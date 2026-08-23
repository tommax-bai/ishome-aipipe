"""Design Orchestrator v1 测试（全部离线，FakeLLM 注入——单测不打真网络）。

覆盖：意图分类解析、事实抽取合并与认知状态、结构类拒信（§8.3 红线）、
确认清单生成与 user_confirmed 升级、修正回路、quick_reply 降级、幂等、
LLM 故障兜底、gRPC 全链路（进程内）。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

import grpc
import pytest
from design import orchestrator, service
from design.channel_client import ChannelClient
from design.grpc_server import build_server
from design.intent import parse_intent, route_intent
from design.repo import find_or_create_project, reset_conversations, reset_seen_messages
from ishome.channel.v1 import message_pb2
from ishome.channel.v1 import service_pb2 as channel_service_pb2
from ishome.channel.v1 import service_pb2_grpc as channel_service_pb2_grpc
from ishome.common.v1 import channel_type_pb2
from ishome.design.v1 import service_pb2 as design_service_pb2
from ishome.design.v1 import service_pb2_grpc as design_service_pb2_grpc

MOCK_INSTANCE = "mock:local"
USER = "u-test"

ALL_SLOT_FACTS: list[dict[str, Any]] = [
    {
        "target_id": "floorplan",
        "property": "estate_name",
        "value": "翠湖天地",
        "cognitive_state": "observed",
    },
    {
        "target_id": "floorplan",
        "property": "source",
        "value": "library",
        "cognitive_state": "observed",
    },
    {
        "target_id": "scale-anchor",
        "property": "entry_door_width",
        "value": 900,
        "unit": "mm",
        "cognitive_state": "inferred",
    },
    {
        "target_id": "household",
        "property": "composition",
        "value": "三口之家",
        "cognitive_state": "observed",
    },
    {
        "target_id": "need-1",
        "property": "core_need",
        "value": "收纳要多",
        "cognitive_state": "observed",
    },
    {
        "target_id": "no-go-1",
        "property": "constraint",
        "value": "无",
        "cognitive_state": "observed",
    },
]


class FakeLlm:
    """按逻辑模型名分发的脚本化 LLM（列表耗尽即测试脚本错误）。"""

    def __init__(self, intents: list[str] | None = None, turns: list[str] | None = None) -> None:
        self.intents = intents or []
        self.turns = turns or []
        self.calls: list[str] = []

    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        *,
        json_mode: bool = False,
    ) -> str:
        self.calls.append(model)
        if model == "design-intent.default":
            return self.intents.pop(0)
        if model == "design-orchestrator.default":
            return self.turns.pop(0)
        raise AssertionError(f"unexpected logical model: {model}")


class BrokenLlm:
    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        *,
        json_mode: bool = False,
    ) -> str:
        raise RuntimeError("gateway down")


class CapturingSender:
    def __init__(self) -> None:
        self.sent: list[message_pb2.UnifiedMessage] = []
        self.idempotency_keys: list[str] = []

    async def send(self, message: message_pb2.UnifiedMessage, idempotency_key: str) -> str:
        self.sent.append(message)
        self.idempotency_keys.append(idempotency_key)
        return message.message_id


class FixedCapability:
    def __init__(self, supports: bool) -> None:
        self.supports = supports

    async def supports_quick_reply(self, channel_type: int, channel_instance: str) -> bool:
        return self.supports


def intent_json(intent: str) -> str:
    return json.dumps({"intent": intent})


def turn_json(facts: list[dict[str, Any]], reply: str) -> str:
    return json.dumps({"facts": facts, "reply": reply}, ensure_ascii=False)


def make_inbound(
    text: str | None = None,
    message_id: str = "in-1",
    quick_reply_option: str | None = None,
) -> message_pb2.UnifiedMessage:
    msg = message_pb2.UnifiedMessage(
        message_id=message_id,
        channel_type=channel_type_pb2.CHANNEL_TYPE_MOCK,
        channel_instance=MOCK_INSTANCE,
        direction=message_pb2.MESSAGE_DIRECTION_INBOUND,
        external_user_id=USER,
    )
    if quick_reply_option is not None:
        msg.quick_reply.CopyFrom(
            message_pb2.QuickReplyContent(selected_option_id=quick_reply_option)
        )
    else:
        msg.text.CopyFrom(message_pb2.TextContent(text=text or ""))
    return msg


def conversation_key() -> str:
    return f"{channel_type_pb2.CHANNEL_TYPE_MOCK}:{MOCK_INSTANCE}:{USER}"


@pytest.fixture(autouse=True)
def _isolate() -> None:
    reset_seen_messages()
    reset_conversations()


# --- 意图路由 ---


def test_parse_intent_variants() -> None:
    assert parse_intent(intent_json("ask_reason")) == "ask_reason"
    assert parse_intent('```json\n{"intent": "other"}\n```') == "other"
    assert parse_intent("not json at all") == "provide_info"
    assert parse_intent(intent_json("nonsense-intent")) == "provide_info"


@pytest.mark.asyncio
async def test_confirm_shortcut_skips_llm() -> None:
    llm = FakeLlm()  # 空脚本：若走 LLM 会 pop 空列表报错
    assert await route_intent(llm, "确认", checklist_open=True) == "confirm_checklist"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_confirm_intent_without_open_checklist_degrades() -> None:
    llm = FakeLlm(intents=[intent_json("confirm_checklist")])
    assert await route_intent(llm, "都对", checklist_open=False) == "provide_info"


# --- 事实抽取与认知状态 ---


@pytest.mark.asyncio
async def test_fact_extraction_and_cognitive_states() -> None:
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")],
        turns=[
            turn_json(
                [ALL_SLOT_FACTS[0], ALL_SLOT_FACTS[2]],
                "记下了。想先聊聊家里都有谁住吗？",
            )
        ],
    )
    await service.ingest_message(make_inbound("我家在翠湖天地"), sender, llm)
    project = await find_or_create_project(conversation_key())
    states = {(f.target_id, f.property): f.cognitive_state for f in project.base_facts.facts}
    assert states[("floorplan", "estate_name")] == "observed"
    assert states[("scale-anchor", "entry_door_width")] == "inferred"
    assert project.base_facts.scale_anchor is not None
    assert len(sender.sent) == 1
    assert "记下了" in sender.sent[0].text.text


@pytest.mark.asyncio
async def test_extraction_never_emits_user_confirmed() -> None:
    turn = orchestrator.parse_turn(
        turn_json([{**ALL_SLOT_FACTS[0], "cognitive_state": "user_confirmed"}], "好的")
    )
    assert turn.facts[0].cognitive_state == "inferred"  # 钳制：确认只能由确认闭环授予


# --- 结构类红线（§8.3） ---


@pytest.mark.asyncio
async def test_structural_facts_never_confirmable() -> None:
    sender = CapturingSender()
    structural_fact = {
        "target_id": "wall-living-north",
        "property": "load_bearing",
        "value": True,
        "fact_kind": "structural",
        "cognitive_state": "observed",
    }
    llm = FakeLlm(
        intents=[intent_json("provide_info"), intent_json("provide_info")],
        turns=[
            turn_json([structural_fact], "了解，你提到想动那面墙。"),
            turn_json(ALL_SLOT_FACTS, "信息齐了。"),
        ],
    )
    await service.ingest_message(make_inbound("客厅北墙是承重墙，我想砸掉", "in-s1"), sender, llm)
    # 结构说明的固定文案随回复附带
    assert "不能作为设计依据" in sender.sent[0].text.text
    # 集齐五槽位后出清单：结构类事实不在清单里
    await service.ingest_message(make_inbound("其他信息都给你", "in-s2"), sender, llm)
    checklist = sender.sent[-1].text.text
    assert checklist.startswith(orchestrator.CHECKLIST_MARKER)
    assert "load_bearing" not in checklist and "承重" not in checklist
    # 确认后：结构类事实认知状态原样，永不 user_confirmed
    await service.ingest_message(
        make_inbound(None, "in-s3", quick_reply_option=orchestrator.CONFIRM_OPTION_ID),
        sender,
        llm,
    )
    project = await find_or_create_project(conversation_key())
    structural = [f for f in project.base_facts.facts if f.fact_kind == "structural"]
    assert structural and all(f.cognitive_state != "user_confirmed" for f in structural)
    dimensional = [f for f in project.base_facts.facts if f.fact_kind == "dimensional"]
    assert dimensional and all(f.cognitive_state == "user_confirmed" for f in dimensional)


# --- 确认清单 ---


@pytest.mark.asyncio
async def test_checklist_quick_reply_when_supported() -> None:
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")],
        turns=[turn_json(ALL_SLOT_FACTS, "都记下了。")],
    )
    await service.ingest_message(make_inbound("一口气全告诉你"), sender, llm, FixedCapability(True))
    checklist = sender.sent[-1]
    assert checklist.WhichOneof("content") == "quick_reply"
    assert checklist.quick_reply.prompt_text.startswith(orchestrator.CHECKLIST_MARKER)
    assert "约900mm" in checklist.quick_reply.prompt_text  # inferred 值带"约"（§8.1）
    option_ids = [o.option_id for o in checklist.quick_reply.options]
    assert option_ids == [orchestrator.CONFIRM_OPTION_ID, orchestrator.CORRECT_OPTION_ID]
    project = await find_or_create_project(conversation_key())
    assert project.open_confirmation_ids


@pytest.mark.asyncio
async def test_checklist_degrades_to_text_without_capability() -> None:
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")],
        turns=[turn_json(ALL_SLOT_FACTS, "都记下了。")],
    )
    await service.ingest_message(
        make_inbound("一口气全告诉你"), sender, llm, FixedCapability(False)
    )
    checklist = sender.sent[-1]
    assert checklist.WhichOneof("content") == "text"
    assert checklist.text.text.startswith(orchestrator.CHECKLIST_MARKER)


@pytest.mark.asyncio
async def test_confirm_upgrades_to_user_confirmed() -> None:
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")],
        turns=[turn_json(ALL_SLOT_FACTS, "都记下了。")],
    )
    await service.ingest_message(make_inbound("信息都给你", "in-c1"), sender, llm)
    await service.ingest_message(make_inbound("确认", "in-c2"), sender, llm)
    ack = sender.sent[-1].text.text
    assert ack.startswith(orchestrator.CONFIRM_ACK_MARKER)
    project = await find_or_create_project(conversation_key())
    assert project.minimum_inputs_confirmed
    assert not project.open_confirmation_ids
    assert all(
        f.cognitive_state == "user_confirmed" for f in orchestrator.confirmable_facts(project)
    )


@pytest.mark.asyncio
async def test_correction_reopens_checklist() -> None:
    sender = CapturingSender()
    corrected = {**ALL_SLOT_FACTS[3], "value": "四口之家"}
    llm = FakeLlm(
        intents=[
            intent_json("provide_info"),
            intent_json("correct_checklist"),
        ],
        turns=[
            turn_json(ALL_SLOT_FACTS, "都记下了。"),
            turn_json([corrected], "已改为四口之家。"),
        ],
    )
    await service.ingest_message(make_inbound("信息都给你", "in-r1"), sender, llm)
    await service.ingest_message(make_inbound("家庭结构错了，是四口之家", "in-r2"), sender, llm)
    checklist = sender.sent[-1].text.text
    assert checklist.startswith(orchestrator.CHECKLIST_MARKER)
    assert "四口之家" in checklist
    project = await find_or_create_project(conversation_key())
    assert not project.minimum_inputs_confirmed
    assert project.open_confirmation_ids


# --- 幂等与兜底 ---


@pytest.mark.asyncio
async def test_duplicate_inbound_replies_once() -> None:
    sender = CapturingSender()
    llm = FakeLlm(intents=[intent_json("other")], turns=[turn_json([], "你好呀。")])
    inbound = make_inbound("你好", "in-dup")
    await service.ingest_message(inbound, sender, llm)
    await service.ingest_message(inbound, sender, llm)
    assert len(sender.sent) == 1


@pytest.mark.asyncio
async def test_llm_failure_sends_fallback() -> None:
    sender = CapturingSender()
    await service.ingest_message(make_inbound("你好", "in-broken"), sender, BrokenLlm())
    assert len(sender.sent) == 1
    assert sender.sent[0].text.text == service.FALLBACK_REPLY


# --- gRPC 全链路（进程内） ---


class CapturingChannelServicer(channel_service_pb2_grpc.ChannelServiceServicer):
    def __init__(self) -> None:
        self.requests: list[channel_service_pb2.SendMessageRequest] = []

    async def SendMessage(
        self, request: channel_service_pb2.SendMessageRequest, context: Any
    ) -> channel_service_pb2.SendMessageResponse:
        self.requests.append(request)
        return channel_service_pb2.SendMessageResponse(
            message_id=request.message.message_id, channel_message_id="mock-channel-msg-1"
        )


@pytest.fixture
async def grpc_harness() -> AsyncIterator[tuple[CapturingChannelServicer, Any]]:
    captured = CapturingChannelServicer()
    channel_server = grpc.aio.server()
    channel_service_pb2_grpc.add_ChannelServiceServicer_to_server(captured, channel_server)
    channel_port = channel_server.add_insecure_port("127.0.0.1:0")
    await channel_server.start()
    channel_client = ChannelClient(f"127.0.0.1:{channel_port}")
    llm = FakeLlm(intents=[intent_json("other")], turns=[turn_json([], "你好，我是你的设计顾问。")])
    design_server = build_server(channel_client, llm)
    design_port = design_server.add_insecure_port("127.0.0.1:0")
    await design_server.start()
    async with grpc.aio.insecure_channel(f"127.0.0.1:{design_port}") as caller:
        yield captured, design_service_pb2_grpc.DesignServiceStub(caller)
    await channel_client.aclose()
    await design_server.stop(grace=None)
    await channel_server.stop(grace=None)


@pytest.mark.asyncio
async def test_grpc_roundtrip_with_fake_llm(
    grpc_harness: tuple[CapturingChannelServicer, Any],
) -> None:
    captured, stub = grpc_harness
    request = design_service_pb2.IngestMessageRequest(message=make_inbound("你好", "in-grpc"))
    response = cast(design_service_pb2.IngestMessageResponse, await stub.IngestMessage(request))
    assert response.message_id == "in-grpc"
    assert len(captured.requests) == 1
    reply = captured.requests[0].message
    assert reply.direction == message_pb2.MESSAGE_DIRECTION_OUTBOUND
    assert reply.external_user_id == USER
    assert "设计顾问" in reply.text.text
