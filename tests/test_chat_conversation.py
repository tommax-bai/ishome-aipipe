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
from chat import orchestrator, service
from chat.channel_client import ChannelClient
from chat.grpc_server import build_server
from chat.intent import parse_intent, route_intent
from chat.models import ConversationRef, Fact, ProjectState
from chat.repo import active_store, find_or_create_project, reset_conversations, reset_messages
from chat.repo.memory import MemoryChatStore
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
        # 面积是业主开头唯一还需要给的数（裁决 8-31）
        "target_id": "floorplan",
        "property": "building_area_sqm",
        "value": 138,
        "unit": "sqm",
        "cognitive_state": "observed",
    },
    {
        # 得房率不再问了（按面积推 80%），业主主动给的照收
        "target_id": "floorplan",
        "property": "floor_area_ratio",
        "value": 81,
        "unit": "percent",
        "cognitive_state": "observed",
    },
    {
        # 比例锚点不是必答题（裁决 2026-08-31），但业主主动给的实测值照收
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
        self.turn_prompts: list[str] = []

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
            self.turn_prompts.append(messages[0]["content"])
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


def conversation_ref() -> ConversationRef:
    return ConversationRef(
        channel_type=channel_type_pb2.CHANNEL_TYPE_MOCK,
        channel_instance=MOCK_INSTANCE,
        external_user_id=USER,
    )


@pytest.fixture(autouse=True)
def _isolate() -> None:
    reset_messages()
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
                [ALL_SLOT_FACTS[0], ALL_SLOT_FACTS[4]],
                "记下了。想先聊聊家里都有谁住吗？",
            )
        ],
    )
    await service.ingest_message(make_inbound("我家在翠湖天地"), sender, llm)
    project = await find_or_create_project(conversation_ref())
    states = {(f.target_id, f.property): f.cognitive_state for f in project.base_facts.facts}
    assert states[("floorplan", "estate_name")] == "observed"
    assert states[("scale-anchor", "entry_door_width")] == "inferred"
    assert project.base_facts.scale_anchor is not None
    assert len(sender.sent) == 1
    assert "记下了" in sender.sent[0].text.text


@pytest.mark.asyncio
async def test_replies_go_out_as_separate_messages() -> None:
    # 一轮说几件事就发几条：模型自己标思维停顿，发送侧照数组发（用户裁决 2026-08-31）
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")],
        turns=[
            json.dumps(
                {
                    "facts": [ALL_SLOT_FACTS[0]],
                    "replies": ["翠湖天地，记下了。", "家里都有谁住？"],
                },
                ensure_ascii=False,
            )
        ],
    )
    await service.ingest_message(make_inbound("我家在翠湖天地"), sender, llm)
    assert [m.text.text for m in sender.sent] == ["翠湖天地，记下了。", "家里都有谁住？"]
    # 幂等键按序号排：同一入站消息重试不会把两条都再发一遍
    assert sender.idempotency_keys == ["reply-in-1-0", "reply-in-1-1"]


def test_reply_shapes_are_all_accepted() -> None:
    """键名与形态都收宽：`replies`/`reply` 两个键、数组/单串两种值，四种组合都收。

    真机踩过：模型照做了分条（回了数组），但键名仍写 `reply`——首版只认"replies 是数组、
    reply 是字符串"这两种，那一种两边都不沾，**整轮回复被丢光**，业主收到的是兜底话。
    丢掉它的是我们的解析器，不是模型的输出。
    """
    both = ["收到户型图了", "您家在哪个小区？"]
    assert orchestrator.parse_turn(json.dumps({"facts": [], "reply": both})).replies == both
    assert orchestrator.parse_turn(json.dumps({"facts": [], "replies": both})).replies == both
    assert orchestrator.parse_turn(turn_json([], "就说一句")).replies == ["就说一句"]
    assert orchestrator.parse_turn(json.dumps({"facts": [], "replies": "就说一句"})).replies == [
        "就说一句"
    ]


@pytest.mark.asyncio
async def test_uploading_an_image_closes_the_floorplan_gap_in_the_same_turn() -> None:
    """图刚传上来就不该再问"你有户型图吗"。

    真机问出过"您家在哪个小区？几室几厅？"——成因是缺口按上一轮末的状态算，而"他传了图"
    这件事等着模型去抽，有一轮延迟。会话侧自己就知道这条入站是图片，**代码知道的事不问模型**。
    """
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")],
        turns=[turn_json([], "收到户型图啦，我这就开始识别～")],
    )
    inbound = make_inbound(None, "in-img")
    # 只要内容类型是图片就够——会话侧判的是"这条是不是图"，不是图里有什么
    # （对象键要等本仓的 contracts 依赖从 v0.2.0 抬到带 objectKey 那一版，时点＝做上报业务侧那一段）
    inbound.image.SetInParent()

    await service.ingest_message(inbound, sender, llm)

    project = await find_or_create_project(conversation_ref())
    assert "floorplan" not in orchestrator.missing_slots(project)
    # 递给模型的那段上下文里，也就不会再出现"还没有户型图"
    assert "还没有户型图" not in llm.turn_prompts[0]


def test_floorplan_gap_never_asks_for_what_the_image_answers() -> None:
    """不问小区名、不问几室几厅、不问面积——图到手这些都算得出来。"""
    hint = orchestrator._SLOT_HINTS["floorplan"]

    assert "不要问小区名" in hint and "不要问几室几厅" in hint


def test_context_given_to_the_model_carries_no_internal_identifiers() -> None:
    """递给模型的已知信息用客户语域写，**不带字段名、取值码与单位缩写**。

    真机漏过一次：上下文里写着 `floorplan/source = upload_pending（observed）`，
    模型原样转述给业主"您提到的户型图我们已收到（来源：upload_pending）"。
    **它看不见就说不出**——与其加一条"不许说内部字段名"的禁令，不如根本不给它看。
    """
    lines = [
        orchestrator._render_fact_line(f)
        for f in (
            orchestrator.upload_fact(),
            Fact(
                target_id="floorplan",
                property="building_area_sqm",
                value=138,
                unit="sqm",
                cognitive_state="observed",
                source="user_message",
            ),
            Fact(
                target_id="scale-anchor",
                property="entry_door_width",
                value=900,
                unit="mm",
                cognitive_state="inferred",
                source="orchestrator_inference",
            ),
        )
    ]
    rendered = "\n".join(lines)

    for leak in ("upload_pending", "uploaded", "building_area_sqm", "sqm", "observed", "inferred"):
        assert leak not in rendered, leak
    assert "建筑面积：138 平方米" in rendered
    assert "入户门宽实测" in rendered


def test_only_area_and_floorplan_are_ever_asked_for() -> None:
    """业主开头只需要给两样：面积 + 户型图（用户裁决 2026-08-31）。

    这一条收窄了三批问法，每一批都是同一种病——**系统在问它自己能算或能推的东西**：
    比例锚点（逼他拿卷尺量房）、小区名与几室几厅（图里就写着）、得房率与家庭结构（面积推得出）。
    """
    project = ProjectState(project_id="p-slots", user_id="u-slots")

    assert orchestrator.missing_slots(project) == ["building_area_sqm", "floorplan"]

    orchestrator.merge_facts(
        project,
        orchestrator.parse_turn(
            turn_json(
                [
                    {
                        "target_id": "floorplan",
                        "property": "building_area_sqm",
                        "value": 138,
                        "unit": "sqm",
                        "cognitive_state": "observed",
                    }
                ],
                "记下了。",
            )
        ).facts,
    )
    orchestrator.merge_facts(project, [orchestrator.upload_fact()])

    assert orchestrator.missing_slots(project) == []


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
    # 结构说明**自成一条**：它是另一件事，拼在回话尾巴上正好把那条撑成长文（用户 2026-08-31）
    assert "不能作为设计依据" not in sender.sent[0].text.text
    assert "不能作为设计依据" in sender.sent[1].text.text
    # 结构类事实永不进可确认集合——口述的结构信息不作设计依据（§8.3）
    await service.ingest_message(make_inbound("其他信息都给你", "in-s2"), sender, llm)
    project = await find_or_create_project(conversation_ref())
    structural = [f for f in project.base_facts.facts if f.fact_kind == "structural"]
    assert structural
    assert all(f.fact_kind == "dimensional" for f in orchestrator.confirmable_facts(project))


# --- 两样齐了：摊开说假设 ---


@pytest.mark.asyncio
async def test_assumptions_are_told_once_the_two_inputs_are_in() -> None:
    """面积与户型图都齐了就把按面积推的那套摊开说，**不出确认清单、不要他按确认**。

    形态是"先做出来再让他改"（裁决 8-31）：给了就用、不给就算。
    """
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")],
        turns=[turn_json(ALL_SLOT_FACTS, "都记下了。")],
    )

    await service.ingest_message(make_inbound("138 平，图发你了"), sender, llm)

    told = sender.sent[-1].text.text
    assert "138 平" in told
    assert "我按 4 个人来安排" in told  # 138/34 → 4，与用户给的锚点一致
    assert "80%" in told
    assert "不说也没关系" in told  # 邀请不是追问
    assert not any(m.WhichOneof("content") == "quick_reply" for m in sender.sent)


@pytest.mark.asyncio
async def test_assumptions_are_never_repeated() -> None:
    """只说一次：每轮再说一遍就从"告知"变成"催问"了。"""
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info"), intent_json("provide_info")],
        turns=[turn_json(ALL_SLOT_FACTS, "都记下了。"), turn_json([], "好的。")],
    )
    await service.ingest_message(make_inbound("138 平，图发你了", "in-a1"), sender, llm)
    before = len(sender.sent)

    await service.ingest_message(make_inbound("再说一句", "in-a2"), sender, llm)

    assert all("我按" not in m.text.text for m in sender.sent[before:])


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


# --- 消息原文落存（IngestMessage 链路，入站与出站都存） ---


@pytest.mark.asyncio
async def test_messages_recorded_inbound_and_outbound() -> None:
    sender = CapturingSender()
    llm = FakeLlm(intents=[intent_json("other")], turns=[turn_json([], "你好呀。")])
    await service.ingest_message(make_inbound("你好", "in-store"), sender, llm)
    store = active_store()
    assert isinstance(store, MemoryChatStore)
    records = store.list_messages(conversation_ref())
    assert [(m.direction, m.content_type, m.text) for m in records] == [
        ("inbound", "text", "你好"),
        ("outbound", "text", "你好呀。"),
    ]
    assert records[0].idempotency_key == "in-store"  # 入站幂等键 = 渠道消息 id
    assert records[1].idempotency_key == "reply-in-store-0"  # 出站幂等键与发送键同源


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
    chat_server = build_server(channel_client, llm)
    chat_port = chat_server.add_insecure_port("127.0.0.1:0")
    await chat_server.start()
    async with grpc.aio.insecure_channel(f"127.0.0.1:{chat_port}") as caller:
        yield captured, design_service_pb2_grpc.DesignServiceStub(caller)
    await channel_client.aclose()
    await chat_server.stop(grace=None)
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
