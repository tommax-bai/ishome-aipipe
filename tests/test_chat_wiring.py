"""会话侧串联（2026-09-04）：事实上报业务侧、收产物发图、没做出来如实说、送达幂等。

业务侧与渠道侧都是假件：这里验的是会话侧的判据——报什么、什么时候报、报不上怎么说、
图怎么发、假设什么时候说。全部离线。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

import grpc
import pytest
from chat import service
from chat.assumptions import DEFAULT_FLOOR_AREA_RATIO_PERCENT
from chat.channel_client import ChannelClient
from chat.grpc_server import build_server
from chat.models import ConversationRef
from chat.project_client import (
    BusinessProject,
    MilestoneProgress,
    ProjectClientError,
    SlotFill,
)
from chat.repo import find_or_create_project, reset_conversations, reset_messages
from ishome.channel.v1 import message_pb2
from ishome.channel.v1 import service_pb2 as channel_service_pb2
from ishome.channel.v1 import service_pb2_grpc as channel_service_pb2_grpc
from ishome.common.v1 import channel_type_pb2
from ishome.design.v1 import service_pb2 as design_service_pb2
from ishome.design.v1 import service_pb2_grpc as design_service_pb2_grpc

MOCK_INSTANCE = "mock:local"
USER = "ou_wiring"
FLOORPLAN_KEY = "uploads/" + "f" * 64 + "/original.png"


class FakeLlm:
    def __init__(self, intents: list[str], turns: list[str]) -> None:
        self.intents = intents
        self.turns = turns

    async def complete(
        self, model: str, messages: Sequence[Mapping[str, str]], *, json_mode: bool = False
    ) -> str:
        if model == "design-intent.default":
            return self.intents.pop(0)
        if model == "design-orchestrator.default":
            return self.turns.pop(0)
        raise AssertionError(f"unexpected logical model: {model}")


class CapturingSender:
    def __init__(self) -> None:
        self.sent: list[message_pb2.UnifiedMessage] = []
        self.idempotency_keys: list[str] = []

    async def send(self, message: message_pb2.UnifiedMessage, idempotency_key: str) -> str:
        self.sent.append(message)
        self.idempotency_keys.append(idempotency_key)
        return message.message_id


class FakeBusiness:
    """记录上报的假业务侧；可设为当场失败。"""

    def __init__(self, *, failing: bool = False) -> None:
        self.failing = failing
        self.find_calls: list[tuple[int, str, str]] = []
        self.fills: list[tuple[str, list[SlotFill]]] = []

    async def find_or_create_project(
        self, channel_type: int, channel_instance: str, external_user_id: str
    ) -> BusinessProject:
        if self.failing:
            raise ProjectClientError("业务侧连不上（测试）")
        self.find_calls.append((channel_type, channel_instance, external_user_id))
        return BusinessProject("01PROJ", "M0", "v1", created=not self.find_calls[:-1])

    async def fill_slots(self, project_id: str, slots: Sequence[SlotFill]) -> MilestoneProgress:
        if self.failing:
            raise ProjectClientError("业务侧连不上（测试）")
        self.fills.append((project_id, list(slots)))
        keys = {slot.slot_key for _, batch in self.fills for slot in batch}
        advanced = {"floorplan", "building_area_sqm"} <= keys
        return MilestoneProgress(
            project_id, "M0.5" if advanced else "M0", advanced, [], ["01TASK"] if advanced else []
        )


def intent_json(intent: str) -> str:
    return json.dumps({"intent": intent})


def turn_json(facts: list[dict[str, Any]], reply: str) -> str:
    return json.dumps({"facts": facts, "reply": reply}, ensure_ascii=False)


def inbound_text(text: str, message_id: str) -> message_pb2.UnifiedMessage:
    msg = _envelope(message_id)
    msg.text.CopyFrom(message_pb2.TextContent(text=text))
    return msg


def inbound_image(
    message_id: str, object_key: str | None = FLOORPLAN_KEY
) -> message_pb2.UnifiedMessage:
    msg = _envelope(message_id)
    msg.image.CopyFrom(message_pb2.ImageContent(mime_type="image/png", object_key=object_key or ""))
    return msg


def _envelope(message_id: str) -> message_pb2.UnifiedMessage:
    return message_pb2.UnifiedMessage(
        message_id=message_id,
        channel_type=channel_type_pb2.CHANNEL_TYPE_MOCK,
        channel_instance=MOCK_INSTANCE,
        direction=message_pb2.MESSAGE_DIRECTION_INBOUND,
        external_user_id=USER,
    )


def conversation_ref() -> ConversationRef:
    return ConversationRef(
        channel_type=channel_type_pb2.CHANNEL_TYPE_MOCK,
        channel_instance=MOCK_INSTANCE,
        external_user_id=USER,
    )


AREA_FACT = {
    "target_id": "floorplan",
    "property": "building_area_sqm",
    "value": 138,
    "unit": "sqm",
    "cognitive_state": "observed",
}


@pytest.fixture(autouse=True)
def _isolate() -> None:
    reset_messages()
    reset_conversations()


# ---------------------------------------------------------------------------
# 事实上报
# ---------------------------------------------------------------------------


async def test_image_then_area_reports_key_then_area_and_inferred_ratio() -> None:
    business = FakeBusiness()
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info"), intent_json("provide_info")],
        turns=[turn_json([], "收到图了。"), turn_json([AREA_FACT], "好的。")],
    )

    await service.ingest_message(inbound_image("m-1"), sender, llm, business=business)
    assert business.find_calls == [(channel_type_pb2.CHANNEL_TYPE_MOCK, MOCK_INSTANCE, USER)]
    assert len(business.fills) == 1
    first_project_id, first_batch = business.fills[0]
    assert first_project_id == "01PROJ"
    assert [(s.slot_key, s.value, s.cognitive_state, s.source_event_id) for s in first_batch] == [
        ("floorplan", FLOORPLAN_KEY, "observed", "m-1")
    ]

    await service.ingest_message(inbound_text("138平", "m-2"), sender, llm, business=business)
    # 属主只问一次（缓存了业务侧项目 id）；第二批只报新东西：面积 + 按面积推的得房率
    assert len(business.find_calls) == 1
    _, second_batch = business.fills[1]
    assert [(s.slot_key, s.value, s.cognitive_state) for s in second_batch] == [
        ("building_area_sqm", "138", "observed"),
        ("floor_area_ratio_percent", str(DEFAULT_FLOOR_AREA_RATIO_PERCENT), "inferred"),
    ]
    # 两样齐了：只说开始设计（假设那套等图回来再说）
    texts = [m.text.text for m in sender.sent if m.WhichOneof("content") == "text"]
    assert any("开始" in t for t in texts)
    assert not any("得房率按" in t for t in texts)

    project = await find_or_create_project(conversation_ref())
    assert project.business_project_id == "01PROJ"
    assert set(project.reported_slots) == {
        "floorplan",
        "building_area_sqm",
        "floor_area_ratio_percent",
    }


async def test_nothing_new_means_no_round_trip() -> None:
    business = FakeBusiness()
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")] * 2,
        turns=[turn_json([AREA_FACT], "记下了。"), turn_json([], "好的。")],
    )
    await service.ingest_message(inbound_text("138平", "m-1"), sender, llm, business=business)
    await service.ingest_message(inbound_text("嗯", "m-2"), sender, llm, business=business)
    assert len(business.fills) == 1


async def test_owner_given_ratio_is_reported_as_observed() -> None:
    business = FakeBusiness()
    sender = CapturingSender()
    ratio_fact = {
        "target_id": "floorplan",
        "property": "floor_area_ratio",
        "value": 81,
        "unit": "percent",
        "cognitive_state": "observed",
    }
    llm = FakeLlm(
        intents=[intent_json("provide_info")], turns=[turn_json([AREA_FACT, ratio_fact], "好。")]
    )
    await service.ingest_message(
        inbound_text("138平，得房率81", "m-1"), sender, llm, business=business
    )
    _, batch = business.fills[0]
    assert ("floor_area_ratio_percent", "81", "observed") in [
        (s.slot_key, s.value, s.cognitive_state) for s in batch
    ]


async def test_report_failure_is_told_honestly_and_retried_next_turn() -> None:
    business = FakeBusiness(failing=True)
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")] * 2,
        turns=[turn_json([], "收到。"), turn_json([], "好。")],
    )

    await service.ingest_message(inbound_image("m-1"), sender, llm, business=business)
    texts = [m.text.text for m in sender.sent]
    assert service.REPORT_FAILED_MESSAGES[0] in texts
    assert all(len(t) <= 60 for t in service.REPORT_FAILED_MESSAGES)

    business.failing = False
    await service.ingest_message(inbound_text("嗯", "m-2"), sender, llm, business=business)
    # 上一轮没报上的键这一轮补报
    assert [s.slot_key for _, batch in business.fills for s in batch] == ["floorplan"]


async def test_image_without_object_key_is_recorded_but_not_reported() -> None:
    business = FakeBusiness()
    sender = CapturingSender()
    llm = FakeLlm(intents=[intent_json("provide_info")], turns=[turn_json([], "收到。")])
    await service.ingest_message(
        inbound_image("m-1", object_key=None), sender, llm, business=business
    )
    assert business.fills == []
    assert business.find_calls == []


async def test_without_business_gateway_nothing_is_reported() -> None:
    sender = CapturingSender()
    llm = FakeLlm(intents=[intent_json("provide_info")], turns=[turn_json([], "收到。")])
    await service.ingest_message(inbound_image("m-1"), sender, llm)
    project = await find_or_create_project(conversation_ref())
    assert project.reported_slots == {}


# ---------------------------------------------------------------------------
# 产物呈现（PresentDeliverables）
# ---------------------------------------------------------------------------


def _present_request(delivery_id: str = "01DELIV") -> design_service_pb2.PresentDeliverablesRequest:
    return design_service_pb2.PresentDeliverablesRequest(
        delivery_id=delivery_id,
        project_id="01PROJ",
        owner=design_service_pb2.ConversationOwner(
            channel_type=channel_type_pb2.CHANNEL_TYPE_MOCK,
            channel_instance=MOCK_INSTANCE,
            external_user_id=USER,
        ),
        deliverables=[
            design_service_pb2.Deliverable(
                artifact_id="a1",
                artifact_type="vision_mood_image",
                object_key="uploads/f/atmosphere-cream-journal-captioned.png",
            ),
            design_service_pb2.Deliverable(
                artifact_id="a2",
                artifact_type="vision_brief_image",
                object_key="uploads/f/plan-brief.png",
            ),
            design_service_pb2.Deliverable(
                artifact_id="a3",
                artifact_type="vision_style_image",
                object_key="uploads/f/atmosphere-lifestyle-notebook-handwritten.jpg",
                caption="第三张：手账写字版",
            ),
        ],
    )


async def _seed_area() -> None:
    """先让快照里有面积——假设那套要按它说。"""
    sender = CapturingSender()
    llm = FakeLlm(intents=[intent_json("provide_info")], turns=[turn_json([AREA_FACT], "好。")])
    await service.ingest_message(inbound_text("138平", "seed"), sender, llm)


async def test_deliverables_are_sent_as_images_in_order_then_assumptions_follow() -> None:
    await _seed_area()
    sender = CapturingSender()

    delivered, message_ids = await service.present_deliverables(_present_request(), sender)

    assert delivered is True
    kinds = [m.WhichOneof("content") for m in sender.sent]
    # 三张图各带一句系统文案：文 图 文 图 文 图，然后五条假设
    assert kinds[:6] == ["text", "image", "text", "image", "text", "image"]
    assert [m.image.object_key for m in sender.sent if m.WhichOneof("content") == "image"] == [
        "uploads/f/atmosphere-cream-journal-captioned.png",
        "uploads/f/plan-brief.png",
        "uploads/f/atmosphere-lifestyle-notebook-handwritten.jpg",
    ]
    captions = [m.text.text for m in sender.sent[:6] if m.WhichOneof("content") == "text"]
    assert captions[2] == "第三张：手账写字版"  # 业务侧给了说明就用它的
    assert len(message_ids) == 6
    assert sender.idempotency_keys[:6] == [f"deliver-01DELIV-{i}" for i in range(6)]
    tail = [m.text.text for m in sender.sent[6:]]
    assert any("138" in t for t in tail) and any("得房率" in t for t in tail)
    assert sender.idempotency_keys[6].startswith("assumptions-01DELIV-")


async def test_same_delivery_is_not_sent_twice() -> None:
    await _seed_area()
    sender = CapturingSender()
    await service.present_deliverables(_present_request("01SAME"), sender)
    sent_before = len(sender.sent)

    delivered, message_ids = await service.present_deliverables(_present_request("01SAME"), sender)

    assert delivered is False and message_ids == []
    assert len(sender.sent) == sent_before


async def test_failure_is_told_honestly_without_assumptions() -> None:
    await _seed_area()
    sender = CapturingSender()
    request = design_service_pb2.PresentDeliverablesRequest(
        delivery_id="01FAIL",
        project_id="01PROJ",
        owner=_present_request().owner,
        failure=design_service_pb2.GenerationFailure(
            code="plan-2d-render", detail="外圈闭合率 64%", task_type="vision_image"
        ),
    )

    delivered, _ = await service.present_deliverables(request, sender)

    assert delivered is True
    assert [m.text.text for m in sender.sent] == list(service.GENERATION_FAILED_MESSAGES)
    assert all(len(t) <= 60 for t in service.GENERATION_FAILED_MESSAGES)
    assert sender.idempotency_keys == ["failure-01FAIL-0", "failure-01FAIL-1"]


async def test_empty_delivery_is_rejected() -> None:
    sender = CapturingSender()
    with pytest.raises(ValueError):
        await service.present_deliverables(
            design_service_pb2.PresentDeliverablesRequest(
                delivery_id="x", project_id="p", owner=_present_request().owner
            ),
            sender,
        )
    with pytest.raises(ValueError):
        await service.present_deliverables(_present_request(delivery_id=""), sender)


# ---------------------------------------------------------------------------
# gRPC 全链路（进程内）：PresentDeliverables 经 stub 打进来，图经 ChannelService 发出去
# ---------------------------------------------------------------------------


class CapturingChannelServicer(channel_service_pb2_grpc.ChannelServiceServicer):
    def __init__(self) -> None:
        self.requests: list[channel_service_pb2.SendMessageRequest] = []

    async def SendMessage(
        self, request: channel_service_pb2.SendMessageRequest, context: Any
    ) -> channel_service_pb2.SendMessageResponse:
        self.requests.append(request)
        return channel_service_pb2.SendMessageResponse(
            message_id=request.message.message_id, channel_message_id="mock-1"
        )


@pytest.fixture
async def grpc_harness() -> AsyncIterator[tuple[CapturingChannelServicer, FakeBusiness, Any]]:
    captured = CapturingChannelServicer()
    channel_server = grpc.aio.server()
    channel_service_pb2_grpc.add_ChannelServiceServicer_to_server(captured, channel_server)
    channel_port = channel_server.add_insecure_port("127.0.0.1:0")
    await channel_server.start()
    channel_client = ChannelClient(f"127.0.0.1:{channel_port}")
    business = FakeBusiness()
    llm = FakeLlm(intents=[intent_json("provide_info")], turns=[turn_json([AREA_FACT], "好。")])
    chat_server = build_server(channel_client, llm, business=business)
    chat_port = chat_server.add_insecure_port("127.0.0.1:0")
    await chat_server.start()
    async with grpc.aio.insecure_channel(f"127.0.0.1:{chat_port}") as caller:
        yield captured, business, design_service_pb2_grpc.DesignServiceStub(caller)
    await channel_client.aclose()
    await chat_server.stop(grace=None)
    await channel_server.stop(grace=None)


async def test_grpc_ingest_reports_and_present_sends_images(
    grpc_harness: tuple[CapturingChannelServicer, FakeBusiness, Any],
) -> None:
    captured, business, stub = grpc_harness

    await stub.IngestMessage(
        design_service_pb2.IngestMessageRequest(message=inbound_text("138平", "g-1"))
    )
    assert [s.slot_key for _, batch in business.fills for s in batch][0] == "building_area_sqm"

    response = cast(
        design_service_pb2.PresentDeliverablesResponse,
        await stub.PresentDeliverables(_present_request("01GRPC")),
    )
    assert response.delivered is True
    assert len(response.message_ids) == 6
    images = [r.message for r in captured.requests if r.message.WhichOneof("content") == "image"]
    assert [m.image.object_key for m in images][1] == "uploads/f/plan-brief.png"
    assert images[0].external_user_id == USER
    assert images[0].direction == message_pb2.MESSAGE_DIRECTION_OUTBOUND

    again = cast(
        design_service_pb2.PresentDeliverablesResponse,
        await stub.PresentDeliverables(_present_request("01GRPC")),
    )
    assert again.delivered is False
