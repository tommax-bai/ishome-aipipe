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
from chat import orchestrator, service
from chat.assumptions import DEFAULT_FLOOR_AREA_RATIO_PERCENT
from chat.channel_client import ChannelClient
from chat.grpc_server import build_server
from chat.models import ConversationRef
from chat.project_client import (
    BusinessProject,
    BusinessSlot,
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
    """记录上报的假业务侧；可设为当场失败，也可预置"表里本来就有的槽位"。

    `existing_slots` ＝ 业主此前给过、已经落在业务侧真相里的那些——会话侧重启后按属主问回来的就是它。
    """

    def __init__(
        self, *, failing: bool = False, existing_slots: Sequence[BusinessSlot] = ()
    ) -> None:
        self.failing = failing
        self.existing_slots = tuple(existing_slots)
        self.find_calls: list[tuple[int, str, str]] = []
        self.fills: list[tuple[str, list[SlotFill]]] = []

    async def find_or_create_project(
        self, channel_type: int, channel_instance: str, external_user_id: str
    ) -> BusinessProject:
        if self.failing:
            raise ProjectClientError("业务侧连不上（测试）")
        self.find_calls.append((channel_type, channel_instance, external_user_id))
        return BusinessProject(
            "01PROJ",
            "M0",
            "v1",
            created=not self.find_calls[:-1],
            slots=self.existing_slots,
        )

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
    # 两样齐了那一轮：只说那一句（假设那套等图回来再说，裁决 8-31；一句话说完，裁决 9-07）
    texts = [m.text.text for m in sender.sent if m.WhichOneof("content") == "text"]
    assert texts[-1] == service.DESIGN_START_MESSAGES[0]
    assert not any("得房率按" in t for t in texts)

    project = await find_or_create_project(conversation_ref())
    assert project.business_project_id == "01PROJ"
    assert set(project.reported_slots) == {
        "floorplan",
        "building_area_sqm",
        "floor_area_ratio_percent",
    }


async def test_chitchat_turn_means_no_round_trip() -> None:
    """业主没再给新东西的那一轮（"嗯"）不往返——省的是那一跳无谓的往返与它连带的里程碑判定。"""
    business = FakeBusiness()
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")] * 2,
        turns=[turn_json([AREA_FACT], "记下了。"), turn_json([], "好的。")],
    )
    await service.ingest_message(inbound_text("138平", "m-1"), sender, llm, business=business)
    await service.ingest_message(inbound_text("嗯", "m-2"), sender, llm, business=business)
    assert len(business.fills) == 1


async def test_resending_the_same_floorplan_reports_again() -> None:
    """重发同一张户型图必须再报一次（用户裁决 2026-09-07："重发一张图的话，就再来一次"）。

    这是 2026-09-06 真机第三跑的门禁：生成失败后系统请业主重发，他重发了同一张图，
    而户型图槽位的值是内容寻址的对象键、同一张图值恒等——按"值变没变"判就把这一轮整个滤空，
    会话侧一个 HTTP 都没打。判据换成"他这一轮又给了没有"之后，这一跳必须发生。
    """
    business = FakeBusiness()
    sender = CapturingSender()
    llm = FakeLlm(
        intents=[intent_json("provide_info")] * 2,
        turns=[turn_json([], "收到图了。"), turn_json([], "再试一次。")],
    )

    await service.ingest_message(inbound_image("m-1"), sender, llm, business=business)
    await service.ingest_message(inbound_image("m-2"), sender, llm, business=business)

    assert len(business.fills) == 2
    _, second_batch = business.fills[1]
    # 值一模一样也照报；source_event_id 是这一轮的入站 message_id（契约不动）
    assert [(s.slot_key, s.value, s.source_event_id) for s in second_batch] == [
        ("floorplan", FLOORPLAN_KEY, "m-2")
    ]


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
    # 没有对象键就没有可报的事实：一条槽位都不报。
    # 按属主那一跳照打——它是每轮开头"业主已经给过什么"的读面，与报不报无关。
    assert business.fills == []
    assert len(business.find_calls) == 1


async def test_restart_does_not_ask_again_what_the_owner_already_answered() -> None:
    """重启后（＝会话快照为空）业主发图，不再问他已经回答过的面积。

    2026-09-06 19:59 业主说过"138 平米，81% 得房率"，落进了业务侧真相；9-07 部署重启后他重发
    户型图，系统又问了一遍"这套房的建筑面积是多少平方米"——他当场说"不应该再出现咨询我建筑面积
    的情况"。会话快照是进程内的、重启即空，槽位真相一直在业务侧表里：算缺口之前先向它要一次。

    **编排模型这一轮一次都不该调**（`turns=[]`：真调了就 IndexError，兜底话会顶掉那一句）——
    两样齐了那一轮由系统文案接管，模型没有产回复的位置也就问不出话来（裁决 9-07）。
    """
    business = FakeBusiness(
        existing_slots=[
            BusinessSlot("building_area_sqm", "138", "observed", "m-0906"),
            BusinessSlot("floor_area_ratio_percent", "81", "observed", "m-0906"),
        ]
    )
    sender = CapturingSender()
    llm = FakeLlm(intents=[intent_json("provide_info")], turns=[])

    await service.ingest_message(inbound_image("m-1"), sender, llm, business=business)

    texts = [m.text.text for m in sender.sent if m.WhichOneof("content") == "text"]
    assert texts == [service.DESIGN_START_MESSAGES[0]]
    assert not any("面积" in t for t in texts)
    # 读回来的算已经报过：这一轮只报他又给了一次的那张图
    assert len(business.find_calls) == 1
    assert [(s.slot_key, s.value) for _, batch in business.fills for s in batch] == [
        ("floorplan", FLOORPLAN_KEY)
    ]


async def test_business_side_inferred_ratio_is_not_restored_as_the_owners_word() -> None:
    """按面积推的得房率是我们自己填进业务侧的，读回来不能当成业主说过的话。

    还原成 observed 事实等于把自己的猜测洗成他的话（《纪律·拿不到就说没有，不许填猜的值》）。
    不还原也不会招来重复上报——`reported_slots` 里记着，值一样就不再往返。
    """
    business = FakeBusiness(
        existing_slots=[
            BusinessSlot("building_area_sqm", "138", "observed"),
            BusinessSlot(
                "floor_area_ratio_percent", str(DEFAULT_FLOOR_AREA_RATIO_PERCENT), "inferred"
            ),
        ]
    )
    sender = CapturingSender()
    llm = FakeLlm(intents=[intent_json("provide_info")], turns=[])

    await service.ingest_message(inbound_image("m-1"), sender, llm, business=business)

    project = await find_or_create_project(conversation_ref())
    assert orchestrator.find_building_area_sqm(project) == 138
    assert orchestrator.find_floor_area_ratio_percent(project) is None
    assert project.reported_slots["floor_area_ratio_percent"] == str(
        DEFAULT_FLOOR_AREA_RATIO_PERCENT
    )
    assert [s.slot_key for _, batch in business.fills for s in batch] == ["floorplan"]


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
