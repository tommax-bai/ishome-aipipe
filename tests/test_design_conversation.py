"""纵切冒烟（Python 侧，全进程内）：mock 渠道入站 → DesignService.IngestMessage → 回话回发。

链路：测试客户端 → design gRPC 服务端（真实 build_server + ChannelClient）
→ 假 ChannelService 服务端（捕获 SendMessage）。断言回话前缀（联调契约钉死）、
渠道字段回传一致、入站重复投递幂等。
"""

from collections.abc import AsyncIterator
from typing import Any, cast

import grpc
import pytest
from design.channel_client import ChannelClient
from design.grpc_server import build_server
from design.repo import reset_seen_messages
from design.service import REPLY_PREFIX
from ishome.channel.v1 import message_pb2
from ishome.channel.v1 import service_pb2 as channel_service_pb2
from ishome.channel.v1 import service_pb2_grpc as channel_service_pb2_grpc
from ishome.common.v1 import channel_type_pb2
from ishome.design.v1 import service_pb2 as design_service_pb2
from ishome.design.v1 import service_pb2_grpc as design_service_pb2_grpc

MOCK_INSTANCE = "mock:local"


class CapturingChannelServicer(channel_service_pb2_grpc.ChannelServiceServicer):
    """假 ChannelService：捕获出站回话供断言。"""

    def __init__(self) -> None:
        self.requests: list[channel_service_pb2.SendMessageRequest] = []

    async def SendMessage(
        self, request: channel_service_pb2.SendMessageRequest, context: Any
    ) -> channel_service_pb2.SendMessageResponse:
        self.requests.append(request)
        return channel_service_pb2.SendMessageResponse(
            message_id=request.message.message_id, channel_message_id="mock-channel-msg-1"
        )


class ConversationHarness:
    def __init__(
        self, captured: CapturingChannelServicer, stub: design_service_pb2_grpc.DesignServiceStub
    ) -> None:
        self.captured = captured
        self.stub = stub


@pytest.fixture
async def harness() -> AsyncIterator[ConversationHarness]:
    reset_seen_messages()
    # 假 channel 服务端
    captured = CapturingChannelServicer()
    channel_server = grpc.aio.server()
    channel_service_pb2_grpc.add_ChannelServiceServicer_to_server(captured, channel_server)
    channel_port = channel_server.add_insecure_port("127.0.0.1:0")
    await channel_server.start()
    # 真实 design 服务端，出站指向假 channel
    channel_client = ChannelClient(f"127.0.0.1:{channel_port}")
    design_server = build_server(channel_client)
    design_port = design_server.add_insecure_port("127.0.0.1:0")
    await design_server.start()
    async with grpc.aio.insecure_channel(f"127.0.0.1:{design_port}") as caller:
        yield ConversationHarness(captured, design_service_pb2_grpc.DesignServiceStub(caller))
    await channel_client.aclose()
    await design_server.stop(grace=None)
    await channel_server.stop(grace=None)
    reset_seen_messages()


def _inbound_text(message_id: str, text: str) -> message_pb2.UnifiedMessage:
    return message_pb2.UnifiedMessage(
        message_id=message_id,
        channel_type=channel_type_pb2.CHANNEL_TYPE_MOCK,
        channel_instance=MOCK_INSTANCE,
        direction=message_pb2.MESSAGE_DIRECTION_INBOUND,
        external_user_id="mock-user-1",
        user_id="u-mock-1",
        text=message_pb2.TextContent(text=text),
    )


async def test_text_message_gets_prefixed_reply_with_channel_fields_echoed(
    harness: ConversationHarness,
) -> None:
    inbound = _inbound_text("01MSGTEXT0001", "客厅太挤了")
    response = cast(
        design_service_pb2.IngestMessageResponse,
        await harness.stub.IngestMessage(design_service_pb2.IngestMessageRequest(message=inbound)),
    )
    assert response.message_id == "01MSGTEXT0001"

    assert len(harness.captured.requests) == 1
    sent = harness.captured.requests[0]
    reply = sent.message
    assert reply.direction == message_pb2.MESSAGE_DIRECTION_OUTBOUND
    assert reply.channel_type == channel_type_pb2.CHANNEL_TYPE_MOCK
    assert reply.channel_instance == MOCK_INSTANCE
    assert reply.external_user_id == "mock-user-1"
    assert reply.user_id == "u-mock-1"
    assert reply.WhichOneof("content") == "text"
    assert reply.text.text.startswith(REPLY_PREFIX)
    assert "客厅太挤了" in reply.text.text
    assert reply.message_id != inbound.message_id  # 回话是新消息（ULID）
    assert sent.idempotency_key == "reply-01MSGTEXT0001"


async def test_duplicate_inbound_delivery_replies_only_once(
    harness: ConversationHarness,
) -> None:
    inbound = _inbound_text("01MSGDUP00001", "书房不要双人位")
    request = design_service_pb2.IngestMessageRequest(message=inbound)
    await harness.stub.IngestMessage(request)
    await harness.stub.IngestMessage(request)
    assert len(harness.captured.requests) == 1


async def test_image_message_gets_receipt_reply(harness: ConversationHarness) -> None:
    inbound = message_pb2.UnifiedMessage(
        message_id="01MSGIMG00001",
        channel_type=channel_type_pb2.CHANNEL_TYPE_MOCK,
        channel_instance=MOCK_INSTANCE,
        direction=message_pb2.MESSAGE_DIRECTION_INBOUND,
        external_user_id="mock-user-1",
        user_id="u-mock-1",
        image=message_pb2.ImageContent(image_url="oss://mock/floorplan.png", mime_type="image/png"),
    )
    await harness.stub.IngestMessage(design_service_pb2.IngestMessageRequest(message=inbound))
    assert len(harness.captured.requests) == 1
    reply_text = harness.captured.requests[0].message.text.text
    assert reply_text.startswith(REPLY_PREFIX)
    assert "图片" in reply_text


async def test_unimplemented_rpcs_signal_their_wiring_points(
    harness: ConversationHarness,
) -> None:
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await harness.stub.GetProject(design_service_pb2.GetProjectRequest(project_id="p-1"))
    assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
