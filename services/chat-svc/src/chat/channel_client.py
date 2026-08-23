"""出站边缘：channel-svc gRPC 客户端（contracts 生成 stub，禁手写协议）。

依赖方向（import-linter 锁定）：本模块只依赖 contracts 生成代码与运行库，
不感知 service/router 等上层——由组合根（grpc_server.serve）注入 service 层
的 OutboundSender 协议位。联调契约：channel-svc gRPC 默认 localhost:9102
（env `CHANNEL_GRPC_TARGET` 覆盖）。
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, cast

import grpc
from ishome.channel.v1 import message_pb2, service_pb2, service_pb2_grpc
from ishome.common.v1 import channel_type_pb2

DEFAULT_CHANNEL_GRPC_TARGET = "localhost:9102"


class ChannelClient:
    """ChannelService.SendMessage 的薄封装（实现 service.OutboundSender 协议）。"""

    def __init__(self, target: str = DEFAULT_CHANNEL_GRPC_TARGET) -> None:
        self.target = target
        self._channel: Any | None = None

    def _stub(self) -> Any:
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(self.target)
        return service_pb2_grpc.ChannelServiceStub(self._channel)

    async def send(self, message: message_pb2.UnifiedMessage, idempotency_key: str) -> str:
        """出站发送；幂等键随行（发进聊天线程的消息不可撤回，重试不得重发）。"""
        request = service_pb2.SendMessageRequest(message=message, idempotency_key=idempotency_key)
        response = cast(
            service_pb2.SendMessageResponse,
            await self._stub().SendMessage(request),
        )
        return response.message_id

    async def supports_quick_reply(self, channel_type: int, channel_instance: str) -> bool:
        """能力查询薄访问器（实现 service.CapabilityLookup 协议；按能力分支，R5）。"""
        request = service_pb2.GetCapabilityRequest(
            # 协议位用 int（proto 枚举 ValueType 即 int NewType），构造处收窄
            channel_type=cast("channel_type_pb2.ChannelType", channel_type),
            channel_instance=channel_instance,
        )
        response = cast(
            service_pb2.GetCapabilityResponse,
            await self._stub().GetCapability(request),
        )
        return bool(response.capability.supports_quick_reply)

    async def aclose(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None

    async def __aenter__(self) -> ChannelClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
