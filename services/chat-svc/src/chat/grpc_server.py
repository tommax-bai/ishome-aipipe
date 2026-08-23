"""gRPC 入站适配：contracts `DesignService` 服务端（与 router 同级的入站层，兼组合根）。

契约入口沿用 contracts design.v1 `DesignService`——V1.5 更名 chat-svc 不动跨仓契约。

- 入站 → service 单向（import-linter 锁定，禁越层触 repo）；
- 联调契约：本服务默认监听 :9101（env `CHAT_GRPC_PORT`），出站回话经
  channel-svc gRPC（env `CHANNEL_GRPC_TARGET`，默认 localhost:9102）；
- 存储：env `CHAT_DATABASE_URL` 设置时会话消息落 PG（schema svc_chat，先跑
  `uv run chat-migrate` 建表），未设时内存（e2e-mock-smoke 裸起可跑）——
  后端选择在 repo 层，此处零装配；
- 启动方式：`uv run chat-grpc`。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import grpc
from ishome.design.v1 import service_pb2, service_pb2_grpc

from chat import service as chat_service
from chat.channel_client import DEFAULT_CHANNEL_GRPC_TARGET, ChannelClient
from chat.llm_client import LiteLlmClient

DEFAULT_CHAT_GRPC_PORT = 9101

logger = logging.getLogger(__name__)


class DesignGrpcServicer(service_pb2_grpc.DesignServiceServicer):
    """contracts DesignService 实现。会话入口已接 Orchestrator v1。"""

    def __init__(
        self,
        sender: chat_service.OutboundSender,
        llm: chat_service.LlmCompletion,
        capability: chat_service.CapabilityLookup | None = None,
    ) -> None:
        self._sender = sender
        self._llm = llm
        self._capability = capability

    async def IngestMessage(
        self, request: service_pb2.IngestMessageRequest, context: Any
    ) -> service_pb2.IngestMessageResponse:
        message_id = await chat_service.ingest_message(
            request.message, self._sender, self._llm, self._capability
        )
        return service_pb2.IngestMessageResponse(message_id=message_id)

    async def SubmitConfirmation(
        self, request: service_pb2.SubmitConfirmationRequest, context: Any
    ) -> service_pb2.SubmitConfirmationResponse:
        # 接入点：确认项升级 user_confirmed → artifact_confirmed 业务事实发往 project-svc
        # （V1.5：里程碑引擎事件驱动，原 DesignProjectWorkflow signal 方案作废）
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "确认闭环待接入 project-svc 事实上报")
        raise AssertionError("unreachable")  # abort 恒抛出；此行安抚类型检查

    async def SubmitPatch(
        self, request: service_pb2.SubmitPatchRequest, context: Any
    ) -> service_pb2.SubmitPatchResponse:
        # 接入点：patch_engine 校验 → 新 revision → outbox 事件 → 受影响产物重算
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Patch 引擎待接入")
        raise AssertionError("unreachable")

    async def GetProject(
        self, request: service_pb2.GetProjectRequest, context: Any
    ) -> service_pb2.GetProjectResponse:
        # 接入点：service.get_project → ProjectSummary 映射（待 repo 落库）
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "项目查询待 repo 落库后实装")
        raise AssertionError("unreachable")

    async def ListProjects(
        self, request: service_pb2.ListProjectsRequest, context: Any
    ) -> service_pb2.ListProjectsResponse:
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "项目查询待 repo 落库后实装")
        raise AssertionError("unreachable")


def build_server(
    sender: chat_service.OutboundSender,
    llm: chat_service.LlmCompletion,
    capability: chat_service.CapabilityLookup | None = None,
) -> Any:
    """组装 grpc.aio server（未绑定端口——测试用 :0 随机端口，serve 用配置端口）。"""
    server = grpc.aio.server()
    service_pb2_grpc.add_DesignServiceServicer_to_server(
        DesignGrpcServicer(sender, llm, capability), server
    )
    return server


async def serve() -> None:
    port = int(os.environ.get("CHAT_GRPC_PORT", DEFAULT_CHAT_GRPC_PORT))
    bind = os.environ.get("CHAT_GRPC_BIND", "0.0.0.0")
    channel_target = os.environ.get("CHANNEL_GRPC_TARGET", DEFAULT_CHANNEL_GRPC_TARGET)
    llm = LiteLlmClient()
    async with ChannelClient(channel_target) as channel_client:
        try:
            server = build_server(channel_client, llm, channel_client)
            server.add_insecure_port(f"{bind}:{port}")
            await server.start()
            logger.info(
                "chat-svc gRPC listening on :%d, channel target %s, llm gateway %s",
                port,
                channel_target,
                llm.base_url,
            )
            await server.wait_for_termination()
        finally:
            await llm.aclose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(serve())
