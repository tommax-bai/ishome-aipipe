"""gRPC 入站适配：contracts `DesignService` 服务端（与 router 同级的入站层，兼组合根）。

- 入站 → service 单向（import-linter 锁定，禁越层触 repo/workflows）；
- 联调契约：本服务默认监听 :9101（env `DESIGN_GRPC_PORT`），出站回话经
  channel-svc gRPC（env `CHANNEL_GRPC_TARGET`，默认 localhost:9102）；
- 启动方式：`uv run design-grpc`。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import grpc
from ishome.design.v1 import service_pb2, service_pb2_grpc

from design import service as design_service
from design.channel_client import DEFAULT_CHANNEL_GRPC_TARGET, ChannelClient
from design.llm_client import LiteLlmClient

DEFAULT_DESIGN_GRPC_PORT = 9101

logger = logging.getLogger(__name__)


class DesignGrpcServicer(service_pb2_grpc.DesignServiceServicer):
    """contracts DesignService 实现。会话入口已接 Orchestrator v1。"""

    def __init__(
        self,
        sender: design_service.OutboundSender,
        llm: design_service.LlmCompletion,
        capability: design_service.CapabilityLookup | None = None,
    ) -> None:
        self._sender = sender
        self._llm = llm
        self._capability = capability

    async def IngestMessage(
        self, request: service_pb2.IngestMessageRequest, context: Any
    ) -> service_pb2.IngestMessageResponse:
        message_id = await design_service.ingest_message(
            request.message, self._sender, self._llm, self._capability
        )
        return service_pb2.IngestMessageResponse(message_id=message_id)

    async def SubmitConfirmation(
        self, request: service_pb2.SubmitConfirmationRequest, context: Any
    ) -> service_pb2.SubmitConfirmationResponse:
        # 接入点：确认项升级 user_confirmed + DesignProjectWorkflow confirm_facts signal
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "确认闭环待接入 workflow signal")
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
    sender: design_service.OutboundSender,
    llm: design_service.LlmCompletion,
    capability: design_service.CapabilityLookup | None = None,
) -> Any:
    """组装 grpc.aio server（未绑定端口——测试用 :0 随机端口，serve 用配置端口）。"""
    server = grpc.aio.server()
    service_pb2_grpc.add_DesignServiceServicer_to_server(
        DesignGrpcServicer(sender, llm, capability), server
    )
    return server


async def serve() -> None:
    port = int(os.environ.get("DESIGN_GRPC_PORT", DEFAULT_DESIGN_GRPC_PORT))
    channel_target = os.environ.get("CHANNEL_GRPC_TARGET", DEFAULT_CHANNEL_GRPC_TARGET)
    llm = LiteLlmClient()
    async with ChannelClient(channel_target) as channel_client:
        try:
            server = build_server(channel_client, llm, channel_client)
            server.add_insecure_port(f"0.0.0.0:{port}")
            await server.start()
            logger.info(
                "design-svc gRPC listening on :%d, channel target %s, llm gateway %s",
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
