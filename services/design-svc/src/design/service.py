"""用例层：设计会话编排（Design Orchestrator / Intent Router 落点）。

- Intent Router 前置输入归一化（语音转文字、短时间窗多消息聚合）归本层——
  渠道层不理解语义；
- LLM 一律经 LiteLLM 网关调用，Langfuse 逐会话记成本；
- Patch 校验经 patch_engine + genpipe plan-rule-check activity。
"""

from __future__ import annotations

import logging
from typing import Protocol

from ishome.channel.v1 import message_pb2
from ulid import ULID

from design.models import ProjectState
from design.repo import find_project, mark_message_seen

logger = logging.getLogger(__name__)

REPLY_PREFIX = "[design-svc] 收到你的消息："
"""回话前缀：纵切 E2E 断言依据（联调契约钉死，改动需与 channel-svc E2E 同步）。"""

_TEXT_SUMMARY_MAX_CHARS = 200


class OutboundSender(Protocol):
    """出站回话协议位：service 层不感知具体渠道客户端（组合根注入）。"""

    async def send(self, message: message_pb2.UnifiedMessage, idempotency_key: str) -> str: ...


async def get_project(project_id: str) -> ProjectState:
    """get = 必得（取不到抛异常）。"""
    project = await find_project(project_id)
    if project is None:
        raise KeyError(f"project not found: {project_id}")
    return project


async def ingest_message(inbound: message_pb2.UnifiedMessage, sender: OutboundSender) -> str:
    """会话入站处理（骨架回话）。

    TODO(Orchestrator)：此处将依次接入——输入归一化（语音转文字、短时间窗
    多消息聚合）→ Intent Router（意图路由）→ LLM（经 LiteLLM 网关，Langfuse
    逐会话记成本）→ `DesignProjectWorkflow` signal（workflows.py）。当前仅
    确认收到并回显，验证 channel ↔ design 双向链路。
    """
    if not await mark_message_seen(inbound.message_id):
        # 幂等：同一入站消息重复投递不重复回话
        logger.info("duplicate inbound skipped: message_id=%s", inbound.message_id)
        return inbound.message_id

    logger.info(
        "inbound message: message_id=%s channel=%s/%s content=%s",
        inbound.message_id,
        inbound.channel_type,
        inbound.channel_instance,
        inbound.WhichOneof("content"),
    )
    reply = message_pb2.UnifiedMessage(
        message_id=str(ULID()),
        channel_type=inbound.channel_type,
        channel_instance=inbound.channel_instance,
        direction=message_pb2.MESSAGE_DIRECTION_OUTBOUND,
        external_user_id=inbound.external_user_id,
        user_id=inbound.user_id,
        text=message_pb2.TextContent(text=_reply_text_for(inbound)),
    )
    reply.occurred_at.GetCurrentTime()
    # 幂等键从入站消息派生：同一入站消息的回话重试不会在聊天线程里发两遍
    await sender.send(reply, idempotency_key=f"reply-{inbound.message_id}")
    logger.info("reply sent: message_id=%s in_reply_to=%s", reply.message_id, inbound.message_id)
    return inbound.message_id


def _reply_text_for(inbound: message_pb2.UnifiedMessage) -> str:
    """按五类内容生成回话文本（统一模型词汇，无渠道方言）。"""
    match inbound.WhichOneof("content"):
        case "text":
            summary = inbound.text.text[:_TEXT_SUMMARY_MAX_CHARS]
        case "image":
            summary = "已收到你发来的图片"
        case "quick_reply":
            summary = f"已收到你的选择（{inbound.quick_reply.selected_option_id}）"
        case "audio":
            summary = "已收到你发来的语音"
        case "card":
            summary = "已收到你分享的卡片"
        case _:
            summary = "已收到你的消息"
    return f"{REPLY_PREFIX}{summary}"
