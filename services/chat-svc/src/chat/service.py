"""用例层：设计会话编排入口（Design Orchestrator v1 已接入）。

流程：入站落存（svc_chat.messages，幂等键防重存兼去重门）→ 输入归一化
（v1：quick_reply 直通；TODO(normalize) 多消息聚合、语音转文字）→ Intent Router
→ Orchestrator（事实抽取 + 回复）→ 确认闭环（最小必要输入集齐 → 文本版确认
清单 → user_confirmed 升级）→ 出站回话（发送后落存出站原文）。

存储：`CHAT_DATABASE_URL` 设置时消息原文落 PG（schema svc_chat），未设时内存
（e2e-mock-smoke 裸起可跑）——选择在 repo 层，本层不感知。会话态（项目快照/
上下文历史）为进程内缓存，Redis 接入位在 repo.SessionCache。

- LLM 一律经 LiteLLM 网关（llm_client），业务只引用任务级逻辑模型名；
- 结构类红线（§8.3）：口述结构信息永不进入可确认集合，回复附两条路径说明；
- TODO(identity)：会话键 → identity 归一 user_id；
- TODO(project-svc)：确认完成 → artifact_confirmed 业务事实发往 project-svc
  （V1.5：里程碑引擎事件驱动，原设计项目长周期 workflow 方案作废）；
- TODO(h5-pointing)：看图点错确认形态（H5 指图时刻）。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Protocol

from ishome.channel.v1 import message_pb2
from ulid import ULID

from chat import intent as intent_router
from chat import orchestrator
from chat.models import ChatMessage, ConversationRef, ConversationTurn, ProjectState
from chat.repo import (
    append_history,
    find_or_create_project,
    find_project,
    get_history,
    record_inbound,
    record_outbound,
    save_project,
)

logger = logging.getLogger(__name__)

FALLBACK_REPLY = "这条我没处理好，麻烦再发一次。"
"""LLM 或编排故障时的兜底回话——每条入站必有一条出站（E2E 不变量）。"""

LlmCompletion = orchestrator.LlmCompletion
"""LLM 协议位（结构化子集；实现 llm_client.LiteLlmClient，测试 FakeLLM）。"""


class OutboundSender(Protocol):
    """出站回话协议位：service 层不感知具体渠道客户端（组合根注入）。"""

    async def send(self, message: message_pb2.UnifiedMessage, idempotency_key: str) -> str: ...


class CapabilityLookup(Protocol):
    """渠道能力查询协议位（按能力分支不按身份分支，R5）。"""

    async def supports_quick_reply(self, channel_type: int, channel_instance: str) -> bool: ...


async def get_project(project_id: str) -> ProjectState:
    """get = 必得（取不到抛异常）。"""
    project = await find_project(project_id)
    if project is None:
        raise KeyError(f"project not found: {project_id}")
    return project


async def ingest_message(
    inbound: message_pb2.UnifiedMessage,
    sender: OutboundSender,
    llm: LlmCompletion,
    capability: CapabilityLookup | None = None,
) -> str:
    """会话入站处理；返回入站 message_id。"""
    conversation = _conversation_ref(inbound)
    # 入站原文落存即幂等门：幂等键（=渠道消息 id）已存过说明是渠道重投，跳过
    if not await record_inbound(conversation, _inbound_message(inbound)):
        logger.info("duplicate inbound skipped: message_id=%s", inbound.message_id)
        return inbound.message_id

    logger.info(
        "inbound message: message_id=%s channel=%s/%s content=%s",
        inbound.message_id,
        inbound.channel_type,
        inbound.channel_instance,
        inbound.WhichOneof("content"),
    )
    project = await find_or_create_project(conversation)
    user_text = _inbound_text(inbound)

    try:
        reply_texts, quick_reply_checklist = await _converse(
            inbound, project, conversation, user_text, llm, capability
        )
    except Exception:
        logger.exception("conversation turn failed: message_id=%s", inbound.message_id)
        reply_texts, quick_reply_checklist = [FALLBACK_REPLY], None

    await append_history(conversation, ConversationTurn(role="user", text=user_text))
    outbounds = [_text_reply(inbound, text) for text in reply_texts]
    if quick_reply_checklist is not None:
        outbounds.append(_quick_reply_checklist(inbound, quick_reply_checklist))
    for seq, outbound in enumerate(outbounds):
        # 幂等键从入站消息派生：同一入站消息的回话重试不会在聊天线程里发两遍
        idempotency_key = f"reply-{inbound.message_id}-{seq}"
        await sender.send(outbound, idempotency_key=idempotency_key)
        await record_outbound(conversation, _outbound_message(outbound, idempotency_key))
        await append_history(
            conversation, ConversationTurn(role="assistant", text=_outbound_text(outbound))
        )
        logger.info(
            "reply sent: message_id=%s in_reply_to=%s", outbound.message_id, inbound.message_id
        )
    await save_project(conversation, project)
    return inbound.message_id


async def _converse(
    inbound: message_pb2.UnifiedMessage,
    project: ProjectState,
    conversation: ConversationRef,
    user_text: str,
    llm: LlmCompletion,
    capability: CapabilityLookup | None,
) -> tuple[list[str], str | None]:
    """一轮会话：返回（文本回话列表, 需 quick_reply 形态发送的确认清单文本或 None）。"""
    checklist_open = bool(project.open_confirmation_ids)
    intent = await _route(inbound, user_text, llm, checklist_open=checklist_open)

    if intent == "confirm_checklist" and checklist_open:
        upgraded = orchestrator.upgrade_confirmed(project)
        logger.info("checklist confirmed: project=%s upgraded=%d", project.project_id, upgraded)
        return [orchestrator.confirm_ack_text()], None

    turn = await orchestrator.step(llm, project, await get_history(conversation), user_text)
    structural = orchestrator.merge_facts(project, turn.facts)
    # 修正已确认信息 → 撤下确认标记，走重新确认回路
    if project.minimum_inputs_confirmed and any(
        f.cognitive_state != "user_confirmed" for f in orchestrator.confirmable_facts(project)
    ):
        project.minimum_inputs_confirmed = False

    reply_text = turn.reply or FALLBACK_REPLY
    if structural:
        reply_text = f"{reply_text}\n\n{orchestrator.structural_note()}"

    if project.minimum_inputs_confirmed or orchestrator.missing_slots(project):
        return [reply_text], None

    # 最小必要输入集齐：出确认清单（§8.2 文本形态）
    checklist_text, open_ids = orchestrator.build_checklist_text(project)
    project.open_confirmation_ids = open_ids
    logger.info(
        "confirmation checklist issued: project=%s items=%d", project.project_id, len(open_ids)
    )
    texts = [reply_text] if turn.reply else []
    if await _supports_quick_reply(inbound, capability):
        return texts, checklist_text
    return [*texts, checklist_text], None


async def _route(
    inbound: message_pb2.UnifiedMessage,
    user_text: str,
    llm: LlmCompletion,
    *,
    checklist_open: bool,
) -> intent_router.Intent:
    """意图路由；quick_reply 选择直通（输入归一化 v1 路径），文本走分类模型。"""
    if inbound.WhichOneof("content") == "quick_reply":
        selected = inbound.quick_reply.selected_option_id
        if selected == orchestrator.CONFIRM_OPTION_ID:
            return "confirm_checklist"
        if selected == orchestrator.CORRECT_OPTION_ID:
            return "correct_checklist"
        return "other"
    return await intent_router.route_intent(llm, user_text, checklist_open=checklist_open)


async def _supports_quick_reply(
    inbound: message_pb2.UnifiedMessage, capability: CapabilityLookup | None
) -> bool:
    if capability is None:
        return False
    try:
        return await capability.supports_quick_reply(inbound.channel_type, inbound.channel_instance)
    except Exception:
        # 能力查询失败按不支持降级（纯文本清单照发，流程不断）
        logger.warning("capability lookup failed, degrade to text checklist", exc_info=True)
        return False


def _conversation_ref(inbound: message_pb2.UnifiedMessage) -> ConversationRef:
    # TODO(identity)：identity 归一后改为渠道无关 user_id 键控（对齐 §6.5）
    return ConversationRef(
        channel_type=inbound.channel_type,
        channel_instance=inbound.channel_instance,
        external_user_id=inbound.external_user_id or inbound.user_id,
    )


def _inbound_message(inbound: message_pb2.UnifiedMessage) -> ChatMessage:
    """入站原文的持久化形态；幂等键 = 渠道消息 id（渠道重投防重存）。"""
    return ChatMessage(
        external_message_id=inbound.message_id,
        direction="inbound",
        content_type=inbound.WhichOneof("content") or "unknown",
        text=_inbound_text(inbound),
        idempotency_key=inbound.message_id,
        occurred_at=_occurred_at(inbound),
    )


def _outbound_message(outbound: message_pb2.UnifiedMessage, idempotency_key: str) -> ChatMessage:
    """出站原文的持久化形态；幂等键与发送键同源（重试重放不重存）。"""
    return ChatMessage(
        external_message_id=outbound.message_id,
        direction="outbound",
        content_type=outbound.WhichOneof("content") or "unknown",
        text=_outbound_text(outbound),
        idempotency_key=idempotency_key,
        occurred_at=_occurred_at(outbound),
    )


def _occurred_at(message: message_pb2.UnifiedMessage) -> datetime | None:
    if not message.HasField("occurred_at"):
        return None
    return message.occurred_at.ToDatetime(tzinfo=UTC)


def _inbound_text(inbound: message_pb2.UnifiedMessage) -> str:
    match inbound.WhichOneof("content"):
        case "text":
            return inbound.text.text
        case "quick_reply":
            return f"[用户选择了：{inbound.quick_reply.selected_option_id}]"
        case "image":
            return "[用户发来一张图片]"  # TODO(genpipe)：floorplan-parse 备用路径接入
        case "audio":
            return "[用户发来一条语音]"  # TODO(normalize)：语音转文字
        case "card":
            return "[用户分享了一张卡片]"
        case _:
            return "[用户发来一条消息]"


def _text_reply(inbound: message_pb2.UnifiedMessage, text: str) -> message_pb2.UnifiedMessage:
    reply = _reply_envelope(inbound)
    reply.text.CopyFrom(message_pb2.TextContent(text=text))
    return reply


def _quick_reply_checklist(
    inbound: message_pb2.UnifiedMessage, checklist_text: str
) -> message_pb2.UnifiedMessage:
    reply = _reply_envelope(inbound)
    reply.quick_reply.CopyFrom(
        message_pb2.QuickReplyContent(
            prompt_text=checklist_text,
            options=[
                message_pb2.QuickReplyOption(
                    option_id=orchestrator.CONFIRM_OPTION_ID, label="确认无误"
                ),
                message_pb2.QuickReplyOption(
                    option_id=orchestrator.CORRECT_OPTION_ID, label="有要修正的"
                ),
            ],
        )
    )
    return reply


def _reply_envelope(inbound: message_pb2.UnifiedMessage) -> message_pb2.UnifiedMessage:
    reply = message_pb2.UnifiedMessage(
        message_id=str(ULID()),
        channel_type=inbound.channel_type,
        channel_instance=inbound.channel_instance,
        direction=message_pb2.MESSAGE_DIRECTION_OUTBOUND,
        external_user_id=inbound.external_user_id,
        user_id=inbound.user_id,
    )
    reply.occurred_at.GetCurrentTime()
    return reply


def _outbound_text(outbound: message_pb2.UnifiedMessage) -> str:
    if outbound.WhichOneof("content") == "quick_reply":
        return outbound.quick_reply.prompt_text
    return outbound.text.text
