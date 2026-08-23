"""存取层：会话与消息落库（Postgres schema `svc_chat`，禁止跨 schema join）。

V1.5 拆分后本服务持 `svc_chat`（conversations / messages / user_profiles /
commitments，表结构见对齐文档 §5.1；episodic_memories 待 pgvector 镜像后 V2
建表——见 migrations/V1 头注）。红线：槽位真相唯一在 `svc_project.slots`
（project-svc 属主），本层只存会话消息原文与画像/承诺，永不落槽位真相。

存储端口 `ChatStore` 双实现：
- `memory`（默认）：进程内存——单测与无 PG 场景（backend e2e-mock-smoke 裸起
  chat-grpc 必须可跑）；
- `pg`：env `CHAT_DATABASE_URL` 设置时启用（psycopg3 直写 SQL，禁 ORM；
  行记录类 XxxRecord 与 pydantic 领域模型区分）。

会话期状态（ProjectState 快照、LLM 上下文历史）两实现都放进程内——它们是
会话态而非事实真相，Redis 接入位在 `memory.SessionCache`（TODO(redis)）。
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from chat.models import ChatMessage, ConversationRef, ConversationTurn, ProjectState
from chat.repo.memory import HISTORY_MAX_TURNS as HISTORY_MAX_TURNS  # re-export（语义沿用）
from chat.repo.memory import MemoryChatStore

logger = logging.getLogger(__name__)


class ChatStore(Protocol):
    """存储端口（原 repo.py 模块函数语义收拢为协议位；实现见 memory / pg）。"""

    async def record_inbound(self, conversation: ConversationRef, message: ChatMessage) -> bool:
        """存入站消息原文；首见返回 True，幂等键重复返回 False（调用方跳过重复处理）。"""
        ...

    async def record_outbound(self, conversation: ConversationRef, message: ChatMessage) -> None:
        """存出站消息原文（幂等：同键重放不重存）。"""
        ...

    async def find_project(self, project_id: str) -> ProjectState | None:
        """find = 可空查询。"""
        ...

    async def find_or_create_project(self, conversation: ConversationRef) -> ProjectState:
        """按会话取项目快照，无则创建（v1 一人一项目；多项目管理后置）。"""
        ...

    async def save_project(self, conversation: ConversationRef, project: ProjectState) -> None: ...

    async def append_history(
        self, conversation: ConversationRef, turn: ConversationTurn
    ) -> None: ...

    async def get_history(self, conversation: ConversationRef) -> list[ConversationTurn]: ...


_store: ChatStore | None = None


def active_store() -> ChatStore:
    """当前后端（懒初始化）：`CHAT_DATABASE_URL` 设置 → PG，未设 → 内存。"""
    global _store
    if _store is None:
        url = os.environ.get("CHAT_DATABASE_URL")
        if url:
            # 懒导入：无 PG 场景（e2e-mock-smoke 裸起）不触 psycopg
            from chat.repo.pg import PgChatStore

            schema = os.environ.get("CHAT_DB_SCHEMA", "svc_chat")
            _store = PgChatStore(url, schema=schema)
            logger.info("chat storage backend: pg (schema=%s)", schema)
        else:
            _store = MemoryChatStore()
            logger.info("chat storage backend: memory (CHAT_DATABASE_URL 未设置)")
    return _store


def use_store(store: ChatStore | None) -> None:
    """注入/重置后端——仅供测试使用（None = 下次调用按 env 重选）。"""
    global _store
    _store = store


# --- 模块级委托函数（service 层的调用形态沿用原 repo.py） ---


async def record_inbound(conversation: ConversationRef, message: ChatMessage) -> bool:
    return await active_store().record_inbound(conversation, message)


async def record_outbound(conversation: ConversationRef, message: ChatMessage) -> None:
    await active_store().record_outbound(conversation, message)


async def find_project(project_id: str) -> ProjectState | None:
    return await active_store().find_project(project_id)


async def find_or_create_project(conversation: ConversationRef) -> ProjectState:
    return await active_store().find_or_create_project(conversation)


async def save_project(conversation: ConversationRef, project: ProjectState) -> None:
    await active_store().save_project(conversation, project)


async def append_history(conversation: ConversationRef, turn: ConversationTurn) -> None:
    await active_store().append_history(conversation, turn)


async def get_history(conversation: ConversationRef) -> list[ConversationTurn]:
    return await active_store().get_history(conversation)


def reset_conversations() -> None:
    """清空项目快照与上下文历史——仅供测试隔离使用（内存后端）。"""
    store = active_store()
    if isinstance(store, MemoryChatStore):
        store.session.reset()


def reset_messages() -> None:
    """清空消息记录与幂等键（原 reset_seen_messages）——仅供测试隔离使用（内存后端）。"""
    store = active_store()
    if isinstance(store, MemoryChatStore):
        store.reset_messages()
