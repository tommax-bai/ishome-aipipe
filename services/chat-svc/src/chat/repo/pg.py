"""PG 实现：psycopg3 直写 SQL（轻量，禁 ORM）；行记录类 XxxRecord 与 pydantic 领域模型区分。

- 本批落库：conversations / messages（入站与出站消息原文，幂等键防重存）；
  user_profiles / commitments 建表已就绪（migrations/V1），写入随画像与承诺
  功能接入；episodic_memories 待 pgvector（V2）。
- 会话态（ProjectState 快照 / LLM 上下文历史）不落 PG：复用 memory.SessionCache
  （TODO(redis)：会话态归 Redis 的接入位即该类）。
- 红线：槽位真相唯一在 svc_project.slots（project-svc 属主）——本实现永不写
  槽位真相，禁止跨 schema join；SQL 全部显式限定本服务 schema。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from psycopg import AsyncConnection, sql
from psycopg.rows import TupleRow, class_row
from psycopg_pool import AsyncConnectionPool
from ulid import ULID

from chat.models import ChatMessage, ConversationRef, ConversationTurn, ProjectState
from chat.repo.memory import SessionCache


@dataclass(frozen=True)
class ConversationRecord:
    """svc_chat.conversations 行记录（Record 后缀：存储行，非领域模型）。"""

    id: str
    channel_type: int
    channel_instance: str
    external_user_id: str
    user_id: str | None
    project_id: str | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None


@dataclass(frozen=True)
class MessageRecord:
    """svc_chat.messages 行记录。"""

    id: str
    conversation_id: str
    direction: str
    content_type: str
    text: str
    external_message_id: str
    idempotency_key: str
    occurred_at: datetime | None
    created_at: datetime
    deleted_at: datetime | None


class PgChatStore:
    """ChatStore 的 PG 后端（psycopg AsyncConnectionPool，首次使用时懒打开）。"""

    def __init__(self, conninfo: str, schema: str = "svc_chat") -> None:
        self._schema = schema
        self._pool = AsyncConnectionPool(conninfo, min_size=1, max_size=4, open=False)
        self._open_lock = asyncio.Lock()
        self._opened = False
        self.session = SessionCache()

    async def aclose(self) -> None:
        await self._pool.close()

    # --- 消息落库（本批核心） ---

    async def record_inbound(self, conversation: ConversationRef, message: ChatMessage) -> bool:
        pool = await self._pool_ready()
        async with pool.connection() as conn, conn.transaction():
            conversation_id = await self._ensure_conversation(conn, conversation)
            return await self._insert_message(conn, conversation_id, message)

    async def record_outbound(self, conversation: ConversationRef, message: ChatMessage) -> None:
        pool = await self._pool_ready()
        async with pool.connection() as conn, conn.transaction():
            conversation_id = await self._ensure_conversation(conn, conversation)
            await self._insert_message(conn, conversation_id, message)

    # --- 会话期状态：进程内缓存（TODO(redis)：会话态归 Redis，接入位在 SessionCache） ---

    async def find_project(self, project_id: str) -> ProjectState | None:
        return self.session.find_project(project_id)

    async def find_or_create_project(self, conversation: ConversationRef) -> ProjectState:
        return self.session.find_or_create_project(conversation)

    async def save_project(self, conversation: ConversationRef, project: ProjectState) -> None:
        self.session.save_project(conversation, project)
        pool = await self._pool_ready()
        update = sql.SQL(
            "UPDATE {} SET project_id = %s, updated_at = now()"
            " WHERE channel_type = %s AND channel_instance = %s AND external_user_id = %s"
            " AND deleted_at IS NULL"
        ).format(self._table("conversations"))
        async with pool.connection() as conn:
            await conn.execute(
                update,
                (
                    project.project_id,
                    conversation.channel_type,
                    conversation.channel_instance,
                    conversation.external_user_id,
                ),
            )

    async def append_history(self, conversation: ConversationRef, turn: ConversationTurn) -> None:
        self.session.append_history(conversation, turn)

    async def get_history(self, conversation: ConversationRef) -> list[ConversationTurn]:
        return self.session.get_history(conversation)

    # --- 读侧（测试观察与后续 GetProject/ListProjects 落库实装的地基） ---

    async def find_conversation(self, conversation: ConversationRef) -> ConversationRecord | None:
        pool = await self._pool_ready()
        query = sql.SQL(
            "SELECT id, channel_type, channel_instance, external_user_id, user_id,"
            " project_id, created_at, updated_at, deleted_at FROM {}"
            " WHERE channel_type = %s AND channel_instance = %s AND external_user_id = %s"
        ).format(self._table("conversations"))
        async with (
            pool.connection() as conn,
            conn.cursor(row_factory=class_row(ConversationRecord)) as cur,
        ):
            await cur.execute(
                query,
                (
                    conversation.channel_type,
                    conversation.channel_instance,
                    conversation.external_user_id,
                ),
            )
            return await cur.fetchone()

    async def list_messages(self, conversation: ConversationRef) -> list[MessageRecord]:
        pool = await self._pool_ready()
        query = sql.SQL(
            "SELECT m.id, m.conversation_id, m.direction, m.content_type, m.text,"
            " m.external_message_id, m.idempotency_key, m.occurred_at, m.created_at,"
            " m.deleted_at FROM {} m JOIN {} c ON c.id = m.conversation_id"
            " WHERE c.channel_type = %s AND c.channel_instance = %s"
            " AND c.external_user_id = %s AND m.deleted_at IS NULL"
            " ORDER BY m.created_at, m.id"
        ).format(self._table("messages"), self._table("conversations"))
        async with (
            pool.connection() as conn,
            conn.cursor(row_factory=class_row(MessageRecord)) as cur,
        ):
            await cur.execute(
                query,
                (
                    conversation.channel_type,
                    conversation.channel_instance,
                    conversation.external_user_id,
                ),
            )
            return await cur.fetchall()

    # --- 内部 ---

    async def _pool_ready(self) -> AsyncConnectionPool:
        if not self._opened:
            async with self._open_lock:
                if not self._opened:
                    await self._pool.open()
                    self._opened = True
        return self._pool

    def _table(self, name: str) -> sql.Identifier:
        return sql.Identifier(self._schema, name)

    async def _ensure_conversation(
        self, conn: AsyncConnection[TupleRow], conversation: ConversationRef
    ) -> str:
        """会话行 upsert（自然键=渠道三元组）；软删行沿用不复活（v1 口径）。"""
        insert = sql.SQL(
            "INSERT INTO {} (id, channel_type, channel_instance, external_user_id)"
            " VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (channel_type, channel_instance, external_user_id) DO NOTHING"
            " RETURNING id"
        ).format(self._table("conversations"))
        params = (
            conversation.channel_type,
            conversation.channel_instance,
            conversation.external_user_id,
        )
        cur = await conn.execute(insert, (str(ULID()), *params))
        row = await cur.fetchone()
        if row is not None:
            return cast(str, row[0])
        select = sql.SQL(
            "SELECT id FROM {} WHERE channel_type = %s AND channel_instance = %s"
            " AND external_user_id = %s"
        ).format(self._table("conversations"))
        cur = await conn.execute(select, params)
        row = await cur.fetchone()
        if row is None:  # upsert 后必可见；防御性抛错而非静默
            raise RuntimeError(f"conversation upsert 后不可见：{conversation.key}")
        return cast(str, row[0])

    async def _insert_message(
        self, conn: AsyncConnection[TupleRow], conversation_id: str, message: ChatMessage
    ) -> bool:
        """幂等落存：唯一键 (conversation_id, idempotency_key) 冲突 → 不重存，返回 False。"""
        insert = sql.SQL(
            "INSERT INTO {} (id, conversation_id, direction, content_type, text,"
            " external_message_id, idempotency_key, occurred_at)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (conversation_id, idempotency_key) DO NOTHING RETURNING id"
        ).format(self._table("messages"))
        cur = await conn.execute(
            insert,
            (
                str(ULID()),
                conversation_id,
                message.direction,
                message.content_type,
                message.text,
                message.external_message_id,
                message.idempotency_key,
                message.occurred_at,
            ),
        )
        return await cur.fetchone() is not None
