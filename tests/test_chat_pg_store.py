"""PG 集成测试：连本地 PG（localhost:15432）实跑迁移执行与仓储读写；不可达时整文件 skip。

纪律：只动独立测试 schema `svc_chat_test`（每例前重建、跑完 DROP CASCADE 清理），
绝不触 svc_chat 真实数据，更绝不触 svc_project / svc_channel——本地 PG 实例与
其他服务共享。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import psycopg
import pytest
from chat.migrate import run_migrations
from chat.models import ChatMessage, ConversationRef
from chat.repo.pg import PgChatStore

DSN = os.environ.get(
    "CHAT_TEST_DATABASE_URL", "postgresql://ishome:ishome-local-dev@localhost:15432/ishome"
)
TEST_SCHEMA = "svc_chat_test"


def _pg_reachable() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_reachable(), reason="本地 PG（localhost:15432）不可达，跳过 PG 集成测试"
)


def _drop_test_schema() -> None:
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS svc_chat_test CASCADE")


@pytest.fixture
def migrated_schema() -> Iterator[str]:
    _drop_test_schema()
    applied = run_migrations(DSN, schema=TEST_SCHEMA)
    assert applied == ["V1__init_svc_chat.sql"]
    yield TEST_SCHEMA
    _drop_test_schema()


@pytest.fixture
async def store(migrated_schema: str) -> AsyncIterator[PgChatStore]:
    pg_store = PgChatStore(DSN, schema=migrated_schema)
    yield pg_store
    await pg_store.aclose()


def _ref(user: str = "u-pg") -> ConversationRef:
    return ConversationRef(channel_type=99, channel_instance="mock:pgtest", external_user_id=user)


def _inbound(idempotency_key: str = "m-1", text: str = "你好") -> ChatMessage:
    return ChatMessage(
        external_message_id=idempotency_key,
        direction="inbound",
        content_type="text",
        text=text,
        idempotency_key=idempotency_key,
    )


def test_migrations_apply_once_then_noop(migrated_schema: str) -> None:
    """迁移执行：V1 已应用（fixture 断言）；重复执行幂等；表清单符合首批口径。"""
    assert run_migrations(DSN, schema=migrated_schema) == []
    with psycopg.connect(DSN) as conn:
        rows = conn.execute(
            "SELECT version, filename FROM svc_chat_test.schema_migrations ORDER BY version"
        ).fetchall()
        assert rows == [(1, "V1__init_svc_chat.sql")]
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
                (migrated_schema,),
            ).fetchall()
        }
    assert {"conversations", "messages", "user_profiles", "commitments"} <= tables
    assert "episodic_memories" not in tables  # 本批不建：待 pgvector 镜像（V2）


@pytest.mark.asyncio
async def test_write_read_conversation_and_messages(store: PgChatStore) -> None:
    """写读会话与消息：入站/出站原文落表，会话行按渠道三元组建立并关联项目。"""
    ref = _ref()
    assert await store.record_inbound(ref, _inbound()) is True
    await store.record_outbound(
        ref,
        ChatMessage(
            external_message_id="o-1",
            direction="outbound",
            content_type="text",
            text="你好，我是你的设计顾问。",
            idempotency_key="reply-m-1-0",
        ),
    )

    conversation = await store.find_conversation(ref)
    assert conversation is not None
    assert conversation.channel_instance == "mock:pgtest"
    assert conversation.deleted_at is None

    records = await store.list_messages(ref)
    assert [(r.direction, r.content_type, r.text) for r in records] == [
        ("inbound", "text", "你好"),
        ("outbound", "text", "你好，我是你的设计顾问。"),
    ]
    assert all(r.conversation_id == conversation.id for r in records)
    assert all(len(r.id) == 26 for r in records)  # ULID 主键

    project = await store.find_or_create_project(ref)
    await store.save_project(ref, project)
    linked = await store.find_conversation(ref)
    assert linked is not None and linked.project_id == project.project_id


@pytest.mark.asyncio
async def test_idempotent_replay_not_restored(store: PgChatStore) -> None:
    """幂等重放不重存：入站重投返回 False，出站重试不产生重复行。"""
    ref = _ref("u-replay")
    assert await store.record_inbound(ref, _inbound("m-r1")) is True
    assert await store.record_inbound(ref, _inbound("m-r1")) is False

    outbound = ChatMessage(
        external_message_id="o-r1",
        direction="outbound",
        content_type="text",
        text="回话",
        idempotency_key="reply-m-r1-0",
    )
    await store.record_outbound(ref, outbound)
    await store.record_outbound(ref, outbound)

    records = await store.list_messages(ref)
    assert len(records) == 2
    assert sorted(r.idempotency_key for r in records) == ["m-r1", "reply-m-r1-0"]
