"""存储端口单测：内存实现语义 + 双实现按 env 切换（无 PG——PG 实跑见 test_chat_pg_store.py）。"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from chat.models import ChatMessage, ConversationRef, MessageDirection
from chat.repo import active_store, use_store
from chat.repo.memory import MemoryChatStore


@pytest.fixture(autouse=True)
def _fresh_store() -> Iterator[None]:
    """每例重置后端单例（按当次 env 重选），跑完恢复默认。"""
    use_store(None)
    yield
    use_store(None)


def _ref() -> ConversationRef:
    return ConversationRef(channel_type=99, channel_instance="mock:local", external_user_id="u-r")


def _message(direction: MessageDirection, idempotency_key: str, text: str = "hi") -> ChatMessage:
    return ChatMessage(
        external_message_id=idempotency_key,
        direction=direction,
        content_type="text",
        text=text,
        idempotency_key=idempotency_key,
    )


@pytest.mark.asyncio
async def test_memory_inbound_dedup_by_idempotency_key() -> None:
    store = MemoryChatStore()
    assert await store.record_inbound(_ref(), _message("inbound", "m-1")) is True
    assert await store.record_inbound(_ref(), _message("inbound", "m-1")) is False
    assert len(store.list_messages(_ref())) == 1


@pytest.mark.asyncio
async def test_memory_outbound_replay_not_duplicated() -> None:
    store = MemoryChatStore()
    await store.record_outbound(_ref(), _message("outbound", "reply-m-1-0"))
    await store.record_outbound(_ref(), _message("outbound", "reply-m-1-0"))
    assert len(store.list_messages(_ref())) == 1


def test_default_backend_is_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHAT_DATABASE_URL 未设 → 内存后端（backend e2e-mock-smoke 裸起 chat-grpc 依赖此路径）。"""
    monkeypatch.delenv("CHAT_DATABASE_URL", raising=False)
    assert isinstance(active_store(), MemoryChatStore)


def test_pg_backend_selected_when_url_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHAT_DATABASE_URL 设置 → PG 后端（构造不连库——连接池懒打开，离线可跑）。"""
    from chat.repo.pg import PgChatStore

    monkeypatch.setenv("CHAT_DATABASE_URL", "postgresql://u:p@localhost:1/nowhere")
    monkeypatch.setenv("CHAT_DB_SCHEMA", "svc_chat_test")
    store = active_store()
    assert isinstance(store, PgChatStore)
