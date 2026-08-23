"""内存实现（默认后端）：消息原文与幂等键只存进程内（重启即失）。

单测与无 PG 场景用；另提供 `SessionCache` 供 PG 后端复用会话态缓存。
"""

from __future__ import annotations

from ulid import ULID

from chat.models import ChatMessage, ConversationRef, ConversationTurn, ProjectState

HISTORY_MAX_TURNS = 20
"""会话历史有界保留（LLM 上下文窗），超出裁掉最旧。"""


class SessionCache:
    """会话期状态（项目快照 + LLM 上下文历史）——会话态，不是事实真相。

    TODO(redis)：对齐 §5.1 会话态（阶段/情绪轨迹/槽位缓存）归 Redis——接入时
    本类换 Redis 实现（键 = ConversationRef.key），存储端口不动。槽位真相唯一
    在 svc_project.slots，此处 ProjectState 仅为会话期缓存。
    """

    def __init__(self) -> None:
        self._projects: dict[str, ProjectState] = {}
        self._histories: dict[str, list[ConversationTurn]] = {}

    def find_project(self, project_id: str) -> ProjectState | None:
        return next((p for p in self._projects.values() if p.project_id == project_id), None)

    def find_or_create_project(self, conversation: ConversationRef) -> ProjectState:
        project = self._projects.get(conversation.key)
        if project is None:
            project = ProjectState(project_id=str(ULID()), user_id=conversation.key)
            self._projects[conversation.key] = project
        return project

    def save_project(self, conversation: ConversationRef, project: ProjectState) -> None:
        self._projects[conversation.key] = project

    def append_history(self, conversation: ConversationRef, turn: ConversationTurn) -> None:
        turns = self._histories.setdefault(conversation.key, [])
        turns.append(turn)
        del turns[:-HISTORY_MAX_TURNS]

    def get_history(self, conversation: ConversationRef) -> list[ConversationTurn]:
        return list(self._histories.get(conversation.key, []))

    def reset(self) -> None:
        """清空——仅供测试隔离使用。"""
        self._projects.clear()
        self._histories.clear()


class MemoryChatStore:
    """ChatStore 全内存实现。"""

    def __init__(self) -> None:
        self.session = SessionCache()
        self._messages: dict[str, list[ChatMessage]] = {}
        self._seen_keys: set[str] = set()

    async def record_inbound(self, conversation: ConversationRef, message: ChatMessage) -> bool:
        return self._record(conversation, message)

    async def record_outbound(self, conversation: ConversationRef, message: ChatMessage) -> None:
        self._record(conversation, message)

    async def find_project(self, project_id: str) -> ProjectState | None:
        return self.session.find_project(project_id)

    async def find_or_create_project(self, conversation: ConversationRef) -> ProjectState:
        return self.session.find_or_create_project(conversation)

    async def save_project(self, conversation: ConversationRef, project: ProjectState) -> None:
        self.session.save_project(conversation, project)

    async def append_history(self, conversation: ConversationRef, turn: ConversationTurn) -> None:
        self.session.append_history(conversation, turn)

    async def get_history(self, conversation: ConversationRef) -> list[ConversationTurn]:
        return self.session.get_history(conversation)

    def _record(self, conversation: ConversationRef, message: ChatMessage) -> bool:
        """幂等落存：同一会话内幂等键重复 → 不重存，返回 False（对齐 PG 唯一约束语义）。"""
        dedup_key = f"{conversation.key}#{message.idempotency_key}"
        if dedup_key in self._seen_keys:
            return False
        self._seen_keys.add(dedup_key)
        self._messages.setdefault(conversation.key, []).append(message)
        return True

    def list_messages(self, conversation: ConversationRef) -> list[ChatMessage]:
        """按落存顺序取消息——仅供测试观察使用。"""
        return list(self._messages.get(conversation.key, []))

    def reset_messages(self) -> None:
        """清空消息与幂等键——仅供测试隔离使用。"""
        self._messages.clear()
        self._seen_keys.clear()
