"""存取层：会话态与项目快照持久化（Postgres schema `svc_chat`，禁止跨 schema join）。

V1.5：原 `svc_design` 设计作废，拆为 `svc_project`（project-svc 属主，
slot/artifact/task/revision 唯一真相）+ `svc_chat`（本服务：会话、记忆、
承诺区，表结构见对齐文档 §5.1）。骨架阶段无真实存储实现；
ORM 类命名规则：`ProjectStateRecord`（Record 后缀）。
"""

from __future__ import annotations

from ulid import ULID

from chat.models import ConversationTurn, ProjectState

HISTORY_MAX_TURNS = 20
"""会话历史有界保留（LLM 上下文窗），超出裁掉最旧。"""


async def find_project(project_id: str) -> ProjectState | None:
    """find = 可空查询。骨架阶段恒返回 None。"""
    _ = project_id
    return None


# --- 会话键控的项目快照与历史（进程内存骨架；落库归 svc_chat 会话表） ---
# 会话键 = f"{channel_type}:{channel_instance}:{external_user_id}"。
# TODO(identity)：identity 归一后改为渠道无关的 user_id 键控（对齐 §6.5）。

_projects: dict[str, ProjectState] = {}
_histories: dict[str, list[ConversationTurn]] = {}


async def find_or_create_project(conversation_key: str) -> ProjectState:
    """按会话键取项目，无则创建（v1 一人一项目；多项目管理后置）。"""
    project = _projects.get(conversation_key)
    if project is None:
        project = ProjectState(project_id=str(ULID()), user_id=conversation_key)
        _projects[conversation_key] = project
    return project


async def save_project(conversation_key: str, project: ProjectState) -> None:
    _projects[conversation_key] = project


async def append_history(conversation_key: str, turn: ConversationTurn) -> None:
    turns = _histories.setdefault(conversation_key, [])
    turns.append(turn)
    del turns[:-HISTORY_MAX_TURNS]


async def get_history(conversation_key: str) -> list[ConversationTurn]:
    return list(_histories.get(conversation_key, []))


def reset_conversations() -> None:
    """清空项目与历史——仅供测试隔离使用。"""
    _projects.clear()
    _histories.clear()


# 入站消息去重（幂等）。骨架阶段为进程内存实现；落库时归 svc_chat 的
# 入站消息表（与 outbox 同事务），进程重启不丢。
_seen_message_ids: set[str] = set()


async def mark_message_seen(message_id: str) -> bool:
    """首见返回 True 并记录；重复返回 False（调用方据此跳过重复处理）。"""
    if message_id in _seen_message_ids:
        return False
    _seen_message_ids.add(message_id)
    return True


def reset_seen_messages() -> None:
    """清空去重记录——仅供测试隔离使用。"""
    _seen_message_ids.clear()
