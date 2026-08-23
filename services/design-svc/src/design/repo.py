"""存取层：ProjectState 持久化（Postgres schema `svc_design`，禁止跨 schema join）。

表结构见对齐文档 §5.1：projects / project_revisions / facts / decisions /
scenes / artifacts / open_questions / outbox（本地事务 + outbox 发事件）。
骨架阶段无真实存储实现；ORM 类命名规则：`ProjectStateRecord`（Record 后缀）。
"""

from __future__ import annotations

from design.models import ProjectState


async def find_project(project_id: str) -> ProjectState | None:
    """find = 可空查询。骨架阶段恒返回 None。"""
    _ = project_id
    return None


# 入站消息去重（幂等）。骨架阶段为进程内存实现；落库时归 svc_design 的
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
