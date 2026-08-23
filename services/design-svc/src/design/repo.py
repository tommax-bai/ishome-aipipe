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
