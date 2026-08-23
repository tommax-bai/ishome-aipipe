"""存取层：批次与门禁状态持久化（Postgres schema `svc_genpipe`，禁止跨 schema join）。

骨架阶段无真实存储实现。
"""

from __future__ import annotations


async def find_batch_status(batch_id: str) -> str | None:
    """find = 可空查询（命名规范：get 必得 / find 可空 / list 集合 / count 计数）。"""
    _ = batch_id
    return None
