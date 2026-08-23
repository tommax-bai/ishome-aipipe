"""用例层：批次生命周期编排；事务边界与跨模块协调收口在此。"""

from __future__ import annotations

from genpipe.repo import find_batch_status


async def get_batch_status(batch_id: str) -> str:
    """get = 必得；骨架阶段未落库的批次按 unknown 返回。"""
    status = await find_batch_status(batch_id)
    return status or "unknown"
