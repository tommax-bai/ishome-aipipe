"""存取层：批次与任务启动记录（暂存内存；后续落 Postgres schema `svc_genpipe`，
禁止跨 schema join）。命名规范：get 必得 / find 可空 / list 集合 / count 计数。"""

from __future__ import annotations

from genpipe.models import WorkflowStartReceipt

_batch_receipts: dict[str, WorkflowStartReceipt] = {}
_task_receipts: dict[str, WorkflowStartReceipt] = {}


async def save_batch_receipt(batch_id: str, receipt: WorkflowStartReceipt) -> None:
    _batch_receipts[batch_id] = receipt


async def find_batch_receipt(batch_id: str) -> WorkflowStartReceipt | None:
    return _batch_receipts.get(batch_id)


async def find_batch_status(batch_id: str) -> str | None:
    return "started" if batch_id in _batch_receipts else None


async def save_task_receipt(task_id: str, receipt: WorkflowStartReceipt) -> None:
    _task_receipts[task_id] = receipt


async def find_task_receipt(task_id: str) -> WorkflowStartReceipt | None:
    return _task_receipts.get(task_id)


def reset_receipts() -> None:
    """测试专用：清空内存暂存。"""
    _batch_receipts.clear()
    _task_receipts.clear()
