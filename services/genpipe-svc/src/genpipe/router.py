"""FastAPI 入站适配：排产触发与批次查询（router → service 单向）。"""

from __future__ import annotations

from fastapi import APIRouter

from genpipe.service import get_batch_status

router = APIRouter(prefix="/api/v1/genpipe")


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> dict[str, str]:
    status = await get_batch_status(batch_id)
    return {"batch_id": batch_id, "status": status}
