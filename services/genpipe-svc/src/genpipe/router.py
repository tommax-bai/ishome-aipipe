"""FastAPI 入站适配：排产/任务触发与批次查询（router → service 单向）。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from temporalio.exceptions import WorkflowAlreadyStartedError

from genpipe.models import GenBatchSpec, GenerationTaskSpec, WorkflowStartReceipt
from genpipe.service import get_batch_status, start_gen_batch, start_generation_task

router = APIRouter(prefix="/api/v1/genpipe")


@router.post("/batches", status_code=202)
async def create_batch(spec: GenBatchSpec) -> WorkflowStartReceipt:
    try:
        return await start_gen_batch(spec)
    except WorkflowAlreadyStartedError as err:
        raise HTTPException(status_code=409, detail=f"batch 已启动：{spec.batch_id}") from err


@router.post("/tasks", status_code=202)
async def create_task(spec: GenerationTaskSpec) -> WorkflowStartReceipt:
    try:
        return await start_generation_task(spec)
    except WorkflowAlreadyStartedError as err:
        raise HTTPException(status_code=409, detail=f"task 已启动：{spec.task_id}") from err


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> dict[str, str]:
    status = await get_batch_status(batch_id)
    return {"batch_id": batch_id, "status": status}
