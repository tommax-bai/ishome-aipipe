"""FastAPI 入站适配：排产/任务/成文线触发与批次查询（router → service 单向）。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from temporalio.exceptions import WorkflowAlreadyStartedError

from genpipe.models import (
    GenBatchSpec,
    GenerationTaskSpec,
    ReportComposeSpec,
    WorkflowStartReceipt,
)
from genpipe.service import (
    get_batch_status,
    start_gen_batch,
    start_generation_task,
    start_report_compose,
)

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


@router.post("/reports", status_code=202)
async def create_report(spec: ReportComposeSpec) -> WorkflowStartReceipt:
    """成文线派发入口：求值线（project-svc）出报告数据包后调用，body 即 spec 原样透传。"""
    try:
        return await start_report_compose(spec)
    except WorkflowAlreadyStartedError as err:
        raise HTTPException(status_code=409, detail=f"report 已启动：{spec.report_id}") from err


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> dict[str, str]:
    status = await get_batch_status(batch_id)
    return {"batch_id": batch_id, "status": status}
