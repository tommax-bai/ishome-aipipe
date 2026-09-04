"""FastAPI 入站适配：排产/任务/成文线触发与批次查询（router → service 单向）。"""

from __future__ import annotations

import os

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from temporalio.exceptions import WorkflowAlreadyStartedError

from genpipe.models import (
    FloorplanVisualsSpec,
    GenBatchSpec,
    GenerationTaskSpec,
    ReportComposeSpec,
    WorkflowStartReceipt,
)
from genpipe.service import (
    get_batch_status,
    start_floorplan_visuals,
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


@router.post("/floorplan-visuals", status_code=202)
async def create_floorplan_visuals(spec: FloorplanVisualsSpec) -> WorkflowStartReceipt:
    """三张免费图派发入口：project-svc 铸任务后调用，body 即 spec 原样透传
    （contracts genpipe.v1）。"""
    try:
        return await start_floorplan_visuals(spec)
    except WorkflowAlreadyStartedError as err:
        raise HTTPException(status_code=409, detail=f"task 已启动：{spec.task_id}") from err


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> dict[str, str]:
    status = await get_batch_status(batch_id)
    return {"batch_id": batch_id, "status": status}


app = FastAPI(
    title="genpipe-svc",
    description=(
        "生成编排入站面。成文线派发通道走 HTTP 而非 Java Temporal SDK 直连"
        "（裁决④：直连会把里程碑引擎裁决刚收缩掉的 Temporal 依赖放回 Java 服务）。"
    ),
)
app.include_router(router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """存活探针：只答本进程活着。

    **不代下游背书**——不连 Temporal、不探 reportgen 队列：派发能不能成由派发本身
    的响应码说话，探针替下游打包票会把"网关活着"误读成"整条线活着"。
    """
    return {"status": "ok"}


def main() -> None:
    """入口脚本 `genpipe-http`（pyproject [project.scripts]）。

    监听地址由 `GENPIPE_HTTP_HOST` / `GENPIPE_HTTP_PORT` 覆盖；默认只绑回环——
    本地联调不对外暴露，部署形态由 infra 决定。
    """
    uvicorn.run(
        app,
        host=os.environ.get("GENPIPE_HTTP_HOST", "127.0.0.1"),
        port=int(os.environ.get("GENPIPE_HTTP_PORT", "8104")),
    )
