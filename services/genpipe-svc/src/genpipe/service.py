"""用例层：批次/任务生命周期编排；Temporal client（懒连接）与 workflow 启动收口在此。

start = 启动即返回（workflow_id / run_id 回执）：任务层异步，结果经事件回流，
不在请求内等待（对齐文档 §2.3）。
"""

from __future__ import annotations

import asyncio
import os

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from genpipe import repo
from genpipe.models import GenBatchSpec, GenerationTaskSpec, WorkflowStartReceipt
from genpipe.workflows import WORKFLOW_TASK_QUEUE, GenBatchWorkflow, GenerationTaskWorkflow

GENPIPE_NAMESPACE = "genpipe"

_client: Client | None = None
_client_lock = asyncio.Lock()


def temporal_address() -> str:
    return os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")


def temporal_namespace() -> str:
    return os.environ.get("TEMPORAL_NAMESPACE", GENPIPE_NAMESPACE)


async def get_temporal_client() -> Client:
    """懒连接单例：首个用例触发连接，此后复用（pydantic data converter 全链一致）。"""
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = await Client.connect(
                    temporal_address(),
                    namespace=temporal_namespace(),
                    data_converter=pydantic_data_converter,
                )
    return _client


def reset_temporal_client() -> None:
    """测试专用：丢弃懒连接单例。"""
    global _client
    _client = None


async def start_gen_batch(spec: GenBatchSpec) -> WorkflowStartReceipt:
    """启动批量预生成 workflow；workflow_id 以 batch_id 定址（重复提交即冲突上抛）。"""
    client = await get_temporal_client()
    handle = await client.start_workflow(
        GenBatchWorkflow.run,
        spec,
        id=f"gen-batch-{spec.batch_id}",
        task_queue=WORKFLOW_TASK_QUEUE,
    )
    receipt = WorkflowStartReceipt(workflow_id=handle.id, run_id=handle.result_run_id or "")
    await repo.save_batch_receipt(spec.batch_id, receipt)
    return receipt


async def start_generation_task(spec: GenerationTaskSpec) -> WorkflowStartReceipt:
    """启动交互侧生成任务 workflow（project-svc"创建任务 → 启动 workflow"的入口）。"""
    client = await get_temporal_client()
    handle = await client.start_workflow(
        GenerationTaskWorkflow.run,
        spec,
        id=f"gen-task-{spec.task_id}",
        task_queue=WORKFLOW_TASK_QUEUE,
    )
    receipt = WorkflowStartReceipt(workflow_id=handle.id, run_id=handle.result_run_id or "")
    await repo.save_task_receipt(spec.task_id, receipt)
    return receipt


async def get_batch_status(batch_id: str) -> str:
    """get = 必得；未落暂存的批次按 unknown 返回。"""
    status = await repo.find_batch_status(batch_id)
    return status or "unknown"
