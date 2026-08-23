"""workflow worker 进程装配：真连 Temporal（namespace: genpipe）并注册本服务 workflow。

监听 workflow 专属队列 `genpipe-workflows`（服务内约定，非跨服务契约）；activity
执行不在本进程——workflow 按 contracts `registries/task_queues.md` 把 activity 派往
genpipe-worker 与三个绘图服务各自的队列。优雅停止：SIGINT / SIGTERM。
"""

from __future__ import annotations

import asyncio
import os
import signal

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from genpipe.workflows import WORKFLOW_TASK_QUEUE, GenBatchWorkflow, GenerationTaskWorkflow

GENPIPE_NAMESPACE = "genpipe"


def temporal_address() -> str:
    return os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")


def temporal_namespace() -> str:
    return os.environ.get("TEMPORAL_NAMESPACE", GENPIPE_NAMESPACE)


async def run_worker(address: str | None = None, namespace: str | None = None) -> None:
    client = await Client.connect(
        address or temporal_address(),
        namespace=namespace or temporal_namespace(),
        data_converter=pydantic_data_converter,
    )
    worker = Worker(
        client,
        task_queue=WORKFLOW_TASK_QUEUE,
        workflows=[GenBatchWorkflow, GenerationTaskWorkflow],
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        async with worker:
            await stop.wait()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def main() -> None:
    """入口脚本 `genpipe-workflow-worker`（pyproject [project.scripts]）。"""
    asyncio.run(run_worker())
