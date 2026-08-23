"""worker 进程装配：连接 Temporal（namespace: genpipe）并注册全部 activities。"""

from __future__ import annotations

from temporalio.client import Client
from temporalio.worker import Worker

from genpipe_worker.activities import ACTIVITY_REGISTRY

GENPIPE_NAMESPACE = "genpipe"
GENPIPE_TASK_QUEUE = "genpipe-activities"


async def run_worker(temporal_address: str) -> None:
    client = await Client.connect(temporal_address, namespace=GENPIPE_NAMESPACE)
    worker = Worker(
        client,
        task_queue=GENPIPE_TASK_QUEUE,
        activities=list(ACTIVITY_REGISTRY.values()),
    )
    await worker.run()
