"""worker 进程装配：真连 Temporal（namespace: genpipe）并注册本仓全部 activities。

监听队列 `genpipe-activities`（contracts `registries/task_queues.md`）；重试/心跳/
取消/背压沿用 Temporal activity 原生语义，不引入服务间 HTTP 调用（对齐文档 §3.1）——
唯一的出站 HTTP 是 `task-result-deliver` 往**派发方注入的回调地址**送结论，那是回调不是调用。
优雅停止：SIGINT / SIGTERM 触发 worker graceful shutdown（等待在飞 activity 收尾）。

**组合根在此**：私有桶连接在这里装好并当场校验——装不上就起不来，绝不带着半套配置上线
等第一张图来了才发现取不到（同 render2d worker 的口径）。网关客户端按 `LITELLM_BASE_URL` /
`LITELLM_API_KEY` 装配（本机 `source ~/.ishome/llm-local.env`，
`LITELLM_API_KEY=$LITELLM_MASTER_KEY`）。
"""

from __future__ import annotations

import asyncio
import os
import signal

import httpx
from temporalio.client import Client
from temporalio.worker import Worker

from genpipe_worker.activities import FloorplanActivities, activity_registry
from genpipe_worker.llm_client import LiteLlmVisionClient
from genpipe_worker.object_store import ObjectStoreError, OssSettings, OssUploadStore

GENPIPE_NAMESPACE = "genpipe"
GENPIPE_TASK_QUEUE = "genpipe-activities"

_CALLBACK_TIMEOUT_SECONDS = 15.0
"""回调只做"把结论交出去"，业务侧登记是同步落库，秒级；超过即当瞬时故障交给重试。"""


def temporal_address() -> str:
    return os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")


def temporal_namespace() -> str:
    return os.environ.get("TEMPORAL_NAMESPACE", GENPIPE_NAMESPACE)


async def run_worker(address: str | None = None, namespace: str | None = None) -> None:
    try:
        store = OssUploadStore(OssSettings.from_env())
    except ObjectStoreError as e:
        # 缺配置是**运维要看的一句话**，不是给开发看的调用栈
        raise SystemExit("；".join(e.details)) from None
    llm = LiteLlmVisionClient()
    client = await Client.connect(
        address or temporal_address(),
        namespace=namespace or temporal_namespace(),
    )
    async with httpx.AsyncClient(timeout=_CALLBACK_TIMEOUT_SECONDS) as http:
        floorplan = FloorplanActivities(store, llm, http)
        worker = Worker(
            client,
            task_queue=GENPIPE_TASK_QUEUE,
            activities=list(activity_registry(floorplan).values()),
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
            await llm.aclose()


def main() -> None:
    """入口脚本 `genpipe-worker`（pyproject [project.scripts]）。"""
    asyncio.run(run_worker())
