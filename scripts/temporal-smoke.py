"""真实接线冒烟：起真 worker + 提交 GenBatchWorkflow，验证失败路径可观测。

流程：以入口脚本拉起两个真 worker 子进程（genpipe-worker 监听 genpipe-activities，
genpipe-workflow-worker 监听 genpipe-workflows）→ 提交 GenBatchWorkflow（默认队列）
→ 首个 activity `plan-layout-solve` 命中真实 NotImplementedError 存根 → 按
RetryPolicy（maximum_attempts=3）重试耗尽 → workflow 返回 failed verdict（绝不静默
假成功）。跑完 SIGTERM 杀干净后台 worker（顺带验证优雅停止）。

用法：uv run python scripts/temporal-smoke.py
需要本地 Temporal（localhost:7233，namespace genpipe；env TEMPORAL_ADDRESS /
TEMPORAL_NAMESPACE 可覆写）。CI 不跑本脚本（真网络）。
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from genpipe.models import GenBatchSpec
from genpipe.workflows import WORKFLOW_TASK_QUEUE, GenBatchWorkflow
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

REPO_ROOT = Path(__file__).resolve().parent.parent
RETRY_CAP = 3  # 与 workflows._ACTIVITY_RETRY.maximum_attempts 一致


def start_worker(entry: str) -> subprocess.Popen[bytes]:
    print(f"启动 worker 子进程：uv run {entry}")
    return subprocess.Popen(["uv", "run", entry], cwd=REPO_ROOT)


def stop_worker(name: str, proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    print(f"worker {name} 已停止（exit={proc.returncode}）")


async def main() -> int:
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "genpipe")
    client = await Client.connect(
        address, namespace=namespace, data_converter=pydantic_data_converter
    )
    workers = {
        "genpipe-worker": start_worker("genpipe-worker"),
        "genpipe-workflow-worker": start_worker("genpipe-workflow-worker"),
    }
    try:
        batch_id = f"smoke-{uuid.uuid4().hex[:8]}"
        spec = GenBatchSpec(
            batch_id=batch_id,
            floorplan_id="smoke-floorplan",
            template_ids=["tpl-smoke"],
            candidate_count=1,
        )
        started_at = time.monotonic()
        handle = await client.start_workflow(
            GenBatchWorkflow.run,
            spec,
            id=f"gen-batch-{batch_id}",
            task_queue=WORKFLOW_TASK_QUEUE,
        )
        print(f"已提交：workflow_id={handle.id}")
        print(f"UI 可观测：http://localhost:8233/namespaces/{namespace}/workflows/{handle.id}")
        result = await asyncio.wait_for(handle.result(), timeout=120)
        elapsed = time.monotonic() - started_at
        print(f"workflow 完成（{elapsed:.1f}s）：verdict={result.verdict}")
        print(f"failed_checks={result.failed_checks}")

        history = await handle.fetch_history()
        scheduled = [
            e.activity_task_scheduled_event_attributes.activity_type.name
            for e in history.events
            if e.HasField("activity_task_scheduled_event_attributes")
        ]
        final_attempts = [
            e.activity_task_started_event_attributes.attempt
            for e in history.events
            if e.HasField("activity_task_started_event_attributes")
        ]
        print(f"history：scheduled={scheduled}，最终 attempt={final_attempts}")

        ok = (
            result.verdict == "failed"
            and any("NotImplementedError" in check for check in result.failed_checks)
            and scheduled == ["plan-layout-solve"]
            and final_attempts == [RETRY_CAP]
        )
        if ok:
            print(
                f"SMOKE OK：plan-layout-solve NotImplementedError 重试 {RETRY_CAP} 次耗尽，"
                "workflow 以 failed verdict 收场（失败路径可观测）"
            )
            return 0
        print("SMOKE FAIL：结果与预期失败路径不符，见上方输出")
        return 1
    finally:
        for name, proc in workers.items():
            stop_worker(name, proc)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
