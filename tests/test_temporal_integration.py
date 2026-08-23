"""集成冒烟：真连本地 Temporal（localhost:7233 / namespace genpipe）跑两个 workflow。

测试 Worker 把 mock activity 实现注册到唯一随机 task queue（spec.queues 全部收拢到
该队列），覆盖 GenBatchWorkflow 门禁通过 / 不通过自动重生成 / 重试超限失败三条路径，
以及 GenerationTaskWorkflow 的 atmosphere-visual 路由链。服务器不可达时整体 skip
（CI 无 Temporal 也能绿）；mock 只存在于测试内，生产 activity 存根保持 NotImplementedError。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from genpipe.models import GenBatchSpec, GenerationTaskSpec, TaskQueues
from genpipe.workflows import GenBatchWorkflow, GenerationTaskWorkflow
from temporalio import activity
from temporalio.api.workflowservice.v1 import DescribeNamespaceRequest
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
TEMPORAL_NAMESPACE = os.environ.get("TEMPORAL_NAMESPACE", "genpipe")

MockImpl = Callable[[Any], dict[str, Any]]
CallLog = list[tuple[str, Any]]


async def _client_or_skip() -> Client:
    try:
        client = await asyncio.wait_for(
            Client.connect(
                TEMPORAL_ADDRESS,
                namespace=TEMPORAL_NAMESPACE,
                data_converter=pydantic_data_converter,
            ),
            timeout=5,
        )
    except Exception as err:  # noqa: BLE001 - 环境探测：不可达即跳过
        pytest.skip(f"Temporal 服务器不可达（{TEMPORAL_ADDRESS}）：{err}")
    try:
        await client.workflow_service.describe_namespace(
            DescribeNamespaceRequest(namespace=TEMPORAL_NAMESPACE)
        )
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"Temporal namespace {TEMPORAL_NAMESPACE} 未注册：{err}")
    return client


def _make_mock_activities(behaviors: dict[str, MockImpl], log: CallLog) -> list[Callable[..., Any]]:
    """按注册名批量构造 mock activity（注册名与 contracts 逐字一致，实现仅测试内存在）。"""

    def _build(name: str, impl: MockImpl) -> Callable[..., Any]:
        @activity.defn(name=name)
        async def mock(arg: Any) -> dict[str, Any]:
            log.append((name, arg))
            return impl(arg)

        return mock

    return [_build(name, impl) for name, impl in behaviors.items()]


def _prep_behaviors(log_counter: dict[str, int]) -> dict[str, MockImpl]:
    """母版备制 + 候选生成的标准 mock：产物 id 递增可追踪。"""

    def atmosphere(_: Any) -> dict[str, Any]:
        log_counter["candidate"] = log_counter.get("candidate", 0) + 1
        return {"artifact_id": f"cand-{log_counter['candidate']}"}

    return {
        "plan-layout-solve": lambda _: {"plan_revision_id": "rev-1"},
        "plan-rule-check": lambda _: {"passed": True},
        "plan-2d-render": lambda _: {"artifact_id": "master-1"},
        "atmosphere-visual": atmosphere,
    }


async def _run_gen_batch(
    client: Client, behaviors: dict[str, MockImpl], log: CallLog, spec: GenBatchSpec
) -> Any:
    task_queue = spec.queues.genpipe
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[GenBatchWorkflow, GenerationTaskWorkflow],
        activities=_make_mock_activities(behaviors, log),
    )
    async with worker:
        return await client.execute_workflow(
            GenBatchWorkflow.run,
            spec,
            id=f"it-batch-{spec.batch_id}",
            task_queue=task_queue,
        )


def _batch_spec(queue: str, **overrides: Any) -> GenBatchSpec:
    queues = TaskQueues(genpipe=queue, render2d=queue, imagegen=queue, render3d=queue)
    defaults: dict[str, Any] = {
        "batch_id": uuid.uuid4().hex,
        "floorplan_id": "fp-1",
        "template_ids": ["tpl-a", "tpl-b", "tpl-c"],
        "candidate_count": 2,
        "queues": queues,
    }
    defaults.update(overrides)
    return GenBatchSpec(**defaults)


async def test_gen_batch_gate_pass_publishes_all_candidates() -> None:
    """门禁通过路径：母版备制 → 双候选 fan-out → 两检全过 → verdict passed。"""
    client = await _client_or_skip()
    log: CallLog = []
    counter: dict[str, int] = {}
    behaviors = _prep_behaviors(counter) | {
        "consistency-check": lambda _: {"passed": True},
        "compliance-check": lambda _: {"passed": True},
    }
    queue = f"it-{uuid.uuid4().hex}"
    result = await _run_gen_batch(client, behaviors, log, _batch_spec(queue))
    assert result.verdict == "passed"
    assert result.rounds == 1
    assert sorted(result.passed_artifact_ids) == ["cand-1", "cand-2"]
    assert log[0][0] == "plan-layout-solve"
    assert [name for name, _ in log].count("atmosphere-visual") == 2


async def test_gen_batch_gate_fail_regenerates_then_passes() -> None:
    """门禁不通过路径：首轮 consistency 拒绝 → 自动重生成（换模板）→ 次轮通过。"""
    client = await _client_or_skip()
    log: CallLog = []
    counter: dict[str, int] = {}

    def flaky_consistency(artifact_id: Any) -> dict[str, Any]:
        # 首个候选拒绝，之后放行：驱动一轮自动重生成
        return {"passed": artifact_id != "cand-1"}

    behaviors = _prep_behaviors(counter) | {
        "consistency-check": flaky_consistency,
        "compliance-check": lambda _: {"passed": True},
    }
    queue = f"it-{uuid.uuid4().hex}"
    spec = _batch_spec(queue, candidate_count=1, max_regen_rounds=2)
    result = await _run_gen_batch(client, behaviors, log, spec)
    assert result.verdict == "passed"
    assert result.rounds == 2
    assert result.passed_artifact_ids == ["cand-2"]
    # 重生成轮用了下一个模板（轮换纯函数在真实链路中生效）
    atmosphere_templates = [arg["template_id"] for name, arg in log if name == "atmosphere-visual"]
    assert atmosphere_templates == ["tpl-a", "tpl-b"]


async def test_gen_batch_gate_exhausted_returns_failed_verdict() -> None:
    """重试超限路径：compliance 持续拒绝 → 超限返回 failed verdict，绝不静默假成功。"""
    client = await _client_or_skip()
    log: CallLog = []
    counter: dict[str, int] = {}
    behaviors = _prep_behaviors(counter) | {
        "consistency-check": lambda _: {"passed": True},
        "compliance-check": lambda _: {"passed": False},
    }
    queue = f"it-{uuid.uuid4().hex}"
    spec = _batch_spec(queue, candidate_count=1, max_regen_rounds=1)
    result = await _run_gen_batch(client, behaviors, log, spec)
    assert result.verdict == "failed"
    assert result.rounds == 2
    assert result.passed_artifact_ids == []
    assert any("compliance-check" in failure for failure in result.failed_checks)


async def test_generation_task_atmosphere_chain_routes_and_gates() -> None:
    """GenerationTaskWorkflow 路由链：atmosphere-visual → realism-pass → 门禁收尾。"""
    client = await _client_or_skip()
    log: CallLog = []
    behaviors: dict[str, MockImpl] = {
        "atmosphere-visual": lambda _: {"artifact_id": "atm-1"},
        "realism-pass": lambda arg: {
            "artifact_id": "real-1",
            "base": arg["base_render_artifact_id"],
        },
        "consistency-check": lambda _: {"passed": True},
        "compliance-check": lambda _: {"passed": True},
    }
    queue = f"it-{uuid.uuid4().hex}"
    queues = TaskQueues(genpipe=queue, render2d=queue, imagegen=queue, render3d=queue)
    spec = GenerationTaskSpec(
        task_id=uuid.uuid4().hex,
        task_type="atmosphere-visual",
        params={"plan_master_artifact_id": "master-1", "template_id": "tpl-a", "style_ref": "s-1"},
        queues=queues,
    )
    worker = Worker(
        client,
        task_queue=queue,
        workflows=[GenBatchWorkflow, GenerationTaskWorkflow],
        activities=_make_mock_activities(behaviors, log),
    )
    async with worker:
        result = await client.execute_workflow(
            GenerationTaskWorkflow.run,
            spec,
            id=f"it-task-{spec.task_id}",
            task_queue=queue,
        )
    assert result.verdict == "passed"
    assert result.artifact_ids == ["atm-1", "real-1"]
    # 链路顺序 + 上游产物注入 + 门禁收尾在最终产物上执行
    assert [name for name, _ in log] == [
        "atmosphere-visual",
        "realism-pass",
        "consistency-check",
        "compliance-check",
    ]
    realism_arg = log[1][1]
    assert realism_arg["base_render_artifact_id"] == "atm-1"
    assert log[2][1] == "real-1"
