"""Temporal workflow：只编排，禁止任何 IO（可重放硬约束，import-linter 契约锁定）。

机检门禁状态机（全自动，运行时无任何队列）：
    多候选生成 → 打分（scorer 阈值，打分为 packages/scoring 库能力、暂无独立 activity）
    → consistency-check → compliance-check → 达标 → 发布；
    不达标 → 自动重生成（带轮数上限，超限返回 failed verdict，绝不静默假成功）

activity 一律以 contracts 注册表中的注册名字符串 + task_queue 派发（跨服务不 import
存根签名），唯一真源：ishome-contracts `activities/registry.md` 与
`registries/task_queues.md`（只增不改）。V1.5 裁决：Temporal 收缩至任务层——
本文件只承载生成管线任务 workflow，无长周期项目 workflow。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    import pydantic_core  # noqa: F401  # pydantic 转换器运行时依赖：预载省去沙箱逐次告警

    from genpipe.models import (
        GenBatchSpec,
        GenerationTaskResult,
        GenerationTaskSpec,
        MachineGateResult,
        TaskStep,
    )

# activity 注册名常量（与 contracts 注册表逐字一致，只增不改）
ACTIVITY_PLAN_LAYOUT_SOLVE = "plan-layout-solve"
ACTIVITY_PLAN_RULE_CHECK = "plan-rule-check"
ACTIVITY_PLAN_2D_RENDER = "plan-2d-render"
ACTIVITY_ATMOSPHERE_VISUAL = "atmosphere-visual"
ACTIVITY_REALISM_PASS = "realism-pass"
ACTIVITY_SCENE_COMPILE = "scene-compile"
ACTIVITY_BASE_RENDER = "base-render"
ACTIVITY_CONSISTENCY_CHECK = "consistency-check"
ACTIVITY_COMPLIANCE_CHECK = "compliance-check"

WORKFLOW_TASK_QUEUE = "genpipe-workflows"
"""workflow 专属队列：起点（service.py）与执行者（genpipe workflow worker）同属本服务，
非跨服务契约，不进 contracts `registries/task_queues.md`（该表只登记 activity 队列归属）。"""

_COMPUTE_TIMEOUT = timedelta(minutes=5)
"""解析/求解/校验/门禁类 activity 的 start_to_close 上限。"""
_RENDER_TIMEOUT = timedelta(minutes=15)
"""绘图/渲染类长跑 activity 的 start_to_close 上限。"""
_RENDER_HEARTBEAT = timedelta(seconds=60)
"""长跑绘图 activity 必须心跳：worker 失联在一个心跳窗口内被发现并重派。"""
_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)
"""activity 级重试上限：耗尽后 ActivityError 上抛，由 workflow 记入 failed verdict。"""


class PipelineDataError(Exception):
    """管线数据形态违约（缺参/缺产物 id/非 dict 结果）：按失败处理，绝不静默假成功。"""


def evaluate_gate(consistency: object, compliance: object) -> list[str]:
    """机检门禁判定（纯函数，可直测）。

    两检都必须显式返回 passed=True 才算通过；结果缺失、形态异常一律按未通过。
    """
    failures: list[str] = []
    for check_name, result in (
        (ACTIVITY_CONSISTENCY_CHECK, consistency),
        (ACTIVITY_COMPLIANCE_CHECK, compliance),
    ):
        if not (isinstance(result, dict) and result.get("passed") is True):
            failures.append(check_name)
    return failures


def pick_template(template_ids: list[str], round_index: int, slot: int, fanout: int) -> str:
    """候选模板轮换（纯函数）：重生成轮次自动换到下一批模板。"""
    return template_ids[(round_index * fanout + slot) % len(template_ids)]


def _require_param(spec: GenerationTaskSpec, key: str) -> Any:
    value = spec.params.get(key)
    if value is None:
        raise PipelineDataError(f"missing-param:{key}")
    return value


def build_task_chain(spec: GenerationTaskSpec) -> list[TaskStep]:
    """task_type → activity 链路由（纯函数，可直测）。

    队列归属与 contracts `registries/task_queues.md` 一致；链路出处：对齐文档 §3.1。
    """
    queues = spec.queues
    if spec.task_type == "plan-2d-render":
        return [
            TaskStep(
                activity=ACTIVITY_PLAN_2D_RENDER,
                task_queue=queues.render2d,
                arg={
                    "revision_id": _require_param(spec, "revision_id"),
                    "purpose": spec.params.get("purpose", "plan_master"),
                },
                long_running=True,
            )
        ]
    if spec.task_type == "atmosphere-visual":
        return [
            TaskStep(
                activity=ACTIVITY_ATMOSPHERE_VISUAL,
                task_queue=queues.imagegen,
                arg={
                    "plan_master_artifact_id": _require_param(spec, "plan_master_artifact_id"),
                    "template_id": _require_param(spec, "template_id"),
                    "render_tier": spec.render_tier,
                },
                long_running=True,
            ),
            TaskStep(
                activity=ACTIVITY_REALISM_PASS,
                task_queue=queues.imagegen,
                arg={
                    "style_ref": _require_param(spec, "style_ref"),
                    "render_tier": spec.render_tier,
                },
                arg_from_upstream={"base_render_artifact_id": "artifact_id"},
                long_running=True,
            ),
        ]
    if spec.task_type == "scene-compile":
        return [
            TaskStep(
                activity=ACTIVITY_SCENE_COMPILE,
                task_queue=queues.render3d,
                arg=_require_param(spec, "deep_revision_id"),
                long_running=True,
            ),
            TaskStep(
                activity=ACTIVITY_BASE_RENDER,
                task_queue=queues.render3d,
                arg={
                    "camera_id": _require_param(spec, "camera_id"),
                    "render_tier": spec.render_tier,
                },
                arg_from_upstream={"scene_package_key": "scene_package_key"},
                long_running=True,
            ),
        ]
    raise PipelineDataError(f"unknown-task-type:{spec.task_type}")


def resolve_step_arg(step: TaskStep, upstream: dict[str, Any]) -> Any:
    """把上游 activity 结果并入本步入参（纯函数，可直测）；缺上游键按失败处理。"""
    if not step.arg_from_upstream:
        return step.arg
    if not isinstance(step.arg, dict):
        raise PipelineDataError(f"{step.activity}:arg-not-mergeable")
    merged: dict[str, Any] = dict(step.arg)
    for arg_key, upstream_key in step.arg_from_upstream.items():
        value = upstream.get(upstream_key)
        if value is None:
            raise PipelineDataError(f"{step.activity}:missing-upstream:{upstream_key}")
        merged[arg_key] = value
    return merged


def describe_failure(error: BaseException) -> str:
    """失败原因可观测化（纯函数）：ActivityError 取根因 type 与 message。"""
    if isinstance(error, PipelineDataError):
        return str(error)
    cause = error.__cause__ or error
    label = getattr(cause, "type", None) or type(cause).__name__
    message = str(cause)
    return f"{label}: {message}" if message else str(label)


async def _execute(
    activity_name: str,
    arg: Any,
    *,
    task_queue: str,
    long_running: bool = False,
) -> dict[str, Any]:
    """按注册名字符串派发 activity；显式 task_queue / 超时 / 重试，长跑加心跳窗口。"""
    result = await workflow.execute_activity(
        activity_name,
        arg,
        task_queue=task_queue,
        start_to_close_timeout=_RENDER_TIMEOUT if long_running else _COMPUTE_TIMEOUT,
        heartbeat_timeout=_RENDER_HEARTBEAT if long_running else None,
        retry_policy=_ACTIVITY_RETRY,
    )
    if not isinstance(result, dict):
        raise PipelineDataError(f"{activity_name}:non-dict-result")
    return result


@workflow.defn
class GenBatchWorkflow:
    """批量预生成：母版备制 → 多候选 fan-out → 机检门禁 → 达标发布 / 超限失败。"""

    @workflow.run
    async def run(self, spec: GenBatchSpec) -> MachineGateResult:
        if spec.candidate_count < 1:
            return MachineGateResult(verdict="failed", failed_checks=["invalid-candidate-count"])
        if not spec.template_ids:
            return MachineGateResult(verdict="failed", failed_checks=["missing-template-ids"])

        # 阶段一：母版备制（spec 已带现成母版则跳过）
        try:
            master_id = spec.plan_master_artifact_id or await self._prepare_plan_master(spec)
        except (ActivityError, PipelineDataError) as err:
            return MachineGateResult(verdict="failed", failed_checks=[describe_failure(err)])

        # 阶段二：候选 fan-out + 门禁；不达标自动重生成（换模板轮换），超限终判失败
        passed: list[str] = []
        last_round_failures: list[str] = []
        rounds = 0
        rounds_allowed = 1 + max(spec.max_regen_rounds, 0)
        while len(passed) < spec.candidate_count and rounds < rounds_allowed:
            shortfall = spec.candidate_count - len(passed)
            round_failures: list[str] = []
            generation = [
                _execute(
                    ACTIVITY_ATMOSPHERE_VISUAL,
                    {
                        "plan_master_artifact_id": master_id,
                        "template_id": pick_template(
                            spec.template_ids, rounds, slot, spec.candidate_count
                        ),
                        "render_tier": spec.render_tier,
                    },
                    task_queue=spec.queues.imagegen,
                    long_running=True,
                )
                for slot in range(shortfall)
            ]
            outcomes = await asyncio.gather(*generation, return_exceptions=True)
            candidates: list[str] = []
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    round_failures.append(
                        f"{ACTIVITY_ATMOSPHERE_VISUAL}:{describe_failure(outcome)}"
                    )
                    continue
                artifact_id = outcome.get("artifact_id")
                if not isinstance(artifact_id, str):
                    round_failures.append(f"{ACTIVITY_ATMOSPHERE_VISUAL}:missing-artifact-id")
                    continue
                candidates.append(artifact_id)
            for artifact_id in candidates:
                try:
                    gate_failures = await self._run_machine_gate(artifact_id, spec.queues.genpipe)
                except (ActivityError, PipelineDataError) as err:
                    round_failures.append(f"{artifact_id}:{describe_failure(err)}")
                    continue
                if gate_failures:
                    round_failures.extend(f"{artifact_id}:{name}" for name in gate_failures)
                else:
                    passed.append(artifact_id)
            last_round_failures = round_failures
            rounds += 1

        if len(passed) >= spec.candidate_count:
            return MachineGateResult(verdict="passed", passed_artifact_ids=passed, rounds=rounds)
        return MachineGateResult(
            verdict="failed",
            failed_checks=last_round_failures,
            passed_artifact_ids=passed,
            rounds=rounds,
        )

    async def _prepare_plan_master(self, spec: GenBatchSpec) -> str:
        """solve → rule-check（本队列）→ plan-2d-render 母版（render2d 队列）。"""
        solved = await _execute(
            ACTIVITY_PLAN_LAYOUT_SOLVE, spec.floorplan_id, task_queue=spec.queues.genpipe
        )
        plan_revision_id = solved.get("plan_revision_id")
        if not isinstance(plan_revision_id, str):
            raise PipelineDataError(f"{ACTIVITY_PLAN_LAYOUT_SOLVE}:missing-plan-revision-id")
        ruled = await _execute(
            ACTIVITY_PLAN_RULE_CHECK, plan_revision_id, task_queue=spec.queues.genpipe
        )
        if ruled.get("passed") is not True:
            raise PipelineDataError(f"{ACTIVITY_PLAN_RULE_CHECK}:not-passed")
        master = await _execute(
            ACTIVITY_PLAN_2D_RENDER,
            {"revision_id": plan_revision_id, "purpose": "plan_master"},
            task_queue=spec.queues.render2d,
            long_running=True,
        )
        artifact_id = master.get("artifact_id")
        if not isinstance(artifact_id, str):
            raise PipelineDataError(f"{ACTIVITY_PLAN_2D_RENDER}:missing-artifact-id")
        return artifact_id

    async def _run_machine_gate(self, artifact_id: str, gate_queue: str) -> list[str]:
        consistency = await _execute(ACTIVITY_CONSISTENCY_CHECK, artifact_id, task_queue=gate_queue)
        compliance = await _execute(ACTIVITY_COMPLIANCE_CHECK, artifact_id, task_queue=gate_queue)
        return evaluate_gate(consistency, compliance)


@workflow.defn
class GenerationTaskWorkflow:
    """交互侧生成任务的任务层入口：task_type 路由 activity 链 + 机检门禁收尾。

    project-svc 创建任务后启动本 workflow 即返回；完成/失败经事件回流，任务超时
    由 Temporal timeout 兜底（对齐文档 §2.3，无手工 deadline 扫描）。
    """

    @workflow.run
    async def run(self, spec: GenerationTaskSpec) -> GenerationTaskResult:
        try:
            chain = build_task_chain(spec)
        except PipelineDataError as err:
            return GenerationTaskResult(
                task_id=spec.task_id, verdict="failed", failed_checks=[str(err)]
            )

        artifact_ids: list[str] = []
        upstream: dict[str, Any] = {}
        for step in chain:
            try:
                arg = resolve_step_arg(step, upstream)
                result = await _execute(
                    step.activity, arg, task_queue=step.task_queue, long_running=step.long_running
                )
            except (ActivityError, PipelineDataError) as err:
                return GenerationTaskResult(
                    task_id=spec.task_id,
                    verdict="failed",
                    artifact_ids=artifact_ids,
                    failed_checks=[f"{step.activity}:{describe_failure(err)}"],
                )
            upstream = result
            artifact_id = result.get("artifact_id")
            if isinstance(artifact_id, str):
                artifact_ids.append(artifact_id)

        if not artifact_ids:
            return GenerationTaskResult(
                task_id=spec.task_id, verdict="failed", failed_checks=["missing-artifact-id"]
            )

        # 门禁收尾（本队列）：交互产物同样过机检，两条路径都强制
        final_artifact_id = artifact_ids[-1]
        try:
            consistency = await _execute(
                ACTIVITY_CONSISTENCY_CHECK, final_artifact_id, task_queue=spec.queues.genpipe
            )
            compliance = await _execute(
                ACTIVITY_COMPLIANCE_CHECK, final_artifact_id, task_queue=spec.queues.genpipe
            )
        except (ActivityError, PipelineDataError) as err:
            return GenerationTaskResult(
                task_id=spec.task_id,
                verdict="failed",
                artifact_ids=artifact_ids,
                failed_checks=[describe_failure(err)],
            )
        gate_failures = evaluate_gate(consistency, compliance)
        if gate_failures:
            return GenerationTaskResult(
                task_id=spec.task_id,
                verdict="regenerate",
                artifact_ids=artifact_ids,
                failed_checks=gate_failures,
            )
        return GenerationTaskResult(
            task_id=spec.task_id, verdict="passed", artifact_ids=artifact_ids
        )
