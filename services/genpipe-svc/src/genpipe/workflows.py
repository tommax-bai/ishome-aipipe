"""Temporal workflow：只编排，禁止任何 IO（可重放硬约束，import-linter 契约锁定）。

机检门禁状态机（全自动，运行时无任何队列）：
    多候选生成 → 打分（scorer 阈值，打分为 packages/scoring 库能力、暂无独立 activity）
    → consistency-check → compliance-check → 达标 → 发布；
    不达标 → 自动重生成（带轮数上限，超限返回 failed verdict，绝不静默假成功）

报告成文线（图 v0.2 §2 第二条流水线，ReportComposeWorkflow）：求值线在派发前把数字全部
算完并封进报告数据包，本文件只做"派发—收结论"的编排——数据包是不透明载荷，原样透传。

activity 一律以 contracts 注册表中的注册名字符串 + task_queue 派发（跨服务不 import
存根签名），唯一真源：ishome-contracts `activities/registry.md` 与
`registries/task_queues.md`（只增不改）。V1.5 裁决：Temporal 收缩至任务层——
本文件只承载生成管线任务 workflow，无长周期项目 workflow。
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    import pydantic_core  # noqa: F401  # pydantic 转换器运行时依赖：预载省去沙箱逐次告警

    from genpipe.models import (
        FloorplanVisualsResult,
        FloorplanVisualsSpec,
        GenBatchSpec,
        GenerationTaskResult,
        GenerationTaskSpec,
        MachineGateResult,
        ReportComposeResult,
        ReportComposeSpec,
        ReportStage,
        TaskProduct,
        TaskStep,
        UnitFanoutOutcome,
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
ACTIVITY_REPORT_UNIT_COMPOSE = "report-unit-compose"
ACTIVITY_REPORT_PAGE_ASSEMBLE = "report-page-assemble"
ACTIVITY_REPORT_BOOK_CHECK = "report-book-check"
ACTIVITY_REPORT_BOOK_RENDER = "report-book-render"
ACTIVITY_FLOORPLAN_PARSE = "floorplan-parse"
ACTIVITY_FLOORPLAN_GEOMETRY_EXTRACT = "floorplan-geometry-extract"
ACTIVITY_PLAN_NOTES_WRITE = "plan-notes-write"
ACTIVITY_PLAN_COPY_WRITE = "plan-copy-write"
ACTIVITY_STYLE_CAPTION_OVERLAY = "style-caption-overlay"
ACTIVITY_TASK_RESULT_DELIVER = "task-result-deliver"

WORKFLOW_TASK_QUEUE = "genpipe-workflows"
"""workflow 专属队列：起点（service.py）与执行者（genpipe workflow worker）同属本服务，
非跨服务契约，不进 contracts `registries/task_queues.md`（该表只登记 activity 队列归属）。"""

_COMPUTE_TIMEOUT = timedelta(minutes=5)
"""解析/求解/校验/门禁类 activity 的 start_to_close 上限。"""
_RENDER_TIMEOUT = timedelta(minutes=15)
"""绘图/渲染类长跑 activity 的 start_to_close 上限。"""
_RENDER_HEARTBEAT = timedelta(seconds=60)
"""长跑绘图 activity 必须心跳：worker 失联在一个心跳窗口内被发现并重派。"""
_COMPOSE_TIMEOUT = timedelta(minutes=10)
"""单元成文 activity 的 start_to_close 上限：一次派发内含 1+max_rewrites 轮 LLM 推理。
reportgen 侧不打心跳，故**不设** heartbeat_timeout——设了等于按心跳窗口误杀正常推理。"""
_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)
"""activity 级重试上限：耗尽后 ActivityError 上抛，由 workflow 记入 failed verdict。"""
_IMAGEGEN_TIMEOUT = timedelta(minutes=10)
"""图生图一张的 start_to_close 上限（实测 18–60 s，含网关排队）。imagegen / render2d 都
**不打心跳**，故不置 long_running——设了心跳窗口等于按 60 s 误杀正常出图。"""
_DELIVER_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=1),
    maximum_attempts=10,
)
"""结果回流的重试：业务侧可能正在重启，多等几轮比丢一次结论便宜——结论丢了业务侧就永远不知道。"""


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


def partition_unit_outcomes(
    domains: list[str], outcomes: Sequence[dict[str, Any] | BaseException]
) -> UnitFanoutOutcome:
    """各 dom- 单元并行成文的结果归并（纯函数，可直测）。

    三类都算失败，绝不静默假成功：派发异常、verdict 非 ok、verdict=ok 却零卡片（空内容顶替）。
    成功单元原样留给装配，失败单元原样回传（自带 domain 与 violations）。
    """
    fanout = UnitFanoutOutcome()
    for domain, outcome in zip(domains, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            fanout.failed_domains.append(domain)
            fanout.dispatch_failures.append(
                f"{ACTIVITY_REPORT_UNIT_COMPOSE}:{domain}:{describe_failure(outcome)}"
            )
            continue
        rewrites_used = outcome.get("rewrites_used")
        if isinstance(rewrites_used, int):
            fanout.rewrite_rounds_by_domain[domain] = rewrites_used
        if outcome.get("verdict") != "ok":
            fanout.failed_domains.append(domain)
            fanout.failed_units.append(outcome)
            continue
        if not outcome.get("cards"):
            fanout.failed_domains.append(domain)
            fanout.dispatch_failures.append(f"{ACTIVITY_REPORT_UNIT_COMPOSE}:{domain}:no-cards")
            continue
        fanout.composed_units.append(outcome)
    return fanout


def merge_retried_units(base: UnitFanoutOutcome, retried: UnitFanoutOutcome) -> UnitFanoutOutcome:
    """把一轮"只重开失败域"的归并结果并回上一轮（纯函数，可直测）。

    - **成功单元累加**：先前成的章不重开，原样留着进装配（重开的是章不是册）；
    - **失败三件套整体换成本轮的**（失败域 / 失败单元 / 派发失败码）：它们回答的是"最后一次
      尝试为什么没成"，跨轮累加会读出一个根本没发生过的失败面——某域第一轮失败第二轮成了，
      它就不该再出现在 failed_domains 里；
    - **重写轮数按域覆盖**：同一域重开后，观测量取最后一次尝试的轮数（章内重写是 activity 内部
      的旋钮，混算跨轮总和会把它和重开次数搅成一个数）。
    """
    return UnitFanoutOutcome(
        composed_units=[*base.composed_units, *retried.composed_units],
        failed_domains=list(retried.failed_domains),
        failed_units=list(retried.failed_units),
        dispatch_failures=list(retried.dispatch_failures),
        rewrite_rounds_by_domain={
            **base.rewrite_rounds_by_domain,
            **retried.rewrite_rounds_by_domain,
        },
    )


def collect_violations(result: dict[str, Any]) -> list[dict[str, Any]]:
    """取 activity 的违规清单（纯函数）：原样透传不改写——判据编号归 release 数据，编排不解释。"""
    raw = result.get("violations")
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def unexplained_failure_checks(activity_name: str, violations: list[dict[str, Any]]) -> list[str]:
    """failed 却不给违规清单 = 失败无理由（纯函数）：补一条编排层失败码，杜绝失败被吞掉。"""
    return [] if violations else [f"{activity_name}:failed-without-violations"]


async def _execute(
    activity_name: str,
    arg: Any,
    *,
    task_queue: str,
    long_running: bool = False,
    start_to_close: timedelta | None = None,
    retry_policy: RetryPolicy | None = None,
) -> dict[str, Any]:
    """按注册名字符串派发 activity；显式 task_queue / 超时 / 重试，长跑加心跳窗口。

    `start_to_close` 覆写默认两档超时（长跑绘图 / 计算校验）；不打心跳的长跑 activity
    只覆写超时、不置 long_running，否则心跳窗口会误杀正常执行。
    `retry_policy` 覆写默认重试（回调那一步要多等几轮）。
    """
    result = await workflow.execute_activity(
        activity_name,
        arg,
        task_queue=task_queue,
        start_to_close_timeout=start_to_close
        or (_RENDER_TIMEOUT if long_running else _COMPUTE_TIMEOUT),
        heartbeat_timeout=_RENDER_HEARTBEAT if long_running else None,
        retry_policy=retry_policy or _ACTIVITY_RETRY,
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


@workflow.defn
class ReportComposeWorkflow:
    """报告成文线编排（图 v0.2 §2 第二条流水线）：dom- 单元并行成文（失败章单独重开）→
    页面装配 → 册级校验。

    图 v0.2 §0 的硬性条件在本类逐条兑现：

    - **只编排不计算**：数字在求值线（project-svc 规则引擎）算完，报告数据包对本编排是
      不透明载荷，原样透传给 activity——本仓不建模包内任何字段；
    - **一次生成 = 一次短 run**：不长驻、不做图内 checkpoint 恢复、不建第二台状态机；
      "改一条决策局部重跑" = 里程碑引擎判定下次派哪些单元（规则 8.1），不在图内恢复；
    - **单元间零通信**：各域同刻 fan-out，依赖只存在于求值线。

    三个 activity 的实现与队列归属见 ishome-reportgen（`reportgen-activities`）；本编排
    只按 contracts 注册名派发，不 import 对侧存根签名。
    """

    @workflow.run
    async def run(self, spec: ReportComposeSpec) -> ReportComposeResult:
        # 入参违约在派发前拦：域集为空/重复（重复域 → 装配出重复页 → 册检必挂）、数据包为空
        # （求值线什么都没算出来）——都属于派了也注定失败，早拦省掉整轮 LLM 推理
        for failure_code, violated in (
            ("missing-domains", not spec.domains),
            ("duplicate-domains", len(set(spec.domains)) != len(spec.domains)),
            ("missing-package", not spec.package),
        ):
            if violated:
                return ReportComposeResult(
                    report_id=spec.report_id,
                    verdict="failed",
                    failed_stage="unit-compose",
                    failed_checks=[failure_code],
                )

        fanout = await self._compose_units(spec, spec.domains)

        # 失败的那一章单独重开（用户裁决 2026-08-31：「失败让重开，我觉得没问题，单独开失败的
        # 这一张就可以了」）。为什么重开有用——**章失败是随机的不是必然的**：同一份输入连跑五次，
        # dom-budget 在第 1/3/4 跑失败、第 2/5 跑通过，dom-lighting 只在第 5 跑失败。单章成功率
        # 约 0.75，六章连乘即整册约 0.18（实测四跑成册 1/4，对得上）；只重开失败章两次，整册可
        # 拉到约 0.91。整册重来则要把已成的五章连同它们的 LLM 推理一起扔掉，还得再赌一次全过。
        # 注意这与 `max_rewrites` 是两个旋钮：那个是 activity **章内**的自我修正轮数（≤2 轮，
        # 图 v0.2 §3 的设计定数，不许私改）；这里是章内轮数耗尽、整章终判 failed 之后，编排侧
        # 拿同样入参**另起一次派发**。也不能靠 `_ACTIVITY_RETRY` 顶替：那条只在 activity 抛异常
        # 时生效，而一章过不了检是正常返回 verdict=failed，Temporal 不会重试它。
        unit_retries_by_domain: dict[str, int] = {}
        retries_left = max(spec.max_unit_retries, 0)
        while fanout.failed_domains and retries_left > 0:
            retries_left -= 1
            retried_domains = list(fanout.failed_domains)
            for domain in retried_domains:
                unit_retries_by_domain[domain] = unit_retries_by_domain.get(domain, 0) + 1
            fanout = merge_retried_units(fanout, await self._compose_units(spec, retried_domains))

        def failed(
            stage: ReportStage,
            *,
            violations: list[dict[str, Any]] | None = None,
            failed_checks: list[str] | None = None,
        ) -> ReportComposeResult:
            """失败结论一律带上失败单元与违规清单：不吞、不空顶（图 v0.2 §3 出口纪律）。"""
            return ReportComposeResult(
                report_id=spec.report_id,
                verdict="failed",
                failed_stage=stage,
                failed_domains=fanout.failed_domains,
                failed_units=fanout.failed_units,
                violations=violations or [],
                failed_checks=failed_checks or [],
                rewrite_rounds_by_domain=fanout.rewrite_rounds_by_domain,
                unit_retries_by_domain=unit_retries_by_domain,
            )

        if fanout.failed_domains:
            # 走到这里 = 重开次数也耗尽了（见上方重开理由），以下策略才生效。
            # 失败策略（本编排唯一的策略取舍，理由写在此处以免后人两头下注）：
            # **任一域失败 → 整册失败，不装配、不出"其余页"**。三条理由：
            # ① 下游 report-page-assemble 自身即以 gate-unit-failed 拒收残缺单元集，派过去
            #    只是把同一裁决绕一圈；
            # ② 册级判据是 set-closure / promise-fulfilled / concept-through（图 v0.2 §4），
            #    判的是整册闭合——缺域的册在册检必然不合格，"出其余页"产的是注定不合格的册；
            # ③ 报告是一次性交付物不是信息流：半本报告对用户是错的成品，不是少一节的成品。
            # 注：并行 fan-out 不做提前取消——一次收齐全部域的违规，是自迭代回路（规则 4.17）
            # 的输入信号，比早退省下的那点推理更值钱。
            return failed("unit-compose", failed_checks=fanout.dispatch_failures)

        try:
            assembled = await _execute(
                ACTIVITY_REPORT_PAGE_ASSEMBLE,
                {"units": fanout.composed_units},
                task_queue=spec.queues.reportgen,
            )
        except (ActivityError, PipelineDataError) as err:
            return failed(
                "page-assemble",
                failed_checks=[f"{ACTIVITY_REPORT_PAGE_ASSEMBLE}:{describe_failure(err)}"],
            )
        if assembled.get("verdict") != "ok":
            page_violations = collect_violations(assembled)
            return failed(
                "page-assemble",
                violations=page_violations,
                failed_checks=unexplained_failure_checks(
                    ACTIVITY_REPORT_PAGE_ASSEMBLE, page_violations
                ),
            )
        pages = assembled.get("pages")
        if not isinstance(pages, list) or not pages:
            # verdict=ok 却没有页：空内容顶替，按失败处理
            return failed(
                "page-assemble", failed_checks=[f"{ACTIVITY_REPORT_PAGE_ASSEMBLE}:missing-pages"]
            )

        try:
            checked = await _execute(
                ACTIVITY_REPORT_BOOK_CHECK,
                {"pages": pages, "package": spec.package},
                task_queue=spec.queues.reportgen,
            )
        except (ActivityError, PipelineDataError) as err:
            return failed(
                "book-check",
                failed_checks=[f"{ACTIVITY_REPORT_BOOK_CHECK}:{describe_failure(err)}"],
            )
        if checked.get("verdict") != "ok":
            book_violations = collect_violations(checked)
            return failed(
                "book-check",
                violations=book_violations,
                failed_checks=unexplained_failure_checks(
                    ACTIVITY_REPORT_BOOK_CHECK, book_violations
                ),
            )

        # 册检过了才出册：渲染层是 gen-locked 的执行者，不该把不合格的册排得漂漂亮亮存起来。
        # 它派往另一条队列（`reportrender-activities`）而不是成文线那条——伸缩轴不同
        # （渲染是 CPU + IO，成文是 LLM 推理），物理隔离的同一条判据。
        try:
            rendered = await _execute(
                ACTIVITY_REPORT_BOOK_RENDER,
                {"report_id": spec.report_id, "pages": pages, "package": spec.package},
                task_queue=spec.queues.reportrender,
            )
        except (ActivityError, PipelineDataError) as err:
            return failed(
                "book-render",
                failed_checks=[f"{ACTIVITY_REPORT_BOOK_RENDER}:{describe_failure(err)}"],
            )
        if rendered.get("verdict") != "ok":
            render_violations = collect_violations(rendered)
            return failed(
                "book-render",
                violations=render_violations,
                failed_checks=unexplained_failure_checks(
                    ACTIVITY_REPORT_BOOK_RENDER, render_violations
                ),
            )
        book_key = rendered.get("book_key")
        if not isinstance(book_key, str) or not book_key:
            # verdict=ok 却没说册落在哪：册没落地却回报成功是本条线最贵的失效形态
            # （调用方会去签一个指向空气的链接，业主点开是 404）——按失败处理。
            return failed(
                "book-render", failed_checks=[f"{ACTIVITY_REPORT_BOOK_RENDER}:missing-book-key"]
            )

        return ReportComposeResult(
            report_id=spec.report_id,
            verdict="ok",
            pages=pages,
            book_key=book_key,
            rewrite_rounds_by_domain=fanout.rewrite_rounds_by_domain,
            unit_retries_by_domain=unit_retries_by_domain,
        )

    async def _compose_units(
        self, spec: ReportComposeSpec, domains: list[str]
    ) -> UnitFanoutOutcome:
        """把给定域并行派往成文 activity 并归并结论（首轮派全集，重开轮只派失败集）。

        入参逐次相同（`package` / `max_rewrites` 不随重开变化）：重开赌的是同一次推理的随机性，
        改入参就成了"换个题面再试"，那是另一件事，不在这条裁决里。
        """
        outcomes = await asyncio.gather(
            *[
                _execute(
                    ACTIVITY_REPORT_UNIT_COMPOSE,
                    {
                        "domain": domain,
                        "package": spec.package,
                        "max_rewrites": spec.max_rewrites,
                    },
                    task_queue=spec.queues.reportgen,
                    start_to_close=_COMPOSE_TIMEOUT,
                )
                for domain in domains
            ],
            return_exceptions=True,
        )
        return partition_unit_outcomes(domains, outcomes)


class StepFailure(Exception):
    """三张图主链某一步没过（activity 抛错、verdict≠ok、缺关键出参）：带上是哪一步、为什么。"""

    def __init__(self, activity_name: str, detail: str) -> None:
        super().__init__(f"{activity_name}:{detail}")
        self.activity_name = activity_name
        self.detail = detail

    @property
    def check(self) -> str:
        return f"{self.activity_name}:{self.detail}"


def summarize_violations(result: dict[str, Any]) -> str:
    """把 activity 的 violations 压成一行失败码（纯函数）：`check=detail; check=detail`；
    没给即 failed-without-violations。"""
    violations = collect_violations(result)
    if not violations:
        return "failed-without-violations"
    return "; ".join(f"{v.get('check', '?')}={v.get('detail', '')}" for v in violations)


CAPTION_LAYOUT_CHECK = "style-caption-failed"
"""render2d 叠字那一步"版面放不下"的失败码（该仓 activities 的 violations.check）。
只有这一种失败值得重生成底图；桶取不到、入参不对等其余失败重生成也不会变好。"""


def is_caption_layout_failure(err: StepFailure) -> bool:
    """叠字失败是不是"版面放不下"（纯函数）。"""
    return (
        err.activity_name == ACTIVITY_STYLE_CAPTION_OVERLAY and CAPTION_LAYOUT_CHECK in err.detail
    )


def annotations_from_notes(notes: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """批注 → 写上风格图的注释（纯函数）：只留**不含数字**的句子（imagegen 拒收带数字的注释——
    数字上图走叠印那条线，不让模型画数字），只带 room 与 text（cites 是说明图那边的事）。"""
    picked: list[dict[str, str]] = []
    for note in notes:
        room = note.get("room")
        text = note.get("text")
        if not isinstance(room, str) or not isinstance(text, str) or not text.strip():
            continue
        if any(ch.isdigit() for ch in text):
            continue
        picked.append({"room": room, "text": text})
    return picked


def captioned_key_of(style_object_key: str) -> str:
    """情绪图叠字后的对象键（纯函数）：
    `…/atmosphere-{template}.{ext}` → `…/atmosphere-{template}-captioned.png`。

    与 render2d `style-caption-overlay` 写键的规则**逐字同构**（那边是真源，本处只用来在
    结论里预告键、便于调用方核对）；ext 固定 png——叠字后的图由确定性绘制层重新编码。
    """
    stem = style_object_key.rsplit(".", 1)[0]
    return f"{stem}-captioned.png"


def build_task_result(
    spec: FloorplanVisualsSpec,
    *,
    verdict: str,
    products: Sequence[TaskProduct],
    failed_checks: Sequence[str],
    failure: dict[str, Any] | None,
    workflow_id: str,
    run_id: str,
) -> dict[str, Any]:
    """回调报文（纯函数）：contracts `openapi/project.v1.yaml` `generation_task_result`。

    只装契约里有的字段——业务侧按契约收货，多一个字段就是两侧口径漂了一格。
    failed_checks 并进 failure.detail：失败原因要让业务侧一眼看到，不靠回来翻 Temporal 历史。
    """
    payload: dict[str, Any] = {
        "task_id": spec.task_id,
        "status": "completed" if verdict == "ok" else "failed",
        "products": [product.model_dump() for product in products],
        "workflow_id": workflow_id,
        "run_id": run_id,
    }
    if verdict != "ok":
        code = (failure or {}).get("code") or "floorplan-visuals-failed"
        detail = (failure or {}).get("detail") or ""
        extra = [check for check in failed_checks if check and check != f"{code}:{detail}"]
        if extra:
            detail = f"{detail}；其余：{' | '.join(extra)}" if detail else " | ".join(extra)
        payload["failure"] = {"code": code, "detail": detail}
    return payload


@workflow.defn
class FloorplanVisualsWorkflow:
    """三张免费图生成线（2026-09-04 接线）：户型图对象键 → 三张图 → 结论回流业务侧。

    主链（任一步失败即整任务失败，已出的半成品只留血缘）：
        floorplan-geometry-extract（勘测 + 几何 + 事实）
        → plan-notes-write ∥ plan-copy-write
        → plan-2d-render（母版 + 功能说明图，render2d）
        → atmosphere-visual ×2（情绪图底图 ∥ 手账写字版 + 注释，imagegen）
        → style-caption-overlay（情绪图叠字，render2d）
    旁路：floorplan-parse（户型特征解析，给报告那条腿用）与主链并行；它失败不影响三张图，
    只在 failed_checks 里记一条、结论里少一件 floorplan_reading。
    收尾：task-result-deliver 把结论 `POST` 到派发时注入的回调地址——**没送到不算完**。

    与 GenerationTaskWorkflow 的关系：那条是骨架期的通用路由链，派的字段与 render2d / imagegen
    两个真 activity 已对不上（那两仓的收货模型早改了形态）；本 workflow 按两仓现行的收货单派，
    不复用那条链。门禁（consistency / compliance）两个 activity 仍是存根，本线暂不派——
    时点写死＝那两个 activity 实装时。
    """

    @workflow.run
    async def run(self, spec: FloorplanVisualsSpec) -> FloorplanVisualsResult:
        products: list[TaskProduct] = []
        failed_checks: list[str] = []
        failure: dict[str, Any] | None = None

        # 旁路先起：特征解析要 11 次调用、约 40 s，与主链并行不占三张图的时间
        parse_task = asyncio.create_task(self._parse_floorplan(spec))
        try:
            await self._render_visuals(spec, products)
        except StepFailure as err:
            failed_checks.append(err.check)
            failure = {"code": err.activity_name, "detail": err.detail}
        reading_product, reading_failed = await parse_task
        if reading_product is not None:
            products.append(reading_product)
        failed_checks.extend(reading_failed)

        verdict = "ok" if failure is None else "failed"
        info = workflow.info()
        result_payload = build_task_result(
            spec,
            verdict=verdict,
            products=products,
            failed_checks=failed_checks,
            failure=failure,
            workflow_id=info.workflow_id,
            run_id=info.run_id,
        )
        delivered = await self._deliver(spec, result_payload, failed_checks)
        return FloorplanVisualsResult(
            task_id=spec.task_id,
            verdict="ok" if verdict == "ok" else "failed",
            products=products,
            failed_checks=failed_checks,
            failure=failure,
            delivered=delivered,
        )

    async def _render_visuals(
        self, spec: FloorplanVisualsSpec, products: list[TaskProduct]
    ) -> None:
        """主链。每一步的产物一出来就登记进 products——失败时业务侧也拿得到已出的半成品血缘。"""
        queues = spec.queues
        key = spec.floorplan_object_key

        geometry = await _run_step(
            ACTIVITY_FLOORPLAN_GEOMETRY_EXTRACT,
            {"floorplan_object_key": key},
            task_queue=queues.genpipe,
        )
        products.append(
            TaskProduct(
                product="floorplan_geometry",
                object_key=_require_str(
                    geometry, "geometry_key", ACTIVITY_FLOORPLAN_GEOMETRY_EXTRACT
                ),
                content_type="application/json",
            )
        )
        facts = geometry.get("facts") or []
        room_names = geometry.get("room_names") or []

        notes_result, copy_result = await _gather_steps(
            _run_step(
                ACTIVITY_PLAN_NOTES_WRITE,
                {"facts": facts, "room_names": room_names},
                task_queue=queues.genpipe,
            ),
            _run_step(ACTIVITY_PLAN_COPY_WRITE, {"facts": facts}, task_queue=queues.genpipe),
        )
        notes = notes_result.get("notes") or []
        copy = copy_result.get("copy") or {}

        render = await _run_step(
            ACTIVITY_PLAN_2D_RENDER,
            {"floorplan_object_key": key, "geometry": geometry.get("geometry"), "notes": notes},
            task_queue=queues.render2d,
            start_to_close=_RENDER_TIMEOUT,
        )
        master_key = _require_str(render, "master_key", ACTIVITY_PLAN_2D_RENDER)
        room_anchors_key = _require_str(render, "room_anchors_key", ACTIVITY_PLAN_2D_RENDER)
        products.append(
            TaskProduct(product="plan_master", object_key=master_key, content_type="image/png")
        )
        products.append(
            TaskProduct(
                product="brief_image",
                object_key=_require_str(render, "brief_key", ACTIVITY_PLAN_2D_RENDER),
                content_type="image/png",
                gen_params={"note_count": len(notes)},
            )
        )

        # 注释只在物件清单也能给的时候递（见 FloorplanVisualTemplates.annotate_style）；
        # 不递就一个字段都不带——imagegen 那边"默认空＝不写注释"，旧派发行为逐字节同前
        annotations = annotations_from_notes(notes) if spec.templates.annotate_style else []
        style_request: dict[str, Any] = {
            "master_object_key": master_key,
            "room_anchors_object_key": room_anchors_key,
            "template_id": spec.templates.style,
        }
        if annotations:
            style_request["annotations"] = annotations
        mood_result, style_result = await _gather_steps(
            _run_step(
                ACTIVITY_ATMOSPHERE_VISUAL,
                {
                    "master_object_key": master_key,
                    "room_anchors_object_key": room_anchors_key,
                    "template_id": spec.templates.mood,
                },
                task_queue=queues.imagegen,
                start_to_close=_IMAGEGEN_TIMEOUT,
            ),
            _run_step(
                ACTIVITY_ATMOSPHERE_VISUAL,
                style_request,
                task_queue=queues.imagegen,
                start_to_close=_IMAGEGEN_TIMEOUT,
            ),
        )
        products.append(
            TaskProduct(
                product="style_image",
                object_key=_require_str(
                    style_result, "image_object_key", ACTIVITY_ATMOSPHERE_VISUAL
                ),
                content_type=_optional_str(style_result, "content_type"),
                gen_params={
                    "template_id": spec.templates.style,
                    "annotation_count": len(annotations),
                },
            )
        )

        # 叠字放不下＝重生成一张底图再叠（不把字压在画面上），上限 max_caption_retries
        mood_key = _require_str(mood_result, "image_object_key", ACTIVITY_ATMOSPHERE_VISUAL)
        caption_attempts = 0
        while True:
            try:
                captioned = await _run_step(
                    ACTIVITY_STYLE_CAPTION_OVERLAY,
                    {"style_object_key": mood_key, "copy": copy},
                    task_queue=queues.render2d,
                    start_to_close=_RENDER_TIMEOUT,
                )
                break
            except StepFailure as err:
                if not is_caption_layout_failure(err) or caption_attempts >= max(
                    spec.max_caption_retries, 0
                ):
                    raise
                caption_attempts += 1
                mood_result = await _run_step(
                    ACTIVITY_ATMOSPHERE_VISUAL,
                    {
                        "master_object_key": master_key,
                        "room_anchors_object_key": room_anchors_key,
                        "template_id": spec.templates.mood,
                    },
                    task_queue=queues.imagegen,
                    start_to_close=_IMAGEGEN_TIMEOUT,
                )
                mood_key = _require_str(mood_result, "image_object_key", ACTIVITY_ATMOSPHERE_VISUAL)
        products.append(
            TaskProduct(
                product="mood_image",
                object_key=_require_str(
                    captioned, "image_object_key", ACTIVITY_STYLE_CAPTION_OVERLAY
                ),
                content_type="image/png",
                gen_params={
                    "template_id": spec.templates.mood,
                    "base_object_key": mood_key,
                    "caption_regenerations": caption_attempts,
                },
            )
        )

    async def _parse_floorplan(
        self, spec: FloorplanVisualsSpec
    ) -> tuple[TaskProduct | None, list[str]]:
        """旁路：特征解析。失败不致命——三张图不等它，报告那条腿届时按键发现没有再说。"""
        try:
            reading = await _run_step(
                ACTIVITY_FLOORPLAN_PARSE,
                {"floorplan_object_key": spec.floorplan_object_key},
                task_queue=spec.queues.genpipe,
            )
            reading_key = _require_str(reading, "reading_key", ACTIVITY_FLOORPLAN_PARSE)
        except StepFailure as err:
            return None, [err.check]
        return (
            TaskProduct(
                product="floorplan_reading",
                object_key=reading_key,
                content_type="application/json",
                gen_params={"layout_features": reading.get("layout_features") or {}},
            ),
            [],
        )

    async def _deliver(
        self, spec: FloorplanVisualsSpec, result_payload: dict[str, Any], failed_checks: list[str]
    ) -> bool:
        try:
            delivered = await _execute(
                ACTIVITY_TASK_RESULT_DELIVER,
                {"result_callback_url": spec.result_callback_url, "result": result_payload},
                task_queue=spec.queues.genpipe,
                retry_policy=_DELIVER_RETRY,
            )
        except (ActivityError, PipelineDataError) as err:
            failed_checks.append(f"{ACTIVITY_TASK_RESULT_DELIVER}:{describe_failure(err)}")
            return False
        if delivered.get("verdict") != "ok":
            failed_checks.append(
                f"{ACTIVITY_TASK_RESULT_DELIVER}:{summarize_violations(delivered)}"
            )
            return False
        return True


async def _run_step(
    activity_name: str,
    arg: Any,
    *,
    task_queue: str,
    start_to_close: timedelta | None = None,
) -> dict[str, Any]:
    """派一步主链 activity：抛错、verdict≠ok 都收成 StepFailure（带步名与原因）。"""
    try:
        result = await _execute(
            activity_name, arg, task_queue=task_queue, start_to_close=start_to_close
        )
    except (ActivityError, PipelineDataError) as err:
        raise StepFailure(activity_name, describe_failure(err)) from err
    if result.get("verdict") != "ok":
        raise StepFailure(activity_name, summarize_violations(result))
    return result


async def _gather_steps(
    first: Coroutine[Any, Any, dict[str, Any]], second: Coroutine[Any, Any, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """两步并行，**两步都收齐再判**：一步先失败也不取消另一步——另一步的产物是血缘，
    且它的失败原因是同一轮该一起看到的信号。第一个失败上抛。"""
    outcomes = await asyncio.gather(first, second, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, StepFailure):
            raise outcome
        if isinstance(outcome, BaseException):
            raise outcome
    left, right = outcomes
    assert isinstance(left, dict) and isinstance(right, dict)  # noqa: S101 - 上面已排除异常
    return left, right


def _require_str(result: dict[str, Any], field: str, activity_name: str) -> str:
    """verdict=ok 却缺关键出参（键为空）＝产物没落地却回报成功，按失败处理。"""
    value = result.get(field)
    if not isinstance(value, str) or not value:
        raise StepFailure(activity_name, f"missing-{field.replace('_', '-')}")
    return value


def _optional_str(result: dict[str, Any], field: str) -> str | None:
    value = result.get(field)
    return value if isinstance(value, str) and value else None
