"""Temporal workflow：只编排，禁止任何 IO（可重放硬约束，import-linter 契约锁定）。

机检门禁状态机（全自动，运行时无任何队列）：
    多候选生成 → 打分（scorer 阈值） → consistency-check → compliance-check
    → 达标 → 发布；不达标 → 自动重生成或换模板重试

activity 一律以 contracts 注册表中的注册名字符串引用，唯一真源：
ishome-contracts `activities/registry.md`（只增不改）。
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

from genpipe.models import GenBatchSpec, MachineGateResult

# activity 注册名常量（与 contracts 注册表逐字一致）
ACTIVITY_ATMOSPHERE_VISUAL = "atmosphere-visual"
ACTIVITY_CONSISTENCY_CHECK = "consistency-check"
ACTIVITY_COMPLIANCE_CHECK = "compliance-check"

_GATE_TIMEOUT = timedelta(minutes=5)


@workflow.defn
class GenBatchWorkflow:
    """批量预生成：多候选 → 打分 → 机检门禁 → 发布（骨架）。"""

    @workflow.run
    async def run(self, spec: GenBatchSpec) -> MachineGateResult:
        # 骨架占位：单候选走一遍机检门禁；后续在此展开
        # 多候选 fan-out、scorer 阈值判定与自动重生成/换模板循环。
        await workflow.execute_activity(
            ACTIVITY_CONSISTENCY_CHECK,
            spec.batch_id,
            start_to_close_timeout=_GATE_TIMEOUT,
        )
        await workflow.execute_activity(
            ACTIVITY_COMPLIANCE_CHECK,
            spec.batch_id,
            start_to_close_timeout=_GATE_TIMEOUT,
        )
        return MachineGateResult(verdict="passed")
