"""DesignProjectWorkflow：设计项目 = 一个长周期 Temporal workflow（namespace: design）。

- 可跨月存续，历史长度经 continue-as-new 控制；
- 用户确认、进入深度设计 = signal（V1.1"任意节点可暂停恢复、中断后按最后
  事件继续"即 Temporal 原生语义，不自研状态机）；
- 只编排禁 IO（import-linter 契约锁定）；重计算按 contracts 注册名字符串
  下发 genpipe activities；
- 阶段门为 workflow 内状态：结构门=硬证据机检（户型库结构数据优先，其次
  用户上传图纸），无证据自动降级为不动结构方案。
"""

from __future__ import annotations

from temporalio import workflow

from design.models import PatchOp, ProjectPhase

# activity 注册名常量（唯一真源：ishome-contracts activities/registry.md，只增不改）
ACTIVITY_PLAN_2D_RENDER = "plan-2d-render"
ACTIVITY_ATMOSPHERE_VISUAL = "atmosphere-visual"


@workflow.defn
class DesignProjectWorkflow:
    """确认闭环 → 初步方案 → 母版+交付图集 → （用户显式进入）深度设计。"""

    def __init__(self) -> None:
        self._phase: ProjectPhase = "preliminary"
        self._facts_confirmed: bool = False
        self._pending_patches: list[list[PatchOp]] = []

    @workflow.run
    async def run(self, project_id: str) -> None:
        # 确认闭环：提取 → 确认清单 → 看图点错 → user_confirmed 后才进入设计
        await workflow.wait_condition(lambda: self._facts_confirmed)
        # 骨架占位：建立 PreliminaryPlan → 基础合理性校验 → 冻结 Revision
        # → 母版（plan-2d-render）→ 模板组合交付图集（atmosphere-visual，
        #    每张独立回读母版）→ 展示并等待用户事件。
        # 历史增长到阈值时：workflow.continue_as_new(project_id)
        _ = project_id

    @workflow.signal
    def confirm_facts(self) -> None:
        """确认闭环通过：关键字段升级 user_confirmed（设计只引用确认后的关键信息）。"""
        self._facts_confirmed = True

    @workflow.signal
    def enter_deep_design(self) -> None:
        """用户明确选择进入深度设计——不得由浏览、停留时长或一般反馈自动推断。"""
        self._phase = "deep"

    @workflow.signal
    def submit_patch(self, ops: list[PatchOp]) -> None:
        """结构化 Patch 入队：校验→新 revision→受影响产物重算（默认只重算 preview 档）。"""
        self._pending_patches.append(ops)
