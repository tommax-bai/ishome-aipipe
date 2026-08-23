"""genpipe 领域模型（pydantic，领域名词原样不带后缀；ORM 类加 Record 后缀区分）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RenderTier = Literal["preview", "final"]
"""渲染两档：preview 供会话内迭代与 Patch 即时反馈，final 为正式出图。"""

GateVerdict = Literal["passed", "regenerate", "switch_template"]
"""机检门禁结论：达标发布 / 自动重生成 / 换模板重试。"""


class GenBatchSpec(BaseModel):
    """一次批量预生成的排产输入（estate 交付日历驱动）。"""

    batch_id: str
    floorplan_id: str
    template_ids: list[str]
    candidate_count: int = 3
    render_tier: RenderTier = "preview"


class MachineGateResult(BaseModel):
    """机检门禁（consistency-check + compliance-check + scorer 阈值）汇总结论。"""

    verdict: GateVerdict
    scorer_score: float | None = None
    failed_checks: list[str] = Field(default_factory=list)
