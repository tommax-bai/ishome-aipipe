"""Patch 操作模型与机械校验。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PatchStage = Literal["preliminary", "deep"]


class PatchOp(BaseModel):
    """单个操作（Agent 方案 §11 示例的结构化形态）。"""

    op: Literal["add", "remove", "update"]
    target_id: str | None = None
    type: str | None = None
    """add 时的对象类型，如 furniturePlacement / activity。"""
    space_id: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class Patch(BaseModel):
    """一次意图对应的一组操作；写入新 Revision 时整组原子生效。"""

    intent: str
    stage: PatchStage
    operations: list[PatchOp]
    reason: str


class PatchValidationError(ValueError):
    """机械校验不通过（几何与规则校验另在 plan-rule-check activity）。"""


def validate_patch(patch: Patch) -> None:
    """机械校验：操作非空、remove/update 必有 target、add 必有 type。"""
    if not patch.operations:
        raise PatchValidationError("patch has no operations")
    for op in patch.operations:
        if op.op in ("remove", "update") and not op.target_id:
            raise PatchValidationError(f"{op.op} requires target_id")
        if op.op == "add" and not op.type:
            raise PatchValidationError("add requires type")
