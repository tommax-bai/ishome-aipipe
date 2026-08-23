"""结构化 Patch：操作模型 + 机械校验。

执行链（Agent 方案 §11）：Agent 生成 Patch → 阶段与修改权限检查 → 几何与
规则校验（经 genpipe plan-rule-check activity，不在本包）→ 写新 Revision
→ outbox 发事件 → 受影响产物重算。本包只承担操作模型与机械校验（op 枚举、
必填字段、阶段权限矩阵）。
PatchOp 的跨语言契约后续以 ishome-contracts `ishome.design.v1` 为唯一真源，
本模型与其保持字段一致。
库不感知使用者：禁止依赖 services/*（import-linter 锁定）。
"""

from patch_engine.ops import Patch, PatchOp, PatchValidationError, validate_patch

__all__ = ["Patch", "PatchOp", "PatchValidationError", "validate_patch"]
