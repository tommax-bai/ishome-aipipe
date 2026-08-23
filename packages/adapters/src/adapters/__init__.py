"""各模型的提示词与出入参变换层。

铁律（规范 §5.2 轴 3，R7 一轴一层）：**LiteLLM 已是模型供应商轴的 API 层
adapter，本包禁止再造一层 API 适配。** 业务代码只引用任务级逻辑模型名
（如 `base-render.default`、`atmosphere-visual.cream`）；逻辑名 → 物理
model_id 的映射在 LiteLLM 配置里——换模型是改配置，不是改代码。
本包只做：各模型的提示词组装差异、出入参格式变换。
库不感知使用者：禁止依赖 services/*（import-linter 锁定）。
"""

from adapters.base import PromptTransform

__all__ = ["PromptTransform"]
