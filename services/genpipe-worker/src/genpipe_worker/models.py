"""genpipe_worker activity 出入参模型（pydantic）。

跨 domain 纪律：worker 不 import 其他 domain 的内部模块，activity 入参出参
以本模块与（后续）contracts 生成 SDK 为准。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

RenderTier = Literal["preview", "final"]
"""渲染两档：失效传播默认只重算 preview，final 由用户显式请求或交付节点触发。"""


class PlanRenderRequest(BaseModel):
    """plan-2d-render 输入。

    同一管线两种用途：确认底图（画 BaseFacts 识别结果）/ 母版（画冻结后的
    PreliminaryPlan）。除 PNG 外必须输出房间遮罩与墙体图层（机器可读层）。
    """

    revision_id: str
    purpose: Literal["confirmation_base", "plan_master"]


class AtmosphereVisualRequest(BaseModel):
    """atmosphere-visual 输入：模板库驱动，几何输入=母版固定遮罩。"""

    plan_master_artifact_id: str
    template_id: str
    render_tier: RenderTier = "preview"


class BaseRenderRequest(BaseModel):
    """base-render 输入：三维底渲（几何/深度/线稿/遮罩输出）。"""

    scene_package_key: str
    camera_id: str
    render_tier: RenderTier = "preview"


class RealismPassRequest(BaseModel):
    """realism-pass 输入：生成式写实化（工厂效果图同用）。"""

    base_render_artifact_id: str
    style_ref: str
    render_tier: RenderTier = "preview"
