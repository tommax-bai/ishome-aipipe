"""Temporal activities：所有 IO 与重计算收口在此。

注册名唯一真源：ishome-contracts `activities/registry.md`，**只增不改**——改注册名
会破坏历史 workflow 重放，等同于改线上协议；新增走 contracts 仓 PR 评审。
命名规则（规范 §2.4）：注册名 = kebab-case 显式声明；函数名 = 同词 snake_case
动词前置。
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from temporalio import activity

from genpipe_worker.models import (
    AtmosphereVisualRequest,
    BaseRenderRequest,
    PlanRenderRequest,
    RealismPassRequest,
)

ActivityResult = dict[str, Any]


@activity.defn(name="floorplan-parse")
async def parse_floorplan(floorplan_asset_key: str) -> ActivityResult:
    """户型图解析（备用路径）：工厂建库 / 交互上传同一实现。"""
    raise NotImplementedError


@activity.defn(name="plan-layout-solve")
async def solve_plan_layout(plan_revision_id: str) -> ActivityResult:
    """自动布局与尺寸计算（确定性求解）。"""
    raise NotImplementedError


@activity.defn(name="plan-rule-check")
async def check_plan_rules(plan_revision_id: str) -> ActivityResult:
    """空间规则校验（碰撞/通道/边界闭合）。"""
    raise NotImplementedError


@activity.defn(name="plan-2d-render")
async def render_plan_2d(request: PlanRenderRequest) -> ActivityResult:
    """母版与确定性图层绘制：确认底图/母版同一管线；输出 PNG + 房间遮罩 + 墙体图层。"""
    raise NotImplementedError


@activity.defn(name="atmosphere-visual")
async def generate_atmosphere_visual(request: AtmosphereVisualRequest) -> ActivityResult:
    """风格化交付图生成（模板库驱动，固定遮罩；每张独立回读母版）。"""
    raise NotImplementedError


@activity.defn(name="scene-compile")
async def compile_scene(deep_revision_id: str) -> ActivityResult:
    """DeepDesign → Scene Graph → 场景包编译（交互引擎专用）。"""
    raise NotImplementedError


@activity.defn(name="base-render")
async def render_base(request: BaseRenderRequest) -> ActivityResult:
    """三维底渲：几何/深度/线稿/遮罩输出（交互引擎专用）。"""
    raise NotImplementedError


@activity.defn(name="realism-pass")
async def apply_realism_pass(request: RealismPassRequest) -> ActivityResult:
    """生成式写实化（工厂效果图同用）。"""
    raise NotImplementedError


@activity.defn(name="consistency-check")
async def check_consistency(artifact_id: str) -> ActivityResult:
    """户型与跨视角一致性校验（含母版遮罩比对的确定性 QA）。"""
    raise NotImplementedError


@activity.defn(name="compliance-check")
async def check_compliance(artifact_id: str) -> ActivityResult:
    """内容安全机检：工厂与交互两条路径都强制。"""
    raise NotImplementedError


ACTIVITY_REGISTRY: dict[str, Callable[..., Coroutine[Any, Any, ActivityResult]]] = {
    "floorplan-parse": parse_floorplan,
    "plan-layout-solve": solve_plan_layout,
    "plan-rule-check": check_plan_rules,
    "plan-2d-render": render_plan_2d,
    "atmosphere-visual": generate_atmosphere_visual,
    "scene-compile": compile_scene,
    "base-render": render_base,
    "realism-pass": apply_realism_pass,
    "consistency-check": check_consistency,
    "compliance-check": check_compliance,
}
"""注册名 → 实现。键与 contracts 注册表逐字一致（tests/test_activity_registry.py 断言）。"""
