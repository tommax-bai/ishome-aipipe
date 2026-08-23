"""Temporal activities：所有 IO 与重计算收口在此。

注册名唯一真源：ishome-contracts `activities/registry.md`，**只增不改**——改注册名
会破坏历史 workflow 重放，等同于改线上协议；新增走 contracts 仓 PR 评审。
命名规则（规范 §2.4）：注册名 = kebab-case 显式声明；函数名 = 同词 snake_case
动词前置。

V1.4 裁决（2026-08-23）：绘图 activity 物理拆分迁出本仓——
`plan-2d-render` → ishome-render2d（队列 `render2d-activities`）；
`atmosphere-visual` / `realism-pass` → ishome-imagegen（队列 `imagegen-activities`）；
`scene-compile` / `base-render` → ishome-render3d（队列 `render3d-activities`）。
注册名与函数名逐字不变，仅归属仓库变化（contracts `registries/task_queues.md`）。
本仓保留非绘图 activity：解析、求解、校验、门禁（队列 `genpipe-activities`）。
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from temporalio import activity

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
    "consistency-check": check_consistency,
    "compliance-check": check_compliance,
}
"""注册名 → 实现。键与 contracts 注册表中归属 `genpipe-activities` 队列的子集逐字一致
（tests/test_activity_registry.py 按 registries/task_queues.md 口径断言）。"""
