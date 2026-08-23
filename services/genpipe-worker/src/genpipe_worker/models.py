"""genpipe_worker activity 出入参模型（pydantic）。

跨 domain 纪律：worker 不 import 其他 domain 的内部模块，activity 入参出参
以本模块与（后续）contracts 生成 SDK 为准。

V1.4 裁决（2026-08-23）：绘图 activity 请求模型（PlanRenderRequest /
AtmosphereVisualRequest / BaseRenderRequest / RealismPassRequest）随绘图能力
物理拆分迁出，分别由 ishome-render2d / ishome-imagegen / ishome-render3d
各自持有。本仓保留的非绘图 activity 当前入参均为标量（str），暂无请求模型。
"""

from __future__ import annotations
