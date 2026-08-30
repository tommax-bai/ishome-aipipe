"""genpipe_worker activity 出入参模型（pydantic）。

跨 domain 纪律：worker 不 import 其他 domain 的内部模块，activity 入参出参
以本模块与（后续）contracts 生成 SDK 为准。

V1.4 裁决（2026-08-23）：绘图 activity 请求模型（PlanRenderRequest /
AtmosphereVisualRequest / BaseRenderRequest / RealismPassRequest）随绘图能力
物理拆分迁出，分别由 ishome-render2d / ishome-imagegen / ishome-render3d
各自持有。本仓保留的非绘图 activity 当前入参均为标量（str），暂无请求模型。

户型图解析产物（2026-08-30 落地）：`FloorplanFeatures` 三段分明——闭集内的标记进
`layout_features`（下发给报告求值线），闭集外的观察进 `observations`（**记录但不下发**），
读不出的东西进 `unreadable`（响亮说明，不猜不留空）。序列化用 camelCase 别名对齐
contracts 报告数据包的 `anonymousProfile.layoutFeatures`（Jackson 口径，同 reportgen 侧）。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _FloorplanModel(BaseModel):
    """解析产物基类：camelCase 别名对齐契约序列化；extra=forbid 拒收模型自造的字段。"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class LayoutObservation(_FloorplanModel):
    """闭集外的观察项：读到了、但**没有规则消费它**，故不进 `layout_features`。

    它不是垃圾桶而是**后续立规则的素材**（契约 §三：新增标记必须与消费它的规则同批提交，
    先有规则后有标记）。`subject` 是部位或主题（"厨房"/"通风"），`finding` 是图上看到的事实。
    """

    subject: str
    finding: str


class UnreadableGap(_FloorplanModel):
    """这张图上读不出来的东西：**要响亮说明，不许猜、不许留空混过去**。

    `subject` 是读不出的东西（"分房间尺寸"），`reason` 是为什么读不出（"全图无任何尺寸标注"）。
    读不出是事实不是降级——相关规则不触发是正确行为（同"朝向缺席时遮光规则不触发"口径）。
    """

    subject: str
    reason: str


class FloorplanFeatures(_FloorplanModel):
    """一张户型图的解析产物：**只有标记与依据，没有任何数字**。

    尺寸/面积/比例不在本产物射程内——它们要等比例标定链路，且算术由确定性代码做不由模型做
    （红线"数字不由 LLM 决定"）。

    `layout_features` 的键 ⊆ 契约闭集（`layout_features.check_features_within_closed_set`
    强制，越界响亮失败），值是**这条标记成立的依据**（人话，图上真实可见的证据）。
    """

    layout_features: dict[str, str] = {}
    observations: list[LayoutObservation] = []
    unreadable: list[UnreadableGap] = []


class FloorplanReading(_FloorplanModel):
    """一次读图的完整留痕：产物 + 用的逻辑模型名 + 模型原文。

    留 `raw_output` 是为了**判定可复核**——产物里每条依据都能回到模型原文对一遍，
    差异不必靠回忆（同报告线"真跑逐字对照"纪律）。
    """

    logical_model: str
    raw_output: str
    features: FloorplanFeatures
