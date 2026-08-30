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


class FeatureVerdict(_FloorplanModel):
    """模型对**一条候选标记**的判定：成不成立，以及为什么。

    这个字段是 2026-08-30 改造的全部要点。上一版让模型直接产 `{标记名: 依据}`，
    结构里**没有"否"的位置**——而模型面对四条候选的自然做法是逐条作答（这本身是负责任的），
    于是"否"只能写进依据栏，产出成了"名字全在、依据全是否定句"。在"键存在即触发"的语义下，
    那份产物会让四条规则全部触发。

    问题不在模型在结构。给"成不成立"一个位置之后，逐条作答变成合法输出，代码只投影
    `holds=True` 的那些（:func:`floorplan_parse.to_floorplan_features`）——
    **结构性堵死，不是纪律禁止**（同"推导步看不见落点的值所以产不出数字"）。

    `evidence` 两种 holds 都要有：`True` 时是这条标记成立的依据（会投影进产物、随规则下发），
    `False` 时是判定不成立的理由（**只留痕、不下发**）。
    """

    feature: str
    holds: bool
    evidence: str


class FloorplanVerdicts(_FloorplanModel):
    """模型输出层：逐条判定 + 观察区 + 读不出区。**这一层不下发**，投影后才是产物。

    与产物层 :class:`FloorplanFeatures` 分开是刻意的：契约要的是"只含成立的标记"，
    而模型要的是"每条都能作答"——两个需求各有自己的形态，中间夹一次确定性投影。
    """

    verdicts: list[FeatureVerdict] = []
    observations: list[LayoutObservation] = []
    unreadable: list[UnreadableGap] = []


class FloorplanFeatures(_FloorplanModel):
    """一张户型图的解析产物：**只有成立的标记与依据，没有任何数字**。

    尺寸/面积/比例不在本产物射程内——它们要等比例标定链路，且算术由确定性代码做不由模型做
    （红线"数字不由 LLM 决定"）。

    `layout_features` 的键 ⊆ 契约闭集（`layout_features.check_features` 强制，越界响亮失败），
    值是**这条标记成立的依据**（人话，图上真实可见的证据）。本层形态与语义是下游契约，
    模型输出层怎么改都不动它。
    """

    layout_features: dict[str, str] = {}
    observations: list[LayoutObservation] = []
    unreadable: list[UnreadableGap] = []


class FloorplanReading(_FloorplanModel):
    """一次读图的完整留痕：产物 + 逐条判定 + 用的逻辑模型名 + 模型原文。

    留 `raw_output` 是为了**判定可复核**——产物里每条依据都能回到模型原文对一遍，
    差异不必靠回忆（同报告线"真跑逐字对照"纪律）。
    留 `verdicts` 是因为**没成立的那几条也是数据**：模型为什么判它不成立，是下一轮改判据、
    改闭集、改读图方式的素材；投影之后这些就看不见了。
    """

    logical_model: str
    raw_output: str
    verdicts: list[FeatureVerdict]
    features: FloorplanFeatures
