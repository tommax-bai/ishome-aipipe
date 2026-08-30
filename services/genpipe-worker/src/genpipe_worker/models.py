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

分区读与朝向换算（2026-08-30 晚，用户裁决）新增的中间层模型：`FloorplanSurvey`（整图勘测：
房间在图上的区域 + 窗开在哪面墙 + 指北针指向）、`RoomLegend`（每块裁剪放大后读到的图例）、
`RoomOrientation`（**代码换算**出来的朝向）。三者都是解析器内部形态，**不下发**——
下发面只有 `FloorplanFeatures`。
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

PageSide = Literal["top", "bottom", "left", "right"]
"""图面四边。房间的窗开在哪一边、指北针的 N 指向哪一边，都用这套词。

用图面词而不是方位词是刻意的：**模型只报它看得见的事（窗画在房间的哪条边上），
方位由代码算**（`orientation` 模块）——同"数字不由 LLM 决定"，方位也不由 LLM 决定。
真跑证据：让模型自己推时，同一张图的主卧朝向出过"西南/南/西"三个答案，
因为它每次都在现场心算、每次推的路径都不一样。
"""


class VisionReader(Protocol):
    """视觉补全协议位：由组合根注入（生产＝LiteLLM 网关客户端，单测＝桩件）。

    放在类型层而不是某个消费方模块里，是因为解析的三步（勘测 / 逐块读图例 / 判定）
    都要用它，而它们互不可见——端口下沉到共同的底层，避免为共用一个协议开横向依赖。
    """

    async def complete_with_image(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        image_media_type: str,
        *,
        temperature: float = 0.0,
    ) -> str: ...


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


class RoomRegion(_FloorplanModel):
    """勘测一步里的一个房间：图上写的名字、它占图的哪一块、窗开在它的哪几条边上。

    `box` 是归一化坐标 `[x0, y0, x1, y1]`（0~1，左上角为原点），**给代码裁剪用**——
    裁剪与放大是确定性动作，不由模型做（模型只指位置，剪刀在代码手里）。

    这一层**不报窗开在哪面墙**：那件事在放大后的近景里才看得准（真跑证据见 `RoomLegend`），
    整图勘测只负责"这个房间在哪一块"。
    """

    name: str
    box: tuple[float, float, float, float]


class FloorplanSurvey(_FloorplanModel):
    """整图勘测产物：指北针指向 + 每个房间占图的哪一块。

    `north_points_to` 为 `None` ＝**图上没有指北针**（是事实不是缺失）；换算时退到
    通行约定，见 `orientation.DEFAULT_NORTH_POINTS_TO`。
    """

    north_points_to: PageSide | None = None
    rooms: list[RoomRegion] = Field(default_factory=list)


class RoomLegend(_FloorplanModel):
    """一块裁剪放大后读到的东西：图例（自由文本）+ 窗开在这一块的哪几条边。

    存在理由是九次真跑的结论：整图单次读**看不见**阳台端头那两个虚线设备位，
    而模型被单独问那一小块时逐个说得出来——不是 prompt 问题，是分辨率问题。

    `window_walls` 放在这一层而不是勘测层，同一个道理：**事实在看得最清楚的地方定**。
    真跑证据（2026-08-30 晚）——整图勘测把次卧的飘窗报成右墙（实为下墙），
    还给没有窗的卫生间报了一面西窗；而同一次跑里，这两个房间的近景读出的是对的。
    那面凭空多出来的西窗经确定性换算变成"卫生间朝西"，直接催出了一次 `west_facing` 误报。
    """

    room: str
    legend: str
    window_walls: list[PageSide] = Field(default_factory=list)


class RoomOrientation(_FloorplanModel):
    """**代码算出来的**房间朝向：窗所在墙面对应的方位。

    朝向取决于**窗开在哪面墙**，不取决于房间在图上的位置——样本那张图主卧在左下角、
    飘窗画在下侧墙上，所以是朝南；按房间位置推会得出"西南"，那是错的。
    `facings` 为空＝这个房间没有画窗（是事实，不猜）。

    换算是算术、必然可复现；**但它算不出比输入更准的东西**——窗墙报错，这里就跟着错，
    而且错得像个确定结论（真跑里有过这种误报，见 :class:`RoomLegend`）。
    """

    room: str
    window_walls: list[PageSide] = Field(default_factory=list)
    facings: list[str] = Field(default_factory=list)


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

    勘测、逐块图例、换算出的朝向三样一并留：分区读把一次调用变成了 1 + N + 1 次，
    **贵到必须能回答"这钱花得值不值"**——留着它们才看得出漏判是勘测框错了、这一块没读到、
    还是判定那一步没用上。`model_call_count` 就是这轮的调用次数。
    """

    logical_model: str
    raw_output: str
    survey: FloorplanSurvey
    room_legends: list[RoomLegend]
    orientations: list[RoomOrientation]
    verdicts: list[FeatureVerdict]
    features: FloorplanFeatures
    model_call_count: int
