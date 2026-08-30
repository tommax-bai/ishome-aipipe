"""户型图解析编排：一张户型图 → 户型特征标记 + 依据（`floorplan-parse` 的实现件）。

**形态（同渲染层先例 2026-08-29）**：首版不成 activity，以工具（纯库 + CLI）形式存在；
`activities.parse_floorplan` 存根保持不动，**接线时点写死＝上传入口就绪时**——那时图从用户
上传落私有 OSS，activity 拿到的是资产键而不是本地文件路径，入参形态与本模块现在的 bytes 入口
不是一回事，提前接线等于接一遍再改一遍。

四件事在这里收口：**组装 prompt（由闭集数据驱动）→ 解析模型输出的逐条判定 → 投影成产物
→ 过两层校验**。校验是硬门禁（`layout_features` 模块，越界响亮失败），prompt 只是第一道
不是门禁——判据下沉次序 schema > 规则 > prompt（同报告成文线），故 prompt 说的每条约束
凡能机检的都在 `layout_features` 里另有一道。

**模型输出层与产物层分开**（2026-08-30 改造，用户裁决）：模型逐条给 `holds` 与依据
（:class:`~genpipe_worker.models.FloorplanVerdicts`），代码只把判成立的投影成产物
（:func:`to_floorplan_features`）。**下游契约一个字没动**，改的是解析器与模型之间那一层。
判据是"结构性堵死 > 纪律禁止"：上一版的结构里"这条不成立"没有位置，模型只好把否定写进
依据栏，于是产出成了"名字全在、依据全是否定句"——在键存在即触发的语义下四条规则全会触发。
给"成不成立"一个位置之后，逐条作答变成合法输出，不再需要靠下游检查去发现它。

读图是**分区读**（用户裁决 2026-08-30 晚）：勘测定位 → 代码裁剪放大 → 逐房间读图例 → 判定。
理由与代价见 :func:`read_floorplan_features`。朝向不在这里推——它由 `orientation`
按指北针与窗墙**算**出来再喂进判定提示（方位不由 LLM 决定）。

**本步不出任何数字**：尺寸、面积、比例都要等比例标定链路，且算术由确定性代码做不由模型做
（红线"数字不由 LLM 决定"）。模型只出标记与依据。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from genpipe_worker.floorplan_regions import read_room_legends
from genpipe_worker.floorplan_survey import survey_floorplan
from genpipe_worker.layout_features import (
    check_feature_names,
    check_features,
    load_closed_set,
)
from genpipe_worker.models import (
    FloorplanFeatures,
    FloorplanReading,
    FloorplanVerdicts,
    RoomLegend,
    RoomOrientation,
    VisionReader,
)
from genpipe_worker.orientation import to_room_orientations

PARSE_LOGICAL_MODEL = "floorplan-parse.default"
"""任务级逻辑模型名（变化轴 3）：物理 model_id 映射在 infra 的 LiteLLM 配置，换模型不改代码。"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$")

_SYSTEM_PROMPT = """\
你是户型图判读器。给你一张户型图和一份候选标记清单，**清单上每一条你都要判一次**：
它对这套户型成不成立，以及你这样判的依据。

五条纪律：
1. 只认图上真实可见的东西。图上没画、没写的一律不判成立——包括常识上很可能但图上没有的。
2. 只判清单上的那些。图上读到的其他东西放观察区（那里是自由的，看到什么写什么）。
3. 判成立要指得出图上的东西（哪个图例、画在哪、哪段文字）；指不出来就判不成立，
   并把缺什么写进读不出区。误报比漏报贵得多——漏报只是少触发一条规则，误报会让整份报告写错。
   下面给了**逐房间图例**（每个房间单独放大看过一遍的结果）和**已经算好的朝向**，
   判之前先查这两份材料里与这条标记相关的那几行。
4. 不出数字。依据里禁止出现尺寸、面积、比例（"800 毫米""约 92 平方米""三分之一"这类），
   也禁止引用任何标准号——依据只说**图上看见了什么**（画了什么图例、标了什么房间名、
   有没有指北针）。尺寸不在这一步的射程内，它由另一条确定性链路算。
5. 读不出的东西要响亮说明，不许猜、不许留空混过去。

读图约定（通用，不针对任何一条标记）：户型图靠图例说话——实线是墙体与门窗；**虚线框通常是家具
或设备的示意位**（洗衣机、柜体、床、餐桌）；带圆点或十字的小图形通常是地漏、插座一类点位；
房间名与卖点文案是图上的文字。**图例本身就是图上可见的证据**：你说得出它画在哪、长什么样，
它就能当依据；说不出就别用。图例画的是"这里放得下什么"，不是尺寸——别拿它推算大小。
**图例认不出具体是哪件东西不影响它是个位置**：户型图上的家具设备一律是示意画法
（图纸自己的免责声明就写着"仅为示意参考"），本来就认不出型号。一个设备位虚线框画在那里，
说明的是"这里预留了一个位置"；要求它长得像某件具体电器才承认，等于要求图纸做它不做的事。

输出严格 JSON，不要代码围栏、不要任何解释文字，**字段按下面的顺序写**：
{
  "observations": [{"subject": "<部位或主题>", "finding": "<图上看到的事实>"}],
  "verdicts": [{"feature": "<清单上的标记名>", "holds": true,
                "evidence": "<你这样判的依据，一句人话>"}],
  "unreadable": [{"subject": "<读不出的东西>", "reason": "<为什么读不出>"}]
}
先写 `observations`：说你在图上和逐房间图例里看到的事实（哪个房间画了什么、门窗怎么开），
这是在看图不是在下结论。看完再写 `verdicts`。
`verdicts` 里**清单上的每一条各出现一次**，`holds` 填 true 或 false，两种都要写依据。
`feature` 只能是清单上的名字，不要自己造名字。
"""

_USER_PROMPT_TEMPLATE = """\
【候选标记清单】（标记名：含义）
{closed_set_lines}

【逐房间图例】（每个房间单独裁出来放大看过一遍，这是图上真实画着的东西）
{room_legends}

【朝向】（已按图上的指北针换算好，直接用，**不要自己再推一遍**）
{orientations}

判读这张户型图，清单上每条各判一次，按系统提示里的形态输出 JSON。
朝向那一栏是算出来的结论，与你对整图的印象不一致时**以那一栏为准**。
朝向栏里没列到的房间＝那个房间没画窗，不要替它补一个朝向。
"""


class FloorplanParseError(Exception):
    """模型输出不可解析为产物——响亮失败，不吞不猜。

    `details` 逐条人话，CLI 原样上报。
    """

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


def build_system_prompt() -> str:
    """系统提示：角色与五条纪律，与闭集无关（闭集在用户提示里，随契约走）。

    **提示里不描述任何错误形态**：本线实测过一次反效果——把"依据不许写否定句"讲得越细，
    模型越是照着写（同报告线的 prompt 铁律"禁止词面进 prompt 就会被照抄"）。
    这里只正面说每条都要判、判不了的写读不出区；错误形态归机检。
    """
    return _SYSTEM_PROMPT


def build_user_prompt(
    closed_set: Mapping[str, str],
    room_legends: Sequence[RoomLegend],
    orientations: Sequence[RoomOrientation],
) -> str:
    """用户提示：候选标记清单 + 逐房间图例 + **算好的**朝向。

    清单**从契约数据生成、不手写进 prompt**——契约加一条标记，prompt 自动多一行；
    手写就会长出第二份闭集，两份必漂（同"映射表一旦存在就会与数据漂移"）。
    含义原样取自契约，不在本侧加判定口径：加了就是把规则知识塞进 prompt，
    而扩集纪律是"先有规则、后有标记"，判定口径该长在契约里。

    朝向以**换算结果**的形态进提示，不是让模型自己看指北针推——推的那一版同一张图出过
    三个答案。没画窗的房间不进这一栏（提示里明说"没列到＝没画窗"，免得模型替它补一个）。
    """
    lines = "\n".join(f"- {name}：{meaning}" for name, meaning in sorted(closed_set.items()))
    legend_lines = (
        "\n".join(f"【{legend.room}】{legend.legend}" for legend in room_legends)
        or "（没有逐房间图例）"
    )
    orientation_lines = (
        "\n".join(
            f"- {item.room}：窗开在{'、'.join(item.window_walls)}，朝{'、'.join(item.facings)}"
            for item in orientations
            if item.facings
        )
        or "（没有一个房间画了窗，或图上读不出窗的位置）"
    )
    return _USER_PROMPT_TEMPLATE.format(
        closed_set_lines=lines,
        room_legends=legend_lines,
        orientations=orientation_lines,
    )


def parse_model_output(raw: str) -> FloorplanVerdicts:
    """把模型原文解析成**逐条判定**（模型输出层）。围栏与前后缀寒暄容忍，形态不对即失败。"""
    text = _FENCE_RE.sub("", raw.strip()).strip()
    match = _JSON_BLOCK_RE.search(text)
    if match is None:
        raise FloorplanParseError([f"模型输出里找不到 JSON 对象：{raw.strip()[:400]}"])
    try:
        payload = json.loads(match.group(0))
    except ValueError as e:
        raise FloorplanParseError([f"模型输出不是合法 JSON：{e}"]) from e
    try:
        return FloorplanVerdicts.model_validate(payload)
    except ValidationError as e:
        raise FloorplanParseError(
            [f"{'.'.join(str(x) for x in err['loc'])}：{err['msg']}" for err in e.errors()]
        ) from e


def to_floorplan_features(verdicts: FloorplanVerdicts) -> FloorplanFeatures:
    """投影：逐条判定 → 产物。**只取 `holds=True` 的**，键＝标记名、值＝依据。

    同一个标记判两次即失败，不静默取其中一条——两条判定说的是相反的事时，"取后一条"
    这种默认行为会把矛盾藏起来；观察区与读不出区原样带过去（它们本来就不参与匹配）。
    """
    seen = Counter(v.feature for v in verdicts.verdicts)
    duplicated = sorted(name for name, count in seen.items() if count > 1)
    if duplicated:
        raise FloorplanParseError(
            [f"标记 `{name}` 在判定里出现了不止一次：一条标记只能判一次" for name in duplicated]
        )
    return FloorplanFeatures(
        layout_features={v.feature: v.evidence for v in verdicts.verdicts if v.holds},
        observations=verdicts.observations,
        unreadable=verdicts.unreadable,
    )


async def read_floorplan_features(
    image_bytes: bytes,
    image_media_type: str,
    reader: VisionReader,
    *,
    logical_model: str = PARSE_LOGICAL_MODEL,
    closed_set: Mapping[str, str] | None = None,
) -> FloorplanReading:
    """读一张户型图：**勘测 → 裁剪放大逐块读 → 换算朝向 → 逐条判定 → 名字校验 → 投影 → 产物校验**。

    **分区读**（用户裁决 2026-08-30 晚）。判据来自九次真跑：整图单次读一直看不见阳台端头
    那两个虚线设备位，三种 prompt 框架都没救回来；而模型被单独问那一小块时逐个说得出来
    ——不是 prompt 问题，是分辨率问题。所以先让模型指出每个房间占图的哪一块，
    **代码按块裁剪放大**（剪刀在代码手里），每块单独送一次读图例，汇总回判定层。
    **按每个房间规划，不是整体规划。**

    **朝向由代码换算不由模型判断**（同批裁决）：模型只报指北针指向与窗开在哪面墙，
    `orientation` 把它换算成方位再喂进判定提示——方位与数字同族，都不由 LLM 决定。
    窗墙取自**近景**那一步：整图勘测报的窗墙实测错过两处（次卧报成右墙、给无窗的卫生间
    报了一面西窗），后者经换算变成"卫生间朝西"、催出一次误报；近景两处都读对了。

    **代价**：一张图从 1 次调用变成 1 + N + 1 次（N＝房间数），`model_call_count` 记下实数。

    名字校验在**投影之前**，覆盖 `holds` 真假两种——判不成立的越界名同样是编造标记名。
    校验不过就抛（`LayoutFeatureViolation`）——**不修剪、不丢弃、不降级**：静默剔掉越界键
    等于把"解析侧在编造标记"藏起来，而下游拿到的是剔完的结果，问题永远浮不出来。
    """
    features_closed_set = dict(closed_set) if closed_set is not None else load_closed_set()
    survey = await survey_floorplan(image_bytes, image_media_type, reader, logical_model)
    room_legends = await read_room_legends(image_bytes, survey.rooms, reader, logical_model)
    orientations = to_room_orientations(survey.north_points_to, room_legends)
    raw = await reader.complete_with_image(
        logical_model,
        build_system_prompt(),
        build_user_prompt(features_closed_set, room_legends, orientations),
        image_bytes,
        image_media_type,
    )
    verdicts = parse_model_output(raw)
    check_feature_names((v.feature for v in verdicts.verdicts), features_closed_set)
    features = to_floorplan_features(verdicts)
    check_features(features.layout_features, features_closed_set)
    return FloorplanReading(
        logical_model=logical_model,
        raw_output=raw,
        survey=survey,
        room_legends=room_legends,
        orientations=orientations,
        verdicts=verdicts.verdicts,
        features=features,
        model_call_count=len(survey.rooms) + 2,
    )
