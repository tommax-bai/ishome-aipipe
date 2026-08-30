"""户型图解析编排：一张户型图 → 户型特征标记 + 依据（`floorplan-parse` 的实现件）。

**形态（同渲染层先例 2026-08-29）**：首版不成 activity，以工具（纯库 + CLI）形式存在；
`activities.parse_floorplan` 存根保持不动，**接线时点写死＝上传入口就绪时**——那时图从用户
上传落私有 OSS，activity 拿到的是资产键而不是本地文件路径，入参形态与本模块现在的 bytes 入口
不是一回事，提前接线等于接一遍再改一遍。

三件事在这里收口：**组装 prompt（由闭集数据驱动）→ 解析模型输出 → 过产出侧校验**。
其中校验是硬门禁（`layout_features` 模块，越界响亮失败），prompt 只是第一道不是门禁
——判据下沉次序 schema > 规则 > prompt（同报告成文线），故 prompt 说的每条约束
凡能机检的都在 `layout_features` 里另有一道。

读图是**一次调用读整张图**；两步读（清点 → 判定）试过并撤回，理由见
:func:`read_floorplan_features`。

**本步不出任何数字**：尺寸、面积、比例都要等比例标定链路，且算术由确定性代码做不由模型做
（红线"数字不由 LLM 决定"）。模型只出标记与依据。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Protocol

from pydantic import ValidationError

from genpipe_worker.layout_features import check_features, load_closed_set
from genpipe_worker.models import FloorplanFeatures, FloorplanReading

PARSE_LOGICAL_MODEL = "floorplan-parse.default"
"""任务级逻辑模型名（变化轴 3）：物理 model_id 映射在 infra 的 LiteLLM 配置，换模型不改代码。"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$")

_SYSTEM_PROMPT = """\
你是户型图判读器。唯一任务：从一张户型图上读出**户型特征标记**，并为每条标记给出图上可见的依据。

六条纪律：
0. **layoutFeatures 只放成立的标记**：不成立、判不准、图上没依据的，不要出现在里面，
   写进读不出区（subject 用标记名）。举个形态上的例子，闭集里若有 aaa_bbb 这条标记而
   图上没有对应的东西：
   错 →  "layoutFeatures": {"aaa_bbb": "图上没画 aaa"}
   对 →  "unreadable": [{"subject": "aaa_bbb", "reason": "图上没画 aaa"}]
1. 只认图上真实可见的东西。图上没画、没写的一律不判——包括常识上很可能但图上没有的。
2. 标记名**只能**取自给定的闭集。闭集外的任何发现都放进观察区，绝不写进标记里：
   写进去等于宣称"有规则会用它"，而那是假的。
3. 拿不准就不判。误报比漏报贵得多——漏报只是少触发一条规则，误报会让整份报告写错。
   判不准的放观察区，并在读不出区说明缺什么。
4. 不出数字。依据里禁止出现尺寸、面积、比例（"800 毫米""约 92 平方米""三分之一"这类），
   也禁止引用任何标准号——依据只说**图上看见了什么**（画了什么图例、标了什么房间名、
   有没有指北针）。尺寸不在这一步的射程内，它由另一条确定性链路算。
5. 读不出的东西要响亮说明，不许猜、不许留空混过去。

读图约定（通用，不针对任何一条标记）：户型图靠图例说话——实线是墙体与门窗；**虚线框通常是家具
或设备的示意位**（洗衣机、柜体、床、餐桌）；带圆点或十字的小图形通常是地漏、插座一类点位；
房间名与卖点文案是图上的文字。**图例本身就是图上可见的证据**：你说得出它画在哪、长什么样，
它就能当依据；说不出就别用。图例画的是"这里放得下什么"，不是尺寸——别拿它推算大小。

输出严格 JSON，不要代码围栏、不要任何解释文字，形如：
{
  "layoutFeatures": {"<闭集内的标记名>": "<这条标记成立的依据，一句人话>"},
  "observations": [{"subject": "<部位或主题>", "finding": "<图上看到的事实>"}],
  "unreadable": [{"subject": "<读不出的东西>", "reason": "<为什么读不出>"}]
}
一条标记都不成立时 layoutFeatures 就是 {}——那是正常结果，不是失败。
**依据是一句肯定句**，说图上画了什么让这条标记成立；写不出肯定句，就说明这条标记不该出现。
"""

_USER_PROMPT_TEMPLATE = """\
【特征标记闭集】（标记名：含义）
{closed_set_lines}

判读这张户型图，按系统提示里的形态输出 JSON。

需要图外信息才能判定的标记（朝向这类），先在图上找到对应依据（指北针、朝向文字标注）再判；
找不到依据就不判，并把缺的东西写进读不出区。
"""


class FloorplanParseError(Exception):
    """模型输出不可解析为产物——响亮失败，不吞不猜。

    `details` 逐条人话，CLI 原样上报。
    """

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


class VisionReader(Protocol):
    """视觉补全协议位：由组合根注入（生产＝LiteLLM 网关客户端，单测＝桩件）。"""

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


def build_system_prompt() -> str:
    """系统提示：角色与六条纪律，与闭集无关（闭集在用户提示里，随契约走）。"""
    return _SYSTEM_PROMPT


def build_user_prompt(closed_set: Mapping[str, str]) -> str:
    """用户提示：闭集逐条列出。

    闭集**从契约数据生成、不手写进 prompt**——契约加一条标记，prompt 自动多一行；
    手写就会长出第二份闭集，两份必漂（同"映射表一旦存在就会与数据漂移"）。
    含义原样取自契约，不在本侧加判定口径：加了就是把规则知识塞进 prompt，
    而扩集纪律是"先有规则、后有标记"，判定口径该长在契约里。
    """
    lines = "\n".join(f"- {name}：{meaning}" for name, meaning in sorted(closed_set.items()))
    return _USER_PROMPT_TEMPLATE.format(closed_set_lines=lines)


def parse_model_output(raw: str) -> FloorplanFeatures:
    """把模型原文解析成产物。围栏与前后缀寒暄容忍，形态不对即失败。"""
    text = _FENCE_RE.sub("", raw.strip()).strip()
    match = _JSON_BLOCK_RE.search(text)
    if match is None:
        raise FloorplanParseError([f"模型输出里找不到 JSON 对象：{raw.strip()[:400]}"])
    try:
        payload = json.loads(match.group(0))
    except ValueError as e:
        raise FloorplanParseError([f"模型输出不是合法 JSON：{e}"]) from e
    try:
        return FloorplanFeatures.model_validate(payload)
    except ValidationError as e:
        raise FloorplanParseError(
            [f"{'.'.join(str(x) for x in err['loc'])}：{err['msg']}" for err in e.errors()]
        ) from e


async def read_floorplan_features(
    image_bytes: bytes,
    image_media_type: str,
    reader: VisionReader,
    *,
    logical_model: str = PARSE_LOGICAL_MODEL,
    closed_set: Mapping[str, str] | None = None,
) -> FloorplanReading:
    """读一张户型图，产出经校验的特征标记。

    **一次调用读完整张图**（实现判断 2026-08-30，可推翻）。试过的另一条路是两步读
    （先逐房间清点图例、再带着清点判定），实测**不成立并已撤回**：清点那一步同样答
    "阳台内无任何图例"（漏的东西照旧漏），却把指北针读错了方向，还诱发模型把闭集当成
    逐条打勾的清单填——四条标记全带着"非 U 形""无法判定"这类否定依据出现，而匹配语义是
    键存在即触发，这比误报更贵。留下来的是那次实测逼出来的两道门禁（`layout_features`
    的否定句闸 + 系统提示第 0 条）。**要真正读准图例级细节得按房间分区放大读**，那属于
    解析实现路径的选型，**时点写死＝拿到第二批样本（手机翻拍图与带标注图）那一批一起定**。

    校验不过就抛（`LayoutFeatureViolation`）——**不修剪、不丢弃、不降级**：静默剔掉越界键
    等于把"解析侧在编造标记"藏起来，而下游拿到的是剔完的结果，问题永远浮不出来。
    """
    features_closed_set = dict(closed_set) if closed_set is not None else load_closed_set()
    raw = await reader.complete_with_image(
        logical_model,
        build_system_prompt(),
        build_user_prompt(features_closed_set),
        image_bytes,
        image_media_type,
    )
    features = parse_model_output(raw)
    check_features(features.layout_features, features_closed_set)
    return FloorplanReading(logical_model=logical_model, raw_output=raw, features=features)
