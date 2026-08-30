"""户型特征标记闭集：加载 + 两层校验（契约 `rulebook/layout_features.md` §四）。

**没有这道校验，键写错就是永远不触发且不报错**——本项目最贵的失效形态（同"判据入册 ≠ 有执行器"）。
故本模块是纯确定性的：不依赖模型调用、不依赖网络、不依赖产物模型，只吃名字与 map 出判定
（依赖方向由 import-linter 锁死）。

**校验分两层，对应解析的两层**（2026-08-30 改造）：

- :func:`check_feature_names` 作用在**模型输出层的逐条判定**上，只管名字 ⊆ 闭集，
  **不论这条判成立还是不成立**——`holds=False` 的越界名同样说明模型在编造标记名；
- :func:`check_features` 作用在**投影后的产物**上，管要下发的那些标记干不干净。

名字越界前移到判定层，是因为那才是它发生的地方；产物层照旧再查一遍名字，拦的是投影这段
代码自己写错（两道不是重复，是两个失效源各一道）。

闭集真源在 ishome-contracts `rulebook/layout_features.json`；本仓持一份**逐字副本**
（`layout_features.json`，与本模块同目录），两处不一致时以 contracts 仓为准并回改此处——
副本口径同 activity 注册名（`tests/test_activity_registry.py`），一致性由
`tests/test_floorplan_parse.py` 断言。之所以持副本而不在运行时读兄弟仓：
解析件要能在只有本仓的环境里跑（容器内没有 contracts 工作副本）。

匹配语义（契约 §一）：**键存在即触发**，值是依据留痕、不参与匹配——但值**必须有**且必须干净：
空依据等于"这条标记成立但说不出为什么"，是编造的入口；依据会随触发的规则下发进报告当
"因为你家……"的来源，故其中的量纲数字与标准号一并判为违规（依据只说图上看见了什么）。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

CLOSED_SET_FILE = Path(__file__).with_name("layout_features.json")
"""契约闭集副本（数据文件；配置只放数据，逻辑在本模块）。"""


class LayoutFeatureSetError(Exception):
    """闭集副本读不出来——解析件不许在不知道闭集的情况下产出标记，故直接失败。"""


class LayoutFeatureViolation(Exception):
    """标记不合契约：越界名（判定层或产物层）、空依据、自相矛盾的依据、量纲数字、标准号。

    ``details`` 逐条给出人话，CLI 与 activity 都原样上报——响亮失败要报得出**是哪个键**，
    只说"校验不通过"等于没报。
    """

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


def load_closed_set(path: Path | None = None) -> dict[str, str]:
    """读闭集副本，返回 `标记名 → 含义`（含义原样取自契约，进 prompt 用）。"""
    source = path or CLOSED_SET_FILE
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LayoutFeatureSetError(f"闭集副本读取失败：{source}（{e}）") from e
    features = raw.get("features") if isinstance(raw, dict) else None
    if not isinstance(features, dict) or not features:
        raise LayoutFeatureSetError(f"闭集副本形态不对（缺 features 或为空）：{source}")
    closed_set: dict[str, str] = {}
    for name, spec in features.items():
        meaning = spec.get("meaning") if isinstance(spec, dict) else None
        if not isinstance(name, str) or not isinstance(meaning, str) or not meaning.strip():
            raise LayoutFeatureSetError(f"闭集副本条目缺 meaning：{name!r} @ {source}")
        closed_set[name] = meaning
    return closed_set


def check_feature_names(names: Iterable[str], closed_set: Mapping[str, str]) -> None:
    """判定层校验：逐条判定里的标记名 ⊆ 闭集，**不论 holds 真假**。

    `holds=False` 的越界名不是"反正不会下发所以无害"——它说明模型在编造闭集里没有的标记名，
    而下一次它可能把同一个名字判成成立。名字这件事在这里拦，比等到投影之后再拦更接近现场。

    不做修剪、不做丢弃——静默剔除越界名等于把"解析侧在编造标记"这件事藏起来。
    """
    details = [
        f"越界标记 `{name}`：不在闭集内（闭集＝{'、'.join(sorted(closed_set))}）"
        "——闭集外的东西进观察区，扩集要与消费它的规则同批提交"
        for name in sorted(set(names))
        if name not in closed_set
    ]
    if details:
        raise LayoutFeatureViolation(details)


def check_features(features: Mapping[str, str], closed_set: Mapping[str, str]) -> None:
    """产物层校验（作用在投影后、要下发的那些标记上），任一不过抛 :class:`LayoutFeatureViolation`：

    1. **键 ⊆ 闭集**（契约 §四）——名字越界主要由 :func:`check_feature_names` 在判定层拦，
       这里再查一遍拦的是投影代码自己写错；本项目最贵的失效形态是键写错后永远不触发且不报错；
    2. **依据非空**——"标记成立但说不出为什么"是编造的入口；
    3. **依据不自相矛盾**——投影进来的都是 `holds=True` 的，依据却在说"判不准""不成立"
       "图上没画"，那这条判定本身是坏的：`holds` 与 `evidence` 各说各话时，下游只看键在不在，
       会按"成立"用它。这道 2026-08-30 立案时是拦"把闭集当逐条打勾的清单填"的最后一关；
       判定层给了"否"一个合法位置之后，它的职责收窄为自相矛盾检查；
    4. **依据里没有量纲数字、没有标准号**——依据会随触发的规则下发到报告里当"因为你家……"
       的可追溯来源，模型自己写的尺寸/面积/比例混进去就等于报告里出现了 LLM 决定的数字
       （红线）；标准号则从来不会画在户型图上，出现即编造。这几条 prompt 里都说了，
       但 prompt 是第一道不是门禁，机检才是。
    """
    details: list[str] = []
    for name in sorted(features):
        if name not in closed_set:
            details.append(
                f"越界标记 `{name}`：不在闭集内（闭集＝{'、'.join(sorted(closed_set))}）"
                "——投影后仍出现越界键，说明投影这段代码本身有问题"
            )
            continue
        evidence = features[name]
        if not evidence.strip():
            details.append(f"标记 `{name}` 的依据为空：标记成立必须说得出图上的依据，禁留空")
            continue
        details.extend(_evidence_violations(name, evidence))
    if details:
        raise LayoutFeatureViolation(details)


_MEASURED_NUMBER_RE = re.compile(
    r"[0-9０-９一二两三四五六七八九十百千零点．.]+\s*"
    r"(?:mm|cm|㎜|㎝|m2|m²|㎡|%|％|毫米|厘米|平方米|平米|平方|米|分之)"
)
"""量纲数字：数（含中文数字）紧跟长度/面积/比例单位。纯计数（"两处设备位"）不在此列
——它描述图上画了什么，不是设计参数。"""

_STANDARD_CODE_RE = re.compile(r"(?:GB|JGJ|CJJ|DBJ|DB|JG|T)\s*/?\s*T?\s*[-—]?\s*\d{3,}")
"""标准号：户型图上从不画标准号，依据里出现它只可能是模型自己编的。"""

_NEGATED_EVIDENCE_RE = re.compile(
    r"无法|不能|未能|无从|不足以|难以确认"  # 判不准
    r"|不成立|不满足|不构成|不符合|不适用|不属于|并非|不是|非[UuＵ]\s*形"  # 判为不成立
    r"|未画|未标|未设|未见|未提供|未标注|无任何|没有任何"  # 说"图上没有"
)
"""否定/不确定措辞。投影进产物的都是判成立的，依据却是这三族之一＝这条判定自相矛盾。

三族分别是：判不准、判为不成立、说图上没有。措辞取自 2026-08-30 真跑里模型写过的原话
（"未画出洗衣机位""无法判定是否西晒"）——那一版结构里没有"否"的位置，模型只能这么写；
现在有了，再出现就说明 `holds` 与 `evidence` 各说各话。

**不收单个"无/未/没有"**：`阳台与客厅之间无隔墙` 是描述图面的正当依据，一竿子打死会把
真依据也拦掉；收的是"未+动词"与"无任何"这种明确在说"图上不存在"的构造。
"""


def _evidence_violations(name: str, evidence: str) -> list[str]:
    violations: list[str] = []
    negated = _NEGATED_EVIDENCE_RE.search(evidence)
    if negated is not None:
        violations.append(
            f"标记 `{name}` 判成立、依据却在说它不成立（出现「{negated.group(0)}」）："
            "判定与依据自相矛盾，而下游只看键在不在、不读依据，会按成立用它"
        )
    measured = _MEASURED_NUMBER_RE.search(evidence)
    if measured is not None:
        violations.append(
            f"标记 `{name}` 的依据里出现量纲数字「{measured.group(0).strip()}」："
            "依据只说图上看见了什么，尺寸/面积/比例由标定链路的确定性代码算，不由模型出"
        )
    standard = _STANDARD_CODE_RE.search(evidence)
    if standard is not None:
        violations.append(
            f"标记 `{name}` 的依据里出现标准号「{standard.group(0).strip()}」："
            "户型图上不画标准号，依据里出现它即编造"
        )
    return violations
