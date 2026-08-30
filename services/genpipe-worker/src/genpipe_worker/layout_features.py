"""户型特征标记闭集：加载 + 解析产出侧校验（契约 `rulebook/layout_features.md` §四 第一行）。

**没有这道校验，键写错就是永远不触发且不报错**——本项目最贵的失效形态（同"判据入册 ≠ 有执行器"）。
故本模块是纯确定性的：不依赖模型调用、不依赖网络、不依赖产物模型，只吃 map 出判定
（依赖方向由 import-linter 锁死）。

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
from collections.abc import Mapping
from pathlib import Path

CLOSED_SET_FILE = Path(__file__).with_name("layout_features.json")
"""契约闭集副本（数据文件；配置只放数据，逻辑在本模块）。"""


class LayoutFeatureSetError(Exception):
    """闭集副本读不出来——解析件不许在不知道闭集的情况下产出标记，故直接失败。"""


class LayoutFeatureViolation(Exception):
    """产出的标记不合契约：越界键、空依据、否定句依据、依据里的量纲数字或标准号。

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


def check_features(features: Mapping[str, str], closed_set: Mapping[str, str]) -> None:
    """产出侧四道校验，任一不过抛 :class:`LayoutFeatureViolation`：

    1. **键 ⊆ 闭集**（契约 §四）——本项目最贵的失效形态是键写错后永远不触发且不报错；
    2. **依据非空**——"标记成立但说不出为什么"是编造的入口；
    2b. **依据不是否定句**——匹配语义是"键存在即触发"，下游只看键在不在、**根本不读依据**；
       模型把闭集当成逐条打勾的清单填（依据写"非 U 形""无法判定"）时，四条规则会带着
       否定的理由全部触发。这是真跑里出现过两次的形态（2026-08-30，两步读与单次读各一次），比误报更贵；
    3. **依据里没有量纲数字、没有标准号**——依据会随触发的规则下发到报告里当"因为你家……"
       的可追溯来源，模型自己写的尺寸/面积/比例混进去就等于报告里出现了 LLM 决定的数字
       （红线）；标准号则从来不会画在户型图上，出现即编造。这几条 prompt 里都说了，
       但 prompt 是第一道不是门禁，机检才是。

    不做修剪、不做丢弃——静默剔除越界键等于把"解析侧在编造标记"这件事藏起来，
    而下游报告求值线读到的是剔除后的结果，问题永远浮不出来。
    """
    details: list[str] = []
    for name in sorted(features):
        if name not in closed_set:
            details.append(
                f"越界标记 `{name}`：不在闭集内（闭集＝{'、'.join(sorted(closed_set))}）"
                "——下发闭集外的键等于宣称有规则会用它，那是假的；"
                "闭集外的东西进观察区，扩集要与消费它的规则同批提交"
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
"""否定/不确定措辞：这些出现在依据里，说的是"这条标记不成立"，而键写上去等于说它成立。

真跑里模型两次把闭集当逐条打勾的清单填（2026-08-30），依据写的正是"未画出洗衣机位"
"无法判定是否西晒"这一类——所以三族都收：判不准、判为不成立、说图上没有。

**不收单个"无/未/没有"**：`阳台与客厅之间无隔墙` 是描述图面的正当依据，一竿子打死会把
真依据也拦掉；收的是"未+动词"与"无任何"这种明确在说"图上不存在"的构造。
"""


def _evidence_violations(name: str, evidence: str) -> list[str]:
    violations: list[str] = []
    negated = _NEGATED_EVIDENCE_RE.search(evidence)
    if negated is not None:
        violations.append(
            f"标记 `{name}` 的依据是否定句（出现「{negated.group(0)}」）："
            "匹配语义是键存在即触发、下游不读依据，"
            "所以不成立的标记根本不该出现在这里——理由写进读不出区"
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
