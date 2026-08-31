"""从面积推出来的默认假设：住几个人、得房率、装修倾向、业主年龄。**确定性，不经模型。**

**为什么这些不问业主**（用户裁决 2026-08-31）：他开头只需要给两样——**面积**与**户型图**。
其余一概推断，产出之后把假设摊开说、给他一个改的入口；**给了就用，不给就算，不追问**。
形态是"先做出来再让他改"，不是"问齐了再做"。

**为什么由代码推而不是模型推**：这些数会进画像、进图、进报告——数字不由 LLM 决定。
表里的数是经验值、锚点只有用户给的一个（138㎡ 住 4~5 人、按 4 人做、得房率 80），
**复看时点＝有真实用户数据时**（同"阈值有数据才定"）。
"""

from __future__ import annotations

from dataclasses import dataclass

SQM_PER_OCCUPANT = 34
"""每人多少平。锚点是用户给的那一个：138 ㎡ → 4~5 人、按 4 人做（138/34 = 4.06）。"""

DEFAULT_FLOOR_AREA_RATIO_PERCENT = 80
"""得房率默认值。**这是对业主说的口径**；进几何与报告数字的计算仍取区间下限
（裁决 2026-08-30：取大则尺寸偏大、业主按报告买的柜子装不进，错误方向代价不对称）。
一个是沟通口径一个是计算口径，不同不是打架。"""

DEFAULT_OWNER_AGE_YEARS = 30
"""业主年龄默认值（用户裁决 2026-08-31）。它影响的是口味的倾向，不影响任何尺寸。"""

_TASTE_BY_AREA: tuple[tuple[int, str, str], ...] = (
    (75, "收纳优先", "地方紧，先解决东西装得下、一物多用"),
    (110, "功能齐整", "面积够住，先把每个功能安顿好、动线走顺"),
    (150, "改善讲究", "地方宽裕了，待客与独处可以分开安排"),
    (10_000, "整体调性", "空间足够，可以从整体气质和材质质感上下功夫"),
)
"""装修倾向按面积分档。**面积决定的是"先要解决什么"**——不是审美偏好，审美偏好问不出来也推不出来。
每档都带一句理由，是因为这句话要摊开给业主看：让他知道我们凭什么这么定，才谈得上他要不要改。"""


@dataclass(frozen=True)
class PlanAssumptions:
    """一套从面积推出来的默认值。业主没给的那些，系统按这个做。"""

    building_area_sqm: float
    occupants_low: int
    occupants_high: int
    occupants_assumed: int
    floor_area_ratio_percent: int
    owner_age_years: int
    taste: str
    taste_reason: str


def infer_from_area(building_area_sqm: float) -> PlanAssumptions:
    """面积 → 一套默认假设。同一个面积推两次结果相同。"""
    if building_area_sqm <= 0:
        raise ValueError(f"面积得是正数：{building_area_sqm}")
    low = max(1, round(building_area_sqm / SQM_PER_OCCUPANT))
    taste, reason = next(
        (taste, reason) for upper, taste, reason in _TASTE_BY_AREA if building_area_sqm < upper
    )
    return PlanAssumptions(
        building_area_sqm=building_area_sqm,
        occupants_low=low,
        occupants_high=low + 1,
        occupants_assumed=low,
        floor_area_ratio_percent=DEFAULT_FLOOR_AREA_RATIO_PERCENT,
        owner_age_years=DEFAULT_OWNER_AGE_YEARS,
        taste=taste,
        taste_reason=reason,
    )


def assumption_messages(assumptions: PlanAssumptions) -> list[str]:
    """把假设摊开说，并给出改的入口——**一条一件事，若干条短消息**。

    **文字由系统确定，不经 LLM**（同确认清单的先例）：每个数都来自上面推出来的那套。

    **为什么是数组不是一段**（用户 2026-08-31 晚）：分条那条规矩此前只管住了模型的回复，
    系统自己写死的文案没被管住——真机上这段是一整段发出去的，五件事（按什么面积做、人数、
    得房率、装修倾向、改的入口）挤在一条里，业主看到的是一堵墙。发送侧本就按数组循环发并
    带停顿节拍，这里给它源头，**不为系统文案另开一条发送路径**。

    最后一条是**邀请不是追问**：给了就用，不给就算——裁决里"如果用户填写就填写，
    不填写就不填写"说的就是这件事，所以这段话之后不再追第二次。
    """
    area = (
        f"{assumptions.building_area_sqm:.0f}"
        if float(assumptions.building_area_sqm).is_integer()
        else f"{assumptions.building_area_sqm:.1f}"
    )
    return [
        f"再说下我这边是按什么做的：面积按你说的 {area} 平。",
        f"这么大的房子一般住 {assumptions.occupants_low}~{assumptions.occupants_high} 个人，"
        f"我按 {assumptions.occupants_assumed} 个人来安排。",
        f"得房率按常见的 {assumptions.floor_area_ratio_percent}% 算。",
        f"装修方向按「{assumptions.taste}」走——{assumptions.taste_reason}。",
        "这几条要是跟你家不一样，直接告诉我住几个人、得房率多少、小区叫什么就行；"
        "不说也没关系，我就按这套做。",
    ]
