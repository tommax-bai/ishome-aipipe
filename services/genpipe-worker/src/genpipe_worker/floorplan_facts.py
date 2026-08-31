"""几何 → **带 id 的结构化户型事实**。确定性，零模型调用。

**这批事实是空间推理背书通道的地基**（用户裁决 2026-08-30）：说明句、报告里"讲你这套户型"
的句子，都必须**声明自己引用了哪几条事实**；机检两条——引用的 id 必须在包里存在、一条都没引用
的句子直接打回。**不是禁止模型编，是编了引不到**。没有这批带 id 的事实，那条通道就是一句话。

**为什么只从几何产、不从模型产**：这批事实的定义就是"读者看图即可自验"的那种——
几间房、哪两间通着、哪间有几个采光口、哪间狭长。它们从像素里算得出来，就不该让模型来说
（同"数字不由 LLM 决定""几何不由 LLM 决定"）。模型该做的是**拿这些事实去推理**，不是产事实。

**陈述句是给模型读的，不进客户产物**：句子里的数是算出来的，模板是确定的——它是事实的可读形态，
不是给业主看的文案。业主看到的话由写作步产出，且必须引用这里的 id。
"""

from __future__ import annotations

from genpipe_worker.models import FloorplanGeometry, PlanFact, RoomOutline

NARROW_ASPECT_RATIO = 1.8
"""长宽比超过它就叫狭长。经验值、单张样本，**复看时点＝拿到第二批样本时**。"""

SMALL_SHARE_RATIO = 0.06
"""占内部自由面积不到这个数的房间叫"小"。同上，经验值。"""


def _room_span(room: RoomOutline, geometry: FloorplanGeometry) -> tuple[float, float]:
    """房间的面宽与进深（像素）。房间是若干矩形拼的，取它们并集的外接框。"""
    if not room.boxes:
        return (0.0, 0.0)
    left = min(box[0] for box in room.boxes) * geometry.frame_width_px
    right = max(box[2] for box in room.boxes) * geometry.frame_width_px
    top = min(box[1] for box in room.boxes) * geometry.frame_height_px
    bottom = max(box[3] for box in room.boxes) * geometry.frame_height_px
    return (right - left, bottom - top)


def _room_facts(geometry: FloorplanGeometry) -> list[PlanFact]:
    facts: list[PlanFact] = []
    for room in geometry.rooms:
        share = room.area_ratio
        facts.append(
            PlanFact(
                fact_id=f"plan-share-{room.name}",
                subject=room.name,
                statement=(
                    f"{room.name}占户型内部面积的 {share:.0%}"
                    + ("，是全屋里偏小的一间" if share < SMALL_SHARE_RATIO else "")
                ),
            )
        )
        width_px, depth_px = _room_span(room, geometry)
        if width_px > 0 and depth_px > 0:
            long_side, short_side = max(width_px, depth_px), min(width_px, depth_px)
            aspect = long_side / short_side
            facts.append(
                PlanFact(
                    fact_id=f"plan-shape-{room.name}",
                    subject=room.name,
                    statement=(
                        f"{room.name}的长边约是短边的 {aspect:.1f} 倍"
                        + (
                            "，是一间狭长的房间"
                            if aspect >= NARROW_ASPECT_RATIO
                            else "，形状接近方正"
                        )
                    ),
                )
            )
    return facts


def _daylight_facts(geometry: FloorplanGeometry) -> list[PlanFact]:
    """每间房外墙上有几个开口。**这是采光面的数量，不是"采光好不好"**——好不好是判断，
    留给推理那一步去说，事实这一层只给它数得出来的东西。"""
    counted: dict[str, int] = {room.name: 0 for room in geometry.rooms}
    for opening in geometry.openings:
        if not opening.is_on_outer_wall:
            continue
        for name in opening.connects:
            if name in counted:
                counted[name] += 1
    return [
        PlanFact(
            fact_id=f"plan-daylight-{name}",
            subject=name,
            statement=(f"{name}的外墙上有 {count} 处开口" if count else f"{name}的外墙上没有开口"),
        )
        for name, count in counted.items()
    ]


# **"哪两间是通的"这类事实本轮不出**（首个真实样例实测立案）：洞两侧房间的归属在开阔处不可信。
# 那一跑产出过「卫生间与小孩房直接相通」「卫生间与餐厅直接相通」两条不成立的关系，而真正的枢纽
# 客厅只连出一条。成因不是探测距离没调好，是**走廊不是房间**——LDK 那片开阔区被分给了客厅和餐厅，
# 门开在走廊上，探两侧就探到了隔壁。它与《交接文档-户型图解析与出图启动》追记九 §五-1 登记的
# "开阔处的房间边界不稳"是同一件事。
#
# 一条错的事实被句子引用，**比没有这条事实更糟**——背书通道正是为拦这个建的：机检只保证
# "引用的 id 存在"，不保证那条事实本身对。所以事实这一层的门槛只能更高，不能更低。
# **恢复的时点写死＝房间边界在开阔处稳定之后**（即那条技术债处置时）。


def _outline_facts(geometry: FloorplanGeometry) -> list[PlanFact]:
    left, top, right, bottom = geometry.plan_box
    width_px = (right - left) * geometry.frame_width_px
    depth_px = (bottom - top) * geometry.frame_height_px
    facts = [
        PlanFact(
            fact_id="plan-rooms",
            subject="户型",
            statement=(
                f"这套户型一共 {len(geometry.rooms)} 间："
                + "、".join(room.name for room in geometry.rooms)
            ),
        )
    ]
    if width_px > 0 and depth_px > 0:
        facts.append(
            PlanFact(
                fact_id="plan-outline",
                subject="户型",
                statement=f"整套户型的面宽与进深之比约为 {width_px / depth_px:.2f}",
            )
        )
    return facts


def derive_facts(geometry: FloorplanGeometry) -> list[PlanFact]:
    """一份几何 → 一批带 id 的事实。同一份几何产两次，逐条相同。

    id 语义命名、前缀即命名空间（`plan-`），**禁纯序号**——序号一换位置就指向别的东西，
    而这些 id 是要被句子引用的，引用关系不能因为顺序变了就错位。
    """
    facts = [
        *_outline_facts(geometry),
        *_room_facts(geometry),
        *_daylight_facts(geometry),
    ]
    seen: set[str] = set()
    unique: list[PlanFact] = []
    for fact in facts:
        if fact.fact_id in seen:
            continue
        seen.add(fact.fact_id)
        unique.append(fact)
    return unique
