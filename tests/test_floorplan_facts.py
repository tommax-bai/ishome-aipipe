"""户型事实红线：确定性、id 语义命名不带序号、只出站得住的那几类。"""

from __future__ import annotations

from genpipe_worker.floorplan_facts import derive_facts
from genpipe_worker.models import FloorplanGeometry, PlanOpening, RoomOutline

_GEOMETRY = FloorplanGeometry(
    frame_width_px=1000,
    frame_height_px=1000,
    plan_box=(0.2, 0.2, 0.8, 0.7),
    rooms=[
        RoomOutline(
            name="客厅", boxes=[(0.2, 0.2, 0.5, 0.7)], area_ratio=0.6, centroid=(0.35, 0.45)
        ),
        RoomOutline(
            name="阳台", boxes=[(0.5, 0.2, 0.8, 0.28)], area_ratio=0.04, centroid=(0.65, 0.24)
        ),
    ],
    openings=[
        PlanOpening(
            axis="horizontal",
            position_ratio=0.2,
            start_ratio=0.25,
            end_ratio=0.35,
            is_on_outer_wall=True,
            connects=["客厅"],
        ),
        PlanOpening(
            axis="vertical",
            position_ratio=0.5,
            start_ratio=0.22,
            end_ratio=0.26,
            is_on_outer_wall=False,
            connects=["客厅", "阳台"],
        ),
    ],
)


def _by_id(geometry: FloorplanGeometry) -> dict[str, str]:
    return {fact.fact_id: fact.statement for fact in derive_facts(geometry)}


def test_facts_are_deterministic() -> None:
    """同一份几何产两次逐条相同——这批事实是要被句子按 id 引用的，id 不能换位置。"""
    assert derive_facts(_GEOMETRY) == derive_facts(_GEOMETRY)


def test_ids_are_semantic_never_ordinal() -> None:
    # 红线：命名禁纯序号、前缀即命名空间。序号一换位置就指向别的东西，而引用关系不能因此错位
    for fact in derive_facts(_GEOMETRY):
        assert fact.fact_id.startswith("plan-"), fact.fact_id
        assert not fact.fact_id.removeprefix("plan-").split("-")[0].isdigit(), fact.fact_id


def test_counts_daylight_only_from_outer_openings() -> None:
    """采光面数的是外墙上的开口。**只给数得出来的，"采光好不好"是判断留给推理那一步。"""
    facts = _by_id(_GEOMETRY)

    assert "1 处开口" in facts["plan-daylight-客厅"]
    assert "没有开口" in facts["plan-daylight-阳台"]


def test_calls_out_the_narrow_room() -> None:
    facts = _by_id(_GEOMETRY)

    assert "狭长" in facts["plan-shape-阳台"]
    assert "接近方正" in facts["plan-shape-客厅"]


def test_does_not_emit_room_links_yet() -> None:
    """洞两侧房间的归属在开阔处不可信（真实样例产出过两条不成立的关系）。

    一条错的事实被句子引用比没有它更糟：机检只保证"引用的 id 存在"，不保证事实本身对，
    所以事实这一层的门槛只能更高。恢复时点＝房间边界在开阔处稳定之后。
    """
    assert not [fact for fact in derive_facts(_GEOMETRY) if "link" in fact.fact_id]


def test_every_room_gets_share_shape_and_daylight() -> None:
    ids = _by_id(_GEOMETRY)
    for name in ("客厅", "阳台"):
        assert f"plan-share-{name}" in ids
        assert f"plan-shape-{name}" in ids
        assert f"plan-daylight-{name}" in ids
