"""户型图几何提取：合成图上的正路径与四条响亮失败路径。

**用合成图不用样本图**，理由与解析侧的守门测试同族：样本图只有一张，且它验的是
"这套阈值在那张图上跑得通"（那件事由 `_iteration/` 下的真跑存档留档，不是单测的题目）。
单测要验的是**管线的判据本身**——墙线找不找得到、断口算不算洞、房间落不落得下、
自证不过时会不会响亮失败。这些在一张画得出来的图上验得更干净，也不依赖任何外部文件。

合成图按真图的画法画：墙是实心黑条（外墙 12px、内墙 8px），家具是细灰线（要被开运算
抹掉），门是墙上的一段空白。
"""

from __future__ import annotations

import io

import pytest
from genpipe_worker.floorplan_geometry import (
    MIN_OPENING_LENGTH_RATIO,
    FloorplanGeometryError,
    extract_geometry,
    render_geometry_overlay,
)
from genpipe_worker.models import RoomRegion
from PIL import Image, ImageDraw

_IMAGE_SIZE_PX = (600, 600)
_PLAN_LEFT_PX, _PLAN_TOP_PX, _PLAN_RIGHT_PX, _PLAN_BOTTOM_PX = 80, 80, 520, 520
_PARTITION_X_PX = 300
_DOOR_TOP_PX, _DOOR_BOTTOM_PX = 260, 330


def _blank_page() -> Image.Image:
    return Image.new("RGB", _IMAGE_SIZE_PX, (255, 255, 255))


def _two_room_plan(*, with_door: bool = True, with_furniture: bool = True) -> bytes:
    """两间房 + 一道隔墙 + 一个门洞。隔墙竖着，门洞在隔墙中段。"""
    page = _blank_page()
    pen = ImageDraw.Draw(page)
    pen.rectangle(
        [_PLAN_LEFT_PX, _PLAN_TOP_PX, _PLAN_RIGHT_PX, _PLAN_BOTTOM_PX],
        outline=(0, 0, 0),
        width=12,
    )
    if with_door:
        pen.rectangle(
            [_PARTITION_X_PX - 4, _PLAN_TOP_PX, _PARTITION_X_PX + 4, _DOOR_TOP_PX],
            fill=(0, 0, 0),
        )
        pen.rectangle(
            [_PARTITION_X_PX - 4, _DOOR_BOTTOM_PX, _PARTITION_X_PX + 4, _PLAN_BOTTOM_PX],
            fill=(0, 0, 0),
        )
    else:
        pen.rectangle(
            [_PARTITION_X_PX - 4, _PLAN_TOP_PX, _PARTITION_X_PX + 4, _PLAN_BOTTOM_PX],
            fill=(0, 0, 0),
        )
    if with_furniture:
        # 细灰线：家具、标注、尺寸线。开运算应当把它们整类抹掉，不留成墙。
        for offset in range(0, 160, 20):
            pen.line([(120, 140 + offset), (260, 140 + offset)], fill=(150, 150, 150), width=1)
        pen.rectangle([360, 140, 470, 240], outline=(150, 150, 150), width=1)
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    return buffer.getvalue()


def _regions(**overrides: tuple[float, float, float, float]) -> list[RoomRegion]:
    boxes = {
        "西屋": (0.16, 0.16, 0.48, 0.85),
        "东屋": (0.52, 0.16, 0.85, 0.85),
    }
    boxes.update(overrides)
    return [RoomRegion(name=name, box=box) for name, box in boxes.items()]


def test_two_room_plan_yields_both_rooms_and_the_door() -> None:
    geometry = extract_geometry(_two_room_plan(), _regions())

    assert sorted(room.name for room in geometry.rooms) == ["东屋", "西屋"]
    # 两间房大小相仿：隔墙在正中偏左，两边差不到两成。
    west, east = ({room.name: room for room in geometry.rooms}[name] for name in ("西屋", "东屋"))
    assert abs(west.area_ratio - east.area_ratio) < 0.2
    # 锚点落在各自房间里——房间标注要挂在这上面（追记一的三层保障之一）。
    assert west.centroid[0] < _PARTITION_X_PX / _IMAGE_SIZE_PX[0] < east.centroid[0]

    inner = [opening for opening in geometry.openings if not opening.is_on_outer_wall]
    assert len(inner) == 1, "隔墙上只开了一个门洞"
    door = inner[0]
    assert door.axis == "vertical"
    assert door.start_ratio * _IMAGE_SIZE_PX[1] == pytest.approx(_DOOR_TOP_PX, abs=12)
    assert door.end_ratio * _IMAGE_SIZE_PX[1] == pytest.approx(_DOOR_BOTTOM_PX, abs=12)


def test_furniture_lines_do_not_become_walls() -> None:
    """细线画的家具与不画家具，墙与洞必须一模一样——开运算这一步是整条管线的前提。"""
    with_furniture = extract_geometry(_two_room_plan(with_furniture=True), _regions())
    without = extract_geometry(_two_room_plan(with_furniture=False), _regions())

    assert len(with_furniture.walls) == len(without.walls)
    assert len(with_furniture.openings) == len(without.openings)


def test_solid_partition_leaves_no_inner_opening() -> None:
    """隔墙不开门时内墙上不该有洞：洞是墙上的口子，不是"两个房间挨着"。"""
    geometry = extract_geometry(_two_room_plan(with_door=False), _regions())

    assert [opening for opening in geometry.openings if not opening.is_on_outer_wall] == []


def test_room_box_off_the_plan_fails_loud() -> None:
    """框落到图幅外＝勘测报偏了（真跑里阳台就这么整条报到底线以外）。

    响亮失败，且**要说出是哪个房间**——不猜、不把它并进邻居（红线一）。
    """
    with pytest.raises(FloorplanGeometryError) as failure:
        extract_geometry(_two_room_plan(), _regions(东屋=(0.90, 0.90, 0.99, 0.99)))

    assert "东屋" in "；".join(failure.value.details)


def test_page_without_walls_fails_loud() -> None:
    """白纸、或者墙细到过不了开运算：说"找不到墙"，不给一个空结构。"""
    buffer = io.BytesIO()
    _blank_page().save(buffer, format="PNG")

    with pytest.raises(FloorplanGeometryError) as failure:
        extract_geometry(buffer.getvalue(), _regions())

    assert "找不到墙" in "；".join(failure.value.details)


def test_no_regions_fails_loud() -> None:
    """墙能定出来，但没有谁给房间起名——这不是"出一份没名字的几何"，是缺输入。"""
    with pytest.raises(FloorplanGeometryError) as failure:
        extract_geometry(_two_room_plan(), [])

    assert "房间框" in "；".join(failure.value.details)


def test_junction_notches_are_not_openings() -> None:
    """洞长下限：短于图幅长边 3% 的断口一律不算洞（那是墙交叉处的豁口）。

    这条门槛之所以能写成相对值，是因为最窄的门也有 700mm、住宅图幅长边到不了 23m——
    测试盯住的就是这个不变量，而不是某张图上的某个像素数。
    """
    geometry = extract_geometry(_two_room_plan(), _regions())
    floor_px = MIN_OPENING_LENGTH_RATIO * max(_IMAGE_SIZE_PX)

    for opening in geometry.openings:
        along_px = max(_IMAGE_SIZE_PX)
        assert (opening.end_ratio - opening.start_ratio) * along_px >= floor_px


def test_overlay_renders_at_source_size() -> None:
    """核验叠图是自证材料：与原图同尺寸才叠得上，验收判据全靠肉眼看这一张。"""
    image_bytes = _two_room_plan()
    geometry = extract_geometry(image_bytes, _regions())

    with Image.open(io.BytesIO(render_geometry_overlay(image_bytes, geometry))) as overlay:
        assert overlay.size == _IMAGE_SIZE_PX


def test_geometry_carries_the_frame_it_is_normalised_against() -> None:
    """光有比例画不出形状：x 按图宽归一、y 按图高归一，两个方向除的不是同一个数。

    不给参照系，一张长方形的户型会被消费方画成正方形——所以参照系与比例同进同出。
    这**不违反"产物里不许出现绝对尺寸"**（下一条）：那条禁的是没有比例尺就冒出来的真实
    世界尺寸（毫米、米），而参照系是像素、是坐标系本身，两回事。
    """
    geometry = extract_geometry(_two_room_plan(), _regions())

    assert (geometry.frame_width_px, geometry.frame_height_px) == _IMAGE_SIZE_PX
    # 消费方按参照系还原形状：图幅的宽高比得跟原图上量出来的一致
    left, top, right, bottom = geometry.plan_box
    aspect = ((right - left) * geometry.frame_width_px) / (
        (bottom - top) * geometry.frame_height_px
    )
    assert aspect == pytest.approx(1.0, abs=0.35), "两室样例的图幅接近方形"


def test_geometry_carries_no_absolute_size() -> None:
    """产物里不许出现绝对尺寸：坐标全在 0~1，比例尺是报告那条线的事，出图不需要。

    参照系（`frame_*_px`）不在此列——它是像素、是这套比例的分母，不是真实世界的尺寸。
    """
    geometry = extract_geometry(_two_room_plan(), _regions())

    values = [
        *geometry.plan_box,
        *(wall.position_ratio for wall in geometry.walls),
        *(wall.thickness_ratio for wall in geometry.walls),
        *(opening.position_ratio for opening in geometry.openings),
        *(value for room in geometry.rooms for box in room.boxes for value in box),
    ]
    assert all(0.0 <= value <= 1.0 for value in values)
