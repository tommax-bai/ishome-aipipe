"""洞口类型基线：两张真图上每个洞的类型钉死（2026-09-05）。

`test_floorplan_geometry.py` 用合成图验判据本身；这里验的是**这套判据在真图上给出的答案不变**——
改阈值、改证据的顺序、改画法依赖，任何一处动了，表就会动，动了就得说清为什么。
判据与阈值的数据出处见 `_iteration/run-2026-09-05-opening-kind/run.md`。

- 92㎡ 楼书图在仓里（`services/genpipe-worker/samples/`），勘测用 8-30 的存档，**总是跑**；
- 138㎡ 真户型图不进仓（用户桌面 `test.png`，见 render3d `_iteration/真户型-基准/README.md`），
  图不在就跳过并说明；勘测框抄自 render3d 留档的 `几何产物.json`（两个次卧、两个卫生间同名，
  房间归属会并成一间——那是勘测的事，不是本表的题目）。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from genpipe_worker.floorplan_geometry import extract_geometry
from genpipe_worker.floorplan_geometry_cli import load_survey
from genpipe_worker.models import PlanOpening, RoomRegion

REPO_ROOT = Path(__file__).resolve().parent.parent
_BROCHURE_92_PNG = REPO_ROOT / "services/genpipe-worker/samples/floorplan-brochure-92sqm-3b2l1b.png"
_BROCHURE_92_SURVEY = REPO_ROOT / "_iteration/run-2026-08-30-floorplan-features/regions-1.json"
_REAL_138_PNG = Path(
    os.environ.get("ISHOME_REAL_FLOORPLAN_138_PNG", str(Path.home() / "Desktop/test.png"))
)
_REAL_138_SHA256 = "dfa2b010f28d01d92febed78df01a35d8749f9d513a054975b9082c38a26606a"

_REAL_138_SURVEY = [
    RoomRegion(name="主卧", box=(100.0, 570.0, 348.0, 850.0)),
    RoomRegion(name="次卧", box=(175.0, 56.0, 388.0, 262.0)),
    RoomRegion(name="客厅", box=(360.0, 560.0, 640.0, 810.0)),
    RoomRegion(name="厨房", box=(390.0, 56.0, 575.0, 270.0)),
    RoomRegion(name="卫生间", box=(175.0, 265.0, 310.0, 415.0)),
    RoomRegion(name="卫生间", box=(100.0, 420.0, 240.0, 565.0)),
    RoomRegion(name="阳台", box=(360.0, 825.0, 640.0, 940.0)),
    RoomRegion(name="书房", box=(575.0, 56.0, 725.0, 270.0)),
    RoomRegion(name="次卧", box=(725.0, 475.0, 855.0, 845.0)),
]
"""render3d `_iteration/真户型-基准/几何产物.json` 里的勘测框，逐字抄来
（像素坐标，套准到图幅由几何提取做）。"""

# 每行：(类型, 依据里必须出现的词)。序号与产物 `openings` 的次序一致；目视真值见 run.md。
_BROCHURE_92_KINDS = [
    ("window", "跨洞平行线"),  # 0 卫生间西外墙小窗
    ("passage", "无门弧"),  # 1 走廊 → 餐厅，没有墙的开口
    ("unknown", "墙线错位"),  # 2 阳台右侧，两条几乎平行的墙线相隔 13px
    ("unknown", "管井"),  # 3 入户门旁的管井小门，两侧都不是房间
    ("window", "跨洞平行线"),  # 4 厨房飘窗
    ("window", "跨洞平行线"),  # 5 小孩房飘窗
    ("door", "门弧"),  # 6 小孩房门
    ("passage", "无门弧"),  # 7 餐厅 ↔ 客厅，LDK 一体
    ("entry-door", "门弧"),  # 8 入户门，门弧画在户外
    ("door", "门弧"),  # 9 次卧门
    ("unknown", "像门扇线"),  # 10 主卧门——门开在一截没投出墙线的短墙上，只剩门扇线穿过断口
    ("window", "跨洞平行线"),  # 11 次卧飘窗
    ("window", "跨洞平行线"),  # 12 主卧飘窗
    ("unknown", "墙角豁口"),  # 13 主卧飘窗与阳台交界 19px 的豁口
    ("window", "跨洞平行线"),  # 14 阳台外沿
]

_REAL_138_KINDS = [
    ("window", "跨洞平行线"),  # 0 上卫生间西外墙小窗
    ("door", "门弧"),  # 1 下卫生间门
    ("door", "门弧"),  # 2 主卧门
    ("unknown", "管井"),  # 3 入户门旁的管井小门
    ("window", "跨洞平行线"),  # 4 左上次卧北窗
    ("window", "跨洞平行线"),  # 5 厨房北窗
    ("window", "跨洞平行线"),  # 6 书房北窗
    ("door", "门弧"),  # 7 左上次卧门
    ("door", "推拉门"),  # 8 厨房推拉门
    ("window", "跨洞平行线"),  # 9 下卫生间北墙小窗
    ("entry-door", "门弧"),  # 10 入户门，门弧画在公共走廊里
    ("door", "门弧"),  # 11 右下次卧门
    ("door", "推拉门"),  # 12 客厅 → 阳台推拉门
    ("window", "跨洞平行线"),  # 13 主卧南窗
    ("window", "跨洞平行线"),  # 14 右下次卧南窗
    ("window", "跨洞平行线"),  # 15 阳台外沿
]


def _assert_table(openings: Sequence[PlanOpening], expected: list[tuple[str, str]]) -> None:
    got = [(opening.kind, opening.kind_evidence) for opening in openings]
    assert len(got) == len(expected), f"洞的个数变了：{len(got)} ≠ {len(expected)}"
    mismatches = [
        f"[{index}] 期望 {kind}（{keyword}），得到 {got_kind}：{evidence}"
        for index, ((kind, keyword), (got_kind, evidence)) in enumerate(
            zip(expected, got, strict=True)
        )
        if got_kind != kind or keyword not in evidence
    ]
    assert not mismatches, "\n".join(mismatches)


def test_brochure_92_opening_kinds_are_pinned() -> None:
    geometry = extract_geometry(
        _BROCHURE_92_PNG.read_bytes(), load_survey(_BROCHURE_92_SURVEY).rooms
    )

    _assert_table(list(geometry.openings), _BROCHURE_92_KINDS)
    assert geometry.opening_kind_coverage_ratio == pytest.approx(11 / 15, abs=1e-4)
    assert [opening.kind for opening in geometry.openings].count("entry-door") == 1


@pytest.mark.skipif(
    not _REAL_138_PNG.is_file(),
    reason=(
        f"138㎡ 真户型图不在仓里，没找到 {_REAL_138_PNG}"
        "（可用环境变量 ISHOME_REAL_FLOORPLAN_138_PNG 指过去）"
    ),
)
def test_real_138_opening_kinds_are_pinned() -> None:
    image_bytes = _REAL_138_PNG.read_bytes()
    assert hashlib.sha256(image_bytes).hexdigest() == _REAL_138_SHA256, "不是那张图"

    geometry = extract_geometry(image_bytes, _REAL_138_SURVEY)

    _assert_table(list(geometry.openings), _REAL_138_KINDS)
    assert geometry.opening_kind_coverage_ratio == pytest.approx(15 / 16, abs=1e-4)
    assert [opening.kind for opening in geometry.openings].count("entry-door") == 1
