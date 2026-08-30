"""朝向换算：**图面事实 → 方位**，纯确定性（用户裁决 2026-08-30 晚）。

**方位不由 LLM 决定**——同族于"数字不由 LLM 决定"。模型只报它看得见的两样事实：
指北针的 N 指向图面哪一边、每个房间的窗画在房间的哪条边上；由哪条边推出哪个方位是算术，
在这里算。

立案的证据是真跑：同一张图三次跑，模型把主卧朝向说成"西南""南""西"三个答案——它每次都在
现场心算（看房间在左下角、看指北针、再推），没有固定规则，每次推的路径都不一样。
换算搬进代码之后，同样的输入必然得出同样的答案。

**朝向取决于窗所在墙面的方位，不取决于房间在图上的位置**：样本那张图主卧在左下角，
但飘窗画在下侧墙上，所以是朝南；按"房间在左下角"推会得出"西南"，那是错的。

**换算算不出比输入更准的东西**：窗墙报错，这里就跟着错，而且错得像个确定结论——
真跑里整图勘测给没有窗的卫生间报了一面西窗，换算成"卫生间朝西"后直接催出一次误报。
所以窗墙改由近景（放大后的那一块）来定，见 :class:`~genpipe_worker.models.RoomLegend`。
"""

from __future__ import annotations

from collections.abc import Sequence

from genpipe_worker.models import PageSide, RoomLegend, RoomOrientation

PAGE_SIDES_CLOCKWISE: tuple[PageSide, ...] = ("top", "right", "bottom", "left")
"""图面四边，顺时针。"""

CARDINALS_CLOCKWISE: tuple[str, ...] = ("北", "东", "南", "西")
"""方位四向，顺时针，与 :data:`PAGE_SIDES_CLOCKWISE` 同序——换算就是把两圈对齐。"""

DEFAULT_NORTH_POINTS_TO: PageSide = "top"
"""没有指北针时的退路：制图通行约定「上北下南左西右东」（用户给的通用规则，2026-08-30）。

**有指北针以指北针为准**，两者冲突时指北针赢——约定只是没有指北针时的默认，不是判据。
写成有出处的常量而不是散在代码里：这条约定哪天要改（比如某类图源惯例不同），改这一处。
"""


def to_cardinal(wall: PageSide, north_points_to: PageSide | None = None) -> str:
    """一条图面边 → 一个方位。`north_points_to` 为 None 时退到通行约定。

    把"图面四边顺时针"与"方位四向顺时针"两圈对齐：指北针指哪条边，哪条边就是北，
    其余三边跟着转。
    """
    north_side = north_points_to or DEFAULT_NORTH_POINTS_TO
    offset = PAGE_SIDES_CLOCKWISE.index(north_side)
    return CARDINALS_CLOCKWISE[(PAGE_SIDES_CLOCKWISE.index(wall) - offset) % 4]


def to_room_orientations(
    north_points_to: PageSide | None, room_legends: Sequence[RoomLegend]
) -> list[RoomOrientation]:
    """近景读到的窗墙 + 指北针 → 逐房间朝向。没画窗的房间 `facings` 为空（是事实，不猜）。

    窗墙取自**近景**而不是整图勘测：事实在看得最清楚的地方定
    （详见 :class:`~genpipe_worker.models.RoomLegend`）。
    """
    return [
        RoomOrientation(
            room=legend.room,
            window_walls=list(legend.window_walls),
            facings=[to_cardinal(wall, north_points_to) for wall in legend.window_walls],
        )
        for legend in room_legends
    ]
