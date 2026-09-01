"""几何提取：一张户型图 → 轮廓、墙线、门窗洞、房间遮罩。**全程不调模型**。

**这是"先出图"的第一道门**（用户裁决 2026-08-30 晚）。母版要的是坐标，而实测已经证明
**让模型直接报坐标，精度到不了画图的要求**——分区读那一轮里，整图勘测把阳台的框整体报低
约半个身位，代码照着裁出来是一张空白页（交接文档追记六 §三）。所以这里的分工是：
**模型只说哪块是哪个房间（它擅长的），墙在哪儿、洞在哪儿、边界在哪儿全部由代码从像素里算**。
同"数字不由 LLM 决定""方位不由 LLM 决定"，这条是**几何不由 LLM 决定**——
而且它由 import-linter 执行：本模块禁止依赖 `llm_client` 与 httpx，算得对不对不必靠人相信。

管线六步，每步都是确定性的：

1. **墙体掩膜**：暗于阈值的像素二值化，再做一次开运算。户型图的墙画成实心黑条（本样本
   内墙约 7px、外墙约 11px），家具与标注是细线（1~2px）与字形笔画——开运算按线宽把后者
   整类抹掉，留下的就是墙。这一步是整条管线成立的原因，也是它对"画法"的唯一依赖。
2. **图幅定位**：连通域里长边不足整图 8% 的一律丢（楼书页上的大字标题就是这么丢掉的），
   剩下的并集即户型图在页面上的位置。**不依赖模型给的框**。
3. **墙线**：逐行/逐列取宽度不超过墙厚上限的暗条，把它的中心投票给一个坐标；票数够的
   聚成一条墙线。用"窄条投票"而不是投影求和，是因为**投影会被 L 形转角带偏**——
   转角处横墙竖墙连成一片，质心落在拐点上，那不是任何一条墙的中心线。
4. **户型轮廓**：把每条墙线上的缺口补齐得到一张"封死"的掩膜，从图边向内漫灌，灌不到的
   就是户型内部。补缺口是必需的：飘窗在图上画成细线，开运算之后外墙在飘窗处是断的，
   不补就会从那里漏到户外，把页面空白也算成房间。**封死掩膜只用于定轮廓**，
   分房间时用的仍是原掩膜——门与过口必须留着开。
5. **房间**：墙线织成网格，格子按"里面有没有墙"筛成自由格；相邻两格之间那条线上墙覆盖
   不足即视为通（门、过口、开放式连通）。再以模型给的房间框为种子长开，
   **穿墙洞的代价按洞的窄度加权**——房间之间被墙隔开，门只是小口子，
   不加权时一间卫生间能顺着走廊把半个户型认领走（首轮实测如此）。
6. **自证**：房间格拼起来占户型内部自由面积的比例。对不上即边界提取有问题，
   **响亮失败**（红线一：宁可说不出，不把没把握的结构往下游传）。

**产物没有任何绝对尺寸**。比例标定服务的是报告里的数字，出图只要相对关系对
（交接文档追记七 §八-3）；洞宽如实给出去，是留给门洞反标定那一级标定物用的输入。

**画法归 render2d**：本模块只出坐标。这里唯一画的东西是
:func:`render_geometry_overlay` 的核验叠图——它是解析件的自证材料（"提取出来的东西
和原图叠不叠得上"，验收判据本身），不是产物；母版是 `plan-2d-render` 的产物。

**常量全部是单张样本实测值**（那张 92㎡ 楼书级矢量渲染图，1080×1466）。跨图与脏图没有数据
（技术债"只测过一张图"，处置时点＝拿到第二批样本）。它们按图的**相对尺度**取值而不是写死
像素，但相对尺度本身也只在一张图上验过。
"""

from __future__ import annotations

import heapq
import io
from collections import deque
from collections.abc import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from genpipe_worker.models import (
    FloorplanGeometry,
    PlanAxis,
    PlanOpening,
    PlanWall,
    PlanWallBand,
    RoomOutline,
    RoomRegion,
)

WALL_DARKNESS_MAX = 90
"""墙体二值化阈值（0~255 灰度）：暗于此即墙体候选。楼书图的墙是纯黑，家具线是浅灰。"""

WALL_OPENING_KERNEL_PX = 5
"""开运算核宽：细于此的笔画被整类抹掉。取 5 是因为本样本的最细墙约 7px、字形笔画 ≤ 4px。"""

MIN_WALL_COMPONENT_LONG_SIDE_RATIO = 0.08
"""连通域长边下限（占整图长边）：短于此不算墙。楼书页上的"92"两个字就是这么丢掉的。"""

MAX_WALL_THICKNESS_RATIO = 0.05
"""墙厚上限（占**图幅**长边，不是整页长边）：宽于此的暗条不是墙的横截面，是顺着墙走的那一段。

按图幅取而不是按页面取：墙厚随图的画幅缩放，与页面留多少白边无关。同一套户型印在
半版和整版上，页面长边差一倍，墙厚占图幅的比例却不变。取 0.05 是宽松的——十一米开间的
户型上相当于 550mm，比任何一堵墙都厚；再厚就只能是顺着墙切出来的那一长条了。
"""

MIN_WALL_LINE_VOTES = 16
"""一条墙线的票数下限：投它的行（列）少于此即噪声。约当图上 16px 长的一段墙。"""

WALL_LINE_MERGE_GAP_PX = 2
"""投票聚类的允许间断：中心线因抗锯齿会摊在相邻一两个像素上。"""

EDGE_PROBE_HALF_WIDTH_PX = 6
"""判断线上某点有没有墙时，向两侧探的半宽：墙线中心与实际墙体有一两像素偏差。"""

MIN_WALL_LINE_SPACING_PX = 2 * EDGE_PROBE_HALF_WIDTH_PX
"""两条墙线的最小间距：更近的合成一条。

跟着探测半宽走而不是另取一个数：探针判"这儿有没有墙"时向两侧各探半宽，两条线比这还近，
探针本来就分不开它们，留着两条只会让下游以为分得开——一处墙报两遍、一个洞报两遍，
中间还夹出一条比格子下限还窄的缝，那条缝会作为"不属于任何房间的地方"从图幅一头贯到另一头。
"""

MIN_CELL_SIDE_PX = 10
"""网格格子的最小边长：更窄的是墙带本身，不是可站人的地方。"""

CELL_INSET_PX = 5
"""判断格子空不空时从四边缩进的量：不缩进会把边界上的墙算进格子内部。"""

MAX_CELL_WALL_RATIO = 0.25
"""格子内部允许的墙占比：超过即这格是墙不是屋。"""

MIN_CELL_INSIDE_RATIO = 0.5
"""格子落在户型轮廓内的比例下限：低于此即页面空白，不是屋（图幅框是矩形，户型不是）。"""

MAX_EDGE_WALL_RATIO_FOR_PASSAGE = 0.75
"""两格之间那条线上的墙覆盖率上限：低于此即通（门、过口、开放式连通）。"""

EDGE_PROBE_MARGIN_PX = 4
"""探测时从线段两端让开的量：转角处两条墙交叠，不让开会把每条线的端点都算成有墙。"""

MIN_ROOM_CELL_OVERLAP_RATIO = 0.5
"""一个格子被认作某房间种子所需的重合比例：模型给的框是粗的，过半才算数。"""

MIN_CELL_COVERAGE_RATIO = 0.80
"""自证门槛：房间格拼起来占户型内部自由面积不足此比例即判提取失败（红线一）。"""

MIN_OPENING_LENGTH_RATIO = 0.03
"""洞长下限（占**图幅**长边）：短于此的断口是墙交叉处的豁口，不是门也不是窗。

这条门槛能用相对值写死，是因为**它两头都够得着**：最窄的门也有 700mm，而住宅户型图的
长边到不了 23m——700mm 在任何一张住宅户型图上都不止长边的 3%。所以 3% 之下的断口
不可能是真洞，与比例尺是多少无关（本模块不做标定，也不需要）。
"""

MIN_GRID_LINES = 2
"""每个方向的墙线条数下限：**围出一间屋的最少条数**，每个方向两条。

取 2 不是留余地，是这条判据的下界本来就在这儿：少于两条，连一间封闭的屋都围不出来，
说明这张图上根本没读出墙网。多于两条要几条不该由这里定——一室一厅与四室两厅的墙线条数
差着一倍，把门槛抬上去等于按户型大小挑图。
"""

_OVERLAY_ROOM_COLORS = (
    (0, 122, 204),
    (0, 153, 102),
    (204, 102, 0),
    (153, 51, 153),
    (0, 153, 153),
    (204, 51, 51),
    (102, 102, 0),
    (51, 102, 204),
    (153, 102, 51),
)
"""核验叠图的房间配色。只用于自证材料，不是产品配色（那归模板库）。"""

_CJK_FONT_CANDIDATES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)
"""核验叠图上写房间名要中文字库；找不到就不写名字（叠图照出，颜色仍能对照）。"""


class FloorplanGeometryError(Exception):
    """几何提取失败——响亮失败，说清缺什么，不给"差不多的"结构（红线一）。"""

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


Bitmap = list[list[bool]]


class _Grid:
    """一次提取的中间状态：掩膜、图幅、墙线、轮廓。只在本模块内流转，不下发。"""

    def __init__(
        self,
        wall_mask: Bitmap,
        width_px: int,
        height_px: int,
        plan_left_px: int,
        plan_top_px: int,
        plan_right_px: int,
        plan_bottom_px: int,
    ) -> None:
        self.wall_mask = wall_mask
        self.width_px = width_px
        self.height_px = height_px
        self.left_px = plan_left_px
        self.top_px = plan_top_px
        self.right_px = plan_right_px
        self.bottom_px = plan_bottom_px
        self.vertical_lines_px: list[int] = []
        self.horizontal_lines_px: list[int] = []
        self.line_thickness_px: dict[tuple[str, int], int] = {}
        self.is_inside: Bitmap = []
        self.plan_long_side_px = float(
            max(plan_right_px - plan_left_px, plan_bottom_px - plan_top_px)
        )
        """图幅长边。墙厚上限与洞长下限都按它取——两者都是图上的尺度，与页面留白无关。"""
        self.parallel_wall: dict[str, Bitmap] = {}
        """按轴向分开的墙体图：`parallel_wall["vertical"]` 里为真的像素属于一条**竖**墙。

        分轴是必需的。判断"这条线上这一点有没有墙"时若直接问原掩膜，**横穿过去的那条墙
        也会答有**——于是每条线都显得贯穿整个图幅，线两端之间就多出一堆本不存在的断口
        （首轮 51 个洞，大半是这么来的）。同向与否用截面宽度分：顺着一条竖墙横切，
        截面就是墙厚；横切一条横墙，截到的是它的长度。
        """


# ---------------------------------------------------------------------------
# 一、墙体掩膜与图幅定位
# ---------------------------------------------------------------------------


def _to_wall_mask(image_bytes: bytes) -> tuple[Bitmap, int, int]:
    """二值化 + 开运算：留下墙，抹掉家具线、尺寸线与字形笔画。"""
    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = image.convert("L")
        binary = gray.point(lambda value: 255 if value < WALL_DARKNESS_MAX else 0)
        opened = binary.filter(ImageFilter.MinFilter(WALL_OPENING_KERNEL_PX)).filter(
            ImageFilter.MaxFilter(WALL_OPENING_KERNEL_PX)
        )
        width_px, height_px = opened.size
        # 取整幅原始字节而不是逐点 getpixel：单通道下每字节即一个像素，
        # 一次拷贝换掉一百五十万次调用。
        raw = opened.tobytes()
    mask = [[raw[y * width_px + x] > 127 for x in range(width_px)] for y in range(height_px)]
    return mask, width_px, height_px


def _components(mask: Bitmap, width_px: int, height_px: int) -> list[list[tuple[int, int]]]:
    """四连通连通域。只走墙体像素，故与整图面积无关、与墙体总量有关。"""
    visited = [[False] * width_px for _ in range(height_px)]
    found: list[list[tuple[int, int]]] = []
    for start_y in range(height_px):
        for start_x in range(width_px):
            if not mask[start_y][start_x] or visited[start_y][start_x]:
                continue
            queue = deque([(start_x, start_y)])
            visited[start_y][start_x] = True
            component: list[tuple[int, int]] = []
            while queue:
                x, y = queue.popleft()
                component.append((x, y))
                for step_x, step_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_x, next_y = x + step_x, y + step_y
                    if (
                        0 <= next_x < width_px
                        and 0 <= next_y < height_px
                        and mask[next_y][next_x]
                        and not visited[next_y][next_x]
                    ):
                        visited[next_y][next_x] = True
                        queue.append((next_x, next_y))
            found.append(component)
    return found


def _locate_plan(mask: Bitmap, width_px: int, height_px: int) -> _Grid:
    """丢掉够不上墙的连通域，其余的并集即图幅；图幅外的墙体像素一并抹掉。

    **不依赖模型给的框**：楼书页上还有标题、卖点文案、区位图，它们要么细、要么短，
    都过不了长边这一关。
    """
    long_side_min_px = MIN_WALL_COMPONENT_LONG_SIDE_RATIO * max(width_px, height_px)
    kept: list[list[tuple[int, int]]] = []
    for component in _components(mask, width_px, height_px):
        xs = [x for x, _ in component]
        ys = [y for _, y in component]
        if max(max(xs) - min(xs), max(ys) - min(ys)) >= long_side_min_px:
            kept.append(component)
        else:
            for x, y in component:
                mask[y][x] = False
    if not kept:
        raise FloorplanGeometryError(
            ["这张图上找不到墙：没有一条够长的实心暗色线条——不是户型图，或者墙画得太细"]
        )
    left_px = min(x for component in kept for x, _ in component)
    top_px = min(y for component in kept for _, y in component)
    right_px = max(x for component in kept for x, _ in component)
    bottom_px = max(y for component in kept for _, y in component)
    return _Grid(mask, width_px, height_px, left_px, top_px, right_px, bottom_px)


# ---------------------------------------------------------------------------
# 二、墙线
# ---------------------------------------------------------------------------


def _dark_runs(grid: _Grid, axis: PlanAxis, along: int) -> Iterable[tuple[int, int]]:
    """沿一行（竖墙）或一列（横墙）扫出暗条的起讫。"""
    mask = grid.wall_mask
    start, end = (
        (grid.left_px, grid.right_px) if axis == "vertical" else (grid.top_px, grid.bottom_px)
    )
    at = start
    while at <= end:
        hit = mask[along][at] if axis == "vertical" else mask[at][along]
        if not hit:
            at += 1
            continue
        run_start = at
        while at <= end and (mask[along][at] if axis == "vertical" else mask[at][along]):
            at += 1
        yield run_start, at - 1


def _build_parallel_wall_mask(grid: _Grid, axis: PlanAxis) -> Bitmap:
    """标出属于**同向**墙的像素：截面不宽于墙厚上限的那些暗条。"""
    thickness_max_px = MAX_WALL_THICKNESS_RATIO * grid.plan_long_side_px
    parallel: Bitmap = [[False] * grid.width_px for _ in range(grid.height_px)]
    scan = (
        range(grid.top_px, grid.bottom_px + 1)
        if axis == "vertical"
        else range(grid.left_px, grid.right_px + 1)
    )
    for along in scan:
        for run_start, run_end in _dark_runs(grid, axis, along):
            if run_end - run_start + 1 > thickness_max_px:
                continue
            for across in range(run_start, run_end + 1):
                if axis == "vertical":
                    parallel[along][across] = True
                else:
                    parallel[across][along] = True
    return parallel


def _vote_wall_lines(grid: _Grid, axis: PlanAxis) -> tuple[list[int], dict[int, int]]:
    """窄条投票：每条不超过墙厚上限的暗条，把中心投给一个坐标。

    返回墙线坐标与每条线的厚度（取投它那些暗条宽度的中位数——外墙比内墙厚，母版要照画）。
    """
    thickness_max_px = MAX_WALL_THICKNESS_RATIO * grid.plan_long_side_px
    votes: dict[int, int] = {}
    widths: dict[int, list[int]] = {}
    scan = (
        range(grid.top_px, grid.bottom_px + 1)
        if axis == "vertical"
        else range(grid.left_px, grid.right_px + 1)
    )
    for along in scan:
        for run_start, run_end in _dark_runs(grid, axis, along):
            width_px = run_end - run_start + 1
            if width_px > thickness_max_px:
                continue
            center = (run_start + run_end) // 2
            votes[center] = votes.get(center, 0) + 1
            widths.setdefault(center, []).append(width_px)

    lines: list[int] = []
    thickness_px: dict[int, int] = {}
    positions = sorted(votes)
    index = 0
    while index < len(positions):
        last = index
        while (
            last + 1 < len(positions)
            and positions[last + 1] - positions[last] <= WALL_LINE_MERGE_GAP_PX
        ):
            last += 1
        cluster = positions[index : last + 1]
        total_votes = sum(votes[position] for position in cluster)
        if total_votes >= MIN_WALL_LINE_VOTES:
            center = round(sum(position * votes[position] for position in cluster) / total_votes)
            cluster_widths = sorted(w for position in cluster for w in widths[position])
            lines.append(center)
            thickness_px[center] = cluster_widths[len(cluster_widths) // 2]
        index = last + 1
    return lines, thickness_px


def _merge_close_lines(lines: Sequence[int], thickness_px: dict[int, int]) -> list[int]:
    """挨得比探测半宽还近的墙线并成一条，位置取中点。

    不并会出双份：两条线各自报一遍同一处墙、同一个洞（首轮 287 与 292、584 与 590 皆如此）。
    并的门槛跟着探测半宽走而不是另取一个数——比半宽还近的两条线，探针本来就分不开它们，
    留着两条只是让下游以为分得开。
    """
    ordered = sorted(set(lines))
    if not ordered:
        return []
    merged: list[int] = []
    cluster = [ordered[0]]
    for position in ordered[1:]:
        if position - cluster[-1] < MIN_WALL_LINE_SPACING_PX:
            cluster.append(position)
            continue
        merged.append(_collapse(cluster, thickness_px))
        cluster = [position]
    merged.append(_collapse(cluster, thickness_px))
    return merged


def _collapse(cluster: list[int], thickness_px: dict[int, int]) -> int:
    center = (cluster[0] + cluster[-1]) // 2
    thickness_px[center] = max(thickness_px.get(position, 0) for position in cluster)
    return center


def _build_wall_lines(grid: _Grid) -> None:
    """墙线只从像素投票来——**图幅四边不作墙线**。

    图幅框是外墙的外缘，而投票已经给出外墙的中心线，把两者都当线用会在每道外墙上
    多出一条几乎重合的线，墙与洞跟着出双份。图幅框留作产物里的 `plan_box`，不进网格。
    """
    vertical, vertical_thickness = _vote_wall_lines(grid, "vertical")
    horizontal, horizontal_thickness = _vote_wall_lines(grid, "horizontal")
    grid.vertical_lines_px = _merge_close_lines(vertical, vertical_thickness)
    grid.horizontal_lines_px = _merge_close_lines(horizontal, horizontal_thickness)
    for position, thickness in vertical_thickness.items():
        grid.line_thickness_px[("vertical", position)] = thickness
    for position, thickness in horizontal_thickness.items():
        grid.line_thickness_px[("horizontal", position)] = thickness
    if (
        len(grid.vertical_lines_px) < MIN_GRID_LINES
        or len(grid.horizontal_lines_px) < MIN_GRID_LINES
    ):
        raise FloorplanGeometryError(
            [
                f"墙网读不出来：竖墙 {len(grid.vertical_lines_px)} 条、"
                f"横墙 {len(grid.horizontal_lines_px)} 条，各自至少要 {MIN_GRID_LINES} 条"
            ]
        )


# ---------------------------------------------------------------------------
# 三、户型轮廓
# ---------------------------------------------------------------------------


def _seal_line_gaps(grid: _Grid) -> Bitmap:
    """把每条墙线上的缺口补齐，得到一张只用来定轮廓的封死掩膜。

    补的是**这条线上最早与最晚那两处墙体之间**的所有位置——门、窗、飘窗留下的断口
    因此一并补上。飘窗尤其要补：它在图上画成细线，开运算之后外墙在飘窗处是断的，
    不补就会从那里漏到户外，把页面空白也算成房间（首轮实测：小孩房一路涨到页边）。

    **补出来的墙只影响轮廓，不影响分房间**：往内部加墙不改变"从户外灌不灌得进来"，
    而分房间用的是原掩膜——门与过口必须留着开。
    """
    sealed = [row[:] for row in grid.wall_mask]
    half = EDGE_PROBE_HALF_WIDTH_PX // 2
    for position in grid.vertical_lines_px:
        walled = [
            y
            for y in range(grid.top_px, grid.bottom_px + 1)
            if _is_walled_near(grid, "vertical", position, y)
        ]
        if len(walled) < MIN_WALL_LINE_VOTES:
            continue
        for y in range(min(walled), max(walled) + 1):
            for offset in range(-half, half + 1):
                if 0 <= position + offset < grid.width_px:
                    sealed[y][position + offset] = True
    for position in grid.horizontal_lines_px:
        walled = [
            x
            for x in range(grid.left_px, grid.right_px + 1)
            if _is_walled_near(grid, "horizontal", position, x)
        ]
        if len(walled) < MIN_WALL_LINE_VOTES:
            continue
        for x in range(min(walled), max(walled) + 1):
            for offset in range(-half, half + 1):
                if 0 <= position + offset < grid.height_px:
                    sealed[position + offset][x] = True
    return sealed


def _is_walled_near(grid: _Grid, axis: PlanAxis, position: int, along: int) -> bool:
    """墙线上某一点有没有**同向**的墙。向两侧探半宽，容忍中心线与墙体的一两像素偏差。

    问的是同向墙图不是原掩膜：横穿过去的那条墙不算这条线上的墙，否则每条线都显得
    从图幅一头贯到另一头（见 :attr:`_Grid.parallel_wall`）。
    """
    parallel = grid.parallel_wall[axis]
    for offset in range(-EDGE_PROBE_HALF_WIDTH_PX, EDGE_PROBE_HALF_WIDTH_PX + 1):
        at = position + offset
        if axis == "vertical":
            if 0 <= at < grid.width_px and parallel[along][at]:
                return True
        elif 0 <= at < grid.height_px and parallel[at][along]:
            return True
    return False


def _flood_inside(grid: _Grid, sealed: Bitmap) -> Bitmap:
    """从图边向内漫灌，灌不到的就是户型内部（含墙体本身）。"""
    outside = [[False] * grid.width_px for _ in range(grid.height_px)]
    queue: deque[tuple[int, int]] = deque()

    def push(x: int, y: int) -> None:
        if not sealed[y][x] and not outside[y][x]:
            outside[y][x] = True
            queue.append((x, y))

    for x in range(grid.width_px):
        push(x, 0)
        push(x, grid.height_px - 1)
    for y in range(grid.height_px):
        push(0, y)
        push(grid.width_px - 1, y)
    while queue:
        x, y = queue.popleft()
        for step_x, step_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_x, next_y = x + step_x, y + step_y
            if 0 <= next_x < grid.width_px and 0 <= next_y < grid.height_px:
                push(next_x, next_y)
    return [[not outside[y][x] for x in range(grid.width_px)] for y in range(grid.height_px)]


MIN_OUTLINE_RUN_PX = 3
"""外轮廓上一段最短多少像素才算数：再短就是掩膜边缘的锯齿，不是一段墙。"""

_MIN_OUTLINE_THICKNESS_PX = 4
"""一段墙像素都量不到、且全图也没有可借的中位数时，外轮廓按这个厚度画（兜底的兜底）。"""

OUTLINE_SKIN_PX = 3
"""量墙厚前允许跨过的"皮"：`is_inside` 的边界是封缝之后的结果，可能比真墙外沿再往外一两像素。"""


def _outline_runs(is_edge: Sequence[bool]) -> list[tuple[int, int]]:
    """沿一条线扫出连续的边界段。短到只剩锯齿的丢掉。"""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for at, edge in enumerate(is_edge):
        if edge:
            start = at if start is None else start
        elif start is not None:
            if at - start >= MIN_OUTLINE_RUN_PX:
                runs.append((start, at - 1))
            start = None
    if start is not None and len(is_edge) - start >= MIN_OUTLINE_RUN_PX:
        runs.append((start, len(is_edge) - 1))
    return runs


def _band_depth_px(grid: _Grid, axis: PlanAxis, position: int, along: int, step: int) -> int:
    """从边界往里走，墙有多厚就走多远。

    **问的是同向墙不是原掩膜**：一条横墙的上沿往下走，若问原掩膜，会顺着与它相交的那条竖墙
    一路走到底——首版就量出过一段"厚 242px 的外墙"。同一个坑《追记九》在判洞时踩过一次
    （"横穿过去的那条墙也答有"），grid 为此备了按轴向分开的墙体图，这里照用。

    边界是 `is_inside` 的边界、也就是墙的外沿，而封缝那一步可能让它比真墙外沿再往外一点点；
    因此**允许先跨过几像素的皮**再开始数，跨不过去才算这一点上没墙。
    """
    band = grid.parallel_wall[axis]
    depth = 0
    at = position
    limit = grid.width_px if axis == "vertical" else grid.height_px
    skin = OUTLINE_SKIN_PX
    while 0 <= at < limit:
        hit = band[along][at] if axis == "vertical" else band[at][along]
        if not hit:
            if depth > 0 or skin <= 0:
                break
            skin -= 1
        else:
            depth += 1
        at += step
    return depth


def _measure_depth_px(
    grid: _Grid, axis: PlanAxis, position: int, run: tuple[int, int], step: int
) -> int:
    """这一段边界上墙有多厚（取沿线若干点的中位数，躲开单点噪声）。0 = 这一段上没有墙像素。"""
    start, end = run
    depths = sorted(
        _band_depth_px(grid, axis, position, along, step)
        for along in range(start, end + 1, max(1, (end - start) // 8 or 1))
    )
    return depths[len(depths) // 2]


def _outline_wall(
    grid: _Grid, axis: PlanAxis, position: int, run: tuple[int, int], step: int, depth: int
) -> PlanWall:
    """一段边界 → 一段外墙。位置取墙带的中心线，与 `walls` 同一口径（都是墙心）。"""
    start, end = run
    across = float(grid.width_px if axis == "vertical" else grid.height_px)
    along_px = float(grid.height_px if axis == "vertical" else grid.width_px)
    return PlanWall(
        axis=axis,
        position_ratio=(position + step * (depth - 1) / 2) / across,
        start_ratio=start / along_px,
        end_ratio=end / along_px,
        thickness_ratio=depth / across,
    )


def _trace_outline(grid: _Grid) -> list[PlanWall]:
    """户型外轮廓，按外墙的**中心线**给出，与 :func:`_build_wall_lines` 同一口径。

    **为什么 `walls` 不够**：那是网格投票出来的线，投不上票的外墙不在里面——飘窗那种墙往外
    折一个台阶的段，整段会被读成"洞"，台阶本身那截短墙又短到投不出线。首个真实样例
    92㎡ 九个飘窗，**外轮廓只剩 64% 有墙**，母版画出来外圈是漏的。这个洞不是"再调调阈值"
    能补的：轴对齐的线模型表达不了台阶，只能另给一条来路。

    来路就是 `is_inside`——它是从图边向内漫灌灌不到的地方，边界正是外墙的外沿，
    **本来就是从像素里算出来的**（"墙在哪儿全部从像素里算"这条没有松动）。
    沿边界扫出连续段、往里量墙带有多厚，得到的就是外墙。

    与 `walls` 重合的那些段照出不去重：两边都是墙、画出来是同一笔黑；**去重要判"这两段是不是
    同一道墙"，那是又一个会错的判断**，而重复画一遍没有任何代价。
    """
    inside = grid.is_inside
    found: list[tuple[PlanAxis, int, tuple[int, int], int, int]] = []
    for x in range(grid.width_px):
        for step in (-1, 1):
            neighbour = x - step  # step=+1 时边界在左侧，往右量厚度；step=-1 反之
            outside_here = not 0 <= neighbour < grid.width_px
            edges = [
                inside[y][x] and (outside_here or not inside[y][neighbour])
                for y in range(grid.height_px)
            ]
            for run in _outline_runs(edges):
                found.append(
                    ("vertical", x, run, step, _measure_depth_px(grid, "vertical", x, run, step))
                )
    for y in range(grid.height_px):
        row = inside[y]
        for step in (-1, 1):
            other = y - step
            neighbour_row = inside[other] if 0 <= other < grid.height_px else None
            edges = [
                row[x] and (neighbour_row is None or not neighbour_row[x])
                for x in range(grid.width_px)
            ]
            for run in _outline_runs(edges):
                found.append(
                    (
                        "horizontal",
                        y,
                        run,
                        step,
                        _measure_depth_px(grid, "horizontal", y, run, step),
                    )
                )

    # 量不到墙的那些段照出，厚度借用其他外墙的中位数——**边界在那儿是事实，墙像素不在是画法**：
    # 飘窗在楼书图上画的是两条细窗线，去家具线那一步的开运算把它们连同尺寸线一起抹了，
    # 于是那几条边一个墙像素都不剩。首个真实样例四个飘窗，外轮廓因此缺了整整四条边。
    # 丢掉它们等于把户型画成漏风的，而它们是不是"墙"这件事，洞的清单已经如实标着了。
    measured = sorted(depth for *_, depth in found if depth > 0)
    fallback = measured[len(measured) // 2] if measured else _MIN_OUTLINE_THICKNESS_PX
    return [
        _outline_wall(grid, axis, position, run, step, depth or fallback)
        for axis, position, run, step, depth in found
    ]


# ---------------------------------------------------------------------------
# 四、房间
# ---------------------------------------------------------------------------


def _rect_ratio(mask: Bitmap, left: int, top: int, right: int, bottom: int) -> float:
    """一个矩形里 True 的占比。"""
    total = 0
    hits = 0
    for y in range(top, bottom + 1):
        row = mask[y]
        for x in range(left, right + 1):
            total += 1
            hits += row[x]
    return hits / max(total, 1)


def _free_cells(grid: _Grid) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """墙线织成网格，筛出"里面没墙、且落在户型内"的格子。"""
    cells: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for column in range(len(grid.vertical_lines_px) - 1):
        for row in range(len(grid.horizontal_lines_px) - 1):
            left = grid.vertical_lines_px[column]
            right = grid.vertical_lines_px[column + 1]
            top = grid.horizontal_lines_px[row]
            bottom = grid.horizontal_lines_px[row + 1]
            if right - left < MIN_CELL_SIDE_PX or bottom - top < MIN_CELL_SIDE_PX:
                continue
            inner = (
                left + CELL_INSET_PX,
                top + CELL_INSET_PX,
                right - CELL_INSET_PX,
                bottom - CELL_INSET_PX,
            )
            if _rect_ratio(grid.wall_mask, *inner) >= MAX_CELL_WALL_RATIO:
                continue
            if _rect_ratio(grid.is_inside, *inner) < MIN_CELL_INSIDE_RATIO:
                continue
            cells[(column, row)] = (left, top, right, bottom)
    return cells


def _edge_wall_ratio(grid: _Grid, axis: PlanAxis, position: int, start: int, end: int) -> float:
    """两格之间那条线上，有墙的比例。1.0 ＝ 一堵完整的墙，0.0 ＝ 完全通开。"""
    covered = 0
    sampled = 0
    for along in range(start + EDGE_PROBE_MARGIN_PX, end - EDGE_PROBE_MARGIN_PX + 1):
        sampled += 1
        if _is_walled_near(grid, axis, position, along):
            covered += 1
    return covered / max(sampled, 1)


def _cell_passages(
    grid: _Grid, cells: dict[tuple[int, int], tuple[int, int, int, int]]
) -> dict[tuple[int, int], list[tuple[tuple[int, int], float]]]:
    """相邻格之间通不通，以及那条边有多少是墙（越接近全墙，穿过去越贵）。"""
    passages: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {
        cell: [] for cell in cells
    }
    for (column, row), (left, top, right, bottom) in cells.items():
        right_neighbour = (column + 1, row)
        if right_neighbour in cells:
            ratio = _edge_wall_ratio(
                grid, "vertical", grid.vertical_lines_px[column + 1], top, bottom
            )
            if ratio < MAX_EDGE_WALL_RATIO_FOR_PASSAGE:
                passages[(column, row)].append((right_neighbour, ratio))
                passages[right_neighbour].append(((column, row), ratio))
        below = (column, row + 1)
        if below in cells:
            ratio = _edge_wall_ratio(
                grid, "horizontal", grid.horizontal_lines_px[row + 1], left, right
            )
            if ratio < MAX_EDGE_WALL_RATIO_FOR_PASSAGE:
                passages[(column, row)].append((below, ratio))
                passages[below].append(((column, row), ratio))
    return passages


def _register_regions(grid: _Grid, regions: Sequence[RoomRegion]) -> list[RoomRegion]:
    """把勘测给的房间框整体套准到算出来的图幅上：**相对位置归模型，绝对定位归代码**。

    做法是一次轴向线性映射——所有房间框的并集就是这套户型（每个房间都在户型里，
    户型也正是这些房间拼起来的），所以那个并集应当与图幅重合，不重合的部分是模型
    整体估偏了。把并集拉到图幅上，偏移就没了。

    真跑证据（同一张图，三次勘测存档）：三次的框并集横向都只差千分之四，纵向下缘一次
    差了 0.041——**恰好是把阳台整条报到图幅底线以外的那一次**，套准前它连一个格子都落不到，
    整份提取因此响亮失败；套准后落回阳台本身。这与分区读那一轮"勘测把阳台的框整体报低
    半个身位、裁出空白页"是同一个毛病的同一次现形（交接文档追记六 §三）。

    **成立的前提是勘测把房间报全了**：漏掉一个贴边的房间，并集就小于图幅，映射会把
    所有框往外拉。当前三次存档都报全了九个房间，没有反例；**复看时点写死＝拿到第二批
    样本时**——那批图的房间数与画法都不同，是这条假设的第一次真考。
    """
    left = min(region.box[0] for region in regions)
    top = min(region.box[1] for region in regions)
    right = max(region.box[2] for region in regions)
    bottom = max(region.box[3] for region in regions)
    if right - left <= 0 or bottom - top <= 0:
        return list(regions)
    plan_left = grid.left_px / grid.width_px
    plan_top = grid.top_px / grid.height_px
    plan_right = grid.right_px / grid.width_px
    plan_bottom = grid.bottom_px / grid.height_px
    scale_x = (plan_right - plan_left) / (right - left)
    scale_y = (plan_bottom - plan_top) / (bottom - top)

    def fit_x(value: float) -> float:
        return plan_left + (value - left) * scale_x

    def fit_y(value: float) -> float:
        return plan_top + (value - top) * scale_y

    return [
        RoomRegion(
            name=region.name,
            box=(
                fit_x(region.box[0]),
                fit_y(region.box[1]),
                fit_x(region.box[2]),
                fit_y(region.box[3]),
            ),
        )
        for region in regions
    ]


def _seed_cells(
    cells: dict[tuple[int, int], tuple[int, int, int, int]],
    regions: Sequence[RoomRegion],
    width_px: int,
    height_px: int,
) -> dict[tuple[int, int], str]:
    """模型给的房间框 → 种子格。过半重合才算数（框是粗的，不能当边界用）。"""
    claims: dict[tuple[int, int], tuple[str, float]] = {}
    for region in regions:
        box_left = region.box[0] * width_px
        box_top = region.box[1] * height_px
        box_right = region.box[2] * width_px
        box_bottom = region.box[3] * height_px
        for cell, (left, top, right, bottom) in cells.items():
            overlap_x = min(right, box_right) - max(left, box_left)
            overlap_y = min(bottom, box_bottom) - max(top, box_top)
            if overlap_x <= 0 or overlap_y <= 0:
                continue
            share = overlap_x * overlap_y / max(1.0, (right - left) * (bottom - top))
            if share < MIN_ROOM_CELL_OVERLAP_RATIO:
                continue
            claimed = claims.get(cell)
            if claimed is None or share > claimed[1]:
                claims[cell] = (region.name, share)
    return {cell: name for cell, (name, _) in claims.items()}


def _grow_rooms(
    cells: dict[tuple[int, int], tuple[int, int, int, int]],
    passages: dict[tuple[int, int], list[tuple[tuple[int, int], float]]],
    seeds: dict[tuple[int, int], str],
) -> dict[tuple[int, int], str]:
    """从种子格长开，代价 ＝ 距离 ÷ 开口的通畅度。

    除以通畅度这一步是首轮实测逼出来的：不加权时，卫生间顺着它那道门一路认领了走廊、
    主卧门口与半个客厅（8.7% 的户型面积算成卫生间）。加权之后，穿一道七成是墙的门要付
    四倍的代价，房间就停在自己的门口了。
    """
    labels: dict[tuple[int, int], str] = {}
    frontier: list[tuple[float, tuple[int, int], str]] = [
        (0.0, cell, name) for cell, name in seeds.items()
    ]
    heapq.heapify(frontier)
    while frontier:
        cost, cell, name = heapq.heappop(frontier)
        if cell in labels:
            continue
        labels[cell] = name
        left, top, right, bottom = cells[cell]
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        for neighbour, wall_ratio in passages[cell]:
            if neighbour in labels:
                continue
            other_left, other_top, other_right, other_bottom = cells[neighbour]
            step = abs(center_x - (other_left + other_right) / 2) + abs(
                center_y - (other_top + other_bottom) / 2
            )
            heapq.heappush(frontier, (cost + step / max(0.05, 1.0 - wall_ratio), neighbour, name))
    return labels


# ---------------------------------------------------------------------------
# 五、墙段与洞
# ---------------------------------------------------------------------------


_BAND_FACE_STEP_PX = 3
"""面走位多少像素才算厚度/位置真的变了。样本实测：同段内的墨宽抖动 ≤2px（矢量渲染的
11↔12、6↔7 交替），真实的段间突变 ≥5px（12→22、12→6、6→11）——取 3 落在分离带上。"""

_BAND_MIN_RUN_PX = 4
"""突变要持续多少像素才开新段。路口行剔除后残余的孤点毛刺 ≤2px（run 首尾的单行 13/14），
最短的真实段 8px（小孩房下墙 12px 那截）——取 4，两头各留一倍。"""


def _ink_faces_px(grid: _Grid, axis: PlanAxis, position: int, along: int) -> tuple[int, int] | None:
    """墙线上某一点的**实测墨带两面**：在 position 附近找同向墨带，向两侧扩到墨的边缘。

    问的是同向墙图不是原掩膜（理由同 :func:`_is_walled_near`：横穿的墙不算这条线的墨）。
    容差与探针同一把尺——中心线与墙体差一两像素是常态，比半宽还远就不是这条线的墙了。
    """
    band = grid.parallel_wall[axis]
    limit = grid.width_px if axis == "vertical" else grid.height_px

    def is_ink(at: int) -> bool:
        if not 0 <= at < limit:
            return False
        return band[along][at] if axis == "vertical" else band[at][along]

    seed = next(
        (
            position + offset
            for offset in sorted(
                range(-EDGE_PROBE_HALF_WIDTH_PX, EDGE_PROBE_HALF_WIDTH_PX + 1), key=abs
            )
            if is_ink(position + offset)
        ),
        None,
    )
    if seed is None:
        return None
    low = seed
    while is_ink(low - 1):
        low -= 1
    high = seed
    while is_ink(high + 1):
        high += 1
    return low, high


def _junction_alongs(grid: _Grid, axis: PlanAxis, start: int, end: int) -> set[int]:
    """这段墙上压在**横穿墙线的墙带里**的那些位置：那儿量出来的不是这条墙的厚度。

    短小的横穿墙段会与本墙连成一条不超过墙厚上限的墨带（h855 那截 13px 长的横墙让竖墙
    在 y 849~861 量出 24px"厚"——真厚 6px），逐点实测躲不开它，只能按结构剔除：
    横穿线的位置与厚度都是投票投出来的，压在它墙带里的行一律不采样，厚度归两侧邻段。
    """
    crossing_axis: PlanAxis = "horizontal" if axis == "vertical" else "vertical"
    crossing_lines = grid.horizontal_lines_px if axis == "vertical" else grid.vertical_lines_px
    found: set[int] = set()
    for line in crossing_lines:
        half = grid.line_thickness_px.get((crossing_axis, line), 0) / 2
        first = max(start, int(line - half))
        last = min(end, int(line + half) + 1)
        found.update(range(first, last + 1))
    return found


def _face_step_px(group: list[tuple[int, int, int]], sample: tuple[int, int, int]) -> int:
    """一个采样点相对当前段的面走位：两面各与段内中位数比，取大的那个。

    比面不比宽是刻意的：宽度不变、整条带子横着挪的"错位"也得换段——那正是两轮定罪里
    "中心错开 7~8px"的形态。中位数抗孤点毛刺，被吸收进段里的单行噪声带不偏它。
    """
    lows = sorted(low for _, low, _ in group)
    highs = sorted(high for _, _, high in group)
    _, low, high = sample
    return max(abs(low - lows[len(lows) // 2]), abs(high - highs[len(highs) // 2]))


def _measure_wall_bands(
    grid: _Grid, axis: PlanAxis, position: int, start: int, end: int
) -> list[PlanWallBand]:
    """一段墙的**按段实测厚度**：逐点量墨带两面，面走位持续超阈值就换段。

    这是"一条线只给一个厚度"的补法（2026-09-01 两轮线稿定罪的共同根因）：投票的厚度取
    整条线的中位数，厚度沿长度变的墙（次卧右外墙的 22px 墙角、玄关交界收成 6px 的那截）
    被套成一个数。这里按实测分段，段内取中位、段界取两个采样点的中点——相邻段共用边界，
    拼起来正好盖满整段墙。

    面对齐不用推断：两面各自实测，突变处哪面数值没变哪面就是没动的。全段都压在路口上
    （量不出）就不给段——如实缺，不给凑的数。
    """
    junction = _junction_alongs(grid, axis, start, end)
    samples: list[tuple[int, int, int]] = []
    for along in range(start, end + 1):
        if along in junction:
            continue
        faces = _ink_faces_px(grid, axis, position, along)
        if faces is not None:
            samples.append((along, faces[0], faces[1]))
    if not samples:
        return []

    groups: list[list[tuple[int, int, int]]] = [[samples[0]]]
    index = 1
    while index < len(samples):
        sample = samples[index]
        if _face_step_px(groups[-1], sample) < _BAND_FACE_STEP_PX:
            groups[-1].append(sample)
            index += 1
            continue
        upcoming = samples[index : index + _BAND_MIN_RUN_PX]
        if len(upcoming) == _BAND_MIN_RUN_PX and all(
            _face_step_px(groups[-1], one) >= _BAND_FACE_STEP_PX for one in upcoming
        ):
            groups.append([sample])
        else:
            groups[-1].append(sample)  # 不持续的孤点毛刺吸收进当前段，中位数不受它带偏
        index += 1

    across = float(grid.width_px if axis == "vertical" else grid.height_px)
    along_px = float(grid.height_px if axis == "vertical" else grid.width_px)
    bands: list[PlanWallBand] = []
    for at, group in enumerate(groups):
        lows = sorted(low for _, low, _ in group)
        highs = sorted(high for _, _, high in group)
        low = lows[len(lows) // 2]
        high = highs[len(highs) // 2]
        band_start = float(start) if at == 0 else (groups[at - 1][-1][0] + group[0][0]) / 2
        band_end = (
            float(end) if at == len(groups) - 1 else (group[-1][0] + groups[at + 1][0][0]) / 2
        )
        bands.append(
            PlanWallBand(
                start_ratio=band_start / along_px,
                end_ratio=band_end / along_px,
                # 墨带占的是 [low, high] 这些整像素，两面在像素格的外缘（±0.5）——
                # 这样厚度恰等于实测墨宽，与 thickness_ratio 的口径（像素数/图宽高）一致
                face_low_ratio=(low - 0.5) / across,
                face_high_ratio=(high + 0.5) / across,
            )
        )
    return bands


def _walls_and_openings(
    grid: _Grid, room_at: list[list[int]], room_names: Sequence[str]
) -> tuple[list[PlanWall], list[PlanOpening]]:
    """沿每条墙线走一遍：连着的墙体是墙段，中间的断口是**候选**洞。

    只取首尾两处墙体之间的断口（线两端之外的空白不是洞，是这条墙到头了），再过一道
    **两侧属谁**：断口两边落在同一个房间里的一律不算洞。

    这道过滤不是保险丝而是判据本身。墙线是一条贯穿图幅的直线，它免不了要横穿几个房间——
    首版没有这道过滤时，一条穿过客厅的线在客厅当中留下的那截空白也被记成了"洞"，
    51 个洞里大半是这么来的，画在叠图上就是客厅正中央凭空几道门。
    **洞的定义是"隔开两边的墙上有个口子"，那就得先知道两边是不是两个地方。**
    """
    walls: list[PlanWall] = []
    openings: list[PlanOpening] = []
    long_side_px = grid.plan_long_side_px
    for axis, positions, along_start, along_end in (
        ("vertical", grid.vertical_lines_px, grid.top_px, grid.bottom_px),
        ("horizontal", grid.horizontal_lines_px, grid.left_px, grid.right_px),
    ):
        axis_name: PlanAxis = "vertical" if axis == "vertical" else "horizontal"
        for position in positions:
            walled = [
                along
                for along in range(along_start, along_end + 1)
                if _is_walled_near(grid, axis_name, position, along)
            ]
            if len(walled) < MIN_WALL_LINE_VOTES:
                continue
            thickness_px = grid.line_thickness_px.get((axis, position), WALL_OPENING_KERNEL_PX)
            present = set(walled)
            at = walled[0]
            last = walled[-1]
            while at <= last:
                if at in present:
                    run_start = at
                    while at <= last and at in present:
                        at += 1
                    walls.append(
                        _to_wall(
                            grid,
                            axis_name,
                            position,
                            run_start,
                            at - 1,
                            thickness_px,
                            _measure_wall_bands(grid, axis_name, position, run_start, at - 1),
                        )
                    )
                else:
                    gap_start = at
                    while at <= last and at not in present:
                        at += 1
                    opening = _to_opening(
                        grid,
                        axis_name,
                        position,
                        gap_start,
                        at - 1,
                        long_side_px,
                        room_at,
                        room_names,
                    )
                    if opening is not None:
                        openings.append(opening)
    return walls, openings


def _to_wall(
    grid: _Grid,
    axis: PlanAxis,
    position: int,
    start: int,
    end: int,
    thickness_px: int,
    bands: list[PlanWallBand],
) -> PlanWall:
    across = float(grid.width_px if axis == "vertical" else grid.height_px)
    along = float(grid.height_px if axis == "vertical" else grid.width_px)
    return PlanWall(
        axis=axis,
        position_ratio=position / across,
        start_ratio=start / along,
        end_ratio=end / along,
        thickness_ratio=thickness_px / across,
        bands=bands,
    )


OUTSIDE_ROOM = -1
"""房间图上的"户外"：轮廓之外。"""

UNCLAIMED_ROOM = -2
"""房间图上的"没归着"：轮廓之内但不属于任何房间格（墙带本身、贴墙的窄条）。"""


def _room_bitmap(
    grid: _Grid,
    cells: dict[tuple[int, int], tuple[int, int, int, int]],
    labels: dict[tuple[int, int], str],
    order: Sequence[str],
) -> list[list[int]]:
    """逐像素的房间图：每个点属于哪个房间（或户外、或没归着）。判断洞的两侧要用它。"""
    index_of = {name: index for index, name in enumerate(order)}
    room_at = [
        [OUTSIDE_ROOM if not grid.is_inside[y][x] else UNCLAIMED_ROOM for x in range(grid.width_px)]
        for y in range(grid.height_px)
    ]
    for cell, name in labels.items():
        left, top, right, bottom = cells[cell]
        index = index_of[name]
        for y in range(top, bottom + 1):
            row = room_at[y]
            for x in range(left, right + 1):
                if row[x] != OUTSIDE_ROOM:
                    row[x] = index
    return room_at


def _side_of(
    grid: _Grid, room_at: list[list[int]], axis: PlanAxis, position: int, along: int, sign: int
) -> int:
    """洞的一侧属谁。由近及远探几步——紧贴洞口的那一两个像素常落在墙带里，问不出名字。"""
    limit = grid.width_px if axis == "vertical" else grid.height_px
    for step in range(EDGE_PROBE_HALF_WIDTH_PX, EDGE_PROBE_HALF_WIDTH_PX * 4, 2):
        at = position + sign * step
        if not 0 <= at < limit:
            return OUTSIDE_ROOM
        found = room_at[along][at] if axis == "vertical" else room_at[at][along]
        if found != UNCLAIMED_ROOM:
            return found
    return UNCLAIMED_ROOM


def _to_opening(
    grid: _Grid,
    axis: PlanAxis,
    position: int,
    start: int,
    end: int,
    long_side_px: float,
    room_at: list[list[int]],
    room_names: Sequence[str],
) -> PlanOpening | None:
    """一个断口是不是洞：够长，且两侧属于不同的地方。

    两道判据各挡一类假洞。**够长**挡的是墙交叉处的豁口——同向墙图在 T 字与十字路口
    必然缺一小块（那几个像素属于横穿的那条墙），首轮里五到十几像素的"洞"全是这么来的。
    **两侧不同**挡的是线横穿房间时留下的空白：次卧当中那条线两头都有墙（飘窗的两侧墙），
    中间一百四十像素既够长、也确实夹在两段墙之间，只有"两边都是次卧"能说明它不是门。
    """
    if end - start + 1 < MIN_OPENING_LENGTH_RATIO * long_side_px:
        return None
    middle = (start + end) // 2
    near = _side_of(grid, room_at, axis, position, middle, -1)
    far = _side_of(grid, room_at, axis, position, middle, 1)
    if near == far and near != OUTSIDE_ROOM:
        return None
    across = float(grid.width_px if axis == "vertical" else grid.height_px)
    along = float(grid.height_px if axis == "vertical" else grid.width_px)
    return PlanOpening(
        axis=axis,
        position_ratio=position / across,
        start_ratio=start / along,
        end_ratio=end / along,
        is_on_outer_wall=OUTSIDE_ROOM in (near, far),
        connects=[room_names[side] for side in (near, far) if 0 <= side < len(room_names)],
    )


# ---------------------------------------------------------------------------
# 六、对外入口
# ---------------------------------------------------------------------------


def extract_geometry(image_bytes: bytes, regions: Sequence[RoomRegion]) -> FloorplanGeometry:
    """一张户型图 + 模型给的房间框 → 几何产物。**全程不调模型**。

    `regions` 只做两件事：给房间**起名**、给区域生长**下种**。墙在哪儿、洞在哪儿、
    户型边界在哪儿，一律从像素里算——模型报的坐标精度到不了画图的要求（追记六 §三 实测）。
    """
    if not regions:
        raise FloorplanGeometryError(["没有房间框：几何提取能定出墙，但没有谁给房间起名"])
    mask, width_px, height_px = _to_wall_mask(image_bytes)
    grid = _locate_plan(mask, width_px, height_px)
    grid.parallel_wall = {
        "vertical": _build_parallel_wall_mask(grid, "vertical"),
        "horizontal": _build_parallel_wall_mask(grid, "horizontal"),
    }
    _build_wall_lines(grid)
    grid.is_inside = _flood_inside(grid, _seal_line_gaps(grid))

    cells = _free_cells(grid)
    if not cells:
        raise FloorplanGeometryError(["墙网里一个可站人的格子都没有：墙线读错了，不往下走"])
    seeds = _seed_cells(cells, _register_regions(grid, regions), width_px, height_px)
    unseeded = [region.name for region in regions if region.name not in set(seeds.values())]
    if unseeded:
        raise FloorplanGeometryError(
            [
                f"这几个房间在墙网里落不到地：{'、'.join(unseeded)}"
                "——模型给的框与算出来的墙对不上，可能是框偏了，也可能是这一块的墙没读出来"
            ]
        )
    labels = _grow_rooms(cells, _cell_passages(grid, cells), seeds)

    rooms = _to_room_outlines(grid, cells, labels)
    room_at = _room_bitmap(grid, cells, labels, [room.name for room in rooms])
    walls, openings = _walls_and_openings(grid, room_at, [room.name for room in rooms])
    coverage_ratio = _cell_coverage_ratio(grid, cells, labels)
    if coverage_ratio < MIN_CELL_COVERAGE_RATIO:
        raise FloorplanGeometryError(
            [
                f"房间拼不满户型：认领到的地方只占内部自由面积的 {coverage_ratio:.0%}"
                f"（门槛 {MIN_CELL_COVERAGE_RATIO:.0%}）——边界提取有问题，不把这份结构往下游传"
            ]
        )
    return FloorplanGeometry(
        outline=_trace_outline(grid),
        frame_width_px=width_px,
        frame_height_px=height_px,
        plan_box=(
            grid.left_px / width_px,
            grid.top_px / height_px,
            grid.right_px / width_px,
            grid.bottom_px / height_px,
        ),
        walls=walls,
        openings=openings,
        rooms=rooms,
        cell_coverage_ratio=round(coverage_ratio, 4),
    )


def _to_room_outlines(
    grid: _Grid,
    cells: dict[tuple[int, int], tuple[int, int, int, int]],
    labels: dict[tuple[int, int], str],
) -> list[RoomOutline]:
    by_room: dict[str, list[tuple[int, int, int, int]]] = {}
    for cell, name in labels.items():
        by_room.setdefault(name, []).append(cells[cell])
    total_px = sum((right - left) * (bottom - top) for left, top, right, bottom in cells.values())
    outlines: list[RoomOutline] = []
    for name, boxes in by_room.items():
        area_px = sum((right - left) * (bottom - top) for left, top, right, bottom in boxes)
        centroid_x = (
            sum(
                (left + right) / 2 * (right - left) * (bottom - top)
                for left, top, right, bottom in boxes
            )
            / area_px
        )
        centroid_y = (
            sum(
                (top + bottom) / 2 * (right - left) * (bottom - top)
                for left, top, right, bottom in boxes
            )
            / area_px
        )
        outlines.append(
            RoomOutline(
                name=name,
                boxes=[
                    (
                        left / grid.width_px,
                        top / grid.height_px,
                        right / grid.width_px,
                        bottom / grid.height_px,
                    )
                    for left, top, right, bottom in sorted(boxes)
                ],
                area_ratio=round(area_px / max(total_px, 1), 4),
                centroid=(centroid_x / grid.width_px, centroid_y / grid.height_px),
            )
        )
    return sorted(outlines, key=lambda outline: -outline.area_ratio)


def _cell_coverage_ratio(
    grid: _Grid,
    cells: dict[tuple[int, int], tuple[int, int, int, int]],
    labels: dict[tuple[int, int], str],
) -> float:
    """自证数：**被房间认领到的自由像素** ÷ 户型内部的全部自由像素（内部里不是墙的那些）。

    两边都按像素数，不按格子面积——格子是从墙线量到墙线的，把墙带算了半条进去，
    拿它当分子会算出超过 100% 的覆盖率（首轮如此，102.1%）。一个能超过 100% 的自证数
    只拦得住漏，拦不住多，等于半道闸。
    """
    claimed = [[False] * grid.width_px for _ in range(grid.height_px)]
    for cell in labels:
        left, top, right, bottom = cells[cell]
        for y in range(top, bottom + 1):
            row = claimed[y]
            for x in range(left, right + 1):
                row[x] = True
    free_px = 0
    claimed_px = 0
    for y in range(grid.top_px, grid.bottom_px + 1):
        for x in range(grid.left_px, grid.right_px + 1):
            if grid.is_inside[y][x] and not grid.wall_mask[y][x]:
                free_px += 1
                claimed_px += claimed[y][x]
    return claimed_px / max(free_px, 1)


# ---------------------------------------------------------------------------
# 七、核验叠图（解析件的自证材料，不是产物）
# ---------------------------------------------------------------------------


def render_geometry_overlay(image_bytes: bytes, geometry: FloorplanGeometry) -> bytes:
    """把提取出来的墙、洞、房间画回原图上——"叠不叠得上"是验收判据本身。

    **这不是母版**。母版是 `plan-2d-render` 的产物，画法归 render2d 仓；这里画的是
    解析件的自证材料，看的人是我们不是业主。两者放在一起会让"独立仓"退化成措辞。
    """
    with Image.open(io.BytesIO(image_bytes)) as source:
        canvas = source.convert("RGBA")
    width_px, height_px = canvas.size
    layer = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    pen = ImageDraw.Draw(layer)

    for index, room in enumerate(geometry.rooms):
        red, green, blue = _OVERLAY_ROOM_COLORS[index % len(_OVERLAY_ROOM_COLORS)]
        for left, top, right, bottom in room.boxes:
            pen.rectangle(
                [left * width_px, top * height_px, right * width_px, bottom * height_px],
                fill=(red, green, blue, 70),
            )
    for wall in geometry.walls:
        half = max(1.0, wall.thickness_ratio * width_px / 2)
        if wall.axis == "vertical":
            center_x = wall.position_ratio * width_px
            pen.rectangle(
                [
                    center_x - half,
                    wall.start_ratio * height_px,
                    center_x + half,
                    wall.end_ratio * height_px,
                ],
                fill=(20, 20, 20, 210),
            )
        else:
            center_y = wall.position_ratio * height_px
            pen.rectangle(
                [
                    wall.start_ratio * width_px,
                    center_y - half,
                    wall.end_ratio * width_px,
                    center_y + half,
                ],
                fill=(20, 20, 20, 210),
            )
    for opening in geometry.openings:
        colour = (230, 90, 20, 235) if opening.is_on_outer_wall else (40, 160, 230, 235)
        if opening.axis == "vertical":
            center_x = opening.position_ratio * width_px
            pen.rectangle(
                [
                    center_x - 2,
                    opening.start_ratio * height_px,
                    center_x + 2,
                    opening.end_ratio * height_px,
                ],
                fill=colour,
            )
        else:
            center_y = opening.position_ratio * height_px
            pen.rectangle(
                [
                    opening.start_ratio * width_px,
                    center_y - 2,
                    opening.end_ratio * width_px,
                    center_y + 2,
                ],
                fill=colour,
            )

    font = _load_cjk_font(max(12, round(min(width_px, height_px) * 0.014)))
    if font is not None:
        for room in geometry.rooms:
            pen.text(
                (room.centroid[0] * width_px, room.centroid[1] * height_px),
                room.name,
                fill=(0, 0, 0, 255),
                font=font,
                anchor="mm",
                stroke_width=3,
                stroke_fill=(255, 255, 255, 220),
            )

    merged = Image.alpha_composite(canvas, layer).convert("RGB")
    buffer = io.BytesIO()
    merged.save(buffer, format="PNG")
    return buffer.getvalue()


def _load_cjk_font(size_px: int) -> ImageFont.FreeTypeFont | None:
    """找一个中文字库写房间名；找不到就不写（叠图照出，颜色仍能对照）。"""
    for path in _CJK_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size_px)
        except OSError:
            continue
    return None
