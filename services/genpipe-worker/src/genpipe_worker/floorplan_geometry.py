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
7. **洞口类型**（2026-09-05 加）：门、窗、过口在掩膜上长得一样，区别在被开运算抹掉的细线上，
   所以回原图灰度量三样证据——门弧（四分之一圆上多数采样角压到孤立细线）、跨洞平行线
   （横向剖面里的暗谷）、门扇线（垂直于墙穿过断口的实线）。一样都没有就 `unknown` 并说明
   缺什么，**不许默认成门**；入户门＝外轮廓上带门弧的洞，全户唯一。

**产物没有任何绝对尺寸**。比例标定服务的是报告里的数字，出图只要相对关系对
（交接文档追记七 §八-3）；洞宽如实给出去，是留给门洞反标定那一级标定物用的输入。

**画法归 render2d**：本模块只出坐标。这里唯一画的东西是
:func:`render_geometry_overlay` 的核验叠图——它是解析件的自证材料（"提取出来的东西
和原图叠不叠得上"，验收判据本身），不是产物；母版是 `plan-2d-render` 的产物。

**常量全部是单张样本实测值**（那张 92㎡ 楼书级矢量渲染图，1080×1466）。跨图与脏图没有数据
（技术债"只测过一张图"，处置时点＝拿到第二批样本）。它们按图的**相对尺度**取值而不是写死
像素，但相对尺度本身也只在一张图上验过。

**2026-09-05 补**：第二批样本到了（138㎡ 彩色渲染户型图，1254×1254）。墙厚上限已改成随图实测
（:data:`WALL_THICKNESS_STROKE_MULTIPLE`），格子墙占比按 92+138 两张 retune（:data:`MAX_CELL_WALL_RATIO`）——
这两条不再是单样本值；其余常量仍只在 92 上验过。
"""

from __future__ import annotations

import heapq
import io
import math
from collections import deque
from collections.abc import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from genpipe_worker.models import (
    FloorplanGeometry,
    OpeningKind,
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

WALL_THICKNESS_STROKE_MULTIPLE = 2.0
"""墙厚上限＝本图实测墙宽的几倍。宽于此的暗条不是墙的横截面，是顺着墙走的那一段
（或路口两墙叠出来的一坨）。

**为什么改成随图量、不写死比例**：原先 0.05×图幅长边是照单张样本（92㎡）定的。第二批
样本（138㎡ 渲染户型图）到了才发现它太松——图幅大时 0.05 折算成五六十像素，把路口墙角
那一坨（约 56px）也当墙投了票，投出一条偏内侧的假墙线，把房间边界顶得离真墙差一个墙厚，
外圈闭合率因此只有 67%。改成 2×实测墙宽后：138 实测墙约 18px、上限 36px，路口那坨被挡在外；
92 的墙约 11px、上限 22px，都够得着真墙。取 2 而不是别的倍数，是因为一条真墙横切最宽也就到
约两倍墙厚（转角处），再宽必是顺墙或路口。实测墙宽＝墙掩膜里每个墙像素横竖行程取小、全图取中位。
"""

MAX_WALL_THICKNESS_RATIO = 0.05
"""墙厚上限的**兜底**比例（占图幅长边）：只有实测墙宽量不出来（图上没有墙，本就要响亮失败）
时才退回它；正常路径走 :data:`WALL_THICKNESS_STROKE_MULTIPLE`。"""

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

MAX_CELL_WALL_RATIO = 0.50
"""格子内部允许的墙占比：超过即这格是墙不是屋。

取 0.50 不 0.25：0.25 照 92㎡ 单图定，太严——贴着外墙那圈地板格里混进小半格墙带，
就被判成"墙不是屋"，房间边界因此缩进一个墙厚、够不着外墙（138 闭合 67% 的另一半原因）。
放到 0.50 后 138 闭合 0.67→0.98、92 也 0.937→0.963，两张都过。**这是 retune 不是公式**：
只在 92+138 两张上验过、落在 0.45~0.60 都能过的平台中段，样本更多前不当定值（《纪律·阈值有数据才定》）。
"""

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

# --- 洞口类型（2026-09-05）。下面每个数都只在 92㎡ 楼书图 + 138㎡ 渲染图两张上验过，
# --- 各自的数据写在 docstring 里；样本更多前不当定值（《纪律·阈值有数据才定》）。

OPENING_LINE_DARKNESS_MARGIN = 25
"""细线"暗"的判据：比它所在那片底色暗多少才算线（灰度差）。底色按洞旁的地面实测，不写死——
138 的地面是米色（灰度 205~230）、线是灰的（门弧 138~178、窗线 135~170），92 的纸面 236~243、
线是黑的（≤140）。两张图里线与底色的最小差是 27（138 入户门弧 178 对地面 205）。"""

DOOR_ARC_ANGLE_SAMPLES_DEG = tuple(range(10, 90, 5))
"""门弧采样角：10°~85° 每 5° 一个，16 个。避开 0°（弧的起点就在墙上，压到的是墙）
与 90°（那是门扇线）。"""

DOOR_ARC_RADIUS_RANGE = (0.7, 1.2)
"""门弧半径的搜索范围（占洞宽）。真跑 8 个门弧的半径落在 0.86~1.02 洞宽（掩膜开运算把断口
两头各吃掉一两像素，所以弧常比断口短一点）。"""

DOOR_ARC_RADIUS_TOLERANCE_PX = 2
"""同一个半径 R 上"压到线"的容差：抗锯齿的细线摊在相邻一两个像素上。"""

DOOR_ARC_ISOLATION_BAND_PX = (4, 9)
"""孤立细线判据：R 两侧 4~9px 内不许再有暗像素。门弧是单根细线，家具的排线、飘窗里的
填充、字形都不是——没有这一条，飘窗与家具会以 0.56~0.75 的命中率冒充门弧（真跑数据）；
加上之后非门的孤立命中率最高 0.25。"""

DOOR_ARC_MIN_ISOLATED_HIT_RATIO = 0.6
"""门弧成立的门槛：16 个采样角里至少六成压到孤立细线。真跑 8 个门弧 0.75~0.94，
23 个非门 ≤0.25（一个 19px 的墙角豁口 0.50，但它在前一道判据就被挡掉了）。取两者中间。"""

DOOR_HINGE_ALONG_OFFSETS_PX = (-8, -6, -4, -2, 0, 2)
"""铰链沿墙相对断口端点的可能偏移（负＝往断口里挪）。真跑 8 个门弧的最佳铰链在端点往里 0~6px：
门框与开运算都让掩膜里的断口比门本身宽一点。"""

DOOR_HINGE_ACROSS_STROKE_FRACTIONS = (0.0, 0.5)
"""铰链横向相对墙线中心的偏移（占墙宽，朝摆向一侧）。铰链装在墙面上不在墙中心：
真跑 8 个门弧里 6 个最佳在 0.5 墙宽、2 个在 0。"""

CROSS_LINE_PROFILE_HALF_WIDTH_STROKES = 1.5
"""跨洞平行线的横向剖面取墙线两侧各 1.5 倍墙宽：窗线画在墙带里（两面各一条 + 中间一两条玻璃线），
再远就是屋里的东西了。"""

CROSS_LINE_MIN_DIP_DEPTH = 40
"""剖面上一条暗线的最小凸显度（比两侧 4px 内的亮处暗多少）。真跑 14 个窗 3~5 条（最紧的一个是
138 阳台外沿：3 条，凸显 42/51/43）；带弧的门 0~1 条（138 的门画了门槛线）；过口 0 条。"""

MIN_WINDOW_CROSS_LINES = 3
"""外轮廓上的洞判成窗至少要几条跨洞暗线：窗的画法是两面各一条 + 玻璃线，最少三条。"""

MIN_SLIDING_DOOR_CROSS_LINES = 2
"""内墙上的洞、没有门弧，几条跨洞暗线判成推拉门：两扇各一条。
真跑两个推拉门 2、3 条，两个过口 0 条。"""

DOOR_LEAF_MIN_LENGTH_RATIO = 0.7
"""门扇线（垂直于墙、穿过断口的实线）最短占洞宽几成。**只有一个样本支持这条**（92 主卧门：
门开在一截没投出墙线的短墙上，落在产物里的断口只被门扇线穿过），而存档复判里它又在一个墙线
错位的假洞上认过一条家具边——所以门扇线今天**只写进依据、不定类型**（结果是 `unknown`）。"""

DOOR_LEAF_INTERIOR_RANGE = (0.1, 0.9)
"""门扇线只在断口内部找（占洞宽的位置）：两端那两条垂直线是窗框、墙面轮廓，窗和门都有。"""

DOOR_LEAF_MIN_CROSSING_PX = 3
"""门扇线要在墙线两侧各露出至少这么多像素：不穿过墙位的垂直线是家具边。"""

MAX_OPENING_WALL_COVER_RATIO = 0.5
"""断口处（墙线 ±2px）沿洞长被任一方向的墙盖住的比例上限：超过即墙角豁口（同向墙图在转角
必然缺一块，那几个像素属于另一条墙），不是洞。真跑一例 100%（92 主卧飘窗与阳台交界 19px）。"""

MIN_PARALLEL_WALL_COVER_RATIO = 0.8
"""墙线两侧 1.5 倍墙宽内若有一条横向偏移上沿洞长 ≥ 八成都是墙，这个断口是墙线错位（真墙在旁边），
不是洞。真跑一例 100%（92 阳台右侧：两条几乎平行的墙线相隔 13px）。"""

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

_OVERLAY_OPENING_COLORS: dict[OpeningKind, tuple[int, int, int, int]] = {
    "door": (40, 160, 230, 235),
    "window": (230, 140, 20, 235),
    "entry-door": (210, 30, 30, 235),
    "passage": (30, 170, 90, 235),
    "unknown": (120, 120, 120, 235),
}
"""核验叠图的洞口配色，按类型：门蓝、窗橙、入户门红、过口绿、推不出灰。类型对不对一眼可查。"""

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


def _median_wall_stroke_px(mask: Bitmap, left: int, top: int, right: int, bottom: int) -> float:
    """本图实测墙宽：图幅框内每个墙像素，取它横行程与竖行程的较小者（≈局部墙厚），全图取中位。

    直墙内部的像素：顺墙那向行程很长、跨墙那向就是墙厚，取小即墙厚；路口/转角处两向都长，
    是少数，被中位数摊掉。所以中位≈这张图上典型的一堵墙有多厚，随画幅自动缩放。
    """
    w = right - left + 1
    h = bottom - top + 1
    if w <= 0 or h <= 0:
        return 0.0
    h_run = [[0] * w for _ in range(h)]
    for gy in range(h):
        row = mask[top + gy]
        gx = 0
        while gx < w:
            if row[left + gx]:
                run_start = gx
                while gx < w and row[left + gx]:
                    gx += 1
                length = gx - run_start
                for xx in range(run_start, gx):
                    h_run[gy][xx] = length
            else:
                gx += 1
    strokes: list[int] = []
    for gx in range(w):
        gy = 0
        while gy < h:
            if mask[top + gy][left + gx]:
                run_start = gy
                while gy < h and mask[top + gy][left + gx]:
                    gy += 1
                length = gy - run_start
                for yy in range(run_start, gy):
                    horizontal = h_run[yy][gx]
                    strokes.append(horizontal if horizontal < length else length)
            else:
                gy += 1
    if not strokes:
        return 0.0
    strokes.sort()
    return float(strokes[len(strokes) // 2])


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
        """图幅长边。洞长下限按它取——是图上的尺度，与页面留白无关。"""
        self.wall_stroke_px = _median_wall_stroke_px(
            wall_mask, plan_left_px, plan_top_px, plan_right_px, plan_bottom_px
        )
        """本图实测墙宽（中位）。见 :func:`_median_wall_stroke_px`。"""
        self.wall_thickness_max_px = (
            WALL_THICKNESS_STROKE_MULTIPLE * self.wall_stroke_px
            if self.wall_stroke_px > 0
            else MAX_WALL_THICKNESS_RATIO * self.plan_long_side_px
        )
        """墙厚上限（像素）：**给墙线定位用**——截面宽于此的暗条不投票（挡住路口那一坨、别投出假墙线）。
        随图实测，见 :data:`WALL_THICKNESS_STROKE_MULTIPLE`。圈墙像素那步用兜底比例，见 :func:`_build_parallel_wall_mask`。"""
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
    """标出属于**同向**墙的像素：截面不宽于墙厚上限的那些暗条。

    这里用**宽松**上限（兜底比例，非实测那条紧的）：路口两墙交叠处，这一向的墙被横穿的
    墙撑宽，那几行仍然**是这堵墙的像素**，丢了会把一堵连续的墙从路口劈成两段
    （单测 `test_junction_rows_do_not_fake_a_thick_band` 就在防这个）。定位墙线要精，
    那用紧的（见 :func:`_vote_wall_lines`）；圈墙像素要全，这里用松的。
    """
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
    thickness_max_px = grid.wall_thickness_max_px
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
# 六、洞口类型：门弧 / 跨洞平行线 / 门扇线（2026-09-05）
# ---------------------------------------------------------------------------
#
# 洞是墙掩膜上判出来的（第五步），可门、窗、过口在掩膜上长得一样——都是墙上的一段空白。
# 区别在被开运算抹掉的那些细线上：门画一道四分之一圆弧（门扇摆过去的轨迹），窗在墙带里画
# 两三条与墙平行的线，过口什么都不画。所以类型要回**原图灰度**上量，掩膜只用来避开墙。
# 三样证据各挡一类，一样都没有就 `unknown`——**不许默认成门**：默认成门正是三维那边今天
# 把窗渲成黑板、把门渲成落地玻璃时手里没有真值可判的原因（失效清单 B4）。


_ARC_SAMPLES = tuple(
    (math.cos(math.radians(degree)), math.sin(math.radians(degree)))
    for degree in DOOR_ARC_ANGLE_SAMPLES_DEG
)


def _to_gray_planes(image_bytes: bytes) -> tuple[bytes, bytes]:
    """原图灰度与它的 3×3 极小值图，都按行铺成一串字节。

    细线抗锯齿会摊在相邻像素上，各像素都比线本身淡；取 3×3 极小值等于把线加粗到一定压得中，
    又不会把两条相隔 4px 以上的线粘成一条（孤立判据要靠这个间隔）。
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = image.convert("L")
        return gray.tobytes(), gray.filter(ImageFilter.MinFilter(3)).tobytes()


class _OpeningProbe:
    """量洞口证据要的四样：原图灰度、极小值图、墙掩膜（任一方向）、本图墙宽。只在本模块内流转。"""

    def __init__(
        self,
        gray: bytes,
        min3: bytes,
        wall_mask: Bitmap,
        width_px: int,
        height_px: int,
        wall_stroke_px: float,
    ) -> None:
        self.gray = gray
        self.min3 = min3
        self.wall_mask = wall_mask
        self.width_px = width_px
        self.height_px = height_px
        self.stroke_px = wall_stroke_px if wall_stroke_px > 0 else float(WALL_OPENING_KERNEL_PX)

    def gray_at(self, x: int, y: int) -> int:
        """图外按白算：图外没有线。"""
        if 0 <= x < self.width_px and 0 <= y < self.height_px:
            return self.gray[y * self.width_px + x]
        return 255

    def min3_at(self, x: int, y: int) -> int:
        if 0 <= x < self.width_px and 0 <= y < self.height_px:
            return self.min3[y * self.width_px + x]
        return 255

    def is_wall(self, x: int, y: int) -> bool:
        return 0 <= x < self.width_px and 0 <= y < self.height_px and self.wall_mask[y][x]

    @staticmethod
    def point(axis: PlanAxis, along: float, across: float) -> tuple[int, int]:
        """(沿墙, 横向) → (x, y)。竖墙沿 y 走、横向是 x；横墙反过来。"""
        if axis == "vertical":
            return round(across), round(along)
        return round(along), round(across)


class _OpeningPx:
    """一个洞在像素坐标里的位置：墙线位置、沿墙起讫、洞宽。"""

    def __init__(self, opening: PlanOpening, width_px: int, height_px: int) -> None:
        self.axis: PlanAxis = opening.axis
        if opening.axis == "vertical":
            self.position = opening.position_ratio * width_px
            self.start = opening.start_ratio * height_px
            self.end = opening.end_ratio * height_px
        else:
            self.position = opening.position_ratio * height_px
            self.start = opening.start_ratio * width_px
            self.end = opening.end_ratio * width_px
        self.length = max(self.end - self.start, 1.0)
        self.is_on_outer_wall = opening.is_on_outer_wall
        self.connects = opening.connects


class _ArcVote:
    """四个候选（铰链在起点/终点 × 朝低位/高位面摆）里最好的那个门弧投票结果。"""

    def __init__(self) -> None:
        self.isolated_hits = 0
        self.hits = 0
        self.radius_ratio = 0.0
        self.hinge_at_start = True
        self.toward_high = True

    @property
    def isolated_ratio(self) -> float:
        return self.isolated_hits / len(_ARC_SAMPLES)


def _wall_cover_ratio(probe: _OpeningProbe, opening: _OpeningPx) -> float:
    """断口处（墙线 ±2px）沿洞长有多大比例被任一方向的墙像素盖着。"""
    total = 0
    covered = 0
    for along in range(round(opening.start), round(opening.end) + 1):
        total += 1
        if any(
            probe.is_wall(*probe.point(opening.axis, along, opening.position + offset))
            for offset in (-2, -1, 0, 1, 2)
        ):
            covered += 1
    return covered / total if total else 0.0


def _parallel_wall_ratio(probe: _OpeningProbe, opening: _OpeningPx, half: int) -> tuple[float, int]:
    """墙线两侧 ±half 内，哪个横向偏移上沿洞长（让开两端一成）墙像素最多；返回 (占比, 偏移)。"""
    inset = 0.1 * opening.length
    first = round(opening.start + inset)
    last = round(opening.end - inset)
    best_ratio = 0.0
    best_offset = 0
    for offset in range(-half, half + 1):
        total = 0
        walled = 0
        for along in range(first, last + 1):
            total += 1
            if probe.is_wall(*probe.point(opening.axis, along, opening.position + offset)):
                walled += 1
        if total and walled / total > best_ratio:
            best_ratio = walled / total
            best_offset = offset
    return best_ratio, best_offset


def _cross_profile(probe: _OpeningProbe, opening: _OpeningPx, half: int) -> list[int]:
    """横向剖面：墙线两侧 ±half 每个偏移上，沿洞长（让开两端 15%）的灰度中位。

    窗线、推拉门扇都与墙平行、贯穿整个洞，所以在它们的偏移上中位是暗的；家具边只占洞长一截，
    中位压不下去。
    """
    inset = 0.15 * opening.length
    first = round(opening.start + inset)
    last = round(opening.end - inset)
    profile: list[int] = []
    for offset in range(-half, half + 1):
        values = sorted(
            probe.gray_at(*probe.point(opening.axis, along, opening.position + offset))
            for along in range(first, last + 1)
        )
        profile.append(values[len(values) // 2] if values else 255)
    return profile


def _count_cross_lines(profile: Sequence[int], depth: int) -> int:
    """剖面里凸显度 ≥ depth 的暗谷个数：比两侧 4px 内的亮处至少暗 depth；相邻 3px 内只数一条。"""
    count = 0
    last = -10
    index = 1
    size = len(profile)
    while index < size - 1:
        value = profile[index]
        if value <= profile[index - 1] and value <= profile[index + 1]:
            plateau_end = index
            while plateau_end + 1 < size and profile[plateau_end + 1] == value:
                plateau_end += 1
            left = max(profile[max(0, index - 4) : index])
            right_slice = profile[plateau_end + 1 : plateau_end + 5]
            right = max(right_slice) if right_slice else value
            if min(left, right) - value >= depth and index - last >= 3:
                count += 1
                last = index
            index = plateau_end + 1
        else:
            index += 1
    return count


def _sector_background(
    probe: _OpeningProbe, opening: _OpeningPx, hinge: float, direction: int, sign: int
) -> int:
    """门弧会画在的那片扇形（半径 0.3~0.6 洞宽）里的灰度中位——那片地面的底色。"""
    values: list[int] = []
    for cos_t, sin_t in _ARC_SAMPLES[::2]:
        for fraction in (0.3, 0.4, 0.5, 0.6):
            radius = fraction * opening.length
            x, y = probe.point(
                opening.axis,
                hinge + direction * radius * cos_t,
                opening.position + sign * radius * sin_t,
            )
            values.append(probe.gray_at(x, y))
    values.sort()
    return values[len(values) // 2]


def _door_arc(probe: _OpeningProbe, opening: _OpeningPx) -> _ArcVote:
    """找门弧：以断口一端附近为圆心的四分之一圆，多数采样角上都压到一根孤立细线。

    四个候选＝铰链在起点或终点 × 门朝低位面或高位面摆。每个候选再让圆心在铰链附近微移
    （沿墙 :data:`DOOR_HINGE_ALONG_OFFSETS_PX`、横向 :data:`DOOR_HINGE_ACROSS_STROKE_FRACTIONS`），
    半径在 :data:`DOOR_ARC_RADIUS_RANGE` 内投票：哪个半径 R 让最多采样角在 R±容差 内有暗像素，
    且那根线两侧 4~9px 内干净（:data:`DOOR_ARC_ISOLATION_BAND_PX`）。落在墙掩膜上的像素不算暗——
    弧的两端本来就搭在墙上。
    """
    length = opening.length
    radius_low = int(DOOR_ARC_RADIUS_RANGE[0] * length)
    radius_high = int(DOOR_ARC_RADIUS_RANGE[1] * length) + 1
    band_low, band_high = DOOR_ARC_ISOLATION_BAND_PX
    tolerance = DOOR_ARC_RADIUS_TOLERANCE_PX
    best = _ArcVote()
    for hinge_at_start in (True, False):
        hinge = opening.start if hinge_at_start else opening.end
        direction = 1 if hinge_at_start else -1
        for toward_high in (False, True):
            sign = 1 if toward_high else -1
            threshold = (
                _sector_background(probe, opening, hinge, direction, sign)
                - OPENING_LINE_DARKNESS_MARGIN
            )
            for along_offset in DOOR_HINGE_ALONG_OFFSETS_PX:
                center_along = hinge - direction * along_offset
                for across_fraction in DOOR_HINGE_ACROSS_STROKE_FRACTIONS:
                    center_across = opening.position + sign * across_fraction * probe.stroke_px
                    dark_by_angle: list[set[int]] = []
                    for cos_t, sin_t in _ARC_SAMPLES:
                        dark: set[int] = set()
                        for radius in range(radius_low - band_high, radius_high + band_high):
                            x, y = probe.point(
                                opening.axis,
                                center_along + direction * radius * cos_t,
                                center_across + sign * radius * sin_t,
                            )
                            if probe.is_wall(x, y):
                                continue
                            if probe.min3_at(x, y) < threshold:
                                dark.add(radius)
                        dark_by_angle.append(dark)
                    for radius in range(radius_low, radius_high):
                        hits = 0
                        isolated = 0
                        for dark in dark_by_angle:
                            if not any(
                                radius + delta in dark for delta in range(-tolerance, tolerance + 1)
                            ):
                                continue
                            hits += 1
                            if not any(
                                radius + delta in dark or radius - delta in dark
                                for delta in range(band_low, band_high + 1)
                            ):
                                isolated += 1
                        if (isolated, hits) > (best.isolated_hits, best.hits):
                            best.isolated_hits = isolated
                            best.hits = hits
                            best.radius_ratio = radius / length
                            best.hinge_at_start = hinge_at_start
                            best.toward_high = toward_high
    return best


def _door_leaf(
    probe: _OpeningProbe, opening: _OpeningPx, profile: Sequence[int]
) -> tuple[float, float] | None:
    """门扇线：断口内部一条垂直于墙、穿过墙位、够长、够细的实线。返回 (长/洞宽, 位置/洞宽)。

    只挡一种情形：门开在一截没投出墙线的短墙上（92 主卧门），产物里的断口是隔壁那条墙线上
    与它相交的空白，门弧的圆心离这条线有半个门那么远、四个候选都够不着，只有门扇线穿过来。
    """
    length = opening.length
    threshold = min(profile[0], profile[-1]) - OPENING_LINE_DARKNESS_MARGIN
    half_stroke = probe.stroke_px / 2
    span = int(1.2 * length)
    interior_low, interior_high = DOOR_LEAF_INTERIOR_RANGE
    best: tuple[float, float] | None = None

    def dark(along: int, offset: int) -> bool:
        x, y = probe.point(opening.axis, along, opening.position + offset)
        return not probe.is_wall(x, y) and probe.min3_at(x, y) < threshold

    for along in range(
        round(opening.start + interior_low * length),
        round(opening.start + interior_high * length) + 1,
    ):
        longest = (0, 0)
        offset = -span
        while offset <= span:
            if not dark(along, offset):
                offset += 1
                continue
            run_start = offset
            run_end = offset
            offset += 1
            while offset <= span:
                if dark(along, offset):
                    run_end = offset
                    offset += 1
                elif offset + 1 <= span and dark(along, offset + 1):
                    offset += 1
                elif offset + 2 <= span and dark(along, offset + 2):
                    offset += 2
                else:
                    break
            if run_end - run_start > longest[1] - longest[0]:
                longest = (run_start, run_end)
        run_start, run_end = longest
        run_length = run_end - run_start + 1
        if run_length < DOOR_LEAF_MIN_LENGTH_RATIO * length:
            continue
        if run_start > -half_stroke - DOOR_LEAF_MIN_CROSSING_PX:
            continue
        if run_end < half_stroke + DOOR_LEAF_MIN_CROSSING_PX:
            continue
        # 细：同一段横向范围、沿墙错开 4px 的两侧都不该暗（否则是墙或色块，不是一根线）
        thin = True
        for side in (-4, 4):
            samples = range(run_start, run_end + 1, 3)
            dark_count = sum(
                1
                for offset in samples
                if probe.gray_at(
                    *probe.point(opening.axis, along + side, opening.position + offset)
                )
                < threshold
            )
            if dark_count > len(samples) / 2:
                thin = False
        if not thin:
            continue
        candidate = (run_length / length, (along - opening.start) / length)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def _kind_of(probe: _OpeningProbe, opening: _OpeningPx) -> tuple[OpeningKind, str]:
    """一个洞的类型与依据。判据按"先排除不是洞的、再找门弧、再找窗线、最后门扇线"的次序。"""
    half = max(2, round(CROSS_LINE_PROFILE_HALF_WIDTH_STROKES * probe.stroke_px))
    cover = _wall_cover_ratio(probe, opening)
    if cover >= MAX_OPENING_WALL_COVER_RATIO:
        return "unknown", f"墙角豁口：断口处 {cover:.0%} 被另一方向的墙盖着，不是洞"
    parallel, offset = _parallel_wall_ratio(probe, opening, half)
    if parallel >= MIN_PARALLEL_WALL_COVER_RATIO:
        return (
            "unknown",
            f"断口旁 {offset:+d}px 处贴着一道平行墙（沿洞长 {parallel:.0%}）——墙线错位，不是洞",
        )
    if not opening.connects:
        return "unknown", "两侧都不属于任何房间（管井/设备位一类），不是户内的门窗"
    outer = opening.is_on_outer_wall
    where = "外轮廓上" if outer else "内墙上"
    arc = _door_arc(probe, opening)
    if arc.isolated_ratio >= DOOR_ARC_MIN_ISOLATED_HIT_RATIO:
        evidence = (
            f"门弧：{arc.isolated_hits}/{len(_ARC_SAMPLES)} 个采样角压到孤立细线，"
            f"半径≈{arc.radius_ratio:.2f} 洞宽，铰链在{'起点' if arc.hinge_at_start else '终点'}、"
            f"朝{'高位' if arc.toward_high else '低位'}面一侧摆；洞在{where}"
        )
        return ("entry-door" if outer else "door"), evidence
    profile = _cross_profile(probe, opening, half)
    lines = _count_cross_lines(profile, CROSS_LINE_MIN_DIP_DEPTH)
    if outer and lines >= MIN_WINDOW_CROSS_LINES:
        return (
            "window",
            f"跨洞平行线：横向剖面 {lines} 条暗线（凸显 ≥{CROSS_LINE_MIN_DIP_DEPTH}）、无门弧；"
            f"洞在{where}",
        )
    if not outer and lines >= MIN_SLIDING_DOOR_CROSS_LINES:
        return "door", f"跨洞平行线：横向剖面 {lines} 条暗线、无门弧；洞在{where}——推拉门画法"
    leaf = _door_leaf(probe, opening, profile)
    if leaf is not None:
        # 门扇线单独不定类型：只有一个样本支持（92 主卧门），且存档复判里它在一个墙线错位的
        # 假洞上认过一条家具边（真户型旧产物第 3 个洞）。先只如实写进依据，样本够了再升成判据。
        length_ratio, position_ratio = leaf
        return (
            "unknown",
            f"{where}的断口：无门弧、跨洞暗线 {lines} 条；断口内 {position_ratio:.2f} 处有一条"
            f"垂直于墙、长 {length_ratio:.2f} 洞宽的实线穿过墙位（像门扇线，单样本判据不定）",
        )
    if outer:
        return (
            "unknown",
            f"外轮廓上的断口：无门弧、跨洞暗线 {lines} 条不足 {MIN_WINDOW_CROSS_LINES}、"
            "无门扇线——是窗是门定不了",
        )
    return "passage", f"内墙上的断口：无门弧、跨洞暗线 {lines} 条、无门扇线——没有门扇的过口"


def _infer_opening_kinds(
    probe: _OpeningProbe, openings: Sequence[PlanOpening]
) -> list[PlanOpening]:
    """给每个洞定类型。入户门全户唯一（方法论 T1）：外轮廓上带门弧的洞超过一个，一个都不认。"""
    judged = [
        _kind_of(probe, _OpeningPx(opening, probe.width_px, probe.height_px))
        for opening in openings
    ]
    entry_count = sum(1 for kind, _ in judged if kind == "entry-door")
    result: list[PlanOpening] = []
    for opening, (kind, evidence) in zip(openings, judged, strict=True):
        if kind == "entry-door" and entry_count > 1:
            kind = "unknown"
            evidence = (
                f"外轮廓上有 {entry_count} 个带门弧的洞，入户门只能有一个（方法论 T1）；"
                f"原判据：{evidence}"
            )
        result.append(opening.model_copy(update={"kind": kind, "kind_evidence": evidence}))
    return result


def _opening_kind_coverage(openings: Sequence[PlanOpening]) -> float:
    """给出了非 unknown 类型的洞占全部洞的比例。一个洞都没有时记 1.0：没有一个是没推出来的。"""
    if not openings:
        return 1.0
    return sum(1 for opening in openings if opening.kind != "unknown") / len(openings)


def classify_opening_kinds(image_bytes: bytes, geometry: FloorplanGeometry) -> FloorplanGeometry:
    """给一份已有的几何产物补洞口类型（原图 + 产物 → 带 `kind` 的产物）。**全程不调模型**。

    :func:`extract_geometry` 已经顺手做了这一步；这个入口给的是**存档复判**——老产物
    （比如三维那边留档的真户型 19 个洞）不重跑提取也能拿到类型，与新产物同一套判据。
    """
    mask, width_px, height_px = _to_wall_mask(image_bytes)
    if geometry.frame_width_px and (width_px, height_px) != (
        geometry.frame_width_px,
        geometry.frame_height_px,
    ):
        raise FloorplanGeometryError(
            [
                f"图与产物对不上：图是 {width_px}×{height_px}，"
                f"产物按 {geometry.frame_width_px}×{geometry.frame_height_px} 归一"
            ]
        )
    grid = _locate_plan(mask, width_px, height_px)
    gray, min3 = _to_gray_planes(image_bytes)
    probe = _OpeningProbe(gray, min3, mask, width_px, height_px, grid.wall_stroke_px)
    openings = _infer_opening_kinds(probe, geometry.openings)
    return geometry.model_copy(
        update={
            "openings": openings,
            "opening_kind_coverage_ratio": round(_opening_kind_coverage(openings), 4),
        }
    )


# ---------------------------------------------------------------------------
# 七、对外入口
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
    gray, min3 = _to_gray_planes(image_bytes)
    openings = _infer_opening_kinds(
        _OpeningProbe(gray, min3, mask, width_px, height_px, grid.wall_stroke_px), openings
    )
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
        opening_kind_coverage_ratio=round(_opening_kind_coverage(openings), 4),
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
        colour = _OVERLAY_OPENING_COLORS[opening.kind]
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
