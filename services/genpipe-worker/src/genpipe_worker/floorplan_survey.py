"""勘测一步：整图一次调用，问出**每个房间在图上的哪一块**，外加指北针指向。

它不下任何结论——**位置归模型，裁剪归代码**。产物喂给 `floorplan_regions` 按 `box` 裁剪放大。
窗开在哪面墙**不在这一步问**：真跑里整图勘测把次卧的飘窗报成右墙（实为下墙）、
还给没有窗的卫生间报了一面西窗，而同一次跑的近景读对了——那件事挪到近景里定
（`RoomLegend.window_walls`）。

**按每个房间规划，不是整体规划**（用户裁决 2026-08-30 晚）：整图一次读漏掉了阳台端头那两个
虚线设备位，九次改 prompt 都没救回来；而模型被单独问阳台那一小块时逐个说得出来——
所以要按房间切块，不是把整张图切成几大块。
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from genpipe_worker.models import FloorplanSurvey, RoomRegion, VisionReader

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_THOUSANDTH_GRID = 1000.0
"""Qwen-VL 系模型报框的自家惯例：整图按 0~1000 的网格给坐标。提示词要的是 0~1，但真跑
（2026-09-04，qwen3-vl-plus 读 1240px 的真户型）回了 [96, 572, 350, 856] 这种千分网格——
按它自家惯例机械换算，不是猜：判据是**所有坐标都落在 0~1000 且至少一个大于 1**。
两种都不像的（负数、超过 1000）仍响亮失败，由裁剪那一步拦。"""
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$")

_SURVEY_SYSTEM_PROMPT = """\
你是户型图判读器的勘测员。这一步**只报图面事实，不下结论**：每个房间在图上占哪一块，
以及图上有没有指北针。功能推断、朝向结论、尺寸估计都不是这一步的事。

坐标用归一化数值：整张图左上角是 (0, 0)，右下角是 (1, 1)。一个房间的 box 写成
[x0, y0, x1, y1]，要把这个房间的墙体连同里面画的东西整个框住，宁可框大一点也别框漏。

输出严格 JSON，不要代码围栏、不要任何解释文字：
{
  "northPointsTo": "top" | "bottom" | "left" | "right" | null,
  "rooms": [
    {"name": "<图上写的房间名>", "box": [x0, y0, x1, y1]}
  ]
}
`northPointsTo` 是**指北针的 N 箭头指向图面的哪一边**；图上没有指北针就填 null，不要猜。
`name` 用图上写的字，图上没写名字的空间不要列。
**只列房间（人能走进去的空间）**：主卧、次卧、客厅、厨房、卫生间、阳台、玄关这一类。
"飘窗""设备平台"这些是房间里的构件或图上的标注，不是房间，不要单独列一条——
它们会在所属房间那一条里被看到。
"""

_SURVEY_USER_PROMPT = """\
勘测这张户型图：把每个有名字的房间各列一条（它占图的哪一块），
再报一次指北针的 N 指向图面哪一边。按系统提示里的形态输出 JSON。
"""


class FloorplanSurveyError(Exception):
    """勘测输出不可用——响亮失败，不吞不猜。"""

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


def build_survey_prompts() -> tuple[str, str]:
    """勘测两段提示（系统、用户）。与闭集无关：这一步不认识"特征"，只认识房间在哪一块。"""
    return _SURVEY_SYSTEM_PROMPT, _SURVEY_USER_PROMPT


def parse_survey_output(raw: str) -> FloorplanSurvey:
    """把勘测原文解析成结构。围栏与前后缀寒暄容忍，形态不对即失败。"""
    text = _FENCE_RE.sub("", raw.strip()).strip()
    match = _JSON_BLOCK_RE.search(text)
    if match is None:
        raise FloorplanSurveyError([f"勘测输出里找不到 JSON 对象：{raw.strip()[:400]}"])
    try:
        payload = json.loads(match.group(0))
    except ValueError as e:
        raise FloorplanSurveyError([f"勘测输出不是合法 JSON：{e}"]) from e
    try:
        survey = FloorplanSurvey.model_validate(payload)
    except ValidationError as e:
        raise FloorplanSurveyError(
            [f"{'.'.join(str(x) for x in err['loc'])}：{err['msg']}" for err in e.errors()]
        ) from e
    if not survey.rooms:
        raise FloorplanSurveyError(["勘测没有报出任何房间：这张图读不出房间划分，不往下走"])
    return normalize_grid_boxes(survey)


def normalize_grid_boxes(survey: FloorplanSurvey) -> FloorplanSurvey:
    """千分网格 → 归一化（纯函数）。已经是 0~1 的原样返回。"""
    coords = [coordinate for room in survey.rooms for coordinate in room.box]
    if not coords or not any(c > 1.0 for c in coords):
        return survey
    if not all(0.0 <= c <= _THOUSANDTH_GRID for c in coords):
        return survey
    rooms = [
        RoomRegion(
            name=room.name,
            box=(
                room.box[0] / _THOUSANDTH_GRID,
                room.box[1] / _THOUSANDTH_GRID,
                room.box[2] / _THOUSANDTH_GRID,
                room.box[3] / _THOUSANDTH_GRID,
            ),
        )
        for room in survey.rooms
    ]
    return FloorplanSurvey(north_points_to=survey.north_points_to, rooms=rooms)


async def survey_floorplan(
    image_bytes: bytes,
    image_media_type: str,
    reader: VisionReader,
    logical_model: str,
) -> FloorplanSurvey:
    """跑勘测一步。"""
    system_prompt, user_prompt = build_survey_prompts()
    raw = await reader.complete_with_image(
        logical_model, system_prompt, user_prompt, image_bytes, image_media_type
    )
    return parse_survey_output(raw)
