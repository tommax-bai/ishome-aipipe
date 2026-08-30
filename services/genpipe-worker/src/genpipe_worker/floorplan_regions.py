"""分区读：**代码按勘测给的区域裁剪并放大**，每块单独送一次，问这一块画了什么图例。

立案证据是九次真跑（2026-08-30）：整图单次读一直看不见阳台端头那两个虚线设备位，
三种 prompt 框架都没救回来；而模型被单独问阳台那一小块时，逐个说得出"虚线框""实心黑点"。
**不是 prompt 问题，是分辨率问题**——所以给它看大的。

裁剪与放大是确定性动作，在这里做不在模型里做：模型只指位置（勘测的 `box`），剪刀在代码手里。
代价写明：一张图从 1 次调用变成 1 + N + 1 次（N＝房间数）。
"""

from __future__ import annotations

import asyncio
import io
import json
import re

from PIL import Image
from pydantic import ValidationError

from genpipe_worker.models import RoomLegend, RoomRegion, VisionReader

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$")

BOX_PADDING_RATIO = 0.08
"""裁剪外扩比例（按框自身尺寸算）：勘测的框略偏时不至于把墙体或角落的图例切掉。"""

# 试过并撤回：按整图尺寸算的"外扩下限"（2026-08-30 晚，三次真跑各测一轮）。
# 动机是实测发现**框越小越不准，而按框自身比例外扩恰好给得越少**——阳台那条窄横带被勘测
# 整体报低了约半个身位，按比例只补千分之三，裁出来是空白页。加了 5% 的下限之后阳台那块确实
# 裁对了、近景重新看见了那两个虚线设备位；**代价是每块都带进了邻居**：窗墙归属跟着乱——
# 主卧的飘窗三次里两次被报成"没画窗"，阳台被报出一面左墙的窗、经换算成"朝西"又催出一次
# `west_facing` 误报。误报比漏报贵，故撤回。
# **真正的瓶颈是勘测框的坐标精度**，不是外扩补得够不够——那要换一种定位办法（非模型的
# 版面定位，或按图幅切片），属解析实现路径选型，时点＝拿到第二批样本那一批一起定。

MIN_CROP_LONG_SIDE_PX = 1024
"""放大目标：裁块的长边至少到这个像素数——"看得更大"就是这一步的全部作用。"""

MAX_CROP_LONG_SIDE_PX = 1536
"""放大上限：再大只是徒增每次调用的图像成本，读不出更多东西。"""

MIN_BOX_AREA_RATIO = 0.001
"""框太小即判勘测出错：一个房间不可能只占整图千分之一。响亮失败，不硬裁。"""

_ROOM_SYSTEM_PROMPT = """\
你在看一张户型图里**截出来放大的一小块**。只描述你看见的线条与图形，不要下结论、
不要推断功能、不要给尺寸。

读图约定：实线是墙体与门窗；**虚线框通常是家具或设备的示意位**（洗衣机、柜体、床、餐桌）；
带圆点或十字的小图形通常是地漏、插座一类点位；框里的字是房间名或标注。
**很粗的黑实线是墙，墙上没有开口就是没有窗**。

输出严格 JSON，不要代码围栏、不要任何解释文字：
{
  "legend": "<你看见的东西，一段话说清>",
  "windowWalls": ["top" | "bottom" | "left" | "right", ...]
}
`windowWalls` 是**窗（含飘窗、落地窗）画在这一块的哪几条边上**；这一块没画窗就写空数组，
不要因为"房间应该有窗"就补一个。
"""

_ROOM_USER_PROMPT = """\
这一块是「{room}」。逐个说出你在这里看见的东西：画了哪些图例（虚线框、实线框、小方块、
圆点、台面、洁具、床、桌椅、设备），各画在这一块的哪一端；墙上的门窗怎么开；写了哪些字。
再单独回答：窗画在这一块的哪几条边上（上/下/左/右），没有窗就是空的。

只说看见的，看不清就说看不清。不要写"这是家政阳台"这类结论。
"""


class RoomLegendError(Exception):
    """近景读图输出不可用——响亮失败，不吞不猜。"""

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


class RoomCropError(Exception):
    """裁剪不出可用的块——勘测给的框有问题，响亮失败不硬裁。"""

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


def crop_room(image_bytes: bytes, region: RoomRegion) -> bytes:
    """按归一化 box 裁一块出来并放大，返回 PNG 字节。"""
    x0, y0, x1, y1 = region.box
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise RoomCropError([f"房间 `{region.name}` 的区域不成立：{list(region.box)}"])
    if (x1 - x0) * (y1 - y0) < MIN_BOX_AREA_RATIO:
        raise RoomCropError([f"房间 `{region.name}` 的区域小到不可能是个房间：{list(region.box)}"])
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        pad_x = (x1 - x0) * BOX_PADDING_RATIO
        pad_y = (y1 - y0) * BOX_PADDING_RATIO
        left = int(max(0.0, x0 - pad_x) * width)
        top = int(max(0.0, y0 - pad_y) * height)
        right = int(min(1.0, x1 + pad_x) * width)
        bottom = int(min(1.0, y1 + pad_y) * height)
        crop = image.crop((left, top, right, bottom))
        long_side = max(crop.size)
        if long_side < MIN_CROP_LONG_SIDE_PX:
            scale = min(MIN_CROP_LONG_SIDE_PX / long_side, MAX_CROP_LONG_SIDE_PX / long_side)
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        crop.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def parse_room_legend(room: str, raw: str) -> RoomLegend:
    """把一块近景的原文解析成结构。围栏与前后缀寒暄容忍，形态不对即失败。"""
    text = _FENCE_RE.sub("", raw.strip()).strip()
    match = _JSON_BLOCK_RE.search(text)
    if match is None:
        raise RoomLegendError([f"房间 `{room}` 的近景输出里找不到 JSON 对象：{raw.strip()[:300]}"])
    try:
        payload = json.loads(match.group(0))
    except ValueError as e:
        raise RoomLegendError([f"房间 `{room}` 的近景输出不是合法 JSON：{e}"]) from e
    payload["room"] = room
    try:
        return RoomLegend.model_validate(payload)
    except ValidationError as e:
        raise RoomLegendError(
            [
                f"房间 `{room}` 近景 {'.'.join(str(x) for x in err['loc'])}：{err['msg']}"
                for err in e.errors()
            ]
        ) from e


async def read_room_legends(
    image_bytes: bytes,
    regions: list[RoomRegion],
    reader: VisionReader,
    logical_model: str,
    *,
    max_concurrency: int = 4,
) -> list[RoomLegend]:
    """逐房间裁剪放大并读图例与窗墙。

    并发有上限——网关那头是按次计费的外部服务，别一次全推过去。
    **窗开在哪面墙在这里定，不在整图勘测里定**：同一张图里，整图勘测把次卧的飘窗报成右墙
    （实为下墙）、给没有窗的卫生间报了一面西窗，而近景两处都读对了。
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def read_one(region: RoomRegion) -> RoomLegend:
        crop = crop_room(image_bytes, region)
        async with semaphore:
            legend = await reader.complete_with_image(
                logical_model,
                _ROOM_SYSTEM_PROMPT,
                _ROOM_USER_PROMPT.format(room=region.name),
                crop,
                "image/png",
            )
        return parse_room_legend(region.name, legend)

    return list(await asyncio.gather(*(read_one(region) for region in regions)))
