"""CLI：`floorplan-geometry --image 户型图.png [--survey 存档.json] [-o 目录]`。

工具形态先行，同 `floorplan-parse`（裁决 2026-08-29：不成服务，以工具形式存在；
接进 Temporal 的时点写死＝上传入口就绪时）。

**`--survey` 是这个工具的要点**：几何提取本身一次模型都不调，只有"哪块是哪个房间"要问一次。
把那一次的答案存下来之后，几何这条链路可以**零成本、可复现地反复跑**——调阈值、看叠图、
再调，不必每轮都付一次读图钱，也不必担心两轮之间模型换了答案让改动的效果说不清。
没给 `--survey` 时才现问一次（一次调用，不做分区读）。

产出两样：几何 JSON 与**核验叠图**。验收判据就是叠图——提取出来的墙、洞、房间画回原图上，
叠得上就是对的，叠不上一眼能看见错在哪儿。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from genpipe_worker.floorplan_geometry import (
    FloorplanGeometryError,
    extract_geometry,
    render_geometry_overlay,
)
from genpipe_worker.floorplan_parse import PARSE_LOGICAL_MODEL
from genpipe_worker.floorplan_survey import FloorplanSurveyError, survey_floorplan
from genpipe_worker.llm_client import LiteLlmVisionClient, LlmGatewayError
from genpipe_worker.models import FloorplanSurvey

_MEDIA_TYPE_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _media_type_of(path: Path) -> str:
    media_type = _MEDIA_TYPE_BY_SUFFIX.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(
            f"不认识的图片格式 `{path.suffix}`（认得：{'、'.join(sorted(_MEDIA_TYPE_BY_SUFFIX))}）"
        )
    return media_type


def load_survey(path: Path) -> FloorplanSurvey:
    """从存档读勘测结果。既吃 `floorplan-parse` 的整份存档，也吃单独存的勘测。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "survey" in payload:
        payload = payload["survey"]
    return FloorplanSurvey.model_validate(payload)


async def _survey_once(
    image_bytes: bytes, media_type: str, args: argparse.Namespace
) -> FloorplanSurvey:
    client = LiteLlmVisionClient(base_url=args.gateway, api_key=args.api_key)
    try:
        return await survey_floorplan(image_bytes, media_type, client, args.model)
    finally:
        await client.aclose()


def _run(args: argparse.Namespace) -> int:
    image_path: Path = args.image
    try:
        image_bytes = image_path.read_bytes()
        media_type = _media_type_of(image_path)
    except (OSError, ValueError) as e:
        print(f"读图失败：{e}", file=sys.stderr)
        return 2

    if args.survey is not None:
        try:
            survey = load_survey(args.survey)
        except (OSError, ValueError) as e:
            print(f"读勘测存档失败：{e}", file=sys.stderr)
            return 2
        model_call_count = 0
    else:
        try:
            survey = asyncio.run(_survey_once(image_bytes, media_type, args))
        except (FloorplanSurveyError, LlmGatewayError) as e:
            print(f"勘测失败：{e}", file=sys.stderr)
            return 2
        model_call_count = 1

    try:
        geometry = extract_geometry(image_bytes, survey.rooms)
    except FloorplanGeometryError as e:
        # 响亮失败：说清缺什么，不给"差不多的"结构（红线一）。
        print("几何提取不通过（fail loud，不降级往下游传）：", file=sys.stderr)
        for line in e.details:
            print(f"  - {line}", file=sys.stderr)
        return 3

    print(f"模型调用 {model_call_count} 次（几何提取本身零次）")
    print(
        f"图幅：{geometry.plan_box[0]:.3f},{geometry.plan_box[1]:.3f}"
        f" → {geometry.plan_box[2]:.3f},{geometry.plan_box[3]:.3f}（归一化）"
    )
    print(f"墙段 {len(geometry.walls)} 段；洞 {len(geometry.openings)} 个（")
    outer = sum(1 for opening in geometry.openings if opening.is_on_outer_wall)
    print(f"  外墙上 {outer} 个、内墙上 {len(geometry.openings) - outer} 个）")
    print(f"房间 {len(geometry.rooms)} 个（占内部自由面积之比）：")
    for room in geometry.rooms:
        print(
            f"  {room.name}：{room.area_ratio:6.1%}"
            f"  锚点 {room.centroid[0]:.3f},{room.centroid[1]:.3f}"
            f"  {len(room.boxes)} 块"
        )
    print(f"自证：房间拼起来占内部自由面积 {geometry.cell_coverage_ratio:.1%}")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        payload = {
            "image": {
                "path": str(image_path),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "bytes": len(image_bytes),
            },
            "modelCallCount": model_call_count,
            "survey": survey.model_dump(by_alias=True),
            "geometry": geometry.model_dump(by_alias=True),
        }
        geometry_path = args.out / f"{stem}-geometry.json"
        overlay_path = args.out / f"{stem}-overlay.png"
        geometry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        overlay_path.write_bytes(render_geometry_overlay(image_bytes, geometry))
        print(f"存档：{geometry_path}")
        print(f"核验叠图：{overlay_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="floorplan-geometry",
        description="户型图几何提取：一张户型图 → 墙线、门窗洞、房间遮罩（归一化坐标，无绝对尺寸）",
    )
    parser.add_argument("--image", required=True, type=Path, help="户型图文件（png/jpg/webp）")
    parser.add_argument(
        "--survey",
        type=Path,
        default=None,
        help="勘测存档 JSON（给了就零调用复跑；不给则现问一次）",
    )
    parser.add_argument(
        "--model",
        default=PARSE_LOGICAL_MODEL,
        help=f"勘测用的逻辑模型名（默认 {PARSE_LOGICAL_MODEL}）",
    )
    parser.add_argument(
        "--gateway", default=None, help="LiteLLM 网关地址（默认取 LITELLM_BASE_URL 或 :4000）"
    )
    parser.add_argument("--api-key", default=None, help="网关 key（默认取 LITELLM_API_KEY）")
    parser.add_argument("-o", "--out", type=Path, default=None, help="产物目录（几何 JSON + 叠图）")
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
