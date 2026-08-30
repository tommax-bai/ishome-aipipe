"""CLI：`floorplan-parse --image 户型图.png [-o 产物.json]`。

工具形态先行（同渲染层裁决 2026-08-29：不成服务，以工具形式存在）——本入口就是"工具"的全部：
读图、调模型、过校验、落盘、汇报，失败带明细退出非零。**接进 Temporal 的时点写死＝上传入口
就绪时**：那时图落私有 OSS、activity 拿到的是资产键，入参形态与本地文件路径不是一回事。

模型经本机 LiteLLM 网关按逻辑名调用。**改了网关配置要重启才认新逻辑名**，而 4000 是常驻网关
——新逻辑名的第一次真跑另起临时端口（infra `litellm/run-dev.sh 4001`），用
`--gateway http://127.0.0.1:4001/v1` 指过去，跑完停掉临时网关，常驻的不动。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from genpipe_worker.floorplan_parse import (
    PARSE_LOGICAL_MODEL,
    FloorplanParseError,
    read_floorplan_features,
)
from genpipe_worker.layout_features import LayoutFeatureSetError, LayoutFeatureViolation
from genpipe_worker.llm_client import LiteLlmVisionClient, LlmGatewayError

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


async def _run(args: argparse.Namespace) -> int:
    image_path: Path = args.image
    try:
        image_bytes = image_path.read_bytes()
        media_type = _media_type_of(image_path)
    except (OSError, ValueError) as e:
        print(f"读图失败：{e}", file=sys.stderr)
        return 2

    client = LiteLlmVisionClient(base_url=args.gateway, api_key=args.api_key)
    try:
        reading = await read_floorplan_features(
            image_bytes, media_type, client, logical_model=args.model
        )
    except LayoutFeatureViolation as e:
        # 硬门禁：产出的标记不合契约。报出**是哪个键**，不修剪不降级。
        print("闭集校验不通过（fail loud，不剔除不静默）：", file=sys.stderr)
        for line in e.details:
            print(f"  - {line}", file=sys.stderr)
        return 3
    except (FloorplanParseError, LlmGatewayError, LayoutFeatureSetError) as e:
        print(f"解析失败：{e}", file=sys.stderr)
        return 2
    finally:
        await client.aclose()

    features = reading.features
    payload = {
        "image": {
            "path": str(image_path),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "bytes": len(image_bytes),
        },
        "logicalModel": reading.logical_model,
        "rawOutput": reading.raw_output,
        "product": features.model_dump(by_alias=True),
    }
    if args.out is not None:
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(f"特征标记（{len(features.layout_features)} 条，均在闭集内）：")
    for name, evidence in sorted(features.layout_features.items()):
        print(f"  {name} ← {evidence}")
    if not features.layout_features:
        print("  （一条都不成立——这是正常结果，不是失败）")
    print(f"观察区（{len(features.observations)} 条，闭集外，记录但不下发）：")
    for observation in features.observations:
        print(f"  {observation.subject}：{observation.finding}")
    print(f"读不出（{len(features.unreadable)} 条）：")
    for gap in features.unreadable:
        print(f"  {gap.subject}：{gap.reason}")
    if args.out is not None:
        print(f"存档：{args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="floorplan-parse",
        description="户型图解析：一张户型图 → 户型特征标记 + 依据（闭集校验，不出任何数字）",
    )
    parser.add_argument("--image", required=True, type=Path, help="户型图文件（png/jpg/webp）")
    parser.add_argument(
        "--model",
        default=PARSE_LOGICAL_MODEL,
        help=f"任务级逻辑模型名（默认 {PARSE_LOGICAL_MODEL}）",
    )
    parser.add_argument(
        "--gateway", default=None, help="LiteLLM 网关地址（默认取 LITELLM_BASE_URL 或 :4000）"
    )
    parser.add_argument("--api-key", default=None, help="网关 key（默认取 LITELLM_API_KEY）")
    parser.add_argument("-o", "--out", type=Path, default=None, help="产物与原文的存档 JSON")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
