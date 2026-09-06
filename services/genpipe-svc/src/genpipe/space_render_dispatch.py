"""三维线的派发形态与结果回流装配（草案，等用户拍键形态与产物词表）。

只装配，禁止任何 IO（与 `render3d_pipeline` 同一条可重放硬约束，import-linter 契约锁定）。

- 派发入参 `SpaceRenderDispatchSpec`＝`SpaceRenderSpec` + `result_callback_url`（三张图线先例：
  回调地址随派发注入，编排侧不知业务侧在哪）。入口校验在此响亮失败：机位清单给了不许空、机位 id
  不许重、机位 id 与风格模板 id 要能当键的一段（imagegen `check_camera_id` 同款）、回调地址要是
  http(s) URL。
- 回流产物词表 `SpaceRenderProduct`（contracts `openapi/genpipe.v1.yaml` `space_render_product`，
  草案）：产物名＝键的末段（contracts `registries/render_products.md`），不另发前缀词表。
- 回调报文 `build_space_render_task_result`：project.v1 `generation_task_result` 同形态——写实图是
  交业主的产物，场景包与底渲五路是血缘原料，都进 `products`；失败机位单列 `failed_cameras`
  （本线新增字段，**待拍**：要不要进 project.v1）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

from genpipe.render3d_pipeline import (
    CameraRender,
    SpaceCamera,
    SpaceRenderResult,
    SpaceRenderSpec,
)

KEY_SEGMENT_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
"""能当对象键一段的 id（imagegen `check_camera_id` 同款）：机位 id 与风格模板 id 都进写实图键。"""

SpaceRenderProduct = Literal[
    "design-package",
    "scene-package",
    "geometry",
    "depth",
    "line",
    "mask",
    "mask-index",
    "sketch",
    "realism",
]
"""三维线交回的产物词表（contracts genpipe.v1 `space_render_product`，草案、只增不改）。
`design-package` 是派发入参不是本线产物，本线不回流它；留在词表里给填包步进管线那一天用。"""

SPACE_RENDER_FAILED_CODE = "space-render-failed"
"""整户失败却没带 failure 时的兜底失败码（不该发生：`run_space_render` 失败路径都带 failure）。"""

_ROUTE_PRODUCTS: tuple[tuple[SpaceRenderProduct, str, str], ...] = (
    ("geometry", "geometry_key", "image/png"),
    ("depth", "depth_key", "image/png"),
    ("line", "line_key", "image/png"),
    ("mask", "mask_key", "image/png"),
    ("mask-index", "mask_index_key", "application/json"),
)
"""底渲五路：产物词 → `CameraRender` 上的键字段 → 内容类型（形态照 render_products.md）。"""


class SpaceRenderDispatchSpec(SpaceRenderSpec):
    """三维线一次派发的入参（contracts genpipe.v1 `space_render_spec`）：编排输入 + 回调地址。

    未知字段拒收（继承 `extra=forbid`）；下面的校验都在派发前拦——派进去再失败要多花一次
    scene-compile / base-render 的钱。
    """

    result_callback_url: str
    """结果回流地址：由派发方注入，编排侧不知道业务侧在哪（规范 §1.0 向上通信只走回调）。"""

    @field_validator("task_id", "design_package_key", "revision_id")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("不许空")
        return value

    @field_validator("style_template_id")
    @classmethod
    def _style_template_id_is_key_segment(cls, value: str) -> str:
        if KEY_SEGMENT_PATTERN.match(value) is None:
            raise ValueError(f"要能当键的一段（{KEY_SEGMENT_PATTERN.pattern}）：{value!r}")
        return value

    @field_validator("cameras")
    @classmethod
    def _cameras_nonempty_unique_key_segments(
        cls, cameras: list[SpaceCamera] | None
    ) -> list[SpaceCamera] | None:
        if cameras is None:
            return None
        if not cameras:
            raise ValueError("机位清单给了就不许空；要全部机位就不给这个字段")
        seen: set[str] = set()
        for camera in cameras:
            if KEY_SEGMENT_PATTERN.match(camera.camera_id) is None:
                raise ValueError(
                    f"机位 id 要能当键的一段（{KEY_SEGMENT_PATTERN.pattern}）：{camera.camera_id!r}"
                )
            if camera.camera_id in seen:
                raise ValueError(f"机位 id 重复：{camera.camera_id!r}")
            seen.add(camera.camera_id)
        return cameras

    @field_validator("width_px", "height_px")
    @classmethod
    def _positive_pixels(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"像素尺寸要是正整数：{value}")
        return value

    @field_validator("result_callback_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"回调地址要是完整的 http(s) URL：{value!r}")
        return value


class SpaceRenderTaskProduct(BaseModel):
    """一件交回业务侧的产物（project.v1 `generation_task_product` 的三维线形态）。"""

    product: SpaceRenderProduct
    object_key: str
    content_type: str | None = None
    gen_params: dict[str, Any] = Field(default_factory=dict)


class SpaceRenderDispatchResult(SpaceRenderResult):
    """派发形态的三维线结论：整份 `SpaceRenderResult` + 回流用的产物清单 + 送没送到。"""

    products: list[SpaceRenderTaskProduct] = Field(default_factory=list)
    """回调报文里的 products 原样：verdict=ok 时含场景包 + 每台成功机位的五路与写实图；failed 时只有
    已出的场景包（失败机位的五路键在 `failed_cameras[].render_keys` 里，不当产物回）。"""
    delivered: bool = False
    """结论是否已送到回调地址。**没送到不算完**：业务侧不知道就等于没做。"""


def camera_products(
    render: CameraRender, *, revision_id: str, style_template_id: str
) -> list[SpaceRenderTaskProduct]:
    """一台成功机位的产物（纯函数）：五路（有 sketch 加一路）是血缘原料，写实图是交业主的那张。

    写实图的 `gen_params` 带门禁回执、自证数与重派次数——"靠重派出来的"要在业务侧也判得出来。
    """
    lineage = {
        "camera_id": render.camera_id,
        "view_kind": render.view_kind,
        "revision_id": revision_id,
    }
    products = [
        SpaceRenderTaskProduct(
            product=product,
            object_key=getattr(render, key_field),
            content_type=content_type,
            gen_params=dict(lineage),
        )
        for product, key_field, content_type in _ROUTE_PRODUCTS
    ]
    if render.sketch_key:
        products.append(
            SpaceRenderTaskProduct(
                product="sketch",
                object_key=render.sketch_key,
                content_type="image/png",
                gen_params=dict(lineage),
            )
        )
    products.append(
        SpaceRenderTaskProduct(
            product="realism",
            object_key=render.image_object_key,
            content_type=render.content_type,
            gen_params={
                **lineage,
                "style_template_id": style_template_id,
                "control_key": render.control_key,
                "fidelity_score": render.fidelity_score,
                "gate": render.gate,
                "evidence": render.evidence,
                "gate_redispatches": render.gate_redispatches,
            },
        )
    )
    return products


def space_render_products(
    spec: SpaceRenderSpec, result: SpaceRenderResult
) -> list[SpaceRenderTaskProduct]:
    """整次运行的产物清单（纯函数）：场景包在前，随后按成功机位顺序各自的五路与写实图。"""
    products: list[SpaceRenderTaskProduct] = []
    if result.scene_package_key:
        products.append(
            SpaceRenderTaskProduct(
                product="scene-package",
                object_key=result.scene_package_key,
                content_type="application/json",
                gen_params={"revision_id": spec.revision_id, "evidence": result.scene_evidence},
            )
        )
    for render in result.renders:
        products.extend(
            camera_products(
                render, revision_id=spec.revision_id, style_template_id=spec.style_template_id
            )
        )
    return products


def build_space_render_task_result(
    spec: SpaceRenderSpec,
    result: SpaceRenderResult,
    *,
    products: Sequence[SpaceRenderTaskProduct],
    workflow_id: str,
    run_id: str,
) -> dict[str, Any]:
    """回调报文（纯函数）：contracts genpipe.v1 `space_render_result`——project.v1
    `generation_task_result` 的字段逐字同形态，另加 `failed_cameras`。

    `completed`＝至少一台机位出了写实图（`verdict=ok`）；失败机位照样单列，业务侧决定怎么告知。
    `failed`＝整户失败：`failure` 取编排结论的 {code, detail}，其余整户级失败码并进 detail
    （三张图线同口径：失败原因让业务侧一眼看到，不靠回来翻 Temporal 历史）。
    """
    payload: dict[str, Any] = {
        "task_id": spec.task_id,
        "status": "completed" if result.verdict == "ok" else "failed",
        "products": [product.model_dump() for product in products],
        "failed_cameras": [camera.model_dump() for camera in result.failed_cameras],
        "workflow_id": workflow_id,
        "run_id": run_id,
    }
    if result.verdict != "ok":
        code = (result.failure or {}).get("code") or SPACE_RENDER_FAILED_CODE
        detail = (result.failure or {}).get("detail") or ""
        extra = [check for check in result.failed_checks if check and check != f"{code}:{detail}"]
        if extra:
            detail = f"{detail}；其余：{' | '.join(extra)}" if detail else " | ".join(extra)
        payload["failure"] = {"code": code, "detail": detail}
    return payload
