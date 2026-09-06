"""三维线派发形态直测，不起 Temporal：

入站面——`POST /space-renders` 挂在 app 上、body 原样到 service、重复派发 409、入参不合格 422；
用例层——`start_space_render` 起的是 SpaceRenderDispatchWorkflow、以 task_id 定址（假 Temporal
client）；
装配——结论 → 产物清单 → 回调报文（写实图是交业主的产物，场景包与五路是血缘，失败机位单列）。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from genpipe import repo, service
from genpipe.models import WorkflowStartReceipt
from genpipe.render3d_pipeline import run_space_render
from genpipe.router import app
from genpipe.space_render_dispatch import (
    SpaceRenderDispatchSpec,
    SpaceRenderTaskProduct,
    build_space_render_task_result,
    space_render_products,
)
from genpipe.workflows import WORKFLOW_TASK_QUEUE, SpaceRenderDispatchWorkflow
from pydantic import ValidationError
from temporalio.exceptions import WorkflowAlreadyStartedError
from test_render3d_pipeline import (
    PREFIX,
    SCENE_KEY,
    FakeRunner,
    three_camera_behaviors,
    three_camera_spec,
)

CALLBACK_URL = "http://127.0.0.1:8103/api/v1/generation-tasks/01J0SPACE/result"


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "task_id": "01J0SPACE",
        "design_package_key": f"{PREFIX}/design-package.json",
        "revision_id": "rev-1",
        "cameras": [
            {"camera_id": "cam-bird-dollhouse", "view_kind": "bird"},
            {"camera_id": "cam-room-living", "view_kind": "room", "seed": 7},
        ],
        "style_template_id": "modern-minimal",
        "result_callback_url": CALLBACK_URL,
    }
    body.update(overrides)
    return body


def _dispatch_spec(**overrides: Any) -> SpaceRenderDispatchSpec:
    return SpaceRenderDispatchSpec.model_validate(_body(**overrides))


# ---------------------------------------------------------------------------
# 入站面
# ---------------------------------------------------------------------------


def test_space_render_route_is_mounted() -> None:
    """路由写了 ≠ 端点可达（test_genpipe_http 的教训）：断言打在 OpenAPI 面上。"""
    assert "post" in app.openapi()["paths"]["/api/v1/genpipe/space-renders"]


def test_create_space_render_passes_spec_through_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[SpaceRenderDispatchSpec] = []

    async def fake_start(spec: SpaceRenderDispatchSpec) -> WorkflowStartReceipt:
        seen.append(spec)
        return WorkflowStartReceipt(workflow_id=f"space-render-{spec.task_id}", run_id="r")

    monkeypatch.setattr("genpipe.router.start_space_render", fake_start)
    with TestClient(app) as client:
        response = client.post("/api/v1/genpipe/space-renders", json=_body())

    assert response.status_code == 202
    assert response.json() == {"workflow_id": "space-render-01J0SPACE", "run_id": "r"}
    spec = seen[0]
    assert spec.design_package_key == f"{PREFIX}/design-package.json"
    assert spec.revision_id == "rev-1"
    assert spec.cameras is not None
    assert [c.camera_id for c in spec.cameras] == ["cam-bird-dollhouse", "cam-room-living"]
    assert spec.cameras[1].seed == 7
    assert spec.style_template_id == "modern-minimal"
    assert spec.result_callback_url == CALLBACK_URL
    assert (spec.width_px, spec.height_px) == (1024, 768)
    assert spec.queues.render3d == "render3d-activities"


def test_create_space_render_conflict_on_duplicate_task(monkeypatch: pytest.MonkeyPatch) -> None:
    async def already_started(spec: SpaceRenderDispatchSpec) -> WorkflowStartReceipt:
        raise WorkflowAlreadyStartedError(f"space-render-{spec.task_id}", "SpaceRenderDispatch")

    monkeypatch.setattr("genpipe.router.start_space_render", already_started)
    with TestClient(app) as client:
        response = client.post("/api/v1/genpipe/space-renders", json=_body())
    assert response.status_code == 409
    assert "01J0SPACE" in response.json()["detail"]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"deep_revision_id": "deep-1"}, id="unknown-field"),
        pytest.param({"cameras": []}, id="empty-cameras"),
        pytest.param(
            {"cameras": [{"camera_id": "cam-a"}, {"camera_id": "cam-a"}]}, id="duplicate-camera"
        ),
        pytest.param({"cameras": [{"camera_id": "Cam_A"}]}, id="camera-id-not-key-segment"),
        pytest.param({"style_template_id": "Modern Minimal"}, id="style-id-not-key-segment"),
        pytest.param({"result_callback_url": "127.0.0.1:8103/result"}, id="callback-not-url"),
        pytest.param({"result_callback_url": "ftp://x/result"}, id="callback-not-http"),
        pytest.param({"width_px": 0}, id="zero-width"),
        pytest.param({"task_id": "  "}, id="blank-task-id"),
        pytest.param({"design_package_key": ""}, id="blank-package-key"),
    ],
)
def test_create_space_render_rejects_bad_input_before_dispatch(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]
) -> None:
    """入参不合格当场 422，service 一次都不被调——派进去再失败要多花一次 scene-compile 的钱。"""
    calls: list[SpaceRenderDispatchSpec] = []

    async def fake_start(spec: SpaceRenderDispatchSpec) -> WorkflowStartReceipt:
        calls.append(spec)
        return WorkflowStartReceipt(workflow_id="x", run_id="")

    monkeypatch.setattr("genpipe.router.start_space_render", fake_start)
    with TestClient(app) as client:
        response = client.post("/api/v1/genpipe/space-renders", json=_body(**overrides))
    assert response.status_code == 422, response.text
    assert calls == []


def test_dispatch_spec_accepts_all_cameras_when_list_omitted() -> None:
    """不给 cameras＝场景包里全部机位（base-render cameraIds=None），届时视角只能来自回执。"""
    body = _body()
    del body["cameras"]
    assert SpaceRenderDispatchSpec.model_validate(body).cameras is None
    with pytest.raises(ValidationError):
        SpaceRenderDispatchSpec.model_validate(_body(cameras=[]))


# ---------------------------------------------------------------------------
# 用例层：起 workflow
# ---------------------------------------------------------------------------


class _FakeHandle:
    def __init__(self, workflow_id: str) -> None:
        self.id = workflow_id
        self.result_run_id = "run-space-1"


class _FakeTemporalClient:
    def __init__(self) -> None:
        self.started: list[tuple[Any, Any, dict[str, Any]]] = []

    async def start_workflow(self, run: Any, arg: Any, **kwargs: Any) -> _FakeHandle:
        self.started.append((run, arg, kwargs))
        return _FakeHandle(kwargs["id"])


async def test_start_space_render_starts_dispatch_workflow_addressed_by_task_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeTemporalClient()

    async def fake_client() -> _FakeTemporalClient:
        return fake

    monkeypatch.setattr("genpipe.service.get_temporal_client", fake_client)
    repo.reset_receipts()
    spec = _dispatch_spec()

    receipt = await service.start_space_render(spec)

    assert receipt == WorkflowStartReceipt(
        workflow_id="space-render-01J0SPACE", run_id="run-space-1"
    )
    run, arg, kwargs = fake.started[0]
    assert run is SpaceRenderDispatchWorkflow.run
    assert arg is spec
    assert kwargs == {"id": "space-render-01J0SPACE", "task_queue": WORKFLOW_TASK_QUEUE}
    assert await repo.find_task_receipt("01J0SPACE") == receipt
    repo.reset_receipts()


# ---------------------------------------------------------------------------
# 装配：结论 → 产物清单 → 回调报文
# ---------------------------------------------------------------------------


async def _three_camera_result() -> tuple[SpaceRenderDispatchSpec, Any]:
    """三台机位：鸟瞰取景失败、客厅门禁不过重派后过（有 sketch）、卧室直接过（无 sketch）。"""
    spec = _dispatch_spec(cameras=[c.model_dump() for c in three_camera_spec().cameras or []])
    runner = FakeRunner(three_camera_behaviors({}))
    return spec, await run_space_render(spec, runner)


async def test_products_put_realism_as_deliverable_and_routes_as_lineage() -> None:
    spec, result = await _three_camera_result()
    products = space_render_products(spec, result)

    assert [p.product for p in products] == [
        "scene-package",
        *["geometry", "depth", "line", "mask", "mask-index", "sketch", "realism"],
        *["geometry", "depth", "line", "mask", "mask-index", "realism"],
    ]
    scene = products[0]
    assert scene.object_key == SCENE_KEY and scene.content_type == "application/json"
    assert scene.gen_params == {
        "revision_id": "rev-1",
        "evidence": {
            "bucket": "ishome-private",
            "metre_per_unit": 14.27,
            "area_match_ratio": 0.951,
            "heights_source": "mock-default",
        },
    }
    living = {p.product: p for p in products[1:8]}
    assert living["line"].object_key == f"{PREFIX}/cam-room-living/line.png"
    assert living["line"].content_type == "image/png"
    assert living["mask-index"].content_type == "application/json"
    assert living["sketch"].gen_params == {
        "camera_id": "cam-room-living",
        "view_kind": "room",
        "revision_id": "rev-1",
    }
    realism = living["realism"]
    # 键末段只有风格模板 id，机位靠前缀那一段（用户裁决 2026-09-06"现在删掉"）
    assert realism.object_key.endswith("/cam-room-living/realism-modern-minimal.png")
    assert realism.content_type == "image/png"
    assert realism.gen_params["style_template_id"] == "modern-minimal"
    assert realism.gen_params["control_key"].endswith("/cam-room-living/sketch.png")
    assert realism.gen_params["fidelity_score"] == 0.83
    assert realism.gen_params["gate"]["passed"] is True
    assert realism.gen_params["gate_redispatches"] == 1
    assert realism.gen_params["evidence"]["backend_name"] == "wanx-sketch"
    assert realism.gen_params["evidence"]["seed"] == 7
    bedroom_realism = products[-1]
    assert bedroom_realism.gen_params["control_key"].endswith("/cam-room-bedroom/line.png")
    assert bedroom_realism.gen_params["gate_redispatches"] == 0


async def test_task_result_payload_completed_lists_failed_cameras_without_failure() -> None:
    spec, result = await _three_camera_result()
    products = space_render_products(spec, result)
    payload = build_space_render_task_result(
        spec, result, products=products, workflow_id="wf", run_id="run"
    )

    assert set(payload) == {
        "task_id",
        "status",
        "products",
        "failed_cameras",
        "workflow_id",
        "run_id",
    }
    assert payload["task_id"] == "01J0SPACE" and payload["status"] == "completed"
    assert len(payload["products"]) == 14
    assert payload["products"][0] == products[0].model_dump()
    assert [c["camera_id"] for c in payload["failed_cameras"]] == ["cam-bird-dollhouse"]
    failed = payload["failed_cameras"][0]
    assert failed["stage"] == "base-render"
    assert failed["failed_checks"] == [
        "base-render-camera-failed=画面几乎全空（covered_pixel_ratio 0.01）"
    ]
    assert failed["render_keys"] == {} and failed["gate_redispatches"] == 0


async def test_task_result_payload_failed_carries_failure_and_scene_package_lineage() -> None:
    """整户失败：底渲一台都没出来——status=failed、failure 取编排结论、产物只剩已出的场景包，
    失败机位照样单列（五路键在它们自己的 render_keys 里，不当产物回）。"""
    spec = _dispatch_spec(cameras=[c.model_dump() for c in three_camera_spec().cameras or []])
    behaviors = three_camera_behaviors({}) | {
        "base-render": lambda _: {
            "verdict": "failed",
            "violations": [{"check": "base-render-scene-missing", "detail": "场景包取不到"}],
        }
    }
    result = await run_space_render(spec, FakeRunner(behaviors))
    assert result.verdict == "failed"
    products = space_render_products(spec, result)
    payload = build_space_render_task_result(
        spec, result, products=products, workflow_id="wf", run_id="run"
    )

    assert payload["status"] == "failed"
    assert payload["failure"] == {
        "code": "base-render",
        "detail": "base-render-scene-missing=场景包取不到",
    }
    assert [p["product"] for p in payload["products"]] == ["scene-package"]
    assert [c["camera_id"] for c in payload["failed_cameras"]] == [
        "cam-bird-dollhouse",
        "cam-room-living",
        "cam-room-bedroom",
    ]


async def test_task_result_payload_scene_compile_failure_has_no_products() -> None:
    spec = _dispatch_spec()
    behaviors = {
        "scene-compile": lambda _: {
            "verdict": "failed",
            "violations": [{"check": "scene-package-missing-plan", "detail": "输入包没有 plan"}],
        }
    }
    result = await run_space_render(spec, FakeRunner(behaviors))
    payload = build_space_render_task_result(
        spec, result, products=space_render_products(spec, result), workflow_id="wf", run_id="r"
    )
    assert payload["status"] == "failed"
    assert payload["products"] == [] and payload["failed_cameras"] == []
    assert payload["failure"]["code"] == "scene-compile"
    assert payload["failure"]["detail"] == "scene-package-missing-plan=输入包没有 plan"


def test_task_result_payload_folds_other_house_level_checks_into_failure_detail() -> None:
    """整户级失败码不止 failure 那一条时（如回流前记的编排层码），并进 detail——三张图线同口径。"""
    spec = _dispatch_spec()
    from genpipe.render3d_pipeline import SpaceRenderResult

    result = SpaceRenderResult(
        task_id=spec.task_id,
        verdict="failed",
        failed_checks=["scene-compile:x", "space-render:other"],
        failure={"code": "scene-compile", "detail": "x"},
    )
    payload = build_space_render_task_result(
        spec, result, products=[], workflow_id="wf", run_id="r"
    )
    assert payload["failure"] == {"code": "scene-compile", "detail": "x；其余：space-render:other"}


def test_task_product_vocabulary_matches_contract_draft() -> None:
    """产物词与 contracts genpipe.v1 `space_render_product` 逐字一致（产物名＝键的末段）。

    仍叫 draft：**字风格待拍**——这里 kebab-case（`mask-index`），三张图线
    `floorplan_visuals_product` 是 snake_case。拍了统一成哪一种，这一串词就得整批换。"""
    for word in (
        "design-package",
        "scene-package",
        "geometry",
        "depth",
        "line",
        "mask",
        "mask-index",
        "sketch",
        "realism",
    ):
        SpaceRenderTaskProduct(product=word, object_key="k")
    with pytest.raises(ValidationError):
        SpaceRenderTaskProduct(product="mood_image", object_key="k")  # type: ignore[arg-type]
