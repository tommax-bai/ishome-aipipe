"""三维线编排直测（不起 Temporal）：假的派发器按注册名回回执，断言机位分列、门禁重派与入参形态。

回执形态照契约（contracts `activities/registry.md`"三维线与写实化的出入参"节；realism-pass 照
imagegen 实装 e859aa6），只把内容换成假的。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from genpipe import workflows
from genpipe.models import GenerationTaskSpec, TaskQueues
from genpipe.render3d_pipeline import (
    ACTIVITY_BASE_RENDER,
    ACTIVITY_REALISM_PASS,
    ACTIVITY_SCENE_COMPILE,
    MAX_GATE_REDISPATCHES,
    DispatchFailure,
    SpaceCamera,
    SpaceRenderSpec,
    is_gate_rejection,
    partition_base_render,
    realism_request,
    resolve_view_kind,
    run_space_render,
    space_render_spec_from_task,
    violation_checks,
)
from genpipe.workflows import fold_space_render_result

PREFIX = "uploads/" + "a" * 64 + "/render3d/rev-1"
SCENE_KEY = f"{PREFIX}/scene-package.json"
QUEUES = TaskQueues(render3d="q-render3d", imagegen="q-imagegen")

MockImpl = Callable[[Any], dict[str, Any]]


class FakeRunner:
    """StepRunner 的测试实现：按注册名回回执、记派发日志；没 mock 的按派发失败抛。"""

    def __init__(self, behaviors: dict[str, MockImpl]) -> None:
        self.behaviors = behaviors
        self.log: list[tuple[str, Any, str]] = []

    async def __call__(self, activity_name: str, arg: Any, task_queue: str) -> dict[str, Any]:
        self.log.append((activity_name, arg, task_queue))
        impl = self.behaviors.get(activity_name)
        if impl is None:
            raise DispatchFailure(activity_name, "ApplicationError: stub")
        return impl(arg)

    def calls(self, activity_name: str) -> list[Any]:
        return [arg for name, arg, _ in self.log if name == activity_name]


def render_entry(
    camera_id: str, *, sketch: bool = False, kind: str | None = None
) -> dict[str, Any]:
    base = f"{PREFIX}/{camera_id}"
    entry: dict[str, Any] = {
        "camera_id": camera_id,
        "geometry_key": f"{base}/geometry.png",
        "depth_key": f"{base}/depth.png",
        "line_key": f"{base}/line.png",
        "mask_key": f"{base}/mask.png",
        "mask_index_key": f"{base}/mask-index.json",
    }
    if sketch:
        entry["sketch_key"] = f"{base}/sketch.png"
    if kind is not None:
        entry["view_kind"] = kind
    return entry


def scene_compile_ok(_: Any) -> dict[str, Any]:
    return {
        "verdict": "ok",
        "scene_package_key": SCENE_KEY,
        "bucket": "ishome-private",
        "mm_per_unit": 14270.0,
        "area_match_ratio": 0.951,
        "heights_source": "mock-default",
    }


def gate_rejected(camera_id: str) -> dict[str, Any]:
    """imagegen 门禁判了不过的回执形态：failed + fidelity-gate-failed + gate 与自证数照带、
    无图键。"""
    return {
        "verdict": "failed",
        "violations": [
            {"check": "fidelity-gate-failed", "detail": "保真度 0.4100 低于下限 0.6000"}
        ],
        "camera_id": camera_id,
        "backend_name": "wanx-sketch",
        "seed": 7,
        "fidelity_score": 0.41,
        "prompt_sha256": "deadbeef",
        "gate": {
            "fidelity_score": 0.41,
            "min_fidelity_score": 0.6,
            "judged": True,
            "passed": False,
            "reason": "保真度 0.4100 低于下限 0.6000",
        },
    }


def realism_ok(arg: Any) -> dict[str, Any]:
    """imagegen realism-pass 的成功回执（mock）。

    写实图键＝`{控制稿前缀}/realism-{style_template_id}.{ext}`——前缀末段已经是机位目录，文件名
    里不再重复 `camera_id`（用户裁决 2026-09-06"现在删掉"）。键由 imagegen 铸，本仓只原样收下。
    """
    source = arg.get("sketchKey") or arg["lineKey"]
    camera_id = arg["cameraId"]
    prefix = source.rsplit("/", 1)[0]  # 形如 {PREFIX}/cam-room-living：机位段在前缀里
    return {
        "verdict": "ok",
        "image_object_key": f"{prefix}/realism-{arg['styleTemplateId']}.png",
        "bucket": "ishome-private",
        "content_type": "image/png",
        "source_object_key": source,
        "camera_id": camera_id,
        "view_kind": arg["viewKind"],
        "backend_name": "wanx-sketch",
        "seed": arg.get("seed", 12345),
        "fidelity_score": 0.83,
        "elapsed_seconds": 31.2,
        "prompt_sha256": "cafebabe",
        "gate": {
            "fidelity_score": 0.83,
            "min_fidelity_score": 0.6,
            "judged": True,
            "passed": True,
            "reason": None,
        },
    }


THREE_CAMERAS = [
    SpaceCamera(camera_id="cam-bird-dollhouse", view_kind="bird"),
    SpaceCamera(camera_id="cam-room-living", view_kind="room", seed=7),
    SpaceCamera(camera_id="cam-room-bedroom"),
]


def three_camera_spec(**overrides: Any) -> SpaceRenderSpec:
    fields: dict[str, Any] = {
        "task_id": "01J0SPACE",
        "design_package_key": f"{PREFIX}/design-package.json",
        "revision_id": "rev-1",
        "cameras": THREE_CAMERAS,
        "style_template_id": "modern-minimal",
        "queues": QUEUES,
    }
    fields.update(overrides)
    return SpaceRenderSpec(**fields)


def three_camera_behaviors(attempts: dict[str, int]) -> dict[str, MockImpl]:
    """鸟瞰机位取景失败（violation 报出、不进 renders）；客厅门禁一次不过再派即过；卧室直接过。"""

    def base_render(arg: Any) -> dict[str, Any]:
        return {
            "verdict": "ok",
            "bucket": "ishome-private",
            "width_px": arg["widthPx"],
            "height_px": arg["heightPx"],
            "renders": [
                render_entry("cam-room-living", sketch=True),
                render_entry("cam-room-bedroom", kind="room"),
            ],
            "violations": [
                {
                    "check": "base-render-camera-failed",
                    "detail": "画面几乎全空（covered_pixel_ratio 0.01）",
                    "camera_id": "cam-bird-dollhouse",
                }
            ],
        }

    def realism(arg: Any) -> dict[str, Any]:
        camera_id = arg["cameraId"]
        attempts[camera_id] = attempts.get(camera_id, 0) + 1
        if camera_id == "cam-room-living" and attempts[camera_id] == 1:
            return gate_rejected(camera_id)
        return realism_ok(arg)

    return {
        ACTIVITY_SCENE_COMPILE: scene_compile_ok,
        ACTIVITY_BASE_RENDER: base_render,
        ACTIVITY_REALISM_PASS: realism,
    }


async def test_three_cameras_one_lost_one_redispatched_one_straight_through() -> None:
    attempts: dict[str, int] = {}
    runner = FakeRunner(three_camera_behaviors(attempts))
    result = await run_space_render(three_camera_spec(), runner)

    assert result.verdict == "ok"
    assert result.failure is None
    assert result.scene_package_key == SCENE_KEY
    # 自证数原样带回、不判；键本身不重复进自证数
    assert result.scene_evidence["mm_per_unit"] == 14270.0
    assert result.scene_evidence["heights_source"] == "mock-default"
    assert "scene_package_key" not in result.scene_evidence
    assert result.base_render_evidence == {
        "bucket": "ishome-private",
        "width_px": 1024,
        "height_px": 768,
    }

    # 成功机位：两台；失败机位单列：鸟瞰那台，取景阶段，理由是回执 violation 原话
    assert [r.camera_id for r in result.renders] == ["cam-room-living", "cam-room-bedroom"]
    assert [c.camera_id for c in result.failed_cameras] == ["cam-bird-dollhouse"]
    lost = result.failed_cameras[0]
    assert lost.stage == "base-render"
    assert lost.failed_checks == [
        "base-render-camera-failed=画面几乎全空（covered_pixel_ratio 0.01）"
    ]
    assert lost.violations[0]["camera_id"] == "cam-bird-dollhouse"
    assert lost.render_keys == {}

    living, bedroom = result.renders
    # 客厅：sketch 有就用 sketch；门禁一次不过，同样入参再派一次即过，次数记在结论里
    assert living.control_key.endswith("/cam-room-living/sketch.png")
    assert living.sketch_key is not None and living.gate_redispatches == 1
    assert living.image_object_key.endswith("/cam-room-living/realism-modern-minimal.png")
    assert living.gate["passed"] is True and living.fidelity_score == 0.83
    assert living.evidence["backend_name"] == "wanx-sketch"
    assert living.evidence["prompt_sha256"] == "cafebabe"
    assert "image_object_key" not in living.evidence and "gate" not in living.evidence
    # 卧室：没有 sketch 用 line；一次过
    assert bedroom.control_key.endswith("/cam-room-bedroom/line.png")
    assert bedroom.sketch_key is None and bedroom.gate_redispatches == 0
    assert bedroom.view_kind == "room"

    # 派发形态：场景编译一次、底渲一次带全部机位、写实化 2 + 1 次；队列各归各的
    assert [name for name, _, _ in runner.log[:2]] == [ACTIVITY_SCENE_COMPILE, ACTIVITY_BASE_RENDER]
    assert runner.log[0][1] == {
        "designPackageKey": f"{PREFIX}/design-package.json",
        "revisionId": "rev-1",
    }
    assert runner.log[0][2] == "q-render3d"
    assert runner.log[1][1] == {
        "scenePackageKey": SCENE_KEY,
        "cameraIds": ["cam-bird-dollhouse", "cam-room-living", "cam-room-bedroom"],
        "widthPx": 1024,
        "heightPx": 768,
    }
    assert runner.log[1][2] == "q-render3d"
    assert attempts == {"cam-room-living": 2, "cam-room-bedroom": 1}
    realism_calls = runner.calls(ACTIVITY_REALISM_PASS)
    assert len(realism_calls) == 3
    assert all(
        queue == "q-imagegen" for name, _, queue in runner.log if name == ACTIVITY_REALISM_PASS
    )
    living_calls = [c for c in realism_calls if c["cameraId"] == "cam-room-living"]
    # 重派入参逐字相同（seed 也不换）；sketch 与 line 二选一只给一个；视角取 spec（回执没给）
    assert living_calls[0] == living_calls[1]
    assert living_calls[0] == {
        "styleTemplateId": "modern-minimal",
        "viewKind": "room",
        "cameraId": "cam-room-living",
        "sketchKey": f"{PREFIX}/cam-room-living/sketch.png",
        "seed": 7,
    }
    bedroom_call = next(c for c in realism_calls if c["cameraId"] == "cam-room-bedroom")
    # 卧室 spec 没给视角，取回执的；没给 seed 就一个字段都不带
    assert bedroom_call["viewKind"] == "room" and "seed" not in bedroom_call
    assert (
        bedroom_call["lineKey"].endswith("/cam-room-bedroom/line.png")
        and "sketchKey" not in bedroom_call
    )
    # 一档：三条入参都不带档位
    assert all("renderTier" not in arg and "render_tier" not in arg for _, arg, _ in runner.log)


async def test_view_kind_unknown_is_a_failed_camera_not_a_guess() -> None:
    """回执与 spec 都没给视角：这台机位按失败记，不派写实化，不按 id 字样猜。"""
    behaviors = three_camera_behaviors({}) | {
        ACTIVITY_BASE_RENDER: lambda _: {
            "verdict": "ok",
            "renders": [
                render_entry("cam-room-bedroom"),
                render_entry("cam-room-living", kind="room"),
            ],
        }
    }
    runner = FakeRunner(behaviors)
    spec = three_camera_spec(
        cameras=[
            SpaceCamera(camera_id="cam-room-bedroom"),
            SpaceCamera(camera_id="cam-room-living"),
        ]
    )
    result = await run_space_render(spec, runner)

    assert result.verdict == "ok"
    assert [r.camera_id for r in result.renders] == ["cam-room-living"]
    failed = result.failed_cameras[0]
    assert failed.camera_id == "cam-room-bedroom" and failed.stage == "realism-pass"
    assert failed.failed_checks == ["missing-view-kind"]
    assert set(failed.render_keys) == {
        "geometry_key",
        "depth_key",
        "line_key",
        "mask_key",
        "mask_index_key",
    }
    # 卧室一次写实化都不派；客厅照常（这套 mock 里它门禁一次不过再派即过，故两次）
    assert [c["cameraId"] for c in runner.calls(ACTIVITY_REALISM_PASS)] == [
        "cam-room-living",
        "cam-room-living",
    ]


async def test_gate_rejection_redispatches_up_to_the_limit_then_fails_that_camera() -> None:
    attempts: dict[str, int] = {}

    def always_rejected(arg: Any) -> dict[str, Any]:
        attempts[arg["cameraId"]] = attempts.get(arg["cameraId"], 0) + 1
        return gate_rejected(arg["cameraId"])

    behaviors = three_camera_behaviors({}) | {ACTIVITY_REALISM_PASS: always_rejected}
    runner = FakeRunner(behaviors)
    result = await run_space_render(three_camera_spec(), runner)

    # 两台都被门禁拦到上限、鸟瞰取景失败：一张都没出来才是整户失败
    assert result.verdict == "failed"
    assert result.failure == {"code": "space-render", "detail": "all-cameras-failed"}
    assert result.renders == []
    assert result.scene_package_key == SCENE_KEY  # 血缘照留
    assert attempts == {
        "cam-room-living": 1 + MAX_GATE_REDISPATCHES,
        "cam-room-bedroom": 1 + MAX_GATE_REDISPATCHES,
    }
    by_id = {c.camera_id: c for c in result.failed_cameras}
    assert set(by_id) == {"cam-bird-dollhouse", "cam-room-living", "cam-room-bedroom"}
    living = by_id["cam-room-living"]
    assert living.stage == "realism-pass"
    assert living.gate_redispatches == MAX_GATE_REDISPATCHES
    assert living.gate is not None and living.gate["passed"] is False
    assert living.failed_checks == ["fidelity-gate-failed=保真度 0.4100 低于下限 0.6000"]
    assert living.render_keys["sketch_key"].endswith("/sketch.png")


async def test_backend_failure_is_not_redispatched() -> None:
    """不是门禁不过（网关拒绝 / 线稿解不成图）：不重派——每重派一次都要再花一张图的钱。"""
    attempts: dict[str, int] = {}

    def backend_down(arg: Any) -> dict[str, Any]:
        attempts[arg["cameraId"]] = attempts.get(arg["cameraId"], 0) + 1
        if arg["cameraId"] == "cam-room-bedroom":
            return realism_ok(arg)
        return {
            "verdict": "failed",
            "violations": [{"check": "realism-failed", "detail": "网关 503"}],
        }

    runner = FakeRunner(three_camera_behaviors({}) | {ACTIVITY_REALISM_PASS: backend_down})
    result = await run_space_render(three_camera_spec(), runner)

    assert result.verdict == "ok"
    assert [r.camera_id for r in result.renders] == ["cam-room-bedroom"]
    assert attempts == {"cam-room-living": 1, "cam-room-bedroom": 1}
    living = next(c for c in result.failed_cameras if c.camera_id == "cam-room-living")
    assert living.failed_checks == ["realism-failed=网关 503"] and living.gate_redispatches == 0
    assert living.gate is None


async def test_unjudged_gate_never_redispatches() -> None:
    """下限没配（judged=false）imagegen 不会以 failed 回；就算回了也不重派——没判就没有"不过"。"""
    attempts: dict[str, int] = {}

    def unjudged_failed(arg: Any) -> dict[str, Any]:
        attempts[arg["cameraId"]] = attempts.get(arg["cameraId"], 0) + 1
        return {
            "verdict": "failed",
            "violations": [{"check": "fidelity-gate-failed", "detail": "量分抛错"}],
            "gate": {"judged": False, "passed": None, "reason": None},
        }

    runner = FakeRunner(three_camera_behaviors({}) | {ACTIVITY_REALISM_PASS: unjudged_failed})
    result = await run_space_render(three_camera_spec(), runner)
    assert result.verdict == "failed"
    assert attempts == {"cam-room-living": 1, "cam-room-bedroom": 1}


async def test_scene_compile_failure_fails_the_whole_task_before_base_render() -> None:
    runner = FakeRunner(
        three_camera_behaviors({})
        | {
            ACTIVITY_SCENE_COMPILE: lambda _: {
                "verdict": "failed",
                "violations": [{"check": "design-package-missing-scale", "detail": "没有建筑面积"}],
            }
        }
    )
    result = await run_space_render(three_camera_spec(), runner)
    assert result.verdict == "failed"
    assert result.failure == {
        "code": "scene-compile",
        "detail": "design-package-missing-scale=没有建筑面积",
    }
    assert result.failed_checks == ["scene-compile:design-package-missing-scale=没有建筑面积"]
    assert result.scene_package_key is None
    assert [name for name, _, _ in runner.log] == [ACTIVITY_SCENE_COMPILE]


async def test_scene_compile_ok_without_key_is_a_failure() -> None:
    """verdict=ok 却没说场景包落在哪：包没落地却回报成功，按失败处理。"""
    runner = FakeRunner(
        three_camera_behaviors({}) | {ACTIVITY_SCENE_COMPILE: lambda _: {"verdict": "ok"}}
    )
    result = await run_space_render(three_camera_spec(), runner)
    assert result.verdict == "failed"
    assert result.failure == {"code": "scene-compile", "detail": "missing-scene-package-key"}


async def test_base_render_dispatch_failure_keeps_scene_lineage() -> None:
    behaviors = three_camera_behaviors({})
    del behaviors[ACTIVITY_BASE_RENDER]  # 没 mock ＝ 派发失败（重试耗尽）
    runner = FakeRunner(behaviors)
    result = await run_space_render(three_camera_spec(), runner)
    assert result.verdict == "failed"
    assert result.failure == {"code": "base-render", "detail": "ApplicationError: stub"}
    assert result.scene_package_key == SCENE_KEY
    assert result.scene_evidence["area_match_ratio"] == 0.951
    assert runner.calls(ACTIVITY_REALISM_PASS) == []


async def test_base_render_with_no_renders_fails_and_lists_every_requested_camera() -> None:
    runner = FakeRunner(
        three_camera_behaviors({})
        | {
            ACTIVITY_BASE_RENDER: lambda _: {
                "verdict": "failed",
                "violations": [{"check": "scene-package-missing", "detail": "桶里没有"}],
            }
        }
    )
    result = await run_space_render(three_camera_spec(), runner)
    assert result.verdict == "failed"
    assert result.failure == {"code": "base-render", "detail": "scene-package-missing=桶里没有"}
    # 点名的三台都记为取景失败，理由是那条没带 camera_id 的 violation（回执不区分，编排不替它区分）
    assert [c.camera_id for c in result.failed_cameras] == [
        "cam-bird-dollhouse",
        "cam-room-living",
        "cam-room-bedroom",
    ]
    assert all(c.failed_checks == ["scene-package-missing=桶里没有"] for c in result.failed_cameras)
    assert runner.calls(ACTIVITY_REALISM_PASS) == []


async def test_base_render_failed_verdict_with_partial_renders_still_realizes_the_rest() -> None:
    """底渲整体 verdict=failed 但有机位出齐了：出齐的照走写实化，没出的单列。"""
    runner = FakeRunner(
        three_camera_behaviors({})
        | {
            ACTIVITY_BASE_RENDER: lambda _: {
                "verdict": "failed",
                "renders": [render_entry("cam-room-bedroom", kind="room")],
                "violations": [
                    {"check": "base-render-camera-failed", "detail": "机位不在场景包里"}
                ],
            }
        }
    )
    result = await run_space_render(three_camera_spec(), runner)
    assert result.verdict == "ok"
    assert [r.camera_id for r in result.renders] == ["cam-room-bedroom"]
    assert sorted(c.camera_id for c in result.failed_cameras) == [
        "cam-bird-dollhouse",
        "cam-room-living",
    ]


async def test_all_cameras_mode_takes_view_kind_from_receipt_only() -> None:
    """cameras=None：底渲 cameraIds=None、机位集来自回执；缺席机位只能从带 camera_id 的
    violation 知道。"""
    runner = FakeRunner(
        three_camera_behaviors({})
        | {
            ACTIVITY_BASE_RENDER: lambda arg: {
                "verdict": "ok",
                "renders": [
                    render_entry("cam-bird-dollhouse", kind="bird"),
                    render_entry("cam-room-living"),
                ],
                "violations": [{"check": "x", "detail": "y", "camera_id": "cam-room-study"}],
            }
        }
    )
    result = await run_space_render(three_camera_spec(cameras=None), runner)
    assert runner.log[1][1]["cameraIds"] is None
    assert result.verdict == "ok"
    assert [r.camera_id for r in result.renders] == ["cam-bird-dollhouse"]
    assert result.renders[0].view_kind == "bird"
    by_id = {c.camera_id: c for c in result.failed_cameras}
    assert by_id["cam-room-living"].failed_checks == ["missing-view-kind"]
    assert by_id["cam-room-study"].stage == "base-render"
    assert by_id["cam-room-study"].failed_checks == ["x=y"]


async def test_realism_ok_without_image_key_is_a_failed_camera() -> None:
    runner = FakeRunner(
        three_camera_behaviors({})
        | {ACTIVITY_REALISM_PASS: lambda _: {"verdict": "ok", "gate": {}}}
    )
    result = await run_space_render(three_camera_spec(), runner)
    assert result.verdict == "failed"
    living = next(c for c in result.failed_cameras if c.camera_id == "cam-room-living")
    assert living.failed_checks == ["missing-image-object-key"]


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def test_partition_marks_incomplete_five_routes_as_failed() -> None:
    entry = render_entry("cam-a")
    del entry["depth_key"]
    complete, failed = partition_base_render(
        {"verdict": "ok", "renders": [entry, render_entry("cam-b")]},
        [SpaceCamera(camera_id="cam-a"), SpaceCamera(camera_id="cam-b")],
    )
    assert [keys["camera_id"] for keys in complete] == ["cam-b"]
    assert failed[0].camera_id == "cam-a" and failed[0].failed_checks == ["missing-depth-key"]


def test_partition_absent_camera_without_any_violation_gets_orchestration_code() -> None:
    _, failed = partition_base_render(
        {"verdict": "ok", "renders": [render_entry("cam-a")]},
        [SpaceCamera(camera_id="cam-a"), SpaceCamera(camera_id="cam-b")],
    )
    assert failed[0].camera_id == "cam-b"
    assert failed[0].failed_checks == ["missing-from-renders"]


def test_is_gate_rejection_reads_gate_first_then_violation_check() -> None:
    assert is_gate_rejection({"gate": {"judged": True, "passed": False}})
    assert not is_gate_rejection({"gate": {"judged": False, "passed": None}})
    assert not is_gate_rejection({"gate": {"judged": True, "passed": True}})
    # gate 缺席时看 violations.check
    assert is_gate_rejection({"violations": [{"check": "fidelity-gate-failed", "detail": ""}]})
    assert not is_gate_rejection({"violations": [{"check": "realism-failed", "detail": ""}]})
    # gate 在但 judged=false，violations 说 fidelity-gate-failed 也不算（没判就没有"不过"）
    assert not is_gate_rejection(
        {
            "gate": {"judged": False, "passed": None},
            "violations": [{"check": "fidelity-gate-failed", "detail": ""}],
        }
    )


def test_resolve_view_kind_prefers_receipt_then_spec_never_guesses() -> None:
    requested = SpaceCamera(camera_id="cam-bird-dollhouse", view_kind="room")
    assert resolve_view_kind({"view_kind": "bird"}, requested) == "bird"
    assert resolve_view_kind({"kind": "bird"}, requested) == "bird"
    assert resolve_view_kind({"view_kind": "aerial"}, requested) == "room"
    assert resolve_view_kind({}, SpaceCamera(camera_id="cam-bird-dollhouse")) is None
    assert resolve_view_kind({}, None) is None


def test_realism_request_picks_sketch_over_line_and_omits_absent_seed() -> None:
    keys = {"camera_id": "cam-a", "line_key": "l.png", "sketch_key": "s.png"}
    request = realism_request(keys, view_kind="room", style_template_id="modern-minimal", seed=None)
    assert request == {
        "styleTemplateId": "modern-minimal",
        "viewKind": "room",
        "cameraId": "cam-a",
        "sketchKey": "s.png",
    }
    without_sketch = realism_request(
        {"camera_id": "cam-a", "line_key": "l.png"},
        view_kind="bird",
        style_template_id="modern-minimal",
        seed=3,
    )
    assert without_sketch["lineKey"] == "l.png" and "sketchKey" not in without_sketch
    assert without_sketch["seed"] == 3


def test_violation_checks_flatten_or_name_the_missing_reason() -> None:
    assert violation_checks({"verdict": "failed"}) == ["failed-without-violations"]
    assert violation_checks({"violations": [{"check": "a", "detail": "x"}, {"check": "b"}]}) == [
        "a=x",
        "b=",
    ]


def test_activity_names_match_workflows_constants() -> None:
    """两处字面量必须相同：注册名是线上协议，本模块不能 import workflows 才各写一份。"""
    assert ACTIVITY_SCENE_COMPILE == workflows.ACTIVITY_SCENE_COMPILE
    assert ACTIVITY_BASE_RENDER == workflows.ACTIVITY_BASE_RENDER
    assert ACTIVITY_REALISM_PASS == workflows.ACTIVITY_REALISM_PASS


def test_spec_from_task_params_rejects_unknown_fields_and_builds_cameras() -> None:
    queues = TaskQueues(render3d="q")
    spec = space_render_spec_from_task(
        "t-1",
        {
            "design_package_key": "k",
            "revision_id": "rev-1",
            "style_template_id": "modern-minimal",
            "cameras": [{"camera_id": "cam-a", "view_kind": "bird", "seed": 1}],
            "width_px": 640,
        },
        queues,
    )
    assert spec.task_id == "t-1" and spec.queues.render3d == "q"
    assert spec.cameras == [SpaceCamera(camera_id="cam-a", view_kind="bird", seed=1)]
    assert (spec.width_px, spec.height_px) == (640, 768)
    # 骨架期那套参数名（deep_revision_id / camera_id）不认：多一个字段就拒收，不静默丢
    with pytest.raises(ValueError, match="deep_revision_id"):
        space_render_spec_from_task(
            "t-2", {"deep_revision_id": "deep-1", "camera_id": "cam-1"}, queues
        )
    with pytest.raises(ValueError, match="view_kind"):
        space_render_spec_from_task(
            "t-3",
            {
                "design_package_key": "k",
                "revision_id": "r",
                "style_template_id": "s",
                "cameras": [{"camera_id": "cam-a", "view_kind": "aerial"}],
            },
            queues,
        )


async def test_fold_into_generation_task_result_keeps_image_keys_and_camera_failures() -> None:
    runner = FakeRunner(three_camera_behaviors({}))
    result = await run_space_render(three_camera_spec(), runner)
    folded = fold_space_render_result(result)
    assert folded.verdict == "passed"
    assert folded.task_id == "01J0SPACE"
    # 键末段不再带 camera_id，两台机位靠前缀里的机位目录区分（用户裁决 2026-09-06"现在删掉"）
    assert ["/".join(key.rsplit("/", 2)[1:]) for key in folded.artifact_ids] == [
        "cam-room-living/realism-modern-minimal.png",
        "cam-room-bedroom/realism-modern-minimal.png",
    ]
    assert folded.failed_checks == [
        "cam-bird-dollhouse:base-render:"
        "base-render-camera-failed=画面几乎全空（covered_pixel_ratio 0.01）"
    ]


def test_generation_task_spec_still_accepts_scene_compile_type() -> None:
    """任务类型闭集未动：scene-compile 仍是三维线的路由词。"""
    spec = GenerationTaskSpec(task_id="t", task_type="scene-compile", params={})
    assert spec.task_type == "scene-compile"
