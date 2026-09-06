"""集成冒烟：真连本地 Temporal 跑三维线（SpaceRenderWorkflow 与 GenerationTaskWorkflow 的
scene-compile 分支）。做法照 test_temporal_integration：mock activity 注册到唯一随机队列，
服务器不可达整体 skip。三台机位：一台取景失败、一台门禁不过重派一次后过、一台直接过。
"""

from __future__ import annotations

import uuid
from typing import Any

from genpipe.models import GenerationTaskSpec, TaskQueues
from genpipe.render3d_pipeline import SpaceCamera, SpaceRenderSpec
from genpipe.workflows import GenerationTaskWorkflow, SpaceRenderWorkflow
from temporalio.client import Client
from temporalio.worker import Worker
from test_render3d_pipeline import PREFIX, SCENE_KEY, three_camera_behaviors
from test_temporal_integration import CallLog, MockImpl, _client_or_skip, _make_mock_activities

CAMERAS = [
    SpaceCamera(camera_id="cam-bird-dollhouse", view_kind="bird"),
    SpaceCamera(camera_id="cam-room-living", view_kind="room", seed=7),
    SpaceCamera(camera_id="cam-room-bedroom"),
]


def _queues(queue: str) -> TaskQueues:
    return TaskQueues(genpipe=queue, render2d=queue, imagegen=queue, render3d=queue)


async def _worker(
    client: Client, queue: str, behaviors: dict[str, MockImpl], log: CallLog
) -> Worker:
    return Worker(
        client,
        task_queue=queue,
        workflows=[SpaceRenderWorkflow, GenerationTaskWorkflow],
        activities=_make_mock_activities(behaviors, log),
    )


async def test_space_render_workflow_lists_failed_camera_and_redispatches_gate_rejection() -> None:
    client = await _client_or_skip()
    log: CallLog = []
    attempts: dict[str, int] = {}
    queue = f"it-{uuid.uuid4().hex}"
    spec = SpaceRenderSpec(
        task_id=uuid.uuid4().hex,
        design_package_key=f"{PREFIX}/design-package.json",
        revision_id="rev-1",
        cameras=CAMERAS,
        style_template_id="modern-minimal",
        queues=_queues(queue),
    )
    async with await _worker(client, queue, three_camera_behaviors(attempts), log):
        result = await client.execute_workflow(
            SpaceRenderWorkflow.run, spec, id=f"it-space-{spec.task_id}", task_queue=queue
        )

    assert result.verdict == "ok", result
    assert result.scene_package_key == SCENE_KEY
    assert [r.camera_id for r in result.renders] == ["cam-room-living", "cam-room-bedroom"]
    assert [c.camera_id for c in result.failed_cameras] == ["cam-bird-dollhouse"]
    assert result.failed_cameras[0].stage == "base-render"
    living, bedroom = result.renders
    assert living.gate_redispatches == 1 and living.control_key.endswith("/sketch.png")
    assert bedroom.gate_redispatches == 0 and bedroom.control_key.endswith("/line.png")
    assert living.image_object_key.endswith("/cam-room-living/realism-modern-minimal.png")
    assert living.gate["passed"] is True and living.evidence["backend_name"] == "wanx-sketch"

    dispatched = [name for name, _ in log]
    assert dispatched[:2] == ["scene-compile", "base-render"]
    assert dispatched.count("realism-pass") == 3
    assert attempts == {"cam-room-living": 2, "cam-room-bedroom": 1}
    base_render_arg = log[1][1]
    assert base_render_arg["scenePackageKey"] == SCENE_KEY
    assert base_render_arg["cameraIds"] == [
        "cam-bird-dollhouse",
        "cam-room-living",
        "cam-room-bedroom",
    ]
    living_calls = [
        arg for name, arg in log if name == "realism-pass" and arg["cameraId"] == "cam-room-living"
    ]
    assert living_calls[0] == living_calls[1] and living_calls[0]["seed"] == 7


async def test_generation_task_scene_compile_branch_folds_into_task_result() -> None:
    """交互侧任务层入口：scene-compile 类型走三维线，结论折成 GenerationTaskResult；
    门禁两个存根 activity 不派（与三张图线同口径）。"""
    client = await _client_or_skip()
    log: CallLog = []
    queue = f"it-{uuid.uuid4().hex}"
    behaviors: dict[str, MockImpl] = three_camera_behaviors({}) | {
        "consistency-check": lambda _: {"passed": True},
        "compliance-check": lambda _: {"passed": True},
    }
    params: dict[str, Any] = {
        "design_package_key": f"{PREFIX}/design-package.json",
        "revision_id": "rev-1",
        "cameras": [camera.model_dump() for camera in CAMERAS],
        "style_template_id": "modern-minimal",
    }
    spec = GenerationTaskSpec(
        task_id=uuid.uuid4().hex, task_type="scene-compile", params=params, queues=_queues(queue)
    )
    async with await _worker(client, queue, behaviors, log):
        result = await client.execute_workflow(
            GenerationTaskWorkflow.run, spec, id=f"it-task-{spec.task_id}", task_queue=queue
        )

    assert result.verdict == "passed"
    assert ["/".join(key.rsplit("/", 2)[1:]) for key in result.artifact_ids] == [
        "cam-room-living/realism-modern-minimal.png",
        "cam-room-bedroom/realism-modern-minimal.png",
    ]
    assert result.failed_checks == [
        "cam-bird-dollhouse:base-render:"
        "base-render-camera-failed=画面几乎全空（covered_pixel_ratio 0.01）"
    ]
    dispatched = [name for name, _ in log]
    assert "consistency-check" not in dispatched and "compliance-check" not in dispatched


async def test_generation_task_scene_compile_rejects_skeleton_era_params() -> None:
    """骨架期那套参数（deep_revision_id / camera_id）派进来：派发前拦，一步 activity 都不派。"""
    client = await _client_or_skip()
    log: CallLog = []
    queue = f"it-{uuid.uuid4().hex}"
    spec = GenerationTaskSpec(
        task_id=uuid.uuid4().hex,
        task_type="scene-compile",
        params={"deep_revision_id": "deep-1", "camera_id": "cam-1"},
        queues=_queues(queue),
    )
    async with await _worker(client, queue, three_camera_behaviors({}), log):
        result = await client.execute_workflow(
            GenerationTaskWorkflow.run, spec, id=f"it-task-{spec.task_id}", task_queue=queue
        )
    assert result.verdict == "failed"
    assert result.failed_checks[0].startswith("invalid-space-render-spec:")
    assert log == []
