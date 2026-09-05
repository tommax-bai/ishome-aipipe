"""集成冒烟：真连本地 Temporal 跑三维线派发形态（SpaceRenderDispatchWorkflow）——跑完把结论经
task-result-deliver 送回派发时注入的回调地址。做法照 test_temporal_space_render：mock activity
注册到唯一随机队列，服务器不可达整体 skip。
"""

from __future__ import annotations

import uuid
from typing import Any

from genpipe.models import TaskQueues
from genpipe.space_render_dispatch import SpaceRenderDispatchSpec
from genpipe.workflows import SpaceRenderDispatchWorkflow
from temporalio.client import Client
from temporalio.worker import Worker
from test_render3d_pipeline import PREFIX, SCENE_KEY, three_camera_behaviors
from test_temporal_integration import CallLog, MockImpl, _client_or_skip, _make_mock_activities
from test_temporal_space_render import CAMERAS

CALLBACK_URL = "http://127.0.0.1:1/api/v1/generation-tasks/x/result"


def _spec(queue: str) -> SpaceRenderDispatchSpec:
    return SpaceRenderDispatchSpec(
        task_id=uuid.uuid4().hex,
        design_package_key=f"{PREFIX}/design-package.json",
        revision_id="rev-1",
        cameras=CAMERAS,
        style_template_id="modern-minimal",
        result_callback_url=CALLBACK_URL,
        queues=TaskQueues(genpipe=queue, render2d=queue, imagegen=queue, render3d=queue),
    )


def _deliver_ok(_: Any) -> dict[str, Any]:
    return {"verdict": "ok", "status_code": 200, "receipt": {"accepted": True}}


async def _run(client: Client, behaviors: dict[str, MockImpl], log: CallLog, queue: str) -> Any:
    spec = _spec(queue)
    worker = Worker(
        client,
        task_queue=queue,
        workflows=[SpaceRenderDispatchWorkflow],
        activities=_make_mock_activities(behaviors, log),
    )
    async with worker:
        return await client.execute_workflow(
            SpaceRenderDispatchWorkflow.run,
            spec,
            id=f"it-space-dispatch-{spec.task_id}",
            task_queue=queue,
        )


async def test_space_render_dispatch_delivers_completed_result_with_failed_camera_listed() -> None:
    client = await _client_or_skip()
    log: CallLog = []
    queue = f"it-{uuid.uuid4().hex}"
    behaviors = three_camera_behaviors({}) | {"task-result-deliver": _deliver_ok}
    result = await _run(client, behaviors, log, queue)

    assert result.verdict == "ok", result
    assert result.delivered is True
    assert result.failed_checks == []
    assert [r.camera_id for r in result.renders] == ["cam-room-living", "cam-room-bedroom"]
    assert [p.product for p in result.products][:2] == ["scene-package", "geometry"]
    assert len(result.products) == 14

    dispatched = [name for name, _ in log]
    assert dispatched[-1] == "task-result-deliver"
    deliver_call = log[-1][1]
    assert deliver_call["result_callback_url"] == CALLBACK_URL
    body = deliver_call["result"]
    assert body["task_id"] == result.task_id and body["status"] == "completed"
    assert "failure" not in body
    assert body["workflow_id"] == f"it-space-dispatch-{result.task_id}" and body["run_id"]
    assert [p["product"] for p in body["products"]].count("realism") == 2
    assert body["products"][0]["object_key"] == SCENE_KEY
    assert [c["camera_id"] for c in body["failed_cameras"]] == ["cam-bird-dollhouse"]
    living = next(
        p
        for p in body["products"]
        if p["product"] == "realism" and p["gen_params"]["camera_id"] == "cam-room-living"
    )
    assert living["gen_params"]["gate_redispatches"] == 1
    assert living["gen_params"]["evidence"]["seed"] == 7


async def test_space_render_dispatch_house_failure_still_delivers_failed_result() -> None:
    client = await _client_or_skip()
    log: CallLog = []
    queue = f"it-{uuid.uuid4().hex}"
    behaviors = three_camera_behaviors({}) | {
        "scene-compile": lambda _: {
            "verdict": "failed",
            "violations": [{"check": "scene-package-missing-plan", "detail": "输入包没有 plan"}],
        },
        "task-result-deliver": _deliver_ok,
    }
    result = await _run(client, behaviors, log, queue)

    assert result.verdict == "failed" and result.delivered is True
    assert result.products == [] and result.renders == []
    body = log[-1][1]["result"]
    assert body["status"] == "failed"
    assert body["failure"]["code"] == "scene-compile"
    assert body["products"] == [] and body["failed_cameras"] == []
    assert [name for name, _ in log] == ["scene-compile", "task-result-deliver"]


async def test_space_render_dispatch_records_callback_rejection() -> None:
    """业务侧明确拒收（4xx → activity 回 failed）：结论照样是 ok，但 delivered=False、
    failed_checks 记下回流那一步——没送到不算完。"""
    client = await _client_or_skip()
    log: CallLog = []
    queue = f"it-{uuid.uuid4().hex}"
    behaviors = three_camera_behaviors({}) | {
        "task-result-deliver": lambda _: {
            "verdict": "failed",
            "violations": [{"check": "callback-rejected-404", "detail": "任务不存在"}],
        }
    }
    result = await _run(client, behaviors, log, queue)

    assert result.verdict == "ok" and result.delivered is False
    assert result.failed_checks == ["task-result-deliver:callback-rejected-404=任务不存在"]
    assert len(result.products) == 14
