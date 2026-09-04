"""入站面装配：路由必须真的挂在 ASGI app 上，且 body 原样透传到 service。

存在理由（2026-08-30 真跑打脸）：`router.py` 里三条 POST 路由早已写好，但全仓没有任何
`FastAPI()` 把它挂上去——端点处于"声明态"，`curl` 打过去是连不上，而单测只测纯函数，
一路全绿。**路由写了 ≠ 端点可达**，这条只能由挂载后的真实请求断言。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from genpipe.models import ReportComposeSpec, WorkflowStartReceipt
from genpipe.router import app
from temporalio.exceptions import WorkflowAlreadyStartedError

_PACKAGE: dict[str, Any] = {
    "entitlement": "paid",
    "domains": ["dom-ergonomics"],
    "releases": [{"domain": "ergonomics", "releaseTag": "ergonomics@v8"}],
    "anchors": [],
    "withheldAnchors": [],
    "gaps": [],
    "personasByDomain": {},
    "checksByDomain": {},
    "bannedTermsByDomain": {},
    "anonymousProfile": {"layoutFeatures": {}},
}


def test_report_dispatch_routes_are_mounted() -> None:
    """成文线派发入口在 app 上可达（裁决④ 的 HTTP 通道）。

    断言打在 OpenAPI 面而非 `app.routes` 上：后者随 FastAPI 版本会把 include 进来的
    路由折进嵌套节点，扁平遍历漏判；OpenAPI 是对外真正发布的那张面。
    """
    paths = app.openapi()["paths"]
    assert "post" in paths["/api/v1/genpipe/reports"]
    assert "post" in paths["/api/v1/genpipe/batches"]
    assert "post" in paths["/api/v1/genpipe/tasks"]
    assert "post" in paths["/api/v1/genpipe/floorplan-visuals"]
    assert "get" in paths["/healthz"]


def test_healthz_answers_without_touching_temporal() -> None:
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_report_passes_spec_through_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """求值线送来的 report_id / domains / package 一字不改地到达 service 层。"""
    seen: list[ReportComposeSpec] = []

    async def fake_start(spec: ReportComposeSpec) -> WorkflowStartReceipt:
        seen.append(spec)
        return WorkflowStartReceipt(workflow_id=f"report-compose-{spec.report_id}", run_id="run-1")

    monkeypatch.setattr("genpipe.router.start_report_compose", fake_start)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/genpipe/reports",
            json={
                "report_id": "rpt-2026-08-30-0001",
                "domains": ["dom-ergonomics", "dom-budget"],
                "package": _PACKAGE,
            },
        )

    assert response.status_code == 202
    assert response.json() == {
        "workflow_id": "report-compose-rpt-2026-08-30-0001",
        "run_id": "run-1",
    }
    assert len(seen) == 1
    assert seen[0].report_id == "rpt-2026-08-30-0001"
    assert seen[0].domains == ["dom-ergonomics", "dom-budget"]
    assert seen[0].package == _PACKAGE
    assert seen[0].max_rewrites == 2
    assert seen[0].queues.reportgen == "reportgen-activities"


def test_duplicate_report_id_answers_409(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一 report_id 重复派发即冲突——幂等键在求值线那侧（裁决③）。"""

    async def fake_start(spec: ReportComposeSpec) -> WorkflowStartReceipt:
        raise WorkflowAlreadyStartedError(
            workflow_id=f"report-compose-{spec.report_id}",
            workflow_type="ReportComposeWorkflow",
        )

    monkeypatch.setattr("genpipe.router.start_report_compose", fake_start)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/genpipe/reports",
            json={"report_id": "rpt-dup", "domains": ["dom-budget"], "package": _PACKAGE},
        )

    assert response.status_code == 409
    assert "rpt-dup" in response.json()["detail"]


def test_create_floorplan_visuals_passes_spec_through_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三张图派发入口：task_id / 对象键 / 回调地址一字不改到达 service 层，模板走默认。"""
    from genpipe.models import FloorplanVisualsSpec

    seen: list[FloorplanVisualsSpec] = []

    async def fake_start(spec: FloorplanVisualsSpec) -> WorkflowStartReceipt:
        seen.append(spec)
        return WorkflowStartReceipt(workflow_id=f"floorplan-visuals-{spec.task_id}", run_id="r")

    monkeypatch.setattr("genpipe.router.start_floorplan_visuals", fake_start)
    body = {
        "task_id": "01J0TASK",
        "floorplan_object_key": "uploads/" + "a" * 64 + "/original.png",
        "result_callback_url": "http://127.0.0.1:8103/api/v1/generation-tasks/01J0TASK/result",
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/genpipe/floorplan-visuals", json=body)
    assert response.status_code == 202
    assert response.json()["workflow_id"] == "floorplan-visuals-01J0TASK"
    assert seen[0].floorplan_object_key == body["floorplan_object_key"]
    assert seen[0].result_callback_url == body["result_callback_url"]
    assert seen[0].templates.mood == "cream-journal"
    assert seen[0].templates.style == "lifestyle-notebook-handwritten"


def test_create_floorplan_visuals_conflict_on_duplicate_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from genpipe.models import FloorplanVisualsSpec

    async def already_started(spec: FloorplanVisualsSpec) -> WorkflowStartReceipt:
        raise WorkflowAlreadyStartedError(f"floorplan-visuals-{spec.task_id}", "FloorplanVisuals")

    monkeypatch.setattr("genpipe.router.start_floorplan_visuals", already_started)
    body = {
        "task_id": "01J0DUP",
        "floorplan_object_key": "uploads/" + "b" * 64 + "/original.jpg",
        "result_callback_url": "http://127.0.0.1:8103/x",
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/genpipe/floorplan-visuals", json=body)
    assert response.status_code == 409
