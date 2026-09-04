"""project-svc 客户端：报文对齐 contracts project.v1（snake_case、注册表小写渠道标识），
失败响亮。"""

from __future__ import annotations

import json

import httpx
import pytest
from chat.project_client import (
    ProjectClient,
    ProjectClientError,
    SlotFill,
    channel_type_registry_id,
)
from ishome.common.v1 import channel_type_pb2


def test_channel_type_maps_enum_to_registry_id_without_literals() -> None:
    assert channel_type_registry_id(channel_type_pb2.CHANNEL_TYPE_MOCK) == "mock"
    assert channel_type_registry_id(channel_type_pb2.CHANNEL_TYPE_FEISHU) == "feishu"
    with pytest.raises(ProjectClientError):
        channel_type_registry_id(channel_type_pb2.CHANNEL_TYPE_UNSPECIFIED)
    with pytest.raises(ProjectClientError):
        channel_type_registry_id(999)


def _client(handler: httpx.MockTransport) -> ProjectClient:
    client = ProjectClient("http://project.test/")
    client._client = httpx.AsyncClient(base_url="http://project.test", transport=handler)  # noqa: SLF001
    return client


async def test_find_or_create_and_fill_slots_send_contract_shaped_bodies() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/v1/projects":
            return httpx.Response(
                200,
                json={
                    "project_id": "01PROJ",
                    "current_milestone": "M0",
                    "process_version": "v1",
                    "created": True,
                },
            )
        return httpx.Response(
            200,
            json={
                "project_id": "01PROJ",
                "current_milestone": "M0.5",
                "advanced": True,
                "entered_milestones": ["M0.5"],
                "created_task_ids": ["01TASK"],
            },
        )

    async with _client(httpx.MockTransport(handler)) as client:
        project = await client.find_or_create_project(
            channel_type_pb2.CHANNEL_TYPE_MOCK, "mock:local", "ou_1"
        )
        progress = await client.fill_slots(
            "01PROJ",
            [
                SlotFill("floorplan", "uploads/x/original.png", "observed", "m-1"),
                SlotFill("building_area_sqm", "138", "observed", "m-2", 0.9),
            ],
        )

    assert project.project_id == "01PROJ" and project.created is True
    assert json.loads(seen[0].content) == {
        "owner": {
            "channel_type": "mock",
            "channel_instance": "mock:local",
            "external_user_id": "ou_1",
        }
    }
    assert seen[1].url.path == "/api/v1/projects/01PROJ/slots"
    body = json.loads(seen[1].content)
    assert body["slots"][0] == {
        "slot_key": "floorplan",
        "value": "uploads/x/original.png",
        "cognitive_state": "observed",
        "source_event_id": "m-1",
        "confidence": 1.0,
    }
    assert body["slots"][1]["confidence"] == 0.9
    assert progress.advanced and progress.created_task_ids == ["01TASK"]


async def test_non_2xx_and_transport_errors_fail_loud() -> None:
    async with _client(
        httpx.MockTransport(lambda r: httpx.Response(404, json={"detail": "项目不存在"}))
    ) as client:
        with pytest.raises(ProjectClientError, match="404"):
            await client.fill_slots("nope", [SlotFill("floorplan", "k", "observed")])

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with _client(httpx.MockTransport(down)) as client:
        with pytest.raises(ProjectClientError, match="连不上"):
            await client.find_or_create_project(channel_type_pb2.CHANNEL_TYPE_MOCK, "i", "u")

    async with _client(httpx.MockTransport(lambda r: httpx.Response(200))) as client:
        with pytest.raises(ProjectClientError):
            await client.fill_slots("p", [])
