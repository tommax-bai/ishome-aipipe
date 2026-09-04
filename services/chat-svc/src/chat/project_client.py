"""出站边缘：project-svc REST 客户端（contracts `openapi/project.v1.yaml`，禁手写协议之外的约定）。

会话侧只做两件事：把事实上报、把产物呈现。本模块是"上报"那一跳——会话侧不判里程碑、不建任务，
它把"业主传了户型图（在这个键）、说了建筑面积"这两件事交给业务侧，业务侧判定并派发。

依赖方向（import-linter 锁定）：本模块只依赖运行库（httpx）与 contracts 生成的枚举，不感知上层——
由组合根（grpc_server.serve）注入 service 层的协议位。联调契约：project-svc 默认
http://127.0.0.1:8103（env `PROJECT_HTTP_BASE_URL` 覆盖）。

渠道类型的换算：会话键里是 proto 枚举 int，契约要的是注册表小写标识（feishu / mock …）——
从枚举名机械去前缀转小写，不出现任何渠道字面量。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import httpx
from ishome.common.v1 import channel_type_pb2

DEFAULT_PROJECT_HTTP_BASE_URL = "http://127.0.0.1:8103"
_TIMEOUT_SECONDS = 15.0
"""填槽位那一跳可能连带派发（业务侧同步打编排侧入口），比纯落库长；超过即当故障。"""

_CHANNEL_TYPE_ENUM_PREFIX = "CHANNEL_TYPE_"


class ProjectClientError(Exception):
    """业务侧那一跳没走通（连不上 / 非 2xx / 回执解析不了）——响亮失败，调用方决定怎么对业主说。"""


@dataclass(frozen=True)
class BusinessProject:
    project_id: str
    current_milestone: str
    process_version: str
    created: bool


@dataclass(frozen=True)
class SlotFill:
    """一条 slot_filled（contracts project.v1 `slot_fill`）：值一律字面字符串，认知状态六值小写。"""

    slot_key: str
    value: str
    cognitive_state: str
    source_event_id: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class MilestoneProgress:
    project_id: str
    current_milestone: str
    advanced: bool
    entered_milestones: list[str] = field(default_factory=list)
    created_task_ids: list[str] = field(default_factory=list)


def channel_type_registry_id(channel_type: int) -> str:
    """proto 枚举 int → 注册表小写标识。UNSPECIFIED / 未知值即失败：
    不许把不认识的渠道报给业务侧。"""
    try:
        name = channel_type_pb2.ChannelType.Name(channel_type)
    except ValueError as e:
        raise ProjectClientError(f"渠道类型枚举值不认识：{channel_type}") from e
    if not name.startswith(_CHANNEL_TYPE_ENUM_PREFIX) or name == "CHANNEL_TYPE_UNSPECIFIED":
        raise ProjectClientError(f"渠道类型没有注册表标识：{name}")
    return name[len(_CHANNEL_TYPE_ENUM_PREFIX) :].lower()


class ProjectClient:
    """project.v1 的薄封装（实现 service.BusinessSideGateway 协议）。"""

    def __init__(
        self, base_url: str | None = None, timeout_seconds: float = _TIMEOUT_SECONDS
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("PROJECT_HTTP_BASE_URL", DEFAULT_PROJECT_HTTP_BASE_URL)
        ).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)

    async def find_or_create_project(
        self, channel_type: int, channel_instance: str, external_user_id: str
    ) -> BusinessProject:
        payload = await self._post(
            "/api/v1/projects",
            {
                "owner": {
                    "channel_type": channel_type_registry_id(channel_type),
                    "channel_instance": channel_instance,
                    "external_user_id": external_user_id,
                }
            },
        )
        return BusinessProject(
            project_id=str(payload.get("project_id") or ""),
            current_milestone=str(payload.get("current_milestone") or ""),
            process_version=str(payload.get("process_version") or ""),
            created=bool(payload.get("created")),
        )

    async def fill_slots(self, project_id: str, slots: Sequence[SlotFill]) -> MilestoneProgress:
        if not slots:
            raise ProjectClientError("没有槽位可报")
        payload = await self._post(
            f"/api/v1/projects/{project_id}/slots",
            {
                "slots": [
                    {
                        "slot_key": slot.slot_key,
                        "value": slot.value,
                        "cognitive_state": slot.cognitive_state,
                        "source_event_id": slot.source_event_id,
                        "confidence": slot.confidence,
                    }
                    for slot in slots
                ]
            },
        )
        return MilestoneProgress(
            project_id=str(payload.get("project_id") or project_id),
            current_milestone=str(payload.get("current_milestone") or ""),
            advanced=bool(payload.get("advanced")),
            entered_milestones=[str(m) for m in payload.get("entered_milestones") or []],
            created_task_ids=[str(t) for t in payload.get("created_task_ids") or []],
        )

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=body)
        except httpx.HTTPError as e:
            raise ProjectClientError(f"业务侧连不上（{self.base_url}{path}）：{e}") from e
        if response.status_code >= 400:
            raise ProjectClientError(
                f"业务侧回了 {response.status_code}（{path}）：{response.text[:300]}"
            )
        try:
            payload = response.json()
        except ValueError as e:
            raise ProjectClientError(f"业务侧回执不是 JSON（{path}）：{response.text[:200]}") from e
        if not isinstance(payload, dict):
            raise ProjectClientError(f"业务侧回执形态不对（{path}）：{str(payload)[:200]}")
        return payload

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ProjectClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
