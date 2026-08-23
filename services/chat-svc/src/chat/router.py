"""FastAPI 入站适配：项目查询等（router → service 单向）。

会话消息入站（channel-svc → chat-svc）与确认/Patch 接口以 contracts
`ishome.design.v1` gRPC 契约为准，SDK 生成后在此挂接；禁止手写客户端。
"""

from __future__ import annotations

from fastapi import APIRouter

from chat.models import ProjectState
from chat.service import get_project

router = APIRouter(prefix="/api/v1/design")


@router.get("/projects/{project_id}")
async def get_project_endpoint(project_id: str) -> ProjectState:
    return await get_project(project_id)
