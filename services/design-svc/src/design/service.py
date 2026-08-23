"""用例层：设计会话编排（Design Orchestrator / Intent Router 落点）。

- Intent Router 前置输入归一化（语音转文字、短时间窗多消息聚合）归本层——
  渠道层不理解语义；
- LLM 一律经 LiteLLM 网关调用，Langfuse 逐会话记成本；
- Patch 校验经 patch_engine + genpipe plan-rule-check activity。
"""

from __future__ import annotations

from design.models import ProjectState
from design.repo import find_project


async def get_project(project_id: str) -> ProjectState:
    """get = 必得（取不到抛异常）。"""
    project = await find_project(project_id)
    if project is None:
        raise KeyError(f"project not found: {project_id}")
    return project
