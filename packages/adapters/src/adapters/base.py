"""提示词/出入参变换端口。"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class ModelTaskInput(BaseModel):
    """一次模型任务调用的统一输入：逻辑模型名 + 受控载荷。

    模板/实例分离纪律：户型专属内容（房间表、批注、标题）来自 PreliminaryPlan
    槽位填充，由系统组装后作为受控文字载荷传入，不手写进 prompt。
    """

    logical_model: str
    """任务级逻辑模型名 `{activity}.{variant}`，物理 model_id 映射在 LiteLLM 配置。"""
    slots: dict[str, str] = Field(default_factory=dict)


class PromptTransform(Protocol):
    """单个模型家族的提示词组装差异（仅此，不做 API 适配）。"""

    def to_prompt(self, task: ModelTaskInput) -> str: ...
