"""Scorer 端口定义。"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class ScorerResult(BaseModel):
    scorer_id: str
    score: float
    """归一化分值 0.0–1.0；机检门禁按阈值判定。"""
    details: dict[str, float] = Field(default_factory=dict)


class Scorer(Protocol):
    """打分维度端口（一维一实现，scorer_id 入 contracts 注册表）。"""

    scorer_id: str

    def score(self, artifact_key: str) -> ScorerResult: ...
