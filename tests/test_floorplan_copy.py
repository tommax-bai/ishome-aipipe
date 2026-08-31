"""获客文案红线：不强制引用（裁决 8-31），但**数字必须在事实清单里**——数字不由 LLM 决定。"""

from __future__ import annotations

import json

import pytest
from genpipe_worker.floorplan_copy import PlanCopyError, check_copy, write_copy
from genpipe_worker.models import PlanCopy, PlanFact

_FACTS = [
    PlanFact(fact_id="plan-rooms", subject="户型", statement="这套户型一共 9 间"),
    PlanFact(fact_id="plan-share-玄关", subject="玄关", statement="玄关占户型内部面积的 5%"),
]


class _FakeLlm:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    async def complete_text(
        self, model: str, system_prompt: str, user_prompt: str, *, temperature: float = 0.0
    ) -> str:
        self.prompts.append(user_prompt)
        return json.dumps(self.payload, ensure_ascii=False)


_TIPS = ["玄关放个换鞋凳", "阳台留一条通道", "厨房把常用的放手边"]


def _copy(
    title: str = "温暖小窝",
    summary: str = "住进来就想赖着不走",
    tips: list[str] | None = None,
) -> PlanCopy:
    return PlanCopy(title=title, summary=summary, tips=list(tips or _TIPS))


def test_copy_without_citations_passes() -> None:
    """获客图不是报告：不逐句要求引用，不因没引就打回。"""
    assert check_copy(_copy(), _FACTS) == []


def test_number_not_in_the_fact_list_is_rejected() -> None:
    # "数字不由 LLM 决定"是全项目红线，不只报告域
    problems = check_copy(_copy(summary="全屋收纳多出 12 处，东西都有家"), _FACTS)

    assert any("12" in problem for problem in problems)


def test_number_that_came_from_a_fact_is_fine() -> None:
    assert check_copy(_copy(summary="9 间房，各归各位"), _FACTS) == []


def test_precision_claim_is_rejected() -> None:
    problems = check_copy(_copy(summary="尺寸误差很小，放心照着买"), _FACTS)

    assert any("精度声明" in problem for problem in problems)


def test_promise_is_rejected() -> None:
    problems = check_copy(_copy(summary="保证住得下一家五口"), _FACTS)

    assert any("承诺" in problem for problem in problems)


def test_overlong_title_is_rejected() -> None:
    problems = check_copy(_copy(title="温" * 30), _FACTS)

    assert any("标题超过" in problem for problem in problems)


@pytest.mark.asyncio
async def test_facts_are_handed_over_as_raw_material() -> None:
    """不给素材的"直接推导"就是编——事实清单照样递过去。"""
    llm = _FakeLlm(_copy().model_dump())

    await write_copy(_FACTS, llm)

    assert "玄关占户型内部面积的 5%" in llm.prompts[0]


@pytest.mark.asyncio
async def test_empty_fact_list_never_reaches_the_model() -> None:
    llm = _FakeLlm({})

    with pytest.raises(PlanCopyError, match="就是编"):
        await write_copy([], llm)
    assert llm.prompts == []
