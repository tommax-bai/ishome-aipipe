"""批注红线：一条依据都没引的打回、引了不存在的 id 的打回、留下的不够即整步失败。

这是空间推理背书通道的第一次实装。机检**只保证依据来自解析产出，不保证推理对不对**——
同数字那条机制：判推理对不对需要一把这里没有的尺子，而"依据是不是编的"有尺子。
"""

from __future__ import annotations

import json

import pytest
from genpipe_worker.floorplan_notes import (
    MAX_NOTE_CHARS,
    PlanNotesError,
    build_user_prompt,
    check_notes,
    write_notes,
)
from genpipe_worker.models import PlanFact, PlanNote

_FACTS = [
    PlanFact(fact_id="plan-daylight-主卧", subject="主卧", statement="主卧的外墙上有 2 处开口"),
    PlanFact(fact_id="plan-shape-阳台", subject="阳台", statement="阳台的长边约是短边的 2.9 倍"),
    PlanFact(fact_id="plan-share-玄关", subject="玄关", statement="玄关占户型内部面积的 5%"),
]
_ROOMS = ["主卧", "阳台", "玄关"]


class _FakeLlm:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    async def complete_text(
        self, model: str, system_prompt: str, user_prompt: str, *, temperature: float = 0.0
    ) -> str:
        self.prompts.append(user_prompt)
        return json.dumps(self.payload, ensure_ascii=False)


def _note(room: str, text: str, cites: list[str]) -> PlanNote:
    return PlanNote(room=room, text=text, cites=cites)


def test_note_citing_nothing_is_rejected() -> None:
    kept, rejected = check_notes([_note("主卧", "主卧很亮", [])], _FACTS, _ROOMS)

    assert kept == []
    assert "一条依据都没引" in rejected[0]


def test_note_citing_an_unknown_fact_is_rejected() -> None:
    """想编一条依据，得先编出一个包里不存在的 id——**不是禁止它编，是编了引不到**。"""
    kept, rejected = check_notes(
        [_note("主卧", "主卧朝南采光足", ["plan-daylight-主卧", "plan-orientation-主卧"])],
        _FACTS,
        _ROOMS,
    )

    assert kept == []
    assert "plan-orientation-主卧" in rejected[0]


def test_note_on_a_room_that_does_not_exist_is_rejected() -> None:
    kept, rejected = check_notes([_note("书房", "书房安静", ["plan-share-玄关"])], _FACTS, _ROOMS)

    assert kept == []
    assert "书房" in rejected[0]


def test_overlong_note_is_rejected() -> None:
    # 批注要挂在房间旁边而不盖住图，长了就挂不住
    kept, rejected = check_notes(
        [_note("主卧", "主" * (MAX_NOTE_CHARS + 1), ["plan-daylight-主卧"])], _FACTS, _ROOMS
    )

    assert kept == []
    assert "挂不住" in rejected[0]


def test_well_grounded_note_is_kept() -> None:
    kept, rejected = check_notes(
        [_note("阳台", "阳台又长又窄，晾晒和洗衣得排着放", ["plan-shape-阳台"])], _FACTS, _ROOMS
    )

    assert [note.room for note in kept] == ["阳台"]
    assert rejected == []


def test_prompt_carries_every_fact_id_and_room() -> None:
    """模型只能引这份清单里的 id，所以这份清单必须完整递进去。"""
    prompt = build_user_prompt(_FACTS, _ROOMS)

    for fact in _FACTS:
        assert fact.fact_id in prompt and fact.statement in prompt
    for room in _ROOMS:
        assert room in prompt


@pytest.mark.asyncio
async def test_too_few_surviving_notes_fails_loud() -> None:
    """半套批注比没有更糟：图上挂着空引线，业主看见的是"这儿本来该有话"。"""
    llm = _FakeLlm({"notes": [{"room": "主卧", "text": "亮", "cites": []}]})

    with pytest.raises(PlanNotesError, match="不足"):
        await write_notes(_FACTS, _ROOMS, llm)


@pytest.mark.asyncio
async def test_empty_fact_list_never_reaches_the_model() -> None:
    llm = _FakeLlm({"notes": []})

    with pytest.raises(PlanNotesError, match="事实清单是空的"):
        await write_notes([], _ROOMS, llm)
    assert llm.prompts == []
