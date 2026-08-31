"""户型批注：模型拿**带 id 的事实**去推理，产出挂在房间上的短句。

**这一步是空间推理背书通道的第一次实装**（用户裁决 2026-08-30）。机制照搬数字那条：
不判断推理对不对，只保证**依据来自解析产出**——每句必须声明它引用了哪几条 `fact_id`，
机检两条：①引用的 id 必须在事实清单里存在；②一条都没引用的句子直接打回。
**不是禁止模型编，是编了引不到**：它想编一条依据，得先编出一个包里不存在的 id。

**为什么不把图递给它**：批注读的是算好的事实清单。递图等于给它第二个可以照着编的来源，
而事实这一层的整个用处，就是让它只有一个来源。

**句子归模型，判据归代码**（同"判据只判不写"那条裁决的另一面）：这里不给模型"该怎么说"的
替代话术，只给它事实和禁令；说得不合规就打回，换成什么由它自己想。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from pydantic import ValidationError

from genpipe_worker.models import PlanFact, PlanNote

NOTES_MODEL = "floorplan-notes.default"
"""任务级逻辑模型名。物理映射唯一落点是 infra 的网关配置，换模型改配置不改代码。"""

MAX_NOTES = 6
"""一张图上最多挂几条。多了图就成了分析表——第一阶段要的是"有获得感"，不是"信息量大"。"""

MIN_NOTES = 3
"""少于这个数就不算一张说明图了，整步响亮失败。"""

MAX_NOTE_CHARS = 24
"""一条批注最多多少字：它要能挂在房间旁边而不盖住图。"""

_SYSTEM_PROMPT = """\
你在给一张户型图写房间批注，读者是刚拿到自己家户型图的业主。
只输出一个 JSON 对象：{"notes": [{"room": "...", "text": "...", "cites": ["..."]}]}

规则：
- 每条批注挂在一个房间上，`room` 必须逐字取自给定的房间清单
- `text` 是一句话，最多 24 个字，说这间房**因为图上这个事实、所以生活里会怎样**
- `cites` 列出这句话用到的事实 id，**至少一条**，且必须逐字取自给定的事实清单
- **绝不写事实清单里没有的数字**：面积、尺寸、间数、开口数只能来自你引用的那几条事实
- 绝不写精度声明（"误差 ±2%""大约准确"这类），你无从知道
- 绝不承诺施工做法、报价或改造方案
- 说人话，不用"动线""格局""采光"这类行话堆砌；一句只说一件事
- 一共写 3 到 6 条，挑值得说的房间写，不必每间都写
"""


class TextCompletion(Protocol):
    """纯文本补全协议位（实现见 llm_client；测试用桩件）。"""

    async def complete_text(
        self, model: str, system_prompt: str, user_prompt: str, *, temperature: float = 0.0
    ) -> str: ...


class PlanNotesError(Exception):
    """批注产不出来。响亮失败，不给半套批注让图上挂着空引线。"""

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


def build_user_prompt(facts: Sequence[PlanFact], room_names: Sequence[str]) -> str:
    """事实清单 + 房间清单 → 递给模型的那段话。纯函数，顺序固定。"""
    listed = "\n".join(f"- {fact.fact_id}：{fact.statement}" for fact in facts)
    return f"房间清单：{'、'.join(room_names)}\n\n事实清单（只能引用这里的 id）：\n{listed}"


def _parse(raw: str) -> list[PlanNote]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].removesuffix("```").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanNotesError([f"模型没回出 JSON：{raw[:300]}"]) from e
    items = payload.get("notes") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise PlanNotesError([f"回执里没有 notes 数组：{raw[:300]}"])
    notes: list[PlanNote] = []
    for item in items:
        try:
            notes.append(PlanNote.model_validate(item))
        except ValidationError:
            continue  # 形态就不对的那条丢掉，下面按条数判够不够
    return notes


def check_notes(
    notes: Sequence[PlanNote], facts: Sequence[PlanFact], room_names: Sequence[str]
) -> tuple[list[PlanNote], list[str]]:
    """逐条过机检。返回（留下的, 打回原因逐条）。

    **机检只保证依据来自解析产出，不保证推理对不对**——同数字那条机制：不判断数字对不对，
    只保证它来自求值线。判推理对不对需要一把这里没有的尺子；而"依据是不是编的"有尺子。
    """
    known = {fact.fact_id for fact in facts}
    rooms = set(room_names)
    kept: list[PlanNote] = []
    rejected: list[str] = []
    for note in notes:
        if note.room not in rooms:
            rejected.append(f"挂在不存在的房间「{note.room}」上：{note.text}")
            continue
        if not note.cites:
            rejected.append(f"一条依据都没引：{note.text}")
            continue
        missing = [cited for cited in note.cites if cited not in known]
        if missing:
            rejected.append(f"引了不存在的事实 {'、'.join(missing)}：{note.text}")
            continue
        if len(note.text) > MAX_NOTE_CHARS:
            rejected.append(f"超过 {MAX_NOTE_CHARS} 字挂不住：{note.text}")
            continue
        kept.append(note)
    return kept[:MAX_NOTES], rejected


async def write_notes(
    facts: Sequence[PlanFact],
    room_names: Sequence[str],
    client: TextCompletion,
    model: str = NOTES_MODEL,
) -> tuple[list[PlanNote], list[str]]:
    """产一批批注并过机检。返回（留下的, 打回原因）。留下的不够即响亮失败。"""
    if not facts:
        raise PlanNotesError(["事实清单是空的：没有可引的东西，句子必然是编的"])
    raw = await client.complete_text(model, _SYSTEM_PROMPT, build_user_prompt(facts, room_names))
    kept, rejected = check_notes(_parse(raw), facts, room_names)
    if len(kept) < MIN_NOTES:
        raise PlanNotesError([f"过检的批注只有 {len(kept)} 条，不足 {MIN_NOTES} 条", *rejected])
    return kept, rejected
