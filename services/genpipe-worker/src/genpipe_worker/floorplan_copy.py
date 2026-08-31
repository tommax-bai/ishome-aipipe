"""获客图上的文案：标题、情绪总结、三块小贴士。

**这一处不走强制引用**（用户裁决 2026-08-31）：三张免费图是获客工具不是严谨报告数据，
文案不逐句声明引用、不因没引就打回。背书通道仍管报告正文与功能说明图的房间批注——
分界看**这句话是不是要被人当依据用**：报告里的话业主会照着买东西、照着画线，
这里的话是让他愿意点开。

**但数字仍然不许编**："数字不由 LLM 决定"是全项目红线不是报告域专有。文案里出现的每个数
都必须在事实清单里出现过，机检可判（数字串是可枚举的）。**事实清单照样递过去当素材**——
"直接推导"要有可推的东西，不给素材的直接推导就是编。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import ValidationError

from genpipe_worker.models import PlanCopy, PlanFact

COPY_MODEL = "floorplan-copy.default"
"""任务级逻辑模型名。与批注那一步分开：批注要引得到事实，文案不要求，两步的判据不同。"""

TIP_COUNT = 3
MAX_TITLE_CHARS = 12
MAX_SUMMARY_CHARS = 40
MAX_TIP_CHARS = 22

_DIGITS = re.compile(r"\d+(?:\.\d+)?")
_BANNED_PATTERNS = (
    ("误差", "自造精度声明"),
    ("精度", "自造精度声明"),
    ("保证", "承诺"),
    ("包您", "承诺"),
    ("报价", "报价承诺"),
    ("施工", "施工承诺"),
)

_SYSTEM_PROMPT = f"""\
你在给一张户型分享图写页面文案，读者是刚拿到自己家户型图的业主。
只输出一个 JSON 对象：{{"title": "...", "summary": "...", "tips": ["...", "...", "..."]}}

- `title`：这个家的名字，最多 {MAX_TITLE_CHARS} 字，温暖、有画面感，**不提具体房间或数字**
- `summary`：一句打中人的话，最多 {MAX_SUMMARY_CHARS} 字，说住进来之后生活会怎样
- `tips`：正好 {TIP_COUNT} 条小贴士，每条最多 {MAX_TIP_CHARS} 字，实用、生活化、可保存

硬规矩：
- **只能用给定事实清单里出现过的数字**；想不起来就别写数，一个数都不写完全可以
- 不写"误差""精度"这类说法，你无从知道
- 不承诺施工做法、报价或效果
- 说人话，不堆"动线""格局""采光"这类行话
"""


class TextCompletion(Protocol):
    """纯文本补全协议位。**与批注那一步各持一份，不共用**——两步同层互不可见是刻意的：
    它们的判据不同（那边要引得到事实，这边不要求），共用一个位置会让"改一处顺手改两处"变得容易。
    同"出站边缘各自持有是既定分层，不是重复"。"""

    async def complete_text(
        self, model: str, system_prompt: str, user_prompt: str, *, temperature: float = 0.0
    ) -> str: ...


class PlanCopyError(Exception):
    """文案产不出来。响亮失败，不给半套让版面空着。"""

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


def build_user_prompt(facts: Sequence[PlanFact]) -> str:
    listed = "\n".join(f"- {fact.statement}" for fact in facts)
    return f"这套户型的事实（写文案的素材，可用可不用）：\n{listed}"


def _known_numbers(facts: Sequence[PlanFact]) -> set[str]:
    return {found for fact in facts for found in _DIGITS.findall(fact.statement)}


def check_copy(copy: PlanCopy, facts: Sequence[PlanFact]) -> list[str]:
    """过机检，返回问题逐条（空即通过）。**只拦编数字与禁语，不拦"说得好不好"**。"""
    problems: list[str] = []
    known = _known_numbers(facts)
    lines = [("标题", copy.title, MAX_TITLE_CHARS), ("总结", copy.summary, MAX_SUMMARY_CHARS)]
    lines += [(f"贴士{'一二三'[at]}", tip, MAX_TIP_CHARS) for at, tip in enumerate(copy.tips[:3])]
    for label, text, limit in lines:
        if len(text) > limit:
            problems.append(f"{label}超过 {limit} 字：{text}")
        for number in _DIGITS.findall(text):
            if number not in known:
                problems.append(
                    f"{label}里的数「{number}」不在事实清单里——数字不由模型决定：{text}"
                )
        for needle, why in _BANNED_PATTERNS:
            if needle in text:
                problems.append(f"{label}里有{why}（「{needle}」）：{text}")
    if len(copy.tips) != TIP_COUNT:
        problems.append(f"贴士要正好 {TIP_COUNT} 条，收到 {len(copy.tips)} 条")
    return problems


async def write_copy(
    facts: Sequence[PlanFact], client: TextCompletion, model: str = COPY_MODEL
) -> PlanCopy:
    """产一份页面文案并过机检。不合格即响亮失败——版面上空着一块比说错更显眼。"""
    if not facts:
        raise PlanCopyError(["事实清单是空的：不给素材的'直接推导'就是编"])
    raw = await client.complete_text(model, _SYSTEM_PROMPT, build_user_prompt(facts))
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].removesuffix("```").strip()
    try:
        copy = PlanCopy.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise PlanCopyError([f"模型没回出可用的 JSON：{raw[:300]}"]) from e
    problems = check_copy(copy, facts)
    if problems:
        raise PlanCopyError(problems)
    return copy
