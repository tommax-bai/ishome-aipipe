"""Design Orchestrator v1（Agent 方案 §3.1/§5.1/§8）：最小必要输入收集 + 确认闭环第一节点。

服务层助手：只依赖 models 与 LlmCompletion 协议位（import-linter 锁定不越层）。
职责边界：Agent 决定设计意图，结构化状态保存设计——本模块产出结构化事实与
文案载荷；重计算（解析/布局/绘制）归 genpipe activities（后续接入）。

v1 范围：初步阶段的最小必要输入（§3.1）经自然对话收集，集齐后出文本版
确认清单（§8.2 的文本形态；H5 看图点错后置——TODO(h5-pointing)）。
TODO(workflow)：确认完成后 signal DesignProjectWorkflow（workflows.py）。
TODO(events)：design.fact.confirmed / corrected 经 outbox 上总线（RocketMQ 接入后）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from design.models import ConversationTurn, Fact, ProjectState, fact_key

ORCHESTRATOR_MODEL = "design-orchestrator.default"
"""主对话逻辑模型名（物理映射见 infra LiteLLM 配置）。"""

CHECKLIST_MARKER = "【确认清单】"
"""确认清单消息的稳定前缀（E2E 与测试断言不变量）。"""

CONFIRM_OPTION_ID = "confirm-ok"
CORRECT_OPTION_ID = "confirm-edit"

CONFIRM_ACK_MARKER = "【已确认】"
"""确认完成回执的稳定前缀（E2E 与测试断言不变量）。"""


class LlmCompletion(Protocol):
    """LLM 补全协议位（结构化子集，实现见 llm_client；测试用 FakeLLM）。"""

    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        *,
        json_mode: bool = False,
    ) -> str: ...


class OrchestratorTurn(BaseModel):
    """一轮编排输出：抽取的事实 + 回复文案。"""

    facts: list[Fact] = Field(default_factory=list)
    reply: str = ""


_SYSTEM_PROMPT = """\
你是"是我的家"的 AI 家装设计顾问，正在与业主进行初步设计前的对话。
本轮同时完成两件事，只输出一个 JSON 对象：{"facts": [...], "reply": "..."}

一、事实提取（facts）：从用户最新消息提取与设计相关的事实，每条形如：
{"target_id": "...", "property": "...", "value": ..., "unit": null,
 "fact_kind": "dimensional", "cognitive_state": "observed", "confidence": 0.9}

target_id / property 必须用下列口径：
- 小区名 → floorplan / estate_name
- 户型（如"89平两室两厅"） → floorplan / floorplan_label
- 户型图来源 → floorplan / source，值 "library"（可按小区检索）或 "upload_pending"（待上传）
- 比例锚点（用户实测/确认的一个已知尺寸，如入户门宽）
  → scale-anchor / <部位英文snake_case>，unit 填 "mm"
- 家庭结构 → household / composition
- 核心诉求（最多3条） → need-1、need-2、need-3 / core_need
- 不可接受项 → no-go-1、no-go-2… / constraint（用户明确说没有时值填 "无"）
- 承重墙、梁柱、烟道、管井等结构信息 → fact_kind 必须是 "structural"

规则：用户明确陈述的 cognitive_state 记 "observed"，你推断的记 "inferred"；
长度尺寸一律毫米（mm）；没有新事实时 facts 为空数组；不确定的不要编造。

二、回复（reply）：自然、专业、简洁的中文，规则：
- 先回应用户刚说的内容；再针对缺口信息提问，一次只问一件事，禁止问卷式连问
- 口头提到推断尺寸时带"约"
- 用户问设计原因时给出真实依据
- 不承诺系统尚未具备的能力；当前阶段：收集并确认信息，之后生成初步设计方案
- 禁止任何免责、责任划分类表述
- 用户口述结构信息（承重墙/梁柱/烟道/管井）时，reply 中说明：口述的结构信息
  不能作为设计依据；两条路——默认方案不动任何结构，或提供原始结构图纸/物业
  文件作为硬证据
"""

# 最小必要输入五槽位（§3.1）；键为槽位标识，值为缺口提示（喂给 LLM 的口径）
_SLOT_HINTS: dict[str, str] = {
    "floorplan": "户型（小区名+户型，或标记待上传户型图）",
    "scale_anchor": "一个比例锚点（用户实测/确认的已知尺寸，如入户门宽，毫米）",
    "household": "家庭结构（谁住在这个家）",
    "core_need": "一至三个核心诉求",
    "no_go": "明确的不可接受项（没有也需用户明确说）",
}

_DISPLAY_LABELS: dict[tuple[str, str], str] = {
    ("floorplan", "estate_name"): "小区",
    ("floorplan", "floorplan_label"): "户型",
    ("floorplan", "source"): "户型图来源",
    ("household", "composition"): "家庭结构",
}

_SOURCE_DISPLAY: dict[str, str] = {"library": "户型库检索", "upload_pending": "待上传户型图"}


async def step(
    llm: LlmCompletion,
    project: ProjectState,
    history: Sequence[ConversationTurn],
    user_text: str,
) -> OrchestratorTurn:
    """一轮对话编排：抽取事实 + 生成回复（单次 LLM 调用控制延迟与成本）。"""
    state_lines = [_render_fact_line(f) for f in project.base_facts.facts]
    missing = missing_slots(project)
    context = ("已收集的信息：\n" + ("\n".join(state_lines) if state_lines else "（暂无）")) + (
        "\n\n仍缺少：\n" + "\n".join(f"- {_SLOT_HINTS[s]}" for s in missing) if missing else ""
    )
    messages: list[Mapping[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT + "\n\n" + context}
    ]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.text})
    messages.append({"role": "user", "content": user_text})
    raw = await llm.complete(ORCHESTRATOR_MODEL, messages, json_mode=True)
    return parse_turn(raw)


def parse_turn(raw: str) -> OrchestratorTurn:
    """容错解析 LLM 输出；坏 JSON 时返回空事实与空回复（service 层兜底文案）。"""
    try:
        data = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError:
        return OrchestratorTurn()
    facts: list[Fact] = []
    for item in data.get("facts", []) if isinstance(data, dict) else []:
        fact = _coerce_fact(item)
        if fact is not None:
            facts.append(fact)
    reply = data.get("reply", "") if isinstance(data, dict) else ""
    return OrchestratorTurn(facts=facts, reply=reply if isinstance(reply, str) else "")


def _coerce_fact(item: object) -> Fact | None:
    if not isinstance(item, dict):
        return None
    payload = dict(item)
    # 认知状态钳制：抽取产物只允许 observed / inferred——user_confirmed 只能由
    # 确认闭环授予，measured/verified 只能由量房与硬证据授予（§8.1）。
    if payload.get("cognitive_state") not in ("observed", "inferred"):
        payload["cognitive_state"] = "inferred"
    payload.setdefault(
        "source",
        "user_message"
        if payload.get("cognitive_state") == "observed"
        else "orchestrator_inference",
    )
    try:
        return Fact.model_validate(payload)
    except ValidationError:
        return None


def merge_facts(project: ProjectState, new_facts: Sequence[Fact]) -> list[Fact]:
    """按 fact_key 合并（同键替换=修正）；返回本轮新记录的结构类事实。

    结构类红线（§8.3）：structural 事实照记（作为线索保留），但永不因口述
    进入可确认集合——upgrade_confirmed 与确认清单都会跳过它们。
    """
    structural_recorded: list[Fact] = []
    for fact in new_facts:
        key = fact_key(fact)
        existing = {fact_key(f): i for i, f in enumerate(project.base_facts.facts)}
        if key in existing:
            project.base_facts.facts[existing[key]] = fact
        else:
            project.base_facts.facts.append(fact)
        if fact.fact_kind == "structural":
            structural_recorded.append(fact)
        if fact.target_id == "scale-anchor":
            project.base_facts.scale_anchor = fact
    return structural_recorded


def missing_slots(project: ProjectState) -> list[str]:
    """最小必要输入缺口（§3.1 五槽位）。"""
    facts = project.base_facts.facts
    missing: list[str] = []
    has_floorplan = any(
        f.target_id == "floorplan" and f.property in ("estate_name", "source") for f in facts
    )
    if not has_floorplan:
        missing.append("floorplan")
    if not any(f.target_id == "scale-anchor" for f in facts):
        missing.append("scale_anchor")
    if not any(f.target_id == "household" for f in facts):
        missing.append("household")
    if not any(f.property == "core_need" for f in facts):
        missing.append("core_need")
    if not any(f.property == "constraint" for f in facts):
        missing.append("no_go")
    return missing


def confirmable_facts(project: ProjectState) -> list[Fact]:
    """进确认清单的事实：尺寸/事实轨全部；结构类永不进（§8.3 口述不采信）。"""
    return [f for f in project.base_facts.facts if f.fact_kind == "dimensional"]


def build_checklist_text(project: ProjectState) -> tuple[str, list[str]]:
    """渲染确认清单（文字由系统确定，不经 LLM）；返回（文本, 待确认 fact_key 列表）。"""
    items = confirmable_facts(project)
    lines = [f"{CHECKLIST_MARKER}请核对这些信息，确认后它们将作为设计依据："]
    for i, fact in enumerate(items, start=1):
        lines.append(f"{i}. {_display_label(fact)}：{_display_value(fact)}")
    lines.append("都对就回复“确认”；有要改的直接说哪一项。")
    return "\n".join(lines), [fact_key(f) for f in items]


def upgrade_confirmed(project: ProjectState) -> int:
    """确认清单通过：清单内事实升级 user_confirmed（结构类永不经此路径）。"""
    open_ids = set(project.open_confirmation_ids)
    upgraded = 0
    for fact in project.base_facts.facts:
        if fact_key(fact) in open_ids and fact.fact_kind == "dimensional":
            fact.cognitive_state = "user_confirmed"
            upgraded += 1
    project.open_confirmation_ids = []
    project.minimum_inputs_confirmed = True
    # TODO(workflow)：signal DesignProjectWorkflow.confirm_facts
    # TODO(events)：design.fact.confirmed 上总线
    return upgraded


def confirm_ack_text() -> str:
    return (
        f"{CONFIRM_ACK_MARKER}信息已全部确认，从现在起它们就是这个项目的设计依据。\n"
        "下一步是基于确认信息生成初步设计方案与交付图集——这部分能力正在接入，"
        "完成后会直接发到这个对话里。期间你随时可以补充或修改信息。"
    )


def structural_note() -> str:
    """口述结构信息的固定说明（确定性文案，随本轮回复附带）。"""
    return (
        "关于结构（承重墙/梁柱/烟道/管井）：口述信息我会记下来，但不能作为设计依据。"
        "两条路任选——默认方案不动任何结构；或提供原始结构图纸、物业确认文件，"
        "经校验后作为硬证据使用。"
    )


def _display_label(fact: Fact) -> str:
    label = _DISPLAY_LABELS.get((fact.target_id, fact.property))
    if label:
        return label
    if fact.target_id == "scale-anchor":
        return f"比例锚点（{fact.property}）"
    if fact.property == "core_need":
        return "核心诉求"
    if fact.property == "constraint":
        return "不可接受项"
    return fact.property


def _display_value(fact: Fact) -> str:
    value = fact.value
    if fact.property == "source" and isinstance(value, str):
        rendered = _SOURCE_DISPLAY.get(value, value)
    elif isinstance(value, bool):
        rendered = "是" if value else "否"
    else:
        rendered = f"{value}{fact.unit or ''}"
    # 数据诚实展示（§8.1）：inferred 值带"约"——服务于用户理解
    return f"约{rendered}" if fact.cognitive_state == "inferred" else rendered


def _render_fact_line(fact: Fact) -> str:
    kind = "结构类·仅线索" if fact.fact_kind == "structural" else fact.cognitive_state
    return f"- {fact.target_id}/{fact.property} = {fact.value}{fact.unit or ''}（{kind}）"


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()
