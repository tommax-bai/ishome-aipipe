"""Design Orchestrator v1（Agent 方案 §3.1/§5.1/§8）：最小必要输入收集 + 确认闭环第一节点。

服务层助手：只依赖 models 与 LlmCompletion 协议位（import-linter 锁定不越层）。
职责边界：Agent 决定设计意图，结构化状态保存设计——本模块产出结构化事实与
文案载荷；重计算（解析/布局/绘制）归 genpipe activities（后续接入）。

v1 范围：初步阶段的最小必要输入（§3.1）经自然对话收集，集齐后出文本版
确认清单（§8.2 的文本形态；H5 看图点错后置——TODO(h5-pointing)）。
TODO(project-svc)：确认完成后 artifact_confirmed 业务事实发往 project-svc
（V1.5：里程碑引擎事件驱动，原设计项目长周期 workflow 方案作废）。
TODO(events)：design.fact.confirmed / corrected 经 outbox 上总线（RocketMQ 接入后）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from chat.models import ConversationTurn, Fact, ProjectState, fact_key

logger = logging.getLogger(__name__)

ORCHESTRATOR_MODEL = "design-orchestrator.default"
"""主对话逻辑模型名（物理映射见 infra LiteLLM 配置）。"""

CHECKLIST_MARKER = "【确认清单】"
"""确认清单消息的稳定前缀（E2E 与测试断言不变量）。"""

CONFIRM_OPTION_ID = "confirm-ok"
CORRECT_OPTION_ID = "confirm-edit"

CONFIRM_ACK_MARKER = "【已确认】"
"""确认完成回执的稳定前缀（E2E 与测试断言不变量）。"""

ONE_THING_MAX_CHARS = 60
"""一条消息最多多少字——超过就说明这条里塞了不止一件事（分条规矩的兜底判据）。

**这个数只写在这一处**：下面的提示词从它拼出来（对模型说的分条护栏），
`tests/test_chat_conversation.py` 的门禁按它判系统自己写死的文案（对我们自己说的同一条）。
分条判据本身是"这是另一件事"，字数只是兜底——两边共用一个兜底数，才不会一边松一边紧。
"""


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
    """一轮编排输出：抽取的事实 + 回复文案（若干短句）。

    **回复是数组不是一个字符串**：人在 IM 里不会把回应、追问、说明挤成一段发出来，
    而模型只要拿到一个字符串位、又被要求一轮说到好几件事，就只能挤成一段（真机实测）。

    **断点由写的这一步自己标，不另起一个切分模型**（用户裁决 2026-08-31）：思维停顿是
    写的时候产生的，不是文本的属性——写的那一方知道"这句是回应、下句是另一件事"，
    停顿正从这儿来；整段写完再交给第二个模型，它只能从字面反推意图结构，信息更少，
    还多一次调用的延迟与失败面。分条是一次想清楚、分几口说出来，**不是想几次**，
    所以仍是单次 LLM 调用，编排不变。发送侧本就按数组循环发（幂等键带序号），这里给它源头。
    **它标不好时的后续路径写死＝真机上出现"总是不分"或"在同一件事中间断开"时，
    再加一步切分**——先用便宜的假设，被证伪了再上贵的。
    """

    facts: list[Fact] = Field(default_factory=list)
    replies: list[str] = Field(default_factory=list)


_SYSTEM_PROMPT = (
    """\
你是"是我的家"的 AI 家装设计顾问，正在与业主进行初步设计前的对话。
本轮同时完成两件事，只输出一个 JSON 对象：{"facts": [...], "reply": "..."}

一、事实提取（facts）：从用户最新消息提取与设计相关的事实，每条形如：
{"target_id": "...", "property": "...", "value": ..., "unit": null,
 "fact_kind": "dimensional", "cognitive_state": "observed", "confidence": 0.9}

target_id / property 必须用下列口径：
- 小区名 → floorplan / estate_name
- 户型（如"89平两室两厅"） → floorplan / floorplan_label
- 户型图来源 → floorplan / source，值 "library"（可按小区检索）或 "upload_pending"（待上传）
- 建筑面积 → floorplan / building_area_sqm，unit 填 "sqm"
- 得房率 → floorplan / floor_area_ratio，值填百分数的数字（如 81），unit 填 "percent"；
  用户说不知道、说不清或让系统自己推断时，值填 "unknown"
- 用户**主动**给出的实测尺寸（如"我量了入户门宽 900"）
  → scale-anchor / <部位英文snake_case>，unit 填 "mm"。他不主动给就没有这条，不要索取
- 家庭结构 → household / composition
- 核心诉求（最多3条） → need-1、need-2、need-3 / core_need
- 不可接受项 → no-go-1、no-go-2… / constraint（用户明确说没有时值填 "无"）
- 承重墙、梁柱、烟道、管井等结构信息 → fact_kind 必须是 "structural"

规则：用户明确陈述的 cognitive_state 记 "observed"，你推断的记 "inferred"；
长度尺寸一律毫米（mm）；没有新事实时 facts 为空数组；不确定的不要编造。

二、回复（replies）：**一个数组，1 到 3 条**，中文，像人在 IM 里说话那样分条。

怎么分条：
- **判据是"这是另一件事"**——回应他刚说的是一件事，追问是另一件事，补充说明是第三件。
  换一件事说就另起一条；**只有一件事要说就只发一条**，不要为了凑数硬拆
- **不要按长度切**：把同一件事从中间断开比不分更难读
"""
    f"- 兜底护栏：**一条超过 {ONE_THING_MAX_CHARS} 字**就说明这条里塞了不止一件事，"
    "回头看看该拆在哪儿\n"
    """\
- **整轮加起来最多一个问号**。想问三件事就只问最要紧的那一件，剩下的下一轮再问——
  一次问完对你省事，对他是一张问卷

内容：
- 先回应用户刚说的内容，再针对缺口信息提问
- **绝不编造具体数字**：门宽、开间、面积这类数只能来自用户说过的或系统给你的；
  想不起来就不说，禁止"大约 860mm""通常 3600mm"这类凭空举例
- **绝不自造精度声明**：禁止"误差 ±2%""精度 95%"这类说法，你无从知道
- 口头提到系统推断出来的尺寸时带"约"
- 用户问设计原因时给出真实依据
- 不承诺系统尚未具备的能力；当前阶段：收集并确认信息，之后生成初步设计方案
- 禁止任何免责、责任划分类表述
- **不要请用户去实测任何尺寸**（不提卷尺、不说"方便时补测"）：户型尺寸由系统按户型图
  自己标定，这是主路径不是降级路径。他主动给实测值就收下并道谢
- 用户口述结构信息（承重墙/梁柱/烟道/管井）时说明：口述的结构信息不能作为设计依据；
  两条路——默认方案不动任何结构，或提供原始结构图纸/物业文件作为硬证据
"""
)

# **业主开头只需要给两样：面积 + 户型图**（用户裁决 2026-08-31）。其余一概推断——
# 人数、得房率、装修倾向、年龄由 :mod:`chat.assumptions` 从面积算出来，产出之后摊开说、
# 给他一个改的入口；**给了就用，不给就算，不追问**。形态是"先做出来再让他改"，不是"问齐了再做"。
#
# 这条一路收窄了三批问法，每一批都是同一种病——**系统在问它自己能算或能推的东西**：
# 比例锚点（逼他拿卷尺量房）、小区名与几室几厅（图里就写着）、得房率与家庭结构（面积推得出）。
_SLOT_HINTS: dict[str, str] = {
    "building_area_sqm": "建筑面积（多少平，他一句话就能说；这是唯一还需要他给的数）",
    "floorplan": (
        "还没有户型图：请他发一张户型图的照片或截图。"
        "**不要问小区名、不要问几室几厅、不要问面积以外的任何数**——图到手这些都算得出来，"
        "问他一遍等于让他替系统干活"
    ),
}

_DISPLAY_LABELS: dict[tuple[str, str], str] = {
    ("floorplan", "estate_name"): "小区",
    ("floorplan", "floorplan_label"): "户型",
    ("floorplan", "source"): "户型图来源",
    ("floorplan", "building_area_sqm"): "建筑面积",
    ("floorplan", "floor_area_ratio"): "得房率",
    ("household", "composition"): "家庭结构",
}

_ANCHOR_PARTS: dict[str, str] = {
    "entry_door_width": "入户门宽",
    "bedroom_door_width": "卧室门宽",
    "wall_length": "墙长",
    "window_width": "窗宽",
}
"""业主主动给实测尺寸时，那个部位叫什么。表外的不猜——认不出就只说"实测尺寸"。"""

_UNIT_DISPLAY: dict[str, str] = {"sqm": " 平方米", "mm": " 毫米", "percent": "%"}
"""单位也是内部写法：`138sqm` 递进去，模型就照着说出来了。表外的原样出，不猜。"""

_SOURCE_DISPLAY: dict[str, str] = {
    "library": "户型库检索",
    "upload_pending": "待上传户型图",
    "uploaded": "业主已发来户型图",
}


async def step(
    llm: LlmCompletion,
    project: ProjectState,
    history: Sequence[ConversationTurn],
    user_text: str,
) -> OrchestratorTurn:
    """一轮对话编排：抽取事实 + 生成回复（单次 LLM 调用控制延迟与成本）。"""
    state_lines = [_render_fact_line(f) for f in project.base_facts.facts]
    # **一次只递一个缺口**：模型看不见第二个缺口，就问不出第二个问题。
    # 提示词里写过"一次只问一件事"，真机两跑都是一轮问三件——纪律禁不住，结构禁得住
    # （同"推导步看不见落点的值所以产不出数字"）。缺口清单本身不动，判确认闭环还要用整份。
    missing = missing_slots(project)
    context = ("已收集的信息：\n" + ("\n".join(state_lines) if state_lines else "（暂无）")) + (
        f"\n\n现在还缺这一件（**只问它，别的下一轮再说**）：\n- {_SLOT_HINTS[missing[0]]}"
        if missing
        # 缺口清空之后必须**明说问完了**：不说的话，"针对缺口提问"那条规则会让它自己编一个
        # 要问的——真机上它问了"家里常住几口人、主要诉求是什么"，而系统紧接着的下一条就是
        # "我按 4 个人来安排…不说也没关系"，等于先为难他再说不用。
        else (
            "\n\n**该要的都要到了：这一轮不要再向他要任何信息**，"
            "不要问人数、诉求、预算、小区名，也不要提出下一步要他配合的事。"
            "这些系统自己会推，推完了会主动告诉他按什么做的。他想补充什么，等他自己说。"
        )
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
    """容错解析 LLM 输出；坏 JSON 时返回空事实与空回复（service 层兜底文案）。

    **一条回复都没解析出来时把原文记进日志**：业主那头看到的是兜底话，而日志里若只写
    "reply sent"，这种失败就是隐形的——真机上正是这样丢了一整轮回复才被发现。
    """
    try:
        data = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError:
        logger.warning("编排输出不是 JSON，本轮回话丢空：%s", raw[:400])
        return OrchestratorTurn()
    facts: list[Fact] = []
    for item in data.get("facts", []) if isinstance(data, dict) else []:
        fact = _coerce_fact(item)
        if fact is not None:
            facts.append(fact)
    replies = _coerce_replies(data)
    if not replies:
        logger.warning("编排输出里没有可用的回复，本轮回话丢空：%s", raw[:400])
    return OrchestratorTurn(facts=facts, replies=replies)


def _coerce_replies(data: object) -> list[str]:
    """回复数组。**键名与形态都收宽**：`replies`/`reply` 两个键、数组/单串两种值，四种组合都收。

    真机踩过：模型照做了分条（回了数组），但键名仍写 `reply`——首版只认"replies 是数组、
    reply 是字符串"这两种，那一种两边都不沾，**整轮回复被丢光**，业主收到的是兜底话。
    形态收宽的判据：模型的意思清清楚楚，丢掉它的是我们的解析器不是它的输出。
    """
    if not isinstance(data, dict):
        return []
    for key in ("replies", "reply"):
        value = data.get(key)
        if isinstance(value, list):
            return [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return []


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


def upload_fact() -> Fact:
    """用户传了户型图这件事，**由代码记，不由模型抽**。

    会话侧自己就知道这条入站消息是图片（统一消息的内容类型摆在那儿），不需要模型来告诉它。
    交给模型抽有一轮延迟：缺口是按上一轮末的状态算的，于是**图刚传上来、系统还在问"你有户型图吗"**
    ——真机就是这么问出"您家在哪个小区？几室几厅？"的。同"几何不由 LLM 决定"的同一条理由：
    代码知道的事不问模型。
    """
    return Fact(
        target_id="floorplan",
        property="source",
        value="uploaded",
        cognitive_state="observed",
        source="channel_upload",
    )


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
    """还差业主给的那几样。**只剩两样：面积与户型图**（用户裁决 2026-08-31）。

    面积排在户型图前面：他一句话就说得出，而传图要找文件。先要好说的那个。
    """
    facts = project.base_facts.facts
    missing: list[str] = []
    if not any(f.property == "building_area_sqm" for f in facts):
        missing.append("building_area_sqm")
    has_floorplan = any(
        f.target_id == "floorplan" and f.property in ("estate_name", "source") for f in facts
    )
    if not has_floorplan:
        missing.append("floorplan")
    return missing


def find_building_area_sqm(project: ProjectState) -> float | None:
    """业主给的建筑面积。find = 可空（规范 §三：取不到返回 None，不抛）。"""
    for fact in project.base_facts.facts:
        if fact.property == "building_area_sqm" and isinstance(fact.value, int | float):
            return float(fact.value)
    return None


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
    # TODO(project-svc)：artifact_confirmed 业务事实发往 project-svc（V1.5）
    # TODO(events)：design.fact.confirmed 上总线
    return upgraded


def confirm_ack_text() -> str:
    return (
        f"{CONFIRM_ACK_MARKER}信息已全部确认，从现在起它们就是这个项目的设计依据。\n"
        "下一步是基于确认信息生成初步设计方案与交付图集——这部分能力正在接入，"
        "完成后会直接发到这个对话里。期间你随时可以补充或修改信息。"
    )


def structural_notes() -> list[str]:
    """口述结构信息的固定说明（确定性文案，随本轮回复附带）——**一条一件事，两条**。

    一条说"口述的不作依据"，另一条说"那有哪两条路走"：**判据是"这是另一件事"**，
    前一条是拒绝、后一条是出路。首版一条里塞了这两件，正好是分条规矩管不到系统文案的漏点
    （用户 2026-08-31 晚）。这一段今天真会发给业主，所以先拆的是它。
    """
    return [
        "关于结构（承重墙/梁柱/烟道/管井）：口述的我会记下来，但不能作为设计依据。",
        "两条路任选：默认方案不动任何结构；或给我原始结构图纸、物业确认文件，校验过就能当依据。",
    ]


def _display_label(fact: Fact) -> str:
    label = _DISPLAY_LABELS.get((fact.target_id, fact.property))
    if label:
        return label
    if fact.target_id == "scale-anchor":
        part = _ANCHOR_PARTS.get(fact.property)
        # 认不出的部位就只说"实测尺寸"——**宁可少说一个词，不把内部名说给业主**
        return f"{part}实测" if part else "实测尺寸"
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
        rendered = f"{value}{_UNIT_DISPLAY.get(fact.unit or '', fact.unit or '')}"
    # 数据诚实展示（§8.1）：inferred 值带"约"——服务于用户理解
    return f"约{rendered}" if fact.cognitive_state == "inferred" else rendered


_STATE_WORDS: dict[str, str] = {
    "observed": "他说的",
    "inferred": "系统推的",
    "user_confirmed": "他确认过的",
    "measured": "量过的",
    "verified": "核过的",
}


def _render_fact_line(fact: Fact) -> str:
    """递给模型的那份已知信息，**用客户语域写，不用内部字段名**。

    首版直接把 `floorplan/source = upload_pending（observed）` 喂进去——真机上模型原样
    转述给了业主："您提到的户型图我们已收到（来源：upload_pending）"。
    **它看不见就说不出**：与其加一条"不许说内部字段名"的禁令，不如根本不给它看
    （同"推导步看不见落点的值所以产不出数字"）。
    """
    kind = (
        "结构类·仅线索"
        if fact.fact_kind == "structural"
        else _STATE_WORDS.get(fact.cognitive_state, fact.cognitive_state)
    )
    return f"- {_display_label(fact)}：{_display_value(fact)}（{kind}）"


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()
