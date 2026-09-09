"""Intent Router（Agent 方案 §5.2）：入站消息意图分类，防"说颜色改布局"式误路由。

服务层助手：只依赖 models 与 LlmCompletion 协议位（import-linter 锁定不越层）。
输入归一化前置层归本层职责；v1 已就位的是 quick_reply 直通路径，
TODO(normalize)：短时间窗多消息聚合、语音转文字（IM 输入形态，对齐 §6.6）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

INTENT_MODEL = "design-intent.default"
"""意图路由逻辑模型名（低延迟档；物理映射见 infra LiteLLM 配置）。"""

CALL_POINT = "chat-intent"
"""这一处 AI 判断在调用记录里的名字（判官台的基本信息表定的，照抄不改）。

与逻辑模型名不是一回事：模型名说"用哪个模型"，这个名字说"这次调用在判什么"。
"""

Intent = Literal[
    "provide_info",  # 提供信息（户型/家庭/诉求/尺寸……）
    "confirm_checklist",  # 确认清单：确认无误
    "correct_checklist",  # 确认清单：有要修正的
    "ask_reason",  # 询问设计原因/解释
    "other",  # 闲聊与其他
]

_VALID_INTENTS: frozenset[str] = frozenset(
    ("provide_info", "confirm_checklist", "correct_checklist", "ask_reason", "other")
)

# 确认清单打开时的确定性快路径（不依赖 LLM，真机可靠性兜底）
_CONFIRM_SHORTCUTS = frozenset(("确认", "确认无误", "没问题", "对的", "都对", "ok", "OK", "好的"))

_INTENT_SYSTEM_PROMPT = """\
你是家装设计助手的意图分类器。将用户消息分类为以下之一，只输出 JSON：
{"intent": "<类别>"}

类别定义：
- provide_info：用户在提供与房子/家庭/需求相关的信息（小区、户型、尺寸、家庭成员、诉求、限制等）
- confirm_checklist：用户在确认"确认清单"内容无误（仅当存在待确认清单时）
- correct_checklist：用户指出"确认清单"里有错、要修正某项（仅当存在待确认清单时）
- ask_reason：用户在询问设计原因、依据或解释
- other：闲聊、问候及其他
"""


class LlmCompletion(Protocol):
    """LLM 补全协议位（结构化子集，实现见 llm_client；测试用 FakeLLM）。

    `call_point`（这次调用是哪一处 AI 判断）必填不给默认值——漏传要当场露出来；
    `run_ref`（属于哪次运行）取不到就是 None。两样都只随请求送进网关的调用记录。
    """

    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        *,
        call_point: str,
        run_ref: str | None = None,
        json_mode: bool = False,
    ) -> str: ...


def parse_intent(raw: str) -> Intent:
    """解析分类输出；解析失败回落 provide_info（对话主流形态，误差可被主模型吸收）。"""
    try:
        data = json.loads(_strip_code_fence(raw))
        intent = data.get("intent")
        if isinstance(intent, str) and intent in _VALID_INTENTS:
            return intent  # type: ignore[return-value]
    except (json.JSONDecodeError, AttributeError):
        pass
    return "provide_info"


async def route_intent(
    llm: LlmCompletion, text: str, *, checklist_open: bool, run_ref: str | None = None
) -> Intent:
    """分类一条文本消息（quick_reply 选择在 service 层直通，不经过这里）。

    `run_ref` 是这一轮的运行编号，由 service 层给（会话键 + 本轮入站消息 id），
    只往下透传进调用记录——分类与编排是同一轮里的两次调用，靠它才串得回同一轮。
    """
    if checklist_open and text.strip() in _CONFIRM_SHORTCUTS:
        return "confirm_checklist"
    raw = await llm.complete(
        INTENT_MODEL,
        [
            {
                "role": "system",
                # 拼接而非 str.format——模板含 JSON 花括号
                "content": _INTENT_SYSTEM_PROMPT
                + f"\n当前是否存在待确认清单：{'是' if checklist_open else '否'}\n",
            },
            {"role": "user", "content": text},
        ],
        call_point=CALL_POINT,
        run_ref=run_ref,
        json_mode=True,
    )
    intent = parse_intent(raw)
    if not checklist_open and intent in ("confirm_checklist", "correct_checklist"):
        # 无清单可确认时的分类噪声，收敛为提供信息
        return "provide_info"
    return intent


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()
