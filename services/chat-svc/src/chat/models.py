"""chat 领域模型（pydantic，领域名词原样不带后缀；ORM 类加 Record 后缀区分）。

词表唯一真源：ishome-contracts `glossary.md` + 枚举 proto。认知状态六值与
Agent 方案 §8 逐字一致，禁止同义变体（如裸 confirmed）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from patch_engine.ops import PatchOp as PatchOp  # 领域内可直接引用（re-export）
from pydantic import BaseModel, Field

CognitiveState = Literal[
    "observed", "inferred", "proposed", "user_confirmed", "measured", "verified"
]
"""认知状态六值（枚举唯一真源：contracts CognitiveState）。"""

FactKind = Literal["dimensional", "structural"]
"""数据信任分轨：尺寸类确认后即为设计依据；结构类只走默认不动结构/硬证据。"""

ProjectPhase = Literal["preliminary", "deep"]


class ConversationRef(BaseModel):
    """会话定位三元组（渠道类型/渠道实例/渠道侧用户）——svc_chat.conversations 的自然键。

    TODO(identity)：identity 归一后改为渠道无关 user_id 键控（对齐 §6.5）。
    """

    channel_type: int
    channel_instance: str
    external_user_id: str

    @property
    def key(self) -> str:
        """进程内会话键（内存实现与会话态缓存的字典键）。"""
        return f"{self.channel_type}:{self.channel_instance}:{self.external_user_id}"


class ConversationTurn(BaseModel):
    """会话历史一轮（LLM 上下文用，有界保留）。

    上下文历史是会话态（Redis 接入位见 repo.SessionCache）；消息原文的持久化
    单位是 ChatMessage → svc_chat.messages，两者职责不同不合并。
    """

    role: Literal["user", "assistant"]
    text: str


MessageDirection = Literal["inbound", "outbound"]
"""消息方向：入站（用户→系统）/出站（系统→用户）。"""


class ChatMessage(BaseModel):
    """会话消息原文（svc_chat.messages 的领域形态，渠道协议无关）。"""

    external_message_id: str
    """UnifiedMessage.message_id：入站=渠道消息 id，出站=本服务生成 ULID。"""
    direction: MessageDirection
    content_type: str
    """text / quick_reply / image / audio / card / unknown（proto oneof 名原样）。"""
    text: str
    """消息原文文本；非文本内容存归一化占位文本（与 LLM 上下文一致）。"""
    idempotency_key: str
    """防重存幂等键：入站=渠道消息 id；出站=派生键 reply-{入站id}-{seq}。"""
    occurred_at: datetime | None = None
    """渠道侧发生时间（UTC；渠道未给则空，落库时间以 created_at 为准）。"""


class Fact(BaseModel):
    """带认知状态的关键数据（Agent 方案 §8.1 的 JSON 示例直接落列）。"""

    target_id: str
    property: str
    value: float | int | str | bool
    unit: str | None = None
    cognitive_state: CognitiveState
    fact_kind: FactKind = "dimensional"
    source: str
    confidence: float | None = None
    stage: ProjectPhase = "preliminary"


def fact_key(fact: Fact) -> str:
    """Fact 的确认清单键（confirmation_item 定位用）：target_id + property。"""
    return f"{fact.target_id}#{fact.property}"


class BaseFacts(BaseModel):
    """原始资料中可观察或确认的事实。

    原则上不可被设计方案覆盖；发现错误时经新证据与修订事件更正。
    """

    facts: list[Fact] = Field(default_factory=list)
    scale_anchor: Fact | None = None
    floorplan_ref: str | None = None  # estate floorplanId 或私有上传 key


class PlanMaster(BaseModel):
    """母版：plan-2d-render 产物，唯一几何真源，绑定 PreliminaryPlan Revision。"""

    artifact_id: str
    plan_revision: int
    png_key: str
    room_mask_key: str
    wall_layer_key: str


class ProjectState(BaseModel):
    """设计项目状态快照（渠道无关，按 user + project 键控）。

    V1.5：唯一真相属主已移交 project-svc（svc_project）；chat 侧仅为会话期
    快照，用于对话上下文与确认闭环，事实真相以 project-svc 为准。
    """

    project_id: str
    user_id: str
    phase: ProjectPhase = "preliminary"
    revision: int = 0
    base_facts: BaseFacts = Field(default_factory=BaseFacts)
    assumptions_told: bool = False
    """按面积推的那套默认假设，跟他说过没有。

    只说一次：裁决 8-31 的形态是"摊开说 + 给改的入口，给了就用不给就算"，
    每轮再说一遍就从"告知"变成"催问"了。
    """

    # 确认闭环（§8.2）：当前待确认的 fact_key 清单（会话侧状态，落库归 svc_chat）
    open_confirmation_ids: list[str] = Field(default_factory=list)
    # 最小必要输入（§3.1）是否已全部 user_confirmed——初步方案生成的前置门
    minimum_inputs_confirmed: bool = False
