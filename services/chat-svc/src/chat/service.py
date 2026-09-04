"""用例层：设计会话编排入口（Design Orchestrator v1 已接入）。

流程：入站落存（svc_chat.messages，幂等键防重存兼去重门）→ 输入归一化
（v1：quick_reply 直通；TODO(normalize) 多消息聚合、语音转文字）→ Intent Router
→ Orchestrator（事实抽取 + 回复）→ 两样（面积 + 户型图）齐了说"开始设计"
→ 出站回话（发送后落存出站原文）。

**假设那套不在这条流程里**：它由 `deliverables_delivered` 在图送回业主之后主动发
（裁决 8-31 原话"产出结果之后也告诉用户"）。确认闭环（清单 → user_confirmed 升级）
机件保留，时点同样挪到真有产出可确认时——两处都不是删掉，是等它们该发生的那一刻。

**2026-09-04 接线**：每轮回话之前把新到的事实（户型图对象键、建筑面积、按面积推的得房率）
报给业务侧（`BusinessSideGateway`，contracts project.v1）——会话侧不判里程碑不建任务，
业务侧判定并派发三张图；图好了业务侧经 `PresentDeliverables` 回来，本层经渠道发进聊天线程，
随后才说假设。上报失败对业主如实说（`REPORT_FAILED_MESSAGES`），事实留在快照里下一轮再报。

存储：`CHAT_DATABASE_URL` 设置时消息原文落 PG（schema svc_chat），未设时内存
（e2e-mock-smoke 裸起可跑）——选择在 repo 层，本层不感知。会话态（项目快照/
上下文历史）为进程内缓存，Redis 接入位在 repo.SessionCache。

- LLM 一律经 LiteLLM 网关（llm_client），业务只引用任务级逻辑模型名；
- 结构类红线（§8.3）：口述结构信息永不进入可确认集合，回复附两条路径说明；
- TODO(identity)：会话键 → identity 归一 user_id；
- TODO(project-svc)：确认完成 → artifact_confirmed 业务事实发往 project-svc
  （V1.5：里程碑引擎事件驱动，原设计项目长周期 workflow 方案作废）；
- TODO(h5-pointing)：看图点错确认形态（H5 指图时刻）。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol, cast

from ishome.channel.v1 import message_pb2
from ishome.common.v1 import channel_type_pb2
from ishome.design.v1 import service_pb2 as design_service_pb2
from ulid import ULID

from chat import intent as intent_router
from chat import orchestrator
from chat.assumptions import DEFAULT_FLOOR_AREA_RATIO_PERCENT, assumption_messages, infer_from_area
from chat.models import ChatMessage, ConversationRef, ConversationTurn, ProjectState
from chat.project_client import BusinessProject, MilestoneProgress, ProjectClientError, SlotFill
from chat.repo import (
    append_history,
    find_or_create_project,
    find_project,
    get_history,
    record_inbound,
    record_outbound,
    save_project,
)

logger = logging.getLogger(__name__)

FALLBACK_REPLY = "这条我没处理好，麻烦再发一次。"
"""LLM 或编排故障时的兜底回话——每条入站必有一条出站（E2E 不变量）。"""

DESIGN_START_MESSAGES: tuple[str, ...] = (
    "面积和户型图都齐了，我这就开始给你做设计。",
    "做好了我直接把图发过来，你先不用再准备什么。",
)
"""两样齐了那一轮只说这一件事：**开始设计**（用户 2026-08-31 晚纠正）。

**这里一个假设都不提**：假设那套要等图发到业主手里之后才说（`deliverables_delivered`）。
真机上图还没影，业主先收到"我按 4 个人来安排"——他不知道这是在说哪份东西。
第二条也**不再向他要任何信息**：说清"你不用再准备什么"，比只在提示词里禁止追问更牢靠。
"""

REPORT_FAILED_MESSAGES: tuple[str, ...] = (
    "你发的我都记下了，不过我这边的设计系统刚才没接上。",
    "过一会儿再随便发我一句话，我接着往下做。",
)
"""上报业务侧失败时对业主说的——如实说没接上，不装作在做（红线"失败要说人话"）。
事实仍在快照里，下一轮入站会再报一次；第二条给的是"怎么触发再试"的出路，不是追问信息。"""

GENERATION_FAILED_MESSAGES: tuple[str, ...] = (
    "这套图我这边没做出来。",
    "你可以换一张更清楚的户型图发我，或者过一会儿再发一遍，我再试一次。",
)
"""业务侧送来"没做出来"时对业主说的（v0.3 §9 失败路径显式）：诚实告知 + 两条出路。"""

DELIVERABLE_CAPTIONS: dict[str, str] = {
    "vision_mood_image": "第一张：你家的样子",
    "vision_brief_image": "第二张：每间房怎么用",
    "vision_style_image": "第三张：手账版",
}
"""三张图各自随图一句（系统文案，分条规矩管；产物类型是业务侧的数据值，本侧只查表不理解）。"""

# 分条之间的停顿（用户裁决 2026-08-31）：一轮多条要按上一条的长度歇一下再发下一条——
# 三条瞬间刷屏，读的人还没看完第一条就被第二条盖过去了，分条反而比不分更难读。
# 不模拟打字速度（那会慢到烦人），只给一个"够看完上一句"的短拍。
_PACING_SECONDS_PER_CHAR = 0.03
_PACING_MIN_SECONDS = 0.4
_PACING_MAX_SECONDS = 1.5

LlmCompletion = orchestrator.LlmCompletion
"""LLM 协议位（结构化子集；实现 llm_client.LiteLlmClient，测试 FakeLLM）。"""


class OutboundSender(Protocol):
    """出站回话协议位：service 层不感知具体渠道客户端（组合根注入）。"""

    async def send(self, message: message_pb2.UnifiedMessage, idempotency_key: str) -> str: ...


class CapabilityLookup(Protocol):
    """渠道能力查询协议位（按能力分支不按身份分支，R5）。"""

    async def supports_quick_reply(self, channel_type: int, channel_instance: str) -> bool: ...


class BusinessSideGateway(Protocol):
    """业务侧（project-svc）协议位：按属主取或建项目、报一批槽位
    （实现 project_client.ProjectClient）。"""

    async def find_or_create_project(
        self, channel_type: int, channel_instance: str, external_user_id: str
    ) -> BusinessProject: ...

    async def fill_slots(self, project_id: str, slots: Sequence[SlotFill]) -> MilestoneProgress: ...


async def get_project(project_id: str) -> ProjectState:
    """get = 必得（取不到抛异常）。"""
    project = await find_project(project_id)
    if project is None:
        raise KeyError(f"project not found: {project_id}")
    return project


async def ingest_message(
    inbound: message_pb2.UnifiedMessage,
    sender: OutboundSender,
    llm: LlmCompletion,
    capability: CapabilityLookup | None = None,
    business: BusinessSideGateway | None = None,
) -> str:
    """会话入站处理；返回入站 message_id。

    `business` 为空＝没接业务侧（e2e-mock-smoke 裸起、旧单测）：事实只留在会话快照里，不上报。
    """
    conversation = _conversation_ref(inbound)
    # 入站原文落存即幂等门：幂等键（=渠道消息 id）已存过说明是渠道重投，跳过
    if not await record_inbound(conversation, _inbound_message(inbound)):
        logger.info("duplicate inbound skipped: message_id=%s", inbound.message_id)
        return inbound.message_id

    logger.info(
        "inbound message: message_id=%s channel=%s/%s content=%s",
        inbound.message_id,
        inbound.channel_type,
        inbound.channel_instance,
        inbound.WhichOneof("content"),
    )
    project = await find_or_create_project(conversation)
    user_text = _inbound_text(inbound)

    try:
        reply_texts, quick_reply_checklist = await _converse(
            inbound, project, conversation, user_text, llm, capability
        )
    except Exception:
        logger.exception("conversation turn failed: message_id=%s", inbound.message_id)
        reply_texts, quick_reply_checklist = [FALLBACK_REPLY], None

    # 上报业务侧：在回话之前——"开始设计"这句要建立在业务侧真的接了活的基础上。
    # 没接上就如实说（不装作在做），事实留在快照里下一轮再报。
    if business is not None:
        try:
            await report_facts(conversation, project, business, source_event_id=inbound.message_id)
        except ProjectClientError:
            logger.exception("business-side report failed: message_id=%s", inbound.message_id)
            reply_texts = [*reply_texts, *REPORT_FAILED_MESSAGES]

    await append_history(conversation, ConversationTurn(role="user", text=user_text))
    outbounds = [_text_reply(inbound, text) for text in reply_texts]
    if quick_reply_checklist is not None:
        outbounds.append(_quick_reply_checklist(inbound, quick_reply_checklist))
    # 幂等键从入站消息派生：同一入站消息的回话重试不会在聊天线程里发两遍
    await _send_all(
        conversation, sender, outbounds, idempotency_prefix=f"reply-{inbound.message_id}"
    )
    await save_project(conversation, project)
    return inbound.message_id


def pending_slot_fills(project: ProjectState, *, source_event_id: str) -> list[SlotFill]:
    """快照里有、还没报给业务侧（或值变了）的槽位（纯函数）。

    只报三样：户型图对象键、建筑面积（业主给的）、得房率（业主给的按 observed，没给按面积推
    为 inferred 的默认值——数字不由 LLM 决定，推的那一步在 `assumptions`）。业务侧判据只看前两样。
    """
    candidates: list[tuple[str, str, str]] = []
    object_key = orchestrator.find_floorplan_object_key(project)
    if object_key:
        candidates.append(("floorplan", object_key, "observed"))
    area_sqm = orchestrator.find_building_area_sqm(project)
    if area_sqm is not None:
        candidates.append(("building_area_sqm", _number_text(area_sqm), "observed"))
        given_ratio = orchestrator.find_floor_area_ratio_percent(project)
        if given_ratio is not None:
            candidates.append(("floor_area_ratio_percent", _number_text(given_ratio), "observed"))
        else:
            candidates.append(
                ("floor_area_ratio_percent", str(DEFAULT_FLOOR_AREA_RATIO_PERCENT), "inferred")
            )
    return [
        SlotFill(slot_key=key, value=value, cognitive_state=state, source_event_id=source_event_id)
        for key, value, state in candidates
        if project.reported_slots.get(key) != value
    ]


async def report_facts(
    conversation: ConversationRef,
    project: ProjectState,
    business: BusinessSideGateway,
    *,
    source_event_id: str,
) -> MilestoneProgress | None:
    """把新到的事实报给业务侧（contracts project.v1）。没有新东西就不打这一跳。

    会话侧不判里程碑、不建任务：业务侧回来的 `created_task_ids` 只记日志，不据此改会话形态——
    图好没好，等它经 `PresentDeliverables` 回来。失败上抛 `ProjectClientError`，
    由调用方决定怎么对业主说；
    已报成功的槽位记进 `reported_slots`，重启丢了缓存也只是多报一次（业务侧 upsert 幂等）。
    """
    fills = pending_slot_fills(project, source_event_id=source_event_id)
    if not fills:
        return None
    if project.business_project_id is None:
        business_project = await business.find_or_create_project(
            conversation.channel_type, conversation.channel_instance, conversation.external_user_id
        )
        project.business_project_id = business_project.project_id
        logger.info(
            "business project %s: id=%s milestone=%s",
            "created" if business_project.created else "found",
            business_project.project_id,
            business_project.current_milestone,
        )
    progress = await business.fill_slots(project.business_project_id, fills)
    for fill in fills:
        project.reported_slots[fill.slot_key] = fill.value
    logger.info(
        "facts reported: project=%s slots=%s milestone=%s advanced=%s tasks=%s",
        project.business_project_id,
        [fill.slot_key for fill in fills],
        progress.current_milestone,
        progress.advanced,
        progress.created_task_ids,
    )
    return progress


async def present_deliverables(
    request: design_service_pb2.PresentDeliverablesRequest,
    sender: OutboundSender,
) -> tuple[bool, list[str]]:
    """业务侧送来一批产物（或"没做出来"）：经渠道发进聊天线程，随后说假设。
    返回（这次发没发, 消息 id）。

    **幂等**：同一 delivery_id 第二次到达不再发（业务侧中继会重投）。
    产物按业务侧给的顺序发，每张前面一句系统文案（`DELIVERABLE_CAPTIONS`，查不到就不加）；
    图都发完才调 `deliverables_delivered`——假设那套的时点写死在"图发到业主手里之后"。
    """
    conversation = ConversationRef(
        channel_type=request.owner.channel_type,
        channel_instance=request.owner.channel_instance,
        external_user_id=request.owner.external_user_id,
    )
    if not request.delivery_id:
        raise ValueError("delivery_id 为空：没有幂等键的送达不发")
    project = await find_or_create_project(conversation)
    if request.delivery_id in project.deliveries_seen:
        logger.info("delivery already presented, skipped: delivery_id=%s", request.delivery_id)
        return False, []

    outbounds: list[message_pb2.UnifiedMessage] = []
    if request.HasField("failure"):
        logger.warning(
            "generation failed for owner=%s task_type=%s code=%s detail=%s",
            conversation.key,
            request.failure.task_type,
            request.failure.code,
            request.failure.detail,
        )
        outbounds.extend(_text_message(conversation, text) for text in GENERATION_FAILED_MESSAGES)
        prefix = f"failure-{request.delivery_id}"
    else:
        if not request.deliverables:
            raise ValueError("既没有产物也没有失败说明：这次送达没有内容")
        for item in request.deliverables:
            caption = item.caption or DELIVERABLE_CAPTIONS.get(item.artifact_type, "")
            if caption:
                outbounds.append(_text_message(conversation, caption))
            outbounds.append(_image_message(conversation, item.object_key))
        prefix = f"deliver-{request.delivery_id}"

    message_ids = await _send_all(conversation, sender, outbounds, idempotency_prefix=prefix)
    project.deliveries_seen.append(request.delivery_id)
    await save_project(conversation, project)
    if not request.HasField("failure"):
        await deliverables_delivered(conversation, sender, delivery_id=request.delivery_id)
    return True, message_ids


def _number_text(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:g}"


async def deliverables_delivered(
    conversation: ConversationRef,
    sender: OutboundSender,
    *,
    delivery_id: str,
) -> bool:
    """**三张图已经发回业主之后**：把按面积推的那套假设摊开说，并给一个改的入口。

    **时点是"产出之后"，不是"输入齐了之后"**（用户 2026-08-31 晚纠正）：裁决原话写的就是
    "产出结果之后也告诉用户……如果他想修改可以再进行修改"，首版却落成了"缺口一空就说"——
    真机上业主刚发完图，先收到一条"我按 4 个人来安排、得房率按 80% 算"，图还没影，
    他不知道这是在说哪份东西。两样齐了那一轮只说"开始设计"（`DESIGN_START_MESSAGES`）。

    **今天没有调用方，接线时点写死＝"渠道出站发我们自己桶里的图"那一段接通时**——图眼下还
    送不到业主手里（《现在在哪儿.md》"图从会话进来"五段里的第三段未做）。同渲染件与
    `floorplan-parse` 那两处先例："机件先做好、留一个真能调的入口，接线时点写死成事件名"，
    不留一句 TODO 注释——注释调不了，入口调得了，测试也就拦得住。

    只说一次（`ProjectState.assumptions_told`）。面积取不到就**响亮记一条日志并返回 False**，
    不拿默认面积顶上（《纪律·拿不到就说没有，不许填猜的值》）。

    参数 `delivery_id`：这次送达的标识，用来派生出站幂等键——同一次送达重投不会说两遍。
    返回：这次是否真的说了。
    """
    project = await find_or_create_project(conversation)
    if project.assumptions_told:
        return False
    area_sqm = orchestrator.find_building_area_sqm(project)
    if area_sqm is None:
        logger.warning(
            "assumptions not told: project=%s 没有建筑面积，按什么做的说不出来",
            project.project_id,
        )
        return False

    project.assumptions_told = True
    told = assumption_messages(infer_from_area(area_sqm))
    outbounds = [_text_message(conversation, text) for text in told]
    await _send_all(
        conversation, sender, outbounds, idempotency_prefix=f"assumptions-{delivery_id}"
    )
    await save_project(conversation, project)
    logger.info("assumptions told: project=%s area_sqm=%s", project.project_id, area_sqm)
    return True


async def _send_all(
    conversation: ConversationRef,
    sender: OutboundSender,
    outbounds: Sequence[message_pb2.UnifiedMessage],
    *,
    idempotency_prefix: str,
) -> list[str]:
    """逐条发出，条与条之间按分条节拍歇一下，并把出站原文与上下文历史一并记上。

    **回话与主动消息共用这一个出口**：分条那条规矩（用户裁决 2026-08-31）此前只管住了模型的
    回复数组；系统写死的文案若自己另写一段发送逻辑，就绕过了停顿节拍、幂等键与落存——
    出口只留一个，绕不过去。
    """
    message_ids: list[str] = []
    for seq, outbound in enumerate(outbounds):
        if seq > 0:
            await asyncio.sleep(_pacing_seconds(_outbound_text(outbounds[seq - 1])))
        idempotency_key = f"{idempotency_prefix}-{seq}"
        message_ids.append(await sender.send(outbound, idempotency_key=idempotency_key))
        await record_outbound(conversation, _outbound_message(outbound, idempotency_key))
        await append_history(
            conversation, ConversationTurn(role="assistant", text=_outbound_text(outbound))
        )
        logger.info(
            "outbound sent: message_id=%s idempotency_key=%s",
            outbound.message_id,
            idempotency_key,
        )
    return message_ids


async def _converse(
    inbound: message_pb2.UnifiedMessage,
    project: ProjectState,
    conversation: ConversationRef,
    user_text: str,
    llm: LlmCompletion,
    capability: CapabilityLookup | None,
) -> tuple[list[str], str | None]:
    """一轮会话：返回（文本回话列表, 需 quick_reply 形态发送的确认清单文本或 None）。"""
    checklist_open = bool(project.open_confirmation_ids)
    intent = await _route(inbound, user_text, llm, checklist_open=checklist_open)

    if intent == "confirm_checklist" and checklist_open:
        upgraded = orchestrator.upgrade_confirmed(project)
        logger.info("checklist confirmed: project=%s upgraded=%d", project.project_id, upgraded)
        return [orchestrator.confirm_ack_text()], None

    # 图片入站：先把"他传了户型图"记上再算缺口——否则这一轮还按"还没有图"问，
    # 而他刚传的就是图（真机上问出了"您家在哪个小区？几室几厅？"）
    if inbound.WhichOneof("content") == "image":
        facts = [orchestrator.upload_fact()]
        if inbound.image.object_key:
            facts.append(orchestrator.upload_object_key_fact(inbound.image.object_key))
        else:
            # 渠道侧没落桶就转过来了：图没有键，后面一步都做不了——响亮记日志，不猜一个键
            logger.warning(
                "image inbound without object_key: message_id=%s（渠道侧未落桶）",
                inbound.message_id,
            )
        orchestrator.merge_facts(project, facts)

    turn = await orchestrator.step(llm, project, await get_history(conversation), user_text)
    structural = orchestrator.merge_facts(project, turn.facts)
    # 修正已确认信息 → 撤下确认标记，走重新确认回路
    if project.minimum_inputs_confirmed and any(
        f.cognitive_state != "user_confirmed" for f in orchestrator.confirmable_facts(project)
    ):
        project.minimum_inputs_confirmed = False

    reply_texts = turn.replies or [FALLBACK_REPLY]
    if structural:
        # 结构说明**自成两条**，不再拼在回话尾巴上：拒绝是一件事、两条出路是另一件事，
        # 而拼上去正好把那一条撑成真机上被吐槽的长文（用户 2026-08-31）
        reply_texts = [*reply_texts, *orchestrator.structural_notes()]

    if orchestrator.missing_slots(project) or project.design_start_told:
        return reply_texts, None

    # 面积与户型图两样齐了：**只说开始设计**（用户 2026-08-31 晚纠正）。不出确认清单、
    # 不再要任何信息，也**不在这儿说按什么假设做的**——那套要等图送到业主手里之后才说
    # （`deliverables_delivered`），裁决原话就是"产出结果之后也告诉用户"。
    # 确认闭环那套机件同样没废，时点同样挪到真有产出可确认时。
    project.design_start_told = True
    logger.info("design start told: project=%s", project.project_id)
    return [*reply_texts, *DESIGN_START_MESSAGES], None


def _pacing_seconds(previous_text: str) -> float:
    """下一条之前歇多久：按上一条的长度算，钳在一个短区间里。"""
    return min(
        max(len(previous_text) * _PACING_SECONDS_PER_CHAR, _PACING_MIN_SECONDS),
        _PACING_MAX_SECONDS,
    )


async def _route(
    inbound: message_pb2.UnifiedMessage,
    user_text: str,
    llm: LlmCompletion,
    *,
    checklist_open: bool,
) -> intent_router.Intent:
    """意图路由；quick_reply 选择直通（输入归一化 v1 路径），文本走分类模型。"""
    if inbound.WhichOneof("content") == "quick_reply":
        selected = inbound.quick_reply.selected_option_id
        if selected == orchestrator.CONFIRM_OPTION_ID:
            return "confirm_checklist"
        if selected == orchestrator.CORRECT_OPTION_ID:
            return "correct_checklist"
        return "other"
    return await intent_router.route_intent(llm, user_text, checklist_open=checklist_open)


async def _supports_quick_reply(
    inbound: message_pb2.UnifiedMessage, capability: CapabilityLookup | None
) -> bool:
    if capability is None:
        return False
    try:
        return await capability.supports_quick_reply(inbound.channel_type, inbound.channel_instance)
    except Exception:
        # 能力查询失败按不支持降级（纯文本清单照发，流程不断）
        logger.warning("capability lookup failed, degrade to text checklist", exc_info=True)
        return False


def _conversation_ref(inbound: message_pb2.UnifiedMessage) -> ConversationRef:
    # TODO(identity)：identity 归一后改为渠道无关 user_id 键控（对齐 §6.5）
    return ConversationRef(
        channel_type=inbound.channel_type,
        channel_instance=inbound.channel_instance,
        external_user_id=inbound.external_user_id or inbound.user_id,
    )


def _inbound_message(inbound: message_pb2.UnifiedMessage) -> ChatMessage:
    """入站原文的持久化形态；幂等键 = 渠道消息 id（渠道重投防重存）。"""
    return ChatMessage(
        external_message_id=inbound.message_id,
        direction="inbound",
        content_type=inbound.WhichOneof("content") or "unknown",
        text=_inbound_text(inbound),
        idempotency_key=inbound.message_id,
        occurred_at=_occurred_at(inbound),
    )


def _outbound_message(outbound: message_pb2.UnifiedMessage, idempotency_key: str) -> ChatMessage:
    """出站原文的持久化形态；幂等键与发送键同源（重试重放不重存）。"""
    return ChatMessage(
        external_message_id=outbound.message_id,
        direction="outbound",
        content_type=outbound.WhichOneof("content") or "unknown",
        text=_outbound_text(outbound),
        idempotency_key=idempotency_key,
        occurred_at=_occurred_at(outbound),
    )


def _occurred_at(message: message_pb2.UnifiedMessage) -> datetime | None:
    if not message.HasField("occurred_at"):
        return None
    return message.occurred_at.ToDatetime(tzinfo=UTC)


def _inbound_text(inbound: message_pb2.UnifiedMessage) -> str:
    match inbound.WhichOneof("content"):
        case "text":
            return inbound.text.text
        case "quick_reply":
            return f"[用户选择了：{inbound.quick_reply.selected_option_id}]"
        case "image":
            return "[用户发来一张图片]"  # TODO(genpipe)：floorplan-parse 备用路径接入
        case "audio":
            return "[用户发来一条语音]"  # TODO(normalize)：语音转文字
        case "card":
            return "[用户分享了一张卡片]"
        case _:
            return "[用户发来一条消息]"


def _text_reply(inbound: message_pb2.UnifiedMessage, text: str) -> message_pb2.UnifiedMessage:
    reply = _reply_envelope(inbound)
    reply.text.CopyFrom(message_pb2.TextContent(text=text))
    return reply


def _text_message(conversation: ConversationRef, text: str) -> message_pb2.UnifiedMessage:
    """主动消息的文本形态：**没有入站消息可挂**，信封只能从会话三元组来。

    渠道侧 user_id 这里给不出（会话键里没有）——TODO(identity)：identity 归一后
    与 `_reply_envelope` 一起改为渠道无关 user_id。
    """
    message = _outbound_envelope(
        channel_type=conversation.channel_type,
        channel_instance=conversation.channel_instance,
        external_user_id=conversation.external_user_id,
    )
    message.text.CopyFrom(message_pb2.TextContent(text=text))
    return message


def _image_message(conversation: ConversationRef, object_key: str) -> message_pb2.UnifiedMessage:
    """主动消息的图片形态：只带私有桶对象键，渠道侧按键取桶再发（渠道出站那一段 9-01 已通）。"""
    message = _outbound_envelope(
        channel_type=conversation.channel_type,
        channel_instance=conversation.channel_instance,
        external_user_id=conversation.external_user_id,
    )
    message.image.CopyFrom(message_pb2.ImageContent(object_key=object_key))
    return message


def _quick_reply_checklist(
    inbound: message_pb2.UnifiedMessage, checklist_text: str
) -> message_pb2.UnifiedMessage:
    reply = _reply_envelope(inbound)
    reply.quick_reply.CopyFrom(
        message_pb2.QuickReplyContent(
            prompt_text=checklist_text,
            options=[
                message_pb2.QuickReplyOption(
                    option_id=orchestrator.CONFIRM_OPTION_ID, label="确认无误"
                ),
                message_pb2.QuickReplyOption(
                    option_id=orchestrator.CORRECT_OPTION_ID, label="有要修正的"
                ),
            ],
        )
    )
    return reply


def _reply_envelope(inbound: message_pb2.UnifiedMessage) -> message_pb2.UnifiedMessage:
    return _outbound_envelope(
        channel_type=inbound.channel_type,
        channel_instance=inbound.channel_instance,
        external_user_id=inbound.external_user_id,
        user_id=inbound.user_id,
    )


def _outbound_envelope(
    *,
    channel_type: int,
    channel_instance: str,
    external_user_id: str,
    user_id: str = "",
) -> message_pb2.UnifiedMessage:
    """出站消息信封（回话与主动消息共用）。"""
    message = message_pb2.UnifiedMessage(
        message_id=str(ULID()),
        # 会话键里的渠道类型是裸 int（ConversationRef 与渠道协议解耦），proto 侧那个枚举
        # 本身就是 int 的子类——这里只还原类型声明，不做任何取值换算
        channel_type=cast(channel_type_pb2.ChannelType, channel_type),
        channel_instance=channel_instance,
        direction=message_pb2.MESSAGE_DIRECTION_OUTBOUND,
        external_user_id=external_user_id,
        user_id=user_id,
    )
    message.occurred_at.GetCurrentTime()
    return message


def _outbound_text(outbound: message_pb2.UnifiedMessage) -> str:
    match outbound.WhichOneof("content"):
        case "quick_reply":
            return outbound.quick_reply.prompt_text
        case "image":
            return f"[发出一张图：{outbound.image.object_key}]"
        case _:
            return outbound.text.text
