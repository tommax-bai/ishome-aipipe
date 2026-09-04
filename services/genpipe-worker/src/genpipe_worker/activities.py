"""Temporal activities：所有 IO 与重计算收口在此。

注册名唯一真源：ishome-contracts `activities/registry.md`，**只增不改**——改注册名
会破坏历史 workflow 重放，等同于改线上协议；新增走 contracts 仓 PR 评审。
命名规则（规范 §2.4）：注册名 = kebab-case 显式声明；函数名 = 同词 snake_case
动词前置。

V1.4 裁决（2026-08-23）：绘图 activity 物理拆分迁出本仓——
`plan-2d-render` → ishome-render2d（队列 `render2d-activities`）；
`atmosphere-visual` / `realism-pass` → ishome-imagegen（队列 `imagegen-activities`）；
`scene-compile` / `base-render` → ishome-render3d（队列 `render3d-activities`）。
本仓保留非绘图 activity：解析、求解、校验、门禁（队列 `genpipe-activities`）。

**2026-09-04 接线**（上传入口就绪，`floorplan-parse` 当初写死的时点到了）：解析这一族从
"纯库 + CLI"接进 activity。入参是**对象键**不是本地路径（当初后置的理由正是"提前接线等于
接一遍再改一遍"）；入参与出参都是不透明字典——派发方（genpipe 编排）不 import 本仓存根签名，
两边只靠 contracts 注册名接头（同 render2d / reportrender 的口径）。

实现件做成类（同 render2d `PlanRenderer`）：它要用三样**进程级**的东西——私有桶连接、
网关客户端、回调用的 HTTP 客户端——都该在起进程时装好并当场校验，不是等第一张图来了才发现。

失败形态两种，分得开：
- **结构性失败**（键不合形态、图不认识、几何不通过、批注不够）→ 返回 `verdict=failed` 带
  violations，编排按失败收——重派也不会变好；
- **瞬时失败**（网关 5xx/连不上、回调地址连不上）→ **抛异常**，交给 Temporal activity 重试。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
from pydantic import ValidationError
from temporalio import activity

from genpipe_worker.floorplan_copy import PlanCopyError, write_copy
from genpipe_worker.floorplan_facts import derive_facts
from genpipe_worker.floorplan_geometry import FloorplanGeometryError, extract_geometry
from genpipe_worker.floorplan_notes import PlanNotesError, write_notes
from genpipe_worker.floorplan_parse import (
    PARSE_LOGICAL_MODEL,
    FloorplanParseError,
    read_floorplan_features,
)
from genpipe_worker.floorplan_regions import RoomCropError, RoomLegendError
from genpipe_worker.floorplan_survey import FloorplanSurveyError, survey_floorplan
from genpipe_worker.layout_features import LayoutFeatureSetError, LayoutFeatureViolation
from genpipe_worker.llm_client import LiteLlmVisionClient
from genpipe_worker.models import FloorplanReading, PlanFact
from genpipe_worker.object_store import (
    GEOMETRY_ARTIFACT,
    READING_ARTIFACT,
    ObjectStoreError,
    OssUploadStore,
)

ActivityResult = dict[str, Any]

ACTIVITY_FLOORPLAN_PARSE = "floorplan-parse"
ACTIVITY_FLOORPLAN_GEOMETRY_EXTRACT = "floorplan-geometry-extract"
ACTIVITY_PLAN_NOTES_WRITE = "plan-notes-write"
ACTIVITY_PLAN_COPY_WRITE = "plan-copy-write"
ACTIVITY_TASK_RESULT_DELIVER = "task-result-deliver"
"""contracts 注册名（#1 / #15 / #16 / #17 / #19）。字符串在此声明一次，worker 与守门测试都引它。"""

_IMAGE_MEDIA_TYPES: tuple[tuple[bytes, int, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"WEBP", 8, "image/webp"),
)
"""按**字节首部**认图片格式（同渠道侧落桶时的判法），不按键的扩展名猜——键里的 ext 也是
字节判出来的，两边同一把尺子。gif / bmp 能落桶但解析模型不吃，这里就认不出、当场失败。"""


class TaskResultDeliveryError(Exception):
    """回调送不到（连不上 / 5xx）——抛出让 Temporal 重试，不在本层吞掉。"""


def image_media_type_of(image_bytes: bytes) -> str:
    """字节首部 → MIME 类型；认不出即失败（纯函数）。"""
    for magic, offset, media_type in _IMAGE_MEDIA_TYPES:
        if image_bytes[offset : offset + len(magic)] == magic:
            if media_type == "image/webp" and image_bytes[:4] != b"RIFF":
                continue
            return media_type
    raise ValueError("图片格式认不出（认得：png / jpeg / webp）")


def reading_archive(
    reading: FloorplanReading, image_bytes: bytes, elapsed_seconds: float
) -> dict[str, Any]:
    """解析存档的形态，与 `floorplan-parse` CLI 的 `-o` 产物逐字同构（少一个本地 path）。纯函数。"""
    features = reading.features
    return {
        "image": {"sha256": hashlib.sha256(image_bytes).hexdigest(), "bytes": len(image_bytes)},
        "logicalModel": reading.logical_model,
        "modelCallCount": reading.model_call_count,
        "elapsedSeconds": round(elapsed_seconds, 1),
        "survey": reading.survey.model_dump(mode="json", by_alias=True),
        "roomLegends": [
            legend.model_dump(mode="json", by_alias=True) for legend in reading.room_legends
        ],
        "orientations": [
            item.model_dump(mode="json", by_alias=True) for item in reading.orientations
        ],
        "rawOutput": reading.raw_output,
        "verdicts": [v.model_dump(mode="json", by_alias=True) for v in reading.verdicts],
        "product": features.model_dump(mode="json", by_alias=True),
    }


class FloorplanActivities:
    """解析一族 activity 的实现件，依赖由组合根（worker）注入。"""

    def __init__(
        self, store: OssUploadStore, llm: LiteLlmVisionClient, http: httpx.AsyncClient
    ) -> None:
        self._store = store
        self._llm = llm
        self._http = http

    async def _load_original(
        self, request: dict[str, Any]
    ) -> tuple[str, bytes, str] | ActivityResult:
        """取源图：键 → 字节 → MIME。三步任一不成即返回失败结果（调用方原样返回）。"""
        floorplan_object_key = str(request.get("floorplan_object_key") or "")
        if not floorplan_object_key:
            return _failed(
                "gate-missing-floorplan-key", "没有 floorplan_object_key：不知道去桶里取哪张图"
            )
        try:
            image_bytes = await asyncio.to_thread(self._store.get_original, floorplan_object_key)
        except ObjectStoreError as e:
            return _violations("object-store-failed", e.details)
        try:
            media_type = image_media_type_of(image_bytes)
        except ValueError as e:
            return _failed("gate-unsupported-image", f"{e}（键 {floorplan_object_key}）")
        return floorplan_object_key, image_bytes, media_type

    @activity.defn(name=ACTIVITY_FLOORPLAN_PARSE)
    async def parse_floorplan(self, request: dict[str, Any]) -> ActivityResult:
        """户型图解析：对象键 → 勘测 → 分区读 → 逐条判定 → 特征标记；存档写回同前缀。

        产物给**报告那条腿**用（特征进画像）。三张图那条链不等它——它与主链并行派发。
        """
        loaded = await self._load_original(request)
        if isinstance(loaded, dict):
            return loaded
        floorplan_object_key, image_bytes, media_type = loaded

        started_at = time.monotonic()
        try:
            reading = await read_floorplan_features(image_bytes, media_type, self._llm)
        except LayoutFeatureViolation as e:
            # 硬门禁：标记名越界或产物不合契约——报出是哪个键，不修剪不降级
            return _violations("layout-feature-violation", e.details)
        except (
            FloorplanParseError,
            FloorplanSurveyError,
            RoomCropError,
            RoomLegendError,
            LayoutFeatureSetError,
        ) as e:
            return _failed("floorplan-parse-failed", str(e))
        elapsed_seconds = time.monotonic() - started_at

        archive = reading_archive(reading, image_bytes, elapsed_seconds)
        try:
            reading_key = await asyncio.to_thread(
                self._store.put_derived_json,
                floorplan_object_key,
                READING_ARTIFACT,
                _json_bytes(archive),
            )
        except ObjectStoreError as e:
            return _violations("object-store-failed", e.details)
        return {
            "verdict": "ok",
            "reading_key": reading_key,
            "layout_features": dict(reading.features.layout_features),
            "room_count": len(reading.survey.rooms),
            "model_call_count": reading.model_call_count,
            "elapsed_seconds": round(elapsed_seconds, 1),
        }

    @activity.defn(name=ACTIVITY_FLOORPLAN_GEOMETRY_EXTRACT)
    async def extract_floorplan_geometry(self, request: dict[str, Any]) -> ActivityResult:
        """对象键 → 勘测一次（唯一一次模型调用）→ 几何（确定性）→ 户型事实（确定性）。

        几何与事实**内联回给编排**：它们是不大的 JSON，母版那一步今天就吃内联几何
        （render2d 改走键的触发条件＝几何产物进 contracts 对象键表）。存档同时写回同前缀，
        report 那条腿与人工复核都能按键取到勘测 + 几何。
        """
        loaded = await self._load_original(request)
        if isinstance(loaded, dict):
            return loaded
        floorplan_object_key, image_bytes, media_type = loaded

        try:
            survey = await survey_floorplan(image_bytes, media_type, self._llm, PARSE_LOGICAL_MODEL)
        except FloorplanSurveyError as e:
            return _failed("floorplan-survey-failed", str(e))
        try:
            geometry = await asyncio.to_thread(extract_geometry, image_bytes, survey.rooms)
        except FloorplanGeometryError as e:
            # 响亮失败：说清缺什么，不给"差不多的"结构（红线一），下游母版拿几何当唯一源
            return _violations("floorplan-geometry-failed", e.details)
        facts = derive_facts(geometry)

        # mode="json"：元组落成列表——内联回编排的那份与存档那份要逐字相同（都走 JSON）
        geometry_payload = geometry.model_dump(mode="json", by_alias=True)
        facts_payload = [fact.model_dump(mode="json", by_alias=True) for fact in facts]
        archive = {
            "image": {"sha256": hashlib.sha256(image_bytes).hexdigest(), "bytes": len(image_bytes)},
            "modelCallCount": 1,
            "survey": survey.model_dump(mode="json", by_alias=True),
            "geometry": geometry_payload,
            "facts": facts_payload,
        }
        try:
            geometry_key = await asyncio.to_thread(
                self._store.put_derived_json,
                floorplan_object_key,
                GEOMETRY_ARTIFACT,
                _json_bytes(archive),
            )
        except ObjectStoreError as e:
            return _violations("object-store-failed", e.details)
        outer = sum(1 for opening in geometry.openings if opening.is_on_outer_wall)
        return {
            "verdict": "ok",
            "geometry_key": geometry_key,
            "geometry": geometry_payload,
            "facts": facts_payload,
            "room_names": [room.name for room in geometry.rooms],
            "model_call_count": 1,
            "wall_count": len(geometry.walls),
            "opening_count": len(geometry.openings),
            "outer_opening_count": outer,
            "room_count": len(geometry.rooms),
            "cell_coverage_ratio": geometry.cell_coverage_ratio,
        }

    @activity.defn(name=ACTIVITY_PLAN_NOTES_WRITE)
    async def write_plan_notes(self, request: dict[str, Any]) -> ActivityResult:
        """事实清单 + 房间清单 → 批注（每句引得到 fact_id，机检在 `floorplan_notes`）。"""
        try:
            facts = [PlanFact.model_validate(item) for item in request.get("facts") or []]
        except (ValidationError, TypeError) as e:
            return _failed("gate-bad-facts", f"事实清单解析失败：{e}")
        room_names = [str(name) for name in request.get("room_names") or []]
        try:
            kept, rejected = await write_notes(facts, room_names, self._llm)
        except PlanNotesError as e:
            return _violations("plan-notes-failed", e.details)
        return {
            "verdict": "ok",
            "notes": [note.model_dump(by_alias=True) for note in kept],
            "rejected": rejected,
        }

    @activity.defn(name=ACTIVITY_PLAN_COPY_WRITE)
    async def write_plan_copy(self, request: dict[str, Any]) -> ActivityResult:
        """事实清单 → 页面文案（标题 / 总结 / 贴士）；数字必须在事实清单里出现过。"""
        try:
            facts = [PlanFact.model_validate(item) for item in request.get("facts") or []]
        except (ValidationError, TypeError) as e:
            return _failed("gate-bad-facts", f"事实清单解析失败：{e}")
        try:
            copy = await write_copy(facts, self._llm)
        except PlanCopyError as e:
            return _violations("plan-copy-failed", e.details)
        return {"verdict": "ok", "copy": copy.model_dump(by_alias=True)}

    @activity.defn(name=ACTIVITY_TASK_RESULT_DELIVER)
    async def deliver_task_result(self, request: dict[str, Any]) -> ActivityResult:
        """把编排归并好的结论 `POST` 到派发时注入的回调地址（project.v1 `generation_task_result`）。

        编排侧不知道业务侧在哪——地址由上层注入（规范 §1.0"向上通信只走事件/回调"）。
        连不上 / 5xx 抛异常交给 Temporal 重试；4xx 是对方明确拒收，重试也不会变好，按失败收。
        """
        callback_url = str(request.get("result_callback_url") or "")
        result = request.get("result")
        if not callback_url or not isinstance(result, dict):
            return _failed("gate-missing-callback", "没有回调地址或结果体：结论无处可送")
        try:
            response = await self._http.post(callback_url, json=result)
        except httpx.HTTPError as e:
            raise TaskResultDeliveryError(f"回调 {callback_url} 送不到：{e}") from e
        if response.status_code >= 500:
            raise TaskResultDeliveryError(
                f"回调 {callback_url} 回了 {response.status_code}：{response.text[:300]}"
            )
        if response.status_code >= 400:
            return _failed(
                f"callback-rejected-{response.status_code}",
                f"业务侧拒收（{callback_url}）：{response.text[:300]}",
            )
        receipt: Any
        try:
            receipt = response.json()
        except ValueError:
            receipt = {}
        return {"verdict": "ok", "status_code": response.status_code, "receipt": receipt}


@activity.defn(name="plan-layout-solve")
async def solve_plan_layout(plan_revision_id: str) -> ActivityResult:
    """自动布局与尺寸计算（确定性求解）。"""
    raise NotImplementedError


@activity.defn(name="plan-rule-check")
async def check_plan_rules(plan_revision_id: str) -> ActivityResult:
    """空间规则校验（碰撞/通道/边界闭合）。"""
    raise NotImplementedError


@activity.defn(name="consistency-check")
async def check_consistency(artifact_id: str) -> ActivityResult:
    """户型与跨视角一致性校验（含母版遮罩比对的确定性 QA）。"""
    raise NotImplementedError


@activity.defn(name="compliance-check")
async def check_compliance(artifact_id: str) -> ActivityResult:
    """内容安全机检：工厂与交互两条路径都强制。"""
    raise NotImplementedError


ActivityCallable = Callable[..., Coroutine[Any, Any, ActivityResult]]


def activity_registry(floorplan: FloorplanActivities) -> dict[str, ActivityCallable]:
    """注册名 → 实现。键与 contracts 注册表中归属 `genpipe-activities` 队列的子集逐字一致
    （tests/test_activity_registry.py 按 registries/task_queues.md 口径断言）。"""
    return {
        ACTIVITY_FLOORPLAN_PARSE: floorplan.parse_floorplan,
        "plan-layout-solve": solve_plan_layout,
        "plan-rule-check": check_plan_rules,
        "consistency-check": check_consistency,
        "compliance-check": check_compliance,
        ACTIVITY_FLOORPLAN_GEOMETRY_EXTRACT: floorplan.extract_floorplan_geometry,
        ACTIVITY_PLAN_NOTES_WRITE: floorplan.write_plan_notes,
        ACTIVITY_PLAN_COPY_WRITE: floorplan.write_plan_copy,
        ACTIVITY_TASK_RESULT_DELIVER: floorplan.deliver_task_result,
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _failed(check: str, detail: str) -> ActivityResult:
    return {"verdict": "failed", "violations": [{"check": check, "detail": detail}]}


def _violations(check: str, details: list[str]) -> ActivityResult:
    return {"verdict": "failed", "violations": [{"check": check, "detail": d} for d in details]}
