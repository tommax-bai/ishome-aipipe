"""解析一族 activity（2026-09-04 接线）：吃对象键、走桶、失败分两种。

依赖全用假件（桶、网关、回调 HTTP），只验 activity 这一层的判据：键怎么验、图怎么认、
产物写到哪、结论长什么样、什么情况返回 failed、什么情况抛出让 Temporal 重试。
几何提取本身的判据在 test_floorplan_geometry，批注/文案的判据在各自的测试里，这里不重验。
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from typing import Any

import httpx
import pytest
from genpipe_worker.activities import (
    FloorplanActivities,
    TaskResultDeliveryError,
    image_media_type_of,
    reading_archive,
)
from genpipe_worker.models import FloorplanFeatures, FloorplanReading, FloorplanSurvey, RoomRegion
from genpipe_worker.object_store import GEOMETRY_ARTIFACT, READING_ARTIFACT, ObjectStoreError
from PIL import Image, ImageDraw

_PLAN_LEFT_PX, _PLAN_TOP_PX, _PLAN_RIGHT_PX, _PLAN_BOTTOM_PX = 80, 80, 520, 520
_PARTITION_X_PX = 300
_DOOR_TOP_PX, _DOOR_BOTTOM_PX = 260, 330


def _two_room_plan() -> bytes:
    """两间房 + 一道隔墙 + 一个门洞（与 test_floorplan_geometry 同一张合成图）。"""
    page = Image.new("RGB", (600, 600), (255, 255, 255))
    pen = ImageDraw.Draw(page)
    pen.rectangle(
        [_PLAN_LEFT_PX, _PLAN_TOP_PX, _PLAN_RIGHT_PX, _PLAN_BOTTOM_PX], outline=(0, 0, 0), width=12
    )
    pen.rectangle(
        [_PARTITION_X_PX - 4, _PLAN_TOP_PX, _PARTITION_X_PX + 4, _DOOR_TOP_PX], fill=(0, 0, 0)
    )
    pen.rectangle(
        [_PARTITION_X_PX - 4, _DOOR_BOTTOM_PX, _PARTITION_X_PX + 4, _PLAN_BOTTOM_PX],
        fill=(0, 0, 0),
    )
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    return buffer.getvalue()


_PLAN_PNG = _two_room_plan()
_SHA = hashlib.sha256(_PLAN_PNG).hexdigest()
_KEY = f"uploads/{_SHA}/original.png"
_SURVEY = {
    "northPointsTo": None,
    "rooms": [
        {"name": "西屋", "box": [0.16, 0.16, 0.48, 0.85]},
        {"name": "东屋", "box": [0.52, 0.16, 0.85, 0.85]},
    ],
}


class _FakeStore:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.written: dict[str, bytes] = {}

    def get_original(self, key: str) -> bytes:
        if not re.match(r"^uploads/[0-9a-f]{64}/original\.(jpg|png|webp|gif|bmp)$", key):
            raise ObjectStoreError([f"键不合形态：{key}"])
        if key not in self.objects:
            raise ObjectStoreError([f"桶里没有 {key}"])
        return self.objects[key]

    def put_derived_json(self, key: str, artifact: str, payload: bytes) -> str:
        derived = f"{key.rsplit('/', 1)[0]}/{artifact}"
        self.written[derived] = payload
        return derived


class _FakeLlm:
    """勘测回固定的两间房；批注引用提示词里出现的第一条事实 id；文案不带数字。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete_with_image(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        image_media_type: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(f"image:{model}")
        return json.dumps(_SURVEY, ensure_ascii=False)

    async def complete_text(
        self, model: str, system_prompt: str, user_prompt: str, *, temperature: float = 0.0
    ) -> str:
        self.calls.append(f"text:{model}")
        if "notes" in model:
            ids = re.findall(r"^- (plan-[^：]+)：", user_prompt, flags=re.MULTILINE)
            rooms = re.search(r"房间清单：(.+)", user_prompt)
            names = rooms.group(1).split("、") if rooms else ["西屋"]
            notes = [
                {
                    "room": names[i % len(names)],
                    "text": f"这间房值得说的第{'一二三'[i]}件事",
                    "cites": [ids[i]],
                }
                for i in range(min(3, len(ids)))
            ]
            return json.dumps({"notes": notes}, ensure_ascii=False)
        return json.dumps(
            {
                "title": "光照进来的家",
                "summary": "两间房各有各的用处。",
                "tips": ["先定动线", "再定收纳", "最后挑灯"],
            },
            ensure_ascii=False,
        )


def _activities(
    store: _FakeStore | None = None,
    llm: _FakeLlm | None = None,
    http: httpx.AsyncClient | None = None,
) -> FloorplanActivities:
    return FloorplanActivities(
        store or _FakeStore({_KEY: _PLAN_PNG}),  # type: ignore[arg-type]
        llm or _FakeLlm(),  # type: ignore[arg-type]
        http or httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )


# ---------------------------------------------------------------------------
# 图怎么认
# ---------------------------------------------------------------------------


def test_media_type_is_read_from_bytes_not_extension() -> None:
    assert image_media_type_of(_PLAN_PNG) == "image/png"
    assert image_media_type_of(b"\xff\xd8\xff\xe0" + b"\x00" * 16) == "image/jpeg"
    assert image_media_type_of(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    with pytest.raises(ValueError):
        image_media_type_of(b"GIF89a" + b"\x00" * 16)


# ---------------------------------------------------------------------------
# 几何提取：键 → 桶 → 勘测 → 几何 → 事实 → 存档
# ---------------------------------------------------------------------------


async def test_geometry_extract_returns_geometry_facts_and_archives_them() -> None:
    store = _FakeStore({_KEY: _PLAN_PNG})
    llm = _FakeLlm()
    result = await _activities(store, llm).extract_floorplan_geometry(
        {"floorplan_object_key": _KEY}
    )

    assert result["verdict"] == "ok", result
    assert result["geometry_key"] == f"uploads/{_SHA}/{GEOMETRY_ARTIFACT}"
    assert sorted(result["room_names"]) == ["东屋", "西屋"]
    assert result["facts"], "几何一出来事实就该跟着出来（确定性）"
    assert result["model_call_count"] == 1 and llm.calls == ["image:floorplan-parse.default"]
    archive = json.loads(store.written[result["geometry_key"]])
    assert set(archive) == {"image", "modelCallCount", "survey", "geometry", "facts"}
    assert archive["image"]["sha256"] == _SHA
    # 内联回去的几何与存档里的是同一份（母版今天吃内联，将来改走键时两边不会对不上）
    assert archive["geometry"] == result["geometry"]


async def test_geometry_extract_rejects_missing_key_before_touching_anything() -> None:
    llm = _FakeLlm()
    result = await _activities(llm=llm).extract_floorplan_geometry({})
    assert result["verdict"] == "failed"
    assert result["violations"][0]["check"] == "gate-missing-floorplan-key"
    assert llm.calls == []


async def test_geometry_extract_reports_store_failure_loud() -> None:
    result = await _activities(_FakeStore({})).extract_floorplan_geometry(
        {"floorplan_object_key": _KEY}
    )
    assert result["verdict"] == "failed"
    assert result["violations"][0]["check"] == "object-store-failed"


async def test_unsupported_image_bytes_fail_without_calling_the_model() -> None:
    gif_key = f"uploads/{'d' * 64}/original.gif"
    llm = _FakeLlm()
    result = await _activities(
        _FakeStore({gif_key: b"GIF89a" + b"\x00" * 32}), llm
    ).extract_floorplan_geometry({"floorplan_object_key": gif_key})
    assert result["violations"][0]["check"] == "gate-unsupported-image"
    assert llm.calls == []


# ---------------------------------------------------------------------------
# 批注 / 文案
# ---------------------------------------------------------------------------


async def test_notes_and_copy_take_facts_and_return_checked_products() -> None:
    impl = _activities()
    geometry = await impl.extract_floorplan_geometry({"floorplan_object_key": _KEY})
    notes = await impl.write_plan_notes(
        {"facts": geometry["facts"], "room_names": geometry["room_names"]}
    )
    assert notes["verdict"] == "ok", notes
    assert len(notes["notes"]) == 3
    assert all(note["cites"] for note in notes["notes"])

    copy = await impl.write_plan_copy({"facts": geometry["facts"]})
    assert copy["verdict"] == "ok", copy
    assert copy["copy"]["title"] == "光照进来的家"
    assert len(copy["copy"]["tips"]) == 3


async def test_notes_reject_malformed_facts_before_calling_the_model() -> None:
    llm = _FakeLlm()
    result = await _activities(llm=llm).write_plan_notes({"facts": [{"nope": 1}], "room_names": []})
    assert result["violations"][0]["check"] == "gate-bad-facts"
    assert llm.calls == []


# ---------------------------------------------------------------------------
# 特征解析：只验 activity 这一层（读图三步件各有各的测试）
# ---------------------------------------------------------------------------


async def test_parse_archives_reading_and_returns_layout_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reading = FloorplanReading(
        logical_model="floorplan-parse.default",
        raw_output="{}",
        survey=FloorplanSurvey(rooms=[RoomRegion(name="西屋", box=(0.1, 0.1, 0.5, 0.9))]),
        room_legends=[],
        orientations=[],
        verdicts=[],
        features=FloorplanFeatures(layout_features={"balcony_utility": "阳台画了洗衣机"}),
        model_call_count=3,
    )

    async def fake_read(*args: Any, **kwargs: Any) -> FloorplanReading:
        return reading

    monkeypatch.setattr("genpipe_worker.activities.read_floorplan_features", fake_read)
    store = _FakeStore({_KEY: _PLAN_PNG})
    result = await _activities(store).parse_floorplan({"floorplan_object_key": _KEY})

    assert result["verdict"] == "ok"
    assert result["reading_key"] == f"uploads/{_SHA}/{READING_ARTIFACT}"
    assert result["layout_features"] == {"balcony_utility": "阳台画了洗衣机"}
    assert result["model_call_count"] == 3
    archive = json.loads(store.written[result["reading_key"]])
    assert archive["product"]["layoutFeatures"] == {"balcony_utility": "阳台画了洗衣机"}
    assert archive == reading_archive(reading, _PLAN_PNG, archive["elapsedSeconds"])


# ---------------------------------------------------------------------------
# 结果回流：2xx 收、4xx 按失败收、5xx/连不上抛出交给重试
# ---------------------------------------------------------------------------


async def test_deliver_posts_result_to_the_injected_callback_url() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"task_id": "t", "accepted": True, "duplicate": False})

    impl = _activities(http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await impl.deliver_task_result(
        {
            "result_callback_url": "http://project/api/v1/generation-tasks/t/result",
            "result": {"task_id": "t", "status": "completed", "products": []},
        }
    )
    assert result == {
        "verdict": "ok",
        "status_code": 200,
        "receipt": {"task_id": "t", "accepted": True, "duplicate": False},
    }
    assert seen[0].method == "POST" and seen[0].url.path == "/api/v1/generation-tasks/t/result"
    assert json.loads(seen[0].content)["status"] == "completed"


async def test_deliver_treats_4xx_as_rejection_not_retry() -> None:
    impl = _activities(
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(404, text="no such task"))
        )
    )
    result = await impl.deliver_task_result(
        {"result_callback_url": "http://project/x", "result": {"task_id": "t"}}
    )
    assert result["verdict"] == "failed"
    assert result["violations"][0]["check"] == "callback-rejected-404"


async def test_deliver_raises_on_5xx_and_transport_errors_so_temporal_retries() -> None:
    flaky = _activities(
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    )
    with pytest.raises(TaskResultDeliveryError):
        await flaky.deliver_task_result({"result_callback_url": "http://project/x", "result": {}})

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    down = _activities(http=httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    with pytest.raises(TaskResultDeliveryError):
        await down.deliver_task_result({"result_callback_url": "http://project/x", "result": {}})


async def test_deliver_without_callback_is_a_gate_failure() -> None:
    result = await _activities().deliver_task_result({"result": {}})
    assert result["violations"][0]["check"] == "gate-missing-callback"
