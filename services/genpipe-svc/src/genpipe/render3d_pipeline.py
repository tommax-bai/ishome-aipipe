"""三维线编排：scene-compile → base-render（多台机位一次）→ realism-pass（每台机位一张，并行）。

只编排，禁止任何 IO（与 `workflows.py` 同一条可重放硬约束，import-linter 契约锁定）。本模块
不 import Temporal：派发经注入的 `StepRunner` 完成，所以这条链既能在 workflow 沙箱里跑，也能
用假的派发器直测，不起 Temporal。

链路出处：对齐文档 §3.1（三维底渲交互引擎专用、写实化复用）；三个 activity 的出入参照 contracts
`activities/registry.md`"三维线与写实化的出入参"节（realism-pass 照 imagegen 实装 e859aa6，
另两条是草案）——入参 camelCase、回执 snake_case，两侧实装如此。

编排取舍（本模块的全部策略，写在此处以免两头下注）：

- **单机位失败不拖垮整户**：某台机位取景失败或写实化失败，记进 `failed_cameras`，其余机位照出；
  **全部机位都失败才整体失败**。一张效果图对业主是一张图不是半本册（与报告成文线"任一域失败整册
  失败"相反——那边的判据是册级闭合，这边没有跨机位的闭合判据）。
- **门禁不过重派一次**：realism-pass 的门禁 `judged=true & passed=false`（回执
  `fidelity-gate-failed`）时用**同样入参**再派一次；`judged=false`（下限没配、只记录不判）不重派——没判就没有"不过"。
  上限 `MAX_GATE_REDISPATCHES=1`，**待用户定 K**。重派入参逐次相同（seed 也不换）：赌的是同一次
  生成的随机性，改入参就成了"换个题面再试"，那是另一件事（同报告成文线"整章重开"的口径）。
- **视角不猜**：`viewKind` 先取 base-render 回执里那台机位的 `view_kind`/`kind`，回执没有就取
  spec 给的；两处都没有，这台机位按失败记（`missing-view-kind`），不按机位 id 的字样猜。
- **几何源**：`sketch_key` 有就用它（专为控制通道画的稿），没有用 `line_key`。
- **渲染只有一档**（用户裁决 2026-09-04）：三条 activity 入参都不带档位。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from genpipe.models import TaskQueues

# activity 注册名（与 contracts 注册表逐字一致，只增不改）。`workflows.py` 里有同名常量，两处
# 字面量相同由 tests/test_render3d_pipeline.py 守门——本模块不能 import workflows
# （那边 import 这边）。
ACTIVITY_SCENE_COMPILE = "scene-compile"
ACTIVITY_BASE_RENDER = "base-render"
ACTIVITY_REALISM_PASS = "realism-pass"

FIDELITY_GATE_CHECK = "fidelity-gate-failed"
"""imagegen realism-pass 门禁不过时 violations.check 的值（e859aa6）。"""

MAX_GATE_REDISPATCHES = 1
"""门禁不过后**同样入参再派**的次数上限。**待用户定 K**：先按 1，一次重派＝再花一张图的钱。"""

ViewKind = Literal["bird", "room"]
"""与 imagegen `RealismPassRequest.view_kind` 同词表：`bird` 揭顶鸟瞰 / `room` 室内机位。"""

SpaceRenderVerdict = Literal["ok", "failed"]

CameraStage = Literal["base-render", "realism-pass"]
"""某台机位失败在哪一步：取景（底渲）还是写实化。"""


class SpaceCamera(BaseModel):
    """派发方指定的一台机位：id 是场景包 `cameras[].id`；视角与 seed 都可选。"""

    model_config = ConfigDict(extra="forbid")

    camera_id: str
    view_kind: ViewKind | None = None
    """base-render 回执不带视角时的兜底；两处都没有这台机位按失败记，不猜。"""
    seed: int | None = None
    """原样送 realism-pass；不给＝那一跑不可复现，编排侧不替派发方铸。"""


class SpaceRenderSpec(BaseModel):
    """三维线一次派发的编排输入。未知字段拒收：两侧字段口径对不上就是接不上头，不静默丢。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    design_package_key: str
    """三维输入包在私有桶里的键（activity 吃键不吃本地路径：包几百 KB，workflow 历史不该背它）。"""
    revision_id: str
    cameras: list[SpaceCamera] | None = None
    """要渲的机位；None＝场景包里全部机位（base-render `cameraIds=None`），届时视角只能来自回执。"""
    style_template_id: str
    """写实风格模板 id（imagegen `templates/realism/*.json`）。"""
    width_px: int = 1024
    height_px: int = 768
    queues: TaskQueues = Field(default_factory=TaskQueues)


class CameraRender(BaseModel):
    """一台机位出齐了：底渲五路键 + 写实图键 + 自证数 + 门禁结果。"""

    camera_id: str
    view_kind: ViewKind
    geometry_key: str
    depth_key: str
    line_key: str
    mask_key: str
    mask_index_key: str
    sketch_key: str | None = None
    control_key: str
    """实际送给 realism-pass 当几何源的那一路（sketch 有则 sketch，否则 line）。"""
    image_object_key: str
    content_type: str | None = None
    fidelity_score: float | None = None
    gate: dict[str, Any] = Field(default_factory=dict)
    """imagegen 门禁回执原样：`{fidelity_score, min_fidelity_score, judged, passed, reason}`。"""
    evidence: dict[str, Any] = Field(default_factory=dict)
    """写实化自证数原样：`backend_name` / `seed` / `prompt_sha256` / `elapsed_seconds` 等。"""
    gate_redispatches: int = 0
    """这张图是第几次重派出来的：0＝一次过。留着它是为了让"靠重派出来的"在结论里判得出来。"""


class FailedCamera(BaseModel):
    """一台机位没出来：在哪一步、为什么；底渲若已成功，五路键照留（血缘）。"""

    camera_id: str
    stage: CameraStage
    failed_checks: list[str]
    """`check=detail` 逐条（activity 的 violations 压平）或编排层失败码。"""
    violations: list[dict[str, Any]] = Field(default_factory=list)
    """activity 的违规清单原样透传，不改写。"""
    render_keys: dict[str, str] = Field(default_factory=dict)
    """底渲成功但写实化失败时的五路键（含 sketch），失败机位也留血缘。"""
    gate: dict[str, Any] | None = None
    """写实化最后一次尝试的门禁回执（门禁不过那条路径才有）。"""
    gate_redispatches: int = 0


class SpaceRenderResult(BaseModel):
    """三维线一次运行的结论：成功机位与失败机位分列，整体只在全部失败时才是 failed。"""

    task_id: str
    verdict: SpaceRenderVerdict
    scene_package_key: str | None = None
    scene_evidence: dict[str, Any] = Field(default_factory=dict)
    """scene-compile 自证数原样（`metre_per_unit` / `area_match_ratio` / `heights_source` …），
    不判。"""
    base_render_evidence: dict[str, Any] = Field(default_factory=dict)
    """base-render 回执里机位之外的自证数原样（`width_px` / `height_px` …），不判。"""
    renders: list[CameraRender] = Field(default_factory=list)
    failed_cameras: list[FailedCamera] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    """整户级失败码（scene-compile 失败、base-render 一台都没出来）；机位级的在
    failed_cameras 里。"""
    failure: dict[str, Any] | None = None
    """整体失败原因 {code, detail}；verdict=ok 时为空。"""


class DispatchFailure(Exception):
    """派发层失败（activity 抛错重试耗尽 / 回执不是 dict）：带是哪一步、为什么。

    由注入的 `StepRunner` 抛出——本模块不 import Temporal，认不出 ActivityError，认这个。
    """

    def __init__(self, activity_name: str, detail: str) -> None:
        super().__init__(f"{activity_name}:{detail}")
        self.activity_name = activity_name
        self.detail = detail


StepRunner = Callable[[str, Any, str], Awaitable[dict[str, Any]]]
"""(activity 注册名, 入参, task_queue) → 回执原样（ok 与 failed 都回，本模块自己看 verdict）；
派发本身失败抛 `DispatchFailure`。workflow 侧用 `workflow.execute_activity` 实现它，测试用假的。"""


_RECEIPT_HOUSEKEEPING_KEYS = frozenset({"verdict", "violations"})


def violation_checks(receipt: Mapping[str, Any]) -> list[str]:
    """把回执的 violations 压成 `check=detail` 逐条（纯函数）；failed 却不给清单＝失败无理由，
    补一条编排层失败码，杜绝失败被吞掉。"""
    raw = receipt.get("violations")
    items = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    if not items:
        return ["failed-without-violations"]
    return [f"{item.get('check', '?')}={item.get('detail', '')}" for item in items]


def collect_violations(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = receipt.get("violations")
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def evidence_of(receipt: Mapping[str, Any], *exclude: str) -> dict[str, Any]:
    """回执里除 verdict/violations 与指定键之外的全部字段（自证数原样，不判）。"""
    skip = _RECEIPT_HOUSEKEEPING_KEYS.union(exclude)
    return {key: value for key, value in receipt.items() if key not in skip}


def _nonempty_str(receipt: Mapping[str, Any], field: str) -> str | None:
    value = receipt.get(field)
    return value if isinstance(value, str) and value else None


def is_gate_rejection(receipt: Mapping[str, Any]) -> bool:
    """realism-pass 回执是不是"门禁判了且不过"（纯函数）：只有这一种失败值得同样入参再派。

    判据取回执 `gate.judged is True and gate.passed is False`；没有 gate 字段时退回看
    violations.check 是否 `fidelity-gate-failed`（imagegen 两处同时给，缺一按另一处）。
    `judged=false` 不算：没判就没有"不过"，重派也不会变成"过"。
    """
    gate = receipt.get("gate")
    if isinstance(gate, dict) and "judged" in gate:
        return gate.get("judged") is True and gate.get("passed") is False
    return any(item.get("check") == FIDELITY_GATE_CHECK for item in collect_violations(receipt))


def resolve_view_kind(entry: Mapping[str, Any], requested: SpaceCamera | None) -> ViewKind | None:
    """一台机位的视角（纯函数）：先回执（`view_kind` 或 `kind`），后 spec；都没有回 None，不猜。"""
    for field in ("view_kind", "kind"):
        value = entry.get(field)
        if value in ("bird", "room"):
            return "bird" if value == "bird" else "room"
    if requested is not None and requested.view_kind is not None:
        return requested.view_kind
    return None


def realism_request(
    render: Mapping[str, str],
    *,
    view_kind: ViewKind,
    style_template_id: str,
    seed: int | None,
) -> dict[str, Any]:
    """realism-pass 入参（纯函数，字段照 imagegen `RealismPassRequest` camelCase）：
    sketch 有用 sketch、没有用 line，二者只给一个；seed 不给就一个字段都不带。"""
    request: dict[str, Any] = {
        "styleTemplateId": style_template_id,
        "viewKind": view_kind,
        "cameraId": render["camera_id"],
    }
    if render.get("sketch_key"):
        request["sketchKey"] = render["sketch_key"]
    else:
        request["lineKey"] = render["line_key"]
    if seed is not None:
        request["seed"] = seed
    return request


_RENDER_KEY_FIELDS = ("geometry_key", "depth_key", "line_key", "mask_key", "mask_index_key")


def partition_base_render(
    receipt: Mapping[str, Any], requested: list[SpaceCamera] | None
) -> tuple[list[dict[str, str]], list[FailedCamera]]:
    """base-render 回执按机位拆成"出齐五路的"与"没出来的"（纯函数，可直测）。

    - `renders[]` 里五路键齐的机位算成功（`sketch_key` 可空）；缺任一路按失败记、不半用；
    - 派发时点名了机位（`requested` 非 None）：点名了却不在 `renders[]` 里的按取景失败记，
      理由取回执 violations——带 `camera_id` 的只归那台，不带的归全部缺席机位（回执不区分时
      编排不替它区分）；
    - 没点名（全部机位）：缺席机位只能从 violations 的 `camera_id` 知道，没写就无从记。
    """
    raw_renders = receipt.get("renders")
    entries = (
        [item for item in raw_renders if isinstance(item, dict)]
        if isinstance(raw_renders, list)
        else []
    )
    violations = collect_violations(receipt)
    complete: list[dict[str, str]] = []
    failed: list[FailedCamera] = []
    seen: set[str] = set()
    for entry in entries:
        camera_id = _nonempty_str(entry, "camera_id")
        if camera_id is None or camera_id in seen:
            continue
        seen.add(camera_id)
        missing = [field for field in _RENDER_KEY_FIELDS if _nonempty_str(entry, field) is None]
        if missing:
            failed.append(
                FailedCamera(
                    camera_id=camera_id,
                    stage="base-render",
                    failed_checks=[f"missing-{field.replace('_', '-')}" for field in missing],
                )
            )
            continue
        keys = {field: entry[field] for field in _RENDER_KEY_FIELDS}
        keys["camera_id"] = camera_id
        sketch_key = _nonempty_str(entry, "sketch_key")
        if sketch_key is not None:
            keys["sketch_key"] = sketch_key
        view_kind = entry.get("view_kind") or entry.get("kind")
        if view_kind in ("bird", "room"):
            keys["view_kind"] = view_kind
        complete.append(keys)

    unattributed = [item for item in violations if not _nonempty_str(item, "camera_id")]
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for item in violations:
        camera_id = _nonempty_str(item, "camera_id")
        if camera_id is not None:
            by_camera.setdefault(camera_id, []).append(item)

    absent_ids = (
        [camera.camera_id for camera in requested if camera.camera_id not in seen]
        if requested is not None
        else [camera_id for camera_id in by_camera if camera_id not in seen]
    )
    for camera_id in absent_ids:
        own = by_camera.get(camera_id, []) + unattributed
        failed.append(
            FailedCamera(
                camera_id=camera_id,
                stage="base-render",
                failed_checks=violation_checks({"violations": own})
                if own
                else ["missing-from-renders"],
                violations=own,
            )
        )
    return complete, failed


async def run_space_render(spec: SpaceRenderSpec, run_step: StepRunner) -> SpaceRenderResult:
    """三维线主链。scene-compile 与 base-render 任一整体失败即整户失败；机位级失败分列。"""
    queues = spec.queues

    def failed(code: str, detail: str, **fields: Any) -> SpaceRenderResult:
        return SpaceRenderResult(
            task_id=spec.task_id,
            verdict="failed",
            failed_checks=[f"{code}:{detail}"],
            failure={"code": code, "detail": detail},
            **fields,
        )

    try:
        compiled = await run_step(
            ACTIVITY_SCENE_COMPILE,
            {"designPackageKey": spec.design_package_key, "revisionId": spec.revision_id},
            queues.render3d,
        )
    except DispatchFailure as err:
        return failed(ACTIVITY_SCENE_COMPILE, err.detail)
    if compiled.get("verdict") != "ok":
        return failed(ACTIVITY_SCENE_COMPILE, "; ".join(violation_checks(compiled)))
    scene_package_key = _nonempty_str(compiled, "scene_package_key")
    if scene_package_key is None:
        # verdict=ok 却没说场景包落在哪：包没落地却回报成功，按失败处理
        return failed(ACTIVITY_SCENE_COMPILE, "missing-scene-package-key")
    scene_evidence = evidence_of(compiled, "scene_package_key")

    requested = spec.cameras
    try:
        rendered = await run_step(
            ACTIVITY_BASE_RENDER,
            {
                "scenePackageKey": scene_package_key,
                "cameraIds": (
                    [camera.camera_id for camera in requested] if requested is not None else None
                ),
                "widthPx": spec.width_px,
                "heightPx": spec.height_px,
            },
            queues.render3d,
        )
    except DispatchFailure as err:
        return failed(
            ACTIVITY_BASE_RENDER,
            err.detail,
            scene_package_key=scene_package_key,
            scene_evidence=scene_evidence,
        )
    complete, failed_cameras = partition_base_render(rendered, requested)
    base_render_evidence = evidence_of(rendered, "renders")
    if not complete:
        # 一台都没出来才是整户失败：verdict=failed 或 ok 却零机位（空内容顶替）都算
        detail = (
            "; ".join(violation_checks(rendered))
            if rendered.get("verdict") != "ok"
            else "no-renders"
        )
        return failed(
            ACTIVITY_BASE_RENDER,
            detail,
            scene_package_key=scene_package_key,
            scene_evidence=scene_evidence,
            base_render_evidence=base_render_evidence,
            failed_cameras=failed_cameras,
        )

    requested_by_id = {camera.camera_id: camera for camera in requested or []}
    outcomes = await asyncio.gather(
        *[
            _realize_camera(spec, keys, requested_by_id.get(keys["camera_id"]), run_step)
            for keys in complete
        ]
    )
    renders = [outcome for outcome in outcomes if isinstance(outcome, CameraRender)]
    failed_cameras.extend(outcome for outcome in outcomes if isinstance(outcome, FailedCamera))

    if not renders:
        return failed(
            "space-render",
            "all-cameras-failed",
            scene_package_key=scene_package_key,
            scene_evidence=scene_evidence,
            base_render_evidence=base_render_evidence,
            failed_cameras=failed_cameras,
        )
    return SpaceRenderResult(
        task_id=spec.task_id,
        verdict="ok",
        scene_package_key=scene_package_key,
        scene_evidence=scene_evidence,
        base_render_evidence=base_render_evidence,
        renders=renders,
        failed_cameras=failed_cameras,
    )


async def _realize_camera(
    spec: SpaceRenderSpec,
    keys: dict[str, str],
    requested: SpaceCamera | None,
    run_step: StepRunner,
) -> CameraRender | FailedCamera:
    """一台机位的写实化：门禁判了不过就同样入参再派（上限 MAX_GATE_REDISPATCHES），
    其余失败不重派。"""
    camera_id = keys["camera_id"]
    render_keys = {field: value for field, value in keys.items() if field.endswith("_key")}
    view_kind = resolve_view_kind(keys, requested)
    if view_kind is None:
        return FailedCamera(
            camera_id=camera_id,
            stage="realism-pass",
            failed_checks=["missing-view-kind"],
            render_keys=render_keys,
        )
    seed = requested.seed if requested is not None else None
    request = realism_request(
        keys, view_kind=view_kind, style_template_id=spec.style_template_id, seed=seed
    )
    control_key = request.get("sketchKey") or request["lineKey"]

    redispatches = 0
    while True:
        try:
            receipt = await run_step(ACTIVITY_REALISM_PASS, request, spec.queues.imagegen)
        except DispatchFailure as err:
            return FailedCamera(
                camera_id=camera_id,
                stage="realism-pass",
                failed_checks=[err.detail],
                render_keys=render_keys,
                gate_redispatches=redispatches,
            )
        gate = receipt.get("gate")
        gate_dict = gate if isinstance(gate, dict) else None
        if receipt.get("verdict") == "ok":
            image_object_key = _nonempty_str(receipt, "image_object_key")
            if image_object_key is None:
                # 图没落地却回报成功，按失败处理
                return FailedCamera(
                    camera_id=camera_id,
                    stage="realism-pass",
                    failed_checks=["missing-image-object-key"],
                    render_keys=render_keys,
                    gate=gate_dict,
                    gate_redispatches=redispatches,
                )
            score = receipt.get("fidelity_score")
            return CameraRender(
                camera_id=camera_id,
                view_kind=view_kind,
                geometry_key=keys["geometry_key"],
                depth_key=keys["depth_key"],
                line_key=keys["line_key"],
                mask_key=keys["mask_key"],
                mask_index_key=keys["mask_index_key"],
                sketch_key=keys.get("sketch_key"),
                control_key=control_key,
                image_object_key=image_object_key,
                content_type=_nonempty_str(receipt, "content_type"),
                fidelity_score=float(score) if isinstance(score, int | float) else None,
                gate=gate_dict or {},
                evidence=evidence_of(
                    receipt, "image_object_key", "content_type", "fidelity_score", "gate"
                ),
                gate_redispatches=redispatches,
            )
        if is_gate_rejection(receipt) and redispatches < MAX_GATE_REDISPATCHES:
            redispatches += 1
            continue
        return FailedCamera(
            camera_id=camera_id,
            stage="realism-pass",
            failed_checks=violation_checks(receipt),
            violations=collect_violations(receipt),
            render_keys=render_keys,
            gate=gate_dict,
            gate_redispatches=redispatches,
        )


def space_render_spec_from_task(
    task_id: str, params: Mapping[str, Any], queues: TaskQueues
) -> SpaceRenderSpec:
    """交互侧任务参数快照 → 三维线编排输入（纯函数）。字段名与 `SpaceRenderSpec` 同（snake_case），
    多一个不认识的字段就拒收（pydantic ValidationError，属 ValueError）。"""
    return SpaceRenderSpec.model_validate({**params, "task_id": task_id, "queues": queues})
