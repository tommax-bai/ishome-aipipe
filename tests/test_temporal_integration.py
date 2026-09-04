"""集成冒烟：真连本地 Temporal（localhost:7233 / namespace genpipe）跑三个 workflow。

测试 Worker 把 mock activity 实现注册到唯一随机 task queue（spec.queues 全部收拢到
该队列），覆盖 GenBatchWorkflow 门禁通过 / 不通过自动重生成 / 重试超限失败三条路径，
GenerationTaskWorkflow 的 atmosphere-visual 路由链，以及 ReportComposeWorkflow 的
成文线主干、失败章单独重开（重开即过 / 重开耗尽）与三阶段失败（单元 / 装配 / 册检）。
服务器不可达时整体 skip（CI 无 Temporal
也能绿）；mock 只存在于测试内，生产 activity 存根保持 NotImplementedError。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from genpipe.models import (
    FloorplanVisualsSpec,
    FloorplanVisualTemplates,
    GenBatchSpec,
    GenerationTaskSpec,
    ReportComposeSpec,
    TaskQueues,
)
from genpipe.workflows import (
    FloorplanVisualsWorkflow,
    GenBatchWorkflow,
    GenerationTaskWorkflow,
    ReportComposeWorkflow,
)
from temporalio import activity
from temporalio.api.workflowservice.v1 import DescribeNamespaceRequest
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
TEMPORAL_NAMESPACE = os.environ.get("TEMPORAL_NAMESPACE", "genpipe")

MockImpl = Callable[[Any], dict[str, Any]]
CallLog = list[tuple[str, Any]]


async def _client_or_skip() -> Client:
    try:
        client = await asyncio.wait_for(
            Client.connect(
                TEMPORAL_ADDRESS,
                namespace=TEMPORAL_NAMESPACE,
                data_converter=pydantic_data_converter,
            ),
            timeout=5,
        )
    except Exception as err:  # noqa: BLE001 - 环境探测：不可达即跳过
        pytest.skip(f"Temporal 服务器不可达（{TEMPORAL_ADDRESS}）：{err}")
    try:
        await client.workflow_service.describe_namespace(
            DescribeNamespaceRequest(namespace=TEMPORAL_NAMESPACE)
        )
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"Temporal namespace {TEMPORAL_NAMESPACE} 未注册：{err}")
    return client


def _make_mock_activities(behaviors: dict[str, MockImpl], log: CallLog) -> list[Callable[..., Any]]:
    """按注册名批量构造 mock activity（注册名与 contracts 逐字一致，实现仅测试内存在）。"""

    def _build(name: str, impl: MockImpl) -> Callable[..., Any]:
        @activity.defn(name=name)
        async def mock(arg: Any) -> dict[str, Any]:
            log.append((name, arg))
            return impl(arg)

        return mock

    return [_build(name, impl) for name, impl in behaviors.items()]


def _prep_behaviors(log_counter: dict[str, int]) -> dict[str, MockImpl]:
    """母版备制 + 候选生成的标准 mock：产物 id 递增可追踪。"""

    def atmosphere(_: Any) -> dict[str, Any]:
        log_counter["candidate"] = log_counter.get("candidate", 0) + 1
        return {"artifact_id": f"cand-{log_counter['candidate']}"}

    return {
        "plan-layout-solve": lambda _: {"plan_revision_id": "rev-1"},
        "plan-rule-check": lambda _: {"passed": True},
        "plan-2d-render": lambda _: {"artifact_id": "master-1"},
        "atmosphere-visual": atmosphere,
    }


async def _run_gen_batch(
    client: Client, behaviors: dict[str, MockImpl], log: CallLog, spec: GenBatchSpec
) -> Any:
    task_queue = spec.queues.genpipe
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[GenBatchWorkflow, GenerationTaskWorkflow, ReportComposeWorkflow],
        activities=_make_mock_activities(behaviors, log),
    )
    async with worker:
        return await client.execute_workflow(
            GenBatchWorkflow.run,
            spec,
            id=f"it-batch-{spec.batch_id}",
            task_queue=task_queue,
        )


def _batch_spec(queue: str, **overrides: Any) -> GenBatchSpec:
    queues = TaskQueues(genpipe=queue, render2d=queue, imagegen=queue, render3d=queue)
    defaults: dict[str, Any] = {
        "batch_id": uuid.uuid4().hex,
        "floorplan_id": "fp-1",
        "template_ids": ["tpl-a", "tpl-b", "tpl-c"],
        "candidate_count": 2,
        "queues": queues,
    }
    defaults.update(overrides)
    return GenBatchSpec(**defaults)


async def test_gen_batch_gate_pass_publishes_all_candidates() -> None:
    """门禁通过路径：母版备制 → 双候选 fan-out → 两检全过 → verdict passed。"""
    client = await _client_or_skip()
    log: CallLog = []
    counter: dict[str, int] = {}
    behaviors = _prep_behaviors(counter) | {
        "consistency-check": lambda _: {"passed": True},
        "compliance-check": lambda _: {"passed": True},
    }
    queue = f"it-{uuid.uuid4().hex}"
    result = await _run_gen_batch(client, behaviors, log, _batch_spec(queue))
    assert result.verdict == "passed"
    assert result.rounds == 1
    assert sorted(result.passed_artifact_ids) == ["cand-1", "cand-2"]
    assert log[0][0] == "plan-layout-solve"
    assert [name for name, _ in log].count("atmosphere-visual") == 2


async def test_gen_batch_gate_fail_regenerates_then_passes() -> None:
    """门禁不通过路径：首轮 consistency 拒绝 → 自动重生成（换模板）→ 次轮通过。"""
    client = await _client_or_skip()
    log: CallLog = []
    counter: dict[str, int] = {}

    def flaky_consistency(artifact_id: Any) -> dict[str, Any]:
        # 首个候选拒绝，之后放行：驱动一轮自动重生成
        return {"passed": artifact_id != "cand-1"}

    behaviors = _prep_behaviors(counter) | {
        "consistency-check": flaky_consistency,
        "compliance-check": lambda _: {"passed": True},
    }
    queue = f"it-{uuid.uuid4().hex}"
    spec = _batch_spec(queue, candidate_count=1, max_regen_rounds=2)
    result = await _run_gen_batch(client, behaviors, log, spec)
    assert result.verdict == "passed"
    assert result.rounds == 2
    assert result.passed_artifact_ids == ["cand-2"]
    # 重生成轮用了下一个模板（轮换纯函数在真实链路中生效）
    atmosphere_templates = [arg["template_id"] for name, arg in log if name == "atmosphere-visual"]
    assert atmosphere_templates == ["tpl-a", "tpl-b"]


async def test_gen_batch_gate_exhausted_returns_failed_verdict() -> None:
    """重试超限路径：compliance 持续拒绝 → 超限返回 failed verdict，绝不静默假成功。"""
    client = await _client_or_skip()
    log: CallLog = []
    counter: dict[str, int] = {}
    behaviors = _prep_behaviors(counter) | {
        "consistency-check": lambda _: {"passed": True},
        "compliance-check": lambda _: {"passed": False},
    }
    queue = f"it-{uuid.uuid4().hex}"
    spec = _batch_spec(queue, candidate_count=1, max_regen_rounds=1)
    result = await _run_gen_batch(client, behaviors, log, spec)
    assert result.verdict == "failed"
    assert result.rounds == 2
    assert result.passed_artifact_ids == []
    assert any("compliance-check" in failure for failure in result.failed_checks)


async def test_generation_task_atmosphere_chain_routes_and_gates() -> None:
    """GenerationTaskWorkflow 路由链：atmosphere-visual → realism-pass → 门禁收尾。"""
    client = await _client_or_skip()
    log: CallLog = []
    behaviors: dict[str, MockImpl] = {
        "atmosphere-visual": lambda _: {"artifact_id": "atm-1"},
        "realism-pass": lambda arg: {
            "artifact_id": "real-1",
            "base": arg["base_render_artifact_id"],
        },
        "consistency-check": lambda _: {"passed": True},
        "compliance-check": lambda _: {"passed": True},
    }
    queue = f"it-{uuid.uuid4().hex}"
    queues = TaskQueues(genpipe=queue, render2d=queue, imagegen=queue, render3d=queue)
    spec = GenerationTaskSpec(
        task_id=uuid.uuid4().hex,
        task_type="atmosphere-visual",
        params={"plan_master_artifact_id": "master-1", "template_id": "tpl-a", "style_ref": "s-1"},
        queues=queues,
    )
    worker = Worker(
        client,
        task_queue=queue,
        workflows=[GenBatchWorkflow, GenerationTaskWorkflow, ReportComposeWorkflow],
        activities=_make_mock_activities(behaviors, log),
    )
    async with worker:
        result = await client.execute_workflow(
            GenerationTaskWorkflow.run,
            spec,
            id=f"it-task-{spec.task_id}",
            task_queue=queue,
        )
    assert result.verdict == "passed"
    assert result.artifact_ids == ["atm-1", "real-1"]
    # 链路顺序 + 上游产物注入 + 门禁收尾在最终产物上执行
    assert [name for name, _ in log] == [
        "atmosphere-visual",
        "realism-pass",
        "consistency-check",
        "compliance-check",
    ]
    realism_arg = log[1][1]
    assert realism_arg["base_render_artifact_id"] == "atm-1"
    assert log[2][1] == "real-1"


# ---------------------------------------------------------------------------
# 报告成文线（ReportComposeWorkflow）
# ---------------------------------------------------------------------------

# 报告数据包对编排是不透明载荷：夹具只需"是个包"，字段形状由 contracts schema 与
# reportgen 侧负责——本仓不建模，故这里也不构造真包（camelCase 提示其生产方是 Java 求值线）
OPAQUE_PACKAGE: dict[str, Any] = {
    "domains": ["dom-lighting", "dom-budget"],
    "anchors": [{"lkpId": "lkp-desk-height", "value": {"mm": 720}}],
}


def _report_spec(queue: str, domains: list[str], **overrides: Any) -> ReportComposeSpec:
    queues = TaskQueues(
        genpipe=queue,
        render2d=queue,
        imagegen=queue,
        render3d=queue,
        reportgen=queue,
        reportrender=queue,
    )
    defaults: dict[str, Any] = {
        "report_id": uuid.uuid4().hex,
        "domains": domains,
        "package": OPAQUE_PACKAGE,
        "queues": queues,
    }
    defaults.update(overrides)
    return ReportComposeSpec(**defaults)


async def _run_report_compose(
    client: Client, behaviors: dict[str, MockImpl], log: CallLog, spec: ReportComposeSpec
) -> Any:
    task_queue = spec.queues.reportgen
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[GenBatchWorkflow, GenerationTaskWorkflow, ReportComposeWorkflow],
        activities=_make_mock_activities(behaviors, log),
    )
    async with worker:
        return await client.execute_workflow(
            ReportComposeWorkflow.run,
            spec,
            id=f"it-report-{spec.report_id}",
            task_queue=task_queue,
        )


def _compose_unit(arg: Any) -> dict[str, Any]:
    """单元成文 mock：按域回卡片；budget 域回一轮重写，供观测量断言。"""
    domain = arg["domain"]
    return {
        "verdict": "ok",
        "domain": domain,
        "cards": [
            {"thesis": f"{domain}-thesis", "body": "见 {lkp-desk-height}", "number_refs": []}
        ],
        "violations": [],
        "rewrites_used": 1 if domain == "dom-budget" else 0,
        "releases": [{"domain": domain, "release_tag": f"{domain}@v1"}],
    }


def _assemble_pages(arg: Any) -> dict[str, Any]:
    """页面装配 mock：确定性按域成页（与 reportgen 首版口径一致）。"""
    units = sorted(arg["units"], key=lambda u: u["domain"])
    return {
        "verdict": "ok",
        "pages": [
            {"page_id": f"page-{u['domain']}", "domain": u["domain"], "cards": u["cards"]}
            for u in units
        ],
        "violations": [],
    }


def _report_behaviors(**overrides: MockImpl) -> dict[str, MockImpl]:
    behaviors: dict[str, MockImpl] = {
        "report-unit-compose": _compose_unit,
        "report-page-assemble": _assemble_pages,
        "report-book-check": lambda _: {"verdict": "ok", "violations": []},
        "report-book-render": lambda arg: {
            "verdict": "ok",
            "book_key": f"reports/{arg['report_id']}/book.html",
            "violations": [],
        },
    }
    behaviors.update(overrides)
    return behaviors


async def test_report_compose_fans_out_units_then_assembles_checks_and_renders() -> None:
    """主干：两域并行成文 → 装配 → 册检 → 出册全过；报告数据包沿途原样透传（不透明载荷纪律）。"""
    client = await _client_or_skip()
    log: CallLog = []
    queue = f"it-{uuid.uuid4().hex}"
    spec = _report_spec(queue, ["dom-lighting", "dom-budget"])
    result = await _run_report_compose(client, _report_behaviors(), log, spec)

    assert result.verdict == "ok"
    assert result.failed_stage is None
    assert [page["page_id"] for page in result.pages] == ["page-dom-budget", "page-dom-lighting"]
    assert result.rewrite_rounds_by_domain == {"dom-lighting": 0, "dom-budget": 1}

    dispatched = [name for name, _ in log]
    # 单元并行（完成次序不定，只断言集合与计数），装配/册检严格在其后依次发生
    assert dispatched.count("report-unit-compose") == 2
    assert dispatched[-3:] == ["report-page-assemble", "report-book-check", "report-book-render"]
    # 册落地了才算成功：键随结论回来，业务侧据此签一条能打开的链接
    assert result.book_key == f"reports/{spec.report_id}/book.html"
    unit_args = [arg for name, arg in log if name == "report-unit-compose"]
    assert sorted(arg["domain"] for arg in unit_args) == ["dom-budget", "dom-lighting"]
    assert all(arg["package"] == OPAQUE_PACKAGE for arg in unit_args)
    assert all(arg["max_rewrites"] == 2 for arg in unit_args)
    assert log[-2][1]["package"] == OPAQUE_PACKAGE  # 册检同样拿到原包
    assert log[-1][1]["package"] == OPAQUE_PACKAGE  # 出册也拿到原包（渲染要靠它解数字引用）


def _failed_unit(domain: str, rewrites_used: int = 2) -> dict[str, Any]:
    """单元成文 mock 的失败形态：章内重写轮数用满仍不过检，正常返回 verdict=failed。

    注意它**不抛异常**——`_ACTIVITY_RETRY` 只对异常生效，这条路径正是"整章重开"要补的缺口。
    """
    return {
        "verdict": "failed",
        "domain": domain,
        "cards": [],
        "violations": [{"check": "cr-budget-stale-price", "detail": "单价过期"}],
        "rewrites_used": rewrites_used,
        "releases": [],
    }


async def test_report_compose_unit_failure_stops_before_assemble() -> None:
    """某域 failed：整册失败，不装配也不出"其余页"——失败单元与其 violations 如实上抛。

    这条只验失败策略本身，故把重开关掉（`max_unit_retries=0`）；重开路径另有两条用例。
    """
    client = await _client_or_skip()
    log: CallLog = []

    def flaky_unit(arg: Any) -> dict[str, Any]:
        if arg["domain"] != "dom-budget":
            return _compose_unit(arg)
        return _failed_unit("dom-budget")

    queue = f"it-{uuid.uuid4().hex}"
    spec = _report_spec(queue, ["dom-lighting", "dom-budget"], max_unit_retries=0)
    result = await _run_report_compose(
        client, _report_behaviors(**{"report-unit-compose": flaky_unit}), log, spec
    )

    assert result.verdict == "failed"
    assert result.failed_stage == "unit-compose"
    assert result.failed_domains == ["dom-budget"]
    assert result.failed_units[0]["violations"] == [
        {"check": "cr-budget-stale-price", "detail": "单价过期"}
    ]
    assert result.pages == []
    assert result.rewrite_rounds_by_domain == {"dom-lighting": 0, "dom-budget": 2}
    assert result.unit_retries_by_domain == {}  # 关掉重开就一次都不重开
    # 装配/册检一步都不派：下游本就以 gate-unit-failed 拒收，缺域的册在册检必然不合格
    assert [name for name, _ in log] == ["report-unit-compose", "report-unit-compose"]


async def test_report_compose_retries_only_the_failed_domain_and_ships_the_book() -> None:
    """失败的那一章单独重开：只重派失败域，已成的章不重跑；重开后成册，重开次数记在结论里。

    用户裁决 2026-08-31：「失败让重开，我觉得没问题，单独开失败的这一张就可以了。」
    """
    client = await _client_or_skip()
    log: CallLog = []
    attempts: dict[str, int] = {}

    def flaky_once(arg: Any) -> dict[str, Any]:
        """dom-budget 首次失败、重开即过——章失败是随机的不是必然的（实测形态）。"""
        domain = arg["domain"]
        attempts[domain] = attempts.get(domain, 0) + 1
        if domain == "dom-budget" and attempts[domain] == 1:
            return _failed_unit(domain)
        return _compose_unit(arg)

    queue = f"it-{uuid.uuid4().hex}"
    spec = _report_spec(queue, ["dom-lighting", "dom-budget"])
    result = await _run_report_compose(
        client, _report_behaviors(**{"report-unit-compose": flaky_once}), log, spec
    )

    assert result.verdict == "ok"
    assert result.failed_domains == []
    assert result.unit_retries_by_domain == {"dom-budget": 1}
    assert [page["page_id"] for page in result.pages] == ["page-dom-budget", "page-dom-lighting"]
    assert result.book_key == f"reports/{spec.report_id}/book.html"
    # 只重开失败那一章：lighting 派 1 次、budget 派 2 次（整册重来会是 2/2）
    assert attempts == {"dom-lighting": 1, "dom-budget": 2}
    unit_args = [arg for name, arg in log if name == "report-unit-compose"]
    assert len(unit_args) == 3
    # 重开用同样入参（不改 max_rewrites：那是章内旋钮的设计定数，重开是另一个旋钮）
    assert all(arg["max_rewrites"] == 2 for arg in unit_args)
    assert all(arg["package"] == OPAQUE_PACKAGE for arg in unit_args)
    # 章内重写轮数取最后一次尝试的（budget 首轮 failed 记 2，重开成的那次记 1）——
    # 它是章内旋钮的观测量，与重开次数各记各的，不混算成一个数
    assert result.rewrite_rounds_by_domain == {"dom-lighting": 0, "dom-budget": 1}


async def test_report_compose_exhausts_unit_retries_then_fails_the_book() -> None:
    """重开次数耗尽仍失败：整册失败，失败面是**最后一次**尝试的，重开次数照记。"""
    client = await _client_or_skip()
    log: CallLog = []
    attempts: dict[str, int] = {}

    def always_failing_budget(arg: Any) -> dict[str, Any]:
        domain = arg["domain"]
        attempts[domain] = attempts.get(domain, 0) + 1
        return _failed_unit(domain) if domain == "dom-budget" else _compose_unit(arg)

    queue = f"it-{uuid.uuid4().hex}"
    spec = _report_spec(queue, ["dom-lighting", "dom-budget"], max_unit_retries=2)
    result = await _run_report_compose(
        client, _report_behaviors(**{"report-unit-compose": always_failing_budget}), log, spec
    )

    assert result.verdict == "failed"
    assert result.failed_stage == "unit-compose"
    assert result.failed_domains == ["dom-budget"]
    assert result.failed_units[0]["domain"] == "dom-budget"
    assert result.pages == []
    assert result.unit_retries_by_domain == {"dom-budget": 2}
    # 首轮 + 两次重开 = budget 派 3 次；lighting 首轮就成了，一次都不重派
    assert attempts == {"dom-lighting": 1, "dom-budget": 3}
    # 重开耗尽才判整册失败，装配/册检/出册仍一步都不派
    assert [name for name, _ in log] == ["report-unit-compose"] * 4


async def test_report_compose_page_assemble_failure_surfaces_violations() -> None:
    """装配 failed：违规原样上抛，册检不再派，pages 不回传。"""
    client = await _client_or_skip()
    log: CallLog = []
    assemble_violations = [{"check": "cr-one-thesis-per-page", "detail": "dom-budget 页双论点"}]
    behaviors = _report_behaviors(
        **{
            "report-page-assemble": lambda _: {
                "verdict": "failed",
                "pages": [],
                "violations": assemble_violations,
            }
        }
    )
    queue = f"it-{uuid.uuid4().hex}"
    spec = _report_spec(queue, ["dom-lighting", "dom-budget"])
    result = await _run_report_compose(client, behaviors, log, spec)

    assert result.verdict == "failed"
    assert result.failed_stage == "page-assemble"
    assert result.violations == assemble_violations
    assert result.failed_checks == []
    assert result.pages == []
    assert [name for name, _ in log].count("report-book-check") == 0


async def test_report_compose_book_check_failure_withholds_pages() -> None:
    """册检 failed：装配成功也不回内容——失败册不给 pages，杜绝"拿到就发布"的误用路径。"""
    client = await _client_or_skip()
    log: CallLog = []
    book_violations = [{"check": "gate-domain-page-missing", "detail": "dom-storage 无页"}]
    behaviors = _report_behaviors(
        **{"report-book-check": lambda _: {"verdict": "failed", "violations": book_violations}}
    )
    queue = f"it-{uuid.uuid4().hex}"
    spec = _report_spec(queue, ["dom-lighting", "dom-budget"])
    result = await _run_report_compose(client, behaviors, log, spec)

    assert result.verdict == "failed"
    assert result.failed_stage == "book-check"
    assert result.violations == book_violations
    assert result.pages == []
    assert [name for name, _ in log][-1] == "report-book-check"


async def test_report_compose_rejects_duplicate_domains_before_dispatch() -> None:
    """同域派两次 → 装配出重复页 → 册检必挂：派发前就拦，省掉整轮 LLM 推理。"""
    client = await _client_or_skip()
    log: CallLog = []
    queue = f"it-{uuid.uuid4().hex}"
    spec = _report_spec(queue, ["dom-lighting", "dom-lighting"])
    result = await _run_report_compose(client, _report_behaviors(), log, spec)

    assert result.verdict == "failed"
    assert result.failed_stage == "unit-compose"
    assert result.failed_checks == ["duplicate-domains"]
    assert log == []


# ---------------------------------------------------------------------------
# 三张免费图生成线（FloorplanVisualsWorkflow）：主链 / 旁路不致命 / 主链失败也回流
# ---------------------------------------------------------------------------

_VISUALS_KEY = "uploads/" + "e" * 64 + "/original.png"
_VISUALS_PREFIX = "uploads/" + "e" * 64


def _visuals_behaviors(log: CallLog) -> dict[str, MockImpl]:
    """全链 mock：出参形态照各仓真 activity 的回执（键名逐字），只把内容换成假的。"""
    return {
        "floorplan-geometry-extract": lambda _: {
            "verdict": "ok",
            "geometry_key": f"{_VISUALS_PREFIX}/floorplan-geometry.json",
            "geometry": {"planBox": [0, 0, 1, 1], "walls": [], "openings": [], "rooms": []},
            "facts": [{"factId": "plan-share-主卧", "subject": "主卧", "statement": "主卧占 30%"}],
            "room_names": ["主卧", "客厅"],
        },
        "plan-notes-write": lambda _: {
            "verdict": "ok",
            "notes": [
                {"room": "主卧", "text": "早上先亮起来的是这间", "cites": ["plan-share-主卧"]},
                {"room": "客厅", "text": "客厅占了 30% 的面积", "cites": ["plan-share-主卧"]},
            ],
            "rejected": [],
        },
        "plan-copy-write": lambda _: {
            "verdict": "ok",
            "copy": {"title": "光照进来的家", "summary": "一句话", "tips": ["a", "b", "c"]},
        },
        "plan-2d-render": lambda arg: {
            "verdict": "ok",
            "master_key": f"{_VISUALS_PREFIX}/plan-master.png",
            "room_anchors_key": f"{_VISUALS_PREFIX}/plan-rooms.json",
            "brief_key": f"{_VISUALS_PREFIX}/plan-brief.png" if arg.get("notes") else None,
        },
        "atmosphere-visual": lambda arg: {
            "verdict": "ok",
            "image_object_key": f"{_VISUALS_PREFIX}/atmosphere-{arg['template_id']}.jpg",
            "content_type": "image/jpeg",
        },
        "style-caption-overlay": lambda arg: {
            "verdict": "ok",
            "image_object_key": arg["style_object_key"].rsplit(".", 1)[0] + "-captioned.png",
        },
        "floorplan-parse": lambda _: {
            "verdict": "ok",
            "reading_key": f"{_VISUALS_PREFIX}/floorplan-reading.json",
            "layout_features": {"balcony_utility": "阳台画了洗衣机"},
        },
        "task-result-deliver": lambda _: {"verdict": "ok", "status_code": 200, "receipt": {}},
    }


async def _run_visuals(
    client: Client,
    behaviors: dict[str, MockImpl],
    log: CallLog,
    queue: str,
    *,
    annotate_style: bool = False,
) -> Any:
    spec = FloorplanVisualsSpec(
        task_id=uuid.uuid4().hex,
        floorplan_object_key=_VISUALS_KEY,
        result_callback_url="http://127.0.0.1:1/api/v1/generation-tasks/x/result",
        templates=FloorplanVisualTemplates(annotate_style=annotate_style),
        queues=TaskQueues(genpipe=queue, render2d=queue, imagegen=queue, render3d=queue),
    )
    worker = Worker(
        client,
        task_queue=queue,
        workflows=[FloorplanVisualsWorkflow],
        activities=_make_mock_activities(behaviors, log),
    )
    async with worker:
        return await client.execute_workflow(
            FloorplanVisualsWorkflow.run, spec, id=f"it-visuals-{spec.task_id}", task_queue=queue
        )


async def test_floorplan_visuals_happy_path_yields_six_products_and_delivers() -> None:
    client = await _client_or_skip()
    log: CallLog = []
    queue = f"it-{uuid.uuid4().hex}"
    result = await _run_visuals(client, _visuals_behaviors(log), log, queue)

    assert result.verdict == "ok", result
    assert result.delivered is True
    assert {p.product for p in result.products} == {
        "floorplan_geometry",
        "plan_master",
        "brief_image",
        "style_image",
        "mood_image",
        "floorplan_reading",
    }
    by_product = {p.product: p for p in result.products}
    assert by_product["mood_image"].object_key.endswith("atmosphere-cream-journal-captioned.png")
    assert by_product["style_image"].object_key.endswith(
        "atmosphere-lifestyle-notebook-handwritten.jpg"
    )
    # 两张风格图：默认（annotate_style=False）两张都不带注释字段——物件清单生成步落地前不递注释
    atmosphere_calls = [arg for name, arg in log if name == "atmosphere-visual"]
    assert {c["template_id"] for c in atmosphere_calls} == {
        "cream-journal",
        "lifestyle-notebook-handwritten",
    }
    assert all("annotations" not in c for c in atmosphere_calls)
    # 母版吃内联几何 + 全部批注（含带数字的那条——说明图画得了数字）
    render_call = next(arg for name, arg in log if name == "plan-2d-render")
    assert render_call["floorplan_object_key"] == _VISUALS_KEY and len(render_call["notes"]) == 2
    # 回调报文按契约：completed + 六件产物
    deliver_call = next(arg for name, arg in log if name == "task-result-deliver")
    assert deliver_call["result_callback_url"].endswith("/result")
    assert deliver_call["result"]["status"] == "completed"
    assert len(deliver_call["result"]["products"]) == 6
    assert "failure" not in deliver_call["result"]


async def test_floorplan_visuals_parse_failure_is_not_fatal() -> None:
    client = await _client_or_skip()
    log: CallLog = []
    behaviors = _visuals_behaviors(log) | {
        "floorplan-parse": lambda _: {
            "verdict": "failed",
            "violations": [{"check": "floorplan-parse-failed", "detail": "网关没这个逻辑名"}],
        }
    }
    queue = f"it-{uuid.uuid4().hex}"
    result = await _run_visuals(client, behaviors, log, queue)

    assert result.verdict == "ok"
    assert "floorplan_reading" not in {p.product for p in result.products}
    assert any(check.startswith("floorplan-parse:") for check in result.failed_checks)
    deliver_call = next(arg for name, arg in log if name == "task-result-deliver")
    assert deliver_call["result"]["status"] == "completed"


async def test_floorplan_visuals_main_chain_failure_still_delivers_failed_result() -> None:
    client = await _client_or_skip()
    log: CallLog = []
    behaviors = _visuals_behaviors(log) | {
        "plan-2d-render": lambda _: {
            "verdict": "failed",
            "violations": [{"check": "plan-master-failed", "detail": "外圈闭合率 64%"}],
        }
    }
    queue = f"it-{uuid.uuid4().hex}"
    result = await _run_visuals(client, behaviors, log, queue)

    assert result.verdict == "failed"
    assert result.failure == {
        "code": "plan-2d-render",
        "detail": "plan-master-failed=外圈闭合率 64%",
    }
    assert result.delivered is True
    # 半成品血缘：几何有了，图一张都没有
    assert {p.product for p in result.products} == {"floorplan_geometry", "floorplan_reading"}
    assert not any(name == "atmosphere-visual" for name, _ in log)
    deliver_call = next(arg for name, arg in log if name == "task-result-deliver")
    assert deliver_call["result"]["status"] == "failed"
    assert deliver_call["result"]["failure"]["code"] == "plan-2d-render"


async def test_floorplan_visuals_annotates_style_only_when_switched_on() -> None:
    """annotate_style=True：手账写字版只带不含数字的那条注释；情绪图底图仍不带。"""
    client = await _client_or_skip()
    log: CallLog = []
    queue = f"it-{uuid.uuid4().hex}"
    result = await _run_visuals(client, _visuals_behaviors(log), log, queue, annotate_style=True)

    assert result.verdict == "ok"
    atmosphere_calls = [arg for name, arg in log if name == "atmosphere-visual"]
    style_call = next(c for c in atmosphere_calls if c["template_id"].endswith("handwritten"))
    assert style_call["annotations"] == [{"room": "主卧", "text": "早上先亮起来的是这间"}]
    assert "annotations" not in next(
        c for c in atmosphere_calls if c["template_id"] == "cream-journal"
    )


async def test_floorplan_visuals_regenerates_mood_when_caption_does_not_fit() -> None:
    """叠字放不下 → 重生成一张情绪图底图再叠；第二张放下了即成功，产物记着重生成了一次。"""
    client = await _client_or_skip()
    log: CallLog = []
    caption_calls = {"n": 0}

    def flaky_caption(arg: Any) -> dict[str, Any]:
        caption_calls["n"] += 1
        if caption_calls["n"] == 1:
            return {
                "verdict": "failed",
                "violations": [
                    {
                        "check": "style-caption-failed",
                        "detail": "版面上没有一段连续空白放得下这些字",
                    }
                ],
            }
        return {"verdict": "ok", "image_object_key": arg["style_object_key"] + "-captioned.png"}

    behaviors = _visuals_behaviors(log) | {"style-caption-overlay": flaky_caption}
    queue = f"it-{uuid.uuid4().hex}"
    result = await _run_visuals(client, behaviors, log, queue)

    assert result.verdict == "ok", result
    mood = next(p for p in result.products if p.product == "mood_image")
    assert mood.gen_params["caption_regenerations"] == 1
    assert [name for name, _ in log].count("atmosphere-visual") == 3
    assert caption_calls["n"] == 2


async def test_floorplan_visuals_gives_up_captioning_after_retries() -> None:
    client = await _client_or_skip()
    log: CallLog = []
    behaviors = _visuals_behaviors(log) | {
        "style-caption-overlay": lambda _: {
            "verdict": "failed",
            "violations": [{"check": "style-caption-failed", "detail": "放不下"}],
        }
    }
    queue = f"it-{uuid.uuid4().hex}"
    result = await _run_visuals(client, behaviors, log, queue)

    assert result.verdict == "failed"
    assert result.failure is not None and result.failure["code"] == "style-caption-overlay"
    # 默认两次重生成：情绪图底图共出了 1 + 2 张，风格图 1 张
    assert [name for name, _ in log].count("atmosphere-visual") == 4
