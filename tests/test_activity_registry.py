"""activity 注册名与 contracts 注册表逐字一致的守门测试。

唯一真源：ishome-contracts `activities/registry.md`（注册名，只增不改）与
`registries/task_queues.md`（队列归属，V1.4 绘图拆分口径）。本清单为其副本；
两处不一致时以 contracts 仓为准并回改此处。
"""

from __future__ import annotations

from genpipe.models import TaskQueues
from genpipe.workflows import (
    ACTIVITY_REPORT_BOOK_CHECK,
    ACTIVITY_REPORT_BOOK_RENDER,
    ACTIVITY_REPORT_PAGE_ASSEMBLE,
    ACTIVITY_REPORT_UNIT_COMPOSE,
)
from genpipe_worker.activities import FloorplanActivities, activity_registry
from temporalio import activity

# 守门测试只看注册名与函数名，依赖给三个哑对象即可（activity 不在这里被调用）
ACTIVITY_REGISTRY = activity_registry(
    FloorplanActivities(store=object(), llm=object(), http=object())  # type: ignore[arg-type]
)

# 注册名 → 函数名（kebab-case ↔ snake_case 动词前置，规范 §2.4）——contracts 全量 14 项
CONTRACTS_ACTIVITY_REGISTRY: dict[str, str] = {
    "floorplan-parse": "parse_floorplan",
    "plan-layout-solve": "solve_plan_layout",
    "plan-rule-check": "check_plan_rules",
    "plan-2d-render": "render_plan_2d",
    "atmosphere-visual": "generate_atmosphere_visual",
    "scene-compile": "compile_scene",
    "base-render": "render_base",
    "realism-pass": "apply_realism_pass",
    "consistency-check": "check_consistency",
    "compliance-check": "check_compliance",
    # #11-13 报告成文线（contracts 2026-08-28 入册）：实现在 ishome-reportgen，本仓只按名派发
    "report-unit-compose": "compose_report_unit",
    "report-page-assemble": "assemble_report_pages",
    "report-book-check": "check_report_book",
    # #14 报告渲染层（contracts 2026-08-30 晚入册）：实现在 ishome-reportrender，本仓只按名派发
    "report-book-render": "render_report_book",
    # #15-#19 串联（contracts 2026-09-04 入册）：解析一族接进 activity + 情绪图叠字 + 结果回流
    "floorplan-geometry-extract": "extract_floorplan_geometry",
    "plan-notes-write": "write_plan_notes",
    "plan-copy-write": "write_plan_copy",
    "style-caption-overlay": "overlay_style_caption",
    "task-result-deliver": "deliver_task_result",
}

# task queue → 承接 activity（contracts registries/task_queues.md，逐字一致）
CONTRACTS_TASK_QUEUE_OWNERSHIP: dict[str, set[str]] = {
    "genpipe-activities": {
        "floorplan-parse",
        "plan-layout-solve",
        "plan-rule-check",
        "consistency-check",
        "compliance-check",
        "floorplan-geometry-extract",
        "plan-notes-write",
        "plan-copy-write",
        "task-result-deliver",
    },
    "render2d-activities": {"plan-2d-render", "style-caption-overlay"},
    "imagegen-activities": {"atmosphere-visual", "realism-pass"},
    "render3d-activities": {"scene-compile", "base-render"},
    "reportgen-activities": {
        "report-unit-compose",
        "report-page-assemble",
        "report-book-check",
    },
    "reportrender-activities": {"report-book-render"},
}


def test_contracts_registry_total_unchanged() -> None:
    """注册表只增不改：V1.4 绘图拆分只搬家不改名（10 项），成文线 3 项、渲染层 1 项即 14；
    2026-09-04 串联再加 5 项即 19。"""
    assert len(CONTRACTS_ACTIVITY_REGISTRY) == 19


def test_queue_ownership_partitions_contracts_registry() -> None:
    """六条队列的承接集合两两不交，且并集恰为 contracts 全量注册表。"""
    union: set[str] = set()
    for owned in CONTRACTS_TASK_QUEUE_OWNERSHIP.values():
        assert not (union & owned), "同一 activity 不得归属多条队列"
        union |= owned
    assert union == set(CONTRACTS_ACTIVITY_REGISTRY)


def test_worker_registry_matches_genpipe_queue_ownership() -> None:
    """genpipe-worker 只承接 genpipe-activities 队列的 9 个非绘图 activity
    （5 个骨架 + 4 个串联）。"""
    assert set(ACTIVITY_REGISTRY) == CONTRACTS_TASK_QUEUE_OWNERSHIP["genpipe-activities"]
    assert len(ACTIVITY_REGISTRY) == 9


def test_report_dispatch_names_and_queue_match_contracts() -> None:
    """成文线 activity 实现在 ishome-reportgen，本仓只按名派发——注册名与队列的唯一校验点。"""
    dispatched = {
        ACTIVITY_REPORT_UNIT_COMPOSE,
        ACTIVITY_REPORT_PAGE_ASSEMBLE,
        ACTIVITY_REPORT_BOOK_CHECK,
    }
    assert dispatched == CONTRACTS_TASK_QUEUE_OWNERSHIP["reportgen-activities"]
    assert TaskQueues().reportgen == "reportgen-activities"
    # 本仓 worker 不得承接成文线 activity（物理隔离：LLM 推理伸缩轴独立部署）
    assert not (set(ACTIVITY_REGISTRY) & dispatched)


def test_book_render_dispatch_name_and_queue_match_contracts() -> None:
    """出册 activity 实现在 ishome-reportrender，本仓只按名派发。

    它与成文线**分队列**：渲染是 CPU + IO，成文是 LLM 推理，伸缩轴不同——
    同一条"逻辑异质→物理隔离"的判据（这是它的第四次应用）。
    """
    assert {ACTIVITY_REPORT_BOOK_RENDER} == CONTRACTS_TASK_QUEUE_OWNERSHIP[
        "reportrender-activities"
    ]
    assert TaskQueues().reportrender == "reportrender-activities"
    assert ACTIVITY_REPORT_BOOK_RENDER not in set(ACTIVITY_REGISTRY)


def test_registered_temporal_names_match_keys() -> None:
    """字典键、@activity.defn(name=...) 注册名、函数名三者一致。"""
    for registry_name, fn in ACTIVITY_REGISTRY.items():
        defn = activity._Definition.from_callable(fn)  # noqa: SLF001
        assert defn is not None, f"{registry_name} is not a temporal activity"
        assert defn.name == registry_name
        assert fn.__name__ == CONTRACTS_ACTIVITY_REGISTRY[registry_name]
