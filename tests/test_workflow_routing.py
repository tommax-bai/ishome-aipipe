"""两条线的编排纯函数直测（不依赖 Temporal 运行时）：

生成管线——GenerationTask 路由链与机检门禁判定；
报告成文线——各 dom- 单元并行成文的结果归并、失败章重开的跨轮归并与违规透传。
"""

from __future__ import annotations

from typing import Any

import pytest
from genpipe.models import GenerationTaskSpec, TaskQueues, TaskStep
from genpipe.workflows import (
    PipelineDataError,
    build_task_chain,
    collect_violations,
    describe_failure,
    evaluate_gate,
    merge_retried_units,
    partition_unit_outcomes,
    pick_template,
    resolve_step_arg,
    unexplained_failure_checks,
)


def test_plan_2d_render_chain_routes_to_render2d_queue() -> None:
    spec = GenerationTaskSpec(
        task_id="t-1", task_type="plan-2d-render", params={"revision_id": "rev-1"}
    )
    chain = build_task_chain(spec)
    assert [(s.activity, s.task_queue) for s in chain] == [
        ("plan-2d-render", "render2d-activities")
    ]
    assert chain[0].arg == {"revision_id": "rev-1", "purpose": "plan_master"}
    assert chain[0].long_running


def test_atmosphere_visual_chain_routes_to_imagegen_and_pipes_artifact() -> None:
    spec = GenerationTaskSpec(
        task_id="t-2",
        task_type="atmosphere-visual",
        params={"plan_master_artifact_id": "master-1", "template_id": "tpl-a", "style_ref": "s-1"},
    )
    chain = build_task_chain(spec)
    assert [(s.activity, s.task_queue) for s in chain] == [
        ("atmosphere-visual", "imagegen-activities"),
        ("realism-pass", "imagegen-activities"),
    ]
    assert chain[1].arg_from_upstream == {"base_render_artifact_id": "artifact_id"}


def test_scene_compile_chain_routes_to_render3d_and_pipes_scene_package() -> None:
    spec = GenerationTaskSpec(
        task_id="t-3",
        task_type="scene-compile",
        params={"deep_revision_id": "deep-1", "camera_id": "cam-1"},
    )
    chain = build_task_chain(spec)
    assert [(s.activity, s.task_queue) for s in chain] == [
        ("scene-compile", "render3d-activities"),
        ("base-render", "render3d-activities"),
    ]
    assert chain[0].arg == "deep-1"
    assert chain[1].arg_from_upstream == {"scene_package_key": "scene_package_key"}


def test_chain_queues_follow_spec_routing_override() -> None:
    """集成测试收拢派发的口子：queues 覆写后链路整体改道。"""
    queues = TaskQueues(genpipe="q", render2d="q", imagegen="q", render3d="q")
    spec = GenerationTaskSpec(
        task_id="t-4", task_type="plan-2d-render", params={"revision_id": "rev-1"}, queues=queues
    )
    assert build_task_chain(spec)[0].task_queue == "q"


def test_missing_param_fails_loud() -> None:
    spec = GenerationTaskSpec(task_id="t-5", task_type="plan-2d-render", params={})
    with pytest.raises(PipelineDataError, match="missing-param:revision_id"):
        build_task_chain(spec)


def test_unknown_task_type_fails_loud() -> None:
    spec = GenerationTaskSpec.model_construct(
        task_id="t-6", task_type="mystery", params={}, queues=TaskQueues()
    )
    with pytest.raises(PipelineDataError, match="unknown-task-type:mystery"):
        build_task_chain(spec)


def test_evaluate_gate_requires_explicit_pass_from_both_checks() -> None:
    assert evaluate_gate({"passed": True}, {"passed": True}) == []
    assert evaluate_gate({"passed": False}, {"passed": True}) == ["consistency-check"]
    assert evaluate_gate({"passed": True}, {}) == ["compliance-check"]
    # 缺失/异常形态一律按未通过（绝不静默假成功）
    assert evaluate_gate(None, "ok") == ["consistency-check", "compliance-check"]


def test_resolve_step_arg_merges_upstream_and_fails_on_missing_key() -> None:
    step = TaskStep(
        activity="realism-pass",
        task_queue="imagegen-activities",
        arg={"style_ref": "s-1"},
        arg_from_upstream={"base_render_artifact_id": "artifact_id"},
    )
    merged = resolve_step_arg(step, {"artifact_id": "atm-1"})
    assert merged == {"style_ref": "s-1", "base_render_artifact_id": "atm-1"}
    with pytest.raises(PipelineDataError, match="missing-upstream:artifact_id"):
        resolve_step_arg(step, {})


def test_pick_template_rotates_across_regen_rounds() -> None:
    templates = ["a", "b", "c"]
    first_round = [pick_template(templates, 0, slot, 2) for slot in range(2)]
    second_round = [pick_template(templates, 1, slot, 2) for slot in range(2)]
    assert first_round == ["a", "b"]
    assert second_round == ["c", "a"]


def test_describe_failure_surfaces_pipeline_error_code() -> None:
    assert describe_failure(PipelineDataError("plan-rule-check:not-passed")) == (
        "plan-rule-check:not-passed"
    )


# ---------------------------------------------------------------------------
# 报告成文线
# ---------------------------------------------------------------------------


def _unit(domain: str, verdict: str = "ok", **overrides: Any) -> dict[str, Any]:
    """单元成文结果的最小形状（reportgen UnitComposeResult 的 dict 形态，编排侧只透传）。"""
    unit: dict[str, Any] = {
        "verdict": verdict,
        "domain": domain,
        "cards": [{"thesis": "t", "body": "b", "number_refs": []}] if verdict == "ok" else [],
        "violations": [],
        "rewrites_used": 0,
        "releases": [],
    }
    unit.update(overrides)
    return unit


def test_partition_keeps_ok_units_intact_and_records_rewrite_rounds() -> None:
    """全 ok：单元结果原样留给装配（不拆包重组），重写轮数按域归集。"""
    fanout = partition_unit_outcomes(
        ["dom-lighting", "dom-budget"],
        [_unit("dom-lighting"), _unit("dom-budget", rewrites_used=2)],
    )
    assert [u["domain"] for u in fanout.composed_units] == ["dom-lighting", "dom-budget"]
    assert fanout.composed_units[1]["rewrites_used"] == 2
    assert fanout.failed_domains == []
    assert fanout.rewrite_rounds_by_domain == {"dom-lighting": 0, "dom-budget": 2}


def test_partition_marks_failed_unit_and_keeps_its_violations() -> None:
    """某域 failed：失败单元原样保留（自带 domain 与 violations），不并入成功集。"""
    failed_unit = _unit(
        "dom-budget",
        verdict="failed",
        rewrites_used=2,
        violations=[{"check": "cr-budget-stale-price", "detail": "单价过期"}],
    )
    fanout = partition_unit_outcomes(
        ["dom-lighting", "dom-budget"], [_unit("dom-lighting"), failed_unit]
    )
    assert fanout.failed_domains == ["dom-budget"]
    assert fanout.failed_units == [failed_unit]
    assert [u["domain"] for u in fanout.composed_units] == ["dom-lighting"]
    assert fanout.rewrite_rounds_by_domain["dom-budget"] == 2


def test_partition_treats_dispatch_error_and_empty_cards_as_failure() -> None:
    """派发异常与"ok 却零卡片"都算失败——绝不静默假成功，也不拿空内容顶替。"""
    fanout = partition_unit_outcomes(
        ["dom-lighting", "dom-storage", "dom-material"],
        [
            PipelineDataError("report-unit-compose:non-dict-result"),
            _unit("dom-storage", cards=[]),
            _unit("dom-material"),
        ],
    )
    assert fanout.failed_domains == ["dom-lighting", "dom-storage"]
    assert fanout.dispatch_failures == [
        "report-unit-compose:dom-lighting:report-unit-compose:non-dict-result",
        "report-unit-compose:dom-storage:no-cards",
    ]
    assert [u["domain"] for u in fanout.composed_units] == ["dom-material"]


def test_merge_retried_units_keeps_earlier_successes_and_clears_healed_failures() -> None:
    """重开成了：成功单元累加，失败面整体换成本轮的——已治好的域不许还留在 failed_domains 里。"""
    base = partition_unit_outcomes(
        ["dom-lighting", "dom-budget"],
        [_unit("dom-lighting"), _unit("dom-budget", verdict="failed", rewrites_used=2)],
    )
    retried = partition_unit_outcomes(["dom-budget"], [_unit("dom-budget", rewrites_used=1)])
    merged = merge_retried_units(base, retried)

    assert [u["domain"] for u in merged.composed_units] == ["dom-lighting", "dom-budget"]
    assert merged.failed_domains == []
    assert merged.failed_units == []
    assert merged.dispatch_failures == []
    # 章内重写轮数按域取最后一次尝试的值（重开次数是另一个旋钮，不混算进来）
    assert merged.rewrite_rounds_by_domain == {"dom-lighting": 0, "dom-budget": 1}


def test_merge_retried_units_reports_last_attempt_failures_only() -> None:
    """重开还是不成：失败面是最后一次尝试的，不跨轮累加（否则读出根本没发生过的失败面）。"""
    base = partition_unit_outcomes(
        ["dom-lighting", "dom-budget"],
        [
            PipelineDataError("report-unit-compose:non-dict-result"),
            _unit("dom-budget", verdict="failed"),
        ],
    )
    retried = partition_unit_outcomes(
        ["dom-lighting", "dom-budget"],
        [
            _unit("dom-lighting"),
            _unit("dom-budget", verdict="failed", violations=[{"check": "cr-budget-stale-price"}]),
        ],
    )
    merged = merge_retried_units(base, retried)

    assert merged.failed_domains == ["dom-budget"]
    assert merged.failed_units == [retried.failed_units[0]]
    assert merged.failed_units[0]["violations"] == [{"check": "cr-budget-stale-price"}]
    assert merged.dispatch_failures == []  # 首轮那条派发失败码不跟着走：那一域这轮已经成了
    assert [u["domain"] for u in merged.composed_units] == ["dom-lighting"]


def test_collect_violations_passes_through_and_ignores_broken_shapes() -> None:
    assert collect_violations({"violations": [{"check": "cr-x", "detail": "d"}]}) == [
        {"check": "cr-x", "detail": "d"}
    ]
    assert collect_violations({"violations": "boom"}) == []
    assert collect_violations({}) == []


def test_unexplained_failure_gets_orchestration_code() -> None:
    """failed 却不给违规清单 = 失败无理由：补编排层失败码，杜绝失败被吞掉。"""
    assert unexplained_failure_checks("report-book-check", []) == [
        "report-book-check:failed-without-violations"
    ]
    assert unexplained_failure_checks("report-book-check", [{"check": "cr-set-closure"}]) == []
