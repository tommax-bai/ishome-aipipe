"""GenerationTask 路由链与机检门禁判定的纯函数直测（不依赖 Temporal 运行时）。"""

from __future__ import annotations

import pytest
from genpipe.models import GenerationTaskSpec, TaskQueues, TaskStep
from genpipe.workflows import (
    PipelineDataError,
    build_task_chain,
    describe_failure,
    evaluate_gate,
    pick_template,
    resolve_step_arg,
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
