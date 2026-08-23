"""冒烟：全部工作区成员可导入，关键领域模型可实例化。"""

from __future__ import annotations


def test_import_all_members() -> None:
    import adapters
    import design.models
    import design.repo
    import design.router
    import design.service
    import design.workflows
    import genpipe.models
    import genpipe.repo
    import genpipe.router
    import genpipe.service
    import genpipe.workflows
    import genpipe_worker.activities
    import genpipe_worker.models
    import genpipe_worker.worker
    import patch_engine
    import scoring

    assert adapters and patch_engine and scoring
    assert design.models and design.repo and design.router and design.service and design.workflows
    assert genpipe.models and genpipe.repo and genpipe.router and genpipe.service
    assert genpipe.workflows and genpipe_worker.activities and genpipe_worker.models
    assert genpipe_worker.worker


def test_fact_cognitive_states() -> None:
    from design.models import Fact

    fact = Fact(
        target_id="space-living-room",
        property="width",
        value=4200,
        unit="mm",
        cognitive_state="inferred",
        fact_kind="dimensional",
        source="source_plan_scale",
        confidence=0.76,
        stage="preliminary",
    )
    assert fact.cognitive_state == "inferred"


def test_patch_validation() -> None:
    import pytest
    from patch_engine import Patch, PatchOp, PatchValidationError, validate_patch

    patch = Patch(
        intent="replace_double_desk_with_reading_corner",
        stage="preliminary",
        operations=[PatchOp(op="remove", target_id="desk-02")],
        reason="用户将书房目标调整为单人办公和阅读",
    )
    validate_patch(patch)

    with pytest.raises(PatchValidationError):
        validate_patch(Patch(intent="x", stage="preliminary", operations=[], reason="r"))
