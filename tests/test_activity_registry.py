"""activity 注册名与 contracts 注册表逐字一致的守门测试。

唯一真源：ishome-contracts `activities/registry.md`（只增不改）。本清单为其
副本；两处不一致时以 contracts 仓为准并回改此处。
"""

from __future__ import annotations

from genpipe_worker.activities import ACTIVITY_REGISTRY
from temporalio import activity

# 注册名 → 函数名（kebab-case ↔ snake_case 动词前置，规范 §2.4）
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
}


def test_registry_matches_contracts() -> None:
    assert set(ACTIVITY_REGISTRY) == set(CONTRACTS_ACTIVITY_REGISTRY)


def test_registered_temporal_names_match_keys() -> None:
    """字典键、@activity.defn(name=...) 注册名、函数名三者一致。"""
    for registry_name, fn in ACTIVITY_REGISTRY.items():
        defn = activity._Definition.from_callable(fn)  # noqa: SLF001
        assert defn is not None, f"{registry_name} is not a temporal activity"
        assert defn.name == registry_name
        assert fn.__name__ == CONTRACTS_ACTIVITY_REGISTRY[registry_name]
