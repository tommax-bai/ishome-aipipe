"""activity 注册名与 contracts 注册表逐字一致的守门测试。

唯一真源：ishome-contracts `activities/registry.md`（注册名，只增不改）与
`registries/task_queues.md`（队列归属，V1.4 绘图拆分口径）。本清单为其副本；
两处不一致时以 contracts 仓为准并回改此处。
"""

from __future__ import annotations

from genpipe_worker.activities import ACTIVITY_REGISTRY
from temporalio import activity

# 注册名 → 函数名（kebab-case ↔ snake_case 动词前置，规范 §2.4）——contracts 全量 10 项
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

# task queue → 承接 activity（contracts registries/task_queues.md，逐字一致）
CONTRACTS_TASK_QUEUE_OWNERSHIP: dict[str, set[str]] = {
    "genpipe-activities": {
        "floorplan-parse",
        "plan-layout-solve",
        "plan-rule-check",
        "consistency-check",
        "compliance-check",
    },
    "render2d-activities": {"plan-2d-render"},
    "imagegen-activities": {"atmosphere-visual", "realism-pass"},
    "render3d-activities": {"scene-compile", "base-render"},
}


def test_contracts_registry_total_unchanged() -> None:
    """V1.4 绘图拆分只搬家不改名：contracts 注册表总量仍为 10。"""
    assert len(CONTRACTS_ACTIVITY_REGISTRY) == 10


def test_queue_ownership_partitions_contracts_registry() -> None:
    """四条队列的承接集合两两不交，且并集恰为 contracts 全量注册表。"""
    union: set[str] = set()
    for owned in CONTRACTS_TASK_QUEUE_OWNERSHIP.values():
        assert not (union & owned), "同一 activity 不得归属多条队列"
        union |= owned
    assert union == set(CONTRACTS_ACTIVITY_REGISTRY)


def test_worker_registry_matches_genpipe_queue_ownership() -> None:
    """genpipe-worker 只承接 genpipe-activities 队列的 5 个非绘图 activity。"""
    assert set(ACTIVITY_REGISTRY) == CONTRACTS_TASK_QUEUE_OWNERSHIP["genpipe-activities"]
    assert len(ACTIVITY_REGISTRY) == 5


def test_registered_temporal_names_match_keys() -> None:
    """字典键、@activity.defn(name=...) 注册名、函数名三者一致。"""
    for registry_name, fn in ACTIVITY_REGISTRY.items():
        defn = activity._Definition.from_callable(fn)  # noqa: SLF001
        assert defn is not None, f"{registry_name} is not a temporal activity"
        assert defn.name == registry_name
        assert fn.__name__ == CONTRACTS_ACTIVITY_REGISTRY[registry_name]
