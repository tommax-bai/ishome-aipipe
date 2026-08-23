"""genpipe 领域模型（pydantic，领域名词原样不带后缀；ORM 类加 Record 后缀区分）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RenderTier = Literal["preview", "final"]
"""渲染两档：preview 供会话内迭代与 Patch 即时反馈，final 为正式出图。"""

GateVerdict = Literal["passed", "regenerate", "switch_template", "failed"]
"""机检门禁结论：达标发布 / 自动重生成 / 换模板重试 / 重试超限终判失败（绝不静默假成功）。"""

GenerationTaskType = Literal["plan-2d-render", "atmosphere-visual", "scene-compile"]
"""交互侧生成任务类型：值与链路首个 activity 注册名同名（路由词表，只增不改）。"""


class TaskQueues(BaseModel):
    """activity 派发队列路由。

    默认值与 contracts `registries/task_queues.md` 逐字一致（只增不改）；
    字段存在的唯一目的，是让集成测试能把全部派发收拢到唯一随机队列——
    生产调用方一律不覆写。
    """

    genpipe: str = "genpipe-activities"
    render2d: str = "render2d-activities"
    imagegen: str = "imagegen-activities"
    render3d: str = "render3d-activities"


class GenBatchSpec(BaseModel):
    """一次批量预生成的排产输入（estate 交付日历驱动）。"""

    batch_id: str
    floorplan_id: str
    template_ids: list[str]
    candidate_count: int = 3
    render_tier: RenderTier = "preview"
    plan_master_artifact_id: str | None = None
    """已有母版则直接进候选 fan-out；为空则先走 solve → rule-check → plan-2d-render 备制。"""
    max_regen_rounds: int = 2
    """门禁不达标后允许的自动重生成轮数上限；超限返回 failed verdict。"""
    queues: TaskQueues = Field(default_factory=TaskQueues)


class MachineGateResult(BaseModel):
    """机检门禁（consistency-check + compliance-check + scorer 阈值）汇总结论。"""

    verdict: GateVerdict
    scorer_score: float | None = None
    failed_checks: list[str] = Field(default_factory=list)
    passed_artifact_ids: list[str] = Field(default_factory=list)
    rounds: int = 0
    """实际执行的候选生成轮数（首轮 + 重生成轮），失败路径可观测。"""


class GenerationTaskSpec(BaseModel):
    """交互侧生成任务的任务层输入（project-svc 创建任务 → 启动 workflow 的入口）。

    params 为任务参数快照：任务创建时刻定格，重放/重试不受后续修订影响。
    """

    task_id: str
    task_type: GenerationTaskType
    params: dict[str, Any] = Field(default_factory=dict)
    render_tier: RenderTier = "preview"
    queues: TaskQueues = Field(default_factory=TaskQueues)


class GenerationTaskResult(BaseModel):
    """交互侧生成任务的任务层结论（链路产物 + 门禁收尾）。"""

    task_id: str
    verdict: GateVerdict
    artifact_ids: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)


class TaskStep(BaseModel):
    """GenerationTaskWorkflow 路由链中的一步（纯数据，路由逻辑可直测）。

    绘图 activity 一律按注册名字符串 + task_queue 派发，跨服务不 import 存根签名。
    """

    activity: str
    task_queue: str
    arg: Any
    arg_from_upstream: dict[str, str] = Field(default_factory=dict)
    """payload 键 → 上游 activity 结果键：执行期把上游结果并入本步入参（arg 须为 dict）。"""
    long_running: bool = False
    """绘图/渲染类长跑 activity：派发时附加 heartbeat_timeout。"""


class WorkflowStartReceipt(BaseModel):
    """start workflow 即返回的回执（任务层异步：结果经事件回流，不在请求内等待）。"""

    workflow_id: str
    run_id: str
