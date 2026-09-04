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

ReportVerdict = Literal["ok", "failed"]
"""成文线结论词表：与 reportgen activity 的 verdict 同词——编排侧只透传不改写。"""

ReportStage = Literal["unit-compose", "page-assemble", "book-check", "book-render"]
"""成文线三阶段（图 v0.2 §2 流水线 / §4 校验三作用域）：值 = activity 注册名去 `report-`
前缀，失败定位无需翻译表。"""


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
    reportgen: str = "reportgen-activities"
    reportrender: str = "reportrender-activities"


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


class ReportComposeSpec(BaseModel):
    """报告成文线一次生成的编排输入（求值线出包 → 派发成文线，图 v0.2 §2）。"""

    report_id: str
    """本次生成的定址 id：workflow_id 由其派生，重复派发即冲突上抛。里程碑真相在
    `svc_project` 表，本 id 不承载状态（规则 8.1 禁第二台状态机）。"""
    domains: list[str]
    """本次派发的 dom- 单元集合，由求值线给出。**不从数据包里读**——数据包对编排是不透明
    载荷；派发集与包内不符时由 activity 出 gate-domain-not-in-package，响亮失败不静默兜底。"""
    package: dict[str, Any]
    """报告数据包原样载荷：schema 归 contracts `rulebook/report_data_package.schema.json`，
    本仓不建模其字段——数字在求值线已算完，成文线只搬运（图 v0.2 §0）。"""
    max_rewrites: int = 2
    """单元出口过检不合格后的**章内**重写轮数上限，随派发下发（图 v0.2 §3：≤2 轮，仍不过即
    failed）。设计定数，不许私改；与 `max_unit_retries` 是两个旋钮，不要混。"""
    max_unit_retries: int = 2
    """某章终判 failed 后允许的**整章重开**次数上限（编排侧旋钮，**不下发给 activity**）。

    与 `max_rewrites` 的分工：`max_rewrites` 是一次派发**内部**的自我修正轮数，由 reportgen
    侧执行；本字段是那一次派发整体失败之后，编排侧拿同样入参**另起一次**派发。之所以有效，
    是因为章失败是随机的不是必然的（实测同一份输入连跑五次：dom-budget 在第 1/3/4 跑失败、
    第 2/5 跑通过，dom-lighting 只在第 5 跑失败）。生产调用方不传，走默认。"""
    queues: TaskQueues = Field(default_factory=TaskQueues)


class UnitFanoutOutcome(BaseModel):
    """各 dom- 单元并行成文的归并结果（纯数据，归并逻辑可直测）。"""

    composed_units: list[dict[str, Any]] = Field(default_factory=list)
    """verdict=ok 的单元结果，原样进装配——编排侧不拆包重组（避免在成文线复刻包结构）。"""
    failed_domains: list[str] = Field(default_factory=list)
    """失败单元的域：派发异常与 verdict=failed 两类合并归集，是"下次派哪些单元"的输入。"""
    failed_units: list[dict[str, Any]] = Field(default_factory=list)
    """verdict=failed 的单元结果原样保留：自带 domain 与本域 violations，回传即带归属。"""
    dispatch_failures: list[str] = Field(default_factory=list)
    """派发层失败码（activity 异常 / 结果形态违约）：无 activity 结论可透传时的兜底记录。"""
    rewrite_rounds_by_domain: dict[str, int] = Field(default_factory=dict)
    """各域实际重写轮数：出口过检拦截强度的观测量（规则 4.17 自迭代回路的输入信号）。"""


class ReportComposeResult(BaseModel):
    """报告成文线一次生成的结论（成功 = 四阶段全过：成文 → 装配 → 册检 → 出册）。"""

    report_id: str
    verdict: ReportVerdict
    failed_stage: ReportStage | None = None
    """失败发生在哪一阶段；verdict=ok 时为空。"""
    pages: list[dict[str, Any]] = Field(default_factory=list)
    """装配产物原样回传，**只在 verdict=ok 时非空**：失败册不回内容，杜绝调用方"拿到 pages
    就发布"的误用路径；失败诊断走 failed_units / violations。"""
    failed_domains: list[str] = Field(default_factory=list)
    failed_units: list[dict[str, Any]] = Field(default_factory=list)
    """单元级失败原样回传（各自带 violations）——图 v0.2 §4 三作用域之首。"""
    violations: list[dict[str, Any]] = Field(default_factory=list)
    """页级 / 册级违规原样回传（三作用域之二三）；单元级违规在 failed_units 内，
    不并流以免丢掉域归属。"""
    failed_checks: list[str] = Field(default_factory=list)
    """编排层失败码（入参违约 / 派发异常 / 结果形态违约），与既有 workflow 同名同义。"""
    rewrite_rounds_by_domain: dict[str, int] = Field(default_factory=dict)
    """各域**章内**重写轮数（activity 内部的自我修正），语义与 `max_rewrites` 对应。"""
    unit_retries_by_domain: dict[str, int] = Field(default_factory=dict)
    """各域**整章重开**次数（编排侧另起派发），语义与 `max_unit_retries` 对应；**只记 >0 的**，
    空字典 = 这一趟一次都没重开。留着它是为了让"这册是不是靠重开出来的"在结论里直接判得出来——
    否则真跑成册与一次过成册在结论上长得一模一样，重开是否值得就成了只能猜的事。"""
    book_key: str | None = None
    """出册后册在私有对象存储里的键（`reports/{report_id}/book.html`，contracts
    `registries/object_keys.md`）。**只是回执不是台账**——键由 report_id 确定性推得，
    调用方不必留着它也能算出来；这里带上是为了让"这一趟到底出没出册"在结论里一眼可见。
    编排侧不签链接：签名是"给谁看、看多久"的事，属业务侧（生成侧不知用户是谁）。"""


FloorplanVisualsProduct = Literal[
    "mood_image",
    "brief_image",
    "style_image",
    "plan_master",
    "floorplan_geometry",
    "floorplan_reading",
]
"""三张图生成线交回的产物词表（contracts `openapi/genpipe.v1.yaml`
floorplan_visuals_product，只增不改）。业务侧把它映射成自己的 artifact_type——两套词表各归各的仓。"""


class FloorplanVisualTemplates(BaseModel):
    """三张免费图用哪些模板（数据，不是代码）：情绪图底图 + 手账写字版风格图。

    默认值＝现行拍定组合（contracts `registries/templates.md` 2026-09-04 落定）；
    生产调用方不传。功能说明图不经图像模型（render2d 制图），不在此列。
    """

    mood: str = "cream-journal"
    style: str = "lifestyle-notebook-handwritten"
    annotate_style: bool = False
    """风格图上写不写注释。**默认关**：imagegen 的数据自洽门禁要求"注释提到的实体必须在该房间的
    物件清单里"，而物件清单今天没有生成步（9-01 那批是人手写的派发数据）——只递注释不递清单，
    当场被拒（2026-09-04 真派发第四跑：客厅/主卧/餐厅三条注释提到灯与床，
    被 atmosphere-failed 拦下）。
    裁决"图上要有注释"（2026-09-01）不变；**打开的时点写死＝物件清单生成步（按事实与家庭假设推，
    机检同批注）落地那一次**，届时注释与清单同批递。"""


class FloorplanVisualsSpec(BaseModel):
    """三张免费图一次生成的编排输入（project-svc 铸任务 → 派发入口，contracts genpipe.v1）。"""

    task_id: str
    """生成任务 id（project-svc 铸造的 ULID）：workflow_id 由其派生，重复派发即冲突上抛。"""
    floorplan_object_key: str
    """用户上传的户型图在私有桶里的键（`uploads/{content_sha256}/original.{ext}`）。"""
    building_area_sqm: float | None = None
    floor_area_ratio_percent: float | None = None
    result_callback_url: str
    """结果回流地址：由派发方注入，编排侧不知道业务侧在哪（规范 §1.0 向上通信只走回调）。"""
    templates: FloorplanVisualTemplates = Field(default_factory=FloorplanVisualTemplates)
    max_caption_retries: int = 2
    """情绪图叠字放不下时，**重生成底图再叠**的次数上限（编排侧旋钮，生产调用方不传）。

    叠字那一步的失败形态本来就是"重生成一张，不把字压在画面上"（render2d `style_caption`）：模板向
    模型要了留白，它给多少是它的事——2026-09-04 真派发第五跑顶部只给 138px（需 235px）。重生成赌的是
    同一次生成的随机性，每次一张图的钱（0.60 元）；耗尽仍放不下即整任务失败，不硬叠。这与挑图门禁
    "生成→量分→不过线重出"是同一形态的第一处落地。"""
    queues: TaskQueues = Field(default_factory=TaskQueues)


class TaskProduct(BaseModel):
    """一件交回业务侧的产物（project.v1 `generation_task_product` 的编排侧形态）。"""

    product: FloorplanVisualsProduct
    object_key: str
    content_type: str | None = None
    gen_params: dict[str, Any] = Field(default_factory=dict)


class FloorplanVisualsResult(BaseModel):
    """三张图生成线一次运行的结论（同时也是回调报文的来源）。"""

    task_id: str
    verdict: ReportVerdict
    products: list[TaskProduct] = Field(default_factory=list)
    """verdict=ok 时含三张图 + 母版 + 几何（+ 特征解析）；failed 时是已出的半成品，只留血缘。"""
    failed_checks: list[str] = Field(default_factory=list)
    """编排层失败码：主链某步失败、特征解析失败（不致命）、回调送不到，都记在这里。"""
    failure: dict[str, Any] | None = None
    """主链失败原因 {code, detail}；verdict=ok 时为空。"""
    delivered: bool = False
    """结论是否已送到回调地址。**没送到不算完**：业务侧不知道就等于没做。"""


class WorkflowStartReceipt(BaseModel):
    """start workflow 即返回的回执（任务层异步：结果经事件回流，不在请求内等待）。"""

    workflow_id: str
    run_id: str
