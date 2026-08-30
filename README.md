# ishome-aipipe

《是我的家》生成域 monorepo（uv workspace）。基线文档在中控仓 `ishome`（技术架构 §五 仓库拆分、开发规范 §1.3 分层）。

## Workspace 地图

```
services/
  genpipe-svc/      # 生成工作流编排（Temporal workflow）+ 机检门禁状态机（全自动）  [Python, FastAPI]
  genpipe-worker/   # 10 个 activity 执行：解析/布局/规则/母版/风格图/场景/渲染/机检   [Python]
  chat-svc/         # 会话引擎（V1.5，由 design-svc 更名）：理解/情绪/封闭动作集/记忆画像/承诺区/主动消息；ProjectState 属主已移交 project-svc（backend）[Python, FastAPI]
packages/
  scoring/          # 打分器端口（变化轴 #6，scorer_id 入 contracts 注册表）
  adapters/         # 各模型提示词/出入参变换（LiteLLM 已是 API 层 adapter，本包禁止再造 API 适配）
  patch-engine/     # 结构化 Patch 操作模型 + 机械校验
template/           # copier 脚手架：新 domain 纵切目录（分层靠模板生成，不靠人记）
tests/              # 工作区级守门测试（activity 注册名一致性、目录白名单）
```

## 分层与 import 契约（规范文档 §1.3，import-linter 真实生效，违规挂流水线）

- 纵切文件形态：`router.py / service.py / repo.py / models.py / workflows.py / activities.py`（纵切件可评审拆为同名子包——chat 的 `repo/` 已拆 memory/pg 双实现）
- `workflows.py` 只编排禁 IO（只许 import models 与 activity 注册名）——Temporal 可重放硬约束
- `router → service → repo` 单向；跨 domain 只许 import 对方 `service`
- `packages/*` 禁止反向依赖 `services/*`
- 禁 `utils/ common/ helpers/` 目录（tests/test_layout.py 结构测试执行）

## Temporal 口径

- **Temporal 收缩至任务层**（V1.5 裁决）：仅 `genpipe` namespace（生成管线 workflow/activity，重试/心跳/超时用原生语义）。原"设计项目=长周期 Temporal workflow（continue-as-new）"方案作废——里程碑真相在表 + 事件驱动 checkCompletion（project-svc）。
- **activity 注册名只增不改**，唯一真源 ishome-contracts `activities/registry.md`；本仓 `tests/test_activity_registry.py` 与其保持逐字一致。
- **报告成文线**（`ReportComposeWorkflow`，图 v0.2 §2）：各 dom- 单元并行成文 → 页面装配 → 册级校验，三个 activity 全部派往 `reportgen-activities`（实现在 ishome-reportgen）。数字在求值线（project-svc 规则引擎）算完，报告数据包对本仓是**不透明载荷**（schema 归 contracts `rulebook/`，本仓不建模其字段）；任一域失败即整册失败，不出"其余页"。

## 户型图解析（`floorplan-parse`，genpipe-worker）

一张户型图 → **户型特征标记 + 依据**，喂报告求值线的 `layoutFeatures`（口子早就留好、一直传空）。

**形态（2026-08-30，同渲染层先例）**：不成 activity，以工具（纯库 + CLI）形式存在；
`floorplan-parse` 的 activity 存根保持不动，**接线时点写死＝上传入口就绪时**——那时图落私有 OSS、
activity 拿到的是资产键，与现在的本地文件入参不是一回事。

```bash
LITELLM_API_KEY=$LITELLM_MASTER_KEY uv run floorplan-parse \
  --image services/genpipe-worker/samples/floorplan-brochure-92sqm-3b2l1b.png \
  --gateway http://127.0.0.1:4001/v1 -o reading.json
```

- **模型逐条判定、代码只投影成立的那些**（2026-08-30 用户裁决）：模型对闭集里每条候选给
  `holds` 与依据，代码把判成立的投影成产物。让模型直接产 map 时，结构里"这条不成立"没有位置，
  模型只能把否定写进依据栏——真跑出现过"名字全在、依据全是否定句"，在键存在即触发的语义下
  四条规则全会触发。**结构性堵死 > 纪律禁止**：给"否"一个位置，这类产出不再产生。
  **下游契约形态与语义不变**；
- 产出三段：`layoutFeatures`（**键 ⊆ 契约闭集**，值＝这条标记成立的依据）、`observations`
  （闭集外的观察，**记录但不下发**——下发闭集外的键等于宣称有规则会用它）、`unreadable`
  （读不出的东西，响亮说明不留空）；
- **闭集校验是硬门禁，分两层**：判定层查名字 ⊆ 闭集（**不论判成立与否**），产物层查名字 + 空依据
  + 判定与依据自相矛盾 + 依据里的量纲数字/标准号。任一命中即**响亮失败并报出是哪个键**
  （退出码 3），不修剪不静默——键写错就永远不触发且不报错是本项目最贵的失效形态；
- **这一步不出任何数字**：尺寸/面积/比例要等比例标定链路，且算术由确定性代码做不由模型做；
- 模型经 LiteLLM 网关按逻辑名调用；**改了网关配置要重启才认新逻辑名**，而 4000 是常驻网关
  ——真跑另起临时端口（`ishome-infra/litellm/run-dev.sh 4001`）跑完即停。

真跑留档（被否的两个候选模型、两步读为什么撤回、"把闭集当逐条打勾的清单填"这个失败形态，
以及改造后复跑的收益与代价——一整类失效消失，但这张样本上唯一那条真阳性丢了）：
`_iteration/run-2026-08-30-floorplan-features/run.md`。

## chat-svc 会话存储（svc_chat）

- 迁移：`CHAT_DATABASE_URL=postgresql://... uv run chat-migrate`——纯 SQL 迁移在
  `services/chat-svc/migrations/`（`V{n}__` 命名），`schema_migrations` 记账、按序幂等，无重型框架。
- 运行：`CHAT_DATABASE_URL` 设置时消息原文（入站+出站）落 PG schema `svc_chat`；未设时内存后端
  （backend 的 e2e-mock-smoke 裸起 chat-grpc 即此路径）。会话态（项目快照/上下文历史）为进程内
  缓存，Redis 接入位在 `chat.repo.memory.SessionCache`。
- 红线：槽位真相唯一在 `svc_project.slots`（project-svc 属主），svc_chat 永不落槽位真相；
  `episodic_memories` 待 pgvector 基础镜像切换后 V2 建表（见 V1 迁移头注）。

## 常用命令

```bash
uv sync                 # 安装全部成员与 dev 工具
uv run ruff check .     # lint
uv run lint-imports     # import 方向契约
uv run mypy             # strict 类型检查
uv run pytest           # 测试
uv run copier copy template/ services/<new-svc>/   # 新 domain 脚手架
```

## contracts SDK 消费

Python SDK 以 git 依赖消费 ishome-contracts 生成代码（`gen/python`）：

```toml
# TODO：contracts 首个 tag 发布后取消注释并锁定 tag
# ishome-contracts = { git = "ssh://git@github.com/tommax-bai/ishome-contracts.git", subdirectory = "gen/python", tag = "v0.1.0" }
```

禁止手写跨服务客户端（技术架构 §2.2）。

## 本机开发备注

国内网络下 `uv sync` 走官方 index 很慢，可用：

```bash
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple uv sync --all-packages
```

CI（GitHub Actions）在境外 runner 上直连官方 index，无需镜像。

## 本地质量门（pre-push）

云端 CI 停用期间的本地把关：push 前自动跑本仓全套检查。新 clone 后执行一次 `git config core.hooksPath .githooks` 启用；紧急绕过用 `git push --no-verify`。
