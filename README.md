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

- 纵切文件形态：`router.py / service.py / repo.py / models.py / workflows.py / activities.py`
- `workflows.py` 只编排禁 IO（只许 import models 与 activity 注册名）——Temporal 可重放硬约束
- `router → service → repo` 单向；跨 domain 只许 import 对方 `service`
- `packages/*` 禁止反向依赖 `services/*`
- 禁 `utils/ common/ helpers/` 目录（tests/test_layout.py 结构测试执行）

## Temporal 口径

- **Temporal 收缩至任务层**（V1.5 裁决）：仅 `genpipe` namespace（生成管线 workflow/activity，重试/心跳/超时用原生语义）。原"设计项目=长周期 Temporal workflow（continue-as-new）"方案作废——里程碑真相在表 + 事件驱动 checkCompletion（project-svc）。
- **activity 注册名只增不改**，唯一真源 ishome-contracts `activities/registry.md`；本仓 `tests/test_activity_registry.py` 与其保持逐字一致。

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
