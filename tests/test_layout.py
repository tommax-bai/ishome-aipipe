"""目录白名单结构测试：禁止 utils / common / helpers 垃圾场目录（规范 §6.3）。

import-linter 锁 import 方向；本测试锁目录形态——两者合起来执行 §1.3 的
"禁止 utils/、common/、helpers/ 目录"。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_DIR_NAMES = {"utils", "common", "helpers"}

ALLOWED_VERTICAL_FILES = {
    "__init__.py",
    "router.py",
    "service.py",
    "repo.py",
    "models.py",
    "workflows.py",
    "activities.py",
    "worker.py",
    # 2026-08-23 纵切评审通过的入站/出站边缘（import-linter 有对应方向契约）：
    "grpc_server.py",  # gRPC 入站适配，与 router 同级（contracts DesignService 服务端）
    "channel_client.py",  # 出站边缘：channel-svc gRPC 客户端（只依赖 contracts 生成代码）
    # 2026-08-23 Orchestrator v1 评审通过的服务层助手与出站边缘（import-linter 有对应契约）：
    "llm_client.py",  # 出站边缘：LiteLLM 网关客户端（任务级逻辑模型名，物理映射在 infra）
    "intent.py",  # 服务层助手：Intent Router（§5.2），只依赖 models 与协议位
    "orchestrator.py",  # 服务层助手：Design Orchestrator（§5.1），只依赖 models 与协议位
    # 2026-08-23 会话存储落库评审通过：
    "migrate.py",  # 迁移执行器边缘（chat-migrate 入口，纯 SQL 迁移；import-linter 契约禁触上层）
    # 2026-08-30 户型图解析评审通过（首版纯库 + CLI，同渲染层先例）；三件都有 import-linter
    # 方向契约。llm_client.py 不新增——genpipe_worker 的出站边缘与 chat 同名同角色，复用上面那条。
    "floorplan_cli.py",  # 工具入口，与 worker 同层互不可见（接 Temporal 的时点＝上传入口就绪时）
    "floorplan_parse.py",  # 解析编排：prompt 组装 / 模型输出解析 / 过产出侧校验
    "layout_features.py",  # 户型特征标记闭集（契约副本）加载与产出侧校验，纯确定性
    # 2026-08-30 晚 分区读与朝向换算评审通过（都有 import-linter 方向契约）：
    "floorplan_survey.py",  # 勘测一步：整图定位每个房间的区域 + 窗墙 + 指北针，不下结论
    "floorplan_regions.py",  # 确定性裁剪放大 + 逐块读图例（剪刀在代码手里，不在模型手里）
    "orientation.py",  # 朝向换算，纯确定性——方位与数字同族，都不由 LLM 决定
    # 2026-08-30 晚 几何提取评审通过（都有 import-linter 方向契约）：
    "floorplan_geometry.py",  # 墙线/门窗洞/房间遮罩，纯确定性——几何与方位、数字同族
    "floorplan_geometry_cli.py",  # 工具入口，与另两个入口同层互不可见（--survey 可零调用复跑）
    # 2026-08-31 户型事实评审通过（有 import-linter 方向契约）：
    "floorplan_facts.py",  # 几何 → 带 id 的结构化户型事实，纯确定性——空间推理背书通道的地基
    "floorplan_notes.py",  # 批注生成 + 引用机检：句子归模型、判据归代码（背书通道第一次实装）
}

# 纵切件允许拆为同名子包（文件 → 目录），当前评审通过的只有 repo（双实现：
# memory 默认 / pg 按 CHAT_DATABASE_URL 启用）；子包内模块方向由 import-linter 锁定。
ALLOWED_VERTICAL_PACKAGES = {"repo"}


def _source_dirs() -> list[Path]:
    return [p for p in REPO_ROOT.glob("services/*/src")] + [
        p for p in REPO_ROOT.glob("packages/*/src")
    ]


def test_no_dumping_ground_directories() -> None:
    offenders = [
        sub
        for src in _source_dirs()
        for sub in src.rglob("*")
        if sub.is_dir() and sub.name in FORBIDDEN_DIR_NAMES
    ]
    assert not offenders, f"垃圾场目录违规：{offenders}"


def test_services_keep_vertical_slice_shape() -> None:
    """services 的 domain 包顶层只允许纵切六件套文件/子包（新增横层先过评审）。"""
    offenders: list[Path] = []
    for src in [p for p in REPO_ROOT.glob("services/*/src")]:
        for domain_dir in [d for d in src.iterdir() if d.is_dir()]:
            for py in domain_dir.glob("*.py"):
                if py.name not in ALLOWED_VERTICAL_FILES:
                    offenders.append(py)
            for sub in domain_dir.iterdir():
                if (
                    sub.is_dir()
                    and sub.name != "__pycache__"
                    and sub.name not in ALLOWED_VERTICAL_PACKAGES
                ):
                    offenders.append(sub)
    assert not offenders, f"纵切形态外的顶层文件/目录：{offenders}"
