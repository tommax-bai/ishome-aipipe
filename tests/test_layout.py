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
}


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
    """services 的 domain 包顶层只允许纵切六件套文件（新增横层文件先过评审）。"""
    offenders: list[Path] = []
    for src in [p for p in REPO_ROOT.glob("services/*/src")]:
        for domain_dir in [d for d in src.iterdir() if d.is_dir()]:
            for py in domain_dir.glob("*.py"):
                if py.name not in ALLOWED_VERTICAL_FILES:
                    offenders.append(py)
    assert not offenders, f"纵切形态外的顶层文件：{offenders}"
