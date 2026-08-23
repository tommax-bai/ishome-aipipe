"""svc_chat 迁移执行器（入口 `chat-migrate`）：纯 SQL 迁移，按序幂等执行。

- 迁移文件：services/chat-svc/migrations/V{n}__{描述}.sql（纯 SQL；表名不带
  schema 前缀——执行器先建 schema 并 SET search_path，同一组文件可原样应用到
  测试 schema）；
- 记账：{schema}.schema_migrations(version, filename, applied_at)，已应用版本
  跳过（重复执行为空操作）；每个迁移单事务应用 + 事务级咨询锁防并发重复执行；
- 不引重型迁移框架（Flyway/Alembic）——首批表量级用不上；V{n}__ 命名沿其惯例。

env：CHAT_DATABASE_URL（必填，如 postgresql://user:pass@localhost:15432/ishome）、
CHAT_DB_SCHEMA（默认 svc_chat）、CHAT_MIGRATIONS_DIR（默认仓内 migrations/）。

红线：只动本服务 schema（schema-per-service）；槽位真相唯一在
svc_project.slots，本 schema 永不建槽位真相表。
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA = "svc_chat"
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
"""workspace 成员为可编辑安装，__file__ 即仓内源码位——migrations/ 在服务根下。"""

_FILENAME_RE = re.compile(r"^V(\d+)__[A-Za-z0-9_\-]+\.sql$")
_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class Migration:
    version: int
    path: Path


def discover_migrations(directory: Path) -> list[Migration]:
    """扫描 V{n}__*.sql 并按版本号升序返回；命名不合规或版本重复即报错。"""
    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME_RE.match(path.name)
        if match is None:
            raise ValueError(f"迁移文件名不合规（应为 V{{n}}__{{描述}}.sql）：{path.name}")
        migrations.append(Migration(version=int(match.group(1)), path=path))
    migrations.sort(key=lambda m: m.version)
    versions = [m.version for m in migrations]
    if len(set(versions)) != len(versions):
        raise ValueError(f"迁移版本号重复：{versions}")
    if not migrations:
        raise ValueError(f"迁移目录为空：{directory}")
    return migrations


def run_migrations(
    conninfo: str, schema: str = DEFAULT_SCHEMA, directory: Path = MIGRATIONS_DIR
) -> list[str]:
    """应用未执行的迁移，返回本次应用的文件名（全部已应用则为空列表）。"""
    if _SCHEMA_RE.match(schema) is None:
        raise ValueError(f"schema 名不合规：{schema}")
    migrations = discover_migrations(directory)
    applied: list[str] = []
    # autocommit 连接：DDL 原子性由每个迁移自己的显式事务块保证
    with psycopg.connect(conninfo, autocommit=True) as conn:
        with conn.transaction():
            _advisory_lock(conn, schema)
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        with conn.transaction():
            _advisory_lock(conn, schema)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version integer PRIMARY KEY,"
                " filename text NOT NULL,"
                " applied_at timestamptz NOT NULL DEFAULT now())"
            )
        done_rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        done: set[int] = {row[0] for row in done_rows}
        for migration in migrations:
            if migration.version in done:
                continue
            with conn.transaction():
                _advisory_lock(conn, schema)
                # bytes 形态提交：psycopg Query 类型只收字面量/bytes，文件内容走 bytes
                conn.execute(migration.path.read_text(encoding="utf-8").encode("utf-8"))
                conn.execute(
                    "INSERT INTO schema_migrations (version, filename) VALUES (%s, %s)",
                    (migration.version, migration.path.name),
                )
            applied.append(migration.path.name)
            logger.info("applied %s -> schema %s", migration.path.name, schema)
    return applied


def _advisory_lock(conn: psycopg.Connection[TupleRow], schema: str) -> None:
    """事务级咨询锁：同 schema 的迁移互斥（事务结束自动释放）。"""
    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"chat-migrate:{schema}",))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    conninfo = os.environ.get("CHAT_DATABASE_URL")
    if not conninfo:
        raise SystemExit(
            "CHAT_DATABASE_URL 未设置（如 postgresql://user:pass@localhost:15432/ishome）"
        )
    schema = os.environ.get("CHAT_DB_SCHEMA", DEFAULT_SCHEMA)
    directory = Path(os.environ.get("CHAT_MIGRATIONS_DIR", str(MIGRATIONS_DIR)))
    applied = run_migrations(conninfo, schema=schema, directory=directory)
    if applied:
        logger.info("migrations applied: %s", ", ".join(applied))
    else:
        logger.info("migrations up-to-date (schema=%s)", schema)
