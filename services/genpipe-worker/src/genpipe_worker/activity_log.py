"""worker 侧的运行留痕：每个 activity 一进一出各一条，失败那条带上判据原文。

**为什么是独立一层**：日志怎么配、一次 activity 该留下哪几条、入参出参里哪些字段可以进日志——
这三样是同一个问题（"事后靠什么复盘"）的三个面，散进 `worker` 与 `activities` 就会各写一半。
它挂在 `activities` 下一层：上面的 activity 用它留痕，下面的纯库层（解析、勘测、几何、
批注、文案）看不见它——**那一层响亮失败靠抛异常，留痕是调用方的事**。
出站边缘（`llm_client`）自己用标准库 `logging` 记每次网关调用，不经本模块：它按契约不许感知
上层（模型轴 R7 一轴一层），一条"调了哪个逻辑名、花了多久"用不着 activity 上下文。

口径的来处（2026-09-07 真机取证）：三个执行侧 worker 当天 journal 零输出，两次生成失败除了
业务库 `generation_tasks.result` 里那段 JSON 无从复盘——不知道每步花了多久、不知道失败前一步
是什么状态。按那次定的：

- **进**：注册名、workflow / run 标识、第几次尝试、入参里的键与计数；
- **出**：verdict、耗时、产物键与自证数；
- **失败**：violations 逐条原文（WARNING）；抛出去的异常带 traceback（ERROR）后原样再抛。

**入参出参不整个打**：`floorplan-geometry-extract` 的回执里内联着整份几何与事实清单，
`plan-notes-write` 的入参是整张事实清单，原样打出来一次就是几十 KB，翻日志的人反而找不到
那一行。只放"名字像标识与量"的字段（`_LOGGED_SUFFIXES` / `_LOGGED_NAMES`），列表只留长度
——**这是生产服务，不是调试脚本**。几何与事实要复盘时按 `geometry_key` 去桶里取，
那里本来就落了一份存档。
"""

from __future__ import annotations

import functools
import io
import logging
import os
import sys
import time
from collections.abc import Callable, Coroutine
from typing import Any

from temporalio import activity

logger = logging.getLogger(__name__)

LOG_LEVEL_ENV = "ISHOME_LOG_LEVEL"
"""级别从环境来（unit 的 EnvironmentFile 里加一行就能调），不写死在代码里也不做成入参。"""

_DEFAULT_LEVEL = "INFO"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
"""与 chat-svc 既有形态一致（`chat.grpc_server` / `chat.migrate`），本仓不另发明一套。"""

_QUIET_LIBRARIES = ("oss2", "PIL", "urllib3", "httpx", "httpcore")
"""第三方库的 INFO 压到 WARNING：oss2 每次取放对象记一条、httpx 每次请求记一条，解析一张
户型图要调十来次模型，业务节点会被冲没。网关调用另有 `llm_client` 记的那条——那条带逻辑
模型名与耗时，httpx 那条只有 URL 与状态码。要看它们时 `ISHOME_LOG_LEVEL=DEBUG` 一并让开。"""

_LOGGED_SUFFIXES = (
    "_key",
    "_id",
    "_count",
    "_px",
    "_ratio",
    "_seconds",
    "_bytes",
    "_score",
    # 地址：结论往哪儿回调、图从哪个网关出——"送没送到"排查的第一个问题就是"送到哪儿了"
    "_url",
)
_LOGGED_NAMES = frozenset(
    {"verdict", "bucket", "seed", "view_kind", "content_type", "status_code", "backend_name"}
)
"""可以进日志的字段按**名字**挑，不按值挑：标识（键 / id）与量（计数 / 像素 / 比率 / 耗时 /
字节数 / 分数）。名字没进这两张表的一律不打——几何、提示词、文案正文都在此列。"""

_MAX_VALUE_CHARS = 120
_MAX_DIGEST_CHARS = 600
_MAX_VIOLATIONS_CHARS = 2000
"""violations 给得比别处宽：几何门禁一次报十几条缺什么、标记越界一次报一批键名，砍到几百字
就只剩"有错"而看不出错在哪一条——它恰恰是要靠日志回答的那一类。"""


def configure_logging() -> None:
    """进程入口调一次（组合根 `worker.main`）。

    **不在 import 时调**：CLI 与单测各有各的口径，一个库模块不该替进程决定日志往哪儿写。

    写 stderr 并**显式要行缓冲**：systemd 把 stdout / stderr 都接进 journald，但 stdout 在
    非终端下是块缓冲——攒够几 KB 才落盘，进程被 kill 时"死之前在干什么"跟着一起丢。走 stderr
    没这个问题，再显式要一次行缓冲，`PYTHONUNBUFFERED` 配没配都一样。
    """
    level_name = os.environ.get(LOG_LEVEL_ENV, _DEFAULT_LEVEL).strip().upper()
    level = logging.getLevelNamesMapping().get(level_name, logging.INFO)
    stream = sys.stderr
    if isinstance(stream, io.TextIOWrapper):
        stream.reconfigure(line_buffering=True)
    # force=True：谁在 import 期间先装过 handler 都让位，否则本进程的级别与格式是别人定的
    logging.basicConfig(level=level, format=_LOG_FORMAT, stream=stream, force=True)
    if level > logging.DEBUG:
        for name in _QUIET_LIBRARIES:
            logging.getLogger(name).setLevel(logging.WARNING)


def _context(fallback: str) -> str:
    """这一跑的身份：注册名 + workflow / run + 第几次尝试。

    不在 activity 上下文里（单测直接调实现件）就退回函数名——**留痕不该成为"这段代码只能在
    Temporal 里跑"的理由**。
    """
    if not activity.in_activity():
        return f"activity={fallback}"
    info = activity.info()
    return (
        f"activity={info.activity_type} workflow={info.workflow_id}"
        f" run={info.workflow_run_id} attempt={info.attempt}"
    )


def current_run_ref() -> str | None:
    """这一跑的运行编号：Temporal 的 workflow id，不在 activity 上下文里就是 None。

    给网关那侧的调用记录用（请求体 `metadata.run_ref`）——网关只看得见逻辑模型名，
    一次生成里解析一张图要调十来次，串得回"这些是同一跑"靠的就是它。

    **不新造标识**：workflow id 是编排本来就有的那一个，与上面 `_context` 打进日志的
    `workflow=` 是同一个值，日志与调用记录对得上。取不到就是 None，不编一个——
    CLI 与单测直接调实现件时没有 workflow，那时"没有运行编号"是事实。
    """
    if not activity.in_activity():
        return None
    return activity.info().workflow_id


def _one_line(text: object) -> str:
    """压成一行：换行与连续空白都并成一个空格。

    pydantic 一次报几十条、每条三行，原样进日志就是一条记录横跨上百行——**一次事件一行**
    才 grep 得动，翻日志的人也才找得到那一行。同 worker 装模板时"不把 pydantic 的多行报告
    原样甩出来"的既有口径。
    """
    return " ".join(str(text).split())


def digest(payload: object) -> str:
    """入参 / 出参里可以进日志的那部分，压成一行 `k=v`。

    列表只留长度：批注、事实清单、violations 都是列表，长度回答的是"到这一步手里有几条"，
    内容要么另行打（violations），要么本来就不该进日志（整份几何）。
    """
    if not isinstance(payload, dict):
        return f"<{type(payload).__name__}>"
    parts: list[str] = []
    for raw_name, value in payload.items():
        name = str(raw_name)
        if isinstance(value, list):
            parts.append(f"{name}=[{len(value)}]")
            continue
        if not (name in _LOGGED_NAMES or name.endswith(_LOGGED_SUFFIXES)):
            continue
        if value is None:
            # 值是 None 与"没有这个键"不是一回事（前者是"这一跑没有"，后者是形态变了），要留得住
            parts.append(f"{name}=None")
        elif isinstance(value, str | int | float | bool):
            text = _one_line(value)
            if len(text) > _MAX_VALUE_CHARS:
                text = f"{text[:_MAX_VALUE_CHARS]}…（共{len(text)}字）"
            parts.append(f"{name}={text}")
    line = " ".join(parts) or "（无可记字段）"
    return line if len(line) <= _MAX_DIGEST_CHARS else f"{line[:_MAX_DIGEST_CHARS]}…"


def _violations_text(result: dict[str, Any]) -> str:
    """失败判据逐条原文——它就是复盘时唯一要读的那一段，不做归并、不改措辞，只压行。"""
    raw = result.get("violations")
    if not isinstance(raw, list) or not raw:
        return "（没给 violations）"
    items = [
        f"{item.get('check', '?')}={_one_line(item.get('detail', ''))}"
        if isinstance(item, dict)
        else _one_line(item)
        for item in raw
    ]
    text = " | ".join(items)
    return text if len(text) <= _MAX_VIOLATIONS_CHARS else f"{text[:_MAX_VIOLATIONS_CHARS]}…"


ActivityMethod = Callable[[Any, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]
"""activity 实现件的形状：`(self, 不透明字典) -> 不透明字典`，三个仓的每一个都长这样。

写成 `Coroutine` 不是 `Awaitable`：注册表那边的类型就是 `Coroutine`，套上这一层之后要能
原样放回去——形状对不上，"套了留痕的还是不是同一个 activity"就得靠人看而不是靠类型检查。"""


def logged_activity(fn: ActivityMethod) -> ActivityMethod:
    """给一个 activity 实现件套上一进一出两条日志。**贴在 `@activity.defn` 底下**——
    注册给 Temporal 的是套好的这一层，注册名与签名由 `functools.wraps` 原样带过去。

    只在这里套一层，不往每个 return 前面手写一行：本层的失败是**返回值**不是异常
    （`verdict=failed` 交给编排定重不重试），一个 activity 七八个 return，手写必漏——
    漏掉的那条往往恰是出事那次要看的。
    """

    @functools.wraps(fn)
    async def logged(self: Any, request: dict[str, Any]) -> dict[str, Any]:
        context = _context(fn.__name__)
        logger.info("%s 开始 入参 %s", context, digest(request))
        started = time.monotonic()
        try:
            result = await fn(self, request)
        except Exception as e:
            # 抛出去的异常＝瞬时故障，Temporal 会重试。带 traceback 记一条再**原样抛**：
            # 不吞、不改形态；重试到第几次由上面 attempt= 那一段读得出来。
            logger.error(
                "%s 抛异常 耗时 %.3fs %s：%s",
                context,
                time.monotonic() - started,
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise
        elapsed = time.monotonic() - started
        if result.get("verdict") == "failed":
            logger.warning(
                "%s 失败 耗时 %.3fs 判据 %s 现场 %s",
                context,
                elapsed,
                _violations_text(result),
                digest(result),
            )
        else:
            logger.info("%s 成功 耗时 %.3fs 产物 %s", context, elapsed, digest(result))
        return result

    return logged
