"""两个网关客户端送出去的请求体里，那两个标记长什么样。

**为什么单立一份**：`call_point`（这次调用是哪一处 AI 判断）与 `run_ref`（属于哪次运行）
放在请求体 `metadata` 下、就叫这两个名字——这不是本仓自己定的，是网关那侧记调用记录的
`custom/ledger_callback.py` 认的两个键（判官台 `gateway/ledger_callback.py` 是它的源文件）。
键名写错不会报错，只会让每条记录的 AI 判断名默默落成 `unregistered`，而那正是没人会发现的
那种失败——所以要有一处按字面盯住它。

两个客户端不共码（chat-svc 与 genpipe-worker 各持一份出站边缘，是既定分层不是重复），
但送出去的形状必须一样，所以两边的断言并排放在这里。
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from chat.llm_client import LiteLlmClient
from genpipe_worker.llm_client import LiteLlmVisionClient

_OK = {"choices": [{"message": {"content": "回文"}}]}


class _Capture:
    """接住请求体的假传输层：不打网络，只把送出去的 JSON 留下。"""

    def __init__(self) -> None:
        self.body: dict[str, Any] = {}

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            import json

            self.body = json.loads(request.content)
            return httpx.Response(200, json=_OK)

        return httpx.MockTransport(handle)


@pytest.mark.asyncio
async def test_vision_client_puts_both_marks_under_metadata() -> None:
    capture = _Capture()
    client = LiteLlmVisionClient(base_url="http://gw/v1", api_key="k")
    # 直接塞进假传输层：客户端自己懒建 AsyncClient，测试不为此给生产代码开注入口
    client._client = httpx.AsyncClient(transport=capture.transport())

    await client.complete_with_image(
        "floorplan-parse.default",
        "系统",
        "用户",
        b"\x89PNG\r\n\x1a\n",
        "image/png",
        call_point="floorplan-survey",
        run_ref="wf-1",
    )

    assert capture.body["metadata"] == {"call_point": "floorplan-survey", "run_ref": "wf-1"}
    await client.aclose()


@pytest.mark.asyncio
async def test_vision_client_text_call_carries_the_marks_too() -> None:
    """纯文本那条路（批注 / 文案）与读图那条路走的是同一个 `_complete`，两条都要带。"""
    capture = _Capture()
    client = LiteLlmVisionClient(base_url="http://gw/v1", api_key="k")
    client._client = httpx.AsyncClient(transport=capture.transport())

    await client.complete_text("floorplan-notes.default", "系统", "用户", call_point="x")

    # 取不到运行编号时 `run_ref` 是 null，不是"键不见了"——落记录时"这一跑没有"要留得住
    assert capture.body["metadata"] == {"call_point": "x", "run_ref": None}
    await client.aclose()


@pytest.mark.asyncio
async def test_chat_client_puts_both_marks_under_metadata() -> None:
    capture = _Capture()
    client = LiteLlmClient(base_url="http://gw/v1", api_key="k")
    client._client = httpx.AsyncClient(transport=capture.transport())

    await client.complete(
        "design-intent.default",
        [{"role": "user", "content": "你好"}],
        call_point="chat-intent",
        run_ref="mock:local:ou_x#in-1",
        json_mode=True,
    )

    assert capture.body["metadata"] == {
        "call_point": "chat-intent",
        "run_ref": "mock:local:ou_x#in-1",
    }
    # 标记是加上去的，原来送什么还送什么
    assert capture.body["response_format"] == {"type": "json_object"}
    await client.aclose()
