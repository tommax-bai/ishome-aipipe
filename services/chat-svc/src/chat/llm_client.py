"""出站边缘：LiteLLM 网关客户端（OpenAI 兼容 /chat/completions）。

依赖方向（import-linter 锁定）：本模块只依赖运行库（httpx），不感知
service / models 等上层——由组合根（grpc_server.serve）注入 service 层的
LlmCompletion 协议位。

模型轴纪律（规范 §5.2 轴 3）：本模块与业务代码只出现**任务级逻辑模型名**
（如 design-orchestrator.default）；逻辑名 → 物理 model_id 的映射唯一落点是
infra 仓的 LiteLLM 配置（ishome-infra/litellm/config.yaml），换模型改配置不改代码。

TODO(langfuse)：Langfuse 逐会话成本追踪在网关侧接入（success_callback），
本客户端届时透传 trace 元数据（session/user 标识）。
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

DEFAULT_LITELLM_BASE_URL = "http://127.0.0.1:4000/v1"

_TIMEOUT_SECONDS = 60.0


class LiteLlmClient:
    """LiteLLM 网关的薄封装（实现 service.LlmCompletion 协议）。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (
            base_url or os.environ.get("LITELLM_BASE_URL", DEFAULT_LITELLM_BASE_URL)
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY", "")
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)
        return self._client

    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        *,
        call_point: str,
        run_ref: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """一次补全调用，返回首个 choice 的文本内容。

        model 只接受任务级逻辑模型名；json_mode 请求 JSON 输出（网关
        drop_params 兜底不支持的物理模型）。

        `call_point`（这次调用是哪一处 AI 判断）与 `run_ref`（属于哪次运行）随请求体
        `metadata` 送给网关，落进网关那侧的调用记录。键名与形状由记录那边定
        （infra 的 `custom/ledger_callback.py` 从 `metadata` 里认这两个键），这里照抄。

        `call_point` **必填、不给默认值**：漏传的调用点要在类型检查与测试里当场露出来；
        给个默认值等于让它悄悄记到 `unregistered` 名下。`run_ref` 取不到就是 None，不编。
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "metadata": {"call_point": call_point, "run_ref": run_ref},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = await self._http().post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError(f"unexpected completion content type: {type(content)!r}")
        return content

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
