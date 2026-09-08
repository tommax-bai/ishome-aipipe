"""出站边缘：LiteLLM 网关的视觉调用客户端（OpenAI 兼容 /chat/completions，多模态 content）。

依赖方向（import-linter 锁定）：本模块只依赖运行库（httpx + stdlib），不感知产物模型与
解析编排——上层注入即可换实现，单测用桩件跑通全链路不打网络。

模型轴纪律（规范 §5.2 轴 3）：本模块与业务代码只出现**任务级逻辑模型名**
（户型图解析＝`floorplan-parse.default`）；逻辑名 → 物理 model_id 的映射唯一落点是
infra 仓的 LiteLLM 配置（ishome-infra/litellm/config.yaml），换模型改配置不改代码。
一轴一层：本模块不做第二层 API 适配（R7），只把消息按 OpenAI 兼容形态送出去。

与 chat-svc 的 `llm_client` 同名同角色、不共用代码：跨 domain 只许 import 对方 service，
出站边缘各自持有是既定分层，不是重复。
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)
"""模型调用**必须在日志里看得见**（2026-09-07 定）：解析一张户型图要调十来次网关，
"解析卡住了"与"某一次调用超时了"在业务库那段 JSON 里长得一模一样。级别与落点由进程入口
（`activity_log.configure_logging`）统一决定，本模块只管记、不碰配置——出站边缘不感知上层。"""

DEFAULT_LITELLM_BASE_URL = "http://127.0.0.1:4000/v1"

_TIMEOUT_SECONDS = 180.0
"""视觉模型读一张印刷级户型图明显慢于文本补全，超时按分钟级给。"""


class LlmGatewayError(Exception):
    """网关调用失败：**带上网关返回的正文**再抛。

    只说"HTTP 400"排不了错——逻辑模型名没在网关配置里（改了配置没重启是常见形态）
    与凭证失效返回的是同一个状态码，正文才分得开。
    """


def _first_choice_text(data: Any) -> Any:
    """首个 choice 的 content；形态不认识回 None（纯函数）。

    记日志与判形态**共用这一处**：日志要量回文多长、`_complete` 要判它是不是文本，
    两边各写一次取法就会出现"日志说有回文、判定说形态不认识"这种自相矛盾的现场。
    """
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


class LiteLlmVisionClient:
    """LiteLLM 网关的薄封装：一次「一张图 + 一段提示词」的补全调用。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = _TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("LITELLM_BASE_URL", DEFAULT_LITELLM_BASE_URL)
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY", "")
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client

    async def complete_text(
        self, model: str, system_prompt: str, user_prompt: str, *, temperature: float = 0.0
    ) -> str:
        """纯文本补全。**不读图的那一步就别把图递进去**——户型批注读的是算好的事实清单，
        递图等于给模型第二个可以照着编的来源，而事实这一层的整个用处就是让它只有一个来源。"""
        return await self._complete(
            {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        )

    async def complete_with_image(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        image_media_type: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        """送出「系统提示 + 用户提示 + 一张图」，返回首个 choice 的文本内容。

        图以 data URL 内联（本地文件驱动的工具形态没有可供模型回读的公网地址；
        上传入口就绪、图落私有 OSS 之后，此处可改传临时签名地址）。
        `model` 只接受任务级逻辑模型名。
        """
        data_url = f"data:{image_media_type};base64,{base64.b64encode(image_bytes).decode()}"
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        }
        return await self._complete(payload)

    async def _complete(self, payload: dict[str, Any]) -> str:
        """一次补全调用：送出、要回首个 choice 的文本。两种调用形态共用这一段。

        **进出各记一条**：逻辑模型名 + 带没带图 + 花了多久。记的是逻辑名不是物理模型
        （`floorplan-parse.default` 一类）——换模型改的是网关配置，日志里出现物理模型名
        等于把映射抄了第二份，换了就对不上。带没带图要记：读图的调用比纯文本慢一个量级，
        分不开这两类就看不出"慢"是慢在哪儿。
        """
        model = str(payload.get("model", "?"))
        with_image = any(
            isinstance(message.get("content"), list) for message in payload.get("messages", [])
        )
        shape = "图+文" if with_image else "纯文本"
        logger.info("网关调用 model=%s 形态=%s 超时=%.0fs", model, shape, self.timeout_seconds)
        started = time.monotonic()
        try:
            response = await self._http().post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 失败这条也带耗时：网关拒绝是秒级、超时是分钟级，这个数把"配置不对"与
            # "模型真的慢"分开——只看状态码分不出来。
            logger.warning(
                "网关调用失败 model=%s 形态=%s 耗时 %.3fs HTTP %d：%s",
                model,
                shape,
                time.monotonic() - started,
                e.response.status_code,
                e.response.text.strip()[:400],
            )
            raise LlmGatewayError(
                f"网关返回 {e.response.status_code}：{e.response.text.strip()[:800]}"
            ) from e
        except httpx.HTTPError as e:
            logger.warning(
                "网关调用失败 model=%s 形态=%s 耗时 %.3fs 不可达（%s）：%s",
                model,
                shape,
                time.monotonic() - started,
                self.base_url,
                e,
            )
            raise LlmGatewayError(f"网关不可达（{self.base_url}）：{e}") from e
        data = response.json()
        logger.info(
            "网关返回 model=%s 形态=%s 耗时 %.3fs 回文=%d字",
            model,
            shape,
            time.monotonic() - started,
            len(str(_first_choice_text(data))),
        )
        content = _first_choice_text(data)
        if content is None:
            raise LlmGatewayError(f"网关返回形态不认识：{str(data)[:800]}")
        if not isinstance(content, str):
            raise LlmGatewayError(f"补全内容不是文本：{type(content)!r}")
        return content

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
