"""真网络会话冒烟（可选，需本机 LiteLLM 网关在线，默认 http://127.0.0.1:4000）。

用途：验证 design-svc Orchestrator 走真实网关（逻辑模型名路由）的一轮多消息
对话——事实抽取、追问、确认清单触发。不启动 gRPC，直接驱动 service 层，
出站用内存捕获。CI 不跑本脚本（真网络）。

用法：
    set -a; source ~/.ishome/llm-local.env; set +a
    LITELLM_API_KEY=$LITELLM_MASTER_KEY uv run python scripts/live-conversation-smoke.py
"""

from __future__ import annotations

import asyncio
import sys

from design import service
from design.llm_client import LiteLlmClient
from design.repo import reset_conversations, reset_seen_messages
from ishome.channel.v1 import message_pb2
from ishome.common.v1 import channel_type_pb2

USER_TURNS = [
    "你好，我想设计我的家",
    "我家在翠湖天地，89平两室两厅，一家三口，孩子5岁",
    "入户门宽我量过，900毫米。最想要的是收纳多、客厅显大；预算别超过20万。没有绝对不能接受的，哦对了阳台不能封",
]


class PrintingSender:
    def __init__(self) -> None:
        self.count = 0

    async def send(self, message: message_pb2.UnifiedMessage, idempotency_key: str) -> str:
        self.count += 1
        text = (
            message.quick_reply.prompt_text
            if message.WhichOneof("content") == "quick_reply"
            else message.text.text
        )
        print(f"\n<<< 设计顾问（{message.WhichOneof('content')}）：\n{text}")
        return message.message_id


async def main() -> int:
    reset_seen_messages()
    reset_conversations()
    llm = LiteLlmClient()
    sender = PrintingSender()
    print(f"gateway = {llm.base_url}")
    try:
        for i, text in enumerate(USER_TURNS):
            print(f"\n>>> 用户：{text}")
            inbound = message_pb2.UnifiedMessage(
                message_id=f"smoke-{i}",
                channel_type=channel_type_pb2.CHANNEL_TYPE_MOCK,
                channel_instance="mock:local",
                direction=message_pb2.MESSAGE_DIRECTION_INBOUND,
                external_user_id="u-smoke",
            )
            inbound.text.CopyFrom(message_pb2.TextContent(text=text))
            await service.ingest_message(inbound, sender, llm)
    finally:
        await llm.aclose()
    if sender.count < len(USER_TURNS):
        print(f"\nSMOKE FAIL：入站 {len(USER_TURNS)} 条，出站仅 {sender.count} 条")
        return 1
    print(f"\nSMOKE OK：{len(USER_TURNS)} 入站 / {sender.count} 出站")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
