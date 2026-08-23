"""contracts Python SDK（git 依赖 ishome-contracts@v0.1.2，subdirectory=gen/python）冒烟。

三域生成代码可导入、可构造即通过；类型以 contracts 仓为唯一真源，本仓禁手写契约类型。
"""

from ishome.channel.v1 import message_pb2
from ishome.common.v1 import channel_type_pb2, cognitive_state_pb2
from ishome.design.v1 import service_pb2_grpc


def test_contracts_types_resolve_across_all_three_domains() -> None:
    msg = message_pb2.UnifiedMessage(
        channel_type=channel_type_pb2.CHANNEL_TYPE_FEISHU,
        text=message_pb2.TextContent(text="hello"),
    )
    assert msg.channel_type == channel_type_pb2.CHANNEL_TYPE_FEISHU
    assert cognitive_state_pb2.COGNITIVE_STATE_USER_CONFIRMED != 0
    assert service_pb2_grpc.DesignServiceStub is not None
