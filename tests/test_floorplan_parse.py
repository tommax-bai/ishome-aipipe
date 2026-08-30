"""户型图解析：闭集校验（硬门禁）、模型输出解析、编排与 CLI 的守门测试。

重点在**越界路径**：闭集校验是"键写错就永远不触发且不报错"这一最贵失效形态的唯一拦截点，
所以它的测试要覆盖的不是"通过"而是"不通过时报不报得出是哪个键"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from genpipe_worker.floorplan_cli import main as floorplan_cli_main
from genpipe_worker.floorplan_parse import (
    FloorplanParseError,
    build_user_prompt,
    parse_model_output,
    read_floorplan_features,
)
from genpipe_worker.layout_features import (
    CLOSED_SET_FILE,
    LayoutFeatureSetError,
    LayoutFeatureViolation,
    check_features,
    load_closed_set,
)

CONTRACTS_CLOSED_SET = (
    Path.home() / "codes" / "ishome-contracts" / "rulebook" / "layout_features.json"
)

# 契约闭集首版四条（只增不改，故断言"含"而不是"等于"）。
CONTRACT_FEATURES = {"west_facing", "kitchen_u_shape", "bedroom_east_facing", "balcony_service"}


class StubVisionReader:
    """视觉补全桩件：记下收到的 prompt，吐回预设原文（单测不打网络）。"""

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_bytes": image_bytes,
                "image_media_type": image_media_type,
            }
        )
        return self.output

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 闭集副本与契约一致
# ---------------------------------------------------------------------------


def test_closed_set_copy_matches_contracts() -> None:
    """本仓闭集副本与 contracts 真源逐字一致（真源不在本机时跳过，同 activity 注册名口径）。"""
    if not CONTRACTS_CLOSED_SET.exists():
        pytest.skip("本机无 ishome-contracts 工作副本")
    assert CLOSED_SET_FILE.read_text(encoding="utf-8") == CONTRACTS_CLOSED_SET.read_text(
        encoding="utf-8"
    ), "闭集副本与 contracts 真源不一致：以 contracts 为准回改本仓副本"


def test_closed_set_contains_contract_features() -> None:
    closed_set = load_closed_set()
    assert set(closed_set) >= CONTRACT_FEATURES
    assert all(meaning.strip() for meaning in closed_set.values())


def test_broken_closed_set_file_fails_loud(tmp_path: Path) -> None:
    """闭集读不出来就直接失败——不许在不知道闭集的情况下产出标记。"""
    broken = tmp_path / "layout_features.json"
    broken.write_text(json.dumps({"features": {}}), encoding="utf-8")
    with pytest.raises(LayoutFeatureSetError):
        load_closed_set(broken)
    with pytest.raises(LayoutFeatureSetError):
        load_closed_set(tmp_path / "缺这个文件.json")


# ---------------------------------------------------------------------------
# 越界路径（本模块存在的理由）
# ---------------------------------------------------------------------------


def test_out_of_set_key_fails_loud_and_names_the_key() -> None:
    closed_set = load_closed_set()
    with pytest.raises(LayoutFeatureViolation) as excinfo:
        check_features({"kitchen_l_shape": "厨房两排操作台呈 L 形"}, closed_set)
    assert any("kitchen_l_shape" in line for line in excinfo.value.details)


def test_every_out_of_set_key_is_reported_not_just_the_first() -> None:
    closed_set = load_closed_set()
    with pytest.raises(LayoutFeatureViolation) as excinfo:
        check_features(
            {
                "balcony_service": "阳台端头画了洗衣机设备位",
                "kitchen_l_shape": "厨房是 L 形",
                "bay_window_all_bedrooms": "每间卧室都画了飘窗",
            },
            closed_set,
        )
    # 每条明细形如 "越界标记 `X`：…"，被点名的键取自反引号内（消息里还会列出闭集全集）。
    offending = {line.split("`")[1] for line in excinfo.value.details}
    assert offending == {"kitchen_l_shape", "bay_window_all_bedrooms"}  # 闭集内那条不被连坐


def test_empty_evidence_fails() -> None:
    closed_set = load_closed_set()
    with pytest.raises(LayoutFeatureViolation) as excinfo:
        check_features({"balcony_service": "   "}, closed_set)
    assert "依据为空" in "；".join(excinfo.value.details)


def test_measured_number_in_evidence_fails() -> None:
    """依据里的量纲数字＝报告里出现 LLM 决定的数字（红线），机检拦。"""
    closed_set = load_closed_set()
    for evidence in ("阳台净宽约 1500 毫米", "厨房操作面约 4 平方米", "占开间三分之一"):
        with pytest.raises(LayoutFeatureViolation) as excinfo:
            check_features({"balcony_service": evidence}, closed_set)
        assert "量纲数字" in "；".join(excinfo.value.details), evidence


def test_standard_code_in_evidence_fails() -> None:
    closed_set = load_closed_set()
    with pytest.raises(LayoutFeatureViolation) as excinfo:
        check_features({"balcony_service": "按 GB 50096 阳台应设家政位"}, closed_set)
    assert "标准号" in "；".join(excinfo.value.details)


def test_negated_evidence_fails() -> None:
    """把闭集当逐条打勾的清单填＝四条规则带着否定的理由全部触发（2026-08-30 真跑形态）。"""
    closed_set = load_closed_set()
    for evidence in (
        "厨房台面呈 L 形，非U形厨房",
        "图上无朝向标注，无法判定是否西晒",
        "阳台区域未画出家政柜、洗衣机位，仅标为‘阳台’",  # 2026-08-30 真跑逐字取样
    ):
        with pytest.raises(LayoutFeatureViolation) as excinfo:
            check_features({"kitchen_u_shape": evidence}, closed_set)
        assert "否定句" in "；".join(excinfo.value.details), evidence


def test_descriptive_absence_in_evidence_passes() -> None:
    """描述图面的"无"是正当依据，不该被否定句这道闸误伤。"""
    check_features(
        {"balcony_service": "阳台与客厅之间无隔墙，阳台端头画了洗衣机设备位"},
        load_closed_set(),
    )


def test_counting_words_in_evidence_pass() -> None:
    """纯计数描述的是图上画了什么，不是设计参数——校验不该把它误伤。"""
    check_features(
        {"balcony_service": "阳台端头画了两处虚线设备位，其中一处是洗衣机图例"},
        load_closed_set(),
    )


def test_clean_features_pass() -> None:
    check_features(
        {
            "balcony_service": "阳台内画有洗衣机设备位与地漏",
            "kitchen_u_shape": "厨房三面均画了操作台",
        },
        load_closed_set(),
    )


# ---------------------------------------------------------------------------
# 模型输出解析
# ---------------------------------------------------------------------------


def test_parse_model_output_tolerates_code_fence_and_chatter() -> None:
    raw = """好的，这是判读结果：
```json
{"layoutFeatures": {"balcony_service": "阳台画了洗衣机位"},
 "observations": [{"subject": "厨房", "finding": "开放式，与餐厅连通"}],
 "unreadable": [{"subject": "分房间尺寸", "reason": "全图无任何尺寸标注"}]}
```
"""
    features = parse_model_output(raw)
    assert features.layout_features == {"balcony_service": "阳台画了洗衣机位"}
    assert features.observations[0].subject == "厨房"
    assert features.unreadable[0].reason == "全图无任何尺寸标注"


def test_parse_model_output_rejects_unknown_top_level_key() -> None:
    with pytest.raises(FloorplanParseError):
        parse_model_output('{"layoutFeatures": {}, "dimensions": {"living": 4200}}')


def test_parse_model_output_rejects_non_json() -> None:
    with pytest.raises(FloorplanParseError):
        parse_model_output("这张图我看不清楚，无法判读。")


def test_product_serializes_with_contract_key_names() -> None:
    features = parse_model_output('{"layoutFeatures": {"balcony_service": "画了洗衣机位"}}')
    dumped = features.model_dump(by_alias=True)
    assert set(dumped) == {"layoutFeatures", "observations", "unreadable"}


# ---------------------------------------------------------------------------
# 编排：prompt 由闭集数据驱动 + 端到端越界
# ---------------------------------------------------------------------------


def test_user_prompt_lists_every_closed_set_feature() -> None:
    """闭集从契约数据生成、不手写进 prompt——契约加一条，prompt 自动多一行。"""
    closed_set = load_closed_set()
    prompt = build_user_prompt(closed_set)
    for name, meaning in closed_set.items():
        assert name in prompt
        assert meaning in prompt


async def test_read_floorplan_features_happy_path() -> None:
    reader = StubVisionReader(
        '{"layoutFeatures": {"balcony_service": "阳台端头画了洗衣机设备位"},'
        ' "observations": [{"subject": "厨房", "finding": "L 形开放式"}],'
        ' "unreadable": [{"subject": "尺寸", "reason": "无任何尺寸标注"}]}'
    )
    reading = await read_floorplan_features(b"\x89PNG-fake", "image/png", reader)
    assert reading.logical_model == "floorplan-parse.default"
    assert reading.features.layout_features == {"balcony_service": "阳台端头画了洗衣机设备位"}
    assert len(reader.calls) == 1  # 一次调用读整张图（两步读试过并撤回，见解析编排模块）
    assert reader.calls[0]["image_media_type"] == "image/png"
    assert reader.calls[0]["image_bytes"] == b"\x89PNG-fake"


async def test_empty_features_is_a_normal_result() -> None:
    """一条标记都不成立是正常结果不是失败——漏报只少触发规则，误报会让报告写错。"""
    reader = StubVisionReader('{"layoutFeatures": {}}')
    reading = await read_floorplan_features(b"x", "image/png", reader)
    assert reading.features.layout_features == {}
    assert reading.features.observations == []


async def test_read_floorplan_features_raises_on_out_of_set_key() -> None:
    """模型把闭集外的东西写进标记时，整次解析失败——不静默剔除。"""
    reader = StubVisionReader('{"layoutFeatures": {"kitchen_l_shape": "厨房呈 L 形"}}')
    with pytest.raises(LayoutFeatureViolation) as excinfo:
        await read_floorplan_features(b"x", "image/png", reader)
    assert any("kitchen_l_shape" in line for line in excinfo.value.details)


# ---------------------------------------------------------------------------
# CLI：退出码分道（越界 3 / 其他失败 2 / 成功 0）
# ---------------------------------------------------------------------------


def _install_stub_client(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    def factory(base_url: str | None = None, api_key: str | None = None) -> StubVisionReader:
        return StubVisionReader(output)

    monkeypatch.setattr("genpipe_worker.floorplan_cli.LiteLlmVisionClient", factory)


def test_cli_exits_nonzero_and_names_the_key_on_out_of_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "floorplan.png"
    image.write_bytes(b"\x89PNG-fake")
    _install_stub_client(monkeypatch, '{"layoutFeatures": {"kitchen_l_shape": "厨房呈 L 形"}}')
    exit_code = floorplan_cli_main(["--image", str(image)])
    assert exit_code == 3
    assert "kitchen_l_shape" in capsys.readouterr().err


def test_cli_writes_archive_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "floorplan.png"
    image.write_bytes(b"\x89PNG-fake")
    out = tmp_path / "reading.json"
    _install_stub_client(
        monkeypatch, '{"layoutFeatures": {"balcony_service": "阳台画了洗衣机设备位"}}'
    )
    exit_code = floorplan_cli_main(["--image", str(image), "-o", str(out)])
    assert exit_code == 0
    archived = json.loads(out.read_text(encoding="utf-8"))
    assert archived["product"]["layoutFeatures"] == {"balcony_service": "阳台画了洗衣机设备位"}
    assert archived["logicalModel"] == "floorplan-parse.default"
    assert archived["rawOutput"]  # 模型原文随档，判定可逐字复核
    assert archived["image"]["sha256"]
    assert "balcony_service" in capsys.readouterr().out


def test_cli_rejects_unknown_image_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "floorplan.pdf"
    image.write_bytes(b"%PDF-fake")
    _install_stub_client(monkeypatch, "{}")
    assert floorplan_cli_main(["--image", str(image)]) == 2
    assert "不认识的图片格式" in capsys.readouterr().err
