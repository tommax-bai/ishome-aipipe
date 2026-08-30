"""户型图解析：两层校验（硬门禁）、逐条判定的解析与投影、编排与 CLI 的守门测试。

重点在**越界路径**：闭集校验是"键写错就永远不触发且不报错"这一最贵失效形态的唯一拦截点，
所以它的测试要覆盖的不是"通过"而是"不通过时报不报得出是哪个键"。

2026-08-30 改造后模型输出层是**逐条判定**（feature/holds/evidence），产物由投影得到；
故越界路径分两处测：判定层的名字（`holds` 真假都要拦）与投影后的产物。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from genpipe_worker.floorplan_cli import main as floorplan_cli_main
from genpipe_worker.floorplan_parse import (
    FloorplanParseError,
    build_system_prompt,
    build_user_prompt,
    parse_model_output,
    read_floorplan_features,
    to_floorplan_features,
)
from genpipe_worker.floorplan_regions import (
    RoomCropError,
    RoomLegendError,
    crop_room,
    parse_room_legend,
    read_room_legends,
)
from genpipe_worker.floorplan_survey import FloorplanSurveyError, parse_survey_output
from genpipe_worker.layout_features import (
    CLOSED_SET_FILE,
    LayoutFeatureSetError,
    LayoutFeatureViolation,
    check_feature_names,
    check_features,
    load_closed_set,
)
from genpipe_worker.models import (
    FeatureVerdict,
    FloorplanVerdicts,
    LayoutObservation,
    RoomLegend,
    RoomOrientation,
    RoomRegion,
    UnreadableGap,
)
from genpipe_worker.orientation import (
    DEFAULT_NORTH_POINTS_TO,
    to_cardinal,
    to_room_orientations,
)

CONTRACTS_CLOSED_SET = (
    Path.home() / "codes" / "ishome-contracts" / "rulebook" / "layout_features.json"
)

# 契约闭集首版四条（只增不改，故断言"含"而不是"等于"）。
CONTRACT_FEATURES = {"west_facing", "kitchen_u_shape", "bedroom_east_facing", "balcony_service"}


SURVEY_STUB = (
    '{"northPointsTo": "top", "rooms": ['
    '{"name": "\u9633\u53f0", "box": [0.40, 0.55, 0.66, 0.66]},'
    '{"name": "\u53a8\u623f", "box": [0.45, 0.18, 0.68, 0.34]}]}'
)
"""勘测桩：阳台在下、厨房在上。窗墙不在这一层——它在近景里定。"""

LEGEND_TEXT = "端头画有两处虚线框，靠墙一侧有一个带圆点的小图形"
LEGEND_STUB = f'{{"legend": "{LEGEND_TEXT}", "windowWalls": ["bottom"]}}'
"""近景桩：读到图例，并报窗开在下侧墙。"""


def make_png(width: int = 400, height: int = 600) -> bytes:
    """造一张真 PNG——裁剪那一步要真解码，假字节过不了。"""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (250, 250, 250)).save(buffer, format="PNG")
    return buffer.getvalue()


class StubVisionReader:
    """视觉补全桩件：记下收到的 prompt，按**这一步是哪一步**吐回预设原文（单测不打网络）。

    解析分三步（勘测 / 逐块读图例 / 判定），逐块那步是并发的、次序不保证，
    所以按系统提示认步，不按调用序号认步。
    """

    def __init__(
        self,
        output: str,
        survey_output: str = SURVEY_STUB,
        legend_output: str = LEGEND_STUB,
    ) -> None:
        self.output = output
        self.survey_output = survey_output
        self.legend_output = legend_output
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
        if "勘测员" in system_prompt:
            return self.survey_output
        if "放大的一小块" in system_prompt:
            return self.legend_output
        return self.output

    @property
    def verdict_prompt(self) -> str:
        """判定那一步收到的用户提示（图例与朝向都在里面）。"""
        return str(
            next(
                call["user_prompt"] for call in self.calls if "候选标记清单" in call["user_prompt"]
            )
        )

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


def test_out_of_set_name_fails_loud_and_names_the_key() -> None:
    closed_set = load_closed_set()
    with pytest.raises(LayoutFeatureViolation) as excinfo:
        check_feature_names(["kitchen_l_shape"], closed_set)
    assert any("kitchen_l_shape" in line for line in excinfo.value.details)


async def test_out_of_set_name_fails_even_when_judged_not_holding() -> None:
    """`holds=False` 的越界名同样拦：它说明模型在编造闭集里没有的标记名。

    "反正不会下发所以无害"不成立——下一次它可能把同一个名字判成成立。
    """
    reader = StubVisionReader(
        '{"verdicts": [{"feature": "kitchen_l_shape", "holds": false,'
        ' "evidence": "厨房两排操作台呈 L 形"}]}'
    )
    with pytest.raises(LayoutFeatureViolation) as excinfo:
        await read_floorplan_features(make_png(), "image/png", reader)
    assert any("kitchen_l_shape" in line for line in excinfo.value.details)


def test_out_of_set_key_after_projection_fails_loud() -> None:
    """产物层照旧查名字——这一道拦的是投影这段代码自己写错。"""
    closed_set = load_closed_set()
    with pytest.raises(LayoutFeatureViolation) as excinfo:
        check_features({"kitchen_l_shape": "厨房两排操作台呈 L 形"}, closed_set)
    assert any("kitchen_l_shape" in line for line in excinfo.value.details)


def test_every_out_of_set_key_is_reported_not_just_the_first() -> None:
    closed_set = load_closed_set()
    with pytest.raises(LayoutFeatureViolation) as excinfo:
        check_feature_names(
            ["balcony_service", "kitchen_l_shape", "bay_window_all_bedrooms"],
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


def test_self_contradicting_evidence_fails() -> None:
    """判成立、依据却在说它不成立：`holds` 与 `evidence` 各说各话，而下游只看键在不在。

    三条措辞逐字取自 2026-08-30 真跑（那一版结构里没有"否"的位置，模型只能这么写）。
    """
    closed_set = load_closed_set()
    for evidence in (
        "厨房台面呈 L 形，非U形厨房",
        "图上无朝向标注，无法判定是否西晒",
        "阳台区域未画出家政柜、洗衣机位，仅标为‘阳台’",
    ):
        with pytest.raises(LayoutFeatureViolation) as excinfo:
            check_features({"kitchen_u_shape": evidence}, closed_set)
        assert "自相矛盾" in "；".join(excinfo.value.details), evidence


def test_descriptive_absence_in_evidence_passes() -> None:
    """描述图面的"无"是正当依据，不该被自相矛盾这道闸误伤。"""
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
# 模型输出层：逐条判定的解析
# ---------------------------------------------------------------------------


def test_parse_model_output_tolerates_code_fence_and_chatter() -> None:
    raw = """好的，这是判读结果：
```json
{"verdicts": [{"feature": "balcony_service", "holds": true, "evidence": "阳台画了洗衣机位"},
              {"feature": "kitchen_u_shape", "holds": false, "evidence": "厨房台面沿两面墙"}],
 "observations": [{"subject": "厨房", "finding": "开放式，与餐厅连通"}],
 "unreadable": [{"subject": "分房间尺寸", "reason": "全图无任何尺寸标注"}]}
```
"""
    verdicts = parse_model_output(raw)
    assert [(v.feature, v.holds) for v in verdicts.verdicts] == [
        ("balcony_service", True),
        ("kitchen_u_shape", False),
    ]
    assert verdicts.observations[0].subject == "厨房"
    assert verdicts.unreadable[0].reason == "全图无任何尺寸标注"


def test_parse_model_output_rejects_unknown_top_level_key() -> None:
    with pytest.raises(FloorplanParseError):
        parse_model_output('{"verdicts": [], "dimensions": {"living": 4200}}')


def test_parse_model_output_rejects_verdict_without_holds() -> None:
    """判定必须给 holds——缺了就退回"这条成不成立没有位置"的旧形态。"""
    with pytest.raises(FloorplanParseError):
        parse_model_output(
            '{"verdicts": [{"feature": "balcony_service", "evidence": "画了洗衣机位"}]}'
        )


def test_parse_model_output_rejects_non_json() -> None:
    with pytest.raises(FloorplanParseError):
        parse_model_output("这张图我看不清楚，无法判读。")


# ---------------------------------------------------------------------------
# 投影：判定 → 产物（下游契约在这一层，形态与语义不变）
# ---------------------------------------------------------------------------


def test_projection_keeps_only_holding_verdicts() -> None:
    verdicts = FloorplanVerdicts(
        verdicts=[
            FeatureVerdict(feature="balcony_service", holds=True, evidence="阳台画了洗衣机位"),
            FeatureVerdict(feature="kitchen_u_shape", holds=False, evidence="厨房台面沿两面墙"),
            FeatureVerdict(feature="west_facing", holds=False, evidence="图上无朝向依据"),
        ],
        observations=[LayoutObservation(subject="厨房", finding="L 形开放式")],
        unreadable=[UnreadableGap(subject="尺寸", reason="无任何尺寸标注")],
    )
    features = to_floorplan_features(verdicts)
    assert features.layout_features == {"balcony_service": "阳台画了洗衣机位"}
    assert features.observations[0].subject == "厨房"  # 观察区与读不出区原样带过去
    assert features.unreadable[0].subject == "尺寸"


def test_projection_rejects_a_feature_judged_twice() -> None:
    """同一条判两次即失败——"取后一条"这种默认行为会把矛盾藏起来。"""
    verdicts = FloorplanVerdicts(
        verdicts=[
            FeatureVerdict(feature="balcony_service", holds=True, evidence="画了洗衣机位"),
            FeatureVerdict(feature="balcony_service", holds=False, evidence="阳台是空的"),
        ]
    )
    with pytest.raises(FloorplanParseError) as excinfo:
        to_floorplan_features(verdicts)
    assert "balcony_service" in "；".join(excinfo.value.details)


def test_product_serializes_with_contract_key_names() -> None:
    """下游契约形态一个字没动：产物仍是 layoutFeatures / observations / unreadable。"""
    features = to_floorplan_features(
        FloorplanVerdicts(
            verdicts=[
                FeatureVerdict(feature="balcony_service", holds=True, evidence="画了洗衣机位")
            ]
        )
    )
    dumped = features.model_dump(by_alias=True)
    assert set(dumped) == {"layoutFeatures", "observations", "unreadable"}
    assert dumped["layoutFeatures"] == {"balcony_service": "画了洗衣机位"}


# ---------------------------------------------------------------------------
# 编排：prompt 由闭集数据驱动 + 端到端越界
# ---------------------------------------------------------------------------


def test_user_prompt_lists_every_closed_set_feature() -> None:
    """候选清单从契约数据生成、不手写进 prompt——契约加一条，prompt 自动多一行。"""
    closed_set = load_closed_set()
    prompt = build_user_prompt(closed_set, [], [])
    for name, meaning in closed_set.items():
        assert name in prompt
        assert meaning in prompt


def test_user_prompt_carries_legends_and_computed_orientations() -> None:
    """判定那一步吃的是**逐房间图例**与**算好的朝向**，不是让它自己再看一遍指北针。"""
    prompt = build_user_prompt(
        load_closed_set(),
        [RoomLegend(room="阳台", legend="端头画有两处虚线框")],
        [RoomOrientation(room="主卧", window_walls=["bottom"], facings=["南"])],
    )
    assert "端头画有两处虚线框" in prompt
    assert "主卧：窗开在bottom，朝南" in prompt
    assert "不要自己再推一遍" in prompt


def test_user_prompt_omits_rooms_without_windows_from_orientations() -> None:
    """没画窗的房间不进朝向栏——提示里另说了"没列到＝没画窗"，免得模型替它补一个。"""
    prompt = build_user_prompt(
        load_closed_set(),
        [],
        [RoomOrientation(room="卫生间", window_walls=[], facings=[])],
    )
    assert "卫生间：窗开在" not in prompt


def test_system_prompt_shows_no_negative_example() -> None:
    """提示里不描述错误形态：本线实测过反效果——把"依据不许写否定句"讲得越细模型越照着写。

    正面说"每条都要判"，错误形态归机检（判据下沉次序 prompt < 机检）。
    """
    prompt = build_system_prompt()
    for wording in ("非U形", "无法判定", "未画出", "错 →", "不要出现在"):
        assert wording not in prompt


async def test_read_floorplan_features_happy_path() -> None:
    image = make_png()
    reader = StubVisionReader(
        '{"verdicts": ['
        '{"feature": "balcony_service", "holds": true, "evidence": "阳台端头画了洗衣机设备位"},'
        '{"feature": "kitchen_u_shape", "holds": false, "evidence": "厨房台面沿两面墙布置"}],'
        ' "observations": [{"subject": "厨房", "finding": "L 形开放式"}],'
        ' "unreadable": [{"subject": "尺寸", "reason": "无任何尺寸标注"}]}'
    )
    reading = await read_floorplan_features(image, "image/png", reader)
    assert reading.logical_model == "floorplan-parse.default"
    assert reading.features.layout_features == {"balcony_service": "阳台端头画了洗衣机设备位"}
    # 没成立的那条不下发，但留在判定里（为什么判不成立是下一轮的素材）
    assert [(v.feature, v.holds) for v in reading.verdicts] == [
        ("balcony_service", True),
        ("kitchen_u_shape", False),
    ]
    # 分区读的成本形态：勘测 1 + 房间 N + 判定 1
    assert reading.model_call_count == len(reading.survey.rooms) + 2 == 4
    assert len(reader.calls) == 4
    assert [item.room for item in reading.orientations] == ["阳台", "厨房"]
    assert [legend.room for legend in reading.room_legends] == ["阳台", "厨房"]
    # 逐块读到的图例与算好的朝向，真的进了判定那一步的提示
    assert LEGEND_TEXT in reader.verdict_prompt
    assert "阳台：窗开在bottom，朝南" in reader.verdict_prompt


async def test_all_verdicts_negative_is_a_normal_result() -> None:
    """四条全判不成立是正常结果不是失败——漏报只少触发规则，误报会让报告写错。"""
    reader = StubVisionReader(
        '{"verdicts": [{"feature": "west_facing", "holds": false, "evidence": "图上无朝向依据"}]}'
    )
    reading = await read_floorplan_features(make_png(), "image/png", reader)
    assert reading.features.layout_features == {}
    assert len(reading.verdicts) == 1


async def test_read_floorplan_features_raises_on_out_of_set_key() -> None:
    """模型判成立的是闭集外的名字时，整次解析失败——不静默剔除。"""
    reader = StubVisionReader(
        '{"verdicts": [{"feature": "kitchen_l_shape", "holds": true, "evidence": "厨房呈 L 形"}]}'
    )
    with pytest.raises(LayoutFeatureViolation) as excinfo:
        await read_floorplan_features(make_png(), "image/png", reader)
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
    image.write_bytes(make_png())
    _install_stub_client(
        monkeypatch,
        '{"verdicts": [{"feature": "kitchen_l_shape", "holds": true, "evidence": "厨房呈 L 形"}]}',
    )
    exit_code = floorplan_cli_main(["--image", str(image)])
    assert exit_code == 3
    assert "kitchen_l_shape" in capsys.readouterr().err


def test_cli_writes_archive_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "floorplan.png"
    image.write_bytes(make_png())
    out = tmp_path / "reading.json"
    _install_stub_client(
        monkeypatch,
        '{"verdicts": ['
        '{"feature": "balcony_service", "holds": true, "evidence": "阳台画了洗衣机设备位"},'
        '{"feature": "west_facing", "holds": false, "evidence": "图上无朝向依据"}]}',
    )
    exit_code = floorplan_cli_main(["--image", str(image), "-o", str(out)])
    assert exit_code == 0
    archived = json.loads(out.read_text(encoding="utf-8"))
    assert archived["product"]["layoutFeatures"] == {"balcony_service": "阳台画了洗衣机设备位"}
    assert archived["logicalModel"] == "floorplan-parse.default"
    assert archived["rawOutput"]  # 模型原文随档，判定可逐字复核
    # 判成立与判不成立的都随档：投影之后没成立的那几条就看不见了
    assert [(v["feature"], v["holds"]) for v in archived["verdicts"]] == [
        ("balcony_service", True),
        ("west_facing", False),
    ]
    assert archived["image"]["sha256"]
    assert "balcony_service" in capsys.readouterr().out


def test_cli_rejects_unknown_image_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "floorplan.pdf"
    image.write_bytes(b"%PDF-fake")
    _install_stub_client(monkeypatch, '{"verdicts": []}')
    assert floorplan_cli_main(["--image", str(image)]) == 2
    assert "不认识的图片格式" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 朝向换算：纯算术，同样的输入必然同样的答案（方位不由 LLM 决定）
# ---------------------------------------------------------------------------


def test_cardinal_follows_the_compass() -> None:
    """指北针指哪条边，哪条边就是北，其余三边跟着转。"""
    assert [to_cardinal(side, "top") for side in ("top", "right", "bottom", "left")] == [
        "北",
        "东",
        "南",
        "西",
    ]
    assert [to_cardinal(side, "right") for side in ("top", "right", "bottom", "left")] == [
        "西",
        "北",
        "东",
        "南",
    ]
    assert [to_cardinal(side, "bottom") for side in ("top", "right", "bottom", "left")] == [
        "南",
        "西",
        "北",
        "东",
    ]
    assert [to_cardinal(side, "left") for side in ("top", "right", "bottom", "left")] == [
        "东",
        "南",
        "西",
        "北",
    ]


def test_no_compass_falls_back_to_the_drafting_convention() -> None:
    """图上没有指北针才用「上北下南左西右东」；有指北针以指北针为准。"""
    assert DEFAULT_NORTH_POINTS_TO == "top"
    assert to_cardinal("bottom", None) == "南"
    assert to_cardinal("bottom", "bottom") == "北"  # 指北针与约定冲突时指北针赢


def test_orientation_follows_the_window_wall_not_the_room_position() -> None:
    """样本那张图的实况：主卧在图**左下角**，但飘窗画在**下侧墙**上 → 朝南，不是西南。

    模型自己推时按房间位置推，出过"西南"，那是错的；换算按窗所在墙面走。
    """
    orientations = to_room_orientations(
        "top",
        [
            RoomLegend(room="主卧", legend="底部墙上标飘窗", window_walls=["bottom"]),
            RoomLegend(room="卫生间", legend="左侧是很粗的黑实线墙，无开口", window_walls=[]),
        ],
    )
    assert orientations[0].room == "主卧"
    assert orientations[0].facings == ["南"]
    assert orientations[1].facings == []  # 没画窗＝没朝向，是事实不猜


def test_orientation_is_reproducible() -> None:
    """换算是算术，跑几次都一样——这条就是"主卧朝向出过三个答案"那件事的处置。"""
    legends = [RoomLegend(room="主卧", legend="底部墙上标飘窗", window_walls=["bottom"])]
    assert {tuple(to_room_orientations("top", legends)[0].facings) for _ in range(5)} == {("南",)}


# ---------------------------------------------------------------------------
# 分区裁剪：剪刀在代码手里
# ---------------------------------------------------------------------------


def test_crop_enlarges_the_region() -> None:
    """裁出来的块要比原图上那一小块**大**——"看得更大"就是分区读的全部作用。"""
    import io

    from PIL import Image

    image_bytes = make_png(400, 600)
    region = RoomRegion(name="阳台", box=(0.40, 0.55, 0.66, 0.66))
    crop = crop_room(image_bytes, region)
    with Image.open(io.BytesIO(crop)) as cropped:
        # 原图上这块约 104×66 像素；放大后长边不低于阈值
        assert max(cropped.size) >= 1024


def test_crop_rejects_a_broken_box() -> None:
    image_bytes = make_png()
    for box in ((0.6, 0.1, 0.2, 0.9), (-0.1, 0.1, 0.5, 0.9), (0.1, 0.1, 1.5, 0.9)):
        with pytest.raises(RoomCropError) as excinfo:
            crop_room(image_bytes, RoomRegion(name="阳台", box=box))
        assert "阳台" in "；".join(excinfo.value.details)


def test_crop_rejects_a_box_too_small_to_be_a_room() -> None:
    with pytest.raises(RoomCropError) as excinfo:
        crop_room(make_png(), RoomRegion(name="阳台", box=(0.5, 0.5, 0.51, 0.51)))
    assert "小到不可能是个房间" in "；".join(excinfo.value.details)


async def test_每个房间各读一次() -> None:
    reader = StubVisionReader("{}")
    legends = await read_room_legends(
        make_png(),
        [
            RoomRegion(name="阳台", box=(0.40, 0.55, 0.66, 0.66)),
            RoomRegion(name="厨房", box=(0.45, 0.18, 0.68, 0.34)),
        ],
        reader,
        "floorplan-parse.default",
    )
    assert [legend.room for legend in legends] == ["阳台", "厨房"]
    assert legends[0].window_walls == ["bottom"]  # 窗墙在近景里定，不在整图勘测里定
    assert len(reader.calls) == 2
    # 送出去的是裁剪放大后的块，不是原图
    assert all(call["image_media_type"] == "image/png" for call in reader.calls)
    assert len({call["image_bytes"] for call in reader.calls}) == 2


# ---------------------------------------------------------------------------
# 勘测输出解析
# ---------------------------------------------------------------------------


def test_survey_parses_compass_and_rooms() -> None:
    survey = parse_survey_output(SURVEY_STUB)
    assert survey.north_points_to == "top"
    assert [room.name for room in survey.rooms] == ["阳台", "厨房"]


def test_survey_without_compass_is_a_fact_not_a_failure() -> None:
    survey = parse_survey_output(
        '{"northPointsTo": null, "rooms": [{"name": "阳台", "box": [0.4, 0.5, 0.6, 0.7]}]}'
    )
    assert survey.north_points_to is None  # 没有指北针是事实不是缺失，换算退到通行约定


def test_survey_with_no_rooms_fails_loud() -> None:
    """读不出房间划分就不往下走——裁剪没有区域可裁，硬走等于把整图又读一遍。"""
    with pytest.raises(FloorplanSurveyError):
        parse_survey_output('{"northPointsTo": "top", "rooms": []}')


def test_survey_rejects_an_unknown_compass_direction() -> None:
    with pytest.raises(FloorplanSurveyError):
        parse_survey_output('{"northPointsTo": "northeast", "rooms": []}')


def test_room_legend_parses_legend_and_window_walls() -> None:
    legend = parse_room_legend("阳台", LEGEND_STUB)
    assert legend.room == "阳台"
    assert legend.window_walls == ["bottom"]
    assert LEGEND_TEXT in legend.legend


def test_room_legend_without_window_is_a_fact() -> None:
    """近景说这一块没画窗，就是没画窗——不许因为"房间应该有窗"补一个。

    立案证据：整图勘测给没有窗的卫生间报过一面西窗，换算成"卫生间朝西"后催出一次误报。
    """
    legend = parse_room_legend("卫生间", '{"legend": "左侧是很粗的黑实线墙", "windowWalls": []}')
    assert legend.window_walls == []
    assert to_room_orientations("top", [legend])[0].facings == []


def test_room_legend_rejects_a_broken_output() -> None:
    with pytest.raises(RoomLegendError):
        parse_room_legend("阳台", "这一块我看不清")
    with pytest.raises(RoomLegendError):
        parse_room_legend("阳台", '{"legend": "有虚线框", "windowWalls": ["northeast"]}')
