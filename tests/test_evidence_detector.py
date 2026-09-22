"""证据层（Phase 4.1 / 4.2）单元测试。

覆盖：

1. :class:`EvidenceResult` 创建与"不绑定状态"的约束；
2. ``confidence`` 边界与 ``detected`` 的一致性；
3. :class:`BaseEvidenceDetector` 基类接口（抽象、别名、detect 契约）；
4. :class:`EvidenceRegistry` 注册/重名/启停/批量检测；
5. Fake Frame 输入（不依赖真实截图与真实数据集）；
6. 每个已实现的 detector 都返回合法 :class:`EvidenceResult`；
7. :class:`EvidenceAggregator` 的规则聚合（battle / yard / exploration / 无证据 → unknown）；
8. **Phase 4.2**：battle 证据链（新增 detector 注册、无模板契约、配置形态、``requires + any_of`` 规则）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from src.core.config import (
    EvidenceAggregatorConfig,
    EvidenceConfig,
    EvidenceDetectorConfig,
    EvidenceRuleConfig,
)
from src.core.exceptions import EvidenceError, EvidenceTemplateLoadError
from src.core.geometry import Rect
from src.vision.capture.models import Frame
from src.vision.evidence import (
    DETECTOR_CLASSES,
    BaseDetector,
    BaseEvidenceDetector,
    EvidenceAggregator,
    EvidenceRegistry,
    EvidenceResult,
    EvidenceScore,
    EvidenceSource,
    EvidenceType,
    Roi,
    StateCandidate,
    TemplateEvidenceDetector,
    create_detectors,
    load_feature_templates,
)
from tests.helpers import build_scene_image

#: 测试帧尺寸 ``(width, height)``
FRAME_SIZE: tuple[int, int] = (400, 300)


def write_png(path: Path, image: np.ndarray) -> Path:
    """写出 PNG（路径可含中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", image)
    assert success
    buffer.tofile(str(path))
    return path


def make_frame(seed: int = 1, size: tuple[int, int] = FRAME_SIZE) -> Frame:
    """构造一张有结构的合成帧（Fake Frame：无 coordinate_transform）。"""
    return Frame.create(build_scene_image(size=size, seed=seed), source="test")


def make_templates(tmp_path: Path, detector: str, count: int = 2) -> Path:
    """在临时目录下造 ``<detector>_<source>_<NNN>.png`` 模板文件。"""
    directory = tmp_path / detector / "positive"
    for index in range(1, count + 1):
        write_png(
            directory / f"{detector}_cap_{index:03d}.png",
            build_scene_image(size=(60, 40), seed=index),
        )
    return directory


class FakeDetector(BaseEvidenceDetector):
    """固定返回给定证据的测试 detector。"""

    def __init__(self, name: str, result: EvidenceResult, *, enabled: bool = True) -> None:
        self._name = name
        self._result = result
        self._enabled = enabled

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    def detect(self, frame: Frame) -> EvidenceResult:
        return self._result


def hit(name: str, confidence: float, *, evidence_type: EvidenceType = EvidenceType.CUSTOM) -> EvidenceResult:
    """构造一条命中证据。"""
    return EvidenceResult(
        name=name,
        detected=True,
        confidence=confidence,
        bbox=Rect(x=1, y=2, width=3, height=4),
        source=EvidenceSource.TEMPLATE,
        evidence_type=evidence_type,
    )


# --------------------------------------------------------------------------
# 1~2. EvidenceResult / EvidenceScore 模型
# --------------------------------------------------------------------------


class TestEvidenceResult:
    """EvidenceResult 的字段约束与"证据-状态解耦"。"""

    def test_create_hit(self) -> None:
        result = hit("battle_auto_button", 0.95)
        assert result.detected is True
        assert result.is_hit is True
        assert result.bbox == Rect(x=1, y=2, width=3, height=4)
        assert "confidence=0.9500" in result.describe()

    def test_create_miss(self) -> None:
        result = EvidenceResult.miss("victory_icon", metadata={"reason": "no template"})
        assert result.detected is False
        assert result.confidence == 0.0
        assert result.metadata["reason"] == "no template"
        assert "miss" in result.describe()

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0, -1.0])
    def test_confidence_out_of_range(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            EvidenceResult(name="x", detected=True, confidence=confidence)

    def test_detected_requires_positive_confidence(self) -> None:
        with pytest.raises(ValueError, match="detected=True"):
            EvidenceResult(name="x", detected=True, confidence=0.0)

    def test_miss_requires_zero_confidence(self) -> None:
        with pytest.raises(ValueError, match="detected=False"):
            EvidenceResult(name="x", detected=False, confidence=0.5)

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            EvidenceResult(name="   ", detected=True, confidence=0.5)

    def test_evidence_cannot_carry_state(self) -> None:
        """证据不允许绑定状态：不存在的字段必须直接报错，而不是被悄悄接收。"""
        with pytest.raises(TypeError):
            EvidenceResult(name="x", detected=True, confidence=0.9, state="battle")  # type: ignore[call-arg]

    def test_default_type_is_custom(self) -> None:
        assert hit("x", 0.5).evidence_type is EvidenceType.CUSTOM

    def test_metadata_excluded_from_equality(self) -> None:
        """metadata 不影响相等比较（便于断言结果本身）。"""
        assert hit("x", 0.5) == hit("x", 0.5)
        assert hit("x", 0.5) != hit("x", 0.6)

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_evidence_score_range(self, score: float) -> None:
        with pytest.raises(ValueError, match="score"):
            EvidenceScore(state="battle", score=score)

    def test_state_candidate_defaults(self) -> None:
        candidate = StateCandidate(state="unknown", confidence=0.0)
        assert candidate.is_known is False
        assert StateCandidate(state="yard", confidence=0.9).is_known is True


# --------------------------------------------------------------------------
# 3. 基类接口
# --------------------------------------------------------------------------


class TestBaseDetectorInterface:
    """基类契约：抽象、别名、ROI 刻度与模板加载。"""

    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseEvidenceDetector()  # type: ignore[abstract]

    def test_missing_detect_is_abstract(self) -> None:
        class OnlyName(BaseEvidenceDetector):
            @property
            def name(self) -> str:
                return "only_name"

        with pytest.raises(TypeError):
            OnlyName()  # type: ignore[abstract]

    def test_base_detector_is_alias(self) -> None:
        assert BaseDetector is BaseEvidenceDetector

    def test_fake_detector_can_be_used(self) -> None:
        detector = FakeDetector("probe", hit("probe", 0.8))
        assert detector.detect(make_frame()).name == "probe"

    @pytest.mark.parametrize(
        ("roi", "expected"),
        [
            ((0.0, 0.8, 1.0, 1.0), Rect(x=0, y=240, width=400, height=60)),
            ((0.5, 0.0, 1.0, 0.5), Rect(x=200, y=0, width=200, height=150)),
        ],
    )
    def test_roi_to_rect(self, roi: tuple[float, float, float, float], expected: Rect) -> None:
        assert Roi.from_sequence(roi).to_rect(*FRAME_SIZE) == expected

    @pytest.mark.parametrize("roi", [(0.5, 0.0, 0.4, 0.5), (0.0, 0.9, 1.0, 0.5), (-0.1, 0.0, 1.0, 1.0)])
    def test_roi_rejects_invalid_range(self, roi: tuple[float, float, float, float]) -> None:
        with pytest.raises(ValueError, match="ROI"):
            Roi.from_sequence(roi)

    def test_crop_roi_uses_frame_pixels(self, tmp_path: Path) -> None:
        detector = TemplateEvidenceDetector(
            "probe",
            EvidenceDetectorConfig(roi=(0.0, 0.8, 1.0, 1.0)),
        )
        cropped, rect = detector.crop_roi(make_frame())
        assert rect == Rect(x=0, y=240, width=400, height=60)
        assert cropped.shape[:2] == (60, 400)

    def test_load_feature_templates(self, tmp_path: Path) -> None:
        directory = make_templates(tmp_path, "probe", count=3)
        templates = load_feature_templates(directory, "probe", limit=10, root=tmp_path)
        assert [item.template_id for item in templates] == [
            "probe_cap_001",
            "probe_cap_002",
            "probe_cap_003",
        ]

    def test_load_feature_templates_respects_limit(self, tmp_path: Path) -> None:
        directory = make_templates(tmp_path, "probe", count=6)
        assert len(load_feature_templates(directory, "probe", limit=2, root=tmp_path)) == 2

    def test_load_feature_templates_missing_dir(self, tmp_path: Path) -> None:
        assert load_feature_templates(tmp_path / "nope", "probe", limit=3, root=tmp_path) == []

    def test_load_feature_templates_wrong_prefix(self, tmp_path: Path) -> None:
        directory = make_templates(tmp_path, "probe", count=1)
        with pytest.raises(EvidenceTemplateLoadError):
            load_feature_templates(directory, "other", limit=3, root=tmp_path)

    def test_from_config_requires_name(self) -> None:
        class NoName(TemplateEvidenceDetector):
            pass

        with pytest.raises(EvidenceError, match="NAME"):
            NoName.from_config(EvidenceConfig())


# --------------------------------------------------------------------------
# 4~5. Registry（Fake Frame 输入）
# --------------------------------------------------------------------------


class TestEvidenceRegistry:
    """注册表的注册、重名保护与批量检测。"""

    def test_register_and_detect_all(self) -> None:
        registry = EvidenceRegistry()
        registry.register(FakeDetector("a", hit("a", 0.9)))
        registry.register(FakeDetector("b", EvidenceResult.miss("b")))
        results = registry.detect_all(make_frame())
        assert [item.name for item in results] == ["a", "b"]
        assert sum(1 for item in results if item.detected) == 1

    def test_duplicate_name_rejected(self) -> None:
        registry = EvidenceRegistry([FakeDetector("a", hit("a", 0.9))])
        with pytest.raises(EvidenceError, match="重复"):
            registry.register(FakeDetector("a", hit("a", 0.5)))

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(EvidenceError, match="name"):
            EvidenceRegistry([FakeDetector("  ", hit("x", 0.5))])

    def test_get_and_unregister(self) -> None:
        registry = EvidenceRegistry([FakeDetector("a", hit("a", 0.9))])
        assert registry.get("a").name == "a"
        registry.unregister("a")
        assert len(registry) == 0
        with pytest.raises(KeyError):
            registry.get("a")
        assert registry.find("a") is None

    def test_disabled_detector_skipped(self) -> None:
        registry = EvidenceRegistry(
            [
                FakeDetector("on", hit("on", 0.9)),
                FakeDetector("off", hit("off", 0.9), enabled=False),
            ]
        )
        assert registry.names == ("on", "off")
        assert [item.name for item in registry.detect_all(make_frame())] == ["on"]
        assert len(registry.enabled_detectors) == 1

    def test_detect_map(self) -> None:
        registry = EvidenceRegistry([FakeDetector("a", hit("a", 0.9))])
        mapped = registry.detect_map(make_frame())
        assert mapped["a"].confidence == pytest.approx(0.9)

    def test_create_detectors_from_config(self) -> None:
        detectors = create_detectors(EvidenceConfig())
        assert [item.name for item in detectors] == list(DETECTOR_CLASSES)
        assert "bottom_menu_yard" in EvidenceRegistry(detectors)


# --------------------------------------------------------------------------
# 6. 每个 detector 的返回契约
# --------------------------------------------------------------------------


class TestDetectorContract:
    """所有已实现 detector 在任何输入下都返回合法证据，且不抛异常。"""

    @pytest.mark.parametrize("name", sorted(DETECTOR_CLASSES))
    def test_returns_valid_evidence(self, name: str) -> None:
        detector = DETECTOR_CLASSES[name](name, EvidenceDetectorConfig())
        result = detector.detect(make_frame())
        assert isinstance(result, EvidenceResult)
        assert result.name == name
        assert 0.0 <= result.confidence <= 1.0
        assert result.detected is False  # 无模板时必须是否定结果，而不是瞎猜

    @pytest.mark.parametrize("name", sorted(DETECTOR_CLASSES))
    def test_disabled_detector_returns_miss(self, name: str) -> None:
        detector = DETECTOR_CLASSES[name](name, EvidenceDetectorConfig(enabled=False))
        result = detector.detect(make_frame())
        assert result.detected is False
        assert result.metadata["reason"] == "detector disabled"

    @pytest.mark.parametrize("name", sorted(DETECTOR_CLASSES))
    def test_evidence_type_is_declared(self, name: str) -> None:
        detector = DETECTOR_CLASSES[name](name, EvidenceDetectorConfig())
        assert detector.evidence_type is not EvidenceType.CUSTOM

    def test_template_hit_produces_evidence(self, tmp_path: Path) -> None:
        """模板来自同一张帧的 ROI 时应当命中（自匹配 1 次即足够）。"""
        frame = make_frame()
        roi_image = frame.image[240:300, 0:400]
        write_png(tmp_path / "probe" / "positive" / "probe_cap_001.png", roi_image)
        detector = TemplateEvidenceDetector(
            "probe",
            EvidenceDetectorConfig(roi=(0.0, 0.8, 1.0, 1.0), min_templates=1),
            tuple(load_feature_templates(tmp_path / "probe" / "positive", "probe", limit=1, root=tmp_path)),
        )
        result = detector.detect(frame)
        assert result.detected is True
        assert result.confidence > 0.9
        assert result.source is EvidenceSource.TEMPLATE
        assert result.bbox is not None

    def test_insufficient_templates_are_penalized(self, tmp_path: Path) -> None:
        """模板数低于门槛时必须折价并标记，而不是伪造高置信度。"""
        frame = make_frame()
        write_png(tmp_path / "probe" / "positive" / "probe_cap_001.png", frame.image[240:300, 0:400])
        detector = TemplateEvidenceDetector(
            "probe",
            EvidenceDetectorConfig(
                roi=(0.0, 0.8, 1.0, 1.0),
                min_templates=5,
                shortfall_factor=0.5,
            ),
            tuple(load_feature_templates(tmp_path / "probe" / "positive", "probe", limit=1, root=tmp_path)),
        )
        result = detector.detect(frame)
        assert result.detected is True
        assert result.metadata["insufficient_templates"] is True
        assert result.confidence == pytest.approx(result.metadata["raw_confidence"] * 0.5)


# --------------------------------------------------------------------------
# 7. Aggregator
# --------------------------------------------------------------------------


def rule(state: str, requires: list[str], min_confidence: float = 0.6) -> EvidenceRuleConfig:
    """构造一条测试规则。"""
    return EvidenceRuleConfig(state=state, requires=requires, min_confidence=min_confidence)


def aggregator(*rules: EvidenceRuleConfig) -> EvidenceAggregator:
    """构造只含给定规则的聚合器。"""
    return EvidenceAggregator(EvidenceAggregatorConfig(rules=list(rules)))


class TestEvidenceAggregator:
    """证据 -> 候选状态的规则聚合。"""

    def test_battle_evidence_maps_to_battle(self) -> None:
        result = aggregator(rule("battle", ["battle_auto_button"])).aggregate(
            [hit("battle_auto_button", 0.95)]
        )
        assert result.state == "battle"
        assert result.confidence == pytest.approx(0.95)
        assert "battle_auto_button=0.9500" in result.reason

    def test_yard_evidence_maps_to_yard(self) -> None:
        result = aggregator(rule("yard", ["bottom_menu_yard"])).aggregate(
            [hit("bottom_menu_yard", 0.9)]
        )
        assert result.state == "yard"

    def test_exploration_requires_both_evidences(self) -> None:
        agg = aggregator(rule("exploration", ["bottom_menu_exploration", "chapter_panel"]))
        both = agg.aggregate([hit("bottom_menu_exploration", 0.9), hit("chapter_panel", 0.8)])
        assert both.state == "exploration"
        assert both.confidence == pytest.approx(0.85)

        only_one = agg.aggregate([hit("bottom_menu_exploration", 0.9)])
        assert only_one.state == "unknown"
        assert "chapter_panel" in only_one.reason

    def test_no_evidence_maps_to_unknown(self) -> None:
        result = aggregator(rule("battle", ["battle_auto_button"])).aggregate([])
        assert result.state == "unknown"
        assert result.confidence == pytest.approx(0.0)

    def test_low_confidence_evidence_is_ignored(self) -> None:
        agg = aggregator(rule("battle", ["battle_auto_button"], min_confidence=0.9))
        assert agg.aggregate([hit("battle_auto_button", 0.5)]).state == "unknown"
        assert agg.aggregate([hit("battle_auto_button", 0.95)]).state == "battle"

    def test_scoring_prefers_higher_confidence(self) -> None:
        agg = aggregator(
            rule("battle", ["battle_auto_button"]),
            rule("yard", ["bottom_menu_yard"]),
        )
        result = agg.aggregate([hit("battle_auto_button", 0.7), hit("bottom_menu_yard", 0.99)])
        assert result.state == "yard"

    def test_score_lists_satisfied_first(self) -> None:
        agg = aggregator(
            rule("battle", ["battle_auto_button"]),
            rule("yard", ["bottom_menu_yard"]),
        )
        scores = agg.score([hit("bottom_menu_yard", 0.9)])
        assert [item.state for item in scores] == ["yard", "battle"]
        assert scores[0].satisfied is True
        assert scores[1].missing == ("battle_auto_button",)

    def test_disabled_aggregator_falls_back(self) -> None:
        config = EvidenceAggregatorConfig(enabled=False, rules=[rule("battle", ["battle_auto_button"])])
        result = EvidenceAggregator(config).aggregate([hit("battle_auto_button", 0.9)])
        assert result.state == "unknown"
        assert "disabled" in result.reason

    def test_rule_requires_condition(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceRuleConfig(state="battle")

    def test_rule_needs_state(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceRuleConfig(state="  ", requires=["battle_auto_button"])


# --------------------------------------------------------------------------
# 8. Phase 4.2：battle 证据链
# --------------------------------------------------------------------------

#: battle 相关的全部证据名（与配置顺序一致）
BATTLE_EVIDENCE_NAMES: tuple[str, ...] = (
    "battle_auto_button",
    "battle_skill_area",
    "battle_action_bar",
    "battle_shikigami_icons",
)
#: Phase 4.2 新增的三个 detector
NEW_BATTLE_DETECTORS: tuple[str, ...] = (
    "battle_skill_area",
    "battle_action_bar",
    "battle_shikigami_icons",
)


class TestBattleEvidenceChain:
    """Phase 4.2：battle 证据链的注册、无模板契约、配置形态与聚合规则。"""

    def test_all_detectors_registered(self) -> None:
        assert set(DETECTOR_CLASSES) == {
            "bottom_menu_yard",
            "bottom_menu_exploration",
            "chapter_panel",
            *BATTLE_EVIDENCE_NAMES,
            "victory_icon",
            "reward_panel",
        }

    @pytest.mark.parametrize("name", NEW_BATTLE_DETECTORS)
    def test_new_detector_declares_battle_ui_type(self, name: str) -> None:
        assert DETECTOR_CLASSES[name].NAME == name
        assert DETECTOR_CLASSES[name].EVIDENCE_TYPE is EvidenceType.BATTLE_UI

    @pytest.mark.parametrize("name", NEW_BATTLE_DETECTORS)
    def test_no_template_returns_miss(self, name: str) -> None:
        """没有模板时必须是"未检出"并给出原因，而不是抛异常或假装命中。"""
        detector = DETECTOR_CLASSES[name](name, EvidenceDetectorConfig())
        result = detector.detect(make_frame())
        assert isinstance(result, EvidenceResult)
        assert result.name == name
        assert result.detected is False
        assert result.confidence == pytest.approx(0.0)
        assert result.bbox is None
        assert result.metadata["reason"]

    @pytest.mark.parametrize("name", NEW_BATTLE_DETECTORS)
    def test_new_detectors_are_configurable(self, name: str) -> None:
        config = EvidenceConfig()
        assert name in config.detectors
        expected = f"dataset/features/battle/{name.removeprefix('battle_')}/positive"
        assert config.detectors[name].templates == expected

    def test_config_roi_matches_dataset_layout(self) -> None:
        """ROI 必须与 dataset/README.md 的 detector 表一致（防止配置漂移）。"""
        detectors = EvidenceConfig().detectors
        assert detectors["battle_skill_area"].roi == (0.60, 0.74, 0.92, 1.00)
        assert detectors["battle_action_bar"].roi == (0.90, 0.04, 1.00, 0.72)
        assert detectors["battle_shikigami_icons"].roi == (0.925, 0.09, 1.00, 0.60)

    def test_default_battle_rule_shape(self) -> None:
        config = EvidenceConfig()
        battle = next(rule for rule in config.aggregator.rules if rule.state == "battle")
        assert battle.requires == ["battle_auto_button"]
        assert battle.any_of == list(NEW_BATTLE_DETECTORS)

    @staticmethod
    def battle_rule() -> EvidenceRuleConfig:
        """Phase 4.2 的 battle 规则：必需自动按钮 + 任一辅助证据。"""
        return EvidenceRuleConfig(
            state="battle",
            requires=["battle_auto_button"],
            any_of=list(NEW_BATTLE_DETECTORS),
        )

    @pytest.mark.parametrize("aux", NEW_BATTLE_DETECTORS)
    def test_auto_plus_one_aux_is_battle(self, aux: str) -> None:
        """自动按钮 + 任一辅助证据即可判定 battle（覆盖技能动画遮挡单个证据的场景）。"""
        result = aggregator(self.battle_rule()).aggregate(
            [hit("battle_auto_button", 0.95), hit(aux, 0.9)]
        )
        assert result.state == "battle"
        assert result.confidence == pytest.approx(0.925)
        assert len(result.evidence) == 2

    def test_auto_alone_is_not_enough(self) -> None:
        """只有自动按钮、没有任何辅助证据时不判定 battle（避免单证据误判）。"""
        result = aggregator(self.battle_rule()).aggregate([hit("battle_auto_button", 0.95)])
        assert result.state == "unknown"
        assert "battle_skill_area" in result.reason

    def test_aux_alone_is_not_enough(self) -> None:
        """辅助证据齐全但缺少必需的自动按钮 → 不判定 battle。"""
        result = aggregator(self.battle_rule()).aggregate(
            [hit(name, 0.95) for name in NEW_BATTLE_DETECTORS]
        )
        assert result.state == "unknown"
        assert "battle_auto_button" in result.reason

    def test_full_battle_evidence(self) -> None:
        result = aggregator(self.battle_rule()).aggregate(
            [hit(name, 0.9) for name in BATTLE_EVIDENCE_NAMES]
        )
        assert result.state == "battle"
        assert len(result.evidence) == 4

    def test_any_of_respects_min_confidence(self) -> None:
        """any_of 里的证据同样受 min_confidence 约束，弱证据不算命中。"""
        agg = aggregator(
            EvidenceRuleConfig(
                state="battle",
                requires=["battle_auto_button"],
                any_of=list(NEW_BATTLE_DETECTORS),
                min_confidence=0.9,
            )
        )
        assert agg.aggregate(
            [hit("battle_auto_button", 0.95), hit("battle_skill_area", 0.5)]
        ).state == "unknown"
        assert agg.aggregate(
            [hit("battle_auto_button", 0.95), hit("battle_skill_area", 0.95)]
        ).state == "battle"
