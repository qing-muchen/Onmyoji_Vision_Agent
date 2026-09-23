"""转场证据迁移测试（Phase 4.6.4）：Transition → Evidence → State。

覆盖：

1. transition detector 的登记与工厂装配；
2. 未接线（无判定器）→ 未检出 + ``metadata`` 记录数据缺失（不抛异常、不伪造命中）；
3. 禁用开关 → 未检出（沿用既有 detector 契约的 reason 文案）；
4. 合成数据的证据契约（confidence / source / reliability / group / metadata）；
5. **真实数据集**端到端：``dataset/states/transition_bottom`` 正样本全部命中、
   ``non_transition_bottom`` 负样本全部被否决（数据集不存在时自动跳过）；
6. 证据 → 状态：``transition_banner`` → ``StateName.TRANSITION``（reason / evidence / source），
   且不影响其它状态的判定、同分时保持"转场优先"；
7. 旧帧路径仍可用（``source=transition_bottom_match``）；
8. 硬编码检查：``classifier.py`` 不含 ``transition_banner`` 字面量（映射来自配置）。
"""

from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import EvidenceConfig, EvidenceDetectorConfig, StateTransitionConfig
from src.vision.capture import Frame
from src.vision.evidence import (
    DETECTOR_CLASSES,
    EvidenceResult,
    EvidenceSource,
    EvidenceType,
    TransitionBannerDetector,
    create_detectors,
)
from src.vision.state import StateClassifier, StateName, StateSource
from src.vision.state.transition.classifier import TransitionClassifier
from src.vision.state.transition.dataset import SampleKind, TransitionSample

#: 转场数据集（Phase 3.0 建立；Phase 4.6.4 由证据层复用）
TRANSITION_DIR = Path("dataset/states/transition_bottom")
#: 非转场负样本目录（完整界面截图）
NON_TRANSITION_DIR = Path("dataset/states/non_transition_bottom")
#: 合成条带尺寸（与真实条带比例一致）
STRIP_HEIGHT = 58
STRIP_WIDTH = 1706


def load_image(path: Path) -> np.ndarray:
    """读取图像（兼容非 ASCII 路径）。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def frames(directory: Path) -> list[Frame]:
    """把目录下的 PNG 读成帧（按文件名排序）。"""
    return [
        Frame.create(load_image(path), source=path.name)
        for path in sorted(directory.glob("*.png"))
    ]


def strip_image(value: int = 0) -> np.ndarray:
    """合成条带（宽高比 29:1，判定器会把它当作"已经是条带"处理）。"""
    return np.full((STRIP_HEIGHT, STRIP_WIDTH, 3), value, dtype=np.uint8)


def frame_of(image: np.ndarray) -> Frame:
    """把图像包装为帧（状态/证据判定都不需要坐标映射）。"""
    return Frame.create(image, source="test")


def synthetic_probe(image: np.ndarray | None = None) -> TransitionClassifier:
    """用合成条带作为唯一正样本构造判定器（不依赖真实数据集）。"""
    sample = TransitionSample(
        sample_id="transition_bottom_001",
        kind=SampleKind.POSITIVE,
        path=Path("transition_bottom_001.png"),
        image=strip_image(0) if image is None else image,
    )
    return TransitionClassifier(StateTransitionConfig(), [sample], [])


def detector(probe: object | None = None) -> TransitionBannerDetector:
    """按证据配置构造 transition detector（判定器由调用方注入）。"""
    return TransitionBannerDetector.from_config(EvidenceConfig(), probe=probe)  # type: ignore[arg-type]


def evidence(name: str, confidence: float = 0.9) -> EvidenceResult:
    """构造一条命中证据。"""
    return EvidenceResult(name=name, detected=True, confidence=confidence)


def classifier() -> StateClassifier:
    """纯证据驱动分类器（规则来自 evidence.aggregator.rules）。"""
    return StateClassifier.from_evidence_config(EvidenceConfig())


@pytest.fixture(scope="module")
def real_probe() -> TransitionClassifier:
    """按配置装配真实转场判定器（数据集缺失时跳过依赖它的用例）。"""
    if not TRANSITION_DIR.is_dir():
        pytest.skip("transition 数据集不存在")
    return TransitionClassifier.from_dataset(StateTransitionConfig())


# ---------------------------------------------------------------------------
# 1/2/3. 登记、装配与数据缺失行为
# ---------------------------------------------------------------------------


def test_transition_detector_is_registered() -> None:
    """transition detector 登记在 DETECTOR_CLASSES，并被工厂装配。"""
    assert "transition_banner" in DETECTOR_CLASSES
    assert DETECTOR_CLASSES["transition_banner"] is TransitionBannerDetector
    assert TransitionBannerDetector.NAME == "transition_banner"
    assert TransitionBannerDetector.EVIDENCE_TYPE is EvidenceType.TRANSITION

    names = [item.name for item in create_detectors(EvidenceConfig())]
    assert "transition_banner" in names


def test_factory_builds_without_probe() -> None:
    """未注入判定器时工厂仍能装配（detector 只输出未检出，不抛异常）。"""
    detectors = {item.name: item for item in create_detectors(EvidenceConfig())}
    transition = detectors["transition_banner"]

    assert isinstance(transition, TransitionBannerDetector)
    assert transition.probe is None


def test_detector_without_probe_reports_missing_dataset() -> None:
    """没有可用判定器（数据缺失/未接线）时输出未检出，并记录原因与 missing_dataset。"""
    result = detector().detect(frame_of(strip_image(0)))

    assert isinstance(result, EvidenceResult)
    assert result.detected is False
    assert result.confidence == pytest.approx(0.0)
    assert result.metadata["reason"]
    assert result.metadata["missing_dataset"] is True
    assert result.evidence_type is EvidenceType.TRANSITION


def test_disabled_detector_returns_miss() -> None:
    """禁用开关生效（reason 文案与其它 detector 契约一致）。"""
    instance = TransitionBannerDetector(
        "transition_banner", EvidenceDetectorConfig(enabled=False), probe=synthetic_probe()
    )

    result = instance.detect(frame_of(strip_image(0)))

    assert result.detected is False
    assert result.metadata["reason"] == "detector disabled"


# ---------------------------------------------------------------------------
# 4. 证据契约（合成数据）
# ---------------------------------------------------------------------------


def test_detector_reports_strip_evidence() -> None:
    """条带自匹配时命中，并带上可靠的定位/打分信息。"""
    instance = detector(synthetic_probe())
    result = instance.detect(frame_of(strip_image(0)))

    assert result.detected is True
    assert result.confidence == pytest.approx(1.0)
    assert result.name == "transition_banner"
    assert result.source is EvidenceSource.ROI
    assert result.evidence_type is EvidenceType.TRANSITION
    assert result.group == ""
    assert result.reliability == pytest.approx(
        EvidenceConfig().reliability_for("transition_banner")
    )
    assert result.metadata["best_positive_file"] == "transition_bottom_001.png"
    assert result.metadata["threshold"] == pytest.approx(0.85)


def test_detector_reports_miss_with_diagnostics() -> None:
    """不相似的画面必须给出未检出，并带上打分细节便于排查。"""
    instance = detector(synthetic_probe())
    result = instance.detect(frame_of(np.full((961, 1706, 3), 235, dtype=np.uint8)))

    assert result.detected is False
    assert result.metadata["reason"] == "transition banner not matched"
    assert "positive_score" in result.metadata
    assert "negative_score" in result.metadata


# ---------------------------------------------------------------------------
# 5. 真实数据集
# ---------------------------------------------------------------------------


def test_real_dataset_hits_all_positives(real_probe: TransitionClassifier) -> None:
    """真实转场条带正样本全部命中（数据可用性的直接证据）。"""
    instance = detector(real_probe)
    positive_frames = frames(TRANSITION_DIR)

    assert len(positive_frames) >= 10
    assert all(instance.detect(frame).detected for frame in positive_frames)


def test_real_dataset_rejects_all_negatives(real_probe: TransitionClassifier) -> None:
    """真实非转场界面（完整截图）全部被否决。"""
    instance = detector(real_probe)
    negative_frames = frames(NON_TRANSITION_DIR)

    assert negative_frames
    assert not any(instance.detect(frame).detected for frame in negative_frames)


def test_real_frame_flows_to_transition_state(real_probe: TransitionClassifier) -> None:
    """真实条带 → 证据 → 状态：端到端得到 transition（含 reason / evidence）。"""
    frame = frames(TRANSITION_DIR)[2]
    evidence_result = detector(real_probe).detect(frame)

    result = classifier().classify([evidence_result])

    assert result.state_name is StateName.TRANSITION
    assert result.source is StateSource.EVIDENCE
    assert result.confidence == pytest.approx(evidence_result.confidence)
    assert "matched transition evidence rule" in result.reason
    assert result.evidence == ("transition_banner",)


# ---------------------------------------------------------------------------
# 6. 证据 → 状态
# ---------------------------------------------------------------------------


def test_evidence_maps_to_transition_state() -> None:
    """transition_banner 命中即判定为 transition（映射来自配置规则）。"""
    result = classifier().classify([evidence("transition_banner")])

    assert result.state_name is StateName.TRANSITION
    assert result.source is StateSource.EVIDENCE
    assert result.reason
    assert "matched transition evidence rule" in result.reason
    assert result.evidence == ("transition_banner",)


def test_transition_order_independent() -> None:
    """证据顺序不影响结果。"""
    items = [evidence("transition_banner"), evidence("bottom_menu_yard")]

    forward = classifier().classify(items)
    backward = classifier().classify(list(reversed(items)))

    assert forward == backward
    assert forward.state_name is StateName.TRANSITION


def test_transition_wins_tie_by_rule_order() -> None:
    """同分时按规则声明顺序择优：transition 规则在首位（保持 Phase 3.0 的"转场优先"）。"""
    result = classifier().classify(
        [evidence("bottom_menu_yard"), evidence("transition_banner")]
    )

    assert result.state_name is StateName.TRANSITION


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([evidence("bottom_menu_yard")], StateName.YARD),
        (
            [evidence("bottom_menu_exploration"), evidence("chapter_panel")],
            StateName.EXPLORATION,
        ),
        ([evidence("battle_auto_button"), evidence("battle_skill_area")], StateName.BATTLE),
        ([evidence("victory_icon")], StateName.BATTLE_RESULT),
        ([evidence("reward_panel")], StateName.BATTLE_REWARD),
    ],
)
def test_other_states_are_unaffected(items: list[EvidenceResult], expected: StateName) -> None:
    """加入转场证据后，其它状态的判定不受影响。"""
    result = classifier().classify(items)

    assert result.state_name is expected
    assert "transition_banner" not in result.evidence


# ---------------------------------------------------------------------------
# 7. 旧帧路径
# ---------------------------------------------------------------------------


def test_frame_path_keeps_legacy_source() -> None:
    """旧帧路径仍产出 transition，且 source 保持 transition_bottom_match。"""
    instance = StateClassifier(transition=synthetic_probe(), yard=None)

    result = instance.classify(frame_of(strip_image(0)))

    assert result.state_name is StateName.TRANSITION
    assert result.source is StateSource.TRANSITION_BOTTOM_MATCH
    assert result.detail


# ---------------------------------------------------------------------------
# 8. 禁止硬编码
# ---------------------------------------------------------------------------


def test_classifier_has_no_hardcoded_transition_evidence() -> None:
    """classifier 源码不得出现证据名：映射必须来自 evidence.aggregator.rules。"""
    source = Path(inspect.getsourcefile(StateClassifier) or "").read_text(encoding="utf-8")

    assert "transition_banner" not in source


def test_rule_is_declared_in_config() -> None:
    """转场规则与可靠度必须声明在配置里（而不是代码里）。"""
    config = EvidenceConfig()
    rule = next(item for item in config.aggregator.rules if item.state == "transition")

    assert rule.any_of == ["transition_banner"]
    assert config.reliability_for("transition_banner") > 0.0
    assert config.detectors["transition_banner"].enabled is True
