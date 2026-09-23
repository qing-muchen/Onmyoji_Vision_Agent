"""置信度契约测试（Phase 4.6.6）：证据得分 ≠ 状态可信度。

覆盖：

1. 两个概念分离：``victory_icon`` 证据得分 0.70 → 状态可信度 ≥ 0.85；
2. 校准后 ``battle_result`` / ``battle_reward`` 能通过稳定化门槛（连续 3 帧 → 稳定）；
3. 低质量证据（低于证据准入阈值 ``min_confidence``）不能进入状态；
4. 校准必须满足 ``min_reliability``（证据类不够可信时不抬高结论）；
5. 未配置校准的规则（battle）行为不变：状态可信度 = 证据得分；
6. 校准完全**配置驱动**（改配置即改结果；``classifier.py`` 里没有浮点阈值字面量）；
7. 真实截图端到端：``dataset/raw`` 的 battle_result / battle_reward → 状态 → 稳定确认。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import (
    EvidenceAggregatorConfig,
    EvidenceConfig,
    EvidenceRuleConfig,
    StateConfidenceConfig,
    load_config,
)
from src.core.paths import resolve_from_project_root
from src.runtime.state_monitor import StateStabilizer
from src.vision.capture import Frame
from src.vision.evidence import (
    EvidenceAggregator,
    EvidenceRegistry,
    EvidenceResult,
    StateCandidate,
    create_detectors,
)
from src.vision.evidence.aggregator import RULE_BASED_MODE
from src.vision.state import StateClassifier, StateName, StateSource

#: 真实样本（用于端到端验证；缺失时跳过）
REAL_SAMPLES: dict[str, str] = {
    "battle_result": "dataset/raw/battle_result/battle_result_win_001.png",
    "battle_reward": "dataset/raw/battle_reward/battle_reward_item_001.png",
}


def evidence(
    name: str, confidence: float = 0.9, reliability: float = 1.0, group: str = ""
) -> EvidenceResult:
    """构造一条命中证据。"""
    return EvidenceResult(
        name=name, detected=True, confidence=confidence, reliability=reliability, group=group
    )


def classifier(config: EvidenceConfig | None = None) -> StateClassifier:
    """纯证据驱动分类器。"""
    return StateClassifier.from_evidence_config(config if config is not None else EvidenceConfig())


def aggregate(
    items: list[EvidenceResult], config: EvidenceConfig | None = None
) -> StateCandidate:
    """直接调用聚合器（观察证据得分与状态可信度两个分量）。"""
    active = config if config is not None else EvidenceConfig()
    return EvidenceAggregator(active.aggregator).aggregate(items)


def stabilizer() -> StateStabilizer:
    """按仓库配置装配稳定器（门槛与帧数来自配置，不在测试里硬编码）。"""
    return StateStabilizer(load_config(environ={}).runtime.stabilization)


def load_image(path: Path) -> np.ndarray:
    """读取图像（兼容非 ASCII 路径）。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# 1. 证据得分 ≠ 状态可信度
# ---------------------------------------------------------------------------


def test_evidence_score_and_state_confidence_differ() -> None:
    """单样本证据：证据得分 0.70，状态可信度 0.85（两个概念不等价）。"""
    candidate = aggregate([evidence("victory_icon", 0.7, 0.95)])

    assert candidate.state == "battle_result"
    assert candidate.confidence == pytest.approx(0.7)  # 证据得分
    assert candidate.state_confidence == pytest.approx(0.85)  # 状态可信度
    assert candidate.is_calibrated is True

    result = classifier().classify([evidence("victory_icon", 0.7, 0.95)])

    assert result.confidence == pytest.approx(0.85)
    assert result.confidence > candidate.confidence


def test_calibrated_state_confidence_passes_stabilization_threshold() -> None:
    """校准后的状态可信度必须达到稳定化门槛（否则状态永远无法确认）。"""
    threshold = load_config(environ={}).runtime.stabilization.confidence_threshold

    result = classifier().classify([evidence("victory_icon", 0.7, 0.95)])

    assert result.confidence >= threshold


# ---------------------------------------------------------------------------
# 2/3. 稳定化：battle_result / battle_reward 能确认
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([evidence("victory_icon", 0.7, 0.95)], StateName.BATTLE_RESULT),
        ([evidence("reward_panel", 0.7, 0.95)], StateName.BATTLE_REWARD),
    ],
)
def test_single_sample_states_reach_stable_state(
    items: list[EvidenceResult], expected: StateName
) -> None:
    """单样本状态连续 3 帧后成为稳定状态（Phase 4.6.6 要解决的问题）。"""
    instance = classifier()
    stab = stabilizer()

    results = [stab.update(instance.classify(items)) for _ in range(3)]

    assert results[-1].state_name is expected
    assert stab.stable_state is expected
    assert results[0].changed is False  # 首次确认建立基线，不算变化


def test_calibration_does_not_break_multi_evidence_states() -> None:
    """校准不改变其它状态的判定与可信度（battle 仍等于证据得分）。"""
    items = [evidence("battle_auto_button", 1.0, 0.8), evidence("battle_skill_area", 1.0, 0.8)]
    stab = stabilizer()
    instance = classifier()

    for _ in range(3):
        result = stab.update(instance.classify(items))

    assert result.state_name is StateName.BATTLE
    assert result.confidence == pytest.approx(1.0)
    assert stab.stable_state is StateName.BATTLE


# ---------------------------------------------------------------------------
# 4. 证据准入与可靠度门槛
# ---------------------------------------------------------------------------


def test_low_quality_evidence_is_rejected() -> None:
    """低于证据准入阈值（min_confidence）的证据不参与状态判断。"""
    result = classifier().classify([evidence("victory_icon", 0.5, 0.95)])

    assert result.state_name is StateName.UNKNOWN
    assert result.confidence == pytest.approx(0.0)
    assert result.evidence == ()


def test_calibration_respects_min_reliability() -> None:
    """平均可靠度低于 min_reliability 时不抬高状态可信度。"""
    calibrated = aggregate([evidence("victory_icon", 0.7, 0.95)])
    uncalibrated = aggregate([evidence("victory_icon", 0.7, 0.5)])

    assert calibrated.state_confidence == pytest.approx(0.85)
    assert uncalibrated.state_confidence == pytest.approx(0.7)
    assert uncalibrated.is_calibrated is False


# ---------------------------------------------------------------------------
# 5. 未校准规则保持既有行为
# ---------------------------------------------------------------------------


def test_unconfigured_rule_keeps_evidence_score() -> None:
    """未声明 state_confidence 的规则：状态可信度等于证据得分。"""
    candidate = aggregate(
        [evidence("battle_auto_button", 1.0, 0.8), evidence("battle_skill_area", 1.0, 0.8)]
    )

    assert candidate.state == "battle"
    assert candidate.state_confidence == candidate.confidence
    assert candidate.is_calibrated is False


def test_transition_and_yard_are_not_calibrated() -> None:
    """transition / yard 未声明校准：状态可信度 = 证据得分（1.0）。"""
    transition = classifier().classify([evidence("transition_banner", 1.0, 0.95)])
    yard = classifier().classify([evidence("bottom_menu_yard", 1.0, 0.9)])

    assert transition.confidence == pytest.approx(1.0)
    assert yard.confidence == pytest.approx(1.0)
    assert "calibrated state confidence" not in transition.reason
    assert "calibrated state confidence" not in yard.reason


# ---------------------------------------------------------------------------
# 6. 配置驱动
# ---------------------------------------------------------------------------


def test_calibration_is_config_driven() -> None:
    """改配置即改结果：校准下限完全来自 ``state_confidence``。"""
    config = EvidenceConfig(
        aggregator=EvidenceAggregatorConfig(
            rules=[
                EvidenceRuleConfig(
                    state="battle_result",
                    requires=["victory_icon"],
                    state_confidence=StateConfidenceConfig(
                        mode="rule_based", min=0.95, min_reliability=0.9
                    ),
                )
            ]
        )
    )

    candidate = aggregate([evidence("victory_icon", 0.7, 0.95)], config)

    assert candidate.state_confidence == pytest.approx(0.95)


def test_calibration_can_be_disabled() -> None:
    """不配置 ``state_confidence`` 时不做校准（状态可信度回到证据得分）。"""
    config = EvidenceConfig(
        aggregator=EvidenceAggregatorConfig(
            rules=[
                EvidenceRuleConfig(
                    state="battle_result",
                    requires=["victory_icon"],
                    state_confidence=None,
                )
            ]
        )
    )

    candidate = aggregate([evidence("victory_icon", 0.7, 0.95)], config)

    assert candidate.state_confidence == pytest.approx(0.7)


def test_repository_config_calibrates_single_sample_rules() -> None:
    """仓库配置：只有单样本规则（battle_result / battle_reward）声明校准，且能达到稳定门槛。"""
    config = load_config(environ={})
    threshold = config.runtime.stabilization.confidence_threshold
    calibrated = {
        rule.state: rule.state_confidence
        for rule in config.evidence.aggregator.rules
        if rule.state_confidence is not None
    }

    assert set(calibrated) == {"battle_result", "battle_reward"}
    for calibration in calibrated.values():
        assert calibration is not None
        assert calibration.mode == RULE_BASED_MODE
        assert calibration.min >= threshold
        assert calibration.min_reliability > 0.0


CLASSIFIER_SOURCE = Path(inspect.getsourcefile(StateClassifier) or "").read_text(encoding="utf-8")


def numeric_literals(source: str) -> list[float]:
    """收集源码中的数字字面量（用于确认没有硬编码阈值）。"""
    tree = ast.parse(source)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]


def test_classifier_has_no_hardcoded_confidence_thresholds() -> None:
    """状态置信度阈值必须来自配置：classifier 里不应出现浮点字面量（0.85 / 0.7 等）。"""
    floats = [value for value in numeric_literals(CLASSIFIER_SOURCE) if isinstance(value, float)]

    assert floats == [], f"classifier 里出现浮点字面量（疑似硬编码阈值）: {floats}"


# ---------------------------------------------------------------------------
# 7. reason 说明
# ---------------------------------------------------------------------------


def test_reason_explains_calibration() -> None:
    """reason 必须解释"为什么达到该状态"，并在校准时明确标注。"""
    result = classifier().classify([evidence("victory_icon", 0.7, 0.95)])

    assert "matched battle_result evidence rule" in result.reason
    assert "calibrated state confidence" in result.reason
    assert result.evidence == ("victory_icon",)
    assert result.source is StateSource.EVIDENCE


# ---------------------------------------------------------------------------
# 8. 真实截图端到端
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_registry() -> EvidenceRegistry:
    """真实 detector 装配（模板从 dataset/features 加载；缺失时跳过）。"""
    if not resolve_from_project_root("dataset/raw").is_dir():
        pytest.skip("dataset/raw 不存在")
    return EvidenceRegistry(create_detectors(EvidenceConfig()))


@pytest.mark.parametrize("state", ["battle_result", "battle_reward"])
def test_real_frames_reach_stable_state(real_registry: EvidenceRegistry, state: str) -> None:
    """真实截图 → 证据 → 状态（可信度 ≥ 门槛）→ 连续 3 帧稳定确认。"""
    path = resolve_from_project_root(REAL_SAMPLES[state])
    if not path.is_file():
        pytest.skip(f"{path} 不存在")

    frame = Frame.create(load_image(path), source=path.name)
    items = real_registry.detect_all(frame)
    instance = classifier()
    threshold = load_config(environ={}).runtime.stabilization.confidence_threshold

    result = instance.classify(items)
    assert result.state_name is StateName(state)
    assert result.confidence >= threshold
    assert result.evidence
    assert result.reason

    stab = stabilizer()
    for _ in range(3):
        stable = stab.update(instance.classify(items))

    assert stable.state_name is StateName(state)
    assert stab.stable_state is StateName(state)
