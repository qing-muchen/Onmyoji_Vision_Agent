"""StateClassifier 的证据驱动路径测试（Phase 4.6.3）：Evidence → State。

覆盖：

1. 各状态的规则判定（battle / battle_result / battle_reward / yard / exploration）；
2. 证据不足与空证据 → unknown（并可解释）；
3. 顺序无关、group 去重与计分沿用证据聚合器（不重复实现）；
4. 输入校验（非法元素 / 非序列输入 / 实例未配置 aggregator）；
5. 与帧驱动路径的兼容（同一入口按输入类型分派）；
6. **禁止硬编码**：classifier 源码中不存在证据名或状态名的字符串常量
   （"证据名 → 状态"的映射必须来自配置规则）。
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from src.core.config import EvidenceAggregatorConfig, EvidenceConfig, StateTransitionConfig
from src.vision.capture import Frame
from src.vision.evidence import EvidenceResult
from src.vision.state import StateClassifier, StateName, StateSource
from src.vision.state.transition.classifier import TransitionClassifier
from src.vision.state.transition.dataset import SampleKind, TransitionSample


def evidence(name: str, confidence: float = 0.9, group: str = "") -> EvidenceResult:
    """构造一条命中证据。

    ``group`` 在生产路径上由 detector 按 ``evidence.groups`` 注入（同一视觉区域的证据
    共享一个 group 名）；此处默认空字符串表示"未分组"，聚合器会以证据名各自成组。
    """
    return EvidenceResult(name=name, detected=True, confidence=confidence, group=group)


def classifier(config: EvidenceConfig | None = None) -> StateClassifier:
    """按证据配置构造纯证据驱动分类器（不加载任何图像数据集）。"""
    return StateClassifier.from_evidence_config(config if config is not None else EvidenceConfig())


def frame_only_classifier() -> StateClassifier:
    """只装配帧（转场）分支的分类器；正样本用内存图像，不依赖数据集。"""
    sample = TransitionSample(
        sample_id="transition_bottom_001",
        kind=SampleKind.POSITIVE,
        path=Path("transition_bottom_001.png"),
        image=np.zeros((58, 1706, 3), dtype=np.uint8),
    )
    return StateClassifier(
        transition=TransitionClassifier(StateTransitionConfig(), [sample], []),
        yard=None,
    )


# ---------------------------------------------------------------------------
# 1. 证据 → 状态（各状态规则）
# ---------------------------------------------------------------------------


def test_battle_from_two_visual_regions() -> None:
    """battle：两个不同视觉区域的证据同时命中。"""
    result = classifier().classify([evidence("battle_auto_button"), evidence("battle_skill_area")])

    assert result.state_name is StateName.BATTLE
    assert result.source is StateSource.EVIDENCE
    assert 0.0 < result.confidence <= 1.0
    assert result.reason
    assert "matched battle evidence rule" in result.reason
    assert result.evidence == ("battle_auto_button", "battle_skill_area")


def test_battle_result_from_victory_icon() -> None:
    """battle_result：胜利图标。"""
    result = classifier().classify([evidence("victory_icon")])

    assert result.state_name is StateName.BATTLE_RESULT
    assert result.source is StateSource.EVIDENCE
    assert result.evidence == ("victory_icon",)
    assert "matched battle_result evidence rule" in result.reason


def test_battle_reward_from_reward_panel() -> None:
    """battle_reward：奖励面板。"""
    result = classifier().classify([evidence("reward_panel")])

    assert result.state_name is StateName.BATTLE_REWARD
    assert result.evidence == ("reward_panel",)


def test_yard_from_bottom_menu() -> None:
    """yard：庭院型底部功能栏。"""
    result = classifier().classify([evidence("bottom_menu_yard")])

    assert result.state_name is StateName.YARD
    assert result.evidence == ("bottom_menu_yard",)


def test_exploration_from_two_required_evidence() -> None:
    """exploration：探索底部功能栏 + 章节栏（requires 两者都命中）。"""
    result = classifier().classify([evidence("bottom_menu_exploration"), evidence("chapter_panel")])

    assert result.state_name is StateName.EXPLORATION
    assert result.evidence == ("bottom_menu_exploration", "chapter_panel")


# ---------------------------------------------------------------------------
# 2. 证据不足 / 空证据
# ---------------------------------------------------------------------------


def test_battle_requires_two_visual_regions() -> None:
    """单条战斗证据（只覆盖一个视觉区域）不足以判定 battle。"""
    result = classifier().classify([evidence("battle_auto_button")])

    assert result.state_name is StateName.UNKNOWN
    assert result.confidence == pytest.approx(0.0)
    assert result.source is StateSource.EVIDENCE
    assert "no evidence rule matched" in result.reason
    assert result.evidence == ()


def test_same_visual_region_counts_once() -> None:
    """同一 group 内的多条证据只计一次——沿用聚合器的空间去重（不在此重新实现）。

    行动条与式神头像在配置里同属 ``battle_right_ui``：两条证据只覆盖**一个**视觉区域，
    不足以满足 ``min_groups=2``，因此不能判定为 battle。
    """
    result = classifier().classify(
        [
            evidence("battle_action_bar", group="battle_right_ui"),
            evidence("battle_shikigami_icons", group="battle_right_ui"),
        ]
    )

    assert result.state_name is StateName.UNKNOWN
    assert result.evidence == ()


def test_two_distinct_regions_are_enough() -> None:
    """两个**不同**视觉区域命中即可判定 battle（同区域内多条仍只算一个区域）。"""
    result = classifier().classify(
        [
            evidence("battle_action_bar", group="battle_right_ui"),
            evidence("battle_shikigami_icons", group="battle_right_ui"),
            evidence("battle_auto_button", group="battle_left_ui"),
        ]
    )

    assert result.state_name is StateName.BATTLE
    # 参与计分的是去重后的代表：右侧区域只贡献一条
    assert len(result.evidence) == 2
    assert "battle_auto_button" in result.evidence


def test_ungrouped_evidence_counts_per_name() -> None:
    """未分组证据（``group=""``）以自身证据名成组——沿用聚合器约定，本层不额外推理 group。"""
    result = classifier().classify([evidence("battle_auto_button"), evidence("battle_skill_area")])

    assert result.state_name is StateName.BATTLE


@pytest.mark.parametrize("empty", [[], ()])
def test_empty_evidence_returns_unknown(empty: Sequence[EvidenceResult]) -> None:
    """空证据（list / tuple）→ unknown，且 reason 明确说明"没有证据"。"""
    result = classifier().classify(empty)

    assert result.state_name is StateName.UNKNOWN
    assert result.confidence == pytest.approx(0.0)
    assert result.source is StateSource.EVIDENCE
    assert result.reason == "no evidence available"
    assert result.evidence == ()


def test_disabled_aggregator_reports_no_rule() -> None:
    """聚合器被配置关闭时同样返回 unknown，并把原因带进 reason（不静默）。"""
    config = EvidenceConfig(aggregator=EvidenceAggregatorConfig(enabled=False))
    result = classifier(config).classify([evidence("bottom_menu_yard")])

    assert result.state_name is StateName.UNKNOWN
    assert result.confidence == pytest.approx(0.0)
    assert "no evidence rule matched" in result.reason
    assert "disabled" in result.reason


# ---------------------------------------------------------------------------
# 3. 顺序无关与计分沿用聚合器
# ---------------------------------------------------------------------------


def test_result_is_order_independent() -> None:
    """同一组证据的不同顺序必须给出相同结果。"""
    items = [evidence("battle_auto_button"), evidence("battle_skill_area")]

    forward = classifier().classify(items)
    backward = classifier().classify(list(reversed(items)))

    assert forward == backward
    assert forward.state_name is StateName.BATTLE


def test_confidence_follows_evidence_scores_when_uncalibrated() -> None:
    """未配置校准的规则（battle）：状态可信度等于证据得分（计分沿用聚合器，不在此重算）。

    Phase 4.6.6 起 ``StateResult.confidence`` 是**状态可信度**；对未配置
    ``state_confidence`` 的规则，它等于证据得分。
    """
    low = classifier().classify(
        [evidence("battle_auto_button", 0.7), evidence("battle_skill_area", 0.7)]
    )
    high = classifier().classify(
        [evidence("battle_auto_button", 0.95), evidence("battle_skill_area", 0.95)]
    )

    assert low.state_name is StateName.BATTLE
    assert high.state_name is StateName.BATTLE
    assert low.confidence == pytest.approx(0.7)
    assert high.confidence == pytest.approx(0.95)
    assert low.confidence < high.confidence


def test_explicit_entry_matches_dispatcher() -> None:
    """显式 classify_evidence 与自动分派的 classify 结果一致。"""
    items = [evidence("victory_icon")]
    instance = classifier()

    assert instance.classify(items) == instance.classify_evidence(items)


# ---------------------------------------------------------------------------
# 4. 输入校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["battle_auto_button", b"bytes", 42, 3.5, None, {"name": "battle_auto_button"}, {"x"}],
)
def test_classify_rejects_non_evidence_input(bad: object) -> None:
    """既不是帧也不是证据序列的输入必须报 TypeError。"""
    with pytest.raises(TypeError, match="Frame"):
        classifier().classify(bad)  # type: ignore[arg-type]


def test_classify_rejects_non_evidence_items() -> None:
    """序列里混入非 EvidenceResult 元素时必须报错，并指出下标。"""
    with pytest.raises(TypeError, match=r"evidence\[1\]"):
        classifier().classify([evidence("battle_auto_button"), "battle_skill_area"])  # type: ignore[list-item]


def test_evidence_input_requires_aggregator() -> None:
    """只装配帧分支的实例收到证据输入时必须报错（而不是静默返回 unknown）。"""
    with pytest.raises(ValueError, match="aggregator"):
        frame_only_classifier().classify([evidence("bottom_menu_yard")])


def test_aggregator_exposure() -> None:
    """聚合器通过属性暴露，便于调用方检查规则数量。"""
    instance = classifier()

    assert instance.aggregator is not None
    assert len(instance.aggregator.rules) > 0
    assert frame_only_classifier().aggregator is None


# ---------------------------------------------------------------------------
# 5. 与帧驱动路径的兼容
# ---------------------------------------------------------------------------


def test_frame_input_uses_frame_path() -> None:
    """同一入口收到 Frame 时仍走 Phase 3.0 的视觉分支（source 为 FALLBACK 而非 EVIDENCE）。"""
    frame = Frame.create(np.zeros((8, 8, 3), dtype=np.uint8), source="test")

    result = classifier().classify(frame)

    assert result.state_name is StateName.UNKNOWN
    assert result.source is StateSource.FALLBACK
    assert result.evidence == ()


# ---------------------------------------------------------------------------
# 6. 禁止硬编码：规则必须来自配置
# ---------------------------------------------------------------------------

#: classifier 源码（用于静态检查）
CLASSIFIER_SOURCE = Path(inspect.getsourcefile(StateClassifier) or "").read_text(encoding="utf-8")


def string_literals(source: str) -> set[str]:
    """收集源码中的字符串常量（剔除文档字符串，它们只是说明文字）。"""
    tree = ast.parse(source)
    docstrings = {
        ast.get_docstring(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }


def test_classifier_has_no_hardcoded_evidence_names() -> None:
    """源码里不得出现证据名字面量（映射来自 evidence.aggregator.rules）。"""
    literals = string_literals(CLASSIFIER_SOURCE)

    assert literals.isdisjoint(set(EvidenceConfig().detectors))


def test_classifier_has_no_hardcoded_state_names() -> None:
    """源码里不得出现状态名字面量（状态名只在 StateName 中登记、在配置里使用）。"""
    literals = string_literals(CLASSIFIER_SOURCE)

    assert literals.isdisjoint({item.value for item in StateName})
