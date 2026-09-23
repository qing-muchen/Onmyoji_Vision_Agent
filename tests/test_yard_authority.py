"""yard 权威来源迁移测试（Phase 4.6.5）：**证据路径是 yard 的唯一权威**。

覆盖：

1. 证据路径：``bottom_menu_yard`` → ``StateName.YARD``（``source=evidence``，``reason`` / ``evidence`` 完整）；
2. 帧路径：庭院截图**不再**经 ``SceneClassifier`` 产出 yard（返回 unknown + 明确 reason）；
3. ``SceneClassifier`` / ``YardClassifier`` 仍可单独运行（组件保留，只是退出状态判定链）；
4. yard 与其它状态互不干扰（exploration / battle / battle_result / battle_reward / transition）；
5. 配置层面：``state=yard`` 规则**唯一**且只引用 ``bottom_menu_yard``（不存在 scene 直连规则）；
6. 禁止硬编码：``classifier.py`` 不含 ``bottom_menu_yard``。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import (
    EvidenceConfig,
    SceneMatcherConfig,
    StateTransitionConfig,
    StateYardConfig,
    load_config,
)
from src.core.paths import resolve_from_project_root
from src.vision.capture import Frame
from src.vision.evidence import EvidenceResult
from src.vision.scene.classifier import SceneClassifier
from src.vision.scene.loader import SceneLoader
from src.vision.state import StateClassifier, StateName, StateSource
from src.vision.state.classifier import FRAME_YARD_DISABLED_REASON
from src.vision.state.transition.classifier import TransitionClassifier
from src.vision.state.transition.dataset import SampleKind, TransitionSample
from src.vision.state.yard.classifier import YardClassifier
from tests.helpers import build_scene_image

#: 合成整图尺寸（与既有 state 测试一致）
FRAME_WIDTH = 1706
FRAME_HEIGHT = 961
STRIP_HEIGHT = 58
#: 临时场景数据集里的相对路径（层级必须符合 SceneLoader 规范）
YARD_RELATIVE = Path("yard") / "skin_001" / "scene_001" / "yard_skin001_scene001_frame001.png"
#: 真实数据集（用于端到端用例；缺失时跳过）
SCENES_DIR = Path("dataset/scenes")
REAL_YARD_FRAME = "dataset/raw/yard/yard_cap_001.png"


def write_png(path: Path, image: np.ndarray) -> Path:
    """写出 PNG（路径可含中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", image)
    assert success
    buffer.tofile(str(path))
    return path


def load_image(path: Path) -> np.ndarray:
    """读取图像（兼容非 ASCII 路径）。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def yard_image() -> np.ndarray:
    """合成"庭院"整图（有结构，避免纯色导致的相似度歧义）。"""
    return build_scene_image(size=(FRAME_WIDTH, FRAME_HEIGHT), seed=5)


def frame_of(image: np.ndarray) -> Frame:
    """把图像包装为帧（状态识别不需要坐标映射）。"""
    return Frame.create(image, source="test")


def evidence(name: str, confidence: float = 0.9) -> EvidenceResult:
    """构造一条命中证据。"""
    return EvidenceResult(name=name, detected=True, confidence=confidence)


def classifier() -> StateClassifier:
    """纯证据驱动分类器（yard 的权威路径）。"""
    return StateClassifier.from_evidence_config(EvidenceConfig())


def yard_component(tmp_path: Path, image: np.ndarray) -> YardClassifier:
    """用临时场景数据集构造庭院视觉组件（走真实 from_config 装配路径）。"""
    write_png(tmp_path / "scenes" / YARD_RELATIVE, image)
    return YardClassifier.from_config(
        StateYardConfig(),
        SceneMatcherConfig(),
        loader=SceneLoader(tmp_path / "scenes"),
    )


def scene_component(tmp_path: Path, image: np.ndarray) -> SceneClassifier:
    """用临时场景数据集构造场景分类器（独立组件）。"""
    write_png(tmp_path / "scenes" / YARD_RELATIVE, image)
    return SceneClassifier.from_dataset(
        SceneMatcherConfig(), loader=SceneLoader(tmp_path / "scenes")
    )


def transition_component(image: np.ndarray) -> TransitionClassifier:
    """用给定图像作为唯一正样本构造转场判定器（帧路径的另一半）。"""
    sample = TransitionSample(
        sample_id="transition_bottom_001",
        kind=SampleKind.POSITIVE,
        path=Path("transition_bottom_001.png"),
        image=image,
    )
    return TransitionClassifier(StateTransitionConfig(), [sample], [])


# ---------------------------------------------------------------------------
# 1. 证据路径是权威
# ---------------------------------------------------------------------------


def test_evidence_yard_is_the_authority() -> None:
    """bottom_menu_yard 命中即判定为 yard，且解释信息完整。"""
    result = classifier().classify([evidence("bottom_menu_yard")])

    assert result.state_name is StateName.YARD
    assert result.source is StateSource.EVIDENCE
    assert result.confidence == pytest.approx(0.9)
    assert result.reason
    assert "matched yard evidence rule" in result.reason
    assert result.evidence == ("bottom_menu_yard",)


def test_yard_does_not_require_frame_pixels() -> None:
    """yard 判定只消费证据：分类器不需要帧、不需要场景模板。"""
    instance = classifier()

    assert instance.aggregator is not None
    assert instance.transition is None
    assert instance.yard is None  # 纯证据驱动实例不装配任何视觉分支
    assert instance.classify_evidence([evidence("bottom_menu_yard", 0.95)]).state_name is (
        StateName.YARD
    )


# ---------------------------------------------------------------------------
# 2. 帧路径不再产出 yard
# ---------------------------------------------------------------------------


def test_frame_path_does_not_produce_yard(tmp_path: Path) -> None:
    """庭院截图走帧路径时不得返回 yard——即使场景组件确实识别出了它。"""
    image = yard_image()
    instance = StateClassifier(
        transition=transition_component(np.full((STRIP_HEIGHT, FRAME_WIDTH, 3), 0, dtype=np.uint8)),
        yard=yard_component(tmp_path, image),
    )

    # 前提：场景组件本身能识别该图（证明"组件被停用"，而不是"组件没命中"）
    assert instance.yard is not None
    assert instance.yard.classify(frame_of(image)) is not None

    result = instance.classify(frame_of(image))

    assert result.state_name is not StateName.YARD
    assert result.state_name is StateName.UNKNOWN
    assert result.source is StateSource.FALLBACK
    assert result.confidence == pytest.approx(0.0)
    assert result.reason == FRAME_YARD_DISABLED_REASON
    assert result.evidence == ()


def test_frame_path_still_reports_transition(tmp_path: Path) -> None:
    """帧路径保留的能力（转场）不受影响。"""
    strip = np.full((STRIP_HEIGHT, FRAME_WIDTH, 3), 0, dtype=np.uint8)
    instance = StateClassifier(
        transition=transition_component(strip),
        yard=yard_component(tmp_path, yard_image()),
    )

    result = instance.classify(frame_of(strip))

    assert result.state_name is StateName.TRANSITION
    assert result.source is StateSource.TRANSITION_BOTTOM_MATCH


@pytest.mark.skipif(
    not (SCENES_DIR.is_dir() and resolve_from_project_root(REAL_YARD_FRAME).is_file()),
    reason="真实 scene / 庭院截图不存在",
)
def test_real_yard_frame_no_longer_classified_by_frame_path() -> None:
    """真实庭院截图 + 仓库配置：帧路径给出 unknown，并把 yard 指回证据路径。"""
    config = load_config(environ={})
    instance = StateClassifier.from_config(config)
    frame = Frame.create(
        load_image(resolve_from_project_root(REAL_YARD_FRAME)), source="real_yard"
    )

    result = instance.classify(frame)

    assert result.state_name is StateName.UNKNOWN
    assert result.reason == FRAME_YARD_DISABLED_REASON


# ---------------------------------------------------------------------------
# 3. 组件仍可单独运行（只是退出状态判定链）
# ---------------------------------------------------------------------------


def test_scene_classifier_still_runs_standalone(tmp_path: Path) -> None:
    """SceneClassifier 未删除、仍可独立使用。"""
    image = yard_image()
    component = scene_component(tmp_path, image)

    results = component.classify(frame_of(image))

    assert results
    assert results[0].confidence > 0.9


def test_yard_classifier_still_runs_standalone(tmp_path: Path) -> None:
    """YardClassifier（Phase 4.6.5 起标记 deprecated）仍可作为视觉组件使用。"""
    image = yard_image()
    component = yard_component(tmp_path, image)

    result = component.classify(frame_of(image))

    assert result is not None
    assert result.confidence > 0.9


# ---------------------------------------------------------------------------
# 4. yard 与其它状态互不干扰
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (
            [evidence("bottom_menu_exploration"), evidence("chapter_panel")],
            StateName.EXPLORATION,
        ),
        ([evidence("battle_auto_button"), evidence("battle_skill_area")], StateName.BATTLE),
        ([evidence("victory_icon")], StateName.BATTLE_RESULT),
        ([evidence("reward_panel")], StateName.BATTLE_REWARD),
        ([evidence("transition_banner")], StateName.TRANSITION),
    ],
)
def test_other_states_unaffected(items: list[EvidenceResult], expected: StateName) -> None:
    """其它状态的判定不受 yard 权威迁移影响。"""
    result = classifier().classify(items)

    assert result.state_name is expected
    assert "bottom_menu_yard" not in result.evidence


def test_yard_and_others_are_mutually_exclusive_evidence() -> None:
    """只有 yard 证据时不会误判成其它状态（反之亦然：yard 证据不与其它规则共享）。"""
    instance = classifier()

    yard_only = instance.classify([evidence("bottom_menu_yard")])
    battle_only = instance.classify([evidence("battle_auto_button"), evidence("battle_skill_area")])

    assert yard_only.state_name is StateName.YARD
    assert battle_only.state_name is StateName.BATTLE


# ---------------------------------------------------------------------------
# 5. 配置层面：yard 规则唯一且基于证据
# ---------------------------------------------------------------------------


def test_yard_rule_is_unique_and_evidence_based() -> None:
    """``state=yard`` 的规则必须恰好一条，且只引用 bottom_menu_yard。"""
    rules = [rule for rule in EvidenceConfig().aggregator.rules if rule.state == "yard"]

    assert len(rules) == 1
    assert rules[0].requires == ["bottom_menu_yard"]
    assert rules[0].any_of == []


def test_all_rules_reference_registered_evidence_only() -> None:
    """所有规则引用的名字都必须是**已登记的 detector**（不存在 scene 直连规则）。"""
    config = EvidenceConfig()
    referenced = {
        name for rule in config.aggregator.rules for name in (*rule.requires, *rule.any_of)
    }

    assert referenced <= set(config.detectors)
    assert "bottom_menu_yard" in referenced


# ---------------------------------------------------------------------------
# 6. 禁止硬编码
# ---------------------------------------------------------------------------


def string_literals(source: str) -> set[str]:
    """收集源码中的字符串常量。

    剔除文档字符串：docstring 是说明文字（例如"yard 的权威来源是 ``bottom_menu_yard``
    证据"），不属于硬编码映射；真正要禁止的是**代码里出现证据名**。
    """
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


def test_classifier_has_no_hardcoded_yard_evidence() -> None:
    """classifier 的**代码**里不得出现 bottom_menu_yard（映射必须来自配置）。"""
    source = Path(inspect.getsourcefile(StateClassifier) or "").read_text(encoding="utf-8")

    literals = string_literals(source)

    assert "bottom_menu_yard" not in literals
    assert literals.isdisjoint(set(EvidenceConfig().detectors))


def test_yard_disabled_reason_is_explicit() -> None:
    """帧路径的"不判定庭院"理由必须可读且指向证据路径。"""
    assert "yard" in FRAME_YARD_DISABLED_REASON
    assert "evidence path" in FRAME_YARD_DISABLED_REASON
