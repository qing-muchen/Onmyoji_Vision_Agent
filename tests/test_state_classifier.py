"""StateClassifier 的单元测试：三态判定、分支优先级与降级行为。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import (
    AppConfig,
    SceneConfig,
    SceneMatcherConfig,
    StateConfig,
    StateTransitionConfig,
    StateYardConfig,
)
from src.vision.capture.models import Frame
from src.vision.scene.loader import SceneLoader
from src.vision.state import StateName, StateClassifier, StateResult, StateSource
from src.vision.state.transition.classifier import TransitionClassifier
from src.vision.state.transition.dataset import SampleKind, TransitionSample, TransitionSampleLoader
from src.vision.state.yard.classifier import YardClassifier
from tests.helpers import build_scene_image

#: 合成图片尺寸
STRIP_HEIGHT = 58
STRIP_WIDTH = 1706
FRAME_HEIGHT = 961
#: 场景数据集中的相对路径
YARD_RELATIVE = Path("yard") / "skin_001" / "scene_001" / "yard_skin001_scene001_frame001.png"


def write_png(path: Path, image: np.ndarray) -> Path:
    """写出 PNG（路径可含中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", image)
    assert success
    buffer.tofile(str(path))
    return path


def flat_strip(value: int) -> np.ndarray:
    """纯色条带。"""
    return np.full((STRIP_HEIGHT, STRIP_WIDTH, 3), value, dtype=np.uint8)


def bright_frame() -> np.ndarray:
    """亮灰整图：与纯黑正样本、与有结构的场景模板都相距很远，用于"均未命中"用例。"""
    return np.full((FRAME_HEIGHT, STRIP_WIDTH, 3), 235, dtype=np.uint8)


def yard_image() -> np.ndarray:
    """合成"庭院"整图（有结构，避免纯色导致的相似度歧义）。"""
    return build_scene_image(size=(STRIP_WIDTH, FRAME_HEIGHT), seed=5)


def transition_classifier(image: np.ndarray) -> TransitionClassifier:
    """用给定整图作为唯一正样本构造转场分类器。"""
    sample = TransitionSample(
        sample_id="transition_bottom_001",
        kind=SampleKind.POSITIVE,
        path=Path("transition_bottom_001.png"),
        image=image,
    )
    return TransitionClassifier(StateTransitionConfig(), [sample], [])


def yard_classifier(tmp_path: Path, image: np.ndarray) -> YardClassifier:
    """用临时场景数据集构造庭院分类器（走真实 from_config 装配路径）。"""
    write_png(tmp_path / "scenes" / YARD_RELATIVE, image)
    return YardClassifier.from_config(
        StateYardConfig(),
        SceneMatcherConfig(),
        loader=SceneLoader(tmp_path / "scenes"),
    )


def frame_of(image: np.ndarray) -> Frame:
    """把图像包装为帧（状态识别不需要坐标映射）。"""
    return Frame.create(image, source="test")


def test_requires_at_least_one_branch() -> None:
    """两个分支都未提供时拒绝构造。"""
    with pytest.raises(ValueError, match="至少需要一个可用分支"):
        StateClassifier()


def test_transition_branch_wins() -> None:
    """transition 命中时直接返回 transition（即使 yard 也会命中）。"""
    image = yard_image()
    classifier = StateClassifier(
        transition=transition_classifier(image),
        yard=None,
    )

    result = classifier.classify(frame_of(image))

    assert isinstance(result, StateResult)
    assert result.state_name is StateName.TRANSITION
    assert result.source is StateSource.TRANSITION_BOTTOM_MATCH
    assert "transition_bottom_001.png" in (result.detail or "")
    assert result.confidence == pytest.approx(1.0)


def test_transition_takes_priority_over_yard(tmp_path: Path) -> None:
    """同一张图同时满足两个分支时，必须先返回 transition（强特征优先）。"""
    image = yard_image()
    classifier = StateClassifier(
        transition=transition_classifier(image),
        yard=yard_classifier(tmp_path, image),
    )

    result = classifier.classify(frame_of(image))

    assert result.state_name is StateName.TRANSITION
    assert result.source is StateSource.TRANSITION_BOTTOM_MATCH


def test_yard_branch_used_when_transition_misses(tmp_path: Path) -> None:
    """转场未命中时回落到庭院分支。"""
    image = yard_image()
    classifier = StateClassifier(
        transition=transition_classifier(flat_strip(0)),  # 与亮色庭院图的底部相距很远
        yard=yard_classifier(tmp_path, image),
    )

    result = classifier.classify(frame_of(image))

    assert result.state_name is StateName.YARD
    assert result.source is StateSource.SCENE_CLASSIFIER
    assert "yard/skin001/scene001" in (result.detail or "")


def test_unknown_when_no_branch_matches(tmp_path: Path) -> None:
    """两个分支都不命中时返回 unknown，并带上诊断信息。"""
    classifier = StateClassifier(
        transition=transition_classifier(flat_strip(0)),
        yard=yard_classifier(tmp_path, yard_image()),
    )

    result = classifier.classify(frame_of(bright_frame()))

    assert result.state_name is StateName.UNKNOWN
    assert result.source is StateSource.FALLBACK
    assert result.confidence == pytest.approx(0.0)
    assert result.is_known is False
    assert "transition positive=" in (result.detail or "")
    assert "yard below threshold" in (result.detail or "")


def test_unknown_when_yard_disabled() -> None:
    """庭院分支未启用时，场景命中也不会返回 yard。"""
    classifier = StateClassifier(
        transition=transition_classifier(flat_strip(0)),
        yard=None,
    )

    result = classifier.classify(frame_of(yard_image()))

    assert result.state_name is StateName.UNKNOWN
    assert "yard" not in (result.detail or "")


def test_unknown_when_transition_disabled(tmp_path: Path) -> None:
    """转场分支未启用时，仅靠庭院分支判定。"""
    image = yard_image()
    classifier = StateClassifier(transition=None, yard=yard_classifier(tmp_path, image))

    result = classifier.classify(frame_of(image))

    assert result.state_name is StateName.YARD
    assert "transition" not in (result.detail or "")


def test_classify_rejects_non_frame_input() -> None:
    """公共入口只接受 Frame。"""
    classifier = StateClassifier(transition=transition_classifier(flat_strip(0)), yard=None)

    with pytest.raises(TypeError, match="Frame"):
        classifier.classify(np.zeros((10, 10, 3), dtype=np.uint8))  # type: ignore[arg-type]


def test_from_config_builds_enabled_branches(tmp_path: Path) -> None:
    """from_config 按配置装配：两个分支都启用时可识别 yard 与 transition。"""
    yard_data = yard_image()
    write_png(tmp_path / "scenes" / YARD_RELATIVE, yard_data)
    write_png(tmp_path / "states" / "transition_bottom" / "transition_bottom_001.png", flat_strip(0))
    write_png(
        tmp_path / "states" / "non_transition_bottom" / "好友界面.png",
        build_scene_image(size=(STRIP_WIDTH, FRAME_HEIGHT), seed=11),
    )
    config = AppConfig(scene=SceneConfig(matcher=SceneMatcherConfig()), state=StateConfig())

    classifier = StateClassifier.from_config(
        config,
        state_loader=TransitionSampleLoader(tmp_path / "states"),
        scene_loader=SceneLoader(tmp_path / "scenes"),
    )

    assert classifier.name == "state"
    assert classifier.transition is not None
    assert classifier.yard is not None
    assert classifier.transition.negative_count == 1
    assert classifier.classify(frame_of(flat_strip(0))).state_name is StateName.TRANSITION
    assert classifier.classify(frame_of(yard_data)).state_name is StateName.YARD


def test_from_config_honours_disabled_switches(tmp_path: Path) -> None:
    """配置开关为 false 时不装配对应分支。"""
    write_png(tmp_path / "states" / "transition_bottom" / "transition_bottom_001.png", flat_strip(0))
    config = AppConfig(
        state=StateConfig(
            transition=StateTransitionConfig(enabled=False),
            yard=StateYardConfig(enabled=False),
        )
    )

    with pytest.raises(ValueError, match="至少需要一个可用分支"):
        StateClassifier.from_config(config, state_loader=TransitionSampleLoader(tmp_path / "states"))


def test_exposed_branches_for_logging() -> None:
    """暴露分支对象，便于 Demo 展示样本数量与排查。"""
    classifier = StateClassifier(transition=transition_classifier(flat_strip(0)), yard=None)

    assert classifier.name == "state"
    assert classifier.transition is not None
    assert classifier.transition.positive_count == 1
    assert classifier.yard is None
