"""TransitionClassifier 的单元测试：正样本阈值、负样本否决与配置驱动。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.config import StateTransitionConfig
from src.core.exceptions import StateDataError
from src.vision.capture.models import Frame
from src.vision.state.transition.classifier import TransitionClassifier
from src.vision.state.transition.dataset import SampleKind, TransitionSample

#: 合成样本尺寸（纯色/块状图便于得到确定的相似度）
STRIP_HEIGHT = 58
STRIP_WIDTH = 1706
#: 块状样本中被点亮的宽度（约 10% → 与纯黑样本的相似度约 0.9）
BLOCK_WIDTH = 170


def sample(
    sample_id: str,
    kind: SampleKind,
    image: np.ndarray,
) -> TransitionSample:
    """直接构造样本（不走文件系统）。"""
    return TransitionSample(
        sample_id=sample_id,
        kind=kind,
        path=Path(f"{sample_id}.png"),
        image=image,
    )


def flat(value: int) -> np.ndarray:
    """纯色条带。"""
    return np.full((STRIP_HEIGHT, STRIP_WIDTH, 3), value, dtype=np.uint8)


def blocked() -> np.ndarray:
    """左侧 10% 为白的条带（与纯黑条带的相似度约 0.9）。"""
    image = flat(0)
    image[:, :BLOCK_WIDTH] = 255
    return image


def frame_of(image: np.ndarray) -> Frame:
    """把条带包装为帧。"""
    return Frame.create(image, source="test")


def build_classifier(
    positives: list[TransitionSample],
    negatives: list[TransitionSample] | None = None,
    **overrides: float,
) -> TransitionClassifier:
    """构造转场分类器（默认配置 + 可覆盖项）。"""
    config = StateTransitionConfig(**overrides)  # type: ignore[arg-type]
    return TransitionClassifier(config, positives, negatives or [])


def test_requires_at_least_one_positive() -> None:
    """没有正样本时拒绝构造（数据缺失不能静默变成"永远不是转场"）。"""
    with pytest.raises(StateDataError, match="正样本"):
        build_classifier([])


def test_matching_strip_is_transition() -> None:
    """与正样本一致的条带判定为转场，且给出命中样本信息。"""
    classifier = build_classifier([sample("transition_bottom_001", SampleKind.POSITIVE, flat(0))])

    match = classifier.match(frame_of(flat(0)))

    assert match.is_transition is True
    assert match.confidence == pytest.approx(1.0)
    assert match.positive_score == pytest.approx(1.0)
    assert match.best_positive_id == "transition_bottom_001"
    assert match.best_positive_file == "transition_bottom_001.png"


def test_unrelated_strip_is_not_transition() -> None:
    """与正样本完全不同的条带（反向颜色）不判定为转场。"""
    classifier = build_classifier([sample("positive", SampleKind.POSITIVE, flat(0))])

    match = classifier.match(frame_of(flat(255)))

    assert match.is_transition is False
    assert match.confidence == 0.0
    assert match.positive_score < 0.85


def test_threshold_is_configuration_driven() -> None:
    """阈值来自配置：调高阈值后原本命中的条带不再判定为转场。"""
    positives = [sample("positive", SampleKind.POSITIVE, flat(0))]
    query = frame_of(blocked())

    lenient = build_classifier(positives, threshold=0.85).match(query)
    strict = build_classifier(positives, threshold=0.99).match(query)

    assert lenient.is_transition is True
    assert strict.is_transition is False
    assert lenient.positive_score == pytest.approx(strict.positive_score)


def test_negative_veto_blocks_false_positive() -> None:
    """负样本否决：即使很像正样本，只要更像某个负样本就不算转场。"""
    query_image = blocked()
    positives = [sample("positive", SampleKind.POSITIVE, flat(0))]
    negatives = [sample("negative", SampleKind.NEGATIVE, query_image)]

    with_veto = build_classifier(positives, negatives).match(frame_of(query_image))
    without_veto = build_classifier(positives).match(frame_of(query_image))

    assert with_veto.positive_score >= 0.85  # 正样本分数本身足够高
    assert with_veto.is_transition is False  # 但被负样本否决
    assert without_veto.is_transition is True


def test_negative_margin_is_configuration_driven() -> None:
    """负样本间隔来自配置：同一个查询在间隔为 0 时通过、为 0.03 时被否。"""
    image = flat(0)
    positives = [sample("positive", SampleKind.POSITIVE, image)]
    negatives = [sample("negative", SampleKind.NEGATIVE, image)]

    loose = build_classifier(positives, negatives, negative_margin=0.0).match(frame_of(image))
    strict = build_classifier(positives, negatives, negative_margin=0.03).match(frame_of(image))

    assert loose.is_transition is True  # gap = 0 >= 0
    assert strict.is_transition is False  # gap = 0 < 0.03


def test_without_negatives_only_threshold_applies() -> None:
    """没有负样本时退化为"仅正样本阈值"，且过程数据仍完整。"""
    classifier = build_classifier([sample("positive", SampleKind.POSITIVE, flat(0))])

    match = classifier.match(frame_of(flat(0)))

    assert match.is_transition is True
    assert match.negative_score == 0.0
    assert match.best_negative_id is None
    assert "n/a" in match.describe()


def test_negative_samples_are_cropped_to_strip() -> None:
    """负样本是完整截图时会先裁出底部条带再比较（与运行时表示一致）。"""
    strip = flat(0)
    full_frame = np.zeros((961, STRIP_WIDTH, 3), dtype=np.uint8)
    full_frame[961 - STRIP_HEIGHT :, :] = strip  # 底部与正样本一致

    classifier = build_classifier(
        [sample("positive", SampleKind.POSITIVE, strip)],
        [sample("negative", SampleKind.NEGATIVE, full_frame)],
    )
    match = classifier.match(frame_of(strip))

    assert match.negative_score == pytest.approx(1.0)
    assert match.is_transition is False  # 被"底部相同"的负样本否决


def test_counts_and_name_are_exposed() -> None:
    """暴露名称与正/负样本数量，便于日志与 Demo 展示。"""
    classifier = build_classifier(
        [sample("p", SampleKind.POSITIVE, flat(0))],
        [sample("n", SampleKind.NEGATIVE, flat(255))],
    )

    assert classifier.name == "transition"
    assert classifier.positive_count == 1
    assert classifier.negative_count == 1
    assert classifier.config.threshold == pytest.approx(0.85)


def test_match_rejects_non_frame_input() -> None:
    """公共入口只接受 Frame。"""
    classifier = build_classifier([sample("p", SampleKind.POSITIVE, flat(0))])

    with pytest.raises(TypeError, match="Frame"):
        classifier.match(flat(0))  # type: ignore[arg-type]


def test_from_dataset_loads_from_directory(tmp_path: Path) -> None:
    """from_dataset 从 ``dataset/states`` 结构加载样本。"""
    import cv2

    from src.vision.state.transition.dataset import TransitionSampleLoader

    positive_dir = tmp_path / "transition_bottom"
    positive_dir.mkdir(parents=True)
    success, buffer = cv2.imencode(".png", flat(0))
    assert success
    buffer.tofile(str(positive_dir / "transition_bottom_001.png"))
    negative_dir = tmp_path / "non_transition_bottom"
    negative_dir.mkdir(parents=True)
    buffer.tofile(str(negative_dir / "对照界面.png"))

    classifier = TransitionClassifier.from_dataset(
        StateTransitionConfig(),
        loader=TransitionSampleLoader(tmp_path),
    )

    assert classifier.positive_count == 1
    assert classifier.negative_count == 1
