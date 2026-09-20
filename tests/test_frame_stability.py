"""FrameStabilityChecker 的单元测试。"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.core.config import StabilityConfig
from src.utils.logger import DEFAULT_LOGGER_NAME
from src.vision.capture.buffer import FrameBuffer
from src.vision.capture.comparator import FrameComparator
from src.vision.capture.models import Frame
from src.vision.capture.stability import FrameStabilityChecker


def make_frame(
    value: int = 0,
    *,
    noise: float = 0.0,
    seed: int = 0,
    size: tuple[int, int] = (16, 12),
) -> Frame:
    """构造测试帧：纯色，可选加入高斯噪声以模拟轻微画面变化。"""
    width, height = size
    image = np.full((height, width, 3), value, dtype=np.uint8)
    if noise:
        rng = np.random.default_rng(seed)
        noisy = image.astype(np.float64) + rng.normal(0.0, noise, image.shape)
        image = np.clip(noisy, 0, 255).astype(np.uint8)
    return Frame.create(image, source="test")


class FixedComparator(FrameComparator):
    """返回固定相似度的替身比较器。"""

    def __init__(self, similarity: float) -> None:
        self._similarity = similarity

    def compare(self, frame_a: Frame, frame_b: Frame) -> float:
        """忽略输入，始终返回预设相似度。"""
        return self._similarity


def test_stable_frames_are_detected() -> None:
    """几乎相同的连续帧判定为稳定。"""
    checker = FrameStabilityChecker(minimum_frames=3, similarity_threshold=0.98)
    frames = [make_frame(120, noise=0.5, seed=index) for index in range(3)]

    assert checker.is_stable(frames) is True
    assert checker.last_state is True


def test_unstable_frames_are_detected() -> None:
    """变化剧烈的连续帧判定为不稳定。"""
    checker = FrameStabilityChecker(minimum_frames=3, similarity_threshold=0.98)
    frames = [make_frame(value) for value in (0, 255, 10)]

    assert checker.is_stable(frames) is False
    assert checker.last_state is False


def test_empty_input_is_unstable() -> None:
    """空输入判定为不稳定。"""
    checker = FrameStabilityChecker(minimum_frames=3, similarity_threshold=0.98)

    assert checker.is_stable([]) is False
    assert checker.similarity([]) == 0.0


def test_insufficient_frames_are_unstable() -> None:
    """帧数少于 minimum_frames 时判定为不稳定，即使画面完全一致。"""
    checker = FrameStabilityChecker(minimum_frames=3, similarity_threshold=0.98)

    assert checker.is_stable([make_frame(50), make_frame(50)]) is False


def test_single_frame_similarity_is_zero() -> None:
    """单帧无法比较，相似度为 0.0。"""
    checker = FrameStabilityChecker()

    assert checker.similarity([make_frame(0)]) == 0.0


@pytest.mark.parametrize("minimum_frames", [0, 1, -1])
def test_invalid_minimum_frames_are_rejected(minimum_frames: int) -> None:
    """minimum_frames 必须不小于 2。"""
    with pytest.raises(ValueError, match="最少帧数"):
        FrameStabilityChecker(minimum_frames=minimum_frames)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    """相似度阈值必须落在 (0, 1] 区间内。"""
    with pytest.raises(ValueError, match="相似度阈值"):
        FrameStabilityChecker(similarity_threshold=threshold)


def test_from_config_reads_values() -> None:
    """由配置构造检查器，并保留默认比较器。"""
    checker = FrameStabilityChecker.from_config(
        StabilityConfig(enabled=True, minimum_frames=5, similarity_threshold=0.9)
    )

    assert checker.minimum_frames == 5
    assert checker.similarity_threshold == pytest.approx(0.9)
    assert isinstance(checker.comparator, FrameComparator)


def test_similarity_averages_adjacent_pairs() -> None:
    """平均相似度是相邻帧两两相似度的平均（3 帧 -> 2 对）。"""
    scores = iter([1.0, 0.5])

    class SequenceComparator(FrameComparator):
        """按调用顺序返回预设相似度。"""

        def compare(self, frame_a: Frame, frame_b: Frame) -> float:
            return next(scores, 0.0)

    checker = FrameStabilityChecker(
        minimum_frames=2, similarity_threshold=0.5, comparator=SequenceComparator()
    )
    frames = [make_frame(index) for index in range(3)]

    assert checker.similarity(frames) == pytest.approx((1.0 + 0.5) / 2)


def test_threshold_comparison_is_inclusive() -> None:
    """平均相似度恰好等于阈值时视为稳定。"""
    checker = FrameStabilityChecker(
        minimum_frames=2, similarity_threshold=0.5, comparator=FixedComparator(0.5)
    )

    assert checker.is_stable([make_frame(0), make_frame(200)]) is True


def test_state_change_is_logged_once_per_transition(caplog: pytest.LogCaptureFixture) -> None:
    """状态翻转时记录日志，且同一状态不会重复打印。"""
    checker = FrameStabilityChecker(minimum_frames=2, similarity_threshold=0.9)
    stable_frames = [make_frame(100), make_frame(100)]
    unstable_frames = [make_frame(0), make_frame(255)]

    with caplog.at_level(logging.INFO, logger=DEFAULT_LOGGER_NAME):
        checker.is_stable(stable_frames)
        checker.is_stable(stable_frames)
        checker.is_stable(unstable_frames)

    transitions = [message for message in caplog.messages if "Frame stability changed" in message]
    assert transitions == ["Frame stability changed: stable -> unstable"]


def test_reset_clears_state_history() -> None:
    """reset 后状态与相似度历史被清除。"""
    checker = FrameStabilityChecker(minimum_frames=2, similarity_threshold=0.9)
    checker.is_stable([make_frame(0), make_frame(255)])
    assert checker.last_state is False
    assert checker.last_similarity is not None

    checker.reset()

    assert checker.last_state is None
    assert checker.last_similarity is None


def test_last_similarity_is_cached_for_reuse() -> None:
    """判定后可复用 last_similarity，无需重复计算。"""
    checker = FrameStabilityChecker(
        minimum_frames=2, similarity_threshold=0.9, comparator=FixedComparator(0.75)
    )

    assert checker.last_similarity is None

    checker.is_stable([make_frame(0), make_frame(10)])

    assert checker.last_similarity == pytest.approx(0.75)


def test_works_with_frame_buffer_recent() -> None:
    """与 FrameBuffer 串联：recent(minimum_frames) 可直接用于判断。"""
    buffer = FrameBuffer(capacity=5)
    buffer.add_many([make_frame(200, noise=0.2, seed=index) for index in range(5)])
    checker = FrameStabilityChecker(minimum_frames=3, similarity_threshold=0.95)

    assert checker.is_stable(buffer.recent(checker.minimum_frames)) is True
