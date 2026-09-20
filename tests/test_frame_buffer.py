"""FrameBuffer 的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.vision.capture.buffer import FrameBuffer
from src.vision.capture.models import Frame


def make_frame(value: int = 0, *, size: tuple[int, int] = (16, 12), timestamp: float | None = None) -> Frame:
    """构造一张纯色测试帧。

    Args:
        value: BGR 三通道的像素值。
        size: ``(width, height)``。
        timestamp: 覆盖采集时间戳。
    """
    width, height = size
    image = np.full((height, width, 3), value, dtype=np.uint8)
    return Frame.create(image, source="test", timestamp=timestamp)


def test_new_buffer_is_empty() -> None:
    """新建缓冲为空，且 latest/recent 返回空结果。"""
    buffer = FrameBuffer(capacity=3)

    assert buffer.capacity == 3
    assert buffer.is_empty is True
    assert len(buffer) == 0
    assert not buffer
    assert buffer.latest() is None
    assert buffer.recent(2) == []


def test_add_and_latest_return_most_recent_frame() -> None:
    """latest 返回最后添加的帧。"""
    buffer = FrameBuffer(capacity=5)
    frames = [make_frame(index) for index in range(3)]

    buffer.add_many(frames)

    assert len(buffer) == 3
    assert buffer.latest() is frames[-1]


def test_recent_returns_frames_in_chronological_order() -> None:
    """recent 按时间升序返回最近的若干帧。"""
    buffer = FrameBuffer(capacity=5)
    frames = [make_frame(index) for index in range(5)]
    buffer.add_many(frames)

    assert buffer.recent(3) == frames[-3:]
    assert buffer.recent(1) == [frames[-1]]


def test_recent_clamps_to_available_frames() -> None:
    """请求数量超过缓存数量时返回全部帧。"""
    buffer = FrameBuffer(capacity=10)
    frames = [make_frame(index) for index in range(2)]
    buffer.add_many(frames)

    assert buffer.recent(5) == frames
    assert buffer.recent(2) == frames


@pytest.mark.parametrize("count", [0, -1])
def test_recent_with_non_positive_count_returns_empty(count: int) -> None:
    """count <= 0 时返回空列表。"""
    buffer = FrameBuffer(capacity=3)
    buffer.add(make_frame(1))

    assert buffer.recent(count) == []


def test_capacity_drops_oldest_frame() -> None:
    """超出容量时自动丢弃最旧帧。"""
    buffer = FrameBuffer(capacity=3)
    frames = [make_frame(index) for index in range(5)]

    buffer.add_many(frames)

    assert len(buffer) == 3
    assert list(buffer) == frames[-3:]
    assert frames[0] not in list(buffer)
    assert buffer.latest() is frames[-1]


def test_clear_empties_buffer_but_keeps_capacity() -> None:
    """clear 清空缓冲并保留容量设置。"""
    buffer = FrameBuffer(capacity=2)
    buffer.add_many([make_frame(1), make_frame(2)])

    buffer.clear()

    assert buffer.is_empty is True
    assert len(buffer) == 0
    assert buffer.latest() is None
    assert buffer.capacity == 2


@pytest.mark.parametrize("capacity", [0, -5])
def test_invalid_capacity_is_rejected(capacity: int) -> None:
    """容量必须为正数。"""
    with pytest.raises(ValueError, match="缓冲容量"):
        FrameBuffer(capacity=capacity)


def test_repr_mentions_capacity_and_size() -> None:
    """__repr__ 便于调试。"""
    buffer = FrameBuffer(capacity=4)
    buffer.add(make_frame(0))

    assert "capacity=4" in repr(buffer)
    assert "frames=1" in repr(buffer)
