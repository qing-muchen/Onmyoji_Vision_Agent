"""连续帧环形缓冲。

只负责保存"最近若干帧"并提供按时间顺序的访问，不做任何画面解读。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from src.core.config import DEFAULT_BUFFER_SIZE
from src.utils.logger import get_logger
from src.vision.capture.models import Frame

__all__ = ["FrameBuffer"]

logger = get_logger("vision.capture.buffer")


class FrameBuffer:
    """保存最近若干帧的环形缓冲。

    使用 :class:`collections.deque` 的 ``maxlen`` 实现容量控制：添加新帧时自动丢弃最旧帧。

    说明：缓冲保存的是 :class:`~src.vision.capture.models.Frame` 引用，**不复制**图像数据。
    ``Frame.image`` 为 numpy 数组，若需要长期保留某一帧的像素，请自行复制。

    Args:
        capacity: 最大保存帧数。

    Raises:
        ValueError: 容量不是正数。
    """

    def __init__(self, capacity: int = DEFAULT_BUFFER_SIZE) -> None:
        if capacity < 1:
            raise ValueError(f"缓冲容量必须为正数: {capacity}")
        self._capacity = capacity
        self._frames: deque[Frame] = deque(maxlen=capacity)
        logger.info("FrameBuffer initialized: capacity=%d", capacity)

    @property
    def capacity(self) -> int:
        """最大保存帧数。"""
        return self._capacity

    @property
    def is_empty(self) -> bool:
        """缓冲是否为空。"""
        return not self._frames

    def __len__(self) -> int:
        """当前保存的帧数。"""
        return len(self._frames)

    def __bool__(self) -> bool:
        """缓冲非空时为真。"""
        return bool(self._frames)

    def __iter__(self) -> Iterator[Frame]:
        """按时间升序（最旧 -> 最新）迭代缓冲中的帧。"""
        return iter(self._frames)

    def __repr__(self) -> str:
        return f"FrameBuffer(capacity={self._capacity}, frames={len(self._frames)})"

    def add(self, frame: Frame) -> None:
        """添加一帧；超出容量时自动丢弃最旧帧。

        Args:
            frame: 待添加的帧。
        """
        self._frames.append(frame)

    def add_many(self, frames: Iterable[Frame]) -> None:
        """按给定顺序批量添加帧。

        Args:
            frames: 可迭代的帧集合（按时间升序）。
        """
        for frame in frames:
            self.add(frame)

    def latest(self) -> Frame | None:
        """返回最新一帧。

        Returns:
            最新帧；缓冲为空时返回 ``None``。
        """
        return self._frames[-1] if self._frames else None

    def recent(self, count: int) -> list[Frame]:
        """返回最近 ``count`` 帧，按时间升序排列。

        Args:
            count: 需要的帧数；超过当前缓存数量时返回全部帧。

        Returns:
            帧列表；``count <= 0`` 时返回空列表。
        """
        if count <= 0:
            return []
        if count >= len(self._frames):
            return list(self._frames)
        return list(self._frames)[-count:]

    def clear(self) -> None:
        """清空缓冲。"""
        self._frames.clear()
