"""连续帧稳定性判断。

只回答一个问题："画面是否已经停止变化"。不解读画面内容，也不判断界面状态
（这些属于后续 Phase 的 Detector / State 职责）。
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.config import (
    DEFAULT_MINIMUM_STABLE_FRAMES,
    DEFAULT_SIMILARITY_THRESHOLD,
    MINIMUM_COMPARABLE_FRAMES,
    StabilityConfig,
)
from src.utils.logger import get_logger
from src.vision.capture.comparator import FrameComparator
from src.vision.capture.models import Frame

__all__ = ["STABLE_STATE_TEXT", "UNSTABLE_STATE_TEXT", "FrameStabilityChecker"]

#: 稳定状态的日志文本
STABLE_STATE_TEXT: str = "stable"
#: 不稳定状态的日志文本
UNSTABLE_STATE_TEXT: str = "unstable"

logger = get_logger("vision.capture.stability")


def _state_text(stable: bool) -> str:
    """把稳定状态转换为日志文本。"""
    return STABLE_STATE_TEXT if stable else UNSTABLE_STATE_TEXT


class FrameStabilityChecker:
    """判断连续帧是否稳定（画面是否停止变化）。

    判定方式：对相邻帧两两比较相似度并取平均，平均值达到阈值即视为稳定。

    Args:
        minimum_frames: 参与判断的最少帧数；不足该数量时判定为不稳定。
        similarity_threshold: 判定稳定所需的平均相似度阈值，取值 ``(0, 1]``。
        comparator: 帧比较器；省略时使用默认的灰度 MAE 实现。

    Raises:
        ValueError: 参数越界。
    """

    def __init__(
        self,
        *,
        minimum_frames: int = DEFAULT_MINIMUM_STABLE_FRAMES,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        comparator: FrameComparator | None = None,
    ) -> None:
        if minimum_frames < MINIMUM_COMPARABLE_FRAMES:
            raise ValueError(
                f"最少帧数不能小于 {MINIMUM_COMPARABLE_FRAMES}（至少需要一对帧才能比较）: {minimum_frames}"
            )
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(f"相似度阈值必须落在 (0, 1] 区间内: {similarity_threshold}")
        self._minimum_frames = minimum_frames
        self._similarity_threshold = similarity_threshold
        self._comparator = comparator if comparator is not None else FrameComparator()
        self._last_stable: bool | None = None
        self._last_similarity: float | None = None

    @classmethod
    def from_config(
        cls,
        config: StabilityConfig,
        *,
        comparator: FrameComparator | None = None,
    ) -> FrameStabilityChecker:
        """由配置构造检查器。

        Args:
            config: 稳定性配置（``enabled`` 由调用方决定是否使用检查器）。
            comparator: 自定义比较器；省略时使用默认实现。

        Returns:
            检查器实例。
        """
        return cls(
            minimum_frames=config.minimum_frames,
            similarity_threshold=config.similarity_threshold,
            comparator=comparator,
        )

    @property
    def minimum_frames(self) -> int:
        """参与判断的最少帧数。"""
        return self._minimum_frames

    @property
    def similarity_threshold(self) -> float:
        """判定稳定所需的平均相似度阈值。"""
        return self._similarity_threshold

    @property
    def comparator(self) -> FrameComparator:
        """当前使用的帧比较器。"""
        return self._comparator

    @property
    def last_state(self) -> bool | None:
        """上一次的判定结果；尚未判断过时为 ``None``。"""
        return self._last_stable

    @property
    def last_similarity(self) -> float | None:
        """上一次判定使用的平均相似度；尚未判断过时为 ``None``。

        供日志与调用方复用，避免重复计算。
        """
        return self._last_similarity

    def similarity(self, frames: Sequence[Frame]) -> float:
        """计算相邻帧相似度的平均值。

        Args:
            frames: 按时间升序排列的帧序列。

        Returns:
            平均相似度；帧数少于 2 时返回 ``0.0``。
        """
        if len(frames) < MINIMUM_COMPARABLE_FRAMES:
            return 0.0
        scores = [
            self._comparator.compare(previous, current)
            for previous, current in zip(frames, frames[1:])
        ]
        return sum(scores) / len(scores)

    def is_stable(self, frames: Sequence[Frame]) -> bool:
        """判断给定帧序列是否稳定。

        状态发生翻转时记录一条日志（正常流程不会每帧打印日志）。

        Args:
            frames: 按时间升序排列的帧序列，通常取自
                :meth:`~src.vision.capture.buffer.FrameBuffer.recent`。

        Returns:
            ``True`` 表示帧数足够且平均相似度达到阈值。
        """
        similarity = self.similarity(frames)
        stable = len(frames) >= self._minimum_frames and similarity >= self._similarity_threshold
        self._last_similarity = similarity
        self._record_state(stable)
        return stable

    def reset(self) -> None:
        """清除上一次的判定记录（下次判断不会产生状态变化日志）。"""
        self._last_stable = None
        self._last_similarity = None

    def _record_state(self, stable: bool) -> None:
        """记录判定结果，仅在状态翻转时输出日志。"""
        previous = self._last_stable
        self._last_stable = stable
        if previous is None or previous == stable:
            return
        logger.info(
            "Frame stability changed: %s -> %s",
            _state_text(previous),
            _state_text(stable),
        )
