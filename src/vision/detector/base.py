"""检测器抽象接口。

约定：

* :meth:`Detector.detect` 的输入必须是 :class:`~src.vision.capture.models.Frame`
  ——帧携带时间戳与坐标映射，检测结果才能附带来源信息并正确映射到参考坐标系，
  因此本接口**不接受**裸 numpy 图像；
* 输出统一为 :class:`~src.vision.detector.result.DetectionResult` 列表。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.vision.capture.models import Frame
from src.vision.detector.result import DetectionResult

__all__ = ["Detector"]


class Detector(ABC):
    """视觉检测器的统一接口。

    实现类只需实现 :attr:`name` 与 :meth:`detect`；:meth:`detect_all` 提供了对多帧的
    便捷批处理。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """检测器名称（用于日志）。"""

    @abstractmethod
    def detect(self, frame: Frame) -> list[DetectionResult]:
        """在给定帧上执行检测。

        Args:
            frame: 待检测的帧；必须携带 ``coordinate_transform``，否则无法输出参考坐标。

        Returns:
            检测结果列表，按 ``confidence`` 降序；无命中时返回空列表。
        """

    def detect_all(self, frames: Sequence[Frame]) -> list[list[DetectionResult]]:
        """对多帧依次执行检测。

        Args:
            frames: 帧序列。

        Returns:
            与输入帧一一对应的结果列表。
        """
        return [self.detect(frame) for frame in frames]
