"""状态分类器抽象接口（Phase 3.0）。

约定：

* 输入必须是 :class:`~src.vision.capture.models.Frame`——帧携带时间戳与来源信息，
  便于后续状态机按"哪一帧得出的结论"回溯；因此本接口**不接受裸 numpy 图像**。
* 状态识别是整图 / 区域判断，**不要求**帧携带 ``coordinate_transform``
  （与 :mod:`src.vision.scene` 一致，与 :mod:`src.vision.detector` 的定位型检测不同）。
* 输出统一为单个 :class:`~src.vision.state.models.StateResult`（三态之一），
  **不做决策、不触发任何交互**。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.vision.capture.models import Frame
from src.vision.state.models import StateResult

__all__ = ["BaseStateClassifier"]


class BaseStateClassifier(ABC):
    """状态分类器的统一接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """分类器名称（用于日志）。"""

    @abstractmethod
    def classify(self, frame: Frame) -> StateResult:
        """判断当前帧所处的游戏状态。

        Args:
            frame: 待识别的帧。

        Returns:
            状态结果；无法判定时返回 ``unknown``（而不是抛异常）。
        """
