"""场景分类器抽象接口。

约定：

* 输入必须是 :class:`~src.vision.capture.models.Frame`；
* 场景识别是整图判断、不涉及坐标，因此**不要求**帧携带 ``coordinate_transform``
  （这与 :mod:`src.vision.detector` 的定位型检测不同）；
* 输出统一为 :class:`~src.vision.scene.result.SceneResult` 列表。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.vision.capture.models import Frame
from src.vision.scene.result import SceneResult

__all__ = ["BaseSceneClassifier"]


class BaseSceneClassifier(ABC):
    """场景分类器的统一接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """分类器名称（用于日志）。"""

    @abstractmethod
    def classify(self, frame: Frame) -> list[SceneResult]:
        """判断当前帧所属场景。

        Args:
            frame: 待识别的帧。

        Returns:
            场景结果列表，按 ``confidence`` 降序；无命中时返回空列表。
        """

    @abstractmethod
    def best_match(self, frame: Frame) -> SceneResult | None:
        """返回置信度最高的场景。

        Args:
            frame: 待识别的帧。

        Returns:
            最高置信度的场景结果；无命中时返回 ``None``。
        """
