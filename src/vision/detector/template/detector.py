"""基于模板匹配的检测器实现。

:class:`TemplateDetector` 是 :class:`~src.vision.detector.base.Detector` 的实现：
把一个模板集合应用到帧上，并把命中区域统一转换到**参考分辨率坐标系**后再输出。
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.config import TemplateDetectorConfig
from src.core.exceptions import MissingTransformError
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.detector.base import Detector
from src.vision.detector.models import TemplateInfo
from src.vision.detector.result import DetectionResult
from src.vision.detector.template.loader import TemplateLoader
from src.vision.detector.template.matcher import TemplateMatcher

__all__ = ["TemplateDetector"]

logger = get_logger("vision.detector.template.detector")


class TemplateDetector(Detector):
    """把一组模板应用到帧上的检测器。

    流程：``Frame`` → 灰度化 → 逐模板多尺度匹配 → 阈值过滤 → 坐标转换（→ reference）
    → :class:`~src.vision.detector.result.DetectionResult`。

    Args:
        templates: 已加载的模板集合（至少一个）。
        config: 模板检测配置。
        matcher: 自定义匹配器；省略时按配置创建。

    Raises:
        ValueError: 模板集合为空。
    """

    def __init__(
        self,
        templates: Sequence[TemplateInfo],
        config: TemplateDetectorConfig,
        *,
        matcher: TemplateMatcher | None = None,
    ) -> None:
        if not templates:
            raise ValueError("至少需要一个模板才能构造 TemplateDetector")
        self._templates = tuple(templates)
        self._config = config
        self._matcher = matcher if matcher is not None else TemplateMatcher(config)

    @classmethod
    def from_dataset(
        cls,
        config: TemplateDetectorConfig,
        *,
        loader: TemplateLoader | None = None,
        category: str | None = None,
        element_type: str | None = None,
        element_name: str | None = None,
    ) -> TemplateDetector:
        """从模板数据集构造检测器。

        Args:
            config: 模板检测配置。
            loader: 自定义加载器；省略时使用默认模板目录。
            category: 只使用该类别目录。
            element_type: 只使用该元素类型。
            element_name: 只使用该元素名称。

        Returns:
            检测器实例。

        Raises:
            TemplateLoadError: 模板目录不存在或图像无法读取。
            TemplateParseError: 目录层级或文件名不符合规范。
            ValueError: 过滤后没有任何模板。
        """
        active_loader = loader if loader is not None else TemplateLoader()
        templates = active_loader.load(
            category=category,
            element_type=element_type,
            element_name=element_name,
        )
        return cls(templates, config)

    @property
    def name(self) -> str:
        """检测器名称。"""
        return "template"

    @property
    def templates(self) -> tuple[TemplateInfo, ...]:
        """已加载的模板集合。"""
        return self._templates

    @property
    def config(self) -> TemplateDetectorConfig:
        """当前配置。"""
        return self._config

    @property
    def matcher(self) -> TemplateMatcher:
        """当前使用的匹配器。"""
        return self._matcher

    def detect(self, frame: Frame) -> list[DetectionResult]:
        """在帧上执行模板检测。

        Args:
            frame: 待检测的帧；必须携带 ``coordinate_transform``。

        Returns:
            检测结果列表（按 ``confidence`` 降序，坐标为参考分辨率坐标系）；
            无命中时返回空列表。

        Raises:
            MissingTransformError: 帧未携带坐标映射，无法输出参考坐标。
        """
        transform = frame.coordinate_transform
        if transform is None:
            raise MissingTransformError(
                "帧未携带 coordinate_transform，无法把检测结果映射到参考坐标系"
            )

        frame_gray = self._matcher.prepare(frame.image)
        results: list[DetectionResult] = []
        for template in self._templates:
            match = self._matcher.match(frame_gray, template)
            if match is None:
                continue
            results.append(
                DetectionResult(
                    element_name=template.element_name,
                    category=template.category,
                    confidence=match.confidence,
                    rect=transform.current_rect_to_reference(match.rect),
                    template_id=template.template_id,
                )
            )

        results.sort(key=lambda item: item.confidence, reverse=True)
        if results:
            logger.debug(
                "模板检测命中 %d 项，最高 confidence=%.4f（%s）",
                len(results),
                results[0].confidence,
                results[0].template_id,
            )
        return results

    @staticmethod
    def best_per_element(results: Sequence[DetectionResult]) -> list[DetectionResult]:
        """按元素去重，只保留每个元素置信度最高的一条结果。

        Args:
            results: 检测结果（通常已是 :meth:`detect` 的降序输出）。

        Returns:
            每个 ``category/element_name`` 的最高分结果，按 ``confidence`` 降序。
        """
        best: dict[str, DetectionResult] = {}
        for result in results:
            current = best.get(result.element_key)
            if current is None or result.confidence > current.confidence:
                best[result.element_key] = result
        return sorted(best.values(), key=lambda item: item.confidence, reverse=True)
