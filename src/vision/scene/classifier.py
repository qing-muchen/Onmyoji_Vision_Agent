"""场景分类器：多模板投票 + 阈值过滤。

同一场景（``scene_type + skin + scene_id``）通常采集了多张截图，因此分类时先按场景分组，
再取组内**最高**相似度作为该场景的分数——任意一张截图匹配成功即说明"当前在该场景"，
取平均会被角度/动画差异最大的那张拖低。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.config import SceneMatcherConfig
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.scene.base import BaseSceneClassifier
from src.vision.scene.loader import SceneLoader
from src.vision.scene.matcher import SceneMatcher
from src.vision.scene.models import SceneTemplate
from src.vision.scene.result import SceneResult

__all__ = ["SceneClassifier"]

logger = get_logger("vision.scene.classifier")


class SceneClassifier(BaseSceneClassifier):
    """基于场景数据集的整图匹配分类器。

    Args:
        templates: 已加载的场景模板（至少一个）。
        config: 场景匹配配置。
        matcher: 自定义匹配器；省略时按配置创建。

    Raises:
        ValueError: 模板集合为空。
    """

    def __init__(
        self,
        templates: Sequence[SceneTemplate],
        config: SceneMatcherConfig,
        *,
        matcher: SceneMatcher | None = None,
    ) -> None:
        if not templates:
            raise ValueError("至少需要一个场景模板才能构造 SceneClassifier")
        self._templates = tuple(templates)
        self._config = config
        self._matcher = matcher if matcher is not None else SceneMatcher(config)
        # 模板只预处理一次，后续每次 classify 都复用（同一场景多张截图时收益明显）
        self._prepared: dict[str, np.ndarray] = {
            template.source_id: self._matcher.prepare(template.image) for template in self._templates
        }

    @classmethod
    def from_dataset(
        cls,
        config: SceneMatcherConfig,
        *,
        loader: SceneLoader | None = None,
        scene_type: str | None = None,
        skin: int | None = None,
        scene_id: int | None = None,
    ) -> SceneClassifier:
        """从场景数据集构造分类器。

        Args:
            config: 场景匹配配置。
            loader: 自定义加载器；省略时使用默认场景目录。
            scene_type: 只使用该场景类型。
            skin: 只使用该皮肤编号。
            scene_id: 只使用该场景编号。

        Returns:
            分类器实例。

        Raises:
            SceneLoadError: 场景目录不存在或图像无法读取。
            SceneTemplateError: 目录层级、目录名或文件名不符合规范。
            ValueError: 过滤后没有任何场景模板。
        """
        active_loader = loader if loader is not None else SceneLoader()
        templates = active_loader.load(scene_type=scene_type, skin=skin, scene_id=scene_id)
        return cls(templates, config)

    @property
    def name(self) -> str:
        """分类器名称。"""
        return "scene"

    @property
    def templates(self) -> tuple[SceneTemplate, ...]:
        """已加载的场景模板集合。"""
        return self._templates

    @property
    def config(self) -> SceneMatcherConfig:
        """当前配置。"""
        return self._config

    @property
    def matcher(self) -> SceneMatcher:
        """当前使用的匹配器。"""
        return self._matcher

    def classify(self, frame: Frame) -> list[SceneResult]:
        """判断当前帧所属场景。

        Args:
            frame: 待识别的帧；整图判断，不要求携带坐标映射。

        Returns:
            场景结果列表（按 ``confidence`` 降序，已按阈值过滤）；无命中时返回空列表。

        Raises:
            TypeError: ``frame`` 不是 :class:`~src.vision.capture.models.Frame`。
        """
        self._validate_frame(frame)
        prepared_frame = self._matcher.prepare(frame.image)

        best_by_template: dict[str, tuple[SceneTemplate, float]] = {}
        for template in self._templates:
            confidence = self._matcher.similarity(prepared_frame, self._prepared[template.source_id])
            current = best_by_template.get(template.template_id)
            if current is None or confidence > current[1]:
                best_by_template[template.template_id] = (template, confidence)

        results = [
            SceneResult(
                scene_type=template.scene_type,
                skin=template.skin,
                scene_id=template.scene_id,
                confidence=confidence,
                template_id=template.template_id,
            )
            for template, confidence in best_by_template.values()
            if confidence >= self._config.threshold
        ]
        results.sort(key=lambda item: item.confidence, reverse=True)

        logger.debug(
            "场景分类完成：候选场景 %d 个，命中 %d 个（阈值 %.2f）",
            len(best_by_template),
            len(results),
            self._config.threshold,
        )
        return results

    def best_match(self, frame: Frame) -> SceneResult | None:
        """返回置信度最高的场景。

        Args:
            frame: 待识别的帧。

        Returns:
            最高置信度的场景结果；无命中时返回 ``None``。

        Raises:
            TypeError: ``frame`` 不是 :class:`~src.vision.capture.models.Frame`。
        """
        results = self.classify(frame)
        return results[0] if results else None

    @staticmethod
    def _validate_frame(frame: Frame) -> None:
        """校验输入类型。

        Raises:
            TypeError: 输入不是 :class:`~src.vision.capture.models.Frame`。
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"classify 需要 Frame 输入，实际为 {type(frame).__name__}")
