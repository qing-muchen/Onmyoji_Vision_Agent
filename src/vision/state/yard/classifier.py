"""庭院状态识别（Phase 3.0）：复用场景分类器。

"当前是不是庭院界面"本质上就是 Phase 2.2 的**整图场景识别**问题，因此本模块只做
**包装**：把判定阈值换成 ``state.yard.threshold``，其余全部复用
:class:`~src.vision.scene.classifier.SceneClassifier`，不复制任何场景匹配代码。

阈值约定：庭院判定以 ``state.yard.threshold`` 为准，它会覆盖
``scene.matcher.threshold``（state 层是判定方，避免两处阈值含义重叠）。

状态判定的权威归属（Phase 4.6.5）——**本组件不再是状态判定来源**：

* 庭院状态（``yard``）的唯一权威来源是**证据路径**：``bottom_menu_yard`` 证据
  （``dataset/features/bottom_menu/yard`` 的稳定 UI 结构）+ ``evidence.aggregator.rules``
  的 yard 规则。它依据"底部功能栏的结构"，不随庭院皮肤 / 布局变化而漂移。
* 本组件（整图场景相似度）依赖庭院背景 / 皮肤 / 场景模板，容易受皮肤变化与布局移动影响，
  因此退出 ``StateResult`` 生产链；保留为**独立视觉组件**，可用于离线分析或后续其它
  视觉任务（例如判断"是否处于某个特定皮肤"）。
* :meth:`~src.vision.state.classifier.StateClassifier.classify_frame` 不再调用本组件
  （返回 ``unknown`` 并在 ``reason`` 说明原因）；需要单独使用时，通过
  :attr:`~src.vision.state.classifier.StateClassifier.yard` 或直接构造本类。

``SceneClassifier`` 与 ``dataset/scenes`` 数据**未删除**（Phase 4.6.5 只改变状态解释权）。
"""

from __future__ import annotations

from src.core.config import SceneMatcherConfig, StateYardConfig
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.scene.classifier import SceneClassifier
from src.vision.scene.loader import SceneLoader
from src.vision.scene.result import SceneResult

__all__ = ["YardClassifier"]

logger = get_logger("vision.state.yard.classifier")


class YardClassifier:
    """把场景分类结果解释为"是否处于庭院"。

    .. deprecated:: Phase 4.6.5
       本组件**不再作为 ``StateResult`` 的生产者**——庭院状态的权威来源是证据路径
       （``bottom_menu_yard`` 证据 + ``evidence.aggregator.rules`` 的 yard 规则）。
       保留为独立视觉组件，供离线分析或后续其它视觉任务使用；接口与行为未改动。

    Args:
        config: 状态识别配置中的 ``yard`` 段（阈值）。
        classifier: 已构建的场景分类器（复用其匹配能力）。
    """

    def __init__(self, config: StateYardConfig, classifier: SceneClassifier) -> None:
        self._config = config
        self._classifier = classifier

    @classmethod
    def from_config(
        cls,
        config: StateYardConfig,
        scene_config: SceneMatcherConfig,
        *,
        loader: SceneLoader | None = None,
    ) -> YardClassifier:
        """按配置构造庭院分类器。

        Args:
            config: 状态识别配置中的 ``yard`` 段。
            scene_config: 场景匹配配置（``AppConfig.scene.matcher``）；其中的
                ``threshold`` 会被 ``state.yard.threshold`` 覆盖。
            loader: 自定义场景加载器；省略时使用默认场景目录。

        Returns:
            庭院分类器实例。

        Raises:
            SceneLoadError: 场景目录不存在或图像无法读取。
            SceneTemplateError: 场景目录层级或命名不符合规范。
            ValueError: 过滤后没有任何场景模板。
        """
        effective = scene_config.model_copy(update={"threshold": config.threshold})
        classifier = SceneClassifier.from_dataset(effective, loader=loader)
        logger.info(
            "庭院分类器就绪：场景模板 %d 个，threshold=%.2f（覆盖 scene.matcher.threshold）",
            len(classifier.templates),
            config.threshold,
        )
        return cls(config, classifier)

    @property
    def name(self) -> str:
        """分类器名称。"""
        return "yard"

    @property
    def config(self) -> StateYardConfig:
        """当前配置。"""
        return self._config

    @property
    def classifier(self) -> SceneClassifier:
        """底层场景分类器（复用对象，便于查看模板数量）。"""
        return self._classifier

    @property
    def template_count(self) -> int:
        """可用的场景模板数量。"""
        return len(self._classifier.templates)

    def classify(self, frame: Frame) -> SceneResult | None:
        """判断帧是否处于庭院场景。

        Args:
            frame: 待识别的帧。

        Returns:
            命中的场景结果；低于阈值时返回 ``None``。

        Raises:
            TypeError: ``frame`` 不是 :class:`~src.vision.capture.models.Frame`。
        """
        result = self._classifier.best_match(frame)
        if result is None:
            logger.debug("庭院判定：无场景达到阈值 %.2f", self._config.threshold)
            return None
        logger.debug("庭院判定命中：%s", result.describe())
        return result
