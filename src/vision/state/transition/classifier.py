"""转场状态判定：正样本阈值 + 负样本否决（Phase 3.0）。

判定规则（阈值全部来自 ``state.transition`` 配置）::

    positive_score = max 相似度(帧底部条带, 各正样本)      # 条带 vs 已裁剪条带
    negative_score = max 相似度(帧底部条带, 各负样本底部)   # 条带 vs 整图裁出的底部
    is_transition  ⟺  positive_score >= threshold
                      且（存在负样本时）positive_score - negative_score >= negative_margin

"负样本否决"是必需的：普通界面（商店 / 式神录 / 战斗……）的底部同样可能是深色 UI 栏，
只看"像不像转场"会误判；因此额外要求"比最像的非转场界面更像转场"。

★ 样本与运行时输入统一表示：正样本（已是条带）、负样本（完整截图）与运行时帧都会经过
:meth:`~src.vision.state.transition.extractor.TransitionBottomExtractor.crop` ——
完整截图会被裁出底部条带，本来就是条带的输入保持原样（见 ``strip_aspect_ratio``）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.core.config import StateTransitionConfig
from src.core.exceptions import StateDataError
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.state.transition.dataset import TransitionSample, TransitionSampleLoader
from src.vision.state.transition.extractor import TransitionBottomExtractor
from src.vision.state.transition.matcher import TransitionBottomMatcher

__all__ = ["TransitionClassifier", "TransitionMatch"]

logger = get_logger("vision.state.transition.classifier")


@dataclass(frozen=True, slots=True)
class TransitionMatch:
    """一次转场判定的完整过程数据（即使未判定为转场也会返回，便于调参）。

    Attributes:
        is_transition: 是否判定为转场状态。
        confidence: 判定置信度（判定为转场时为 ``positive_score``，否则为 ``0.0``）。
        positive_score: 与正样本的最高相似度。
        negative_score: 与负样本的最高相似度（无负样本时为 ``0.0``）。
        best_positive_id: 最像的正样本 ID；无正样本时为 ``None``。
        best_positive_file: 最像的正样本文件名（用于 ``detail`` 展示）。
        best_negative_id: 最像的负样本 ID（无负样本时为 ``None``）。
        threshold: 本次使用的正样本阈值。
        negative_margin: 本次使用的负样本间隔。
    """

    is_transition: bool
    confidence: float
    positive_score: float
    negative_score: float
    best_positive_id: str | None
    best_positive_file: str | None
    best_negative_id: str | None
    threshold: float
    negative_margin: float

    def describe(self) -> str:
        """返回可读描述（含正/负分数与阈值，便于判断"差在哪一步"）。"""
        negative = f"{self.negative_score:.4f}" if self.best_negative_id else "n/a"
        return (
            f"transition={self.is_transition} positive={self.positive_score:.4f} "
            f"negative={negative} threshold={self.threshold:.2f} "
            f"margin={self.negative_margin:.2f} best={self.best_positive_file or 'none'}"
        )


class TransitionClassifier:
    """基于底部条带正/负样本的转场分类器。

    Args:
        config: 状态识别配置中的 ``transition`` 段。
        positives: 正样本（**已裁剪的底部条带**，直接使用）。
        negatives: 负样本（**完整界面截图**，使用前按配置比例裁出底部）。
        extractor: 自定义裁剪器；省略时按配置创建。
        matcher: 自定义匹配器；省略时按配置创建。

    Raises:
        StateDataError: 正样本为空（启用转场识别却没有可用样本）。
    """

    def __init__(
        self,
        config: StateTransitionConfig,
        positives: Sequence[TransitionSample],
        negatives: Sequence[TransitionSample],
        *,
        extractor: TransitionBottomExtractor | None = None,
        matcher: TransitionBottomMatcher | None = None,
    ) -> None:
        if not positives:
            raise StateDataError("转场识别已启用，但没有可用的正样本（transition_bottom 为空）")
        self._config = config
        self._extractor = extractor if extractor is not None else TransitionBottomExtractor(config)
        self._matcher = matcher if matcher is not None else TransitionBottomMatcher(config)
        self._files = {sample.sample_id: sample.file_name for sample in positives}
        # 统一表示：完整截图裁出底部条带，本身已是条带的正样本则保持原样（自适应）
        self._positives = {
            sample.sample_id: self._matcher.prepare(self._extractor.crop(sample.image))
            for sample in positives
        }
        self._negatives = {
            sample.sample_id: self._matcher.prepare(self._extractor.crop(sample.image))
            for sample in negatives
        }
        logger.info(
            "转场分类器就绪：正样本 %d 个，负样本 %d 个，crop_height_ratio=%.4f, 尺寸=%s",
            len(self._positives),
            len(self._negatives),
            config.crop_height_ratio,
            self._matcher.resize_size,
        )

    @classmethod
    def from_dataset(
        cls,
        config: StateTransitionConfig,
        *,
        loader: TransitionSampleLoader | None = None,
    ) -> TransitionClassifier:
        """从 ``dataset/states`` 构造分类器。

        Args:
            config: 状态识别配置中的 ``transition`` 段。
            loader: 自定义加载器；省略时使用默认样本目录。

        Returns:
            转场分类器实例。

        Raises:
            StateDataError: 正样本目录缺失/为空，或样本图像无法读取。
        """
        active_loader = loader if loader is not None else TransitionSampleLoader()
        positives, negatives = active_loader.load()
        return cls(config, positives, negatives)

    @property
    def name(self) -> str:
        """分类器名称。"""
        return "transition"

    @property
    def config(self) -> StateTransitionConfig:
        """当前配置。"""
        return self._config

    @property
    def positive_count(self) -> int:
        """正样本数量。"""
        return len(self._positives)

    @property
    def negative_count(self) -> int:
        """负样本数量。"""
        return len(self._negatives)

    def match(self, frame: Frame) -> TransitionMatch:
        """判断帧是否处于转场界面。

        Args:
            frame: 待识别的帧。

        Returns:
            判定过程数据（未判定为转场时 ``is_transition=False``，分数仍然有效）。

        Raises:
            TypeError: ``frame`` 不是 :class:`~src.vision.capture.models.Frame`。
        """
        _, strip = self._extractor.extract(frame)
        prepared = self._matcher.prepare(strip)

        positive = self._matcher.best_match(prepared, self._positives)
        negative = self._matcher.best_match(prepared, self._negatives)
        positive_score = positive[1] if positive is not None else 0.0
        negative_score = negative[1] if negative is not None else 0.0

        above_threshold = positive_score >= self._config.threshold
        # 没有负样本时不做间隔要求（退化为"仅正样本阈值"，并由加载器给出告警）
        margin_ok = (
            True
            if negative is None
            else (positive_score - negative_score) >= self._config.negative_margin
        )
        is_transition = above_threshold and margin_ok

        match = TransitionMatch(
            is_transition=is_transition,
            confidence=positive_score if is_transition else 0.0,
            positive_score=positive_score,
            negative_score=negative_score,
            best_positive_id=positive[0] if positive is not None else None,
            best_positive_file=(
                self._files.get(positive[0]) if positive is not None else None
            ),
            best_negative_id=negative[0] if negative is not None else None,
            threshold=self._config.threshold,
            negative_margin=self._config.negative_margin,
        )
        logger.debug("转场判定：%s", match.describe())
        return match
