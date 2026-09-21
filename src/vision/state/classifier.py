"""状态分类器：把各状态分支组合成三态判定（Phase 3.0）。

判定顺序（先强特征、后弱特征）::

    Frame
      ↓  TransitionClassifier（底部条带 + 正负样本）——最强、最专用
      ↓  YardClassifier（复用场景识别）
      ↓  Fallback → unknown

为什么 transition 优先：转场底部条带是**专属强特征**（深棕底纹 + 官网提示文字），而
场景识别是整图相似度，对"庭院风格"的插画可能给出偏高的分数；反过来条带匹配几乎不会
对庭院界面（底部是普通 UI 栏）给高分。先判 transition 可避免"转场插画被误判为庭院"。

本模块只做**视觉状态识别**：不延迟、不等待、不做状态转移、不触发任何交互。
"""

from __future__ import annotations

from src.core.config import AppConfig, StateConfig
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.scene.loader import SceneLoader
from src.vision.state.base import BaseStateClassifier
from src.vision.state.models import StateName, StateResult, StateSource
from src.vision.state.transition.classifier import TransitionClassifier
from src.vision.state.transition.dataset import TransitionSampleLoader
from src.vision.state.yard.classifier import YardClassifier

__all__ = ["StateClassifier"]

logger = get_logger("vision.state.classifier")


class StateClassifier(BaseStateClassifier):
    """基于"转场底部条带 + 庭院场景"的三态状态分类器。

    Args:
        transition: 转场分支；``None`` 表示未启用。
        yard: 庭院分支；``None`` 表示未启用。

    Raises:
        ValueError: 两个分支都未提供（无法判定任何状态）。
    """

    def __init__(
        self,
        transition: TransitionClassifier | None = None,
        yard: YardClassifier | None = None,
    ) -> None:
        if transition is None and yard is None:
            raise ValueError("StateClassifier 至少需要一个可用分支（transition 或 yard）")
        self._transition = transition
        self._yard = yard

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        state_loader: TransitionSampleLoader | None = None,
        scene_loader: SceneLoader | None = None,
    ) -> StateClassifier:
        """按应用配置装配状态分类器。

        Args:
            config: 应用配置（``state`` 与 ``scene`` 两段）。
            state_loader: 自定义转场样本加载器（便于测试或指定数据目录）。
            scene_loader: 自定义场景加载器。

        Returns:
            状态分类器实例。

        Raises:
            StateDataError: 启用了转场识别但没有正样本。
            SceneLoadError: 启用了庭院识别但场景目录不存在。
            ValueError: 两个分支都被关闭。
        """
        state_config: StateConfig = config.state
        transition = (
            TransitionClassifier.from_dataset(state_config.transition, loader=state_loader)
            if state_config.transition.enabled
            else None
        )
        yard = (
            YardClassifier.from_config(state_config.yard, config.scene.matcher, loader=scene_loader)
            if state_config.yard.enabled
            else None
        )
        logger.info(
            "状态分类器就绪：transition=%s(%d/%d), yard=%s",
            transition is not None,
            transition.positive_count if transition is not None else 0,
            transition.negative_count if transition is not None else 0,
            yard is not None,
        )
        return cls(transition, yard)

    @property
    def name(self) -> str:
        """分类器名称。"""
        return "state"

    @property
    def transition(self) -> TransitionClassifier | None:
        """转场分支；未启用时为 ``None``。"""
        return self._transition

    @property
    def yard(self) -> YardClassifier | None:
        """庭院分支；未启用时为 ``None``。"""
        return self._yard

    def classify(self, frame: Frame) -> StateResult:
        """判断当前帧所处的游戏状态。

        Args:
            frame: 待识别的帧（不需要 ``coordinate_transform``）。

        Returns:
            三态结果之一：``transition`` / ``yard`` / ``unknown``。

        Raises:
            TypeError: ``frame`` 不是 :class:`~src.vision.capture.models.Frame`。
        """
        self._validate_frame(frame)
        diagnosed: list[str] = []

        if self._transition is not None:
            match = self._transition.match(frame)
            if match.is_transition:
                return StateResult(
                    state_name=StateName.TRANSITION,
                    confidence=match.confidence,
                    source=StateSource.TRANSITION_BOTTOM_MATCH,
                    detail=(
                        f"matched template {match.best_positive_file}"
                        f" (positive={match.positive_score:.4f}, negative={match.negative_score:.4f})"
                    ),
                )
            diagnosed.append(
                f"transition positive={match.positive_score:.4f}"
                f" negative={match.negative_score:.4f} threshold={match.threshold:.2f}"
            )

        if self._yard is not None:
            scene = self._yard.classify(frame)
            if scene is not None:
                return StateResult(
                    state_name=StateName.YARD,
                    confidence=scene.confidence,
                    source=StateSource.SCENE_CLASSIFIER,
                    detail=(
                        f"{scene.scene_key} template={scene.template_id}"
                        f" (threshold={self._yard.config.threshold:.2f})"
                    ),
                )
            diagnosed.append(f"yard below threshold {self._yard.config.threshold:.2f}")

        detail = "; ".join(diagnosed) if diagnosed else "no state branch available"
        logger.debug("未匹配任何状态：%s", detail)
        return StateResult(
            state_name=StateName.UNKNOWN,
            confidence=0.0,
            source=StateSource.FALLBACK,
            detail=detail,
        )

    @staticmethod
    def _validate_frame(frame: Frame) -> None:
        """校验输入类型。

        Raises:
            TypeError: 输入不是 :class:`~src.vision.capture.models.Frame`。
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"classify 需要 Frame 输入，实际为 {type(frame).__name__}")
