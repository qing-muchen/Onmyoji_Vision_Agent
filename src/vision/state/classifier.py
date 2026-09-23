"""状态分类器：把「证据」或「帧」映射为游戏状态（Phase 3.0；Phase 4.6.3 起支持证据输入）。

本类有两条输入路径，共用同一个 :class:`~src.vision.state.models.StateResult` 输出：

1. **证据驱动（Phase 4.6.3，推荐）**——消费证据层产出的
   :class:`~src.vision.evidence.models.EvidenceResult` 序列::

       [EvidenceResult, ...]
         ↓  EvidenceAggregator.aggregate（可靠度加权 + group 去重；规则来自配置）
         ↓  StateClassifier.classify_evidence
       StateResult(state_name, confidence, source=EVIDENCE, reason, evidence)

   语义映射**全部来自配置规则**（``evidence.aggregator.rules``），本模块不含
   "证据名 → 状态"的硬编码分支；本类也不读像素、不调用 detector、不调用 OCR。

2. **帧驱动（Phase 3.0；Phase 4.6.5 起只判转场，兼容保留）**——直接消费
   :class:`~src.vision.capture.models.Frame`::

       Frame
         ↓  TransitionClassifier（底部条带 + 正负样本）——需要整帧像素
         ↓  Fallback → unknown

   **庭院已迁出帧路径**：``yard`` 的唯一权威来源是证据路径
   （``bottom_menu_yard`` 证据 + ``evidence.aggregator.rules`` 的 yard 规则）。
   帧路径因此不会再返回 ``yard``（即使实例装配了
   :class:`~src.vision.state.yard.classifier.YardClassifier`）——它返回 ``unknown``
   并在 ``reason`` / ``detail`` 说明"帧路径的庭院判定已禁用，请使用证据路径"。
   帧路径本身保留，是因为转场判定需要整帧像素，且 ``StateMonitor`` 等旧调用方仍可能传 ``Frame``。
   被装配的 ``YardClassifier`` 作为**独立视觉组件**保留（见 ``vision.state.yard`` 模块文档），
   不再作为 ``StateResult`` 生产者。

两条路径的职责划分：**证据层回答"看到了什么"，本类回答"这些证据意味着什么"**。
本模块不延迟、不等待、不做状态转移、不触发任何交互。
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.config import AppConfig, EvidenceConfig, StateConfig
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.evidence import EvidenceAggregator, EvidenceResult
from src.vision.scene.loader import SceneLoader
from src.vision.state.base import BaseStateClassifier
from src.vision.state.models import MIN_CONFIDENCE, StateName, StateResult, StateSource
from src.vision.state.transition.classifier import TransitionClassifier
from src.vision.state.transition.dataset import TransitionSampleLoader
from src.vision.state.yard.classifier import YardClassifier

__all__ = ["FRAME_YARD_DISABLED_REASON", "StateClassifier"]

#: 帧路径不再判定庭院时给出的理由（Phase 4.6.5）。
#:
#: 庭院判定的权威来源是证据路径（``bottom_menu_yard`` 证据 + 配置规则）；
#: 帧路径命中庭院场景时也必须显式说明"这里不再负责"，而不是静默返回 unknown。
FRAME_YARD_DISABLED_REASON: str = (
    "frame based yard classification disabled; use evidence path"
)

logger = get_logger("vision.state.classifier")


class StateClassifier(BaseStateClassifier):
    """状态分类器：支持「证据驱动」与「帧驱动」两种输入。

    职责边界（Phase 4.6.5）：

    * **证据驱动（权威路径）**——把 :class:`~src.vision.evidence.models.EvidenceResult`
      序列交给 :class:`~src.vision.evidence.aggregator.EvidenceAggregator` 聚合（规则来自配置），
      判定包括 ``yard`` 在内的全部状态；不读像素、不调用 detector / OCR、不生成证据。
    * **帧驱动（兼容路径）**——只判转场（``TransitionClassifier`` 需要整帧像素），
      其余情况返回 ``unknown``；**不再判定 ``yard``**（庭院已收敛到证据路径，
      见 :meth:`classify_frame`）。

    Args:
        transition: 转场分支（帧驱动）；``None`` 表示未启用。
        yard: 庭院视觉组件；``None`` 表示未装配。**它不参与状态判定**（Phase 4.6.5 起），
            仅作为可独立使用的视觉组件保留（离线分析 / 后续其它视觉任务）。
        aggregator: 证据聚合器（证据驱动）；``None`` 表示本实例不接受证据输入。

    Raises:
        ValueError: transition / yard / aggregator 全部未提供（没有任何可用判定路径）。
    """

    def __init__(
        self,
        transition: TransitionClassifier | None = None,
        yard: YardClassifier | None = None,
        *,
        aggregator: EvidenceAggregator | None = None,
    ) -> None:
        if transition is None and yard is None and aggregator is None:
            raise ValueError(
                "StateClassifier 至少需要一个可用分支（transition / yard / aggregator）"
            )
        self._transition = transition
        self._yard = yard
        self._aggregator = aggregator

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        state_loader: TransitionSampleLoader | None = None,
        scene_loader: SceneLoader | None = None,
    ) -> StateClassifier:
        """按应用配置装配状态分类器。

        Phase 4.6.5 起，``state.yard`` 只决定"是否装配独立的庭院视觉组件"：装配出来的
        :class:`~src.vision.state.yard.classifier.YardClassifier` **不参与状态判定**
        （庭院状态的权威来源是证据路径），可通过 :attr:`yard` 属性单独使用/查询。

        Args:
            config: 应用配置（``state`` 与 ``scene`` 两段）。
            state_loader: 自定义转场样本加载器（便于测试或指定数据目录）。
            scene_loader: 自定义场景加载器（装配庭院视觉组件用）。

        Returns:
            状态分类器实例。

        Raises:
            StateDataError: 启用了转场识别但没有正样本。
            SceneLoadError: 启用了庭院视觉组件但场景目录不存在。
            ValueError: transition / yard / aggregator 全部未提供。
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

    @classmethod
    def from_evidence_config(cls, config: EvidenceConfig) -> StateClassifier:
        """按证据层配置装配**证据驱动**分类器（Phase 4.6.3）。

        与 :meth:`from_config` 的区别：本方法**不加载任何图像数据集**（不构造 transition /
        yard 分支），只装配证据聚合器，因此适合"证据由
        :class:`~src.vision.evidence.registry.EvidenceRegistry` + detector 产出"的流水线。
        状态映射全部来自 :attr:`EvidenceConfig.aggregator` 的规则（配置驱动，本类不含硬编码）。

        Args:
            config: 证据层配置（``AppConfig.evidence``）。

        Returns:
            只接受证据序列输入的分类器（``transition`` / ``yard`` 均为 ``None``）。

        Raises:
            ValueError: 配置无法构造聚合器（例如规则非法）。
        """
        aggregator = EvidenceAggregator(config.aggregator)
        logger.info(
            "证据驱动状态分类器就绪：rules=%d, fallback=%s",
            len(aggregator.rules),
            aggregator.fallback_state,
        )
        return cls(aggregator=aggregator)

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
        """庭院视觉组件；未装配时为 ``None``。

        Phase 4.6.5 起它**不再参与状态判定**（:meth:`classify_frame` 不会调用它）：
        庭院状态的唯一权威来源是证据路径（``bottom_menu_yard`` + 配置规则）。
        保留该组件供独立使用（离线分析、后续其它视觉任务）。
        """
        return self._yard

    @property
    def aggregator(self) -> EvidenceAggregator | None:
        """证据聚合器（证据驱动）；未配置时为 ``None``（此时不接受证据输入）。"""
        return self._aggregator

    def classify(self, input: Frame | Sequence[EvidenceResult]) -> StateResult:
        """判断当前状态（同一入口兼容「帧输入」与「证据输入」）。

        * 输入 :class:`~src.vision.capture.models.Frame` → :meth:`classify_frame`
          （帧路径：Phase 4.6.5 起只判转场，庭院改由证据路径判定）；
        * 输入 :class:`~src.vision.evidence.models.EvidenceResult` 序列
          （``list`` / ``tuple``，可以是空序列）→ :meth:`classify_evidence`
          （Phase 4.6.3 的证据驱动路径）。

        Args:
            input: 待识别的帧，或证据结果序列。

        Returns:
            状态结果。

        Raises:
            TypeError: 输入既不是 :class:`Frame` 也不是证据序列（例如裸 numpy 图像数组）。
            ValueError: 证据路径不可用（实例未配置 aggregator），或聚合产出的状态名非法。
        """
        if isinstance(input, Frame):
            return self.classify_frame(input)
        if isinstance(input, Sequence) and not isinstance(input, (str, bytes)):
            return self.classify_evidence(input)
        raise TypeError(
            f"classify 需要 Frame 或 EvidenceResult 序列输入，实际为 {type(input).__name__}"
        )

    def classify_frame(self, frame: Frame) -> StateResult:
        """由**帧**判断当前游戏状态（Phase 4.6.5 起只判转场）。

        帧路径只负责转场（底部条带 + 正负样本），因为该判定需要整帧像素；
        庭院（``yard``）已迁出帧路径——它的唯一权威来源是证据路径
        （``bottom_menu_yard`` 证据 + ``evidence.aggregator.rules`` 的 yard 规则）。

        因此本方法**不会**返回 :attr:`~src.vision.state.models.StateName.YARD`：
        即使实例装配了 :class:`~src.vision.state.yard.classifier.YardClassifier`，
        该组件也不会被调用，结果会是 ``unknown``，且 ``reason`` 为
        :data:`FRAME_YARD_DISABLED_REASON`（说明"请使用证据路径"）。

        Args:
            frame: 待识别的帧（不需要 ``coordinate_transform``）。

        Returns:
            ``transition``（条带命中）或 ``unknown``（其余情况）。

        Raises:
            TypeError: ``frame`` 不是 :class:`~src.vision.capture.models.Frame`。
        """
        self._validate_frame(frame)
        diagnosed: list[str] = []
        reason = ""

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
            # Phase 4.6.5：庭院判定的权威来源是证据路径（bottom_menu_yard + 配置规则）。
            # 帧路径不再调用 YardClassifier——这里只记录"为什么不判"，绝不据此返回 yard。
            diagnosed.append(FRAME_YARD_DISABLED_REASON)
            reason = FRAME_YARD_DISABLED_REASON

        detail = "; ".join(diagnosed) if diagnosed else "no state branch available"
        logger.debug("未匹配任何状态：%s", detail)
        return StateResult(
            state_name=StateName.UNKNOWN,
            confidence=MIN_CONFIDENCE,
            source=StateSource.FALLBACK,
            detail=detail,
            reason=reason,
        )

    def classify_evidence(self, evidence: Sequence[EvidenceResult]) -> StateResult:
        """由**视觉证据**判断当前状态（Phase 4.6.3）。

        语义映射全部来自配置规则（``evidence.aggregator.rules``）；计分（可靠度加权）、
        空间去重（同一 group 只取一条代表）与规则匹配都由
        :class:`~src.vision.evidence.aggregator.EvidenceAggregator` 负责，本方法只把
        聚合结果翻译成 :class:`StateResult`——因此不存在"证据名 → 状态"的硬编码分支。

        输入应当是**全量证据集**（``EvidenceRegistry.detect_all`` 的输出，含未命中项）：
        未命中的证据不参与计分，但缺失条目会让解释信息不完整。

        Args:
            evidence: 证据结果序列（``list`` / ``tuple``，可以为空）。

        Returns:
            状态结果。``source`` 恒为 :attr:`~src.vision.state.models.StateSource.EVIDENCE`；
            ``confidence`` 是**状态可信度**（Phase 4.6.6：聚合器按规则校准后的值，可能高于
            证据得分——见 ``evidence.aggregator.rules[*].state_confidence``）；
            命中规则时 ``evidence`` 为参与计分的证据名（按聚合器排序：置信度降序），
            没有可用证据时为 ``()`` 且 ``confidence`` 为 ``0.0``、``reason`` 为
            ``"no evidence available"``。发生校准（状态可信度高于证据得分）时，``reason``
            会追加 ``"; calibrated state confidence"``。

        Raises:
            ValueError: 实例未配置 aggregator（既未走 :meth:`from_evidence_config`，
                也未注入 ``aggregator``），或聚合产出的状态名不在
                :class:`~src.vision.state.models.StateName` 中（配置加载阶段的校验
                已拦截该情况，此处属于兜底失败）。
            TypeError: 序列中存在非 :class:`EvidenceResult` 元素。
        """
        if self._aggregator is None:
            raise ValueError(
                "StateClassifier 未配置 evidence aggregator："
                "请使用 from_evidence_config(config.evidence) 或注入 aggregator"
            )

        items = tuple(evidence)
        for index, item in enumerate(items):
            if not isinstance(item, EvidenceResult):
                raise TypeError(
                    f"evidence[{index}] 必须是 EvidenceResult，实际为 {type(item).__name__}"
                )

        if not items:
            logger.debug("证据为空，无法判定状态")
            return StateResult(
                state_name=StateName.UNKNOWN,
                confidence=MIN_CONFIDENCE,
                source=StateSource.EVIDENCE,
                detail=None,
                reason="no evidence available",
                evidence=(),
            )

        candidate = self._aggregator.aggregate(items)
        rule_matched = candidate.confidence > MIN_CONFIDENCE
        reason = (
            f"matched {candidate.state} evidence rule: {candidate.reason}"
            if rule_matched
            else f"no evidence rule matched: {candidate.reason}"
        )
        if candidate.is_calibrated:
            # 状态可信度被规则校准抬高（证据得分低于规则声明的下限）
            reason += "; calibrated state confidence"
        logger.debug("证据驱动判定：%s", reason)
        return StateResult(
            state_name=candidate.state,
            # Phase 4.6.6：StateResult.confidence 是**状态可信度**（校准后）；
            # 证据得分见 candidate.confidence（仅用于规则满足判定与解释）
            confidence=candidate.state_confidence,
            source=StateSource.EVIDENCE,
            detail=None,
            reason=reason,
            evidence=tuple(item.name for item in candidate.evidence),
        )

    @staticmethod
    def _validate_frame(frame: Frame) -> None:
        """校验输入类型。

        Raises:
            TypeError: 输入不是 :class:`~src.vision.capture.models.Frame`。
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"classify 需要 Frame 输入，实际为 {type(frame).__name__}")
