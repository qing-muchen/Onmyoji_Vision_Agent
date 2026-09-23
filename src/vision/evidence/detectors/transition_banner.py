"""转场条带证据（Phase 4.6.4）。

职责：只回答"当前画面里是否存在**转场底部条带**这一视觉事实"，**不回答**
"当前状态是不是 transition"——状态映射由 ``evidence.aggregator.rules`` 完成。

判定逻辑**不在这里重新实现**：本 detector 包装 Phase 3.0 的
:class:`~src.vision.state.transition.classifier.TransitionClassifier`（底部条带裁剪 +
正样本阈值 + 负样本否决），只把它的判定翻译成一条 :class:`EvidenceResult`。

依赖方向（Phase 4.5 冻结）：``vision.evidence`` **不得依赖** ``vision.state``。
本模块因此**不在运行期导入** ``src.vision.state``——需要判定的对象由**装配层**
（``create_detectors(..., transition_classifier=...)`` 的调用方，即编排层）注入，
本模块只按鸭子类型调用它的 ``match(frame)``；相关类型注解在 ``TYPE_CHECKING`` 下引入。
这样可以避免 ``vision.state.classifier`` → ``vision.evidence`` →
``vision.state.transition`` 的循环导入。

数据缺失时的行为（Phase 4.6.4 明确要求）：未注入判定器（例如
``dataset/states/transition_bottom`` 不可用）时返回 ``detected=False``，并在
``metadata`` 中记录 ``missing_dataset=True`` 与原因——**不伪造命中、不降低阈值**。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.core.config import EvidenceConfig, EvidenceDetectorConfig
from src.core.exceptions import EvidenceError
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.evidence.base import BaseEvidenceDetector
from src.vision.evidence.models import (
    DEFAULT_RELIABILITY,
    MIN_CONFIDENCE,
    EvidenceResult,
    EvidenceSource,
    EvidenceType,
)

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型注解；运行期不导入 state 层（避免循环依赖）
    from src.vision.state.transition.classifier import TransitionClassifier, TransitionMatch

__all__ = ["TransitionBannerDetector", "TransitionProbe"]

logger = get_logger("vision.evidence.detectors.transition_banner")


class TransitionProbe(Protocol):
    """转场判定器的最小接口（鸭子类型）。

    生产实现是 :class:`~src.vision.state.transition.classifier.TransitionClassifier`；
    测试可以注入任意满足该接口的替身。
    """

    def match(self, frame: Frame) -> TransitionMatch:
        """判断帧是否处于转场界面。"""
        ...


class TransitionBannerDetector(BaseEvidenceDetector):
    """报告"画面里是否有转场底部条带"的证据检测器。

    这是**包装型** detector：自身不加载模板、不做图像比较，判定完全交给注入的
    :class:`TransitionProbe`（生产实现为 ``TransitionClassifier``）。因此
    ``evidence.detectors.transition_banner`` 只需要 ``enabled`` 开关，
    判定参数（裁剪比例、阈值、负样本间隔）仍由 ``state.transition`` 统一提供，
    不存在第二份阈值。

    Args:
        name: 证据名称（``transition_banner``）。
        config: detector 配置（只使用 ``enabled``）。
        probe: 转场判定器；``None`` 表示当前没有可用数据/未接线（此时输出"未检出"并记录原因）。
        reliability: 该类证据的长期可靠程度（来自 ``evidence.reliability``）。
        group: 该证据观察的视觉区域名（未分组时为空字符串）。
    """

    #: detector 名（同时作为配置键、证据名）
    NAME: str = "transition_banner"
    #: 证据粗分类
    EVIDENCE_TYPE: EvidenceType = EvidenceType.TRANSITION

    @classmethod
    def from_config(
        cls,
        config: EvidenceConfig,
        *,
        probe: TransitionProbe | None = None,
    ) -> TransitionBannerDetector:
        """按证据层配置构造 detector。

        Args:
            config: 证据层配置（提供开关与可靠度）。
            probe: 转场判定器（由装配层构造并注入）；``None`` 表示数据缺失/未接线。

        Returns:
            detector 实例。

        Raises:
            EvidenceError: 配置中缺少 :data:`NAME` 对应的条目。
        """
        detector_config = config.detectors.get(cls.NAME)
        if detector_config is None:
            raise EvidenceError(f"evidence.detectors 缺少 {cls.NAME!r} 的配置")
        if probe is None:
            logger.warning(
                "转场证据无可用判定器（dataset/states/transition_bottom 缺失或未注入）："
                "%s 将始终输出未检出",
                cls.NAME,
            )
        return cls(
            cls.NAME,
            detector_config,
            probe=probe,
            reliability=config.reliability_for(cls.NAME),
            group=config.group_of(cls.NAME) or "",
        )

    def __init__(
        self,
        name: str,
        config: EvidenceDetectorConfig,
        *,
        probe: TransitionProbe | None = None,
        reliability: float = DEFAULT_RELIABILITY,
        group: str = "",
    ) -> None:
        self._name = name
        self._config = config
        self._probe = probe
        self._reliability = reliability
        self._group = group

    @property
    def name(self) -> str:
        """证据名称。"""
        return self._name

    @property
    def enabled(self) -> bool:
        """该 detector 是否启用。"""
        return self._config.enabled

    @property
    def config(self) -> EvidenceDetectorConfig:
        """当前配置。"""
        return self._config

    @property
    def evidence_type(self) -> EvidenceType:
        """该 detector 输出的证据分类。"""
        return self.EVIDENCE_TYPE

    @property
    def reliability(self) -> float:
        """该类证据的长期可靠程度（写入每条证据）。"""
        return self._reliability

    @property
    def group(self) -> str:
        """该证据观察的视觉区域名（空字符串 = 未分组）。"""
        return self._group

    @property
    def probe(self) -> TransitionProbe | None:
        """当前注入的转场判定器；未接线时为 ``None``。"""
        return self._probe

    def detect(self, frame: Frame) -> EvidenceResult:
        """报告帧中是否存在转场条带证据。

        Args:
            frame: 待检测的帧（**整帧**；条带裁剪由判定器按 ``state.transition`` 配置完成）。

        Returns:
            证据结果：判定为转场条带时 ``detected=True`` 且置信度为条带正样本相似度；
            否则 ``detected=False``（含"无可用判定器"的情况，原因写入 ``metadata``）。
        """
        if not self._config.enabled:
            return self.miss(reason="detector disabled")

        if self._probe is None:
            return self.miss(
                reason="transition evidence unavailable",
                missing_dataset=True,
                hint="dataset/states/transition_bottom 不可用或未注入判定器",
            )

        match = self._probe.match(frame)
        metadata: dict = {
            "positive_score": match.positive_score,
            "negative_score": match.negative_score,
            "threshold": match.threshold,
            "negative_margin": match.negative_margin,
            "best_positive_file": match.best_positive_file,
        }
        if not match.is_transition or match.confidence <= MIN_CONFIDENCE:
            return self.miss(reason="transition banner not matched", **metadata)

        return EvidenceResult(
            name=self._name,
            detected=True,
            confidence=match.confidence,
            bbox=None,  # 条带位置由判定器内部处理；证据只需报告"是否存在"
            source=EvidenceSource.ROI,
            metadata=metadata,
            evidence_type=self.EVIDENCE_TYPE,
            reliability=self._reliability,
            group=self._group,
        )

    def miss(self, *, reason: str, **extra: object) -> EvidenceResult:
        """组装一条"未检出"证据（含原因与判定细节）。"""
        metadata: dict = {"reason": reason}
        metadata.update(extra)
        return EvidenceResult.miss(
            self._name,
            metadata=metadata,
            evidence_type=self.EVIDENCE_TYPE,
            reliability=self._reliability,
            group=self._group,
        )

    def describe(self) -> str:
        """返回可读描述（含是否已接线）。"""
        return f"{self._name} (enabled={self.enabled}, probe={'yes' if self._probe else 'no'})"
