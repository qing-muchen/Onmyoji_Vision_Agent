"""帧到状态的证据流水线（Phase 4.7）。

把 Phase 4.6 建立的"证据驱动状态判定"封装为**一次完整判定**，为 Runtime 迁移做准备::

    Frame
      ↓  EvidenceRegistry.detect_all(frame)      # 证据层：看到了什么
    [EvidenceResult, ...]
      ↓  StateClassifier.classify(evidence)      # 状态层：这些证据意味着什么
    StateResult

本模块**只做编排**，不做任何视觉判定，也不做时间相关的事：

* **不做时间滤波**——唯一归属是
  :class:`~src.runtime.state_monitor.stabilizer.StateStabilizer`（运行时层）；
* **不做循环 / 节拍 / 采集**——属于 ``StateMonitor`` 与 ``FrameSource``（Phase 5）；
* **不读文件**——模板在 detector 的构造期加载，``detect`` 只消费内存与帧像素。

为什么需要它：Phase 4.6 之后 :class:`~src.vision.state.classifier.StateClassifier`
既能接收 ``Frame`` 也能接收证据，但 ``StateMonitor`` 仍调用 ``classify(frame)``
（帧路径自 4.6.5 起只剩转场）。本类是两个世界之间的**适配器**：Runtime 只需持有一个
:class:`EvidenceFrameProcessor`，即可在不改动自身主流程的前提下切换到证据驱动判定
（Phase 5 接线）。
"""

from __future__ import annotations

from pathlib import Path

from src.core.config import AppConfig
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.evidence import (
    EvidenceRegistry,
    EvidenceResult,
    TransitionProbe,
    create_detectors,
)
from src.vision.state import StateClassifier, StateResult

__all__ = ["EvidenceFrameProcessor"]

logger = get_logger("vision.pipeline.frame_processor")


class EvidenceFrameProcessor:
    """把一帧画面判定为一个 :class:`~src.vision.state.models.StateResult`。

    证据驱动、无时间滤波：同一个处理器的多次调用彼此独立（抖动抑制交给
    ``StateStabilizer``）。

    Args:
        registry: 证据检测器注册表（``Frame → [EvidenceResult]``）。
        classifier: 状态分类器（``[EvidenceResult] → StateResult``）；必须是**证据驱动**
            实例（``aggregator`` 不为 ``None``）。

    Raises:
        ValueError: ``classifier`` 未配置 evidence aggregator（无法消费证据）。
    """

    def __init__(self, registry: EvidenceRegistry, classifier: StateClassifier) -> None:
        if classifier.aggregator is None:
            raise ValueError(
                "EvidenceFrameProcessor 需要证据驱动的 StateClassifier（缺少 evidence aggregator）："
                "请使用 StateClassifier.from_evidence_config(...) 或 from_config(...) 装配"
            )
        self._registry = registry
        self._classifier = classifier

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        transition_classifier: TransitionProbe | None = None,
        ocr_engine: object | None = None,
        root: Path | None = None,
    ) -> EvidenceFrameProcessor:
        """按应用配置装配处理单元（证据 detector + 证据驱动分类器）。

        Args:
            config: 应用配置（``evidence`` 段提供 detector 与规则）。
            transition_classifier: 转场判定器（通常为 ``TransitionClassifier``）；
                由**编排层**构造后注入——本包不导入 ``vision.state.transition`` 的实现，
                以避免证据层与状态层之间的循环依赖（见
                :class:`~src.vision.evidence.detectors.transition_banner.TransitionBannerDetector`）。
            ocr_engine: 可选的 OCR 引擎（注入后启用文字锚点证据）。
            root: 解析模板相对路径的基准目录；省略时使用项目根。

        Returns:
            处理单元实例。
        """
        detectors = create_detectors(
            config.evidence,
            ocr_engine=ocr_engine,
            root=root,
            transition_classifier=transition_classifier,
        )
        registry = EvidenceRegistry(detectors)
        classifier = StateClassifier.from_evidence_config(config.evidence)
        logger.info(
            "证据流水线就绪：detectors=%d, rules=%d, transition=%s",
            len(registry),
            len(classifier.aggregator.rules) if classifier.aggregator is not None else 0,
            transition_classifier is not None,
        )
        return cls(registry, classifier)

    @property
    def registry(self) -> EvidenceRegistry:
        """证据检测器注册表。"""
        return self._registry

    @property
    def classifier(self) -> StateClassifier:
        """证据驱动状态分类器。"""
        return self._classifier

    def detect(self, frame: Frame) -> list[EvidenceResult]:
        """检测一帧的全部证据（含未命中项，顺序与注册顺序一致）。

        Args:
            frame: 待检测的帧。

        Returns:
            证据结果列表。
        """
        return self._registry.detect_all(frame)

    def process(self, frame: Frame) -> StateResult:
        """判定一帧的状态（``Frame → [EvidenceResult] → StateResult``）。

        Args:
            frame: 待判定的帧。

        Returns:
            状态结果；``source`` 为 ``StateSource.EVIDENCE``，并带有 ``reason`` / ``evidence``。

        Raises:
            ValueError: 证据集导致的状态名非法（配置校验已在加载阶段拦截，此处为兜底）。
            TypeError: 证据内部类型异常（detector 契约被破坏）。
        """
        evidence = self.detect(frame)
        result = self._classifier.classify(evidence)
        logger.debug("证据流水线判定：%s", result.describe())
        return result

    def describe(self) -> str:
        """返回可读描述。"""
        aggregator = self._classifier.aggregator
        rules = len(aggregator.rules) if aggregator is not None else 0
        return f"EvidenceFrameProcessor(detectors={len(self._registry)}, rules={rules})"

    def __repr__(self) -> str:
        return self.describe()
