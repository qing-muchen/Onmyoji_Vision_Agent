"""基于 OCR 文字的检测器：把"固定文字"当作 UI 元素的语义锚点。

为什么需要它：探索按钮的外观随皮肤 / 活动主题剧烈变化（灯笼、圆球、兔子、不同配色），
纯像素模板匹配无法稳定命中；但按钮上的文字（``探索``）是固定的，因此用 OCR 作为
**语义来源**，与模板（视觉定位来源）互补。

数据流::

    Frame
      ↓  OCREngine.recognize(frame)（复用 src/vision/ocr，不复制任何 OCR 代码）
    list[TextResult]（reference 坐标）
      ↓  与配置中的 OCRTextTarget 文字比对
    list[DetectionResult]（category / element_name 来自目标配置，rect 直接复用 TextResult.rect）

本模块只做"文字 -> 元素"的映射，不解读文字含义、不做决策、不产生任何交互。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.core.config import OCRConfig, OCRTextDetectorConfig, OCRTextTarget
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.detector.base import Detector
from src.vision.detector.result import DetectionResult
from src.vision.ocr.base import BaseOCREngine
from src.vision.ocr.engine import OCREngine
from src.vision.ocr.result import TextResult

__all__ = ["OCR_TEMPLATE_PREFIX", "OCRTextDetector", "normalize_text"]

#: OCR 结果的 ``template_id`` 前缀（OCR 命中没有模板，用该前缀标注来源）
OCR_TEMPLATE_PREFIX: str = "ocr:"
#: 所有空白字符（Tesseract 常在汉字之间插入空格，比对前统一清理）
WHITESPACE_PATTERN = re.compile(r"\s+")

logger = get_logger("vision.detector.ocr.detector")


def normalize_text(text: str) -> str:
    """规范化文本用于比对：去掉全部空白字符。

    OCR 输出常出现 ``探 索`` / ``探索 `` 这类空白差异，若直接做 ``==`` 比对会漏检。

    Args:
        text: 原始文本。

    Returns:
        去掉所有空白后的文本。
    """
    return WHITESPACE_PATTERN.sub("", text)


class OCRTextDetector(Detector):
    """把 OCR 识别到的目标文字转换为 :class:`DetectionResult` 的检测器。

    坐标说明：``OCREngine`` 已把结果统一转换到**参考分辨率坐标系**，因此本检测器
    **不做二次坐标变换**（再做一次会得到错误坐标），``rect`` 直接复用 ``TextResult.rect``。

    Args:
        engine: OCR 引擎（复用 ``src/vision/ocr``，不重新实现识别逻辑）。
        config: OCR 辅助检测配置（文字 -> 元素的映射表）。
    """

    def __init__(self, engine: BaseOCREngine, config: OCRTextDetectorConfig) -> None:
        self._engine = engine
        self._config = config
        self._targets = tuple(config.targets)
        #: 规范化文字 -> 目标（配置中重复文字时后者生效，并记录一条告警）
        self._targets_by_text: dict[str, OCRTextTarget] = {}
        for target in self._targets:
            key = normalize_text(target.text)
            if key in self._targets_by_text:
                logger.warning("OCR 文字目标重复，后者生效: %s", target.text)
            self._targets_by_text[key] = target

    @classmethod
    def from_config(
        cls,
        config: OCRTextDetectorConfig,
        ocr_config: OCRConfig,
        *,
        engine: BaseOCREngine | None = None,
    ) -> OCRTextDetector:
        """按配置构造检测器。

        Args:
            config: OCR 辅助检测配置（文字 -> 元素映射表）。
            ocr_config: OCR 引擎配置（``AppConfig.ocr``），用于创建引擎。
            engine: 直接注入的 OCR 引擎；省略时按 ``ocr_config`` 创建
                （便于测试或复用已建实例）。

        Returns:
            检测器实例。
        """
        active_engine = engine if engine is not None else OCREngine.from_config(ocr_config)
        return cls(active_engine, config)

    @property
    def name(self) -> str:
        """检测器名称。"""
        return "ocr"

    @property
    def engine(self) -> BaseOCREngine:
        """当前使用的 OCR 引擎。"""
        return self._engine

    @property
    def config(self) -> OCRTextDetectorConfig:
        """当前配置。"""
        return self._config

    @property
    def targets(self) -> tuple[OCRTextTarget, ...]:
        """已配置的文字目标。"""
        return self._targets

    def detect(self, frame: Frame) -> list[DetectionResult]:
        """在帧中通过 OCR 寻找配置里声明的目标文字。

        Args:
            frame: 待检测的帧；必须携带 ``coordinate_transform``（由 OCR 引擎校验）。

        Returns:
            检测结果列表（按 ``confidence`` 降序）；无目标命中时返回空列表。

        Raises:
            MissingTransformError: 帧未携带坐标映射。
            OCRProviderUnavailableError: OCR 引擎不可用（明确报错，不静默返回空结果）。
            OCRError: 识别过程失败。
        """
        if not self._targets_by_text:
            logger.debug("未配置 OCR 文字目标，跳过 OCR 检测")
            return []

        matches = self._engine.recognize(frame)
        results: list[DetectionResult] = []
        for match in matches:
            target = self._targets_by_text.get(normalize_text(match.text))
            if target is None:
                continue
            results.append(self._to_detection(target, match))

        results.sort(key=lambda item: item.confidence, reverse=True)
        logger.debug("OCR 检测完成：文本块 %d 个 -> 命中目标 %d 项", len(matches), len(results))
        return results

    @staticmethod
    def _to_detection(target: OCRTextTarget, match: TextResult) -> DetectionResult:
        """把文字结果转换为检测结果（语义来自配置，区域直接复用 OCR 结果）。

        Args:
            target: 命中的文字目标。
            match: OCR 识别结果（reference 坐标）。

        Returns:
            与模板检测同构的检测结果。
        """
        return DetectionResult(
            element_name=target.element_name,
            category=target.category,
            confidence=match.confidence,
            rect=match.rect,
            template_id=f"{OCR_TEMPLATE_PREFIX}{match.text}",
        )

    @staticmethod
    def matched_texts(results: Sequence[DetectionResult]) -> list[str]:
        """从检测结果中还原命中的文字（便于日志与调试输出）。

        Args:
            results: :meth:`detect` 的输出。

        Returns:
            命中文字列表（按原顺序）；``template_id`` 不含 OCR 前缀的条目会被跳过。
        """
        return [
            result.template_id[len(OCR_TEMPLATE_PREFIX) :]
            for result in results
            if result.template_id.startswith(OCR_TEMPLATE_PREFIX)
        ]
