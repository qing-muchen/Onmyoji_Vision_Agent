"""证据检测器注册表（Phase 4.1）。

统一管理 detector 的注册与批量执行::

    registry = EvidenceRegistry()
    registry.register(BattleAutoDetector("battle_auto_button", config))
    results = registry.detect_all(frame)

新增 detector **不需要修改核心流程**：写一个
:class:`~src.vision.evidence.base.BaseEvidenceDetector` 子类，注册进来即可。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from src.core.exceptions import EvidenceError
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.evidence.base import BaseEvidenceDetector
from src.vision.evidence.models import EvidenceResult

__all__ = ["EvidenceRegistry"]

logger = get_logger("vision.evidence.registry")


class EvidenceRegistry:
    """证据检测器的注册表。

    Args:
        detectors: 初始 detector 序列（可选）。

    Raises:
        EvidenceError: 初始序列中存在重名或空名的 detector。
    """

    def __init__(self, detectors: Sequence[BaseEvidenceDetector] = ()) -> None:
        self._detectors: dict[str, BaseEvidenceDetector] = {}
        self.register_all(detectors)

    def register(self, detector: BaseEvidenceDetector) -> None:
        """注册一个 detector。

        Args:
            detector: 待注册的 detector。

        Raises:
            EvidenceError: 名称为空，或该名称已被注册（不做静默覆盖）。
        """
        name = getattr(detector, "name", "")
        if not isinstance(name, str) or not name.strip():
            raise EvidenceError(f"detector.name 必须是非空字符串，实际为 {name!r}")
        if name in self._detectors:
            raise EvidenceError(
                f"detector 名称重复: {name!r}（已注册 {type(self._detectors[name]).__name__}，"
                f"新注册 {type(detector).__name__}）"
            )
        self._detectors[name] = detector
        logger.debug("已注册证据 detector: %s", name)

    def register_all(self, detectors: Sequence[BaseEvidenceDetector]) -> None:
        """批量注册。

        Args:
            detectors: 待注册的 detector 序列。
        """
        for detector in detectors:
            self.register(detector)

    def unregister(self, name: str) -> None:
        """注销一个 detector。

        Args:
            name: detector 名。

        Raises:
            KeyError: 该名称未注册。
        """
        if name not in self._detectors:
            raise KeyError(f"未注册的 detector: {name!r}")
        del self._detectors[name]
        logger.debug("已注销证据 detector: %s", name)

    def get(self, name: str) -> BaseEvidenceDetector:
        """按名称取 detector。

        Args:
            name: detector 名。

        Returns:
            对应的 detector。

        Raises:
            KeyError: 该名称未注册。
        """
        if name not in self._detectors:
            raise KeyError(
                f"未注册的 detector: {name!r}（已注册: {', '.join(self._detectors) or '无'}）"
            )
        return self._detectors[name]

    def find(self, name: str) -> BaseEvidenceDetector | None:
        """按名称取 detector，未注册时返回 ``None``。"""
        return self._detectors.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        """已注册的 detector 名（注册顺序）。"""
        return tuple(self._detectors)

    @property
    def detectors(self) -> tuple[BaseEvidenceDetector, ...]:
        """已注册的 detector（注册顺序）。"""
        return tuple(self._detectors.values())

    @property
    def enabled_detectors(self) -> tuple[BaseEvidenceDetector, ...]:
        """已注册且启用的 detector。"""
        return tuple(detector for detector in self._detectors.values() if detector.enabled)

    def detect_all(self, frame: Frame) -> list[EvidenceResult]:
        """对所有启用的 detector 执行检测。

        结果顺序与注册顺序一致；某个 detector 内部异常会直接抛出——证据缺失必须
        让调用方看到，而不是静默当成"画面里没有该证据"。

        Args:
            frame: 待检测的帧。

        Returns:
            证据结果列表（可能包含 ``detected=False`` 的条目）。
        """
        results: list[EvidenceResult] = []
        for detector in self._detectors.values():
            if not detector.enabled:
                logger.debug("跳过已停用的 detector: %s", detector.name)
                continue
            results.append(detector.detect(frame))
        hit = sum(1 for item in results if item.detected)
        logger.debug("证据检测完成：detector=%d, 命中=%d", len(results), hit)
        return results

    def detect_map(self, frame: Frame) -> dict[str, EvidenceResult]:
        """执行检测并返回 ``证据名 -> 结果`` 的映射（便于按名称查证）。

        Args:
            frame: 待检测的帧。

        Returns:
            证据名到结果的映射。
        """
        return {result.name: result for result in self.detect_all(frame)}

    def describe(self) -> str:
        """返回可读描述。"""
        return f"EvidenceRegistry({len(self._detectors)} detectors: {', '.join(self._detectors) or '无'})"

    def __len__(self) -> int:
        return len(self._detectors)

    def __iter__(self) -> Iterator[BaseEvidenceDetector]:
        return iter(self._detectors.values())

    def __contains__(self, name: object) -> bool:
        return name in self._detectors

    def __repr__(self) -> str:
        return f"EvidenceRegistry(detectors={list(self._detectors)})"
