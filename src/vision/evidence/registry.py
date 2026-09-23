"""证据检测器注册表（Phase 4.1）。

统一管理 detector 的注册与批量执行::

    registry = EvidenceRegistry()
    registry.register(BattleAutoDetector("battle_auto_button", config))
    results = registry.detect_all(frame)

新增 detector **不需要修改核心流程**：写一个
:class:`~src.vision.evidence.base.BaseEvidenceDetector` 子类，注册进来即可。
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from src.core.exceptions import EvidenceError
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.evidence.base import BaseEvidenceDetector
from src.vision.evidence.models import EvidenceResult

__all__ = ["DetectorProfile", "EvidenceRegistry", "frame_fingerprint"]

logger = get_logger("vision.evidence.registry")


def frame_fingerprint(frame: Frame) -> bytes:
    """计算帧**内容指纹**（Phase 5.1，用于检测缓存）。

    指纹只由像素内容与形状/类型决定（**不含时间戳**）：内容相同 → 结果必然相同，
    因此缓存复用不会改变任何证据/状态；内容不同 → 指纹必然不同。

    Args:
        frame: 待计算指纹的帧。

    Returns:
        16 字节的 ``blake2b`` 摘要。
    """
    image = np.ascontiguousarray(frame.image)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(image.shape).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(image.tobytes())
    return digest.digest()


@dataclass(frozen=True, slots=True)
class DetectorProfile:
    """单个 detector 的耗时与静态信息（Phase 5.1，仅在 ``profile=True`` 时采集）。

    Attributes:
        name: detector（证据）名。
        elapsed_ms: 本次检测耗时（毫秒）。
        detected: 是否命中该证据。
        detector_type: detector 类名（区分模板型 / 包装型）。
        templates: 已加载的模板数量（非模板型为 0）。
        roi: ROI 的像素尺寸描述（例如 ``1706x192``）；非 ROI 型为空字符串。
        scales: 实际参与搜索的 scale 序列（非模板型为空元组）。
    """

    name: str
    elapsed_ms: float
    detected: bool
    detector_type: str
    templates: int
    roi: str
    scales: tuple[float, ...]

    def describe(self) -> str:
        """返回可读描述（用于 ``debug`` 级别的性能输出）。"""
        scales = f"{len(self.scales)}" if self.scales else "n/a"
        return (
            f"{self.name:<26} type={self.detector_type:<26} templates={self.templates:<2} "
            f"roi={self.roi or 'n/a':<10} scales={scales:<3} time={self.elapsed_ms:7.2f}ms "
            f"hit={self.detected}"
        )


class EvidenceRegistry:
    """证据检测器的注册表。

    Args:
        detectors: 初始 detector 序列（可选）。
        cache_size: 证据缓存容量（Phase 5.1）；``0``（默认）表示**关闭缓存**，
            行为与 Phase 5 完全一致。开启后按**帧内容指纹**复用检测结果
            （内容相同的帧不会重复检测；不会按时间缓存，也不会跨内容复用）。

    Raises:
        EvidenceError: 初始序列中存在重名或空名的 detector。
    """

    def __init__(
        self,
        detectors: Sequence[BaseEvidenceDetector] = (),
        *,
        cache_size: int = 0,
    ) -> None:
        self._detectors: dict[str, BaseEvidenceDetector] = {}
        self.register_all(detectors)
        self._cache_size = max(0, cache_size)
        self._cache: OrderedDict[bytes, tuple[EvidenceResult, ...]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_profile: tuple[DetectorProfile, ...] | None = None

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

    def detect_all(self, frame: Frame, *, profile: bool = False) -> list[EvidenceResult]:
        """对所有启用的 detector 执行检测。

        结果顺序与注册顺序一致；某个 detector 内部异常会直接抛出——证据缺失必须
        让调用方看到，而不是静默当成"画面里没有该证据"。

        Phase 5.1 新增两项**默认关闭**的能力（都不改变结果）：

        * ``profile=True``：逐个 detector 计时并采集静态信息（模板数 / ROI 像素尺寸 /
          scale 数），写入 :attr:`last_profile` 并打一条 ``debug`` 日志——
          用于定位"该优化谁"，**不是**永久打印；
        * 构造时 ``cache_size > 0``：按**帧内容指纹**复用检测结果（内容相同的帧不重复
          检测；不按时间缓存）。

        Args:
            frame: 待检测的帧。
            profile: 是否采集耗时画像（默认 ``False``，零额外开销）。

        Returns:
            证据结果列表（可能包含 ``detected=False`` 的条目）。
        """
        use_cache = self._cache_size > 0
        cache_key: bytes | None = None
        if use_cache:
            cache_key = frame_fingerprint(frame)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                self._cache.move_to_end(cache_key)
                logger.debug(
                    "证据缓存命中：%s（累计命中 %d）", cache_key.hex()[:12], self._cache_hits
                )
                return list(cached)
            self._cache_misses += 1

        results = self._detect_all_uncached(frame, profile=profile)

        if use_cache and cache_key is not None:
            self._cache[cache_key] = tuple(results)
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return results

    def _detect_all_uncached(self, frame: Frame, *, profile: bool) -> list[EvidenceResult]:
        """真正执行检测（缓存未命中或未启用缓存时）；``profile`` 时采集画像。"""
        results: list[EvidenceResult] = []
        profiles: list[DetectorProfile] = []
        for detector in self._detectors.values():
            if not detector.enabled:
                logger.debug("跳过已停用的 detector: %s", detector.name)
                continue
            if profile:
                started = time.perf_counter()
                result = detector.detect(frame)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                profiles.append(self._profile_entry(detector, frame, elapsed_ms, result.detected))
            else:
                result = detector.detect(frame)
            results.append(result)
        hit = sum(1 for item in results if item.detected)
        logger.debug("证据检测完成：detector=%d, 命中=%d", len(results), hit)
        if profile:
            self._last_profile = tuple(profiles)
            logger.debug("证据检测画像：\n%s", self.profile_summary())
        return results

    @staticmethod
    def _profile_entry(
        detector: BaseEvidenceDetector,
        frame: Frame,
        elapsed_ms: float,
        detected: bool,
    ) -> DetectorProfile:
        """采集单个 detector 的静态信息与耗时（缺失属性按"无"处理）。"""
        templates = tuple(getattr(detector, "templates", ()) or ())
        roi = getattr(detector, "roi", None)
        roi_label = ""
        if roi is not None:
            rect = roi.to_rect(frame.width, frame.height)
            roi_label = f"{rect.width}x{rect.height}"
        matcher_config = getattr(detector, "matcher_config", None)
        scales = tuple(getattr(matcher_config, "effective_scales", ()) or ())
        return DetectorProfile(
            name=detector.name,
            elapsed_ms=elapsed_ms,
            detected=detected,
            detector_type=type(detector).__name__,
            templates=len(templates),
            roi=roi_label,
            scales=scales,
        )

    @property
    def last_profile(self) -> tuple[DetectorProfile, ...] | None:
        """最近一次 ``profile=True`` 的检测画像；未采集时为 ``None``。"""
        return self._last_profile

    def profile_summary(self) -> str:
        """把最近一次画像压成可读多行文本（按耗时降序，末行是合计）。"""
        if not self._last_profile:
            return "(no profile collected; call detect_all(..., profile=True))"
        ordered = sorted(self._last_profile, key=lambda item: item.elapsed_ms, reverse=True)
        lines = [item.describe() for item in ordered]
        total = sum(item.elapsed_ms for item in ordered)
        lines.append(f"{'TOTAL':<26} time={total:7.2f}ms")
        return "\n".join(lines)

    @property
    def cache_size(self) -> int:
        """证据缓存容量（``0`` = 关闭）。"""
        return self._cache_size

    @property
    def cache_hits(self) -> int:
        """缓存命中次数。"""
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """缓存未命中次数。"""
        return self._cache_misses

    def clear_cache(self) -> None:
        """清空证据缓存（不清空命中/未命中统计）。"""
        self._cache.clear()

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
