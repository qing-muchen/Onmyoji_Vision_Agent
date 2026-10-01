"""Phase 11.6-A 测试：证据 detector 的 ROI 结果缓存（``cache_size``）。

用户约束（本阶段只加缓存）：**不改匹配算法、不增加模板、不改配置**。
因此这里的测试全部围绕"缓存只是跳过重复计算"：

1. 默认容量 ``0`` = 关闭 ⇒ 统计为空、行为与没有缓存时一致；
2. 命中返回的结果与"关闭缓存"逐字段**完全相同**（detected / confidence / source / bbox /
   metadata / reliability / group / evidence_type）；
3. 命中时**不再调用**匹配路径（``best_match`` 调用次数不增加）⇒ 确实跳过了 matchTemplate；
4. 键语义保守：不同帧、**单像素变化**、**帧尺寸变化**都必须不命中（宁可重算，不给错结果）；
5. LRU 淘汰（容量上限被遵守）；
6. 未检出（``detected=False``）的结果同样可缓存且语义一致；
7. ``roi_fingerprint`` 只由 ROI 像素内容决定；
8. 带缓存的完整识别链与不带缓存给出**同一个 StateResult**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.core.config import EvidenceDetectorConfig
from src.core.paths import get_project_root
from src.vision.capture.models import Frame
from src.vision.evidence.base import (
    LruEvidenceCache,
    TemplateEvidenceDetector,
    roi_fingerprint,
)
from src.vision.evidence.models import EvidenceResult, EvidenceType
from src.vision.evidence.templates import load_feature_templates

ROOT = get_project_root()


class _ProbeDetector(TemplateEvidenceDetector):
    """测试用模板型 detector（不依赖 dataset 模板目录）。"""

    NAME = "probe"
    EVIDENCE_TYPE = EvidenceType.CUSTOM


def write_png(path: Path, image: np.ndarray) -> Path:
    """写真实 PNG（非 ASCII 路径安全）。"""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def noise_image(seed: int, *, size: int = 24, channel: int | None = None) -> np.ndarray:
    """确定性噪声图；``channel`` 指定时只改某个通道的一个像素（单像素扰动）。"""
    rng = np.random.default_rng(seed)
    image = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    if channel is not None:
        image[0, 0, channel] = int((int(image[0, 0, channel]) + 7) % 256)
    return image


def frame_of(image: np.ndarray, *, timestamp: float = 1.0) -> Frame:
    """把图像包成帧（``Frame.create`` 会复制，便于构造"同内容不同对象"的帧）。"""
    return Frame.create(image, source="test", timestamp=timestamp)


def make_detector(
    tmp_path: Path, *, cache_size: int, template: np.ndarray | None = None, threshold: float = 0.5
) -> _ProbeDetector:
    """构造模板型 detector（ROI = 整帧、单尺度，便于确定性比较）。"""
    write_png(tmp_path / "probe_cap_001.png", template if template is not None else noise_image(1))
    config = EvidenceDetectorConfig(
        roi=(0.0, 0.0, 1.0, 1.0),
        templates=str(tmp_path),
        threshold=threshold,
        multi_scale=False,
    )
    templates = tuple(load_feature_templates(tmp_path, "probe", limit=1, root=tmp_path))
    assert len(templates) == 1
    return _ProbeDetector("probe", config, templates, cache_size=cache_size)


def assert_same_result(first: EvidenceResult, second: EvidenceResult) -> None:
    """断言两条证据结果逐字段一致（缓存不得改变任何判定结果）。"""
    assert first.detected == second.detected
    assert first.confidence == pytest.approx(second.confidence)
    assert first.source == second.source
    assert first.evidence_type == second.evidence_type
    assert first.reliability == pytest.approx(second.reliability)
    assert first.group == second.group
    assert (first.bbox is None) == (second.bbox is None)
    if first.bbox is not None and second.bbox is not None:
        assert first.bbox.describe() == second.bbox.describe()
    assert dict(first.metadata) == dict(second.metadata)
    assert first.describe() == second.describe()


# ---------------------------------------------------------------------------
# 1 / 2 / 3. 默认关闭、等价性、真的跳过匹配
# ---------------------------------------------------------------------------


def test_cache_disabled_by_default(tmp_path: Path) -> None:
    """默认 ``cache_size=0``：不启用、无统计、行为与无缓存一致。"""
    detector = make_detector(tmp_path, cache_size=0)
    frame = frame_of(noise_image(1))

    first = detector.detect(frame)
    second = detector.detect(frame)

    assert detector.cache_size == 0
    assert detector.cache_stats() == {
        "enabled": False,
        "capacity": 0,
        "size": 0,
        "hits": 0,
        "misses": 0,
        "hit_rate": 0.0,
    }
    assert_same_result(first, second)


def test_cache_hit_matches_uncached_result_and_skips_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """命中结果与无缓存逐字段一致，且命中时**不再调用** ``best_match``。"""
    cached_detector = make_detector(tmp_path, cache_size=4, threshold=0.999)
    plain_detector = make_detector(tmp_path, cache_size=0, threshold=0.999)
    image = noise_image(1)
    frame = frame_of(image)

    reference = plain_detector.detect(frame)

    calls = {"count": 0}
    original = cached_detector.best_match

    def counting_best_match(target: Frame):  # noqa: ANN202 - 测试用间谍
        calls["count"] += 1
        return original(target)

    monkeypatch.setattr(cached_detector, "best_match", counting_best_match)

    first = cached_detector.detect(frame)
    calls_after_first = calls["count"]
    second = cached_detector.detect(frame)

    assert calls_after_first == 1  # 第一次真的算了一次
    assert calls["count"] == 1  # 第二次完全跳过了匹配路径
    assert_same_result(reference, first)
    assert_same_result(first, second)
    stats = cached_detector.cache_stats()
    assert stats["enabled"] is True and stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. 键语义保守：不同内容 / 单像素 / 不同尺寸都不命中
# ---------------------------------------------------------------------------


def test_cache_misses_on_different_frames(tmp_path: Path) -> None:
    """不同画面（同一尺寸）不命中。"""
    detector = make_detector(tmp_path, cache_size=4)

    detector.detect(frame_of(noise_image(1)))
    detector.detect(frame_of(noise_image(2)))

    stats = detector.cache_stats()
    assert stats["misses"] == 2 and stats["hits"] == 0
    assert stats["size"] == 2


def test_cache_misses_on_single_pixel_change(tmp_path: Path) -> None:
    """单像素变化**不命中**（宁可重算，也不复用可能过期的结果）。"""
    detector = make_detector(tmp_path, cache_size=4)
    base = noise_image(9)
    tweaked = noise_image(9, channel=0)

    detector.detect(frame_of(base))
    detector.detect(frame_of(tweaked))

    assert detector.cache_stats()["hits"] == 0


def test_cache_key_includes_frame_size(tmp_path: Path) -> None:
    """帧尺寸不同必然不命中（``bbox`` 位于帧坐标系，不能跨尺寸复用）。"""
    detector = make_detector(tmp_path, cache_size=4)
    image = noise_image(3)
    smaller = image[::2, ::2]

    first = detector.detect(frame_of(image))
    second = detector.detect(frame_of(smaller))

    assert detector.cache_stats()["hits"] == 0
    if first.bbox is not None and second.bbox is not None:
        # 两次都是真实计算的结果（对应各自尺寸），不是跨尺寸复用
        assert second.bbox.right <= smaller.shape[1] + 1


# ---------------------------------------------------------------------------
# 5 / 6. LRU 上限与"未检出"结果
# ---------------------------------------------------------------------------


def test_cache_evicts_least_recently_used(tmp_path: Path) -> None:
    """容量上限被遵守：超出后淘汰最久未使用的一条。"""
    detector = make_detector(tmp_path, cache_size=2)
    frames = [frame_of(noise_image(100 + index)) for index in range(3)]

    for frame in frames:
        detector.detect(frame)
    assert detector.cache_stats()["size"] == 2  # 只保留 2 条

    # 重新访问最早那条 ⇒ 必须重新计算（已被淘汰）
    detector.cache_stats()  # 触发一次读取（无副作用）
    detector.detect(frames[0])
    assert detector.cache_stats()["hits"] == 0

    # 但刚写入的两条仍在缓存里（访问第 3 条 ⇒ 命中）
    detector.detect(frames[2])
    assert detector.cache_stats()["hits"] == 1


def test_cache_caches_miss_results_too(tmp_path: Path) -> None:
    """``detected=False`` 的结果同样被缓存，且语义与未缓存一致。"""
    # 模板与画面完全不同 ⇒ 阈值 0.999 下必然未检出
    detector = make_detector(tmp_path, cache_size=4, template=noise_image(1), threshold=0.999)
    plain = make_detector(tmp_path, cache_size=0, template=noise_image(1), threshold=0.999)
    frame = frame_of(noise_image(2))

    cached_first = detector.detect(frame)
    cached_second = detector.detect(frame)

    assert cached_first.detected is False
    assert_same_result(plain.detect(frame), cached_first)
    assert_same_result(cached_first, cached_second)
    assert detector.cache_stats()["hits"] == 1


# ---------------------------------------------------------------------------
# 7. 指纹与缓存对象本身
# ---------------------------------------------------------------------------


def test_roi_fingerprint_depends_only_on_pixels() -> None:
    """指纹只由 ROI 像素内容决定（同内容同指纹；不同内容/不同形状不同指纹）。"""
    first = noise_image(5)
    same = first.copy()
    other = noise_image(6)
    reshaped = np.zeros((48, 12, 3), dtype=np.uint8)

    assert roi_fingerprint(first) == roi_fingerprint(same)
    assert roi_fingerprint(first) != roi_fingerprint(other)
    assert roi_fingerprint(first) != roi_fingerprint(reshaped)


def test_lru_cache_statistics_and_clear() -> None:
    """缓存对象的统计与清空语义（``clear`` 保留统计，``reset_statistics`` 清零）。"""
    cache = LruEvidenceCache(2)
    cache.put(("a",), "value-a")
    assert cache.get(("a",)) == "value-a"
    assert cache.get(("b",)) is None
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 1

    cache.clear()
    assert cache.size == 0 and cache.stats()["hits"] == 1  # 统计保留
    cache.reset_statistics()
    assert cache.stats()["hits"] == 0 and cache.stats()["misses"] == 0

    disabled = LruEvidenceCache(0)
    disabled.put(("a",), "value-a")
    assert disabled.get(("a",)) is None
    assert disabled.stats()["enabled"] is False and disabled.stats()["misses"] == 0


# ---------------------------------------------------------------------------
# 8. 完整识别链的等价性（缓存不改变状态判定）
# ---------------------------------------------------------------------------


def test_cached_chain_matches_uncached_state_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一帧上，带 ROI 缓存的识别链与不带缓存给出**同一个 StateResult**。"""
    from src.vision.evidence import EvidenceRegistry
    from src.vision.pipeline.frame_processor import (
        EvidenceFrameProcessor,
        build_transition_probe,
    )
    from src.vision.state import StateClassifier

    from src.core.config import load_config

    dataset_yard = ROOT / "dataset" / "raw" / "yard"
    images = sorted(dataset_yard.glob("*.png"))[:2]
    if len(images) < 2:
        pytest.skip("缺少 yard 原图")
    import cv2

    config = load_config(environ={})
    frames = []
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None
        frames.append(Frame.create(image, source=path.name, timestamp=1.0))

    def build(cache_size: int) -> EvidenceFrameProcessor:
        from src.vision.evidence import create_detectors

        detectors = create_detectors(
            config.evidence, transition_classifier=build_transition_probe(config)
        )
        if cache_size > 0:
            for detector in detectors:
                if isinstance(detector, TemplateEvidenceDetector):
                    detector._cache = LruEvidenceCache(cache_size)  # noqa: SLF001 - 测试注入
        registry = EvidenceRegistry(detectors)
        classifier = StateClassifier.from_evidence_config(config.evidence)
        return EvidenceFrameProcessor(registry, classifier)

    plain = build(0)
    cached = build(8)

    for frame in frames:
        plain_result = plain.process(frame)
        cached_first = cached.process(frame)
        cached_second = cached.process(frame)  # 第二次走缓存
        assert cached_first.state_name == plain_result.state_name
        assert cached_first.confidence == pytest.approx(plain_result.confidence)
        assert cached_second.state_name == plain_result.state_name
        assert cached_second.confidence == pytest.approx(plain_result.confidence)

    # 缓存确实被用上了（同内容帧重复出现 ⇒ 至少一次命中）
    yard_detector = cached.registry.find("bottom_menu_yard")
    assert yard_detector is not None
    assert yard_detector.cache_stats()["hits"] >= 1
