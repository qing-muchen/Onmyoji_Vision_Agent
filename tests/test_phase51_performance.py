"""Phase 5.1 性能与健壮性测试。

覆盖（任务七）：

1. ``profile=True`` 不改变证据结果；
2. **优化等价性**：模板金字塔缓存（优化后）与“逐帧重算”（测试内朴素参考实现）**逐字段一致**；
3. 帧指纹缓存不改变状态（命中缓存与冷启动结果相同；不同内容不复用）；
4. 空帧 / 极小帧的异常处理（明确报错或不崩溃）；
5. detector 失败时 Runtime 继续运行（含 ``timing=True`` 路径）；
6. 模拟 500ms 固定节拍：clock 恰好推进 ``cycles × interval``，且 cycle timing 累积正确。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import EvidenceConfig, RuntimeStateMonitorConfig, load_config
from src.core.exceptions import EvidenceError
from src.core.geometry import Rect
from src.core.paths import resolve_from_project_root
from src.runtime.state_monitor import StateMonitor, StateStabilizer
from src.vision.capture import Frame
from src.vision.detector.models import TemplateInfo
from src.vision.detector.template.matcher import (
    LOWER_IS_BETTER_METHODS,
    TemplateMatch,
    TemplateMatcher,
)
from src.vision.detector.template.preprocess import resize_image, to_grayscale
from src.vision.evidence import EvidenceRegistry, create_detectors, load_feature_templates
from src.vision.pipeline import EvidenceFrameProcessor
from src.vision.pipeline.frame_processor import build_transition_probe
from src.vision.state import StateClassifier, StateName, StateSource
from tests.monitor_fakes import FakeClock, FakeFrameSource
from tests.test_runtime_pipeline import FailingDetector, ScriptedDetector

#: 真实数据（缺失时跳过依赖它们的用例）
YARD_FRAME = "dataset/raw/yard/yard_cap_001.png"
YARD_TEMPLATES = "dataset/features/bottom_menu/yard/positive"


def load_image(path: Path) -> np.ndarray:
    """读取图像（兼容非 ASCII 路径）。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def tiny_frame(value: int = 0, size: tuple[int, int] = (2, 2)) -> Frame:
    """构造一个极小的合成帧。"""
    height, width = size
    return Frame.create(np.full((height, width, 3), value, dtype=np.uint8), source="tiny")


def naive_match(
    frame_gray: np.ndarray,
    template: TemplateInfo,
    config: object,
) -> TemplateMatch | None:
    """**朴素参考实现**：Phase 5.1 优化前的匹配逻辑（每帧重算灰度与缩放）。

    用来证明"模板金字塔缓存"是纯性能优化——两个实现必须逐字段给出相同结果。
    """
    method_name = config.method  # type: ignore[attr-defined]
    threshold = config.threshold  # type: ignore[attr-defined]
    method = getattr(cv2, method_name)
    template_gray = to_grayscale(template.image)
    frame_height, frame_width = frame_gray.shape[:2]
    best: TemplateMatch | None = None

    for scale in config.effective_scales:  # type: ignore[attr-defined]
        scaled = resize_image(template_gray, scale)
        height, width = scaled.shape[:2]
        if height > frame_height or width > frame_width:
            continue
        result = cv2.matchTemplate(frame_gray, scaled, method)
        min_value, max_value, min_location, max_location = cv2.minMaxLoc(result)
        if method_name in LOWER_IS_BETTER_METHODS:
            confidence = min(1.0, max(0.0, 1.0 - float(min_value)))
            location = (int(min_location[0]), int(min_location[1]))
        else:
            confidence = min(1.0, max(0.0, float(max_value)))
            location = (int(max_location[0]), int(max_location[1]))
        if best is None or confidence > best.confidence:
            best = TemplateMatch(
                template=template,
                confidence=confidence,
                rect=Rect(x=location[0], y=location[1], width=width, height=height),
                scale=scale,
            )

    if best is None or best.confidence < threshold:
        return None
    return best


def build_registry(*, cache_size: int = 0) -> EvidenceRegistry:
    """按仓库配置装配真实 detector 注册表（模板缺失时返回空注册表）。"""
    config = load_config(environ={})
    detectors = create_detectors(config.evidence, transition_classifier=build_transition_probe(config))
    return EvidenceRegistry(detectors, cache_size=cache_size)


def build_processor(*, cache_size: int = 0) -> EvidenceFrameProcessor:
    """装配真实证据流水线（可选缓存）。"""
    config = load_config(environ={})
    return EvidenceFrameProcessor(
        build_registry(cache_size=cache_size),
        StateClassifier.from_evidence_config(config.evidence),
    )


def monitor_config(tmp_path: Path, *, interval_ms: int = 10) -> RuntimeStateMonitorConfig:
    """测试用监控配置（不写默认历史文件）。"""
    return RuntimeStateMonitorConfig(
        interval_ms=interval_ms,
        save_history=False,
        history_path=str(tmp_path / "state_history.json"),
    )


def stabilizer() -> StateStabilizer:
    """按仓库配置装配稳定器。"""
    return StateStabilizer(load_config(environ={}).runtime.stabilization)


@pytest.fixture(scope="module")
def matching_setup() -> tuple[TemplateMatcher, list[TemplateInfo], np.ndarray]:
    """真实模板 + 真实帧的灰度图（用于等价性对比）。"""
    config = load_config(environ={})
    frame_path = resolve_from_project_root(YARD_FRAME)
    directory = resolve_from_project_root(YARD_TEMPLATES)
    if not frame_path.is_file() or not directory.is_dir():
        pytest.skip("需要 dataset/raw 与 dataset/features 的模板")
    detector_config = config.evidence.detectors["bottom_menu_yard"]
    templates = load_feature_templates(
        YARD_TEMPLATES, "bottom_menu_yard", limit=detector_config.max_templates
    )
    if not templates:
        pytest.skip("无可用模板")
    matcher = TemplateMatcher(detector_config.matcher_config())
    frame_gray = TemplateMatcher.prepare(load_image(frame_path))
    return matcher, templates, frame_gray


# ---------------------------------------------------------------------------
# 2. 优化等价性（模板金字塔缓存 vs 朴素实现）
# ---------------------------------------------------------------------------


def test_template_pyramid_matches_naive_reference(
    matching_setup: tuple[TemplateMatcher, list[TemplateInfo], np.ndarray],
) -> None:
    """优化后的匹配结果与"逐帧重算"的朴素实现**逐字段完全一致**。"""
    matcher, templates, frame_gray = matching_setup
    config = matcher.config

    for template in templates:
        optimized = matcher.match(frame_gray, template)
        reference = naive_match(frame_gray, template, config)

        if reference is None:
            assert optimized is None, template.template_id
            continue
        assert optimized is not None, template.template_id
        assert optimized.confidence == reference.confidence
        assert optimized.rect == reference.rect
        assert optimized.scale == reference.scale
        assert optimized.template.template_id == reference.template.template_id


def test_template_pyramid_is_reused_across_calls(
    matching_setup: tuple[TemplateMatcher, list[TemplateInfo], np.ndarray],
) -> None:
    """同一模板第二次匹配命中金字塔缓存（并记录 scale 统计）。"""
    matcher, templates, frame_gray = matching_setup
    matcher.reset_statistics()
    before = matcher.pyramid_cache_hits

    first = matcher.match(frame_gray, templates[0])
    second = matcher.match(frame_gray, templates[0])

    assert matcher.pyramid_cache_hits > before
    assert first == second
    if first is not None:
        assert matcher.scale_hits.get(first.scale, 0) >= 2


# ---------------------------------------------------------------------------
# 1. profiling 不影响结果
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not resolve_from_project_root("dataset/features").is_dir(), reason="dataset/features 不存在"
)
def test_profiling_does_not_change_results() -> None:
    """``profile=True`` 只采集耗时；证据结果与默认路径完全一致。"""
    frame_path = resolve_from_project_root(YARD_FRAME)
    if not frame_path.is_file():
        pytest.skip("样本不存在")

    registry = build_registry()
    frame = Frame.create(load_image(frame_path), source="bench")

    baseline = [item.describe() for item in registry.detect_all(frame)]
    profiled = [item.describe() for item in registry.detect_all(frame, profile=True)]

    assert baseline == profiled
    assert registry.last_profile is not None
    assert len(registry.last_profile) == len(profiled)
    assert all(entry.elapsed_ms >= 0.0 for entry in registry.last_profile)
    assert all(entry.name for entry in registry.last_profile)
    # 画像里必须带可定位的信息（模板数 / ROI 尺寸）
    assert any(entry.templates > 0 for entry in registry.last_profile)
    assert any(entry.roi for entry in registry.last_profile)


def test_profile_summary_is_readable() -> None:
    """画像摘要包含每个 detector 与合计行（不是永久打印，只在取用时生成）。"""
    registry = EvidenceRegistry([ScriptedDetector([True])])

    assert "no profile" in registry.profile_summary()
    registry.detect_all(tiny_frame(), profile=True)
    summary = registry.profile_summary()

    assert "bottom_menu_yard" in summary
    assert "TOTAL" in summary


# ---------------------------------------------------------------------------
# 3. 帧指纹缓存
# ---------------------------------------------------------------------------


def test_frame_cache_does_not_change_state(tmp_path: Path) -> None:
    """缓存命中时返回完全相同的结果；不同内容的帧不被复用。"""
    frame_path = resolve_from_project_root(YARD_FRAME)
    if not frame_path.is_file():
        pytest.skip("样本不存在")

    processor = build_processor(cache_size=4)
    frame = Frame.create(load_image(frame_path), source="bench")

    first = processor.process(frame)
    second = processor.process(frame)

    assert first == second
    assert processor.registry.cache_hits == 1
    assert processor.registry.cache_misses == 1

    other = tiny_frame(value=17)
    processor.process(other)

    assert processor.registry.cache_misses == 2
    assert processor.registry.cache_hits == 1


def test_cache_is_disabled_by_default() -> None:
    """默认不启用缓存（行为与 Phase 5 一致）。"""
    registry = EvidenceRegistry([ScriptedDetector([True])])

    registry.detect_all(tiny_frame())

    assert registry.cache_size == 0
    assert registry.cache_hits == 0
    assert registry.cache_misses == 0


def test_fingerprint_distinguishes_content() -> None:
    """指纹由像素内容决定：内容不同 → 指纹不同。"""
    from src.vision.evidence.registry import frame_fingerprint

    first = frame_fingerprint(tiny_frame(value=1))
    second = frame_fingerprint(tiny_frame(value=1))
    changed = frame_fingerprint(tiny_frame(value=2))

    assert first == second
    assert first != changed


# ---------------------------------------------------------------------------
# 4. 空帧 / 极小帧
# ---------------------------------------------------------------------------


def test_empty_image_is_rejected() -> None:
    """空图像不能构造帧（明确报错，而不是产生无意义结果）。"""
    with pytest.raises(ValueError):
        Frame.create(np.zeros((0, 0, 3), dtype=np.uint8), source="empty")


def test_tiny_frame_does_not_crash(tmp_path: Path) -> None:
    """极小帧（2×2）在完整流水线上不崩溃：所有 detector 未检出 → unknown。"""
    processor = build_processor()

    result = processor.process(tiny_frame())

    assert result.state_name is StateName.UNKNOWN
    assert result.source is StateSource.EVIDENCE
    assert result.confidence == 0.0
    assert result.evidence == ()


def test_tiny_frame_with_timing(tmp_path: Path) -> None:
    """计时开启时极小帧同样安全（耗时画像可用）。"""
    processor = build_processor()

    processor.process(tiny_frame(), profile=True)

    assert processor.last_timing is not None
    assert processor.last_timing.total_ms >= 0.0


# ---------------------------------------------------------------------------
# 5. detector 失败时 Runtime 继续运行
# ---------------------------------------------------------------------------


def test_detector_failure_continues_with_timing(tmp_path: Path) -> None:
    """detector 抛异常 + 开启 timing：该帧降级为 unknown，循环继续且画像仍生成。"""
    registry = EvidenceRegistry([FailingDetector(EvidenceError("boom"))])
    processor = EvidenceFrameProcessor(registry, StateClassifier.from_evidence_config(EvidenceConfig()))
    monitor = StateMonitor(
        processor,
        FakeFrameSource(),
        monitor_config(tmp_path),
        stabilizer=stabilizer(),
        clock=FakeClock().clock,
        sleep=lambda _seconds: None,
        timing=True,
    )

    report = monitor.run(max_cycles=2)

    assert report.classified == 2
    assert monitor.errors == 2
    assert monitor.last_result is not None
    assert monitor.last_result.state_name is StateName.UNKNOWN
    assert "evidence pipeline failure" in monitor.last_result.reason
    assert monitor.last_timing is not None


# ---------------------------------------------------------------------------
# 6. 模拟 500ms 固定节拍 + cycle timing
# ---------------------------------------------------------------------------


def test_simulated_500ms_loop(tmp_path: Path) -> None:
    """模拟 500ms 节拍：clock 恰好推进 ``cycles × interval``，cycle timing 累积正确。"""
    clock = FakeClock()
    processor = EvidenceFrameProcessor(
        EvidenceRegistry([ScriptedDetector([True])]),
        StateClassifier.from_evidence_config(EvidenceConfig()),
    )
    monitor = StateMonitor(
        processor,
        FakeFrameSource(),
        monitor_config(tmp_path, interval_ms=500),
        stabilizer=stabilizer(),
        clock=clock.clock,
        sleep=clock.sleep,
        timing=True,
    )

    report = monitor.run(max_cycles=4)

    assert report.cycles == 4
    assert clock.value == pytest.approx(4 * 0.5, abs=1e-6)
    assert monitor.last_timing is not None
    assert monitor.last_timing.cycle == 4
    assert monitor.last_timing.total_ms >= 0.0
    assert monitor.last_timing.state is StateName.YARD


def test_timing_is_off_by_default(tmp_path: Path) -> None:
    """默认不采集耗时画像（零额外开销）。"""
    monitor = StateMonitor(
        EvidenceFrameProcessor(
            EvidenceRegistry([ScriptedDetector([True])]),
            StateClassifier.from_evidence_config(EvidenceConfig()),
        ),
        FakeFrameSource(),
        monitor_config(tmp_path),
        stabilizer=stabilizer(),
        clock=FakeClock().clock,
        sleep=lambda _seconds: None,
    )

    monitor.run(max_cycles=2)

    assert monitor.last_timing is None


def test_timing_does_not_change_events(tmp_path: Path) -> None:
    """开启 timing 不改变事件逻辑（同样 3 帧命中 → 1 次基线，无事件）。"""
    def build(timing: bool) -> StateMonitor:
        return StateMonitor(
            EvidenceFrameProcessor(
                EvidenceRegistry([ScriptedDetector([True])]),
                StateClassifier.from_evidence_config(EvidenceConfig()),
            ),
            FakeFrameSource(),
            monitor_config(tmp_path),
            stabilizer=stabilizer(),
            clock=FakeClock().clock,
            sleep=lambda _seconds: None,
            timing=timing,
        )

    without = build(False).run(max_cycles=3)
    with_timing = build(True).run(max_cycles=3)

    assert with_timing.events == without.events == ()
    assert with_timing.initial_state is without.initial_state is StateName.YARD
    assert with_timing.cycles == without.cycles == 3
