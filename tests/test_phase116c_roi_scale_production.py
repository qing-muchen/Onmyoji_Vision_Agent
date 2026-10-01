"""Phase 11.6-C 测试：ROI 降采样的**生产能力**（配置开关 + bbox clamp + 回归）。

用户约束：默认行为保持 ``roi_scale = 1.0``、新增配置开关、首次只接入 ``0.75``、
不改 ``threshold``、不增加模板、不改 action / resolver / safety。因此这里的测试围绕：

1. 配置层：默认值 ``1.0``、非法值被拒、``matcher_config()`` 只缩放搜索尺度；
2. **默认行为回归**：``roi_scale = 1.0`` 与 Phase 11.6-B 的参考实现逐字段一致，
   且 ``metadata`` 不新增任何键（既有下游不受影响）；
3. **bbox clamp**：任意比例下命中框严格落在 ROI 内（含 11.6-B 曾越界 1 像素的样本）；
4. **语义不变**：生产接入的 ``0.75`` 与 ``1.0`` 判定一致、confidence 与参考实现相等；
5. 生产配置只接入 ``bottom_menu_yard``，其余 detector 保持 ``1.0``，阈值与模板数未变；
6. OCR 路径仍使用**原尺寸** ROI（降采样只作用于模板匹配）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config import EvidenceDetectorConfig, load_config
from src.core.geometry import Rect, Size
from src.core.paths import get_project_root
from src.vision.capture.models import Frame
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector

ROOT = get_project_root()
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import benchmark_roi_downscale as bench  # noqa: E402

YARD = "bottom_menu_yard"
YARD_DIR = ROOT / "dataset" / "raw" / "yard"
EXPLORATION_DIR = ROOT / "dataset" / "raw" / "exploration"
PRODUCTION_ROI_SCALE = 0.75
#: Phase 11.6-B 实测中恢复后越出 ROI 1 像素的两张图（生产 clamp 后必须归零）
OVERSHOOT_SAMPLES = ("yard_cap_001.png", "yard_cap_005.png")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_config():  # noqa: ANN201 - 配置对象类型由 load_config 决定
    return load_config()


@pytest.fixture(scope="module")
def yard_config(app_config):  # noqa: ANN001, ANN201
    return app_config.evidence.detectors[YARD]


@pytest.fixture(scope="module")
def templates(app_config) -> tuple:  # noqa: ANN001
    """生产加载的那一批模板（同一批复用给各比例的 detector）。"""
    return BottomMenuYardDetector.from_config(app_config.evidence).templates


def detector_with_scale(yard_config, templates, scale: float) -> BottomMenuYardDetector:  # noqa: ANN001
    """按给定 ``roi_scale`` 构造 detector（内存副本，不写配置）。"""
    return bench.build_scale_detector(yard_config, templates, scale)


@pytest.fixture(scope="module")
def frames() -> list[tuple[str, Frame]]:
    """若干真实 yard 帧（含 11.6-B 的越界样本）。"""
    names = list(OVERSHOOT_SAMPLES) + [path.name for path in sorted(YARD_DIR.glob("*.png"))[:4]]
    frames: list[tuple[str, Frame]] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        image = bench.read_image(YARD_DIR / name)
        if image is None:
            continue
        frames.append((name, Frame.create(image, source=name, timestamp=1.0)))
    if not frames:
        pytest.skip("缺少 yard 原图")
    return frames


# ---------------------------------------------------------------------------
# 1. 配置层
# ---------------------------------------------------------------------------


def test_roi_scale_defaults_to_identity() -> None:
    """新字段默认 ``1.0``（未配置的 detector 行为不变）。"""
    assert EvidenceDetectorConfig().roi_scale == 1.0


@pytest.mark.parametrize("value", [0.0, -0.1, 1.0001, 2.0])
def test_roi_scale_rejects_out_of_range(value: float) -> None:
    """非法比例（非正 / 大于 1）直接拒绝，避免静默放大或零尺寸。"""
    with pytest.raises(ValidationError):
        EvidenceDetectorConfig(roi_scale=value)


def test_matcher_config_scales_only(yard_config) -> None:  # noqa: ANN001
    """``matcher_config()`` 只把搜索尺度乘以 ``roi_scale``，阈值与方法不变。"""
    production = yard_config.matcher_config()
    assert production.threshold == yard_config.threshold  # 约束：不改阈值
    assert production.method == yard_config.method

    scaled = yard_config.model_copy(update={"roi_scale": 0.5}).matcher_config()
    assert scaled.scales == [0.5 * value for value in yard_config.scales]
    assert scaled.threshold == yard_config.threshold
    assert scaled.method == yard_config.method

    identity = yard_config.model_copy(update={"roi_scale": 1.0}).matcher_config()
    assert identity.scales == list(yard_config.scales)


def test_production_config_only_yard_is_downscaled(yard_config, app_config) -> None:  # noqa: ANN001
    """生产配置：只有 ``bottom_menu_yard`` 接入 ``0.75``，其余 detector 仍是 ``1.0``。"""
    assert yard_config.roi_scale == PRODUCTION_ROI_SCALE
    others = {
        name: config.roi_scale
        for name, config in app_config.evidence.detectors.items()
        if name != YARD
    }
    assert others, "至少应存在其它 detector"
    assert set(others.values()) == {1.0}, others


def test_production_config_keeps_threshold_and_templates(yard_config) -> None:  # noqa: ANN001
    """接入降采样不得顺带改动阈值、模板预算与 ROI。"""
    assert yard_config.threshold == 0.85
    assert yard_config.max_templates == 12
    assert yard_config.roi == (0.0, 0.8, 1.0, 1.0)
    assert yard_config.template_source == "dataset/features/bottom_menu/yard/positive"


# ---------------------------------------------------------------------------
# 2. 默认行为回归（roi_scale = 1.0）
# ---------------------------------------------------------------------------


def test_identity_scale_matches_reference_implementation(yard_config, templates, frames) -> None:  # noqa: ANN001
    """``roi_scale = 1.0`` 与 11.6-B 的参考实现逐字段一致（含命中框）。"""
    detector = detector_with_scale(yard_config, templates, 1.0)
    reference_matcher = bench.build_downscale_matcher(detector.matcher_config, 1.0)

    for name, frame in frames:
        production, roi_rect = detector.best_match(frame)
        reference, reference_roi = bench.detect_roi_downscaled(
            detector, frame, 1.0, reference_matcher
        )
        assert reference_roi.describe() == roi_rect.describe(), name
        assert (production is None) == (reference is None), name
        if production is None:
            continue
        assert production.confidence == reference.confidence, name
        assert production.rect.describe() == reference.rect.describe(), name
        assert production.template.template_id == reference.template.template_id, name


def test_identity_scale_keeps_metadata_unchanged(yard_config, templates, frames) -> None:  # noqa: ANN001
    """``roi_scale = 1.0`` 时 ``metadata`` 不含 ``roi_scale`` 键（既有字段集不变）。"""
    detector = detector_with_scale(yard_config, templates, 1.0)
    name, frame = frames[0]
    result = detector.detect(frame)
    assert "roi_scale" not in result.metadata, name
    assert {"template_id", "raw_confidence", "roi", "template_count"} <= set(result.metadata)


def test_downscaled_metadata_records_scale(yard_config, templates, frames) -> None:  # noqa: ANN001
    """降采样时 ``metadata`` 如实记录 ``roi_scale``。"""
    detector = detector_with_scale(yard_config, templates, PRODUCTION_ROI_SCALE)
    for _name, frame in frames:
        result = detector.detect(frame)
        if not result.detected:
            continue
        assert result.metadata["roi_scale"] == PRODUCTION_ROI_SCALE
        return
    pytest.skip("该批图上没有命中样本")


# ---------------------------------------------------------------------------
# 3. bbox clamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 0.75, 0.5])
def test_hit_bbox_stays_inside_roi(yard_config, templates, frames, scale: float) -> None:  # noqa: ANN001
    """命中框严格落在 ROI 内（**不放宽**容差 —— 生产实现已 clamp）。"""
    detector = detector_with_scale(yard_config, templates, scale)

    for name, frame in frames:
        match, roi_rect = detector.best_match(frame)
        if match is None:
            continue
        rect = match.rect
        assert rect.width >= 1 and rect.height >= 1, name
        assert rect.left >= roi_rect.left and rect.top >= roi_rect.top, name
        assert rect.right <= roi_rect.right and rect.bottom <= roi_rect.bottom, name
        # 同时不越出帧
        assert rect.right <= frame.width and rect.bottom <= frame.height, name


def test_restore_rect_clamps_every_corner(yard_config, templates) -> None:  # noqa: ANN001
    """``_restore_rect`` 对越界输入一律 clamp 回 ROI（含四边与退化宽高）。"""
    roi_size = Size(width=1706, height=192)
    cases = [
        # scale, 降采样空间的命中框, 说明
        (0.75, Rect(x=0, y=0, width=1280, height=144), "1.0 档恢复到 1707 宽"),
        (0.5, Rect(x=0, y=0, width=853, height=96), "右/下边界取整溢出"),
        (0.5, Rect(x=850, y=94, width=4, height=4), "右下角极小框"),
        (0.75, Rect(x=0, y=0, width=1, height=1), "左上角单像素"),
    ]
    for scale, rect, note in cases:
        detector = detector_with_scale(yard_config, templates, scale)
        assert detector.roi_scale == scale
        restored = detector._restore_rect(rect, roi_size)  # noqa: SLF001 - 直接验证 clamp
        assert restored.width >= 1 and restored.height >= 1, note
        assert restored.left >= 0 and restored.top >= 0, note
        assert restored.right <= roi_size.width, note
        assert restored.bottom <= roi_size.height, note


def test_known_overshoot_samples_are_clamped(yard_config, templates) -> None:  # noqa: ANN001
    """11.6-B 曾越界 1 像素的两张图，在生产 clamp 下不再越界。"""
    for scale in (0.75, 0.5):
        detector = detector_with_scale(yard_config, templates, scale)
        for name in OVERSHOOT_SAMPLES:
            image = bench.read_image(YARD_DIR / name)
            if image is None:
                continue
            frame = Frame.create(image, source=name, timestamp=1.0)
            match, roi_rect = detector.best_match(frame)
            if match is None:
                continue
            assert bench.overshoot_of(match.rect, roi_rect) == 0, f"{name} scale={scale}"


# ---------------------------------------------------------------------------
# 4. 语义不变（生产 0.75 vs 1.0，以及 vs 参考实现）
# ---------------------------------------------------------------------------


def test_production_scale_agrees_with_identity_on_real_images(yard_config, templates) -> None:  # noqa: ANN001
    """生产 ``0.75`` 与 ``1.0`` 在真实正/负样本上判定一致。"""
    positives = sorted(YARD_DIR.glob("*.png"))[:6]
    negatives = sorted(EXPLORATION_DIR.glob("*.png"))[:4]
    if not positives:
        pytest.skip("缺少 yard 原图")

    baseline = detector_with_scale(yard_config, templates, 1.0)
    production = detector_with_scale(yard_config, templates, PRODUCTION_ROI_SCALE)

    compared = 0
    for path in positives + negatives:
        image = bench.read_image(path)
        if image is None:
            continue
        frame = Frame.create(image, source=path.name, timestamp=1.0)
        expected, _ = baseline.best_match(frame)
        actual, _ = production.best_match(frame)
        compared += 1
        assert (expected is None) == (actual is None), path.name
    assert compared >= 1


def test_production_confidence_equals_reference_implementation(yard_config, templates, frames) -> None:  # noqa: ANN001
    """生产实现与参考实现的 ``confidence`` 完全相等（生产化没有改变匹配行为）。"""
    base = detector_with_scale(yard_config, templates, 1.0)
    production = detector_with_scale(yard_config, templates, PRODUCTION_ROI_SCALE)
    matcher = bench.build_downscale_matcher(base.matcher_config, PRODUCTION_ROI_SCALE)

    hits = 0
    for name, frame in frames:
        expected, _ = production.best_match(frame)
        reference, _ = bench.detect_roi_downscaled(base, frame, PRODUCTION_ROI_SCALE, matcher)
        assert (expected is None) == (reference is None), name
        if expected is None:
            continue
        hits += 1
        assert expected.confidence == reference.confidence, name
        assert expected.template.template_id == reference.template.template_id, name
    assert hits >= 1, "该批图上没有命中样本"


# ---------------------------------------------------------------------------
# 5. OCR 路径不降采样
# ---------------------------------------------------------------------------


def test_roi_frame_keeps_full_resolution(yard_config, templates, frames) -> None:  # noqa: ANN001
    """``_roi_frame``（OCR 输入）始终是 ROI 的**原始**像素尺寸。"""
    detector = detector_with_scale(yard_config, templates, PRODUCTION_ROI_SCALE)
    for name, frame in frames:
        roi_rect = detector.roi.to_rect(frame.width, frame.height)
        roi_frame = detector._roi_frame(frame)  # noqa: SLF001 - 断言 OCR 输入尺寸
        assert roi_frame.width == roi_rect.width, name
        assert roi_frame.height == roi_rect.height, name


# ---------------------------------------------------------------------------
# 6. benchmark 集成（生产实现 / 参考实现）
# ---------------------------------------------------------------------------


def test_benchmark_production_mode_reports_zero_overshoot(app_config) -> None:  # noqa: ANN001
    """benchmark 的 production 模式：生产配置被识别、越界为 0、交叉验证不冲突。"""
    report = bench.benchmark(
        config=app_config, scales=(1.0, PRODUCTION_ROI_SCALE), image_limit=3, negative_limit=1, repeat=1
    )
    assert "error" not in report
    assert report["mode"] == "production"
    assert report["production_roi_scale"] == PRODUCTION_ROI_SCALE
    # 生产 matcher 的搜索尺度含 roi_scale 缩放
    assert report["production_matcher_scales"] == [
        PRODUCTION_ROI_SCALE * value for value in app_config.evidence.detectors[YARD].scales
    ]
    for key, row in report["results"].items():
        assert row["mode"] == "production", key
        assert row["effective_roi_scale"] == float(key), key
        assert row["boundary_overshoot"]["max_px"] == 0, key
    alignment = report["reference_alignment"]
    assert alignment["decision_mismatches"] == 0
    assert alignment["confidence_mismatches"] == 0


def test_benchmark_rejects_unknown_mode() -> None:
    """未知模式直接报错，不静默降级。"""
    report = bench.benchmark(mode="turbo", scales=(1.0,), image_limit=1, negative_limit=1)
    assert "error" in report


# ---------------------------------------------------------------------------
# 7. 只读：接通生产不等于写配置
# ---------------------------------------------------------------------------


def test_production_path_does_not_touch_config_file(app_config) -> None:  # noqa: ANN001
    """构造各比例的 detector + 跑 detection 之后，``configs/config.yaml`` 字节不变。"""
    config_path = ROOT / "configs" / "config.yaml"
    before = config_path.read_bytes()

    templates = BottomMenuYardDetector.from_config(app_config.evidence).templates
    yard_config = app_config.evidence.detectors[YARD]
    for scale in (1.0, PRODUCTION_ROI_SCALE, 0.5):
        detector = detector_with_scale(yard_config, templates, scale)
        image = bench.read_image(sorted(YARD_DIR.glob("*.png"))[0])
        if image is not None:
            detector.detect(Frame.create(image, source="cfg", timestamp=1.0))

    assert config_path.read_bytes() == before
