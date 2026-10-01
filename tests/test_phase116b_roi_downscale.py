"""Phase 11.6-B 测试：ROI 降采样 benchmark 的**只读性与语义不变性**。

用户约束（本阶段只实现 benchmark）：不改阈值、不加模板、不改 detector 语义、
不接入真实输入、不改 action/resolver/safety。因此这里的测试全部围绕：

1. 降采样 matcher 只改**搜索尺度**（= 生产尺度 × scale），``threshold`` 与 ``method`` 不变；
2. ``scale == 1.0`` 与生产 ``best_match`` **逐字段一致**（等价性 ⇒ benchmark 可信）；
3. 降采样后的命中坐标按 ``1/scale`` 恢复，仍落在帧内（坐标恢复正确）；
4. 三个 scale 在真实数据上给出**相同判定**（降采样不改变 detector 语义）；
5. benchmark 报告口径正确（阈值/模板数取自生产配置、比例越小耗时越低）；
6. 运行 benchmark **不写入**任何生产配置。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.core.config import TemplateDetectorConfig, load_config
from src.core.geometry import Rect
from src.core.paths import get_project_root
from src.vision.capture.models import Frame
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector

ROOT = get_project_root()
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import benchmark_roi_downscale as bench  # noqa: E402

YARD_DIR = ROOT / "dataset" / "raw" / "yard"
EXPLORATION_DIR = ROOT / "dataset" / "raw" / "exploration"
SCALES = (1.0, 0.75, 0.5)


@pytest.fixture(scope="module")
def yard_detector() -> BottomMenuYardDetector:
    """本文件的**基准** detector：``roi_scale = 1.0``（真实配置 + 真实模板，只读）。

    Phase 11.6-C 起 ``configs/config.yaml`` 的 ``bottom_menu_yard`` 已接入
    ``roi_scale: 0.75``；而本文件验证的是 Phase 11.6-B 的**参考实现**
    （它自己负责降采样，因而要求基准 detector 不降采样），故这里显式置回 ``1.0``。
    """
    config = load_config()
    production = BottomMenuYardDetector.from_config(config.evidence)
    baseline_config = config.evidence.detectors[bench.YARD_EVIDENCE].model_copy(
        update={"roi_scale": 1.0}
    )
    return BottomMenuYardDetector(bench.YARD_EVIDENCE, baseline_config, production.templates)


@pytest.fixture(scope="module")
def yard_frames() -> list[tuple[str, Frame]]:
    """若干真实 yard 帧（图像解码与 Frame 构造不属于被测对象）。"""
    paths = sorted(YARD_DIR.glob("*.png"))[:4]
    frames: list[tuple[str, Frame]] = []
    for path in paths:
        image = bench.read_image(path)
        if image is None:
            continue
        frames.append((path.name, Frame.create(image, source=path.name, timestamp=1.0)))
    if not frames:
        pytest.skip("缺少 yard 原图")
    return frames


# ---------------------------------------------------------------------------
# 1. 只改搜索尺度，阈值与方法不变
# ---------------------------------------------------------------------------


def test_downscale_matcher_only_scales(yard_detector: BottomMenuYardDetector) -> None:
    """``scales`` 整体乘以比例；``threshold`` / ``method`` 保持生产值。"""
    production = yard_detector.matcher_config
    expected = production.effective_scales

    for scale in SCALES:
        matcher = bench.build_downscale_matcher(production, scale)
        assert matcher.scales == tuple(scale * value for value in expected)
        assert matcher.config.threshold == production.threshold  # 约束 1：不改阈值
        assert matcher.config.method == production.method  # 不改匹配算法

    # scale = 1.0 时与生产完全同构
    assert bench.build_downscale_matcher(production, 1.0).scales == expected


def test_downscale_matcher_rejects_non_positive_scale(
    yard_detector: BottomMenuYardDetector,
) -> None:
    """非正数比例直接拒绝（避免静默产生错误结果）。"""
    with pytest.raises(ValueError):
        bench.build_downscale_matcher(yard_detector.matcher_config, 0.0)
    with pytest.raises(ValueError):
        bench.build_downscale_matcher(yard_detector.matcher_config, -0.5)


# ---------------------------------------------------------------------------
# 2. scale == 1.0 与生产 best_match 逐字段一致
# ---------------------------------------------------------------------------


def test_scale_one_is_identical_to_production(
    yard_detector: BottomMenuYardDetector, yard_frames: list[tuple[str, Frame]]
) -> None:
    """``scale == 1.0`` 的降采样路径 == 生产 ``best_match``（benchmark 可信的前提）。"""
    matcher = bench.build_downscale_matcher(yard_detector.matcher_config, 1.0)

    for name, frame in yard_frames:
        production, roi_rect = yard_detector.best_match(frame)
        candidate, candidate_roi = bench.detect_roi_downscaled(yard_detector, frame, 1.0, matcher)

        assert candidate_roi.describe() == roi_rect.describe(), name
        assert (production is None) == (candidate is None), name
        if production is None:
            continue
        assert candidate.confidence == production.confidence, name
        assert candidate.rect.describe() == production.rect.describe(), name
        assert candidate.template.template_id == production.template.template_id, name
        assert candidate.scale == production.scale, name


def test_reference_alignment_helper_reports_no_mismatch(
    yard_detector: BottomMenuYardDetector,
) -> None:
    """交叉验证辅助函数在真实图上报告 0 处不一致（``scale = 1.0`` 时 clamp 是恒等）。"""
    paths = sorted(YARD_DIR.glob("*.png"))[:3]
    if not paths:
        pytest.skip("缺少 yard 原图")
    report = bench.check_reference_alignment(
        yard_detector,
        yard_detector.config,
        yard_detector.templates,
        paths,
        (1.0,),
        samples=3,
    )
    assert report["checked"] >= 1
    assert report["decision_mismatches"] == 0
    assert report["confidence_mismatches"] == 0
    assert report["rect_mismatches"] == 0


# ---------------------------------------------------------------------------
# 3. 坐标恢复：命中矩形仍在帧内，且与 1.0 的重叠极高
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", SCALES)
def test_restored_rect_stays_inside_frame(
    yard_detector: BottomMenuYardDetector, yard_frames: list[tuple[str, Frame]], scale: float
) -> None:
    """坐标恢复后的矩形位于帧坐标系内（不越界、宽高为正）。"""
    matcher = bench.build_downscale_matcher(yard_detector.matcher_config, scale)

    for name, frame in yard_frames:
        match, roi_rect = bench.detect_roi_downscaled(yard_detector, frame, scale, matcher)
        if match is None:
            continue
        rect = match.rect
        assert rect.width > 0 and rect.height > 0, name
        assert 0 <= rect.left and 0 <= rect.top, name
        # 降采样有取整量化误差：恢复后的框最多比 ROI 边界多出 1 像素（见下一条测试）
        # 命中仍必须落在这个 detector 的 ROI 内（坐标确实恢复了）
        assert rect.left >= roi_rect.left - 1 and rect.top >= roi_rect.top - 1, name
        assert rect.right <= roi_rect.right + 1 and rect.bottom <= roi_rect.bottom + 1, name
        assert rect.right <= frame.width + 1 and rect.bottom <= frame.height + 1, name


def test_downscale_quantization_overshoot_is_bounded(
    yard_detector: BottomMenuYardDetector, yard_frames: list[tuple[str, Frame]]
) -> None:
    """降采样 + 取整的坐标误差上界为 1 像素（不掩盖这一副作用，而是钉住它）。"""
    for scale in SCALES:
        matcher = bench.build_downscale_matcher(yard_detector.matcher_config, scale)
        for name, frame in yard_frames:
            match, roi_rect = bench.detect_roi_downscaled(yard_detector, frame, scale, matcher)
            if match is None:
                continue
            overshoot = max(
                roi_rect.left - match.rect.left,
                roi_rect.top - match.rect.top,
                match.rect.right - roi_rect.right,
                match.rect.bottom - roi_rect.bottom,
            )
            assert overshoot <= 1, f"{name} scale={scale}: 越界 {overshoot} 像素"


def test_downscale_preserves_hit_position(
    yard_detector: BottomMenuYardDetector, yard_frames: list[tuple[str, Frame]]
) -> None:
    """1.0 与 0.75 都命中时，坐标重叠度高（IoU 高 ⇒ 恢复正确）。"""
    baseline = bench.build_downscale_matcher(yard_detector.matcher_config, 1.0)
    downscaled = bench.build_downscale_matcher(yard_detector.matcher_config, 0.75)

    compared = 0
    for name, frame in yard_frames:
        first, _ = bench.detect_roi_downscaled(yard_detector, frame, 1.0, baseline)
        second, _ = bench.detect_roi_downscaled(yard_detector, frame, 0.75, downscaled)
        if first is None or second is None:
            continue
        compared += 1
        # 降采样会带来 ±几像素的量化误差，但应远小于矩形尺寸
        assert bench.rect_iou(first.rect, second.rect) > 0.9, name
    if compared == 0:
        pytest.skip("该批图上没有同时命中的样本")


# ---------------------------------------------------------------------------
# 4. 语义不变性：三个 scale 的判定一致
# ---------------------------------------------------------------------------


def test_decisions_agree_across_scales(yard_detector: BottomMenuYardDetector) -> None:
    """同一批正/负样本上，1.0 / 0.75 / 0.5 的检出判定一致（参考实现）。"""
    positives = sorted(YARD_DIR.glob("*.png"))[:6]
    negatives = sorted(EXPLORATION_DIR.glob("*.png"))[:4]
    if not positives:
        pytest.skip("缺少 yard 原图")

    detector = yard_detector
    matchers = {
        scale: bench.build_downscale_matcher(detector.matcher_config, scale)
        for scale in SCALES
    }

    for path in positives + negatives:
        image = bench.read_image(path)
        if image is None:
            continue
        frame = Frame.create(image, source=path.name, timestamp=1.0)
        decisions = {}
        for scale, matcher in matchers.items():
            match, _roi = bench.detect_roi_downscaled(detector, frame, scale, matcher)
            decisions[scale] = match is not None
        assert len(set(decisions.values())) == 1, f"{path.name}: {decisions}"


# ---------------------------------------------------------------------------
# 5. benchmark 报告口径
# ---------------------------------------------------------------------------


def test_benchmark_report_uses_production_settings() -> None:
    """报告里的阈值/模板数来自生产配置，且自检无不一致。"""
    config = load_config()
    detector_config = config.evidence.detectors[bench.YARD_EVIDENCE]
    report = bench.benchmark(
        config=config, scales=SCALES, image_limit=2, negative_limit=1, repeat=1
    )
    assert "error" not in report
    assert report["threshold"] == float(detector_config.threshold)
    assert report["templates_used"] == len(
        BottomMenuYardDetector.from_config(config.evidence).templates
    )
    # production 实现与参考实现交叉验证：判定与 confidence 必须完全一致
    alignment = report["reference_alignment"]
    assert alignment["decision_mismatches"] == 0
    assert alignment["confidence_mismatches"] == 0
    assert set(report["results"]) == {"1.00", "0.75", "0.50"}

    for key, row in report["results"].items():
        assert row["tp"] + row["fn"] == report["positives_total"], key
        assert row["fp"] + row["tn"] == report["negatives_total"], key
        assert len(row["matcher_scales"]) == len(detector_config.matcher_config().effective_scales)


def test_benchmark_latency_decreases_with_scale() -> None:
    """比例越小 ⇒ 单位耗时越低（本阶段的核心待验证结论）。"""
    report = bench.benchmark(
        scales=SCALES, image_limit=3, negative_limit=2, repeat=2
    )
    assert "error" not in report
    latencies = {key: row["latency_ms"]["p50"] for key, row in report["results"].items()}
    assert latencies["0.50"] is not None and latencies["1.00"] is not None
    assert latencies["0.50"] < latencies["1.00"]
    assert latencies["0.75"] < latencies["1.00"]


# ---------------------------------------------------------------------------
# 6. 只读：不写入生产配置
# ---------------------------------------------------------------------------


def test_benchmark_does_not_touch_production_config() -> None:
    """跑一次 benchmark 前后，``configs/config.yaml`` 字节不变。"""
    config_path = ROOT / "configs" / "config.yaml"
    before = config_path.read_bytes()

    bench.benchmark(scales=(1.0, 0.5), image_limit=2, negative_limit=1, repeat=1)

    assert config_path.read_bytes() == before


def test_helpers() -> None:
    """``rect_iou`` / ``percentile`` / ``_rect_payload`` 的基本语义。"""
    first = Rect(x=0, y=0, width=10, height=10)
    assert bench.rect_iou(first, Rect(x=0, y=0, width=10, height=10)) == pytest.approx(1.0)
    assert bench.rect_iou(first, Rect(x=100, y=100, width=10, height=10)) == 0.0
    # 半重叠：交 50，并 150 ⇒ 1/3
    assert bench.rect_iou(first, Rect(x=5, y=0, width=10, height=10)) == pytest.approx(1 / 3)

    assert bench.percentile([], 50) is None
    assert bench.percentile([3.0], 50) == pytest.approx(3.0)
    assert bench.percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)

    assert bench._rect_payload(None) is None
    assert bench._rect_payload(first) == {"x": 0, "y": 0, "width": 10, "height": 10}


def test_collect_samples_limits(tmp_path: Path) -> None:
    """``collect_samples`` 的上限参数生效（不读入图像内容）。"""
    dataset = tmp_path / "dataset"
    for index in range(4):
        target = dataset / "raw" / "yard" / f"yard_cap_{index:03d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    for index in range(3):
        target = dataset / "raw" / "exploration" / f"exp_{index:03d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")

    positives, negatives = bench.collect_samples(dataset, image_limit=2, negative_limit=1)
    assert len(positives) == 2
    assert len(negatives) == 1

    all_positives, all_negatives = bench.collect_samples(dataset)
    assert len(all_positives) == 4
    assert len(all_negatives) == 3


def test_template_detector_config_unchanged_by_helper(
    yard_detector: BottomMenuYardDetector,
) -> None:
    """构造降采样 matcher 不会修改生产配置对象（``TemplateDetectorConfig`` 不可变语义）。"""
    production = yard_detector.matcher_config
    snapshot = TemplateDetectorConfig(
        method=production.method,
        threshold=production.threshold,
        multi_scale=production.multi_scale,
        scales=list(production.scales),
    )
    for scale in SCALES:
        bench.build_downscale_matcher(production, scale)
    assert production.method == snapshot.method
    assert production.threshold == snapshot.threshold
    assert list(production.scales) == list(snapshot.scales)
    assert np.allclose(production.effective_scales, snapshot.effective_scales)
