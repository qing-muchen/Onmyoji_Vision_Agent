"""Phase 11.8-R1 测试：模板/ROI 1~2 px 边界缺陷的 bounded fit-to-ROI（30 项）。

真实运行期背景（Phase 11.8-R）：``roi_scale=0.75`` 下 ROI 与模板各自独立 ``round`` 取整。
当模板比当前匹配 ROI 宽/高 1~2 px 时，最正确的 pyramid 档位会被 ``TemplateMatcher.match()``
的"模板大于帧"检查**整档跳过** —— 实测同一内容仅差 1 px，分数从 0.9999 崩到 0.7444、
``detect()`` 由 True 变 False。

本文件先于实现编写（红），覆盖用户指定的 30 个点；实现位置在 ``TemplateMatcher.match()``
（外加 ``best_match`` 保留 fit 字段这一必要传递）。
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import TemplateDetectorConfig, load_config
from src.core.paths import get_project_root
from src.vision.capture.models import Frame
from src.vision.detector.models import TemplateInfo
from src.vision.detector.template.matcher import FIT_TOLERANCE_PX, TemplateMatcher
from src.vision.pipeline.frame_processor import (
    EvidenceFrameProcessor,
    build_transition_probe,
)

ROOT = get_project_root()
DATASET = ROOT / "dataset"
YARD_DIR = DATASET / "raw" / "yard"
YARD_NAME = "bottom_menu_yard"
SAMPLE_1704 = YARD_DIR / "yard_skin001_001.png"
SAMPLE_1706 = YARD_DIR / "yard_cap_006.png"
VICTIM_1703 = YARD_DIR / "yard_cap_024.png"
MANIFEST = DATASET / "manifests" / "raw_sha256.json"
ANNOTATIONS = DATASET / "annotations" / "elements.jsonl"
MATCHER_SRC = ROOT / "src" / "vision" / "detector" / "template" / "matcher.py"
PREPROCESS_SRC = ROOT / "src" / "vision" / "detector" / "template" / "preprocess.py"


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def build_template(width: int, height: int, *, seed: int = 0) -> tuple[TemplateInfo, np.ndarray]:
    """构造模板（BGR）与其灰度基准图。

    基准图先做高斯平滑：真实画面相邻像素高度相关，而纯随机噪声在 1 px 重采样后内容
    完全错位，会把"几何是否放得下"误判成"相似度是否够高"。
    """
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(2.0, width / 64.0))
    template = TemplateInfo(
        template_id=f"tpl_{width}x{height}",
        element_type="element",
        element_name="boundary_probe",
        skin="000",
        scene="000",
        category="probe",
        path=Path(f"tpl_{width}x{height}.png"),
        image=cv2.cvtColor(base, cv2.COLOR_GRAY2BGR),
    )
    return template, base


def geometry_matcher(*, threshold: float = 0.1) -> TemplateMatcher:
    """单尺度（1.0）匹配器：模板尺寸即 pyramid 档位尺寸，便于精确构造边界。"""
    config = TemplateDetectorConfig(
        method="TM_CCOEFF_NORMED", threshold=threshold, multi_scale=False
    )
    return TemplateMatcher(config)


def crop(base: np.ndarray, width: int, height: int) -> np.ndarray:
    """从基准图裁出 ``width x height`` 的灰度帧（保证连续内存）。"""
    return np.ascontiguousarray(base[:height, :width])


def load_frame(path: Path, *, width: int | None = None) -> Frame:
    """读取真实原图为 ``Frame``（可截断到指定宽度）。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, path
    if width is not None:
        image = np.ascontiguousarray(image[:, :width])
    height, real_width = image.shape[:2]
    return Frame(
        image=image,
        timestamp=time.monotonic(),
        width=real_width,
        height=height,
        source="phase118r",
    )


@pytest.fixture(scope="module")
def yard_detector():  # noqa: ANN201
    """生产 yard detector（Production36 + thumb Top-4）。"""
    config = load_config()
    processor = EvidenceFrameProcessor.from_config(
        config, transition_classifier=build_transition_probe(config)
    )
    detector = processor.registry.find(YARD_NAME)
    assert detector is not None
    return detector


# ---------------------------------------------------------------------------
# 1~9：overflow 判定（fit / skip 边界）
# ---------------------------------------------------------------------------


def test_overflow_zero_keeps_original_path() -> None:
    """[1] overflow = 0 ⇒ 完全走原路径，且不触发 fit。"""
    template, base = build_template(1277, 144)
    matcher = geometry_matcher()
    match = matcher.match(crop(base, 1277, 144), template)
    assert matcher.fit_count == 0
    assert match is not None
    assert match.fit_applied is False
    assert match.fit_ratio == 1.0
    assert match.rect.width == 1277 and match.rect.height == 144


def test_width_overflow_one_applies_bounded_fit() -> None:
    """[2] 宽度溢 1 px ⇒ 轻微 fit 后参与匹配（修复前整档 skip）。"""
    template, base = build_template(1278, 144)
    matcher = geometry_matcher()
    match = matcher.match(crop(base, 1277, 144), template)
    assert matcher.fit_count == 1
    assert match is not None, "宽度溢 1 px 不应被整档跳过"
    assert match.fit_applied is True
    assert match.scale == 1.0, "scale 语义必须仍是配置的逻辑比例"
    assert match.rect.width <= 1277 and match.rect.height <= 144


def test_width_overflow_two_applies_bounded_fit() -> None:
    """[3] 宽度溢 2 px ⇒ fit 后参与匹配。"""
    template, base = build_template(1280, 144)
    matcher = geometry_matcher()
    match = matcher.match(crop(base, 1278, 144), template)
    assert matcher.fit_count == 1
    assert match is not None, "宽度溢 2 px 不应被整档跳过"
    assert match.fit_applied is True
    assert match.rect.width <= 1278 and match.rect.height <= 144


def test_width_overflow_three_is_skipped() -> None:
    """[4] 宽度溢 3 px ⇒ 仍必须 skip（容差外不得静默 fit）。"""
    template, base = build_template(1280, 144)
    matcher = geometry_matcher()
    assert matcher.match(crop(base, 1277, 144), template) is None
    assert matcher.fit_count == 0


def test_height_overflow_one_applies_bounded_fit() -> None:
    """[5] 高度溢 1 px ⇒ fit。"""
    template, base = build_template(1277, 145)
    matcher = geometry_matcher()
    match = matcher.match(crop(base, 1277, 144), template)
    assert matcher.fit_count == 1
    assert match is not None and match.fit_applied is True
    assert match.rect.height <= 144


def test_height_overflow_two_applies_bounded_fit() -> None:
    """[6] 高度溢 2 px ⇒ fit。"""
    template, base = build_template(1277, 146)
    matcher = geometry_matcher()
    match = matcher.match(crop(base, 1277, 144), template)
    assert matcher.fit_count == 1
    assert match is not None and match.fit_applied is True
    assert match.rect.height <= 144


def test_height_overflow_three_is_skipped() -> None:
    """[7] 高度溢 3 px ⇒ skip。"""
    template, base = build_template(1277, 147)
    matcher = geometry_matcher()
    assert matcher.match(crop(base, 1277, 144), template) is None
    assert matcher.fit_count == 0


def test_combined_small_overflow_applies_bounded_fit() -> None:
    """[8] 宽高同时 <= 2 px ⇒ fit。"""
    template, base = build_template(1278, 146)
    matcher = geometry_matcher()
    match = matcher.match(crop(base, 1277, 144), template)
    assert matcher.fit_count == 1
    assert match is not None and match.fit_applied is True
    assert match.rect.width <= 1277 and match.rect.height <= 144


def test_large_overflow_is_skipped() -> None:
    """[9] 显著超界（≈2 倍）⇒ 必须始终 skip。"""
    template, base = build_template(2554, 144)
    matcher = geometry_matcher()
    assert matcher.match(crop(base, 1277, 144), template) is None
    assert matcher.fit_count == 0


# ---------------------------------------------------------------------------
# 10~14：fit 几何正确性与缓存/顺序不变量
# ---------------------------------------------------------------------------


def test_fit_keeps_aspect_ratio() -> None:
    """[10] fit 必须等比缩放（不做非等比拉伸）。"""
    template, base = build_template(1278, 292)
    matcher = geometry_matcher()
    match = matcher.match(crop(base, 1276, 292), template)
    assert matcher.fit_count == 1
    assert match is not None and match.fit_applied is True
    original = 1278 / 292
    fitted = match.rect.width / match.rect.height
    assert abs(fitted - original) / original < 0.01


def test_fit_result_never_exceeds_frame() -> None:
    """[11] fit 后尺寸不超过 frame，且至少 1 像素。"""
    for template_w, template_h, frame_w, frame_h in (
        (1278, 144, 1277, 144),
        (1280, 144, 1278, 144),
        (1278, 146, 1277, 144),
        (1280, 146, 1278, 144),
    ):
        template, base = build_template(template_w, template_h)
        matcher = geometry_matcher()
        match = matcher.match(crop(base, frame_w, frame_h), template)
        assert matcher.fit_count == 1
        assert match is not None
        assert 1 <= match.rect.width <= frame_w
        assert 1 <= match.rect.height <= frame_h


def test_cached_pyramid_is_not_mutated() -> None:
    """[12] bounded fit 不得写回 template pyramid 缓存。"""
    matcher = geometry_matcher()
    template, base = build_template(1278, 144)
    before = [(scale, level.copy()) for scale, level in matcher.template_pyramid(template)]
    result = matcher.match(crop(base, 1277, 144), template)
    assert result is not None and result.fit_applied is True
    after = matcher.template_pyramid(template)
    assert len(after) == len(before)
    for (scale_before, level_before), (scale_after, level_after) in zip(before, after):
        assert scale_before == scale_after
        assert level_after.shape == level_before.shape
        assert np.array_equal(level_after, level_before)


def test_normal_roi_unchanged_after_fit_roi() -> None:
    """[13] 先匹配需要 fit 的小 ROI，再匹配正常 ROI ⇒ 正常 ROI 结果不受影响。"""
    template, base = build_template(1278, 144)
    matcher = geometry_matcher()
    fitted = matcher.match(crop(base, 1277, 144), template)
    normal = matcher.match(crop(base, 1278, 144), template)
    assert fitted is not None and fitted.fit_applied is True
    assert normal is not None and normal.fit_applied is False
    assert normal.rect.width == 1278 and normal.rect.height == 144
    assert matcher.fit_count == 1


def test_fit_roi_unchanged_after_normal_roi() -> None:
    """[14] 反向顺序（先正常、后 fit）⇒ 行为与 [13] 一致。"""
    template, base = build_template(1278, 144)
    matcher = geometry_matcher()
    normal = matcher.match(crop(base, 1278, 144), template)
    fitted = matcher.match(crop(base, 1277, 144), template)
    assert normal is not None and normal.fit_applied is False
    assert fitted is not None and fitted.fit_applied is True
    assert normal.rect.width == 1278
    assert fitted.rect.width <= 1277
    assert matcher.fit_count == 1


# ---------------------------------------------------------------------------
# 15~18：真实图片路径
# ---------------------------------------------------------------------------


def test_real_1704_path_does_not_trigger_fit(yard_detector) -> None:  # noqa: ANN001
    """[15] 1704 正常 yard 路径不触发 fit。"""
    frame = load_frame(SAMPLE_1704)
    assert frame.width == 1704
    match, _roi = yard_detector.best_match(frame)  # noqa: SLF001
    assert match is not None
    assert match.fit_applied is False
    assert match.fit_ratio == 1.0
    assert match.confidence > 0.99


def test_real_1706_path_does_not_trigger_fit(yard_detector) -> None:  # noqa: ANN001
    """[16] 1706 正常 yard 路径不触发 fit。"""
    frame = load_frame(SAMPLE_1706)
    assert frame.width == 1706
    match, _roi = yard_detector.best_match(frame)  # noqa: SLF001
    assert match is not None
    assert match.fit_applied is False
    assert match.confidence > 0.99


def test_real_1703_boundary_path_applies_fit(yard_detector) -> None:  # noqa: ANN001
    """[17] 1703 边界路径（模板比 ROI 宽 1~2 px）被正确处理：走 bounded fit。"""
    frame = load_frame(SAMPLE_1704, width=1703)
    assert frame.width == 1703
    match, _roi = yard_detector.best_match(frame)  # noqa: SLF001
    assert match is not None, "1703 帧应通过 bounded fit 恢复可用档位"
    assert match.fit_applied is True
    assert match.fit_ratio < 1.0


def test_bbox_after_fit_stays_inside_roi(yard_detector) -> None:  # noqa: ANN001
    """[18] fit 后 bbox 仍走既有恢复路径，严格落在 ROI 与帧内。"""
    frame = load_frame(SAMPLE_1704, width=1703)
    result = yard_detector.detect(frame)
    assert result.detected is True
    bbox = result.bbox
    assert bbox is not None
    assert bbox.x >= 0 and bbox.y >= 0
    assert bbox.x + bbox.width <= frame.width
    assert bbox.y + bbox.height <= frame.height
    assert bbox.width >= 1 and bbox.height >= 1
    # yard ROI = 帧底部 20%（768..960）：bbox 必须落在 ROI 内
    assert bbox.y >= 768
    assert bbox.y + bbox.height <= 960


# ---------------------------------------------------------------------------
# 19~27：Production36 / Top-4 契约与安全配置
# ---------------------------------------------------------------------------


def test_production_loads_36_templates(yard_detector) -> None:  # noqa: ANN001
    """[19] Production36 仍加载 36 张模板。"""
    assert len(yard_detector.templates) == 36


def test_prefilter_top_k_candidates_within_bound(yard_detector) -> None:  # noqa: ANN001
    """[20] thumb Top-4：候选数 <= 4，且 fit 不被计入额外候选模板。"""
    assert yard_detector.prefilter_top_k == 4
    frame = load_frame(SAMPLE_1704, width=1703)
    metadata = yard_detector.detect(frame).metadata
    assert metadata["prefilter_candidate_count"] <= 4
    assert metadata["prefilter_total_templates"] == 36
    assert metadata.get("prefilter_fallback") is not True


def test_threshold_remains_085() -> None:
    """[21] 生产 threshold 仍为 0.85。"""
    assert load_config().evidence.detectors[YARD_NAME].threshold == 0.85


def test_roi_scale_remains_075() -> None:
    """[22] 生产 roi_scale 仍为 0.75，模板集合与预筛未变。"""
    yard_config = load_config().evidence.detectors[YARD_NAME]
    assert yard_config.roi_scale == 0.75
    assert len(yard_config.template_include) == 36
    assert yard_config.template_prefilter == "thumb"
    assert yard_config.prefilter_top_k == 4


def test_canonical_negatives_count_is_80() -> None:
    """[23] canonical 正/负样本仍为 98 / 80（递归口径）。"""
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import benchmark_template_prefilter as bench  # noqa: PLC0415

    assert len(bench.collect_positives(DATASET)) == 98
    assert len(bench.collect_negatives(DATASET)) == 80


def test_representative_negatives_false_positive_free(yard_detector) -> None:  # noqa: ANN001
    """[24] 代表性困难负样本（unknown/friends、unknown/shop）在 0.85 下无假阳性。"""
    checked = 0
    for group in ("unknown/friends", "unknown/shop"):
        directory = DATASET / "raw" / group
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.png")):
            frame = load_frame(path)
            assert yard_detector.detect(frame).detected is False, path
            checked += 1
    assert checked > 0


def test_action_disabled() -> None:
    """[25] action.enabled 必须仍为 false。"""
    assert load_config().action.enabled is False


def test_backend_dry_run() -> None:
    """[26] backend 必须仍为 dry_run。"""
    assert load_config().action.backend == "dry_run"


def test_real_input_disabled() -> None:
    """[27] 不存在真实输入路径：matcher/preprocess 不接触 GUI 与 Action。"""
    for source_path in (MATCHER_SRC, PREPROCESS_SRC):
        source = source_path.read_text(encoding="utf-8")
        assert "pyautogui" not in source
        assert "SetForegroundWindow" not in source
        assert "src.action" not in source
        assert "src.runtime" not in source


# ---------------------------------------------------------------------------
# 28~30：数据不变性与 AST 边界
# ---------------------------------------------------------------------------


def test_raw_manifest_unchanged() -> None:
    """[28] raw 178 张 SHA256 清单不变。"""
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["count"] == 178
    assert len(payload["entries"]) == 178


def test_annotations_unchanged() -> None:
    """[29] annotations 不变（8 条，均为待审状态）。"""
    records = [
        json.loads(line)
        for line in ANNOTATIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 8
    assert all(record["annotation_status"] == "pending_review" for record in records)
    assert all(record["review_status"] == "not_reviewed" for record in records)


def test_ast_boundary_unchanged() -> None:
    """[30] TemplateMatcher 不引入 GUI/输入/跨层依赖，也不调用窗口 API。"""
    tree = ast.parse(MATCHER_SRC.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for forbidden in ("pyautogui", "pynput", "src.action", "src.runtime", "src.task"):
        assert not any(name.startswith(forbidden) for name in imported), forbidden

    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name:
                called.add(name)
    for forbidden in ("SetForegroundWindow", "click", "press", "write", "hotkey"):
        assert forbidden not in called


def test_fit_tolerance_is_exactly_two_pixels() -> None:
    """常量契约：FIT_TOLERANCE 严格为 2 px（不使用百分比放大容差）。"""
    assert FIT_TOLERANCE_PX == 2
    assert isinstance(FIT_TOLERANCE_PX, int)
