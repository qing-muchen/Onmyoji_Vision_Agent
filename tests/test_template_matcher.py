"""TemplateMatcher 与预处理工具的单元测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.config import TemplateDetectorConfig
from src.vision.detector import TemplateInfo, TemplateMatcher
from src.vision.detector.template.preprocess import resize_image, to_grayscale
from tests.helpers import build_scene

TEMPLATE_ID = "button_explore_skin001_scene001"


def make_template(image: np.ndarray, template_id: str = TEMPLATE_ID) -> TemplateInfo:
    """由图像构造模板信息（不读取磁盘）。"""
    return TemplateInfo(
        template_id=template_id,
        element_type="button",
        element_name="explore",
        skin="001",
        scene="001",
        category="buttons",
        path=Path(f"{template_id}.png"),
        image=image,
    )


def test_identical_template_is_matched_with_full_confidence() -> None:
    """帧中贴入完全相同的模板：置信度接近 1，位置正确。"""
    scene = build_scene()
    matcher = TemplateMatcher(TemplateDetectorConfig())

    match = matcher.match(matcher.prepare(scene.frame_image), make_template(scene.template_image))

    assert match is not None
    assert match.confidence == pytest.approx(1.0, abs=1e-3)
    assert match.rect == scene.rect_in_frame
    assert match.scale == pytest.approx(1.0)


def test_absent_template_returns_none() -> None:
    """模板不在帧中时返回 None（低于阈值）。"""
    scene = build_scene(seed=3)
    other = np.random.default_rng(99).integers(0, 256, scene.template_image.shape, dtype=np.uint8)
    matcher = TemplateMatcher(TemplateDetectorConfig())

    assert matcher.match(matcher.prepare(scene.frame_image), make_template(other)) is None


def test_confidence_stays_within_unit_range() -> None:
    """置信度始终落在 [0, 1] 区间内。"""
    scene = build_scene(seed=11)
    matcher = TemplateMatcher(TemplateDetectorConfig(threshold=0.01))

    match = matcher.match(matcher.prepare(scene.frame_image), make_template(scene.template_image))

    assert match is not None
    assert 0.0 <= match.confidence <= 1.0


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.2])
def test_multi_scale_selects_the_matching_scale(scale: float) -> None:
    """模板以某比例出现在帧中时，应选中该比例并定位准确。"""
    scene = build_scene(scale=scale, seed=5)
    matcher = TemplateMatcher(TemplateDetectorConfig())

    match = matcher.match(matcher.prepare(scene.frame_image), make_template(scene.template_image))

    assert match is not None
    assert match.scale == pytest.approx(scale)
    assert match.rect == scene.rect_in_frame
    assert match.confidence > 0.95


def test_multi_scale_can_be_disabled() -> None:
    """关闭多尺度时只使用 1.0 比例，缩放后的模板置信度下降。"""
    scene = build_scene(scale=1.2, seed=6)
    matcher = TemplateMatcher(TemplateDetectorConfig(multi_scale=False, threshold=0.01))

    match = matcher.match(matcher.prepare(scene.frame_image), make_template(scene.template_image))

    assert matcher.scales == (1.0,)
    assert match is not None
    assert match.scale == pytest.approx(1.0)
    assert match.confidence < 0.99


def test_template_larger_than_frame_returns_none() -> None:
    """模板大于帧时跳过，不抛异常。"""
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    template = np.zeros((80, 80, 3), dtype=np.uint8)
    matcher = TemplateMatcher(TemplateDetectorConfig())

    assert matcher.match(matcher.prepare(frame), make_template(template)) is None


def test_threshold_filters_weak_matches() -> None:
    """未达阈值的结果被过滤掉。"""
    scene = build_scene(seed=8)
    other = np.random.default_rng(123).integers(0, 256, scene.template_image.shape, dtype=np.uint8)
    matcher = TemplateMatcher(TemplateDetectorConfig(threshold=0.99))

    assert matcher.match(matcher.prepare(scene.frame_image), make_template(other)) is None


def test_sqdiff_normed_converts_score_to_confidence() -> None:
    """TM_SQDIFF_NORMED 的分数被转换为"越大越像"。"""
    scene = build_scene(seed=9)
    matcher = TemplateMatcher(TemplateDetectorConfig(method="TM_SQDIFF_NORMED"))

    match = matcher.match(matcher.prepare(scene.frame_image), make_template(scene.template_image))

    assert matcher.method_name == "TM_SQDIFF_NORMED"
    assert match is not None
    assert match.confidence == pytest.approx(1.0, abs=1e-3)


def test_prepare_returns_single_channel_image() -> None:
    """prepare 输出单通道灰度图。"""
    scene = build_scene()

    gray = TemplateMatcher.prepare(scene.frame_image)

    assert gray.ndim == 2
    assert gray.shape == scene.frame_image.shape[:2]


def test_to_grayscale_returns_single_channel_input_unchanged() -> None:
    """已是单通道的输入原样返回。"""
    gray = np.zeros((10, 10), dtype=np.uint8)

    assert to_grayscale(gray) is gray


def test_resize_image_rejects_non_positive_scale() -> None:
    """缩放比例必须为正数。"""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="缩放比例"):
        resize_image(image, 0.0)


def test_resize_image_identity_returns_same_array() -> None:
    """比例为 1.0 时不做缩放。"""
    image = np.zeros((10, 10), dtype=np.uint8)

    assert resize_image(image, 1.0) is image
