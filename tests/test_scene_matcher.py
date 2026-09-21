"""SceneMatcher 与场景相似度计算的单元测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import SceneMatcherConfig
from src.vision.capture.models import Frame
from src.vision.scene import SceneMatcher, SceneTemplate, normalized_similarity
from src.vision.scene.preprocess import resize_to, to_grayscale
from tests.helpers import build_scene_image, perturb_image

MATCHER_CONFIG = SceneMatcherConfig(threshold=0.5, resize_width=64, resize_height=36)
TEMPLATE_ID = "yard_skin001_scene001"


def make_template(image: np.ndarray, *, frame: int = 1) -> SceneTemplate:
    """由图像构造场景模板（不读取磁盘）。"""
    return SceneTemplate(
        template_id=TEMPLATE_ID,
        scene_type="yard",
        skin=1,
        scene_id=1,
        frame=frame,
        path=Path(f"{TEMPLATE_ID}_frame{frame:03d}.png"),
        image=image,
    )


def make_frame(image: np.ndarray) -> Frame:
    """构造场景帧（场景识别不需要坐标映射）。"""
    return Frame.create(image, source="test")


def test_identical_images_have_full_confidence() -> None:
    """完全相同的图像相似度为 1.0。"""
    image = build_scene_image(seed=1)
    matcher = SceneMatcher(MATCHER_CONFIG)

    assert matcher.match(make_frame(image), make_template(image)) == pytest.approx(1.0)


def test_different_images_score_lower() -> None:
    """不同画面的相似度明显低于同一画面。"""
    matcher = SceneMatcher(MATCHER_CONFIG)
    reference = build_scene_image(seed=2)
    other = build_scene_image(seed=99)

    same = matcher.match(make_frame(reference), make_template(reference))
    different = matcher.match(make_frame(reference), make_template(other))

    assert same == pytest.approx(1.0)
    assert different < same - 0.05


def test_similarity_decreases_as_image_diverges() -> None:
    """画面差异越大，相似度越低。"""
    matcher = SceneMatcher(MATCHER_CONFIG)
    base = build_scene_image(seed=3)
    template = make_template(base)

    small = matcher.match(make_frame(perturb_image(base, noise=5, seed=4)), template)
    large = matcher.match(make_frame(perturb_image(base, noise=40, seed=5)), template)

    assert small > large
    assert small <= 1.0


def test_confidence_stays_within_unit_range() -> None:
    """置信度始终落在 [0, 1]：全黑与全白为 0。"""
    matcher = SceneMatcher(MATCHER_CONFIG)
    black = np.zeros((36, 64, 3), dtype=np.uint8)
    white = np.full((36, 64, 3), 255, dtype=np.uint8)

    assert matcher.match(make_frame(black), make_template(white)) == pytest.approx(0.0)
    assert matcher.match(make_frame(black), make_template(black)) == pytest.approx(1.0)


def test_images_of_different_sizes_are_resized_before_compare() -> None:
    """不同尺寸的输入先统一缩放，因此窗口尺寸变化不会破坏匹配。"""
    matcher = SceneMatcher(MATCHER_CONFIG)
    base = build_scene_image(seed=6, size=(160, 90))
    enlarged = cv2.resize(base, (640, 360), interpolation=cv2.INTER_LINEAR)

    confidence = matcher.match(make_frame(enlarged), make_template(base))

    assert confidence > 0.9


def test_similarity_helper_matches_matcher_result() -> None:
    """预处理的比较结果与 match 一致（分类器复用的路径）。"""
    matcher = SceneMatcher(MATCHER_CONFIG)
    image = build_scene_image(seed=7)
    template = make_template(image)

    direct = matcher.match(make_frame(image), template)
    prepared = matcher.similarity(matcher.prepare(image), matcher.prepare(template.image))

    assert prepared == pytest.approx(direct)


def test_prepare_returns_uniform_grayscale() -> None:
    """prepare 输出统一尺寸的单通道灰度图。"""
    matcher = SceneMatcher(MATCHER_CONFIG)
    prepared = matcher.prepare(build_scene_image(seed=8, size=(333, 111)))

    assert prepared.ndim == 2
    assert prepared.shape == (MATCHER_CONFIG.resize_height, MATCHER_CONFIG.resize_width)


def test_normalized_similarity_bounds() -> None:
    """相似度计算的边界：一致为 1，差异最大为 0。"""
    same = np.zeros((4, 4), dtype=np.uint8)
    opposite = np.full((4, 4), 255, dtype=np.uint8)
    middle = np.full((4, 4), 128, dtype=np.uint8)

    assert normalized_similarity(same, same) == pytest.approx(1.0)
    assert normalized_similarity(same, opposite) == pytest.approx(0.0)
    assert 0.0 < normalized_similarity(same, middle) < 1.0


def test_resize_to_validates_target_size() -> None:
    """目标尺寸必须为正数。"""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="目标尺寸"):
        resize_to(image, (0, 10))


def test_resize_to_returns_same_array_when_size_matches() -> None:
    """尺寸已匹配时不做缩放。"""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    assert resize_to(image, (10, 10)) is image


def test_to_grayscale_keeps_single_channel_input() -> None:
    """已是单通道的输入原样返回。"""
    gray = np.zeros((10, 10), dtype=np.uint8)

    assert to_grayscale(gray) is gray
