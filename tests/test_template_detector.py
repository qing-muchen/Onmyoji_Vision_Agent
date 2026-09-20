"""TemplateDetector 的单元测试：输入 Frame、输出 DetectionResult、坐标转换。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import TemplateDetectorConfig
from src.core.exceptions import MissingTransformError
from src.core.geometry import CoordinateTransform, Rect, Size
from src.vision.capture.models import Frame
from src.vision.detector import DetectionResult, TemplateDetector, TemplateInfo, TemplateLoader
from tests.helpers import build_scene

#: 当前分辨率（窗口客户区）与参考分辨率：参考恰好是当前的 5 倍，便于精确断言
FRAME_SIZE = (200, 150)
REFERENCE_SIZE = Size(width=1000, height=750)
REFERENCE_FACTOR = REFERENCE_SIZE.width / FRAME_SIZE[0]
TEMPLATE_ID = "button_explore_skin001_scene001"


def build_transform() -> CoordinateTransform:
    """构造 current -> reference 的坐标映射（reference = 5 x current）。"""
    return CoordinateTransform.from_sizes(
        reference=REFERENCE_SIZE,
        current=Size(width=FRAME_SIZE[0], height=FRAME_SIZE[1]),
    )


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


def make_frame(image: np.ndarray, *, with_transform: bool = True) -> Frame:
    """由图像构造帧（可选是否携带坐标映射）。"""
    return Frame.create(
        image,
        source="test",
        coordinate_transform=build_transform() if with_transform else None,
    )


def test_detect_returns_results_sorted_by_confidence() -> None:
    """检测输出 DetectionResult 列表并按 confidence 降序。"""
    scene = build_scene()
    detector = TemplateDetector(
        [make_template(scene.template_image, f"{TEMPLATE_ID}_{index}") for index in range(2)],
        TemplateDetectorConfig(),
    )

    results = detector.detect(make_frame(scene.frame_image))

    assert len(results) == 2
    assert all(isinstance(result, DetectionResult) for result in results)
    assert results[0].confidence >= results[1].confidence
    assert detector.name == "template"


def test_detect_result_fields_match_template() -> None:
    """结果字段来自所用模板（元素、类别、模板 ID）。"""
    scene = build_scene()
    detector = TemplateDetector([make_template(scene.template_image)], TemplateDetectorConfig())

    result = detector.detect(make_frame(scene.frame_image))[0]

    assert result.element_name == "explore"
    assert result.category == "buttons"
    assert result.template_id == TEMPLATE_ID
    assert result.element_key == "buttons/explore"
    assert result.confidence > 0.9


def test_detect_maps_rect_to_reference_coordinates() -> None:
    """命中区域必须从当前（窗口）坐标转换到参考分辨率坐标。"""
    scene = build_scene(paste=(50, 40))
    detector = TemplateDetector([make_template(scene.template_image)], TemplateDetectorConfig())

    result = detector.detect(make_frame(scene.frame_image))[0]

    expected = Rect(
        x=int(scene.rect_in_frame.x * REFERENCE_FACTOR),
        y=int(scene.rect_in_frame.y * REFERENCE_FACTOR),
        width=int(scene.rect_in_frame.width * REFERENCE_FACTOR),
        height=int(scene.rect_in_frame.height * REFERENCE_FACTOR),
    )
    assert result.rect == expected
    assert result.rect.width > scene.rect_in_frame.width


def test_detect_requires_coordinate_transform() -> None:
    """帧缺少坐标映射时必须报错，而不是输出无法解释的坐标。"""
    scene = build_scene()
    detector = TemplateDetector([make_template(scene.template_image)], TemplateDetectorConfig())

    with pytest.raises(MissingTransformError):
        detector.detect(make_frame(scene.frame_image, with_transform=False))


def test_detect_returns_empty_list_when_nothing_matches() -> None:
    """无命中时返回空列表。"""
    scene = build_scene(template_size=(32, 24))
    unrelated = np.random.default_rng(42).integers(0, 256, scene.frame_image.shape, dtype=np.uint8)
    detector = TemplateDetector([make_template(unrelated)], TemplateDetectorConfig())

    assert detector.detect(make_frame(scene.frame_image)) == []


def test_detector_requires_at_least_one_template() -> None:
    """模板集合为空时拒绝构造。"""
    with pytest.raises(ValueError, match="至少需要一个模板"):
        TemplateDetector([], TemplateDetectorConfig())


def test_detect_all_processes_every_frame() -> None:
    """detect_all 对多帧逐一检测。"""
    scene = build_scene()
    detector = TemplateDetector([make_template(scene.template_image)], TemplateDetectorConfig())
    frames = [make_frame(scene.frame_image), make_frame(scene.frame_image)]

    results = detector.detect_all(frames)

    assert len(results) == 2
    assert all(len(frame_results) == 1 for frame_results in results)


def test_best_per_element_keeps_highest_confidence() -> None:
    """按元素去重时保留置信度最高的结果。"""
    rect = Rect(x=0, y=0, width=10, height=10)
    results = [
        DetectionResult("explore", "buttons", 0.90, rect, "t1"),
        DetectionResult("explore", "buttons", 0.95, rect, "t2"),
        DetectionResult("confirm", "buttons", 0.80, rect, "t3"),
    ]

    best = TemplateDetector.best_per_element(results)

    assert [item.template_id for item in best] == ["t2", "t3"]


def test_best_per_element_handles_empty_input() -> None:
    """空输入返回空列表。"""
    assert TemplateDetector.best_per_element([]) == []


def test_from_dataset_loads_templates_from_directory(tmp_path: Path) -> None:
    """from_dataset 通过 loader 加载模板并可直接检测。"""
    scene = build_scene()
    target = tmp_path / "buttons" / "explore" / f"{TEMPLATE_ID}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(target), scene.template_image)

    detector = TemplateDetector.from_dataset(
        TemplateDetectorConfig(),
        loader=TemplateLoader(tmp_path),
    )
    results = detector.detect(make_frame(scene.frame_image))

    assert len(detector.templates) == 1
    assert results
    assert results[0].template_id == TEMPLATE_ID
