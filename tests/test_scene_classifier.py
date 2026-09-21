"""SceneClassifier 的单元测试：返回 SceneResult、多模板组合、best_match 与输入校验。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import SceneMatcherConfig
from src.vision.capture.models import Frame
from src.vision.scene import SceneClassifier, SceneLoader, SceneResult, SceneTemplate
from tests.helpers import build_scene_image, perturb_image

CONFIG = SceneMatcherConfig(threshold=0.5, resize_width=64, resize_height=36)
STRICT_CONFIG = SceneMatcherConfig(threshold=0.99, resize_width=64, resize_height=36)


def make_template(
    image: np.ndarray,
    *,
    scene_type: str = "yard",
    skin: int = 1,
    scene_id: int = 1,
    frame: int = 1,
) -> SceneTemplate:
    """由图像构造场景模板（不读取磁盘）。"""
    template_id = f"{scene_type}_skin{skin:03d}_scene{scene_id:03d}"
    return SceneTemplate(
        template_id=template_id,
        scene_type=scene_type,
        skin=skin,
        scene_id=scene_id,
        frame=frame,
        path=Path(f"{template_id}_frame{frame:03d}.png"),
        image=image,
    )


def make_frame(image: np.ndarray) -> Frame:
    """构造场景帧（场景识别是整图判断，不需要坐标映射）。"""
    return Frame.create(image, source="test")


def test_classify_returns_scene_result() -> None:
    """输出 SceneResult，字段来自命中的场景模板。"""
    image = build_scene_image(seed=1)
    classifier = SceneClassifier([make_template(image)], CONFIG)

    results = classifier.classify(make_frame(image))

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, SceneResult)
    assert result.scene_type == "yard"
    assert result.skin == 1
    assert result.scene_id == 1
    assert result.confidence == pytest.approx(1.0)
    assert result.template_id == "yard_skin001_scene001"
    assert result.scene_key == "yard/skin001/scene001"
    assert classifier.name == "scene"


def test_classify_sorts_results_by_confidence() -> None:
    """结果按 confidence 降序排列。"""
    reference = build_scene_image(seed=2)
    classifier = SceneClassifier(
        [
            make_template(reference, scene_id=2),
            make_template(reference, scene_id=1),
            make_template(build_scene_image(seed=77), scene_id=3),
        ],
        SceneMatcherConfig(threshold=0.01, resize_width=64, resize_height=36),
    )

    results = classifier.classify(make_frame(reference))

    confidences = [result.confidence for result in results]
    assert confidences == sorted(confidences, reverse=True)
    assert results[0].scene_id == 1 or results[0].scene_id == 2
    assert results[0].confidence == pytest.approx(1.0)


def test_multi_frame_templates_take_highest_confidence() -> None:
    """同一场景的多张截图取组内最高置信度（单结果）。"""
    base = build_scene_image(seed=3)
    variant = perturb_image(base, noise=40, block_ratio=0.5, seed=4)
    classifier = SceneClassifier(
        [make_template(base, frame=1), make_template(variant, frame=2)],
        CONFIG,
    )

    results = classifier.classify(make_frame(base))

    assert len(results) == 1
    assert results[0].template_id == "yard_skin001_scene001"
    assert results[0].confidence == pytest.approx(1.0)


def test_different_scenes_are_reported_separately() -> None:
    """不同场景分别给出结果。"""
    base = build_scene_image(seed=5)
    classifier = SceneClassifier(
        [
            make_template(base, scene_id=1),
            make_template(base, skin=2, scene_id=1),
        ],
        CONFIG,
    )

    results = classifier.classify(make_frame(base))

    assert {result.scene_key for result in results} == {
        "yard/skin001/scene001",
        "yard/skin002/scene001",
    }


def test_threshold_filters_low_confidence() -> None:
    """低于阈值的场景不输出。"""
    base = build_scene_image(seed=6)
    classifier = SceneClassifier([make_template(base)], STRICT_CONFIG)

    diverged = make_frame(perturb_image(base, noise=30, block_ratio=0.3, seed=7))

    assert classifier.classify(diverged) == []
    assert classifier.best_match(diverged) is None


def test_best_match_returns_top_scene() -> None:
    """best_match 返回置信度最高的场景。"""
    reference = build_scene_image(seed=8)
    classifier = SceneClassifier(
        [
            make_template(build_scene_image(seed=99), scene_id=1),
            make_template(reference, scene_id=2),
        ],
        SceneMatcherConfig(threshold=0.01, resize_width=64, resize_height=36),
    )

    best = classifier.best_match(make_frame(reference))

    assert best is not None
    assert best.scene_id == 2
    assert best.confidence == pytest.approx(1.0)


def test_empty_templates_are_rejected() -> None:
    """空模板集合拒绝构造。"""
    with pytest.raises(ValueError, match="至少需要一个场景模板"):
        SceneClassifier([], CONFIG)


def test_classify_rejects_non_frame_input() -> None:
    """输入必须是 Frame（不允许裸 numpy 图像）。"""
    classifier = SceneClassifier([make_template(build_scene_image(seed=9))], CONFIG)

    with pytest.raises(TypeError, match="Frame"):
        classifier.classify(np.zeros((10, 10, 3), dtype=np.uint8))  # type: ignore[arg-type]


def test_classify_does_not_require_coordinate_transform() -> None:
    """场景识别是整图判断，不要求帧携带坐标映射。"""
    image = build_scene_image(seed=10)
    frame = make_frame(image)
    classifier = SceneClassifier([make_template(image)], CONFIG)

    assert frame.coordinate_transform is None
    assert classifier.classify(frame)


def test_from_dataset_loads_templates_and_classifies(tmp_path: Path) -> None:
    """from_dataset 通过 loader 加载场景模板并可直接分类。"""
    image = build_scene_image(seed=11)
    for frame_index in (1, 2):
        target = (
            tmp_path
            / "yard"
            / "skin_001"
            / "scene_001"
            / f"yard_skin001_scene001_frame{frame_index:03d}.png"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(target), image)

    classifier = SceneClassifier.from_dataset(CONFIG, loader=SceneLoader(tmp_path))
    results = classifier.classify(make_frame(image))

    assert len(classifier.templates) == 2
    assert len(results) == 1
    assert results[0].template_id == "yard_skin001_scene001"
