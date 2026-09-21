"""SceneLoader 与场景文件名解析的单元测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.exceptions import SceneLoadError, SceneTemplateError
from src.core.paths import get_project_root
from src.vision.scene import SceneLoader, parse_scene_filename

SCENE_IMAGE_SIZE = (32, 24)


def write_scene(directory: Path, relative: str, *, value: int = 100) -> Path:
    """在目录下写入一张场景图并返回其路径。"""
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = SCENE_IMAGE_SIZE
    image = np.full((height, width, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_parse_valid_filename() -> None:
    """按规范解析场景文件名（编号转为整数）。"""
    parsed = parse_scene_filename("yard_skin001_scene002_frame003.png")

    assert parsed.scene_type == "yard"
    assert parsed.skin == 1
    assert parsed.scene_id == 2
    assert parsed.frame == 3


@pytest.mark.parametrize(
    "filename",
    [
        "yard_skin1_scene001_frame001.png",  # skin 未补零
        "yard_skin001_scene1_frame001.png",  # scene 未补零
        "yard_skin001_scene001_frame1.png",  # frame 未补零
        "yard_skin001_scene001.png",  # 缺少 frame
        "Yard_skin001_scene001_frame001.png",  # 大写
        "yard-skin001-scene001-frame001.png",  # 使用连字符
        "yard_skin001_scene001_frame001.jpg",  # 非 PNG
    ],
)
def test_parse_rejects_invalid_filename(filename: str) -> None:
    """不符合规范的文件名必须报错，而不是被猜测。"""
    with pytest.raises(SceneTemplateError):
        parse_scene_filename(filename)


def test_load_reads_scene_templates_with_metadata(tmp_path: Path) -> None:
    """加载场景目录并填充元数据（同场景多帧共享 template_id）。"""
    write_scene(tmp_path, "yard/skin_001/scene_001/yard_skin001_scene001_frame001.png")
    write_scene(tmp_path, "yard/skin_001/scene_001/yard_skin001_scene001_frame002.png", value=140)

    templates = SceneLoader(tmp_path).load_all()

    assert [template.source_id for template in templates] == [
        "yard_skin001_scene001_frame001",
        "yard_skin001_scene001_frame002",
    ]
    first = templates[0]
    assert first.template_id == "yard_skin001_scene001"
    assert first.scene_type == "yard"
    assert first.skin == 1
    assert first.scene_id == 1
    assert first.frame == 1
    assert first.scene_key == "yard/skin001/scene001"
    assert first.image.shape == (SCENE_IMAGE_SIZE[1], SCENE_IMAGE_SIZE[0], 3)
    assert templates[1].template_id == first.template_id


def test_load_filters_by_metadata(tmp_path: Path) -> None:
    """支持按 scene_type / skin / scene_id 过滤。"""
    write_scene(tmp_path, "yard/skin_001/scene_001/yard_skin001_scene001_frame001.png")
    write_scene(tmp_path, "yard/skin_002/scene_003/yard_skin002_scene003_frame001.png")

    loader = SceneLoader(tmp_path)

    assert len(loader.load()) == 2
    assert len(loader.load(skin=1)) == 1
    assert len(loader.load(scene_id=3)) == 1
    assert len(loader.load(scene_type="yard")) == 2
    assert loader.load(skin=9) == []
    assert loader.load(scene_type="battle") == []


def test_load_missing_root_raises(tmp_path: Path) -> None:
    """场景目录不存在时抛出 SceneLoadError。"""
    with pytest.raises(SceneLoadError, match="场景数据目录不存在"):
        SceneLoader(tmp_path / "not_exists").load()


def test_load_invalid_layout_depth_raises(tmp_path: Path) -> None:
    """层级不符合 <scene_type>/skin_NNN/scene_NNN/<file> 时抛出异常。"""
    write_scene(tmp_path, "yard_skin001_scene001_frame001.png")

    with pytest.raises(SceneTemplateError, match="层级"):
        SceneLoader(tmp_path).load()


@pytest.mark.parametrize(
    "relative",
    [
        "yard/skin_1/scene_001/yard_skin001_scene001_frame001.png",  # skin 目录未补零
        "yard/skin_001/scene1/yard_skin001_scene001_frame001.png",  # scene 目录未补零
        "yard/skins_001/scene_001/yard_skin001_scene001_frame001.png",  # 目录命名非法
    ],
)
def test_load_invalid_directory_name_raises(tmp_path: Path, relative: str) -> None:
    """目录名不符合 skin_NNN / scene_NNN 时抛出异常。"""
    write_scene(tmp_path, relative)

    with pytest.raises(SceneTemplateError):
        SceneLoader(tmp_path).load()


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("yard/skin_001/scene_001/other_skin001_scene001_frame001.png", "场景类型与目录不一致"),
        ("yard/skin_001/scene_001/yard_skin002_scene001_frame001.png", "skin 编号与目录不一致"),
        ("yard/skin_001/scene_001/yard_skin001_scene003_frame001.png", "scene 编号与目录不一致"),
    ],
)
def test_load_number_mismatch_raises(tmp_path: Path, relative: str, message: str) -> None:
    """文件名中的编号必须与目录一致。"""
    write_scene(tmp_path, relative)

    with pytest.raises(SceneTemplateError, match=message):
        SceneLoader(tmp_path).load()


def test_load_unreadable_image_raises(tmp_path: Path) -> None:
    """图像无法解码时抛出 SceneLoadError。"""
    target = tmp_path / "yard" / "skin_001" / "scene_001" / "yard_skin001_scene001_frame001.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a png")

    with pytest.raises(SceneLoadError, match="读取失败"):
        SceneLoader(tmp_path).load()


def test_load_real_dataset_when_available() -> None:
    """若本地已准备场景数据集，应能全部加载。"""
    dataset_root = get_project_root() / "dataset" / "scenes"
    if not dataset_root.is_dir() or not any(dataset_root.rglob("*.png")):
        pytest.skip("本地未准备场景数据集（图片不入库）")

    templates = SceneLoader().load()

    assert len(templates) >= 2
    # 场景级 template_id 数量应少于文件数（同一场景存在多张截图）
    assert len({template.template_id for template in templates}) < len(templates)
    assert all(template.scene_type for template in templates)
