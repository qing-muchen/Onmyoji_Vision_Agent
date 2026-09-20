"""TemplateLoader 与模板文件名解析的单元测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.exceptions import TemplateLoadError, TemplateParseError
from src.core.paths import get_project_root
from src.vision.detector import TemplateLoader, parse_template_filename

TEMPLATE_SIZE = (20, 16)
DEFAULT_TEMPLATE_ID = "button_explore_skin001_scene001"


def write_template(directory: Path, name: str, *, value: int = 128) -> Path:
    """写入一张纯色 PNG 模板并返回其路径。"""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = TEMPLATE_SIZE
    image = np.full((height, width, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_parse_valid_filename() -> None:
    """按规范解析模板文件名。"""
    parsed = parse_template_filename(f"{DEFAULT_TEMPLATE_ID}.png")

    assert parsed.element_type == "button"
    assert parsed.element_name == "explore"
    assert parsed.skin == "001"
    assert parsed.scene == "001"


def test_parse_keeps_zero_padded_numbers() -> None:
    """skin/scene 保留三位零填充的字符串形式。"""
    parsed = parse_template_filename("button_explore_skin018_scene004.png")

    assert parsed.skin == "018"
    assert parsed.scene == "004"


@pytest.mark.parametrize(
    "filename",
    [
        "button_explore_skin1_scene001.png",  # skin 未补零
        "button_explore_skin001_scene1.png",  # scene 未补零
        "button_explore.png",  # 缺少 skin/scene
        "Button_Explore_skin001_scene001.png",  # 大写
        "button-explore-skin001-scene001.png",  # 使用连字符
        "button_explore_skin001_scene001.jpg",  # 非 PNG
        "button_explore_skin001_scene001",  # 无扩展名
        "explore_skin001_scene001.png",  # 缺少元素名称
    ],
)
def test_parse_rejects_invalid_filenames(filename: str) -> None:
    """不符合规范的文件名必须报错，而不是被猜测。"""
    with pytest.raises(TemplateParseError):
        parse_template_filename(filename)


def test_load_reads_templates_with_metadata(tmp_path: Path) -> None:
    """加载目录下的模板并填充元数据。"""
    write_template(tmp_path / "buttons" / "explore", f"{DEFAULT_TEMPLATE_ID}.png")
    write_template(tmp_path / "buttons" / "explore", "button_explore_skin002_scene003.png")

    templates = TemplateLoader(tmp_path).load_all()

    assert [template.template_id for template in templates] == [
        DEFAULT_TEMPLATE_ID,
        "button_explore_skin002_scene003",
    ]
    first = templates[0]
    assert first.element_type == "button"
    assert first.element_name == "explore"
    assert first.skin == "001"
    assert first.scene == "001"
    assert first.category == "buttons"
    assert first.image.shape == (TEMPLATE_SIZE[1], TEMPLATE_SIZE[0], 3)
    assert first.element_key == "buttons/explore"


def test_load_filters_by_metadata(tmp_path: Path) -> None:
    """支持按 category / element_type / element_name 过滤。"""
    write_template(tmp_path / "buttons" / "explore", f"{DEFAULT_TEMPLATE_ID}.png")
    write_template(tmp_path / "buttons" / "confirm", "button_confirm_skin001_scene001.png")
    write_template(tmp_path / "icons" / "auto", "icon_auto_skin001_scene001.png")

    loader = TemplateLoader(tmp_path)

    assert len(loader.load()) == 3
    assert len(loader.load(category="buttons")) == 2
    assert len(loader.load(element_type="icon")) == 1
    assert len(loader.load(element_name="explore")) == 1
    assert loader.load(element_name="missing") == []


def test_load_missing_root_raises(tmp_path: Path) -> None:
    """模板目录不存在时抛出 TemplateLoadError。"""
    with pytest.raises(TemplateLoadError, match="模板目录不存在"):
        TemplateLoader(tmp_path / "not_exists").load()


def test_load_invalid_filename_raises(tmp_path: Path) -> None:
    """目录中存在非法文件名时抛出 TemplateParseError。"""
    write_template(tmp_path / "buttons" / "explore", "explore_button_skin001_scene001.png")

    with pytest.raises(TemplateParseError):
        TemplateLoader(tmp_path).load()


def test_load_name_directory_mismatch_raises(tmp_path: Path) -> None:
    """文件名中的元素名称与所在目录不一致时抛出异常。"""
    write_template(tmp_path / "buttons" / "confirm", f"{DEFAULT_TEMPLATE_ID}.png")

    with pytest.raises(TemplateParseError, match="元素名称与目录不一致"):
        TemplateLoader(tmp_path).load()


def test_load_invalid_layout_depth_raises(tmp_path: Path) -> None:
    """层级不符合 <category>/<element_name>/<file> 时抛出异常。"""
    write_template(tmp_path, f"{DEFAULT_TEMPLATE_ID}.png")

    with pytest.raises(TemplateParseError, match="层级"):
        TemplateLoader(tmp_path).load()


def test_load_unreadable_image_raises(tmp_path: Path) -> None:
    """图像无法解码时抛出 TemplateLoadError。"""
    target = tmp_path / "buttons" / "explore" / f"{DEFAULT_TEMPLATE_ID}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a png")

    with pytest.raises(TemplateLoadError, match="读取失败"):
        TemplateLoader(tmp_path).load()


def test_load_real_dataset_when_available() -> None:
    """若本地已准备模板数据集，应能全部加载。"""
    dataset_root = get_project_root() / "dataset" / "templates"
    if not dataset_root.is_dir() or not any(dataset_root.rglob("*.png")):
        pytest.skip("本地未准备模板数据集（图片不入库）")

    loader = TemplateLoader()
    templates = loader.load(element_name="explore")

    assert templates
    assert all(template.element_name == "explore" for template in templates)
    assert loader.root.as_posix().endswith("dataset/templates")
