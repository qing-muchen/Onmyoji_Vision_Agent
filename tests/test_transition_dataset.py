"""TransitionSampleLoader 的单元测试：宽松命名、缺失降级与错误处理。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.exceptions import StateDataError
from src.vision.state.transition.dataset import (
    NEGATIVE_DIRNAME,
    POSITIVE_DIRNAME,
    SampleKind,
    TransitionSampleLoader,
)

#: 条带样本尺寸（模拟真实数据：宽而扁）
STRIP_SIZE = (1706, 58)
#: 完整界面样本尺寸
FRAME_SIZE = (1706, 961)


def write_png(path: Path, size: tuple[int, int] = STRIP_SIZE) -> Path:
    """写出一个合法 PNG（路径可含中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[1], size[0], 3), 32, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    buffer.tofile(str(path))
    return path


def build_loader(tmp_path: Path) -> TransitionSampleLoader:
    """构造指向临时目录的加载器。"""
    return TransitionSampleLoader(tmp_path)


def test_load_positive_reads_png_and_sorts_by_name(tmp_path: Path) -> None:
    """正样本按文件名排序加载，sample_id 为文件名（不含扩展名）。"""
    write_png(tmp_path / POSITIVE_DIRNAME / "transition_bottom_002.png")
    write_png(tmp_path / POSITIVE_DIRNAME / "transition_bottom_001.png")
    loader = build_loader(tmp_path)

    samples = loader.load_positive()

    assert [sample.sample_id for sample in samples] == [
        "transition_bottom_001",
        "transition_bottom_002",
    ]
    assert all(sample.kind is SampleKind.POSITIVE for sample in samples)
    assert samples[0].image.shape == (STRIP_SIZE[1], STRIP_SIZE[0], 3)


def test_load_negative_accepts_non_ascii_names(tmp_path: Path) -> None:
    """负样本文件名允许中文（人工标注），不会因命名不规范而失败。"""
    write_png(tmp_path / NEGATIVE_DIRNAME / "好友界面.png", FRAME_SIZE)
    write_png(tmp_path / NEGATIVE_DIRNAME / "战斗准备阶段.png", FRAME_SIZE)
    loader = build_loader(tmp_path)

    samples = loader.load_negative()

    assert [sample.sample_id for sample in samples] == ["好友界面", "战斗准备阶段"]
    assert samples[0].file_name == "好友界面.png"
    assert all(sample.kind is SampleKind.NEGATIVE for sample in samples)


def test_positive_directory_missing_raises(tmp_path: Path) -> None:
    """启用转场识别却没有正样本目录时必须明确报错（不静默返回空）。"""
    with pytest.raises(StateDataError, match="不存在"):
        build_loader(tmp_path).load_positive()


def test_positive_directory_empty_raises(tmp_path: Path) -> None:
    """正样本目录为空同样视为错误。"""
    (tmp_path / POSITIVE_DIRNAME).mkdir(parents=True)

    with pytest.raises(StateDataError, match="为空"):
        build_loader(tmp_path).load_positive()


def test_negative_directory_missing_degrades(tmp_path: Path) -> None:
    """负样本目录缺失时降级为空列表（不影响转场识别可用性）。"""
    write_png(tmp_path / POSITIVE_DIRNAME / "transition_bottom_001.png")

    assert build_loader(tmp_path).load_negative() == []


def test_negative_directory_empty_degrades(tmp_path: Path) -> None:
    """负样本目录为空时同样降级为空列表。"""
    (tmp_path / NEGATIVE_DIRNAME).mkdir(parents=True)

    assert build_loader(tmp_path).load_negative() == []


def test_invalid_image_raises_state_data_error(tmp_path: Path) -> None:
    """损坏的图片文件必须报错，而不是被当成"没有样本"。"""
    broken = tmp_path / POSITIVE_DIRNAME / "transition_bottom_001.png"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"not a png")

    with pytest.raises(StateDataError, match="无法读取"):
        build_loader(tmp_path).load_positive()


def test_non_png_files_are_ignored(tmp_path: Path) -> None:
    """目录内的非 PNG 文件（例如档案、说明）不参与加载。"""
    write_png(tmp_path / POSITIVE_DIRNAME / "transition_bottom_001.png")
    (tmp_path / POSITIVE_DIRNAME / "README.md").write_text("note", encoding="utf-8")
    (tmp_path / NEGATIVE_DIRNAME).mkdir(parents=True, exist_ok=True)
    (tmp_path / NEGATIVE_DIRNAME / "archive.zip").write_bytes(b"zip")

    loader = build_loader(tmp_path)

    assert len(loader.load_positive()) == 1
    assert loader.load_negative() == []


def test_load_returns_both_kinds(tmp_path: Path) -> None:
    """load() 一次返回正、负样本。"""
    write_png(tmp_path / POSITIVE_DIRNAME / "transition_bottom_001.png")
    write_png(tmp_path / NEGATIVE_DIRNAME / "庭院界面.png", FRAME_SIZE)

    positives, negatives = build_loader(tmp_path).load()

    assert len(positives) == 1
    assert len(negatives) == 1


def test_directory_properties_point_to_expected_subdirs(tmp_path: Path) -> None:
    """目录属性指向规范约定的子目录。"""
    loader = build_loader(tmp_path)

    assert loader.positive_dir == tmp_path / POSITIVE_DIRNAME
    assert loader.negative_dir == tmp_path / NEGATIVE_DIRNAME
    assert loader.root == tmp_path


def test_sample_describe_and_repr(tmp_path: Path) -> None:
    """样本提供可读描述，便于日志与 detail 展示。"""
    write_png(tmp_path / POSITIVE_DIRNAME / "transition_bottom_001.png")
    sample = build_loader(tmp_path).load_positive()[0]

    assert "transition_bottom_001.png" in sample.describe()
    assert "positive" in sample.describe()
    assert "TransitionSample" in repr(sample)
