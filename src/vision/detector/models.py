"""检测器输入侧的数据模型：模板信息与文件名解析结果。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["TemplateFileName", "TemplateInfo"]


@dataclass(frozen=True, slots=True)
class TemplateFileName:
    """模板文件名的解析结果（对应文件名 ``<element_type>_<element_name>_skinNNN_sceneNNN.png``）。

    Attributes:
        element_type: 元素类型，例如 ``button``。
        element_name: 元素名称，例如 ``explore``。
        skin: 皮肤编号（保留三位零填充的字符串形式，例如 ``001``）。
        scene: 场景编号（保留三位零填充的字符串形式，例如 ``001``）。
    """

    element_type: str
    element_name: str
    skin: str
    scene: str


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class TemplateInfo:
    """一个已加载的模板及其元数据。

    说明：本类不生成 ``__eq__``（``image`` 为 numpy 数组，比较会产生布尔歧义）；
    模板的唯一标识请使用 :attr:`template_id`。

    Attributes:
        template_id: 模板唯一标识（文件名去掉扩展名）。
        element_type: 元素类型，例如 ``button``。
        element_name: 元素名称，例如 ``explore``。
        skin: 皮肤编号（字符串，三位零填充）。
        scene: 场景编号（字符串，三位零填充）。
        category: 模板类别目录名，例如 ``buttons``。
        path: 模板文件路径。
        image: 模板图像（BGR，``numpy.ndarray``）。
    """

    template_id: str
    element_type: str
    element_name: str
    skin: str
    scene: str
    category: str
    path: Path
    image: np.ndarray

    def __repr__(self) -> str:
        return (
            f"TemplateInfo(template_id={self.template_id!r}, category={self.category!r}, "
            f"size={self.image.shape[1]}x{self.image.shape[0]})"
        )

    @property
    def element_key(self) -> str:
        """``<category>/<element_name>`` 形式的元素键。"""
        return f"{self.category}/{self.element_name}"

    def describe(self) -> str:
        """返回可读的模板描述。"""
        height, width = self.image.shape[:2]
        return (
            f"{self.template_id} ({self.element_key}, {self.element_type}, "
            f"skin={self.skin}, scene={self.scene}, {width}x{height})"
        )
