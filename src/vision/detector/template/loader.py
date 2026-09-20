"""模板数据集加载（``dataset/templates/``）。

目录结构与 Phase 2.0 的规范一致::

    dataset/templates/<category>/<element_name>/<element_type>_<element_name>_skinNNN_sceneNNN.png

命名或结构不符合规范时**直接抛出异常**，不做任何猜测。
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from src.core.exceptions import TemplateLoadError, TemplateParseError
from src.core.paths import resolve_from_project_root
from src.utils.logger import get_logger
from src.vision.detector.models import TemplateFileName, TemplateInfo

__all__ = [
    "DEFAULT_TEMPLATE_ROOT",
    "TEMPLATE_FILENAME_PATTERN",
    "TEMPLATE_RELATIVE_DEPTH",
    "TemplateLoader",
    "parse_template_filename",
]

#: 模板根目录（相对项目根）
DEFAULT_TEMPLATE_ROOT: str = "dataset/templates"
#: 模板相对层级：``<category>/<element_name>/<file>``
TEMPLATE_RELATIVE_DEPTH: int = 3
#: 模板文件名规范（与 ``dataset/README.md``、``tools/check_dataset.py`` 保持一致）
TEMPLATE_FILENAME_PATTERN = re.compile(
    r"^(?P<element_type>[a-z][a-z0-9]*)_(?P<element_name>[a-z][a-z0-9]*)"
    r"_skin(?P<skin>\d{3})_scene(?P<scene>\d{3})\.png$"
)

logger = get_logger("vision.detector.template.loader")


def parse_template_filename(name: str) -> TemplateFileName:
    """解析模板文件名。

    Args:
        name: 文件名（含扩展名），例如 ``button_explore_skin001_scene001.png``。

    Returns:
        解析出的各字段。

    Raises:
        TemplateParseError: 文件名不符合规范（不做任何猜测）。
    """
    matched = TEMPLATE_FILENAME_PATTERN.match(name)
    if matched is None:
        raise TemplateParseError(
            f"模板文件名无法解析: {name!r}"
            "（期望 <element_type>_<element_name>_skinNNN_sceneNNN.png）"
        )
    return TemplateFileName(
        element_type=matched.group("element_type"),
        element_name=matched.group("element_name"),
        skin=matched.group("skin"),
        scene=matched.group("scene"),
    )


def _imread(path: Path) -> np.ndarray | None:
    """读取图像，失败时返回 ``None``。

    先读字节再解码，以兼容非 ASCII 路径（OpenCV 的 ``imread`` 在 Windows 上不支持）。
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


class TemplateLoader:
    """从模板目录加载模板。

    Args:
        root: 模板根目录；相对路径以项目根目录为基准。
    """

    def __init__(self, root: str | Path = DEFAULT_TEMPLATE_ROOT) -> None:
        self._root = resolve_from_project_root(root)

    @property
    def root(self) -> Path:
        """模板根目录（绝对路径）。"""
        return self._root

    def load_all(self) -> list[TemplateInfo]:
        """加载全部模板。

        Returns:
            模板列表，按 ``template_id`` 排序。
        """
        return self.load()

    def load(
        self,
        *,
        category: str | None = None,
        element_type: str | None = None,
        element_name: str | None = None,
    ) -> list[TemplateInfo]:
        """按条件加载模板。

        所有模板都会先做命名与结构校验（即便随后被过滤），以保证数据集质量。

        Args:
            category: 只加载该类别目录，例如 ``buttons``。
            element_type: 只加载该元素类型，例如 ``button``。
            element_name: 只加载该元素名称，例如 ``explore``。

        Returns:
            模板列表，按 ``template_id`` 排序。

        Raises:
            TemplateLoadError: 模板目录不存在或模板图像无法读取。
            TemplateParseError: 目录层级或文件名不符合规范。
        """
        if not self._root.is_dir():
            raise TemplateLoadError(f"模板目录不存在: {self._root}")

        templates: list[TemplateInfo] = []
        for path in sorted(self._root.rglob("*.png")):
            relative = path.relative_to(self._root)
            file_category, directory_element = self._validate_layout(relative)
            parsed = parse_template_filename(path.name)
            if parsed.element_name != directory_element:
                raise TemplateParseError(
                    f"模板文件名中的元素名称与目录不一致: {relative.as_posix()}"
                    f"（目录 {directory_element!r} != 文件名 {parsed.element_name!r}）"
                )
            if category is not None and file_category != category:
                continue
            if element_type is not None and parsed.element_type != element_type:
                continue
            if element_name is not None and parsed.element_name != element_name:
                continue
            templates.append(self._read(path, file_category, parsed))

        logger.info(
            "已加载模板 %d 个（root=%s, category=%s, element_type=%s, element_name=%s）",
            len(templates),
            self._root,
            category or "*",
            element_type or "*",
            element_name or "*",
        )
        return templates

    def _validate_layout(self, relative: Path) -> tuple[str, str]:
        """校验模板的相对层级。

        Args:
            relative: 相对模板根目录的路径。

        Returns:
            ``(category, element_name_dir)``。

        Raises:
            TemplateParseError: 层级不符合 ``<category>/<element_name>/<file>``。
        """
        parts = relative.parts
        if len(parts) != TEMPLATE_RELATIVE_DEPTH:
            raise TemplateParseError(
                f"模板目录层级不符合 <category>/<element_name>/<file>: {relative.as_posix()}"
            )
        return parts[0], parts[1]

    @staticmethod
    def _read(path: Path, category: str, parsed: TemplateFileName) -> TemplateInfo:
        """读取模板图像并组装 :class:`~src.vision.detector.models.TemplateInfo`。

        Raises:
            TemplateLoadError: 图像无法读取或为空。
        """
        image = _imread(path)
        if image is None:
            raise TemplateLoadError(f"模板图像读取失败: {path}")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise TemplateLoadError(f"模板图像尺寸非法: {path} {image.shape}")
        return TemplateInfo(
            template_id=path.stem,
            element_type=parsed.element_type,
            element_name=parsed.element_name,
            skin=parsed.skin,
            scene=parsed.scene,
            category=category,
            path=path,
            image=image,
        )
