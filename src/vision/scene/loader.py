"""场景数据集加载（``dataset/scenes/``）。

目录结构与 Phase 2.0 的规范一致::

    dataset/scenes/<scene_type>/skin_NNN/scene_NNN/<scene_type>_skinNNN_sceneNNN_frameNNN.png

命名或结构不符合规范时**直接抛出异常**，不做任何猜测。
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from src.core.exceptions import SceneLoadError, SceneTemplateError
from src.core.paths import resolve_from_project_root
from src.utils.logger import get_logger
from src.vision.scene.models import SceneFileName, SceneTemplate

__all__ = [
    "DEFAULT_SCENE_ROOT",
    "SCENE_FILENAME_PATTERN",
    "SCENE_RELATIVE_DEPTH",
    "SceneLoader",
    "parse_scene_filename",
]

#: 场景根目录（相对项目根）
DEFAULT_SCENE_ROOT: str = "dataset/scenes"
#: 场景文件相对层级：``<scene_type>/skin_NNN/scene_NNN/<file>``
SCENE_RELATIVE_DEPTH: int = 4
#: 场景文件名规范（与 ``dataset/README.md``、``tools/check_dataset.py`` 保持一致）
SCENE_FILENAME_PATTERN = re.compile(
    r"^(?P<scene_type>[a-z][a-z0-9]*)_skin(?P<skin>\d{3})_scene(?P<scene>\d{3})"
    r"_frame(?P<frame>\d{3})\.png$"
)
#: 目录命名规范
SKIN_DIR_PATTERN = re.compile(r"^skin_(?P<skin>\d{3})$")
SCENE_DIR_PATTERN = re.compile(r"^scene_(?P<scene>\d{3})$")

logger = get_logger("vision.scene.loader")


def parse_scene_filename(name: str) -> SceneFileName:
    """解析场景文件名。

    Args:
        name: 文件名（含扩展名），例如 ``yard_skin001_scene001_frame001.png``。

    Returns:
        解析出的各字段。

    Raises:
        SceneTemplateError: 文件名不符合规范（不做任何猜测）。
    """
    matched = SCENE_FILENAME_PATTERN.match(name)
    if matched is None:
        raise SceneTemplateError(
            f"场景文件名无法解析: {name!r}"
            "（期望 <scene_type>_skinNNN_sceneNNN_frameNNN.png）"
        )
    return SceneFileName(
        scene_type=matched.group("scene_type"),
        skin=int(matched.group("skin")),
        scene_id=int(matched.group("scene")),
        frame=int(matched.group("frame")),
    )


def _imread(path: Path) -> np.ndarray | None:
    """读取图像，失败时返回 ``None``（先读字节再解码，兼容非 ASCII 路径）。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


class SceneLoader:
    """从场景目录加载场景模板。

    Args:
        root: 场景根目录；相对路径以项目根目录为基准。
    """

    def __init__(self, root: str | Path = DEFAULT_SCENE_ROOT) -> None:
        self._root = resolve_from_project_root(root)

    @property
    def root(self) -> Path:
        """场景根目录（绝对路径）。"""
        return self._root

    def load_all(self) -> list[SceneTemplate]:
        """加载全部场景模板。

        Returns:
            场景模板列表，按 ``source_id`` 排序。
        """
        return self.load()

    def load(
        self,
        *,
        scene_type: str | None = None,
        skin: int | None = None,
        scene_id: int | None = None,
    ) -> list[SceneTemplate]:
        """按条件加载场景模板。

        所有文件都会先做命名与结构校验（即便随后被过滤），以保证数据集质量。

        Args:
            scene_type: 只加载该场景类型，例如 ``yard``。
            skin: 只加载该皮肤编号。
            scene_id: 只加载该场景编号。

        Returns:
            场景模板列表。

        Raises:
            SceneLoadError: 场景目录不存在或图像无法读取。
            SceneTemplateError: 目录层级、目录名或文件名不符合规范，或编号与目录不一致。
        """
        if not self._root.is_dir():
            raise SceneLoadError(f"场景数据目录不存在: {self._root}")

        templates: list[SceneTemplate] = []
        for path in sorted(self._root.rglob("*.png")):
            relative = path.relative_to(self._root)
            directory_type, directory_skin, directory_scene = self._validate_layout(relative)
            parsed = parse_scene_filename(path.name)
            self._validate_consistency(
                relative,
                (directory_type, directory_skin, directory_scene),
                parsed,
            )
            if scene_type is not None and parsed.scene_type != scene_type:
                continue
            if skin is not None and parsed.skin != skin:
                continue
            if scene_id is not None and parsed.scene_id != scene_id:
                continue
            templates.append(self._read(path, parsed))

        logger.info(
            "已加载场景模板 %d 个（root=%s, scene_type=%s, skin=%s, scene_id=%s）",
            len(templates),
            self._root,
            scene_type or "*",
            f"{skin:03d}" if skin is not None else "*",
            f"{scene_id:03d}" if scene_id is not None else "*",
        )
        return templates

    def _validate_layout(self, relative: Path) -> tuple[str, int, int]:
        """校验目录层级与目录名。

        Args:
            relative: 相对场景根目录的路径。

        Returns:
            ``(scene_type, skin, scene_id)``。

        Raises:
            SceneTemplateError: 层级或目录名不符合 ``<scene_type>/skin_NNN/scene_NNN/<file>``。
        """
        parts = relative.parts
        if len(parts) != SCENE_RELATIVE_DEPTH:
            raise SceneTemplateError(
                f"场景目录层级不符合 <scene_type>/skin_NNN/scene_NNN/<file>: {relative.as_posix()}"
            )
        scene_type, skin_dir, scene_dir, _name = parts
        skin_matched = SKIN_DIR_PATTERN.match(skin_dir)
        if skin_matched is None:
            raise SceneTemplateError(
                f"皮肤目录名应为 skin_NNN: {relative.as_posix()}（实际 {skin_dir!r}）"
            )
        scene_matched = SCENE_DIR_PATTERN.match(scene_dir)
        if scene_matched is None:
            raise SceneTemplateError(
                f"场景目录名应为 scene_NNN: {relative.as_posix()}（实际 {scene_dir!r}）"
            )
        return scene_type, int(skin_matched.group("skin")), int(scene_matched.group("scene"))

    @staticmethod
    def _validate_consistency(
        relative: Path,
        directory: tuple[str, int, int],
        parsed: SceneFileName,
    ) -> None:
        """校验文件名中的编号与所在目录一致。

        Raises:
            SceneTemplateError: 任一项不一致。
        """
        scene_type, skin, scene_id = directory
        if parsed.scene_type != scene_type:
            raise SceneTemplateError(
                f"文件名中的场景类型与目录不一致: {relative.as_posix()}"
                f"（目录 {scene_type!r} != 文件名 {parsed.scene_type!r}）"
            )
        if parsed.skin != skin:
            raise SceneTemplateError(
                f"文件名中的 skin 编号与目录不一致: {relative.as_posix()}"
                f"（目录 {skin:03d} != 文件名 {parsed.skin:03d}）"
            )
        if parsed.scene_id != scene_id:
            raise SceneTemplateError(
                f"文件名中的 scene 编号与目录不一致: {relative.as_posix()}"
                f"（目录 {scene_id:03d} != 文件名 {parsed.scene_id:03d}）"
            )

    @staticmethod
    def _read(path: Path, parsed: SceneFileName) -> SceneTemplate:
        """读取场景图像并组装 :class:`~src.vision.scene.models.SceneTemplate`。

        Raises:
            SceneLoadError: 图像无法读取。
        """
        image = _imread(path)
        if image is None:
            raise SceneLoadError(f"场景图像读取失败: {path}")
        return SceneTemplate(
            template_id=f"{parsed.scene_type}_skin{parsed.skin:03d}_scene{parsed.scene_id:03d}",
            scene_type=parsed.scene_type,
            skin=parsed.skin,
            scene_id=parsed.scene_id,
            frame=parsed.frame,
            path=path,
            image=image,
        )
