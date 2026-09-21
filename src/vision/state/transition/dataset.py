"""转场底部样本加载（``dataset/states/``，Phase 3.0）。

目录结构与数据形态约定::

    dataset/states/
    ├── transition_bottom/          # 正样本：**已裁剪**的底部条带（如 1706x58）
    │   └── transition_bottom_001.png ...
    └── non_transition_bottom/      # 负样本：**完整界面截图**（如 1706x961）
        └── 好友界面.png ...

★ 两类样本的形态**不同**：正样本已经是条带，负样本是整图。因此使用侧必须对负样本
补一次底部裁剪（见 :class:`~src.vision.state.transition.classifier.TransitionClassifier`），
正样本则直接使用。这一点是数据本身决定的，不做任何猜测。

★ 命名放宽：这两类样本由人工标注（文件名含中文、短横线等），因此本加载器只要求
``*.png``，``sample_id`` 取文件名（不含扩展名），用于日志与 ``detail`` 展示。

★ 图片不入 git（``.gitignore`` 忽略 ``dataset/**/*.png``），因此从 git 克隆的工作区
不含样本；缺失时抛出 :class:`~src.core.exceptions.StateDataError`，不静默返回空结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

from src.core.exceptions import StateDataError
from src.core.paths import resolve_from_project_root
from src.utils.logger import get_logger

__all__ = [
    "DEFAULT_STATE_ROOT",
    "NEGATIVE_DIRNAME",
    "POSITIVE_DIRNAME",
    "SampleKind",
    "TransitionSample",
    "TransitionSampleLoader",
]

#: 状态样本根目录（相对项目根）
DEFAULT_STATE_ROOT: str = "dataset/states"
#: 正样本（转场底部条带）目录名
POSITIVE_DIRNAME: str = "transition_bottom"
#: 负样本（非转场界面截图）目录名
NEGATIVE_DIRNAME: str = "non_transition_bottom"

logger = get_logger("vision.state.transition.dataset")


class SampleKind(StrEnum):
    """样本类型。"""

    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class TransitionSample:
    """一张底部条带样本及其元数据。

    说明：本类不生成 ``__eq__``（``image`` 为 numpy 数组，比较会产生布尔歧义），
    样本的唯一标识使用 :attr:`sample_id`。

    Attributes:
        sample_id: 样本 ID（文件名去掉扩展名），例如 ``transition_bottom_003``。
        kind: 样本类型（正/负）。
        path: 样本文件路径。
        image: 样本图像（BGR）。
    """

    sample_id: str
    kind: SampleKind
    path: Path
    image: np.ndarray

    def __repr__(self) -> str:
        height, width = self.image.shape[:2]
        return (
            f"TransitionSample(sample_id={self.sample_id!r}, kind={self.kind.value}, "
            f"size={width}x{height})"
        )

    @property
    def file_name(self) -> str:
        """样本文件名（含扩展名），用于 ``detail`` 展示。"""
        return self.path.name

    def describe(self) -> str:
        """返回可读描述。"""
        height, width = self.image.shape[:2]
        return f"{self.file_name} ({self.kind.value}, {width}x{height})"


def _imread(path: Path) -> np.ndarray | None:
    """读取图像，失败时返回 ``None``。

    先读字节再解码，以兼容非 ASCII 路径（负样本文件名含中文，``cv2.imread`` 在
    Windows 上不支持）。与 ``scene`` / ``detector`` 中的同类实现行为一致，但各层
    不互相导入私有函数。
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


class TransitionSampleLoader:
    """从 ``dataset/states`` 加载转场正/负样本。

    Args:
        root: 样本根目录；相对路径以项目根目录为基准。
    """

    def __init__(self, root: str | Path = DEFAULT_STATE_ROOT) -> None:
        self._root = resolve_from_project_root(root)

    @property
    def root(self) -> Path:
        """样本根目录（绝对路径）。"""
        return self._root

    @property
    def positive_dir(self) -> Path:
        """正样本目录（``transition_bottom``）。"""
        return self._root / POSITIVE_DIRNAME

    @property
    def negative_dir(self) -> Path:
        """负样本目录（``non_transition_bottom``）。"""
        return self._root / NEGATIVE_DIRNAME

    def load_positive(self) -> list[TransitionSample]:
        """加载全部正样本（已裁剪的底部条带）。

        Returns:
            正样本列表，按 ``sample_id`` 排序。

        Raises:
            StateDataError: 目录不存在、目录为空，或存在无法读取的图片。
        """
        return self._load_dir(self.positive_dir, SampleKind.POSITIVE, required=True)

    def load_negative(self) -> list[TransitionSample]:
        """加载全部负样本（完整界面截图）。

        负样本用于"否决"误判，缺失时**不报错**：返回空列表并记录一条告警，
        转场判定会退化为"仅正样本阈值"。

        Returns:
            负样本列表，按 ``sample_id`` 排序；目录缺失时为空列表。

        Raises:
            StateDataError: 目录存在但存在无法读取的图片。
        """
        if not self.negative_dir.is_dir():
            logger.warning(
                "负样本目录不存在，转场判定将不使用负样本否决: %s", self.negative_dir
            )
            return []
        samples = self._load_dir(self.negative_dir, SampleKind.NEGATIVE, required=False)
        if not samples:
            logger.warning("负样本目录为空，转场判定将不使用负样本否决: %s", self.negative_dir)
        return samples

    def load(self) -> tuple[list[TransitionSample], list[TransitionSample]]:
        """一次性加载正、负样本。

        Returns:
            ``(正样本列表, 负样本列表)``。
        """
        return self.load_positive(), self.load_negative()

    def _load_dir(
        self,
        directory: Path,
        kind: SampleKind,
        *,
        required: bool,
    ) -> list[TransitionSample]:
        """加载单个目录下的全部 PNG 样本。

        Args:
            directory: 样本目录。
            kind: 样本类型。
            required: 目录缺失/为空时是否视为错误。

        Returns:
            样本列表，按 ``sample_id`` 排序。

        Raises:
            StateDataError: ``required=True`` 且目录缺失或为空，或图片无法读取。
        """
        if not directory.is_dir():
            if required:
                raise StateDataError(f"状态样本目录不存在: {directory}")
            return []

        paths = sorted(path for path in directory.glob("*.png") if path.is_file())
        if not paths and required:
            raise StateDataError(f"状态样本目录为空: {directory}")

        samples: list[TransitionSample] = []
        for path in paths:
            image = _imread(path)
            if image is None:
                raise StateDataError(f"状态样本图像无法读取: {path}")
            samples.append(
                TransitionSample(sample_id=path.stem, kind=kind, path=path, image=image)
            )

        logger.info(
            "已加载%s样本 %d 个（%s）",
            "正" if kind is SampleKind.POSITIVE else "负",
            len(samples),
            directory.name,
        )
        return samples
