"""场景模板数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["SceneFileName", "SceneTemplate"]


@dataclass(frozen=True, slots=True)
class SceneFileName:
    """场景文件名的解析结果（``<scene_type>_skinNNN_sceneNNN_frameNNN.png``）。

    Attributes:
        scene_type: 场景类型，例如 ``yard``。
        skin: 皮肤编号（整数）。
        scene_id: 场景编号（整数）。
        frame: 帧编号（整数）。
    """

    scene_type: str
    skin: int
    scene_id: int
    frame: int


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class SceneTemplate:
    """场景数据集中的一张截图及其元数据。

    :attr:`template_id` 是**场景级**标识（``yard_skin001_scene001``）：同一场景的多张截图
    共享同一个 ``template_id``，用 :attr:`frame` 区分；需要唯一标识单个文件时使用
    :attr:`source_id`。

    说明：本类不生成 ``__eq__``（``image`` 为 numpy 数组，比较会产生布尔歧义）。

    Attributes:
        template_id: 场景模板 ID（场景级，不含帧号）。
        scene_type: 场景类型，例如 ``yard``。
        skin: 皮肤编号（整数）。
        scene_id: 场景编号（整数）。
        frame: 帧编号（整数）—— 同一场景下的第几张截图。
        path: 场景图像路径。
        image: 场景图像（BGR）。
    """

    template_id: str
    scene_type: str
    skin: int
    scene_id: int
    frame: int
    path: Path
    image: np.ndarray

    def __repr__(self) -> str:
        height, width = self.image.shape[:2]
        return (
            f"SceneTemplate(template_id={self.template_id!r}, frame={self.frame}, "
            f"size={width}x{height})"
        )

    @property
    def source_id(self) -> str:
        """唯一标识单个文件的 ID：``<template_id>_frameNNN``。"""
        return f"{self.template_id}_frame{self.frame:03d}"

    @property
    def scene_key(self) -> str:
        """``<scene_type>/skin{NNN}/scene{NNN}`` 形式的场景键。"""
        return f"{self.scene_type}/skin{self.skin:03d}/scene{self.scene_id:03d}"

    def describe(self) -> str:
        """返回可读描述。"""
        height, width = self.image.shape[:2]
        return f"{self.source_id} ({self.scene_key}, {width}x{height})"
