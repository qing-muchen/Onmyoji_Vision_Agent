"""场景识别结果模型。

场景识别是**整图判断**，只回答"当前在哪里"，因此结果不包含任何坐标或矩形
（与 detector 的定位型结果不同）。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MAX_CONFIDENCE", "MIN_CONFIDENCE", "SceneResult"]

#: 允许的最小置信度
MIN_CONFIDENCE: float = 0.0
#: 允许的最大置信度
MAX_CONFIDENCE: float = 1.0


@dataclass(frozen=True, slots=True)
class SceneResult:
    """一次场景识别命中的结果。

    Attributes:
        scene_type: 场景类型，例如 ``yard``。
        skin: 皮肤编号（整数，例如 ``1``）；展示时可格式化为 ``001``。
        scene_id: 场景编号（整数，例如 ``1``）。
        confidence: 匹配置信度，范围 ``[0, 1]``。
        template_id: 命中的场景模板 ID，例如 ``yard_skin001_scene001``。

    Raises:
        ValueError: 置信度越界或字段非法。
    """

    scene_type: str
    skin: int
    scene_id: int
    confidence: float
    template_id: str

    def __post_init__(self) -> None:
        if not self.scene_type:
            raise ValueError("scene_type 不能为空")
        if self.skin < 1:
            raise ValueError(f"skin 必须为正整数: {self.skin}")
        if self.scene_id < 1:
            raise ValueError(f"scene_id 必须为正整数: {self.scene_id}")
        if not self.template_id:
            raise ValueError("template_id 不能为空")
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence 必须落在 [{MIN_CONFIDENCE:g}, {MAX_CONFIDENCE:g}] 区间内: {self.confidence}"
            )

    @property
    def scene_key(self) -> str:
        """``<scene_type>/skin{NNN}/scene{NNN}`` 形式的场景键。"""
        return f"{self.scene_type}/skin{self.skin:03d}/scene{self.scene_id:03d}"

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"scene_type={self.scene_type} skin={self.skin:03d} scene={self.scene_id:03d} "
            f"confidence={self.confidence:.4f} template={self.template_id}"
        )
