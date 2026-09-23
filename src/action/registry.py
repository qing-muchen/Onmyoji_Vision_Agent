"""语义目标注册表（Phase 7）：语义名 → 窗口相对位置。

Phase 7 的定位策略是**静态注册表**（人工声明 + 标定），因此：

* 注册表是**唯一**的语义名 → 比例来源，未注册的目标**解析失败**（绝不猜测坐标）；
* 每条记录带 :attr:`TargetEntry.calibrated` 标记：**未标定**（占位值）的目标在真实输入
  后端下会被安全门拒绝（见 :class:`~src.action.safety.guard.SafetyGuard`）；
* 允许注册任意自定义名称（例如 ``buttons/explore``），不限于 :class:`~src.action.models.ActionTarget`
  枚举里的内置名（枚举只是规范拼写，不是白名单）；
* 注册表**不读像素、不调 detector**（Phase 7.1 的 vision-aware 解析器是它的替代实现，
  而不是它的内部依赖）。

坐标一律以**窗口客户区相对比例**（``x_ratio`` / ``y_ratio``，0..1）表达，
实际屏幕坐标由 :meth:`ResolvedTarget.client_point` 用窗口客户区矩形换算。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from src.action.models import (
    ActionTarget,
    ActionTargetError,
    ResolvedTarget,
    validate_target_name,
)
from src.utils.logger import get_logger

__all__ = ["BUILTIN_TARGETS", "TargetEntry", "TargetRegistry", "registry_from_mapping"]

logger = get_logger("action.registry")


@dataclass(frozen=True, slots=True)
class TargetEntry:
    """注册表中的一条语义目标记录。

    Attributes:
        name: 语义目标名。
        x_ratio: 窗口客户区横向比例（0..1）。
        y_ratio: 窗口客户区纵向比例（0..1）。
        confidence: 声明置信度（0..1；静态注册表通常为 1.0）。
        calibrated: 是否已实机标定（占位值必须为 ``False``）。
        description: 说明（人类可读）。
        source: 来源标记（写入解析结果）。

    Raises:
        ActionTargetError: 名称非法、比例 / 置信度越界。
    """

    name: str
    x_ratio: float
    y_ratio: float
    confidence: float = 1.0
    calibrated: bool = False
    description: str = ""
    source: str = "registry"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_target_name(self.name))
        for label, value in (("x_ratio", self.x_ratio), ("y_ratio", self.y_ratio)):
            if not 0.0 <= float(value) <= 1.0:
                raise ActionTargetError(f"TargetEntry[{self.name}].{label} 必须在 0..1 之间: {value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ActionTargetError(
                f"TargetEntry[{self.name}].confidence 必须在 0..1 之间: {self.confidence}"
            )

    def to_resolved(self) -> ResolvedTarget:
        """转换为解析结果。"""
        return ResolvedTarget(
            target=self.name,
            x_ratio=float(self.x_ratio),
            y_ratio=float(self.y_ratio),
            confidence=float(self.confidence),
            source=self.source,
            calibrated=self.calibrated,
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含像素坐标）。"""
        return {
            "name": self.name,
            "window_relative_position": {"x_ratio": self.x_ratio, "y_ratio": self.y_ratio},
            "confidence": self.confidence,
            "calibrated": self.calibrated,
            "description": self.description,
            "source": self.source,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        flag = "calibrated" if self.calibrated else "placeholder"
        return (
            f"TargetEntry({self.name}, x_ratio={self.x_ratio:.4f}, y_ratio={self.y_ratio:.4f}, "
            f"{flag})"
        )


#: 内置目标的**占位**比例（右下角自动战斗 / 弹窗确认等常见位置）。
#: ⚠️ 这些值**未经实机标定**（``calibrated=False``）：只用于 dry-run 演示与测试；
#: 真实输入前必须通过注册表显式声明标定值（``TargetRegistry.add(...)`` /
#: ``registry_from_mapping(...)``），或由后续的 vision 解析器提供（见
#: ``docs/PHASE7_ACTION_LAYER_REPORT.md`` 的"坐标策略"）。
BUILTIN_TARGETS: tuple[TargetEntry, ...] = (
    TargetEntry(
        name=ActionTarget.BATTLE_AUTO_BUTTON.value,
        x_ratio=0.92,
        y_ratio=0.95,
        description="战斗中右下角『自动』按钮（占位比例，未标定）",
    ),
    TargetEntry(
        name=ActionTarget.BATTLE_RESULT_CONFIRM.value,
        x_ratio=0.50,
        y_ratio=0.86,
        description="战斗结算界面确认按钮（占位比例，未标定）",
    ),
    TargetEntry(
        name=ActionTarget.REWARD_CONFIRM.value,
        x_ratio=0.50,
        y_ratio=0.82,
        description="奖励弹窗确认按钮（占位比例，未标定）",
    ),
    TargetEntry(
        name=ActionTarget.EXPLORATION_ENTRY.value,
        x_ratio=0.85,
        y_ratio=0.90,
        description="庭院右下角『探索』入口（占位比例，未标定）",
    ),
    TargetEntry(
        name=ActionTarget.YARD_ENTRY.value,
        x_ratio=0.06,
        y_ratio=0.06,
        description="左上角返回庭院按钮（占位比例，未标定）",
    ),
)


class TargetRegistry:
    """语义目标注册表（可变；线程安全由使用方保证：单线程事件循环）。

    Args:
        entries: 初始记录（同名后者覆盖前者）。
    """

    def __init__(self, entries: Iterable[TargetEntry] = ()) -> None:
        self._entries: dict[str, TargetEntry] = {}
        for entry in entries:
            self.register(entry)

    @classmethod
    def empty(cls) -> TargetRegistry:
        """空注册表（**生产安全默认**：一切目标都需显式注册）。"""
        return cls()

    @classmethod
    def default(cls) -> TargetRegistry:
        """内置注册表（含 :data:`BUILTIN_TARGETS` 的占位记录，便于 dry-run 演示与测试）。"""
        return cls(BUILTIN_TARGETS)

    # ---- 读写 ---------------------------------------------------------------

    def register(self, entry: TargetEntry) -> TargetEntry:
        """注册（或覆盖）一条目标。

        Args:
            entry: 目标记录。

        Returns:
            写入的记录（便于链式调用）。
        """
        if not isinstance(entry, TargetEntry):
            raise ActionTargetError("registry.register 需要 TargetEntry")
        self._entries[entry.name] = entry
        logger.debug("注册语义目标: %s", entry.describe())
        return entry

    def add(
        self,
        name: str,
        x_ratio: float,
        y_ratio: float,
        *,
        confidence: float = 1.0,
        calibrated: bool = True,
        description: str = "",
        source: str = "registry",
    ) -> TargetEntry:
        """便捷注册：按比例登记一个语义目标。

        Args:
            name: 语义目标名。
            x_ratio: 窗口客户区横向比例（0..1）。
            y_ratio: 窗口客户区纵向比例（0..1）。
            confidence: 声明置信度。
            calibrated: **默认 ``True``**（显式调用 ``add`` 视为有意声明，可被真实后端使用）。
            description: 说明。
            source: 来源标记。

        Returns:
            写入的记录。
        """
        return self.register(
            TargetEntry(
                name=name,
                x_ratio=x_ratio,
                y_ratio=y_ratio,
                confidence=confidence,
                calibrated=calibrated,
                description=description,
                source=source,
            )
        )

    def unregister(self, name: str) -> bool:
        """移除一条目标。

        Returns:
            ``True`` 表示确实移除了。
        """
        return self._entries.pop(name, None) is not None

    def clear(self) -> None:
        """清空注册表。"""
        self._entries.clear()

    # ---- 查询 ---------------------------------------------------------------

    def get(self, name: str) -> TargetEntry | None:
        """按名称取记录（不存在时返回 ``None``）。"""
        return self._entries.get(name)

    def resolve(self, name: str) -> ResolvedTarget | None:
        """按名称解析为 :class:`ResolvedTarget`（不存在时返回 ``None``）。"""
        entry = self._entries.get(name)
        return entry.to_resolved() if entry is not None else None

    def names(self) -> tuple[str, ...]:
        """已注册目标名（按注册顺序）。"""
        return tuple(self._entries)

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {name: entry.to_dict() for name, entry in self._entries.items()}

    def describe(self) -> str:
        """返回可读描述。"""
        return f"TargetRegistry(entries={len(self._entries)}, names={list(self._entries)})"

    # ---- 协议支持 -----------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[TargetEntry]:
        return iter(self._entries.values())

    def __repr__(self) -> str:
        return self.describe()


def registry_from_mapping(data: Mapping[str, Any]) -> TargetRegistry:
    """从配置映射构造注册表（``{name: {x_ratio, y_ratio, ...}}``）。

    Args:
        data: 形如 ``{"BATTLE_AUTO_BUTTON": {"x_ratio": 0.92, "y_ratio": 0.95, "calibrated": true}}``。

    Returns:
        注册表（空映射 → 空注册表）。

    Raises:
        ActionTargetError: 记录缺少比例或取值非法。
    """
    registry = TargetRegistry.empty()
    for name, payload in data.items():
        if not isinstance(payload, Mapping):
            raise ActionTargetError(f"注册表记录必须是映射: {name}")
        if "x_ratio" not in payload or "y_ratio" not in payload:
            raise ActionTargetError(f"注册表记录缺少 x_ratio / y_ratio: {name}")
        registry.register(
            TargetEntry(
                name=str(name),
                x_ratio=float(payload["x_ratio"]),
                y_ratio=float(payload["y_ratio"]),
                confidence=float(payload.get("confidence", 1.0)),
                calibrated=bool(payload.get("calibrated", False)),
                description=str(payload.get("description", "") or ""),
                source=str(payload.get("source", "registry")),
            )
        )
    return registry
