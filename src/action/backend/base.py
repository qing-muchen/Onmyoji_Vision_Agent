"""输入后端契约与默认实现（Phase 7）。

:class:`InputBackend` 是 Action 层与真实输入设备之间的**唯一**接口：

```text
ActionDispatcher ──▶ InputBackend.move/click/press/release/key_press/key_release
```

* 默认实现 :class:`DryRunBackend`：**记录调用，不产生任何真实输入**（安全默认）；
* 真实实现位于 :mod:`src.action.backend.windows`（Win32 ``SendInput``）；
* 后端接口只接受**屏幕坐标点**（:class:`~src.core.geometry.Point`）与按键名，
  坐标的计算属于 Dispatcher（由窗口客户区矩形 × 比例得出），后端不做定位与判断；
* 任何后端都必须提供 :attr:`InputBackend.is_dry_run`：安全门用它决定是否要求
  "目标已标定"（占位比例在真实后端下会被拒绝）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.core.geometry import Point, Rect
from src.utils.logger import get_logger

__all__ = [
    "MOUSE_BUTTONS",
    "BackendCall",
    "BackendStats",
    "DryRunBackend",
    "InputBackend",
    "NullWindowProbe",
    "WindowProbe",
]

logger = get_logger("action.backend")

#: 支持的鼠标按键名
MOUSE_BUTTONS: frozenset[str] = frozenset({"left", "right", "middle"})


@dataclass(frozen=True, slots=True)
class BackendCall:
    """一次后端调用记录（dry-run 与实际调用共用，便于审计与测试）。

    Attributes:
        operation: ``move`` / ``click`` / ``press`` / ``release`` / ``key_press`` / ``key_release``。
        point: 屏幕坐标（鼠标操作）。
        button: 鼠标按键名。
        key: 按键名（键盘操作）。
        count: 点击次数。
    """

    operation: str
    point: Point | None = None
    button: str = ""
    key: str = ""
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含截图 / 用户数据）。"""
        payload: dict[str, Any] = {"operation": self.operation}
        if self.point is not None:
            payload["point"] = list(self.point.to_int_tuple())
        if self.button:
            payload["button"] = self.button
        if self.key:
            payload["key"] = self.key
        if self.count != 1:
            payload["count"] = self.count
        return payload

    def describe(self) -> str:
        """返回可读描述。"""
        if self.point is not None:
            return f"{self.operation}({self.point.describe()}, button={self.button}, count={self.count})"
        if self.key:
            return f"{self.operation}(key={self.key})"
        return f"{self.operation}()"


@runtime_checkable
class InputBackend(Protocol):
    """输入后端协议（鼠标 + 键盘的最小原语集合）。"""

    @property
    def is_dry_run(self) -> bool:
        """是否为"只记录、不输入"的空后端。"""
        ...

    def move(self, point: Point, /) -> None:
        """把指针移动到屏幕坐标。"""
        ...

    def click(self, point: Point, /, *, button: str = "left", count: int = 1) -> None:
        """在屏幕坐标点击（``count`` 次）。"""
        ...

    def press(self, point: Point, /, *, button: str = "left") -> None:
        """在屏幕坐标按下（长按起点）。"""
        ...

    def release(self, point: Point, /, *, button: str = "left") -> None:
        """在屏幕坐标释放（长按终点）。"""
        ...

    def key_press(self, key: str, /) -> None:
        """按下按键。"""
        ...

    def key_release(self, key: str, /) -> None:
        """释放按键。"""
        ...


@runtime_checkable
class WindowProbe(Protocol):
    """目标窗口探测协议（安全门控的输入来源）。

    实现方负责提供"窗口是否有效 / 是否在前台 / 客户区矩形"三项**事实**；
    Action 层据此决定是否放行（不自行查找窗口、不抢焦点）。
    """

    def is_valid(self) -> bool:
        """窗口是否存在且可见（标题栏 / 最小化状态由实现判断）。"""
        ...

    def is_foreground(self) -> bool:
        """目标窗口是否为前台窗口。"""
        ...

    def client_rect(self) -> Rect | None:
        """客户区在**屏幕坐标**下的矩形（用于比例 → 屏幕坐标换算）。"""
        ...

    def describe(self) -> str:
        """返回可读描述。"""
        ...


class NullWindowProbe:
    """空窗口探测器：**永远报告窗口无效**（安全默认）。

    没有真实窗口信息时**不允许**任何动作——避免"窗口未知也照点"的 fallback 行为。
    """

    def is_valid(self) -> bool:
        """恒返回 ``False``（安全默认）。"""
        return False

    def is_foreground(self) -> bool:
        """恒返回 ``False``。"""
        return False

    def client_rect(self) -> Rect | None:
        """恒返回 ``None``。"""
        return None

    def describe(self) -> str:
        """返回可读描述。"""
        return "NullWindowProbe(always-invalid)"


class DryRunBackend:
    """默认后端：**记录调用，不产生任何真实输入**。

    用途：Phase 7 的默认后端（`action.enabled=false`）、测试替身与离线演示。
    记录 :class:`BackendCall` 供测试断言与审计（不含截图 / 用户数据）。
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._calls: list[BackendCall] = []
        self._clock = clock

    @property
    def is_dry_run(self) -> bool:
        """恒为 ``True``（空后端）。"""
        return True

    @property
    def calls(self) -> tuple[BackendCall, ...]:
        """已记录的调用（按顺序）。"""
        return tuple(self._calls)

    @property
    def count(self) -> int:
        """已记录的调用数量。"""
        return len(self._calls)

    def clear(self) -> None:
        """清空记录。"""
        self._calls.clear()

    def _record(self, operation: str, **kwargs: Any) -> BackendCall:
        """记录一次调用。"""
        call = BackendCall(operation=operation, **kwargs)
        self._calls.append(call)
        logger.debug("dry-run 后端记录: %s", call.describe())
        return call

    def move(self, point: Point, /) -> None:
        """记录移动（不产生输入）。"""
        self._record("move", point=point)

    def click(self, point: Point, /, *, button: str = "left", count: int = 1) -> None:
        """记录点击（不产生输入）。"""
        self._record("click", point=point, button=button, count=count)

    def press(self, point: Point, /, *, button: str = "left") -> None:
        """记录按下（不产生输入）。"""
        self._record("press", point=point, button=button)

    def release(self, point: Point, /, *, button: str = "left") -> None:
        """记录释放（不产生输入）。"""
        self._record("release", point=point, button=button)

    def key_press(self, key: str, /) -> None:
        """记录按键按下（不产生输入）。"""
        self._record("key_press", key=key)

    def key_release(self, key: str, /) -> None:
        """记录按键释放（不产生输入）。"""
        self._record("key_release", key=key)

    def describe(self) -> str:
        """返回可读描述。"""
        return f"DryRunBackend(calls={self.count})"

    def __repr__(self) -> str:
        return self.describe()


@dataclass(slots=True)
class BackendStats:
    """后端的轻量统计（可选观测）。"""

    moves: int = 0
    clicks: int = 0
    presses: int = 0
    releases: int = 0
    keys: int = 0
    errors: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "moves": self.moves,
            "clicks": self.clicks,
            "presses": self.presses,
            "releases": self.releases,
            "keys": self.keys,
            "errors": self.errors,
            **self.detail,
        }
