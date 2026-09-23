"""Action 端口（Phase 7）：Task 与 Action 之间的**字段级**契约。

Action 层**不允许** import ``src.task``（依赖边界），Task 层也不 import Action 层；
两层通过**字段视图（duck typing）**对接：

```text
Task 侧（Phase 6.2）                     Action 侧（Phase 7）
─────────────────                        ───────────────────
ActionRequest                            ActionRequest
  action_id   ─────────────────────────▶ request_id
  kind        ─────────────────────────▶ action_type
  target {element|name|x_ratio,y_ratio} ▶ target: SemanticTarget
  params      ─────────────────────────▶ parameters
  timeout_s / attempt / task_id / step_id 同名字段直传

ActionOutcome.is_failure  ◀──────────── ActionOutcome.status ∈ {REJECTED, FAILED}
```

因此 :class:`~src.action.dispatcher.ActionDispatcher` **本身就是** Task 层的 ``ActionPort``
实现（``request(request) -> outcome``），不需要任何适配器，也不需要修改 Task 层。

本模块只定义协议（:class:`ActionPort`）与最小自检工具：便于静态检查与文档化。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.action.models import ActionOutcome, ActionRequest
from src.utils.logger import get_logger

__all__ = ["ActionPort", "NullActionPort", "as_action_port", "is_action_request"]

logger = get_logger("action.port")


@runtime_checkable
class ActionPort(Protocol):
    """动作端口协议（与 Task 层 ``ActionPort`` 同名同形）。

    实现方负责：接收**语义**动作请求，返回处理结果；不负责定位判断（那是 Resolver 的职责），
    也不负责安全策略（那是 Guard 的职责）。
    """

    def request(self, request: Any, /) -> ActionOutcome:
        """提交一次语义动作请求。

        Args:
            request: :class:`ActionRequest`，或与其字段兼容的 Task 侧请求对象。

        Returns:
            处理结果。
        """
        ...


class NullActionPort:
    """空端口：接收请求、记录请求，**不产生任何输入**（默认安全实现）。

    与 Task 层 ``NullActionPort`` 语义一致：用于"只跑决策链路、不落地输入"的场景。
    """

    def __init__(self) -> None:
        self._requests: list[Any] = []

    @property
    def requests(self) -> tuple[Any, ...]:
        """已记录的请求（按提交顺序）。"""
        return tuple(self._requests)

    @property
    def count(self) -> int:
        """已记录的请求数量。"""
        return len(self._requests)

    def request(self, request: Any, /) -> ActionOutcome:
        """记录请求并返回"未执行"。

        Args:
            request: 语义动作请求。

        Returns:
            ``SKIPPED`` 结果。
        """
        from src.action.models import ActionStatus

        self._requests.append(request)
        logger.debug("空端口收到动作请求（未执行）: %r", request)
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="null action port")


@runtime_checkable
class _HasRequest(Protocol):
    """具备 ``request`` 方法的对象（用于端口自检）。"""

    def request(self, request: Any, /) -> ActionOutcome: ...


def as_action_port(candidate: Any) -> ActionPort:
    """校验对象是否满足 :class:`ActionPort` 协议。

    Args:
        candidate: 待检查对象。

    Returns:
        同一个对象（便于链式使用）。

    Raises:
        TypeError: 不满足协议。
    """
    if not isinstance(candidate, _HasRequest):
        raise TypeError(f"{type(candidate).__name__} 不满足 ActionPort 协议（缺少 request()）")
    return candidate  # type: ignore[return-value]


def is_action_request(value: Any) -> bool:
    """判断对象是否为 Action 层请求（或字段兼容的 Task 侧请求）。

    Args:
        value: 待检查对象。

    Returns:
        ``True`` 表示可作为请求提交给 :meth:`ActionPort.request`。
    """
    if isinstance(value, ActionRequest):
        return True
    for name in ("action_id", "request_id", "kind", "action_type"):
        if hasattr(value, name) or (isinstance(value, dict) and name in value):
            return True
    return False
