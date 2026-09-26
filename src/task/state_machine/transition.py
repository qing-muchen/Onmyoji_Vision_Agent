"""转移模型（Phase 8）：``StateTransition`` —— 条件驱动与事件驱动的状态转移。

```text
source ──▶ target        转移的起点与终点（``target == "COMPLETE"`` 表示流程成功结束）
├── condition            条件驱动：谓词为真时转移（例如 state == "battle_ready"）
└── event                事件驱动：``timeout`` / ``action_failed``（由运行期产生）
```

* **条件驱动**（``event == ""``）：由 :meth:`TaskFlowStateMachine.handle_state` 评估——
  输入是 :class:`~src.vision.state.models.StateResult` 的**字段视图**（Task 层不 import vision）；
* **事件驱动**（``event != ""``）：由运行期事件触发（超时 / 动作失败），
  优先级高于条件驱动（同一状态同时命中时，事件优先）；
* 条件在**加载期**编译为谓词（复用 Phase 6.2 的 :func:`~src.task.hooks.build_condition`），
  **禁止** ``eval`` / 动态执行字符串；
* 未命中的转移不产生任何副作用（调用方保持当前状态）。

依赖边界（AST 用例固化）：只允许标准库 + ``src.core`` + ``src.utils`` + ``src.task``。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.task.models import TERMINAL_STATE_NAME, TaskDefinitionError

__all__ = [
    "EVENT_ACTION_FAILED",
    "EVENT_NONE",
    "EVENT_TIMEOUT",
    "TRANSITION_FIELDS",
    "StateTransition",
]

#: 无事件（纯条件驱动）
EVENT_NONE: str = ""
#: 状态等待窗口超时
EVENT_TIMEOUT: str = "timeout"
#: 状态内动作被拒 / 失败
EVENT_ACTION_FAILED: str = "action_failed"

#: 允许的事件名
KNOWN_EVENTS: tuple[str, ...] = (EVENT_NONE, EVENT_TIMEOUT, EVENT_ACTION_FAILED)

#: 转移允许出现的字段
TRANSITION_FIELDS: frozenset[str] = frozenset(
    {"source", "target", "from", "to", "condition", "when", "event", "name"}
)


@dataclass(frozen=True, slots=True)
class StateTransition:
    """一条状态转移（Phase 8）。

    Attributes:
        source: 来源状态名。
        target: 目标状态名，或保留字 :data:`~src.task.models.TERMINAL_STATE_NAME`（成功结束）。
        condition: **编译后**的谓词（条件驱动；``None`` 表示无条件，仅事件驱动时使用）。
        condition_spec: 条件声明原文（审计）。
        event: 事件名（``""`` = 条件驱动；``timeout`` / ``action_failed`` = 事件驱动）。
        name: 转移标识（缺省 ``<source>-><target>``）。

    Raises:
        TaskDefinitionError: 缺少来源 / 目标、事件名未知，或事件驱动却没有事件。
    """

    source: str
    target: str
    condition: Callable[[Any, Any], bool] | None = None
    condition_spec: Any = None
    event: str = EVENT_NONE
    name: str = ""

    def __post_init__(self) -> None:
        source = str(self.source).strip()
        target = str(self.target).strip()
        if not source:
            raise TaskDefinitionError("转移缺少 source（来源状态）")
        if not target:
            raise TaskDefinitionError("转移缺少 target（目标状态）")
        event = str(self.event or "").strip().lower()
        if event not in KNOWN_EVENTS:
            raise TaskDefinitionError(
                f"未知事件名: {self.event!r}（可选: {', '.join(KNOWN_EVENTS)}）"
            )
        if event == EVENT_NONE and self.condition is None:
            raise TaskDefinitionError(
                f"转移 {source} -> {target} 既没有条件也没有事件（至少需要一个）"
            )
        if self.condition is not None and not callable(self.condition):
            raise TaskDefinitionError("转移条件必须是已编译的谓词（禁止字符串动态执行）")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "name", str(self.name or "") or f"{source}->{target}")

    @property
    def is_terminal(self) -> bool:
        """是否指向流程成功结束。"""
        return self.target == TERMINAL_STATE_NAME

    @property
    def is_event_driven(self) -> bool:
        """是否事件驱动（``timeout`` / ``action_failed``）。"""
        return self.event != EVENT_NONE

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        condition: Callable[[Any, Any], bool] | None = None,
        where: str = "transition",
    ) -> StateTransition:
        """从映射构造。

        支持 ``from`` / ``to`` 与 ``source`` / ``target`` 两套键名，
        条件可写在 ``condition`` 或 ``when``：

        ```yaml
        transitions:
          - from: prepare
            to: running
            when: {state: battle_running}      # 或 condition: state == battle_running
          - from: running
            to: recovery
            event: timeout
        ```

        Args:
            data: 转移定义。
            condition: 已编译的谓词（由加载器用 ``build_condition`` 编译后传入）。
            where: 出错提示使用的位置描述。

        Returns:
            转移定义。

        Raises:
            TaskDefinitionError: 字段非法 / 未知。
        """
        unknown = sorted(set(map(str, data)) - TRANSITION_FIELDS)
        if unknown:
            raise TaskDefinitionError(
                f"{where} 出现未知字段: {unknown}（可选: {sorted(TRANSITION_FIELDS)}）"
            )
        source = data.get("source", data.get("from"))
        target = data.get("target", data.get("to"))
        if source is None:
            raise TaskDefinitionError(f"{where} 缺少 from/source")
        if target is None:
            raise TaskDefinitionError(f"{where} 缺少 to/target")
        spec = data.get("condition", data.get("when"))
        return cls(
            source=str(source),
            target=str(target),
            condition=condition,
            condition_spec=spec,
            event=str(data.get("event", "") or ""),
            name=str(data.get("name", "") or ""),
        )

    def matches(
        self, *, state_name: str, context: Any = None, snapshot: Any = None
    ) -> bool:
        """判断本条转移是否命中（条件驱动）。

        Args:
            state_name: 当前状态名（必须与 :attr:`source` 相同）。
            context: 条件上下文（任务上下文视图）。
            snapshot: 状态视图。

        Returns:
            ``True`` 表示命中。
        """
        if self.is_event_driven or state_name != self.source or self.condition is None:
            return False
        return bool(self.condition(context, snapshot))

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        condition = self.condition_spec
        return {
            "source": self.source,
            "target": self.target,
            "condition": dict(condition) if isinstance(condition, Mapping) else condition,
            "event": self.event,
            "name": self.name,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        if self.is_event_driven:
            return f"{self.source} --[{self.event}]--> {self.target}"
        condition = self.condition_spec
        shown = dict(condition) if isinstance(condition, Mapping) else condition
        return f"{self.source} --{shown}--> {self.target}"
