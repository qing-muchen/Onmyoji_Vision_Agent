"""状态 → 任务触发映射（Phase 7.5）。

Runtime 的稳定状态可以触发一个任务（例如"进入战斗状态 → 启动自动战斗任务"）。
映射关系由**配置/调用方注册**，本模块**不做任何游戏逻辑判断**：

```python
triggers = TaskTriggerRegistry()
triggers.register("battle", "auto_battle")     # 状态名 -> 任务 id
triggers.resolve("battle")                     # -> "auto_battle"
```

* 状态名匹配**大小写不敏感**，且接受 :class:`~src.vision.state.models.StateName` 枚举
  （通过 ``.value`` 归一化为字符串，不 import vision）；
* 未注册的状态返回 ``None``（由编排层决定"不触发"，绝不猜测任务）；
* 同一状态重复注册默认报错（避免隐式覆盖），可用 ``replace=True`` 显式覆盖。

依赖边界（架构决策 D-4）：只允许标准库 + ``src.core`` + ``src.utils`` + ``src.task``。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from src.task.models import TaskDefinitionError
from src.utils.logger import get_logger

__all__ = ["TaskTriggerRegistry", "trigger_registry_from_mapping"]

logger = get_logger("task.triggers")


def _state_key(value: Any) -> str:
    """把状态名（枚举 / 字符串）归一化为小写字符串。"""
    inner = getattr(value, "value", value)
    return str(inner).strip().casefold()


class TaskTriggerRegistry:
    """状态名 → 任务标识的映射表（**纯声明**，不含判定逻辑）。

    Args:
        mapping: 初始映射（``{state_name: task_id}``）。
    """

    def __init__(self, mapping: Mapping[str, str] | None = None) -> None:
        self._mapping: dict[str, str] = {}
        for state_name, task_id in (mapping or {}).items():
            self.register(state_name, task_id, replace=True)

    # ---- 注册 / 注销 --------------------------------------------------------

    def register(self, state_name: Any, task_id: str, *, replace: bool = False) -> str:
        """注册一条触发映射。

        Args:
            state_name: 状态名（字符串或枚举）。
            task_id: 目标任务标识。
            replace: 已存在时是否覆盖（默认 ``False`` = 报错）。

        Returns:
            写入的状态键（归一化后的小写字符串）。

        Raises:
            TaskDefinitionError: 状态名为空、任务标识为空，或已存在且未允许覆盖。
        """
        key = _state_key(state_name)
        target = str(task_id).strip()
        if not key:
            raise TaskDefinitionError("触发映射的状态名不能为空")
        if not target:
            raise TaskDefinitionError(f"触发映射 [{key}] 的任务标识不能为空")
        if key in self._mapping and not replace:
            raise TaskDefinitionError(
                f"状态 {key!r} 已绑定任务 {self._mapping[key]!r}"
                f"（如需覆盖请传 replace=True）"
            )
        self._mapping[key] = target
        logger.debug("注册触发映射: %s -> %s", key, target)
        return key

    def unregister(self, state_name: Any) -> bool:
        """移除一条映射。

        Returns:
            ``True`` 表示确实移除了。
        """
        return self._mapping.pop(_state_key(state_name), None) is not None

    def clear(self) -> None:
        """清空映射。"""
        self._mapping.clear()

    # ---- 查询 ---------------------------------------------------------------

    def resolve(self, state_name: Any) -> str | None:
        """解析状态对应的任务标识（未注册时返回 ``None``）。

        Args:
            state_name: 状态名（字符串或枚举）。

        Returns:
            任务标识；未注册时 ``None``（调用方据此"不触发"，绝不猜测）。
        """
        return self._mapping.get(_state_key(state_name))

    def states(self) -> tuple[str, ...]:
        """已注册状态（按注册顺序）。"""
        return tuple(self._mapping)

    def task_ids(self) -> tuple[str, ...]:
        """已引用的任务标识（去重，按注册顺序）。"""
        seen: dict[str, None] = {}
        for task_id in self._mapping.values():
            seen.setdefault(task_id, None)
        return tuple(seen)

    def items(self) -> tuple[tuple[str, str], ...]:
        """全部映射项。"""
        return tuple(self._mapping.items())

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return dict(self._mapping)

    def describe(self) -> str:
        """返回可读描述。"""
        return f"TaskTriggerRegistry(entries={len(self._mapping)}, states={list(self._mapping)})"

    # ---- 协议支持 -----------------------------------------------------------

    def __contains__(self, state_name: object) -> bool:
        return _state_key(state_name) in self._mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._mapping.items())

    def __repr__(self) -> str:
        return self.describe()


def trigger_registry_from_mapping(data: Mapping[str, Any]) -> TaskTriggerRegistry:
    """从配置映射构造触发表（``{state_name: task_id}``）。

    Args:
        data: 映射。

    Returns:
        触发表。

    Raises:
        TaskDefinitionError: 任一映射非法。
    """
    return TaskTriggerRegistry({str(key): str(value) for key, value in data.items()})
