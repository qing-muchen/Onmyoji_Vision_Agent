"""观察存储（Phase 10）：``ObservationStore`` —— 最近 N 次观察的**短期记忆**。

```text
ObservationStore(max_size=1000)
├── append(observation)      写入（满则 FIFO 淘汰最旧；失败隔离，不影响 Runtime）
├── latest()                 最近一条
├── history(limit)           最近 N 条（时间顺序）
├── query(filter)            按条件筛选（filter 只需有 matches(observation)）
├── clear()                  清空
└── stats                    写入 / 淘汰 / 失败计数
```

**职责与边界**：

* 只保存 :class:`~src.runtime.observation.models.Observation`（状态描述），
  **不保存画面**（构造期已校验）、**不认识 Task 层**（纯 Runtime memory）；
* 容量语义：FIFO（超限淘汰最旧），默认 1000（"短期记忆"）；
* 失败策略：写入失败**不抛异常**（计数 + 日志），调用方（Runtime）不受影响；
* 线程模型：Runtime 单线程循环使用，本类不加锁。

依赖边界（AST 用例固化）：只 import 标准库 + ``src.core`` / ``src.utils`` + 本包 models。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from src.runtime.observation.models import Observation
from src.utils.logger import get_logger

__all__ = ["DEFAULT_OBSERVATION_MAX_SIZE", "ObservationStore", "ObservationStoreStats"]

logger = get_logger("runtime.observation.store")

#: 默认短期记忆容量（"最近 1000 次观察"）
DEFAULT_OBSERVATION_MAX_SIZE: int = 1000


@dataclass(slots=True)
class ObservationStoreStats:
    """观察存储统计（fail-safe 观测点）。"""

    appended: int = 0
    evicted: int = 0
    errors: int = 0
    cleared: int = 0
    last_reason: str = ""
    last_error: str = ""

    def reset(self) -> None:
        """复位为初始值。"""
        for name in self.__slots__:
            setattr(self, name, 0 if isinstance(getattr(self, name), int) else "")

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {name: getattr(self, name) for name in self.__slots__}

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"appended={self.appended} evicted={self.evicted} "
            f"cleared={self.cleared} errors={self.errors}"
        )


class ObservationStore:
    """最近 N 次观察的环形短期记忆（Phase 10）。

    Args:
        max_size: 容量上限（``>= 1``）；超出后 FIFO 淘汰最旧观察。
        clock: 单调时钟（保留给未来的"按时间窗口查询"；当前仅用于时间戳兜底）。

    Raises:
        ValueError: ``max_size < 1``。
    """

    def __init__(self, max_size: int = DEFAULT_OBSERVATION_MAX_SIZE, *, clock: Callable[[], float] | None = None) -> None:
        if int(max_size) < 1:
            raise ValueError(f"ObservationStore 容量必须 >= 1（当前 {max_size}）")
        self._max_size = int(max_size)
        self._items: deque[Observation] = deque(maxlen=self._max_size)
        self._stats = ObservationStoreStats()
        self._clock = clock

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def max_size(self) -> int:
        """容量上限。"""
        return self._max_size

    @property
    def stats(self) -> ObservationStoreStats:
        """统计。"""
        return self._stats

    @property
    def count(self) -> int:
        """当前观察条数。"""
        return len(self._items)

    @property
    def is_empty(self) -> bool:
        """是否为空。"""
        return not self._items

    @property
    def is_full(self) -> bool:
        """是否已达容量上限。"""
        return len(self._items) >= self._max_size

    def latest(self) -> Observation | None:
        """最近一次观察（空时 ``None``）。"""
        return self._items[-1] if self._items else None

    def oldest(self) -> Observation | None:
        """最早一次观察（空时 ``None``）。"""
        return self._items[0] if self._items else None

    def history(self, limit: int | None = None) -> tuple[Observation, ...]:
        """返回最近的观察序列（时间顺序，最早在前）。

        Args:
            limit: 只取最近 ``limit`` 条；``None`` 或 ``<= 0`` 表示全部（受容量限制）。

        Returns:
            观察元组。
        """
        items = list(self._items)
        if limit is not None and int(limit) > 0:
            items = items[-int(limit) :]
        return tuple(items)

    def states(self, limit: int | None = None) -> tuple[str, ...]:
        """返回最近若干次的状态名序列（便于快速观察轨迹）。"""
        return tuple(item.state_name for item in self.history(limit))

    def latest_state(self) -> str | None:
        """最近一次的状态名（空时 ``None``）。"""
        latest = self.latest()
        return None if latest is None else latest.state_name

    def query(self, criteria: Any = None) -> tuple[Observation, ...]:
        """按条件筛选观察（时间顺序）。

        Args:
            criteria: ``None``（全部）/ 可调用对象（``obs -> bool``）/
                具备 ``matches(observation)`` 方法的对象（例如
                :class:`~src.runtime.observation.query.ObservationQuery`）。

        Returns:
            命中的观察元组。

        Raises:
            TypeError: ``criteria`` 既不可调用也没有 ``matches``。
        """
        if criteria is None:
            return self.history()
        if hasattr(criteria, "apply") and not callable(criteria):
            # 具备 apply(observations) 的条件对象（例如 ObservationQuery）：
            # 由它统一处理 matches + limit（保证 limit 语义一致）。
            return tuple(criteria.apply(tuple(self._items)))
        matcher: Callable[[Observation], bool]
        if callable(criteria):
            matcher = criteria
        elif hasattr(criteria, "matches"):
            matcher = criteria.matches
        else:
            raise TypeError(
                "query 条件必须是可调用对象或具备 matches(observation) 的对象"
                f"（当前: {type(criteria).__name__}）"
            )
        return tuple(item for item in self._items if matcher(item))

    def get(self, observation_id: str) -> Observation | None:
        """按标识取观察（不存在时 ``None``）。"""
        key = str(observation_id)
        for item in self._items:
            if item.observation_id == key:
                return item
        return None

    # ---- 写入 ---------------------------------------------------------------

    def append(self, observation: Any) -> Observation | None:
        """写入一条观察（满则 FIFO 淘汰最旧）。

        **失败隔离**：任何异常（非法元数据 / 无法识别的值）都被捕获并计数，
        返回 ``None``，**不影响 Runtime 主循环**。

        Args:
            observation: :class:`~src.runtime.observation.models.Observation`
                或任何可按字段视图识别的状态结果。

        Returns:
            实际写入的观察；写入失败时 ``None``。
        """
        try:
            if isinstance(observation, Observation):
                item = observation
            elif isinstance(observation, Mapping):
                # 映射输入：元数据一并参与校验（与 Observation 构造规则一致）
                payload = dict(observation)
                metadata = payload.pop("metadata", None)
                item = Observation.from_state_result(payload, metadata=metadata)
            else:
                item = Observation.from_state_result(observation)
            if self.is_full and self._items:
                self._stats.evicted += 1
            self._items.append(item)
            self._stats.appended += 1
            self._stats.last_reason = f"append {item.describe()}"
            return item
        except Exception as exc:  # noqa: BLE001 - 写入失败绝不影响 Runtime
            self._stats.errors += 1
            self._stats.last_error = str(exc)
            logger.warning("观察写入失败（已隔离）: %s", exc)
            return None

    def extend(self, observations: Iterable[Any]) -> tuple[Observation, ...]:
        """批量写入（逐条走 :meth:`append` 的失败隔离）。

        Args:
            observations: 观察序列。

        Returns:
            实际写入的观察元组。
        """
        written: list[Observation] = []
        for item in observations:
            result = self.append(item)
            if result is not None:
                written.append(result)
        return tuple(written)

    def clear(self) -> int:
        """清空全部观察。

        Returns:
            被清空的条数。
        """
        count = len(self._items)
        self._items.clear()
        if count:
            self._stats.cleared += 1
        return count

    def reset_stats(self) -> None:
        """只复位统计（保留观察）。"""
        self._stats.reset()

    # ---- 描述 ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "max_size": self._max_size,
            "count": len(self._items),
            "stats": self._stats.to_dict(),
            "observations": [item.to_dict() for item in self._items],
        }

    def describe(self) -> str:
        """返回可读描述。"""
        latest = self.latest()
        return (
            f"ObservationStore(size={len(self._items)}/{self._max_size}, "
            f"latest={latest.describe() if latest else '-'}, {self._stats.describe()})"
        )

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Observation]:
        return iter(list(self._items))

    def __repr__(self) -> str:
        return self.describe()
