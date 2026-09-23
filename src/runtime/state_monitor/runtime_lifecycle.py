"""运行时生命周期状态机（Phase 5.4）。

把"Runtime 现在处于什么阶段"变成**显式、可校验**的状态，避免"监控器到底有没有在跑 /
是不是正在恢复"只能靠猜：

```text
INITIALIZING ──initialize──▶ READY ──start──▶ RUNNING ──pause──▶ PAUSED
      ▲                        │                │  ▲                  │
      │                        │                │  └──resume──────────┘
      │                        │             capture failure
      │                        │                ▼
      │                        │            RECOVERING ──recovered──▶ RUNNING
      │                        │                │
      │                        │                └──retry exhausted──▶ FAILED
      │                        │                                       │
      └──re-initialize─────────┴───────────────stop────────────▶ STOPPED
```

设计要点：

* **转换表驱动**：:data:`ALLOWED_TRANSITIONS` 是唯一事实来源，非法转换抛
  :class:`InvalidRuntimeTransition`（不会静默改变状态）；
* **语义化入口**：:meth:`~RuntimeLifecycle.initialize` / ``start`` / ``pause`` / ``resume`` /
  ``begin_recovery`` / ``finish_recovery`` / ``stop`` / ``fail``，调用方无需关心跳转细节；
* **幂等**：重复调用同一入口不产生非法转换（例如已 ``RUNNING`` 时再 ``start()`` 直接返回）；
* **可重启**：``STOPPED`` / ``FAILED`` 可以通过 :meth:`~RuntimeLifecycle.initialize` 重新进入
  ``INITIALIZING → READY``，因此既有 ``StateMonitor.run()`` 在 ``stop()`` 之后仍可再次运行；
* **只描述状态**：不采集、不判定、不做恢复动作（恢复策略见
  :mod:`src.runtime.state_monitor.recovery`）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from src.core.exceptions import OnmyojiVisionError
from src.utils.logger import get_logger

from src.runtime.state_monitor.report import local_now_iso

__all__ = [
    "ALLOWED_TRANSITIONS",
    "InvalidRuntimeTransition",
    "LifecycleTransition",
    "RuntimeLifecycle",
    "RuntimeState",
]

logger = get_logger("runtime.state_monitor.runtime_lifecycle")


class RuntimeState(StrEnum):
    """运行时的生命周期状态。"""

    #: 已构造、尚未初始化
    INITIALIZING = "initializing"
    #: 初始化完成，可以开始运行
    READY = "ready"
    #: 正在运行（采集 / 判定 / 稳定化 / 事件）
    RUNNING = "running"
    #: 已暂停：保持节拍但**不处理新帧**
    PAUSED = "paused"
    #: 出现可恢复故障（采集失败 / 窗口丢失），正在按策略重试
    RECOVERING = "recovering"
    #: 已停止（资源已释放）；可通过 ``initialize()`` 重新进入
    STOPPED = "stopped"
    #: 重试用尽（无法恢复），需要人工处理
    FAILED = "failed"


#: 合法状态转换表（唯一事实来源）
ALLOWED_TRANSITIONS: Mapping[RuntimeState, frozenset[RuntimeState]] = {
    RuntimeState.INITIALIZING: frozenset(
        {RuntimeState.READY, RuntimeState.STOPPED, RuntimeState.FAILED}
    ),
    RuntimeState.READY: frozenset(
        {RuntimeState.RUNNING, RuntimeState.STOPPED, RuntimeState.FAILED}
    ),
    RuntimeState.RUNNING: frozenset(
        {
            RuntimeState.READY,
            RuntimeState.PAUSED,
            RuntimeState.RECOVERING,
            RuntimeState.STOPPED,
            RuntimeState.FAILED,
        }
    ),
    RuntimeState.PAUSED: frozenset(
        {RuntimeState.RUNNING, RuntimeState.STOPPED, RuntimeState.FAILED}
    ),
    RuntimeState.RECOVERING: frozenset(
        {RuntimeState.READY, RuntimeState.RUNNING, RuntimeState.STOPPED, RuntimeState.FAILED}
    ),
    RuntimeState.STOPPED: frozenset({RuntimeState.INITIALIZING}),
    RuntimeState.FAILED: frozenset({RuntimeState.INITIALIZING}),
}

#: 终止态（不会再自行推进）
TERMINAL_STATES: frozenset[RuntimeState] = frozenset({RuntimeState.STOPPED, RuntimeState.FAILED})


class InvalidRuntimeTransition(OnmyojiVisionError):
    """非法的生命周期状态转换。"""


@dataclass(frozen=True, slots=True)
class LifecycleTransition:
    """一次状态转换记录。

    Attributes:
        previous: 转换前的状态。
        current: 转换后的状态。
        reason: 触发原因（人类可读）。
        timestamp: 转换时刻（ISO 8601 本地时间）。
    """

    previous: RuntimeState
    current: RuntimeState
    reason: str = ""
    timestamp: str = ""

    def describe(self) -> str:
        """返回可读描述。"""
        suffix = f"（{self.reason}）" if self.reason else ""
        return f"{self.previous.value} -> {self.current.value}{suffix}"


class RuntimeLifecycle:
    """运行时生命周期状态机（Phase 5.4）。

    典型用法::

        lifecycle = RuntimeLifecycle()
        lifecycle.initialize()          # INITIALIZING -> READY
        lifecycle.start()               # READY -> RUNNING
        lifecycle.begin_recovery("capture_failed")   # RUNNING -> RECOVERING
        lifecycle.finish_recovery()     # RECOVERING -> RUNNING
        lifecycle.stop()                # RUNNING -> STOPPED

    Args:
        now: 时间戳函数（默认本地 ISO 8601；可在测试中注入）。
    """

    def __init__(self, *, now: Callable[[], str] = local_now_iso) -> None:
        self._now = now
        self._state = RuntimeState.INITIALIZING
        self._transitions: list[LifecycleTransition] = []

    # ---- 查询 ---------------------------------------------------------------

    @property
    def state(self) -> RuntimeState:
        """当前状态。"""
        return self._state

    @property
    def transitions(self) -> tuple[LifecycleTransition, ...]:
        """状态转换历史（按发生顺序）。"""
        return tuple(self._transitions)

    @property
    def transition_count(self) -> int:
        """状态转换次数。"""
        return len(self._transitions)

    @property
    def last_transition(self) -> LifecycleTransition | None:
        """最近一次转换；尚未发生转换时为 ``None``。"""
        return self._transitions[-1] if self._transitions else None

    @property
    def is_terminal(self) -> bool:
        """是否处于终止态（``STOPPED`` / ``FAILED``）。"""
        return self._state in TERMINAL_STATES

    @property
    def is_running(self) -> bool:
        """是否处于 ``RUNNING``。"""
        return self._state is RuntimeState.RUNNING

    @property
    def is_paused(self) -> bool:
        """是否处于 ``PAUSED``。"""
        return self._state is RuntimeState.PAUSED

    @property
    def is_recovering(self) -> bool:
        """是否处于 ``RECOVERING``。"""
        return self._state is RuntimeState.RECOVERING

    def can_transition_to(self, target: RuntimeState) -> bool:
        """判断能否转换到目标状态。

        Args:
            target: 目标状态。

        Returns:
            ``True`` 表示该转换合法（含"目标即当前状态"的幂等情形）。
        """
        if target is self._state:
            return True
        return target in ALLOWED_TRANSITIONS[self._state]

    # ---- 转换 ---------------------------------------------------------------

    def transition(self, target: RuntimeState, *, reason: str = "") -> LifecycleTransition:
        """执行一次状态转换（非法转换抛异常，不静默忽略）。

        Args:
            target: 目标状态。
            reason: 触发原因（人类可读，写入转换历史）。

        Returns:
            本次转换记录。

        Raises:
            InvalidRuntimeTransition: 当前状态不允许转换到 ``target``。
        """
        if target is self._state:
            # 幂等：返回"自转换"记录但不写入历史，避免历史被重复状态灌满
            return LifecycleTransition(
                previous=self._state, current=self._state, reason=reason, timestamp=self._now()
            )
        allowed = ALLOWED_TRANSITIONS[self._state]
        if target not in allowed:
            raise InvalidRuntimeTransition(
                f"非法状态转换: {self._state.value} -> {target.value}"
                f"（允许: {sorted(item.value for item in allowed)}）"
            )
        record = LifecycleTransition(
            previous=self._state, current=target, reason=reason, timestamp=self._now()
        )
        self._state = target
        self._transitions.append(record)
        logger.info("运行时状态: %s", record.describe())
        return record

    # ---- 语义化入口 ---------------------------------------------------------

    def initialize(self, *, reason: str = "initialize") -> RuntimeState:
        """初始化：``INITIALIZING`` / ``STOPPED`` / ``FAILED`` → ``READY``。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。
        """
        if self._state is RuntimeState.READY:
            return self._state
        if self._state in TERMINAL_STATES:
            self.transition(RuntimeState.INITIALIZING, reason="re-initialize")
        if self._state is RuntimeState.INITIALIZING:
            self.transition(RuntimeState.READY, reason=reason)
        return self._state

    def start(self, *, reason: str = "start") -> RuntimeState:
        """开始运行：``READY`` / ``PAUSED`` / ``RECOVERING`` → ``RUNNING``。

        未初始化（``INITIALIZING``）或已终止（``STOPPED`` / ``FAILED``）时会先
        :meth:`initialize`，因此"停机后再次启动"始终可用。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。
        """
        if self._state is RuntimeState.RUNNING:
            return self._state
        if self._state is not RuntimeState.READY and self._state not in (
            RuntimeState.PAUSED,
            RuntimeState.RECOVERING,
        ):
            self.initialize(reason="auto-initialize")
        return self.transition(RuntimeState.RUNNING, reason=reason).current

    def pause(self, *, reason: str = "pause") -> RuntimeState:
        """暂停：``RUNNING`` → ``PAUSED``（停止处理新帧，节拍继续）。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。

        Raises:
            InvalidRuntimeTransition: 当前不是 ``RUNNING``。
        """
        return self.transition(RuntimeState.PAUSED, reason=reason).current

    def resume(self, *, reason: str = "resume") -> RuntimeState:
        """恢复：``PAUSED`` → ``RUNNING``。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。
        """
        if self._state is RuntimeState.RUNNING:
            return self._state
        return self.transition(RuntimeState.RUNNING, reason=reason).current

    def begin_recovery(self, reason: str) -> RuntimeState:
        """进入恢复：``RUNNING`` → ``RECOVERING``。

        Args:
            reason: 失败原因（写入转换历史）。

        Returns:
            转换后的状态。
        """
        if self._state is RuntimeState.RECOVERING:
            return self._state
        return self.transition(RuntimeState.RECOVERING, reason=reason).current

    def finish_recovery(self, *, reason: str = "recovered") -> RuntimeState:
        """恢复成功：``RECOVERING`` → ``RUNNING``。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。
        """
        if self._state is not RuntimeState.RECOVERING:
            return self._state
        return self.transition(RuntimeState.RUNNING, reason=reason).current

    def stop(self, *, reason: str = "stop") -> RuntimeState:
        """停止：任何非终止状态 → ``STOPPED``（已停止时幂等）。

        Args:
            reason: 转换原因。

        Returns:
            转换后的状态。
        """
        if self._state is RuntimeState.STOPPED:
            return self._state
        if self._state is RuntimeState.FAILED:
            # FAILED 是终止态：显式"停机"先回到 INITIALIZING 再 STOPPED
            self.transition(RuntimeState.INITIALIZING, reason="re-initialize")
        return self.transition(RuntimeState.STOPPED, reason=reason).current

    def fail(self, *, reason: str = "unrecoverable") -> RuntimeState:
        """进入失败态：``RUNNING`` / ``RECOVERING`` → ``FAILED``。

        Args:
            reason: 失败原因。

        Returns:
            转换后的状态。
        """
        if self._state is RuntimeState.FAILED:
            return self._state
        return self.transition(RuntimeState.FAILED, reason=reason).current

    def describe(self) -> str:
        """返回可读描述。"""
        last = self.last_transition
        suffix = f"，最近转换: {last.describe()}" if last is not None else ""
        return f"RuntimeLifecycle(state={self._state.value}, transitions={self.transition_count}{suffix})"
