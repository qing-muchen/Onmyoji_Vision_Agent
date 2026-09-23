"""周期调度：deadline based 节拍（Phase 5.2）。

为什么不用 ``sleep(interval)``：单轮处理耗时会被**累加**进周期，节拍逐轮漂移
（500ms 周期 + 185ms 处理 → 实际约 685ms/轮）。本模块按"上一个 deadline + interval"
推进等待，处理耗时被吸收进周期内部，只有单轮耗时超过 interval 时才是异常：

```text
cycle N:  begin() ── 处理（frame → judge → stabilize）── finish()
                                                            │
                                    deadline += interval    │
                                    remaining = deadline - now
                                    remaining > 0  → sleep(remaining)          # 正常：吸收耗时
                                    remaining <= 0 → 不 sleep + deadline = now # late：重对齐并记录
```

契约（与 Phase 3.1 :class:`~src.runtime.state_monitor.monitor.StateMonitor` 既有节拍一致，
因此替换实现不改变可观测行为）：

* **不漂移**：deadline 累加而非"从当前时刻重新计时"，n 轮后总时间约为 ``n × interval``；
* **late cycle 记录**：``elapsed_ms > interval_ms`` 的轮次记为 ``late``，
  记录 ``overrun_ms``（超出量）与累计统计（``late_cycles`` / ``overrun_total_ms`` / ``overrun_max_ms``）；
* **重对齐**：late 时**不等待**并把握节拍基准重置到当前时刻，避免"追赶风暴"
  （否则连续慢轮会退化成无等待的忙循环）——重对齐次数记入 ``re_alignments``；
* **不吞异常**：``sleep`` 抛出的异常向上传播（调度失败不是可忽略的观测问题）。

本模块只做节拍计算，不取帧、不判定状态、不做任何决策。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from src.core.config import MILLISECONDS_PER_SECOND, RuntimeStateMonitorConfig
from src.utils.logger import get_logger

__all__ = ["CycleSchedule", "CycleScheduler"]

logger = get_logger("runtime.state_monitor.scheduler")


@dataclass(frozen=True, slots=True)
class CycleSchedule:
    """单轮的节拍结果。

    Attributes:
        cycle_id: 轮次序号（从 1 开始）。
        started_at: 本轮开始时刻（注入时钟的秒数）。
        ended_at: 本轮处理结束时刻（未含等待）。
        elapsed_ms: 本轮处理耗时（``ended_at - started_at``）。
        interval_ms: 目标周期。
        sleep_ms: 本轮**请求**的等待时长（late 时为 ``0.0``）。
        overrun_ms: 超时量 ``max(0, elapsed_ms - interval_ms)``。
        late: 本轮处理耗时是否超过周期（``overrun_ms > 0``）。
        re_aligned: 是否因超时而重对齐节拍基准（此时 ``sleep_ms == 0``）。
    """

    cycle_id: int
    started_at: float
    ended_at: float
    elapsed_ms: float
    interval_ms: float
    sleep_ms: float
    overrun_ms: float
    late: bool
    re_aligned: bool

    @property
    def occupancy(self) -> float:
        """周期占用率（``elapsed_ms / interval_ms``，1.0 表示刚好用满）。"""
        return self.elapsed_ms / self.interval_ms if self.interval_ms > 0 else 0.0

    def describe(self) -> str:
        """返回可读描述。"""
        state = "late" if self.late else "ok"
        overrun = f" overrun={self.overrun_ms:.1f}ms" if self.late else ""
        re_aligned = " re-aligned" if self.re_aligned else ""
        return (
            f"cycle={self.cycle_id} elapsed={self.elapsed_ms:.1f}ms "
            f"sleep={self.sleep_ms:.1f}ms interval={self.interval_ms:.1f}ms "
            f"[{state}]{overrun}{re_aligned}"
        )


class CycleScheduler:
    """按固定周期推进的 deadline based 调度器。

    典型用法::

        scheduler = CycleScheduler(500, clock=time.monotonic, sleep=time.sleep)
        while running:
            started = scheduler.begin()
            do_work()
            schedule = scheduler.finish(started)   # 内部完成等待与统计
            if schedule.late:
                logger.warning("late cycle: %s", schedule.describe())

    Args:
        interval_ms: 目标周期（毫秒），必须为正数。
        clock: 单调时钟函数（秒）；可在测试中注入。
        sleep: 等待函数（秒）；可在测试中注入。

    Raises:
        ValueError: ``interval_ms`` 不是正数。
    """

    def __init__(
        self,
        interval_ms: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError(f"周期必须为正数: {interval_ms}")
        self._interval_ms = float(interval_ms)
        self._clock = clock
        self._sleep = sleep
        self._deadline: float = clock()
        self._cycles = 0
        self._late_cycles = 0
        self._overrun_total_ms = 0.0
        self._overrun_max_ms = 0.0
        self._re_alignments = 0
        self._last: CycleSchedule | None = None

    @classmethod
    def from_config(
        cls,
        config: RuntimeStateMonitorConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> CycleScheduler:
        """按监控配置构造调度器（周期取自 ``runtime.state_monitor.interval_ms``）。

        Args:
            config: 状态监控配置。
            clock: 单调时钟函数。
            sleep: 等待函数。

        Returns:
            调度器实例。
        """
        return cls(config.interval_ms, clock=clock, sleep=sleep)

    @property
    def interval_ms(self) -> float:
        """目标周期（毫秒）。"""
        return self._interval_ms

    @property
    def interval_seconds(self) -> float:
        """目标周期（秒）。"""
        return self._interval_ms / MILLISECONDS_PER_SECOND

    @property
    def cycles(self) -> int:
        """已完成的轮次数。"""
        return self._cycles

    @property
    def late_cycles(self) -> int:
        """处理耗时超过周期的轮次数。"""
        return self._late_cycles

    @property
    def overrun_total_ms(self) -> float:
        """累计超时量（毫秒）。"""
        return self._overrun_total_ms

    @property
    def overrun_max_ms(self) -> float:
        """单轮最大超时量（毫秒）。"""
        return self._overrun_max_ms

    @property
    def re_alignments(self) -> int:
        """因超时而重对齐节拍基准的次数。"""
        return self._re_alignments

    @property
    def last(self) -> CycleSchedule | None:
        """最近一轮的节拍结果。"""
        return self._last

    def reset(self, *, start_at: float | None = None) -> None:
        """重置节拍基准与统计（用于开始一次新的运行）。

        Args:
            start_at: 本轮运行的起始时刻；省略时取当前时钟值。
        """
        self._deadline = self._clock() if start_at is None else start_at
        self._cycles = 0
        self._late_cycles = 0
        self._overrun_total_ms = 0.0
        self._overrun_max_ms = 0.0
        self._re_alignments = 0
        self._last = None

    def begin(self) -> float:
        """标记一轮开始。

        Returns:
            本轮开始时刻（注入时钟的秒数）。
        """
        return self._clock()

    def finish(self, started_at: float) -> CycleSchedule:
        """结束一轮：计算耗时、推进 deadline，必要时等待。

        Args:
            started_at: :meth:`begin` 返回的本轮开始时刻。

        Returns:
            本轮的节拍结果（含耗时、等待时长与超时量）。
        """
        ended_at = self._clock()
        elapsed_ms = max(0.0, (ended_at - started_at) * MILLISECONDS_PER_SECOND)
        overrun_ms = max(0.0, elapsed_ms - self._interval_ms)
        late = overrun_ms > 0.0

        self._deadline += self.interval_seconds
        sleep_ms = 0.0
        re_aligned = False
        remaining = self._deadline - ended_at
        if remaining > 0:
            sleep_ms = remaining * MILLISECONDS_PER_SECOND
            self._sleep(remaining)
        else:
            # 单轮耗时超过周期：不等待，并把节拍基准重对齐到当前时刻，
            # 避免"每轮都欠账"导致的追赶风暴。
            re_aligned = True
            self._deadline = ended_at

        self._cycles += 1
        if late:
            self._late_cycles += 1
            self._overrun_total_ms += overrun_ms
            self._overrun_max_ms = max(self._overrun_max_ms, overrun_ms)
        if re_aligned:
            self._re_alignments += 1

        schedule = CycleSchedule(
            cycle_id=self._cycles,
            started_at=started_at,
            ended_at=ended_at,
            elapsed_ms=elapsed_ms,
            interval_ms=self._interval_ms,
            sleep_ms=sleep_ms,
            overrun_ms=overrun_ms,
            late=late,
            re_aligned=re_aligned,
        )
        self._last = schedule
        if late:
            logger.warning("late cycle（处理耗时超过周期）: %s", schedule.describe())
        else:
            logger.debug("本轮节拍: %s", schedule.describe())
        return schedule
