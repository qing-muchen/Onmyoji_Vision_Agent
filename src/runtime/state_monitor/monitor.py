"""实时状态监控循环（Phase 3.1）。

把单帧 :class:`~src.vision.state.classifier.StateClassifier` 变成**按固定周期运行**的监控：

```text
每 interval_ms:
  1. frame = source.get_frame()            # None → 跳过该轮（计入 skipped）
  2. result = classifier.classify(frame)   # 复用 Phase 3.0
  3. 回调 on_result（Demo 打印当前状态）
  4. 只有 state_name 改变 → StateEvent → 回调 on_event + history 即时落盘
```

设计要点：

* **固定节拍**：等待时长按"上次节拍 + interval"计算，而不是"每轮 sleep(interval)"，
  否则单轮耗时会让周期不断漂移；
* **可注入时钟与睡眠**（``clock`` / ``sleep`` / ``now``）：循环逻辑无需真实等待即可测试；
* **首次识别只建立基线**（不产生事件，因为没有"前一个状态"），基线写入 ``initial_state``；
* **不吞异常**：只捕获 ``KeyboardInterrupt`` 以正常收尾（中断时也会落盘）；
  历史写入失败等真实错误向上传播（见 :class:`~src.core.exceptions.MonitorError`）。

运行摘要与停止原因见 :mod:`src.runtime.state_monitor.report`。
本模块不做状态转移、不做决策、不触发任何键鼠交互。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.core.config import AppConfig, RuntimeStateMonitorConfig
from src.runtime.state_monitor.events import StateEvent
from src.runtime.state_monitor.history import StateHistoryStore
from src.runtime.state_monitor.report import MonitorReport, StopReason, local_now_iso
from src.runtime.state_monitor.source import FrameSource
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.state.classifier import StateClassifier
from src.vision.state.models import StateName, StateResult

__all__ = ["StateMonitor"]

logger = get_logger("runtime.state_monitor.monitor")


class StateMonitor:
    """按固定周期采集帧、识别状态并记录变化的监控器。

    Args:
        classifier: 状态分类器（Phase 3.0 的 :class:`StateClassifier` 或同接口替身）。
        source: 帧来源（生产环境为
            :class:`~src.runtime.state_monitor.source.CaptureFrameSource`）。
        config: 状态监控配置（周期、历史路径、事件上限）。
        history: 历史存储；省略时按配置创建。
        clock: 单调时钟（秒），用于固定节拍；测试可注入假时钟。
        sleep: 等待函数；测试可注入空实现以避免真实等待。
        now: 时间戳函数（ISO 8601）；测试可注入固定值。
        on_result: 每次识别后的回调（Demo 用来打印当前状态）。
        on_event: 每次产生状态变化事件后的回调。
    """

    def __init__(
        self,
        classifier: StateClassifier,
        source: FrameSource,
        config: RuntimeStateMonitorConfig,
        *,
        history: StateHistoryStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], str] = local_now_iso,
        on_result: Callable[[StateResult], None] | None = None,
        on_event: Callable[[StateEvent], None] | None = None,
    ) -> None:
        self._classifier = classifier
        self._source = source
        self._config = config
        self._history = history if history is not None else StateHistoryStore(config)
        self._clock = clock
        self._sleep = sleep
        self._now = now
        self._on_result = on_result
        self._on_event = on_event
        self._events: list[StateEvent] = []
        self._current_state: StateName | None = None
        self._last_result: StateResult | None = None
        self._cycles = 0
        self._classified = 0
        self._skipped = 0

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        source: FrameSource,
        classifier: StateClassifier | None = None,
        history: StateHistoryStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], str] = local_now_iso,
        on_result: Callable[[StateResult], None] | None = None,
        on_event: Callable[[StateEvent], None] | None = None,
    ) -> StateMonitor:
        """按应用配置装配监控器。

        Args:
            config: 应用配置（``runtime.state_monitor`` 与 ``state`` / ``scene`` 两段）。
            source: 帧来源。
            classifier: 状态分类器；省略时按 ``config`` 创建 ``StateClassifier``。
            history: 历史存储；省略时按配置创建。
            clock: 单调时钟；默认 ``time.monotonic``。
            sleep: 等待函数；默认 ``time.sleep``。
            now: 时间戳函数；默认本地 ISO 8601。
            on_result: 识别回调。
            on_event: 事件回调。

        Returns:
            监控器实例。
        """
        active = classifier if classifier is not None else StateClassifier.from_config(config)
        return cls(
            active,
            source,
            config.runtime.state_monitor,
            history=history,
            clock=clock,
            sleep=sleep,
            now=now,
            on_result=on_result,
            on_event=on_event,
        )

    @property
    def config(self) -> RuntimeStateMonitorConfig:
        """当前监控配置。"""
        return self._config

    @property
    def classifier(self) -> StateClassifier:
        """使用的状态分类器。"""
        return self._classifier

    @property
    def source(self) -> FrameSource:
        """使用的帧来源。"""
        return self._source

    @property
    def history(self) -> StateHistoryStore:
        """历史存储。"""
        return self._history

    @property
    def current_state(self) -> StateName | None:
        """最近一次识别到的状态；尚未识别时为 ``None``。"""
        return self._current_state

    @property
    def last_result(self) -> StateResult | None:
        """最近一次识别结果。"""
        return self._last_result

    @property
    def events(self) -> tuple[StateEvent, ...]:
        """本次运行已产生的状态变化事件。"""
        return tuple(self._events)

    @property
    def cycles(self) -> int:
        """已完成的周期数。"""
        return self._cycles

    @property
    def skipped(self) -> int:
        """因帧不可用而跳过的周期数。"""
        return self._skipped

    @property
    def event_count(self) -> int:
        """事件数量。"""
        return len(self._events)

    def step(self, frame: Frame) -> StateResult:
        """处理一帧：识别状态、检测变化、记录事件（可独立测试）。

        Args:
            frame: 待识别的帧。

        Returns:
            本次识别的状态结果。
        """
        result = self._classifier.classify(frame)
        self._last_result = result
        self._classified += 1

        previous = self._current_state
        if previous is None:
            # 首次识别只建立基线：没有"前一个状态"，不构成状态变化
            self._current_state = result.state_name
            self._history.set_initial_state(result.state_name)
            logger.info("初始状态基线: %s（不产生事件）", result.state_name.value)
        elif result.state_name is not previous:
            event = StateEvent(
                previous_state=previous,
                current_state=result.state_name,
                timestamp=self._now(),
                confidence=result.confidence,
            )
            self._events.append(event)
            self._current_state = result.state_name
            logger.info("状态变化: %s", event.describe())
            self._history.append(event, timestamp=event.timestamp)
            if self._on_event is not None:
                self._on_event(event)

        if self._on_result is not None:
            self._on_result(result)
        return result

    def run(
        self,
        *,
        duration_s: float | None = None,
        max_cycles: int | None = None,
    ) -> MonitorReport:
        """按固定周期持续监控，直到达到限制或收到中断。

        Args:
            duration_s: 最长运行时长（秒）；``None`` 表示不限时。
            max_cycles: 最多运行的周期数；``None`` 表示不限次。

        Returns:
            运行摘要（含事件、跳过次数与停止原因）。

        Raises:
            MonitorError: 历史文件无法写入。
        """
        self._reset()
        started_at = self._now()
        self._history.start(timestamp=started_at, interval_ms=self._config.interval_ms)
        interval = self._config.interval_seconds
        start_clock = self._clock()
        deadline = start_clock
        stop_reason = StopReason.MAX_CYCLES
        logger.info(
            "状态监控开始：interval=%dms, max_cycles=%s, duration=%ss",
            self._config.interval_ms,
            max_cycles,
            duration_s,
        )
        try:
            while True:
                if max_cycles is not None and self._cycles >= max_cycles:
                    stop_reason = StopReason.MAX_CYCLES
                    break
                if duration_s is not None and (self._clock() - start_clock) >= duration_s:
                    stop_reason = StopReason.DURATION
                    break

                frame = self._source.get_frame()
                if frame is None:
                    self._skipped += 1
                    logger.debug("本轮无可用帧，跳过（累计 %d）", self._skipped)
                else:
                    self.step(frame)

                self._cycles += 1
                self._history.update_progress(
                    cycles=self._cycles,
                    skipped=self._skipped,
                    timestamp=self._now(),
                )

                deadline += interval
                remaining = deadline - self._clock()
                if remaining > 0:
                    self._sleep(remaining)
                else:
                    # 单轮耗时超过周期：重新对齐节拍，避免"追赶风暴"
                    deadline = self._clock()
        except KeyboardInterrupt:
            stop_reason = StopReason.INTERRUPT
            logger.warning("收到中断信号（Ctrl+C），正在收尾并保存历史")

        ended_at = self._now()
        self._history.update_progress(
            cycles=self._cycles,
            skipped=self._skipped,
            timestamp=ended_at,
        )
        self._history.save()
        report = MonitorReport(
            cycles=self._cycles,
            classified=self._classified,
            skipped=self._skipped,
            events=tuple(self._events),
            initial_state=self._history.initial_state,
            final_state=self._current_state,
            started_at=started_at,
            ended_at=ended_at,
            stop_reason=stop_reason,
            history_path=str(self._history.path) if self._config.save_history else None,
        )
        logger.info("状态监控结束：%s", report.describe())
        return report

    def _reset(self) -> None:
        """重置本次运行的计数与事件（支持重复调用 :meth:`run`）。"""
        self._events.clear()
        self._current_state = None
        self._last_result = None
        self._cycles = 0
        self._classified = 0
        self._skipped = 0
