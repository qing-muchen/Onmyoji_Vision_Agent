"""实时状态监控循环（Phase 3.1 / 3.2）。

把"单帧状态判定"变成**按固定周期运行**的监控。判定入口可以是**证据流水线**（Phase 5 推荐）
或状态分类器本身（Phase 3.0 行为，兼容保留）：

```text
每 interval_ms: 取帧（None → 跳过，计入 skipped）
                → 判定：EvidenceFrameProcessor.process(frame)（推荐）
                       或 StateClassifier.classify(frame)（旧路径）
                → stabilizer（逐帧 → 稳定状态）
                → 只有稳定状态改变才产生 StateEvent → 回调 + history 即时落盘
```

设计要点：

* **固定节拍**：等待按"上次节拍 + interval"计算，单轮耗时不会让周期漂移；
* **稳定化先行**：事件只由**稳定状态**驱动，逐帧抖动不产生事件；首次确认只建立基线；
* **可注入时钟与睡眠**（``clock`` / ``sleep`` / ``now``）：循环逻辑无需真实等待即可测试；
* **判定失败不崩溃**（Phase 5）：证据检测 / OCR / 模板异常被捕获，该帧降级为 ``unknown``
  并把原因写入 ``reason`` / ``detail``（计入 :attr:`StateMonitor.errors`）；
* **不吞其余异常**：只捕获 ``KeyboardInterrupt`` 以正常收尾（中断时也会落盘），
  历史写入失败等真实错误向上传播（见 :class:`~src.core.exceptions.MonitorError`）。

运行摘要见 :mod:`src.runtime.state_monitor.report`；本模块不做状态转移与任何交互。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.core.config import AppConfig, RuntimeStabilizationConfig, RuntimeStateMonitorConfig
from src.core.exceptions import OnmyojiVisionError
from src.runtime.state_monitor.events import StateEvent
from src.runtime.state_monitor.history import StateHistoryStore
from src.runtime.state_monitor.report import MonitorReport, StopReason, local_now_iso
from src.runtime.state_monitor.source import FrameSource
from src.runtime.state_monitor.stabilizer import StableStateResult, StateStabilizer
from src.utils.logger import get_logger
from src.vision.capture.models import Frame
from src.vision.pipeline.frame_processor import EvidenceFrameProcessor, build_transition_probe
from src.vision.state.classifier import StateClassifier
from src.vision.state.models import MIN_CONFIDENCE, StateName, StateResult, StateSource

__all__ = ["StateMonitor"]

logger = get_logger("runtime.state_monitor.monitor")


class StateMonitor:
    """按固定周期采集帧、判定状态、稳定化并记录变化的监控器。

    数据流：``capture → judge（证据流水线或分类器）→ stabilizer → stable state → event``；
    事件只由稳定状态驱动，逐帧抖动不产生事件。

    Args:
        classifier: 状态判定入口：:class:`~src.vision.state.classifier.StateClassifier`（旧路径）
            或 :class:`~src.vision.pipeline.frame_processor.EvidenceFrameProcessor`（Phase 5 推荐）。
        source: 帧来源（生产环境为 ``CaptureFrameSource``）。
        config: 状态监控配置（周期、历史路径、事件上限）。
        processor: 证据流水线；显式传入时优先于 ``classifier``（渐进迁移）。
        stabilizer / history: 稳定器与历史存储；省略时按各自配置创建。
        clock / sleep / now: 时钟、等待与时间戳函数（可在测试中注入）。
        on_result / on_event / on_stable: 识别 / 稳定状态变化 / 稳定化处理回调。
    """

    def __init__(
        self,
        classifier: StateClassifier | EvidenceFrameProcessor,
        source: FrameSource,
        config: RuntimeStateMonitorConfig,
        *,
        processor: EvidenceFrameProcessor | None = None,
        stabilizer: StateStabilizer | None = None,
        history: StateHistoryStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], str] = local_now_iso,
        on_result: Callable[[StateResult], None] | None = None,
        on_event: Callable[[StateEvent], None] | None = None,
        on_stable: Callable[[StableStateResult], None] | None = None,
    ) -> None:
        # 渐进迁移（Phase 5）：第一个位置参数既可以是证据流水线（推荐），也可以是状态分类器
        # （Phase 3.0 旧行为）；显式传入的 ``processor`` 优先级最高。
        active_processor = processor
        active_classifier: StateClassifier | None = None
        if isinstance(classifier, EvidenceFrameProcessor):
            if active_processor is None:
                active_processor = classifier
        else:
            active_classifier = classifier
        self._processor = active_processor
        self._classifier = active_classifier
        self._source = source
        self._config = config
        self._stabilizer = (
            stabilizer if stabilizer is not None else StateStabilizer(RuntimeStabilizationConfig())
        )
        self._history = history if history is not None else StateHistoryStore(config)
        self._clock = clock
        self._sleep = sleep
        self._now = now
        self._on_result = on_result
        self._on_event = on_event
        self._on_stable = on_stable
        self._events: list[StateEvent] = []
        self._current_state: StateName | None = None
        self._last_result: StateResult | None = None
        self._last_stable: StableStateResult | None = None
        self._cycles = 0
        self._classified = 0
        self._skipped = 0
        self._errors = 0

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        source: FrameSource,
        processor: EvidenceFrameProcessor | None = None,
        classifier: StateClassifier | None = None,
        stabilizer: StateStabilizer | None = None,
        history: StateHistoryStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], str] = local_now_iso,
        on_result: Callable[[StateResult], None] | None = None,
        on_event: Callable[[StateEvent], None] | None = None,
        on_stable: Callable[[StableStateResult], None] | None = None,
    ) -> StateMonitor:
        """按应用配置装配监控器。

        Args:
            config: 应用配置（``runtime`` / ``evidence`` / ``state`` / ``scene`` 各段）。
            source: 帧来源。
            processor: 证据流水线；与 ``classifier`` 都省略时按 ``config`` 装配**完整证据流水线**
                （含转场证据；转场数据集不可用时自动降级，不会抛异常）。
            classifier: 显式使用状态分类器（帧路径，Phase 3.x 行为）；仅在 ``processor`` 省略时生效。
            stabilizer / history: 省略时按 ``config`` 自动创建。
            clock / sleep / now / on_result / on_event / on_stable: 见 :meth:`__init__`。

        Returns:
            监控器实例。
        """
        active_stabilizer = (
            stabilizer if stabilizer is not None else StateStabilizer(config.runtime.stabilization)
        )
        active_processor = processor
        if active_processor is None and classifier is None:
            # Phase 5 默认：证据驱动的完整流水线（含转场证据；数据缺失时优雅降级）
            active_processor = EvidenceFrameProcessor.from_config(
                config, transition_classifier=build_transition_probe(config)
            )
        entry: StateClassifier | EvidenceFrameProcessor = (
            active_processor if active_processor is not None else classifier  # type: ignore[assignment]
        )
        return cls(
            entry,
            source,
            config.runtime.state_monitor,
            processor=active_processor,
            stabilizer=active_stabilizer,
            history=history,
            clock=clock,
            sleep=sleep,
            now=now,
            on_result=on_result,
            on_event=on_event,
            on_stable=on_stable,
        )

    @property
    def config(self) -> RuntimeStateMonitorConfig:
        """当前监控配置。"""
        return self._config

    @property
    def processor(self) -> EvidenceFrameProcessor | None:
        """使用的证据流水线；帧路径模式（显式注入 ``classifier``）下为 ``None``。"""
        return self._processor

    @property
    def classifier(self) -> StateClassifier:
        """使用的状态分类器。

        证据流水线模式下返回**流水线内部**使用的分类器（``processor.classifier``），
        因此 ``monitor.classifier`` 始终是 :class:`StateClassifier` 实例。
        """
        if self._processor is not None:
            return self._processor.classifier
        assert self._classifier is not None  # 构造时保证二者必有其一
        return self._classifier

    @property
    def errors(self) -> int:
        """判定失败（降级为 ``unknown``）的帧数（Phase 5）。"""
        return self._errors

    @property
    def source(self) -> FrameSource:
        """使用的帧来源。"""
        return self._source

    @property
    def history(self) -> StateHistoryStore:
        """历史存储。"""
        return self._history

    @property
    def stabilizer(self) -> StateStabilizer:
        """状态稳定器。"""
        return self._stabilizer

    @property
    def current_state(self) -> StateName | None:
        """当前**稳定状态**；尚未确认为 ``None``。"""
        return self._current_state

    @property
    def current_raw_state(self) -> StateName | None:
        """最近一帧的**原始**识别状态（未经稳定化）。"""
        return self._last_result.state_name if self._last_result is not None else None

    @property
    def last_result(self) -> StateResult | None:
        """最近一次原始识别结果。"""
        return self._last_result

    @property
    def last_stable(self) -> StableStateResult | None:
        """最近一次稳定化结果（含 raw/stable 与累积进度）。"""
        return self._last_stable

    @property
    def events(self) -> tuple[StateEvent, ...]:
        """本次运行已产生的稳定状态变化事件。"""
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
        """处理一帧：识别 → 稳定化 → 变化检测 → 记录事件（可独立测试）。

        事件只由稳定状态驱动（首次确认只建立基线）；返回本帧的原始识别结果，
        稳定化结果见 :attr:`last_stable`。

        Args:
            frame: 待识别的帧。
        """
        result = self._classify(frame)
        self._last_result = result
        self._classified += 1

        stable = self._stabilizer.update(result)
        self._last_stable = stable

        if self._current_state is None and stable.is_confirmed and stable.state_name is not None:
            # 首次确认稳定状态：建立基线（没有"前一个稳定状态"，不构成变化）
            self._current_state = stable.state_name
            self._history.set_initial_state(stable.state_name)
            logger.info("初始稳定状态基线: %s（不产生事件）", stable.state_name.value)
        elif stable.changed and stable.state_name is not None:
            event = StateEvent(
                previous_state=self._current_state,
                current_state=stable.state_name,
                timestamp=self._now(),
                confidence=stable.confidence,
            )
            self._events.append(event)
            self._current_state = stable.state_name
            logger.info("稳定状态变化: %s", event.describe())
            self._history.append(event, timestamp=event.timestamp)
            if self._on_event is not None:
                self._on_event(event)

        if self._on_stable is not None:
            self._on_stable(stable)
        if self._on_result is not None:
            self._on_result(result)
        return result

    def _classify(self, frame: Frame) -> StateResult:
        """判定一帧的状态：证据流水线优先，失败降级为 ``unknown``（Phase 5）。

        Runtime 的职责是**调度**，不负责 detect / aggregate / OCR / 模板匹配；
        证据侧抛出的异常（检测失败、OCR 不可用、模板缺失等
        :class:`~src.core.exceptions.OnmyojiVisionError`）在此被捕获——该帧降级为
        ``unknown`` 并记录失败原因，**绝不让监控循环崩溃**（后续帧照常处理）。

        Args:
            frame: 待判定的帧。

        Returns:
            状态结果；判定失败时为 ``unknown``（``reason`` / ``detail`` 含失败原因）。
        """
        try:
            if self._processor is not None:
                return self._processor.process(frame)
            assert self._classifier is not None
            return self._classifier.classify(frame)
        except OnmyojiVisionError as exc:
            self._errors += 1
            logger.error("状态判定失败（第 %d 次），本帧降级为 unknown: %s", self._errors, exc)
            return StateResult(
                state_name=StateName.UNKNOWN,
                confidence=MIN_CONFIDENCE,
                source=StateSource.FALLBACK,
                detail=f"state resolution failed: {exc}",
                reason=f"evidence pipeline failure: {exc}",
            )

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
            运行摘要（含事件、跳过次数与停止原因）；历史写入失败时抛
            :class:`~src.core.exceptions.MonitorError`。
        """
        self._events.clear()
        self._current_state = self._last_result = self._last_stable = None
        self._cycles = self._classified = self._skipped = 0
        self._errors = 0
        self._stabilizer.reset()

        started_at = self._now()
        self._history.start(timestamp=started_at, interval_ms=self._config.interval_ms)
        interval = self._config.interval_seconds
        start_clock = self._clock()
        deadline = start_clock
        stop_reason = StopReason.MAX_CYCLES
        logger.info(
            "状态监控开始：interval=%dms, max_cycles=%s, duration=%ss, stabilization=%s(%d 帧)",
            self._config.interval_ms,
            max_cycles,
            duration_s,
            self._stabilizer.is_enabled,
            self._stabilizer.required_frames,
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
        self._history.update_progress(cycles=self._cycles, skipped=self._skipped, timestamp=ended_at)
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
