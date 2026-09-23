"""实时状态监控循环（Phase 3.1 / 3.2 / 5.2）。

把"单帧状态判定"变成**按固定周期运行**的监控。判定入口可以是**证据流水线**（Phase 5 推荐）
或状态分类器本身（Phase 3.0 行为，兼容保留）：

```text
每 interval_ms: 取帧（None → 跳过，计入 skipped）
                → 判定：EvidenceFrameProcessor.process(frame)（推荐）
                       或 StateClassifier.classify(frame)（旧路径）
                → stabilizer（逐帧 → 稳定状态）
                → 只有稳定状态改变才产生 StateEvent → 回调 + history 即时落盘
                → 运行指标（可选，Phase 5.2）：cycle_id / timestamp / state / stable / latency
```

节拍由 :class:`~src.runtime.state_monitor.scheduler.CycleScheduler` 提供（Phase 5.2）：

* **deadline based**：等待按"上一次 deadline + interval"计算，单轮耗时被吸收进周期，
  不会让周期漂移（不是 ``sleep(interval)``）；
* **late cycle 记录**：单轮耗时超过 interval 时记为 ``late`` 并累计 ``overrun``，
  同时重对齐节拍基准，避免"追赶风暴"（详见调度器模块文档）。

设计要点：

* **稳定化先行**：事件只由**稳定状态**驱动，逐帧抖动不产生事件；首次确认只建立基线；
* **可注入时钟与睡眠**（``clock`` / ``sleep`` / ``now``）：循环逻辑无需真实等待即可测试；
* **判定失败不崩溃**（Phase 5）：证据检测 / OCR / 模板异常被捕获，该帧降级为 ``unknown``
  并把原因写入 ``reason`` / ``detail``（计入 :attr:`StateMonitor.errors`）；
* **采集失败不崩溃**（Phase 5.2）：真实窗口帧来源把窗口/采集问题转成"本轮无帧"
  （``source.get_frame() is None``），循环跳过该轮并继续，原因写入运行指标
  （``metrics`` 的 ``skip_reason``）；
* **不进 Action 层**：本模块只采集、判定、记录；不点击、不键入、不切换窗口、不抢焦点；
* **不吞其余异常**：只捕获 ``KeyboardInterrupt`` 以正常收尾（中断时也会落盘），
  历史写入失败等真实错误向上传播（见 :class:`~src.core.exceptions.MonitorError`）。

运行摘要见 :mod:`src.runtime.state_monitor.report`，逐轮指标见
:mod:`src.runtime.state_monitor.metrics`；本模块不做状态转移与任何交互。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.core.config import AppConfig, RuntimeStabilizationConfig, RuntimeStateMonitorConfig
from src.core.exceptions import OnmyojiVisionError
from src.runtime.state_monitor.events import StateEvent
from src.runtime.state_monitor.history import StateHistoryStore
from src.runtime.state_monitor.metrics import (
    CycleMetrics,
    CycleStatus,
    RuntimeMetricsRecorder,
)
from src.runtime.state_monitor.report import MonitorReport, StopReason, local_now_iso
from src.runtime.state_monitor.scheduler import CycleSchedule, CycleScheduler
from src.runtime.state_monitor.source import FrameSource, StatusAwareSource
from src.runtime.state_monitor.stabilizer import StableStateResult, StateStabilizer
from src.runtime.state_monitor.timing import CycleTiming
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
        source: 帧来源（生产环境为 ``WindowFrameSource`` 或 ``CaptureFrameSource``）。
        config: 状态监控配置（周期、历史路径、事件上限）。
        processor: 证据流水线；显式传入时优先于 ``classifier``（渐进迁移）。
        stabilizer / history: 稳定器与历史存储；省略时按各自配置创建。
        scheduler: 节拍调度器；省略时按 ``config.interval_ms`` 创建（Phase 5.2）。
        metrics: 逐轮运行指标记录器；省略时不记录（``from_config`` 会按配置装配）。
        clock / sleep / now: 时钟、等待与时间戳函数（可在测试中注入）。
        timing: 是否采集每轮耗时画像（Phase 5.1；默认 ``False`` = 零额外开销）。
        on_result / on_event / on_stable: 识别 / 稳定状态变化 / 稳定化处理回调。
        on_cycle: 每轮结束（含跳过轮）后的回调，接收该轮
            :class:`~src.runtime.state_monitor.scheduler.CycleSchedule`；
            回调执行时本轮运行指标**已**写入 ``metrics``（Phase 5.2）。
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
        scheduler: CycleScheduler | None = None,
        metrics: RuntimeMetricsRecorder | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], str] = local_now_iso,
        timing: bool = False,
        on_result: Callable[[StateResult], None] | None = None,
        on_event: Callable[[StateEvent], None] | None = None,
        on_stable: Callable[[StableStateResult], None] | None = None,
        on_cycle: Callable[[CycleSchedule], None] | None = None,
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
        self._scheduler = (
            scheduler
            if scheduler is not None
            else CycleScheduler(config.interval_ms, clock=clock, sleep=sleep)
        )
        self._metrics = metrics
        self._clock = clock
        self._now = now
        self._timing_enabled = timing
        self._last_timing: CycleTiming | None = None
        self._on_result = on_result
        self._on_event = on_event
        self._on_stable = on_stable
        self._on_cycle = on_cycle
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
        scheduler: CycleScheduler | None = None,
        metrics: RuntimeMetricsRecorder | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], str] = local_now_iso,
        timing: bool = False,
        on_result: Callable[[StateResult], None] | None = None,
        on_event: Callable[[StateEvent], None] | None = None,
        on_stable: Callable[[StableStateResult], None] | None = None,
        on_cycle: Callable[[CycleSchedule], None] | None = None,
    ) -> StateMonitor:
        """按应用配置装配监控器。

        Args:
            config: 应用配置（``runtime`` / ``evidence`` / ``state`` / ``scene`` 各段）。
            source: 帧来源。
            processor: 证据流水线；与 ``classifier`` 都省略时按 ``config`` 装配**完整证据流水线**
                （含转场证据；转场数据集不可用时自动降级，不会抛异常）。
            classifier: 显式使用状态分类器（帧路径，Phase 3.x 行为）；仅在 ``processor`` 省略时生效。
            stabilizer / history: 省略时按 ``config`` 自动创建。
            scheduler: 节拍调度器；省略时按 ``runtime.state_monitor.interval_ms`` 创建。
            metrics: 运行指标记录器；省略时按 ``runtime.state_monitor.save_metrics`` 自动创建
                （默认保留在内存中，开启后写入 ``metrics_path``）。
            clock / sleep / now: 时钟、等待与时间戳函数（可在测试中注入）。
            timing: 是否采集每轮耗时画像（Phase 5.1；默认 ``False``）。
            on_result / on_event / on_stable: 见 :meth:`__init__`。

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
        active_metrics = (
            metrics
            if metrics is not None
            else RuntimeMetricsRecorder.from_config(config.runtime.state_monitor)
        )
        return cls(
            entry,
            source,
            config.runtime.state_monitor,
            processor=active_processor,
            stabilizer=active_stabilizer,
            history=history,
            scheduler=scheduler,
            metrics=active_metrics,
            clock=clock,
            sleep=sleep,
            now=now,
            timing=timing,
            on_result=on_result,
            on_event=on_event,
            on_stable=on_stable,
            on_cycle=on_cycle,
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
    def scheduler(self) -> CycleScheduler:
        """节拍调度器（含 late cycle / overrun 统计，Phase 5.2）。"""
        return self._scheduler

    @property
    def metrics(self) -> RuntimeMetricsRecorder | None:
        """逐轮运行指标记录器；未启用时为 ``None``（直接构造时可省略）。"""
        return self._metrics

    @property
    def last_schedule(self) -> CycleSchedule | None:
        """最近一轮的节拍结果；尚未运行过时为 ``None``。"""
        return self._scheduler.last

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

    def step(self, frame: Frame, *, frame_ms: float = 0.0) -> StateResult:
        """处理一帧：识别 → 稳定化 → 变化检测 → 记录事件（可独立测试）。

        事件只由稳定状态驱动（首次确认只建立基线）；返回本帧的原始识别结果，
        稳定化结果见 :attr:`last_stable`。

        Args:
            frame: 待识别的帧。
            frame_ms: 取帧耗时（由 :meth:`run` 传入；**仅**用于耗时画像，默认 ``0.0``）。
        """
        judge_started = time.perf_counter()
        result = self._classify(frame)
        judge_ms = (time.perf_counter() - judge_started) * 1000.0
        self._last_result = result
        self._classified += 1

        stabilize_started = time.perf_counter()
        stable = self._stabilizer.update(result)
        stabilize_ms = (time.perf_counter() - stabilize_started) * 1000.0
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
        if self._timing_enabled:
            self._record_timing(frame_ms, judge_ms, stabilize_ms, result)
        return result

    def _record_timing(
        self,
        frame_ms: float,
        judge_ms: float,
        stabilize_ms: float,
        result: StateResult,
    ) -> None:
        """记录本轮耗时画像（Phase 5.1；不参与任何判定与事件逻辑）。

        ``detect_ms`` / ``state_ms`` 来自 :attr:`EvidenceFrameProcessor.last_timing`
        （仅在 ``timing=True`` 时采集）；帧路径下两者为 ``0.0``。
        """
        detect_ms = 0.0
        state_ms = 0.0
        if self._processor is not None and self._processor.last_timing is not None:
            timing = self._processor.last_timing
            detect_ms = timing.detect_ms
            state_ms = timing.classify_ms
        self._last_timing = CycleTiming(
            cycle=self._cycles + 1,
            frame_ms=frame_ms,
            detect_ms=detect_ms,
            state_ms=state_ms,
            judge_ms=judge_ms,
            stabilize_ms=stabilize_ms,
            total_ms=frame_ms + judge_ms + stabilize_ms,
            state=result.state_name,
        )
        logger.debug("本轮耗时画像：%s", self._last_timing.describe())

    @property
    def last_timing(self) -> CycleTiming | None:
        """最近一轮的耗时画像；未启用 ``timing`` 时为 ``None``。"""
        return self._last_timing

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
                if self._timing_enabled:
                    # 采集耗时分解（Phase 5.1）；不影响判定结果
                    return self._processor.process(frame, profile=True)
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

    def _record_metrics(
        self,
        *,
        schedule: CycleSchedule,
        timestamp: str,
        frame_ms: float,
        frame: Frame | None,
        result: StateResult | None,
        failed: bool,
    ) -> None:
        """记录本轮运行指标（Phase 5.2；未启用 ``metrics`` 时是空操作）。

        只记录"运行"本身：轮次、时间戳、原始/稳定状态、耗时与跳过原因；
        **不记录截图，也不记录任何用户数据**。跳过原因来自实现了
        :class:`~src.runtime.state_monitor.source.StatusAwareSource` 的帧来源
        （例如 ``WindowFrameSource``：窗口不存在 / 最小化 / 不在前台 / 采集失败）。

        Args:
            schedule: 本轮节拍结果（提供耗时与超时量）。
            timestamp: 本轮时间戳（与历史记录使用同一个值）。
            frame_ms: 取帧耗时。
            frame: 本轮帧；未取到时为 ``None``。
            result: 本轮原始判定结果；未取到帧时为 ``None``。
            failed: 本轮判定是否失败（降级为 ``unknown``）。
        """
        metrics = self._metrics
        if metrics is None:
            return
        if frame is None:
            status = CycleStatus.SKIPPED
        elif failed:
            status = CycleStatus.ERROR
        else:
            status = CycleStatus.OK
        skip_reason = None
        if frame is None and isinstance(self._source, StatusAwareSource):
            skip_reason = self._source.last_skip_reason
        metrics.record(
            CycleMetrics(
                cycle_id=self._cycles,
                timestamp=timestamp,
                started_at=schedule.started_at,
                latency_ms=schedule.elapsed_ms,
                frame_ms=frame_ms,
                state=result.state_name if result is not None else None,
                confidence=result.confidence if result is not None else 0.0,
                stable_state=self._current_state,
                status=status,
                skip_reason=skip_reason,
                frame_source=frame.source if frame is not None else None,
            ),
            overrun_ms=schedule.overrun_ms,
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
        start_clock = self._clock()
        self._scheduler.reset(start_at=start_clock)
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

                cycle_started = self._scheduler.begin()
                errors_before = self._errors
                frame_started = time.perf_counter()
                frame = self._source.get_frame()
                frame_ms = (time.perf_counter() - frame_started) * 1000.0
                result: StateResult | None = None
                if frame is None:
                    self._skipped += 1
                    logger.debug("本轮无可用帧，跳过（累计 %d）", self._skipped)
                else:
                    result = self.step(frame, frame_ms=frame_ms)

                self._cycles += 1
                timestamp = self._now()
                self._history.update_progress(
                    cycles=self._cycles,
                    skipped=self._skipped,
                    timestamp=timestamp,
                )

                schedule = self._scheduler.finish(cycle_started)
                self._record_metrics(
                    schedule=schedule,
                    timestamp=timestamp,
                    frame_ms=frame_ms,
                    frame=frame,
                    result=result,
                    failed=self._errors > errors_before,
                )
                if self._on_cycle is not None:
                    self._on_cycle(schedule)
        except KeyboardInterrupt:
            stop_reason = StopReason.INTERRUPT
            logger.warning("收到中断信号（Ctrl+C），正在收尾并保存历史")

        ended_at = self._now()
        self._history.update_progress(cycles=self._cycles, skipped=self._skipped, timestamp=ended_at)
        self._history.save()
        if self._scheduler.late_cycles:
            logger.warning(
                "本轮运行有 %d/%d 个 late cycle（单轮耗时超过 %dms）：累计超时 %.1fms，最大 %.1fms",
                self._scheduler.late_cycles,
                self._scheduler.cycles,
                self._config.interval_ms,
                self._scheduler.overrun_total_ms,
                self._scheduler.overrun_max_ms,
            )
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
