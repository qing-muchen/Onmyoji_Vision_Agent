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
  （Phase 7.3 起可**可选**接线 :class:`~src.runtime.vision_bridge.VisionSnapshotBridge`：
  它只把"本轮的视觉结果"转换成数据快照推送给 Action 侧的 resolver，仍然**不调用**
  ``ActionDispatcher``、不产生任何输入；未接线时行为与此前完全一致。）
  （Phase 7.5 起可**可选**接线 :class:`~src.runtime.task_bridge.TaskRuntimeBridge`：
  它把「稳定状态 + 每轮节拍」交给任务编排（内部持有 ``TaskEngine`` 与任务清单），
  同样不调用 ``ActionDispatcher``、不产生任何输入；本模块自身**不 import** ``src.task`` /
  ``src.action``。）
* **不吞其余异常**：只捕获 ``KeyboardInterrupt`` 以正常收尾（中断时也会落盘），
  历史写入失败等真实错误向上传播（见 :class:`~src.core.exceptions.MonitorError`）。

运行摘要见 :mod:`src.runtime.state_monitor.report`，逐轮指标见
:mod:`src.runtime.state_monitor.metrics`；本模块不做状态转移与任何交互。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

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
from src.runtime.state_monitor.recovery import (
    RecoveryAction,
    RecoveryDecision,
    RecoveryManager,
    RecoveryPolicy,
)
from src.runtime.state_monitor.runtime_diagnostics import (
    CycleDiagnosis,
    RuntimeDiagnostics,
    RuntimeHealth,
)
from src.runtime.vision_bridge import (  # Phase 11.1：快照失效原因（同层常量）
    INVALIDATION_RUNTIME_STOPPED,
    INVALIDATION_WINDOW_UNAVAILABLE,
)
from src.runtime.state_monitor.runtime_lifecycle import (
    LifecycleTransition,
    RuntimeLifecycle,
    RuntimeState,
)
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

#: Phase 11.8：yard 观测目标（证据名）。仅用于**读取**已存在的证据 / 画像，
#: 不代表 Runtime 参与视觉判定（判定仍完全由 vision 层完成）。
YARD_EVIDENCE_NAME: str = "bottom_menu_yard"


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
        lifecycle: RuntimeLifecycle | None = None,
        recovery: RecoveryManager | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], str] = local_now_iso,
        timing: bool = False,
        on_result: Callable[[StateResult], None] | None = None,
        on_event: Callable[[StateEvent], None] | None = None,
        on_stable: Callable[[StableStateResult], None] | None = None,
        on_cycle: Callable[[CycleSchedule], None] | None = None,
        vision_bridge: Any | None = None,
        task_bridge: Any | None = None,
        yard_metrics: bool = False,
        action_enabled: bool | None = None,
        action_backend: str | None = None,
    ) -> None:
        """（Phase 7.3 / 7.5 新增参数见末尾说明。）

        Args:
            classifier: 分类器或证据流水线（见类文档）。
            source: 帧来源。
            config: 监控配置。
            processor: 证据流水线（优先级最高）。
            stabilizer: 状态稳定化器。
            history: 状态历史存储。
            scheduler: 节拍调度器。
            metrics: 逐轮指标记录器。
            lifecycle: 生命周期状态机。
            recovery: 失败恢复管理器。
            clock: 单调时钟。
            sleep: 睡眠函数。
            now: 时间戳函数。
            timing: 是否采集耗时画像。
            on_result: 每帧原始结果回调。
            on_event: 状态变化事件回调。
            on_stable: 每帧稳定结果回调。
            on_cycle: 每轮结束（含跳过轮）回调。
            vision_bridge: **可选**的视觉快照桥（Phase 7.3）。传入后，``run()`` 会在每轮末尾
                把"本轮的帧 + 本轮检测到的证据 + 帧号（``cycles``）+ 轮号（``cycle_id``）"
                交给它，由它转换成 ``TargetSnapshot`` 并单向推送给 Action 侧的 resolver。
                **默认 ``None`` = 完全不接线**（行为与 Phase 5.4 / 7.2 一致）。
                接入原因：Runtime 是唯一能同时看到"帧 / 视觉结果 / 动作端口"的编排层
                （Action 不 import vision、Vision 不 import action），因此由本层完成编排。
                该桥**只推送快照数据**，既不调用 ``ActionDispatcher``、也不产生任何输入。
            task_bridge: **可选**的任务编排桥（Phase 7.5，例如
                :class:`~src.runtime.task_bridge.TaskRuntimeBridge`）。传入后：

                * 每帧稳定状态 → ``task_bridge.on_state_stable(stable)``（触发 / 驱动任务）；
                * 每轮结束 → ``task_bridge.on_cycle(schedule)``（任务计时；在快照推送**之后**，
                  因此任务决策能看到本轮最新快照）。

                **默认 ``None`` = 完全不接线**（行为与此前一致）。编排桥内部持有
                ``TaskEngine`` 与任务清单，本层**不**直接依赖 ``src.task`` / ``src.action``；
                它同样不调用 ``ActionDispatcher``、不产生任何输入。
            yard_metrics: **是否采集 yard 运行期观测**（Phase 11.8；默认 ``False`` = 零额外开销）。

                开启后每轮额外记录：evidence ``profile`` 里的 yard 单项耗时、yard 证据
                ``metadata`` 里的预筛候选数 / 回退状态 / 回退原因、视觉快照状态与累计失效、
                活动任务 id、以及动作层安全配置（``action_enabled`` / ``action_backend``）。
                全部为标量与短字符串，**不含图像、候选列表或模板列表**；只在
                :class:`~src.runtime.state_monitor.metrics.CycleMetrics` 上追加字段，
                不改变任何判定。
            action_enabled / action_backend: 动作层安全快照（仅用于观测记录；
                本层**不**接触 ``ActionDispatcher`` 或输入后端）。
        """
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
        self._diagnostics = RuntimeDiagnostics()
        self._lifecycle = (
            lifecycle if lifecycle is not None else RuntimeLifecycle(now=now)
        )
        self._recovery = (
            recovery if recovery is not None else RecoveryManager(RecoveryPolicy(), sleep=sleep)
        )
        self._stop_requested = False
        self._paused_cycles = 0
        self._clock = clock
        self._now = now
        self._timing_enabled = timing
        self._last_timing: CycleTiming | None = None
        self._on_result = on_result
        self._on_event = on_event
        self._on_stable = on_stable
        self._on_cycle = on_cycle
        self._vision_bridge = vision_bridge
        self._task_bridge = task_bridge
        # Phase 11.8：yard 运行期观测（默认关闭 ⇒ 零额外开销，且指标输出与既有版本一致）
        self._yard_metrics = bool(yard_metrics)
        self._action_enabled = action_enabled
        self._action_backend = action_backend
        self._last_detect_ms: float | None = None
        self._events: list[StateEvent] = []
        self._current_state: StateName | None = None
        self._last_result: StateResult | None = None
        self._last_stable: StableStateResult | None = None
        self._cycles = 0
        self._classified = 0
        self._skipped = 0
        self._errors = 0
        self._last_evidence_hits: int | None = None
        # Phase 7.3：最近一轮的证据结果（供视觉快照桥消费；不在本层做任何动作）
        self._last_evidences: list[Any] | None = None

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
        lifecycle: RuntimeLifecycle | None = None,
        recovery: RecoveryManager | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], str] = local_now_iso,
        timing: bool = False,
        on_result: Callable[[StateResult], None] | None = None,
        on_event: Callable[[StateEvent], None] | None = None,
        on_stable: Callable[[StableStateResult], None] | None = None,
        on_cycle: Callable[[CycleSchedule], None] | None = None,
        vision_bridge: Any | None = None,
        task_bridge: Any | None = None,
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
        active_recovery = (
            recovery
            if recovery is not None
            else RecoveryManager(
                RecoveryPolicy.from_config(config.runtime.recovery), sleep=sleep
            )
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
            lifecycle=lifecycle,
            recovery=active_recovery,
            clock=clock,
            sleep=sleep,
            now=now,
            timing=timing,
            on_result=on_result,
            on_event=on_event,
            on_stable=on_stable,
            on_cycle=on_cycle,
            # Phase 11.8：可选接线（省略 = 完全不接线，行为与此前一致）
            vision_bridge=vision_bridge,
            task_bridge=task_bridge,
            # Phase 11.8：yard 观测开关 + 动作层安全快照（仅用于记录，不改变任何行为）
            yard_metrics=bool(config.runtime.state_monitor.yard_metrics),
            action_enabled=bool(config.action.enabled),
            action_backend=str(config.action.backend),
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
    def diagnostics(self) -> RuntimeDiagnostics:
        """运行时诊断收集器（Phase 5.3；逐轮累积，供 :attr:`health` 使用）。"""
        return self._diagnostics

    @property
    def health(self) -> RuntimeHealth:
        """运行时健康快照（区分 runtime failure 与 vision unknown，Phase 5.3）。"""
        return self._diagnostics.health

    @property
    def last_evidence_hits(self) -> int | None:
        """最近一轮命中的证据条数（Phase 5.3；帧路径或尚未判定时为 ``None``）。"""
        return self._last_evidence_hits

    @property
    def last_evidences(self) -> tuple[Any, ...]:
        """最近一轮检测到的证据结果（Phase 7.3）。

        由 :meth:`step` → :meth:`_classify` 在**同一次检测**中顺带保留（不重复检测），
        供 :class:`~src.runtime.vision_bridge.VisionSnapshotBridge` 消费；
        没有走证据路径或判定失败时为空元组。**只读**，本层不据此做任何动作。
        """
        return tuple(self._last_evidences or ())

    @property
    def vision_bridge(self) -> Any | None:
        """当前接线的视觉快照桥（Phase 7.3；未接线时为 ``None``）。"""
        return self._vision_bridge

    @property
    def task_bridge(self) -> Any | None:
        """当前接线的任务编排桥（Phase 7.5；未接线时为 ``None``）。"""
        return self._task_bridge

    @property
    def lifecycle(self) -> RuntimeLifecycle:
        """生命周期状态机（Phase 5.4：``INITIALIZING`` / ``READY`` / ``RUNNING`` / …）。"""
        return self._lifecycle

    @property
    def recovery(self) -> RecoveryManager:
        """失败恢复策略管理器（Phase 5.4）。"""
        return self._recovery

    @property
    def runtime_state(self) -> RuntimeState:
        """当前运行时状态（等价于 ``monitor.lifecycle.state``）。"""
        return self._lifecycle.state

    @property
    def paused_cycles(self) -> int:
        """暂停期间空转的节拍数（不计入 ``cycles``）。"""
        return self._paused_cycles

    @property
    def stop_requested(self) -> bool:
        """是否已请求停止（``stop()`` 后为 ``True``，直到下次 ``run()``）。"""
        return self._stop_requested

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
        result, detect_ms, state_ms = self._classify(
            frame, profile=self._timing_enabled or self._yard_metrics
        )
        judge_ms = (time.perf_counter() - judge_started) * 1000.0
        self._last_detect_ms = detect_ms
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
        # 注：Phase 7.5 的任务编排在 run() 每轮末尾驱动（在快照推送之后），
        # 因此这里不通知任务桥——避免"本轮快照尚未就绪就触发任务"。
        if self._on_result is not None:
            self._on_result(result)
        if self._timing_enabled:
            self._record_timing(frame_ms, judge_ms, stabilize_ms, result, detect_ms, state_ms)
        return result

    def _record_timing(
        self,
        frame_ms: float,
        judge_ms: float,
        stabilize_ms: float,
        result: StateResult,
        detect_ms: float = 0.0,
        state_ms: float = 0.0,
    ) -> None:
        """记录本轮耗时画像（Phase 5.1；不参与任何判定与事件逻辑）。

        ``detect_ms`` / ``state_ms`` 由 :meth:`_classify` 在同一次判定中采集
        （与 :meth:`~src.vision.pipeline.frame_processor.EvidenceFrameProcessor.process`
        的 ``profile=True`` 同口径）；帧路径下两者为 ``0.0``。
        """
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

    def _classify(
        self, frame: Frame, *, profile: bool = False
    ) -> tuple[StateResult, float, float]:
        """判定一帧的状态：证据流水线优先，失败降级为 ``unknown``（Phase 5）。

        Runtime 的职责是**调度**，不负责 detect / aggregate / OCR / 模板匹配；
        证据侧抛出的异常（检测失败、OCR 不可用、模板缺失等
        :class:`~src.core.exceptions.OnmyojiVisionError`）在此被捕获——该帧降级为
        ``unknown`` 并记录失败原因，**绝不让监控循环崩溃**（后续帧照常处理）。

        证据路径下本方法显式走 ``detect → classify`` 两步（与
        :meth:`~src.vision.pipeline.frame_processor.EvidenceFrameProcessor.process`
        内部完全一致，**不重复检测**），唯一目的是在同一次检测中统计"命中了多少条证据"，
        供 Phase 5.3 的 ``unknown`` 原因细化使用（``no_evidence`` 与
        ``unsupported_scene`` 的区分点）。

        Args:
            frame: 待判定的帧。
            profile: 是否采集 ``detect`` / ``classify`` 耗时分解（Phase 5.1；
                默认 ``False`` = 不额外计时）。

        Returns:
            ``(状态结果, detect_ms, state_ms)``；判定失败时为 ``unknown``
            （``reason`` / ``detail`` 含失败原因），两个耗时均为 ``0.0``。
            本轮命中的证据条数见 :attr:`last_evidence_hits`。
        """
        detect_ms = 0.0
        state_ms = 0.0
        try:
            if self._processor is not None:
                started = time.perf_counter()
                # Phase 11.8：``profile`` 同时驱动逐 detector 耗时采集（供 yard 观测使用）。
                # 仅在**需要画像时**才传该关键字参数 —— 保持 ``detect(frame)`` 这一既有契约
                # 对鸭子类型的 processor 实现（例如测试替身）依然成立。
                evidence = (
                    self._processor.detect(frame, profile=True)
                    if profile
                    else self._processor.detect(frame)
                )
                if profile:
                    detect_ms = (time.perf_counter() - started) * 1000.0
                    started = time.perf_counter()
                result = self._processor.classifier.classify(evidence)
                if profile:
                    state_ms = (time.perf_counter() - started) * 1000.0
                self._last_evidence_hits = sum(1 for item in evidence if item.detected)
                # Phase 7.3：保留本轮证据（同一次检测的产物，供视觉快照桥消费）
                self._last_evidences = list(evidence)
                return result, detect_ms, state_ms
            assert self._classifier is not None
            result = self._classifier.classify(frame)
            self._last_evidence_hits = None  # 帧路径不提供证据计数
            self._last_evidences = None  # 帧路径没有证据结果
            return result, detect_ms, state_ms
        except OnmyojiVisionError as exc:
            self._errors += 1
            self._last_evidence_hits = None
            self._last_evidences = None  # 判定失败：本轮证据不可信
            logger.error("状态判定失败（第 %d 次），本帧降级为 unknown: %s", self._errors, exc)
            return (
                StateResult(
                    state_name=StateName.UNKNOWN,
                    confidence=MIN_CONFIDENCE,
                    source=StateSource.FALLBACK,
                    detail=f"state resolution failed: {exc}",
                    reason=f"evidence pipeline failure: {exc}",
                ),
                detect_ms,
                state_ms,
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
        skip_reason: str | None = None,
        diagnosis: CycleDiagnosis | None = None,
    ) -> None:
        """记录本轮运行指标（Phase 5.2；未启用 ``metrics`` 时是空操作）。

        只记录"运行"本身：轮次、时间戳、原始/稳定状态、耗时与跳过原因；
        **不记录截图，也不记录任何用户数据**。跳过原因来自实现了
        :class:`~src.runtime.state_monitor.source.StatusAwareSource` 的帧来源
        （例如 ``WindowFrameSource``：窗口不存在 / 最小化 / 不在前台 / 采集失败）。

        Phase 5.3 追加诊断字段（:class:`~src.runtime.state_monitor.runtime_diagnostics.CycleDiagnosis`）：
        ``capture_status`` / ``foreground_status`` / ``unknown_reason``；既有字段含义与格式不变。

        Args:
            schedule: 本轮节拍结果（提供耗时与超时量）。
            timestamp: 本轮时间戳（与历史记录使用同一个值）。
            frame_ms: 取帧耗时。
            frame: 本轮帧；未取到时为 ``None``。
            result: 本轮原始判定结果；未取到帧时为 ``None``。
            failed: 本轮判定是否失败（降级为 ``unknown``）。
            skip_reason: 帧来源报告的跳过原因（已由调用方解析）。
            diagnosis: 本轮诊断三元组（采集 / 前台 / ``unknown`` 原因）。
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
                capture_status=diagnosis.capture_status if diagnosis is not None else None,
                foreground_status=diagnosis.foreground_status if diagnosis is not None else None,
                unknown_reason=diagnosis.unknown_reason if diagnosis is not None else None,
                runtime_state=self._lifecycle.state.value,
                recovery_count=self._recovery.count,
                last_recovery_reason=self._recovery.last_reason,
                # Phase 11.8：yard 运行期观测（未开启时为空 dict ⇒ 字段保持 None）
                **self._yard_observation(),
            ),
            overrun_ms=schedule.overrun_ms,
        )

    def _yard_observation(self) -> dict[str, Any]:
        """采集本轮 yard 运行期观测（Phase 11.8；未开启时返回空字典）。

        数据**全部来自 Runtime 已持有的对象**，不跨层 import，也不访问 detector 的私有属性：

        * yard 单项耗时 → ``processor.registry.last_profile``（``profile=True`` 的既有产物）；
        * 预筛候选 / 回退 → yard 证据的 ``metadata``（vision 层在启用预筛时写入）；
        * 快照状态 / 累计失效 → ``vision_bridge.stats``；
        * 活动任务 → ``task_bridge.active_task_id``；
        * 安全确认 → 装配期记录的 action 配置。

        取不到的项一律留空（``None``），不猜测、不伪造。
        """
        if not self._yard_metrics:
            return {}
        observation: dict[str, Any] = {
            "frame_id": self._cycles,
            "vision_ms": self._last_detect_ms,
            "action_enabled": self._action_enabled,
            "action_backend": self._action_backend,
        }

        profile = self._profile_of(YARD_EVIDENCE_NAME)
        if profile is not None:
            observation["yard_ms"] = profile.elapsed_ms
            observation["yard_total_templates"] = profile.templates or None
        evidence = self._evidence_of(YARD_EVIDENCE_NAME)
        if evidence is not None:
            observation["yard_detected"] = bool(getattr(evidence, "detected", False))
            observation["yard_confidence"] = float(getattr(evidence, "confidence", 0.0) or 0.0)
            metadata = dict(getattr(evidence, "metadata", None) or {})
            observation["yard_template_id"] = metadata.get("template_id")
            if metadata.get("template_prefilter") is not None:
                observation["yard_prefilter"] = metadata.get("template_prefilter")
                observation["yard_top_k"] = metadata.get("prefilter_top_k")
                observation["yard_candidates"] = metadata.get("prefilter_candidate_count")
                observation["yard_prefilter_ms"] = metadata.get("prefilter_ms")
                observation["yard_fallback"] = bool(metadata.get("prefilter_fallback", False))
                observation["yard_fallback_reason"] = metadata.get("prefilter_fallback_reason")
                total = metadata.get("prefilter_total_templates")
                if total is not None:
                    observation["yard_total_templates"] = total
                candidates = observation.get("yard_candidates")
                scales = len(profile.scales) if profile is not None else 0
                if isinstance(candidates, int) and scales:
                    observation["yard_match_calls"] = candidates * scales

        bridge = self._vision_bridge
        stats = getattr(bridge, "stats", None) if bridge is not None else None
        if stats is not None:
            observation["snapshot_status"] = getattr(stats, "last_reason", "") or None
            observation["snapshot_invalidated"] = getattr(stats, "invalidated", None)

        task_bridge = self._task_bridge
        if task_bridge is not None:
            observation["task_active"] = getattr(task_bridge, "active_task_id", None)
        return observation

    def _evidence_of(self, name: str) -> Any | None:
        """本轮证据里按名查找（只读；未采集时为 ``None``）。"""
        for item in self._last_evidences or ():
            if getattr(item, "name", None) == name:
                return item
        return None

    def _profile_of(self, name: str) -> Any | None:
        """最近一次 evidence profile 里按名查找（仅在采集过 ``profile`` 时有值）。"""
        registry = self._processor.registry if self._processor is not None else None
        if registry is None:
            return None
        for item in getattr(registry, "last_profile", None) or ():
            if getattr(item, "name", None) == name:
                return item
        return None

    def _source_skip_reason(self) -> str | None:
        """本轮跳过原因（来自实现了 :class:`StatusAwareSource` 的帧来源）。"""
        if isinstance(self._source, StatusAwareSource):
            return self._source.last_skip_reason
        return None

    def _source_skip_message(self) -> str | None:
        """本轮跳过的可读说明（帧来源提供 ``status.message`` 时可用）。"""
        status = getattr(self._source, "status", None)
        message = getattr(status, "message", None)
        return message if isinstance(message, str) and message else None

    def _foreground_flags(self) -> tuple[bool, bool]:
        """返回 ``(是否检查过前台, 是否通过)``（帧来源支持前台门控时为实际值）。

        只读取帧来源暴露的属性（鸭子类型），因此对 ``CaptureFrameSource`` 等
        不具备门控信息的来源返回 ``(False, False)``——诊断层会把它记为 ``n/a``。
        """
        guard = getattr(self._source, "guard", None)
        checked = bool(getattr(guard, "enabled", False))
        status = getattr(self._source, "status", None)
        passed = getattr(status, "foreground", None) is True
        return checked, passed

    # ---- 生命周期（Phase 5.4） ----------------------------------------------

    def initialize(self) -> RuntimeState:
        """初始化运行时：重置统计、启动帧来源，状态 ``INITIALIZING``/``STOPPED`` → ``READY``。

        幂等；``step()`` / ``run()`` 的行为不受影响（``run()`` 自己也会重置统计）。

        Returns:
            转换后的运行时状态。
        """
        self._reset_runtime()
        state = self._lifecycle.initialize()
        self._start_source()
        logger.info("运行时已初始化：%s", self._lifecycle.describe())
        return state

    def start(self) -> RuntimeState:
        """启动运行：状态 → ``RUNNING``（幂等，不阻塞）。

        循环本身仍由 :meth:`run` 驱动；本方法用于显式表达"开始运行"并把帧来源投入工作
        （``FrameSource`` 提供 ``start()`` 时调用）。

        Returns:
            转换后的运行时状态。
        """
        if self._lifecycle.state in (
            RuntimeState.INITIALIZING,
            RuntimeState.STOPPED,
            RuntimeState.FAILED,
        ):
            self._reset_runtime()
            self._lifecycle.initialize(reason="auto-initialize")
        state = self._lifecycle.start()
        self._stop_requested = False
        self._start_source()
        logger.info("运行时已启动：%s", self._lifecycle.describe())
        return state

    def pause(self) -> RuntimeState:
        """暂停：停止**处理新帧**（节拍与 ``duration`` 计时继续推进）。

        Returns:
            转换后的运行时状态。

        Raises:
            InvalidRuntimeTransition: 当前状态不是 ``RUNNING``。
        """
        state = self._lifecycle.pause()
        logger.info("运行时已暂停（不再处理新帧）：%s", self._lifecycle.describe())
        return state

    def resume(self) -> RuntimeState:
        """恢复：从 ``PAUSED`` 回到 ``RUNNING``（幂等）。

        Returns:
            转换后的运行时状态。
        """
        state = self._lifecycle.resume()
        logger.info("运行时已恢复：%s", self._lifecycle.describe())
        return state

    def stop(self) -> RuntimeState:
        """停止：请求循环退出并**释放采集资源**（幂等、异常安全）。

        * 设置停止请求：``run()`` 会在下一轮检测到并结束（``StopReason.STOP_REQUESTED``）；
        * 立即释放帧来源（``FrameSource`` 提供 ``stop()`` 时调用）→ dxcam 会话与内部线程结束、
          mss 会话关闭；
        * 状态 → ``STOPPED``（可用 :meth:`initialize` / :meth:`start` 重新开始）。

        Returns:
            转换后的运行时状态。
        """
        self._stop_requested = True
        state = self._lifecycle.stop()
        self._stop_source()
        logger.info("运行时已停止（资源已释放）：%s", self._lifecycle.describe())
        return state

    @property
    def lifecycle_transitions(self) -> tuple[LifecycleTransition, ...]:
        """生命周期转换历史（便于诊断 / 测试）。"""
        return self._lifecycle.transitions

    def _reset_runtime(self) -> None:
        """重置一次运行的全部累积状态（事件、计数、稳定化、诊断、恢复）。"""
        self._events.clear()
        self._current_state = self._last_result = self._last_stable = None
        self._cycles = self._classified = self._skipped = 0
        self._errors = 0
        self._paused_cycles = 0
        self._last_evidence_hits = None
        self._last_evidences = None
        self._stabilizer.reset()
        self._diagnostics.reset()
        self._recovery.reset()

    def _publish_vision_snapshot(
        self, *, frame: Frame | None, schedule: CycleSchedule, timestamp: str
    ) -> None:
        """把本轮的视觉结果推送给 Action 侧的快照桥（Phase 7.3）。

        设计要点：

        * **未接线即空操作**（``vision_bridge=None`` 时立即返回，零开销、零行为变化）；
        * **同一次检测**：使用 :attr:`last_evidences`（``_classify`` 已检测好的证据），
          本层不重复调用 ``detect()``；
        * **帧号来自 Runtime**：``frame_id = self._cycles``（本轮的轮次序号；capture 失败轮
          不推送，因此帧号只增不减），``cycle_id`` 取自 ``CycleSchedule.cycle_id`` 以便追踪；
        * **异常隔离**：桥或 Action 侧抛出的任何异常都在此被捕获并记录，
          绝不影响监控主循环（后续帧照常处理）。
        """
        bridge = self._vision_bridge
        if bridge is None:
            return
        try:
            bridge.update(
                frame=frame,
                evidences=self._last_evidences or (),
                frame_id=self._cycles,
                cycle_id=schedule.cycle_id,
                timestamp=frame.timestamp if frame is not None else None,
            )
        except Exception as exc:  # noqa: BLE001 - 桥的异常绝不影响主循环
            logger.error("推送视觉快照失败（已隔离，不影响监控循环）: %s", exc)

    def _invalidate_vision_snapshot(self, reason: str) -> None:
        """**主动失效**视觉快照（Phase 11.1）。

        用于"画面结论不再可信"的场景（暂停 / Runtime 停止）——此前只有 capture 失败
        （``frame=None``）会触发失效，暂停与结束路径会留下**仍然有效**的旧快照。

        Args:
            reason: 失效原因（见 :mod:`src.runtime.vision_bridge` 的 ``INVALIDATION_*``）。
        """
        bridge = self._vision_bridge
        if bridge is None:
            return
        invalidate = getattr(bridge, "invalidate", None)
        if not callable(invalidate):
            return
        try:
            invalidate(reason=reason, frame_id=self._cycles)
        except TypeError:
            # 兼容仅有 ``invalidate(reason)`` 的桥实现
            try:
                invalidate(reason)
            except Exception as exc:  # noqa: BLE001 - 绝不影响主循环
                logger.error("失效视觉快照失败（已隔离）: %s", exc)
        except Exception as exc:  # noqa: BLE001 - 桥的异常绝不影响主循环
            logger.error("失效视觉快照失败（已隔离）: %s", exc)

    def _notify_task_bridge(
        self, *, stable: StableStateResult | None, schedule: CycleSchedule
    ) -> None:
        """每轮末尾驱动任务编排桥（Phase 7.5）。

        顺序与语义：

        1. **在快照推送之后调用**（``run()`` 内），因此任务决策总能使用**本轮**的视觉快照；
        2. 仅当本轮**确实处理了一帧**（``frame is not None``）时才上报稳定状态——
           跳过轮没有新的状态结论，不应重复触发任务（但节拍照常上报，用于计时 / 超时）；
        3. **未接线即空操作**（``task_bridge=None``）；
        4. **异常隔离**：桥或任务侧的异常都在此捕获，绝不影响监控主循环。
        """
        bridge = self._task_bridge
        if bridge is None:
            return
        # 两个通知各自独立隔离：一个失败不影响另一个，也绝不影响监控主循环
        if stable is not None:
            try:
                bridge.on_state_stable(stable)
            except Exception as exc:  # noqa: BLE001 - 任务编排异常绝不影响主循环
                logger.error("任务编排处理稳定状态失败（已隔离）: %s", exc)
        try:
            bridge.on_cycle(schedule)
        except Exception as exc:  # noqa: BLE001
            logger.error("任务编排处理节拍失败（已隔离）: %s", exc)

    def _start_source(self) -> None:
        """启动帧来源（若其提供 ``start()``；幂等，异常不致命）。"""
        starter = getattr(self._source, "start", None)
        if not callable(starter):
            return
        try:
            starter()
        except Exception as exc:  # noqa: BLE001 - 启动失败由采集层按"跳过"处理
            logger.warning("帧来源启动失败（将继续按轮次重试）：%s", exc)

    def _stop_source(self) -> None:
        """停止帧来源并释放资源（若其提供 ``stop()``；异常安全）。"""
        stopper = getattr(self._source, "stop", None)
        if not callable(stopper):
            return
        try:
            stopper()
        except Exception as exc:  # noqa: BLE001 - 清理路径不应影响停止流程
            logger.warning("帧来源停止异常（已忽略）：%s", exc)

    def _handle_recovery(self, *, frame_ok: bool, reason: str | None) -> bool:
        """按恢复策略处理本轮结果。

        Args:
            frame_ok: 本轮是否取到帧。
            reason: 失败原因（``SkipReason`` 值或 ``judge_failure``）。

        Returns:
            ``True`` 表示重试已用尽（调用方应结束本次运行）。
        """
        if not self._recovery.enabled:
            return False
        if frame_ok:
            if self._recovery.register_success():
                self._lifecycle.finish_recovery()
            return False

        decision: RecoveryDecision = self._recovery.register_failure(reason)
        if decision.action is RecoveryAction.RETRY:
            if self._lifecycle.is_running:
                self._lifecycle.begin_recovery(reason or "capture_failure")
            return False
        if decision.action is RecoveryAction.EXHAUSTED:
            if not self._lifecycle.is_terminal:
                self._lifecycle.fail(reason=reason or "recovery exhausted")
            self._stop_source()
            logger.error(
                "恢复重试用尽，结束本次运行（连续失败 %d 次，原因=%s）",
                self._recovery.attempts,
                reason or "unknown",
            )
            return True
        return False

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
        self._reset_runtime()
        # 生命周期同步（Phase 5.4）：run() 始终可运行，因此必要时自动初始化并进入 RUNNING
        if self._lifecycle.state in (
            RuntimeState.INITIALIZING,
            RuntimeState.STOPPED,
            RuntimeState.FAILED,
        ):
            self._lifecycle.initialize(reason="auto-initialize")
        if not self._lifecycle.is_running:
            self._lifecycle.start(reason="run")
        self._stop_requested = False

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
                if self._stop_requested:
                    stop_reason = StopReason.STOP_REQUESTED
                    break
                if max_cycles is not None and self._cycles >= max_cycles:
                    stop_reason = StopReason.MAX_CYCLES
                    break
                if duration_s is not None and (self._clock() - start_clock) >= duration_s:
                    stop_reason = StopReason.DURATION
                    break
                if self._lifecycle.is_paused:
                    # 暂停：保持节拍，但**不处理新帧**（不计轮次、不判定、不写指标）
                    paused_started = self._scheduler.begin()
                    self._scheduler.finish(paused_started)
                    if self._paused_cycles == 0:
                        # Phase 11.1：进入暂停的**首个**轮次立即失效视觉快照——
                        # 暂停期间画面不可信（不再更新），旧目标不应继续可用。
                        self._invalidate_vision_snapshot(INVALIDATION_WINDOW_UNAVAILABLE)
                    logger.debug("已暂停，本轮不处理新帧（累计暂停轮次 %d）", self._paused_cycles + 1)
                    self._paused_cycles += 1
                    continue

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
                failed = self._errors > errors_before
                skip_reason = self._source_skip_reason()
                foreground_checked, foreground_passed = self._foreground_flags()
                diagnosis = self._diagnostics.observe(
                    frame_ok=frame is not None,
                    result=result,
                    judge_failed=failed,
                    skip_reason=skip_reason,
                    skip_message=self._source_skip_message(),
                    foreground_checked=foreground_checked,
                    foreground_passed=foreground_passed,
                    latency_ms=schedule.elapsed_ms,
                    backend=frame.source if frame is not None else None,
                    evidence_hits=self._last_evidence_hits if frame is not None else None,
                )
                # 失败恢复（Phase 5.4）：仅当策略启用时生效；退避发生在耗时统计之后，
                # 因此不会污染本轮 latency，但本轮指标能反映恢复后的状态
                exhausted = self._handle_recovery(
                    frame_ok=frame is not None,
                    reason=skip_reason or ("judge_failure" if failed else None),
                )
                self._record_metrics(
                    schedule=schedule,
                    timestamp=timestamp,
                    frame_ms=frame_ms,
                    frame=frame,
                    result=result,
                    failed=failed,
                    skip_reason=skip_reason,
                    diagnosis=diagnosis,
                )
                if self._on_cycle is not None:
                    self._on_cycle(schedule)
                # Phase 7.3：把本轮的视觉结果推送给 Action 侧（未接线时是空操作）
                self._publish_vision_snapshot(
                    frame=frame, schedule=schedule, timestamp=timestamp
                )
                # Phase 7.5：驱动任务编排（在快照推送**之后**，任务能看到本轮最新快照）
                self._notify_task_bridge(
                    stable=self._last_stable if frame is not None else None,
                    schedule=schedule,
                )
                if exhausted:
                    stop_reason = StopReason.RECOVERY_EXHAUSTED
                    break
        except KeyboardInterrupt:
            stop_reason = StopReason.INTERRUPT
            logger.warning("收到中断信号（Ctrl+C），正在收尾并保存历史")

        # Phase 11.1：本轮监控结束 ⇒ 快照失效（停止后画面结论不再可信）
        self._invalidate_vision_snapshot(INVALIDATION_RUNTIME_STOPPED)
        ended_at = self._now()
        self._history.update_progress(cycles=self._cycles, skipped=self._skipped, timestamp=ended_at)
        self._history.save()
        if self._lifecycle.is_running or self._lifecycle.is_recovering:
            # 一次运行结束（未显式 stop）：回到 READY，可再次 run()
            self._lifecycle.transition(RuntimeState.READY, reason="run finished")
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
