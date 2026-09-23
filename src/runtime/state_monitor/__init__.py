"""实时状态监控子包（Phase 3.1 / 5.2）。

把"单帧状态判定"扩展为**可持续运行**的监控过程。判定入口自 Phase 5 起默认是
**证据流水线**（:class:`~src.vision.pipeline.frame_processor.EvidenceFrameProcessor`），
也兼容旧的状态分类器：

```text
每 interval_ms（CycleScheduler：deadline based，late cycle 有记录）:
  FrameSource.get_frame()
        │  帧不可用（窗口不存在/最小化/不在前台/采集失败）→ 跳过该轮并记录原因
        ▼
  判定（二选一，processor 优先）
    · EvidenceFrameProcessor.process(frame)   # Phase 5 推荐：Frame → 证据 → StateResult
    · StateClassifier.classify(frame)         # Phase 3.0 兼容路径
        │  StateResult（判定失败 → unknown + 失败原因，不崩溃）
        ▼
  StateStabilizer.update(result)               # 唯一时间层：逐帧 → 稳定状态
        │  只有稳定状态改变才产生事件；首次确认只建立基线
        ▼
  StateEvent → StateHistoryStore（即时原子写入 state_history.json）
        │
  CycleMetrics → RuntimeMetricsRecorder（可选 JSONL：runtime_metrics.jsonl，Phase 5.2）
```

对外入口：

* :class:`~src.runtime.state_monitor.monitor.StateMonitor` —— 固定节拍循环 + 变化检测
  （可注入 ``processor`` / ``classifier`` / ``scheduler`` / ``metrics``）；
* :class:`~src.runtime.state_monitor.scheduler.CycleScheduler` —— deadline based 节拍
  （late cycle / overrun 统计）；
* :class:`~src.runtime.state_monitor.metrics.RuntimeMetricsRecorder` —— 逐轮运行指标
  （cycle_id / timestamp / state / stable_state / latency，**不含截图与用户数据**）；
* :class:`~src.runtime.state_monitor.window_source.WindowFrameSource` /
  :class:`~src.runtime.state_monitor.frame_source.WindowFrameSource` /
  :class:`~src.runtime.state_monitor.window_binding.OnmyojiWindowSource` /
  :class:`~src.runtime.state_monitor.foreground_guard.ForegroundGuard` —— 真实窗口帧来源
  （dxcam 优先、mss 回退；前台门控只检查不抢占；窗口/采集异常转为"跳过"而非崩溃）；
  Phase 5.3 拆分后三个模块各自单一职责，``window_source`` 仅作兼容层（旧 import 路径仍可用）；
* :class:`~src.runtime.state_monitor.runtime_diagnostics.RuntimeHealth` /
  :class:`~src.runtime.state_monitor.runtime_diagnostics.UnknownReason` —— 运行时诊断：
  区分 **runtime failure** 与 **vision unknown**（Phase 5.3）；
* :class:`~src.runtime.state_monitor.monitor.MonitorReport` —— 一次运行的摘要；
* :class:`~src.runtime.state_monitor.events.StateEvent` —— 状态变化事件；
* :class:`~src.runtime.state_monitor.history.StateHistoryStore` —— 历史记录（JSON）；
* :class:`~src.runtime.state_monitor.source.FrameSource` /
  :class:`~src.runtime.state_monitor.source.CaptureFrameSource` —— 帧来源抽象与整屏采集实现。

依赖方向：``core`` → ``capture`` → ``scene`` / ``evidence`` / ``state`` / ``pipeline`` →
``runtime``（不得反向）。本层不做视觉判定（detect / aggregate / OCR / 模板匹配）、
不做状态转移、不做决策、**不触发任何交互**（不点击、不键入、不抢焦点、不切换窗口）。
"""

from src.runtime.state_monitor.events import StateEvent
from src.runtime.state_monitor.foreground_guard import (
    ForegroundGuard,
    GuardVerdict,
    SkipReason,
)
from src.runtime.state_monitor.frame_source import (
    DEFAULT_RETRY_INTERVAL_S,
    CaptureStatus,
    WindowFrameSource,
)
from src.runtime.state_monitor.history import StateHistoryStore
from src.runtime.state_monitor.metrics import (
    CycleMetrics,
    CycleStatus,
    RuntimeMetricsRecorder,
    RuntimeMetricsSummary,
)
from src.runtime.state_monitor.monitor import StateMonitor
from src.runtime.state_monitor.report import MonitorReport, StopReason, local_now_iso
from src.runtime.state_monitor.runtime_diagnostics import (
    CaptureOutcome,
    CycleDiagnosis,
    ForegroundOutcome,
    RuntimeDiagnostics,
    RuntimeHealth,
    UnknownReason,
    resolve_unknown_reason,
)
from src.runtime.state_monitor.scheduler import CycleSchedule, CycleScheduler
from src.runtime.state_monitor.source import (
    CaptureFrameSource,
    ForegroundGate,
    FrameSource,
    StatusAwareSource,
    make_foreground_gate,
)
from src.runtime.state_monitor.stabilizer import StableStateResult, StateStabilizer
from src.runtime.state_monitor.timing import CycleTiming
from src.runtime.state_monitor.window_binding import OnmyojiWindowSource

__all__ = [
    "CaptureFrameSource",
    "CaptureOutcome",
    "CaptureStatus",
    "CycleDiagnosis",
    "CycleMetrics",
    "CycleSchedule",
    "CycleScheduler",
    "CycleStatus",
    "CycleTiming",
    "DEFAULT_RETRY_INTERVAL_S",
    "ForegroundGate",
    "ForegroundGuard",
    "ForegroundOutcome",
    "FrameSource",
    "GuardVerdict",
    "MonitorReport",
    "OnmyojiWindowSource",
    "RuntimeDiagnostics",
    "RuntimeHealth",
    "RuntimeMetricsRecorder",
    "RuntimeMetricsSummary",
    "SkipReason",
    "StableStateResult",
    "StateEvent",
    "StateHistoryStore",
    "StateMonitor",
    "StateStabilizer",
    "StatusAwareSource",
    "StopReason",
    "UnknownReason",
    "WindowFrameSource",
    "local_now_iso",
    "make_foreground_gate",
    "resolve_unknown_reason",
]
