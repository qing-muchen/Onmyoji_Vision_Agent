"""实时状态监控子包（Phase 3.1）。

把"单帧状态判定"扩展为**可持续运行的监控过程**。判定入口自 Phase 5 起默认是
**证据流水线**（:class:`~src.vision.pipeline.frame_processor.EvidenceFrameProcessor`），
也兼容旧的状态分类器：

```text
每 interval_ms:
  FrameSource.get_frame()
        │  帧不可用（后端暂无画面 / 前台门控未通过）→ 跳过该轮
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
```

对外入口：

* :class:`~src.runtime.state_monitor.monitor.StateMonitor` —— 固定节拍循环 + 变化检测
  （可注入 ``processor`` 或 ``classifier``）；
* :class:`~src.runtime.state_monitor.monitor.MonitorReport` —— 一次运行的摘要；
* :class:`~src.runtime.state_monitor.events.StateEvent` —— 状态变化事件；
* :class:`~src.runtime.state_monitor.history.StateHistoryStore` —— 历史记录（JSON）；
* :class:`~src.runtime.state_monitor.source.FrameSource` /
  :class:`~src.runtime.state_monitor.source.CaptureFrameSource` —— 帧来源抽象与采集实现。

依赖方向：``core`` → ``capture`` → ``scene`` / ``evidence`` / ``state`` / ``pipeline`` →
``runtime``（不得反向）。本层不做视觉判定（detect / aggregate / OCR / 模板匹配）、
不做状态转移、不做决策、不触发任何交互。
"""

from src.runtime.state_monitor.events import StateEvent
from src.runtime.state_monitor.history import StateHistoryStore
from src.runtime.state_monitor.monitor import StateMonitor
from src.runtime.state_monitor.report import MonitorReport, StopReason, local_now_iso
from src.runtime.state_monitor.source import (
    CaptureFrameSource,
    ForegroundGate,
    FrameSource,
    make_foreground_gate,
)
from src.runtime.state_monitor.stabilizer import StableStateResult, StateStabilizer

__all__ = [
    "CaptureFrameSource",
    "ForegroundGate",
    "FrameSource",
    "MonitorReport",
    "StableStateResult",
    "StateEvent",
    "StateHistoryStore",
    "StateMonitor",
    "StateStabilizer",
    "StopReason",
    "local_now_iso",
    "make_foreground_gate",
]
