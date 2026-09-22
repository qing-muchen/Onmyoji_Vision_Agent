"""实时状态监控子包（Phase 3.1）。

把 Phase 3.0 的单帧 :class:`~src.vision.state.classifier.StateClassifier` 扩展为
**可持续运行的监控过程**：

```text
每 interval_ms:
  FrameSource.get_frame()
        │  帧不可用（后端暂无画面 / 前台门控未通过）→ 跳过该轮
        ▼
  StateClassifier.classify(frame)      # 复用 Phase 3.0，未改动
        │  StateResult
        ▼
  状态变化检测（只有 state_name 改变才产生事件）
        │  StateEvent
        ▼
  StateHistoryStore（即时原子写入 state_history.json）
```

对外入口：

* :class:`~src.runtime.state_monitor.monitor.StateMonitor` —— 固定节拍循环 + 变化检测；
* :class:`~src.runtime.state_monitor.monitor.MonitorReport` —— 一次运行的摘要；
* :class:`~src.runtime.state_monitor.events.StateEvent` —— 状态变化事件；
* :class:`~src.runtime.state_monitor.history.StateHistoryStore` —— 历史记录（JSON）；
* :class:`~src.runtime.state_monitor.source.FrameSource` /
  :class:`~src.runtime.state_monitor.source.CaptureFrameSource` —— 帧来源抽象与采集实现。

依赖方向：``core`` → ``capture`` → ``scene`` / ``state`` → ``runtime``（不得反向）。
本层不做状态转移、不做决策、不触发任何交互。
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

__all__ = [
    "CaptureFrameSource",
    "ForegroundGate",
    "FrameSource",
    "MonitorReport",
    "StateEvent",
    "StateHistoryStore",
    "StateMonitor",
    "StopReason",
    "local_now_iso",
    "make_foreground_gate",
]
