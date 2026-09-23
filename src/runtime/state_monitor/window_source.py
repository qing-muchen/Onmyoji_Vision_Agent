"""兼容层：真实窗口帧来源的旧 import 路径（Phase 5.3 / 5.4 拆分后保留）。

Phase 5.3 把原本 670 行、职责过多的 ``window_source.py`` 拆成四个单一职责模块，
Phase 5.4 又把采集细节拆成三个：

| 模块 | 职责 |
| --- | --- |
| :mod:`~src.runtime.state_monitor.window_binding` | 目标窗口查找 / 绑定 / 几何刷新（``exact`` / ``contains`` / ``regex``） |
| :mod:`~src.runtime.state_monitor.foreground_guard` | 前台门控（只检查、不抢占）+ ``SkipReason`` / ``GuardVerdict`` |
| :mod:`~src.runtime.state_monitor.window_capture` | hwnd 校验 / 采集区域 / 几何刷新 / 缩放检测 / 查找冷却（Phase 5.4） |
| :mod:`~src.runtime.state_monitor.capture_backend` | dxcam / mss 后端创建、复用、重建与释放（Phase 5.4） |
| :mod:`~src.runtime.state_monitor.frame_source` | 编排 + 重试 + 状态：``WindowFrameSource`` / ``CaptureStatus`` |

本模块**不删除、不改行为**，只把上述符号重新导出，因此：

```python
from src.runtime.state_monitor.window_source import SkipReason, WindowFrameSource  # 旧路径仍可用
from src.runtime.state_monitor.frame_source import WindowFrameSource              # 新路径（推荐）
```

两者是**同一个对象**（``is`` 判定为真），因此 isinstance / 类型注解 / 异常处理完全一致。

> 注意：若要替换采集后端工厂（测试替身），请 patch **实现所在的模块**：
> ``src.runtime.state_monitor.capture_backend.create_capture_backend``
> （patch 本兼容层上的同名属性不会影响 ``WindowFrameSource`` 的行为）。
"""

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
from src.runtime.state_monitor.window_binding import OnmyojiWindowSource

__all__ = [
    "CaptureStatus",
    "DEFAULT_RETRY_INTERVAL_S",
    "ForegroundGuard",
    "GuardVerdict",
    "OnmyojiWindowSource",
    "SkipReason",
    "WindowFrameSource",
]
