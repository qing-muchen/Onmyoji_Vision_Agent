"""安全子系统（Phase 7）：紧急停止与动作安全门。

```text
EmergencyStop（API + logs/ESTOP 哨兵，禁止热键）
SafetyGuard  （5 道门：E-STOP → 窗口 → 前台 → 目标已解析 → 频率限制）
RateLimiter  （最小间隔 + 每分钟上限）
```

任一检查失败 → 动作被 ``REJECTED``；**禁止 fallback click、禁止猜坐标**。
"""

from src.action.safety.estop import (
    DEFAULT_ACTION_SENTINEL_PATH,
    EmergencyStop,
    EmergencyStopSource,
    EmergencyStopState,
)
from src.action.safety.guard import (
    DEFAULT_MAX_ACTIONS_PER_MINUTE,
    DEFAULT_MIN_INTERVAL_S,
    GuardResult,
    GuardStage,
    RateLimiter,
    SafetyGuard,
    build_guard,
)

__all__ = [
    "DEFAULT_ACTION_SENTINEL_PATH",
    "DEFAULT_MAX_ACTIONS_PER_MINUTE",
    "DEFAULT_MIN_INTERVAL_S",
    "EmergencyStop",
    "EmergencyStopSource",
    "EmergencyStopState",
    "GuardResult",
    "GuardStage",
    "RateLimiter",
    "SafetyGuard",
    "build_guard",
]
