"""输入控制层（Action Layer，Phase 7）：把 Task 的**语义动作请求**变成受控的真实键鼠输入。

```text
TaskEngine ──▶ ActionRequest ──▶ ActionDispatcher ──▶ TargetResolver ──▶ InputBackend ──▶ Mouse/Keyboard
                （语义目标）      （5 道安全门）        （语义 → 比例）      （SendInput）
```

```text
action/
├── models.py       ActionRequest / ActionOutcome / ActionStatus / ActionType / ActionTarget
│                   + SemanticTarget / ResolvedTarget（**禁止像素坐标**）
├── port.py         ActionPort 协议（与 Task 层字段兼容）+ NullActionPort
├── resolver.py     TargetResolver 协议 + RegistryTargetResolver（不读 Frame、不调 detector）
├── registry.py     TargetRegistry（语义名 → 窗口相对比例；占位值 calibrated=False）
├── dispatcher.py   ActionDispatcher（门控 → 解析 → 换算 → 执行 → 结果）
├── backend/        InputBackend 协议 / DryRunBackend（默认）/ Win32 SendInput 后端
└── safety/         EmergencyStop（API + logs/ESTOP）/ SafetyGuard（5 道门）/ RateLimiter
```

**本阶段边界（Phase 7 第一阶段）**：

* 实现"数据模型 / 端口 / 语义目标解析 / Windows 输入后端 / 安全门控 / 分发流程"；
* **不做**游戏按钮识别（模板 / OCR / YOLO）、不做任务逻辑、不做自动刷任务；
* 默认后端是 :class:`~src.action.backend.base.DryRunBackend`（**不产生任何输入**）；
  真实输入需要 ``configs/config.yaml`` 的 ``action.enabled=true`` + ``backend: windows``；
* 未修改 ``runtime`` / ``vision`` / ``evidence`` / ``capture`` / ``window`` / ``task``。

依赖边界（AST 测试固化）：只 import 标准库与 ``src.core`` / ``src.utils`` / ``src.action``；
**禁止** import ``src.vision`` / ``src.evidence`` / ``src.task`` / ``src.runtime`` /
``cv2`` / ``pyautogui`` / ``pynput``。
"""

from src.action.backend import (
    MOUSE_BUTTONS,
    BackendCall,
    DryRunBackend,
    InputBackend,
    NullWindowProbe,
    WindowProbe,
    create_input_backend,
)
from src.action.dispatcher import ActionDispatcher, DispatchStats
from src.action.models import (
    FORBIDDEN_PARAMETER_KEYS,
    ActionError,
    ActionOutcome,
    ActionRequest,
    ActionRequestError,
    ActionResolutionError,
    ActionStatus,
    ActionTarget,
    ActionTargetError,
    ActionType,
    InputBackendError,
    InputBackendUnavailableError,
    ResolvedTarget,
    SemanticTarget,
    now_iso,
    outcome_from_any,
    validate_target_name,
)
from src.action.port import ActionPort, NullActionPort, as_action_port, is_action_request
from src.action.registry import (
    BUILTIN_TARGETS,
    TargetEntry,
    TargetRegistry,
    registry_from_mapping,
)
from src.action.resolver import (
    DEFAULT_VISION_TARGET_MIN_CONFIDENCE,
    DEFAULT_VISION_TARGET_TTL_S,
    EXPLICIT_RATIO_TARGET,
    VISION_TARGET_SOURCE,
    RegistryTargetResolver,
    ResolverWithReason,
    TargetResolution,
    TargetResolutionReason,
    TargetResolver,
    VisionAwareTargetResolver,
)
from src.action.safety import (
    DEFAULT_ACTION_SENTINEL_PATH,
    DEFAULT_MAX_ACTIONS_PER_MINUTE,
    DEFAULT_MIN_INTERVAL_S,
    EmergencyStop,
    EmergencyStopSource,
    EmergencyStopState,
    GuardResult,
    GuardStage,
    RateLimiter,
    SafetyGuard,
    build_guard,
)

__all__ = [
    "BUILTIN_TARGETS",
    "DEFAULT_ACTION_SENTINEL_PATH",
    "DEFAULT_MAX_ACTIONS_PER_MINUTE",
    "DEFAULT_MIN_INTERVAL_S",
    "DEFAULT_VISION_TARGET_MIN_CONFIDENCE",
    "DEFAULT_VISION_TARGET_TTL_S",
    "EXPLICIT_RATIO_TARGET",
    "FORBIDDEN_PARAMETER_KEYS",
    "MOUSE_BUTTONS",
    "VISION_TARGET_SOURCE",
    "ActionDispatcher",
    "ActionError",
    "ActionOutcome",
    "ActionPort",
    "ActionRequest",
    "ActionRequestError",
    "ActionResolutionError",
    "ActionStatus",
    "ActionTarget",
    "ActionTargetError",
    "ActionType",
    "BackendCall",
    "DispatchStats",
    "DryRunBackend",
    "EmergencyStop",
    "EmergencyStopSource",
    "EmergencyStopState",
    "GuardResult",
    "GuardStage",
    "InputBackend",
    "InputBackendError",
    "InputBackendUnavailableError",
    "NullActionPort",
    "NullWindowProbe",
    "RateLimiter",
    "RegistryTargetResolver",
    "ResolvedTarget",
    "ResolverWithReason",
    "SafetyGuard",
    "SemanticTarget",
    "TargetEntry",
    "TargetRegistry",
    "TargetResolution",
    "TargetResolutionReason",
    "TargetResolver",
    "VisionAwareTargetResolver",
    "WindowProbe",
    "as_action_port",
    "build_guard",
    "create_input_backend",
    "is_action_request",
    "now_iso",
    "outcome_from_any",
    "registry_from_mapping",
    "validate_target_name",
]
