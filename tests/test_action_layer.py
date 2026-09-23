"""Phase 7 测试：Action Layer（数据模型 / 端口 / 目标解析 / 输入后端 / 安全门控 / 分发流程）。

覆盖用户要求的十项：

1. ``ActionRequest`` schema（必填字段、默认值、序列化）；
2. **禁止 pixel coordinate**（字段层 + 参数层 + 目标写法层）；
3. ``SemanticTarget`` 解析（注册表 / 显式比例 / 未注册）；
4. Resolver 不依赖 vision（AST 依赖边界）；
5. DryRun 执行（默认后端只记录、不产生输入）；
6. E-STOP 拒绝（API + 文件哨兵）；
7. Foreground / Window 拒绝；
8. Backend 异常隔离（转换为 FAILED，不抛出）；
9. Dispatcher 流程（门控顺序、限流、延迟统计、统计计数）；
10. Windows backend 接口（结构 / 键码 / 坐标归一化 / dry-run 安全）。

**安全承诺**：本文件**绝不产生真实键鼠输入**——所有用例使用
:class:`DryRunBackend`（默认）或 :class:`RecordingBackend`（不发送输入的真实感替身），
Windows 后端只用 ``dry_run=True`` 与纯函数（键码映射 / 坐标归一化）验证。

另含 **AST 依赖检查**：``src/action`` 禁止 import ``src.vision`` / ``src.evidence`` /
``src.task`` / ``src.runtime`` / ``cv2`` / ``pyautogui`` / ``pynput``。
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.action import (
    BUILTIN_TARGETS,
    FORBIDDEN_PARAMETER_KEYS,
    ActionDispatcher,
    ActionOutcome,
    ActionPort,
    ActionRequest,
    ActionRequestError,
    ActionStatus,
    ActionTarget,
    ActionTargetError,
    ActionType,
    DryRunBackend,
    EmergencyStop,
    EmergencyStopSource,
    GuardStage,
    NullActionPort,
    RateLimiter,
    RegistryTargetResolver,
    ResolvedTarget,
    SafetyGuard,
    SemanticTarget,
    TargetEntry,
    TargetRegistry,
    as_action_port,
    create_input_backend,
    is_action_request,
    registry_from_mapping,
    validate_target_name,
)
from src.action.backend import BackendCall, NullWindowProbe
from src.core.geometry import Point, Rect
from src.core.paths import get_project_root

# ---------------------------------------------------------------------------
# 测试替身（**不产生任何真实输入**）
# ---------------------------------------------------------------------------

DEFAULT_RECT = Rect(x=100, y=200, width=1600, height=900)


class FakeClock:
    """可注入的单调时钟。"""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def clock(self) -> float:
        """当前时间（秒）。"""
        return self.value

    def advance(self, seconds: float) -> None:
        """推进时间。"""
        self.value += seconds


class StaticWindowProbe:
    """固定窗口探测器（可配置有效 / 前台 / 客户区）。"""

    def __init__(
        self,
        *,
        valid: bool = True,
        foreground: bool = True,
        rect: Rect | None = DEFAULT_RECT,
    ) -> None:
        self.valid = valid
        self.foreground = foreground
        self.rect = rect
        self.calls: list[str] = []

    def is_valid(self) -> bool:
        """记录并返回配置的有效性。"""
        self.calls.append("is_valid")
        return self.valid

    def is_foreground(self) -> bool:
        """记录并返回配置的前台状态。"""
        self.calls.append("is_foreground")
        return self.foreground

    def client_rect(self) -> Rect | None:
        """返回配置的客户区矩形。"""
        self.calls.append("client_rect")
        return self.rect

    def describe(self) -> str:
        """返回可读描述。"""
        return f"StaticWindowProbe(valid={self.valid}, foreground={self.foreground})"


class RecordingBackend:
    """记录调用但**不发送任何输入**的真实感替身（``is_dry_run=False``）。

    用途：验证 ``EXECUTED`` 路径与门控"已标定"规则，同时保证测试零副作用。
    """

    def __init__(self) -> None:
        self.calls: list[BackendCall] = []

    @property
    def is_dry_run(self) -> bool:
        """``False``：让门控按真实后端处理（要求目标已标定）。"""
        return False

    def _record(self, operation: str, **kwargs: object) -> None:
        self.calls.append(BackendCall(operation=operation, **kwargs))  # type: ignore[arg-type]

    def move(self, point: Point, /) -> None:
        """记录移动。"""
        self._record("move", point=point)

    def click(self, point: Point, /, *, button: str = "left", count: int = 1) -> None:
        """记录点击。"""
        self._record("click", point=point, button=button, count=count)

    def press(self, point: Point, /, *, button: str = "left") -> None:
        """记录按下。"""
        self._record("press", point=point, button=button)

    def release(self, point: Point, /, *, button: str = "left") -> None:
        """记录释放。"""
        self._record("release", point=point, button=button)

    def key_press(self, key: str, /) -> None:
        """记录按键按下。"""
        self._record("key_press", key=key)

    def key_release(self, key: str, /) -> None:
        """记录按键释放。"""
        self._record("key_release", key=key)


class ExplodingBackend(RecordingBackend):
    """执行时抛异常的替身（验证异常隔离）。"""

    def click(self, point: Point, /, *, button: str = "left", count: int = 1) -> None:
        """抛出异常。"""
        raise RuntimeError("boom")


def make_registry(*, calibrated: bool = True) -> TargetRegistry:
    """构造带一个已注册目标的注册表。"""
    registry = TargetRegistry.empty()
    registry.add("TEST_BUTTON", 0.5, 0.5, calibrated=calibrated, description="测试目标")
    return registry


def make_dispatcher(
    *,
    backend: object | None = None,
    registry: TargetRegistry | None = None,
    probe: object | None = None,
    estop: EmergencyStop | None = None,
    clock: FakeClock | None = None,
    require_foreground: bool = True,
    min_interval_s: float = 0.0,
    trust_explicit_ratio: bool = False,
    sleeper: object | None = None,
) -> tuple[ActionDispatcher, FakeClock]:
    """构造分发器（默认：dry-run 后端 + 静态窗口 + 空限流）。"""
    active_clock = clock if clock is not None else FakeClock()
    guard = SafetyGuard(
        estop=estop,
        window_probe=probe if probe is not None else StaticWindowProbe(),  # type: ignore[arg-type]
        rate_limiter=RateLimiter(min_interval_s=min_interval_s, clock=active_clock.clock),
        require_foreground=require_foreground,
    )
    dispatcher = ActionDispatcher(
        backend=backend,  # type: ignore[arg-type]
        resolver=RegistryTargetResolver(
            registry if registry is not None else make_registry(),
            trust_explicit_ratio=trust_explicit_ratio,
        ),
        guard=guard,
        clock=active_clock.clock,
        now=lambda: "2025-06-01T12:00:00+08:00",
        sleeper=sleeper if sleeper is not None else (lambda _seconds: None),  # type: ignore[arg-type]
    )
    return dispatcher, active_clock


def click_request(
    target: SemanticTarget | None = None,
    *,
    action_type: ActionType = ActionType.CLICK,
    parameters: dict | None = None,
) -> ActionRequest:
    """构造一个点击请求。"""
    return ActionRequest(
        request_id="req-1",
        action_type=action_type,
        target=target if target is not None else SemanticTarget.from_name("TEST_BUTTON"),
        parameters=parameters or {},
    )


# ---------------------------------------------------------------------------
# 1. ActionRequest schema
# ---------------------------------------------------------------------------


def test_action_request_required_fields() -> None:
    """必需字段齐全（request_id / action_type / target / parameters / created_at）。"""
    request = click_request()

    names = {item.name for item in dataclasses.fields(ActionRequest)}

    assert {"request_id", "action_type", "target", "parameters", "created_at"} <= names
    assert request.created_at  # 自动填充
    assert request.action_type is ActionType.CLICK
    assert request.parameters == {}
    assert request.attempt == 1


def test_action_request_serialization_has_no_coordinates() -> None:
    """``to_dict()`` 只含语义信息（无像素坐标 / 无截图）。"""
    payload = click_request().to_dict()

    assert payload["request_id"] == "req-1"
    assert payload["action_type"] == "click"
    assert payload["target"] == {"name": "TEST_BUTTON"}
    assert "image" not in payload
    assert "x" not in payload and "y" not in payload


def test_action_request_missing_id_rejected() -> None:
    """缺少 request_id → :class:`ActionRequestError`。"""
    with pytest.raises(ActionRequestError):
        ActionRequest(request_id="", action_type=ActionType.CLICK, target=SemanticTarget.from_name("X"))


def test_action_request_unknown_type_rejected() -> None:
    """未知动作类型 → :class:`ActionRequestError`。"""
    with pytest.raises(ActionRequestError):
        ActionRequest(request_id="r", action_type="teleport", target=None)  # type: ignore[arg-type]


def test_action_request_requires_target_except_wait() -> None:
    """需要目标的动作必须给目标；``wait`` 可以没有目标。"""
    with pytest.raises(ActionRequestError):
        ActionRequest(request_id="r", action_type=ActionType.CLICK, target=None)

    wait = ActionRequest(request_id="r", action_type=ActionType.WAIT, target=None)

    assert wait.target is None
    assert "wait" in wait.describe()


def test_action_request_rejects_invalid_numeric_fields() -> None:
    """``timeout_s`` / ``attempt`` 取值非法 → 报错。"""
    for kwargs in ({"timeout_s": 0.0}, {"timeout_s": -1.0}, {"attempt": 0}):
        with pytest.raises(ActionRequestError):
            ActionRequest(
                request_id="r",
                action_type=ActionType.WAIT,
                target=None,
                **kwargs,
            )


# ---------------------------------------------------------------------------
# 2. 禁止 pixel coordinate
# ---------------------------------------------------------------------------


def test_action_request_has_no_pixel_fields() -> None:
    """数据类字段里不存在像素 / 屏幕坐标。"""
    names = {item.name.lower() for item in dataclasses.fields(ActionRequest)}

    assert not names & {"x", "y", "screen_x", "screen_y", "pixel_x", "pixel_y", "position"}


def test_action_request_rejects_coordinate_parameters() -> None:
    """参数里出现坐标键 → 直接报错（不允许绕过语义目标）。"""
    for key in sorted(FORBIDDEN_PARAMETER_KEYS):
        with pytest.raises(ActionRequestError, match="坐标字段"):
            ActionRequest(
                request_id="r",
                action_type=ActionType.CLICK,
                target=SemanticTarget.from_name("TEST_BUTTON"),
                parameters={key: 123},
            )


def test_semantic_target_rejects_pixel_writings() -> None:
    """语义目标不接受 ``reference`` / ``x`` / ``y`` 等像素写法。"""
    for payload in ({"reference": [100, 200]}, {"x": 1, "y": 2}, {"screen_x": 1, "screen_y": 2}):
        with pytest.raises(ActionTargetError):
            SemanticTarget.from_mapping(payload)


def test_semantic_target_rejects_coordinates_in_name() -> None:
    """目标名不得编码像素坐标（``CLICK_1530_850``）。"""
    with pytest.raises(ActionTargetError, match="像素坐标"):
        validate_target_name("CLICK_1530_850")

    with pytest.raises(ActionTargetError):
        SemanticTarget.from_name("TAP_100_200")

    assert validate_target_name("BATTLE_AUTO_BUTTON") == "BATTLE_AUTO_BUTTON"


def test_semantic_target_ratio_validation() -> None:
    """比例必须在 0..1；名称与比例互斥。"""
    assert SemanticTarget.from_ratio(0.0, 1.0).is_ratio is True
    for x_ratio, y_ratio in ((1.5, 0.5), (-0.1, 0.5), (0.5, 2.0)):
        with pytest.raises(ActionTargetError):
            SemanticTarget.from_ratio(x_ratio, y_ratio)
    with pytest.raises(ActionTargetError):
        SemanticTarget(name="X", x_ratio=0.5, y_ratio=0.5)
    with pytest.raises(ActionTargetError):
        SemanticTarget()


def test_resolved_target_exposes_only_ratios() -> None:
    """``ResolvedTarget`` 只有窗口相对位置，没有像素字段。"""
    resolved = ResolvedTarget(target="X", x_ratio=0.92, y_ratio=0.95, confidence=0.8)

    assert resolved.window_relative_position == {"x_ratio": 0.92, "y_ratio": 0.95}
    payload = resolved.to_dict()
    assert "window_relative_position" in payload
    assert not {"x", "y", "screen_x", "pixel"} & set(payload)


def test_from_task_request_rejects_reference_target() -> None:
    """Task 侧形如 ``{"reference": [x, y]}`` 的目标被拒绝（禁止像素）。"""
    task_like = {
        "action_id": "farming:STEP:1",
        "kind": "click",
        "target": {"reference": [1530, 850]},
        "params": {},
    }

    with pytest.raises(ActionTargetError):
        ActionRequest.from_task_request(task_like)


# ---------------------------------------------------------------------------
# 3. SemanticTarget resolution
# ---------------------------------------------------------------------------


def test_registry_resolution() -> None:
    """注册表解析：语义名 → 窗口相对位置 + 置信度 + 来源。"""
    registry = make_registry()
    resolver = RegistryTargetResolver(registry)

    resolved = resolver.resolve(SemanticTarget.from_name("TEST_BUTTON"))

    assert resolved is not None
    assert resolved.x_ratio == pytest.approx(0.5)
    assert resolved.y_ratio == pytest.approx(0.5)
    assert resolved.confidence == pytest.approx(1.0)
    assert resolved.source == "registry"
    assert resolved.calibrated is True
    assert resolved.target == "TEST_BUTTON"


def test_builtin_registry_entries_are_placeholders() -> None:
    """内置目标存在但**未标定**（占位比例：不允许用于真实输入）。"""
    registry = TargetRegistry.default()

    resolved = registry.resolve(ActionTarget.BATTLE_AUTO_BUTTON.value)

    assert resolved is not None
    assert resolved.calibrated is False
    assert len(BUILTIN_TARGETS) == len(TargetRegistry.default())
    assert ActionTarget.is_builtin("BATTLE_AUTO_BUTTON") is True
    assert ActionTarget.is_builtin("buttons/explore") is False


def test_unregistered_target_is_not_resolved() -> None:
    """未注册目标解析为 ``None``（Dispatcher 据此拒绝，不猜坐标）。"""
    resolver = RegistryTargetResolver(TargetRegistry.empty())

    assert resolver.resolve(SemanticTarget.from_name("NOPE")) is None
    assert resolver.resolve(None) is None


def test_explicit_ratio_resolution_and_trust() -> None:
    """显式比例可用，但默认不"信任"（未标定 → 真实后端会拒绝）。"""
    strict = RegistryTargetResolver(trust_explicit_ratio=False)
    trusted = RegistryTargetResolver(trust_explicit_ratio=True)

    strict_resolved = strict.resolve(SemanticTarget.from_ratio(0.3, 0.4))
    trusted_resolved = trusted.resolve(SemanticTarget.from_ratio(0.3, 0.4))

    assert strict_resolved is not None and strict_resolved.source == "explicit_ratio"
    assert strict_resolved.calibrated is False
    assert trusted_resolved is not None and trusted_resolved.calibrated is True


def test_registry_mapping_and_custom_names() -> None:
    """注册表支持自定义名称与映射构造（不限于内置枚举）。"""
    registry = registry_from_mapping(
        {
            "buttons/explore": {"x_ratio": 0.85, "y_ratio": 0.9, "calibrated": True},
            "REWARD_CONFIRM": {"x_ratio": 0.5, "y_ratio": 0.8},
        }
    )

    assert registry.names() == ("buttons/explore", "REWARD_CONFIRM")
    assert registry.get("REWARD_CONFIRM") is not None
    assert registry.get("REWARD_CONFIRM").calibrated is False  # type: ignore[union-attr]
    assert registry.resolve("buttons/explore") is not None
    assert "buttons/explore" in registry

    with pytest.raises(ActionTargetError):
        registry_from_mapping({"bad": {"x_ratio": 0.5}})


def test_target_entry_validation() -> None:
    """记录级校验（比例越界 / 置信度越界 / 名称非法）。"""
    with pytest.raises(ActionTargetError):
        TargetEntry(name="X", x_ratio=1.5, y_ratio=0.5)
    with pytest.raises(ActionTargetError):
        TargetEntry(name="X", x_ratio=0.5, y_ratio=0.5, confidence=1.5)
    with pytest.raises(ActionTargetError):
        TargetEntry(name="CLICK_10_20", x_ratio=0.5, y_ratio=0.5)
    assert TargetRegistry().unregister("X") is False


# ---------------------------------------------------------------------------
# 4. Resolver 不依赖 vision（AST 依赖边界）
# ---------------------------------------------------------------------------

ALLOWED_IMPORT_PREFIXES: tuple[str, ...] = ("src.core", "src.utils", "src.action")
FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "cv2",
    "numpy",
    "PIL",
    "torch",
    "pyautogui",
    "pynput",
    "keyboard",
    "src.vision",
    "src.evidence",
    "src.task",
    "src.runtime",
)


def _iter_action_imports() -> Iterator[tuple[Path, str]]:
    """遍历 ``src/action/**/*.py`` 的导入语句。"""
    for path in sorted((get_project_root() / "src" / "action").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield path, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield path, node.module


def test_action_layer_forbidden_imports_absent() -> None:
    """``src/action`` 不得 import vision / evidence / task / runtime / cv2 / 输入注入库。"""
    violations = [
        f"{path.name}: {module}"
        for path, module in _iter_action_imports()
        for forbidden in FORBIDDEN_IMPORTS
        if module == forbidden or module.startswith(f"{forbidden}.")
    ]

    assert violations == [], f"Action 层出现被禁止的依赖: {violations}"


def test_action_layer_allowed_imports_only() -> None:
    """依赖白名单：标准库 / ``src.core`` / ``src.utils`` / ``src.action``。"""
    violations: list[str] = []
    for path, module in _iter_action_imports():
        top = module.split(".")[0]
        if top in sys.stdlib_module_names or top == "__future__":
            continue
        if module.startswith(ALLOWED_IMPORT_PREFIXES):
            continue
        violations.append(f"{path.name}: {module}")

    assert violations == [], f"Action 层出现越界依赖: {violations}"


def test_action_layer_has_no_eval_or_exec() -> None:
    """Action 层不得使用 ``eval`` / ``exec`` / ``compile``。"""
    violations: list[str] = []
    for path in sorted((get_project_root() / "src" / "action").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec", "compile"}:
                    violations.append(f"{path.name}:{node.lineno} {node.func.id}()")

    assert violations == [], f"Action 层出现动态执行: {violations}"


def test_resolver_module_does_not_touch_vision_concepts() -> None:
    """``resolver.py`` 只处理语义目标 ↔ 比例：代码里不出现 Frame / detector / cv2。

    只检查**可执行代码**（名称 / 属性 / 导入），注释与 docstring 不参与
    （文档里明确写着"不读 Frame、不调 detector"是设计说明）。
    """
    source = (get_project_root() / "src" / "action" / "resolver.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])

    assert not names & {"Frame", "detect", "detector", "cv2", "numpy", "vision"}


# ---------------------------------------------------------------------------
# 5. DryRun 执行 / EXECUTED 流程
# ---------------------------------------------------------------------------


def test_dry_run_backend_records_without_input() -> None:
    """默认后端：动作被记录但标记为未执行（``SKIPPED``），且坐标换算正确。"""
    dispatcher, _ = make_dispatcher()
    backend = dispatcher.backend
    assert isinstance(backend, DryRunBackend)

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.SKIPPED
    assert "dry-run" in outcome.reason
    assert backend.count == 1
    call = backend.calls[0]
    assert call.operation == "click"
    assert call.point is not None
    # 客户区 (x=100, y=200, 1600x900) × (0.5, 0.5)
    assert call.point.to_int_tuple() == (900, 650)
    assert outcome.resolved_target is not None
    assert outcome.resolved_target.source == "registry"


def test_executed_path_with_real_like_backend() -> None:
    """真实感替身（``is_dry_run=False``）：动作进入 ``EXECUTED`` 并携带解析结果。"""
    backend = RecordingBackend()
    dispatcher, _ = make_dispatcher(backend=backend)

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.EXECUTED
    assert outcome.executed_at is not None
    assert outcome.resolved_target is not None
    assert [call.operation for call in backend.calls] == ["click"]
    assert backend.calls[0].count == 1


def test_double_click_and_long_press_and_key() -> None:
    """``double_click`` / ``long_press`` / ``key`` 的后端调用形态正确。"""
    backend = RecordingBackend()
    sleeper_calls: list[float] = []
    dispatcher, _ = make_dispatcher(
        backend=backend, sleeper=lambda seconds: sleeper_calls.append(seconds)
    )

    dispatcher.request(click_request(action_type=ActionType.DOUBLE_CLICK))
    dispatcher.request(click_request(action_type=ActionType.LONG_PRESS, parameters={"duration_s": 0.9}))
    dispatcher.request(
        click_request(action_type=ActionType.KEY, parameters={"key": "enter", "hold_s": 0.1})
    )

    operations = [call.operation for call in backend.calls]
    assert operations == ["click", "press", "release", "key_press", "key_release"]
    assert backend.calls[0].count == 2
    assert sleeper_calls == [0.9, 0.1]
    assert backend.calls[-2].key == "enter"


def test_wait_needs_no_target_and_skips() -> None:
    """``wait``：不需要目标、不产生输入（``SKIPPED``）。"""
    backend = RecordingBackend()
    dispatcher, _ = make_dispatcher(backend=backend)

    outcome = dispatcher.request(
        ActionRequest(request_id="w", action_type=ActionType.WAIT, target=None)
    )

    assert outcome.status is ActionStatus.SKIPPED
    assert backend.calls == []


def test_unimplemented_action_type_rejected() -> None:
    """``drag`` 尚未实现 → 明确 ``REJECTED``（不静默降级）。"""
    dispatcher, _ = make_dispatcher()

    outcome = dispatcher.request(click_request(action_type=ActionType.DRAG))

    assert outcome.status is ActionStatus.REJECTED
    assert "尚未实现" in outcome.reason


# ---------------------------------------------------------------------------
# 6. E-STOP 拒绝
# ---------------------------------------------------------------------------


def test_estop_api_triggers_rejection(tmp_path: Path) -> None:
    """API 触发 E-STOP → 动作被拒绝（第一道门）。"""
    estop = EmergencyStop(path=tmp_path / "ESTOP")
    dispatcher, _ = make_dispatcher(estop=estop)

    estop.trigger("user abort")
    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.ESTOP
    assert estop.source == EmergencyStopSource.API.value
    assert dispatcher.backend.count == 0  # type: ignore[attr-defined]


def test_estop_sentinel_file_rejection(tmp_path: Path) -> None:
    """文件哨兵存在 → 动作被拒绝（人工创建文件即可停止）。"""
    sentinel = tmp_path / "ESTOP"
    estop = EmergencyStop(path=sentinel)
    dispatcher, _ = make_dispatcher(estop=estop)

    sentinel.write_text("manual stop\n", encoding="utf-8")
    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.ESTOP
    assert estop.source == EmergencyStopSource.SENTINEL.value


def test_estop_reset_allows_actions_again(tmp_path: Path) -> None:
    """复位后动作恢复（复位必须显式调用，且默认删除哨兵）。"""
    estop = EmergencyStop(path=tmp_path / "ESTOP")
    dispatcher, _ = make_dispatcher(estop=estop)
    estop.trigger("stop")
    assert dispatcher.request(click_request()).status is ActionStatus.REJECTED

    estop.reset()

    assert dispatcher.request(click_request()).status is ActionStatus.SKIPPED


def test_estop_state_contains_no_user_data(tmp_path: Path) -> None:
    """E-STOP 状态快照只含运行信息。"""
    estop = EmergencyStop(path=tmp_path / "ESTOP")
    estop.trigger("because")

    assert set(estop.state.to_dict()) == {"triggered", "reason", "source", "triggered_at"}
    assert "inactive" in EmergencyStop(path=tmp_path / "other").state.describe()


# ---------------------------------------------------------------------------
# 7. Foreground / Window 拒绝
# ---------------------------------------------------------------------------


def test_window_invalid_rejection() -> None:
    """窗口无效 → ``REJECTED``（stage=window）。"""
    dispatcher, _ = make_dispatcher(probe=StaticWindowProbe(valid=False))

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.WINDOW


def test_foreground_rejection_and_opt_out() -> None:
    """失焦 → 拒绝；显式关闭前台要求后放行（仍不点真实输入）。"""
    dispatcher, _ = make_dispatcher(probe=StaticWindowProbe(foreground=False))
    lenient, _ = make_dispatcher(
        probe=StaticWindowProbe(foreground=False), require_foreground=False
    )

    strict_outcome = dispatcher.request(click_request())
    lenient_outcome = lenient.request(click_request())

    assert strict_outcome.status is ActionStatus.REJECTED
    assert strict_outcome.detail["stage"] == GuardStage.FOREGROUND
    assert lenient_outcome.status is ActionStatus.SKIPPED


def test_missing_client_rect_rejection() -> None:
    """拿不到客户区矩形 → 拒绝（不猜坐标）。"""
    dispatcher, _ = make_dispatcher(probe=StaticWindowProbe(rect=None))

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.WINDOW


def test_null_window_probe_is_safe_default() -> None:
    """``NullWindowProbe`` 报告窗口无效 → 一律拒绝（安全默认）。"""
    dispatcher = ActionDispatcher(
        backend=DryRunBackend(),
        guard=SafetyGuard(window_probe=NullWindowProbe()),
        resolver=RegistryTargetResolver(make_registry()),
    )
    assert NullWindowProbe().is_valid() is False

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.WINDOW


def test_unregistered_target_rejection() -> None:
    """未注册目标 → ``REJECTED``（stage=target，绝不 fallback click）。"""
    dispatcher, _ = make_dispatcher(registry=TargetRegistry.empty())

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.TARGET
    # Phase 7.1：失败原因由解析器细化（既有 stage 语义不变）
    assert outcome.detail["resolution_reason"] == "target_not_found"
    assert "target_not_found" in outcome.reason


def test_uncalibrated_target_rejected_for_real_backend() -> None:
    """占位比例（未标定）在真实后端下被拒绝；dry-run 不受影响。"""
    registry = make_registry(calibrated=False)
    real_backend = RecordingBackend()
    strict, _ = make_dispatcher(backend=real_backend, registry=registry)
    loose, _ = make_dispatcher(registry=registry)  # 默认 dry-run

    strict_outcome = strict.request(click_request())
    loose_outcome = loose.request(click_request())

    assert strict_outcome.status is ActionStatus.REJECTED
    assert strict_outcome.detail["stage"] == GuardStage.TARGET
    assert "未标定" in strict_outcome.reason
    assert real_backend.calls == []
    assert loose_outcome.status is ActionStatus.SKIPPED


# ---------------------------------------------------------------------------
# 8. Backend 异常隔离
# ---------------------------------------------------------------------------


def test_backend_exception_is_isolated() -> None:
    """后端抛异常 → ``FAILED``（不传播），统计记录错误。"""
    dispatcher, _ = make_dispatcher(backend=ExplodingBackend())

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.FAILED
    assert "backend_error" in outcome.reason
    assert outcome.detail["error_type"] == "RuntimeError"
    assert dispatcher.stats.failed == 1


def test_dispatcher_survives_backend_failure() -> None:
    """一次后端异常不影响后续动作（门控与状态仍可继续）。"""
    backend = ExplodingBackend()
    dispatcher, _ = make_dispatcher(backend=backend)

    first = dispatcher.request(click_request())
    backend.__class__ = RecordingBackend  # 恢复为正常替身
    second = dispatcher.request(click_request())

    assert first.status is ActionStatus.FAILED
    assert second.status is ActionStatus.EXECUTED
    assert dispatcher.stats.requests == 2


# ---------------------------------------------------------------------------
# 9. Dispatcher 流程（顺序 / 限流 / 统计）
# ---------------------------------------------------------------------------


def test_guard_order_estop_before_window(tmp_path: Path) -> None:
    """门控顺序：E-STOP 先于窗口检查（两者都不满足时报告 E-STOP）。"""
    estop = EmergencyStop(path=tmp_path / "ESTOP")
    probe = StaticWindowProbe(valid=False)
    dispatcher, _ = make_dispatcher(estop=estop, probe=probe)

    estop.trigger("stop")
    outcome = dispatcher.request(click_request())

    assert outcome.detail["stage"] == GuardStage.ESTOP
    assert probe.calls == []  # 窗口探测根本没被调用


def test_probe_call_order() -> None:
    """窗口 → 前台 → 客户区的调用顺序（解析前只做有效性 / 前台检查）。"""
    probe = StaticWindowProbe()
    dispatcher, _ = make_dispatcher(probe=probe)

    dispatcher.request(click_request())

    assert probe.calls == ["is_valid", "is_foreground", "client_rect"]


def test_rate_limit_rejection_and_recovery() -> None:
    """限流：间隔不足即拒绝；时钟推进后放行。"""
    clock = FakeClock()
    dispatcher, _ = make_dispatcher(clock=clock, min_interval_s=1.0)

    first = dispatcher.request(click_request())
    second = dispatcher.request(
        ActionRequest(
            request_id="req-2",
            action_type=ActionType.CLICK,
            target=SemanticTarget.from_name("TEST_BUTTON"),
        )
    )

    assert first.status is ActionStatus.SKIPPED
    assert second.status is ActionStatus.REJECTED
    assert second.detail["stage"] == GuardStage.RATE_LIMIT

    clock.advance(1.5)
    third = dispatcher.request(
        ActionRequest(
            request_id="req-3",
            action_type=ActionType.CLICK,
            target=SemanticTarget.from_name("TEST_BUTTON"),
        )
    )

    assert third.status is ActionStatus.SKIPPED


def test_rate_limiter_window_limit() -> None:
    """每分钟上限生效。"""
    clock = FakeClock()
    limiter = RateLimiter(min_interval_s=0.0, max_actions_per_minute=2, clock=clock.clock)

    assert limiter.check().allowed is True
    limiter.record()
    clock.advance(0.1)
    limiter.record()
    clock.advance(0.1)

    blocked = limiter.check()

    assert blocked.allowed is False
    assert blocked.stage == GuardStage.RATE_LIMIT
    assert limiter.actions_in_window == 2

    clock.advance(61.0)
    assert limiter.check().allowed is True
    limiter.reset()
    assert limiter.total == 0


def test_latency_and_stats_recorded() -> None:
    """延迟按注入时钟统计；统计字段完整。"""
    clock = FakeClock()
    dispatcher, _ = make_dispatcher(backend=RecordingBackend(), clock=clock)
    original_clock = clock.clock
    values = iter([10.0, 10.25])  # 两次取时钟：开始 + 结束 → 250ms

    dispatcher._clock = lambda: next(values)  # type: ignore[attr-defined]
    outcome = dispatcher.request(click_request())
    dispatcher._clock = original_clock  # type: ignore[attr-defined]

    assert outcome.latency_ms == pytest.approx(250.0, abs=0.01)
    payload = dispatcher.stats.to_dict()
    assert payload["requests"] == 1
    assert payload["executed"] == 1
    assert payload["mean_latency_ms"] == pytest.approx(250.0, abs=0.01)
    assert "reject_stages" in payload


def test_stats_reject_stage_counting() -> None:
    """拒绝阶段被分类统计（便于观测"为什么没动作"）。"""
    dispatcher, _ = make_dispatcher(probe=StaticWindowProbe(foreground=False))

    dispatcher.request(click_request())
    dispatcher.request(click_request())

    payload = dispatcher.stats.to_dict()

    assert payload["rejected"] == 2
    assert payload["reject_stages"][GuardStage.FOREGROUND] == 2


def test_dispatcher_accepts_task_side_request_object() -> None:
    """字段视图对接：Task 侧请求对象可直接提交（Dispatcher 即 Task 的 ActionPort）。"""
    backend = RecordingBackend()
    dispatcher, _ = make_dispatcher(backend=backend)
    task_like = type(
        "TaskSideRequest",
        (),
        {
            "action_id": "farming:IN_BATTLE:2",
            "kind": "click",
            "target": {"element": "TEST_BUTTON"},
            "params": {},
            "task_id": "farming",
            "step_id": "IN_BATTLE",
            "timeout_s": 2.0,
            "attempt": 2,
        },
    )()

    outcome = dispatcher.request(task_like)

    assert outcome.status is ActionStatus.EXECUTED
    assert outcome.is_failure is False
    assert backend.calls[0].operation == "click"


def test_invalid_request_is_rejected_not_raised() -> None:
    """非法请求（未知类型 / 含坐标）转换为 ``REJECTED``，不向调用方抛出。"""
    dispatcher, _ = make_dispatcher()

    bad_type = dispatcher.request({"action_id": "x", "kind": "teleport"})
    bad_coords = dispatcher.request(
        {"action_id": "y", "kind": "click", "target": {"x": 10, "y": 20}}
    )

    assert bad_type.status is ActionStatus.REJECTED
    assert bad_type.detail["stage"] == "request"
    assert bad_coords.status is ActionStatus.REJECTED


def test_is_action_request_and_as_action_port() -> None:
    """端口自检工具可用。"""
    dispatcher, _ = make_dispatcher()

    assert as_action_port(dispatcher) is dispatcher
    assert is_action_request(click_request()) is True
    assert is_action_request({"kind": "click"}) is True
    assert is_action_request(42) is False
    assert isinstance(dispatcher, ActionPort)
    assert NullActionPort().request({"kind": "click"}).status is ActionStatus.SKIPPED


# ---------------------------------------------------------------------------
# 10. Windows backend 接口（纯接口 / 纯函数；不发真实输入）
# ---------------------------------------------------------------------------

IS_WINDOWS = sys.platform == "win32"


@pytest.mark.skipif(not IS_WINDOWS, reason="Win32 后端只在 Windows 上可构造")
def test_windows_backend_classes_and_dry_run() -> None:
    """Windows 后端类存在、方法齐全，且 dry-run 模式不发送输入。"""
    from src.action.backend.windows import (
        WindowsInputBackend,
        WindowsKeyboardBackend,
        WindowsMouseBackend,
        Win32WindowProbe,
    )

    mouse = WindowsMouseBackend(dry_run=True)
    keyboard = WindowsKeyboardBackend(dry_run=True)
    combined = WindowsInputBackend(mouse=mouse, keyboard=keyboard)

    assert mouse.is_dry_run is True
    assert combined.is_dry_run is True
    for name in ("move", "click", "press", "release"):
        assert callable(getattr(mouse, name))
    for name in ("key_press", "key_release"):
        assert callable(getattr(keyboard, name))

    mouse.click(Point(10.0, 20.0))
    keyboard.key_press("enter")

    assert [call.operation for call in mouse.calls] == ["click"]
    assert [call.operation for call in keyboard.calls] == ["key_press"]
    assert Win32WindowProbe(0).is_valid() is False


@pytest.mark.skipif(not IS_WINDOWS, reason="Win32 结构定义只在 Windows 上可用")
def test_windows_sendinput_structures() -> None:
    """``SendInput`` 结构定义有效（大小非零、字段齐全）。"""
    import ctypes

    from src.action.backend.windows import _INPUT, _KEYBDINPUT, _MOUSEINPUT

    assert ctypes.sizeof(_INPUT) > ctypes.sizeof(_MOUSEINPUT) - 1
    assert ctypes.sizeof(_INPUT) >= ctypes.sizeof(_MOUSEINPUT)
    assert ctypes.sizeof(_INPUT) >= ctypes.sizeof(_KEYBDINPUT)
    assert {field[0] for field in _MOUSEINPUT._fields_} >= {"dx", "dy", "dwFlags"}
    assert {field[0] for field in _KEYBDINPUT._fields_} >= {"wVk", "wScan", "dwFlags"}


def test_key_code_resolution() -> None:
    """按键名 → 虚拟键码（未知键报错，不猜）。"""
    from src.action.backend.windows import KEY_CODES, resolve_key_code

    assert resolve_key_code("enter") == 0x0D
    assert resolve_key_code("ENTER") == 0x0D
    assert resolve_key_code("f1") == 0x70
    assert resolve_key_code("a") == 0x41
    assert resolve_key_code("7") == 0x37
    assert "space" in KEY_CODES

    from src.action.models import InputBackendError

    for key in ("", "hyperkey"):
        with pytest.raises(InputBackendError):
            resolve_key_code(key)


def test_normalize_absolute_mapping() -> None:
    """屏幕坐标 → ``0..65535`` 归一化（含边界与裁剪）。"""
    from src.action.backend.windows import normalize_absolute

    assert normalize_absolute(0, 0, 2560) == 0
    assert normalize_absolute(2559, 0, 2560) == 65535
    assert normalize_absolute(1280, 0, 2561) == 32768
    assert normalize_absolute(-500, 0, 2560) == 0
    assert normalize_absolute(99999, 0, 2560) == 65535
    assert normalize_absolute(10, 0, 1) == 0  # 退化尺寸 → 0
    assert normalize_absolute(-100, -100, 200) == 0
    assert normalize_absolute(99, -100, 200) == 65535
    assert normalize_absolute(100, -100, 200) == 65535  # 越界裁剪


def test_platform_guard_for_windows_backend() -> None:
    """请求 Windows 后端：Windows 上创建真实后端（**不发输入**），其它平台明确报错。"""
    from src.action import backend as backend_pkg
    from src.action.models import InputBackendUnavailableError

    class FakeConfig:
        enabled = True
        backend = "windows"

    if not IS_WINDOWS:
        with pytest.raises(InputBackendUnavailableError):
            backend_pkg.create_input_backend(FakeConfig())
        return

    from src.action.backend.windows import WindowsInputBackend

    backend = backend_pkg.create_input_backend(FakeConfig())

    assert isinstance(backend, WindowsInputBackend)
    assert backend.is_dry_run is False
    assert backend.calls == ()  # 构造后端不会产生任何调用


def test_create_input_backend_defaults_to_dry_run() -> None:
    """工厂默认返回 dry-run 后端（无配置 / enabled=False / backend=dry_run）。"""
    class DisabledConfig:
        enabled = False
        backend = "windows"

    class DryRunConfig:
        enabled = True
        backend = "dry_run"

    assert isinstance(create_input_backend(None), DryRunBackend)
    assert isinstance(create_input_backend(DisabledConfig()), DryRunBackend)
    assert isinstance(create_input_backend(DryRunConfig()), DryRunBackend)


def test_dispatcher_from_config_uses_dry_run() -> None:
    """``from_config`` 在默认配置下使用 dry-run 后端（安全默认）。"""
    from src.core import load_config

    config = load_config()
    dispatcher = ActionDispatcher.from_config(config, window_probe=StaticWindowProbe())

    assert dispatcher.is_dry_run is True
    assert isinstance(dispatcher.backend, DryRunBackend)
    assert "ActionDispatcher" in dispatcher.describe()

    outcome = dispatcher.request(click_request(target=SemanticTarget.from_name("NOPE")))
    assert outcome.status is ActionStatus.REJECTED


# ---------------------------------------------------------------------------
# 端口 / 结果对象的字段兼容性
# ---------------------------------------------------------------------------


def test_outcome_is_task_compatible() -> None:
    """``ActionOutcome`` 满足 Task 层的字段视图（status / reason / is_failure）。"""
    outcome = ActionOutcome(status=ActionStatus.REJECTED, reason="nope")

    assert outcome.is_failure is True
    assert outcome.is_executed is False
    assert ActionOutcome(status=ActionStatus.EXECUTED).is_failure is False
    assert ActionOutcome(status=ActionStatus.SKIPPED).is_failure is False
    assert "rejected" in outcome.describe()


def test_backend_call_serialization() -> None:
    """后端调用记录可序列化（无截图 / 用户数据）。"""
    call = BackendCall(operation="click", point=Point(1.0, 2.0), button="left", count=2)

    payload = call.to_dict()

    assert payload["operation"] == "click"
    assert payload["point"] == [1, 2]
    assert payload["count"] == 2
    assert "image" not in payload
