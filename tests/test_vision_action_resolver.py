"""Phase 7.1 测试：Vision-aware Action Resolver。

覆盖用户要求的九项：

1. ``SemanticTarget`` → ``VisionTargetCandidate`` → ``ResolvedTarget`` 成功链路；
2. 目标不存在 → Rejected（``target_not_found``）；
3. 低 confidence → Rejected（``low_confidence``）；
4. TTL 超时 → Rejected（``target_expired``）；
5. 坐标非法 → Rejected（``invalid_target``）；
6. 旧 ``RegistryTargetResolver`` 行为不变；
7. Dispatcher **不知道** Vision 存在（AST + 运行时视图）；
8. Action 包依赖检查（禁止 vision / evidence / runtime / task / cv2 / numpy）；
9. **无真实输入**（全部使用 ``DryRunBackend`` / ``RecordingBackend``）。

另含：注入机制（单向 push）、别名、O(1) 查询与"不触发检测/截图"的性能约束、
以及 TTL / 置信度边界值。

**安全承诺**：本文件不产生任何真实键鼠输入。
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.action import (
    ActionDispatcher,
    ActionRequest,
    ActionStatus,
    ActionType,
    DryRunBackend,
    GuardStage,
    NullWindowProbe,
    RegistryTargetResolver,
    ResolvedTarget,
    SafetyGuard,
    SemanticTarget,
    TargetEntry,
    TargetRegistry,
    TargetResolution,
    TargetResolutionReason,
    TargetResolver,
    VisionAwareTargetResolver,
)
from src.action.backend import BackendCall
from src.core.geometry import Point, Rect, Size
from src.core.paths import get_project_root
from src.core.vision_target import (
    TargetSnapshot,
    VisionTargetCandidate,
    ratio_from_rect,
    ratio_from_screen_point,
)

# ---------------------------------------------------------------------------
# 替身（不产生任何真实输入）
# ---------------------------------------------------------------------------

DEFAULT_RECT = Rect(x=47, y=473, width=1704, height=960)


class FakeClock:
    """可注入单调时钟（秒）。"""

    def __init__(self, value: float = 100.0) -> None:
        self.value = float(value)
        self.reads = 0

    def clock(self) -> float:
        """返回当前时间并计数（性能断言用）。"""
        self.reads += 1
        return self.value

    def advance(self, seconds: float) -> None:
        """推进时间。"""
        self.value += float(seconds)


class StaticWindowProbe:
    """固定窗口探测器。"""

    def __init__(
        self, *, valid: bool = True, foreground: bool = True, rect: Rect | None = DEFAULT_RECT
    ) -> None:
        self.valid = valid
        self.foreground = foreground
        self.rect = rect
        self.probe_calls = 0

    def is_valid(self) -> bool:
        """窗口是否有效。"""
        return self.valid

    def is_foreground(self) -> bool:
        """是否前台。"""
        return self.foreground

    def client_rect(self) -> Rect | None:
        """客户区矩形。"""
        return self.rect

    def describe(self) -> str:
        """可读描述。"""
        return "StaticWindowProbe"


class RecordingBackend:
    """记录调用但不发送输入的真实感替身。"""

    def __init__(self) -> None:
        self.calls: list[BackendCall] = []

    @property
    def is_dry_run(self) -> bool:
        """``False``：让门控按真实后端规则处理。"""
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


def make_candidate(
    name: str = "battle_auto_button",
    *,
    x_ratio: float = 0.82,
    y_ratio: float = 0.91,
    confidence: float = 0.96,
    source: str = "evidence",
) -> VisionTargetCandidate:
    """构造一个视觉候选。"""
    return VisionTargetCandidate(
        target_name=name,
        x_ratio=x_ratio,
        y_ratio=y_ratio,
        confidence=confidence,
        source=source,
    )


def make_vision_resolver(
    *,
    clock: FakeClock | None = None,
    ttl_s: float = 1.0,
    minimum_confidence: float = 0.6,
    **kwargs: object,
) -> tuple[VisionAwareTargetResolver, FakeClock]:
    """构造 Vision-aware 解析器（默认 TTL 1.0s、min_confidence 0.6）。"""
    active_clock = clock if clock is not None else FakeClock()
    resolver = VisionAwareTargetResolver(
        ttl_s=ttl_s,
        minimum_confidence=minimum_confidence,
        clock=active_clock.clock,
        **kwargs,  # type: ignore[arg-type]
    )
    return resolver, active_clock


def make_dispatcher(
    resolver: object,
    *,
    backend: object | None = None,
    clock: FakeClock | None = None,
    probe: object | None = None,
) -> tuple[ActionDispatcher, FakeClock]:
    """构造使用指定 resolver 的 Dispatcher（默认 dry-run 后端）。"""
    active_clock = clock if clock is not None else FakeClock()
    dispatcher = ActionDispatcher(
        backend=backend,  # type: ignore[arg-type]
        resolver=resolver,  # type: ignore[arg-type]
        guard=SafetyGuard(
            window_probe=probe if probe is not None else StaticWindowProbe(),  # type: ignore[arg-type]
        ),
        clock=active_clock.clock,
        now=lambda: "2025-06-01T12:00:00+08:00",
        sleeper=lambda _seconds: None,
    )
    return dispatcher, active_clock


def click_request(name: str = "battle_auto_button") -> ActionRequest:
    """构造点击请求。"""
    return ActionRequest(
        request_id=f"req-{name}",
        action_type=ActionType.CLICK,
        target=SemanticTarget.from_name(name),
    )


# ---------------------------------------------------------------------------
# 1. 成功链路：SemanticTarget → VisionTargetCandidate → ResolvedTarget
# ---------------------------------------------------------------------------


def test_vision_candidate_resolves_to_resolved_target() -> None:
    """候选注入后，语义名可解析为 ResolvedTarget（source=vision）。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate()])

    resolution = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button"))

    assert resolution.ok is True
    resolved = resolution.resolved
    assert isinstance(resolved, ResolvedTarget)
    assert resolved.target == "battle_auto_button"
    assert resolved.x_ratio == pytest.approx(0.82)
    assert resolved.y_ratio == pytest.approx(0.91)
    assert resolved.confidence == pytest.approx(0.96)
    assert resolved.source == "vision"
    assert resolved.calibrated is True
    assert resolution.reason is TargetResolutionReason.RESOLVED
    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is not None


def test_resolver_satisfies_frozen_protocol() -> None:
    """新实现满足 Phase 7.0 冻结的 ``TargetResolver`` 协议（含可选增强协议）。"""
    resolver, _ = make_vision_resolver()
    registry_resolver = RegistryTargetResolver()

    assert isinstance(resolver, TargetResolver)
    assert isinstance(registry_resolver, TargetResolver)
    from src.action import ResolverWithReason

    assert isinstance(resolver, ResolverWithReason)
    assert isinstance(registry_resolver, ResolverWithReason)


def test_candidate_matching_is_case_insensitive() -> None:
    """候选名匹配大小写不敏感（`BATTLE_AUTO_BUTTON` ↔ `battle_auto_button`）。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate("battle_auto_button")])

    assert resolver.resolve(SemanticTarget.from_name("BATTLE_AUTO_BUTTON")) is not None
    assert resolver.resolve(SemanticTarget.from_name("Battle_Auto_Button")) is not None


def test_alias_mapping_supports_different_naming() -> None:
    """别名映射让"语义名 → 视觉名"可对齐（例如 `buttons/explore` → `explore_button`）。"""
    resolver, _ = make_vision_resolver(aliases={"buttons/explore": "explore_button"})
    resolver.update_candidates([make_candidate("explore_button")])

    resolution = resolver.resolve_detail(SemanticTarget.from_name("buttons/explore"))

    assert resolution.ok is True
    assert resolution.resolved is not None
    assert resolution.resolved.target == "explore_button"


def test_snapshot_replacement_overrides_previous_candidates() -> None:
    """新快照完全替换旧快照（旧候选立即不可用）。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate("old_button")])

    assert resolver.resolve(SemanticTarget.from_name("old_button")) is not None

    resolver.update_candidates([make_candidate("new_button")])

    assert resolver.resolve(SemanticTarget.from_name("old_button")) is None
    assert resolver.resolve(SemanticTarget.from_name("new_button")) is not None
    assert resolver.updates == 2


def test_clear_snapshot_makes_targets_unresolvable() -> None:
    """清空快照后一切目标都不可解析（不做任何兜底）。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate()])

    resolver.clear()

    assert resolver.snapshot is None
    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is None
    assert resolver.is_stale() is True


# ---------------------------------------------------------------------------
# 2/3/4/5. 四类拒绝原因
# ---------------------------------------------------------------------------


def test_target_not_found_rejection() -> None:
    """目标不存在 → ``target_not_found``。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate("other_button")])

    resolution = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button"))

    assert resolution.ok is False
    assert resolution.resolved is None
    assert resolution.reason is TargetResolutionReason.TARGET_NOT_FOUND
    assert resolution.detail["name"] == "battle_auto_button"
    assert "target_not_found" in resolution.message()


def test_target_not_found_without_snapshot() -> None:
    """从未注入过快照 → 同样 ``target_not_found``（并标注 snapshot=none）。"""
    resolver, _ = make_vision_resolver()

    resolution = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button"))

    assert resolution.reason is TargetResolutionReason.TARGET_NOT_FOUND
    assert resolution.detail["snapshot"] == "none"


def test_low_confidence_rejection() -> None:
    """置信度不足 → ``low_confidence``（附带实际值与阈值）。"""
    resolver, _ = make_vision_resolver(minimum_confidence=0.6)
    resolver.update_candidates([make_candidate(confidence=0.42)])

    resolution = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button"))

    assert resolution.reason is TargetResolutionReason.LOW_CONFIDENCE
    assert resolution.detail["confidence"] == pytest.approx(0.42)
    assert resolution.detail["minimum"] == pytest.approx(0.6)


def test_confidence_boundary_is_inclusive() -> None:
    """恰等于阈值时允许（``<`` 判定）。"""
    resolver, _ = make_vision_resolver(minimum_confidence=0.6)
    resolver.update_candidates([make_candidate(confidence=0.6)])

    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is not None


def test_target_expired_rejection() -> None:
    """TTL 过期 → ``target_expired``（含年龄与 TTL）。"""
    clock = FakeClock(100.0)
    resolver, _ = make_vision_resolver(clock=clock, ttl_s=1.0)
    resolver.update_candidates([make_candidate()], captured_monotonic=100.0)

    clock.advance(1.5)
    resolution = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button"))

    assert resolution.reason is TargetResolutionReason.TARGET_EXPIRED
    assert resolution.detail["age_s"] == pytest.approx(1.5)
    assert resolution.detail["ttl_s"] == pytest.approx(1.0)
    assert resolver.is_stale() is True


def test_ttl_boundary_is_inclusive() -> None:
    """恰等于 TTL 时仍然有效（``>`` 判定）。"""
    clock = FakeClock(100.0)
    resolver, _ = make_vision_resolver(clock=clock, ttl_s=1.0)
    resolver.update_candidates([make_candidate()], captured_monotonic=100.0)

    clock.advance(1.0)

    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is not None


def test_ttl_zero_disables_expiry() -> None:
    """``ttl_s=0`` 表示不做过期判定（长期有效，由调用方负责刷新）。"""
    clock = FakeClock(100.0)
    resolver, _ = make_vision_resolver(clock=clock, ttl_s=0.0)
    resolver.update_candidates([make_candidate()], captured_monotonic=100.0)

    clock.advance(3600.0)

    assert resolver.resolve(SemanticTarget.from_name("battle_auto_button")) is not None
    assert resolver.is_stale() is False


def test_invalid_target_rejection() -> None:
    """坐标非法（比例越界）→ ``invalid_target``（不构造 ResolvedTarget）。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates(
        [VisionTargetCandidate(target_name="bad_button", x_ratio=1.4, y_ratio=0.5, confidence=0.9)]
    )

    resolution = resolver.resolve_detail(SemanticTarget.from_name("bad_button"))

    assert resolution.reason is TargetResolutionReason.INVALID_TARGET
    assert resolution.resolved is None

    nan_resolution = resolver.resolve_detail(SemanticTarget.from_name("nan_button"))
    assert nan_resolution.reason is TargetResolutionReason.TARGET_NOT_FOUND  # 未注入 → not found

    resolver.update_candidates([make_candidate("negative", x_ratio=-0.2, confidence=0.9)])
    assert (
        resolver.resolve_detail(SemanticTarget.from_name("negative")).reason
        is TargetResolutionReason.INVALID_TARGET
    )


def test_no_target_rejection() -> None:
    """请求本身没有目标（``wait``）→ ``no_target``。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate()])

    resolution = resolver.resolve_detail(None)

    assert resolution.reason is TargetResolutionReason.NO_TARGET
    assert resolution.resolved is None


def test_explicit_ratio_still_supported() -> None:
    """显式比例目标仍然可用（默认不信任 → calibrated=False，与 Phase 7.0 一致）。"""
    strict, _ = make_vision_resolver()
    trusted, _ = make_vision_resolver(trust_explicit_ratio=True)
    target = SemanticTarget.from_ratio(0.3, 0.4)

    strict_resolution = strict.resolve_detail(target)
    trusted_resolution = trusted.resolve_detail(target)

    assert strict_resolution.ok is True
    assert strict_resolution.resolved is not None
    assert strict_resolution.resolved.source == "explicit_ratio"
    assert strict_resolution.resolved.calibrated is False
    assert trusted_resolution.resolved is not None
    assert trusted_resolution.resolved.calibrated is True


def test_trust_vision_targets_flag_controls_calibration() -> None:
    """``trust_vision_targets=False`` → 视觉目标也不视为已标定（真实后端会拒绝）。"""
    resolver, _ = make_vision_resolver(trust_vision_targets=False)
    resolver.update_candidates([make_candidate()])

    resolution = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button"))

    assert resolution.resolved is not None
    assert resolution.resolved.calibrated is False


# ---------------------------------------------------------------------------
# 6. 旧 RegistryTargetResolver 行为不变
# ---------------------------------------------------------------------------


def test_registry_resolver_behaviour_unchanged() -> None:
    """注册表解析器的既有行为保持不变（注册 / 未注册 / 占位 / 显式比例）。"""
    registry = TargetRegistry.empty()
    registry.add("TEST_BUTTON", 0.5, 0.5, calibrated=True)
    resolver = RegistryTargetResolver(registry)

    resolved = resolver.resolve(SemanticTarget.from_name("TEST_BUTTON"))
    assert resolved is not None
    assert resolved.x_ratio == pytest.approx(0.5)
    assert resolved.source == "registry"
    assert resolved.calibrated is True

    assert resolver.resolve(SemanticTarget.from_name("NOPE")) is None
    assert resolver.resolve(None) is None

    detail = resolver.resolve_detail(SemanticTarget.from_name("NOPE"))
    assert detail.reason is TargetResolutionReason.TARGET_NOT_FOUND

    default_resolver = RegistryTargetResolver()
    placeholder = default_resolver.resolve(SemanticTarget.from_name("BATTLE_AUTO_BUTTON"))
    assert placeholder is not None and placeholder.calibrated is False

    strict = RegistryTargetResolver(trust_explicit_ratio=False)
    explicit = strict.resolve(SemanticTarget.from_ratio(0.3, 0.4))
    assert explicit is not None and explicit.calibrated is False
    assert explicit.source == "explicit_ratio"


def test_registry_entries_accept_vision_style_names() -> None:
    """注册表与 vision 解析器使用同一套"语义名"（大小写写法可不同）。"""
    registry = TargetRegistry.empty()
    registry.register(TargetEntry(name="battle_auto_button", x_ratio=0.9, y_ratio=0.9))

    resolved = RegistryTargetResolver(registry).resolve(
        SemanticTarget.from_name("battle_auto_button")
    )

    assert resolved is not None


# ---------------------------------------------------------------------------
# 7. Dispatcher 不知道 Vision 存在
# ---------------------------------------------------------------------------


def test_dispatcher_is_agnostic_of_vision() -> None:
    """``dispatcher.py`` 的代码与文本里都不出现视觉概念（它只认识 ``TargetResolver``）。"""
    source = (get_project_root() / "src" / "action" / "dispatcher.py").read_text(encoding="utf-8")
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

    assert not names & {"TargetSnapshot", "VisionTargetCandidate", "vision_target"}

    # 连字符串都不出现（实现选择被放在 resolver 模块）
    for banned in ("VisionAware", "VisionTargetCandidate", "TargetSnapshot", "vision_target"):
        assert banned not in source, f"dispatcher.py 不应出现: {banned}"


def test_dispatcher_uses_injected_resolver_without_knowing_source() -> None:
    """Dispatcher 只依赖注入的 resolver 协议：换成视觉实现即可，无需改动调用方。"""
    resolver, _ = make_vision_resolver()
    dispatcher, _ = make_dispatcher(resolver)
    backend = dispatcher.backend
    assert isinstance(backend, DryRunBackend)

    # 未注入视觉结果 → 拒绝（stage=target）
    before = dispatcher.request(click_request())
    assert before.status is ActionStatus.REJECTED
    assert before.detail["stage"] == GuardStage.TARGET
    assert before.detail["resolution_reason"] == "target_not_found"

    # 外部注入视觉结果 → 同一 Dispatcher 立即可解析（无需重建、无需告知）
    resolver.update_candidates([make_candidate()])
    after = dispatcher.request(click_request())

    assert after.status is ActionStatus.SKIPPED  # dry-run：记录但不产生输入
    assert after.resolved_target is not None
    assert after.resolved_target.source == "vision"
    assert backend.count == 1
    assert backend.calls[0].point is not None
    # 客户区 (47, 473, 1704x960) × (0.82, 0.91) → (47+1704×0.82, 473+960×0.91)
    assert backend.calls[0].point.to_int_tuple() == (1444, 1347)


def test_dispatcher_set_resolver_switches_implementation() -> None:
    """``set_resolver`` 可在运行期切换实现（注册表 ↔ 视觉）。"""
    dispatcher, _ = make_dispatcher(RegistryTargetResolver(TargetRegistry.empty()))
    assert dispatcher.request(click_request()).status is ActionStatus.REJECTED

    vision, _ = make_vision_resolver()
    vision.update_candidates([make_candidate()])
    dispatcher.set_resolver(vision)

    outcome = dispatcher.request(click_request())
    assert outcome.status is ActionStatus.SKIPPED
    assert outcome.resolved_target is not None
    assert outcome.resolved_target.source == "vision"


def test_resolver_failure_is_rejected_not_failed() -> None:
    """Resolver 失败属于 target gate：``REJECTED``（不是 ``FAILED``、不抛异常）。"""

    class BrokenResolver:
        """故意抛异常的解析器。"""

        def resolve(self, target: object, /) -> object:
            """抛出异常。"""
            raise RuntimeError("resolver exploded")

    dispatcher, _ = make_dispatcher(BrokenResolver())

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.TARGET
    assert outcome.detail["resolution_reason"] == "resolver_error"


def test_guard_order_unchanged_with_vision_resolver() -> None:
    """5 道门顺序不变（E-STOP → 窗口 → 前台 → 目标 → 限流）；视觉解析不改变门控语义。"""
    probe = StaticWindowProbe(foreground=False)
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate()])
    dispatcher, _ = make_dispatcher(resolver, probe=probe)

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.FOREGROUND  # 前台先于目标


def test_vision_target_still_requires_valid_window() -> None:
    """窗口无效时，即使视觉目标可解析也不执行（门 2 先于解析）。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate()])
    dispatcher, _ = make_dispatcher(resolver, probe=StaticWindowProbe(valid=False))

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.WINDOW


def test_null_window_probe_keeps_vision_resolution_safe() -> None:
    """``NullWindowProbe`` 下即使有视觉候选也拒绝（安全默认不变）。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate()])
    dispatcher = ActionDispatcher(
        backend=DryRunBackend(),
        resolver=resolver,
        guard=SafetyGuard(window_probe=NullWindowProbe()),
    )

    outcome = dispatcher.request(click_request())

    assert outcome.status is ActionStatus.REJECTED
    assert outcome.detail["stage"] == GuardStage.WINDOW


# ---------------------------------------------------------------------------
# 8. Action 包依赖检查
# ---------------------------------------------------------------------------

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

ALLOWED_IMPORT_PREFIXES: tuple[str, ...] = ("src.core", "src.utils", "src.action")


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
    """``src/action`` 不得 import vision / evidence / runtime / task / cv2 / numpy 等。"""
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


def test_contract_module_has_no_vision_types() -> None:
    """契约模块（``src/core/vision_target.py``）的**代码**不含 Frame / numpy / 视觉层类型。"""
    path = get_project_root() / "src" / "core" / "vision_target.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    names: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
            names.add(node.module.split(".")[0])

    assert not names & {"Frame", "numpy", "cv2", "EvidenceResult", "DetectionResult"}
    # 项目内依赖只有 core 的几何类型（其余为 stdlib）
    assert {module for module in imported_modules if module.startswith("src.")} == {
        "src.core.geometry"
    }


# ---------------------------------------------------------------------------
# 9. 无真实输入 + 性能约束
# ---------------------------------------------------------------------------


def test_no_real_input_is_produced() -> None:
    """全流程（vision resolver + dry-run 后端）不产生任何真实输入。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate()])
    backend = DryRunBackend()
    dispatcher, _ = make_dispatcher(resolver, backend=backend)

    outcome = dispatcher.request(click_request())

    assert dispatcher.is_dry_run is True
    assert outcome.status is ActionStatus.SKIPPED
    assert backend.count == 1  # 只记录，不发送
    assert "dry-run" in outcome.reason


def test_resolution_is_constant_time_dict_lookup() -> None:
    """解析为 O(1)：字典查找 + 常数次比较，不触发检测 / 截图 / 模型调用。"""
    clock = FakeClock()
    resolver, _ = make_vision_resolver(clock=clock)
    candidates = [make_candidate(f"button_{index}") for index in range(500)]
    resolver.update_candidates(candidates)

    target = SemanticTarget.from_name("button_499")
    clock.reads = 0
    for _ in range(100):
        assert resolver.resolve(target) is not None

    # 每次解析只读一次时钟（TTL 判定），没有任何其它开销
    assert clock.reads == 100


def test_candidate_construction_does_not_require_frames() -> None:
    """契约只接受纯数据（无 Frame / 无图像）：可用最朴素的字面量构造。"""
    candidate = VisionTargetCandidate(
        target_name="reward_panel",
        x_ratio=0.5,
        y_ratio=0.82,
        confidence=0.9,
        bbox=Rect(x=700, y=700, width=200, height=80),
        source="detector",
        timestamp=123.0,
        metadata={"template_id": "button_reward_skin001"},
    )

    assert candidate.is_valid is True
    assert candidate.to_dict()["target_name"] == "reward_panel"
    assert "image" not in candidate.to_dict()

    snapshot = TargetSnapshot.create([candidate])
    assert snapshot.get("REWARD_PANEL") is not None


# ---------------------------------------------------------------------------
# 契约与换算工具
# ---------------------------------------------------------------------------


def test_ratio_from_rect_uses_center() -> None:
    """帧内矩形 → 比例（取中心点），与 bbox 语义一致。"""
    x_ratio, y_ratio = ratio_from_rect(
        Rect(x=100, y=200, width=200, height=100), Size(width=1000, height=500)
    )

    assert x_ratio == pytest.approx(0.2)
    assert y_ratio == pytest.approx(0.5)


def test_ratio_from_rect_rejects_bad_frame_size() -> None:
    """帧尺寸非正 → ``ValueError``（不做静默兜底）。"""
    with pytest.raises(ValueError):
        ratio_from_rect(Rect(x=0, y=0, width=10, height=10), Size(width=0, height=100))


def test_ratio_from_screen_point_matches_client_rect_mapping() -> None:
    """屏幕点 → 客户区比例，与 Dispatcher 的反向换算自洽。"""
    client_rect = Rect(x=47, y=473, width=1704, height=960)
    x_ratio, y_ratio = ratio_from_screen_point(Point(x=1495.4, y=1337.0), client_rect)

    assert x_ratio == pytest.approx(0.85, abs=1e-3)
    assert y_ratio == pytest.approx(0.90, abs=1e-3)

    resolved = ResolvedTarget(
        target="roundtrip", x_ratio=x_ratio, y_ratio=y_ratio, calibrated=True
    )
    assert resolved.client_point(client_rect).to_int_tuple() == (1495, 1337)


def test_rect_outside_frame_is_flagged_invalid() -> None:
    """窗口外的目标不会被静默裁剪，而是标记为非法并被拒绝。"""
    x_ratio, y_ratio = ratio_from_rect(
        Rect(x=1100, y=100, width=100, height=100), Size(width=1000, height=500)
    )
    candidate = VisionTargetCandidate(
        target_name="outside", x_ratio=x_ratio, y_ratio=y_ratio, confidence=0.9
    )
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([candidate])

    assert candidate.is_valid is False
    assert (
        resolver.resolve_detail(SemanticTarget.from_name("outside")).reason
        is TargetResolutionReason.INVALID_TARGET
    )


def test_target_resolution_serialization() -> None:
    """解析结果可序列化（不含图像 / 用户数据）。"""
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([make_candidate()])

    ok_payload = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button")).to_dict()
    bad_payload = resolver.resolve_detail(SemanticTarget.from_name("nope")).to_dict()

    assert ok_payload["ok"] is True
    assert "window_relative_position" in ok_payload["resolved"]
    assert bad_payload["ok"] is False
    assert bad_payload["reason"] == "target_not_found"
    assert "image" not in str(ok_payload)


def test_candidate_metadata_does_not_affect_resolution() -> None:
    """候选的 ``metadata`` 只作审计：携带模板名 / OCR 文本也不改变解析结果。"""
    candidate = VisionTargetCandidate(
        target_name="battle_auto_button",
        x_ratio=0.82,
        y_ratio=0.91,
        confidence=0.96,
        source="evidence",
        metadata={"template_id": "button_auto_skin001", "ocr_text": "自动", "group": "battle"},
    )
    resolver, _ = make_vision_resolver()
    resolver.update_candidates([candidate])

    resolution = resolver.resolve_detail(SemanticTarget.from_name("battle_auto_button"))

    assert resolution.ok is True
    assert resolution.resolved is not None
    assert resolution.resolved.x_ratio == pytest.approx(0.82)
    assert resolution.resolved.source == "vision"
    assert candidate.metadata["template_id"] == "button_auto_skin001"
