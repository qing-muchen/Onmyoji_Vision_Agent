"""Phase 5.4 测试：Runtime 可靠性与生命周期管理。

覆盖：

1. **状态转换合法**：``ALLOWED_TRANSITIONS`` 转换表、非法转换抛
   :class:`InvalidRuntimeTransition`、转换历史与幂等性；
2. **start / pause / resume / stop**：``StateMonitor`` 生命周期入口（暂停期间不处理新帧）；
3. **capture failure recovery**：退避 + 次数上限 + 用尽后结束运行（``FAILED``）；
4. **资源释放**：``stop()`` 关闭采集后端（dxcam/mss）、幂等、异常安全；
5. **window lost recovery**：窗口丢失 → ``RECOVERING`` → 恢复 → ``RUNNING``；
6. **metrics 兼容**：既有键名与含义不变，Phase 5.4 字段为追加；
7. **旧 demo 不受影响**：``live_runtime_demo`` 仍可装配、CLI 参数向后兼容。

本文件不依赖真实屏幕、真实等待与真实窗口。
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from src.core.config import (
    RuntimeRecoveryConfig,
    RuntimeStabilizationConfig,
    RuntimeStateMonitorConfig,
    load_config,
)
from src.core.exceptions import CaptureError, WindowNotFoundError
from src.core.geometry import Rect
from src.runtime.state_monitor import (
    ALLOWED_TRANSITIONS,
    CycleScheduler,
    CycleStatus,
    InvalidRuntimeTransition,
    LifecycleTransition,
    RecoveryAction,
    RecoveryManager,
    RecoveryPolicy,
    RuntimeCaptureBackend,
    RuntimeLifecycle,
    RuntimeMetricsRecorder,
    RuntimeState,
    SkipReason,
    StateMonitor,
    StateStabilizer,
    WindowCaptureTarget,
    WindowFrameSource,
)
from src.runtime.state_monitor import capture_backend as capture_backend_module
from src.vision.capture import CaptureBackend, Frame
from src.vision.state import StateName
from src.vision.window import WindowInfo
from tests.monitor_fakes import FakeClassifier, FakeClock

#: 测试用周期（毫秒）
INTERVAL_MS = 500
#: 固定的状态历史 / 事件时间戳
FIXED_TIMESTAMP = "2025-01-01T00:00:00+08:00"


def make_frame() -> Frame:
    """构造一个极小的测试帧。"""
    return Frame.create(np.zeros((8, 8, 3), dtype=np.uint8), source="fake")


def make_window(*, hwnd: int = 0x5150, x: int = 20, y: int = 30) -> WindowInfo:
    """构造一个假窗口。"""
    return WindowInfo(hwnd=hwnd, title="阴阳师-生命周期测试", x=x, y=y, width=800, height=600)


class LifecycleBinding:
    """``OnmyojiWindowSource`` 的替身（可切换窗口存活状态）。"""

    def __init__(self, window: WindowInfo | None = None, *, alive: bool = True) -> None:
        self._window = window
        self.alive = alive
        self.calls: dict[str, int] = {"find": 0, "require": 0, "clear": 0}

    @property
    def window(self) -> WindowInfo | None:
        """当前绑定窗口。"""
        return self._window

    @property
    def keywords(self) -> tuple[str, ...]:
        """标题关键词。"""
        return ("阴阳师", "Onmyoji")

    @property
    def match_mode(self) -> str:
        """标题匹配模式。"""
        return "contains"

    def find(self, *, refresh: bool = False) -> WindowInfo | None:
        """重新查找（不重建绑定，便于模拟"窗口消失后仍未回来"）。"""
        self.calls["find"] += 1
        return self._window if self.alive else None

    def require(self) -> WindowInfo:
        """读回最新几何。"""
        self.calls["require"] += 1
        if self._window is None or not self.alive:
            raise WindowNotFoundError("窗口不存在")
        return self._window

    def clear(self) -> None:
        """清除绑定。"""
        self.calls["clear"] += 1
        self._window = None

    def retarget(self, window: WindowInfo) -> None:
        """直接绑定窗口。"""
        self._window = window
        self.alive = True

    def is_alive(self) -> bool:
        """是否仍可见。"""
        return self._window is not None and self.alive

    def is_minimized(self) -> bool:
        """是否最小化。"""
        return False


class TrackedBackend(CaptureBackend):
    """记录启动/停止与抓帧次数的后端替身。"""

    name = "tracked"

    def __init__(self, *, region: Rect, fail: bool = False) -> None:
        super().__init__(region=region, fps=1)
        self.starts = 0
        self.stops = 0
        self.grabs = 0
        self.fail = fail

    def _open(self) -> None:
        """记录启动。"""
        self.starts += 1

    def _close(self) -> None:
        """记录停止（资源释放）。"""
        self.stops += 1

    def _grab_image(self) -> np.ndarray | None:
        """按配置失败或返回全黑图像。"""
        self.grabs += 1
        if self.fail:
            raise CaptureError("模拟设备丢失")
        return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def tracked_backends(monkeypatch: pytest.MonkeyPatch) -> list[TrackedBackend]:
    """把采集工厂替换为 :class:`TrackedBackend`，返回创建记录。"""
    created: list[TrackedBackend] = []

    def factory(
        config: object, *, window: WindowInfo | None = None, transform: object | None = None
    ) -> TrackedBackend:
        region = window.rect if window is not None else Rect(x=0, y=0, width=8, height=8)
        backend = TrackedBackend(region=region)
        created.append(backend)
        return backend

    monkeypatch.setattr(capture_backend_module, "create_capture_backend", factory)
    return created


class CountingSource:
    """记录取帧次数的帧来源替身（可选总是返回 ``None`` 以模拟采集失败）。

    实现 :class:`~src.runtime.state_monitor.source.StatusAwareSource`
    （``last_skip_reason``），因此恢复原因会带上 ``no_frame``。
    """

    def __init__(self, *, frames: bool = True, fail_first: int = 0) -> None:
        self.calls = 0
        self.starts = 0
        self.stops = 0
        self._frames = frames
        self._fail_first = fail_first
        self._last_skip_reason: str | None = None

    def start(self) -> None:
        """记录启动。"""
        self.starts += 1

    def stop(self) -> None:
        """记录停止（资源释放）。"""
        self.stops += 1

    @property
    def last_skip_reason(self) -> str | None:
        """最近一次跳过原因（成功时为 ``None``）。"""
        return self._last_skip_reason

    def get_frame(self) -> Frame | None:
        """返回帧或 ``None``（前 ``fail_first`` 次失败）。"""
        self.calls += 1
        if self.calls <= self._fail_first or not self._frames:
            self._last_skip_reason = SkipReason.NO_FRAME.value
            return None
        self._last_skip_reason = None
        return make_frame()


def make_source(
    binding: LifecycleBinding,
    *,
    clock: FakeClock,
    retry_interval_s: float = 1.0,
) -> WindowFrameSource:
    """构造真实窗口帧来源（采集工厂由 autouse fixture 替换）。"""
    config = load_config(environ={})
    return WindowFrameSource(
        binding,  # type: ignore[arg-type]
        capture=config.capture,
        calibration=config.calibration,
        guard_enabled=False,
        retry_interval_s=retry_interval_s,
        clock=clock.clock,
    )


def build_monitor(
    tmp_path: Path,
    *,
    source: object,
    metrics: RuntimeMetricsRecorder | None = None,
    clock: FakeClock | None = None,
    recovery: RecoveryManager | None = None,
    lifecycle: RuntimeLifecycle | None = None,
    states: Sequence[StateName] = (StateName.YARD,),
    on_cycle: object | None = None,
) -> tuple[StateMonitor, RuntimeMetricsRecorder, FakeClock]:
    """装配带指标与生命周期的监控器（不写历史文件）。"""
    active_clock = clock if clock is not None else FakeClock()
    recorder = metrics if metrics is not None else RuntimeMetricsRecorder()
    monitor = StateMonitor(
        FakeClassifier(list(states)),  # type: ignore[arg-type]
        source,  # type: ignore[arg-type]
        RuntimeStateMonitorConfig(
            interval_ms=INTERVAL_MS,
            save_history=False,
            history_path=str(tmp_path / "state_history.json"),
        ),
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(min_frames=1)),
        scheduler=CycleScheduler(INTERVAL_MS, clock=active_clock.clock, sleep=active_clock.sleep),
        metrics=recorder,
        lifecycle=lifecycle,
        recovery=recovery,
        clock=active_clock.clock,
        sleep=active_clock.sleep,
        now=lambda: FIXED_TIMESTAMP,
        on_cycle=on_cycle,  # type: ignore[arg-type]
    )
    return monitor, recorder, active_clock


def simple_recovery(
    *,
    max_retry: int = 3,
    backoff_s: float = 0.1,
    clock: FakeClock | None = None,
) -> RecoveryManager:
    """构造启用状态下的恢复管理器（退避用假时钟，测试不真实等待）。"""
    active_clock = clock if clock is not None else FakeClock()
    return RecoveryManager(
        RecoveryPolicy(
            enabled=True, max_retry=max_retry, backoff_s=backoff_s, backoff_factor=2.0
        ),
        sleep=active_clock.sleep,
    )


# ---------------------------------------------------------------------------
# 1. 状态转换合法
# ---------------------------------------------------------------------------


def test_lifecycle_starts_initializing() -> None:
    """新建的生命周期处于 ``INITIALIZING``，且没有转换历史。"""
    lifecycle = RuntimeLifecycle()

    assert lifecycle.state is RuntimeState.INITIALIZING
    assert lifecycle.transitions == ()
    assert lifecycle.transition_count == 0
    assert lifecycle.is_terminal is False


def test_lifecycle_happy_path_transitions() -> None:
    """initialize → start → pause → resume → stop 的转换序列明确。"""
    lifecycle = RuntimeLifecycle(now=lambda: FIXED_TIMESTAMP)

    assert lifecycle.initialize() is RuntimeState.READY
    assert lifecycle.start() is RuntimeState.RUNNING
    assert lifecycle.pause() is RuntimeState.PAUSED
    assert lifecycle.resume() is RuntimeState.RUNNING
    assert lifecycle.stop() is RuntimeState.STOPPED

    assert [item.current for item in lifecycle.transitions] == [
        RuntimeState.READY,
        RuntimeState.RUNNING,
        RuntimeState.PAUSED,
        RuntimeState.RUNNING,
        RuntimeState.STOPPED,
    ]
    assert all(isinstance(item, LifecycleTransition) for item in lifecycle.transitions)
    assert lifecycle.transitions[0].reason == "initialize"
    assert lifecycle.is_terminal is True


def test_lifecycle_rejects_invalid_transition() -> None:
    """非法转换（例如 ``INITIALIZING`` → ``PAUSED``）抛异常且不改变状态。"""
    lifecycle = RuntimeLifecycle()

    with pytest.raises(InvalidRuntimeTransition):
        lifecycle.pause()

    assert lifecycle.state is RuntimeState.INITIALIZING
    assert lifecycle.can_transition_to(RuntimeState.PAUSED) is False
    assert lifecycle.can_transition_to(RuntimeState.READY) is True


@pytest.mark.parametrize(
    ("source", "target", "allowed"),
    [
        (RuntimeState.INITIALIZING, RuntimeState.READY, True),
        (RuntimeState.INITIALIZING, RuntimeState.RUNNING, False),
        (RuntimeState.READY, RuntimeState.RUNNING, True),
        (RuntimeState.RUNNING, RuntimeState.PAUSED, True),
        (RuntimeState.RUNNING, RuntimeState.RECOVERING, True),
        (RuntimeState.RUNNING, RuntimeState.READY, True),
        (RuntimeState.PAUSED, RuntimeState.RECOVERING, False),
        (RuntimeState.RECOVERING, RuntimeState.RUNNING, True),
        (RuntimeState.RECOVERING, RuntimeState.FAILED, True),
        (RuntimeState.STOPPED, RuntimeState.RUNNING, False),
        (RuntimeState.STOPPED, RuntimeState.INITIALIZING, True),
        (RuntimeState.FAILED, RuntimeState.INITIALIZING, True),
    ],
)
def test_transition_table_is_explicit(
    source: RuntimeState, target: RuntimeState, allowed: bool
) -> None:
    """转换表逐条锁定（唯一事实来源）。"""
    assert (target in ALLOWED_TRANSITIONS[source]) is allowed


def test_lifecycle_stop_is_idempotent_and_restartable() -> None:
    """``stop()`` 幂等；``STOPPED`` 可通过 ``initialize()`` 重新进入 READY。"""
    lifecycle = RuntimeLifecycle()
    lifecycle.initialize()
    lifecycle.start()

    assert lifecycle.stop() is RuntimeState.STOPPED
    assert lifecycle.stop() is RuntimeState.STOPPED
    assert lifecycle.transition_count == 3

    assert lifecycle.initialize() is RuntimeState.READY
    assert lifecycle.start() is RuntimeState.RUNNING


def test_lifecycle_recovery_flow_and_failure() -> None:
    """恢复流程：``RUNNING`` → ``RECOVERING`` → ``RUNNING``；用尽 → ``FAILED``。"""
    lifecycle = RuntimeLifecycle()
    lifecycle.start()

    assert lifecycle.begin_recovery("capture_failed") is RuntimeState.RECOVERING
    assert lifecycle.finish_recovery() is RuntimeState.RUNNING
    assert lifecycle.begin_recovery("window_lost") is RuntimeState.RECOVERING
    assert lifecycle.fail(reason="retry exhausted") is RuntimeState.FAILED
    assert lifecycle.is_terminal is True
    assert lifecycle.last_transition is not None
    assert lifecycle.last_transition.reason == "retry exhausted"


# ---------------------------------------------------------------------------
# 2. start / pause / resume / stop
# ---------------------------------------------------------------------------


def test_monitor_lifecycle_entrypoints(tmp_path: Path) -> None:
    """``start`` / ``pause`` / ``resume`` / ``stop`` 驱动状态并可查询。"""
    monitor, _, _ = build_monitor(tmp_path, source=CountingSource())
    assert monitor.runtime_state is RuntimeState.INITIALIZING

    assert monitor.initialize() is RuntimeState.READY
    assert monitor.start() is RuntimeState.RUNNING
    assert monitor.pause() is RuntimeState.PAUSED
    assert monitor.resume() is RuntimeState.RUNNING
    assert monitor.stop() is RuntimeState.STOPPED

    assert monitor.lifecycle.state is RuntimeState.STOPPED
    assert monitor.stop_requested is True
    assert len(monitor.lifecycle_transitions) >= 5


def test_monitor_pause_skips_frame_processing(tmp_path: Path) -> None:
    """暂停后不再取帧、不再计轮（节拍继续，直到 duration 到期）。"""
    source = CountingSource()
    holder: dict[str, StateMonitor] = {}

    def on_cycle(_schedule: object) -> None:
        monitor = holder["monitor"]
        if monitor.cycles == 1:
            monitor.pause()

    monitor, recorder, clock = build_monitor(tmp_path, source=source, on_cycle=on_cycle)
    holder["monitor"] = monitor
    report = monitor.run(duration_s=2.0)

    assert report.cycles == 1  # 暂停后不再处理新帧
    assert source.calls == 1
    assert monitor.paused_cycles >= 1  # 节拍仍在空转
    assert monitor.runtime_state is RuntimeState.PAUSED
    assert clock.value >= 2.0
    assert recorder.summary().cycles == 1


def test_monitor_resume_continues_after_pause(tmp_path: Path) -> None:
    """``resume()`` 后可继续运行（暂停不改变 ``run()`` / ``step()`` 语义）。"""
    source = CountingSource()
    holder: dict[str, StateMonitor] = {}
    paused_once = {"done": False}

    def on_cycle(_schedule: object) -> None:
        monitor = holder["monitor"]
        if not paused_once["done"] and monitor.cycles == 1:
            paused_once["done"] = True
            monitor.pause()

    monitor, _, _ = build_monitor(tmp_path, source=source, on_cycle=on_cycle)
    holder["monitor"] = monitor
    monitor.run(duration_s=1.0)
    assert monitor.runtime_state is RuntimeState.PAUSED

    monitor.resume()
    second = monitor.run(duration_s=1.0)

    assert monitor.runtime_state is RuntimeState.READY  # 运行结束后回到 READY
    assert second.cycles >= 1
    assert source.calls > 1


def test_monitor_run_after_stop_still_works(tmp_path: Path) -> None:
    """``stop()`` 之后再次 ``run()`` 仍然可用（生命周期可重启）。"""
    monitor, _, _ = build_monitor(tmp_path, source=CountingSource())
    monitor.start()
    monitor.stop()
    assert monitor.runtime_state is RuntimeState.STOPPED

    report = monitor.run(max_cycles=2)

    assert report.cycles == 2
    assert monitor.runtime_state is RuntimeState.READY


def test_monitor_run_transitions_to_running(tmp_path: Path) -> None:
    """``run()`` 期间运行时状态为 ``RUNNING``（直接构造的监控器也会自动初始化）。"""
    monitor, recorder, _ = build_monitor(tmp_path, source=CountingSource())

    monitor.run(max_cycles=1)

    assert recorder.records[0].runtime_state == RuntimeState.RUNNING.value
    assert monitor.runtime_state is RuntimeState.READY


# ---------------------------------------------------------------------------
# 3. capture failure recovery
# ---------------------------------------------------------------------------


def test_recovery_policy_backoff_is_capped() -> None:
    """退避按倍数增长并封顶。"""
    policy = RecoveryPolicy(enabled=True, max_retry=5, backoff_s=0.5, backoff_factor=2.0, max_backoff_s=2.0)

    assert policy.unlimited is False
    assert policy.delay_for(1) == pytest.approx(0.5)
    assert policy.delay_for(2) == pytest.approx(1.0)
    assert policy.delay_for(3) == pytest.approx(2.0)
    assert policy.delay_for(4) == pytest.approx(2.0)  # 封顶


def test_recovery_manager_retries_then_exhausts() -> None:
    """连续失败按上限重试，超出后 ``EXHAUSTED``（不会死循环）。"""
    clock = FakeClock()
    manager = simple_recovery(max_retry=2, backoff_s=0.5, clock=clock)

    first = manager.register_failure("capture_failed")
    second = manager.register_failure("capture_failed")
    third = manager.register_failure("capture_failed")

    assert first.action is RecoveryAction.RETRY
    assert second.action is RecoveryAction.RETRY
    assert third.action is RecoveryAction.EXHAUSTED
    assert manager.attempts == 3
    assert manager.count == 3
    assert manager.is_exhausted is True
    assert clock.value == pytest.approx(1.5)  # 0.5 + 1.0（第三次不再等待）


def test_recovery_manager_unlimited_policy_never_exhausts() -> None:
    """``max_retry=0`` 表示不放弃（仍按最大退避等待，不忙等）。"""
    manager = simple_recovery(max_retry=0, backoff_s=0.25)

    for _ in range(10):
        decision = manager.register_failure("window_not_found")

    assert decision.action is RecoveryAction.RETRY
    assert decision.delay_s == pytest.approx(8.0)  # 默认封顶
    assert manager.is_exhausted is False


def test_recovery_manager_disabled_is_noop() -> None:
    """未启用策略时不做任何等待 / 计数（既有行为不变）。"""
    clock = FakeClock()
    manager = RecoveryManager(RecoveryPolicy(), sleep=clock.sleep)

    decision = manager.register_failure("capture_failed")

    assert decision.action is RecoveryAction.NONE
    assert manager.attempts == 0
    assert manager.count == 0
    assert clock.value == 0.0


def test_recovery_manager_success_clears_attempts() -> None:
    """成功后连续失败计数清零，并报告"从恢复中恢复"。"""
    manager = simple_recovery(max_retry=3)
    manager.register_failure("capture_failed")

    recovered = manager.register_success()

    assert recovered is True
    assert manager.attempts == 0
    assert manager.register_success() is False


def test_monitor_recovery_exhausted_stops_run(tmp_path: Path) -> None:
    """恢复用尽 → 结束本次运行并进入 ``FAILED``（stop_reason 可见）。"""
    source = CountingSource(frames=False)  # 永远采不到帧
    clock = FakeClock()
    monitor, recorder, _ = build_monitor(
        tmp_path,
        source=source,
        clock=clock,
        recovery=simple_recovery(max_retry=2, backoff_s=0.5, clock=clock),
    )

    report = monitor.run()

    from src.runtime.state_monitor import StopReason

    assert report.stop_reason is StopReason.RECOVERY_EXHAUSTED
    assert report.cycles == 3  # 2 次重试 + 第 3 次用尽
    assert monitor.runtime_state is RuntimeState.FAILED
    assert monitor.recovery.count == 3
    assert monitor.recovery.total_backoff_s == pytest.approx(1.5)
    assert recorder.records[-1].last_recovery_reason == SkipReason.NO_FRAME.value
    assert recorder.records[-1].runtime_state == RuntimeState.FAILED.value


def test_monitor_recovery_returns_to_running_after_success(tmp_path: Path) -> None:
    """失败后恢复成功 → 生命周期回到 ``RUNNING``，恢复计数保留。"""
    source = CountingSource(fail_first=2)
    clock = FakeClock()
    monitor, recorder, _ = build_monitor(
        tmp_path,
        source=source,
        clock=clock,
        recovery=simple_recovery(max_retry=3, backoff_s=0.25, clock=clock),
    )

    report = monitor.run(max_cycles=4)

    assert report.skipped == 2
    assert report.classified == 2
    assert monitor.recovery.attempts == 0  # 已恢复
    assert monitor.recovery.count == 2  # 两次失败各记一次恢复
    assert monitor.runtime_state is RuntimeState.READY
    states = [record.runtime_state for record in recorder.records]
    assert RuntimeState.RECOVERING.value in states
    assert states[-1] == RuntimeState.RUNNING.value


def test_recovery_policy_from_config() -> None:
    """配置 → 策略映射（``runtime.recovery``）。"""
    config = RuntimeRecoveryConfig(
        enabled=True, max_retry=5, backoff_s=0.2, backoff_factor=3.0, max_backoff_s=4.0
    )

    policy = RecoveryPolicy.from_config(config)

    assert policy.enabled is True
    assert policy.max_retry == 5
    assert policy.delay_for(2) == pytest.approx(0.6)
    assert "enabled=True" in policy.describe()


def test_monitor_from_config_uses_recovery_config(tmp_path: Path) -> None:
    """``from_config`` 按 ``runtime.recovery`` 装配恢复策略（默认关闭）。"""
    config = load_config(environ={})
    default_monitor = StateMonitor.from_config(config, source=CountingSource())
    assert default_monitor.recovery.enabled is False

    enabled_config = config.model_copy(
        update={"runtime": config.runtime.model_copy(
            update={
                "recovery": RuntimeRecoveryConfig(
                    enabled=True, max_retry=1, backoff_s=0.0, max_backoff_s=0.0
                )
            }
        )}
    )
    enabled_monitor = StateMonitor.from_config(enabled_config, source=CountingSource())
    assert enabled_monitor.recovery.enabled is True
    assert enabled_monitor.recovery.policy.max_retry == 1


# ---------------------------------------------------------------------------
# 4. 资源释放
# ---------------------------------------------------------------------------


def test_stop_releases_capture_backend(tmp_path: Path, tracked_backends: list[TrackedBackend]) -> None:
    """``stop()`` 关闭采集后端（后端 ``stop()`` 被调用），且幂等。"""
    binding = LifecycleBinding(window=make_window())
    source = make_source(binding, clock=FakeClock())
    monitor, _, _ = build_monitor(tmp_path, source=source)

    monitor.initialize()  # 启动帧来源 → 创建后端
    assert source.backend.is_ready is True
    assert len(tracked_backends) == 1
    backend = tracked_backends[0]
    assert backend.starts == 1

    monitor.stop()
    assert backend.stops == 1
    assert source.backend.is_ready is False
    assert source.is_running is False

    monitor.stop()  # 幂等：不重复释放
    assert backend.stops == 1


def test_source_stop_is_exception_safe(tmp_path: Path) -> None:
    """帧来源 ``stop()`` 抛异常时不影响 ``StateMonitor.stop()``。"""

    class ExplodingSource(CountingSource):
        def stop(self) -> None:
            """模拟清理时报错。"""
            raise OSError("模拟清理失败")

    source = ExplodingSource()
    monitor, _, _ = build_monitor(tmp_path, source=source)

    assert monitor.stop() is RuntimeState.STOPPED
    assert source.stops == 0  # 抛异常前未计数（异常被吞掉并记录）


def test_runtime_capture_backend_release_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RuntimeCaptureBackend.release()`` 幂等，且区域变化时才重建。"""
    created: list[TrackedBackend] = []

    def factory(
        config: object, *, window: WindowInfo | None = None, transform: object | None = None
    ) -> TrackedBackend:
        region = window.rect if window is not None else Rect(x=0, y=0, width=8, height=8)
        backend = TrackedBackend(region=region)
        created.append(backend)
        return backend

    monkeypatch.setattr(capture_backend_module, "create_capture_backend", factory)
    config = load_config(environ={})
    handle = RuntimeCaptureBackend(config.capture)
    window = make_window()

    first = handle.ensure(window=window, region=window.rect, transform=None)
    second = handle.ensure(window=window, region=window.rect, transform=None)

    assert first.ok is True
    assert second.ok is True
    assert len(created) == 1  # 区域未变 → 不重建
    assert handle.rebuilds == 1

    moved = make_window(x=500, y=600)
    third = handle.ensure(window=moved, region=moved.rect, transform=None)

    assert third.ok is True
    assert len(created) == 2
    assert created[0].stops == 1  # 旧后端被释放

    handle.release()
    handle.release()
    assert created[1].stops == 1
    assert handle.is_ready is False


def test_window_capture_target_detects_resize() -> None:
    """``WindowCaptureTarget`` 记录区域变化（窗口移动 / 缩放检测）。"""
    binding = LifecycleBinding(window=make_window(x=10, y=10))
    config = load_config(environ={})
    target = WindowCaptureTarget(binding, calibration=config.calibration)  # type: ignore[arg-type]

    first = target.resolve()
    second = target.resolve()
    binding.retarget(make_window(x=400, y=300))
    third = target.resolve()

    assert first.ok and second.ok and third.ok
    assert target.resizes == 2  # 首次 + 移动后
    assert target.region is not None
    assert target.region_changed(None) is True
    assert target.region_changed(third.region) is False
    assert first.region != third.region


# ---------------------------------------------------------------------------
# 5. window lost recovery
# ---------------------------------------------------------------------------


def test_window_lost_triggers_recovery_then_recovers(tmp_path: Path) -> None:
    """窗口丢失 → ``RECOVERING``；窗口回来后 → ``RUNNING``。"""
    window = make_window()
    binding = LifecycleBinding(window=window)
    clock = FakeClock()
    source = make_source(binding, clock=clock, retry_interval_s=0.0)
    holder: dict[str, StateMonitor] = {}

    # 第 2 轮起窗口丢失，第 4 轮恢复
    def on_cycle(_schedule: object) -> None:
        if holder["monitor"].cycles == 1:
            binding.alive = False
        elif holder["monitor"].cycles == 3:
            binding.alive = True
            binding.retarget(window)

    monitor, recorder, _ = build_monitor(
        tmp_path,
        source=source,
        clock=clock,
        recovery=simple_recovery(max_retry=5, backoff_s=0.25, clock=clock),
        on_cycle=on_cycle,
    )
    holder["monitor"] = monitor
    monitor.start()  # 启动帧来源（StateMonitor.start() 会调用 source.start()）

    report = monitor.run(max_cycles=5)

    assert report.skipped >= 1
    states = [record.runtime_state for record in recorder.records]
    assert RuntimeState.RECOVERING.value in states
    assert monitor.recovery.count >= 1
    assert monitor.recovery.attempts == 0  # 最终恢复
    assert monitor.runtime_state is RuntimeState.READY  # 运行正常结束


def test_window_lost_exhausts_when_never_returns(tmp_path: Path) -> None:
    """窗口一直不回来 → 重试用尽 → ``FAILED``（不会无限死循环）。"""
    binding = LifecycleBinding(window=make_window(), alive=False)
    clock = FakeClock()
    source = make_source(binding, clock=clock, retry_interval_s=0.0)
    monitor, _, _ = build_monitor(
        tmp_path,
        source=source,
        clock=clock,
        recovery=simple_recovery(max_retry=2, backoff_s=0.1, clock=clock),
    )
    monitor.start()

    report = monitor.run()

    from src.runtime.state_monitor import StopReason

    assert report.stop_reason is StopReason.RECOVERY_EXHAUSTED
    assert monitor.runtime_state is RuntimeState.FAILED
    assert monitor.recovery.is_exhausted is True
    assert report.cycles == 3


# ---------------------------------------------------------------------------
# 6. metrics 兼容
# ---------------------------------------------------------------------------

#: Phase 5.2 既有字段（名称与含义必须保持不变）
LEGACY_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "cycle_id",
        "timestamp",
        "state",
        "confidence",
        "stable_state",
        "latency_ms",
        "frame_ms",
        "status",
        "skip_reason",
        "frame_source",
    }
)
#: Phase 5.4 追加字段
LIFECYCLE_METRIC_KEYS: frozenset[str] = frozenset(
    {"runtime_state", "recovery_count", "last_recovery_reason"}
)


def test_metrics_keep_legacy_keys_and_add_lifecycle(tmp_path: Path) -> None:
    """JSONL 既有键名不变，新增生命周期 / 恢复字段。"""
    metrics_path = tmp_path / "metrics.jsonl"
    recorder = RuntimeMetricsRecorder(path=metrics_path)
    monitor, _, _ = build_monitor(tmp_path, source=CountingSource(), metrics=recorder)

    monitor.run(max_cycles=2)

    payloads = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 2
    for payload in payloads:
        assert LEGACY_METRIC_KEYS <= set(payload)
        assert LIFECYCLE_METRIC_KEYS <= set(payload)
        assert payload["runtime_state"] == RuntimeState.RUNNING.value
        assert payload["recovery_count"] == 0
        assert payload["last_recovery_reason"] is None
        # 不记录截图 / 用户数据
        assert "image" not in payload


def test_metrics_records_recovery_reason(tmp_path: Path) -> None:
    """恢复原因写入指标（便于长跑后统计"为什么恢复"）。"""
    recorder = RuntimeMetricsRecorder()
    clock = FakeClock()
    monitor, _, _ = build_monitor(
        tmp_path,
        source=CountingSource(fail_first=1),
        metrics=recorder,
        clock=clock,
        recovery=simple_recovery(max_retry=3, backoff_s=0.1, clock=clock),
    )

    monitor.run(max_cycles=2)

    first = recorder.records[0]
    assert first.runtime_state == RuntimeState.RECOVERING.value
    assert first.recovery_count == 1
    assert first.last_recovery_reason is not None
    assert recorder.records[-1].recovery_count == 1  # 恢复成功后计数保留


def test_metrics_legacy_construction_still_valid(tmp_path: Path) -> None:
    """不传新字段构造 ``CycleMetrics`` 仍然可用。"""
    from src.runtime.state_monitor import CycleMetrics

    metrics = CycleMetrics(
        cycle_id=1,
        timestamp=FIXED_TIMESTAMP,
        started_at=0.0,
        latency_ms=1.0,
        frame_ms=0.5,
        state=StateName.YARD,
        confidence=0.9,
        stable_state=StateName.YARD,
        status=CycleStatus.OK,
    )

    payload = metrics.to_dict()

    assert LEGACY_METRIC_KEYS <= set(payload)
    assert payload["runtime_state"] is None
    assert payload["recovery_count"] == 0
    assert payload["last_recovery_reason"] is None
    assert payload["capture_status"] is None


# ---------------------------------------------------------------------------
# 7. 旧 demo 不受影响
# ---------------------------------------------------------------------------


def test_live_demo_cli_is_backward_compatible() -> None:
    """``live_runtime_demo`` 仍可导入，既有 CLI 参数含义不变（Phase 5.4 只新增）。"""
    demo = importlib.import_module("examples.live_runtime_demo")
    parser = demo.build_parser()

    legacy = parser.parse_args(
        ["--duration", "10", "--max-cycles", "5", "--no-metrics", "--allow-background"]
    )
    assert legacy.duration == 10.0
    assert legacy.max_cycles == 5
    assert legacy.no_metrics is True
    assert legacy.allow_background is True
    assert legacy.interval_ms is None
    assert legacy.timing is False

    # Phase 5.4 新增开关（默认关闭 = 旧行为）
    lifecycle = parser.parse_args([])
    assert getattr(lifecycle, "lifecycle_demo", False) is False
    assert getattr(lifecycle, "recovery", False) is False
    assert getattr(lifecycle, "recovery_max_retry", 3) == 3


def test_default_recovery_keeps_previous_behaviour(tmp_path: Path) -> None:
    """恢复策略默认关闭：采集失败只记 ``skipped`` 并继续（Phase 5.2/5.3 行为）。"""
    source = CountingSource(frames=False)
    monitor, recorder, _ = build_monitor(tmp_path, source=source)

    report = monitor.run(max_cycles=4)

    assert report.cycles == 4
    assert report.skipped == 4
    assert monitor.recovery.enabled is False
    assert monitor.recovery.count == 0
    assert monitor.runtime_state is RuntimeState.READY


def test_demo_print_helpers_still_work() -> None:
    """demo 的逐轮打印辅助函数仍接受既有参数（含 Phase 5.4 生命周期状态）。"""
    demo = importlib.import_module("examples.live_runtime_demo")
    from src.runtime.state_monitor import CycleMetrics

    clock = FakeClock()
    scheduler = CycleScheduler(INTERVAL_MS, clock=clock.clock, sleep=clock.sleep)
    started = scheduler.begin()
    schedule = scheduler.finish(started)
    metrics = CycleMetrics(
        cycle_id=1,
        timestamp=FIXED_TIMESTAMP,
        started_at=0.0,
        latency_ms=1.0,
        frame_ms=0.5,
        state=StateName.YARD,
        confidence=0.9,
        stable_state=None,
        status=CycleStatus.OK,
        runtime_state=RuntimeState.RUNNING.value,
    )

    demo.print_cycle_line(metrics, schedule, None)  # 不应抛异常

    assert schedule.cycle_id == 1


def test_examples_import_is_safe() -> None:
    """``examples.live_runtime_demo`` 可被安全导入（不触发采集 / 不要求真实窗口）。"""
    module = importlib.import_module("examples.live_runtime_demo")

    assert hasattr(module, "main")
    assert callable(module.build_parser)
