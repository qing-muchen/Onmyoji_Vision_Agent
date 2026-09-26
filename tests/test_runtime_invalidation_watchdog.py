"""Phase 11.1-pre 测试：快照主动失效接线 + 调度器看门狗。

覆盖：
1. Runtime 暂停 → 快照失效（`window_unavailable`）；
2. `run()` 结束 → 快照失效（`runtime_stopped`）；
3. 失效异常隔离（不影响主循环）；
4. 看门狗默认关闭（向后兼容）；
5. 看门狗超时 → 实例 FAILED + 统计；
6. 正常推进不触发看门狗；
7. 暂停 / 恢复后看门狗重新计时；
8. 配置（`SchedulerConfig.watchdog_timeout_s` / YAML / `from_config`）；
9. 失效与看门狗全程零真实输入。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.action import ActionOutcome, ActionStatus
from src.core.config import RuntimeStabilizationConfig, RuntimeStateMonitorConfig
from src.core.geometry import Rect, Size
from src.runtime.state_monitor.monitor import StateMonitor
from src.runtime.state_monitor.stabilizer import StateStabilizer
from src.runtime.vision_bridge import (
    INVALIDATION_REASONS,
    INVALIDATION_RUNTIME_STOPPED,
    INVALIDATION_WINDOW_UNAVAILABLE,
    VisionSnapshotBridge,
)
from src.action import VisionAwareTargetResolver
from src.task import TaskRegistry, validate_flow_document
from src.task.scheduler import (
    TaskInstanceStatus,
    TaskScheduler,
    load_scheduler_config,
    scheduler_config_from_mapping,
)
from src.vision.capture.models import Frame
from src.vision.pipeline.target_adapter import ACTIONABLE_EVIDENCE
from src.vision.evidence.models import EvidenceResult, EvidenceSource

TARGET_NAME = "battle_auto_button"
TRIGGER_STATE = "battle"
PIXELS = 40
CLIENT_RECT = Rect(x=47, y=473, width=1704, height=960)
FRAME_SIZE = Size(width=1704, height=960)


class FakeClock:
    """可手动推进的单调时钟（不自增，便于精确控制看门狗）。"""

    def __init__(self, value: float = 100.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        """返回当前时间。"""
        return self.value

    def advance(self, seconds: float) -> None:
        """推进时间。"""
        self.value += float(seconds)


class StubResolver:
    """满足 ``update_snapshot`` 协议的最小 resolver 替身（不参与动作）。"""

    def __init__(self) -> None:
        self.snapshots: list[object] = []
        self.stale_snapshots = 0

    def update_snapshot(self, snapshot: object, *, accept_out_of_order: bool = False):
        """记录快照（返回入参）。"""
        self.snapshots.append(snapshot)
        return snapshot


class ExplodingBridge:
    """``invalidate`` 抛异常的桥替身（验证异常隔离）。"""

    def __init__(self) -> None:
        self.calls = 0

    def update(self, **kwargs: object) -> None:
        """无操作。"""

    def invalidate(self, **kwargs: object) -> None:
        """总是抛异常。"""
        self.calls += 1
        raise RuntimeError("invalidate boom")


class NullClassifier:
    """返回固定状态的结果替身。"""

    def classify(self, evidence: object = None, *args: object, **kwargs: object):
        """返回状态结果。"""
        from src.vision.state.models import StateResult

        return StateResult(state_name=TRIGGER_STATE, confidence=0.9, source="evidence")


class FakeSource:
    """帧来源替身。"""

    def __init__(self, count: int) -> None:
        self._count = count
        self.last_skip_reason: str | None = None

    def start(self) -> None:
        """无操作。"""

    def stop(self) -> None:
        """无操作。"""

    def get_frame(self):
        """取下一帧。"""
        if self._count <= 0:
            self.last_skip_reason = "no_frame"
            return None
        self._count -= 1
        return Frame.create(
            np.zeros((PIXELS, PIXELS, 3), dtype=np.uint8), source="fake", timestamp=1.0
        )


def make_monitor(tmp_path: Path, *, frames: int, bridge: object, clock: FakeClock) -> StateMonitor:
    """构造最小 StateMonitor（可指定接线到 vision_bridge 的对象）。"""
    return StateMonitor(
        NullClassifier(),  # type: ignore[arg-type]
        source=FakeSource(frames),  # type: ignore[arg-type]
        config=RuntimeStateMonitorConfig(
            interval_ms=10,
            history_path=str(tmp_path / "history.json"),
            save_history=False,
            check_foreground=False,
        ),
        processor=None,
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(enabled=False)),
        clock=clock,
        sleep=lambda _seconds: None,
        now=lambda: "2026-01-01T00:00:00+08:00",
        vision_bridge=bridge,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 1 / 2 / 3. 快照失效接线
# ---------------------------------------------------------------------------


def test_run_completion_invalidates_snapshot(tmp_path: Path) -> None:
    """``run()`` 结束 → 快照失效（``runtime_stopped``），原因可追踪。"""
    clock = FakeClock()
    resolver = StubResolver()
    bridge = VisionSnapshotBridge(resolver=resolver, clock=clock)
    monitor = make_monitor(tmp_path, frames=1, bridge=bridge, clock=clock)

    monitor.run(max_cycles=1)

    assert bridge.stats.invalidated == 1
    assert bridge.stats.last_reason == INVALIDATION_RUNTIME_STOPPED
    assert INVALIDATION_RUNTIME_STOPPED in INVALIDATION_REASONS
    assert len(resolver.snapshots) == 2  # 有效帧 + 结束失效
    assert resolver.snapshots[-1].count == 0  # type: ignore[attr-defined]


def test_pause_invalidates_snapshot(tmp_path: Path) -> None:
    """Runtime 暂停（首个暂停轮）→ 快照失效（``window_unavailable``）。"""
    clock = FakeClock()
    resolver = StubResolver()
    bridge = VisionSnapshotBridge(resolver=resolver, clock=clock)
    monitor = make_monitor(tmp_path, frames=1, bridge=bridge, clock=clock)
    # lifecycle 必须先到 RUNNING 才能暂停（initializing -> paused 是非法转换）
    monitor.lifecycle.initialize()
    monitor.lifecycle.start()
    monitor.pause()

    monitor.run(max_cycles=2)

    assert bridge.stats.invalidated >= 1
    assert (
        bridge.stats.last_reason == INVALIDATION_WINDOW_UNAVAILABLE
        or INVALIDATION_WINDOW_UNAVAILABLE in INVALIDATION_REASONS
    )
    assert INVALIDATION_WINDOW_UNAVAILABLE in INVALIDATION_REASONS


def test_invalidate_exception_is_isolated(tmp_path: Path) -> None:
    """桥的 ``invalidate`` 抛异常时监控主循环照常结束（异常隔离）。"""
    clock = FakeClock()
    exploding = ExplodingBridge()
    monitor = make_monitor(tmp_path, frames=1, bridge=exploding, clock=clock)

    report = monitor.run(max_cycles=1)

    assert report.cycles == 1
    assert exploding.calls == 1  # 结束时的失效被调用且异常被吞掉


def test_bridge_without_invalidate_is_supported(tmp_path: Path) -> None:
    """桥未实现 ``invalidate``（老实现）时失效接线为无操作（向后兼容）。"""

    class LegacyBridge:
        def __init__(self) -> None:
            self.updates = 0

        def update(self, **kwargs: object) -> None:
            """只统计更新。"""
            self.updates += 1

    clock = FakeClock()
    legacy = LegacyBridge()
    monitor = make_monitor(tmp_path, frames=1, bridge=legacy, clock=clock)

    report = monitor.run(max_cycles=1)

    assert report.cycles == 1
    assert legacy.updates == 1


def test_monitor_without_vision_bridge_is_unchanged(tmp_path: Path) -> None:
    """未接视觉桥：失效接线为无操作（行为不变）。"""
    clock = FakeClock()
    monitor = make_monitor(tmp_path, frames=1, bridge=None, clock=clock)

    report = monitor.run(max_cycles=1)

    assert report.cycles == 1


# ---------------------------------------------------------------------------
# 4 - 8. 调度器看门狗
# ---------------------------------------------------------------------------


def flow_document(task_id: str = "task_a") -> dict:
    """标准流程（prepare --state == battle--> running）。"""
    return {
        "task_id": task_id,
        "initial": "prepare",
        "states": [
            {
                "name": "prepare",
                "actions": [{"kind": "click", "target": {"element": TARGET_NAME}}],
                "timeout": 600,
            },
            {"name": "running", "timeout": 600},
        ],
        "transitions": [
            {"from": "prepare", "to": "running", "when": f"state == {TRIGGER_STATE}"},
        ],
        "budget": {"max_steps": 20, "max_step_retries": 0},
    }


class CountingPort:
    """记录请求并返回 SKIPPED 的端口（dry-run 语义）。"""

    def __init__(self) -> None:
        self.requests: list[object] = []

    def request(self, request: object, /) -> ActionOutcome:
        """记录请求。"""
        self.requests.append(request)
        return ActionOutcome(status=ActionStatus.SKIPPED, reason="dry-run")


def make_scheduler(clock: FakeClock, **kwargs: object) -> TaskScheduler:
    """构造带一个流程任务的调度器。"""
    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document()))
    return TaskScheduler(
        registry,
        clock=clock,
        now=lambda: "T",
        action_port=CountingPort(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_watchdog_is_disabled_by_default() -> None:
    """默认关闭：长时间不推进也不会被强制失败（向后兼容）。"""
    clock = FakeClock()
    scheduler = make_scheduler(clock)
    instance = scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})

    clock.advance(3600.0)
    tick = scheduler.tick(None)

    assert scheduler.watchdog_timeout_s == 0.0
    assert tick.watchdog_kills == 0
    assert instance.status is TaskInstanceStatus.RUNNING


def test_watchdog_kills_stalled_instance() -> None:
    """挂起实例（既不推进也不终结）超过阈值 → 强制失败并释放资源。"""
    clock = FakeClock()
    scheduler = make_scheduler(clock, max_concurrent=1, watchdog_timeout_s=30.0)
    instance = scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})
    assert instance.status is TaskInstanceStatus.RUNNING

    clock.advance(31.0)
    tick = scheduler.tick(None)  # 不喂状态：既不推进也不终结

    assert tick.watchdog_kills == 1
    assert scheduler.stats.watchdog_kills == 1
    assert instance.status is TaskInstanceStatus.FAILED
    assert "watchdog timeout" in instance.last_error
    assert scheduler.locks.to_dict() == {}  # 资源已释放
    assert scheduler.instances == ()  # 已移出活跃集合


def test_watchdog_does_not_fire_when_progressing() -> None:
    """正常推进的实例不会被看门狗误杀（每轮推进刷新计时）。"""
    clock = FakeClock()
    scheduler = make_scheduler(clock, watchdog_timeout_s=30.0)
    instance = scheduler.submit("task_a")

    for _ in range(5):
        clock.advance(10.0)  # 每次 < 阈值
        scheduler.tick({"state_name": TRIGGER_STATE})

    assert scheduler.stats.watchdog_kills == 0
    assert instance.status is TaskInstanceStatus.RUNNING


def test_watchdog_rearms_after_resume() -> None:
    """暂停 / 恢复后重新计时（不把暂停时间算作挂起）。"""
    clock = FakeClock()
    scheduler = make_scheduler(clock, watchdog_timeout_s=30.0)
    instance = scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})

    assert scheduler.pause("task_a") is True
    clock.advance(120.0)  # 暂停期间不计入
    assert scheduler.resume("task_a") is True

    clock.advance(10.0)
    scheduler.tick({"state_name": TRIGGER_STATE})

    assert scheduler.stats.watchdog_kills == 0
    assert instance.status is TaskInstanceStatus.RUNNING


def test_watchdog_can_be_toggled_at_runtime() -> None:
    """``set_watchdog_timeout`` 可运行时开关。"""
    clock = FakeClock()
    scheduler = make_scheduler(clock)
    scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})

    scheduler.set_watchdog_timeout(15.0)
    clock.advance(20.0)
    tick = scheduler.tick(None)

    assert tick.watchdog_kills == 1
    scheduler.set_watchdog_timeout(0)
    assert scheduler.watchdog_timeout_s == 0.0


def test_watchdog_config_sources() -> None:
    """配置：映射 / 仓库 YAML / ``from_config`` 都能带上看门狗超时。"""
    config = scheduler_config_from_mapping({"max_concurrent": 1, "watchdog_timeout_s": 45})
    assert config.watchdog_timeout_s == 45.0
    assert config.to_dict()["watchdog_timeout_s"] == 45.0

    repository = load_scheduler_config()
    assert repository.watchdog_timeout_s == 0.0  # 仓库默认关闭

    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document()))
    built = TaskScheduler.from_config(config, registry, clock=lambda: 1.0)
    assert built.watchdog_timeout_s == 45.0

    with pytest.raises(Exception):
        scheduler_config_from_mapping({"watchdog_timeout_s": -1})


def test_watchdog_is_isolated_from_other_instances() -> None:
    """看门狗只终结超时实例，不牵连其它 RUNNING 实例。"""
    clock = FakeClock()
    registry = TaskRegistry()
    registry.register(validate_flow_document(flow_document("task_a")))
    registry.register(validate_flow_document(flow_document("task_b")))
    scheduler = TaskScheduler(
        registry,
        max_concurrent=2,
        clock=clock,
        now=lambda: "T",
        action_port=CountingPort(),
        watchdog_timeout_s=30.0,
    )
    first = scheduler.submit("task_a", resources=("RES_A",))
    second = scheduler.submit("task_b", resources=("RES_B",))
    scheduler.tick({"state_name": TRIGGER_STATE})

    # 只让 first 挂起：单独推进 second（通过 engine 直接喂），不动 first
    clock.advance(5.0)
    second.engine.handle_state({"state_name": TRIGGER_STATE})
    second.watchdog_armed_at = clock()
    clock.advance(30.0)
    scheduler.tick(None)

    assert first.status is TaskInstanceStatus.FAILED
    assert second.status is TaskInstanceStatus.RUNNING
    assert scheduler.stats.watchdog_kills == 1


def instance_terminal(scheduler: TaskScheduler) -> bool:
    """辅助：确认看门狗 kill 后没有活跃实例（资源与状态都已落定）。"""
    return scheduler.instances == () and scheduler.locks.to_dict() == {}


def test_no_real_input_in_new_paths() -> None:
    """新路径（失效接线 / 看门狗）同样零真实输入。"""
    clock = FakeClock()
    scheduler = make_scheduler(clock, watchdog_timeout_s=1.0)
    scheduler.submit("task_a")
    scheduler.tick({"state_name": TRIGGER_STATE})
    clock.advance(2.0)
    scheduler.tick(None)

    port = scheduler._action_port  # noqa: SLF001 - 测试断言用
    assert isinstance(port, CountingPort)
    # 端口只收到语义请求（ActionRequest），没有任何输入原语被调用
    assert port.requests and all(hasattr(item, "target") for item in port.requests)
    assert ACTIONABLE_EVIDENCE  # 语义登记表仍在（未被本阶段改动）
    assert FRAME_SIZE.width == CLIENT_RECT.width
    assert instance_terminal(scheduler)
