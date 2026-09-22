"""StateMonitor 的单元测试：初始化与固定周期运行。

状态变化事件与历史落盘的用例见 ``test_state_monitor_tracking.py``；
历史存储自身的用例见 ``test_state_monitor_history.py``。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import AppConfig, RuntimeStateMonitorConfig
from src.runtime.state_monitor import StateHistoryStore, StateMonitor, StopReason
from src.vision.capture.models import Frame
from src.vision.state import StateName
from tests.monitor_fakes import (
    INTERVAL_MS,
    FakeClassifier,
    FakeClock,
    FakeFrameSource,
    build_monitor,
    make_frame,
)

# --- 初始化 -------------------------------------------------------------------


def test_monitor_exposes_config_and_components(tmp_path: Path) -> None:
    """初始化后暴露配置、分类器、帧来源与历史存储。"""
    monitor, classifier, source, _ = build_monitor([StateName.YARD], tmp_path=tmp_path)

    assert monitor.config.interval_ms == INTERVAL_MS
    assert monitor.config.interval_seconds == pytest.approx(INTERVAL_MS / 1000)
    assert monitor.classifier is classifier
    assert monitor.source is source
    assert monitor.history.path == tmp_path / "state_history.json"
    assert monitor.current_state is None
    assert monitor.last_result is None
    assert monitor.events == ()
    assert monitor.event_count == 0


def test_history_store_defaults_to_config(tmp_path: Path) -> None:
    """未注入 history 时按配置自行创建（路径来自配置）。"""
    config = RuntimeStateMonitorConfig(history_path=str(tmp_path / "own.json"))
    monitor = StateMonitor(
        FakeClassifier([StateName.YARD]),  # type: ignore[arg-type]
        FakeFrameSource(),
        config,
    )

    assert isinstance(monitor.history, StateHistoryStore)
    assert monitor.history.path == tmp_path / "own.json"


@pytest.mark.parametrize("interval_ms", [0, -1, 60001])
def test_interval_must_be_within_range(interval_ms: int) -> None:
    """监控周期必须在合法范围内（配置层校验）。"""
    with pytest.raises(ValueError):
        RuntimeStateMonitorConfig(interval_ms=interval_ms)


def test_from_config_uses_injected_classifier() -> None:
    """from_config 注入分类器时不再自动装配（便于测试与自定义）。"""
    classifier = FakeClassifier([StateName.YARD])
    monitor = StateMonitor.from_config(
        AppConfig(),
        source=FakeFrameSource(),
        classifier=classifier,  # type: ignore[arg-type]
    )

    assert monitor.classifier is classifier
    assert monitor.config.interval_ms == INTERVAL_MS


# --- 周期运行 -----------------------------------------------------------------


def test_run_respects_max_cycles(tmp_path: Path) -> None:
    """max_cycles 决定循环次数，且每轮都调用一次帧来源与分类器。"""
    monitor, classifier, source, _ = build_monitor([StateName.YARD], tmp_path=tmp_path)

    report = monitor.run(max_cycles=3)

    assert report.cycles == 3
    assert report.classified == 3
    assert source.calls == 3
    assert classifier.calls == 3
    assert report.stop_reason is StopReason.MAX_CYCLES
    assert report.started_at == "2025-01-01T12:00:00+08:00"


def test_loop_uses_fixed_period(tmp_path: Path) -> None:
    """等待时长约为配置周期（固定节拍，假时钟下可精确断言）。"""
    monitor, _, _, clock = build_monitor([StateName.YARD], tmp_path=tmp_path, interval_ms=500)

    monitor.run(max_cycles=3)

    assert clock.sleeps == pytest.approx([0.5, 0.5, 0.5])


def test_interval_override_changes_period(tmp_path: Path) -> None:
    """周期来自配置：改配置即改等待时长。"""
    monitor, _, _, clock = build_monitor([StateName.YARD], tmp_path=tmp_path, interval_ms=200)

    monitor.run(max_cycles=2)

    assert clock.sleeps == pytest.approx([0.2, 0.2])


def test_slow_cycle_does_not_drift(tmp_path: Path) -> None:
    """单轮耗时超过周期时重新对齐节拍，不产生"追赶式"连续等待。"""

    class SlowFrameSource:
        """取帧时推进时钟，模拟"单轮处理耗时"超过监控周期。"""

        def __init__(self, clock: FakeClock, cost: float) -> None:
            self._clock = clock
            self._cost = cost

        def get_frame(self) -> Frame:
            self._clock.value += self._cost
            return make_frame()

    clock = FakeClock()
    config = RuntimeStateMonitorConfig(
        interval_ms=500, history_path=str(tmp_path / "h.json"), save_history=False
    )
    monitor = StateMonitor(
        FakeClassifier([StateName.YARD]),  # type: ignore[arg-type]
        SlowFrameSource(clock, cost=0.8),  # 每轮 0.8s > 0.5s 周期
        config,
        clock=clock.clock,
        sleep=clock.sleep,
    )

    monitor.run(max_cycles=3)

    # 每轮已耗时 0.8s：不应再额外等待，也不应累积补偿
    assert clock.sleeps == []


def test_duration_limit_stops_loop(tmp_path: Path) -> None:
    """duration_s 到点即停（由注入时钟推进）。"""
    monitor, _, _, _ = build_monitor([StateName.YARD], tmp_path=tmp_path, interval_ms=500)

    report = monitor.run(duration_s=1.0)

    assert report.stop_reason is StopReason.DURATION
    assert report.cycles == 2


def test_unavailable_frames_are_skipped(tmp_path: Path) -> None:
    """帧不可用的轮次计入 skipped，且不调用分类器。"""
    plan: list[Frame | None] = [make_frame(), None, make_frame()]
    monitor, classifier, source, _ = build_monitor([StateName.YARD], plan=plan, tmp_path=tmp_path)

    report = monitor.run(max_cycles=3)

    assert source.calls == 3
    assert report.skipped == 1
    assert report.classified == 2
    assert classifier.calls == 2


def test_run_can_be_repeated(tmp_path: Path) -> None:
    """重复 run 会重置计数与事件。"""
    monitor, _, _, _ = build_monitor(
        [StateName.UNKNOWN, StateName.YARD], tmp_path=tmp_path, interval_ms=200
    )

    first = monitor.run(max_cycles=2)
    second = monitor.run(max_cycles=2)

    assert first.cycles == 2 and second.cycles == 2
    assert first.event_count == 1
    assert second.event_count == 1


def test_keyboard_interrupt_returns_report_and_saves(tmp_path: Path) -> None:
    """中断信号不会让监控崩溃：返回摘要并保留已记录的历史。"""
    monitor, _, _, _ = build_monitor(
        [StateName.UNKNOWN, StateName.YARD],
        tmp_path=tmp_path,
        interval_ms=200,
        interrupt_at=3,
    )

    report = monitor.run()

    assert report.stop_reason is StopReason.INTERRUPT
    assert report.cycles == 3
    assert report.history_path is not None
    assert Path(report.history_path).is_file()
