"""StateMonitor 的状态变化事件与历史落盘用例。

初始化与周期运行的用例见 ``test_state_monitor.py``；历史存储自身的用例见
``test_state_monitor_history.py``。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import (
    AppConfig,
    RuntimeStabilizationConfig,
    RuntimeStateMonitorConfig,
)
from src.runtime.state_monitor import StateEvent, StateMonitor, StateStabilizer
from src.vision.capture.models import Frame
from src.vision.state import StateClassifier, StateName
from tests.monitor_fakes import (
    FIXED_TIMESTAMP,
    INTERVAL_MS,
    FakeClassifier,
    FakeFrameSource,
    build_monitor,
    make_frame,
)

# --- 状态变化事件 -------------------------------------------------------------


def test_first_classification_sets_baseline_without_event(tmp_path: Path) -> None:
    """首次识别只建立基线，不产生事件（没有"前一个状态"）。"""
    monitor, _, _, _ = build_monitor([StateName.YARD], tmp_path=tmp_path)

    report = monitor.run(max_cycles=1)

    assert report.initial_state is StateName.YARD
    assert report.final_state is StateName.YARD
    assert report.event_count == 0
    assert monitor.events == ()


def test_state_change_produces_event(tmp_path: Path) -> None:
    """只有 state_name 改变才产生事件（连续相同状态不重复）。"""
    states = [StateName.UNKNOWN, StateName.YARD, StateName.YARD, StateName.TRANSITION]
    monitor, _, _, _ = build_monitor(states, tmp_path=tmp_path)

    report = monitor.run(max_cycles=len(states))

    assert report.event_count == 2
    first, second = report.events
    assert first.previous_state is StateName.UNKNOWN
    assert first.current_state is StateName.YARD
    assert first.confidence == pytest.approx(0.9)
    assert first.timestamp == FIXED_TIMESTAMP
    assert second.state_key == "yard -> transition"
    assert report.initial_state is StateName.UNKNOWN
    assert report.final_state is StateName.TRANSITION


def test_state_regression_also_produces_event(tmp_path: Path) -> None:
    """状态"回落"（transition -> unknown）同样是变化，必须产生事件。"""
    states = [StateName.TRANSITION, StateName.UNKNOWN]
    monitor, _, _, _ = build_monitor(states, tmp_path=tmp_path)

    report = monitor.run(max_cycles=len(states))

    assert report.event_count == 1
    assert report.events[0].state_key == "transition -> unknown"


def test_skipped_cycles_do_not_create_events(tmp_path: Path) -> None:
    """无帧的轮次不会产生"状态变化"（画面都不知道，不能编造状态）。"""
    plan: list[Frame | None] = [make_frame(), None, None]
    monitor, _, _, _ = build_monitor([StateName.UNKNOWN, StateName.YARD], plan=plan, tmp_path=tmp_path)

    report = monitor.run(max_cycles=3)

    assert report.skipped == 2
    assert report.classified == 1
    assert report.event_count == 0


def test_on_result_and_on_event_callbacks(tmp_path: Path) -> None:
    """回调按预期触发（Demo 依赖它们打印输出）。"""
    seen_results: list[StateName] = []
    seen_events: list[StateEvent] = []
    config = RuntimeStateMonitorConfig(interval_ms=INTERVAL_MS, save_history=False)
    monitor = StateMonitor(
        FakeClassifier([StateName.UNKNOWN, StateName.YARD]),  # type: ignore[arg-type]
        FakeFrameSource(),
        config,
        # 直通稳定化（窗口 1 帧）：本用例只验证回调接线，稳定化语义见 test_state_stabilizer.py
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(min_frames=1)),
        on_result=lambda result: seen_results.append(result.state_name),
        on_event=seen_events.append,
    )

    monitor.run(max_cycles=2)

    assert seen_results == [StateName.UNKNOWN, StateName.YARD]
    assert [event.state_key for event in seen_events] == ["unknown -> yard"]


def test_step_can_be_used_standalone(tmp_path: Path) -> None:
    """step 可独立使用：单帧识别 + 变化检测（history 需自行 start）。"""
    monitor, _, _, _ = build_monitor([StateName.YARD], tmp_path=tmp_path)
    monitor.history.start(timestamp=FIXED_TIMESTAMP)

    first = monitor.step(make_frame())
    second = monitor.step(make_frame())

    assert first.state_name is StateName.YARD
    assert second.state_name is StateName.YARD
    assert monitor.current_state is StateName.YARD
    assert monitor.last_result is second
    assert monitor.event_count == 0


# --- 历史落盘 -----------------------------------------------------------------


def test_events_are_persisted_immediately(tmp_path: Path) -> None:
    """事件产生后立即落盘（不依赖运行结束）。"""
    monitor, _, _, _ = build_monitor([StateName.UNKNOWN, StateName.YARD], tmp_path=tmp_path)
    monitor.history.start(timestamp=FIXED_TIMESTAMP)

    monitor.step(make_frame())
    monitor.step(make_frame())

    assert monitor.history.path.is_file()
    data = monitor.history.read()
    assert data["initial_state"] == "unknown"
    assert data["events"] == [
        {
            "previous_state": "unknown",
            "current_state": "yard",
            "timestamp": FIXED_TIMESTAMP,
            "confidence": 0.9,
        }
    ]


def test_history_records_progress_and_initial_state(tmp_path: Path) -> None:
    """运行结束后历史文件包含周期数、跳过数与初始状态。"""
    plan: list[Frame | None] = [make_frame(), None, make_frame()]
    monitor, _, _, _ = build_monitor(
        [StateName.UNKNOWN, StateName.YARD], plan=plan, tmp_path=tmp_path, interval_ms=200
    )

    report = monitor.run(max_cycles=3)

    data = monitor.history.read()
    assert data["cycles"] == 3
    assert data["skipped"] == 1
    assert data["interval_ms"] == 200
    assert data["initial_state"] == "unknown"
    assert Path(report.history_path or "").is_file()
    assert len(monitor.history.read_events()) == report.event_count


def test_history_disabled_leaves_no_file(tmp_path: Path) -> None:
    """save_history=false 时不写文件，摘要中的 history_path 为 None。"""
    monitor, _, _, _ = build_monitor(
        [StateName.UNKNOWN, StateName.YARD], tmp_path=tmp_path, save_history=False
    )

    report = monitor.run(max_cycles=2)

    assert report.history_path is None
    assert monitor.history.event_count == 1
    assert not (tmp_path / "state_history.json").exists()


@pytest.mark.skipif(
    not (
        (Path(__file__).resolve().parents[1] / "dataset" / "states").is_dir()
        and (Path(__file__).resolve().parents[1] / "dataset" / "scenes").is_dir()
    ),
    reason="需要 dataset/states 与 dataset/scenes（图片不入库）",
)
def test_from_config_builds_real_classifier() -> None:
    """from_config 未注入分类器时按应用配置装配真实的 StateClassifier。"""
    monitor = StateMonitor.from_config(AppConfig(), source=FakeFrameSource())

    assert isinstance(monitor.classifier, StateClassifier)
    assert monitor.classifier.name == "state"
