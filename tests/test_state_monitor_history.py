"""StateHistoryStore 的单元测试：即时落盘、原子写、上限截断与错误处理。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.config import RuntimeStateMonitorConfig
from src.core.exceptions import MonitorError
from src.runtime.state_monitor import StateEvent, StateHistoryStore
from src.runtime.state_monitor.history import TEMP_SUFFIX
from src.vision.state import StateName

#: 固定时间戳
TIMESTAMP = "2025-01-01T12:00:00+08:00"
#: 默认监控周期（毫秒）
INTERVAL_MS = 500


def build_config(tmp_path: Path, **overrides: object) -> RuntimeStateMonitorConfig:
    """构造指向临时目录的监控配置。"""
    values: dict[str, object] = {
        "history_path": str(tmp_path / "state_history.json"),
        "interval_ms": INTERVAL_MS,
    }
    values.update(overrides)
    return RuntimeStateMonitorConfig(**values)  # type: ignore[arg-type]


def build_event(
    previous: StateName = StateName.UNKNOWN,
    current: StateName = StateName.YARD,
    confidence: float = 0.9,
    timestamp: str = TIMESTAMP,
) -> StateEvent:
    """构造测试事件。"""
    return StateEvent(
        previous_state=previous,
        current_state=current,
        timestamp=timestamp,
        confidence=confidence,
    )


def test_path_is_resolved_against_project_root(tmp_path: Path) -> None:
    """相对路径按项目根解析，绝对路径原样保留。"""
    relative = StateHistoryStore(RuntimeStateMonitorConfig(history_path="logs/x.json"))
    absolute = StateHistoryStore(build_config(tmp_path))

    assert relative.path.is_absolute()
    assert relative.path.name == "x.json"
    assert absolute.path == (tmp_path / "state_history.json")


def test_start_resets_state_and_records_metadata(tmp_path: Path) -> None:
    """start 记录元数据并清空上一轮内存事件。"""
    store = StateHistoryStore(build_config(tmp_path))
    store.append(build_event())

    store.start(timestamp=TIMESTAMP, interval_ms=250)

    assert store.is_started is True
    assert store.event_count == 0
    assert store.initial_state is None
    snapshot = store.snapshot()
    assert snapshot["started_at"] == TIMESTAMP
    assert snapshot["interval_ms"] == 250


def test_append_persists_immediately(tmp_path: Path) -> None:
    """每次 append 都会立即落盘（不依赖进程退出）。"""
    store = StateHistoryStore(build_config(tmp_path))
    store.start(timestamp=TIMESTAMP)
    store.set_initial_state(StateName.UNKNOWN)

    store.append(build_event(), timestamp="2025-01-01T12:00:05+08:00")

    assert store.path.is_file()
    data = store.read()
    assert data["initial_state"] == "unknown"
    assert data["updated_at"] == "2025-01-01T12:00:05+08:00"
    assert data["events"] == [
        {
            "previous_state": "unknown",
            "current_state": "yard",
            "timestamp": TIMESTAMP,
            "confidence": 0.9,
        }
    ]


def test_progress_is_included_in_snapshot(tmp_path: Path) -> None:
    """周期数与跳过数写入历史，便于判断监控是否连续。"""
    store = StateHistoryStore(build_config(tmp_path))
    store.start(timestamp=TIMESTAMP)
    store.update_progress(cycles=42, skipped=3, timestamp=TIMESTAMP)

    store.save()

    data = store.read()
    assert data["cycles"] == 42
    assert data["skipped"] == 3


def test_events_beyond_limit_drop_oldest(tmp_path: Path) -> None:
    """超过 max_events 时丢弃最旧事件，只保留最近若干条。"""
    store = StateHistoryStore(build_config(tmp_path, max_events=2))
    store.start(timestamp=TIMESTAMP)

    store.append(build_event(current=StateName.YARD))
    store.append(build_event(previous=StateName.YARD, current=StateName.TRANSITION))
    store.append(build_event(previous=StateName.TRANSITION, current=StateName.UNKNOWN))

    events = store.read_events()
    assert [event.current_state for event in events] == [
        StateName.TRANSITION,
        StateName.UNKNOWN,
    ]


def test_save_is_skipped_when_disabled(tmp_path: Path) -> None:
    """save_history=false 时不产生文件（只保留内存数据）。"""
    store = StateHistoryStore(build_config(tmp_path, save_history=False))
    store.start(timestamp=TIMESTAMP)

    store.append(build_event())

    assert store.event_count == 1
    assert not store.path.exists()


def test_save_leaves_no_temporary_file(tmp_path: Path) -> None:
    """原子写不留下临时文件。"""
    store = StateHistoryStore(build_config(tmp_path))
    store.start(timestamp=TIMESTAMP)

    store.save()

    leftovers = list(tmp_path.glob(f"*{TEMP_SUFFIX}"))
    assert leftovers == []
    assert store.path.is_file()


def test_read_events_round_trips(tmp_path: Path) -> None:
    """落盘后可以回读并重建事件对象。"""
    store = StateHistoryStore(build_config(tmp_path))
    store.start(timestamp=TIMESTAMP)
    store.set_initial_state(StateName.YARD)
    store.append(build_event(previous=StateName.YARD, current=StateName.TRANSITION))

    events = store.read_events()

    assert len(events) == 1
    assert events[0] == build_event(previous=StateName.YARD, current=StateName.TRANSITION)
    assert store.read()["initial_state"] == "yard"


def test_write_failure_raises_monitor_error(tmp_path: Path) -> None:
    """写入失败必须明确报错（不静默丢弃状态变化）。"""
    blocker = tmp_path / "blocked.json"
    blocker.mkdir()
    store = StateHistoryStore(RuntimeStateMonitorConfig(history_path=str(blocker)))

    with pytest.raises(MonitorError, match="写入失败"):
        store.save()


def test_read_missing_file_raises_monitor_error(tmp_path: Path) -> None:
    """读取不存在的历史文件时明确报错。"""
    store = StateHistoryStore(build_config(tmp_path))

    with pytest.raises(MonitorError, match="读取失败"):
        store.read()


def test_read_invalid_json_raises_monitor_error(tmp_path: Path) -> None:
    """历史文件不是合法 JSON 时明确报错。"""
    store = StateHistoryStore(build_config(tmp_path))
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not json", encoding="utf-8")

    with pytest.raises(MonitorError, match="JSON"):
        store.read()


def test_invalid_event_fields_raise_monitor_error(tmp_path: Path) -> None:
    """历史文件中的事件字段非法时明确报错。"""
    store = StateHistoryStore(build_config(tmp_path))
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps({"events": [{"previous_state": "unknown", "current_state": "battle"}]}),
        encoding="utf-8",
    )

    with pytest.raises(MonitorError, match="事件字段非法"):
        store.read_events()
