"""StateStabilizer 的单元测试：稳定窗口、unknown 恢复、置信度门槛、状态回退、首次初始化。

同时覆盖与 StateMonitor 的集成：**事件只由稳定状态驱动**。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from src.core.config import RuntimeStabilizationConfig
from src.runtime.state_monitor import StableStateResult, StateStabilizer
from src.vision.state import StateName, StateResult, StateSource
from tests.monitor_fakes import build_monitor

#: 默认稳定窗口（与 configs/config.yaml 一致）
MIN_FRAMES = 3
#: 默认置信度门槛
THRESHOLD = 0.85


def result(state: StateName, confidence: float) -> StateResult:
    """构造一条原始识别结果。"""
    return StateResult(
        state_name=state,
        confidence=confidence,
        source=StateSource.FALLBACK,
        detail="test",
    )


def build_stabilizer(**overrides: object) -> StateStabilizer:
    """构造稳定器（默认启用、窗口 3 帧、门槛 0.85）。"""
    values: dict[str, object] = {"min_frames": MIN_FRAMES, "confidence_threshold": THRESHOLD}
    values.update(overrides)
    return StateStabilizer(RuntimeStabilizationConfig(**values))  # type: ignore[arg-type]


def feed(stabilizer: StateStabilizer, states: Sequence[StateName], confidence: float = 0.9) -> list[StableStateResult]:
    """按顺序喂入状态序列，返回每次的稳定化结果。"""
    return [stabilizer.update(result(state, confidence)) for state in states]


# --- ① 连续 3 帧确认 / 首次状态初始化 -------------------------------------------


def test_first_state_initialized_after_three_frames() -> None:
    """连续 3 帧相同状态才确认；首次确认是建立基线（``changed=False``）。"""
    stabilizer = build_stabilizer()

    results = feed(stabilizer, [StateName.YARD] * MIN_FRAMES)

    assert [item.is_confirmed for item in results] == [False, False, True]
    assert [item.changed for item in results] == [False, False, False]  # 首次不算变化
    assert results[-1].state_name is StateName.YARD
    assert results[-1].support_frames == MIN_FRAMES
    assert results[-1].required_frames == MIN_FRAMES
    assert stabilizer.stable_state is StateName.YARD


def test_two_frames_are_not_enough() -> None:
    """不足窗口帧数时保持未确认，且不产生稳定状态。"""
    stabilizer = build_stabilizer()

    results = feed(stabilizer, [StateName.YARD] * (MIN_FRAMES - 1))

    assert all(item.is_confirmed is False for item in results)
    assert stabilizer.stable_state is None
    assert results[-1].candidate is StateName.YARD
    assert results[-1].support_frames == MIN_FRAMES - 1


def test_candidate_change_resets_counter() -> None:
    """候选状态改变时计数从 1 重新开始。"""
    stabilizer = build_stabilizer()

    feed(stabilizer, [StateName.YARD, StateName.YARD, StateName.TRANSITION])

    assert stabilizer.candidate is StateName.TRANSITION
    assert stabilizer.support_frames == 1
    assert stabilizer.stable_state is None


# --- ② 中途 unknown 恢复 -------------------------------------------------------


def test_unknown_interrupts_then_recovers() -> None:
    """累积过程中插入 unknown 会打断累积，随后重新累积仍能确认 yard。"""
    stabilizer = build_stabilizer()

    results = feed(
        stabilizer,
        [
            StateName.YARD,
            StateName.YARD,
            StateName.UNKNOWN,
            StateName.YARD,
            StateName.YARD,
            StateName.YARD,
        ],
    )

    assert [item.is_confirmed for item in results] == [False] * 5 + [True]
    assert results[2].candidate is StateName.UNKNOWN  # unknown 参与统计
    assert results[2].support_frames == 1  # 打断了 yard 的累积
    assert results[-1].state_name is StateName.YARD


def test_unknown_accumulates_despite_zero_confidence() -> None:
    """unknown 的置信度恒为 0.0，但**豁免**门槛，因此仍能累积成稳定状态。"""
    stabilizer = build_stabilizer()

    results = [
        stabilizer.update(result(StateName.UNKNOWN, 0.0)) for _ in range(MIN_FRAMES)
    ]

    assert all(item.counted for item in results)
    assert results[-1].is_confirmed is True
    assert results[-1].state_name is StateName.UNKNOWN


# --- ③ 置信度不足 -------------------------------------------------------------


def test_low_confidence_frames_are_ignored() -> None:
    """低于门槛的已知状态帧不计入统计：既不累加，也不重置已累积的证据。"""
    stabilizer = build_stabilizer()

    first = stabilizer.update(result(StateName.YARD, 0.9))
    second = stabilizer.update(result(StateName.YARD, 0.5))  # 不计入
    third = stabilizer.update(result(StateName.YARD, 0.9))
    fourth = stabilizer.update(result(StateName.YARD, 0.9))

    assert second.counted is False
    assert second.support_frames == 1  # 未累加
    assert third.support_frames == 2
    assert fourth.is_confirmed is True
    assert first.changed is False and fourth.changed is False


def test_low_confidence_does_not_become_candidate() -> None:
    """低置信度帧不会成为候选状态（不会把噪声状态带进累积）。"""
    stabilizer = build_stabilizer()

    stabilizer.update(result(StateName.YARD, 0.9))
    stabilizer.update(result(StateName.TRANSITION, 0.4))  # 低置信度，忽略

    assert stabilizer.candidate is StateName.YARD
    assert stabilizer.support_frames == 1


def test_threshold_is_configuration_driven() -> None:
    """门槛来自配置：调低门槛后原本被忽略的帧开始计入。"""
    lenient = build_stabilizer(confidence_threshold=0.4)

    results = [lenient.update(result(StateName.YARD, 0.5)) for _ in range(MIN_FRAMES)]

    assert all(item.counted for item in results)
    assert results[-1].is_confirmed is True


# --- ④ 状态回退 ---------------------------------------------------------------


def test_stable_state_falls_back_after_window() -> None:
    """已确认状态被新状态覆盖时：必须满足窗口，且 ``changed=True``。"""
    stabilizer = build_stabilizer()
    feed(stabilizer, [StateName.YARD] * MIN_FRAMES)
    assert stabilizer.stable_state is StateName.YARD

    results = feed(stabilizer, [StateName.UNKNOWN] * MIN_FRAMES, confidence=0.0)

    assert [item.changed for item in results] == [False, False, True]
    assert results[-1].state_name is StateName.UNKNOWN
    assert results[-1].confidence == pytest.approx(0.0)


def test_single_frame_cannot_override_high_confidence_state() -> None:
    """单帧 unknown 不能覆盖高可信的已知稳定状态（必须满足窗口）。"""
    stabilizer = build_stabilizer()
    feed(stabilizer, [StateName.YARD] * MIN_FRAMES)

    for _ in range(MIN_FRAMES - 1):
        item = stabilizer.update(result(StateName.UNKNOWN, 0.0))
        assert item.changed is False
        assert item.state_name is StateName.YARD


def test_repeated_same_state_keeps_confidence() -> None:
    """稳定状态持续期间不再改变置信度（保持确认时的均值）。"""
    stabilizer = build_stabilizer()
    feed(stabilizer, [StateName.YARD] * MIN_FRAMES)
    confirmed_confidence = stabilizer.stable_confidence

    results = feed(stabilizer, [StateName.YARD] * MIN_FRAMES, confidence=0.95)

    assert all(item.changed is False for item in results)
    assert stabilizer.stable_confidence == pytest.approx(confirmed_confidence)


# --- 其它 ---------------------------------------------------------------------


def test_stabilization_disabled_is_pass_through() -> None:
    """关闭稳定化时退化为"每帧立即确认"。"""
    stabilizer = build_stabilizer(enabled=False)

    results = feed(stabilizer, [StateName.YARD, StateName.TRANSITION])

    assert stabilizer.required_frames == 1
    assert [item.is_confirmed for item in results] == [True, True]
    assert results[1].changed is True
    assert results[1].state_name is StateName.TRANSITION


def test_reset_clears_accumulation() -> None:
    """reset 清空累积与稳定状态（支持重复运行）。"""
    stabilizer = build_stabilizer()
    feed(stabilizer, [StateName.YARD] * MIN_FRAMES)

    stabilizer.reset()

    assert stabilizer.stable_state is None
    assert stabilizer.candidate is None
    assert stabilizer.support_frames == 0
    assert stabilizer.is_confirmed is False
    assert stabilizer.last is None


def test_update_rejects_non_state_result() -> None:
    """只接受 StateResult 输入。"""
    stabilizer = build_stabilizer()

    with pytest.raises(TypeError, match="StateResult"):
        stabilizer.update("yard")  # type: ignore[arg-type]


def test_result_describe_includes_raw_and_progress() -> None:
    """稳定化结果提供 raw/stable 与累积进度的可读描述。"""
    stabilizer = build_stabilizer()

    item = stabilizer.update(result(StateName.YARD, 0.9))

    assert "raw=yard" in item.describe()
    assert "stable=pending" in item.describe()
    assert f"1/{MIN_FRAMES}" in item.describe()


def test_min_frames_must_be_positive() -> None:
    """窗口帧数必须在配置层校验。"""
    with pytest.raises(ValueError):
        RuntimeStabilizationConfig(min_frames=0)


# --- 与 StateMonitor 的集成：事件只由稳定状态驱动 --------------------------------


def test_monitor_emits_event_only_on_stable_change(tmp_path: Path) -> None:
    """逐帧抖动不产生事件；只有稳定状态变化才产生。"""
    states = [
        StateName.YARD,
        StateName.TRANSITION,  # 抖动（仅 1 帧）：不产生事件
        StateName.YARD,
        StateName.YARD,
        StateName.YARD,  # 连续 3 帧 yard -> 首次确认（基线，无事件）
        StateName.UNKNOWN,
        StateName.UNKNOWN,
        StateName.UNKNOWN,  # 连续 3 帧 unknown -> 稳定状态变化（1 个事件）
    ]
    monitor, _, _, _ = build_monitor(states, tmp_path=tmp_path, stabilize=True)

    report = monitor.run(max_cycles=len(states))

    assert report.initial_state is StateName.YARD
    assert report.final_state is StateName.UNKNOWN
    assert report.event_count == 1
    assert report.events[0].state_key == "yard -> unknown"
    assert report.events[0].confidence == pytest.approx(0.0)  # unknown 的稳定置信度


def test_monitor_exposes_raw_and_stable(tmp_path: Path) -> None:
    """监控器同时暴露原始状态与稳定状态（Demo 需要展示两者对照）。"""
    monitor, _, _, _ = build_monitor(
        [StateName.YARD, StateName.YARD, StateName.YARD], tmp_path=tmp_path, stabilize=True
    )

    monitor.run(max_cycles=3)

    assert monitor.current_raw_state is StateName.YARD
    assert monitor.current_state is StateName.YARD
    assert monitor.last_stable is not None
    assert monitor.last_stable.support_frames == 3
    assert monitor.stabilizer.is_enabled is True


def test_monitor_records_stable_initial_state_in_history(tmp_path: Path) -> None:
    """历史文件里的 initial_state 是**稳定状态**（不是逐帧原始值）。"""
    states = [StateName.TRANSITION, StateName.YARD, StateName.YARD, StateName.YARD]
    monitor, _, _, _ = build_monitor(states, tmp_path=tmp_path, stabilize=True)

    monitor.run(max_cycles=len(states))

    assert monitor.history.read()["initial_state"] == "yard"
