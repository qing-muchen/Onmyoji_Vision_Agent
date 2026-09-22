"""监控运行摘要与停止原因（Phase 3.1）。

与 :mod:`src.runtime.state_monitor.monitor` 分开的原因：摘要模型是**输出契约**
（Demo / 未来任务层都读它），而 monitor 模块承载循环逻辑，拆开后两者都能保持精简。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from src.runtime.state_monitor.events import StateEvent
from src.vision.state.models import StateName

__all__ = ["MonitorReport", "StopReason", "local_now_iso"]


class StopReason(StrEnum):
    """监控循环的停止原因。"""

    #: 达到 ``max_cycles``
    MAX_CYCLES = "max_cycles"
    #: 达到 ``duration_s``
    DURATION = "duration"
    #: 收到中断信号（Ctrl+C）
    INTERRUPT = "interrupt"


def local_now_iso() -> str:
    """返回本地时间的 ISO 8601 字符串（含时区偏移）。

    历史文件与事件时间戳都用它：可读、带时区、可直接 JSON 序列化。
    """
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class MonitorReport:
    """一次监控运行的摘要。

    说明：``initial_state`` / ``final_state`` / ``events`` 都是**稳定状态**
    （经 :class:`~src.runtime.state_monitor.stabilizer.StateStabilizer` 处理后的结果），
    不是逐帧的原始识别值。

    Attributes:
        cycles: 已完成的周期数。
        classified: 实际完成状态识别的周期数。
        skipped: 因帧不可用而跳过的周期数（后端暂无画面 / 前台门控未通过）。
        events: 本次运行产生的稳定状态变化事件。
        initial_state: 首次确认的稳定状态（基线，不属于事件）。
        final_state: 运行结束时的稳定状态。
        started_at: 开始时间（ISO 8601）。
        ended_at: 结束时间（ISO 8601）。
        stop_reason: 停止原因。
        history_path: 历史文件路径字符串；未落盘时为 ``None``。
    """

    cycles: int
    classified: int
    skipped: int
    events: tuple[StateEvent, ...]
    initial_state: StateName | None
    final_state: StateName | None
    started_at: str
    ended_at: str
    stop_reason: StopReason
    history_path: str | None = None

    @property
    def event_count(self) -> int:
        """事件数量。"""
        return len(self.events)

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"cycles={self.cycles} classified={self.classified} skipped={self.skipped} "
            f"events={self.event_count} "
            f"initial={self.initial_state.value if self.initial_state else 'none'} "
            f"final={self.final_state.value if self.final_state else 'none'} "
            f"stop={self.stop_reason.value}"
        )
