"""状态监控测试的共享替身：帧来源、分类器、时钟与装配辅助。

这些替身让监控循环**不依赖真实屏幕与真实等待**即可确定性测试（见 ``test_state_monitor*``）。
本文件不以 ``test_`` 开头，pytest 不会把它当成测试模块。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.core.config import RuntimeStateMonitorConfig
from src.runtime.state_monitor import StateMonitor
from src.vision.capture.models import Frame
from src.vision.state import StateName, StateResult, StateSource

#: 测试使用的监控周期（毫秒）
INTERVAL_MS = 500
#: 测试使用的时间戳
FIXED_TIMESTAMP = "2025-01-01T12:00:00+08:00"


def make_frame() -> Frame:
    """构造一个极小的测试帧（监控只关心"有没有帧"）。"""
    return Frame.create(np.zeros((4, 4, 3), dtype=np.uint8), source="fake")


class FakeClassifier:
    """按预设序列返回状态结果的分类器替身（序列循环使用，支持重复 run）。"""

    def __init__(self, states: Sequence[StateName], *, confidence: float = 0.9) -> None:
        self._results = [
            StateResult(
                state_name=state,
                confidence=confidence,
                source=StateSource.FALLBACK,
                detail=f"stub {index}",
            )
            for index, state in enumerate(states)
        ]
        self.calls = 0

    def classify(self, frame: Frame) -> StateResult:
        """返回序列中的下一个结果（用尽后从头循环）。"""
        index = self.calls % len(self._results)
        self.calls += 1
        return self._results[index]


class FakeFrameSource:
    """按预设计划返回帧的帧来源替身（可返回 ``None``，也可在指定轮抛中断）。"""

    def __init__(
        self,
        plan: Sequence[Frame | None] | None = None,
        *,
        interrupt_at: int | None = None,
    ) -> None:
        self._plan = list(plan) if plan else []
        self._interrupt_at = interrupt_at
        self.calls = 0

    def get_frame(self) -> Frame | None:
        """返回计划中的下一帧。"""
        index = self.calls
        self.calls += 1
        if self._interrupt_at is not None and index == self._interrupt_at:
            raise KeyboardInterrupt
        if not self._plan:
            return make_frame()
        return self._plan[index % len(self._plan)]


class FakeClock:
    """可注入的时钟与睡眠：``sleep`` 会推进时间，使周期逻辑无需真实等待即可测试。"""

    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        """当前时间（秒）。"""
        return self.value

    def sleep(self, seconds: float) -> None:
        """记录等待时长并推进时间。"""
        self.sleeps.append(seconds)
        self.value += seconds


def build_monitor(
    states: Sequence[StateName],
    *,
    plan: Sequence[Frame | None] | None = None,
    tmp_path: Path | None = None,
    clock: FakeClock | None = None,
    interrupt_at: int | None = None,
    **config_overrides: object,
) -> tuple[StateMonitor, FakeClassifier, FakeFrameSource, FakeClock]:
    """装配监控器与全部替身。

    Args:
        states: 分类器按顺序返回的状态序列。
        plan: 帧来源的返回计划（``None`` 表示该轮无可用帧）。
        tmp_path: 历史文件所在目录；省略时使用配置默认路径（测试中应尽量传入）。
        clock: 可复用的假时钟；省略时新建。
        interrupt_at: 在该轮索引抛出 ``KeyboardInterrupt``。
        **config_overrides: 覆盖监控配置字段（例如 ``interval_ms`` / ``save_history``）。

    Returns:
        ``(监控器, 分类器替身, 帧来源替身, 假时钟)``。
    """
    active_clock = clock if clock is not None else FakeClock()
    values: dict[str, object] = {"interval_ms": INTERVAL_MS}
    if tmp_path is not None:
        values["history_path"] = str(tmp_path / "state_history.json")
    values.update(config_overrides)
    config = RuntimeStateMonitorConfig(**values)  # type: ignore[arg-type]

    classifier = FakeClassifier(states)
    source = FakeFrameSource(plan, interrupt_at=interrupt_at)
    monitor = StateMonitor(
        classifier,  # type: ignore[arg-type]
        source,
        config,
        clock=active_clock.clock,
        sleep=active_clock.sleep,
        now=lambda: FIXED_TIMESTAMP,
    )
    return monitor, classifier, source, active_clock
