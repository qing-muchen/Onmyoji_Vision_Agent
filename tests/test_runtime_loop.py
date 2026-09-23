"""Phase 5.2 测试：真实运行时循环（节拍调度 / 跳过路径 / 运行指标 / 既有语义不变）。

覆盖：

1. 500ms scheduler 准确（deadline based，无累计漂移）；
2. 单轮处理耗时超过 interval 时仍继续，且记为 late cycle（overrun / 重对齐统计）；
3. 采集失败（窗口不存在 / 最小化 / 后端异常 / 后端不可用）不中断循环，转为跳过并记录原因；
4. 判定（证据流水线）异常不中断循环，该帧降级为 ``unknown``；
5. 目标窗口不在前台时跳过本轮（前台门控只检查、不抢占、不切换窗口）；
6. 事件逻辑不变（只有稳定状态变化才产生事件）；
7. ``StateStabilizer`` 继续工作（连续窗口 + 置信度门槛）；
8. 运行指标字段与 JSONL 落盘（**不含截图与用户数据**），写入失败不影响主循环。

除标记 ``integration`` 的真实窗口烟测外，本文件不依赖真实屏幕与真实等待。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from src.core.config import (
    AppConfig,
    EvidenceConfig,
    RuntimeStabilizationConfig,
    RuntimeStateMonitorConfig,
    load_config,
)
from src.core.exceptions import CaptureError, EvidenceError, WindowNotFoundError
from src.core.geometry import Rect
from src.runtime.state_monitor import (
    CycleMetrics,
    CycleScheduler,
    CycleStatus,
    ForegroundGuard,
    OnmyojiWindowSource,
    RuntimeMetricsRecorder,
    SkipReason,
    StateMonitor,
    StateStabilizer,
    WindowFrameSource,
)
from src.runtime.state_monitor import window_source as window_source_module
from src.utils.logger import get_logger
from src.vision.capture import CaptureBackend, Frame
from src.vision.evidence import BaseEvidenceDetector, EvidenceRegistry, EvidenceResult
from src.vision.pipeline import EvidenceFrameProcessor
from src.vision.state import StateClassifier, StateName, StateResult, StateSource
from src.vision.window import WindowInfo
from tests.monitor_fakes import FakeClassifier, FakeClock, FakeFrameSource

#: 测试用周期（毫秒）
INTERVAL_MS = 500
#: 真实窗口关键词（integration 烟测使用）
REAL_KEYWORDS: tuple[str, ...] = ("阴阳师", "Onmyoji")

logger = get_logger("tests.test_runtime_loop")


def make_frame() -> Frame:
    """构造一个极小的测试帧（循环只关心"有没有帧"）。"""
    return Frame.create(np.zeros((8, 8, 3), dtype=np.uint8), source="fake")


def make_window(
    *,
    hwnd: int = 0x1234,
    x: int = 100,
    y: int = 100,
    width: int = 800,
    height: int = 600,
) -> WindowInfo:
    """构造一个假的窗口信息。"""
    return WindowInfo(hwnd=hwnd, title="阴阳师-测试窗口", x=x, y=y, width=width, height=height)


class FakeBinding:
    """``OnmyojiWindowSource`` 的替身（不触碰真实窗口）。

    Args:
        window: 初始绑定窗口（``None`` = 未绑定）。
        find_result: :meth:`find` 的返回值；省略时返回当前绑定窗口。
        minimized / alive / foreground: 窗口状态模拟。
    """

    def __init__(
        self,
        window: WindowInfo | None = None,
        *,
        find_result: WindowInfo | None = None,
        keywords: Sequence[str] = REAL_KEYWORDS,
        match_mode: str = "contains",
        minimized: bool = False,
        alive: bool = True,
        foreground: bool = True,
    ) -> None:
        self._window = window
        self._find_result = find_result
        self._keywords = tuple(keywords)
        self._match_mode = match_mode
        self.minimized = minimized
        self.alive = alive
        self.foreground = foreground
        self.calls: dict[str, int] = {"find": 0, "require": 0, "clear": 0}

    @property
    def window(self) -> WindowInfo | None:
        """当前绑定窗口。"""
        return self._window

    @property
    def keywords(self) -> tuple[str, ...]:
        """标题关键词。"""
        return self._keywords

    @property
    def match_mode(self) -> str:
        """标题匹配模式。"""
        return self._match_mode

    def find(self, *, refresh: bool = False) -> WindowInfo | None:
        """模拟窗口查找（``find_result`` 优先）。"""
        self.calls["find"] += 1
        if self._find_result is not None:
            self._window = self._find_result
        return self._window

    def require(self) -> WindowInfo:
        """模拟几何刷新。"""
        self.calls["require"] += 1
        if self._window is None:
            raise WindowNotFoundError("窗口不存在")
        return self._window

    def clear(self) -> None:
        """清除绑定。"""
        self.calls["clear"] += 1
        self._window = None

    def retarget(self, window: WindowInfo) -> None:
        """直接替换绑定窗口（模拟窗口被拖动 / 缩放）。"""
        self._window = window

    def is_alive(self) -> bool:
        """是否仍可见。"""
        return self._window is not None and self.alive

    def is_minimized(self) -> bool:
        """是否最小化。"""
        return self.minimized

    def is_foreground(self) -> bool:
        """是否在前台。"""
        return self.foreground


class FakeBackend(CaptureBackend):
    """采集后端替身（可配置连续失败次数与产出图像尺寸）。"""

    name = "fake"

    def __init__(self, *, region: Rect, fail_times: int = 0, size: tuple[int, int] = (8, 8)) -> None:
        super().__init__(region=region, fps=1)
        self._fail_times = fail_times
        self._size = size
        self.grabs = 0
        self.stops = 0

    def _open(self) -> None:
        """无需资源。"""

    def _close(self) -> None:
        """记录释放次数。"""
        self.stops += 1

    def _grab_image(self) -> np.ndarray | None:
        """按配置失败或返回全黑图像。"""
        self.grabs += 1
        if self.grabs <= self._fail_times:
            raise CaptureError("模拟设备丢失")
        height, width = self._size
        return np.zeros((height, width, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def fake_backend_factory(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> list[tuple[object, WindowInfo | None]]:
    """把 ``WindowFrameSource`` 使用的后端工厂替换为替身（integration 用例除外）。

    Returns:
        ``[(capture_config, window), ...]`` 形式的工厂调用记录。
    """
    calls: list[tuple[object, WindowInfo | None]] = []
    if request.node.get_closest_marker("integration") is not None:
        return calls  # 真实窗口烟测需要真实采集后端

    def factory(
        config: object,
        *,
        window: WindowInfo | None = None,
        transform: object | None = None,
    ) -> FakeBackend:
        calls.append((config, window))
        region = window.rect if window is not None else Rect(x=0, y=0, width=8, height=8)
        return FakeBackend(region=region)

    monkeypatch.setattr(window_source_module, "create_capture_backend", factory)
    return calls


def build_window_source(
    binding: FakeBinding,
    *,
    guard: ForegroundGuard | None = None,
    guard_enabled: bool = False,
    clock: FakeClock | None = None,
    retry_interval_s: float = 1.0,
) -> WindowFrameSource:
    """用假绑定构造真实窗口帧来源（后端工厂由 autouse fixture 替换）。

    默认**关闭**前台门控：测试环境的前台窗口不是假窗口，门控会（正确地）阻止采集，
    从而干扰窗口/采集路径的验证。需要验证门控的用例显式注入 ``guard``。
    """
    config = load_config(environ={})
    active_clock = clock if clock is not None else FakeClock()
    return WindowFrameSource(
        binding,  # type: ignore[arg-type]
        capture=config.capture,
        calibration=config.calibration,
        guard=guard,
        guard_enabled=guard_enabled,
        retry_interval_s=retry_interval_s,
        clock=active_clock.clock,
    )


class FailingDetector(BaseEvidenceDetector):
    """总是抛异常的 detector 替身（模拟 detector / OCR / 模板失败）。"""

    def __init__(self, error: Exception, *, name: str = "bottom_menu_yard") -> None:
        self._error = error
        self._name = name

    @property
    def name(self) -> str:
        """证据名。"""
        return self._name

    def detect(self, frame: Frame) -> EvidenceResult:
        """抛出配置的异常。"""
        raise self._error


def failing_processor(error: Exception) -> EvidenceFrameProcessor:
    """构造"检测必失败"的真实证据流水线。"""
    registry = EvidenceRegistry([FailingDetector(error)])
    return EvidenceFrameProcessor(registry, StateClassifier.from_evidence_config(EvidenceConfig()))


def build_monitor_config(tmp_path: Path, *, interval_ms: int = INTERVAL_MS) -> RuntimeStateMonitorConfig:
    """构造测试用监控配置（不写历史文件）。"""
    return RuntimeStateMonitorConfig(
        interval_ms=interval_ms,
        save_history=False,
        history_path=str(tmp_path / "state_history.json"),
    )


def build_scripted_monitor(
    states: Sequence[StateName],
    tmp_path: Path,
    *,
    min_frames: int = 1,
    interval_ms: int = INTERVAL_MS,
    source: object | None = None,
    metrics: RuntimeMetricsRecorder | None = None,
    clock: FakeClock | None = None,
    processor: object | None = None,
    confidence: float = 0.9,
) -> tuple[StateMonitor, FakeClock]:
    """装配"假分类器 + 假帧来源 + scheduler + metrics"的监控器。"""
    active_clock = clock if clock is not None else FakeClock()
    recorder = metrics if metrics is not None else RuntimeMetricsRecorder()
    entry = processor if processor is not None else FakeClassifier(states, confidence=confidence)
    monitor = StateMonitor(
        entry,  # type: ignore[arg-type]
        source if source is not None else FakeFrameSource(),  # type: ignore[arg-type]
        build_monitor_config(tmp_path, interval_ms=interval_ms),
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(min_frames=min_frames)),
        scheduler=CycleScheduler(interval_ms, clock=active_clock.clock, sleep=active_clock.sleep),
        metrics=recorder,
        clock=active_clock.clock,
        sleep=active_clock.sleep,
        now=lambda: "2025-01-01T00:00:00+08:00",
    )
    return monitor, active_clock


# ---------------------------------------------------------------------------
# 1. 500ms scheduler 准确
# ---------------------------------------------------------------------------


def test_scheduler_keeps_deadline_without_drift() -> None:
    """4 轮 × 10ms 处理 + 500ms 周期：总时间恰好 2.0s，每轮等待 490ms。"""
    clock = FakeClock()
    scheduler = CycleScheduler(INTERVAL_MS, clock=clock.clock, sleep=clock.sleep)
    scheduler.reset(start_at=clock.clock())

    for _ in range(4):
        started = scheduler.begin()
        clock.value += 0.01  # 模拟 10ms 处理
        schedule = scheduler.finish(started)
        assert schedule.late is False
        assert schedule.sleep_ms == pytest.approx(490.0)

    assert clock.value == pytest.approx(2.0)
    assert clock.sleeps == pytest.approx([0.49] * 4)
    assert scheduler.cycles == 4
    assert scheduler.late_cycles == 0
    assert scheduler.re_alignments == 0


def test_scheduler_rejects_non_positive_interval() -> None:
    """周期必须为正数。"""
    with pytest.raises(ValueError):
        CycleScheduler(0)


def test_monitor_cycles_land_on_500ms_deadline(tmp_path: Path) -> None:
    """监控循环使用 scheduler：4 轮后时钟恰好前进 2.0s（无漂移）。"""
    monitor, clock = build_scripted_monitor([StateName.YARD], tmp_path)

    report = monitor.run(max_cycles=4)

    assert report.cycles == 4
    assert clock.value == pytest.approx(2.0)
    assert monitor.scheduler.cycles == 4
    assert monitor.last_schedule is not None
    assert monitor.last_schedule.interval_ms == pytest.approx(INTERVAL_MS)


# ---------------------------------------------------------------------------
# 2. 处理耗时超过 interval：仍继续 + late cycle 记录
# ---------------------------------------------------------------------------


class SlowFrameSource:
    """每轮消耗固定时长的帧来源（模拟真实环境里处理超过周期）。"""

    def __init__(self, clock: FakeClock, *, cost: float) -> None:
        self._clock = clock
        self._cost = cost
        self.calls = 0

    def get_frame(self) -> Frame | None:
        """推进时钟并返回一帧（0.8s > 0.5s 周期）。"""
        self.calls += 1
        self._clock.value += self._cost
        return make_frame()


def test_late_cycle_is_recorded_and_loop_continues(tmp_path: Path) -> None:
    """单轮 0.8s > 0.5s 周期：循环继续、每轮记为 late、不等待并记录 overrun。"""
    clock = FakeClock()
    recorder = RuntimeMetricsRecorder()
    monitor, _ = build_scripted_monitor(
        [StateName.YARD],
        tmp_path,
        source=SlowFrameSource(clock, cost=0.8),
        metrics=recorder,
        clock=clock,
    )

    report = monitor.run(max_cycles=3)

    assert report.cycles == 3  # 循环没有被超时打断
    assert report.classified == 3
    assert monitor.scheduler.late_cycles == 3
    assert monitor.scheduler.re_alignments == 3
    assert monitor.scheduler.overrun_total_ms == pytest.approx(900.0)  # 3 × (800 - 500)
    assert monitor.scheduler.overrun_max_ms == pytest.approx(300.0)
    assert clock.sleeps == []  # late 时不等待，避免追赶风暴
    assert recorder.summary().late_cycles == 3
    assert all(record.status is CycleStatus.OK for record in recorder.records)


# ---------------------------------------------------------------------------
# 3. 采集失败不中断
# ---------------------------------------------------------------------------


def test_missing_window_is_reported_and_loop_continues(tmp_path: Path) -> None:
    """窗口不存在 → 跳过该轮并记录原因，循环继续（不抛异常）。"""
    binding = FakeBinding(window=None, find_result=None)
    clock = FakeClock()
    source = build_window_source(binding, clock=clock)
    recorder = RuntimeMetricsRecorder()
    monitor, _ = build_scripted_monitor(
        [StateName.YARD], tmp_path, source=source, metrics=recorder, clock=clock
    )

    with source:
        report = monitor.run(max_cycles=3)

    assert report.cycles == 3
    assert report.skipped == 3
    assert report.classified == 0
    assert monitor.errors == 0
    assert source.last_skip_reason == SkipReason.WINDOW_NOT_FOUND.value
    # start() 也会尝试绑定一次，因此计数不少于轮数
    assert source.skip_counts[SkipReason.WINDOW_NOT_FOUND.value] >= report.skipped
    summary = recorder.summary()
    assert summary.skipped == 3
    assert summary.ok == 0
    assert all(
        record.skip_reason == SkipReason.WINDOW_NOT_FOUND.value for record in recorder.records
    )


def test_window_lookup_is_throttled_while_missing(tmp_path: Path) -> None:
    """窗口不存在时按冷却间隔重新查找（500ms 周期内不重复枚举窗口）。"""
    binding = FakeBinding(window=None, find_result=None)
    clock = FakeClock()
    source = build_window_source(binding, clock=clock, retry_interval_s=10.0)
    monitor, _ = build_scripted_monitor([StateName.YARD], tmp_path, source=source, clock=clock)

    with source:
        monitor.run(max_cycles=3)

    assert binding.calls["find"] == 1  # 冷却期内只查找一次


def test_minimized_window_is_reported(tmp_path: Path) -> None:
    """窗口最小化 → 明确报告 window_minimized（不误报"窗口不存在"）。"""
    binding = FakeBinding(window=make_window(), minimized=True)
    clock = FakeClock()
    source = build_window_source(binding, clock=clock)

    with source:
        frame = source.get_frame()

    assert frame is None
    assert source.last_skip_reason == SkipReason.WINDOW_MINIMIZED.value
    assert binding.calls["require"] == 0  # 最小化时无需读取几何


def test_lost_window_is_relooked_up(tmp_path: Path) -> None:
    """窗口句柄失效（已关闭）→ 清除绑定并按冷却重新查找。"""
    binding = FakeBinding(window=make_window(), alive=False, find_result=None)
    clock = FakeClock()
    source = build_window_source(binding, clock=clock)

    with source:
        frame = source.get_frame()

    assert frame is None
    assert binding.calls["clear"] == 1
    assert binding.calls["find"] == 1
    assert source.last_skip_reason == SkipReason.WINDOW_NOT_FOUND.value


def test_capture_failure_is_reported_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """后端抓帧失败 → 记录 capture_failed 并在下一轮重建后端后恢复。"""
    binding = FakeBinding(window=make_window())
    clock = FakeClock()
    source = build_window_source(binding, clock=clock)
    created = {"count": 0}

    def factory(
        config: object, *, window: WindowInfo | None = None, transform: object | None = None
    ) -> FakeBackend:
        created["count"] += 1
        region = window.rect if window is not None else Rect(x=0, y=0, width=8, height=8)
        return FakeBackend(region=region, fail_times=1 if created["count"] == 1 else 0)

    monkeypatch.setattr(window_source_module, "create_capture_backend", factory)

    with source:
        first = source.get_frame()
        second = source.get_frame()

    assert first is None
    assert source.skip_counts[SkipReason.CAPTURE_FAILED.value] == 1
    assert second is not None  # 下轮重建后端后成功
    assert source.frames == 1
    assert source.status is not None
    assert source.status.ok is True


def test_backend_unavailable_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """所有后端都不可用 → 记录 backend_unavailable，循环继续。"""
    binding = FakeBinding(window=make_window())
    clock = FakeClock()
    source = build_window_source(binding, clock=clock)
    recorder = RuntimeMetricsRecorder()
    monitor, _ = build_scripted_monitor(
        [StateName.YARD], tmp_path, source=source, metrics=recorder, clock=clock
    )

    def factory(
        config: object, *, window: WindowInfo | None = None, transform: object | None = None
    ) -> object:
        raise CaptureError("所有候选采集后端均不可用")

    monkeypatch.setattr(window_source_module, "create_capture_backend", factory)

    with source:
        report = monitor.run(max_cycles=2)

    assert report.cycles == 2
    assert report.skipped == 2
    # start() 也会尝试创建一次后端，因此计数不少于轮数
    assert source.skip_counts[SkipReason.BACKEND_UNAVAILABLE.value] >= report.skipped


def test_capture_config_is_forwarded_to_factory(
    fake_backend_factory: list[tuple[object, WindowInfo | None]],
) -> None:
    """``capture`` 配置（backend=auto → dxcam 优先、mss 回退）原样交给采集工厂。"""
    binding = FakeBinding(window=make_window())
    source = build_window_source(binding)

    with source:
        frame = source.get_frame()

    assert frame is not None
    assert fake_backend_factory, "后端工厂应被调用"
    config, window = fake_backend_factory[0]
    assert getattr(config, "backend") == "auto"
    assert window is not None


def test_window_region_change_rebuilds_backend(
    fake_backend_factory: list[tuple[object, WindowInfo | None]],
) -> None:
    """窗口移动导致采集区域变化 → 重建后端（采集区域跟随窗口）。"""
    binding = FakeBinding(window=make_window(x=10, y=10))
    source = build_window_source(binding)

    with source:
        assert source.get_frame() is not None
        binding.retarget(make_window(x=400, y=300))  # 窗口被拖动
        assert source.get_frame() is not None

    assert len(fake_backend_factory) == 2
    first_window = fake_backend_factory[0][1]
    second_window = fake_backend_factory[1][1]
    assert first_window is not None and second_window is not None
    assert first_window.rect != second_window.rect


# ---------------------------------------------------------------------------
# 4. 判定异常不中断
# ---------------------------------------------------------------------------


def test_processor_failure_does_not_interrupt_loop(tmp_path: Path) -> None:
    """证据流水线抛异常 → 降级为 unknown、计入 errors，循环继续且指标记录 error。"""
    recorder = RuntimeMetricsRecorder()
    monitor, _ = build_scripted_monitor(
        [StateName.YARD],
        tmp_path,
        processor=failing_processor(EvidenceError("模拟判定失败")),
        metrics=recorder,
    )

    report = monitor.run(max_cycles=2)

    assert report.cycles == 2
    assert report.skipped == 0
    assert monitor.errors == 2
    assert monitor.last_result is not None
    assert monitor.last_result.state_name is StateName.UNKNOWN
    assert monitor.last_result.source is StateSource.FALLBACK
    assert "evidence pipeline failure" in monitor.last_result.reason
    assert all(record.status is CycleStatus.ERROR for record in recorder.records)
    assert recorder.summary().errors == 2


def test_failure_in_one_frame_does_not_break_next_frame(tmp_path: Path) -> None:
    """失败帧之后的帧照常处理（异常不污染循环状态）。"""
    monitor, _ = build_scripted_monitor(
        [StateName.UNKNOWN, StateName.YARD], tmp_path, min_frames=1
    )

    monitor.step(make_frame())  # 第 1 帧：unknown
    monitor.step(make_frame())  # 第 2 帧：yard

    assert monitor.errors == 0
    assert monitor.last_result is not None
    assert monitor.last_result.state_name is StateName.YARD


# ---------------------------------------------------------------------------
# 5. 前台门控
# ---------------------------------------------------------------------------


def test_foreground_guard_only_checks() -> None:
    """门控只做检查：不在前台返回 blocked，不改变窗口状态（不抢焦点）。"""
    calls: list[int] = []

    def foreground() -> int:
        calls.append(1)
        return 0x9999  # 别的窗口在前台

    guard = ForegroundGuard(hwnd=0x1234, foreground=foreground)

    verdict = guard.verdict()

    assert verdict.allowed is False
    assert verdict.reason is SkipReason.NOT_FOREGROUND
    assert verdict.foreground_hwnd == 0x9999
    assert guard.skips == 1
    assert calls == [1]
    # 门控不提供任何激活 / 切换窗口的入口
    assert not hasattr(guard, "activate")
    assert not hasattr(guard, "set_foreground")


def test_foreground_guard_allows_when_target_is_foreground() -> None:
    """目标窗口就是前台窗口时放行。"""
    guard = ForegroundGuard(hwnd=0x1234, foreground=lambda: 0x1234)

    assert guard.allows() is True
    assert guard.skips == 0
    assert guard.checks == 1


def test_foreground_guard_reports_unbound_window() -> None:
    """尚未绑定窗口时门控不予放行（并说明原因）。"""
    guard = ForegroundGuard(hwnd=0, foreground=lambda: 0x1234)

    verdict = guard.verdict()

    assert verdict.allowed is False
    assert verdict.reason is SkipReason.WINDOW_NOT_FOUND


def test_window_source_skips_when_not_foreground(tmp_path: Path) -> None:
    """目标窗口不在前台 → 本轮跳过（不采集、不抢焦点），循环继续。"""
    window = make_window()
    binding = FakeBinding(window=window, foreground=False)
    clock = FakeClock()
    guard = ForegroundGuard(hwnd=window.hwnd, foreground=lambda: 0x7777)
    source = build_window_source(binding, guard=guard, clock=clock)
    recorder = RuntimeMetricsRecorder()
    monitor, _ = build_scripted_monitor(
        [StateName.YARD], tmp_path, source=source, metrics=recorder, clock=clock
    )

    with source:
        report = monitor.run(max_cycles=3)

    assert report.cycles == 3
    assert report.skipped == 3
    assert guard.skips == 3
    assert source.last_skip_reason == SkipReason.NOT_FOREGROUND.value
    assert source.frames == 0
    assert all(record.skip_reason == SkipReason.NOT_FOREGROUND.value for record in recorder.records)


def test_window_source_enables_guard_from_config() -> None:
    """``from_config`` 默认启用前台门控（真实运行的安全边界），“不抢焦点”是默认行为。"""
    config = load_config(environ={})
    source = WindowFrameSource.from_config(config, binding=FakeBinding(window=make_window()))  # type: ignore[arg-type]

    assert source.guard.enabled is True
    assert source.guard.hwnd == 0  # 尚未绑定，绑定后由 retarget 更新


def test_window_source_captures_while_guard_allows() -> None:
    """门控放行（目标窗口在前台）时正常采集，且状态标记 foreground=True。"""
    window = make_window()
    binding = FakeBinding(window=window)
    guard = ForegroundGuard(hwnd=window.hwnd, foreground=lambda: window.hwnd)
    source = build_window_source(binding, guard=guard, guard_enabled=True)

    with source:
        frame = source.get_frame()

    assert frame is not None
    assert source.guard.enabled is True
    assert source.status is not None
    assert source.status.foreground is True


def test_window_source_reports_not_started() -> None:
    """未 ``start()`` 就取帧 → 明确报告（不崩溃）。"""
    binding = FakeBinding(window=make_window())
    source = build_window_source(binding)

    assert source.get_frame() is None
    assert source.last_skip_reason == SkipReason.NOT_STARTED.value


# ---------------------------------------------------------------------------
# 6. 事件逻辑不变
# ---------------------------------------------------------------------------


def test_events_only_on_stable_change(tmp_path: Path) -> None:
    """3 帧命中建立基线（无事件），随后 3 帧未命中 → 恰好 1 个事件。"""
    monitor, _ = build_scripted_monitor(
        [*[StateName.YARD] * 3, *[StateName.UNKNOWN] * 3], tmp_path
    )

    report = monitor.run(max_cycles=6)

    assert report.initial_state is StateName.YARD
    assert report.final_state is StateName.UNKNOWN
    assert len(report.events) == 1
    assert report.events[0].previous_state is StateName.YARD
    assert report.events[0].current_state is StateName.UNKNOWN


def test_single_frame_jitter_produces_no_event(tmp_path: Path) -> None:
    """单帧抖动不产生事件（稳定化仍然先行）。"""
    monitor, _ = build_scripted_monitor(
        [StateName.YARD, StateName.YARD, StateName.YARD, StateName.BATTLE, StateName.YARD],
        tmp_path,
        min_frames=3,
    )

    report = monitor.run(max_cycles=5)

    assert report.initial_state is StateName.YARD
    assert report.final_state is StateName.YARD
    assert report.events == ()


# ---------------------------------------------------------------------------
# 7. StateStabilizer 继续工作
# ---------------------------------------------------------------------------


def test_stabilizer_requires_consecutive_frames(tmp_path: Path) -> None:
    """连续 3 帧同一状态才确认（前两帧为 pending）。"""
    monitor, _ = build_scripted_monitor([StateName.YARD], tmp_path, min_frames=3)
    states: list[StateName | None] = []

    for _ in range(3):
        monitor.step(make_frame())
        states.append(monitor.last_stable.state_name if monitor.last_stable else None)

    assert states[0] is None
    assert states[1] is None
    assert states[2] is StateName.YARD


def test_stabilizer_confidence_threshold_still_applied(tmp_path: Path) -> None:
    """置信度低于稳定化门槛的帧不计入统计（0.7 < 0.85）。"""
    monitor, _ = build_scripted_monitor(
        [StateName.YARD], tmp_path, min_frames=3, confidence=0.7
    )

    for _ in range(3):
        monitor.step(make_frame())

    assert monitor.last_result is not None
    assert monitor.last_result.state_name is StateName.YARD  # 判定成立
    assert monitor.current_state is None  # 但 0.7 < 0.85，永不确认
    assert monitor.last_stable is not None
    assert monitor.last_stable.counted is False


# ---------------------------------------------------------------------------
# 8. 运行指标
# ---------------------------------------------------------------------------


def test_metrics_records_required_fields(tmp_path: Path) -> None:
    """指标字段齐全（cycle_id / timestamp / state / stable_state / latency），且不含截图与用户数据。"""
    metrics_path = tmp_path / "runtime_metrics.jsonl"
    recorder = RuntimeMetricsRecorder(path=metrics_path)
    monitor, _ = build_scripted_monitor([StateName.YARD], tmp_path, metrics=recorder)

    monitor.run(max_cycles=2)

    payloads = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(payloads) == 2
    first = payloads[0]
    assert set(first) == {
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
    assert first["cycle_id"] == 1
    assert first["state"] == StateName.YARD.value
    assert first["status"] == CycleStatus.OK.value
    assert first["latency_ms"] >= 0
    # 不保存截图 / 图像内容 / 窗口标题等用户数据
    assert "image" not in first
    assert "screenshot" not in first
    assert "window" not in first


def test_metrics_records_skip_reason(tmp_path: Path) -> None:
    """跳过轮次的指标带有原因（便于长跑后统计"为什么没有帧"）。"""
    recorder = RuntimeMetricsRecorder()
    clock = FakeClock()
    source = build_window_source(FakeBinding(window=None, find_result=None), clock=clock)
    monitor, _ = build_scripted_monitor(
        [StateName.YARD], tmp_path, source=source, metrics=recorder, clock=clock
    )

    with source:
        monitor.run(max_cycles=2)

    assert recorder.skip_counts() == {SkipReason.WINDOW_NOT_FOUND.value: 2}
    assert recorder.records[0].stable_state is None
    assert recorder.records[0].frame_source is None


def test_metrics_write_failure_is_tolerated(tmp_path: Path) -> None:
    """指标写入失败不影响主循环（只累计 write_errors）。"""
    blocked = tmp_path / "blocked"
    blocked.mkdir()  # 目录不能当作文件追加写入
    recorder = RuntimeMetricsRecorder(path=blocked)
    monitor, _ = build_scripted_monitor([StateName.YARD], tmp_path, metrics=recorder)

    report = monitor.run(max_cycles=2)

    assert report.cycles == 2
    assert recorder.write_errors == 2
    assert len(recorder.records) == 2  # 内存记录仍然完整


def test_metrics_summary_statistics(tmp_path: Path) -> None:
    """汇总统计：fps / mean / p95 / skipped / errors。"""
    recorder = RuntimeMetricsRecorder()
    monitor, _ = build_scripted_monitor([StateName.YARD], tmp_path, metrics=recorder)

    monitor.run(max_cycles=20)

    summary = recorder.summary()
    assert summary.cycles == 20
    assert summary.ok == 20
    assert summary.skipped == 0
    assert summary.fps == pytest.approx(2.0, abs=0.15)
    assert summary.p95_latency_ms >= summary.mean_latency_ms >= 0.0
    assert summary.describe()


def test_metrics_from_config_defaults_to_memory_only(tmp_path: Path) -> None:
    """``save_metrics=False``（默认）时只保留内存记录，不落盘。"""
    config = build_monitor_config(tmp_path)

    recorder = RuntimeMetricsRecorder.from_config(config)
    recorder.record(
        CycleMetrics(
            cycle_id=1,
            timestamp="2025-01-01T00:00:00+08:00",
            started_at=0.0,
            latency_ms=1.0,
            frame_ms=0.5,
            state=StateName.YARD,
            confidence=0.9,
            stable_state=StateName.YARD,
            status=CycleStatus.OK,
        )
    )

    assert recorder.path is None
    assert recorder.summary().cycles == 1
    assert not (tmp_path / "runtime_metrics.jsonl").exists()


def test_metrics_from_config_writes_when_enabled(tmp_path: Path) -> None:
    """``save_metrics=True`` 时写入 ``metrics_path``（相对项目根解析）。"""
    config = build_monitor_config(tmp_path).model_copy(
        update={"save_metrics": True, "metrics_path": str(tmp_path / "metrics.jsonl")}
    )

    recorder = RuntimeMetricsRecorder.from_config(config)

    assert recorder.path == tmp_path / "metrics.jsonl"


# ---------------------------------------------------------------------------
# 真实窗口烟测（integration）
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_window_source_smoke() -> None:
    """真实窗口烟测：绑定 / 采集 / 跳过都不抛异常（无窗口时只验证跳过路径）。"""
    config: AppConfig = load_config(environ={})
    binding = OnmyojiWindowSource.from_config(config, keywords=REAL_KEYWORDS)
    source = WindowFrameSource.from_config(config, binding=binding)

    with source:
        frame = source.get_frame()

    if frame is None:
        assert source.last_skip_reason is not None
        logger.info("真实窗口不可用，跳过原因：%s", source.last_skip_reason)
    else:
        assert frame.width > 0 and frame.height > 0
        assert frame.source in {"dxcam", "mss"}
        assert frame.coordinate_transform is not None
