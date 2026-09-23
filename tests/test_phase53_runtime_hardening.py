"""Phase 5.3 测试：Runtime 架构加固（模块拆分边界 / 诊断层 / 指标兼容）。

覆盖：

1. **旧 import 仍工作**：``window_source`` 兼容层导出的 7 个符号与新模块**同一对象**；
2. **拆分前后行为一致**：同一批替身场景（正常 / 抓帧失败 / 恢复 / 最小化 / 失焦 / 句柄失效）
   走旧路径与新路径得到**完全相同**的结果序列（行为快照回归护栏）；
3. **capture 失败不崩**：后端不可用 / 抓帧失败 / 窗口丢失都不会中断循环；
4. **foreground skip 正确**：失焦 → 跳过 + 诊断 ``blocked`` / ``not_foreground``；
5. **unknown reason 正确**：``no_evidence`` / ``unsupported_scene`` / ``detector_failure`` /
   ``not_foreground`` / ``skipped`` 五类判据，以及"状态明确时无 unknown 原因"；
6. **metrics schema 兼容**：既有字段名与含义不变，诊断字段为**追加**；旧构造方式仍可用。

本文件不依赖真实屏幕与真实等待。
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from src.core.config import (
    EvidenceConfig,
    RuntimeStabilizationConfig,
    RuntimeStateMonitorConfig,
    load_config,
)
from src.core.exceptions import CaptureError, EvidenceError, WindowNotFoundError
from src.core.geometry import Rect
from src.runtime.state_monitor import (
    CaptureOutcome,
    CycleMetrics,
    CycleScheduler,
    CycleStatus,
    ForegroundGuard,
    ForegroundOutcome,
    RuntimeDiagnostics,
    RuntimeMetricsRecorder,
    SkipReason,
    StateMonitor,
    StateStabilizer,
    UnknownReason,
    WindowFrameSource,
    resolve_unknown_reason,
)
from src.runtime.state_monitor import foreground_guard, frame_source, window_binding
from src.runtime.state_monitor import window_source as legacy_window_source
from src.vision.capture import CaptureBackend, Frame
from src.vision.evidence import BaseEvidenceDetector, EvidenceRegistry, EvidenceResult
from src.vision.pipeline import EvidenceFrameProcessor
from src.vision.state import StateClassifier, StateName, StateResult, StateSource
from src.vision.window import WindowInfo
from tests.monitor_fakes import FakeClassifier, FakeClock, FakeFrameSource

#: 测试用周期（毫秒）
INTERVAL_MS = 500
#: 旧兼容层必须保留的符号
LEGACY_SYMBOLS: tuple[str, ...] = (
    "CaptureStatus",
    "DEFAULT_RETRY_INTERVAL_S",
    "ForegroundGuard",
    "GuardVerdict",
    "OnmyojiWindowSource",
    "SkipReason",
    "WindowFrameSource",
)
#: 场景共享状态（由 autouse fixture 在每个测试前重置）
SCENARIO: dict[str, object] = {"grab_failures": [], "foreground": 0}


def make_frame() -> Frame:
    """构造一个极小的测试帧。"""
    return Frame.create(np.zeros((8, 8, 3), dtype=np.uint8), source="fake")


def make_window(*, hwnd: int = 0x2468, x: int = 50, y: int = 60) -> WindowInfo:
    """构造一个假窗口。"""
    return WindowInfo(hwnd=hwnd, title="阴阳师-加固测试", x=x, y=y, width=800, height=600)


class MiniBinding:
    """``OnmyojiWindowSource`` 的最小替身（可切换窗口状态）。"""

    def __init__(
        self,
        window: WindowInfo | None = None,
        *,
        minimized: bool = False,
        alive: bool = True,
    ) -> None:
        self._window = window
        self.minimized = minimized
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
        """模拟查找（不新建绑定）。"""
        self.calls["find"] += 1
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

    def is_alive(self) -> bool:
        """是否仍可见。"""
        return self._window is not None and self.alive

    def is_minimized(self) -> bool:
        """是否最小化。"""
        return self.minimized


class MiniBackend(CaptureBackend):
    """采集后端替身；按 :data:`SCENARIO` 中的脚本决定某次抓帧是否失败。"""

    name = "fake"

    def __init__(self, *, region: Rect) -> None:
        super().__init__(region=region, fps=1)
        self.grabs = 0

    def _open(self) -> None:
        """无需资源。"""

    def _close(self) -> None:
        """无需清理。"""

    def _grab_image(self) -> np.ndarray | None:
        """按场景脚本失败或返回全黑图像。"""
        self.grabs += 1
        failures = SCENARIO["grab_failures"]
        assert isinstance(failures, list)
        if failures and failures.pop(0):
            raise CaptureError("模拟设备丢失")
        return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """把采集后端工厂替换为替身，并重置场景状态。"""
    SCENARIO["grab_failures"] = []
    SCENARIO["foreground"] = make_window().hwnd

    def factory(
        config: object, *, window: WindowInfo | None = None, transform: object | None = None
    ) -> MiniBackend:
        region = window.rect if window is not None else Rect(x=0, y=0, width=8, height=8)
        return MiniBackend(region=region)

    monkeypatch.setattr(frame_source, "create_capture_backend", factory)


def scenario_foreground() -> int:
    """返回当前场景下的前台窗口句柄（``0`` = 没有前台窗口）。"""
    value = SCENARIO["foreground"]
    return int(value)  # type: ignore[arg-type]


def make_guard(hwnd: int) -> ForegroundGuard:
    """构造读取场景状态的前台门控。"""
    return ForegroundGuard(hwnd=hwnd, foreground=scenario_foreground)


def make_source(
    binding: MiniBinding,
    *,
    guard: ForegroundGuard | None = None,
    guard_enabled: bool = False,
    clock: FakeClock | None = None,
    retry_interval_s: float = 1.0,
    factory_class: type[WindowFrameSource] = WindowFrameSource,
) -> WindowFrameSource:
    """构造真实窗口帧来源（采集工厂由 autouse fixture 替换）。"""
    config = load_config(environ={})
    active_clock = clock if clock is not None else FakeClock()
    return factory_class(
        binding,  # type: ignore[arg-type]
        capture=config.capture,
        calibration=config.calibration,
        guard=guard,
        guard_enabled=guard_enabled,
        retry_interval_s=retry_interval_s,
        clock=active_clock.clock,
    )


class ScriptedDetector(BaseEvidenceDetector):
    """按脚本命中/未命中的 detector 替身。"""

    def __init__(self, *, name: str, script: Sequence[bool], confidence: float = 0.9) -> None:
        self._name = name
        self._script = list(script)
        self._confidence = confidence
        self.calls = 0

    @property
    def name(self) -> str:
        """证据名。"""
        return self._name

    def detect(self, frame: Frame) -> EvidenceResult:
        """按脚本返回证据。"""
        hit = self._script[self.calls % len(self._script)]
        self.calls += 1
        if not hit:
            return EvidenceResult.miss(self._name, reliability=0.9)
        return EvidenceResult(
            name=self._name, detected=True, confidence=self._confidence, reliability=0.9
        )


class FailingDetector(BaseEvidenceDetector):
    """总是抛异常的 detector 替身。"""

    def __init__(self, *, name: str = "bottom_menu_yard") -> None:
        self._name = name

    @property
    def name(self) -> str:
        """证据名。"""
        return self._name

    def detect(self, frame: Frame) -> EvidenceResult:
        """抛出证据层异常。"""
        raise EvidenceError("模拟判定失败")


def evidence_processor(*detectors: BaseEvidenceDetector) -> EvidenceFrameProcessor:
    """用给定 detector 构造真实证据流水线（不依赖数据集）。"""
    registry = EvidenceRegistry(list(detectors))
    return EvidenceFrameProcessor(registry, StateClassifier.from_evidence_config(EvidenceConfig()))


def build_monitor(
    tmp_path: Path,
    *,
    source: object,
    processor: object | None = None,
    states: Sequence[StateName] = (StateName.UNKNOWN,),
    metrics: RuntimeMetricsRecorder | None = None,
    clock: FakeClock | None = None,
) -> tuple[StateMonitor, RuntimeMetricsRecorder, FakeClock]:
    """装配带 metrics 的监控器。"""
    active_clock = clock if clock is not None else FakeClock()
    recorder = metrics if metrics is not None else RuntimeMetricsRecorder()
    entry = processor if processor is not None else FakeClassifier(list(states))
    monitor = StateMonitor(
        entry,  # type: ignore[arg-type]
        source,  # type: ignore[arg-type]
        RuntimeStateMonitorConfig(
            interval_ms=INTERVAL_MS,
            save_history=False,
            history_path=str(tmp_path / "state_history.json"),
        ),
        stabilizer=StateStabilizer(RuntimeStabilizationConfig(min_frames=1)),
        scheduler=CycleScheduler(INTERVAL_MS, clock=active_clock.clock, sleep=active_clock.sleep),
        metrics=recorder,
        clock=active_clock.clock,
        sleep=active_clock.sleep,
        now=lambda: "2025-01-01T00:00:00+08:00",
    )
    return monitor, recorder, active_clock


# ---------------------------------------------------------------------------
# 1. 旧 import 仍工作
# ---------------------------------------------------------------------------


def test_legacy_window_source_import_path_still_works() -> None:
    """``src.runtime.state_monitor.window_source`` 仍导出全部旧符号。"""
    module = importlib.import_module("src.runtime.state_monitor.window_source")

    for name in LEGACY_SYMBOLS:
        assert hasattr(module, name), f"兼容层缺少旧符号: {name}"
    assert set(module.__all__) == set(LEGACY_SYMBOLS)


def test_split_symbols_are_identical_objects() -> None:
    """兼容层与新模块导出的是**同一个对象**（isinstance / 异常处理完全一致）。"""
    assert legacy_window_source.WindowFrameSource is frame_source.WindowFrameSource
    assert legacy_window_source.CaptureStatus is frame_source.CaptureStatus
    assert (
        legacy_window_source.DEFAULT_RETRY_INTERVAL_S is frame_source.DEFAULT_RETRY_INTERVAL_S
    )
    assert legacy_window_source.OnmyojiWindowSource is window_binding.OnmyojiWindowSource
    assert legacy_window_source.ForegroundGuard is foreground_guard.ForegroundGuard
    assert legacy_window_source.GuardVerdict is foreground_guard.GuardVerdict
    assert legacy_window_source.SkipReason is foreground_guard.SkipReason


def test_new_modules_import_paths_work() -> None:
    """三个新模块可被直接导入（推荐路径）。"""
    assert frame_source.WindowFrameSource.__name__ == "WindowFrameSource"
    assert window_binding.OnmyojiWindowSource.__name__ == "OnmyojiWindowSource"
    assert foreground_guard.ForegroundGuard.__name__ == "ForegroundGuard"


# ---------------------------------------------------------------------------
# 2. 拆分前后行为一致
# ---------------------------------------------------------------------------

#: 行为快照：正常 / 抓帧失败 / 恢复 / 最小化 / 失焦 / 句柄失效
BEHAVIOUR_EXPECTED: list[tuple[bool, str | None]] = [
    (True, None),
    (False, SkipReason.CAPTURE_FAILED.value),
    (True, None),
    (False, SkipReason.WINDOW_MINIMIZED.value),
    (False, SkipReason.NOT_FOREGROUND.value),
    (False, SkipReason.WINDOW_NOT_FOUND.value),
]


def run_behaviour_scenario(
    factory_class: type[WindowFrameSource], *, clock: FakeClock
) -> list[tuple[bool, str | None]]:
    """跑一遍固定的场景脚本，返回每轮 ``(是否取到帧, 跳过原因)``。"""
    window = make_window()
    binding = MiniBinding(window=window)
    guard = make_guard(window.hwnd)
    source = make_source(
        binding, guard=guard, guard_enabled=True, clock=clock, factory_class=factory_class
    )
    # 场景脚本：第 2 次抓帧失败，其余成功；初始目标窗口在前台
    SCENARIO["grab_failures"] = [False, True, False, False, False, False]
    SCENARIO["foreground"] = window.hwnd

    results: list[tuple[bool, str | None]] = []
    with source:
        results.append((source.get_frame() is not None, source.last_skip_reason))  # 正常
        results.append((source.get_frame() is not None, source.last_skip_reason))  # 抓帧失败
        results.append((source.get_frame() is not None, source.last_skip_reason))  # 重建后恢复
        binding.minimized = True
        results.append((source.get_frame() is not None, source.last_skip_reason))  # 最小化
        binding.minimized = False
        SCENARIO["foreground"] = 0  # 其它窗口在前台
        results.append((source.get_frame() is not None, source.last_skip_reason))  # 失焦
        SCENARIO["foreground"] = window.hwnd
        binding.alive = False  # 句柄失效
        results.append((source.get_frame() is not None, source.last_skip_reason))
    return results


def test_split_preserves_behaviour() -> None:
    """同一场景脚本下，旧兼容路径与新路径给出完全一致的结果序列。"""
    new_results = run_behaviour_scenario(frame_source.WindowFrameSource, clock=FakeClock())
    legacy_results = run_behaviour_scenario(
        legacy_window_source.WindowFrameSource, clock=FakeClock()
    )

    assert new_results == BEHAVIOUR_EXPECTED
    assert legacy_results == BEHAVIOUR_EXPECTED
    assert new_results == legacy_results


def test_behaviour_includes_skip_reason_counter() -> None:
    """拆分后跳过原因统计仍按原因累计（诊断与指标同源）。"""
    window = make_window()
    binding = MiniBinding(window=window)
    source = make_source(binding, guard=make_guard(window.hwnd), guard_enabled=True)
    SCENARIO["foreground"] = 0

    with source:
        source.get_frame()
        source.get_frame()

    assert source.skip_counts == {SkipReason.NOT_FOREGROUND.value: 2}
    assert source.frames == 0
    assert source.last_skip_reason == SkipReason.NOT_FOREGROUND.value


# ---------------------------------------------------------------------------
# 3. capture 失败不崩
# ---------------------------------------------------------------------------


def test_backend_unavailable_does_not_crash_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """所有后端不可用 → 每轮跳过、循环跑满、诊断标记为 runtime failure。"""
    binding = MiniBinding(window=make_window())
    source = make_source(binding)
    monitor, recorder, _ = build_monitor(tmp_path, source=source)

    def factory(
        config: object, *, window: WindowInfo | None = None, transform: object | None = None
    ) -> object:
        raise CaptureError("所有候选采集后端均不可用")

    monkeypatch.setattr(frame_source, "create_capture_backend", factory)

    with source:
        report = monitor.run(max_cycles=3)

    assert report.cycles == 3  # 没有崩溃
    assert report.skipped == 3
    assert source.skip_counts[SkipReason.BACKEND_UNAVAILABLE.value] >= 3
    health = monitor.health
    assert health.processor_ok is True  # 判定链路没坏
    assert health.capture_ok is False
    assert health.last_error
    assert recorder.records[-1].capture_status is CaptureOutcome.SKIPPED
    assert recorder.records[-1].unknown_reason is UnknownReason.SKIPPED


def test_capture_failure_recovers_within_loop(tmp_path: Path) -> None:
    """抓帧失败后自动重建后端恢复（循环继续，不崩）。"""
    binding = MiniBinding(window=make_window())
    source = make_source(binding)
    monitor, recorder, _ = build_monitor(tmp_path, source=source)
    SCENARIO["grab_failures"] = [True, False, False]

    with source:
        report = monitor.run(max_cycles=3)

    assert report.cycles == 3
    assert report.skipped == 1  # 仅第 1 轮失败
    assert report.classified == 2
    assert recorder.records[0].skip_reason == SkipReason.CAPTURE_FAILED.value
    assert recorder.records[-1].status is CycleStatus.OK


def test_missing_window_does_not_crash_loop(tmp_path: Path) -> None:
    """窗口不存在 → 跳过（不是崩溃），且保持节拍。"""
    binding = MiniBinding(window=None)
    source = make_source(binding, retry_interval_s=10.0)
    clock = FakeClock()
    monitor, recorder, _ = build_monitor(tmp_path, source=source, clock=clock)

    with source:
        report = monitor.run(max_cycles=4)

    assert report.cycles == 4
    assert report.skipped == 4
    assert clock.value == pytest.approx(2.0)  # 500ms × 4
    assert all(record.unknown_reason is UnknownReason.SKIPPED for record in recorder.records)


# ---------------------------------------------------------------------------
# 4. foreground skip 正确
# ---------------------------------------------------------------------------


def test_foreground_skip_is_reported_in_diagnostics(tmp_path: Path) -> None:
    """失焦 → 跳过 + 诊断 ``blocked`` / ``not_foreground``，且不抢焦点。"""
    window = make_window()
    binding = MiniBinding(window=window)
    guard = make_guard(window.hwnd)
    source = make_source(binding, guard=guard, guard_enabled=True)
    monitor, recorder, _ = build_monitor(tmp_path, source=source)
    SCENARIO["foreground"] = 0

    with source:
        report = monitor.run(max_cycles=2)

    assert report.skipped == 2
    assert guard.skips == 2
    assert guard.checks == 2
    assert not hasattr(guard, "activate")  # 门控不提供抢焦点入口
    first = recorder.records[0]
    assert first.foreground_status is ForegroundOutcome.BLOCKED
    assert first.capture_status is CaptureOutcome.SKIPPED
    assert first.unknown_reason is UnknownReason.NOT_FOREGROUND
    health = monitor.health
    assert health.foreground_ok is False
    assert health.is_vision_unknown is True  # "没有画面"属于 vision unknown，而非故障
    assert health.is_runtime_failure is False


def test_foreground_pass_is_reported_in_diagnostics(tmp_path: Path) -> None:
    """门控通过 → 诊断 ``ok``，``unknown`` 原因来自视觉侧而不是门控。"""
    window = make_window()
    binding = MiniBinding(window=window)
    guard = make_guard(window.hwnd)
    source = make_source(binding, guard=guard, guard_enabled=True)
    processor = evidence_processor(ScriptedDetector(name="bottom_menu_yard", script=[False]))
    monitor, recorder, _ = build_monitor(tmp_path, source=source, processor=processor)

    with source:
        report = monitor.run(max_cycles=2)

    assert report.classified == 2
    first = recorder.records[0]
    assert first.foreground_status is ForegroundOutcome.OK
    assert first.capture_status is CaptureOutcome.OK
    assert first.unknown_reason is UnknownReason.NO_EVIDENCE
    assert monitor.health.foreground_ok is True


# ---------------------------------------------------------------------------
# 5. unknown reason 正确
# ---------------------------------------------------------------------------


def test_unknown_reason_no_evidence(tmp_path: Path) -> None:
    """证据全未命中 → ``no_evidence``（链路正常，画面里没有已知元素）。"""
    binding = MiniBinding(window=make_window())
    source = make_source(binding)
    processor = evidence_processor(ScriptedDetector(name="bottom_menu_yard", script=[False]))
    monitor, recorder, _ = build_monitor(tmp_path, source=source, processor=processor)

    with source:
        monitor.run(max_cycles=2)

    assert monitor.last_result is not None
    assert monitor.last_result.state_name is StateName.UNKNOWN
    assert monitor.last_evidence_hits == 0
    assert recorder.records[-1].unknown_reason is UnknownReason.NO_EVIDENCE
    assert monitor.health.is_vision_unknown is True
    assert monitor.health.is_runtime_failure is False


def test_unknown_reason_unsupported_scene(tmp_path: Path) -> None:
    """有证据命中但没有任何规则满足 → ``unsupported_scene``（界面不在覆盖范围）。"""
    binding = MiniBinding(window=make_window())
    source = make_source(binding)
    # battle_action_bar 命中 1 条：battle 规则要求至少 2 个不同视觉区域 → 不满足
    processor = evidence_processor(ScriptedDetector(name="battle_action_bar", script=[True]))
    monitor, recorder, _ = build_monitor(tmp_path, source=source, processor=processor)

    with source:
        monitor.run(max_cycles=2)

    assert monitor.last_result is not None
    assert monitor.last_result.state_name is StateName.UNKNOWN
    assert monitor.last_evidence_hits == 1
    assert recorder.records[-1].unknown_reason is UnknownReason.UNSUPPORTED_SCENE


def test_unknown_reason_detector_failure(tmp_path: Path) -> None:
    """判定异常降级 → ``detector_failure`` + ``processor_ok=False``。"""
    binding = MiniBinding(window=make_window())
    source = make_source(binding)
    processor = evidence_processor(FailingDetector())
    monitor, recorder, _ = build_monitor(tmp_path, source=source, processor=processor)

    with source:
        monitor.run(max_cycles=2)

    assert monitor.errors == 2
    assert recorder.records[-1].status is CycleStatus.ERROR
    assert recorder.records[-1].unknown_reason is UnknownReason.DETECTOR_FAILURE
    health = monitor.health
    assert health.processor_ok is False
    assert health.is_runtime_failure is True
    assert health.last_error


def test_unknown_reason_not_foreground(tmp_path: Path) -> None:
    """门控跳过 → ``not_foreground``。"""
    window = make_window()
    binding = MiniBinding(window=window)
    source = make_source(binding, guard=make_guard(window.hwnd), guard_enabled=True)
    monitor, recorder, _ = build_monitor(tmp_path, source=source)
    SCENARIO["foreground"] = 0

    with source:
        monitor.run(max_cycles=1)

    assert recorder.records[-1].unknown_reason is UnknownReason.NOT_FOREGROUND


def test_unknown_reason_skipped_for_other_capture_problems(tmp_path: Path) -> None:
    """其它采集跳过（窗口最小化等）→ ``skipped``。"""
    binding = MiniBinding(window=make_window(), minimized=True)
    source = make_source(binding)
    monitor, recorder, _ = build_monitor(tmp_path, source=source)

    with source:
        monitor.run(max_cycles=1)

    assert recorder.records[-1].skip_reason == SkipReason.WINDOW_MINIMIZED.value
    assert recorder.records[-1].unknown_reason is UnknownReason.SKIPPED


def test_unknown_reason_none_when_state_is_known(tmp_path: Path) -> None:
    """状态明确时没有 unknown 原因（诊断字段为 ``None``）。"""
    binding = MiniBinding(window=make_window())
    source = make_source(binding)
    processor = evidence_processor(ScriptedDetector(name="bottom_menu_yard", script=[True]))
    monitor, recorder, _ = build_monitor(tmp_path, source=source, processor=processor)

    with source:
        monitor.run(max_cycles=1)

    assert monitor.last_result is not None
    assert monitor.last_result.state_name is StateName.YARD
    assert recorder.records[-1].unknown_reason is None


@pytest.mark.parametrize(
    ("skip_reason", "expected"),
    [
        (SkipReason.NOT_FOREGROUND.value, UnknownReason.NOT_FOREGROUND),
        (SkipReason.WINDOW_NOT_FOUND.value, UnknownReason.SKIPPED),
        (SkipReason.CAPTURE_FAILED.value, UnknownReason.SKIPPED),
        (SkipReason.BACKEND_UNAVAILABLE.value, UnknownReason.SKIPPED),
    ],
)
def test_resolve_unknown_reason_for_skipped_cycles(
    skip_reason: str, expected: UnknownReason
) -> None:
    """纯函数：本轮没有画面时按跳过原因分类。"""
    assert resolve_unknown_reason(None, skip_reason=skip_reason) is expected


@pytest.mark.parametrize(
    ("state", "source", "evidence", "hits", "expected"),
    [
        (StateName.UNKNOWN, StateSource.EVIDENCE, (), 0, UnknownReason.NO_EVIDENCE),
        (StateName.UNKNOWN, StateSource.EVIDENCE, (), 2, UnknownReason.UNSUPPORTED_SCENE),
        (StateName.UNKNOWN, StateSource.FALLBACK, (), None, UnknownReason.DETECTOR_FAILURE),
        (StateName.YARD, StateSource.EVIDENCE, ("bottom_menu_yard",), 1, None),
        (StateName.BATTLE, StateSource.EVIDENCE, (), 0, None),
    ],
)
def test_resolve_unknown_reason_pure_function(
    state: StateName,
    source: StateSource,
    evidence: tuple[str, ...],
    hits: int | None,
    expected: UnknownReason | None,
) -> None:
    """纯函数分支：判定成功但 unknown 时按命中证据数细分；状态明确时为 ``None``。"""
    result = StateResult(
        state_name=state, confidence=0.9, source=source, reason="test", evidence=evidence
    )

    assert resolve_unknown_reason(result, evidence_hits=hits) is expected


def test_resolve_unknown_reason_judge_failure() -> None:
    """判定失败标志优先于结果内容。"""
    result = StateResult(
        state_name=StateName.UNKNOWN,
        confidence=0.0,
        source=StateSource.EVIDENCE,
        reason="no evidence available",
        evidence=(),
    )

    assert resolve_unknown_reason(result, judge_failed=True) is UnknownReason.DETECTOR_FAILURE


def test_diagnostics_reset_clears_counters() -> None:
    """``RuntimeDiagnostics`` 可重置（供多次运行复用）。"""
    diagnostics = RuntimeDiagnostics()
    diagnostics.observe(frame_ok=False, result=None, skip_reason=SkipReason.NOT_FOREGROUND.value)
    assert diagnostics.health.cycles == 1
    assert diagnostics.health.skipped == 1

    diagnostics.reset()

    health = diagnostics.health
    assert health.cycles == 0
    assert health.skipped == 0
    assert health.unknown_reason is None
    assert health.skip_counts == {}


# ---------------------------------------------------------------------------
# 6. metrics schema 兼容
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
#: Phase 5.3 追加的诊断字段
DIAGNOSTIC_METRIC_KEYS: frozenset[str] = frozenset(
    {"capture_status", "foreground_status", "unknown_reason"}
)


def test_metrics_schema_is_backward_compatible(tmp_path: Path) -> None:
    """JSONL 同时包含既有字段（含义不变）与追加的诊断字段。"""
    metrics_path = tmp_path / "metrics.jsonl"
    recorder = RuntimeMetricsRecorder(path=metrics_path)
    binding = MiniBinding(window=make_window())
    source = make_source(binding)
    processor = evidence_processor(ScriptedDetector(name="bottom_menu_yard", script=[False]))
    monitor, _, _ = build_monitor(tmp_path, source=source, processor=processor, metrics=recorder)

    with source:
        monitor.run(max_cycles=1)

    payload = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert LEGACY_METRIC_KEYS <= set(payload)
    assert DIAGNOSTIC_METRIC_KEYS <= set(payload)
    assert set(payload) == LEGACY_METRIC_KEYS | DIAGNOSTIC_METRIC_KEYS
    assert payload["status"] == CycleStatus.OK.value
    assert payload["state"] == StateName.UNKNOWN.value
    assert payload["capture_status"] == CaptureOutcome.OK.value
    assert payload["foreground_status"] == ForegroundOutcome.NOT_CHECKED.value
    assert payload["unknown_reason"] == UnknownReason.NO_EVIDENCE.value
    # 仍然不保存截图 / 用户数据
    assert "image" not in payload
    assert "window" not in payload


def test_cycle_metrics_legacy_construction_still_valid() -> None:
    """旧构造方式（不传诊断字段）仍然可用，且序列化出完整 schema。"""
    metrics = CycleMetrics(
        cycle_id=7,
        timestamp="2025-01-01T00:00:00+08:00",
        started_at=0.0,
        latency_ms=1.5,
        frame_ms=0.5,
        state=StateName.YARD,
        confidence=0.9,
        stable_state=StateName.YARD,
        status=CycleStatus.OK,
    )

    payload = metrics.to_dict()

    assert LEGACY_METRIC_KEYS <= set(payload)
    assert set(payload) == LEGACY_METRIC_KEYS | DIAGNOSTIC_METRIC_KEYS
    assert payload["capture_status"] is None
    assert payload["foreground_status"] is None
    assert payload["unknown_reason"] is None
    assert payload["skip_reason"] is None


def test_metrics_summary_unaffected_by_diagnostics(tmp_path: Path) -> None:
    """诊断字段不影响既有汇总统计（mean / p95 / skipped / errors）。"""
    recorder = RuntimeMetricsRecorder()
    binding = MiniBinding(window=make_window())
    source = make_source(binding)
    monitor, _, _ = build_monitor(tmp_path, source=source, metrics=recorder)

    with source:
        report = monitor.run(max_cycles=4)

    summary = recorder.summary()
    assert summary.cycles == 4
    assert summary.ok == 4
    assert summary.skipped == 0
    assert summary.p95_latency_ms >= summary.mean_latency_ms >= 0.0
    assert report.classified == 4


def test_monitor_health_snapshot_is_json_serializable(tmp_path: Path) -> None:
    """健康快照可序列化（供上层日志 / 诊断使用），且不含用户数据。"""
    binding = MiniBinding(window=make_window())
    source = make_source(binding)
    monitor, _, _ = build_monitor(tmp_path, source=source)

    with source:
        monitor.run(max_cycles=2)

    payload = monitor.health.to_dict()

    assert payload["cycles"] == 2
    assert payload["frames"] == 2
    assert payload["processor_ok"] is True
    assert "window" not in payload
    assert "image" not in payload
    assert json.dumps(payload, ensure_ascii=False)


def test_source_without_diagnostics_capability_still_works(tmp_path: Path) -> None:
    """不提供诊断能力的帧来源（如 ``FakeFrameSource``）仍可运行且不报错。"""
    monitor, recorder, _ = build_monitor(tmp_path, source=FakeFrameSource())

    report = monitor.run(max_cycles=2)

    assert report.cycles == 2
    assert recorder.summary().ok == 2
    assert recorder.records[-1].foreground_status is ForegroundOutcome.NOT_CHECKED
    assert recorder.records[-1].frame_source == "fake"
