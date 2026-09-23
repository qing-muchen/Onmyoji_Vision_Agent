"""Phase 5 Runtime 证据流水线测试。

覆盖：

1. ``EvidenceFrameProcessor`` 输出 ``StateResult``（证据路径）；
2. ``StateMonitor`` 接受 ``processor``（并以证据路径判定）；
3. 旧 ``classifier`` 注入方式仍工作（帧路径，Phase 3.x 兼容）；
4. ``StateStabilizer`` 不受迁移影响（连续窗口 + 置信度门槛）；
5. ``battle_result`` 能进入稳定状态（真实截图，校准后的状态可信度）；
6. ``transition`` 能进入稳定状态（真实条带）；
7. 判定失败（detector / OCR / 模板异常）不会让 Runtime 崩溃，降级为 ``unknown`` 并记录原因；
8. 事件只由**稳定状态变化**触发（单帧抖动不产生事件）。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import EvidenceConfig, RuntimeStateMonitorConfig, load_config
from src.core.exceptions import EvidenceError, OCRProviderUnavailableError
from src.core.paths import resolve_from_project_root
from src.runtime.state_monitor import FrameSource, StateMonitor, StateStabilizer
from src.vision.capture import Frame
from src.vision.evidence import BaseEvidenceDetector, EvidenceRegistry, EvidenceResult
from src.vision.pipeline import EvidenceFrameProcessor
from src.vision.pipeline.frame_processor import build_transition_probe
from src.vision.state import StateClassifier, StateName, StateSource
from tests.monitor_fakes import FakeClassifier, FakeClock, FakeFrameSource, build_monitor

#: 用于脚本化判定的证据名（命中即满足 yard 规则）
SCRIPT_EVIDENCE = "bottom_menu_yard"
#: 真实样本（缺失时跳过依赖它们的用例）
REAL_SAMPLES: dict[str, str] = {
    "battle_result": "dataset/raw/battle_result/battle_result_win_001.png",
    "transition": "dataset/states/transition_bottom/transition_bottom_003.png",
}


def make_frame(seed: int = 0) -> Frame:
    """构造一个极小的测试帧（流水线只需要"有帧"）。"""
    return Frame.create(np.full((8, 8, 3), seed, dtype=np.uint8), source="fake")


def load_image(path: Path) -> np.ndarray:
    """读取图像（兼容非 ASCII 路径）。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


class ScriptedDetector(BaseEvidenceDetector):
    """按脚本返回证据的 detector 替身（``True`` = 命中，``False`` = 未检出）。

    用它可以不依赖任何数据集就驱动"证据 → 状态 → 稳定 → 事件"的完整链路，
    并且完全确定（每一帧返回什么由脚本决定）。

    Args:
        script: 每帧命中与否的序列（用尽后循环）。
        name: 证据名（默认 :data:`SCRIPT_EVIDENCE`）。
        confidence: 命中时的证据置信度。
    """

    def __init__(
        self,
        script: Sequence[bool],
        *,
        name: str = SCRIPT_EVIDENCE,
        confidence: float = 0.9,
        reliability: float = 0.9,
    ) -> None:
        self._script = list(script)
        self._name = name
        self._confidence = confidence
        self._reliability = reliability
        self.calls = 0

    @property
    def name(self) -> str:
        """证据名。"""
        return self._name

    def detect(self, frame: Frame) -> EvidenceResult:
        """按脚本返回命中或未检出（帧内容被忽略）。"""
        hit = self._script[self.calls % len(self._script)]
        self.calls += 1
        if not hit:
            return EvidenceResult.miss(
                self._name, metadata={"reason": "scripted miss"}, reliability=self._reliability
            )
        return EvidenceResult(
            name=self._name,
            detected=True,
            confidence=self._confidence,
            reliability=self._reliability,
        )


class FailingDetector(BaseEvidenceDetector):
    """总是抛异常的 detector 替身（模拟 detector / OCR / 模板失败）。"""

    def __init__(self, error: Exception, *, name: str = SCRIPT_EVIDENCE) -> None:
        self._error = error
        self._name = name

    @property
    def name(self) -> str:
        """证据名。"""
        return self._name

    def detect(self, frame: Frame) -> EvidenceResult:
        """抛出配置的异常。"""
        raise self._error


def scripted_processor(script: Sequence[bool]) -> EvidenceFrameProcessor:
    """用脚本化 detector 构造证据流水线（不依赖数据集）。"""
    registry = EvidenceRegistry([ScriptedDetector(script)])
    classifier = StateClassifier.from_evidence_config(EvidenceConfig())
    return EvidenceFrameProcessor(registry, classifier)


def scripted_monitor(
    script: Sequence[bool],
    tmp_path: Path,
    *,
    max_cycles: int,
    processor: EvidenceFrameProcessor | None = None,
    source: FrameSource | None = None,
) -> StateMonitor:
    """装配"脚本化证据流水线 + StateMonitor"（不真实等待、不写默认历史文件）。"""
    active_processor = processor if processor is not None else scripted_processor(script)
    config = RuntimeStateMonitorConfig(
        interval_ms=10,
        save_history=False,
        history_path=str(tmp_path / "state_history.json"),
    )
    return StateMonitor(
        active_processor,
        source if source is not None else FakeFrameSource(),
        config,
        stabilizer=StateStabilizer(load_config(environ={}).runtime.stabilization),
        clock=FakeClock().clock,
        sleep=lambda _seconds: None,
    )


# ---------------------------------------------------------------------------
# 1. FrameProcessor 输出 StateResult
# ---------------------------------------------------------------------------


def test_frame_processor_returns_state_result() -> None:
    """处理器把一帧判定为 ``StateResult``：命中脚本时得到 yard（source=evidence）。"""
    processor = scripted_processor([True])

    result = processor.process(make_frame())

    assert result.state_name is StateName.YARD
    assert result.source is StateSource.EVIDENCE
    assert result.evidence == (SCRIPT_EVIDENCE,)
    assert result.confidence > 0.0
    assert result.reason


def test_frame_processor_miss_returns_unknown() -> None:
    """脚本未命中时得到 unknown（证据路径，不抛异常）。"""
    result = scripted_processor([False]).process(make_frame())

    assert result.state_name is StateName.UNKNOWN
    assert result.source is StateSource.EVIDENCE
    assert result.confidence == pytest.approx(0.0)
    assert result.evidence == ()


# ---------------------------------------------------------------------------
# 2. StateMonitor 使用 processor
# ---------------------------------------------------------------------------


def test_state_monitor_uses_processor(tmp_path: Path) -> None:
    """监控器接受 processor，并按其判定结果生成稳定状态。"""
    monitor = scripted_monitor([True], tmp_path, max_cycles=3)

    report = monitor.run(max_cycles=3)

    assert monitor.processor is not None
    assert report.classified == 3
    assert report.initial_state is StateName.YARD
    assert monitor.current_state is StateName.YARD
    # 判定来自证据路径（而不是帧路径）
    assert monitor.last_result is not None
    assert monitor.last_result.source is StateSource.EVIDENCE


def test_state_monitor_exposes_inner_classifier(tmp_path: Path) -> None:
    """processor 模式下 ``monitor.classifier`` 仍返回状态分类器（接口兼容）。"""
    monitor = scripted_monitor([True], tmp_path, max_cycles=1)

    assert isinstance(monitor.classifier, StateClassifier)
    assert monitor.classifier.name == "state"


# ---------------------------------------------------------------------------
# 3. 旧 classifier 模式仍工作
# ---------------------------------------------------------------------------


def test_legacy_classifier_mode_still_works(tmp_path: Path) -> None:
    """显式注入 ``classifier``（帧路径）保持 Phase 3.x 行为。"""
    classifier = FakeClassifier([StateName.YARD])
    monitor = StateMonitor(
        classifier,  # type: ignore[arg-type]
        FakeFrameSource(),
        RuntimeStateMonitorConfig(
            interval_ms=10, save_history=False, history_path=str(tmp_path / "h.json")
        ),
        stabilizer=StateStabilizer(load_config(environ={}).runtime.stabilization),
        sleep=lambda _seconds: None,
    )

    report = monitor.run(max_cycles=3)

    assert monitor.processor is None
    assert report.initial_state is StateName.YARD
    assert monitor.current_state is StateName.YARD


def test_processor_keyword_takes_priority(tmp_path: Path) -> None:
    """同时传入二者时 ``processor`` 优先（渐进迁移）。"""
    processor = scripted_processor([True])
    monitor = StateMonitor(
        FakeClassifier([StateName.BATTLE]),  # type: ignore[arg-type]
        FakeFrameSource(),
        RuntimeStateMonitorConfig(
            interval_ms=10, save_history=False, history_path=str(tmp_path / "h.json")
        ),
        processor=processor,
        stabilizer=StateStabilizer(load_config(environ={}).runtime.stabilization),
        sleep=lambda _seconds: None,
    )

    monitor.run(max_cycles=3)

    assert monitor.processor is processor
    assert monitor.current_state is StateName.YARD  # 来自 processor，而非 FakeClassifier


def test_monitor_fakes_build_monitor_still_works(tmp_path: Path) -> None:
    """既有测试辅助（``build_monitor``）保持可用。"""
    monitor, _, _, _ = build_monitor([StateName.YARD], tmp_path=tmp_path)

    report = monitor.run(max_cycles=2)

    assert report.cycles == 2
    assert monitor.current_state is StateName.YARD


# ---------------------------------------------------------------------------
# 4. StateStabilizer 不受影响
# ---------------------------------------------------------------------------


def test_stabilizer_requires_consecutive_frames(tmp_path: Path) -> None:
    """稳定化语义不变：连续 3 帧同状态才确认，前两帧为 pending。"""
    monitor = scripted_monitor([True], tmp_path, max_cycles=3)
    states: list[StateName | None] = []

    for _ in range(3):
        monitor.step(make_frame())
        states.append(monitor.last_stable.state_name if monitor.last_stable else None)

    assert states[0] is None
    assert states[1] is None
    assert states[2] is StateName.YARD


def test_stabilizer_threshold_still_applied(tmp_path: Path) -> None:
    """置信度门槛不变：状态可信度低于门槛的帧不计入稳定统计。

    证据置信度取 ``0.7``：它能通过**证据准入阈值**（``min_confidence=0.6``）→ 判定出
    ``yard``（状态可信度 0.7，yard 规则未配置校准）；但低于稳定化门槛 ``0.85`` →
    该帧**不计入**统计，因此永远无法确认稳定状态。
    """
    registry = EvidenceRegistry([ScriptedDetector([True], confidence=0.7)])
    classifier = StateClassifier.from_evidence_config(EvidenceConfig())
    processor = EvidenceFrameProcessor(registry, classifier)
    monitor = scripted_monitor([True], tmp_path, max_cycles=3, processor=processor)

    monitor.run(max_cycles=3)

    assert monitor.last_result is not None
    assert monitor.last_result.state_name is StateName.YARD  # 判定成立
    assert monitor.current_state is None  # 但 0.7 < confidence_threshold(0.85)，永不确认
    assert monitor.last_stable is not None
    assert monitor.last_stable.counted is False


# ---------------------------------------------------------------------------
# 5/6. 真实数据：battle_result / transition 可以稳定
# ---------------------------------------------------------------------------

requires_datasets = pytest.mark.skipif(
    not resolve_from_project_root("dataset/raw").is_dir(), reason="dataset/raw 不存在"
)


def build_real_monitor(tmp_path: Path, paths: Sequence[Path], *, repeat: int = 3) -> StateMonitor:
    """用真实图片序列 + 真实证据流水线装配监控器。"""
    config = load_config(environ={})
    processor = EvidenceFrameProcessor.from_config(
        config, transition_classifier=build_transition_probe(config)
    )

    class SequenceSource:
        """把图片按顺序重复 ``repeat`` 次后循环（保证连续帧数达到稳定窗口）。"""

        def __init__(self, items: Sequence[Path]) -> None:
            self._items = list(items)
            self._index = 0

        def get_frame(self) -> Frame | None:
            if not self._items:
                return None
            path = self._items[(self._index // repeat) % len(self._items)]
            self._index += 1
            return Frame.create(load_image(path), source=path.name)

    return StateMonitor(
        processor,
        SequenceSource(paths),  # type: ignore[arg-type]
        RuntimeStateMonitorConfig(
            interval_ms=10, save_history=False, history_path=str(tmp_path / "h.json")
        ),
        stabilizer=StateStabilizer(config.runtime.stabilization),
        sleep=lambda _seconds: None,
    )


@requires_datasets
@pytest.mark.parametrize("state", ["battle_result", "transition"])
def test_real_samples_reach_stable_state(tmp_path: Path, state: str) -> None:
    """真实截图（单样本状态）经证据流水线达到稳定状态。"""
    path = resolve_from_project_root(REAL_SAMPLES[state])
    if not path.is_file():
        pytest.skip(f"{path} 不存在")

    monitor = build_real_monitor(tmp_path, [path])
    monitor.run(max_cycles=3)

    assert monitor.current_state is StateName(state)
    assert monitor.last_result is not None
    assert monitor.last_result.source is StateSource.EVIDENCE
    assert monitor.last_result.confidence >= load_config(environ={}).runtime.stabilization.confidence_threshold
    assert monitor.errors == 0


@requires_datasets
def test_real_sequence_produces_events_on_state_change(tmp_path: Path) -> None:
    """真实图片序列：状态切换（连续 3 帧）产生事件，切换前不产生。"""
    paths = [
        resolve_from_project_root(REAL_SAMPLES["battle_result"]),
        resolve_from_project_root(REAL_SAMPLES["transition"]),
    ]
    if not all(path.is_file() for path in paths):
        pytest.skip("样本不存在")

    monitor = build_real_monitor(tmp_path, paths)
    report = monitor.run(max_cycles=6)

    assert report.initial_state is StateName.BATTLE_RESULT
    assert report.final_state is StateName.TRANSITION
    assert len(report.events) == 1
    event = report.events[0]
    assert event.previous_state is StateName.BATTLE_RESULT
    assert event.current_state is StateName.TRANSITION


# ---------------------------------------------------------------------------
# 7. 判定失败不会让 Runtime 崩溃
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        EvidenceError("detector 崩溃"),
        OCRProviderUnavailableError("OCR 不可用"),
    ],
)
def test_failing_detector_does_not_crash(tmp_path: Path, error: Exception) -> None:
    """detector / OCR / 模板异常被捕获：该帧降级为 unknown 并记录原因。"""
    registry = EvidenceRegistry([FailingDetector(error)])
    classifier = StateClassifier.from_evidence_config(EvidenceConfig())
    processor = EvidenceFrameProcessor(registry, classifier)
    monitor = scripted_monitor([True], tmp_path, max_cycles=2, processor=processor)

    report = monitor.run(max_cycles=2)

    assert report.classified == 2  # 循环继续，没有崩溃
    assert monitor.errors == 2
    assert monitor.last_result is not None
    assert monitor.last_result.state_name is StateName.UNKNOWN
    assert monitor.last_result.source is StateSource.FALLBACK
    assert "evidence pipeline failure" in monitor.last_result.reason
    assert str(error) in monitor.last_result.reason


def test_failure_in_one_frame_does_not_break_next_frame(tmp_path: Path) -> None:
    """失败帧之后的帧照常处理（异常不会污染循环状态）。"""
    registry = EvidenceRegistry([ScriptedDetector([False, True])])
    classifier = StateClassifier.from_evidence_config(EvidenceConfig())
    processor = EvidenceFrameProcessor(registry, classifier)
    monitor = scripted_monitor([], tmp_path, max_cycles=2, processor=processor)

    monitor.step(make_frame())  # 第 1 帧：miss → unknown
    monitor.step(make_frame())  # 第 2 帧：hit → yard

    assert monitor.errors == 0
    assert monitor.last_result is not None
    assert monitor.last_result.state_name is StateName.YARD


# ---------------------------------------------------------------------------
# 8. 事件只由稳定状态变化触发
# ---------------------------------------------------------------------------


def test_events_only_on_stable_change(tmp_path: Path) -> None:
    """3 帧命中 → 建立基线（无事件）；切到未命中 3 帧 → 恰好 1 个事件。"""
    monitor = scripted_monitor([True, True, True, False, False, False], tmp_path, max_cycles=6)

    report = monitor.run(max_cycles=6)

    assert report.initial_state is StateName.YARD
    assert report.final_state is StateName.UNKNOWN
    assert len(report.events) == 1
    assert report.events[0].previous_state is StateName.YARD
    assert report.events[0].current_state is StateName.UNKNOWN


def test_jitter_does_not_trigger_events(tmp_path: Path) -> None:
    """抖动（交替命中）在未确认前不产生事件。"""
    monitor = scripted_monitor([True, False, True, False], tmp_path, max_cycles=4)

    report = monitor.run(max_cycles=4)

    assert report.events == ()
    assert monitor.current_state is None
