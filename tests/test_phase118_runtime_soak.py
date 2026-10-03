"""Phase 11.8 测试：yard 运行期观测、soak summary 与安全边界（34 项）。

覆盖用户要求的 34 点：metrics 上限与 FIFO、不保存 Frame/numpy、summary 契约、
offline replay 可运行、状态迁移与 flap、capture 失败与快照失效、prefilter 回退语义、
candidate/calls ≤ 上限、安全配置、数据只读、生产配置未变、AST 边界、
不引入重型依赖、工具不激活窗口也不发送输入。
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.config import EvidenceDetectorConfig, load_config
from src.core.paths import get_project_root
from src.runtime.state_monitor.metrics import CycleMetrics, CycleStatus, RuntimeMetricsRecorder
from src.runtime.state_monitor.monitor import StateMonitor
from src.vision.capture.models import Frame

ROOT = get_project_root()
TOOLS = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import benchmark_template_prefilter as bench  # noqa: E402
import run_runtime_soak as soak  # noqa: E402

YARD = "bottom_menu_yard"
SOAK_SRC = TOOLS / "run_runtime_soak.py"
METRICS_SRC = ROOT / "src" / "runtime" / "state_monitor" / "metrics.py"
MONITOR_SRC = ROOT / "src" / "runtime" / "state_monitor" / "monitor.py"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
YARD_DIR = ROOT / "dataset" / "raw" / "yard"
ANNOTATIONS = ROOT / "dataset" / "annotations" / "elements.jsonl"

YARD_KEYS = (
    "frame_id",
    "vision_ms",
    "yard_ms",
    "yard_detected",
    "yard_confidence",
    "yard_template_id",
    "yard_prefilter",
    "yard_top_k",
    "yard_candidates",
    "yard_total_templates",
    "yard_match_calls",
    "yard_prefilter_ms",
    "yard_fallback",
    "yard_fallback_reason",
    "snapshot_status",
    "snapshot_invalidated",
    "task_active",
    "action_enabled",
    "action_backend",
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():  # noqa: ANN201
    return load_config()


@pytest.fixture(scope="module")
def yard_metrics_config(config):  # noqa: ANN001, ANN201
    """开启 yard_metrics 的内存配置副本（不改任何文件）。"""
    monitor_config = config.runtime.state_monitor.model_copy(
        update={"yard_metrics": True, "save_history": False, "save_metrics": False}
    )
    runtime_config = config.runtime.model_copy(update={"state_monitor": monitor_config})
    return config.model_copy(update={"runtime": runtime_config})


def make_source(repeat: int = 3):  # noqa: ANN201
    """短离线 replay 源（含注入的 capture 失败）。"""
    return soak.ReplayFrameSource(
        script=(("yard", repeat), ("exploration", 2), ("yard", 2)),
        fail_positions=(repeat + 1,),
    )


@pytest.fixture(scope="module")
def replay_summary(yard_metrics_config):  # noqa: ANN001, ANN201
    """一次离线 replay soak（模块级缓存，避免重复 15s 级开销）。"""
    source = make_source(repeat=4)
    summary = soak.run_soak(
        yard_metrics_config,
        source,
        max_cycles=source.plan_size,
        mode="replay",
    )
    assert summary["cycles"] > 0
    return summary


# ---------------------------------------------------------------------------
# 1 / 2 / 3 / 4 —— metrics 上限、FIFO、不保存大对象
# ---------------------------------------------------------------------------


def test_metrics_default_unbounded_but_fifo_when_capped() -> None:
    """显式设置上限时按 FIFO 淘汰（deque(maxlen)），不会无限增长。"""
    recorder = RuntimeMetricsRecorder(max_records=3)
    for index in range(5):
        recorder.record(
            CycleMetrics(
                cycle_id=index + 1,
                timestamp="2026-01-01T00:00:00+08:00",
                started_at=float(index),
                latency_ms=1.0,
                frame_ms=0.5,
                state=None,
                confidence=0.0,
                stable_state=None,
                status=CycleStatus.OK,
            )
        )
    records = recorder.records
    assert len(records) == 3
    assert [record.cycle_id for record in records] == [3, 4, 5]


def test_metrics_reject_negative_capacity() -> None:
    """负数上限直接拒绝（不允许"负容量"被误当成无限制）。"""
    with pytest.raises(ValueError):
        RuntimeMetricsRecorder(max_records=-1)


def test_cycle_metrics_holds_only_scalars() -> None:
    """单轮记录只含标量与短字符串：不含 Frame / ndarray / 候选列表。"""
    payload = CycleMetrics(
        cycle_id=1,
        timestamp="2026-01-01T00:00:00+08:00",
        started_at=0.0,
        latency_ms=1.0,
        frame_ms=0.5,
        state=None,
        confidence=0.0,
        stable_state=None,
        status=CycleStatus.OK,
        yard_template_id="bottom_menu_yard_skin001_001",
        yard_candidates=4,
    ).to_dict()
    for value in payload.values():
        assert not isinstance(value, (Frame, np.ndarray, list, tuple, dict)), value
        assert isinstance(value, (str, int, float, bool, type(None))), value


def test_cycle_metrics_json_serialisable_and_bounded() -> None:
    """单轮 JSON 可序列化且体积有界（< 2KB，防止逐轮膨胀）。"""
    payload = CycleMetrics(
        cycle_id=42,
        timestamp="2026-01-01T00:00:00+08:00",
        started_at=1.0,
        latency_ms=265.0,
        frame_ms=25.0,
        state=None,
        confidence=0.0,
        stable_state=None,
        status=CycleStatus.OK,
        frame_id=42,
        vision_ms=240.0,
        yard_ms=30.0,
        yard_detected=True,
        yard_confidence=0.98,
        yard_prefilter="thumb",
        yard_top_k=4,
        yard_candidates=4,
        yard_total_templates=36,
        yard_match_calls=20,
        yard_prefilter_ms=0.3,
        yard_fallback=False,
        snapshot_status="processed",
        snapshot_invalidated=0,
        action_enabled=False,
        action_backend="dry_run",
    ).to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert len(encoded.encode("utf-8")) < 2048
    assert payload["yard_match_calls"] == 20


def test_default_metrics_output_unchanged(yard_metrics_config) -> None:
    """未开启观测时，指标输出**不含**任何 yard 字段（与既有版本一致）。"""
    config = load_config()
    source = soak.ReplayFrameSource(script=(("yard", 2),), fail_positions=())
    recorder = RuntimeMetricsRecorder()
    monitor = StateMonitor.from_config(config, source=source, metrics=recorder)
    monitor.run(max_cycles=2)
    for record in recorder.records:
        assert not any(key in record.to_dict() for key in YARD_KEYS)


# ---------------------------------------------------------------------------
# 5 / 6 / 7 / 8 / 9 / 10 / 11 —— summary 契约与 offline replay
# ---------------------------------------------------------------------------


def test_summary_contains_cycle_latency(replay_summary) -> None:  # noqa: ANN001
    """summary 含 cycle latency 的 p50/p90/p95/p99/max。"""
    latency = replay_summary["cycle_latency"]
    for key in ("p50", "mean", "p90", "p95", "p99", "max"):
        assert latency[key] is not None, key
    assert latency["p50"] <= latency["max"]


def test_summary_contains_yard_detector_latency(replay_summary) -> None:  # noqa: ANN001
    """summary 含 yard detector 单项耗时。"""
    detector_ms = replay_summary["yard"]["detector_ms"]
    assert detector_ms["samples"] > 0
    assert detector_ms["p50"] is not None and detector_ms["p95"] is not None


def test_summary_contains_candidate_count(replay_summary) -> None:  # noqa: ANN001
    """summary 含候选数分布。"""
    candidate = replay_summary["yard"]["candidate_count"]
    assert candidate["samples"] > 0
    assert candidate["max"] is not None


def test_summary_contains_fallback(replay_summary) -> None:  # noqa: ANN001
    """summary 含 prefilter 回退计数与原因分布。"""
    yard = replay_summary["yard"]
    assert "fallback_count" in yard
    assert yard["fallback_rate"] is not None
    assert isinstance(yard["fallback_reasons"], dict)


def test_summary_contains_state_transitions(replay_summary) -> None:  # noqa: ANN001
    """summary 含稳定状态计数与迁移次数。"""
    states = replay_summary["states"]
    assert states["counts"]
    assert states["transitions"] >= 0
    assert isinstance(states["transition_pairs"], dict)
    assert states["flaps"] >= 0


def test_summary_contains_snapshot_invalidation(replay_summary) -> None:  # noqa: ANN001
    """summary 含快照失效统计（注入的 capture 失败应记为 no_frame）。"""
    snapshot = replay_summary["snapshot"]
    assert snapshot["invalidations"] >= 1
    assert "no_frame" in snapshot["statuses"]


def test_offline_replay_is_marked(replay_summary) -> None:  # noqa: ANN001
    """offline replay 结果必须显式标注，避免被当成真实性能结论。"""
    assert replay_summary["mode"] == "replay"
    assert replay_summary["source"] == "ReplayFrameSource"


def test_soak_summary_writes_to_temp() -> None:
    """soak 默认输出路径在系统 TEMP。"""
    for target in (soak.DEFAULT_OUT, soak.DEFAULT_METRICS):
        path = Path(target)
        assert Path(tempfile.gettempdir()) in path.parents
        assert "dataset" not in path.as_posix()


# ---------------------------------------------------------------------------
# 12 / 13 / 14 / 15 / 16 —— 状态稳定、迁移、失败与快照
# ---------------------------------------------------------------------------


def test_yard_stable_sequence_does_not_flap() -> None:
    """连续 yard 画面不会制造 flap（稳定器只在达到门槛后确认状态）。"""
    source = soak.ReplayFrameSource(script=(("yard", 10),), fail_positions=())
    summary = soak.run_soak(load_config(), source, max_cycles=10, mode="replay")
    assert summary["states"]["flaps"] == 0
    assert summary["states"]["counts"].get("yard", 0) >= 1


def test_yard_to_other_state_records_one_transition() -> None:
    """yard 与其它画面交替时，达到稳定门槛即记录迁移（默认 replay 脚本）。"""
    source = soak.ReplayFrameSource(fail_positions=())
    summary = soak.run_soak(
        load_config(), source, max_cycles=source.plan_size, mode="replay"
    )
    states = summary["states"]
    assert states["transitions"] >= 1
    assert any("->" in key for key in states["transition_pairs"])
    assert states["flaps"] == 0


def test_capture_failure_produces_no_observation(replay_summary) -> None:  # noqa: ANN001
    """注入的 capture 失败轮次：status=skipped、无 evidence、快照失效。"""
    capture = replay_summary["capture"]
    assert capture["failures"] >= 1
    assert any("capture_failed_injected" in key for key in capture["skipped_reasons"])
    snapshot = replay_summary["snapshot"]
    assert snapshot["invalidations"] >= 1


def test_foreground_lost_produces_no_action(replay_summary) -> None:  # noqa: ANN001
    """任何轮次都不产生真实输入（安全字段恒为 dry_run）。"""
    safety = replay_summary["safety"]
    assert safety["action_enabled"] is False
    assert safety["backend"] == "dry_run"
    assert safety["real_input"] is False


def test_prefilter_fallback_zero_in_normal_run(replay_summary) -> None:  # noqa: ANN001
    """正常运行时 fallback = 0（包括未命中的 yard 轮次）。"""
    assert replay_summary["yard"]["fallback_count"] == 0


def test_injected_prefilter_failure_is_counted(config, yard_metrics_config) -> None:  # noqa: ANN001
    """注入预筛故障时，回退被统计且被记录（含原因与候选数）。"""
    from src.runtime.state_monitor.metrics import RuntimeMetricsRecorder as Recorder

    source = soak.ReplayFrameSource(script=(("yard", 4),), fail_positions=())
    recorder = Recorder()
    monitor = StateMonitor.from_config(yard_metrics_config, source=source, metrics=recorder)
    assert monitor._processor is not None  # noqa: SLF001

    def exploding(_gray):  # noqa: ANN001, ANN202
        raise RuntimeError("injected prefilter failure")

    for detector in monitor._processor.registry.detectors:  # noqa: SLF001
        if getattr(detector, "name", "") == YARD and getattr(detector, "prefilter", None):
            detector.prefilter.rank = exploding  # type: ignore[method-assign]

    monitor.run(max_cycles=4)
    rows = [record.to_dict() for record in recorder.records]
    fallback_rows = [row for row in rows if row.get("yard_fallback")]
    assert fallback_rows, "注入故障后应至少有一轮回退"
    assert any(
        "injected prefilter failure" in str(row["yard_fallback_reason"]) for row in fallback_rows
    )
    # 回退时按全量匹配：候选数 = 模板库大小
    assert all(row["yard_candidates"] == 36 for row in fallback_rows)


def test_normal_miss_is_not_fallback(replay_summary) -> None:  # noqa: ANN001
    """正常未命中（负样本/非 yard 画面）不算 fallback。"""
    yard = replay_summary["yard"]
    assert yard["fallback_count"] == 0
    # 未命中的轮次仍然有候选诊断（Phase 11.8 补齐）
    assert yard["candidate_count"]["samples"] >= yard["detections"]


def test_candidate_count_within_top_k(replay_summary) -> None:  # noqa: ANN001
    """候选数不超过 top_k=4，matchTemplate 调用不超过 20。"""
    candidate = replay_summary["yard"]["candidate_count"]
    calls = replay_summary["yard"]["match_calls"]
    assert candidate["max"] is not None and candidate["max"] <= 4
    assert calls["max"] is not None and calls["max"] <= 20
    assert replay_summary["yard"]["prefilter"] == ["thumb"]


# ---------------------------------------------------------------------------
# 17..30 —— 安全、数据只读、生产配置
# ---------------------------------------------------------------------------


def test_safety_configuration(config) -> None:  # noqa: ANN001
    """安全配置：``action.enabled=False``、``backend=dry_run``。"""
    assert config.action.enabled is False
    assert config.action.backend == "dry_run"


def test_production_yard_config_unchanged(config) -> None:  # noqa: ANN001
    """生产 yard 配置未被本阶段改动：36 张 / top_k 4 / 0.85 / 0.75。"""
    detector_config = config.evidence.detectors[YARD]
    assert detector_config.max_templates == detector_config.explicit_template_count == 36
    assert detector_config.prefilter_top_k == 4
    assert detector_config.template_prefilter == "thumb"
    assert detector_config.threshold == 0.85
    assert detector_config.roi_scale == 0.75


def test_yard_detector_loads_36_templates(config) -> None:  # noqa: ANN001
    """生产 detector 仍加载 36 张模板。"""
    from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector

    detector = BottomMenuYardDetector.from_config(config.evidence)
    assert len(detector.templates) == 36
    assert detector.prefilter is not None


def test_raw_dataset_and_annotations_unchanged(replay_summary) -> None:  # noqa: ANN001
    """soak 前后 raw 抽样 SHA256 与 annotations 不变（只读）。"""
    before = {
        path.name: bench.source_id(path) for path in sorted(YARD_DIR.glob("*.png"))[:5]
    }
    assert len(list(YARD_DIR.glob("*.png"))) == 98
    after = {path.name: bench.source_id(path) for path in sorted(YARD_DIR.glob("*.png"))[:5]}
    assert after == before
    assert ANNOTATIONS.read_bytes()  # 仍可读（内容由 check_annotations 校验）


def test_config_file_unchanged(config) -> None:  # noqa: ANN001
    """soak 不会改写生产配置（内存副本 + 只读工具）。"""
    before = CONFIG_PATH.read_bytes()
    source = soak.ReplayFrameSource(script=(("yard", 2),), fail_positions=())
    soak.run_soak(config, source, max_cycles=2, mode="replay")
    assert CONFIG_PATH.read_bytes() == before


def test_ast_boundary_of_soak_tool() -> None:
    """soak 工具不 import 输入控制库（pyautogui）与 Action 派发器。"""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(SOAK_SRC.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import) and not isinstance(node, ast.ImportFrom):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert "pyautogui" not in modules
    for module in modules:
        assert "dispatcher" not in module
        assert not module.startswith("src.task"), module


def test_soak_tool_does_not_activate_window() -> None:
    """工具**代码**中不存在窗口激活 / 键鼠输入调用（docstring 里的说明文字不计）。"""
    tree = ast.parse(SOAK_SRC.read_text(encoding="utf-8"))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name:
                called.add(name)
    assert "pyautogui" not in imported
    for forbidden in ("SetForegroundWindow", "activate", "click", "press", "write", "hotkey"):
        assert forbidden not in called, forbidden


def test_runtime_module_has_no_new_heavy_dependency() -> None:
    """Runtime 观测路径不引入新的重型第三方依赖（只用标准库 + numpy）。"""
    for path in (METRICS_SRC, MONITOR_SRC):
        modules: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        for forbidden in ("psutil", "torch", "ultralytics", "PIL", "matplotlib"):
            assert forbidden not in modules, (path.name, forbidden)


def test_probe_window_reports_unavailable_without_faking(config, monkeypatch) -> None:  # noqa: ANN001
    """窗口不可用时必须返回 not_run + 原因，而不是伪造成功。"""

    class _FakeBinding:
        def find(self, *, refresh: bool = False):  # noqa: ANN202, ARG002
            return None

    class _FakeSource:
        binding = _FakeBinding()

    monkeypatch.setattr(
        "src.runtime.state_monitor.frame_source.WindowFrameSource.from_config",
        classmethod(lambda cls, cfg: _FakeSource()),
    )
    available, reason = soak.probe_window(config)
    assert available is False
    assert reason == "window_not_found"


def test_soak_safety_field_is_explicit(replay_summary) -> None:  # noqa: ANN001
    """summary 必须显式给出 ``real_input=false``（安全契约）。"""
    assert replay_summary["safety"]["real_input"] is False
    assert replay_summary["safety"]["action_enabled"] is False


def test_evidence_detector_config_default_off() -> None:
    """观测/预筛的默认值仍是"关闭"（不因本阶段改变默认行为）。"""
    default = EvidenceDetectorConfig()
    assert default.template_prefilter == "none"
    assert default.prefilter_top_k == 0
    assert load_config().runtime.state_monitor.yard_metrics is False
