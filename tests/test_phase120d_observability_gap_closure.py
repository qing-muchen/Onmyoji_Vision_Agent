"""Phase 12.0-D 测试：观测缺口补齐（只补观测，不改判定逻辑）。

关闭 Phase 12.0-C 记录的 4 个 ``observability_gap``：

* **①** ``yard_fields_leak_into_skipped_cycles`` —— 跳过轮（失焦 / 最小化 / 采集失败）
  不得携带上一轮的 ``yard_detected`` / ``yard_confidence`` / ``yard_template_id`` /
  ``yard_bbox_oob``；统计分母改为"真正产生过 yard 观测的轮次"。
* **②** ``match_calls_is_budget_not_actual`` —— 保留 ``yard_match_calls``（预算）历史语义，
  新增 ``yard_match_calls_actual``（**真正执行** ``cv2.matchTemplate`` 的次数），
  并可派生 ``skipped_match_calls = budget - actual``。
* **③** ``unknown_reason`` **保持状态层语义**（不扩展枚举、不改取值集合）；
  **具体原因统计一律以 ``skip_reason`` 为唯一权威**。
* **④** ``rss_sampler_keeps_no_timeseries`` —— RSS 采样保留时序，可算前/中/后 10 分钟
  median 与 MB/hour 斜率与结论。

安全/纪律：不修改 threshold / roi_scale / 模板集合 / prefilter / top_k / matcher 判定 /
``FIT_TOLERANCE`` / Stabilizer / Task 策略；不启用真实输入。
"""

from __future__ import annotations

import ast
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_runtime_soak as soak  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.runtime.state_monitor.metrics import (  # noqa: E402
    CycleMetrics,
    CycleStatus,
    RuntimeMetricsRecorder,
)
from src.runtime.state_monitor.monitor import StateMonitor  # noqa: E402
from src.runtime.state_monitor.runtime_diagnostics import (  # noqa: E402
    UnknownReason,
    resolve_unknown_reason,
)
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.detector.template.matcher import FIT_TOLERANCE_PX, TemplateMatcher  # noqa: E402
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector  # noqa: E402

MONITOR_SRC = ROOT / "src" / "runtime" / "state_monitor" / "monitor.py"
MATCHER_SRC = ROOT / "src" / "vision" / "detector" / "template" / "matcher.py"
SOAK_SRC = ROOT / "tools" / "run_runtime_soak.py"
YARD = "bottom_menu_yard"
YARD_DIR = ROOT / "dataset" / "raw" / "yard"
REPLAY_CYCLES = 24


def metric(cycle_id: int = 1, **kwargs: object) -> CycleMetrics:
    """构造 ``CycleMetrics``（补齐必填字段；观测字段按需覆盖）。"""
    base: dict[str, object] = {
        "cycle_id": cycle_id,
        "timestamp": "2026-10-05T00:00:00+08:00",
        "started_at": float(cycle_id),
        "latency_ms": 1.0,
        "frame_ms": 0.5,
        "state": None,
        "confidence": 0.0,
        "stable_state": None,
        "status": CycleStatus.OK,
    }
    base.update(kwargs)
    return CycleMetrics(**base)  # type: ignore[arg-type]


def yard_detector():  # noqa: ANN201
    """生产 yard detector（真实配置与模板集合）。"""
    return BottomMenuYardDetector.from_config(load_config().evidence)


def _replay_config():  # noqa: ANN202
    """带 ``yard_metrics=True`` 运行期 override 的配置（与 soak 同法，不改生产默认）。"""
    config = load_config()
    monitor_config = config.runtime.state_monitor.model_copy(
        update={"yard_metrics": True, "save_metrics": False, "save_history": False}
    )
    runtime_config = config.runtime.model_copy(update={"state_monitor": monitor_config})
    return config.model_copy(update={"runtime": runtime_config})


def _replay_run() -> tuple[list[dict], dict]:  # noqa: ANN401
    """跑一次含**注入采集失败**的 replay soak，返回 ``(逐轮 rows, summary)``。"""
    recorder = RuntimeMetricsRecorder(path=None, max_records=1000)
    source = soak.ReplayFrameSource(loop=True)  # 默认 fail_positions 会注入失败
    monitor = StateMonitor.from_config(
        _replay_config(), source=source, metrics=recorder, vision_bridge=None
    )
    monitor.run(max_cycles=REPLAY_CYCLES)
    rows = [record.to_dict() for record in recorder.records]
    summary = soak.build_summary(
        monitor,
        recorder,
        mode="replay",
        duration_s=1.0,
        cycles=len(rows),
        memory=soak.measure_memory(
            {"rss_start_bytes": None, "rss_peak_bytes": None, "rss_end_bytes": None,
             "rss_samples": 0, "rss_timeseries": None}
        ),
    )
    return rows, summary


# ---------------------------------------------------------------------------
# 缺口①：跳过轮不携带残留 yard 观测
# ---------------------------------------------------------------------------


def test_skipped_cycle_has_no_yard_evidence_fields() -> None:
    """[1] 端到端：注入采集失败 ⇒ 该轮 metrics **不含**任何 yard 证据字段。"""
    rows, _summary = _replay_run()
    skipped = [row for row in rows if row.get("status") == "skipped"]
    assert skipped, "replay 注入失败后必须出现 skipped 轮"
    for row in skipped:
        assert row.get("yard_detected") is None, row
        assert row.get("yard_confidence") is None, row
        assert row.get("yard_template_id") is None, row
        assert row.get("yard_bbox_oob") is None, row
        assert row.get("vision_ms") is None, row
        # 与"本轮是否产出"无关的字段照常写入（配置类 / 累计类）
        assert row.get("frame_id") is not None
        assert row.get("action_enabled") is False
        assert row.get("action_backend") == "dry_run"


def test_ok_cycle_still_has_yard_evidence_fields() -> None:
    """[2] 正常轮次照常有 yard 证据字段（确认修复没有把观测整体关掉）。"""
    rows, _summary = _replay_run()
    ok = [row for row in rows if row.get("status") == "ok"]
    assert ok, "replay 必须有成功轮"
    assert any(row.get("yard_detected") is not None for row in ok)
    assert any(row.get("vision_ms") is not None for row in ok)


def test_yard_stats_excludes_unobserved_from_denominator() -> None:
    """[3] 识别率分母只含"真正产生过 yard 观测"的轮次。"""
    stats = soak._yard_stats(  # noqa: SLF001
        [
            metric(1, yard_detected=True, yard_confidence=0.9),
            metric(2, yard_detected=True, yard_confidence=0.9),
            metric(3, status=CycleStatus.SKIPPED),  # 跳过：yard_detected 保持 None
        ]
    )
    assert stats["cycles"] == 3
    assert stats["yard_observed"] == 2
    assert stats["yard_unobserved"] == 1
    assert stats["detections"] == 2
    assert stats["detected_ratio"] == 1.0
    # 兼容视图保留（会高估），仅作历史对照
    assert stats["detected_ratio_all_cycles"] == pytest.approx(2 / 3, abs=1e-6)


def test_yard_stats_ratio_is_none_when_nothing_observed() -> None:
    """[4] 全部未采集 ⇒ 识别率为 ``None``（既不是 0，也不是 1）。"""
    stats = soak._yard_stats([metric(1, status=CycleStatus.SKIPPED)])  # noqa: SLF001
    assert stats["yard_observed"] == 0
    assert stats["detected_ratio"] is None


def test_replay_summary_ratio_matches_observed_only() -> None:
    """[5] 端到端：summary 的 ``detected_ratio`` 只按已观测轮次计算，且与逐轮一致。"""
    rows, summary = _replay_run()
    observed = [row for row in rows if row.get("yard_detected") is not None]
    detected = [row for row in observed if row.get("yard_detected")]
    yard = summary["yard"]
    assert yard["yard_observed"] == len(observed)
    assert yard["detections"] == len(detected)
    assert yard["detected_ratio"] == pytest.approx(len(detected) / len(observed), abs=1e-6)
    assert yard["yard_observed"] < yard["cycles"], "本场景含跳过轮，分母必须小于总轮次"


def test_monitor_signature_requires_produced_flag() -> None:
    """[6] ``_yard_observation`` 必须以关键字 ``produced`` 调用（防止回退到泄漏写法）。"""
    source = MONITOR_SRC.read_text(encoding="utf-8")
    assert "def _yard_observation(self, *, produced: bool)" in source
    assert "if not produced:" in source
    assert "**self._yard_observation(produced=" in source


def test_skipped_cycle_still_records_skip_reason() -> None:
    """[7] 跳过轮必须留下**精确** ``skip_reason``（供缺口③的权威统计使用）。"""
    rows, summary = _replay_run()
    skipped = [row for row in rows if row.get("status") == "skipped"]
    assert skipped
    assert all(row.get("skip_reason") for row in skipped)
    assert summary["capture"]["skip_reasons"] != {"(none)": len(rows)}


# ---------------------------------------------------------------------------
# 缺口②：budget vs actual
# ---------------------------------------------------------------------------


def test_metrics_has_match_calls_actual_field() -> None:
    """[8] ``CycleMetrics`` 新增 ``yard_match_calls_actual``；旧字段语义与顺序不变。"""
    fields = {f.name: f for f in dataclasses.fields(CycleMetrics)}
    assert fields["yard_match_calls_actual"].default is None
    assert fields["yard_match_calls"].default is None
    payload = metric(1, yard_match_calls=20, yard_match_calls_actual=20).to_dict()
    assert payload["yard_match_calls"] == 20
    assert payload["yard_match_calls_actual"] == 20
    keys = list(payload)
    assert keys.index("yard_match_calls_actual") == keys.index("yard_match_calls") + 1


def test_matcher_does_not_count_overflow_skipped_levels() -> None:
    """[9] 模板远大于匹配图（overflow > FIT_TOLERANCE_PX）⇒ 整档跳过 ⇒ 不计数。"""
    matcher = TemplateMatcher(load_config().evidence.detectors[YARD].matcher_config())
    frame = np.zeros((60, 60), dtype=np.uint8)
    huge = dataclasses.replace(
        yard_detector().templates[0], image=np.zeros((200, 200, 3), dtype=np.uint8)
    )
    before = matcher.match_calls
    matcher.match(frame, huge)
    assert matcher.match_calls == before, "溢出超容差的档位不得计入 actual"


def test_matcher_counts_bounded_fit_calls() -> None:
    """[10] bounded fit 之后实际执行了匹配 ⇒ 计入 actual。

    精确构造：匹配图 240×120、模板 401×201 ⇒ 0.6 档缩放后为 241×121 ⇒ 溢出 **1 px**
    （≤ ``FIT_TOLERANCE_PX``）⇒ 走 bounded fit 并真的执行 ``cv2.matchTemplate``。
    """
    matcher = TemplateMatcher(load_config().evidence.detectors[YARD].matcher_config())
    frame = np.zeros((120, 240), dtype=np.uint8)
    near = dataclasses.replace(
        yard_detector().templates[0], image=np.zeros((201, 401, 3), dtype=np.uint8)
    )
    before_calls = matcher.match_calls
    before_fit = matcher.fit_count
    matcher.match(frame, near)
    assert matcher.fit_count == before_fit + 1, "应触发一次 bounded fit"
    assert matcher.match_calls > before_calls, "fit 之后真的执行了匹配 ⇒ 计入 actual"


def test_matcher_actual_is_lower_than_budget_at_production_workpoint() -> None:
    """[11] 生产工作点：actual(12) < budget(20) —— 每候选有 2 档因 overflow 被跳过。

    这正是缺口②的价值：``yard_match_calls`` 只反映**预算**（候选 × 档位），
    并不代表真的执行了那么多次 ``cv2.matchTemplate``。
    """
    import cv2

    detector = yard_detector()
    image = cv2.imread(str(YARD_DIR / "yard_skin001_001.png"), cv2.IMREAD_COLOR)
    detector.detect(Frame.create(image, source="probe", timestamp=1.0))
    metadata = detector._prefilter_diagnostics()  # noqa: SLF001
    candidates = metadata["prefilter_candidate_count"]
    scales = len(detector.matcher_config.effective_scales)
    actual = metadata["prefilter_match_calls_actual"]
    assert candidates == 4
    assert scales == 5
    assert actual == 12, "0.825 / 0.9 两档在 1278 宽匹配图上溢出 ⇒ 每候选只跑 3 档"
    assert actual < candidates * scales


def test_soak_derives_skipped_match_calls() -> None:
    """[12] soak 同时给出 budget / actual / 派生 skipped。"""
    stats = soak._yard_stats(  # noqa: SLF001
        [
            metric(1, yard_detected=True, yard_match_calls=20, yard_match_calls_actual=20),
            metric(2, yard_detected=False, yard_match_calls=20, yard_match_calls_actual=0),
        ]
    )
    assert stats["match_calls"]["distribution"] == {20: 2}
    assert stats["match_calls"]["note"].startswith("**预算**")
    assert stats["match_calls_actual"]["distribution"] == {0: 1, 20: 1}
    assert stats["skipped_match_calls"]["distribution"] == {0: 1, 20: 1}
    assert stats["skipped_match_calls"]["total"] == 20
    assert "预算" in stats["skipped_match_calls"]["note"]


def test_replay_summary_reports_both_call_kinds() -> None:
    """[13] 端到端：summary 同时含 budget 与 actual，且 actual ≤ budget。"""
    _rows, summary = _replay_run()
    yard = summary["yard"]
    budget = yard["match_calls"]
    actual = yard["match_calls_actual"]
    assert budget["samples"] == actual["samples"] > 0
    assert (budget["max"] or 0) >= (actual["max"] or 0)
    assert yard["skipped_match_calls"]["total"] is not None


# ---------------------------------------------------------------------------
# 缺口③：skip_reason 是唯一权威原因
# ---------------------------------------------------------------------------


def test_unknown_reason_enum_values_unchanged() -> None:
    """[14] ``UnknownReason`` 取值集合**未扩展**（状态层语义与序列化兼容性不变）。"""
    assert {item.value for item in UnknownReason} == {
        "no_evidence",
        "unsupported_scene",
        "detector_failure",
        "not_foreground",
        "skipped",
    }


def test_resolve_unknown_reason_behaviour_unchanged() -> None:
    """[15] ``resolve_unknown_reason`` 行为不变：只特判 not_foreground，其余归 skipped。"""
    assert (
        resolve_unknown_reason(None, skip_reason="not_foreground")
        is UnknownReason.NOT_FOREGROUND
    )
    assert resolve_unknown_reason(None, skip_reason="window_minimized") is UnknownReason.SKIPPED
    assert resolve_unknown_reason(None, skip_reason="window_not_found") is UnknownReason.SKIPPED


def test_skip_reason_distinguishes_focus_loss_from_minimize() -> None:
    """[16] 失焦 vs 最小化必须在 ``skip_reason`` 上可区分（不再被 unknown_reason 抹平）。"""
    rows = [
        metric(1, yard_detected=True),
        metric(2, status=CycleStatus.SKIPPED, skip_reason="not_foreground",
               unknown_reason=UnknownReason.NOT_FOREGROUND),
        metric(3, status=CycleStatus.SKIPPED, skip_reason="window_minimized",
               unknown_reason=UnknownReason.SKIPPED),
    ]
    payloads = [row.to_dict() for row in rows]
    counted: dict[str, int] = {}
    for row in payloads:
        key = row.get("skip_reason") or "(none)"
        counted[key] = counted.get(key, 0) + 1
    assert counted == {"(none)": 1, "not_foreground": 1, "window_minimized": 1}
    # 状态层未变：最小化在 unknown_reason 上仍是 skipped（未扩展枚举）
    assert payloads[2]["unknown_reason"] == "skipped"


def test_summary_separates_skip_reasons_and_diagnosis() -> None:
    """[17] summary：``capture.skip_reasons``（权威）+ ``diagnosis.unknown_reasons``（状态层）。"""
    _rows, summary = _replay_run()
    assert "skip_reasons" in summary["capture"]
    assert "unknown_reasons" not in summary["capture"], "具体原因不得再用 unknown_reason 统计"
    assert "unknown_reasons" in summary["diagnosis"]
    assert "唯一权威" in summary["diagnosis"]["note"]
    source = SOAK_SRC.read_text(encoding="utf-8")
    assert "capture.skip_reasons" in source


# ---------------------------------------------------------------------------
# 缺口④：RSS 时序 / 分段 median / 斜率
# ---------------------------------------------------------------------------


def test_rss_sampler_keeps_timeseries() -> None:
    """[18] 采样器保留 ``(t, rss)`` 时序，两序列等长且首点为 0。"""
    pytest.importorskip("psutil")
    sampler = soak._RssSampler(interval_s=0.05)  # noqa: SLF001
    sampler.start()
    result = sampler.stop()
    series = result["rss_timeseries"]
    assert isinstance(series, dict)
    assert len(series["t_s"]) == len(series["rss_bytes"]) == result["rss_samples"]
    assert series["t_s"][0] == 0.0
    assert all(value > 0 for value in series["rss_bytes"])


def test_rss_sampler_without_psutil_has_no_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """[19] 无 psutil ⇒ 不启动线程、无时序（不伪造）。"""
    monkeypatch.setattr(soak, "_rss_bytes", lambda: None)
    sampler = soak._RssSampler()  # noqa: SLF001
    sampler.start()
    result = sampler.stop()
    assert result["rss_samples"] == 0
    assert result["rss_timeseries"] is None


def test_rss_window_stats_reports_three_windows() -> None:
    """[20] 满 1 小时 ⇒ 给出前/中/后 10 分钟 median，且随增长单调上升。"""
    times = [float(i) for i in range(0, 3601, 10)]
    values = [int(200 * soak.MB) + i * 1000 for i in range(len(times))]
    windows = soak._rss_window_stats(times, values)  # noqa: SLF001
    assert windows["available"] is True
    for key in ("first_10min", "middle_10min", "last_10min"):
        assert windows[key]["median_mb"] is not None
        assert windows[key]["samples"] > 0
    assert (
        windows["first_10min"]["median_mb"]
        < windows["middle_10min"]["median_mb"]
        < windows["last_10min"]["median_mb"]
    )


def test_rss_window_stats_flags_short_run() -> None:
    """[21] 时长不足 ⇒ 如实标注，不臆造窗口。"""
    windows = soak._rss_window_stats([0.0, 60.0], [100, 101])  # noqa: SLF001
    assert windows["available"] is False
    assert "无法切分" in windows["note"]


def test_rss_trend_stabilized_after_warmup() -> None:
    """[22] warmup 上升后平台 ⇒ ``stabilized_after_warmup``（**不**因 peak>start 报泄漏）。"""
    times = [float(i) for i in range(0, 3601, 10)]
    values = [
        int((200 + t / 60.0) * soak.MB) if t <= soak.RSS_WARMUP_S else int(205 * soak.MB)
        for t in times
    ]
    trend = soak._rss_trend(times, values)  # noqa: SLF001
    assert trend["verdict"] == "stabilized_after_warmup"
    assert trend["tail_slope_mb_per_hour"] is not None
    assert "不" in trend["note_disclaimer"]


def test_rss_trend_persistent_growth() -> None:
    """[23] warmup 后仍持续上升 ⇒ ``possible_memory_growth``（只描述、不自动改缓存/GC）。"""
    times = [float(i) for i in range(0, 3601, 10)]
    values = [int((200 + t / 60.0) * soak.MB) for t in times]  # 60 MB/hour
    trend = soak._rss_trend(times, values)  # noqa: SLF001
    assert trend["verdict"] == "possible_memory_growth"
    assert trend["tail_slope_mb_per_hour"] > soak.RSS_SLOPE_FLAT_MB_PER_H
    assert "未修改缓存/GC 策略" in trend["note_disclaimer"]


def test_rss_trend_insufficient_without_series() -> None:
    """[24] 无时序 ⇒ ``insufficient_data``（不猜）。"""
    assert soak._rss_trend([], [])["verdict"] == "insufficient_data"  # noqa: SLF001
    assert soak._rss_trend([0.0, 1.0], [100, 101])["verdict"] == "insufficient_data"  # noqa: SLF001


def test_measure_memory_exposes_windows_and_trend() -> None:
    """[25] ``measure_memory`` 输出 ``rss_windows`` / ``rss_trend``，且可 JSON 序列化。"""
    memory = soak.measure_memory(
        {"rss_start_bytes": None, "rss_peak_bytes": None, "rss_end_bytes": None,
         "rss_samples": 0, "rss_timeseries": None}
    )
    assert memory["rss_start"] is None
    assert memory["rss_trend"]["verdict"] == "insufficient_data"
    assert "not_available" in memory["rss_note"]
    times = [float(i) for i in range(0, 1201, 10)]
    values = [int(200 * soak.MB)] * len(times)
    memory2 = soak.measure_memory(
        {"rss_start_bytes": values[0], "rss_peak_bytes": max(values),
         "rss_end_bytes": values[-1], "rss_samples": len(values),
         "rss_timeseries": {"t_s": times, "rss_bytes": values}}
    )
    assert memory2["rss_windows"] is not None
    assert memory2["rss_trend"]["verdict"] in {
        "stabilized_after_warmup", "possible_memory_growth", "declining_after_warmup",
    }
    json.dumps(memory2)


# ---------------------------------------------------------------------------
# 向后兼容与纪律
# ---------------------------------------------------------------------------


def test_no_yard_keys_when_metrics_disabled() -> None:
    """[26] 未采集观测时仍**不**输出任何 yard / vision 键（无新增序列化开销）。"""
    payload = metric().to_dict()
    assert not any(key.startswith("yard_") for key in payload)
    assert "vision_ms" not in payload
    assert "rss_windows" not in payload


def test_yard_metrics_default_still_false() -> None:
    """[27] 生产默认 ``yard_metrics`` 仍为 ``False``。"""
    assert load_config().runtime.state_monitor.yard_metrics is False


def test_decision_parameters_untouched() -> None:
    """[28] 判定参数与安全开关未被本轮观测改动影响。"""
    config = load_config()
    yard = config.evidence.detectors[YARD]
    assert yard.threshold == 0.85
    assert yard.roi_scale == 0.75
    assert yard.template_prefilter == "thumb"
    assert yard.prefilter_top_k == 4
    assert len(yard.template_include) == 36
    assert FIT_TOLERANCE_PX == 2
    assert config.action.enabled is False
    assert config.action.backend == "dry_run"


def test_matcher_counting_is_pure_increment() -> None:
    """[29] 计数只有一处 ``+= 1``，且不参与任何控制流（判定路径不变）。"""
    tree = ast.parse(MATCHER_SRC.read_text(encoding="utf-8"))
    increments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Attribute)
        and node.target.attr == "_match_calls"
    ]
    assert len(increments) == 1
    source = MATCHER_SRC.read_text(encoding="utf-8")
    assert "if self._match_calls" not in source


def test_monitor_still_reads_metadata_only() -> None:
    """[30] monitor 仍只读 metadata，不跨层 import vision 内部实现。"""
    body = MONITOR_SRC.read_text(encoding="utf-8").split("def _yard_observation")[1]
    for forbidden in ("from src.vision", "import cv2"):
        assert forbidden not in body, forbidden
    assert 'metadata.get("prefilter_match_calls_actual")' in body
