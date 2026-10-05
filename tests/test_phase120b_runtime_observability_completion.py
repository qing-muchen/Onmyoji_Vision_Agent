"""Phase 12.0-B 测试：Runtime Observability Completion（观测补全，不改判定逻辑）。

覆盖两块**纯观测**增强：

1. **bbox 观测**：``_restore_rect_ex`` 透出「clamp 是否被触发」（= 不修正本会越出 ROI），
   经 ``TemplateEvidenceDetector.detect()`` 写入 ``metadata["bbox_clamped"]``（**仅触发时**），
   再由 ``StateMonitor`` 注入 ``CycleMetrics.yard_bbox_oob``，最后被 soak 聚合成
   ``yard.bbox.{samples,invalid}``；
2. **RSS 观测**：``psutil`` 作为**可选**依赖，缺失时全部 RSS 字段保持 ``not_available``
   （**不得**用 Python 分配量冒充常驻内存）。

关键向后兼容契约：``yard_metrics=False``（生产默认）时，``CycleMetrics.to_dict()`` 输出
**不得**包含任何 yard/观测键；单测逐字段固化。
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
from src.core.geometry import Rect  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.detector.models import TemplateInfo  # noqa: E402
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector  # noqa: E402
from src.runtime.state_monitor.metrics import CycleMetrics, CycleStatus  # noqa: E402

BASE_SRC = ROOT / "src" / "vision" / "evidence" / "base.py"
MONITOR_SRC = ROOT / "src" / "runtime" / "state_monitor" / "monitor.py"
METRICS_SRC = ROOT / "src" / "runtime" / "state_monitor" / "metrics.py"
SOAK_SRC = ROOT / "tools" / "run_runtime_soak.py"
YARD = "bottom_menu_yard"
YARD_DIR = ROOT / "dataset" / "raw" / "yard"
FORBIDDEN_IMPORTS = (
    "pyautogui",
    "pynput",
    "keyboard",
    "mouse",
    "src.action.dispatcher",
    "src.task",
)


def yard_detector():  # noqa: ANN201
    """生产 yard detector（真实配置与模板集合）。"""
    return BottomMenuYardDetector.from_config(load_config().evidence)


def metric(cycle_id: int = 1, **kwargs: object) -> CycleMetrics:
    """构造 ``CycleMetrics``（补齐必填字段；观测字段按需覆盖）。"""
    base: dict[str, object] = {
        "cycle_id": cycle_id,
        "timestamp": "2026-10-03T00:00:00+08:00",
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


# ---------------------------------------------------------------------------
# 1~5：bbox 观测
# ---------------------------------------------------------------------------


def test_restore_rect_reports_no_clamp_for_interior_rect() -> None:
    """[1] 内部矩形不触发 clamp。"""
    detector = yard_detector()
    roi = Rect(x=0, y=0, width=1704, height=192)
    raw = Rect(x=10, y=10, width=100, height=50)
    restored, clamped = detector._restore_rect_ex(raw, roi.size)  # noqa: SLF001
    assert clamped is False
    assert restored.width >= 1 and restored.height >= 1
    assert restored.x + restored.width <= roi.width
    assert restored.y + restored.height <= roi.height


def test_restore_rect_reports_clamp_when_beyond_roi() -> None:
    """[2] 越出 ROI 的矩形必须被 clamp 且报告 ``clamped=True``。"""
    detector = yard_detector()
    roi = Rect(x=0, y=0, width=1704, height=192)
    raw = Rect(x=1260, y=100, width=200, height=200)  # 右/下均越界
    restored, clamped = detector._restore_rect_ex(raw, roi.size)  # noqa: SLF001
    assert clamped is True
    assert restored.x + restored.width <= roi.width
    assert restored.y + restored.height <= roi.height


def test_restore_rect_keeps_api_and_agrees_with_ex() -> None:
    """[3] ``_restore_rect`` 仍返回单个 Rect，且与 ``_restore_rect_ex`` 结果一致。"""
    detector = yard_detector()
    roi = Rect(x=0, y=0, width=1704, height=192)
    for raw in (
        Rect(x=5, y=5, width=40, height=20),
        Rect(x=1270, y=180, width=20, height=40),
    ):
        plain = detector._restore_rect(raw, roi.size)  # noqa: SLF001
        extended, _clamped = detector._restore_rect_ex(raw, roi.size)  # noqa: SLF001
        assert (plain.x, plain.y, plain.width, plain.height) == (
            extended.x,
            extended.y,
            extended.width,
            extended.height,
        )


def test_match_and_restore_agrees_with_best_match() -> None:
    """[4] ``_match_and_restore`` 与 ``best_match`` 判定/bbox 完全一致（仅多一个标记）。"""
    import cv2

    detector = yard_detector()
    image = cv2.imread(str(YARD_DIR / "yard_skin001_001.png"), cv2.IMREAD_COLOR)
    frame = Frame.create(image, source="probe", timestamp=1.0)
    match_a, roi_a = detector.best_match(frame)
    match_b, roi_b, clamped = detector._match_and_restore(frame)  # noqa: SLF001
    assert (match_a is None) == (match_b is None)
    assert (roi_a.x, roi_a.y, roi_a.width, roi_a.height) == (roi_b.x, roi_b.y, roi_b.width, roi_b.height)
    if match_a is not None and match_b is not None:
        assert match_a.rect == match_b.rect
        assert match_a.template.template_id == match_b.template.template_id
        assert isinstance(clamped, bool)


def test_detect_metadata_bbox_clamped_only_when_triggered() -> None:
    """[5] ``metadata`` 只在 clamp 触发时含 ``bbox_clamped``（未触发时不得出现该键）。"""
    import cv2

    detector = yard_detector()
    image = cv2.imread(str(YARD_DIR / "yard_skin001_001.png"), cv2.IMREAD_COLOR)
    result = detector.detect(Frame.create(image, source="probe", timestamp=1.0))
    assert result.detected is True
    if "bbox_clamped" in result.metadata:
        assert result.metadata["bbox_clamped"] is True


# ---------------------------------------------------------------------------
# 6~8：CycleMetrics 字段与向后兼容
# ---------------------------------------------------------------------------


def test_yard_bbox_oob_field_exists_with_none_default() -> None:
    """[6] ``CycleMetrics`` 新增 ``yard_bbox_oob``，默认 ``None``。"""
    field = next(f for f in dataclasses.fields(CycleMetrics) if f.name == "yard_bbox_oob")
    assert field.default is None
    assert metric().yard_bbox_oob is None


def test_metrics_output_omits_observability_when_not_collected() -> None:
    """[7] 未采集观测时 ``to_dict`` 不含任何 yard/观测键（生产默认路径）。"""
    payload = metric().to_dict()
    observability_keys = {
        "frame_id", "vision_ms", "yard_ms", "yard_detected", "yard_confidence",
        "yard_template_id", "yard_prefilter", "yard_top_k", "yard_candidates",
        "yard_total_templates", "yard_match_calls", "yard_prefilter_ms", "yard_fallback",
        "yard_fallback_reason", "yard_bbox_oob", "snapshot_status", "snapshot_invalidated",
        "task_active", "action_enabled", "action_backend",
    }
    assert not (observability_keys & set(payload)), sorted(observability_keys & set(payload))


def test_metrics_output_includes_bbox_flag_when_collected() -> None:
    """[8] 采集到观测时按固定顺序输出，且 bbox 的 ``False`` 也会写出。"""
    payload = metric(
        1,
        yard_detected=True,
        yard_confidence=0.9,
        yard_fallback=False,
        yard_fallback_reason="none",
        yard_bbox_oob=False,
    ).to_dict()
    assert payload["yard_bbox_oob"] is False
    keys = list(payload)
    assert keys.index("yard_bbox_oob") == keys.index("yard_fallback_reason") + 1


# ---------------------------------------------------------------------------
# 9~11：monitor 注入语义
# ---------------------------------------------------------------------------


def test_monitor_injects_bbox_only_on_detection() -> None:
    """[9] 命中注入 ``True/False``；未命中保持未设置（不写 ``False``）。"""
    source = MONITOR_SRC.read_text(encoding="utf-8")
    body = source.split("def _yard_observation")[1]
    assert 'observation["yard_bbox_oob"]' in body
    assert 'if observation["yard_detected"]:' in body, "必须在命中时才注入 bbox 观测"


def test_monitor_reads_bbox_clamped_metadata_key() -> None:
    """[10] 观测来源是 evidence metadata 的 ``bbox_clamped``（跨层不 import）。"""
    body = MONITOR_SRC.read_text(encoding="utf-8").split("def _yard_observation")[1]
    assert 'metadata.get("bbox_clamped", False)' in body
    for forbidden in ("src.vision", "import cv2", "from src.vision"):
        assert forbidden not in body, forbidden


def test_detect_writes_bbox_clamped_key() -> None:
    """[11] vision 层写入键名与 runtime 读取键名一致（契约对齐）。"""
    base = BASE_SRC.read_text(encoding="utf-8")
    monitor = MONITOR_SRC.read_text(encoding="utf-8")
    assert 'metadata["bbox_clamped"] = True' in base
    assert 'metadata.get("bbox_clamped", False)' in monitor


# ---------------------------------------------------------------------------
# 12~16：RSS 观测
# ---------------------------------------------------------------------------


def test_rss_bytes_returns_int_when_psutil_available() -> None:
    """[12] 装了 psutil ⇒ RSS 为可读字节数。"""
    psutil = pytest.importorskip("psutil")
    value = soak._rss_bytes()  # noqa: SLF001
    assert isinstance(value, int) and value > 0
    assert int(psutil.Process().memory_info().rss) > 0


def test_rss_bytes_returns_none_without_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """[13] 无 psutil ⇒ ``None``（不伪造、不退化到 Python 分配量）。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):  # noqa: ANN202
        if name == "psutil":
            raise ImportError("simulated missing psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert soak._rss_bytes() is None  # noqa: SLF001


def test_rss_sampler_without_psutil_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """[14] 无 psutil ⇒ 采样器不启动线程，``stop`` 返回全 ``None``。"""
    monkeypatch.setattr(soak, "_rss_bytes", lambda: None)
    sampler = soak._RssSampler()  # noqa: SLF001
    sampler.start()
    result = sampler.stop()
    assert result["rss_start_bytes"] is None
    assert result["rss_peak_bytes"] is None
    assert result["rss_end_bytes"] is None
    assert result["rss_samples"] == 0


def test_measure_memory_notes_unavailable_without_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """[15] 无 psutil ⇒ ``rss_note`` 明确 not_available，Python 分配量仍提供。"""
    monkeypatch.setattr(soak, "_rss_bytes", lambda: None)
    memory = soak.measure_memory()
    assert memory["rss_start"] is None and memory["rss_end"] is None
    assert "not_available" in memory["rss_note"]
    assert isinstance(memory["python_alloc_current_bytes"], int)


def test_measure_memory_reports_rss_when_available() -> None:
    """[16] 有 psutil ⇒ ``rss_note`` 说明来源，且 start/peak/end 可读。"""
    pytest.importorskip("psutil")
    sampler = soak._RssSampler(interval_s=0.05)  # noqa: SLF001
    sampler.start()
    sampler.stop()
    memory = soak.measure_memory(sampler.stop())
    assert memory["rss_start"] is not None
    assert memory["rss_peak"] is not None
    assert "psutil" in memory["rss_note"]


# ---------------------------------------------------------------------------
# 17~19：soak 聚合
# ---------------------------------------------------------------------------


def test_yard_stats_aggregates_bbox() -> None:
    """[17] ``_yard_stats`` 聚合 bbox（只统计命中轮次）。"""
    records = [
        metric(1, yard_detected=True, yard_bbox_oob=False, yard_confidence=0.9),
        metric(2, yard_detected=True, yard_bbox_oob=True, yard_confidence=0.9),
        metric(3, yard_detected=False),  # 未命中：不计入
    ]
    stats = soak._yard_stats(records)  # noqa: SLF001
    assert stats["bbox"]["samples"] == 2
    assert stats["bbox"]["invalid"] == 1
    assert stats["bbox"]["clamped"] == 1


def test_yard_stats_bbox_empty_when_no_detection() -> None:
    """[18] 全部未命中 ⇒ bbox 样本数为 0（不得把它当成"非法 0"）。"""
    stats = soak._yard_stats([metric(1, yard_detected=False)])  # noqa: SLF001
    assert stats["bbox"]["samples"] == 0
    assert stats["bbox"]["invalid"] == 0


def test_print_summary_handles_missing_bbox_and_rss(capsys: pytest.CaptureFixture[str]) -> None:
    """[19] 摘要打印对"缺 bbox 键 / RSS not_available"都要能工作。"""
    summary = {
        "mode": "replay", "source": "ReplayFrameSource", "cycles": 3, "duration_s": 1.5,
        "interval_ms": 500, "errors": 0,
        "cycle_latency": {"p50": 1.0, "mean": 1.0, "p90": 1.0, "p95": 1.0, "p99": 1.0, "max": 2.0},
        "deadline": {"misses": 0, "rate": 0.0, "budget_ms": 500},
        "yard": {
            "detections": 3, "observations": 3,
            "detector_ms": {"p50": 1.0, "p95": 1.0, "max": 1.0},
            "prefilter": "thumb", "candidate_count": {"distribution": {4: 3}},
            "match_calls": {"distribution": {20: 3}}, "fallback_count": 0,
            "fallback_reasons": {},
        },
        "states": {"counts": {}, "transitions": 0, "flaps": 0},
        "capture": {"success": 3, "failures": 0, "skipped_reasons": {}},
        "snapshot": {"statuses": {}, "invalidations": 0},
        "memory": {"rss_start": None, "rss_peak": None, "rss_end": None, "rss_samples": 0,
                   "rss_note": "psutil 未安装", "python_alloc_current_bytes": 1,
                   "python_alloc_peak_bytes": 2},
        "safety": {"action_enabled": False, "backend": "dry_run", "real_input": False},
    }
    soak.print_summary(summary)
    out = capsys.readouterr().out
    assert "not_available" in out
    assert "bbox" not in out  # 无 bbox 键时不打印该行


# ---------------------------------------------------------------------------
# 20~26：纪律与不变性
# ---------------------------------------------------------------------------


def test_soak_summary_json_contains_new_fields(tmp_path: Path) -> None:
    """[20] 汇总字典（写 TEMP 的 JSON）含 ``bbox`` 与 ``rss_*`` 新键。"""
    source = SOAK_SRC.read_text(encoding="utf-8")
    assert '"bbox": {' in source
    assert '"rss_start"' in source and '"rss_samples"' in source
    payload = {"yard": soak._yard_stats([]), "memory": soak.measure_memory()}  # noqa: SLF001
    assert "bbox" in payload["yard"]
    assert "rss_note" in payload["memory"]
    json.dumps(payload)  # 必须可 JSON 序列化


def test_observability_does_not_change_decision_logic() -> None:
    """[21] 观测补全不得触碰判定阈值/预筛/matcher 语义。"""
    config = load_config()
    yard = config.evidence.detectors[YARD]
    assert yard.threshold == 0.85
    assert yard.roi_scale == 0.75
    assert yard.template_prefilter == "thumb"
    assert yard.prefilter_top_k == 4
    assert len(yard.template_include) == 36
    for path in (BASE_SRC, MONITOR_SRC, METRICS_SRC):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("FIT_TOLERANCE_PX =", "threshold = 0.", "roi_scale = 0."):
            assert forbidden not in text, (path.name, forbidden)


def test_yard_metrics_default_remains_false() -> None:
    """[22] 观测开关的生产默认值仍为 ``False``（零额外开销）。"""
    assert load_config().runtime.state_monitor.yard_metrics is False


def test_safety_flags_unchanged() -> None:
    """[23] 安全开关未被观测改动影响。"""
    config = load_config()
    assert config.action.enabled is False
    assert config.action.backend == "dry_run"
    assert not hasattr(config.action, "real_input")


def test_no_input_control_dependencies_added() -> None:
    """[24] 新增观测不引入任何输入控制依赖（``src.action.resolver`` 只读定位**允许**）。"""
    for path in (BASE_SRC, MONITOR_SRC, METRICS_SRC, SOAK_SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        called: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name:
                    called.add(name)
        for forbidden in FORBIDDEN_IMPORTS:
            assert not any(name.startswith(forbidden) for name in imported), (path.name, forbidden)
        # 只读的视觉定位 resolver 允许出现；真正的动作执行入口一律禁止
        for forbidden in ("ActionDispatcher", "InputBackend", "click", "press", "typewrite",
                          "SetForegroundWindow", "keyDown", "keyUp"):
            assert forbidden not in called, (path.name, forbidden)


def test_psutil_is_optional_dependency_only() -> None:
    """[25] ``psutil`` 只作为可选依赖出现在开发依赖清单里。"""
    text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "psutil" in text
    runtime = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "psutil" not in runtime, "RSS 观测不得成为运行时硬依赖"


def test_rss_sampler_only_reports_real_values() -> None:
    """[26] 采样器只汇报真实读数（不可用时绝不返回 0 或臆造值）。"""
    pytest.importorskip("psutil")
    sampler = soak._RssSampler(interval_s=0.05)  # noqa: SLF001
    sampler.start()
    result = sampler.stop()
    assert result["rss_samples"] >= 1
    assert result["rss_peak_bytes"] >= result["rss_start_bytes"] > 0


def _unused_numpy_guard() -> None:  # pragma: no cover - 仅保留 numpy 依赖的显式引用
    """占位：确保测试模块对 numpy 的引用是有意的（图像构造辅助）。"""
    assert np is not None
