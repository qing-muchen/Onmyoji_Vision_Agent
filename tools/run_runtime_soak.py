"""Phase 11.8：Runtime soak 运行器（**只读观测**；绝不发送真实输入）。

用途：让 ``Production36 + thumb Top-4`` 在**连续运行**环境中接受验证 —— 节拍、状态稳定性、
预筛回退、快照失效与资源趋势。两种模式：

============================  =============================================================
模式                           说明
============================  =============================================================
``--mode real``                真实 Runtime（``WindowFrameSource``，真实窗口）。
                               窗口不可用时**明确返回** ``window_unavailable``，不伪造成功。
                               **不**激活窗口、**不**切换前台、**不**发送任何键鼠输入。
``--mode replay``              离线图片序列（``dataset/raw`` 的真实原图），可复现。
                               用于验证 metrics / 状态迁移 / buffer / summary / 快照失效；
                               **不代表**真实运行期性能结论（报告中标为 ``offline_replay``）。
============================  =============================================================

安全承诺（本阶段硬约束）：

* 全程 ``action.enabled=false``、``backend=dry_run`` ⇒ ``real_input=false``；
* 不调用 ``pyautogui`` / ``ActionDispatcher`` / ``InputBackend``，也不做窗口激活；
* 采集失败/失焦时按既有语义跳过该轮（不伪造帧、不复用旧帧）；
* 所有输出（JSON summary / 可选 JSONL 指标）只写系统 TEMP。

用法::

    python tools/run_runtime_soak.py --mode replay --max-cycles 40 --summary-every 10
    python tools/run_runtime_soak.py --mode real --duration 300      # 5 分钟真实 soak
    python tools/run_runtime_soak.py --mode real --duration 30       # 短测
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tracemalloc
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.config import AppConfig, load_config  # noqa: E402
from src.runtime.state_monitor.metrics import RuntimeMetricsRecorder, percentile  # noqa: E402
from src.runtime.state_monitor.monitor import StateMonitor  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ova118" / "runtime_soak_summary.json"
DEFAULT_METRICS = Path(tempfile.gettempdir()) / "ova118" / "runtime_soak_metrics.jsonl"

#: 观测目标证据
YARD_EVIDENCE = "bottom_menu_yard"
#: 内存中保留的最大逐轮记录数（**有界**：FIFO 淘汰最旧记录，防止长跑无界增长）
MAX_SOAK_RECORDS: int = 20000
#: 判定"状态抖动"的时间窗（秒）：``A→B→A`` 在窗口内发生即记为一次 flap
FLAP_WINDOW_S = 2.0
#: 离线 replay 的默认场景脚本（group, 连续轮数）——对应 11.8 的 A/C 场景
DEFAULT_REPLAY_SCRIPT: tuple[tuple[str, int], ...] = (
    ("yard", 8),
    ("exploration", 3),
    ("yard", 5),
    ("unknown/other", 3),
    ("yard", 5),
)
#: 离线 replay 中注入 capture 失败的位置（1-based；用于验证快照失效与跳过语义）
DEFAULT_REPLAY_FAILURES: tuple[int, ...] = (12, 13)


# ---------------------------------------------------------------------------
# 离线 replay 帧源
# ---------------------------------------------------------------------------


def read_image(path: Path) -> np.ndarray | None:
    """读图（兼容非 ASCII 路径）。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


class ReplayFrameSource:
    """按离线场景脚本产出帧（**可复现**；支持注入采集失败）。

    实现 :class:`~src.runtime.state_monitor.source.FrameSource` 协议；同时提供
    ``last_skip_reason`` 以配合 :class:`StatusAwareSource` 的跳过原因记录。

    Args:
        script: ``(raw 子目录, 连续轮数)`` 序列。
        dataset: 数据根目录。
        fail_positions: 在这些 1-based 轮次返回 ``None``（模拟 capture 失败）。
        loop: 脚本走完后是否从头循环（``max_cycles`` 大于脚本长度时需要）。
    """

    def __init__(
        self,
        script: tuple[tuple[str, int], ...] = DEFAULT_REPLAY_SCRIPT,
        *,
        dataset: Path = DATASET,
        fail_positions: tuple[int, ...] = DEFAULT_REPLAY_FAILURES,
        loop: bool = True,
    ) -> None:
        self._plan: list[tuple[str, Path]] = []
        for group, count in script:
            paths = sorted((dataset / "raw" / group).glob("*.png"))
            if not paths:
                continue
            for index in range(count):
                self._plan.append((group, paths[index % len(paths)]))
        if not self._plan:
            raise ValueError("replay 脚本为空：dataset/raw 下没有可用原图")
        self._fail = set(fail_positions)
        self._loop = loop
        self._index = 0
        self._last_skip_reason: str | None = None

    @property
    def plan_size(self) -> int:
        """脚本长度（不含循环）。"""
        return len(self._plan)

    @property
    def last_skip_reason(self) -> str | None:
        """最近一次跳过的原因（配合 ``StatusAwareSource``）。"""
        return self._last_skip_reason

    def get_frame(self) -> Frame | None:
        """产出下一帧；注入位置返回 ``None``（视为采集失败）。"""
        position = self._index + 1
        if not self._loop and self._index >= len(self._plan):
            return None
        group, path = self._plan[self._index % len(self._plan)]
        self._index += 1
        if position in self._fail:
            self._last_skip_reason = f"capture_failed_injected(position={position})"
            return None
        image = read_image(path)
        if image is None:
            self._last_skip_reason = f"image_unreadable({path.name})"
            return None
        self._last_skip_reason = None
        return Frame.create(image, source=f"{group}:{path.name}", timestamp=float(position))


def build_vision_bridge(config: AppConfig) -> tuple[Any | None, str]:
    """构造**只读**的视觉快照桥（用于观测快照失效）。

    只用于观测：桥只把每轮视觉结果 push 给一个**内存中**的 resolver，
    **不**接 ``ActionDispatcher``、**不**接输入后端、不产生任何键鼠事件，
    因此不会改变生产安全语义（``action.enabled`` 仍为 ``false``）。

    Returns:
        ``(bridge, note)``；无法构造时 ``bridge=None`` 并给出 ``not_available`` 原因。
    """
    try:
        from src.action.resolver import VisionAwareTargetResolver
        from src.runtime.vision_bridge import VisionSnapshotBridge
    except Exception as exc:  # noqa: BLE001
        return None, f"not_available:import_failed:{type(exc).__name__}"
    try:
        resolver = VisionAwareTargetResolver(
            ttl_s=float(config.action.vision_target_ttl_s),
            minimum_confidence=float(config.action.vision_target_min_confidence),
        )
        return VisionSnapshotBridge(resolver=resolver), "attached(observation_only)"
    except Exception as exc:  # noqa: BLE001
        return None, f"not_available:resolver_init_failed:{type(exc).__name__}: {exc}"


def _source_kind(source: Any) -> str:
    """帧来源类型名（用于报告标注）。"""
    return type(source).__name__


# ---------------------------------------------------------------------------
# 真实窗口可用性
# ---------------------------------------------------------------------------


def probe_window(config: AppConfig) -> tuple[bool, str]:
    """检查真实游戏窗口是否可用（**只检查，不激活、不抢占**）。

    Returns:
        ``(available, reason)``；不可用时 ``reason`` 为 ``window_unavailable`` 的细分。
    """
    try:
        from src.runtime.state_monitor.frame_source import WindowFrameSource
    except Exception as exc:  # noqa: BLE001
        return False, f"import_failed:{type(exc).__name__}: {exc}"
    try:
        source = WindowFrameSource.from_config(config)
    except Exception as exc:  # noqa: BLE001
        return False, f"source_init_failed:{type(exc).__name__}: {exc}"
    binding = source.binding
    window = binding.find(refresh=True)
    if window is None:
        return False, "window_not_found"
    if binding.is_minimized():
        return False, "window_minimized"
    if not binding.is_foreground():
        # 注意：**不**抢焦点，只报告
        return False, "window_not_foreground"
    return True, "ok"


# ---------------------------------------------------------------------------
# 运行与统计
# ---------------------------------------------------------------------------


def _state_timeline(monitor: StateMonitor) -> list[tuple[str, float]]:
    """稳定状态时间线 ``[(state, epoch_seconds), ...]``（按事件顺序）。"""
    timeline: list[tuple[str, float]] = []
    for event in monitor.events:
        try:
            stamp = datetime.fromisoformat(event.timestamp).timestamp()
        except ValueError:  # pragma: no cover - 时间戳格式异常时退化为顺序
            stamp = float(len(timeline))
        timeline.append((event.current_state.value, stamp))
    return timeline


def count_flaps(timeline: list[tuple[str, float]], *, window_s: float = FLAP_WINDOW_S) -> int:
    """统计状态抖动次数：``A → B → A`` 在 ``window_s`` 内发生即记一次。"""
    flaps = 0
    for index in range(len(timeline) - 2):
        first_state, first_time = timeline[index]
        middle_state, _middle_time = timeline[index + 1]
        last_state, last_time = timeline[index + 2]
        if first_state == last_state and middle_state != first_state:
            if (last_time - first_time) <= window_s:
                flaps += 1
    return flaps


def _yard_stats(records: list[Any]) -> dict[str, Any]:
    """从逐轮指标聚合 yard 观测（未采集到的项为 ``None``）。"""
    rows = [record.to_dict() for record in records]
    detected = [row for row in rows if row.get("yard_detected")]
    yard_ms = [row["yard_ms"] for row in rows if row.get("yard_ms") is not None]
    candidates = [row["yard_candidates"] for row in rows if row.get("yard_candidates") is not None]
    calls = [row["yard_match_calls"] for row in rows if row.get("yard_match_calls") is not None]
    prefilter_ms = [
        row["yard_prefilter_ms"] for row in rows if row.get("yard_prefilter_ms") is not None
    ]
    confidence = [
        row["yard_confidence"] for row in detected if row.get("yard_confidence") is not None
    ]
    fallback_rows = [row for row in rows if row.get("yard_fallback")]
    reasons = Counter(row.get("yard_fallback_reason") or "(unknown)" for row in fallback_rows)
    prefilter_names = {row.get("yard_prefilter") for row in rows if row.get("yard_prefilter")}

    return {
        "observations": len(rows),
        "detections": len(detected),
        "detector_ms": _latency_stats(yard_ms),
        "prefilter_ms": _latency_stats(prefilter_ms),
        "candidate_count": {
            "samples": len(candidates),
            "min": min(candidates) if candidates else None,
            "max": max(candidates) if candidates else None,
            "p50": percentile([float(value) for value in candidates], 0.5) if candidates else None,
            "distribution": dict(sorted(Counter(candidates).items())),
        },
        "match_calls": {
            "samples": len(calls),
            "min": min(calls) if calls else None,
            "max": max(calls) if calls else None,
            "distribution": dict(sorted(Counter(calls).items())),
        },
        "best_confidence": {
            "count": len(confidence),
            "max": round(max(confidence), 6) if confidence else None,
            "mean": round(float(np.mean(confidence)), 6) if confidence else None,
            "min": round(min(confidence), 6) if confidence else None,
        },
        "prefilter": sorted(name for name in prefilter_names if name),
        "fallback_count": len(fallback_rows),
        "fallback_rate": round(len(fallback_rows) / len(rows), 4) if rows else None,
        "fallback_reasons": dict(sorted(reasons.items())),
        "fallback_cycles": [
            {
                "cycle_id": row.get("cycle_id"),
                "reason": row.get("yard_fallback_reason"),
                "state": row.get("state"),
                "candidate_count": row.get("yard_candidates"),
            }
            for row in fallback_rows[:20]
        ],
    }


def _latency_stats(values: list[float]) -> dict[str, Any]:
    """耗时分位数（p50/p90/p95/p99/max/mean）。"""
    if not values:
        return {"samples": 0, "p50": None, "mean": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "samples": len(values),
        "p50": percentile(values, 0.5),
        "mean": round(float(np.mean(values)), 3),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": round(max(values), 3),
    }


def build_summary(
    monitor: StateMonitor,
    recorder: RuntimeMetricsRecorder,
    *,
    mode: str,
    duration_s: float,
    cycles: int,
    memory: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装结构化 soak summary（Phase 11.8 契约）。"""
    records = list(recorder.records)
    rows = [record.to_dict() for record in records]
    summary = recorder.summary()
    latencies = [record.latency_ms for record in records]
    stable_states = Counter(row.get("stable_state") or "(pending)" for row in rows)
    timeline = _state_timeline(monitor)

    capture_failures = sum(1 for row in rows if row.get("status") == "skipped")
    snapshot_statuses = Counter(
        row.get("snapshot_status") or "(bridge_not_attached)" for row in rows
    )
    payload: dict[str, Any] = {
        "phase": "11.8",
        "mode": mode,
        "duration_s": round(duration_s, 3),
        "cycles": cycles,
        "interval_ms": monitor.config.interval_ms,
        "cycle_latency": _latency_stats(latencies),
        "deadline": {
            "misses": summary.late_cycles,
            "rate": round(summary.late_cycles / len(records), 4) if records else None,
            "budget_ms": monitor.config.interval_ms,
            "note": "miss 沿用 CycleScheduler 的 overrun 语义（elapsed > interval）",
        },
        "states": {
            "counts": dict(sorted(stable_states.items())),
            "transitions": len(timeline),
            "transition_pairs": dict(
                sorted(
                    Counter(
                        f"{timeline[i][0]}->{timeline[i + 1][0]}" for i in range(len(timeline) - 1)
                    ).items()
                )
            ),
            "flaps": count_flaps(timeline),
            "flap_window_s": FLAP_WINDOW_S,
        },
        "yard": _yard_stats(records),
        "capture": {
            "success": sum(1 for row in rows if row.get("status") == "ok"),
            "failures": capture_failures,
            "skipped_reasons": recorder.skip_counts(),
            "capture_status": dict(
                sorted(Counter(row.get("capture_status") or "(none)" for row in rows).items())
            ),
            "foreground_status": dict(
                sorted(Counter(row.get("foreground_status") or "(none)" for row in rows).items())
            ),
            "unknown_reasons": dict(
                sorted(Counter(row.get("unknown_reason") or "(none)" for row in rows).items())
            ),
        },
        "snapshot": {
            "statuses": dict(sorted(snapshot_statuses.items())),
            "invalidations": max(
                [row["snapshot_invalidated"] for row in rows if row.get("snapshot_invalidated") is not None]
                or [0]
            ),
            "expired": None,
            "expired_note": "TTL 过期由 Action 侧 resolver 判定，运行期不可见 ⇒ not_available",
        },
        "memory": memory,
        "metrics_buffer": {
            "max_records": MAX_SOAK_RECORDS,
            "note": "逐轮记录**有界**（FIFO 淘汰最旧）；统计基于保留的最近 N 轮，长跑不会无界增长",
        },
        "safety": {
            "action_enabled": monitor._action_enabled,  # noqa: SLF001 - 装配期记录的安全快照
            "backend": monitor._action_backend,  # noqa: SLF001
            "real_input": False,
            "note": "本工具不调用 ActionDispatcher / InputBackend，也不激活或切换窗口",
        },
        "errors": monitor.errors,
        "report": monitor._last_report.describe() if getattr(monitor, "_last_report", None) else None,  # noqa: SLF001
    }
    if extra:
        payload.update(extra)
    return payload


def measure_memory() -> dict[str, Any]:
    """内存快照：``tracemalloc`` 的 Python 分配量（RSS 需要 psutil，未安装 ⇒ not_available）。"""
    current, peak = tracemalloc.get_traced_memory()
    return {
        "rss_start": None,
        "rss_end": None,
        "rss_peak": None,
        "rss_note": "psutil 未安装 ⇒ RSS 为 not_available；使用 tracemalloc 的 Python 分配量",
        "python_alloc_current_bytes": current,
        "python_alloc_peak_bytes": peak,
    }


def run_soak(
    config: AppConfig,
    source: Any,
    *,
    max_cycles: int | None = None,
    duration_s: float | None = None,
    mode: str = "replay",
    record_metrics: bool = False,
    summary_every: int = 0,
) -> dict[str, Any]:
    """运行一次 soak 并返回 summary。

    Args:
        config: 应用配置（``action.enabled=false`` / ``backend=dry_run`` 由配置保证）。
        source: 帧来源（真实窗口或 replay）。
        max_cycles: 最多轮数。
        duration_s: 最长时长（秒）。
        mode: ``real`` / ``replay``（仅用于报告标注）。
        record_metrics: 是否额外写 JSONL 指标到 TEMP。
        summary_every: 每 N 轮打印一次进度（``0`` = 不打印）。

    Returns:
        结构化 summary。
    """
    monitor_config = config.runtime.state_monitor.model_copy(
        update={
            "yard_metrics": True,
            "save_metrics": False,
            "metrics_path": str(DEFAULT_METRICS if record_metrics else config.runtime.state_monitor.metrics_path),
            "save_history": False,
        }
    )
    runtime_config = config.runtime.model_copy(update={"state_monitor": monitor_config})
    active = config.model_copy(update={"runtime": runtime_config})

    recorder = RuntimeMetricsRecorder(
        path=DEFAULT_METRICS if record_metrics else None, max_records=MAX_SOAK_RECORDS
    )
    vision_bridge, bridge_note = build_vision_bridge(config)
    monitor = StateMonitor.from_config(
        active,
        source=source,
        metrics=recorder,
        vision_bridge=vision_bridge,
        on_cycle=(
            (lambda schedule: _print_progress(recorder, schedule, every=summary_every))
            if summary_every > 0
            else None
        ),
    )

    tracemalloc.start()
    started = time.perf_counter()
    monitor.run(duration_s=duration_s, max_cycles=max_cycles)
    elapsed = time.perf_counter() - started

    memory = measure_memory()
    tracemalloc.stop()
    return build_summary(
        monitor,
        recorder,
        mode=mode,
        duration_s=elapsed,
        cycles=len(recorder.records),
        memory=memory,
        extra={"source": _source_kind(source), "snapshot_bridge": bridge_note},
    )


def _print_progress(recorder: RuntimeMetricsRecorder, schedule: Any, *, every: int) -> None:
    """每 ``every`` 轮打印一次进度（保持输出简洁）。"""
    if every <= 0 or len(recorder.records) % every:
        return
    summary = recorder.summary()
    print(
        f"    … {len(recorder.records)} cycles | latency p50={summary.p50_latency_ms:.1f}ms "
        f"p95={summary.p95_latency_ms:.1f}ms late={summary.late_cycles}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def print_summary(summary: dict[str, Any]) -> None:
    """打印可读摘要。"""
    print("=" * 96)
    print(
        f"Phase 11.8 Runtime soak（mode={summary['mode']}，source={summary.get('source')}）"
    )
    print(
        f"  cycles={summary['cycles']} duration={summary['duration_s']}s "
        f"interval={summary['interval_ms']}ms errors={summary['errors']}"
    )
    latency = summary["cycle_latency"]
    print(
        f"  cycle latency(ms): p50={latency['p50']} mean={latency['mean']} p90={latency['p90']} "
        f"p95={latency['p95']} p99={latency['p99']} max={latency['max']}"
    )
    deadline = summary["deadline"]
    print(f"  deadline: misses={deadline['misses']} rate={deadline['rate']} budget={deadline['budget_ms']}ms")
    yard = summary["yard"]
    print(
        f"  yard: detections={yard['detections']}/{yard['observations']} "
        f"detector_ms p50={yard['detector_ms']['p50']} p95={yard['detector_ms']['p95']} "
        f"max={yard['detector_ms']['max']}"
    )
    print(
        f"  prefilter: {yard['prefilter']} candidates={yard['candidate_count']['distribution']} "
        f"match_calls={yard['match_calls']['distribution']} "
        f"fallback={yard['fallback_count']} ({yard['fallback_reasons']})"
    )
    print(f"  states: counts={summary['states']['counts']} transitions={summary['states']['transitions']} flaps={summary['states']['flaps']}")
    capture = summary["capture"]
    print(f"  capture: success={capture['success']} failures={capture['failures']} reasons={capture['skipped_reasons']}")
    print(
        f"  snapshot: statuses={summary['snapshot']['statuses']} invalidations={summary['snapshot']['invalidations']}"
    )
    memory = summary["memory"]
    print(
        f"  memory(python): start={memory['python_alloc_current_bytes']} "
        f"peak={memory['python_alloc_peak_bytes']} | rss: not_available"
    )
    print(f"  safety: {summary['safety']}")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="Runtime soak 运行器（只读观测；不发送真实输入、不激活窗口）"
    )
    parser.add_argument("--mode", choices=("real", "replay"), default="replay")
    parser.add_argument("--duration", type=float, default=None, help="最长时长（秒）")
    parser.add_argument("--max-cycles", type=int, default=None, help="最多轮数")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="summary JSON 路径（TEMP）")
    parser.add_argument("--summary-every", type=int, default=0, help="每 N 轮打印进度")
    parser.add_argument("--record-metrics", action="store_true", help="额外写 JSONL 指标到 TEMP")
    parser.add_argument("--replay-failures", default=",".join(str(v) for v in DEFAULT_REPLAY_FAILURES))
    parser.add_argument("--no-loop", action="store_true", help="replay 脚本不循环")
    args = parser.parse_args(argv)

    config = load_config()
    if config.action.enabled or config.action.backend != "dry_run":
        print("!! 安全配置不是 action.enabled=false / backend=dry_run，拒绝运行")
        return 1

    mode = args.mode
    if mode == "real":
        available, reason = probe_window(config)
        if not available:
            summary: dict[str, Any] = {
                "phase": "11.8",
                "mode": "real",
                "real_runtime_soak": "not_run",
                "reason": reason,
                "note": "窗口不可用：不伪造结果，也不自动启动或激活游戏窗口",
                "safety": {
                    "action_enabled": config.action.enabled,
                    "backend": config.action.backend,
                    "real_input": False,
                },
            }
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        from src.runtime.state_monitor.frame_source import WindowFrameSource

        with WindowFrameSource.from_config(config) as source:
            summary = run_soak(
                config,
                source,
                max_cycles=args.max_cycles,
                duration_s=args.duration,
                mode="real",
                record_metrics=args.record_metrics,
                summary_every=args.summary_every,
            )
    else:
        failures = tuple(
            int(item) for item in str(args.replay_failures).split(",") if item.strip()
        )
        source = ReplayFrameSource(
            fail_positions=failures, loop=not args.no_loop
        )
        summary = run_soak(
            config,
            source,
            max_cycles=args.max_cycles or source.plan_size,
            duration_s=args.duration,
            mode="replay",
            record_metrics=args.record_metrics,
            summary_every=args.summary_every,
        )
        summary["offline_replay"] = True
        summary["replay"] = {
            "plan_size": source.plan_size,
            "script": [list(item) for item in DEFAULT_REPLAY_SCRIPT],
            "injected_capture_failures": list(failures),
            "warning": "offline_replay 不代表真实运行期性能；真实 wall-clock 结论需真实窗口 soak",
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print_summary(summary)
    print("=" * 96)
    print(f"summary JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
