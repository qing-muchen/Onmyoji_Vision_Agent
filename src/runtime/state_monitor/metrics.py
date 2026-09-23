"""运行期观测指标（Phase 5.2）：每轮的 ``cycle_id / timestamp / state / stable_state / latency``。

**只记录"运行"本身**，不记录画面内容，也不记录任何用户数据：

* 记录：轮次序号、时间戳（本地时间）、原始判定状态与置信度、稳定状态、本轮耗时、
  取帧耗时、本轮结果（ok / skipped / error）、跳过原因、采集后端名称；
* **不记录**：截图（像素一律不落盘）、窗口标题、鼠标键盘输入、任何账号或任务数据。

落盘格式为 JSON Lines（``logs/runtime_metrics.jsonl``，每轮一行）：
崩溃/中断时已写入的行不会丢失，也便于长跑后直接用脚本统计。
写入失败**不影响主循环**（观测数据不应该让运行中断），只累加 ``write_errors`` 并告警。

与历史记录（:class:`~src.runtime.state_monitor.history.StateHistoryStore`）的区别：
历史记录是**状态变化事件**（业务契约，写入失败会抛 :class:`~src.core.exceptions.MonitorError`），
本模块是**逐轮观测**（诊断用，尽最大努力写入）。
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.core.config import RuntimeStateMonitorConfig
from src.core.paths import resolve_from_project_root
from src.utils.logger import get_logger
from src.vision.state.models import StateName

__all__ = [
    "CycleMetrics",
    "CycleStatus",
    "RuntimeMetricsRecorder",
    "RuntimeMetricsSummary",
    "percentile",
]

logger = get_logger("runtime.state_monitor.metrics")

#: 内存中保留的最大记录数；``0`` 表示不限制（长跑时可用它约束内存占用）
DEFAULT_METRICS_MAX_RECORDS: int = 0
#: 统计用分位数
P95_FRACTION: float = 0.95


class CycleStatus(StrEnum):
    """单轮运行结果。"""

    #: 取到帧并完成判定
    OK = "ok"
    #: 本轮无可用帧（窗口不可用 / 前台门控未通过 / 采集未产出），已跳过
    SKIPPED = "skipped"
    #: 判定过程失败并降级为 ``unknown``（异常被 Runtime 捕获，循环继续）
    ERROR = "error"


def percentile(values: list[float], fraction: float) -> float:
    """按最近秩（nearest-rank）计算分位数。

    Args:
        values: 样本（未排序也可以）。
        fraction: 分位数位置，``0 < fraction <= 1``。

    Returns:
        分位数；样本为空时返回 ``0.0``。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass(frozen=True, slots=True)
class CycleMetrics:
    """单轮运行指标。

    Attributes:
        cycle_id: 轮次序号（从 1 开始）。
        timestamp: 本地时间戳（ISO 8601，与 :class:`~src.runtime.state_monitor.events.StateEvent` 同格式）。
        started_at: 本轮开始时刻（单调时钟秒数，仅用于本地统计，不做落盘）。
        latency_ms: 本轮端到端耗时（取帧 → 判定 → 稳定化）。
        frame_ms: 取帧耗时（含前台门控与采集）。
        state: 本轮**原始**判定状态；跳过的轮次为 ``None``。
        confidence: 本轮判定置信度（跳过时为 ``0.0``）。
        stable_state: 本轮结束时的**稳定状态**；尚未确认为 ``None``。
        status: 本轮结果（ok / skipped / error）。
        skip_reason: 跳过原因（仅 ``status=skipped`` 时非 ``None``）。
        frame_source: 采集后端名称（例如 ``dxcam``）；跳过时为 ``None``。
    """

    cycle_id: int
    timestamp: str
    started_at: float
    latency_ms: float
    frame_ms: float
    state: StateName | None
    confidence: float
    stable_state: StateName | None
    status: CycleStatus
    skip_reason: str | None = None
    frame_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        """返回可 JSON 序列化的字典（枚举转字符串，不含图像与用户数据）。"""
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "state": self.state.value if self.state is not None else None,
            "confidence": round(self.confidence, 6),
            "stable_state": self.stable_state.value if self.stable_state is not None else None,
            "latency_ms": round(self.latency_ms, 3),
            "frame_ms": round(self.frame_ms, 3),
            "status": self.status.value,
            "skip_reason": self.skip_reason,
            "frame_source": self.frame_source,
        }

    def describe(self) -> str:
        """返回可读描述（单行，便于逐轮打印）。"""
        state = self.state.value if self.state is not None else "-"
        stable = self.stable_state.value if self.stable_state is not None else "pending"
        reason = f" reason={self.skip_reason}" if self.skip_reason else ""
        return (
            f"cycle={self.cycle_id} latency={self.latency_ms:.1f}ms frame={self.frame_ms:.1f}ms "
            f"state={state} stable={stable} status={self.status.value}{reason}"
        )


@dataclass(frozen=True, slots=True)
class RuntimeMetricsSummary:
    """一次运行的观测统计。

    Attributes:
        cycles: 记录到的轮次数。
        ok: 完成判定的轮次数。
        skipped: 因无可用帧而跳过的轮次数。
        errors: 判定失败的轮次数。
        late_cycles: 处理耗时超过周期的轮次数（来自调度器）。
        duration_s: 运行时长（秒，按首末轮的单调时钟计算）。
        mean_latency_ms / p50_latency_ms / p95_latency_ms / max_latency_ms: 每轮耗时统计。
        mean_frame_ms: 取帧耗时均值。
        mean_overrun_ms: 平均超时量（含 0 值的全部轮次）。
    """

    cycles: int
    ok: int
    skipped: int
    errors: int
    late_cycles: int
    duration_s: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    mean_frame_ms: float
    mean_overrun_ms: float

    @property
    def fps(self) -> float:
        """平均每秒完成轮次数（运行时长过短时返回 ``0.0``）。"""
        return self.cycles / self.duration_s if self.duration_s > 0 else 0.0

    def describe(self) -> str:
        """返回可读的多行摘要。"""
        return (
            f"cycles={self.cycles} fps={self.fps:.2f} ok={self.ok} skipped={self.skipped} "
            f"errors={self.errors} late={self.late_cycles}\n"
            f"  latency(ms): mean={self.mean_latency_ms:.1f} p50={self.p50_latency_ms:.1f} "
            f"p95={self.p95_latency_ms:.1f} max={self.max_latency_ms:.1f} "
            f"frame_mean={self.mean_frame_ms:.1f} overrun_mean={self.mean_overrun_ms:.1f}"
        )


class RuntimeMetricsRecorder:
    """逐轮记录运行指标（内存 + 可选 JSONL 落盘）。

    典型用法::

        recorder = RuntimeMetricsRecorder.from_config(config.runtime.state_monitor)
        monitor = StateMonitor.from_config(config, source=source, metrics=recorder)
        monitor.run(duration_s=600)
        print(recorder.summary().describe())

    Args:
        path: JSONL 输出路径；``None`` 表示只记录在内存中（不落盘）。
        max_records: 内存中保留的最大记录数（``0`` = 不限制，丢弃最旧记录）。

    Raises:
        ValueError: ``max_records`` 为负数。
    """

    def __init__(self, *, path: Path | None = None, max_records: int = DEFAULT_METRICS_MAX_RECORDS) -> None:
        if max_records < 0:
            raise ValueError(f"max_records 不能为负数: {max_records}")
        self._path = path
        self._records: deque[CycleMetrics] = deque(maxlen=max_records or None)
        self._late_cycles = 0
        self._overrun_total_ms = 0.0
        self._write_errors = 0

    @classmethod
    def from_config(cls, config: RuntimeStateMonitorConfig) -> RuntimeMetricsRecorder:
        """按监控配置构造记录器。

        ``runtime.state_monitor.save_metrics`` 为 ``True`` 时把指标写入
        ``runtime.state_monitor.metrics_path``（相对路径以项目根为基准）；
        否则只保留在内存中。

        Args:
            config: 状态监控配置。

        Returns:
            记录器实例。
        """
        if not config.save_metrics:
            return cls(path=None)
        return cls(path=resolve_from_project_root(config.metrics_path))

    @property
    def path(self) -> Path | None:
        """JSONL 输出路径；未落盘时为 ``None``。"""
        return self._path

    @property
    def records(self) -> tuple[CycleMetrics, ...]:
        """已记录的轮次指标。"""
        return tuple(self._records)

    @property
    def last(self) -> CycleMetrics | None:
        """最近一轮的指标。"""
        return self._records[-1] if self._records else None

    @property
    def write_errors(self) -> int:
        """落盘失败次数（不影响主循环）。"""
        return self._write_errors

    def reset(self) -> None:
        """清空记录与统计（不删除已落盘的文件内容）。"""
        self._records.clear()
        self._late_cycles = 0
        self._overrun_total_ms = 0.0
        self._write_errors = 0

    def record(self, metrics: CycleMetrics, *, overrun_ms: float = 0.0) -> None:
        """记录一轮指标。

        Args:
            metrics: 本轮指标。
            overrun_ms: 本轮超时量（来自 :class:`~src.runtime.state_monitor.scheduler.CycleSchedule`）；
                仅用于统计，不写入单行 JSON。
        """
        self._records.append(metrics)
        if overrun_ms > 0:
            self._late_cycles += 1
            self._overrun_total_ms += overrun_ms
        if self._path is not None:
            self._append_line(metrics)
        logger.debug("运行指标: %s", metrics.describe())

    def summary(self) -> RuntimeMetricsSummary:
        """汇总统计。

        Returns:
            统计对象；没有任何记录时各计数为 0、耗时统计为 0.0。
        """
        records = list(self._records)
        if not records:
            return RuntimeMetricsSummary(
                cycles=0,
                ok=0,
                skipped=0,
                errors=0,
                late_cycles=self._late_cycles,
                duration_s=0.0,
                mean_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                max_latency_ms=0.0,
                mean_frame_ms=0.0,
                mean_overrun_ms=0.0,
            )

        latencies = [record.latency_ms for record in records]
        frame_times = [record.frame_ms for record in records]
        first = records[0]
        last = records[-1]
        duration_s = max(0.0, last.started_at + last.latency_ms / 1000.0 - first.started_at)
        return RuntimeMetricsSummary(
            cycles=len(records),
            ok=sum(1 for record in records if record.status is CycleStatus.OK),
            skipped=sum(1 for record in records if record.status is CycleStatus.SKIPPED),
            errors=sum(1 for record in records if record.status is CycleStatus.ERROR),
            late_cycles=self._late_cycles,
            duration_s=duration_s,
            mean_latency_ms=sum(latencies) / len(latencies),
            p50_latency_ms=percentile(latencies, 0.5),
            p95_latency_ms=percentile(latencies, P95_FRACTION),
            max_latency_ms=max(latencies),
            mean_frame_ms=sum(frame_times) / len(frame_times),
            mean_overrun_ms=self._overrun_total_ms / len(records),
        )

    def skip_counts(self) -> dict[str, int]:
        """按跳过原因统计轮次数（仅 ``status=skipped`` 的记录）。"""
        counts: dict[str, int] = {}
        for record in self._records:
            if record.status is not CycleStatus.SKIPPED:
                continue
            reason = record.skip_reason or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def extend(self, records: Iterable[CycleMetrics]) -> None:
        """批量记录（按顺序）。"""
        for record in records:
            self.record(record)

    def close(self) -> None:
        """收尾（幂等）。

        当前实现每轮独立追加一行（进程崩溃也不会丢已写入的内容），因此无需关闭文件句柄；
        保留本方法是为了让调用方的生命周期代码统一，并为将来改为持久句柄留出位置。
        """

    def _append_line(self, metrics: CycleMetrics) -> None:
        """把一轮指标追加为一行 JSON（失败只计数，不抛异常）。"""
        try:
            path = self._path
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            self._write_errors += 1
            logger.warning("运行指标落盘失败（第 %d 次，不影响主循环）: %s", self._write_errors, exc)

    def __enter__(self) -> RuntimeMetricsRecorder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
