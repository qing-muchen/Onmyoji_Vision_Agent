"""运行时诊断层（Phase 5.3）：把"不知道"和"坏了"分开。

Runtime 有两种截然不同的"没有结论"：

| 现象 | 真实含义 | 处理方向 |
| --- | --- | --- |
| `RuntimeHealth.processor_ok=False` / `last_error` 非空 | **runtime failure**：采集 / 门控 / 判定链路出了问题 | 查窗口、采集后端、detector 异常 |
| `state=unknown` + `unknown_reason=no_evidence` | **vision unknown**：链路正常，但画面里没有已知证据 | 数据侧（采集样本 / 新增证据） |
| `state=unknown` + `unknown_reason=unsupported_scene` | 有部分证据命中但无规则满足（界面不在覆盖范围） | 规则 / 数据侧 |
| `state=unknown` + `unknown_reason=not_foreground` | 本轮根本没有可用画面（前台门控跳过） | 运行环境（把窗口置于最前） |

本模块提供：

* :class:`UnknownReason` —— `unknown` 的细分类别（**不修改 `StateResult` 契约**，
  只在 Runtime 侧记录为诊断字段）；
* :class:`CaptureOutcome` / :class:`ForegroundOutcome` —— 采集与前台门控的本轮结果；
* :class:`CycleDiagnosis` —— 单轮诊断三元组（供运行指标落盘）；
* :class:`RuntimeDiagnostics` —— 逐轮累积的运行健康状态；
* :class:`RuntimeHealth` —— 健康快照（`capture_ok` / `foreground_ok` / `processor_ok` /
  `last_error` / `last_state` / `cycle_latency` / `unknown_reason`）。

依赖方向：本模块只依赖 ``StateResult``（只读其既有字段）与 ``scheduler.CycleSchedule`` 的
数值，不采集、不判定、不做时间滤波、不写文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.vision.state.models import StateName, StateResult, StateSource

from src.runtime.state_monitor.foreground_guard import SkipReason

__all__ = [
    "CaptureOutcome",
    "CycleDiagnosis",
    "ForegroundOutcome",
    "RuntimeDiagnostics",
    "RuntimeHealth",
    "UnknownReason",
    "resolve_unknown_reason",
]


class UnknownReason(StrEnum):
    """``unknown`` 的细分原因（Runtime 侧诊断，不改变 ``StateResult`` 契约）。"""

    #: 链路正常、证据全未命中：画面里没有系统认识的元素
    NO_EVIDENCE = "no_evidence"
    #: 有部分证据命中但没有任何规则满足：界面不在数据集覆盖范围内
    UNSUPPORTED_SCENE = "unsupported_scene"
    #: 判定过程失败（detector / OCR / 模板异常），该帧降级为 ``unknown``
    DETECTOR_FAILURE = "detector_failure"
    #: 本轮因目标窗口不在前台被跳过（没有可用画面）
    NOT_FOREGROUND = "not_foreground"
    #: 其它采集跳过（窗口不存在 / 最小化 / 后端不可用 / 抓帧失败 / 无帧）
    SKIPPED = "skipped"


class CaptureOutcome(StrEnum):
    """本轮取帧结果。"""

    OK = "ok"
    SKIPPED = "skipped"


class ForegroundOutcome(StrEnum):
    """本轮前台门控结果。"""

    #: 门控通过（目标窗口在前台）
    OK = "ok"
    #: 目标窗口不在前台，本轮被跳过
    BLOCKED = "blocked"
    #: 未启用门控或本轮未做检查
    NOT_CHECKED = "n/a"


def resolve_unknown_reason(
    result: StateResult | None,
    *,
    judge_failed: bool = False,
    skip_reason: str | None = None,
    evidence_hits: int | None = None,
) -> UnknownReason | None:
    """推导本轮的 ``unknown`` 原因（链路正常时为 ``None``）。

    规则（顺序即优先级）：

    1. ``skip_reason`` 存在 → 本轮没有画面：``not_foreground`` 或其它跳过（``skipped``）；
    2. ``judge_failed`` → :data:`UnknownReason.DETECTOR_FAILURE`（判定过程异常）；
    3. 没有结果或结果**不是** ``unknown`` → ``None``（有明确状态，无需诊断）；
    4. ``unknown`` + ``StateSource.FALLBACK`` → :data:`UnknownReason.DETECTOR_FAILURE`；
    5. ``unknown`` + **有证据命中**（``evidence_hits > 0``，或结果自带证据）→
       :data:`UnknownReason.UNSUPPORTED_SCENE`：界面里有系统认识的元素，但没有规则满足
       （通常意味着界面不在数据集覆盖范围内）；
    6. 其余 → :data:`UnknownReason.NO_EVIDENCE`：证据全未命中，链路本身正常。

    Args:
        result: 本轮原始判定结果；未取到帧时为 ``None``。
        judge_failed: 判定过程是否失败（Runtime 捕获异常并降级为 ``unknown``）。
        skip_reason: 帧来源报告的跳过原因（:class:`SkipReason` 的字符串值）。
        evidence_hits: 本轮命中的证据条数（由 Runtime 在同一次检测中统计）；
            ``None`` 表示不可用（例如帧路径 / 外部注入的分类器），此时回退到
            ``result.evidence`` 是否为空来判断。

    Returns:
        细分原因；链路正常且状态明确时为 ``None``。
    """
    if skip_reason is not None:
        if skip_reason == SkipReason.NOT_FOREGROUND.value:
            return UnknownReason.NOT_FOREGROUND
        return UnknownReason.SKIPPED
    if judge_failed:
        return UnknownReason.DETECTOR_FAILURE
    if result is None or result.state_name is not StateName.UNKNOWN:
        return None
    if result.source is StateSource.FALLBACK:
        return UnknownReason.DETECTOR_FAILURE
    if evidence_hits is None:
        has_evidence = bool(result.evidence)
    else:
        has_evidence = evidence_hits > 0
    if has_evidence:
        return UnknownReason.UNSUPPORTED_SCENE
    return UnknownReason.NO_EVIDENCE


@dataclass(frozen=True, slots=True)
class CycleDiagnosis:
    """单轮诊断三元组（写入运行指标，见 :class:`~src.runtime.state_monitor.metrics.CycleMetrics`）。

    Attributes:
        capture_status: 取帧结果（``ok`` / ``skipped``）。
        foreground_status: 前台门控结果（``ok`` / ``blocked`` / ``n/a``）。
        unknown_reason: ``unknown`` 的细分原因；状态明确时为 ``None``。
    """

    capture_status: CaptureOutcome
    foreground_status: ForegroundOutcome
    unknown_reason: UnknownReason | None

    def describe(self) -> str:
        """返回可读描述。"""
        reason = self.unknown_reason.value if self.unknown_reason is not None else "-"
        return (
            f"capture={self.capture_status.value} "
            f"foreground={self.foreground_status.value} unknown_due_to={reason}"
        )


@dataclass(frozen=True, slots=True)
class RuntimeHealth:
    """运行时健康快照：区分 **runtime failure** 与 **vision unknown**。

    Attributes:
        capture_ok: 最近一轮是否取到帧。
        foreground_ok: 最近一轮前台门控是否通过（未启用门控时为 ``False`` 但见 ``foreground_status``）。
        processor_ok: 最近一轮判定是否正常完成（未因异常降级）。
        last_error: 最近一次失败说明（判定异常 / 采集跳过消息）；无失败时为 ``None``。
        last_state: 最近一轮的**原始**判定状态。
        cycle_latency_ms: 最近一轮端到端耗时（毫秒）。
        unknown_reason: 最近一轮 ``unknown`` 的细分原因。
        cycles / frames / skipped / errors: 累计轮次、成功取帧、跳过、判定失败次数。
        skip_counts: 按 :class:`SkipReason` 统计的跳过次数快照。
        backend: 最近使用的采集后端名称。
    """

    capture_ok: bool
    foreground_ok: bool
    processor_ok: bool
    last_error: str | None
    last_state: StateName | None
    cycle_latency_ms: float | None
    unknown_reason: UnknownReason | None
    cycles: int
    frames: int
    skipped: int
    errors: int
    skip_counts: dict[str, int]
    backend: str | None

    @property
    def is_runtime_failure(self) -> bool:
        """是否存在运行期故障（判定失败 / 采集失败，而非"画面里没有已知元素"）。"""
        if not self.processor_ok:
            return True
        return self.unknown_reason in {
            UnknownReason.DETECTOR_FAILURE,
            UnknownReason.SKIPPED,
        }

    @property
    def is_vision_unknown(self) -> bool:
        """是否属于"链路正常但认不出画面"（数据侧问题，不是 runtime 故障）。"""
        return self.unknown_reason in {
            UnknownReason.NO_EVIDENCE,
            UnknownReason.UNSUPPORTED_SCENE,
            UnknownReason.NOT_FOREGROUND,
        }

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（**不含截图、不含用户数据**）。"""
        return {
            "capture_ok": self.capture_ok,
            "foreground_ok": self.foreground_ok,
            "processor_ok": self.processor_ok,
            "last_error": self.last_error,
            "last_state": self.last_state.value if self.last_state is not None else None,
            "cycle_latency_ms": self.cycle_latency_ms,
            "unknown_reason": self.unknown_reason.value if self.unknown_reason else None,
            "cycles": self.cycles,
            "frames": self.frames,
            "skipped": self.skipped,
            "errors": self.errors,
            "skip_counts": dict(self.skip_counts),
            "backend": self.backend,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        state = self.last_state.value if self.last_state is not None else "-"
        latency = f"{self.cycle_latency_ms:.1f}ms" if self.cycle_latency_ms is not None else "n/a"
        reason = self.unknown_reason.value if self.unknown_reason else "-"
        return (
            f"cycles={self.cycles} frames={self.frames} skipped={self.skipped} errors={self.errors} "
            f"capture_ok={self.capture_ok} foreground_ok={self.foreground_ok} "
            f"processor_ok={self.processor_ok} state={state} latency={latency} "
            f"unknown_due_to={reason} backend={self.backend or 'n/a'}"
        )


class RuntimeDiagnostics:
    """逐轮收集运行健康状态（**观测**，不参与任何判定）。

    由 :class:`~src.runtime.state_monitor.monitor.StateMonitor` 每轮调用
    :meth:`observe` 更新；:attr:`health` 返回当前快照。

    典型用法::

        diagnostics = RuntimeDiagnostics()
        diagnosis = diagnostics.observe(frame_ok=True, skip_reason=None, result=result, ...)
        print(diagnostics.health.describe())
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """清空全部累积状态（用于重新开始一次运行）。"""
        self._cycles = 0
        self._frames = 0
        self._skipped = 0
        self._errors = 0
        self._capture_ok = False
        self._foreground_ok = False
        self._processor_ok = True
        self._last_error: str | None = None
        self._last_state: StateName | None = None
        self._last_unknown_reason: UnknownReason | None = None
        self._latency_ms: float | None = None
        self._backend: str | None = None
        self._skip_counts: dict[str, int] = {}

    def observe(
        self,
        *,
        frame_ok: bool,
        result: StateResult | None,
        judge_failed: bool = False,
        skip_reason: str | None = None,
        skip_message: str | None = None,
        foreground_checked: bool = False,
        foreground_passed: bool = False,
        latency_ms: float | None = None,
        backend: str | None = None,
        evidence_hits: int | None = None,
    ) -> CycleDiagnosis:
        """记录一轮观测并返回本轮诊断。

        Args:
            frame_ok: 本轮是否取到帧。
            result: 本轮原始判定结果；未取到帧时为 ``None``。
            judge_failed: 判定是否因异常降级为 ``unknown``。
            skip_reason: 帧来源报告的跳过原因。
            skip_message: 跳过的可读说明（作为 ``last_error`` 的可选补充）。
            foreground_checked: 本轮是否做了前台门控检查。
            foreground_passed: 前台门控是否通过。
            latency_ms: 本轮端到端耗时（毫秒）。
            backend: 本轮采集后端名称。
            evidence_hits: 本轮命中的证据条数（``None`` = 不可用）。

        Returns:
            本轮诊断（采集 / 前台 / ``unknown`` 原因三元组）。
        """
        self._cycles += 1
        self._capture_ok = frame_ok
        if frame_ok:
            self._frames += 1
        else:
            self._skipped += 1
            if skip_reason:
                self._skip_counts[skip_reason] = self._skip_counts.get(skip_reason, 0) + 1
        if judge_failed:
            self._errors += 1
        self._processor_ok = not judge_failed
        self._latency_ms = latency_ms
        if backend:
            self._backend = backend
        if result is not None:
            self._last_state = result.state_name

        unknown_reason = resolve_unknown_reason(
            result,
            judge_failed=judge_failed,
            skip_reason=skip_reason,
            evidence_hits=evidence_hits,
        )
        self._last_unknown_reason = unknown_reason

        if judge_failed:
            detail = result.reason if result is not None and result.reason else "判定失败"
            self._last_error = detail
        elif not frame_ok:
            self._last_error = skip_message or skip_reason or "本轮无可用帧"
        else:
            self._last_error = None

        if not foreground_checked:
            foreground_status = ForegroundOutcome.NOT_CHECKED
        elif foreground_passed:
            foreground_status = ForegroundOutcome.OK
            self._foreground_ok = True
        else:
            foreground_status = ForegroundOutcome.BLOCKED
            self._foreground_ok = False

        return CycleDiagnosis(
            capture_status=CaptureOutcome.OK if frame_ok else CaptureOutcome.SKIPPED,
            foreground_status=foreground_status,
            unknown_reason=unknown_reason,
        )

    @property
    def health(self) -> RuntimeHealth:
        """当前健康快照。"""
        return RuntimeHealth(
            capture_ok=self._capture_ok,
            foreground_ok=self._foreground_ok,
            processor_ok=self._processor_ok,
            last_error=self._last_error,
            last_state=self._last_state,
            cycle_latency_ms=self._latency_ms,
            unknown_reason=self._last_unknown_reason,
            cycles=self._cycles,
            frames=self._frames,
            skipped=self._skipped,
            errors=self._errors,
            skip_counts=dict(self._skip_counts),
            backend=self._backend,
        )

    def describe(self) -> str:
        """返回可读描述（等价于 :attr:`health` 的描述）。"""
        return self.health.describe()
