"""Task 层数据模型（Phase 6.1）。

本模块定义 Task Engine 的**声明式定义**与**运行期视图**，并严格遵守 Phase 6.0 冻结的边界
（架构决策 D-1 ~ D-6）：

* **零上层依赖**：不 `import cv2` / `src.vision.*` / `src.runtime.*` / `src.action.*`，
  只依赖标准库与 `src.core` / `src.utils`。因此 Task 层可以被完全离线驱动与测试；
* **只消费状态数据的字段**：:class:`StateSnapshot` 从 `StateResult` / `StableStateResult`
  等对象**按字段提取**（duck typing），而不是导入视觉层类型——这样"消费 StateResult"
  与"不依赖 vision"同时成立；
* **语义目标而非坐标**（D-3）：:class:`Step` 的 ``action`` 用语义定位器（元素名）或
  参考坐标表达，**禁止屏幕绝对坐标**；
* **生命周期**（用户给定）：``CREATED → READY → RUNNING → PAUSED → BLOCKED →
  COMPLETED / FAILED``，转换规则见 :mod:`src.task.state_machine`。

时间戳格式与 Runtime 的 ``local_now_iso`` 保持一致（ISO 8601 含时区），但本层**独立实现**
（避免跨层依赖）。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from time import monotonic
from typing import Any

from src.task.state_machine import (
    TaskError,
    TaskState,
    TaskStateMachine,
    local_now_iso,
)

__all__ = [
    "Budget",
    "RuntimeSnapshot",
    "StateSnapshot",
    "Step",
    "Task",
    "TaskContext",
    "TaskDefinitionError",
    "TaskPolicies",
    "TaskResult",
    "TaskState",
    "TaskStatus",
    "local_now_iso",
]

#: 未知 / 无证据类原因（Runtime 侧 ``UnknownReason`` 的取值，Task 只做字符串比较）
VISION_UNKNOWN_REASONS: frozenset[str] = frozenset({"no_evidence", "unsupported_scene"})
#: 运行时故障类原因（判定异常 / 采集跳过）
RUNTIME_FAILURE_REASONS: frozenset[str] = frozenset({"detector_failure", "skipped"})
#: 采集阻塞类原因（没有可用画面：失焦 / 窗口问题）
CAPTURE_BLOCKED_REASONS: frozenset[str] = frozenset({"not_foreground"})
#: 允许的动作类型（本阶段只校验，不执行——执行属 Phase 7）
ACTION_KINDS: frozenset[str] = frozenset(
    {"click", "double_click", "long_press", "drag", "key", "wait"}
)
#: 允许的策略取值
UNKNOWN_POLICIES: frozenset[str] = frozenset({"wait", "fail"})
RUNTIME_FAILURE_POLICIES: frozenset[str] = frozenset({"block", "fail"})
MISMATCH_POLICIES: frozenset[str] = frozenset({"retry", "fail"})
TIMEOUT_POLICIES: frozenset[str] = frozenset({"retry", "fail", "wait"})


class TaskDefinitionError(TaskError):
    """任务定义非法（空步骤、未知动作类型、策略取值非法等）。"""


class TaskStatus(StrEnum):
    """任务的对外结论（由生命周期状态派生，便于日志与上层判断）。"""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def from_state(cls, state: TaskState) -> TaskStatus:
        """把生命周期状态映射为对外结论。

        Args:
            state: 任务生命周期状态。

        Returns:
            ``CREATED`` → ``PENDING``；``READY`` / ``RUNNING`` / ``PAUSED`` / ``BLOCKED``
            → ``ACTIVE``；``COMPLETED`` → ``COMPLETED``；``FAILED`` → ``FAILED``。
        """
        if state is TaskState.CREATED:
            return cls.PENDING
        if state is TaskState.COMPLETED:
            return cls.COMPLETED
        if state is TaskState.FAILED:
            return cls.FAILED
        return cls.ACTIVE

    @property
    def is_finished(self) -> bool:
        """任务是否已结束（完成或失败）。"""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED)


def _text(value: Any) -> str | None:
    """把枚举 / 字符串取值统一成字符串（``None`` 原样返回）。"""
    if value is None:
        return None
    inner = getattr(value, "value", value)
    return inner if isinstance(inner, str) else str(inner)


def _pick(source: Any, *names: str, default: Any = None) -> Any:
    """按名称从对象或映射中取值（找不到时返回 ``default``）。

    Args:
        source: 任意对象或映射（``StateResult`` / ``StableStateResult`` / ``dict`` …）。
        *names: 候选字段名（按顺序取第一个存在者）。
        default: 全部缺失时的返回值。

    Returns:
        取到的值；都不存在时为 ``default``。
    """
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Task 侧的**状态视图**（从 ``StateResult`` / ``StableStateResult`` 提取字段）。

    本类型是 Task 与视觉层之间的唯一数据通道：Task 不导入视觉层类型，只消费这些字段
    （架构决策 D-4）。

    Attributes:
        state: 状态名（``StateName`` 的取值字符串，例如 ``battle`` / ``unknown``）。
        confidence: 置信度（``[0, 1]``）。
        source: 判定来源字符串（例如 ``evidence`` / ``fallback``）。
        reason: 判定原因（人类可读，来自 ``StateResult.reason``）。
        evidence: 参与判定的证据名（仅用于审计，**不做决策**，见 D-4）。
        stable: 是否来自稳定化结果：``True`` = ``StableStateResult``，``False`` = 原始 ``StateResult``。
        support_frames: 稳定化连续帧数（原始结果时为 ``0``）。
        required_frames: 稳定化所需帧数（原始结果时为 ``0``）。
        counted: 该帧是否计入稳定统计（低置信度帧为 ``False``）。
        raw_state: 稳定化前的原始状态（仅稳定化结果携带）。
        previous_state: 稳定化结果中"上一个稳定状态"。
        changed: 该帧是否改变了稳定状态。
        timestamp: 事件 / 采样时间（ISO 8601）。
    """

    state: str
    confidence: float = 0.0
    source: str = ""
    reason: str = ""
    evidence: tuple[str, ...] = ()
    stable: bool = False
    support_frames: int = 0
    required_frames: int = 0
    counted: bool = True
    raw_state: str | None = None
    previous_state: str | None = None
    changed: bool = False
    timestamp: str = ""

    @property
    def is_unknown(self) -> bool:
        """状态是否为 ``unknown``。"""
        return self.state == "unknown"

    @property
    def effective_state(self) -> str:
        """用于判定的状态：稳定化未确认时回退到原始状态。"""
        if self.stable and self.raw_state is not None and self.state == "unknown":
            return self.raw_state
        return self.state

    @classmethod
    def from_object(cls, value: Any, *, timestamp: str = "") -> StateSnapshot:
        """从任意状态对象 / 映射 / 字符串构造快照（duck typing）。

        支持：

        * :class:`StateSnapshot` —— 原样返回；
        * ``str`` —— 视为状态名；
        * ``Mapping`` —— 读取 ``state`` / ``state_name`` 等键；
        * 其它对象（例如 ``StateResult`` / ``StableStateResult``）—— 按同名字段读取。

        Args:
            value: 待提取的状态来源。
            timestamp: 缺省时间戳（来源未携带时使用）。

        Returns:
            状态快照。

        Raises:
            TaskDefinitionError: 无法从 ``value`` 提取状态名。
        """
        if isinstance(value, StateSnapshot):
            return value

        if isinstance(value, str):
            return cls(state=value, timestamp=timestamp)

        state = _text(_pick(value, "state", "state_name"))
        if state is None:
            # 稳定化结果可能在未确认时 state_name 为 None：尝试原始状态
            state = _text(_pick(value, "raw_state"))
        if state is None:
            raise TaskDefinitionError(f"无法从 {type(value).__name__} 提取状态名")

        evidence = _pick(value, "evidence") or ()
        if isinstance(evidence, str):
            evidence = (evidence,)

        return cls(
            state=state,
            confidence=float(_pick(value, "confidence") or 0.0),
            source=_text(_pick(value, "source")) or "",
            reason=str(_pick(value, "reason") or ""),
            evidence=tuple(str(item) for item in evidence),
            stable=bool(_pick(value, "stable") or _is_stable_result(value)),
            support_frames=int(_pick(value, "support_frames") or 0),
            required_frames=int(_pick(value, "required_frames") or 0),
            counted=bool(_pick(value, "counted", default=True)),
            raw_state=_text(_pick(value, "raw_state")),
            previous_state=_text(_pick(value, "previous_state")),
            changed=bool(_pick(value, "changed") or False),
            timestamp=str(_pick(value, "timestamp") or timestamp),
        )

    def describe(self) -> str:
        """返回可读描述。"""
        kind = "stable" if self.stable else "raw"
        support = f" support={self.support_frames}/{self.required_frames}" if self.stable else ""
        return (
            f"{self.state}({self.confidence:.2f}, {kind})"
            f"{support}{'' if self.counted else ' [uncounted]'}"
        )


def _is_stable_result(value: Any) -> bool:
    """判断对象是否"看起来像"稳定化结果（有 ``is_confirmed`` / ``support_frames`` 字段）。"""
    return hasattr(value, "is_confirmed") or hasattr(value, "support_frames")


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Task 侧的**运行时视图**（从 ``RuntimeHealth`` / ``StateMonitor`` 提取字段）。

    Attributes:
        state: 运行时生命周期状态字符串（``RuntimeState`` 的取值，例如 ``running``）。
        capture_ok: 最近一轮是否取到帧。
        foreground_ok: 前台门控是否通过。
        processor_ok: 判定链路是否正常（``False`` = 判定失败降级）。
        unknown_reason: ``unknown`` 的细分原因（``no_evidence`` / ``not_foreground`` …）。
        last_error: 最近一次失败说明。
        cycles / skipped / errors: 运行时计数。
    """

    state: str | None = None
    capture_ok: bool | None = None
    foreground_ok: bool | None = None
    processor_ok: bool | None = None
    unknown_reason: str | None = None
    last_error: str | None = None
    cycles: int = 0
    skipped: int = 0
    errors: int = 0

    @property
    def is_known(self) -> bool:
        """是否携带任何运行时信息（未知快照返回 ``False``）。"""
        return self.state is not None or self.processor_ok is not None

    @property
    def is_runtime_failure(self) -> bool:
        """是否为**运行时故障**（判定异常 / 采集跳过类），对应架构决策 D-5。"""
        if self.processor_ok is False:
            return True
        return self.unknown_reason in RUNTIME_FAILURE_REASONS

    @property
    def is_capture_blocked(self) -> bool:
        """是否"没有可用画面"（失焦 / 采集未取到帧 / 正在恢复 / 已失败 / 已停止）。"""
        if self.unknown_reason in CAPTURE_BLOCKED_REASONS:
            return True
        if self.state in {"recovering", "failed", "stopped"}:
            return True
        return self.capture_ok is False and self.state not in {None, "ready", "running"}

    @property
    def is_vision_unknown(self) -> bool:
        """是否"链路正常但认不出画面"（数据侧问题，任务应等待而非失败）。"""
        return self.unknown_reason in VISION_UNKNOWN_REASONS

    @property
    def is_blocking(self) -> bool:
        """是否应让任务进入 ``BLOCKED``（D-5：不消耗任务预算）。"""
        return self.is_runtime_failure or self.is_capture_blocked

    @property
    def is_terminal(self) -> bool:
        """运行时是否已处于终止态（``failed`` / ``stopped``）。"""
        return self.state in {"failed", "stopped"}

    @classmethod
    def from_object(cls, value: Any) -> RuntimeSnapshot:
        """从任意运行时对象 / 映射构造快照（duck typing）。

        支持 :class:`RuntimeSnapshot`（原样）、``Mapping``、以及具备
        ``runtime_state`` / ``health`` / ``capture_ok`` 等字段的对象
        （例如 ``StateMonitor`` 或 ``RuntimeHealth``）。

        Args:
            value: 待提取的来源；``None`` 表示"无运行时信息"。

        Returns:
            运行时快照。
        """
        if isinstance(value, RuntimeSnapshot):
            return value
        if value is None:
            return cls()

        # StateMonitor：优先读取其 health 快照，再补充 lifecycle 状态与计数
        health = _pick(value, "health")
        source = health if health is not None else value
        state = _text(_pick(value, "runtime_state")) or _text(_pick(source, "runtime_state"))
        return cls(
            state=state,
            capture_ok=_pick_bool(source, "capture_ok"),
            foreground_ok=_pick_bool(source, "foreground_ok"),
            processor_ok=_pick_bool(source, "processor_ok"),
            unknown_reason=_text(_pick(source, "unknown_reason")),
            last_error=_pick_text(source, "last_error"),
            cycles=int(_pick(source, "cycles") or 0),
            skipped=int(_pick(source, "skipped") or 0),
            errors=int(_pick(source, "errors") or 0),
        )

    def describe(self) -> str:
        """返回可读描述。"""
        if not self.is_known:
            return "runtime=unknown"
        return (
            f"runtime={self.state or 'n/a'} capture_ok={self.capture_ok} "
            f"foreground_ok={self.foreground_ok} processor_ok={self.processor_ok} "
            f"unknown_due_to={self.unknown_reason or '-'} "
            f"cycles={self.cycles} skipped={self.skipped} errors={self.errors}"
        )


def _pick_bool(source: Any, name: str) -> bool | None:
    """读取布尔字段（缺失时返回 ``None``）。"""
    value = _pick(source, name)
    return None if value is None else bool(value)


def _pick_text(source: Any, name: str) -> str | None:
    """读取文本字段（空串归一为 ``None``）。"""
    value = _pick(source, name)
    if value is None:
        return None
    text = str(value)
    return text or None


@dataclass(frozen=True, slots=True)
class Budget:
    """任务级预算（硬上限，防止任务失控）。

    Attributes:
        max_steps: 单任务最多执行的步骤数（含重试派发）。
        deadline_s: 单任务墙钟上限（秒）。
        max_step_retries: 单步默认重试次数（步骤可覆盖）。
        min_stable_frames: 达成判定需要的连续满足帧数（默认 ``1`` = 信任 Runtime 的稳定化）。
    """

    max_steps: int = 20
    deadline_s: float = 300.0
    max_step_retries: int = 1
    min_stable_frames: int = 1

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise TaskDefinitionError(f"budget.max_steps 必须为正数: {self.max_steps}")
        if self.deadline_s <= 0:
            raise TaskDefinitionError(f"budget.deadline_s 必须为正数: {self.deadline_s}")
        if self.max_step_retries < 0:
            raise TaskDefinitionError(
                f"budget.max_step_retries 不能为负数: {self.max_step_retries}"
            )
        if self.min_stable_frames < 1:
            raise TaskDefinitionError(
                f"budget.min_stable_frames 至少为 1: {self.min_stable_frames}"
            )

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"Budget(max_steps={self.max_steps}, deadline={self.deadline_s}s, "
            f"max_retries={self.max_step_retries}, min_stable_frames={self.min_stable_frames})"
        )


@dataclass(frozen=True, slots=True)
class TaskPolicies:
    """任务级策略（架构决策 D-5 与 Phase 6.0 的失败分类）。

    Attributes:
        on_unknown: ``unknown``（``no_evidence`` / ``unsupported_scene``）时的行为：
            ``wait``（默认，不动作、不消耗重试）或 ``fail``。
        on_runtime_failure: 运行时故障时的行为：``block``（默认，进入 ``BLOCKED``，
            不消耗预算）或 ``fail``。
        on_mismatch: 观测到与期望冲突的状态（``forbid`` 命中）时：``retry``（默认）或 ``fail``。
        on_step_timeout: 步骤验证窗口超时：``retry``（默认）/ ``fail`` / ``wait``。
    """

    on_unknown: str = "wait"
    on_runtime_failure: str = "block"
    on_mismatch: str = "retry"
    on_step_timeout: str = "retry"

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("on_unknown", self.on_unknown, UNKNOWN_POLICIES),
            ("on_runtime_failure", self.on_runtime_failure, RUNTIME_FAILURE_POLICIES),
            ("on_mismatch", self.on_mismatch, MISMATCH_POLICIES),
            ("on_step_timeout", self.on_step_timeout, TIMEOUT_POLICIES),
        ):
            if value not in allowed:
                raise TaskDefinitionError(
                    f"policies.{name} 非法: {value!r}（可选: {', '.join(sorted(allowed))}）"
                )

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"TaskPolicies(unknown={self.on_unknown}, runtime_failure={self.on_runtime_failure}, "
            f"mismatch={self.on_mismatch}, step_timeout={self.on_step_timeout})"
        )


@dataclass(frozen=True, slots=True)
class Step:
    """任务步骤的声明式定义（**不含执行逻辑**）。

    Attributes:
        step_id: 步骤标识（在任务内唯一）。
        description: 人类可读说明。
        expect: 达成判定（谓词 spec：``dict`` / ``str`` / 可调用对象；``None`` = 立即达成）。
        forbid: 冲突判定（谓词 spec；命中即 ``VIOLATED``，默认 ``None``）。
        action: 语义动作请求（``Mapping``，见 :attr:`action_kind` / :attr:`action_target`）；
            ``None`` 表示"只等待状态达成、不请求动作"。
        timeout_s: 验证窗口（秒）；超时按策略重试或失败。
        retries: 本步骤重试上限；``None`` = 使用 :class:`Budget.max_step_retries`。
        min_stable_frames: 达成所需连续满足帧数；``None`` = 使用预算值。
    """

    step_id: str
    description: str = ""
    expect: Any | None = None
    forbid: Any | None = None
    action: Mapping[str, Any] | None = None
    timeout_s: float = 10.0
    retries: int | None = None
    min_stable_frames: int | None = None

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise TaskDefinitionError("step.step_id 不能为空")
        if self.timeout_s <= 0:
            raise TaskDefinitionError(
                f"step[{self.step_id}].timeout_s 必须为正数: {self.timeout_s}"
            )
        if self.retries is not None and self.retries < 0:
            raise TaskDefinitionError(
                f"step[{self.step_id}].retries 不能为负数: {self.retries}"
            )
        if self.min_stable_frames is not None and self.min_stable_frames < 1:
            raise TaskDefinitionError(
                f"step[{self.step_id}].min_stable_frames 至少为 1: {self.min_stable_frames}"
            )
        if self.action is not None:
            kind = str(self.action.get("kind", "")).strip()
            if kind not in ACTION_KINDS:
                raise TaskDefinitionError(
                    f"step[{self.step_id}].action.kind 非法: {kind!r}"
                    f"（可选: {', '.join(sorted(ACTION_KINDS))}）"
                )
            if kind != "wait" and not self.action.get("target"):
                raise TaskDefinitionError(
                    f"step[{self.step_id}].action 需要 target（语义元素或参考坐标，D-3）"
                )

    @property
    def action_kind(self) -> str | None:
        """语义动作类型（``None`` 表示本步骤不请求动作）。"""
        if self.action is None:
            return None
        return str(self.action.get("kind", "")) or None

    @property
    def action_target(self) -> Mapping[str, Any] | None:
        """语义目标（``element`` / ``reference``；见 D-3）。"""
        if self.action is None:
            return None
        target = self.action.get("target")
        return target if isinstance(target, Mapping) else None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, index: int = 0) -> Step:
        """从映射（YAML 友好）构造步骤。

        Args:
            data: 步骤定义，键与 :class:`Step` 字段对应；``id`` / ``name`` 等价于 ``step_id``。
            index: 缺省 ``step_id`` 序号（形如 ``step_1``）。

        Returns:
            步骤实例。

        Raises:
            TaskDefinitionError: 缺少标识或字段非法（由 ``__post_init__`` 校验）。
        """
        step_id = str(data.get("step_id") or data.get("id") or data.get("name") or "").strip()
        if not step_id:
            step_id = f"step_{index + 1}"
        action = data.get("action")
        return cls(
            step_id=step_id,
            description=str(data.get("description", "") or ""),
            expect=data.get("expect"),
            forbid=data.get("forbid"),
            action=action if isinstance(action, Mapping) else None,
            timeout_s=float(data.get("timeout_s", 10.0)),
            retries=data.get("retries"),
            min_stable_frames=data.get("min_stable_frames"),
        )

    def describe(self) -> str:
        """返回可读描述。"""
        parts = [f"step={self.step_id}"]
        if self.action_kind:
            parts.append(f"action={self.action_kind}")
        if self.expect is not None:
            parts.append("expect=set")
        if self.forbid is not None:
            parts.append("forbid=set")
        parts.append(f"timeout={self.timeout_s:g}s")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class Task:
    """任务的声明式定义（YAML 流程的承载对象，架构决策 D-6）。

    典型用法::

        task = Task.from_mapping(
            {
                "id": "demo.wait_for_yard",
                "description": "等待庭院稳定（不请求任何动作）",
                "steps": [{"id": "wait_yard", "expect": {"state_is": "yard"}, "timeout_s": 30}],
            }
        )

    Attributes:
        task_id: 任务标识。
        description: 人类可读说明。
        steps: 步骤序列（按顺序执行，全部达成即 ``COMPLETED``）。
        budget: 预算（步数 / 时长 / 重试 / 稳定帧）。
        policies: 策略（unknown / runtime failure / mismatch / timeout）。
        metadata: 附加信息（例如来源文件；不参与判定）。
    """

    task_id: str
    description: str = ""
    steps: tuple[Step, ...] = ()
    budget: Budget = field(default_factory=Budget)
    policies: TaskPolicies = field(default_factory=TaskPolicies)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise TaskDefinitionError("task.task_id 不能为空")
        if not self.steps:
            raise TaskDefinitionError(f"task[{self.task_id}] 至少需要一个步骤")
        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise TaskDefinitionError(f"task[{self.task_id}] 步骤标识重复: {step.step_id}")
            seen.add(step.step_id)

    @property
    def step_count(self) -> int:
        """步骤数量。"""
        return len(self.steps)

    @property
    def step_ids(self) -> tuple[str, ...]:
        """步骤标识序列。"""
        return tuple(step.step_id for step in self.steps)

    def step_at(self, index: int) -> Step | None:
        """按序号取步骤（越界返回 ``None``）。"""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Task:
        """从映射（YAML 友好）构造任务。

        Args:
            data: 任务定义；键与 :class:`Task` 字段对应（``id`` 等价于 ``task_id``）。

        Returns:
            任务实例。

        Raises:
            TaskDefinitionError: 定义非法（缺标识 / 无步骤 / 字段取值非法）。
        """
        task_id = str(data.get("task_id") or data.get("id") or "").strip()
        raw_steps = data.get("steps") or ()
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
            raise TaskDefinitionError(f"task[{task_id}].steps 必须是步骤列表")
        steps = tuple(
            Step.from_mapping(item, index=index)
            for index, item in enumerate(raw_steps)
            if isinstance(item, Mapping)
        )
        budget_data = data.get("budget") or {}
        policies_data = data.get("policies") or {}
        if not isinstance(budget_data, Mapping) or not isinstance(policies_data, Mapping):
            raise TaskDefinitionError(f"task[{task_id}] 的 budget / policies 必须是映射")
        return cls(
            task_id=task_id,
            description=str(data.get("description", "") or ""),
            steps=steps,
            budget=Budget(**dict(budget_data)),
            policies=TaskPolicies(**dict(policies_data)),
            metadata=dict(data.get("metadata") or {}),
        )

    def with_budget(self, **updates: Any) -> Task:
        """返回替换了部分预算字段的副本（便于测试与派生任务）。"""
        return replace(self, budget=replace(self.budget, **updates))

    def with_policies(self, **updates: Any) -> Task:
        """返回替换了部分策略字段的副本。"""
        return replace(self, policies=replace(self.policies, **updates))

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"Task(id={self.task_id}, steps={self.step_count} "
            f"[{', '.join(self.step_ids)}], {self.budget.describe()}, {self.policies.describe()})"
        )


@dataclass(frozen=True, slots=True)
class TaskResult:
    """任务结束时的结果快照（终态：``COMPLETED`` / ``FAILED``）。

    Attributes:
        task_id: 任务标识。
        state: 终态生命周期状态。
        status: 对外结论。
        reason: 结束原因（人类可读，例如 ``completed`` / ``step_timeout`` / ``stopped``）。
        steps_executed: 实际执行的步骤数（含重试派发）。
        steps_total: 任务定义的步骤总数。
        cycles: 任务期间经历的 Runtime 节拍数。
        elapsed_s: 任务墙钟时长（秒）。
        started_at / ended_at: ISO 8601 时间戳。
    """

    task_id: str
    state: TaskState
    status: TaskStatus
    reason: str = ""
    steps_executed: int = 0
    steps_total: int = 0
    cycles: int = 0
    elapsed_s: float = 0.0
    started_at: str = ""
    ended_at: str = ""

    @property
    def succeeded(self) -> bool:
        """任务是否成功完成。"""
        return self.status is TaskStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含截图 / 用户数据）。"""
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "status": self.status.value,
            "reason": self.reason,
            "steps_executed": self.steps_executed,
            "steps_total": self.steps_total,
            "cycles": self.cycles,
            "elapsed_s": round(self.elapsed_s, 3),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"task={self.task_id} status={self.status.value} state={self.state.value} "
            f"reason={self.reason or '-'} steps={self.steps_executed}/{self.steps_total} "
            f"cycles={self.cycles} elapsed={self.elapsed_s:.1f}s"
        )


class TaskContext:
    """任务运行期上下文（**可变**，由 :class:`~src.task.engine.TaskEngine` 独占写入）。

    保存用户要求的内容：

    * 当前任务状态（:attr:`state` / :attr:`machine`）；
    * 当前 ``StateResult`` 视图（:attr:`snapshot`）；
    * runtime 信息（:attr:`runtime`）；
    * elapsed time（:attr:`elapsed_s`）；
    * counters（:attr:`cycles` / :attr:`stable_ticks` / :attr:`steps_dispatched` / …）。

    外部只读访问；观测方法（``observe_*`` / ``mark_*``）仅供引擎调用。

    Args:
        task: 任务定义。
        machine: 任务状态机（引擎创建并注入）。
        clock: 单调时钟函数（用于 elapsed 计算；可在测试中注入）。
        started_at: 任务开始时间（ISO 8601）。
    """

    def __init__(
        self,
        task: Task,
        machine: TaskStateMachine,
        *,
        clock: Any = None,
        started_at: str = "",
    ) -> None:
        self._task = task
        self._machine = machine
        self._clock = clock if clock is not None else monotonic
        self._started_at = started_at or local_now_iso()
        self._started_monotonic = self._clock()

        self._snapshot: StateSnapshot | None = None
        self._last_event: StateSnapshot | None = None
        self._runtime = RuntimeSnapshot()

        self._cycles = 0
        self._last_cycle_id: int | None = None
        self._last_cycle_timestamp: str = ""
        self._stable_ticks = 0
        self._events = 0
        self._steps_dispatched = 0
        self._steps_satisfied = 0
        self._steps_retried = 0
        self._timeouts = 0
        self._actions_requested = 0
        self._blocks = 0
        self._unblocks = 0
        self._pauses = 0
        self._resumes = 0
        self._last_reason = ""
        self._step_index = 0

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def task(self) -> Task:
        """任务定义。"""
        return self._task

    @property
    def machine(self) -> TaskStateMachine:
        """任务状态机（含转换历史）。"""
        return self._machine

    @property
    def state(self) -> TaskState:
        """当前任务状态。"""
        return self._machine.state

    @property
    def status(self) -> TaskStatus:
        """当前对外结论。"""
        return TaskStatus.from_state(self._machine.state)

    @property
    def snapshot(self) -> StateSnapshot | None:
        """最近一次状态视图（``on_stable`` 提供）。"""
        return self._snapshot

    @property
    def last_event(self) -> StateSnapshot | None:
        """最近一次"稳定状态变化"视图（若来源携带 ``changed`` 信息）。"""
        return self._last_event

    @property
    def runtime(self) -> RuntimeSnapshot:
        """当前运行时视图。"""
        return self._runtime

    @property
    def started_at(self) -> str:
        """任务开始时间（ISO 8601）。"""
        return self._started_at

    def elapsed_s(self) -> float:
        """任务已运行时长（秒，基于注入的单调时钟）。"""
        return max(0.0, self._clock() - self._started_monotonic)

    @property
    def cycles(self) -> int:
        """经历的 Runtime 节拍数（``on_cycle`` 次数）。"""
        return self._cycles

    @property
    def last_cycle_id(self) -> int | None:
        """最近一次节拍序号（来自 ``CycleSchedule.cycle_id``，若有）。"""
        return self._last_cycle_id

    @property
    def stable_ticks(self) -> int:
        """收到的稳定状态帧数（``on_stable`` 次数）。"""
        return self._stable_ticks

    @property
    def events(self) -> int:
        """观测到的稳定状态变化次数。"""
        return self._events

    @property
    def step_index(self) -> int:
        """当前步骤序号（从 0 开始）。"""
        return self._step_index

    @property
    def current_step(self) -> Step | None:
        """当前步骤（任务已结束时为 ``None``）。"""
        return self._task.step_at(self._step_index)

    @property
    def steps_dispatched(self) -> int:
        """已派发的步骤次数（含重试）。"""
        return self._steps_dispatched

    @property
    def steps_satisfied(self) -> int:
        """已达成并推进的步骤数。"""
        return self._steps_satisfied

    @property
    def steps_retried(self) -> int:
        """步骤重试次数。"""
        return self._steps_retried

    @property
    def timeouts(self) -> int:
        """观测到的步骤超时次数。"""
        return self._timeouts

    @property
    def actions_requested(self) -> int:
        """向 Action 端口提出的动作请求数（本阶段不执行）。"""
        return self._actions_requested

    @property
    def blocks(self) -> int:
        """进入 ``BLOCKED`` 的次数。"""
        return self._blocks

    @property
    def unblocks(self) -> int:
        """解除 ``BLOCKED`` 的次数。"""
        return self._unblocks

    @property
    def pauses(self) -> int:
        """暂停次数。"""
        return self._pauses

    @property
    def resumes(self) -> int:
        """恢复次数。"""
        return self._resumes

    @property
    def last_reason(self) -> str:
        """最近一次状态转换原因。"""
        return self._last_reason

    # ---- 观测（引擎调用） ---------------------------------------------------

    def observe_cycle(self, *, cycle_id: int | None = None, timestamp: str = "") -> None:
        """记录一次节拍（``on_cycle``）。

        Args:
            cycle_id: Runtime 的轮次序号（``CycleSchedule.cycle_id``）；未知时为 ``None``。
            timestamp: 节拍时间戳（ISO 8601）。
        """
        self._cycles += 1
        if cycle_id is not None:
            self._last_cycle_id = int(cycle_id)
        if timestamp:
            self._last_cycle_timestamp = timestamp

    def observe_stable(self, snapshot: StateSnapshot) -> None:
        """记录一次稳定状态帧（``on_stable``）。"""
        self._snapshot = snapshot
        self._stable_ticks += 1
        if snapshot.changed:
            self._events += 1
            self._last_event = snapshot

    def observe_runtime(self, snapshot: RuntimeSnapshot) -> None:
        """记录运行时视图。"""
        self._runtime = snapshot

    def advance_step(self) -> None:
        """推进到下一步骤（引擎在步骤达成后调用）。"""
        self._step_index += 1
        self._steps_satisfied += 1

    def note_dispatch(self, *, retry: bool = False) -> None:
        """记录一次步骤派发（含重试）。"""
        self._steps_dispatched += 1
        if retry:
            self._steps_retried += 1

    def note_timeout(self) -> None:
        """记录一次步骤超时。"""
        self._timeouts += 1

    def note_action(self) -> None:
        """记录一次动作请求（Action 端口调用）。"""
        self._actions_requested += 1

    def note_block(self, reason: str) -> None:
        """记录一次阻塞。"""
        self._blocks += 1
        self._last_reason = reason

    def note_unblock(self, reason: str) -> None:
        """记录一次解除阻塞。"""
        self._unblocks += 1
        self._last_reason = reason

    def note_pause(self, reason: str = "pause") -> None:
        """记录一次暂停。"""
        self._pauses += 1
        self._last_reason = reason

    def note_resume(self, reason: str = "resume") -> None:
        """记录一次恢复。"""
        self._resumes += 1
        self._last_reason = reason

    # ---- 输出 ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含截图 / 用户数据）。"""
        return {
            "task_id": self._task.task_id,
            "state": self.state.value,
            "status": self.status.value,
            "step_index": self._step_index,
            "step_id": self.current_step.step_id if self.current_step else None,
            "cycles": self._cycles,
            "stable_ticks": self._stable_ticks,
            "events": self._events,
            "steps_dispatched": self._steps_dispatched,
            "steps_satisfied": self._steps_satisfied,
            "steps_retried": self._steps_retried,
            "timeouts": self._timeouts,
            "actions_requested": self._actions_requested,
            "blocks": self._blocks,
            "unblocks": self._unblocks,
            "pauses": self._pauses,
            "resumes": self._resumes,
            "elapsed_s": round(self.elapsed_s(), 3),
            "last_reason": self._last_reason,
            "state_snapshot": self._snapshot.state if self._snapshot else None,
            "runtime": self._runtime.state,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        step = self.current_step.step_id if self.current_step else "-"
        state = self._snapshot.describe() if self._snapshot else "n/a"
        return (
            f"TaskContext(task={self._task.task_id}, state={self.state.value}, "
            f"step={step}({self._step_index + 1}/{self._task.step_count}), state_view={state}, "
            f"cycles={self._cycles}, elapsed={self.elapsed_s():.1f}s, {self._runtime.describe()})"
        )

    def __repr__(self) -> str:
        return self.describe()
