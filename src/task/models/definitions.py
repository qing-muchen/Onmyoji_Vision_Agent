"""Task 定义模型（Phase 6.2：从 ``models.py`` 拆出）。

本模块只放**声明式定义**与相关枚举 / 常量：

* 线性步骤模型（Phase 6.1）：:class:`Task` / :class:`Step` / :class:`Budget` / :class:`TaskPolicies`；
* 状态机式工作流定义（Phase 6.2，来自 ``configs/tasks/*.yaml``）：
  :class:`TaskDefinition` / :class:`WorkflowState` / :class:`WorkflowTransition` / :class:`WorkflowAction`。

运行期视图（状态 / 运行时快照）在 :mod:`src.task.models.context`，结果对象在
:mod:`src.task.models.result`。

依赖边界（架构决策 D-4）：本模块不 import ``cv2`` / ``src.vision`` / ``src.runtime`` /
``src.action``；状态名与运行时状态一律以**字符串**表达（Q-1）。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Callable, Protocol, runtime_checkable

from src.task.state_machine.lifecycle import TaskError, TaskState

__all__ = [
    "ACTION_KINDS",
    "CAPTURE_BLOCKED_REASONS",
    "DEFAULT_ACTION_TIMEOUT_S",
    "DEFAULT_STEP_TIMEOUT_S",
    "MISMATCH_POLICIES",
    "RUNTIME_FAILURE_POLICIES",
    "RUNTIME_FAILURE_REASONS",
    "TERMINAL_STATE_NAME",
    "TIMEOUT_POLICIES",
    "UNKNOWN_POLICIES",
    "VISION_UNKNOWN_REASONS",
    "Budget",
    "Step",
    "StepPhase",
    "Task",
    "TaskDefinition",
    "TaskDefinitionError",
    "TaskLike",
    "TaskPolicies",
    "TaskStatus",
    "WorkflowAction",
    "WorkflowState",
    "WorkflowTransition",
]

#: 状态机式工作流中表示"成功结束"的保留目标名（不需要在 ``states`` 中定义）
TERMINAL_STATE_NAME: str = "COMPLETE"



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
#: 默认的动作执行超时（秒；与步骤验证窗口 `step.timeout` 是两回事）
DEFAULT_ACTION_TIMEOUT_S: float = 1.0
#: 默认的步骤 / 状态等待窗口（秒）——与 Phase 6.1 的 ``Step.timeout_s`` 默认值一致
DEFAULT_STEP_TIMEOUT_S: float = 10.0


class TaskDefinitionError(TaskError):
    """任务定义非法（空步骤、未知动作类型、策略取值非法等）。"""


class TaskAdmissionDenied(TaskDefinitionError):
    """调度**准入被拒绝**（admission guard 判定；Phase 12.1-C1）。

    语义边界（与 :class:`TaskDefinitionError` 严格区分）：

    * 只用于**明确的准入拒绝**（例如 E-STOP 有效、任务被禁用）——**不是**任务定义错误、
      **不是**运行期异常、**不是**代码缺陷；
    * 被拒绝时**不会创建** ``TaskInstance``（``submit()`` 成功时的返回契约不变）；
    * 携带**结构化**拒绝原因 :attr:`reason`（如 ``estop_active`` / ``guard_error``）：
      调用方按字段判定，**不要**解析异常文本。

    Attributes:
        task_id: 被拒绝的任务标识。
        reason: 结构化拒绝原因（机器可读）：``admission_guard`` 的返回值，或 guard 自身
            抛异常时的 ``guard_error``（fail-closed）。
        detail: 附加说明（人类可读；可为空）。
    """

    def __init__(self, task_id: str, reason: str, *, detail: str = "") -> None:
        self.task_id = str(task_id)
        self.reason = str(reason)
        self.detail = str(detail)
        message = f"调度准入被拒绝: {self.task_id}（reason={self.reason}）"
        if self.detail:
            message = f"{message}: {self.detail}"
        super().__init__(message)

    def to_dict(self) -> dict[str, str]:
        """返回可 JSON 序列化的字典（审计用；不含截图 / 用户数据）。"""
        return {"task_id": self.task_id, "reason": self.reason, "detail": self.detail}


class StepPhase(StrEnum):
    """步骤 / 状态的执行阶段（引擎与工作流执行器共用，Phase 6.1 / 6.2）。"""

    #: 任务未启动 / 步骤尚未进入
    IDLE = "idle"
    #: 等待观测到满足（尚未派发动作）
    PENDING = "pending"
    #: 已派发动作（或进入等待），等待达成
    AWAITING = "awaiting"
    #: 已达成并推进
    SATISFIED = "satisfied"
    #: 与期望冲突（``forbid`` 命中）
    VIOLATED = "violated"
    #: 动作被拒绝 / 失败
    REJECTED = "rejected"
    #: 超时
    TIMEOUT = "timeout"
    #: 任务已结束
    DONE = "done"


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

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Budget:
        """从 YAML 映射构造预算（YAML 用 ``timeout_s`` 表示任务时长上限）。

        Args:
            data: 预算定义，键为 :class:`Budget` 字段；``timeout_s`` 等价于 ``deadline_s``。

        Returns:
            预算实例。

        Raises:
            TaskDefinitionError: 字段非法（由 ``__post_init__`` 校验）或出现未知字段。
        """
        fields = {key: value for key, value in data.items() if key != "timeout_s"}
        if "timeout_s" in data and "deadline_s" not in data:
            fields["deadline_s"] = data["timeout_s"]
        allowed = {"max_steps", "deadline_s", "timeout_s", "max_step_retries", "min_stable_frames"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise TaskDefinitionError(f"budget 出现未知字段: {unknown}（可选: {sorted(allowed)}）")
        return cls(**fields)


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


# ---- 状态机式工作流定义（Phase 6.2：来自 configs/tasks/*.yaml） ----------------


@runtime_checkable
class TaskLike(Protocol):
    """``TaskContext`` 可接受的"任务定义"最小接口。

    :class:`Task`（线性步骤）与 :class:`TaskDefinition`（状态机式工作流）都满足，
    因此引擎与上下文不必关心任务来自 Python 还是 YAML。
    """

    @property
    def task_id(self) -> str:
        """任务标识。"""
        ...

    @property
    def step_count(self) -> int:
        """步骤 / 状态数量。"""
        ...

    def step_at(self, index: int) -> Any | None:
        """按序号取步骤；状态机式定义返回 ``None``（它没有"第 N 步"语义）。"""
        ...

    def describe(self) -> str:
        """返回可读描述。"""
        ...


@dataclass(frozen=True, slots=True)
class WorkflowAction:
    """工作流状态里的一个动作声明（**语义动作**，不含屏幕绝对坐标，D-3）。

    Attributes:
        kind: 动作类型（``ACTION_KINDS`` 之一）；``wait`` 表示"只等待、不请求输入"。
        target: 语义目标（``{"element": ...}`` 或 ``{"reference": [x, y]}``）；``wait`` 时为 ``None``。
        timeout_s: 执行层超时（秒）。
        params: 附加参数（按键 / 时长等）。

    Raises:
        TaskDefinitionError: 动作类型缺失或非法、需要目标却没有目标。
    """

    kind: str
    target: Mapping[str, Any] | None = None
    timeout_s: float = DEFAULT_ACTION_TIMEOUT_S
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise TaskDefinitionError(
                f"action.kind 非法: {self.kind!r}（可选: {', '.join(sorted(ACTION_KINDS))}）"
            )
        if self.timeout_s <= 0:
            raise TaskDefinitionError(f"action.timeout 必须为正数: {self.timeout_s}")
        if self.kind != "wait" and not self.target:
            raise TaskDefinitionError(f"action[{self.kind}] 需要 target（语义元素或参考坐标）")

    @property
    def is_wait(self) -> bool:
        """是否"只等待"（不请求任何输入）。"""
        return self.kind == "wait"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> WorkflowAction:
        """从 YAML 映射构造动作声明。

        支持两种写法：

        * ``{"kind": "click", "target": {"element": "buttons/explore"}}``（显式 kind）；
        * ``{"target": "wait", "timeout": 30}``（动作类型写在 ``target`` 上，常见于"等待"）。

        Args:
            data: 动作定义。

        Returns:
            动作声明。

        Raises:
            TaskDefinitionError: 字段缺失或非法。
        """
        raw_kind = str(data.get("kind", "") or "").strip().lower()
        target = data.get("target")
        if not raw_kind and isinstance(target, str):
            raw_kind = target.strip().lower()
            target = None
        params = data.get("params") or {}
        if not isinstance(params, Mapping):
            raise TaskDefinitionError("action.params 必须是映射")
        timeout = data.get("timeout", data.get("timeout_s", DEFAULT_ACTION_TIMEOUT_S))
        return cls(
            kind=raw_kind,
            target=dict(target) if isinstance(target, Mapping) else None,
            timeout_s=float(timeout),
            params=dict(params),
        )

    def describe(self) -> str:
        """返回可读描述。"""
        target = ""
        if self.target:
            key = "element" if "element" in self.target else "reference"
            target = f":{self.target.get(key)}"
        return f"{self.kind}{target}(timeout={self.timeout_s:g}s)"


@dataclass(frozen=True, slots=True)
class WorkflowTransition:
    """工作流转移：满足 ``condition`` 时进入 ``next_state``。

    Attributes:
        condition_spec: 条件声明（``str`` 表达式 / ``Mapping`` / 可调用对象；保留原文便于审计）。
        predicate: **编译后**的谓词（由 :class:`~src.task.loader.TaskLoader` 用
            :func:`~src.task.hooks.build_condition` 构建——**禁止** ``eval`` 与动态执行字符串）。
        next_state: 目标状态名，或保留字 :data:`TERMINAL_STATE_NAME`（成功结束）。
        name: 转移标识（缺省为 ``to_<next_state>``）。

    Raises:
        TaskDefinitionError: 缺少条件或目标、或谓词未编译。
    """

    condition_spec: Any
    predicate: Callable[[Any, Any], bool]
    next_state: str
    name: str = ""

    def __post_init__(self) -> None:
        if self.condition_spec is None:
            raise TaskDefinitionError("transition.condition 不能为空")
        if not self.next_state.strip():
            raise TaskDefinitionError("transition.next 不能为空")
        if not callable(self.predicate):
            raise TaskDefinitionError("transition 需要一个已编译的谓词（禁止字符串动态执行）")

    @property
    def is_terminal(self) -> bool:
        """是否指向工作流成功结束。"""
        return self.next_state == TERMINAL_STATE_NAME

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], predicate: Callable[[Any, Any], bool]
    ) -> WorkflowTransition:
        """从 YAML 映射构造转移（``predicate`` 由 loader 编译后传入）。

        Args:
            data: 转移定义（``condition`` / ``next`` / 可选 ``name``）。
            predicate: 已编译的条件谓词。

        Returns:
            转移定义。
        """
        spec = data.get("condition")
        next_state = str(data.get("next", "") or "").strip()
        return cls(
            condition_spec=spec,
            predicate=predicate,
            next_state=next_state,
            name=str(data.get("name", "") or "") or f"to_{next_state}",
        )

    def describe(self) -> str:
        """返回可读描述。"""
        condition = self.condition_spec
        if isinstance(condition, Mapping):
            condition = dict(condition)
        return f"{self.name}: {condition} -> {self.next_state}"


@dataclass(frozen=True, slots=True)
class WorkflowState:
    """工作流中的一个状态（Step 定义）。

    Attributes:
        name: 状态名。
        description: 人类可读说明。
        actions: 进入该状态时请求的动作（按顺序提交；``wait`` 只等待）。
        transitions: 条件转移（**按声明顺序**评估，第一个满足者生效）。
        timeout_s: 该状态的等待窗口（秒）；超时按策略重试或失败。
        retry: 该状态的重试上限（``None`` = 使用预算默认值）。
        entry_condition_spec / entry_predicate: 可选的"进入条件"（不满足则不进入该状态）。

    Raises:
        TaskDefinitionError: 缺少状态名、字段非法、使用保留名、进入条件未编译。
    """

    name: str
    description: str = ""
    actions: tuple[WorkflowAction, ...] = ()
    transitions: tuple[WorkflowTransition, ...] = ()
    timeout_s: float = DEFAULT_STEP_TIMEOUT_S
    retry: int | None = None
    entry_condition_spec: Any | None = None
    entry_predicate: Callable[[Any, Any], bool] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise TaskDefinitionError("state 名不能为空")
        if self.name == TERMINAL_STATE_NAME:
            raise TaskDefinitionError(
                f"state 名不能使用保留字 {TERMINAL_STATE_NAME}（它表示工作流成功结束）"
            )
        if self.timeout_s <= 0:
            raise TaskDefinitionError(f"state[{self.name}].timeout 必须为正数: {self.timeout_s}")
        if self.retry is not None and self.retry < 0:
            raise TaskDefinitionError(f"state[{self.name}].retry 不能为负数: {self.retry}")
        if self.entry_condition_spec is not None and self.entry_predicate is None:
            raise TaskDefinitionError(
                f"state[{self.name}].entry_condition 需要编译后的谓词（loader 负责编译）"
            )

    @property
    def has_actions(self) -> bool:
        """是否声明了动作。"""
        return bool(self.actions)

    def first_action(self) -> WorkflowAction | None:
        """第一个动作（本阶段每个状态通常只有一个）。"""
        return self.actions[0] if self.actions else None

    def describe(self) -> str:
        """返回可读描述。"""
        actions = ",".join(action.describe() for action in self.actions) or "none"
        targets = ",".join(item.next_state for item in self.transitions) or "none"
        return (
            f"state={self.name} actions=[{actions}] timeout={self.timeout_s:g}s "
            f"retry={self.retry if self.retry is not None else 'default'} -> [{targets}]"
        )


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """状态机式工作流定义（Phase 6.2；由 :class:`~src.task.loader.TaskLoader` 校验后产出）。

    与 Phase 6.1 的线性 :class:`Task` 并存：引擎的 ``start()`` 两种模式都支持。

    Attributes:
        name: 任务名（YAML ``name``）。
        initial: 起始状态名（YAML ``initial``）。
        states: 状态名 → :class:`WorkflowState`。
        budget: 预算（``max_steps`` / ``deadline_s``；YAML 可用 ``timeout_s`` 表示时长）。
        policies: 策略（unknown / runtime failure / mismatch / timeout）。
        description: 人类可读说明。
        metadata: 附加信息（不参与判定）。
        source_path: 来源文件路径（审计用）。
    """

    name: str
    initial: str
    states: Mapping[str, WorkflowState]
    budget: Budget = field(default_factory=Budget)
    policies: TaskPolicies = field(default_factory=TaskPolicies)
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @property
    def task_id(self) -> str:
        """任务标识（等价于 :attr:`name`，供 :class:`TaskLike` 使用）。"""
        return self.name

    @property
    def step_count(self) -> int:
        """状态数量。"""
        return len(self.states)

    @property
    def state_names(self) -> tuple[str, ...]:
        """状态名序列（保持 YAML 声明顺序）。"""
        return tuple(self.states)

    @property
    def initial_state(self) -> WorkflowState:
        """起始状态。"""
        return self.states[self.initial]

    def state(self, name: str) -> WorkflowState | None:
        """按名称取状态（不存在时返回 ``None``）。"""
        return self.states.get(name)

    def step_at(self, index: int) -> None:
        """状态机式定义没有"第 N 步"语义，恒返回 ``None``（满足 :class:`TaskLike`）。"""
        return None

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含截图 / 用户数据）。"""
        return {
            "name": self.name,
            "description": self.description,
            "initial": self.initial,
            "states": {
                key: {
                    "actions": [
                        {
                            "kind": action.kind,
                            "target": dict(action.target) if action.target else None,
                        }
                        for action in state.actions
                    ],
                    "transitions": [
                        {
                            "name": item.name,
                            "condition": (
                                dict(item.condition_spec)
                                if isinstance(item.condition_spec, Mapping)
                                else str(item.condition_spec)
                            ),
                            "next": item.next_state,
                        }
                        for item in state.transitions
                    ],
                    "timeout_s": state.timeout_s,
                    "retry": state.retry,
                }
                for key, state in self.states.items()
            },
            "budget": {
                "max_steps": self.budget.max_steps,
                "deadline_s": self.budget.deadline_s,
            },
            "source_path": self.source_path,
        }

    def describe(self) -> str:
        """可读描述。"""
        return (
            f"TaskDefinition(name={self.name}, initial={self.initial}, "
            f"states={self.step_count} [{', '.join(self.state_names)}], "
            f"{self.budget.describe()}, {self.policies.describe()})"
        )
