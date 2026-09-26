"""任务动作适配器（Phase 7.4）：把"任务动作意图"规范成 Task 侧的 :class:`ActionRequest`。

```text
TaskDefinition ──▶ TaskEngine ──▶ ActionRequest ──▶ ActionPort ──▶ ActionDispatcher
   （声明式 / 编程式）                                          ├─▶ TargetResolver（语义名 → 位置）
                                                               ├─▶ SafetyGuard（5 道门）
                                                               └─▶ InputBackend（默认 DryRun）
```

**本模块只做两件事**：

1. 把动作意图（类型 + 语义目标 + 参数）规范成 :class:`~src.task.workflow.ActionRequest`
   —— 字段与 Action 侧契约同形（``action_id`` / ``kind`` / ``target`` / ``params`` /
   ``task_id`` / ``step_id`` / ``timeout_s`` / ``attempt``），供 ``ActionPort.request()`` 消费；
2. 在**任务侧**就把"语义目标"限定为**语义名**（``{"element": "buttons/explore"}``），
   明确拒绝像素 / 参考坐标 / 比例（``reference`` / ``x`` / ``y`` / ``x_ratio`` / ``bbox`` …）
   —— 与 Action 侧的 :class:`~src.action.models.SemanticTarget` 规则一致（两层防线）。

**明确不做**（架构决策 D-3 / D-4）：

* 不调用 ``InputBackend``（不产生输入）、不调用 ``TargetResolver``（不解析位置）、
  不读 Frame / Evidence、不推送快照（``update_snapshot`` 属 Runtime 的编排职责）；
* 不 import ``src.action`` / ``src.vision`` / ``src.capture`` / ``cv2`` / ``numpy``
  （既有 AST 用例固化的任务层边界），因此本模块产出的是**任务侧**请求对象，
  由 Action 侧用字段视图适配（``ActionRequest.from_task_request``）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.task.models import ACTION_KINDS, TaskDefinitionError
from src.task.workflow import ActionRequest
from src.utils.logger import get_logger

__all__ = [
    "FORBIDDEN_TARGET_KEYS",
    "SEMANTIC_TARGET_KEYS",
    "TARGETLESS_KINDS",
    "SemanticTargetSpec",
    "TaskActionAdapter",
    "normalize_semantic_target",
]

logger = get_logger("task.action_adapter")

#: 语义目标允许出现的字段（任务侧只接受"语义名"）
SEMANTIC_TARGET_KEYS: tuple[str, ...] = ("element", "name", "target_name")

#: 明确拒绝的坐标类字段（像素 / 参考分辨率 / 比例 / 区域）
FORBIDDEN_TARGET_KEYS: frozenset[str] = frozenset(
    {
        "reference",
        "x",
        "y",
        "x_ratio",
        "y_ratio",
        "bbox",
        "rect",
        "point",
        "position",
        "coords",
        "coordinates",
    }
)

#: 不需要语义目标的动作类型（与 Action 侧 ``ActionType.needs_target`` 对齐：仅 ``wait``）
TARGETLESS_KINDS: frozenset[str] = frozenset({"wait"})


@dataclass(frozen=True, slots=True)
class SemanticTargetSpec:
    """任务侧的语义目标声明（**只允许语义名**，禁止坐标）。

    Attributes:
        name: 语义目标名（例如 ``buttons/explore`` / ``battle_auto_button``）。

    Raises:
        TaskDefinitionError: 名称为空、类型不支持，或携带坐标 / 比例字段。
    """

    name: str

    def __post_init__(self) -> None:
        normalized = str(self.name).strip()
        if not normalized:
            raise TaskDefinitionError("语义目标名不能为空")
        object.__setattr__(self, "name", normalized)

    @classmethod
    def from_value(cls, value: Any) -> SemanticTargetSpec:
        """接受字符串或映射（``{"element": ...}`` / ``{"name": ...}``）。

        Args:
            value: 语义目标声明。

        Returns:
            目标声明。

        Raises:
            TaskDefinitionError: 类型不受支持或携带坐标字段。
        """
        if isinstance(value, SemanticTargetSpec):
            return value
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        raise TaskDefinitionError(
            f"语义目标必须是字符串或映射（当前: {type(value).__name__}）"
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SemanticTargetSpec:
        """从映射构造（``{"element": "buttons/explore"}``）。

        Args:
            data: 目标声明映射。

        Returns:
            目标声明。

        Raises:
            TaskDefinitionError: 出现坐标 / 比例字段，或没有语义名。
        """
        forbidden = sorted(str(key) for key in data if str(key) in FORBIDDEN_TARGET_KEYS)
        if forbidden:
            raise TaskDefinitionError(
                f"任务侧语义目标禁止携带坐标 / 比例字段: {forbidden}"
                f"（定位由 Action 侧 resolver 决定；请只给语义名，例如 {{'element': 'battle_auto_button'}}）"
            )
        for key in SEMANTIC_TARGET_KEYS:
            if key in data and str(data[key]).strip():
                return cls(name=str(data[key]))
        raise TaskDefinitionError(
            f"语义目标缺少名称（可用的键: {', '.join(SEMANTIC_TARGET_KEYS)}）"
        )

    def to_mapping(self) -> dict[str, str]:
        """转换为与 Action 侧 ``SemanticTarget.from_mapping()`` 兼容的写法。"""
        return {"element": self.name}

    def describe(self) -> str:
        """返回可读描述。"""
        return f"SemanticTargetSpec(name={self.name})"


def normalize_semantic_target(value: Any) -> dict[str, str] | None:
    """把语义目标规范成 ``{"element": <name>}``（``None`` 表示"无目标"，仅 ``wait`` 允许）。

    Args:
        value: 字符串 / 映射 / :class:`SemanticTargetSpec` / ``None``。

    Returns:
        规范化的目标映射；``None`` 原样返回。

    Raises:
        TaskDefinitionError: 目标非法（空名、坐标字段、类型不支持）。
    """
    if value is None:
        return None
    return SemanticTargetSpec.from_value(value).to_mapping()


class TaskActionAdapter:
    """构造任务侧动作请求的便利层（**不执行任何动作**）。

    Args:
        task_id: 任务标识（写入 ``ActionRequest.task_id``，用于审计与幂等键）。
        step_id: 默认步骤标识（写入 ``ActionRequest.step_id``）。
        timeout_s: 默认执行层超时（秒）。
        attempt: 默认尝试序号（从 1 开始）。
    """

    def __init__(
        self,
        *,
        task_id: str = "",
        step_id: str = "",
        timeout_s: float = 1.0,
        attempt: int = 1,
    ) -> None:
        self._task_id = str(task_id)
        self._step_id = str(step_id)
        self._timeout_s = float(timeout_s)
        self._attempt = int(attempt)

    # ---- 只读 ---------------------------------------------------------------

    @property
    def task_id(self) -> str:
        """任务标识。"""
        return self._task_id

    @property
    def step_id(self) -> str:
        """默认步骤标识。"""
        return self._step_id

    def with_step(self, step_id: str, *, timeout_s: float | None = None) -> TaskActionAdapter:
        """派生一个绑定到具体步骤的适配器。

        Args:
            step_id: 步骤标识。
            timeout_s: 覆盖默认超时。

        Returns:
            新的适配器实例。
        """
        return TaskActionAdapter(
            task_id=self._task_id,
            step_id=str(step_id),
            timeout_s=self._timeout_s if timeout_s is None else float(timeout_s),
            attempt=self._attempt,
        )

    # ---- 构造 ---------------------------------------------------------------

    def build(
        self,
        kind: str,
        target: Any = None,
        *,
        params: Mapping[str, Any] | None = None,
        step_id: str | None = None,
        timeout_s: float | None = None,
        attempt: int | None = None,
        suffix: str | None = None,
    ) -> ActionRequest:
        """构造一个动作请求（不提交、不执行）。

        Args:
            kind: 动作类型（``ACTION_KINDS`` 之一）。
            target: 语义目标（字符串 / ``{"element": ...}``）；``wait`` 可省略。
            params: 附加参数（按键名 / 时长等）。
            step_id: 覆盖步骤标识。
            timeout_s: 覆盖执行层超时。
            attempt: 覆盖尝试序号。
            suffix: 幂等键后缀（多条同步骤动作时用于区分）。

        Returns:
            任务侧 :class:`~src.task.workflow.ActionRequest`。

        Raises:
            TaskDefinitionError: 动作类型非法、需要目标却没有目标、目标携带坐标。
        """
        action_kind = str(kind).strip().lower()
        if action_kind not in ACTION_KINDS:
            raise TaskDefinitionError(
                f"动作类型非法: {kind!r}（可选: {', '.join(sorted(ACTION_KINDS))}）"
            )
        active_step = str(step_id if step_id is not None else self._step_id)
        normalized = normalize_semantic_target(target)
        if action_kind not in TARGETLESS_KINDS and normalized is None:
            raise TaskDefinitionError(f"{action_kind} 需要语义目标（target 不能为空）")
        active_attempt = int(attempt if attempt is not None else self._attempt)
        identifier = f"{self._task_id or 'task'}:{active_step or 'adhoc'}:{active_attempt}"
        if suffix:
            identifier = f"{identifier}:{suffix}"
        request = ActionRequest(
            action_id=identifier,
            kind=action_kind,
            target=normalized,
            task_id=self._task_id,
            step_id=active_step,
            params=dict(params or {}),
            timeout_s=float(timeout_s if timeout_s is not None else self._timeout_s),
            attempt=active_attempt,
        )
        logger.debug("已构造任务动作请求: %s", request.describe())
        return request

    def click(self, target: Any, *, params: Mapping[str, Any] | None = None, **kwargs: Any) -> ActionRequest:
        """构造 ``click`` 请求（目标必须是语义名）。"""
        return self.build("click", target, params=params, **kwargs)

    def double_click(
        self, target: Any, *, params: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> ActionRequest:
        """构造 ``double_click`` 请求。"""
        return self.build("double_click", target, params=params, **kwargs)

    def long_press(
        self, target: Any, *, params: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> ActionRequest:
        """构造 ``long_press`` 请求。"""
        return self.build("long_press", target, params=params, **kwargs)

    def drag(
        self, source: Any, destination: Any, *, params: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> ActionRequest:
        """构造 ``drag`` 请求（起点 + 终点，均为语义名）。

        注意：Action 侧目前对 ``drag`` 返回 ``REJECTED``（尚未实现），
        此处允许构造以便任务定义先行表达意图。
        """
        merged = dict(params or {})
        merged.setdefault("from_target", normalize_semantic_target(source))
        merged.setdefault("to_target", normalize_semantic_target(destination))
        return self.build("drag", destination, params=merged, **kwargs)

    def key(
        self,
        key_name: str,
        target: Any = None,
        *,
        params: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ActionRequest:
        """构造 ``key`` 请求。

        注意：Action 侧契约要求"除 ``wait`` 外的动作都要有语义目标"（``ActionType.needs_target``），
        因此这里也接受（但不使用）一个语义名——当前实现只发送按键，不点击目标
        （见报告"未解决问题"中的 Q-7.4-1）。

        Args:
            key_name: 按键名（例如 ``enter``）。
            target: 语义目标（字符串 / ``{"element": ...}``）。
            params: 附加参数（例如 ``hold_s``）。
            **kwargs: 透传给 :meth:`build`（``step_id`` / ``timeout_s`` / ``attempt`` …）。

        Returns:
            任务侧动作请求。
        """
        merged = dict(params or {})
        merged.setdefault("key", str(key_name))
        return self.build("key", target, params=merged, **kwargs)

    def wait(self, *, params: Mapping[str, Any] | None = None, **kwargs: Any) -> ActionRequest:
        """构造 ``wait`` 请求（只等待、不产生输入）。"""
        return self.build("wait", None, params=params, **kwargs)

    # ---- 校验 ---------------------------------------------------------------

    @staticmethod
    def normalize_target(value: Any) -> dict[str, str] | None:
        """把语义目标规范成 ``{"element": <name>}``（见 :func:`normalize_semantic_target`）。"""
        return normalize_semantic_target(value)

    @staticmethod
    def spec(value: Any) -> SemanticTargetSpec:
        """解析语义目标（返回 :class:`SemanticTargetSpec`）。"""
        return SemanticTargetSpec.from_value(value)

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"TaskActionAdapter(task_id={self._task_id or 'n/a'}, "
            f"step_id={self._step_id or 'n/a'}, timeout_s={self._timeout_s:g}, "
            f"attempt={self._attempt})"
        )

    def __repr__(self) -> str:
        return self.describe()
