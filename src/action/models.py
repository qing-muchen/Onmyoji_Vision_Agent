"""Action 层数据模型（Phase 7）：语义动作请求、语义目标、解析结果与执行结果。

设计红线（本阶段不可突破）：

1. **禁止像素坐标**：:class:`ActionRequest` / :class:`SemanticTarget` / :class:`ResolvedTarget`
   都不携带屏幕绝对坐标或像素 x/y；坐标只以**窗口相对比例**（``x_ratio`` / ``y_ratio``）表达，
   实际屏幕坐标在 :class:`~src.action.dispatcher.ActionDispatcher` 内由
   ``window_client_rect × ratio`` 计算得出；:class:`ActionRequest` 的参数里出现坐标键会直接报错。
2. **禁止坐标即名字**：目标名不得形如 ``CLICK_1530_850``（见 :func:`validate_target_name`）。
3. **ActionRequest 只描述"做什么"**：不含 Frame / 证据 / 状态，也不引用 detector；
   定位由 :class:`~src.action.resolver.TargetResolver` 完成（Phase 7 用静态注册表，
   Phase 7.1 接入 vision-aware 解析）。

依赖边界（AST 测试固化）：只 import 标准库与 ``src.core`` / ``src.utils``；
**不** import ``src.vision`` / ``src.evidence`` / ``src.task`` / ``src.runtime`` / ``cv2`` /
``pyautogui`` / ``pynput``。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from src.core.exceptions import OnmyojiVisionError
from src.core.geometry import Point, Rect

__all__ = [
    "ClockFn",
    "now_iso",
    "FORBIDDEN_PARAMETER_KEYS",
    "ActionError",
    "ActionOutcome",
    "ActionRequest",
    "ActionRequestError",
    "ActionResolutionError",
    "ActionStatus",
    "ActionTarget",
    "ActionTargetError",
    "ActionType",
    "InputBackendError",
    "InputBackendUnavailableError",
    "ResolvedTarget",
    "SemanticTarget",
    "outcome_from_any",
    "validate_target_name",
]


# ---- 异常 -------------------------------------------------------------------


class ActionError(OnmyojiVisionError):
    """Action 层错误基类。"""


class ActionRequestError(ActionError):
    """动作请求非法（缺字段 / 含像素坐标 / 类型未知）。"""


class ActionTargetError(ActionError):
    """语义目标非法（名称为空、比例越界、把坐标写进名字）。"""


class ActionResolutionError(ActionError):
    """目标解析失败（未注册 / 解析器不可用）。"""


class InputBackendError(ActionError):
    """输入后端错误（平台不支持、调用失败）。"""


class InputBackendUnavailableError(InputBackendError):
    """输入后端不可用（例如非 Windows 平台请求 Windows 后端）。"""


# ---- 枚举 -------------------------------------------------------------------


class ActionType(StrEnum):
    """动作类型（与 Task 层 ``ACTION_KINDS`` 同名，保持 YAML 一致）。"""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    LONG_PRESS = "long_press"
    DRAG = "drag"
    KEY = "key"
    WAIT = "wait"

    @property
    def needs_target(self) -> bool:
        """是否需要语义目标（``wait`` 不需要）。"""
        return self is not ActionType.WAIT

    @property
    def is_implemented(self) -> bool:
        """本阶段是否已实现（``drag`` 留待 Phase 7.1）。"""
        return self is not ActionType.DRAG


class ActionStatus(StrEnum):
    """动作请求的处理结果（与 Task 层 ``ActionStatus`` 语义一致）。"""

    #: 已受理并执行（**不代表界面已响应**：效果由后续状态验证）
    EXECUTED = "executed"
    #: 已受理但未执行（``wait`` 或 dry-run 后端）
    SKIPPED = "skipped"
    #: 被安全门拒绝（E-STOP / 窗口无效 / 失焦 / 未解析 / 限流 / 未实现）
    REJECTED = "rejected"
    #: 执行失败（后端异常）
    FAILED = "failed"


class ActionTarget(StrEnum):
    """内置语义目标名。

    Registry 也**接受任意自定义名称**（例如 ``buttons/explore``）：本枚举只是"内置常用目标"
    的规范拼写，便于代码中引用与文档化，不构成白名单。

    禁止把坐标编码进目标名（``CLICK_1530_850``）——见 :func:`validate_target_name`。
    """

    BATTLE_AUTO_BUTTON = "BATTLE_AUTO_BUTTON"
    BATTLE_RESULT_CONFIRM = "BATTLE_RESULT_CONFIRM"
    REWARD_CONFIRM = "REWARD_CONFIRM"
    EXPLORATION_ENTRY = "EXPLORATION_ENTRY"
    YARD_ENTRY = "YARD_ENTRY"

    @classmethod
    def is_builtin(cls, name: str) -> bool:
        """``name`` 是否为内置目标名。"""
        return name in cls._value2member_map_

    @classmethod
    def builtin_names(cls) -> tuple[str, ...]:
        """内置目标名（按声明顺序）。"""
        return tuple(item.value for item in cls)


#: 形如 ``CLICK_1530_850`` 的目标名（把像素坐标写进名字：禁止）
_PIXEL_NAME_PATTERN = re.compile(r"^(?:click|tap|move|point|press)_\d+_\d+$", re.IGNORECASE)


def validate_target_name(name: str) -> str:
    """校验语义目标名（禁止把像素坐标编码进名字）。

    Args:
        name: 目标名。

    Returns:
        规范化后的目标名（去除首尾空白）。

    Raises:
        ActionTargetError: 名称为空或形如 ``CLICK_1530_850``。
    """
    normalized = name.strip()
    if not normalized:
        raise ActionTargetError("语义目标名不能为空")
    if _PIXEL_NAME_PATTERN.match(normalized):
        raise ActionTargetError(
            f"语义目标名不能编码像素坐标: {normalized!r}"
            f"（请使用语义名，如 BATTLE_AUTO_BUTTON；"
            f"确需比例定位请用 x_ratio / y_ratio）"
        )
    return normalized


# ---- 目标 -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticTarget:
    """语义目标：**语义名** 或 **窗口相对比例**（二选一，禁止像素坐标）。

    Attributes:
        name: 语义目标名（查 :class:`~src.action.registry.TargetRegistry`）。
        x_ratio: 窗口客户区横向比例（0..1，0 = 左边缘）。
        y_ratio: 窗口客户区纵向比例（0..1，0 = 上边缘）。

    Raises:
        ActionTargetError: 名称与比例同时给出或都未给出、比例越界、名称编码坐标。
    """

    name: str = ""
    x_ratio: float | None = None
    y_ratio: float | None = None

    def __post_init__(self) -> None:
        has_name = bool(self.name.strip())
        has_ratio = self.x_ratio is not None or self.y_ratio is not None
        if has_name and has_ratio:
            raise ActionTargetError("语义目标只能给定 名称 或 比例 之一")
        if not has_name and not has_ratio:
            raise ActionTargetError("语义目标必须给定 名称 或 x_ratio / y_ratio")
        if has_name:
            object.__setattr__(self, "name", validate_target_name(self.name))
            return
        if self.x_ratio is None or self.y_ratio is None:
            raise ActionTargetError("x_ratio 与 y_ratio 必须同时给出")
        for label, value in (("x_ratio", self.x_ratio), ("y_ratio", self.y_ratio)):
            if not 0.0 <= value <= 1.0:
                raise ActionTargetError(f"{label} 必须在 0..1 之间: {value}")

    @property
    def is_named(self) -> bool:
        """是否按语义名定位。"""
        return bool(self.name)

    @property
    def is_ratio(self) -> bool:
        """是否按窗口相对比例定位。"""
        return self.x_ratio is not None

    @classmethod
    def from_name(cls, name: str) -> SemanticTarget:
        """按语义名构造。"""
        return cls(name=name)

    @classmethod
    def from_ratio(cls, x_ratio: float, y_ratio: float) -> SemanticTarget:
        """按窗口相对比例构造。"""
        return cls(x_ratio=float(x_ratio), y_ratio=float(y_ratio))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> SemanticTarget | None:
        """从声明映射构造语义目标（``None`` 表示"无目标"，仅 ``wait`` 允许）。

        支持的写法：

        * ``{"element": "buttons/explore"}`` / ``{"name": "BATTLE_AUTO_BUTTON"}``；
        * ``{"x_ratio": 0.92, "y_ratio": 0.95}``；
        * ``{"target": <上述之一>}``（Task 层 YAML 的嵌套写法）。

        Raises:
            ActionTargetError: 出现像素 / 参考分辨率坐标（``reference`` / ``x`` / ``y``），
                或写法不被支持。
        """
        if data is None:
            return None
        payload = data.get("target") if "target" in data and isinstance(data.get("target"), Mapping) else data
        if not isinstance(payload, Mapping):
            raise ActionTargetError(f"语义目标必须是映射: {type(payload).__name__}")
        for key in ("reference", "coordinates", "coords", "x", "y", "screen_x", "screen_y", "pixel"):
            if key in payload:
                raise ActionTargetError(
                    f"语义目标不接受像素 / 参考坐标字段: {key!r}"
                    f"（请使用语义名或窗口相对比例 x_ratio / y_ratio）"
                )
        if "x_ratio" in payload or "y_ratio" in payload:
            return cls.from_ratio(
                float(payload.get("x_ratio", 0.0)), float(payload.get("y_ratio", 0.0))
            )
        name = payload.get("element") or payload.get("name") or payload.get("target_name")
        if name:
            return cls.from_name(str(name))
        raise ActionTargetError(f"无法识别的语义目标写法: {sorted(map(str, payload))}")

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含像素坐标）。"""
        if self.is_named:
            return {"name": self.name}
        return {"x_ratio": self.x_ratio, "y_ratio": self.y_ratio}

    def describe(self) -> str:
        """返回可读描述。"""
        if self.is_named:
            return f"SemanticTarget(name={self.name})"
        return f"SemanticTarget(x_ratio={self.x_ratio:.4f}, y_ratio={self.y_ratio:.4f})"


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """解析结果：窗口相对位置 + 置信度 + 来源（不含任何像素坐标）。

    Attributes:
        target: 语义目标名（``<explicit-ratio>`` 表示调用方直接给的比例）。
        x_ratio: 窗口客户区横向比例（0..1）。
        y_ratio: 窗口客户区纵向比例（0..1）。
        confidence: 解析置信度（0..1；静态注册表为 1.0，Phase 7.1 的 vision 解析为检测得分）。
        source: 解析来源（``registry`` / ``explicit_ratio``；Phase 7.1 会有 ``vision``）。
        calibrated: 是否经过实机标定（占位值 ``False``；**非 dry-run 后端拒绝未标定目标**）。

    Raises:
        ActionTargetError: 比例或置信度越界、来源为空。
    """

    target: str
    x_ratio: float
    y_ratio: float
    confidence: float = 1.0
    source: str = "registry"
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise ActionTargetError("ResolvedTarget.target 不能为空")
        if not self.source.strip():
            raise ActionTargetError("ResolvedTarget.source 不能为空")
        for label, value in (("x_ratio", self.x_ratio), ("y_ratio", self.y_ratio)):
            if not 0.0 <= value <= 1.0:
                raise ActionTargetError(f"ResolvedTarget.{label} 必须在 0..1 之间: {value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ActionTargetError(f"ResolvedTarget.confidence 必须在 0..1 之间: {self.confidence}")

    @property
    def window_relative_position(self) -> dict[str, float]:
        """窗口相对位置（比例形式）。"""
        return {"x_ratio": self.x_ratio, "y_ratio": self.y_ratio}

    def client_point(self, client_rect: Rect) -> Point:
        """把窗口相对比例换算为**屏幕坐标**（唯一的坐标计算入口）。

        Args:
            client_rect: 目标窗口客户区在屏幕上的矩形（来自窗口探测器 / 窗口层）。

        Returns:
            屏幕坐标点（客户区比例映射结果）。
        """
        return Point(
            x=float(client_rect.x) + client_rect.width * self.x_ratio,
            y=float(client_rect.y) + client_rect.height * self.y_ratio,
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "target": self.target,
            "window_relative_position": self.window_relative_position,
            "confidence": self.confidence,
            "source": self.source,
            "calibrated": self.calibrated,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        position = self.window_relative_position
        return (
            f"ResolvedTarget(target={self.target}, x_ratio={position['x_ratio']:.4f}, "
            f"y_ratio={position['y_ratio']:.4f}, confidence={self.confidence:.2f}, "
            f"source={self.source}, calibrated={self.calibrated})"
        )


# ---- 请求与结果 -------------------------------------------------------------

#: 参数中禁止出现的键（像素 / 坐标语义）
FORBIDDEN_PARAMETER_KEYS: frozenset[str] = frozenset(
    {
        "x",
        "y",
        "screen_x",
        "screen_y",
        "client_x",
        "client_y",
        "pixel_x",
        "pixel_y",
        "coordinate",
        "coordinates",
        "coords",
        "position",
        "point",
        "pos",
    }
)


def now_iso() -> str:
    """返回本地时间的 ISO 8601 字符串（与 Runtime / Task 同格式）。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _pick(source: Any, *names: str, default: Any = None) -> Any:
    """按名称从对象或映射中取值（找不到时返回 ``default``）。"""
    for name in names:
        if isinstance(source, Mapping):
            if name in source:
                return source[name]
            continue
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _text(value: Any) -> str:
    """把枚举 / 字符串取值统一成字符串（``None`` → 空串）。"""
    if value is None:
        return ""
    inner = getattr(value, "value", value)
    return inner.strip() if isinstance(inner, str) else str(inner)


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """一次**语义**动作请求（Task → Action 的契约）。

    必需字段（Phase 7 规范）：``request_id`` / ``action_type`` / ``target`` / ``parameters`` /
    ``created_at``。**禁止**出现屏幕坐标与像素 x/y：坐标只能通过 :class:`SemanticTarget` 的
    窗口相对比例表达。

    Attributes:
        request_id: 幂等键（``task:step:attempt`` 或调用方自定义）。
        action_type: 动作类型。
        target: 语义目标（``wait`` 可为 ``None``）。
        parameters: 动作参数（**禁止坐标键**，见 :data:`FORBIDDEN_PARAMETER_KEYS`）。
        created_at: 创建时间（ISO 8601）。
        task_id / step_id: 来源标识（审计用；可空）。
        timeout_s: 执行层超时（秒）。
        attempt: 第几次尝试（从 1 开始）。

    Raises:
        ActionRequestError: 缺字段、类型非法、参数含坐标键、需要目标却没有目标。
    """

    request_id: str
    action_type: ActionType
    target: SemanticTarget | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    task_id: str = ""
    step_id: str = ""
    timeout_s: float = 1.0
    attempt: int = 1

    def __post_init__(self) -> None:
        if not str(self.request_id).strip():
            raise ActionRequestError("ActionRequest.request_id 不能为空")
        if not isinstance(self.action_type, ActionType):
            raise ActionRequestError(
                f"ActionRequest.action_type 非法: {self.action_type!r}"
                f"（可选: {', '.join(item.value for item in ActionType)}）"
            )
        if self.action_type.needs_target and self.target is None:
            raise ActionRequestError(f"{self.action_type.value} 需要语义目标（target 不能为空）")
        if self.target is not None and not isinstance(self.target, SemanticTarget):
            raise ActionRequestError("ActionRequest.target 必须是 SemanticTarget")
        if self.timeout_s <= 0:
            raise ActionRequestError(f"ActionRequest.timeout_s 必须为正数: {self.timeout_s}")
        if self.attempt < 1:
            raise ActionRequestError(f"ActionRequest.attempt 至少为 1: {self.attempt}")
        payload = dict(self.parameters)
        forbidden = sorted(key for key in payload if str(key).lower() in FORBIDDEN_PARAMETER_KEYS)
        if forbidden:
            raise ActionRequestError(
                f"ActionRequest.parameters 不允许携带坐标字段: {forbidden}"
                f"（定位必须通过语义目标：语义名或 x_ratio / y_ratio）"
            )
        if not str(self.created_at).strip():
            object.__setattr__(self, "created_at", now_iso())

    @classmethod
    def from_task_request(
        cls, source: Any, *, fallback_type: ActionType | None = None
    ) -> ActionRequest:
        """从 **Task 侧的请求对象**构造（字段视图 duck typing，不 import ``src.task``）。

        Task 层（Phase 6.2）的 ``ActionRequest`` 具备 ``action_id`` / ``kind`` / ``target`` /
        ``params`` / ``task_id`` / ``step_id`` / ``timeout_s`` / ``attempt`` 字段；
        两层的 ``ActionOutcome`` 都有 ``status`` / ``reason`` / ``is_failure``，
        因此 :class:`~src.action.dispatcher.ActionDispatcher` 可以直接充当 Task 的动作端口。

        Args:
            source: Task 侧请求对象（或映射 / 本层 :class:`ActionRequest`）。
            fallback_type: 无法识别 ``kind`` 时使用的默认类型。

        Returns:
            本层的 :class:`ActionRequest`。

        Raises:
            ActionRequestError: 类型未知且没有 ``fallback_type``、字段缺失或含坐标参数。
        """
        if isinstance(source, ActionRequest):
            return source
        raw_type = _text(_pick(source, "kind", "action_type", "type"))
        if raw_type:
            try:
                action_type = ActionType(raw_type)
            except ValueError as exc:
                if fallback_type is None:
                    raise ActionRequestError(
                        f"未知动作类型: {raw_type!r}"
                        f"（可选: {', '.join(item.value for item in ActionType)}）"
                    ) from exc
                action_type = fallback_type
        elif fallback_type is not None:
            action_type = fallback_type
        else:
            raise ActionRequestError("无法从来源识别 action_type（缺少 kind / action_type 字段）")
        raw_target = _pick(source, "target", "action_target")
        target = SemanticTarget.from_mapping(raw_target) if isinstance(raw_target, Mapping) else None
        if target is None and raw_target is not None and not isinstance(raw_target, Mapping):
            target = SemanticTarget.from_name(str(raw_target))
        request_id = _text(_pick(source, "action_id", "request_id")) or now_iso()
        return cls(
            request_id=request_id,
            action_type=action_type,
            target=target,
            parameters=dict(_pick(source, "params", "parameters", default={}) or {}),
            created_at=_text(_pick(source, "created_at")) or now_iso(),
            task_id=_text(_pick(source, "task_id")),
            step_id=_text(_pick(source, "step_id")),
            timeout_s=float(_pick(source, "timeout_s", default=1.0) or 1.0),
            attempt=int(_pick(source, "attempt", default=1) or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典（不含截图 / 用户数据 / 像素坐标）。"""
        return {
            "request_id": self.request_id,
            "action_type": self.action_type.value,
            "target": self.target.to_dict() if self.target is not None else None,
            "parameters": dict(self.parameters),
            "created_at": self.created_at,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "timeout_s": self.timeout_s,
            "attempt": self.attempt,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        target = self.target.describe() if self.target is not None else "no-target"
        return (
            f"ActionRequest(id={self.request_id}, type={self.action_type.value}, "
            f"{target}, attempt={self.attempt})"
        )


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """一次动作请求的最终结果（Task 侧只依赖 ``status`` / ``reason`` / ``is_failure``）。

    Attributes:
        status: 处理结果状态。
        reason: 人类可读原因（拒绝 / 失败说明）。
        latency_ms: 本次处理耗时（毫秒，含门控 / 解析 / 执行）。
        resolved_target: 解析结果（被拒绝时为已解析到的目标或 ``None``）。
        executed_at: 实际执行时间（ISO 8601）；未执行时为 ``None``。
        detail: 附加信息（不含截图与用户数据）。
    """

    status: ActionStatus
    reason: str = ""
    latency_ms: float = 0.0
    resolved_target: ResolvedTarget | None = None
    executed_at: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        """是否失败（被拒绝或执行出错）。"""
        return self.status in (ActionStatus.REJECTED, ActionStatus.FAILED)

    @property
    def is_executed(self) -> bool:
        """是否真正执行了输入。"""
        return self.status is ActionStatus.EXECUTED

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "status": self.status.value,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 3),
            "resolved_target": self.resolved_target.to_dict() if self.resolved_target else None,
            "executed_at": self.executed_at,
            "detail": dict(self.detail),
        }

    def describe(self) -> str:
        """返回可读描述。"""
        suffix = f"（{self.reason}）" if self.reason else ""
        target = f" @ {self.resolved_target.describe()}" if self.resolved_target else ""
        return f"{self.status.value}{suffix}, {self.latency_ms:.1f}ms{target}"


def outcome_from_any(value: Any) -> ActionOutcome | None:
    """把任意"结果对象"（duck typing）转成本层 :class:`ActionOutcome`。

    Args:
        value: 具备 ``status`` / ``reason`` 字段的对象或映射。

    Returns:
        本层结果对象；无法识别时返回 ``None``。
    """
    if isinstance(value, ActionOutcome):
        return value
    status = _text(_pick(value, "status"))
    if not status:
        return None
    try:
        parsed = ActionStatus(status)
    except ValueError:
        return None
    return ActionOutcome(
        status=parsed,
        reason=_text(_pick(value, "reason")),
        latency_ms=float(_pick(value, "latency_ms", default=0.0) or 0.0),
    )


#: 计时函数类型（注入便于测试）
ClockFn = Callable[[], float]
