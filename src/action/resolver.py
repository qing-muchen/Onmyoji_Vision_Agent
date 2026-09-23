"""语义目标解析（Phase 7：静态注册表；Phase 7.1：Vision-aware）。

职责边界（**本层最重要的架构约束**）：

* Resolver **不读 Frame**、**不调用 detector / OCR / EvidenceRegistry / StateClassifier**、
  **不属于 vision**；
* Resolver 的输入只有 :class:`~src.action.models.SemanticTarget`（语义名或窗口相对比例），
  输出只有 :class:`~src.action.models.ResolvedTarget`（窗口相对位置 + 置信度 + 来源）；
* 两种实现：

| 实现 | 定位来源 | 使用场景 |
| --- | --- | --- |
| :class:`RegistryTargetResolver` | 静态注册表（人工标定） | 默认（Phase 7.0 行为不变） |
| :class:`VisionAwareTargetResolver` | **外部注入的视觉快照**（Phase 7.1） | 由视觉结果动态定位 |

解析失败（未注册、快照缺失、**过期**、置信度不足、坐标非法）一律**不返回位置**：
调用方（Dispatcher）据此拒绝动作，**绝不猜测坐标、绝不使用默认值、绝无 fallback**。

失败原因通过 :meth:`resolve_detail` 以 :class:`TargetResolution` 暴露（**增量能力**）：
:class:`TargetResolver` 协议保持与 Phase 7.0 完全一致，既有调用方零改动。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any, Protocol, runtime_checkable

from src.action.models import (
    ActionTargetError,
    ResolvedTarget,
    SemanticTarget,
    now_iso,
)
from src.action.registry import TargetRegistry
from src.core.vision_target import TargetSnapshot, VisionTargetCandidate
from src.utils.logger import get_logger

__all__ = [
    "DEFAULT_VISION_TARGET_MIN_CONFIDENCE",
    "DEFAULT_VISION_TARGET_TTL_S",
    "EXPLICIT_RATIO_TARGET",
    "RegistryTargetResolver",
    "ResolverWithReason",
    "SemanticTarget",
    "TargetResolution",
    "TargetResolutionReason",
    "TargetResolver",
    "VisionAwareTargetResolver",
    "VISION_TARGET_SOURCE",
    "resolver_from_config",
]

logger = get_logger("action.resolver")

#: 显式比例目标在解析结果中的名字标记
EXPLICIT_RATIO_TARGET: str = "<explicit-ratio>"

#: 视觉来源标记（写入 :attr:`ResolvedTarget.source`）
VISION_TARGET_SOURCE: str = "vision"

#: 视觉候选的默认有效期（秒）：默认 500ms 节拍下约 3 个周期
DEFAULT_VISION_TARGET_TTL_S: float = 1.5

#: 视觉候选的默认最低置信度（与 evidence 规则下限一致）
DEFAULT_VISION_TARGET_MIN_CONFIDENCE: float = 0.6


class TargetResolutionReason(StrEnum):
    """解析结果的原因（失败原因可观测，便于上层决策与排障）。"""

    #: 解析成功
    RESOLVED = "resolved"
    #: 请求本身不带目标（``wait``）
    NO_TARGET = "no_target"
    #: 视觉快照中没有该目标（未注册 / 未检出 / 快照为空）
    TARGET_NOT_FOUND = "target_not_found"
    #: 目标存在但已过期（超过 TTL）
    TARGET_EXPIRED = "target_expired"
    #: 目标置信度不足
    LOW_CONFIDENCE = "low_confidence"
    #: 目标数据非法（比例越界 / NaN / 无法构造解析结果）
    INVALID_TARGET = "invalid_target"
    #: 解析器内部异常（已隔离）
    RESOLVER_ERROR = "resolver_error"


@dataclass(frozen=True, slots=True)
class TargetResolution:
    """一次解析的完整结果（成功 + 原因 + 细节）。

    Attributes:
        resolved: 解析出的目标位置；失败时为 ``None``。
        reason: 原因（成功时为 :attr:`TargetResolutionReason.RESOLVED`）。
        detail: 附加信息（如 ``age_s`` / ``confidence`` / ``ttl_s``），不含图像与用户数据。
    """

    resolved: ResolvedTarget | None = None
    reason: TargetResolutionReason = TargetResolutionReason.RESOLVED
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.resolved is not None and self.reason is not TargetResolutionReason.RESOLVED:
            object.__setattr__(self, "reason", TargetResolutionReason.RESOLVED)

    @property
    def ok(self) -> bool:
        """是否解析成功。"""
        return self.resolved is not None

    @classmethod
    def success(cls, resolved: ResolvedTarget) -> TargetResolution:
        """构造成功结果。"""
        return cls(resolved=resolved, reason=TargetResolutionReason.RESOLVED)

    @classmethod
    def reject(cls, reason: TargetResolutionReason, **detail: Any) -> TargetResolution:
        """构造失败结果。"""
        return cls(resolved=None, reason=reason, detail=dict(detail))

    def message(self) -> str:
        """返回可直接写入 ``ActionOutcome.reason`` 的一句话说明。"""
        if self.ok and self.resolved is not None:
            return f"resolved: {self.resolved.describe()}"
        text = self.reason.value
        if self.detail:
            pairs = ", ".join(f"{key}={value}" for key, value in self.detail.items())
            text = f"{text}（{pairs}）"
        return text

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "reason": self.reason.value,
            "ok": self.ok,
            "resolved": self.resolved.to_dict() if self.resolved is not None else None,
            "detail": dict(self.detail),
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return f"TargetResolution({self.message()})"


@runtime_checkable
class TargetResolver(Protocol):
    """语义目标解析器协议（Phase 7.0 冻结；Phase 7.1 实现必须满足它）。"""

    def resolve(self, target: SemanticTarget | None, /) -> ResolvedTarget | None:
        """解析语义目标。

        Args:
            target: 语义目标（``None`` 表示无目标，仅 ``wait`` 允许）。

        Returns:
            解析结果；无法解析时返回 ``None``（调用方必须据此拒绝动作）。
        """
        ...


@runtime_checkable
class ResolverWithReason(Protocol):
    """可选的增强协议：在 ``resolve()`` 之外提供失败原因。

    实现该协议的解析器可被 :class:`~src.action.dispatcher.ActionDispatcher` 优先使用，
    从而把原因写入 ``ActionOutcome``（``stage`` 仍为 ``target``）。
    """

    def resolve_detail(self, target: SemanticTarget | None, /) -> TargetResolution:
        """解析语义目标并返回原因。"""
        ...


def _explicit_ratio_resolution(
    target: SemanticTarget, *, trusted: bool
) -> TargetResolution:
    """把"调用方显式给出的窗口相对比例"转换为解析结果。

    Args:
        target: 带比例的目标（调用方需保证 ``target.is_ratio``）。
        trusted: 是否把该比例视为可信（决定 ``calibrated``）。

    Returns:
        解析结果；比例越界等异常被转换为 ``invalid_target``。
    """
    try:
        return TargetResolution.success(
            ResolvedTarget(
                target=EXPLICIT_RATIO_TARGET,
                x_ratio=float(target.x_ratio or 0.0),
                y_ratio=float(target.y_ratio or 0.0),
                confidence=1.0,
                source="explicit_ratio",
                calibrated=trusted,
            )
        )
    except (ActionTargetError, ValueError) as exc:  # pragma: no cover - 防御性
        return TargetResolution.reject(
            TargetResolutionReason.INVALID_TARGET, error=str(exc), target=EXPLICIT_RATIO_TARGET
        )


class RegistryTargetResolver:
    """基于静态注册表的解析器（Phase 7.0 默认实现，行为保持不变）。

    Args:
        registry: 目标注册表；省略时使用 :meth:`TargetRegistry.default`（内置占位记录）。
        trust_explicit_ratio: 是否信任调用方直接给出的窗口相对比例。
            **默认 ``False``**：显式比例也标记为"未标定"，真实输入后端会拒绝它
            （安全默认：不允许把猜测的比例直接点下去；dry-run 不受影响）。
    """

    def __init__(
        self,
        registry: TargetRegistry | None = None,
        *,
        trust_explicit_ratio: bool = False,
    ) -> None:
        self._registry = registry if registry is not None else TargetRegistry.default()
        self._trust_explicit_ratio = trust_explicit_ratio

    @property
    def registry(self) -> TargetRegistry:
        """当前注册表。"""
        return self._registry

    @property
    def source(self) -> str:
        """解析来源标记。"""
        return "registry"

    def resolve(self, target: SemanticTarget | None, /) -> ResolvedTarget | None:
        """解析语义目标（注册表查找或显式比例转换）。

        Args:
            target: 语义目标。

        Returns:
            解析结果；未注册 / 无目标时返回 ``None``（并记录 debug 日志）。
        """
        return self.resolve_detail(target).resolved

    def resolve_detail(self, target: SemanticTarget | None, /) -> TargetResolution:
        """解析语义目标并给出原因（Phase 7.1 新增的增强接口）。

        Args:
            target: 语义目标。

        Returns:
            解析结果（含 ``target_not_found`` / ``no_target`` 等失败原因）。
        """
        if target is None:
            return TargetResolution.reject(TargetResolutionReason.NO_TARGET)
        if target.is_ratio:
            return _explicit_ratio_resolution(target, trusted=self._trust_explicit_ratio)
        resolved = self._registry.resolve(target.name)
        if resolved is None:
            logger.debug("语义目标未注册，无法解析: %s", target.name)
            return TargetResolution.reject(
                TargetResolutionReason.TARGET_NOT_FOUND,
                name=target.name,
                origin=self.source,
            )
        return TargetResolution.success(resolved)

    def resolve_name(self, name: str) -> ResolvedTarget | None:
        """按名称解析（便捷方法，等价于 ``resolve(SemanticTarget.from_name(name))``）。"""
        return self._registry.resolve(name)

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"RegistryTargetResolver(source={self.source}, "
            f"trust_explicit_ratio={self._trust_explicit_ratio}, "
            f"entries={len(self._registry)})"
        )

    def __repr__(self) -> str:
        return self.describe()


class VisionAwareTargetResolver:
    """Vision-aware 解析器（Phase 7.1）：用**外部注入的视觉快照**定位语义目标。

    数据流（单向，Action 从不主动读取 vision）：

    ```text
    Vision（adapter，外部）──▶ VisionTargetCandidate[] ──▶ TargetSnapshot
                                                            │ update_snapshot() / update_candidates()
                                                            ▼
                                            VisionAwareTargetResolver.resolve(SemanticTarget)
    ```

    检查顺序（任一失败即返回 ``None`` + 原因，**绝不猜位置**）：

    ```text
    1) 快照中有该目标吗？      否 → target_not_found
    2) 快照过期了吗？          是 → target_expired      （age > vision_target_ttl_s）
    3) 置信度达标吗？          否 → low_confidence      （confidence < minimum_confidence）
    4) 坐标合法吗？            否 → invalid_target      （比例越界 / NaN）
    5) 通过 → ResolvedTarget(source="vision")
    ```

    性能：查询为**一次字典查找 + 常数次比较**（O(1)）；不跑检测、不读截图、不调用模型。

    Args:
        ttl_s: 视觉候选有效期（秒），``0`` 或负数表示不做过期判定。
        minimum_confidence: 最低置信度。
        clock: 单调时钟函数（TTL 判定；测试可注入）。
        aliases: 语义名 → 视觉候选名的别名映射（大小写不敏感）。
        trust_vision_targets: 视觉解析结果是否视为"已标定"（默认 ``True``：
            当帧实测 + 通过 TTL/置信度门；置 ``False`` 时真实后端会拒绝）。
        trust_explicit_ratio: 调用方直接给出的比例是否视为可信（默认 ``False``，同注册表实现）。
        snapshot: 初始快照（可选）。
    """

    def __init__(
        self,
        *,
        ttl_s: float = DEFAULT_VISION_TARGET_TTL_S,
        minimum_confidence: float = DEFAULT_VISION_TARGET_MIN_CONFIDENCE,
        clock: Any = monotonic,
        aliases: Mapping[str, str] | None = None,
        trust_vision_targets: bool = True,
        trust_explicit_ratio: bool = False,
        snapshot: TargetSnapshot | None = None,
    ) -> None:
        self._ttl_s = float(ttl_s)
        self._minimum_confidence = float(minimum_confidence)
        self._clock = clock
        self._aliases = {
            str(key).strip().casefold(): str(value).strip()
            for key, value in (aliases or {}).items()
        }
        self._trust_vision_targets = bool(trust_vision_targets)
        self._trust_explicit_ratio = bool(trust_explicit_ratio)
        self._snapshot: TargetSnapshot | None = snapshot
        self._updates = 0
        self._stale_snapshots = 0

    # ---- 只读 ---------------------------------------------------------------

    @property
    def snapshot(self) -> TargetSnapshot | None:
        """当前视觉快照（未被注入过时为 ``None``）。"""
        return self._snapshot

    @property
    def ttl_s(self) -> float:
        """候选有效期（秒）。"""
        return self._ttl_s

    @property
    def minimum_confidence(self) -> float:
        """最低置信度。"""
        return self._minimum_confidence

    @property
    def source(self) -> str:
        """解析来源标记。"""
        return VISION_TARGET_SOURCE

    @property
    def updates(self) -> int:
        """快照更新次数（观测用）。"""
        return self._updates

    @property
    def stale_snapshots(self) -> int:
        """被拒收的乱序快照数量（观测用；Phase 7.2）。"""
        return self._stale_snapshots

    # ---- 注入（单向：Vision → Resolver）-------------------------------------

    def update_snapshot(
        self,
        snapshot: TargetSnapshot | None,
        *,
        accept_out_of_order: bool = False,
    ) -> TargetSnapshot | None:
        """替换当前视觉快照（**单向注入**：Vision → Resolver）。

        乱序保护（Phase 7.2）：若新旧快照都带 ``frame_id`` 且新快照**更旧**，
        默认拒收并保留较新的快照（避免"迟到的旧帧"把有效位置冲掉）。调用方若确实需要
        覆盖（例如手动调试），可传 ``accept_out_of_order=True``。

        Args:
            snapshot: 新快照；``None`` 表示清空（等价于"没有视觉结果"）。
            accept_out_of_order: 是否接受更旧的帧（默认 ``False``）。

        Returns:
            当前生效的快照（被拒收时返回原有快照）。
        """
        if (
            not accept_out_of_order
            and snapshot is not None
            and self._snapshot is not None
            and snapshot.frame_id is not None
            and self._snapshot.frame_id is not None
            and snapshot.frame_id < self._snapshot.frame_id
        ):
            self._stale_snapshots += 1
            logger.warning(
                "拒收乱序快照: 收到 frame=%s，当前 frame=%s（保留较新者）",
                snapshot.frame_id,
                self._snapshot.frame_id,
            )
            return self._snapshot
        self._snapshot = snapshot
        self._updates += 1
        if snapshot is None:
            logger.debug("视觉快照已清空（目标将无法解析）")
        else:
            logger.debug("视觉快照已更新: %s", snapshot.describe())
        return snapshot

    def update_candidates(
        self,
        candidates: Iterable[VisionTargetCandidate] | Mapping[str, Any],
        *,
        captured_at: str | None = None,
        captured_monotonic: float | None = None,
        frame_id: int | None = None,
        accept_out_of_order: bool = False,
    ) -> TargetSnapshot:
        """用候选集合构造并替换快照（便捷入口）。

        Args:
            candidates: 候选序列或 ``{名称: 候选}`` 映射。
            captured_at: 覆盖快照的 ISO 时间戳（默认取当前时间）。
            captured_monotonic: 覆盖快照的单调时钟时间（默认取 :attr:`_clock`）。
            frame_id: 来源帧序号（乱序保护用）。
            accept_out_of_order: 是否接受更旧的帧（默认 ``False``）。

        Returns:
            新建（或被接受）的快照。
        """
        snapshot = TargetSnapshot.create(
            candidates, frame_id=frame_id, clock=self._clock, now=now_iso
        )
        if captured_at is not None or captured_monotonic is not None:
            snapshot = TargetSnapshot(
                candidates=snapshot.candidates,
                captured_at=captured_at if captured_at is not None else snapshot.captured_at,
                captured_monotonic=(
                    captured_monotonic
                    if captured_monotonic is not None
                    else snapshot.captured_monotonic
                ),
                frame_id=snapshot.frame_id,
            )
        result = self.update_snapshot(snapshot, accept_out_of_order=accept_out_of_order)
        return result if result is not None else snapshot

    def clear(self) -> None:
        """清空快照（目标将无法解析；不删除已配置的别名与阈值）。"""
        self.update_snapshot(None)

    def is_stale(self) -> bool:
        """当前快照是否已过期（无快照视为过期）。"""
        if self._snapshot is None:
            return True
        if self._ttl_s <= 0:
            return False
        return self._snapshot.age_s(clock=self._clock) > self._ttl_s

    # ---- 解析 ---------------------------------------------------------------

    def resolve(self, target: SemanticTarget | None, /) -> ResolvedTarget | None:
        """解析语义目标（协议方法）。

        Args:
            target: 语义目标。

        Returns:
            解析结果；任一检查失败返回 ``None``（原因见 :meth:`resolve_detail`）。
        """
        return self.resolve_detail(target).resolved

    def resolve_detail(self, target: SemanticTarget | None, /) -> TargetResolution:
        """解析语义目标并给出失败原因。

        Args:
            target: 语义目标。

        Returns:
            :class:`TargetResolution`（``target_not_found`` / ``target_expired`` /
            ``low_confidence`` / ``invalid_target`` …）。
        """
        if target is None:
            return TargetResolution.reject(TargetResolutionReason.NO_TARGET)
        if target.is_ratio:
            return _explicit_ratio_resolution(target, trusted=self._trust_explicit_ratio)

        name = target.name
        lookup = self._aliased(name)
        snapshot = self._snapshot
        if snapshot is None:
            return TargetResolution.reject(
                TargetResolutionReason.TARGET_NOT_FOUND,
                name=name,
                origin=self.source,
                snapshot="none",
            )
        candidate = snapshot.get(lookup)
        if candidate is None:
            return TargetResolution.reject(
                TargetResolutionReason.TARGET_NOT_FOUND,
                name=name,
                looked_up=lookup,
                snapshot_names=len(snapshot.names),
            )

        age_s = snapshot.age_s(clock=self._clock)
        if self._ttl_s > 0 and age_s > self._ttl_s:
            return TargetResolution.reject(
                TargetResolutionReason.TARGET_EXPIRED,
                name=candidate.target_name,
                age_s=round(age_s, 3),
                ttl_s=self._ttl_s,
            )

        if float(candidate.confidence) < self._minimum_confidence:
            return TargetResolution.reject(
                TargetResolutionReason.LOW_CONFIDENCE,
                name=candidate.target_name,
                confidence=round(float(candidate.confidence), 3),
                minimum=self._minimum_confidence,
            )

        if not candidate.is_valid:
            return TargetResolution.reject(
                TargetResolutionReason.INVALID_TARGET,
                name=candidate.target_name,
                x_ratio=candidate.x_ratio,
                y_ratio=candidate.y_ratio,
                confidence=candidate.confidence,
            )

        try:
            resolved = ResolvedTarget(
                target=candidate.target_name,
                x_ratio=float(candidate.x_ratio),
                y_ratio=float(candidate.y_ratio),
                confidence=float(candidate.confidence),
                source=VISION_TARGET_SOURCE,
                calibrated=self._trust_vision_targets,
            )
        except (ActionTargetError, ValueError) as exc:  # pragma: no cover - 防御性
            return TargetResolution.reject(
                TargetResolutionReason.INVALID_TARGET,
                name=candidate.target_name,
                error=str(exc),
            )
        return TargetResolution.success(resolved)

    def _aliased(self, name: str) -> str:
        """按别名映射解析候选名（无别名时返回原名）。"""
        return self._aliases.get(name.strip().casefold(), name.strip())

    # ---- 描述 ---------------------------------------------------------------

    def describe(self) -> str:
        """返回可读描述。"""
        snapshot = self._snapshot
        return (
            f"VisionAwareTargetResolver(ttl_s={self._ttl_s}, "
            f"min_confidence={self._minimum_confidence}, "
            f"trust_vision_targets={self._trust_vision_targets}, "
            f"aliases={len(self._aliases)}, "
            f"updates={self._updates}, stale={self._stale_snapshots}, "
            f"snapshot={'none' if snapshot is None else snapshot.count})"
        )

    def __repr__(self) -> str:
        return self.describe()


def resolver_from_config(config: Any) -> TargetResolver:
    """按配置选择解析器实现（**默认 registry**，即 Phase 7.0 行为）。

    把"实现选择"放在本模块（而不是 Dispatcher）：这样
    :mod:`src.action.dispatcher` 无需知道任何视觉概念（它只认识 ``TargetResolver`` 协议）。

    * ``resolver == "vision"`` → :class:`VisionAwareTargetResolver`
      （需由外部通过 ``resolver.update_snapshot()`` 注入视觉快照）；
    * 其它取值（含 ``None``）→ :class:`RegistryTargetResolver`。

    Args:
        config: ``AppConfig.action`` 或 ``None``。

    Returns:
        解析器实例。
    """
    choice = str(getattr(config, "resolver", "registry") or "registry").strip().lower()
    if choice != VISION_TARGET_SOURCE:
        return RegistryTargetResolver()
    return VisionAwareTargetResolver(
        ttl_s=float(getattr(config, "vision_target_ttl_s", DEFAULT_VISION_TARGET_TTL_S)),
        minimum_confidence=float(
            getattr(config, "vision_target_min_confidence", DEFAULT_VISION_TARGET_MIN_CONFIDENCE)
        ),
    )
