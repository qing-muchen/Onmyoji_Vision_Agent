"""语义目标解析（Phase 7）：:class:`SemanticTarget` → :class:`ResolvedTarget`。

职责边界（**这是本阶段最重要的架构约束**）：

* Resolver **不读 Frame**、**不调用 detector**、**不属于 vision**；
* Resolver 的输入只有 :class:`~src.action.models.SemanticTarget`（语义名或窗口相对比例），
  输出只有 :class:`~src.action.models.ResolvedTarget`（窗口相对位置 + 置信度 + 来源）；
* Phase 7 的默认实现是**静态注册表**（:class:`RegistryTargetResolver`）；
  Phase 7.1 的 **Vision-aware Resolver** 只需实现同一个 :class:`TargetResolver` 协议
  （输入语义名，输出解析结果，允许失败），无需修改 Dispatcher / Guard / Backend。

解析失败（未注册、比例缺失）一律返回 ``None``：调用方（Dispatcher）据此拒绝动作，
**绝不猜测坐标**（没有 fallback click）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.action.models import ResolvedTarget, SemanticTarget
from src.action.registry import TargetRegistry
from src.utils.logger import get_logger

__all__ = [
    "EXPLICIT_RATIO_TARGET",
    "RegistryTargetResolver",
    "TargetResolver",
    "SemanticTarget",
]

logger = get_logger("action.resolver")

#: 显式比例目标在解析结果中的名字标记
EXPLICIT_RATIO_TARGET: str = "<explicit-ratio>"


@runtime_checkable
class TargetResolver(Protocol):
    """语义目标解析器协议（Phase 7.1 的 vision-aware 实现需满足此协议）。"""

    def resolve(self, target: SemanticTarget | None, /) -> ResolvedTarget | None:
        """解析语义目标。

        Args:
            target: 语义目标（``None`` 表示无目标，仅 ``wait`` 允许）。

        Returns:
            解析结果；无法解析时返回 ``None``（调用方必须据此拒绝动作）。
        """
        ...


class RegistryTargetResolver:
    """基于静态注册表的解析器（Phase 7 默认实现）。

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
        if target is None:
            return None
        if target.is_ratio:
            assert target.x_ratio is not None and target.y_ratio is not None  # noqa: S101
            return ResolvedTarget(
                target=EXPLICIT_RATIO_TARGET,
                x_ratio=target.x_ratio,
                y_ratio=target.y_ratio,
                confidence=1.0,
                source="explicit_ratio",
                calibrated=self._trust_explicit_ratio,
            )
        resolved = self._registry.resolve(target.name)
        if resolved is None:
            logger.debug("语义目标未注册，无法解析: %s", target.name)
        return resolved

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
