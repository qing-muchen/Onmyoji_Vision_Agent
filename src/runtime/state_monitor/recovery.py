"""失败恢复策略（Phase 5.4）：重试 / 次数上限 / 指数退避。

Runtime 长时间运行时一定会遇到"暂时采不到"的情况：游戏没开、窗口被最小化、dxcam 设备
会话失效、显示器切换……这些**不该让循环崩溃，也不该忙等**。本模块把"怎么重试"抽成策略：

```text
失败（capture failure / window lost / temporary exception）
   │  attempt += 1
   ├─ attempt <= max_retry            → RETRY：按指数退避等待（backoff_s × factor^(n-1)，封顶 max_backoff_s）
   └─ attempt >  max_retry            → EXHAUSTED：交给调用方（Runtime 进入 FAILED 并结束运行）
成功（取到帧）
   └─ attempt 清零（RECOVERING → RUNNING）
```

设计要点：

* **不会死循环**：``max_retry`` 是硬上限；达到上限后不再重试（``EXHAUSTED``）；
* **不会忙等**：每次重试前有退避（``sleep`` 注入，可在测试中用假时钟验证）；
* **可选能力（默认关闭）**：``RecoveryPolicy.enabled`` 默认 ``False`` = **保持 Phase 5.2/5.3
  的既有行为**（失败只记入 ``skipped`` 并继续下一轮，不结束运行）。开启后失败会按策略退避，
  并在超过上限时结束当前运行（``StopReason.RECOVERY_EXHAUSTED``）。长跑场景建议在配置中开启
  （``runtime.recovery.enabled: true``）；
* **只做策略与计数**：不采集、不判定、不改变窗口与后端（由
  :mod:`~src.runtime.state_monitor.frame_source` 负责重建）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from src.core.config import RuntimeRecoveryConfig
from src.utils.logger import get_logger

__all__ = [
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryManager",
    "RecoveryPolicy",
]

logger = get_logger("runtime.state_monitor.recovery")


class RecoveryAction(StrEnum):
    """一次失败处理的动作。"""

    #: 无需恢复（本轮成功，或未启用恢复策略）
    NONE = "none"
    #: 退避后重试
    RETRY = "retry"
    #: 重试次数用尽（调用方应结束运行或人工介入）
    EXHAUSTED = "exhausted"


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    """恢复策略参数。

    Attributes:
        enabled: 是否启用恢复策略（默认 ``False`` = 保持既有行为）。
        max_retry: 连续失败的最大重试次数；``<= 0`` 表示不放弃（无限退避，仍不会忙等）。
        backoff_s: 首次重试的等待时长（秒）。
        backoff_factor: 退避倍数（指数增长）。
        max_backoff_s: 单次等待上限（秒）。
    """

    enabled: bool = False
    max_retry: int = 3
    backoff_s: float = 0.5
    backoff_factor: float = 2.0
    max_backoff_s: float = 8.0

    @classmethod
    def from_config(cls, config: RuntimeRecoveryConfig) -> RecoveryPolicy:
        """按配置构造策略。

        Args:
            config: ``runtime.recovery`` 配置段。

        Returns:
            策略实例。
        """
        return cls(
            enabled=config.enabled,
            max_retry=config.max_retry,
            backoff_s=config.backoff_s,
            backoff_factor=config.backoff_factor,
            max_backoff_s=config.max_backoff_s,
        )

    @property
    def unlimited(self) -> bool:
        """是否不放弃重试（``max_retry <= 0``）。"""
        return self.max_retry <= 0

    def delay_for(self, attempt: int) -> float:
        """计算第 ``attempt`` 次重试的退避时长。

        Args:
            attempt: 失败序号（从 1 开始）。

        Returns:
            等待时长（秒），不超过 :attr:`max_backoff_s`。
        """
        if attempt <= 0:
            return 0.0
        delay = self.backoff_s * (self.backoff_factor ** (attempt - 1))
        return max(0.0, min(self.max_backoff_s, delay))

    def describe(self) -> str:
        """返回可读描述。"""
        limit = "unlimited" if self.unlimited else str(self.max_retry)
        return (
            f"RecoveryPolicy(enabled={self.enabled}, max_retry={limit}, "
            f"backoff={self.backoff_s}s ×{self.backoff_factor} (max {self.max_backoff_s}s))"
        )


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """一次失败处理的结果。

    Attributes:
        action: 采取的动作。
        reason: 失败原因（来自帧来源的 ``SkipReason`` 或判定异常说明）。
        attempt: 连续失败序号（从 1 开始）。
        delay_s: 本次退避时长（未启用的策略与 ``EXHAUSTED`` 时为 ``0.0``）。
    """

    action: RecoveryAction
    reason: str | None = None
    attempt: int = 0
    delay_s: float = 0.0

    @property
    def retrying(self) -> bool:
        """是否处于"退避重试"中。"""
        return self.action is RecoveryAction.RETRY

    @property
    def exhausted(self) -> bool:
        """重试是否已用尽。"""
        return self.action is RecoveryAction.EXHAUSTED

    def describe(self) -> str:
        """返回可读描述。"""
        if self.action is RecoveryAction.NONE:
            return "recovery=none"
        suffix = f" backoff={self.delay_s:.2f}s" if self.delay_s else ""
        return (
            f"recovery={self.action.value} attempt={self.attempt} "
            f"reason={self.reason or 'unknown'}{suffix}"
        )


class RecoveryManager:
    """按 :class:`RecoveryPolicy` 处理连续失败（计数 + 退避 + 上限）。

    典型用法::

        manager = RecoveryManager(policy, sleep=monitor_sleep)
        decision = manager.register_failure("capture_failed")
        if decision.exhausted:
            monitor.stop()
        ...
        manager.register_success()      # 连续失败计数清零

    Args:
        policy: 恢复策略。
        sleep: 等待函数（秒）；可在测试中注入假时钟。
    """

    def __init__(
        self,
        policy: RecoveryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._policy = policy if policy is not None else RecoveryPolicy()
        self._sleep = sleep
        self.reset()

    @property
    def policy(self) -> RecoveryPolicy:
        """当前策略。"""
        return self._policy

    @property
    def attempts(self) -> int:
        """当前连续失败次数。"""
        return self._attempts

    @property
    def count(self) -> int:
        """累计恢复次数（进入恢复态的次数，跨成功/失败）。"""
        return self._count

    @property
    def last_reason(self) -> str | None:
        """最近一次恢复原因。"""
        return self._last_reason

    @property
    def last_delay_s(self) -> float:
        """最近一次退避时长（秒）。"""
        return self._last_delay_s

    @property
    def total_backoff_s(self) -> float:
        """累计退避等待时长（秒）。"""
        return self._total_backoff_s

    @property
    def is_exhausted(self) -> bool:
        """是否已用尽重试。"""
        return self._exhausted

    @property
    def enabled(self) -> bool:
        """恢复策略是否启用。"""
        return self._policy.enabled

    def reset(self) -> None:
        """清空计数与状态（用于重新开始一次运行）。"""
        self._attempts = 0
        self._count = 0
        self._last_reason: str | None = None
        self._last_delay_s = 0.0
        self._total_backoff_s = 0.0
        self._exhausted = False

    def register_failure(self, reason: str | None) -> RecoveryDecision:
        """记录一次采集/判定失败，并按策略退避。

        Args:
            reason: 失败原因（``SkipReason`` 值或异常说明）。

        Returns:
            处理决定；``RETRY`` 时**已经等待**了 ``delay_s``（``sleep`` 注入）。
        """
        if not self._policy.enabled:
            return RecoveryDecision(RecoveryAction.NONE, reason, 0, 0.0)

        self._attempts += 1
        self._count += 1
        self._last_reason = reason
        if not self._policy.unlimited and self._attempts > self._policy.max_retry:
            self._exhausted = True
            logger.error(
                "恢复重试用尽（连续失败 %d 次，原因=%s）", self._attempts, reason or "unknown"
            )
            return RecoveryDecision(RecoveryAction.EXHAUSTED, reason, self._attempts, 0.0)

        delay = self._policy.delay_for(self._attempts)
        self._last_delay_s = delay
        self._total_backoff_s += delay
        logger.warning(
            "采集失败（第 %d/%s 次，原因=%s），退避 %.2fs 后重试",
            self._attempts,
            "∞" if self._policy.unlimited else str(self._policy.max_retry),
            reason or "unknown",
            delay,
        )
        if delay > 0:
            self._sleep(delay)
        return RecoveryDecision(RecoveryAction.RETRY, reason, self._attempts, delay)

    def register_success(self) -> bool:
        """记录一次成功（连续失败计数清零）。

        Returns:
            ``True`` 表示本次成功是从"恢复中"恢复过来的（调用方应把生命周期切回
            ``RUNNING``）。
        """
        recovered = self._attempts > 0 and not self._exhausted
        self._attempts = 0
        if recovered:
            logger.info("采集已恢复（累计恢复 %d 次）", self._count)
        return recovered

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"RecoveryManager(attempts={self._attempts}, recoveries={self._count}, "
            f"last_reason={self._last_reason or 'n/a'}, total_backoff={self._total_backoff_s:.2f}s, "
            f"exhausted={self._exhausted}, {self._policy.describe()})"
        )
