"""Runtime 视觉快照桥（Phase 7.3）：把每轮的视觉结果推送到 Action 消费链路。

```text
Runtime Cycle（StateMonitor.run）
  │  frame = source.get_frame()                 ← capture（失败 → 本轮不更新）
  │  processor.detect(frame) → [EvidenceResult] ← 已有视觉流水线（不新增 detector）
  │  classifier.classify(...) → StateResult     ← 状态判定（既有）
  ▼
VisionSnapshotBridge.update(frame=…, evidences=…, frame_id=monitor.cycles, cycle_id=…)
  │  ① VisionTargetAdapter → [VisionTargetCandidate]   （Phase 7.2）
  │  ② TargetSnapshot{frame_id, timestamp, candidates}
  │  ③ resolver.update_snapshot(snapshot)               （单向 push；Phase 7.1）
  ▼
Action：VisionAwareTargetResolver → ActionDispatcher → SafetyGuard → InputBackend（默认 DryRun）
```

**为什么放在 Runtime**：Runtime 是唯一可以同时看到"帧 / 视觉结果 / 动作端口"的编排层
（Action 不 import vision；Vision 不 import action）。本模块只做**搬运与转换**：

* 不新增视觉算法（复用 `EvidenceFrameProcessor.detect()` 的产物）；
* 不调用 `ActionDispatcher`、不产生任何输入（只 push 快照数据）；
* 不做判定（TTL / 置信度 / 合法性仍由 Action 侧的 resolver 与安全门决定）。

**失败策略（fail-safe）**：

| 情况 | 处理 | 后果 |
| --- | --- | --- |
| capture 失败（`frame is None`） | **不更新**（既不 push 空快照，也不保留"假装有效"） | 旧快照自然超时 → Action 侧 `target_expired` |
| 视觉无结果（`evidences` 为空） | push **空快照**（覆盖旧目标） | 目标立即不可解析（`target_not_found`） |
| adapter 异常 | 记录日志 + push **空快照** | 不保留上一帧目标；主循环不受影响 |
| resolver 拒收旧帧（乱序） | 保持 Phase 7.2 的 stale 保护（不覆盖） | 保留较新的快照 |
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import monotonic
from typing import Any

from src.core.vision_target import TargetSnapshot
from src.utils.logger import get_logger
from src.vision.pipeline.target_adapter import VisionTargetAdapter

__all__ = ["VisionBridgeStats", "VisionSnapshotBridge"]

logger = get_logger("runtime.vision_bridge")


@dataclass(slots=True)
class VisionBridgeStats:
    """桥接统计（观测 / 排障；不含截图与用户数据）。"""

    #: ``update()`` 调用次数（= Runtime 轮次）
    updates: int = 0
    #: 被 resolver 接受并生效的快照数
    published: int = 0
    #: 被 resolver 拒收的乱序旧帧数（保持 Phase 7.2 的 stale 保护）
    rejected_stale: int = 0
    #: 推送过程出错（resolver 抛异常，已隔离）
    push_errors: int = 0
    #: capture 失败轮数（本轮不更新快照）
    no_frame: int = 0
    #: adapter 异常次数（已隔离，推送空快照）
    adapter_errors: int = 0
    #: 空快照次数（视觉无结果或 adapter 异常）
    empty_snapshots: int = 0
    #: 累计候选数量
    candidates: int = 0
    #: 最近一轮的帧号 / 轮号 / 候选数
    last_frame_id: int | None = None
    last_cycle_id: int | None = None
    last_candidates: int = 0
    #: 最近一轮的说明（审计用）
    last_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "updates": self.updates,
            "published": self.published,
            "rejected_stale": self.rejected_stale,
            "push_errors": self.push_errors,
            "no_frame": self.no_frame,
            "adapter_errors": self.adapter_errors,
            "empty_snapshots": self.empty_snapshots,
            "candidates": self.candidates,
            "last_frame_id": self.last_frame_id,
            "last_cycle_id": self.last_cycle_id,
            "last_candidates": self.last_candidates,
            "last_reason": self.last_reason,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"VisionBridgeStats(updates={self.updates}, published={self.published}, "
            f"stale={self.rejected_stale}, push_errors={self.push_errors}, "
            f"no_frame={self.no_frame}, adapter_errors={self.adapter_errors}, "
            f"empty={self.empty_snapshots}, candidates={self.candidates})"
        )

    def reset(self) -> None:
        """归零（一次 run 之前调用）。"""
        self.updates = 0
        self.published = 0
        self.rejected_stale = 0
        self.push_errors = 0
        self.no_frame = 0
        self.adapter_errors = 0
        self.empty_snapshots = 0
        self.candidates = 0
        self.last_frame_id = None
        self.last_cycle_id = None
        self.last_candidates = 0
        self.last_reason = ""


class VisionSnapshotBridge:
    """把"每轮的帧与证据"转换成 :class:`TargetSnapshot` 并**单向**推给 resolver。

    Args:
        resolver: 目标解析器（鸭子类型：需要 ``update_snapshot(snapshot)``；
            例如 :class:`~src.action.resolver.VisionAwareTargetResolver`）。
        adapter: 视觉目标适配器；省略时使用默认配置的 :class:`VisionTargetAdapter`。
        clock: 单调时钟（写入快照的 ``captured_monotonic``，与 Action 侧 TTL 同基准）。
        now: ISO 时间戳函数（写入快照的 ``captured_at``）。

    Raises:
        TypeError: ``resolver`` 缺少 ``update_snapshot`` 方法。
    """

    def __init__(
        self,
        *,
        resolver: Any,
        adapter: VisionTargetAdapter | None = None,
        clock: Callable[[], float] = monotonic,
        now: Callable[[], str] | None = None,
    ) -> None:
        updater = getattr(resolver, "update_snapshot", None)
        if not callable(updater):
            raise TypeError(
                f"resolver 必须提供 update_snapshot()（当前: {type(resolver).__name__}）"
            )
        self._resolver = resolver
        self._adapter = adapter if adapter is not None else VisionTargetAdapter()
        self._clock = clock
        self._now = now
        self._stats = VisionBridgeStats()

    # ---- 只读 ---------------------------------------------------------------

    @property
    def resolver(self) -> Any:
        """当前解析器（Action 侧）。"""
        return self._resolver

    @property
    def adapter(self) -> VisionTargetAdapter:
        """当前视觉目标适配器。"""
        return self._adapter

    @property
    def stats(self) -> VisionBridgeStats:
        """桥接统计。"""
        return self._stats

    def reset(self) -> None:
        """重置统计（不改变 resolver 中已有的快照）。"""
        self._stats.reset()

    # ---- 每轮入口 -----------------------------------------------------------

    def update(
        self,
        *,
        frame: Any = None,
        evidences: Iterable[Any] | None = None,
        frame_id: int | None = None,
        cycle_id: int | None = None,
        timestamp: float | None = None,
        frame_size: Any = None,
    ) -> TargetSnapshot | None:
        """处理一轮：转换证据 → 构造快照 → 单向 push。

        Args:
            frame: 本轮的帧（``None`` 表示 capture 失败 → 本轮不更新快照）。
            evidences: 本轮的证据结果（``[]`` / ``None`` 视为"视觉无结果" → 推送空快照）。
            frame_id: 帧号（**必须来自 Runtime 轮次**，不在此生成；见
                :attr:`~src.runtime.state_monitor.monitor.StateMonitor.cycles`）。
            cycle_id: 节拍轮号（``CycleSchedule.cycle_id``；仅用于审计与追踪）。
            timestamp: 覆盖快照时间戳（默认取 ``frame.timestamp``）。
            frame_size: 覆盖帧尺寸（默认从 ``frame`` 读取；读不到则按缺料跳过所有候选）。

        Returns:
            resolver 中当前生效的快照；capture 失败时为 ``None``（未推送）。
        """
        self._stats.updates += 1
        self._stats.last_frame_id = frame_id
        self._stats.last_cycle_id = cycle_id

        # 情况 1：capture 失败 → 本轮不更新（旧快照自然超时，Action 侧给 target_expired）
        if frame is None:
            self._stats.no_frame += 1
            self._stats.last_reason = "no_frame"
            logger.debug("本轮无可用帧（cycle=%s），不更新视觉快照", cycle_id)
            return None

        self._stats.last_reason = "processed"
        active_frame_size = (
            frame_size if frame_size is not None else self._adapter.frame_size_of(frame)
        )
        active_timestamp = (
            timestamp
            if timestamp is not None
            else self._adapter.frame_timestamp_of(frame, default=float(self._clock()))
        )

        # 情况 2/3：正常转换；adapter 异常 → 空快照（不保留上一帧目标）
        try:
            snapshot = self._adapter.snapshot_from_evidence(
                evidences or (),
                frame_size=active_frame_size,
                timestamp=active_timestamp,
                frame_id=frame_id,
                clock=self._clock,
                now=self._now,
            )
        except Exception as exc:  # noqa: BLE001 - 适配器异常绝不影响主循环
            self._stats.adapter_errors += 1
            self._stats.last_reason = "adapter_error"
            logger.error("视觉目标适配失败（已隔离，本轮推送空快照）: %s", exc)
            snapshot = self._empty_snapshot(frame_id=frame_id)

        if snapshot.count == 0:
            self._stats.empty_snapshots += 1

        # 情况 4：resolver 可能拒收乱序旧帧（保持 Phase 7.2 的 stale 保护）
        accepted = self._push(snapshot)

        self._stats.last_candidates = snapshot.count
        self._stats.candidates += snapshot.count
        if accepted is snapshot:
            self._stats.published += 1
        elif accepted is None:
            self._stats.push_errors += 1
        else:
            self._stats.rejected_stale += 1
        return accepted

    # ---- 内部 ---------------------------------------------------------------

    def _push(self, snapshot: TargetSnapshot) -> TargetSnapshot | None:
        """把快照推给 resolver（异常隔离：绝不影响 Runtime 主循环）。

        Returns:
            resolver 中当前生效的快照；推送出错时返回 ``None``。
        """
        try:
            accepted = self._resolver.update_snapshot(snapshot)
        except Exception as exc:  # noqa: BLE001 - Action 侧异常同样隔离
            logger.error("推送视觉快照失败（已隔离）: %s", exc)
            return None
        if accepted is None:  # 解析器清空了快照（异常用法）
            return None
        logger.debug(
            "视觉快照已推送: %s（accepted=%s）",
            snapshot.describe(),
            "self" if accepted is snapshot else "previous",
        )
        return accepted

    def _empty_snapshot(self, *, frame_id: int | None) -> TargetSnapshot:
        """构造空快照（视觉无结果 / 适配失败时使用）。"""
        return TargetSnapshot.create(
            [], frame_id=frame_id, clock=self._clock, now=self._now
        )

    def describe(self) -> str:
        """返回可读描述。"""
        return (
            f"VisionSnapshotBridge(resolver={type(self._resolver).__name__}, "
            f"{self._stats.describe()})"
        )

    def __repr__(self) -> str:
        return self.describe()
