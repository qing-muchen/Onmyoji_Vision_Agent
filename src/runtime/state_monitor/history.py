"""状态历史持久化（Phase 3.1）。

把监控过程中产生的 :class:`~src.runtime.state_monitor.events.StateEvent` 写成 JSON：

```json
{
  "started_at": "2025-01-01T12:00:00+08:00",
  "updated_at": "2025-01-01T12:05:30+08:00",
  "interval_ms": 500,
  "cycles": 660,
  "skipped": 3,
  "initial_state": "unknown",
  "events": [{"previous_state": "unknown", "current_state": "transition",
              "timestamp": "2025-01-01T12:03:11+08:00", "confidence": 0.9412}]
}
```

写入策略（配置驱动）：

* **每次事件即时落盘**——事件频率远低于监控周期，代价很小，但进程被中断也不会丢事件；
* 采用**原子写**（同目录临时文件 + ``replace``），避免读到半截 JSON；
* ``max_events`` 限制事件数量，超出后丢弃**最旧**事件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.config import RuntimeStateMonitorConfig
from src.core.exceptions import MonitorError
from src.core.paths import resolve_from_project_root
from src.runtime.state_monitor.events import StateEvent
from src.utils.logger import get_logger
from src.vision.state.models import StateName

__all__ = ["StateHistoryStore"]

#: 原子写使用的临时文件后缀
TEMP_SUFFIX: str = ".tmp"
#: JSON 缩进（便于人工查看历史文件）
JSON_INDENT: int = 2

logger = get_logger("runtime.state_monitor.history")


class StateHistoryStore:
    """状态历史的内存缓冲 + JSON 持久化。

    Args:
        config: 状态监控配置（``history_path`` / ``save_history`` / ``max_events``）。
    """

    def __init__(self, config: RuntimeStateMonitorConfig) -> None:
        self._config = config
        self._path = resolve_from_project_root(config.history_path)
        self._events: list[StateEvent] = []
        self._initial_state: StateName | None = None
        self._started_at: str | None = None
        self._updated_at: str | None = None
        self._interval_ms: int = config.interval_ms
        self._cycles: int = 0
        self._skipped: int = 0

    @property
    def config(self) -> RuntimeStateMonitorConfig:
        """当前配置。"""
        return self._config

    @property
    def path(self) -> Path:
        """历史文件绝对路径。"""
        return self._path

    @property
    def events(self) -> tuple[StateEvent, ...]:
        """已记录的事件（只读视图）。"""
        return tuple(self._events)

    @property
    def event_count(self) -> int:
        """事件数量。"""
        return len(self._events)

    @property
    def initial_state(self) -> StateName | None:
        """本次监控首次识别到的状态（基线，不产生事件）。"""
        return self._initial_state

    @property
    def is_started(self) -> bool:
        """是否已调用 :meth:`start`。"""
        return self._started_at is not None

    def start(self, *, timestamp: str, interval_ms: int | None = None) -> None:
        """开始一次监控记录（清空上一轮内存数据）。

        Args:
            timestamp: 开始时间（ISO 8601）。
            interval_ms: 本轮使用的监控周期；省略时沿用配置值。
        """
        self._events.clear()
        self._initial_state = None
        self._started_at = timestamp
        self._updated_at = timestamp
        self._interval_ms = self._config.interval_ms if interval_ms is None else interval_ms
        self._cycles = 0
        self._skipped = 0
        logger.info("状态历史记录开始：path=%s, interval=%dms", self._path, self._interval_ms)

    def set_initial_state(self, state: StateName) -> None:
        """记录本次监控的初始状态（首次识别结果，不产生事件）。

        Args:
            state: 首次识别到的状态。
        """
        self._initial_state = state

    def update_progress(self, *, cycles: int, skipped: int, timestamp: str) -> None:
        """更新运行进度计数（只改内存，不落盘）。

        Args:
            cycles: 已完成的周期数。
            skipped: 因帧不可用而跳过的周期数。
            timestamp: 当前时间（ISO 8601），用于 ``updated_at``。
        """
        self._cycles = cycles
        self._skipped = skipped
        self._updated_at = timestamp

    def append(self, event: StateEvent, *, timestamp: str | None = None) -> None:
        """追加一条事件并**立即落盘**。

        Args:
            event: 状态变化事件。
            timestamp: 落盘时刻（ISO 8601）；省略时沿用上次的 ``updated_at``。
        """
        self._events.append(event)
        if len(self._events) > self._config.max_events:
            # 保留最近 max_events 条（丢弃最旧）
            del self._events[: len(self._events) - self._config.max_events]
            logger.debug("事件数量超过上限 %d，已丢弃最旧事件", self._config.max_events)
        if timestamp is not None:
            self._updated_at = timestamp
        self.save()

    def snapshot(self) -> dict[str, Any]:
        """组装待写入的历史结构。"""
        return {
            "started_at": self._started_at,
            "updated_at": self._updated_at,
            "interval_ms": self._interval_ms,
            "cycles": self._cycles,
            "skipped": self._skipped,
            "initial_state": self._initial_state.value if self._initial_state else None,
            "events": [event.to_dict() for event in self._events],
        }

    def save(self) -> None:
        """把当前历史原子写入 JSON 文件。

        Raises:
            MonitorError: 目录创建失败或文件写入失败。
        """
        if not self._config.save_history:
            logger.debug("save_history=false，跳过历史落盘")
            return
        payload = json.dumps(self.snapshot(), ensure_ascii=False, indent=JSON_INDENT)
        temporary = self._path.with_name(self._path.name + TEMP_SUFFIX)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(payload + "\n", encoding="utf-8")
            temporary.replace(self._path)
        except OSError as exc:
            raise MonitorError(f"状态历史写入失败: {self._path}: {exc}") from exc

    def read(self) -> dict[str, Any]:
        """读取已落盘的历史内容（不修改任何文件）。

        Returns:
            历史结构字典。

        Raises:
            MonitorError: 文件不存在或不是合法 JSON。
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MonitorError(f"状态历史读取失败: {self._path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MonitorError(f"状态历史不是合法 JSON: {self._path}: {exc}") from exc
        if not isinstance(data, dict):
            raise MonitorError(f"状态历史根节点必须是对象: {self._path}")
        return data

    def read_events(self) -> list[StateEvent]:
        """读取并重建事件列表（便于测试与离线分析）。

        Returns:
            事件列表。

        Raises:
            MonitorError: 文件缺失、JSON 非法或事件字段不合法。
        """
        data = self.read()
        raw_events = data.get("events", [])
        if not isinstance(raw_events, list):
            raise MonitorError(f"状态历史 events 必须是数组: {self._path}")
        try:
            return [StateEvent.from_dict(item) for item in raw_events]
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitorError(f"状态历史事件字段非法: {self._path}: {exc}") from exc
