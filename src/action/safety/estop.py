"""Action 层的紧急停止（E-STOP，Phase 7）。

与 Task 层（``src/task/estop.py``）**同机制、同哨兵文件、独立实现**：Action 层不允许
import ``src.task``（依赖边界），但两侧读同一个 ``logs/ESTOP``：

```text
人工 / 上层                     Task 侧                    Action 侧
──────────                     ───────                    ─────────
EmergencyStop.trigger(...) ─┐
                            ├──▶ TaskEngine.on_*   ──▶   SafetyGuard.check_pre
echo x > logs/ESTOP       ──┘    任务 FAILED                动作 REJECTED
```

要点：

* **两种触发方式**：API（:meth:`EmergencyStop.trigger`）与**文件哨兵**（默认 ``logs/ESTOP``）；
* **禁止全局热键**：不注册任何键盘钩子（无 ``pynput`` / ``keyboard`` / Win32 钩子）；
* **动作前必查**：:class:`~src.action.safety.guard.SafetyGuard` 在每次动作前检查（第一个门）；
* **不自动复位**：必须人工显式 :meth:`EmergencyStop.reset`（默认同时删除哨兵文件）；
* **只描述状态**：不停止线程、不修改任务状态（由调用方响应）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from src.core.paths import resolve_from_project_root
from src.utils.logger import get_logger

__all__ = [
    "DEFAULT_ACTION_SENTINEL_PATH",
    "EmergencyStop",
    "EmergencyStopSource",
    "EmergencyStopState",
]

logger = get_logger("action.estop")

#: 默认哨兵路径（与 Task 层一致：人工创建一个文件即可同时阻止 Task 与 Action）
DEFAULT_ACTION_SENTINEL_PATH: str = "logs/ESTOP"


class EmergencyStopSource(StrEnum):
    """触发来源（与 Task 层 ``EStopSource`` 对齐）。"""

    #: 通过 API 触发
    API = "api"
    #: 通过文件哨兵触发
    SENTINEL = "sentinel"


def _now_iso() -> str:
    """返回本地时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class EmergencyStopState:
    """E-STOP 状态快照。

    Attributes:
        triggered: 是否触发。
        reason: 触发原因。
        source: 触发来源（``api`` / ``sentinel``）。
        triggered_at: 触发时刻（ISO 8601）。
    """

    triggered: bool = False
    reason: str = ""
    source: str = ""
    triggered_at: str = ""

    def to_dict(self) -> dict[str, object]:
        """返回可 JSON 序列化的字典。"""
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "source": self.source,
            "triggered_at": self.triggered_at,
        }

    def describe(self) -> str:
        """返回可读描述。"""
        if not self.triggered:
            return "estop=inactive"
        return f"estop=active source={self.source} reason={self.reason} at {self.triggered_at}"


class EmergencyStop:
    """紧急停止开关（API + 文件哨兵；禁止热键）。

    Args:
        path: 哨兵文件路径；省略时使用 ``logs/ESTOP``（相对项目根解析）。
        now: 时间戳函数（可在测试中注入）。
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        now: Callable[[], str] = _now_iso,
    ) -> None:
        self._path = (
            resolve_from_project_root(path)
            if path is not None
            else resolve_from_project_root(DEFAULT_ACTION_SENTINEL_PATH)
        )
        self._now = now
        self._triggered = False
        self._reason = ""
        self._source = ""
        self._triggered_at = ""

    # ---- 查询 ---------------------------------------------------------------

    @property
    def path(self) -> Path:
        """哨兵文件路径。"""
        return self._path

    @property
    def is_triggered(self) -> bool:
        """API 侧是否已触发（不含哨兵）。"""
        return self._triggered

    @property
    def reason(self) -> str:
        """触发原因（未触发时为空串）。"""
        return self._reason

    @property
    def source(self) -> str:
        """触发来源（未触发时为空串）。"""
        return self._source

    @property
    def state(self) -> EmergencyStopState:
        """状态快照（不含哨兵检查）。"""
        return EmergencyStopState(
            triggered=self._triggered,
            reason=self._reason,
            source=self._source,
            triggered_at=self._triggered_at,
        )

    # ---- 触发与检查 ---------------------------------------------------------

    def trigger(
        self, reason: str = "manual", *, source: EmergencyStopSource = EmergencyStopSource.API
    ) -> None:
        """触发紧急停止（幂等：保留首次原因与时刻）。

        Args:
            reason: 触发原因。
            source: 触发来源。
        """
        if self._triggered:
            return
        self._triggered = True
        self._reason = reason
        self._source = source.value
        self._triggered_at = self._now()
        logger.error("Action E-STOP 已触发: %s（source=%s）", reason, source.value)

    def check_sentinel(self) -> bool:
        """检查文件哨兵是否存在。

        Returns:
            ``True`` 表示哨兵文件存在（若此前未触发，则记为哨兵触发）。
        """
        if not self._path.exists():
            return False
        if not self._triggered:
            self.trigger(
                f"sentinel file present: {self._path}", source=EmergencyStopSource.SENTINEL
            )
        return True

    def check(self) -> bool:
        """检查是否处于紧急停止状态（API 触发或哨兵存在）。

        Returns:
            ``True`` 表示必须立即拒绝动作。
        """
        if self._triggered:
            return True
        return self.check_sentinel()

    # ---- 人工复位 -----------------------------------------------------------

    def create_sentinel(self, reason: str = "manual") -> Path:
        """创建哨兵文件（便于人工 / 脚本触发）。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(f"{reason}\n", encoding="utf-8")
        logger.warning("已创建 Action E-STOP 哨兵文件: %s", self._path)
        return self._path

    def clear_sentinel(self) -> bool:
        """删除哨兵文件（若存在）。"""
        try:
            self._path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:  # pragma: no cover - 文件系统异常
            logger.warning("删除 Action E-STOP 哨兵失败: %s", exc)
            return False
        return True

    def reset(self, *, remove_sentinel: bool = True) -> None:
        """复位（**必须显式调用**）。

        Args:
            remove_sentinel: 是否同时删除哨兵文件。
        """
        if remove_sentinel:
            self.clear_sentinel()
        if self._triggered:
            logger.warning("Action E-STOP 已复位（此前原因: %s）", self._reason)
        self._triggered = False
        self._reason = ""
        self._source = ""
        self._triggered_at = ""

    def describe(self) -> str:
        """返回可读描述。"""
        return f"EmergencyStop(path={self._path}, {self.state.describe()})"

    def __repr__(self) -> str:
        return self.describe()
