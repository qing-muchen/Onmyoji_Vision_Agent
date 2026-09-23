"""紧急停止（E-STOP，Phase 6.2，架构决策 Q-3）。

失控保护的单点开关：**任何层都可以触发，任何层都可以检查**。

```text
人工 / 上层                     TaskEngine               未来的 Action 层
──────────                     ───────────              ────────────────
EStop.trigger("manual") ─┐
                         ├──▶  每轮 check() ──▶ 任务 FAILED(reason="estop: ...")
echo x > logs/ESTOP     ─┘                          且不再请求任何动作
```

设计要点：

* **两种触发方式**（Q-3）：API（:meth:`EStop.trigger`）与**文件哨兵**
  （默认 ``logs/ESTOP`` 存在即视为触发）—— 不需要热键、不需要交互；
* **禁止全局热键**：本模块**不**注册任何键盘钩子（无 ``pynput`` / ``keyboard`` / Win32 钩子）；
* **双检查点**：Task 每轮检查（引擎调用），Phase 7 的 Action 层在每次请求前也要检查
  （见 Phase 6.0 第 8.3 节的安全设计）；
* **不自动复位**：触发后必须由人工显式 :meth:`EStop.reset`（默认同时删除哨兵文件）；
* **只描述状态**：不停止线程、不修改任务状态（由调用方决定如何响应）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from src.core.paths import resolve_from_project_root
from src.utils.logger import get_logger

__all__ = ["DEFAULT_SENTINEL_PATH", "EStop", "EStopSource", "EStopState"]

logger = get_logger("task.estop")

#: 默认的文件哨兵路径（相对项目根；``logs/`` 已被 .gitignore 忽略）
DEFAULT_SENTINEL_PATH: str = "logs/ESTOP"


class EStopSource(StrEnum):
    """触发来源。"""

    #: 通过 API 触发（上层调用）
    API = "api"
    #: 通过文件哨兵触发（人工创建文件）
    SENTINEL = "sentinel"


@dataclass(frozen=True, slots=True)
class EStopState:
    """E-STOP 状态快照。

    Attributes:
        triggered: 是否处于触发状态。
        reason: 触发原因（人类可读）。
        source: 触发来源。
        triggered_at: 触发时刻（ISO 8601）。
    """

    triggered: bool = False
    reason: str = ""
    source: str = ""
    triggered_at: str = ""

    def to_dict(self) -> dict[str, object]:
        """返回可 JSON 序列化的字典（不含截图 / 用户数据）。"""
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


def _now_iso() -> str:
    """返回本地时间的 ISO 8601 字符串（与 Task / Runtime 其它模块同格式）。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class EStop:
    """紧急停止开关（API + 文件哨兵；线程安全由使用方保证：单线程事件循环）。

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
            else resolve_from_project_root(DEFAULT_SENTINEL_PATH)
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
    def state(self) -> EStopState:
        """当前状态快照（**不含哨兵检查**：仅反映 API 触发情况）。"""
        return EStopState(
            triggered=self._triggered,
            reason=self._reason,
            source=self._source,
            triggered_at=self._triggered_at,
        )

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
    def triggered_at(self) -> str:
        """触发时刻（未触发时为空串）。"""
        return self._triggered_at

    # ---- 触发与检查 ---------------------------------------------------------

    def trigger(self, reason: str = "manual", *, source: EStopSource = EStopSource.API) -> None:
        """触发紧急停止（幂等：保留首次触发的原因与时刻）。

        Args:
            reason: 触发原因（人类可读）。
            source: 触发来源。
        """
        if self._triggered:
            return
        self._triggered = True
        self._reason = reason
        self._source = source.value
        self._triggered_at = self._now()
        logger.error("紧急停止已触发: %s（source=%s）", reason, source.value)

    def check_sentinel(self) -> bool:
        """检查文件哨兵是否存在。

        Returns:
            ``True`` 表示哨兵文件存在（若此前未触发，则记录为哨兵触发）。
        """
        if not self._path.exists():
            return False
        if not self._triggered:
            self.trigger(f"sentinel file present: {self._path}", source=EStopSource.SENTINEL)
        return True

    def check(self) -> bool:
        """检查是否处于紧急停止状态（API 触发或哨兵存在）。

        引擎每轮调用本方法（``on_stable`` 与 ``on_cycle`` 都会检查）。

        Returns:
            ``True`` 表示必须立即停止任务且不再请求任何动作。
        """
        if self._triggered:
            return True
        return self.check_sentinel()

    # ---- 人工复位 -----------------------------------------------------------

    def create_sentinel(self, reason: str = "manual") -> Path:
        """创建哨兵文件（便于人工 / 脚本触发）。

        Args:
            reason: 写入文件首行的原因。

        Returns:
            哨兵文件路径。
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(f"{reason}\n", encoding="utf-8")
        logger.warning("已创建 E-STOP 哨兵文件: %s", self._path)
        return self._path

    def clear_sentinel(self) -> bool:
        """删除哨兵文件（若存在）。

        Returns:
            ``True`` 表示确实删除了文件。
        """
        try:
            self._path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:  # pragma: no cover - 文件系统异常
            logger.warning("删除 E-STOP 哨兵失败: %s", exc)
            return False
        return True

    def reset(self, *, remove_sentinel: bool = True) -> None:
        """复位（**必须显式调用**，不自动复位）。

        Args:
            remove_sentinel: 是否同时删除哨兵文件。
        """
        if remove_sentinel:
            self.clear_sentinel()
        if self._triggered:
            logger.warning("紧急停止已复位（此前原因: %s）", self._reason)
        self._triggered = False
        self._reason = ""
        self._source = ""
        self._triggered_at = ""

    def describe(self) -> str:
        """返回可读描述（含哨兵路径）。"""
        return f"EStop(path={self._path}, {self.state.describe()})"

    def __repr__(self) -> str:
        return self.describe()
