"""任务控制状态（Phase 12.1-C2）：Task Enable / Disable 与统一准入优先级。

```text
装配侧（Bridge / 未来 orchestration）                Task 层
────────────────────────────────────               ────────────
TaskControlState(estop=...)  ──┐
  .disable(task_id)            │  持有"哪些任务被禁用"（纯状态，不含调度逻辑）
  .enable(task_id)             │
  .admission_reason(task_id)  ─┴──▶ 作为 ``TaskScheduler(admission_guard=...)`` 的注入点
```

设计（Phase 12.1-C0 已批准方向）：

* **状态所有权**：由上层任务控制组件（本类）拥有；``TaskScheduler`` 只通过既有
  ``admission_guard`` **消费**判定，``TaskTriggerRegistry`` 保持纯映射职责，
  ``TaskDefinition`` **不被修改**，也不新增 Scheduler 状态机；
* **统一准入优先级**（同一周期内判定顺序）：
  ``E-STOP`` > ``Shutdown``（未来 C3，本模块不实现） > ``Task Disabled`` > 正常准入；
* **结构化拒绝原因**（调用方按字段判定，**不**解析异常文本）：
  :data:`ADMISSION_ESTOP_ACTIVE` / :data:`ADMISSION_TASK_DISABLED`；
  :data:`ADMISSION_GUARD_ERROR` 由 ``TaskScheduler`` 在 guard 自身抛异常时产生（fail-closed）；
* **不引入额外确认规则**：Enable 只恢复触发资格 —— 是否触发仍由既有
  ``StateStabilizer`` / 触发映射与**新的** stable observation 决定。

依赖边界（与其它 ``src.task`` 模块一致）：**禁止** import ``src.vision`` /
``src.action`` / ``src.runtime`` / ``cv2`` / ``numpy``。
"""

from __future__ import annotations

from collections.abc import Iterable

from src.task.estop import EStop

__all__ = [
    "ADMISSION_ESTOP_ACTIVE",
    "ADMISSION_GUARD_ERROR",
    "ADMISSION_TASK_DISABLED",
    "TaskControlState",
]

#: 结构化拒绝原因：E-STOP 有效（**最高**优先级）
ADMISSION_ESTOP_ACTIVE: str = "estop_active"
#: 结构化拒绝原因：任务被显式禁用
ADMISSION_TASK_DISABLED: str = "task_disabled"
#: 结构化拒绝原因：准入 guard 自身故障（由 ``TaskScheduler`` 产生，fail-closed）
ADMISSION_GUARD_ERROR: str = "guard_error"


class TaskControlState:
    """任务级控制状态（Enable / Disable）+ 准入判定（**可直接作为** ``admission_guard``）。

    Args:
        estop: 可选 E-STOP 开关（**只读消费**：本类既不触发也不复位它）。
        disabled: 初始禁用集合。

    Notes:
        * :meth:`admission_reason` 的签名与 ``TaskScheduler`` 的 ``admission_guard``
          完全一致（``(task_id) -> str | None``），因此可直接注入；
        * 本类**不**捕获 E-STOP 检查异常：``EStop.check()`` 若因文件系统故障抛异常，
          由 ``TaskScheduler`` 记为 ``guard_error`` 并 **fail-closed**（不默认放行）；
        * 线程安全由使用方保证（与项目其它模块一致：单线程事件循环）。
    """

    def __init__(self, *, estop: EStop | None = None, disabled: Iterable[str] = ()) -> None:
        self._estop = estop
        self._disabled: set[str] = {str(item) for item in disabled}

    # ---- 只读视图 -----------------------------------------------------------

    @property
    def estop(self) -> EStop | None:
        """绑定的 E-STOP 开关（未绑定时 ``None``）。"""
        return self._estop

    @property
    def disabled_tasks(self) -> tuple[str, ...]:
        """当前被禁用的任务标识（字典序，输出稳定）。"""
        return tuple(sorted(self._disabled))

    def is_disabled(self, task_id: str) -> bool:
        """该任务是否被显式禁用。"""
        return str(task_id) in self._disabled

    # ---- 控制（幂等） -------------------------------------------------------

    def disable(self, task_id: str) -> bool:
        """禁用任务（阻止其为**新** stable observation 创建新实例）。

        Returns:
            ``True`` 表示状态发生变化；重复禁用返回 ``False``（幂等）。
        """
        key = str(task_id)
        if key in self._disabled:
            return False
        self._disabled.add(key)
        return True

    def enable(self, task_id: str) -> bool:
        """恢复任务的触发资格（**不**创建实例、**不**恢复旧实例、**不**清空旧队列）。

        Returns:
            ``True`` 表示状态发生变化；重复启用返回 ``False``（幂等）。
        """
        key = str(task_id)
        if key not in self._disabled:
            return False
        self._disabled.discard(key)
        return True

    # ---- 准入判定（注入 ``TaskScheduler.admission_guard``） -----------------

    def admission_reason(self, task_id: str) -> str | None:
        """准入判定：``None`` = 允许；否则返回**结构化**拒绝原因。

        优先级：``estop_active`` > ``task_disabled``。
        """
        estop = self._estop
        if estop is not None and estop.check():
            return ADMISSION_ESTOP_ACTIVE
        if str(task_id) in self._disabled:
            return ADMISSION_TASK_DISABLED
        return None

    # ---- 诊断 ---------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """返回可 JSON 序列化的字典（审计用；不含截图 / 用户数据）。"""
        return {"disabled": list(self.disabled_tasks), "estop_bound": self._estop is not None}

    def describe(self) -> str:
        """返回可读描述（**不**访问文件系统：只用 E-STOP 的 API 侧状态）。"""
        if self._estop is None:
            estop = "none"
        else:
            estop = "triggered" if self._estop.is_triggered else "clear"
        return f"TaskControlState(disabled={list(self.disabled_tasks)}, estop={estop})"

    def __repr__(self) -> str:
        return self.describe()
