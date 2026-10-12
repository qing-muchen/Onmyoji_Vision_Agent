# Phase 12.1-A1 — Workflow ActionFeedback Closure Report

> 基线 commit：`2a38737 feat(task): add offline task dry-run harness`（Phase 12.1-A）
>
> 目标：**只**修复 Phase 12.1-A 审计缺口 **G6** —— ``WorkflowRunner`` 主路径的
> ``ActionOutcome`` 未写回对应 ``TaskInstance`` 的 ``DecisionContext``。
> **不**进入 12.1-B、**不**做 Task 架构重构。

---

## 1. G6 根因

| 环节 | 事实 |
|---|---|
| 任务表示 | 仓库 ``TaskDefinition`` **只有 workflow 形式**（``name/initial/states/...``）⇒ 任务**主路径**必然是 ``WorkflowRunner`` |
| 派发点 | ``WorkflowRunner.enter()``（``src/task/workflow.py:243-269``）**自行** ``outcome = port.request(request)`` |
| 反馈点 | ``TaskEngine._record_action_feedback``（``engine.py:1155``）只在 **step 路径**（``_dispatch``, ``:1034``）与 flow 路径（``:768``）被调用 |
| 结果 | workflow 路径**绕过**了反馈写入 ⇒ ``DecisionContext.actions`` 恒为空（``context.stats.actions = 0``），反馈可观测性断裂 |

**安全影响：无** —— 动作仍完整经过 5 道安全门 + ``DryRunBackend``，``executed_count`` 恒 0。

## 2. 最小修复路径

**只包装传给 ``WorkflowRunner`` 的端口**，复用既有反馈机制（不开发第二套模型）：

```
_start_definition()                     engine.py:422-428
   └─ WorkflowRunner(..., port=_WorkflowFeedbackPort(self, context))   ← 唯一改动点
          └─ request(request)
               ├─ outcome = self._action_port.request(request)   ← 真实端口（不变）
               └─ self._engine._record_action_feedback(context, outcome, …)  ← 既有机制
```

新增（均在 ``src/task/engine.py`` 内，**未新建模块**）：

* ``_WorkflowFeedbackPort``：单入口包装器；每次 ``request()`` **恰好**记录一次；
  **不修改** outcome、**不改变** ``StepActionOutcome.ok``（工作流是否推进仍由真实结果决定）；
* ``_target_name_of(request)``：仅按**字段视图**提取语义目标名（不 import ``src.action``）。

**未修改**：``WorkflowRunner``、``TaskEngine._dispatch``（step 路径）、``_record_action_feedback``、
``ActionDispatcher`` / 安全门 / ``StateStabilizer`` / ``Scheduler`` / 生产配置。

## 3. 修改文件与依赖检查

| 文件 | 变更 |
|---|---|
| `src/task/engine.py` | **+60 / −2**（新增 `_WorkflowFeedbackPort`、`_target_name_of`；改一处 `port=` 绑定） |
| `tests/test_phase121a_task_integration.py` | 更新 G6 测试为"已修复"断言 + 2 项新断言 |
| `tests/test_phase121a1_workflow_feedback_closure.py` | **新增 22 项**回归测试 |
| `docs/PHASE12_1A1_WORKFLOW_FEEDBACK_CLOSURE_REPORT.md` | 本报告 |
| `docs/PHASE12_1A_TASK_INTEGRATION_AUDIT_AND_DRY_RUN_REPORT.md` | 更新：G6 → **已修复** |

**依赖边界检查**（AST 遍历 ``src/task/**``）：**零** ``cv2`` / ``numpy`` / ``src.vision`` / ``src.action``
import ⇒ 未引入循环依赖，Task 层仍不接触坐标 / Frame / OpenCV。
**结构约束检查**：`_WorkflowFeedbackPort` 在引擎中**只有 1 处**构造 ⇒ 不可能重复记录；
`WorkflowRunner` 源码不含 `_record_action_feedback`（runner 仍与引擎反馈解耦）。

## 4. 三种反馈证据（真实运行）

| 场景 | intent | **recorded** | status | dry_run | ok | reason |
|---|---|---|---|---|---|---|
| **正常 workflow（dry-run）** | 1 | **1** | `skipped` | **True** | True | `dry-run 后端：已记录请求，未产生输入` |
| **E-STOP 拒绝** | 4 | **4** | `rejected` | False | False | `紧急停止生效（…）` |
| **Target 不存在拒绝** | 2 | **2** | `rejected` | False | False | `语义目标不可用（target_not_found（name=bottom_menu_yard, origin=registry））` |

⇒ 三类真实结果均**如实写入**；**dry-run 未被解释为"已执行 / 已确认"**（``status=skipped`` ≠ ``executed``）。

## 5. ActionOutcome 与 DecisionContext 记录字段一致性

单条记录（正常场景）实测：

```
kind=click · target=battle_auto_button · status=skipped · dry_run=True · ok=True
reason=dry-run 后端：已记录请求，未产生输入
task_id=dry_run_click · step_id=CLICK · action_id=dry_run_click:CLICK:1
attempt=1 · runtime_cycle_id=2
```

| 关联字段 | 与原始请求一致 |
|---|---|
| `task_id` | ✅ 与 ``ActionRequest.task_id`` 相同 |
| `step_id` | ✅ 与 ``ActionRequest.step_id`` 相同（= 状态名） |
| `action_id` | ✅ 与 ``ActionRequest.action_id`` 相同（幂等键 ``task:step:attempt``） |
| `attempt` | ✅ 相同 |
| `kind` / `target` | ✅ 相同（target 为语义元素名） |
| `runtime_cycle_id` | ✅ 来自 ``DecisionContext.cycle_id``（G5 的 ≤1 轮滞后语义**未改变**） |

## 6. 写入次数与 outcome 数量一致性

| 场景 | intent | outcomes | 上下文记录 | 一致 |
|---|---|---|---|---|
| A（正常） | 1 | 1 | 1 | ✅ |
| D | 1 | 1 | 1 | ✅ |
| E（target 缺失） | 2 | 2 | 2 | ✅ |
| F（E-STOP） | 4 | 4 | 4 | ✅ |
| F2（多任务隔离） | 1 | 1 | 1 | ✅ |
| F3（停止 + 限流） | 2 | 2 | 2 | ✅ |
| 重复回调（12 轮稳定） | 1 | 1 | 1 | ✅ 不重复 |
| 直接连续两次端口调用 | +2 | +2 | +2 | ✅ 逐次记录 |

**结论**：`recorded == intent_count == len(outcomes)`，**不重不漏**。

## 7. 多任务隔离与终止后反馈行为

| 检查 | 结果 |
|---|---|
| 两个独立 ``TaskInstance`` 的 ``DecisionContext`` | **不同对象**，互不串写 ✅ |
| 未运行实例收到的反馈 | **0 条**（`intent_count = 0`）✅ |
| 停止（``stop_task``）后 | 记录数仍 == outcome 数；`active_task_id` 为 `None` ✅ |
| 已失败实例 | 保留其**全部真实**反馈（4 条 rejected）✅ |
| 拒绝反馈是否错误推进工作流 | **否** —— 任务状态为 `failed`，`active_task_id = None` ✅ |

## 8. step 路径语义回归防护

| 检查 | 结果 |
|---|---|
| step 路径仍直接用 `self._action_port.request` | ✅（源码断言） |
| `_WorkflowFeedbackPort` 构造点数量 | **1**（AST 断言）✅ |
| `WorkflowRunner` 源码未改动 | ✅（仍 `port.request(request)`，且不含 `_record_action_feedback`） |
| 既有 `test_runtime_task_bridge.py` / `test_scheduler.py` / `test_state_machine.py` 等 | **全量通过** ✅ |
| ``TaskEngine._record_action_feedback`` 本身 | **未修改** ✅ |

## 9. `real_input=false` 验证

* 全部反馈场景 `executed_count == 0`、`real_input is False`、`action_enabled is False`（参数化断言）；
* `DryRunBackend` 只记录 `BackendCall`；`SendInput` 仅存在于未启用的 `backend/windows.py`；
* 生产配置未修改：`action.enabled=false` · `backend=dry_run` · `yard_metrics=false`。

## 10. 测试与验收

| 项 | 结果 |
|---|---|
| `python -m pytest -q` | **1969 passed, 1 skipped**（1947 + 22 新增） |
| `python -m pytest tests/test_phase121a1_workflow_feedback_closure.py tests/test_phase121a_task_integration.py -q` | **55 passed** |
| `tools/check_dataset.py` | **PASS** |
| `tools/raw_manifest.py` | **`ok: true`**（178/178 + 29/29） |
| `tools/check_annotations.py` | **PASS** |
| `src/` diff | **仅 `src/task/engine.py`（+60/−2）** |
| `configs/` · `dataset/` | **零改动** |

## 11. 提交

| 项 | 值 |
|---|---|
| 修改文件 | `src/task/engine.py`、`tests/test_phase121a_task_integration.py`、`tests/test_phase121a1_workflow_feedback_closure.py`、本报告、12.1-A 报告、README、ARCHITECTURE_OVERVIEW |
| commit | 见提交记录（`fix(task): close workflow action feedback loop`） |

## 12. 严格限制遵守情况

| 限制 | 遵守 |
|---|---|
| 保持 `action.enabled=false` / `backend=dry_run` | ✅ |
| 不改 Vision / Matcher / Dataset / 生产配置 | ✅ |
| 不改 Action safety gates | ✅ |
| 不改 `StateStabilizer` / `Scheduler` 判定逻辑 | ✅ |
| 不接入真实客户端 Task、不允许 `SendInput` | ✅ |
| 不调整 G5 的 `cycle_id` 行为 | ✅（`runtime_cycle_id` 语义未变） |
| 不修改历史任务成功判定语义 | ✅（`_record_action_feedback` 未改；``StepActionOutcome.ok`` 仍由真实 outcome 决定） |
| 未新建第二套反馈模型 / 未扩大为重构 | ✅ 复用 `_record_action_feedback` + `ActionRecord` |
| 未发现"必须改变 Workflow 生命周期或状态机语义"的情况 | ✅（故未停止上报） |

**Phase 12.1-A1 判定：PASS —— G6 已修复并验证。**
