# Phase 7.4 报告 —— Task → Action 集成（Vision-aware Action 执行闭环）

> 状态：**已完成**（等待下一步指令）
> 前置：Phase 7.3（Runtime 视觉快照接入，提交 `f459230`）
> 本阶段只完成：**"任务动作 → 语义目标 → Action Layer"的闭环**；
> 不新增任务类型、不新增游戏逻辑、不新增状态机、不新增视觉算法、不接入真实鼠标键盘。

---

## 1. Task → Action 数据流

```text
TaskDefinition（YAML / 编程式）
      │
      ▼
TaskEngine ──▶ StepExecutor ──▶ ActionRequest ──▶ ActionPort.request()
      │                                                  │
      │  （Task 只知道"语义目标 + 请求形状"，不知道坐标）      ▼
      │                                        ActionDispatcher ──┬─▶ TargetResolver（语义名 → 位置）
      │                                                          ├─▶ SafetyGuard（5 道门）
      │                                                          └─▶ InputBackend（默认 DryRun）
      │                                                                   ▲
      │                                                                   │
      └──────────────────────────── Runtime：Frame → 证据 → 快照 ──────────┘
                                    （Phase 7.3 VisionSnapshotBridge → resolver）
```

**两条链路在 ActionDispatcher 汇合**：

```text
Vision Snapshot ──▶ resolver.update_snapshot()   （Runtime 推送，Phase 7.3）
                              │
Task ──▶ ActionRequest ──▶ ActionDispatcher ──▶ resolver.resolve() ──▶ Backend
```

### 1.1 本阶段的实际改动（先评估"是否需要重构"）

用户要求"**如果**当前 TaskEngine 和 ActionDispatcher 耦合过深，新增 `src/task/action_adapter.py`"。
评估结论：

| 现状 | 结论 |
| --- | --- |
| `TaskEngine` 只依赖 `ActionPort` **协议**（``request(request)``），不 import `src.action` | 耦合**已经很松**（Phase 6.1 的 D-4 决策） |
| 请求对象用**字段视图**传递（`action_id` / `kind` / `target` / `params` / `timeout_s` / `attempt`） | Action 侧 `ActionRequest.from_task_request()` 适配 ⇒ 无需共享类型 |
| `TaskEngine` 不触碰 resolver / snapshot / Frame / Evidence | 边界已满足（AST 用例固化） |

因此本阶段**没有重构引擎**，新增的 `src/task/action_adapter.py` 是一个**可选的便利层**，
解决三件实际存在的事：

1. **编程式构造动作请求**（无需先写 YAML）：`adapter.click("battle_auto_button")`；
2. **在任务侧就把"语义目标"限定为语义名**（`{"element": ...}`），**明确拒绝坐标 / 比例**
   —— 与 Action 侧 `SemanticTarget` 规则一致，形成两层防线；
3. **显式声明字段契约**（幂等键 / 任务与步骤标识 / 超时 / 尝试序号），便于任务库与测试复用。

---

## 2. ActionRequest 设计（不重新设计，只固化契约）

沿用 Phase 7.0 冻结的**两层同形契约**（字段视图，不互相 import）：

| 任务侧 `src/task/workflow.py::ActionRequest` | Action 侧 `src/action/models.py::ActionRequest` | 说明 |
| --- | --- | --- |
| `action_id`（`task:step:attempt`） | `request_id` | 幂等键（Action 层可据此去重） |
| `kind`（`click` / `key` / `wait` …） | `action_type` | 动作类型，取值集合一致（`ACTION_KINDS`） |
| `target`（`{"element": "buttons/explore"}`） | `target: SemanticTarget` | **语义名**；Action 侧解析为窗口相对比例 |
| `params` | `parameters` | 附加参数（按键名 / 时长等） |
| `task_id` / `step_id` | 同名字段 | 审计 |
| `timeout_s` / `attempt` | 同名字段 | 执行层超时 / 尝试序号 |

**Task 侧不知道**：`x` / `y` / `x_ratio` / `y_ratio` / `bbox` / `rect` / `point` …（
`FORBIDDEN_TARGET_KEYS` 在适配器里**直接拒绝**这些字段，`Action` 侧 `SemanticTarget.from_mapping`
同样拒绝——两层防线）。

新增的 `TaskActionAdapter` 接口：

```python
TaskActionAdapter(*, task_id="", step_id="", timeout_s=1.0, attempt=1)
  .build(kind, target=None, *, params=None, step_id=None, timeout_s=None, attempt=None, suffix=None)
  .click(target, **)      .double_click(target, **)   .long_press(target, **)
  .drag(source, destination, **)   .key(key_name, target=None, **)   .wait(**)
  .with_step(step_id, *, timeout_s=None)      # 派生步骤级适配器
  @staticmethod normalize_target(value) / spec(value)

SemanticTargetSpec(name)  # 只允许语义名；from_value / from_mapping / to_mapping
normalize_semantic_target(value) -> {"element": name} | None
```

* `wait` 是唯一**不需要目标**的动作（与 Action 侧 `ActionType.needs_target` 对齐）；
* `key` 与 `click` 一样需要一个语义名（Action 侧契约要求，见 §7 Q-7.4-1）；
* 适配器**不调用 backend / resolver**，只构造 `ActionRequest`（AST 用例断言代码中不出现输入原语）。

---

## 3. Vision 边界（严格保持）

| 规则 | 落实 |
| --- | --- |
| Task 不知道 Vision | `src/task` **不 import** `src.vision` / `src.capture` / `src.runtime` / `src.action` / `cv2` / `numpy`（AST 用例 + 白名单用例） |
| Task 只知道 `ActionRequest` / 语义目标 | 请求里只有 `{"element": <name>}`；适配器拒绝一切坐标字段 |
| TaskEngine 不推送快照 | AST + 文本双检：`engine.py` / `workflow.py` / `action_adapter.py` 中**不出现** `update_snapshot` / `TargetSnapshot` / `Frame` / `EvidenceResult` |
| TaskEngine 不读 Frame / Evidence | 同上；任务只消费 `StateResult` 视图（Phase 6.1 既有边界） |
| Vision 不知道 Task | 新增用例：`src/vision` **不得** import `src.task` |
| Runtime 仍是唯一编排层 | 快照推送仍只发生在 `src/runtime/vision_bridge.py`（Phase 7.3）；本阶段未改 Runtime |

任务侧的"状态输入"依旧来自 Runtime（`engine.on_stable(StateResult视图)` / `on_cycle`），
与 Phase 6.1/6.2 完全一致——**本阶段没有给 Task 新增任何视觉依赖**。

---

## 4. 失败策略

| 情况 | Action 层结果 | Task 层结果（实测） |
| --- | --- | --- |
| Task 请求的目标不在快照里 | `REJECTED`，`resolution_reason=target_not_found` | 任务 `FAILED`，`reason` 含 `target_not_found`；backend **0** 次调用 |
| 从未推送过快照 | `REJECTED`，`target_not_found`（`snapshot=none`） | 同上（**绝不猜坐标**） |
| 快照过期（超过 TTL） | `REJECTED`，`target_expired`（含 `age_s` / `ttl_s`） | 任务 `FAILED`；backend **0** 次调用（**不使用上一帧位置**） |
| 后端为 dry-run | `SKIPPED`（已受理、无输入） | 任务正常推进（链路完整执行）；backend 只记录 `click` |
| E-STOP 生效 | `REJECTED`，`stage=estop` | 任务 `FAILED`；backend 0 次调用（安全门未被绕过） |
| 窗口失焦 | `REJECTED`，`stage=foreground` | 同上 |
| 动作被拒但仍有预算 | 引擎按 `max_step_retries` 重试（测试用例把预算设为 0 以获得确定性） | — |

关键点：**Task 侧的失败是"显式失败"而不是"忽略"**——`ActionOutcome.is_failure` 被引擎识别为
动作失败，任务进入 `FAILED` 并记录原因（便于上层重试或告警）。

---

## 5. 安全门（未被绕过）

Task 的动作请求与其它来源的请求走**完全相同的**门控链（Phase 7.0 冻结，本阶段未改动）：

```text
E-STOP → Window → Foreground → TargetResolver → TargetValidation → RateLimit → Backend
```

* Task **不直接调用** backend / resolver / SafetyGuard：只调用注入的 `ActionPort`（即 `ActionDispatcher`）；
* 门控失败的 stage 会进入 `ActionOutcome.detail.stage` 与 `dispatcher.stats.reject_stages`，
  测试用例断言了 `estop` 与 `foreground` 两条路径；
* dry-run 后端下**链路完整执行**（门控 → 解析 → 坐标换算 → 后端记录），只是不产生真实输入；
* `action.enabled=false` / `backend=dry_run` / `resolver=registry` 三个默认值本阶段未修改。

---

## 6. 测试结果

```text
$ python -m pytest -q
1247 passed, 1 skipped in 60.02s

$ python tools/check_dataset.py
Dataset: PASS   # 1602 张图；invalid / unresolved / duplicate = 0

$ git status
clean
```

| 项 | 数值 |
| --- | --- |
| 全套基线（Phase 7.3 后） | 1218 passed / 1 skipped |
| 本阶段新增 | **+29**（`tests/test_task_action_integration.py`） |
| 现基线 | **1247 passed / 1 skipped** |

### 6.1 覆盖矩阵（用户要求的 6 项）

| # | 要求 | 用例 |
| --- | --- | --- |
| 1 | Task 产生 click 动作 → Dispatcher 收到请求 | `test_task_produces_click_request`（字段 / 语义目标 / 幂等键 / 无坐标）、`test_dispatcher_receives_request_and_counts_it`、`test_dispatcher_rejects_task_request_without_snapshot`（反向验证请求真的到达 Action 层） |
| 2 | Vision 存在目标 → completed + backend 记录 click | `test_task_completes_when_vision_has_target`（含坐标闭式解断言）、`test_task_completes_through_runtime_bridge_pipeline` |
| 3 | Vision 不存在目标 → `target_not_found` | `test_missing_target_is_rejected_with_reason`、`test_missing_snapshot_is_rejected` |
| 4 | TTL 过期 → `target_expired` | `test_expired_snapshot_is_rejected` |
| 5 | Task 不知道 Vision（AST） | `test_task_layer_does_not_import_vision_or_capture`、`test_task_layer_allowed_imports_only`、`test_vision_layer_does_not_import_task`、`test_action_layer_does_not_import_vision_or_task`、`test_engine_never_pushes_snapshots_or_reads_frames`、`test_action_adapter_does_not_touch_backend_or_resolver` |
| 6 | 端到端 dry-run（FakeFrame → … → DryRunBackend） | `test_end_to_end_dry_run_full_chain`（Frame → 证据 → 桥 → 快照 → resolver → TaskEngine → Dispatcher → DryRunBackend）、`test_task_action_never_produces_real_input` |

另含：安全门未被绕过（`test_estop_blocks_task_action`、`test_foreground_gate_blocks_task_action`）、
适配器行为（`test_adapter_builds_requests_with_semantic_targets`、`test_adapter_rejects_coordinates`、
`test_adapter_requires_target_for_click`、`test_adapter_idempotency_key_and_suffix`、
`test_adapter_derives_step_scoped_copy`、`test_adapter_supports_all_action_kinds`、
`test_semantic_target_spec_and_normalizer`、`test_adapter_request_is_accepted_by_dispatcher`、
`test_adapter_drag_is_rejected_by_action_layer`、`test_task_definition_action_still_flows_through_port`、
`test_snapshot_type_never_leaks_into_task_layer`）。

### 6.2 端到端冒烟（dry-run，零真实输入）

```text
adapter 示例请求: click(battle_auto_button) attempt=1 id=daily:click_auto:1

=== A) 未推送任何视觉快照 ===
  task: failed | action_failure: 语义目标不可用（target_not_found（name=battle_auto_button, snapshot=none））
  dispatcher: {requests: 1, rejected: 1, reject_stages: {target: 1}}   | backend calls: 0

=== B) Runtime 推送快照后重试 ===
  bridge: {updates: 1, published: 1, candidates: 1, last_frame_id: 1}
  resolver: VisionAwareTargetResolver(ttl_s=1.0, updates=1, stale=0, snapshot=1)
  backend call: click((899, 953), button=left, count=1)        ← 客户区 (47,473,1704,960) × (0.5,0.5)
  task: completed | status=completed steps=1/1

=== C) 快照过期后 ===
  task: failed | action_failure: 语义目标不可用（target_expired（age_s=5.0, ttl_s=1.0））

backend calls total: 1 | real input: False
```

---

## 7. 未解决问题

| # | 问题 | 现状 / 建议 |
| --- | --- | --- |
| Q-7.4-1 | `key` 动作为什么需要一个（用不到的）语义目标？ | Action 侧 `ActionType.needs_target` 只豁免 `wait`，因此 `key` 也必须有目标；若要放宽，应改 Action 侧模型（本阶段未改，保持契约一致）。建议 Phase 7.5 决定"按键是否绑定焦点元素" |
| Q-7.4-2 | Task 与 Runtime 的自动接线（谁在何时驱动 `on_stable` / `on_cycle`） | 仍未做：本阶段只提供"Task → Action"闭环，二者的编排（`StateMonitor.on_stable → TaskEngine`）属 Phase 7.5 |
| Q-7.4-3 | `drag` 的语义 | 适配器已能表达 `from_target` / `to_target`，但 Action 侧仍明确 `REJECTED`（未实现） |
| Q-7.4-4 | 任务级失败后的重试/降级策略 | 当前"动作被拒 → 任务 FAILED"；是否要"重新解析（等下一帧）"由任务策略决定（`on_mismatch` / `max_step_retries` 已有，但缺少"解析失败专用策略"） |
| Q-7.4-5 | 语义名命名空间统一 | Task 用 `BATTLE_AUTO_BUTTON` / `battle_auto_button`（大小写不敏感匹配），视觉侧用 `battle_auto_button`；跨层命名规范建议在任务库阶段统一（可用 resolver 的 `aliases` 过渡） |
| K-7.4-1 | `tests/test_task_action_integration.py` 819 行 / `action_adapter.py` 356 行 | 超出 300 行规范（与既有 K-1 / K-7.x 同类） |

---

## 附录 A：复核命令

```powershell
# 全套 + 数据集
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\check_dataset.py

# Phase 7.4 定点测试
.\.venv\Scripts\python.exe -m pytest tests/test_task_action_integration.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_task_action_integration.py -q -k "vision or not_found or expired or dry_run"

# 边界固化（Task 不依赖 Vision / Action；Vision 不依赖 Task）
.\.venv\Scripts\python.exe -m pytest tests/test_task_action_integration.py -q -k "imports or allowed or engine_never or adapter_does_not"

# 适配器示例（任务侧只给出语义名）
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from src.task import TaskActionAdapter; a=TaskActionAdapter(task_id='demo', step_id='s'); print(a.click('battle_auto_button').to_dict())"
```

## 附录 B：公开 API 增量

```text
src.task.action_adapter（新增模块）
  TaskActionAdapter(*, task_id, step_id, timeout_s, attempt)
    .build / .click / .double_click / .long_press / .drag / .key / .wait
    .with_step / .normalize_target / .spec / .describe
  SemanticTargetSpec(name)  + from_value / from_mapping / to_mapping / describe
  normalize_semantic_target(value)
  SEMANTIC_TARGET_KEYS / FORBIDDEN_TARGET_KEYS / TARGETLESS_KINDS

src.task（包导出）
  + TaskActionAdapter / SemanticTargetSpec / normalize_semantic_target
  + SEMANTIC_TARGET_KEYS / FORBIDDEN_TARGET_KEYS / TARGETLESS_KINDS
```

## 附录 C：术语表

| 术语 | 含义 |
| --- | --- |
| 任务侧 `ActionRequest` | 任务产出的动作请求（字段与 Action 侧同形，不含坐标） |
| 语义目标 | `{"element": "battle_auto_button"}`——只给名字，位置由 Action 侧 resolver 决定 |
| TaskActionAdapter | 任务侧的可选便利层：动作意图 → `ActionRequest`（不执行、不解析、不读视觉） |
| 汇合点 | `ActionDispatcher`：同时消费 Task 的请求与 Runtime 推送的快照 |
| 两层防线 | 适配器拒绝坐标字段 + Action 侧 `SemanticTarget` 拒绝坐标写法 |
