# Phase 6.2 报告 —— Task Definition Loading & Workflow Execution

> 状态：**已完成**（等待 Phase 7 Action Layer）
> 前置：Phase 6.0（架构 Review，提交 `011d160`）、Phase 6.1（Task Engine 骨架，提交 `051daae`）
> 本阶段范围：**任务定义加载 + schema 校验 + 工作流执行模型 + 条件绑定**；
> **不实现**鼠标点击 / 键盘输入 / 游戏操作 / 真实 Action Adapter（`ActionPort` 保持 `NullActionPort`）。

---

## 1. 本阶段目标与范围

用户指令（Phase 6.2）：

```text
进入 Phase 6.2：Task Definition Loading & Workflow Execution。

实现：1. Task YAML 加载  2. Task Definition Schema
      3. Workflow Step 执行模型  4. Predicate / Hook 绑定
不要实现：鼠标点击 / 键盘输入 / 游戏操作 / 真实 Action Adapter
```

已确认的架构决策（沿用 Phase 6.1 的 Q&A）：

| 决策 | 内容 | 本阶段落实 |
| --- | --- | --- |
| Q-1 | Task 不依赖 `src.vision.state`，继续使用**字符串状态视图** | `StateSnapshot.state` 为字符串；`build_condition` 的字段白名单只做字符串 / 数值比较 |
| Q-2 | **不新增** `CANCELLED`；`stop` 使用 `FAILED(reason="stopped")` | 未新增状态；E-STOP 亦复用 `FAILED(reason="estop: ...")` |
| Q-3 | E-STOP = **API + `logs/ESTOP` 文件哨兵**；**禁止全局热键** | `src/task/estop.py`；AST 测试固化"无键盘钩子依赖" |
| Q-4 | 拆分 `task/models.py` | `task/models/{definitions,context,result}.py` + 兼容 re-export |
| Q-5 | 任务定义采用 `configs/tasks/*.yaml` | `TaskLoader` + 两个示例定义 |

**明确不做**（本阶段验收检查项）：

- ❌ 鼠标点击 / 键盘输入 / 任何输入注入（无 `pyautogui` / `pynput` / `win32api` 调用）；
- ❌ 真实 `ActionPort` 实现（仍为 `NullActionPort`：只记录请求）；
- ❌ 与 `StateMonitor` 的自动接线（Task 的 `on_stable` / `on_cycle` 由调用方驱动）；
- ❌ 修改 `runtime/` `vision/` `evidence/` `capture/` `window/` `action/`（本阶段 diff 未触及这些目录）；
- ❌ 全局热键（无键盘钩子库依赖，见第 6.3 节）。

---

## 2. 交付物清单

### 2.1 新增文件

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `src/task/loader.py` | 422 | `TaskLoader`：YAML 读取 → schema 校验 → 条件编译 → 图校验 → `TaskDefinition` |
| `src/task/workflow.py` | 656 | 动作端口契约（`ActionRequest`/`ActionOutcome`/`ActionPort`/`NullActionPort`）+ `StepExecutor` + `StepTransition` + `WorkflowRunner` + `WorkflowDecision` |
| `src/task/estop.py` | 243 | `EStop` / `EStopSource` / `EStopState`：API + 文件哨兵（**禁止热键**） |
| `src/task/models/__init__.py` | 76 | `models` 包聚合导出（旧 `from src.task.models import ...` 路径不变） |
| `src/task/models/definitions.py` | 783 | 声明式定义：`Task` / `Step` / `Budget` / `TaskPolicies` / `TaskStatus` / `StepPhase` + `TaskDefinition` / `WorkflowState` / `WorkflowTransition` / `WorkflowAction` / `TaskLike` |
| `src/task/models/context.py` | 588 | 运行期视图：`StateSnapshot` / `RuntimeSnapshot` / `TaskContext` |
| `src/task/models/result.py` | 72 | `TaskResult` |
| `configs/tasks/farming.yaml` | 61 | 示例任务（用户给定形状：等待战斗 → 战斗 → 结算 → `COMPLETE`） |
| `configs/tasks/open_exploration.yaml` | 61 | 示例任务（语义 click / 结构化条件 / 重试 / 多分支） |
| `tests/test_task_loader.py` | 1001 | Phase 6.2 测试（63 用例） |
| `docs/PHASE6_2_TASK_DEFINITION_REPORT.md` | 本文 | 阶段报告 |

### 2.2 修改文件

| 文件 | 变更 |
| --- | --- |
| `src/task/models.py` | **删除**（拆分为上面的 `models/` 包；对外导入路径不变） |
| `src/task/engine.py` | `start(TaskDefinition)` 与 `start(Task)` 双模式；`_tick_workflow` / `_tick_workflow_timeout` / `_workflow_dispatch_guard` / `_check_estop`；E-STOP 构造参数；动作端口契约改为从 `workflow` 导入 |
| `src/task/hooks.py` | 条件绑定：`build_condition`（统一入口）/ `parse_condition_expression`（受限解析）/ `CONDITION_FIELDS` / `CONDITION_TYPE_ALIASES` |
| `src/task/__init__.py` | 公开符号 35 → 61 |
| `tests/test_task_engine.py` | AST 依赖白名单加入 `yaml`（配置解析依赖） |
| `README.md` | Phase 6.0 / 6.1 / 6.2 段落、主链路、分层表、目录结构、路线图、"尚未实现" |
| `docs/ARCHITECTURE_OVERVIEW.md` | 分层表补 `src/task` 行、阶段索引补 Phase 6.x |

---

## 3. 架构变化

### 3.1 目录结构（本阶段目标形状）

```text
src/task/
├── models/
│   ├── __init__.py       # 聚合导出（兼容旧 import）
│   ├── definitions.py    # Task / Step / Budget / TaskPolicies + TaskDefinition / WorkflowState / ...
│   ├── context.py        # TaskContext / StateSnapshot / RuntimeSnapshot
│   └── result.py         # TaskResult
├── loader.py             # TaskLoader（YAML parsing + 校验）
├── workflow.py           # StepExecutor / StepTransition / WorkflowRunner + ActionPort 契约
├── engine.py             # TaskEngine（扩展：双模式）
├── state_machine.py      # （Phase 6.1，未改）
├── hooks.py              # （Phase 6.1 + 条件绑定）
├── estop.py              # EStop（Phase 6.2 新增；Q-3）
└── __init__.py
```

依赖方向（无环）：

```text
__init__ ─▶ engine ─▶ workflow ─▶ models/hooks ─▶ state_machine
              │           │
              │           └─▶ models.definitions / models.context
              ├─▶ estop ─▶ core.paths / utils.logger
              └─▶ loader ─▶ hooks + models + yaml（配置解析）
```

`ActionPort` / `ActionRequest` / `ActionOutcome` / `ActionStatus` / `NullActionPort` **从 `engine.py`
迁移到 `workflow.py`**（它们属于"工作流执行模型"），但 `src.task.engine.ActionPort` 仍可导入
（engine 从 workflow 导入并放入 `__all__`），旧路径不破坏。

### 3.2 `StepPhase` 归属调整

`StepPhase`（步骤 / 状态执行阶段）由 `engine.py` 移到 `models/definitions.py`：
引擎与工作流执行器共用同一枚举，避免 `workflow ⇄ engine` 的循环依赖；
`src.task.engine.StepPhase` 与 `src.task.StepPhase` 均仍可用。

### 3.3 分层视图

```text
Vision（看到什么） ─▶ State（现在是什么） ─▶ Task（下一步目标是什么） ─▶ Action（执行什么操作）
   cv2/detector        StateResult/Event       TaskDefinition（YAML）        ActionPort（Phase 7）
   禁止 Task import ───────────────────────────┘  Task 只消费 StateResult / Runtime 状态
```

Task 层的输入只有三类：**状态视图**（`StateSnapshot`）、**运行时视图**（`RuntimeSnapshot`）、
**计时回调**（`on_cycle(timestamp)`）——不读像素、不读证据、不 import `cv2`。

---

## 4. Task Definition Schema

### 4.1 顶层字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `name` | ✅ | 任务名（`TaskDefinition.task_id`） |
| `description` | — | 人类可读说明 |
| `version` | — | 定义版本（保留字段，参与校验但不参与执行） |
| `initial` | ✅ | 起始状态名，必须出现在 `states` 中 |
| `states` | ✅ | 状态表（非空映射），见 4.2 |
| `budget` | — | `max_steps`（默认 20）/ `timeout_s`（=`deadline_s`，默认 300）/ `max_step_retries` / `min_stable_frames` |
| `policies` | — | `on_unknown`（`wait`/`fail`）/ `on_runtime_failure`（`block`/`fail`）/ `on_mismatch`（`retry`/`fail`）/ `on_step_timeout`（`retry`/`fail`/`wait`） |
| `metadata` | — | 附加信息（不参与判定） |

### 4.2 状态字段（`states.<NAME>`）

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `description` | — | 状态说明 |
| `actions` | — | 进入该状态时请求的动作列表（见 4.4） |
| `on` | ✅ | 转移列表（**至少一条**；缺省即"死路状态"→ 加载失败） |
| `timeout` / `timeout_s` | — | 该状态的等待窗口（秒，默认 10） |
| `retry` / `retries` | — | 该状态的重试上限（默认取 `budget.max_step_retries`） |
| `entry_condition` | — | 进入条件（谓词；不满足则视为尚未就绪） |

### 4.3 转移字段（`on[*]`）

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `condition` | ✅ | 条件声明（字符串表达式 / `{type, value}` 结构 / `{state_is: ...}` 键式 / `all`·`any`） |
| `next` | ✅ | 目标状态名，或保留字 **`COMPLETE`**（工作流成功结束，无需在 `states` 中定义） |
| `name` | — | 转移名（缺省 `to_<next>`；用于审计与钩子事件） |

**评估顺序**：按**声明顺序**逐个求值，第一个满足者生效（无优先级字段，靠顺序表达优先级）。

### 4.4 动作字段（`actions[*]`）

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `kind` | 见下 | 动作类型：`click` / `double_click` / `long_press` / `drag` / `key` / `wait` |
| `target` | 见下 | 语义目标：`{element: "buttons/explore"}`（元素名）或 `{reference: [x, y]}`（**参考分辨率**坐标，D-3：禁止屏幕绝对坐标） |
| `timeout` / `timeout_s` | — | 执行层超时（秒，默认 1） |
| `params` | — | 附加参数（按键名 / 时长等） |

两种写法均支持：

```yaml
actions:
  - target: wait          # 动作类型写在 target 上（只等待，不请求输入）
    timeout: 30
  - kind: click           # 显式 kind + 语义元素
    target: {element: buttons/explore}
    timeout: 2
```

`kind: wait` → `requested=False`（**不会**产生 `ActionRequest`）；其余类型才提交请求。
`kind != wait` 且无 `target` → 加载失败。

### 4.5 完整示例（`configs/tasks/farming.yaml`）

```yaml
name: farming
description: "等待进入战斗 → 战斗进行中 → 结算收集 → 回到庭院"
initial: WAIT_BATTLE
states:
  WAIT_BATTLE:
    description: "等待进入战斗界面"
    on:
      - condition: state == battle
        next: IN_BATTLE

  IN_BATTLE:
    description: "战斗进行中：只等待，不请求任何输入"
    actions:
      - target: wait
        timeout: 30
    on:
      - condition: state == battle_result
        next: COLLECT

  COLLECT:
    description: "等待结算 / 奖励结束后回到庭院"
    on:
      - condition: state == yard
        next: COMPLETE
budget:
  max_steps: 100
  timeout_s: 3600
policies:
  on_unknown: wait
  on_runtime_failure: block
  on_step_timeout: retry
```

---

## 5. 加载期校验（fail fast）

`TaskLoader.load(path)` → `yaml.safe_load`（**不使用** `full_load` / `unsafe_load`）→ `validate()`：

| # | 校验项 | 失败信息关键词 |
| --- | --- | --- |
| 1 | 文件存在 / 可读 / YAML 可解析 / 顶层为映射 / 非空 | "不存在" / "YAML 解析失败" / "必须是映射" / "文件为空" |
| 2 | 顶层**未知字段** | "出现未知字段"（并列出可选字段） |
| 3 | 缺少 `name` / `initial` / `states`（空或非映射） | "缺少 name" / "缺少 initial" / "必须是非空映射" |
| 4 | 每个 state 未知字段 / 非法 `timeout`·`retry` | "出现未知字段" / "必须为正数" / "不能为负数" |
| 5 | state 无 `on`（死路状态） | "没有任何 on 转移（死路状态…）" |
| 6 | transition 未知字段 / 缺 `condition` / 缺 `next` | "出现未知字段" / "缺少 condition" / "缺少 next" |
| 7 | action 未知字段 / 非法 `kind` / 需要 `target` 却缺失 | "出现未知字段" / "action.kind 非法" / "需要 target" |
| 8 | `initial` 未定义 | "initial 指向未定义的状态" |
| 9 | `next` 悬空（既不是已定义状态，也不是 `COMPLETE`） | "next 指向未定义的状态" |
| 10 | 状态名使用保留字 `COMPLETE` | "不能使用保留字 COMPLETE" |
| 11 | **循环引用**（DFS 三色标记检测） | "存在循环引用: A -> B -> A" |
| 12 | 从 `initial` **无法到达** `COMPLETE`（防御性校验） | "无法到达 COMPLETE（工作流没有成功出口）" |
| 13 | 条件表达式 / 结构化条件非法（未知字段、未知类型、`and`·`or`、括号、空值） | "不支持的条件表达式" / "未知条件字段" / "不支持的条件类型" |
| 14 | `budget` / `policies` 未知字段与越界取值 | "未知字段" / "必须为正数" 等 |

### 5.1 YAML 1.1 细节：`on:` 会被解析成布尔键

PyYAML 遵循 **YAML 1.1**，其中 `on` / `off` / `yes` / `no` 是布尔字面量，因此

```yaml
states:
  A:
    on:                    # ← 键被解析为 True（bool），而不是字符串 "on"
      - condition: state == yard
        next: COMPLETE
```

会让朴素实现把 `True` 当成"未知字段"。**本阶段在 loader 内做了键规范化**
（`_normalize_keys`：`True → "on"`、`False → "off"`），因此**用户无需给 `on:` 加引号**；
该行为有专门用例 `test_yaml_boolean_on_key_is_normalized` 固化。

### 5.2 循环引用为什么必须拒绝

工作流是**有向无环图 + 唯一成功出口**：Task 不拥有循环（D-1），若允许环，则"任务是否会结束"
无法静态判定，预算与超时策略会退化为"只能靠 deadline 兜底"。拒绝环并强制可达 `COMPLETE`，
使"任务定义了成功终点"成为**加载期可验证的性质**。

---

## 6. 条件绑定（D-6：YAML → Predicate，**禁止 eval**）

### 6.1 三种写法（`hooks.build_condition` 统一入口）

```yaml
# 1) 字符串表达式（受限解析，只支持单一比较）
- condition: state == battle
  next: IN_BATTLE

# 2) 结构化（推荐）
- condition:
    type: state_equals
    value: battle
  next: IN_BATTLE

# 3) 组合（all / any）
- condition:
    type: any
    value:
      - state == battle
      - state == transition
  next: WAIT_BATTLE_END
```

Python 侧同样可用（`build_condition(None)` / `True` / `"always"` / 可调用对象 / 列表）。

### 6.2 受限表达式语法

```text
<field> <op> <literal>
```

| 部分 | 允许取值 |
| --- | --- |
| `field` | `state` / `raw_state` / `source` / `confidence` / `stable` / `counted` / `unknown` / `runtime_state` / `runtime_healthy` / `elapsed_s` / `cycles` |
| `op` | `==` `!=` `>=` `<=` `>` `<` |
| `literal` | 数字 / `true`·`false` / `none` / 引号字符串 / **不含空白的**裸字符串（状态名） |

明确**拒绝**（加载期报错，绝不"尽力解析"）：

- 逻辑组合与括号：`state == battle and confidence > 0.9`、`(state == yard)`；
- 未知字段：`mystery == 1`；
- 空比较值：`state == `；
- 含空白的裸值（多半是把 `and` / `or` 写进来了）：`state === yard`；
- 任何危险内容：`__import__('os')` 等（不匹配受限语法即失败）。

结构化条件同样严格：`type` 必须是 `CONDITION_TYPE_ALIASES` 中的键（`state_equals`、`state_in`、
`state_not_in`、`confidence_at_least`、`min_confidence`、`counted`、`known_state`、
`runtime_state_equals`、`runtime_state_is`、`runtime_ok`、`elapsed_at_least`、`cycles_at_least`，
或组合类型 `all` / `any`），且只允许 `{type, value}` 两个键。

### 6.3 为什么不可能发生动态执行

| 层 | 保障 |
| --- | --- |
| 语法层 | 正则 + 字段白名单：只有 `field / op / literal` 三种片段，没有函数调用与属性访问 |
| 类型层 | `WorkflowTransition.predicate` 必须是 **callable**（`__post_init__` 校验）；字符串无法冒充 |
| 架构层 | `TaskDefinition` / `WorkflowState` 只能由 `TaskLoader` 构建（编译发生在加载期） |
| 测试层 | `test_task_layer_has_no_eval_or_exec`：AST 扫描 `src/task/**`，禁止 `eval` / `exec` / `compile` |

---

## 7. 工作流执行模型

```text
TaskEngine.start(TaskDefinition)
  └─ WorkflowRunner.enter_initial()      # 进入 initial：提交 actions（wait 只等待）
        ↓
  on_stable(StateResult) ──▶ StepExecutor.evaluate()   # 按声明顺序求值转移条件
        ├─ 无满足者 → 继续等待（决策 NONE）
        ├─ 命中非终止 → WorkflowRunner.enter_state(next) → 决策 TRANSITION
        └─ 命中 COMPLETE → 决策 COMPLETE → 引擎：任务 COMPLETED
        ↓
  on_cycle(timestamp) ────▶ 等待窗口到期？
        ├─ policies.on_unknown == wait 且当前为 unknown → 延长窗口（**不消耗重试**）
        ├─ policies.on_step_timeout == wait → 延长窗口
        ├─ on_step_timeout == retry 且未超上限 → 重新进入该状态（重新请求动作）→ 决策 RETRY
        └─ 否则 → 决策 FAILED(reason="step_timeout")
```

### 7.1 关键不变量

| 不变量 | 说明 |
| --- | --- |
| **每轮最多一次状态转移** | 避免同一帧内连锁穿过多个状态（可预测性优先；YAML 工作流不是"瞬时展开"的宏） |
| **计时只在 `on_cycle`** | D-2：`on_stable` 只做决策，超时判定只用 `on_cycle` 的注入时钟（测试可注入假时钟） |
| **派发前检查预算** | 与线性模式一致：`max_steps` 在**真正派发前**检查（避免"最后一步已在等待达成"被误判超限） |
| **动作被拒即失败** | `ActionOutcome.is_failure`（`REJECTED` / `FAILED`）→ 决策 `FAILED(reason=...)`，不静默重试 |
| **unknown 不消耗重试** | `on_unknown: wait` + 当前 `state == unknown` → 只延长时间窗（与 Runtime 的"暂不可知"语义一致） |
| **Blocked 不派发** | 运行时故障 / 失焦 → `BLOCKED`：**不进入初始状态**、不请求动作、不消耗预算（D-5） |

### 7.2 `StepExecutor` / `StepTransition` / `WorkflowDecision`

- `StepExecutor(state)`：`enter(task_id, attempt, port)` → 提交语义动作（返回 `StepActionOutcome`）；
  `evaluate(context, snapshot)` → 返回第一个命中的 `WorkflowTransition`（谓词异常包成 `TaskError`）；
- `StepTransition(from_state, to_state, trigger, condition_spec, timestamp)`：跳转审计记录，
  `to_dict()` 可序列化；
- `WorkflowDecision(kind, reason, transition, outcome)`：`NONE` / `TRANSITION` / `COMPLETE` /
  `RETRY` / `FAILED`。**运行器不修改任务生命周期**（不 emit 钩子、不改状态机）——决策由引擎应用。

---

## 8. 引擎变化

### 8.1 双模式启动

```python
engine.start(task)                 # Task：Phase 6.1 线性步骤（行为未变，67 用例守护）
engine.start(definition)           # TaskDefinition：Phase 6.2 YAML 工作流
```

分派依据是类型（`isinstance(task, TaskDefinition)`）；两种模式共享门控、预算、E-STOP、钩子与结果模型。

### 8.2 新增观测点

| API | 说明 |
| --- | --- |
| `engine.definition` / `engine.runner` | 当前工作流定义与运行器（线性模式为 `None`） |
| `engine.current_state_name` | 工作流当前状态名（线性模式 `None`） |
| `engine.phase` | 工作流模式返回运行器阶段（`AWAITING` / `REJECTED` / `TIMEOUT` / `DONE`） |
| `runner.history` | 跳转历史（`StepTransition` 元组） |
| `runner.attempts` / `retries_used` / `timeouts` | 计数（观测与审计） |

钩子事件（`HookRegistry`）在工作流模式下补齐 `step_id`（用状态名）：
`TASK_STARTED` / `TASK_READY` / `STEP_DISPATCHED` / `STEP_SATISFIED` / `STEP_RETRIED` /
`STEP_TIMEOUT` / `ACTION_REQUESTED` / `TASK_COMPLETED` / `TASK_FAILED`。

### 8.3 E-STOP（Q-3）

```python
estop = EStop()                      # 默认哨兵路径 logs/ESTOP
engine = TaskEngine(estop=estop)

estop.trigger("manual abort")        # 方式 1：API
# 方式 2：人工创建文件（无需交互）
#   New-Item logs/ESTOP -ItemType File
engine.on_cycle()                    # 每轮检查 → 任务 FAILED(reason="estop: manual abort")
```

| 特性 | 行为 |
| --- | --- |
| 检查点 | `on_stable` 与 `on_cycle` **每轮**都检查（先于门控与决策） |
| 触发语义 | 任务立即 `FAILED(reason="estop: <原因>")`，后续轮次不再请求任何动作 |
| 幂等 | 重复 `trigger` 保留**首次**原因与时刻 |
| 复位 | 必须显式 `reset()`（默认同时删除哨兵文件），**不自动复位** |
| 热键 | **不实现**（无 `pynput` / `keyboard` / Win32 键盘钩子；AST 测试固化） |
| 日志 | 触发写 `ERROR` 级日志（无需看代码即可定位原因） |

### 8.4 未接线的部分（诚实记录）

- Task Engine **尚未**自动挂到 `StateMonitor`：`on_stable(result)` / `on_cycle(timestamp)` 的
  接线属 Phase 6.3（用户已确认"Task 与 Runtime 接线属 6.2"之前的表述，本阶段实际未做；
  引擎已接受 `set_runtime_source(monitor)` 与 `runtime=` 参数，可手工驱动）；
- 任务库 / 多任务调度 / 任务选择策略未实现（属 Phase 6.3 / 7）；
- 动作**未执行**：`NullActionPort` 只 `logger.info` 记录请求。

---

## 9. 测试结果

```text
$ python -m pytest -q
1070 passed, 1 skipped in 43.01s
```

| 项 | 数值 |
| --- | --- |
| 全套基线（Phase 6.1 后） | 1007 passed / 1 skipped |
| 本阶段新增 | **+63**（`tests/test_task_loader.py`） |
| 现基线 | **1070 passed / 1 skipped** |
| 既有 6.1 用例 | 67 passed（未修改行为，仅 AST 白名单加 `yaml`） |

### 9.1 覆盖矩阵（用户要求 6 项 + 额外）

| 要求 | 用例（节选） |
| --- | --- |
| YAML 读取 | `test_load_repository_task_definitions`、`test_load_all_from_repository_directory`、`test_load_from_temp_file`、`test_load_invalid_yaml_raises`、`test_load_missing_file_raises` |
| schema 验证 | `test_unknown_*_rejected`（顶层 / state / transition / action）、`test_missing_required_fields_rejected`、`test_budget_alias_and_validation`、`test_policies_validated`、`test_state_timeout_and_retry_validated` |
| 非法配置 | `test_dangling_next_rejected`、`test_cyclic_workflow_rejected`、`test_dead_end_state_rejected`、`test_unreachable_terminal_rejected`、`test_reserved_state_name_rejected` |
| transition | `test_transitions_evaluated_in_declaration_order`、`test_completes_when_reaching_complete`、`test_workflow_timeout_retries_then_fails` |
| predicate 绑定 | `test_condition_expression_parsing`、`test_unsupported_condition_expression_rejected`（6 组参数）、`test_typed_condition_binding`、`test_typed_condition_unknown_type_rejected`、`test_condition_in_workflow_compiled_at_load_time` |
| engine 执行 | `test_workflow_step_action_is_recorded_not_executed`、`test_workflow_vision_unknown_does_not_consume_retries`、`test_workflow_budget_max_steps_fails`、`test_workflow_blocks_on_runtime_failure_without_budget`、`test_workflow_and_linear_task_coexist` |
| AST 边界（新增） | `test_task_layer_forbidden_imports_absent`、`test_task_layer_has_no_eval_or_exec`、`test_task_layer_allowed_imports_only` |
| E-STOP | `test_estop_api_trigger_fails_task`、`test_estop_sentinel_file_triggers`、`test_estop_is_idempotent_and_resettable`、`test_estop_module_registers_no_hotkeys` |

### 9.2 端到端验收（离线脚本化状态，`NullActionPort`）

```text
$ python -c "…TaskLoader().load('configs/tasks/farming.yaml') + TaskEngine 驱动…"
start -> running wf = WAIT_BATTLE
  observe yard           -> task=running    wf=WAIT_BATTLE  actions=0
  observe battle         -> task=running    wf=IN_BATTLE    actions=0
  observe battle_result  -> task=running    wf=COLLECT      actions=0
  observe yard           -> task=completed  wf=COLLECT      actions=0
result: task=farming status=completed state=completed reason=COLLECT -> COMPLETE steps=3/3
transitions: ['WAIT_BATTLE -> IN_BATTLE', 'IN_BATTLE -> COLLECT', 'COLLECT -> COMPLETE']
```

结论：**决策链路完整可跑通，且全程未产生任何输入**（`actions=0`，`NullActionPort` 只记录
非 wait 类型的请求）。

### 9.3 数据集与依赖边界

```text
$ python tools/check_dataset.py
Dataset: PASS   # 1602 张图；invalid / unresolved / duplicate = 0
```

`src/task` 的 import roots（复核用）：

```text
['__future__', 'collections', 'dataclasses', 'datetime', 'enum', 'pathlib', 're', 'src', 'time', 'typing', 'yaml']
```

即：**标准库 + PyYAML（配置解析）+ `src.core` / `src.utils` / `src.task`**；
`cv2` / `numpy` / `PIL` / `torch` / `pyautogui` / `pynput` / `src.vision` / `src.runtime` /
`src.action` 均**不在**其中。

---

## 10. 与 Phase 6.0 / 6.1 的差异与未实现项

| 项 | Phase 6.0 设计 | 本阶段实现 | 差异原因 |
| --- | --- | --- | --- |
| 任务定义来源 | YAML 流程 + Python predicate（D-6） | ✅ `configs/tasks/*.yaml` + `build_condition` | 一致 |
| 生命周期状态 | （6.0 草案含 `WAITING` / `CANCELLED`） | 沿用 6.1 的 7 态；E-STOP 用 `FAILED(reason=...)` | Q-2：不新增 `CANCELLED` |
| Action 端口 | `ActionPort` 协议 | ✅ 契约就位，实现为 `NullActionPort` | 真实输入属 Phase 7 |
| 工作流模型 | "步骤序列 + 条件分支" | **状态机式**（`states` + `on`） | 用户给定的 YAML 形状 |
| 与 Runtime 接线 | Phase 6.2 提及 | **未做**（手工驱动） | 保持"不修改 runtime"边界 |

**未实现（明确留待后续）**：

1. 任何输入（点击 / 键盘）与真实 `ActionPort` 实现 —— Phase 7；
2. Task Engine ↔ `StateMonitor` 的自动接线（回调注册、生命周期联动）—— Phase 6.3；
3. 任务库 / 多任务调度 / 任务优先级与抢占；YAML 中的 `metadata` 目前仅保留；
4. `entry_condition` 已在 schema 与模型中支持，但**进入语义尚未在运行器中强制**
   （当前作为"就绪判据"预留，等待 Phase 6.3 与任务调度一起定稿）；
5. 文件级 E-STOP 的**自动轮询**（当前由引擎每轮 `check()` 驱动，未启动独立 watcher 线程）。

**已知偏差（K-1）**：单文件行数超过 README 规范的 300 行
（`engine.py` 1012 / `definitions.py` 783 / `workflow.py` 656 / `context.py` 588）。
本阶段以"行为不变 + 边界清晰"优先，进一步拆分（例如 `engine` + `engine_workflow`、
`definitions` + `workflow_definitions`）建议放到 Phase 6.3 或 Phase 7 之前统一处理。

---

## 11. 验收结论与下一步

| 验收项 | 结果 |
| --- | --- |
| `pytest` 全套通过 | ✅ 1070 passed / 1 skipped |
| `check_dataset` PASS | ✅（1602 张图，invalid/unresolved/duplicate = 0） |
| 未修改 `runtime/` `vision/` `evidence/` `capture/` `window/` `action/` | ✅（`git status` 仅含 `src/task/`、`configs/tasks/`、`tests/`、`docs/`、`README.md`） |
| 未实现点击 / 键盘 / 游戏操作 | ✅（`ActionPort` = `NullActionPort`；无输入库依赖） |
| 旧 import 兼容 | ✅ `from src.task.models import Task`、`src.task.engine.ActionPort`、`src.task.StepPhase` 均可用 |
| 依赖边界（AST） | ✅ 三条 AST 用例（禁止依赖 / 禁止 `eval` / 白名单） |

**下一步（唯一入口）**：等待用户给出 **Phase 7 Action Layer** 指令。届时需要回答/确认的点：

- 真实 `ActionPort` 的元素解析来源（模板检测结果 / 语义锚点表 / 手工注册的坐标映射）；
- 输入安全约束（坐标必须经 `CoordinateTransform.reference_to_current()`、点击前的前台校验、
  频率限制、E-STOP 二次检查）；
- 动作失败的重试与观测口径（是否进入任务预算、是否记入 Runtime 指标）。

---

## 附录 A：本阶段复核命令

```powershell
# 全套测试与数据集校验
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\check_dataset.py

# 任务定义加载 + 端到端离线驱动
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from src.task import TaskLoader, TaskEngine; d=TaskLoader().load('configs/tasks/farming.yaml'); e=TaskEngine(); e.start(d); [e.on_stable({'state':s}) or e.on_cycle() for s in ('yard','battle','battle_result','yard')]; print(e.state.value, e.result.describe())"

# 依赖边界（AST）
.\.venv\Scripts\python.exe -m pytest tests/test_task_loader.py -q -k "import or eval"

# 非法定义必须失败（示例：循环引用）
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from src.task import validate_task_document as v; v({'name':'x','initial':'A','states':{'A':{'on':[{'condition':'state == yard','next':'B'}]},'B':{'on':[{'condition':'state == yard','next':'A'}]}}})"
```

## 附录 B：公开 API 一览（`src.task`，61 个符号）

```text
定义与模型：Task / Step / Budget / TaskPolicies / TaskStatus / StepPhase（=TaskStatus 旁）
           TaskDefinition / WorkflowState / WorkflowTransition / WorkflowAction / TaskLike
           StateSnapshot / RuntimeSnapshot / TaskContext / TaskResult
加载与校验：TaskLoader / load_task / validate_task_document / DEFAULT_TASK_DIR
           TaskDefinitionError
工作流执行：StepExecutor / StepTransition / WorkflowRunner / WorkflowDecision / WorkflowDecisionKind
           StepActionOutcome / ActionPort / ActionRequest / ActionOutcome / ActionStatus
           NullActionPort / TERMINAL_STATE_NAME
条件与钩子：build_condition / build_predicate / parse_condition_expression / Predicate
           Hook / HookEvent / HookKind / HookRegistry
           CONDITION_FIELDS / CONDITION_TYPE_ALIASES / PREDICATE_BUILDERS
生命周期  ：TaskEngine / TaskEngineError / TaskEngineBusyError
           TaskState / TaskStateMachine / TaskTransition / ALLOWED_TRANSITIONS
           TERMINAL_TASK_STATES / InvalidTaskTransition / TaskError / local_now_iso
安全      ：EStop / EStopSource / EStopState / DEFAULT_SENTINEL_PATH
常量      ：ACTION_KINDS / DEFAULT_ACTION_TIMEOUT_S / DEFAULT_STEP_TIMEOUT_S
```

## 附录 C：术语表

| 术语 | 含义 |
| --- | --- |
| Task Definition | 声明式任务定义（YAML），本阶段为**状态机式工作流** |
| Workflow State | 工作流中的状态（= 用户语言里的"Step"）：进入时请求动作，等待条件满足后转移 |
| Transition | 条件 + 目标状态；`next: COMPLETE` 表示工作流成功结束 |
| Semantic Action | 语义动作：元素名 / 参考分辨率坐标（D-3：禁止屏幕绝对坐标） |
| ActionPort | Task 与未来 `src.action` 之间的端口契约（本阶段 `NullActionPort`） |
| E-STOP | 紧急停止：API + 文件哨兵；触发即任务失败且不再请求动作 |
| Blocked | 运行时不可用（故障 / 失焦）时的任务状态；不消耗预算（D-5） |
