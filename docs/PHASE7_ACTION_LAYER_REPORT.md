# Phase 7 报告 —— Action Layer Implementation（第一阶段）

> 状态：**第一阶段已完成**（等待 Phase 7.1：Vision-aware Action Resolver）
> 前置：Phase 6.2（任务定义加载与工作流执行，提交 `0850a10`）
> 本阶段范围：**Action 数据模型 / ActionPort / 语义目标解析 / Windows 输入后端 / 安全门控 / 分发流程**；
> **不实现**游戏按钮识别（模板 / OCR / YOLO）、任务逻辑与自动刷任务。

---

## 1. 本阶段目标与范围

用户指令（Phase 7）：

```text
数据流：TaskEngine → ActionRequest → ActionDispatcher → ActionResolver → InputBackend → Mouse / Keyboard
第一阶段只实现：1. Action 数据模型  2. ActionPort  3. 语义目标解析  4. Windows 输入后端
不要实现：游戏按钮识别 / OCR / YOLO / 自动刷任务逻辑 / Task 逻辑
不要修改：runtime/ vision/ evidence/ task/
```

**红线与落实**

| 红线 | 落实方式 |
| --- | --- |
| Task 层只能产生 `ActionRequest`，禁止 import vision / evidence / cv2 / window / pyautogui | Phase 6.2 的 AST 用例继续锁定 `src/task`；本阶段未修改 `src/task` |
| `ActionRequest` 必须含 `request_id` / `action_type` / `target` / `parameters` / `created_at` | `src/action/models.py::ActionRequest`（数据类字段同名，`created_at` 缺省自动填充） |
| **禁止** screen coordinate / pixel x/y | 数据类无坐标字段；`parameters` 的坐标键直接报错；语义目标拒绝 `reference` / `x` / `y` 写法；目标名不得形如 `CLICK_1530_850` |
| `ActionTarget` 用 enum / string，禁止 `CLICK_1530_850` | `ActionTarget`（`StrEnum` 内置名）+ 注册表接受任意自定义名；`validate_target_name()` 拦截坐标式命名 |
| Resolver 不读 Frame、不调 detector、不属于 vision | `resolver.py` 只做"语义名 → 比例"；AST 用例锁定依赖边界；代码级检查（第 8.4 节） |
| 坐标只允许窗口相对比例（`x_ratio` / `y_ratio`） | `ResolvedTarget.client_point(client_rect)` 是**唯一**换算点 |
| 后端用 win32api / ctypes，**不用** pyautogui | `backend/windows.py` 用 `ctypes` + `SendInput`；AST 用例禁止 `pyautogui` / `pynput` |
| 每次动作前 5 道检查，任一失败 → `REJECTED` | `safety/guard.py::SafetyGuard.check_pre/check_post` |
| 禁止 fallback click、禁止猜坐标 | 未解析 → `REJECTED`；未标定 → 真实后端下 `REJECTED`；无"默认坐标"分支 |

---

## 2. 文件清单

### 2.1 新增（`src/action/`）

| 文件 | 行数 | 内容 |
| --- | --- | --- |
| `__init__.py` | 133 | 公共 API 导出（48 个符号）+ 架构说明 |
| `models.py` | 605 | `ActionType` / `ActionStatus` / `ActionTarget` / `SemanticTarget` / `ResolvedTarget` / `ActionRequest` / `ActionOutcome` / 异常 / `validate_target_name` |
| `port.py` | 130 | `ActionPort` 协议 + `NullActionPort` + `as_action_port` / `is_action_request` |
| `resolver.py` | 121 | `TargetResolver` 协议 + `RegistryTargetResolver`（+ `SemanticTarget` re-export） |
| `registry.py` | 295 | `TargetEntry` / `TargetRegistry` / `BUILTIN_TARGETS`（占位）/ `registry_from_mapping` |
| `dispatcher.py` | 486 | `ActionDispatcher`（9 步流程）/ `DispatchStats` |
| `backend/__init__.py` | 66 | `create_input_backend`（安全默认 dry-run） |
| `backend/base.py` | 255 | `InputBackend` 协议 / `BackendCall` / `BackendStats` / `DryRunBackend` / `WindowProbe` / `NullWindowProbe` |
| `backend/windows.py` | 740 | `WindowsMouseBackend` / `WindowsKeyboardBackend` / `WindowsInputBackend` / `Win32WindowProbe` / `resolve_key_code` / `normalize_absolute` |
| `safety/__init__.py` | 40 | 安全子系统导出 |
| `safety/estop.py` | 233 | `EmergencyStop` / `EmergencyStopSource` / `EmergencyStopState`（API + `logs/ESTOP`） |
| `safety/guard.py` | 367 | `SafetyGuard`（5 道门）/ `GuardResult` / `GuardStage` / `RateLimiter` / `build_guard` |

（`src/action` 合计 3471 行。）

### 2.2 新增（其它）

| 文件 | 内容 |
| --- | --- |
| `tests/test_action_layer.py` | Phase 7 测试（58 用例；1128 行；含 AST 依赖边界） |
| `docs/PHASE7_ACTION_LAYER_REPORT.md` | 本文 |

### 2.3 修改

| 文件 | 变更 |
| --- | --- |
| `src/core/config.py` | 新增 `ActionConfig`（`enabled` / `backend` / `require_foreground` / `min_interval_s` / `max_actions_per_minute` / `estop_path` / `move_before_click`）+ 6 个默认常量；`AppConfig.action` |
| `src/core/__init__.py` | 导出 `ActionConfig` |
| `configs/config.yaml` | 新增 `action:` 段（默认 `enabled: false` + `backend: dry_run`） |
| `README.md` | Phase 7 段、目录结构、路线图、"尚未实现" |

**未修改**：`runtime/` `vision/` `evidence/` `capture/` `window/` `task/`（`git diff --name-only` 可复核）。

---

## 3. Action 架构

### 3.1 数据流与模块职责

```text
TaskEngine（Phase 6.2）
   │  ActionRequest（语义目标：{element: ...} / {x_ratio, y_ratio}）
   ▼
ActionDispatcher ──── 编排：门控 → 解析 → 换算 → 执行 → 结果
   │        │
   │        ├── SafetyGuard ──▶ EmergencyStop / WindowProbe / RateLimiter
   │        ├── TargetResolver ──▶ TargetRegistry（Phase 7.1：Vision-aware Resolver）
   │        └── InputBackend ──▶ DryRunBackend（默认）/ WindowsInputBackend（SendInput）
   ▼
ActionOutcome（status / reason / latency_ms / resolved_target）
```

| 模块 | 职责 | 明确不做 |
| --- | --- | --- |
| `models.py` | 语义请求 / 结果 / 目标的数据契约；禁止像素坐标 | 不做定位、不做 IO |
| `port.py` | 协议与空实现（`request()` 形状与 Task 层一致） | 不做策略 |
| `registry.py` | 语义名 → 窗口相对比例（含标定标记） | 不读像素、不做识别 |
| `resolver.py` | `SemanticTarget` → `ResolvedTarget` | **不读 Frame、不调 detector** |
| `dispatcher.py` | 全流程编排 + 统计 + 异常隔离 | 不做识别、不做任务逻辑 |
| `backend/*` | 输入原语与平台实现（dry-run 默认） | 不判断"该不该点" |
| `safety/*` | E-STOP 与 5 道安全门 | 不做定位、不做输入 |

### 3.2 依赖边界

```text
src/action  →  stdlib + src.core + src.utils + src.action      ✅
src/action  →  src.vision / src.evidence / src.task / src.runtime / cv2 / pyautogui / pynput   ❌
```

AST 用例（`tests/test_action_layer.py`）锁定：禁止依赖清单、白名单依赖清单、
无 `eval` / `exec` / `compile`、`resolver.py` 的代码不含 `Frame` / `detector` / `cv2` 名称。

### 3.3 与 Task 层的接口：字段视图（无需适配器、无需改 Task）

```text
Task 侧 ActionRequest                Action 侧 ActionRequest
  action_id   ─────────────────────▶ request_id
  kind        ─────────────────────▶ action_type
  target {element|name} / {x_ratio,y_ratio} ▶ SemanticTarget
  params      ─────────────────────▶ parameters
  task_id / step_id / timeout_s / attempt   同名字段直传

ActionOutcome.is_failure  ◀────────── status ∈ {REJECTED, FAILED}
```

`ActionDispatcher.request()` 精确满足 Task 层 `ActionPort` 的调用形状，
因此**可以直接注入** `TaskEngine(action_port=dispatcher)`，两层互不 import。
端到端冒烟已验证（第 8.3 节）。

### 3.4 为什么用 `SendInput` 而不是 `pyautogui`

| 维度 | `SendInput`（本实现） | `pyautogui`（未采用） |
| --- | --- | --- |
| 输入通道 | 与真实设备同一队列，携带扫描码与时间戳 | `mouse_event` / `SetCursorPos` 等旧接口 |
| 游戏兼容性 | 多数 DirectInput / Raw Input 场景可识别 | 部分游戏收不到或延迟明显 |
| 延迟 | 单次系统调用批量提交事件 | 逐条调用 + 内部 sleep |
| 依赖 | `ctypes`（标准库） | 第三方库（且会连带引入键盘钩子能力） |

因此本项目**不引入** `pyautogui` / `pynput` / 任何全局热键库（Phase 6.2 起即为禁令，本阶段继续）。

---

## 4. 输入流程（一次 `request()` 的 9 步）

```text
0) 请求规范化    Task 侧对象 → ActionRequest（字段视图；非法 / 含坐标 → REJECTED stage=request）
1) E-STOP       门 1：EmergencyStop.check()（API 触发 或 logs/ESTOP 存在）
2) 窗口有效性    门 2：WindowProbe.is_valid()（存在 / 可见 / 未最小化）
3) 前台         门 3：WindowProbe.is_foreground()（require_foreground 可关，但默认开启）
   ── wait 类型在此返回（SKIPPED：不产生输入、不需要目标）──
4) 目标解析      SemanticTarget → ResolvedTarget（未注册 → REJECTED，**不猜坐标**）
5) 目标可用性    门 4：已解析 +（非 dry-run 时）已标定 + confidence > 0
6) 频率限制      门 5：最小间隔 + 每分钟上限
7) 坐标换算      client_rect × (x_ratio, y_ratio) → 屏幕坐标（唯一换算点）
   ── 记账：门控全部通过即占用限流额度 ──
8) 后端执行      move / click / press / release / key_press / key_release（异常隔离）
9) 结果汇总      ActionOutcome（status / reason / latency_ms / resolved_target / detail）
```

### 4.1 状态语义

| `ActionStatus` | 何时返回 | 含义 |
| --- | --- | --- |
| `EXECUTED` | 后端为真实实现且调用成功 | **已提交输入**（不代表界面已响应；效果由状态验证） |
| `SKIPPED` | `wait` 动作 / dry-run 后端 | 已受理但**未产生输入** |
| `REJECTED` | 任一门控失败 / 未实现的动作类型 / 请求非法 | 明确拒绝（带 `stage` 与原因，**绝不 fallback**） |
| `FAILED` | 后端抛异常 | 执行失败（异常已隔离，不会传播给调用方） |

### 4.2 动作类型的后端映射

| 动作 | 后端调用 |
| --- | --- |
| `click` | `click(point, button, count=1)` |
| `double_click` | `click(point, button, count=2)` |
| `long_press` | `press(point)` → 睡眠 `duration_s`（默认 0.6s，注入 sleeper）→ `release(point)` |
| `key` | `key_press(key)` → 睡眠 `hold_s`（默认 0）→ `key_release(key)` |
| `wait` | **无后端调用**（不产生输入） |
| `drag` | `REJECTED`：本阶段未实现（留待 Phase 7.1，避免半成品语义） |

---

## 5. 坐标策略

### 5.1 三层表达（只有最后一层是像素）

```text
① 语义目标      SemanticTarget(name="BATTLE_AUTO_BUTTON")      ← Task / YAML 使用
② 窗口相对比例   ResolvedTarget{x_ratio: 0.92, y_ratio: 0.95}   ← Resolver 输出 / 注册表存储
③ 屏幕坐标      Point(x=47+1704×0.92, y=473+960×0.95)           ← 仅在 Dispatcher 内、送后端前算出
```

规则：

* ①→② 由 `TargetResolver` 完成（本阶段：注册表查表；Phase 7.1：vision 解析）；
* ②→③ 由 `ResolvedTarget.client_point(client_rect)` 完成，**这是唯一的坐标换算点**；
* 客户区矩形来自 `WindowProbe.client_rect()`（`GetClientRect` + `ClientToScreen` → 屏幕坐标）；
* 比例按**客户区**计算（不含标题栏与边框），窗口移动 / 缩放后自动跟随（每次动作前重取）；
* 任何环节拿不到客户区矩形 → `REJECTED`（不猜）。

### 5.2 为什么禁止像素 / 参考分辨率坐标

* 硬编码像素在换分辨率、缩放、窗口移动后全部失效，且会掩盖"目标其实不存在"的事实；
* 参考分辨率坐标（`{"reference": [x, y]}`）在本阶段被**显式拒绝**：它是"另一套绝对坐标"，
  与本阶段的"语义定位"目标冲突——待 Phase 7.1 引入 vision 解析后再讨论是否需要它；
* 目标名同样不得编码坐标（`CLICK_1530_850` 会被 `validate_target_name` 拒绝）。

### 5.3 标定（`calibrated`）与占位值

* `BUILTIN_TARGETS` 里的比例是**占位示例值**（`calibrated=False`），只用于 dry-run 演示与测试；
* 真实输入后端（`is_dry_run=False`）下，未标定目标会被门 4 拒绝：
  `"目标 X 未标定（占位比例不允许用于真实输入）"`；
* 显式比例（`{x_ratio, y_ratio}`）默认也视为未标定，需
  `RegistryTargetResolver(trust_explicit_ratio=True)` 或注册表显式 `add(..., calibrated=True)`；
* 也就是说：**要真实点击，必须先声明"这个比例是可信的"**，而不是让系统猜。

### 5.4 DPI 与多显示器

* 创建 Windows 后端时调用一次 `ensure_dpi_aware()`（`SetProcessDpiAwarenessContext(-4)`，
  失败回退 `SetProcessDPIAware`；幂等）——避免 150% 缩放下的坐标偏移；
* 绝对坐标按**虚拟桌面**归一化到 `0..65535`（`MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK`），
  因此多显示器 / 负坐标显示器同样正确（`normalize_absolute` 有单元测试覆盖）。

---

## 6. 安全机制

### 6.1 五道门（每次动作前必查）

| # | 检查 | 失败原因示例 | 结果 |
| --- | --- | --- | --- |
| 1 | E-STOP（API / `logs/ESTOP`） | `紧急停止生效（user abort）` | `REJECTED` stage=`estop` |
| 2 | 窗口有效性 | `目标窗口无效（Win32WindowProbe(hwnd=0x0, valid=False)）` | `REJECTED` stage=`window` |
| 3 | 前台 | `目标窗口不在前台（不抢焦点、不后台点击）` | `REJECTED` stage=`foreground` |
| 4 | 目标已解析 + 已标定 + 置信度 | `语义目标未解析（不会使用默认坐标兜底）` | `REJECTED` stage=`target` |
| 5 | 频率限制 | `动作间隔过短，请等待 0.180s` | `REJECTED` stage=`rate_limit` |

**顺序是强约束**：E-STOP 先于一切（E-STOP 生效时窗口探测根本不会被调用，有专门用例验证）。

### 6.2 E-STOP（与 Task 层同机制、同哨兵）

```python
estop = EmergencyStop()            # 默认 logs/ESTOP
estop.trigger("manual abort")      # 方式 1：API
# 方式 2：人工创建文件（无需交互）
#   New-Item logs/ESTOP -ItemType File
```

* 触发即 `REJECTED`（原因写入 `detail.estop_source`）；
* 幂等（保留首次原因与时刻）；
* **不自动复位**：必须 `reset()`（默认删除哨兵文件）；
* **禁止全局热键**：不注册任何键盘钩子（AST 用例禁止 `pynput` / `keyboard` 依赖）；
* Task 层与 Action 层**共用同一哨兵文件**，人工停止一次即可同时阻断"决策"与"输入"。

### 6.3 限流

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `min_interval_s` | `0.25` | 两次动作最小间隔（4 次/秒） |
| `max_actions_per_minute` | `60` | 每分钟上限（滑动窗口） |

记账时机：**门控全部通过后**即占用额度（dry-run 与执行失败同样记账）——
保证"决策频率"受控且行为可预测；被拒绝的动作不占额度。

### 6.4 异常隔离

* 后端抛出的任何异常都会转换为 `FAILED(reason="backend_error: ...")`，**不向调用方传播**；
* 解析器异常视为"未解析" → `REJECTED`；
* 请求规范化失败 → `REJECTED(stage=request)`；
* 一次失败不影响后续动作（有专门用例验证"异常后仍能正常执行下一条"）。

### 6.5 禁止项（代码层保证，不是靠约定）

| 禁止 | 实现方式 |
| --- | --- |
| fallback click | 不存在"目标解析失败就点默认位置"的分支；未解析直接拒绝 |
| 猜坐标 | 未注册目标不解析；未标定目标在真实后端下拒绝 |
| 静默降级 | 请求 Windows 后端但平台不支持 → 抛 `InputBackendUnavailableError`（**不**回退 dry-run） |
| 半成品动作 | `drag` 明确 `REJECTED`（提示 Phase 7.1），不降级成 click |
| 后台点击 | 默认 `require_foreground=true`，失焦即拒绝（不抢焦点） |

---

## 7. 配置

`configs/config.yaml`：

```yaml
action:
  enabled: false            # 安全默认：真实输入必须显式开启
  backend: "dry_run"        # dry_run（默认）/ windows（SendInput）
  require_foreground: true  # 失焦即拒绝（不抢焦点）
  min_interval_s: 0.25
  max_actions_per_minute: 60
  estop_path: "logs/ESTOP"
  move_before_click: true
```

```python
from src.core import load_config
from src.action import ActionDispatcher

dispatcher = ActionDispatcher.from_config(load_config())   # → DryRunBackend（安全默认）
```

`create_input_backend()` 的规则：无配置 / `enabled=false` / `backend != windows` → `DryRunBackend`；
`enabled=true` + `backend=windows` → `WindowsInputBackend`（平台不支持时报错）。

---

## 8. 测试结果

```text
$ python -m pytest -q
1128 passed, 1 skipped in 87.27s
```

| 项 | 数值 |
| --- | --- |
| 全套基线（Phase 6.2 后） | 1070 passed / 1 skipped |
| 本阶段新增 | **+58**（`tests/test_action_layer.py`） |
| 现基线 | **1128 passed / 1 skipped** |

### 8.1 覆盖矩阵（用户要求的 10 项）

| # | 要求 | 用例（节选） |
| --- | --- | --- |
| 1 | ActionRequest schema | `test_action_request_required_fields`、`test_action_request_serialization_has_no_coordinates`、`test_action_request_requires_target_except_wait`、`test_action_request_rejects_invalid_numeric_fields` |
| 2 | 禁止 pixel coordinate | `test_action_request_has_no_pixel_fields`、`test_action_request_rejects_coordinate_parameters`、`test_semantic_target_rejects_pixel_writings`、`test_semantic_target_rejects_coordinates_in_name`、`test_resolved_target_exposes_only_ratios`、`test_from_task_request_rejects_reference_target` |
| 3 | SemanticTarget resolution | `test_registry_resolution`、`test_builtin_registry_entries_are_placeholders`、`test_unregistered_target_is_not_resolved`、`test_explicit_ratio_resolution_and_trust`、`test_registry_mapping_and_custom_names` |
| 4 | Resolver 不依赖 vision | `test_action_layer_forbidden_imports_absent`、`test_action_layer_allowed_imports_only`、`test_action_layer_has_no_eval_or_exec`、`test_resolver_module_does_not_touch_vision_concepts` |
| 5 | DryRun 执行 | `test_dry_run_backend_records_without_input`（含坐标断言 900,650）、`test_executed_path_with_real_like_backend`、`test_double_click_and_long_press_and_key`、`test_wait_needs_no_target_and_skips` |
| 6 | E-STOP 拒绝 | `test_estop_api_triggers_rejection`、`test_estop_sentinel_file_rejection`、`test_estop_reset_allows_actions_again`、`test_estop_state_contains_no_user_data` |
| 7 | Foreground 拒绝 | `test_foreground_rejection_and_opt_out`、`test_window_invalid_rejection`、`test_missing_client_rect_rejection`、`test_null_window_probe_is_safe_default`、`test_uncalibrated_target_rejected_for_real_backend` |
| 8 | Backend 异常隔离 | `test_backend_exception_is_isolated`、`test_dispatcher_survives_backend_failure` |
| 9 | Dispatcher 流程 | `test_guard_order_estop_before_window`、`test_probe_call_order`、`test_rate_limit_rejection_and_recovery`、`test_rate_limiter_window_limit`、`test_latency_and_stats_recorded`、`test_stats_reject_stage_counting`、`test_dispatcher_accepts_task_side_request_object`、`test_invalid_request_is_rejected_not_raised` |
| 10 | Windows backend 接口 | `test_windows_backend_classes_and_dry_run`、`test_windows_sendinput_structures`、`test_key_code_resolution`、`test_normalize_absolute_mapping`、`test_platform_guard_for_windows_backend`、`test_create_input_backend_defaults_to_dry_run` |

**测试安全**：全部用例使用 `DryRunBackend` 或 `RecordingBackend`（记录但不发送）；
Windows 后端只用 `dry_run=True` 与纯函数（键码 / 归一化）验证——**测试过程零真实输入**。

### 8.2 数据集校验

```text
$ python tools/check_dataset.py
Dataset: PASS   # 1602 张图；invalid / unresolved / duplicate = 0
```

### 8.3 端到端冒烟（Task → Action，dry-run，零真实输入）

```text
start -> running state = WAIT_YARD
  observe yard         -> task=running    wf=OPEN_EXPLORATION backend_calls=1
  observe exploration  -> task=completed  wf=OPEN_EXPLORATION backend_calls=1
result: task=open_exploration status=completed reason=OPEN_EXPLORATION -> COMPLETE steps=2/3
stats: {'requests': 1, 'executed': 0, 'skipped': 1, 'rejected': 0, 'failed': 0, ...}
  backend call: click((1495.4, 1337), button=left, count=1)
transitions: ['WAIT_YARD -> OPEN_EXPLORATION', 'OPEN_EXPLORATION -> COMPLETE']
```

坐标换算核对：客户区 `Rect(47, 473, 1704, 960)`（本机实测阴阳师窗口客户区）×

`x_ratio=0.85` → 47 + 1704 × 0.85 = **1495.4**；`y_ratio=0.90` → 473 + 960 × 0.90 = **1337.0** ✓

结论：`TaskEngine(action_port=ActionDispatcher(...))` 可直接工作（字段视图对接），
决策链路完整，且 dry-run 下**未产生任何真实输入**。

### 8.4 依赖边界复核

```text
src/action 的 import roots:
['__future__', 'collections', 'ctypes', 'dataclasses', 'datetime', 'enum', 'pathlib', 're',
 'src', 'sys', 'time', 'typing']

不包含: cv2 / numpy / PIL / torch / pyautogui / pynput / keyboard /
        src.vision / src.evidence / src.task / src.runtime
```

---

## 9. 未实现与已知偏差

| 项 | 说明 | 计划 |
| --- | --- | --- |
| **游戏按钮识别** | 本阶段不识别界面元素（无模板匹配 / OCR / YOLO 接入） | Phase 7.1：Vision-aware Action Resolver |
| `drag` 动作 | 需要"起点 + 终点"两个语义目标，语义未定 | Phase 7.1 / 7.2 |
| 真实输入验证 | 本阶段**未**做过任何真实点击（安全考虑） | 由用户在有 E-STOP 就绪时人工验收 |
| 与 `StateMonitor` 的接线 | Task/Action 均由调用方手工驱动 | Phase 7.2（编排） |
| 任务库 / 多任务调度 | 未实现 | Phase 8 |
| **已知偏差 K-2** | `backend/windows.py` 与 `src/vision/win32.py` 存在 Win32 绑定重复实现（Action 层不得 import vision） | 建议后续统一下沉到 `src/core`（属重构，本阶段不动 vision） |

**容量说明 K-3**：`models.py` 600 行 / `backend/windows.py` 700 行 / `dispatcher.py` 501 行，
超出 README 的"单文件 300 行"规范（与 Phase 6.2 的 K-1 同类），后续可继续拆分。

---

## 10. 验收结论与下一步

| 验收项 | 结果 |
| --- | --- |
| `pytest` 全套通过 | ✅ 1128 passed / 1 skipped |
| `check_dataset` PASS | ✅（1602 张图，invalid/unresolved/duplicate = 0） |
| `git clean` | ✅（提交后工作区干净） |
| 未修改 `runtime/` `vision/` `evidence/` `capture/` `window/` `task/` | ✅ |
| 未实现点击以外的游戏逻辑 | ✅（无识别、无任务逻辑） |
| 默认无真实输入 | ✅（`action.enabled=false` → `DryRunBackend`） |

**下一步**：等待 **Phase 7.1：Vision-aware Action Resolver**。届时需要确定的点：

1. **解析输入**：resolver 从"最近一次检测结果 / 证据"取元素位置（谁提供、如何注入、TTL 多长）；
2. **坐标来源**：检测框 → 比例（经 `CoordinateTransform` 或元素 bbox 归一化）；
3. **置信度语义**：检测得分如何映射到 `ResolvedTarget.confidence`，阈值多少才允许动作；
4. **失败与回退**：解析失败 / 置信度不足时的任务级策略（重试解析 / 等下一帧 / 任务失败）；
5. **标定**：`calibrated` 是否由 vision 来源自动置真，还是仍需人工声明。

---

## 附录 A：复核命令

```powershell
# 全套测试 + 数据集
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\check_dataset.py

# 依赖边界 / 无动态执行
.\.venv\Scripts\python.exe -m pytest tests/test_action_layer.py -q -k "import or eval or vision"

# 默认安全（dry-run + 未注册目标 → REJECTED）
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from src.action import ActionDispatcher, ActionRequest, ActionType, SemanticTarget; d=ActionDispatcher(); o=d.request(ActionRequest(request_id='r', action_type=ActionType.CLICK, target=SemanticTarget.from_name('NOPE'))); print(o.status.value, o.detail.get('stage'), o.reason)"

# E-STOP 哨兵（人工停止）
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from src.action import EmergencyStop; e=EmergencyStop(); e.create_sentinel('manual'); print(e.check(), e.source); e.reset()"
```

## 附录 B：公开 API（`src.action`，48 个符号）

```text
模型      ：ActionType / ActionStatus / ActionTarget / SemanticTarget / ResolvedTarget
            ActionRequest / ActionOutcome / now_iso / outcome_from_any / validate_target_name
            FORBIDDEN_PARAMETER_KEYS
异常      ：ActionError / ActionRequestError / ActionTargetError / ActionResolutionError
            InputBackendError / InputBackendUnavailableError
端口      ：ActionPort / NullActionPort / as_action_port / is_action_request
解析      ：TargetResolver / RegistryTargetResolver / EXPLICIT_RATIO_TARGET
注册表    ：TargetRegistry / TargetEntry / BUILTIN_TARGETS / registry_from_mapping
分发      ：ActionDispatcher / DispatchStats
后端      ：InputBackend / DryRunBackend / BackendCall / MOUSE_BUTTONS / create_input_backend
            WindowProbe / NullWindowProbe
安全      ：EmergencyStop / EmergencyStopSource / EmergencyStopState / DEFAULT_ACTION_SENTINEL_PATH
            SafetyGuard / GuardResult / GuardStage / RateLimiter / build_guard
            DEFAULT_MIN_INTERVAL_S / DEFAULT_MAX_ACTIONS_PER_MINUTE
```

## 附录 C：术语表

| 术语 | 含义 |
| --- | --- |
| SemanticTarget | 语义目标：语义名（`BATTLE_AUTO_BUTTON`）或窗口相对比例（`x_ratio` / `y_ratio`） |
| ResolvedTarget | 解析结果：窗口相对位置 + 置信度 + 来源 + 是否已标定（**无像素**） |
| TargetResolver | 语义目标 → 解析结果（Phase 7 用注册表；Phase 7.1 用 vision） |
| calibrated | 该比例是否经过实机标定；未标定目标不允许用于真实输入 |
| SafetyGuard | 5 道门：E-STOP → 窗口 → 前台 → 目标已解析 → 频率限制 |
| DryRunBackend | 默认后端：记录调用、不产生任何输入 |
| stage | 拒绝发生的阶段（`estop` / `window` / `foreground` / `target` / `rate_limit` / `request` / `action_type`） |
