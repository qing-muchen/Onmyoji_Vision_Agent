# Phase 7.1 报告 —— Vision-aware Action Resolver

> 状态：**已完成**（等待下一步指令；本阶段不进入 Action 执行优化）
> 前置：Phase 7.0（Action Layer 第一阶段，提交 `2d6b186`）、
> 本阶段审查：`docs/PHASE7_1_ARCHITECTURE_REVIEW.md`
> 本阶段只完成：**视觉目标 → 语义解析 → 安全动作目标**；不实现点击优化、不实现识别、不做任务逻辑。

---

## 1. 架构变化

### 1.1 分层职责（保持不变，只是被"接通"）

```text
Vision   ：回答"目标在哪里？"                  → EvidenceResult.bbox / DetectionResult.rect
Resolver ：回答"这个语义目标对应哪个可执行位置？" → 本阶段新增 VisionAwareTargetResolver
Action   ：回答"是否允许执行输入？"              → SafetyGuard（5 道门，职责未变）
```

**Action 层依旧不 import `src.vision` / `src.runtime` / `src.task`**（AST 用例固化），
也依旧不读 Frame、不调 detector / OCR / EvidenceRegistry / StateClassifier。

### 1.2 新增文件

| 文件 | 行数 | 内容 |
| --- | --- | --- |
| `src/core/vision_target.py` | 309 | **Vision Target Contract**：`VisionTargetCandidate` / `TargetSnapshot` + 比例换算纯函数 |
| `docs/PHASE7_1_ARCHITECTURE_REVIEW.md` | 285 | 本阶段架构审查（先于代码） |
| `tests/test_vision_action_resolver.py` | 854 | Phase 7.1 测试（38 用例） |

### 1.3 修改文件（最小且不破坏冻结边界）

| 文件 | 变更 |
| --- | --- |
| `src/action/resolver.py` | 新增 `TargetResolution` / `TargetResolutionReason` / `ResolverWithReason`（增量协议）、`VisionAwareTargetResolver`、`resolver_from_config()`；`RegistryTargetResolver` 只**新增** `resolve_detail()`（原有 `resolve()` 行为一字未改） |
| `src/action/dispatcher.py` | `_resolve()` 优先消费 `resolve_detail()` 并把失败原因写入 `ActionOutcome`（`stage` 仍为 `target`）；`_with_resolution_reason()`；`from_config()` 改用 `resolver_from_config`（**Dispatcher 仍不知道任何视觉概念**） |
| `src/action/__init__.py` | 导出 55 个符号（+7） |
| `src/core/__init__.py` | 导出契约（4 个符号） |
| `src/core/config.py` | `ActionConfig` 新增 `resolver` / `vision_target_ttl_s` / `vision_target_min_confidence`（+3 个默认常量） |
| `configs/config.yaml` | `action:` 段新增三项（默认 `resolver: registry` = Phase 7.0 行为） |
| `tests/test_action_layer.py` | 1 处断言细化（未注册目标的 reason 现在更具体：`target_not_found`） |
| `README.md` / `docs/ARCHITECTURE_OVERVIEW.md` | 同步 |

### 1.4 冻结边界（**未改动**）

`TargetResolver.resolve(SemanticTarget) -> ResolvedTarget | None`、`ResolvedTarget`、
`SemanticTarget → ResolvedTarget → client_point(client_rect)` 的坐标链路、
`SafetyGuard` 的 5 道门与其顺序、`InputBackend` 契约 —— 全部保持 Phase 7.0 的形态。
新增能力一律以**可选协议 / 新类**的方式提供。

### 1.5 新增接口一览

```python
# src/core/vision_target.py —— Vision 与 Action 共同依赖的契约（Action 仍不依赖 vision）
VisionTargetCandidate(target_name, x_ratio, y_ratio, confidence, bbox=None, source, timestamp, metadata)
TargetSnapshot.create(candidates, *, clock=monotonic, now=now_iso)
snapshot.get(name) -> VisionTargetCandidate | None        # 大小写不敏感，O(1)
ratio_from_rect(rect, frame_size) -> (x_ratio, y_ratio)
ratio_from_screen_point(point, client_rect) -> (x_ratio, y_ratio)

# src/action/resolver.py —— 解析实现（协议不变，能力增强）
VisionAwareTargetResolver(*, ttl_s, minimum_confidence, clock, aliases, trust_vision_targets, ...)
resolver.update_snapshot(snapshot) / resolver.update_candidates(candidates, ...)   # 单向注入
resolver.resolve_detail(target) -> TargetResolution          # 可选增强协议
TargetResolutionReason = RESOLVED | NO_TARGET | TARGET_NOT_FOUND | TARGET_EXPIRED
                       | LOW_CONFIDENCE | INVALID_TARGET | RESOLVER_ERROR
```

---

## 2. Vision → Resolver → Action 数据流

```text
┌─ Vision（现有模块，本阶段未修改）────────────────────────────────────────────┐
│ EvidenceFrameProcessor → EvidenceResult{name, bbox(帧像素), confidence, source}      │
│ DetectionResult{element_key, rect(参考分辨率), confidence}                            │
└───────────────────────────┬──────────────────────────────────────────────────────┘
                            │ ① adapter（由调用方 / 后续编排层提供；见 2.2 示例）
                            ▼
             VisionTargetCandidate[]{target_name, x_ratio, y_ratio, confidence, source}
                            │ ② TargetSnapshot.create(...)（打时间戳 + 建索引）
                            ▼
              TargetSnapshot  ──③ push（单向）──▶ resolver.update_snapshot(snapshot)
                            │
┌─ Action（本阶段新增实现；边界不变）──────────────────────────────────────────┐
│ VisionAwareTargetResolver.resolve_detail(SemanticTarget) → TargetResolution        │
│ ActionDispatcher（**不知道 Vision 存在**：只持有 TargetResolver 协议）               │
│   SafetyGuard 5 道门 → client_rect × ratio → InputBackend（默认 dry-run）           │
└────────────────────────────────────────────────────────────────────────────────────┘
```

关键性质：

| 性质 | 落实方式 |
| --- | --- |
| **单向流动** | Action 从不主动读取 vision；只有外部把快照 push 给 resolver（`update_snapshot` / `update_candidates`） |
| **Dispatcher 无感知** | `dispatcher.py` 内不出现 `VisionAware*` / `TargetSnapshot` / `VisionTargetCandidate` 字样（AST + 文本断言） |
| **实现选择在 resolver 模块** | `resolver_from_config()`（不在 dispatcher），因此工厂的"视觉知识"不会渗入分发器 |
| **可替换** | `dispatcher.set_resolver()` 运行期切换实现（注册表 ↔ 视觉） |

### 2.1 端到端实测（Task → Vision → Resolver → Action，dry-run）

```text
=== 1) 无视觉快照：目标无法解析（拒绝，不猜坐标） ===
task: failed | backend calls: 0
   → reason: 语义目标不可用（target_not_found（name=buttons/explore, origin=vision, snapshot=none））

=== 2) 注入视觉候选后重试（别名 buttons/explore → explore_button） ===
task: running | wf: OPEN_EXPLORATION
  backend call: click((1495.4, 1337), button=left, count=1)      ← dry-run，只记录
  result: completed（OPEN_EXPLORATION -> COMPLETE），steps=2/3

=== 3) 快照过期（TTL 1.0s）后请求被拒绝 ===
  is_stale: True
  outcome: rejected | stage: target | reason: 语义目标不可用（target_expired（name=explore_button, age_s=5.0, ttl_s=1.0））
```

坐标核对：客户区 `Rect(47, 473, 1704, 960)` × 候选 `(0.85, 0.90)` = `(47+1448.4, 473+864)` = **(1495.4, 1337.0)** ✓ 与视觉候选一致。

### 2.2 Vision 侧 adapter 示例（本阶段**不落地为模块**，保持 vision 目录不动）

```python
# 由 orchestration 层 / 视觉侧（Phase 7.2）实现；依赖 EvidenceResult 与 Frame，因此不能放在 src/action
from src.core.geometry import Size, Point
from src.core.vision_target import VisionTargetCandidate, TargetSnapshot, ratio_from_rect

def candidate_from_evidence(item: EvidenceResult, frame: Frame) -> VisionTargetCandidate | None:
    if not item.detected or item.bbox is None:
        return None                                   # 未检出 → 不上报（Resolver 会拒绝）
    x_ratio, y_ratio = ratio_from_rect(item.bbox, Size(width=frame.width, height=frame.height))
    return VisionTargetCandidate(
        target_name=item.name,                        # 例如 battle_auto_button
        x_ratio=x_ratio, y_ratio=y_ratio,
        confidence=item.confidence,
        bbox=item.bbox, source=item.source, timestamp=frame.timestamp,
        metadata={**item.metadata, "group": item.group},
    )

snapshot = TargetSnapshot.create([c for c in map(partial(...), evidences) if c is not None])
resolver.update_snapshot(snapshot)                    # 单向 push
```

对 `DetectionResult`（参考分辨率的 `rect`）：先 `transform.reference_to_current(rect.center())`
得到客户区像素点，再加客户区屏幕原点，最后用 `ratio_from_screen_point(point, client_rect)`。

---

## 3. 坐标转换策略

```text
① 参考分辨率坐标   DetectionResult.rect          1920×1080 逻辑坐标
                     │ CoordinateTransform.reference_to_current（既有工具，纯缩放）
                     ▼
② 客户区像素坐标   EvidenceResult.bbox / Frame 像素   （采集区域 == 窗口客户区）
                     │ ratio_from_rect / ratio_from_screen_point（本阶段新增的纯函数）
                     ▼
③ 窗口相对比例     VisionTargetCandidate{x_ratio, y_ratio}    ← 契约只接受这一种定位表达
                     │ ResolvedTarget.client_point(client_rect)（Phase 7.0，唯一换算点）
                     ▼
④ 屏幕绝对坐标     Point → SendInput 归一化（虚拟桌面 0..65535）
```

* **没有第二套坐标系**：采集区域就是客户区（`window_capture` 的头注释已声明），
  所以帧像素与客户区像素是同一坐标系；比例换算只需除以帧尺寸；
* **不静默裁剪**：矩形中心若落在窗口之外，`ratio_from_rect` 会返回越界值，
  `VisionTargetCandidate.is_valid` 判定为非法 → 解析失败（`invalid_target`），而不是被夹到边缘；
* **往返自洽**：`ratio_from_screen_point(...)` 与 `ResolvedTarget.client_point(...)` 互为逆运算
  （有单测断言 `(1495.4, 1337.0) ↔ (0.85, 0.90)`）；
* 窗口客户区屏幕矩形仍由 Action 自己的 `Win32WindowProbe` 提供（未新增 vision 接口）。

---

## 4. TTL / confidence 策略

### 4.1 检查顺序与拒绝原因（**任一失败即返回 ``None``**）

| # | 检查 | 不通过时 | 判定式 |
| --- | --- | --- | --- |
| 1 | 快照中存在该目标吗？ | `target_not_found` | `snapshot.get(name) is None` |
| 2 | 快照过期了吗？ | `target_expired` | `age_s > vision_target_ttl_s`（**严格大于**） |
| 3 | 置信度达标吗？ | `low_confidence` | `confidence < vision_target_min_confidence`（**严格小于**） |
| 4 | 坐标合法吗？ | `invalid_target` | `not candidate.is_valid`（比例越界 / NaN） |
| 5 | 通过 | — | `ResolvedTarget(source="vision")` |

额外原因：`no_target`（请求本身没有目标，例如 `wait`）、`resolver_error`（解析器异常被隔离）。

### 4.2 参数与默认值

| 参数 | 默认 | 位置 | 理由 |
| --- | --- | --- | --- |
| `vision_target_ttl_s` | `1.5` | `configs/config.yaml` → `action.vision_target_ttl_s` | 500ms 节拍下约 3 个周期：容忍 1–2 帧抖动，但不允许"陈旧坐标" |
| `vision_target_min_confidence` | `0.6` | `action.vision_target_min_confidence` | 与 `evidence.aggregator.rules[*].min_confidence` 的默认下限一致 |
| `ttl_s = 0` | — | 代码参数 | 显式关闭过期判定（由调用方负责刷新；默认不这么做） |
| `aliases` | 空 | 代码参数 | 语义名 ↔ 视觉名对齐（例如 `buttons/explore` → `explore_button`） |
| `trust_vision_targets` | `True` | 代码参数 | 视觉来源 + 通过 TTL/置信度门 ⇒ 视为"已标定"；置 `False` 时真实后端会拒绝（Q-7.1-1） |

### 4.3 性能（用户要求 O(1)）

* 查询 = **一次字典查找（`casefold` 索引）+ 常数次比较**；
* 不跑检测、不读截图、不调用视觉模型、不做 IO；
* 快照是**不可变对象**，注入即整体替换（无锁、无增量合并）；
* 单测断言：500 个候选下连续 100 次解析只触发 100 次时钟读取（每次解析 1 次，用于 TTL 判定）✓

---

## 5. 安全边界

### 5.1 门控顺序（Job 保持 Phase 7.0 冻结的顺序）

```text
E-STOP → Window → Foreground → **Target Resolver（本阶段接入）** → Target Validation → Rate Limit → Backend
```

Resolver 被夹在"前台检查"与"目标校验/限流"之间，**没有新增门、没有改变顺序**：

| 场景 | 结果 |
| --- | --- |
| E-STOP 生效（即使视觉目标可用） | `REJECTED`，`stage=estop`（窗口探测都不被调用） |
| 窗口无效 / 失焦 | `REJECTED`，`stage=window` / `stage=foreground`（**先于解析**） |
| 目标在快照中缺失 / 过期 / 低置信 / 非法 | `REJECTED`，`stage=target`（`detail.resolution_reason` 给出具体原因） |
| Resolver 抛异常 | `REJECTED`，`stage=target`，`resolution_reason=resolver_error`（**不是** `FAILED`，不抛出） |
| 频率超限 | `REJECTED`，`stage=rate_limit` |

**禁止项**（代码层保证）：无 fallback click、无默认坐标、无"低置信度就用最像的那个"、
无"过期就用上一次"、`RegistryTargetResolver` 不会被当作视觉失败的回退。
`drag` 仍未实现（明确 `REJECTED`）。

### 5.2 默认仍为"零真实输入"

```text
action.enabled = false      → DryRunBackend（只记录，不产生任何输入）
action.backend = dry_run
action.resolver = registry  → 默认不使用视觉解析（Phase 7.0 行为）
```

只有显式把 `resolver` 设为 `vision` **并且**外部注入快照、**并且**开启真实后端时，
才可能产生输入；届时仍需通过 5 道门（含"已标定"这一项）。

### 5.3 禁止修改范围的遵守情况

| 路径 | 状态 |
| --- | --- |
| `src/runtime/`、`src/task/` | **未修改** |
| `src/vision/evidence/`、`src/vision/state/` | **未修改** |
| `src/vision/capture/`、`src/vision/detector/`、`src/vision/window/` 等 | **未修改** |
| `EvidenceResult` / `StateResult` / Runtime 生命周期 | **未修改** |
| 新增 vision 侧文件 | **未新增**（adapter 示例放在报告里，等编排层落地） |

---

## 6. 测试结果

```text
$ python -m pytest -q
1166 passed, 1 skipped in 62.13s
```

| 项 | 数值 |
| --- | --- |
| 全套基线（Phase 7.0 后） | 1128 passed / 1 skipped |
| 本阶段新增 | **+38**（`tests/test_vision_action_resolver.py`） |
| 现基线 | **1166 passed / 1 skipped** |
| 既有 7.0 用例 | 58 passed（仅 1 处断言细化：未注册目标的 reason 由"未解析"改为 `target_not_found`） |

### 6.1 覆盖矩阵（用户要求的 9 项）

| # | 要求 | 用例（节选） |
| --- | --- | --- |
| 1 | SemanticTarget → Candidate → ResolvedTarget 成功 | `test_vision_candidate_resolves_to_resolved_target`、`test_candidate_matching_is_case_insensitive`、`test_alias_mapping_supports_different_naming`、`test_snapshot_replacement_overrides_previous_candidates`、`test_clear_snapshot_makes_targets_unresolvable`、`test_resolver_satisfies_frozen_protocol` |
| 2 | 目标不存在 → Rejected | `test_target_not_found_rejection`、`test_target_not_found_without_snapshot`、`test_no_target_rejection` |
| 3 | 低 confidence → Rejected | `test_low_confidence_rejection`、`test_confidence_boundary_is_inclusive` |
| 4 | TTL 超时 → Rejected | `test_target_expired_rejection`、`test_ttl_boundary_is_inclusive`、`test_ttl_zero_disables_expiry` |
| 5 | 坐标非法 → Rejected | `test_invalid_target_rejection`、`test_rect_outside_frame_is_flagged_invalid` |
| 6 | 旧 RegistryTargetResolver 行为不变 | `test_registry_resolver_behaviour_unchanged`、`test_registry_entries_accept_vision_style_names` |
| 7 | Dispatcher 不知道 Vision | `test_dispatcher_is_agnostic_of_vision`（AST + 文本）、`test_dispatcher_uses_injected_resolver_without_knowing_source`、`test_dispatcher_set_resolver_switches_implementation` |
| 8 | Action 包依赖检查（禁 vision / evidence / runtime / task / cv2 / numpy） | `test_action_layer_forbidden_imports_absent`、`test_action_layer_allowed_imports_only`、`test_contract_module_has_no_vision_types` |
| 9 | 无真实输入（DryRun / Recording） | `test_no_real_input_is_produced`、`test_resolution_is_constant_time_dict_lookup`、`test_candidate_construction_does_not_require_frames` |

另含安全语义回归：`test_resolver_failure_is_rejected_not_failed`（Resolver 异常 → `REJECTED` 而非 `FAILED`）、
`test_guard_order_unchanged_with_vision_resolver`、`test_vision_target_still_requires_valid_window`、
`test_null_window_probe_keeps_vision_resolution_safe`、`test_trust_vision_targets_flag_controls_calibration`。

### 6.2 数据集校验

```text
$ python tools/check_dataset.py
Dataset: PASS   # 1602 张图；invalid / unresolved / duplicate = 0
```

### 6.3 依赖边界复核

```text
src/action/**    → 只 import: stdlib + src.core + src.utils + src.action
                     （不含 cv2 / numpy / PIL / torch / pyautogui / pynput / keyboard /
                       src.vision / src.evidence / src.task / src.runtime）
src/core/vision_target.py → 只 import stdlib + src.core.geometry
dispatcher.py    → 文本与 AST 中均不出现 VisionAware / TargetSnapshot / VisionTargetCandidate
```

---

## 7. 未解决问题

| # | 问题 | 现状 / 建议 |
| --- | --- | --- |
| Q-7.1-1 | 视觉来源是否自动视为"已标定"（`calibrated=True`）？ | 暂定：**是**，但需同时满足"通过 TTL/置信度门 + `trust_vision_targets=True`"；`source="vision"` 保留在解析结果中，便于审计与后续收紧 |
| Q-7.1-2 | TTL / 置信度默认值是否合适？ | 暂定 `1.5s` / `0.6`；建议按实机节拍（500ms）与误点代价在 Phase 7.2 联调时校准 |
| Q-7.1-3 | 快照是否需要"帧号/单调递增 ID"以防止旧快照晚到覆盖新快照？ | 未实现（当前以 `captured_monotonic` 判新旧；若注入方乱序，可加 `frame_id` 单调校验） |
| Q-7.1-4 | 多目标歧义（同名多个候选 / 多个相似按钮） | 未实现：当前"名称精确匹配（+别名）+ 取该名称的唯一候选"；如需要，应由**生成候选的一方**先做筛选（Resolver 不做启发式选择） |
| Q-7.1-5 | Vision adapter 落在哪个模块？ | 本阶段不落地（需依赖 `EvidenceResult` / `Frame`）；建议 Phase 7.2 在编排层新增 `src/orchestration/vision_target_adapter.py`（或视觉侧新增文件），报告 2.2 已给出实现示例 |
| Q-7.1-6 | `drag`（起点 + 终点）语义 | 仍未实现；Phase 7.1 只做"视觉目标 → 语义解析"，`drag` 需要两个目标，建议在 Phase 7.2 与编排一起定 |
| K-7.1-1 | `resolver.py` 565 行 / `dispatcher.py` 538 行 | 超出 300 行规范（与 K-1 / K-3 同类）；后续可按"协议 / 实现 / 工厂"再拆分 |
| K-7.1-2 | Win32 绑定重复（K-2 未变） | `src/action/backend/windows.py` 与 `src/vision/win32.py` 仍各有一份；建议后续下沉到 `src/core` |

---

## 附录 A：复核命令

```powershell
# 全套测试 + 数据集
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\check_dataset.py

# Phase 7.1 定点测试（解析语义 / TTL / 依赖边界）
.\.venv\Scripts\python.exe -m pytest tests/test_vision_action_resolver.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_vision_action_resolver.py -q -k "expired or confidence or invalid or not_found"

# 默认安全（resolver=registry；未注册目标 → target_not_found）
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from src.action import ActionDispatcher, ActionRequest, ActionType, SemanticTarget; d=ActionDispatcher(); o=d.request(ActionRequest(request_id='r', action_type=ActionType.CLICK, target=SemanticTarget.from_name('NOPE'))); print(o.status.value, o.detail.get('stage'), o.detail.get('resolution_reason'))"

# 视觉解析（无快照 → target_not_found；注入后 → source=vision）
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from src.action import VisionAwareTargetResolver, SemanticTarget; from src.core.vision_target import VisionTargetCandidate; r=VisionAwareTargetResolver(); print(r.resolve_detail(SemanticTarget.from_name('battle_auto_button')).reason.value); r.update_candidates([VisionTargetCandidate(target_name='battle_auto_button', x_ratio=0.82, y_ratio=0.91, confidence=0.96)]); print(r.resolve_detail(SemanticTarget.from_name('battle_auto_button')).describe())"
```

## 附录 B：公开 API 增量（`src.action`，55 个符号）

```text
新增：VisionAwareTargetResolver / TargetResolution / TargetResolutionReason / ResolverWithReason
      VISION_TARGET_SOURCE / DEFAULT_VISION_TARGET_TTL_S / DEFAULT_VISION_TARGET_MIN_CONFIDENCE
      resolver_from_config（位于 src.action.resolver）
src.core 新增：VisionTargetCandidate / TargetSnapshot / ratio_from_rect / ratio_from_screen_point
配置新增：action.resolver / action.vision_target_ttl_s / action.vision_target_min_confidence
```

## 附录 C：术语表

| 术语 | 含义 |
| --- | --- |
| VisionTargetCandidate | 视觉层上报的定位候选（名称 + 窗口相对比例 + 置信度 + 可选 bbox + 来源 + 时间戳） |
| TargetSnapshot | 某一时刻的全部候选（按名称 `casefold` 索引；含生成时刻，供 TTL 判定） |
| TTL（`vision_target_ttl_s`） | 候选有效期：超过即 `target_expired`（防止用陈旧坐标点击） |
| `resolution_reason` | `ActionOutcome.detail` 中的解析失败原因（`target_not_found` / `target_expired` / `low_confidence` / `invalid_target` / `resolver_error`） |
| `calibrated` | 该位置是否可视为可信；视觉来源需通过 TTL/置信度门且 `trust_vision_targets=True` |
| 反向注入（push） | 视觉结果由外部推给 Resolver；Action 从不主动读取 vision |
