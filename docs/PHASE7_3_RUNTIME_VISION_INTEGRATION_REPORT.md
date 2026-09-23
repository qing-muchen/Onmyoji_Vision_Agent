# Phase 7.3 报告 —— Runtime Vision Snapshot Integration

> 状态：**已完成**（等待下一步指令）
> 前置：Phase 7.2（VisionTargetAdapter，提交 `cc3ef51`）
> 本阶段只完成："**实时视觉结果如何进入 Action 消费链路**"；
> 不新增视觉算法、不做 OCR / YOLO / 模板匹配、不做点击逻辑、不做 Task 设计。

---

## 1. 架构变化

### 1.1 链路贯通

```text
Runtime Cycle（StateMonitor.run）
  │  frame = source.get_frame()                     ← capture（失败 → 本轮不更新）
  │  processor.detect(frame) → [EvidenceResult]     ← **既有**视觉流水线（不新增 detector）
  │  classifier.classify(evidence) → StateResult    ← **既有**状态判定
  ▼
VisionSnapshotBridge.update(frame, evidences, frame_id=monitor.cycles, cycle_id=schedule.cycle_id)
  │  ① VisionTargetAdapter → [VisionTargetCandidate]      （Phase 7.2）
  │  ② TargetSnapshot{frame_id, captured_at, captured_monotonic, candidates}
  │  ③ resolver.update_snapshot(snapshot)                  （单向 push；Phase 7.1）
  ▼
Action：VisionAwareTargetResolver → ActionDispatcher → SafetyGuard → InputBackend（默认 DryRun）
```

### 1.2 新增 / 修改

| 文件 | 行数 | 变更 |
| --- | --- | --- |
| `src/runtime/vision_bridge.py` | 288（新增） | `VisionSnapshotBridge` + `VisionBridgeStats` |
| `tests/test_runtime_vision_bridge.py` | 678（新增） | 21 用例 |
| `src/runtime/state_monitor/monitor.py` | +90 行 | 可选 `vision_bridge` 参数、`last_evidences`（同一次检测的产物）、`_publish_vision_snapshot()`、`run()` 每轮调用点 |
| `src/runtime/__init__.py` | docstring 更新 | 声明 Runtime 作为唯一编排层：可依赖 vision / action 的**解析器接口**，仍不触发输入 |
| `README.md` / `docs/ARCHITECTURE_OVERVIEW.md` | 文档同步 | Phase 7.3 段、目录结构、主链路、阶段索引、路线图 |

### 1.3 对既有模块的改动（逐条说明"为什么必要"）

用户要求"不修改 `SafetyGuard` / `dispatcher.py` 门控顺序 / `EvidenceResult` / `StateResult` /
Runtime 生命周期接口，除非必要，需先说明原因"。本阶段的实际改动：

| 对象 | 是否改动 | 说明 |
| --- | --- | --- |
| `src/action/SafetyGuard` | ❌ 未改 | 5 道门与顺序完全不变 |
| `src/action/dispatcher.py` 核心门控顺序 | ❌ 未改 | 本阶段未触碰 `src/action/**` |
| `EvidenceResult` / `StateResult` | ❌ 未改 | 只**读取**既有字段（`bbox` / `name` / `confidence` / `source` / `group` / `metadata`） |
| Runtime 生命周期接口（`initialize/start/pause/resume/stop`、状态机） | ❌ 未改 | 未触碰生命周期与状态转换表 |
| `StateMonitor.__init__` | ✅ 新增**可选**参数 `vision_bridge=None` | **必要**：Runtime 是唯一能同时看到"帧 / 视觉结果 / 动作端口"的层，接线点必须在 Runtime。默认 `None` ⇒ 行为与 Phase 5.4 / 7.2 **完全一致**（既有 1000+ 用例零改动通过） |
| `StateMonitor._classify()` | ✅ 保留 `self._last_evidences`（1 行 + 2 处置空） | **必要**：证据列表就在该方法内产出（`detect()` 的返回值），保存引用即可让桥消费**同一次检测**的结果，避免重复 detect（性能 ×2）。不改变任何返回值与行为 |
| `StateMonitor.run()` | ✅ 每轮末尾调用 `_publish_vision_snapshot()` | **必要**：把推送挂在既有循环中（在 `on_cycle` 之后）。未接线时该方法是空操作 |
| `src/runtime/__init__.py` docstring | ✅ 更新 | **必要**：原文写"不触发任何键鼠交互（`src/action`）"，与 Phase 7.3 的编排职责不符；现精确表述为"不调用 `ActionDispatcher`、不产生输入；可向 Action 侧 resolver **推送快照数据**" |

### 1.4 严格架构边界（复核结论）

| 边界 | 结论 |
| --- | --- |
| Action 不 import `src.vision` / `src.runtime` / `src.capture` / `src.window` / `cv2` / `numpy` | ✅ **本阶段未触碰 `src/action/**`**；既有 AST 用例继续通过 |
| Action 只知道 `TargetResolver`，不知道 `EvidenceResult` / `Frame` / `VisionTargetAdapter` | ✅ 桥把数据转换成 `core.vision_target` 契约后才推送；Action 侧只看到 `TargetSnapshot` |
| Vision 不调用 Action | ✅ `src/vision/pipeline/target_adapter.py` 不 import action（Phase 7.2 的 AST 用例断言，本阶段未改） |
| Runtime 作为唯一编排层 | ✅ 只有 `src/runtime/vision_bridge.py` 同时依赖 `vision.pipeline` 与 action 侧 resolver |

---

## 2. Runtime → Vision → Action 数据流

### 2.1 每轮顺序（`StateMonitor.run()` 内）

```text
1. cycle_started = scheduler.begin()            节拍开始
2. frame = source.get_frame()                   采集（失败 → None）
3. step(frame) → _classify → processor.detect → [EvidenceResult]（保存到 last_evidences）
                → classifier.classify → StateResult → 稳定化 → 事件
4. cycles += 1                                  轮次计数（frame_id 的来源）
5. schedule = scheduler.finish(...)              节拍结束（含 cycle_id）
6. 诊断 / 恢复 / 指标
7. on_cycle(schedule)                           既有回调（Task 侧计时用）
8. _publish_vision_snapshot(frame, schedule, timestamp)   ← Phase 7.3 新增
      bridge.update(frame=frame, evidences=last_evidences,
                    frame_id=self._cycles, cycle_id=schedule.cycle_id, timestamp=frame.timestamp)
```

### 2.2 端到端实测（dry-run，零真实输入）

```text
=== 1) 运行 4 轮（3 帧 + 1 次 capture 失败） ===
  cycles: 4 | skipped: 1
  bridge: {updates: 4, published: 3, rejected_stale: 0, push_errors: 0,
           no_frame: 1, adapter_errors: 0, empty_snapshots: 0, candidates: 3,
           last_frame_id: 4, last_cycle_id: 4, last_reason: 'no_frame'}
  resolver: VisionAwareTargetResolver(ttl_s=1.0, min_confidence=0.6, updates=3, stale=0, snapshot=1)
  snapshot: TargetSnapshot(count=1, frame=3, names=['battle_auto_button'])

=== 2) Action：正常目标 ===
  status: skipped | ResolvedTarget(battle_auto_button, 0.5000, 0.5000, source=vision, calibrated=True)
  backend call: click((899, 953), button=left, count=1)     ← dry-run，只记录

=== 3) TTL 过期后（1.0s） ===
  status: rejected | target_expired
```

坐标核对：客户区 `Rect(47, 473, 1704, 960)` × `(0.5, 0.5)` = **`(899, 953)`** ✓

---

## 3. frame_id 来源（解决 Q-7.2-2）

| 项 | 取值 | 依据 |
| --- | --- | --- |
| `frame_id` | **`StateMonitor.cycles`**（本轮的轮次序号） | 用户要求"优先 StateMonitor cycle id"；它只在**处理过一帧**（或跳过轮）后递增，天然单调 |
| `cycle_id` | `CycleSchedule.cycle_id`（节拍轮号） | 既有节拍序号，用于审计与追踪（两者可能因暂停轮而不同） |
| 谁生成 | **Runtime**（桥只透传，**不生成**） | 用户要求"不要由 adapter 随机生成，必须来自 runtime cycle" |

追踪链路：

```text
cycle_id（节拍序号，含暂停轮）──┐
                                ├──▶ bridge.update(frame_id=monitor.cycles, cycle_id=schedule.cycle_id)
frame_id（处理过的帧序号，单调）──┘
                                        │
                                        ▼
                            TargetSnapshot.frame_id / VisionTargetCandidate.frame_id
                                        │
                                        ▼
                    resolver.update_snapshot（更旧的 frame_id → 拒收，stale_snapshots += 1）
                                        │
                                        ▼
                        ActionOutcome.detail.resolution_reason（解析 / 拒绝原因）
```

注意：capture 失败的轮次**不推送**（见第 4 节），因此 `frame_id` 可能出现"空洞"，
但始终**单调递增** ⇒ 不会触发乱序保护，也不会让 Action 用陈旧坐标。

---

## 4. 失败策略（fail-safe）

| 情况 | 桥的行为 | 依据 / 后果 |
| --- | --- | --- |
| **1. capture 失败**（`frame is None`） | **不更新**（不 push、也不 push 空快照），`no_frame += 1` | 旧快照保持原样，超过 `vision_target_ttl_s` 后 Action 侧返回 `target_expired` —— **不会"假装还有目标"** |
| **2. Vision 无结果**（`evidences` 为 `[]`/`None`，或全部未检出） | push **空快照**（`count == 0`），`empty_snapshots += 1` | 旧目标立即不可解析（`target_not_found`）—— **禁止保留上一帧目标** |
| **3. Adapter 异常** | `logger.error` + push **空快照**，`adapter_errors += 1` | 主循环不受影响；同样不保留上一帧目标 |
| **4. Resolver 拒收旧帧** | 保持 Phase 7.2 的 stale 保护（不覆盖），`rejected_stale += 1` | 保留较新的快照 |
| **5. Resolver 抛异常** | `logger.error` + `push_errors += 1`（返回 `None`） | Action 侧异常同样被隔离，不影响监控循环 |
| **6. 桥未接线**（`vision_bridge=None`） | 完全空操作 | 行为与 Phase 5.4 / 7.2 一致（既有测试全绿） |
| **7. 桥本身抛异常** | `_publish_vision_snapshot()` 捕获 + 记录 | 监控循环继续（有用例专门验证：跑满 `max_cycles` 且 `errors == 0`） |

设计要点：**"没有新证据"与"采集失败"被刻意区分**——前者说明"画面里确实没有目标"（推空快照、
立即失效），后者说明"这一轮我们什么都没看到"（不更新、让 TTL 决定何时失效）。

---

## 5. 安全边界

* 桥**只推送数据**：不调用 `ActionDispatcher`、不触碰 `SafetyGuard`、不产生任何输入；
* 真正的输入仍由 Action 侧决定：5 道门（E-STOP → 窗口 → 前台 → 目标已解析/已标定 → 限流）
  + `InputBackend`（默认 `DryRunBackend`，`action.enabled=false` 仍为默认值，本阶段未改配置）；;
* 视觉来源的"可信"仍受 Phase 7.1 的策略约束：`source="vision"` + 通过 TTL / 置信度门 +
  `trust_vision_targets`（默认 `True`）；
* 桥不读文件、不落盘、不记录截图（`VisionBridgeStats` 只有计数与 frame_id / cycle_id）；
* 未接线 ⇒ 零行为变化（既有 1000+ 用例继续通过）。

---

## 6. 测试结果

```text
$ python -m pytest -q
1218 passed, 1 skipped in 63.00s

$ python tools/check_dataset.py
Dataset: PASS   # 1602 张图；invalid / unresolved / duplicate = 0

$ git status
clean
```

| 项 | 数值 |
| --- | --- |
| 全套基线（Phase 7.2 后） | 1197 passed / 1 skipped |
| 本阶段新增 | **+21**（`tests/test_runtime_vision_bridge.py`） |
| 现基线 | **1218 passed / 1 skipped** |

### 6.1 覆盖矩阵（用户要求的 6 项）

| # | 要求 | 用例 |
| --- | --- | --- |
| 1 | 正常循环（frame_id=1 → snapshot.frame_id=1，resolver 生效） | `test_single_cycle_publishes_snapshot_with_frame_id`、`test_bridge_requires_resolver_with_update_snapshot` |
| 2 | 连续循环（1/2/3 → resolver 为 3） | `test_consecutive_cycles_advance_frame_id` |
| 3 | 空视觉结果（`Evidence=[]` → 空快照、不崩溃） | `test_empty_evidences_publish_empty_snapshot`、`test_none_evidences_treated_as_empty`、`test_undetected_evidences_produce_empty_snapshot` |
| 4 | capture 失败（不更新；旧快照随后 `target_expired`） | `test_capture_failure_does_not_update_snapshot`、`test_capture_failure_then_recovery_publishes_again` |
| 5 | 旧 frame（10 → 9：保留 10，`stale_snapshots += 1`） | `test_stale_frame_is_rejected_and_counted` |
| 6 | 端到端 dry-run（FakeFrame → … → TaskEngine → Dispatcher → DryRunBackend） | `test_end_to_end_runtime_to_action_dry_run`、`test_end_to_end_without_snapshot_is_rejected` |

另含集成与隔离用例：`StateMonitor` 自动推送（`test_monitor_publishes_snapshot_every_cycle`）、
未接线行为不变（`test_monitor_without_bridge_behaviour_unchanged`）、
同一次检测（`test_monitor_exposes_last_evidences_for_same_detection`）、
capture 失败轮不推送（`test_monitor_capture_failure_skips_publish`）、
桥异常不影响主循环（`test_monitor_bridge_exception_does_not_break_loop`）、
adapter 异常（`test_adapter_failure_publishes_empty_snapshot`）、resolver 异常（`test_resolver_failure_is_isolated`）、
统计与 reset（`test_bridge_stats_serialization`、`test_bridge_reset_clears_stats_only`）、
字段视图读取帧（`test_bridge_uses_adapter_frame_helpers`）。

### 6.2 "零真实输入"的证据

* 全部用例使用 `DryRunBackend`（`dispatcher.is_dry_run is True` 有断言）；
* `test_end_to_end_runtime_to_action_dry_run` 断言 `backend.count == 1` 且**只有 dry-run 记录**，
  坐标等于"客户区 × 比例"的闭式解；
* 未修改 `configs/config.yaml`（`enabled: false` / `backend: "dry_run"` / `resolver: "registry"` 保持默认）。

---

## 7. 未解决问题

| # | 问题 | 现状 / 建议 |
| --- | --- | --- |
| Q-7.3-1 | 是否把 `DetectionResult`（模板检测）也接进桥？ | 当前桥只消费 `EvidenceResult`（`last_evidences`）。`DetectionResult` 已可用作候选来源（Phase 7.2 支持），但需要另一次检测调用 ⇒ 建议 Phase 7.4 决定是否复用 detector 结果 |
| Q-7.3-2 | 暂停（`pause()`）期间是否应清空快照？ | 当前：暂停轮不处理帧 ⇒ 不推送 ⇒ 快照自然超时失效（fail-safe）✓ 如需"立刻失效"可显式 `resolver.clear()`（未实现策略） |
| Q-7.3-3 | `frame_id` 空洞（capture 失败轮）是否影响审计？ | 单调即可（乱序保护只需单调）；如需逐帧对齐可改用 `cycle_id` 作为 `frame_id`（会因暂停轮而不连续） |
| Q-7.3-4 | Task 侧（`TaskEngine`）的接线 | 本阶段只接"视觉 → 动作解析"；`StateMonitor.on_stable/on_cycle → TaskEngine` 的自动接线仍未做（Phase 6.3 / 7.4） |
| Q-7.3-5 | 快照节流 / 去重（同一帧重复推送） | 当前每轮一次（`run()` 内），无重复；若上层循环重复调用 `bridge.update` 需自行保证 frame_id 单调 |
| K-7.3-1 | `monitor.py` 1020 行 / `vision_bridge.py` 288 行 | monitor 已超 300 行规范（与 K-1 / K-3 / K-7.1-1 / K-7.2-1 同类） |

---

## 附录 A：复核命令

```powershell
# 全套 + 数据集
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\check_dataset.py

# Phase 7.3 定点测试
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_vision_bridge.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_vision_bridge.py -q -k "monitor or stale or capture or empty"

# 未接线时行为不变（既有 runtime 用例）
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_loop.py tests/test_phase53_runtime_hardening.py tests/test_phase54_runtime_lifecycle.py -q

# Action 层依赖边界（本阶段未触碰）
.\.venv\Scripts\python.exe -m pytest tests/test_vision_action_resolver.py -q -k "imports or allowed"
```

## 附录 B：公开 API 增量

```text
src.runtime.vision_bridge（新增模块）
  VisionSnapshotBridge(resolver, *, adapter=None, clock=monotonic, now=None)
    .update(*, frame, evidences, frame_id, cycle_id=None, timestamp=None, frame_size=None)
    .reset() / .stats / .resolver / .adapter / .describe()
  VisionBridgeStats
    updates / published / rejected_stale / push_errors / no_frame / adapter_errors
    / empty_snapshots / candidates / last_frame_id / last_cycle_id / last_candidates
    / last_reason + to_dict() / describe() / reset()

StateMonitor（Phase 7.3 增量）
  __init__(..., vision_bridge: Any | None = None)
  .last_evidences -> tuple[EvidenceResult, ...]     最近一轮证据（同一次检测的产物）
  .vision_bridge -> Any | None                      当前接线
```

## 附录 C：术语表

| 术语 | 含义 |
| --- | --- |
| VisionSnapshotBridge | Runtime 的视觉快照桥：帧 + 证据 → `TargetSnapshot` → 单向 push 给 resolver |
| frame_id | 处理过的帧序号（= `StateMonitor.cycles`）；单调递增，用于乱序保护与追踪 |
| cycle_id | 节拍轮号（`CycleSchedule.cycle_id`，含暂停轮）；审计用 |
| no_frame | capture 失败轮（本轮不更新快照） |
| empty_snapshots | 推送了空快照的轮数（视觉无结果 / adapter 异常） |
| 单向 push | Action 不主动读取 Vision；由 Runtime 把快照推给 resolver |
