# Phase 11.8 — Yard Runtime Soak Validation 与视觉可观测性报告

> 本阶段是**运行期验证阶段**：不训练模型、不加模板、不降阈值、不改 Action 安全门、不启用真实输入。
> 目的：验证 Phase 11.7-C 已接入生产的 `Production36 + thumb Top-4` 在**连续运行**中是否满足
> 节拍、稳定性、预筛行为、快照失效与资源增长等运行期要求。
> commit：``feat(runtime): add yard runtime soak observability``

---

## 0. 一句话结论（先读）

| 关注点 | 结论（实测） |
| --- | --- |
| **500ms 节拍** | 真实窗口 60 轮：p50 **297ms**、p90 328ms、**p99 375ms**、**deadline miss 0** ✅ |
| **预筛行为** | 两种模式下 `candidates = 4`、`matchTemplate = 20 次/帧`、**fallback = 0** ✅ |
| **yard 单项耗时** | 真实窗口 p50 **40.5ms** / p95 49.5ms / max 57.8ms ✅ |
| **状态稳定性** | `flaps = 0`（含注入失败的 replay）✅ |
| **快照失效** | 注入的 capture 失败**正确触发失效**（`no_frame`，invalidations=2）✅ |
| **安全** | 全程 `action.enabled=false` / `backend=dry_run` / **`real_input=false`** ✅ |
| **资源** | 30s（60 轮）Python 分配 44.4MB → peak 54.2MB，无异常跃升；RSS 为 `not_available`（缺 psutil） |
| ⚠️ **未覆盖** | **真实 yard 画面**：窗口真实存在且在前台（`阴阳师-桌面版` 1704×960），但采集到的画面**不是庭院** ⇒ `detections = 0/60`，场景 A–F **未覆盖** |

**因此**：`real_runtime_soak = run_but_scene_not_yard` —— 节拍 / 预筛 / 快照 / 安全 / 资源
均在真实采集下得到验证；**yard 状态稳定性在真实环境尚未验证**（需要人工把游戏切到庭院）。

---

## 1. Runtime 可观测性审查（先审查后编码）

### 1.1 现状：每轮已记录什么

| 项 | 现状（改造前） |
| --- | --- |
| `CycleMetrics`（逐轮 JSONL） | `cycle_id / timestamp / state / confidence / stable_state / latency_ms / frame_ms / status / skip_reason / frame_source / capture_status / foreground_status / unknown_reason / runtime_state / recovery_count / last_recovery_reason` |
| `CycleTiming`（`timing=True` 时） | `frame_ms / detect_ms / state_ms / judge_ms / stabilize_ms / total_ms` |
| `DetectorProfile`（`profile=True` 时） | 每 detector 的 `elapsed_ms / detected / templates / roi / scales` |
| `VisionBridgeStats` | `updates / published / stale / push_errors / no_frame / **invalidated** / adapter_errors / empty_snapshots / candidates / last_reason` |
| `StateHistoryStore` + `monitor.events` | 稳定状态迁移事件 |
| 结构化 JSON | ✅ 已有（`logs/runtime_metrics.jsonl`，JSONL 逐轮一行） |

### 1.2 逐项回答（本次审查的 10 个问题）

| # | 问题 | 结论 |
| --- | --- | --- |
| 1 | 每 cycle 已记录哪些耗时 | `latency_ms`（端到端）+ `frame_ms`（取帧）；`timing=True` 时另有 detect/state/judge/stabilize |
| 2 | 能否单独观察 yard detector 耗时 | **原不能**（`_classify` 调 `processor.detect(frame)` 不采集 profile）⇒ **本阶段补齐** |
| 3 | 能否观察 prefilter 候选数 | 原仅存在于证据 `metadata`（且**只在命中时**）⇒ **本阶段补齐**并覆盖未命中轮次 |
| 4 | 能否观察 fallback 次数与原因 | 同上 ⇒ **本阶段补齐** |
| 5 | 能否观察实际 `matchTemplate` 数量 | **原不能** ⇒ **本阶段补齐**（候选数 × 档位数） |
| 6 | 能否观察 template cache hit/miss | API 存在（`EvidenceRegistry.cache_hits/cache_misses`），但**生产未启用**（`cache_size=0`）⇒ 报告 `cache disabled` |
| 7 | 能否观察 stable state 变化次数 | ✅ 已有（`events` / history）；本阶段在 summary 中加入 `transitions` 与 `flaps` |
| 8 | 能否区分真实 unknown / capture 失败 / processor 失败 / 失焦 / 快照失效 | ✅ 已有：`unknown_reason`（`no_evidence`/`unsupported_scene`/`detector_failure`/`not_foreground`/`skipped`）+ `capture_status` + `foreground_status` + `vision_bridge.invalidated/last_reason`（`no_frame`/`window_unavailable`/`runtime_stopped`） |
| 9 | metrics 是否有长期增长风险 | `RuntimeMetricsRecorder(max_records=0)` 默认**不限制**；`maxlen` 机制已存在 ⇒ 本阶段 soak 使用**有界** recorder（`MAX_SOAK_RECORDS=20000`） |
| 10 | 是否已有结构化 JSON 输出 | ✅ 有；本阶段新增**结构化 soak summary**（JSON）到 TEMP |

> 复用既有设施：**没有建立第二套 Telemetry**。

---

## 2. 新增的 runtime metrics（只补确实缺失的）

`CycleMetrics` 追加 **19 个可选字段**（全部标量 / 短字符串，默认 `None`）：

```
frame_id  vision_ms  yard_ms  yard_detected  yard_confidence  yard_template_id
yard_prefilter  yard_top_k  yard_candidates  yard_total_templates  yard_match_calls
yard_prefilter_ms  yard_fallback  yard_fallback_reason
snapshot_status  snapshot_invalidated  task_active  action_enabled  action_backend
```

**低开销与上限设计**：

* 开关：`runtime.state_monitor.yard_metrics`（**默认 `False`** = 零额外开销）；关闭时
  `to_dict()` **逐字节**与既有版本一致（字段仅在采集到时写入）；
* 只记录标量与短字符串：**不记录** Frame / screenshot / ndarray / 完整候选列表 / 完整模板列表 / ROI 图像；
* 单轮 JSON 有界（实测满字段 < 2KB，测试断言 < 2048 字节）；
* history buffer **必须有 max_size**：soak 用 `max_records=20000`（FIFO 淘汰最旧）。

**数据来源（全部是 Runtime 已持有的对象，无跨层 import、无私有属性访问）**：

```
yard 单项耗时          ← processor.registry.last_profile（profile=True 的既有产物）
预筛候选/回退/耗时      ← yard 证据的 metadata（vision 层在启用预筛时写入）
快照状态 / 累计失效     ← vision_bridge.stats
活动任务               ← task_bridge.active_task_id（本阶段未接线 ⇒ None）
安全确认               ← 装配期记录的 action 配置（action.enabled / backend）
```

**为此补的两处观测接口（不改判定）**：

1. `EvidenceFrameProcessor.detect(frame, *, profile=False)` —— 透传 `registry.detect_all(profile=…)`；
   **仅当需要画像时才传该关键字参数**，因此 `detect(frame)` 这一既有契约对鸭子类型实现仍然成立；
2. `TemplateEvidenceDetector._prefilter_diagnostics()` —— 把预筛诊断（计数/回退/耗时）统一注入
   **命中与未命中**两条路径的 `metadata`（默认未启用预筛时返回空 ⇒ `metadata` 逐字段不变）。

---

## 3. Soak runner

新增 `tools/run_runtime_soak.py`：

| 参数 | 说明 |
| --- | --- |
| `--mode real / replay` | 真实窗口 / 离线图片序列 |
| `--duration` | 最长时长（秒）；**不写死**（支持 30s / 5min / 15min / 30min） |
| `--max-cycles` | 最多轮数 |
| `--output` | summary JSON 路径（默认 TEMP） |
| `--summary-every` | 每 N 轮打印进度 |
| `--record-metrics` | 额外把逐轮 JSONL 写到 TEMP |
| `--replay-failures` / `--no-loop` | replay 的注入失败位置与循环控制 |

**安全设计（硬约束）**：

* 启动前校验 `action.enabled=false` 且 `backend=dry_run`，否则**拒绝运行**；
* `--mode real` 只做**只读探测**（`binding.find/is_minimized/is_foreground`），
  **不激活窗口、不切换前台、不发送任何键鼠输入**（AST 测试钉住：不 import `pyautogui`、
  不出现 `SetForegroundWindow` / `click` / `press` / `write` / `hotkey` 调用）；
* 窗口不可用时**明确返回** `real_runtime_soak = not_run` + 原因（`window_not_found` /
  `window_minimized` / `window_not_foreground`），**不伪造成功**，也不自动启动游戏；
* 输出（summary JSON / JSONL）只写系统 TEMP。

---

## 4. 离线 replay

`ReplayFrameSource`：按 `dataset/raw` 的真实原图构造可复现序列，并支持**注入 capture 失败**。

默认脚本（对应 A/C 场景）：

```text
yard × 8 → exploration × 3 → yard × 5 → unknown/other × 3 → yard × 5
注入 capture 失败：第 12、13 轮
```

**明确标注**：`mode = replay`、`offline_replay = true`，且 summary 内带
``"warning": "offline_replay 不代表真实运行期性能；真实 wall-clock 结论需真实窗口 soak"``。

**用途**（用户指定）：验证 metrics / 状态迁移 / buffer / summary / fallback / snapshot invalidation 的**正确性**；
真实 wall-clock 性能仍由 `--mode real` 决定。

---

## 5. 实际测试环境

| 项 | 值 |
| --- | --- |
| OS / Python | Windows / Python 3.12.10（venv） |
| OpenCV / NumPy | 4.14.0 / 2.5.3 |
| 真实窗口 | **存在且在前台**：`阴阳师-桌面版`，hwnd 328260，客户区 **1704×960**（`is_alive=True`，`is_minimized=False`，`is_foreground=True`） |
| 游戏画面状态 | **不是庭院**（检测结果全部为 unknown）⇒ 未覆盖 yard 场景 |
| psutil | **未安装** ⇒ RSS 为 `not_available`（改用 `tracemalloc`） |
| 生产配置 | yard：36 张显式模板 + `thumb` + `top_k=4`，`threshold=0.85`，`roi_scale=0.75`；`action.enabled=false` / `backend=dry_run` |

---

## 6. 实际运行时长

| 运行 | 模式 | 轮数 | 时长 | 说明 |
| --- | --- | --- | --- | --- |
| R1 | **real** | **60** | **30.0s** | 真实窗口 + 真实采集（画面非庭院） |
| R2 | replay | 24 | 12.0s | 含 2 处注入 capture 失败 |
| R3 | replay | 10 | 5.0s | 连续 yard（flap 检查） |
| R4 | replay | 19 | 9.5s | 默认脚本（迁移检查） |

> **未做** 5/15/30 分钟长跑（见 §18 未测试项）。

---

## 7. Cycle latency（端到端，`CycleScheduler` 语义）

| 运行 | p50 | mean | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| **real（60 轮）** | **297.0** | 295.3 | 328.0 | 343.0 | **375.0** | **375.0** |
| replay（24 轮） | 265.0 | 246.4 | 312.0 | 312.0 | 312.0 | 312.0 |

（单位 ms；`interval_ms = 500`）

* 真实窗口下 **p99 = 375ms < 500ms**，且 **max = 375ms**（未出现接近预算的尖峰）；
* 与离线基准（yard 单项 p50 45ms + 其它 detector ≈ 265ms）一致 ⇒ **端到端预算仍有余量**。

---

## 8. Deadline miss

| 运行 | misses | rate | 语义 |
| --- | --- | --- | --- |
| real | **0** | 0.0 | 沿用 `CycleScheduler` 的 overrun 语义（`elapsed > interval_ms`） |
| replay | **0** | 0.0 | 同上 |

* **未自创第二套 deadline 定义**：直接使用 `RuntimeMetricsRecorder` 的 `late_cycles`
  （由 `CycleSchedule.overrun_ms > 0` 驱动）与 `interval_ms` 作为预算；
* **未观察到持续积压或 drift**（60 轮 / 30s 内无任何 miss）。

---

## 9. Yard detector 运行期指标

| 运行 | detections | detector p50 | p95 | max |
| --- | --- | --- | --- | --- |
| **real** | **0 / 60** ⚠️ | **40.5ms** | 49.5ms | 57.8ms |
| replay | 11 / 24 | 25.0ms | 35.9ms | 39.0ms |

* 真实窗口下 yard detector 耗时 p50 40.5ms（与离线 45.1ms 同量级 ⇒ 真实采集未显著放大）；
* **`detections = 0/60`**：采集到的画面不是庭院（既无 yard 也无其它已知状态）⇒
  这是**场景覆盖问题**，不是 detector 故障（`unknown_reason` 为 `no_evidence`）。

---

## 10. Candidate count 与 matchTemplate 调用分布

| 运行 | 候选数分布 | `matchTemplate` 分布 |
| --- | --- | --- |
| real（60 轮） | `{4: 60}` | `{20: 60}` |
| replay（24 轮） | `{4: 24}` | `{20: 24}` |

* **候选数从未超过 `top_k=4`**（含未命中轮次）；
* `matchTemplate = 候选数 × 5 档 = 20`，**从未出现 180 次（全量回退）** ⇒ 与 `fallback=0` 自洽；
* 说明：Phase 11.8 补齐了**未命中轮次**的候选诊断（此前仅命中时可见），因此 60/60、24/24 轮都有候选记录。

---

## 11. Prefilter fallback

| 运行 | fallback_count | rate | reasons |
| --- | --- | --- | --- |
| real | **0** | 0.0 | `{}` |
| replay | **0** | 0.0 | `{}` |
| 注入故障（测试） | ≥1 | — | `RuntimeError: injected prefilter failure`（回退全量，候选数 = 36） |

* 正常运行时 **fallback 接近 0（实测为 0）**；
* 注入故障时**被统计且被记录**（`cycle_id` / `reason` / `state` / `candidate_count`），
  但**不输出原图**（遵守约束）；
* **"Top-4 内没有模板超过阈值"不算 fallback**：真实窗口 60 轮全部未命中，fallback 仍为 0 ✅

---

## 12. Template cache 状态

**`cache disabled`**：生产 detector 的 `EvidenceRegistry.cache_size` 仍为 **0**（Phase 5.1 默认），
即 11.6-A 的 `LruEvidenceCache` 在生产路径**未启用**。

* 本阶段**没有**为了观测擅自开启（遵守"不修改 cache 配置"）；
* 因此**无法**给出静止 yard / 动画 yard 的 hit/miss 对比 ⇒ 记为 `not_observed`；
* API 存在（`registry.cache_hits / cache_misses / cache_size`），如需启用属 Phase 11.9 决策。

---

## 13. State transitions 与 flaps

| 运行 | 稳定状态计数 | transitions | **flaps** |
| --- | --- | --- | --- |
| real | `{pending: 2, unknown: 58}` | 0 | **0** |
| replay | `{pending: 4, unknown: 5, yard: 15}` | 2 | **0** |

* 稳定化参数（未改动）：`enabled=True`, `min_frames=3`, `confidence_threshold=0.85`；
* **flaps = 0**：连续 yard 片段（10 轮）与含注入失败的混合序列**都没有**出现 `A→B→A` 抖动；
* 未观察到 `yard ↔ unknown` 抖动 —— 原因是稳定化只在置信度 ≥ 0.85 连续 3 帧后确认，
  低置信度（`unknown` 的 confidence=0）**不会推翻**已确认状态（这是**设计使然**，不是巧合）；
* 首次确认稳定状态只建立基线、不产生迁移事件（既有语义）。

---

## 14. Capture / Processor 失败

| 运行 | capture success | failures | 跳过原因 | 影响 |
| --- | --- | --- | --- | --- |
| real | 60 | 0 | — | 未自然出现失败 ⇒ **`not_observed`** |
| replay（注入 2 处） | 22 | **2** | `capture_failed_injected(position=12/13)` | 该轮 `status=skipped`、**不产生新 Observation**、不判定、不写 yard 指标 |

* **未为了制造错误主动破坏系统**：失败是 replay 注入的（可复现）；
* `processor_failure`（`unknown_reason=detector_failure`）在两种运行中均为 **0** ⇒ `not_observed`。

---

## 15. Snapshot invalidation

| 运行 | snapshot 状态分布 | invalidations |
| --- | --- | --- |
| real | `{processed: 59, (首轮尚未推送): 1}` | **0**（无 capture 失败，符合预期） |
| replay | `{no_frame: 2, processed: 21, (首轮): 1}` | **2** ✅ |

* 注入的 capture 失败**正确触发主动失效**（`reason=no_frame`）—— 不是等待 TTL 过期；
* 快照桥在 soak 中以**只读方式**接线（`build_vision_bridge`）：只 push 快照给一个**内存中**的
  `VisionAwareTargetResolver`，**不接** `ActionDispatcher`、**不接**输入后端 ⇒ 不影响安全语义；
* `expired`（TTL 过期）由 Action 侧 resolver 判定，运行期不可见 ⇒ 记为 `not_available`。

---

## 16. 内存趋势

| 运行 | Python alloc 起始 | peak | RSS |
| --- | --- | --- | --- |
| real（30s / 60 轮） | 44.36 MB | 54.17 MB | `not_available`（无 psutil） |
| replay（12s / 24 轮） | 52.70 MB | 66.18 MB | `not_available` |

* 用 `tracemalloc` 观测 Python 层分配：30s 内从 44.4MB 到 peak 54.2MB，**无异常跃升**；
* **逐轮记录有界**（`max_records=20000`，FIFO）⇒ `ObservationStore` / metrics buffer / state history
  均不会无界增长（history 由 `max_events` 限制、观察存储由既有上限限制）；
* ⚠️ **时间不足**：30s 不足以判断长期趋势；**RSS 未测**（psutil 未安装）⇒ 需要 30 分钟长跑复核（§18）。

---

## 17. Safety 确认

| 项 | 值 |
| --- | --- |
| `action.enabled` | **false** |
| `action.backend` | **dry_run** |
| `real_input` | **false** |
| 工具是否激活窗口 / 发送输入 | **否**（AST 测试钉住：不 import `pyautogui`、无 `SetForegroundWindow`/`click`/`press`/`write`/`hotkey` 调用） |
| 是否接触 `ActionDispatcher` / `InputBackend` | **否**（soak 只读；快照桥仅接内存 resolver） |

---

## 18. 未测试项目（`not_tested`）

| 项 | 原因 |
| --- | --- |
| **真实 yard 画面 soak**（场景 A/B/C/D/E/F） | 真实窗口存在但在**非庭院界面** ⇒ `detections=0/60`；需**人工**把游戏切到庭院后重跑（程序**不**代劳） |
| 场景 D（覆盖式弹窗）| 同上（未覆盖 yard 基线） |
| 场景 E（Alt-Tab 失焦）| 同上；`foreground_status` 恒为 `n/a`（真实窗口路径的前台门控由 `WindowFrameSource` 内部处理） |
| 30 分钟长跑 / RSS 趋势 | 时间不足 + `psutil` 未安装（RSS `not_available`） |
| template cache hit/miss | 生产 **cache disabled**（`cache_size=0`） |
| `task_active` 观测 | `TaskRuntimeBridge` 需要 `TaskEngine` + 任务清单才可构造，本阶段**未接线** ⇒ `None` |
| 结构化 replay 的 `expired`（TTL） | 由 Action 侧 resolver 判定，运行期不可见 ⇒ `not_available` |
| `processor_failure`（detector 异常降级） | 未自然出现，也未注入 ⇒ `not_observed` |

---

## 19. 未解决问题

| 编号 | 内容 |
| --- | --- |
| U-1 | **真实 yard 场景未验证**：需要人工把游戏切到庭院（含动画、yard↔探索切换、弹窗、失焦、回前台六个场景）后重跑 `--mode real` |
| U-2 | **RSS 未测**：安装 `psutil` 后可记录 `rss_start/end/peak`（当前为 `not_available`） |
| U-3 | **长跑不足**：30s 的窗口不足以判断内存长期趋势与 drift；建议 30 分钟 |
| U-4 | **template cache 未启用** ⇒ 无法观测 hit/miss 与"静止 vs 动画"差异 |
| U-5 | **`task_active` 未接线**（需要 `TaskRuntimeBridge`） |
| U-6 | `snapshot expired`（TTL）在运行期不可见，只能从 Action 侧观测 |
| U-7 | 真实窗口的 `foreground_status` 恒为 `n/a`（门控在 `WindowFrameSource` 内部），失焦场景无法从 metrics 直接读出 |

---

## 20. Phase 11.9 建议

1. **人工配合的真实 yard soak**：切到庭院后按场景 A→F 各跑 2–5 分钟（`--mode real --duration 300`），
   验证 `detections`、`flaps`、`transitions`、`snapshot` 与 `foreground` 的真实行为；
2. **长跑 30 分钟 + `psutil`**：记录 RSS 趋势，确认无持续增长；
3. **决定是否启用 template cache**（11.6-A）：启用后可观测静止/动画 yard 的 hit/miss 比；
4. **接 `TaskRuntimeBridge`**：补齐 `task_active` 观测（仍保持 `dry_run`）；
5. 把 soak summary 接入既有诊断出口（`logs/diagnostics/`），让日常运行也能留下同类证据；
6. 在**真实 yard 数据**上复核 `candidate_count` 分布与 `prefilter_ms`（当前真实窗口未命中 yard，
   prefilter 耗时只在 replay 与离线数据上验证过）。

---

## 21. Git commit

```text
feat(runtime): add yard runtime soak observability
```
