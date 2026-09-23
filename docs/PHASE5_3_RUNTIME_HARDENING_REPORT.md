# Phase 5.3 — Runtime Architecture Hardening 报告

> 本阶段只做 **Runtime 工程化收敛**：拆分职责边界、增加诊断层、细化 `unknown` 原因、
> 兼容地扩展运行指标。**不改视觉判定**（Evidence / detector / StateResult / 状态规则 / dataset 全部未动），
> **不新增 Action**，**不做性能优化**。

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 5.3（任务一～六） |
| 入口 | `src/runtime/state_monitor/`（拆分 + 诊断）、`tests/test_phase53_runtime_hardening.py` |
| 行数口径 | 下文行数统一为 `Measure-Object -Line`（含空行与文档） |
| 结论 | 拆分完成且**行为完全一致**（896 passed / 1 skipped，`check_dataset` PASS，live demo 正常） |

---

## 1. 拆分前结构

`src/runtime/state_monitor/window_source.py`（Phase 5.2 交付物）**约 870 行**，单个模块承担 5 件事：

```text
window_source.py
├── SkipReason / CaptureStatus / GuardVerdict   ← 数据模型（3 种语义）
├── ForegroundGuard                             ← 前台门控（判定"能不能采集"）
├── OnmyojiWindowSource                         ← 窗口绑定（查找 / 几何 / 失效重绑）
└── WindowFrameSource                           ← 采集编排（门控 → 后端 → 帧 → 失败处理）
```

| 问题 | 影响 |
| --- | --- |
| 一个文件 5 类职责 | 阅读 / 评审成本高；改动窗口绑定逻辑会与采集逻辑混在同一 diff 中 |
| 数据模型与行为混排 | `SkipReason`（门控与采集共用）夹在两类实现之间，导入关系不直观 |
| 依赖方向不显式 | "谁依赖谁"只能靠阅读代码判断，无法从模块结构读出 |

---

## 2. 拆分后结构

```text
src/runtime/state_monitor/
├── window_binding.py     123 行  OnmyojiWindowSource —— 窗口查找 / 绑定 / 几何刷新（exact/contains/regex）
├── foreground_guard.py   140 行  SkipReason + GuardVerdict + ForegroundGuard —— 前台门控（只检查、不抢占）
├── frame_source.py       433 行  CaptureStatus + WindowFrameSource —— 采集编排（dxcam 优先 / mss 回退 / 重试）
├── window_source.py       34 行  **兼容层**：重新导出上述 7 个符号（旧 import 路径继续可用）
├── runtime_diagnostics.py 约 400 行  RuntimeHealth / UnknownReason / RuntimeDiagnostics（Phase 5.3 新增）
├── scheduler.py          190 行  （未改动）
└── source.py / monitor.py / metrics.py / stabilizer.py / history.py / report.py / events.py / timing.py
```

依赖方向（单向、无环）：

```text
core ──▶ vision.window / vision.capture ──▶ window_binding ──▶ foreground_guard ──▶ frame_source
                                                                        │
                                              runtime_diagnostics ◀──────┘（SkipReason）
                                                       ▲
                                                  monitor（逐轮 observe）
```

| 模块 | 单一职责 | 明确不做 |
| --- | --- | --- |
| `window_binding` | 找到目标窗口 + 读回实时几何 | 不采集、不门控、不判定 |
| `foreground_guard` | 判断"这一轮能不能采集"并说明原因 | 不抢焦点、不切换窗口、不采集 |
| `frame_source` | 把窗口客户区采集成 `Frame`，失败转成"跳过" | 不判定、不做时间滤波 |
| `runtime_diagnostics` | 收集运行健康状态、细分 `unknown` 原因 | 不采集、不判定、不写文件 |

---

## 3. API 兼容

| 兼容项 | 状态 | 验证 |
| --- | --- | --- |
| 旧 import 路径 `window_source.{WindowFrameSource, CaptureStatus, DEFAULT_RETRY_INTERVAL_S, OnmyojiWindowSource, ForegroundGuard, GuardVerdict, SkipReason}` | ✅ 全部可用 | `test_legacy_window_source_import_path_still_works` |
| 兼容层与新模块导出**同一对象**（`is`） | ✅ | `test_split_symbols_are_identical_objects` |
| `StateMonitor` API | ✅ 未改签名；新增只读 property `health` / `diagnostics` / `last_evidence_hits` | `tests/test_state_monitor*.py` 全绿 |
| `WindowFrameSource` API | ✅ 构造参数、属性、`get_frame()` 语义未变 | `tests/test_runtime_loop.py` 全绿 |
| `Scheduler` 行为 | ✅ `scheduler.py` 未改动 | `test_scheduler_*` 全绿 |
| `Metrics` JSONL 格式 | ✅ 既有 10 个字段名与含义不变，诊断字段为**追加** | `test_metrics_schema_is_backward_compatible` |
| 旧 `CycleMetrics(...)` 构造方式（不传诊断字段） | ✅ 仍可用 | `test_cycle_metrics_legacy_construction_still_valid` |
| 旧的采集工厂替换路径 | ⚠️ patch 目标改为实现所在模块：`src.runtime.state_monitor.frame_source.create_capture_backend`（兼容层文档已注明） | `tests/test_phase53_runtime_hardening.py` 全部用例 |

---

## 4. 诊断层（任务二）与 `unknown` 细化（任务三）

### 4.1 区分"坏了"与"不知道"

```text
RuntimeHealth
├── capture_ok       最近一轮是否取到帧
├── foreground_ok    最近一轮前台门控是否通过
├── processor_ok     最近一轮判定是否正常完成（未因异常降级）
├── last_error       最近一次失败说明（判定异常 / 采集跳过消息）
├── last_state       最近一轮原始判定状态
├── cycle_latency    最近一轮端到端耗时（ms）
├── unknown_reason   unknown 的细分原因
└── cycles / frames / skipped / errors / skip_counts / backend
```

两个派生判断（把"两种没有结论"分开）：

| 判断 | 含义 | 处理方向 |
| --- | --- | --- |
| `health.is_runtime_failure` | 判定失败（`detector_failure`）或采集失败（`skipped`） | 查 detector / 采集后端 / 窗口 |
| `health.is_vision_unknown` | `no_evidence` / `unsupported_scene` / `not_foreground` | 数据侧或运行环境 |

### 4.2 `unknown` 原因分类（**不改 `StateResult` 契约**）

`StateResult` 一个字都没动：细分类别只作为 Runtime 侧 metadata 进入诊断与运行指标。

| `unknown_reason` | 判据 | 含义 |
| --- | --- | --- |
| `no_evidence` | 判定成功、状态 `unknown`、命中证据数 = 0 | 链路正常，画面里没有系统认识的元素 |
| `unsupported_scene` | 判定成功、状态 `unknown`、命中证据数 > 0 | 看到了已知元素，但没有任何规则满足（界面不在覆盖范围） |
| `detector_failure` | 判定异常降级（`judge_failed`，或 `source=FALLBACK`） | 视觉链路出问题 |
| `not_foreground` | 本轮因窗口不在前台被跳过 | 没有可用画面（运行环境） |
| `skipped` | 其它采集跳过（窗口不存在 / 最小化 / 后端不可用 / 抓帧失败 / 无帧） | 没有可用画面（采集问题） |

**如何拿到"命中证据数"而不重复检测**：`StateMonitor` 的证据路径由
`processor.process(frame)` 改为**等价的** `processor.detect(frame)` + `processor.classifier.classify(evidence)`
两行调用（`process` 内部本来就是这两步），因此同一次检测既完成判定、又得到命中计数；
耗时口径不变，`CycleTiming` 的 `detect_ms` / `state_ms` 由 monitor 自身按同样计时点填充。
帧路径（显式注入 `classifier`）不提供证据计数，判据回退为 `StateResult.evidence` 是否为空。

### 4.3 运行指标扩展（任务四）

`logs/runtime_metrics.jsonl` 每行现在包含（**旧字段不变 + 3 个追加字段**）：

```json
{"cycle_id": 1, "timestamp": "2026-09-23T21:10:15+08:00", "state": "unknown", "confidence": 0.0,
 "stable_state": "unknown", "latency_ms": 266.0, "frame_ms": 1.9, "status": "ok",
 "skip_reason": null, "frame_source": "dxcam",
 "capture_status": "ok", "foreground_status": "ok", "unknown_reason": "no_evidence"}
```

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `capture_status` | `ok` / `skipped` | 采集结果 |
| `foreground_status` | `ok` / `blocked` / `n/a` | 前台门控结果（未启用门控的来源为 `n/a`） |
| `unknown_reason` | 上表五类 / `null` | 状态明确时为 `null` |

仍然**不记录截图、不记录用户数据**；写入失败不影响主循环（既有行为）。

---

## 5. 测试

```text
python -m pytest -q                       → 896 passed, 1 skipped（跳过项：本机未安装 Tesseract）
python tools/check_dataset.py             → Dataset: PASS（1602 张图；invalid/unresolved/duplicate = 0）
python examples/live_runtime_demo.py --max-cycles 6 --no-metrics
                                          → 真实窗口 6 轮全部采集成功，0 late、0 errors
```

`tests/test_phase53_runtime_hardening.py`（32 个用例，任务五要求 6 项全覆盖）：

| # | 覆盖项 | 用例 |
| --- | --- | --- |
| 1 | 旧 import 仍工作 | `test_legacy_window_source_import_path_still_works`、`test_split_symbols_are_identical_objects`、`test_new_modules_import_paths_work` |
| 2 | 拆分前后行为一致 | `test_split_preserves_behaviour`（6 步场景快照：正常 → 抓帧失败 → 恢复 → 最小化 → 失焦 → 句柄失效，新旧路径结果序列完全一致）、`test_behaviour_includes_skip_reason_counter` |
| 3 | capture 失败不崩 | `test_backend_unavailable_does_not_crash_loop`、`test_capture_failure_recovers_within_loop`、`test_missing_window_does_not_crash_loop` |
| 4 | foreground skip 正确 | `test_foreground_skip_is_reported_in_diagnostics`、`test_foreground_pass_is_reported_in_diagnostics` |
| 5 | unknown reason 正确 | `test_unknown_reason_{no_evidence,unsupported_scene,detector_failure,not_foreground,skipped_for_other_capture_problems,none_when_state_is_known}`、`test_resolve_unknown_reason_{pure_function,for_skipped_cycles,judge_failure}`、`test_diagnostics_reset_clears_counters` |
| 6 | metrics schema 兼容 | `test_metrics_schema_is_backward_compatible`、`test_cycle_metrics_legacy_construction_still_valid`、`test_metrics_summary_unaffected_by_diagnostics`、`test_monitor_health_snapshot_is_json_serializable`、`test_source_without_diagnostics_capability_still_works` |

原有测试 **全部保留并通过**（Phase 5.2 的 32 个 runtime 用例、Phase 5.1 的性能/计时用例、
`StateMonitor` / `StateStabilizer` / 证据层 / 状态层用例均未修改语义）。

### 5.1 live demo 真实输出（Phase 5.3 新增诊断列）

```text
[21:10:15] #0001 raw=unknown      confidence=0.0000 stable=pending   cycle=  422.0ms unknown_due_to=no_evidence
[21:10:16] #0003 raw=unknown      confidence=0.0000 stable=unknown   cycle=  266.0ms unknown_due_to=no_evidence
Summary:
  cycles=6 fps=2.17 ok=6 skipped=0 errors=0 late=0
  health:  cycles=6 frames=6 skipped=0 errors=0 capture_ok=True foreground_ok=True processor_ok=True state=unknown latency=266.0ms unknown_due_to=no_evidence backend=dxcam
```

---

## 6. 风险与后续

| # | 风险 | 现状 / 缓解 |
| --- | --- | --- |
| R-1 | 兼容层 `window_source.py` 长期存在会造成双入口 | 本阶段按要求**不删除**旧模块；兼容层只做 re-export（34 行）并在文档中指明"实现位于三个新模块"，建议 Phase 5.4 起统一使用新路径 |
| R-2 | 判定路径改为 `detect + classify` 提升了耦合面 | 与 `process()` 内部实现完全一致（**无重复检测**），行为等价由 896 个用例守护；`EvidenceFrameProcessor.process(profile=True)` 的公共语义未改（外部仍可自行调用） |
| R-3 | `monitor.last_timing` 的 `detect_ms/state_ms` 来源改为自身计时 | 计时点与 `process(profile=True)` 相同（同一段代码），Phase 5.1 的 timing 用例全绿 |
| R-4 | `unknown` 分类依赖"命中证据数" | 帧路径下不可用 → 回退到 `StateResult.evidence` 判据（已单测覆盖两种入口）；分类逻辑本身是纯函数、可独立测试 |
| R-5 | `frame_source.py` 仍 433 行（> 项目 300 行建议） | 其中约 90 行为模块/类文档；若要继续拆分，建议把"窗口再绑定 + 查找冷却"抽成独立的窗口解析策略对象（Phase 5.4 候选） |
| R-6 | metrics 新增键可能影响严格 schema 的消费方 | 追加字段对 JSON 消费者向后兼容（旧键名与含义均未改）；本仓库内文档与测试已同步 |
| R-7 | 诊断字段的语义扩张风险 | `RuntimeHealth.to_dict()` 与 `CycleMetrics.to_dict()` 都显式列出字段，**不含截图 / 窗口标题 / 用户数据**，并有测试锁定 |

**本阶段明确未做**：未修改 Evidence / detector / 状态规则 / `StateResult` / dataset；
未新增 Action Layer；未做性能优化（真实环境 detect 仍约 240–290 ms，见
`docs/PHASE5_2_REAL_RUNTIME_REPORT.md`）。

---

## 7. 复现方式

```powershell
python -m pytest -q                                   # 896 passed, 1 skipped
python tools/check_dataset.py                         # Dataset: PASS
python examples/live_runtime_demo.py --max-cycles 20  # 真实窗口（含 unknown_due_to 诊断列 + health 快照）
python -c "import src.runtime.state_monitor.window_source as m; print(m.__all__)"   # 旧路径兼容性检查
```
