# PHASE 5 Report — Runtime Evidence Pipeline Integration

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 5 Runtime Evidence Pipeline Integration |
| 目标 | 把 Runtime 主链路迁移到证据流水线：`Frame → Evidence → StateResult → StateStabilizer → StateMonitor` |
| 性质 | **运行链路接入**（不新增视觉能力、不新增 detector、不训练模型、不实现点击） |
| 前置 | Phase 4.1–4.7（证据层、词汇/契约、证据驱动分类器、transition/yard 迁移、置信度校准、架构收敛） |
| 完成状态 | 已实现、已测试（818 passed）、Demo 与 dataset 校验通过；**已提交**（信息见第 8 节） |

**总体约束核对**

| 约束 | 结果 |
| --- | --- |
| `StateMonitor` 公共接口兼容 | ✓ 位置参数与关键字参数都保持可用（`classifier` 参数现同时接受 processor）；新增 `processor` / `errors` 属性为**加法** |
| `StateEvent` 结构兼容 | ✓ 未改动（`previous_state` / `current_state` / `timestamp` / `confidence`） |
| `StateHistoryStore` 兼容 | ✓ 未改动，历史文件结构不变 |
| 已有测试全部通过 | ✓ 818 passed（含全部既有用例，无回归） |
| 禁止新增状态 / detector / 改 dataset / 改 evidence scoring / 改 confidence calibration / 改 Action Layer | ✓ 全部未触碰 |
| 禁止实现自动点击 | ✓ 未实现（无输入控制代码） |

---

## 1. 迁移前后架构

### 1.1 迁移前（Phase 4.7 结束时）

```text
FrameSource.get_frame()
      ↓
StateClassifier.classify(frame)        # 帧路径：自 4.6.5 起只判 transition
      ↓
StateResult
      ↓
StateStabilizer.update ──→ StateEvent ──→ StateHistoryStore
```

问题：`StateMonitor` 拿到的是**帧路径**结论——`yard` / `exploration` / `battle` /
`battle_result` / `battle_reward` 全部退化为 `unknown`（它们的权威来源在证据路径），
证据层已经具备完整能力却没有进入运行时。

### 1.2 迁移后（Phase 5）

```text
FrameSource.get_frame()
      ↓
EvidenceFrameProcessor.process(frame)   # Frame → [EvidenceResult] → StateResult
      ↓                                 （复用证据层与状态层，不重复实现任何判定）
StateResult（source=evidence，带 reason / evidence）
      ↓
StateStabilizer.update(result)          # 时间滤波唯一归属：连续窗口 + 置信度门槛
      ↓
StableStateResult → StateEvent → StateHistoryStore
```

回退路径（兼容）：显式注入 `classifier` 时保持 Phase 3.x 的帧路径行为。

实测（`examples/runtime_pipeline_demo.py`，6 张真实样本 × 每张连续 3 帧）：

```text
[19:58:43] frame=009 state=battle         confidence=0.9286 source=evidence stable=battle           EVENT: exploration -> battle
[19:58:44] frame=012 state=battle_result  confidence=0.8500 source=evidence stable=battle_result    EVENT: battle -> battle_result
[19:58:45] frame=015 state=battle_reward  confidence=0.8500 source=evidence stable=battle_reward    EVENT: battle_result -> battle_reward
[19:58:45] frame=018 state=transition     confidence=1.0000 source=evidence stable=transition       EVENT: battle_reward -> transition

Summary: cycles=18 classified=18 skipped=0 errors=0 initial=yard final=transition events=5
```

---

## 2. StateMonitor 变化

| 位置 | 迁移前 | 迁移后 |
| --- | --- | --- |
| 判定入口 | `self._classifier.classify(frame)`（硬编码帧路径） | `self._classify(frame)`：**processor 优先**，否则回退到 classifier |
| 构造参数 | `StateMonitor(classifier, source, config, *, stabilizer, history, clock, sleep, now, on_*)` | `StateMonitor(classifier_or_processor, source, config, *, processor=None, stabilizer, history, clock, sleep, now, on_*)` |
| 第一位置参数 | 仅 `StateClassifier` | `StateClassifier` **或** `EvidenceFrameProcessor`（多态；显式 `processor=` 优先级最高） |
| `from_config` | `classifier` 省略时构造 `StateClassifier.from_config(config)`（帧路径） | `processor` / `classifier` 都省略时构造**完整证据流水线**（含转场判定器；数据缺失自动降级）；显式传 `classifier` 可回退帧路径 |
| 异常处理 | 不捕获（异常向上抛） | 捕获 `OnmyojiVisionError` → 该帧降级为 `unknown`（`reason`/`detail` 含原因）、`errors += 1`、记录 error 日志；**循环继续** |
| 新属性 | — | `processor`（流水线或 `None`）、`errors`（降级次数） |
| `classifier` 属性 | 返回注入的分类器 | processor 模式下返回 `processor.classifier`（因此**仍是** `StateClassifier` 实例，接口兼容） |
| `step()` | 直接 classify | 走 `_classify`，其余逻辑（稳定化 → 变化检测 → 事件 → 落盘）**逐行未变** |
| `run()` | 固定节拍循环 | 未改（仅重置计数时同时重置 `errors`） |
| `MonitorReport` / `StateEvent` / `StateHistoryStore` | — | **未改** |

**职责边界（本阶段明确）**：`StateMonitor` 只负责 **Runtime 调度 / 状态稳定 / 事件生成**；
**不**做 detect、aggregate、OCR、模板匹配（这些都在证据层与状态层，由 processor 组合）。

---

## 3. Processor 接入方式

| 项 | 内容 |
| --- | --- |
| 复用既有实现 | `src/vision/pipeline/frame_processor.py`（Phase 4.7 新增）；**未**重新实现 `detect_all` / `aggregate` / `classify` |
| 调用链 | `monitor._classify(frame)` → `processor.process(frame)` → `registry.detect_all(frame)` + `classifier.classify(evidence)` |
| 装配（推荐） | `StateMonitor.from_config(config, source=...)` —— 默认装配证据流水线（含转场证据） |
| 装配（显式） | `EvidenceFrameProcessor.from_config(config, transition_classifier=probe)` 后 `StateMonitor(processor, source, config)` |
| 转场判定器 | `build_transition_probe(config)`（`src/vision/pipeline/frame_processor.py` 内的辅助函数）：按 `config.state.transition` 构造 `TransitionClassifier`；数据不可用时返回 `None`（`transition_banner` 输出未检出 + `metadata.missing_dataset`，**不抛异常**） |
| 依赖方向 | `runtime → vision.pipeline → {vision.evidence, vision.state}` ✓ 反向无依赖（Phase 4.5 冻结的方向保持不变） |
| 失败语义 | detector / OCR / 模板异常（`OnmyojiVisionError` 家族）→ 由 `StateMonitor._classify` 捕获 → unknown + 原因（见第 5.3 节测试） |

---

## 4. 性能测试

**完整测量见 `docs/PHASE5_RUNTIME_PERFORMANCE_REPORT.md`**；摘要：

| 阶段 | mean | p95 | 占单帧 |
| --- | --- | --- | --- |
| capture（`Frame.create`，不含解码） | 0.01 ms | 0.02 ms | ~0% |
| **detect（10 个 detector 的模板匹配）** | **356.45 ms** | **513.64 ms** | **≈99.98%** |
| aggregate（规则 + 可靠度加权 + 校准） | 0.08 ms | 0.10 ms | ≈0.02% |
| classify（含 aggregate） | 0.08 ms | 0.13 ms | ≈0.02% |
| process（detect + classify） | 350.61 ms | 515.31 ms | — |

- **500ms interval 当前不可行**：p95 ≈ 514 ms 已越过周期（mean 356 ms 占预算 71%）；
- **瓶颈位置**：`bottom_menu_exploration` 32.7% + `bottom_menu_yard` 26.8% + `chapter_panel` 15.1% = **74.6%**；
  耗时 ≈ `ROI 面积 × 模板数 × 尺度数`（`bottom_menu` 的 ROI 是**全宽**底部 1706×192，且各带 8 个模板 × 5 档尺度）；
- **按阶段要求不做优化**；可选优化方向（缩小 ROI / 降模板数与尺度 / 证据分级 / 并行 / 模板灰度缓存）记录在性能报告第 4 节，供后续独立阶段决策。

---

## 5. 兼容性说明

### 5.1 接口兼容

| 调用方式 | 结果 |
| --- | --- |
| `StateMonitor(classifier, source, config, *, stabilizer=..., ...)` | ✓ 仍可用（帧路径） |
| `StateMonitor(processor, source, config)` | ✓ 新支持（证据路径） |
| `StateMonitor(fake_classifier, source, config)`（既有测试替身） | ✓ 仍可用（`tests/monitor_fakes.build_monitor` 全通过） |
| `StateMonitor.from_config(config, source=...)` | ✓ 可用，**默认改为证据流水线**（行为变化，见 5.2） |
| `StateMonitor.from_config(config, source=..., classifier=...)` | ✓ 显式帧路径（Phase 3.x 行为） |
| `monitor.classifier` | ✓ 仍是 `StateClassifier` 实例（processor 模式下返回其内部分类器） |
| `monitor.step(frame)` / `monitor.run(...)` / `report.events` / `monitor.history` | ✓ 未变 |

### 5.2 行为变化（必须披露）

| 变化 | 说明 |
| --- | --- |
| `from_config` 的默认判定路径 | 之前是帧路径（只能判 `transition`），现在是**证据路径**（可判全部 7 个状态）——这正是本阶段目标；需要旧行为时显式传 `classifier=` |
| 未接线/数据缺失时 | 证据路径下模板缺失只会让对应证据 miss（不再报错）；转场数据缺失时 `transition_banner` 输出 `missing_dataset` 提示 |
| 判定失败时 | 新增降级语义：unknown + 原因（之前异常会直接抛出并中断循环） |
| `examples/state_monitor_demo.py` | **未改动**：它显式注入 `StateClassifier.from_config(config)` → 仍走帧路径（Demo 行为不变） |
| `examples/runtime_pipeline_demo.py` | 新增（证据路径的固定节拍 Demo） |

### 5.3 异常与失败处理（任务三）

| 失败类型 | 行为 | 覆盖测试 |
| --- | --- | --- |
| detector 抛 `EvidenceError` | 该帧 unknown（`source=fallback`、`reason="evidence pipeline failure: …"`），`monitor.errors += 1`，循环继续 | `test_failing_detector_does_not_crash[EvidenceError]` |
| OCR 不可用（`OCRProviderUnavailableError`） | 同上 | `test_failing_detector_does_not_crash[OCRProviderUnavailableError]` |
| 模板缺失 | 证据 miss（不抛异常）；`metadata` 记录 `insufficient_templates` / 目录告警 | 既有 `tests/test_evidence_detector.py` |
| 转场数据缺失 | `build_transition_probe` 返回 `None` → `transition_banner` 输出 miss + `missing_dataset=True` | 既有 `tests/test_transition_evidence.py` |
| 失败帧之后的帧 | 照常处理（失败不污染循环状态） | `test_failure_in_one_frame_does_not_break_next_frame` |

---

## 6. 测试结果

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# → 818 passed, 1 skipped, 0 failed

# 阶段基线（Phase 4.7 完成时）
# → 801 passed, 1 skipped

# 新增测试文件
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_pipeline.py -q
# → 17 passed

# Demo（离线固定节拍，6 样本 × 每张连续 3 帧 = 18 轮）
.\.venv\Scripts\python.exe examples/runtime_pipeline_demo.py
# → 18 帧逐行输出；Summary: cycles=18 classified=18 skipped=0 errors=0
#   initial=yard final=transition events=5（每个状态切换恰好 1 个事件）

# 既有监控 Demo（显式帧路径）仍可用
.\.venv\Scripts\python.exe examples/state_monitor_demo.py --max-cycles 3 --interval-ms 10
# → 正常运行（行为不变）

# dataset 校验
.\.venv\Scripts\python.exe tools/check_dataset.py
# → exit=0（Invalid 0 / Unresolved 0 / Duplicate 0）
```

| 指标 | Phase 4.7 | 现在 | 差异 |
| --- | --- | --- | --- |
| passed | 801 | **818** | **+17**（全部来自新测试文件；既有用例数量未变） |
| failed | 0 | **0** | 无回归 |
| skipped | 1（Tesseract） | 1 | 无关 |

### 6.1 新增用例对照（任务七的 8 项要求）

| 要求 | 覆盖用例 |
| --- | --- |
| 1. FrameProcessor 输出 `StateResult` | `test_frame_processor_returns_state_result`、`test_frame_processor_miss_returns_unknown` |
| 2. StateMonitor 使用 processor | `test_state_monitor_uses_processor`、`test_state_monitor_exposes_inner_classifier`、`test_processor_keyword_takes_priority` |
| 3. 旧 classifier 模式仍工作 | `test_legacy_classifier_mode_still_works`、`test_monitor_fakes_build_monitor_still_works` |
| 4. StateStabilizer 不受影响 | `test_stabilizer_requires_consecutive_frames`、`test_stabilizer_threshold_still_applied` |
| 5. `battle_result` 可以稳定 | `test_real_samples_reach_stable_state[battle_result]` |
| 6. `transition` 可以稳定 | `test_real_samples_reach_stable_state[transition]` |
| 7. 异常 detector 不导致 crash | `test_failing_detector_does_not_crash[...]`（2 个参数化）、`test_failure_in_one_frame_does_not_break_next_frame` |
| 8. 事件只由 stable state change 触发 | `test_events_only_on_stable_change`、`test_jitter_does_not_trigger_events`、`test_real_sequence_produces_events_on_state_change` |

### 6.2 任务完成对照

| 任务 | 状态 | 交付物 |
| --- | --- | --- |
| 一、迁移 StateMonitor 输入链路 | ✓ | `monitor._classify`（processor 优先）+ 职责边界（runtime 不做 detect/aggregate/OCR/模板） |
| 二、构造方式扩展 | ✓ | 第一位置参数多态 + `processor=` 关键字（优先）+ `from_config(processor=?, classifier=?)` |
| 三、Runtime Processor 接入（含异常处理） | ✓ | 复用 `frame_processor.py`；`OnmyojiVisionError` → unknown + 原因；`build_transition_probe` 数据缺失降级 |
| 四、StateStabilizer 接入验证 | ✓ | 连续 3 帧、门槛 0.85 语义不变；`battle_result` / `battle_reward` 实测 confidence=0.8500 → 稳定 |
| 五、实时循环 Demo | ✓ | `examples/runtime_pipeline_demo.py`（模拟 500ms，`--real-time` 可真实等待；输出 timestamp/raw/confidence/source/stable/event） |
| 六、性能测量 | ✓ | `docs/PHASE5_RUNTIME_PERFORMANCE_REPORT.md`（含分阶段与分 detector 瓶颈；未做优化） |
| 七、测试 | ✓ | `tests/test_runtime_pipeline.py`（17 个用例，覆盖 8 项要求） |

---

## 7. 剩余风险

| 编号 | 风险 | 影响 | 缓解 / 归属 |
| --- | --- | --- | --- |
| R1 | **性能超预算**：`detect` p95 ≈ 514 ms > 500 ms 周期 | 实际节拍被拉长到 ~0.36–0.55 s（不丢帧但延迟不稳定） | 本阶段按要求**不优化**；优化方向已记录（性能报告第 4 节），建议独立阶段处理并在此之前先缩小 `bottom_menu` ROI |
| R2 | 真实采集未纳入测量 | `capture`（dxcam/mss）+ 前台门控的真实成本未知 | 需在有游戏窗口的环境实测（Phase 5 后续 / Phase 6） |
| R3 | OCR 若启用会进一步增加耗时（`chapter_panel` / `battle_auto_button` / `victory_icon` / `transition_banner` 带 `text_targets`） | 更易超周期 | 性能报告已记录；建议 OCR 移出热路径 |
| R4 | `from_config` 默认路径变化（帧 → 证据） | 仅传 `Frame` 的既有调用方现在得到更丰富的状态（**意图如此**），但依赖旧行为者需显式传 `classifier=` | 已在 5.2 节披露；`examples/state_monitor_demo.py` 保持显式注入（行为不变） |
| R5 | 转场判定器依赖注入；遗漏则 `transition` 恒为未检出 | 运行时会丢掉 transition（有 warning + `missing_dataset` 标记，不会静默错判） | `from_config` 已默认注入 `build_transition_probe(config)` |
| R6 | `errors`（降级次数）只在 `MonitorReport` 之外暴露 | 运行摘要里看不到失败次数 | 已在 `MonitorReport.describe()` 之外提供 `monitor.errors`；如需进报告需改 `MonitorReport`（会动既有结构，未做） |
| R7 | 真实长时间运行未验证（无游戏窗口） | 长跑稳定性未知 | Demo 为离线模拟；真实运行验证属后续阶段 |
| R8 | `EvidenceHistory` 仍存在（deprecated，Phase 4.7 决策） | 读者可能误用 | 已在 Phase 4.7 双重标注；随本阶段完成，其删除可作为独立清理项 |
| R9 | `config.py` 仍单文件（拆分计划已有） | 维护成本 | Phase 4.7 报告第 6.3 节保留迁移计划 |

**本阶段明确未做的事**：未新增 detector / 状态 / dataset；未改 `src/vision/evidence/`（仅在 `pipeline` 内新增辅助函数）；
未改 detector 规则与 confidence 校准；未删除任何 deprecated 文件；未实现点击或输入控制。

---

## 8. Git commit 信息

本阶段改动已提交：

```text
feat(runtime): integrate evidence state pipeline

把 Runtime 主链路迁移到证据流水线（Phase 5）：FrameSource → EvidenceFrameProcessor
→ StateResult → StateStabilizer → StateMonitor；不新增视觉能力、不实现点击。

- src/runtime/state_monitor/monitor.py:
  - 判定入口改为 _classify(frame)：证据流水线优先，回退到状态分类器（渐进迁移）
  - __init__ 第一位置参数同时接受 StateClassifier 与 EvidenceFrameProcessor；
    新增关键字 processor=（优先级最高）
  - from_config 支持 processor= / classifier=；两者都省略时默认装配完整证据流水线
    （含转场判定器，数据缺失自动降级）
  - 判定失败（detector / OCR / 模板异常，OnmyojiVisionError）被捕获：该帧降级为
    unknown 并把原因写入 reason/detail，计数到新增属性 monitor.errors，循环不崩溃
  - 新增 processor 属性；classifier 属性在流水线模式下返回其内部分类器（接口兼容）
  - 未改：固定节拍、稳定化、事件生成、历史落盘、MonitorReport / StateEvent 结构
- src/runtime/state_monitor/__init__.py: 数据流文档（判定入口与职责边界）
- src/vision/pipeline/frame_processor.py: 新增 build_transition_probe(config)
  （构造转场判定器，数据不可用时返回 None，不抛异常）
- examples/runtime_pipeline_demo.py（新增）: 固定节拍演示（模拟 500ms / --real-time），
  离线图片序列 → 证据 → 状态 → 稳定 → 事件，逐帧输出 timestamp/raw/confidence/source/stable/event
- tests/test_runtime_pipeline.py（新增）: 17 个用例（处理器输出、monitor 接入、旧 classifier
  兼容、稳定化语义、真实样本 battle_result/transition 稳定、异常降级不崩溃、事件只由稳定状态变化触发）
- docs/PHASE5_RUNTIME_PIPELINE_REPORT.md、docs/PHASE5_RUNTIME_PERFORMANCE_REPORT.md（新增）

性能（本阶段测量，未优化）：detect 阶段 mean 356ms / p95 514ms，500ms 周期不可行；
瓶颈为 bottom_menu_exploration(32.7%) + bottom_menu_yard(26.8%) + chapter_panel(15.1%)。

验证：pytest 818 passed, 1 skipped；tools/check_dataset.py PASS（exit=0）；
examples/runtime_pipeline_demo.py 18 轮 → 5 事件（initial=yard → final=transition）。
Refs: docs/PHASE5_RUNTIME_PERFORMANCE_REPORT.md
```

### 提交前检查清单

```powershell
.\.venv\Scripts\python.exe -m pytest -q            # 期望 818 passed, 1 skipped
.\.venv\Scripts\python.exe tools/check_dataset.py   # 期望 exit=0
.\.venv\Scripts\python.exe examples/runtime_pipeline_demo.py   # 期望逐帧输出 + Summary
rtk git status                                      # 期望 clean（提交后）
```
