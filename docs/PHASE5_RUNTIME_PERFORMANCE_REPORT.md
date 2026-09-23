# PHASE 5 Report — Runtime Performance（证据流水线单帧延迟）

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 5 Runtime Evidence Pipeline Integration（任务六：性能测量） |
| 目的 | 首次测量**完整 evidence pipeline** 的单帧延迟，确认 **500ms interval 是否可行** |
| 结论 | **当前配置下不可行**：`detect` 阶段 p95 ≈ **514 ms** > 500 ms（mean 356 ms ≈ 预算的 71%）。**按阶段要求不做优化，仅记录瓶颈位置** |
| 环境 | 本机 Windows / Python 3.13（`.venv`）/ OpenCV 4.14；CPU 执行，无 GPU |
| 测量脚本 | 临时脚本（`time.perf_counter`，跑完即删；不在仓库保留） |

---

## 1. 测量方法

| 项 | 值 |
| --- | --- |
| 样本 | 6 张真实图（yard / exploration / battle / battle_result / battle_reward / transition 条带），帧尺寸 1706×961（条带 1706×56） |
| 迭代 | 每张样本 × 5 轮 = **30 次**（阶段计时）；单 detector 细分 = 每张 × 3 轮 = 18 次 |
| 计时 | `time.perf_counter()` 包裹单个调用，统计 min / mean / p50 / p95 / max |
| 阶段口径 | `capture` = 构造 `Frame`（图片解码**预置在测量之外**）；`detect` = `EvidenceRegistry.detect_all`；`aggregate` = `EvidenceAggregator.aggregate`；`classify` = `StateClassifier.classify(evidence)`（**内含 aggregate**）；`process` = `EvidenceFrameProcessor.process`（= detect + classify） |
| 配置 | `configs/config.yaml` 默认值（10 个 detector、6 条规则、每 detector `max_templates ≤ 8`、`scales` 5 档） |

**已知局限**（不影响"瓶颈位置"的结论）：

1. `capture` 只测了 `Frame.create`（图片解码在预热阶段完成）→ **不代表真实采集**；真实 `dxcam`/`mss` 的采集耗时需在有游戏窗口的环境单独测量；
2. 各阶段是**分别独立测量**（不是同一次调用内的分解），聚合/分类会被重复计算一次；
3. 无 GPU 加速、无并行；CPU 型号与频率会影响绝对值（相对占比更有参考意义）。

---

## 2. 分阶段结果（30 次）

```text
capture (Frame.create)     n=30 min=  0.01ms mean=  0.01ms p50=  0.01ms p95=  0.02ms max=  0.03ms
detect (detect_all)        n=30 min= 26.25ms mean=356.45ms p50=408.82ms p95=513.64ms max=513.90ms
aggregate (rules)          n=30 min=  0.04ms mean=  0.08ms p50=  0.08ms p95=  0.10ms max=  0.13ms
classify (incl. aggregate) n=30 min=  0.04ms mean=  0.08ms p50=  0.07ms p95=  0.13ms max=  0.15ms
process (detect+classify)  n=30 min= 25.12ms mean=350.61ms p50=405.15ms p95=515.31ms max=542.20ms

estimated linked frame latency = capture + detect + classify = 356.54 ms
classification-only (classify − aggregate) ≈ 0.00 ms
```

| 阶段 | 平均 | p95 | 占单帧比例（mean） |
| --- | --- | --- | --- |
| capture（`Frame.create`，不含解码） | 0.01 ms | 0.02 ms | ~0% |
| **detect（`detect_all`）** | **356.45 ms** | **513.64 ms** | **≈99.98%** |
| aggregate（规则聚合） | 0.08 ms | 0.10 ms | ≈0.02% |
| classify（含 aggregate） | 0.08 ms | 0.13 ms | ≈0.02% |

**读法**：证据聚合（规则匹配、可靠度加权、状态可信度校准）与状态判定的成本可以忽略（0.08 ms 量级）；
**全部耗时都在证据检测（模板匹配）**。`min=26 ms` 出现在 transition 条带（1706×56，ROI 裁剪后极小）——
说明耗时随**帧/ROI 面积**剧烈变化。

---

## 3. 瓶颈位置：按 detector 细分（10 个 detector）

```text
bottom_menu_exploration    mean=124.15ms p95=177.42ms share=32.7%
bottom_menu_yard           mean=101.65ms p95=179.10ms share=26.8%
chapter_panel              mean= 57.34ms p95= 84.79ms share=15.1%
reward_panel               mean= 33.63ms p95= 50.06ms share= 8.9%
battle_skill_area          mean= 19.81ms p95= 29.83ms share= 5.2%
victory_icon               mean= 18.52ms p95= 28.20ms share= 4.9%
battle_action_bar          mean=  8.48ms p95= 12.05ms share= 2.2%
battle_auto_button         mean=  6.36ms p95=  9.48ms share= 1.7%
battle_shikigami_icons     mean=  6.11ms p95=  8.10ms share= 1.6%
transition_banner          mean=  3.24ms p95=  3.75ms share= 0.9%
TOTAL                      mean=379.29ms
```

| 排名 | detector | ROI（比例 → 1706×961 下的像素） | 模板数 × 尺度 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | `bottom_menu_exploration` | `[0, 0.80, 1, 1]` → **1706×192** | 8 × 5 = 40 次匹配 | 全宽 + 8 个模板 + 5 档尺度 → 最贵 |
| 2 | `bottom_menu_yard` | `[0, 0.80, 1, 1]` → **1706×192** | 8 × 5 = 40 次匹配 | 同上（同一 ROI，两套模板） |
| 3 | `chapter_panel` | `[0.80, 0.15, 1, 0.70]` → 341×528 | 6 × 5 = 30 次匹配 | 面积次之 |
| 4 | `reward_panel` | `[0.15, 0.15, 0.62, 0.90]` → 802×721 | 1 × 5 = 5 次匹配 | 单模板但 ROI 面积大 |
| 5–10 | battle 系列 + victory + transition | ROI 更小（多为窄条） | ≤ 3 × 5 | 面积小 → 便宜 |

**瓶颈结论**：耗时 ≈ `ROI 面积 × 模板数 × 尺度数`。
**前 3 个 detector（两个 `bottom_menu` + `chapter_panel`）合计占 74.6%**；
`bottom_menu` 的两个 detector 共用同一块**全宽**底部 ROI（1706×192），是首要瓶颈。

---

## 4. 500ms interval 是否可行？

| 结论 | 依据 |
| --- | --- |
| **不可行（边界超支）** | `detect` p95 = **513.64 ms** > 500 ms；`process` p95 = 515.31 ms；mean 356 ms 已占预算 71% |
| 即"平均能跑、尾部会超" | mean 356 ms < 500 ms，但 p95/max 已越过周期；一旦超时，`StateMonitor` 的固定节拍会按"单轮耗时超过周期 → 重新对齐"处理（**不追赶**），表现为**实际周期被拉长到 ≈0.36–0.55 s**，而非丢帧 |
| 数据缺失时更慢 | 本次测量已包含 `win32` 无关的纯 CPU 检测；若注入 OCR（`text_targets` 且 Tesseract 可用），`chapter_panel` / `battle_auto_button` / `victory_icon` / `transition_banner` 还会各自附加一次 OCR 耗时 |
| 真实采集未计入 | `capture` 只测了 `Frame.create`；真实 `dxcam` 采集通常几毫秒，但会进一步压缩余量 |

**按阶段要求：不做优化**。以下为**记录**（供后续独立阶段决策，不在本阶段实施）：

| 可选方向 | 预期收益 | 代价 |
| --- | --- | --- |
| 缓存模板灰度图（现在每次 `match` 都重算 `to_grayscale(template.image)`） | 小（模板很小） | 低 |
| 缩小 `bottom_menu` ROI（当前全宽；两套模板的差异只在局部图标） | 大（占 59.5%） | 需重新采集/验证 ROI 与样本一致 |
| 降低 `max_templates`（8 → 4）或 `scales`（5 → 3） | 大（线性） | 牺牲对窗口尺寸差异的鲁棒性 |
| 证据分级（低频证据每 N 帧检测一次，例如 `bottom_menu` 隔帧） | 大 | 引入检测延迟与状态抖动（需与稳定化窗口协同标定） |
| 多线程/进程并行 detector；GPU 加速 | 大 | 复杂度与可测试性 |
| OCR 移出热路径（仅对已命中区域触发） | 中 | 需改证据层触发逻辑 |

---

## 5. 复现方式

测量脚本是临时脚本（不在仓库保留）。复现步骤：

1. 用真实图片构造 6 张样本（与 `examples/evidence_state_demo.py` 的默认样本一致）；
2. `EvidenceRegistry` + `create_detectors(config.evidence, transition_classifier=build_transition_probe(config))`；
3. `StateClassifier.from_evidence_config(config.evidence)`（取 `classifier.aggregator` 用于单独测 aggregate）；
4. 用 `time.perf_counter()` 分别包裹 `Frame.create` / `detect_all` / `aggregate` / `classify` / `processor.process`；
5. 也可直接用 `examples/runtime_pipeline_demo.py --real-time --interval-ms 500` 观察真实节拍下的表现
   （该 Demo 会体现"单轮耗时超过周期时重新对齐"的固定节拍行为）。

---

## 6. 与 Phase 5 报告的关系

本文件只负责**性能事实与瓶颈记录**；链路迁移、兼容性与测试结论见
`docs/PHASE5_RUNTIME_PIPELINE_REPORT.md`。
