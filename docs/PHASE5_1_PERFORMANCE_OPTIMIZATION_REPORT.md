# PHASE 5.1 Report — Performance Optimization

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 5.1（任务二～七：profiling / 优化 / 统计 / 缓存 / 计时 / 测试） |
| 目标 | 把实时检测能力提升到 **500ms interval 可稳定运行** |
| 原则 | **正确性 > 性能**：任何优化必须证明"同一 Frame → 优化前后结果完全相同" |
| 结果 | `detect` **mean 356.45 ms → 197.20 ms（−44.7%）**、**p95 513.64 ms → 301.27 ms（−41.3%）**；验收标准达成（mean <200ms、p95 <400ms） |
| 未做 | 未改判定逻辑、未降阈值、未删 detector、未改 dataset、未改规则语义、未改 confidence contract、未实现点击 |

---

## 1. 优化前 baseline（Phase 5 实测）

```text
capture (Frame.create)     n=30  mean=  0.01ms p95=  0.02ms
detect (detect_all)        n=30  mean=356.45ms p50=408.82ms p95=513.64ms max=513.90ms
aggregate (rules)          n=30  mean=  0.08ms p95=  0.10ms
classify (incl. aggregate) n=30  mean=  0.08ms p95=  0.13ms
process (detect+classify)  n=30  mean=350.61ms p95=515.31ms

结论（Phase 5）：p95 ≈ 514 ms 已越过 500 ms 周期 → 500 ms interval 不可行。
```

按 detector 细分（Phase 5 实测）：

```text
bottom_menu_exploration    mean=124.15ms p95=177.42ms share=32.7%
bottom_menu_yard           mean=101.65ms p95=179.10ms share=26.8%
chapter_panel              mean= 57.34ms p95= 84.79ms share=15.1%     ← 前三名合计 74.6%
reward_panel               mean= 33.63ms p95= 50.06ms share= 8.9%
battle_skill_area          mean= 19.81ms p95= 29.83ms share= 5.2%
victory_icon               mean= 18.52ms p95= 28.20ms share= 4.9%
battle_action_bar          mean=  8.48ms p95= 12.05ms share= 2.2%
battle_auto_button         mean=  6.36ms p95=  9.48ms share= 1.7%
battle_shikigami_icons     mean=  6.11ms p95=  8.10ms share= 1.6%
transition_banner          mean=  3.24ms p95=  3.75ms share= 0.9%
TOTAL                      mean=379.29ms（各 detector 求和）
```

---

## 2. 瓶颈分析

**耗时模型**：`ROI 面积 × 模板数 × 尺度数 × 单次 matchTemplate 成本`。

| 观察 | 数据 |
| --- | --- |
| 耗时集中在 3 个 detector | `bottom_menu_exploration` + `bottom_menu_yard` + `chapter_panel` = **74.6%** |
| 共同特征 | ROI 面积大（前者全宽 1706×192、后者 341×528）× 模板数多（8 / 8 / 6）× 5 档尺度 |
| 每帧的重复计算 | `TemplateMatcher.match` 对每个模板都要重算 `to_grayscale(template.image)`，并对每个 scale 重算 `resize_image(...)`——**模板与其缩放图在运行期完全不变**，属于纯重复开销 |
| 为什么 Phase 5 没看出来 | Phase 5 只测到"detector 级"粒度，没有区分"模板预处理"与"滑窗匹配" |

**关键判断**：真正不可省的是 `cv2.matchTemplate` 的滑窗成本（由 ROI/模板/尺度决定）；
而**模板预处理（灰度 + 缩放）是每帧可省的重复计算**——这正是本阶段优化的落点。

---

## 3. 优化措施

| # | 措施 | 位置 | 收益 | 风险 |
| --- | --- | --- | --- | --- |
| 1 | **模板金字塔缓存**：把"灰度化 + 各 scale 缩放"从每帧每模板移到首次使用并缓存（键含模板标识与 scale 元组） | `TemplateMatcher.template_pyramid` / `match` | **大（−45%）** | **零**：模板与缩放结果都是确定性的，逐字段等价（第 5 节有证明测试） |
| 2 | **可选 profiling**：`EvidenceRegistry.detect_all(frame, profile=True)` 逐个 detector 计时并采集静态信息（name / templates / roi / scales / elapsed_ms），写入 `last_profile` + `profile_summary()`；默认 `False` 零开销 | `evidence/registry.py` | 观测能力（定位优化对象） | 无（默认关闭，且已测"不改变结果"） |
| 3 | **帧指纹缓存**（默认关闭）：`EvidenceRegistry(cache_size=N)` 按**帧内容指纹**（blake2b，含 shape/dtype/像素，不含时间戳）复用检测结果 | `evidence/registry.py` | 场景相关（完全相同的画面） | 低：内容相同 → 结果必然相同；不按时间缓存、不跨内容复用 |
| 4 | **scale 命中统计**：记录每个 scale 被选为最佳匹配的次数 | `TemplateMatcher.scale_hits` | 观测（评估 scale 有效性） | 无（只读统计） |
| 5 | **极小帧健壮性**：`Roi.to_rect` 增加边界裁剪；`TemplateEvidenceDetector.best_match` 对"裁不出像素"直接按未检出处理 | `evidence/models.py` / `evidence/base.py` | 修复真实缺陷（严重） | 低：正常尺寸下比例换算本来就落在图内，行为不变 |
| 6 | **耗时画像**：`EvidenceFrameProcessor.process(frame, profile=True)`（detect/classify 分解）与 `StateMonitor(timing=True)`（`CycleTiming`：frame/detect/state/stabilize/total） | `pipeline/frame_processor.py` / `runtime/state_monitor/timing.py` | 观测能力（每轮预算可见） | 无（默认关闭，且已测"不改变事件"） |

**明确未做（按要求或按风险）**：

| 未做 | 原因 |
| --- | --- |
| 缩小 `bottom_menu` / `chapter_panel` 的 ROI | 会改变证据结果（可能漏检），属**配置决策**，需要重新采样与回归验证 |
| 减少模板数（8→4）或尺度数（5→3） | 会改变置信度分布与判定，需数据验证；本阶段只提供 `scale_hits` 统计 |
| 粗到精（下采样先匹配） | 会改变定位精度与漏检行为 |
| 更换匹配方法（`TM_CCOEFF_NORMED` → 其它） | 会改变分数口径 |
| 多线程 / GPU 加速 | 架构级改动，不属本阶段 |
| 缓存默认开启 | 真实连续截图的字节相同概率低（噪声/压缩），默认开启只会增加哈希成本；保持默认关闭 |

---

## 4. 前后性能对比

同一基准（6 张真实图 × 5 轮 = 30 次，同一机器）：

| 指标 | 优化前（Phase 5） | 优化后 | 变化 |
| --- | --- | --- | --- |
| `detect` mean | 356.45 ms | **197.20 ms** | **−44.7%** |
| `detect` p50 | 408.82 ms | 225.19 ms | −44.9% |
| `detect` p95 | 513.64 ms | **301.27 ms** | **−41.3%** |
| `detect` p99 / max | 513.90 ms | 320.93 ms | −37.5% |
| `detect` min | 26.25 ms | 1.42 ms | −94.6% |
| `aggregate` | 0.08 ms | 0.06 ms | 不变（本就不是瓶颈） |
| `classification` | 0.08 ms | 0.05 ms | 不变 |

**验收标准**：

| 目标 | 要求 | 实测 | 结论 |
| --- | --- | --- | --- |
| `detect` mean | < 200 ms | **184.86–197.20 ms** | **达成** |
| `detect` p95 | < 400 ms | **274.44–301.27 ms** | **达成** |

**500ms 周期可行性**（含真实采集，详见 `docs/PHASE5_1_REAL_RUNTIME_REPORT.md`）：

| 链路 | mean | p95 | 预算占用 |
| --- | --- | --- | --- |
| dxcam + detect + classify + stabilize | ≈185 ms | ≈276 ms | 37% / 55% |
| mss + detect + classify + stabilize | ≈215 ms | ≈318 ms | 43% / 64% |

→ **500 ms interval 现在可行**，且留有约 45% 余量（未启用 OCR 的前提下）。

---

## 5. 准确性验证（"同一 Frame → 结果相同"）

| 验证 | 方法 | 结果 |
| --- | --- | --- |
| **模板金字塔等价性** | 测试内实现**朴素参考匹配**（Phase 5.1 前的逻辑：每帧重算灰度与缩放），对**同一真实帧 + 全部真实模板**逐字段比较 `confidence` / `rect` / `scale` / `template_id` | **完全一致**（`test_template_pyramid_matches_naive_reference`，含 `None` 情形） |
| profiling 无副作用 | `detect_all(frame)` vs `detect_all(frame, profile=True)` 比较全部证据的 `describe()` | 完全一致 |
| 缓存无副作用 | 同一帧连续两次 `process(frame)`：结果 `==`；不同内容的帧不复用 | 一致 |
| 计时无副作用 | `timing=True/False` 两次运行的 `events` / `initial_state` / `cycles` | 完全一致 |
| 契约未变 | `EvidenceResult` / `StateResult` / `StateMonitor` / `StateEvent` 字段与语义 | 未改动（既有契约测试全绿） |
| 既有行为 | 全量测试（含 detector 契约、聚合、迁移、Runtime） | **832 passed, 1 skipped, 0 failed** |

**顺带修复的真实缺陷**：极小帧（2×2）下 `Roi.to_rect` 会算出越界矩形 → 裁剪出空数组 →
`cv2.cvtColor` 抛 `cv2.error`，而 Runtime 只捕获 `OnmyojiVisionError` → **监控循环会崩溃**。
现已修复（边界裁剪 + 空裁剪守卫），并新增 `test_tiny_frame_does_not_crash` /
`test_empty_image_is_rejected` 守护。

---

## 6. 测试结果

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# → 832 passed, 1 skipped, 0 failed

# 阶段基线（Phase 5 完成时）
# → 818 passed, 1 skipped

.\.venv\Scripts\python.exe -m pytest tests/test_phase51_performance.py -q
# → 14 passed
```

### 6.1 新增用例对照（任务七的 6 项要求）

| 要求 | 覆盖用例 |
| --- | --- |
| 1. profiling 不影响结果 | `test_profiling_does_not_change_results`、`test_profile_summary_is_readable` |
| 2. 优化前后 StateResult 相同 | `test_template_pyramid_matches_naive_reference`（逐字段）、`test_template_pyramid_is_reused_across_calls` |
| 3. 缓存不会改变状态 | `test_frame_cache_does_not_change_state`、`test_cache_is_disabled_by_default`、`test_fingerprint_distinguishes_content` |
| 4. 空 frame 异常处理 | `test_empty_image_is_rejected`、`test_tiny_frame_does_not_crash`、`test_tiny_frame_with_timing` |
| 5. detector 失败继续运行 | `test_detector_failure_continues_with_timing`（+ Phase 5 的 2 个同类用例） |
| 6. 500ms loop 模拟运行 | `test_simulated_500ms_loop`（clock 恰好推进 `cycles × interval`）、`test_timing_is_off_by_default`、`test_timing_does_not_change_events` |

### 6.2 任务完成对照

| 任务 | 状态 | 交付物 |
| --- | --- | --- |
| 一、真实 FrameSource 测量 | ✓ | `docs/PHASE5_1_REAL_RUNTIME_REPORT.md`（dxcam 0.42ms / mss 29.83ms；分阶段 mean/p50/p95/p99） |
| 二、detector profiling | ✓ | `detect_all(profile=True)` + `last_profile` + `profile_summary()`（默认关闭，不永久打印） |
| 三、模板匹配优化 | ✓ | 模板金字塔缓存（唯一显著收益项，逐字段等价） |
| 四、多尺度统计 | ✓ | `TemplateMatcher.scale_hits`（记录命中次数；**未删除任何 scale**） |
| 五、检测缓存 | ✓ | `cache_size` + 帧指纹（内容 hash，不按时间；默认关闭） |
| 六、Runtime Loop 计时 | ✓ | `StateMonitor(timing=True)` + `CycleTiming`（frame/detect/state/stabilize/total）+ Demo `--timing` |
| 七、性能回归测试 | ✓ | `tests/test_phase51_performance.py`（14 个用例，覆盖 6 项要求） |

---

## 7. 剩余限制

| 编号 | 限制 | 说明 / 建议 |
| --- | --- | --- |
| R1 | `detect` 仍是绝对瓶颈（≈185 ms，占链路 99.97%） | 进一步提速只能靠**配置决策**（缩小 ROI / 降模板数与尺度）或算法改动（下采样、并行），均需独立验证 |
| R2 | **scale 1.1 / 1.2 在 6 张样本上从未被选中**（命中集中在 0.8 / 0.9 / 1.0） | 样本来自**单一分辨率**，不足以证明可删；建议在多样本/多分辨率上复测后再决定（届时可省约 40% 的 matchTemplate 次数） |
| R3 | 帧指纹缓存在**真实连续截图**上命中率低（字节不会完全相同） | 价值主要在离线回放、暂停画面、静态界面；默认关闭，需要时显式 `cache_size>0` |
| R4 | 未启用 OCR 时的预算结论 | 一旦开启 `text_targets`，带 OCR 的 4 个 detector 会各自附加 OCR 耗时，需重新测量 |
| R5 | 未做长跑（小时级）与多分辨率验证 | 内存/句柄、温度降频、窗口尺寸变化均未覆盖 |
| R6 | 真实采集测量是**整屏**而非游戏窗口客户区 | 客户区更小 → 采集更快；但需在有游戏的环境复测 |
| R7 | `roi` 的像素尺寸随帧尺寸变化 | profiling 输出的是**当帧**的 ROI 像素尺寸（比例不变），跨分辨率对比时需看比例 |
| R8 | `pyramid_cache_hits` 统计的是"复用次数" | 首次使用必然 miss；`cache_hits==0` 不代表未生效（`pyramid_cache_size>0` 才是判据） |

**本阶段未做的事**：未新增 detector / 状态 / dataset；未改规则语义、confidence contract、`EvidenceResult` / `StateResult` / `StateMonitor` / `StateEvent` 契约；未删除任何 detector 或 scale；未实现点击；未做架构重构。
