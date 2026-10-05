# Phase 12.0-C Scene D-F Runtime Validation Report

> 阶段目标：完成**真实 Scene D / E / F** 验证 —— D 窗口尺寸变化与恢复、E 失焦 / capture interruption / snapshot 恢复、F ≥1 小时稳定长跑。
> **本阶段不做任何生产代码修改**（`git status` 全程仅含历史未提交的 `tools/build_annotation_review_pack.py`，未修改 / 未 stage / 未 commit）。
> 原则：**只测量，不修复**；发现问题先记录、停止分析，**不边跑边改算法**。

---

## 1. baseline commit

```
46916ab feat(runtime): add bbox and rss runtime observability   ← HEAD（Phase 12.0-B）
71f9a0f docs(runtime): add yard runtime soak validation report  （12.0-A）
63f5998 feat(vision): switch yard production templates to proposed36
```

## 2. production config（**全程未变**）

| 项 | 值 |
|---|---|
| `template_include` | **Proposed36**（`template_count = 36`） |
| `template_prefilter` / `prefilter_top_k` | `thumb` / `4` |
| `threshold` / `roi_scale` | `0.85` / `0.75` |
| `FIT_TOLERANCE_PX` | `2`（matcher scales = {0.600, 0.675, 0.750, 0.825, 0.900}） |
| `max_templates` / `dedupe_threshold` / `per_source_limit` / `round_robin` | `36` / `0` / `0` / `False` |
| `runtime.state_monitor.yard_metrics` | **`False`**（生产默认；soak 仅用运行期 `config.model_copy` override） |
| `action.enabled` / `backend` | **`false`** / **`dry_run`**（`real_input = false`） |

## 3. tested environment

| 项 | 值 |
|---|---|
| OS / shell | Windows 11（?）/ PowerShell |
| Python | 3.12.10（`.venv`） |
| 采集后端 | `dxcam`（`BACKEND_PRIORITY` 含 `DxcamCapture` / `MssCapture`） |
| 目标窗口 | `阴阳师-网易游戏`（hwnd 在 D0–D3 为 `0x130ab0`，E/E3/F 变为 `0x461048` —— 窗口被重建过） |
| DPI / 缩放 | 150%（逻辑尺寸 = 像素 ÷ 1.5） |
| 客户端 | **单设备、单 DPI、单客户端**（因此本报告**不**称"跨分辨率泛化验证"） |
| soak 工具 | `tools/run_runtime_soak.py`（**未新造工具**） |

## 4. actual window sizes（用户手动调整，程序从不 resize）

| 段 | 客户区（真实像素） | 与生产工作点关系 |
|---|---|---|
| D0 / D3 / E / E3 / F | **1704 × 960** | **= 生产工作点**（模板来源尺寸） |
| D1 | **1156 × 651** | 等比缩小 **0.678×** |
| D2 | **2305 × 1299** | 等比放大 **1.353×**（纵横比 1.7745，与原 1.7750 一致） |

---

## 5. Scene D — 窗口尺寸变化与恢复

### 5.1 各段

| 段 | 客户区 | cycles / duration | detected | **bbox invalid/samples** | detector_ms p50 | **deadline miss** | candidate / match_calls | fallback | states / transitions / flaps | capture |
|---|---|---|---|---|---|---|---|---|---|---|
| **D0** | 1704×960 | 120 / 60.012 s | **120/120 (1.000)** | **0 / 120** | 26.5 | **0** | 4 / 20 | 0 | yard 118+pending 2 / 0 / 0 | success 120 · failures 0 |
| **D1** | 1156×651 | 240 / 120.007 s | **0/240 (0.000)** | 0 / **0** | **0.66** | **0** | 4 / 20 | 0 | unknown 238+pending 2 / 0 / 0 | success 240 · failures 0 |
| **D2** | 2305×1299 | 130 / 120.238 s | **0/130 (0.000)** | 0 / **0** | **101.3** | **130/130 (1.0)** | 4 / 20 | 0 | unknown 128+pending 2 / 0 / 0 | success 130 · failures 0 |
| **D3** | 1704×960 | 120 / 60.104 s | **120/120 (1.000)** | **0 / 120** | 27.8 | **1**（首轮 593 ms 冷启动） | 4 / 20 | 0 | yard 118+pending 2 / 0 / 0 | success 120 · failures 0 |

### 5.2 Scene D 的 confidence / latency

| 段 | confidence min / p50 / max | cycle latency p50 / p95 / max | yard_ms p50 / p95 |
|---|---|---|---|
| D0 | 0.8749 / 0.9195 / 0.9327 | 250 / 313 / 421 | 26.5 / 42.4 |
| D1 | —（未命中，无样本） | 0 / 16 / 32 | 0.66 / 1.17 |
| D2 | —（未命中，无样本） | 891 / 1125 / 1203 | 101.3 / 135.7 |
| D3 | —（见 §7 F 段同尺寸 0.8729~0.9352） | 250 / 344 / 593 | 27.8 / 39.4 |

### 5.3 fit / bounded fit 行为（**关键机制**）

`cap_028` 模板原图 **1706×192**；各客户区下的匹配图（`roi_scale = 0.75`）与模板档位：

| 客户区 | ROI | 匹配图 | @0.600 | @0.675 | @0.750 | @0.825 | @0.900 | 结果 |
|---|---|---|---|---|---|---|---|---|
| **1704×960** | 1704×192 | **1278×144** | 1024×115 fits | 1152×130 fits | **1280×144 overflow 2** | overflow 129 | overflow 257 | **bounded fit 生效（在 `FIT_TOLERANCE_PX=2` 内）⇒ 命中** ✅ |
| 1156×651 | 1156×130 | 867×98 | overflow **157** | 285 | 413 | 540 | 668 | **全档超容差 ⇒ 全跳过 ⇒ 0 检出** |
| 2305×1299 | 2305×260 | 1728×195 | fits | fits | fits | fits | fits | 尺寸可容纳，但**内容已被等比放大 1.353×** ⇒ 需要 scale≈1.35（**档位只到 0.9**）⇒ 分数低于 0.85 ⇒ 0 检出 |

### 5.4 Scene D 结论要点

* **尺寸变化链路正确**：每次尺寸变化后 `capture` 都拿到**新的真实客户区尺寸**，ROI 随之正确重算（D3 恢复后 `detector_ms` 27.8 ms、detected 120/120 完全回归）✅
* **bbox 恒合法**：所有**命中**轮次 `bbox.invalid = 0`（D0 120/120、D3 120/120）✅
* **无陈旧尺寸快照**：D3 `snapshot invalidations = 0` 且 119/120 轮 `processed`（用**新**帧）✅
* **无崩溃/无 flap**：四段 `errors 0`、`transitions 0`、`flaps 0`、`fallback 0` ✅
* **识别率下降的两侧根因同一**：模板是**固定像素源**，`roi_scale` 只降采样 ROI、**不缩放模板**，matcher 档位 **0.6~0.9 只覆盖同一客户区下的尺度抖动**。D1 需要 <0.6、D2 需要 ≈1.35 ⇒ 均落在档位之外 ⇒ **已知架构限制**，**不是 detector failure**。
* **D2 附带发现（性能缩放）**：匹配图面积 1.83×（1278×144 → 1728×195）时 `yard_ms` 26.5 → 101.3 ms、**每轮 891~1203 ms > 500 ms 预算** ⇒ 120 s 只跑 130 轮（D0 为 240 轮）。调度器正确 `[late] re-aligned`、`errors 0`、**无 flap** ⇒ **降级行为健康，但吞吐减半**。
* 表述纪律：本段为 **real-window-size robustness diagnostic**（单设备 / 单 DPI / 单客户端），**不**称"完整跨分辨率泛化验证"。

## 6. Scene E — 失焦 / capture interruption / snapshot 恢复

### 6.1 E1 失焦（用户手动 Alt+Tab，**约 10 s**）

| 字段 | 值 | 判定 |
|---|---|---|
| 实测窗口 | cycle **118~140 = 23 轮 ≈ 11.5 s** | 与用户"约 10 s"吻合 |
| `status` / `skip_reason` | `skipped` / **`not_foreground`** 23/23 | ✅ |
| `capture_status` / `foreground_status` | `skipped` / **`blocked`** 23/23 | ✅ **前台门控，不抢占焦点** |
| `unknown_reason` | **`not_foreground`**（明确区别于 `detector_failure` / `no_evidence`） | ✅ |
| `state` / `confidence` | `None` / `0.0`（**不产出判定**） | ✅ **无 stale target** |
| `stable_state` | **保持 `yard` 23/23** | ✅ 短失焦不打断稳定态 |
| `snapshot_status` | cycle 119~140 全 **`no_frame`** ⇒ 快照按设计失效 | ✅ |
| `action` | `action_enabled=False`、**action invocation = 0** | ✅ |
| `fallback` | 0（**无 fallback 风暴**） | ✅ |
| `errors` | 0 | ✅ |

### 6.2 E2 恢复（cycle 141 起，460 轮）

| 指标 | 值 |
|---|---|
| **time-to-first-valid-frame** | **1 轮（≈0.5 s）**（cycle 141 `capture_status=ok`） |
| **time-to-yard-detected** | **1 轮**（cycle 141 `yard_detected=True`，`cap_028`，conf **0.9220**） |
| **time-to-stable-yard** | **0 轮**（`stable_state=yard` 在失焦期从未被打断；恢复后 **460/460 全 yard**） |
| snapshot 恢复 | 141 `no_frame` → **142 `processed`（1 轮重建）** |
| transitions / flaps | **0 / 0** |
| E2 detected | **460/460** |

### 6.3 E3 最小化（用户手动点「—」，**实测 13.5 s**）

| 字段 | 值 | 判定 |
|---|---|---|
| 实测窗口 | cycle **174~200 = 27 轮 ≈ 13.5 s**（`20:09:47 → 20:10:00`） | 用户描述为 T+90s→T+120s（30 s），**以实测为准** |
| `skip_reason` | **`window_minimized`** 27/27 | ✅ **与 E1 的 `not_foreground` 精确区分** |
| `frame_ms` | **0.05~0.43 ms** | ✅ **根本没进入采集** ⇒ dxcam 不可能拿到黑帧/不崩溃 |
| `state` / `confidence` / `stable_state` | `None` / `0.0` / `yard` 27/27 | ✅ 无 stale target、稳定态不抖动 |
| `action` / `snapshot` | `(False, dry_run)` 27/27；`no_frame` 26 | ✅ |
| 恢复 | cycle **201** 即 `ok` / `yard` / conf **0.9200**；snapshot 202 起 `processed` | ✅ **1 轮恢复** |
| 有效轮复核 | `status=ok` **573 轮 ⇒ detected 573/573 = 1.000** | ✅ |

### 6.4 明确区分「预期失焦」与「detector failure」

失焦/最小化期间**从不**进入判定路径：`state = None`、`confidence = 0.0`、`unknown_reason` 明确标注原因（`not_foreground` / `skipped`），**未被**误标为 `detector_failure` 或 `no_evidence` ⇒ 二者在本阶段**完全可区分** ✅

## 7. Scene F — ≥1 小时稳定长跑

| 项 | 值 |
|---|---|
| duration / cycles | **3600.025 s** / **7200**（interval 500 ms） |
| 时段 | `20:15:56 → 21:15:55`（+08:00） |
| **passive / manual segments** | 用户申报 **无人工干预**；数据侧 **`status = ok 7200/7200`**、`skip_reason` 全 `None`、`snapshot invalidations = 0`、`capture failures = 0` ⇒ **全程为单一 passive 段**，无 manual-intervention 段可分离 |
| `status` | **ok 7200/7200** |
| cycle latency | min 203 · **p50 235** · p90 266 · p95 282 · p99 313 · **max 515** |
| **deadline misses** | **1 / 7200（0.01%）** —— cycle 798，elapsed 515 ms、overrun **15 ms**，调度器 `[late] re-aligned`，**无 error** |
| yard latency | p50 **25.7** · p90 28.6 · p95 31.6 · p99 37.9 · max 57.6 |
| vision_ms | p50 227.0 · p90 258.7 · p95 276.8 · p99 308.4 · max 511.3 |
| frame_ms | p50 1.4 · p95 1.9 · max 25.0 |
| **detection ratio** | **7200/7200 = 1.000**（passive） |
| confidence | min **0.872929** · p50 0.919113 · p90 0.925164 · p95 0.926401 · max 0.935152 |
| candidates | **{4: 7200}** |
| match calls | **{20: 7200}**（`yard_total_templates = 36` 恒定） |
| fallback | **0**（`reason = None` 7200） |
| **bbox samples / invalid** | **7200 / 0** ✅（`yard_bbox_oob = False` 7200/7200） |
| state transitions / flaps | **0 / 0**；states `yard` 7200；`stable_state` yard 7198 + None 2（起始 pending） |
| snapshot | `processed` 7199；**`snapshot_invalidated` 累计 0** |
| 命中模板 | `cap_028` 7150 + `cap_007` 50 |
| 分段（每 10 分钟） | lat p50 **234→235→235→235→235→235**（**无热漂移**）；conf p50 0.9185~0.9197；**bbox_oob=True 在 6/6 窗口均为 0** |

## 8. RSS

| 项 | 值 |
|---|---|
| `rss_start` | **200.87 MB**（200,867,840 B） |
| `rss_peak` | **274.28 MB**（274,276,352 B） |
| `rss_end` | **261.60 MB**（261,595,136 B） |
| `rss_samples` | **3573**（1 s 间隔） |
| python alloc current / peak | 37.65 MB（37,646,788 B）/ 47.58 MB（47,581,635 B） |

跨段旁证（每段都是**独立进程**，故 `start` 不应视为"上段泄漏"）：

| 段 | 时长 | start | peak | end |
|---|---|---|---|---|
| D0 | 60 s | 200.96 MB | 250.19 MB | 238.92 MB |
| D3 | 60 s | 200.86 MB | 258.32 MB | 242.42 MB |
| E | 300 s | 199.60 MB | 252.67 MB | 243.49 MB |
| E3 | 300 s | 200.67 MB | 252.70 MB | 242.96 MB |
| **F** | **3600 s** | **200.87 MB** | **274.28 MB** | **261.60 MB** |

**趋势结论：`insufficient_data`（无法判定）** —— 见 §16 缺口 ④。
按纪律：**不**把 `peak > start` 称为内存泄漏；**不**臆断 warmup vs 持续增长。（若只看 `end - start = +60.7 MB`，需 12.0-D 补时序才能定性。）

## 9. safety

| 指标 | 值 |
|---|---|
| `action_enabled` | **False**（7200/7200） |
| `backend` | **`dry_run`**（7200/7200） |
| `real_input` | **false** |
| **action invocation count** | **0** |
| 前台门控 | 只检查、**不抢占**（`not_foreground` / `window_minimized` 均为"跳过本轮"） |
| 窗口操作 | **无** `SetForegroundWindow`、无 resize、无最小化、无 Alt-Tab、无导航、无移动角色 |
| 输入 | **无**点击 / **无**键盘输入 |

## 10. pytest

```
python -m pytest -q  ⇒  1884 passed, 1 skipped
```

（= Phase 12.0-B 基线，未新增 / 未删除测试，未降低数量，未新增无理由 skip。）

## 11. dataset integrity

| 命令 | 结果 |
|---|---|
| `python tools/check_dataset.py` | **PASS**（`raw` total **178**；`yard_live` total **29**） |
| `python tools/raw_manifest.py` | **`ok: true`** —— 历史 `178/178`、live `29/29`，`missing` / `added` / `changed` 全空 ✅ |
| `python tools/check_annotations.py` | **PASS** |
| 生产配置 / 模板集合 | **未变**（36 张 Proposed36） |
| `git status --short` | 仅 ` M tools/build_annotation_review_pack.py`（历史未提交，未修改 / 未 stage / 未 commit） |

## 12. Scene D verdict

**CONDITIONAL（PASS）**

* ✅ Runtime 侧全部通过：尺寸变化 → capture/ROI 正确重算、`bbox.invalid = 0`、无陈旧快照、无崩溃、无 flap、恢复后识别 **100% 回归**；
* ⚠️ D1(1156×651) / D2(2305×1299) 识别 **0/240、0/130**，**忠实反映**模板档位之外（需 <0.6 / ≈1.35），属**已知架构限制**，**不调参**；
* ⚠️ D2 暴露"大窗口下 500 ms 周期无法维持"（**deadline miss 130/130**、吞吐减半），**降级行为健康**但需作为已知性能缩放限制记录。

## 13. Scene E verdict

**PASS**

* E0 前台基线 117/117 ok；
* E1 失焦 23 轮：门控 `not_foreground`、**0 action**、**0 stale target**、snapshot 正确失效、**无 fallback 风暴**、0 error；
* E2 恢复：**1 轮**内重新有效采帧并判定 yard，`transitions/flaps = 0`，460/460 detected；
* E3 最小化 27 轮：`window_minimized` 独立原因、**未进入采集**、`frame_ms ≈ 0.05 ms`、**0 action**、**1 轮恢复**、有效轮 573/573。

## 14. Scene F verdict

**PASS**

* duration **3600.025 s ≥ 3600**、cycles **7200**；
* **deadline miss 1/7200（0.01%）**，唯一 1 次为 15 ms 系统调度尖峰（已单独分析，**不是**算法问题，**未**改算法）；
* **flaps = 0**、**fallback = 0**、**bbox invalid = 0 / samples = 7200**、candidates ≤ 4、match_calls = 20、`action invocation = 0`、`real_input = false`；
* passive detected ratio **1.000**，与短测（D0/E 的 100%）**一致**，**无 miss 簇需要分析**；
* 分段无热漂移（lat p50 [234,235,235,235,235,235]、confidence 稳定）。

## 15. overall verdict

**Phase 12.0-C = PASS**

**没有必须修复的 runtime 安全问题**：

1. 无任何真实输入 / action 调用 / 窗口控制（全程 `action_enabled=false` + `dry_run`）；
2. 失焦 / 最小化 / 尺寸变化 / 长跑均**无崩溃、无 stale target、无 fallback 风暴、无 state flap**；
3. 前台门控与快照失效按设计工作，恢复均 ≤1 轮；
4. **12.0-A 缺口①（运行期 bbox 观测）正式关闭**：`bbox samples = 7200`、**`invalid = 0`**；
5. **12.0-A 缺口②（RSS 不可用）已解决**：`psutil` 提供了 `start/peak/end/samples`（但见缺口 §16-④）。

## 16. unresolved issues

| # | 类型 | 描述 | 影响 | 处理 |
|---|---|---|---|---|
| ① | **observability_gap** | `yard_fields_leak_into_skipped_cycles`：`status=skipped`（失焦/最小化）的轮次仍**携带上一轮**的 `yard_detected / yard_confidence / yard_template_id / yard_bbox_oob`（例：cycle 118 `status=skipped` 但 `yard_detected=true, conf=0.9217`） | `_yard_stats` 报 `detections = 600/600` 而真实有效轮为 577 ⇒ **识别率被高估** | 本阶段**未改代码**；分析一律改用 `status == "ok"` 作分母（E: 577/577、E3: 573/573、F: 7200/7200 不受影响）。`bbox.invalid` 不受影响（泄漏值恒 `False`，只会漏报不会虚报）。**建议 12.0-D 修** |
| ② | **observability_gap** | `match_calls_is_budget_not_actual`：`yard_match_calls = 20` 是**预算值**（candidates 4 × scales 5），D1 中实际 `matchTemplate` 调用为 **0**（全被尺寸检查跳过） | 无法区分"跑了但低分"与"根本没跑" | 本阶段未改；以 `detector_ms`（0.66 ms vs 25.7 ms）作旁证。**建议 12.0-D 修** |
| ③ | **observability_gap** | `unknown_reason` 不一致：E1 失焦为 `not_foreground`，E3 最小化为 `skipped`（同为"未采集"） | 细分原因不一致（`skip_reason` 字段本身精确） | 记录，**建议 12.0-D 统一** |
| ④ | **observability_gap** | `rss_sampler_keeps_no_timeseries`：`_RssSampler` 只保留 `start/peak/end/samples`，**不保留时间序列** ⇒ 无法计算前/中/后 10 分钟 median 与 slope MB/hour | 无法判定 warmup vs 持续增长 | **建议 12.0-D 加时序**（本阶段按纪律不改代码） |
| ⑤ | **known limitation**（非缺陷） | 生产模板为**固定像素源**，`roi_scale` 只降采样 ROI、**不缩放模板**；matcher 档位 0.6~0.9 仅覆盖同一客户区下的尺度抖动 | 窗口尺寸显著变化（D1 缩小 / D2 放大）时**无法识别** | **设计意图**（`FIT_TOLERANCE_PX=2` 只吸收取整误差）；跨分辨率需模板多尺度，属**未来阶段** |
| ⑥ | **known limitation**（非缺陷） | 大窗口下匹配图面积增大 ⇒ 单轮耗时超 500 ms 周期（D2: 891~1203 ms） | 吞吐减半；调度器 `[late] re-aligned` 正确降级 | 记录；**未改算法** |
| ⑦ | note | 游戏窗口 hwnd 在 D3 → E 之间由 `0x130ab0` 变为 `0x461048`（窗口被重建） | 不影响结论（`find(refresh=True)` 每次重新解析） | 记录 |

## 17. whether ready for Task integration

**可以进入 Task 接线的前置条件已满足**，但建议**先处理缺口 ①（`yard_fields_leak_into_skipped_cycles`）**，因为 Task 接线后会依赖 `yard_detected` / `yard_bbox_oob` 作为**决策输入**，而当前该字段在 skipped 轮**语义被污染**；以及缺口 ④（RSS 时序）若"长跑无内存增长"是 Task 接线的验收门槛，也需先补。

**建议顺序**：Phase 12.0-D（补 ①③④ 观测 + 明确 ② 语义）→ 再接线 Task。

> 其余结论：Runtime 在**尺寸变化 / 失焦 / 最小化 / 1 小时长跑**四类压力下**稳定性已验证**；识别率下降全部可归因于**用户主动离开生产工作点**（窗口尺寸）或**已知档位限制**，**不构成 detector failure**。
