# Phase 12.0-D-R — Runtime Observability Real-World Verification Report

> 目标：在真实 Windows《阴阳师》客户端中复验 Phase 12.0-D 新增的观测字段，
> **不修改视觉识别算法、生产参数或 Task/Action 架构**。
> 基线 commit：`5526ef8 feat(observability): close runtime observability gaps`
>
> **结论：未发现需要修复的缺陷；本轮零源代码改动**（仅新增本报告）。

---

## 1. 前置基线（已确认）

| 项 | 值 |
|---|---|
| HEAD | `5526ef8` |
| `git status` | 仅 ` M tools/build_annotation_review_pack.py`（历史无关改动，**保留、未混入提交**） |
| `configs/` / `dataset/` / `src/` / `tests/` | 本轮**无改动**（`git diff` 为空） |
| 生产 yard | **Proposed36（36 张）** · `threshold 0.85` · `roi_scale 0.75` · `prefilter thumb` · `top_k 4` · `max_templates 36` |
| matcher 档位 | `(0.600, 0.675, 0.750, 0.825, 0.900)` |
| 安全 | `action.enabled=false` · `backend=dry_run` · **无 `real_input`** |
| `yard_metrics` 生产默认 | **`false`**（全程只用运行期临时诊断开关 `config.model_copy`，生产配置未改） |
| `psutil` | 可用（`RssSampler` 正常工作） |
| 离线对照探针（1704×960 原图） | detected `True` · candidates `4` · budget `20` · **actual `12`** |

## 2. 真实运行环境

| 项 | 值 |
|---|---|
| 目标窗口 | `阴阳师-网易游戏` · hwnd **`0x70d52`** |
| 客户区 | **1704 × 960**（三个场景均未改变） |
| 采集后端 | `dxcam` |
| DPI / 缩放 | 150% |
| soak 工具 | `tools/run_runtime_soak.py`（未新造工具、未改代码） |
| 采集时间 | 2026-10-12 18:20 ~ 19:01（+08:00） |

---

## 3. 各场景实际采样与关键结果

### 3.1 Scene D-R-a（纯被动基线，300 s）

| 项 | 值 |
|---|---|
| **实际采样周期数** | **600**（`duration 299.998 s`，`errors 0`） |
| `status=ok` / skip | **ok 600 / skip 0** |
| **`yard_observed`（正确分母）** | **600** · `unobserved = 0` |
| `detections` / `detected_ratio` | **513 / 0.855** |
| `capture` | success 600 · failures 0 · `skip_reasons {'(none)': 600}` |
| **理论 budget** | **{20: 600}** |
| **实际 actual** | **{12: 600}** ← **与 12.0-D 报告完全一致** |
| **派生 skipped** | **{8: 600}** · 违例 **0** |
| `bbox` | **invalid 0 / samples 513**（`{False: 513, None: 87}`） |
| `fallback` / `candidates` / `templates` / `prefilter` | `{False:600}` / `{4:600}` / `{36:600}` / `thumb 600` |
| `skip_reason` 分布 | `{None: 600}` |
| `unknown_reason` 分布（状态层） | `{None: 513, no_evidence: 87}` |
| `snapshot` | `processed 599` · **invalidations 0** |
| `state` / `stable_state` | `yard 513 / unknown 87`；`yard 544 / unknown 54 / pending 2`；**transitions 26 · flaps 0** |
| **Runtime latency** | p50 **266.0** · p90 313 · p95 328 · p99 344 · max 391 ms |
| `yard_ms` / `vision_ms` | p50 28.1 / 256.3 |
| **deadline miss** | **0** |
| **state flap** | **0** |
| confidence（513 命中） | min **0.850014** · p50 0.874935 · p90 0.884632 · p99 0.891084 · max 0.893798 |
| **action invocation** | **0** · `real_input=false` |

### 3.2 Scene D-R-b1（手动失焦，300 s）

| 项 | 值 |
|---|---|
| **实际采样周期数** | **600** |
| `status=ok` / skip | **ok 580 / skip 20** |
| 失焦窗口 | **cycle 168~187 = 20 轮 ≈ 10 s**（`18:50:00 → 18:50:09`） |
| **`yard_observed`（正确分母）** | **580** · `unobserved = 20` |
| `detections` / `detected_ratio` | **441 / 0.760345** |
| **理论 budget / 实际 actual / 派生 skipped** | **{20:580} / {12:580} / {8:580}** · 违例 **0** |
| **`skip_reason` 分布** | **`{'(none)': 580, 'not_foreground': 20}`** |
| **`unknown_reason` 分布** | **`{'(none)': 441, 'no_evidence': 139, 'not_foreground': 20}`** |
| 失焦轮残留证据字段 | **`leaked non-None: NONE (clean)`** —— `yard_detected` / `yard_confidence` / `yard_template_id` / `yard_bbox_oob` / `vision_ms` / `yard_ms` / `yard_candidates` / `yard_match_calls` / `yard_match_calls_actual` / `yard_prefilter` / `yard_total_templates` / `yard_fallback` **全部 `None`** |
| 失焦轮保留字段 | `frame_id` · `action_enabled=false` · `action_backend=dry_run` · `snapshot_status` · `snapshot_invalidated` |
| 失焦期间 | `state=None`（**无 stale target**）· `stable_state` 保持 `yard` · **action 0** |
| **snapshot invalidation** | `no_frame 20` · `invalidated` 19 → **20** |
| **恢复结果** | cycle **188**（失焦后第 1 轮）即 `ok` / `yard` / `cap_028` / conf 0.8601 / `budget 20 actual 12` ⇒ **立即重新观测，不复用过期数据** |
| `bbox` | invalid 0 / samples 441 |
| latency / deadline miss / flaps | p50 265.0 / **0** / **0** |

### 3.3 Scene D-R-b2（手动最小化，300 s）

| 项 | 值 |
|---|---|
| **实际采样周期数** | **600** |
| `status=ok` / skip | **ok 587 / skip 13** |
| 最小化窗口 | **cycle 175~187 = 13 轮**（`18:56:13 → 18:56:19`） |
| **`yard_observed`（正确分母）** | **587** · `unobserved = 13` |
| `detections` / `detected_ratio` | **554 / 0.943782** |
| **理论 budget / 实际 actual / 派生 skipped** | **{20:587} / {12:587} / {8:587}** · 违例 **0** |
| **`skip_reason` 分布** | **`{'(none)': 587, 'window_minimized': 13}`** |
| **`unknown_reason` 分布** | **`{'(none)': 554, 'no_evidence': 33, 'skipped': 13}`** ← 最小化在状态层仍是 **`skipped`**（**未被替换**成底层原因） |
| 最小化轮残留证据字段 | **`leaked non-None: NONE (clean)`** |
| `frame_ms`（最小化轮） | **0.048 ~ 0.149 ms**（1 个边界轮 1.766 ms）⇒ **未进入采集**（dxcam 不可能拿到黑帧） |
| **snapshot invalidation** | `no_frame 13` · `invalidated` **13** |
| **恢复结果** | cycle **188** 恢复采集（`capture_status=ok`）且该轮 `yard_detected=False`（**没有**沿用失焦前的 `True`）⇒ **不复用过期数据**；cycle **189** 恢复 `yard` / `cap_028` / conf 0.8882 |
| `state` / `stable_state` | `yard 554 / unknown 33 / None 13`；`yard 580 / unknown 18 / pending 2`；**transitions 10 · flaps 0** |
| `bbox_oob` | `{False: 554, None: 46}`（46 = 33 未命中 + 13 最小化）⇒ 语义正确 |
| latency / deadline miss / flaps | p50 250.0 / **0** / **0** |

### 3.4 `skip_reason` 对比（缺口③ 的核心验证）

| 场景 | `skip_reason`（唯一权威具体原因） | `unknown_reason`（状态层） |
|---|---|---|
| 手动失焦（b1） | **`not_foreground` × 20** | `not_foreground` × 20 |
| 手动最小化（b2） | **`window_minimized` × 13** | **`skipped` × 13** |

⇒ 两个场景在**具体原因**上**完全可区分**，而 `unknown_reason` **保持原有状态层语义、取值集合未扩展**。
这与 Phase 12.0-C 中"两者被抹平成同一个值"的问题形成直接对照 —— **缺口③ 已关闭**。

## 4. RSS 时序验证

| 项 | 场景 a | 场景 b1 | 场景 b2 |
|---|---|---|---|
| `rss_start` / `peak` / `end` | 197.99 / 255.63 / 241.89 MB | 200.27 / 255.48 / 245.49 MB | 201.16 / 256.54 / 246.29 MB |
| `rss_samples` | **299** | **299** | **299** |
| `rss_windows.available` | **`false`** | **`false`** | **`false`** |
| `rss_windows.note` | "运行时长 300.0s < 600s，无法切分前/中/后 10 分钟窗口" | 同左 | 同左 |
| `rss_trend.verdict` | **`insufficient_data`** | **`insufficient_data`** | **`insufficient_data`** |

**独立采样探针（8 s、interval 0.5 s）**：17 个采样点 · `t[0]=0.0` · `t[-1]=8.0` ·
**时间戳严格单调递增** ✅ · 两序列等长 ✅ · RSS 恒正 ✅ · 窗口与趋势同样如实标注 `insufficient`。

⇒ **采样点正常产生、时间戳单调、统计逻辑有效** —— 三项均已确认 ✅
⇒ 按纪律：**300 s 不足以生成三个独立的 10 分钟窗口，已明确标记 `insufficient_duration`（`rss_windows.available=false`），未宣称长期 RSS 趋势通过** ✅

**完整长期 RSS 趋势验证**需单独进行，至少采集覆盖前/中/后三个 10 分钟窗口（建议 ≥ 35 分钟，理想 60 分钟）。

## 5. 汇总验收表

| 验收项 | 结果 |
|---|---|
| 每场景采样周期数 | **600 / 600 / 600** |
| `status=ok` vs skip | 600/0 · 580/20 · 587/13 |
| `yard_observed` 正确分母 | **600 / 580 / 587**（**均不含跳过轮**）✅ |
| 理论 / 实际 / 跳过 match calls | **20 / 12 / 8**（三场景一致，**零违例**）✅ |
| `skip_reason` 分布 | `{(none):600}` · `{not_foreground:20}` · `{window_minimized:13}` ✅ |
| `unknown_reason` 分布 | `{no_evidence:87}` · `{no_evidence:139, not_foreground:20}` · `{no_evidence:33, skipped:13}` ✅ |
| 跳过轮残留 yard 证据 | **零残留**（两场景均 `NONE (clean)`）✅ |
| snapshot invalidation / 恢复 | 0 / **20** / **13**；恢复均 **≤1~2 轮**且**不复用过期数据** ✅ |
| RSS 采样与时序 | 299 采样点 · 时间戳严格单调 · `insufficient_data`（**未宣称通过**）✅ |
| Runtime latency（p50） | 266.0 / 265.0 / 250.0 ms |
| deadline miss | **0 / 0 / 0** ✅ |
| state flap | **0 / 0 / 0** ✅ |
| **action invocation** | **0** ✅ |
| **real_input** | **false** ✅ |
| `errors` | **0 / 0 / 0** ✅ |
| pytest | **1914 passed, 1 skipped** |
| `check_dataset` / `raw_manifest` / `check_annotations` | **PASS** / **`ok: true`**（178/178 + 29/29）/ **PASS** |

## 6. 判定

| 场景 | 判定 |
|---|---|
| Scene D-R-a（被动基线） | **PASS** |
| Scene D-R-b1（失焦） | **PASS** |
| Scene D-R-b2（最小化） | **PASS** |
| RSS 时序（短时） | **PASS（限定）** —— 仅验证采样/单调/统计逻辑；**长期趋势未验证** |

**Phase 12.0-D-R 总体判定：PASS（未发现缺陷，零源代码改动）**

## 7. 未覆盖 / 受限项（如实列出）

| # | 项 | 说明 |
|---|---|---|
| ① | **长期 RSS 趋势** | 本轮每场景仅 300 s ⇒ `insufficient_data`。需 ≥ 35 分钟（最好 60 分钟）单独复验，覆盖前/中/后三个 10 分钟窗口。 |
| ② | **置信度整体偏低**（非缺陷） | 本轮 p50 **0.8749** / min **0.8500**，明显低于 12.0-C F 段的 p50 0.9191 / min 0.8729；同一模板与阈值 ⇒ 属**当前庭院画面差异**（摆件/时段/NPC/其它玩家）。未命中 87/139/33 轮呈**零散小簇（1~5 轮）**、`flaps=0`，非系统性失败。**按要求未调整任何参数**。 |
| ③ | `rss_trend.note` 为空（**轻微瑕疵**） | 当采样点足够但 warmup 后点不足时，`verdict=insufficient_data` 正确，但 `note` 为空字符串（缺原因说明）。**本轮未修**（按"优先只交付证据与报告"）。 |
| ④ | `capture_failed` / `backend_unavailable` / `window_not_found` 场景 | 本轮未触发（只覆盖了 `not_foreground` 与 `window_minimized` 两种）。 |
| ⑤ | 跨分辨率 | 本轮固定在 1704×960（生产工作点），未涉及窗口尺寸变化。 |
| ⑥ | 单设备 / 单 DPI / 单客户端 | 结论不构成跨分辨率或跨设备泛化验证。 |
