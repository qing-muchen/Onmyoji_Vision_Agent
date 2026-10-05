# Phase 12.0-D — Observability Gap Closure Report

> 目标：关闭 Phase 12.0-C 记录的 4 个 `observability_gap`，**只补观测**，
> **不改变任何判定逻辑**（threshold / roi_scale / 模板集合 / prefilter / top_k /
> matcher 判定 / `FIT_TOLERANCE_PX` / StateStabilizer / Task 策略全部未动；
> 不启用真实输入，不自动控制游戏）。
>
> 起点 commit：`b7d0e80 docs(runtime): document scene d-f runtime soak validation`（12.0-C）

---

## 1. 4 个缺口与采用的修法

| # | 缺口 | 采纳的修法 |
|---|---|---|
| **①** | `yard_fields_leak_into_skipped_cycles`：跳过轮仍携带上一轮 yard 观测 ⇒ 识别率被高估 | **`_yard_observation(produced=...)`**：证据类观测只在**本轮真的产出**时写入；统计分母改为"真正产生过 yard 观测的轮次" |
| **②** | `match_calls_is_budget_not_actual`：`yard_match_calls` 只是预算 | **保留预算字段语义不变**，新增 **`yard_match_calls_actual`**（真正执行 `cv2.matchTemplate` 的次数）+ 派生 `skipped_match_calls` |
| **③** | `unknown_reason` 语义不统一（失焦=`not_foreground`、最小化=`skipped`） | **不扩展枚举、不改取值集合**：`unknown_reason` 保持**状态层**语义；**具体原因一律以 `skip_reason` 为唯一权威**，soak 统计改用 `skip_reason` |
| **④** | `rss_sampler_keeps_no_timeseries`：只有 start/peak/end | **保留 RSS 时序**，派生前/中/后 10 分钟 median + MB/hour 斜率 + 趋势结论 |

## 2. 改动清单

| 文件 | 改动 |
|---|---|
| `src/vision/detector/template/matcher.py` | 新增 `self._match_calls` 与只读属性 `match_calls`；**只在** `_match_once()`（真正执行 `cv2.matchTemplate` 处）`+= 1` |
| `src/vision/evidence/base.py` | `best_match()` 用"调用前后增量"得出本轮 actual 并存入 `self._last_match_calls_actual`；`_prefilter_diagnostics()` 输出 `prefilter_match_calls_actual` |
| `src/runtime/state_monitor/metrics.py` | 新增 `yard_match_calls_actual: int \| None`，并列入 `_observability_items()`（紧跟 `yard_match_calls`） |
| `src/runtime/state_monitor/monitor.py` | `_yard_observation(*, produced: bool)`：跳过轮**不写** `vision_ms` / `yard_*`；`frame_id` / `action_*` / `snapshot_*` / `task_active` 照常 |
| `tools/run_runtime_soak.py` | `_RssSampler` 保留 `(t, rss)` 时序；新增 `_rss_window_stats()` / `_rss_trend()`；`measure_memory()` 输出 `rss_windows` / `rss_trend`；`_yard_stats()` 输出 `yard_observed` / `yard_unobserved` / `detected_ratio` / `match_calls_actual` / `skipped_match_calls`；summary 用 `capture.skip_reasons` + `diagnosis.unknown_reasons`；`print_summary` 输出上述全部 |
| `tests/test_phase120d_observability_gap_closure.py` | **新增 30 项测试** |

## 3. 关键发现（缺口②的价值立刻显现）

生产工作点（客户区 1704×960、匹配图 1278×144）下：

```
candidates = 4      scales = 5      budget = 20
actual = 12         （0.825 / 0.9 两档在 1278 宽匹配图上 overflow 129 / 257 ⇒ 每候选跳过 2 档）
skipped_match_calls = 8
```

⇒ 历史字段 `yard_match_calls = 20` **从未**表示"真的调用了 20 次 `cv2.matchTemplate`"。
Phase 12.0-C 的 D1（窗口缩小到 1156×651）会得到 **actual = 0 / budget = 20**，
`skipped_match_calls = 20` —— 这类"跑了但低分"与"根本没跑"的区分现在**可观测**。

> 该发现**不改变判定逻辑**，也**不修改** `FIT_TOLERANCE_PX` 或档位配置。

## 4. 各缺口的语义（写入代码注释与本文档，避免未来 consumer 混用）

* **`skip_reason`（唯一权威的具体原因）**：`not_started` / `window_not_found` /
  `window_minimized` / `not_foreground` / `backend_unavailable` / `capture_failed` / `no_frame`
  + replay 注入值。**所有具体原因统计（capture / 前台 / 最小化 / 采集失败）只看它。**
* **`unknown_reason`（状态层语义，取值集合未变）**：`no_evidence` / `unsupported_scene` /
  `detector_failure` / `not_foreground` / `skipped`。它回答的是"**为什么这一轮是 unknown**"，
  **不是**"采集为什么没成功"；因此 `window_minimized` 在状态层仍是 `skipped`。
  ⇒ 也**不复制**底层 Runtime 原因。
* **`yard_detected` 等证据字段**：只有 `status=ok`（真的采到帧并判定过）的轮次才有值；
  跳过轮一律 `None`。⇒ 识别率的分母是 `yard_observed`，**不是**总轮次。
* **`yard_match_calls` vs `yard_match_calls_actual`**：前者是**预算**（candidate × scales），
  后者是**真实调用**；`skipped_match_calls = 前者 - 后者`。

## 5. 向后兼容

| 项 | 状态 |
|---|---|
| `yard_metrics=False`（生产默认） | `to_dict()` 仍**不含**任何 `yard_*` / `vision_ms` 键 ⇒ **无新增序列化开销**（单测固化） |
| `yard_match_calls` 字段名 / 取值 / 历史语义 | **未改** |
| `UnknownReason` 取值集合与 `resolve_unknown_reason()` 行为 | **未改**（单测固化） |
| `yard_bbox_oob`（12.0-B）语义 | 未改 |
| `best_match()` / `_restore_rect()` / `detect()` 调用路径 | 未改（`detect()` 仍必经 `best_match()`） |
| 判定参数 | `threshold 0.85` · `roi_scale 0.75` · `thumb` · `top_k 4` · 36 张 · `FIT_TOLERANCE_PX 2` **全部未变** |

## 6. 测试（+30）

`tests/test_phase120d_observability_gap_closure.py`：

* **①**：端到端 replay（注入采集失败）⇒ 跳过轮无 yard 字段、正常轮仍有；跳过轮仍留精确 `skip_reason`；
  `_yard_stats` 的 `yard_observed` / `detected_ratio` / 全未采集时 `detected_ratio is None`；
  summary 的 `detected_ratio` 与逐轮一致；
* **②**：新字段与顺序；overflow 超容差**不**计数；bounded fit 后**计数**（构造 401×201 模板 ⇒ 0.6 档溢出 1 px）；
  生产工作点 `actual(12) < budget(20)`；soak 派生 `skipped_match_calls`；
* **③**：`UnknownReason` 取值集合未扩展；`resolve_unknown_reason` 行为不变；
  失焦 vs 最小化在 `skip_reason` 上可区分；summary 的 `capture.skip_reasons`（权威）与
  `diagnosis.unknown_reasons`（状态层）分离；
* **④**：时序保留（含无 psutil 分支）；前/中/后 10 分钟 median；时长不足时如实标注；
  `stabilized_after_warmup` / `possible_memory_growth` / `insufficient_data` 三种结论；
  `measure_memory` 可 JSON 序列化；
* **纪律**：无观测时不输出 yard 键；`yard_metrics` 默认仍 `False`；判定参数未动；
  matcher 计数是**纯 `+= 1`**（AST 校验只有一处、且不出现在条件里）；monitor 仍只读 metadata。

## 7. 验收

| 项 | 结果 |
|---|---|
| `python -m pytest -q` | **1914 passed, 1 skipped**（1884 + 30 新增） |
| `python tools/check_dataset.py` | **PASS**（raw 178 / `yard_live` 29） |
| `python tools/raw_manifest.py` | **`ok: true`**（178/178 + 29/29，missing/added/changed 空） |
| `python tools/check_annotations.py` | **PASS** |
| `configs/` / `dataset/` | **未改**（`git diff` 为空） |
| 生产配置 | `36` / `0.85` / `0.75` / `thumb` / `4` / `yard_metrics=False` / `action=false+dry_run` |

## 8. 明确未做

* 未修改 threshold / roi_scale / 模板集合 / prefilter / top_k / matcher 判定 /
  `FIT_TOLERANCE_PX` / StateStabilizer / Task 策略；
* 未扩展 `UnknownReason` 枚举、未改其取值集合与序列化；
* 未把 `psutil` 加入运行时依赖；
* **未**在运行期运行过 soak 验证新字段（本轮只做离线/单元与 replay 验证）；
  真实运行期复验见 §9。

## 9. 下一步

1. **（建议）一次短的真实 soak（约 5 分钟）复验新观测**：确认 `bbox samples=invalid=0`、
   `yard_observed/detected_ratio`、`match_calls_actual`、`skip_reasons`、`rss_windows/rss_trend`
   在真实运行期都如预期出现（含一次手动失焦，验证 `skip_reasons` 把 `not_foreground`
   与 `window_minimized` 分开统计）。
2. 通过后再进入 **Task 真正接线**（当前 `task_active` 恒为 `None`）。
