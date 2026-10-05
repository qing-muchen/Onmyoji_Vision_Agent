# Phase 12.0-B — Runtime Observability Completion Report

> 目标：补齐 Phase 12.0-A 暴露的**两个观测缺口**（① 运行期无 bbox 观测；② 无 `psutil` ⇒ 无 RSS 趋势）。
> **只增加观测**：不改 threshold / roi_scale / prefilter / top_k / matcher / `FIT_TOLERANCE` /
> 判定流程 / 安全开关；`configs/config.yaml` **未修改**。

---

## 1. 背景：12.0-A 的两个缺口

| 缺口 | 12.0-A 的表现 | 影响 |
| --- | --- | --- |
| **运行期无 bbox 观测** | `CycleMetrics` 没有 bbox 字段 ⇒ soak 无法判定"bbox invalid" | 验收标准 "bbox invalid = 0" 无法在运行期判定，只能引用离线结论 |
| **无 `psutil` ⇒ RSS `not_available`** | 只有 `tracemalloc` 的 Python 分配量 | 无法给出 RSS start/peak/end，也无法证明"无内存增长" |

## 2. 改动清单（全部为"新增观测"）

| 文件 | 改动 |
| --- | --- |
| `src/vision/evidence/base.py` | 新增 **`_restore_rect_ex()`**（返回 rect + 「是否被 clamp 修正」）与 **`_match_and_restore()`**（与 `best_match` 同路径 + 透出标记）；`_restore_rect()` **API 不变**；`detect()` 在 **clamp 触发时**向 metadata 写 `bbox_clamped = True` |
| `src/runtime/state_monitor/metrics.py` | `CycleMetrics` 新增 **`yard_bbox_oob: bool \| None`**，并追加到 `_observability_items()`（`None` 不输出） |
| `src/runtime/state_monitor/monitor.py` | `_yard_observation()` 在**命中**时注入 `yard_bbox_oob`（未命中保持未设置） |
| `tools/run_runtime_soak.py` | 新增 `_rss_bytes()` + `_RssSampler`（后台 1s 采样）；`measure_memory()` 支持 RSS；`_yard_stats()` 聚合 `bbox`；`print_summary` 增加 bbox 行并动态显示 RSS |
| `requirements-dev.txt` | 新增**可选**依赖 `psutil>=5.9`（**不入** `requirements.txt`） |

## 3. bbox 观测设计

* **语义**：`clamp` 被触发 ⟺ "若不修正，bbox 本会越出 ROI" ⇒ 这正是"bbox invalid"的运行期等价观测；
* **判定位置**：`_restore_rect_ex()` 比较 clamp **前**的原始四边与 clamp **后**的结果，**完全由输入决定**（**无实例状态**、无共享可变状态 ⇒ 不影响线程安全）；
* **写入纪律**：只在**触发时**写 `metadata["bbox_clamped"] = True`（未触发不写 ⇒ `roi_scale=1.0` 等未触发路径的 metadata 与既有版本**逐字段一致**）；
* **注入纪律**：runtime 只在 `yard_detected=True` 时注入 `yard_bbox_oob`；**未命中保持未设置**（`None`）—— 避免把"没有 bbox"误读成"bbox 合法"；
* **聚合**：`_yard_stats()` 的 `bbox.samples` 只统计**命中**轮次，`bbox.invalid = bbox.clamped`（计数）。
* **无跨层 import**：runtime 只读 evidence 的 `metadata` 键，不 import vision 内部实现。

## 4. RSS 观测设计

* `psutil` 为**可选依赖**：缺失时 `_rss_bytes()` 返回 `None`，`_RssSampler.start()` **不启动线程**（零开销），`measure_memory()` 的 `rss_note` 明确写 `not_available`；
* 存在时：后台守护线程按 **1 s** 间隔采样 `Process().memory_info().rss`，`stop()` 返回 `start / peak / end / samples`；
* **绝不**用 Python 分配量冒充常驻内存；采样失败（异常）时静默降级为 `None`，不影响 soak 主流程。

## 5. 向后兼容契约（关键）

* **`yard_metrics = False`（生产默认）时**：`CycleMetrics.to_dict()` **不含**任何 yard / 观测键 ——
  由单测逐字段固化（`test_metrics_output_omits_observability_when_not_collected`）；
* **`detect()` 仍必须经由 `best_match()`** —— 该调用路径是既有契约（ROI 结果缓存、子类覆写、
  以及既有测试的 spy 都依赖它）。因此 clamp 标记以 **`_last_bbox_clamped` + 只读属性
  `last_bbox_clamped`** 暴露（与既有 `_last_prefilter` 完全同风格），**而不是**让 `detect()`
  改走另一条内部路径。该路径由 `tests/test_phase116_template_cache.py` 的 `best_match` spy
  测试固化 —— 实现过程中确实先踩中、再修正。
* `best_match()` 签名与返回**不变**（`(TemplateMatch | None, Rect)`），`_restore_rect()` 签名与返回**不变**；
* `detect()` 的 `EvidenceResult` 结构不变，仅**可能**多一个 `metadata` 键（且仅在被修正时）；
* 生产 `runtime.state_monitor.yard_metrics` 默认值仍为 **`False`**（单测固化）。

## 6. 测试

新增 `tests/test_phase120b_runtime_observability_completion.py`（**26 项**），覆盖：

1~5. bbox 观测：内部矩形不触发 clamp / 越界必触发且被 clamp / `_restore_rect` API 与 `_restore_rect_ex` 一致 /
`_match_and_restore` 与 `best_match` 判定与 bbox 完全一致 / `detect()` 只在触发时写 `bbox_clamped`；
6~8. `CycleMetrics` 新字段默认 `None`、未采集时不输出任何观测键、采集时按固定顺序输出；
9~11. monitor 仅命中时注入、数据来源为 `bbox_clamped`（**无跨层 import**）、vision 写入键与 runtime 读取键**契约对齐**；
12~16. RSS：有 `psutil` 返回真实字节数 / 无 `psutil` 返回 `None`（monkeypatch 模拟）/ 采样器无 `psutil` 时 no-op /
`measure_memory` 的 `not_available` 分支 / 有 `psutil` 时 start·peak·end 可读；
17~19. soak 聚合：`bbox` 只统计命中轮次 / 全未命中时样本为 0 / 摘要打印对缺键与 `not_available` 都能工作；
20~26. 纪律与不变性：汇总含新键且可 JSON 序列化 / 观测不改判定（threshold·roi_scale·prefilter·top_k·36 张不变，
且源码不含改这些常量的语句）/ `yard_metrics` 默认仍 `False` / 安全开关不变 / 不引入输入控制依赖
（`src.action.resolver` 只读定位**允许**，`ActionDispatcher`·`InputBackend`·点击·按键一律禁止）/ `psutil` **只**在
开发依赖里（运行时依赖不得包含）/ 采样器只汇报真实读数。

## 7. 验收

| 命令 | 结果 |
| --- | --- |
| `python -m pytest -q` | **1884 passed, 1 skipped**（1858 基线 + 26 新增） || `python tools/check_dataset.py` | PASS |
| `python tools/raw_manifest.py` | `ok: true`（历史 178/178 + live 29/29，`missing/added/changed` 空） |
| `python tools/check_annotations.py` | PASS |

生产不变性：`template_include = 36`（Proposed36）· `threshold 0.85` · `roi_scale 0.75` ·
`prefilter thumb` · `top_k 4` · `action.enabled=false` · `backend=dry_run` —— 全部未变。
`configs/config.yaml` **未修改**。

## 8. 明确未做

* 未修改任何判定逻辑、阈值、ROI 缩放、预筛、matcher、`FIT_TOLERANCE_PX`；
* 未新增模板、未重选模板、未触碰 `dataset/**`；
* 未把 `psutil` 加入运行时依赖；
* 未修改 `CycleMetrics` 既有字段的语义或输出顺序（只在 `yard_fallback_reason` **之后**插入新键）。

## 9. 下一步

1. **Phase 12.0-A 复跑**（Scene A/B′/C + 30 分钟长跑）以取得：**RSS start/peak/end 趋势**与
   **运行期 `bbox.invalid`**，从而补齐 12.0-A 验收标准的最后一项；
2. 之后进入 **Scene D–F 扩展与更长 soak（≥1 h）**；
3. 最后进入 **Task 真正接线**（当前 `task_active` 恒为 `None`）。
