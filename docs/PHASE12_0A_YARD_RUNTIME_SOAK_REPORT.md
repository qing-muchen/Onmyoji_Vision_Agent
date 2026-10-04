# Phase 12.0-A — Yard Runtime Soak Validation Report

> 目标：验证真实运行链路 **Capture → VisionBridge → YardDetector → StateStabilizer →
> RuntimeObservation** 的长期运行可靠性。
> **本阶段不进行任何识别优化**，也**未修改任何代码 / 生产配置**（`configs/config.yaml` 未动，
> `threshold` / `roi_scale` / `template_include` / matcher / prefilter / `FIT_TOLERANCE` 全部保持
> 11.9-C 切换后的状态）。

---

## 1. commit baseline

* **`63f5998 feat(vision): switch yard production templates to proposed36`**（HEAD）
* 本阶段**无代码改动**；`tools/build_annotation_review_pack.py` 仍为历史未提交 —— **未修改 / 未 stage / 未 commit**。

## 2. runtime configuration

| 项 | 值 |
| --- | --- |
| 模板集合 | **Proposed36**（`freeze_hash 7851a45ebc5f9252c9bd4cca38c8063298bf87cef0341513b9aa8f92eabb00c5`） |
| threshold / roi_scale | **0.85 / 0.75**（未改） |
| template_prefilter / top_k | **thumb / 4**（未改） |
| `runtime.state_monitor.yard_metrics` | 生产默认 **`False`**（未改）；soak 以 `config.model_copy(update=…)` **内存开启** |
| 观测入口 | **`tools/run_runtime_soak.py`**（复用既有工具，未新造；`--mode real --duration … --record-metrics`） |
| cycle interval / deadline budget | 500 ms / 500 ms |
| 前置就绪门控 | 外层只读轮询"目标窗口成为前台"后再启动 soak（soak 自身在启动瞬间判定前台，非前台即 `not_run`、不重试、不伪造） |

## 3. window information

* 目标窗口 **`阴阳师-网易游戏`**，hwnd **`0x6807de`**，客户区 **1704×960**，未最小化；
* 采集后端 `dxcam`；前台门控启用（**不抢占焦点、不激活窗口**）。

## 4. scene coverage

| 场景 | 定义 | 时长 | cycles | 人为操作 |
| --- | --- | --- | --- | --- |
| **A** | 静态庭院 | 300.2 s | 600 | 无 |
| B（**作废重跑**） | 原定"自然动态"，实际混入 UI 点击 / 角色移动 | 300.0 s | 600 | 有 |
| **C** | 轻度 UI 变化（+ 期间窗口切换） | 300.2 s | 593 | 有（UI） |
| **B′** | **严格被动观察**（只等动画 / NPC / 背景相位） | 300.0 s | 600 | 无 |
| **长跑** | 30 分钟 | **1800.0 s** | **3600** | **有**（走动 / 点 UI / 离开庭院） |

## 5. cycles

* 长跑 **3600**（≥ 3600 ✅ 达标）；四段短场景合计 1793；
* 全部运行 `errors = 0`。

## 6. duration

* 长跑 **1800.003 s**；A 300.223 s · B 299.989 s · C 300.227 s · B′ 300.011 s。

## 7. latency

**长跑段（n = 3600）**：

| 指标 | cycle `latency_ms` | `vision_ms` | `yard_ms` |
| --- | --- | --- | --- |
| p50 | **234.0** | 233.5 | **25.9** |
| p90 | 266.0 | 268.3 | 33.5 |
| p95 | **296.0** | 286.7 | **36.6** |
| p99 | 328.0 | 321.3 | 43.4 |
| max | **406.0** | 400.1 | 49.7 |

**deadline misses = 0 / 3600（rate 0.0）** ✅ 达标（budget 500 ms，p50 仅占 47%）。

各段 deadline misses：A **4**（0.67%，overrun 16/93/32/47 ms）· B 0 · C **1**（0.17%，对应切窗口的 4218 ms）· B′ **0** · 长跑 **0**。所有 late cycle 均由工具 `re-aligned`，**无累积漂移**。

## 8. yard detection

| 段 | detected ratio | confidence min / p50 / p90 / p95 / max |
| --- | --- | --- |
| A（无操作） | **1.000**（600/600） | 0.8770 / 0.9200 / 0.9260 / 0.9270 / 0.9351 |
| B（有操作，作废） | 0.955（573/600） | 0.8500 / 0.9001 / 0.9121 / 0.9199 / 0.9340 |
| **B′（无操作）** | **1.000（600/600）** | 0.8507 / 0.8985 / 0.9044 / 0.9054 / 0.9158 |
| C（UI 被动） | **1.000（593/593）** | 0.8537 / 0.8981 / 0.9043 / 0.9059 / 0.9106 |
| **长跑（有操作）** | **0.9311（3352/3600）** | 0.8526 / 0.8999 / 0.9144 / 0.9205 / 0.9353 |

**长跑未检出 248 轮（6.9%）成 3 簇**：`23.0 s`（cycle 1199 起 46 轮）、`0.5 s`（1 轮）、`100.5 s`（cycle 3065 起 201 轮）。

**归因（已由操作者确认）**：长跑期间操作者**移动了角色 / 点击了 UI / 离开过庭院** ⇒ 上述未检出属**人为干预引起的画面变化**，**不是**链路或算法退化。

### 归因闭合表（本阶段最重要的结论）

| 人为操作 | detected ratio | flaps |
| --- | --- | --- |
| 无（静止 A / 被动动态 B′ / 被动 UI C） | **1.000** | 0 |
| 有（走动 / 点 UI / 离开庭院：B 与长跑） | 0.931 ~ 0.955 | **0** |

⇒ **只有人为操作（角色移动、UI 交互、离开场景）才会造成短暂成簇未检出**，且状态机**始终未判为抖动**（`flaps = 0`）。

## 9. template runtime

* **loaded templates = 36**（全程恒定）· **candidate_count = 4**（3600/3600）· **match_calls = 20**（3600/3600）· prefilter `thumb`（3600/3600）；
* **未出现 180 calls/frame** ✅；
* **fallback = 0（0%）** ✅ 达标（≤ 1%），reason 全为 `None`；`yard_prefilter_ms` p50 0.357 / p95 0.640 / max 1.206；
* 命中模板分布：`cap_028` **3191** + `cap_007` **161** + miss 248 ⇒ **多模板参与**（`cap_007` 承接 4.5% 命中）。

## 10. state stability

* 长跑：`yard 3352` / `unknown 246` / `pending 2`，**transitions = 4 · flaps = 0** ✅ 达标；
* 状态序列含 3 轮 `transition` 过渡态（正常过渡，非抖动）；
* 四段短场景（含混入 UI 的 B）flaps 均为 **0**。

## 11. capture recovery

* 长跑：**capture success 3600 / failures 0**、`snapshot invalidations = 0`（全程前台）；
* **Scene C 实测到恢复链路**：4 次 `not_foreground` skip + **3 次 `snapshot_invalidated`**（切窗口），期间 **0 次错误 action**、`transitions/flaps = 0`；恢复后 `capture ok 589`、`detections 593/593` ⇒ **中断期不产生错误动作、恢复后快照与 detector 均正常** ✅。

## 12. memory trend

⚠️ **测量缺口（如实记录）**：

* 本机**无 `psutil`** ⇒ **RSS 为 `not_available`**（`rss_start / rss_end / rss_peak` 均为 `null`）；
* 仅有 `tracemalloc` 的 Python 分配量：长跑 **current 47.69 MB / peak 57.60 MB**；
* 各段（均为**独立进程**）：current 32.42 / 32.45 / 32.46 / 48.44 / 47.69 MB；peak 42.33 / 42.32 / 42.34 / 58.31 / 57.60 MB ⇒ **区间稳定、无单调增长**，是"无显著泄漏"的**弱证据**；
* **不得**据此宣称"已证明无内存泄漏"。建议后续补 `psutil` 并增加进程中采样点。

## 13. safety verification

* 5993 条 metrics：`action_enabled = false`、`action_backend = dry_run`（**100%**）；
* **action invocation = 0**（本工具不调用 `ActionDispatcher` / `InputBackend`）；
* **real_input = false**；全程未激活 / 切换 / 抢占窗口，未发送任何输入；
* `bbox invalid`：⚠️ **运行期 `CycleMetrics` 不含 bbox 字段 ⇒ `not_available`**；仅能引用 **11.9-C 生产路径离线验证**：命中 82、**bbox 越界 0**、IoU p50/min 1.0、模板选择差异 0。

## 14. pytest

**1858 passed, 1 skipped**（= 基线；本阶段未改代码）

## 15. dataset integrity

* `dataset/raw/yard` 98 张 · `dataset/raw/yard_live` 29 张（s01 5 / s02 8 / s03 8 / s04 8）；
* `check_dataset` **PASS** · `raw_manifest` **`ok: true`**（历史 178/178 + live 29/29）· `check_annotations` **PASS**；
* `git status --short`：仅 ` M tools/build_annotation_review_pack.py`（历史未提交，未触碰）。

## 16. conclusion

**链路可靠性：通过。** 真实窗口下 30 分钟 / 3600 轮连续运行：`errors = 0`、**deadline miss = 0**、
**flaps = 0**、**fallback = 0**、模板契约恒定（**36 / 4 / 20**）、`action` 全程关闭、capture 无失败、
快照无失效；轻度 UI 被动观察下检出 **100%**、零误判。

**识别侧观察（非本阶段优化目标）：** 静止 / 被动动态 / 被动 UI 均 **100%** 检出；一旦出现
**人为操作**（角色移动、UI 交互、离开庭院）会出现**成簇短暂未检出**（长跑 6.9%，最长 100.5 s），
但 **`flaps` 始终为 0**，状态机未将其判为抖动。

**两个必须注明的缺口：** ① 运行期无 bbox 字段 ⇒ 越界只能引用离线结论；② 无 `psutil` ⇒ 无 RSS 趋势。

## 17. next phase recommendation

1. **补 `psutil`**（可选依赖）后重跑 30 分钟长跑，补齐 **RSS start / peak / end** 与进程中采样点；
2. 若需在运行期覆盖 **bbox invalid**，为 `CycleMetrics` 增加一个"bbox 是否越界"的**标量观测**
   （属"新增观测"，需单独排期，不得改判定逻辑）；
3. **长跑未检出段**（23.0 s / 100.5 s）已归因为人为操作；若要进一步区分"移动 vs UI vs 离开场景"，
   建议在 Phase 12.1 做**带时间戳的场景标注**（用户按时间窗标注操作类型）；
4. 生产已切 `proposed36` 且 soak 无退化 ⇒ 可推进：**Scene D–F 扩展**、**更长 soak（≥1 h）**、
   **真实任务接线**（当前 `task_active` 恒为 `None` ⇒ Task 未接线），以及
   **yard variant 覆盖策略**（`s01` 云景阆苑这一独立 domain 仍未被覆盖）。
