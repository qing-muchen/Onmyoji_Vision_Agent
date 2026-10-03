# Phase 11.9-B — Candidate-Only Yard Template Reselection and Blind-Holdout Preparation

> 本阶段**只从现有 61 张模板里重选 36 张**（不新增模板、不改 threshold / roi_scale / top_k、
> 不改 matcher、不改生产配置）。成功标准不是"根据已看过的 s02 选出一个对 s02 最好的集合"，
> 而是"只用 historical LOO + s03 candidate 重选 → **冻结** → **一次性**验证 s02 →
> 最终留出全新的 s04 作真正 blind holdout"。

---

## 1. 为什么 s02 降级为 validation

Phase 11.9-A 已经查看过 s02 的**逐帧 Production36 结果、逐帧 All61 结果、漏检帧的最佳模板**
（`cap_026` / `cap_020` / `cap_009`）。如果再用 s02 选模板、又拿 s02 当最终验收集，就会产生
**选择泄漏**。因此：

* s02 的角色从 `holdout` 改为 **`validation`**；
* 重选过程**完全不读** `dataset/raw/yard_live/yard_live_20261003_s02`；
* 只在集合**冻结之后**做**一次** validation，且结果**不得回流** selection；
* 真正 blind 的角色交给**新采集**的 `yard_live_20261003_s04`。

> 未物理改动原始图片，也未改 `capture_role` schema（避免为本阶段强行扩 schema）；
> `logical_role = validation` 只记录在本报告与 selection protocol 中。

## 2. selection 数据边界

| 用途 | 数据 | 说明 |
| --- | --- | --- |
| 允许（选择） | `historical98`（`dataset/raw/yard/*.png`） | **必须 source-aware LOO**：评估某张原图时排除同源派生模板 `bottom_menu_<stem>` |
| 允许（选择） | `s03 candidate`（`raw/yard_live/yard_live_20261003_s03/*.png`，8 张） | 当前主题适配数据；当前主题**没有**对应模板 ⇒ 不存在自匹配 |
| 允许（**仅安全约束**） | `canonical80`（`bench.collect_negatives()`） | 只用于 FP 必须为 0 与 negative risk 排序；**不**用于调 threshold |
| **禁止** | `s02`（validation） | 冻结前绝不读取 |
| **禁止** | `s01`（diagnostic，云景阆苑） | 与茨心乐园属不同 domain，混入会让目标失焦 |

实现层用 `assert_selection_only()` 显式拒绝任何 s02 / s01 路径，测试固化（含"选择样本分组恰好
等于 `{raw/yard, raw/yard_live/<s03>}`"）。

## 3. 当前 Production36 贡献分析

按"历史 LOO unique / candidate unique / shared / 负样本风险"逐模板统计（选取摘录）：

| 模板 | 历史 LOO unique | candidate unique | shared | negative risk | proxy 冗余 |
| --- | --- | --- | --- | --- | --- |
| `cap_006` | 1 | **7** | 3 | 0.2724 | — |
| `skin003_003` | 2 | 0 | 1 | 0.4614 | — |
| `skin007_001` | 2 | 0 | 6 | 0.5892 | — |
| `skin009_001` | 2 | 0 | 0 | **0.7793** | — |
| `cap_003` / `cap_007` | 1 | 0 | 0 / 3 | 0.5077 / 0.2689 | — |
| `skin003_001` | 0 | 0 | — | — | ✅ |
| `skin003_004` | 0 | 0 | — | — | ✅ |
| `skin006_001` | 0 | 0 | — | — | ✅ |
| `skin006_003` | 0 | 0 | — | — | ✅ |
| `skin006_005` | 0 | 0 | — | — | ✅ |

**proxy 冗余（unique 贡献为 0）的 5 张**：`skin003_001` / `skin003_004` / `skin006_001` /
`skin006_003` / `skin006_005` —— 这些位置就是"释放预算"的首选来源（**不是**按皮肤名判断，
而是按 unique 贡献）。

> 注意 `skin009_001` 的 negative risk **0.7793** 正好等于当前 negative max ⇒ 它是最"贴阈值"
> 的模板，但仍有 unique 贡献（2），因此未被判为冗余。

## 4~5. greedy 策略与 swap search

* **Stage 1（proxy）**：用低阈值探针构建 **template × sample 分数矩阵**
  （61 模板 × (98 historical + 8 candidate + 80 negative)），每格记录
  `confidence` / `detected_at_threshold` / `selected_scale` / `fit_applied`，
  历史样本与模板同源时标 `excluded_by_loo=true`（**该分数不得参与 objective**）。
  贪心按**词典序增益** `(Δcandidate 覆盖, Δhistorical 覆盖)` 选择，平手一律退回
  `template_id` 字典序 ⇒ 结果确定、可复现；若增益归零则用"负样本风险最低"的模板补齐到 36。
* **Stage 2（真实语义）**：用生产 `detect()`（`36 loaded → thumb prefilter → Top-4 →
  threshold 0.85`，`roi_scale=0.75`）复核；**最终指标一律取自 Stage 2**。
* **swap search**：1-for-1 替换，先用 proxy 过滤、只允许 proxy 有增益的替换进入 Stage 2
  （每轮取 proxy 最优的前 6 个），接受能使 `ranking_key` 提升的替换，最多 6 轮。

**本次结果**：proxy 贪心已直接达到 `(candidate 8, historical 58)`，**swap 轮数 = 0**
（贪心解即为局部最优，无替换被接受）⇒ 说明 swap 阶段没有"额外收益"，也没有引入过拟合式的
反复调整。

## 6. proposed36 完整列表（PROPOSED_36_FROZEN）

```
cap  (6) : cap_001  cap_003  cap_005  cap_007  cap_027  cap_028
skin (30): skin001_001 skin001_003 skin001_005 skin002_001 skin002_003 skin002_005
           skin003_001 skin003_003 skin003_005 skin004_001 skin004_003 skin004_005
           skin005_001 skin005_005 skin006_001 skin006_003 skin007_001 skin008_001
           skin008_003 skin008_005 skin009_001 skin010_001 skin011_001 skin012_001
           skin013_001 skin014_001 skin015_001 skin016_001 skin017_001 skin018_001
```

共 **36** 张（`cap` 6 + `skin` 30），全部带 `bottom_menu_yard_` 前缀。完整 stable id 列表见
`TEMP/ova119b/proposed_36_frozen.json` 的 `proposed_36` 字段；测试断言其数量为 36、无重复、
且全部存在于现有 61 张库内。

## 7. 与 current36 的 added / removed diff

| 变更 | 模板 |
| --- | --- |
| **added**（4） | `bottom_menu_yard_cap_001`、`bottom_menu_yard_cap_005`、`bottom_menu_yard_cap_027`、`bottom_menu_yard_cap_028` |
| **removed**（4） | `bottom_menu_yard_cap_006`、`bottom_menu_yard_skin003_004`、`bottom_menu_yard_skin006_005`、`bottom_menu_yard_skin007_003` |

> ⚠️ **算法没有选中 Phase 11.9-A 里"看起来有用"的 `cap_008` / `cap_026` / `cap_009`**。
> 这符合既定纪律（不硬编码、数据驱动），也说明在 **Stage 2 真实语义**（Top-4 + threshold）下
> 那几张并不比入选者更有用 —— 印证了"Stage 1 proxy 会高估"这一设计前提。

## 8. historical ordinary

| 集合 | TP | FN | 说明 |
| --- | --- | --- | --- |
| current36 | **81** / 98 | 17 | 与 Phase 11.7-C 基线一致 |
| **proposed36** | **82** / 98 | 16 | **+1** |

## 9. historical LOO（source-aware）

| 集合 | LOO TP | LOO FN |
| --- | --- | --- |
| current36 | **56** / 98 | 42 |
| **proposed36** | **58** / 98 | 40 |

⇒ **LOO 提升 2**，满足"不得退化"的硬约束，且是**真实泛化收益**（自匹配已被剔除）。

## 10. s03 candidate（训练内）

| 集合 | TP | FN | confidence min / p50 / p90 / max |
| --- | --- | --- | --- |
| current36 | 7 / 8 | 1 | 0.8542 / 0.8693 / 0.8794 / 0.8828 |
| **proposed36** | **8 / 8** | **0** | **0.8563 / 0.8837 / 0.9012 / 0.9019** |

⇒ 达到 All61 已证实的理论上限 **8/8**，且整体置信度上移。

## 11. canonical80（安全约束）

| 集合 | FP | TN | negative max | p95 | gap to 0.85 |
| --- | --- | --- | --- | --- | --- |
| current36 | **0** | 80 | **0.7793** | 0.7250 | **0.0707** |
| **proposed36** | **0** | 80 | **0.7793** | 0.7250 | **0.0707** |

⇒ FP 仍为 0，且 **negative max 与 p95 完全不变**（0.7793 / 0.7250）⇒ 新集合**没有**让任何
困难负样本更贴近阈值，安全余量未被侵蚀。

## 12. threshold margin

阈值 **0.85 未变**；上面 gap 0.0707 未变。`cap_028` 等新入选模板在 80 张负样本上的最高分
仍低于 0.78 —— 说明它们是"对庭院强、对非庭院弱"的模板，而非靠放宽判定获得收益。

## 13. Top-4 candidate recall

* 生产语义下 `candidate_count` 每帧恒为 **4**（= `top_k`），未出现 `candidate > 4`；
* Stage 2 逐帧 `detected` 与全量判定一致（`fallback = 0` 全程）；
* 因此在 36 张集合上，`recall@4` 与 `recall@全部` 在本轮数据上等价（无候选漏选）。

## 14. latency

**⚠️ 测量方法学更正（重要）**：单侧顺序测量（先 current 再 proposed，各 1 轮）会给出
**+48% 的虚假回归**——那是系统负载/热效应造成的顺序偏差。改用**交替测量**
（A/B/A/B 各 4 轮、共 744 样本、双方各预热 60 帧）后：

| 集合 | p50 | p90 | p95 | mean |
| --- | --- | --- | --- | --- |
| current36 | 53.25 ms | 59.27 | 61.92 | 53.37 |
| **proposed36** | **52.17 ms** | 57.46 | 59.65 | 52.39 |
| delta | **−2.0%** | −3.1% | −3.7% | −1.8% |

⇒ **无性能回归（略优）**。模板尺寸也不是因素（Top-4 候选像素比 1.001）。
`matchTemplate` 预算保持 **20 calls/frame**（4 候选 × 5 scales）。

## 15. freeze manifest / hash

```
freeze_hash = 7851a45ebc5f9252c9bd4cca38c8063298bf87cef0341513b9aa8f92eabb00c5
```

hash 输入包含：`algorithm = lexicographic-greedy+proxy-filtered-swap`、
`algorithm_version = 11.9b-1`、`seed = 1191`、`threshold = 0.85`、`roi_scale = 0.75`、
`top_k = 4`、`prefilter = thumb`、`template_count = 36`、全部 61 个库内模板 id、
36 个选中 id、以及**每个 selection / negative 样本的 SHA256**。冻结清单写入
`TEMP/ova119b/proposed_36_frozen.json`（`status = PROPOSED_36_FROZEN`），
**未写入 `configs/config.yaml`**。

## 16. s02 一次性 validation（冻结后）

```
count = 8 · TP = 5 · FN = 3 · recall = 0.625
confidence min / p50 / p90 / max = 0.8555 / 0.8850 / 0.9187 / 0.9362
```

逐帧：

| 帧 | detected | confidence | selected template |
| --- | --- | --- | --- |
| `s02_001` | ✅ | 0.9362 | `cap_028` |
| `s02_002` | ✅ | 0.8850 | `cap_028` |
| `s02_003` | ❌ | 0.0 | — |
| `s02_004` | ❌ | 0.0 | — |
| `s02_005` | ✅ | 0.8826 | `cap_028` |
| `s02_006` | ✅ | 0.8555 | `cap_028` |
| `s02_007` | ✅ | 0.8925 | `cap_028` |
| `s02_008` | ❌ | 0.0 | — |

**关键观察**：命中的 5 张**全部**由**新入选**的 `cap_028` 命中（current36 下是 `cap_006`），
3 个 FN 的 `candidate_count` 均为 4、`fallback = False` ⇒ 属于"分数不足"，**不是**候选漏选。

## 17. s02 结果是否支持泛化

**部分支持，但不足以宣称跨会话收益**：

* validation recall 与 current36 **相同**（5/8 → 5/8），**未改善**；
* 但命中质量提升（同一批帧的置信度区间上移到 0.8555~0.9362，且由新模板命中）；
* 3 个 FN 中 `s02_004` 是**已知 residual**（All61 也只有 0.7255）；另 2 张（`003` / `008`）
  在 All61 下可过（0.9345 / 0.8987），其最佳模板是**未入选**的 `cap_026` / `cap_020`
  ⇒ 说明 **只有 8 张 candidate 支撑的选择无法覆盖全部跨会话变化**；
* 因此**不能**据此宣称"生产切换会改善真实场景" ⇒ 必须等 **s04 blind holdout**。

⇒ 结论：**训练侧达标（可冻结）**，但"跨会话收益"仍**未证实**。这是本阶段最重要的诚实结论。

## 18. known residual gap

* **`s02_004`**：`existing-library-gap` 候选样本（All61 best ≈ **0.7255** < 0.85）。本阶段
  **不**从它生成模板、**不**因它改变 selection，保留为 **known residual validation gap**；
* `s02_003` / `s02_008`（100% 属于**当前 36/61 内的选择问题**，因 All61 可过）⇒ 属于
  "candidate 覆盖不足"，应由 **Phase 11.9-C**（在 s04 之后、结合新 candidate 采集）处理；
* 结论：**主要缺口是 production-selection-gap（已被本阶段部分解决），残余是局部覆盖缺口**。

## 19. s01 domain shift 隔离

`s01`（云景阆苑）Production36 recall **0**、All61 ≈ **0.2** ⇒ 同一套模板对不同 yard 皮肤的
覆盖差异巨大。本阶段**严格未**把 s01 混入 objective（`assert_selection_only` 显式拒绝），
s01 保持 `diagnostic only`；yard variant 覆盖策略留作**独立课题**，不在 11.9-B 解决。

## 20. blind s04 protocol（下一步，本阶段**未**执行）

```
session_id : yard_live_20261003_s04
role       : holdout
variant    : 茨心乐园
目标       : 约 8 个独立视觉状态
```

要求：新 session（用户**重新离开并进入庭院**）、SHA 与现有 live 全部不同、
`source leakage = 0`、dHash 近重复**只作 warning 不作 hard fail**、**不要求**人为制造极端构图。

**纪律**：
1. s04 采集完成后，**第一次**正式 probe **只允许**运行 `Frozen Proposed36`，输出
   TP/FN / confidence / fallback / candidate_count / bbox；
2. 只有这次 blind 结果**完整保存**之后，才允许为**诊断**运行 All61；
3. s04 采集前 `PROPOSED_36_FROZEN` 不再改动。

## 是否建议进入 Phase 11.9-C

**建议进入，但顺序必须是**：先采 s04 → 跑一次 Frozen Proposed36（blind）→ 再决定是否把
proposed36 写入生产配置。理由：训练侧硬约束全部满足（LOO 56→58、candidate 7/8→8/8、FP=0、
calls=20、fallback=0、latency 无回归），但**跨会话收益尚未证实**（s02 validation 未改善）
⇒ 生产切换需要 blind holdout 背书。**本阶段不改 `configs/config.yaml`。**

---

## 附：本阶段新增/改动文件

| 类型 | 文件 |
| --- | --- |
| 新工具 | `tools/reselect_yard_templates.py`（数据边界 + Stage0 矩阵 + Stage1 贪心 + Stage2 生产语义 + swap + freeze + 一次性 validation） |
| 新测试 | `tests/test_phase119b_yard_template_reselection.py`（**37 项**） |
| 文档 | 本文件 + README / ARCHITECTURE 同步 |
| **未改动** | `configs/config.yaml`（生产仍为 current36）、`dataset/raw/**`、`features/.../positive`、`positive_sources.json` |
