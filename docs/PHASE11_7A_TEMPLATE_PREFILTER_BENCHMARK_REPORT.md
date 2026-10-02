# Phase 11.7-A — yard 模板候选预筛选 benchmark 报告

> 本阶段**只做 benchmark**：不接入生产、不改 `threshold`、不加模板、不改
> action / resolver / safety、不修改 `configs/config.yaml` 与 `src/vision/evidence/base.py`。
> commit：``test(vision): benchmark yard template candidate prefilter``

**核心问题**：能否在不损失"33 张模板覆盖收益"与安全性的前提下，只对少量候选执行
现有 `cv2.matchTemplate` 精匹配，从而把 33 张模板规模重新压回 500ms 实时预算内？

**答案**：**能**。用"32×4 灰度缩略图 L1 距离 + Top-4 候选"，判定与 33 张 naive 全匹配
**逐图完全一致**（含 LOO），而单项耗时从 **188.3ms 降到 23.3ms（8.1×）**，pre-filter
自身仅 **0.288ms**。

---

## 0. 一句话结论（先读）

| 指标 | Production 12 | **Naive 33** | **33 + thumb Top-4** |
| --- | --- | --- | --- |
| TP / FN（98 正样本） | 41 / 57 | **73 / 25** | **73 / 25** |
| TP（LOO 泛化） | 29 / 98 | **50 / 98** | **50 / 98** |
| FP / TN（80 负样本） | 0 / 80 | 0 / 80 | **0 / 80** |
| candidate recall@4 | — | — | **1.0000（73/73）** |
| 判定翻转 vs Naive 33 | — | — | **0** |
| bbox 选中模板差异 | — | — | **0** |
| bbox 越界 | 0 | 0 | **0** |
| matchTemplate 调用/帧 | 60 | 165 | **20（−88%）** |
| 单项耗时 p50 | 71.2ms | 188.3ms | **23.3ms（8.1×）** |
| 推算整链（+282ms 基线） | 353ms | 470ms | **305ms** |

---

## 1. 当前复杂度分析（源码 + 实测）

### 1.1 真实瓶颈：`cv2.matchTemplate`（占 99.8%）

实测拆分（98 正样本，33 模板）：

| 阶段 | p50 | 占比 |
| --- | --- | --- |
| ROI 准备（`crop_roi` 切片 + `roi_scale=0.75` 降采样 + `to_grayscale`） | **0.27ms** | **0.14%** |
| 33 张模板 × 5 档精匹配 | **186.74ms** | **99.86%** |
| 合计 | 186.98ms | 100% |

⇒ **单模板精匹配 ≈ 5.66ms**；模板数与耗时**线性**（12 张 ≈ 70.9ms ⇒ 5.91ms/张）。

复杂度：每帧 `cv2.matchTemplate` 调用数 = **模板数 × 有效档位数**（yard 是 5 档）：
Production 12 → **60 次**；Naive 33 → **165 次**。单次代价 ≈ `O((W−w+1)(H−h+1)·w·h)`，
而 yard 的模板几乎与 ROI 同大（`ROI 192×1706`、模板 `192×1704/1706`），位置搜索维度
已退化为 1 ⇒ 单次代价 ≈ **模板像素数**。

**结论**：唯一值得优化的维度是"每帧要匹配多少张模板"——这正是 pre-filter 的切入点。

### 1.2 用户提出的 9 个源码问题

**Q1：33 张模板运行前经过哪些预处理？**
`load_feature_templates` 只做两件事：`np.fromfile` + `cv2.imdecode` 读成 BGR `ndarray`，
再按文件名解析 `skin`/`scene` 构造 `TemplateInfo`。本阶段 33 张的加载参数为
`limit=0, dedupe_threshold=0, per_source_limit=0, round_robin=True`（并按 `skin*` 过滤）
⇒ **不做 dHash 去重**。（生产 12 张用 `dedupe_threshold=6`：61 → 57 → round_robin 抽 12。）
此时**没有**灰度化/缩放 —— 那些是惰性的。

**Q2：哪些预处理已在 TemplateMatcher 构造时缓存？**
**构造期不缓存任何东西**（`__init__` 只保存 config 并建空 dict）。
`template_pyramid()` 是**惰性**的：首次用到某模板时才 `to_grayscale` + 逐档 `resize_image`，
结果存入 `_template_pyramids`（键 = `template_id` + path + scales 元组），之后跨帧复用。
⇒ 首帧付出全部 `33 × 5` 次 resize 的代价，稳态为零。

**Q3：一帧 yard ROI 在精匹配前做过哪些操作？**

| 操作 | ROI 侧 | 说明 |
| --- | --- | --- |
| resize | **是** | `roi_scale=0.75` ⇒ `resize_image(cropped, 0.75)`（INTER_LINEAR） |
| grayscale | **是** | `to_grayscale` |
| pyramid | **否** | 金字塔只作用于**模板**，ROI 不做 |
| hash | **否**（默认） | 11.6-A 的 `roi_fingerprint` 仅在 `cache_size>0` 时计算，**默认关闭** |
| histogram | **否** | — |
| statistics | **否** | — |

⇒ 本阶段 pre-filter 用到的所有廉价特征（dHash / 缩略图 / 颜色统计 / 直方图 / 梯度）
**全部是新增计算**，不与他人复用。

**Q4：TemplateMatcher 是否必须逐模板调用 `cv2.matchTemplate`？**
**是**。`match(frame_gray, template)` 对该模板的 pyramid 逐档调用 `_match_once`
（一次 `cv2.matchTemplate` + 一次 `cv2.minMaxLoc`）。没有任何批量/向量化路径。

**Q5：full-match 的模板选择顺序是否影响最终结果？**
**判定不受影响**（`detected` = 最高分 ≥ 阈值）。但**选中的模板受顺序影响**：
`if best is None or match.confidence > best.confidence` 用**严格大于** ⇒ 同分时先遍历者胜出。
因此候选集变化**可能**改变 `bbox`（本阶段实测：thumb@4 与 naive33 的选中模板**零差异**，见 §10）。

**Q6：33 张模板构成**

| 项 | 值 |
| --- | --- |
| cap 来源 | **0 张**（naive 33 全是皮肤模板） |
| skin 来源 | **33 张 / 18 种皮肤** |
| 各皮肤张数 | skin001: 3、skin002: 3、skin003: 4、skin004: 3、skin005: 2、skin006: 3、skin007: 2、skin008: 3、skin009–skin018: 各 1 |
| 可追溯性 | `positive_sources.json`：18 来源、33 条映射，每张含 `from_image` + `source_id` |
| dHash 去重（阈值 6）后 | **32 张**（移除 1 张近重复）；本阶段按用户要求用**未去重的 33 张** |
| 高度相似模板 | 同一皮肤的多张不同画面（如 skin001 的 001/003/005、skin003 的 4 张） |

**Q7：round_robin 的职责 —— 与运行时预筛严格区分**

| | `round_robin` | 本阶段的 pre-filter |
| --- | --- | --- |
| 生效时机 | **模板加载期**（离线、一次性） | **运行时、逐帧** |
| 决定什么 | 哪 12 张模板进入 detector 的**模板库** | 这一帧拿哪 K 张去**精匹配** |
| 对覆盖的影响 | **缩小**（12 来源 ⇒ 皮肤覆盖下降：LOO 29/98） | **不变**（候选来自完整 33 张） |
| 对耗时的影响 | 间接（模板库小了） | 直接（每帧匹配数 165 → 20） |

⇒ 二者不可混为一谈：round_robin 是"用覆盖换速度"，pre-filter 是"**不换覆盖地省速度**"。

**Q8：0.75 ROI 下单模板精匹配耗时**
**5.66ms**（33 张 186.74ms）；生产 12 张为 5.91ms/张 ⇒ 线性。

**Q9：CPU 主要花在哪？**
**`matchTemplate`（99.8%）**。ROI resize + grayscale 共 0.27ms（0.14%）；
模板 pyramid 只在首帧一次性建立；Python 循环开销可忽略。

---

## 2. 数据口径与 reference 定义

### 2.1 样本

| | 数量 | 构成 |
| --- | --- | --- |
| 正样本 | **98** | `dataset/raw/yard/*.png` = cap 28 + skin 70（18 源）；**不含** 派生图（`states/`、`features/`、`scenes/`） |
| 负样本 | **80** | Phase 11.5 canonical 集合（**递归**收集）：exploration 36、unknown/other 22、unknown/collection 4、unknown/shop 4、unknown/shikigami 4、unknown/equipment 3、battle 3、unknown/friends 2、battle_result 1、battle_reward 1 |

**口径修正说明（重要）**：`tools/audit_yard_dataset.py` 的 `list_images` 用 `rglob("*")`
⇒ canonical 负样本是**80 张**（含 `raw/unknown/*` 子目录 39 张）。
而 Phase 11.6-B/C 的 `tools/benchmark_roi_downscale.py` 用的是非递归 `glob("*.png")`
⇒ 当时只统计到 **41 张**，其报告中的"负样本 41 张与 audit 口径一致"**不准确**。
本阶段工具改用递归收集，与 11.5 canonical 对齐（不修改历史报告）。

### 2.2 两个 reference（均为 `roi_scale=0.75`、`threshold=0.85`）

| reference | 模板集合 | 说明 |
| --- | --- | --- |
| **A：Production 12** | `dedupe_threshold=6 + round_robin` 抽 12 张 = `cap_001` + `skin001..skin011` | 当前生产配置 |
| **B：Naive 33** | 全部 33 张皮肤模板（18/18 皮肤） | 覆盖能力上限，**无** pre-filter，逐模板全匹配 |

> 另在报告中列出 `33 + cap 代表 = 34` 作为参考信息（本阶段**未**单独评测）；
> cap 覆盖的价值已由 Production 12（含 `cap_001`）实测反映。

**LOO**：按 `own_template_id(stem) = "bottom_menu_" + stem` 排除"由该测试图自身派生"的模板
（与 `tools/audit_yard_dataset.py` 同口径），并给出每个 case 的 `_loo` 变体。
本阶段 178 帧全部带 LOO（`excluded_pairs = 178`）。

---

## 3. Production 12 基线

| 指标 | 值 |
| --- | --- |
| TP / FN | **41 / 57**（recall 0.4184） |
| TP / FN（LOO） | **29 / 69**（recall 0.2959）—— 与 Phase 11.5-B 报告的 `LOO 29/98` 完全一致 |
| FP / TN | 0 / 80（precision 1.0000） |
| matchTemplate 调用/帧 | 60 |
| 单项耗时 p50 | 71.16ms |

---

## 4. Naive 33 reference（覆盖上限）

| 指标 | 值 |
| --- | --- |
| TP / FN | **73 / 25**（recall 0.7449） |
| TP / FN（LOO） | **50 / 48**（recall 0.5102） |
| FP / TN | 0 / 80（precision 1.0000） |
| matchTemplate 调用/帧 | 165 |
| 单项耗时 p50 | 188.33ms |
| 分来源命中 | **skin 70/70**；cap 3/28 |

**关键发现**：33 张皮肤模板对 70 张皮肤原图**全部命中**，但对 28 张 cap 原图只命中 3 张
（模板库里没有 cap 模板）⇒ 相对 Production 12（含 `cap_001`），Naive 33 多命中 **34 张**、
少命中 **2 张**，净增 **+32 TP**（LOO 下净增 **+21**：50 vs 29）。
⇒ **33 张模板的覆盖增益是真实且显著的**，这正是 pre-filter 要保住的东西。

---

## 5. Pre-filter 方法（5 种廉价、确定性、无训练算法）

流程（唯一允许的形态）：

```text
ROI → cheap pre-filter → Top-K template ids → 既有 TemplateMatcher 精匹配
    → 既有 threshold 0.85 → 既有 bbox 恢复(clamp) → 判定
```

pre-filter **只输出模板 id 排序**：不产生 `EvidenceResult`、不产生 bbox、不碰阈值、
不决定 yard 状态（测试 `test_prefilter_returns_only_ids` 静态检查了这一点）。

模板侧特征在**构造期**预计算一次（不计入每帧成本）；每帧只对 ROI 提一次特征，
与全部 33 张模板比较后排序：

| 算法 | 特征 | 距离 |
| --- | --- | --- |
| **`thumb`** | 灰度缩略图 `32×4`（`INTER_AREA`） | 逐元素 L1 均值 |
| `dhash` | 8×8 dHash（64 bit，与 `templates.template_dhash` 同口径） | 汉明距离 |
| `color` | BGR 通道 mean/std（6 维） | L1 均值 |
| `edge` | Sobel 幅值的 mean/std/高梯度占比（3 维） | L1 均值 |
| `combined` | 上述（含 16-bin 灰度直方图）的**排名归一化求和** | 秩和 |

**禁止项**（未使用）：YOLO / CNN / embedding / SIFT / 在线学习 / 机器学习分类器。
**未预设结论**：dHash 并非最优 —— 实测 `thumb` 显著更好（见 §6）。

---

## 6. Candidate Recall@K（首要指标）

在 Naive 33 能命中的 **73 张**图上，检查"Naive 33 选中的最佳模板"是否落在 pre-filter 的 Top-K：

| pre-filter | @4 | @6 | @8 | @10 | @12 | @16 | 最佳模板排名 p50 / max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`thumb`** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1 / 1** |
| `combined` | 0.9863 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 / 6 |
| `color` | 0.9726 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 / 6 |
| `dhash` | 0.9863 | 0.9863 | 0.9863 | 0.9863 | 0.9863 | 0.9863 | 1 / **28** |
| `edge` | 0.3562 | 0.4110 | 0.5616 | 0.6438 | 0.7671 | 0.9041 | 8 / 21 |

* **`thumb` 在 Top-4 就达到 100%**，且最佳模板**几乎总是排第 1**（p50 = max = 1）。
* `miss_count = 0`：没有任何图的最佳模板**完全**落出 33 张的排序（候选集本就是全集）。
* dHash 有 1 张图的最佳模板排到第 **28** 位（`yard_cap_004` ⇒ 见 §11 失败案例）。
* `edge` 单独用完全不够（@4 仅 0.36）——**不预设 dHash 最佳**这一要求得到验证。

---

## 7. 判定一致性：TP / FN / FP / TN

（98 正 + 80 负；`_loo` = 排除同源模板）

| case | TP | FN | FP | TN | recall | precision |
| --- | --- | --- | --- | --- | --- | --- |
| `production12` | 41 | 57 | 0 | 80 | 0.4184 | 1.0000 |
| `production12_loo` | 29 | 69 | 0 | 80 | 0.2959 | 1.0000 |
| `naive33` | **73** | 25 | 0 | 80 | 0.7449 | 1.0000 |
| `naive33_loo` | **50** | 48 | 0 | 80 | 0.5102 | 1.0000 |
| **`thumb@4`** | **73** | 25 | 0 | 80 | 0.7449 | 1.0000 |
| **`thumb@4_loo`** | **50** | 48 | 0 | 80 | 0.5102 | 1.0000 |
| `thumb@6` / `thumb@8` / `…@16` | 73 | 25 | 0 | 80 | 0.7449 | 1.0000 |
| （各 `thumb@K_loo`） | 50 | 48 | 0 | 80 | 0.5102 | 1.0000 |
| `combined@4` | 72 | 26 | 0 | 80 | 0.7347 | 1.0000 |
| `combined@6..16` | 73 | 25 | 0 | 80 | 0.7449 | 1.0000 |
| `color@4..16` | 73 | 25 | 0 | 80 | 0.7449 | 1.0000 |
| `color@4_loo` | 46 | 52 | 0 | 80 | 0.4694 | 1.0000 |
| `color@8..16_loo` | 50 | 48 | 0 | 80 | 0.5102 | 1.0000 |
| `dhash@4..16` | 72 | 26 | 0 | 80 | 0.7347 | 1.0000 |
| `edge@8` | 48 | 50 | 0 | 80 | 0.4898 | 1.0000 |
| `edge@16` | 66 | 32 | 0 | 80 | 0.6735 | 1.0000 |

### 7.1 逐图翻转（分类翻转，含方向）

| 比较 | 结果 |
| --- | --- |
| `naive33` → **`thumb@4..16`** | **0 翻转** |
| `naive33_loo` → **`thumb@4..16_loo`** | **0 翻转** |
| `naive33` → `color@4..16` | 0 翻转 |
| `naive33_loo` → `color@4_loo` | **4**（全部 positive→negative） |
| `naive33` → `combined@4` | 1 |
| `naive33` → `dhash@4..16` | **1**（`yard_cap_004`，positive→negative） |
| `naive33` → `edge@4` | **40**；@8 → 25；@16 → 7 |
| `naive33` → `production12` | **34**（全部 positive→negative） |
| `naive33_loo` → `production12_loo` | **21**（全部 positive→negative） |

**没有任何 negative→positive 翻转**（FP 全程为 0）——pre-filter **没有**让任何负样本变得更容易命中。

> 注意一个有价值的不对称：`color@4` 的 candidate recall@4 只有 0.9726（2 张图的最佳模板不在
> Top-4），但**最终判定仍与 naive33 完全相同**（73 TP）——因为候选集内还有别的模板同样
> 达到 ≥0.85。这说明"候选召回"比"判定一致性"更严格，是更保守的首选指标。

---

## 8. 困难负样本安全边际

诊断探针（`threshold≈0.001`，**只读分数、不进生产**）在 33 张模板上的原始最高分：

| 分组 | 张数 | max | p50 | p90 | p95 |
| --- | --- | --- | --- | --- | --- |
| **全 80 张** | 80 | **0.7793** | 0.4108 | 0.5960 | 0.7250 |
| `raw/unknown/friends`（困难） | 2 | **0.7793** | 0.7793 | 0.7793 | 0.7793 |
| `raw/unknown/other` | 22 | 0.7591 | 0.4087 | 0.7197 | 0.7322 |
| `raw/unknown/shop`（困难） | 4 | 0.6981 | 0.4372 | 0.6419 | 0.6700 |
| `raw/unknown/collection` | 4 | 0.5919 | 0.4754 | 0.5638 | 0.5779 |
| `raw/exploration` | 36 | 0.5436 | 0.3993 | 0.4824 | 0.4989 |
| `raw/unknown/shikigami` | 4 | 0.5310 | 0.4283 | 0.5012 | 0.5161 |
| `raw/unknown/equipment` | 3 | 0.5191 | 0.4591 | 0.5071 | 0.5131 |
| `raw/battle` | 3 | 0.2903 | 0.2846 | 0.2892 | 0.2897 |
| `raw/battle_result` | 1 | 0.2478 | 0.2478 | 0.2478 | 0.2478 |
| `raw/battle_reward` | 1 | 0.2141 | 0.2141 | 0.2141 | 0.2141 |

* **距离阈值 0.85 的余量 = 0.0707**（最高分 0.7793 来自 `unknown/friends`）；
* **超阈值 0 张**（与 11.5 canonical 的 `FP 0/80` 一致）；
* pre-filter **不改变**这一结论：判定仍建立在"候选集内的最高分"上，而候选集 ⊆ 33 张全量
  ⇒ 候选集的最高分 ≤ 全量最高分 ⇒ **FP 不可能因 pre-filter 增加**（§7 实测亦为 0）；
* 风险最高的是 `unknown/friends`（0.7793）与 `unknown/other`（0.7591），
  若未来降阈值到 0.75 会立刻出现 FP（与 11.5 的阈值扫描结论一致）。

---

## 9. Latency 分解

### 9.1 确定性操作数预算（与硬件无关）

| case | 模板数 | 档位数 | `matchTemplate` 调用/帧 | 相对 Naive 33 |
| --- | --- | --- | --- | --- |
| Production 12 | 12 | 5 | 60 | −64% |
| **Naive 33** | 33 | 5 | **165** | — |
| **thumb@4** | 4 | 5 | **20** | **−88%** |
| thumb@8 | 8 | 5 | 40 | −76% |
| thumb@12 | 12 | 5 | 60 | −64% |
| thumb@16 | 16 | 5 | 80 | −52% |

### 9.2 Wall-clock（**硬件敏感**，98 正样本 × repeat 3 + warmup 1，中位数）

| case | prefilter p50 | 精匹配 p50 | **total p50** | mean | p90 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| Production 12 | 0.0005ms | 70.90ms | **71.16ms** | 71.03 | 76.71 | 79.25 |
| Naive 33 | 0.0005ms | 188.09ms | **188.33ms** | 188.29 | 203.66 | 212.01 |
| **thumb@4** | **0.288ms** | 22.71ms | **23.25ms** | 23.41 | 25.17 | **26.50** |
| thumb@6 | 0.289ms | 33.90ms | **34.40ms** | 34.72 | 37.49 | 40.53 |
| thumb@8 | 0.290ms | 45.28ms | **45.84ms** | 45.88 | 48.76 | 51.33 |
| thumb@10 | 0.289ms | 57.27ms | **57.79ms** | 57.72 | 62.27 | 65.10 |
| thumb@12 | 0.291ms | 69.82ms | **70.28ms** | 71.23 | 82.27 | 88.03 |
| thumb@16 | 0.295ms | 99.57ms | **100.23ms** | 110.11 | 145.23 | 174.22 |

* **pre-filter 自身只花 0.288ms**，占 `thumb@4` total 的 **1.2%** —— 完全不会吃掉收益；
* 这些数字**不是线性外推**：33 张模板库 + Top-K 精匹配是**实测**的；
* 硬件敏感：同一配置两次运行的生产 12 项 p50 相差约 8%（65.68 vs 71.16ms），
  因此**不应把单次 <500ms 当作可重复事实** —— 见 §9.3 的余量设计建议。

### 9.3 推算整链（其它 detector + 状态判定基线 282ms）

| case | 单项 p50 | 推算整链 | 500ms 余量 |
| --- | --- | --- | --- |
| Production 12 | 71.16ms | 353ms | 29% |
| **Naive 33** | 188.33ms | **470ms** | **6%（危险）** |
| **33 + thumb@4** | 23.25ms | **305ms** | **39%** |
| 33 + thumb@8 | 45.84ms | 328ms | 34% |
| 33 + thumb@12 | 70.28ms | 352ms | 30% |
| 33 + thumb@16 | 100.23ms | 382ms | 24% |

⇒ **Naive 33 已经贴近上限（470ms，余量 6%）**，这正是本阶段要解决的问题；
`thumb@4` 把 33 张模板的覆盖带回 **305ms（余量 39%）**，甚至**比当前生产 12 张更快**。

---

## 10. bbox 一致性与坐标安全

bbox 仍走 Phase 11.6-C 的生产路径（`roi_scale=0.75` → `_restore_rect` → clamp），
pre-filter 不参与坐标计算。

| case | 共同命中 | IoU p50 | IoU min | 中心误差 max | **选中模板不同** |
| --- | --- | --- | --- | --- | --- |
| **thumb@4** | 73 | 1.0000 | 1.0000 | **0.0px** | **0** |
| **thumb@8** | 73 | 1.0000 | 1.0000 | 0.0px | **0** |
| dhash@4 | 72 | 1.0000 | 1.0000 | 0.0px | 0 |
| color@4 | 73 | 1.0000 | 1.0000 | 0.0px | **2** |
| edge@4 | 33 | 1.0000 | 1.0000 | 0.0px | 7 |

* **bbox 越界 = 0**（`max_px = 0`，`count = 0`），覆盖 `naive_all` / `naive_loo` / `production12` 三个判定；
* `thumb@4` 与 Naive 33 选中的**模板完全相同**（`template_selection_mismatch = 0`），
  bbox 逐像素一致；
* `color@4` / `edge@4` 有少量图选中了**同皮肤的另一张模板**（score 更低的候选），
  由于同皮肤模板的 ROI 位置相同，IoU 仍为 1.0、中心误差 0 —— 但**模板 id 变了**，
  如实报告（例：`color@4` 让 `yard_skin007_007` 从 `skin007_001` 改为 `skin007_003`）。

---

## 11. 失败案例

| case | 现象 | 原因 |
| --- | --- | --- |
| `dhash@4..16` | `yard_cap_004` 由命中变漏检（0.8821 → 未达阈值） | 该图的最佳模板 `skin009_001` 在 dHash 排序里只排第 **28** 位；dHash 只有 64 bit，且对"底部功能栏这类大面积相似 UI"分辨力不足 |
| `edge@4` | 40 张正样本漏检 | 3 维梯度统计（mean/std/高梯度占比）过于粗糙，几乎无法区分 18 种皮肤 |
| `color@4_loo` | 4 张 LOO 漏检 | 颜色统计对"同皮肤不同背景亮度"敏感；@6 起恢复正常 |
| `combined@4` | 1 张漏检 | 秩和融合被 `edge` 分量拖累（`combined` 的 @4 = 0.9863 与 dhash 相同） |
| `thumb@4` | **无** | — |

另有 25 张正样本在 **Naive 33 下也不命中**（其中 25 张多为 cap 原图）⇒ 属**模板库覆盖**问题，
不是 pre-filter 的问题（`naive33` 与 `thumb@4` 在这 25 张上判定一致）。

---

## 12. 结论：推荐方案（`prefilter_viable`）

**推荐：`thumb`（32×4 灰度缩略图 L1 距离）+ Top-4**（保守可退 Top-8）。

逐条对照验收决策规则：

| # | 规则 | `thumb@4` 实测 |
| --- | --- | --- |
| 1 | 相对 Naive 33 的 candidate recall 足够高 | **1.0000**（73/73，最佳模板 p50=max=1） |
| 2 | TP 明显高于 Production 12 或保留 Naive 33 主要增益 | **73 vs 41**（保留全部 +32 增益；LOO 50 vs 29） |
| 3 | FP 不高于当前基线 | **0**（negative→positive 翻转 0） |
| 4 | threshold 保持 0.85 | 是（无任何改动） |
| 5 | bbox 合法率 100% | 是（越界 0、IoU 1.0、选中模板差异 0） |
| 6 | total 显著低于 Naive 33 | **23.25ms vs 188.33ms（8.1×）** |
| 7 | 整链可稳定 <500ms | **305ms（余量 39%）** |
| 8 | pre-filter 自身不吃掉收益 | **0.288ms（占 1.2%）** |

**8 项全部满足** ⇒ 建议进入下一阶段（Phase 11.7-B）做生产接入评估。

**Top-4 vs Top-8 的取舍**：两者 candidate recall 均为 1.0、判定均与 Naive 33 一致。
考虑到本阶段正样本仅 98 张（18 皮肤），未来出现新皮肤时"最佳模板恰好排第 1"未必总成立，
**更稳的默认值是 Top-8**（整链 328ms，余量 34%，仍远优于 Naive 33）；若追求极致延迟则用 Top-4。

**不推荐**：`dhash`（差 1 张，且排名最大 28，稳定性弱于 thumb）、`edge`（单独不可用）、
`combined`（无额外收益，却多算 5 种特征）。

---

## 13. 本阶段局限

| 编号 | 局限 |
| --- | --- |
| L-1 | 正样本仅 **98 张 / 18 皮肤**，未覆盖新皮肤、新活动 UI、跨分辨率与多 DPI |
| L-2 | 只评测了 `bottom_menu_yard` 一个 detector；其它 detector（如 `battle_auto_button` / `victory_icon`）模板更小，需各自验证 |
| L-3 | latency 是 **wall-clock**，受机器负载影响（同配置两次运行差约 8%）；操作数预算是确定性的，二者已在 §9 明确区分 |
| L-4 | 未接入真实运行期（约束禁止）⇒ 未观测真实采集的编码噪声、缩放抖动对 `thumb` 距离的影响 |
| L-5 | pre-filter 基于"ROI 外观相似度"；若未来出现"外观相似但语义不同"的元素，候选召回可能下降 |
| L-6 | 33 张模板**不含 cap 模板** ⇒ 28 张 cap 原图只命中 3 张；"33 vs 34（+cap 代表）"未单独评测 |
| L-7 | 分数表用 `threshold≈0` 的探针 matcher 读取，分数恰为 0 的模板会被过滤（每帧 `template_count` 26~33），对判定无影响（0 ≪ 0.85） |
| L-8 | 未做"模板库变化后 Top-K 稳定性"的敏感性分析（例如换成 57 张全量模板） |

---

## 14. Phase 11.7-B 建议

1. **把 pre-filter 抽成可注入的纯函数**（`TemplatePrefilter`：输入 prepared ROI + 模板特征库，
   输出候选 id 列表），放在 `tools/` 或显式 experimental 模块；**默认关闭**，保持现有行为零变化。
2. **配置形态**（与 11.6-C 的 `roi_scale` 同构）：`evidence.detectors.<name>.template_prefilter`
   + `top_k`；默认 `top_k: 0` = 关闭（等价全量匹配）。
3. **安全兜底**：候选为空/异常时**回退全量匹配**；`top_k >= 模板数` 时等价全量（已有测试）。
4. **可观测性**：命中时把 `prefilter` 名称、`top_k`、候选数写入 `metadata`，便于事后审计。
5. **接入前补测**：其它 detector、更多皮肤、跨分辨率；并复核 `thumb` 在"低分辨率 / 不同 DPI"
   下的候选召回是否仍为 1.0。
6. **覆盖换速度的次序**：先用 pre-filter 解锁 33 张模板（覆盖 41 → 73），
   再考虑是否顺带把 `roi_scale` 从 0.75 降到 0.5（11.6-C 的 R-1 遗留）——
   两者叠加后整链余量充足，但 0.5 的 bbox 精度代价需单独评估。
7. **回归基线**：接入后必须保证 `production12` 的既有行为可由配置完全复现（`top_k: 0`）。

---

## 15. 测试与验收

```text
$ python -m pytest -q                          → 1644 passed, 1 skipped   # 1618 → +26
$ python tools/check_dataset.py                → PASS
$ python tools/raw_manifest.py                 → raw 178 张 SHA256 未变
$ python tools/check_annotations.py            → PASS
$ python tools/benchmark_template_prefilter.py --latency --latency-repeat 3   → 见 §6-§9
```

新增 `tests/test_phase117a_template_prefilter_benchmark.py`（**26 项**，覆盖用户要求的 20 点）：
候选排序确定性 / Top-K 尺寸 / `K ≥ 模板数` 等价全量 / 不修改 threshold / 不修改 production
config / 不修改模板文件 / 不修改 raw dataset / pre-filter 不产生 `EvidenceResult` /
精匹配仍是既有 `TemplateMatcher` / candidate recall 用 records 独立重算 / LOO 排除同源模板 /
同分排序稳定 / 探针不进生产 matcher / bbox 走生产 `_restore_rect` / bbox 越界 0 /
latency 分项（prefilter·match·total，含 p50/p90/p95）/ 报告含样本数与 `source_id` /
输出只写 TEMP / `action.enabled=False` / `backend=dry_run`。

**本阶段未新增任何 skip**，也未删除既有测试。

---

## 16. Git commit

```text
test(vision): benchmark yard template candidate prefilter
```
