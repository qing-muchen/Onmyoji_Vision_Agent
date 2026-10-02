# Phase 11.7-B — 模板预筛生产加固报告

> 本阶段只建设**生产能力 + 混合模板库 benchmark + 稳定性验证**：
> 预筛实现完成且**默认关闭**，``bottom_menu_yard`` **仍未启用**新配置（仍是 12 模板 / roi_scale 0.75）。
> 不修改 ``threshold``、不新增正式模板文件、不修改 action / resolver / safety。
> commit：``feat(vision): harden template prefilter for production``

**核心问题**：在补齐 ``Production12`` 的独有覆盖、且模板库发生合理变化后，Top-K 预筛是否仍然稳定？
是否已经具备"默认关闭、可回滚、可观测"的生产实现条件？

**答案**：是。``Production12`` 的独有覆盖**只有 1 张且是自匹配**（LOO 下为 0）；
推荐库 **Skin33 + 3 张 cap = 36 张**（LOO 56/98，相对 Skin33 的 +6 全部来自泛化）；
Top-4 的 candidate recall 在 **6 种模板库扰动下均为 1.0**；预筛已实现为默认关闭的生产组件。

---

## 0. 一句话结论（先读）

| 指标（98 正 / 80 负，阈值 0.85） | Production 12 | Skin33 | **Skin33+cap3（36）** |
| --- | --- | --- | --- |
| TP / FN | 41 / 57 | 73 / 25 | **81 / 17** |
| **LOO TP / FN** | 29 / 69 | 50 / 48 | **56 / 42** |
| FP / TN | 0 / 80 | 0 / 80 | **0 / 80** |
| candidate recall@4 | — | 1.0 | **1.0** |
| bbox 越界 | 0 | 0 | **0** |

* **cap 补集**（贪心）：``cap_006``(+5 TP / **+4 LOO**) → ``cap_003``(+2 / +1) → ``cap_007``(+1 / **+1**) → 之后 LOO 增益 **0**（只剩自匹配）。
* **不需要"补回 Production12 独有覆盖"**：那唯一 1 张（``yard_cap_001``）是 ``cap_001`` 的**自匹配**，LOO 下不成立。
* 36 张库 + Top-4：单项 p50 **29.88ms**（naive 256.09ms ⇒ **8.6×**），``matchTemplate`` **180 → 20 次/帧（−89%）**。
* 负数安全边际与 33 张时**完全相同**（max 0.7793，距阈值 0.0707）⇒ 加 cap 模板**没有**抬高负样本分数。

---

## 1. Production12 与 Skin33 的真实差异

用**一张 61 模板全量分数表**推导（避免重复精匹配），逐图四分类：

| 维度 | A（both） | B（Skin33 only） | C（Production12 only） | D（neither） | 合计 |
| --- | --- | --- | --- | --- | --- |
| **普通** | 40 | 33 | **1** | 24 | 98 |
| **LOO** | 29 | 21 | **0** | 48 | 98 |

* 普通：``Production12`` TP = A + C = **41** ✅；``Skin33`` TP = A + B = **73** ✅
* LOO：``Production12`` = 29 ✅；``Skin33`` = 50 ✅，且 **C = 0** ⇒ **在泛化意义上 Skin33 是 Production12 的严格超集**
* 唯一的生产独有命中：``raw/yard/yard_cap_001.png`` → 命中模板 ``bottom_menu_yard_cap_001``，
  ``confidence = 1.0``（**同源自匹配**），同帧上 Skin33 的最高分仅 **0.6457**，且 ``loo_detected = False``。

> 这正面回答了本阶段的第一个缺口：**不存在"Production12 独有、且非自匹配"的覆盖**。
> 因此"补回 Production12 独有覆盖"这个目标本身在 LOO 意义上是**空集**。

---

## 2. 34 个 flip 的方向分布

| 方向 | 普通 | LOO |
| --- | --- | --- |
| ``positive → negative``（Skin33 命中、Production12 未命中） | **33** | **21** |
| ``negative → positive``（Production12 命中、Skin33 未命中） | **1** | **0** |
| **合计** | **34** | **21** |

* TP 差 = 73 − 41 = **32**，与翻转数 34 的差额恰为 ``C − B`` 的反向贡献（33 − 1 = 32）✅ **完全自洽**；
* 那 1 个 ``negative → positive`` 就是 §1 的 ``yard_cap_001``（自匹配）；
* **没有任何 negative 样本发生翻转**（FP 全程为 0）。

---

## 3. cap 补集分析（数据驱动贪心）

目标函数：新增**普通 TP** 最大，并列时取 LOO TP 最大。

| 步 | 模板 | Cap TP | Cap LOO | 新增 TP | **新增 LOO** | 新增覆盖 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ``cap_006`` | 78 | 54 | +5 | **+4** | cap_006/007/016/020/028 |
| 2 | ``cap_003`` | 80 | 55 | +2 | **+1** | cap_003/027 |
| 3 | ``cap_007`` | 81 | 56 | +1 | **+1** | cap_019 |
| 4 | ``cap_001`` | 82 | 56 | +1 | **+0** | cap_001（自匹配） |
| 5 | ``cap_002`` | 83 | 56 | +1 | **+0** | cap_002 |
| … | （后续每步 +1 TP / +0 LOO，直到 TP 饱和于 98/98） | | | | | |

**关键**：**LOO 在第 3 步（``cap_007``）饱和于 56**。此后每增加一张 cap 模板只提升"自匹配式"的普通 TP，
**不提升泛化**。⇒ 最小完整库 = **Skin33 + cap{006, 003, 007} = 36 张**。

> 注意：贪心选的**不是**生产在用的 ``cap_001`` —— ``cap_001`` 只在自匹配上有效，
> 这正是"不能随便挑一张 cap 模板"（用户约束）的量化证据。

---

## 4. 最终 mixed library 建议

| 库 | 模板数 | TP | FN | LOO TP | LOO recall | recall@4 | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| production12（现状） | 12 | 41 | 57 | 29 | 0.2959 | 1.0 | 0 |
| skin33 | 33 | 73 | 25 | 50 | 0.5102 | 1.0 | 0 |
| skin33+cap1 | 34 | 78 | 20 | 54 | 0.5510 | 1.0 | 0 |
| skin33+cap2 | 35 | 80 | 18 | 55 | 0.5612 | 1.0 | 0 |
| **skin33+cap3 ★推荐** | **36** | **81** | **17** | **56** | **0.5714** | **1.0** | **0** |
| skin33+cap4 | 37 | 82 | 16 | 56 | 0.5714 | 1.0 | 0 |
| … | | | | | | | |
| skin33+cap20（TP 饱和） | 53 | 98 | 0 | 56 | 0.5714 | 1.0 | 0 |
| all61（上限参考） | 61 | 98 | 0 | 57 | 0.5816 | 1.0 | 0 |

**推荐 36 张**，理由：

1. 它是**泛化（LOO）达到上限的最小集合**：再加 17 张 cap 才把普通 TP 推到 98，但 LOO 一分不涨（成本 +17 × ~5ms/帧）；
2. 相对现状 Production12：**LOO 29 → 56（+93%）**、TP 41 → 81（+98%）；
3. 每张新模板的边际泛化收益：21 张 skin ⇒ **+21 LOO**（1.0/张）、3 张 cap ⇒ **+6 LOO**（**2.0/张**）——
   cap 模板的边际效率是 skin 的 2 倍（cap 原图有 28 张，而 Skin33 对它们覆盖很差）。

> 若目标是"离线识别率最大化"而非泛化，则可选 ``skin33+cap20``（53 张，TP 98/98）；但 LOO 不变，
> 且每帧精匹配成本 ×2.7。**本阶段推荐 36 张**。

---

## 5. LOO 结果

| 库 | 普通 TP | LOO TP | 差额（自匹配贡献） |
| --- | --- | --- | --- |
| production12 | 41 | 29 | 12 |
| skin33 | 73 | 50 | 23 |
| **skin33+cap3** | **81** | **56** | **25** |
| all61 | 98 | 57 | 41 |

* LOO 按 ``own_template_id(stem) = "bottom_menu_" + stem`` 排除同源模板（与 ``tools/audit_yard_dataset.py`` 同口径）；
* **加 cap 模板的收益不是自匹配造成的**：前 3 步的 LOO 增益为 +4 / +1 / +1（普通增益 +5 / +2 / +1），
  即 6 个新增 TP 中有 **6 个**是泛化（``cap_006`` 的 5 个里 4 个泛化、``cap_003`` 的 2 个里 1 个、``cap_007`` 的 1 个全泛化）；
* 第 4 步起 LOO 增益为 0 ⇒ 该步及以后**纯自匹配**，不纳入推荐库。

---

## 6. Top-K 稳定性（模板库扰动）

对推荐库（36 张）做 6 种扰动，固定 ``seed = 11701``（可复现）：

| 变体 | 模板数 | 命中图 | recall@4 | recall@6 | recall@8 | 最佳模板 rank（p50 / p95 / max） |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 36 | 81 | **1.0** | **1.0** | **1.0** | 1.0 / 1.0 / 2 |
| **顺序随机重排** | 36 | 81 | **1.0** | **1.0** | **1.0** | 1.0 / 1.0 / 2 |
| 加 1 张无关合法模板 | 37 | 82 | **1.0** | **1.0** | **1.0** | 1.0 / 1.0 / 2 |
| 删 1 张非关键模板 | 35 | 80 | **1.0** | **1.0** | **1.0** | 1.0 / 1.0 / 1 |
| 加 cap 模板 | 38 | 83 | **1.0** | **1.0** | **1.0** | 1.0 / 1.0 / 2 |
| 加同皮肤近似模板 | 37 | 82 | **1.0** | **1.0** | **1.0** | 1.0 / 1.0 / 2 |

* **Top-4 的成功不是文件顺序或特定库构成的巧合**：重排后 recall 与排名分布完全不变；
* 全部变体下最佳模板的排名 **max ≤ 2**（几乎总是第 1 名）；
* 同一测试也在 53 张库（``skin33+cap20``）上跑过，结论一致（recall@4 = 1.0、rank max ≤ 4）。

⇒ **生产推荐 Top-4**；若偏好更大冗余，Top-8 同样稳定（成本 p50 67.91ms，仍在预算内，见 §10）。

---

## 7. 生产预筛架构

### 7.1 新增模块

``src/vision/evidence/template_prefilter.py``：

| 成员 | 职责 |
| --- | --- |
| ``thumb_vector(gray)`` | 32×4 灰度缩略图（``INTER_AREA``）→ 128 维 ``float32`` |
| ``TemplatePrefilter(templates)`` | **构造期**一次性缓存全部模板特征；``build_ms`` 记录成本 |
| ``.rank(roi_gray)`` | 返回 :class:`PrefilterRanking`（确定性排序 + 分项耗时） |
| ``.select(roi_gray, top_k)`` | 取 Top-K（``top_k >= 模板数`` 时直接返回全部） |

**接口语义**：输入"已准备好的 ROI 灰度图 + 当前已加载模板集合"，输出**排序后的模板 id**。
**不产生** ``EvidenceResult`` / bbox / 状态；**不读取** Task / Runtime / Action；
**不改变** ``TemplateMatcher`` / ``cv2.matchTemplate`` / ``threshold`` 语义（由 AST 测试守护）。

**算法一致性**：``thumb`` 的定义（尺寸、插值、L1 距离、``stable`` 排序）与 Phase 11.7-A benchmark
**逐位一致**，并由 ``test_thumb_consistent_with_11_7a_benchmark`` 长期守护 —— 不存在"benchmark 里是 A、生产里是 B"。

### 7.2 接线点

``src/vision/evidence/base.py``：``best_match`` 的模板循环由 ``self._templates`` 改为
``self._candidate_templates(frame_gray)``；新增 ``_build_prefilter`` / ``_candidate_templates`` /
``_record_prefilter_fallback``。**ROI 提取、降采样、坐标恢复、clamp 全部不变**。

### 7.3 特征缓存（静态 vs 每帧）

| 阶段 | 内容 | 成本 |
| --- | --- | --- |
| 构造期（一次） | 33 张模板 → 灰度 → 32×4 缩略图 → 缓存 | **14.88ms**（33 张） |
| 每帧 | ROI → 32×4 缩略图 + 向量化 L1 + 稳定排序 | **0.05–0.25ms**（实测 roi_feature_ms + ranking_ms） |

⇒ 预筛本身处于**亚毫秒级**，且**不会**每帧重算模板特征（测试用 ``thumb_vector`` 调用计数钉住）。

---

## 8. 默认关闭语义（三层保证）

| 层 | 行为 |
| --- | --- |
| 配置 | ``template_prefilter: "none"``（默认）+ ``prefilter_top_k: 0``（默认）⇒ ``prefilter_enabled = False`` |
| 构造 | ``prefilter_enabled=False`` ⇒ **不构造** ``TemplatePrefilter``（不付特征构建成本） |
| 每帧 | ``_candidate_templates`` 首行即返回 ``self._templates`` 并置 ``_last_prefilter = None`` ⇒ 模板顺序、``matchTemplate`` 次数、``metadata`` 全部与既有版本逐字段一致 |

**与 ``max_templates`` 的严格分离**（本阶段的核心设计）：

| 字段 | 控制什么 | 现状 |
| --- | --- | --- |
| ``max_templates`` | **模板库规模**（加载期，经 dedupe → per_source → round_robin 抽样） | 12 |
| ``prefilter_top_k`` | **每帧进入精匹配的候选数量**（运行期） | 0 |

现状下二者**耦合**（``best_match`` 遍历全部已加载模板）；本阶段把"每帧匹配数"独立出来：

```yaml
max_templates: 36        # 加载 36 张
prefilter_top_k: 4       # 每帧只精匹配 Top-4
```

**``prefilter_top_k = 0`` 只表示"不做预筛"（每帧全量匹配已加载模板），绝不用它表示"回滚到 12 模板"** ——
回滚靠 ``max_templates: 12``（见 §15）。

---

## 9. fallback 语义

| 情形 | 行为 | ``metadata`` |
| --- | --- | --- |
| 未启用预筛 | 全量匹配 | 不写入任何 prefilter 字段 |
| 预筛**初始化**失败（特征构建异常） | 全量匹配 | ``prefilter_fallback=True`` + ``prefilter_fallback_reason`` |
| 未知算法名（``model_copy`` 绕过校验） | 全量匹配 | 同上（reason 含 ``unknown algorithm``） |
| ROI 特征 / 排序抛异常 | 全量匹配 | 同上（reason 含异常类型与消息） |
| 候选为空 / id 非法 | 全量匹配 | 同上 |
| ``top_k >= 已加载模板数`` | 直接全量（**不**额外跑预筛） | ``prefilter_fallback=False``，reason 说明等价全量 |
| **Top-K 里没有模板超过阈值** | **正常"未检出"** | **不 fallback**（这是"画面里没有 yard"，不是故障） |

最后一条是关键：若"未命中就二次全量匹配"，则**非 yard 帧与未被覆盖的 yard 帧**会把省下的成本全部付回去，
预筛就失去意义。``test_no_match_does_not_trigger_fallback`` 钉住了这一点。

---

## 10. Latency（实测生产 detector，36 张库）

```bash
python tools/benchmark_template_library.py --stability --latency --latency-repeat 3
```

### 10.1 确定性操作数预算（与硬件无关）

| case | 模板库 | 档位 | ``matchTemplate`` 调用/帧 | 相对 naive 36 |
| --- | --- | --- | --- | --- |
| production12_naive | 12 | 5 | 60 | — |
| skin33_naive | 33 | 5 | 165 | — |
| **skin33+cap3_naive** | 36 | 5 | **180** | — |
| **skin33+cap3_top4** | 36 | 5 | **20** | **−89%** |
| skin33+cap3_top6 | 36 | 5 | 30 | −83% |
| skin33+cap3_top8 | 36 | 5 | 40 | −78% |

### 10.2 Wall-clock（98 正样本 × repeat 3 + warmup 1，中位数，**硬件敏感**）

| case | p50 | p90 | p95 |
| --- | --- | --- | --- |
| production12_naive | 88.47ms | 110.01ms | 117.99ms |
| skin33_naive | 247.68ms | 290.87ms | 302.16ms |
| **skin33+cap3_naive** | **256.09ms** | 336.58ms | 378.32ms |
| **skin33+cap3_top4** | **29.88ms** | 42.33ms | 47.03ms |
| skin33+cap3_top6 | 40.82ms | 53.71ms | 55.67ms |
| skin33+cap3_top8 | 67.91ms | 97.29ms | 106.86ms |

* **36 张库 + Top-4：29.88ms vs naive 256.09ms ⇒ 8.6×**；推算整链 = 282 + 29.88 ≈ **312ms**（500ms 余量 38%）；
* ⚠️ **wall-clock 波动显著**：同一 ``production12_naive`` 在相邻两次运行中为 67.77ms / 88.47ms（**±30%**），
  ``skin33+cap3_top4`` 为 30.21 / 29.88ms（±1%）——
  **绝对耗时受机器负载影响，操作数预算（180 → 20）才是稳定事实**；
* 因此本阶段不把任何单次 "<500ms" 当作可重复结论，而是以"操作数 −89% + 预筛亚毫秒"为决策依据。

---

## 11. 负样本安全边际（36 张库，canonical 80 张）

| 分组 | 张数 | max | p50 | p90 | p95 |
| --- | --- | --- | --- | --- | --- |
| **全 80 张** | 80 | **0.7793** | 0.4108 | 0.5960 | 0.7250 |
| ``raw/unknown/friends``（困难） | 2 | **0.7793** | 0.7793 | 0.7793 | 0.7793 |
| ``raw/unknown/other`` | 22 | 0.7591 | 0.4087 | 0.7197 | 0.7322 |
| ``raw/unknown/shop``（困难） | 4 | 0.6981 | 0.4372 | 0.6419 | 0.6700 |
| ``raw/unknown/collection`` | 4 | 0.5919 | 0.4754 | 0.5639 | 0.5779 |
| ``raw/exploration`` | 36 | 0.5436 | 0.3994 | 0.4824 | 0.4988 |
| ``raw/unknown/shikigami`` | 4 | 0.5310 | 0.4283 | 0.5012 | 0.5161 |
| ``raw/unknown/equipment`` | 3 | 0.5191 | 0.4591 | 0.5071 | 0.5131 |
| ``raw/battle`` | 3 | 0.2903 | 0.2846 | 0.2892 | 0.2898 |
| ``raw/battle_result`` | 1 | 0.2478 | — | — | — |
| ``raw/battle_reward`` | 1 | 0.2141 | — | — | — |

* 距阈值 **0.0707**，**超阈值 0 张**；与 Skin33（33 张）**逐位相同**（max/p50 完全一致）
  ⇒ **增加 3 张 cap 模板没有抬高任何负样本分数**；
* 预筛在"候选集 ⊆ 全量"上取最高分 ⇒ 最高分不可能升高 ⇒ **FP 不可能因预筛增加**
  （4 个算法的 3 个 Top-K × 80 张负样本实测：FP = 0）。

---

## 12. bbox 一致性（36 张库）

| case | 共同命中 | IoU p50 | IoU min | 中心误差 max | 选中模板不同 | **越界** |
| --- | --- | --- | --- | --- | --- | --- |
| **top4** | 81 | **1.0000** | **1.0000** | **0.0px** | **0** | **0** |
| top6 | 81 | 1.0000 | 1.0000 | 0.0px | 0 | 0 |
| top8 | 81 | 1.0000 | 1.0000 | 0.0px | 0 | 0 |

* Top-K 与 naive 全库**选中同一张模板**（``template_selection_mismatch = 0``）⇒ bbox 逐像素一致；
* bbox 仍由生产 ``_restore_rect`` + clamp 产生，**越界 0**。
* 对照：Phase 11.7-A 中 ``color@4`` 有 2 处、``edge@4`` 有 7 处选到不同模板 ⇒ **``thumb`` 在这一点上也更优**。

---

## 13. 测试结果

```text
$ python -m pytest -q                 → 1680 passed, 1 skipped   # 1644 → +36
$ python tools/check_dataset.py       → PASS（仅 warning）
$ python tools/raw_manifest.py        → raw 178 张 SHA256 未变
$ python tools/check_annotations.py   → PASS
$ python tools/benchmark_template_library.py --stability --latency --latency-repeat 3   → 见 §4-§12
$ python tools/benchmark_template_prefilter.py --latency --latency-repeat 3             → 11.7-A 口径复跑
```

新增 ``tests/test_phase117b_template_prefilter_production.py``（**36 项**，覆盖用户要求的 30 点）：
默认关闭 / 默认 metadata 与候选数不变 / thumb 与 11.7-A 逐位一致 / 模板特征静态缓存 /
ROI 特征每帧一次 / 排序确定性与 tie-break / Top-K 语义（=0、≥N）/ 不产生 ``EvidenceResult`` 与 bbox /
AST 依赖边界 / 异常与未知算法回退 / **正常无匹配不回退** / 阈值不变 / FP 不增 / bbox 合法 /
canonical 负样本 80 张 / LOO 正确 / 贪心数据驱动 / 稳定性可复现 / configs 与模板文件与 raw 不变 /
安全配置（``action.enabled=False``、``backend=dry_run``）。

**本阶段未新增任何 skip，未删除既有测试。**

---

## 14. Phase 11.7-C 推荐生产配置

```yaml
bottom_menu_yard:
  threshold: 0.85            # 不变
  roi_scale: 0.75            # 不变
  max_templates: 36          # 12 → 36（库规模）
  round_robin: false         # 需要按"Skin33 + 指定 3 张 cap"精确选库 ⇒ 不能用轮转抽样
  template_prefilter: thumb  # none → thumb
  prefilter_top_k: 4         # 0 → 4（每帧精匹配数）
```

**⚠️ 前置缺口（本阶段未解决）**：现有加载逻辑（``dedupe_threshold`` + ``per_source_limit`` + ``round_robin``）
**无法表达**"全部 33 张 skin + 指定的 3 张 cap"（轮转抽样不会跳过 ``cap_002..cap_005``）。
Phase 11.7-C 需要先补一个**明确的加载策略**，二选一：

| 方案 | 改动 | 评价 |
| --- | --- | --- |
| A. 新增加载过滤（如 ``include_sources`` / 显式模板清单） | 改 ``load_feature_templates`` 与配置模型 | **推荐**：不新增文件、可追溯、可回滚 |
| B. 新建派生目录（把 36 张复制到 ``.../yard/positive_prod/``） | 新增 36 个模板文件 | 需重跑 ``raw_manifest`` 之外的模板校验；与"正式模板不新增"冲突 |

---

## 15. Rollback 配置（精确回滚到当前生产行为）

```yaml
bottom_menu_yard:
  threshold: 0.85
  max_templates: 12
  round_robin: true
  dedupe_threshold: 6
  per_source_limit: 0
  roi_scale: 0.75
  template_prefilter: none     # 或不写（默认）
  prefilter_top_k: 0           # 或不写（默认）
```

**回滚靠 ``max_templates: 12``，不靠 ``prefilter_top_k``** —— 后者只控制"每帧匹配多少张"。
回滚路径已由 ``test_default_candidate_count_unchanged`` / ``test_default_metadata_unchanged`` 覆盖。

---

## 16. 未解决问题

| 编号 | 内容 |
| --- | --- |
| U-1 | **加载策略未实现**：如何精确加载"33 skin + 3 cap"（§14 方案 A/B 未决） |
| U-2 | **跨分辨率未验证**：raw 全部为 1703–1706 × 960–963，本阶段**不做**合成缩放冒充跨分辨率验证（留 Phase 11.8） |
| U-3 | **只验证了 ``bottom_menu_yard``**；其它 detector（模板更小）需各自 benchmark |
| U-4 | **只产品化了 ``thumb``**；``dhash`` / ``color`` / ``combined`` 仅停留在 11.7-A 的实验实现 |
| U-5 | **wall-clock 波动 ±30%**（同配置两次运行），latency 结论应以操作数预算为准 |
| U-6 | **未接入真实运行期**（本阶段约束），未观测真实采集噪声对 ``thumb`` 距离的影响 |
| U-7 | **稳定性只在 178 帧上验证**；模板库若长期增长（>60 张）需重新评估 Top-K |
| U-8 | ``all61`` 的 LOO 57 比推荐库的 56 高 1 ⇒ 仍存在 1 个"未被 36 张覆盖的泛化命中"，本阶段未定位其价值/成本比 |

---

## 17. Git commit

```text
feat(vision): harden template prefilter for production
```
