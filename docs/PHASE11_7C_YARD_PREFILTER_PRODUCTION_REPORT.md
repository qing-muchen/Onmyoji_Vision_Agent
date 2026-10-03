# Phase 11.7-C — 精确模板集合选择与 yard 生产启用报告

> 本阶段把 Phase 11.7-B 验证过的 **36 张模板 + `thumb` Top-4** 正式接入生产：
> 新增**显式模板集合**能力、精确加载、并把 `bottom_menu_yard` 切换到新配置。
> 保持 `threshold=0.85`、`roi_scale=0.75`；不修改 action / resolver / safety；不启用真实输入。
> commit：``feat(vision): enable yard template prefilter in production``

**成功标准**（本阶段自证）：生产 detector **真实加载** 11.7-B 验证过的那 36 张模板，
在**真正的生产路径**上以 Top-4 精匹配复现 **TP 81/98、LOO 56/98、FP 0/80**，
并且可以通过明确配置恢复到旧的 Production12 行为。

---

## 0. 一句话结论（先读）

| 指标（98 正 / 80 canonical 负，阈值 0.85） | **Production12**（旧） | **Production36**（新，生产路径实测） |
| --- | --- | --- |
| 库大小 / 每帧精匹配数 | 12 / 12 | **36 / 4**（Top-4） |
| `matchTemplate` 调用/帧 | 60 | **20（−67%）** |
| TP / FN | 41 / 57 | **81 / 17** |
| **LOO** TP / FN | 29 / 69 | **56 / 42** |
| FP / TN | 0 / 80 | **0 / 80** |
| candidate recall@4 | — | **1.0000**（81/81） |
| bbox 越界 / 选中模板差异 | 0 / — | **0 / 0** |
| prefilter fallback | — | **0** |
| `detect()` p50 / p90 / p95 | — | **45.1 / 51.5 / 53.6 ms** |

* 全部数字来自**生产 detector**（`BottomMenuYardDetector.from_config` + `detect()`），不是离线分数表推导；
* 相对旧配置：**多命中 41 张、少命中 1 张**（那 1 张是 `yard_cap_001` 的**自匹配**，见 §3）；
* 恢复旧行为只需按 §15 的 rollback 配置改回 12 张 —— 已由自动化测试验证。

---

## 1. 精确模板选择的设计

### 1.1 现有加载顺序（先审查后编码）

``load_feature_templates`` 的实际顺序：

```text
目录解析 → sorted(*.png) 目录发现 → 文件名前缀/尾部解析（source/num）→ _imread
        → dedupe_templates(dedupe_threshold)      ← dHash 去重
        → limit_per_source(per_source_limit)      ← 每来源限量
        → _sample_round_robin / _sample_evenly(limit)   ← 抽样到 max_templates
```

* ``max_templates`` = **最终加载库大小上限**（不是每帧精匹配数量）；
* ``round_robin`` = **加载阶段的来源均衡策略**（与运行期 ``prefilter_top_k`` 完全无关）。

### 1.2 为什么选 "stable template id" 而不是 "source_id"

| 方案 | 事实 | 结论 |
| --- | --- | --- |
| `source_id` | ``positive_sources.json`` **只覆盖 33 张 skin 模板**；28 张 cap 模板**没有来源映射**（须靠命名规则 + SHA256 推导） | **不满足"可审计"** |
| **`template_id`（文件名去扩展名）** | 61 张全部唯一、稳定、人类可读；skin 部分还能与 json 的 ``template_name`` 交叉验证 | **采用** |

> 用户规则允许："如果一个 source_id 对应多个候选模板而无法唯一选择，则改用 stable template identifier"。
> 本次的对应情形是"source_id 对部分模板**不可得**"，同样落到 stable identifier。
> 完整映射表见 §2（含每张的 ``source_id`` 与 ``from_image``）。

### 1.3 选择器语义

```yaml
template_include: [<stable template id>, ...]   # 空列表（默认）= 不做显式过滤，旧行为完全不变
```

**非空时**：

```text
目录 templates
   ↓ explicit include filter（按声明顺序精确选取；任何 selector 无法解析 ⇒ 直接报错）
   ↓ [跳过] dedupe / per_source_limit / round_robin
   ↓ max_templates 作为安全上限（集合大于上限 ⇒ 报错，不静默截断）
最终集合（顺序 = 声明顺序）
```

**为什么跳过 dedupe**（本阶段必须专门回答的问题）：实测 **36 张 + `dedupe_threshold=6` ⇒ 35 张**
（会移除 ``skin006_005``）。显式集合的语义是"我就要这些"，再自动去重会**破坏可复现性**。
因此显式模式**跳过**去重/限量/抽样，并且配置校验**强制**这三者为关闭值（0/0/false），
避免"字段写了但没生效"的含糊配置。

### 1.4 完整性校验（全部 fail-fast）

| # | 校验 | 位置 | 失败行为 |
| --- | --- | --- | --- |
| 1 | selector 形态（只允许字母/数字/下划线） | config `field_validator` | `ValidationError` |
| 2 | selector 不重复 | config `field_validator` | `ValidationError` |
| 3 | 显式模式下 dedupe/per_source/round_robin 必须关闭 | config `model_validator` | `ValidationError` |
| 4 | 集合不得大于 `max_templates` | config `model_validator` | `ValidationError` |
| 5 | selector 全部能解析（模板存在） | `_select_explicit_templates` | `EvidenceTemplateLoadError` |
| 6 | 目录内 `template_id` 唯一 | `_select_explicit_templates` | `EvidenceTemplateLoadError` |
| 7 | 集合不得大于 `limit` | `_select_explicit_templates` | `EvidenceTemplateLoadError` |
| 8 | 路径越界（``../``、分隔符） | 由 1 覆盖（selector 不是路径） | `ValidationError` |

⇒ 绝不"静默加载 35 或 37 张后继续运行"。

---

## 2. 实际加载的 36 张模板（完整列表与来源映射）

``idx  template_id | skin | source_id | from_image``

| # | template_id | skin | source_id | from_image |
| --- | --- | --- | --- | --- |
| 1 | bottom_menu_yard_skin001_001 | skin001 | src_30c19de243049464 | raw/yard/yard_skin001_001.png |
| 2 | bottom_menu_yard_skin001_003 | skin001 | src_8f6abbf8f9b4aec8 | raw/yard/yard_skin001_003.png |
| 3 | bottom_menu_yard_skin001_005 | skin001 | src_bfd7c72f1766c200 | raw/yard/yard_skin001_005.png |
| 4 | bottom_menu_yard_skin002_001 | skin002 | src_a067ae318923bc14 | raw/yard/yard_skin002_001.png |
| 5 | bottom_menu_yard_skin002_003 | skin002 | src_2a505a150765dcc7 | raw/yard/yard_skin002_003.png |
| 6 | bottom_menu_yard_skin002_005 | skin002 | src_4c0e825b0dcec737 | raw/yard/yard_skin002_005.png |
| 7 | bottom_menu_yard_skin003_001 | skin003 | src_ee9b81ee83b8dd9a | raw/yard/yard_skin003_001.png |
| 8 | bottom_menu_yard_skin003_003 | skin003 | src_6887997be8dfd4a2 | raw/yard/yard_skin003_003.png |
| 9 | bottom_menu_yard_skin003_004 | skin003 | src_52b0284458555f54 | raw/yard/yard_skin003_004.png |
| 10 | bottom_menu_yard_skin003_005 | skin003 | src_3cb44d53ecf79e76 | raw/yard/yard_skin003_005.png |
| 11 | bottom_menu_yard_skin004_001 | skin004 | src_c8d50df97d2d5c2e | raw/yard/yard_skin004_001.png |
| 12 | bottom_menu_yard_skin004_003 | skin004 | src_6d9a9770f56609b2 | raw/yard/yard_skin004_003.png |
| 13 | bottom_menu_yard_skin004_005 | skin004 | src_b6a7a6662838939c | raw/yard/yard_skin004_005.png |
| 14 | bottom_menu_yard_skin005_001 | skin005 | src_897e81d520c21cba | raw/yard/yard_skin005_001.png |
| 15 | bottom_menu_yard_skin005_005 | skin005 | src_fbc6b0d4f1bd4317 | raw/yard/yard_skin005_005.png |
| 16 | bottom_menu_yard_skin006_001 | skin006 | src_c98e153c0efd84e4 | raw/yard/yard_skin006_001.png |
| 17 | bottom_menu_yard_skin006_003 | skin006 | src_29319c627574e324 | raw/yard/yard_skin006_003.png |
| 18 | bottom_menu_yard_skin006_005 | skin006 | src_46b0706b0db124f2 | raw/yard/yard_skin006_005.png |
| 19 | bottom_menu_yard_skin007_001 | skin007 | src_6d138f7ff6f936a9 | raw/yard/yard_skin007_001.png |
| 20 | bottom_menu_yard_skin007_003 | skin007 | src_fb05d8881a20df69 | raw/yard/yard_skin007_003.png |
| 21 | bottom_menu_yard_skin008_001 | skin008 | src_c02d3ff680f34253 | raw/yard/yard_skin008_001.png |
| 22 | bottom_menu_yard_skin008_003 | skin008 | src_b760c4994e0351a4 | raw/yard/yard_skin008_003.png |
| 23 | bottom_menu_yard_skin008_005 | skin008 | src_9dd8b6760956202e | raw/yard/yard_skin008_005.png |
| 24 | bottom_menu_yard_skin009_001 | skin009 | src_fc33f404603c1872 | raw/yard/yard_skin009_001.png |
| 25 | bottom_menu_yard_skin010_001 | skin010 | src_1bdb879a150f5916 | raw/yard/yard_skin010_001.png |
| 26 | bottom_menu_yard_skin011_001 | skin011 | src_0cf0563e18f83e27 | raw/yard/yard_skin011_001.png |
| 27 | bottom_menu_yard_skin012_001 | skin012 | src_9cb6b85aab584c1d | raw/yard/yard_skin012_001.png |
| 28 | bottom_menu_yard_skin013_001 | skin013 | src_41ce5bdfa708edb9 | raw/yard/yard_skin013_001.png |
| 29 | bottom_menu_yard_skin014_001 | skin014 | src_be4c05746b4ba8e0 | raw/yard/yard_skin014_001.png |
| 30 | bottom_menu_yard_skin015_001 | skin015 | src_c6fb7bd3dab955d9 | raw/yard/yard_skin015_001.png |
| 31 | bottom_menu_yard_skin016_001 | skin016 | src_491e8627c3803afa | raw/yard/yard_skin016_001.png |
| 32 | bottom_menu_yard_skin017_001 | skin017 | src_317ca460781823fd | raw/yard/yard_skin017_001.png |
| 33 | bottom_menu_yard_skin018_001 | skin018 | src_d441d3daf96f7924 | raw/yard/yard_skin018_001.png |
| 34 | bottom_menu_yard_cap_006 | cap | src_809bc978eb2186a3 | raw/yard/yard_cap_006.png ★ |
| 35 | bottom_menu_yard_cap_003 | cap | src_f738392e406aa9ce | raw/yard/yard_cap_003.png ★ |
| 36 | bottom_menu_yard_cap_007 | cap | src_267221ddea83c7f9 | raw/yard/yard_cap_007.png ★ |

★ = cap 模板的来源映射不在 ``positive_sources.json`` 中，``from_image``/``source_id``
由**命名规则 + SHA256** 推导（``bottom_menu_yard_cap_006`` ↔ ``raw/yard/yard_cap_006.png``），
工具输出中标记 ``[推导]``。33 张 skin 的映射直接取自 json，**未做任何推导**。

覆盖：**18/18 种皮肤** + 1 个 cap 来源；61 张目录中实际使用 36 张（其余 25 张 cap **未使用**、也**未删除**）。

---

## 3. include 与 dedupe 的语义（本阶段专门回答）

| 问题 | 实测/设计 |
| --- | --- |
| 36 张 + `dedupe_threshold=6` 会剩多少？ | **35 张**（移除 `bottom_menu_yard_skin006_005`） |
| 显式模式下 dedupe 是否生效？ | **不生效**（跳过）；即使显式传入 `dedupe_threshold=6` 也不改变集合 |
| 为什么？ | 显式集合是"人工/benchmark 已筛选的最终结果"，再自动去重会破坏可复现性 |
| 配置层如何防止误用？ | `model_validator` 强制"显式模式 ⇒ dedupe/per_source/round_robin 必须关闭" |
| 预期 = 实际？ | **36 = 36** ✅（`test_explicit_loads_exactly_36`） |

---

## 4. Production12 基线（rollback 配置，生产路径实测）

| 指标 | 值 |
| --- | --- |
| 模板 / 每帧精匹配 | **12 / 12**（`cap_001` + `skin001-011`） |
| `matchTemplate` 调用/帧 | **60** |
| TP / FN | **41 / 57**（recall 0.4184） |
| LOO TP | **29**（Phase 11.5-B/11.7-B 同值） |
| FP / TN | **0 / 80**（precision 1.0） |
| prefilter | `None`（未构造） |
| fallback | 0 |

---

## 5. Production36 结果（生产路径实测，98 正 + 80 负）

| 指标 | 值 |
| --- | --- |
| 模板 / prefilter / top_k | **36 / thumb / 4** |
| `matchTemplate` 调用/帧 | **20** |
| TP / FN | **81 / 17**（recall 0.8265） |
| FP / TN | **0 / 80**（precision 1.0） |
| prefilter fallback | **0** |
| 每帧候选上限 | **4** |

**相对 Production12 的逐图翻转**：`positive_to_negative = 1`、`negative_to_positive = 41`、合计 42。

* 唯一退化：``raw/yard/yard_cap_001.png``（旧配置由 ``cap_001`` **自匹配**命中，新库不含 ``cap_001``）；
* 41 张新增覆盖：`cap_003/006/007` 与 `skin012-018` 等；
* 净变化 41 − 1 = **+40 = 81 − 41** ✅ 与 TP 差自洽。

---

## 6. LOO（按真实加载库、排除同源模板）

| 配置 | 普通 TP | **LOO TP** | LOO recall |
| --- | --- | --- | --- |
| Production12 | 41 | 29 | 0.2959 |
| **Production36** | 81 | **56** | **0.5714** |

* LOO 用生产 detector **实际加载的 36 张**做分数表并排除 ``own_template_id``（同源模板）；
* 与 Phase 11.7-B 的预测 **完全一致（56/98）** ⇒ 显式集合、加载顺序、来源映射都没有偏差。

---

## 7. canonical 80 负样本与阈值安全边际

| 分组 | 张数 | max | p50 | p90 | p95 |
| --- | --- | --- | --- | --- | --- |
| **全 80 张** | 80 | **0.7793** | 0.4108 | 0.5960 | 0.7250 |
| `raw/unknown/friends`（困难） | 2 | **0.7793** | 0.7793 | 0.7793 | 0.7793 |
| `raw/unknown/other` | 22 | 0.7591 | 0.4087 | 0.7197 | 0.7322 |
| `raw/unknown/shop`（困难） | 4 | 0.6981 | 0.4372 | 0.6419 | 0.6700 |
| `raw/exploration` | 36 | 0.5436 | 0.3994 | 0.4824 | 0.4988 |
| 其余（collection/equipment/shikigami/battle×3） | 15 | ≤0.5919 | — | — | — |

* **FP = 0 / TN = 80**；**超阈值 0 张**，距阈值 **0.0707**；`threshold` 仍为 **0.85**（未改）；
* 口径为**递归 80 张**（含 `raw/unknown/*` 子目录），**未退回** 11.6-B/C 的 41 张非递归错误口径；
* 与 11.7-B（33 张库）**逐位相同** ⇒ 加入 3 张 cap 模板没有抬高任何负样本分数。

---

## 8. 生产路径 candidate recall@4

用**生产 detector + 生产 `TemplatePrefilter` 实例**（不是实验工具）：

| K | recall | 说明 |
| --- | --- | --- |
| **@4** | **1.0000** | 81 张命中图全部覆盖（最佳模板 rank p50 = 1.0、max = **2**） |
| @6 | 1.0000 | |
| @8 | 1.0000 | |

同时检查：实际加载模板数 **36** ✅、每帧候选 **≤4** ✅、`matchTemplate` 调用 **≤20** ✅、
额外 fallback 次数 **0** ✅。

---

## 9. 性能（生产 detector，`detect()` 全路径）

| 指标 | **Production36 + Top-4** | 参照 |
| --- | --- | --- |
| `matchTemplate` 调用/帧（**确定性**） | **20** | 12 张 = 60；36 张 naive = 180 |
| `detect()` p50 | **45.1 ms** | — |
| `detect()` mean | 44.7 ms | — |
| `detect()` p90 | **51.5 ms** | — |
| `detect()` p95 | **53.6 ms** | — |
| `best_match()` 单项 p50（11.7-B 口径） | 29.9 ms | 36 张 naive ≈ 256 ms |

* **整链推算** = 其它 detector + 状态判定基线 282ms + 45.1ms ≈ **327ms**（500ms 余量 **35%**）；
* wall-clock 波动明显（同一配置三次 `detect()` p50 = 42.1 / 46.0 / 45.1 ms，±5–10%；
  同批次的 12 张 naive 曾测得 67.8 / 88.5 ms，±30%）⇒ **决策依据是确定的 20 次调用/帧**，
  而不是"某一次 <500ms"；
* 与 11.7-B 的 29.9ms 差异来自**口径不同**（那里只测 `best_match`，这里测 `detect()` 全路径）
  与机器负载，两者都如实列出，不做取舍。

---

## 10. bbox 一致性与坐标安全

| 项 | 值 |
| --- | --- |
| 命中带 bbox | **81** |
| bbox 越界（相对 ROI） | **0** |
| bbox 落在帧外 | **0** |
| 与"同库 naive 全量匹配"的 IoU p50 / min | **1.0 / 1.0** |
| 中心误差 max | **0.0 px** |
| 选中模板不同 | **0** |

⇒ 预筛只改变了"匹配了哪些模板"，**没有**改变坐标语义；bbox 仍由生产 ``_restore_rect`` + clamp 产生。

---

## 11. fallback 次数与原因

| 场景 | 次数 | 说明 |
| --- | --- | --- |
| **正常扫描（98 正 + 80 负）** | **0** | 包含 17 张未命中正样本与 80 张负样本，**均未**触发 fallback |
| 预筛异常（测试注入 `RuntimeError`） | 1 | 回退全量匹配 + `metadata.prefilter_fallback/_reason` |

之所以重要：若"Top-4 没超过阈值"也去跑全量，则**非 yard 帧会把省下的成本全部付回去**。
本阶段保持 11.7-B 语义（只有预筛**自身故障**才回退）。

---

## 12. Rollback（实测，非文档承诺）

rollback 配置（内存副本，走生产装配路径）：

```yaml
bottom_menu_yard:
  template_include: []      # 清空显式集合
  max_templates: 12
  round_robin: true
  dedupe_threshold: 6
  per_source_limit: 0
  roi_scale: 0.75
  template_prefilter: none
  prefilter_top_k: 0
```

| 断言 | 结果 |
| --- | --- |
| 加载模板数 | **12** ✅（`cap_001` + `skin001-011`，顺序与旧配置一致） |
| prefilter 是否构造 | **未构造**（`detector.prefilter is None`） ✅ |
| TP / FP / TN | **41 / 0 / 80** ✅（与旧 Production12 一致） |
| `matchTemplate` 调用/帧 | **60** ✅ |
| metadata | **无任何 prefilter 字段** ✅ |
| fallback | 0 ✅ |

⇒ 回滚**不依赖** ``prefilter_top_k`` 的特殊解释，而是靠 ``max_templates: 12`` 恢复旧库
（``prefilter_top_k=0`` 的语义始终是"不做预筛"）。

---

## 13. 其它 detector 与安全边界

| 项 | 结果 |
| --- | --- |
| 其它 detector 的 `template_include` | 全为 `[]`（未使用显式集合） |
| 其它 detector 的 prefilter | 全为 `none` / `0`；装配后 `prefilter is None` |
| `action.enabled` | **false**（未因 yard 召回提升而启用真实输入） |
| `action.backend` | **dry_run** |
| `raw` 178 张 SHA256 | **未变** |
| `dataset/annotations` | **未变** |
| 模板目录 | **仍 61 张**，无新增/删除/复制（未创建第二份 36 张目录） |

---

## 14. 测试结果

```text
$ python -m pytest -q                 → 见下方（1680 → +38）
$ python tools/check_dataset.py       → PASS（仅 warning）
$ python tools/raw_manifest.py        → raw 178 张 SHA256 未变
$ python tools/check_annotations.py   → PASS
$ python tools/verify_yard_production_prefilter.py --repeat 2 --warmup 1   → 见 §4–§12
```

新增 ``tests/test_phase117c_yard_prefilter_production.py``（**38 项**，覆盖用户要求的 40 点）：
selector 默认关闭 / 36 个 selector 可解析 / 恰好 36 张 / 无重复 / 来源映射可追溯 /
越界与非法字符拒绝 / 不存在 selector 拒绝 / 重复 selector 拒绝 / `max_templates` 与 selector 职责分离 /
`prefilter_top_k` 与 `max_templates` 职责分离 / 显式集合不被重新抽样 / 与 dedupe 的语义 /
生产配置（thumb / top4 / 0.85 / 0.75 / 36 张）/ 生产路径 TP 81 / LOO 56 / candidate recall@4 = 1.0 /
calls ≤ 20 / canonical 80 负样本 / FP = 0 / 正常未命中不 fallback / 预筛异常才 fallback /
bbox 越界 0 / 与 naive36 bbox 一致 / rollback 12 张 + 预筛关闭 + 与旧 Production12 一致 /
其它 detector 不受影响 / metadata 仅在启用时含诊断字段 / 安全配置 / 数据只读 / AST 边界 / 输出写 TEMP。

**同时更新了 4 个既有测试文件**中"硬编码旧生产配置"的断言（保持原意、不降低强度）：
`test_phase115b`（3 项）、`test_phase116c`（2 项）、`test_phase117b`（4 项），
并把 11.7-A/B 工具中的 **Production12 基线固定为常量**（不再随当前生产配置漂移，
保证历史报告的数字仍可复现）。**未删除任何测试、未新增任何 skip。**

---

## 15. 未解决问题

| 编号 | 内容 |
| --- | --- |
| U-1 | **跨分辨率未验证**：raw 全为 1703–1706 × 960–963；本阶段**不做**合成缩放冒充（留 Phase 11.8） |
| U-2 | **只接入 `bottom_menu_yard`**；其它 detector 需各自 benchmark 后再决定 |
| U-3 | 唯一退化：`yard_cap_001` 的自匹配命中消失（净收益 +40，但**该单帧**识别率下降） |
| U-4 | 显式集合把"选哪些模板"变成**手工清单**：模板库演进需要一个可复现的重新选择流程（贪心脚本已具备，但未产品化） |
| U-5 | wall-clock 波动 ±5–30%，未做长时间稳定性观测（本阶段不启用真实输入） |
| U-6 | `prefilter_top_k = 4` 的冗余度：本阶段数据里 rank max = 2，但样本仅 178 帧；若未来换库建议复核 Top-8 |
| U-7 | 25 张未使用的 cap 模板仍在目录中（未删除），需要时可由新的显式集合纳入 |

---

## 16. Phase 11.8 建议

1. **跨分辨率 / 多 DPI 验证**：补采真实数据（不用合成缩放），重跑 candidate recall@4 与 LOO；
2. 把"模板集合的重新选择流程"产品化（贪心覆盖 + LOO 饱和判定，输出可直接粘贴的 ``template_include``）；
3. 评估 `roi_scale: 0.5`（11.6-C 的 R-1 遗留）—— 现已在 36 张 + Top-4 下留有余量，但需权衡 bbox 精度；
4. 其它 detector 的 prefilter 评估（先 benchmark，再逐个启用，方法与本阶段一致）；
5. 把 ``prefilter`` 命中率 / fallback 纳入运行期观测（``metadata`` 已有字段，需要一个聚合出口）；
6. 真实运行期长跑（仍保持 ``action.enabled=false``），观测命中率与节拍余量。

---

## 17. Git commit

```text
feat(vision): enable yard template prefilter in production
```
