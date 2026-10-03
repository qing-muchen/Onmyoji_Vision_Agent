# Phase 11.9-C — Production Switch（PROPOSED_36 → 生产）

> 目标：把 11.9-B 冻结的 `PROPOSED_36_FROZEN` **正式写入生产配置**。
> **必读措辞（本报告的唯一正确表述）**：
> **"s04 confirmed no regression, but was not discriminative for incremental recall."**
> —— **不得**表述为"blind 证明 proposed 提升了召回"。

---

## 1. 切换前保护

| 项 | 结果 |
| --- | --- |
| `git status --short`（切换前） | 唯一未提交：` M tools/build_annotation_review_pack.py`（**未修改、未 stage、未 commit**） |
| `git diff --stat`（切换前） | 1 file, +2/−1（同上） |
| HEAD（切换前） | `1f12aea feat(dataset): add blind holdout s04 and one-shot acceptance` |
| **rollback source** | `current36` 的 36 个 `template_include` id（完整记录，见 §18） |
| 配置快照 | `TEMP/ova119c/config_before_119c.yaml`（切换前 yaml 完整副本） |
| rollback JSON | `TEMP/ova119c/rollback_current36.json`（`rollback_source` / `proposed_36` / `added` / `removed` / 快照路径） |

## 2. 生产配置修改（只改 `configs/config.yaml`）

* 只替换 `evidence.detectors.bottom_menu_yard.template_include` 为 `PROPOSED_36_FROZEN` 的 36 个 id；
* 在块前追加 Phase 11.9-C 说明注释（含 `freeze_hash`、数据依据、**s04 无区分度**警示、rollback 路径）；
* 保持不变：`threshold: 0.85`、`roi_scale: 0.75`、`template_prefilter: thumb`、
  `prefilter_top_k: 4`、`max_templates: 36`、`dedupe_threshold: 0`、`per_source_limit: 0`、
  `round_robin: false`、`action.enabled: false`、`backend: dry_run`；
* **未修改**：matcher、threshold、roi_scale、top_k、prefilter 算法、`FIT_TOLERANCE_PX`、安全开关；
* 变更规模：`configs/config.yaml` **+14 / −7**（仅该文件）。

> ⚠️ 记录一次**过程失误**：首次替换脚本误把 `template_include:` **这一行 key 本身**也纳入删除范围，
> 导致 YAML 解析失败（`expected <block end>, but found block sequence start`）。
> 已**从快照恢复**后用"保留 key 行 + 8 空格列表缩进"重做，并通过 `yaml.safe_load` + `load_config()`
> 双重校验。教训：结构化文本替换必须保留锚点行本身。

## 3. 配置验证

| 检查 | 结果 |
| --- | --- |
| `yaml.safe_load` + `load_config()` | **OK** |
| `template_include` 数量 | **36** |
| 配置内 id == `PROPOSED_36_FROZEN` | **True** |
| **`cap_028` 已进入生产** | **True** |
| `cap_006` 已离开生产 | True（`False` in include） |
| 生产 detector（**经 `template_include` 加载路径**）| `loaded templates = 36`、`prefilter = thumb`、`top_k = 4` |
| **loaded ids checksum** | `619bbc9dd44cca8ba7648e3816f5aa5751b59ca26552fdfb17450ba9869bede0` |
| **frozen checksum** | 同上（**完全一致** ✅） |
| rollback 可恢复 | 以 `rollback_source` 构造 detector ⇒ 加载 **36** 张、id 逐项一致 ✅（**未实际回滚**） |

## 4. 完整回归

| 命令 | 结果 |
| --- | --- |
| `python -m pytest -q` | **1858 passed, 1 skipped**（≥ 基线 1858 ✅） |
| `python tools/check_dataset.py` | **PASS**（`yard_live total: 29`） |
| `python tools/raw_manifest.py` | `ok: true` · 历史 **178/178** · live **29/29** · `missing/added/changed` 全空 |
| `python tools/check_annotations.py` | **PASS** |

历史 raw **unchanged**；live **unchanged**（s01 5 / s02 8 / s03 8 / s04 8）。

## 5. 生产路径验证（**走生产 detector，不是离线脚本**）

| 指标 | 切换后 | 冻结结果 | 要求 | 判定 |
| --- | --- | --- | --- | --- |
| historical **TP / FN** | **82 / 16** | 82 / 98 | ≥ 82 | ✅ |
| historical **LOO TP / FN** | **58 / 40** | 58 / 98 | ≥ 58 | ✅ |
| canonical80 **FP / TN** | **0 / 80** | 0 / 80 | = 0 | ✅ |
| candidate recall@4 | **1.0**（hit 82，best_rank p50 1.0 / max 2） | 1.0 | 不退 | ✅ |
| negative raw max / p95 / gap | **0.7793 / 0.7250 / 0.0707** | 0.7793 | 标注 | ✅ 不变 |
| negative `above_threshold` | **0** | 0 | 0 | ✅ |
| 最难负样本组 | `unknown/friends` 0.7793 · `unknown/other` 0.7591 · `unknown/shop` 0.6981 | — | — | 与切换前一致 |

## 6. Latency（交替 benchmark）

⚠️ **必须交替测量**：单侧顺序测量会给出虚假回归（11.9-B 曾出现 +48%）；轮次少时亦有 ±10% 波动
（4 轮曾读到 +6.6%）。

**8 轮交替（各 1424 样本，双方预热 80 帧）**：

| 集合 | p50 | p90 | p95 | mean |
| --- | --- | --- | --- | --- |
| current36 | 26.98 ms | 40.20 | 44.56 | 29.46 |
| **proposed36（生产）** | **26.94 ms** | **37.01** | **39.69** | **28.71** |
| **delta** | **−0.1%** | **−7.9%** | **−10.9%** | **−2.5%** |

同时保持：`candidate_count_max = 4` · `match_template_calls_per_frame = **20**` ·
`fallback_count = **0**`（生产路径实测）。

## 7. Rollback 验证（只验证可恢复，**未真实回滚**）

* 从 `rollback_current36.json` 的 `rollback_source` 构造 detector ⇒ **加载 36 张、id 逐项一致** ✅；
* rollback checksum：`06b7c2a195adf83dac036fa8c310b70f50729e387c394bcd285a46035f46bdd0`；
* 恢复方式：把 `configs/config.yaml` 的 `template_include` 换回该 36 个 id，或直接恢复
  `TEMP/ova119c/config_before_119c.yaml`。

## 8. 最终报告（18 项）

| # | 项 | 值 |
| --- | --- | --- |
| 1 | commit hash | `feat(vision): switch yard production templates to proposed36`（见 `git log`） |
| 2 | production template count | **36** |
| 3 | proposed36 hash（freeze） | `7851a45ebc5f9252c9bd4cca38c8063298bf87cef0341513b9aa8f92eabb00c5` |
| 4 | current36 rollback hash（ids checksum） | `06b7c2a195adf83dac036fa8c310b70f50729e387c394bcd285a46035f46bdd0` |
| 5 | loaded template ids checksum | `619bbc9dd44cca8ba7648e3816f5aa5751b59ca26552fdfb17450ba9869bede0`（== freeze） |
| 6 | historical TP / FN | **82 / 16** |
| 7 | historical LOO TP / FN | **58 / 40** |
| 8 | canonical FP / TN | **0 / 80** |
| 9 | s03 / s04 状态说明 | s03 candidate 8/8（训练内）；**s04 blind 8/8，且 current36 亦 8/8** ⇒ 见下方措辞 |
| 10 | latency current vs proposed | p50 **26.98 → 26.94**（−0.1%）、p95 44.56 → **39.69**（−10.9%） |
| 11 | candidate_count | **4**（恒 = top_k） |
| 12 | match calls | **20 / frame** |
| 13 | fallback | **0** |
| 14 | bbox | 命中 82、**越界 0**、IoU p50/min **1.0**、中心误差 max 0.0、模板选择差异 0 |
| 15 | pytest | **1858 passed, 1 skipped** |
| 16 | dataset integrity | 历史 raw **unchanged**、live **unchanged**（s04 added only 已入库清单）、annotations PASS |
| 17 | safety | `action.enabled=false` · `backend=dry_run`；采集/工具链未引入任何输入控制 |
| 18 | rollback 方式 | 见 §7（配置快照 + `rollback_source` 36 id，可一键恢复） |

### 关于 s04 的唯一正确表述

> **s04 confirmed no regression, but was not discriminative for incremental recall.**
>
> 增益主要来自 **historical LOO（56→58）** 与 **candidate 集合（s03 7/8→8/8）**；
> blind holdout s04 确认了**无退化**，但 current36 在同一批上同样 8/8，因此该批次
> **不能**证明 proposed36 带来增量召回。**不得**表述为"blind 证明提升"。

### 生产切换要点（给后续阶段的提醒）

* 生产阈值 / ROI 缩放 / 预筛 / 安全开关**均未改动**，切换只换模板集合；
* 若后续要再次重选模板，必须重新走 11.9-B 的"candidate-only + freeze + 一次性 validation"
  纪律，并**另采**新的 candidate（`s04` 已永久保留为 blind holdout，**不得**作为模板来源）。
