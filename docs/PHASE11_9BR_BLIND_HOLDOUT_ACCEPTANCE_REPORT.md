# Phase 11.9-BR — Blind Holdout Acquisition and One-Shot Acceptance

> 目标：在 **`PROPOSED_36_FROZEN` 完全冻结**的前提下，采集一个**全新** session
> （`yard_live_20261003_s04`）作真正 blind holdout，做**一次、不可回溯调参**的 blind 验收，
> 以判断 11.9-B 的模板重选是否具备**跨 session 泛化**能力。
> 本阶段**不修改生产配置**（切换到 Phase 11.9-C）。

---

## 1. Blind 纪律（执行前锁定）

冻结后不得修改：proposed36 template ids、selection algorithm、threshold、roi_scale、top_k、
prefilter、matcher、`FIT_TOLERANCE`、historical/candidate objective。**不得**因 s04 的任何结果
重新进入 selection loop。第一次正式 s04 probe **只使用 Frozen Proposed36**；在 blind 结果
完整落盘前**禁止**运行 All61。

**执行前的可复现性校验**（开始前检查）：

| 项 | 结果 |
| --- | --- |
| `freeze_hash`（stored / recomputed / expected） | `7851a45ebc5f9252c9bd4cca38c8063298bf87cef0341513b9aa8f92eabb00c5` **三者一致** ✅ |
| `PROPOSED_36_FROZEN` | `status = PROPOSED_36_FROZEN`、36 张 |
| s01 / s02 / s03 | 5 / 8 / 8 张（未变） |
| s04 | **采集前不存在** ✅ |
| 生产 `template_include` | 仍 **36**（current36） |
| 历史未提交文件 | `tools/build_annotation_review_pack.py` —— 未修改、未 stage、未 commit |

## 2. s04 采集（用户手动，程序只截图）

* session：`yard_live_20261003_s04`；`role = holdout`；`variant = 茨心乐园`；
* **一次 1 张**（`--count 1` + `--wait-foreground 1800` + `--settle-seconds 12`），
  每张由用户手动改变自然状态后触发；
* 覆盖序列：静态庭院 → 角色不同位置 → 不同背景动画相位 → 式神/NPC 状态 → 不同镜头方向 →
  主题装饰另一动画阶段 → 底部菜单清晰但上半屏变化 → 自然非极端构图；
* **未**为拉大 dHash 而遮挡画面或制造极端角度；
* 全程 `action.enabled=false` / `backend=dry_run`；采集工具只截图，未点击、未键盘输入、
  未 `SetForegroundWindow`、未自动导航、未自动切换皮肤。

## 3~5. 数据质量审计（**未运行任何视觉评分**）

| 项 | 结果 |
| --- | --- |
| s04 样本数 | **8**（全部 **1704×960**） |
| **SHA 重复** | **0**（s04 8/8 唯一；live 29/29 唯一） |
| **source leakage** | **False**（`hard_failures = []`） |
| `exact_duplicate_count` | **0** ✅ |
| **dHash overlap warnings** | `cross_session_near_duplicate_count = 7`（`cross_session_dhash_min = 3`）⇒ **possible_visual_overlap（warning only，不删除、不要求重采）** |
| s04 内部 dHash 最小距离 | **11**（28 对中无 ≤6） |
| session metadata | 8/8 条均为 `session_id=yard_live_20261003_s04`、`role=holdout`、`yard_variant=茨心乐园`、`client=1704×960`、`backend=dxcam`、`captured_at` 连续递增（23:20:22 → 23:26:10） |
| 审计 `status` | **`ok`** |
| manifest | 历史 **178/178**；live **29/29**（s01:5 s02:8 s03:8 s04:8）；`missing/added/changed` 全空 |

## 6~7. Blind 结果文件

```
freeze_hash        = 7851a45ebc5f9252c9bd4cca38c8063298bf87cef0341513b9aa8f92eabb00c5
template_set_hash  = 619bbc9dd44cca8ba7648e3816f5aa5751b59ca26552fdfb17450ba9869bede0
文件               = TEMP/ova119b/phase119b_s04_blind_acceptance.json
文件 sha256（前 32）= 0abc70070b75b8aa7a59e3456b94ef08…
status             = BLIND_RESULT_FROZEN   （一经写出不得覆盖）
probe_scope        = Frozen Proposed36 ONLY (no current36, no All61)
```

文件含：`freeze_hash` / `template_set_hash` / `session_id` / 每张样本 SHA256 /
`threshold=0.85` / `roi_scale=0.75` / `top_k=4` / `prefilter=thumb`，以及逐帧
`detected` / `confidence` / `selected_template` / `selected_scale` / `candidate_count` /
`fallback` / `fit_applied` / `bbox_valid`。

## 8~13. Blind 验收结果（Frozen Proposed36，一次性）

| 指标 | 值 |
| --- | --- |
| **8. TP / FN** | **8 / 0** |
| **9. recall** | **1.0** |
| **10. confidence min / p50 / p90 / max** | **0.8765 / 0.8895 / 0.8970 / 0.8985** |
| **11. candidate_count 分布** | `{4: 8}`（恒定 = top_k，未超预算） |
| **12. fallback** | **0** |
| **13. bbox 越界** | **0**（全部 `bbox_valid = True`） |
| latency p50 / p90 | 27.14 / 32.38 ms |
| 逐帧命中模板 | **8/8 全部由 `cap_028` 命中**，`scale=0.75`，`fit_applied=True` |

**解释（按既定框架）**：8/8 ⇒ **较强泛化支持**（该档为 7/8 或 8/8）。

## 14~17. Post-hoc 对比（blind 冻结后才运行）

| 集合 | s04 TP / FN | 命中帧置信度区间 |
| --- | --- | --- |
| **14. current36** | **8 / 0** | 0.865 ~ 0.883 |
| **8. proposed36** | **8 / 0** | 0.876 ~ 0.898 |
| **15. All61** | **8 / 0** | raw 0.914 ~ 0.938（最佳模板多为 `cap_026`，另有 `cap_009`） |

**16. current36 → proposed36 增益**：**召回增益 = 0**（两者都 8/8），但 **置信度全面上移**
（逐帧 +0.008 ~ +0.021，`p50` 更高），安全余量更大。

**17. proposed36 → All61 剩余差距**：召回上**无差距**（均 8/8）；置信度上 All61 更高
（0.914~0.938 vs 0.876~0.898），因为 All61 可用未入选模板 `cap_026`。

逐帧 A/B/C/D 分类：**全部为 D（all hit）** —— 本批 s04 帧三套模板集合都能命中。

## 18. 每个 FN 的归因

**s04 无 FN**（Proposed36 FN = 0）⇒ 本批不存在 `selection_gap` / `library_gap` / `borderline`。

历史遗留的 FN 归因（沿用 11.9-B 结论，本轮未因 s04 改动任何设定）：

* `s02_004`：`library_gap`（All61 best ≈ **0.7255** < 0.85）⇒ known residual；
* `s02_003` / `s02_008`：`selection_gap`（All61 可过 0.9345 / 0.8987，最佳模板为未入选的 `cap_026`/`cap_020`）。

## 19. 是否支持 Proposed36 跨 session 泛化

**结论必须分成两半，且不可混淆**：

* ✅ **不劣化已被证明**：proposed36 在全部四个集合上都不低于 current36
  （s03 **7/8 → 8/8**、historical LOO **56 → 58**、s02 5/8 → 5/8、s04 8/8 → 8/8），
  且在 s04 / s03 上置信度更高、negative max/p95 完全不变；
* ❌ **正向跨 session 召回增益未被证实**：s02 validation 与 s04 blind **都没有**出现召回提升
  —— s04 更关键，因为 **current36 在本批也 8/8** ⇒ 该批次对"重选增益"**没有区分度**。

⇒ 因此："Proposed36 具备跨 session **稳健性**"成立；"Proposed36 具备跨 session **召回增量**"
**不成立（证据不足）**。不为了达成某一档而调整算法。

## 20. 是否建议进入 Phase 11.9-C 生产切换

**建议进入，但需带着明确的期望管理**：

* **支持切换**：无任何集合退化；historical LOO +2、s03 candidate +1 是真实提升；
  置信度与余量整体上移；安全约束（FP=0、calls=20、fallback=0、bbox 合法、latency 无回归）全部满足；
* **不应声称**：切换会改善"真实场景召回"——blind 数据不支持这一点；
* 若 11.9-C 决定切换，建议**同时保留 rollback 路径**（current36 的 36 个 id 已完整记录），
  并把"s04 无区分度"作为已知限制写入切换说明。

## 21. s04 永久保留（不得污染）

`s04` 已是**永久 blind holdout**：即使某张非常适合做模板，也**不得**在 Phase 11.9-B/C 里
作为模板来源；若未来需要新的 candidate，另采 `s05 candidate`。

---

## 附：本阶段产物

| 类型 | 位置 |
| --- | --- |
| 新数据 | `dataset/raw/yard_live/yard_live_20261003_s04/`（8 张，PNG 不入库，SHA 由 manifest 记录） |
| 元数据 | `dataset/annotations/sources.jsonl`（+8 条 provenance） |
| manifest | `dataset/manifests/raw_sha256.json`（live 29） |
| **blind 结果（冻结）** | `TEMP/ova119b/phase119b_s04_blind_acceptance.json`（`BLIND_RESULT_FROZEN`） |
| **未改动** | `configs/config.yaml`（生产仍 current36）、生产模板 PNG、`positive_sources.json`、历史 raw |
