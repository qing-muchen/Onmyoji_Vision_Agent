# Phase 11.4 — 单人开发复核工作流（Self-Review Workflow）报告

> 目标：本项目由开发者**独立开发**，不存在第二名固定审核者。在**不削弱** Phase 11.3
> 独立审核规则的前提下，补一条**明确标注为非独立**的个人复核路径：
> 让 8 条草稿能被逐条复核、留下可追溯记录，用于**开发期离线对照**，
> 但**永远**不能自动变成真值、`clickable_verified` 或真实输入资格。
>
> **不训练模型、不启用真实游戏输入、不自动批准任何人工标注、不虚构审核人员与标注数据。**

---

## 0. 一句话结论（先读）

本轮交付：**个人复核数据契约 + `self-review` CLI + 审核包个人复核表 + 双口径离线评估 + 校验规则**。

* 保留原有独立审核与 `promote --approve`（唯一真值入口），**未放宽**任何约束；
* 新增 `AnnotationStatus.SELF_REVIEWED` 与 `self_review_*` 审计字段（复核人 / 结论 / 意见 / 时间）；
* `self-review` 命令**没有** `--reviewer` 参数 ⇒ 结构上无法借个人复核伪造独立审核；
* 离线评估**严格区分** `independently_verified`（正式口径）与
  `exploratory_self_reviewed`（开发期诊断，`admissible_as_official_metric=False`）；
* `template_hit` 与 `element_bbox` 的语义不可比约束**照旧**，不因个人复核放宽。

**当前数量（如实）**：等待独立审核 **8** 条；个人复核 **0** 条（工具不代填、本阶段未替开发者做决定）；
**经独立审核真值 0 条**；`clickable_verified` **0 条** ⇒ 定位指标仍全部 `insufficient_data`。
`action.enabled=false` / `backend=dry_run` 与既有 Action 安全门**未改动**。

---

## 1. 实际源码审查（改动前的问题 → 本轮修复）

| # | 改动前现状 | 依据 | 本轮处置 |
| --- | --- | --- | --- |
| 1 | 只有"独立审核"一条路径；单人开发时无法留下任何复核记录 | `tools/annotate.py` 仅有 `review` / `revise` / `promote`，且都拒绝自审 | 新增 `self-review` 命令与 `self_review_*` 字段（**不触碰**审核字段） |
| 2 | 伪造独立审核的口子必须结构性堵死 | `cmd_promote` 强制 `annotator != reviewer` | `self-review` **不提供** `--reviewer`；`promote` / `review` 的自审拒绝**保持**并新增测试 |
| 3 | 无法区分"开发者自己看过"与"独立审核通过" | `is_verified_ground_truth` 只看 `status/review_status/annotator/reviewer` | 新增 `has_self_review` / `is_self_reviewed_only` / `is_independently_verified`；**个人复核不参与真值判定** |
| 4 | 离线评估只有 verified / all 两态，缺"开发期探索"口径 | `evaluate_localization.evaluate(include_all_status=…)` | 新增 `truth_mode` 三口径（`independently_verified` / `exploratory_self_reviewed` / `all`）与 `admissible_as_official_metric` 标记 |
| 5 | 审核包只服务"另一名审核者" | `tools/build_annotation_review_pack.py` 生成 `review_form.md` | 复用同一审核包，**新增** `self_review_form.md`（逐条空栏，工具不预填） |
| 6 | 校验脚本不检查个人复核记录 | `tools/check_annotations.py` 10 项检查 | 新增第 11 项：`self_reviewed` 必须记录完整，且**不得**写独立审核字段 |
| 7 | 修订历史与原始标注必须保留 | Phase 11.3 `revise` 追加 + `supersedes` 链 | 保持不变；`self-review` 只改审计字段，不改 bbox / annotator / region / note |

---

## 2. 个人复核数据契约

### 2.1 新增字段（增量，全部有默认值 ⇒ 旧 JSONL 直接可读）

| 字段 | 含义 |
| --- | --- |
| `self_reviewer` | 复核人（通常是开发者本人；**与 `reviewer` 无关**） |
| `self_review_decision` | 个人复核结论（`SelfReviewDecision`：`approved` / `rejected` / `needs_revision`） |
| `self_review_comment` | 个人复核意见 |
| `self_reviewed_at` | 个人复核时间（ISO 8601） |

新状态：`AnnotationStatus.SELF_REVIEWED`（`self_reviewed`）=  **仅个人复核过**。

**模型层硬校验**：`self_review_decision` 与 `self_reviewed_at` **必须成对出现**
（只有时间没有结论、或只有结论没有时间，一律 `AnnotationError`）。

### 2.2 真值判定（**未放宽**）

```text
is_independently_verified  ← 主判定（Phase 11.4 命名）
  = annotation_status == verified
  AND review_status == approved
  AND review_decision 显式 approved
  AND annotator 与 reviewer 都非空且不是同一人
  AND visibility != absent
is_verified_ground_truth   = is_independently_verified   # 历史名称，向后兼容
is_self_reviewed_only      = has_self_review AND NOT is_independently_verified
```

* 个人复核字段**完全不出现在**主判定里 ⇒ 即使 `self_review_decision=approved`，
  只要没有独立审核者，`is_verified_ground_truth` 仍为 `False`；
* `awaits_review` 现在把 `self_reviewed` 也计入（"个人看过"**仍需**独立审核）；
* `annotation_stats()` 新增 `independently_verified` / `self_reviewed` / `self_reviewed_only`，
  与 `verified` / `clickable_verified` / `absent` 一样**分类计数，不合并**。

---

## 3. CLI 与工作流

```bash
# 个人复核（单人开发；写 self_review_*，不写 reviewer / review_status）
python tools/annotate.py self-review --id <annotation_id> \
    --decision approved|rejected|needs_revision --comment "…" [--developer <名字>]

# 独立审核（唯一真值入口，保持不变）
python tools/annotate.py promote --id <annotation_id> \
    --annotator <A> --reviewer <B> --approve --comment "已核对原图"
```

| 个人结论 | `annotation_status` 变化 | 是否产生真值 |
| --- | --- | --- |
| `approved` | → `self_reviewed`（仍等待独立审核） | **否** |
| `rejected` | **不变**（只记录 `self_review_decision=rejected`） | **否** |
| `needs_revision` | **不变**（只记录 `self_review_decision=needs_revision`） | **否** |

> `annotation_status` 的 `verified` / `rejected` 属于**独立审核语义域**，个人复核**不得**覆盖；
> 命令输出带 `status_effect`（`self_reviewed` 或 `unchanged`）如实回报本次写入对状态的影响。

* 写入方式与 Phase 11.3 一致：**先备份 → 临时文件 → `os.replace` 原子替换**；
* `self-review` **拒绝**已通过独立审核的标注（避免把"独立真值"与"自我复核"写进同一条记录）；
* 修订流程不变：`revise` **追加**新标注（`supersedes_annotation_id` + `revision+1`），
  旧条连同其 `self_review_*` 记录**完整保留**，新条回到 `pending_review`、不带个人复核记录；
* 命令输出显式回报边界：`independent_review_untouched` / `counts_as_ground_truth`。

---

## 4. 离线评估：两种口径严格区分

`tools/evaluate_localization.py`（`--status`）：

| CLI | `truth_mode` | 选中集合 | `admissible_as_official_metric` |
| --- | --- | --- | --- |
| `verified`（默认） | `independently_verified` | **仅**独立审核真值 | **True** |
| `self_reviewed` | `exploratory_self_reviewed` | **仅**个人复核、未经独立审核 | **False** |
| `all` | `all(status)` | 全部草稿（自检） | **False** |

* 两个口径**互斥**（`exploratory` 用 `self_reviewed_only`，天然排除已独立验证的条目）；
* 探索口径报告带 `exploratory_diagnostic=True` +
  "`admissible_as_official_metric=False`，只能作为开发期诊断，严禁计入正式指标或对外宣称准确率"；
* 6 类**不可比原因**与 `template_hit` / `element_bbox` 语义约束**完全不变** ⇒
  个人复核**不会**让模板命中框可以冒充元素外框；
* `include_all_status` 保留为**已弃用**兼容参数（显式给出时覆盖 `truth_mode`）。

**实测（正式数据）**：

```text
$ python tools/evaluate_localization.py                      → independently_verified：真值 0 ⇒ INSUFFICIENT_DATA
$ python tools/evaluate_localization.py --status self_reviewed → exploratory_self_reviewed：样本 0 ⇒ INSUFFICIENT_DATA
$ 副本上做 1 条 self-review 后：
  verified 口径            → 真值仍 0（个人复核**不**进入正式口径）
  self_reviewed 口径        → 命中 1 条样本，但预测全为模板命中框 ⇒
                             incomparable_region_kind ⇒ INSUFFICIENT_DATA（跨语义拒绝比较）
```

---

## 5. 审核包：复用现有包，新增"个人复核表"

`tools/build_annotation_review_pack.py` **不新增包形态**，只在既有审核包里加一份
`self_review_form.md`：

```text
review_pack/
├── README.md              审核须知 + **单人开发的个人复核**流程（含"没有 --reviewer"说明）
├── review_form.md         独立审核表（Phase 11.3，保持不变）
├── self_review_form.md    **新增**：逐条个人复核表（工具**不预填**任何结论）
├── manifest.json / index.md / near_duplicates.json
└── images/<原图名>/{full_preview.jpg, zoom_*.jpg, meta_*.json}
```

* `meta_*.json` 增加 `self_review_command`（可直接复制的命令提示）与
  `self_review_note`（强调"不是独立审核"）；
* 返回结果新增 `self_review_decisions: 0` 与 `self_review_form`，
  与既有的 `promoted: 0` 一起**显式声明工具没有替任何人做决定**；
* 实测：`4 原图 / 8 标注 / promote 0 次 / 工具代填结论 0 条`；
  个人复核表恰有 **8** 行（对应 8 条草稿），且**没有任何勾选**。

---

## 6. 独立审核规则未被削弱（防护清单）

| 防护 | 实现 |
| --- | --- |
| `promote` 仍是唯一 `verified` 入口 | `cmd_promote` 未改动；`self-review` 永不写 `annotation_status=verified` |
| 自审一律拒绝 | `promote` / `review` 保留 `annotator == reviewer` 拒绝；新增测试覆盖 |
| 无法伪造 reviewer | `self-review` **没有** `--reviewer` 参数（argparse 直接报错），也从不写 `reviewer` / `review_status` |
| 个人复核不得冒充真值 | `is_independently_verified` 不读 `self_review_*`；统计口径新增 `self_reviewed_only` 明确区分 |
| 校验兜底 | `check_annotations` 第 11 项：`self_reviewed` 缺复核记录、或写入独立审核字段 ⇒ **报错** |
| 不授予可点击 / 真实输入资格 | 授权只认 `src/core/elements.py` 代码登记表（Phase 11.3），与标注**完全解耦**；`self-review` 不改 `region_kind` |
| 已独立审核的条目不可再写个人复核 | `cmd_self_review` 前置拒绝，且断言文件**未被写入** |

---

## 7. 当前数量口径

| 口径 | 数量 | 说明 |
| --- | --- | --- |
| 试点累计草稿 | 8 | Phase 11.2，全部 `pending_review` |
| **等待独立审核**（`awaiting_review`） | **8** | 含未来"已个人复核但仍待独立审核"的条目 |
| **个人复核记录**（`self_reviewed`） | **0** | 工具不代填；逐条判断属开发者本人的决定 |
| **经独立审核真值**（`independently_verified`） | **0** | 无第二名审核者 ⇒ 如实为 0 |
| **`clickable_verified`** | **0** | 真实游戏输入仍关闭（Phase 11.3 结论沿用） |

> 本阶段**不要求**取得独立审核真值。个人复核的填写是**人的决定**，
> Agent 只提供工具、空表与记录格式，**不代替**开发者做出任何审核结论。

---

## 8. 测试结果

### 8.1 新增测试（`tests/test_phase114_self_review_workflow.py`，15 个）

覆盖用户 7 条要求：个人复核记录完整且原始标注不被改写 / 三种个人结论都不成真值 /
不授予 `clickable_verified` 与真实输入资格 / `self-review` 无 `--reviewer` /
`promote` 仍强制独立审核 / 已独立审核的条目拒绝个人复核 / 结论与时间必须成对 /
修订链保留旧条与复核记录 / 双口径互斥（默认口径**不**含个人复核样本）/
探索口径 `admissible_as_official_metric=False` / `template_hit` 语义不可比**不放宽** /
CLI 接受 `--status self_reviewed` / 审核包个人复核表**零预填**（`promoted=0`、
`self_review_decisions=0`、标注与原图 SHA256 不变）/ 真实 8 条草稿的审核包就绪且只读 /
校验脚本识别不完整或伪装的个人复核 / `action.enabled=false` + `backend=dry_run` 未变。

### 8.2 既有测试

无需修改任何既有测试（新增字段与状态均为**增量**、有默认值）；旧 JSONL 与旧调用
（`include_all_status=`）继续可用。

### 8.3 验收

```text
$ python -m pytest -q                                  → 1545 passed, 1 skipped   # 基线 1530 → +15
$ python tools/check_dataset.py                        → PASS（exit 0）
$ python tools/raw_manifest.py                         → raw manifest: PASS（178 张 SHA256 未变）
$ python tools/check_annotations.py                    → Annotations: PASS
                                                         stats: verified=0, self_reviewed=0, awaiting_review=8
$ python tools/evaluate_localization.py                → INSUFFICIENT_DATA（independently_verified，真值 0）
$ python tools/evaluate_localization.py --status self_reviewed
                                                       → INSUFFICIENT_DATA（exploratory，样本 0；不可作正式指标）
$ python tools/build_annotation_review_pack.py         → 4 原图 / 8 标注 / promote 0 次 / 代填结论 0 条
$ git diff --stat -- dataset                            → 空（正式标注与 raw 未被改动）
```

> 注：控制台为 GBK 时，工具输出含 `⇒`（U+21D2），建议 `set PYTHONIOENCODING=utf-8`
> 或使用 UTF-8 终端（既有行为，非本轮引入）。

---

## 9. 尚未解决的风险

| 编号 | 风险 |
| --- | --- |
| R-11.4-1 | **仍无独立审核者** ⇒ `independently_verified = 0`、定位指标仍不可计算（继承 R-11.3-1） |
| R-11.4-2 | 个人复核是**自证**，无法证明"现实中确有独立判断"；只能用于开发期诊断，不得对外引用 |
| R-11.4-3 | 个人复核记录同样为本地文件形态，无签名 / 时间戳链，长期审计需外部存档（继承 R-11.3-6） |
| R-11.4-4 | 试点仍 4 张原图 / 8 条草稿；`battle_reward` 仍未目视核查（继承 R-11.3-3） |
| R-11.4-5 | detector 仍只输出"模板命中框"口径 ⇒ 即使有个人复核样本也无法算元素级定位精度（继承 R-11.3-2） |
| R-11.4-6 | 单人开发下"修订 → 复核 → 修订"的循环可能让人只维护一套口径，存在**过拟合自证**风险 |
| K-11.4-1 | 文件长度：`annotations.py` 796 / `annotate.py` 546 / `check_annotations.py` 277 / `evaluate_localization.py` 537 / `build_annotation_review_pack.py` 417 / `tests/test_phase114_self_review_workflow.py` 615 行 |

---

## 10. Git commit

```text
feat(dataset): add single developer self-review workflow
```
