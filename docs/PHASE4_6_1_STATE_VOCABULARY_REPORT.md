# PHASE 4.6.1 Report — State Vocabulary Unification

| 项 | 值 |
| --- | --- |
| 阶段 | Phase 4.6.1 State Vocabulary Unification |
| 目标 | 统一状态枚举：让 `StateName` 能表达证据层聚合规则推断出的**全部**状态，并让配置里的状态名声明受同一词汇约束 |
| 前置 | Phase 4.5 Architecture Review（`docs/PHASE4_5_ARCHITECTURE_REVIEW.md`，F1 / F3 / D-5 / I-5） |
| 代码基线 | `0fd83c2` + Phase 4.5 文档（未提交） |
| 完成状态 | 已实现、已测试；**未执行 `git commit`**（commit 信息见第 5 节） |

**范围遵守声明**（对照 Phase 4.6.1 的允许/禁止清单）：

| 范围 | 结果 |
| --- | --- |
| 允许修改：`src/vision/state/models.py` | 已改（唯一改动的 vision 源文件） |
| 允许修改：`src/core/config.py` | 已改 |
| 允许修改：`tests/` | 已改（2 个更新 + 1 个新建 + 1 个用例修正） |
| 禁止修改：`src/vision/evidence/` | 未触碰 |
| 禁止修改：`src/vision/capture/`、`src/vision/window/` | 未触碰 |
| 禁止修改：`src/runtime/`（含 `StateMonitor` / `StateStabilizer`） | 未触碰（`StateEvent` / `StateStabilizer` 无需改动即可支持新状态，见第 2.3 节） |
| 禁止修改：`dataset/` | 未触碰（只读统计） |
| 禁止：Action Layer、状态识别逻辑、Phase 4.6.2 内容 | 未触碰（`StateClassifier` 行为完全不变） |

---

## 1. 修改文件

| 文件 | 类型 | 变更摘要 |
| --- | --- | --- |
| `src/vision/state/models.py` | 修改 | `StateName` 扩展为 7 个状态；模块 docstring 记录"新增状态的登记顺序"；`StateResult` 的 `state_name` / `is_known` 文档同步（**字段与签名不变**） |
| `src/core/config.py` | 修改 | 新增 `KNOWN_STATE_NAMES`（状态词汇镜像）与 `_require_known_state_name()`；`EvidenceRuleConfig.state` 与 `EvidenceAggregatorConfig.fallback_state` 改为校验合法性 |
| `tests/test_state_vocabulary.py` | **新增** | 跨层一致性测试：枚举 ↔ 配置常量 ↔ 默认规则 ↔ 仓库配置；非法状态名的拒绝与错误信息 |
| `tests/test_state_models.py` | 修改 | 枚举取值断言更新为 7 值；新增"旧状态名仍可用""每个状态都能被 `StateResult` 表达""`is_known` 只排除 unknown" |
| `tests/test_config.py` | 修改 | 新增：合法状态名的规则可通过加载；非法状态名（拼写错误 / 大小写不符 / 不支持的值 / 非法兜底名）必须抛 `ConfigError` |
| `tests/test_state_monitor_events.py` | 修改 | `StateEvent.from_dict` 的"未知状态"样例由 `battle` 改为词汇表外的 `not_a_state`（`battle` 现在是合法状态名），并新增 4 个"统一后的状态名可解析"用例 |

**未新增/未删除任何生产代码文件**；`KNOWN_STATE_NAMES` 之外没有引入新的运行时开关。

### 1.1 `StateName` 最终取值

| 成员 | 值 | 说明 |
| --- | --- | --- |
| `YARD` | `yard` | 庭院界面（Phase 3.0 既有） |
| `EXPLORATION` | `exploration` | 探索界面（章节栏 + 探索型底部功能栏） |
| `BATTLE` | `battle` | 战斗进行中 |
| `BATTLE_RESULT` | `battle_result` | 战斗结算 |
| `BATTLE_REWARD` | `battle_reward` | 战斗奖励展示 |
| `TRANSITION` | `transition` | 转场 / 加载插画（Phase 3.0 既有） |
| `UNKNOWN` | `unknown` | 未识别 / 兜底（Phase 3.0 既有） |

### 1.2 配置校验行为

| 配置项 | 变更前 | 变更后 |
| --- | --- | --- |
| `evidence.aggregator.rules[*].state` | 仅校验"非空" | 非空 **且** ∈ `KNOWN_STATE_NAMES`，错误信息给出非法值与合法集合 |
| `evidence.aggregator.fallback_state` | 无校验 | 非空 **且** ∈ `KNOWN_STATE_NAMES`（字段路径出现在错误信息中） |

实测错误信息（由 pydantic 包装为 `Value error, ...`；以下为消息正文）：

```text
evidence 规则的 state 不是系统支持的状态名: 'battel'（合法值: battle, battle_result, battle_reward, exploration, transition, unknown, yard）
evidence.aggregator.fallback_state 不是系统支持的状态名: 'nothing'（合法值: ...）
```

---

## 2. 架构变化

### 2.1 状态词汇成为单点事实（三处镜像 + 一处锁）

```text
src/vision/state/models.py        StateName（语义来源：7 个成员）
        │  一致性由测试锁定
        ▼
src/core/config.py                KNOWN_STATE_NAMES（配置校验用镜像）
        │  校验
        ▼
configs/config.yaml               evidence.aggregator.rules[*].state / fallback_state
        │  证据层
        ▼
EvidenceAggregator → StateCandidate.state（str，写法受限为合法词汇）
```

**为什么是"镜像"而不是直接导入**：`core` 是最底层基础设施，依赖方向只允许 `core → vision`；
若 `config.py` 在导入期 `from src.vision.state.models import StateName`，就会形成
`core → vision → core` 的导入环（`vision.state.classifier` 依赖 `core.config`）。
因此 `KNOWN_STATE_NAMES` 保存同一词汇的镜像，并由
`tests/test_state_vocabulary.py::test_enum_matches_config_vocabulary` 机械锁定二者相等
—— 单一语义来源仍是 `StateName`，配置层只是拿到一份**可校验的拷贝**。

### 2.2 消除的缺口（Phase 4.5 的 F1 / F3）

| 缺口 | Phase 4.5 现状 | 现在 |
| --- | --- | --- |
| 表达能力（F1） | 证据层可产出 `exploration` / `battle` / `battle_result` / `battle_reward`，`StateName` 只有 3 态 → `StateResult` 无法表达 | 7 态枚举，全部可表达；`tests/test_state_models.py::test_every_state_name_can_be_expressed` 逐个验证 |
| 配置拼写错误（F3） | 规则 `state` 只校验非空 → `battel` 静默生效，产生"幽灵状态" | 配置加载阶段即失败（`ConfigError`），错误信息给出合法集合 |

### 2.3 未改变的行为（刻意的"最小改动"）

| 组件 | 为什么不改 | 现状 |
| --- | --- | --- |
| `StateClassifier`（`state/classifier.py`） | 本阶段只解决"名称表达能力"，不改识别逻辑；且该文件不在允许修改范围 | 仍只产出 `yard` / `transition` / `unknown`（逐行未动） |
| `StateStabilizer` | 新状态会自然走"已知状态"路径（置信度门槛），逻辑无需改动；`unknown` 豁免规则不受影响（`runtime` 禁止修改） | `is StateName.UNKNOWN` 判断语义不变 |
| `StateEvent` / `StateHistoryStore` | `StateName(value)` 构造与 `.value` 序列化对新增取值天然可用；旧历史 JSON 的读法不变 | 旧记录可继续解析（新增 4 个用例验证） |
| `StateResult` 字段 | `state_name` / `confidence` / `source` / `detail` 全部保留（向后兼容要求） | `result.state_name` 用法零改动 |

### 2.4 与 Phase 4.5 冻结契约的关系

- 落实 **I-5**（状态名单一事实来源：枚举登记 + 配置校验）；
- 落实迁移计划 **4.6.1**（"`StateName` 扩展 + 规则状态名校验"）；
- **未触碰** I-2（时间层唯一所有者）、I-3（阈值口径分离）、4.6.2–4.6.6 的任何内容
  —— 本阶段不涉及 `StateResult.reason/evidence`、`transition` 证据化、yard 权威、阈值重标定。

---

## 3. 测试结果

### 3.1 执行命令与结果

```powershell
# 基线（Phase 4.4 commit 0fd83c2，改动前）
.\.venv\Scripts\python.exe -m pytest -q -m "not integration"
# → 593 passed, 1 skipped, 3 deselected

# 改动后：完整 pytest（用户要求必须执行）
.\.venv\Scripts\python.exe -m pytest -q
# → 655 passed, 1 skipped, 0 failed

# 改动后：非 integration 子集（与基线同一口径，便于逐项对比）
.\.venv\Scripts\python.exe -m pytest -q -m "not integration"
# → 652 passed, 1 skipped, 3 deselected
```

| 指标 | 基线 | 现在 | 差异 |
| --- | --- | --- | --- |
| passed（not integration 口径） | 593 | 652 | **+59** |
| skipped | 1 | 1 | 0（Tesseract 未安装导致，与本次改动无关） |
| failed | 0 | 0 | **0（无下降）** |
| 完整 pytest | —（上次未跑全量） | 655 passed | 0 failed |

### 3.2 新增/更新用例核算（+59）

| 文件 | 变更 | 用例数 |
| --- | --- | --- |
| `tests/test_state_vocabulary.py`（新增） | 一致性 5 项 + 正向参数化 7+7 + 非法拒绝 7+7 + 错误信息 2 | 35 |
| `tests/test_state_models.py` | 旧名兼容 1 + 全状态可表达 7 + `is_known` 6+1 | 15 |
| `tests/test_config.py` | 合法配置 1 + 非法配置参数化 4 | 5 |
| `tests/test_state_monitor_events.py` | 新增"统一状态名可解析"参数化 4（另有 1 个原用例按新词汇修正样例值） | 4 |
| 合计 | | **59**（与 +59 精确吻合） |

### 3.3 关键验证点（对应本阶段三项测试要求）

1. **原有测试不下降**：652 passed（含全部 593 个基线用例），0 failed；
2. **新增状态枚举测试**：7 个成员的取值逐一断言、旧 3 态构造仍可用、
   `StateResult` 能表达每一个状态、`is_known` 仅排除 `unknown`；
3. **配置状态合法性测试**：
   - 合法：7 个状态名都能作为规则 `state` 与 `fallback_state` 通过；
   - 非法：`battel` / `Battle` / `BATTLE` / `battle ` / ` explore` / `""` / `"  "` 全部被拒；
   - 端到端：`load_config()` 对非法状态名的 YAML 抛 `ConfigError`；
   - 仓库配置：`configs/config.yaml` 的实际规则状态名全部合法；
   - 一致性：`{StateName} == KNOWN_STATE_NAMES`，且默认证据规则不引用词汇表外的名字。

### 3.4 一处必要的既有测试修正（必须披露）

`tests/test_state_monitor_events.py::test_from_dict_rejects_unknown_state` 原先用
`current_state="battle"` 作为"未知状态"样例；`battle` 现在是合法状态名，用例因此失效
（改动后首次全量跑：`1 failed, 647 passed`）。修正方式：把样例值换成词汇表外的
`not_a_state`（该用例的意图"未知状态名必须报错"未变），并额外新增 4 个用例验证
`exploration` / `battle` / `battle_result` / `battle_reward` 现在能从历史字典解析。

---

## 4. 未解决问题

| 编号 | 问题 | 影响 | 建议归属 |
| --- | --- | --- | --- |
| U1 | `src/vision/state/__init__.py`、`classifier.py`、`base.py` 的 docstring 仍写"只支持三态 / 三态判定（`yard` / `transition` / `unknown`）" | 文档与枚举不一致，可能误导阅读者 | 不在 4.6.1 允许的修改范围（仅允许 `models.py`）；建议 4.6.3 重构 `StateClassifier` 时一并更新 |
| U2 | `tools/build_dataset.py:33-41` 另有一份状态词汇定义（`TARGET_STATES` + `UNKNOWN`），未包含 `transition` | 第三处词汇定义，未来可能与枚举分叉 | `tools/` 不在允许范围；建议在后续阶段改为引用 `StateName`（或至少加入一致性测试） |
| U3 | `EvidenceAggregator` 产出的 `StateCandidate.state` 仍是自由 `str`（`src/vision/evidence/` 禁止修改） | 目前合法性**只由配置校验**保证；运行期构造的 `StateCandidate` 仍可携带任意字符串 | 4.6.3（`StateClassifier` 改为消费证据集）时给出强保证 |
| U4 | README 的 Phase 3 章节与路线图尚未记录 Phase 4.1–4.6.1；`examples/state_stabilizer_demo.py` 的 `--help` 已知状态列表会自动变成 7 项（行为改善，但文案未同步） | 文档滞后（Phase 4.5 的 R9 仍然存在） | 4.6.x 或阶段收尾时统一更新 README |
| U5 | `KNOWN_STATE_NAMES` 与 `StateName` 的同步仍靠测试而非机制 | 新增状态时若忘记同步常量，`test_enum_matches_config_vocabulary` 会失败（**会失败，不会静默**） | 保持现状（显式失败优于隐式机制）；若希望零成本同步，可在后续阶段评估"由 core 提供注册表"方案 |
| U6 | `StateStabilizer.confidence_threshold`（0.85）仍是 State 层口径；新状态一旦由证据层产出，需重新标定（Phase 4.5 的 R3） | 未解决（本阶段刻意不碰） | 4.6.6 |
| U7 | 未验证真实数据下的状态识别效果（本阶段不改识别逻辑，`StateClassifier` 仍不产出新状态） | 无（本阶段无识别能力变更） | 4.6.3 起 |

**本阶段明确未做的事**：未改证据层代码、未改 `runtime`、未改 `StateClassifier` 的判定分支、
未引入 `StateResult.reason/evidence`、未做 `transition` 证据化、未删除 `EvidenceHistory`、
未建立编排层（即：Phase 4.6.2 及以后的内容一律未启动）。

---

## 5. Git commit 信息

**未执行 `git commit`**（本阶段只输出改动与信息）。建议把 Phase 4.5 文档与本次代码改动**分开提交**：

### 5.1 建议提交 A：Phase 4.5 文档（若尚未提交）

```text
docs(vision): add Phase 4.5 evidence-state architecture review

记录 Evidence Layer 与 State Layer 的职责边界冻结结论：现状 12 项事实（F1–F12）、
6 项重复职责（D-1–D-6）、目标流水线与接口契约、时间层归属决策（方案 A：删除
EvidenceHistory，时间层唯一归属 StateStabilizer）、迁移步骤与 14 项风险（R1–R14）。
只读 Review，不修改任何代码。
```

### 5.2 建议提交 B：Phase 4.6.1 代码改动（本次）

```text
feat(state): unify state vocabulary with the evidence layer

扩展 StateName 为完整状态集，并让配置里的状态名声明必须落在同一词汇内，
消除"证据层推断得出、状态层表达不了"的缺口与拼写错误静默生效的问题（Phase 4.6.1）。

- src/vision/state/models.py: StateName 新增 exploration / battle / battle_result /
  battle_reward（yard / transition / unknown 保持兼容）；docstring 记录新增状态的
  登记顺序（先枚举 → 再配置镜像 → 再规则）
- src/core/config.py: 新增 KNOWN_STATE_NAMES（状态词汇镜像，避免 core 在导入期依赖
  vision 而形成导入环）与 _require_known_state_name()；EvidenceRuleConfig.state 与
  EvidenceAggregatorConfig.fallback_state 现在校验合法性并给出合法集合
- tests/test_state_vocabulary.py: 新增跨层一致性测试（枚举 ↔ 配置常量 ↔ 默认规则 ↔
  仓库配置）与非法状态名拒绝用例
- tests/test_config.py: 新增配置层状态合法性用例（合法通过 / 非法抛 ConfigError）
- tests/test_state_models.py: 枚举断言更新为 7 值，新增旧名兼容、全状态可表达、
  is_known 语义用例
- tests/test_state_monitor_events.py: StateEvent.from_dict 的"未知状态"样例改为
  词汇表外的 not_a_state，并新增统一状态名可解析用例

行为不变：状态识别逻辑未改动（StateClassifier 仍只产出 yard / transition / unknown）。
兼容性：词汇扩展是加法，result.state_name 用法与旧历史 JSON 解析保持可用。

验证：pytest 655 passed, 1 skipped（基线 593 passed 无下降，新增 59 个用例）。
Refs: docs/PHASE4_5_ARCHITECTURE_REVIEW.md（F1 / F3 / D-5 / I-5）
```

### 5.3 提交前检查清单

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # 期望 655 passed, 1 skipped
git status --short                                # 期望仅上述 5 个修改 + 1 个新增（+ 4.5 文档）
```
