# Dataset

## Dataset Overview

本目录存放《阴阳师》视觉理解模块所需的图像数据。数据仅用于屏幕视觉理解与 GUI 自动化，
不涉及游戏内部数据读取、内存修改、网络协议分析或程序注入。

整体结构分为 **新旧两套，并存兼容**：

| 体系 | 目录 | 用途 | 状态 |
| --- | --- | --- | --- |
| **新结构（Phase 4）** | `raw/` | **唯一真源**：未修改的原始截图归档 | 由 `tools/build_dataset.py` 生成 |
| | `states/` | 状态级正/负样本（`<state>/positive\|negative/`） | 同上（与旧分组并存） |
| | `features/` | 证据级 ROI 样本（一个 detector 一种证据） | 同上 |
| **旧结构（历史数据）** | `scenes/` | 场景数据（Scene Classifier） | 保留不动 |
| | `templates/` | UI 模板数据（Template Detector） | 保留不动 |
| | `states/transition_bottom/`、`states/non_transition_bottom/` | Phase 3.0 转场条带样本 | 保留不动 |

设计目标：让数据集支撑
`Frame → Feature Detector → Evidence Score → State Classifier → StateMonitor`，
而不是简单的图片分类。
**`features/` 与 `states/` 的分工是核心**：`states/` 回答"整屏属于哪个状态"，
`features/` 回答"某个 ROI 是否出现某条 UI 证据"；两者可以交叉（例如某张图属于
`exploration` 状态，但它的 `chapter_panel` 仍是负样本）。

## Directory Structure

```text
dataset/
├── metadata.json                     # 全部源截图的标注（178 条）
├── raw/                              # 唯一真源：原始截图归档（不修改像素）
│   ├── yard/                         #   yard_cap_NNN.png / yard_skinNNN_NNN.png
│   ├── exploration/                  #   exploration_ch26_NNN.png
│   ├── battle/                       #   battle_auto_NNN.png
│   ├── battle_result/                #   battle_result_win_NNN.png
│   ├── battle_reward/                #   battle_reward_item_NNN.png
│   └── unknown/                      #   无法归入上述状态的有效游戏界面
│       ├── shop/ collection/ friends/ shikigami/ equipment/ other/
│
├── states/                           # 状态级样本
│   ├── yard/positive/  negative/
│   ├── exploration/positive/  negative/
│   ├── battle/positive/  negative/
│   ├── battle_result/positive/  negative/
│   ├── battle_reward/positive/  negative/
│   ├── unknown/                      #   兜底状态，按来源子分类（无 positive/negative）
│   │   └── shop/ collection/ friends/ shikigami/ equipment/ other/
│   ├── transition_bottom/            # [旧] 扁平样本，保留兼容
│   └── non_transition_bottom/        # [旧] 扁平样本，保留兼容
│
├── features/                         # 证据级 ROI 样本（一个 detector 一种证据）
│   ├── bottom_menu/
│   │   ├── yard/positive/  negative/
│   │   └── exploration/positive/  negative/
│   ├── chapter_panel/positive/  negative/
│   ├── battle/
│   │   ├── auto_button/positive/  negative/
│   │   ├── skill_area/positive/  negative/
│   │   ├── action_bar/positive/  negative/
│   │   └── shikigami_icons/positive/  negative/
│   ├── victory_icon/positive/  negative/
│   └── reward_panel/positive/  negative/
│
├── scenes/yard/skin_XXX/scene_XXX/   # [旧] 场景数据
└── templates/buttons/explore/        # [旧] UI 模板数据
```

## State Definitions

| state | 含义 | 判定依据（features） |
| --- | --- | --- |
| `yard` | 庭院 | `bottom_menu_yard` |
| `exploration` | 探索（章节地图） | `bottom_menu_exploration` + `chapter_panel` |
| `battle` | 战斗中 | `battle_auto_button` + `battle_skill_area` + `battle_action_bar` + `battle_shikigami_icons` |
| `battle_result` | 战斗胜利结算 | `victory_icon` |
| `battle_reward` | 奖励展示 | `reward_panel` |
| `unknown` | 无法归入上述五类**但仍是有效游戏界面**（商店 / 图鉴 / 好友 / 式神录 / 御魂 / 阴阳寮 / 斗技 / 珍旅居 …） | 无（兜底类别） |

`positive` = 明确属于该状态的完整截图（且带该状态的 UI 证据）；
`negative` = 容易被误判为该状态、但实际属于其它状态的截图副本。

## Known Gaps

| 缺口 | status | reason | need recollection |
| --- | --- | --- | --- |
| `transition` 状态 | **missing** | 现有样本不含真实转场画面 | 是 |
| `battle_result` / `battle_reward` | **insufficient** | 各仅 1 张正样本 | 是 |
| `victory_icon` / `reward_panel` detector | **insufficient** | 正样本各仅 1 张，证据置信度已被折价 | 是 |

### `transition`：missing

* **status**：missing
* **reason**：现有素材只有 `states/transition_bottom/` 的 10 张，分辨率 `1706x57`，
  内容仅为黑色条带与一行暗色官网文字（`yys.163.com`），**不包含任何真实游戏转场 /
  加载画面**。把它当作 positive 会让模型学到"黑色区域 = transition"这一错误规则。
* **处理**：
  1. **不建立** `states/transition/`；
  2. `states/transition_bottom/` **原样保留**，标记为**历史素材（待验证）**，仅作兼容用途，
     新结构以 `state_name` 为准；
  3. 该目录不参与 `states/` 的状态样本统计与判定。
* **need recollection**：需专项采集以下场景后，再建立 `states/transition/positive/`：

  1. 地图 → 副本加载
  2. 探索 → 战斗切换
  3. 战斗结束 → 结算
  4. 庭院 → 功能界面切换
  5. 黑屏加载
  6. 过场动画

* **采集要求**：放**完整截图**（不要只截底部条带），并同步归档到 `raw/transition/`。

## Naming Convention

新结构统一为 **`{prefix}_{source}_{NNN}.png`**：全小写、下划线分隔、编号三位零填充，
**文件名前缀必须与所在目录一致**（由 `tools/check_dataset.py` 校验）。

| 位置 | prefix | source 取值 | 示例 |
| --- | --- | --- | --- |
| `raw/<state>/`、`states/<state>/positive/` | `<state>` | `cap` / `skinNNN` / `ch26` / `auto` / `win` / `item` | `yard_skin001_001.png`、`exploration_ch26_001.png`、`battle_result_win_001.png` |
| `states/<state>/negative/` | `<state>` | 误判来源状态 | `yard_exploration_001.png` |
| `states/unknown/<cat>/`、`raw/unknown/<cat>/` | `unknown` | 来源子分类 | `unknown_shop_001.png` |
| `features/bottom_menu/<kind>/` | `bottom_menu_<kind>` | 正向来源 / 负向来源 | `bottom_menu_yard_cap_001.png`、`bottom_menu_yard_exploration_001.png` |
| `features/chapter_panel/` | `chapter_panel` | `ch26` / 负向来源 | `chapter_panel_ch26_001.png` |
| `features/battle/<part>/` | `battle_<part>` | `auto` / 负向来源 | `battle_auto_button_auto_001.png` |
| `features/victory_icon/`、`features/reward_panel/` | 同名 | `win` / `item` / 负向来源 | `victory_icon_win_001.png` |

旧结构（`scenes/`、`templates/`）沿用各自既有规范；
`states/transition_bottom/`、`states/non_transition_bottom/` 为人工标注的宽松命名，不做规范校验。

**禁止**：`IMG001.png`、`截图1.png`、`微信图片.png` 这类无信息文件名。

## Feature ROI Definition

ROI 用**相对完整截图的百分比**表示（`x0, y0, x1, y1`），裁剪只做 `crop`：
不缩放、不压缩、不改色，保持原始像素与比例。

| detector | x0 | y0 | x1 | y1 | 覆盖的证据 |
| --- | --- | --- | --- | --- | --- |
| `bottom_menu/yard` | 0.00 | 0.80 | 1.00 | 1.00 | 庭院型底部功能栏（图鉴/珍旅居/纳物库/组队/阴阳寮/商店/花合战/好友/阴阳术/式神录） |
| `bottom_menu/exploration` | 0.00 | 0.80 | 1.00 | 1.00 | 探索型底部功能栏（觉醒材料/御魂/结界突破/御灵/式神委派/秘闻副本/地域鬼王/妖气探索/英杰试炼/六道之门/契灵之境） |
| `chapter_panel` | 0.80 | 0.15 | 1.00 | 0.70 | 主线/玩法 + 第二十六~二十八章 |
| `battle/auto_button` | 0.00 | 0.74 | 0.12 | 1.00 | 左下角自动/手动按钮 |
| `battle/skill_area` | 0.60 | 0.74 | 0.92 | 1.00 | 底部技能按钮区 |
| `battle/action_bar` | 0.90 | 0.04 | 1.00 | 0.72 | 右侧行动顺序条 |
| `battle/shikigami_icons` | 0.925 | 0.09 | 1.00 | 0.60 | 行动条上的式神头像 |
| `victory_icon` | 0.25 | 0.05 | 0.85 | 0.40 | 中心胜利图标与「胜利」字样 |
| `reward_panel` | 0.15 | 0.15 | 0.62 | 0.90 | 奖励卡片 + 中央奖励宝箱 |

> `bottom_menu` 刻意**拆成 yard / exploration 两个 detector**：两类底部功能栏的图标组成、
> 排列与上下文完全不同，合并成单一 `bottom_menu=true` 会丢失区分度。使用侧应分别产出
> `bottom_menu_yard` 与 `bottom_menu_exploration` 两条证据。
>
> `battle` 同理拆成 4 个 detector：战斗 UI 的有效证据分散在左下、底部、右侧三个区域，
> 单一 ROI 要么丢证据、要么引入大量无关背景。**不要裁整屏作为 battle detector 数据。**

## Metadata

`dataset/metadata.json` 以**源截图**为单位（178 条），每条记录该图的标注与全部落盘路径：

| 字段 | 含义 |
| --- | --- |
| `image` | `raw/` 下的相对路径（如 `raw/exploration/exploration_ch26_001.png`） |
| `state` | 状态标签：`yard` / `exploration` / `battle` / `battle_result` / `battle_reward` / `unknown` |
| `source` | 来源标识：`cap` / `skinNNN` / `ch26` / `auto` / `win` / `item` / unknown 子分类 |
| `features` | 该图**含**的 feature 证据列表（如 `["bottom_menu_exploration", "chapter_panel"]`） |
| `confidence` | 标注置信度，当前一律 `manual` |
| `note` | 人工备注（含困难样本说明） |
| `subcategory` | 仅 `unknown`：`shop` / `collection` / `friends` / `shikigami` / `equipment` / `other` |
| `scene_only` | 仅无 UI 的纯场景图：`true`（不进任何 positive，只作 negative） |
| `paths.raw` | 归档路径 |
| `paths.state_positive` | 状态正样本路径（`scene_only` / `unknown` 无此字段） |
| `paths.original` | **原始素材路径**，用于回溯整理前的来源 |

示例：

```json
{
  "image": "raw/exploration/exploration_ch26_035.png",
  "state": "exploration",
  "source": "ch26",
  "features": ["bottom_menu_exploration", "chapter_panel"],
  "confidence": "manual",
  "note": "探索地图，章节栏完整（主线/玩法 + 第二十六~二十八章）",
  "paths": {
    "raw": "raw/exploration/exploration_ch26_035.png",
    "original": "dataset/states/non_transition_bottom/探索界面.png",
    "state_positive": "states/exploration/positive/exploration_ch26_035.png"
  }
}
```

## Build & Validation

```powershell
# 从既有素材重建 raw/ states/ features/ 与 metadata.json（幂等，可重复运行）
python tools/build_dataset.py
python tools/build_dataset.py --dry-run     # 只统计，不落盘

# 校验目录结构、命名规范与 PNG 有效性（只读）
python tools/check_dataset.py
```

`build_dataset.py` 只写入自己负责的**生成目录**（`raw/`、`features/`、
`states/<state>/positive|negative/`、`states/unknown/`），对 `scenes/`、`templates/`、
`states/transition_bottom/`、`states/non_transition_bottom/` 以及 `states/<state>/` 下的
历史扁平样本一律不动（脚本内有 `GENERATED_DIRS` / `PROTECTED_DIRS` 防误删校验）。

`check_dataset.py` 校验内容：

1. 目录结构（旧结构 + 新结构白名单，含 `states/` 分组、`raw/` 状态、`features/` detector）；
2. 图片命名（`{prefix}_{source}_{NNN}.png`，含"前缀与所在目录一致"与编号重复）；
3. PNG 文件是否有效（签名 + 完整解码）；
4. 无法解析的文件输出 `Unresolved Files List` 等待人工确认。

退出码 `0` = PASS，`1` = FAIL。

入库策略：图片体积大，**一律不进入 git 历史**（`.gitignore` 忽略 `dataset/**/*.png`、
`*.jpg`、`*.jpeg`、`*.zip`）；仓库只跟踪目录结构（`.gitkeep`）、`metadata.json`、
本说明与 `tools/` 下脚本。从 git 克隆出的工作区不含图片，需要本地单独准备数据后运行
`build_dataset.py`。

## Adding New Data

1. 把新截图放进对应 `raw/<state>/`（或新增状态时同步更新 `tools/check_dataset.py` 与
   `tools/build_dataset.py` 的白名单），命名遵循 `{state}_{source}_{NNN}.png`；
2. 在 `tools/build_dataset.py` 的映射表里登记新素材（`SOURCE` 映射 / `SAMPLE_NOTES`）；
3. 运行 `python tools/build_dataset.py` 重建派生数据；
4. 运行 `python tools/check_dataset.py` 确认 PASS。

新增 detector 时：在 `build_dataset.py` 的 `FEATURE_ROIS` 登记 ROI、
在 `check_dataset.py` 的 `ALLOWED_FEATURE_DETECTORS` 登记目录。
