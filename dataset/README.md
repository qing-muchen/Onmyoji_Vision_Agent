# Dataset

## Dataset Overview

本目录存放《阴阳师》视觉理解模块所需的图像数据，分为三类：

- **Scene Dataset（场景数据）**：用于判断当前游戏画面属于什么场景；
- **UI Template Dataset（UI 模板数据）**：用于 Template Detector 定位按钮、图标等 UI 元素；
- **State Samples（状态样本）**：用于 Phase 3.0 的状态识别（转场底部条带的正/负样本）。

数据仅用于屏幕视觉理解与 GUI 自动化，不涉及游戏内部数据读取、内存修改、网络协议分析或程序注入。

> **入库策略**：图片体积较大（当前 158 张：场景 70 / 模板 35 / 状态样本 53），
> **图片与数据归档一律不进入 git 历史**
> （见根目录 `.gitignore`：忽略 `dataset/**/*.png`、`*.jpg`、`*.jpeg`、`*.zip`）。
> 仓库只跟踪目录结构（`.gitkeep`）、本说明与校验脚本 `tools/check_dataset.py`；
> 因此从 git 克隆出的工作区不含图片，需要在本地单独准备数据。

## Directory Structure

```text
dataset/
├── README.md                                        # 本说明
├── scenes/                                          # 场景数据（用于场景识别）
│   └── yard/                                        #   场景类型：庭院（yard）
│       └── skin_XXX/                                #     皮肤编号
│           └── scene_XXX/                           #       场景编号
│               └── yard_skinXXX_sceneXXX_frameXXX.png
├── templates/                                       # UI 模板数据（用于 UI 元素检测）
│   └── buttons/                                     #   元素类别：按钮
│       └── explore/                                 #     元素名称：探索
│           └── button_explore_skinXXX_sceneXXX.png
└── states/                                          # 状态样本（用于状态识别，Phase 3.0）
    ├── transition_bottom/                           #   正样本：**已裁剪的底部条带**
    │   └── transition_bottom_NNN.png
    └── non_transition_bottom/                       #   负样本：**完整界面截图**
        └── <界面名>.png                              #     例如 好友界面.png
```

- `scenes/`：场景识别数据，目录层级与文件名**同时**编码 `skin / scene / frame`，便于批量遍历，
  单个文件被移动或单独取出时也不会丢失来源信息。
- `templates/`：UI 元素模板，按「元素类别 / 元素名称」分目录，文件名以**元素类型前缀**开头。
- `states/`：状态样本，按`状态分组`分目录；**两类样本的图像形态不同**（见下表），
  使用侧会统一归一化到"底部条带"再比较，因此新增数据时**不要**手工裁剪
  `non_transition_bottom` 的截图。

## Naming Convention

| 类型 | 命名规范 | 示例 |
| --- | --- | --- |
| Scene Image | `{scene_kind}_skin{NNN}_scene{NNN}_frame{NNN}.png` | `yard_skin001_scene002_frame003.png` |
| Template Image | `{element_type}_{element_name}_skin{NNN}_scene{NNN}.png` | `button_explore_skin001_scene002.png` |
| State Sample（正样本） | `transition_bottom_{NNN}.png` | `transition_bottom_003.png` |
| State Sample（负样本） | `<界面名>.png`（**宽松命名**，允许中文） | `好友界面.png`、`探索-地域鬼王.png` |

规则：

1. 编号统一**三位零填充**（`1` → `001`，`18` → `018`）。
2. 全部小写，字段以下划线分隔，不使用中文、空格、短横线或其它分隔符。
3. 模板命名以元素类型开头，便于扩展更多 UI 元素：`button_explore_skin001_scene001.png`、
   `button_confirm_skin001_scene002.png`、`icon_auto_skin001_scene001.png`。
   **不使用** `explore_button_xxx` 这类"名称在前、类型在后"的写法。
4. 场景文件名中的 `skin/scene` 编号必须与所在目录一致（由 `tools/check_dataset.py` 校验）。
5. 统一 PNG 格式，保留原始分辨率与像素内容：不压缩、不缩放、不转换格式。
6. **`states/` 例外**：状态样本由人工标注（负样本就是界面截图的名字），因此**不要求**
   小写/编号规范，只要求是合法 PNG；`transition_bottom` 的编号便于人工索引，但不强制连续。

## Metadata Meaning

| 字段 | 含义 |
| --- | --- |
| `skin` | 皮肤编号：同一场景在不同皮肤下的外观差异 |
| `scene` | 场景编号：同一皮肤下的第几个场景 |
| `frame` | 采集帧编号：同一场景下的第几次采样（仅场景数据使用） |
| `scene_kind` | 场景类型名，当前为 `yard`（庭院） |
| `element_type` | UI 元素类型，当前为 `button`（按钮），后续可扩展 `icon` 等 |
| `element_name` | UI 元素名称，当前为 `explore`（探索） |
| `buttons/` | 模板分组目录（按元素类别粗分），后续可扩展 `icons/` 等 |
| `transition_bottom/` | 状态样本分组：**转场/加载界面的底部条带**（正样本） |
| `non_transition_bottom/` | 状态样本分组：**非转场界面的完整截图**（负样本） |

状态样本的两个分组**图像形态不同**（由采集方式决定，使用侧会自动归一化）：

| 分组 | 形态 | 实测尺寸示例 | 使用侧处理 |
| --- | --- | --- | --- |
| `transition_bottom/` | 已裁剪的底部条带 | 1704×58、1706×56~60 | 宽高比已足够大 → **原样使用** |
| `non_transition_bottom/` | 完整界面截图 | 1706×961、1706×962 | 按 `state.transition.crop_height_ratio` **裁出底部条带** |

因此新增负样本时**直接放完整截图**即可（不要预先裁剪），新增正样本时放**底部条带**。

## Validation

```powershell
python tools/check_dataset.py
```

校验内容：

1. 目录结构是否符合规范（含 `states/` 的分组目录名白名单）；
2. 图片命名是否符合规范（含"文件名编号与目录编号一致"；`states/` 只要求是合法 PNG）；
3. PNG 文件是否有效（签名 + 完整解码校验）；
4. 是否存在重复编号；
5. 是否存在无法解析的文件（输出 `Unresolved Files List` 等待人工确认）。

脚本只读取数据、不修改任何文件，可重复运行；退出码 `0` 表示 PASS，`1` 表示 FAIL。

## Adding New Data

1. 按上表命名规范放入对应目录（新增场景类型 / 模板分组 / 状态分组时，同步更新
   `tools/check_dataset.py` 中的 `ALLOWED_SCENE_KINDS` / `ALLOWED_TEMPLATE_GROUPS` /
   `ALLOWED_STATE_GROUPS` 白名单）；
2. 运行 `python tools/check_dataset.py` 确认 PASS；
3. 图片不需要 `git add`（已被 `.gitignore` 忽略），只需保证目录结构与命名规范同步。

状态样本的扩展方式（Phase 3.0）：

- **增加负样本**：把新的界面截图（完整截图，任意文件名）放进
  `dataset/states/non_transition_bottom/` 即可，无需改代码；
- **增加正样本**：把转场界面的底部条带放进 `dataset/states/transition_bottom/`；
- 样本数量变化不影响配置，但**建议在扩充数据后重跑 Demo / 校准探针**，确认
  `state.transition.negative_margin` 仍然把正负两类分开。
