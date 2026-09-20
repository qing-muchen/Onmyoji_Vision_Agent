# Dataset

## Dataset Overview

本目录存放《阴阳师》视觉理解模块所需的图像数据，分为两类：

- **Scene Dataset（场景数据）**：用于判断当前游戏画面属于什么场景；
- **UI Template Dataset（UI 模板数据）**：用于后续 Template Detector 定位按钮、图标等 UI 元素。

数据仅用于屏幕视觉理解与 GUI 自动化，不涉及游戏内部数据读取、内存修改、网络协议分析或程序注入。

> **入库策略**：图片体积较大（当前 105 张 / 约 129 MB），**图片一律不进入 git 历史**
> （见根目录 `.gitignore`：忽略 `dataset/**/*.png`、`*.jpg`、`*.jpeg`）。
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
└── templates/                                       # UI 模板数据（用于 UI 元素检测）
    └── buttons/                                     #   元素类别：按钮
        └── explore/                                 #     元素名称：探索
            └── button_explore_skinXXX_sceneXXX.png
```

- `scenes/`：场景识别数据，目录层级与文件名**同时**编码 `skin / scene / frame`，便于批量遍历，
  单个文件被移动或单独取出时也不会丢失来源信息。
- `templates/`：UI 元素模板，按「元素类别 / 元素名称」分目录，文件名以**元素类型前缀**开头。

## Naming Convention

| 类型 | 命名规范 | 示例 |
| --- | --- | --- |
| Scene Image | `{scene_kind}_skin{NNN}_scene{NNN}_frame{NNN}.png` | `yard_skin001_scene002_frame003.png` |
| Template Image | `{element_type}_{element_name}_skin{NNN}_scene{NNN}.png` | `button_explore_skin001_scene002.png` |

规则：

1. 编号统一**三位零填充**（`1` → `001`，`18` → `018`）。
2. 全部小写，字段以下划线分隔，不使用中文、空格、短横线或其它分隔符。
3. 模板命名以元素类型开头，便于扩展更多 UI 元素：`button_explore_skin001_scene001.png`、
   `button_confirm_skin001_scene002.png`、`icon_auto_skin001_scene001.png`。
   **不使用** `explore_button_xxx` 这类"名称在前、类型在后"的写法。
4. 场景文件名中的 `skin/scene` 编号必须与所在目录一致（由 `tools/check_dataset.py` 校验）。
5. 统一 PNG 格式，保留原始分辨率与像素内容：不压缩、不缩放、不转换格式。

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

## Validation

```powershell
python tools/check_dataset.py
```

校验内容：

1. 目录结构是否符合规范；
2. 图片命名是否符合规范（含"文件名编号与目录编号一致"）；
3. PNG 文件是否有效（签名 + 完整解码校验）；
4. 是否存在重复编号；
5. 是否存在无法解析的文件（输出 `Unresolved Files List` 等待人工确认）。

脚本只读取数据、不修改任何文件，可重复运行；退出码 `0` 表示 PASS，`1` 表示 FAIL。

## Adding New Data

1. 按上表命名规范放入对应目录（新增场景类型时，同步更新 `tools/check_dataset.py` 中的
   `ALLOWED_SCENE_KINDS` / `ALLOWED_TEMPLATE_GROUPS` 白名单）；
2. 运行 `python tools/check_dataset.py` 确认 PASS；
3. 图片不需要 `git add`（已被 `.gitignore` 忽略），只需保证目录结构与命名规范同步。
