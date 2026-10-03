# `raw/yard_live/` —— 当前版本庭院新采集（Phase 11.9-A）

本目录存放**当前版本真实客户端**的庭院截图，与历史 baseline `dataset/raw/yard/`（98 张，
永不改动）**严格分离**。

## 目录结构

```text
dataset/raw/yard_live/
├── README.md                     ← 本文件
├── .gitkeep
└── <capture_session_id>/         ← 一次连续采集 = 一个会话目录
    ├── <capture_session_id>_001.png
    ├── <capture_session_id>_002.png
    └── ...
```

**会话 ID 契约**：`yard_live_<YYYYMMDD>_s<NN>`（例如 `yard_live_20261003_s01`）。

**帧文件契约**：`<capture_session_id>_<NNN>.png`（`NNN` 从 `001` 开始）。

## 采集角色（`capture_role`）

| 角色 | 含义 | 能否用于模板生成 / 选择 |
| --- | --- | --- |
| `candidate` | 候选池 | ✅ **唯一**允许 |
| `holdout` | 留出池 | ❌ 禁止（也不得参与调参） |
| `diagnostic` | 诊断池（只定位问题） | ❌ 禁止 |
| `unknown` | 历史素材或未标注 | ❌ 禁止 |

**角色按会话整体分配**，不允许同一会话内混用（`session_role_conflicts` 会检出）。
未登记的新图默认 `unknown` —— **不会**被自动当成 candidate 或 holdout。

## 采集流程（由用户手动控制游戏）

```bash
# 0) 先确认能不能采（只探测窗口与前台状态，不落盘）
python tools/capture_yard_dataset.py --role candidate --dry-run

# 1) 用户手动进入庭院，保持窗口在前台，然后采集 candidate 会话
python tools/capture_yard_dataset.py --role candidate --count 5 --interval-ms 1500 \
    --session-id yard_live_20261003_s01

# 2) **离开庭院再重新进入**（制造真实会话边界），采集 holdout 会话
python tools/capture_yard_dataset.py --role holdout --count 5 --interval-ms 1500 \
    --session-id yard_live_20261003_s02

# 3) 审计：SHA / dHash / candidate↔holdout 泄漏 + Production36 / All61 探针
python tools/audit_yard_live.py
```

## 硬性约束

* 程序**只截图**：不激活窗口、不 `SetForegroundWindow`、不点击、不发送键盘、不自动导航；
* 窗口不存在 / 最小化 / **非前台** ⇒ 明确失败（退出码 `3` / `4`），**不写任何文件**；
* 保存**原客户区像素**：不 resize、不裁 ROI、不做增强；
* 重复 SHA256 拒绝、已有文件不覆盖、失败清理 `.tmp` 半文件；
* 路径严格限制在本目录内（越界写入会被拒绝）；
* 本目录**不会**被现有 benchmark 扫描（`benchmark_template_prefilter.collect_positives`
  只做 `raw/yard` 的非递归 `glob`），也不会被 `tools/raw_manifest.py` 计入历史 178 张计数
  （它单独出现在清单的 `live` 分区里）。

## 与模板的关系

本阶段（11.9-A）**不生成任何生产模板**：这里的新图既不会进入
`dataset/features/bottom_menu/yard/positive/`，也不会改动 `template_include` /
`threshold` / `roi_scale` / `prefilter_top_k`。模板重选属 Phase 11.9-B。
