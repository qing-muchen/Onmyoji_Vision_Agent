"""按本项目的状态/特征数据集规范，从既有素材构建 ``dataset/`` 新结构。

设计要点
--------
* **raw 是唯一真源**：``dataset/raw/<state>/`` 保存未修改的原始截图副本；
  ``states/`` 与 ``features/`` 一律**从 raw 派生**，不再直接读取旧素材目录。
* **states 是状态级派生**：``dataset/states/<state>/positive/`` 是"明确属于该状态的完整截图"，
  ``negative/`` 是"容易被误判为该状态、但实际属于其它状态"的截图副本。
* **features 是证据级派生**：按固定 ROI 裁剪，一个 detector 一种证据，各自 positive/negative。
  裁剪只做 ``crop``，不缩放、不压缩、不改色。
* 旧的 ``scenes/``、``templates/``、``states/transition_bottom``、``states/non_transition_bottom``
  以及 ``states/<state>/`` 下的历史扁平样本一律保留不动；本脚本只写入并重建自己负责的
  生成目录（``raw/`` 与 ``features/`` 整棵树、``states/<state>/positive|negative/``）。

脚本只读取旧素材，可重复运行。
用法::

    python tools/build_dataset.py            # 构建
    python tools/build_dataset.py --dry-run  # 只统计，不落盘
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "dataset"

TARGET_STATES: tuple[str, ...] = (
    "yard",
    "exploration",
    "battle",
    "battle_result",
    "battle_reward",
)
UNKNOWN = "unknown"
ALL_STATES = TARGET_STATES + (UNKNOWN,)

#: features 的 detector 定义：相对 ``features/`` 的目录路径 -> (ROI 比例 (x0, y0, x1, y1), 说明)
#: detector 名 = 路径中的 ``/`` 换成 ``_``（用于文件名前缀与 metadata.features）。
FEATURE_ROIS: dict[str, tuple[tuple[float, float, float, float], str]] = {
    "bottom_menu/yard": (
        (0.00, 0.80, 1.00, 1.00),
        "庭院型底部功能栏（图鉴/珍旅居/纳物库/组队/阴阳寮/商店/花合战/好友/阴阳术/式神录）",
    ),
    "bottom_menu/exploration": (
        (0.00, 0.80, 1.00, 1.00),
        "探索型底部功能栏（觉醒材料/御魂/结界突破/御灵/式神委派/秘闻副本/地域鬼王/妖气探索/英杰试炼/六道之门/契灵之境）",
    ),
    "chapter_panel": (
        (0.80, 0.15, 1.00, 0.70),
        "探索界面右侧章节选择栏（主线/玩法 + 第二十六~二十八章）",
    ),
    "battle/auto_button": (
        (0.00, 0.74, 0.12, 1.00),
        "战斗左下角自动/手动按钮",
    ),
    "battle/skill_area": (
        (0.60, 0.74, 0.92, 1.00),
        "战斗底部技能按钮区（妖术/普攻）",
    ),
    "battle/action_bar": (
        (0.90, 0.04, 1.00, 0.72),
        "战斗右侧行动顺序条",
    ),
    "battle/shikigami_icons": (
        (0.925, 0.09, 1.00, 0.60),
        "行动条上的式神头像",
    ),
    "victory_icon": (
        (0.25, 0.05, 0.85, 0.40),
        "战斗结算中心的胜利图标与「胜利」字样",
    ),
    "reward_panel": (
        (0.15, 0.15, 0.62, 0.90),
        "奖励展示中心区（奖励卡片 + 中央奖励宝箱）",
    ),
}

#: detector 目录路径 -> detector 名
def detector_name(path: str) -> str:
    """把 features 相对目录路径转成 detector 名（用于文件名前缀）。"""
    return path.replace("/", "_")


UNKNOWN_SUBDIRS: tuple[str, ...] = ("shop", "collection", "friends", "shikigami", "equipment", "other")

#: 战斗界面的 4 个证据（battle_002 实测右侧行动条不可见，单独处理）
BATTLE_FULL_FEATURES = (
    "battle_auto_button",
    "battle_skill_area",
    "battle_action_bar",
    "battle_shikigami_icons",
)
BATTLE_PARTIAL_FEATURES = ("battle_auto_button", "battle_skill_area")


@dataclass
class Sample:
    """一条源截图记录。"""

    src: Path
    state: str
    source: str
    name: str
    raw_rel: str = ""
    features: list[str] = field(default_factory=list)
    note: str = ""
    scene_only: bool = False
    subdir: str | None = None  # 仅 unknown 使用


# --------------------------------------------------------------------------
# 素材映射（显式登记，便于人工核对与后续扩展）
# --------------------------------------------------------------------------

#: non_transition_bottom 中实际属于目标状态的完整截图（其余归 unknown）
NON_TRANSITION_STATE_SAMPLES: dict[str, tuple[str, str]] = {
    "庭院界面.png": ("yard", "cap"),
    "探索界面.png": ("exploration", "ch26"),
    "探索-式神委派.png": ("exploration", "ch26"),
    "战斗界面.png": ("battle", "auto"),
}

#: unknown 的子分类归属
UNKNOWN_SUBDIR_OF: dict[str, str] = {
    "召唤-商店.png": "shop",
    "商店-皮肤屋.png": "shop",
    "商店-礼包屋.png": "shop",
    "珍旅居-千物宝库.png": "shop",
    "图鉴-式神.png": "collection",
    "图鉴界面.png": "collection",
    "契灵录.png": "collection",
    "纳物库.png": "collection",
    "好友.png": "friends",
    "好友界面.png": "friends",
    "式神录-队伍预设.png": "shikigami",
    "式神录.png": "shikigami",
    "阴阳术.png": "shikigami",
    "阵容助手界面.png": "shikigami",
    "式神录-御魂-整理界面.png": "equipment",
    "式神录-御魂-装配界面.png": "equipment",
    "探索-御魂-八岐大蛇-神罚副本.png": "equipment",
}

#: 旧 states/battle/ 中混装了 3 种状态，按实测画面拆分
BATTLE_SPLIT: dict[str, tuple[str, str]] = {
    "battle_001.png": ("battle", "auto"),
    "battle_002.png": ("battle", "auto"),
    "battle_003.png": ("battle_result", "win"),
    "battle_004.png": ("battle_reward", "item"),
}

#: 逐图人工标注
SAMPLE_NOTES: dict[str, str] = {
    "庭院界面.png": "庭院完整截图，底部功能栏与右侧竖排入口同时可见",
    "探索界面.png": "探索地图，章节栏完整（主线/玩法 + 第二十六~二十八章）",
    "探索-式神委派.png": "探索地图，但右侧为式神委派任务面板、无章节栏（chapter_panel 的困难负样本）",
    "战斗界面.png": "战斗界面，左下自动按钮 + 底部技能区 + 右侧行动条",
    "battle_001.png": "自动战斗，左下自动按钮 + 底部技能区 + 右侧行动条齐全",
    "battle_002.png": "战斗进行中，右侧行动条不可见（action_bar / shikigami_icons 的负样本）",
    "battle_003.png": "战斗胜利结算：中心胜利图标与「胜利」字样 + 式神结算列表",
    "battle_004.png": "奖励展示：上方奖励卡片 + 中央奖励宝箱 + 点击屏幕继续",
    "召唤界面.png": "召唤（抽卡）界面，无法归入五个目标状态",
    "战斗准备阶段.png": "战斗准备阶段：式神阵容 + 准备按钮，尚无战斗 UI（battle 的困难负样本）",
}


def _note_for(p: Path, state: str, source: str) -> str:
    if p.name in SAMPLE_NOTES:
        return SAMPLE_NOTES[p.name]
    if state == "yard" and source == "cap":
        return "庭院完整截图，底部功能栏完整（图鉴/珍旅居/纳物库/…）"
    if state == "yard":
        return "无 UI 纯场景图：仅返回/公告/切换按钮，无底部功能栏（scene_only）"
    if state == "exploration":
        return "探索地图，右侧章节栏与底部功能栏可见（章节 26~28）"
    if state == UNKNOWN:
        return "无法归入 yard/exploration/battle/battle_result/battle_reward 的有效游戏界面"
    return ""


def collect_samples() -> list[Sample]:
    """扫描旧素材，产出全部源截图记录（此时 raw_rel 尚未填充）。"""
    samples: list[Sample] = []

    # 1) states/yard 与 states/exploration（历史采集批次）
    for i, p in enumerate(sorted((DATASET / "states" / "yard").glob("*.png")), start=1):
        samples.append(Sample(p, "yard", "cap", f"yard_cap_{i:03d}.png",
                              features=["bottom_menu_yard"], note=_note_for(p, "yard", "cap")))
    for i, p in enumerate(sorted((DATASET / "states" / "exploration").glob("*.png")), start=1):
        samples.append(Sample(p, "exploration", "ch26", f"exploration_ch26_{i:03d}.png",
                              features=["bottom_menu_exploration", "chapter_panel"],
                              note=_note_for(p, "exploration", "ch26")))

    # 2) states/battle 的 4 张 -> 拆成 battle / battle_result / battle_reward
    counter = {"battle": 0, "battle_result": 0, "battle_reward": 0}
    for p in sorted((DATASET / "states" / "battle").glob("*.png")):
        state, source = BATTLE_SPLIT[p.name]
        counter[state] += 1
        if state == "battle":
            feats = list(BATTLE_PARTIAL_FEATURES if p.name == "battle_002.png" else BATTLE_FULL_FEATURES)
        elif state == "battle_result":
            feats = ["victory_icon"]
        else:
            feats = ["reward_panel"]
        samples.append(Sample(p, state, source, f"{state}_{source}_{counter[state]:03d}.png",
                              features=feats, note=_note_for(p, state, source)))

    # 3) non_transition_bottom：4 张归目标状态 + 其余归 unknown
    nt_dir = DATASET / "states" / "non_transition_bottom"
    yard_next = len(list((DATASET / "states" / "yard").glob("*.png"))) + 1
    expl_next = len(list((DATASET / "states" / "exploration").glob("*.png"))) + 1
    unknown_counter: dict[str, int] = {s: 0 for s in UNKNOWN_SUBDIRS}
    for p in sorted(nt_dir.glob("*.png")):
        if p.name in NON_TRANSITION_STATE_SAMPLES:
            state, source = NON_TRANSITION_STATE_SAMPLES[p.name]
            if state == "yard":
                name = f"yard_cap_{yard_next:03d}.png"
                yard_next += 1
                feats = ["bottom_menu_yard"]
            elif state == "exploration":
                name = f"exploration_ch26_{expl_next:03d}.png"
                expl_next += 1
                feats = ["bottom_menu_exploration"]
                if "式神委派" not in p.name:
                    feats.append("chapter_panel")
            else:
                counter["battle"] += 1
                name = f"battle_auto_{counter['battle']:03d}.png"
                feats = list(BATTLE_FULL_FEATURES)
            samples.append(Sample(p, state, source, name, features=feats,
                                  note=_note_for(p, state, source)))
        else:
            sub = UNKNOWN_SUBDIR_OF.get(p.name, "other")
            unknown_counter[sub] += 1
            samples.append(Sample(p, UNKNOWN, sub, f"unknown_{sub}_{unknown_counter[sub]:03d}.png",
                                  features=[], note=_note_for(p, UNKNOWN, sub), subdir=sub))

    # 4) scenes/yard：无 UI 纯场景图，归档进 raw 且只作 negative
    skin_counter: dict[str, int] = {}
    for p in sorted((DATASET / "scenes" / "yard").rglob("*.png")):
        skin_dir = p.parent.parent.name  # skin_XXX
        skin = f"skin{skin_dir.split('_')[-1]}"  # skin001（与目录 skin_001 对应）
        skin_counter[skin] = skin_counter.get(skin, 0) + 1
        samples.append(Sample(p, "yard", skin, f"yard_{skin}_{skin_counter[skin]:03d}.png",
                              features=[], note=_note_for(p, "yard", skin), scene_only=True))

    for s in samples:
        rel = Path(s.state) / (s.subdir or "") / s.name
        s.raw_rel = rel.as_posix()
    return samples


# --------------------------------------------------------------------------
# 构建
# --------------------------------------------------------------------------


def _copy(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if not src.is_file():
        raise FileNotFoundError(f"源素材缺失（请先恢复素材）: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _crop(src: Path, dst: Path, roi: tuple[float, float, float, float], dry_run: bool) -> None:
    if dry_run:
        return
    from PIL import Image

    x0, y0, x1, y1 = roi
    with Image.open(src) as im:
        w, h = im.size
        box = (int(round(w * x0)), int(round(h * y0)), int(round(w * x1)), int(round(h * y1)))
        with im.convert("RGB").crop(box) as out:
            dst.parent.mkdir(parents=True, exist_ok=True)
            out.save(dst, format="PNG", optimize=False)


#: 允许被清空重建的生成目录（相对 dataset/）——**绝不包含**存放历史素材的目录
GENERATED_DIRS: tuple[str, ...] = (
    "raw",
    "features",
    "states/yard/positive",
    "states/yard/negative",
    "states/exploration/positive",
    "states/exploration/negative",
    "states/battle/positive",
    "states/battle/negative",
    "states/battle_result/positive",
    "states/battle_result/negative",
    "states/battle_reward/positive",
    "states/battle_reward/negative",
    "states/unknown",
)

#: 历史素材目录：出现在这些目录下的文件一律不动
PROTECTED_DIRS: tuple[str, ...] = (
    "scenes",
    "templates",
    "states/transition_bottom",
    "states/non_transition_bottom",
    "states/yard",
    "states/exploration",
    "states/battle",
)


def _reset(path: Path, dry_run: bool) -> None:
    """清空一个**生成目录**（保留 .gitkeep）。

    只允许作用于 ``GENERATED_DIRS`` 登记的路径，且其下不得存在历史素材文件，
    以避免误删旧数据。
    """
    relative = path.relative_to(DATASET).as_posix()
    if relative not in GENERATED_DIRS:
        raise RuntimeError(f"拒绝清空未登记的目录（防误删）: {relative}")
    for protected in PROTECTED_DIRS:
        if relative == protected:
            raise RuntimeError(f"拒绝清空历史素材目录: {relative}")
    if dry_run or not path.exists():
        return
    for child in path.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _ensure_gitkeep(dry_run: bool) -> int:
    """为新结构目录补 ``.gitkeep``。

    图片被 ``.gitignore`` 忽略（``dataset/**/*.png``），目录本身需要占位文件才能进入 git。
    """
    roots = [DATASET / "raw", DATASET / "features"]
    roots += [DATASET / "states" / state for state in ALL_STATES]
    created = 0
    for root in roots:
        if not root.exists():
            continue
        for directory in [root, *(p for p in root.rglob("*") if p.is_dir())]:
            marker = directory / ".gitkeep"
            if marker.exists():
                continue
            if not dry_run:
                marker.touch()
            created += 1
    return created


def build(dry_run: bool = False) -> dict:
    samples = collect_samples()
    by_state: dict[str, list[Sample]] = {s: [] for s in ALL_STATES}
    for s in samples:
        by_state[s.state].append(s)

    # 正向样本 = 真正属于该状态、且带该状态 UI 证据的截图（scene_only 不算）
    positives: dict[str, list[Sample]] = {
        s: [x for x in by_state[s] if not x.scene_only] for s in ALL_STATES
    }
    full_shots = [x for s in ALL_STATES for x in by_state[s]]

    # ---- 1) raw/ ：唯一真源 ---------------------------------------------
    raw_root = DATASET / "raw"
    _reset(raw_root, dry_run)
    for s in samples:
        _copy(s.src, raw_root / s.raw_rel, dry_run)

    def raw_of(s: Sample) -> Path:
        """派生阶段的源文件：一律指向 raw/，保证 raw 是唯一真源。"""
        return raw_root / s.raw_rel

    # ---- 2) states/ ：从 raw 派生 ---------------------------------------
    states_root = DATASET / "states"
    state_counts: dict[str, dict[str, int]] = {}
    for state in TARGET_STATES:
        _reset(states_root / state / "positive", dry_run)
        _reset(states_root / state / "negative", dry_run)
        for s in positives[state]:
            _copy(raw_of(s), states_root / state / "positive" / s.name, dry_run)
        counter = 0
        for other in ALL_STATES:
            if other == state:
                continue
            for s in positives[other]:
                counter += 1
                _copy(raw_of(s), states_root / state / "negative" / f"{state}_{other}_{counter:03d}.png", dry_run)
        state_counts[state] = {"positive": len(positives[state]), "negative": counter}
    _reset(states_root / UNKNOWN, dry_run)
    for s in positives[UNKNOWN]:
        _copy(raw_of(s), states_root / UNKNOWN / (s.subdir or "other") / s.name, dry_run)
    state_counts[UNKNOWN] = {"categories": len({s.subdir for s in positives[UNKNOWN]})}

    # ---- 3) features/ ：从 raw 派生 -------------------------------------
    features_root = DATASET / "features"
    _reset(features_root, dry_run)
    feature_counts: dict[str, dict[str, int]] = {}
    for path, (roi, _desc) in FEATURE_ROIS.items():
        det = detector_name(path)
        positive_shots = [s for s in full_shots if det in s.features]
        negative_shots = [s for s in full_shots if det not in s.features]
        pos_counts: dict[str, int] = {}
        for s in positive_shots:
            pos_counts[s.source] = pos_counts.get(s.source, 0) + 1
            _crop(raw_of(s), features_root / path / "positive" / f"{det}_{s.source}_{pos_counts[s.source]:03d}.png", roi, dry_run)
        neg_counts: dict[str, int] = {}
        for s in negative_shots:
            key = "scene_only" if s.scene_only else s.state
            neg_counts[key] = neg_counts.get(key, 0) + 1
            _crop(raw_of(s), features_root / path / "negative" / f"{det}_{key}_{neg_counts[key]:03d}.png", roi, dry_run)
        feature_counts[path] = {"positive": len(positive_shots), "negative": len(negative_shots)}

    # ---- 4) metadata.json ----------------------------------------------
    metadata = _build_metadata(samples)
    if not dry_run:
        with (DATASET / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    gitkeeps = _ensure_gitkeep(dry_run)

    return {
        "samples": len(samples),
        "states": state_counts,
        "scene_only": sum(1 for s in samples if s.scene_only),
        "features": feature_counts,
        "metadata_entries": len(metadata),
        "gitkeep_created": gitkeeps,
        "dry_run": dry_run,
    }


def _build_metadata(samples: list[Sample]) -> list[dict]:
    entries: list[dict] = []
    for s in samples:
        paths = {
            "raw": f"raw/{s.raw_rel}",
            "original": s.src.relative_to(REPO_ROOT).as_posix(),
        }
        if not s.scene_only:
            if s.state == UNKNOWN:
                paths["state"] = f"states/{UNKNOWN}/{s.subdir or 'other'}/{s.name}"
            else:
                paths["state_positive"] = f"states/{s.state}/positive/{s.name}"
        entry: dict = {
            "image": f"raw/{s.raw_rel}",
            "state": s.state,
            "source": s.source,
            "features": s.features,
            "confidence": "manual",
            "note": s.note,
        }
        if s.subdir:
            entry["subcategory"] = s.subdir
        if s.scene_only:
            entry["scene_only"] = True
        entry["paths"] = paths
        entries.append(entry)
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 dataset/ 新结构（raw / states / features）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args(argv)

    report = build(dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
