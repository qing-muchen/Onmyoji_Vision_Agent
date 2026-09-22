"""数据集结构与命名校验脚本。

校验 ``dataset/`` 目录，新旧两套结构并存：

**旧结构（历史数据，保留兼容）**

1. ``scenes/<scene_kind>/skin_XXX/scene_XXX/``：场景数据；
2. ``templates/<category>/<group>/``：UI 模板数据；
3. ``states/transition_bottom/``、``states/non_transition_bottom/``：扁平状态样本
   （人工标注、允许中文名，因此**只校验目录结构与 PNG 有效性**）。

**新结构（Phase 4，由 ``tools/build_dataset.py`` 生成）**

4. ``raw/<state>/``：唯一真源，未修改的原始截图归档（``unknown/`` 下按子分类）；
5. ``states/<state>/positive|negative/``：状态级正/负样本（``unknown/`` 按子分类）；
6. ``features/<detector>/positive|negative/``：证据级 ROI 裁剪样本。

新增结构一律要求命名符合 ``{prefix}_{source}_{NNN}.png``（三位编号、全小写、
下划线分隔），且文件名前缀与所在目录一致。

脚本只读取数据、不修改任何文件，可重复运行。

用法::

    python tools/check_dataset.py
    python tools/check_dataset.py --dataset dataset

退出码：``0`` = PASS，``1`` = FAIL。

新增场景类型 / UI 元素类别 / 状态 / detector 时，请同步更新下方的
``ALLOWED_SCENE_KINDS``、``ALLOWED_TEMPLATE_GROUPS``、``ALLOWED_STATE_GROUPS``、
``ALLOWED_RAW_STATES`` 与 ``ALLOWED_FEATURE_DETECTORS`` 白名单。
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: 场景文件命名：<scene_kind>_skinNNN_sceneNNN_frameNNN.png
SCENE_FILE_PATTERN = re.compile(
    r"^(?P<kind>[a-z][a-z0-9]*)_skin(?P<skin>\d{3})_scene(?P<scene>\d{3})_frame(?P<frame>\d{3})\.png$"
)
#: 模板文件命名：<element_type>_<element_name>_skinNNN_sceneNNN.png
TEMPLATE_FILE_PATTERN = re.compile(
    r"^(?P<element_type>[a-z][a-z0-9]*)_(?P<element_name>[a-z][a-z0-9]*)"
    r"_skin(?P<skin>\d{3})_scene(?P<scene>\d{3})\.png$"
)
#: 目录命名
SKIN_DIR_PATTERN = re.compile(r"^skin_\d{3}$")
SCENE_DIR_PATTERN = re.compile(r"^scene_\d{3}$")

#: 允许的场景类型（新增场景时在此登记）
ALLOWED_SCENE_KINDS: tuple[str, ...] = ("yard",)
#: 允许的模板分组（相对 ``templates/`` 的路径，新增 UI 元素类别时在此登记）
ALLOWED_TEMPLATE_GROUPS: tuple[str, ...] = ("buttons/explore",)
#: 旧状态样本分组：扁平 PNG，人工标注（允许中文名），不校验命名规范
LEGACY_STATE_GROUPS: tuple[str, ...] = ("transition_bottom", "non_transition_bottom")
#: 新状态分组：``states/<state>/positive|negative/``，要求命名规范
STATE_GROUPS: tuple[str, ...] = ("yard", "exploration", "battle", "battle_result", "battle_reward")
#: 兜底状态分组：``states/unknown/<category>/``
UNKNOWN_GROUP = "unknown"
#: 允许的状态样本分组（相对 ``states/`` 的目录名，新增状态数据时在此登记）
ALLOWED_STATE_GROUPS: tuple[str, ...] = LEGACY_STATE_GROUPS + STATE_GROUPS + (UNKNOWN_GROUP,)
#: ``states/unknown/`` 下允许的来源子分类
ALLOWED_UNKNOWN_CATEGORIES: tuple[str, ...] = (
    "shop",
    "collection",
    "friends",
    "shikigami",
    "equipment",
    "other",
)
#: ``raw/`` 下允许的状态目录
ALLOWED_RAW_STATES: tuple[str, ...] = STATE_GROUPS + (UNKNOWN_GROUP,)
#: 允许的 features detector（相对 ``features/`` 的目录路径，可含一级分组）
ALLOWED_FEATURE_DETECTORS: tuple[str, ...] = (
    "bottom_menu/yard",
    "bottom_menu/exploration",
    "chapter_panel",
    "battle/auto_button",
    "battle/skill_area",
    "battle/action_bar",
    "battle/shikigami_icons",
    "victory_icon",
    "reward_panel",
)
#: 正/负样本子目录名
SAMPLE_POLARITIES: tuple[str, ...] = ("positive", "negative")
#: ``dataset/`` 根目录下允许出现的条目
ALLOWED_ROOT_ENTRIES: tuple[str, ...] = (
    ".gitkeep",
    "README.md",
    "metadata.json",
    "scenes",
    "templates",
    "states",
    "raw",
    "features",
)
#: 数据集目录内允许出现的非图片文件（占位与说明，不参与校验）
ALLOWED_NON_PNG_FILES: tuple[str, ...] = (".gitkeep", "README.md", "metadata.json")
#: PNG 文件头签名
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: 场景文件相对层级：<scene_kind>/skin_XXX/scene_XXX/<file>
SCENE_RELATIVE_DEPTH = 4
#: 模板文件相对层级：<category>/<group>/<file>（例如 buttons/explore/xxx.png）
TEMPLATE_RELATIVE_DEPTH = 3
#: 新结构样本命名：<prefix>_<source>_<三位编号>.png —— prefix 之后的部分
SAMPLE_TAIL_PATTERN = re.compile(r"^(?P<source>[a-z][a-z0-9_]*?)_(?P<num>\d{3})\.png$")


@dataclass
class ValidationReport:
    """校验结果汇总。

    Attributes:
        scenes: 合法的场景图片数量。
        templates: 合法的模板图片数量。
        state_samples: 各状态样本分组的合法图片数量（``分组名 -> 数量``）。
        legacy_state_samples: 旧扁平状态样本分组的图片数量。
        raw_samples: ``raw/<state>/`` 下的合法图片数量（``state -> 数量``）。
        feature_samples: 各 detector 的正/负样本数量（``detector -> {"positive": n, "negative": n}``）。
        skin_directories: 出现过的皮肤目录名集合。
        scene_directories: 出现过的场景目录相对路径集合。
        template_groups: 出现过的模板分组集合。
        invalid: 结构或文件非法的问题列表。
        unresolved: 无法解析的文件列表（需要人工确认）。
        duplicates: 重复编号列表。
        warnings: 非致命提示。
    """

    scenes: int = 0
    templates: int = 0
    state_samples: dict[str, int] = field(default_factory=dict)
    legacy_state_samples: dict[str, int] = field(default_factory=dict)
    raw_samples: dict[str, int] = field(default_factory=dict)
    feature_samples: dict[str, dict[str, int]] = field(default_factory=dict)
    skin_directories: set[str] = field(default_factory=set)
    scene_directories: set[str] = field(default_factory=set)
    template_groups: set[str] = field(default_factory=set)
    invalid: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """是否通过校验（无非法项、无未解析项、无重复编号）。"""
        return not (self.invalid or self.unresolved or self.duplicates)


def check_png(path: Path) -> str | None:
    """校验单个 PNG 文件是否有效。

    Args:
        path: PNG 文件路径。

    Returns:
        错误描述；文件有效时返回 ``None``。
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(len(PNG_SIGNATURE))
    except OSError as exc:
        return f"无法读取文件: {exc}"
    if header != PNG_SIGNATURE:
        return "不是合法的 PNG（文件头签名不匹配）"

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - pillow 属于运行时依赖
        return None
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - 任何解码异常都视为文件无效
        return f"PNG 解码校验失败: {type(exc).__name__}: {exc}"
    return None


def _collect_non_png_files(root: Path, label: str, report: ValidationReport) -> None:
    """把非 PNG 文件记为未解析项（数据集只接受 PNG，占位/说明文件除外）。"""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() == ".png":
            continue
        if path.name in ALLOWED_NON_PNG_FILES:
            continue
        report.unresolved.append(f"{label}/{path.relative_to(root).as_posix()}（非 PNG 文件）")


def _validate_scene_directories(root: Path, report: ValidationReport) -> None:
    """校验 ``scenes/`` 下的目录层级与命名。"""
    for kind_dir in sorted(entry for entry in root.iterdir() if entry.is_dir()):
        if kind_dir.name not in ALLOWED_SCENE_KINDS:
            report.invalid.append(
                f"{kind_dir.as_posix()}: 未知场景类型（允许: {', '.join(ALLOWED_SCENE_KINDS)}）"
            )
            continue
        for skin_dir in sorted(entry for entry in kind_dir.iterdir() if entry.is_dir()):
            if not SKIN_DIR_PATTERN.match(skin_dir.name):
                report.invalid.append(f"{skin_dir.as_posix()}: 目录名应为 skin_XXX")
                continue
            report.skin_directories.add(skin_dir.name)
            for scene_dir in sorted(entry for entry in skin_dir.iterdir() if entry.is_dir()):
                if not SCENE_DIR_PATTERN.match(scene_dir.name):
                    report.invalid.append(f"{scene_dir.as_posix()}: 目录名应为 scene_XXX")
                    continue
                report.scene_directories.add(f"{kind_dir.name}/{skin_dir.name}/{scene_dir.name}")
                if any(entry.is_dir() for entry in scene_dir.iterdir()):
                    report.invalid.append(f"{scene_dir.as_posix()}: scene 目录下不应再包含子目录")


def _validate_scene_files(root: Path, report: ValidationReport) -> None:
    """校验 ``scenes/`` 下的文件命名、编号一致性与 PNG 有效性。"""
    seen: dict[tuple[str, str, str, str], str] = {}
    frames_by_scene: dict[str, set[int]] = {}

    for path in sorted(root.rglob("*.png")):
        relative = path.relative_to(root)
        label = f"scenes/{relative.as_posix()}"
        parts = relative.parts
        if len(parts) != SCENE_RELATIVE_DEPTH:
            report.invalid.append(f"{label}: 层级应为 <scene_kind>/skin_XXX/scene_XXX/<file>")
            continue

        kind, skin_dir, scene_dir, name = parts
        matched = SCENE_FILE_PATTERN.match(name)
        if matched is None:
            report.unresolved.append(f"{label}（文件名无法解析）")
            continue
        if matched.group("kind") != kind:
            report.invalid.append(f"{label}: 文件名中的场景类型与目录（{kind}）不一致")
            continue
        if f"skin_{matched.group('skin')}" != skin_dir:
            report.invalid.append(f"{label}: 文件名中的 skin 编号与目录（{skin_dir}）不一致")
            continue
        if f"scene_{matched.group('scene')}" != scene_dir:
            report.invalid.append(f"{label}: 文件名中的 scene 编号与目录（{scene_dir}）不一致")
            continue

        key = (kind, matched.group("skin"), matched.group("scene"), matched.group("frame"))
        if key in seen:
            report.duplicates.append(f"{label} 与 {seen[key]} 编号重复")
            continue
        seen[key] = label
        frames_by_scene.setdefault(f"{kind}/{skin_dir}/{scene_dir}", set()).add(
            int(matched.group("frame"))
        )

        error = check_png(path)
        if error:
            report.invalid.append(f"{label}: {error}")
            continue
        report.scenes += 1

    for scene_key, frames in sorted(frames_by_scene.items()):
        missing = sorted(set(range(1, max(frames) + 1)) - frames)
        if missing:
            report.warnings.append(f"scenes/{scene_key}: frame 编号不连续，缺少 {missing}")


def _validate_scenes(root: Path, report: ValidationReport) -> None:
    """校验场景数据目录。"""
    if not root.is_dir():
        report.warnings.append("scenes/ 不存在（尚未采集场景数据？）")
        return
    _validate_scene_directories(root, report)
    _validate_scene_files(root, report)
    _collect_non_png_files(root, "scenes", report)


def _validate_templates(root: Path, report: ValidationReport) -> None:
    """校验 UI 模板数据目录。"""
    if not root.is_dir():
        report.warnings.append("templates/ 不存在（尚未采集模板数据？）")
        return

    seen: dict[tuple[str, str, str, str], str] = {}
    for path in sorted(root.rglob("*.png")):
        relative = path.relative_to(root)
        label = f"templates/{relative.as_posix()}"
        parts = relative.parts
        if len(parts) != TEMPLATE_RELATIVE_DEPTH:
            report.invalid.append(f"{label}: 层级应为 <category>/<group>/<file>")
            continue

        group_path = "/".join(parts[:-1])
        if group_path not in ALLOWED_TEMPLATE_GROUPS:
            report.invalid.append(
                f"{label}: 未知模板分组（允许: {', '.join(ALLOWED_TEMPLATE_GROUPS)}）"
            )
            continue
        report.template_groups.add(group_path)

        matched = TEMPLATE_FILE_PATTERN.match(parts[-1])
        if matched is None:
            report.unresolved.append(f"{label}（文件名无法解析）")
            continue
        if matched.group("element_name") != parts[-2]:
            report.invalid.append(
                f"{label}: 文件名中的元素名称与目录（{parts[-2]}）不一致；"
                "命名应为 <element_type>_<element_name>_skinNNN_sceneNNN.png"
            )
            continue

        key = (
            matched.group("element_type"),
            matched.group("element_name"),
            matched.group("skin"),
            matched.group("scene"),
        )
        if key in seen:
            report.duplicates.append(f"{label} 与 {seen[key]} 编号重复")
            continue
        seen[key] = label

        error = check_png(path)
        if error:
            report.invalid.append(f"{label}: {error}")
            continue
        report.templates += 1

    _collect_non_png_files(root, "templates", report)


def _validate_sample_files(
    directory: Path,
    label: str,
    expected_prefix: str,
    report: ValidationReport,
) -> int:
    """校验一个样本目录下 PNG 的命名与有效性。

    命名规范：``<expected_prefix>_<source>_<三位编号>.png``。

    Args:
        directory: 待校验的目录（只校验其直接子文件）。
        label: 报告中使用的路径标签。
        expected_prefix: 文件名前缀（应与所在目录一致）。
        report: 校验结果汇总。

    Returns:
        合法样本数量。
    """
    count = 0
    seen: dict[tuple[str, str], str] = {}
    prefix_part = f"{expected_prefix}_"
    for path in sorted(directory.rglob("*.png")):
        relative = path.relative_to(directory).as_posix()
        full = f"{label}/{relative}"
        if path.parent != directory:
            report.invalid.append(f"{full}: 样本目录下不应再包含子目录")
            continue
        if not path.name.startswith(prefix_part):
            report.invalid.append(
                f"{full}: 文件名应以 {expected_prefix}_ 开头（前缀须与所在目录一致）"
            )
            continue
        matched = SAMPLE_TAIL_PATTERN.match(path.name[len(prefix_part):])
        if matched is None:
            report.unresolved.append(
                f"{full}（文件名无法解析，应为 {expected_prefix}_<source>_<三位编号>.png）"
            )
            continue
        key = (matched.group("source"), matched.group("num"))
        if key in seen:
            report.duplicates.append(f"{full} 与 {seen[key]} 编号重复")
            continue
        seen[key] = full

        error = check_png(path)
        if error:
            report.invalid.append(f"{full}: {error}")
            continue
        count += 1
    return count


def _validate_states(root: Path, report: ValidationReport) -> None:
    """校验 ``states/`` 状态样本目录（新旧结构并存）。

    * 旧扁平分组（``transition_bottom`` / ``non_transition_bottom``）：人工标注、
      文件名含中文，因此只校验结构与 PNG 有效性；
    * 新状态分组（``yard`` / ``exploration`` / …）：``positive`` / ``negative`` 子目录，
      要求命名规范；同目录下的历史扁平 PNG 保留兼容、宽松校验；
    * ``unknown``：按来源子分类（``shop`` / ``collection`` / …）。
    """
    if not root.is_dir():
        report.warnings.append("states/ 不存在（尚未采集状态样本？）")
        return

    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            if entry.name not in ALLOWED_STATE_GROUPS:
                report.invalid.append(
                    f"{entry.as_posix()}: 未知状态样本目录"
                    f"（允许: {', '.join(ALLOWED_STATE_GROUPS)}）"
                )
            continue
        if entry.name not in ALLOWED_NON_PNG_FILES:
            report.warnings.append(f"{entry.as_posix()}: states 根目录下的非样本条目（已忽略）")

    # ---- 旧扁平分组 ----
    for group in LEGACY_STATE_GROUPS:
        directory = root / group
        if not directory.is_dir():
            report.warnings.append(f"states/{group} 不存在（尚未采集该组样本？）")
            continue
        count = 0
        for path in sorted(directory.rglob("*.png")):
            label = f"states/{path.relative_to(root).as_posix()}"
            if path.parent != directory:
                report.invalid.append(f"{label}: 状态样本目录下不应再包含子目录")
                continue
            error = check_png(path)
            if error:
                report.invalid.append(f"{label}: {error}")
                continue
            count += 1
        report.legacy_state_samples[group] = count

    # ---- 新状态分组 ----
    for group in STATE_GROUPS:
        directory = root / group
        if not directory.is_dir():
            report.warnings.append(f"states/{group} 不存在（尚未构建新结构？）")
            continue
        total = 0
        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                if entry.name not in SAMPLE_POLARITIES:
                    report.invalid.append(
                        f"{entry.as_posix()}: states/{group} 下只允许 "
                        f"{'/'.join(SAMPLE_POLARITIES)} 子目录"
                    )
                    continue
                total += _validate_sample_files(
                    entry, f"states/{group}/{entry.name}", group, report
                )
            elif entry.suffix.lower() == ".png":
                # 历史扁平样本：保留兼容
                error = check_png(entry)
                if error:
                    report.invalid.append(f"{entry.as_posix()}: {error}")
        report.state_samples[group] = total

    # ---- unknown ----
    unknown_dir = root / UNKNOWN_GROUP
    if not unknown_dir.is_dir():
        report.warnings.append(f"states/{UNKNOWN_GROUP} 不存在（尚未构建新结构？）")
        return
    total = 0
    for entry in sorted(unknown_dir.iterdir()):
        if entry.is_dir():
            if entry.name not in ALLOWED_UNKNOWN_CATEGORIES:
                report.invalid.append(
                    f"{entry.as_posix()}: 未知 unknown 子分类"
                    f"（允许: {', '.join(ALLOWED_UNKNOWN_CATEGORIES)}）"
                )
                continue
            total += _validate_sample_files(
                entry, f"states/{UNKNOWN_GROUP}/{entry.name}", UNKNOWN_GROUP, report
            )
        elif entry.suffix.lower() == ".png":
            error = check_png(entry)
            if error:
                report.invalid.append(f"{entry.as_posix()}: {error}")
    report.state_samples[UNKNOWN_GROUP] = total


def _validate_raw(root: Path, report: ValidationReport) -> None:
    """校验 ``raw/`` 原始截图归档（唯一真源）。

    结构：``raw/<state>/``，``unknown`` 下再分来源子分类；
    文件命名：``{state}_{source}_{NNN}.png``。
    """
    if not root.is_dir():
        report.warnings.append("raw/ 不存在（尚未构建新结构？运行 tools/build_dataset.py）")
        return

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            if entry.name not in ALLOWED_NON_PNG_FILES:
                report.warnings.append(f"{entry.as_posix()}: raw 根目录下的非样本条目（已忽略）")
            continue
        if entry.name not in ALLOWED_RAW_STATES:
            report.invalid.append(
                f"{entry.as_posix()}: 未知状态目录（允许: {', '.join(ALLOWED_RAW_STATES)}）"
            )
            continue
        if entry.name == UNKNOWN_GROUP:
            total = 0
            for sub in sorted(entry.iterdir()):
                if not sub.is_dir():
                    if sub.name not in ALLOWED_NON_PNG_FILES:
                        report.warnings.append(f"{sub.as_posix()}: raw/unknown 下的非目录条目（已忽略）")
                    continue
                if sub.name not in ALLOWED_UNKNOWN_CATEGORIES:
                    report.invalid.append(
                        f"{sub.as_posix()}: 未知 unknown 子分类"
                        f"（允许: {', '.join(ALLOWED_UNKNOWN_CATEGORIES)}）"
                    )
                    continue
                total += _validate_sample_files(
                    sub, f"raw/{UNKNOWN_GROUP}/{sub.name}", UNKNOWN_GROUP, report
                )
            report.raw_samples[UNKNOWN_GROUP] = total
        else:
            report.raw_samples[entry.name] = _validate_sample_files(
                entry, f"raw/{entry.name}", entry.name, report
            )


def _validate_features(root: Path, report: ValidationReport) -> None:
    """校验 ``features/`` 证据级 ROI 样本。

    结构：``features/<detector>/positive|negative/``（detector 可含一级分组，
    如 ``bottom_menu/yard``、``battle/auto_button``）；
    文件命名：``{detector}_{source}_{NNN}.png``。
    """
    if not root.is_dir():
        report.warnings.append("features/ 不存在（尚未构建新结构？运行 tools/build_dataset.py）")
        return

    found: set[str] = set()
    for detector in ALLOWED_FEATURE_DETECTORS:
        directory = root / detector
        if not directory.is_dir():
            report.invalid.append(f"features/{detector} 不存在（detector 目录缺失）")
            continue
        found.add(detector)
        label = detector.replace("/", "_")
        counts = {"positive": 0, "negative": 0}
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                if entry.name not in ALLOWED_NON_PNG_FILES:
                    report.warnings.append(f"{entry.as_posix()}: detector 下的非目录条目（已忽略）")
                continue
            if entry.name not in SAMPLE_POLARITIES:
                report.invalid.append(
                    f"{entry.as_posix()}: detector 下只允许 {'/'.join(SAMPLE_POLARITIES)} 子目录"
                )
                continue
            counts[entry.name] = _validate_sample_files(
                entry, f"features/{detector}/{entry.name}", label, report
            )
        report.feature_samples[detector] = counts

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if any(entry.name == d.split("/")[0] for d in ALLOWED_FEATURE_DETECTORS):
            continue
        report.invalid.append(f"{entry.as_posix()}: 未知 features 分组")


def _validate_root_entries(dataset_root: Path, report: ValidationReport) -> None:
    """检查 ``dataset/`` 根目录下是否有未登记的条目（例如旧结构残留）。"""
    for entry in sorted(dataset_root.iterdir()):
        if entry.name not in ALLOWED_ROOT_ENTRIES:
            report.warnings.append(f"{entry.name}: dataset 根目录下的未登记条目，请确认是否为旧结构残留")


def validate(dataset_root: Path) -> ValidationReport:
    """校验数据集目录。

    Args:
        dataset_root: 数据集根目录。

    Returns:
        校验结果汇总。
    """
    report = ValidationReport()
    if not dataset_root.is_dir():
        report.warnings.append(f"dataset 目录不存在: {dataset_root}")
        return report
    _validate_root_entries(dataset_root, report)
    _validate_scenes(dataset_root / "scenes", report)
    _validate_templates(dataset_root / "templates", report)
    _validate_states(dataset_root / "states", report)
    _validate_raw(dataset_root / "raw", report)
    _validate_features(dataset_root / "features", report)
    return report


def print_report(dataset_root: Path, report: ValidationReport) -> None:
    """打印校验报告。

    Args:
        dataset_root: 数据集根目录。
        report: 校验结果。
    """
    print("Dataset Validation Report")
    print("-" * 25)
    print()
    print("Dataset Path:")
    print(dataset_root.as_posix())
    print()
    print("Dataset:")
    print("PASS" if report.passed else "FAIL")
    print()
    print("Scenes:")
    print(
        f"{report.scenes} images "
        f"({len(report.scene_directories)} scene dirs, {len(report.skin_directories)} skins)"
    )
    print()
    print("Templates:")
    print(f"{report.templates} images ({len(report.template_groups)} groups)")
    print()
    print("States:")
    if report.state_samples or report.legacy_state_samples:
        for group, count in report.legacy_state_samples.items():
            print(f"{group} (legacy): {count} images")
        for group, count in report.state_samples.items():
            print(f"{group}: {count} images")
    else:
        print("none")
    print()
    print("Raw:")
    if report.raw_samples:
        for state, count in report.raw_samples.items():
            print(f"{state}: {count} images")
        print(f"total: {sum(report.raw_samples.values())} images")
    else:
        print("none")
    print()
    print("Features:")
    if report.feature_samples:
        for detector, counts in report.feature_samples.items():
            print(f"{detector}: positive {counts['positive']}, negative {counts['negative']}")
        print(f"total: {sum(c['positive'] + c['negative'] for c in report.feature_samples.values())} images")
    else:
        print("none")
    print()

    sections = (
        ("Invalid Files", report.invalid),
        ("Unresolved Files", report.unresolved),
        ("Duplicate Entries", report.duplicates),
    )
    for title, items in sections:
        print(f"{title}:")
        print(str(len(items)))
        for item in items:
            print(f"  - {item}")
        print()

    if report.warnings:
        print("Warnings:")
        for item in report.warnings:
            print(f"  - {item}")
        print()


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        已注册参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="check_dataset",
        description="校验 dataset 目录结构、命名规范与 PNG 有效性（只读，可重复运行）",
    )
    parser.add_argument("--dataset", "-d", default="dataset", help="数据集根目录，默认 dataset")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数列表，省略时读取 ``sys.argv[1:]``。

    Returns:
        ``0`` 表示 PASS，``1`` 表示 FAIL。
    """
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset)
    report = validate(dataset_root)
    print_report(dataset_root, report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
