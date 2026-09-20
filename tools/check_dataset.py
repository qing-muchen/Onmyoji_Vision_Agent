"""数据集结构与命名校验脚本。

校验 ``dataset/`` 目录：

1. 目录结构（``scenes/<scene_kind>/skin_XXX/scene_XXX/``、``templates/<group>/``）；
2. 图片命名（含"文件名编号与所在目录一致"）；
3. PNG 文件是否有效（文件头签名 + 完整解码）；
4. 是否存在重复编号；
5. 是否存在无法解析的文件（输出 Unresolved Files List，等待人工确认）。

脚本只读取数据、不修改任何文件，可重复运行。

用法::

    python tools/check_dataset.py
    python tools/check_dataset.py --dataset dataset

退出码：``0`` = PASS，``1`` = FAIL。

新增场景类型 / UI 元素类别时，请同步更新下方的 ``ALLOWED_SCENE_KINDS`` 与
``ALLOWED_TEMPLATE_GROUPS`` 白名单。
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
#: ``dataset/`` 根目录下允许出现的条目
ALLOWED_ROOT_ENTRIES: tuple[str, ...] = (".gitkeep", "README.md", "scenes", "templates")
#: 数据集目录内允许出现的非图片文件（占位与说明，不参与校验）
ALLOWED_NON_PNG_FILES: tuple[str, ...] = (".gitkeep", "README.md")
#: PNG 文件头签名
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: 场景文件相对层级：<scene_kind>/skin_XXX/scene_XXX/<file>
SCENE_RELATIVE_DEPTH = 4
#: 模板文件相对层级：<category>/<group>/<file>（例如 buttons/explore/xxx.png）
TEMPLATE_RELATIVE_DEPTH = 3


@dataclass
class ValidationReport:
    """校验结果汇总。

    Attributes:
        scenes: 合法的场景图片数量。
        templates: 合法的模板图片数量。
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
