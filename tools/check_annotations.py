"""元素级标注校验工具（Phase 11.1）：``tools/check_annotations.py``。

```bash
# 默认只读校验（不写任何文件）
python tools/check_annotations.py
python tools/check_annotations.py --annotations dataset/annotations/elements.jsonl
python tools/check_annotations.py --json
```

校验项（用户要求 10 条）：

1. 所有 ``source_id`` 指向真实 raw 原图；
2. 所有 bbox 坐标有限且合法；
3. bbox 宽高必须大于 0；
4. bbox 必须位于原图尺寸之内；
5. ``element_name`` 必须属于已登记语义目标（可执行登记表 ∪ 状态证据名）；
6. 同一 ``source_id`` 的标注不能出现冲突（同元素同区域多条不同 bbox / 重复 ID）；
7. 不允许已审核标注缺少必要审核信息（``verified`` 必须有 ``annotator`` 与 ``reviewer``）；
8. 不允许将未审核标注计入 verified ground truth（统计口径检查）；
9. 原图 SHA256 不得与清单登记值不一致（有清单时）；
10. 图片读取失败必须记录明确原因（``np.fromfile`` + ``cv2.imdecode``，中文路径不误报）。

**默认只读**：本工具**从不**修改 ``dataset/raw/`` 与标注文件。
自动修复需要显式 ``--write-normalized``（仅重写标注 JSONL，且不动 bbox 语义）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.geometry import Rect  # noqa: E402
from src.vision.dataset.annotations import (  # noqa: E402
    AnnotationStatus,
    AnnotationStore,
    ElementAnnotation,
    ReviewStatus,
    _parse_bbox,
)
from src.vision.dataset.sources import image_size  # noqa: E402
from src.vision.dataset.elements import known_names as registered_names  # noqa: E402
from src.vision.pipeline.target_adapter import ACTIONABLE_EVIDENCE  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_ANNOTATIONS = DATASET / "annotations" / "elements.jsonl"
DEFAULT_MANIFEST = DATASET / "manifests" / "raw_sha256.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

#: 允许出现在标注里的元素名 —— **唯一来源**：``src.vision.dataset.elements.ELEMENT_REGISTRY``
#: （Phase 11.3 收敛；此前这里另有一份硬编码副本）
KNOWN_ELEMENT_NAMES: frozenset[str] = frozenset(registered_names()) | {"transition_bottom"}


def _raw_index(dataset: Path) -> dict[str, Path]:
    """``raw/`` 下全部图片：``相对路径 → 绝对路径``。"""
    index: dict[str, Path] = {}
    raw_root = dataset / "raw"
    if not raw_root.is_dir():
        return index
    for path in sorted(raw_root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.relative_to(dataset).as_posix().casefold()] = path
    return index


def _manifest_digests(path: Path) -> dict[str, str]:
    """读取真源清单：``相对路径 → sha256``（无清单时为空）。"""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["path"]): str(item["sha256"]) for item in payload.get("entries", [])}


def check_annotations(
    *,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    dataset: Path = DATASET,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    """执行全部校验（只读）。

    Args:
        annotations_path: 标注 JSONL 路径。
        dataset: 数据根目录。
        manifest_path: ``raw`` 真源清单路径。

    Returns:
        报告字典（``ok`` / ``errors`` / ``warnings`` / ``stats`` / ``insufficient_data``）。
    """
    errors: list[str] = []
    warnings: list[str] = []
    store = AnnotationStore.load(annotations_path)
    for line_error in store.errors:
        errors.append(f"[解析] {line_error}")

    raw_index = _raw_index(dataset)
    digests = _manifest_digests(manifest_path)
    #: 反向索引：sha256 → raw 相对路径（用于"source_id 必须指向真实原图"的完整性口径）
    digest_to_path = {value: key for key, value in digests.items()}

    size_cache: dict[str, tuple[int, int]] = {}
    file_errors: list[str] = []

    for item in store:
        where = f"{item.source_id}/{item.element_name}"
        # 1) source_id → image_path 必须存在且位于 raw/
        image_path = item.image_path
        if not image_path:
            warnings.append(f"[1] {where}: 缺少 image_path（无法定位原图）")
        else:
            normalized = image_path.replace("\\", "/")
            if not normalized.lower().startswith("raw/"):
                errors.append(f"[1] {where}: image_path 不在 raw/ 下: {image_path}")
            elif normalized.casefold() not in raw_index:
                errors.append(f"[1] {where}: 原图不存在: {image_path}")
            else:
                absolute = raw_index[normalized.casefold()]
                key = normalized.casefold()
                if key not in size_cache:
                    width, height = image_size(absolute)
                    size_cache[key] = (width, height)
                    if width == 0 or height == 0:
                        file_errors.append(f"[10] {where}: 图片无法解码: {image_path}")
                width, height = size_cache[key]
                # 4) bbox 必须位于原图内（用**真实读取到的尺寸**，不用标注自报值）
                if width > 0 and height > 0:
                    rect = item.bbox
                    if not (0 <= rect.x and 0 <= rect.y and rect.right <= width and rect.bottom <= height):
                        errors.append(
                            f"[4] {where}: bbox {rect.describe()} 超出原图 {width}x{height}"
                                )
                    if item.image_width and int(item.image_width) != width:
                        warnings.append(
                            f"[4] {where}: image_width 自报 {item.image_width} != 实际 {width}"
                        )
                    if item.image_height and int(item.image_height) != height:
                        warnings.append(
                            f"[4] {where}: image_height 自报 {item.image_height} != 实际 {height}"
                        )
                # 9) SHA256 一致性（有清单时）
                registered = digests.get(normalized)
                if registered:
                    from src.vision.dataset.sources import compute_sha256

                    actual = compute_sha256(absolute)
                    if actual != registered:
                        errors.append(
                            f"[9] {where}: SHA256 与清单不一致（{image_path}）"
                        )

        # 2/3) bbox 合法与正宽高（构造期已校验，这里再走一遍显式路径以便报错定位）
        try:
            _parse_bbox([item.bbox.x, item.bbox.y, item.bbox.width, item.bbox.height])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[2/3] {where}: {exc}")

        # 5) element_name 必须登记
        if item.element_name not in KNOWN_ELEMENT_NAMES:
            errors.append(
                f"[5] {where}: 未登记的元素名（可选: {sorted(KNOWN_ELEMENT_NAMES)}）"
            )

        # 7) 已审核标注必须带审核信息，且必须是**独立**审核
        if item.annotation_status is AnnotationStatus.VERIFIED and item.review_status is ReviewStatus.APPROVED:
            if not item.annotator:
                errors.append(f"[7] {where}: 已审核标注缺少 annotator")
            if not item.reviewer:
                errors.append(f"[7] {where}: 已审核标注缺少 reviewer")
            if item.is_self_reviewed:
                errors.append(
                    f"[7] {where}: 自审（annotator == reviewer）不构成独立审核，"
                    "不得计入 ground truth"
                )

    # 6) 冲突
    for problem in store.conflicts():
        errors.append(f"[6] {problem}")

    # 8) 统计口径：verified 必须都满足 审核通过 + 独立审核通过
    verified = store.verified()
    for item in verified:
        if item.annotation_status is not AnnotationStatus.VERIFIED:
            errors.append(f"[8] {item.annotation_id}: 未审核却被计入 verified")
        if item.review_status is not ReviewStatus.APPROVED:
            errors.append(f"[8] {item.annotation_id}: 未独立审核却被计入 verified")

    stats = store.stats()
    insufficient = stats["annotations"] == 0
    return {
        "ok": not errors,
        "annotations_path": str(annotations_path.relative_to(ROOT))
        if ROOT in annotations_path.parents
        else str(annotations_path),
        "errors": errors,
        "warnings": warnings,
        "file_errors": file_errors,
        "stats": stats,
        "raw_count": len(raw_index),
        "manifest_present": bool(digests),
        "known_elements": sorted(KNOWN_ELEMENT_NAMES),
        "insufficient_data": insufficient,
        "insufficient_reason": (
            "尚无任何元素级人工标注（annotations == 0）；"
            "本阶段只建立契约与工具，不代表定位精度已验证"
            if insufficient
            else ""
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="元素级标注校验（默认只读）")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS), help="标注 JSONL 路径")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="raw 真源清单路径")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    parser.add_argument(
        "--write-normalized",
        action="store_true",
        help="把解析成功的标注规范化后**重写标注文件**（不修改 raw 图片；显式危险模式）",
    )
    args = parser.parse_args(argv)

    report = check_annotations(
        annotations_path=Path(args.annotations), manifest_path=Path(args.manifest)
    )
    if args.write_normalized:
        store = AnnotationStore.load(Path(args.annotations))
        written = store.save(Path(args.annotations))
        report["normalized_written"] = str(written)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"标注文件: {report['annotations_path']}")
        print(f"raw 原图: {report['raw_count']} 张 | 真源清单: {'有' if report['manifest_present'] else '无'}")
        print(f"标注统计: {json.dumps(report['stats'], ensure_ascii=False)}")
        if report["insufficient_data"]:
            print(f"!! insufficient_data: {report['insufficient_reason']}")
        for warning in report["warnings"]:
            print(f"  [warn] {warning}")
        for error in report["errors"]:
            print(f"  [error] {error}")
        print("Annotations: " + ("PASS" if report["ok"] else "FAIL"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
