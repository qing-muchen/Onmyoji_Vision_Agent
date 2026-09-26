"""最小本地元素级标注工具（Phase 11.2）：``tools/annotate.py``。

```bash
# 1) 新增一条**候选草稿**（默认 pending_review；agent / 人工都从这个入口写）
python tools/annotate.py add \
    --image raw/yard/yard_cap_001.png --element battle_auto_button \
    --bbox 121,112,96,164 --region ui_element --visibility visible \
    --annotator agent-draft --note "目视确认：左侧挂饰「探索」"

# 2) 列出当前标注与状态统计
python tools/annotate.py list

# 3) 渲染 bbox 预览图（写到 --out 指定的临时目录，**不修改原图**）
python tools/annotate.py render --out %TEMP%/ova113/preview

# 4) 人工/独立审核（唯一能产生 verified 的入口；自审被拒绝）
python tools/annotate.py promote --id <annotation_id> \
    --annotator human-a --reviewer human-b --approve

# 5) 导出待审核清单（供独立审核者使用）
python tools/annotate.py pending
```

**安全与边界**（用户要求）：

* 只写 ``dataset/annotations/`` 下的 JSONL：读取 → 修改 → **原子替换**
  （同目录临时文件 + ``os.replace``），任何一步失败都不破坏已有标注；写前自动备份一次；
* **从不修改** ``dataset/raw/`` 与任何图片；
* **不向游戏窗口发送任何输入**（工具只做文件与图像渲染）；
* bbox 是"原始帧客户区像素"，工具只做**坐标校验**（有限 / 正宽高 / 在原图内），不做裁剪；
* ``promote`` 是**唯一**能把状态推进到 ``verified`` 的入口，且强制
  ``annotator != reviewer``（独立审核）与显式 ``--approve``。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.geometry import Rect  # noqa: E402
from src.vision.dataset.annotations import (  # noqa: E402
    AnnotationError,
    AnnotationStatus,
    AnnotationStore,
    ElementAnnotation,
    ElementRegionKind,
    ElementVisibility,
    ReviewStatus,
)
from src.vision.dataset.sources import compute_sha256, image_size  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_ANNOTATIONS = DATASET / "annotations" / "elements.jsonl"
DEFAULT_SOURCES = DATASET / "annotations" / "sources.jsonl"
BACKUP_SUFFIX = ".bak"

#: 只有这些元素名允许被标注（可执行登记表 ∪ 状态证据登记表）
ALLOWED_ELEMENTS: frozenset[str] = frozenset(
    {
        "battle_auto_button",
        "battle_skill_area",
        "battle_action_bar",
        "battle_shikigami_icons",
        "bottom_menu_yard",
        "bottom_menu_exploration",
        "chapter_panel",
        "victory_icon",
        "reward_panel",
    }
)

#: 允许被登记为"可点击"的元素（与 ``src.vision.pipeline.target_adapter`` 对齐）
CLICKABLE_ALLOWED: frozenset[str] = frozenset({"battle_auto_button", "REWARD_CONFIRM"})


def _parse_bbox(text: str) -> Rect:
    """解析 ``x,y,w,h``。"""
    parts = [item.strip() for item in str(text).replace(";", ",").split(",") if item.strip()]
    if len(parts) != 4:
        raise AnnotationError(f"--bbox 需要 4 个数（x,y,w,h），当前: {text!r}")
    try:
        x, y, width, height = (float(item) for item in parts)
    except ValueError as exc:
        raise AnnotationError(f"--bbox 数值非法: {text!r}") from exc
    return Rect(x=x, y=y, width=width, height=height)


def _source_id_for(image_relative: str) -> str:
    """由原图内容派生 ``source_id``（缺失原图时报错）。"""
    target = DATASET / image_relative.replace("\\", "/")
    if not target.is_file():
        raise AnnotationError(f"原图不存在: {image_relative}")
    return f"src_{compute_sha256(target)[:16]}"


def _atomic_write(path: Path, text: str) -> None:
    """原子写入（临时文件 + ``os.replace``），并先做一次备份。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        shutil.copy2(path, path.with_suffix(path.suffix + BACKUP_SUFFIX))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def cmd_add(args: argparse.Namespace) -> int:
    """新增一条候选草稿（默认 pending_review）。"""
    store = AnnotationStore.load(Path(args.annotations))
    image_relative = str(args.image).replace("\\", "/")
    element = str(args.element)
    if element not in ALLOWED_ELEMENTS:
        raise AnnotationError(f"未登记的元素名: {element}（可选: {sorted(ALLOWED_ELEMENTS)}）")
    bbox = _parse_bbox(args.bbox)
    width, height = image_size(DATASET / image_relative)
    if width <= 0 or height <= 0:
        raise AnnotationError(f"原图无法读取: {image_relative}")
    if not (0 <= bbox.x and 0 <= bbox.y and bbox.right <= width and bbox.bottom <= height):
        raise AnnotationError(
            f"bbox {bbox.describe()} 超出原图 {width}x{height}（工具不做裁剪）"
        )
    region = ElementRegionKind(args.region)
    if region is ElementRegionKind.CLICKABLE_VERIFIED and element not in CLICKABLE_ALLOWED:
        raise AnnotationError(
            f"{element} 未登记为可点击目标，禁止标记 clickable_verified"
            f"（可点击登记表: {sorted(CLICKABLE_ALLOWED)}）"
        )
    annotation = ElementAnnotation(
        source_id=_source_id_for(image_relative),
        image_path=image_relative,
        element_name=element,
        bbox=bbox,
        region_kind=region,
        visibility=ElementVisibility(args.visibility),
        image_width=width,
        image_height=height,
        annotation_status=AnnotationStatus(args.status),
        annotator=str(args.annotator),
        review_status=ReviewStatus.NOT_REVIEWED,
        note=str(args.note or ""),
    )
    store.add(annotation)
    _atomic_write(Path(args.annotations), store.to_jsonl())
    print(json.dumps(annotation.to_dict(), ensure_ascii=False))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """列出标注与统计。"""
    store = AnnotationStore.load(Path(args.annotations))
    for item in store:
        print(
            f"{item.annotation_status.value:>14} | {item.review_status.value:>13} | "
            f"{item.region_kind.value:>18} | {item.element_name:>22} | "
            f"{item.bbox.describe():>26} | {item.image_path}"
        )
    print("-" * 120)
    print(json.dumps(store.stats(), ensure_ascii=False))
    print(f"独立原图: {len(store.source_ids())} | 冲突: {len(store.conflicts())} | 错误: {len(store.errors)}")
    return 0


def cmd_pending(args: argparse.Namespace) -> int:
    """导出待审核清单（供独立审核者）。"""
    store = AnnotationStore.load(Path(args.annotations))
    pending = [
        item
        for item in store
        if item.annotation_status
        in (AnnotationStatus.LABELED, AnnotationStatus.PENDING_REVIEW)
    ]
    payload = [item.to_dict() for item in pending]
    print(json.dumps({"pending": len(payload), "items": payload}, ensure_ascii=False, indent=2))
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """审核入口（唯一能产生 verified 的路径；强制独立审核）。"""
    path = Path(args.annotations)
    store = AnnotationStore.load(path)
    target = next((item for item in store if item.annotation_id == args.id), None)
    if target is None:
        raise AnnotationError(f"未找到标注: {args.id}")
    if not args.annotator or not args.reviewer:
        raise AnnotationError("promote 必须同时提供 --annotator 与 --reviewer")
    target.annotator = str(args.annotator)
    target.reviewer = str(args.reviewer)
    if target.is_self_reviewed:
        raise AnnotationError("标注人与审核人不得相同（独立审核）")
    if args.approve:
        target.review_status = ReviewStatus.APPROVED
        target.annotation_status = AnnotationStatus.VERIFIED
    elif args.reject:
        target.review_status = ReviewStatus.REJECTED
        target.annotation_status = AnnotationStatus.REJECTED
    else:
        target.review_status = ReviewStatus.NEEDS_CHANGES
        target.annotation_status = AnnotationStatus.PENDING_REVIEW
    target.updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    _atomic_write(path, store.to_jsonl())
    print(json.dumps(target.to_dict(), ensure_ascii=False))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """渲染 bbox 预览图到 ``--out``（**不修改原图**，仅供人工核对）。"""
    import cv2
    import numpy as np

    store = AnnotationStore.load(Path(args.annotations))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        ElementRegionKind.STATE_EVIDENCE.value: (0, 200, 255),
        ElementRegionKind.UI_ELEMENT.value: (80, 220, 80),
        ElementRegionKind.CLICKABLE_VERIFIED.value: (60, 60, 255),
    }
    rendered = 0
    for relative in dict.fromkeys(item.image_path for item in store):
        source = DATASET / relative
        if not source.is_file():
            continue
        buffer = np.fromfile(str(source), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            continue
        for item in store:
            if item.image_path != relative:
                continue
            rect = item.bbox
            top_left = (int(rect.x), int(rect.y))
            bottom_right = (int(rect.right), int(rect.bottom))
            color = colors.get(item.region_kind.value, (255, 255, 255))
            cv2.rectangle(image, top_left, bottom_right, color, 3)
            cv2.putText(
                image,
                f"{item.element_name} [{item.region_kind.value}]",
                (int(rect.x), max(20, int(rect.y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
        target = out_dir / (source.stem + "_preview.jpg")
        scale = min(1.0, 1400.0 / max(1, image.shape[1]))
        resized = cv2.resize(
            image,
            (int(image.shape[1] * scale), int(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            target.write_bytes(encoded.tobytes())
            rendered += 1
    print(json.dumps({"rendered": rendered, "out": str(out_dir)}, ensure_ascii=False))
    return 0


def cmd_verify_sources(args: argparse.Namespace) -> int:
    """为所有被标注的原图生成/刷新来源登记（历史素材：session 留空）。"""
    from src.vision.dataset.sources import CaptureSource, SourceRegistry

    store = AnnotationStore.load(Path(args.annotations))
    registry = SourceRegistry.load(Path(args.sources))
    added = 0
    for relative in dict.fromkeys(item.image_path for item in store):
        source_path = DATASET / relative
        if not source_path.is_file():
            continue
        digest = compute_sha256(source_path)
        if registry.by_sha256(digest) is not None:
            continue
        registry.add(
            CaptureSource.legacy(relative_path=relative, digest=digest)
        )
        added += 1
    if added:
        registry.save(Path(args.sources))
    print(json.dumps({**registry.stats(), "added": added}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    global DATASET
    parser = argparse.ArgumentParser(description="元素级标注工具（只写 annotations/，不碰 raw/）")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES))
    parser.add_argument(
        "--dataset",
        default=str(DATASET),
        help="数据根目录（默认 dataset/；测试或复刻场景可指向别处）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="新增候选草稿（默认 pending_review）")
    add.add_argument("--image", required=True, help="相对 dataset/ 的原图路径")
    add.add_argument("--element", required=True)
    add.add_argument("--bbox", required=True, help="x,y,w,h（原始帧客户区像素）")
    add.add_argument("--region", default=ElementRegionKind.UI_ELEMENT.value)
    add.add_argument("--visibility", default=ElementVisibility.VISIBLE.value)
    add.add_argument("--status", default=AnnotationStatus.PENDING_REVIEW.value)
    add.add_argument("--annotator", default="agent-draft")
    add.add_argument("--note", default="")
    add.set_defaults(func=cmd_add)

    listing = sub.add_parser("list", help="列出标注与统计")
    listing.set_defaults(func=cmd_list)

    pending = sub.add_parser("pending", help="导出待审核清单")
    pending.set_defaults(func=cmd_pending)

    promote = sub.add_parser("promote", help="审核（唯一能产生 verified 的入口）")
    promote.add_argument("--id", required=True)
    promote.add_argument("--annotator", default="")
    promote.add_argument("--reviewer", default="")
    promote.add_argument("--approve", action="store_true")
    promote.add_argument("--reject", action="store_true")
    promote.set_defaults(func=cmd_promote)

    render = sub.add_parser("render", help="渲染 bbox 预览图（不修改原图）")
    render.add_argument("--out", required=True)
    render.set_defaults(func=cmd_render)

    sources = sub.add_parser("verify-sources", help="刷新来源登记（历史素材 session 留空）")
    sources.set_defaults(func=cmd_verify_sources)

    args = parser.parse_args(argv)
    DATASET = Path(args.dataset)
    try:
        return int(args.func(args))
    except AnnotationError as exc:
        print(f"[annotate] 拒绝: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
