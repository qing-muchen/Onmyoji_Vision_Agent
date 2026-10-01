"""yard 皮肤模板派生（Phase 11.5-B）：``tools/build_yard_skin_templates.py``。

把 ``dataset/raw/yard`` 里**已经存在**的 ``skin001…skin018`` 原图，按现有
``bottom_menu_yard`` 的 ROI 规则派生成候选模板，写入**派生目录**
``dataset/features/bottom_menu/yard/positive/``。

```bash
# ① 只预览计划（不写任何文件）
python tools/build_yard_skin_templates.py --dry-run

# ② 正式生成（幂等：已存在的同名模板不会被覆盖）
python tools/build_yard_skin_templates.py

# ③ 生成并把来源清单一并写入派生目录（便于长期追溯）
python tools/build_yard_skin_templates.py --manifest dataset/features/bottom_menu/yard/positive_sources.json
```

**边界（用户要求）**：

* **不修改** ``dataset/raw``（只读取原图）；
* **不修改**任何 annotation 与 ``dataset/annotations``；
* 派生模板**不计入**独立样本（模板是 ROI 派生图，独立样本仍只有 98 张原图）；
* **不降低** ``threshold``、**不引入**新 detector —— 本工具只产生模板文件；
* 生成**必须**落在派生目录：目标目录不在 ``dataset/features/`` 下时直接拒绝；
* 幂等：同名模板已存在时**跳过**（``--force`` 才覆盖），且从不删除既有 cap 模板。

**去重与代表性**：每个皮肤**组内**按 dHash（汉明距离 ≤ ``--dedupe-threshold``）去重，
只保留代表帧 ⇒ 既去掉同一场景的连拍冗余，又保证**每种皮肤至少 1 张**候选模板。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.config import load_config  # noqa: E402
from src.vision.dataset.sources import compute_sha256, image_size  # noqa: E402
from src.vision.detector.models import TemplateInfo  # noqa: E402
from src.vision.evidence.templates import (  # noqa: E402
    EVIDENCE_TEMPLATE_TYPE,
    dedupe_templates,
    load_feature_templates,
)

DATASET = ROOT / "dataset"
YARD_EVIDENCE = "bottom_menu_yard"
YARD_DIR = "raw/yard"
#: 允许写入的派生根目录（越界直接拒绝）
DERIVED_ROOT = "features"
DEFAULT_DEDUPE_THRESHOLD = 6
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
#: ``yard_skin001_002`` → 来源前缀 + 序号
SOURCE_PATTERN = re.compile(r"^yard_(?P<source>skin\d{3})_(?P<num>\d+)$")


# ---------------------------------------------------------------------------
# 只读辅助
# ---------------------------------------------------------------------------


def read_image(path: Path) -> np.ndarray | None:
    """读图（兼容非 ASCII 路径）。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def parse_skin(path: Path) -> tuple[str, str] | None:
    """解析皮肤来源（非 ``skinNNN`` 命名返回 ``None``）。"""
    matched = SOURCE_PATTERN.match(path.stem)
    if matched is None:
        return None
    return matched.group("source"), matched.group("num")


def crop_roi(image: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    """按 detector 的 ROI 规则裁剪（与运行期 ``crop_roi`` 同一像素算法）。"""
    height, width = image.shape[:2]
    left, top = int(roi[0] * width), int(roi[1] * height)
    right, bottom = int(roi[2] * width), int(roi[3] * height)
    return image[top:bottom, left:right]


# ---------------------------------------------------------------------------
# 计划：候选模板
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateTemplate:
    """一个候选模板（来自哪张原图、裁到哪里）。

    Attributes:
        source: 皮肤来源（``skin001`` …）。
        number: 原图序号（三位）。
        image_path: 原图相对 ``dataset/`` 的路径。
        source_id: 原图身份（``src_<sha256 前 16>``）。
        template_name: 目标模板文件名（``bottom_menu_yard_skin001_001.png``）。
        crop_size: ROI 裁剪尺寸 ``"宽x高"``。
    """

    source: str
    number: str
    image_path: str
    source_id: str
    template_name: str
    crop_size: str


def plan_candidates(
    dataset: Path, *, dedupe_threshold: int = DEFAULT_DEDUPE_THRESHOLD
) -> list[CandidateTemplate]:
    """规划候选模板：按皮肤分组 → 组内 dHash 去重 → 每组保留代表帧。

    Args:
        dataset: 数据根目录。
        dedupe_threshold: 组内去重的 dHash 汉明距离阈值（``<=0`` = 不去重，
            即每个皮肤的所有帧都进入候选）。

    Returns:
        候选模板列表（按皮肤、序号排序）；**不写任何文件**。
    """
    detector_config = load_config().evidence.detectors[YARD_EVIDENCE]
    roi = tuple(float(value) for value in detector_config.roi)

    grouped: dict[str, list[Path]] = defaultdict(list)
    yard_dir = dataset / YARD_DIR
    if yard_dir.is_dir():
        for path in sorted(yard_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            parsed = parse_skin(path)
            if parsed is not None:
                grouped[parsed[0]].append(path)

    candidates: list[CandidateTemplate] = []
    for source, paths in sorted(grouped.items()):
        representatives: list[Path] = []
        infos: list[TemplateInfo] = []
        for path in paths:
            image = read_image(path)
            if image is None:
                continue
            infos.append(
                TemplateInfo(
                    template_id=path.stem,
                    element_type=EVIDENCE_TEMPLATE_TYPE,
                    element_name=YARD_EVIDENCE,
                    skin=source,
                    scene=parse_skin(path)[1] if parse_skin(path) else "000",
                    category=YARD_EVIDENCE.split("_")[0],
                    path=path,
                    image=image,
                )
            )
        kept = dedupe_templates(infos, threshold=dedupe_threshold)
        representatives = [info.path for info in kept]

        for path in representatives:
            parsed = parse_skin(path)
            assert parsed is not None  # grouped 时已解析
            _source, number = parsed
            image = read_image(path)
            if image is None:
                continue
            crop = crop_roi(image, roi)
            if crop.size == 0:
                continue
            height, width = image_size(path)
            candidates.append(
                CandidateTemplate(
                    source=source,
                    number=number,
                    image_path=f"{YARD_DIR}/{path.name}",
                    source_id=f"src_{compute_sha256(path)[:16]}",
                    template_name=f"{YARD_EVIDENCE}_{source}_{number}.png",
                    crop_size=f"{crop.shape[1]}x{crop.shape[0]}",
                )
            )
    return candidates


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


def _assert_derived_dir(out_dir: Path, dataset: Path) -> None:
    """目标目录必须位于 ``dataset/features/`` 下（派生目录），否则拒绝写入。

    Raises:
        ValueError: 目标目录越界。
    """
    try:
        relative_parts = out_dir.resolve().relative_to(dataset.resolve()).parts
    except ValueError as exc:
        raise ValueError(f"派生目录必须位于 dataset 内: {out_dir}") from exc
    if not relative_parts or relative_parts[0] != DERIVED_ROOT:
        raise ValueError(
            f"派生模板只能写入 dataset/{DERIVED_ROOT}/ 下（当前 {out_dir}）——"
            "禁止写入 raw / annotations 或任何非派生目录"
        )


def build_templates(
    candidates: list[CandidateTemplate],
    *,
    dataset: Path = DATASET,
    out_dir: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """把候选模板写进派生目录（幂等；不删除既有模板）。

    Args:
        candidates: :func:`plan_candidates` 的结果。
        dataset: 数据根目录。
        out_dir: 目标派生目录（缺省 ``dataset/features/bottom_menu/yard/positive``）。
        dry_run: 只报告计划，不写文件。
        force: 覆盖已存在的同名模板（默认**跳过**）。

    Returns:
        报告字典（``candidates`` / ``written`` / ``skipped_existing`` /
        ``sources_covered`` / ``per_source`` / ``templates``）。

    Raises:
        ValueError: 目标目录不在派生目录下，或原图缺失/不可读。
    """
    detector_config = load_config().evidence.detectors[YARD_EVIDENCE]
    roi = tuple(float(value) for value in detector_config.roi)
    target_dir = out_dir or (dataset / "features/bottom_menu/yard/positive")
    _assert_derived_dir(target_dir, dataset)

    written: list[str] = []
    skipped: list[str] = []
    templates: list[dict[str, Any]] = []
    per_source: dict[str, int] = defaultdict(int)

    for candidate in candidates:
        source_path = dataset / candidate.image_path
        target_path = target_dir / candidate.template_name
        entry = {
            "template_name": candidate.template_name,
            "from_image": candidate.image_path,
            "source_id": candidate.source_id,
            "source": candidate.source,
            "number": candidate.number,
            "crop_size": candidate.crop_size,
            "roi": list(roi),
            "target_path": target_path.relative_to(dataset).as_posix(),
        }
        if target_path.is_file() and not force:
            entry["action"] = "skip_existing"
            skipped.append(candidate.template_name)
            templates.append(entry)
            per_source[candidate.source] += 1
            continue
        image = read_image(source_path)
        if image is None:
            raise ValueError(f"原图不可读: {candidate.image_path}")
        crop = crop_roi(image, roi)
        if crop.size == 0:
            raise ValueError(f"ROI 裁剪为空: {candidate.image_path}（ROI={roi}）")
        if dry_run:
            entry["action"] = "would_write"
            templates.append(entry)
            per_source[candidate.source] += 1
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        ok, buffer = cv2.imencode(".png", crop)
        if not ok:
            raise ValueError(f"PNG 编码失败: {candidate.template_name}")
        target_path.write_bytes(buffer.tobytes())
        entry["action"] = "written"
        written.append(candidate.template_name)
        templates.append(entry)
        per_source[candidate.source] += 1

    return {
        "out_dir": out_dir.relative_to(dataset).as_posix() if out_dir else
        "features/bottom_menu/yard/positive",
        "dry_run": dry_run,
        "roi": list(roi),
        "candidates": len(candidates),
        "written": len(written),
        "skipped_existing": len(skipped),
        "sources_covered": sorted({item.source for item in candidates}),
        "source_count": len({item.source for item in candidates}),
        "per_source": dict(sorted(per_source.items())),
        "written_names": written,
        "skipped_names": skipped,
        "templates": templates,
        "note": (
            "派生模板来自 dataset/raw/yard 的 ROI 裁剪，**不计入独立样本**；"
            "本工具不修改 raw、不修改任何 annotation、不删除既有模板"
        ),
    }


def audit_after_build(*, dataset: Path = DATASET, root: Path | None = None) -> dict[str, Any]:
    """生成后清点模板目录（供报告用；**只读**）。"""
    detector_config = load_config().evidence.detectors[YARD_EVIDENCE]
    templates = load_feature_templates(
        detector_config.template_source, YARD_EVIDENCE, limit=0, root=root or ROOT
    )
    per_source: dict[str, int] = defaultdict(int)
    for item in templates:
        per_source[item.skin] += 1
    return {
        "templates_in_directory": len(templates),
        "sources": dict(sorted(per_source.items())),
        "source_count": len(per_source),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="按 bottom_menu_yard 的 ROI 规则派生皮肤模板（只写派生目录；不改 raw / annotations）"
    )
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--dry-run", action="store_true", help="只预览计划，不写文件")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的同名模板（默认跳过）")
    parser.add_argument(
        "--dedupe-threshold",
        type=int,
        default=DEFAULT_DEDUPE_THRESHOLD,
        help=f"皮肤组内 dHash 去重阈值（默认 {DEFAULT_DEDUPE_THRESHOLD}）",
    )
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    parser.add_argument(
        "--manifest",
        default="features/bottom_menu/yard/positive_sources.json",
        help="模板来源映射的输出路径（相对 dataset/ 或绝对路径；空字符串则不写）",
    )
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    try:
        candidates = plan_candidates(dataset, dedupe_threshold=args.dedupe_threshold)
        report = build_templates(
            candidates,
            dataset=dataset,
            dry_run=args.dry_run,
            force=args.force,
        )
        if not args.dry_run:
            report["after"] = audit_after_build(dataset=dataset)
            if args.manifest:
                manifest_path = Path(args.manifest)
                if not manifest_path.is_absolute():
                    manifest_path = dataset / manifest_path
                _assert_derived_dir(manifest_path.parent, dataset)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps(
                        {
                            "generated_from": "dataset/raw/yard",
                            "roi": report["roi"],
                            "dedupe_threshold": args.dedupe_threshold,
                            "candidates": report["candidates"],
                            "written": report["written"],
                            "skipped_existing": report["skipped_existing"],
                            "sources_covered": report["sources_covered"],
                            "source_count": report["source_count"],
                            "per_source": report["per_source"],
                            "templates": [
                                {key: value for key, value in item.items() if key != "roi"}
                                for item in report["templates"]
                            ],
                            "note": (
                                "派生模板**不计入**独立样本（独立样本只有 dataset/raw 的原始截图）；"
                                "本映射用于追溯每个模板来自哪张原图"
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                report["manifest"] = manifest_path.relative_to(dataset).as_posix()
    except ValueError as exc:
        print(f"[build_yard_skin_templates] 拒绝: {exc}")
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"ROI                : {report['roi']}")
        print(f"候选模板           : {report['candidates']}（覆盖皮肤 {report['source_count']} 种）")
        print(f"写入 / 跳过已存在  : {report['written']} / {report['skipped_existing']}"
              f"{'（--dry-run，未写盘）' if report['dry_run'] else ''}")
        print(f"每皮肤候选         : {report['per_source']}")
        if report.get("manifest"):
            print(f"来源映射           : dataset/{report['manifest']}")
        if report.get("after"):
            print(f"模板目录现有       : {report['after']['templates_in_directory']} 张，"
                  f"来源 {report['after']['source_count']} 种 {report['after']['sources']}")
        print("模板来源清单：")
        for item in report["templates"]:
            print(
                f"  {item['template_name']:<40} ← {item['from_image']:<28}"
                f" {item['crop_size']:<12} [{item['action']}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
