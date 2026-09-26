"""离线元素定位评估（Phase 11.2）：``tools/evaluate_localization.py``。

```bash
# ① 用**现有 Vision 检测输出**为被标注原图生成预测，再评估（默认只吃 verified 真值）
python tools/evaluate_localization.py --generate

# ② 只做标注侧统计（不跑检测 / 或没有预测文件）
python tools/evaluate_localization.py

# ③ 指定预测文件与真值口径
python tools/evaluate_localization.py --predictions preds.jsonl --status all --json
```

**核心约束（用户要求）**：

* **状态分类正确** 与 **元素定位正确** **分开统计**，绝不合并成一个"准确率"；
* 预测框（模板命中框）与真值框（元素完整外框）可能**语义不同** ⇒ 每条对比都记录
  ``semantics_match``；语义不一致的样本**不计入**定位指标，只计入"语义待对齐"计数；
* 真值不足（``verified`` 数为 0）⇒ 输出 ``insufficient_data``，**不给任何定位指标**；
* 绝不使用"在训练数据上自比"的结果宣称泛化性能；
* 只读：不修改 ``dataset/raw/`` 与标注文件。

指标：``IoU``、中心点误差（像素）、命中率（``IoU ≥ --iou-threshold``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.core.geometry import Rect  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.dataset.annotations import (  # noqa: E402
    AnnotationStatus,
    AnnotationStore,
    ElementRegionKind,
)

DATASET = ROOT / "dataset"
DEFAULT_ANNOTATIONS = DATASET / "annotations" / "elements.jsonl"
DEFAULT_PREDICTIONS = DATASET / "annotations" / "predictions.jsonl"

#: 不可比原因（Phase 11.3）：这些情况**不得**被算成定位正确或错误，只报告原因
INCOMPARABLE_REASONS: tuple[str, ...] = (
    "incomparable_region_kind",     # 预测框与真值框语义不同（如 template_hit vs element_bbox）
    "unverified_ground_truth",      # 真值尚未经独立审核
    "element_name_mismatch",        # 元素名不一致
    "coordinate_space_mismatch",    # 坐标空间不一致
    "ambiguous_instance",           # 同画面多个同名元素，无法判定归属
    "missing_prediction",           # 没有对应预测
)

#: 预测框的语义标签（模板命中框 / OCR 文字框 —— 与"元素完整外框"不同）
PREDICTION_SEMANTICS: dict[str, str] = {
    "template_hit": "模板命中框（ROI 内匹配区域）",
    "ocr_text_box": "OCR 文字框",
    "element_bbox": "元素完整外框（人工标注口径）",
}


def read_image(path: Path) -> np.ndarray | None:
    """读图（兼容非 ASCII 路径）。"""
    import cv2

    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def intersection_over_union(first: Rect, second: Rect) -> float:
    """计算两个矩形的 IoU（并集为 0 时返回 0.0）。"""
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    inter = width * height
    union = first.width * first.height + second.width * second.height - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def center_offset(first: Rect, second: Rect) -> float:
    """中心点欧氏距离（像素）。"""
    dx = first.center.x - second.center.x
    dy = first.center.y - second.center.y
    return float((dx * dx + dy * dy) ** 0.5)


def generate_predictions(
    *,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    out_path: Path = DEFAULT_PREDICTIONS,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """用现有 Vision 检测输出生成预测（只跑被标注的原图；dry-run，无输入）。

    Args:
        annotations_path: 标注文件（决定要跑哪些原图）。
        out_path: 预测输出（JSONL）。
        min_confidence: 置信度下限。

    Returns:
        报告字典（``predictions`` / ``images`` / ``error``）。
    """
    from src.core.config import load_config
    from src.vision.pipeline.frame_processor import EvidenceFrameProcessor

    store = AnnotationStore.load(annotations_path)
    images = list(dict.fromkeys(item.image_path for item in store))
    if not images:
        return {"predictions": 0, "images": 0, "error": "没有可预测的原图（标注为空）"}

    try:
        processor = EvidenceFrameProcessor.from_config(load_config())
    except Exception as exc:  # noqa: BLE001 - 配置 / 模板缺失时不阻塞评估
        return {"predictions": 0, "images": len(images), "error": f"processor 构建失败: {exc}"}

    records: list[dict[str, Any]] = []
    for index, relative in enumerate(images, start=1):
        image = read_image(DATASET / relative)
        if image is None:
            continue
        height, width = image.shape[:2]
        frame = Frame.create(image, source="dataset-pilot", timestamp=float(index))
        try:
            evidences = processor.detect(frame)
        except Exception as exc:  # noqa: BLE001 - 单图失败不影响其它
            records.append({"image_path": relative, "error": str(exc)})
            continue
        for evidence in evidences:
            if not getattr(evidence, "detected", False) or evidence.bbox is None:
                continue
            if float(evidence.confidence) < min_confidence:
                continue
            bbox = evidence.bbox
            records.append(
                {
                    "image_path": relative,
                    "element_name": str(evidence.name),
                    "bbox": [float(bbox.x), float(bbox.y), float(bbox.width), float(bbox.height)],
                    "confidence": float(evidence.confidence),
                    "source": str(getattr(evidence.source, "value", evidence.source)),
                    "prediction_semantics": "template_hit",
                    "image_width": width,
                    "image_height": height,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    return {"predictions": len(records), "images": len(images), "out": str(out_path)}


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """读取预测 JSONL（缺失文件返回空列表）。"""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("bbox"):
            records.append(payload)
    return records


def evaluate(
    *,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    include_all_status: bool = False,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """评估元素定位（真值 vs 预测），状态分类与定位**分开**。

    Args:
        annotations_path: 标注文件。
        predictions_path: 预测文件。
        include_all_status: ``True`` 时把草稿也当作真值（**仅用于自检，不可用于结论**）。
        iou_threshold: 视为命中的 IoU 阈值。

    Returns:
        报告字典（``status`` / ``insufficient_reason`` / ``localization`` /
        ``semantics_mismatch`` / ``annotation_stats``）。
    """
    store = AnnotationStore.load(annotations_path)
    truth = [
        item
        for item in store
        if (include_all_status or item.is_verified_ground_truth)
    ]
    predictions = load_predictions(predictions_path)
    stats = store.stats()

    report: dict[str, Any] = {
        "annotations_path": str(annotations_path),
        "predictions_path": str(predictions_path),
        "truth_mode": "all(status)" if include_all_status else "verified_ground_truth",
        "annotation_stats": stats,
        "truth_count": len(truth),
        "prediction_count": len(predictions),
        "iou_threshold": iou_threshold,
        "status": "ok",
        "insufficient_reason": "",
        "localization": {},
        "semantics_mismatch": {},
        "notes": [],
    }

    if not truth:
        report["status"] = "insufficient_data"
        report["insufficient_reason"] = (
            "没有任何**经独立审核**的元素级真值"
            f"（annotations={stats['annotations']}, verified={stats['verified']}）"
            " ⇒ 不输出任何定位指标"
        )
        report["notes"].append(
            "当前只有待审核草稿（pending_review）：草稿可用于核对流程，"
            "不能作为定位准确率的依据"
        )
        return report

    # 按 (原图, 元素) 建立预测索引：同一 key 取置信度最高者
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in predictions:
        key = (str(item.get("image_path", "")), str(item.get("element_name", "")))
        current = index.get(key)
        if current is None or float(item.get("confidence", 0.0)) > float(
            current.get("confidence", 0.0)
        ):
            index[key] = item

    per_truth: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    matched = missed = semantics_mismatch = 0
    ious: list[float] = []
    offsets: list[float] = []

    for item in truth:
        key = (item.image_path, item.element_name)
        predicted = index.get(key)
        entry: dict[str, Any] = {
            "annotation_id": item.annotation_id,
            "image_path": item.image_path,
            "element_name": item.element_name,
            "truth_bbox": [item.bbox.x, item.bbox.y, item.bbox.width, item.bbox.height],
            "truth_region_kind": item.region_kind.value,
            "found": predicted is not None,
        }
        if predicted is None:
            missed += 1
            entry["incomparable_reason"] = "missing_prediction"
            per_truth.append(entry)
            continue
        entry["prediction_semantics"] = str(predicted.get("prediction_semantics", ""))
        entry["prediction_confidence"] = float(predicted.get("confidence", 0.0))

        # ---- Phase 11.3：六项可比性前置检查（缺一即不计入定位指标）----
        reasons: list[str] = []
        if not include_all_status and not item.is_verified_ground_truth:
            reasons.append("unverified_ground_truth")
        if str(predicted.get("element_name", "")) != item.element_name:
            reasons.append("element_name_mismatch")
        if str(predicted.get("image_path", "")) != item.image_path:
            reasons.append("coordinate_space_mismatch")
        prediction_space = str(predicted.get("coordinate_space", "frame_client_pixels"))
        if prediction_space != item.coordinate_space.value:
            reasons.append("coordinate_space_mismatch")
        if item.instance_index and predicted.get("instance_index") in (None, ""):
            reasons.append("ambiguous_instance")
        # 语义可比性：只有"元素完整外框"口径才与真值同义
        semantics_match = entry["prediction_semantics"] == "element_bbox"
        entry["semantics_match"] = bool(semantics_match)
        if not semantics_match:
            reasons.append("incomparable_region_kind")
        try:
            x, y, width, height = (float(value) for value in predicted["bbox"])
            predicted_rect = Rect(x=x, y=y, width=width, height=height)
        except Exception:  # noqa: BLE001 - 预测非法 ⇒ 记为不可比
            entry["error"] = "预测 bbox 非法"
            entry["incomparable_reason"] = "incomparable_region_kind"
            missed += 1
            per_truth.append(entry)
            continue
        if reasons:
            entry["incomparable_reason"] = sorted(set(reasons))[0]
            entry["incomparable_reasons"] = sorted(set(reasons))
            for reason in set(reasons):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            per_truth.append(entry)
            continue
        iou = intersection_over_union(item.bbox, predicted_rect)
        offset = center_offset(item.bbox, predicted_rect)
        entry["predicted_bbox"] = [predicted_rect.x, predicted_rect.y, predicted_rect.width, predicted_rect.height]
        entry["iou"] = round(iou, 4)
        entry["center_offset_px"] = round(offset, 2)
        ious.append(iou)
        offsets.append(offset)
        if iou >= iou_threshold:
            matched += 1
        per_truth.append(entry)

    comparable = len(ious)
    report["localization"] = {
        "truth_total": len(truth),
        "found": len(truth) - missed,
        "missed": missed,
        "comparable": comparable,
        "hit_at_iou": matched,
        "hit_rate": round(matched / comparable, 4) if comparable else None,
        "iou_mean": round(float(np.mean(ious)), 4) if ious else None,
        "iou_p50": round(float(np.median(ious)), 4) if ious else None,
        "center_offset_px_mean": round(float(np.mean(offsets)), 2) if offsets else None,
        "center_offset_px_p95": round(float(np.percentile(offsets, 95)), 2) if offsets else None,
    }
    report["semantics_mismatch"] = {
        "count": reason_counts.get("incomparable_region_kind", 0),
        "comparable_threshold": '同义口径 = prediction_semantics == "element_bbox"',
    }
    report["incomparable_reasons"] = dict(sorted(reason_counts.items()))
    report["draft_diagnostic"] = bool(include_all_status)
    if include_all_status:
        report["notes"].append(
            "本报告使用 --status all（把待审核草稿也当真值）⇒ 只能作为 "
            "**draft_diagnostic**，严禁混入正式评估指标或对外宣称准确率"
        )
    report["per_truth"] = per_truth
    report["notes"].append(
        "状态分类与元素定位分开统计：本报告**只**给出元素定位指标（IoU / 中心误差）；"
        "状态分类结果见既有 Runtime 报告"
    )
    if comparable == 0:
        report["status"] = "insufficient_data"
        report["insufficient_reason"] = (
            "真值存在，但**没有**与真值同义口径的预测（全部为模板命中框）"
            " ⇒ 无法计算定位指标（避免跨语义比较）"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="离线元素定位评估（只读；状态与定位分开统计）")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--generate", action="store_true", help="先用现有 Vision 检测生成预测")
    parser.add_argument(
        "--status",
        choices=("verified", "all"),
        default="verified",
        help="真值口径：verified（默认）或 all（把草稿也当真值，仅用于自检）",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = parser.parse_args(argv)

    if args.generate:
        summary = generate_predictions(
            annotations_path=Path(args.annotations), out_path=Path(args.predictions)
        )
        print("[generate]", json.dumps(summary, ensure_ascii=False))

    report = evaluate(
        annotations_path=Path(args.annotations),
        predictions_path=Path(args.predictions),
        include_all_status=args.status == "all",
        iou_threshold=args.iou_threshold,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        stats = report["annotation_stats"]
        print(f"标注: {stats['annotations']} 条（verified={stats['verified']}, "
              f"clickable={stats['clickable_verified']}, 独立原图={stats['sources']}）")
        print(f"真值口径: {report['truth_mode']} | 真值数: {report['truth_count']} | "
              f"预测数: {report['prediction_count']}")
        if report["status"] != "ok":
            print(f"!! {report['status']}: {report['insufficient_reason']}")
        else:
            print("定位指标:", json.dumps(report["localization"], ensure_ascii=False))
            print("语义待对齐:", json.dumps(report["semantics_mismatch"], ensure_ascii=False))
        for note in report["notes"]:
            print(f"  [note] {note}")
        print("Localization: " + ("PASS" if report["status"] == "ok" else "INSUFFICIENT_DATA"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
