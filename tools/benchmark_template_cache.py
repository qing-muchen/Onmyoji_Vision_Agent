"""模板缓存 benchmark（Phase 11.6-A）：``tools/benchmark_template_cache.py``。

只测**缓存本身**的收益，不修改任何配置、不增加模板、不改匹配算法：

```bash
python tools/benchmark_template_cache.py                 # 默认场景 + cache_size=8
python tools/benchmark_template_cache.py --cache-size 16
python tools/benchmark_template_cache.py --repeats 30 --json
```

对照场景（同一批真实 yard 原图，同一 detector 配置，只切换 ``cache_size``）：

========================== ==========================================================
场景                        含义
========================== ==========================================================
``cold_single``             单帧首次检测（缓存没有收益的下界）
``repeat_same_frame``       同一画面连续重复（庭院站桩 / 静止界面）
``distinct_frames``         连续不同画面（缓存无法命中的上界成本）
``tiny_change_in_roi``      同一画面、ROI **内**改动 1 个像素（保守语义 ⇒ 不命中）
``change_outside_roi``      同一画面、ROI **外**改动 1 个像素（ROI 缓存仍命中 ⇒ 帧级缓存做不到）
========================== ==========================================================

输出：每场景的调用数 / 平均与中位耗时 / 命中率，以及"整链耗时 ≈ 单项 + 其它 detector 基线"
（基线取 Phase 11.5-B 稳健基准的 282ms）的推算值。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.config import load_config  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "ova116" / "template_cache_benchmark.json"
YARD_DIR = "raw/yard"
YARD_EVIDENCE = "bottom_menu_yard"
#: 其它 detector + 状态判定的基线耗时（Phase 11.5-B 稳健基准实测，用于推算整链）
CHAIN_BASE_MS = 282.0
SCENARIOS: tuple[str, ...] = (
    "cold_single",
    "repeat_same_frame",
    "distinct_frames",
    "tiny_change_in_roi",
    "change_outside_roi",
)


def read_image(path: Path) -> np.ndarray | None:
    """读图（兼容非 ASCII 路径）。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def list_yard_images(dataset: Path) -> list[Path]:
    """``raw/yard`` 下的原图（排序）。"""
    directory = dataset / YARD_DIR
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.png"))


def tweak_inside_roi(image: np.ndarray, roi: tuple[float, float, float, float], index: int) -> np.ndarray:
    """复制一份图像，并在 **ROI 内**改动 1 个像素。"""
    changed = image.copy()
    height, width = image.shape[:2]
    left, top = int(roi[0] * width), int(roi[1] * height)
    right, bottom = int(roi[2] * width), int(roi[3] * height)
    changed[min(bottom - 1, top + index % max(1, bottom - top)), left] = [0, 0, 0]
    return changed


def tweak_outside_roi(
    image: np.ndarray, roi: tuple[float, float, float, float], index: int
) -> np.ndarray:
    """复制一份图像，并在 **ROI 之外**（顶部区域）改动 1 个像素。"""
    changed = image.copy()
    height, width = image.shape[:2]
    top_limit = max(1, int(roi[1] * height) - 1)
    changed[index % top_limit, width // 2] = [0, 0, 0]
    return changed


def scenario_images(
    scenario: str,
    *,
    base: np.ndarray,
    others: list[np.ndarray],
    roi: tuple[float, float, float, float],
    repeats: int,
) -> list[np.ndarray]:
    """按场景生成待检测的图像序列。"""
    if scenario == "cold_single":
        return [base]
    if scenario == "repeat_same_frame":
        return [base] * repeats
    if scenario == "distinct_frames":
        pool = others if others else [base]
        return [pool[index % len(pool)] for index in range(repeats)]
    if scenario == "tiny_change_in_roi":
        return [tweak_inside_roi(base, roi, index) for index in range(repeats)]
    if scenario == "change_outside_roi":
        return [tweak_outside_roi(base, roi, index) for index in range(repeats)]
    raise ValueError(f"未知场景: {scenario}")


def run_scenario(
    detector: BottomMenuYardDetector,
    images: list[np.ndarray],
    *,
    repeats: int,
) -> dict[str, Any]:
    """跑一个场景：逐张构造帧（含拷贝）后调用 detector，记录耗时与缓存统计。"""
    detector.clear_cache()
    detector.cache.reset_statistics()
    detect_ms: list[float] = []
    frame_ms: list[float] = []
    for index, image in enumerate(images, start=1):
        started = time.perf_counter()
        frame = Frame.create(image, source="benchmark", timestamp=float(index))
        frame_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        detector.detect(frame)
        detect_ms.append((time.perf_counter() - started) * 1000.0)
    stats = detector.cache_stats()
    detect_median = float(np.median(detect_ms))
    return {
        "calls": len(images),
        "detect_ms_mean": round(float(np.mean(detect_ms)), 2),
        "detect_ms_p50": round(detect_median, 2),
        "detect_ms_total": round(float(np.sum(detect_ms)), 2),
        "frame_ms_mean": round(float(np.mean(frame_ms)), 2) if frame_ms else None,
        "cache": stats,
        "predicted_chain_ms": round(CHAIN_BASE_MS + detect_median, 2),
    }


def benchmark(
    *,
    dataset: Path = DATASET,
    cache_size: int = 8,
    repeats: int = 20,
    evidence: str = YARD_EVIDENCE,
) -> dict[str, Any]:
    """对每个场景分别测"无缓存"与"有缓存"。

    Args:
        dataset: 数据根目录。
        cache_size: 启用缓存时的容量（``<=0`` 时只跑无缓存一遍）。
        repeats: 每个场景的调用次数。
        evidence: 证据名（默认 ``bottom_menu_yard``）。

    Returns:
        报告字典（``scenarios`` / ``cache_size`` / ``repeats`` / ``roi`` / ``note``）。
    """
    config = load_config()
    detector_config = config.evidence.detectors[evidence]
    roi = tuple(float(value) for value in detector_config.roi)
    paths = list_yard_images(dataset)
    if not paths:
        return {"error": f"没有可用的 yard 原图: {dataset / YARD_DIR}"}

    base = read_image(paths[0])
    if base is None:
        return {"error": f"原图不可读: {paths[0]}"}
    others = [image for image in (read_image(path) for path in paths[1:]) if image is not None]

    settings = [0, max(0, int(cache_size))] if cache_size > 0 else [0]
    report: dict[str, Any] = {
        "evidence": evidence,
        "roi": list(roi),
        "threshold": float(detector_config.threshold),
        "templates_loaded": len(
            BottomMenuYardDetector.from_config(config.evidence).templates
        ),
        "repeats": repeats,
        "cache_settings": settings,
        "chain_base_ms": CHAIN_BASE_MS,
        "scenarios": {},
        "note": (
            "只切换 detector 的 cache_size（内存构造，不改配置）；匹配算法与阈值完全不变。"
            "每轮都新建 Frame（含一次图像拷贝），贴近运行期真实开销。"
        ),
    }

    for setting in settings:
        key = f"cache_{setting}"
        report["scenarios"][key] = {}
        for scenario in SCENARIOS:
            images = scenario_images(
                scenario, base=base, others=others, roi=roi, repeats=repeats
            )
            detector = BottomMenuYardDetector.from_config(config.evidence, cache_size=setting)
            report["scenarios"][key][scenario] = run_scenario(
                detector, images, repeats=repeats
            )
    return report


def _print_report(report: dict[str, Any]) -> None:
    """打印可读对照表。"""
    if "error" in report:
        print(f"!! {report['error']}")
        return
    print("=" * 92)
    print(
        f"模板缓存 benchmark（证据={report['evidence']}，模板={report['templates_loaded']} 张，"
        f"阈值={report['threshold']}，ROI={report['roi']}，每场景调用 {report['repeats']} 次）"
    )
    print("-" * 92)
    print("  场景                    缓存  调用  平均ms   中位ms   命中/未命中  命中率   推算整链ms")
    for key, scenarios in report["scenarios"].items():
        setting = key.removeprefix("cache_")
        for name, row in scenarios.items():
            cache = row["cache"]
            print(
                f"  {name:<24} {setting:>4} {row['calls']:>5} {row['detect_ms_mean']:>8}"
                f" {row['detect_ms_p50']:>8}   {cache['hits']:>4}/{cache['misses']:<5}"
                f" {cache['hit_rate']:>7}  {row['predicted_chain_ms']:>9}"
            )
    print("-" * 92)
    print(f"  [基线] 其它 detector + 状态判定 ≈ {report['chain_base_ms']}ms（Phase 11.5-B 稳健基准）")
    print(f"  [说明] {report['note']}")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="模板缓存 benchmark（只读：不改配置 / 不加模板 / 不改匹配算法）"
    )
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--cache-size", type=int, default=8, help="启用缓存时的容量")
    parser.add_argument("--repeats", type=int, default=20, help="每个场景的调用次数")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON 输出路径")
    parser.add_argument("--json", action="store_true", help="只打印完整 JSON")
    args = parser.parse_args(argv)

    report = benchmark(
        dataset=Path(args.dataset), cache_size=args.cache_size, repeats=args.repeats
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
        print("=" * 92)
        print(f"明细 JSON: {out}")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
