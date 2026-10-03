"""raw 真源清单工具（Phase 11.1 / 11.9-A）：生成 / 校验 raw 图片的 SHA256 清单。

用途：**保证原始图片字节不变**，并把"历史 baseline"与"当前版本新采集"**分区**记录。

```bash
# 生成 / 刷新清单（显式写入模式）
python tools/raw_manifest.py --write

# 校验（默认，只读）：确认 raw 内容与清单一致
python tools/raw_manifest.py
python tools/raw_manifest.py --verify dataset/manifests/raw_sha256.json
```

设计要点：

* 默认**只读**（校验）；写清单必须显式 ``--write``；
* 清单路径缺省 ``dataset/manifests/raw_sha256.json``；
* **分区**（Phase 11.9-A）：顶层 ``count`` / ``entries`` 只描述历史 ``raw/``
  （**不含** ``raw/yard_live/``，因此 178 张的语义与旧版逐字段一致）；
  新采集数据单独放在 ``live`` 分区（``root`` / ``count`` / ``sessions`` / ``entries``），
  校验时同样报 ``added`` / ``missing`` / ``changed``，不与历史混在一起；
* 图片读取用 ``np.fromfile`` + ``cv2.imdecode``（兼容非 ASCII 路径）；
* 图片尺寸与 SHA256 都不匹配时分别报告，便于定位"被替换"与"被改动大小"。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.dataset.sources import compute_sha256, image_size  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_MANIFEST = DATASET / "manifests" / "raw_sha256.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
#: 当前版本新采集数据的独立根目录（**不并入**历史 raw 计数）
LIVE_RAW_ROOT = "raw/yard_live"


def _entries_under(root: Path, dataset: Path) -> list[dict[str, object]]:
    """收集某个根目录下全部图片的清单条目（递归，按相对路径排序）。"""
    entries: list[dict[str, object]] = []
    if not root.is_dir():
        return entries
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(dataset).as_posix()
        width, height = image_size(path)
        entries.append(
            {
                "path": relative,
                "sha256": compute_sha256(path),
                "bytes": path.stat().st_size,
                "width": width,
                "height": height,
            }
        )
    return entries


def collect_raw_entries(dataset: Path = DATASET) -> list[dict[str, object]]:
    """收集**历史** ``raw/`` 下全部图片的清单条目（**排除** ``raw/yard_live/``）。

    Args:
        dataset: 数据根目录。

    Returns:
        条目列表（按相对路径排序）。
    """
    raw_root = dataset / "raw"
    live_root = dataset / LIVE_RAW_ROOT
    entries: list[dict[str, object]] = []
    if not raw_root.is_dir():
        return entries
    for path in sorted(raw_root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.is_relative_to(live_root):
            continue
        relative = path.relative_to(dataset).as_posix()
        width, height = image_size(path)
        entries.append(
            {
                "path": relative,
                "sha256": compute_sha256(path),
                "bytes": path.stat().st_size,
                "width": width,
                "height": height,
            }
        )
    return entries


def collect_live_entries(dataset: Path = DATASET) -> list[dict[str, object]]:
    """收集当前版本新采集数据（``raw/yard_live/``）的清单条目。

    Args:
        dataset: 数据根目录。

    Returns:
        条目列表（按相对路径排序；目录不存在时为空）。
    """
    return _entries_under(dataset / LIVE_RAW_ROOT, dataset)


def live_sessions(entries: list[dict[str, object]]) -> dict[str, int]:
    """按会话统计新采集条目数（``raw/yard_live/<session>/...``）。

    Args:
        entries: :func:`collect_live_entries` 的输出。

    Returns:
        ``session_id -> count``。
    """
    counts: dict[str, int] = {}
    for entry in entries:
        parts = str(entry["path"]).split("/")
        session = parts[2] if len(parts) > 3 else "(unparsed)"
        counts[session] = counts.get(session, 0) + 1
    return dict(sorted(counts.items()))


def write_manifest(path: Path = DEFAULT_MANIFEST, *, dataset: Path = DATASET) -> dict[str, object]:
    """生成并写入清单（**只写清单文件**，不触碰任何图片）。

    Args:
        path: 清单路径。
        dataset: 数据根目录。

    Returns:
        清单内容（顶层 = 历史 raw；``live`` = 当前版本新采集）。
    """
    entries = collect_raw_entries(dataset)
    live = collect_live_entries(dataset)
    payload = {
        "kind": "raw-sha256",
        "phase": "11.1",
        "dataset_root": dataset.name,
        "count": len(entries),
        "entries": entries,
        "live": {
            "root": LIVE_RAW_ROOT,
            "count": len(live),
            "sessions": live_sessions(live),
            "entries": live,
            "note": "当前版本庭院新采集（Phase 11.9-A）；不并入历史 178 张计数",
        },
        "note": "原始图片为唯一真源；任何条目变化都说明 raw/ 被改动或补齐",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _compare(
    expected_entries: list[dict[str, object]], actual_entries: list[dict[str, object]]
) -> dict[str, object]:
    """比对两组清单条目（返回 missing / added / changed / size_mismatch）。"""
    expected = {str(item["path"]): item for item in expected_entries}
    actual = {str(item["path"]): item for item in actual_entries}
    changed: list[str] = []
    size_mismatch: list[str] = []
    for key in sorted(set(expected) & set(actual)):
        if expected[key]["sha256"] != actual[key]["sha256"]:
            changed.append(key)
        if int(expected[key].get("bytes", -1)) != int(actual[key].get("bytes", -1)):
            size_mismatch.append(key)
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": sorted(set(expected) - set(actual)),
        "added": sorted(set(actual) - set(expected)),
        "changed": changed,
        "size_mismatch": size_mismatch,
    }


def verify_manifest(path: Path = DEFAULT_MANIFEST, *, dataset: Path = DATASET) -> dict[str, object]:
    """校验 raw 与清单一致（只读），并**分区**报告历史与 ``yard_live``。

    Args:
        path: 清单路径。
        dataset: 数据根目录。

    Returns:
        报告字典（顶层 = 历史 raw；``live`` = 当前版本新采集）。
    """
    empty = {
        "expected_count": 0,
        "actual_count": 0,
        "missing": [],
        "added": [],
        "changed": [],
        "size_mismatch": [],
    }
    if not path.is_file():
        return {
            "ok": False,
            "reason": f"清单不存在: {path}（先运行 --write 生成）",
            **empty,
            "live": dict(empty),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    historical = _compare(payload.get("entries", []), collect_raw_entries(dataset))
    live_expected = payload.get("live", {}).get("entries", [])
    live = _compare(live_expected, collect_live_entries(dataset))
    ok = not (historical["missing"] or historical["added"] or historical["changed"])
    live_ok = not (live["missing"] or live["added"] or live["changed"])
    return {
        "ok": bool(ok and live_ok),
        "manifest": str(path.relative_to(ROOT)) if ROOT in path.parents else str(path),
        "historical_ok": bool(ok),
        "live_ok": bool(live_ok),
        "expected_count": historical["expected_count"],
        "actual_count": historical["actual_count"],
        "missing": historical["missing"],
        "added": historical["added"],
        "changed": historical["changed"],
        "size_mismatch": historical["size_mismatch"],
        "live": {
            "root": payload.get("live", {}).get("root", LIVE_RAW_ROOT),
            **live,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="dataset/raw 真源清单（生成 / 校验）")
    parser.add_argument("manifest", nargs="?", default=str(DEFAULT_MANIFEST), help="清单路径")
    parser.add_argument("--write", action="store_true", help="生成 / 刷新清单（写入模式）")
    parser.add_argument("--quiet", action="store_true", help="只输出结论")
    args = parser.parse_args(argv)

    target = Path(args.manifest)
    if args.write:
        payload = write_manifest(target)
        print(
            json.dumps(
                {
                    "written": str(target),
                    "count": payload["count"],
                    "live_count": payload["live"]["count"],
                    "live_sessions": payload["live"]["sessions"],
                },
                ensure_ascii=False,
            )
        )
        return 0

    report = verify_manifest(target)
    if args.quiet:
        print("raw manifest: PASS" if report["ok"] else "raw manifest: FAIL")
        return 0 if report["ok"] else 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
