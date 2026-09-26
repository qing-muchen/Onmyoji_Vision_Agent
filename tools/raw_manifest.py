"""raw 真源清单工具（Phase 11.1）：生成 / 校验 ``dataset/raw/`` 的 SHA256 清单。

用途：**保证原始图片字节不变**。

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


def collect_raw_entries(dataset: Path = DATASET) -> list[dict[str, object]]:
    """收集 ``raw/`` 下全部图片的清单条目（相对路径 / sha256 / 尺寸 / 字节数）。

    Args:
        dataset: 数据根目录。

    Returns:
        条目列表（按相对路径排序）。
    """
    raw_root = dataset / "raw"
    entries: list[dict[str, object]] = []
    if not raw_root.is_dir():
        return entries
    for path in sorted(raw_root.rglob("*")):
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


def write_manifest(path: Path = DEFAULT_MANIFEST, *, dataset: Path = DATASET) -> dict[str, object]:
    """生成并写入清单（**只写清单文件**，不触碰任何图片）。

    Args:
        path: 清单路径。
        dataset: 数据根目录。

    Returns:
        清单内容。
    """
    entries = collect_raw_entries(dataset)
    payload = {
        "kind": "raw-sha256",
        "phase": "11.1",
        "dataset_root": dataset.name,
        "count": len(entries),
        "entries": entries,
        "note": "原始图片为唯一真源；任何条目变化都说明 raw/ 被改动或补齐",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def verify_manifest(path: Path = DEFAULT_MANIFEST, *, dataset: Path = DATASET) -> dict[str, object]:
    """校验 ``raw/`` 与清单一致（只读）。

    Args:
        path: 清单路径。
        dataset: 数据根目录。

    Returns:
        报告字典（``ok`` / ``missing`` / ``changed`` / ``added`` / ``size_mismatch``）。
    """
    if not path.is_file():
        return {
            "ok": False,
            "reason": f"清单不存在: {path}（先运行 --write 生成）",
            "missing": [],
            "changed": [],
            "added": [],
            "size_mismatch": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(item["path"]): item for item in payload.get("entries", [])}
    actual = {str(item["path"]): item for item in collect_raw_entries(dataset)}

    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed: list[str] = []
    size_mismatch: list[str] = []
    for key in sorted(set(expected) & set(actual)):
        if expected[key]["sha256"] != actual[key]["sha256"]:
            changed.append(key)
        if int(expected[key].get("bytes", -1)) != int(actual[key].get("bytes", -1)):
            size_mismatch.append(key)
    ok = not (missing or added or changed)
    return {
        "ok": ok,
        "manifest": str(path.relative_to(ROOT)) if ROOT in path.parents else str(path),
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "added": added,
        "changed": changed,
        "size_mismatch": size_mismatch,
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
        print(json.dumps({"written": str(target), "count": payload["count"]}, ensure_ascii=False))
        return 0

    report = verify_manifest(target)
    if args.quiet:
        print("raw manifest: PASS" if report["ok"] else "raw manifest: FAIL")
        return 0 if report["ok"] else 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
