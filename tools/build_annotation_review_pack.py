"""审核交接包生成（Phase 11.3）：``tools/build_annotation_review_pack.py``。

```bash
# 为现有全部草稿生成审核包（默认输出到仓库外的临时目录，可用 --out 指定）
python tools/build_annotation_review_pack.py --out "%TEMP%/ova114/review_pack"

# 只做近重复诊断（不写审核包）
python tools/build_annotation_review_pack.py --near-duplicate-only
```

审核包内容（每个原图一份目录）：

```text
review_pack/
├── README.md                    审核须知（含"必须看原图、不要只看裁剪"）
├── review_form.md               审核者填写表（每条标注一行 + 意见栏）
├── manifest.json                全部标注的机器可读清单（含 SHA256）
├── near_duplicates.json         近重复诊断（可疑组 + 依据）
├── index.md                     人工索引（按原图列出问题）
└── images/<原图名>/
    ├── full_preview.jpg         整图叠加标注框（保留完整上下文）
    ├── zoom_<annotation_id>.jpg 该标注的局部放大（含周围留白 ≥ 2× 框宽）
    └── meta_<annotation_id>.json 单条标注元数据 + 待确认问题
```

**边界（用户要求）**：

* 只生成**派生预览**，**从不修改** ``dataset/raw/``；
* **不会**执行 ``promote``（不自动批准任何标注）；
* 近重复诊断只**输出可疑组与依据**（SHA256 相同 / dHash 汉明距离），
  **不删除图片、不合并 ``source_id``、不改写 ``session_id``**。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.vision.dataset.annotations import AnnotationStore  # noqa: E402
from src.vision.dataset.sources import compute_sha256, image_size  # noqa: E402

DATASET = ROOT / "dataset"
DEFAULT_ANNOTATIONS = DATASET / "annotations" / "elements.jsonl"
DEFAULT_OUT = Path(r"C:\Users\Muyu\AppData\Local\Temp\ova114\review_pack")

#: 审核者必须逐条确认的**具体问题**（用户第二节指定）
REVIEW_QUESTIONS: dict[str, tuple[str, ...]] = {
    "bottom_menu_yard": (
        "上边界是否贴合功能栏顶部？（Phase 11.2 曾出现上边界偏高约 70px）",
        "是否应拆分为多个具体入口，而不是整条区域？",
    ),
    "battle_action_bar": (
        "上边界是否贴合操作栏顶部？（Phase 11.2 曾出现上边界偏高约 70px）",
        "与 battle_skill_area 的重叠范围是否合理？",
    ),
    "battle_skill_area": (
        "上边界是否贴合技能栏顶部？（Phase 11.2 曾出现上边界偏高约 70px）",
        "是否应拆成单个技能按钮？",
    ),
    "battle_auto_button": (
        "同画面是否存在**两个**同名按钮（自动 / 手动）？当前标注是否只覆盖其中一个？",
        "实例归属是否正确（instance_index）？",
        "该按钮的**可点击内部区域**是否可从截图判定？（不得仅凭外观设为 clickable_verified）",
    ),
    "battle_shikigami_icons": (
        "visibility=partial 是否准确（最右列是否真被窗口裁切）？",
        "是否需要按头像拆成多个实例？",
    ),
    "chapter_panel": (
        "标注的是**整体面板**还是某个可交互按钮？",
        "面板内的具体按钮是否需要单独登记并标注？",
    ),
    "victory_icon": (
        "确认它只是**状态证据**（不是按钮）？",
        "该页的可交互区域是否只是「整屏点击继续」（无具体元素边界）？",
    ),
    "bottom_menu_exploration": (
        "上边界是否贴合功能栏顶部？",
        "是否应拆分为多个具体入口？",
    ),
}


def read_image(path: Path) -> np.ndarray | None:
    """读图（兼容非 ASCII 路径）。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def dhash(image: np.ndarray, *, size: int = 8) -> str:
    """计算 dHash（差分感知哈希）：返回 16 位十六进制字符串。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = np.packbits(diff.flatten())
    return bits.tobytes().hex()


def hamming(first: str, second: str) -> int:
    """两个十六进制哈希串的汉明距离（按位）。"""
    left = bin(int(first, 16))[2:].zfill(len(first) * 4)
    right = bin(int(second, 16))[2:].zfill(len(second) * 4)
    return sum(1 for a, b in zip(left, right) if a != b)


def near_duplicate_scan(
    *, limit: int = 24, threshold: int = 6, dataset: Path = DATASET
) -> dict[str, object]:
    """近重复诊断（**只读**）：SHA256 完全相同 + dHash 汉明距离过近。

    Args:
        limit: 最多扫描的原图数量（避免一次性读全量）。
        threshold: dHash 汉明距离阈值（≤ 视为可疑近重复）。
        dataset: 数据根目录。

    Returns:
        报告字典（``exact_duplicates`` / ``near_duplicate_pairs`` / ``scanned`` / ``note``）。
    """
    raw_root = dataset / "raw"
    images = sorted(raw_root.rglob("*.png"))[:limit] if raw_root.is_dir() else []
    digests: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for path in images:
        image = read_image(path)
        if image is None:
            continue
        relative = path.relative_to(dataset).as_posix()
        digests[relative] = compute_sha256(path)
        hashes[relative] = dhash(image)

    exact: dict[str, list[str]] = {}
    for relative, digest in digests.items():
        exact.setdefault(digest, []).append(relative)
    exact_duplicates = sorted(
        f"{digest[:12]}: {sorted(paths)}" for digest, paths in exact.items() if len(paths) > 1
    )

    pairs: list[str] = []
    keys = sorted(hashes)
    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            distance = hamming(hashes[first], hashes[second])
            if distance <= threshold:
                pairs.append(f"{distance}: {first} ~ {second}")
    return {
        "scanned": len(hashes),
        "threshold": threshold,
        "exact_duplicates": exact_duplicates,
        "near_duplicate_pairs": sorted(pairs)[:50],
        "note": (
            "近重复仅为**诊断**：不删除图片、不合并 source_id、不改写 session_id；"
            "疑似连续帧 / 高度相似截图应作为**待人工确认**的分组清单"
        ),
    }


def build_pack(
    *,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    out_dir: Path = DEFAULT_OUT,
    verify_sha256: bool = True,
    near_duplicate: bool = True,
    dataset: Path = DATASET,
) -> dict[str, object]:
    """生成审核包（派生预览 + 清单 + 审核表；不修改 raw，不 promote）。"""
    store = AnnotationStore.load(annotations_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    index_lines = ["# 审核索引（按原图）", ""]
    form_lines = ["# 审核表（请逐条填写）", "", "| annotation_id | 判定 | 意见 |", "| --- | --- | --- |"]
    #: 个人复核表（Phase 11.4）：单人开发用。工具**只生成空表**，绝不预填结论。
    self_form_lines = [
        "# 个人复核表（单人开发；Phase 11.4）",
        "",
        "> 供**开发者本人**逐条复核草稿。它**不是**独立审核：",
        "> 逐条判断请由人完成（本表**不预填**任何结论），填完后用",
        "> `annotate.py self-review` 记录 —— 那条记录**不会**产生真值，",
        "> 也不会授予 `clickable_verified` 或真实输入资格。",
        "",
        "| annotation_id | 元素 | 实例 | 个人结论（请手动勾选） | 意见 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for relative in dict.fromkeys(item.image_path for item in store):
        source = dataset / relative
        image = read_image(source)
        if image is None:
            continue
        image_dir = out_dir / "images" / source.stem
        image_dir.mkdir(parents=True, exist_ok=True)
        height, width = image.shape[:2]
        digest = compute_sha256(source) if verify_sha256 else ""

        overlay = image.copy()
        for item in store:
            if item.image_path != relative:
                continue
            rect = item.bbox
            color = {
                "state_evidence": (0, 200, 255),
                "ui_element": (80, 220, 80),
                "clickable_verified": (60, 60, 255),
            }.get(item.region_kind.value, (255, 255, 255))
            cv2.rectangle(
                overlay,
                (int(rect.x), int(rect.y)),
                (int(rect.right), int(rect.bottom)),
                color,
                3,
            )
            label = f"{item.element_name} #{item.instance_index} [{item.region_kind.value}]"
            cv2.putText(
                overlay,
                label,
                (int(rect.x), max(24, int(rect.y) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.resize(
                overlay,
                (int(width * min(1.0, 1500.0 / max(1, width))), int(height * min(1.0, 1500.0 / max(1, width)))),
                interpolation=cv2.INTER_AREA,
            ),
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if ok:
            (image_dir / "full_preview.jpg").write_bytes(encoded.tobytes())

        index_lines.append(f"## {relative}（{width}x{height}, sha256={digest[:12]}）")
        index_lines.append("")
        for item in store:
            if item.image_path != relative:
                continue
            rect = item.bbox
            # 局部放大：含 ≥2× 框宽的上下文，避免"只看裁剪图判断边界"
            pad_x = max(40.0, rect.width)
            pad_y = max(40.0, rect.height)
            left = max(0, int(rect.x - pad_x))
            top = max(0, int(rect.y - pad_y))
            right = min(width, int(rect.right + pad_x))
            bottom = min(height, int(rect.bottom + pad_y))
            crop = image[top:bottom, left:right].copy()
            cv2.rectangle(
                crop,
                (int(rect.x) - left, int(rect.y) - top),
                (int(rect.right) - left, int(rect.bottom) - top),
                (80, 220, 80),
                3,
            )
            ok_crop, encoded_crop = cv2.imencode(
                ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
            )
            if ok_crop:
                (image_dir / f"zoom_{item.element_name}_{item.instance_index}.jpg").write_bytes(
                    encoded_crop.tobytes()
                )
            questions = REVIEW_QUESTIONS.get(item.element_name, ("bbox 是否贴合元素实际边界？",))
            meta = {
                **item.to_dict(),
                "image_sha256": digest,
                "questions": list(questions),
                "must_open_original": "请打开原图（不是只看局部放大）再判断边界",
                "self_review_command": (
                    "python tools/annotate.py self-review --id "
                    f"{item.annotation_id} --decision <approved|rejected|needs_revision> "
                    '--comment "<本人复核意见>" --developer <你的名字>'
                ),
                "self_review_note": (
                    "个人复核**不是**独立审核：不产生真值、不授予 clickable_verified，"
                    "正式的 verified 只能由独立审核者用 promote --approve 产生"
                ),
            }
            (image_dir / f"meta_{item.element_name}_{item.instance_index}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            manifest.append(meta)
            index_lines.append(
                f"- `{item.annotation_id}` **{item.element_name}** "
                f"({item.region_kind.value}, {item.visibility.value}, instance={item.instance_index})"
            )
            for question in questions:
                index_lines.append(f"  - ❓ {question}")
            form_lines.append(
                f"| `{item.annotation_id}` | ☐ 通过 / ☐ 拒绝 / ☐ 需修订 |  |"
            )
            # 个人复核表：**不预填**任何结论，仅给出条目与空栏
            self_form_lines.append(
                f"| `{item.annotation_id}` | {item.element_name} | {item.instance_index} | "
                "☐ 通过 / ☐ 拒绝 / ☐ 需修订 |  |"
            )
        index_lines.append("")

    (out_dir / "manifest.json").write_text(
        json.dumps({"count": len(manifest), "items": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (out_dir / "review_form.md").write_text("\n".join(form_lines) + "\n", encoding="utf-8")
    (out_dir / "self_review_form.md").write_text(
        "\n".join(self_form_lines) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# 元素级标注审核包（Phase 11.3 / 11.4）",
                "",
                "本审核包由 `tools/build_annotation_review_pack.py` 生成（**只生成派生预览，未修改任何原图**）。",
                "",
                "## 审核步骤",
                "1. 打开 `index.md`，逐条阅读待确认问题；",
                "2. 打开 `images/<原图名>/full_preview.jpg` **看整图上下文**；",
                "3. 再打开 `zoom_*.jpg` 查看局部（注意：局部图带 ≥1 倍框宽的留白，避免误判边界）；",
                "4. 在 `review_form.md` 填写判定与意见；",
                "5. 用以下命令落库（**自审会被拒绝**）：",
                "",
                "```bash",
                "python tools/annotate.py review --id <annotation_id> \\",
                "    --reviewer <你的名字> --decision needs_revision --comment \"底部栏上边界再收紧 20px\"",
                "# 仅当你**独立**确认通过时：",
                "python tools/annotate.py promote --id <annotation_id> \\",
                "    --annotator <标注者> --reviewer <你的名字> --approve --comment \"已核对原图\"",
                "```",
                "",
                "## 单人开发的个人复核（Phase 11.4）",
                "若你就是标注者本人（**没有**第二名审核者）：",
                "1. 打开 `self_review_form.md` **逐条**填写个人结论（工具**不会预填**任何结论）；",
                "2. 用 `python tools/annotate.py self-review --id <annotation_id> \\",
                "   --decision approved|rejected|needs_revision --comment \"…\"` 落库；",
                "3. 该记录**不是**独立审核：状态变为 `self_reviewed`，而 `reviewer` / `review_status`",
                "   **保持不变** ⇒ 不产生真值，也不授予 `clickable_verified` 或真实输入资格；",
                "4. 离线评估用 `--status self_reviewed`（`exploratory_self_reviewed` 口径）只看开发期诊断，",
                "   其 `admissible_as_official_metric=False`，**不得**对外宣称准确率。",
                "",
                "## 重要提醒",
                "* 审核包**不会**自动批准任何标注；`approved` 只能由**独立审核者**主动确认；",
                "* `self-review` 命令**没有** `--reviewer` 参数：无法借个人复核伪造独立审核；",
                "* 若对 bbox 有异议，请用 `annotate.py revise` 提交修订（旧草稿**保留**不改写）；",
                "* `victory_icon` 等**状态证据**不应被标为可点击区域。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result: dict[str, object] = {
        "out": str(out_dir),
        "images": len({item.image_path for item in store}),
        "annotations": len(manifest),
        "promoted": 0,
        #: 工具**不代填**任何个人复核结论（逐条判断必须由人完成）
        "self_review_decisions": 0,
        "self_review_form": str(out_dir / "self_review_form.md"),
        "note": "审核包不执行 promote，也不代填个人复核结论；approved 需独立审核者显式确认",
    }
    if near_duplicate:
        result["near_duplicates"] = near_duplicate_scan(dataset=dataset)
        (out_dir / "near_duplicates.json").write_text(
            json.dumps(result["near_duplicates"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="生成元素级标注审核包（只读 raw，不 promote）")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--no-sha256", action="store_true")
    parser.add_argument("--near-duplicate-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.near_duplicate_only:
        report = near_duplicate_scan()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    result = build_pack(
        annotations_path=Path(args.annotations),
        out_dir=Path(args.out),
        verify_sha256=not args.no_sha256,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"审核包输出: {result['out']}")
        print(f"原图 {result['images']} 张 / 标注 {result['annotations']} 条 / promote {result['promoted']} 次")
        print(f"个人复核表: {result['self_review_form']}（工具代填结论 {result['self_review_decisions']} 条）")
        near = result.get("near_duplicates") or {}
        if near:
            print(f"近重复扫描: {near['scanned']} 张；完全重复 {len(near['exact_duplicates'])} 组；"
                  f"近重复 {len(near['near_duplicate_pairs'])} 对（阈值 {near['threshold']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
