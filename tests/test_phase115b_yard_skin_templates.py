"""Phase 11.5-B 测试：yard 皮肤模板派生与模板选择（全部用 ``tmp_path``）。

覆盖用户要求的边界：

1. 派生只读取 ``dataset/raw``，**不修改**任何 raw 原图；
2. **不修改**任何 annotation（``dataset/annotations`` 不在写入范围）；
3. 派生模板**不计入**独立样本（工具只写派生目录，且明确声明）；
4. **不降低** ``threshold``、**不引入**新 detector（工具与模板选择都不碰阈值/detector 集合）；
5. 生成必须落在派生目录，越界路径直接拒绝；
6. 幂等：同名模板默认跳过，``--force`` 才覆盖；
7. 每种皮肤至少 1 张代表模板（组内 dHash 去重只删冗余，不删来源）；
8. 轮转抽样保证"预算 ≥ 来源数 ⇒ 每个来源至少 1 张"；
9. 模板选择新参数默认关闭 ⇒ 既有行为不变。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

from src.core.config import EvidenceDetectorConfig, load_config
from src.core.paths import get_project_root
from src.vision.dataset.sources import compute_sha256
from src.vision.evidence.templates import load_feature_templates

ROOT = get_project_root()
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_yard_skin_templates as skin_tool  # noqa: E402

YARD = "raw/yard"
DERIVED = "features/bottom_menu/yard/positive"


def noise_image(seed: int, *, tweak: int = 0, width: int = 100, height: int = 50) -> np.ndarray:
    """确定性噪声图（同 seed 且不 tweak ⇒ 内容相同）。"""
    rng = np.random.default_rng(seed)
    image = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    if tweak:
        image[0, 0, 0] = int((int(image[0, 0, 0]) + tweak) % 256)
    return image


def write_png(path: Path, image: np.ndarray) -> Path:
    """写真实 PNG（非 ASCII 路径安全）。"""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def directory_digest(directory: Path) -> dict[str, str]:
    """目录内容快照（相对路径 → SHA256）。"""
    return {
        path.relative_to(directory).as_posix(): compute_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def make_yard_dataset(tmp_path: Path) -> Path:
    """构造最小 dataset：3 张皮肤原图（同皮肤 2 张近重复）+ 1 张 cap + 1 张 exploration。"""
    dataset = tmp_path / "dataset"
    write_png(dataset / YARD / "yard_skin001_001.png", noise_image(1))
    write_png(dataset / YARD / "yard_skin001_002.png", noise_image(1, tweak=7))  # 与 001 近重复
    write_png(dataset / YARD / "yard_skin002_001.png", noise_image(2))
    write_png(dataset / YARD / "yard_cap_001.png", noise_image(3))
    write_png(dataset / "raw/exploration" / "exploration_001.png", noise_image(4))
    return dataset


# ---------------------------------------------------------------------------
# 1-3 / 7. 规划与生成
# ---------------------------------------------------------------------------


def test_plan_covers_every_skin_and_dedupes_within_skin(tmp_path: Path) -> None:
    """候选计划：每种皮肤至少 1 张，组内近重复被去掉。"""
    dataset = make_yard_dataset(tmp_path)

    candidates = skin_tool.plan_candidates(dataset, dedupe_threshold=6)

    assert {item.source for item in candidates} == {"skin001", "skin002"}
    skin001 = [item for item in candidates if item.source == "skin001"]
    assert len(skin001) == 1  # 2 张里只有 1 张代表（另一张近重复）
    assert skin001[0].template_name == "bottom_menu_yard_skin001_001.png"
    # 不裁剪、不写盘：计划阶段目录里没有任何派生文件
    assert not (dataset / DERIVED).exists() or not any((dataset / DERIVED).iterdir())


def test_build_writes_only_derived_dir_and_keeps_raw_and_annotations(tmp_path: Path) -> None:
    """生成只写派生目录：raw 与 annotations 逐字节不变。"""
    dataset = make_yard_dataset(tmp_path)
    annotations = dataset / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / "elements.jsonl").write_text('{"source_id": "src_x"}\n', encoding="utf-8")
    raw_before = directory_digest(dataset / "raw")
    annotations_before = directory_digest(annotations)

    candidates = skin_tool.plan_candidates(dataset, dedupe_threshold=6)
    report = skin_tool.build_templates(candidates, dataset=dataset)

    assert report["candidates"] == 2
    assert report["written"] == 2
    assert report["source_count"] == 2
    assert sorted(report["sources_covered"]) == ["skin001", "skin002"]
    written = sorted(path.name for path in (dataset / DERIVED).glob("*.png"))
    assert written == ["bottom_menu_yard_skin001_001.png", "bottom_menu_yard_skin002_001.png"]
    # raw 与 annotations 完全没被触碰
    assert directory_digest(dataset / "raw") == raw_before
    assert directory_digest(annotations) == annotations_before
    # 生成的模板是 ROI 底部 20% 的裁剪（100x50 → 100x10）
    import cv2

    generated = cv2.imread(str(dataset / DERIVED / "bottom_menu_yard_skin001_001.png"))
    assert generated.shape[:2] == (10, 100)
    # 来源可追溯
    entry = next(item for item in report["templates"] if item["source"] == "skin001")
    assert entry["from_image"] == f"{YARD}/yard_skin001_001.png"
    assert entry["source_id"].startswith("src_")
    assert entry["roi"] == [0.0, 0.8, 1.0, 1.0]


def test_build_is_idempotent_and_refuses_out_of_tree_dir(tmp_path: Path) -> None:
    """幂等（默认跳过已存在），且拒绝写入非派生目录。"""
    dataset = make_yard_dataset(tmp_path)
    candidates = skin_tool.plan_candidates(dataset, dedupe_threshold=6)
    first = skin_tool.build_templates(candidates, dataset=dataset)
    assert first["written"] == 2

    digest_after_first = directory_digest(dataset / DERIVED)
    second = skin_tool.build_templates(candidates, dataset=dataset)
    assert second["written"] == 0 and second["skipped_existing"] == 2
    assert directory_digest(dataset / DERIVED) == digest_after_first  # 未重写

    # 越界目录（raw / annotations / dataset 之外）一律拒绝
    for bad in (dataset / "raw/yard", dataset / "annotations", tmp_path / "outside"):
        with pytest.raises(ValueError):
            skin_tool.build_templates(candidates, dataset=dataset, out_dir=bad)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """``--dry-run`` 只报告计划。"""
    dataset = make_yard_dataset(tmp_path)
    candidates = skin_tool.plan_candidates(dataset, dedupe_threshold=6)

    report = skin_tool.build_templates(candidates, dataset=dataset, dry_run=True)

    assert report["dry_run"] is True
    assert report["written"] == 0 and report["candidates"] == 2
    assert not (dataset / DERIVED).exists()
    assert all(item["action"] == "would_write" for item in report["templates"])


# ---------------------------------------------------------------------------
# 8-9. 模板选择：轮转抽样 + 默认行为不变
# ---------------------------------------------------------------------------


def test_round_robin_keeps_every_source(tmp_path: Path, ) -> None:
    """预算 ≥ 来源数时，轮转抽样保证每个来源至少 1 张（均匀抽样可能整类漏掉）。"""
    for index in range(6):
        write_png(tmp_path / f"probe_cap_{index + 1:03d}.png", noise_image(100 + index))
    for index in range(4):
        write_png(tmp_path / f"probe_skin001_{index + 1:03d}.png", noise_image(200 + index))
    write_png(tmp_path / "probe_skin002_001.png", noise_image(300))

    round_robin = load_feature_templates(
        tmp_path, "probe", limit=3, round_robin=True, root=tmp_path
    )
    assert sorted(item.skin for item in round_robin) == ["cap", "skin001", "skin002"]

    # 均匀抽样在同样预算下会漏掉排序靠后的来源（这里 skin002 在末尾）
    evenly = load_feature_templates(tmp_path, "probe", limit=3, root=tmp_path)
    assert "skin002" not in {item.skin for item in evenly}


def test_template_selection_defaults_unchanged() -> None:
    """新参数默认关闭：配置默认值 0/False，且真实配置的 yard 段未被本阶段改动阈值。"""
    config = EvidenceDetectorConfig()
    assert config.dedupe_threshold == 0
    assert config.per_source_limit == 0
    assert config.round_robin is False

    live = load_config(environ={}).evidence.detectors["bottom_menu_yard"]
    assert live.threshold == pytest.approx(0.85)  # 阈值不变
    assert tuple(live.roi) == (0.0, 0.8, 1.0, 1.0)


def test_skin_template_tool_is_read_only_outside_derived_dir() -> None:
    """生成工具不 import Action / 输入原语，且写入目标被限制在派生目录。"""
    path = TOOLS / "build_yard_skin_templates.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(
                    ("src.action", "src.runtime", "pyautogui", "pynput")
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(
                ("src.action", "src.runtime", "pyautogui", "pynput")
            )
    assert "_assert_derived_dir" in text  # 目标目录越界会直接拒绝
    assert "dataset/**" not in text


# ---------------------------------------------------------------------------
# 接入验证（Phase 11.5-B：bottom_menu_yard = 12 张 + round_robin）
# ---------------------------------------------------------------------------

#: 派生模板目录（缺失时相关测试 skip —— 它是本地派生产物，不入库）
LIVE_TEMPLATE_DIR = ROOT / "dataset" / "features/bottom_menu/yard/positive"
LIVE_SOURCES_MANIFEST = ROOT / "dataset" / "features/bottom_menu/yard/positive_sources.json"
#: 稳健耗时基准（30 张 × 3 次取中位数，见 Phase 11.5-B 报告 §4）：
#: 整链 ≈ 其它 detector 基线 + 每模板匹配耗时 × 模板数
CHAIN_BASE_MS = 282.0
CHAIN_PER_TEMPLATE_MS = 17.0
#: 运行时采集/识别周期（configs/config.yaml: runtime.state_monitor.interval_ms）
CYCLE_BUDGET_MS = 500.0


def live_yard_config():
    """取运行期 ``bottom_menu_yard`` 配置（模板目录缺失时 skip）。"""
    if not LIVE_TEMPLATE_DIR.is_dir():
        pytest.skip("派生模板目录不存在（先运行 tools/build_yard_skin_templates.py）")
    return load_config(environ={}).evidence.detectors["bottom_menu_yard"]


def test_live_config_uses_budgeted_templates_and_round_robin() -> None:
    """接入后的真实配置：12 张 + 轮转；阈值 / ROI / 证据集合保持不变。"""
    config = live_yard_config()

    assert config.max_templates == 12
    assert config.round_robin is True
    assert config.dedupe_threshold == 6
    assert config.per_source_limit == 0
    assert config.threshold == pytest.approx(0.85)  # **未降低阈值**
    assert tuple(config.roi) == (0.0, 0.8, 1.0, 1.0)  # ROI 未改
    assert config.text_targets == []  # 未引入第二证据 / 新 detector


def test_live_config_respects_the_500ms_cycle_budget() -> None:
    """确定性预算断言：预测整链耗时必须落在 500ms 节拍内（防止日后盲目调大模板数）。"""
    config = live_yard_config()
    predicted_ms = CHAIN_BASE_MS + CHAIN_PER_TEMPLATE_MS * config.max_templates

    assert predicted_ms <= CYCLE_BUDGET_MS, (
        f"预测整链 {predicted_ms:.0f}ms 超出 {CYCLE_BUDGET_MS:.0f}ms 节拍预算；"
        "若确实要扩模板，必须先做匹配耗时优化（Phase 11.6）"
    )


def test_live_detector_loads_exactly_max_templates_and_spreads_sources() -> None:
    """实际加载模板数 = max_templates，且轮转让 12 张来自 12 个不同来源。"""
    from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector

    detector = BottomMenuYardDetector.from_config(load_config(environ={}).evidence)

    assert len(detector.templates) == 12
    sources = [item.skin for item in detector.templates]
    assert len(set(sources)) == 12  # 每来源 1 张（轮转）


def test_live_templates_have_source_manifest() -> None:
    """模板来源映射已保存：每个模板都能追溯到一张 raw/yard 原图。"""
    if not LIVE_SOURCES_MANIFEST.is_file():
        pytest.skip("来源映射不存在（先运行 tools/build_yard_skin_templates.py）")
    import json

    manifest = json.loads(LIVE_SOURCES_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["generated_from"] == "dataset/raw/yard"
    assert manifest["roi"] == [0.0, 0.8, 1.0, 1.0]
    assert manifest["source_count"] == 18
    assert len(manifest["templates"]) == manifest["candidates"] == 33
    for item in manifest["templates"]:
        assert item["from_image"].startswith("raw/yard/yard_skin")
        assert item["source_id"].startswith("src_")
        assert (ROOT / "dataset" / item["from_image"]).is_file()  # 来源原图存在
    assert "不计入" in manifest["note"]


def test_live_templates_never_false_hit_other_screens() -> None:
    """新模板集在**全部**负样本（80 张）上不产生假命中 ⇒ FP 不增加。"""
    if not LIVE_TEMPLATE_DIR.is_dir():
        pytest.skip("派生模板目录不存在")
    from src.vision.capture.models import Frame
    from src.vision.evidence.detectors.bottom_menu import BottomMenuYardDetector

    detector = BottomMenuYardDetector.from_config(load_config(environ={}).evidence)
    negatives: list[Path] = []
    for group in ("raw/exploration", "raw/unknown", "raw/battle", "raw/battle_result", "raw/battle_reward"):
        negatives.extend(sorted((ROOT / "dataset" / group).rglob("*.png")))

    false_hits: list[str] = []
    for index, path in enumerate(negatives, start=1):
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        frame = Frame.create(image, source=path.name, timestamp=float(index))
        if detector.detect(frame).detected:
            false_hits.append(path.name)

    assert not false_hits, f"负样本出现 yard 证据假命中: {false_hits}"


def test_live_yard_chain_latency_smoke() -> None:
    """实测整链延迟（8 张 yard 图取中位数）落在 500ms 节拍内。

    Notes:
        该断言受机器负载影响；若失败请先运行
        ``python tools/audit_yard_dataset.py --timing``（30 张 × 3 次取中位数）复核 ——
        那里的线性模型（``整链 ≈ 282ms + 17ms×模板数``）是不受单次抖动影响的依据。
    """
    if not LIVE_TEMPLATE_DIR.is_dir():
        pytest.skip("派生模板目录不存在")
    import time

    from src.vision.capture.models import Frame
    from src.vision.pipeline.frame_processor import (
        EvidenceFrameProcessor,
        build_transition_probe,
    )

    config = load_config(environ={})
    processor = EvidenceFrameProcessor.from_config(
        config, transition_classifier=build_transition_probe(config)
    )
    paths = sorted((ROOT / "dataset" / "raw" / "yard").glob("*.png"))[:8]
    if not paths:
        pytest.skip("缺少 yard 原图")

    durations: list[float] = []
    for index, path in enumerate(paths, start=1):
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        frame = Frame.create(image, source=path.name, timestamp=float(index))
        started = time.perf_counter()
        processor.process(frame)
        durations.append((time.perf_counter() - started) * 1000.0)

    median_ms = float(np.median(durations))
    assert median_ms < CYCLE_BUDGET_MS, f"整链延迟中位数 {median_ms:.0f}ms 超出 500ms 节拍预算"

