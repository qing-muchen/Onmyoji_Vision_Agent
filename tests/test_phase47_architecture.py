"""Phase 4.7 架构收敛的守护测试。

覆盖（对应用户要求）：

1. ``EvidenceHistory`` **无生产调用**（只被证据层自身、配置与文档引用）；
2. :class:`~src.vision.pipeline.frame_processor.EvidenceFrameProcessor`
   —— ``Frame → Evidence → StateResult``；
3. ``StateResult`` 契约**不变**（字段集合与序列化键固定，Runtime 依赖它）；
4. 既有 ``StateMonitor`` 链路继续通过（帧路径 + 稳定化 + 事件 + 落盘）；
5. ``SceneClassifier`` 继续可独立运行（保留为独立视觉组件）；
6. ``examples`` 的 Demo 结构健全，且新的证据 Demo 走证据路径。
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import (
    EvidenceConfig,
    SceneMatcherConfig,
    StateTransitionConfig,
    load_config,
)
from src.core.paths import resolve_from_project_root
from src.vision.capture import Frame
from src.vision.evidence import EvidenceRegistry
from src.vision.pipeline import EvidenceFrameProcessor
from src.vision.scene.classifier import SceneClassifier
from src.vision.scene.loader import SceneLoader
from src.vision.state import StateClassifier, StateName, StateResult, StateSource
from tests.helpers import build_scene_image
from tests.monitor_fakes import build_monitor

#: 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: 允许出现 ``EvidenceHistory`` 的位置（证据层自身 / 配置 / 测试 / 文档）
EVIDENCE_HISTORY_ALLOWED = (
    "src/vision/evidence/",
    "src/core/config.py",
    "tests/",
    "docs/",
)
#: 真实样本（用于端到端用例；缺失时跳过）
REAL_BATTLE_FRAME = "dataset/raw/battle/battle_auto_001.png"


def python_files(*relative_dirs: str) -> list[Path]:
    """收集给定目录下的 Python 文件（相对项目根）。"""
    files: list[Path] = []
    for relative in relative_dirs:
        directory = PROJECT_ROOT / relative
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.py")))
    return files


def relative(path: Path) -> str:
    """返回相对项目根的 POSIX 路径。"""
    return path.relative_to(PROJECT_ROOT).as_posix()


def write_png(path: Path, image: np.ndarray) -> Path:
    """写出 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", image)
    assert success
    buffer.tofile(str(path))
    return path


def load_image(path: Path) -> np.ndarray:
    """读取图像（兼容非 ASCII 路径）。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def frame_of(image: np.ndarray) -> Frame:
    """把图像包装为帧。"""
    return Frame.create(image, source="test")


def minimal_processor() -> EvidenceFrameProcessor:
    """不加载任何数据集的流水线（空注册表 + 证据驱动分类器）。"""
    return EvidenceFrameProcessor(
        EvidenceRegistry([]), StateClassifier.from_evidence_config(EvidenceConfig())
    )


# ---------------------------------------------------------------------------
# 1. EvidenceHistory 无生产调用
# ---------------------------------------------------------------------------


def test_evidence_history_has_no_production_callers() -> None:
    """除证据层自身、配置与测试/文档外，任何源码都不得引用 EvidenceHistory。"""
    offenders = [
        relative(path)
        for path in python_files("src")
        if "EvidenceHistory" in path.read_text(encoding="utf-8")
        and not relative(path).startswith(EVIDENCE_HISTORY_ALLOWED)
    ]

    assert offenders == []


def test_time_filter_belongs_to_state_stabilizer_only() -> None:
    """状态层与运行时层不得出现第二处时间滤波实现。"""
    for directory in ("src/vision/state", "src/runtime", "src/vision/pipeline"):
        for path in python_files(directory):
            source = path.read_text(encoding="utf-8")
            assert "EvidenceHistory" not in source, relative(path)


def test_evidence_history_is_still_importable_and_deprecated() -> None:
    """兼容性：模块仍可导入（保留测试兼容），并在文档中标记 deprecated。"""
    from src.vision.evidence import EvidenceHistory
    from src.vision.evidence.history import EvidenceHistoryConfig

    history = EvidenceHistory(EvidenceHistoryConfig(window_size=3, min_state_count=2))

    assert isinstance(history, EvidenceHistory)
    assert "deprecated" in (EvidenceHistory.__doc__ or "")


# ---------------------------------------------------------------------------
# 2. EvidenceFrameProcessor：Frame → StateResult
# ---------------------------------------------------------------------------


def test_frame_processor_maps_frame_to_state_result() -> None:
    """空注册表下也返回合法 ``StateResult``（类型契约，不依赖数据集）。"""
    processor = minimal_processor()

    result = processor.process(frame_of(np.zeros((961, 1706, 3), dtype=np.uint8)))

    assert isinstance(result, StateResult)
    assert result.state_name is StateName.UNKNOWN
    assert result.confidence == pytest.approx(0.0)
    assert result.source is StateSource.EVIDENCE
    assert result.reason == "no evidence available"
    assert result.evidence == ()


def test_frame_processor_exposes_registry_and_classifier() -> None:
    """编排层组合关系可检查（detector 数 / 规则数）。"""
    processor = minimal_processor()

    assert processor.registry is not None
    assert processor.classifier.aggregator is not None
    assert "EvidenceFrameProcessor" in processor.describe()


def test_frame_processor_requires_evidence_classifier() -> None:
    """不支持证据的分类器会被拒绝（避免静默走错路径）。"""
    from src.vision.state.transition.classifier import TransitionClassifier
    from src.vision.state.transition.dataset import SampleKind, TransitionSample

    sample = TransitionSample(
        sample_id="transition_bottom_001",
        kind=SampleKind.POSITIVE,
        path=Path("transition_bottom_001.png"),
        image=np.zeros((58, 1706, 3), dtype=np.uint8),
    )
    frame_only = StateClassifier(
        transition=TransitionClassifier(StateTransitionConfig(), [sample], []),
        yard=None,
    )

    with pytest.raises(ValueError, match="aggregator"):
        EvidenceFrameProcessor(EvidenceRegistry([]), frame_only)


@pytest.mark.skipif(
    not resolve_from_project_root("dataset/raw").is_dir(), reason="dataset/raw 不存在"
)
def test_frame_processor_on_real_frame() -> None:
    """真实截图端到端：Frame → 证据 → StateResult（battle）。"""
    path = resolve_from_project_root(REAL_BATTLE_FRAME)
    if not path.is_file():
        pytest.skip(f"{path} 不存在")

    processor = EvidenceFrameProcessor.from_config(load_config(environ={}))
    result = processor.process(frame_of(load_image(path)))

    assert result.state_name is StateName.BATTLE
    assert result.source is StateSource.EVIDENCE
    assert result.evidence
    assert result.reason


# ---------------------------------------------------------------------------
# 3. StateResult 契约不变
# ---------------------------------------------------------------------------


def test_state_result_contract_is_unchanged() -> None:
    """字段集合与序列化键固定（Runtime / Agent 依赖该契约）。"""
    fields = {field.name for field in dataclasses.fields(StateResult)}

    assert fields == {"state_name", "confidence", "source", "detail", "reason", "evidence"}

    payload = StateResult(
        state_name=StateName.YARD,
        confidence=1.0,
        source=StateSource.EVIDENCE,
        reason="matched yard evidence rule",
        evidence=("bottom_menu_yard",),
    ).to_dict()

    assert set(payload) == fields
    assert StateResult.from_dict(payload).state_name is StateName.YARD


# ---------------------------------------------------------------------------
# 4. StateMonitor（既有运行时链路）继续通过
# ---------------------------------------------------------------------------


def test_state_monitor_still_runs(tmp_path: Path) -> None:
    """固定节拍 + 稳定化 + 事件 + 落盘不受 Phase 4.7 影响。"""
    monitor, _, _, _ = build_monitor([StateName.YARD], tmp_path=tmp_path)

    report = monitor.run(max_cycles=3)

    assert report.cycles == 3
    assert report.classified == 3
    assert report.initial_state is StateName.YARD
    assert monitor.current_state is StateName.YARD


# ---------------------------------------------------------------------------
# 5. SceneClassifier 继续可独立运行
# ---------------------------------------------------------------------------


def test_scene_classifier_still_runs_standalone(tmp_path: Path) -> None:
    """场景识别保留为独立视觉组件（Phase 4.6.5 只改变状态解释权）。"""
    image = build_scene_image(size=(1706, 961), seed=5)
    write_png(tmp_path / "scenes" / "yard" / "skin_001" / "scene_001" / "yard_skin001_scene001_frame001.png", image)
    classifier = SceneClassifier.from_dataset(
        SceneMatcherConfig(), loader=SceneLoader(tmp_path / "scenes")
    )

    results = classifier.classify(frame_of(image))

    assert results
    assert results[0].confidence > 0.9


# ---------------------------------------------------------------------------
# 6. examples 结构健全 + 证据 Demo 走证据路径
# ---------------------------------------------------------------------------

EXAMPLES_DIR = PROJECT_ROOT / "examples"
DEMO_NAMES = (
    "capture_demo.py",
    "frame_pipeline_demo.py",
    "evidence_demo.py",
    "evidence_state_demo.py",
    "state_classifier_demo.py",
    "state_monitor_demo.py",
    "state_stabilizer_demo.py",
)


@pytest.mark.parametrize("name", DEMO_NAMES)
def test_examples_are_parseable_with_standard_entry(name: str) -> None:
    """每个 Demo 都存在、可解析，并提供 ``build_parser`` / ``main`` 入口。"""
    path = EXAMPLES_DIR / name

    assert path.is_file()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert {"build_parser", "main"} <= functions


def test_evidence_state_demo_uses_evidence_pipeline() -> None:
    """新 Demo 走证据路径（并排除帧路径调用）。"""
    source = (EXAMPLES_DIR / "evidence_state_demo.py").read_text(encoding="utf-8")

    assert "EvidenceFrameProcessor" in source
    assert "classify_frame" not in source


def test_legacy_demos_remain_for_frame_path() -> None:
    """旧 Demo 保留（Phase 4.5/4.6 的兼容决策：帧路径不删除）。"""
    for name in ("state_classifier_demo.py", "state_monitor_demo.py"):
        assert (EXAMPLES_DIR / name).is_file()


# ---------------------------------------------------------------------------
# 附加：文档存在性（Architecture Overview）
# ---------------------------------------------------------------------------


def test_architecture_overview_document_exists() -> None:
    """Phase 4.7 新增的架构总览文档存在且包含主链路说明。"""
    path = PROJECT_ROOT / "docs" / "ARCHITECTURE_OVERVIEW.md"
    source = path.read_text(encoding="utf-8")

    assert "EvidenceRegistry" in source
    assert "StateStabilizer" in source
    assert "时间滤波" in source
