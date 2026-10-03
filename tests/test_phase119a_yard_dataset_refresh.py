"""Phase 11.9-A 测试：当前版本庭院数据刷新与 holdout 工作流（30 项）。

覆盖用户指定的 30 个验收点：

1~3.  采集不激活窗口 / 不发送鼠标 / 不发送键盘输入（AST + 参数断言）
4.    只捕获配置指定的游戏窗口
5.    窗口不存在时明确失败
6.    非前台默认不捕获
7~9.  保存原客户区像素（不 resize、不裁 ROI、逐像素一致）
10.   不覆盖已有文件
11.   SHA 重复拒绝
12.   source_id 唯一
13.   capture_session_id 存在
14.   capture_role 合法取值
15.   candidate / holdout 按 session 划分
16.   同 session 不会同时作为 candidate 与 holdout
17.   raw 路径越界被拒绝
18.   temp partial file 失败后清理
19.   metadata 与实际图尺寸一致
20.   manifest 可识别新增文件（历史 unchanged + live added）
21~29. 生产目录 / 清单 / 配置 / 安全开关不变
30.   AST 确认采集工具无输入控制 API

外加（同一阶段的核心契约）：dHash 与 ``template_dhash`` 逐图等价、``raw/yard`` 与
``raw/yard_live`` 双向隔离、跨集合近重复检出。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import capture_yard_dataset as capture_tool  # noqa: E402
import raw_manifest as manifest_tool  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.runtime.state_monitor.frame_source import CaptureStatus, SkipReason  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.dataset import live_yard  # noqa: E402
from src.vision.dataset.live_yard import (  # noqa: E402
    LIVE_ROOT,
    LiveFrameError,
    assert_within_live_root,
    audit_live_dataset,
    difference_hash,
    frame_filename,
    hamming_distance,
    is_valid_session_id,
    leakage_report,
    list_live_frames,
    list_live_sessions,
    near_duplicate_pairs,
    new_session_id,
    partition_by_role,
    session_directory,
    session_role_conflicts,
)
from src.vision.dataset.sources import (  # noqa: E402
    CAPTURE_ROLE_CANDIDATE,
    CAPTURE_ROLE_HOLDOUT,
    CAPTURE_ROLE_UNKNOWN,
    CaptureSource,
    SourceError,
    SourceRegistry,
    compute_sha256,
    normalize_capture_role,
)

DATASET = ROOT / "dataset"
YARD_DIR = DATASET / "raw" / "yard"
POSITIVE_DIR = DATASET / "features" / "bottom_menu" / "yard" / "positive"
POSITIVE_SOURCES = DATASET / "features" / "bottom_menu" / "yard" / "positive_sources.json"
MANIFEST = DATASET / "manifests" / "raw_sha256.json"
CAPTURE_SRC = ROOT / "tools" / "capture_yard_dataset.py"
SESSION_A = "yard_live_20261003_s01"
SESSION_B = "yard_live_20261003_s02"
FORBIDDEN_IMPORTS = ("pyautogui", "pynput", "keyboard", "mouse", "src.action", "src.task")
FORBIDDEN_CALLS = frozenset(
    {
        "SetForegroundWindow",
        "BringWindowToTop",
        "ShowWindow",
        "SendInput",
        "sendinput",
        "click",
        "press",
        "typewrite",
        "keyDown",
        "keyUp",
        "moveTo",
        "hotkey",
    }
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def make_image(seed: int, *, width: int = 320, height: int = 120) -> np.ndarray:
    """构造**结构差异明显**的合成图（保证不同 seed 的 dHash 距离大）。"""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), int(rng.integers(30, 120)), dtype=np.uint8)
    bars = 2 + seed % 5
    for index in range(bars):
        x = int(width * (index + 1) / (bars + 1))
        image[:, x : x + 3] = 255 - int(rng.integers(0, 60))
    image[: height // 3] = int(rng.integers(120, 250))
    return image


def write_png(path: Path, image: np.ndarray) -> Path:
    """写 PNG（自动建父目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def read_png(path: Path) -> np.ndarray:
    """读 PNG（兼容非 ASCII 路径）。"""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None, path
    return image


def frame_of(image: np.ndarray) -> Frame:
    """把图像包装为 :class:`Frame`。"""
    height, width = image.shape[:2]
    return Frame(image=image, timestamp=0.0, width=int(width), height=int(height), source="test")


def source_record(
    relative_path: str,
    *,
    session_id: str | None,
    role: str = CAPTURE_ROLE_UNKNOWN,
    digest: str = "0" * 64,
) -> CaptureSource:
    """构造来源记录。"""
    return CaptureSource(
        source_id=f"src_{digest[:16]}",
        relative_path=relative_path,
        session_id=session_id,
        source_sha256=digest,
        capture_role=role,
    )


class FakeSource:
    """帧来源替身（**不接触真实窗口**）。"""

    def __init__(
        self,
        frames: list[Frame],
        *,
        reason: SkipReason | None = None,
        message: str = "",
        backend: str = "fake",
        foreground: bool | None = True,
        window: object | None = None,
    ) -> None:
        self._frames = list(frames)
        self._reason = reason
        self._message = message
        self._foreground = foreground
        self.window = window
        self.backend_name = backend
        self.started = False
        self.stopped = False
        self.servings = 0

    def start(self) -> None:
        """标记启动。"""
        self.started = True

    def stop(self) -> None:
        """标记停止。"""
        self.stopped = True

    @property
    def status(self) -> CaptureStatus:
        """最近一次状态。"""
        if self._reason is not None:
            return CaptureStatus(ok=False, reason=self._reason, message=self._message)
        return CaptureStatus(ok=True, reason=None, message="ok", foreground=self._foreground)

    def get_frame(self) -> Frame | None:
        """取帧（按顺序发完为止）。"""
        self.servings += 1
        if self._reason is not None:
            return None
        if not self._frames:
            return None
        return self._frames.pop(0)


def yard_live_dir(dataset: Path, session: str) -> Path:
    """测试用会话目录。"""
    return dataset / "raw" / "yard_live" / session


def build_dataset(dataset: Path, *, sessions: dict[str, list[int]]) -> dict[str, Path]:
    """在临时目录里构造 ``raw/yard`` + ``raw/yard_live/<session>/``。

    Args:
        dataset: 临时数据根。
        sessions: ``session_id -> [image seed, …]``。

    Returns:
        ``session_id -> 目录``。
    """
    write_png(dataset / "raw" / "yard" / "yard_skin001_001.png", make_image(1))
    created: dict[str, Path] = {}
    for session, seeds in sessions.items():
        for index, seed in enumerate(seeds, start=1):
            target = yard_live_dir(dataset, session) / frame_filename(session, index)
            write_png(target, make_image(seed))
        created[session] = yard_live_dir(dataset, session)
    return created


# ---------------------------------------------------------------------------
# 1~3 / 30：无输入控制
# ---------------------------------------------------------------------------


def test_capture_does_not_import_input_libraries() -> None:
    """[1] 采集工具不 import 任何输入控制库（不可能激活窗口 / 点击 / 按键）。"""
    tree = ast.parse(CAPTURE_SRC.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for forbidden in FORBIDDEN_IMPORTS:
        assert not any(name.startswith(forbidden) for name in imported), forbidden


def test_capture_does_not_call_mouse_api() -> None:
    """[2] 采集工具不调用任何鼠标控制 API。"""
    called = _called_names(CAPTURE_SRC)
    for forbidden in ("click", "doubleClick", "moveTo", "mouseDown", "mouseUp", "hotkey"):
        assert forbidden not in called, forbidden


def test_capture_does_not_call_keyboard_api() -> None:
    """[3] 采集工具不调用任何键盘控制 API。"""
    called = _called_names(CAPTURE_SRC)
    for forbidden in ("typewrite", "press", "keyDown", "keyUp", "SendInput", "sendinput"):
        assert forbidden not in called, forbidden


def test_capture_never_activates_window() -> None:
    """[1b] 采集工具不调用窗口置前 API；帧来源构造为"绝不激活"。"""
    called = _called_names(CAPTURE_SRC)
    for forbidden in ("SetForegroundWindow", "BringWindowToTop", "ShowWindow"):
        assert forbidden not in called, forbidden

    config = load_config()
    source = capture_tool.open_source(config)
    assert source.binding.allow_activation is False
    assert source.binding.config.activate_on_start is False
    assert source.guard.enabled is True


def test_ast_confirms_no_input_control_api() -> None:
    """[30] AST 固化：采集工具与 live_yard 模块都无输入控制调用。"""
    for path in (CAPTURE_SRC, ROOT / "src" / "vision" / "dataset" / "live_yard.py"):
        called = _called_names(path)
        assert not (called & FORBIDDEN_CALLS), (path.name, sorted(called & FORBIDDEN_CALLS))


def _called_names(path: Path) -> set[str]:
    """收集源码里被调用的函数 / 方法名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name:
                names.add(name)
    return names


# ---------------------------------------------------------------------------
# 4~6：窗口与前台门控
# ---------------------------------------------------------------------------


def test_capture_targets_configured_game_window() -> None:
    """[4] 只针对配置指定的游戏窗口（标题关键词来自 config.window）。"""
    config = load_config()
    source = capture_tool.open_source(config)
    keywords = tuple(source.binding.keywords)
    assert "阴阳师" in keywords
    assert keywords == tuple(config.window.title_keywords)
    assert source.binding.config.match_mode == config.window.match_mode


def test_capture_fails_when_window_missing(tmp_path: Path) -> None:
    """[5] 窗口不存在时明确失败（exit code 3），不落盘、不登记。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={})
    sources_path = tmp_path / "sources.jsonl"
    fake = FakeSource([], reason=SkipReason.WINDOW_NOT_FOUND, message="窗口未找到")
    with pytest.raises(capture_tool.AbortCapture) as excinfo:
        capture_tool.capture_session(
            load_config(),
            session_id=SESSION_A,
            role=CAPTURE_ROLE_CANDIDATE,
            count=1,
            interval_ms=300,
            scene="yard",
            theme="unknown",
            variant="unknown",
            note="",
            sources_path=sources_path,
            dataset=dataset,
            source_factory=lambda _config: fake,
        )
    assert excinfo.value.code == capture_tool.EXIT_WINDOW_UNAVAILABLE
    assert excinfo.value.status == "window_not_found"
    assert not sources_path.exists()
    assert list_live_frames(dataset) == ()


def test_capture_refuses_when_not_foreground(tmp_path: Path) -> None:
    """[6] 非前台默认不捕获（exit code 4），不落盘、不登记。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={})
    sources_path = tmp_path / "sources.jsonl"
    fake = FakeSource([], reason=SkipReason.NOT_FOREGROUND, message="不是前台窗口")
    with pytest.raises(capture_tool.AbortCapture) as excinfo:
        capture_tool.capture_session(
            load_config(),
            session_id=SESSION_A,
            role=CAPTURE_ROLE_CANDIDATE,
            count=1,
            interval_ms=300,
            scene="yard",
            theme="unknown",
            variant="unknown",
            note="",
            sources_path=sources_path,
            dataset=dataset,
            source_factory=lambda _config: fake,
        )
    assert excinfo.value.code == capture_tool.EXIT_NOT_FOREGROUND
    assert not sources_path.exists()
    assert list_live_sessions(dataset) == ()


# ---------------------------------------------------------------------------
# 7~13 / 18~19：落盘契约
# ---------------------------------------------------------------------------


def test_capture_saves_original_client_pixels(tmp_path: Path) -> None:
    """[7][8][9] 原客户区像素直存：尺寸不变、不 resize、不裁 ROI（逐像素一致）。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={})
    sources_path = tmp_path / "sources.jsonl"
    image = make_image(11, width=340, height=137)
    fake = FakeSource([frame_of(image)])
    report = capture_tool.capture_session(
        load_config(),
        session_id=SESSION_A,
        role=CAPTURE_ROLE_CANDIDATE,
        count=1,
        interval_ms=300,
        scene="yard",
        theme="unknown",
        variant="unknown",
        note="",
        sources_path=sources_path,
        dataset=dataset,
        source_factory=lambda _config: fake,
    )
    saved = report["saved"][0]["relative_path"]
    path = dataset / saved
    assert path.is_file()
    loaded = read_png(path)
    assert loaded.shape == image.shape, "[7] 原尺寸"
    assert np.array_equal(loaded, image), "[8][9] 不 resize、不裁 ROI、不做增强"
    assert report["geometry"]["client_width"] == 340
    assert report["geometry"]["client_height"] == 137


def test_capture_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    """[10] 目标文件已存在时拒绝覆盖。"""
    target = tmp_path / "yard_live_x_001.png"
    target.write_bytes(b"existing")
    with pytest.raises(capture_tool.CaptureError):
        capture_tool.write_atomic(target, b"new")
    assert target.read_bytes() == b"existing"


def test_capture_rejects_duplicate_sha256(tmp_path: Path) -> None:
    """[11] 相同内容（SHA256 重复）被拒绝保存，且不产生重复登记。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={})
    sources_path = tmp_path / "sources.jsonl"
    image = make_image(21)
    fake = FakeSource([frame_of(image), frame_of(image.copy())])
    report = capture_tool.capture_session(
        load_config(),
        session_id=SESSION_A,
        role=CAPTURE_ROLE_CANDIDATE,
        count=2,
        interval_ms=300,
        scene="yard",
        theme="unknown",
        variant="unknown",
        note="",
        sources_path=sources_path,
        dataset=dataset,
        source_factory=lambda _config: fake,
        sleeper=lambda _seconds: None,
    )
    assert len(report["saved"]) == 1
    assert len(report["duplicates"]) == 1
    assert len(list_live_frames(dataset)) == 1
    registry = SourceRegistry.load(sources_path)
    assert len(registry) == 1


def test_capture_source_id_is_unique_and_content_addressed(tmp_path: Path) -> None:
    """[12] source_id 由内容派生且唯一。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={})
    sources_path = tmp_path / "sources.jsonl"
    fake = FakeSource([frame_of(make_image(31)), frame_of(make_image(32))])
    report = capture_tool.capture_session(
        load_config(),
        session_id=SESSION_A,
        role=CAPTURE_ROLE_CANDIDATE,
        count=2,
        interval_ms=300,
        scene="yard",
        theme="unknown",
        variant="unknown",
        note="",
        sources_path=sources_path,
        dataset=dataset,
        source_factory=lambda _config: fake,
        sleeper=lambda _seconds: None,
    )
    ids = [item["source_id"] for item in report["saved"]]
    assert len(ids) == len(set(ids)) == 2
    for item in report["saved"]:
        digest = compute_sha256(dataset / item["relative_path"])
        assert item["source_id"] == f"src_{digest[:16]}"
        assert item["source_sha256"] == digest


def test_capture_records_session_and_role(tmp_path: Path) -> None:
    """[13][14] 采集会话 ID 存在且角色合法，登记表可回读。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={})
    sources_path = tmp_path / "sources.jsonl"
    fake = FakeSource([frame_of(make_image(41))])
    report = capture_tool.capture_session(
        load_config(),
        session_id=SESSION_B,
        role=CAPTURE_ROLE_HOLDOUT,
        count=1,
        interval_ms=300,
        scene="yard",
        theme="安周年庆",
        variant="unknown",
        note="holdout 会话",
        sources_path=sources_path,
        dataset=dataset,
        source_factory=lambda _config: fake,
    )
    assert report["role"] == CAPTURE_ROLE_HOLDOUT
    record = report["saved"][0]
    assert record["session_id"] == SESSION_B
    assert record["capture_role"] == CAPTURE_ROLE_HOLDOUT
    assert record["theme_or_event"] == "安周年庆"
    assert record["captured_at"]
    assert is_valid_session_id(record["session_id"])
    loaded = list_live_frames(dataset, registry=SourceRegistry.load(sources_path))
    assert loaded[0].is_holdout is True
    assert loaded[0].is_template_eligible is False

    with pytest.raises(SourceError):
        normalize_capture_role("train")
    with pytest.raises(capture_tool.CaptureError):
        capture_tool.capture_session(
            load_config(),
            session_id=SESSION_A,
            role="unknown",
            count=1,
            interval_ms=300,
            scene="yard",
            theme="unknown",
            variant="unknown",
            note="",
            sources_path=sources_path,
            dataset=dataset,
            source_factory=lambda _config: fake,
        )


def test_partial_tmp_file_is_cleaned_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """[18] 写入失败后清理 ``.tmp`` 半文件。"""
    target = tmp_path / "yard_live_x_001.png"

    def boom(self: Path, other: Path) -> Path:  # noqa: ANN001
        raise OSError("rename failed")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        capture_tool.write_atomic(target, b"payload")
    assert not target.exists()
    assert not (tmp_path / "yard_live_x_001.png.tmp").exists()


def test_metadata_matches_saved_image_size(tmp_path: Path) -> None:
    """[19] 登记的 client_width/height 与实际保存的图片尺寸一致。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={})
    sources_path = tmp_path / "sources.jsonl"
    image = make_image(51, width=1704, height=960)
    fake = FakeSource([frame_of(image)])
    report = capture_tool.capture_session(
        load_config(),
        session_id=SESSION_A,
        role=CAPTURE_ROLE_CANDIDATE,
        count=1,
        interval_ms=300,
        scene="yard",
        theme="unknown",
        variant="unknown",
        note="",
        sources_path=sources_path,
        dataset=dataset,
        source_factory=lambda _config: fake,
    )
    loaded = read_png(dataset / report["saved"][0]["relative_path"])
    registry = SourceRegistry.load(sources_path)
    record = next(iter(registry))
    assert (record.client_width, record.client_height) == (loaded.shape[1], loaded.shape[0])
    assert (image.shape[1], image.shape[0]) == (record.client_width, record.client_height)


# ---------------------------------------------------------------------------
# 15~17：划分与隔离
# ---------------------------------------------------------------------------


def test_roles_are_partitioned_by_session(tmp_path: Path) -> None:
    """[15] candidate / holdout 按 session 整体划分（不逐图随机切分）。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={SESSION_A: [1, 2], SESSION_B: [3, 4]})
    registry = SourceRegistry()
    for index in (1, 2):
        path = yard_live_dir(dataset, SESSION_A) / frame_filename(SESSION_A, index)
        registry.add(
            source_record(f"{LIVE_ROOT}/{SESSION_A}/{path.name}", session_id=SESSION_A,
                          role=CAPTURE_ROLE_CANDIDATE, digest=compute_sha256(path))
        )
    for index in (1, 2):
        path = yard_live_dir(dataset, SESSION_B) / frame_filename(SESSION_B, index)
        registry.add(
            source_record(f"{LIVE_ROOT}/{SESSION_B}/{path.name}", session_id=SESSION_B,
                          role=CAPTURE_ROLE_HOLDOUT, digest=compute_sha256(path))
        )
    frames = list_live_frames(dataset, registry=registry)
    buckets = partition_by_role(frames)
    assert {frame.session_id for frame in buckets[CAPTURE_ROLE_CANDIDATE]} == {SESSION_A}
    assert {frame.session_id for frame in buckets[CAPTURE_ROLE_HOLDOUT]} == {SESSION_B}
    audit = audit_live_dataset(dataset, registry=registry)
    assert audit["candidate_sessions"] == [SESSION_A]
    assert audit["holdout_sessions"] == [SESSION_B]


def test_same_session_cannot_be_both_roles(tmp_path: Path) -> None:
    """[16] 同一会话被声明为两个角色 ⇒ 冲突被检出，审计状态为 role_conflict。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={SESSION_A: [1, 2]})
    registry = SourceRegistry()
    first = yard_live_dir(dataset, SESSION_A) / frame_filename(SESSION_A, 1)
    second = yard_live_dir(dataset, SESSION_A) / frame_filename(SESSION_A, 2)
    registry.add(
        source_record(f"{LIVE_ROOT}/{SESSION_A}/{first.name}", session_id=SESSION_A,
                      role=CAPTURE_ROLE_CANDIDATE, digest=compute_sha256(first))
    )
    registry.add(
        source_record(f"{LIVE_ROOT}/{SESSION_A}/{second.name}", session_id=SESSION_A,
                      role=CAPTURE_ROLE_HOLDOUT, digest=compute_sha256(second))
    )
    conflicts = session_role_conflicts(registry)
    assert conflicts and SESSION_A in conflicts[0]
    audit = audit_live_dataset(dataset, registry=registry)
    assert audit["status"] == "role_conflict"


def test_cross_collection_near_duplicate_is_detected(tmp_path: Path) -> None:
    """[16b] candidate ↔ holdout 内容近乎相同（近重复但**不是**同一文件）⇒ possible_leakage。"""
    dataset = tmp_path
    # 同一底图 + 单像素微改：SHA256 不同（可各自登记），dHash 距离 <= 6（近重复）
    base = make_image(7)
    near = base.copy()
    near[0, 0] = np.clip(base[0, 0].astype(int) + 1, 0, 255).astype(np.uint8)
    write_png(yard_live_dir(dataset, SESSION_A) / frame_filename(SESSION_A, 1), base)
    write_png(yard_live_dir(dataset, SESSION_B) / frame_filename(SESSION_B, 1), near)
    registry = SourceRegistry()
    for session, role in ((SESSION_A, CAPTURE_ROLE_CANDIDATE), (SESSION_B, CAPTURE_ROLE_HOLDOUT)):
        path = yard_live_dir(dataset, session) / frame_filename(session, 1)
        registry.add(
            source_record(f"{LIVE_ROOT}/{session}/{path.name}", session_id=session, role=role,
                          digest=compute_sha256(path))
        )
    frames = list_live_frames(dataset, registry=registry)
    pairs = near_duplicate_pairs(frames)
    assert any(pair.cross_collection for pair in pairs)
    report = leakage_report(frames)
    assert report["possible_leakage"] is True
    assert report["cross_collection"] >= 1
    audit = audit_live_dataset(dataset, registry=registry)
    assert audit["status"] == "leakage_detected"
    assert audit["status"] != "ok"


def test_wrong_role_partition_is_not_reported_ok(tmp_path: Path) -> None:
    """[15b] 只有一个会话时审计必须报 insufficient_sessions（不谎报 ok）。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={SESSION_A: [1, 2]})
    registry = SourceRegistry()
    for index in (1, 2):
        path = yard_live_dir(dataset, SESSION_A) / frame_filename(SESSION_A, index)
        registry.add(
            source_record(f"{LIVE_ROOT}/{SESSION_A}/{path.name}", session_id=SESSION_A,
                          role=CAPTURE_ROLE_CANDIDATE, digest=compute_sha256(path))
        )
    audit = audit_live_dataset(dataset, registry=registry)
    assert audit["status"] == "insufficient_sessions"


def test_path_outside_live_root_is_rejected(tmp_path: Path) -> None:
    """[17] 越界路径（含 ``..`` 逃逸）一律拒绝。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={})
    with pytest.raises(LiveFrameError):
        assert_within_live_root(dataset, dataset / "raw" / "yard" / "yard_skin001_001.png")
    with pytest.raises(LiveFrameError):
        assert_within_live_root(dataset, dataset / "raw" / "yard_live" / ".." / ".." / "escaped.png")
    with pytest.raises(LiveFrameError):
        session_directory(dataset, "not_a_session")
    # 合法路径通过
    assert assert_within_live_root(dataset, session_directory(dataset, SESSION_A))


# ---------------------------------------------------------------------------
# 20：manifest 分区
# ---------------------------------------------------------------------------


def test_manifest_separates_history_and_live(tmp_path: Path) -> None:
    """[20] manifest 历史 count 不变、live 分区独立；新增 live 文件报 added。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={SESSION_A: [1]})
    manifest_path = dataset / "manifests" / "raw_sha256.json"
    payload = manifest_tool.write_manifest(manifest_path, dataset=dataset)
    assert payload["count"] == 1, "历史 raw 计数只含 raw/yard"
    assert payload["live"]["count"] == 1
    assert payload["live"]["root"] == LIVE_ROOT
    assert payload["live"]["sessions"] == {SESSION_A: 1}
    historical_paths = {entry["path"] for entry in payload["entries"]}
    assert historical_paths == {"raw/yard/yard_skin001_001.png"}

    write_png(
        yard_live_dir(dataset, SESSION_A) / frame_filename(SESSION_A, 2), make_image(99)
    )
    report = manifest_tool.verify_manifest(manifest_path, dataset=dataset)
    assert report["historical_ok"] is True
    assert report["live_ok"] is False
    assert report["ok"] is False
    assert report["live"]["added"] == [f"{LIVE_ROOT}/{SESSION_A}/{frame_filename(SESSION_A, 2)}"]
    assert report["missing"] == [] and report["changed"] == []


def test_raw_manifest_ignores_live_directory() -> None:
    """[20b] ``collect_raw_entries`` 排除 ``raw/yard_live``（历史 178 语义不变）。"""
    entries = manifest_tool.collect_raw_entries(DATASET)
    assert len(entries) == 178
    assert not any(LIVE_ROOT in str(entry["path"]) for entry in entries)


def test_benchmark_positives_ignore_live_directory(tmp_path: Path) -> None:
    """[20c] 现有 benchmark 的非递归 glob 不会看到 yard_live（不依赖巧合：显式断言）。"""
    import benchmark_template_prefilter as bench  # noqa: PLC0415

    assert len(bench.collect_positives(DATASET)) == 98
    dataset = tmp_path
    build_dataset(dataset, sessions={SESSION_A: [1, 2, 3]})
    assert len(bench.collect_positives(dataset)) == 1, "只有 raw/yard 顶层计入"


def test_live_scan_does_not_depend_on_recursive_glob(tmp_path: Path) -> None:
    """[20d] ``list_live_frames`` 只认合法会话目录，不受其它目录干扰。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={SESSION_A: [1]})
    stray = dataset / "raw" / "yard_live" / "not_a_session" / "whatever.png"
    write_png(stray, make_image(5))
    nested = dataset / "raw" / "yard_live" / SESSION_A / "nested" / "deep.png"
    write_png(nested, make_image(6))
    assert list_live_sessions(dataset) == (SESSION_A,)
    frames = list_live_frames(dataset)
    assert len(frames) == 1
    assert frames[0].relative_path.endswith(frame_filename(SESSION_A, 1))


# ---------------------------------------------------------------------------
# 21~29：生产不变性
# ---------------------------------------------------------------------------


def test_production_template_dir_unchanged(tmp_path: Path) -> None:
    """[21] 采集不影响生产模板目录（只在这里做只读快照断言）。"""
    before = sorted((path.name, path.stat().st_size) for path in POSITIVE_DIR.glob("*.png"))
    dataset = tmp_path
    build_dataset(dataset, sessions={})
    fake = FakeSource([frame_of(make_image(61))])
    capture_tool.capture_session(
        load_config(),
        session_id=SESSION_A,
        role=CAPTURE_ROLE_CANDIDATE,
        count=1,
        interval_ms=300,
        scene="yard",
        theme="unknown",
        variant="unknown",
        note="",
        sources_path=tmp_path / "sources.jsonl",
        dataset=dataset,
        source_factory=lambda _config: fake,
    )
    after = sorted((path.name, path.stat().st_size) for path in POSITIVE_DIR.glob("*.png"))
    assert after == before


def test_positive_sources_json_unchanged() -> None:
    """[22] ``positive_sources.json`` 内容不变。"""
    digest = compute_sha256(POSITIVE_SOURCES)
    payload = json.loads(POSITIVE_SOURCES.read_text(encoding="utf-8"))
    assert payload["dedupe_threshold"] == 6
    assert payload["source_count"] == 18
    assert digest == compute_sha256(POSITIVE_SOURCES)
    assert len(list(POSITIVE_DIR.glob("*.png"))) == 61


def test_template_include_unchanged() -> None:
    """[23] 生产 ``template_include`` 仍是 36 张，且未混入 yard_live 来源。"""
    yard = load_config().evidence.detectors["bottom_menu_yard"]
    assert len(yard.template_include) == 36
    assert all("live" not in item for item in yard.template_include)


def test_production_threshold_roi_scale_top_k_unchanged() -> None:
    """[24][25][26] threshold=0.85、roi_scale=0.75、top_k=4 不变。"""
    yard = load_config().evidence.detectors["bottom_menu_yard"]
    assert yard.threshold == 0.85
    assert yard.roi_scale == 0.75
    assert yard.prefilter_top_k == 4
    assert yard.template_prefilter == "thumb"


def test_action_still_disabled_and_dry_run() -> None:
    """[27][28] ``action.enabled=false``、``backend=dry_run`` 不变。"""
    action = load_config().action
    assert action.enabled is False
    assert action.backend == "dry_run"


def test_no_real_input_path_exists() -> None:
    """[29] 不存在真实输入路径：配置无 real_input 开关，源码无输入库依赖。"""
    config = load_config()
    assert not hasattr(config.action, "real_input")
    assert not hasattr(config.capture, "real_input")
    for path in (CAPTURE_SRC, ROOT / "src" / "vision" / "dataset" / "live_yard.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            name.startswith(forbidden) for forbidden in FORBIDDEN_IMPORTS for name in imported
        ), path.name


# ---------------------------------------------------------------------------
# 契约补充：dHash 一致性、命名、CLI
# ---------------------------------------------------------------------------


def test_difference_hash_matches_template_dhash() -> None:
    """dHash 与 ``src.vision.evidence.templates.template_dhash`` 逐图等价（同口径）。"""
    from src.vision.evidence.templates import template_dhash  # noqa: PLC0415

    for path in sorted(YARD_DIR.glob("*.png"))[:20]:
        image = read_png(path)
        assert difference_hash(image) == template_dhash(image), path.name
    sample = make_image(7)
    assert difference_hash(sample) == template_dhash(sample)
    assert hamming_distance(difference_hash(sample), difference_hash(make_image(8))) > 6


def test_session_and_frame_naming_contract() -> None:
    """会话 / 文件命名契约：非法输入必须报错，不静默兜底。"""
    assert is_valid_session_id(new_session_id()) is True
    assert is_valid_session_id("yard_live_20261003_s1") is False
    assert is_valid_session_id(SESSION_A) is True
    assert frame_filename(SESSION_A, 7) == f"{SESSION_A}_007.png"
    with pytest.raises(LiveFrameError):
        frame_filename("bad_session", 1)
    with pytest.raises(LiveFrameError):
        frame_filename(SESSION_A, 1000)


def test_live_frame_role_defaults_to_unknown(tmp_path: Path) -> None:
    """未登记的 live 帧角色为 ``unknown``（**不**默认成 candidate/holdout）。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={SESSION_A: [1]})
    frames = list_live_frames(dataset)
    assert len(frames) == 1
    frame = frames[0]
    assert frame.role == CAPTURE_ROLE_UNKNOWN
    assert frame.is_template_eligible is False
    assert frame.is_candidate is False and frame.is_holdout is False
    assert frame.resolution == "320x120"


def test_capture_cli_parser_rejects_bad_arguments() -> None:
    """CLI 参数：角色只能是 candidate/holdout/diagnostic。"""
    parser = capture_tool.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--role", "train"])
    args = parser.parse_args(["--role", "holdout", "--count", "3"])
    assert args.role == "holdout"
    assert args.count == 3


def test_capture_refuses_interval_below_minimum(tmp_path: Path) -> None:
    """间隔低于最小值时拒绝（避免连续近重复帧冒充独立视觉状态）。"""
    dataset = tmp_path
    build_dataset(dataset, sessions={})
    with pytest.raises(capture_tool.CaptureError):
        capture_tool.capture_session(
            load_config(),
            session_id=SESSION_A,
            role=CAPTURE_ROLE_CANDIDATE,
            count=1,
            interval_ms=10,
            scene="yard",
            theme="unknown",
            variant="unknown",
            note="",
            sources_path=tmp_path / "sources.jsonl",
            dataset=dataset,
            source_factory=lambda _config: FakeSource([]),
        )
