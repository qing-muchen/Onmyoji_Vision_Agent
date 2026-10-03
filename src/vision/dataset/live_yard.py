"""当前版本庭院实时采集（Phase 11.9-A）：``raw/yard_live`` 布局、按会话分流角色、跨集合泄漏审计。

目录契约（**显式**，不依赖任何递归 glob 的巧合隔离）::

    dataset/raw/yard_live/<capture_session_id>/yard_live_<capture_session_id>_<NNN>.png

三条硬约束：

1. 历史 baseline ``dataset/raw/yard/*.png`` **严格保持 98 张不动**，新数据一律进
   ``raw/yard_live/``；
2. 既有 benchmark（``tools/benchmark_template_prefilter.py`` 的 ``collect_positives``）只做
   ``raw/yard`` 的**非递归** ``glob("*.png")``，因此天然看不到 ``yard_live``；本模块反过来
   **显式**只扫 ``raw/yard_live/<合法 session>/``，两边都不靠"恰好扫不到"来隔离；
3. 同一张图及其近重复帧**不得**同时出现在 candidate 与 holdout：角色由
   ``CaptureSource.capture_role``（source-level）声明，且**按会话**强制一致。

依赖边界：只 import 标准库 + ``src.core`` / ``src.utils`` + 本包 ``sources``。
dHash 在此**自实现**（语义与 ``src.vision.evidence.templates.template_dhash`` 完全一致，
由测试逐图断言），避免 dataset 包反向依赖 evidence 包。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.vision.dataset.sources import (
    CAPTURE_ROLE_CANDIDATE,
    CAPTURE_ROLE_DIAGNOSTIC,
    CAPTURE_ROLE_HOLDOUT,
    CAPTURE_ROLE_UNKNOWN,
    CAPTURE_ROLES,
    TEMPLATE_ELIGIBLE_ROLES,
    CaptureSource,
    SourceError,
    SourceRegistry,
    compute_sha256,
)

__all__ = [
    "FRAME_NAME_PATTERN",
    "LIVE_ROOT",
    "LiveFrameError",
    "LiveFrame",
    "MIN_CAPTURE_INTERVAL_MS",
    "NEAR_DUPLICATE_THRESHOLD",
    "NearDuplicate",
    "SESSION_ID_PATTERN",
    "assert_within_live_root",
    "audit_live_dataset",
    "difference_hash",
    "frame_filename",
    "hamming_distance",
    "is_valid_session_id",
    "leakage_report",
    "list_live_frames",
    "list_live_sessions",
    "live_root",
    "near_duplicate_pairs",
    "new_session_id",
    "parse_frame_index",
    "partition_by_role",
    "read_image",
    "read_live_frame",
    "session_directory",
    "session_id_for",
    "session_role_conflicts",
    "session_roles",
]

logger = get_logger("vision.dataset.live_yard")

#: 新采集数据的独立根目录（相对 ``dataset/``）—— 历史 ``raw/yard`` 完全不受影响
LIVE_ROOT = "raw/yard_live"
#: 会话目录 / 会话 ID 形态：``yard_live_<YYYYMMDD>_s<NN>``
SESSION_ID_PATTERN = re.compile(r"^yard_live_(?P<date>\d{8})_s(?P<seq>\d{2})$")
#: 帧文件名（**stem**）形态：``yard_live_<YYYYMMDD>_s<NN>_<NNN>``
FRAME_NAME_PATTERN = re.compile(
    r"^(?P<session>yard_live_\d{8}_s\d{2})_(?P<index>\d{3})$"
)
#: 仅接受 PNG（避免"重新编码的 jpg 冒充 raw"）
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png"})
#: 近重复判定的最大汉明距离 —— 与 ``tools/audit_yard_dataset.NEAR_DUPLICATE_THRESHOLD``
#: 及 ``positive_sources.json`` 的 ``dedupe_threshold`` 保持同一口径
NEAR_DUPLICATE_THRESHOLD = 6
#: 连续采样时的最小间隔（毫秒）：低于此值几乎必然产生近重复帧，没有独立视觉价值
MIN_CAPTURE_INTERVAL_MS = 300
#: dHash 网格边长（9x8 差分 ⇒ 8x8 bit ⇒ 16 个十六进制字符）
DHASH_SIZE = 8


class LiveFrameError(ValueError):
    """``raw/yard_live`` 非法（越界路径 / 会话名不合法 / 角色冲突）。"""


# ---------------------------------------------------------------------------
# 图像与哈希（与模板去重同口径）
# ---------------------------------------------------------------------------


def read_image(path: Path | str, *, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读图（兼容非 ASCII 路径：``np.fromfile`` + ``cv2.imdecode``）。

    Args:
        path: 图片路径。
        flags: ``cv2`` 读取标志。

    Returns:
        图像数组；读取失败返回 ``None``。
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, flags)
    except Exception:  # noqa: BLE001 - 读取失败不应中断审计
        return None


def difference_hash(image: np.ndarray, *, size: int = DHASH_SIZE) -> str:
    """dHash（差分感知哈希）十六进制串。

    与 ``src.vision.evidence.templates.template_dhash`` **逐位等价**（同 ``INTER_AREA``、
    同 9x8 差分、同 ``packbits``），因此"近重复"判定与"模板去重"使用同一口径。

    Args:
        image: BGR 或灰度图。
        size: 网格边长。

    Returns:
        十六进制哈希串（``size`` 为 8 时长度 16）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return np.packbits(diff.flatten()).tobytes().hex()


def hamming_distance(first: str, second: str) -> int:
    """两个十六进制哈希串的按位汉明距离。

    Args:
        first: 十六进制哈希。
        second: 十六进制哈希。

    Returns:
        不同位的数量。
    """
    left = bin(int(first, 16))[2:].zfill(len(first) * 4)
    right = bin(int(second, 16))[2:].zfill(len(second) * 4)
    return sum(1 for a, b in zip(left, right, strict=False) if a != b)


# ---------------------------------------------------------------------------
# 目录 / 命名契约
# ---------------------------------------------------------------------------


def live_root(dataset: Path | str) -> Path:
    """``raw/yard_live`` 绝对路径。"""
    return Path(dataset) / LIVE_ROOT


def is_valid_session_id(text: str) -> bool:
    """会话 ID 是否是合法形态（``yard_live_<YYYYMMDD>_s<NN>``）。"""
    return bool(SESSION_ID_PATTERN.match(str(text or "").strip()))


def session_id_for(*, date_text: str, sequence: int = 1) -> str:
    """按日期与序号构造会话 ID。

    Args:
        date_text: ``YYYYMMDD``。
        sequence: 当日会话序号（``1`` ⇒ ``s01``）。

    Returns:
        会话 ID。

    Raises:
        LiveFrameError: 日期形态非法或序号越界。
    """
    text = str(date_text or "").strip()
    if not re.fullmatch(r"\d{8}", text):
        raise LiveFrameError(f"日期必须是 YYYYMMDD: {date_text!r}")
    number = int(sequence)
    if not 1 <= number <= 99:
        raise LiveFrameError(f"会话序号必须在 1..99: {sequence!r}")
    return f"yard_live_{text}_s{number:02d}"


def new_session_id(*, now: datetime | None = None, sequence: int = 1) -> str:
    """按本地日期生成新会话 ID。"""
    stamp = now or datetime.now().astimezone()
    return session_id_for(date_text=stamp.strftime("%Y%m%d"), sequence=sequence)


def frame_filename(session_id: str, index: int) -> str:
    """帧文件名（``<session>_<NNN>.png``）。

    Raises:
        LiveFrameError: 会话 ID 非法或序号越界。
    """
    if not is_valid_session_id(session_id):
        raise LiveFrameError(f"会话 ID 非法: {session_id!r}")
    number = int(index)
    if not 1 <= number <= 999:
        raise LiveFrameError(f"帧序号必须在 1..999: {index!r}")
    return f"{session_id}_{number:03d}.png"


def parse_frame_index(stem: str) -> int | None:
    """解析帧序号（传 **stem**，即不含 ``.png``；不匹配返回 ``None``）。"""
    matched = FRAME_NAME_PATTERN.match(str(stem or ""))
    return int(matched.group("index")) if matched else None


def session_directory(dataset: Path | str, session_id: str) -> Path:
    """会话目录 ``dataset/raw/yard_live/<session_id>/``。

    Raises:
        LiveFrameError: 会话 ID 非法。
    """
    if not is_valid_session_id(session_id):
        raise LiveFrameError(f"会话 ID 非法: {session_id!r}")
    return live_root(dataset) / session_id


def assert_within_live_root(dataset: Path | str, path: Path | str) -> Path:
    """断言路径落在 ``raw/yard_live/`` 内（防越界写入）。

    Args:
        dataset: 数据根目录。
        path: 待检查路径。

    Returns:
        解析后的绝对路径。

    Raises:
        LiveFrameError: 越界（含 ``..`` 逃逸）。
    """
    root = live_root(dataset).resolve()
    target = Path(path)
    resolved = target.resolve()
    if resolved != root and root not in resolved.parents:
        raise LiveFrameError(f"路径越界（必须位于 {LIVE_ROOT}/）: {path}")
    return resolved


# ---------------------------------------------------------------------------
# 帧记录与显式扫描
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiveFrame:
    """``raw/yard_live`` 中的一张原图（只读视图）。

    Attributes:
        relative_path: 相对 ``dataset/`` 的 POSIX 路径。
        session_id: 所属采集会话。
        index: 会话内序号。
        role: 采集角色（``candidate`` / ``holdout`` / ``diagnostic`` / ``unknown``）。
        source_id: ``src_<sha256 前 16>``。
        sha256: 原图 SHA256。
        width / height: 客户区像素尺寸（**不 resize、不裁 ROI**）。
        dhash: dHash 十六进制串。
    """

    relative_path: str
    session_id: str
    index: int
    role: str
    source_id: str
    sha256: str
    width: int
    height: int
    dhash: str

    @property
    def is_candidate(self) -> bool:
        """是否 candidate 池。"""
        return self.role == CAPTURE_ROLE_CANDIDATE

    @property
    def is_holdout(self) -> bool:
        """是否 holdout 池。"""
        return self.role == CAPTURE_ROLE_HOLDOUT

    @property
    def is_template_eligible(self) -> bool:
        """是否可用于模板生成 / 选择（仅 ``candidate``）。"""
        return self.role in TEMPLATE_ELIGIBLE_ROLES

    @property
    def resolution(self) -> str:
        """``1920x1080`` 形态的分辨率串。"""
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "relative_path": self.relative_path,
            "session_id": self.session_id,
            "index": self.index,
            "role": self.role,
            "source_id": self.source_id,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "dhash": self.dhash,
        }


def session_roles(sources: Iterable[CaptureSource]) -> dict[str, str]:
    """从来源记录汇总 ``session_id → role``（同一会话多角色时取**首个冲突前**的值）。

    Args:
        sources: 来源记录。

    Returns:
        ``session_id → capture_role``。
    """
    mapping: dict[str, str] = {}
    for item in sources:
        if not item.session_id:
            continue
        mapping.setdefault(item.session_id, item.capture_role)
    return mapping


def session_role_conflicts(sources: Iterable[CaptureSource]) -> tuple[str, ...]:
    """找出"同一会话被声明为多个角色"的会话（应为空）。

    Args:
        sources: 来源记录。

    Returns:
        冲突描述（已排序）。
    """
    seen: dict[str, set[str]] = {}
    for item in sources:
        if not item.session_id:
            continue
        seen.setdefault(item.session_id, set()).add(item.capture_role)
    return tuple(
        sorted(f"{session}: {sorted(roles)}" for session, roles in seen.items() if len(roles) > 1)
    )


def list_live_sessions(dataset: Path | str) -> tuple[str, ...]:
    """列出 ``raw/yard_live`` 下的**合法**会话目录名（显式，不递归）。

    Args:
        dataset: 数据根目录。

    Returns:
        会话 ID（已排序）。
    """
    root = live_root(dataset)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(entry.name for entry in root.iterdir() if entry.is_dir() and is_valid_session_id(entry.name))
    )


def _frame_files(dataset: Path | str, session_id: str) -> list[Path]:
    """会话目录下的帧文件（**显式**只认 ``<session>_<NNN>.png``，按序号排序）。"""
    directory = session_directory(dataset, session_id)
    if not directory.is_dir():
        return []
    files: list[Path] = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        matched = FRAME_NAME_PATTERN.match(path.stem)
        if matched is None or matched.group("session") != session_id:
            logger.warning("yard_live 中出现不符合命名契约的文件（已忽略）: %s", path.name)
            continue
        files.append(path)
    return sorted(files, key=lambda item: parse_frame_index(item.stem) or 0)


def read_live_frame(
    dataset: Path | str,
    path: Path | str,
    *,
    registry: SourceRegistry | None = None,
) -> LiveFrame:
    """把一张 ``yard_live`` 原图读成 :class:`LiveFrame`。

    角色的**权威来源**是 ``sources.jsonl``（``registry``）；未登记时按 ``unknown`` 处理
    （**不**默认成 candidate/holdout）。

    Args:
        dataset: 数据根目录。
        path: 帧路径（必须位于 ``raw/yard_live/`` 内）。
        registry: 来源登记表（可选）。

    Returns:
        :class:`LiveFrame`。

    Raises:
        LiveFrameError: 路径越界、命名不符合契约或图片不可读。
    """
    resolved = assert_within_live_root(dataset, path)
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise LiveFrameError(f"帧文件必须是 PNG: {resolved.name}")
    matched = FRAME_NAME_PATTERN.match(resolved.stem)
    if matched is None:
        raise LiveFrameError(f"帧文件名不符合契约: {resolved.name}")
    session_id = matched.group("session")
    relative = resolved.relative_to(Path(dataset).resolve()).as_posix()
    image = read_image(resolved)
    if image is None:
        raise LiveFrameError(f"图片不可读: {relative}")
    height, width = image.shape[:2]
    digest = compute_sha256(resolved)
    record = registry.by_sha256(digest) if registry is not None else None
    if record is None and registry is not None:
        record = registry.by_path(relative)
    role = record.capture_role if record is not None else CAPTURE_ROLE_UNKNOWN
    return LiveFrame(
        relative_path=relative,
        session_id=session_id,
        index=parse_frame_index(resolved.stem) or 0,
        role=role,
        source_id=f"src_{digest[:16]}",
        sha256=digest,
        width=int(width),
        height=int(height),
        dhash=difference_hash(image),
    )


def list_live_frames(
    dataset: Path | str,
    *,
    registry: SourceRegistry | None = None,
) -> tuple[LiveFrame, ...]:
    """**显式**扫描 ``raw/yard_live/<合法 session>/*.png``。

    刻意不使用 ``rglob`` —— 目录层级由 :func:`list_live_sessions` 与
    :func:`_frame_files` 明确限定，避免"靠 glob 恰好扫不到"来隔离数据。

    Args:
        dataset: 数据根目录。
        registry: 来源登记表（提供角色）。

    Returns:
        :class:`LiveFrame` 元组（按会话、序号排序）。
    """
    frames: list[LiveFrame] = []
    for session_id in list_live_sessions(dataset):
        for path in _frame_files(dataset, session_id):
            frames.append(read_live_frame(dataset, path, registry=registry))
    return tuple(frames)


def partition_by_role(
    frames: Iterable[LiveFrame],
) -> dict[str, tuple[LiveFrame, ...]]:
    """按角色分流（**按会话整体**，不逐图随机切分）。

    Args:
        frames: 帧记录。

    Returns:
        ``role → frames``（四种角色键恒存在，可能为空）。
    """
    buckets: dict[str, list[LiveFrame]] = {role: [] for role in sorted(CAPTURE_ROLES)}
    for frame in frames:
        buckets.setdefault(frame.role, []).append(frame)
    return {role: tuple(items) for role, items in buckets.items()}


# ---------------------------------------------------------------------------
# 近重复与跨集合泄漏
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NearDuplicate:
    """两张近重复帧（dHash 汉明距离 ``<=`` 阈值）。

    Attributes:
        distance: 汉明距离。
        left / right: 相对路径。
        left_role / right_role: 各自角色。
    """

    distance: int
    left: str
    right: str
    left_role: str
    right_role: str

    @property
    def cross_collection(self) -> bool:
        """是否跨 candidate / holdout（**数据泄漏风险**）。"""
        roles = {self.left_role, self.right_role}
        return roles == {CAPTURE_ROLE_CANDIDATE, CAPTURE_ROLE_HOLDOUT}

    @property
    def within_session(self) -> bool:
        """两个文件是否同属一个会话（同会话高度相关，属预期）。"""
        return self.left.rsplit("/", 1)[:-1] == self.right.rsplit("/", 1)[:-1]

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典。"""
        return {
            "distance": self.distance,
            "left": self.left,
            "right": self.right,
            "left_role": self.left_role,
            "right_role": self.right_role,
            "cross_collection": self.cross_collection,
        }


def near_duplicate_pairs(
    frames: Iterable[LiveFrame],
    *,
    threshold: int = NEAR_DUPLICATE_THRESHOLD,
) -> tuple[NearDuplicate, ...]:
    """找出全部近重复帧对（含会话内、跨会话、跨集合）。

    Args:
        frames: 帧记录。
        threshold: 最大汉明距离。

    Returns:
        :class:`NearDuplicate` 元组（按距离、路径排序）。
    """
    items = sorted(frames, key=lambda frame: frame.relative_path)
    pairs: list[NearDuplicate] = []
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            distance = hamming_distance(first.dhash, second.dhash)
            if distance <= threshold:
                pairs.append(
                    NearDuplicate(
                        distance=distance,
                        left=first.relative_path,
                        right=second.relative_path,
                        left_role=first.role,
                        right_role=second.role,
                    )
                )
    return tuple(sorted(pairs, key=lambda pair: (pair.distance, pair.left, pair.right)))


def leakage_report(
    frames: Iterable[LiveFrame],
    *,
    threshold: int = NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, Any]:
    """candidate ↔ holdout 近重复泄漏报告（**自匹配**防线的核心检查）。

    Args:
        frames: 帧记录。
        threshold: 最大汉明距离。

    Returns:
        报告字典（``pairs`` / ``within_candidate`` / ``within_holdout`` / ``cross`` 等）。
    """
    items = tuple(frames)
    pairs = near_duplicate_pairs(items, threshold=threshold)
    cross = tuple(pair for pair in pairs if pair.cross_collection)
    return {
        "threshold": int(threshold),
        "pairs_total": len(pairs),
        "within_candidate": sum(1 for pair in pairs if pair.left_role == pair.right_role == CAPTURE_ROLE_CANDIDATE),
        "within_holdout": sum(1 for pair in pairs if pair.left_role == pair.right_role == CAPTURE_ROLE_HOLDOUT),
        "cross_collection": len(cross),
        "cross_pairs": [pair.to_dict() for pair in cross],
        "possible_leakage": bool(cross),
        "pairs": [pair.to_dict() for pair in pairs],
    }


# ---------------------------------------------------------------------------
# 汇总审计
# ---------------------------------------------------------------------------


def _resolution_counts(frames: Iterable[LiveFrame]) -> dict[str, int]:
    """分辨率分布。"""
    counts: dict[str, int] = {}
    for frame in frames:
        counts[frame.resolution] = counts.get(frame.resolution, 0) + 1
    return dict(sorted(counts.items()))


def audit_live_dataset(
    dataset: Path | str,
    *,
    registry: SourceRegistry | None = None,
) -> dict[str, Any]:
    """审计 ``raw/yard_live``：会话、角色、SHA、dHash、泄漏与"是否足以进入下一阶段"。

    Args:
        dataset: 数据根目录。
        registry: 来源登记表（角色真相）。

    Returns:
        报告字典，含 ``status``（``empty`` / ``role_conflict`` / ``leakage_detected`` /
        ``insufficient_sessions`` / ``ok``）与 ``reasons``。
    """
    frames = list_live_frames(dataset, registry=registry)
    roles = session_roles(registry) if registry is not None else {}
    conflicts = session_role_conflicts(registry) if registry is not None else ()

    digests: dict[str, list[str]] = {}
    for frame in frames:
        digests.setdefault(frame.sha256, []).append(frame.relative_path)
    exact_duplicates = {
        digest: sorted(paths) for digest, paths in digests.items() if len(paths) > 1
    }

    by_role = partition_by_role(frames)
    role_counts = {role: len(items) for role, items in sorted(by_role.items())}
    sessions: list[dict[str, Any]] = []
    for session_id in list_live_sessions(dataset):
        members = [frame for frame in frames if frame.session_id == session_id]
        sessions.append(
            {
                "session_id": session_id,
                "role": members[0].role if members else roles.get(session_id, CAPTURE_ROLE_UNKNOWN),
                "frames": len(members),
                "resolutions": _resolution_counts(members),
                "unique_sha256": len({frame.sha256 for frame in members}),
            }
        )

    leak = leakage_report(frames)
    candidate_sessions = sorted({frame.session_id for frame in by_role[CAPTURE_ROLE_CANDIDATE]})
    holdout_sessions = sorted({frame.session_id for frame in by_role[CAPTURE_ROLE_HOLDOUT]})

    reasons: list[str] = []
    if not frames:
        status = "empty"
        reasons.append("raw/yard_live 尚无样本（等待人工采集）")
    elif conflicts:
        status = "role_conflict"
        reasons.append(f"同一会话出现多个角色: {list(conflicts)}")
    elif leak["possible_leakage"]:
        status = "leakage_detected"
        reasons.append(
            f"candidate ↔ holdout 存在 {leak['cross_collection']} 对近重复（阈值 {leak['threshold']}）"
        )
    elif len(sessions) < 2 or not candidate_sessions or not holdout_sessions:
        status = "insufficient_sessions"
        reasons.append(
            f"需要 >= 2 个独立会话且 candidate/holdout 各自非空"
            f"（当前会话 {len(sessions)}、candidate {len(candidate_sessions)}、holdout {len(holdout_sessions)}）"
        )
    else:
        status = "ok"

    return {
        "root": LIVE_ROOT,
        "status": status,
        "reasons": reasons,
        "session_count": len(sessions),
        "sessions": sessions,
        "frame_count": len(frames),
        "unique_sha256": len(digests),
        "exact_duplicate_groups": dict(sorted(exact_duplicates.items())),
        "by_role": role_counts,
        "candidate_sessions": candidate_sessions,
        "holdout_sessions": holdout_sessions,
        "resolutions": _resolution_counts(frames),
        "near_duplicate_threshold": leak["threshold"],
        "near_duplicate_pairs_total": leak["pairs_total"],
        "near_duplicate_within_candidate": leak["within_candidate"],
        "near_duplicate_within_holdout": leak["within_holdout"],
        "cross_collection_pairs": leak["cross_collection"],
        "possible_leakage": leak["possible_leakage"],
        "leakage": leak,
        "frames": [frame.to_dict() for frame in frames],
    }


def require_inside_live_root(dataset: Path | str, path: Path | str) -> Path:
    """:func:`assert_within_live_root` 的别名（便于调用方语义化读取）。

    Raises:
        LiveFrameError: 越界。
        SourceError: 路径为空。
    """
    if not str(path or "").strip():
        raise SourceError("路径为空")
    return assert_within_live_root(dataset, path)


def role_of_session(registry: SourceRegistry | None, session_id: str) -> str:
    """某会话的角色（未登记 ⇒ ``unknown``）。

    Args:
        registry: 来源登记表。
        session_id: 会话 ID。

    Returns:
        角色字符串。
    """
    if registry is None:
        return CAPTURE_ROLE_UNKNOWN
    for item in registry:
        if item.session_id == session_id:
            return item.capture_role
    return CAPTURE_ROLE_UNKNOWN


#: 供外部按"角色 → 语义"展示（诊断用，不含未知）
ROLE_LABELS: dict[str, str] = {
    CAPTURE_ROLE_CANDIDATE: "候选（可用于模板生成/选择）",
    CAPTURE_ROLE_HOLDOUT: "留出（禁止参与模板生成/选择/调参）",
    CAPTURE_ROLE_DIAGNOSTIC: "诊断（只定位问题，不选模板）",
    CAPTURE_ROLE_UNKNOWN: "未知（历史素材或未标注）",
}
