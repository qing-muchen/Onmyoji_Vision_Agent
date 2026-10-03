"""Phase 11.9-A：当前版本庭院 raw 采集工具（**只截图，不做任何输入控制**）。

用途：由**用户手动**把游戏切到庭院后，按"独立视觉状态"逐张采集原客户区像素，写入
``dataset/raw/yard_live/<capture_session_id>/``，并登记 source-level provenance。

```bash
# 采集前先看能不能采（不写任何文件）
python tools/capture_yard_dataset.py --role candidate --dry-run

# 采一张静态庭院（candidate 会话，自动生成 yard_live_<YYYYMMDD>_s01）
python tools/capture_yard_dataset.py --role candidate --count 1

# 采集 holdout 会话（必须**另起**一次进入庭院后的连续采集）
python tools/capture_yard_dataset.py --role holdout --session-id yard_live_20261003_s02 --count 5
```

安全边界（硬约束，AST 用例固化）：

* 只捕获**已存在**的游戏窗口；窗口不存在 / 最小化 / **非前台** ⇒ 明确失败并给出原因码，
  **绝不激活窗口、绝不抢占前台、绝不自动导航**；
* 不 import 任何输入控制库（``pyautogui`` / ``pynput`` / ``keyboard`` / ``mouse``），
  不调用 ``SetForegroundWindow`` / ``ShowWindow`` / ``BringWindowToTop``；
* 保存**原客户区像素**：不 resize、不裁 ROI、不做图像增强（避免"加工后的图冒充 raw"）；
* 原子写入（``.png.tmp`` → ``os.replace``）：失败**清理半文件**；
* 重复 SHA256 拒绝、已有文件不覆盖、路径严格限制在 ``dataset/raw/yard_live/`` 内；
* **不**自动写入 ``dataset/features/bottom_menu/yard/positive``（生产模板目录）——
  模板重选属后续阶段。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import AppConfig, load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.vision.capture.models import Frame  # noqa: E402
from src.vision.dataset.live_yard import (  # noqa: E402
    LIVE_ROOT,
    MIN_CAPTURE_INTERVAL_MS,
    assert_within_live_root,
    frame_filename,
    is_valid_session_id,
    new_session_id,
    parse_frame_index,
    session_directory,
)
from src.vision.dataset.sources import (  # noqa: E402
    CAPTURE_ROLE_UNKNOWN,
    CaptureSource,
    SourceRegistry,
    compute_sha256,
    local_now_iso,
    normalize_capture_role,
)
from src.runtime.state_monitor.frame_source import WindowFrameSource  # noqa: E402

logger = get_logger("tools.capture_yard_dataset")

DATASET = ROOT / "dataset"
DEFAULT_SOURCES = DATASET / "annotations" / "sources.jsonl"
DEFAULT_INTERVAL_MS = 800
#: 允许的采集角色（**不允许** ``unknown``：未知角色等同于未标注，不能用于候选/留出）
ALLOWED_ROLES: tuple[str, ...] = ("candidate", "holdout", "diagnostic")

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_WINDOW_UNAVAILABLE = 3
EXIT_NOT_FOREGROUND = 4
EXIT_SAVE_FAILED = 5


class CaptureError(RuntimeError):
    """采集过程不可继续（窗口 / 前台 / 落盘问题）。"""


class AbortCapture(CaptureError):
    """带退出码的中止。"""

    def __init__(self, code: int, status: str, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.code = int(code)
        self.status = str(status)
        self.message = str(message)


def sha256_bytes(payload: bytes) -> str:
    """字节内容 SHA256（与 :func:`compute_sha256` 同算法，无需二次读盘）。"""
    return hashlib.sha256(payload).hexdigest()


def encode_png(image: np.ndarray) -> bytes:
    """把 BGR 客户区图像编码为 PNG 字节（**无损、不做任何变换**）。

    Args:
        image: 客户区原始像素。

    Returns:
        PNG 字节。

    Raises:
        CaptureError: 编码失败。
    """
    ok, buffer = cv2.imencode(".png", image)
    if not ok or buffer is None:
        raise CaptureError("PNG 编码失败")
    return buffer.tobytes()


def write_atomic(path: Path, payload: bytes) -> None:
    """原子写入：先写 ``<name>.tmp``，成功后再 ``replace``；失败清理半文件。

    Args:
        path: 目标文件（必须尚不存在）。
        payload: 字节内容。

    Raises:
        CaptureError: 目标已存在或写入/替换失败。
    """
    if path.exists():
        raise CaptureError(f"目标文件已存在（不覆盖）: {path.name}")
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(payload)
        if tmp.stat().st_size != len(payload):
            raise CaptureError("写入字节数与预期不一致")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def next_index(directory: Path, session_id: str) -> int:
    """会话内下一个可用序号（从 1 开始，跳过已占用）。

    Args:
        directory: 会话目录。
        session_id: 会话 ID。

    Returns:
        可用序号。
    """
    used: set[int] = set()
    if directory.is_dir():
        for entry in directory.iterdir():
            if entry.is_file() and entry.suffix.lower() == ".png" and entry.stem.startswith(session_id):
                number = parse_frame_index(entry.stem)
                if number is not None:
                    used.add(number)
    index = 1
    while index in used:
        index += 1
    return index


def known_digests(registry: SourceRegistry, directory: Path) -> set[str]:
    """已知 SHA256 集合（来源登记表 + 会话目录内既有文件）。

    Args:
        registry: 来源登记表。
        directory: 会话目录。

    Returns:
        小写十六进制摘要集合。
    """
    digests = {item.source_sha256.lower() for item in registry if item.source_sha256}
    if directory.is_dir():
        for entry in directory.iterdir():
            if entry.is_file() and entry.suffix.lower() == ".png":
                digests.add(compute_sha256(entry).lower())
    return digests


def frame_geometry(frame: Frame, source: WindowFrameSource) -> dict[str, Any]:
    """从帧与窗口绑定提取几何元数据（拿不到就记 ``0``，**不猜**）。

    Args:
        frame: 采集到的帧。
        source: 帧来源。

    Returns:
        ``window_*`` / ``client_*`` / ``display_scale`` 字段字典。
    """
    window = source.window
    transform = getattr(frame, "coordinate_transform", None)
    scale = 0.0
    for attr in ("scale", "scale_x"):
        candidate = getattr(transform, attr, None)
        if isinstance(candidate, (int, float)) and float(candidate) > 0:
            scale = float(candidate)
            break
    return {
        "window_width": int(getattr(window, "width", 0) or 0),
        "window_height": int(getattr(window, "height", 0) or 0),
        "client_width": int(getattr(frame, "width", 0) or 0),
        "client_height": int(getattr(frame, "height", 0) or 0),
        "display_scale": scale,
    }


def open_source(config: AppConfig) -> WindowFrameSource:
    """装配真实窗口帧来源（**绝不**允许置前、启用前台门控）。"""
    return WindowFrameSource.from_config(config, allow_activation=False, guard_enabled=True)


def probe(source: WindowFrameSource) -> tuple[Frame | None, str, str]:
    """取一帧并给出结论。

    Args:
        source: 帧来源（需先 ``start()``）。

    Returns:
        ``(frame, status, message)``；``frame`` 为 ``None`` 时 ``status`` 为原因码。
    """
    frame = source.get_frame()
    status = source.status
    reason = status.reason.value if status is not None and status.reason is not None else "ok"
    message = status.message if status is not None else ""
    if frame is None:
        return None, reason, message
    if status is not None and status.foreground is False:
        return None, "not_foreground", "前台门控拒绝本轮（无法确认画面在前台）"
    return frame, "ok", message


def capture_session(
    config: AppConfig,
    *,
    session_id: str,
    role: str,
    count: int,
    interval_ms: int,
    scene: str,
    theme: str,
    variant: str,
    note: str,
    sources_path: Path,
    dataset: Path = DATASET,
    dry_run: bool = False,
    sleeper: Any = time.sleep,
    source_factory: Any = None,
) -> dict[str, Any]:
    """执行一个采集会话（**不激活窗口、不发送任何输入**）。

    Args:
        config: 应用配置。
        session_id: 采集会话 ID。
        role: 采集角色（``candidate`` / ``holdout`` / ``diagnostic``）。
        count: 本会话采集张数。
        interval_ms: 相邻两张的最小间隔（毫秒）。
        scene: ``game_scene`` 元数据。
        theme: ``theme_or_event`` 元数据（不确定写 ``unknown``）。
        variant: ``yard_variant`` 元数据（不确定写 ``unknown``）。
        note: 备注。
        sources_path: 来源登记表路径。
        dataset: 数据根目录。
        dry_run: ``True`` 时只探测窗口与状态，不落盘、不登记。
        sleeper: 间隔函数（测试可注入）。
        source_factory: 帧来源工厂（默认 :func:`open_source`；测试注入替身，**不改变**生产安全参数）。

    Returns:
        结果字典（``status`` / ``saved`` / ``duplicates`` / ``probe`` …）。

    Raises:
        AbortCapture: 采集不可继续（窗口不可用 / 非前台），携带退代码。
        CaptureError: 参数非法或落盘失败。
    """
    if not is_valid_session_id(session_id):
        raise CaptureError(f"会话 ID 非法（应为 yard_live_<YYYYMMDD>_s<NN>）: {session_id!r}")
    resolved_role = normalize_capture_role(role)
    if resolved_role not in ALLOWED_ROLES:
        raise CaptureError(f"采集角色必须是 {ALLOWED_ROLES}，收到: {role!r}")
    if interval_ms < MIN_CAPTURE_INTERVAL_MS:
        raise CaptureError(f"间隔过小（最小 {MIN_CAPTURE_INTERVAL_MS} ms）: {interval_ms}")

    directory = assert_within_live_root(dataset, session_directory(dataset, session_id))
    registry = SourceRegistry.load(sources_path)
    digests = known_digests(registry, directory)

    result: dict[str, Any] = {
        "session_id": session_id,
        "role": resolved_role,
        "requested": int(count),
        "saved": [],
        "duplicates": [],
        "geometry": {},
        "backend": None,
        "probe": {},
        "dry_run": bool(dry_run),
        "window_guard": "no_activation",
    }
    source = (source_factory or open_source)(config)
    source.start()
    try:
        for index in range(int(count)):
            if index > 0:
                sleeper(max(interval_ms, MIN_CAPTURE_INTERVAL_MS) / 1000.0)
            frame, status, message = probe(source)
            result["probe"] = {"index": index + 1, "status": status, "message": message}
            result["backend"] = source.backend_name
            if frame is None:
                if result["saved"]:
                    result["status"] = f"partial:{status}"
                    result["message"] = message
                    result["exit_code"] = EXIT_OK
                    return result
                code = EXIT_NOT_FOREGROUND if status == "not_foreground" else EXIT_WINDOW_UNAVAILABLE
                raise AbortCapture(code, status, message)

            payload = encode_png(frame.image)
            digest = sha256_bytes(payload)
            result["geometry"] = frame_geometry(frame, source)
            if digest.lower() in digests:
                result["duplicates"].append({"reason": "sha256", "digest": digest[:16]})
                logger.warning("重复 SHA256（拒绝保存）: %s", digest[:16])
                continue

            position = next_index(directory, session_id)
            filename = frame_filename(session_id, position)
            target = assert_within_live_root(dataset, directory / filename)
            relative = f"{LIVE_ROOT}/{session_id}/{filename}"
            if dry_run:
                result["saved"].append({"relative_path": relative, "dry_run": True})
                continue

            directory.mkdir(parents=True, exist_ok=True)
            write_atomic(target, payload)
            record = CaptureSource(
                source_id=f"src_{digest[:16]}",
                relative_path=relative,
                capture_id=f"cap_{session_id}_{position:03d}",
                session_id=session_id,
                captured_at=local_now_iso(),
                capture_backend=source.backend_name or CAPTURE_ROLE_UNKNOWN,
                game_scene=scene,
                source_sha256=digest,
                origin="live-capture",
                note=note,
                capture_role=resolved_role,
                theme_or_event=theme,
                yard_variant=variant,
                **result["geometry"],
            )
            registry.add(record)
            registry.save(sources_path)
            digests.add(digest.lower())
            result["saved"].append(record.to_dict())
    finally:
        source.stop()

    if not result["saved"]:
        result["status"] = "no_frame_saved"
        result["message"] = "本轮没有任何新帧被保存（全部重复或不可采）"
    else:
        result["status"] = "ok"
    result["exit_code"] = EXIT_OK
    return result


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="当前版本庭院 raw 采集（只截图；不激活窗口、不发送输入）",
    )
    parser.add_argument("--session-id", default="", help="采集会话 ID（缺省自动生成）")
    parser.add_argument("--role", required=True, choices=ALLOWED_ROLES, help="采集角色")
    parser.add_argument("--count", type=int, default=1, help="采集张数（默认 1）")
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help=f"相邻两张的间隔（最小 {MIN_CAPTURE_INTERVAL_MS}）",
    )
    parser.add_argument("--scene", default="yard", help="game_scene 元数据")
    parser.add_argument("--theme", default="unknown", help="主题/活动（不确定写 unknown）")
    parser.add_argument("--variant", default="unknown", help="庭院变体（不确定写 unknown）")
    parser.add_argument("--note", default="", help="备注")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES), help="来源登记表路径")
    parser.add_argument("--dataset", default=str(DATASET), help="数据根目录")
    parser.add_argument("--dry-run", action="store_true", help="只探测窗口与前台状态，不落盘")
    parser.add_argument("--quiet", action="store_true", help="输出紧凑 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。

    Returns:
        进程退出码（``0`` 成功 / ``2`` 用法 / ``3`` 窗口不可用 / ``4`` 非前台 / ``5`` 落盘失败）。
    """
    args = build_parser().parse_args(argv)
    session_id = args.session_id or new_session_id()
    try:
        config = load_config()
        report = capture_session(
            config,
            session_id=session_id,
            role=args.role,
            count=max(1, int(args.count)),
            interval_ms=int(args.interval_ms),
            scene=args.scene,
            theme=args.theme,
            variant=args.variant,
            note=args.note,
            sources_path=Path(args.sources),
            dataset=Path(args.dataset),
            dry_run=bool(args.dry_run),
        )
    except AbortCapture as abort:
        print(json.dumps({"status": abort.status, "message": abort.message}, ensure_ascii=False))
        return abort.code
    except CaptureError as exc:
        print(json.dumps({"status": "capture_error", "message": str(exc)}, ensure_ascii=False))
        return EXIT_SAVE_FAILED

    print(json.dumps(report, ensure_ascii=False, indent=None if args.quiet else 2))
    return int(report.get("exit_code", EXIT_OK))


if __name__ == "__main__":
    raise SystemExit(main())
