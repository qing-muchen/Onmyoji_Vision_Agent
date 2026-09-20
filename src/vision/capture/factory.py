"""采集后端工厂：按配置与可用性选择后端。

``capture.backend`` 取值语义：

* ``auto`` —— 按 :data:`BACKEND_PRIORITY` 依次尝试，失败自动回退；
* ``dxcam`` / ``mss`` —— 只使用指定后端，不可用时抛出 :class:`~src.core.exceptions.CaptureError`。

采集区域的确定顺序：

1. ``capture.region`` 显式配置（屏幕绝对坐标）；
2. 目标窗口客户区（推荐：窗口尺寸可变，必须动态获取）；
3. 两者都没有时视为配置错误。
"""

from __future__ import annotations

from src.core.config import CaptureConfig
from src.core.exceptions import CaptureError, ConfigError, OnmyojiVisionError
from src.core.geometry import CoordinateTransform, Point, Rect
from src.utils.logger import get_logger
from src.vision.capture.base import CaptureBackend
from src.vision.capture.dxcam_capture import (
    OUTPUT_INDEX_OFFSET,
    DxcamCapture,
)
from src.vision.capture.dxcam_capture import is_available as is_dxcam_available
from src.vision.capture.monitors import (
    find_monitor_index,
    get_monitor_rect,
    get_window_monitor_rect,
)
from src.vision.capture.mss_capture import MssCapture
from src.vision.capture.mss_capture import is_available as is_mss_available
from src.vision.win32 import IS_WINDOWS
from src.vision.window.models import WindowInfo

__all__ = [
    "BACKEND_PRIORITY",
    "candidate_backends",
    "create_capture_backend",
    "resolve_capture_region",
]

#: ``auto`` 模式下的后端优先级（Windows 下 dxcam 性能更优）
BACKEND_PRIORITY: tuple[str, ...] = ("dxcam", "mss")

logger = get_logger("vision.capture.factory")


def candidate_backends(backend: str) -> tuple[str, ...]:
    """返回按优先级排列的候选后端名称。

    Args:
        backend: 配置中的 ``capture.backend`` 取值。

    Returns:
        ``auto`` 返回完整优先级序列；指定后端返回仅含该后端的单元素元组。

    Raises:
        ConfigError: 后端名称不受支持。
    """
    if backend == "auto":
        return BACKEND_PRIORITY
    if backend in BACKEND_PRIORITY:
        return (backend,)
    raise ConfigError(f"不支持的采集后端: {backend}（可选: auto, {', '.join(BACKEND_PRIORITY)}）")


def resolve_capture_region(config: CaptureConfig, window: WindowInfo | None = None) -> Rect:
    """确定采集区域。

    Args:
        config: 采集配置。
        window: 目标窗口信息；``capture.region`` 未配置时使用其客户区。

    Returns:
        屏幕绝对坐标下的采集区域。

    Raises:
        ConfigError: 既未配置 ``capture.region`` 也没有目标窗口。
    """
    if config.region is not None:
        left, top, width, height = config.region
        return Rect(x=left, y=top, width=width, height=height)
    if window is not None:
        return window.rect
    raise ConfigError("未配置 capture.region 且未提供目标窗口，无法确定采集区域")


def create_capture_backend(
    config: CaptureConfig,
    *,
    window: WindowInfo | None = None,
    transform: CoordinateTransform | None = None,
) -> CaptureBackend:
    """按配置创建采集后端。

    返回的后端**尚未启动**，调用方需显式 ``start()`` 或使用 ``with`` 语句。

    Args:
        config: 采集配置（后端、帧率、显示器、区域）。
        window: 目标窗口信息，用于推导采集区域与所在显示器。
        transform: 随每帧携带的坐标映射（通常来自校准结果）。

    Returns:
        可用的采集后端。

    Raises:
        CaptureError: 所有候选后端都不可用。
    """
    region = resolve_capture_region(config, window)
    failures: list[str] = []
    for name in candidate_backends(config.backend):
        try:
            if name == "dxcam":
                return _build_dxcam(config=config, region=region, window=window, transform=transform)
            return _build_mss(config=config, region=region, transform=transform)
        except CaptureError as exc:
            logger.warning("%s 后端不可用: %s", name, exc)
            failures.append(f"{name}: {exc}")
    raise CaptureError("所有候选采集后端均不可用 -> " + "; ".join(failures))


def _build_dxcam(
    *,
    config: CaptureConfig,
    region: Rect,
    window: WindowInfo | None,
    transform: CoordinateTransform | None,
) -> CaptureBackend:
    """构造 dxcam 后端，并将屏幕绝对坐标换算到显示器局部坐标。

    Raises:
        CaptureError: dxcam 不可用、无法确定显示器，或采集区域不在该显示器内。
    """
    if not is_dxcam_available():
        raise CaptureError("dxcam 不可用（未安装或当前平台不支持）")
    try:
        monitor_rect = _resolve_monitor_rect(config, window)
        output_index = find_monitor_index(monitor_rect) - OUTPUT_INDEX_OFFSET
        return DxcamCapture(
            region=region,
            fps=config.fps,
            origin=Point(x=monitor_rect.x, y=monitor_rect.y),
            output_index=max(0, output_index),
            coordinate_transform=transform,
        )
    except OnmyojiVisionError as exc:
        raise CaptureError(str(exc)) from exc


def _build_mss(
    *,
    config: CaptureConfig,
    region: Rect,
    transform: CoordinateTransform | None,
) -> CaptureBackend:
    """构造 mss 后端（使用屏幕绝对坐标，无需显示器换算）。

    Raises:
        CaptureError: mss 不可用。
    """
    if not is_mss_available():
        raise CaptureError("mss 不可用（未安装）")
    return MssCapture(region=region, fps=config.fps, coordinate_transform=transform)


def _resolve_monitor_rect(config: CaptureConfig, window: WindowInfo | None) -> Rect:
    """确定采集依据的显示器矩形。

    窗口模式下跟随窗口所在显示器（多显示器时更符合直觉）；整屏回退时使用
    ``capture.monitor_index`` 指定的显示器。
    """
    if window is not None and window.is_real_window and IS_WINDOWS:
        return get_window_monitor_rect(window.hwnd)
    return get_monitor_rect(config.monitor_index)
