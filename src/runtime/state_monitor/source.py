"""帧来源抽象（Phase 3.1）。

监控循环只依赖 :class:`FrameSource` 协议（``get_frame() -> Frame | None``），因此：

* 生产环境用 :class:`CaptureFrameSource`（包装 :class:`CaptureBackend`）；
* 测试可以注入"固定帧序列"的替身，**不依赖真实屏幕**（循环确定性可测）。

**前台门控**（配置 ``runtime.state_monitor.check_foreground``）：屏幕区域采集（dxcam / mss）
只能拿到"屏幕上的像素"，目标窗口失去前台时截图会包含遮挡窗口，可能造成状态误判。
本模块提供 :func:`make_foreground_gate` —— **只检查、不抢占**：未在前台时本轮返回
``None``（监控侧计入 ``skipped``），不会把游戏窗口抢到前台打扰用户。
"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol, runtime_checkable

from src.utils.logger import get_logger
from src.vision.capture.base import CaptureBackend
from src.vision.capture.models import Frame
from src.vision.window import WindowInfo, is_foreground

__all__ = [
    "CaptureFrameSource",
    "ForegroundGate",
    "FrameSource",
    "make_foreground_gate",
]

logger = get_logger("runtime.state_monitor.source")


@runtime_checkable
class FrameSource(Protocol):
    """帧来源：返回一帧，或在"当前没有可用画面"时返回 ``None``。"""

    def get_frame(self) -> Frame | None:
        """取得一帧画面。

        Returns:
            帧对象；暂无可用画面（后端未产出 / 前台门控未通过）时为 ``None``。
        """
        ...


class ForegroundGate(Protocol):
    """前台门控：返回 ``True`` 表示当前可以采集。"""

    def __call__(self) -> bool:
        """判断是否允许采集。"""
        ...


def make_foreground_gate(window: WindowInfo) -> ForegroundGate:
    """构造"只检查、不抢占"的前台门控。

    Args:
        window: 目标窗口（应为其客户区做采集）。

    Returns:
        无参可调用对象：目标窗口在前台时返回 ``True``。
    """
    hwnd = window.hwnd

    def _gate() -> bool:
        """检查目标窗口是否仍在前台。"""
        return is_foreground(hwnd)

    return _gate


class CaptureFrameSource:
    """把采集后端包装成监控可用的帧来源。

    典型用法::

        with CaptureFrameSource(backend, gate=make_foreground_gate(window)) as source:
            frame = source.get_frame()

    Args:
        backend: 采集后端（由 :func:`~src.vision.capture.create_capture_backend` 创建）。
        gate: 可选的采集门控；返回 ``False`` 时 :meth:`get_frame` 直接给出 ``None``。
    """

    def __init__(
        self,
        backend: CaptureBackend,
        *,
        gate: ForegroundGate | None = None,
    ) -> None:
        self._backend = backend
        self._gate = gate
        self._gated_skips = 0

    @property
    def backend(self) -> CaptureBackend:
        """底层采集后端。"""
        return self._backend

    @property
    def gate(self) -> ForegroundGate | None:
        """当前门控；未启用时为 ``None``。"""
        return self._gate

    @property
    def is_gated(self) -> bool:
        """是否启用了采集门控。"""
        return self._gate is not None

    @property
    def gated_skips(self) -> int:
        """因门控未通过而跳过的次数。"""
        return self._gated_skips

    def start(self) -> None:
        """启动底层采集（幂等）。"""
        self._backend.start()

    def stop(self) -> None:
        """停止底层采集（幂等）。"""
        self._backend.stop()

    def get_frame(self) -> Frame | None:
        """取得一帧。

        Returns:
            帧对象；门控未通过或后端暂无画面时为 ``None``。
        """
        if self._gate is not None and not self._gate():
            self._gated_skips += 1
            logger.debug("前台门控未通过，跳过本轮采集（累计 %d 次）", self._gated_skips)
            return None
        return self._backend.get_frame()

    def __enter__(self) -> CaptureFrameSource:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
