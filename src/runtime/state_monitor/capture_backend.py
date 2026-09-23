"""采集后端的创建与生命周期（Phase 5.4 从 ``frame_source.py`` 拆出）。

职责边界（只负责"后端"这一件事）：

* **创建**：按 ``capture.backend`` 选择后端 —— ``auto`` 时 **dxcam 优先、mss 回退**，
  具体候选顺序由 :func:`~src.vision.capture.factory.create_capture_backend` 决定（本层不重复实现）；
* **持有**：记住当前后端与其采集区域，区域未变则复用（不重建）；
* **重建**：采集区域变化（窗口移动 / 缩放）时释放旧后端并按新区域重建；
* **释放**：``release()`` 停止后端（dxcam 会话与内部线程随之结束），**异常安全**（清理路径不抛）。

本模块不解析窗口、不做前台门控、不判定状态。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.config import CaptureConfig
from src.core.exceptions import CaptureError, OnmyojiVisionError
from src.core.geometry import CoordinateTransform, Rect
from src.utils.logger import get_logger
from src.vision.capture import CaptureBackend, Frame, create_capture_backend
from src.vision.window import WindowInfo

from src.runtime.state_monitor.foreground_guard import SkipReason

__all__ = ["BackendResult", "RuntimeCaptureBackend"]

logger = get_logger("runtime.state_monitor.capture_backend")


@dataclass(frozen=True, slots=True)
class BackendResult:
    """一次后端装配的结果。

    Attributes:
        ok: 后端是否就绪。
        name: 后端名称（``dxcam`` / ``mss``）；失败时为 ``None``。
        region: 该后端服务的采集区域；失败时为 ``None``。
        reason: 失败原因（``backend_unavailable`` / ``capture_failed``）。
        message: 人类可读说明。
        error: 触发失败的原始异常（供 ``strict`` 模式复抛）。
    """

    ok: bool
    name: str | None = None
    region: Rect | None = None
    reason: SkipReason | None = None
    message: str = ""
    error: BaseException | None = None

    def describe(self) -> str:
        """返回可读描述。"""
        if self.ok:
            return f"ok backend={self.name} region={self.region.describe() if self.region else 'n/a'}"
        reason = self.reason.value if self.reason is not None else "unknown"
        return f"unavailable reason={reason} message={self.message}"


class RuntimeCaptureBackend:
    """真实窗口帧来源持有的采集后端（dxcam 优先 / mss 回退 + 区域变化重建）。

    Args:
        capture: 采集配置（后端、帧率、显示器）。
    """

    def __init__(self, capture: CaptureConfig) -> None:
        self._capture = capture
        self._backend: CaptureBackend | None = None
        self._region: Rect | None = None
        self._rebuilds = 0

    @property
    def config(self) -> CaptureConfig:
        """采集配置。"""
        return self._capture

    @property
    def backend(self) -> CaptureBackend | None:
        """当前后端；未就绪时为 ``None``。"""
        return self._backend

    @property
    def name(self) -> str | None:
        """当前后端名称；未就绪时为 ``None``。"""
        return self._backend.name if self._backend is not None else None

    @property
    def region(self) -> Rect | None:
        """当前后端服务的采集区域；未就绪时为 ``None``。"""
        return self._region

    @property
    def is_ready(self) -> bool:
        """后端是否已就绪。"""
        return self._backend is not None

    @property
    def rebuilds(self) -> int:
        """后端创建次数（含首次；窗口移动/缩放会使其增长）。"""
        return self._rebuilds

    def ensure(
        self,
        *,
        window: WindowInfo,
        region: Rect,
        transform: CoordinateTransform | None,
    ) -> BackendResult:
        """确保后端就绪（区域未变则复用，变了则重建）。

        Args:
            window: 目标窗口（后端用于推导所在显示器）。
            region: 需要采集的区域（屏幕绝对坐标）。
            transform: 随帧携带的坐标映射。

        Returns:
            装配结果；失败时 ``reason`` 说明原因（不会抛异常）。
        """
        if self._backend is not None and self._region == region:
            return BackendResult(ok=True, name=self._backend.name, region=self._region)

        if self._backend is not None:
            logger.info(
                "采集区域变化（%s -> %s），重建采集后端",
                self._region.describe() if self._region is not None else "n/a",
                region.describe(),
            )
            self.release()

        try:
            backend = create_capture_backend(self._capture, window=window, transform=transform)
            backend.start()
        except OnmyojiVisionError as exc:
            self.release()
            return BackendResult(
                ok=False,
                reason=SkipReason.BACKEND_UNAVAILABLE,
                message=f"采集后端不可用: {exc}",
                error=exc,
            )
        except OSError as exc:  # 设备级异常（例如显示器会话丢失）
            self.release()
            return BackendResult(
                ok=False,
                reason=SkipReason.CAPTURE_FAILED,
                message=f"采集后端启动失败（系统错误）: {exc}",
                error=exc,
            )

        self._backend = backend
        self._region = region
        self._rebuilds += 1
        logger.info(
            "采集后端就绪: backend=%s region=%s（dxcam 优先 / mss 回退，由 capture.backend 决定）",
            backend.name,
            region.describe(),
        )
        return BackendResult(ok=True, name=backend.name, region=region)

    def grab(self) -> Frame | None:
        """抓取一帧（异常由调用方按既有语义处理）。

        Returns:
            帧对象；后端本轮没有产出时为 ``None``。

        Raises:
            CaptureError: 后端未就绪或抓取失败。
        """
        backend = self._backend
        if backend is None:
            raise CaptureError("采集后端未就绪，无法抓帧")
        return backend.get_frame()

    def release(self) -> None:
        """释放当前后端（幂等，清理路径不抛异常）。

        ``CaptureBackend.stop()`` 会停止底层设备会话：dxcam 的后台取帧线程随
        ``camera.stop()`` 结束，mss 的会话句柄随 ``session.close()`` 关闭。
        """
        backend = self._backend
        self._backend = None
        self._region = None
        if backend is None:
            return
        try:
            backend.stop()
        except Exception as exc:  # noqa: BLE001 - 清理路径：设备异常不应影响主循环
            logger.warning("采集后端停止异常（已忽略）: %s", exc)

    def describe(self) -> str:
        """返回可读描述。"""
        region = self._region.describe() if self._region is not None else "n/a"
        return (
            f"RuntimeCaptureBackend(backend={self.name or 'n/a'}, region={region}, "
            f"rebuilds={self._rebuilds}, running={self.is_ready})"
        )

    def __enter__(self) -> RuntimeCaptureBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
