"""OpenCV 模板匹配（支持多尺度）。

算法：``cv2.matchTemplate``。默认 ``TM_CCOEFF_NORMED``；由于游戏窗口尺寸不固定，
模板会按配置中的比例逐档缩放后再匹配，取置信度最高的一档。

只使用**归一化**匹配方法，保证分数可以稳定映射到 ``[0, 1]``。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from src.core.config import IDENTITY_SCALE, TemplateDetectorConfig
from src.core.geometry import Rect
from src.utils.logger import get_logger
from src.vision.detector.models import TemplateInfo
from src.vision.detector.template.preprocess import resize_image, to_grayscale

__all__ = [
    "FIT_TOLERANCE_PX",
    "LOWER_IS_BETTER_METHODS",
    "MAX_CONFIDENCE",
    "MIN_CONFIDENCE",
    "TemplateMatch",
    "TemplateMatcher",
]

#: 分数越小越好的匹配方法（其余方法越大越好）；仅包含归一化方法
LOWER_IS_BETTER_METHODS: frozenset[str] = frozenset({"TM_SQDIFF_NORMED"})
#: 置信度的合法下界
MIN_CONFIDENCE: float = 0.0
#: 置信度的合法上界
MAX_CONFIDENCE: float = 1.0
#: Phase 11.8-R1：允许做 bounded fit-to-ROI 的**最大像素溢出量**（严格 2 px）。
#:
#: 仅用于吸收"ROI 与模板各自独立 ``round`` 取整"造成的 1~2 px 边界差：当某个 pyramid
#: 档位的模板只比当前匹配图宽/高 1~2 px 时，把它**等比**收紧到不超过匹配图后继续匹配，
#: 而不是整档跳过；超过该容差仍按原逻辑跳过，避免把真正的尺度不匹配静默 fit 掉。
#:
#: **刻意不使用** ``max(2px, 0.5% * frame_width)`` 这类相对容差：在 1278 px 的匹配图上
#: 0.5% 约为 6.4 px，已远超实测确认的取整/边界缺陷量级。
FIT_TOLERANCE_PX: int = 2

logger = get_logger("vision.detector.template.matcher")


def _clamp_confidence(value: float) -> float:
    """把原始匹配分数裁剪到 ``[0, 1]``。"""
    return min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, value))


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    """单个模板的最佳匹配结果。

    Attributes:
        template: 命中的模板。
        confidence: 归一化置信度，范围 ``[0, 1]``。
        rect: 命中区域，位于**当前帧（窗口客户区）坐标系**。
        scale: 命中的缩放比例。**始终是配置的（逻辑）pyramid 比例**（例如 ``0.75``）；
            bounded fit-to-ROI 只是让该档位在几何上放得下，**不改变**该字段语义。
        fit_applied: 该最佳匹配是否对该档模板做过 bounded fit-to-ROI（Phase 11.8-R1）。
            正常路径恒为 ``False``。
        fit_ratio: bounded fit 实际使用的等比缩放系数；未做 fit 时为 ``1.0``。
    """

    template: TemplateInfo
    confidence: float
    rect: Rect
    scale: float
    fit_applied: bool = False
    fit_ratio: float = 1.0

    def describe(self) -> str:
        """返回可读的匹配描述。"""
        return (
            f"{self.template.template_id} confidence={self.confidence:.4f} "
            f"scale={self.scale:.2f} rect={self.rect.describe()}"
        )


class TemplateMatcher:
    """单模板多尺度匹配器。

    Args:
        config: 模板检测配置（方法、阈值、尺度）。

    Raises:
        ValueError: 配置中的匹配方法不被当前 OpenCV 支持。
    """

    def __init__(self, config: TemplateDetectorConfig) -> None:
        self._config = config
        self._method_name = config.method
        self._method = getattr(cv2, config.method, None)
        if self._method is None:
            raise ValueError(f"OpenCV 不支持的模板匹配方法: {config.method}")
        #: 模板 × scale 的灰度金字塔缓存（Phase 5.1）：模板与其缩放结果都是确定性的，
        #: 逐帧重算只是浪费；键包含 scales 元组，配置变化不会复用旧缓存。
        self._template_pyramids: dict[tuple[str, str, tuple[float, ...]], tuple[tuple[float, np.ndarray], ...]] = {}
        #: 各 scale 被选为**最佳匹配**的次数（Phase 5.1 统计，不改变判定逻辑）
        self._scale_hits: Counter[float] = Counter()
        self._pyramid_cache_hits = 0
        #: Phase 11.8-R1：累计执行 bounded fit-to-ROI 的档位次数（诊断用）
        self._fit_count = 0
        #: Phase 12.0-D：累计**真正执行** ``cv2.matchTemplate`` 的次数（诊断用）。
        #: 只在进入 :meth:`_match_once` 时 +1；因模板显著大于匹配图
        #: （``overflow > FIT_TOLERANCE_PX``）而整档跳过的不计入。**不参与判定逻辑**。
        self._match_calls = 0

    @property
    def config(self) -> TemplateDetectorConfig:
        """当前配置。"""
        return self._config

    @property
    def method_name(self) -> str:
        """当前使用的匹配方法名。"""
        return self._method_name

    @property
    def scales(self) -> tuple[float, ...]:
        """实际参与搜索的比例序列（关闭多尺度时只含 1.0）。"""
        return self._config.effective_scales

    @staticmethod
    def prepare(frame_image: np.ndarray) -> np.ndarray:
        """把帧图像转换为匹配用的灰度图。

        同一帧只需转换一次，可在多个模板之间复用。

        Args:
            frame_image: BGR 帧图像。

        Returns:
            灰度帧图像。
        """
        return to_grayscale(frame_image)

    def template_pyramid(
        self, template: TemplateInfo
    ) -> tuple[tuple[float, np.ndarray], ...]:
        """返回模板在各 scale 下的灰度图（**构造期缓存**，Phase 5.1）。

        等价性：原实现每帧对每个模板重算 ``to_grayscale`` 与 ``resize_image``；
        模板及其缩放结果都是确定性的，缓存只是避免重复计算，**不改变任何匹配结果**。
        缓存键包含模板标识与 scale 元组，因此配置变化不会复用旧缓存。

        Args:
            template: 待匹配的模板。

        Returns:
            ``((scale, 缩放后的灰度模板), ...)``，顺序与 :attr:`scales` 完全一致。
        """
        key = (template.template_id, str(template.path), self.scales)
        cached = self._template_pyramids.get(key)
        if cached is not None:
            self._pyramid_cache_hits += 1
            return cached
        gray = to_grayscale(template.image)
        pyramid = tuple((scale, resize_image(gray, scale)) for scale in self.scales)
        self._template_pyramids[key] = pyramid
        return pyramid

    @property
    def scale_hits(self) -> dict[float, int]:
        """各 scale 被选为**最佳匹配**的次数（Phase 5.1 统计，不改变判定逻辑）。"""
        return dict(self._scale_hits)

    @property
    def pyramid_cache_size(self) -> int:
        """已缓存的模板金字塔数量。"""
        return len(self._template_pyramids)

    @property
    def pyramid_cache_hits(self) -> int:
        """命中模板金字塔缓存的次数（每个模板第二次起为命中）。"""
        return self._pyramid_cache_hits

    def reset_statistics(self) -> None:
        """清空 scale 命中统计与 bounded fit 计数（不影响金字塔缓存）。"""
        self._scale_hits.clear()
        self._fit_count = 0

    @property
    def fit_tolerance_px(self) -> int:
        """bounded fit-to-ROI 允许的最大像素溢出量（Phase 11.8-R1）。"""
        return FIT_TOLERANCE_PX

    @property
    def fit_count(self) -> int:
        """累计执行 bounded fit-to-ROI 的档位次数（诊断用，不改变判定逻辑）。"""
        return self._fit_count

    @property
    def match_calls(self) -> int:
        """累计**真正执行** ``cv2.matchTemplate`` 的次数（Phase 12.0-D 诊断用）。

        只统计实际进入匹配的档位：因模板显著大于匹配图（``overflow > FIT_TOLERANCE_PX``）
        而整档跳过的**不计入**；bounded fit-to-ROI 之后若实际执行了匹配则**计入**。
        该计数**不参与任何判定逻辑**，仅供观测（用于区分"跑了但分数低"与"根本没跑"）。
        """
        return self._match_calls

    def match(self, frame_gray: np.ndarray, template: TemplateInfo) -> TemplateMatch | None:
        """在灰度帧中查找单个模板的最佳位置。

        Phase 5.1：模板的灰度图与各 scale 缩放图来自 :meth:`template_pyramid` 缓存
        （不再逐帧重算），匹配与选择逻辑保持不变。

        Phase 11.8-R1：当某个 pyramid 档位的模板只比匹配图宽/高 **1~2 px** 时，不再整档
        跳过，而是先把该档模板**等比**收紧到不超过匹配图（bounded fit-to-ROI）再参与匹配。
        超过 :data:`FIT_TOLERANCE_PX` 的溢出仍按原逻辑跳过。

        Args:
            frame_gray: 与帧同源的灰度图（见 :meth:`prepare`）。
            template: 待匹配的模板。

        Returns:
            最佳匹配结果；未达阈值或模板显著大于帧时返回 ``None``。
        """
        frame_height, frame_width = frame_gray.shape[:2]
        best: TemplateMatch | None = None

        for scale, scaled in self.template_pyramid(template):
            height, width = scaled.shape[:2]
            overflow = max(width - frame_width, height - frame_height)
            fit_applied = False
            fit_ratio = IDENTITY_SCALE
            if overflow > 0:
                if overflow > FIT_TOLERANCE_PX:
                    logger.debug(
                        "跳过比例 %.2f：模板 %dx%d 超出匹配图 %dx%d 达 %d px（容差 %d px）",
                        scale,
                        width,
                        height,
                        frame_width,
                        frame_height,
                        overflow,
                        FIT_TOLERANCE_PX,
                    )
                    continue
                original_width, original_height = width, height
                scaled, fit_ratio = self._fit_to_frame(scaled, frame_width, frame_height)
                height, width = scaled.shape[:2]
                fit_applied = True
                self._fit_count += 1
                logger.debug(
                    "bounded fit 比例 %.2f：模板 %dx%d -> %dx%d（溢出 %d px，ratio=%.5f）",
                    scale,
                    original_width,
                    original_height,
                    width,
                    height,
                    overflow,
                    fit_ratio,
                )

            confidence, location = self._match_once(frame_gray, scaled)
            if best is None or confidence > best.confidence:
                best = TemplateMatch(
                    template=template,
                    confidence=confidence,
                    rect=Rect(x=int(location[0]), y=int(location[1]), width=int(width), height=int(height)),
                    scale=scale,
                    fit_applied=fit_applied,
                    fit_ratio=fit_ratio,
                )

        if best is None:
            return None
        self._scale_hits[best.scale] += 1
        if best.confidence < self._config.threshold:
            return None
        return best

    def _fit_to_frame(
        self, template_gray: np.ndarray, frame_width: int, frame_height: int
    ) -> tuple[np.ndarray, float]:
        """把略大于匹配图的模板**等比**收紧到不超过匹配图（Phase 11.8-R1）。

        只服务于 1~2 px 级的取整/边界溢出：**始终返回新数组**，既不修改入参，也绝不写回
        :meth:`template_pyramid` 缓存，因此不会污染后续帧（某个 1703 帧不会改变 1704/1706
        帧使用的模板尺寸）。

        Args:
            template_gray: 当前 pyramid 档位的灰度模板（不会被就地修改）。
            frame_width: 匹配图宽度。
            frame_height: 匹配图高度。

        Returns:
            ``(收紧后的灰度模板, 实际使用的等比缩放系数)``；宽高均 ``>= 1`` 且不超过匹配图。
        """
        height, width = template_gray.shape[:2]
        ratio = min(frame_width / width, frame_height / height)
        if ratio >= IDENTITY_SCALE:
            return template_gray, IDENTITY_SCALE

        fitted = resize_image(template_gray, ratio)
        # ``resize_image`` 用 round 取整，极端情况下可能进位 1 px；最多再收紧两次，且每次都
        # 严格等比缩小，保证最终尺寸一定落在匹配图内。
        for _ in range(2):
            fitted_height, fitted_width = fitted.shape[:2]
            if fitted_width <= frame_width and fitted_height <= frame_height:
                break
            ratio *= min(frame_width / fitted_width, frame_height / fitted_height)
            fitted = resize_image(template_gray, ratio)
        return fitted, ratio

    def _match_once(self, frame_gray: np.ndarray, template_gray: np.ndarray) -> tuple[float, tuple[int, int]]:
        """执行一次 ``cv2.matchTemplate``。

        Args:
            frame_gray: 灰度帧。
            template_gray: 缩放后的灰度模板。

        Returns:
            ``(归一化置信度, (left, top))``。
        """
        # Phase 12.0-D：只有真正执行匹配才计数（调用方已排除 overflow 超容差的档位）
        self._match_calls += 1
        result = cv2.matchTemplate(frame_gray, template_gray, self._method)
        min_value, max_value, min_location, max_location = cv2.minMaxLoc(result)
        if self._method_name in LOWER_IS_BETTER_METHODS:
            return _clamp_confidence(1.0 - float(min_value)), (int(min_location[0]), int(min_location[1]))
        return _clamp_confidence(float(max_value)), (int(max_location[0]), int(max_location[1]))
