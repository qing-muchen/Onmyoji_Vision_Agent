"""转场底部条带的相似度匹配（Phase 3.0）。

算法：**统一尺寸缩放 → 灰度化 → 归一化 MAE**（相似度 = ``1 - 平均绝对差 / 255``，
完全一致为 ``1.0``）。

为什么这样选：

1. 条带是**固定 UI**（底纹 + 文字，位置固定），不需要滑动搜索 → 不用 ``cv2.matchTemplate``；
2. 帧尺寸不固定（窗口可缩放），所以先把条带统一到配置尺寸再比较，把"分辨率差异"
   归一化掉——这也正是正样本（1706x58）与运行时条带（可能 2560x96）能直接比较的原因；
3. MAE 计算确定、可解释，且**不引入新依赖**（SSIM 需要 scikit-image，直方图会丢失结构）；
4. 复用而不复制：归一化相似度来自
   :func:`src.vision.scene.matcher.normalized_similarity`，缩放来自
   :func:`src.vision.scene.preprocess.resize_to`。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.core.config import StateTransitionConfig
from src.vision.scene.matcher import normalized_similarity
from src.vision.scene.preprocess import resize_to, to_grayscale

__all__ = ["TransitionBottomMatcher"]


class TransitionBottomMatcher:
    """条带相似度匹配器。

    Args:
        config: 状态识别配置中的 ``transition`` 段（提供统一缩放尺寸）。
    """

    def __init__(self, config: StateTransitionConfig) -> None:
        self._config = config

    @property
    def config(self) -> StateTransitionConfig:
        """当前配置。"""
        return self._config

    @property
    def resize_size(self) -> tuple[int, int]:
        """条带统一缩放尺寸 ``(width, height)``。"""
        return (self._config.resize_width, self._config.resize_height)

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """把条带缩放到统一尺寸并转灰度。

        同一张条带只需处理一次，结果可在多个样本之间复用。

        Args:
            image: 条带图像（BGR 或灰度）。

        Returns:
            统一尺寸下的灰度图。
        """
        return to_grayscale(resize_to(image, self.resize_size))

    def similarity(self, prepared_query: np.ndarray, prepared_sample: np.ndarray) -> float:
        """比较两张**已预处理**的条带。

        Args:
            prepared_query: 预处理后的待识别条带。
            prepared_sample: 预处理后的样本条带。

        Returns:
            ``0.0``（差异最大）到 ``1.0``（完全一致）的相似度。
        """
        return normalized_similarity(prepared_query, prepared_sample)

    def best_match(
        self,
        prepared_query: np.ndarray,
        prepared_samples: Mapping[str, np.ndarray],
    ) -> tuple[str, float] | None:
        """在若干已预处理样本中找出最相似的一张。

        Args:
            prepared_query: 预处理后的待识别条带。
            prepared_samples: ``sample_id -> 预处理后的样本条带``。

        Returns:
            ``(sample_id, 最高相似度)``；样本集合为空时返回 ``None``。
        """
        best_id: str | None = None
        best_score = 0.0
        for sample_id, prepared_sample in prepared_samples.items():
            score = self.similarity(prepared_query, prepared_sample)
            if best_id is None or score > best_score:
                best_id, best_score = sample_id, score
        if best_id is None:
            return None
        return best_id, best_score
