"""转场状态识别子包（Phase 3.0）。

职责：把"帧底部固定条带"和正/负样本比较，回答"当前是否处于转场/加载插画界面"。

数据流::

    Frame
      ↓  TransitionBottomExtractor（按配置比例裁底部条带）
    bottom strip
      ↓  TransitionBottomMatcher（统一尺寸 + 灰度 + 归一化 MAE）
    正样本最高分 / 负样本最高分
      ↓  TransitionClassifier（阈值 + 负样本否决）
    TransitionMatch.is_transition

对外入口：

* :class:`~src.vision.state.transition.dataset.TransitionSampleLoader` —— 加载
  ``dataset/states`` 正/负样本；
* :class:`~src.vision.state.transition.extractor.TransitionBottomExtractor` —— 底部区域裁剪；
* :class:`~src.vision.state.transition.matcher.TransitionBottomMatcher` —— 条带相似度；
* :class:`~src.vision.state.transition.classifier.TransitionClassifier` /
  :class:`~src.vision.state.transition.classifier.TransitionMatch` —— 判定与过程数据。
"""

from src.vision.state.transition.classifier import TransitionClassifier, TransitionMatch
from src.vision.state.transition.dataset import (
    DEFAULT_STATE_ROOT,
    NEGATIVE_DIRNAME,
    POSITIVE_DIRNAME,
    SampleKind,
    TransitionSample,
    TransitionSampleLoader,
)
from src.vision.state.transition.extractor import TransitionBottomExtractor
from src.vision.state.transition.matcher import TransitionBottomMatcher

__all__ = [
    "DEFAULT_STATE_ROOT",
    "NEGATIVE_DIRNAME",
    "POSITIVE_DIRNAME",
    "SampleKind",
    "TransitionBottomExtractor",
    "TransitionBottomMatcher",
    "TransitionClassifier",
    "TransitionMatch",
    "TransitionSample",
    "TransitionSampleLoader",
]
