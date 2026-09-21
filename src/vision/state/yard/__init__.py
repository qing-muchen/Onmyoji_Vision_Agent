"""庭院状态识别子包（Phase 3.0）。

对外入口：:class:`~src.vision.state.yard.classifier.YardClassifier` —— 包装
``src/vision/scene`` 的场景分类能力，回答"当前是否处于庭院"。
"""

from src.vision.state.yard.classifier import YardClassifier

__all__ = ["YardClassifier"]
