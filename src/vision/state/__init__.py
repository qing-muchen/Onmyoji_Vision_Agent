"""游戏状态识别层（Phase 3.0：State Classification Foundation）。

职责：把帧**映射到一个游戏状态**——当前只支持三态：

* :attr:`~src.vision.state.models.StateName.YARD` —— 庭院界面（复用 ``scene`` 的场景识别）；
* :attr:`~src.vision.state.models.StateName.TRANSITION` —— 转场 / 加载插画界面
  （只使用底部固定条带，正样本 + 负样本双侧判别）；
* :attr:`~src.vision.state.models.StateName.UNKNOWN` —— 其余所有界面（战斗、探索、商店……）。

数据流::

    Frame
      ↓  StateClassifier.classify(frame)
    TransitionClassifier（强特征，优先）  +  YardClassifier（复用 scene）
      ↓  Fallback
    StateResult(state_name, confidence, source, detail)

对外入口：

* :class:`~src.vision.state.base.BaseStateClassifier` —— 抽象接口（``classify(frame)``）；
* :class:`~src.vision.state.classifier.StateClassifier` —— 三态编排（``from_config`` 装配）；
* :class:`~src.vision.state.models.StateResult` / ``StateName`` / ``StateSource`` —— 结果模型；
* :class:`~src.vision.state.transition.classifier.TransitionClassifier` —— 底部条带判定；
* :class:`~src.vision.state.yard.classifier.YardClassifier` —— 庭院判定（包装 scene）。

依赖方向：``core`` → ``capture`` → ``scene`` → ``state``（本层可依赖 ``scene``；
``scene`` / ``detector`` / ``ocr`` **不得**反向依赖 ``state``）。本层不做任何交互，
也不做状态转移——状态机属于后续阶段（``src/state``）。
"""

from src.vision.state.base import BaseStateClassifier
from src.vision.state.classifier import StateClassifier
from src.vision.state.models import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    StateName,
    StateResult,
    StateSource,
)

__all__ = [
    "MAX_CONFIDENCE",
    "MIN_CONFIDENCE",
    "BaseStateClassifier",
    "StateClassifier",
    "StateName",
    "StateResult",
    "StateSource",
]
