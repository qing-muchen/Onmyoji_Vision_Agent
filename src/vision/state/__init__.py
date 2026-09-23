"""游戏状态识别层（Phase 3.0：State Classification Foundation）。

职责：把**证据**或**帧**映射到一个游戏状态。

两条输入路径（Phase 4.6.3 / 4.6.5），共用 ``StateClassifier.classify`` 入口（按输入类型自动分派）::

    # 1. 证据驱动（**权威路径**）：证据层回答"看到了什么"，本层回答"这些证据意味着什么"
    [EvidenceResult, ...]                  # 由 vision.evidence 的 detector 产出
      ↓  EvidenceAggregator.aggregate       # 规则来自 evidence.aggregator.rules（配置驱动）
      ↓  StateClassifier.classify_evidence
    StateResult(state_name, confidence, source=EVIDENCE, reason, evidence)

    # 2. 帧驱动（Phase 3.0；4.6.5 起只判转场，兼容保留）
    Frame
      ↓  StateClassifier.classify_frame
    TransitionClassifier（底部条带 + 正负样本）   # 需要整帧像素
      ↓  Fallback
    StateResult(state_name, confidence, source, detail)
    # 庭院（yard）已迁出帧路径：帧路径不再调用 YardClassifier，也不再产出 yard

状态集合：:class:`~src.vision.state.models.StateName` 已登记**完整状态集**
（``yard`` / ``exploration`` / ``battle`` / ``battle_result`` / ``battle_reward`` /
``transition`` / ``unknown``），**全部由证据驱动路径按配置规则判定**（含 ``yard``——
其权威来源是 ``bottom_menu_yard`` 证据）；帧驱动路径只判 ``transition``，其余返回 ``unknown``。
本模块不含"证据名 → 状态"的硬编码映射。

对外入口：

* :class:`~src.vision.state.base.BaseStateClassifier` —— 抽象接口（``classify(input)``）；
* :class:`~src.vision.state.classifier.StateClassifier` —— 状态判定编排：

  * ``from_config(config)``：装配帧驱动分支（transition）+ 可选的庭院视觉组件；
  * ``from_evidence_config(config.evidence)``：装配**纯证据驱动**分类器（不加载图像数据集）；
  * ``classify(input)`` 自动分派；``classify_frame`` / ``classify_evidence`` 为显式入口。

* :class:`~src.vision.state.models.StateResult` / ``StateName`` / ``StateSource`` —— 结果模型；
* :class:`~src.vision.state.transition.classifier.TransitionClassifier` —— 底部条带判定；
* :class:`~src.vision.state.yard.classifier.YardClassifier` —— 庭院视觉组件（包装 scene）；
  Phase 4.6.5 起**不再参与状态判定**（见该模块文档），保留为独立视觉组件。

依赖方向：``core`` → ``capture`` → ``scene`` / ``evidence`` → ``state``（本层可依赖
``scene`` 与 ``evidence``；``scene`` / ``detector`` / ``ocr`` / ``evidence`` **不得**
反向依赖 ``state``）。本层不做任何交互，也不做状态转移——状态机属于后续阶段（``src/state``）。
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
