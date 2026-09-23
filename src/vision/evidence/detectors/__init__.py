"""证据 detector 集合与工厂（Phase 4.1 / 4.2；4.6.4 加入包装型转场证据）。

新增 detector 的完整流程（**不需要修改核心流程**）：

1. 继承 :class:`~src.vision.evidence.base.TemplateEvidenceDetector`
   （ROI 模板型）或 :class:`~src.vision.evidence.base.BaseEvidenceDetector`（包装型），
   声明 ``NAME`` 与 ``EVIDENCE_TYPE``；
2. 在本模块的 :data:`DETECTOR_CLASSES` 中登记；
3. 在 ``configs/config.yaml`` 的 ``evidence.detectors`` 下加同名条目（ROI / 模板目录 / 阈值）；
4. 若要参与状态判定，在 ``evidence.aggregator.rules`` 中引用该证据名。

两类 detector 的差异：

* **模板型**（``TemplateEvidenceDetector`` 子类）：配置驱动 ROI 与模板，``create_detectors``
  用统一签名构造；
* **包装型**（例如 :class:`~src.vision.evidence.detectors.transition_banner.TransitionBannerDetector`）：
  复用其它层已有的判定能力，需要调用方通过 ``create_detectors(..., transition_classifier=...)``
  注入判定器（依赖倒置，避免 ``vision.evidence`` → ``vision.state`` 的循环依赖）；
  未注入时该 detector 仍会被构造，但只输出"未检出"并在 ``metadata`` 记录原因。

:func:`create_detectors` 会按配置一次性装配全部登记的 detector（模板在此时加载）。
"""

from __future__ import annotations

from pathlib import Path

from src.core.config import EvidenceConfig
from src.utils.logger import get_logger
from src.vision.evidence.base import BaseEvidenceDetector, TemplateEvidenceDetector
from src.vision.evidence.detectors.battle_action_bar import BattleActionBarDetector
from src.vision.evidence.detectors.battle_auto import BattleAutoDetector
from src.vision.evidence.detectors.battle_shikigami import BattleShikigamiIconsDetector
from src.vision.evidence.detectors.battle_skill_area import BattleSkillAreaDetector
from src.vision.evidence.detectors.bottom_menu import (
    BottomMenuExplorationDetector,
    BottomMenuYardDetector,
)
from src.vision.evidence.detectors.chapter_panel import ChapterPanelDetector
from src.vision.evidence.detectors.reward_panel import RewardPanelDetector
from src.vision.evidence.detectors.transition_banner import (
    TransitionBannerDetector,
    TransitionProbe,
)
from src.vision.evidence.detectors.victory_icon import VictoryIconDetector

__all__ = [
    "DETECTOR_CLASSES",
    "BattleActionBarDetector",
    "BattleAutoDetector",
    "BattleShikigamiIconsDetector",
    "BattleSkillAreaDetector",
    "BottomMenuExplorationDetector",
    "BottomMenuYardDetector",
    "ChapterPanelDetector",
    "RewardPanelDetector",
    "TransitionBannerDetector",
    "TransitionProbe",
    "VictoryIconDetector",
    "create_detectors",
]

logger = get_logger("vision.evidence.detectors")

#: 已实现的 detector：证据名 -> 实现类
#:
#: 其中 :class:`TransitionBannerDetector` 是**包装型** detector：``create_detectors``
#: 需要调用方注入 ``transition_classifier``（未注入时输出"未检出"）。
DETECTOR_CLASSES: dict[str, type[BaseEvidenceDetector]] = {
    BottomMenuYardDetector.NAME: BottomMenuYardDetector,
    BottomMenuExplorationDetector.NAME: BottomMenuExplorationDetector,
    ChapterPanelDetector.NAME: ChapterPanelDetector,
    BattleAutoDetector.NAME: BattleAutoDetector,
    BattleSkillAreaDetector.NAME: BattleSkillAreaDetector,
    BattleActionBarDetector.NAME: BattleActionBarDetector,
    BattleShikigamiIconsDetector.NAME: BattleShikigamiIconsDetector,
    VictoryIconDetector.NAME: VictoryIconDetector,
    RewardPanelDetector.NAME: RewardPanelDetector,
    TransitionBannerDetector.NAME: TransitionBannerDetector,
}


def create_detectors(
    config: EvidenceConfig,
    *,
    ocr_engine: object | None = None,
    root: Path | None = None,
    transition_classifier: TransitionProbe | None = None,
) -> list[BaseEvidenceDetector]:
    """按配置装配全部已登记且已配置的 detector。

    Args:
        config: 证据层配置。
        ocr_engine: 可选的 OCR 引擎；注入后启用文字锚点证据。
        root: 解析模板相对路径的基准目录；省略时使用项目根。
        transition_classifier: 转场判定器（通常为
            :class:`~src.vision.state.transition.classifier.TransitionClassifier`）；
            由**编排层**构造后注入（本层不导入 ``vision.state``，以免形成循环依赖）。
            省略时转场证据会构造出来但始终输出"未检出"，并在 ``metadata`` 记录数据缺失。

    Returns:
        detector 列表（按 :data:`DETECTOR_CLASSES` 的顺序）；``evidence.detectors``
        中没有对应配置的 detector 会被跳过并记录日志。
    """
    detectors: list[BaseEvidenceDetector] = []
    for name, detector_class in DETECTOR_CLASSES.items():
        if name not in config.detectors:
            logger.warning("evidence.detectors 未配置 %s，跳过该 detector", name)
            continue
        if issubclass(detector_class, TransitionBannerDetector):
            detectors.append(
                TransitionBannerDetector.from_config(
                    config, probe=transition_classifier
                )
            )
            continue
        detectors.append(detector_class.from_config(config, ocr_engine=ocr_engine, root=root))
    return detectors
