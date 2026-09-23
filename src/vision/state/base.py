"""状态分类器抽象接口（Phase 3.0；Phase 4.6.3 起输入扩展为「帧或证据」）。

约定：

* 输入是**帧**或**证据序列**：

  * :class:`~src.vision.capture.models.Frame` —— 帧携带时间戳与来源信息，便于后续状态机
    按"哪一帧得出的结论"回溯；本接口**不接受裸 numpy 图像**；
  * :class:`~src.vision.evidence.models.EvidenceResult` 序列（Phase 4.6.3）—— 由证据层
    （``EvidenceRegistry`` / detector）产出，状态层只回答"这些证据意味着什么"。

* 状态识别是整图 / 区域判断，**不要求**帧携带 ``coordinate_transform``
  （与 :mod:`src.vision.scene` 一致，与 :mod:`src.vision.detector` 的定位型检测不同）。
* 输出统一为单个 :class:`~src.vision.state.models.StateResult`；
  :class:`~src.vision.state.models.StateName` 已登记完整状态集
  （``yard`` / ``exploration`` / ``battle`` / ``battle_result`` / ``battle_reward`` /
  ``transition`` / ``unknown``）。**不做决策、不触发任何交互**。

实现可以只支持其中一种输入（例如纯证据驱动的分类器不接受帧）；对不支持的输入必须抛出
明确异常，而不是静默返回兜底状态。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.vision.capture.models import Frame
from src.vision.state.models import StateResult

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型注解，避免抽象接口在运行期依赖证据层
    from src.vision.evidence.models import EvidenceResult

__all__ = ["BaseStateClassifier"]


class BaseStateClassifier(ABC):
    """状态分类器的统一接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """分类器名称（用于日志）。"""

    @abstractmethod
    def classify(self, input: Frame | Sequence[EvidenceResult]) -> StateResult:
        """判断当前状态。

        Args:
            input: 待识别的帧，或证据结果序列。

        Returns:
            状态结果；无法判定时返回 ``unknown``（而不是抛异常）。

        Raises:
            TypeError: 输入类型不受支持（既不是帧也不是证据序列）。
        """
