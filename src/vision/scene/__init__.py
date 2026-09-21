"""场景识别层（Phase 2.2）。

职责：整图判断"当前在哪里"——输入 :class:`~src.vision.capture.models.Frame`，输出
:class:`~src.vision.scene.result.SceneResult`（场景类型 / 皮肤 / 场景编号 / 置信度）。
**不做**点击、任务、状态机，也不涉及坐标与 ``Rect``。

对外入口：

* :class:`~src.vision.scene.base.BaseSceneClassifier` —— 分类器抽象接口；
* :class:`~src.vision.scene.classifier.SceneClassifier` —— 基于场景数据集的实现；
* :class:`~src.vision.scene.loader.SceneLoader` —— 从 ``dataset/scenes`` 加载场景模板；
* :class:`~src.vision.scene.matcher.SceneMatcher` —— 整图相似度计算；
* :class:`~src.vision.scene.models.SceneTemplate` / :class:`~src.vision.scene.result.SceneResult`
  —— 输入模型（模板）与输出模型（结果）。

依赖方向：``core`` → ``capture`` → ``scene``（与 ``detector`` 同级，不依赖 ``detector`` /
``action`` / ``task`` / ``state``）。
"""

from src.vision.scene.base import BaseSceneClassifier
from src.vision.scene.classifier import SceneClassifier
from src.vision.scene.loader import (
    DEFAULT_SCENE_ROOT,
    SCENE_FILENAME_PATTERN,
    SceneLoader,
    parse_scene_filename,
)
from src.vision.scene.matcher import SceneMatcher, normalized_similarity
from src.vision.scene.models import SceneFileName, SceneTemplate
from src.vision.scene.result import SceneResult

__all__ = [
    "DEFAULT_SCENE_ROOT",
    "SCENE_FILENAME_PATTERN",
    "BaseSceneClassifier",
    "SceneClassifier",
    "SceneFileName",
    "SceneLoader",
    "SceneMatcher",
    "SceneResult",
    "SceneTemplate",
    "normalized_similarity",
    "parse_scene_filename",
]
