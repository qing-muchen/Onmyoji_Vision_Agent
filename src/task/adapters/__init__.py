"""任务侧转换层（Phase 10）：把外部结果转成 :mod:`src.task.context` 的数据模型。

```text
src/task/adapters/
├── __init__.py            统一导出
└── action_feedback.py     ActionFeedbackAdapter（ActionOutcome → ActionRecord → DecisionContext）
```

设计原则：**context 是数据模型，adapter 是转换层**。后续新增
``VisionFeedbackAdapter`` / ``SchedulerFeedbackAdapter`` 时只加文件，
不会让 ``src/task/context.py`` 膨胀。

依赖边界（AST 用例固化）：转换层只 import 标准库 + ``src.core`` / ``src.utils`` / ``src.task``；
**不** import ``src.action`` / ``src.vision`` / ``src.runtime``——
外部结果一律按**字段视图**（duck typing）读取。
"""

from src.task.adapters.action_feedback import ActionFeedbackAdapter

__all__ = ["ActionFeedbackAdapter"]
