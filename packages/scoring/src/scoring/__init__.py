"""打分器（scorer）：机检门禁的量化打分端口。

变化轴 #6（规范 §5.2）：新增打分维度 = 新 Scorer 实现 + 注册表一行，既有
代码零修改（R7 验收标准）。scorer_id 注册表唯一真源在 ishome-contracts。
库不感知使用者：本包禁止依赖 services/*（import-linter 锁定）。
"""

from scoring.scorer import Scorer, ScorerResult

__all__ = ["Scorer", "ScorerResult"]
