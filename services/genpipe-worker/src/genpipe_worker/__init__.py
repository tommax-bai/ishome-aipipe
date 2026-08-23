"""genpipe_worker：生成域 activity 执行进程。

所有 IO 与重计算收口在 activities；两台引擎（内容工厂 / 交互设计）复用同一批
activity 实现。伸缩轴独立（外部模型 API 配额、未来 GPU 池）。
"""
