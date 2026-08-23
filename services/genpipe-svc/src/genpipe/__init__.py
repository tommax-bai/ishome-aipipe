"""genpipe：内容工厂生成编排域。

职责：批量预生成编排（Temporal workflow）+ 发布门禁状态机（consistency /
compliance / scorer 机检，全自动）。重计算不在此进程发生——全部经 activity
下发 genpipe-worker。
"""
