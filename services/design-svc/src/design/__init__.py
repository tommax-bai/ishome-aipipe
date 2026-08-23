"""design：交互设计状态机与决策域。

design-svc 管状态和决策，genpipe 管算力和管线：本服务不跑重计算（解析、
布局、渲染全部下发 genpipe activities）；对渠道零感知（只说 channel.v1
统一消息模型的语言，禁止渠道名分支——CI grep 拦截渠道名字面量）。
"""
