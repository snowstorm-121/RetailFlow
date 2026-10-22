# 来源与二次开发

RetailFlow 复用 Agent 循环模板、固定 retail 业务环境和本地模型。项目新增的是执行控制、状态管理与结果检查，而不是模型训练或整套业务规则。

| 来源 | 复用部分 | 本项目的改造 |
| --- | --- | --- |
| [LangChain react-agent](https://github.com/langchain-ai/react-agent) | StateGraph 模型—工具循环、消息与条件路由 | 本地 ChatOllama 接入、受控业务节点、暂停、预算和追踪 |
| [τ2-bench retail](https://github.com/sierra-research/tau2-bench) | 模拟数据、任务政策、RetailTools 方法及数据模型 | 固定文件适配、工具暴露范围、身份绑定、权限检查与独立业务快照 |
| Ollama / Qwen3 | 模型权重、推理服务和原生工具输出能力 | 请求配置、协议反馈及输出检查；未训练或微调模型 |
| SQLite / LangChain 消息类型 | 数据库事务与消息序列化 | 业务和回执同事务提交、会话历史、预算及恢复逻辑 |

上游固定版本与逐文件适配见 [UPSTREAM.md](../UPSTREAM.md)。

## 核心新增代码

- [environment.py](../src/retailflow/environment.py)：身份、权限、提案、原参数确认、业务差异提交与执行回执。
- [storage.py](../src/retailflow/storage.py)：本地事务、业务快照与会话状态。
- [runtime.py](../src/retailflow/runtime.py)、[context.py](../src/retailflow/context.py)：共享运行入口、工具交互、历史恢复、上下文与预算。
- [dataset.py](../src/retailflow/dataset.py)、[user_script.py](../src/retailflow/user_script.py)、[evaluation.py](../src/retailflow/evaluation.py)：任务派生、角色隔离、独立业务状态与结果检查。

模型决策与应用保护分别验证，不能从确定性测试或单次演示推断业务任务普遍可靠。项目不包含真实电商支付、生产身份认证、网页、多 Agent 或模型训练。

## 开发与审查

实现、排障和部分语义复核使用了 Codex 辅助。复用组件和模型能力归属各自来源；辅助语义复核不属于独立人工标注。历史实验及修复记录保留原配置与证据，不因展示文档修订而改变。

## 许可

[react-agent MIT 通知](../LICENSES/react-agent-MIT.txt) 和 [τ2-bench MIT 通知](../vendor/tau_retail/source/LICENSE)保留。原始 retail 数据文件没有独立文件级许可声明，具体来源与限制记录在清单中；模拟数据不应替换为真实客户信息。

新增代码尚未选择独立许可证，GitHub 公开不自动意味着全部内容采用 MIT。
