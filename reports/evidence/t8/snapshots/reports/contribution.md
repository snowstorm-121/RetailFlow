# 来源、二开贡献与作者边界

日期：2026-10-20。此表区分项目新增工程与个人独立完成程度。RetailFlow 的实现、测试和排障由 Codex 协助完成；用户提出范围与约束并参与确认，关键理解仍单独验收。不能把生成的代码说成用户从零独立手写。

| 来源 / 已有能力 | 本项目实际复用 | RetailFlow 二开 | 不应冒称 |
| --- | --- | --- | --- |
| LangChain react-agent，固定 c93701b297e643c7f80926d78247e5dc06c87bfa | StateGraph 模型—工具循环、消息与条件路由模板 | 本地 ChatOllama 接入、retail 受控节点、暂停/预算/追踪；保留 T1 烟测图 | 自研 Agent 框架、发明 ReAct、原生 checkpoint 已全部接入 |
| τ2-bench retail，固定 2174a603f6d014ef94473ffa95957f6ce27100db | 模拟 DB、任务/政策、RetailTools 业务方法、Pydantic 模型 | 薄导入兼容；只暴露八工具；身份绑定、权限和政策错误反馈、提案确认、业务快照隔离 | 自己编写全部订单规则、真实电商支付/退款、官方 τ2 成绩 |
| Ollama / Qwen3 4B Q4_K_M | 已有服务、模型文件及其原生工具输出能力 | 请求配置和协议核查、调用约束和失败记录 | 自训练、微调模型或改进模型算法；原生 FC 可用不等于任务可靠 |
| SQLite / LangChain 消息类型 | 本地事务、序列化消息 | 同事务业务+回执、会话历史、预算与恢复；提案变更失效 | 分布式 exactly-once；LangGraph SqliteSaver 是本项目自研 |
| 固定 retail 源任务 | 13 道完整目标派生题，Dev 9/Test 4 | 同源用户组划分、独立初始状态、角色隔离、最终状态/必要沟通/副作用评分、三组冻结对照 | 80/60 道已完成、自动生成大量独立任务、提升或稳定生产成功率 |

## 文件责任图

- `src/react_agent/`：模板改造点；`prompts.py`、`__init__.py` 等保留来源，详细历史在 [UPSTREAM.md](../UPSTREAM.md)。
- `src/retailflow/upstream/tools.py`、`data_model.py`：固定上游副本，业务方法体与字段未改，导入适配单列。`vendor/tau_retail/source` 原始文件保留且校验哈希。
- `environment.py / storage.py`：应用权限、提案、原参数确认、差异提交、回执和独立快照。最小身份定位是模拟政策绑定，不是生产身份认证。
- `runtime.py / context.py / cli.py`：共享 run_turn、消息恢复、受控上下文、本地交互与显式确认。
- `dataset.py / user_script.py / evaluation.py / frozen.py`：派生与划分、角色可见范围、评分/聚合、配置漂移拒绝；语义复核由助手辅助，非独立人工标注。
- `tests/`：确定性回归，脚本模型只验证路径；真实模型证据另外保存。T8 的 `examples/confirmation_change.py` 是直接应用层演示，不是生产 Agent 或评测新分支。

## 可核对的结果

T1 小工具烟测 10/10 是历史可行性结果，不等于 retail Task 成功率。T8 重新跑了真实模型确认取消和按政策解释拒绝；参数变更/旧确认失效则通过真实业务方法的直接应用演示及 4 个专项回归证明。完整工程回归 104 个通过。正式 Dev 各 0/9、Test 各 0/4；没有改写这些结果，也没有提升幅度。

模型回复不代表执行证据。支付日志、确认回执、其他对象不变与必要沟通分别检查。CLI 与批量任务共用运行时；评测没有直接安排参考工具链。没有新增多 Agent、MCP、训练、网页或部署服务。

## 许可证与公开内容

[react-agent MIT 通知](../LICENSES/react-agent-MIT.txt) 和 [τ2-bench MIT 通知](../vendor/tau_retail/source/LICENSE) 均保留。仓库 LICENSE 表明上游仓库许可，但原始数据文件没有独立文件级许可声明；本项目记录来源、固定 commit、改动和这一限制，不扩大解释为所有数据再分发权已独立核定。用户已明确同意公开固定模拟数据和全部派生题以便复现。

RetailFlow 新增代码尚未选定单独许可证，不因 GitHub 公开就自动变成 MIT；未经用户选择，不擅自给全部原创文件加许可证。公开模拟姓名、订单和付款标识属于特意保留的上游内容，不应替换为真实客户数据。

未修改 ScholarLens、共享环境或 Ollama 配置，未修改简历。可用于展示的是有证据的工程机制和失败复盘；不能把未经口述验收的理解或独立实现能力写成已掌握。
