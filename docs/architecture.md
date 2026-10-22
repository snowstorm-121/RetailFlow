# 执行架构与设计取舍

RetailFlow 关注模型提出的操作如何安全地落到模拟业务上。入口是 CLI，但交互与批量任务均使用同一个 `RetailRuntime.run_turn`；模型选择下一步工具，应用决定是否允许执行。

## 一次取消订单经过什么

1. `cli.py` 把用户消息交给 `runtime.py`，按 thread 恢复消息、预算和业务快照。
2. LangGraph 节点调用本地 ChatOllama，绑定原生工具 schema。
3. 模型定位用户并读取订单，工具结果作为 ToolMessage 回到循环。
4. 取消工具先校验政策并创建提案，原订单暂不改变。
5. 应用展示提案，等待用户确认具体订单和参数。
6. 确认后在 SQLite 事务中提交业务变化和执行回执，再把实际结果交给模型说明。

主要入口：[cli.py](../src/retailflow/cli.py)、[runtime.py](../src/retailflow/runtime.py)、[graph.py](../src/react_agent/graph.py)。

## 为什么确认不交给模型

提示词中的“请先征得同意”不是授权机制。模型可能把普通答复当成确认，也可能在参数变化后沿用先前同意。

`confirm_action` 不在模型可见工具表中。应用把确认绑定到 thread、当前 `proposal_id`、工具名和原参数。换订单、商品或原因会使旧确认失效。具体实现见 [environment.py](../src/retailflow/environment.py)，可运行案例见 [confirmation_change.py](../examples/confirmation_change.py)。

预检在可丢弃的业务副本中执行；只有通过确认后的提交才改变正式会话快照。

## 已经提交，但回复失败了怎么办

业务 JSON 与执行回执在同一个 SQLite 事务中保存。恢复时先查回执；如果同一操作已经提交，就返回原结果，而不是因为模型回复超时再执行一次。

提交只合并业务方法前后的实际差异，避免 Pydantic 补入的默认字段污染无关订单。完整消息、业务状态和模型调用并不处于一个全局事务中，这个方案不等于跨系统 exactly-once。

实现见 [storage.py](../src/retailflow/storage.py) 和 [environment.py](../src/retailflow/environment.py)。

## 完整历史不等于模型上下文

完整用户消息、工具调用与结果落盘，便于恢复和问题回查；模型每次只接收受控视图。

- 有界会话中的用户要求与更正按顺序保留，避免多轮执行后忘记原目标。
- 工具请求和对应结果成组处理，不留下孤立 ToolMessage。
- 工具返回按必要业务字段裁剪，已提交结果由应用提供事实摘要。
- 必需信息超出预算时停止，不静默丢掉当前目标。

预算使用字符估算，不是精确 tokenizer 计数。SQLite 使用自定义表，不是 LangGraph 原生 SqliteSaver。实现见 [context.py](../src/retailflow/context.py) 与 [storage.py](../src/retailflow/storage.py)。

## 原生工具调用与交互恢复

Function Calling 返回的是工具名称与参数，真正执行 Python 方法的是应用。普通文字“已经取消”不会改变订单；文字追问也不会自动伪造成工具调用。

`ask_user` 只接受已定义的缺失字段。模型用文字追问而没有发出原生调用时，运行时可以提供有界协议反馈，仍要求模型重新生成实际调用；反馈计入调用预算，不作为用户身份或执行证据。身份参数需来自实际用户消息。

只对明确发生在执行前的只读临时故障有限重试。权限拒绝、政策拒绝和结果未知的写入不走盲目重试。小模型仍可能在这些约束下提前结束或选错工具。

实现见 [runtime.py](../src/retailflow/runtime.py)、[tools.py](../src/retailflow/tools.py)、[工具图](../src/react_agent/graph.py)。

## 怎样检查一次执行

沿实际轨迹依次核对用户要求、工具参数、工具返回、确认事件、执行回执和最终业务状态。自然语言答复、参数格式正确、业务状态正确是不同检查项，不能互相替代。

`evaluation.py` 将业务末态、必要沟通和禁止副作用分开处理，用户脚本与评分标签不进入模型的规划上下文。确定性测试用于验证代码约束，不代替真实模型业务验证。

相关代码：[dataset.py](../src/retailflow/dataset.py)、[user_script.py](../src/retailflow/user_script.py)、[evaluation.py](../src/retailflow/evaluation.py)。
