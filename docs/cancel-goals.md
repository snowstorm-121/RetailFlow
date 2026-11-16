# 取消目标登记实验

本配置默认不启用，范围限定为“全部待处理订单”或用户直接给出订单编号的取消请求。它不是通用多目标规划器，不支持将“取消某种商品”自动解析为订单目标，也不要用于退货或任意查询。

## 数据流

用户消息 → 原生 update_cancel_goal → 已验证账户的订单清单 → 待完成订单 → 原生取消提案 → 用户逐笔确认 → 订单状态与成功回执 → 完成检查。

模型登记 scope、order_ids、source_quote。scope 支持 none、all_pending、selected；selected 必须包含用户原话中的完整订单号。程序验证结构和原话来源，不声称已证明自然语言解释正确。模型错误理解范围仍可能造成遗漏或无效提案，必须由原确认规则和最终任务检查继续约束。

每条新用户消息重新登记目标，身份信息、原因回答也计入这一步。若登记失败，不进入后续业务。确认事件沿用已有目标，不重新猜测范围。每一笔订单的提案分别确认；登记目标不能代替确认。

全部待处理订单的目标集合在用户身份已确认、get_user_details 成功返回清单后，从该用户的业务快照中冻结。只计入本人当时 pending 的订单。后续订单状态变更不会让目标自动消失。同一范围的重新登记保留冻结集合，真正改变范围或撤销时才重建。目标只有在实际状态为 cancelled 且同一会话存在成功取消回执时才完成；无法访问的订单保持未完成，不暴露他人状态。

目标记录复用 SQLite runtime_state 的 cancel_goal 字段，包含 needs_refresh、scope、order_ids、source_quote、targets。没有新增数据库服务。新用户消息会使旧计划待更新，并由原会话逻辑使旧提案失效。重启后可以沿同一 thread 恢复。

## 运行

使用已有 Ollama 和项目环境：

```bash
.venv/bin/python -m retailflow.cli chat \
  --thread cancel-goals-demo --config configs/cancel-goals-local.json \
  --message 'I want to cancel all of my pending orders.'
```

按实际提示提供身份、取消原因，再逐笔核对提案。下一次恢复同一演示时保留 thread，省略 --message。独立演示使用新 thread。

此配置固定取消目标实验开关及已有信息/提案恢复开关，仍使用原本地模型。目标登记也消耗原模型调用预算，不增加预算、不忽略失败。不将普通文本或 JSON 作为可执行工具请求。

## 失败边界

- 非原生登记：cancel_goal_not_registered。
- 非法范围、空指定列表或引用并非用户原话：invalid_cancel_goal。
- 有未完成目标但模型试图结束：在原恢复预算内尝试一次；仍未完成返回 cancel_goal_incomplete。
- 缺少用户信息仍允许 ask_user；不能把合理追问直接改成写操作。
- 已登记为 none 或超出冻结目标的取消请求不能创建提案。

这是一层执行约束，不保证模型完成任务。当前退货反例曾产生非法登记，说明不能将其默认为所有业务的入口。引用核验也不是通用语义验证或安全认证。
