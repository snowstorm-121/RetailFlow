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

## 原话与订单索引协议（另一个实验开关）

`cancel_goal_references` 默认 false，只有取消目标登记已启用时才生效。使用 `configs/cancel-goals-references-local.json` 可同时启用两项开关。

索引模式向模型提供编号的用户消息和原文中出现的订单号清单。原生 update_cancel_goal 返回 scope、order_indices、source_message_index；应用严格检查索引类型、范围和重复值，再从原文取回 order_ids 和 source_quote。内部账本格式保持不变。负数、布尔值、字符串索引、越界索引和重复订单都会拒绝。

两种登记方式都加入有限的拒绝规则：取消目标的引用需明确包含取消用语；从该引用起的用户消息若包含 `keep #W…`、`do not cancel #W…` 或 `don't cancel #W…` 等支持的明确保留句式，不允许将相应订单登记为取消目标。冲突的 all_pending 目标也会拒绝。这些规则只拒绝，不自动推导其他取消对象；它们不是完整的否定、指代或自然语言语义解析。

模型仍可能选择错误索引或错误理解意图。索引协议解决的是复制原话/订单号时的字符错误，不保证目标理解正确。遇到不支持或冲突的表达，系统可能停止并要求重新表述，仍不能绕过逐笔确认。

## 受限自动启用（完整 Dev 候选配置）

`configs/scoped-cancel-goals.json` 在原 proposal-recovery 配置上开启账本，并设置 `cancel_goal_activation: explicit_all_pending`。只有完整用户消息符合一条受支持命令时才首次进入账本，例如：

- `Cancel all my pending orders.`
- `Please cancel all pending orders.`
- `I want to cancel all of my pending orders.`
- `I'd like to cancel all of my pending orders!`

匹配忽略大小写和首尾空白，可省略句末标点；不接受额外条件、排除项或其他句子。不依赖任务编号、用户编号、订单编号或评测答案。未匹配的请求继续走原业务链路，不代表系统拒绝或判定其没有取消意图。这是有限入口语法，不是通用自然语言分类器，也不是授权或安全检查。

首次进入后，是否启用由已持久化的 cancel_goal 记录保持：补身份、补原因、撤销或改范围不能因当前一句话不匹配而绕开账本。重启同一 thread 仍保留该约束。撤销仍需原生登记 none；若后续改为不受账本支持的业务，原有登记可能失败，不保证任意话题切换。新 thread 才重新从未启用状态开始。

原 `cancel_goal_tracking: true` 未设置该激活模式时仍保持始终登记的旧行为；不修改默认 CLI 开关。模型仍须原生登记目标、读取本人订单、生成提案；应用只在逐笔确认后写入。启用条件本身不生成订单目标、不授权取消，也不增加模型预算。

本次验收顺序：先验证否定/条件/混合请求不被新入口截获及账本重启后仍处理撤销，再真实运行 Dev113；只有完整单题通过才运行九题 Dev，按原语义标准复核，保留失败和副作用记录。
