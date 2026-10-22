# T3 业务工具、权限与确认

日期：2026-10-20。阶段目标：在不依赖模型的情况下，让固定版 retail 查询、取消和退货工具通过同一生产接口运行，并验证未授权写入为零。T3 工程验证已完成，代码实现、测试和排障由 AI 助手辅助完成，不作为独立编码或个人掌握程度的证明。这里的通过率是工程测试，不是 Agent 业务任务成功率。

## 输入、输出与状态流

`RetailEnvironment.create_session(thread_id, initial_state_ref)` 检查原始 `db.json` 的 SHA-256 后，把它复制进该线程的 SQLite 业务快照；源文件始终只读。`dispatch_tool(thread_id, name, args)` 严格校验八个固定工具的名字、参数键和类型，要求先用邮箱或姓名+邮编定位用户；查询个人资料和订单时核对所有权。返回值含 `ok`、`code`、`message` 或 `data`。业务方法本身是固定版上游 `RetailTools` 的原方法，导入适配和修改清单见 `UPSTREAM.md`。

取消与退货的第一次请求在**丢弃的业务副本**上调用上游方法验证政策，只保存提案，返回 `CONFIRMATION_REQUIRED`，不改变业务快照。应用层用户对当前 `proposal_id` 显式确认后，同名且参数完全相同的第二次请求才在新副本上执行一次。提案绑定线程、工具名及规范化参数，15 分钟过期；拒绝、换参数、换目标或过期会使旧确认失效。`confirm_action` 不在八个模型可调用工具内，模型不能用工具参数填写 approval。

SQLite 的同一事务提交更新后的业务 JSON 和操作回执；操作键绑定线程、提案及参数。重复请求先返回原回执，不再次取消、退款或申请退货。工具在复制状态上先变更、后抛错时，原快照保持不变。`state_hash` 只计算业务 JSON，不计提案或回执。这个设计处理本地重放窗口，不宣称跨模型 checkpoint 的分布式 exactly-once；后者仍需 T5 恢复验证。

错误分为 `INVALID_ARGUMENT`（schema）、`AUTH_REQUIRED`、`NOT_FOUND`、`POLICY_DENIED`、`CONFIRMATION_REQUIRED`、`INVALID_CONFIRMATION` 与 `TOOL_ERROR`。原始政策中 pending 取消原因仅限 `no longer needed`/`ordered by mistake`，delivered 退货需原付款方式或用户现有礼品卡；具体判断仍由上游业务方法执行。`transfer_to_human_agents` 在 T3 仅返回上游本地标记，不会向真人发送消息；是否该转人工须由 T4 对话流程和后续评测核对。

## 验证与真实问题

运行项目 `.venv` 中的 `python -m pytest tests/test_tools.py tests/test_write_guard.py -q`：22/22 通过。测试使用固定真实 `db.json`、上游 Pydantic 模型和业务方法，覆盖身份与所有权、八工具白名单、非法状态/退货方式、两个线程隔离、源文件哈希、取消与退货、礼品卡退款、拒绝/过期/跨线程/更换订单或商品确认、回执重放和中途抛错不落盘。随后 fresh run 全项目测试 40/40，通过 `compileall` 和 T2 `validate`（Dev 9/Test 4，错误 0）。

独立演示用 Yara Muller 的 `#W5056519`：身份定位返回 `yara_muller_8652`；第一次取消返回 `CONFIRMATION_REQUIRED`，确认前业务哈希未变；明确确认后订单为 `cancelled`，支付历史从 1 条增至 2 条；同参数重放返回完全相同的回执。演示使用临时 SQLite 文件，没有改动源 DB。

排障中有三项值得复盘：其一，测试里误写了用户邮箱编号，真实上游查询正确返回 `NOT_FOUND`；核对原始用户记录后修正测试数据，没有更改工具逻辑。其二，确认旧提案后请求另一用户的订单时，旧确认最初仍有效；测试复现后把旧提案失效移动到所有权检查之前。审查又发现读取同账户另一订单或识别另一用户也会保留旧确认；两个回归测试先失败，再于目标或身份上下文变化时清除提案，重跑后旧确认无法再获批。其三，方法调用次数测试发现已确认写入原先多跑一次丢弃副本预检；现只在提案阶段预检，获批时执行业务方法一次。

T3 不调用 Ollama、不运行 T2 的 13 道模型任务，也不证明模型会正确选工具。T4 要把这一生产 dispatcher 接入 ReAct 回路和 CLI；T5 再检验跨进程恢复、消息配对与写入结果未知时的处理。
