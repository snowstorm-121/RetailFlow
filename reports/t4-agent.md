# T4 本地模型业务循环与 CLI

日期：2026-10-20。工程目标：在 T1 固定版 LangGraph ReAct 模板旁接入 T3 的同一业务 dispatcher，让本地模型逐步选工具，CLI 能追问、暂停确认和结束；不运行 T2 的正式任务评测。

## 入口与数据流

`python -m retailflow.cli chat --thread <id> --config configs/local.json` 可交互运行；加 `--message` 可提交首条消息。CLI 将消息或显式的 `yes/no` 确认事件交给 `retailflow.runtime.run_turn`。运行时从独立 SQLite 会话取完整消息，构建 retail 专用 `StateGraph`，调用本地 `ChatOllama.bind_tools`（8 个业务工具加 `ask_user`）。每轮只允许一项工具调用；AIMessage 的原始参数及 ID 与对应 ToolMessage 一起保存。`ask_user` 和 `CONFIRMATION_REQUIRED` 会结束当前图回合。写操作仅由 T3 保护层生成提案，CLI 显示具体参数并接收 `yes/no`；模型不能调用 `confirm_action`。确认后应用执行原提案，重新调用模型解释结果。

T1 的烟测图和三项小工具保留。retail 图因需在工具结果处安全暂停、拒绝多调用和混合文本，使用同一 `StateGraph` 模型→工具→模型循环及受控工具节点；没有另造固定的业务执行链。会话消息存于 SQLite `conversations` 表，原始业务状态仍由 T3 的独立快照管理。T5 还需验证故障窗口和长上下文恢复，不能仅凭此处持久化声称完全恢复能力。

## 本地验证证据

本机 Ollama 服务 0.15.6；模型 `qwen3:4b-instruct-2507-q4_K_M`，digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`，已有模型约 2.50 GB。本项目 Python 包：LangGraph 1.2.16、langchain-core 1.6.12、langchain-ollama 1.1.0、Pydantic 2.14.1。`configs/local.json` SHA-256 为 `743883b026a1646048c49715c79c0a2b23e061d93c614ad100bc4d6ab240c2f7`；固定业务 DB SHA-256 为 `413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765`。无下载、安装、云 API 或共享 Ollama 配置改动。

脚本模型测试 `python -m pytest tests/test_agent_loop.py -q`：10/10 通过，覆盖单工具与原始 call ID、两步依赖、`ask_user` 后新输入、多调用和“工具调用＋直接答复”不部分执行、明确确认、预算结束、无工具证据的提案声称拦截、转人工仅本地标记和 CLI 确认选项。全项目 `pytest -q`：50/50；`compileall` 成功；T2 `validate` 为 Dev 9/Test 4、错误 0。这些测试不代表真实模型业务任务成功率。

真实本地演示均使用全九工具 schema，以下是观察到的选择，不是固定脚本：

| 路径 | 轨迹与业务末态 | 结论 |
|---|---|---|
| 身份缺失查询 | `t4-query-20261020-b`：先 `ask_user`，用户提供姓名和 ZIP 后定位身份；模型首次将 `#W5056519` 错传为 `W5056519`，得到 `NOT_FOUND`；用户纠正前缀后读取正确订单，末态仍 pending。公开轨迹：[追问](../examples/traces/query-needs-identity.json)、[错误参数](../examples/traces/query-dropped-prefix.json)、[用户纠正后](../examples/traces/query-recovered.json)。独立直接查询 `t4-query-20261020-g` 使用身份定位→订单查询→答复，原轨迹留在本机 `artifacts/t4/20261020T063636Z-e72d78b4.json`。 | 多轮查询可完成，但缺失前缀的错误需要用户纠正；不能记为自动恢复。 |
| 显式确认后取消 | `t4-cancel-20261020-f`：身份定位→订单查询→取消工具提案；确认前业务哈希未变，CLI 收到明确 `yes` 后订单从 pending 变 cancelled，支付历史 1→2，答复正确说明信用卡退款需 5–7 个工作日。公开轨迹：[提案](../examples/traces/cancel-proposal.json)、[确认后](../examples/traces/cancel-confirmed.json)。 | 一条完整本地写流程通过；模型选取了三个业务工具。 |
| 政策拒绝 | `t4-refuse-20261020-a`：身份定位→查询 `#W1679211`，识别 delivered 并解释不能取消；订单仍 delivered，支付历史仍 1，公开[轨迹](../examples/traces/cancel-refused.json)。 | 无禁止副作用；当前状态是 `answered`，而非结构化 `refused`，需在后续评分中按文本、工具与业务状态共同判断。 |

最终查询 `t4-query-20261020-g` 记录模型 3 次、工具 2 次；模型请求 15.219 秒、工具执行 0.066 秒、状态检查 0.058 秒，输入 3665 / 输出 197 tokens，停止原因 `answered`，业务哈希不变。实际用量来自模型返回；缺失用量的回合保留 null，不编造数字。

## 真实问题和边界

九工具 schema 下，真实轨迹观察到 `ask_user`、`find_user_id_by_name_zip`、`get_order_details`、`cancel_pending_order` 四种调用；其余五种没有在 T4 演示中覆盖。最初模型漏掉订单号前导 `#`；在 schema 中加入格式提示后，新的查询轨迹使用正确参数。取消试验中，模型有时只用文本要求确认、未调用写工具；[一条公开失败轨迹](../examples/traces/cancel-text-only-failure.json)可复查，`t4-cancel-20261020-a/b/d/e` 均保持原订单 pending。`c` 完成取消但跳过显式订单查询，并把信用卡退款说成已处理；补充政策和工具说明后，`f` 完成订单查询、提案、确认与准确退款说明。这些试验的提示版本与用户措辞不同，不能把次数换算成稳定成功率或与 T2 Test 成绩比较。

一次模型仅用文字声称“提案已创建”却无写工具结果。运行时现将这类明显无证据的声明标记为 `failed/unsupported_action_claim`，不展示为成功操作；普通 `answered` 仍仅表示模型给了答复，不是业务成功判定。T5/T6 需继续处理长对话、故障恢复与按末态评分。模型请求中途失败后的图消息恢复尚未验收。
