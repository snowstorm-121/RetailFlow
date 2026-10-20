# 三条演示与故障定位

这是演示操作单，不是新增评测集。真实模型单例与直接业务保护测试分开；不把重放轨迹当作现场推理，不把它们加入 Dev/Test 成功数。已有模型也可能失败，一次失败就保存轨迹并解释，不循环改问法直到挑出成功。

## 启动前

在项目根目录执行，使用已有独立环境；这些命令不安装依赖：

```bash
pwd
.venv/bin/python --version
.venv/bin/python -m retailflow.cli validate --manifest data/manifests/dataset.json
```

`CLI` 是 Command-Line Interface（命令行界面）。`python -m retailflow.cli` 用指定 Python 解释器运行这个模块；`--thread` 是会话 ID，同 ID 恢复已有状态，新 ID 创建独立业务副本。`--config` 指定本地模型配置；`--message` 提交首条消息。换目录不等于切换解释器，用 `.venv/bin/python` 明确选择本项目环境。

模型必须已在本机 Ollama 中，默认端口 11434。以下所有订单和付款是上游模拟数据，不连接真实电商。每次演示使用新 thread，不删除旧 SQLite 或重置别人的会话；确认提案需要在 15 分钟预算内完成。已有环境缺包/模型时停止定位，不在演示中自动下载。

## 1. 真实模型：查询、提案、确认取消

```bash
DEMO_THREAD="demo-cancel-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python -m retailflow.cli chat --thread "$DEMO_THREAD" --config configs/local.json --message 'I am Yara Muller, ZIP 85041. Please cancel order #W5056519 because I ordered it by mistake.'
```

观察模型是否真的依次调用定位身份、查询订单、取消工具。正确暂停为 `needs_confirmation`，此时只创建提案，业务订单仍 pending。核对显示的工具名、订单号、原因后，当前模拟演示输入 `yes`；不匹配就输入 `no`。输入 `/quit` 结束交互。

想演示跨进程恢复，可以在提案提示处按 Ctrl-D（EOF，End Of File，结束当前输入），然后运行：

```bash
.venv/bin/python -m retailflow.cli chat --thread "$DEMO_THREAD" --config configs/local.json
```

它应先显示同一个待确认操作；核对后再输入 yes。不能换 thread 后声称恢复成功。真实业务结果以 SQLite/轨迹中的执行结果为准，不能只看自然语言。

2026-10-20 实际使用独立 `runtime/t8-live.sqlite`，两个 CLI 进程完成：提案阶段模型/工具 3/3；确认阶段模型/dispatcher 尝试 2/2（其中一次是应用执行取消，另一次是模型查询）。最终 cancelled、支付记录 1→2，完整末态检查通过；答复准确说明信用卡退款需 5–7 个工作日，没有称已到账。见 [取消检查](../reports/evidence/t8/cancel-check.json) 和 [实际轨迹映射](../reports/evidence/t8/trace-index.json)。这次提供了完整身份，不能用来掩盖缺身份开场的正式评测失败。

## 2. 应用保护层：只改一个输入字段，旧 yes 失效

```bash
.venv/bin/python examples/confirmation_change.py
```

该脚本**不调用模型**。它直接调用 Agent 使用的真实 `RetailEnvironment` 和固定上游业务方法，创建独立输出目录，按下面顺序验证：

1. 定位 Yara、查询订单，请求 `reason='ordered by mistake'` 的取消并记录确认，尚未执行写入。
2. 仅将输入参数 `reason` 改成 `no longer needed`，再次请求取消。
3. 新请求必须生成不同 proposal_id。拿旧 ID 确认，必须得到 INVALID_CONFIRMATION；此时订单仍 pending，业务哈希未变。
4. 明确确认新 ID 后取消才提交；重开存储重放同参数，必须取原回执，支付记录仍为 2 条。检查其他订单、用户、产品未改变。

脚本断言失败会报错，不打印预设成功答案。实际 [结果](../reports/evidence/t8/confirmation-change.json) 保留旧/新参数、不同提案 ID、拒绝代码和严格末态检查。这是“需求参数变更后重新确认”的工程证明，不能描述成模型自主理解需求变更。已有回归也覆盖切换订单/退货商品：

```bash
.venv/bin/python -m pytest tests/test_write_guard.py -k 'changed_parameters or invalid_new_target or reading_another_order or changed_return_item' -q
```

实际 4 passed，9 deselected；deselected 表示本次按名字筛掉，其余测试未在这个专项命令中执行，不是失败。

## 3. 真实模型：已送达订单不能取消

```bash
.venv/bin/python -m retailflow.cli chat --thread "demo-policy-$(date +%Y%m%d-%H%M%S)" --config configs/local.json --message 'I am Yusuf Rossi, ZIP 19122. Please cancel order #W2378156 because I no longer need it.'
```

2026-10-20 实际模型定位身份→查订单→解释 delivered 不能取消，3 次模型请求、2 次工具调用；未调用取消写工具、未生成提案，完整业务快照相等。见 [政策检查](../reports/evidence/t8/policy-check.json)。运行时状态是 `answered`，不是结构化 `refused`；不能为了演示把状态名称改写。另有固定回归验证业务层对非法状态返回 POLICY_DENIED。

拒绝可以是正确业务结果，但正式任务若另有必要转交或沟通要求，仍需检查那些目标。T7 的零成功结论没有因此改变。

## 当现场失败时

先记 `status / failure_reason / trace_path`。打开轨迹，沿原始 messages 找第一处偏差：用户实际给了什么 → 模型发了哪个工具和参数 → 对应 ToolMessage 的代码 → 是否有确认与提交结果。`call_id` 用来将一次请求与它的结果对应，不是订单号。

| 观察 | 定位证据 | 处理边界 |
| --- | --- | --- |
| 普通文字追问，未出现 ask_user | AIMessage 没有对应工具请求；T6 典型失败 | 保留失败，不把文字改造成伪 Function Calling |
| AUTH_REQUIRED | 定位身份之前调用了受保护工具 | 需要先取得当前用户身份，不替换他人身份绕过 |
| PROTOCOL_ERROR | 同时输出说明文字和工具请求，或多工具请求 | 查原始消息；格式合法不代表协议允许 |
| NOT_FOUND | 参数是否保留 #、用户是否给错标识 | 按信息澄清；不碰运气查询他人订单 |
| timeout / connection_error | 先确认请求是否到本机服务，再看历史工具结果 | 不重启共享服务；写入未知先查回执，不能盲重试写 |
| business committed 但模型回复失败 | 回执、目标状态、支付记录 | 明确报告已提交事实，不重新退款 |

HTTP 200 只表示模型请求收到响应，不代表选对工具或业务成功。现有恢复回归使用注入故障和脚本模型，标明测试性质；本轮没有以故障注入冒充真实模型恢复率。
