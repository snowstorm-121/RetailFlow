# 业务演示指南

这里提供两条真实模型交互路径和一条不调用模型的业务保护层示例。数据均来自固定 retail 模拟环境，不连接真实订单或支付。

## 环境检查

在仓库根目录执行：

```bash
.venv/bin/python --version
ollama list
curl -sS http://127.0.0.1:11434/api/version
.venv/bin/python -m retailflow.cli validate --manifest data/manifests/dataset.json
```

模型标签应包含 `qwen3:4b-instruct-2507-q4_K_M`。缺少依赖、服务或模型时先完成[首页的环境准备](../README.md#快速开始)。下列命令不自动安装或下载模型。

`--thread` 标识会话：同 ID 恢复已有业务快照，新 ID 创建独立副本。配置中的会话预算包含进程退出时间，过期后需新建会话。

## 查询、提案、确认与取消

```bash
DEMO_THREAD="demo-cancel-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python -m retailflow.cli chat \
  --thread "$DEMO_THREAD" --config configs/local.json \
  --message 'I am Yara Muller, ZIP 85041. Please cancel order #W5056519 because I ordered it by mistake.'
```

观察身份定位、订单查询和取消工具请求。等待状态应为 `needs_confirmation`，此时只是提案，订单尚未取消。核对工具名、订单号与原因；一致再输入 `yes`，否则输入 `no`。

要检查跨进程恢复，可在等待确认时按 Ctrl-D 退出，再运行：

```bash
.venv/bin/python -m retailflow.cli chat \
  --thread "$DEMO_THREAD" --config configs/local.json
```

不要追加新 `--message`，普通新消息会使尚未提交的提案失效。恢复后核对同一提案，再确认。最终检查工具返回、订单状态与回执；模型输出“取消成功”不能代替这些证据。

已有演示记录：[取消检查](../reports/evidence/t8/cancel-check.json)、[轨迹映射](../reports/evidence/t8/trace-index.json)。它们来自历史执行，不代表当前命令已重新运行。

## 参数变化使旧确认失效

```bash
.venv/bin/python examples/confirmation_change.py
```

该示例直接使用真实业务保护层，不调用模型：

1. 创建取消原因为 `ordered by mistake` 的提案并记录确认，暂不提交。
2. 把原因改成 `no longer needed`，创建新提案。
3. 使用旧提案确认得到 `INVALID_CONFIRMATION`，订单不变。
4. 确认新提案后才提交；重开存储并重放同一请求时读取回执，不再增加支付记录。

原始执行记录见[参数变更检查](../reports/evidence/t8/confirmation-change.json)。这验证的是应用约束，不是模型自主理解需求变更。

## 已送达订单不能取消

```bash
.venv/bin/python -m retailflow.cli chat \
  --thread "demo-policy-$(date +%Y%m%d-%H%M%S)" --config configs/local.json \
  --message 'I am Yusuf Rossi, ZIP 19122. Please cancel order #W2378156 because I no longer need it.'
```

检查模型是否先查询订单，再解释 delivered 状态不能取消。业务快照应保持不变；拒绝说明不应生成写入提案。历史执行状态为 `answered`，并不叫结构化 `refused`。记录见[政策检查](../reports/evidence/t8/policy-check.json)。

## 遇到失败

保存 `status`、`failure_reason` 和 `trace_path`，从第一次偏差往后看：

| 现象 | 检查位置 |
| --- | --- |
| 普通文字追问，没有暂停 | 实际是否产生 `ask_user` 原生请求，以及协议反馈是否用尽 |
| `AUTH_REQUIRED` | 受保护查询之前是否已定位当前用户 |
| `PROTOCOL_ERROR` | 实际原生调用数量与协议约束；说明文本不提供确认权限 |
| `NOT_FOUND` | 用户提供的标识、订单号 `#` 前缀及所有权 |
| 模型请求失败 | 本地服务、调用异常与已有工具结果 |
| 已写入但答复失败 | 先查回执和业务状态，不重复提交写入 |

独立演示使用新 thread，保留失败轨迹，不通过删除数据库或重用修改后的业务快照制造成功。
