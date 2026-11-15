# RetailFlow

本地零售业务 Agent，基于 LangGraph、Ollama 和固定版 τ2-bench retail 模拟环境，支持订单查询、取消与退货。

[业务演示](#业务演示) · [主要改造](#主要改造) · [执行流程](#执行流程) · [快速开始](#快速开始) · [Langfuse 可观测性](#langfuse-可观测性) · [设计说明](docs/architecture.md)

项目把模型决策和业务提交分开：模型选择工具、提出操作；应用检查身份和权限，展示具体提案，收到用户确认后才修改订单。SQLite 保存业务快照与执行回执，用于进程恢复和重复执行保护。

这是运行在 Apple M5 / 16 GB Mac 上的实验原型。订单、用户和支付记录均为模拟数据，不连接真实电商或支付系统。

## 业务演示

取消一笔待处理订单，沿同一条链路观察查询、提案、确认与提交：

```bash
DEMO_THREAD="cancel-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python -m retailflow.cli chat \
  --thread "$DEMO_THREAD" --config configs/local.json \
  --message 'I am Yara Muller, ZIP 85041. Please cancel order #W5056519 because I ordered it by mistake.'
```

写工具第一次请求只生成提案。CLI 展示订单和原因，等待用户输入 `yes` 或 `no`；模型本身没有确认权限。确认时必须对应当前提案和原参数。

可以在等待确认时退出进程，再用同一个 thread 恢复：

```bash
.venv/bin/python -m retailflow.cli chat \
  --thread "$DEMO_THREAD" --config configs/local.json
```

恢复已有会话时不传新 `--message`。同一个 thread 使用自己的业务副本；新的独立演示使用新 thread。

另一个关键案例是参数变化：只修改取消原因，旧确认应失效，新提案必须重新确认。仓库提供不调用模型的业务保护层示例：

```bash
.venv/bin/python examples/confirmation_change.py
```

[完整演示与结果检查](docs/demos.md)。实际模型可能选错工具或提前结束，业务是否完成要看工具结果与最终状态，不能只看回复文字。

## 主要改造

复用 [react-agent](https://github.com/langchain-ai/react-agent) 的模型—工具循环模板，以及 [τ2-bench](https://github.com/sierra-research/tau2-bench) 的 retail 数据、政策和业务方法。新增实现集中在执行约束与会话管理：

| 设计 | 解决的问题 | 代码入口 |
| --- | --- | --- |
| 显式操作提案 | 防止模型替用户确认，或沿用已经变更的订单、商品与原因 | [environment.py](src/retailflow/environment.py) |
| 业务状态与回执同事务提交 | 已写入后进程退出，恢复时不重复提交同一操作 | [storage.py](src/retailflow/storage.py) |
| 完整历史与模型上下文分开 | 保留用户要求、必要状态与完整工具消息块，同时控制输入预算 | [context.py](src/retailflow/context.py) |
| 交互和批量运行共用入口 | 避免演示与业务任务执行两套逻辑 | [runtime.py](src/retailflow/runtime.py) |
| 原生工具协议处理 | 缺信息时暂停交互；普通文本不冒充工具调用或执行结果 | [运行时](src/retailflow/runtime.py)、[图节点](src/react_agent/graph.py) |
| Langfuse 运行轨迹 | 按会话查看模型调用、工具节点与失败原因，并关联本地日志 | [observability.py](src/retailflow/observability.py) |

## 执行流程

```mermaid
flowchart LR
    U[用户消息] --> R[恢复会话与业务快照]
    R --> C[构建模型上下文]
    C --> M[本地模型选择工具]
    M --> G[参数 / 身份 / 权限检查]
    G -->|查询结果| C
    G -->|缺信息| I[暂停并等待用户输入]
    I --> R
    G -->|写操作| P[展示具体提案]
    P --> H[用户确认]
    H --> W[事务提交业务与回执]
    W --> C
    M --> A[答复或停止]
```

模型可见八个业务工具及一个 `ask_user`。`confirm_action` 不向模型暴露；写入结果未知时不自动重试写操作。恢复与重复执行保护仅针对本地事务和已测试路径，不是分布式 exactly-once 保证。

## 快速开始

参考环境：Python 3.12.15、uv、Ollama 0.15.6，模型为 `qwen3:4b-instruct-2507-q4_K_M`。仓库锁定依赖，不要求云 API Key、Docker 或 PyTorch。

```bash
git clone https://github.com/snowstorm-121/RetailFlow.git
cd RetailFlow
uv sync --locked --python 3.12
ollama list
curl -sS http://127.0.0.1:11434/api/version
.venv/bin/python -m retailflow.cli validate --manifest data/manifests/dataset.json
```

`uv sync` 可能联网下载解释器和依赖；Ollama 服务和完整模型标签需另行准备，项目不会自动下载模型。环境就绪后运行上面的业务演示。工具定义和模拟业务为英文，演示也使用英文。

## Langfuse 可观测性

已接入 **Langfuse Cloud 日本区**，用于复查 Agent 的实际执行过程。模型仍由本地 Ollama 推理；Langfuse 接收运行轨迹，不参与模型推理或业务决策。上报默认关闭，可按需启用。

| 观察层级 | 可以查看的内容 |
| --- | --- |
| Session | 同一个 thread 的多轮交互，便于追踪追问、提案和确认过程 |
| Trace | 每次用户事件的输入、最终状态与失败原因；Trace ID 与本地 JSON 对应 |
| 模型调用 | ChatOllama 输入输出、耗时，以及模型提供的 token 用量 |
| 工具执行 | LangGraph 工具节点、工具请求与结果；图外确认写操作记录在根节点的执行回执中 |

例如，订单查询中途停止时，可以沿 Trace 查看模型选择了什么工具、工具返回了什么，再对照最终失败原因定位问题。工具按图节点展示，不将每个 Python 方法包装成独立 tool span。

### 启用方式

在项目根目录安装可选依赖（可能联网下载依赖）：

```bash
uv sync --locked --extra observability --python 3.12
```

参照 [.env.langfuse.example](.env.langfuse.example) 创建本机 `.env.langfuse`；已有文件直接编辑，不覆盖。填写自己 Langfuse 项目的 Public Key、Secret Key，服务地址使用 `https://jp.cloud.langfuse.com`。密钥文件被 Git 忽略。

```bash
# 仅检查项目鉴权，不调用模型或上传业务轨迹
.venv/bin/python examples/langfuse_check.py

# 应用不会自动读取 .env 文件，需在当前终端加载
set -a
source .env.langfuse
set +a

.venv/bin/python -m retailflow.cli chat \
  --thread "langfuse-demo-$(date +%Y%m%d-%H%M%S)" \
  --config configs/local.json --message 'Hello.'
```

运行后，用 CLI 输出的 Trace ID 在 Langfuse 中定位记录；同一会话继续交互时复用该 thread。关闭上报使用 `export RETAILFLOW_LANGFUSE_ENABLED=false`。

日本区已完成真实本地模型轨迹上传与 Cloud API 回读验证，包括 Session 关联、父子节点、本地 Trace ID 对应及失败状态记录。观测链路可用不代表业务任务成功；业务结果仍以实际状态和回执为准。

启用后，消息、模型上下文和模拟业务数据会上传到 Cloud；当前没有自动个人信息脱敏，不应输入真实客户资料或秘密。上报失败不会自动重放业务操作，本地日志仍保留。完整配置、记录范围和故障边界见 [Langfuse 接入指南](docs/langfuse.md)。

## 设计与源码

- [执行架构与设计取舍](docs/architecture.md)：原生调用、提案、事务、恢复与上下文。
- [Langfuse Cloud 接入](docs/langfuse.md)：可选会话追踪、模型调用与工具节点观测。
- [演示操作指南](docs/demos.md)：取消、参数变更、政策拒绝，以及常见故障。
- [来源与二次开发](reports/contribution.md)：框架、业务方法与新增模块的范围。
- [固定上游清单](UPSTREAM.md)：版本、适配文件与许可信息。

`src/react_agent/` 是模板改造层，`src/retailflow/` 是业务执行与会话层；`tests/` 包含确定性回归。`runtime/`、`artifacts/`、模型和虚拟环境不提交 Git。历史实验记录保留在 `reports/`，与本文的演示说明分别维护。

## 已知限制

当前入口是 CLI，没有网页或真实平台接入。小模型仍可能错误追问、遗漏用户要求或生成没有工具依据的完成声明。应用保护能限制部分副作用，不能代替正确的规划和业务沟通。

上下文 token 预算采用字符估算；身份检查遵循模拟环境政策，不是生产账户认证。项目没有生产可用性或全面安全承诺。

上游许可证保留在 [LICENSES/](LICENSES/) 和 [τ2-bench 来源目录](vendor/tau_retail/source/LICENSE)。新增代码尚未选择独立许可证；开发与辅助审查说明见[来源文档](reports/contribution.md)。
