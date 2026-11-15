# Langfuse Cloud 接入

使用固定 Python SDK `langfuse==5.2.0`，目标日本区 https://jp.cloud.langfuse.com 。模型仍由本地 Ollama 推理，Langfuse 用于存储与查看运行轨迹。默认不启用网络上报。

## 安装与项目密钥

```bash
uv sync --locked --extra observability --python 3.12
```

登录日本区，创建 RetailFlow 项目，在项目 Settings → API Keys 创建 Public Key 和 Secret Key。复制 `.env.langfuse.example` 为 `.env.langfuse`，填写对应字段。若文件已经存在，直接编辑，不覆盖。密钥文件被 Git 忽略，不能发到聊天或放入配置 JSON。

```bash
.venv/bin/python examples/langfuse_check.py
```

该检查读取本地文件中的简单字面量赋值，不执行文件内容，只验证项目鉴权，不调用模型或上传业务轨迹。它成功也不代表实际 trace 已经送达。

## 启用与运行

应用使用环境变量，**不会自动加载 .env 文件**。在自己的终端加载本地文件后运行：

```bash
set -a
source .env.langfuse
set +a
.venv/bin/python -m retailflow.cli chat \
  --thread langfuse-demo --config configs/local.json \
  --message 'Hello.'
```

每次独立业务演示使用新 thread；同一 thread 会恢复会话和业务状态。关闭上报：

```bash
export RETAILFLOW_LANGFUSE_ENABLED=false
```

同进程缓存 SDK 客户端；修改密钥或区域后重启本项目进程。不要重启共享 Ollama。

## 记录范围

每次用户事件对应一条 `retailflow.turn` Trace，同一 thread 通过 session_id 关联多轮。记录范围包括用户消息、确认事件、模型输入输出、LangGraph 节点、工具请求和结果，以及最终状态和失败原因。确认后在图外执行的写操作通过 root Trace 的 tool_attempts 和 confirmed_result 记录。工具节点按图节点展示，不宣称每个自定义 Python 方法都有独立 SDK tool span。

CLI 输出 Trace ID，本地 JSON 同时保存该 ID，可在 Cloud Trace 列表中定位，再从 Session 查看完整会话。SDK 采集的模型 token 取决于提供方是否返回；本地模型费用不自动估算为云模型价格。

Cloud 获得的是运行时上下文和模拟业务数据；本接入不做自动个人信息脱敏，不能直接用于真实客户数据。不会主动上传整个 SQLite 文件、环境变量或配置中的密钥，但用户消息本身若包含秘密也会被上报。历史轨迹不自动回填。

本地完整 JSON、业务数据库和原评分仍是审计依据。接入不改变任务成功标准，也不会把未完成任务标成成功。

## 故障与验证边界

启用但缺密钥时，在业务执行前给出配置错误。SDK 初始化、更新或导出失败时保留本地结果，不自动重放业务操作。短进程每个事件结束 flush；上报可能增加总耗时，比较实验应保持观测配置一致，另记录网络耗时影响。

当前自动化测试使用真实 SDK 的内存 exporter 验证 Session 和父子 span，不向 Cloud 上报；另测试关闭开关、缺密钥、初始化失败、导出失败和业务异常不会导致重复执行。Cloud 鉴权与真实 Trace 可见性必须另行验收，不能用离线测试代替。

参考：[LangChain/LangGraph 集成](https://langfuse.com/integrations/frameworks/langchain)。具体接口以锁定的 SDK 5.2.0 为准。
