# Langfuse 接入验证

日期：2026-11-15。固定 SDK 5.2.0，以 observability 可选依赖安装至项目独立环境。既有业务依赖未主动升级。日本区地址为 https://jp.cloud.langfuse.com 。

## 已验证

- 默认关闭时不初始化 SDK；启用且缺密钥时在业务执行前提示配置错误。
- 真实 SDK 内存 exporter 能捕获 root Trace、LangGraph 子节点和 Session 关联；导出失败或业务异常不会自动重放业务。
- 真实本地 Ollama 烟测捕获 2 个 ChatOllama generation，同属一个 Trace。本次业务状态为 failed，观测链路正常记录失败；这不是业务成功证明。结果见 [offline-smoke-result.json](evidence/langfuse-integration/offline-smoke-result.json)。该烟测没有向 Cloud 导出。
- 完整工程测试 159 项通过；最后调整初始化失败隔离后，6 项观测专项测试再次通过。
- uv pip check 通过；.env.langfuse 被 Git 忽略，示例文件只有空密钥。

## 日本区 Cloud 实测

用户配置本机密钥后，`examples/langfuse_check.py` 项目鉴权通过。随后在独立业务数据库和新 Session 中执行一次真实订单查询，使用已有本地 `qwen3:4b-instruct-2507-q4_K_M`，没有使用云模型。

上传后通过 Langfuse Cloud observations API 回读验证，而非仅凭 SDK flush 返回判断成功：

- 共 18 条观测，包括 4 个 ChatOllama generation、2 个 tools 图节点及 root。
- 所有观测属于同一 Trace、同一 Session；所有非 root 节点的父节点均存在。
- 本地 JSON 的 Trace ID 与 Cloud 一致；generation 包含实际 token 用量。
- root 正确记录 ERROR、业务状态 failed 和 `information_request_not_structured`。本次接入验证成功，但订单查询任务失败，不代表业务指标提升。
- Cloud 观测均为非公开。公开仓库仅保留不含密钥、项目标识和业务原文的验证摘要：[cloud-smoke-result.json](evidence/langfuse-integration/cloud-smoke-result.json)。完整回读数据保存在被忽略的本地 artifacts 目录。

本次已完成鉴权、实际上传及 Cloud API 可读验收，未进行浏览器控制台视觉验收，也未重新运行完整业务评测。旧实验冻结文件保留不变；新增依赖和代码会改变冻结环境，不能冒称仍为旧实验版本。
