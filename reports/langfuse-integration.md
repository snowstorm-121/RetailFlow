# Langfuse 接入验证

日期：2026-11-15。固定 SDK 5.2.0，以 observability 可选依赖安装至项目独立环境。既有业务依赖未主动升级。日本区地址为 https://jp.cloud.langfuse.com 。

## 已验证

- 默认关闭时不初始化 SDK；启用且缺密钥时在业务执行前提示配置错误。
- 真实 SDK 内存 exporter 能捕获 root Trace、LangGraph 子节点和 Session 关联；导出失败或业务异常不会自动重放业务。
- 真实本地 Ollama 烟测捕获 2 个 ChatOllama generation，同属一个 Trace。本次业务状态为 failed，观测链路正常记录失败；这不是业务成功证明。结果见 [offline-smoke-result.json](evidence/langfuse-integration/offline-smoke-result.json)。该烟测没有向 Cloud 导出。
- 完整工程测试 159 项通过；最后调整初始化失败隔离后，6 项观测专项测试再次通过。
- uv pip check 通过；.env.langfuse 被 Git 忽略，示例文件只有空密钥。

## 尚未验证

本机 Public Key 和 Secret Key 仍为空，未完成日本区项目鉴权、真实上传或控制台可见性验收。不能声称 Langfuse Cloud 已连接成功。

配置项目后，先运行 examples/langfuse_check.py，再按 docs/langfuse.md 加载环境变量并执行一次最小对话。检查控制台 Trace ID、Session 关联和 generation 子节点，才算完成端到端接入。旧实验冻结文件保留不变；新增依赖和代码会改变冻结环境，不能冒称仍为旧实验版本。
