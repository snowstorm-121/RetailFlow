# T1 本地模型与原生工具调用验收

日期：2026-10-20。范围：无订单数据、无业务写入的本地工具调用；本报告不代表 retail 任务评测结果。

## 固定环境

- 项目：`/Users/yyy/code/RetailFlow`；独立 CPython 3.12.15 与 `.venv`，均在项目目录内。
- 运行时：Ollama 0.15.6，本地端点 `http://127.0.0.1:11434`。模型 `qwen3:4b-instruct-2507-q4_K_M`，digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`，GGUF Q4_K_M；复用已有约 2.50 GB 的模型文件，未下载模型。
- 模板 commit、许可证、文件及改动清单见 `UPSTREAM.md`。业务环境尚未导入。
- `uv.lock` SHA-256：`99822c1d8d74b026047ace6be5ce0682233cf4227d39fdb0326a0e63d8f7c4f7`。Python 包：langgraph 1.2.16、langchain-core 1.6.12、langchain-ollama 1.1.0、ollama 0.6.3、pydantic 2.14.1、pytest 9.1.1。
- 生成参数：temperature 0、seed 42、num_ctx 8192、num_predict 768、请求超时 120 秒，单并发。没有修改共享 Ollama 配置；LangSmith 追踪关闭。
- 项目内占用：`.tools` 约 151 MiB，`.venv` 约 36 MiB，T1 轨迹约 112 KiB。下载的 uv wheel 为 14.4 MB，Python 包为 23.9 MiB；依赖下载量尚未逐包汇总，未超过先前约定的 300 MiB 下载预算与 2 GiB 磁盘预算。

## 验证结果

命令：

```bash
.venv/bin/python -B -m pytest -q
.venv/bin/python -B -m retailflow.cli smoke-model --config configs/local.json
```

- 最终单测：10 passed，0 warnings。此前先运行失败测试，识别到模板的 Tavily 默认导入；替换成本地小工具后，契约与图路径测试通过。新版 LangGraph 的 ToolNode 单独运行需要图运行时，单测改为沿实际图执行。
- 最终真实烟测：10/10；无需工具 2/2，单工具 4/4，两步依赖 2/2，缺参数澄清 2/2。两道两步任务均经历“查路线分钟数 → 用返回值调用分钟转秒 → 答复”，不预填工具结果。
- 最终轨迹：`artifacts/t1/smoke-20261020T025854Z.json`，另有同时间戳的每题 `*-http.jsonl`。共 18 次模型请求、18 次 HTTP 200 响应、8 次实际工具调用。全部 HTTP 记录的 URL 为 `http://127.0.0.1:11434/api/chat`，请求带原生 `tools` 字段；框架轨迹保留工具名、参数、call_id、对应 ToolMessage 和最终答复。
- 10 题总耗时 24.105 秒，单题中位数 2.728 秒。模型响应报告输入 6873 tokens、输出 327 tokens。这些是本次运行值，不保证跨版本完全复现。
- 把端点改为 `127.0.0.1:1` 后，图返回 `ConnectionError: Failed to connect to Ollama`；没有模型成功回退。共享 Ollama 服务未停止。

## 实际问题与边界

模板默认导入 Tavily，且原仓库依赖清单包含多个云模型适配包；本项目只保留其 LangGraph ReAct 循环，T1 用三个只读本地工具及独立依赖清单。`uv python install` 曾自动在 `~/.local/bin` 建立一个指向项目解释器的链接；确认该链接为本次新建后已移除，项目内解释器保留。

第一次烟测只保存框架消息。为核对原生协议，随后加入本地 HTTP 请求/响应记录并完整重跑 10 题。实际请求使用 `stream=false` 以完整记录 JSON 响应；这项设置是 T1 审计条件，不能直接推断流式模式的结果相同。

通过的是小工具原生 Function Calling 可行性，尚未检验 9 个实际业务工具 schema、订单政策、长对话、确认或恢复。工程 T1 达到计划门槛；用户是否理解及能否独立修改，仍需分别验收。
