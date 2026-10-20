# T2 retail 来源与派生任务协议

日期：2026-10-20。此阶段只构建数据与静态协议，没有运行模型业务评测，也没有得到成功率。任务记录、脚本与 gold 仍是私有评测材料，不应作为生产 prompt 示例或公开原始数据发布。

## 来源与复用边界

固定来源为 [tau2-bench](https://github.com/sierra-research/tau2-bench/tree/2174a603f6d014ef94473ffa95957f6ce27100db)，commit `2174a603f6d014ef94473ffa95957f6ce27100db`，上游包版本 1.0.1。经用户确认后取得的 12 个文件共 3,241,745 字节（约 3.09 MiB），本地源目录占用约 3.1 MiB；逐文件字节数、SHA-256、许可证说明在 `data/manifests/sources.json` 和 `UPSTREAM.md`。源文件未改动，未安装上游包或额外依赖，未下载模型。当前 T2 衍生数据约 32 KiB，远低于批准的 100 MiB 新增预算。原始数据库、任务和私有派生记录均已加入 `.gitignore`，避免将它们默认提交到未来的仓库。

政策要求先通过邮箱或姓名加邮编定位用户，只处理同一用户的订单；取消必须是 pending 且原因限于 `no longer needed` / `ordered by mistake`；退货必须是 delivered、选定原支付方式或用户现有礼品卡，并在修改前取得明确确认。原始取消工具写入 `cancelled`、取消原因及退款记录；退货工具写入 `return requested`、退货 item ID 与退款方式。T2 对这些初始条件进行静态核验；T3 才实际调用业务工具。

下载的 `tools.py` 依赖未列入本轮批准清单的 `tau2.environment.db`、`tau2.environment.tool` 和 `tau2.utils`。因此目前保留固定版源码供 T3 适配，不能称已运行上游业务工具，也不擅自拉取上游全部包。

## 筛选与划分

原始 `tasks.json` 有 114 道，`train` 74、`test` 40，`base` 是两者并集，不能再加一次。按首版八个业务工具检查参考动作，75 道含不支持的动作（换货、订单修改、计算等），整题排除；其余 39 道逐题检查完整请求、条件分支与脚本可表达性，保守保留 13 道，26 道暂未纳入，不能说这 26 道全部不可做。尤其 5、19 虽有退货参考动作，原目标还包括换货，不能剪掉困难部分充数；67 与 test 68 同一用户和近似目标，不能跨组当作独立 Dev 题。

固定版 `t2-v1` 为 Dev 9 / Test 4，总 13，道数距目标 Dev 20 / Test 60 分别少 11 / 56，合计少 67。按主类计：合法状态修改 9、多轮与需求变化 2、拒绝转人工 1、查询 1。分布不均衡，不能据此推出宽泛能力或正式三组对照效果。选中来源 ID 为 Dev `2,24,48,50,69,73,81,92,113`，Test `51,53,68,108`。Dev/Test 沿用上游 train/test 方向，并按同一用户 ID 归组；51 与 53 同组，跨组交集为零。后续若要扩大任务集，必须新建数据版本并重新报告全部分母，不把旧 Test 结果继续当未见数据。

所有 13 条标为 `origin=adapted`：来源业务目标与参考最终状态保留，但首句改成可直接输入 CLI 的用户话语；后续身份、订单、原因、退货商品及付款方式只通过 `needs_input` 字段回答。`needs_confirmation` 使用提案规则；24 在取消提案时反悔并提出 T-shirt 查询，92 要求把智能手表加入提案并追加另一订单的 e-reader。未定义的字段/提案分支标记 `script_unhandled`，最多 8 个用户回合，不按轮数提示正确工具或读取 gold。改写和脚本均在 `configs/evaluation.json` 可逐题复查；原始 `reason_for_call` 留在私有事实中，不直接放入首句，因为其中包含“如果被确认时怎样改意”等隐藏行为指令。

每条任务的 `expected_checks.orders` 描述目标最终状态：取消后的状态/原因，或退货后的状态/item 列表/退款方式；`communicate_info` 要求必要信息确实告知用户，转人工另有标记。`forbidden_effects` 要求未确认不写入、不额外退商品、其他订单不变。T2 的 `validate` 验证来源及衍生文件哈希、原始 split 方向、任务/组 ID、对象存在、脚本分支和写动作的初始政策条件。`check_final_state` 另按目标订单最终字段、取消退款记录和礼品卡余额、其他订单/用户/商品不变、确认写入记录、必要沟通与转人工记录评分。9 道 Dev 已用构造的参考末态逐题通过，并故意修改无关订单验证会失败；这验证的是检查规则的基本可执行性，不是上游工具或 Agent 的真实运行。T3/T6 接入真实轨迹后仍需再次核对字段与评分器。

T2 任务与静态检查规则现在按 `t2-v1` 固定：`configs/evaluation.json` SHA-256 `80b099f2f66a308a4b7fd5ff511e7f5b6334e6c4856b85f9166bebddc58decc4`，`src/retailflow/dataset.py` SHA-256 `5a0cc0d6d60e1a8d104ff3ed3854d678d5db52d6293da9edf593b070088f5315`，`data/manifests/dataset.json` SHA-256 `da234764ff25f869f3cb60120b0e4d64c474608d23f4a4d2ab655cf422dea87c`；派生文件哈希在清单中。若真实轨迹暴露 T2 标注或检查器缺陷，先保存旧版报告，再创建新版本并说明改动；不能在看过 Test 结果后悄悄改旧版评分。

## 验证证据与限制

`python -m retailflow.cli prepare --config configs/evaluation.json` 生成 `data/manifests/{sources,dataset}.json` 与 `data/derived/{dev,test}.jsonl`；相同内容可重复运行，已有冻结文件若不同则拒绝覆盖。`python -m retailflow.cli validate --manifest data/manifests/dataset.json` 输出 `valid: true, errors: []`，计数 9/4/13。`python -m pytest tests/test_dataset.py -q` 为 8/8，包含跨上游划分的近似题阻断、非法来源动作排除、私有字段投影、数据篡改、对象异常和最终状态检查。全项目测试 18/18。本机当前 shell 找不到 `uv` 命令，因此使用项目独立 `.venv` 中的 Python 与 pytest；没有安装或联网。上述审查由助手完成，不是人工独立审核；Test 只审来源和预期，尚未查看 Agent 输出。

后续运行时必须为每题从固定 `db.json` 创建独立状态副本，用户脚本仅读取私有意图和当前请求，评分器独立读取预期与完整轨迹。生产模型入口只使用 `agent_initial_message(task)` 输出的首句，绝不传入 `user_private_facts`、`script`、`expected_checks` 或来源参考动作。
