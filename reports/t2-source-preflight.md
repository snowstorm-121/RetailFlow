# T2 固定来源下载前核查

日期：2026-10-20。状态：下载前核查已完成；用户确认后，以下 12 个文件已下载到 `vendor/tau_retail/source/`，实际大小与 HEAD 一致，SHA-256 见 `data/manifests/sources.json`。

唯一上游仓库：`https://github.com/sierra-research/tau2-bench`，固定 commit `2174a603f6d014ef94473ffa95957f6ce27100db`。下载地址统一为 `https://raw.githubusercontent.com/sierra-research/tau2-bench/2174a603f6d014ef94473ffa95957f6ce27100db/` 加下列相对路径。大小来自固定 URL 的 HTTP HEAD，不是本地文件实测；下载后要登记 SHA-256 与实际字节数。

| 相对路径 | HEAD 字节数 | 用途 |
|---|---:|---|
| `LICENSE` | 1,072 | 仓库代码许可证 |
| `pyproject.toml` | 2,502 | 固定版 Python/依赖说明，不直接安装整个 tau2 包 |
| `CHANGELOG.md` | 19,434 | 核对任务修订记录 |
| `data/tau2/domains/retail/db.json` | 2,811,616 | 只读原始模拟用户、商品、订单状态 |
| `data/tau2/domains/retail/policy.md` | 6,699 | retail 业务规则 |
| `data/tau2/domains/retail/tasks.json` | 345,982 | 来源任务与目标动作 |
| `data/tau2/domains/retail/split_tasks.json` | 3,263 | 上游 train/test/base 边界 |
| `src/tau2/domains/retail/tools.py` | 29,202 | 查询、取消、退货等原始工具语义 |
| `src/tau2/domains/retail/data_model.py` | 8,799 | 业务对象类型 |
| `src/tau2/domains/retail/utils.py` | 248 | retail 数据路径说明 |
| `src/tau2/domains/retail/environment.py` | 1,615 | retail 环境加载方式 |
| `src/tau2/environment/toolkit.py` | 11,313 | 原始工具注册与调用所需公共代码 |
| **合计** | **3,241,745** | **约 3.09 MiB** |

初步磁盘预算：上述原始文件约 3.1 MiB；必要源码快照、筛选后的任务、清单与报告预计另增 5–30 MiB。将 T2 工作包的新增磁盘上限设为 100 MiB；若实际依赖需要额外下载，先单独说明具体文件与原因。现有 `.venv` 已含 Pydantic，本清单不包含安装任何新包、下载模型或完整 tau2 包。

已只读检查 `split_tasks.json`：`base` 收录其 `train` 和 `test` ID，不能相加计数。`tasks.json` 中存在仅凭八个限定业务工具无法完成的目标，例如换货；筛选必须依据完整用户目标和动作，不以参考动作链凑题。仓库声明 MIT，但 retail 数据的文件级使用说明尚待取得原件后核对；不能仅凭仓库许可证推定所有数据可再分发。
