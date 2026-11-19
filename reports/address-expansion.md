# 订单和账户地址扩建

`t5-address-v1` 共 73 道任务（Dev53 / Test20），比上一版多 18 道。已有 55 道沿用原分组。新任务按账户分组，保留来源任务 ID 与来源划分。当前版本距离 80 道还差 7 道，计划中的 Dev20 / Test60 和最终冻结评测仍待完成。

新增两个原生工具，分别修改待处理订单的收货地址和当前认证用户的默认地址。每个提案展示原地址与新地址，独立确认后写入。账户默认地址不会联动修改历史订单。修改地址后再改商品，验收会同时核对这两次提交的状态和回执。已经被更新覆盖的地址回执不能代替重新确认。

18 道新增题均从独立快照执行来源参考动作。测试核对确认、写入结果、其他订单和用户无副作用，以及遗漏确认时的失败状态。参考流程使用本地真实业务方法，沟通字段只做检查器验证，完整沟通仍需真实模型轨迹的语义审查。本版没有执行模型全量评测。

模拟用户通过私有参考参数判断提案，模型只看到用户请求与追问回复。此方式的限制是它可能拒绝其他合理方案，例如语义等价但参数不同的方案。相同目标的换序重复题，以及依赖多阶段改口的题，保留在筛选记录中。

- [新增与暂缓记录](../data/reviews/address-expansion.json)
- [生成脚本](../scripts/build_address_dataset.py)
- [数据版本清单](../data/versions/t5-address-v1/data/manifests/dataset.json)

```sh
.venv/bin/python scripts/build_address_dataset.py
.venv/bin/python -m retailflow.cli validate --manifest data/versions/t5-address-v1/data/manifests/dataset.json
.venv/bin/pytest -q tests/test_address_expansion.py
```
