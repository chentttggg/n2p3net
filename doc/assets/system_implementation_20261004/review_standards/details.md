# Standards 审查证据与建议

标准来源：AGENTS 的 shared identity/channel/fit/time contracts、preserve data/source execution binding；`doc/engineering.zh.md` 的保存重载与只读推理、失败状态纪律；`doc/technical_structure.zh.md` 的 canonical ownership；codebase-design 的小 Interface / 深 Implementation / 接口反例。基线 smells 全部作为判断启发，不作为独立硬缺陷。未重复 Ruff。

运行：

```powershell
.\.venv\Scripts\python.exe doc/assets/system_implementation_20261004/review_standards/probe_standards.py
```

真实模型和 CLI 产物保留 `experiments/runs/system_implementation_20261004/standards_review/`，不可覆盖。六个 probe 的机器结果为 [results.json](results.json)。其中前五种对应摘要四项（noise 与 continuous 合并为冻结元数据一项）：

| 反例 | 实测结果 | 修复方向 |
|---|---|---|
| target-calibration noise scope 与 LDA 不同 | 初次 fit 拒绝；修改公开 `provenance['scope_id']` 后 fit/save/load 成功 | 嵌套 provenance 深冻结，并在公共消费/保存准入核验完整认证状态；新构造与load要验证必需元数据 |
| continuous 原 calibration 记录回放 | 原 gate 拒绝；`provenance['records'].clear()` 后保存/加载再评分成功 | 训练信号/record账本不可变且必要；不得把缺失字段默认为空历史 |
| noise CLI 输入路径替换 | 原文件 SHA `8e0f4d98a9a4da0b211ef13636d541f7191e25edaefe3bee9e6a300afc930786`，报告记录替换SHA `302e1b094a8c100011933ca90c330b96fcde422f6314acbb78a3559278670712`，模型实际input_sha仍是原数组 | 统一已验字节消费对象；输入、计划、模型、metadata 的摘要均取同次读取快照，运行中路径变化不改变报告归属 |
| ERP 全 NaN mask | 仍返回全部finite scores | 按共享 mask 合同在 Interface 拒绝，而非让 NaN truthiness 表示 observed |
| fixed_analytic 任意正交 basis | `from_record`通过；额外实际 `StructuredERP.from_state` probe也通过，score最大变化4.0873 | 校验公式和计划角色；不可只以重新摘要和正交性代替 fixed_cosine |
| xDAWN refit单类标签失败 | 仍 predict 原分数 | 清除fitted资格或实现事务式失败语义；该问题冻结基线已存在，建议修复但不归因本轮新增 |

固定解析基实际保存结构反例见 [analytic_role_replay.json](analytic_role_replay.json)。检查过相对正则、covariance 对称/PSD校验、无需oracle的连续预测接口、原始候选词表覆盖、数值state保存集合与ERP family检查。未见它们因基础缺失而必须推倒；修复成熟工程问题后继续研究即可。

Toeplitz 许可证存在于 `src/baselines/structured_erp.LICENSE`，代码明确归属与来源；本轮freeze helper新增处理相邻LICENSE，未报告无依据的许可证缺失。数值研究 API 的 raw/state坐标需要调用者显式输入，文档也声明不解决物理状态可辨识性；未把这一已声明研究边界误当成既成产品算法收益。

**证据边界**：独立审查不是全套再测试，也不证明算子性能。构造反例演示公开元数据可意外变化与实际文件替换时科学身份失真，不主张抵御有意修改源代码的攻击者。摘要列出的P1应在验收前闭合；此处保留其修前证据。
