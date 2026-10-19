# Operator 审查反例修复与校准右删失协议

日期：2026-10-04。独立 Standards 审查原证据保留于 `review_standards.md` 和 `review_standards/`，修复不覆盖原 probe 结果。此记录对应随后 root 明确授权接管的 `noise_model.py`、`continuous_decision.py`、`operator_lda.py`、`run_operator_decision_study.py`。

## 冻结状态修复

三条公开链反例先写入测试并实跑 RED：noise scope 可改、continuous records 可清空、CLI 报告改后路径 SHA。随后完成：

- `freeze_metadata` 对所有嵌套 dict/list 转为不可修改的 JSON-compatible dict/tuple；不能只冻结数值数组。
- JointNoiseModel、ContinuousDecisionModel、OperatorAwareLDA 均保存构造时完整状态 certificate；project、fit consumer、predict/score、save 重算核验。即使用 `object.__setattr__` 绕过 Python frozen dataclass，更改scope、ledger、权重也会被 public consumer 拒绝。
- 噪声 scope/count/draw IDs/input SHA、连续 training records/signal SHA、LDA class/count/噪声父scope 全部在构造及 load 验证。保存不会为遭改动对象重新签发合法身份。
- continuous learned model 缺失或空 training records 会拒绝，包括删除记录后重新计算 artifact digest 的文件。
- 独立数学 oracle 保留显式 `analytic_reference` 角色，需 `reference_id` 和空 records；不把数值参考冒充拟合模型。该角色用于独立 dense-Gaussian 数学反例。

## 消费字节与报告身份

CLI 的七个入口共用一次性输入快照：边读边 SHA-256、8 MiB 阈值的 spool、同一快照解码。JSON plan/metadata、NPZ数据、NPY basis、已保存model均由同次读取返回摘要；不再拟合/预测后重开原路径补hash。额外记录 basis 原始文件摘要。

新增真实CLI反例覆盖两种时机：

1. noise fit 内替换输入 NPZ 和 plan；报告仍绑定原始执行字节，模型数值摘要与原输入一致。
2. continuous score 内替换 model/input/metadata 三条路径；prediction.json 仍保留实际消费的三个原SHA。

每次 CLI 保存 `execution.json`，标记 running/completed/failed；失败不会被留下的局部产物误作 completed。source 在模块导入、执行前、执行后核对，旧导入状态与新磁盘源码不匹配直接拒绝。原始产物、历史报告和数据未删除。

## 新增科学协议：明确授权的校准右删失

raw Adapter 独立检查发现 BNCI K200/K800 常在 decision 中间，若沿最后授权 flash 后读取完整 1.2s，可能包含后续未标记 flash。root 明确决定：保留全部K已授权labels，但将raw截到下一未授权flash onset；最后若干响应视为右删失。Adapter owner 提供完整已授权event history，不读取后续标签。

`CalibrationRecord.right_censored: bool=False` 为显式权限。缺省仍要求完整响应尾；True 时所有越过当前 record 末端的设计行按实际可观测长度裁basis，不虚构尾部零观测，也不删去对应已授权label。每条fit receipt记录 `right_censored`、`right_censored_events`、`n_samples`。

连续模型保存格式升为 `n2p3_continuous_decision_model/2`，原 `/1` 研究prototype字节留存但不静默补新协议。`score`始终要求完整尾部，校准删失权限不会扩大到测试。

独立公开测试先因缺新字段RED，之后证明：缺省拒绝末尾截断；显式协议成功拟合和save/load；ledger记录56个实测样本、6条授权event、1条右删失；同样截断测试数据仍被score拒绝。真实BNCI Adapter的physical读取与队列验证由其owner单独记录，本文不把数值core测试冒充真实读取验证。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/transfer/test_noise_model.py tests/transfer/test_continuous_decision.py tests/transfer/test_operator_lda.py tests/research/test_operator_decision_cli.py -ra
```

最终 `18 passed in 3.32s`，包含独立 dense-Gaussian covariance 数学对照、联合cross terms、冻结保存重载、删失边界、权威元数据/derived state异常、CLI实际消费字节和失败状态。ownedfiles Ruff、compileall、git diff --check通过。执行源码SHA见 `operator_integrity_runtime.json`。

这解决已验证工程关系与明示校准协议；不证明真实工频/边界state可辨识，不替代完整raw Adapter授权，也不构成十二库或成人设备性能收益。
