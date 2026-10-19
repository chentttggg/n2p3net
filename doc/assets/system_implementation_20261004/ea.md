# E1：EA 授权、实际拟合、输出和物理入口的纵向闭合

日期：2026-10-04。冻结基线为 `frozen/system_implementation_20261004_before/source.manifest.json` 与 `resources.manifest.json`；本轮保留原始数据、旧产物和并行工作。使用 codebase-design、diagnosing-bugs、tdd 的深 Module / 实际反例 / 纵向验证方法。实验统计门禁和全库性能属于其他实施工作，本记录只界定 E1。

## 1. 本轮入口与所有权

`data.ea_artifact.EAFitPlan` 集中持有调用者给定的计划身份、物理行授权、参考模式及完整数值配置。`authorization_from_dataset` 供可信 split/plan owner 构造计划，必须面对实际 attested parent，而不再接受调用者手填的 parent SHA。计划摘要表示内容身份，不是数字签名，也不是科学授权的外部证明。

实际使用链条只有一套 Interface：

```python
plan = EAFitPlan(plan_id, caller_row_authorization, reference_mode, numeric_config)
artifact = fit_ea_feature_artifact(attested_parent, expected_plan=plan)
save_ea_feature_artifact(path, artifact, parent=attested_parent, expected_plan=plan)
restored = load_ea_feature_artifact(path, parent=attested_parent, expected_plan=plan)
features = restored.transform(new_attested_input)
```

- fit 内部从实际 parent 解析授权行，验证实际 participant/row/content 后拟合。保存和加载重放拟合数值与输出，不接受各自独立自证的 state/authorization/output。
- 逐状态核对 scope、domain、dimension、样本数、buffer digest、经验二阶矩、参考矩阵及 inverse；输出绑定实际行顺序、标签（若有）、subject、domain 路由及变换值。
- `source_rows`、`target_calibration`、`unlabelled_buffer` 和 `per_domain` 继续可用；EA 不使用标签，`y=None` 的无标签拟合、保存、加载、推理均可用。
- 逐域授权必须覆盖所有输出域；每个域均须满足数值最低样本数。未授权域拒绝发布；冻结推理遇到新域拒绝，不拟合兜底状态。
- `transform` 只读应用已验证状态。它允许新 attested 数据，核对观测通道、通道顺序、预处理及参考；不要求新的数据等于拟合 parent，也不重拟合。
- 构造产物复制并拥有数组和嵌套元数据，避免冻结输出时把 caller 的数组锁只读。后续重验证不会重复复制整个特征矩阵。
- artifact 加载边读边哈希到 8 MiB 内存阈值的临时快照，超出后落盘；解码同一份已验证快照，不在验 SHA 后重开源路径。

格式为 `n2p3_ea_feature_artifact/2`、`n2p3_euclidean_alignment_fit/3`、`n2p3_ea_fit_plan/1`。旧局部 artifact/1 不静默升级；原字节保留供历史审计。

CLI `experiments/prepare_euclidean_alignment_cache.py` 改为必需的 `--fit-plan` 和 `--fit-plan-sha256`，实际计划由上游 owner 提供。删除 `--fit-scope` / `--fit-rows` / 数值覆盖参数，避免执行器选择拟合行并替自己签发授权。CLI 在读取 parent 前核对计划字节；计划内的 config/reference 也绑定加载者的 expected plan，防止合法换 shrinkage 后偷换实验臂。

## 2. 物理入口及历史读取

`data.epochs.assert_physical_epoch_dataset` 在普通 dataset validate、full/attested load 和公共 checkpoint metadata gate 生效。带 `data_alignment` 的旧伪电压 cache 及 dimensionless EA 坐标不能进入物理模型。`PreparedTransferCheckpoint` 和直接 `load_n2p3_trunk_checkpoint` 共用该 gate。

独立 `load_historical_epoch_dataset` 仍允许只读检查旧 cache 的数值与 provenance；返回对象标记为 historical audit，正常 validate、保存和 checkpoint 加载均拒绝。无 EA 的原有正常 cache、source prepare 和 checkpoint 行为回归通过。

## 3. 实际 RED 与反例

第一条 RED 直接使用改前的真实 `save → load → assert_parent_replay` 链：授权 `[0, 1]`，state 却用 `[2, 3] × 7` 拟合，原实现返回成功，pytest 报 `DID NOT RAISE ValueError`。本轮修复后相同科学反例在发布前因实际 fit replay 不符被拒绝。

独立审查又产生两条 RED：

1. 构造 artifact 后，原 caller 的 `values.flags.writeable` 被改成 false。已改为所有权复制，caller 后续改值不再更改 artifact。
2. 无标签 `unlabelled_buffer` 被误要求 labels。已移除该无关前提，完整保存/加载/冻结预测链通过。

保留的公开 Interface 反例还覆盖：重签伪输出、重签假 buffer/样本数/不同拟合行、不同数值算子的相同 row 授权、parent 加载后内存突变、row-domain 路由错位、缺域、每域样本不足、新推理域、缺导 mask、非法行索引、state 修改后的冻结预测、源路径在 decode 前替换，以及普通 checkpoint 对旧物理 cache 的拒绝。测试造数已集中到 `tests/support/ea.py`，不再跨测试文件导入。

## 4. 真实 Huebner 小切片重放

`probe_ea.py` 读取此前审计保留的真实 Huebner subject1/run0 小切片：16 条事件、FZ/PZ 实测、OZ 未测。原始 cache SHA 为 `079e56aaeb346f286600f4653777bd3a412ef7a46922227daa8649e41e8cb5cf`。

原含未测 OZ 的输入被拒绝。通过共享 `select_epoch_channels` 显式选择 FZ/PZ 后另存 attested parent，仅授权前 8 条拟合，最后 8 条只用于软件重放验证。完成 fit → save → load → frozen transform → 固定 LDA score：

- 合法输出形状 `(16, 2, 179)`，无伪造 OZ 实测槽。
- 授权 8 条的二阶矩相对单位阵最大绝对误差 `5.3277893e-9`。
- 保存/重载特征逐位相同，冻结 LDA 分数最大差值 `0.0`。
- 最终产物 SHA `bdb141b99c718fb2e4a2acd37543a7380b6dc2103a2ed199a907a8b2b059279b`。
- [机器记录](ea_real_replay.json)绑定实际执行时的相关源码 SHA、Python/Numpy 版本、输入 SHA、实际授权 plan SHA；产物位于 `experiments/runs/system_implementation_20261004/ea_owned_buffers/`。阶段性重放产物与报告均保留，未改写原始审计 cache。

这是坐标/权限/重载正确性验证；没有用这 16 条事件估计算法性能或宣称 A1/Q1 改善。

## 5. 命令与验证范围

```powershell
.\.venv\Scripts\python.exe -m pytest tests/data/test_ea_artifact.py tests/data/test_ea_physical_ingress.py tests/data/test_euclidean_alignment.py tests/transfer/test_prepare_euclidean_alignment_cache.py tests/data/test_epochs.py tests/data/test_cache_attestation_consumers.py tests/transfer/test_prepared_checkpoint.py tests/train/test_prepare_multidomain_source.py tests/train/test_prepare_source_selection.py tests/train/test_pretrain_supervised.py -ra
```

结果 `162 passed in 23.31s`。随后嵌套 metadata 防 alias 收尾再次运行 EA artifact 全部 30 项，`30 passed in 5.90s`；数值公式/CLI/历史入口专项此前共 46 项通过。owned files Ruff、compileall 和 `git diff --check` 通过。

```powershell
.\.venv\Scripts\python.exe doc/assets/system_implementation_20261004/probe_ea.py --output experiments/runs/system_implementation_20261004/ea_owned_buffers
```

最终真实小切片重放退出码 0。脚本遇已有 parent 产物会拒绝覆盖；重复运行请使用新 output 目录。

## 6. 界限

本轮未跑十二库性能矩阵，未把 EA 加入仍以物理 electrode 输入为合同的统一 N2P3 模型。类型化 EA 特征可以合法拟合、重载和供特征模型使用，但不以伪物理电压方式兼容旧神经模型入口。全源共同实测子空间不足的问题仍须由实验设计处理；不能把缺导补零后声明全导可观测。

计划来自调用者，不可能仅凭 hash 判断科学计划是否合理；本 Module 保证执行遵循给定计划以及实际 parent。历史 checkpoint 如果从未记录其源输入的非物理变换，单凭权重不能推断出被省略的历史；本次修复负责明确声明过的 cache/输入和新产物链。
