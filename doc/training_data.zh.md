# 训练数据与结果

本页说明数据、模型和结果的存放与恢复方法。目录规模取自2026-11-10清理前盘点，见[紧凑清单](assets/build_cleanup/repository_20261110/training_data_catalog.json)；本次没有训练、重新处理EEG或重算历史预测。压缩包已逐成员验证，散件退休的实际结果由本轮清理回执维护；研究续接仍读[当前checkpoint](assets/research_resume_20261012/pause_checkpoint_current.json)。

## 数据与模型

| 位置 | 内容与清理前规模 | 保留方式 |
|---|---|---|
| `mne_data/` | 公开原始数据、BrainSync录制及具名派生数据，共59,232,954,909逻辑字节 | 原位保留；数据集级文件数与大小见紧凑清单 |
| `cloud_inputs/` | native、EA及source-selection输入；21文件，2,129,932,259字节 | NPZ与record、selection、manifest配套原位保留 |
| `experiments/cache/four_dataset_20260907/` | Huebner合并输入、13人分片、smoke及sidecar；30文件，2,480,523,547字节 | 整目录无损冷存；再次使用前恢复原路径 |
| `experiments/cache/history/` | 历史派生数组与sidecar；12文件，1,378,316,938字节 | 整目录无损冷存；历史缺失项保持原记录 |
| `experiments/cache/legacy/` | 旧输入合同缓存与审计；13文件，211,525,936字节 | 整目录无损冷存；启用前重新核合同 |
| `models/source/` | 当前旧basic入口的源权重；1文件，36,347,053字节 | 保持可直接读取 |
| `experiments/results_history/` | 早期结果、预测、日志和实际producer；369文件，204,607,617字节 | 无损封存原相对路径，按需解压 |
| `frozen/` | 原源码快照、模型/结果恢复清单及历史证据 | 保留原字节与成员身份，按[归档规则](../frozen/README.md)恢复 |

原始采集与派生输入分别保留。`mne_data/BrainSync_startup_cut_4s/`记录启动裁剪后的具名派生，不能替换原EDF；其处理关系见[输入合同](input_contract_math.zh.md)。源库、参与者、通道、标签和实际支持由各次原manifest决定，目录名本身不证明其可进入当前训练。

NPZ是数组容器，扩展名不能说明压缩方式。本次只读ZIP目录得到：`cloud_inputs`的5份NPZ、`history`的4份及`legacy`的2份均已使用DEFLATE；`four_dataset_20260907`的15份NPZ使用STORED，数组尚未压缩。冷存采用外层ZIP，保留每份NPZ与sidecar的原字节和SHA，不重新编码数组。Huebner合并record明确引用13个原分片，恢复时须连同这些分片一起解压。

当前[旧basic配置](../configs/brainsync_old_basic_nn_source.json)的`source.path`指向`models/source/basic12_r1s0m0_u27950.pt`。本次重算SHA-256为`935ee55de598256ca9299595ec899359751b1ebbf5fa0bb15af0f0fbb8bda3bd`，与配置`source.sha256`一致。校准配方仍由该文件的`recipe`持有；这份源模型与三个旧cache目录没有直接路径依赖。

## 保留哪些结果

结果解释沿用四个主题：[校准与选择](evidence/calibration_and_selection.zh.md)、[响应与噪声](evidence/response_and_noise.zh.md)、[公共评估](evidence/public_evaluation.zh.md)、[执行与集成](evidence/execution_and_integration.zh.md)。本页只列取数入口和恢复限制。

9月16日四库公共开发矩阵的本地结构化入口是[results_summary.json](assets/unified_cloud_20260916/results_summary.json)和[mean_auc.csv](assets/unified_cloud_20260916/mean_auc.csv)。JSON的`means.<arm>.<K>.mean_auc`保存AUC，`n_requested_participants`保存参与者分母；`integrity.logical_cells=12150`、`integrity.fits`和`replay`分别保存评分单元、拟合/复用与原保存重放记录。精确K200覆盖123人，A0Q0为0.6947、完整A1Q1为0.6877、scratch为0.6534。完整包对scratch有平均收益，对普通微调没有全面优势；原文沿[公共评估历史索引](evidence/history.zh.md#公共评估)恢复。这份四库矩阵与9月18日十二库大版有各自的计划、producer和结果，恢复时分别定位。

10月31日本人旧basic结果由[矩阵接受绑定](assets/brainsync_matrix/acceptance.json)及[单轮接受绑定](assets/brainsync_matrix/single_round_acceptance.json)定位。前者记录`requested_cells=completed_cells=150`；后者记录`requested_rounds_per_seed=100`和`new_fits=0`。原主题记载：C1到C10，NN source的R1为29.00%到48.67%，完整R5为60.00%到98.33%。R1每轮只用九个数字各一次闪烁，R5累计五轮，完整分母分别为100轮和20个block；神经读数取三个目标seed平均。它们来自同一人、两份目标7/9的旧开发录制，不能作为独立成人确认；三个seed也不增加人数。完整剂量、覆盖和方法配对继续由校准主题维护。

截至本次盘点，`experiments/runs/`在当前仓库原位置不存在。接受绑定仍保有原路径、大小及SHA，其中旧矩阵`complete_execution.tar.gz`为305,844,200字节，SHA为`5d5236ba0a88186093fcb39cc6ff0e0818f61fdf6279f37c5b5cd433c57227d9`；R5的`analysis/summary.json`、`analysis/decision_records.json`以及R1的`single_round/results.json`、`round_predictions.json`也有原SHA。清理没有找到这些载荷，不能凭现有摘要重放完整矩阵。恢复须取得对应原包并核SHA，连同`execution_source`、`recovery_source`及分析producer恢复；当前源码快照不能代替当时执行字节。

[9月20日大版清单](../frozen/major_domain_transfer_20260920/manifest.json)的`executed_sources`分别保存主训练、复用首折、输入准备和工作站默认源的真实producer；`external_assets`指向完整模型/结果包。该外部包记录为11,990,531,333字节，SHA为`db3456fada03f6cc7177c225b822c4884aef4e2204c646218c029ef0bd99c790`。本次没有连接云端，远端可达性仍需原owner核验。[云端原载荷恢复绑定](assets/research_resume_20261011/cloud_payload_recovery_binding.json)另持十月公共程序的原结果与本地恢复位置，不能与十二库大版或本人矩阵混为一份结果。

启动裁剪后的basic/QC输入接受、源准备量及EOG幅值诊断见校准与响应主题。旧模型和旧成绩保持各自输入身份；新链的完整NN比较须有自己的训练与结果凭证。

## 冷存与恢复

本轮[训练输入清单](assets/build_cleanup/repository_20261110/training_inputs.manifest.json)对应`frozen/repository_consolidation_20261110/training_inputs.zip`：上述三个旧cache目录共55成员，从4,070,366,421字节压至2,583,081,719字节，逐成员SHA验证55/55通过。外层包SHA为`2612d50ad9fcdcb9d7402a198d7289d89019d817c15ad868c0d1048d84a488dc`。

[历史记录清单](assets/build_cleanup/repository_20261110/research_records.manifest.json)对应同目录的`research_records.zip`。它保存`experiments/results_history`、旧`tmp/audit`及工作树退休记录的原相对路径，共2,753成员，从245,211,088字节压至35,672,738字节，逐成员SHA验证2,753/2,753通过。包SHA为`d62a1b9f2a1eb09f2b17add9ac2aabf185cd56d588f88d63c717e7b2ea1b3aac`。历史JSON和producer保持原字节。

`experiments/inspect_run_lineage.py index`仍读取`tmp/audit/scratch/g8_inventory.json`和`g8_decl.json`，这两份小输入继续留在原位置；`query`读取已有索引。其他历史审计需要原日志或脚本时，从归档显式解压相应成员，再按成员清单核SHA。

三个旧cache目录未出现在当前源码/配置的具体输入路径或两个当前暂停检查点中，整包验证后已退休散件，实际凭证与逻辑缩减量见[清理回执](assets/build_cleanup/repository_20261110/receipt.json)。恢复训练输入时，将完整目录解压回原相对路径，核NPZ和sidecar原SHA，再由实际消费者检查输入合同及父数据；不能把压缩包的SHA写成原NPZ的SHA。

`artifacts/personal_erp_identity_06/report.json`及空`lean.log`也要保留：[原审阅绑定](assets/lean_personal_erp_20261018/identity_current_source_review_01.json)的`file_sha256`明确保护它们。暂停检查点、在途输入和外部科研交付继续保持原位置，恢复依赖沿[续接入口](assets/research_resume_20261012/HANDOFF.zh.md)核对。
