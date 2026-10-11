# 历史成绩可比性与执行身份审计

日期：2026-09-29。只读调查；未训练、未修改生产源码。范围覆盖历史 LOSO、09-07 四库与 training-chain、09-16 P1/P2、09-16 unified、09-18 unified12。主要证据来自原始 tar 内 record、NPZ、源 checkpoint 与执行源码；报告文字仅用于定位历史文件。

## 1. 已确证的结论

1. **“旧无迁移更高、现在迁移更低”目前不是同一实验的前后下降。** 旧 `0.74511` 是 BI2014a 的 64 折、16 导、2–30 Hz、800 ms 后窗、同库其他人训练后直接留人测试；现在的跨库 few-shot 是六差分输入、0.1–30 Hz、1200 ms 后窗、整库留出、目标前缀校准后测后缀。还更换了 cohort、测试行、训练剂量、BN/dropout、部分网络结构。这些因素足以使绝对值不可直接比较，但不能据此量化哪一个因素导致了多少损失。
2. **09-16 P1/P2 的 `epochs=60` 不是实际 60 轮。** 原 cell 同时设置 `max_updates=200`，实际 K200 为 200 updates/200 遍/40,000 draws，K800 为 200 updates/100 遍/80,000 draws。09-07 chain 同档为 60/120 updates；09-16 unified 为 90/180 updates。只读计划顶层 `target_epochs=60` 会误判“同预算”。
3. **P1/P2 到 unified 同时改变输入和配方。** 原 source checkpoint 直接确认：前者是 causal-v3/forward IIR、无 50 Hz notch、普通 BN、无 residual；后者是双向 IIR+notch、observed-channel BN、额外 feature residual。目标从 online BN + checkpoint dropout 变为校准开始时估计后冻结的 BN + dropout off。不能把其差额归为 A1/Q1 或“十二库扩源”单因素。
4. **重复 scratch/LDA 未见异常漂移。** 同 run、同 target cache、participant、K、target seed、layout、train/test/label 数组摘要的重复基线，09-07 chain 1,220 组、P1/P2 1,830 组、09-16 unified 1,476 组，全部 score 数组哈希一致。它排除了这些已有结果里“同基线被不同 source 方法重跑成不同预测”的具体怀疑，不证明所有可复用键完备。
5. **存在稳定的跨运行锚点。** 09-07 chain 与 09-16 P1 的六导 waveform LDA 在 305 个同 cache、同 participant、同实际 K、同 train/test/y 数组的配对单元中，AUC 逐个完全相同。代表单元 Huebner2017/1/K200 的 score 数值也完全相同；NPZ 仅从 float32 存储变为 float64，不能因字节哈希不同判性能变化。
6. **minimum_test_trials 与实际测试量不一致是独立合同问题。** 09-07 chain、P1/P2、09-16 unified 都存在 BI2014a participant 6/K200、已完成但实际只测 85 trial 的记录，而 plan 声明 `minimum_test_trials=200`。本子任务已核实 NPZ 及记录；根因与当前修复由输入合同审计进一步核查。它不是新统一模型才出现的问题。

## 2. 原始产物复算覆盖

| 实验 | 本轮直接读取 | 成功与原 record 配对的 NPZ | 最大 AUC 复算误差 | 带 prediction integrity 的哈希不一致 |
|---|---|---:|---:|---:|
| 09-07 chain | `completed_prediction_evidence.tar.gz` 中 outer-results 子集 | 37,568 | 0 | 0；早期无该字段的记录不能称已通过此项 |
| 09-16 P1/P2 | `export/results.tar.gz` 全部 target 记录 | 32,238 | 0 | 0 |
| 09-16 unified | `export/verified_results.tar.gz` 全部 target 记录 | 12,150 | 0 | 0 |

这里共 81,956 份原始预测，AUC 由 NPZ 的 `y` 与 `scores` 用 `roc_auc_score` 重算；同时记录实际 train/test 数组摘要、标签摘要、正例数和分母。09-07 chain 不包含本归档全部 stage，不能把 37,568 宣称为历史 45,134 外层评分的全量复算。09-07 四库 b/c 的 3,862 NPZ 由并行评分审计负责，避免重复读取。

复算脚本：[inspect_archives.py](inspect_archives.py)。大型逐单元表、原 cell 示例、inventory 与 source-checkpoint 摘要位于本地忽略目录 [experiments/runs/deep_chain_audit_20260929/comparability](../../../../experiments/runs/deep_chain_audit_20260929/comparability/)。按原路径及 cell index 定位的表为 `chain_0907_cells.csv`、`p1p2_0916_cells.csv`、`unified_0916_cells.csv`；每张对应 `*_inventory.json` 绑定 record SHA、source/evaluation snapshot、source checkpoint SHA 与 target cache SHA。内容摘要与迁移校验保存在 [local_artifacts.manifest.json](local_artifacts.manifest.json)，移动前后 SHA-256 一致，没有删除旧证据或重新计算预测。

## 3. 实验口径对照

| 轴 | 08-28 BI 高分 LOSO | 09-07 training-chain | 09-16 P1/P2 | 09-16 unified | 09-18 unified12 |
|---|---|---|---|---|---|
| 当前核实层级 | 冻结原报告；该轮原预测未在本地找到 | 原 NPZ + record + 冻结源码 | 原 NPZ + record + 源 checkpoint | 原 NPZ + record + 源 checkpoint | 本地分析/收据/执行脚本；逐 cell 包本地缺失 |
| 目标问题 | 同库 63 人学习，1 人不校准直接测试 | 四库整库留出 + 目标本人校准 | 相同四库 LODO + 校准 | 四库 LODO + 校准 | 十二库源、四目标 LODO + 校准 |
| BI cohort | 64，含后来排除 62–64 | 61 | 61 | 61 | 61 |
| 其他普通目标 | 无 | BI2015a 41、BNCI 8、Huebner 13 | 同左 | 同左 | 同左 |
| 通道/参考 | 16 导；不冒认当前 Cz 六差分 | common-three 与六差分分开 | 六差分 | 六差分 | 六差分 |
| 输入 | 128 Hz；2–30 Hz；[-200,800) ms；128 点；mean baseline；零相位 IIR | 128 Hz；0.1–30 Hz；[-200,1200) ms；179 点；mean baseline；forward IIR | 与 chain 对应六导目标 cache SHA 完全相同 | 128 Hz；0.1–30 Hz；[-200,1200) ms；179 点；双向 IIR+50 Hz notch；各 C_K/raw test 独立准备 | 与统一双向合同一致；具体实际执行须以缺失的原包补核 |
| target 监督 | 不使用留出人标签拟合 | K0/200/800/2000，过短记录按原规则回退 | 同左 | K0/200/800 | K0/200/800 |
| split | 同库 LOSO，source 内 group-disjoint validation | BI2014a/BNCI 前缀；BI2015a/Huebner 末 session 留出 | 与 chain 相同原 cache/split | 新 raw 前缀 manifest 与固定最大 K test；索引在 C/test 视图中重建 | 同类新 raw 前缀协议 |
| source 训练 | 最多 30 epochs，patience 6，pos_weight 8 | 多种采样/统计与 source dose；须按 source_arm 单列 | 全部 source 固定 4,000 updates | 每折参考源选 exposure 后共享主源预算 | 十二库参考源选择；不能当固定 4,000 updates 扩源 |
| target full/scratch | 不适用 | 60 epochs，batch cap512 | **200 updates**，batch cap512 | 90 epochs，batch cap512 | 90 epochs，batch cap512 |
| target BN/dropout | 不适用 | full/scratch online BN、legacy train-mode dropout | online BN、checkpoint dropout(.25) | calibration BN、dropout off | calibration BN、dropout off（执行脚本/当前冻结配方依据） |
| target optimizer | 不适用 | AdamW lr.001，wd.0001；实际校准类比加权 CE | AdamW lr.001，wd.0001；实际校准类比加权 CE | 相同 optimizer，Q1 增加先验；BNCI 再加 decision loss | 同左，且有 CE-only scratch 单独对照 |
| 架构 | full_unfold，1,506 参数原报告 | 六导 full_unfold K35，无统一 residual | 六导 full_unfold K35，无统一 residual | observed-channel BN + feature residual；1,279 参数当前已核 | 同 unified 结构 |
| seed | 20260828 | 20260906 | 配对 20260916/17/18 | 配对 20260916/17/18 | 只有 20260918 |
| 主归约 | BI participant-macro | participant within study；固定四库等权 | 同左，source-arm/seed 明确 | 同左，但三库 prior-only 与 BNCI prior+decision 应拆 | 同左；单 seed、固定模型条件推断 |

第一列准确数字不能由本轮直接预测重放保证。旧 `audit_baseline_8sub/n2p3net/record.json` 的 AUC 是 0.49828，且为 250 Hz/350 点/不同输入和模型；它不是 `0.74511` 的原 artifact，不能拿路径里“baseline”替代定位。

## 4. 同一个 epochs 字段为何不能定义相同训练

`target_recipe_examples.json` 保存 Huebner2017/1 的原 cell 完整 recipe、initialization 和裁短的实际 fit ledger。预算如下：

| 目标预算 | 09-07 chain | 09-16 P1/P2 | 09-16 unified |
|---|---:|---:|---:|
| K200 | 60 updates；12,000 draws | 200 updates；40,000 draws | 90 updates；18,000 draws |
| K800 | 120 updates；48,000 draws | 200 updates；80,000 draws | 180 updates；72,000 draws |

P1/P2 在 K200 比 unified 更新多 2.22 倍；在 K800 只多 1.11 倍。新方案即使都写“全网微调”，也改变了学习强度对 K 的函数。history 的 epoch-permutation SHA 绑定实际顺序，不能由相同 seed 推断不同 dataset/recipe 下相同梯度序列。

冻结 09-07 源码还直接确认：`scratch_like_trunk` 在局部 RNG 环境设置 target seed，重新构造整个 N2P3Net，并检查 architecture record 相同；`classifier_fine` 使用 source classifier，普通 `full_fine` 保留整个 source trunk。未发现把随机 linear head 误当此轮 full-fine 的实现证据。不同 source arm 下重用 scratch 的数值一致性见上面的全量检查。

相关原始源码是 `experiments/runs/training_chain_auto_20260907/source/source.source-only.tar.gz` 内：

- `src/transfer/binary.py:307`：scratch 独立构造与 architecture gate。
- `experiments/run_cross_dataset_transfer.py:371`：固定目标配方、`bn_update` 和输入统计；`:471` 为 scratch/source 分支；`:479` 固定目标配置；`:490` classifier-only 冻结 BN。
- `src/transfer/subject_adapter.py:194`：head 初始化分支；`:204` 源 classifier 重用；`:991` 目标训练 RNG；`:1006` AdamW；`:1014` 实际目标类计数和 CE 权重；`:636` 与 `:697` 显式/legacy module mode。

## 5. 强基线与可比性的排除证据

`baseline_reuse_checks.json` 记录重复结果分组键与通过数量。跨 run LDA 对照详见本地结果目录的 `chain_to_p1_lda_exact_inputs.csv`：BI2014a 142 个 K 单元、BI2015a 100、BNCI 24、Huebner 39，总计 305。每组共享精确 target cache SHA 和原始 train/test/y 数组，不依赖猜测被试别名。

上述 LDA 证据表明：旧 causal 输入链中，source 扩展/损失变化不会自动改变相同目标线性基线，现有预测也没有显示 seed/reuse 混乱。它不能证明双向重建后的 LDA应该保持相同，因为其 X、边界支持与 source/target normalization 已改变。

BI2014a participant identity 从 `BI2014a-candidate%3A01/participant/01` 改成 `BI2014a/participant/1`，且统一前缀视图把行号重新编号。因此本子任务**没有**只靠字符串删零或 row index 相等，把 P1 与 unified 自动并成物理相同样本；跨该边界需要 raw recording SHA、事件时刻和显式身份映射。相反，chain→P1 用完全相同 participant key/cache SHA，可合法直接配对。

## 6. minimum_test_trials 的真实反例

09-16 unified 原包成员：`matrix/targets/unified/85daaab93c0bbff693a9094e/shard7/record.json`，`cells[58]`，participant `n2p3-origin://BI2014a/participant/6`、K200、`unified_waveform_lda`，实际 NPZ `n_train=200, n_test=85`。其 plan 声明 `single_recording_prefix` 与 `minimum_test_trials=200`。

同一历史 participant 在以下两代也只有 85 个 test trial：

- chain：`current/source_chain/outer_results/394684542f1c76c99d919d12/shard5/record.json`，`cells[5]`。
- P1/P2：`matrix/targets/P2/6047d5ee7762453e1776c904/shard1/record.json`，`cells[8]`。

它提示实际最小测试量要从数组复核，不能相信 plan 的数字自动被强制执行。短 suffix 下 AUC 的正负配对数量少，会增大参与者间分数离散；没有新增实验时不能把这种不确定性归为网络容量缺陷。

## 7. 旧 EA 结果实际消费链及边界

已在 `frozen/local_history/tmp_20260906.zip` 内找到 `plan_dt_v2_relaxed_dev64_1seed.json`，其中 `source_arms[2]` 明确使用：

`/root/autodl-tmp/n2p3-cloud-inputs/bi2014a_ea_target_calibration_relaxed.npz`

其 arm 为 `bi2014a_ea_target_calibration`、EA mode 为 `target_calibration/native_preserve`，source 为 16 epochs/batch512，target full-fine 为 15 epochs/batch32/lr.0001。`analysis_relaxed_native.json` 则保存已执行的 960 个 result 身份，其中 192 个是三个 EA 方法×64 人，source snapshot SHA 为 `fcbc6c11bced1271ca6235348c83b142c63eb4c540af6fbf6d2f6458b360d698`（8ddab10）。详见 `ea_consumer_trace.json`。

**计划副本有漂移：** 留存 plan 的 `source_snapshot_manifest` 已是 2a75407，实际分析绑定的是 8ddab10。不能拿这个计划副本覆盖真实执行身份。原 execution manifest SHA 存在，但该 manifest 的完整原件未在本地找到；因此本轮只能把 arm、计划 filename 与执行 analysis 对上，尚未把旧 EA cache 的内容 SHA 直接绑定到原执行 manifest。

输入审计另发现 eligible61 派生 EA cache 先对 64 人拟合 W 再删 62–64；该问题是 **09-06 之后派生 cache 的身份污染**。本轮只找到其 retirement 替换账本以及 09-18 云端派生 cache 清理收据，未找到后续真实训练消费 eligible61 的 record。因此不能将这项污染倒扣给当时仍是合法 64 人开发口径的 09-05 run，更不能用它解释不消费 EA 的 plain unified12。

EA 的 V² 尺度相对于固定 `1e-6` eigenvalue floor 问题，可追溯相同旧代码路径；其数值复现属于并行输入审计。即便存在这个算法实现问题，也必须保持“旧 EA 候选结果失去方法忠实性”与“当前主线低分原因”两种结论的区分。

## 8. 当前证据缺口与下一步读取目标

1. **08-28 高分原始 record/NPZ 缺口。** 已在 `frozen/research_evidence_through_20260901-d1db8e4.tar.gz` 中读到 `doc/prior_free_unfold_result_20260828.zh.md` 及原协议；其 `experiments/runs/bi2014a_prior_free_unfold_ablation_20260828_r1/` 在当前本地未找到，冻结 tar 自述不含 raw EEG/model。不能宣称已经重算 `0.74511`。
2. **09-18 原包本地缺口。** `results/archive_receipt.json` 指向云端 `verified_execution.tar.gz`，bytes `11,990,531,333`，SHA `db3456fada03f6cc7177c225b822c4884aef4e2204c646218c029ef0bd99c790`、19,779 verified members。本地仅有分析和收据，未找到该包或逐 cell；48 指 target shard jobs，不是 48 个独立 source fit。它的缺口不影响本轮对 09-16 已有原包的结论。
3. **拆因果需统一执行身份。** 优先准备与历史 causal 同输入、同 test 行、同 target K、同 seed 的配方桥接；一次只更换预算、BN/dropout 或输入合同。此前高分及当前低分只提供现象，不能按哪个分数最高回选窗口/正则/路由。

本报告完成的是证据可比性与已保存结果排查，不是新训练配方的性能验证。小型研究脚本、摘要、报告和哈希清单保存在当前 `doc/assets/.../comparability/`；大型派生数据仅存本地忽略的 `experiments/runs/deep_chain_audit_20260929/comparability/`。既有结果、缓存、源码和快照均保留原样。
