# G4 审计：doc/evidence 关键量化声明的 run 产物回溯

- 仓库：`D:\BCI\n2p3-net`；HEAD `f1a303c`，工作树 429 项未提交（审计期间未改变）。
- 分区：G4（补充分区），只做“文档数字 → `experiments/runs` 产物”的回溯，不重复 data/transfer/tests/工程/repo/docs/security/performance/Lean 分区的代码审查。
- 方法：grep 证据文档取声明 → 定位 run 目录 → 只读解析小体积 JSON/CSV → 用 `hashlib` 复算文件封印与本地归档 SHA。未加载 `.npz`/`.pt` 内容，未解压大归档逐成员读取，未运行验证脚本（部分脚本会写文件），未启动训练。
- 判定：**可回溯** = 产物中有同义字段且数值/分母/失败数一致或可重算；**部分** = 数字存在但语义/覆盖/可复核性有缺口；**不可回溯** = 找不到产物。级别 P0 结论错误；P1 证据链显著缺口或会误导的表述；P2 口径/可维护性。
- 只读声明：仅写本报告与 `tmp/audit/scratch/` 下的只读脚本；未改删任何既有文件，未执行 git 写命令。

## 结论摘要

| # | 声明 | 主要产物 | 判定 | 级别 |
|---|---|---|---|---|
| 1 | BNCI 八人 hit@all 0.4052/0.5905，覆盖 28/29，464 请求 448 评分，448 decision 重放一致 | `experiments/runs/system_implementation_20261004/bnci_continuous_real_v2/` | 可回溯 | P2（重放字段弱） |
| 2 | unified12 48/48 重放；K200 0.7057 | unified12 为 48 个**分片**、3,632 个模型；0.7057 不在 unified12，属 training_chain_auto | 计数可回溯；0.7057 归属需更正 | P1（表述） |
| 3 | P1/P2 45 源模型、16,599 拟合、32,238 评分、失败 0 | `experiments/runs/domain_transfer_cloud_20260916/` | 可回溯 | P2（README 未分解） |
| 4 | compact 52.1 分钟、6,138 NN、4,416 复用 | `experiments/runs/unified_cloud_20260916/` | 可回溯 | P2（命名歧义） |
| 5 | 本地归档逐成员校验 | P1/P2 与 09-16 compact 有本地 tar.gz；unified12 仅云端收据 | 部分（仅 unified12） | P2 |

**总体**：抽查的四组主数字没有发现算错或凭空生成；所有数字均能在 run 产物中找到对应字段，P1/P2 与 09-16 compact 还有本地归档 SHA 可复算。主要问题是**“模型数/分片数/decision 数”的口径转述**，以及 unified12 只有云端收据、重放容差未写入文档。

## 1. BNCI 八人连续诊断

**声明**：`doc/README.md:27`（hit@all 0.4052/0.5905、覆盖 28/29、448 个完整 decision 保存模型重放一致）；`doc/evidence/system_implementation_20261004.zh.md:132-143`（16 cell、0 拟合失败、464 请求/448 评分/16 不完整、448 份 36 维 log_evidence、两 K 表、A02 K200 NLL 33.1674、平均等待 576.73 s/有限核 30.95 s）；`doc/roadmap.zh.md:150`。

**产物**：`experiments/runs/system_implementation_20261004/bnci_continuous_real_v2/result.json`（16 cells + `by_K` + `result_integrity`）；`A01..A08/K{200,800}/predictions.json`（每份 29 条 decision）、`record.json`、`model.npz`；`scopes.json` 8 份；`executed_source/` 35 个源码文件；独立复算脚本/结果在 `doc/assets/system_implementation_20261004/verify_bnci_continuous.py` 与 `bnci_continuous_verification.json`。

**核对**：
- `result.json.by_K`：K200 `participant_macro_hit_all=0.4051724137931034`、K800 `0.5905172413793104`；两 K `participant_macro_coverage=0.9655172413793104`（=28/29）；NLL 6.9018254/3.2426752。与文档表 `system_implementation:135-138` 一致。
- 由 `cells[].requested_hit_all` 重算八人算术平均，得到同一 0.4052/0.5905；16 个 cell 的 `coverage` 全为 28/29，且 `by_K` 的 macro 值相同（两种口径碰巧都为 96.55%）。
- 16 份 `predictions.json` 共 464 条：`correct` 231 + `incorrect` 217 = 448 条可评分，`incomplete` 16 条；448 条 `log_evidence` 长度全为 36；448 条 `saved_reload_scores_exact` 全 true、0 false。与“464 请求、448 评分、16 不完整、448 份 36 维”逐字对应。
- `A02/K200 scored_nll_mean=33.1674166`；16 cell 平均 `mean_actual_offline_latency_s=576.7306`、`mean_finite_response_latency_s=30.9492`——33.1674、576.73、30.95 均可回溯。
- 8 份 `scopes.json` 每人 `requested_decisions=29`、`scorable_decisions=28`、`minimum_gap_seconds=8.25`；16 cell 状态全 `completed`、`fit_failure` 全 null，故“0 拟合失败”成立。
- 只读复算：16 个 `model.npz` + predictions/record 共 34 个受 `result_integrity` 封印的文件，字节数与 SHA-256 全部一致；35 个 `executed_source` 文件全部存在且 SHA 一致；v1 与 v2 的 `by_K` 完全相同（`v1_v2_metric_identity=true`）。
- 独立脚本 `verify_bnci_continuous.py:53-78` 重算 softmax、logsumexp−target、argmax/winner、hit/coverage/NLL；但脚本只断言 `saved_reload_scores_exact` 布尔（:71），没有自行 `ContinuousDecisionModel.load` 后重算。真正的重载相等在 runner 内：`experiments/run_bnci_continuous_study.py:103-104` 逐 decision `assert_array_equal` 两次 log_evidence，失败会抛出未捕获的 AssertionError；而字段本身只是 `nll is not None`（:127）。有文件 SHA 封印，手改会被发现，但该字段不是独立的密码学重放证据。**P2**。

**判定：可回溯**。数值、分母、失败/不完整数全部落到 run 产物并可重算；唯一保留是重放字段的强度。

## 2. unified12 与 0.7057 的真实归属

**声明**：`doc/README.md:54`（48/48 保存模型重放通过）；`doc/evidence/unified12_results_20260918.zh.md:7-10`（48 目标分片、4082 逻辑单元/3166 独立单元、916 重复、2062 NN/246 LDA/450 zero-shot/1324 复用、48/48 通过约 78 秒、归档 19779 成员约 11.18 GiB）。

**产物**：`experiments/runs/unified12_20260918/results/` 的 `verification.json`、`integrity_inventory.json`、`summary_initial.json`、`archive_receipt.json`。

**核对**：
- `verification.json`：`status=passed`、48 条 results 全 passed、`failures=[]`、`elapsed_seconds=78.0936`；48 个 record 路径唯一；`saved_models` 合计 3,632、`cells` 合计 4,082。**“48/48”是分片 job 数，不是 48 个模型**。
- 逐条 `maximum_prediction_difference` 最大 0.0003662109375，常见 1.9e-6～1.5e-5；复核脚本 `experiments/runs/unified12_20260918/verify_unified_results.py:51` 用 `rtol=1e-4, atol=1e-4`，故“通过”是容差内一致，非逐位相等。文档正文未写容差（对比 `unified_results_20260916.zh.md:32-33` 明写），`verification.json` 也无 tolerance 字段。**P2**。
- `integrity_inventory.json`：`source_jobs=48`（实为目标分片，09-29 勘误已说明）、`logical_cells=4082`、`normalized_cells=3166`、`duplicate_baseline_rows_verified=916`、`fits={neural:2062, lda:246, reused:1324, zero_shot:450}`——与文档逐项一致。
- `summary_initial.json.ordinary.descriptives` K200：A0Q0 0.6995、A0Q1 0.6971、A1Q0 0.7003、A1Q1 0.6895、LDA 0.6925、Scratch 0.6551、ScratchCE 0.6581，与 `unified12_results:22-28` 完全一致；四个主对比差值也一致（-0.0100、+0.0344 等）。
- `archive_receipt.json`：`members_verified=19779`、`bytes=11990531333`、SHA 与文档一致；但 archive 路径是云端 `/root/autodl-fs/...`，本地 run 目录仅约 14.8 MiB、无 tar.gz 副本，故 19,779 成员与包体无法本地逐字节复核。另 11,990,531,333 B = 11.167 GiB，文档“约11.18 GiB”高报约 0.01 GiB。**P2**。
- **0.7057 不在 unified12**：`summary_initial.json` 全文无 `0.7057`；unified12 的 K200 A0Q0 是 0.6995。0.7057 来自 `experiments/runs/training_chain_auto_20260907/completed_analysis/primary_metrics.json`：`full_fine/K200 auc=0.7057074933538198`、`target_only=0.6563828492270251`、`shrinkage_lda=0.6917514659532247`，差 +0.0493246；对应 `training_chain_results_20260907.zh.md:63-70` 的 0.7057/0.6564/0.6918/+0.0493。该文与 `doc/README.md:148` 同段（“扩展全量 E1/E3…六导完整微调”），不是十二库统一矩阵。

**判定：unified12 计数与 48/48 可回溯（收据级）；本地不能复跑重放、容差未声明 → 部分（P2）。0.7057 可回溯，但属于 training_chain_auto，不属于 unified12（P1 归属错误风险）。**

## 3. P1/P2 矩阵：45 源、16,599 拟合、32,238 评分

**声明**：`doc/README.md:89-91`；`doc/evidence/domain_transfer_cloud_20260916.zh.md:71-73,98-107,163-175`；`doc/roadmap.zh.md:351`。

**产物**：`experiments/runs/domain_transfer_cloud_20260916/` 的 `completion.json`、`compiled_counts.json`、`verification.json`、`matrix.json`、`analysis/ordinary/p1.json`、`analysis/cells.csv`、`export/results.local_verification.json` 与 1.63 GB `export/results.tar.gz`。

**核对**：
- `completion.json`：`status=completed`、`source_models=45`、`target_jobs=288`、`product_confirmation=false`。四组 counts：ordinary_P1 8,604 cells/0 failed/7,320 NN/915 LDA；family_P1 6,840/0/5,082/0；ordinary_P2 9,342/0/1,830/0；family_P2 7,452/0/1,452/0。合计 **32,238 cells、0 failed、15,684 NN、915 LDA**；复用 0+1,452+6,405+5,082=**12,939**；zero-shot 2,700。与 `compiled_counts.json.expected_*` 完全一致。
- **16,599 = 15,684 + 915**：README 的“目标拟合”是 NN+LDA 的总物理拟合；`verification.json.fitted_models` 的 29,538 是含复用副本的模型文件数，不能当作独立拟合（`domain_transfer_cloud:172-173` 已显式区分）。
- `analysis/cells.csv` 正好 32,238 行，`status` 全 `completed`；不依赖 completion 汇总即可确认分母与 0 失败。
- `verification.json`：288 条 results 全 passed、`failures=[]`、`cells` 合计 32,238、`fitted_models` 合计 29,538、最大绝对差 0.0003366299858384991、耗时 86.7169 秒、11 个 replay worker；与“288 分片全部通过、最大差 0.00033663”一致。
- 本地归档：`results.local_verification.json` 报 94,624 成员、29,538 models、32,238 predictions/cells、SHA `d9ec2c88…d97b9`；我用 `Get-FileHash` 重算 `export/results.tar.gz` 得同一 SHA，字节 1,628,798,632。`sources.local_verification.json` 报 257 成员、45 个源模型、SHA `09017472…`，重算 `sources.tar.gz` 同样一致。故该组不只停在云端收据。
- AUC 表九行全部可从 `analysis/ordinary/p1.json.groups[0].arm_descriptives` 复现：full 0.72494、scratch 0.71068、S1T1 0.70696、S0T1 0.69977、LDA 0.69175、S1T0 0.69088、affine+head 0.67468、residual+head 0.67108、S0T0 0.66837，n=123，与 `domain_transfer_cloud:17-25` 一致。

**判定：可回溯**。**P2**：README 的 16,599 与 32,238 未在句中分解；32,238 含 ordinary/family 两套重叠分析（102 人与 123 人重叠），正文 `domain_transfer_cloud:104-107` 已声明不可相加为独立参与者，README 单句未带此限定。

## 4. compact（09-16 统一精简矩阵）

**声明**：`doc/engineering.zh.md:322`（52.1 分钟、6,138 正式目标 NN、246 LDA、4,416 复用）；`doc/evidence/unified_results_20260916.zh.md:13-33`（6,138/246/4,416/12,150/52.1 分钟/10,800 模型/容差）。

**产物**：`experiments/runs/unified_cloud_20260916/analysis/integrity_inventory.json`、`matrix_matrix.json`、`verification_unified_verification.json`、`analysis/cells.csv`、`export/local_verification.json` 与 1.10 GB `export/verified_results.tar.gz`。

**核对**：
- `integrity_inventory.fits = {"neural":6138,"lda":246,"reused":4416,"zero_shot":1350}`；`logical_cells=12150`、`duplicate_baseline_rows_verified=2700`、`normalized_cells=9450`。6138/246/4416 与 engineering 行逐字一致。
- `matrix_matrix.json`：`status=completed`、`program=unified`、`target_groups=32`、`failed_groups=[]`、`failed_cells=0`、`elapsed_seconds=3126.7848796844482`（=52.113 分钟）；与“52.1 分钟”一致。
- `analysis/cells.csv` 正好 12,150 行且全 `completed`，与 `logical_cells` 及其“失败0”一致。
- `verification_unified_verification.json`：288 results 全 passed、`failures=[]`、cells 合计 12,150、`saved_models` 合计 10,800、最大差 0.0003662109375、重放 24.03 秒；文档“12,150 份预测及 10,800 份保存模型文件”可回溯。文档 32-33 行明写 `atol=rtol=1e-4` 与最大差 0.0003662，与实际一致。
- 本地归档：`local_verification.json` 报 `verified`、36,411 成员、10,800 models、12,150 predictions、SHA `59a7a982…dbbb`；重算 `verified_results.tar.gz` 得同一 SHA，字节 1,097,847,401。故 52.1 分钟结果有本地逐成员校验。
- **同名区分**：`experiments/runs/training_chain_compact512_20260907` 是 09-07 的 1,953 评分小矩阵（README:144-145），其 `analysis.json.compact_full_fine` K200=0.6995、scratch=0.6564、LDA=0.6918；它不是 09-16 的 6,138/4,416 矩阵，也不是 0.7057 表。

**判定：可回溯**。**P2**：engineering 单句把“246 LDA”与 6138 并列，未说明 4,416 复用与 1,350 zero-shot 的口径，也未说明 52.1 分钟含源阶段（正文 `unified_results:26-28` 有说明）。

## 5. 交叉抽查与口径问题

- **“448 个保存模型”是误读**：BNCI run 内只有 16 个 `model.npz`（每 subject×K 一个）；448 是 decision 级重放次数/36 维 log_evidence 条数。文档原句“448 个完整 decision 保存模型重放”尚可，转述成“448 个保存模型”会把模型数夸大 28 倍。**P1（表述）**。
- **“48/48 保存模型重放”是分片口径**：unified12 48 条 job 记录合计 3,632 个 `saved_models`、4,082 个 cells；48 是目标分片 job 数，不是 48 个模型。README:54 单句没有“分片”二字。**P1（表述）**。
- **0.7057 与 0.6995 不能混用**：0.7057 是 `training_chain_auto` 的 S1 源 + full-fine；0.6995 是 09-07 compact512 的 `compact_full_fine`；0.6995 也是 09-18 unified12 的 A0Q0。三者实验不同，数字接近但不可互换。
- **本地归档可用性差异**：P1/P2（1.63 GB）与 09-16 compact（1.10 GB）有本地 tar.gz，且重算 SHA 与 receipt 完全一致；unified12（11.17 GiB）只有云端 receipt，本地目录约 14.8 MiB。若审计要求“任何数字可在本地重跑”，unified12 不满足；若只要求“有签名收据”，则满足。
- **合计数复核**：P1/P2 32,238 = 8,604+6,840+9,342+7,452，15,684+915=16,599；compact 12,150 = 6,138+246+4,416+1,350。两组的 K0/zero-shot 均单列，未发现漏项或重复计入物理拟合。**P2**：README 的 16,599/32,238 未给分解。

## 6. 缺失字段与建议

1. **BNCI 重放证据（P2）**：`saved_reload_scores_exact` 只是 `nll is not None`；建议在 `predictions.json`/cell 中保存两次 log_evidence 的摘要（如 SHA-256）或重放差值，而不是布尔，便于不依赖 runner 代码判断。
2. **unified12 本地可复核性（P2）**：`verification.json` 的 record 路径为云端绝对路径，11.17 GiB 归档不在本地；建议补一份本地（或可获取的）归档副本/分片校验清单，或在证据正文明确“remote-only receipt”。
3. **容差字段（P2）**：unified12 `verification.json` 无 tolerance 字段，文档也未写；建议与 09-16 compact 一样记录 `atol/rtol` 与最大差。
4. **README 口径（P1/P2）**：把“448 个完整 decision 重放（16 个保存模型）”和“48 个分片重放（3,632 个模型）”写清；把 16,599 写成“15,684 NN + 915 LDA”，并保留 32,238 含重叠分析的限定。
5. **命名（P2）**：两个 “compact”（09-07 compact512、09-16 unified compact）在文档与 run 目录并存，引用时统一带日期/run 目录。
6. **数值舍入（P2）**：unified12 归档 11,990,531,333 B = 11.167 GiB，文档“约11.18 GiB”建议改为 11.17 GiB 或写十进制约 11.99 GB。

## 7. 已检查范围与未验证项

- **已检查**：上述 4 个 run 目录的聚合 JSON、`cells.csv`、`verification.json`、`completion.json`、`archive_receipt.json`、本地归档 SHA；BNCI 34 个封印文件与 35 个源码快照；`doc/README.md`、`doc/engineering.zh.md:322`、`doc/roadmap.zh.md:150/351` 与 4 份证据正文的相关段落。
- **未验证**：未运行 `verify_bnci_continuous.py`（会写 `doc/assets` 输出）、未运行云端 replay（需 GPU/远程路径）、未逐成员解压 1.63 GB/1.10 GB 归档（只验证了 SHA 与 receipt 的成员计数）；unified12 的 48 份 record/3,632 个模型不在本地，未能独立重放。
- **未覆盖**：测试计数（2263/2255/8 等）属 tests 分区；训练吞吐/GPU 采样属 performance 分区；run 目录源码快照完整性与绝对路径泄漏属 G8；本报告只在数字回溯层面引用这些字段。
- **只读声明**：未修改/删除/重命名任何既有文件；未执行 git 写命令；未安装依赖；未启动训练或长任务；未创建子代理；唯一写入为 `tmp/audit/G4-evidence-repro.md` 与 `tmp/audit/scratch/g4_*.py`。
