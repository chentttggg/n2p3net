# N2P3-Net 文献综述：面向跨被试 P300 解码的工程化证据

> 版本：2026-09-03。本文是论文 Related Work/Introduction 的证据底稿，不是 PRISMA 系统综述。检索与全文边界见 [`literature_manifest_20260828.md`](literature_manifest_20260828.md)；本项目实现和实验状态以 [`../doc/research_program.zh.md`](../doc/research_program.zh.md) 与 [`../doc/research_status_report_20260902.zh.md`](../doc/research_status_report_20260902.zh.md) 为准。

## 1. 范围与证据规则

本综述只讨论与本项目直接相关的五个问题：

1. 单试次 P300/ERP 的紧凑时空建模；
2. 跨被试、跨数据集和少校准泛化；
3. P300 潜伏期变化、预刺激基线与多试次证据累积；
4. 候选选择、动态停止和 operational 评价；
5. EEG 基础模型/自监督方法在小通道 P300 场景中的适用边界。

证据分为四层：

| 层级 | 含义 | 在本文中的用法 |
|---|---|---|
| E1 | 同行评审且直接研究 P300/ERP 或其 speller 协议 | 支持方法背景和可比机制 |
| E2 | P300/ERP 预印本、会议稿或接受稿 | 支持候选机制；性能数字必须保留协议限制 |
| T | 非 P300 的 EEG 方法、基准或理论工作 | 只能支持可迁移方法假设 |
| P | 本项目源码、合同、测试和运行制品 | 说明 N2P3-Net 实际做了什么 |

“本地 PDF 已下载/可解析”只说明原文可审计，不等于结论已被本项目复现。外部论文的数值只有在同时给出数据集、被试划分单位、校准预算、候选数、重复次数和指标定义时，才可进入比较表；否则仅用于提出假设。

## 2. 问题定义：P300 不是孤立的二分类

P300 speller 的观测首先是单次刺激后的 ERP trial。二分类标签通常表示 target/non-target，但部署目标是从固定候选词汇 (V) 中选择一个候选。对候选 (d) 和重复前缀 (r)，项目的统一语义是先累积该候选的 trial evidence，再进行 row/column 或 candidate-level 选择：

\[
S_{d,r}=\operatorname{Aggregate}\{e_i:c_i=d,\operatorname{rep}_i<r\},
\qquad
\hat d_r=\arg\max_{d\in V}S_{d,r}.
\]

因此，trial AUC/BACC、character/selection accuracy、`hit@R`、动态停止时间和 ITR 是不同 estimand，不能互换。Pan 等的 P300 speller 综述强调范式、刺激时序和性能评价共同决定系统结果；多试次研究也表明，重复聚合后的字符/候选结果不能由单 trial 指标直接推出（`P300_Speller_Advances_Review_2022.pdf`、`MultiTrial_P300_Character_arXiv_2410.08561.pdf`）。本项目把 candidate vocabulary、selection/decision group、repetition index 和 coverage 写入数据合同，正是为了避免把“二分类得分高”误写成“speller 选择可靠”。

## 3. 紧凑 CNN 与多尺度时间建模

EEGNet（Lawhern et al., 2018）以时间卷积、按通道空间卷积和深度可分离卷积构成参数高效的 EEG 基线。其价值不是证明小模型普遍优于大模型，而是提供了可在不同范式、通道数和样本量下复核的结构基线（E1，`EEGNet_Compact_CNN_JNE_2018.pdf`）。P300 专门研究进一步显示，少量卷积滤波器也可能足够，模型容量、训练数据和评估协议需要同时控制（E1，`Few_Filters_P300_arXiv_1909.06970.pdf`）。

MS-EEGNet（Borra et al., 2021）在 EEGNet 风格的空间投影上使用并行时间尺度，并在多个 P300/odd-ball 数据集上与 EEGNet、BranchedNet、OCLNN 和 xDAWN+RG 比较。其直接启示是：P300 的峰形和潜伏期存在跨 trial/被试变化，多尺度时间分支是可检验的归纳偏置；但结果并非所有数据集都一致，不能写成“MS-EEGNet 总是更优”（E1，DOI `10.3389/fnhum.2021.655840`，`MS_EEGNet_P300_Frontiers_2021.pdf`）。大规模多被试 CNN 的研究则提示，数据覆盖与 subject-independent 训练本身可能比继续增大网络更关键（E1，`Large_Multisubject_P300_CNN_arXiv_2001.04225.pdf`）。

本项目的 `N2P3Net` 是这一紧凑、多尺度路线的工程化实现，而不是新的基础模型：`src/models/n2p3net.py` 明确包含 temporal convolution、全通道 spatial projection、并行 MST branches、可审计 pooling/readout 和 `forward_features` 接口。当前默认结构使用 35-sample ST kernel、5/17-sample MST kernels、depthwise spatial projection 与受限空间权重；感受野由 kernel、pool、stride/dilation 和采样率的显式函数计算，替代“层数越深越好”的叙述。`full_unfold`、`global_average`、`ms_flatten` 和 latency-contrast 是 readout 假设，不能在论文中合并称为同一个模型。

## 4. ERP 形态、潜伏期与预刺激信息

P300 的单 trial 形态和 latency jitter 会削弱简单平均、固定时间窗和单一 temporal kernel。Quattrociocchi 等的研究专门比较 single-trial latency realignment，支持把潜伏期变化作为形态建模问题，但其目标是 ERP 形态重建而非证明分类增益（E1，DOI `10.1088/1741-2552/ae7766`，`Improving_P300_Morphology_Single_Trial_Latency_Realignment_JNE_2026.pdf`）。这为时间多尺度和 latency-aware readout 提供机制依据，却不能直接推出本项目的目标域 hit@R 会提高。

当前 N2P3Net 的 `latency_marginal_contrast` readout 在预刺激窗口与 250--600 ms evidence window 之间构造对比；`forward_features` 则允许在 pooling 前对结构化时空特征施加约束。该接口使“预刺激基线是否有用”“latency bank 是否只是在记忆时间窗”“off-window artifact 是否被误用”可以作为独立消融，而不是把它们混入主干结构的效果。`src/data/contract.py` 集中定义了不同 cohort 的可执行合同：默认离线 recipe 为 128 Hz、([-200,1200)) ms、0.1--30 Hz、`mean_only`；论文对齐的 GTN offline/causal 合同另为 0.5--30 Hz。论文必须把这些作为可复核的 analysis recipe，而不是声称所有原始数据都以同一硬件合同采集。

## 5. 跨被试与跨数据集泛化

跨被试 P300 文献有三条互补路线。第一条是大规模 subject-independent supervised CNN：`Zero_Training_Online_P300_CNN_ACCESS_2020.pdf` 报告了离线 subject-held-out 与真实新被试在线实验，说明“大数据 + 严格新被试测试”可以减少校准；`Invariant_Patterns_Subject_Independent_P300_TNSRE_2021.pdf` 进一步将 invariant pattern 与大 EEG 数据结合。它们支持本项目设置 subject-disjoint baseline，但其候选数、重复次数和在线反馈协议与 BrainSync/GTN 不同，不能直接转移准确率。

第二条是传统或几何迁移：xDAWN+Riemannian 方法和 Riemannian transfer 可作为低容量 classical floor，说明协方差几何、空间滤波和 recentering 在少校准下有价值（E1/E2，`xDAWN_Riemannian_Transfer_P300_2020.pdf`、`Riemannian_Transfer_Minimal_Calibration_arXiv_2111.12071.pdf`）。但这些方法不替代本项目的 fold-local artifact QC、通道/坐标合同和候选证据状态。

第三条是显式 domain adaptation。跨数据集 variability 研究显示采样率、通道、参考、刺激设计和设备差异会形成非平凡 domain shift（T，DOI `10.3389/fnhum.2020.00103`，`EEG_Cross_Dataset_Variability_2020.pdf`）。P300 的 Split-MMD 预印本将 target weighting、Split-BN 和 MMD 组合起来，但其主要实验是同批被试的随机 trial split，不能证明 unknown-subject/unknown-decision transfer（E2，`Adaptive_Split_MMD_P300_arXiv_2510.21969.pdf`）。MS-MDA 和 RSVP domain-rectified Transformer 说明 domain-specific normalization、conditional alignment 和 shared/private decomposition 是可研究机制，但它们不是 GTN candidate decision 证据（T，`CrossScale_Transformer_Domain_Rectified_RSVP_TNSRE_2024.pdf` 及 `doc/feature_domain_transfer_research_20260902.zh.md`）。

本项目工程裁决因此不是“加一个 adapter 就完成迁移”：

- `N2P3Net` core 保持 domain-agnostic；迁移路径由 `src/models/adapters.py` 的 `FeatureDomainTransfer` 包裹；
- invariant/private 表示、conditional alignment、target-only residual adapter 和 candidate decision 层属于不同参数/数据边界；
- target calibration 只能更新低容量 adapter/evidence calibration，不能访问 later test decisions；
- `no-transfer`、domain ceiling、domain LODO 和 cross-decision 是必要对照；
- domain probe 或 feature distance 下降只是机制证据，不能替代 operational candidate hit@R。

当前 BNCI-target development matrix 观察到 pooled multi-source CE 的负迁移：`joint_natural` 的 hit@8 低于 `bnci_only`，80/20 域权重只能恢复而未超过单域基线（详见 [`research_status_report_20260902.zh.md`](../doc/research_status_report_20260902.zh.md)）。该矩阵的证据链仍需按冻结 manifest 重建，因此这里将其作为探索性/开发证据，而不是确认性结论或对所有 P300 数据集的普遍定理；它支持继续研究条件对齐和梯度冲突，而不支持直接宣称“多源预训练有效”。

## 6. 少校准适配：哪些证据可以迁移

P300 零校准与短校准工作（`Calibration_Free_P300_2022_Accepted.pdf`、`Bayesian_Signal_Matching_ERP_Transfer_arXiv_2401.07111.pdf`）共同指出：新被试的少量信息应被用于 source matching、校准或不确定性估计，而不是默认解冻整个网络。Bayesian signal matching 还提示，当没有可靠 source 时应允许 reference/no-borrow fallback；这一点与本项目的 fail-closed 迁移策略一致。

MAML、Reptile、原型网络和 LoRA 是一般 few-shot 方法，但现有证据主要来自睡眠分期、运动想象或大模型，不能被写成 P300 已验证方案。`doc/literature/literature_review_few_shot_adaptation.md` 的候选排序应理解为工程假设而非结论：本仓库尚无 matched P300 LOSO 证据证明 FiLM、Reptile 或 LoRA 优于当前 B0。尤其是 N2P3Net 只有紧凑规模，LoRA 的大模型参数效率优势不能直接外推；原型均值也未解决 P300 trial-to-trial latency/morphology variance。

因此当前可接受的适配研究是低容量、可回退和可审计的 residual adapter：identity initialization、显式残差上界、target-only update、无 target BN running-state 漂移、无 source-domain ID 依赖。只有在合法 target calibration/cross-decision 协议中相对 B0 稳定提升 operational hit@R，才可将其写成迁移增益。

## 7. 自监督与 EEG 基础模型：作为对照，不是默认答案

LaBraM、BENDR、EEGPT、BIOT 以及近年的 EEG foundation-model benchmark 展示了跨数据集预训练的潜力，但现有公开材料没有提供与本项目相同的“小通道、P300 trial、subject-disjoint、candidate decision”匹配证据。`EEG-FM-Compass`、`NeuralBench` 和 `OmniEEG-Bench` 的重要贡献是要求统一 task specification、split、数据规模和 downstream adaptation；它们不能被简化成“foundation model 必然胜过 EEGNet”。

负面对照尤其重要。`EEG_Deep_Learning_Data_Leakage_2024.pdf` 说明随机 epoch split 可产生乐观偏差；`Identity_Trap_in_EEG_Foundation_Models_arXiv_2606.06647.pdf` 提醒 subject identity 可能成为捷径；低频偏置工作提示 reconstruction pretraining 可能过度利用 1/f 成分。对本项目而言，这些文献转化为 executable controls：participant-level split、identity probe/erasure、频带敏感性、negative-control protocol、source/cache/checkpoint digest，以及不把 generic masked reconstruction 设为默认训练目标。

自监督可以作为 matched auxiliary arm（例如 time-mask/contrastive ERP），但必须与相同 N2P3Net、相同 source/target split、相同 QC 和相同 candidate metric 比较。`MultiTask_SSL_RSVP_TNSRE_2024.pdf` 与 `SpellerSSL_arXiv_2509.19401.pdf` 只能支持“联合监督/自监督值得消融”，不能支持本项目 GTN 或 BrainSync 性能数字。

## 8. 候选聚合、listwise loss 与动态停止

P300 文献通常把 trial detector 和 character/candidate decoder 分开；动态停止工作进一步把准确率与刺激成本联系起来。项目当前的 candidate contract 要求冻结 vocabulary、完整 evidence coverage、显式 tie/incomplete/fit-failure 状态和请求分母。对 prefix (r)，只有所有候选均有合法 evidence 时才计算 posterior 或 argmax；将缺失候选置为 (-\infty) 后继续选择会改变 estimand，属于无效结果。

候选集 softmax/listwise loss 可作为决策层机制研究：

\[
\mathcal L_{set,r}=-\log\frac{\exp(S_{d^*,r}/\tau)}{\sum_{d\in V}\exp(S_{d,r}/\tau)},
\qquad
\mathcal L_{prefix}=\sum_r\alpha_r\mathcal L_{set,r}.
\]

它把错误候选之间的相对竞争纳入训练，但不等于直接优化 `hit@R`；proper probabilistic loss 与 0--1 selection loss 可能排序相反。ListNet/ListMLE 是一般 learning-to-rank 理论来源，而不是 P300 直接证据。当前本项目的正确顺序是：先闭合 candidate eligibility、subject-macro `hit@R` 和 group-preserving batching，再做 binary CE、pairwise 和 candidate softmax 的 matched ablation；`L_candidate` 单独训练不能晋升。

动态停止同样只能作为独立策略 arm。必须同时报告 operational accuracy、coverage、abstain/tie/failure、实际 repetition、秒数、刺激数和正确选择/分钟；名义 ITR 只在固定候选数和明确时间口径下作为辅助指标。不能以删除 incomplete decision 或只报告 eligible accuracy 换取更高结果。

## 9. 本项目的定位与可证伪缺口

综合现有证据，N2P3-Net 的贡献边界应写成：

1. 一个可审计的 MS-EEGNet-style compact CNN，用显式物理输入合同和感受野记录连接 EEG 工程参数与模型结构；
2. 一个把 source-only baseline、跨域表示约束、低容量 target adapter 和 candidate decision 分开的研究框架；
3. 一个将 trial-level discrimination 与 candidate-level operational outcome 分开的评价合同；
4. 一组针对数据泄漏、身份捷径、缺失候选、tie、causal leakage 和负迁移的反例门禁。

当前不能写成已证实的结论包括：feature alignment 已经提升 target hit@R、bilevel adaptation 已经有效、基础模型优于 N2P3Net、或 BrainSync 产品已达到 90%。这些问题需要新实验，并且必须绑定 source snapshot、数据/cache、checkpoint、环境、plan digest、请求/合格/成功分母和 participant-cluster CI。

## 10. 核心参考文献与本地证据入口

下表只列入论文主文建议优先核对的核心来源；完整去重清单、失败检索和 SHA-256 见 [`literature_manifest_20260828.md`](literature_manifest_20260828.md)。

| 主题 | 文献 | 证据边界 | 本地文件 |
|---|---|---|---|
| 紧凑 EEG CNN | Lawhern et al., 2018, DOI `10.1088/1741-2552/aace8c` | EEGNet 基线；非 P300 专属 | `EEGNet_Compact_CNN_JNE_2018.pdf` |
| 多尺度 P300 CNN | Borra et al., 2021, DOI `10.3389/fnhum.2021.655840` | 直接 P300；不同数据集结果混合 | `MS_EEGNet_P300_Frontiers_2021.pdf` |
| 少滤波器 P300 | `10.1016/j.neucom.2020.10.104` | 直接 P300；协议需匹配 | `Few_Filters_P300_arXiv_1909.06970.pdf` |
| 大规模跨被试 CNN | `10.1016/j.bspc.2019.101837` | subject-independent P300；指标/候选协议不同 | `Large_Multisubject_P300_CNN_arXiv_2001.04225.pdf` |
| 零训练在线 P300 | `10.1109/ACCESS.2020.2988057` | 离线 held-out + 新被试在线；非本项目合同 | `Zero_Training_Online_P300_CNN_ACCESS_2020.pdf` |
| invariant P300 | `10.1109/TNSRE.2021.3083548` | 大数据 subject-independent；候选数不同 | `Invariant_Patterns_Subject_Independent_P300_TNSRE_2021.pdf` |
| xDAWN/Riemannian | `10.3390/app10051804` | classical transfer floor | `xDAWN_Riemannian_Transfer_P300_2020.pdf` |
| 跨数据集漂移 | `10.3389/fnhum.2020.00103` | EEG transfer protocol evidence；非 P300 性能证据 | `EEG_Cross_Dataset_Variability_2020.pdf` |
| latency realignment | `10.1088/1741-2552/ae7766` | 形态/潜伏期机制；非分类增益证明 | `Improving_P300_Morphology_Single_Trial_Latency_Realignment_JNE_2026.pdf` |
| P300 speller 综述 | Pan et al., 2022, DOI `10.3389/fnhum.2022.1077717` | 范式与评价背景 | `P300_Speller_Advances_Review_2022.pdf` |
| 动态停止 | `10.1109/GLOCOM.2017.8255030` | character/ITR 轨道；不等于 trial AUC | `Dynamic_Stopping_Subject_Independent_P300_GLOBECOM_2017.pdf` |
| 数据泄漏 | Brookshire et al., 2024, DOI `10.1101/2024.01.16.24301366` | EEG split/泄漏警示；方法学证据 | `EEG_Deep_Learning_Data_Leakage_2024.pdf` |
| Bayesian source matching | Ma et al., DOI `10.1080/01621459.2025.2563189` | ERP transfer 与 no-borrow 启示 | `Bayesian_Signal_Matching_ERP_Transfer_arXiv_2401.07111.pdf` |
| conditional MMD | Chen & Delorme, arXiv `2510.21969` | P300 预印本；非 LOSO 强证据 | `Adaptive_Split_MMD_P300_arXiv_2510.21969.pdf` |
| 候选多试次 | Shukla et al., arXiv `2410.08561` | repetition-dependent character metric | `MultiTrial_P300_Character_arXiv_2410.08561.pdf` |
| 自监督 speller | Hong et al., arXiv `2509.19401` | 机制候选；source/subject 协议有限 | `SpellerSSL_arXiv_2509.19401.pdf` |
| foundation-model benchmark | Liu et al., arXiv `2601.17883` | benchmark/task specification；无匹配 GTN 结论 | `EEG_FM_Compass_arXiv_2601.17883.pdf` |
| identity shortcut | Lin et al., arXiv `2606.06647` | 诊断性负对照；非 P300 性能 | `Identity_Trap_in_EEG_Foundation_Models_arXiv_2606.06647.pdf` |

## 11. 写入论文时的最低引用纪律

- 外部性能数字必须紧跟 dataset、subject split、calibration、candidate/repetition 和 metric；
- 预印本、接受稿、会议稿和同行评审版本不能隐去发表状态；
- 工程事实引用源码/合同/测试，不能用文献替代；
- 本项目结果引用冻结 run、manifest 和报告，不把文献结果写成项目结果；
- 任何“优于”“有效”“泛化”都要指明 estimand 和证据层级，并同时报告负结果/不确定性。
