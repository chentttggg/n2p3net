# N2P3-Net 文献库索引

更新：2026-08-28

## 检索范围

本轮通过 arXiv API、OpenAlex、DOI/期刊开放页面和作者公开稿检索，覆盖：

- P300/ERP 的跨被试、零/少校准、跨数据集和在线适配；
- 单 trial latency jitter、ERP estimation、候选聚合和 dynamic stopping；
- compact CNN、multi-scale、attention/graph/Transformer；
- 自监督/联合监督、MMD/Riemannian/Bayesian transfer；
- EEG foundation model 的身份捷径、低频偏置、负对照和 benchmark protocol；
- 重参考、通道/坐标、伪迹和可复现性。
- theoretical/effective receptive field、padding、stride、dilation 与 temporal CNN。

截至本次更新，`Paper/` 中 53 个 PDF 均已通过结构与首段文本校验，没有损坏或加密 PDF。
“已下载”只表示原文可审计，不表示论文结论已被本项目接受。
完整 68-source 检索/失败清单见 `literature_manifest_20260828.md`，逐 PDF 的
SHA-256、页数和首段可读性见 `pdf_validation_20260828.json`；本文只保留决策摘要。

## 深度核对后的路线判断

| 本地文件 | 直接证据边界 | 本项目用途 |
|---|---|---|
| `Zero_Training_Online_P300_CNN_ACCESS_2020.pdf` | 55 人 LOSO/session test，另 12 名真实在线新被试；15 repetitions | 零校准多被试 supervised 主线 |
| `Invariant_Patterns_Subject_Independent_P300_TNSRE_2021.pdf` | 150 人训练、独立 50 人测试；40-choice 聚合 | 大规模 supervised + 短校准回退主线 |
| `MultiTask_SSL_RSVP_TNSRE_2024.pdf` | 3 个 RSVP 数据集，但主要是同人时间切分 | `L_cls + auxiliary SSL` 机制消融 |
| `SpellerSSL_arXiv_2509.19401.pdf` | II-A 预训练、II-B 适配/测试，本质仅两个源被试 | time-mask/FFT/G=2 机制对照，数字不可外推 |
| `Estimating_Event_Related_Potential_from_Few_EEG_Trials_arXiv_2511.23162.pdf` | P300 数据 55 人，38/5/12 subject split；目标是 ERP R2 | few-trial ERP 与 predictive variance 机制 |
| `Improving_P300_Morphology_Single_Trial_Latency_Realignment_JNE_2026.pdf` | 模拟 + 10 名真实被试；不评估分类 | latency 约束、shift/jitter/伪峰反例 |
| `ERP_XTTN_arXiv_2606.02939.pdf` | cross-subject ERP benchmark | 外部 specialist 水平和原型注意力对照 |
| `Adaptive_Split_MMD_P300_arXiv_2510.21969.pdf` | 两个 40 人数据集；目标 10 trial/人后混池随机 CV，非 LOSO | 仅 class-conditional MMD 对照 |
| `Bayesian_Signal_Matching_ERP_Transfer_arXiv_2401.07111.pdf` | 新被试仍用 5 字符×5 序列；MCMC 成本高 | 结构化候选后验/不确定性对照 |
| `Active_Sampling_Multicentre_P300_arXiv_2412.17833.pdf` | 17 人；用 test accuracy 选样本量，只报 raw accuracy | 不采用 |
| `Adaptive_Semisupervised_P300_arXiv_2602.15955.pdf` | 15 人；结果/公式不足以复现强结论 | 不采用当前公式版本 |
| `CrossScale_Transformer_Domain_Rectified_RSVP_TNSRE_2024.pdf` | 目标人 60% fine-tune；约 964k 参数 | 重型容量上界，不作 few-shot 证据 |

## 必读方法与负对照

| 主题 | 本地文件 | 使用方式 |
|---|---|---|
| compact P300 backbone | `EEGNet_Compact_CNN_JNE_2018.pdf`、`MS_EEGNet_P300_Frontiers_2021.pdf`、`Few_Filters_P300_arXiv_1909.06970.pdf` | matched compact floors |
| large multi-subject P300 | `Large_Multisubject_P300_CNN_arXiv_2001.04225.pdf` | 验证“大数据优先于大模型” |
| Riemannian transfer | `Riemannian_Means_Field_BCI_arXiv_2504.17352.pdf`、`Riemannian_Transfer_Minimal_Calibration_arXiv_2111.12071.pdf` | classical transfer floor |
| robust covariance/QC | `Riemannian_Artifact_Subspace_Reconstruction_2019.pdf` | 伪迹协方差对照，不直接替换 fold-local QC |
| candidate/dynamic decision | `Dynamic_Stopping_Subject_Independent_P300_GLOBECOM_2017.pdf`、`MultiTrial_P300_Character_arXiv_2410.08561.pdf` | hit-speed/abstain 研究 |
| reference/montage | `P300_Rereferencing_arXiv_2510.10733.pdf`、`P300_Spatial_ROI_arXiv_2511.02735.pdf` | 重参考和坐标 stem 前置证据 |
| Bayesian ERP | `P_SMGP_Bayesian_ERP_arXiv_2605.30775.pdf`、`Sparse_Bayesian_Channel_Interactions_P300_arXiv_2602.17772.pdf` | uncertainty/channel interaction 对照 |
| 数据泄漏 | `EEG_Deep_Learning_Data_Leakage_2024.pdf` | subject/session split 门禁 |
| 跨数据集漂移 | `EEG_Cross_Dataset_Variability_2020.pdf` | 不同 source 不可盲目 concat |
| 身份捷径 | `Identity_Trap_in_EEG_Foundation_Models_arXiv_2606.06647.pdf` | 独立 subject probe/erasure 诊断 |
| 低频偏置 | `Understanding_and_Correcting_Low-Frequency_Bias_in_EEG_Foundation_Models_arXiv_2608.01898.pdf`、`Spectral_Audit_Task_Dependent_Aperiodic_Reliance_arXiv_2606.08583.pdf` | band balance 与敏感性分析；不作单项判死 |
| benchmark 负对照 | `EEG_FM_Negative_Control_Protocol_arXiv_2607.24519.pdf`、`EEG_Task_Specification_NeuroDoc_arXiv_2606.22925.pdf`、`MOABB_Reproducibility_Benchmark_arXiv_2404.15319.pdf` | executable protocol/ledger |
| FM 对照 | `EEG_FM_Compass_arXiv_2601.17883.pdf`、`EEG_FM_Generalization_Framework_arXiv_2605.28563.pdf`、`NeuralBench_arXiv_2605.08495.pdf`、`OmniEEG_Bench_arXiv_2606.00815.pdf` | specialist 先行，FM 后置 |
| 感受野方法 | `Effective_Receptive_Field_Luo_NeurIPS_2016.pdf`、`Temporal_Convolutional_Networks_Lea_CVPR_2017.pdf` | RF/ERF 计算与时序层级先例，不作 P300 性能证据 |

## 仍需谨慎解读

- `Calibration_Free_P300_2022_Accepted.pdf` 使用模拟在线反馈；不等于真实零校准部署。
- `ST_CapsNet_P300_TNSRE_2023.pdf` 和 `ASPEN_Cross_Subject_arXiv_2602.16147.pdf`
  改变结构、容量和训练协议，只能在 compact floors 稳定后进入。
- `Channel_Reflection_EEG_Augmentation_arXiv_2412.03224.pdf` 需要可信左右对称通道；
  3 导 GTN 不满足通用通道反射前提。
- `Contrastive_ERP_CNN_arXiv_2407.04738.pdf` 支持对比机制探索，但必须与相同 backbone
  的 supervised 对照匹配。
- `EEG_FM_BCI_Diverse_Features_arXiv_2506.01867.pdf`、`BLPM_*`、`EEG_PRISM_*`
  只提供通用表示研究背景，不是 GTN 85% 的必要路径。

## 引用纪律

每次引用外部数字必须同时写明 dataset、被试数、split unit、目标人标签量、候选数、
repetitions 和 metric。若缺任一项，只能用论文生成假设，不能放进性能排名表。

完整科研决策见 `../doc/research_program.zh.md`。
