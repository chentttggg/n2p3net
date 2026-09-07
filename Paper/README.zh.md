# N2P3-Net 文献与证据库

更新：2026-09-07

## 论文写作入口

- [`literature_review_n2p3net_20260903.zh.md`](literature_review_n2p3net_20260903.zh.md)：面向论文 Related Work/Introduction 的工程化文献综述底稿，已把外部文献、当前源码和实验边界分层。
- [`literature_manifest_20260828.md`](literature_manifest_20260828.md)：68 个去重来源、检索台账、访问边界和失败记录。
- [`pdf_validation_20260828.json`](pdf_validation_20260828.json)：本地 PDF 的页数、SHA-256、可读性和首段文本校验。
- [`receptive_field_literature_ledger_20260828.md`](receptive_field_literature_ledger_20260828.md)：感受野、padding、stride、dilation 与时序 CNN 的定向检索补充。
- 项目实测与外部方法边界见[证据索引](../doc/evidence/README.md)，代码合同与验证规则见[工程指南](../doc/engineering.zh.md)。旧证据账本和框架验收正文已退役，原路径与哈希见[归档清单](../frozen/technical_audit_before_20260906.manifest.json)。

## 证据边界

本目录的 PDF 是可审计原文，不代表论文结论已被 N2P3-Net 复现。论文中的外部性能比较必须同时写明数据集、被试划分单位、目标被试标签量、候选数、repetitions 和 metric；缺少任一项时，只能用作机制假设。

当前工程主线是 `src/models/n2p3net.py` 中的 MS-EEGNet-style compact CNN，配合集中式 EEG/data contract、fold-local QC、subject/cross-decision split 和 candidate-evidence 评价。旧架构、历史实验或非匹配 EEG foundation-model 结果不得作为默认路线。

## 主题索引

| 主题 | 主要原文 |
|---|---|
| compact / multi-scale P300 | `EEGNet_Compact_CNN_JNE_2018.pdf`, `MS_EEGNet_P300_Frontiers_2021.pdf`, `Few_Filters_P300_arXiv_1909.06970.pdf` |
| subject-independent / zero-calibration | `Zero_Training_Online_P300_CNN_ACCESS_2020.pdf`, `Invariant_Patterns_Subject_Independent_P300_TNSRE_2021.pdf`, `Large_Multisubject_P300_CNN_arXiv_2001.04225.pdf` |
| latency / ERP morphology | `Improving_P300_Morphology_Single_Trial_Latency_Realignment_JNE_2026.pdf`, `Estimating_Event_Related_Potential_from_Few_EEG_Trials_arXiv_2511.23162.pdf` |
| transfer / domain shift | `EEG_Cross_Dataset_Variability_2020.pdf`, `xDAWN_Riemannian_Transfer_P300_2020.pdf`, `Adaptive_Split_MMD_P300_arXiv_2510.21969.pdf` |
| candidate / sequential decision | `Dynamic_Stopping_Subject_Independent_P300_GLOBECOM_2017.pdf`, `MultiTrial_P300_Character_arXiv_2410.08561.pdf`, `P300_Speller_Advances_Review_2022.pdf` |
| self-supervision / foundation models | `SpellerSSL_arXiv_2509.19401.pdf`, `EEG_FM_Compass_arXiv_2601.17883.pdf`, `NeuralBench_arXiv_2605.08495.pdf`, `OmniEEG_Bench_arXiv_2606.00815.pdf` |
| negative controls / reproducibility | `EEG_Deep_Learning_Data_Leakage_2024.pdf`, `Identity_Trap_in_EEG_Foundation_Models_arXiv_2606.06647.pdf`, `MOABB_Reproducibility_Benchmark_arXiv_2404.15319.pdf` |

完整科研路线与当前证据状态见 [`../doc/research_program.zh.md`](../doc/research_program.zh.md)。被吸收的 dated reports 和候选综述保存在 `../frozen/document_reports_through_6c4abe8_20260903.tar.gz`，不再作为活动写作入口。
