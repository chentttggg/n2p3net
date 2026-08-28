# P300/ERP literature manifest (2026-08-28)

## Scope and retrieval boundary

- Question: model-improvement evidence for P300/ERP decoding, with emphasis on cross-subject/domain generalization, latency jitter/alignment, multi-scale temporal modeling, xDAWN/Riemannian methods, spatial graphs, Transformer/SSM ideas, self-supervision/foundation models, uncertainty/calibration, repetition/candidate aggregation, and artifact robustness.
- Date boundary: 2018-2026 for recent work, plus selected classical papers required to interpret xDAWN, Riemannian, compact CNN, and dynamic-stopping baselines.
- Access date: 2026-08-28. This is a broad, reproducible evidence search, not a PRISMA/systematic review. Scopus, Web of Science, formal screening, and risk-of-bias assessment were not performed.
- Parallel Research/Search/Extract could not run because Parallel CLI 0.7.1 had no authenticated account or `PARALLEL_API_KEY`. No ordinary web search is represented as Parallel research.

## Retrieval ledger

| Surface | Bounded calls | Retrieved | Reconciliation |
|---|---:|---:|---|
| OpenAlex broad thematic search | 10 | 366 | 283 unique normalized titles; 9 searches intentionally truncated, sequential/repetition search complete at 6/6 |
| OpenAlex targeted search | 10 | 180 | 9 x 20 bounded results; exact `SpellerSSL` query returned 0 and was recovered through arXiv ID lookup |
| Europe PMC thematic search | 5 | 250 | 50 per query; large hit counts included biomedical `p300` false positives, so records were not accepted without EEG/BCI screening |
| OpenAlex classical-method search | 5 | 75 | 15 per query, intentionally bounded |
| OpenAlex sequential/calibration search | 5 | 112 | includes complete repetition aggregation 21/21 and exact dynamic-stopping DOI 1/1 |
| OpenAlex exact architecture lookup | 5 | 5 | ATCRN, ST-GraphTRNet, P3Net, EEG Conformer, ATCNet all resolved 1/1 |
| arXiv 2024-2026 P300/EEG/ERP query | 1 | 37 | complete 37/37; parser verified the executed query and rejected error-feed semantics |
| arXiv known-ID batch | 1 | 20 | complete 20/20 with abstracts, versions, journal DOI when registered, and PDF URLs |
| Crossref exact DOI validation | 25 | 25 | 25/25 resolved |
| Semantic Scholar exact OA batch | 1 | 7 | 7/7 resolved; used only to distinguish OA claims from actually retrievable files |

Final curated set: **68 unique sources** after DOI, arXiv ID, and normalized-title reconciliation. Of these, **53 PDFs are local and readable**; **15 are metadata-only, HTML/TeX method sources, indirect mechanism references, or failed legal-download attempts**. `pdf_validation_20260828.json` records SHA-256, pages, metadata, and first-page text extraction for every local PDF. All 53 parsed successfully and yielded first-page text.

### Focused receptive-field addendum

On 2026-08-28, a bounded follow-up queried OpenAlex (three 20-result searches plus exact IDs),
Crossref exact DOI, PubMed/Europe PMC P300 latency terms, and Unpaywall. Full request parameters,
count boundaries, transport failures, and retained-source rationale are preserved in
[`receptive_field_literature_ledger_20260828.md`](receptive_field_literature_ledger_20260828.md).
This addendum is methodological, not a
systematic review or direct GTN performance search.

## Evidence labels

- `D1`: directly about P300/ERP/RSVP decoding and peer-reviewed or an established benchmark/review.
- `D2`: directly relevant but a preprint, accepted manuscript, conference paper, or otherwise not equivalent to a matched GTN trial.
- `T`: transferable EEG method/protocol evidence; it is not direct P300 performance evidence.
- `M`: metadata/abstract only in this workspace; conclusions must remain limited to accessible metadata.

## Local direct P300/ERP sources

All rows below have `download_status=verified_local_pdf`.

| Year | Source | Identifier / canonical URL | Access | Level | Local file / boundary |
|---:|---|---|---|---|---|
| 2021 | Comparison of Convolutional and Recurrent Neural Networks for P300 Detection | [10.5220/0010248201860191](https://doi.org/10.5220/0010248201860191) | OA conference | D2 | `102482.pdf`; GTN-family data, reported accuracy is not LOSO AUC/BACC |
| 2024 | Transfer Learning with Active Sampling for Rapid Training and Calibration in BCI-P300 | [arXiv:2412.17833](https://arxiv.org/abs/2412.17833) | preprint | D2 | `Active_Sampling_Multicentre_P300_arXiv_2412.17833.pdf` |
| 2026 | Adaptive Semi-Supervised Training of P300 ERP-BCI Speller System | [arXiv:2602.15955](https://arxiv.org/abs/2602.15955) | preprint | D2 | `Adaptive_Semisupervised_P300_arXiv_2602.15955.pdf` |
| 2025 | Adaptive Split-MMD Training for Small-Sample Cross-Dataset P300 EEG | [arXiv:2510.21969](https://arxiv.org/abs/2510.21969) | preprint | D2 | `Adaptive_Split_MMD_P300_arXiv_2510.21969.pdf` |
| 2026 | ASPEN: Spectral-Temporal Fusion for Cross-Subject Brain Decoding | [arXiv:2602.16147](https://arxiv.org/abs/2602.16147) | preprint | D2 | `ASPEN_Cross_Subject_arXiv_2602.16147.pdf`; six datasets/three paradigms, not P300-only |
| 2024/2025 | Bayesian Signal Matching for Transfer Learning in ERP-BCI | [10.1080/01621459.2025.2563189](https://doi.org/10.1080/01621459.2025.2563189), [arXiv:2401.07111](https://arxiv.org/abs/2401.07111) | preprint/journal metadata | D2 | `Bayesian_Signal_Matching_ERP_Transfer_arXiv_2401.07111.pdf` |
| 2022 | Block-Toeplitz covariance LDA for ERP-BCI | [10.1088/1741-2552/ac9c98](https://doi.org/10.1088/1741-2552/ac9c98), [arXiv:2202.02001](https://arxiv.org/abs/2202.02001) | arXiv manuscript | D1 | `Block_Toeplitz_LDA_ERP_JNE_2022.pdf` |
| 2022 | A Calibration-free Approach to Implementing P300 BCI | [10.1007/s12559-021-09971-1](https://doi.org/10.1007/s12559-021-09971-1) | accepted manuscript | D1 | `Calibration_Free_P300_2022_Accepted.pdf` |
| 2024 | Contrastive Learning Based CNN for ERP-BCIs | [arXiv:2407.04738](https://arxiv.org/abs/2407.04738) | preprint | D2 | `Contrastive_ERP_CNN_arXiv_2407.04738.pdf` |
| 2017 | Subject-Independent P300 BCI with Ensemble Classifier, Dynamic Stopping and Adaptive Learning | [10.1109/GLOCOM.2017.8255030](https://doi.org/10.1109/GLOCOM.2017.8255030) | repository manuscript | D2 | `Dynamic_Stopping_Subject_Independent_P300_GLOBECOM_2017.pdf`; character/ITR outcomes differ from trial AUC |
| 2018 | EEGNet: A Compact CNN for EEG-based BCIs | [10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c), [arXiv:1611.08024](https://arxiv.org/abs/1611.08024) | author manuscript | D1 | `EEGNet_Compact_CNN_JNE_2018.pdf`; multi-paradigm baseline |
| 2026 | ERP-XTTN: Prototype-Guided Cross-Attention for Cross-Subject ERP Classification | [arXiv:2606.02939](https://arxiv.org/abs/2606.02939) | preprint | D2 | `ERP_XTTN_arXiv_2606.02939.pdf` |
| 2020 | A Few Filters Are Enough: CNN for P300 Detection | [10.1016/j.neucom.2020.10.104](https://doi.org/10.1016/j.neucom.2020.10.104), [arXiv:1909.06970](https://arxiv.org/abs/1909.06970) | preprint/published metadata | D1 | `Few_Filters_P300_arXiv_1909.06970.pdf` |
| 2023 | Variant of U-Net Using a Large P300 Dataset | local proceeding, identifier not recovered | local OA PDF | D2 | `gtn_unet_2023.pdf`; two-page report, weak standalone evidence |
| 2026 | Improving P300 Morphology through Single-Trial Latency Realignment | [10.1088/1741-2552/ae7766](https://doi.org/10.1088/1741-2552/ae7766) | OA publisher PDF | D1 | `Improving_P300_Morphology_Single_Trial_Latency_Realignment_JNE_2026.pdf` |
| 2021 | Learning Invariant Patterns for Subject-Independent P300 BCI | [10.1109/TNSRE.2021.3083548](https://doi.org/10.1109/TNSRE.2021.3083548) | OA publisher PDF | D1 | `Invariant_Patterns_Subject_Independent_P300_TNSRE_2021.pdf` |
| 2020 | Evaluation of CNNs Using a Large Multi-Subject P300 Dataset | [10.1016/j.bspc.2019.101837](https://doi.org/10.1016/j.bspc.2019.101837), [arXiv:2001.04225](https://arxiv.org/abs/2001.04225) | preprint | D1 | `Large_Multisubject_P300_CNN_arXiv_2001.04225.pdf` |
| 2021 | Lightweight Multi-Scale CNN for P300 Decoding (MS-EEGNet) | [10.3389/fnhum.2021.655840](https://doi.org/10.3389/fnhum.2021.655840) | CC/open publisher PDF | D1 | `MS_EEGNet_P300_Frontiers_2021.pdf`; mixed advantage across datasets |
| 2024 | Multi-Task Collaborative Supervised/Self-Supervised Network for RSVP | [10.1109/TNSRE.2024.3357863](https://doi.org/10.1109/TNSRE.2024.3357863) | OA publisher PDF | D1 | `MultiTask_SSL_RSVP_TNSRE_2024.pdf`; RSVP, not GTN candidate protocol |
| 2024 | Multi-Trial P300 Character Recognition | [arXiv:2410.08561](https://arxiv.org/abs/2410.08561) | preprint | D2 | `MultiTrial_P300_Character_arXiv_2410.08561.pdf`; repetition-dependent character metric |
| 2026 | Probit-link Split-and-merge Gaussian Process for ERP-BCI | [arXiv:2605.30775](https://arxiv.org/abs/2605.30775) | preprint | D2 | `P_SMGP_Bayesian_ERP_arXiv_2605.30775.pdf` |
| 2025 | Does Re-referencing Matter? Large Laplacian for Single-Trial P300 | [arXiv:2510.10733](https://arxiv.org/abs/2510.10733) | preprint | D2 | `P300_Rereferencing_arXiv_2510.10733.pdf`; 62-channel result cannot be assumed for GTN 3/8-channel montage |
| 2025 | Spatial Insight: Data-Driven ROIs for Single-Trial P300 | [arXiv:2511.02735](https://arxiv.org/abs/2511.02735) | preprint | D2 | `P300_Spatial_ROI_arXiv_2511.02735.pdf`; high-density/source-space setup |
| 2022 | Advances in P300 BCI Spellers | [10.3389/fnhum.2022.1077717](https://doi.org/10.3389/fnhum.2022.1077717) | CC review | D1-review | `P300_Speller_Advances_Review_2022.pdf` |
| 2025 | Riemannian Means Field Classifier for EEG-BCI | [arXiv:2504.17352](https://arxiv.org/abs/2504.17352) | preprint | D2 | `Riemannian_Means_Field_BCI_arXiv_2504.17352.pdf`; multi-paradigm evidence |
| 2021 | Minimizing Subject-Dependent Calibration with Riemannian Transfer Learning | [arXiv:2111.12071](https://arxiv.org/abs/2111.12071) | preprint | D2 | `Riemannian_Transfer_Minimal_Calibration_arXiv_2111.12071.pdf`; MOABB P300/MI/SSVEP meta-analysis |
| 2022 | Classifying Numbers from EEG Data - Which Architecture Performs Best? | [10.3233/SHTI220333](https://doi.org/10.3233/SHTI220333) | OA proceedings | D2 | `SHTI-292-SHTI220333.pdf`; no tested network significantly beat CNN baseline |
| 2026 | Sparse Bayesian Modeling of EEG Channel Interactions for P300 | [arXiv:2602.17772](https://arxiv.org/abs/2602.17772) | preprint | D2 | `Sparse_Bayesian_Channel_Interactions_P300_arXiv_2602.17772.pdf`; character-level/repetition outcomes |
| 2025 | SpellerSSL: Self-Supervised Learning with P300 Aggregation | [arXiv:2509.19401](https://arxiv.org/abs/2509.19401) | preprint | D2 | `SpellerSSL_arXiv_2509.19401.pdf`; direct mechanism, not matched GTN evidence |
| 2023 | ST-CapsNet for P300 Detection | [10.1109/TNSRE.2023.3237319](https://doi.org/10.1109/TNSRE.2023.3237319) | OA publisher PDF | D1 | `ST_CapsNet_P300_TNSRE_2023.pdf` |
| 2020 | P300 Transfer with xDAWN and Riemannian Geometry | [10.3390/app10051804](https://doi.org/10.3390/app10051804) | CC publisher PDF | D1 | `xDAWN_Riemannian_Transfer_P300_2020.pdf` |
| 2020 | CNN with Large Data Achieves True Zero-Training in Online P300 BCI | [10.1109/ACCESS.2020.2988057](https://doi.org/10.1109/ACCESS.2020.2988057) | OA publisher PDF | D1 | `Zero_Training_Online_P300_CNN_ACCESS_2020.pdf` |

## Local transferable, protocol, and negative-control sources

These are useful for hypotheses, implementation controls, or failure analysis, but must not be cited as direct GTN/P300 performance proof.

| Year | Source | Identifier / canonical URL | Level | Local file / boundary |
|---:|---|---|---|---|
| 2026 | Continuous-Latent Predictive Modeling for EEG-Language Foundation Models | [arXiv:2608.11656](https://arxiv.org/abs/2608.11656) | T | `BLPM_Continuous_Latent_Predictive_Modeling_arXiv_2608.11656.pdf`; foundation/latent modeling, not P300 benchmark |
| 2024 | Channel Reflection: Knowledge-Driven EEG Augmentation | [arXiv:2412.03224](https://arxiv.org/abs/2412.03224) | T | `Channel_Reflection_EEG_Augmentation_arXiv_2412.03224.pdf`; montage assumptions require testing |
| 2024 | Cross-Scale Transformer and Domain-Rectified Transfer for RSVP | [10.1109/TNSRE.2024.3359191](https://doi.org/10.1109/TNSRE.2024.3359191) | T | `CrossScale_Transformer_Domain_Rectified_RSVP_TNSRE_2024.pdf`; RSVP evidence only |
| 2020 | Cross-Dataset Variability in EEG Decoding | [10.3389/fnhum.2020.00103](https://doi.org/10.3389/fnhum.2020.00103) | T | `EEG_Cross_Dataset_Variability_2020.pdf` |
| 2024 | Data Leakage in Deep Learning Studies of Translational EEG | [10.1101/2024.01.16.24301366](https://doi.org/10.1101/2024.01.16.24301366) | T/protocol | `EEG_Deep_Learning_Data_Leakage_2024.pdf` |
| 2025 | EEG Foundation Models for BCI Learn Diverse Features | [arXiv:2506.01867](https://arxiv.org/abs/2506.01867) | T | `EEG_FM_BCI_Diverse_Features_arXiv_2506.01867.pdf` |
| 2026 | EEG-FM-Compass | [arXiv:2601.17883](https://arxiv.org/abs/2601.17883) | T/benchmark | `EEG_FM_Compass_arXiv_2601.17883.pdf` |
| 2026 | Multi-dimensional Generalization Framework for EEG Foundation Models | [arXiv:2605.28563](https://arxiv.org/abs/2605.28563) | T/negative | `EEG_FM_Generalization_Framework_arXiv_2605.28563.pdf`; reports limited short-window/channel-constrained gains |
| 2026 | Negative-Control Protocol for EEG Foundation-Model Benchmarks | [arXiv:2607.24519](https://arxiv.org/abs/2607.24519) | T/protocol | `EEG_FM_Negative_Control_Protocol_arXiv_2607.24519.pdf` |
| 2026 | EEG-PRISM Interpretability | [arXiv:2608.13676](https://arxiv.org/abs/2608.13676) | T | `EEG_PRISM_arXiv_2608.13676.pdf` |
| 2026 | NeuroDoc Task-Specification Layer for EEG Benchmarks | [arXiv:2606.22925](https://arxiv.org/abs/2606.22925) | T/protocol | `EEG_Task_Specification_NeuroDoc_arXiv_2606.22925.pdf` |
| 2025 | Estimating ERP from Few EEG Trials | [arXiv:2511.23162](https://arxiv.org/abs/2511.23162) | T | `Estimating_Event_Related_Potential_from_Few_EEG_Trials_arXiv_2511.23162.pdf`; ERP estimation, not classifier proof |
| 2026 | Identity Trap in EEG Foundation Models | [arXiv:2606.06647](https://arxiv.org/abs/2606.06647) | T/negative | `Identity_Trap_in_EEG_Foundation_Models_arXiv_2606.06647.pdf` |
| 2024 | MOABB Reproducibility Benchmark | [arXiv:2404.15319](https://arxiv.org/abs/2404.15319) | T/benchmark | `MOABB_Reproducibility_Benchmark_arXiv_2404.15319.pdf`; 15 P300 datasets, Riemannian pipelines often outperform deep models |
| 2026 | NeuralBench | [arXiv:2605.08495](https://arxiv.org/abs/2605.08495) | T/benchmark | `NeuralBench_arXiv_2605.08495.pdf` |
| 2026 | OmniEEG-Bench | [arXiv:2606.00815](https://arxiv.org/abs/2606.00815) | T/benchmark | `OmniEEG_Bench_arXiv_2606.00815.pdf` |
| 2019 | Riemannian Artifact Subspace Reconstruction | [10.3389/fnhum.2019.00141](https://doi.org/10.3389/fnhum.2019.00141) | T | `Riemannian_Artifact_Subspace_Reconstruction_2019.pdf`; artifact handling is indirect and may erase ERP signal |
| 2026 | Spectral Audit of Task-Dependent Aperiodic Reliance | [arXiv:2606.08583](https://arxiv.org/abs/2606.08583) | T/negative | `Spectral_Audit_Task_Dependent_Aperiodic_Reliance_arXiv_2606.08583.pdf` |
| 2026 | Understanding and Correcting Low-Frequency Bias in EEG Foundation Models | [arXiv:2608.01898](https://arxiv.org/abs/2608.01898) | T/negative | `Understanding_and_Correcting_Low-Frequency_Bias_in_EEG_Foundation_Models_arXiv_2608.01898.pdf` |
| 2016 | Understanding the Effective Receptive Field in Deep CNNs | [arXiv:1701.04128](https://arxiv.org/abs/1701.04128) | T/method | `Effective_Receptive_Field_Luo_NeurIPS_2016.pdf`; image-domain ERF theory, not EEG evidence |
| 2017 | Temporal Convolutional Networks for Action Segmentation | [10.1109/CVPR.2017.113](https://doi.org/10.1109/CVPR.2017.113) | T/method | `Temporal_Convolutional_Networks_Lea_CVPR_2017.pdf`; temporal hierarchy precedent only |

## Metadata-only and failed full-text retrievals

No local PDF is claimed for these rows. OA labels come from OpenAlex/Semantic Scholar and are not equivalent to successful full-text retrieval.

| Year | Source | DOI / URL | Access and download status | Level |
|---:|---|---|---|---|
| 2009 | Original xDAWN algorithm | [10.1109/TBME.2009.2012869](https://doi.org/10.1109/TBME.2009.2012869) | metadata-only; OpenAlex marked closed | M |
| 2021 | P3Net systematic model selection | [10.1109/TSMC.2021.3051136](https://doi.org/10.1109/TSMC.2021.3051136) | paper metadata-only/closed; [code repository](https://github.com/berdakh/P3Net) is open | M |
| 2024 | Signal Alignment for Cross-Dataset P300 | [10.1088/1741-2552/ad430d](https://doi.org/10.1088/1741-2552/ad430d) | hybrid OA claimed; IOP returned bot-captcha HTML, Europe PMC reports no OA full text | M |
| 2023 | Bayesian Uncertainty Modeling for P300 BCI | [10.1109/TNSRE.2023.3286688](https://doi.org/10.1109/TNSRE.2023.3286688) | gold OA claimed; IEEE direct PDF returned HTTP 403 | M |
| 2023 | Joint Alignment of Feature Vectors for P300 Transfer | [10.1109/JBHI.2023.3299837](https://doi.org/10.1109/JBHI.2023.3299837) | green OA claimed; HAL returned anti-bot HTML | M |
| 2022 | ERP-WGAN augmentation | [10.1016/j.jneumeth.2022.109621](https://doi.org/10.1016/j.jneumeth.2022.109621) | hybrid OA claimed; ScienceDirect returned HTTP 403 | M |
| 2019 | Active Inference for adaptive P300 BCI | [10.1088/1741-2552/ab5d5c](https://doi.org/10.1088/1741-2552/ab5d5c) | bronze OA claimed; IOP returned bot-captcha HTML | M |
| 2022 | Cross-State/Cross-Subject Visual ERP with Adversarial Training | [10.1109/TNSRE.2022.3150007](https://doi.org/10.1109/TNSRE.2022.3150007) | gold OA claimed; IEEE returned HTTP 418 | M |
| 2026 | ATCRN for P300 Speller | [10.1016/j.jneumeth.2026.110727](https://doi.org/10.1016/j.jneumeth.2026.110727) | metadata-only; OpenAlex marked closed | M |
| 2026 | ST-GraphTRNet single-trial P300 | [10.1088/1741-2552/ae3d68](https://doi.org/10.1088/1741-2552/ae3d68) | metadata-only; OpenAlex marked closed | M |
| 2026 | Data Aggregation Strategies for a P300 Speller | [10.64898/2026.06.17.732982](https://doi.org/10.64898/2026.06.17.732982) | bioRxiv PDF request returned HTTP 403 | M |
| 2022 | EEG Conformer | [10.1109/TNSRE.2022.3230250](https://doi.org/10.1109/TNSRE.2022.3230250) | OA metadata; not downloaded because it is mechanism-only, not direct P300 evidence | T/M |
| 2022 | ATCNet | [10.1109/TII.2022.3197419](https://doi.org/10.1109/TII.2022.3197419) | closed metadata; motor-imagery mechanism only, not direct P300 evidence | T/M |
| 2019 | Computing Receptive Fields of CNNs | [10.23915/distill.00021](https://doi.org/10.23915/distill.00021) | official OA HTML; no publisher PDF | T/M |
| 2016 | A Guide to Convolution Arithmetic for Deep Learning | [arXiv:1603.07285](https://arxiv.org/abs/1603.07285) | official TeX source verified; arXiv PDF transport failed | T/M |

## High-confidence synthesis and contradictions

1. **Cross-subject transfer is the central evidence-rich direction**, but published metrics are protocol-fragmented. Invariant-pattern CNN, zero-training CNN, calibration-free CNN, xDAWN/Riemannian transfer, Bayesian signal matching, signal alignment, MMD, SSL, and adversarial alignment must be compared on the same outer subjects, calibration budget, channels, epoch contract, and trial metric.
2. **Latency handling is plausible but not settled.** The 2026 latency-realignment paper directly supports modeling trial latency, while MS-EEGNet supports multi-scale receptive fields. However, MS-EEGNet is not uniformly better: prior extracted LOSO results show it trails EEGNet/BranchedNet on some datasets and wins on another. This is negative evidence against making multi-scale depth a default without matched ablation.
3. **Riemannian/xDAWN baselines remain mandatory.** The MOABB benchmark and direct transfer papers make it scientifically weak to evaluate only neural architectures. A fold-local xDAWN plus tangent-space/Riemannian baseline should be a primary comparator, not a secondary appendix.
4. **Graph/attention novelty is under-verified for low-channel GTN.** Sparse Bayesian channel interactions, high-density ROI work, ST-CapsNet, ERP-XTTN, and ST-GraphTRNet motivate relational spatial modeling, but 62-channel/source-space gains do not imply gains with 3 or 8 electrodes. A graph whose edges are learned from three channels may add parameters without identifiable structure.
5. **Foundation/SSM evidence is indirect and includes negative controls.** Current foundation benchmarks report limited gains on short-window, channel-constrained BCI tasks and expose identity, low-frequency, and dataset-leakage traps. Foundation pretraining should be tested against random-weight/frozen-feature and source-identity controls before it is called an improvement.
6. **Uncertainty and sequential aggregation need separate endpoints.** Trial ROC-AUC/BACC, calibration (NLL/Brier/ECE), character/candidate hit rate, stopping time, and ITR are different outcomes. A model can improve character accuracy through repetitions or a language model while leaving single-trial discrimination unchanged.
7. **Direct artifact-robustness evidence for P300 is thin.** Riemannian ASR and channel-reflection augmentation are transferable methods, not proof that a particular rejection/correction policy preserves P300. Artifact methods require constructed counterexamples and clean-vs-corrupted paired tests because aggressive cleaning can delete the ERP itself.
8. **Theoretical RF and trained ERF are different objects.** Arithmetic sources justify exact
   support, jump, center, padding, and dilation calculations; Luo et al. show that high-impact ERF
   can occupy only part of that support. Neither line of work establishes an optimal P300 RF.

## Important interpretation limits

- The 53 PDFs are not 53 mutually comparable GTN experiments. They include reviews, preprints, benchmark/protocol papers, broad EEG foundation work, and indirect methods.
- No external AUC, BACC, F1, AP, character accuracy, hit rate, or ITR value should enter a leaderboard unless dataset, subject split, calibration, repetitions, preprocessing, channels, and metric definition match.
- Preprints dated 2025-2026 are hypothesis sources until peer review and independent reproduction. Metadata-only rows were not treated as full-text evidence.
- Search packets and exact API provenance are preserved under `tmp/literature_search_20260828/`; the stable local integrity record is `Paper/pdf_validation_20260828.json`.
