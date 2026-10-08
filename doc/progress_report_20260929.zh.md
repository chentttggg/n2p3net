# N2P3-Net 项目进展报告（2026-08-21 至 2026-09-29）

**日期**：2026-09-29　**仓库状态**：HEAD `f1a303c`　**报告范围**：P300/oddball 解码研究主线，以及云端训练基础设施和 BrainSync 工作站支援

**证据范围**：本文依据本地仓库 `doc/evidence/` 的 dated 记录、`.workbuddy` 工作日志和 `D:/temp` 的云端协作暂存件整理。所有数值结果均绑定相应执行记录、输入摘要和模型/结果身份；文献仅采用本地 `Paper/` 中已核验的 PDF，标识符沿用 `Paper/literature_manifest_20260828.md` 的检索账本与 D1/D2 证据分级。HEAD 只作为版本锚点，不替代各实验记录中的实际执行快照。

---

## 摘要

本报告总结近五周的研究、工程和证据建设进展。研究方法方面，经机制审查后，项目采用“共享表征—域归一化—源残差/先验—目标适配—物理归属决策”的纵向分层架构；在十二库统一框架中，源初始化相对 CE-only scratch 的平均 AUC 提高 0.0414，而包含源增强与 prior/decision 适配的完整联合包相对普通 full-fine 的差异为 −0.0100（95% CI [−0.0220, 0.0014]，Holm 校正后 p=0.161）。因此，当前证据支持迁移初始化作为研究基线，不支持把完整联合包晋升为默认路径。

评价与工程方面，项目完成了源/目标身份、固定测试行、缓存摘要、模型重放和资源执行记录的联动核验；48/48 个保存模型重放通过。9 月 22 日的只读审计确认了六项口径或复用缺陷，并推翻了一项由默认值推导出的高严重度误报。输入合同方面，对 BI2014a 61 名准入被试的 57,446 次 flash 逐事件复核显示，1200 ms 响应窗内后续 onset 的 participant-macro mean 为 2.221972。该结果证明时序重叠存在，但不能被解释为噪声比例或性能损失；600/800/1200 ms 的原始窗口配对实验仍是下一阶段首要任务。

项目当前的主要优势是：数据、模型、训练、评估和工作站均由显式合同连接；复杂机制能够拆分为可复用、可反例验证的模块；公开数据开发、软件正确性、成人设备确认和产品性能保持分账。上述优势提高了研究结果的可追溯性和后续迭代效率，但不替代多 seed、实机采集或成人九选一性能证据。

## 1. 背景与目标

项目目标是在 oddball P300 范式下建立跨被试、可复核的解码链，最终服务于成人 BrainSync 多 session 场景中的九选一识别。产品读数按研究规范定义为完整 45-flash block 的 `hit@all`；公开数据开发与成人设备确认严格分账。本报告中的数值全部来自公开数据集（BI2014a、GTN 及十二库框架），不构成成人有效性结论，也不等同于 BrainSync 产品验收。

报告期起点（8 月下旬）是：MS-EEGNet 风格紧凑 CNN [2] 在 GTN 上尚未超过 EEGNet 基线 [1]，迁移路线仍在架构选择阶段。报告期终点是：统一框架已给出迁移初始化与组合机制的分离证据，输入合同的主要疑点完成原始事件复核，训练和工作站链条形成了可追踪的工程闭环。

### 1.1 项目优势与技术架构

项目采用从原始信号到产品边界的分层架构，具体实现与接口职责见[当前技术结构](technical_structure.zh.md)。各层的职责、当前实现和直接优势如下：

| 层级 | 代表模块/接口 | 核心职责 | 形成的优势 |
|---|---|---|---|
| 数据与输入合同 | `data.contract`、`PreprocessingSpec`、`source_records`、`SourceCatalog` | 统一采样率、通道/参考、时间支持、raw 范围、candidate membership、source/evaluation/deployment 角色 | 将信号语义集中到单一合同，降低口径漂移和数据泄漏风险 |
| 共享表征与适配 | `N2P3Net`、observed-channel pack/scatter、`Split-BN`、`FeatureResidualAdapter`、`SourceParameterPrior` | 对实测通道进行显式归一化，复用共享 trunk，并把域差异、源残差和先验适配作为独立因素 | 参数共享与机制隔离并存，便于做 matched ablation 和失败归因 |
| 物理决策层 | `MembershipDecision`、CE/decision recipe | 保留真实 event×candidate 归属、完整词表和缺失证据，区分 binary trial 与物理 decision | 不把候选数量相等误写成范式同构，避免将公共 AUC 直接冒充产品命中率 |
| 实验执行与资源 | `unified_target_recipes`、`UnifiedTrainingBudget`、`run_domain_transfer_matrix.py`、资源配置文件 | 由统一 recipe 派生 scratch、迁移、LDA 和 decision 对照；把 CPU/GPU 预算传入实际 worker | 训练请求、拟合范围、资源消耗和结果身份可重放，减少重复拟合和非法复用 |
| 证据与验收 | cache sidecar、source snapshot、SHA-256、manifest、模型 replay | 绑定输入、源码、运行环境、模型字节和输出记录 | 能区分“代码按声明执行”“公开数据开发结果”和“产品性能确认” |
| 工作站与发布 | `transport.py`、`DeviceAdapter`、`RecordingSession`、`SessionStore`、`BrainSync.exe` | 统一串口/BLE 生命周期、250 Hz 采集、EDF/BIDS 收尾、确认事务、后端结果准入和发布运行时 | 采集、保存、推理和退出由明确 owner 管理，降低操作竞态和历史数据误用 |

这套架构的工程性优势不能替代科研证据。当前已验证的是合同、身份、重放、模拟采集和发布启动行为；真实设备吞吐、光学时间同步和成人九选一效能仍需独立验收。

## 2. 文献综述

### 2.1 基准与基线格局

MOABB 复现基准为公共数据对比提供了统一口径 [5]。其中 BI2014a within-session 排序为 EEGITNet 86.66、EEGNeX 85.08、EEGNet-8,2 82.09（AUC×100），经典 xDAWN+LDA 为 66.60。该结果带来两项直接约束：deep 架构之间的 within-session 差距约为 4 个点，通常小于 deep 与经典方法之间的差距；线性 LDA 在跨被试条件下也可能保持竞争力。因此，本项目将波形 LDA 作为并行强基线，而非附录式对照。

### 2.2 紧凑卷积架构

轻量化卷积结构在 P300 解码中已有稳定的文献依据。Cecotti 的少滤波器 CNN [1] 表明，数十个滤波器即可支撑 trial 分类；MS-EEGNet [2] 说明多尺度卷积可能在小参数量下获得收益，但跨数据集稳定性有限；GTN 大规模 CNN 评估 [3] 与独立复核 [4] 均未发现某一网络稳定超过 CNN 基线。这些结果提示，容量本身未必是主要瓶颈。本项目 trunk 采用 MS-EEGNet 风格的多尺度结构；报告期内 `full_unfold` readout（1,506 参数）在 64-fold LOSO 中取得 0.7451 AUC，未显示出明显的容量受限迹象。

### 2.3 跨被试与跨数据集迁移

现有迁移证据可概括为三类。第一类是 Riemannian 路线：xDAWN 协方差与 Riemannian 对齐 [7] 以及少校准迁移的 meta 分析 [8] 表明，源域协方差对齐可能以较少目标数据换取增益。第二类是零校准路线：免校准 P300 系统 [9] 和大数据零训练在线 CNN [10] 说明，在源数据充分时可以减少目标校准。第三类是表征学习路线：invariant 模式学习 [11]、对比学习 CNN [12]、自适应 split-MMD [13]、自监督聚合 [14] 和贝叶斯信号匹配 [15] 均报告过跨被试收益，但其中多项为 D2 级预印本，数据集和评价口径并不一致。它们构成 A1 源增强包的设计动机，不能预先替代本项目的配对验证。

### 2.4 repetition 聚合与决策

动态停止和集成分类器是 subject-independent P300 的成熟组件 [6]；multi-trial 字符识别研究 [16] 系统讨论了 repetition 数与命中率之间的权衡。本项目把候选级聚合、重复数和完整 decision 作为显式变量，区分 score-level 聚合、feature-level 聚合和物理 membership 决策。

### 2.5 评价方法学与泄漏

跨数据集变异性 [17] 和脑电深度学习泄漏研究 [18] 指出，预处理/切分造成的隐性泄漏，以及利用评价集选择超参数或配方，是高频风险。报告期内的输入合同审查、source-only 选择和固定 test 行，均围绕这两类风险设计；其中代码口径仍存在的审计问题见第 7 节。

## 3. 数据与输入合同

- **公开数据**：BI2014a（61 人准入，平均 2.93 repetition/selection [20]）、GTN（242 被试）及十二库框架。8 月下旬通过直连 CERN 的下载脚本补齐 BI2012/2013a/2014b/2015a/2015b 原始 ZIP；下载日志留存于 `D:/temp`。
- **缓存与溯源**：训练 cache 配套 `<cache>.record.json` sidecar，并进行 SHA-256 双端核验；源码以 frozen archive、manifest 和 commit/执行快照共同绑定。该机制与[统一结果记录](evidence/unified12_results_20260918.zh.md)中的模型 replay 和归档校验相互对应。
- **窗口复核**：默认合同为 `[-200, 1200) ms`（`sfreq=128`，`n_times=179`）。对 BI2014a 完整原始 ZIP 逐事件计数后，61 名被试共 57,446 次 flash；1200 ms 截止时后续 onset 的 participant-macro mean 为 2.221972。该统计只说明响应窗与邻近刺激存在时序重叠，不能推出噪声比例、不可识别性或性能损失。250–600 ms 仅是可选 LMBC 证据段，默认 `full_unfold` 使用完整卷积时间支持。历史上 2 Hz/800 ms 与 0.1 Hz/1200 ms 的比较同时改变了滤波和窗口，不能用于单独判定窗口长度。详见[输入合同审查](evidence/input_contract_review_20260922.zh.md)。

## 4. 阶段一：契约修订与架构落地（8 月下旬–9 月上旬）

### 4.1 GTN 契约

在 0.1 Hz/1200 ms LOSO、245 折的既定协议下，K=33 配置的 `hit@all=0.8041` 是报告期内 GTN 结果中的最高值。该数字属于公开数据开发结果，不是成人 BrainSync 产品读数。

### 4.2 BI2014a matched 消融

在 64-fold LOSO 对照中，`full_unfold` 的 AUC/BACC 为 0.7451/0.6774，`ms_flatten` 为 0.7342/0.6678，AUC 高于 EEGNet。

### 4.3 纵向分层架构与软件检验

经机制审查后，项目未采用横向嵌套结构，转而实现三项可独立开关的机制：P0 `Split-BN`（共享 affine、按域维护 buffer，并采用 fail-closed 资格门）；P1 drop-to-consensus 源残差（`p∈[0.2,0.5]`，`r_max=0.5` 时整 run invalid）；P2 direct-membership 决策损失（`λ_trial=0` 时明确拒绝）。该阶段完成 SHA-256 逐文件核对、Ruff 检查和空闲环境下全量 pytest 772 passed 的三次复现。后续发现的 flaky 四模式经受控实验归因于环境，验收规则改为云端或独立 shell 复跑。


## 5. 阶段三：十二库统一框架（9 月 18 日）

### 6.1 设置与执行完整性

源包为 A0（CE）/ A1（CE + 0.05 SupCon + 0.05 条件 MMD）；个人适配为 Q0（普通 full-fine）/ Q1（trial CE + 源中心收缩，并在有真实 membership 的目标上增加 decision CE）。另设 CE-only scratch、decision scratch 和波形 LDA（Block-Toeplitz 型 [19] 的方法族）。单配对 seed 为 `20260918`；评价包含四个整库留出目标和一个 BI 家族联合排除敏感性折；校准预算为 K=0/200/800。

本轮完成 48 个 source job、2,062 次神经拟合、246 次波形 LDA、450 个 zero-shot 单元；保存模型 replay 48/48 通过，重放耗时约 78 秒；归档包含 19,779 个成员、约 11.18 GiB，SHA-256 已入账。上述是执行完整性证据，不等同于多 seed 稳定性或成人产品性能证据。详细结果见[十二库统一结果](evidence/unified12_results_20260918.zh.md)。

需要特别说明：binary-only 目标上的 Q1 是 prior-only 适配；只有 BNCI2014-008 具有真实 membership，才增加 decision CE。因此 A0Q1/A1Q1 的四库总均值不能被解释成一个定义完全一致的单一 decision-loss 效应。后续分析应将 binary-only、BNCI decision 和 BI 敏感性分别报告。

### 6.2 主结果（K=200）

| 处理 | 平均 AUC |
|---|---:|
| A0Q0：CE 源 + 普通 full-fine | **0.6995** |
| A1Q0：增强源 + 普通适配 | 0.7003 |
| A0Q1：CE 源 + prior/decision 适配 | 0.6971 |
| A1Q1：完整联合包 | 0.6895 |
| 波形 LDA | 0.6925 |
| CE-only scratch | 0.6581 |
| decision scratch | 0.6551 |

表中 scratch 标签按 9 月 29 日对原始 `results/bnci_hit_all.json` 和分析器映射的勘误复核。四库均值用于描述冻结矩阵，不掩盖不同目标的监督定义差异。

### 6.3 结果解释

1. **迁移初始化显示出明确的开发收益**：A0Q0 相对 CE-only scratch 的平均 AUC 提高 0.0414。该对照更直接地反映 source 初始化/共享表征相对从头训练的差异。矩阵中 A1Q1 相对 decision scratch 的 +0.0344 仅作为包含不同监督定义的次要对照，不能当作四库共同 estimand。
2. **完整联合包没有优于普通 full-fine 的证据**：A1Q1−A0Q0 为 −0.0100（95% CI [−0.0220, 0.0014]，Holm 校正后 p=0.1611）；K=800 时为 −0.0159（CI [−0.0223, −0.0095]）。A1Q0−A0Q0 为 +0.0008（区间跨 0），说明源端增强的独立贡献接近于零。由于 Q1 在 BNCI 含额外 decision supervision，当前 A×Q interaction（−0.0084）应视为配方层面的探索性现象，不能直接归因给某一个机制。
3. **decision loss 暂不进入默认路径**：BNCI 是本轮唯一具有真实 membership 的目标。K=800 的 `hit@all` 为 A0Q0=0.716、A1Q1=0.509；Q1 在该目标上明显下降。因此 P2 保留为研究对照，不晋升为默认适配策略。
4. **LDA 仍具竞争力**：A1Q1−LDA 为 −0.0031（95% CI [−0.0153, 0.0085]），当前证据不足以宣称神经联合包超过 LDA。默认研究基线定为 A0Q0/full-fine，并保留波形 LDA 作为低复杂度参照。

BI 敏感性折中，A1Q1−A0Q0 为 −0.0166（Holm 校正后 p=0.00016），负效应主要由 BI2015a 驱动（−0.0327）。该结果加强了“迁移相对 scratch 有效、完整联合机制未显示附加收益”的判断，但仍属于冻结单 seed 的公开开发证据。

## 7. 阶段四：审计与工程（9 月 15–29 日）

### 7.1 全链条审计（9 月 22 日）

9 月 22 日完成四路独立只读审查，并对高严重度结论逐条回读源码。审查确认六项问题：`hit@R` 对 incomplete decision 的分母口径不一致；共同 test 行随批内最大 K 漂移；跨运行复用键缺少 `evidence_scope`；验证损失错误使用不可加的 `forward()` 归约；`paired_permutation_test` 在活动树中无生产调用；EOG fit 范围守卫为可选参数、存在 API 层 fail-open。审查同时推翻了“缺导零填充进入 temporal BN 并被 affine 复活”的高严重度结论，因为主链显式启用 `observed_channel_norm=True` 并在 checkpoint 载入时强制核验。该审查本身为只读工作，未把线索直接写成训练结果；完整边界见[全链条审查](evidence/full_chain_audit_20260922.zh.md)。

### 7.2 输入合同窗口审查（9 月 22 日）

原始事件复核确认 1200 ms 窗内存在后续刺激，但没有支持“平均 4 个后续 flash”或“71% 输入为噪声”的证据。窗口重叠不能单独判定可利用 ERP 成分、噪声比例或性能上限。roadmap 已将 600/800/1200 ms 的窗口族实验置于 EA/RPA/GALIA 等迁移路线之前；真实短窗比较必须从 raw 范围重建，不能把长窗 FFT cache 的前缀截取当作等价短窗。详见[输入合同审查](evidence/input_contract_review_20260922.zh.md)。

### 7.3 工程与支援

算法链八项合同修复完成，全套定向验收记录为 1,936 项通过。BrainSync 前端完成串口/BLE 统一、固定 250 Hz、采集收尾和结果准入改造；发布副本采用 PyInstaller onedir，`BrainSync.exe --mock` 在隔离环境下持续运行 8 秒，发布载荷与开发目录边界已核验。云端训练执行器记录 CPU 总预算 25、GPU worker 容量 4，并将预算传入 source/target worker；四路大源预加载验证成功，未发生传输回退或 OOM 重试。上述结果属于软件、资源和模拟链条验收，不构成真实设备吞吐或成人识别效能结论。相关记录见[统一训练性能](evidence/unified12_performance_20260918.zh.md)、[BrainSync 传输统一](evidence/domain_transfer_major_20260920.zh.md)和[发布版验收](evidence/brainsync_release_20260920.zh.md)。

## 8. 讨论、结论与下一步

报告期的知识增量可以归纳为四点：

1. **架构层面**：项目从若干候选迁移机制收敛为显式分层、可拆解的共享 trunk、域归一化、源适配、目标校准和物理决策链。数据合同、recipe、资源计划和证据 manifest 贯通了研究代码与工作站接口，形成了可追踪的闭环。
2. **科研层面**：在冻结的单 seed、固定 study 集合和固定预算下，source 初始化相对 CE-only scratch 有明显开发收益；A1 源增强和 Q1 prior/decision 适配没有显示稳定附加收益，部分条件出现负迁移。该结论支持收缩默认研究基线，不支持把复杂联合包作为产品默认。
3. **证据层面**：窗口疑点已从推测转为可复算的逐事件统计，且历史 2 Hz/800 ms 对 0.1 Hz/1200 ms 的比较被识别为混淆设计。输入合同不再把重叠计数写成噪声比例，也不把 3×3 粗归并当成九选一物理 membership。
4. **工程层面**：训练、缓存、模型重放、资源排程、采集生命周期和发布运行时均有独立的验收记录；这为后续实验提供了可恢复、可复核的基础，但不等同于成人设备或产品效果已通过验收。

下一阶段按以下顺序推进：

1. 从 raw 范围重建并预注册 600/800/1200 ms 窗口比较，先解决输入支持的不确定性。
2. 将十二库结果拆成 binary-only、BNCI decision 和 BI 敏感性三个 estimand，并用 3–5 个配对训练 seed 评估方法稳定性。
3. 统一 `hit@R` 分母、跨运行 `evidence_scope`、验证损失归约和 EOG fit 范围等审计项，再复核受影响的历史比较。
4. 在独立成人设备实验中验证 250 Hz 采集、光学时间、会话确认和九选一 `hit@all`，不把公开数据 AUC 或软件模拟结果晋升为产品成绩。

本报告的主要局限是：所有结果均为公开数据开发证据；统一框架只有一组配对 training seed；K=200 与 K=800 的完整 participant 集合不同；within-session 同口径上界尚未执行；文献综述限于本地已核验 PDF（53/68 全文可读），不构成系统综述。统计区间应理解为条件于冻结 study、模型和 source selection 的开发证据，不应外推为跨训练随机性或成人产品的总体置信区间。

## 参考文献

1. Cecotti H. A Few Filters Are Enough: CNN for P300 Detection. Neurocomputing, 2020. doi:10.1016/j.neucom.2020.10.104（arXiv:1909.06970）
2. Altaheri H, et al. Lightweight Multi-Scale CNN for P300 Decoding (MS-EEGNet). Frontiers in Human Neuroscience, 2021. doi:10.3389/fnhum.2021.655840
3. Evaluation of CNNs Using a Large Multi-Subject P300 Dataset. Biomedical Signal Processing and Control, 2020. doi:10.1016/j.bspc.2019.101837（arXiv:2001.04225）
4. Classifying Numbers from EEG Data — Which Architecture Performs Best? Studies in Health Technology and Informatics, 2022. doi:10.3233/SHTI220333
5. Barachant A, et al. The MOABB: toward reproducible and benchmarkable EEG/MEG pipelines. arXiv:2404.15319, 2024.
6. Wang Y, et al. Subject-Independent P300 BCI with Ensemble Classifier, Dynamic Stopping and Adaptive Learning. IEEE GLOBECOM, 2017. doi:10.1109/GLOCOM.2017.8255030
7. P300 Transfer with xDAWN Covariance and Riemannian Geometry. Applied Sciences, 2020. doi:10.3390/app10051804
8. Sosnik R, et al. Minimizing Subject-Dependent Calibration with Riemannian Transfer Learning. arXiv:2111.12071, 2021.
9. A Calibration-free Approach to Implementing P300 BCI. Cognitive Computation, 2022. doi:10.1007/s12559-021-09971-1
10. CNN with Large Data Achieves True Zero-Training in Online P300 BCI. IEEE ACCESS, 2020. doi:10.1109/ACCESS.2020.2988057
11. Learning Invariant Patterns for Subject-Independent P300 BCI. IEEE TNSRE, 2021. doi:10.1109/TNSRE.2021.3083548
12. Contrastive Learning Based CNN for ERP-BCI. arXiv:2407.04738, 2024.
13. Adaptive Split-MMD Training for Small-Sample Cross-Dataset P300 EEG. arXiv:2510.21969, 2025.
14. SpellerSSL: Self-Supervised Learning with P300 Aggregation. arXiv:2509.19401, 2025.
15. Bayesian Signal Matching for Transfer Learning in ERP-BCI. arXiv:2401.07111, 2024.
16. Multi-Trial P300 Character Recognition. arXiv:2410.08561, 2024.
17. Cross-Dataset Variability in EEG Decoding. Frontiers in Human Neuroscience, 2020. doi:10.3389/fnhum.2020.00103
18. Data Leakage in Deep Learning Studies of Translational EEG. medRxiv 2024.01.16.24301366. doi:10.1101/2024.01.16.24301366
19. Block-Toeplitz Covariance LDA for ERP-BCI. Journal of Neural Engineering, 2022. doi:10.1088/1741-2552/ac9c98（arXiv:2202.02001）
20. Brain Invaders 2014a 官方作者报告. HAL 02171575。（repetition 结构与刺激时序的一手描述；§3 的逐事件窗口统计来自本项目对原始 ZIP 的独立复核，非引自该文）
21. Advances in P300 BCI Spellers（综述）. Frontiers in Human Neuroscience, 2022. doi:10.3389/fnhum.2022.1077717
