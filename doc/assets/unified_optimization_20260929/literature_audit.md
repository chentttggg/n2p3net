# A1/Q1 优化的一手文献审计

日期：2026-09-29。用途：为优化方案提供机制依据与实验边界；不是新增训练结果。已加载 `research`、`experimental-design`、`eeg-bci-transfer-learning`，先读当前文档索引、研究规范、执行计划、十二库结果与批判评估、全链机制审查及 09-20/09-22 路线约束。技能中旧的 FiLM 排名和旧模型规模不作为本轮结论。

## 1. 对下一步最有影响的核查结论

1. **EA 有直接 EEG/ERP 一手先例，但原论文区分了读取全部目标 EEG 的离线实验与只读取已到达校准样本的模拟在线实验。** 本项目适合先验证 calibration-only、拟合后冻结的 EA state；不能引用原论文离线实验并把它称为不读取 test EEG 的协议。
2. **完整 P300 RPA 不是默认零标签路线。** 作者源码用目标校准标签拟合 ERP 模板，又用按类目标协方差均值求旋转。只做无标签 recenter 是可研究的子路线，须单独命名；若其 covariance 特征本身使用目标 ERP 模板，依然需要目标标签。
3. **GALIA 的论文实现和本地共享 Q/逐域对角缩放不同。** 原摘要已确认每域独立变换与 cyclic AJD；本地旧全文审计进一步记录 super-trial tangent vectors、类内 surrogate features 与每域旋转。Fast alignment 的“无需再训练分类器”不能解释成“无需目标数据/标签”。在标签需求核定前不能准入 zero-label 臂。
4. **GALIA 摘要的 `2.12 ± 1.88%` 是 classification accuracy 改善，不能换写为 AUC 改善。** 它相对 subject-specific train/test pipeline，且临床库例外；与本项目相对 plain source zero-shot 的 ΔAUC 不同。项目 `+0.01 AUC` 是预声明开发门槛，不是文献承诺收益。
5. SupCon、源中心正则、PCGrad/CAGrad 都可提供机制假设；现有一手文献没有证明它们在当前十二库、固定小模型和稀少完整 decision 的预算下有效。因此按当前路线先冻结输入合同，保留 CE + full-fine 锚点，再比较合法 geometry state；暂不重启辅助损失网格或默认梯度手术。

## 2. 主要原始来源与适用范围

### R1. He & Wu：Euclidean Alignment

He H, Wu D. *Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach*. IEEE TBME 67(2):399–410, 2020. [DOI](https://doi.org/10.1109/TBME.2019.2913914)；[作者 arXiv 全文](https://arxiv.org/pdf/1808.05464)。

**本轮核实层级：全文。** 已读取公式与离线/模拟在线实验段落。

- p.4 式 (10)–(12)：每域以 trial 二阶矩的算术均值为 reference，`Rbar = mean_i(X_i X_i^T)`，再做 `Xtilde_i = Rbar^(-1/2) X_i`。原式没有额外减均值；经变换后域均值二阶矩为单位阵。除以时间长度的约定必须 source/target 一致。源码增加 shrinkage 或 eigenvalue floor 后，只能期待正则化后的数值稳定，不能声称原未正则化二阶矩仍精确等于 I。
- p.6 离线段明确写：因为可访问全部 EEG，使用 all trials/resting epochs 估 reference，然后做留一被试测试。这是读取目标测试 EEG 的转导协议。
- p.10 模拟在线段明确只用新被试已经获得的小批校准 trials 来估 reference，并强调不能使用当时不可获得的未来 EEG 或标签。后续分类器使用其合法校准标签。
- p.11 明确讨论错误/异常 trial 会污染 reference，以及模拟在线不等于真人闭环。

**支持什么：** 不改 `(N,C,T)` 形状的完整白化 state 有方法学依据；拟合范围、到达时间、异常影响是方法的一部分。

**不支持什么：** 白化保证类条件对齐、保证全部参与者受益、或者把旧未对齐源 checkpoint 配上单独白化的 target 就称为复现 EA。原文 source/target 都独立对齐到共同 reference；源训练输入合同改变时须重新拟合并绑定模型。如果保留未对齐源 checkpoint，必须另行设计 target-to-source reference 映射，并把它标为项目变体。

### R2. Rodrigues, Jutten & Congedo：Riemannian Procrustes Analysis

*Riemannian Procrustes Analysis: Transfer Learning for Brain–Computer Interfaces*. IEEE TBME, 2019. [DOI](https://doi.org/10.1109/TBME.2018.2889705)；[作者代码](https://github.com/plcrodrigues/RPA)；[关键实现，固定 commit](https://github.com/plcrodrigues/RPA/blob/cfcddb3d31b482941a23353dfbe46dffb118d02d/rpa/helpers/transfer_learning/utils.py)。

**本轮核实层级：原论文摘要 + 作者原始代码。** Europe PMC PMID `30596565` 与 HAL 官方元数据均核实；HAL 全文链接本轮返回 bot challenge，未把旧摘要当全文。作者仓库 master 对应 commit `cfcddb3d31b482941a23353dfbe46dffb118d02d`（2021-03-12）。

- 原摘要：SPD covariance 上的 translation、scaling、rotation，测试覆盖 8 个公开 BCI 数据集/3 范式。几何变换作用于 SPD 特征，不等同于通道 z-score 或任意向量对角缩放。
- `get_sourcetarget_split_p300`：只在 target calibration 上 `ERPCovariances(classes=[2], estimator='lwf').fit(X_target[idx_train], y_target[idx_train])`，同一个 ERP estimator 作用于其 test。此步骤已使用目标标签。
- `transform_org2rct_p300`：reference `M_train` 只来自校准协方差，test 使用同一个 `M_test = M_train`；可选 `weight_samples` 按标签加权。
- `transform_rct2rot_p300`：源/目标校准分别按类计算 Riemannian mean，随后求旋转矩阵 R，固定作用于 calibration/test。这是有监督旋转。
- 作者示例按随机 trial 和每类样本数拆分，不复现本项目 chronological prefix / physical-decision 预算；应保留本项目时间边界，不能直接搬其示例评分结果。

**支持什么：** calibration-only 冻结 recenter/stretch/rotation state 的实现结构；同 K 的有标签 RPA 可作为完整 few-shot 路线。

**不支持什么：** 把完整 P300 RPA 标成 strict zero-shot/zero-label；把 SPD 路线不加说明塞入 raw EEG 的 `(N,C,T)` 变换接口。若重构成 raw 变换，必须给出映射等价性及反例测试，不能由“都是矩阵乘法”推断等价。

### R3. Altındış et al.：P300 joint feature alignment / GALIA

*Transfer Learning for P300 Brain-Computer Interfaces by Joint Alignment of Feature Vectors*. IEEE JBHI 27(10):4696–4706, 2023. [DOI](https://doi.org/10.1109/JBHI.2023.3299837)；[HAL](https://hal.science/hal-04255249)；[HAL 元数据 API](https://api.archives-ouvertes.fr/search/?q=halId_s:hal-04255249&fl=title_s,abstract_s,files_s&wt=json)。

**本轮核实层级：作者 HAL 摘要/元数据 + Europe PMC 原论文摘要；详细公式来自既有本地全文审计。** HAL PDF 当前返回 bot challenge；不宣称本轮重新读到了 PDF。

- 摘要明确：group learning 为 many-to-many，fast alignment 为 many-to-one；cyclic AJD 找到 **one transformation for each domain**。实验为 6 个 P300 库、333 sessions、177 subjects；报告的是 **classification accuracy** 平均改善 `2.12 ± 1.88%`，临床库例外。这里仅核实摘要级数值，不推断其误差条定义或与本项目 ΔAUC 等价。
- 既有 `doc/evidence/domain_transfer_chain_primary_source_audit_20260904.zh.md` 记录式 (1)–(10)：super-trial SPD covariance → 域内 Riemannian tangent vector `v_ml`；classwise surrogate `T_mk`；每域 prewhitening `W_m`；跨域交叉外积 `R_ijk`；每域 `U_m` 循环 AJD，`B_m=W_m U_m`；fast alignment 另求新域 `U_x`。
- 本轮读取当前 `src/data/domain_alignment.py` 的模块声明：仍描述 shared approximate joint diagonalizer 加每域 diagonal scale。这不足以实现上述论文序列。

**支持什么：** 对 P300 多域线性几何提出一个有直接先例的完整候选；先按论文构造数据对象与每域变换，再讨论性能。

**不支持什么：** 不重训分类器意味着完全零标签；共享 Q + diagonal-only whitening 复现 GALIA；或将其 accuracy 改善挪用为本项目 AUC 先验。classwise surrogate 与目标 fast-alignment 标签供应方式必须逐步追踪，核定前按未准入处理。

### R4. Zhao et al.：不变表示并不足以保证迁移

*On Learning Invariant Representations for Domain Adaptation*. ICML 2019, PMLR 97:7523–7532. [原文与 PDF](https://proceedings.mlr.press/v97/zhao19a.html)。

**本轮核实层级：全文，重点 §4 与 Theorem 4.1/4.3。** Theorem 4.1 的目标误差界同时包含源误差、边缘分布差异与 source/target labeling-function 差异；Theorem 4.3 在其条件下揭示 label marginals 不同与表征强制不变之间的误差下界冲突。

**支持什么：** 域区分度下降、MMD 下降、均值协方差接近不能单独作为标签判别保留的证据。应记录 class-conditional geometry、类比例与逐库风险。

**不支持什么：** 该定理直接证明本项目 conditional MMD 必然有害。本项目 A1 是带标签的源–源条件对齐；原定理主要讨论源–目标不变表示，其对象与假设不同。当前负交互仍需本项目数据和梯度证据，不能靠引文归因。

### R5. Khosla et al.：SupCon 的同类拉近目标

*Supervised Contrastive Learning*. NeurIPS 2020. [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)。

**本轮核实层级：作者摘要。** 原文明确同类 embedding 拉近、异类拉远；实证为图像分类，不能挪用其 ImageNet 改善到 EEG。

**支持什么：** 如果跨被试/范式的同 binary class 内存在对后续 decision 或适配有用的多模态结构，应检查“哪些变化被压缩”，而不只看 source CE/SupCon 数值。

**不支持什么：** 摘要没有证明有限权重 SupCon 必然把所有 EEG 同类样本压成一点，更没证明当前 A1 已发生这种塌缩。该表述只能作为待检验反例假设；不能把论文机制描述当当前病因证据。

### R6. Li, Grandvalet & Davoine：以预训练点为中心的正则

*Explicit Inductive Bias for Transfer Learning with Convolutional Networks*. ICML 2018. [arXiv:1802.01483](https://arxiv.org/abs/1802.01483)。

**本轮核实层级：作者摘要。** 作者比较使新解靠近源初始化的正则，并推荐以预训练参数为中心的 L2 penalty 作为 transfer baseline。

**支持什么：** 当前有限源中心约束有一般迁移学习先例，值得在同监督、同预算下作为单独假设评价。

**不支持什么：** 对不匹配源中心的强收缩总有利；本项目小型 EEG 网络需要 FiLM/LoRA；或者参数接近保证函数保持。当 input normalization、BN state、特征映射变化时，`theta≈theta_source` 不能保证预测接近。该项保持第二阶段优先级，不据此重新开启正则强度大网格。

### R7. PCGrad / CAGrad：梯度冲突控制是优化工具

Yu et al. *Gradient Surgery for Multi-Task Learning*. NeurIPS 2020. [arXiv:2001.06782](https://arxiv.org/abs/2001.06782)。Liu et al. *Conflict-Averse Gradient Descent for Multi-task Learning*. NeurIPS 2021. [arXiv:2110.14048](https://arxiv.org/abs/2110.14048)。

**本轮核实层级：作者摘要。** PCGrad 对冲突梯度做投影；CAGrad 以最差任务局部改善约束平均目标的更新，并讨论在论文条件下的平均目标收敛。此处没有重新核对完整定理条件，不扩大为本项目保证。

**支持什么：** 如果将来同一步联合 CE/decision/prior 的真实梯度经持续日志证明冲突，可将梯度方向和幅值作为可测量诊断，再提出有主任务优先级的改造。

**不支持什么：** 平均训练目标收敛保证测试 AUC/hit 改善；瞬时内积为负足以证明 A1×Q1 因果；用 PCGrad 修复错误监督单位、目标标签泄漏、极少完整 decision 或不相容源中心。故不是当前默认优化器。

## 3. P300 经典锚点的补充核实

- Rivet et al. *xDAWN algorithm to enhance evoked potentials: application to brain-computer interface*. 2009，[DOI](https://doi.org/10.1109/TBME.2009.2012869)，Europe PMC PMID `19174332` 摘要本轮已核实。空间滤波增强 P300，与 Bayesian LDA 配合；原论文摘要仅三被试，不能据此承诺跨十二库收益。当前 xDAWN + Riemannian 是独立 predictor，其标签、模板、SPD 原点都必须在合法 fit 范围拟合。
- Blankertz et al. *Single-trial analysis and classification of ERP components—a tutorial*. NeuroImage 2011，[DOI](https://doi.org/10.1016/j.neuroimage.2010.06.048)，Europe PMC PMID `20600976` 摘要本轮已核实。说明高维 covariance 估计是瓶颈，收缩 LDA 优于未正则 LDA；支持保留强线性锚点，不证明它一定优于当前网络。该文是 tutorial，证据角色为成熟方法与设计依据。
- 本轮不引用未经读取的动态停止论文来给 Q1 决策聚合背书。已有独立机制审查能给出本项目的直接反例：同一 selection 的多个前缀不是独立决策样本；完整等曝光/均匀先验下，候选共同 offset 与全局正温度不改 argmax。聚合效能仍须在真实 membership 和完整预算账本上测量。

## 4. 可执行的研究设计约束

**先后顺序：** 窗口/参考/membership 合同核定并冻结 → CE 源训练与普通 full-fine、scratch、波形收缩 LDA 锚点 → EA 完整路线 → 几何路线通过开发门槛后再研究有标签 RPA/GALIA。该顺序遵守当前 09-20/09-22 路线；不是宣称 EA 已经有效。

**fit 范围必须拆为三种：** strict zero-shot（目标 EEG 和标签均不读）；无标签 calibration-buffer alignment（只估校准统计并冻结，不读 test）；transductive test alignment（读取将被评分的无标签 test EEG，另列）。有标签 ERP 模板/类均值旋转属于 few-shot，不因 classifier 权重冻结而改变类别。

**源模型配对必须有两个层级：** 同一原输入上的 target state 替换实验，保持它要求的 source 坐标系；完整逐域 EA 训练路线则需要新的源输入/模型身份，使用相同源成员、预算与 seed 作 matched pipeline 对照。不能一边说复现 EA，一边要求复用不相容的旧 source checkpoint。

**推断单位：** participant 嵌套 study、共享 source fit 和训练 seed 都有相关性。首轮固定四库等权的 ΔAUC；K200/K800 共同 cohort 另报；源–目标因子需配对随机种子。bootstrap participant 不等于重训练多 seed。BNCI 的 decision 与其余 binary-only Q1 不混成同一种干预。

**停止条件：** 输入/状态/重放门禁通过但效应未达既定开发门槛，保留全分母失败证据；不按测试成绩为不同目标挑路由。后续 Q1 方案只能在新的预声明机制问题下启动，且保留 CE-only / matched scratch 对照。

## 5. 检索和证据边界

网络检索与原文解析均使用仓库 `.venv/Scripts/python.exe` 的标准库与已安装 `pypdf`；没有安装依赖、启动训练、修改生产源码或发送外部消息。所有 URL 于 2026-09-29 实际请求；arXiv/PMLR 原文可读取，HAL 元数据 API 可读取，但 GALIA/RPA PDF 返回 bot challenge。Europe PMC 为原论文摘要镜像，不能等同已读全文；逐项层级已如上标明。旧本地审计只在明确注明的 GALIA 细式处使用，未伪装成当前在线全文复核。
