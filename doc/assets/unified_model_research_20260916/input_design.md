# 统一模型：物理输入、保信息结构与模块交互设计

2026-09-16；只读设计。已读六份活动指南的相关部分及domain_transfer_plan_20260915，核查当前执行器、模型stem、mask、adapter、SplitBN与EA。未改source/guide，未训练。旧方案中的分阶段淘汰顺序不适用于本次用户要求；本次设计从完整联合图出发，同时检验条件贡献和交互。

## 1. 必须先统一的是可执行测量合同

本文件是输入与交互的独立备选设计，实施顺序以综合evidence方案为准：先固定Cz6共同profile、MAP、residual与统一decision目标；Q七维、集合stem、分解、BN、EA保留为明确后续交互扩展。下述13项矩阵保留作为另一种完整设计，不与首期输入、norm变化合并成无法归因的一轮。

当前source_records.SOURCE_EOG_PREPROCESSING是显式因果合同；公共既有模型通常消费六个Cz差分槽。BrainSync当前默认是完整记录双向、50Hz/Q30陷波、原生左耳垂参考七EEG导；EOG独立。两者同时存在，不能通过同一个shape、改reference字符串或换checkpoint文件名宣称单一输入合同。

首个可落地共同profile建议以当前产品数值程序为准：原始率先50Hz/Q30和0.1–30Hz四阶IIR、zero+mne_reflect_limited，源闭端epoch[-200,1200]ms、FFT至128Hz、截前179点、[-200,0)ms均值基线；全部实际source/target从raw重建。共同比较视图为Fz/P3/Pz/P4/PO8/Oz减实测Cz，严格mask。先在原始域确定QC/参考依赖及空间视图，再按最终依赖并集划分连续滤波段；各导已经按不同段过滤后再减Cz不等价。该空间次序也进入新profile，旧source在float32后投影的程序不被称为逐位相同。若产品另有真实在线需求，应另建完整因果profile，成对重建source/target；不能让同一权重签名兼容两种时间程序。50Hz是明确执行参数，不代表所有公开资料工频相同或残余伪迹已消除。

同一profile仍不等于同一有限记录算子：F_N依赖实际记录长度、分段与padding，source长记录和45闪烁短记录须在receipt中保留N与真实支持。短记录重放/源内匹配上下文是独立训练设计，不能把长记录过滤后crop标成短记录处理。raw隔离必须发生在filtfilt之前；先完整raw双向滤波再按训练/验证/测试标签切行不能撤回已经混入的未来信息。校准/最终测试分开raw，所有EA/QC/EOG/normalizer/BN状态按inner-train或final-calibration范围重新拟合。

该共同Cz6 profile是可核对锚点，不应成为永远丢掉native7的理由。它需要自己的同输入scratch及LDA。当前因果Cz6缓存和checkpoint仍作为旧实验，不原地改写。

## 2. 差分加实测Cz是可逆扩展，值得放入完整图

令x属于R^(7×T)，依次为Fz,Cz,P3,Pz,P4,PO8,Oz，均相对同一次真实采集参考r。D的六行是e_c^T-e_Cz^T；定义：

  u = Qx = [Dx; x_Cz],   Q=[D;e_Cz^T].
  rank(D)=6, D1=0, rank(Q)=7.
  x_Cz=u_7, x_c=u_c+u_7.

共同参考平移x->x+1a(t)不改变Dx，u_7则变成u_7+a(t)。u_7应称“参考敏感的实测Cz分量”，它包含Cz脑活动及参考影响，不能把它直接当噪声/公共模态丢给private分支。一个全导共同ERP分量1p(t)会完全落在ker(D)；没有第七分量不能恢复它。

建议统一输入图保留两个显式角色：contrast tokens和anchor token。contrast共享编码；anchor经参考条件化的小映射进入任务融合头，保证有真实任务梯度。SupCon/MMD先只约束reference-invariant/shared表示，不能强迫不同采集参考下的anchor原始分布相同。anchor与“private domain representation”是不同概念。

约束：

- 第七路必须来自真实raw；Cz6缓存不得补造。Q、原参考身份、每个物理依赖与新执行receipt共同持久化。
- 源u7相对r_d，产品u7相对left earlobe。元数据条件化可组成同一模型的显式多参考输入族，但不使电压物理语义自动相同。只有实际测得可重参考到共同位置时，才能声称共同参考；未知仍未知。陌生参考条件需合法校准，K0只使用预声明的source端初始化/固定策略。
- m_u,c=m_x,c AND m_x,Cz；m_u,7=m_x,Cz。Cz坏时差分路径失效；完整可逆性不能外推成提高缺Cz时的证据覆盖。保留原始native7数据与其独立原生基准，不伪造替代Cz。
- GTN原生只有Fz/Cz/Pz，最多得到2个差分+1个anchor，rank最多3；七槽只是接口容量，GTN没有提供七路信息。单库GTN与目标的迁移效应先在相同两导/三信息维度比较；多源联合覆盖可以让共享模型接收不同mask，不能宣称每个来源均训练了每个导。
- EOG只在冻结清理/QC中使用，先识别原始坏导再做空间混合；不进入EEG token或头皮几何。EOG标签相关并不证明是P300脑证据。

裁决：把anchor保信息开关R纳入完整联合设计，以及R×decomposition、R×normalization的预声明交互；共同Cz6是R=0对照。R=1收益可能来自多一维输入，不能全部归因于迁移。必须同样提供R=1 scratch，比较transfer gain(R=1)-transfer gain(R=0)。

## 3. 固定stem与电极集合stem的真实差别

当前N2P3Net.st_spatial是kernel_size=(n_channels,1)的固定全导混合，forward_features只接x及domain参数。Deep._prepare_input会在标准化后恢复mask零值，但模型自身没有显式trial mask接口；在早期BatchNorm里，缺导零仍可参与矩。不能把此接口描述成已实现完整集合编码。

建议只替换空间聚合入口，复用现有共享时间卷积、MST与读出：

  v_i,c,t = phi_theta(u_i,c, local_time)_t + e(lead_c, role_c, reference_c)
  h_i,j,t = sum_{c:m_i,c=1} a_i,j,c,t V_j v_i,c,t,
  a_i,j,c,t = softmax_{c:m=1}(q_j^T K v_i,c,t).

phi保留有符号波形与事件锁相；空间lead描述对于差分包含被测导与Cz，anchor包含真实采集参考。时间位置只编码相对stimulus时间；不按数组槽号推测电极位置。模板坐标与实测坐标分开记账。同步重排数据、mask、lead metadata后输出应不变；全部mask=False必须拒绝，不能softmax空集合。

固定stem对照采用同一真实输入、预算、mask与尽量匹配的norm/参数量；跨stem直接更换BN为LN同时改变两个因素，只能报告联合策略效果。集合分支可用token内归一化，缺失token完全不进入聚合；如沿用BN，需明确只在实测token上更新统计的数学规则，不能依靠最后再乘mask修复已污染的均值。

BIOT正文支持按channel token组织异构记录及显式缺失处理，但其segment采用频谱能量向量；|FFT(x)|²=|FFT(-x)|²，故不能据此证明适合区分有符号ERP。我们借用集合接口思想，不照搬energy-only tokenizer，也不把其TUEV人工缺导实验等同于本项目GTN三导到七导。

## 4. 所有机制可以在一个训练图工作，但各有不同身份

建议结构图：

  typed raw/receipt -> deterministic reference views + true masks
    -> optional legally fitted EA view -> shared signed temporal/set stem -> MST h
    -> invariant adapter zI + private projection zP
    -> task fusion(zI, reference-sensitive anchor features) -> binary/decision heads.

domain head(zP)为训练支路；它不必进入最终推理才算有效模块。不得把明确的domain/reference ID直接喂给该head再宣称从波形学出了style分解。target adapter从合法calibration拟合，source残差在目标推理仍按现有合同关闭。

  L = L_task + lambda_SC L_SC(zI)
      + lambda_CM L_conditionalMMD(zI)
      + lambda_dom CE(domain_head(zP),d)
      + lambda_orth ||Cov(zI,zP)||_F²
      + lambda_anchor ||theta_T-theta_S||².

这组损失必须按训练阶段解释：源阶段L_task=CE_S且lambda_anchor=0；目标阶段只在合法calibration上拟合任务/锚定项。跨域SC/MMD若没有显式合法源上下文则不在单一目标域强行制造正对；已有AS-MMD联合源/校准模式须另声明，不读取最终测试。统一graph允许训练阶段不同，不意味着所有loss在推理或每个阶段都有效。

SupCon和MMD是同一特征/参数图上的损失函数，不是必须另加encoder的结构模块；复用相同行账本、一次特征前向、原CE批次与更新预算。它们可以同时打开。当前SourceObjectiveConfig.kind为互斥值，实际落地需集中扩展为显式目标组合及各项诊断，而非在不同runner私自相加。

BN/SplitBN是有状态的归一化层；EA是有拟合范围的线性变换；decomposition是参数化分支加训练目标。统一graph不意味着把这些对象都称为可学习子网络。目标校准会改变theta或BN统计时，依赖其激活的其他拟合统计必须有更新顺序；冻结fit-state和输入参数变化不自动相容。

EA只在确实观测的协方差子空间上定义：

  R_O = mean_{i in legal fit, observed O}(X_i,O X_i,O^T/T),
  A_O = [(1-rho)R_O + rho tr(R_O)/|O| I]^-1/2.

不把缺导零当真实样本，不用epsilon制造可观测性。完整六导EA不适用于从未联合观察该六导的GTN。若扩展逐观察子空间EA，必须有合法co-observation支持及冻结矩阵；未知mask的变换不能从不存在的跨导协方差猜出。未经验证的缺失模式可让该EA支路“不适用”并记账，主原始波形路径继续工作；这是一项预声明输入能力规则，不能按测试分数决定。

EA输出是混合信号，不能再把每行当作原电极的真实测量点。集合stem若消费它，须携带实际lead mixing矩阵A_O D_O或放进独立的内部aligned-view类型；保留raw contrast路径。完整观测且A可逆时，[A D;e_Cz^T]仍秩7，信息保留原则可以与EA并存；实际float32只声称数值容差内重构。第一实现可在共同完整子布局复用现有EA，对其余真实通道保留原路径；不能将“共同三导EA收益”解释成完整七导对齐收益。

对齐不保证迁移。Zhao等给出域不变及低源误差仍不足的反例；类比例变化时Z=Y可以完全判别但仍揭示domain。条件MMD避开这个特定边缘先验混淆，却不能保证不同范式的P300标签涵义及类内任务信息完全相同。MMD/SC宜作用于明确shared子空间，anchor保留任务信息。

Domain Separation Networks原论文联合shared/private、任务与重构目标，支持“结构需配合目标”的概念；该证据来自图像域适配，不能证明当前P300正交支路已有生理可解释性。Cov(zI,zP)=0不推出独立，更不推出脑信号/伪迹分离；zI=x，zP=x²在对称x下协方差为0但完全依赖。

## 5. 避免门控无效与缩放不可识别

对h'=h+g f_phi(h)：

  dL/dphi = g J_f^T grad_h'L,
  dL/dg = <grad_h'L, f_phi(h)>.

若g=0且f=0，两类梯度同时为0，模块可能永不启动。当前FeatureResidualAdapter是gate=0而内部投影非零，不应误报为必死：首先可更新gate，之后分支权重才获得任务梯度。但联合设计不能再同时把末层与gate清零，亦不能把sigmoid预设在饱和区。

建议full-stack的科学开关由预声明配置控制，不以自学软gate代替有无因素。identity初始化只选择一种：固定非零gate+零末层，或零gate+非零分支；若保留现有后一种，测试首步gate梯度及第二步分支梯度。学习gate可以是另一个固定候选，不能以其大小排列模块价值。

因为g f=(g/a)(a f)，有界g并不限制整个分支贡献。相邻BN affine与FiLM也可合并为一个affine。不要靠可互相吸收的参数证明机制；删除重复自由度，或固定一个尺度约定。诊断采用实际增量范数||g f||/(||h||+epsilon)、反事实有无支路的输出/损失、有效梯度范数与不同目标梯度夹角。正则值下降、gate缩小或domain分类变难均不能独立证明迁移改善。

## 6. 小而可辨别交互的完整矩阵

在同一输入版本、source pool、source/target seed对、K和训练预算下，固定六个因素：R=实测anchor信息、S=SupCon、M=条件MMD、D=分解支路、B=指定SplitBN策略、A=合法EA view。所有因素的on/off含义在run manifest固定，结构不适用比例另报。

最小核心12个训练程序：

1. full-stack(R,S,M,D,B,A全开)；
2. 6个leave-one-out，分别关闭一项，测其余都在时的条件贡献；
3. 4个双移除：{S,M}、{D,B}、{R,D}、{A,B}；
4. 1个同输入/同骨干的plain task-loss锚点（R=1，保留anchor，其余扩展关闭）。

anchor本身和已有两个单移除共同组成各对2×2：

  I_jk = Y_full - Y_without_j - Y_without_k + Y_without_jk.

这给出其他因素固定开启时的条件交互。full-stack减LOO不是孤立主效应，不能把这些数相加还原总收益。若预算允许，R×B再加一个双移除单元；这是预声明扩展，不由测试结果挑有利pair。

共同Cz6的plain task-loss基线（R=0）还须有一项：只有相同新profile、实际输入和剂量均已执行时才可复用，否则为第13个训练程序。不能用旧causal权重填这个位置。

另设固定stem/集合stem成对对照，至少在同一R=0共同profile与相同训练目标上比较；如果不能匹配norm，则明确它回答stem+norm联合策略。每种信息维度有同输入scratch/LDA，最终必须报告[transfer-scratch]的差异，而非只比较七导迁移与六导旧源。已有合法source/基线拟合按真实输入与方法身份复用。

用合成响应Y=.70+.03*S*M即可反证“单独S/M无增益就不做组合”：两者单开均.70，联合.73。这个数只是数学反例。本矩阵不是逐项淘汰树；每个被授权单元无论先导符号都执行并保留，机制判断由预声明全表支持。

每单元须记录：源/参与者/类/mask实际覆盖、各loss有效项数、CE与aux梯度范数和夹角、BN/EA fit来源及状态更新、参考分支实际使用比例、source/target参数量和训练成本。比较既报告真实operational覆盖，也在共同可评估事件/decision交集作机制分析；不能按模型置信度删行制造配对。

## 7. 代码联动地图

| 位置 | 已有职责及方案变化 |
|---|---|
| data.contract / data.epochs.PreprocessingSpec | 新共同数值profile、参考视图及mask物理角色；旧causal/zero包不重标 |
| data.preprocess.execute_preprocessing | 继续唯一数值执行、真实分段/支持receipt；Q空间投影与QC依赖不能被caller私自交换顺序 |
| data.source_records / data.brainsync / data.domain.adapt_observed_montage | 从raw重建共同profile、保留native anchor、实测坏导/参考门禁；GTN缺导不补成观测 |
| baselines.deep._prepare_input / N2P3Net.st_temporal, st_spatial, forward_features | 复用时序和MST，明确新stem接受mask/lead描述；现有零mask不等于masked latent BN |
| models.adapters.FeatureDomainTransfer | 复用shared/private/target/source残差，anchor任务支路不能等同private nuisance；初始化避免双零死门 |
| models.splitbn | 复用显式域与目标buffer选择，源训练/校准/只读测试状态分开 |
| data.euclidean_alignment | 复用完整观测EA估计核；缺失子空间/混合lead语义另建明确wrapper合同，不在runner补零白化 |
| transfer.source_objective / transfer.alignment | 一次前向共享SC/MMD特征与身份；显式组合配置、合法空项和原始输入finite门禁继续保留 |
| models.parameter_groups / transfer.recipes / SubjectAdapter | graph参数归属完整互斥，结构因素与loss因素分开；同K内层fit和完整refit一致 |
| transfer.checkpoint / calibrated / research.source_matrix / stats.transfer | 架构、输入族、目标函数组合、fit范围、paired seeds及因素层级进入身份和统计；缺项保留分母 |

本设计不把metadata-conditioned输入族伪装成所有记录同一电压参考，也不要求新模型无条件接收所有旧cache。可复用的是明确共享的算子和参数模块；完整执行输入及合法fit状态必须可重建。

## 8. 本轮文献读取层级及证据文件

联网共8次：4篇相关论文摘要成功，其中BIOT进一步全文方法/实验核验；2个候选ID指向无关论文，1次Zhao全文URL为404。错误候选未进入EEG证据，不继续猜测编号。没有援引未核实的论文数值。

- Yang C, Westover MB, Sun J (2023). BIOT: Cross-data Biosignal Learning in the Wild. https://arxiv.org/abs/2305.10351 ; 正文 https://arxiv.org/html/2305.10351v1 。读取§2 tokenization、§3.3缺导/缺片段实验。支持异构channel token和mask思路；其频谱能量编码、人工缺失假设、癫痫等任务不可直接迁移为P300已验证结论。
- He H, Wu D. Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach. https://arxiv.org/abs/1808.05464 ; DOI 10.1109/TBME.2019.2913914（DOI来自已核验项目指南，本轮未重新访问DOI）。本轮摘要：包含ERP/MI离线和模拟在线，unsupervised alignment不等于允许本项目读测试统计。
- Zhao H et al. (2019). On Learning Invariant Representation for Domain Adaptation. https://arxiv.org/abs/1901.09453 。本轮摘要核验；条件偏移反例、标签边缘不同下不变性/联合误差取舍。全文请求失败，不冒称逐定理精读。
- Bousmalis K et al. (2016). Domain Separation Networks. https://arxiv.org/abs/1608.06019 。本轮摘要核验；shared/private联合任务与重构，图像域适配证据，不能背书当前P300生理分解。

本目录的input_retrieval.json保留URL、读取层级、简短摘录和失败/排除记录，不复制整页HTML；原网页临时取证保留在系统temp的n2p3-unified-input-20260916目录。input_math.py以项目.venv运行，输出input_math.json，数学反例全部通过：rank(D)=6、rank(Q)=7、重构误差4.44e-16、共同参考变化下差分误差8.88e-16、正负波形频谱能量完全相同、双零门梯度为0、单零gate尚可更新、交互不能由baseline单项消融发现、零协方差不保证独立。没有模型训练或新效果分数。

技能使用：已读eeg-bci-transfer-learning；其旧模型描述、固定方法排名及旧样本预算不覆盖当前六指南，本轮只采用原文核验与不编数规范。记忆仅辅助定位GTN三导/参考合同；关键事实均由本轮活动指南与源码复核。


决策批次约束补充：一个450-row完整decision满足现行batch上限512，单个完整group可以构成一个batch，不能要求每batch至少两个decision而误拒。超过512的完整group须明确报告当前能力不支持，或另行定义并验证分块合同；本方案不暗中放宽batch上限。未来分块即使保留全group的softmax，也须另外处理BN状态、随机层和梯度缩放，不能把它拆成若干独立decision后平均。450只是明确示例，实际行数以membership和真实事件账本为准；45闪烁产品测试与450闪烁校准会话不能互换分母。


