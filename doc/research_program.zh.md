# N2P3-Net：公开ERP预训练辅助成人猜数字

版本：2026-09-06d，作者拒收被试排除版。唯一活动研究规范，负责目标与科学约束。
公开BI等数据用于学习可迁移P300/ERP特征；目标硬件为七路头皮EEG和一路独立EOG。
阶段、配方和验收统一见[执行计划](roadmap.zh.md)，实际接口见[技术结构](technical_structure.zh.md)，
时间/信号推导见[输入合同](input_contract_math.zh.md)，实测与文献见[证据索引](evidence/README.md)。

## 1. 产品目标与数据角色

最终问题是在成人BrainSync记录中，用预声明的本人已知目标calibration，识别随后独立
decisions中对推理程序隐藏的目标数字。当前用户预算为九个数字各10 block，每block
45次闪烁，共4050 trial/450 target；新测试每次1 block，九选一hit@R5为主要产品读数。
所有数字可参与校准。测试数字身份可重复，但测试EEG/标签必须独立于拟合和配方选择。
目标为1–9九选一，不做认知能力或医疗诊断。
产品开发愿景为participant-macro `hit@R5 >= 0.90`，不是实测成绩或个体保证；
正式验收界限与样本量需结合成人先导数据预声明。
采集槽顺序Fz/Cz/P3/Pz/P4/PO8/EOG/Oz：原PO7槽采PO8，原PO8槽用于EOG。
分类EEG为Fz/Cz/P3/Pz/P4/PO8/Oz，共七导。EOG独立保留于raw/BIDS，用于伪迹处理与QC，
不进入EEG卷积、分类特征或头皮CAR；放置方向/参考未知时不猜作VEOG/HEOG。
不能全局改写公共数据原始电极名，不能把插值/补零当实测通道。

2026-09-08用户确认BrainSync REF为左耳耳垂耳夹，规范值`left earlobe`；旧T7/T3为记录错误。
本人target-only基准优先保留原生七导信息，已有跨源模型继续显式Cz六导适配；后者的迁移收益
仍与同Cz输入scratch配对。这个双路径选择依据可逆性、信息秩与源输入合同，不宣称已经实测
原生七导更优。左/右耳垂不等效，不能将公共源native权重的参考字符串改写成BrainSync参考。
见[科研判断与更正验证](evidence/brainsync_reference_20260908.zh.md)。

| 数据层 | 作用 | 结论边界 |
|---|---|---|
| public binary ERP source | EEG、target/non-target、participant/session/domain和采集provenance | 表征学习与source数据剂量；不要求数字/candidate身份 |
| public pseudo-target | 留出source未见participant/domain，在固定后续段测试 | 是否降低公共目标校准标签/时间成本，development |
| 合法candidate数据 | 有每flash物理membership及真实decision边界 | 该数据自身的固定预算/聚合/停止，不自动是成人九选一 |
| adult BrainSync | 七EEG+EOG、completed、已确认标签、独立target-switch记录 | 成人九选一、真实成本、跨日与最终确认 |
| synthetic/contract | 不变量、梯度、序列、失败状态 | 软件/数学验证，不是性能 |

此前核查的旧BrainSync目录只有error记录；本次用户补充`BrainSync_Tool_newest`后发现
4份completed/confirmed候选记录和3份error记录，见[版本比较](evidence/brainsync_versions_20260906.zh.md)。
2026-09-07用户进一步确认P001目录为同一人，新增记录后共有5份完整session；
派生副本已经通过显式身份、角色、坐标及BIDS读取核验，并完成本人先导云训练，
见[P001证据](evidence/brainsync_p001_pilot_20260907.zh.md)。年龄、光学时钟、可靠成人效能
仍未确认，预选方案后续AUC0.4926且严格R5覆盖仅10%，不能称已经获得产品确认。
error记录、旧GTN、BI和合成信号不能冒充成人确认。
GTN儿童三导且固定thought digit，只能作声明限制下的开发研究。

## 2. 活动算法路径

源池选择与数据准备资格分开记录。B7已在09-09完成扩源比较；09-16统一模型固定四库用于机制比较，
不能据此推断新增库未接入或永远不适合。新source_pools/外折成员必须在评分前明确，
任一成员准备失败都保留完整请求及失败原因，不能沿用原池名删库。BrainForm/Cattan/BI2012的
连续reader与准备合同已于09-18接入；其代码和无训练探针不构成新迁移成绩，
Lee完整下载/转换与Mainsah外部角色仍按各自证据处理。

2026-09-16统一矩阵已完成并重放核验，详见[结果分析](evidence/unified_results_20260916.zh.md)。
公开开发证据支持校准后的迁移收益，尚不支持完整新增损失包优于同结构普通微调；
BNCI是本轮唯一实际启用decision训练的库，其Q1包的AUC和真实36选一hit均下降。
用户随后明确优先从科学推理、源码和原始协议判断，见[独立机制审查](evidence/chain_mechanism_20260916.zh.md)。
研究优先级先看完整决策监督、损失单位和输入信息保留；既有分数不用于倒推唯一病因。
今后的prior/decision效能归因仍须固定源、匹配raw范围和工作集，不能据组合对照直接删除单一模块，
也不能将这些公共36选一或trial AUC结果等同成人BrainSync九选一确认。

原协议Train/Test可来自有文献来源的实验阶段，不限于文件分开。来源没有原生phase字段时，
明确记录派生规则与出处。当前BNCI单录制前缀方案保持其预声明的混合阶段开发含义，
不称官方测试复现；将来官方phase方案另建输入身份。预算同时报告K、正例、完整decision、
重复数、目标覆盖、录制时长与phase，同一selection的多个前缀不当作独立样本。

2026-09-09用户确定应用双向滤波，BrainSync新默认为`brainsync_bidirectional_v1`。
完整校准记录分别处理，未知目标推理在完整记录结束后执行；各轮分数为回顾性证据。
双端有效状态误差经实际epoch算子评估；自然采集端点敏感度作诊断，含伪迹切段的片段使用
完整双端条件界。原始滚动幅度可由漂移造成，新拟合默认只诊断，突变/恒值/缺样仍留原始账本。
六条件本人QC/模型对照及已知漂移/ERP验证见[联合优化](evidence/bidirectional_qc_20260909.zh.md)。
既有四会话及本轮受污染记录仍是开发证据；持续污染、经验包络外输入和新成人效能不能据此认证。

2026-09-07连续质量研究选择先处理原始污染区间及滤波状态，再评估EOG和局部QC，详见
[文献、数学与实施规格](evidence/brainsync_signal_quality_design_20260907.zh.md)。软件实施及旧数据
开发重放见[验证记录](evidence/brainsync_quality_implementation_20260907.zh.md)，不变更窗口/频带配方。
2026-09-08增加逐通道恢复：在明确清洁校准片段上拟合经验状态包络，按实际重采样/基线算子
计算恢复期，参考/EOG依赖共同传播。该界只约束隔离异常后重置的状态误差，不保证持续输入无污染。
62个冻结注入反例中仍有六个小幅30秒斜坡漏检，见[局部恢复证据](evidence/brainsync_recovery_20260908.zh.md)。
单HEOG已明确为左眼眼角，不能称完整眼动校正；这次旧记录重放未使用EOG回归。
标签无关的质量门限仍可能选择性删除真实较大ERP；
须做已知信号保真反例并报告冻结规则后的类保留率、完整决策覆盖与实际成本。
已观察过的P001会话用于开发，不能再无偏确认新配方；新确认使用独立后续采集。
补充污染记录用于分类器训练时，可显式固定原质量校准模型；新增训练记录不得改变该冻结门限。
严格核对原QC拟合行的事件、信号、mask、特征和摘要，保存/推理携带相同QC状态。
校准内重新拟合policy是另一明确模式，不能把已做epoch QC的缓存当作未经选择的训练输入。

2026-09-08个人QC采用`personal_epoch_signal_evidence`：物理PTP、STD和差分粗糙度只描述
逐导相对校准分布的偏移，不能单独作为伪迹硬拒。跨导中位数归一化的相对幅度依赖他导及mask，
低STD校准分位数也不能证明断线，故新个人路径不再使用这两项硬门。公共FoldLocalArtifactPolicy
和旧冻结个人模型保留原方法身份；新旧JSON由method/schema明确分派。
新个人QC要求已应用匹配的逐导连续QC `/4`，在重参考前检测原始数字恒值区间；本次显式设为5秒。
epoch精确恒值另作无动态证据兜底，原始支持、异常、参考依赖、恢复期及最低有效导数仍决定准入。
诊断包络仅在合法校准训练组拟合、随后冻结，测试不能更新；内层选方也遵守各自fit范围。
文献、反例及三臂57拟合/114预测见[QC重设计证据](evidence/brainsync_qc_redesign_20260908.zh.md)。
保留增加不等于污染消失，新方法也未在第五份记录稳定胜过重拟合旧QC；慢漂移和成人效能仍待验证。

其余前置质量问题已按[持续趋势、采样支持与处理对照](evidence/brainsync_qc_remaining_20260908.zh.md)落实。
持续趋势以5/15/30秒逐导稳健斜率、残差MAD和相邻趋势变化描述，不作伪迹认证或自动扣除；
外加漂移能抵消原始趋势，不能以更小斜率证明更干净。恢复证书只覆盖重置初态齐次误差，
不覆盖持续输入污染；当前不确定集内提前一个采样即可构造超1uV反例，不擅自缩短等待。
新采集的刺激前基线和每block响应尾窗必须确认同段真实样本及checkpoint，最终clock与BIDS断点一致。
EOG/频带/参考12条件、72拟合/144预测已完成并复核，结果不支持统一开启EOG或按旧测试选择默认频带。
显式改变频带只用于新模型的声明合同，源checkpoint兼容门禁照常执行；已有公共源不得重标为新频带。

### 统一校准档位（2026-09-07用户规定）

后续校准预算比较统一请求 **K=0、200、800、2000**，K计有标签trial/闪烁数，
不是正例数、epoch更新次数或block数。不再默认K96为主比较，四档并列报告。
K0仅使用源模型及源统计，不在目标上拟合统计、微调、BN或LDA；正档位不重复训练零校准臂。

若样本不足以执行最高请求档并保留原协议要求的测试量，按时间顺序使用约80%可用trial
作为校准上限，剩余约20%作为测试，并继续执行epoch时间支持的purge。
例：单记录150个可用trial时采用实际K120，不能把它标成K200/K800/K2000；
可达的标准低档仍保留，超过上限的请求映射到同一实际档，避免重复计算。
有原始独立Train/Test时，Test保持原划分；80%只作用于不足的Train校准池，不能从Test补样本。
若purge后无测试证据，进一步降低上限；校准/测试缺类别则报告失败，不按结果补样本。

记录必须同时保存请求档、实际K、回退原因、校准正例数、测试样本数及划分规则。
同一被试各可用K（含0）共享按最高实际K确定的test，跨K比较同时报告各档人数和共同被试结果，
不能把不同实际K混为同一标准档求均值。小test结果应明确精度限制。
代码默认权威为`transfer.binary.STANDARD_CALIBRATION_TRIALS`及
`LIMITED_CALIBRATION_FRACTION`，两类binary runner共用解析与回退实现。
历史冻结的K48/96/192等记录保持原身份；既有采集总量4050不是本轮被重写的评估档位。

每次正K迁移实验必须包含同K无迁移消融：同结构N2P3Net随机初始化，
不继承source权重、BN或normalizer，所有目标拟合只用同一校准行。
迁移与scratch共享被试、实际K、校准/test行及标签、通道mask与预处理；
训练预算和调参机会须匹配并显式记录，同时保留原波形收缩LDA基线。
80%回退后按同一实际K配对，不用“更少校准的scratch”衬托迁移。
结果必须给逐人`迁移指标-scratch指标`、负迁移比例及失败配对，不能只报迁移绝对分数。
CLI在正K迁移遗漏scratch时拒绝启动；实际结果以校准/test行digest核对配对，
任一臂失败则保留`incomplete_pair`，不制造差值。K0只有零校准，不能拟合无标签scratch/LDA。

`公开raw → 明确binary source视图 → N2P3Net监督预训练 → 源head保留/目标微调 →
目标独立EEG证据 → 目标协议九选一聚合/停止`。

上述链条现已接入整合后的BrainSync主工作站，见[实施记录](evidence/brainsync_app_integration_20260906.zh.md)：
`transfer.calibrated`和`brainsync_model_task.py`提供内部选配方、完整校准refit、个体模型
保存/加载及无标签completed recording推理；前端新测试采满45闪烁后自动判定，再确认目标。
旧通用manifest仍是preprocess入口，不能把它和新结构化后端任务混为一谈。
本轮完成软件/模拟链验证，未实现逐SDK chunk推理，也不宣称真人或实机效能。

源训练只需可靠二分类标签，不要求BI恢复固定物理row/column、九选一或R8。
对candidate字段显式projection，X/y/participant身份不变，父cache及被移除合同保留provenance。
训练按participant或真实domain分组，不由可疑selection猜测创造监督或独立单位。

第一比较基线：plain N2P3Net supervised CE、zero-shot、classifier_fine、full_fine及
同结构target-only随机初始化。classifier_fine复用源head；random linear只作额外表征诊断，
不能作为唯一无迁移基线。normalization、BN、dropout、标签/更新预算与epoch选择规则必须
明确，完整方法比较不冒称某个机制的因果效应。

EA、AS-MMD、Split-BN、TSD、Dense PDS、源残差、invariant/private等已有模块为
可复用研究能力。模块单独无增益不阻止预声明组合实验；条件贡献和交互须在相同数据、剂量与拟合范围下检验，
不能以CI含0宣判无效，也不因全部叠加便自动成为产品默认。
复用既有模块及成熟MNE/sklearn/PyTorch工具；新机制先证明简单路径留下的具体缺陷。

2026-09-16统一模型的科研目标为一个共享前向图和自足个人包：源预训练、本人校准、只读推理连续使用，
阶段通过合法监督、拟合状态和更新权限区分。首个设计为全参数有限源中心收缩、已有特征残差和共同membership决策层，
源SupCon/MMD可组合；固定层是优化控制，不等于删除该层。模块取舍依赖完整组合下的贡献及成本，
不以单项筛选先淘汰可能的协同。公式、风险归一化、组合设计及反证见
[统一模型方案](evidence/unified_model_plan_20260916.zh.md)，当前尚无新架构效能结果。

首版主链已实施。源内λ选择只消费被排除在参考源拟合之外的参与者，按seed后participant等权开发AUC选择；
外层目标不参与。最新[精简训练裁决](evidence/unified_compact_training_20260916.zh.md)只对每折一个源内参考模型早停，
其e*冻结到该折两种源目标的三个seed；个人固定90轮直接拟合全部授权C，不再强制个人内部选轮和第二次重训。
首轮估计源目标包A与个人适配包Q的2×2交互及相对scratch/LDA收益，不分离包内各模块作用。
共享预算由A1参考选择，三个seed条件于冻结e*/λ；实际剂量及选择成本另报，不宣称各方法分别最优。
membership损失仅从真实物理台账构造；不完整请求保留风险质量，ties仍参与训练，不能按模型当前分数挑可用组。
个人decision分数为固定归属下的logit差累加；softmax只是内部归一化，尚无概率校准证据。

LDA顺序实验作为独立开发对照：微调后冻结特征并拟合收缩LDA，或用校准LDA初始化
线性头再微调；原微调head作共同基线。标准化系数必须折叠到head：若LDA分数为
`a^T((h-m)/s)+b`，则`w=a/s`、`c=b-w^T m`，两logit取`(-g/2,g/2)`使差等于`g=w^T h+c`。
后续交叉熵更新后的head不再称为LDA统计解；本轮不引入可微协方差或交替优化。
校准内选方时，LDA初始化和最终重拟合也必须各自在inner训练范围内重做，不能预先用
完整校准特征拟合后再声称inner validation独立。

## 3. 估计量与迁移收益

2026-09-16算法链修复统一内部训练范围检查：个人学习型raw处理须来自合法inner训练范围，
源权重/统计按源身份合同独立核验；固定LDA无内部选方时使用完整授权校准范围。冻结raw QC/EOG
与最终refit可重拟合的epoch policy分别持久化，不能因QC路由改变统计/BN/配方含义。
配对重复关系从执行计划传到结果与分析，不从成功cell推断；不同源方法保留独立因素，复用scratch/LDA
不增加独立重复数。预声明库缺失时全库主估计不可用，部分库读数另列；已知实际K失败保留分母，
未知K不虚构。缺失AUC对比的识别界按正系数总和及库等权计算，不是置信区间。
实施及数学验证见[八项修复证据](evidence/algorithm_chain_repairs_20260916.zh.md)。

2026-09-16域迁移首批实现只开放显式P1/P2研究处理，蒸馏已取消；实际顺序见路线图。
P1空间组指`st_spatial`，时间组包括`st_temporal`及MST depthwise/pointwise，
读出与BN affine独立。该分组用于代码归因，不把MST pointwise的特征混合称为纯生理时间效应。
P2保留源CE抽样及固定更新剂量；SupCon只用不同真实域、不同参与者的同类正对，
同域/同人同类行不作假负对，重复训练行只在辅助项去重。辅助项采用post-MST展开特征的无参数L2投影。
目标测试不进入源目标函数，任何源目标函数改变都进入配置、轨迹复用键及checkpoint合同。
这些新增能力仅有软件验收，不能提前宣称优于完整微调或减少真人校准量。

\[
\Delta(K)=M_T(\theta_u(K;\theta_S))-M_T(\theta_u(K;\theta_0)).
\]

theta_S包含source参数/显式source统计；theta_0为同结构随机初始化，target-only只用
目标calibration统计。所有K使用共同later test或明确队列变化。source数量、participant
多样性与optimizer updates分开控制；多域权重与类别权重分开记账。

| 估计量 | 标签使用 | 测试范围 |
|---|---|---|
| Z0 | 无target标签 | target未参与source训练；不先用未来统计 |
| S | early已知目标calibration | later独立未知目标decisions |
| O | 同selection prefix标签 | 同selection suffix，仅oracle proxy |
| binary time-split | early有限trial标签 | 单recording固定后续trial，仅binary开发意义 |

`unseen_calibration_codes`保留为额外严格对照，不是当前九数字全校准产品的必要门禁。
`observed_sequence`保留自然目标序列；不能事后删除相邻重复目标来抬高成功率。

LOSO术语须指明标签使用层级：纯zero-shot LOSO在source训练及任何目标拟合中都不使用
留出者标签；“LOSO源预训练+目标自身校准”只保证source被试隔离，随后显式使用该人的
早期校准标签；“8人共同留出”不是64折LOSO。三者不能用一个LOSO标题混报。
全64人、通道与校准规则配对复核见[BI64设计与运行记录](evidence/bi64_protocol_factorial_20260906.zh.md)。
该[64人实测](evidence/bi64_protocol_factorial_results_20260906.zh.md)显示16导强微调优于5导，
selection改trial本身收益很小。五导继续作为真实硬件交集对照，不当成所有公开source的最佳信息保留方案；
如何保留16导source信息并迁移到7导目标，需要另行验证，不能将缺失导插值当实测数据。

公共source健康度用participant-macro AUC、类平衡读数、学习曲线和失败率；不替代成人指标。
产品主读数为requested-participant operational九选一hit@R，实际请求的失败、tie、
incomplete、abstain保留分母；条件valid准确率另报。还报告correct selections/min、
calibration分钟、NLL/Brier、coverage及低表现被试。R由目标数据/产品预算声明，
BI R8不再是binary source训练/晋升门。

sum-logit在条件独立且score为LLR时有Bayes依据；非线性不保证先平均EEG更优。
ERP平均/candidate损失必须有真实membership。正仿射log-odds在等覆盖下不改变argmax，
概率校准的收益应按概率质量及停止风险解释。

## 4. 数据、时间与空间合同

2026-09-15预处理执行科研裁决区分声明兼容、有限算子等价和字节复现。
纯架构重构需保持实际信号、事件支持、QC及失败分母；无此证据不能据同参数宣称同算子，
也不能宣称分类性能提升。拟合范围与数据来源身份分别核验，不要求合法source/target数据摘要相同。
严格执行入口、七处调用和凭证门禁已落实，见[实施验收](evidence/preprocessing_execution_20260915.zh.md)。
原始来源/拟合范围、声明语义、实际执行及字节身份分别绑定；旧无执行凭证数据可解释，但不补造认证。
闭端支持和近等采样率恢复修正不得回写历史缓存/成绩；不以软件等价宣称识别率提高。
理论与反证门槛见[预处理合同执行研究](evidence/preprocessing_contract_research_20260915.zh.md)。

2026-09-09用户明确要求BrainSync工频陷波：新本人校准使用50Hz、Q=30二阶IIR与原带通级联，
参数来自独立BrainSync输入合同，原公共数据配方不按参与者或库名推测工频。
这是输入方法变更：新模型、QC恢复证书和缓存需绑定包含陷波的实际算子；旧成绩保持原处理身份。
信号抑制和已知ERP保真验证不等于真人识别率增益，见[实施证据](evidence/brainsync_notch_20260909.zh.md)。

- 使用项目.venv；绑定raw、cache、环境、源码archive、模型、checkpoint及结果。
  authority为participant_key，本地ID只作provenance；namespace不改变真实身份。
- source与target不共享训练participant，除非另行声明within-participant问题。
  双人同步记录保留不同participant_key，并按dyad/共同记录隔离依赖。
- 当前recipe为128Hz、V、连续0.1–30Hz四阶Butterworth forward、steady-state初态、
  [-200,1200)ms、mean-only基线；不是永远最优，对照变化需新合同/cache。
- 连续滤波后epoch，chunk携带状态。epoch FFT非局部，短窗在线输出必须从允许时间的
  raw重建，不能裁已经使用较晚数据的cache冒充早可用。
- normalization、QC、ICA/EOG回归、EA、早停、特征选择和校准/停止阈值仅在训练或
  声明calibration内fit。EOG清理用成熟工具并记fit范围，缺独立EOG不称完成眼电归因。
- 时序选epoch后refit必须显式声明`prefix_mask`；内层train/validation须属于该校准范围，
  只有范围内的embargo行可回到refit。统计、BN、优化和候选metadata一起按该范围切片。
  两个正式refit入口继续只传已切出的校准X，范围与实际refit行数写入history。
- 内部配方选择还要求学习型预处理仅在inner train拟合。当前fold-local QC仅支持outer
  fit生命周期，不能与target_time_split内部选轮数组合；入口对此明确拒绝。已有清理cache
  上进行内部选择前也必须核对EOG/EA等fit来源，不能把outer隔离当作inner隔离。
- AS-MMD初始拟合和refit共用源上下文与联合损失；合并单样本尾批而不丢校准行。
  启用时batch_size至少2，两阶段实际loss分别记录，不能refit时静默退回普通CE。
- BrainSync允许录制结束后确认目标；校准就绪时刻为`max(session_end, target_confirmed)`，
  后续测试不能早于校准EEG与标签就绪。显式时间列缺失的旧cache要求重建，不推测时间。
  显式声明的离线`independent_recording_holdout`以完整独立会话定义训练/测试，不以录制先后划分；
  仍要求人员、输入合同一致、会话/事件不重叠及所有拟合仅在校准范围。2026-09-08用户指定
  第二+第四次校准、第三/第五次分别测试采用此协议，不混称工作站后续新采集验收。
- 个体模型绑定participant、校准session/数据摘要、输入/参考/通道、EOG状态和模拟声明。
  无标签推理不读target字段；后补真值不改变预测。固定LDA至少1份两类session，源NN内部
  选方至少2份；自动EOG在NN内训练session拟合后冻结，不用验证段估计清理系数。
- reference/电极身份可追溯。不同native参考不伪装一致，CAR是可能损失任务信号的对照。
  共同五导是研究桥接，不宣传完整七导产品性能。
- BI候选冲突不阻塞binary学习；F3/F5歧义不影响已确定五导，相关空间解释另行裁决。
- BI2014a 永久只准入 01-61；62/64 的视力矫正不全、63 的干电极接触不良，以及作者
  明确禁止用于 P300 分析的说明已核实，见[作者依据与删除记录](evidence/bi61_author_exclusion_20260906.zh.md)。
  三人的 raw 和活动缓存行已删除，源/目标/验证及旧 checkpoint 复用均受同一身份门禁限制。
  不能从含这些人的模型删除 ledger 后冒充合规，必须重训；历史64人结果保留原身份。
  不因新模型成绩进一步删其他被试；缺 candidate 数据仍可作 binary source。

## 5. 训练性能与证据

复用DeepBaseline、GpuPerformanceScheduler、源批处理、SubjectAdapter和cache attestation。
GPU沿用验证过的BF16/compile/fused，记录实际fallback、OOM、batch、updates；不能为
提高GPU利用率静默改变科学比较。小样本fit先测开销再选CPU/GPU/并发；按cgroup配额，
不按宿主208核分线程。source训练、target fit、结果分析分别记录资源与耗时。

cache按digest复用、每进程尽量一次解压；任务按日志/checkpoint恢复。物理source archive
对应实际文件，dirty工作树记录父commit和逐文件hash，不把父commit当全部当前源码。
participant/seed/session不确定性分开；固定模型bootstrap不是训练程序总体推断。
多轮选择后的dev不称confirmation。SESOI、非劣界、主要contrast和确认计划先声明；
CI含0只是未决，实用等效需要适当检验与精度。

## 6. 反例与联动

本轮训练只执行[compact512基线与组件消融](roadmap.zh.md#2026-09-07训练链条验证方案)：
固定batch上限512、60遍历，复用四个排他source；不新增源训练、不搜索batch或产品选择器。
batch512是用户指定的资源策略，不是B0选出的最优值。三个消融只估计相对Source full的
条件贡献，不证明完整因素交互、多源独立优势、源选模最优或成人效果。
E1完整因素矩阵、S1/E2/S2和E3软件能力保留，本轮不排程；候选能力不等于候选已经获胜。
E3保留旧产品、仅迁移、无迁移与五候选四个程序，部署默认不因实现完成自动切换。
新选择器的低/高训练预算固定，inner模型丢弃后从原初始化完整refit；实际拟合统计随模型保存。
缺少合法inner记录只允许预声明LDA fallback，并标记没有选模证据。公共AUC与BrainSync
完整45闪烁block的operational hit@R5分别评分；测试逐人oracle不是可部署收益。

来源身份中的原始stream ID、实验namespace及真实采集库是不同轴。来源库/家族排除须由
可追溯的实际归属合同证明，不能直接把按被试编码的stream当作独立库。候选失败保持固定
比较队列，预检排除保留原请求分母；通过0.01下降检查只表示没有触发该工作门槛，
不证明普遍替换或个体保证。

每轮：理念→概念→公式→反例→完整代码链→测试/文档/产物。
重点反例：真实builder身份隔离、metadata-only candidate投影、不同K共同suffix/purge、
source/scratch完整权重与BN来源、EOG不进入EEG张量/CAR/坐标、7EEG+1EOG的BIDS/EDF
顺序、截止后脉冲、合法candidate置换与不完整coverage、预算/seed/分母不静默变化。
改变的合同和测试同步，删除被证伪的旧语义断言，复用仍成立的不变量；不以旧全量测试
替代新科研验证。重大变化前说明理由，之后更新计划状态和证据边界。

## 7. 历史与执行

2026-09-06补充：[工作树清理与数据保留](evidence/workspace_cleanup_20260906.zh.md)记录历史
`tmp/`全部16,389个文件的去向。四份cache及配套记录进入`experiments/cache/history/`，
368份结果/日志进入`experiments/results_history/`，其余物理压缩；原结果不改写。
历史脚本不能作为当前训练入口，历史数据重新使用仍须通过现行合同与划分核验。

被替代指导的原文、哈希与当前去向集中于[证据索引](evidence/README.md)。
dated结果按写作时口径解释，不执行过时默认机制/R8门禁；训练/结果保持原身份。
当前未完成项统一由执行计划维护，不因历史自主运行授权自动重启旧矩阵。

已经完成的额外真实性对照使用BNCI可靠的物理target pair：测试目标未曾作为本人
校准目标，但刺激本身可见过；BI候选冲突未解不作该主张。该对照保留独立科学意义，
不要求产品部署排除所有校准数字。
原微调复核的混合均值、性能反证及refit风险更正见
[校准范围守卫复核](evidence/finetune_leakage_review_20260906.zh.md)。
严格公开对照已完成，见[BNCI未见校准目标结果](evidence/bnci_unseen_target_results_20260906.zh.md)：
同一200个后续decision、8人3seed；校准内部选择配方的600trial AUC为0.8136。
它仍是已观察公共数据的development，不能替代真实BrainSync确认或P300成分归因。
