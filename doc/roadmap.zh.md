# 当前执行计划

更新：2026-09-06。本文件统一当前阶段、实验优先级与验收条件。
研究约束见[研究规范](research_program.zh.md)，模块接口见[技术结构](technical_structure.zh.md)。
日期化设计/运行记录保留在[证据索引](evidence/README.md)，不自动触发旧云端任务。

2026-09-06 最新准入：已核实作者报告明确拒收 BI2014a 62-64，后续源/目标/验证仅用
01-61。raw 和活动缓存行已删除，当前主 cache 为 57,437 epoch。下面已完成的 BI64
结果仅保留历史身份；所有后续实验必须绑定新 eligible61 cache 和合规 source checkpoint，
含62-64的旧BI source必须重新训练。详见[作者依据与删除记录](evidence/bi61_author_exclusion_20260906.zh.md)。

## 已完成与未完成

2026-09-07四库主线已云端启动，见[冻结配方与运行目录](evidence/four_study_launch_20260907.zh.md)。
BI2014a61人、BI2015a41人、BNCI8人、Huebner13人已生成四个整库排他source视图。
用户缩减为单seed=20260906；四折source及123人六臂目标评估全部完成，2214单元无失败。
目标评估实测4worker相对2worker快1.84倍，采用4worker；
保留第一折已生成的结果。严格CUDA数值确定性尚未启用，见运行记录中的重复预测差异。
Mainsah参考及全部288份EDF/BDF转换/角色映射已核验，见
[准入审计](evidence/mainsah_s2_admission_20260907.md)，仍封存未评分。
下文历史待办中的参考/转换阻塞由本次证据解除；外部协议冻结与方法冻结仍需完成。

| 工作 | 状态与证据边界 |
|---|---|
| binary源视图、监督预训练、源head保留与目标校准 | 已有真实公共数据开发实验 |
| BI/BNCI配方、强scratch/LDA、同session与ERP聚合诊断 | 已完成相应开发矩阵；不是全部E0/E1已完成 |
| BI64通道与selection/trial配对复核 | 已完成63→1源LOSO加目标自身校准对照 |
| BNCI未见校准目标、校准内部选择与refit守卫 | 已完成严格开发对照；不称成人确认 |
| BrainSync采集、源模型校准、EOG、模型重载、45闪烁无标签推理 | 模拟Qt端到端已验证；逐SDK chunk推理未实现 |
| 阻抗显示来源区分 | 模拟值不再认证真实接触；实机测量问题仍待核实 |
| 历史数据与结果整理、指导整合 | 已归档并逐文件校验，原数据和结果保留 |
| 跨库扩源、真实设备与成人确认 | 下面阶段继续推进，尚未完成 |

## 下一步顺序

### 本轮执行：已有BI与BNCI跨库训练

2026-09-06已启动`experiments/runs/cross_dataset_start_20260906/`：BI64全部61,011 epoch
与BNCI8全部33,600 epoch分别训练source，对方整库留出，先一个seed=20260906。
两库共同五导CZ/P3/PZ/P4/OZ、native right-earlobe参考，K48/96/192共享purged test。
六核心臂用轻配方；额外完整报告classifier/full的强配方，不按test选择赢家。
主比较预声明K96轻full-fine减合理target-only，结果属于公开开发，不能代表完整七导产品。

当前状态：两个源模型、1,728首轮单元和432个前缀统计/BN归因单元均完成，2,160份预测
全部重算核验。K192前缀统计+更新BN的AUC为BI→BNCI 0.6598、BNCI→BI 0.7067；
相对强LDA尚无稳定优势。设计、完整对照与CPU/GPU计时见[本轮结果](evidence/cross_dataset_start_20260906.zh.md)。
实际source与evaluation从同一物理
快照B执行：`frozen/cross_dataset_start_20260906b.manifest.json`，逐文件比对通过。
快照A的预检误将BI按被试编码的dataset_id计为多库，在训练前失败；B依据保留的
原始数据集合同与逐被试stream映射核验，不改cache、不猜全局身份。
两库未引入目标统计的source训练与本人成本明确分开；三seed、扩源与成人确认仍未完成。

当前源模型仅1,186参数，完整source refit为21/25epoch。下一步先在固定source划分下
比较MST每分支2/8通道×最大30/120epoch，记录实际选轮及训练预算；未验证容量和训练
充分性前，不以本轮负迁移否定源表征。目标CUDA单任务实测约快6.4%，仍需并发吞吐对照，
后端选择与随机性固定入配方，不将CPU/CUDA结果静默混用。

### LDA顺序配对实验

按用户最新指令，先执行`experiments/runs/lda_head_order_20260906/`：复用已训练的两个
跨库source，在72名目标人、K48/96/192上对比原微调线性头、微调后收缩LDA头、
LDA初始化头再微调。前两臂共享同一份微调权重及BN状态，第三臂仅初始head不同。
预声明K96的两项相对原head差，60epoch/lr1e-3/前缀统计/更新BN，统一CUDA，
每人每K两次fit、三个结果，共216对/648份预测，已全部完成并核验，墙钟239.68秒。
所有LDA的scaler/协方差只拟合校准行；未做内部选配方，不使用test选择赢家。
源容量×预算对照仍待执行，不把LDA对照当作容量充分性验证。

结果见[顺序对照](evidence/lda_head_order_20260906.zh.md)：K96后置LDA较原head的AUC差
为BI→BNCI +0.0025、BNCI→BI +0.0068；LDA初始化再微调分别-0.0071、-0.0108。
后置LDA保留为候选，先验证多seed与校准内选择；初始化路线不作为默认继续叠加。
新增两个可复用接口及global-average形状修复已通过93项相关测试，产品默认模型未更换。

用户随后要求提高K。旧BI64大预算任务因新准入规则中断，完成记录48/83组，
只作历史保留，见`frozen/lda_head_budget_20260906.interrupted.json`，不恢复旧source。
当前使用已验证的`bi2014a_binary_common5_20260906_eligible61.npz`（57,437行）重启于
`experiments/runs/lda_head_bi61_20260906/`。BI source从头按原30epoch上限/seed/选轮规则训练，
实际refit8epoch；BNCI source身份无BI而复用。新快照包括准入门禁。
BNCI全8人用K192/600/1200/2400，BI按时间/长度合格的15人用K192/600/1000，
均保留至少200条共同测试行，共77组/231份预测，已全部完成、重算与模型状态核验通过。
不拼接前一轮64人的旧后缀或中断结果；60epoch下大K也增加updates，
这回答实际校准/训练剂量收益，不是仅增加独立标签的纯因果效应。

新61人结果：BNCI目标原head在K192/600/1200/2400为0.6575/0.7348/0.7679/0.8040；
BI15人长记录原head在K192/600/1000为0.6608/0.7641/0.7996。
最大K后置LDA分别0.7989/0.8035，未显示跨方向稳定优势，不设默认；
详见[完整大预算结果](evidence/lda_head_order_20260906.zh.md)。当前优先保持原head基线，
下一项验证同预算target-only/LDA、多seed及源容量/训练预算，不外推成人或K4050成绩。

2026-09-07后验hit重算：同一120个BNCI完整后续decision，K2400时原head的36选一
R5/R10为61.67%/81.67%，后置LDA为64.17%/84.17%，差异CI均含0。
这里R5消耗60次flash，不能当作BrainSync45次flash九选一；实际产品hit仍待采集验证。

### 扩源与产品

| 阶段 | 动作 | 验收与继续条件 |
|---|---|---|
| A 实机接入 | 短录制核对七EEG/EOG、参考、接触变化、事件与样本时钟、丢包及BIDS | 真实SDK回调与raw replay一致，completed记录及标签就绪时间合法 |
| B 新源准入 | BI2015a/Huebner已核验并建cache；Mainsah完整转换、参考、Train/Test映射已核验 | Mainsah保持封存；外部模型评估前冻结预算及条件组成 |
| C cache与划分 | 按当前合同构建binary cache，产品mask/Cz差分视图，多run校准与共同test | 原始顺序可追溯、父cache与行身份一致；source排除后仍满足通道覆盖 |
| D 最小接口 | 整库排他、显式类别权重/updates、多run边界、失败账本 | 真实builder反例和CLI通过，训练与评估snapshot分别绑定 |
| E 公共开发 | 先BI↔BNCI复现，再合法四库单源/多源与六个核心对照 | 先一seed诊断，再三seed复核；配对预测、预算、失败和负迁移完整 |
| F 增量裁定 | 等更新的源剂量/多样性，必要时窗口/伪迹和已有机制增量 | 每次只解决有证据的剩余缺陷；无附加收益则保留plain候选 |
| G 外部与成人 | 冻结方法后Mainsah评估；成人后续session与跨日九选一 | 外部数据准入与未用于选方可核查；真实标签/时间成本与请求分母完整 |

A与B可独立推进。单库未准入时继续其余合法工作，但明确缺失的折；不能将缩小的数据池
称为完整四库结果。没有正式成人数据时，软件/公共开发成果保持各自证据等级。

## 跨库设计与最小矩阵

准入后的开发池为BI2014a、BI2015a、BNCI2014-008、Huebner2017。
每次完整留出一库作伪目标，其余库source训练；目标库任何人、session、统计、teacher或
早停信息都不得进入source。另做BI2014a与BI2015a同时留出的整族压力测试。
Huebner的两种分发不得重复计样本；同源改namespace不产生新的participant或独立域。

Mainsah S2的转换、参考及逐文件Train/Test映射已通过2026-09-07审计，保留为封存外部库。
元数据审计未读取模型成绩；仍需声明历史使用记录，不能自动声称前瞻封存。
方法由开发池决定后，source在开发池refit，再评估Mainsah。若之后将Mainsah纳入全公开源
训练，所得部署候选是另一个checkpoint，不能继承“对Mainsah未见”的结果。

七导布局只映射实测通道，缺导保持mask和零占位；outer/inner排除后检查训练行观测覆盖。
跨库优先核验共同实测Cz参考，输出六差分；前提是原EEG共享参考且上游变换兼容。
未知参考不自动视为Cz。七导native、六差分及已有五导桥接分开比较，不能更名混用。

| 核心臂 | 初始化与目标fit |
|---|---|
| target-only shrinkage LDA | scaler和协方差只拟合校准 |
| target-only N2P3Net | 随机权重及BN，校准统计，合理训练预算 |
| source zero-shot | source权重和统计，不拟合目标 |
| source classifier_fine | 保留源head，冻结trunk |
| source full_fine | 从源初始化，全网仅用校准更新 |
| source feature + shrinkage LDA | 源特征冻结，head只用校准拟合 |

公共binary预算先用K=48/96/192，固定最大K之后且无epoch重叠的共同test。
主比较为K96的transfer减scratch，同时比较LDA，完整报告其他K。K只数闪烁，
各库另报正例数、校准秒数和run数；它不是4050次产品校准预算的替代。
多run优先遵循真实Train/Test与时间顺序，不能按字符串排序或拼接重置的时钟。

source起点沿用batch512、Adam lr1e-3、weight_decay1e-4、最多30epoch、
source内participant分组早停patience6，seed=20260906/07/08。
这些是设计起点，不是已验证跨库最优值。现有正类权重8先复现，再在source开发中比较1；
类别权重、采样、域/participant权重及每步归约必须一起声明。

目标轻/强配方为15epoch/lr1e-4与60epoch/lr1e-3；normalization和BN作为显式因素。
可仅在校准内部选择配方，然后从源初始化在完整校准refit。无法合法内层划分时，使用
开发期预先冻结的fallback，不能看test选方或补标签。scratch有同等开发调优机会。

按用户最新要求，本轮四库×单seed×六臂×三K对应72个汇总组合，实际拟合量还取决于participant/run；
zero-shot和确定性LDA可复用，组合数不等于独立样本数。公开主读数为participant-macro
AUC再库等权，辅报BACC/AP、负迁移、低分位、失败、概率质量与计算成本。
不以CI含0宣判等效，开发期0.01 AUC工作阈值不充当确认性SESOI。

## 接口与反例验收

| 所有权 | 必要增量或不得越过的边界 |
|---|---|
| `data.raw_artifacts/manifest/identity/lineage` | 实际信号+events映射、转换来源、同源去重与真实participant |
| `data.domain`与`prepare_multidomain_source.py` | binary投影、显式参考、缺导mask、每折source覆盖，复用原模块 |
| `run_pretrain_supervised.py`与`baselines.deep` | 显式class weight与固定updates；域/participant权重和validation归约匹配 |
| `transfer.checkpoint`与`research.evaluation` | 整study/家族排他进入真实合同，不能只查当前目标人 |
| `transfer.binary`与其runner | 多recording有序C/T清单，不解除单stream守卫后直接拼接 |
| `SubjectAdapter`与`transfer.calibrated` | 保留prefix守卫及inner-fit范围，复用已有session内部选方 |

必须覆盖：目标库另一人混入source；同源namespace改名；排除后唯一Fz/PO8来源消失；
缺导经参考变换产生假观测；标签码跨数据源误映射；run-10/run-2和时钟重置；
改变later test振幅/标签却影响fitted状态；K变化改变test队列；microbatch改变loss归约；
scratch偷用source统计；无两类校准或拟合失败后从test补数据。
具体公式和更详细接口表见[跨库设计](evidence/architecture_decision_cross_dataset_20260906.zh.md)、
[实施规格](evidence/implementation_plan_cross_dataset_20260906.zh.md)与
[数学反例](evidence/math_verification_cross_dataset_20260906.zh.md)，当前进度只在本文件维护。

## 后续候选与成人验收

尚缺作者准入61人16导within-session E0-A、成对去额/ICA完整E1、EOG生理归因、
固定source更新数的剂量对照、窗口/滤波扫描、BI物理membership冲突恢复及真实动态停止。
这些任务回答不同问题，BI候选恢复不阻塞binary学习；先按对主线的信息价值安排。
窗口600/800/1000/1200ms需按允许截止从raw重建，不能裁FFT cache声称提前可用。

EA、Split-BN、AS-MMD、源选择与全导teacher蒸馏各自对plain做附加比较，
匹配source人数/draws/updates与目标预算，teacher成本另报；有稳定证据后再考虑组合。
完整[72项候选目录](evidence/research_experiment_catalog_20260906.zh.md)保留为资料，不是待全跑清单。

成人阶段固定独立后续decision，比较源Z0/head/full、合理scratch和LDA，
并配对无清理/EOG回归、native七导/Cz六差分。主指标是研究规范规定的operational
九选一`hit@R5`、correct selections/min及校准成本。正式样本量、SESOI和跨日方案
结合真实先导方差冻结，不能用公共AUC或模拟命中估计成人准确率。
