# 当前执行计划

2026-09-09工频陷波：BrainSync新校准/基础报告默认50Hz、Q30；共享滤波、恢复证书、缓存与模型门禁已联动。
103053真实记录P4/PO8的49–51Hz RMS从原带通后的0.987/0.565μV降到0.066/0.063μV。
软件与信号验证见[本轮证据](evidence/brainsync_notch_20260909.zh.md)；未重训公共源/本人模型，未测识别率收益。

更新：2026-09-07。本文件统一当前阶段、实验优先级与验收条件。
研究约束见[研究规范](research_program.zh.md)，模块接口见[技术结构](technical_structure.zh.md)。
日期化设计/运行记录保留在[证据索引](evidence/README.md)，不自动触发旧云端任务。

2026-09-06 最新准入：已核实作者报告明确拒收 BI2014a 62-64，后续源/目标/验证仅用
01-61。raw 和活动缓存行已删除，当前主 cache 为 57,437 epoch。下面已完成的 BI64
结果仅保留历史身份；所有后续实验必须绑定新 eligible61 cache 和合规 source checkpoint，
含62-64的旧BI source必须重新训练。详见[作者依据与删除记录](evidence/bi61_author_exclusion_20260906.zh.md)。

## 已完成与未完成

2026-09-08工作站校准入口修复完成：QC不再使不兼容源静默降级为LDA；固定LDA的全校准
EOG/冻结QC与dry-run范围一致，不进入内部选模。计算设备设置已接通，当前四份规范记录
及随包源的默认配置预检查通过；旧记录缺少EOG接法，本机配置明确关闭自动EOG。
后端全套1539通过、4跳过，前端35项通过，Ruff与差异空白检查通过；Qt设置窗口布局已核对。
RTX5070 Laptop上的合成源微调、保存重载和CLI推理通过，CPU/GPU最大logit差2.67e-6，
推理前后个体包字节不变。本轮未训练真人数据或启动云任务，不新增成人性能证据。

2026-09-08[全库代码质量核查](evidence/code_quality_audit_20260908.zh.md#修复落实)中的七项问题
已在现有模块修复，新增反例覆盖进程副作用、模型输入变换、失败生命周期和产物完整性。
全套1530通过/4跳过，另完成最终前端定向测试及桌面/手机浏览器核对。新启用CUDA还暴露并修复
Dense PDS距离下溢及相消；这些是软件验证，不增加公开或本人准确率证据。

2026-09-08其余四项前置质量联动已完成：低幅/平直线复核、多尺度慢趋势诊断、首刺激及block尾部
实际样本门禁、clock/BIDS断点交叉校验、恢复界证书和EOG/频带/参考12条件对照。
72拟合/144预测及保存重载全部通过；保持原恢复不确定集时不能安全缩短一个采样，持续漂移不能
仅由趋势大小唯一识别。新EOG保留采集电压模式已接前后端，不自动替换默认频带/参考，见
[科研判断与结果](evidence/brainsync_qc_remaining_20260908.zh.md)。

2026-09-08已按[QC裁决](evidence/brainsync_qc_redesign_20260908.zh.md)分开幅度分布诊断与
明确伪迹/不可用证据，完成文献核验、29项独立反例、前后端全生命周期及原始恒值 `/4` 门禁。
第二+第四校准、第三/第五独立测试的三臂云端57拟合/114预测全部完成并重载复核，0失败。
第四份Pz从旧冻结规则30个恢复至367个；四新源共同支持AUC第三份0.4677→0.5235、
第五份0.4409→0.4957，但第五份未胜过旧规则重拟合的0.5089。严格R5尚无命中，慢漂移未解决。
保留率与开发AUC不构成成人确认或部署晋升；旧结果保持原始QC身份。

2026-09-08校准/推理界面完成日常流程精简：研究设置独立，训练完成直接呈现内部验证、
训练曲线、数据采用量和候选比较；猜数字展示九候选证据、R1-R5和拒判。
模型重载恢复报告，刷新保留勾选，推理记录按模型身份和时间范围筛选。
Qt窗口已核对1440×900与1160×720布局；属于软件交互验证，不新增成人效能证据。

2026-09-08已完成左耳耳垂耳夹REF合同更正及前后端联动，9个活动目录/24份元数据更正，
18份EDF摘要不变。四份自采记录分别生成原生七导和Cz六导缓存及独立质量模型；
已复算的三个Cz缓存波形、标签、mask、准入与旧版本相同，没有改变旧成绩或重训网络。
原生七导是本人模型的信息保留基准，Cz仍为既有跨源checkpoint的显式桥接。
尚未完成BrainSync原生七导/Cz六导的配对性能实验，不按保留率宣称准确率提升。
见[参考裁决与验证](evidence/brainsync_reference_20260908.zh.md)。

2026-09-08按用户指定完成BrainSync独立会话留出：合并第二、第四次校准，第三、第五次分别测试。
四库新S1源、旧BNCI源与匹配scratch/LDA共19拟合、38预测已复核；675个NN训练epoch，
新源未稳定优于scratch，严格R5无完整block。详见[本次结果](evidence/brainsync_s1_calibration_20260908.zh.md)。
该划分不以录制先后定义测试；独立会话、事件及全部拟合范围仍严格隔离。

2026-09-07最新执行规则：后续校准比较统一K=0/200/800/2000；不足时按约80%校准、
20%后续测试并执行purge，记录请求K与实际K。原独立Train/Test不改变，
规则与解释边界见[研究规范](research_program.zh.md#统一校准档位2026-09-07用户规定)。
下面K48/96/192数值属于已完成历史实验，不再作为下一轮默认预算。

按新矩阵的重跑已于2026-09-07完成，123人1648单元全部成功且逐预测复算通过：
`/root/autodl-tmp/n2p3-runs/four_study_development_20260907c`，
直接使用修改后的`experiments/runs/four_dataset_20260907/development_plan.json`。
单seed、K0/200/800/2000与80%不足回退，每个实际正K带同校准/test行的scratch和LDA。
四个已核验source checkpoint复用b版，目标评估代码与配置冻结在c版，4worker×1线程。
评估快照SHA256为`e9b7beaa938543c5b4b487c603ea73bbd4f6f533a75d3f96e32766e0a23706ac`。
新结果的`paired_transfer_effects`记录逐人同实际K迁移减scratch差值，失败配对保留。
因为最高K改变，共同test也会重新划定；不与旧小K矩阵混算配对增益。
见[标准预算完整分析](evidence/four_study_standard_budgets_20260907.zh.md)：真实K200/K800/K2000
分别123/59/21人。前两档四库等权全网减scratch为+.0269/+.0107；Huebner K2000为-.0033。
回退K不重标标准档，当前固定轮数结果不能证明早停有效。

2026-09-07四库主线已云端启动，见[冻结配方与运行目录](evidence/four_study_launch_20260907.zh.md)。
BI2014a61人、BI2015a41人、BNCI8人、Huebner13人已生成四个整库排他source视图。
用户缩减为单seed=20260906；四折source及123人六臂目标评估全部完成，2214单元无失败。
结果已从预测逐项复算，见[三档预算完整分析](evidence/four_study_results_20260907.zh.md)：
四库等权full_fine AUC为0.625/0.671/0.724，相对scratch增益0.048/0.047/0.028。
零校准等权AUC0.576；本轮支持有校准迁移的平均收益，不构成可靠零校准或成人九选一证据。
K96没有最优性依据，三档并列报告；产品4050闪烁预算另行验证。
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

### 2026-09-07 目标早停消融

沿用[先前阶段判断](evidence/four_study_standard_budgets_20260907.zh.md)：固定已训练source，
优先检验目标全网微调、头微调和同K scratch；K0与闭式LDA不做早停。高预算迁移作为工作基准，
不把四库平均读数当成逐人最优或早停有效证据。

公共入口新增`--stage ES`，每个正K为三类NN×三策略及一个LDA，共10个结果；K0仅源zero-shot。
默认上限60遍、lr1e-3、batch上限512、source/target seed20260906，其他recipe保持配对。
full/scratch使用校准输入统计、在线BN和可训练affine；head仅分类头更新，源BN buffer/affine冻结。

| 策略 | 内层 | 完整校准拟合 |
|---|---|---|
| fixed | 不选轮 | 原初始化固定60遍 |
| full_trajectory | 在授权校准内按时序/purge划分，跑满60遍，按validation CE选轮 | 丢弃inner状态，从原初始化全K重训选中轮数 |
| patience | 同一内层划分，连续6轮无超过1e-6的CE改善时实际停止 | 同样从原初始化全K重训选中轮数 |

两种选轮策略使用相同最小改善定义和较早轮数tie规则；最终test完全不进入选轮。多记录按真实时间
保留末尾约20%的完整记录，单记录按80/20并purge；无合法inner划分时执行同一固定高预算recipe，
显式记录`selection_evidence=false`与fallback原因，不从test补数据。
先比较full_trajectory与fixed（选轮策略连同验证/重训的整体效果），再比较patience与full_trajectory
（提前终止的附加效果）；不能把fixed与patience的差归因于单纯少跑几轮。

记录实际inner/选中/refit轮数、完整CE曲线、训练行摘要、总updates/draws、物理updates/draws、
两阶段总墙钟及复用。相同完整校准refit按完整键复用，inner完整轨迹与patience分别实际执行。
由此不能用cache命中时间证明某策略独立运行更快；预算上界包括inner与refit，可能超过固定60遍。
`analyze_training_chain.py --stage ES`提供15个配对对比、负迁移/回退分母及成本汇总。

软件合同、CLI/dry-run与四库每库前两名声明participant的GPU试跑已完成；选择名单不使用成绩，
固定已有六导4000-update基准源模型。用户明确要求验证后自动执行全量，小试只是启动前检查。
不重训source、不重新展开S1/E2/S2，不触及Mainsah、额外seed或成人/设备确认。
当前实现及8人GPU试跑完成，218/218评分成功且逐预测/预算复核，源模型未重新拟合。
全套1208通过、4跳过；完整四库dry-run为123人、305正K、3173评分、最多4880次逻辑fit。
试跑中全网patience含refit的总更新反而增加41.5%，未显示准确率收益，固定高预算仍保留；
全量已于2026-09-08 00:01完成，目录`/root/n2p3-runs/early_stopping_full512_20260907/`，
原控制器PID31745，8个CUDA进程，续跑墙钟393.96秒。
复用同一已验证源码快照及8人218评分，续跑余下115人2955评分，合计123人3173评分。
已有结果逐成员SHA、配方、行、seed和评价快照核对后并入最终分析；不重训已完成单元。
3173个评分、3050个保存模型、选轮/预算及配对区间均重算通过，0失败。
全网patience相对fixed的AUC差为K200 -0.0219、K800 -0.0243、K2000 -0.0150；
全网/头/scratch含refit总更新分别增加31.5%/29.6%/9.0%。继续固定60遍，ES保留实验选项，
不将当前CE早停设为默认。见[全量结果](evidence/target_early_stopping_results_20260908.zh.md)，
原实现过程保留于[实施记录](evidence/target_early_stopping_20260907.zh.md)。

### 2026-09-08 固定目标90遍消融

同一123人、305个实际正K，保留上述60遍full/scratch及LDA，新增full90/scratch90共610拟合。
固定既有六导4000-update source、seed20260906、lr1e-3、batch上限512及全部校准/test划分，
只增加目标轮数。新训练96,660更新，无inner选轮、无新source训练。
从原初始化重训90遍，旧60遍模型不具备完整optimizer/RNG继续训练状态，不作不等价续训。
完整dry-run后已自动执行610/610，0失败；1525份新旧预测及模型复核通过。
训练197.84秒，含审计236.89秒。全网90遍AUC为0.71445/0.77698/0.83470，较60遍提高
0.00861/0.00964/0.00135；K2000只有21人/2库，增益区间包含0。Scratch改善更大，迁移差距缩小。
详见[声明与结果](evidence/target_epoch90_20260908.zh.md)。保留90遍为开发高预算对照，
60遍为历史参考，产品默认未改；不按本次test逐人或逐K选择轮数。

### 2026-09-07 P001 本人数据训练

后续质量核查：用户拒绝`0903_170721`，原始/准备记录已移出活动目录，含该会话的旧缓存、
个体模型及结果包已物理压缩校验并隔离。永久删除被自动审批策略拒绝，不称已永久删除。
第三、第五会话原始记录保留，局部污染的隔离/恢复现按下一节开发验证；
不能沿用旧五会话结果或模型，见[质量诊断](evidence/brainsync_transient_quality_20260907.zh.md)。

以下记录撤回前的历史执行范围：
用户确认 P001 文件夹属于同一人后，5 份完整记录经显式身份、EOG标签与坐标迁移及缺失
BIDS sidecar重建，通过严格输入门禁；3份不完整录制移出活动目录并有逐文件校验归档。
云端RTX5090已完成16臂比较和4个全数据个体包：前四session校准1764 trial，最新session
独立测试441 trial，最终全数据refit2205 trial。内部预选方案后续AUC0.4926，严格R5为0/10，
其中9个block因休息后首trial基线跨入休息而拒判，仅第一个block完整；此前末次flash缺尾部的
文字解释已按事件/缓存复核更正。不能据此认定实用迁移收益或成人确认完成。
执行、模型、性能与完整分母见[P001证据](evidence/brainsync_p001_pilot_20260907.zh.md)。

后续[连续质量研究方案](evidence/brainsync_signal_quality_design_20260907.zh.md)已完成三路文献
核查及10组反例；P0样本账本/双侧guard、P1连续区间/状态恢复接口、P2单HEOG和个人分组QC、
P3保存重载及完整分母已实施并作软件验证，见[实施记录](evidence/brainsync_quality_implementation_20260907.zh.md)。
实际前端排程经后端得到10/10组各45个完整epoch。恢复上界、误报率、波形保真与实机光学
时钟仍需验证，不自动启用新阈值。第三/第五份仍只作开发，新采集承担独立验证。

### 2026-09-08局部恢复与补充训练

在原窗口/滤波/网络下，连续逐导QC、参考依赖和经验状态恢复已联动；额外1.4秒异常前沿保护
挡住已检测大漂移开始处的漏入。真实R3/R5在连续QC加冻结epoch QC后保留247/450、229/450，
各至少3/6导；完整六导85/59，通道-epoch1199/1102。原休息边界缺口仍在，严格完整R5均0/10。
62个原定注入反例重放后56个新增误差小于1uV，六个小幅缓慢斜坡仍漏检，不能认证总体无污染。

明确冻结第二份epoch QC状态的接口已贯穿补充校准、复用、预检及保存/推理，工作站可导入同一JSON。
本地GPU对照预声明为三处理管线21拟合，以及冻结质量下R2与R2+R3补充训练14拟合。
两组均已完成并独立重算全部35模型/56预测；R5共同229事件上，补入R3使三seed均值AUC
source从0.4750到0.5459、scratch从0.5126到0.5688。抽取数同时增加7.19%，严格R5仍0/10。
确定性GPU的四次额外重放确认两种同QC路径逐位一致；后端1336通过/4跳过，前端16通过。
逐预测、实际训练量和共同支持结果以[本次证据](evidence/brainsync_recovery_20260908.zh.md)为准；
不根据本轮成绩调整质量门限，也不把旧数据开发结果作为成人/新目标确认。

### 2026-09-07训练链条验证方案

#### 后续授权：优化后的扩展全量

用户在compact结果分析与耗时评估后明确要求启动扩展全量。新运行独立执行E1/E3及S1/E2/S2，
source/target各一个seed=20260906，用户指定目标batch上限512；source batch512，S1/S2每臂4000 updates，
E2候选500/1000/2000/4000/8000 updates。沿用四库整库排他和BI家族排除。
E1保留21个正K处理，排除历史batch32；E3保留transfer-only/no-transfer/five-candidates，排除旧产品batch64。
B0搜索/B0_score、额外seed、Mainsah及成人采集不排程，原compact和B0产物保持原身份。

plan使用`include_historical_controls=false`和明确的batch512规则；E1对compact中完整相同的
full/scratch/LDA与K0产物记录跨快照复用来源，核对环境、数据、配方、行/标签和模型状态。
源链默认按CPU实际配额和GPU可用显存自动调度：各外折/布局独立并行，GPU任务共用全局准入信号量，
源轨迹按语义签名用跨进程文件锁登记，实现并行计算及重复训练去重。当前完整矩阵有9个独立准备组。
E2校准和外层评估也受同一GPU容量预算约束；目标worker按cache/被试分片复用scratch/LDA
及相同源状态评分。`--cpu-threads`表示总预算，缺省读取实际配额；可显式覆盖source/target worker数。
来源合同分别生成，命中次数与物理拟合分开记录。详见工程指南的云端训练性能优先流程。
本次是分析后的开发扩展，不把已经观察过的四库变为前瞻确认集。
旧metadata串行运行完成45个源任务、4组选模后因数据盘空间不足退出；已核验产物继续复用。
自动调度版本于2026-09-07 20:40从`/root/n2p3-runs/training_chain_auto_c_20260907/`启动，
控制器PID24311；25核配额、90 GiB容器上限，解析9个source组、8个GPU准入槽和最多8个目标worker。
新结果及校验临时副本使用另一文件系统，batch仍为512。该队列21:48完成，最后一次续跑68分05秒；
95个源任务、37568个源链外层评分及E1/E3全部成功，0失败。45134外层与7050内层预测已逐项复算，
见[完成分析](evidence/training_chain_results_20260907.zh.md)，运行与优化过程仍见
[实施记录](evidence/training_chain_implementation_20260907.zh.md)。旧控制器不得恢复。

本轮开发证据优先支持六导、校准输入统计、在线BN和完整trunk微调；新增源统计/采样和校准效用
选模的额外收益有限。下一轮建议以固定4000 source updates和完整微调为工作基准，保留scratch/LDA，
暂停重复完整S1/E2/S2网格，只针对等剂量多源的小幅信号复核。五候选提供无迁移退路，但未稳定
超过仅迁移选择器。这些是结果后的建议，未更改产品默认值或新增训练排程。

#### 已完成：compact512

此前按用户精简要求运行 **compact512基线与组件消融**。
旧B0控制器及其后续队列已终止，compact云训练1953/1953完成、0失败，耗时326.83秒并通过逐预测复核；
已完成的E0、已有模型和部分B0结果保留原身份，具体成绩见[实施证据](evidence/training_chain_implementation_20260907.zh.md)。
compact当时取消原B0/B0_score、完整E1、S1、E2、S2及E3排程，后续扩展使用上文的新授权与独立队列。
旧方案与替换前的九份文档已物理压缩并逐成员SHA-256校验，见
[归档清单](../frozen/compact512_redesign_20260907/manifest.json)；压缩包仅在本地保存。
扩展seed、Mainsah评分和成人/设备采集仍不在范围内。

#### 本轮问题与固定条件

本轮只回答：相同本人校准预算下，既有公开ERP初始化相对scratch和原波形LDA是否有益；
目标输入统计、BN运行统计更新及学习型trunk适配分别对完整迁移配方贡献多少。
沿用六槽实测Cz差分和已核验的b版四个整库排他checkpoint，不增加source训练或另建布局。

| 项目 | compact512固定规则 |
|---|---|
| 开发队列 | BI2014a 61人、BI2015a 41人、BNCI2014-008 8人、Huebner2017 13人，共123人 |
| source | 复用b版四个排他checkpoint及原source snapshot；目标库任何人、session、统计和选轮信息不得进入source |
| 校准预算 | K=0/200/800/2000；共用生产builder、80%不足回退、实际K去重及purge，共305个实际正K |
| test | 每人由最高实际K一次确定共同later test；原独立Train/Test不改变，不按得分调整行 |
| 输入 | 128Hz、V、0.1–30Hz forward四阶Butterworth、steady-state、[-200,1200)ms、mean-only baseline |
| 布局 | Fz/P3/Pz/P4/PO8/Oz六槽实测Cz差分，保留真实mask；不补导、不切换三槽 |
| NN优化 | AdamW，lr=1e-3、wd=1e-4、60个完整校准遍历；无inner选轮、无额外refit |
| batch | 用户指定上限512，实际为min(512,n_fit)，保留真实尾批，不复制行补齐；不搜索、不按B0选优、不线性缩放lr |
| 随机性 | target seed=20260906；source保留原seed=20260906，同人同实际K的NN使用相同行序列 |
| dropout | 所有NN臂固定checkpoint声明值；scratch使用相同架构和dropout设定 |
| 独立拟合 | 每人每实际K每臂从原checkpoint或独立随机初始化开始，不跨K/臂续训 |
| 来源 | source与evaluation分别绑定实际执行快照；实际batch、updates/draws、参数/统计摘要、失败与耗时写入产物 |

batch512是用户指定的资源与实验范围策略，不是完成B0后选出的最优batch。
同人同实际K的NN臂匹配遍历、updates和draws；跨K同时改变校准量与训练更新数，不能解释为纯样本量效应。
K不足512时使用实际行数作为batch，不以重采样、梯度累积或OOM重试静默改变预算。

#### 最小矩阵

K0仅保留原source zero-shot，每人1个结果。每个实际正K执行以下六臂：

| 臂 | 输入统计 | BN运行buffer | BN affine | 其他学习参数 |
|---|---|---|---|---|
| 原波形LDA | 当前校准范围内的LDA/scaler | 不适用 | 不适用 | 拟合收缩LDA |
| Scratch NN | 当前校准 | online | 可训练、随机初始化 | 随机初始化全网可训练 |
| Source full NN | 当前校准 | online | 可训练、source初始化 | source初始化全网可训练 |
| 去目标输入统计 | source | online | 同Source full | 同Source full |
| 冻结BN buffer | 当前校准 | source冻结 | 同Source full，仍可训练 | 同Source full |
| 冻结学习型trunk | 当前校准 | online | 冻结source affine | 仅classifier可训练 |

后三臂分别与Source full配对。最后一臂在recipe中同时设置`weights=head`和`bn_affine=frozen`，
它消融整个学习型trunk的参数适配；BN buffer仍按相同online策略更新、dropout仍保持同一设定，
因此不称“完全冻结源特征”。冻结BN buffer臂只改变buffer策略，不能顺手冻结gamma/beta。
输入统计臂只改变统计来源；所有统计、BN更新、LDA/scaler都只使用对应授权校准行。

计划共`123 + 305*6 = 1953`个评分：123个K0、1830个正K结果；
其中`305*5 = 1525`次NN拟合、305次LDA拟合、**0次新增source拟合**。
这些是任务数，独立participant仍为123人。旧E0仅复用通过摘要核对的source/划分/来源证据；
旧batch32训练结果不混写为本轮batch512结果。确定性预测的复用须满足完整数据/状态/配置digest一致。

#### 本轮取消的成本与解释边界

旧S1为`4外折*2布局*4统计/采样配置=32`次source训练；旧S2为
`4外折*(4个全量臂+4个固定剂量臂)+1个家族留出=33`次source训练。
旧E2还要求16条source训练轨迹、80个候选checkpoint、选后从头refit及大量伪目标校准。
这些分别回答源统计/采样、多源独立价值、家族排除和源内选模，当前最小问题不需要同时重训它们。

本轮取消batch搜索与固定updates敏感性、完整I×B×W因素矩阵、一次BN重估、
额外affine/dropout/产品统计/历史batch桥接、三槽源矩阵、多源剂量与家族矩阵及E3选择器benchmark。
相关软件能力继续保留，可在有明确研究问题时另列矩阵；不宣称已被当前结果验证或证明没有价值。
三个消融只能估计相对完整配方的条件贡献，不估计完整交互，不把它们相加当独立总贡献。
本轮不回答多源优于单源、最佳source选模方式、产品选择器收益或成人九选一表现。

#### 执行与验收

先完成compact512共享recipe/CLI/dry-run和相关反例，核对123人/305正K/1953评分及来源账本；
随后从新的物理evaluation快照运行完整四库，保持已授权的RTX5090、4进程、每进程1个CPU线程。
原b版source快照、旧E0与B0产物不重标，新队列单独记录控制器、命令、快照与运行目录。
不得从旧控制器恢复被取消的后续阶段；新队列实际启动后才更新启动状态。

反例至少覆盖：scratch不继承source权重/BN；各臂只改变声明参数组或buffer；
head-only冻结BN affine且保留声明的online buffer/dropout；相同NN行序列与预算；
改动final test不改变任何fitted状态；LDA不提前拟合test；80%回退不重复计K；
来源整库排他、真实身份、缺导与purge边界均通过生产builder验证。
记录实际拟合统计，预测/重载不得更新状态。运行Ruff、相关合同/反例、CLI与`git diff --check`，
广泛代码变更执行全套；软件测试、公开开发与成人/设备确认分别记账。

先按同人同实际K配对，再库内平均、库间等权。精确K200/800/2000和回退实际K分开，
每个标准K明示可估计库数；完整请求、失败和不可估计分母保留，不自动填AUC=0.5。
主比较为Source full减scratch及减LDA；三个消融分别减Source full。
报告AUC/BACC/AP、负迁移率、下降超过0.01比例、p10/最差和逐人结果。
bootstrap在库内按participant配对10000次、analysis_seed=20260907，明确条件于已拟合source和固定四库；
不把任务数/epoch当独立样本，不把反复观察的开发库变成确认集。

ETA以compact512实际完成NN拟合数、真实batch512耗时和尾部分片为依据；
旧B0混有不同batch与固定updates且只拟合inner prefix，不能把其总完成率当新矩阵完成率。
E0的1648份预测与四个b版源权重已复核，未记录的历史batch序列/updates继续标为`not_recorded`。
实现与独立验证见[训练链条实施证据](evidence/training_chain_implementation_20260907.zh.md)。

## 既有开发记录

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

2026-09-08[扩源审计](evidence/dataset_expansion_audit_20260908.zh.md)确认GTN及BI2012/2013a/2014b
具备候选基础：另四BI库139份ZIP官方MD5/大小和CRC通过，GTN真实记录通过六槽缺导适配。
其中BI2015b随后按用户要求删除，释放8.47 GiB，不进入扩源池；BI2014b仅Solo作为后备。
新单人视觉源优先Chailloux2020、EPFLP300与Speier2017，Cattan2019/BrainForm为补充候选，实读见同一审计。
四套所选原始分发共4.22 GiB已全部下载校验；EPFL192份MAT实读通过，Chailloux382份明确电极记录
通过、另2份匿名通道记录列入排除。Cattan舍弃不完整尾采样的边界已明确。
Speier镜像60份有标签原始DAT、483.4 MiB下载与parser核验完成，原站单文件SHA256一致。
BrainForm从24人167份记录恢复校准索引，99953个完整时间窗通过当前预处理，三份记录四个异常导联
写入输入排除清单；仍需逐折QC与runner接入，不将其固定T1校准视作目标切换确认。
Lee ERP原始MAT两会话93.05 GiB；已按用户要求启动连续BDF镜像及元数据38.63 GiB下载，
完整校验状态以扩源审计中的下载账本为准，尚未认证MAT到BDF转换。
当前四库计划不自动扩排；先完成参考/角色/单位和全队列准入，并泛化源数量、
三导完整观测分支及三源固定剂量假设。GTN作为辅助binary源的价值需配对检验，Mainsah继续保留外部角色。

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

公共binary预算统一请求K=0/200/800/2000，固定最高实际K之后且无epoch重叠的共同test。
不足时按约80%可用trial确定实际校准上限并记录回退；原独立测试记录不变。
各档并列比较transfer、scratch和LDA，K0不进行目标拟合。K只数闪烁，
各库另报正例数、校准秒数和run数；它不是4050次产品校准预算的替代。
多run优先遵循真实Train/Test与时间顺序，不能按字符串排序或拼接重置的时钟。

source起点沿用batch512、Adam lr1e-3、weight_decay1e-4、最多30epoch、
source内participant分组早停patience6，本轮按用户要求仅seed=20260906。
这些是设计起点，不是已验证跨库最优值。现有正类权重8先复现，再在source开发中比较1；
类别权重、采样、域/participant权重及每步归约必须一起声明。

目标轻/强配方为15epoch/lr1e-4与60epoch/lr1e-3；normalization和BN作为显式因素。
可仅在校准内部选择配方，然后从源初始化在完整校准refit。无法合法内层划分时，使用
开发期预先冻结的fallback，不能看test选方或补标签。scratch有同等开发调优机会。

已完成历史矩阵四库×单seed×六臂×三K对应72个汇总组合。新默认按K0仅零校准、
正K运行有校准臂，实际拟合量还取决于participant/run及80%回退后的去重档位；
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
