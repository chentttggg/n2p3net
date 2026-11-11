# 校准、温度与整程序选择

本页维护校准与选择的结论和来源。部分详细运行载荷尚未恢复，现存绑定与原地址见[历史索引](history.zh.md#运行载荷缺口)；下列历史读数不代表本次重新训练或重放。

## BrainSync一致EOG与QC重做

2026-10-31用户要求重新拟合两套完整链：basic源预训练→basic本人校准→basic推理，以及
QC源预训练→QC本人校准→同一冻结QC推理。两臂保留相同十二库、原始记录名单和模型结构；
实际EOG才参与回归，无EOG的源明确记为不可用辅助信号，其余QC照常执行。旧矩阵保留原始来源与指标定义。

EOG只进入预处理，保持采集时的EOG电压参考；EEG先确定Cz参考，再使用授权训练/C样本拟合的
中心化OLS系数校正。模型输入仍为Fz/P3/Pz/P4/PO8/Oz六个EEG槽，未测量槽保mask，EOG不进入分类器。
共同信号步骤为50Hz notch、0.1–30Hz四阶双向IIR、[-200,1200)ms/128Hz179点和刺激前均值校正。
QC还复用原始一阶差分median/MAD跳变检测、5秒精确平线检测、按真实依赖传播的通道mask及双向边界状态恢复。
原始差分阈值逐C录制估计后合为包住各记录稳健区间的固定范围；不能将高噪声与低噪声录制混算一个MAD，
以免增加较安静C后反把原C全部拒绝。源单记录使用同一规则的单组情形，拟合与实际阈值保独立摘要。
恢复warmup上限30秒，短记录按授权完整片段长度缩短并保存实际参数，不能借别的记录补足上下文。
EOG与阈值/恢复状态只拟合源训练记录或当前完整C；held信号与标签不参与拟合。OLS也可能去除与EOG相关的真实EEG，
冻结后的线性注入保真不能单独证明生理保真；效能仍由实际配对识别与覆盖检验。

九月五份记录均核到实际EOG槽和左耳垂参考。依据用户“如果也是EOG，那就相同”的确认，
在[交接时的派生声明记录](../assets/handover_20261107/source_inventory.json)绑定
左眼角单端EOG接法；五份原EDF及原sidecar保持不变，派生EDF逐字节哈希一致。
十二库各一份原记录已验证两臂入口；代表记录中只有GTN读取器保留实际EOG。
[交接时的代表输入记录](../assets/handover_20261107/source_inventory.json)
只证明这些实际记录的入口，不能冒充完整源队列或效能结果。

比较展示NN source、NN scratch、xDAWN和source＋波形LDA＋xDAWN等权概率池，LDA是池内组件。
剂量为1/3/5/10份完整录制；目标识别率、R1/R5、覆盖与共同已评分配对遵循[研究规范](../research_program.zh.md)。
basic与QC的预处理方案写入源训练合同、权重和个人包，save/load及当前目标绑定均核一致性，避免旧basic权重接入QC校准。
波形LDA两臂通过同一个完整C fitter和具名recipe保存，避免旧无mask包装器与当前候选包装器混报。
该运行的checkpoint原位已不可用，恢复缺口见[历史索引](history.zh.md#运行载荷缺口)。续接先读[统一状态入口](../README.md#当前状态与续接)，不从本页的旧读数推定任务状态。

用户确认设备启动问题后，本人原始输入改按[启动裁剪合同](../input_contract_math.zh.md)先排除首刺激起4秒并向上取整。
[裁剪接受绑定中的原计划身份](../assets/brainsync_quality/startup_cut_acceptance.json)绑定同一原始记录名单、
新裁剪源与原计划SHA；公共十二库源视图保其原输入。下列旧模型与读数均来自裁剪前输入，
重新比较须在新支持上拟合QC/EOG、准备C/test和训练模型，不能在旧FFT缓存中删几行代替原始采样裁剪。

裁剪后八个预处理单元的实际幅值复核已定位[少量校准下的EOG放大](response_and_noise.zh.md#裁剪后的预处理幅值与eog放大复核)：
C1冻结回归在同两份测试产生228个超过1mV的保留epoch，同支持撤销EOG扣除后为0。
这一诊断要求先解决EOG稳定性并同步源/目标预处理合同，不能把当前QC包直接视为已完成伪迹消除。

2026-11-01本地续接的[交接时记录的阶段结果来源](../assets/handover_20261107/source_inventory.json)
分别列固定未来两份测试与每剂量全部剩余录制；后者随C量改变名单，较早记录明确为回顾性留出。
已完成的16个xDAWN/池内LDA模型及32个原始EDF重放向量保持原producer；新增余集预测只消费冻结模型，
重新计分也核固定测试读数不变。NN source、scratch和三混合在新源预训练完成前明确留空。
fixed测试的xDAWN在C10下为basic R1 `29/100`、R5 `11/20`；QC R1 `21/74`、R5 `10/13`，
QC相应覆盖率为74%和65%。R5较高的条件识别率伴随较低覆盖，不能只据10/13认定QC全面胜出。
完整逐决策结果、共同已评分配对和未评分原因在阶段报告内，不能借旧NN权重补齐新比较。

本次本地外置缓存暴露MOABB BI2012的Windows路径解析缺陷：带点的父目录被误当为`training/online`条件。
`source_parsers`只对BI2012传入POSIX分隔路径；合成点目录反例和同一真实原MAT路径修复前后516个epoch
逐值、标签及事件身份完全一致。新BI2012准备绑定独立执行快照，其余完成源缓存保留原方法身份；
旧失败目录及[交接时记录的数值核验来源](../assets/handover_20261107/source_inventory.json)保留。

源QC若把一份录制的全部epoch拒绝，该录制贡献零训练样本，并在完整原始名单中保留具名处置。
`admit_source_qc_outcomes`仅接收原队列明确的这一类结果：核完整请求名单逐项闭合、重放原冻结数值代码，
再从真实通道支持与双向恢复独立重算“可用epoch数=0”。原始字节、阈值和旧失败账本保持；
hash/解析/普通数值异常及整库无可用样本继续阻止准入。新视图的分片来源、拒绝见证和缓存摘要由独立准入记录绑定，
不把旧失败运行改写成成功。BI2015a的123份请求中120份有EEG、3份无EEG已完成这一核验，见
[紧凑证据绑定](../assets/brainsync_matrix/source_QC_disposition_binding.json)。
源准备队列将完成/失败终态回写本轮checkpoint；全部十二库的两套源视图齐全后，
调用共同的有界内存合并器生成basic/QC池，不在无云端时启动新神经预训练。

完整十二库的[交接时的缓存核验记录](../assets/handover_20261107/source_inventory.json)
已逐文件复算24个源视图的SHA-256，并核六导槽×179点几何。两臂各1963份原始记录，
basic保留1,430,742个epoch，QC保留851,697个epoch；QC的206份记录按上述独立处置产生零训练行，
原始请求仍保留在名单中。这些是预处理保留量，不是识别效能；源池发布与神经训练状态仍由本轮checkpoint维护。

## BrainSync实机的冻结大版与校准剂量矩阵

2026-10-31在用户开启的5090上完成[矩阵接受绑定](../assets/brainsync_matrix/acceptance.json)，
实际来源、模型与验收由[接受绑定](../assets/brainsync_matrix/acceptance.json)维护。
大版指9月20日恢复清单中的十二库主后端与原部署权重；不是工作站默认BNCI源。
原A0Q0、A1Q1分别做零监督校准推理，剂量比较以原A0Q0权重的固定full-fine为神经锚点。
当前专家复用本人模型、xDAWN、连续AR0/AR8与`EnsembleScorer`固定等权池；不包括仍暂停的完整
population/source-response研究续跑。三个seed是同一个冻结源下的目标优化重复。

校准按完整录制分离，预声明1/2/3/5/8/10份、450/900/1350/2250/3600/4500次排程。
每个非零C均包含QC边界恢复的实际拟合记录`P001_20261028_224755_01`，该记录计入剂量；
其后按既定顺序加入旧记录。固定未来测试为10月30日14:27和15:25两份，目标7和9，共20个45闪烁
block；14:45仍按原EDF字节排除。各剂量的全部其余P001记录也在模型冻结后评分，较早余集与
固定未来集分列。P003原身份保持，仅列源冻结推理。共同源输入显式投影为Cz参考六导，Cz缺失的
epoch退出输入并保留排程；raw专家使用同一实测参考下的五导原轴独立block。

下表为固定20请求的命中比例；神经与含神经的池按三目标seed平均。`basic`使用共同基础输入，
`reviewed`应用既有十月QC。旧后端不支持新`/5`，没有放宽旧门禁或将其冒称新QC消费者。

| 校准录制数 | 冻结NN / 当前basic NN | 当前basic NN+LDA+xDAWN | 当前basic再加AR8 | 当前reviewed NN | 当前reviewed四专家 |
|---|---:|---:|---:|---:|---:|
| 1 | 60.0% | 90.0% | 80.0% | 55.0% | 30.0% |
| 2 | 56.7% | 60.0% | 53.3% | 25.0% | 35.0% |
| 3 | 50.0% | 61.7% | 56.7% | 40.0% | 38.3% |
| 5 | 48.3% | 51.7% | 43.3% | 35.0% | 30.0% |
| 8 | 85.0% | 80.0% | 80.0% | 60.0% | 45.0% |
| 10 | 98.3% | 95.0% | 90.0% | 65.0% | 46.7% |

各剂量的可判覆盖分别为basic NN及三专家20/20、basic四专家19/20、reviewed NN15/20、
reviewed四专家11/20；拒判保留在分母。10份时basic NN的三seed平均NLL为0.19468，
三专家为0.66831；1份时分别为1.42588和0.68845。缺概率时完整requested NLL保持不可定义，
条件风险与缺失闪烁诊断另存，未用tiny clip补概率。AR0/8单臂本批大多仅0–4/20，不支持其在本配置下
作为稳定增益来源。零校准时C-only专家无拟合状态，混合保持结构不可用；reviewed零监督校准另消费
既有无标签QC，不与严格无目标统计K0混称。

冻结A0Q0在两份固定测试零校准为5/20；全部13份独立录制为19/130，可判76/130。
原A1Q1相应为2/20、15/130，可判同为76/130。共同输入下18个剂量×seed程序的36份未来logit数组
新旧逐bit相等；360个block另由独立直接求和复核。混合在本批较小C的点估计较好，较大C的NN较好，
没有得到跨剂量稳定支配的模型。剂量同时改变月份和目标组成，不能把曲线变化单独归因于样本量。
只有一人、两份固定测试录制，且数据此前已用于QC/ERP探索；各录制内block相关，三seed不增加独立
样本数。所有臂以整份录制完成为可用时刻，属于回顾性block评分，不是在线单block或纯P300来源确认。
本次点估计不直接晋升默认算法。

一份校准的90%已按[原矩阵接受绑定](../assets/brainsync_matrix/acceptance.json)
只读复核：basic NN、波形LDA、xDAWN分别12/20、14/20、15/20；固定等权候选概率池18/20，
相对NN救回7个、损害1个，三个目标seed的命中组成相同。该校准录制包含450次闪烁、50个target，
NN另持原公共源初始化。实际目标7的第5个block中，NN选2、LDA选3、xDAWN选7；
概率池对7、3、2的归一化分数约0.3559、0.3213、0.2830，仍正确输出7。
两份测试录制各9/10；这解释本批错误互补，不建立稳定90%或独立校准概率的结论。
同剂量reviewed三专家只有6/20、可判11/20，应与basic读数分列。

### reviewed的block覆盖与预测损失

本次四种trial专家路径的basic/reviewed含义及接点见
[QC调用范围清单](../assets/brainsync_matrix/QC_chain_scope.json)。
basic已执行共同数据合同校验、50Hz/Q30陷波、0.1–30Hz四阶双向IIR、原轴有效窗口筛选、
[-200,1200)ms切段、128Hz/179点、刺激前均值基线，以及显式Cz参考六导投影。
无效/休息/填充支持与14:45排除门两路共用；basic没有额外的本批幅度或差分硬拒绝器。
reviewed在此基础上增加已绑定原EDF的通道区间、两份Fz全程屏蔽、5秒精确恒值检测、逐导干净段
滤波重置和经验边界误差门，再将实际X与mask传入C内输入统计/训练、个人包保存重载、预测、
固定专家池及R1/R5完整性记账。混合本身不另行清洗，继承leaf的输入可用性。

该包是`/5 reviewed_only`：仅注册七份十月原EDF，不是任意新记录的自动伪迹分类器。
本次矩阵的九月六份仍直接复用basic缓存；10份个人C实际为五份十月reviewed加五份九月basic，
两份固定测试均是十月reviewed。原公共source预训练没有重跑本批QC；其原输入合同单独保留。
两路本次均未开启ICA/EOG回归或另拟合epoch-QC模型。EOG通道不入分类器不代表头皮EEG内眼电已分离。
差分与范围的通用硬阈值在reviewed-only运行时关闭；500μV规则属于本批离线候选区间编制与复核，
不在未知未来录制重新拟合阈值。经验边界容差是数值不确定性约束，不是纯脑信号认证。

[R5配对分解接受绑定](../assets/brainsync_matrix/block_reviewed_acceptance.json)沿用
四方法、1/3/5/10份校准及原20请求，不重新拟合模型。
10份校准时basic NN source/scratch/xDAWN/三混合分别98.33/83.33/55.00/95.00%，
reviewed分别65.00/60.00/25.00/45.00%。NN可判15/20，经典专家及三混合可判11/20。
当前完整R5要求45次闪烁均有可用分数；缺少一次也会让相应block失去完整候选曝光。
NN允许至少五个输出槽观测，xDAWN/LDA要求拟合时固定所选六导完整观测；三专家池要求所有leaf齐备。

固定basic预测，仅按reviewed的拒判集合扣除，NN source的19.67个正确会先减去5个，
即98.33%降至73.33%；共同可判集合再净少1.67个正确，得到13/20=65%。
三混合是19−9−1=9/20：总50个百分点差距中，45个百分点来自原本正确却被拒判的block，
其余5个百分点来自共同可判集合上的预测净变化。原输出正确/错误/拒判状态各自保留，
不能把reviewed的可判子集13/15=86.67%或9/11=81.82%冒称全部20请求的命中率。

校准程序也不同：NN拟合行4455→4097，xDAWN/LDA因固定六导完整观察约束为4455→3133。
缺导mask、训练归一化/BN及分段滤波均随之改变；仅凭这次比较不能将剩余预测变化分离归因。
在两份测试的共同观测通道-epoch上，basic/reviewed波形RMSE中位数均为0，95%分位分别为
0.00099及0.00241μV，少数局部变化最大3.24644及0.28566μV；没有广泛翻转或整体改坏正常波形的证据。
1份NN在reviewed保留的15个block上，basic原有9个正确、reviewed为11个；
总体仍由60%降至55%，说明覆盖损失与可判集合上的识别能力必须分列。
全部40个seed×方法×剂量配对分解恒等式通过；本轮只是机械支持分解，不是训练QC/测试QC的独立因果消融。

### 单轮九选一目标识别率

用户进一步确认这里的per trial为“9个数字各闪一次后识别，不累计5轮”，随机基线为1/9。
按[单轮R1接受绑定](../assets/brainsync_matrix/single_round_acceptance.json)重聚合原冻结模型的
逐epoch分数；[单轮接受绑定](../assets/brainsync_matrix/single_round_acceptance.json)另存新分析的身份，
原R5结果保持原字节。两份测试录制各50轮，共100个requested九选一；每轮重置聚合，只取本轮9个epoch。
三混合继续是NN source、波形LDA与xDAWN的等权候选概率池；scratch为单独对照，未替换混合中的LDA。
本次新增模型/预处理/阈值/权重fit均为0。

| 校准录制数 | NN source | NN scratch | xDAWN | 三混合 |
|---|---:|---:|---:|---:|
| 1 | 29.00% | 30.00% | 37.00% | 35.00% |
| 3 | 25.00% | 14.67% | 27.00% | 26.00% |
| 5 | 27.67% | 19.00% | 18.00% | 22.67% |
| 10 | 48.67% | 44.00% | 29.00% | 47.67% |

上表为basic，全部100轮可判；神经和混合取原三个目标seed均值，确定性xDAWN仅一份拟合。
reviewed同样保全部100轮分母：NN source及scratch可判90轮，xDAWN及混合可判78轮。
10份校准分别为39.33%、30.33%、20.00%、27.67%。完整逐轮预测、来源及覆盖保存于报告。
8832个完整专家轮与独立直接softmax求值一致，另核不累计其他轮、类别常数偏置不变性、等权池、
缺专家不fallback及tie不强判。输出候选始终为1–9，没有缩成测试中出现的7、9两个数字。

原连续滤波的可用时刻仍为整份录制完成；新计算只改变每次聚合的epoch数，属于回顾性单轮识别，
不作为在线单轮设备验收。100轮来自同一人的两份录制且目标只含7、9；三seed不扩大独立样本量。
此前生成的目标/非目标二分类诊断不属于用户此次要求的九选一指标，不用其88.89%多数类基线解释本表。

与旧GTN scratch LOSO比较时，需同时对齐训练范围与证据深度。
`loso_folds`将其余来源身份作为训练集；scratch表示随机初始化，不表示只用目标个人的少量C。
本次scratch仅用本人C，1/3/5/10份的实际训练epoch分别450/1332/2214/4455，未继承公共源权重。
[单轮比较的现存接受绑定](../assets/brainsync_matrix/single_round_acceptance.json)
按原模型SHA确认：10份同一scratch权重在R1为44.00%，在R5为83.33%；其他三档R1/R5分别为
30.00/63.33%、14.67/31.67%、19.00/35.00%。这些差异没有新fit，不能当作网络版本回退。
可定位的[旧阶段报告](history.zh.md#执行与集成)将GTN 0.8041写作`hit@all`、245折、
时域kernel33；该报告字段不等于R1。用户提及的“70%+”尚未绑定到其确切producer/字段，
因此本次不将其直接改称R1或R5，也不据此将全部差距归因于采集质量。

## 真实label-K与连续观测支持

原`RecordCalibrationPrefixCompiler.compile`将全局K分配给各独立record，授予原binary bits和raw支持；每record右界为`min(next_unlabelled_onset, last_authorized_onset+response_samples, native_sample_count)`。新连续模型接这份实际grant，最后响应按同一预声明边界右删失。若读取更长raw，则未标曝光进入单独声明的潜变量程序；缺bit与non-target分开，held目标不参与C赋值。labels requested、granted、raw used及各epoch profile used分别保存。

现成完整C3软件实例的三个bit均0，已贯通共同响应与共享先验更新；有正bit的真实C及数值反例会直接检验target增量列。K0另按source-only定义，目标raw不用于估noise、reference、shape或scale；正K的C-noise保持其原预算身份。原公共输入、源差异及全请求比较设计见[真实输入就绪性绑定](../assets/research_resume_20261012/public_evaluation_readiness.json)，数值完整程序见[响应与噪声](response_and_noise.zh.md#一次响应c更新到完整九选一)。

新seed20261014的BNCI八人binary给出明确的低剂量概率线索：K200时，预声明0.5NN+0.5xDAWN的source池和scratch池各八人NLL/Brier均劣于各自单NN。source池平均NLL由0.435648增至0.513829、Brier由0.130122增至0.160667；K2000同source池则小幅改善。直觉上，固定平均也给概率误差更大的专家一半权重；专家互补性、概率尺度和校准量需要共同处理。后续沿已实现的C-only OOF温度、ε与整程序选择评价，各预声明固定池保留原结果。这里OOF表示某组样本由未用该组训练的程序预测，避免拿已经练过的题评价学习规则；执行与分析见[公共评估](public_evaluation.zh.md#本轮bnci二分类与个人校准量)。

本文按主题维护已复核的结论、代码接点和下一步。运行日期与源码身份留在各证据及绑定中；使用方式和工程规则由原六指南维护。

## 当前结论与实施

source-informed四响应候选已接入同一before-group选择与保存图。实际合成纵向在一个合法outer held上比较四候选，保存五节点/两C scope，选中mean_shape；outer成绩不参与最终选择，完整selected graph可在不读取future真值、不重新拟合的胶囊内重放。这验证了完整学习程序的接线，公共优劣沿后续完整配对评估。fixed四专家池仍要求原名单齐备，某专家数值不可用时保留整个池的不可用结果；四臂两法的8个专家请求与2个混合池分别记分母。

Joint程序已完成的前两条C内forward OOF显示高置信错答：K0把真position:9判为12，NLL409.5816；K68把真position:24判为11，NLL610.5583，两个最大概率均舍入为1。原marker的target编码与物理membership一致，标签都在原C500授权范围内；完整7行OOF与全部future沿原程序继续。单record连续λ积分保持首条赢家，真答案NLL为470.3023。这把后续研究明确分成两条：准确计算原law，以及用C内证据学习概率尺度/保留备选质量并检验响应与噪声匹配。下一opt-in比较现有有界温度、含uniform锚点的逆温度与ε混合；完整forward/nested学习关系保持，原程序照原政策执行。证据见[预测积分及授权标签读数](../assets/research_resume_20261012/joint_predictive_evidence_20261016.json)。

新的`forward_beta_uniform`采用显式Beta(a,b) prior，只对合法更早C产生的完整OOF概率更新一次。每个selection贡献一个likelihood；正权log-Beta混合用O(n²)递推保存整个后验，未来单decision概率恰为`E(1−ε)p+Eε/J`。直觉上，它综合有限校准数据支持的多种平滑强度，同时为备选保留概率。真empty roster使用原prior，非空未完成/失败roster保持不可用；完整nested learner只使用outer held之前的真实行。数学核心和完整opt-in生产调用链均已接受，见[设计、源码与实施验收](../assets/research_resume_20261012/joint_forward_beta_design_20261016.json)。

实际小型软件纵向完成4个outer组和1个future：fixed、旧前向温度、新Beta三个候选都进入nested选择，n0与n4的整后验保存/加载、原issuer的capsule打开和真实parent数值重放均通过，重放新增fit/采样为0。把C标签、权限和后验一起重签，会被原数值模型的fit身份拒绝；把raw OOF与全部后验/保存记录一致重签，虽可正常加载，仍会在原C/P数值重放中被识别。root核21个实际执行文件、277成员代码档案中的20个涉及模块及独立审阅，验证这条链确实依赖原输入和原预测。真实完整7行及独立future比较继续沿各自programme执行。

这一具名条件模型给出`q(y)≥a/[J(a+b+n)]`。例如显式Beta(1,1)、n=7、J=32时，单次后验预测NLL至多log288，最大候选概率至多257/288；相对uniform的完整prequential累计logloss至多多log8。参数先验由新programme声明，概率/动作/tie和晋升仍由原owner评价。早期两个C内诊断读数、完整7行因果读数与后续独立future各保自己的来源，模型与概率选择沿完整requested比较。

完整7行连续预测及其因果Beta消费者现已接受。第i题使用此前i题的完整后验评分，之后才纳入当前真值；评分的n为0至6，最后n7状态仅保存供future使用。全部真值来自原C500授权，预测先完整封存，actual evaluator随后恢复7个完整组476标签和第二record的24个partial标签。两个连续law都保全7/7概率与动作、0tie、0/7命中。下表为whole law；fixed-prior原NLL为318.6237，校准结果相同。

| 概率程序 | 平均NLL | Brier |
|---|---:|---:|
| 原连续预测 | 319.3663 | 1.999888 |
| 具名Beta(1,1)，只用此前结果校准 | 3.7628 | 1.057072 |
| 32候选均匀概率参照 | 3.4657 | 0.968750 |

NLL是给真答案的概率过低时产生的惩罚。校准让错误的最高置信度由首题约51.56%逐步降到末题15.23%，最终`Eε=8/9`；它学会了为备选保留质量。这7题的原真值概率都极小，因此每次错答近似把后验推进到Beta(n+1,1)；累计NLL相对uniform多`log8`，与这一模型的预声明上界吻合。当前候选排序需要响应模型改进，下一有限delay/width研究会在同source prior和noise下检验这一点。原模型、全部预测、逐题后验及9个实际概率/指标owner保持原身份；独立80位正混合oracle逐步复算14条向量、整后验和n7，最大概率差3.62e−16。新消费者8.13秒、0新fit/采样/λ积分，完整源码和实际读数见[因果7行接受](../assets/research_resume_20261012/joint_forward_beta_design_20261016.json)。

binary候选的排名量固定为fit前requested正负样本对的成功credit；任一输出缺失的pair记0，分母保持`n+×n−`。完整输出等于原AUC，删除输出只能丢失非负credit。可用子集ROC AUC作诊断单列，同风险先保coverage。legacy CE固定共同校准class cost与requested验证mass，完整输出才参与排名；BrainSync继续保全部真实45-flash block，缺失及tie为miss。普通/QC/EOG产品与公共E3共原fit/refit owner，保存支持、评分和final选择关系。两个实际source候选的新旧对照中，full-C参数、独立预测及保存加载逐bit一致；最终入口关系核查随集成源码绑定。

`transfer_scratch_prefix_plan`已将source NN、matched scratch NN及各自NN+xDAWN接入新四候选fixed程序族，只替换新计划的`prefix_selection`。source/scratch共input profile、架构和匹配配方，xDAWN保其自身声明视图；物理decision、预算与可用时间由原owner核。baseline与tie政策预声明，future队列评价最终程序。实际source/scratch初始化、K0、C200/800、五个合法forward选择行、同run图/保存/零fit重放与parent评分已贯通，见[整合绑定](../assets/research_resume_20261012/research_integration_acceptance.json)。下一公共计划沿该新roster冻结，与原entry005分别保存身份。

pool层uniform-contamination已完成fixed-base、严格cross-fitted温度+ε及ordinary各exactpool的完整软件程序：`pε=(1−ε)p+ε/J`，ε从原roster/weights的合法OOF概率以凸NLL目标学习。温度OOF与pool OOF分别保存；pool held组的温度来自更深的排除关系，实际温度expert子集逐包关联。每个ordinary名单/权重组合独立持有自身OOF、ε、模型和parent复算。完整requested失败保留，C800 final保partial80、六meta行使用720完整标签。原fixed首片、selectedfixed及完整T/ordinary各保原producer；最终5项实际合成软件纵向通过，selected29请求和ordinary464预测经正常parent及篡改反例闭合。`937cbae`已集成，来源见[整合绑定](../assets/research_resume_20261012/research_integration_acceptance.json)。直觉上，温度调节分数差距，ε保留其他候选的概率；下一冻结公共比较评价其概率质量和迁移异质性。

fixed使用温度1，grouped按完整selection生成C内OOF概率，并为参与校准的专家学习温度；两个程序共用最终专家。现有K200最终拟合保全部200个raw标签和196个合法trial，完整selection有一个，后80个标签已经参与合法final fit。

整程序选择已作为opt-in接入实际公共caller：每个h/v从原source重编真正更早的prefix，完整运行候选学习算法，再对held group评分；最终选定程序用原授权K全部合法标签refit。候选明确包含“单prefix final + group-LOO温度inner”；inner/final剂量不同是这个算法的一部分。排序复用已有hit/NLL/Brier与baseline policy，同run模型/输出保存后零新EEG拟合重放，父进程从真实专家重算混合概率。当前入口与接受来源见[整程序接线与复签](#整程序接线与复签)。

实际fitter样本、标签、prior/noise/statistics与graph共同关联。selection行数下限是预声明策略参数；单组也可定义学习算法，later独立queue用于评价最终程序。当前group-LOO入口的三组下限继续保留其原程序身份。

原source10/entry003以failed、OS1结束：首四人包保存重放后由parent因JSON list/FrozenMetadata tuple比较拒绝，后四人遇到FS/RAM预算失败。修后entry005已零fit准入原四人、仅补原未完成四人，完整八人152请求及92面板闭合；32份selection replay对应600个整程序节点、3290次exact-score检查。K2000选择保命中并改善主pool概率风险，K800少1命中且概率指标变差；实际结果与目标端nested范围见[公共评估](public_evaluation.zh.md#估计量与下一步)。原failed终态和各producer保留；下一研究进入合法C内风险与异质性，下一执行底座由云owner优化分块重放和只读输入。

## 完整公共前缀的选择行为

原entry005已完成8人、152共同future请求，seed20260918。2026-10-14直接从原`4b249832…`结果的13984条预测复算全部92面板，最大保存指标差`1.78e-15`、结构异常0；来源见[恢复与复算绑定](../assets/research_resume_20261011/cloud_payload_recovery_binding.json)。K200固定/选择均83/152，K800为113/152与112/152，K2000均132/152；三档选择−固定NLL分别约`+0.000357,+0.061106,−0.104747`。

K800五项预测变化集中A04：救回2项、损害3项，正确数5→4，ΔNLL+0.876779；其他七人平均ΔNLL−0.055418。K2000 A06同时有1项救回和1项损害，各人相同正确数仍可包含不同预测。该档7/8人NLL/Brier改善，96.31%的NLL收益来自A03/A05/A06；选中的四种程序继续保原C内选择身份。

同K2000的matched scratch为135/152、NN+xDAWN固定组合133/152，分别提供更高hit与更低NLL的匹配锚点。选择程序相对主pool的最大错误置信下降，错误集合内平均置信上升；这两个条件描述保各自分母。下一轮优先检验低剂量选择稳定性、C内概率校准和source/scratch完整程序选择。上述future分量用于诊断，新默认和阈值仍通过预声明C内程序学习。8项不完整支持及错误尾部详见[公共评估](public_evaluation.zh.md#修后前缀完整八人152请求)。

## 小组、部分标签与采样坐标

BNCI原始MAT与事件坐标为256 Hz；trial/NN模型输入为128 Hz、179点。原common profile的source support为`[-51,308)`；P1 raw continuous的1.2秒basis为307点。两种几何由各自owner提供。修后A01 probe与[来源绑定](../assets/research_resume_20261011/prefix_small_group_fixed_binding.json)保存了实际窗口。

每个selection有10个repetition、120个flash；每个repetition是6 row+6 column，共12个flash，最终词表是36个row/column组合。同一selection目标恒定。R=1/2/5时，完整响应尾跨入下一未纳入flash，适合显式fragment/right-censor研究；这类输出保留原selection的重复测量关系。

A01的后80个标签可单独编为`[11477,16648)`的suffix scope：80 raw labels、76完整trial、4个right-censored事件、5139个conditioning32之后的行；与前120-label完整组有1753 samples的gap。用该suffix训练后评价前组是offline backcast；当前P200 final依旧使用200 raw/196 trial。独立裁剪再处理对held扰动不变；全C双向处理后删行产生`3.26665e-6 V`泄漏，验证了先隔离raw再预处理的必要性。

一个out-of-fit温度观测可得到边界、内点或flat最优；固定source先验的正则温度也可计算。source-only K0预测或独立suffix训练可提供明确的新学习入口，完整算法随后在独立later数据上评价。温度与超参数选择分层记录，已用于拟合温度的那一行作为meta训练数据。[独立评审](../assets/research_resume_20261011/prefix_program_review_binding.json)和root单行解析反例保留了具体来源；先前128/256及广播错误版本保存在本地作为被替代证据。

## B03：整个程序的nested边界

外层held组记为v，选择层held组为h，温度训练held组为t。评价v之前，候选选择只读`C\\v`；评价候选在h上的输出前，其模型与温度只读`C\\{v,h}`；温度的OOF基础模型进一步排除t。固定过滤在各组的合法物理scope内执行，统计、prior和真实expert fit从各自subset产生state；完全相同subset才可复用实际缓存。最后全C选择与refit不读取外层评价结果。

`research.calibration_selection`定义预声明candidate roster，固定或分组温度程序；完整概率支持、hit→NLL→Brier顺序、baseline输出/失败保护与固定tie顺序共同决定选择。失败、不可用和tie保留requested owner，不能缩小expert roster或重归一化绕过失败。run-local缓存按实际group记录、labels/sample digest、program与source bytes绑定，只复用完全相同subset；原expert wrapper冻结，温度派生新对象。

审查复现：重复/重叠raw或record未在split前拒绝、decision身份延迟拒绝、event身份重复被接受、同final数值包可重写label authority、选定温度子集与actual final不一致，以及可变expert wrapper漂移。修复以原ensemble owner的`validate_calibration_ensemble_groups`统一完整roster准入；`_calibration_expert_fit_digest`复用原fit身份公式，constructor/load将最终scope、labels、checkpoint角色和温度子集绑定回原FittedEnsemble。

本切片交付完整组科研API及final包；opt-in inner producer capture/replay、公共B12 prefix温度调用及真正prefix整程序选择已在后续接通。BrainSync入口按工作站计划推进。W06.11/W09.06的程序隔离已有软件出口；W07.13 gating与新participant评价另行推进。

## 原 MAT 保存重放验收

从已接受 `cf48f78` 物理导出源码，在原 A01 MAT（SHA `cdf9bca4…`）的授权 C800 中拟合六个完整组，实际使用720 group labels。原 harness 在有效模型已保存后错误地重评 calibration views，被原 `ensemble prediction repeats a calibration observation` 门正确拒绝。模型状态没有因此失效；没有放宽该门。

修后使用原 independent-prefix owner 的 C800 之后 future queue，只消费 label-free view，不读取 queue truth 选择程序。重放保留全部29 requested decisions，其中28有合法概率，1项不可用仍在分母；原模型→重新保存→加载的证据29/29一致，三份package文件逐字节相同，纠正过程中新增拟合0。这个结果是公开数据上的软件保存重放证据，不报告hit/NLL或新效能。原失败日志、原包和修后消费者身份各自保留，详见[紧凑绑定](../assets/research_resume_20261011/native_nested_replay_binding.json)。

## CalibrationPrefix保存重放与调用端复签

后续prefix修稿已经独立复签、经总监逐读实际fit/OOF/save/load/future/parent admission后接受。`fit_calibration_prefix_programs`捕获同一次实际final专家和inner状态；`persist_prefix_inner_replay`绑定raw/trial样本及标签、requested aliases、实际recipe/checkpoint seed与source role。load从原数值模型的fit binding和final OOF owner核关系，不能仅重签manifest副本。成功OOF必需的inner roster/held score完整；typed inner fit失败、返回不可用evidence及直接raised score异常各保真实凭证，不虚构成功重放、不删除合法fixed程序。

实际independent caller捕获完整truth-free future expert outputs，核共同Kmax test scope，保存/load/replay后才封prediction。父验收从已捕获专家用原EnsembleScorer重新计算plan mixture并核完整概率/失败记录；同一model digest不能授权改概率。iterable只消费一次，不可用温度strategy保全requested未来行。外部JSON容器与内部冻结tuple采用既有semantic owner比较，矩阵、导序、reference、fs、SHA和值变化仍拒绝。fit/cache miss的真实状态也已修复，共用造数归tests/support。

最终独立原六类违规全部拒绝、两个诚实fixed/grouped对照通过；新增consumer7项及真实Cz6 caller/raised-score两项通过，源码七文件前后稳定。原caller probe因缺新必需test_scope仍exit1并保留原字节，合法scope/queue由新增独立harness验收，不把原exit冒记为绿色。作者34项compact/cache/三对caller属于spatial同义修前body；30项prefix/actual spatial属于最终body，native interruption1项另绑定最后测试字节。旧74项、source_changed失败和旧RED保持自己的source身份；详见[紧凑绑定](../assets/research_resume_20261011/prefix_replay_binding.json)。

实际6包receipt从旧10.22–10.95MB缩为新1.48–2.21MB，父验收对每包只解析/核验一次并消费冻结索引，消除原1224次预测重复解析同一大凭证的路径；这里只陈述实测尺寸与调用次数，未声称训练端到端提速。当前事务粒度为整个prefix future capture/replay后逐行封预测，已计算但未封行中断时仍为partial/unknown；native退出验证只证明已封行、已处理strategy和完整requested账本保留，不证明任意中断可零fit恢复。

这是实际prefix/public候选runner的软件接线；它仍没有候选程序的strict nested selection或learned gating，不完成公共效能、跨host整包或成人确认。原MAT窄输入验证另用冻结源码执行，不把上述synthetic通过重签成公共结果。

原A01 MAT后续窄验证已exit0：从clean已接受源码按实际工作字节冻结971成员（保CRLF/LF字节），只跑预声明raw+Toeplitz、K200/K800、fixed/grouped两个程序与共同独立未来队列。29 requested decisions产生232 prediction行；每K捕获58个future strategy行。raw使用200/800标签，trial实际使用196/796标签；K200 grouped不足组具名不可用，K800保6个真实inner producer并成功重放。所有模型load/future replay及实际parent mixture准入通过，原MAT前后SHA一致；源checkpoint拟合0，没有用未来truth选方。它只证明原公共输入的软件链，未签收分类效能。源码/原MAT/脚本/退出码/包摘要见[原MAT前缀绑定](../assets/research_resume_20261011/native_prefix_replay_binding.json)。

## 整程序接线与复签

`fit_nested_prefix_selection`复用原fitter执行完整`A(P)`，`CalibrationPrefixCompiler`只从verified MAT bytes解析一次并编译各真实before-group scope。候选共享结构h schedule，数值失败仍占原请求；`minimum_selection_rows`显式写入计划，结构不足按原baseline policy处理。outer只描述同一参与者的相关learning path，final独立future另行评价。

`prefix_selection_replay`保存同run的h/v/final graph和各原模型包；actual replay从原compiler重建所有prefix及native输入，核真实h概率和选方规则。`prefix_inner_replay/2`绑定final与inner raw/trial原生输入，旧`/1`继续由原冻结owner重放。温度通过唯一`temperature_from_oof`只读复核同一有限meta learner；每包一次小优化，新EEG fit为0。

公共plan新增`prefix_selection`（schema、output_name、baseline_name、显式minimum_selection_rows及candidate records），在原fixed/grouped输出之外追加具名selected请求。整个future捕获、保存、原输入graph重放完成后才封预测；parent重编原MAT并核实际selected mixture。selector的started/failed/partial由自己的receipts持有；final ExpectedSubjectError的完整future保留typed不可用。

真实C800改签C200、full-C meta搬早期v、h概率篡改、温度全摘要自签、constructor自签source权限、selector中断receipt混用等反例已闭合。作者`d0edc170…`的八个文件经root逐读、原字节集成，含当前continuous白化/P1/新horizon依赖的12项相关验证全部通过，Ruff与diff检查通过，见[接受绑定](../assets/research_resume_20261011/prefix_program_acceptance_binding.json)。旧快照/模型保原执行身份；下一步冻结真实opt-in计划，评价完整独立future队列。

真实八人programme保原K0/200/800/2000、ordinary arms和source/scratch；新增primary fixed/grouped与NN+xDAWN fixed/grouped四个完整候选，minimum_selection_rows=3。零fit原MAT census确认K200/800/2000的完整组数为1/6/16、合法final选择行为0/3/13，共同Kmax future152请求/144可评分。云端以实际source checkpoint aabb2a43、递归重编finite profiles和source10/527新consumer冻结执行；本地e083 census保持自己的source身份。首次526 directCLI import在任何fit/score前失败；修复后新entry003重新执行完整矩阵，保留旧失败账本。结果终态后再评价原风险与选择效应。
