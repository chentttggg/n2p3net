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

| 数据层 | 作用 | 结论边界 |
|---|---|---|
| public binary ERP source | EEG、target/non-target、participant/session/domain和采集provenance | 表征学习与source数据剂量；不要求数字/candidate身份 |
| public pseudo-target | 留出source未见participant/domain，在固定后续段测试 | 是否降低公共目标校准标签/时间成本，development |
| 合法candidate数据 | 有每flash物理membership及真实decision边界 | 该数据自身的固定预算/聚合/停止，不自动是成人九选一 |
| adult BrainSync | 七EEG+EOG、completed、已确认标签、独立target-switch记录 | 成人九选一、真实成本、跨日与最终确认 |
| synthetic/contract | 不变量、梯度、序列、失败状态 | 软件/数学验证，不是性能 |

此前核查的旧BrainSync目录只有error记录；本次用户补充`BrainSync_Tool_newest`后发现
4份completed/confirmed候选记录和3份error记录，见[版本比较](evidence/brainsync_versions_20260906.zh.md)。
这些记录尚未完成物理通道、参考、时钟、信号和成人研究准入；不能称已经获得产品确认。
error记录、旧GTN、BI和合成信号不能冒充成人确认。
GTN儿童三导且固定thought digit，只能作声明限制下的开发研究。

## 2. 活动算法路径

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
按计划激活的增量对照，没有matched附加收益前不默认叠加，也不以CI含0宣判无效。
复用既有模块及成熟MNE/sklearn/PyTorch工具；新机制先证明简单路径留下的具体缺陷。

LDA顺序实验作为独立开发对照：微调后冻结特征并拟合收缩LDA，或用校准LDA初始化
线性头再微调；原微调head作共同基线。标准化系数必须折叠到head：若LDA分数为
`a^T((h-m)/s)+b`，则`w=a/s`、`c=b-w^T m`，两logit取`(-g/2,g/2)`使差等于`g=w^T h+c`。
后续交叉熵更新后的head不再称为LDA统计解；本轮不引入可微协方差或交替优化。
校准内选方时，LDA初始化和最终重拟合也必须各自在inner训练范围内重做，不能预先用
完整校准特征拟合后再声称inner validation独立。

## 3. 估计量与迁移收益

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
