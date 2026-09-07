# P300 实验目录的文献与现场核查记录

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

> 后续目录清理：本文本地`tmp/`分析结果已迁入results history，旧方案已物理压缩。
> [迁移索引与保全说明](workspace_cleanup_20260906.zh.md)提供原路径查询；下文保留核查时口径。

访问日期：2026-09-06，Asia/Shanghai；本轮查询始于09-05晚间并跨午夜。
配套文件：[72项实验目录](research_experiment_catalog_20260906.zh.md)。
源码核查基线：`c8ee573`；本轮开始时工作树仅有用户未跟踪文档
`doc/critical_architecture_review_20260905.zh.md`，本次未修改该文件。

## 1. 检索范围与证据等级

已加载项目 `eeg-p300-research`、`scientific-critical-thinking`、
`scientific-brainstorming`、`experimental-design`、`statistical-power`、
`research-lookup`、`paper-lookup` 与PDF技能。三路独立研究分别核查外部基线/协议、
信号/伪迹、模型/聚合，由主agent核对关键原文、源码和数学反例并整合。
这些是AI辅助研究观点，不是多名人类专家的独立实验或统计复现。

这是面向当前工程决策的定向文献检索，不是PRISMA系统综述，也没有证明候选方向的新颖性。
没有为了凑够某一参考数量加入弱相关论文。优先使用原始数据报告、作者论文/源码、
官方MOABB/MNE、Europe PMC、Crossref、NCBI与OpenAlex；只检索公开问题，不上传项目数据。

证据等级：**全文**=读到与所述结论相关的Methods/Results/协议原文；
**摘要**=只支持摘要所能陈述的内容；**实现**=官方/作者代码而非论文全文；
**元数据**=只核实身份/DOI，不证明方法结论。检索结果不直接提供本项目性能证据。

`research-lookup` 默认Parallel后端在本环境没有CLI，也没有可用的Parallel/OpenRouter
环境凭据，因此未调用该服务或启动其付费任务；查询改用已加载 `paper-lookup` 的公开API。
arXiv HTML/部分接口出现TLS EOF，已用本地PDF、HAL全文和官方结果页交叉核验。
PubMed网页cookie阻挡以官方eutils替代；部分Europe PMC全文返回404的论文仅记摘要。
OpenAlex JSON含大小写不同的abstract token键，PowerShell默认解析失败，改用
`ConvertFrom-Json -AsHashtable`/限制select字段后正确读取；未将失败返回当成无文献。

## 2. 需要改变当前研究优先级的原始证据

### 2.1 BI2014a：group身份与游戏边界存在来源冲突

R01全文，PDF p8，原文：

> Note that the groups are constructed randomly at each repetition, thus a physical symbol
> in the matrix does not belong to the same 'row' or 'column' group across repetitions.

同页还明确，这些group不是屏幕物理行列。每次闪烁二分类标签能从第19列得到，
但这不自动给出跨repetition的物理symbol身份。当前代码
`src/data/bi2014a_schedule.py:107`把20–25/40–45/60–65/80–85解为固定row/column；
`src/data/bi2014a_candidate.py:218`用target pair变化及100/104切selection；
`src/transfer/candidate_decision.py:176`按group累加logit。

反向证据必须保留：subject01原始99轮中target pair仅27次变化，有多轮相同pair；
早期R06游戏源码只在loadLevel调用ShuffleMatrix，并在轮间积累判别信息。
该代码未绑定2014采集版本，所以不能凭它推翻技术报告，也不能仅凭报告一句话判定
现有解析器必错。target pair相同也不足以证明其余35个物理候选映射不变。

**裁决状态：关键协议冲突未解决。** 要从实际版本/配置/日志/随机化规则恢复membership，
或把不可恢复的结果明确限制为group-ID surrogate。现有target/non-target分类可以在
标签和时间审计后继续；未恢复物理映射的跨R准确率不得冠以已确认字符准确率。

R01 p8明确解释100为新block、102为repetition结束，未给104/105/106完整含义。
当前104=level/restart的解释需要额外来源。原文还把20–45和60–85两段都写成
“containing target”，有明显文字错误；码表必须与raw第19列和实际发布实现交叉核对。

R01 p7：命中target结束level，八次失败后重启；p6还有实验者按控制状态提前结束。
所以记录长度由原策略和表现共同影响。长R分析估计的是
`P(correct under evaluated decoder at R | original recording supports R)`，
不是随机删失后的总体固定R准确率。新算法希望继续到未记录的R时，离线资料没有该EEG。

### 2.2 队列、通道和参考不是已完全确定的常量

R01 p4–5明确：62/64视力未完全矫正，63接触差，这三人资料虽提供但
“very noisy and should not be used for P300 data analysis”。36/46/52/57的ERP不一致，
作者未记录相应采集问题，不能按新模型表现追加删除。

MOABB与当前cache使用01–64。**复现其结果时保留64；按原作者建议的研究队列01–61
另列预声明分析/敏感性，两者方法差值分别报告。** 71是招募人数，不是可用EEG人数；
65–71资料未提供。不能把71、64、61混写成同一个数据集样本数。

R01 p5正文通道为Fp1/Fp2/**F5/F6**等，当前loader和项目为**F3/F4**；
报告正文参考是右耳，图注另有左耳表述。优先检查Header和实际采集资料，不能静默选
有利版本。Fz是报告中的ground位置，不是一个测量的Fz EEG通道。

### 2.3 原批判文档的深度模型基准数字未获所引来源支持

本地 `Paper/MOABB_Reproducibility_Benchmark_arXiv_2404.15319.pdf`：43页，
CreationDate 2024-04-25；HAL全文另含封面。p16 §4.3明确该版DL重点是MI。
p41 Table D4及R03官方P300/ERP页都只有以下五条BI2014a基线：

| 方法 | AUC均值 ± SD，百分数 |
|---|---:|
| XDAWNCov + TS + SVM | 85.77 ± 9.75 |
| XDAWNCov + MDM | 80.88 ± 11.01 |
| ERPCov(svd_n=4) + MDM | 72.11 ± 11.64 |
| ERPCov + MDM | 71.62 ± 11.17 |
| xDAWN + LDA | 66.60 ± 7.54 |

这里的SD按官方页为先fold平均后的subject/session层级，不是新项目的paired CI。
未在这些来源找到EEGITNet86.66、EEGNeX85.08、EEGNet82.09、ShallowConvNet63.19的
BI2014a P300行。这是**引用来源未支持**，不是证明所有文献都不存在相应数字。

R02正文支持连续4阶Butterworth双向滤波、P300 1–24Hz、session内shuffle分层五折、
训练内层三折选择；R04版本化代码P300默认0–1s、baseline=None、resample=None，
BI2014a全16导/64人。论文精确运行commit/依赖尚未完整绑定，v1.1.2不能自动视为
原论文全部执行环境。R05 SVM含C=[0.1,1,10]及kernel=[rbf,linear]的搜索，
不能把默认无调参模型与85.77硬比较。

### 2.4 文献没有支持“千参数网络必然容量不足”

R08 Borra全文Table2报告1154/1210参数的轻量网络，Table3有较好的P300表现。
其数据是BCIAUT-P300和两个私有数据集，不是BI2014a；三组WS/LOSO AUC分别为
83.52/75.40、89.60/74.82、92.63/86.09（百分数）。这反驳参数量本身的决定性论证，
但既不证明当前N2P3Net充分，也不提供本项目硬门槛。

### 2.5 凝视、眼电、视觉ERP与重叠应分开

R12全文与R13摘要支持P300 speller常依赖凝视，早期视觉ERP也贡献分类。
眼电是候选污染源，真实P1/N1/N2等是其他神经信息，不能两者混同。
R14全文强调deconvolution所需的时间设计变化、近似线性叠加和波形稳定假设；
不应期/疲劳造成的非平稳响应不被线性去重叠自动修复。

R15及R19支持ICA在约1Hz高通副本上拟合、应用回任务带宽；R19提醒ICA前不baseline、
去除ICA后再baseline，并警告额区EEG代理EOG会误删脑信号。ICA结果不是眼电因果真值。
R16的baseline GLM是统计分析方法，不能直接推导“当前CNN无baseline更好”。

## 3. 本轮现场计算与边界

只读文件：`cloud_inputs/l0_bi2014a_causal_v3_64subj.npz`及其record/candidate_audit；
`tmp/analysis_relaxed_native.json`、`tmp/plan_dt_v2_relaxed_dev64_1seed.json`。
缓存计算读取metadata、事件和QC结构，没有为本次报告重新训练或优化模型。

| 核查 | 实际读数 | 解释限制 |
|---|---|---|
| cache shape | 61011×16×179，128Hz，[-200,1200)ms，0.1–30Hz forward，mean_only baseline | 是当前cache，不是MOABB论文配方 |
| parser分组普查 | 1736段、5085轮，均值2.929147 | 依赖当前selection解析，不能先当真实物理目标决策 |
| dev64 requested/complete | 1416请求；R1=1413，R2=964，R3=628，R8=73 | 覆盖99.79/68.08/44.35/5.16%；不能当随机缺失 |
| metadata内同selection相邻间隔 | median0.3515625s；1%0.166015625s；99%1.388671875s | 是解析后的event onset统计，不是屏幕photodiode延迟 |
| 同selection相邻epoch重叠 | gap<1.4s占0.9917840574 | 说明random-trial分割有共享raw风险，不证明每个fold或所有成绩泄漏 |
| 边界只靠target_pair_change | 145/1736；另有1391个104+pair、101个100+104+pair、64起始、31个104、4个100+104 | marker含义/物理映射仍需裁决 |
| 统一timeline | complete=False，candidate_ids均空，repetition_indices全-1 | metadata有信息不能替代所有timeline消费者；不能只篡改complete标志 |
| QC | cache有EpochQCFeatures/1；dev64各eval arm fold_local_qc=False | 已有QC模块，不是全项目“没有QC”；这批确实没启用对应QC |
| dev64机制激活 | feature_alignment_weight=0；SplitBN/源residual字段null；full_fine_no_mmd | 不支持“AS-MMD/Split-BN全部CI含0” |
| linear对照 | shrinkage normalization、target_time_split；full_fine用source、fixed_budget | 随机head以外至少还有配方混杂；实际优化器AdamW |

已读源码确认：filter在完整raw上执行后才epoch，使用steady-state初态；
FFT resample在整个epoch内执行。因此“每个epoch重置IIR”不符合实现，
而裁现有FFT cache后声称更早在线输出也缺少时间支持证据。

七项数学检查均用`.venv/Scripts/python.exe`的NumPy实际断言通过，见目录§13。
另有第八项实际MNE算子反例：717点合成epoch（512Hz，起点-0.2s），sample600放单位
脉冲，其时刻为+0.971875s；执行`mne.filter.resample(up=1,down=4,npad='auto',
window='auto',pad='edge')`后，+0.6s前输出最大绝对值为0.0016904403，大于1e-12。
这验证epoch FFT算子非局部时间支持，不是估计真实EEG的污染幅度或性能影响。
没有运行全量测试，没有把这些synthetic/invariant检查称作E0/E1或生理实验。

## 4. 来源清单

以下21项来源按研究用途列出，R04/R05/R06/R07/R19是实现/指导而非独立性能研究。

| ID | 来源与身份 | 本轮核验级别及用途 |
|---|---|---|
| R01 | Korczowski et al. 2019, *Brain Invaders calibration-less P300-based BCI using dry EEG electrodes Dataset (bi2014a)*；[HAL-02171575v1](https://hal.science/hal-02171575/document)，数据[DOI10.5281/zenodo.3266223](https://doi.org/10.5281/zenodo.3266223) | 全文p4–8；实际获取811706bytes，SHA256 `984c4fcca62f47efe65638579c3ae4e424a8c56c4aa0491c4641c11568e40e0a`；协议/队列/通道冲突 |
| R02 | Chevallier et al. 2024，MOABB benchmark；[arXiv2404.15319](https://arxiv.org/abs/2404.15319)，[HAL全文](https://hal.science/hal-04537061/document) | 本地与HAL全文；P300表、CV与预处理；版本题名略有差异，按ID辨认 |
| R03 | [MOABB官方论文结果页](https://moabb.neurotechx.com/docs/paper_results.html) | 当前页面表格；五条P300经典法，不能混入MI神经网络表 |
| R04 | [MOABB v1.1.2 P300 paradigm](https://raw.githubusercontent.com/NeuroTechX/moabb/v1.1.2/moabb/paradigms/p300.py) | 版本化实现；P300预处理默认，非自动的原论文运行证明 |
| R05 | [v1.1.2 XdawnCov_TS_SVM.yml](https://raw.githubusercontent.com/NeuroTechX/moabb/v1.1.2/pipelines/XdawnCov_TS_SVM.yml) | 版本化实现；OAS、xDAWN SCM、SVM内层参数搜索 |
| R06 | [Daeda88/BrainInvaders早期GameScreen.cpp](https://raw.githubusercontent.com/Daeda88/BrainInvaders/master/Brain%20Invaders/Brain%20Invaders/SpaceInvadersBCI/GameScreen.cpp) | 历史实现反证；loadLevel才ShuffleMatrix，未绑定2014采集版本 |
| R07 | [原作者BI2014a loader](https://raw.githubusercontent.com/plcrodrigues/py.BI.EEG.2014a-GIPSA/master/braininvaders2014a/dataset.py)、[MOABB loader](https://raw.githubusercontent.com/NeuroTechX/moabb/develop/moabb/datasets/braininvaders.py) | 实现；F3/F4、64人、折叠session/run，和R01有冲突 |
| R08 | Borra, Fantozzi & Magosso 2021, *A Lightweight Multi-Scale Convolutional Neural Network for P300 Decoding: Analysis of Training Strategies and Uncovering of Network Decision*；[DOI10.3389/fnhum.2021.655840](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.655840/full) | 全文；轻量网络、WS/LOSO/迁移，数据不含BI2014a |
| R09 | Lawhern et al. 2018, *EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces*；[DOI10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)，[作者实现](https://raw.githubusercontent.com/vlawhern/arl-eegmodels/master/EEGModels.py) | DOI元数据+完整作者实现；物理尺度与dropout，未声称读论文全文 |
| R10 | Blankertz et al. 2011, *Single-trial analysis and classification of ERP components: A tutorial*；[DOI10.1016/j.neuroimage.2010.06.048](https://doi.org/10.1016/j.neuroimage.2010.06.048)，PMID20600976 | 摘要；高维ERP协方差估计与shrinkage LDA |
| R11 | Throckmorton et al. 2013, *Bayesian Approach to Dynamically Controlling Data Collection in P300 Spellers*；[DOI10.1109/TNSRE.2013.2253125](https://doi.org/10.1109/TNSRE.2013.2253125)，[PMC3798004](https://pmc.ncbi.nlm.nih.gov/articles/PMC3798004/) | Methods/Results全文；条件独立likelihood、训练密度、真实与theoretical bitrate区分 |
| R12 | Treder & Blankertz 2010, *(C)overt attention and visual speller design in an ERP-based brain-computer interface*；[DOI10.1186/1744-9081-6-28](https://doi.org/10.1186/1744-9081-6-28)，PMC2904265 | 全文相关段；13人eye-tracking、早晚ERP和重叠控制 |
| R13 | Brunner et al. 2010, *Does the 'P300' speller depend on eye gaze?*；[DOI10.1088/1741-2560/7/5/056013](https://doi.org/10.1088/1741-2560/7/5/056013)，PMID20858924 | 摘要；17人凝视条件，不能等同眼电污染；全文API404 |
| R14 | Ehinger & Dimigen 2019, *Unfold: an integrated toolbox for overlap correction, non-linear modeling, and regression-based EEG analysis*；[DOI10.7717/peerj.7838](https://doi.org/10.7717/peerj.7838)，PMC6815663 | 全文方法/假设段；去重叠的识别条件与限制 |
| R15 | Winkler et al. 2015, *On the influence of high-pass filtering on ICA-based artifact reduction in EEG-ERP*；[DOI10.1109/EMBC.2015.7319296](https://doi.org/10.1109/EMBC.2015.7319296)，PMID26737196 | 摘要；21人听觉oddball，ICA拟合高通不等于任务默认高通 |
| R16 | Alday 2019, *How much baseline correction do we need in ERP research? Extended GLM model can replace baseline correction while lifting its limits*；[DOI10.1111/psyp.13451](https://doi.org/10.1111/psyp.13451)，PMID31403187 | 摘要；baseline作为协变量，不是当前CNN收益证据 |
| R17 | Lu et al. 2013, *The effects of stimulus timing features on P300 speller performance*；[DOI10.1016/j.clinph.2012.08.002](https://doi.org/10.1016/j.clinph.2012.08.002)，PMID22939456 | 摘要；6人时序/通道研究，支持实测accuracy–speed而非通用最优SOA；全文API404 |
| R18 | Dong et al. 2019, *A Comparative Study of Different EEG Reference Choices for Event-Related Potentials Extracted by Independent Component Analysis*；[DOI10.3389/fnins.2019.01068](https://doi.org/10.3389/fnins.2019.01068)，PMID31680810 | 摘要；参考影响IC/ERP，不外推REST在八导干电极必优 |
| R19 | [MNE ICA官方教程](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) | 正文；ICA拟合高通、baseline时机、额区EOG代理风险 |
| R20 | Cawley & Talbot 2010, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*；[JMLR11:2079–2107](https://www.jmlr.org/papers/v11/cawley10a.html) | 官方摘要；有限验证数据上的多次模型选择可过拟合，不能把dev最佳当确认 |
| R21 | Lakens 2017, *Equivalence Tests*；[DOI10.1177/1948550617697177](https://doi.org/10.1177/1948550617697177) | OpenAlex元数据；用于定位等效检验文献，具体设计建议来自statistical-power技能与标准统计定义，未声称全文核实 |

补充已核摘要但不作主要论证：Hoffmann et al. 2008，*An efficient P300-based
brain-computer interface for disabled subjects*，DOI10.1016/j.jneumeth.2007.03.005，
PMID17445904；本轮未从摘要获得足够BLDA实现细节，所以目录优先使用已明确的
shrinkage LDA/ridge，没有声称忠实BLDA已闭合。

## 5. 查询记录与主张映射

直接URL见R01–R09、R19–R20；下列API均使用URL编码参数，检查JSON/XML是否实际含结果。
只做有限页的相关性检索，没有把第一页结果当成某领域全部文献。

| API/端点 | 实际查询与范围 | 使用/失败 |
|---|---|---|
| Europe PMC `/webservices/rest/search` | `EXT_ID:20509913`；`TITLE:"Does the P300 speller depend"`；`TITLE:unfold AND EEG` | resultType=core，pageSize分别1/3/5；R12–14 |
| Europe PMC search | `TITLE:"stimulus timing" AND P300`；`DOI:"10.1109/EMBC.2015.7319296"` | core，pageSize5/1；R17/R15 |
| Europe PMC search | `TITLE:"baseline correction" AND AUTH:Alday`；`TITLE:"reference" AND ERP AND AUTH:Dong` | core，pageSize3/3；R16/R18 |
| Europe PMC `/PMC2904265/fullTextXML`、`/PMC6815663/fullTextXML` | R12/R14全文，skill JATS解析器检查body | 成功；R13/R17全文不可得则仅用摘要 |
| Crossref `/works/{URL-encoded DOI}` | `10.3389/fnhum.2021.655840`；`10.1088/1741-2552/aace8c` | 核对R08/R09身份，再访问正文或作者代码 |
| NCBI `/entrez/eutils/efetch.fcgi` | db=pubmed,id=20600976,17445904,23529202,retmode=xml | R10/R11及补充Hoffmann摘要；网页cookie阻挡后使用 |
| NCBI efetch | db=pmc,id=3798004,retmode=xml | skill JATS解析器确认full_text_available，Methods/Results约2771words |
| OpenAlex `/works` | search=`Equivalence Tests A Practical Primer`，per_page=2 | 找到2016预印本与R21 2017期刊；仅元数据，不混成两份独立证据 |
| OpenAlex `/works` | search=`Cross-validation failure small sample sizes neuroimaging Varoquaux`，per_page=3 | 返回有限样本验证及其他相关研究，未用非精确命中冒充该论文；统计主张改用R20官方摘要 |

Europe PMC完整搜索基址：`https://www.ebi.ac.uk/europepmc/webservices/rest/search`；
Crossref：`https://api.crossref.org/works/`；NCBI：
`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi`；
OpenAlex：`https://api.openalex.org/works`。

| 实验组 | 直接支持来源 | 尚未被文献证明的部分 |
|---|---|---|
| A 协议/身份 | R01、R04、R06、R07 | 实际2014随机化版本、104/105/106完整语义、物理membership、Header冲突 |
| B 基线 | R02–05、R08–10 | 本项目E0数值、确切原论文执行环境、当前管线差距来源 |
| C 时间/预处理 | R12、R14–17、项目输入代码 | 哪个窗口/滤波最优，真实在线显示与时钟延迟 |
| D 伪迹/空间 | R01、R12–13、R15、R18–19 | 本项目眼电贡献比例、产品八导最佳布局、ICA实际收益 |
| E 主干 | R08–10、作者实现 | 加宽/加深/新训练配方的收益 |
| F 聚合 | R01、R11、R14与数学反例 | 当前repetition独立性、ERP平均优效、真实候选身份和反事实停止收益 |
| G 校准 | R08、R10–11、现有SubjectAdapter | 最优标签预算、正则/BN策略、source-centered penalty收益 |
| H 迁移 | 当前代码和已完成dev配置核查；本轮未做EA/AS-MMD全面文献复审 | 所有具体机制在公平对照中的追加收益；不以实现存在作效力证据 |
| I 统计 | R20、R21定位、统计设计技能与现有统计接口 | 确认所需被试/seed数量，需SESOI及方差估计而非猜测 |
| J 产品 | R01、R11–13、R17及当前mission | 成人真实九选一、跨日、眼电独立归因和可靠性 |

## 6. 本轮没有闭合的工作

没有运行新的within-session、ICA、蒙太奇消融、源训练或成人实验。
没有修改已存在的算法、测试、数据合同，也没有把原分组假设直接删除。
文献矛盾与缺数据已转为目录的具体审计任务；binary基线和标签/时间分析可先推进。
对性能、机制效力和产品目标的裁决，仍需目录中规定的数据与实验结果。
