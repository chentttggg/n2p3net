# 统一模型：分阶段停止与精简训练方案

日期：2026-09-16。用户最新要求允许按科研证据决定停止方法，优先节省云端时间。
本方案替换全体神经网络强制早停及R/S/M/D的16格默认排程。原方案和测试已物理冻结，见
[变更前源码](../../frozen/unified_compact_training_20260916_before/source.manifest.json)及
[旧排程](../../frozen/unified_compact_training_20260916_before/previous_program.manifest.json)。
共享模型和损失数学沿用[统一模型推导](unified_model_plan_20260916.zh.md)。训练设计、测试整理以及随后预算/性能实施均已完成；实际状态见[执行器验收](unified_budget_execution_20260916.zh.md)。未启动云训练。

## 1. 为什么改变停止方法

现有123人公开数据实验中，个人CE早停加完整校准重训，相对固定60轮的full AUC差为
K200 −0.0219、K800 −0.0243、K2000 −0.0150；总更新从32,220增至42,359，增加31.5%。
连完整验证轨迹选轮也未超过固定预算，不能只归因于patience太小。
固定90轮相对60轮的full AUC增量为+0.00861/+0.00964/+0.00135，scratch也明显改善。
依据见[早停开发结果](target_early_stopping_results_20260908.zh.md)和[90轮开发结果](target_epoch90_20260908.zh.md)。
这些来自旧输入/旧适配程序，支持一个有证据的起点，不能证明新联合模型90轮最优。

小验证集的误差方差及反复选模的偏差有外部方法依据[1,2]；早停本身是训练时长超参数选择，
选轮后重训的额外成本及“固定更新还是固定数据遍历”也须明确[3]。故只在具有多参与者验证的源内参考阶段选轮，
其余阶段执行预先冻结的预算。固定预算使用全部个人校准数据，并不要求额外切出个人验证集。

| 阶段 | 停止规则 | 拟合与成本 |
|---|---|---|
| 每外折1个源内参考模型 | participant验证CE早停：patience=3、min_delta=1e-4、最多12个曝光周期 | 一次inner选择，再在排除λ开发人员的全部源上重训选中轮数 |
| 两个源配方×三个seed | 使用该折参考模型选出的同一个曝光周期数e* | 每模型从其原seed初始化直接拟合全部合法源；无重复inner |
| 源内λ开发 | 每λ固定90轮，四个候选 | 每人直接拟合C200，独立later query只负责选择λ |
| 个人迁移与scratch | 固定90轮 | 完整C_K一次拟合；不做个人选轮后第二次重训 |
| 波形LDA、BN重估、输入统计 | 各执行一次相应估计 | 没有迭代早停语义；同输入的确定性LDA跨seed复用 |
| K0及未知目标推理 | 只读前向 | 零优化更新 |

源参考使用联合源目标A=1。共享e*让两个源配方有相同曝光预算，但预算由A=1参考选择；
对比是在此共同预算规则下的程序比较，不宣称分别调优后的两个方法性能上限。
三个seed是在冻结e*/λ条件下的优化重复，不是三次独立超参数搜索。

## 2. 数据与固定模型

源与目标执行完整显式产品profile：128Hz、0.1–30Hz、[-200,1200)ms、179点、V单位、
均值基线和已声明50Hz/Q30双向处理，实际参数由`configs/brainsync_bidirectional_preprocessing.json`读取。
首轮用Cz六差分槽及真实mask，固定共有残差R=1的1279参数图；旧因果cache不能重标后复用。
未测电极仍为缺失，坏Cz使依赖它的差分不可用。

公共开发保留四个整库留出折和一个BI家族排除敏感性折。家族折的同一对源模型分别评价BI2014a/BI2015a，
结果另列，不能当成第五独立研究库。源与目标三个seed成对，沿用现有完整流程seed声明，不另做Cartesian乘积。

主预算为K0/200/800。K按原事件授权数计；每K的C_K分别裁raw后处理，later test由该参与者经预声明回退后的最大实际授权范围一次固定。
训练、QC、normalizer和BN只消费C_K。短记录沿用已实现的有限数据回退，requested/effective/retained分别列出；
精确K和回退K不混成同一个学习曲线点。实际队列必须由新profile清单重新枚举。
采用固定个人预算后不生成未消费的inner T/V滤波缓存；源内λ开发只需其合法C200/query。

源优化器Adam，个人AdamW，学习率1e-3、weight decay=1e-4；不增加调参维度。
源沿用participant/domain风险与真实抽样，batch上限512，dropout=0.25；个人dropout关闭，
按C_K重估一次BN buffer后冻结，BN affine和其余学习参数共同更新，batch为min(512,n_fit)。
随机初始化、mask、source prior与联合更新继续复用当前实现。

## 3. 只保留四个组合和两个必要基线

A表示源目标包，Q表示个人适配包；二者是预声明的组合处理。

| 处理 | 源目标 | 本人目标 | 回答的问题 |
|---|---|---|---|
| A0Q0 | CE | trial CE，λ=0，D0同工作集 | 同结构完整微调锚点 |
| A1Q0 | CE+0.05 SupCon+0.05条件MMD | trial CE，λ=0，D0同工作集 | 联合源目标的贡献 |
| A0Q1 | CE | trial CE+λ*源中心罚项+真实decision CE | 个人适配包在普通源上的贡献 |
| A1Q1 | CE+0.05 SupCon+0.05条件MMD | trial CE+λ*源中心罚项+真实decision CE | 完整统一程序 |
| Scratch | 同结构随机初始化 | 与Q1相同decision监督，λ=0 | 迁移是否优于本人从头训练 |
| 波形LDA | 无源拟合 | 合法C_K上的一次收缩LDA | 低成本经典基线 |

SupCon temperature=0.1、最小有效cell样本数2，特征取post-residual；沿用已验证的去重与跨真实域/人的配对。
decision系数γ=1、评分温度τ=1作为固定开发起点；只有真实membership才启用。
二分类数据的Q1只加入源中心罚项，D记N/A；不从binary标签制造九选候选。
先按授权C_K冻结完整物理排程组；QC后证据不全的已请求组保留原风险权重，跨预算或排程不全的组不补齐。
Q0的γ=0仍消费同组工作集，Q1与Q0均只执行一次联合step。

此表保留所有当前核心模块在完整组合中的机会。它能估计两个处理包的交互，不能单独归因SupCon、MMD、
残差、收缩和decision各自的效果。残差固定开启，首轮不回答其单独贡献；后续只为明确未决问题增加成对处理。
如果λ*=0且D不适用，Q0/Q1数学程序相同，只做一次物理拟合并记录退化；不可把重复结果当成新的实验信息。

令Y_aq为同一库/人/K/seed下的指标，预声明

\[
\Delta_{\rm unified}=Y_{11}-Y_{00},\qquad
I_{A,Q}=Y_{11}-Y_{10}-Y_{01}+Y_{00},\qquad
\Delta_{\rm transfer}=Y_{11}-Y_{scratch}.
\]

两因素设计矩阵列为(1,A,Q,AQ)，秩为4[4]。例如Y00=.70、Y10=.69、Y01=.69、Y11=.76，
完整组合提高.06而交互为.08；该四格仍能发现单开无益但组合有益的现象。
这里把原模块明确合成两个包，不能把四格冒称原四因素的无混淆压缩设计。

## 4. 源参考、共享预算和λ选择

对外折f的合法源人员S_f，先按冻结身份seed在每来源库选2名具备合法C200/query的人员组成H_f。
普通折3个来源库，所以每折6人；BI家族排除折2库，所以4人；合计28个人员×外折开发单位。
资格只由真实身份、标签预算、物理时间与最低两类支持决定，不按模型分数筛选。
不足时明确该折开发不可估计，不能从外层目标借人或借统计。

参考源只在S_f\H_f内进一步按参与者划出20%的V_stop，T_stop与V_stop每个来源库都须有合法支持。
H_f不进入参考源的梯度、normalizer、BN或早停验证。参考A1从固定开发seed初始化；
每\(q_T=\lceil n_{Tstop}/512\rceil\)次有放回更新称为一个曝光周期，验证一次分层风险加权CE。
若\(v_t<b-10^{-4}\)则记录最佳轮并清零未改善计数，否则加一；连续3次无改善或达到12周期停止。
从原初始化在S_f\H_f重新拟合e*个曝光周期，形成λ开发的参考checkpoint。
source CE验证曲线、辅助有效暴露、触顶原因与最终权重均保存。触顶且仍改善的情况报告预算受限，不据此宣判模型无效。

对每个h∈H_f，用其C200训练90轮，再在独立query评估。只搜索
\(\lambda\in\{0,10^{-3},10^{-2},10^{-1}\}\)，保持其他参数和seed固定。
有真实membership时用Q1完整目标；其余为trial CE+prior。选择准则为来源库等权、库内人员等权的query AUC：

\[
\lambda_f^*=\arg\max_{\lambda\in\Lambda}
\frac1{|D_f|}\sum_{d\in D_f}\frac1{|H_{f,d}|}\sum_{h\in H_{f,d}}AUC_{h,query}(\lambda).
\]

分数在1e-6内并列选较小λ；失败单元不删除后重算均值，缺任一预声明单元则该λ不具选方资格。
λ=0使是否需要收缩也成为可证伪问题；不能预先保证正收缩有益。
默认112次小规模个人开发拟合，没有60/90×λ网格，也没有每个seed重选λ。

最终30个源模型按各自外折使用冻结e*，从三个原始seed在全部S_f上训练。
每个source的prior中心和尺度来自其自身最终权重；λ_f*在A0/A1间共用。
H_f的重新加入只发生在已经完成源内选择的最终源拟合；外层目标始终排除。
\(U_{f,main}=e_f^*\lceil n_{S_f}/512\rceil\)，不是把inner更新数原样复制成全源曝光。
选轮预算在A1上开发的偏向和仅2人/库的开发方差应随报告披露；它节省搜索，不提供最优性保证。

## 5. 实际任务数与复用

源最终模型为\(2\times5\times3=30\)，参考源模型5个；共35个保存的source artifact，
40次物理source fit（5 inner+5 reference refit+30 main），而旧完整R/S/M表仅主源就有120个artifact。
有放回每周期不保证每行看过一次，实际updates/draws/unique rows必须分别记录。

设E_o为四个普通折的不同person×有效正K数量，E_f为BI家族敏感性折的相应数量。
同输入scratch跨源和家族复用；固定90轮使它不依赖各折的e*/λ。
波形LDA为确定性拟合，在原始范围、特征、质量、实现和数值环境相同时跨seed也复用。

\[
N_{NN}=3(5E_o+4E_f),\qquad N_{LDA}=E_o,\qquad N_{\lambda}=112.
\]

这是去重前Q0/Q1仍不同的保守拟合计数，不包含K0前向或软件故障重试。
每个逻辑请求都留记录；复用只减少物理工作，不增加参与者或seed的独立性。
以旧队列的K≤800回退清单作估算，E_o=246、E_f=204，则main NN≤6138、LDA=246，加λ开发112。
只看旧精确K200/800则E_o=182、E_f=140，main NN4410、LDA182。两者均须由新raw资格重新枚举，不能当成已准入数量。

## 6. 计算排程与时间估算

时间依据及场景计算见[云端计时核查](../assets/unified_compact_training_20260916/cloud_timing_audit.md)。
以历史RTX5090约32GiB、有效CPU配额和内存限制为参照；启动时重新探测当前实例，不能从显卡名称推出当前吞吐。
先准备一次raw/profile和逐K缓存，再按外折执行reference→冻结e*/λ→main sources→共享个人worker→评分/重放。
不同外折依赖互不等待；独立CPU准备与GPU训练可重叠，但不得超过实际内存准入。
source先从4个GPU worker、每worker一个计算线程的已测配置起步；target从同一保守并发起步，
短代表任务测量完成率后再选择4/8worker，保留batch≤512。实测不支持的并发不写成节省时间承诺。

代表探针覆盖A0、A1、Q0/Q1和缺导pack，包含加载、输入统计、BN、保存与重放；
先分开记录冷编译和热执行，再冻结实际运行方式。当前动态observed pack没有CUDA Graph吞吐证据，
不能照搬旧plain模型的compile加速比。探针的成功真实任务可按同一执行身份计入完整排程，避免重训。

\[
T_{total}=T_{prepare}+T_{reference}+T_{source-main}+T_{calibration}+T_{replay/IO}.
\]

每阶段按GPU worker秒、实际并发效率及串行依赖估算；不能简单用模型数量乘单模型中位数后除以worker数。
源步数在历史五折\(\sum_f q_f=2246\)下，即使没有提前停止也满足
\(U_{main}\le6\times12\times2246=161712\)，
参考inner+refit的保守界为\(2\times12\times2246=53904\)，合计≤215616。
它略高于旧P1/P2的180000源更新；节省主要来自减少完整源状态、目标程序及重复个人inner/refit，不能只凭35<45声称源更快。

历史资源为RTX5090约32GiB、25个有效CPU核、90GiB内存；本次未连接云实例，当前资源尚未实查。
旧P1/P2完整matrix墙钟110.83分钟，其中目标阶段约94.26分钟；不能把总时间全部计给目标。
按逐人90轮实测时间重新加权到新程序，再给未测的pack/联合损失留1.25–3倍工程余量，排期为：

| 缓存准备就绪后的阶段 | 时间区间 |
|---|---:|
| 5个源参考及30个主源 | 20–50分钟 |
| 112次λ开发拟合 | 2–6分钟 |
| 目标NN及一次LDA，保留实际K回退 | 60–140分钟 |
| 重放、统计与归档 | 5–15分钟 |
| 合计 | **87–211分钟，约1.5–3.5小时** |

这是排期区间，不是新程序实测或统计置信区间。首次raw重建尚无全库吞吐，
总时长须再加T_raw与缺失数据传输时间；现有三记录探针不能给出这一项的可信上界。
若仅列精确K200/800的历史参照，训练约67–166分钟，但本方案保留真实回退请求，不为缩短估计删除这些人。

## 7. 评价和执行前的精确联动

主终点为精确K200的库等权participant-macro AUC，K800及实际回退档单列。
完整决策数据另报operational hit@R5、覆盖、NLL/Brier、并列及失败数；binary-only不虚构决策指标。
三个配对seed先在人内平均，按研究库再等权；家族排除单列。主对比族固定为unified−A0Q0、unified−scratch、A×Q交互，
报告配对区间、负迁移比例/尾部和失败；需要显著性时对该族作Holm校正，不追加看结果后选择的对比。
当前四库及P001均是开发证据，成人未知目标的新参与者/跨日确认另行冻结。

下表联动已在现有owner完成；真实调用、默认配置与GPU计时见[执行器验收](unified_budget_execution_20260916.zh.md)：

| 现有owner | 精确变更 | 必验反例 |
|---|---|---|
| `train.early_stopping`、source selection/chain | 参考源拥有patience；输出带source范围/配置摘要的e*预算；main消费冻结周期直接fit | 冻结H身份后，其数据不能影响reference状态/e*；外层目标不进入任何fit；主A0/A1同曝光 |
| `research.source_matrix`、domain matrix | 显式选择两种R=1源条件和2×2个人包，source reference可复用已有fit/refit | 只枚举35个source artifact；Q退化去重不丢逻辑请求 |
| `source_prior_selection` | 每库2人，四λ含0，固定90轮、真实Q1与库等权选择 | query人完全排除reference训练/早停；失败不缩分母；λ0合法 |
| `target_experiments`、`calibrated`、CLI | 消费固定90轮target budget；直接完整C fit，单份完整BrainSync校准录制允许进入 | 不调用个人inner/refit；QC/BN/prior均只fit全C；保存重放一致 |
| `source_records`、prefix CLI | 只有实际需要target early stopping的研究处理才生成inner T/V | 固定模式不生成未消费视图；各C_K/test原始时间隔离继续成立 |
| `result_reuse`、checkpoint/report | 将参考预算、λ选择和实际模式纳入身份；确定性LDA复用不受无关seed影响 | 改budget/输入/监督不能命中旧模型；重复请求physical cost=0 |

以上是同一模型的训练生命周期配置，不新增按测试成绩动态挑模块的推理分支。
实施已按源预算、目标一次拟合、数据范围、缓存身份四组反例验收，并进行全套回归。

## 8. 测试整理

已实际整理`tests/data/`及统一个人适配/产品链，生产源码未变。修改前全部代码与测试已物理冻结。
旧10个测试文件归入对应owner，新增1个集中candidate测试文件；测试文件总数144→135，
测试函数1405→1392。按被测接口合并，并将一次真实拟合的多个不变量一起核验。
数据范围、身份、数学梯度、真实membership、失败分母及保存重放仍有独立反例。

相关个人测试成功NN拟合105→89、AdamW更新294→254；prefix准备5→1次，重复EOG校准5→1次。
两个目录合计净减233行测试及辅助代码。定向集合分别58项、162项通过，后续补充反例另记；
这些集合有重叠，不相加当作全套结果。一次墙钟观察不宣称稳定加速比。
映射、删除理由及实际调用计数见[个人测试整理](../assets/unified_compact_training_20260916/tests_transfer_consolidation.md)
和[数据测试整理](../assets/unified_compact_training_20260916/tests_data_consolidation.md)。
旧全局早停等仍有实际研究调用方的生命周期测试继续验证；新unified测试已经切换为分阶段预算合同。

整理后的最终全套为**2079 passed、4 skipped，281.38秒，exit 0**，收集项2099→2083；
Ruff和`git diff --check`通过。4个skip仍为两个独立工作站入口未配置及两个Windows符号链接权限场景。
本次未增加skip或减少环境准入。前后单次全套耗时不构成稳定性能基准，节省的真实拟合/准备次数另有直接计数。
与修改前物理快照比对，生产源码、实验入口、配置及依赖文件未变。
最终源码/测试和文档见[源码清单](../../frozen/unified_compact_training_20260916_acceptance/source.manifest.json)，
方案、文献响应、成本与测试映射、完整日志见[证据清单](../../frozen/unified_compact_training_20260916_acceptance/evidence.manifest.json)。

## 9. 文献核验边界

1. Varoquaux, 2018, *Cross-validation failure: Small sample sizes lead to large error bars*,
   [DOI](https://doi.org/10.1016/j.neuroimage.2017.06.061)。本次核验Europe PMC摘要；支持小样本验证方差问题，不直接证明本P300程序的停止轮数。
2. Cawley & Talbot, 2010, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*,
   [JMLR](https://jmlr.org/papers/v11/cawley10a.html)。本次核验出版方摘要；支持限制反复选模和隔离外层评价。
3. Goodfellow, Bengio & Courville, *Deep Learning*, §7.8，[官方全文](https://www.deeplearningbook.org/contents/regularization.html)。
   本次读取early stopping及Algorithms7.1–7.3；原文明确重训时“相同步数还是相同遍历数”存在选择，本方案显式采用曝光周期。
4. NIST/SEMATECH e-Handbook, [Full factorial designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri333.htm)。
   两因素四格枚举依据；两包组合的科学定义与未识别的细分作用由本方案承担。

检索响应、来源摘要和数学验证保存在`doc/assets/unified_compact_training_20260916/`。
