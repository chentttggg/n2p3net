# 统一模型分模块实施记录

本记录随实施更新。研究依据与完整计划见[科研方案](unified_model_plan_20260916.zh.md)。
用户要求每个较大独立模块结束时验证、整理工作树并物理冻结阶段；并行已有工作及原始数据保留。

## 初始代码冻结

- 实施前完整源码、测试、配置和六指南已压缩，见`frozen/unified_model_20260916_before/source.manifest.json`。
- 当前存在前轮与并行dirty工作；本轮不通过reset或删除旧数据制造干净工作树。

## 模型、实测通道与残差结构

已在`N2P3Net`原处增加实测通道归一化及共有残差。观测通道先pack，再经原temporal卷积和原BatchNorm，
最后scatter回物理槽位；缺导不进入卷积权重梯度，也不会被BN bias复活。
这个变换在数学上给出方案中的观测slot/time均值与方差，复用PyTorch的momentum、running variance及原重估hook，
无需增加自定义统计状态。SplitBN按同样pack后的域索引路由。CUDA动态pack吞吐仍待局部性能核验。

`FeatureResidualAdapter`直接复用，加入core的post-MST位置；只在新结构中启用。新参数在classifier之后初始化，
相同seed的core权重不因新模块消耗随机数而改变。六导候选为1279参数，首次前向与1202参数core逐位相同。
源和个人结构重建持有真实mask/residual声明，旧文件缺字段仍按其原结构读取。

个人适配的fit/validation/refit、BN重估、特征缓存和只读预测已显式传播mask；wrapper继续接收同一mask参数。
scratch配置提取发现`feature_residual`模块与同名配置的冲突，在`plain_trunk_configuration`原处修复为读取配置字段。

定向验证：`test_observed_channels.py` 7项通过，含前向/梯度/BN等价、NaN缺导、无观测失败、残差初始化及本人状态；
`test_structural_adaptation.py` 9项通过。既有N2P3Net模型测试在联动批次中通过；Ruff通过。
这些是软件与合成数据证据，尚未形成真实EEG效能或新训练矩阵结果。

模型阶段快照：`frozen/unified_model_20260916_models/source.manifest.json`。
此快照为分模块检查点，其他并行模块可能仍在集成，不作为整个方案已验收的声明。

## 下一轮统一模型的全局早停

用户同日修订：下一轮全量启用早停、共享全局patience，旧P1/P2保留。共同配置由
`train.early_stopping.GlobalEarlyStoppingPolicy`解释，初始patience6、min_delta1e-6、
source上限32个验证周期、target上限200遍；各拟合独立计数，不按被试/库/方法单独调节。
本节取代新矩阵此前固定4000/200更新的预算声明，不改写旧实验的执行身份。

source在每个来源库按真实participant留出，按内部验证CE选轮，再从原初始化全源refit选中曝光周期。
目标raw-prefix同时准备独立inner训练和验证算子，完整校准与共同later test仍各自处理；
`IndependentEpochSelectionData`核验原始支持、身份和合同，选轮后从原初始化在完整校准范围重训。
无合法验证划分明确失败；LDA和K0不参与早停。新统一入口拒绝固定更新与早停同时声明。

全局policy随source/target计划、checkpoint、恢复及缓存键传递；改patience不能误复用旧选择。
相同真实refit仍可复用；复用cell保留原停止证据，其本次physical updates/draws为0。
记录验证曲线、最佳轮次、停止原因、触顶标记及两阶段更新/抽样/墙钟。实际训练剂量允许不同，
比较的是同一停止规则下的完整程序；不是等更新数机制对照。调参只依据开发验证，最终test不参与。

并行早停模块已有37项独立反例，覆盖计数、改善重置、人员/原始信号隔离、从原初始化refit、缓存与物理成本。
其当时的源码与验证记录见
[源码清单](../../frozen/global_early_stopping_20260916_verified/source.manifest.json)及
[测试清单](../../frozen/global_early_stopping_20260916_verified/validation.manifest.json)。
这些是该阶段的验收身份；本轮统一程序最终检查另记，不将并行修改前的通过数移植为当前全套结果。

## 共有决策、源中心收缩与本人联合拟合

`MembershipDecision`统一真实event×candidate归属；NumPy/Torch共用累加语义，direct与row/column均可反传。
完整词表、原物理事件和QC后的`-1`继续保留。目标侧旧candidate参数组及epoch末额外optimizer步骤已从活动调用链删除；
唯一共同接口是`decision_groups/decision_targets/decision_weights`，D0与D1保持同工作集，单次更新只执行一次step。
跨标签预算边界的组在读取truth前排除，预算内剩余trial继续参与CE；冻结缺失组保留风险质量。

`SourceParameterPrior`集中在既有参数组module。权重尺度为源RMS²加扇入参考方差，bias/BN/gate用单位尺度，
按tensor均值再按结构角色均值求和。它是明确的参数坐标先验，不推测所有历史初始化，也不承诺函数重参数化不变。
先验中心/尺度摘要进入fit history；inner与fresh refit共用原源中心。验证覆盖零中心、冻结引用、零权重尺度、
有限梯度、无源参数突变以及真实拟合更新。此前方案关于“逐层实际初始化分布”的额外记录要求由此收敛。

独立验证：源中心先验7项、共同更新4项均通过；后者检查物理batch上限、一次step、D0/D1工作集相同、
预算外EEG/标签不影响结果及缺失请求。决策owner的82项核心、17项调用及61项迁移/分析分批通过，
这些集合有重叠，不相加作为独立总数。真实recipe→fit→单步joint探针亦通过。

BNCI的strict MAT parser与candidate builder已接raw scope，保留原事件ID及真实10轮声明。
timeline的complete定义为所声明窗口内完整台账；每候选原repeat连续即可保留非零起点，
完整candidate-chain与MembershipDecision继续独立核验从头的完整组。真实合成MAT集成中K200保留1个完整选择，
测试开头的截断组排除，后续33完整组保留；相关40项反例分批通过。

## 个人校准、早停与保存程序

用户明确要求统一模型改用全局早停。旧固定更新方案已物理压缩：
`frozen/unified_model_20260916_fixed_plan/manifest.json`。
复用并行工作树中的`GlobalEarlyStoppingPolicy/PatienceStopper`与既有`fit_with_epoch_selection`，
同一实验固定patience/min_delta/epoch上限，各次fit各自计数，不能共享参与者之间的patience状态。
source使用排他人员验证后fresh refit；target只在授权校准内选轮，再从原初始化完整C重拟合。
固定步数仍用于局部合成梯度/计数探针，不充当统一模型的真实训练预算。

个人`selector_mode=unified`消费一份明确TargetRecipe及全局早停策略，不做完整模型择优。
CLI提供`--calibration-recipe`和`--early-stopping-config`；保存包绑定sum评分、温度及严格coverage，
推理使用同一membership核且不读取目标标签。当前BrainSync入口要求分开的完整校准record用于早停；
公共单record则由独立raw inner scopes处理，不能通过裁已双向过滤的epoch模拟独立验证。
两项完整个人链测试（含CLI）及四项joint反例通过。源prior/个人旧路径77项相关回归通过；
随后发现的只读NumPy mask转换警告已在输入转换原处改为小mask显式copy。

决策与适配阶段快照：`frozen/unified_model_20260916_decision_adaptation/source.manifest.json`。

## 原始范围准备与源内开发

`RawSampleScope`保存原始样本半开区间与授权事件索引，先裁连续raw再调用原滤波owner；
必要响应尾窗可以含后续刺激，其标签仍不自动进入C。每K独立C、独立inner T/V与共同later test都有实际执行凭证。
多会话按显式session/run顺序组合，`target_raw_view_identity`保存有序子缓存身份；worker只加载本人的缓存。
BNCI入口复用严格MAT parser，保留真实10轮membership；其它无真实candidate合同的数据只具备binary研究资格。

实际三记录Huebner输入探针通过：固定test476行；K200的inner为160/35，K600为476/124。
这是读取真实raw并执行预处理的验证，没有训练模型。输入owner相关183项、后续CLI恢复3项及两项集成通过，
集合之间有重叠。详见[原始范围实施](../assets/unified_model_implementation_20260916/raw_scope_implementation.md)及
[真实raw摘要](../assets/unified_model_implementation_20260916/raw_scope_probe_summary.json)。
阶段源码见`frozen/unified_model_20260916_raw_scope/source.manifest.json`。

源阶段`unified_source`生成R/S/M八状态，实际架构、辅助特征位置与目标函数彼此核验。
`source_prior_selection.prefix_manifests`提供合法源内C200/query；先排除留出参与者，再拟合R1/S0/M0参考源。
各λ候选均在C200内选轮并完整重训，以独立query的seed均值、participant等权AUC选择同一外折公共λ。
缺少合法K200或任一请求失败时保留不可估计，不能借外层目标补选。候选、原cache、checkpoint、预测和成本摘要进入计划与恢复身份。
λ任务按person/seed复用已有spawn/GPU/内存调度，单worker内部顺序遍历λ以保持随机初始化对应关系。

联合源目标/runtime、checkpoint及目标矩阵已通过分批验证，含本地CUDA联合损失探针和真实双worker λ合成执行。
相关生命周期套件152项通过；一项旧测试硬要求2worker，在实际可用内存仅约619MB时错误拒绝正确的单worker准入，
已改为核验实际准入与完成任务数，独立复测2项通过。详情与各批次边界见
[源运行实施](../assets/unified_model_research_20260916/source_runtime_implementation.md)。
阶段源码见`frozen/unified_model_20260916_source_training/source.manifest.json`。

## 最终数学与验收口径

源目标为

\[
\mathcal L_S=\mathbb E_{i\sim p}\left[\frac{r_i}{p_i}\,
\operatorname{CE}(f_\theta(x_i),y_i)\right]
+\alpha\mathcal L_{\mathrm{SupCon}}+\beta\mathcal L_{\mathrm{cMMD}}.
\]

辅助项在owner内各乘一次系数；同一更新共用特征。条件MMD沿用无偏估计，有限样本下允许负值。
个人更新为

\[
\widehat{\mathcal L}_C=\widehat{\mathcal R}_{trial}
+\gamma I_d\operatorname{CE}(s_d,y_d)+\lambda\Omega(\theta;\theta_S),
\quad d\sim q_d,
\qquad s_d=\log\pi+A_d^\top(r_d\odot e_d)/\tau.
\]

其中\(q_d\)按participant、原始decision逐级等权；\(I_d\)只由冻结台账与可观测性决定。
不完整组\(I_d=0\)仍占原风险质量；同分组的CE仍可训练。D0保留同一group工作集，令\(\gamma=0\)。
物理batch上限由micro-forward保证，trial/decision/prior只执行一次backward与optimizer step；联合梯度不增加第二次更新。
\(r_d\)是声明的事件权重，与trial采样风险\(r_i\)属于不同层次；不能把二者重复乘进CE。

源中心罚项按结构角色\(g\)及其参数tensor集合\(P_g\)定义

\[
\Omega=\sum_g\frac1{|P_g|}\sum_{p\in P_g}
\frac{\|\theta_p-\theta_{S,p}\|_2^2}{2n_p s_p^2},\qquad
s_p^2=\begin{cases}
\operatorname{mean}(\theta_{S,p}^2)+1/(3\,fan\_in_p),&p\text{为Conv/Linear weight},\\
1,&p\text{为bias、BN affine或gate}.
\end{cases}
\]

中心和尺度始终取原source；fresh refit不以inner赢家作新中心。目标仍复用原AdamW，源仍复用原Adam。
这是参数坐标正则，不能声称对等价网络重参数化不变。

早停每轮比较验证CE：若\(v_t<b-\delta\)，更新\(b\leftarrow v_t,t_*\leftarrow t\)并清零计数，
否则计数加一；连续\(P\)次无改善即停止，最大轮数为\(T_{max}\)。每次拟合独立持有计数，
选中\(t_*\)后从同一原初始化在完整合法范围重训\(t_*\)轮。当前monitor为trial CE，
尚未验证是否优于真实decision级验证损失；这项选择进入整个程序的效能评估。

最终验收区分：数学与软件反例、真实raw范围探针、公开数据完整训练、成人新测试。
本次交付前两层；没有启动新的完整真实矩阵或成人确认。

## 独立复核与整体收尾

复核发现并原处修复了以下调用偏差：

- unified个人预检查沿用旧R5选模门槛，与实际验证CE选轮不一致；现在预检与执行共用CE口径，
  CLI默认调用正常，显式传入旧选择指标给出参数错误。R5校准、R1验证的实际API/CLI反例通过。
- 源内λ参考模型此前忽略自定义残差结构；现在从同一计划构造R1/S0/M0参考，非默认残差反例通过。
- 逐K原始准备中的实际预算回退须保留requested/effective及原因，不能被统一矩阵误当成越界预算；
  实际prefix视图须与target cache绑定同一authority、study及固定test证据，防止同编号跨库误配。

对joint适配的独立复核与实际probe确认：跨预算组先排除再核验目标，QC后全缺组仍保留原风险；
inner/refit都围绕原source中心，inner QC仅拟合内部训练，final QC在完整校准范围重新拟合。
联合损失没有第二次optimizer step，保存包重载保持分数及质量状态。

第一轮全套为2082通过、4跳过、1失败（320.09秒）。失败测试把单次非零原始repeat错误当作断档，
与新的scoped台账语义冲突；现保留非零起点并用同候选多次观测构造真实缺口，同时检验其不能冒充完整candidate chain。
开启QC的新增保存重放测试还曾因原生tuple与JSON list直接比较失败，改为比较实际序列化合同。
上述台账、QC、CE预检及个人链定向复测21项通过（11.11秒）。最终状态另记，旧日志保留供追溯。

2026-09-16最终收尾：矩阵入口新增反例10项、实际source→target调用集成1项通过。
代码稳定后冻结`frozen/unified_model_20260916_validation/source.manifest.json`，从该工作树运行全套：
**2095 passed、4 skipped，294.48秒，exit 0**。验收期间冻结的源码、测试、配置与指南逐文件无变化；
Ruff检查`src tests experiments`和标准`git diff --check`均exit 0。
这些数字包含既有项目回归；联合模型的数学和输入范围证据仍以各定向反例为准。

最终源码、测试、六指南与配置见
[最终源码清单](../../frozen/unified_model_20260916_acceptance/source.manifest.json)，
本记录、完整科研方案、研究资产、测试日志/XML和结构化验收见
[证据清单](../../frozen/unified_model_20260916_acceptance/validation.manifest.json)。
每份压缩包均核验整体SHA-256及成员摘要；最终代码与执行测试的源码一致，文档仅补充本段结果。
并行已有工作、原始数据和历史结果按原层级保留，未将源/目标短探针表述为全矩阵成绩。
