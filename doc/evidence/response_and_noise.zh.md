# 响应模型、空间先验与噪声数值

本文按主题维护已复核的结论、代码接点和下一步。运行日期与源码身份留在各证据及绑定中；使用方式和工程规则由原六指南维护。

## P1完整16格读数

完整16格已完成并独审：8人各一个held selection05、R10、36选1，两噪声臂共576个候选分数，seed20260918，新增fit及response-fit均0。科学exit0、所属树0；[完成绑定](../assets/research_resume_20261011/p1_cloud_completion_binding.json)保原producer、新consumer、结果和两路独立复算。此处覆盖下文“A01已恢复、余14格待评”的阶段状态，原失败面板继续保原身份。

| 噪声臂 | 正确/请求 | hit | NLL | Brier | 请求覆盖 |
|---|---:|---:|---:|---:|---:|
| shared8 | 5/8 | 62.5% | 7.336477 | 0.679094 | 8/8 |
| private8 | 6/8 | 75.0% | 5.554012 | 0.477678 | 8/8 |

private仅救回A06，其余七人命中状态相同。A01贡献约71.82%的NLL改善；其真值概率从8.72e-22变为2.45e-17，两臂仍为近100%置信错误。Brier改善约75.71%来自A06，真值概率从0.203527变为0.925032。因此当前研究优先级是高置信错误尾部与C内概率校准，并用多selection/类别扩展固定响应噪声对照。

新旧stationary law的16格赢家全部相同，最大归一化概率变化约2.45e-8。private原密度修正接近候选共同偏移，softmax会抵消共同常数；quadratic与predictive logdet实际改变。A06救回在旧law中已经存在，属于噪声臂差异。数值修复完成了initial-state正确性、保存准入与完整执行闭环，其效能评价继续绑定真实候选概率。

原观察后八人配对bootstrap的95% Δhit区间为[0,+37.5]个百分点；仅一个discordant pair，精确双侧McNemar p=1。每人一项且八项真值均为同一候选类别，下一实验预声明多selection和类别。2026-10-13按保存报告的舍入表复算尾部贡献及均值，记录在[当前读数](../assets/research_resume_20261011/cloud_completion_readout.json)；原始大载荷留存待核回，没有新增原分数重放。

## 当前模型与数值任务

信息界诊断的`/2`已完成数学与状态审阅：保同C估计mean/inverse-shrinkage-covariance的raw plug-in d²，并完整保存实际或显式canonical activation及全部候选pair geometry。在已声明的独立flash、共同已知covariance和uniform prior模型下，pairwise error为`Phi(-sqrt(d²*h)/2)`；相同activation列按uniform tie，有限Monte Carlo另记录seed、trial数及相关candidate的trial-level误差。`Q=3/F=1`的解析高信噪极限为`2/3`，旧严格比较得到`1/3`的错误已修。候选曝光均衡与line-cycle均衡分列，`Q=9/F=45`每候选均15次，但line counts为8/8/8/7/7/7。

`ExactPropagatedCovariance/2`保存真实有限F对应的`F K_AR F.T`，SPD密度用无ridge的factor；显式正交支撑Q拥有自己的Euclidean测度，支撑外残差拒绝。数值分解未解决时保raw诊断，factor/密度明确不可用；ridge只通过独立显式诊断入口。白化偏差按`L_AR^-1 K_prop L_AR^-T`计算，修正旧square-root chart误差。信息界报告的完整JSON保存/load/model-only replay已闭合；exact类当前是内存诊断接口。相关验证与两轴独审分来源记录在[数学整合绑定](../assets/research_resume_20261012/diagnostic_mathematics_integration.json)，没有新增公共数据实验。

固定响应的完整物理factor由保存的M、上三角precision R和原Sigma导出：`F=kron(R^-1,chol(Sigma_ref))`。它保持common/increment系数及channel坐标，噪声变化时复用同一物理响应分布。

八人shared/private AR8、共同prefix32的16格native诊断已完成：每人600标签fit、120标签held评价、80授权标签未使用，8 anchor/16 noise fits均保存加载。16个held评分均触发数值可分辨性门，概率覆盖0；完整保存态已零拟合重现原失败。[执行绑定](../assets/research_resume_20261011/p1_native_panel_binding.json)保存原来源和完整分母。

数值稳定化已接受并接入科研树：full diffuse quotient保无量纲source-scale界，conditional score保nats界，仍用原两道阈值。实现包含prefix nuisance可逆换图、实际256-row短叶/平衡reduction、真实prefix/tail与列误差传播，以及完整posterior metric中的局部位移界。`Q.T b`显式按`abs(Q).T abs(b)`计算dot误差，logdet的factor2与raw normalizer/输出舍入都入账。

独立审阅核完整AR历史、flat测度/Jacobian、相关proper factor、inverse solve defect、prefix压缩和后验位移。109个独立案例通过；54个conditional案例中32获准、22具名拒绝，获准案例的110位oracle最大误差`1.395e-9 nats`且在各自估计界内。root逐读全部差异及最后finite保护，按原字节接入`5b827aa4…`，核依赖/证据SHA并通过六个新增反例、Ruff与diff检查，见[数值接受绑定](../assets/research_resume_20261011/p1_numerical_acceptance_binding.json)。

八人两臂的16格保存态续评已完成并独立复算，新增fit为0。A01的72候选保`22bf6782…`原评分来源，其余14格504候选由`5b827aa4…`评分；所有576候选满足原两道误差门。shared命中5/8、NLL7.336477、Brier0.679094；private命中6/8、NLL5.554012、Brier0.477679。private救回A06，未损害原正确决策，配对NLL下降1.782464；5/8人的NLL改善，其中A01贡献总改善的71.8%。两臂NLL都高于36候选均匀参考3.583519，严重错判的概率风险继续进入后续模型判断。

本面板每人仅一个C-held decision，八个预声明真值均为row5/col0；结果解释原冻结compiled-W Gaussian程序。真实平稳law修正采用新的producer另做同支持配对，不从本次held结果更改response、先验或候选。终态/独立逐概率复算/CSV与图见[续评接受绑定](../assets/research_resume_20261011/p1_numeric_panel_acceptance_binding.json)，原16格数值失败面板及两个执行失败run保持原身份。

这次A01消费的一次本机CPU采样为80.9%，按实际记录为超限；后续native1且检查总机余量，最多两个score workers。完整数值程序和原producer分开绑定；tree dot也改变新C-fit的舍入路径，原prior和统计模型保持。此处交付数值实现与真实保存态可用性，后续同物理支持的概率分析继续用原requested owner。

Windows控制文件读写竞态已独立复现，修复后原8份完整receipt逐字节复用、未封存A04/A05 private journal归档并重算，仅剩余8格新评分。实际续跑exit0、进程全部退役，CPU峰值44.6%；原terminal的64.1%对应整个历史追加日志。执行器的parent身份混搭缺口由最终独立检查器以原8份receipt SHA和固定script/config pair关闭，执行源码保持其真实历史字节。

真实 stationary-law 新producer 的 stage04 首次启动在严格旧capsule export 阶段发现冻结入口把 manifest 字符串传给只接受 `Path` 的 `BoundInput`，在 export/root compile/fit/score 前退出，owned tree 清理为0。stage05 已修复 export 与 worker 两个实际 caller，并以独立冻结 source/plan 通过 A01 原capsule strict export、独立审阅及7项 Path/资源生命周期验证；代码已整合到科研树。新实验保持原A01–A08两臂16格、seed20260918、C600/held120/unused80与完整原response/noise参数，只编译新的 `/3` stationary prefix root。

stage05 两次、stage06 一次本地实际执行均由 RAM 门记为 `resource_blocked`，exit1，owned tree 全部清理为0。三次都只有 A01 一个 active actor，已实际编译 shared8 partial root并完成1/5/4个候选的conditional resolution；完整score与cell receipt均为0，新增fit为0。stage05的RSS骤降与非同步采样是待核解释；stage06则是available RAM降至约1.46GiB，按reserve/growth公式触发负余量。单worker重规划没有闭合本地完整执行，后续交云窗口按真实资源与平台合同执行同一16格。旧 `/2` panel、三次资源终态和所有partial保持原身份，实际计数与交接见[执行绑定](../assets/research_resume_20261011/stationary_program_execution_binding.json)。

## 平稳前缀的真实模型关系

续评发现private旧入口的线程依赖：A01原保存root在native2重编逐bit相同并通过原关系门，native1重编却拒绝。更深入的两个独立100位oracle对全部36个cross-channel pair求真实平稳covariance，确认问题在旧companion Lyapunov/lag-power编译；全256维root的真实covariance pullback谱差为`0.0772181`，Frobenius差`0.167282`，而raw covariance相对差仅约`1.7583e-9`。原`1.6e-7`门核的是旧数值重编关系，弱方向需要真实law oracle。

尾部实际消费stored PARCOR lattice，新模型以每个binary64 PARCOR作为精确实数，经实数step-up确定AR；跨channel创新Sigma只入一次。完整pairwise mixed Yule–Walker与独立p² Sylvester在36个pair上一致；unequal AR1 closed form和diagonal scalar Yule–Walker另作参照。同真实law的高精度joint Cholesky仅舍入为binary64后，谱差`3.409e-11`、Frobenius差`1.109e-10`，支持保持原门实现稳定编译。

新compiler已由独立树完成并合入科研主树：`stationary_noise_artifact`以保存PARCOR的exact-real step-up、mixed Yule--Walker和joint Decimal96 Cholesky定义真实平稳law；saved root用独立向上误差证书验收，load不重编root。`/3`新state与旧`/2` compiled-W严格分开，旧P1续评仍只消费旧producer和旧law，不追签历史结果。A01新root的独立MP100 pullback谱/Frobenius误差为`3.409e-11`/`1.109e-10`，旧root为`0.0772181`/`0.167282`；stage04 underflow certificate、四种变换、save/load及实际caller均通过，0 fit/score。生产新state仍需绑定自己的producer和scope后再进入新实验。推导、两套oracle、实际旧state与源码绑定见[诊断绑定](../assets/research_resume_20261011/stationary_prefix_diagnosis_binding.json)及作者最终交接。

原Windows严格loader已导出全部8人的原响应与16份完整噪声状态。A01复用既有封存出口，A02–A08七个独立进程均exit0；最终33成员包为5,602,250 B，逐成员摘要、原130输入/158源码/158 capsule及完整请求经独立复审。原导出计划保持`056977…`，与新的云评分计划分别绑定；根窗口复核最终包33成员并写入[跨主机出口绑定](../assets/research_resume_20261011/stationary_crosshost_export_binding.json)。原`/2` strict load包含旧reference-root关系重编，新`/3` root编译、fit与held score均为0。

云科学接线已完成原导出计划对held样本/设计的独立授权及跨attempt互斥，修后独审33项与科研树集成5项分别通过，见[科学接受绑定](../assets/research_resume_20261011/stationary_cloud_scientific_acceptance_binding.json)。合法原A01可进入native5数值池，重签错误样本/设计及原输出争用均在计算前拒绝；原Linux旧loader失败与本地三次RAM失败保留自己的来源。

同支持的新stationary law云端16格已完成：8人×`shared8/private8`、每cell 36候选，共576 slots；每人一个`selection05` held decision，truth均为`row:5|column:0`。科学进程真实exit0，owned processes after exit为0；外层七项queue仍exit1，作为独立队列事实保留。两臂均prediction/probability coverage 8/8：shared命中5/8、NLL `7.3364765832`、Brier `0.6790937536`；private命中6/8、NLL `5.5540121013`、Brier `0.4776784949`。private相对shared救回A06、没有损害正确决策，paired NLL下降`1.7824644819`。A01贡献约71.82%的NLL改善但仍是接近1置信度的错误；A06贡献约75.71%的Brier改善。新旧stationary law在同一16格的winner完全不变；shared最大log-probability差`2.33e-10`，private最大差`8.86e-7`，实际概率最大差约`3.69e-11/2.45e-8`，law替换不是这次shared/private差异的来源。每cell fit/response-fit均为0，原response/noise与held支持保持。逐cell差异、terminal和原始result/queue SHA见[分析绑定](../assets/research_resume_20261012/p1_stationary_analysis.json)。

## P1 数值门诊断与小组协议评估

数值证据的第一、第二道门分别记录。前一道需要full-record quotient的正确算术估计，后一道需要prefix/tail各自的input与压缩误差；prefix basis、真实dot reduction和局部certificate按实际调用联动。只读推导及counterexample位于`tmp/P1_CHART_NEXT_HANDOFF_20261011/`，实码与独立审阅位于`tmp/P1_NUMERICS/`和`tmp/P1_NUMERICS_REVIEW/`。小组/单组结论集中在[校准与选择](calibration_and_selection.zh.md#小组部分标签与采样坐标)。

## B07：固定native新家族及真实坐标关系

唯一`dictionary_artifact/response_dictionary`compiler新增`/4` regular raised cosine和固定coarse6+fine detail3 cubic chart，free/hard-value端点；Gaussian/cosine `/2`及cubic `/3`各保原函数和数值语义。raised先在Decimal96中组合`(1+cos)/2`再一次binary64舍入，与旧原型先舍入cos再组合不同；atom边界精确为0，global free crop与hard-value连续性分开解释。

multiscale九列是fine9空间的可逆chart，并非新增span或正交wavelet。若`D_new=D_fine S`，保持同物理prior要求`Lambda_new=S.T Lambda_fine S`；同坐标ridge会改变prior及有效自由度，不能宣称同容量。g-reference沿真实C设计重编译并接受独立换图预测检验；hard端点为真实ambient chart内的约束。fs、`L=round(fs*T)`、实际端点`L/fs`、最后样本和半开支持继续分别保存。

实际consumer链为spec→compiler→event design→C-reference/prior→continuous posterior/容量→原OOF→ensemble save/load/score。五家族所有现有合法空间的精确off-grid δ/w评价及完整历史预测已完成下述软件切片。专用三臂dictionary comparison panel保持原Gaussian/cosine设计；下一出口为同支持公共评价、wavelet及导数/硬斜率，W03.01保持partial。

后续off-grid/history方向已通过源码、方法原件与独立反例审阅，见[方向绑定](../assets/research_resume_20261011/b07_offgrid_history_direction_binding.json)。12个小矩阵数学反例与7个时钟/授权反例分别验证：同native数组不足以定义连续函数；半开裁剪不能用Taylor或舍入onset替代；邻居与当前必须共用一次kernel后验并保跨事件协方差；正确条件混合为full与prefix各自先积分再相减，prefix同时更新邻居状态。Unfold原全文的完整事件设计/线性叠加给出方法依据，当前模型仍按自己的Gaussian/有限状态假设推导。

五家族精确δ/w评价、原decoder持有的截止内完整历史、冻结C后验的多通道条件预测及capsule保存重放已集成。当前decision完整flash覆盖与response尾部右删失分别记录；next-selection已发生前缀保完整候选词表和零贡献列。每个state的精确半开支持参与事件可见性判断，包络用于读取边界和ledger；恒零轴按完整实际joint质量边缘化，严格正质量不按阈值裁剪。原decoder整数event sample、response latency和width分别保存；实测fractional时钟由后续独立输入authority提供。

B05数值中枢与history adapter已接通：`native_history_marginal_program`消费原`TemporalNoise`、M/R/Σ及实际full/prefix观测，复用增广QR，按通道展开共同flat nuisance，生成full/prefix absolute potential、稳定conditional companion和原source-scale/nats估计。每状态及完整mixture的数值记录随prediction/capsule持久化。mixture先由prefix更新状态权重，再积分direct conditional；误差传播为`2*max(prefix_error)+max(conditional_error)`加实际算术估计，full误差保留且不重复相加。最终candidate scores先减max再logsumexp，保存概率保持归一关系；数值失败返回完整requested词表、capability和源码身份。

冻结C后验的合法state范围按实际C标签条件化的正joint质量判断：每个C-owned selection的正质量状态全部nominal，或它们的full/prefix design均精确为0。显式C-nominal/future-variable joint law获准；更宽C随机响应模型进入新posterior producer。主树54项相关software/synthetic检查通过，最终C-law六案例和多C失败账本另在实际主树复签；mixture独审11项数学案例与32组Decimal110扰动通过。软件验证包含合成拟合/加载/预测，独审曾误调用的4-case fixture按1次合成C fit、7次顶层load、6次prediction attempt单列，public fit/score为0。实际raw源码、行尾差异、验证范围及后续公共评价见[整合绑定](../assets/research_resume_20261012/history_central_integration.json)。真实A01 drift=0与drift=1e3/1e6的原接受/拒绝证据保留其来源。

## B09：可识别效应与source群体forecast

跨reference的共同物理效应输运已接通：直接在source真实joint likelihood中测目标reference-free泛函，保存BLUE均值、完整measurement covariance和native noise归一化，再进入群体forecast与目标prior。共同电极和参考关系来自共享authority；declared CAR的结构零与生成舍入一起核对。三种已声明source路径及source→群体→target→B05有独立逻辑和实际调用证据。最终入口修复冻结tuple/list的等价序列关系后，5项delta验证通过，见[整合绑定](../assets/research_resume_20261012/research_integration_acceptance.json)。

独立审阅确认一般full-row source的下一修复：结构可识别性应由物理空间map、时间design及record-local nuisance的精确因子乘积计算。四导单行`[1,3,5,−9]`的真实beta-nullity为12，现浮点归一化再Kronecker的逐项舍入却能给出0；微小但真实未观测泛函随后可能获准。下一切片保存并重建factorized physical law，数值likelihood/BLUE继续用原程序及逐项witness核对。当前输运先闭合已有生成authority的native、实测参考和CAR三条source路径。

`ERPPhysicalEffect`绑定basis/window/grid/fs/operator/dictionary、真实reference与空间观测map。`ERPChannelEstimate`只从实际joint likelihood提取可识别BLUE，不把已有收缩posterior当独立measurement；measurement noise/dependence由其owner明确绑定。若压缩观测为`y=X beta+e`，物理泛函F必须存在witness`F=W X`，则效应mean为`W y`、covariance为`W W.T`。新关系检查逐entry使用原物理坐标，零观测列不从其它强列借容差；允许界来自浮点算法与运算尺度，不是全输入区间证明。

source group以participant为单位，对效应measurement covariance与人际variance分别记账，使用外部模板、强anchor与整程序上界。forecast covariance为`S_between+V_population_mean`，variance component为plug-in，不声称完整Bayesian积分。`mean_only`与`separable`是两种明确completion；不能把标准化d-prime向量按reference线性推送或手工放大噪声precision。

审查复现并修复tiny未观测泛函被全局sqrt(eps)吞掉、learned目标权限只在fit而可绕constructor/load、source scope/unit在load未完整核、效应coefficient-group关系、measurement schema/d-prime关系与zero-F的虚构population forecast。固定群体保全部participant；超出预声明bounds是整程序typed失败，不删异常人或截断贡献。

该source-group owner已经连接实际spatial posterior/effect/forecast及原B05 given-noise研究入口；独立d-prime学习机制仍未实现，当前continuous ensemble/公共runner/工作站尚未调用learn。真实跨source布局与reference transport、跨人稳定性、超参数选择、variance不确定性与新效能仍是后续出口。

## 后验QR与有效自由度的反例

恢复增量使stationary fit均值、posterior covariance、保存R/RHS/Q、noise residual与score消费同一个compressed QR后验，避免两次QR的舍入差把合法取消均值拒绝。legacy clipped AR1保持原估计程序，数值值不承诺逐bit等同父source；旧包继续由其冻结producer解释。

独立审阅另复现重复basis的大尺度合法反例：3×110样本、2channel、ridge1、实际只有两信号方向。`trace((D.T D)H)`在AR0给26.2414407043，独立两维模型的df为2；错误residual df使Sigma放大8.1413%。legacy AR1在该反例给6.4695454025，Sigma放大1.4076%。这是数值算术错误，不是新生理机制。

修后保持原定义，使用同posterior root和实际information root计算

`df=trace(D.T D H)=||R_D R_posterior^-1||_F^2`。

不clip df，不改LedoitWolf与残差自由度科学定义，fixed32的nTail profile分母保持。独立两noise路径×六amplitude核查中，1e8幅度的df与独立模型差小于4e-14，Sigma倍率回到1；新正式反例先RED2后GREEN。保存mean/root关系门仍拒绝真实变更，不放宽阈值。

`H`仍是保存的factor-Gram诊断；held预测直接消费保留的R，不重新从H分解。fixed32只改变同一producer后验的数值生成入口，profile Sigma仍由原joint tail目标估计。

## P1后续方向：冻结响应与噪声程序的配对评分

下列方向记录对应实施前的决策；软件入口和16格native面板随后均已完成，结果见本页开头及下节。

独立逐读B05/B06/B11后，总监接受同一完整物理response后验下的shared/private AR8、共同C residual与32点prefix，接现有CandidateEvidence、DecisionPlan与paired metric owner。实现由独立`research-mechanism-paired`工作树交付；该树现继续承担数值稳定化。

方向依据包括8个数学/实际owner反例和一次current synthetic fit/save/load上的2个关系核验，均exit0。完整可逆白化不自动损失判别信息；非可逆投影可降低ACF却删除ERP；相同秩不排除错误span；Schur covariance正确而漏候选相关条件均值仍可翻转赢家；无候选信息的prefix仍更新共享不确定response。另实测近似平局在B05原门与共同CandidateEvidence门会得到不同赢家；删困难决策使可用NLL下降却降低coverage，被共同paired gate拒绝。新panel预声明共同赢家程序，原B05 winner仅作其原程序诊断，不追签旧输出。

现代Continuous保存的上三角row precision为R，完整系数M前半为common、后半为target increment；row-major系数×channel物理factor为`kron(R^-1, chol(原Sigma_ref))`，保持同一保存M/R和原spatial covariance。真实小型scorer图像关系误差约`1.39e-17`；误用R转置的图像误差约`0.04657`。不能从派生H重造现代producer，不能随新noise换response不确定性；这是模块化plugin预测程序，不称各臂joint posterior或joint MLE。

新physical入口须消费实际PredictionDecisionView/MembershipDecision的onsets、membership、source/support、事件counts和evidence level，design按原basis核验；旧generic数组manifest不能猜counts补身份。建议后续16格C内诊断为8实际participants×2noise，各用C800授权中最早6完整组、前5组600标签拟合、第6组120标签later-held，其余标签另列。它不称C800全fit、独立公共终测或成人确认。现代转换、共同指标与该实际输入先在代码/反例/保存重放中验收，再执行新panel。原ignored B11 payload未恢复，不重跑旧panel来补签；证据指针与hash见[方向绑定](../assets/research_resume_20261011/p1_direction_binding.json)。

## P1软件入口修后验收

`FrozenERPResponse.from_continuous_model`和paired audit `/2`已集成：保存M/R/原Σ转换为完整物理factor，公共入口从原MAT owner重建view、onsets/membership/design/support及标签权限，输出共同CandidateEvidence/DecisionPlan/paired指标。C-side残差明确不扣nuisance，可含drift/ERP失配；held原flat affine nuisance和noise数值程序保持原定义。本接口是固定base-C后验后的plugin比较，不把新noise当重估joint posterior，不从噪声诊断决定产品默认。

独立审阅否证了作者65项通过的初版：保两份实际source records却同时删A02 participant与plan能缩分母；另将C labels改成另一合法candidate并真正重拟modern R模型，也能越过public admission。修后公共requested集合来自全部actual source CalibrationSelection，resolved与selection-failure由共享DecisionPlan完整分割；每个C target/label逐项核原owner。两攻击均在noise.fit之前拒绝，合法失败participant仍保宏分母。原RED、原65-body与修后身份分开，不累加成一组证据。

修后作者16项、原独立主probe6项及专门admission3项通过；原wrong-label脚本因更早builder拒绝仍exit1，保原字节，另用旧wrong-label输入直接audit确认0fit拒绝。root逐读source、C、modern factor、held、指标及失败消费者后接受有限软件切片，Ruff/diff检查通过。两人四格诚实actual-owner synthetic输入完整fit/save/load/replay通过，不代表新native成绩。八原MAT的16格面板随后完成，但16个held cell均因完整proper-prior/noise chart的数值可分辨性门记为typed `not_available`，概率覆盖0、paired NLL/Brier为空；这不是分类错误率读数。执行身份、8 anchor/16 noise fit、保存态零拟合失败重放和资源边界见[native面板绑定](../assets/research_resume_20261011/p1_native_panel_binding.json)。原面板完整执行说明保留在本地 `tmp/P1_NATIVE_20261011/attempt02_absolute_paths/`。
