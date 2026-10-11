# 响应模型、空间先验与噪声数值

本文按主题维护已复核的结论、代码接点和下一步。运行日期与源码身份留在各证据及绑定中；使用方式和工程规则由原六指南维护。

## 当前模型与数值任务

固定响应的完整物理factor由保存的M、上三角precision R和原Sigma导出：`F=kron(R^-1,chol(Sigma_ref))`。它保持common/increment系数及channel坐标，噪声变化时复用同一物理响应分布。

八人shared/private AR8、共同prefix32的16格native诊断已完成：每人600标签fit、120标签held评价、80授权标签未使用，8 anchor/16 noise fits均保存加载。16个held评分均触发数值可分辨性门，概率覆盖0；完整保存态已零拟合重现原失败。[执行绑定](../assets/research_resume_20261011/p1_native_panel_binding.json)保存原来源和完整分母。

数值稳定化已接受并接入科研树：full diffuse quotient保无量纲source-scale界，conditional score保nats界，仍用原两道阈值。实现包含prefix nuisance可逆换图、实际256-row短叶/平衡reduction、真实prefix/tail与列误差传播，以及完整posterior metric中的局部位移界。`Q.T b`显式按`abs(Q).T abs(b)`计算dot误差，logdet的factor2与raw normalizer/输出舍入都入账。

独立审阅核完整AR历史、flat测度/Jacobian、相关proper factor、inverse solve defect、prefix压缩和后验位移。109个独立案例通过；54个conditional案例中32获准、22具名拒绝，获准案例的110位oracle最大误差`1.395e-9 nats`且在各自估计界内。root逐读全部差异及最后finite保护，按原字节接入`5b827aa4…`，核依赖/证据SHA并通过六个新增反例、Ruff与diff检查，见[数值接受绑定](../assets/research_resume_20261011/p1_numerical_acceptance_binding.json)。

真实A01 shared/private各36候选已经恢复完整概率输出：原保存态由原loader准入，新数值consumer为`22bf6782…`，新增fit为0。source-scale界最大分别为`5.380e-9`/`8.263e-9`，conditional nats界最大`0.000295162`/`0.000337333`，同actual tail预算`0.000946760`；概率mass均为1。随后5b只补无FMA极端中间乘积的typed有限值保护，保72候选原source身份。其余14个保存态cell继续独立评价，原16格失败面板保持原记录。

这次A01消费的一次本机CPU采样为80.9%，按实际记录为超限；后续native1且检查总机余量，最多两个score workers。完整数值程序和原producer分开绑定；tree dot也改变新C-fit的舍入路径，原prior和统计模型保持。此处交付数值实现与真实保存态可用性，后续同物理支持的概率分析继续用原requested owner。

## 平稳前缀的真实模型关系

续评发现private旧入口的线程依赖：A01原保存root在native2重编逐bit相同并通过原关系门，native1重编却拒绝。更深入的两个独立100位oracle对全部36个cross-channel pair求真实平稳covariance，确认问题在旧companion Lyapunov/lag-power编译；全256维root的真实covariance pullback谱差为`0.0772181`，Frobenius差`0.167282`，而raw covariance相对差仅约`1.7583e-9`。原`1.6e-7`门核的是旧数值重编关系，弱方向需要真实law oracle。

尾部实际消费stored PARCOR lattice，新模型以每个binary64 PARCOR作为精确实数，经实数step-up确定AR；跨channel创新Sigma只入一次。完整pairwise mixed Yule–Walker与独立p² Sylvester在36个pair上一致；unequal AR1 closed form和diagonal scalar Yule–Walker另作参照。同真实law的高精度joint Cholesky仅舍入为binary64后，谱差`3.409e-11`、Frobenius差`1.109e-10`，支持保持原门实现稳定编译。

新compiler及constructor/save/load关系在独立`research-stationary-prefix`树推进。现P1续评明确使用各原冻结compiled-W Gaussian law，保持原model与consumer身份；其结果提供该实际数值程序的概率读数。真实stationary VAR的新compiled state将以新来源交付，原文件不修签。推导、两套oracle、实际旧state与下一代码依赖见[诊断绑定](../assets/research_resume_20261011/stationary_prefix_diagnosis_binding.json)。

## P1 数值门诊断与小组协议评估

数值证据的第一、第二道门分别记录。前一道需要full-record quotient的正确算术估计，后一道需要prefix/tail各自的input与压缩误差；prefix basis、真实dot reduction和局部certificate按实际调用联动。只读推导及counterexample位于`tmp/P1_CHART_NEXT_HANDOFF_20261011/`，实码与独立审阅位于`tmp/P1_NUMERICS/`和`tmp/P1_NUMERICS_REVIEW/`。小组/单组结论集中在[校准与选择](calibration_and_selection.zh.md#小组部分标签与采样坐标)。

## B07：固定native新家族及真实坐标关系

唯一`dictionary_artifact/response_dictionary`compiler新增`/4` regular raised cosine和固定coarse6+fine detail3 cubic chart，free/hard-value端点；Gaussian/cosine `/2`及cubic `/3`各保原函数和数值语义。raised先在Decimal96中组合`(1+cos)/2`再一次binary64舍入，与旧原型先舍入cos再组合不同；atom边界精确为0，global free crop与hard-value连续性分开解释。

multiscale九列是fine9空间的可逆chart，并非新增span或正交wavelet。若`D_new=D_fine S`，保持同物理prior要求`Lambda_new=S.T Lambda_fine S`；同坐标ridge会改变prior及有效自由度，不能宣称同容量。g-reference沿真实C设计重编译并接受独立换图预测检验；hard端点为真实ambient chart内的约束。fs、`L=round(fs*T)`、实际端点`L/fs`、最后样本和半开支持继续分别保存。

实际consumer链为spec→compiler→event design→C-reference/prior→continuous posterior/容量→原OOF→ensemble save/load/score。当前专用三臂dictionary comparison panel仍是原Gaussian/cosine设计，不静默扩大；新家族off-grid响应状态、导数/硬斜率、wavelet、完整历史与公共收益仍未完成。W03.01保持partial。

## B09：可识别效应与source群体forecast

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
