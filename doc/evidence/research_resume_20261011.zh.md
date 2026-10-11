# 科研恢复：nested选择、物理字典、source空间先验与QR数值

2026-10-11。本页承接[123项科研进展](wishlist_progress_20261010.zh.md)，记录代码恢复、实际审阅、数学反例与有限验收；整体B00–B14继续活动。完整公共source→target程序和新成人确认仍分别由其实际数据、执行程序与独立协议验收。

## 恢复身份与文件所有权

原`wishlist-integration`工作树已被应用归档。最后已接受科研提交为`1ca4619a79d5a8e4b536ae1960ea7029604801ad`；Git恢复快照`99352070d267141db8651c26c5749a564f6762c6`以它为直接父提交，保存12文件的待审增量。恢复快照不是科研验收。本次从该快照建立`codex/research-integration-20261011`，实际路径为`C:/Users/chent/.codex/worktrees/research-integration/n2p3-net`。原ignored暂停检查点、旧独立review与诊断目录在原路径未找到，不将其称作已恢复产物。

用户原`D:/BCI/n2p3-net`并发dirty保持；云端`deep-research-plan`与accepted496归另一窗口独占。本科研源码没有替换其producer。云accepted496 archive为`4b075a2df69b781f03844ad7efe2e3d4a2b64342118ea291bb85e1aa99dc2127`、manifest为`04ff8a07a7a801fca4d46904572ea9a2acd6696570c72454de280d773900f12e`，其原四source模型及结果由云窗口按原身份解释。

三路作者分别拥有nested/ensemble、dictionary和spatial文件；第四路独立审阅QR与全局依赖。作者结束后交换复审，科研总监逐读差异、完整调用链和反例，再集成。本轮不跑仓库全集。统一使用项目venv并显式`PYTHONPATH`指恢复树；venv的editable安装本身指向用户原树，单凭workdir不能证明实际执行来源。

## B03：整个程序的nested边界

外层held组记为v，选择层held组为h，温度训练held组为t。评价v之前，候选选择只读`C\\v`；评价候选在h上的输出前，其模型与温度只读`C\\{v,h}`；温度的OOF基础模型进一步排除t。固定过滤在各组的合法物理scope内执行，统计、prior和真实expert fit从各自subset产生state；完全相同subset才可复用实际缓存。最后全C选择与refit不读取外层评价结果。

`research.calibration_selection`定义预声明candidate roster，固定或分组温度程序；完整概率支持、hit→NLL→Brier顺序、baseline输出/失败保护与固定tie顺序共同决定选择。失败、不可用和tie保留requested owner，不能缩小expert roster或重归一化绕过失败。run-local缓存按实际group记录、labels/sample digest、program与source bytes绑定，只复用完全相同subset；原expert wrapper冻结，温度派生新对象。

审查复现：重复/重叠raw或record未在split前拒绝、decision身份延迟拒绝、event身份重复被接受、同final数值包可重写label authority、选定温度子集与actual final不一致，以及可变expert wrapper漂移。修复以原ensemble owner的`validate_calibration_ensemble_groups`统一完整roster准入；`_calibration_expert_fit_digest`复用原fit身份公式，constructor/load将最终scope、labels、checkpoint角色和温度子集绑定回原FittedEnsemble。

本切片为完整组科研API及final包；后续 P0 续接已增加 opt-in inner producer capture/replay，但公共B12 prefix runner与BrainSync尚未接nested程序；外层C内相关组读数不等于新participant效能或成人确认。W06.11/W09.06的程序隔离推进，不把W07.13 gating或全部B03算完成。

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

## 验收状态与下一依赖

本页的作者和独立审阅产物在`tmp/RESEARCH_DIRECTOR_20261011/`；源身份、必要验证、原MAT重放与最终接受状态由同目录和本页后续结构化binding保存。目前正在完成交换复审与真实native组合验收，未把本页当最终签收。

全局优先序：保B12基础公共执行独立前进；补nested inner持久重放及prefix/runner consumer；真实source效应权限与跨布局transport在接learn前明确；B07后续wavelet/off-grid/history按实际数学依赖进入。TemporalNoise当前`/5`从保存编译态安装，旧startup再生成RED不能当当前未修事实；完整同版本跨host模型重放仍待实测。B05/B11时间相关失配与ERP/span是竞争解释，不让一个数值门阻塞其它愿望，也不从拟合更好选择产品默认。

## 云端交接边界

2026-10-11 云窗口报告新纯性能 consumer 已实际启动：physical freeze archive `f655dea8b19d1d1a9b2664cf3a175cfed9186349d27cb26940f55acedd437794`、manifest `a8827a42e8a3fe6ea25f8f1fb0871d308dee89350864075c66cf9ac49aaeeb74`、隔离候选 commit `3747bf1bc1a9138f9c771aca7461403db8079222`。该 execution 使用新 owner/API，与旧 accepted496/source producer 分离；科研树不修改其运行文件，也不从未终态云结果选择科研 route。云端实际吞吐、完成矩阵和效能由云窗口继续验收。

## P0续接：inner replay 与 B12 runner

全局审阅把下一顺序定为 B12+B14 单 seed 公共 execution 最高优先，B03 inner producer replay 并行；B05/B06/B11 固定输入机制面板为诊断支线，B09 standalone learner 与 B13 成人确认不阻塞 P0。依据见 `tmp/RESEARCH_DIRECTOR_20261011/GLOBAL_ROUTE/REPORT.zh.md`。

B03 新增 opt-in InnerReplayRecorder，在同一次 `fit_nested_calibration_selection` run 捕获实际 cache producer、grouped-temperature OOF fixed producer/held score、v/h/t context、失败 receipt与 source role。persist_inner_replay 没有原 recorder 会 fail-closed；有 recorder 才保存实际 ensemble package 和完整 manifest，replay_inner_replay 只加载、审计 forbidden scope并重放 score，不二次拟合。8 项 inner 测试、原 selection/ensemble 41 项通过；B12 CalibrationPrefix/public caller仍未接。

B12 新增 `validate_b12_execution_manifest`，由 paired `run_erp_comparator._finalize_comparison` 写入 `b12_execution_binding`，校验 requested K×arm×seed roster、实际K alias、失败/不可用分母、target authority/input manifest和 public source producer catalog。软件 scope不虚构producer。另修复 FrozenTrial list/tuple JSON metadata 比较，保留真实 preprocessing 差异拒绝，相关校验5项通过。云端 candidate8 的6个 input-contract RED已缩小到该类型问题并修复科研树版本；2个FS/RAM失败、四库 Errno28 和完整source→target终态仍保留，未放宽 gate或改写旧producer。

这批 P0 代码已提交到隔离科研分支，之后独立复审继续核对其实际行为；B12 只有 software/synthetic contract evidence，不能宣称完整单seed执行、公共效能或成人确认。

云端后续交接：数据盘清理从约0.949GB free恢复到25.176GB，原科研文件无计划外变化；B08 tuple helper 修正以独立新 producer完成 exit0，1344包/576 energy/48 C reference/28 native/20000 bootstrap等按新 acceptance 身份解释，旧失败不改签。该结果不替代B12 source→target execution，科研侧继续保留 candidate input-contract、资源 admission和四库终态边界。

## 原 MAT 保存重放验收

从已接受 `cf48f78` 物理导出源码，在原 A01 MAT（SHA `cdf9bca4…`）的授权 C800 中拟合六个完整组，实际使用720 group labels。原 harness 在有效模型已保存后错误地重评 calibration views，被原 `ensemble prediction repeats a calibration observation` 门正确拒绝。模型状态没有因此失效；没有放宽该门。

修后使用原 independent-prefix owner 的 C800 之后 future queue，只消费 label-free view，不读取 queue truth 选择程序。重放保留全部29 requested decisions，其中28有合法概率，1项不可用仍在分母；原模型→重新保存→加载的证据29/29一致，三份package文件逐字节相同，纠正过程中新增拟合0。这个结果是公开数据上的软件保存重放证据，不报告hit/NLL或新效能。原失败日志、原包和修后消费者身份各自保留，详见[紧凑绑定](../assets/research_resume_20261011/native_nested_replay_binding.json)。

## B12最终账本修后复签

独立审阅否证了此前manifest初版：coverage只核总数、重复source键消失、可重签的producer声明与plan不一致、unresolved peer掩盖completed peer缺源，以及statistics之后的report不在摘要内、重复finalize滚动绑定旧摘要。原RED、原probe和历史producer保留；本轮修复后才重新接受对应关系，不能沿用初版的通过数量作修后证据。

修后coverage逐bucket核实际referenced cell终态；source catalog拒绝重复和foreign study/seed，checkpoint/procedure与plan一致，public producer的seed/kind/selection/source-study mirror及target exclusion互核。failed actor的缺失source保持显式`unresolved_source_producers`，completed NN/scratch cell缺源直接拒绝。participant聚合遇到同键不同完整凭证拒绝，相同凭证可合并。最终statistics写完后绑定report，摘要排除顶层`b12_execution_binding`，重复finalize不消费旧binding。

原独立15项探针字节未改，修后15 PASS、exit0；真实source7 selection/refit（inner3/refit2）及unresolved-peer正控仍通过。实际paired waveform serial finalizer保留8 requested aliases、6 resolved cells；首次最终report、去binding独立canonical和repeat-finalize摘要均为`13d3bb70…`。正式B12合同与actual caller22项通过。另从`e9ef642`导出不可变副本，仅覆盖本轮B12三文件，对实际spawned participant部分失败、全失败及source dependency unreachable三个聚合路径验证通过；未混入并发prefix WIP。各命令退出码0，相关Ruff与diff检查通过。source/file/日志摘要和实际coverage见[紧凑绑定](../assets/research_resume_20261011/b12_postfix_binding.json)。

总监逐读最终manifest、serial finalizer与participant聚合调用链后接受这组软件关系。聚合冲突分支由源码审阅确认，不冒称独立15案例实测；本轮未重新认证原始source fit/cache或完整checkpoint，跨阶段门继续消费原物理入口已核的凭证。这些证据不代表完整公共矩阵、公共效能或成人确认。CalibrationPrefix/public caller接线仍待另外的输入、OOF、common-test及保存重放反例闭合。

## CalibrationPrefix保存重放与调用端复签

后续prefix修稿已经独立复签、经总监逐读实际fit/OOF/save/load/future/parent admission后接受。`fit_calibration_prefix_programs`捕获同一次实际final专家和inner状态；`persist_prefix_inner_replay`绑定raw/trial样本及标签、requested aliases、实际recipe/checkpoint seed与source role。load从原数值模型的fit binding和final OOF owner核关系，不能仅重签manifest副本。成功OOF必需的inner roster/held score完整；typed inner fit失败、返回不可用evidence及直接raised score异常各保真实凭证，不虚构成功重放、不删除合法fixed程序。

实际independent caller捕获完整truth-free future expert outputs，核共同Kmax test scope，保存/load/replay后才封prediction。父验收从已捕获专家用原EnsembleScorer重新计算plan mixture并核完整概率/失败记录；同一model digest不能授权改概率。iterable只消费一次，不可用温度strategy保全requested未来行。外部JSON容器与内部冻结tuple采用既有semantic owner比较，矩阵、导序、reference、fs、SHA和值变化仍拒绝。fit/cache miss的真实状态也已修复，共用造数归tests/support。

最终独立原六类违规全部拒绝、两个诚实fixed/grouped对照通过；新增consumer7项及真实Cz6 caller/raised-score两项通过，源码七文件前后稳定。原caller probe因缺新必需test_scope仍exit1并保留原字节，合法scope/queue由新增独立harness验收，不把原exit冒记为绿色。作者34项compact/cache/三对caller属于spatial同义修前body；30项prefix/actual spatial属于最终body，native interruption1项另绑定最后测试字节。旧74项、source_changed失败和旧RED保持自己的source身份；详见[紧凑绑定](../assets/research_resume_20261011/prefix_replay_binding.json)。

实际6包receipt从旧10.22–10.95MB缩为新1.48–2.21MB，父验收对每包只解析/核验一次并消费冻结索引，消除原1224次预测重复解析同一大凭证的路径；这里只陈述实测尺寸与调用次数，未声称训练端到端提速。当前事务粒度为整个prefix future capture/replay后逐行封预测，已计算但未封行中断时仍为partial/unknown；native退出验证只证明已封行、已处理strategy和完整requested账本保留，不证明任意中断可零fit恢复。

这是实际prefix/public候选runner的软件接线；它仍没有候选程序的strict nested selection或learned gating，不完成公共效能、跨host整包或成人确认。原MAT窄输入验证另用冻结源码执行，不把上述synthetic通过重签成公共结果。

原A01 MAT后续窄验证已exit0：从clean已接受源码按实际工作字节冻结971成员（保CRLF/LF字节），只跑预声明raw+Toeplitz、K200/K800、fixed/grouped两个程序与共同独立未来队列。29 requested decisions产生232 prediction行；每K捕获58个future strategy行。raw使用200/800标签，trial实际使用196/796标签；K200 grouped不足组具名不可用，K800保6个真实inner producer并成功重放。所有模型load/future replay及实际parent mixture准入通过，原MAT前后SHA一致；源checkpoint拟合0，没有用未来truth选方。它只证明原公共输入的软件链，未签收分类效能。源码/原MAT/脚本/退出码/包摘要见[原MAT前缀绑定](../assets/research_resume_20261011/native_prefix_replay_binding.json)。

## 原云候选面板：完整分母与配对读数

云窗口独立零拟合恢复已完成8人、152共同requested decisions、13376预测、448 fit records、32物理包、88指标与原80配对对照。科研总监实际读取本地terminal/actual exit/delivery/independent statistics，逐字核其SHA与delivery吻合，再读完整80原对照及20指定导出；其中8行已有原对照、12行为原保存预测的新只读派生。原executor exit1、scientific source `75a61532…`及其499冻结manifest均保留，恢复exit0、新source/target fit均0。完整失败账本留本地，不以科研新prefix协议改签。

主迁移pool在正预算fixed/grouped的概率coverage均144/152，8个不可用决策仍在操作分母中。paired NLL仅在144共同概率决策上按原participant/seed层级归约，不把条件概率分母改成152。

| K | fixed hit@all | grouped hit@all | fixed NLL | grouped NLL | grouped−fixed配对门 |
|---|---:|---:|---:|---:|---|
| 200 | 83/152 | 0/152（全部不可用） | 1.744632 | 不可用 | false：输出丢失、无paired NLL |
| 800 | 113/152 | 114/152 | 0.851218 | 0.999665 | false：NLL增加0.148447 |
| 2000 | 132/152 | 136/152 | 0.367592 | 0.485101 | false：NLL增加0.117508 |

因此grouped的少量hit提升没有通过原开发风险门，不能据它晋升默认。完整source→matched scratch方向为candidate−baseline：K2000 fixed单NN的scratch比source多7/152 hit、paired NLL低0.036876；matched scratch pool多1/152、NLL低0.034934，两者原点估计门true。这是scratch在该条件下改善source，不是迁移收益。低预算fixed pool的source读数较scratch高1/152、NLL较低；K800 fixed单NN在hit/NLL上存在取舍，不能合并成统一“source更好”或“source无用”的结论。逐人也有异质性，K2000 fixed单NN的A03/A05在scratch方向hit与NLL均变差。

这些出口只属于BNCI八人、一个实际`20260918` seed的公共校准开发程序；不覆盖训练随机性、独立确认、健康成人九选一或产品保证。当前研究依赖转向合法C内整程序选择、温度风险解释及独立P1机制，不用这份未来读数直接指定participant/budget默认策略。原始result约346MB、完整80/20失败账本、producer及具体读数指针见[紧凑来源与读数](../assets/research_resume_20261011/cloud_candidate_readout_binding.json)；科研根没有再次重放全部大模型或重新计算原346MB预测。

## P1后续方向：冻结响应与噪声程序的配对评分

独立逐读B05/B06/B11后，总监接受下一接口切片为同一完整物理response后验下的shared/private AR8、共同C residual与32点prefix，接现有CandidateEvidence、DecisionPlan与paired metric owner。实现已分配到独立`research-mechanism-paired`工作树，不影响在途B12或当前prefix源码验证；软件接口和实际输入尚待签收，新native panel未执行。

方向依据包括8个数学/实际owner反例和一次current synthetic fit/save/load上的2个关系核验，均exit0。完整可逆白化不自动损失判别信息；非可逆投影可降低ACF却删除ERP；相同秩不排除错误span；Schur covariance正确而漏候选相关条件均值仍可翻转赢家；无候选信息的prefix仍更新共享不确定response。另实测近似平局在B05原门与共同CandidateEvidence门会得到不同赢家；删困难决策使可用NLL下降却降低coverage，被共同paired gate拒绝。新panel预声明共同赢家程序，原B05 winner仅作其原程序诊断，不追签旧输出。

现代Continuous保存的上三角row precision为R，完整系数M前半为common、后半为target increment；row-major系数×channel物理factor为`kron(R^-1, chol(原Sigma_ref))`，保持同一保存M/R和原spatial covariance。真实小型scorer图像关系误差约`1.39e-17`；误用R转置的图像误差约`0.04657`。不能从派生H重造现代producer，不能随新noise换response不确定性；这是模块化plugin预测程序，不称各臂joint posterior或joint MLE。

新physical入口须消费实际PredictionDecisionView/MembershipDecision的onsets、membership、source/support、事件counts和evidence level，design按原basis核验；旧generic数组manifest不能猜counts补身份。建议后续16格C内诊断为8实际participants×2noise，各用C800授权中最早6完整组、前5组600标签拟合、第6组120标签later-held，其余标签另列。它不称C800全fit、独立公共终测或成人确认。现代转换、共同指标与该实际输入先在代码/反例/保存重放中验收，再执行新panel。原ignored B11 payload未恢复，不重跑旧panel来补签；证据指针与hash见[方向绑定](../assets/research_resume_20261011/p1_direction_binding.json)。

## P1软件入口修后验收

`FrozenERPResponse.from_continuous_model`和paired audit `/2`已集成：保存M/R/原Σ转换为完整物理factor，公共入口从原MAT owner重建view、onsets/membership/design/support及标签权限，输出共同CandidateEvidence/DecisionPlan/paired指标。C-side残差明确不扣nuisance，可含drift/ERP失配；held原flat affine nuisance和noise数值程序保持原定义。本接口是固定base-C后验后的plugin比较，不把新noise当重估joint posterior，不从噪声诊断决定产品默认。

独立审阅否证了作者65项通过的初版：保两份实际source records却同时删A02 participant与plan能缩分母；另将C labels改成另一合法candidate并真正重拟modern R模型，也能越过public admission。修后公共requested集合来自全部actual source CalibrationSelection，resolved与selection-failure由共享DecisionPlan完整分割；每个C target/label逐项核原owner。两攻击均在noise.fit之前拒绝，合法失败participant仍保宏分母。原RED、原65-body与修后身份分开，不累加成一组证据。

修后作者16项、原独立主probe6项及专门admission3项通过；原wrong-label脚本因更早builder拒绝仍exit1，保原字节，另用旧wrong-label输入直接audit确认0fit拒绝。root逐读source、C、modern factor、held、指标及失败消费者后接受有限软件切片，Ruff/diff检查通过。两人四格诚实actual-owner synthetic输入完整fit/save/load/replay通过，不代表新native成绩。八原MAT的16格面板随后完成，但16个held cell均因完整proper-prior/noise chart的数值可分辨性门记为typed `not_available`，概率覆盖0、paired NLL/Brier为空；这不是分类错误率读数。执行身份、8 anchor/16 noise fit、保存态零拟合失败重放和资源边界见[native面板绑定](../assets/research_resume_20261011/p1_native_panel_binding.json)。原面板完整执行说明保留在本地 `tmp/P1_NATIVE_20261011/attempt02_absolute_paths/`。

## 缓存与共享评分独立复签

source cache 与 shared programme score 已在科研主树 `4830ef9` 集成。修后 checkpoint 的 whole-study literal 权限只读私有 admitted source ledger；当前 target 的 physical、identity、execution 与 study proof 每次重核，source prototype 不缓存 target fit、BN、OOF 或 temperature。`FittedEnsemble.score_programs` 只在同一 current view、同一实际 experts tuple、fit receipts 与 target contract 下共享 raw score，fixed/grouped 仍分别应用自己的概率程序；capture 与 loaded future replay 各自真实计算，parent 保原混合准入。独立 authority/fresh-state/lifecycle probes 和两条真实 K0/full-C/OOF caller 路径通过，原权限绕过和失败日志保留。固定源码及证据边界见[独立复签 binding](../assets/research_resume_20261011/cache_shared_acceptance_binding.json)；这仍是软件/有限调用链验收，没有完整矩阵吞吐、跨host整包或公共效能声明。

## P1 数值门诊断与小组协议评估

16格 native panel 的失败主要来自全记录 `gamma_N`/nuisance/proper chart 的保守误差上界；实际 Q 正交误差约 `2e-13`，A01 dense 高精度密度差约 `1e-10`，对 nuisance 做可逆坐标变换后 density 逐bit不变而误差估计显著下降。当前建议是稳定、预声明的 nuisance basis 与 prefix/tail 局部误差界，不能直接放宽阈值或 fallback；详细只读诊断留在 `tmp/P1_CHART_DIAGNOSIS_20261011/HANDOFF.zh.md`。

A01 原始 owner 的每个 selection 是120 flash、10 repetition；原始MAT为256 Hz，common profile从原始事件支持取308 samples和51 samples pre-roll，再输出128 Hz/179点trial；P1 raw continuous owner的1.2秒支持为307 samples。R=1/2/5 repetition时响应尾会覆盖未纳入flash，且同一selection的target pair不变。K200的后80 labels可成为另行声明的right-censored C内训练记录，但不是第二个完整selection；用它训练再评价更早完整selection只能叫offline backcast。既有accepted K200 final fit仍是raw=200、trial=196。单完整组可支持fixed baseline或预冻结source-global温度，不能把target grouped OOF下限直接降到1。固定256 Hz窗口与单组温度解析结果见[fixed small-group binding](../assets/research_resume_20261011/prefix_small_group_fixed_binding.json)，原128换算RED保留在父目录；partial scope与当前caller边界见[评审binding](../assets/research_resume_20261011/prefix_program_review_binding.json)。

下一跨数据集队列先做 **BI2014a**，不一次展开全部数据集。目标 owner 是现有 `data.bi2014a_candidate`/`prepare_bi2014a_candidate`，target preflight 已确认61条记录、raw512Hz→model128Hz/179点、976个文件。source07 catalog随后解析出既有 leave-BI2014a source producer（seed20260918，checkpoint SHA `fc8d8cbdef405263ff9c233e592755ceeeb2264bc8287800b8b2ffacfb199c02`）；云端正在核对其 source snapshot、roster、task contract 与 target exclusion，核验完成后再解锁source→target固定配方与matched scratch三臂。target grouped temperature不在该队列学习；每库独立持久保存plan、输入/source binding、participant receipts、模型重放、失败分母、指标和资源终态。该状态与下一动作记录在[BI2014a队列计划](../assets/research_resume_20261011/bi2014a_queue_plan.json)，当前仍没有target fit。
