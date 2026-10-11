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

这批 P0 代码已提交到隔离科研分支 `c3278b7`；B12 只有 software/synthetic contract evidence，不能宣称完整单seed执行、公共效能或成人确认。

云端后续交接：数据盘清理从约0.949GB free恢复到25.176GB，原科研文件无计划外变化；B08 tuple helper 修正以独立新 producer完成 exit0，1344包/576 energy/48 C reference/28 native/20000 bootstrap等按新 acceptance 身份解释，旧失败不改签。该结果不替代B12 source→target execution，科研侧继续保留 candidate input-contract、资源 admission和四库终态边界。
