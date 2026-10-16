# 当前技术结构与完整链条

更新：2026-09-06。本文件依据实际源码与入口核对，描述现状与尚缺接口；科研目标由
[活动规范](research_program.zh.md)定义，发现和验证见[本次审计](evidence/technical_audit_20260906.zh.md)。
源码目录不因行数大或模块多就拆分；优先保留已经集中实现的合同和算法。

## 主链与接口

2026-09-18云端资源预算由`research.source_execution`统一解析并传递至source、源内λ和outer target worker。
`cloud_resource_floor`约束CPU/GPU容量，`HostMemoryAdmission`同时限制嵌套worker占用。
缓存输入的显式只读通道视图、统计分块与`MatrixBatchSource`成功预加载后的CPU释放保持在原owner内。

2026-09-18 `SourceCatalog.validate_fold`集中区分evaluation和deployment源组，部署组必须显式完整成员且不得生成外层目标。
`source_prior_development_fold`接受显式development_domains，未选域全量留在源拟合；source链检查开发域与逐域排除原因完整分割实际source。
源内选方记录包含声明及所有prefix摘要，改变开发支持不能沿用原选择身份。`unified_target_recipes`集中产生按监督匹配的scratch，
有真实membership时增加CE-only对照，无membership时保留单个等价scratch。调用者统一从recipe派生处理名和decision因子。

2026-09-17采集端`transport.py`集中串口/BLE连接入口、地址及固件能力差异；
`DeviceAdapter`共用连接所有权、预览、阻抗、回调录制和清理，删除串口SDK原生EDF分支。
采集端数据合同的250 Hz为两传输及模拟模式唯一采样率，SDK回调batch独立为250包的正整数倍。
`LocalEdfRecorder`统一电压/样本时钟检查点，`RecordingSession`继续持有暂停和EDF/BIDS收尾。
BLE回调按连接generation隔离，固件应答后再次核验句柄；关闭等待原生连接请求与迟到句柄清理。
调用链、反例及验证见[连接统一记录](evidence/brainsync_transport_unification_20260917.zh.md)。

2026-09-16统一程序在既有owner内实现：`N2P3Net`先pack实测通道经原temporal Conv/BN，scatter后继续空间/MST，
post-MST直接复用`FeatureResidualAdapter`，六导含残差共1279参数。`models.parameter_groups.SourceParameterPrior`
保存固定源中心/尺度；`SourceObjective`组合CE外的SupCon和条件MMD，`SubjectAdapter`在一次step内联合trial、decision和prior。
`MembershipDecision`保存物理归属、完整词表和缺失证据，NumPy/Torch共享评分核；不另建候选神经网络。
`data.source_records`准备逐K独立raw校准、inner T/V及固定test，`prepare_source_records.py`输出受核验的prefix manifest。
`unified_source`与目标`unified`复用源调度和目标worker；λ由`source_prior_selection`的源内开发manifest选择。
个人`selector_mode=unified`消费单份显式recipe和固定目标预算，保存完整参数、统计、质量状态及membership评分声明。
调用和反例见[实施记录](evidence/unified_model_implementation_20260916.zh.md)。

2026-09-16统一预算由`train.early_stopping.UnifiedTrainingBudget`集中解释，`PatienceStopper`继续负责独立早停计数。
`source_selection.fit_source_trajectory`执行参考源的参与者选轮/refit，source chain每折只调用一次，
正式源训练消费带reference摘要的`source_budget`并按e*换算固定更新。λ开发和目标worker直接全C拟合。
`transfer.epoch_selection`保留其它显式早停研究；新unified默认不消费个人inner视图。
`source_records`可按是否需要inner准备/读取实际视图；新CLI默认只准备C_K/test。
模块、复用和性能记录见[预算执行器](evidence/unified_budget_execution_20260916.zh.md)。

2026-09-16 `run_domain_transfer_matrix.py`组合既有source chain及共享目标worker，统一执行P1/P2
并复用相同输入/配方/权重的结果；不复制训练循环。`SourceObjective`集中source-only配对、
去重与批量条件MMD；`analyze_transfer_contrasts(seed_pairs=...)`显式区分完整流程配对和交叉seed设计。

2026-09-16前端运行时增加单录制启动owner及通用SDK清理worker终态门禁，禁止超时后的迟到操作
影响新会话。任务完成通知在coroutine清理后发送；`RecordingSession`持有暂停幂等和失败收尾，
`SessionStore`用OS租约排除外部修改活动会话，并在成功发布manifest后更新内存。
预览滤波使用SOS实现并消费实际设备采样率；仅作用于显示。
解码结果部分解析失败会撤销可见数字/模型就绪；反馈按请求身份及已保存目标记录，无判定为`correct=null`。
质量报告重试保留原冻结请求，失败显示与进程回收独立。见[逻辑修复](evidence/brainsync_frontend_logic_20260916.zh.md)。

2026-09-15工作站架构修复将采集上下文、样本支持、暂停账本及EDF/BIDS收尾集中到采集端
`recording.RecordingSession`；主窗口处理显示与操作者动作。真实/模拟device adapter共用该interface，
并发finish等待同一次提交。`backend.ModelTask`持有结构化命令和结果准入：schema、状态、请求编号、
会话路径/身份、受试者、模拟声明及模型摘要共同核对；后端从同一次读取的模型字节加载并生成摘要。
`SessionStore`以待提交记录协调确认标签和事件日志，前后端拒绝消费未完成确认；恢复不重复追加事件，
历史前缀冲突则保留现场并拒绝自动恢复。`ModelJobRunner.shutdown()`统一关闭入口和进程树回收，
`MainWindow.shutdown()`及应用退出finally等待它完成。详见[架构实施](evidence/brainsync_architecture_20260915.zh.md)。

device adapter在采集启动取消时保持原生申请Future，由同一清理owner等待其终态后释放资源；
超时保持recovery并阻止与迟到申请并行退订或断开。已有样本仍先通过EDF持久化路径保存。

`data.contract.BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT`与`configs/brainsync_bidirectional_preprocessing.json`
声明BrainSync新默认；`zero_phase_edge_policy`经PreprocessingSpec、raw builder及模型签名传递。
`data.preprocess.filter_bidirectional_segment`统一实际SOS双遍和两端有效状态，
`data.bidirectional.BidirectionalStateRecovery`负责校准包络、参考投影及逐epoch条件界。
固定epoch线性映射和有限段响应采用有界缓存；zero可用时间保守取完整raw结束。
`continuous_qc`保留原始检测及自然/伪迹端点区分；`transfer.quality`检查缓存中的滤波/QC匹配。
个人入口支持新默认、旧模型按冻结合同重放；MOABB高层无法执行新声明时仍拒绝。
比较入口为`run_bidirectional_qc_study.py`，见[联合优化证据](evidence/bidirectional_qc_20260909.zh.md)。

| 阶段 | 实际模块与入口 | 交付与边界 |
|---|---|---|
| 原始数据 | `data.raw_artifacts`、`data.moabb`、`data.bids_eeg`、`data.brainsync` | 字节来源、真实participant、时间轴、电极角色；校准要求confirmed，推理只要求completed且raw就绪 |
| 输入张量 | `data.contract`、`data.preprocess`、`data.epochs` | 连续滤波、epoch、基线、观测mask、timeline、attestation；当前是完整记录处理 |
| 源视图 | `data.domain`，`adapt_eeg_domain.py`、`prepare_multidomain_source.py` | 实测通道选择/缺导mask/参考、binary投影；保留父cache和身份 |
| 模型构造 | `models.n2p3net`、`train.factory`、`baselines.deep` | 架构和训练配置；`DeepBaseline`承担训练流程，`train.runtime`承担设备/内存/调度 |
| 源预训练 | `run_pretrain_supervised.py`、`transfer.checkpoint` | 权重、源分类头、训练行统计、架构、输入合同和训练身份账本 |
| 本人校准 | `transfer.subject_adapter.SubjectAdapter.fit`、`transfer.calibrated` | source-head微调、完整session内部验证、显式范围/refit、持久化个体模型 |
| 公共验证 | `run_binary_subject_transfer.py`、`run_candidate_cross_decision.py` | binary固定后缀或合法candidate后续decision；各自声明估计量 |
| BrainSync离线验证 | `run_brainsync_cross_decision.py`、`transfer.within_subject` | 同人早期校准与后续独立session，当前只提供固定训练预算 |
| BrainSync先导模型比较 | `run_brainsync_pilot.py`、`transfer.calibrated` | 预声明完整session划分及匹配scratch；校准内选方、先冻结赢家再test、最后独立保存全数据refit包 |
| 决策与分析 | `models.candidate_evidence`、`transfer.outcomes`、`research.evaluation`、`stats.hierarchical` | 完整候选词表、R曲线、失败分母、participant统计与结果身份 |
| 产品block完成推理 | `brainsync_model_task.py`、`data.brainsync.load_brainsync_inference_session`、采集端`decoding.py` | 45闪烁保存后无标签推理、模型重载、R1-R5、事后反馈；模拟端到端已验证 |
| 逐chunk流式推理 | 尚未实现 | 不用完整recording推理冒充每个SDK batch的在线更新 |

```mermaid
flowchart TD
    S[公开原始 ERP] --> V[身份与输入合同 / binary 源视图]
    V --> P[N2P3Net 源预训练]
    P --> K[源 checkpoint 与统计]
    C[本人已知目标校准记录] --> A[SubjectAdapter]
    K --> A
    A --> M[个体模型与版本化推理包]
    T[独立后续已标注记录] --> E[离线评估入口]
    M --> E
    E --> D[候选证据聚合与失败账本]
    M --> B[个体模型保存与重载]
    U[新45闪烁无标签完整记录] --> I[目标无关接入与 R5 决策]
    B --> I
    W[SDK 连续 chunk] -. 待实现 .-> J[逐chunk模型推理]
```

图中虚线表示仍缺少的逐chunk接口；实线产品链已通过模拟Qt端到端验证，不能替代实机测试。
旧manifest只注册preprocess，新工作站通过结构化后端任务接校准/inspect/infer。
采集端`decoding.py`负责选择、任务状态和模型生命周期；`decoding_results.py`使用现有
pyqtgraph展示校准曲线、候选内部AUC及九数字证据。报告消费`brainsync_model_task.py`
已核验的结构化结果，不以相邻目录的summary替代个体包检查。
采集端`theme.py`集中维护工作站配色、Qt控件样式、图表色板和既有图标着色；
`resources/icons/theme/`保存由原图标生成的原生控件位图，不在运行时写入资源。
主体主题不覆盖独立刺激窗口，显示密度保留各布局的基础间距比例。
采集后报告由独立前端`quality_reports.py`排队，通过`report_brainsync_quality.py`调用
`research.recording_quality`生成JSON、HTML和PNG。它复用无标签BrainSync接入、共享预处理及
`transfer.quality.transform_epoch_quality`，零可用epoch也保留完整账本；不执行QC拟合。
个体包的“冻结输入重放”由`CalibratedPredictor.load_inference_session(session_dir)`持有：
从包内原合同恢复预处理、EOG和连续QC，返回已通过输入与前瞻时间门禁的无标签数据；不拟合、不预测。
推理和个体包质量报告共同使用这个interface，不在CLI重建包内处理步骤。
`data.brainsync`集中参考协调与最终通道选择：EOG/连续QC已执行参考时直接选择保存的通道子集；
epoch后执行参考时先保留实测参考通道，完成相减后再选最终通道。
无标签loader对同一次核验的录制对象执行`recording_validator`，再进入信号处理；个体包先核验
人员、校准会话排除、标签就绪时间、模拟声明及已有采集参考声明，零可用epoch也不能绕过。
软件回归与改动前归档见[冻结输入重放验证](evidence/brainsync_input_replay_20260909.zh.md)。
前端仅写派生分析目录，报告指针与会话归属交叉核验，取消及应用关闭沿用ModelJobRunner进程树清理。
legacy与候选选择器均保存每个候选的`fit_history`及独立`refit_history`，
通过个体模型的`selection`与任务结果`internal_validation`恢复曲线；旧包不补造历史。
采集参考声明从实际dataset provenance进入`input_request`，与CLI请求逐项校验。
两个采集端版本的实现、兼容性和候选记录见[源码比较](evidence/brainsync_versions_20260906.zh.md)。
新目录现已合入旧目录的EOG/坐标修正并修复录制并发问题；历史raw没有被静默改写。

## 已有模块应如何复用

2026-09-16 `validate_calibration_raw_fit_scope`成为个人preflight、旧fit、候选及fit缓存共用的raw拟合范围门禁；
legacy配方在QC路由前统一解析为`TargetRecipe`。`continuous_filter_plan`在原预处理module内同时提供
分段坐标和两侧原因，`continuous_filter_segments`保留坐标投影；滤波、恢复和执行凭证共用该解释。
源adapter在数值处理前排除永久坏非参考导，坏Cz拒绝当前投影；最终通过既有montage接口补mask/零槽。
source preparation `/2`绑定实际raw/sidecar及方法身份，恢复同时核验NPZ和record凭证，轻量准入不解压矩阵。
`analysis_seed_pairs`/`analysis_source_arms`贯穿源任务、目标计划、结果；分析器复用同一配对引擎生成组内及跨源对比。
见[完整调用链验收](evidence/algorithm_chain_repairs_20260916.zh.md)。

2026-09-15 `data.preprocess.execute_preprocessing(raw, events, preprocessing, **layout_options)`
集中完整Spec验证、参数解释、实际源sample支持及执行凭证。BrainSync、MOABB raw、GTN、
BI2014a、BNCI2014-008、manifest及source_records七处已迁移；`build_subject/load_dataset`
用显式`preprocessing`接入，低层`preprocess(n_times=None)`仍可做研究。
`transform_epoch_samples`统一实际FFT→截长→半开基线，恢复矩阵仅开放线性能力，
`recovery.epoch_linear_projection`保留有界缓存与分块构造。
`PreprocessingSpec.record()`只表示声明；`provenance.preprocessing_execution`单独保存实际实现、
模块加载时源码摘要、SOS、分段和源网格。cache合并保留全部父凭证，历史缺失记unverified_inputs。
模型门禁分别绑定可迁移的算子身份和不可混同的数据来源，不比较不同录制长度/原始摘要。
来源、单位、候选含义、QC拟合范围和交付时钟仍归各adapter；MOABB高层保留独立能力门禁。
数学依据见[科研裁决](evidence/preprocessing_contract_research_20260915.zh.md)，
实施和验收见[执行器落地](evidence/preprocessing_execution_20260915.zh.md)。

2026-09-18源接入与排程分为三个Module：`data.source_parsers`持有连续格式Adapter、标签与实际伴随文件；
`data.source_admission.PreparedSourceInventory`核验准备状态、merged NPZ/record摘要、物理研究库与预处理，
返回ready/pending/failed；`research.source_catalog.SourceCatalog`解释源池和整库排除，
`research.source_matrix`编译任务及等量剂量。池成员不是数据准入状态，B0--B7仍保留原153任务，
不存在按“候选库”名称额外禁用已执行B7的开关。新组合通过显式JSON加入，不覆盖既有池名。
source chain可用`source_studies`或`source_pool`声明任意数量的源，逐项与实际cache核对；
旧未声明源集合的计划保留原三源/家族两源约束。S2单源剂量按实际源域数推导，
common-three仍要求逐行真观测，不以补零绕过。选取单个matched任务时保留整折比较池与seed来定配额。
BrainForm原始CSV、Cattan连续MAT及BI2012已经接入同一source preparation Interface；
BrainForm保留Pxx、1245采样裁切坐标、校准标签和显式坏导，Cattan不按block重置滤波，BI2012排除Fz ground。
Chailloux现在同时绑定实际读取的vhdr/eeg/vmrk，替换其中任一文件都会使旧准备记录失效。
source pool入口支持配置预处理、无拟合预检及有截止时间的准备等待，输入摘要绑定后变更必须另开结果目录。
`run_source_pool_matrix.py`调用共享`fit_or_reuse_source_trajectory`；公共目标复用`execute_target_tasks`，个人矩阵复用
`fit_calibration_candidate`/`SubjectAdapter`及`CalibratedPredictor`保存与重载。
`data.source_records.SOURCE_EOG_PREPROCESSING`显式绑定本轮单向陷波合同，与工作站日后默认值分离。
`train.source_sampling`将不变CDF预计算一次，保持NumPy choice相同抽样和RNG流；
大源审计向量用`encode_source_vector`无损编码，消费者通过`decode_source_vector`同时接收旧列表与新摘要绑定编码。
等量源按每个目标留出折选择参与者/正负样本配额，先截取每库实际行再合并，避免将全池常驻复制到每个seed。

2026-09-09 `data.preprocess.design_continuous_sos`统一构造“显式陷波→带通”的SOS级联，
实际连续滤波、`data.recovery.filter_sos`与连续QC恢复门禁共用。BrainSync默认50Hz/Q30，
`PreprocessingSpec.notch_freqs_hz/notch_quality_factor`贯穿所有通用raw builder和持久化门禁。
`PreprocessingSpec.record()`统一序列化；历史缺省字段仍表示无陷波，保持原签名，不继承当前默认。
缓存、校准、质量报告的JSON配置通过`load_preprocessing_spec`读取，保存模型的报告/推理恢复其原算子。
MOABB高层离线P300入口未实现显式陷波，遇到该请求明确拒绝；其原生因果raw入口支持。

2026-09-08采集REF合同由`data.contract.brainsync_reference_record()`导出为
`configs/brainsync_acquisition_reference.json`，独立前端从`models/acquisition_reference.json`读取。
session的`montage.reference_details`、BIDS的`EEGReference/EEGReferenceDetails`与EEG通道参考
在接入时交叉校验；EOG保持自身显式导出。模型任务分别保存采集参考和最终软件参考。
`correct_brainsync_reference.py`只在显式旧声明和证言范围内更正元数据，先归档，检查原EDF不变；
连续QC可在全部原拟合EDF摘要一致时生成声明更正副本，系数不重拟合。
原生七导和Cz六导的QC/模型独立；CLI显式native请求不得被QC或EOG配置隐式改成Cz。

2026-09-07连续质量软件接口已实施，见[实施证据](evidence/brainsync_quality_implementation_20260907.zh.md)。
`data.continuous_qc`持有冻结检测与恢复合同，`data.recording_clock`校验前端样本账本，
`data.preprocess`消费完整排除区间。`data.ocular`绑定单HEOG接法和有效fit-sample范围。
`transfer.quality`以真实session/block分组拟合个人信号证据QC或显式历史`data.artifact`，人员身份保持独立；
calibration_candidates执行inner/final生命周期，calibrated保存状态并用NaN显式表示不可用证据。
CLI及缓存记录处理摘要，R5从原始请求账本保留失败分母。没有自动去阶跃或自动认证恢复门限。

2026-09-08`data.recovery.EpochStateRecovery`提供校准经验状态包络，`continuous_qc`通过
`signal_dependency_matrix`投影参考/眼电依赖，`brainsync`输出逐epoch/导的有效mask。
`transfer.quality`统一epoch模型保存/读取、原拟合数据摘要校验和应用；候选、fit复用、个体模型
共用`epoch_quality_model`冻结模式或`epoch_quality_policy`拟合模式，两者互斥。
公开训练入口拒绝预先做过epoch QC的缓存；内部应用一次并将同一质量状态存入个体包。
LDA完整观测行选择和预检共用接口，分别记账请求/实际拟合数据。连续QC的最低有效通道数
在后续epoch QC后仍生效，拒判保持NaN与原450事件账本对齐。
开发对照由`run_brainsync_recovery.py`复用正式拟合/模型保存/预测接口，
`analyze_brainsync_recovery.py`生成同事件波形与可用性图，见[恢复记录](evidence/brainsync_recovery_20260908.zh.md)。
`brainsync_model_task/2`结果分别记录原始排程数、输入epoch、预处理不可用、模型内不可用、
有效证据及总不可用；分层计数必须与完整排程数相等，不能只报告模型内QC损失。

`data.personal_quality`负责逐导物理幅度/STD/差分粗糙度的冻结诊断包络及精确恒值兜底。
`transfer.quality`按method/schema严格分派新旧模型，禁止把旧相对PTP数字解释成新方法。
BrainSync新方法在拟合、读取冻结QC和个体模型加载时均要求已应用匹配的逐导连续QC `/4`，
CLI预检及前端设置同步拒绝缺少参考前恒值检测的配置。连续检测实现仍由`data.continuous_qc`持有。
新个人包写`n2p3_calibrated_predictor/4`，显式兼容通过原合同的 `/3`、`/1`，不接收 `/2`。
fit复用键含训练行QC特征摘要；测试特征不改变拟合身份。冻结fit事件、物理X、mask和特征均核验。
三臂开发runner可在一个worker内复用一次源checkpoint加载和共享合法校准缓存，各臂拟合状态独立。
最终门禁下57模型/114预测只读重放已通过，见[本轮证据](evidence/brainsync_qc_redesign_20260908.zh.md)。

后续四项联动增加`data.slow_trend.analyze_slow_trends`和`FittedContinuousQC.signal_diagnostics`。
loader在原始域计算诊断摘要，写入`continuous_quality_ledger.signal_diagnostics`；完整窗口可由
`include_windows=True`独立导出，不向模型重复写入大型逐窗表，也不据此修改硬mask。
`data.recovery`只读证书接口给出放行成本与持续输入不在认证内的声明，旧恢复模型record不变。
`data.brainsync_contract`交叉核对sample clock与pause/BIDS边界；前端runtime/ui实际样本等待器
覆盖baseline与每block尾窗。跨仓库controller单测必须显式提供模拟等待器，不能冒充真实采样确认。

`IndependentRecordingScope`将已声明的完整校准/测试会话透传所有BrainSync loader入口。
它只替换显式离线实验的时间排序条件，仍检查会话不重叠、fit来源、人员、参考及原始支持。
工作站默认无此scope；新测试仍受校准就绪时间约束。EOG和连续QC的拟合会话均不能超出声明校准集。
`brainsync_model_task --preprocessing-config`传完整`PreprocessingSpec`，fit/preflight的
`declared_preprocessing`显式核对实际输入；未声明检查当前个人合同，推理只从个体包恢复算子。
`run_brainsync_recovery`支持无源scratch/LDA，保存实际EOG模型，统计真实源加载次数。
新EOG参考策略由`data.ocular`持有，dependency矩阵、拟合CLI、个体入口、GUI选择和重载共同消费。

离线runner显式支持`evaluation_mode=independent_recording_holdout`：校验会话/事件不重叠、
同participant及同输入合同，按预声明独立会话留出，不按录制时间排序裁定；缺省仍为
`prospective_recording`。工作站新测试的时间门禁保持原产品生命周期。两种协议随结果保存。

- 数据合同：`data.contract`决定时间/滤波/基线；`data.channel`决定物理通道；
  `data.identity`与`data.lineage`决定身份。runner不得自行猜别名、维度或参考。
- 源模型校验：`transfer.checkpoint`统一校验模型结构、输入统计、源训练身份与目标排除。
  描述性来源列表不能替代实际`training_identity_ledger`。
- 校准训练：`SubjectAdapter`统一初始拟合与完整prefix refit的训练损失。
  AS-MMD有独立源上下文，但复用两阶段相同批次规则，不在runner重写弱化版本。
- head对照：`SubjectAdapter.predict_features`输出冻结推理状态下的head输入；
  `transfer.binary`复用scaler/收缩LDA，并提供等价复制到二分类线性头的接口。
  后置LDA与LDA初始化的顺序实验见证据索引，未设为工作站默认方法。
- 候选聚合：`models.candidate_evidence`持有词表、membership和覆盖规则；
  不因缺少候选、校准目标范围或试验的测试队列而缩小解码词表。
- 结果写入：复用`research.execution.atomic_write_json`。失败时保留原结果，
  不在各runner复制不同的JSON writer。
- 训练和评估代码：candidate/within/BrainSync/binary入口分别绑定训练snapshot与评估snapshot。
  candidate的batch、promotion、manifest和analyzer同步该区别，zero-shot缓存也绑定评估hash。
  `verified_source_snapshot`只核验声明的物理归档，不比对正在执行的checkout；
  实际执行身份仍需从归档运行或另行逐文件核对。

`data`对`research.contracts`/`research.execution`的少量依赖目前只是语义哈希和原子写入，
两者自身不依赖EEG训练实现。不能仅凭包名认为发生训练层反向耦合并新增一套工具包。
真正的重复职责应由调用证据及反例证明后合并。

## 产品校准与验证

### 训练链条共享接口（2026-09-07）

`transfer.recipes.TargetRecipe`集中初始化、输入统计、BN buffer/affine、其他权重、dropout、
batch、epochs/updates及seed；公共实验与产品候选调用`calibration_candidates`同一拟合实现。
`binary.dataset_calibration_splits`负责请求K/实际K和共同test，`calibration_split`只在授权
校准范围生成合法inner训练/验证。`target_experiments`通过`--stage compact`编排当前精简矩阵，
复用生产评价builder和候选拟合接口；原binary入口通过`--training-plan --stage`进入同一整库排他程序。
plan声明`target_stage=compact`、`target_batch_size=512`，使用明确的用户固定batch规则；
三类基线和三个组件消融直接拟合当前完整校准，无inner搜索和额外refit。
`weights=head`配合`bn_affine=frozen`冻结学习型trunk；online buffer和checkpoint dropout
仍按声明运行。随后授权的扩展全量通过`include_historical_controls=false`排除E1/E3历史小batch，
源链默认由`research.source_execution`根据实际CPU配额/显存预算调度外折和布局任务组；
spawn隔离CUDA和随机状态，全局GPU信号量协调source、E2候选和外层目标组的并发。
池内worker顺序复用，每次拟合仍从声明的seed和初始状态开始；Manager只加载调度合同，
训练与评分依赖在实际调用时导入，N2P3训练不加载未使用的Braindecode对照模型。
外层组只持有一个attested目标cache，`TargetResultReuse`按完整处理键复用基线或相同源状态评分，
通过`transfer.result_reuse`核对已完成record/cell、预测与完整模型文件的摘要，
从同一份已验证字节解码，重新生成当前消费合同并记录原结果来源。缺少原始完整性声明的
旧产物不进入自动复用；历史分析保持可读。完整性异常独立于被试拟合失败。

公共源池与本人EOG矩阵的已完成产物统一通过`transfer.result_reuse.admit_completed_result`准入。
该interface要求生产入口的完整当前声明，内部按公共target run与本人排程结果分别核验。
公共`target_context`与`build_manifest`同时服务执行及续跑：身份、具体行、配方、环境及快照均比较，
再核验`tasks.json`、逐cell记录、模型和预测；保留已完成矩阵中的`fit_failed`及其分母。
本人`personal_result_contract`绑定稳定单任务、完整recipe、测试会话、输入摘要及环境。
生产时封存逐session NPZ和正K个体包，K0没有目标拟合模型；NaN保留为无证据，Inf及行错位被拒绝。
个体包从同一份已验字节调用原加载门禁，任务、配方、历史和原计划声明交叉核对。
`--task`子集改变本次队列但不改变单任务科学声明；队列计划追加保存，原`plan.json`和原模型来源保留。
`SourceChainResume`继续持有历史源清单；`verify_pool_matrix.py`继续承担独立预测重放，不代替当前文件核验。
实现和反例见[完成产物准入](evidence/completed_result_admission_20260909.zh.md)。
公共`target_experiments`直接调用`calibration_candidates`，不再经个体包模块重导出候选接口。

`PreparedTransferCheckpoint`为一个固定checkpoint/目标dataset组合完成一次完整校验，
随后从已验证状态复制每人的独立trunk，并逐人检查训练身份排除。E2与公共外层runner
复用其已验证合同摘要，不在每个K重新递归序列化包含逐行账本的训练合同。
`SourceChainResume`仅从停止后的哈希清单读取已完成源任务/选模预测，核对配方、行、cache与原快照，
保留原checkpoint路径与来源；新训练及新评分绑定新的执行快照。

E3的`CalibrationCandidateFitCache`由单个人/实际K的校准scope持有，切换scope即释放。
缓存绑定完整recipe、训练数据与身份、源参数/BN/输入统计和运行配置，区分inner/full-prefix；
命中返回独立状态，并在当前验证集重新评分。按验证CE选epoch时验证内容也进入键，final test不进入。
四选择器可共享相同候选及相同winner的完整prefix refit；`fit_execution`记录实际拟合和复用次数。

`transfer.epoch_selection.EpochSelectionPolicy`集中目标fixed/full_trajectory/patience策略；
`fit_with_epoch_selection`复用校准内时间划分和候选拟合，丢弃inner状态后从原初始化全K重训。
`SubjectAdapter.validation_behavior=select_epoch_full_trajectory`选轮但不提前停止；既有`select_epoch`
保留patience停止，`diagnostics_only`不选轮。停止参数进入精确fit缓存键，旧TargetRecipe字段/默认值
保持不变。`--stage ES`由两个公共CLI进入同一任务builder；保存模型同时记录停止策略、选中配方与
两阶段账本，分析器配对精确K并汇总完整成本。产品E3默认五候选流程未自动加入早停。
ES的`epoch_selection`还可声明`epochs/treatments/modes/include_baselines`，用于固定轮数的
精简配对矩阵；recipe、任务预算、训练与保存模型使用同一显式轮数。缺省仍为三模型三策略60遍。

`DeepBaseline`固定updates分支复用`GpuPerformanceScheduler`、`MatrixBatchSource`和现有优化器策略；
逐更新在设备端累计loss/梯度有限性，在已有进度点、snapshot及最终发布前检查模型状态。
失败清空候选snapshot并使模型失效。`fitted_input_statistics()`统一导出实际推理仿射变换；
关闭标准化时导出零均值/单位尺度，经验统计保留在诊断账本，两个源导出入口共用该Interface。
`SubjectAdapter.fit`从开始到完整refit均按一个成功状态提交；异常和中断统一清理统计、校准
输出及历史，禁止部分更新状态进入预测，保留重新拟合能力。
`train.source_sampling`的普通概率采样与已有TSD/PDS显式分开。`research.source_selection`
持有源内候选轨迹、身份/观测账本、剂量与选中预算的从头refit；
`run_pretrain_supervised.py --source-chain-plan`和`run_source_training_chain.py`提供生产入口。
source runner的共享trajectory registry仅对有序真实event/person/domain、X/y/mask、
完整训练配方、seed、设备及源码一致的已有update快照复用。原checkpoint不改写，消费方容器
保留自己的父cache/fit ledger，并记录原checkpoint SHA及TrainingRunContract digest；
`physical_optimizer_updates=0`明确表示本次未训练，不把复用伪装成新的随机重复。
登记表按语义签名持有OS级跨进程文件锁，完成训练和哈希发布后才允许相同任务复用。
每组保存独立record及轻量进度，父进程聚合；已完成模型可以从显式哈希清单跨多次续跑复用。
GPU任务结束释放缓存并重置该worker的编译缓存，避免旧optimizer对象累积触发全局重编译上限。
`transfer.mask_probe`仅用固定checkpoint删除实测通道，记录激活和预测，不拟合或生成缺失通道。

`research.evaluation.verified_executed_snapshot`同时核验物理归档、逐文件摘要和执行checkout，
检查新增runtime文件是否也已冻结。只有归档整体hash、或只核对manifest声明的部分旧文件，
不足以绑定新训练入口。`freeze_source_snapshot.py --materialize`可生成隔离执行副本。

产品预算、指标和trial/decision定义见[研究规范](research_program.zh.md)，
操作方式见[工作站指南](workstation.zh.md)。配置中的目标顺序是示例，正式采集前须
预声明每人的排列；27个测试decision只是每数字3次的开发起点，不是统计功效结论。
`observed_sequence`保留真实目标顺序，包括相邻相同目标；额外的
`forced_switch`和`unseen_calibration_codes`只回答各自明确声明的研究问题。

校准内部选择的完整目标流程是：内层训练拟合预处理及模型，内层验证选择配方/轮数，
从源初始化在完整校准数据重训，冻结后测试。任何学习型QC、EOG回归或输入统计都必须
服从其阶段范围；仅声明“outer calibration内拟合”不足以证明inner validation独立。
当前BNCI的内部选配方结果是独立保存的公开开发实验。新个体模型模块复用该选择方法，
以完整BrainSync校准session做内部验证；自动EOG只在内部训练session拟合并保持冻结。
个体包的模拟/未声明来源标签、原始校准范围和模型参数都在推理时核对。

本人固定LDA的QC路径直接调用`fit_calibration_candidate`完成一次全校准拟合，
与preflight的固定配方范围一致，不借内部选模失败进入LDA兜底。
`legacy_product`在QC路径共用源失败判定，不兼容源及全部源候选失败均终止任务。
研究选择器的显式兜底和原有独立会话门禁继续保留；QC不能改变入口声明的方法。

## 研究扩展

2026-09-16统一模型设计将现有`N2P3Net`、`FeatureResidualAdapter`、参数组、`SubjectAdapter`和个人包作为复用主体，
计划集中扩展源目标组合、源中心收缩和共同event×candidate membership；每个优化工作集只执行一次联合更新。
`decision_loss/candidate_set_loss/candidate_evidence`需共享物理语义，source/target recipe、固定更新、inner/refit、
checkpoint和resume同步迁移。先补全现有互斥所保护的批次/风险/状态合同，再解除互斥。
逐文件顺序和验收见[统一模型方案](evidence/unified_model_plan_20260916.zh.md#7-逐项工程实施与验收)。
上述为待实施设计；以下P1/P2接口是当前已实现能力，不将两者混称。

2026-09-16首批域迁移实现使用两个集中module。
`models.parameter_groups`验证参数归属完整且不重叠，`N2P3Net/FeatureDomainTransfer.adaptation_parameter_groups()`
由实际module对象声明空间、时间、读出、BN及扩展参数；`SubjectAdapter`统一应用策略并记录训练参数量。
`TargetRecipe.weights`新增`spatial/temporal/adapter`，残差配置显式序列化；个人模型按实际wrapper重建并严格核对架构。
`P1`矩阵、公共runner和分析器共用这些配方，完整校准refit继续使用同一interface。

`transfer.source_objective`持有源目标函数配置、身份准入、批内正负对、重复行处置和诊断。
`DeepConfig.source_objective`仅用于固定更新源轨迹：CE原路径保持原计算，SupCon/MMD复用同一CE批次和优化器。
`source_selection`及监督预训练CLI把实际辅助项写入每个更新点的合同，checkpoint入口核对配置与更新范围。
`P2_source`复用既有源组调度、registry和resume；外层`P2`阶段绑定源臂并复用校验后的scratch/LDA产物。
嵌套配置统一用`asdict`序列化，避免`__dict__`把dataclass实例写进JSON或造成续跑身份不一致。
见[实施与验收](evidence/domain_transfer_implementation_20260916.zh.md)。

| 现有实现 | 定位 | 继续运行的条件 |
|---|---|---|
| `models.adapters`、`models.splitbn` | 特征残差、invariant/private、Split-BN对照 | 显式源domain与目标buffer来源，匹配基线 |
| `transfer.asmmd` | 源流/冻结logit联合适配对照 | 两阶段一致损失、源身份校验、批内至少2样本 |
| `data.euclidean_alignment`、`data.domain_alignment`、`transfer.alignment` | EA、ACW、GALIA各自不同合同 | 不互换名称/公式，fit范围和参考可追溯 |
| `train.source_sampling` | TSD与Dense PDS源样本选择 | target选择依据仅来自声明校准，源行账本保留 |
| `transfer.pretraining`、`run_pretrain.py` | 重建式预训练对照 | 不冒充已训练的监督源分类头 |
| `transfer.candidate_set_loss`等 | 合法membership下的决策损失 | 物理候选合同成立，完整词表/覆盖测试通过 |

这些实现不是已证伪模块。删除的是错误或被取代的默认叙述、重复实现与死接口，
不因机制CI含0或未被默认调用而抹去可复用研究能力。

## 目录与数据层级

| 目录 | 所属层 | 处置原则 |
|---|---|---|
| `src/` | 可复用实现 | 不放一次性实验配方、现场路径或历史执行脚本 |
| `experiments/*.py` | 可运行入口 | 编排已实现模块；直接CLI启动应成立 |
| `configs/` | 当前产品声明和通用实验模板 | 不存已退役的日期化主线模板；JSON不自动代表采集器已执行 |
| `tests/` | 当前行为与反例 | 共享造数归`support/`，测试文件不互相导入；合并重复场景，保留数学与合同反例 |
| `doc/` | 六份当前指导与入口 | `research_program`为研究权威；`roadmap`为统一进度；本文件为实现导航 |
| `doc/evidence/` | 结果、推导、来源与历史审计 | 主题索引；dated状态不覆盖当前指导，旧原文保留于压缩快照 |
| `doc/assets/` | 文档引用的结构化证据 | 与源码分开；已运行审计的JSON不是默认配置 |
| `mne_data/` | 公开原始数据 | 保留来源及校验，不重命名电极或改写原始数据 |
| `experiments/cache/` | 派生训练/评估cache | 凭attestation识别，不凭文件名判有效性 |
| `experiments/runs/`、`experiments/results*/` | 执行记录、预测和研究源码副本 | 旧分数不回写；忽略目录内脚本不当成受支持CLI |
| `cloud_inputs/` | 历史/当前云输入分发 | native、EA、source-selection数据保留，启用前重新核对合同 |
| `experiments/cache/history/` | 保留的历史派生数据 | 数据与sidecar一起保留；启用前重新核对合同 |
| `experiments/results_history/` | 历史结果、日志及迁移索引 | 原始字节/哈希保留，不回写历史分数和内嵌路径 |
| `tmp/` | 当前临时探查 | 旧材料已按索引迁移；新材料先核对用途/哈希再清理 |
| `frozen/` | 物理冻结的源码、旧指导与证据 | Git跟踪manifest，压缩包留本地；不作为可导入源码或默认配置 |
| `Paper/` | 文献原文 | 保留来源，不是模型输入或试验结果 |

2026-09-06审计时工作树包含未提交研究变更，审计前非忽略文件已逐文件校验冻结，
见[manifest](../frozen/technical_audit_before_20260906.manifest.json)。该历史base commit
不等于当时执行源码，后续Git提交整理也不改变旧实验身份。当前提交状态用`git status`核对。

历史`tmp/`的16,389个文件已按[数据保留与清理记录](evidence/workspace_cleanup_20260906.zh.md)
分别移至cache history、results history与物理归档；数据和结果保留，迁移逐文件校验。
这些大型本地材料不随Git提交分发，机器迁移须单独携带索引所列目标。
