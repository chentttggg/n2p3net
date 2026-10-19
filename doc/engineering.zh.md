# 工程指南

本页维护环境、验证、资源和交付规则。研究纪律见[主规范](research_program.zh.md)，证据见[执行与集成](evidence/execution_and_integration.zh.md)；当前工作树、并发限额和运行安排从[文档入口](README.md#当前状态与续接)读取。

## 文档维护

稳定规则更新对应主指南；专题页维护结论、实验读数和来源链接；checkpoint维护当前状态与下一动作。交接和记忆引用这些入口。记忆保留长期偏好与检索指针，项目的PID、暂停位置、运行计数和临时SHA由项目记录持有。

同一规则或状态只维护一处，其他位置改用链接。替换历史指导前保留必要原文和哈希，归档按[分发规则](../frozen/README.md)管理。本次整理的原文恢复关系见[清单](../frozen/guidance_consolidation.manifest.json)。

新源码、JSON、TOML、Markdown默认LF，已有显式producer行尾规则优先；
捕获的`doc/assets`和`frozen`原字节默认不转换。不要全仓renormalize或统一格式化
已绑定源码来修风格告警。忽略规则排除本机`.claude/settings.local.json`、`artifacts/`
及`doc/assets`模型/缓存/归档载荷；JSON、manifest和必要文本继续可跟踪。
ignore规则只控制新增分发，不删除本地科研证据，也不自动移除已跟踪载荷。

## 交付与旧构建清理

新构建确认可用后，删除已无复用价值的旧构建，这是交付完成条件。按以下顺序收尾：

1. 记录新构建身份、实际验收结果及替代关系。验证针对交付行为，复用已成立的证据。
2. 列出旧对象的绝对路径、类别和保留或删除依据，核实际运行、暂停恢复、共享依赖及其他作者的引用。
3. 将可删除对象限定为确认退役的构建输出、构建缓存和冗余打包副本。原始数据、科研原始证据、冻结结果、producer快照、暂停checkpoint及其在途依赖独立保留。
4. 核目标解析路径位于声明范围内，使用cmd工具删除。批量失败后对明确目标逐个处理；删除后复查路径。命令失败或自动审批拒绝时保留实际结果及原因。
5. 在该交付已有的`cleanup.json`或清理清单中记录新构建凭证、候选类别/路径、执行命令、退出码、实际删除文件数与原字节数、删除后核验、保留对象及未完成项。文件字节数与物理释放空间分别解释。

清理结果使用`completed`、`incomplete`或`not_applicable`。已确认的旧对象全部退休或有明确保留用途，且结果已记录，才能将清理项记为完成；没有替代构建或退役对象时记录不适用及实际检查范围。后续清理不隐含恢复暂停的科研进程。

工作树优先放D盘并仅复制代码，环境与数据共用。整合后保留提交和必要交接，核无执行/恢复依赖，再按上述流程退休完成树。

## 环境与入口

后端统一使用项目`.venv/Scripts/python.exe`。依赖权威为
[pyproject.toml](../pyproject.toml)，解析版本见[uv.lock](../uv.lock)；
`requirements.txt`只是便捷安装入口，不另立环境真相。
Python至少3.11；Linux x86_64且Python<3.14的Torch/Torchaudio固定2.8.0，
其他平台/版本按项目声明的Torch>=2.4开发分支安装，不能当作RTX5090训练验证。
历史GPU环境为Torch2.8.0/CUDA12.8，当前硬件与可用资源须在运行前重新核对。

环境解析固定使用uv 0.12.9，`uv.lock`中的多个Torch版本分别服务互斥的平台/Python分支，
不是同一环境中的多个解。Linux x86_64云端默认Python3.12，可运行
`sh scripts/setup_cloud.sh`按锁同步并输出实际Torch/CUDA能力；`N2P3_PYTHON`可显式选择
已声明的其他Python版本。脚本不启动训练，`uv sync --locked`与`uv pip check`都须通过。
本地`.venv-gpu`的历史cu130手工覆盖属于独立、已记录环境，不由默认lock复现；
需要复用时核其实际包、驱动和原运行凭证，不能把覆盖后的环境称为lock完全同步。

wheel保留现有七个包名并显式白名单发布；暂不进行影响所有调用者和旧artifact模块身份的
命名空间迁移。`source_preparation_method/3`只绑定已安装的data源码，CLI另在新准备账本
保存并核对实际`preparation_entrypoint`源码摘要；旧准备账本保持原身份，不能直接重签续跑。
仓库型科研编排入口仍从源码checkout运行，不将wheel描述为独立实验CLI发行包。
CI按Linux/Python3.12和Windows/Python3.14分支执行lock、Ruff、必要合同与wheel构建；
硬件、外部采集端和真实科研矩阵仍需各自环境的明确验收。

dashboard只监听loopback；浏览器写请求须来自相同Origin、具有本机Host及JSON类型，
monitor和kill共用该边界。静态HTTP仅发布页面、active-run指针及声明run的四种遥测文件，
额外run目录限`tmp/runs`。本机同用户进程仍属于可信操作者范围。
Windows使用`open_dashboard.ps1 -SshHost <SSH配置别名> -RemoteRoot <远端绝对项目路径>`；
主机/用户/密钥分别可用`N2P3_SSH_HOST`、`N2P3_SSH_USER`、`N2P3_SSH_IDENTITY`指定，
远端路径也可用`N2P3_REMOTE_ROOT`。默认遵循SSH config，首次连接核对主机指纹后写入
known_hosts，脚本强制`StrictHostKeyChecking=ask`，拒绝已改变的主机key。
隧道只绑定127.0.0.1，复用时同时核SSH进程、目标和转发端口；页面保留一次停止确认。

源checkpoint、公共结果复用模型及BrainSync源模型统一经`transfer.checkpoint.load_weights_only`以`weights_only=True`读取，
仅对NumPy产生的Unicode身份标量提供受限allowlist，不回退到任意对象反序列化；先核验摘要的调用者直接解码同一份已读字节。
2026-10-12本机全部3,806个唯一`.pt`产物逐一比对一致，见[读取兼容绑定](assets/research_resume_20261012/checkpoint_weights_only_integration.json)。
冻结恢复模型跨Windows/Linux重载时保留原序列化模态身份，
重算特征值仅容许1e-12绝对舍入差；原QC摘要不改写，错序/明显变化仍拒绝。
复现实测差1.51e-13且恢复样本数一致，见[新源校准记录](evidence/brainsync_s1_calibration_20260908.zh.md)。

NumPy/SciPy/MNE承担信号与格式处理，sklearn/pyRiemann承担经典基线，PyTorch承担神经网络。
CLI使用argparse与显式JSON合同；Hydra不在依赖中。配置优先来自数据类和已有接口，
不得凭旧说明重新引入已删除的默认YAML。

```powershell
uv sync --locked --extra signals --extra baselines --extra dev
.\.venv\Scripts\python.exe experiments\brainsync_model_task.py --help
.\.venv\Scripts\python.exe -m ruff check src experiments tests
.\.venv\Scripts\python.exe -m pytest tests/brainsync/test_calibrated_predictor.py tests/brainsync/test_brainsync.py
git diff --check
```

测试文件按改动范围选择，上例不是每轮必跑全集。入口/import变更还要从项目外目录验证CLI。
pytest默认`-rs --strict-markers`：汇总列出跳过原因并拒绝未注册marker；独立前端与Linux云端专用测试的跳过不等于通过。
BrainSync原生SDK/Qt使用采集端自己的Python3.10环境，通过结构化进程参数调用后端；
不向后端注入另一minor版本的原生扩展，实际启动方式见[工作站指南](workstation.zh.md)。

采集端自2026-09-15使用自身`pyproject.toml`及`uv.lock`：在采集端目录执行`uv sync --locked`。
默认不扫描D:/Temp或其他项目；仅显式SDK路径覆盖允许注入外部环境，并写入日志。
SDK发行包版本0.3.0的运行时自报0.2.0，来源记录分别保存，不能互相替代；迁移时已核对入口与原生二进制字节。
采集端Ruff目标为Python3.10；运行`python tests/run_headless.py`及相关TEMP模拟验收脚本。
两种传输均按250 Hz运行；`test_transport_sample_contract.py`覆盖相同回调/样本时钟/暂停/BIDS及
拒绝非250请求，`test_transport_connection.py`覆盖迟到连接、旧回调、断开重试与阻抗后配置恢复。
`tests/verify_transport_ui.py <TEMP输出目录>`在真实qasync下使用FakeSdk检验串口/BLE启动与预览，
不扫描或连接硬件。测试证据见[连接统一验收](evidence/domain_transfer_major_20260920.zh.md)。
2026-09-20发布构建位于`D:/BCI/BrainSync_Release_20260920`，使用PyInstaller onedir并随附`_internal`运行时；
构建时排除未使用的Qt WebEngine/Qt3D/Quick等模块，排除`_internal`后的发布载荷128.319 MiB，小于200 MiB，
不复制`.venv`、源码、测试、开发入口或历史数据。启动探针使用`BrainSync.exe --mock`，
不代表真实设备验收。完整清单见[发布版EXE验收](evidence/brainsync_release_20260920.zh.md)。
Qt测试必须注入临时`app_root`，在同一qasync循环内完成`window.shutdown()`，不得读取操作员设置或
以强制`_close_ready`跳过进程回收。应用直接quit与启动中的模型任务须有真实子进程回归。

前端异步测试要区分“请求已取消/超时”和“原生命令已结束”：未结束的worker不得与新任务并行操作同一设备。
测试超时恢复时使用可释放的worker，先证明重试被拒，再释放并验证恢复；不能以修改布尔值伪造原生命令完成。
活动会话租约由OS随进程退出释放；恢复测试覆盖真实子进程崩溃及journal关闭失败。
滤波数值回归包含250/500/1000Hz预览，不能只用有限值断言，须检查幅度和分批连续性。

工作站计算设备通过结构化`--device`传入，`auto`复用`train.device.get_device`，
显式CUDA先检查后端可用性和设备编号；LDA计算仍在CPU，不能按GPU选项声称LDA加速。
修改个体入口时覆盖QC开关下的源失败、固定LDA全范围EOG/冻结QC、dry-run与真实执行一致性，
并验证模型重载及无标签推理不改写个体包。采集端设备设置也须测试持久化与忙碌状态互斥。
冻结输入重放改动同时运行`test_brainsync_input_replay.py`、`test_recording_quality_report.py`、
`test_calibrated_predictor.py`及BrainSync/参考/EOG/连续QC接入测试。用真实合成BIDS覆盖原生参考、
epoch后Cz、EOG和QC拥有Cz参考、保存的通道子集/顺序、旧无陷波包；按实际张量与事件账本比较。
零epoch路径先核验录制准入，普通合同失败不能降级为拒判；报告禁止分类器执行，原始文件和模型只读。
2026-09-08本轮入口修复在当前后端环境全套1539通过、4跳过；前端35项通过。
CUDA实际拟合、保存重载及CLI推理验证见[执行计划](evidence/domain_transfer_major_20260920.zh.md)，
该合成验证不构成CPU/GPU吞吐比较或成人效能结果。

采集后质量报告需要`signals`依赖组，Matplotlib已显式声明并由`uv.lock`锁定；绘图使用Agg后端。
报告后台固定一个计算线程，避免占满采集工作站。Qt集成验证入口位于独立前端的
`tests/verify_quality_report.py`，使用模拟采集和独立输出目录，不能据其波形或报告认定真实信号合格。

## 开发与验证

[本轮实际机制与来源](evidence/research_mechanisms_20261009.zh.md)记录本地A01六格空间prior、native边界及自适应输入挑战；完整公共训练/跨host与一次最终整仓验证仍依各自实际环境验收。source ledger同一次同步只读调用在真实validate后用私有row digest，公共entry继续验证、新调用重新hash当前输入；39.3MB合成账本六对实测约1.34倍，只关闭该重复扫描，不称云全阶段吞吐。

g raw研究expert显式提供`kernel_prior=KernelPriorSpec.calibration_g(g)`，同时不给ridge；默认旧ridge1与新g互斥。compiled root只在fit/C或load ingress验证，数值η循环消费同一immutable root；rank failure有`KernelPriorRankError.record()`结构化证据供runner保存。新/5实际numerical payload和source门禁不接受旧数据/recipe重标，见[实施](evidence/calibration_g_prior_20261006.zh.md)。

[多专家实施](evidence/ensemble_implementation_20261005.zh.md)将概率/producer/物理身份检查集中到入口及frozen载入；
`calibration_state.FrozenTrialPredictor`复用真实decode并取代ERP comparator重复实现。研究CLI冻结全部src与实际入口，
仅捕获明确ExpectedSubjectError/数值分解失败为fit_failure，合同错误传播并保存failed进度。
本轮编辑Python文件以局部.gitattributes固定LF，actual源摘要按执行bytes保存，不以Git父SHA替代工作代码。
完整测试对应`0f8074a`，最后只读调整对应`d3e6027`并另有定向及独立复验；精确命令/退出码、跳过原因和两轴结论见[质量验收](evidence/ensemble_quality_20261005.zh.md)。
最终源码与原始质量证据分别物理封存，Git仅分发清单；执行新公开比较时使用已核验的实际源码归档，不用历史成绩或测试通过替代训练吞吐证据。

[fixed32实施](evidence/continuous_fixed_prefix_20261005.zh.md)的solver/estimator由实际noise strategy与AR阶数统一派生并在载入处核对。
`comparison_role=fixed_prefix_ar_order`要求六阶×Burg/conditional-profile、共同32点prefix及相同basis/ridge；实际scored字段与planned字段分开，未收敛是具名fit_failure。
真实路径先测fit/save/score/reload总墙钟，再优化矩阵/梯度热点；开发数据吞吐探针不能代替完整公共配对矩阵。
fixed32 solver固定maxiter1000、maxcor32、eta_max8、ftol1e-10、gtol1e-6，CPU/float64 exact autograd；逐次求值及初末独立核对都计入真实账本。
本地同输入数值窗口推荐2workers×4threads，实际约7.47核；比1×4吞吐高23.4%，双worker峰值约1.939GB。完整participant的内存、并发与总墙钟须在M1另测，不泛化为云吞吐。
M1从物理冻结的materialized源码运行，绑定计划/原MAT哈希/venv；结束后核全部执行bytes。完整测试对应4b99源，计数修复e982源另有定向及独立复验，不混写测试版本。
实际多进程egress消费已保存JSON record，冻结metadata/state不进入pickle协议。M1包续跑仅准入同src bytes与同科学/资源计划，先验证旧execution/physical freeze和relative包成员，再复制旧bytes；每格fit producer、实际fit/reuse与score来源分列，坏完整包不fallback。

2026-10-05 [分块方案](evidence/deep_research_plan_20261005.zh.md)将123愿望映射到15块，
实际研究入口/helper须一并冻结；旧Phase4凭证遗漏不能消费端补签。
减少防御代码按“外部ingress/fit/load一次验证、不可变内部状态、纯数值核”实施，
保留身份/fit/时间/完整候选/失败分母与artifact守卫，不以多层catch或默认参数吞掉失败。
[B08能量匹配控制](evidence/prior_control_20261006.zh.md)由同prior owner派生`λ_C`并集中reference/kind规则，init/Burg/NumPy/Torch/OOF共享冻结root。真实28臂panel必须固定C、basis、prefix、noise与温度；typed rank failure保持requested分母及actual scored0，矩阵不得从原M1旧包改签复用。
[B06条件路径](evidence/conditional_state_20261006.zh.md)由noise owner持有gain/Schur与支持合同，operator及CLI只消费冻结状态。生产SHA按实际执行字节绑定；相关源码与测试以LF跨checkout保持，修后probe与原RED各用自己的capsule。外部signal/means/operator的finite-real检查集中在ingress，数值核不散落重复防御；真实映射和producer核验仍保留。
按用户2026-10-06要求，代码质量优先通读实际调用链、公式及运行证据，再选能区分新旧行为的必要验证；不以重复本地全集代替阅读。真实先验矩阵的后验误判采用原失败数组→独立QR/SVD→后向关系修复，40项关联/实际spawn通过；413文件旧全集不追认为修后源。全矩阵重新冻结完整执行源，原101包保留历史producer，当前程序不改签复用它们。
[B07字典维护](evidence/erp_dictionary_20261006.zh.md)采用一次compile/载入关系检查、immutable basis内部复用，η中不重造dict或hash；actual C信息与容量/后验共用owner，消除自签mirror。20项必要行为及原MAT三包重放、两轴复验替代重复本地全集；保留修前755/431 cap与RED、修后a981/432 cap的各自source身份。后续完整源训练/seed/caller在B12主线验证。
共享raw/view缓存和ready DAG消费同一CPU总预算、GPU容量及HostMemoryAdmission；四worker各6线程加1调度，
准备/评价从总预算分配，尾部记录capacity/ready/active。本轮仅数学/合同验证，未实测新云吞吐。
有并发原树时采用已绑定当前字节的隔离worktree，分块提交，交付净树；不reset/clean/stash他人内容。

2026-10-04按实际dirty tree先物理冻结，再实施和独立双轴复审，见[实施与验收](evidence/system_implementation_20261004.zh.md)。
`experiments/freeze_source_snapshot.py`包含源码、测试、配置、六指南、Git换行规则及运行JSON/HTML/CSS/JS和许可证；
逐成员核对archive与当前字节。已有脏改动用改前快照区分，不能拿Git HEAD差异替代本轮范围。
历史RED探针保留，修后结果另记；全套通过也不能替代真实输入、失败分母和数学反例审查。

新研究CLI使用显式plan、全新output目录和`evidence_scope`；合成fixture必须标为`software_correctness`。
运行绑定实际读取的同一份输入字节和实际执行源码；任务异常要保存failed状态，不能留下伪running完成记录。
成熟ERP比较逐subject处理全部arm，保留请求但不可达的K及独立fallback账本；source checkpoint必须按真实库/参与者排除和输入合同准入。
每个可执行full-fine在拟合前绑定同输入、实际K、seed、架构和训练配方的scratch及原waveform LDA；
`matched_controls`/`matched_transfer_effects`保存对应关系和失败配对。显式统计contrast也不能换用不匹配scratch冒充迁移效应。
连续BNCI入口为`experiments/run_bnci_continuous_study.py --plan <plan> --output <new-directory>`，
计划示例及真实执行源见实施证据；运行前据本机可用内存确定线程，不能同时物化全四库数组。

2026-09-18源池配置与准备验证集中化，详情见[源模块验收](evidence/domain_transfer_major_20260920.zh.md)。
新研究使用`configs/source_pool_expansion.example.json`描述source_pools、targets的整库排除、seed、
matched_pools、product_pools和独立输入布局。示例默认三seed、四个已用开发目标，不代表准入已完成。
`run_source_pool_matrix.py --matrix-plan <JSON> --prepared <目录> --snapshot <源码清单> --output <新目录> --dry-run`
只核对计划和准备产物。`--preprocessing-config`消费完整PreprocessingSpec，省略时保持旧单向陷波；
固定checkpoint与输入合同仍不得混用。去掉`--dry-run`才执行训练；云端仍先按下节实测资源与吞吐。
准备与拟合重叠时声明`--wait-for-preparation-seconds <秒>`，超时/failed写明相关未执行任务；默认缺输入立即失败。
准备完成数据在`input_bindings.json`绑定manifest、NPZ、record和处理方法；变更需新输出目录，
旧153任务方案及已有产物继续保持原执行身份。新运行记录使用`n2p3_explicit_source_matrix/2`，不覆盖旧/1目录。
`configs/candidate_promotion.example.json`同样是需要真实前置产物的计划模板，
不附带可直接dry-run的EEG缓存或源码freeze。使用前先准备其target/source缓存、原selection
与snapshot清单，保各自身份；不要为运行示例生成冒充真实参与者的空文件。
未传`--root`时相对路径以plan目录解析；传`--root`时全部路径改以显式root解析并受该root
包含边界约束，需先把示例中的`../`改为该root下的实际相对路径。
公共评估`run_public_pool_targets.py`消费同一`--matrix-plan`；每目标的第一个声明池为full对照，
所有配对seed带scratch，第一seed带确定性LDA。`--target-protocols`为额外目标显式声明记录顺序，
不按文件名猜新库会话次序；`--wait-for-input-seconds`允许有截止时间的源/目标流水执行。
`verify_pool_matrix.py`继续按实际结果中的source_task、配方和保存模型重放，无额外池名表。

原始入口继续使用`prepare_source_records.py --manifest <JSON> --output <新目录>`。
`n2p3_source_records/1`每份manifest只含一个真实study，逐record声明subject/session/run、raw相对path、
sha256、sample_rate_hz与reference；压缩成员另给member及成员摘要。整个manifest先验唯一性检查后才启动worker。
BrainForm使用原CSV、study=`BrainForm`、subject=`Pxx`、run=文件时间戳；显式传入既有审计的
excluded_channels及channel_exclusion_reason，输入文件保持完整。Cattan使用study=`Cattan2019_VR`、
原MAT、subject数字、session=`PC`或`VR`；已知subject_01_VR的损坏末采样显式声明
valid_samples_exclusive_end=289814，其余不裁。BI2012直接消费training/online原MAT或ZIP成员，
角色和Fz ground由实际reader保留。Lee不因下载状态或更换名称而获得未实现的格式/转换认证。

2026-09-16统一模型研究资产位于`doc/assets/unified_model_research_20260916/`，
四个独立数学脚本及JSON检验参数/梯度、membership、输入信息和因素设计；主链实施证据另存
`doc/assets/unified_model_implementation_20260916/`。真实raw三记录探针验证输入范围，不推断识别效能。
实施按[分段验收清单](evidence/unified_model_plan_20260916.zh.md#71-必过反例不以旧全套替代)推进：
先对应合同/反例与Ruff，跨主链最终集成再做完整套件。新架构、loss组合、采样风险、源中心尺度、
BN状态或score定义变化均进入checkpoint/cache/resume身份；旧归档不重标。
实际执行前用`experiments/freeze_source_snapshot.py --output <新目录>`冻结将要执行的合并源码并核验逐文件摘要；
不为每次修复另留完整微冻结。阶段落地后保当前骨干、主要可恢复版本及最终模型/预测所必需的真实producer来源，
已解决失败、重复review副本和一次性工具退役。未关闭问题保必要反例与原状态；不清除并行dirty工作、原数据或唯一结果。
统一主链最终全套2095通过、4跳过，定向反例及六份阶段源码校验见
[统一程序实施与验收](evidence/domain_transfer_major_20260920.zh.md)；最终源码与验证证据已分别物理冻结。

2026-09-16算法链修复以失败反例先行，最终全套1936通过、4跳过，Ruff及diff检查通过，见
[验收记录](evidence/domain_transfer_major_20260920.zh.md)。修复前后源码均物理归档，已有dirty改动保留。
源MMD/SupCon在normalize/dedup前验证原特征和索引，CUDA采用一次设备断言，保留批量MMD；
合法无项返回图连通零，CE批序、draws、BN、optimizer及RNG不变，基础双样本MMD仍严格。
新source准备目录使用`n2p3_source_preparation/2`；方法、raw或sidecar改变以及缓存record缺失/篡改均拒绝旧completed复用，
历史目录保留，新建输出。sidecar未预声明时只绑定本次`preparation_start`，不补写旧冻结来源。
P1/P2分析优先读取执行声明；旧结果须显式提供`--seed-pair`或`--seed-design`，P2另声明`--source-arms`，
不得根据已完成行猜完整设计。局部CUDA验证使用Dynamo eager后端，不冒称Linux Inductor吞吐复测。

2026-09-16域迁移首批重点测试：`tests/transfer/test_structural_adaptation.py`、
`tests/train/test_source_objectives.py`以及个人模型重载、source registry/resume、预训练CLI和checkpoint门禁。
P1的残差处理必须核对实际保存架构；P2候选更新点必须保存当时累计诊断，不能填入轨迹末尾诊断。
固定更新辅助项当前显式关闭CUDA Graph重放并记录原因，仍允许普通compile；正式云训练前需实测这一配方吞吐。
静态目标处理通过`TargetResultReuse`复用已核验产物，源初始化处理键绑定权重和统计，scratch/LDA不依赖源权重。
请求次数、实际fit调用和cache命中分别报告，复用不算新的科学重复。

测试按职责分组，可直接对目录运行`python -m pytest tests/<目录>`：

| 目录 | 验证内容 |
|---|---|
| `tests/data/` | 数据接入、事件和身份、预处理、连续质量及恢复 |
| `tests/models/` | 网络、特征、重建和经典/深度基线 |
| `tests/train/` | 训练循环、源拟合、采样、资源及进程执行 |
| `tests/transfer/` | 校准划分、适配、checkpoint、迁移损失和目标实验 |
| `tests/brainsync/` | 采集合同、个体模型、冻结输入重放和工作站报告 |
| `tests/research/` | 评价、统计、研究合同与实验编排；`promotion/`按声明、输入核验和执行恢复分组 |
| `tests/environment/` | 依赖声明、轻量导入与Python兼容性 |

跨组造数显式导入`tests/support/`；仅在一个测试模块使用的fixture留在本模块，
前端运行时fixture由`tests/brainsync/conftest.py`持有。项目路径统一使用`tests.PROJECT_ROOT`，
不按测试文件所在层数推算。源码快照由`support/snapshots.py`构造，模型权重和身份账本由
`support/checkpoints.py`构造，源码归档不加载模型依赖。旧平铺入口已移除，按新目录定位测试。

滤波相位/边界改动检查`tests/data/test_filter_phase.py`、`test_bidirectional_recovery.py`、
`test_preprocess.py`、`test_notch_filter.py`及个人保存/推理/QC入口；共享默认改变后运行全套。
完整记录先划分再滤波，zero禁止同raw内校准前缀/验证后缀；保存逐事件分数、质量掩码与分母。
双端状态界须包含第二遍耦合，固定epoch映射缓存用真实线性算子生成并核验数值；所有缓存有界。
Windows原子JSON替换对短暂PermissionError最多等待0.63秒重试，失败仍保留旧文件并清理临时文件。
源码、性能基准及重载分别绑定实际执行版本，见[双向联合验证](evidence/bidirectional_qc_20260909.zh.md)。

测试共享造数放在`tests/support/`，复用dataset、candidate、BrainSync、quality、checkpoint及物理artifact；
测试文件不互相导入。辅助函数直接构造当前合同，不加兼容别名、异常吞并或猜测字段的兜底。
相同操作的不同输入用参数化用例；一次拟合可覆盖学习效果、logit和概率，避免为重复断言重训。
2026-09-16按此原则合并数据与统一适配测试：测试文件144→135，相关个人NN拟合105→89，
重复prefix准备和EOG拟合均5→1；完整映射及最终验证见[精简方案](evidence/domain_transfer_major_20260920.zh.md)。
测试数量不是质量目标；删除项必须由同接口的独立行为检查覆盖，先归档，再移除旧入口和无用途的假拟合层。
删除没有独立行为价值的退役状态断言、不可达fake与重复构造，不保留空壳；净删减统计包含辅助代码。
参数案例直接声明输入变更与预期，避免用同一个场景名分别驱动造数和断言的分支链。
成功、输入拒绝、更新失败、验证失败和refit失败按实际阶段组织；夹具按具名通道、会话和时间生成，
不依赖参数位置、全流程第几次调用或散落的维度常量。独立数学预期与被测合同常量仍须显式核验。
保留身份/时间/fit隔离、数学反例、失败分母、产物完整性及保存重载。外部数据或独立前端
缺失时明确skip，不能跳过断言后计为通过。全套运行用`python -m pytest`；pytest缓存支持
`--lf`重跑失败项，不能代替改动结束后的相关测试或大范围改动后的全套验证。
Python兼容检查覆盖`src/`、`tests/`、根配置和`experiments/`直接入口，不递归扫描历史运行快照。
科研集成先逐段审阅相关builder、模块、保存加载和真实caller，复用已封存的绿色数学与运行证据；只为新行为或具体未闭合风险增加区分性检查。真实纵向消费者记录自己的拟合、QR重构、评分和重放成本。全套验收独立运行；根`conftest.py`在导入数值库前默认设`OPENBLAS_NUM_THREADS=1`、
`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，避免多个解释器导入时各自铺满线程；显式环境设置优先。
子进程线程预算仍由相应测试显式指定；资源不足不能通过重试或跳过行为断言来掩盖。

原continuous source stage用独立worker/observer保存start→saved、实际PID/birth、scientific/shim各自OS退出、Job回收及导入源码对预启动code-only归档的关系。原NPZ和完整FrozenSource读回后才发布saved；只读catalog准入与aggregate独立封存。保存后导出失败通过`research.original_continuous_source_stage --admit-only`写入全新导出目录，原stage/model/aggregate保留。首个实际software genesis成功1fit、native1、峰值RSS606564352B；公共scope另由原数据issuer与完整计划取得。

显式`target_native_threads_per_worker`在CPU和CUDA任务均由同一phase builder消费，并计入一个inter-op线程；CUDA另保延迟取得GPU的政策。单worker本地软件可声明CPU总预算3＝调度1＋native1＋inter-op1，拟合退出后的ordinary reader和parent沿同native1/总claim2运行；无需为降低native数虚配第二个内存worker。默认未声明路径及既有CUDA描述保持，实际resolver→actor→parent账本反例和源审已通过。
定位耗时用`python -m pytest --durations=20`。真实spawn、快照校验、模型拟合与重载继续执行；
源训练进程池在组间复用worker，Manager和CLI参数入口不提前导入训练依赖，
Braindecode仅在构造其对照模型时加载。调度日志直接格式化plan/builder已解析的路径，
文件读写边界仍做物理路径与产物校验，每次执行或恢复重新核验输入并计算DAG摘要。

工频陷波的回归入口为`tests/data/test_notch_filter.py`，覆盖50/60Hz抑制、通带保真、因果性、
断点及chunk状态、EEG/EOG角色、原始Nyquist、恢复误差、缓存/模型重载及旧源不兼容。
新增陷波频率或Q须同步实际SOS及恢复证书，不能只向滤波器追加调用而保留旧恢复时间。
历史缓存签名由`PreprocessingSpec.record()`保持无陷波原义；元数据不得重标成新处理。

每轮依次完成理念、数学概念、公式、反例、调用链联动和文档/产物整理。
预处理合同执行的科研探针、14项数学/软件反例与93项现有基线测试已归档，
见[2026-09-15裁决](evidence/preprocessing_contract_research_20260915.zh.md)。七处实际入口已统一，
新显式JSON必须包含完整`PreprocessingSpec.record()`（或全部dataclass字段），空/局部配置拒绝；
旧record省略notch/edge仍按原义解码。低层`n_times=None`不自动产生严格执行凭证。
验收须比较X、事件/拒绝分母、源支持、旧record和模型重放，不能只比较kwargs。
实际模块加载时源码摘要随凭证保存；参数改变不能复用旧凭证。先冻结源码，完整测试期间停止
源码和文档修改；本轮归档与测试记录见[执行器验收](evidence/preprocessing_execution_20260915.zh.md)。
模块明确shape、dtype、metadata、fitted状态、失败语义；复用已实现的合同和算法。
数据合同、划分和评价工具先能服务同一比较，再添加模型；不能因追求速度省去真实builder。

测试重点是训练范围、真实身份与group隔离、时间截止、mask/通道角色、完整候选词表、
失败分母和保存/重载。模型变化还检查CPU forward/backward与参数量；
梯度或目标函数变化用数值等价与反例，旧全量测试通过不证明新科学判断正确。
合同改变同步builder、runner、checkpoint、缓存/结果门禁与相关tests；
删除只为退役接口服务的旧断言，保留仍成立的数学不变量。

source、内部校准train/validation、完整calibration refit、final test的状态来源分开。
QC、EOG、EA、BN及阈值选择服从同一范围，不能只查模型优化器。
EA变换后的坐标由`data.ea_artifact`依据调用者`EAFitPlan`集中完成授权fit、state、transform和产物发布；
10-04复审发现的关系缺口已在本轮实施通过实际父输入重放关闭，旧物理EA普通模型入口拒绝，历史审计用独立loader。
活动统计区分物理证据和数值输入：复用须核fit/score/执行身份，算子实验允许ContrastPlan声明的变化轴；
只有行号相同不构成同一输入。相关组合反例及修复证据见[本轮实施](evidence/system_implementation_20261004.zh.md)。
连续质量改动还须运行`test_continuous_qc`、`test_recording_clock`、`test_acquisition_contract`、
`test_personal_quality`、`test_ocular`和个体保存/推理测试；大范围改动运行全量。
逐导恢复另覆盖`test_epoch_state_recovery`、`test_local_recovery_pipeline`、`test_recovery_contamination`、
`test_recovery_lda`、`test_calibration_quality_ingress`、`test_frozen_epoch_quality`；数值注入必须使用真实
loader/参考/重采样/基线和后续QC，不只检验简化滤波脚本。总污染与重置状态误差分开报告。
前端使用其独立`.venv`并先调用`gtn_workstation.app.configure_runtime_paths()`再运行unittest。
真实EDF导出的填充、失败写入与暂停后响应支持必须作反例，不能只检查JSON字段存在。
本次验证和源码归档入口见[实施记录](evidence/brainsync_quality_implementation_20260907.zh.md)。
个人信号证据QC另覆盖`test_personal_signal_quality`、`test_personal_quality_dispatch`和
`test_raw_constant_quality`：他导/mask不变性、低幅动态信号、宽ERP、原始恒值减参考、持续时间边界、
新旧schema分派以及新方法必须依赖已应用连续QC `/4`。公共旧QC测试保留原语义。
新个人模型包写 `/4`；旧 `/3`、`/1`只按原合同读取。未启用原始恒值的连续QC仍写 `/3`，
启用时写 `/4`，明确5秒等实际参数，不修改旧profile摘要。
2026-09-08最终后端全套1404项中1400通过、4跳过、0失败/错误，前端29项通过。
云端57模型/114预测在最终门禁下CPU重放，有限值mask完全一致，最大logit差1.431e-6；
不是CPU/GPU逐位相等声明。详细环境、反例和归档见[QC验证](evidence/brainsync_qc_redesign_20260908.zh.md)。
结果复用`research.execution.atomic_write_json`，序列化失败保留旧结果。
完成目标结果由`transfer.result_reuse`封存`n2p3_result_integrity/1`：record/cell语义摘要绑定
预测和完整模型文件摘要。读取后直接解码同一份已验证字节；产物损坏或缺少原始声明时明确
中止复用，不计作被试拟合失败，也不在消费时补算摘要来重标旧结果。
矩阵续跑使用`admit_completed_result`一次核验完整生产声明和必需产物；不能只读顶层record后标记reused。
本人新结果在生产时封存全套session预测和可选模型；旧未封存结果保持历史用途，不在消费时补签。
当前入口计划保存在`plan.<digest>.json`，首份`plan.json`保持原字节；progress指明当前队列计划。
对准入改动运行`tests/transfer/test_completed_result_admission.py`及现有训练结果复用、目标实验、
个体包保存加载测试。覆盖真实CLI缺件/篡改拒绝、任务改变、失败cell、K0、完整排程NaN及验证后路径替换。
源模型保存实际执行的输入变换；声明关闭标准化而携带非恒等推理统计的checkpoint被拒绝。
固定更新源训练在设备端累计数值健康状态，snapshot/进度点/结束时检查，避免逐步CPU同步；
适配器任何阶段失败后均拒绝推理和统计导出。回归包括NaN loss、非有限梯度/参数和refit中断。
监控PID存活使用已声明依赖`psutil`的只读查询，Windows禁止以`os.kill(pid, 0)`探测。
仪表盘的监控认领在已有锁中完成；缺失数值保持缺失，真实零值仍保留，见
[全库核查与修复](evidence/domain_transfer_major_20260920.zh.md)。
Dense PDS的GPU距离使用直接差值路径，避免矩阵平方展开的相消；坐标指数超出安全平方范围
时复用既有CPU `hypot`计算，不放宽半径。该边界由float64精度和特征维数推导，backend照实记录。
每个实验保存实际配置、训练/评估各自源码snapshot、训练行、预测、指标、包版本及资源记录。
dirty工作树父commit不等于执行源码；不得用新hash重标旧实验。
本轮QC的GPU执行快照`brainsync_qc_redesign_20260908/source/`与最终门禁完善后的
`final_source/`分别冻结。新增入口门禁不改变本轮已满足 `/4` 的输入及预测数学，最终重载审计单独绑定
实际评估源码。研究/前端/输入与结果压缩逐成员校验，Git只保存`frozen/`清单。

其余四项质量联动增加`test_slow_trend`、`test_recovery_certificate`、`test_independent_raw_scope`、
`test_brainsync_clock_segments`、`test_ocular_native_regressor`及跨仓库采集/个体CLI回归。
新频带必须显式声明完整预处理配置，恢复SOS及包络对应同一实际算子；源模型仍核验原合同。
新EOG策略保存 `/4`，旧 `/3`字节语义保持；完整raw重建不能把冻结EOG记录写成None。
12条件72拟合/144预测从`brainsync_qc_remaining_20260908/source_preparation_v2/checkout/`执行，
并在同一冻结源码下完成CPU重放。启动前先更正继承的旧问题文字，初始未执行计划单独归档，
不得把模板残留描述当作实际fit范围。后续测试夹具及GUI改动另绑定最终实现，不重标训练源码。
详细验证、吞吐和独立证据见[四项收尾](evidence/brainsync_qc_remaining_20260908.zh.md)。

目标早停消融使用`--stage ES`，plan可声明`epoch_selection={"patience":6,"min_delta":1e-6}`。
固定轮数扩展可在`epoch_selection`中声明`epochs:90`、`treatments:["full","scratch"]`、
`modes:["fixed"]`、`include_baselines:false`；后者省略已复用的K0/LDA任务。
轮数必须为正整数，模型/策略列表必须非空、无重复且使用受支持名称。
ES轮数读取`epoch_selection.epochs`，缺省60；普通非stage入口的`target_epochs`不覆盖ES。
`--dry-run`同时列出inner和完整refit的成本上界；`--stage ES`分析记录实际与缓存后的物理成本。
先跑满轨迹选轮、再单独比较patience停止，不能用训练loss下降或外层test确定停止时点。
固定source、学习率和batch512；非法校准内划分按同一固定recipe回退并保留无选模证据标记。

## 设备与性能

### 统一模型的分阶段训练预算

用户最新允许按阶段采用不同停止规则。当前[精简设计](evidence/domain_transfer_major_20260920.zh.md)只在每外折一个源内参考上早停：
patience3、min_delta1e-4、最多12曝光周期。reference inner与refit的范围和成本分别记录；
主源两种配方/三个seed消费同一个冻结e*，每次直接拟合全部合法源。个人迁移和scratch固定90轮，完整C一次拟合。
输入统计、BN重估和LDA为一次估计；K0/只读推理没有停止轮数。主K0/200/800，不默认展开更高K或模块大笛卡尔积。

当前代码已由`UnifiedTrainingBudget`路由新预算；省略`--training-budget`时使用dataclass默认值，
`configs/unified_training_budget.json`是可显式传入的示例，不是自动读取的默认文件。
复用`PatienceStopper`、source fit/refit及直接target fit；`GlobalEarlyStoppingPolicy`和`EpochSelectionPolicy`保留为其它显式研究能力。
来源与反例见[实施验收](evidence/domain_transfer_major_20260920.zh.md)，没有新增每算法一个runner。参考source范围/配置、e*、λ、实际模式、目标轮数均进入checkpoint、计划和cache身份。
相同e*对应不同fit行数时按曝光周期换算updates；固定个人模式不请求或生成无用途的inner raw缓存。
确定性LDA按实际输入和数值程序复用，seed标签不能触发相同的重新拟合。

代表吞吐探针需包含observed pack、两个联合源辅助项、个人完整group、输入统计和保存；
GPU计时同步，冷启动/编译与热运行分列。缓存就绪的训练时间按阶段估算；首次raw准备和传输没有实测时单列未知项。
参考源触顶仍在改善时记录预算受限，外层test不用于扩预算或改停止规则。
`experiments/benchmark_unified_training.py`可在项目环境测A0/A1、Q0/Q1及prior独立oracle，输出冷/热计时与实际预算；
本地5070 Laptop的Q1热拟合+预测由逐tensor prior的0.702s降到pack的0.498s，loss/gradient容差核验通过。
它是合成prepared-array探针；云端正式4/8worker及真实raw吞吐需在当前实例重新测量。

### 云端训练性能优先

独立前缀矩阵的完成参与者恢复使用计划中的`prefix_recovery`，与M1的`resume`分开。
恢复build只核原producer元数据并注册ready工作：旧包零拟合重放和新fit后的独立准入共用
`candidate_participant_admission_adapter`，由同一`CloudExecutor`以CPU evaluation phase拥有进程、
线程、RAM和完整树退休。唯一数值checker仍为`_admit_candidate_participant`，完整graph/model/OOF/future门不跳过。
新fit退出后main先做小型source/plan/record结构核验，再动态注册evaluation；main按实际IPC返回SHA核
新sealed receipt和原metadata/source/request后串行ACK，fitted与admitted分别留账，准入失败不能记completed。
fit完成事实在回调入口独立留存；动态evaluation发布前完成进度写入。写入异常保留原callback失败及全部
requested roster，不能把从未注册的evaluation列为已执行或让finalizer丢失失败报告。
并发进程上限来自同resource owner的`max_cache_group_processes`；显式`target_native_threads_per_worker=4`
时，4 GPU-capable fit各4 native＋1 interop、4 CPU replay各1及coordinator1合为25 CPU，GPU容量仍4。
host/staging与每actor RAM继续由原owner核实际准入；配置可行和软件liveness不替代公共吞吐实测。
独立前缀或source链candidate未显式声明staging时，同一资源owner从中央配置、host预算的15%和
工作阶段剩余内存三者取最小值。有cloud floor时先保各source/target阶段的最低并发内存，
source尾部只保真实group数；不预留两个阶段同时满载。source和target容量均用扣除staging后的余额。
显式staging保原值，无法满足floor时明确拒绝。独立ledger反例核四路target实际全部准入及25线程，
并覆盖四路source、提高到六路的floor和CPU容量；nested parent峰值继续按实际账本单独记载。
共享输入最后一棵reader进程树退休后，先清caller/task的Windows大bytes别名，再关闭holder、归还byte租约，
随后接纳pending准备。executor已退出但未获byte准入的准备保存artifact-admission失败及完整请求，
原OS退出事实保持；source链在全部consumer和additional finalization后保存最终CloudRun状态。
`research.prefix_recovery`绑定原队列request、实际science exit、完整树退休、terminal里的原plan/result字节、
原executed source及物理归档，并核精确审阅后的load/replay/admission/resource源码变化。
候选consumer从原MAT和已保存的全部ordinary/selection包做独立parent零拟合重放，保原producer及
`score_source_digest`；完整records/K/seed/策略/选择器不缩减，只有原record index的未恢复补集进入新fit。
新结果分别记录每人的producer和consumer，原failed运行保持只读。恢复在CPU重放冻结predictor，
captured device用于核原拟合身份；实际阶段耗时、RSS和字节读写须另测，不能把零拟合当零成本。
独立恢复consumer的actor入口在prepare和任何输出写入前调用同一producer输出保护门；
资源模块接入已验收的进程私有exec环境、共享subreaper租约和逐pool历史恢复门。只允许
`prefix_recovery._TRANSITION_ROLES`列出的精确路径变化：原load/admission/recovery/resource入口及
新增source_execution、prefix_replay_dispatch、prefix_replay_execution、prefix_replay_inputs、
ensemble_views的resource/immutable-input transport。每个路径仍须绑定原producer与新consumer的
实际SHA和独立审阅，科学模型的其他改动继续拒绝；旧compiler由原亲核字节执行，旧模型保持自己的来源。
接口的作者软件验证和实际公共恢复范围见[执行与集成](evidence/execution_and_integration.zh.md#独立前缀请求与完成参与者恢复)。

条件连续score按本次actual record共享公共白化列；各candidate保完整AR history、prefix/full QR及原密度。公共函数抽取与新score接点已做相关Gaussian、multi-record和precision反例验证；当前科研fit/posterior/df逐函数保持原逻辑。性能以prepare＋load＋score阶段和完整矩阵墙钟实测；旧模型由原loader准入，显式新数值consumer另记身份。

CUDA target可由`target_native_threads_per_worker`显式声明native线程，并另计一个interop
线程；worker在第一次实际fit/predict前按FIFO取得GPU资源，CPU准备可先执行。CUDA
context或模型仍活着时继续持有GPU claim，直到完整所属进程树退出。Linux以subreaper、
每任务继承标记与PID创建时间核验迟到/setsid后代，保留独立peer和Manager；nested GPU
升级同样尊重已排队target。嵌套artifact writer分别登记尚未写出的预算，outer按完整
并发峰值扩容并结算实际留存字节，不能用尚未落盘的零字节覆盖另一个pending writer。
这些资源边界的原失败及修后49项相关云端测试已保存；实际完整训练吞吐随单库运行记录。

候选ensemble由`ensemble_fit._fit_expert`在真正调用神经fitter前进入compute phase，
ordinary、group-LOO与整prefix选择共享该接点。source-only K0、结构上不可拟合的
专家及零拟合重放不取得GPU；当前冻结predictor的候选评分在CPU执行。原云端49项
资源证据保持原源码身份，科研集成后的prefix调用验证另行绑定。

公开数据集按独立队列逐库执行：`experiments/run_dataset_queue.py`消费逐库冻结声明，
每库仍由原科学CLI持有完整矩阵、25线程/至少4个GPU worker和验收。先保存该库模型、预测、
重载结果与真实终态，再进入下一库；一个库失败保留其部分产物并继续其他已声明库。
队列以独立supervisor保存真实OS退出与进程创建身份，重启后接回在途任务，避免重复启动。
所属进程树退休后才推进队列；未知退出不写成成功，也不覆盖迟到的真实退出。
恢复同时核计划、科学源码和真正文件锁实现的字节；增加数据集须严格保留原队列前缀。
队列的退出事实不代替科学CLI的完整模型/预测准入。Linux冻结副本的21项正式测试及
父进程中断、重复启动、逃逸后代、终态篡改、迟到退出、声明追加和锁实现漂移反例已通过。

用户于2026-10-08再次强调多次遇到25核配额只用1核：必须核真实进程创建身份、就绪任务、有效CPU核、GPU利用、内存峰、阶段进度和完成产物；25线程/4worker配置本身不能作为验收。遇到串行压缩/解压、重复加载/哈希、I/O、锁或内存准入等待，要落实代码并验证端到端路径。2394曾GPU95%但worker突然退出、内存接近90 GiB上限，原因未确定；[原始终态与完整输入基准](evidence/public_source_throughput_20261008.zh.md#完整输入基准与后续源运行终态)已物理保留。本地整机CPU<80%与云端配额利用分别控制，不能混用。

用户随后明确本轮交付目标是**优化底层调用算法并彻底模块化云端训练**：相近实验固定复用同一个执行模块，已有优秀训练实现按当前合同复用。实验计划提供依赖、配方、科学调用和真实输入/输出身份；公共底层统一拥有ready队列、进程与数值线程、host/GPU/暂存准入、缓存复用、阶段记录和异常退出诊断。`GpuPerformanceScheduler`、`MatrixBatchSource`、源轨迹登记表、原缓存writer/attested loader及资源owner作为复用基础；CLI消费稳定接口，资源算法集中在SRC。

本轮统一验收包含准备→完整源选择/refit→目标校准→评估的实际调用链。一个cache组内复用加载与已准入输入，完整源产物完成后按真实依赖解锁目标工作；源/目标共用CPU、GPU与内存额度，CPU fallback按实际常驻占用转账。participant任务分片只改变排程，当前阶段声明的配对seed集合、requested预算与失败分母、C/held授权、fixed/grouped程序、模型保存重放和actual producer由原科学owner继续持有。输出与概率合同分别绑定当前binary/candidate入口。

当前源输入/字节调度和退出诊断切片仍在独立两轴审查，共用执行接口及公共消费者接入尚在实施；此目标段不表示模块化或真实吞吐已经验收。合并稳定源码后统一运行一次所需全集、Ruff及合同反例，再做真实完整输入与实际进程/资源/进度验证。短软件探针、单个phase改善和名义容量均分别记录。

用户进一步要求有云端时尽量在云端跑测试：资源敏感case、集成全集和性能基准默认使用本任务云端锁定venv及实际冻结源码，保留archive/member SHA、真实JUnit/退出与资源凭证。Windows特有行为及少量必要本地检查另行核验，本地整机CPU仍低于80%。本机因RAM不足产生的资源RED保留；同一源码的云端GREEN只证明对应环境与实际case，不改签旧记录。

2026-10-08首个[公共源吞吐切片](evidence/public_source_throughput_20261008.zh.md)将QC按epoch分块，
保持逐行float64归约及最终float32结果；新outer工作cache采用免压缩容器，并由同进程真实writer
以独立production凭证交接实际数据，避免马上重新复制、哈希和解压。`source_view_working_directory`可显式指定
独占工作cache根目录，与持久结果根分开；工作cache的恢复/持久镜像由运行owner记录，不能覆盖旧view。
旧cache/模型不换身份；resume仍走原attested loader。
`source_input_stages.json`记录加载、投影/QC、拼接、写盘/核验与source账本的真实耗时和进程CPU；
端点RSS不是峰值，native线程/GPU worker容量也不是实际利用率。完整云端峰值与吞吐另测后才调整merge并发。
验证使用各任务独占的持久`--basetemp`，每轮新目录且先创建父目录；不使用共享pytest临时池清理未封存证据。

用户于2026-09-18明确要求今后云端统一CPU总线程预算至少25、GPU worker容量至少4。
集中配置为`configs/cloud_training_resources.json`；源/目标worker须消费资源解析器给出的实际原生线程预算，
不能只在plan里写25而在调用处固定1。四source worker时每worker总claim为6线程，
新默认CUDA分支包含5个native线程与1个inter-op线程，另保留1个调度线程；
源内子worker同时受GPU任务容量限制，避免嵌套进程超额争抢CPU。合同或内存无法满足声明下限时显式失败，不能静默串行。
依赖末端只有一两个可执行拟合时记录就绪任务数，不通过重复拟合伪造四路工作量。
该配置为资源声明模板，需进入实际runner的plan并记录解析结果；它不是自动启动任务。
按当前4×18 GiB source组、10 GiB共享staging、host fraction .95，4组同时可就绪时
需要至少约86.32 GiB实际可用host RAM；4×4 GiB GPU任务与80%准入余量需要至少
20 GiB实际空闲显存。这是配置估算下限，不是任务实测峰值；其余占用和进程启动后仍须复测。
`source_load_workers=1`只限制当前共享staging的并行loader，不把GPU worker容量降为1；
任何调整必须用实际ready工作、内存与吞吐证据核验。
大源路径同序通道采用显式只读共享视图；全行source refit避免再次复制X；成功GPU预加载释放归一化CPU副本；
统计量保留原逐行float64归约与最终加权内积，只将临时矩阵分块。父调度器给每worker显式GPU常驻字节预算；
`research.source_execution.PreparedSourceStorage`集中读取容量header、选择实际snapshot文件系统、
准入解码/磁盘临时字节并一次分配最终C-order EEG；每source写入自己固定的行区间，科学adapter仍核真实参考/身份。
`train.source_rows.SourceRows`使source选轮和refit按行分块读取，避免完整raw subset复制；有限值与缺导检查同样按完整epoch分块。
`SourceInputAdmission`分开组I/O容量和共享NN归一化字节，ready NN优先；四GPU容量由原资源owner保留。
source组的host RAM声明不转换成输出盘容量；复用固定source view时不预留新增EEG文件。
source checkpoint先对实际CPU payload执行不保留buffer的Torch序列化计数，再由原named writer准入实际档案字节；
选轮checkpoint复用按原文件大小准入，保持原producer文件字节。JSON沿用原精确writer，进度JSONL按实际UTF-8追加字节准入。
新source view默认位于group目录的兄弟`source_views/`，也可显式选择另一文件系统；它由原cache writer按实际payload独立准入，
`PreparedSourceStorage`仍负责resident/decoded临时内存。写盘不足须在覆盖前拒绝，真实失败后留下的partial继续按实际字节计费。
固定名单中的旧派生cache先按原build SHA流式压缩，再逐字节解压还原核验；删除由同一`SourceInputAdmission.remove_external_file`执行。
该接口重核mode/dev/inode/size/mtime/ctime，仅删除不被任何named file、temporary或目录lease覆盖的外部regular file；同路径只可回收一次。
unlink成功先登记removed事实与同路径guard；随后free探测失败保即时信用0、未确认延期上界，由原OSError携小收据，adapter保存真实删除与可恢复archive来源后原样抛错。二次journal失败附诊断note，由原worker stderr保存；不声称durable receipt已更新。
即时磁盘信用取最后link的实际分配块与删除前后实测free增量的较小值；未释放部分只记为 deferred upper bound，不直接提升可用容量。
每次新申请重新读取实际free，并保留所有live snapshot和uncommitted artifact reservation；tmpfs回收不扩固定RAM staging。
raw、model和唯一evidence不属于此清理名单；压缩/删除失败保留原件和surviving partial。
离线清理也按实际引用区分临时副本与正式载荷：检查所有成功source/target的plan、source链和artifact账本，
不能只凭failed、retry或inactive目录名删除。可再生解压stage保原ZIP、唯一member/header、scope及恢复清单；
已结束测试保最终source/JUnit/terminal摘要，逐regular file记录删除事实和实际free增量。正式selection graph仍是重放证据。
`GpuPerformanceScheduler.complete_input_staging`在train/held矩阵仍存活时测实际进程RSS；CPU回退只在整场持有的source组预算内
转为常驻，超预算在optimizer更新前失败并释放暂存，不用长lease静默降低并发。即使没有shared staging，source组预算仍传入实际worker。
这些端点RSS与ZIP容量估计都不等于全程峰值；tmpfs/ramfs snapshot和工作输出额外计RAM，不能当可回收磁盘页。
现切片支持单组I/O、组内资源准入的多个source loader；跨组共享snapshot磁盘额度由统一executor继续接入后才放开组I/O并发。
源固定更新复用`baselines.progress`的轻量JSONL sink，在原100-update/验证点记录实际更新与验证，selection/refit各有独占目录；
显示fold=0保留原seed。scientific report与已完成checkpoint账本不逐batch改写。旧checkpoint仍绑定原执行快照。


2026-09-18十二库合并暴露单个NumPy Unicode标量的itemsize上限；新cache的metadata_json改用一维UTF-8 uint8，
解码兼容历史Unicode标量，解码后数据摘要不变。原始record准备保留原数据模块快照，新合并/训练绑定后续快照；
不把cache容器编码变化误记为重滤波，也不倒填已有准备凭证。


2026-09-16统一精简矩阵总执行器实测52.1分钟，6138个正式目标NN拟合、246个LDA拟合、4416个拟合请求复用。
该墙钟不含此前补传及提前源折；完整计数、保存模型重放、数值规则及压缩结果验证见
[结果与核验](evidence/unified_results_20260916.zh.md)。旧200-update重放器不能直接验证新90-epoch程序；
预算须包含D0/D1匹配的decision工作集，LDA重放须还原实际数值dtype及标准化次序，不靠放宽容差通过。

2026-09-16新统一预算的5090部署、真实输入吞吐和归档删除证据见[云端运行记录](evidence/unified_cloud_20260916.zh.md)。
同机短测与154723行真实源输入均未显示default编译收益，本轮使用eager；输入准备可与已就绪的完整源折并行，
完成模型经现有摘要准入复用。tmpfs计算输入须另存持久压缩包及恢复根路径，不能仅依赖实例内存保存。

2026-09-16 P1/P2统一入口为`run_domain_transfer_matrix.py`；按cache/被试分片把两个阶段与所有
源处理交给同一目标worker，复用已验证的相同拟合。源端条件MMD用固定掩码批量计算，
保持各domain-pair/class独立中位数带宽和无偏公式；真实热运行约3.25倍改善，见
[运行记录](evidence/domain_transfer_cloud_20260916.zh.md)。三个完整流程seed的统计须用
`analyze_training_chain.py --seed-pair SOURCE TARGET`逐对声明，未指定时仍是原Cartesian设计。

2026-09-09源池矩阵已将大源账本向量无损压缩、固定采样CDF预计算和排程分组验证集中到原模块。
不要在逐update重建不变CDF，不要把百万行统计权重重复展开为JSON列表。
内存规划同时计入原始cache、合并视图、float64统计临时数组及CPU验证进程；
关注cgroup `memory.high`节流与`memory.max`，不能只看宿主机可用内存。
大cache的私有校验副本可通过`N2P3_VERIFIED_CACHE_SNAPSHOT_DIR`放到经配额核验的私有tmpfs，
其占用仍计入内存预算。迁移已完成产物必须先校验副本，再保留原逻辑路径；不得丢弃其原执行快照。

用户明确偏好：**优先缩短实际训练墙钟，不以保持代码简单为由保留性能瓶颈**。
必要的任务图、进程调度、缓存、共享登记表和资源准入复杂度可以增加；以端到端吞吐和正确性验收。
已有独立任务时，默认充分使用当前机器实际可用的CPU配额和GPU容量，而不是固定单进程/单核。

每次云端启动或续跑前执行以下步骤：

1. 调用`performance-optimization`及`get-available-resources`技能。检查CPU affinity/cgroup配额、
   可用内存/显存、CUDA运行时及其他活跃任务；沿用同环境已核验的短时证据，避免重复冗长预检。
2. 分别检查数据加载/解压/合同校验、source训练、E2校准、外层评估及结果写盘的真实调用链。
   对有独立工作的阶段启用资源自适应并发；重复模型和固定元数据只处理一次，跨进程复用须有锁和来源摘要。
3. 若实现仍串行或GPU在等待CPU工作，修改算法、任务图和代码后再按实测吞吐确定执行拓扑。
   仅调大线程参数不算完成优化。worker/native/compiler线程总量和GPU准入受真实配额约束，batch服从用户明确上限。
4. 冻结实际执行源码，保存CPU/GPU预算、requested/effective并发、复用、OOM及编译回退。
   已完成产物直接核验复用；不为更换调度器从头重复训练，不把新快照身份倒填旧模型。
5. 启动后核对真实进程树、各阶段进度、GPU/CPU采样和已完成单元速率；空闲时定位剩余依赖或瓶颈并处理，
   不能只报告“配置了并发”。以墙钟改善为目标，不承诺所有依赖阶段的CPU/GPU百分比始终100%。

每次修改须保留身份/时间/fit范围、随机配方、预测和失败分母的可核验性；性能探针与科学结果分别记账。
默认完整落实已授权实现和云训练，不反复请求已有授权。准入、测试和必要的短测应支持快速执行。
用户再次明确：**说训练就是执行已约定的全量矩阵**，除非用户明确限制为小试、只读或dry-run。
必要的软件测试和小规模验证由执行方自行完成，通过后自动进入全量，不再次请求启动许可；
不能停在试跑、执行清单或因小试成绩不理想而自行终止后续训练。真实失败/合同冲突仍须处理，
但已经通过的相同环境和快照检查应直接复用，不重复冗长预检。

`operator_fixed_target_batch_rule`允许记录有明确理由的执行方固定batch配置；
其provenance为`run_configuration`，没有选优证据，不冒充用户逐字指定或完成B0搜索。
P001先导采用上限512，在校准数据上比较1/2个spawn进程后选择双进程、每进程1线程；
最新时耗及云端源码/输入验证见[P001运行](evidence/brainsync_p001_pilot_20260907.zh.md)。

复用`train.runtime.GpuPerformanceScheduler`与`MatrixBatchSource`，设备顺序CUDA、XPU、CPU，
张量经所选`torch.device`迁移，不硬编码`.cuda()`。CPU是正确性参照，
加速器比较允许声明精度误差，不能以CPU合成测试宣称CUDA或实机验证。

监督`DeepConfig`默认请求fused Adam和`reduce-overhead`整train-step编译；CUDA按支持情况
执行，CPU/XPU回退非fused eager，记录requested/effective/fallback。
`--no-fused-adam --compile-mode none`提供显式eager对照。SSL和SubjectAdapter有独立优化循环，
不能将监督训练测得的默认性能直接套用。BF16按设备能力启用，否则FP32；
batch和更新预算显式记录，不通过隐藏梯度累积或OOM重试改变科学比较。

预载矩阵须留显存余量；连续pinned视图可异步拷贝，随机gather走现有同步安全路径。
不能复用仍被DMA读取的单一pinned staging buffer；未来预取须有多缓冲和完成事件。
`--materialize-shuffled-epoch`会额外占一份矩阵，历史配对测量更慢，保持显式可选。
训练loss在设备累积、每epoch读取，预测按整pass回CPU，避免逐batch同步。

GPU folds用spawned进程隔离RNG和CUDA上下文，不能共享训练线程。
自动GPU worker数依设备显存及总预算决定；显式并发覆盖须有当前设备吞吐/显存证据。
`--cpu-threads`是总预算，经`resolve_cpu_worker_plan`与`cpu_thread_budget`按worker分配，
同时约束Torch/BLAS/OpenMP并读取cgroup配额；`threadpoolctl`为显式依赖，缺失时导入即失败，不静默跳过BLAS限制。
源数组共享只读，fold-local拟合状态各自私有。
同一外层划分的公共QC可跨候选复用；校准内部选择不能复用只满足outer-fit范围的QC。

优化先测预处理、训练、目标适配和写结果各段耗时；记录冷/热编译、峰值显存及延迟中位/p95。
大账本checkpoint应在任务组进入时完成整体校验、固定摘要，组内只做逐人身份排除与状态独立化。
不能以GPU空闲直接提高worker数；先检查反复反序列化/合同递归处理。2026-09-07实测该问题
令4个E2 worker各占一CPU核而GPU长期0%，一次合同校验约1.57秒。
优化后的4单元真实5090校验为27.31到4.24秒，确定性GPU条件下预测完全相同；
默认CUDA自身重复存在浮点波动，不将此性能检查声称为独立科学样本。
历史5090耗时、并发策略和归约细节保留在[设备测量记录](evidence/device-portability.md)，
其数值不是当前机器的实时能力。方法效用比较与等draws/等updates的机制归因分别记账。

本次[目标适配计时](evidence/cross_dataset_start_20260906.zh.md)中CUDA单任务约快6.4%，
不支持“小模型必选CPU”的规则，也不替代并发吞吐测试。CPU/CUDA随机训练预测可能明显
不同，运行后端须冻结入配方；性能探针不冒充新增独立科学样本。

## 数据、工作树与文档

科研集成优先逐读涉及模块和实际builder、caller、保存加载与重放路径，再选能区分新旧行为的定向合同和反例；测试扩展围绕尚未解决的集成风险。科研子代理不限我方时间、token或算力额度，主代理负责仔细审阅和最终集成。工作树优先放D盘、只取代码并共用项目环境；整合后保留提交和必要恢复记录，及时移除已完成的任务树。研究结论集中维护既有四主题页，用具体结果、适用假设和下一出口表达。

BrainSync采集REF更正为左耳耳垂耳夹时，活动元数据通过`correct_brainsync_reference.py`
按显式session列表、旧标签和用户证言修订；该工具预检全部记录、压缩校验原声明后再写入。
原EDF和追加式事件日志不改写，旧二进制头中的声明由BIDS/session更正记录明确覆盖。
历史cache/QC/模型/结果保持执行身份；新派生物使用新目录，禁止直接改旧digest假装旧模型已经重训。
见[更正清单](../frozen/brainsync_reference_20260908/metadata/manifest.json)及
[科研/验证说明](evidence/brainsync_reference_20260908.zh.md)。

训练链条入口的`--dry-run`生成任务、真实身份/行范围、recipe、候选失败与预算清单，
不得拟合normalizer、BN、LDA或网络。已完成compact运行使用`--stage compact`，plan声明
`target_stage=compact`、`target_batch_size=512`及用户固定batch规则，先核对123人、305正K、
1953个评分。实际batch为`min(512,n_fit)`，60遍历；无需B0搜索或冻结结果，不据此声称最优性。
旧控制器及后续B0/B0_score/E1/S1/E2/S2/E3队列已终止，不得自动恢复；新compact队列单独冻结和记录。
减少重复拟合只可复用完整数据/状态/配置digest一致的结果，不改变本轮recipe和独立初始化语义。

保留的B0接口先只评分inner validation，
`--declare-batch-cohort-from ... --batch-cohort-design ...`将全部预检与预声明participant清单绑定，
实际B0通过`--batch-cohort ...`使用该范围；
`run_cross_dataset_transfer.py --freeze-batch-from ... --batch-cohort ... --output batch_rule.json`冻结规则；
`--stage B0_score --batch-rule ...`才执行其final test评分。这些是未排程的研究能力，
其缺省batch32不能覆盖本轮用户固定512策略。两类binary入口从项目外目录使用绝对路径调用。
新产品选择器同样使用`--batch-rule`；缺省明确回退32，旧产品独立固定64。
本轮compact512不更改产品默认batch，也不运行E3选择器benchmark。
用户随后授权的扩展全量通过`include_historical_controls=false`仅运行batch512的E1/E3处理；
source plan默认自动并发，各worker使用一个原生计算线程和一个编译线程，按真实CPU配额限制任务组。
GPU采用带余量的全局任务准入预算；任务结束释放临时内存，进程退出释放上下文。
共享文件registry保留跨进程任务复用，JSON发表和checkpoint哈希核验在语义签名锁内完成。
内存预算取主机可用量与容器`memory.max-current`的较小值，并保留15%余量；不能使用宿主机
总内存直接排程。source组暂按6 GiB、E2/外层worker按4 GiB估算，可在plan显式调整；这是
基于已有进程RSS的准入估计，不是内存硬上限。嵌套worker同时预留全局CPU和内存，余量不足时
由已加载数据的父进程评分；source组结束后退出进程，使后续评分可扩大并发。
单任务组也经过相同准入。中断恢复读取并校验各子组记录及外部引用文件，保留原快照身份。
临时attested副本和结果目录分别检查磁盘容量；`N2P3_VERIFIED_CACHE_SNAPSHOT_DIR`可指定
空间充足的私有目录。2026-09-07旧数据盘仅余1.4 GiB，触发校验副本的空间拒绝后，改用另一个
有25 GiB余量的文件系统续跑，旧数据和结果保留原位。

E3共享候选缓存仅驻留当前人/实际K；切换scope清空，避免GPU状态和数据随人数累积。
复用账本区分请求次数、真正fit和cache hit，不能将命中时间记成独立训练速度。
产品dry-run加载确定性输入并报告来源兼容性与inner范围，自动EOG仅列出拟合计划。
分析器默认拒绝software/synthetic或未声明证据；显式`--allow-software-correctness`仍将所有
区间标为软件验证，不进入公共开发或总体推断。

`freeze_source_snapshot.py --output 新目录 --materialize`保存当前文件字节、父commit、dirty状态
和物理归档，生成`checkout/`隔离副本。实验从该副本调用生产CLI；source和evaluation可绑定
不同快照。执行核验也逐项比对归档成员，不能通过重写manifest摘要给旧归档换身份。

2026-09-07本次本地环境为Python3.14.3、Torch2.13.0+cpu；管理工具可见5070 Laptop，
但项目Torch无CUDA运行时。本轮软件/CPU计时不作为目标5090的吞吐、显存或batch选择证据；
当前设备结果及正式执行前的资源缺口见[实施证据](evidence/training_chain_implementation_20260907.zh.md)。

2026-09-08经用户授权新增本地GPU专用`.venv-gpu`，原CPU`.venv`保留。GPU环境先按项目锁定
依赖安装，再显式覆盖为`torch==2.13.0+cu130`；`uv pip check`及真实CUDA运算通过。
本机实测RTX5070 Laptop 8GB、驱动610.62；使用单GPU worker、固定batch32的独立P001开发对照。
该环境不是公共云5090配方。启动前核对实际包版本、显存与源码快照，完整记录依赖覆盖与执行路径。
结果与脚本见[恢复实验](evidence/brainsync_recovery_20260908.zh.md)；不修改默认CPU环境或静默回退CPU训练。

2026-09-08代码质量修复期间，另一安装进程将主`.venv`的Torch改为`2.13.0+cu130`，
最终1530通过/4跳过基于该实际环境，GPU故障注入也已执行。小型CUDA/BF16/fused Adam及
显式`aot_eager`检查通过；本机没有Triton，未验证Inductor/CUDA Graph或云端5090吞吐。
Windows编译验证使用`python -X utf8`，避免当前Torch模板读取采用GBK；这不修改项目默认环境。

校准扫描默认统一使用`transfer.binary.STANDARD_CALIBRATION_TRIALS=(0,200,800,2000)`。
`run_binary_subject_transfer.py`的CLI默认和`run_cross_dataset_transfer.py`缺省plan字段
共用该定义。有限数据按`LIMITED_CALIBRATION_FRACTION=0.8`回退，
结果`budget_adjustments`记录请求与实际预算；同一有效K去重，K0只跑zero-shot。
历史冻结运行脚本保留原显式预算以便复现，不作为新实验配置模板。
正K迁移必须同时包含target-only scratch；两类binary runner共享实际行摘要核对，
并将`paired_transfer_effects`写入结果。默认保留原波形LDA同K对照，
不能仅因迁移臂得分高就宣称迁移有价值。

原始数据、派生cache、运行结果、当前源码和冻结快照按[目录层级](technical_structure.zh.md#目录与数据层级)
分别保存。删旧文件前物理压缩、逐文件校验；结果内嵌路径和分数保持原字节。
冻结材料不得重新作为默认配置/可导入源码；有效唯一数据不当临时垃圾删除。
保留无关dirty修改；stage或commit须属于用户明确要求的提交整理范围。

历史`tmp/`的数据和结果现存cache/results history，完整查找方法见
[数据保留记录](evidence/workspace_cleanup_20260906.zh.md)。Git忽略的大材料迁机时单独携带，
不能把Git clone当数据备份。代码归档与研究成绩各自保留原身份。

2026-09-08再次整理本地临时材料，4,816份历史文件及工具缓存压缩保存在
`frozen/local_history/workspace_cleanup_20260908/`；BCI根目录16份旧传输文件集中到
`D:/BCI/n2p3-audit-archive/transfers_20260831_20260901/`。当天脚本、SSH/EDF支持目录和
三个Git工作树保留原位。旧路径查询、校验与恢复见
[本次整理记录](evidence/workspace_cleanup_20260908.zh.md)。

2026-09-08云端数据盘已清理，旧BI64/GTN/R8派生缓存、早期开发结果和旧部署展开目录
经压缩逐文件校验后退役，归档位于`/root/autodl-fs/n2p3-cleanup-20260908/`。
当前四库输入、完整训练链、源模型和早停/90轮实验仍在原位；历史路径查询、清理清单与
本地证据包见[云端清理记录](evidence/cloud_disk_cleanup_20260908.zh.md)。

GitHub不分发历史压缩归档；`frozen/`只跟踪清单、哈希与交付记录，详见
[归档说明](../frozen/README.md)。移除已进入未推送历史的归档时，先在本地备份原Git历史，
再仅改写未推送提交；保留旧新commit映射，不回写历史manifest的`source_commit`。
推送前检查远端同步、全部待推送对象大小、暂存内容、文档链接、Ruff、相关测试及lock。

当前指导固定在[六份正文](README.md)，实施状态统一更新执行计划。
独立实验/来源/推导放`evidence/`，原JSON、日志、图片放原结果目录或`assets/`。
修改文档核对本地链接；替换的指导先归档再删除，合并关系写入manifest。
