# 公共源吞吐：软件切片与实际验收边界

2026-10-08。真实旧运行 fd62c48 已有四个完整outer就绪，但单个load gate保护下仅一个worker做十一库合并和压缩，05:21采样约0.97 CPU核、GPU0；其余三个worker在IPC等待。旧first merge曾观测RSS25.79 GiB，高于18 GiB静态worker估计。原运行、已完成view与所有准备产物保留。

首切片在 `codex/public-source-throughput` 实施三个直接接点：共享QC不再把全部EEG转换为float64及同规模std临时数组，按完整epoch分块保留旧每行归约；新outer工作cache明确使用uncompressed NPZ；`EpochCacheProduction`把原writer实际保存的canonical数据、容器SHA、recordSHA和decoded摘要交给同进程，不冒充一次attested-load。真实旧resume仍走原cache验证，不给历史模型重签新源码。

production入口明确限制writer实际保存的表示，包括float32 EEG/positions、int64标签与event整数、float64event时间、bool掩码、float32 QC、Unicode身份和显式metadata。保存后同进程核record、文件版本与实际decoded内容，拒绝写后数据或容器变动。首次写入后保留同一对象供三个seed分别独立selection/refit，原source/目标科学合同与数学程序没有更改。

`SourceInputPhases`用独立运行日志记录source load、实测参考/QC/身份投影、完整拼接、工作cache写入/核验、writer交接和source输入账本；CPU明确是整个进程、可能包括其它线程，RSS前后值不称峰值。该日志不进入科学数据或模型身份。当前仍保留单merge gate，不依据短探针放开四个高峰合并。

必要验证使用独占持久目录。QC包括float32/float64、非连续/Fortran/reversed、多块及最后不足一块、缺导、恒零和最后块NaN；最终缓存特征逐元素相同。writer源身份、dtype变更、写后篡改、旧resume，以及真实BDF/MAT两库入口合计22项通过，0失败/错误/跳过；随后新增两个空物理轴反例，连同原7项QC做9项增量核对通过，独立case共24项。工作cache根与持久结果根分离再经原两库入口default/separate-root两个case通过，新增1项，最终独立case共25项。一次新basetemp父目录未创建造成setup错误，原日志保留；随后创建新的任务目录执行正式22项，未回到全局临时池。

70,385,664-byte EEG短probe中，旧QC额外采样峰285,704,192 bytes，新为36,192,256 bytes；耗时约0.229→0.204秒。该样本证明有界临时矩阵与同输出，不证明云端完整merge峰或端到端倍数。16,384-row真实merge调用链软件fixture的压缩+reload与stored+resident都产生相同decoded摘要、X/y；精确计时和实际执行源摘要见本轮receipt，短fixture改善不能推广为完整25核利用。

另从原BNCI A01 MAT和BI2014a subject1 ZIP成员真实准备两库，原信号SHA核对后得到5388×6×179。免压缩writer交接与冷attested loader逐元素核X/y/subject/mask一致；这是公共输入准备证据，NN拟合0次，无效能结论。

来源、JUnit、探针及代码身份由[切片凭证](../assets/public_source_throughput_20261008/slice1_receipt.json)绑定。完整排除BNCI十一库云端bench、实际peak、门禁/并行调度、旧view/model受控接续与新运行利用率仍单列完成；本页没有宣称2500% CPU或四路GPU已经实测达到。

后续原fd62全四view完成但尚未NN拟合时，caller计划非法scope `public_source_development` 在child较晚阶段才拒绝。新切片将scope/software预算合法性集中到 `resolve_source_program`，在任何资源/归档准入及重merge前调用；三个非法enum的无输出/无cache与两合法enum等6项反例通过。该失败与旧view来源保留，0 NN不记完成；[前置凭证](../assets/public_source_throughput_20261008/scope_preflight_receipt.json)独立签本次源码。

## 完整输入基准与后续源运行终态

真实排除BNCI的十一库输入基准已结束：合并529.064秒、writer常驻交接72.428秒，采样RSS峰27,555,393,536 bytes；X/y/身份/掩码/位置/QC/时间与预处理数值摘要全部与原view相同。Chailloux安全加载161.764秒、投影及身份核验36.38秒、完整拼接31.73秒、写盘/核验135.13秒是实际阶段读数。该基准只做输入生产和交接、零NN；旧运行包含四view及失败开销，不能用它的总时间与本单view计算速度倍数。

fd62原运行于05:59:33 UTC因caller非法scope以exit1结束，四完整view保留。改合法`development`后的2394执行直接复用四view，06:37:00–06:53:03 UTC运行962.390秒后exit1，父层报告`BrokenProcessPool`及worker突然终止。187个资源样本中GPU最高95%，GPU≥80%期间CPU中位约4.03核；cgroup内存采样峰96,625,889,280 bytes，接近90 GiB硬上限。这些值说明实际训练曾使用GPU，但不是25核充分利用或内存准入成功的证明。当前`oom=0/oom_kill=0`不能确定原始退出原因；没有已完成source checkpoint，不把在途更新填成零更新或完成。

07:21:29 UTC只读核验无活动source owner/runner，GPU0。下一切片继续减少whole-X与inner/held副本、落实分阶段字节准入和ready NN优先，另补实际worker退出码/phase/native诊断后再恢复完整12程序。科研子代理不设我方期限或额度；局部软件通过不能代替下一次真实内存峰、有效并发和进度验收。

两个失败run及完整基准共26份原计划、owner、退出/资源/runner记录已下载逐SHA核对，创建并逐成员回读[物理证据归档清单](../../frozen/public_source_terminal_20261008/evidence.manifest.json)。归档载荷与云端原文件均保留；[终态读数](../assets/public_source_throughput_20261008/terminal_readout.json)绑定实际archive而不改写历史producer。

## worker退出诊断软件切片

退出观测已由独立SRC owner集中实施并合入a23fe28：每pool实例隔离、PID/创建身份、每任务状态重置、source/load/fit阶段、Python/native stderr与faulthandler；父层保存首次观察及实际process handle退出码。坏JSON、资源传感器异常、诊断写盘失败和executor registry并发清空经真实RED→GREEN处理，原fit/pool异常仍保留。Windows CPython动态spawn遗漏新sentinel的纯stdlib原始frame证据保留；本片交付观测，不宣称修复该调度竞争或2394原始死亡原因。

最终独立21项（11诊断/10资源owner）、Ruff、diff检查及Spec/Standards通过；两份旧真实sourceCLI共16task对应各自此前冻结版本，没有重标为final v3执行。[质量凭证](../assets/public_source_throughput_20261008/diagnostics_quality_receipt.json)、[主树逐文件整合](../assets/public_source_throughput_20261008/diagnostics_integration.json)和[444成员原冻结清单](../../frozen/source_worker_diagnostics_20261008/source.manifest.json)分别绑定实际来源。整合后的共同调度与全矩阵、云端完整吞吐仍按本轮工程接口继续验收。

## 第二软件切片：共享存储、行视图与输入调度

一次最终C-order float32 buffer由`PreparedSourceStorage`拥有；每个原source按原顺序独占行区间，读入/投影后立即交到最终buffer。
不同时保留全部原X再复制最终X；事件、身份、标签、mask、metadata、lineage仍由原canonical concat owner产生。
`source_loading`仍进入canonical provenance；改变loader数量会形成新的decoded/cache身份，数值一致不能追认为旧cache身份。
线程loader使用分配的group native预算并按RAM临时与snapshot磁盘字节准入，父对象在合并后释放。
ENV或显式snapshot目录先经同一central选择，再按实际tmpfs/ramfs额外计RAM。容量是估计，不能当实测峰值。
通用CLI保留原普通save/load路径；production交接只接受canonical C-contiguous final buffer。

`SourceRows`保存原行顺序、重复行和实际通道索引，禁止隐式物化全部raw subset；选轮train/held与全量refit都消费同一只读base。
原逐行float64 moments、风险权重、source sampling、seed、更新和归一化不变；final normalized矩阵只分配一次。
中央finite/static mask/trial mask检查按2048行分块，包括晚块masked NaN，保留拒绝边界。

共享NN暂存按train+held float32确数准入，ready NN优先于下一个组I/O。
`GpuPerformanceScheduler`在两个normalized矩阵仍存活时测RSS；CPU fallback若满足始终持有的source whole-process组预算则归还额外staging，
否则在首个optimizer更新前明确失败并释放lease。四个6GiB抽象stage与四份18GiB组reservation同时就绪的反例通过，未做巨型分配或CUDA拟合。
没有shared pool的source child仍消费组RSS预算；历史0.25GiB软件fixture升为1GiB，原小值不构成Torch进程容量证据。

固定更新复用已有`baselines.progress` sink：既有100-update/验证报告点写小JSONL，selection与refit目录分开，display fold=0不改变cfg.seed。
软件实例核进度真实更新数与inner/refit账本，不能把completed exposure=0解释成in-flight更新=0。

本软件切片44个独立case的所需路径通过。42个主要case由本机必要验证核对；两个always-RSS spawn CLI终态在实际云Linux3.12锁定环境复验，2通过、0失败/错误/跳过、27.120秒，实际执行源为RC e323ea38…的444成员逐bytes冻结。本机资源准入RED及早期无RSS版本GREEN各留原来源。多轮JUnit按classname+case去重，不把重测相加。
包括5036有序/重复行、超过32768行moments两风险、真实两更新CPU dense/lazy的完整history与state逐元素一致、训练和held normalizer exact、
原SharedTrajectoryRegistry实际fit复用、两真实study调用、writer/cold-load decoded摘要一致；spawn CLI两布局以实际RC云CPU记录验收，
模拟disk/tmpfs与ENV snapshot选择、0/1字节预算分配前失败、晚块非法值及lease失败释放。
完整测试由根代理在合并所有独立切片后统一一次；[第二切片凭证](../assets/public_source_throughput_20261008/slice2_receipt.json)绑定源码与原JUnit。

本切片没有重新云端launch。旧full11 benchmark的reader/writer成本和高峰、2394/fd62失败及所有原产物由根保留。
目前组I/O门禁为1，组内可以多个source loader；跨组snapshot磁盘聚合和当前binary/candidate消费者的共享ready滚动仍由统一executor收口。
终态full11真实bench、整个12源程序、CPU/GPU实际利用率与峰值分别验收，不能把本软件切片称为云吞吐或研究效能已经完成。

主树63e38ba已合入第二切片与退出诊断，三处source CLI冲突保留两方意图：诊断范围内创建显式snapshot根、准备调用使用共享`reserve("load")`并在同诊断task内解析结果、父层提交同时保留输入gate和process事实捕获。相应旧parser反例的stage seam同步新接口。十六份受审源码保持原bytes；合成CLI另签实际449成员归档07104d54…，在云Linux锁定环境完成2个真实sourceCLI与1个准备坏JSON诊断case：3通过、0失败/错误/跳过、27.510秒，实际pytest退出0。首版后台验证owner因换行转义在测试前失败，原日志保留；v2负责实际退出码和cgroup采样。新sourceRows/test行尾已集中固定为受审LF，[逐文件整合凭证](../assets/public_source_throughput_20261008/slice2_integration.json)和[原受审444成员清单](../../frozen/public_source_throughput_slice2_20261008/source.manifest.json)分别记录，联合软件通过未签为新四GPU吞吐或通用执行器完成。

## 共用执行模块：云端生命周期验证

尚在独立writer实施的CloudExecutor使用同一资源owner，按cache组/participant组持有实际进程，OS退出后才归还阶段额度。451成员RC1归档7c7c8a89…的9项云端行为case通过；RC2归档56168f3b…加入管道创建失败和原生actor退出后的grandchild回收，11项通过、0失败/错误/跳过、8.529秒，实际pytest退出0。RC2包含实际CUDA上下文/16元素探针，没有EEG或模型拟合；两版重叠case不相加。

[生命周期凭证](../assets/public_source_throughput_20261008/cloud_execution_lifecycle.json)逐文件绑定实际JUnit、退出和资源采样，物理源码清单分别为[RC1](../../frozen/cloud_execution_rc1_20261008/source.manifest.json)与[RC2](../../frozen/cloud_execution_rc2_20261008/source.manifest.json)。原RC2后续SourceInputAdmission、epochs与科学caller修改没有进入该冻结；全局FS/RAM池、持久产物占用、当前source/binary/candidate实际调用及完整programme另验收。本段不将生命周期通过推广为通用底层或实际吞吐全部完成。
