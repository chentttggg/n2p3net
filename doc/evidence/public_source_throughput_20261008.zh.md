# 公共源吞吐：第一软件切片

2026-10-08。真实旧运行 fd62c48 已有四个完整outer就绪，但单个load gate保护下仅一个worker做十一库合并和压缩，05:21采样约0.97 CPU核、GPU0；其余三个worker在IPC等待。旧first merge曾观测RSS25.79 GiB，高于18 GiB静态worker估计。原运行、已完成view与所有准备产物保留。

首切片在 `codex/public-source-throughput` 实施三个直接接点：共享QC不再把全部EEG转换为float64及同规模std临时数组，按完整epoch分块保留旧每行归约；新outer工作cache明确使用uncompressed NPZ；`EpochCacheProduction`把原writer实际保存的canonical数据、容器SHA、recordSHA和decoded摘要交给同进程，不冒充一次attested-load。真实旧resume仍走原cache验证，不给历史模型重签新源码。

production入口明确限制writer实际保存的表示，包括float32 EEG/positions、int64标签与event整数、float64event时间、bool掩码、float32 QC、Unicode身份和显式metadata。保存后同进程核record、文件版本与实际decoded内容，拒绝写后数据或容器变动。首次写入后保留同一对象供三个seed分别独立selection/refit，原source/目标科学合同与数学程序没有更改。

`SourceInputPhases`用独立运行日志记录source load、实测参考/QC/身份投影、完整拼接、工作cache写入/核验、writer交接和source输入账本；CPU明确是整个进程、可能包括其它线程，RSS前后值不称峰值。该日志不进入科学数据或模型身份。当前仍保留单merge gate，不依据短探针放开四个高峰合并。

必要验证使用独占持久目录。QC包括float32/float64、非连续/Fortran/reversed、多块及最后不足一块、缺导、恒零和最后块NaN；最终缓存特征逐元素相同。writer源身份、dtype变更、写后篡改、旧resume，以及真实BDF/MAT两库入口合计22项通过，0失败/错误/跳过；随后新增两个空物理轴反例，连同原7项QC做9项增量核对通过，独立case共24项。工作cache根与持久结果根分离再经原两库入口default/separate-root两个case通过，新增1项，最终独立case共25项。一次新basetemp父目录未创建造成setup错误，原日志保留；随后创建新的任务目录执行正式22项，未回到全局临时池。

70,385,664-byte EEG短probe中，旧QC额外采样峰285,704,192 bytes，新为36,192,256 bytes；耗时约0.229→0.204秒。该样本证明有界临时矩阵与同输出，不证明云端完整merge峰或端到端倍数。16,384-row真实merge调用链软件fixture的压缩+reload与stored+resident都产生相同decoded摘要、X/y；精确计时和实际执行源摘要见本轮receipt，短fixture改善不能推广为完整25核利用。

另从原BNCI A01 MAT和BI2014a subject1 ZIP成员真实准备两库，原信号SHA核对后得到5388×6×179。免压缩writer交接与冷attested loader逐元素核X/y/subject/mask一致；这是公共输入准备证据，NN拟合0次，无效能结论。

来源、JUnit、探针及代码身份由[切片凭证](../assets/public_source_throughput_20261008/slice1_receipt.json)绑定。完整排除BNCI十一库云端bench、实际peak、门禁/并行调度、旧view/model受控接续与新运行利用率仍单列完成；本页没有宣称2500% CPU或四路GPU已经实测达到。

后续原fd62全四view完成但尚未NN拟合时，caller计划非法scope `public_source_development` 在child较晚阶段才拒绝。新切片将scope/software预算合法性集中到 `resolve_source_program`，在任何资源/归档准入及重merge前调用；三个非法enum的无输出/无cache与两合法enum等6项反例通过。该失败与旧view来源保留，0 NN不记完成；[前置凭证](../assets/public_source_throughput_20261008/scope_preflight_receipt.json)独立签本次源码。
