# 执行、保存重放与代码集成

本文按主题维护已复核的结论、代码接点和下一步。运行日期与源码身份留在各证据及绑定中；使用方式和工程规则由原六指南维护。

## 当前职责与交付

科研主树拥有科学合同和最终集成，云窗口拥有唯一云端资源与实际执行。P1数值作者/审阅者在机制树工作，prefix selector作者/审阅者在已接cache的隔离树工作；cloud resource-phase和continuous score优化在云端独立树交付，root按函数逻辑合入。检查实际模块、调用、保存和重放，针对新风险运行必要反例；源码与产物身份随交付记录。

近期已合入source-only准备缓存和current-view共享评分（`4830ef9`），source权限始终来自私有admitted ledger/set，每次current target proof重新核对，target state/BN/statistics保持fresh。live与loaded各自实际计算，wholeprefix捕获/持久化/重放完成后封prediction；parent复算声明混合的完整概率。

BI2014a按库队列由云端推进，数据准备、模型、结果和终态各自持久化；一库完成后解锁下一库。科研日常seed为20260918，大版本使用三个完整独立程序。资源owner传递25 CPU总预算和至少4 GPU容量，真实ready work与阶段资源决定并发；本机合计CPU低于80%。

云端资源交付`c916de4`已逐函数审阅并合入科研树`e0832c5`：显式native+interop预算、FIFO延迟GPU准入、完整进程树退出后释放及nested artifact并发峰值登记。科研随后把candidate的compute phase集中到`ensemble_fit._fit_expert`真实神经拟合接点，覆盖新增prefix selector内部拟合，并避免K0/不可用分支提前取得GPU。原云49项验证保原源码；本地资源与prefix相关验证通过，新增两例分别穿过真实神经group-LOO和完整prefix selector，核实际12/8/8/8及196行。Linux后代退出与tmpfs两例由原云证据验证，本机按平台跳过；没有重复本地全套测试。

近期旧日期页已经先物理归档、逐member验证，再合并为四个主题页；原binding、MAT、缓存、模型和结果保持原位置。源页面恢复方式见[主题整理历史清单](../../frozen/research_topic_documents_history.manifest.json)。后续研究结论更新主题页，代码使用方式更新原六指南。

## 恢复身份与文件所有权

原`wishlist-integration`工作树已被应用归档。最后已接受科研提交为`1ca4619a79d5a8e4b536ae1960ea7029604801ad`；Git恢复快照`99352070d267141db8651c26c5749a564f6762c6`以它为直接父提交，保存12文件的待审增量。恢复快照不是科研验收。本次从该快照建立`codex/research-integration-20261011`，实际路径为`C:/Users/chent/.codex/worktrees/research-integration/n2p3-net`。原ignored暂停检查点、旧独立review与诊断目录在原路径未找到，不将其称作已恢复产物。

用户原`D:/BCI/n2p3-net`并发dirty保持；云端`deep-research-plan`与accepted496归另一窗口独占。本科研源码没有替换其producer。云accepted496 archive为`4b075a2df69b781f03844ad7efe2e3d4a2b64342118ea291bb85e1aa99dc2127`、manifest为`04ff8a07a7a801fca4d46904572ea9a2acd6696570c72454de280d773900f12e`，其原四source模型及结果由云窗口按原身份解释。

三路作者分别拥有nested/ensemble、dictionary和spatial文件；第四路独立审阅QR与全局依赖。作者结束后交换复审，科研总监逐读差异、完整调用链和反例，再集成。本轮不跑仓库全集。统一使用项目venv并显式`PYTHONPATH`指恢复树；venv的editable安装本身指向用户原树，单凭workdir不能证明实际执行来源。

## 验收状态与下一依赖

下列记录对应当时的源码与执行阶段；当前状态见本页开头。

本页的作者和独立审阅产物在`tmp/RESEARCH_DIRECTOR_20261011/`；源身份、必要验证、原MAT重放与最终接受状态由同目录和本页后续结构化binding保存。目前正在完成交换复审与真实native组合验收，未把本页当最终签收。

全局优先序：保B12基础公共执行独立前进；补nested inner持久重放及prefix/runner consumer；真实source效应权限与跨布局transport在接learn前明确；B07后续wavelet/off-grid/history按实际数学依赖进入。TemporalNoise当前`/5`从保存编译态安装，旧startup再生成RED不能当当前未修事实；完整同版本跨host模型重放仍待实测。B05/B11时间相关失配与ERP/span是竞争解释，不让一个数值门阻塞其它愿望，也不从拟合更好选择产品默认。

## 云端交接边界

下列记录对应当时的源码与执行阶段；当前状态见本页开头。

2026-10-11 云窗口报告新纯性能 consumer 已实际启动：physical freeze archive `f655dea8b19d1d1a9b2664cf3a175cfed9186349d27cb26940f55acedd437794`、manifest `a8827a42e8a3fe6ea25f8f1fb0871d308dee89350864075c66cf9ac49aaeeb74`、隔离候选 commit `3747bf1bc1a9138f9c771aca7461403db8079222`。该 execution 使用新 owner/API，与旧 accepted496/source producer 分离；科研树不修改其运行文件，也不从未终态云结果选择科研 route。云端实际吞吐、完成矩阵和效能由云窗口继续验收。

## P0续接

下列记录对应当时的源码与执行阶段；当前状态见本页开头。

全局审阅把下一顺序定为 B12+B14 单 seed 公共 execution 最高优先，B03 inner producer replay 并行；B05/B06/B11 固定输入机制面板为诊断支线，B09 standalone learner 与 B13 成人确认不阻塞 P0。依据见 `tmp/RESEARCH_DIRECTOR_20261011/GLOBAL_ROUTE/REPORT.zh.md`。

B03 新增 opt-in InnerReplayRecorder，在同一次 `fit_nested_calibration_selection` run 捕获实际 cache producer、grouped-temperature OOF fixed producer/held score、v/h/t context、失败 receipt与 source role。persist_inner_replay 没有原 recorder 会 fail-closed；有 recorder 才保存实际 ensemble package 和完整 manifest，replay_inner_replay 只加载、审计 forbidden scope并重放 score，不二次拟合。8 项 inner 测试、原 selection/ensemble 41 项通过；B12 CalibrationPrefix/public caller仍未接。

B12 新增 `validate_b12_execution_manifest`，由 paired `run_erp_comparator._finalize_comparison` 写入 `b12_execution_binding`，校验 requested K×arm×seed roster、实际K alias、失败/不可用分母、target authority/input manifest和 public source producer catalog。软件 scope不虚构producer。另修复 FrozenTrial list/tuple JSON metadata 比较，保留真实 preprocessing 差异拒绝，相关校验5项通过。云端 candidate8 的6个 input-contract RED已缩小到该类型问题并修复科研树版本；2个FS/RAM失败、四库 Errno28 和完整source→target终态仍保留，未放宽 gate或改写旧producer。

这批 P0 代码已提交到隔离科研分支，之后独立复审继续核对其实际行为；B12 只有 software/synthetic contract evidence，不能宣称完整单seed执行、公共效能或成人确认。

云端后续交接：数据盘清理从约0.949GB free恢复到25.176GB，原科研文件无计划外变化；B08 tuple helper 修正以独立新 producer完成 exit0，1344包/576 energy/48 C reference/28 native/20000 bootstrap等按新 acceptance 身份解释，旧失败不改签。该结果不替代B12 source→target execution，科研侧继续保留 candidate input-contract、资源 admission和四库终态边界。

## B12最终账本修后复签

独立审阅否证了此前manifest初版：coverage只核总数、重复source键消失、可重签的producer声明与plan不一致、unresolved peer掩盖completed peer缺源，以及statistics之后的report不在摘要内、重复finalize滚动绑定旧摘要。原RED、原probe和历史producer保留；本轮修复后才重新接受对应关系，不能沿用初版的通过数量作修后证据。

修后coverage逐bucket核实际referenced cell终态；source catalog拒绝重复和foreign study/seed，checkpoint/procedure与plan一致，public producer的seed/kind/selection/source-study mirror及target exclusion互核。failed actor的缺失source保持显式`unresolved_source_producers`，completed NN/scratch cell缺源直接拒绝。participant聚合遇到同键不同完整凭证拒绝，相同凭证可合并。最终statistics写完后绑定report，摘要排除顶层`b12_execution_binding`，重复finalize不消费旧binding。

原独立15项探针字节未改，修后15 PASS、exit0；真实source7 selection/refit（inner3/refit2）及unresolved-peer正控仍通过。实际paired waveform serial finalizer保留8 requested aliases、6 resolved cells；首次最终report、去binding独立canonical和repeat-finalize摘要均为`13d3bb70…`。正式B12合同与actual caller22项通过。另从`e9ef642`导出不可变副本，仅覆盖本轮B12三文件，对实际spawned participant部分失败、全失败及source dependency unreachable三个聚合路径验证通过；未混入并发prefix WIP。各命令退出码0，相关Ruff与diff检查通过。source/file/日志摘要和实际coverage见[紧凑绑定](../assets/research_resume_20261011/b12_postfix_binding.json)。

总监逐读最终manifest、serial finalizer与participant聚合调用链后接受这组软件关系。聚合冲突分支由源码审阅确认；跨阶段门继续消费原物理入口已核的source/checkpoint凭证。CalibrationPrefix/public caller的输入、OOF、common-test及保存重放在后续完成复签，详见[校准与选择](calibration_and_selection.zh.md#calibrationprefix保存重放与调用端复签)。完整公共矩阵与成人确认按各自计划继续推进。

## 缓存与共享评分独立复签

source cache 与 shared programme score 已在科研主树 `4830ef9` 集成。修后 checkpoint 的 whole-study literal 权限只读私有 admitted source ledger；当前 target 的 physical、identity、execution 与 study proof 每次重核，source prototype 不缓存 target fit、BN、OOF 或 temperature。`FittedEnsemble.score_programs` 只在同一 current view、同一实际 experts tuple、fit receipts 与 target contract 下共享 raw score，fixed/grouped 仍分别应用自己的概率程序；capture 与 loaded future replay 各自真实计算，parent 保原混合准入。独立 authority/fresh-state/lifecycle probes 和两条真实 K0/full-C/OOF caller 路径通过，原权限绕过和失败日志保留。固定源码及证据边界见[独立复签 binding](../assets/research_resume_20261011/cache_shared_acceptance_binding.json)；这仍是软件/有限调用链验收，没有完整矩阵吞吐、跨host整包或公共效能声明。

## 条件评分公共白化集成

云source07的固定白化patch（`bf09e072…`）已按函数逻辑接入科研树：只抽取原conditional QR helper，条件score在当前record共享公共列，各candidate继续全record白化增量设计并分别积分。root逐读实际白化、event design、prefix/full QR和precision调用；AST核对全部其它现有函数一致，保留科研树较新的单producer QR及稳定df。相关23项验证、Ruff与diff检查通过，来源见[集成绑定](../assets/research_resume_20261011/common_whitening_integration_binding.json)。

云端原32 records×36 candidates的数值桥与352-record阶段计时保其原来源；科研侧当前接受公式/调用相容和相关行为，完整矩阵耗时继续由云窗口实测。旧模型保持原producer，新训练包使用实际执行源码身份。
