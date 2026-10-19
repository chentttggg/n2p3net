# 训练链条实现与验证记录

本记录保留训练链条实现、软件验证和旧运行的真实身份，并记录用户后续精简决策。
当前执行方案以[roadmap的compact512矩阵](domain_transfer_major_20260920.zh.md)为准。
代码、合同/合成验证、公共开发成绩和成人/设备确认分别验收；扩展seed、Mainsah和真人采集不在本轮范围。

## 优化扩展全量已启动

该节以下保留启动时点的过程证据。自动续跑现已完成：95个源任务、37568个源链外层评分及E1/E3
全部成功；45134外层和7050内层预测复算通过。最后一次续跑68分05秒，0失败；统计结论及完整
归档见[完成结果](training_chain_results_20260907.zh.md)，不再以启动采样代表当前状态。

### 自动资源调度与偏好固化

用户要求每次云端训练先调用相关性能/资源技能，吞吐优先于保持代码简单，允许增加必要的
调度、缓存及并发实现。已写入AGENTS和工程指南；默认按实际资源并行准备、source、校准与评价，
保留batch512、配方、身份/时间/fit范围、完成结果及各自原始快照。

当前运行在`/root/n2p3-runs/training_chain_auto_c_20260907/`，2026-09-07 20:40启动，
控制器PID24311。执行[快照manifest](../../frozen/training_chain_auto_20260907/source.manifest.json)
SHA为`5aa716346e2e15bd39665620cd9e6b33d361a60b6b95e9524be31ba236ec544b`，235个成员。
实际资源为25核cgroup配额、90 GiB容器内存和RTX5090；宿主机显示208核及约600 GiB空闲内存，
这些宿主机数值未用作可用配额。默认解析9个独立source组、8个GPU任务槽，目标最多8个worker。
source/E2嵌套任务共享CPU和内存准入，按组进程退出、编译器reset及CUDA清理释放临时状态。
20:41:39采样确认9个source工作进程同时运行，CPU合计9.01核，GPU15%、显存3194 MiB，
容器内存52.07 GiB；当时处于各组加载/校验与首批新拟合交叠阶段，已导入2个旧任务、0失败。
这是启动利用率观测，不是全程平均或最终完成率；实时结果保存在`runtime_status.json`与各组记录。

旧metadata运行在45个源任务、4组选模完成后因磁盘不足退出，未进入外层评价。
数据盘仅余1.4 GiB，校验一个约2.3 GiB的cache副本需要更多空间，因此新输出与临时副本使用
另一个有25 GiB余量的文件系统。旧源文件及1294个跨代外部引用均核验SHA；新续跑清单
SHA为`1a9adf1366917a200a42b3ce8350c5f1f294ec266322bed557e6bed5ba7d71ba`。
E1 6528和E3 1038个已完成评分继续保留，不重新训练。准备清单、资源JSON与运行日志分别保存。

最终快照在5090通过生产CLI双布局/双spawn进程短测：8个源任务成功，0失败，墙钟31.76秒，
实际worker PID24230/24233；进程退出后无残留训练GPU进程。该短测使用1 update合成输入，
只证明CUDA软件执行及生命周期，不据此推算正式4000/8000更新耗时或科学准确率。
本地全套1186通过、4跳过、0失败（215.25秒）；审查随后补上单组资源准入及中断子组续跑两项
反例，11项资源/resume测试通过。Ruff全源码检查通过。父报告未合并时直接核验子组完成记录，
避免中断后重训；跨进程语义锁覆盖训练到checkpoint发表，避免竞争者重复拟合同一轨迹。

### 元数据瓶颈与续跑

用户反馈GPU未充分使用。实时采样确认4个E2 worker各占满一CPU核，而GPU连续0%、
后续一个采样11%；设备32 GiB显存只用约4.8 GiB，cgroup CPU配额为25核，不是GPU容量不足。
E1的6528个和E3的1038个结果均已完成、0失败。

真实源checkpoint为28.64 MiB，仅1214个权重/缓冲元素；主要体积为逐行训练账本。
每个E2单元重复加载并多次验证该账本，单次合同验证实测1.568秒；cProfile显示
一次验证约2436万Python调用，成本集中在`_canonical_value`、dataclass复制和JSON摘要。
同进程组只校验一次、逐人保留身份门禁后，4个真实单元从26.725到4.047秒（6.60倍）。
默认CUDA重复原实现也有最大约0.0079的logit变化；确定性GPU对照为27.311到4.241秒（6.44倍），
四份预测逐项完全一致。CPU拟合状态和预测的精确等价、身份泄漏与篡改反例均通过。
只读attach采样受容器ptrace权限限制，以上函数画像来自独立cProfile子进程；训练环境依赖未更改。

旧控制器及其已核对的进程树已终止，E1/E3、首外折的22个源任务和已有预测均保留。
`training_chain_full512_20260907/source_resume_inventory.json`冻结1322个原源产物的SHA，
清单自身SHA为`87f0db53f51f1b32ce13e6fd20f0b7288686360c762a5fb8943b7f6fc9c23ce2`；
终止及迁移记录为同目录`metadata_migration.json`。新队列从独立快照续跑，旧权重不重标也不重训，
完成的源内评分保留原评价合同；不完整单元保留原文件，在新目录重新执行。
另观察到source编译在8次重编译后回退警告，该项未因本次metadata修复自动宣称解决。
原1.5–2.5小时估算未充分计入大账本重复校验，剩余时间需按新队列实际吞吐更新。

修复版已从`/root/autodl-tmp/n2p3-runs/training_chain_full512_metadata_20260907/`续跑，
控制器PID21544、source进程21547。新[执行快照](../../frozen/training_chain_metadata_fix_20260907/source.manifest.json)
SHA为`8d8c078e60e12031d15e41c0617c67249518c3e3e593928d0b3a3954cf68120a`，233个成员。
启动后203秒核对时已导入9个源任务，未重新拟合；后续继续核验余下原任务并执行未完成部分。
本次全套为1177通过、4跳过、0失败（191.73秒）；Ruff和diff检查通过。
普通/确定性GPU对照记录已回传本地结果目录的`benchmark_default.json`和`benchmark_deterministic.json`。

用户在成本评估后明确要求启动。新目录为
`/root/autodl-tmp/n2p3-runs/training_chain_full512_20260907/`，控制器PID17593；
依次执行E1、E3、S1/E2/S2及其并发外层评估，source/target seed均20260906，batch上限512。
跳过B0/B0_score、历史batch32、旧产品batch64、扩展seed、Mainsah和成人采集。
新快照见[manifest](../../frozen/training_chain_full512_20260907/source.manifest.json)，
归档SHA-256为`08a13ae56aef5c1f3469b6a762cbb97cd72b73204e97b51eb7cb3e773b28f9fb`，227个成员。
训练从物化checkout执行，controller脚本摘要也写入`execution.json`。

E1四库dry-run核对6528个评分；正式启动42秒检查点完成254个、0失败，其中36个从compact复用。
四个CUDA worker运行，GPU利用率96%、显存3218 MiB。E3预期1038个外层评分；其后源链
保留单父进程的精确轨迹registry，E2候选及外层目标使用4个spawn worker，分别一个CPU线程。
E1同数据/环境/配方/行及状态的compact模型与K0直接导入，新消费合同记录原路径和摘要。
外层目标组共享同一attested cache，scratch/LDA及相同源状态结果按完整键复用，复制状态后独立记账。
预计1.5–2.5小时、预留3小时；启动检查点不等于剩余阶段已经完成，实时状态读取新目录。

本次全套1175项中1170通过、4跳过，1项旧Windows raw snapshot临时文件并发删除发生
`PermissionError [WinError 5]`；该失败项单独重跑通过，未改动raw模块。
新的spawn分片/合同、固定512分支过滤、跨compact拟合复用、缓存错配/模型篡改测试均通过。
Ruff全源码和`git diff --check`通过，JUnit为`tmp/full512_launch_tests.xml`。

## compact512精简排程

用户要求固定batch512、保留少量基线并以组件消融替换巨大研究网格。旧队列已终止，
新compact队列1953/1953全部完成、0失败；不能将旧B0完成率当作新矩阵进度。旧B0/B0_score、完整E1、
S1/E2/S2及E3全部取消本轮排程，保留实现能力及既有产物。

新矩阵复用b版四个整库排他source，123人、305个去重后实际正K；K0只运行source zero-shot。
每个正K包含原波形LDA、Scratch NN、Source full NN、去目标输入统计、冻结BN buffer、
冻结学习型trunk六臂。最后一臂仅classifier可训练，同时冻结BN affine，仍使用online buffer
和相同checkpoint dropout；冻结BN buffer臂仍允许affine更新。
全部NN固定60遍历、AdamW lr=1e-3/wd=1e-4、seed20260906、实际batch=min(512,n_fit)，
同人同K匹配行序列、updates/draws；无inner搜索和额外refit。

计划1953评分=123 K0+305*6；其中1525次NN拟合、305次LDA拟合、0次新增source拟合。
batch512是用户指定策略，不是已完成B0选优。三个消融只回答完整配方下的条件贡献，
不验证完整交互、多源独立价值、source选模方式或产品选择器收益。
原S1=4外折*2布局*4配置=32次source拟合；S2=4外折*(4全量+4剂量)+1家族=33次；
E2另有16条source轨迹、80个候选checkpoint、选后refit和大量内层目标校准，均不再排程。

替换前九份活动文档已压缩到本地
`frozen/compact512_redesign_20260907/active_documents_before_compact512.zip`，
SHA-256为`0c590d5ec82ecf8a95f453c26ec0ea5eafbf2113e86b8494d5f289da3317d426`；
全部成员从压缩流重新计算SHA-256并与原文件一致，见
[归档清单](../../frozen/compact512_redesign_20260907/manifest.json)。旧数据、模型、结果及源码身份未改写。

### 新队列启动核验

运行目录为`/root/autodl-tmp/n2p3-runs/training_chain_compact512_20260907/`，控制器PID11793。
四库生产CLI dry-run核对1953评分、123人、305 LDA及123 K0；正式运行采用4个CUDA worker，
每worker一个CPU线程。首核验点运行33.8秒，完成224/1953、0失败，GPU利用率99%，显存3088 MiB。
这是启动检查点；最终完成状态见该目录的`execution.json`及各分片`record.json`。
执行源码为[独立物理快照](../../frozen/training_chain_compact512_20260907/source.manifest.json)，
archive SHA-256为`c3857d180b776ef55d6c4284464bde957b04c0e848c338ec1864729bdea2c82a`。
后续本地重复拟合优化不改写该运行快照，也不计入本轮吞吐证据。

旧控制器及4个暂停worker已核对命令后终止。旧GPU显存从3122降至2 MiB，退出进程RSS合计约12.7 GiB；
暂停采用SIGSTOP时保留内存，终止后才释放。B0磁盘产物保留3476/3660成功单元、0失败，
取消记录为旧目录`cancelled_for_compact_redesign.json`，新目录`old_queue_retirement.json`记录逐分片SHA。
这些模型拟合inner-train，可作为B0观测和计时证据，不能冒充完整prefix的compact模型。

### 最终结果复核

1953/1953成功、0失败，训练进程墙钟326.828秒（5分27秒），配对分析CLI退出码0。
退出后GPU显存回到2 MiB。独立重算1953份预测的AUC/AP/BACC/Brier/NLL、评价合同摘要，
核对1830份拟合模型状态摘要、123人共同测试后缀、305个正K的六臂配对，以及全部NN的
60遍历、512上限、实际updates/draws与相同排列摘要，全部通过。
结果包已回传`experiments/runs/training_chain_compact512_20260907/results.tar.gz`，
云端/本地SHA-256同为`9070eaa2f4c49a26981ec7d023dbadea3c10e9176eb1a5ce347e6e039df8fa2a`。
同目录`analysis.json`、`integrity.json`、`execution.json`可直接读取。

下表为同人同实际K配对后按库等权的AUC差；K200/800/2000分别123/59/21人。

| 比较 | K200 | K800 | K2000 |
|---|---:|---:|---:|
| 完整迁移 - scratch | +0.0431 | +0.0258 | +0.0050 |
| 完整迁移 - LDA | +0.0078 | +0.0164 | +0.0165 |
| 源输入统计 - 完整迁移 | -0.0049 | -0.0098 | -0.0016 |
| 冻结BN buffer - 完整迁移 | -0.0113 | -0.0193 | -0.0126 |
| 只训练head - 完整迁移 | -0.0414 | -0.0631 | -0.0683 |

K200迁移-scratch的条件95% bootstrap区间为[0.0305, 0.0562]；完整区间、逐库负迁移、
精确K与回退K分别见分析文件。各K可用人群不同，不能把表中跨K差异当作纯标签剂量效应。
这是一组固定源模型和四个固定开发库下的单seed结果，不覆盖新库、重训source或成人设备确认。

K2000的迁移-scratch区间为[-0.0017, 0.0132]，仅BNCI/Huebner的21人可估计；迁移优势不明确。
LDA基线应保留：K200完整迁移-LDA区间跨0，BI2015a的K200/800均值都偏向LDA，
K800有14/18人LDA更好。K200有30/123人迁移低于scratch，其中27人下降超过0.01。
head-only和冻结BN三个K的条件区间均低于0；输入统计切换影响较小且依赖库/K。
这些观察支持保留三基线和当前完整适配，不要求为重复同一结论恢复全部交互网格。

### 重复拟合简化

E3四选择器通过单scope精确缓存复用共同候选。真实1epoch软件探针从16次拟合降至10次，
选中配方、逐预测和状态摘要一致；35项相关测试通过。训练行/源状态/统计/seed/配方改变不命中，
inner与full-prefix隔离；普通候选的验证成绩重新计算，旧产品选epoch的验证数据进入键。
最终测试数据不影响fit缓存；命中返回独立对象，失败仍记录，scope结束释放缓存。
这是等价性及调用次数验证，不是5090性能证据；该优化不在已启动compact的冻结快照中。

E2新增运行内精确source轨迹复用，11项source测试通过：原权重及原合同不改写，消费者容器
绑定自己的父cache和fit ledger，同时记录物理来源SHA/合同摘要及0次本地optimizer更新。
有序训练行、身份、标签、mask、seed或配方不同不会合并，源checkpoint篡改会拒绝复用。
设计上的12条library内轨迹对应6组两库训练集，但只有完整实际签名一致才复用；本轮未重跑E2，
不将潜在16到10条轨迹的减少写成已测云端提速。各外折评分仍独立，不跨目标库共享校准结果。

### 优化后完整研究矩阵的耗时评估

当时按用户要求只估算；随后实际启动见本文开头。这里的“全量”指原S1/E2/S2加E1/E3的研究矩阵，
单source seed、单target seed、batch上限512。跳过B0搜索/B0_score、E1历史batch32和E3旧产品batch64，
不扩展Mainsah/真人；已完成compact无需重跑。既有低batch结果和inner模型不能更名复用。

真实preflight及生产确定性分组给出E2内层915个library正K和90个participant正K，
乘5候选为5025次NN拟合，另2025次K0前向，CE与zero指标共享预测。
源主链81个逻辑任务/388000 updates；同一combined调用的运行内registry可把物理任务降至
54–77个、256000–372000 updates，额外refit为0–16次/0–46000 updates。
实际只完成首外折签名核对（20到19）；54是全部跨外折键一致时的条件下限。
若拆成独立source CLI，当前运行内registry不会自动跨进程共享。

| 执行方式 | 预计墙钟 |
|---|---:|
| 已实现source/E3精确复用，E2校准和源外层评估仍串行 | 3–5小时 |
| 再实现目标4-worker GPU调度、跨源scratch/LDA共享、相同checkpoint多selector合并评分 | 1.5–2.5小时，资源预留3小时 |

估算时第二档尚未实现端到端调度；后续已实现并启动，完整运行提速仍待完成后实测。
估算分项为source约20–35分钟、E2校准20–30分钟、
外层评估40–65分钟、剩余E1/E3与分析10–20分钟。实际来源为本次326.83秒四worker记录：
cell耗时累计1003.50秒，墙钟/累计比0.3257；历史同5090、512的source refit约0.00360–0.00405秒/update。
旧source是epoch分支，新fixed-update采样/编译不同；三槽、feature LDA、预计算及IO另留余量，
时间范围是计划估计而非置信区间。所需66–82个去重后model×target-study外评仍保留真实各折结果。
不能以GPU并发数直接除以四，不能把相同seed视作GPU逐位确定性证据。

机器可读输入、公式及假设在本地结果目录`full512_cost_estimate.json`，
重算入口`tmp/estimate_full512_from_artifacts.py`。本次估算未新增模型拟合；只读计数进程已退出。

## 旧队列启动记录

旧队列曾启动`/root/autodl-tmp/n2p3-runs/training_chain_20260907/launch.py`，控制器PID9240。
设备RTX5090，4个训练进程、每进程1个CPU线程；四库123人、305个合法实际正K，
B0共3660个训练/验证单元。启动后首个核对点已完成463单元、0失败，GPU利用率73%。
这是当时启动核对点，不是当前状态或完整成绩验收。当前旧队列已终止，实际取消记录单独保留。

当时的原计划顺序为B0、完整队列冻结、B0后评分、E1、S1/E2/S2及其外层评估、E3；现已取消该衔接。
source/target seed均20260906。四个原b版source复用原权重；执行源码为
`tmp/training_chain_snapshot_20260907c/`对应的物理归档，已部署到云端`source/checkout/`。
正式source计划及家族输入已准备：8份原cache字节和study门禁通过，家族source为
BNCI8人+Huebner13人、196357行；已有数据未重标。准备这些输入不表示source矩阵已训练完成。

## 实施基线

任务开始HEAD为`f54e45c48014d05a71d9644b0cecd7aa041f873f`；当时已有预算/runner/文档的dirty与
untracked改动。414个非忽略文件按原字节保存在
`tmp/training_chain_baseline_20260907/files/`，逐文件SHA与起始状态见同目录`manifest.json`。
审查以这份实际工作树为基线，不以HEAD差异代替本轮修改，不撤销并发改动。

## 接口清单

| 分类 | 接口及处理 |
|---|---|
| 已存在复用 | 标准K/80%回退/同K配对、生产身份/lineage/评价合同、checkpoint、SubjectAdapter、DeepBaseline、runtime、sklearn LDA |
| 扩展 | SubjectAdapter初始/refit状态、固定updates与batch账本；DeepBaseline采样/统计/候选快照；两类公共runner和产品持久化 |
| 新增明确能力 | TargetRecipe、合法校准内划分、B0选择/E1矩阵/E3候选；S1/E2/S2编排；真实study membership、固定模型缺导探针、分层分析、执行源码核验 |

依赖顺序为共享预算/recipe与E0核验，合法inner划分与B0，E1适配，S1源诊断，E2/S2源选模，
E3产品选择，最后CLI、反例、独立审查与文档。实现分工按现有模块所有权进行。

## E0只读复核

在独立输出`tmp/training_chain_e0_recheck_20260907/`重算c轮1648份预测，123人、16分片。
AUC/BACC与原记录差异小于1e-12，另算AP；同K行、共同test/标签及prefix嵌套检查通过。
四个b版source checkpoint从原结果包重算SHA，训练账本、训练合同、输入统计和source snapshot
一致；1648个评价合同摘要逐项核对。原始结果、分数、模型和旧source/evaluation身份未改写。
实际batch序列和optimizer updates在旧history中不存在，保留`not_recorded`。

标准K200/800/2000分别可估计123/59/21人；305个去重后的正K单元，另123个K0。
历史当前成绩仍以[标准预算结果](four_study_standard_budgets_20260907.zh.md)为准；本记录不新增成绩。

BI来源stream按人编码。`data.study`从原始候选合同及lineage祖先证明stream到真实库的归属，
不按namespace或前缀猜库；历史checkpoint缺少内嵌归属时，使用外部已校验cache的生产来源证明，
仅绑定新的评价，不改写旧模型。当前BI proof cache/sidecar均已重算hash，61个stream归属一致。

## 软件验证

精简矩阵及精确复用修改后，`python -m pytest tests`为**1167 passed / 4 skipped**，164.46秒；
JUnit为`tmp/compact512_final_tests.xml`，Ruff全源码检查通过。
此前完整训练链实现的全套为**1156 passed / 4 skipped**，187.34秒。
两项生产端集成在设置`BRAIN_SYNC_WORKSTATION_ROOT`后补跑，**2 passed**；
剩余两项未执行的是当前Windows权限不允许的符号链接反例，不是算法/GPU通过证据。
Ruff（`src experiments tests`）及`git diff --check`通过。测试期间核对的294个代码/依赖文件
未发生字节变化；JUnit及源文件hash见`tmp/training_chain_validation_20260907/`。
实验使用项目Python3.14.3/Torch2.13.0+cpu；CUDA运行时不可用。
本次生成了四个明确标记为synthetic的生产cache及一个整族留出组合，用于CLI与合同验证；
每库8名合成participant、每人400行，不是公共数据、真实成人或采集记录。

| 阶段 | 实现状态 | 已有验证 | 正式证据 |
|---|---|---|---|
| E0 | 核验复用 | 1648预测、四权重/合同、61个BI来源stream归属复核 | 已有单seed公共开发；历史预算观测缺项保留 |
| B0 | 完成 | batch去重/预算/完整cohort/诊断曲线反例；24个inner与25个后评分CLI单元 | 旧队列已终止；3476/3660部分产物保留，本轮不续跑 |
| E1 | 完成 | 八臂、统计诊断、全部桥接；43个CLI单元；BN/refit反例 | 正式因素矩阵未执行 |
| S1 | 完成 | 加权矩与风险手算、两布局dry-run、五mask固定模型探针 | 32条source轨迹未正式训练 |
| E2 | 完成 | 4条缩小轨迹、2次从头refit、180/180内层评分、6个合法外层plan | 完整8000更新轨迹未正式训练 |
| E3 | 完成 | 四程序9个CLI单元；保存重载、实际统计、无标签/EOG/同recipe对照 | 新公共选择器及成人确认未执行 |
| S2 | 完成 | 单源/多源/固定剂量/整族与seed合同，全部source dry-run | 新全量/剂量/家族矩阵未正式训练 |

source实际合成运行来自物理快照a，目标运行来自快照b，各自保留原manifest；
共101份target及180份source内层预测再次独立复算AUC/相关指标和评价合同摘要，全部一致。
B0/E1/E3/B0后评分的CPU进程墙钟分别约14.14/20.08/11.88/13.79秒，包含解释器启动，
只使用1遍历的软件预算，不能线性换算正式60遍历或5090费用。新增B0诊断曲线另通过
109项相关回归中的状态/预算不变反例；源码与结果版本不混记。

Standards与Spec独立审查共发现并修复：真实库排他、逐session类别、run顺序、失败分母、
普遍收益措辞、外层plan参数/结果字段、B0跨入口规则、诊断曲线、完整队列及相对路径。
审查不要求commit/stash，也未安装issue tracker。当前六份指南和证据索引已同步。

## 当前入口与保留接口

以下命令使用项目`.venv/Scripts/python.exe`；具体plan填写现有cache/checkpoint的实际路径与摘要。
当前公共入口为`--stage compact`，plan声明`target_stage=compact`、`target_batch_size=512`，
`batch_rule`使用`user_fixed_target_batch_rule()`生成的用户固定规则，或通过`batch_rule_file`读取同一规则。

```powershell
.venv/Scripts/python.exe experiments/run_cross_dataset_transfer.py --plan compact_target_plan.json --subjects P01 --stage compact --dry-run --device cpu --output tmp/compact_tasks
.venv/Scripts/python.exe experiments/run_binary_subject_transfer.py --training-plan compact_target_plan.json --target-subjects P01 --stage compact --dry-run --output tmp/compact_binary_tasks
```

真实云端运行使用完整声明participant清单、CUDA和新冻结的evaluation snapshot。
以下为旧矩阵已验证的保留接口示例，不是本轮执行命令：

```powershell
.venv/Scripts/python.exe experiments/freeze_source_snapshot.py --output tmp/chain_snapshot --materialize
.venv/Scripts/python.exe experiments/run_cross_dataset_transfer.py --plan target_plan.json --subjects P01 --stage B0 --dry-run --device cpu --output tmp/b0_tasks
.venv/Scripts/python.exe experiments/run_cross_dataset_transfer.py --declare-batch-cohort-from b0_BI14/tasks.json b0_BI15/tasks.json b0_BNCI/tasks.json b0_Huebner/tasks.json --batch-cohort-design experiments/runs/four_dataset_20260907/development_plan.json --output batch_cohort.json
.venv/Scripts/python.exe experiments/run_cross_dataset_transfer.py --plan target_plan.json --subjects P01 --stage B0 --batch-cohort batch_cohort.json --device cpu --output b0_run
.venv/Scripts/python.exe experiments/run_cross_dataset_transfer.py --freeze-batch-from b0_run/record.json --batch-cohort batch_cohort.json --output batch_rule.json
.venv/Scripts/python.exe experiments/run_cross_dataset_transfer.py --plan target_plan.json --subjects P01 --stage E1 --batch-rule batch_rule.json --device cpu --output tmp/e1_run
.venv/Scripts/python.exe experiments/run_binary_subject_transfer.py --training-plan target_plan.json --target-subjects P01 --stage E3 --dry-run --output tmp/e3_tasks
.venv/Scripts/python.exe experiments/run_pretrain_supervised.py --source-chain-plan source_plan.json --dry-run --output tmp/source_tasks --device cpu
```

真实执行从materialize的`checkout/experiments/`调用，源snapshot和评价snapshot各自绑定。
`study_membership_proofs`允许列出已有NPZ来源证明；源库排他与完整训练身份仍须通过当前门禁。
S1/E2/S2正式配置固定4000/8000更新网格；缩小预算仅允许明确`evidence_scope=software_correctness`，
相关输出不能进入公共开发成绩汇总。E2选中的最终refit次数由完整内层评分确定。
示例中B0的每库预检须覆盖声明的完整名单；实际分片后冻结时列出全部record，遗漏分片将被拒绝。

源plan的每个outer fold包含`source_cache`、`excluded_source_datasets`、逐source协议及目标cache/
协议，家族折用`targets`分别列出两个目标库；source/target seeds独立。
新产品调用可用`brainsync_model_task.py --task calibrate --selector-mode five_candidates
--selection-protocol brainsync_operational_r5 --batch-rule ... --session-dir ... --model-artifact ...
--output-dir ... --dry-run`，实际校准去掉dry-run。source兼容失败不会冒称成功迁移。
`analyze_training_chain.py`接收明确study、source/target seed和record集合，输出精确K、回退K、
预算策略、分库/共同队列、p10/最差/负迁移与条件bootstrap；软件验证必须显式使用
`--allow-software-correctness`。Brier/NLL若报告，明确为未经外部校准的raw sigmoid读数。

## 旧矩阵成本与证据边界

以下数量属于精简前设计，当前取消排程。旧E1在batch32时由已有实际K清单推导6528个评分，在另一个冻结batch时最多7138个；
这只复用已核验预算清单，最终执行仍须逐cache/inner时间和身份预检。B0最多3660个inner任务，
实际batch去重及非法inner会减少数量；E3四程序加每人K0共1343个外层评分。
不能把任务数当独立样本数，不能用CPU计时代替目标GPU预算。

完整合成输入上的source dry-run为81条轨迹：S1 32、E2 16、S2全量/剂量32、家族1，
计划388000 updates/198656000 draws，1200次内层target fit；E2完整source refit尚由选模决定，
每outer/source seed的refit上限15500 updates。该计数不替代真实数据的观测/身份预检。
正式秒数、峰值显存与云费用未给伪精确估计，需目标设备上代表性正确性/吞吐探针及实际费率。

旧矩阵曾要求先冻结B0规则；当前compact直接声明用户固定512，独立绑定新执行源码及实际5090环境。
S1/E2/S2、扩展seed和外部/成人确认没有因本次软件实现自动获得已完成状态。
最初两个过期SSH入口连接失败；后续通过当前有效westc:41560连通并按用户新授权启动全量。
E0所需模型/结果均已从原归档完成核验，已知来源的cache、raw和结果原件全部保留。
