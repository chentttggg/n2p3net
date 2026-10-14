# 科研总监交接

科研已按用户授权持续推进。当前入口为 `D:/BCI/worktrees/research-director`，分支 `codex/research-director-resume-20261013`，最新已提交状态可由该分支解析；当前已接受源码含35adb7e。原暂停检查点和旧producer保持历史身份。活动规范由六指南维护，科学结论由四主题页维护。

## 最新集成与当前工作

- binary固定requested选择已接受：缺失预测保固定正负对分母，普通/QC/EOG及公共E3共fit/refit owner，个人包 `/5`、target-stage `/2` 保存实际选择关系。原新旧产品全C参数、独立预测及保存加载逐bit一致。详见[整合绑定](research_integration_acceptance.json)及主树 `tmp/BINARY_SELECTION_*`。
- B09物理效应/参考输运和精确因子已接受：posterior `/5`、observation `/3`保存原A/D/Z、同次QR/noise生成及原坐标/map/volume。四导实际12维null反例和極端单位QR关系已修；完整measurement covariance、原BLUE与持久态复用。代码集成acaf796、指南7e66453，实际生产SHA `23eb1376…`，原交付回收在 `tmp/spatial_exact_delivery`，独审 `tmp/SPATIAL_EXACT_FACTOR_FINAL_REVIEW.md`。已结束作者树移除，提交保留。
- 四fixed程序factory已接受：source/scratch单NN及各自NN+xDAWN，只替换新计划的prefix selection；匹配recipe/architecture与经典专家自身profile、baseline/tie/row政策由原owner消费。真实CLI、graph、保存、零拟合重放与parent已有软件出口；公共新程序需自己的freeze。
- 云node/chunk/shared-input底座已按模块集成，原history权限、中央资源floor、同checkpoint bytes、完整dynamic registry与同CloudRun finalize保持。Linux原history正控实际OS0，执行前后599source members一致。完整四程序source链case原75ee4第一次为setup失败；云实际第二次source selector/refit完成，但fixture遗漏waveform LDA，target前被原控制门拒绝。1656afa只修测试输入及10面板/完整requested断言；新case完成source/target拟合和25包，prepare因锁定Python缺memfd/fcntl绑定未准入，云owner修nativeABI。target实测RSS1.531GiB超过原fixture1.25，41cecd6明确软件worker2GiB及原20ms whole-treeguard，CPU2/native1不变。旧失败保原签，history不重复。交接 `tmp/CLOUD_FIXED_FAMILY_INTEGRATION_HANDOFF.md`，云正式日志由owner交付。
- B07多通道C-latent已完成的原作者树 `D:/BCI/worktrees/multichannel-c-latent`，作者/数学独审持有新producer：实际C标签权限、同record一次startup/共享beta/原flat measure、完整positive-state后验，explicit conditional-tail/whole-observed两评分法，same-context nominal对照。独立raw AR协方差与实际selection design oracle均闭合。严格BNCI合成fixture119label/8channel首轮6scope通过、3numerical不可用；改small-log/scalar-operation误差和保存F后，再修整数索引/upperT/C1入口。最终旧C producer模型与MAT全SHA复用，newconsumer `tmp/C_LATENT_NATIVE_REUSE` OS0/三完整出口；八prediction/1044continuation、nominal唯一新Cfit、whole/censored/modelonly/heldtruth均触发，旧C0refit/publicfit0。原失败与各body保来源，作者96280已由根审/独审最终接受并集成319da17；全部tmp和runtime源43成员核回主树 `tmp/c_latent_delivery`，clean作者树已移除，branch提交保留。
- B09作者/独审树 `D:/BCI/worktrees/spatial-rowspace` 的rowspace owner4b4c已独审接受并集成b15bdaf：joint `[D Z]` exact time row-space压缩保shared beta和每record独立nuisance核，数值QR/noise/BLUE不变；随后接真实source builder、OOF和ensemble。B03概率作者/独审树 `D:/BCI/worktrees/probability-tail` 的fixed-base exactpool首片e189+bf06已接受并集成bfdc613/1f19e8d，normal selectedtail/publicparent正控与篡改拒绝实际通过；后续T+ε及ordinary各exactpool独立状态继续。pool层 `pε=(1−ε)p+ε/J`，每层只用实际完整pool合法OOF及声明risk学习ε，完整roster/失败及旧producer保持，沿parent/model-only replay保存新程序。两个任务只复制代码，必要数值窗本机native1串行协调。B11作者树 `D:/BCI/worktrees/joint-flat-scale` 实施actual same-record单Gammaλ、globalbeta/recordflatZ共同测度、C→prefix→tail及真实source/C参数权限/持久态；独立初审已固定有效行数/volume/prefixa与ratio-MCSE关系，小oracle和实际纵向待执行。
- 原BI/BNCI、entry005/P1大结果及32份P1 score/receipt已恢复到 `D:/BCI/artifacts/cloud-completed-originals-20261014`。云独立原分数分析获接收；root核37个local payload/分析/脚本/清单SHA与bytes，原binding保持，见[恢复与复算绑定](../research_resume_20261011/cloud_payload_recovery_binding.json)。云73精确临时文件释放4.069GiB；模型、selection graph及有引用或缺重建链的冷缓存保留。科研原dirty main保持。
- memory精简仅通过ad-hoc更新说明记录在 `C:/Users/chent/.codex/memories/extensions/ad_hoc/notes/20261013-215044-research-memory-simplification.md`。长期偏好保全局推导、实际模块/调用链审阅、定向验证、无我方科研额度、D盘代码树和主题文档；原memory不直接编辑。

## 工作身份与恢复

- 当前工作树：`D:/BCI/worktrees/research-director`；从原C盘树迁移并修复Git注册，没有新增科研载荷副本。
- 当前分支：`codex/research-director-resume-20261013`，恢复基点`f1d0ec0b912af0e0dc8e8e5c269c5d9e8bb7b98c`。
- 暂停前完整代码/结果文档提交：`577ac4c68db17cf3ce12431add8f371deaab5759`。本handoff及当前checkpoint随后作为文档提交；最终提交可由该分支tip解析，避免在文件中自引用提交SHA。
- `f6c8ca5`：目标校准scope和严格warm-start完整状态覆盖；`2dc48dd`：Gaussian诊断及中央资源floor；`afa9bbb`：当前科学路线与验收；`577ac4c`：从云端`4c4b987`接收完成读数，保较新的科研记录并解决四个文档冲突。
- 暂停收尾时实际`merge-base --is-ancestor 33603f7431ca50465e2d1678eef51145a1c63f3c HEAD`退出0：Claude review原tip已是当前树祖先，当前无需再次合并旧branch。原提案的替代/否决以及随后修复按固定源码覆盖和活动指南解释。
- `D:/BCI/n2p3-net`仍是`main@f1a303c42c68173e8b1452ceb26bd84923b8a6aa`，暂停收尾时345条porcelain记录，完全保留。恢复时原状态保持；后续仅移除已接受且恢复证据核验过的任务checkout，main、数据、分支提交保持。未push、未prune或gc。
- 多个旧工作树目录已不存在，其Git提交仍保留。prunable注册不等于可安全清理：旧ignored原始结果、模型和专项验证载荷须先核恢复来源。旧`pause_checkpoint_20261012.json`不重写。
- Python：`D:/BCI/n2p3-net/.venv/Scripts/python.exe`；行尾以`.gitattributes`和实际raw-byte冻结为准。

## 已接受代码与证据

| 切片 | 当前行为与验收范围 | 证据 |
|---|---|---|
| 训练scope | `load_pds_selection_record`始终要求目标校准scope；missing/null/非mapping feature record不能旁路。纯数学解析保留独立入口 | [训练准入](training_admission_repairs.json)，51项相关检查 |
| 严格warm-start | 复制前核完整目标parameter和buffer；拒空或缺tensor状态，完整状态逐bit加载 | 同上，4项focused检查及旧源码反例 |
| InformationBound `/2` | raw same-sample shrinkage plugin d²；实际/显式canonical activation、全候选geometry、uniform随机tie；冻结整份报告、独占JSON保存和model-only replay | [数学整合](diagnostic_mathematics_integration.json)、[独审](diagnostic_independent_review.json)，最终29项；早期13项保早期源码 |
| Exact covariance `/2` | 实际`F K_AR F.T`；raw SPD Cholesky、显式orthonormal支持及Euclidean支持测度；未解决factorization保raw诊断；ridge为独立显式law | 同上；原37项、迁移28项、最终6项按真实源码分别绑定 |
| 中央floor | helper和direct resolver共用authority；非豁免CUDA省略/null用中央CPU25/GPU4，显式override不能降轴；明确software/local scope及CPU准备/依赖尾角色保留 | [floor修复](cloud_floor_authority_repair.json)，31项focused检查 |
| Claude覆盖 | 固定`f1d0ec0`：53等价、8正确替代、8退役/历史、1否决、3真实余缺；后三项本轮修复。594 evidence、9 manifests及cleanup manifest的Git路径/blob保留 | [固定源码覆盖](runtime_coverage_review.json)；旧“26待接”不再作为当前待办 |

最终实际源码SHA256已在文档合并后核回：

```text
src/transfer/information_bound.py 18218cb9be6814176279b681734aa25e84e625ca9327fc11c3ea815798c84e58
src/transfer/exact_covariance.py d3f27c60e6a42d25c84c5bc86e7d3aac5882b8aa6b865a82390ae5a410eea721
src/research/source_execution.py af464538c1525dd71b95c952e90d64a3ee930b399d574b4e2205c6bb138a2028
```

独立最终审阅逐读实际数学、模块、caller、不可变状态和五份JUnit，两轴GO，无阻断项；未重复测试。Exact类当前为内存诊断，没有save/load API。原main中local-only `experiments/run_information_bound.py`保留，缺`membership_kind`并需要activation/schema及旧ceiling措辞迁移，尚未准入。当前未增加公共生产caller。

专项原始日志与详细交接仍在本树`tmp/INFORMATION_BOUND_REPAIR/`、`tmp/EXACT_COVARIANCE_REPAIR/`、`tmp/CLOUD_FLOOR_AUTHORITY_REPAIR/`、`tmp/DIAGNOSTIC_FINAL_REVIEW/`；Git中的紧凑binding包含各自摘要和范围，归档本树前须保留这些ignored载荷。本轮未跑本地全集、未启动公共训练或公共fit/score/model-load；执行过合成协方差估计、Gaussian MC、报告load/replay及密度/构造反例，不能称零数值操作。软件案例有重叠，不能合并为科研重复数。

## 完成程序与可读分析

原完成程序已按SHA恢复并独立复算，今天没有重拟合这些完成实验。读[公共评估](../../evidence/public_evaluation.zh.md)、[校准与选择](../../evidence/calibration_and_selection.zh.md)、[响应与噪声](../../evidence/response_and_noise.zh.md)和[恢复绑定](../research_resume_20261011/cloud_payload_recovery_binding.json)。

- BI Q1：61人，hit保1287 requested，概率指标保1243可用；fine−scratch participant-macro hit +10.8958pp，42人命中/52人NLL/46人Brier改善。
- BNCI C600：八人240请求；fine−scratch hit −3.3333pp，A02/A05各净少七个正确，NLL改善主要来自A04；高置信组平均98.276%、实际正确85.897%。
- entry005：八人152请求、92面板、13984预测，原指标归约误差≤1.78e-15/结构异常0。K2000 selector132/152，NLL .262927对primary .367674；scratch135/152、NN+xDAWN133/152。K800少一命中，2救回/3损害集中A04。
- P1：16cells/576分数，private6/8对shared5/8；同输入新旧law16winner完全相同，最大Δp约2.45e-8。科学OS0/owned0，外层queue OS1保历史失败。

## 云窗口分工与当前执行

云owner chat `01a10b0f-c672-7e73-8449-4ebdaf6fc5eb` 独占云资源、freeze、source/target调度与实际利用率检查。新endpoint及原lockedvenv已恢复，软件case使用独立system code checkout、native1/hiddenCUDA、完整599member身份。下一公共执行仍按CPU25/GPU4和完整约定矩阵；base四程序不等待B07/B09新机制收尾。

下一cloud正式交付是native ABI实际holder/reader及组合case真实terminal/JUnit、内层task/source ledger、同run注册/失败分母、input lease与owned-tree RSS、source before/after SHA。41cecd6软件fixture worker2GiB来自原1.531GiB实测，最终whole-tree采样仍核同声明。软件完整接缝已接受，base科学声明也经独立14文件/四fold/123名单审阅finding0；云owner修明确source native5+interop1/claim6的25线程预算、冻结真实plan/source/inputs/resources后直接执行全约定base矩阵。BI新cohort核exact01..61，B07新增C-latent/概率ε另有opt-in freeze。只同步重要接口、阻塞及可整合交付。

## 续接的决策与实施顺序

1. 核branch/HEAD/dirty与真实进程，读本handoff、AGENTS、文档索引及相关六指南；旧checkpoint只供历史定位，不重复已完成测试。
2. 复核云组合case的actual terminal与源/任务/资源证据，按真实问题修复；接收base完整科学计划并核source/input/CPU25/GPU4实际能力，随后完整单seed执行。
3. 审阅C-latent最终运算误差、两评分法、原producer保存/恢复及必要真实出口；结合独立raw-AR oracle集成接受源码，回收证据后清理作者树。
4. B03pool尾部校准及B09长record/实际source消费者各沿理论、反例、深owner接口、纵向caller/save/load/replay审阅与集成；没有future阈值授权或prior completion伪测量。
5. B11固定noise shape K0来源和原flat measure长record程序、B10完整输入增量、B12新数据泛化与B13成人确认按各自依赖继续推进。整个123项计划仍开放。

日常科研一个完整source→target seed20260918，大版本才三个完整程序。科研子代理没有我方人工时限/额度，主代理最终仔细审阅和整合；针对实际变更验证、主题页维护结论，已完成代码树及时清理。
