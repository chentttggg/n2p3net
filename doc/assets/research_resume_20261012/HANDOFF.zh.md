# 科研总监暂停交接

2026-10-13用户明确要求“暂停工作，收尾更新handoff”。本记录是续接入口，活动科学规范仍由六指南和四个主题页持有。科研目标暂停，整体计划未完成；本轮所有科研子代理已结束。云窗口是独立执行owner，本记录不改变其任务和进程。

## 工作身份与恢复

- 当前工作树：`C:/Users/chent/.codex/worktrees/research-director-resume/n2p3-net`。
- 当前分支：`codex/research-director-resume-20261013`，恢复基点`f1d0ec0b912af0e0dc8e8e5c269c5d9e8bb7b98c`。
- 暂停前完整代码/结果文档提交：`577ac4c68db17cf3ce12431add8f371deaab5759`。本handoff及当前checkpoint随后作为文档提交；最终提交可由该分支tip解析，避免在文件中自引用提交SHA。
- `f6c8ca5`：目标校准scope和严格warm-start完整状态覆盖；`2dc48dd`：Gaussian诊断及中央资源floor；`afa9bbb`：当前科学路线与验收；`577ac4c`：从云端`4c4b987`接收完成读数，保较新的科研记录并解决四个文档冲突。
- 暂停收尾时实际`merge-base --is-ancestor 33603f7431ca50465e2d1678eef51145a1c63f3c HEAD`退出0：Claude review原tip已是当前树祖先，当前无需再次合并旧branch。原提案的替代/否决以及随后修复按固定源码覆盖和活动指南解释。
- `D:/BCI/n2p3-net`仍是`main@f1a303c42c68173e8b1452ceb26bd84923b8a6aa`，暂停收尾时345条porcelain记录，完全保留。本轮未切换main、未push、未删除工作树/分支、未prune或gc。
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

四个完成程序的报告/紧凑完成绑定已从保留Git提交恢复，不是本轮重新评分。详见[公共评估](../../evidence/public_evaluation.zh.md)、[响应与噪声](../../evidence/response_and_noise.zh.md)及[原读数](../research_resume_20261011/cloud_completion_readout.json)。

- BI Q1：61人，hit保1287 requested，概率指标保1243可用；fine−scratch participant-macro hit `+10.8958pp`、NLL `−0.590480`。
- BNCI C600：8人，每臂240；fine−scratch hit `−3.3333pp`，NLL改善主要来自A04。
- entry005：8×19=152请求，92面板、13,984预测，结构异常0。K2000 selector132/152，NLL `.262927`对primary `.367674`；scratch135/152、NN+xDAWN133/152。K800少一命中且概率指标变差；K200回基线。
- P1：16 cells/576分数，private6/8对shared5/8，只有A06救回；新旧stationary-law赢家一致。entry005及P1 science exit0/owned0，外层queue exit1保历史失败。

旧ignored大结果/模型的本地路径已失效，当前SSH端点此前拒绝后超时；原始载荷持久性未重新确认。完整完成身份与原始payload恢复是两个状态，不重签旧producer、不从超时推断可重启训练。

## 云窗口分工与未集成实现

云owner chat：`01a10b0f-c672-7e73-8449-4ebdaf6fc5eb`，host `local`。其新树`C:/Users/chent/.codex/worktrees/prefix-replay-integration/n2p3-net`，收尾读取时HEAD `263df81e9f5a675e929477754d5fafec09f7e333`，有`ensemble_views.py`及新增共享输入模块/测试的未提交工作，保持原样。

云窗口持有CLI、dispatch、IO/lifecycle和node/chunk实现。最新即时快照称共享输入记账六例通过、节点串行/分块/逆序归并一致且独审接受，实际调度生命周期仍在核验；整体实现尚未交由科研root最终整合。继续消费owner最终handoff和真实代码，不把局部软件结果写成公共吞吐改善。

旧冻结云路径为`/root/n2p3-runtime-20261011-control/prefix_catalog_recovery_source02/checkout`及`p1_queued_source02/checkout`，目前仅保恢复定位信息。接收新调度前确认：一次graph/package准入、digest绑定node/chunk、原顺序归并、requested失败分母、source/input lease、退出及共享输入预算均闭合。同步本轮PDS scope、warm-start及direct floor接口变化，云owner须以实际source/config/plan重新冻结。

## 续接的决策与实施顺序

1. 先核branch/HEAD/dirty状态，读本handoff、当前checkpoint、`AGENTS.md`、文档索引及相关六指南/四主题页；历史binding按原源码使用，不重跑旧全集。
2. 核云owner最终实现和原始载荷恢复状态。已完成科学结果不为性能改造重复拟合；清理工作树前核恢复包与实际进程依赖。
3. 首个可独立研究切片：binary calibration selector仍在各candidate自己的finite logits上排AUC，选择性abstention可能获利。此前candidate-dependent交集提案已正确否决；需要设计共同requested-risk/availability合同，写反例并检查完整caller/save/replay，再实施。
4. 接云node/chunk候选时root仔细逐模块审阅，不以作者测试数量替代集成验收。优先核内存/lease/reduction与数学score等价，再做必要针对性检查。
5. B07冻结同支持公共opt-in、fractional时钟authority及C-latent新producer；B05更多selection/类别、高置信校准和joint MLE；B09 source builder/reference transport/ensemble；B10/B11/B12完整科研出口及B13成人/设备确认继续按依赖推进。

日常seed `20260918`，大版本才三seed；科研子代理无我方人工期限/额度。没有用户恢复指令前不再推进科研代码、新训练或新代理任务。暂停是用户选择，未完成研究不标complete或blocked。
