# 微调泄漏复核：校准范围守卫与证据更正

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-06。原审查稿已物理保存在
`frozen/finetune_leakage_review_before_refit_fix_20260906.zip`。
本次核对实际源码快照、逐run产物并修复refit接口；不改写任何历史分数。

## 1. 结论

- `_refit_full_prefix`原来使用传入的全部X，而初次fit可以只使用部分train/validation行。
  一旦X还有未授权的test行，原refit路径会把它们重新纳入统计和训练。风险成立。
- 这不是“切回时序验证一定泄漏”：原来的两个正式runner已经先切出校准X再调用fit。
  缺陷是适配器没有独立约束refit范围，不能防止新调用方携带test上下文。
- 已增加显式`prefix_mask`，时序refit缺少它时在任何拟合之前报错。refit只处理授权校准行；
  train/validation必须属于该范围，校准范围内的inner embargo可在refit时重新加入。
- 历史高分路径未触发refit有直接产物证据，不再仅凭默认配置推断。
- 原审查稿表中的0.8150/0.7047来自混合run的原始行均值，不能作同配置24格对照。
  正确对齐后仍有配方排序翻转，但所有这些数据均为development。

## 2. 为什么性能方向不能排除泄漏

`normalization=source`且冻结BN减少了这两类拟合操作，适合作为低复杂度对照。
但“若泄漏，target统计/BN那路应更高”不成立：泄漏可能提高、保持或降低AUC。
单特征线性分数即使使用test估计均值和正标准差，正仿射变换仍保持AUC；
多特征的尺度或BN变化还可能破坏已学习的判别方向。结果方向不能证明没有数据越界。
况且source统计/冻结BN也不能阻止optimizer在错误纳入的test行上训练。
应以输入范围、实际执行记录和反例验证为依据，而不是比较谁的分数更高。

## 3. 真实执行证据

本地并不缺少云端执行材料：源码A–F压缩包位于`frozen/`，
下载结果位于`experiments/runs/p300_mainline_20260906/retrieved/p300_mainline_20260906`。
核查脚本和完整输出见
`experiments/runs/refit_scope_audit_20260906/check_evidence.py`及`execution_and_aggregation_audit.json`。

对该批2400个binary评估cell：1434个实际神经拟合记录均满足
`refit_full_prefix=false`、`refit_epochs=0`、`refit_train_losses=[]`、`best_epoch=null`和`val_losses=[]`。
其中包括246个K192 full-fine cell。对应五个实际评估源码archive已校验SHA，AST核对
`SubjectAdapterConfig(refit_full_prefix=False, val_group_fraction=None)`。
因此旧主路径“未执行refit”既有显式调用配置，也有运行后history，而不是只有条件推断。
本次补丁不需要因此作废这些已完成的固定预算结果；它阻止的是另一条现存可误用路径。

## 4. 原表为什么算错

原数字可以精确复现：仅按dataset/arm/epochs/lr/normalization/BN筛选后直接平均CSV行，
得到BI的0.6596/0.7090/0.8150与BNCI的0.7191/0.7429/0.7047。
但BI每项是32行，BNCI每项68行；有24种participant×seed键不意味着每种只有一个同条件结果。

BI zero/轻微调重复计入seed20260906；BI强微调混入`eval_width4_bi`。
BNCI混合了5导、7导、Cz六差分、不同source及两人pilot。即使先按participant/seed平均，
仍不能消除不同模型和输入方案的混合。必须按cache、source checkpoint、架构、参考、
测试协议及预声明run集合区分配置，再在participant内平均seed。

正确绑定已有主实验的K192结果如下：

| 配方 | BI五导：53人source，8人共同留出 | BNCI七导：7→1 source LOSO，8人 |
|---|---:|---:|
| zero-shot | 0.6565 | 0.7760 |
| 15epoch/lr1e-4/source统计/冻结BN | 0.7075 | 0.7989 |
| 60epoch/lr1e-3/source统计/冻结BN | 0.8153 | 0.7364 |

这支持配方效应依赖数据与协议，不能声称通用P300微调提升+0.155。
跨域翻转本身也不构成“BI选择偏差是唯一成因”的证明；源/目标分布变化可以导致同样现象。
选择偏差成立的直接依据是研究流程确实观察后缀成绩后挑了代表配方。
BNCI同样被多次用于配方比较，不能将事后较好的0.7989改称独立确认。
后续64人配对复核见[实际报告](bi64_protocol_factorial_results_20260906.zh.md)，不能与本表8人pilot混算。

## 5. 守卫的数学与实现

记C为调用方明确授权的校准范围，M为内层训练，V为内层验证，T为外层test。

\[
M\cap V=\varnothing,\quad M\cup V\subseteq C,\quad C\cap T=\varnothing,\qquad
E=C\setminus(M\cup V).
\]

初次优化仅用M，选择epoch仅用V；E可以是校准内部的时间embargo，不是test。
选好epoch后从同一初始化仅用C进行refit。target统计、目标BN缓冲、class prior、
神经优化以及candidate loss的metadata都必须按C切片。source统计模式也遵守X_C训练约束。

实现链：`SubjectAdapter.fit(prefix_mask=...)`验证布尔shape/非空、内层mask包含关系，
自动group/repetition切分仅在C内执行；`_refit_full_prefix`再次验证并同步切片X/y/mask/候选信息。
`run_candidate_cross_decision.py`和`run_within_subject_transfer.py`从既有builder得到校准子数组，
为该数组显式声明完整校准mask，并在`prefix_refit.fit_scope`保存实际行数与mask digest。
固定预算不refit的合法调用不受影响。

原来“未声明范围仍可refit全部X”的行为被删除；原有合法校准embargo语义保留。
mask不是自动识别test的算法：调用方若把test错误标进C，仍需上游数据合同阻止。
因此常规runner继续只向fit提供已经切好的校准数据，mask作为额外接口边界。

反例覆盖：缺mask拒绝、非法mask拒绝、训练/验证越界拒绝；source/target/shrinkage统计与
冻结/更新BN六种组合下，改变C之外波形和标签不改变参数、BN及预测；C内embargo仍用于refit；
越界候选元数据不会进入candidate loss；两个真实runner从builder到refit的范围记录闭合。
这属于泄漏边界修复，不是新增迁移机制，也不证明真实成人效能。

## 6. 更严格的目标隔离

本次审查时用户要求独立目标真实性对照，对应`unseen_calibration_codes`：
测试数字不能曾作为本人校准目标；每个阶段仍闪烁完整1–9候选集，因此不是“数字从未出现”。
该额外对照继续保留；之后用户明确的九数字全部校准产品主线使用独立新decision验证，
不要求排除已校准数字。当前没有正式BrainSync记录；软件与公开数据实验不能冒充成人确认。
公开部分优先使用真实候选membership可核实的BNCI，校准与test使用不同完整decision及
不同物理target pair；预先固定LR/epoch候选，不能在严格test上再选最优后称确认。
