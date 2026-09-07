# BNCI未见校准目标的严格开发对照

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

用户要求：校准与测试不能是同一目标选项，并把配方选择限制在校准内部。
本计划在查看该严格test结果前固定；BNCI本身已经多轮用于开发，不称独立确认。
BrainSync尚无正式记录，只更新前瞻采集配置，不报告成人性能。

## 1. 数据隔离

- 复用BNCI2014-008原始builder校验过的36选一行列membership和七导binary cache。
- 每个目标participant复用其余7人训练的source checkpoint，3个既有seed；target不在source身份表。
- 最大校准为按时间前5个完整selection，分别使用前1/3/5个作为三档预算，实际120/360/600trial。
- test统一取最大校准之后、原始epoch时间无重叠、且物理target pair不属于前5个校准target集合
  的完整selection。三个预算用同一test。每个test pair均未作为该人的校准目标。
- “未见目标”不是“刺激从未出现”：每轮仍闪烁全部12个行列组；只隔离曾被关注的物理选项。
- 同一连续记录中的不同decision不等于跨session/跨日，剩余范式与视觉捷径另需真实对照。

## 2. 固定方法

| 方法 | 拟合数据与固定规则 |
|---|---|
| zero-shot | source参数/统计，无target拟合 |
| full-low | 校准C，15epoch/lr1e-4，source统计/冻结BN |
| full-high | 校准C，60epoch/lr1e-3，source统计/冻结BN |
| inner-selected | 只在C的时序内层验证上比较上述两配方的最佳weighted CE，选其较低者；从同初始化refit完整C |
| target-only | 同结构随机初始化，C统计/更新BN，60epoch/lr1e-3 |
| raw shrinkage LDA | 仅C拟合scaler/covariance，20Hz ERP特征 |
| candidate-identity-only | 只用可观察flash group ID，one-hot+正则logistic；诊断记忆刺激身份捷径 |

inner-selected通过既有`chronological_time_validation_split`选择内层train/validation，
显式`prefix_mask`授权C；test X/y不传入fit、配方选择或refit。固定配方不使用test选择epoch。
这些是协议与训练范围对照，没有新增神经机制。两套固定配方均报告，不在test上挑一个再宣传。

## 3. 输出与检验

participant-macro trial AUC及R1/2/3/5/10的36选一准确率；不是BrainSync九选一。
每cell保留source/cache/评估源码/harness SHA、C/T行、target集合、所选配方、fit_scope和逐trial预测。
选择参与者和test cohort不依赖模型分数，缺失或失败单独保留，不用低分删样本。
训练标签/时间/participant/target pair隔离逐个断言，独立重算AUC与decision outcomes。
三个seed不扩张participant样本量，所有结论限定为公共development。

执行完成与独立重算见[结果与验收](bnci_unseen_target_results_20260906.zh.md)。
解码候选始终保持完整36格，不把校准目标从解码器候选中移除。
