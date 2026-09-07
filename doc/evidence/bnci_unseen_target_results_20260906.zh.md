# 未见校准目标的BNCI开发结果与refit修复验收

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-06。设计见[预声明计划](bnci_unseen_target_plan_20260906.zh.md)，
代码风险与原审查更正见[refit复核](finetune_leakage_review_20260906.zh.md)。
这是公开BNCI2014-008、同记录后续decision的开发实验，不是成人BrainSync确认。

## 1. 严格范围

8人、3个既有seed；每个source模型严格排除当前目标participant。
校准预算为前1/3/5个完整selection，即120/360/600trial、20/60/100个target flash。
test统一来自最大5个校准selection之后，且target pair不属于这些校准target集合。
每人恰好25个完整test selection，8人共200个不同decision；三个seed不扩张独立样本数。
三个预算和全部方法使用相同test row，原始epoch时间不交叉。

解码器仍在完整36格中选择，不因校准目标被排除就缩小候选词表。
“未见”只指未作为本人校准目标：所有行列flash仍可在校准中作为非目标刺激出现。
同一连续记录的不同decision不是不同session/跨日，剩余时序、视觉和凝视线索仍需其他对照。

## 2. AUC

先每participant平均3seed，再平均8人；未在该test上选方法。
low=15epoch/lr1e-4，high=60epoch/lr1e-3，均source统计/冻结BN。
inner-selected仅用校准C的时序内层验证比较两配方的最佳weighted CE，选较低者后从同初始化refit完整C。

| 方法 | 120trial | 360trial | 600trial |
|---|---:|---:|---:|
| zero-shot | 0.7782 | 0.7782 | 0.7782 |
| full-low | 0.7940 | 0.8121 | 0.8190 |
| full-high | 0.7180 | 0.7577 | 0.7884 |
| inner-selected | 0.7858 | 0.7811 | 0.8136 |
| target-only tuned | 0.6368 | 0.7298 | 0.7692 |
| raw shrinkage LDA | 0.6360 | 0.7212 | 0.7650 |
| candidate identity only | 0.4480 | 0.4680 | 0.4800 |

固定high在大校准预算下改善，但仍不优于low；不能把在BI选择的强配方宣称为通用最优。
inner-selected不读取test来选配方，但小校准预算下仍可能选不好；合法选择并不保证最佳效能。
其24个participant-seed单元在120/360/600trial下选择high配方的次数分别为12/19/16，
实际refit轮数由各自校准内层选择，不等于一律训练到60轮。

## 3. 决策表现

600trial校准下的36选一结果，同一200个decision，逐participant平均seed。

| 方法 | R3准确率 | R10准确率 |
|---|---:|---:|
| zero-shot | 0.4467 | 0.7417 |
| full-low | 0.5767 | 0.8683 |
| full-high | 0.5267 | 0.8100 |
| inner-selected | 0.5383 | 0.8717 |
| target-only tuned | 0.4800 | 0.7717 |
| raw shrinkage LDA | 0.4650 | 0.7750 |
| candidate identity only | 0.0000 | 0.0000 |

只看可观察flash编号的one-hot+正则logistic对照在所有预算的decision准确率均为0。
这不支持“只记住校准刺激身份即可解释神经模型表现”。但不能据此排除全部视觉/凝视捷径，
也不能声称全部判别信息就是P300。BNCI已反复用于开发，这不是独立确认结果。
R10与R3、36选一与BrainSync九选一、600trial与旧K192都不得直接混算。

## 4. 验收

源代码冻结：`frozen/refit_scope_guard_20260906.manifest.json`，archive SHA256
`0271ec2f3feff762fa6b14e85c05e61b3bcb00f808642f65d9fb3c32b0e9374c`。
复用source checkpoint训练快照E，没有重新训练source或让target进入source。
新GPU路径完成504个cell，全部成功；144个内层候选实际执行了带显式校准范围的refit。
独立分析逐个检查source训练身份、校准/test row及target pair隔离、相同test cohort、
refit范围和内层配方选择依据，并从保存的预测重算AUC与全部decision准确率。

CPU定向测试209项：208通过、1个需独立采集端路径的可选集成跳过。
新增边界反例及两个真实runner的时序refit集成通过；changed-file ruff通过。
原未声明mask的反例先运行并得到“DID NOT RAISE”，补丁后通过。
GPU实验总墙钟约293.5秒，未改变batch/seed以追求吞吐。

非神经对照不适用统一日志中的神经normalization/BN占位字段；
独立汇总的`method_inputs`明确其真实输入和算子，不以这些占位字段合并配置。
原始分数和合同日志保留，不能用participant×seed覆盖齐全替代配置身份核验。

执行目录：`experiments/runs/bnci_unseen_target_20260906`；云端同名目录位于
`/root/autodl-tmp/n2p3-runs/`。数据、测试、冻结包及结果入口见
`frozen/refit_scope_guard_20260906.delivery.json`。

## 5. BrainSync下一步

本实验对应的额外前瞻对照是`unseen_calibration_codes`：校准目标1/3/5、test目标2/4/6/7/8/9，
仍呈现并解码全部1–9。用户随后明确产品采用九数字全部校准、独立新session测试，
当前采集配置已跟随该产品目标更新；本实验的目标隔离、数据、分数和结论保持原口径。
实际仍需完整采集、目标确认及独立session验证。
真实BrainSync可以增加本人校准量，但配方选择只能发生在校准/开发部分；外层test不能参与。
独立目标结果只回答一个更严格的真实性问题，不替代眼电归因、跨日和重新佩戴验证。
