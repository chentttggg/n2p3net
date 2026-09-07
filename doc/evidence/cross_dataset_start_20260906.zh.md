# BI与BNCI双向跨库训练首轮

日期：2026-09-06。公开ERP跨库开发实验；不是同库LOSO，也不是成人BrainSync确认。
执行状态见[当前计划](../roadmap.zh.md)。完整本地结果位于
`experiments/runs/cross_dataset_start_20260906/`，可核验产物见
[交付清单](../../frozen/cross_dataset_start_20260906.delivery.json)。

## 设计与真实性

BI2014a全部64人/61,011 epoch与BNCI2014-008全部8人/33,600 epoch分别作为source，
目标是另一个完整数据集。共同实测五导CZ/P3/PZ/P4/OZ、native right-earlobe参考，
使用已attested的causal-v3 binary cache。目标库任何被试都不进入source训练。
不将五导模型直接替换工作站当前Cz六差分候选，产品蒙太奇/参考须另行适配验证。

seed固定20260906；目标K48/96/192，各K共享最大前缀之后经epoch时间purge的相同test。
本人早期校准标签可用，测试EEG/标签不用于fit、normalization或BN。
这仍是单recording binary前缀/后缀，未证明物理目标选项未见或跨session独立。
BI62–64保留在原64人请求队列，未按新分数排除。

首轮六核心臂：zero-shot、保留源head的classifier-fine/full-fine、合理target-only、
raw shrinkage LDA与冻结源特征LDA。轻微调15epoch/lr1e-4；额外两个强微调臂60epoch/lr1e-3，
均用source统计/冻结BN。target-only为随机权重、前缀统计、训练BN、60epoch/lr1e-3。
预声明主比较为K96轻full-fine减target-only；旧强/弱配方都报告，未在test上选赢家。

首轮1,728单元后，基于明显跨库退化追加432个配对归因单元：固定强full-fine模型/数据/预算，
仅改为前缀统计，再加入BN更新。该追加设计发生在观察首轮后，明确属于development探索。
总计2,160个单元及预测文件全部完成；逐文件重算AUC、核验C/T行与checkpoint身份，
所有方法/预算的test行与标签一致。source checkpoint的完整训练身份账本与源库盘点相等。
identity核验使用已有source/global authority，不宣称新增了完整跨研究全球身份登记。

## 主要读数

K192 participant-macro trial AUC：

| 方法 | BI→BNCI（8人） | BNCI→BI（64人） |
|---|---:|---:|
| zero-shot | 0.5117 | 0.5108 |
| classifier-fine轻配方 | 0.5201 | 0.5140 |
| classifier-fine强配方 | 0.5818 | 0.5462 |
| 源特征+LDA | 0.5897 | 0.5506 |
| full-fine轻配方，source统计/冻结BN | 0.5288 | 0.5214 |
| full-fine强配方，source统计/冻结BN | 0.6302 | 0.6163 |
| full-fine强配方，前缀统计/冻结BN | 0.5972 | 0.6733 |
| full-fine强配方，前缀统计/更新BN | 0.6598 | 0.7067 |
| 合理target-only NN | 0.6370 | 0.7061 |
| raw shrinkage LDA | 0.6486 | 0.7085 |

预声明K96主比较为负：BI→BNCI -0.0568，95% CI [-0.1039,-0.0164]；
BNCI→BI -0.1182，[-0.1387,-0.0978]。这首先否定当前轻微调配方的效用，
不能单凭它否定更充分的源学习或所有迁移方法。

追加K192前缀统计+BN更新相对target-only：BI→BNCI +0.0228，CI [0.0120,0.0332]；
BNCI→BI +0.0006，[-0.0111,0.0115]。相对LDA分别+0.0112、-0.0017，CI均含0。
这些CI是固定源模型/配方下的participant bootstrap，不包含训练seed不确定性或开发选择偏差，
不能将一个方向的正CI当作普适迁移收益。完整K48/96/192和配对负迁移人数均保存在analysis JSON。

统计与BN改变确实缓解部分退化，但不是单一通用修复：BI→BNCI只换前缀统计会下降。
当前仅能说直接跨库初始化较弱、适配配置影响很大，尚未显示相对强LDA的稳定优势。

## 为什么训练快

本轮复用原紧凑基线，没有削减源数据。每个模型只有1,186个可训练参数：
5×179输入、8个时域滤波器、16个空间通道、两个MST分支各2通道，共4特征通道，
44个特征时点展开为176维，再接二分类线性头，dropout=0.25。

| 来源 | 最多epoch | 实际完整source refit | 选轮拟合耗时 | refit耗时 | source内部所选验证AUC |
|---|---:|---:|---:|---:|---:|
| BI | 30 | 21 | 16.63s | 10.94s | 0.6971 |
| BNCI | 30 | 25 | 12.87s | 8.87s | 0.7204 |

checkpoint的`best_epoch=20/24`是零基索引，不能误写实际只训20/24轮。
两任务并行总墙钟35.48秒，包括加载/编译等开销。RTX5090、batch512、BF16、数据预载、
fused Adam和整train-step编译均实际生效；没有OOM。不能只引述fit_seconds而漏掉refit成本。

快的一部分原因确实是模型容量小和本轮预算有限。当前尚无本轮的容量×训练预算充分性实验，
因此不能称主干已经训练充分，也不能把负迁移归因成数据之间没有可迁移特征。
下一项优先对照是在固定source划分下比较MST每分支2/8通道与最大30/120epoch，
记录实际选择/refit轮数、loss和预算，再做冻结配方的迁移复核；具体步骤由执行计划维护。

## CPU与GPU调度核查

源训练用GPU；首轮1,728目标单元用6个CPU进程、每进程2线程，墙钟224.15秒。
追加432单元同配置用142.92秒。CPU高而GPU低是阶段调度结果，并非源训练漏用CUDA。
但最初选择CPU尚缺这台机器的目标阶段计时依据，不能声称已经证明CPU更快。

补做同源模型、BI01–03、K192、60epoch/lr1e-3、前缀统计+训练BN的CPU/CUDA比较，
各重复两次、第二次反转顺序；完整CLI包含cache加载，单元耗时包含fit和推理：

| 设备 | 每次3人完整CLI墙钟中位数 | 单人单元耗时中位数 |
|---|---:|---:|
| CPU，2线程 | 13.72s | 2.56s |
| CUDA，2线程 | 12.84s | 2.35s |

GPU单任务约快6.4%，不足以代表6并发下的端到端吞吐；后续调度应补并发对照，
不能只追求GPU利用率。CPU/CUDA随机训练的预测不逐位相等，本次个别AUC差异可达约0.058，
不能在同一科学表内偷偷更换后端后拼接结果。基准12个单元只作性能资料，不扩充研究样本数。

## 产物与保全

- 训练/评估源码快照B：`frozen/cross_dataset_start_20260906b.manifest.json`，执行前逐文件校验。
- 快照A在预检时错误假设BI timeline的dataset_id等于研究库，未启动训练；B按保留的原库合同及逐被试stream核验，原cache未改写。
- 初轮与归因结果分别压缩保存，两端SHA-256核验；benchmark另存，旧结果文件不覆盖。
- 本地30项binary/supervised/evaluation合同测试通过；2,160份预测全部独立重算与配对审计。
- 三seed、四库扩源、容量/预算充分性、未见物理目标跨库对照与成人验证仍未完成。
