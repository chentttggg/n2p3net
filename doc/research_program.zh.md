# N2P3-Net 科研总纲：未知数字、多决策与 90% 目标

日期：2026-09-02
状态：living research guide
适用范围：oddball/P300 单试次检测、多试次候选聚合、跨被试预训练和单被试校准。
阶段结果摘要见 [`research_status_report_20260902.zh.md`](research_status_report_20260902.zh.md)。

## 0. 权威边界

本文件是研究状态、估计量和执行顺序的唯一总入口。冲突时按以下顺序裁决：

1. 带 cache/checkpoint SHA、完整预测、环境和完成标记的原始制品；
2. 当前可执行代码、反例测试和真实数据 smoke；
3. 本文件；
4. `constitution.md`、`blueprint.md` 等稳定原则；
5. 日期化报告只保留当时发生过什么，不自动代表当前结论。

模型状态只使用：

| 状态 | 含义 |
|---|---|
| 软件默认 | 代码可运行的默认值，不代表准确率最好 |
| 探索领先 | 在已查看数据上领先，可生成下一轮假设 |
| 确认冠军 | 在未用于选模型/recipe 的冻结外层数据上通过统计门禁 |
| 部署冠军 | 确认冠军再通过失败、覆盖、延迟、内存和校准成本门禁 |

当前没有确认冠军或部署冠军。

## 1. 目标定义

最终产品目标不是单试次 BACC，也不是跨被试 LOSO AUC：

> 在成人 BrainSync 目标域，同一被试先完成预声明的已知目标校准 decisions，
> 随后在独立、目标数字未知的 decisions/session 中，以固定最大重复预算 `R`
> 完成 9 选 1；失败、缺候选和 abstain 均进入分母，subject-macro `hit@R`
> 点估计达到 0.90。

`hit@5` 不是通用标准，只是一个固定预算切片。最终产品的 R 必须由采集允许的最大
刺激/时间预算预先声明。冻结的 v4 核长实验仍按其预注册
`hit@all_balanced` 解释；后继 count-correction 结果已表明，在候选 trial 数不齐时，
当前 GTN 开发默认应使用全部 trial 的 candidate mean (`count_power=0`)。
balanced truncation、sqrt-count 和 raw sum 是共同兼容证据，不再作为新的默认。
必须同时报告：

- `hit@all_balanced`、raw `hit@all`、`hit@1..R`、每个预算的覆盖率、abstain 和
  实际刺激/秒数；
- trial AUC/BACC、group NLL/ECE，但不得代替 9 选命中率；
- 每人多个独立 test decisions 和每人命中率分布；
- requested-cohort operational hit 与 eligible conditional hit；
- point estimate 与 95% CI。点估计 `>=0.90` 只能写“观察达到”；CI 下界
  `>=0.90` 才能写“可靠达到”。

以 175 个独立 decisions 为例，`158/175=0.903` 的 95% 下界约 0.85；约需
`166/175=0.949` 才能让下界超过 0.90。最终样本不能把同一人的多个 decision
当成独立被试；推断需 subject-cluster bootstrap 或层级模型。

## 2. 当前数据能回答什么

### 2.1 GTN

GTN 是 3 导、7--17 岁数据。每名被试只有一个固定 `thought_number` 和一个
selection，目标数字分布也不均衡。它可以回答：

- target-excluded cross-subject / zero-shot 的 9 选一次决策能力；
- 随机刺激流中，等待每候选获得 M/R 个有效证据的 operational evidence-budget；
- 模型、滤波、窗长和聚合机制的开发比较。

它不能回答：

- 成人 8 导 BrainSync 的部署准确率；
- 每个单被试长期 90%，因为每人没有多个独立未知数字 decisions；
- “监督校准后猜未知数字”：同一 selection prefix 的真标签已经泄露 suffix 的
  同一 thought digit，并可能允许 digit-glyph/VEP shortcut。

因此 GTN runner 的合法主模式为：

```text
Z0: M=0, target-excluded checkpoint -> earliest suffix, session-start zero-calibration
Z5: M=5 labels unused -> later suffix, matched-time zero-target-fit sensitivity
O5: M=5 labels used -> same-selection oracle personalization proxy only
```

`O5` 必须显式确认，永远不能进入产品 90% 结论。增大 GTN 的无标签 M 只会把
测试推迟到 session 后段；增大有标签 M 只会加强同 digit oracle，不会增加合法的
未知数字校准量。真正的 calibration M 应按多个目标变化的已知 decisions 计数，
只能在 BI/BrainSync cross-decision 数据上研究。

GTN 没有同步的“每轮 1--9”。`repetition_index` 是 candidate-local occurrence，
不同候选的第 r 次发生在不同真实时间。组级模型可使用“每候选前 R 个累计证据”，
但不能把伪 round 解释成同步 9-candidate context。若需要同步 set，采集端必须提供
真实 `block_id`。

### 2.2 BI2014a

BI2014a 有多个、目标变化的 character selections，可检验：

```text
前 K 个已知 character decisions 校准
-> raw-time embargo
-> 后续未知 character decisions 的 6x6 hit@R
```

它不是 9 选数字最终数据，但能回答“个体 ERP 校准是否迁移到新目标”。旧 candidate
cache 与 causal-v2 结果已压缩隔离；当前必须从 raw 用统一 causal-v3 构造器重建。

### 2.3 BrainSync

BrainSync 成人 8 导、多数字、多 session 数据是 90% 的最终裁决域。正式采集至少需要：

- 每人多个已知 calibration decisions，目标数字均衡随机；
- 多个独立未知 test decisions，目标在 calibration/test 间变化；
- acquisition/source/model sample rate、参考、坐标、EOG/QC 独立记录；
- session 边界、真实 `block_id`、事件 onset 和 evidence-available time。

工程入口现已支持 causal steady-state、严格 analysis-ready、多 session
`started_utc` 排序和显式 target-policy split/runner。活动 session v4 只接受
`completed`，一 session 是一个 decision；每个 block 是完整 9 候选随机排列循环，
block 仍只表示调度/休息分段，不证明 target-switch；显式公共通道 CAR 域适配另行记录。当前
默认固定为 128 Hz、0.1--30 Hz、`[-200,1200) ms`、179 samples；没有 2/800
降级选项。当前
本机 4 个历史 sessions 分别缺 recording、recording_error 或仍为
`analysis_ready=false`/target pending，均不能进入准确率分析；必须重新采集。

## 3. 2026-08-31 证据清算

| 证据 | 当前状态 | 允许结论 |
|---|---|---|
| BI2014a 128 Hz zero-phase LOSO assets | 已进入物理压缩归档 | 只作架构开发；不可作为当前 checkpoint |
| BI2014a causal-v2 cross-decision | 物理归档：64 人、12 checkpoints、13 arms x 3 seeds | 只保留机制结论；v3 必须重建重训 |
| BI2014a+BNCI2014_008 common-CAR5 source | 物理归档：`0.1300/0.0967/0.1239/0.0974` hit@2；2026-09-02 对旧 cache 只读机制复审 | 负迁移来自条件 ERP 冲突、epoch/participant 风险错位和 validation 域漂移；旧 128-sample cache 不可续用 |
| BNCI-target causal-v3 CAR5 正式矩阵（295942e） | 96/96 checkpoint+result 完成；manifest 两次失败已定位修复（`ab6c8e7`）；本地独立复算完成，待云端 checkpoint 下载后重建 manifest 并物理冻结 | hit@8 `bnci_only 0.3417 > bnci80_epoch 0.2736 > bnci80_participant 0.2667 > joint_natural 0.1222`；联合臂不晋升；J2≈J1 关闭统计单元轴；预注册梯度冲突门已触发 |
| GTN 2 Hz/800 ms assets | 仅存在物理压缩归档 | 不提供活跃降级接口 |
| GTN steady-state causal 2x2 bundle | 4 cache records、32 checkpoints、120 eval JSON 已本地独立复算 | 0.1 Hz/1200 ms 是当前开发 signal winner；不作产品确认 |
| GTN Z0 最佳固定-R baseline | hit@5 coverage 230/245；conditional 0.578；operational 0.543；AUC 0.709 | 未达到 0.90；15 人仅因 R_s=2--4 无法到 @5，仍进入 all-evidence 主分析 |
| source QC 100 uV vs none | operational +0.118，95% CI [+0.053,+0.184]；覆盖相同 | 冻结 source QC=100 uV，target-prefix QC 仍关闭 |
| full-unfold K33/K35/K65 v4 | 36 checkpoints、36 ledgers、3 seeds；balanced-all 0.623/0.669/0.654 | K35 临时工程默认；K35 稳定胜 K33，未确认胜 K65 |
| v4 ledger all-evidence count correction | K35 mean/sqrt/sum=0.714/0.703/0.687；K65=0.683/0.683/0.669 | 不丢 trial 的 candidate mean 为当前开发默认；learned decision-only 头不晋升 |
| end-to-end decision-aligned fine-tune | 物理归档：K35/K65 x 3 seed x 4 block；learned=0.688/0.676 | 当前联合微调 recipe 不晋升；v4 不进入活跃链 |
| legacy zero-state causal ranking | 同一 suffix 连续选 QC/epoch/block/aggregation 且有 startup transient | 数值已从当前指导移除 |
| 旧 forward causal cache/checkpoint | IIR 零状态产生严重 startup transient | 永久拒绝；steady-state replacement 已完成 |
| 旧 BI candidate causal cache | repetition metadata 语义错误 | 永久拒绝；causal-v2 replacement 也已归档，当前重建 v3 |
| 当前代码 | 反例和聚焦测试通过 | 只证明合同能力，不是准确率结果 |

旧 `forward` IIR 从零状态启动，在带 mV 直流偏置的 GTN 原始记录上会把前期 trial
制造成 mV 级伪迹，而 later suffix 已趋稳，形成假的 prefix/suffix domain shift。
当前活跃合同只有：

```text
p300_single_subject_causal_v3
gtn_paper_causal_v2
```

所有旧 causal accuracy、checkpoint 和 cache 不得与新版本直接比较。

## 4. 硬门禁与可消融 recipe

### 4.1 硬门禁

这些约束修复的是 estimand 或确定性错误，不因短期准确率下降而关闭：

1. target subject 不得出现在 source checkpoint；用 cache-qualified/global identity 核对；
2. 监督校准和 test 必须是不同、目标变化的 decisions；同目标只可标 oracle proxy；
3. online causal 路径不得用整段 zero-phase 后再切 prefix/suffix；
4. `max(train evidence_available) < min(test epoch_start)`，重叠样本进入 embargo；
5. checkpoint 固化有序通道、参考、完整 preprocessing、cache SHA 和训练被试；
6. 缺候选、tie、训练失败和 abstain 不得从分母静默删除；tie 一律 abstain/miss；
7. 所有训练、标准化、QC、校准只看其被授权的 source 或 calibration 数据；
8. 结果必须保存 trial score/label/candidate/time、fold、seed、环境和完整 exclusion ledger。

### 4.2 准确率消融轴

以下不是永久禁令；每个轴必须记录当前状态，不能把已完成比较继续写成待办：

| 轴 | arms | 当前状态 |
|---|---|---|
| causal high-pass | 0.1 vs 0.5 Hz | GTN steady-state 已完成，development 冻结 0.1 Hz |
| epoch end | 800 vs 1200 ms | GTN 2x2 已完成，development 冻结 1200 ms |
| online/offline | forward steady-state vs split-local zero-phase | 机制敏感性；禁止 whole-record zero-phase |
| normalization | source stats / target-prefix stats / shrinkage | 旧 causal-v2 归档观察：source≈shrinkage、target-prefix 下降；v3 尚未重跑 |
| multi-source normalization | all-source stats / target-dataset source stats | 已完成且等效：BI-source stats 相对 all-source `+0.07 pp`，CI 跨 0 |
| multi-source 域质量/统计单元 | B0 单域 / J0 natural / J1 80-20 epoch / J2 80-20 participant | 两方向已完成：BI-target 归档 `-3.33 pp`（uniform）、`-0.61 pp`（3x/1x）；BNCI-target v3 `J0-B0 -21.94 pp`、`J1-B0 -6.81 pp`、`J2-J1 -0.69 pp` hit@8；联合不晋升，单元轴关闭 |
| target QC | none / prefix-fit fold-local | source QC100 已冻结；target QC 待独立 decision 消融 |
| epoch budget | source/fixed budget / real-time target holdout + full-prefix refit | 代码闭合，性能待独立 decision |
| adaptation | zero-shot / pretrained-classifier fine / scratch head / full fine / adapter-only inner loop | 2026-09-02 机制闭合：`SubjectAdapter(head_kind="adapter")` 在冻结 trunk+classifier 上只训练保留槽 `__target__` 的零初始化残差 adapter（eval-mode 确定性前向、无 BN 统计更新）；性能待 BI/BNCI v3 cross-decision |
| BatchNorm | frozen running stats / target adapt | 待合法 cross-decision |
| feature-domain mechanism | none / S1 per-domain residual adapters / A1 class-conditional mean alignment / AS1 两者 | 2026-09-02 机制闭合（见 Gate 2 预注册）；单轴实验待跑 |
| aggregation | all-evidence mean / tempered effective count / sum / fixed-count trim | mean 当前领先；all/R/cost 共同报告；precision 仅有预测方差时启用 |
| decision objective | trial CE / 9-candidate listwise + trial CE | 30-epoch全参数实验已完成且不晋升；保护 backbone 的分阶段策略待新实验 |

旧 B0->C1 同时改 high-pass 与窗长的结果已被新的配对 2x2 取代。steady-state Z0
factorial 中，1200 ms 平均提升约 `+20.20 pp`，0.1 Hz 平均提升约 `+5.92 pp`，
交互项 CI 跨 0；当前开发 winner 为 0.1 Hz/1200 ms。该选择仍来自已查看 GTN
development cohort，不是成人产品确认。

## 5. 评分与校准数学

设 binary 输出为 `z_i = logit_target - logit_nontarget`。加权 CE 的正类权重为
`w`，训练正类先验为 `pi`，则解析 LLR 为：

```text
LLR_i = (z_i - log(w) - log(pi/(1-pi))) / T,  T > 0.
mu_d = sum_i weight_i * LLR_i / sum_i weight_i
n_eff,d = (sum_i weight_i)^2 / sum_i weight_i^2
S_d = mu_d * n_eff,d^beta, beta in [0,1]
d_hat = argmax_d S_d.
```

单位权重时 `beta=0/0.5/1` 分别为 candidate mean、sqrt-count tempering 和
raw sum。当前 GTN all-evidence 开发结果选择 `beta=0`；它使用全部 trial，但不让
随机 occurrence 数量直接成为数字得分。只有独立 train-only 可靠性证据才能进入
`weight_i`，不得使用 test truth/correctness。

正温度保证不反转排序。固定完整 R、每候选计数相等时，公共正仿射变换不改变
argmax；普通 Platt 在极小 validation 上可能学到负 slope，因此不得用于候选排序。

### 5.1 多源训练风险

共同通道和 CAR 只定义可比较输入，不保证 `P(X|Y,D)` 相同。设域质量为
`alpha_d`，类别 CE 权重为 `c_y`，域 `d` 的合法 source participants 为
`S_d`。与 subject-macro 外层目标一致的主风险是：

```text
L_subject = sum_d alpha_d / |S_d| * sum_(s in S_d)
              [sum_(i in d,s) c_(y_i) CE_i / sum_(i in d,s) c_(y_i)]
sum_d alpha_d = 1.
```

`epoch` 控制臂把 `S_d` 替换为一个包含该域全部 epoch 的单元。实现不重复、不删除
任何物理行；逐行 multiplier 在 inner-train split 与 full refit 内分别重算，使每域
class-weighted coefficient mass 精确等于 `alpha_d`。mini-batch 仍保留原
`CrossEntropyLoss(weight=[1,pos_weight])` 的类别权重分母，因此当 `epoch` 臂的
`alpha_d` 等于自然 class-weighted 行比例时，所有 multiplier 严格为 1，退化回原 CE。

多源 inner validation/calibration 只允许读取显式 `selection_domain` 的 group-disjoint
participants。辅助域长记录不得因一个 participant 有更多 epoch 而占据模型选择目标。
域 ID 来自多源 builder 写入的精确 `source_domain` 行轴，不再从 subject 字符串前缀猜测。

`trim0.2` 固定按排序裁 `floor(0.2*n)` 个两端值；`n<5` 不裁。旧 quantile
实现会在 R=2 时删除两个不同值并产生编码依赖 tie，已废弃。任何并列最高分都
abstain，不按最小数字破 tie。

GTN operational M/R 定义为：先等待每候选 M 个可用 evidence；过全局 raw-time
边界后，再等待每候选 R 个新可用 evidence。它必须同时报告为获得这些证据消耗的
scheduled events 和秒数。它不等同 fixed scheduled-block estimand；后者若运行，缺失
证据必须记 miss/abstain。

## 6. 训练路线

### Gate 0：代码与制品闭环

- causal steady-state GTN 0.1/0.5 x 800/1200 四个 cache 已重建并审计；
- BI/BNCI/GTN 当前 v3 cache 尚未从 raw 重建；旧 cache 已物理压缩隔离；
- checkpoint 完整输入签名、source refit、subject/cache identity 已闭合；
- Z0、Z5、O5 和 BI cross-decision 输出不可混报；
- 所有核心反例测试、Ruff、full pytest 通过。

### Gate 1：信号 recipe

在 target-excluded source pretraining 和同一冻结 zero-shot evaluator 下，比较：

```text
high-pass {0.1,0.5} x tmax {800,1200}
```

配对 2x2 已完成并冻结 0.1 Hz/1200 ms。准确率为主，trial AUC、startup PTP、
保留率、刺激成本和显存为辅助；旧 zero-state suffix 不再参与选择。

### Gate 2：source-supervised zero-shot

每个 target 使用明确排除自己的 checkpoint。source inner validation 只选 epoch，
随后从同一初始化在全部合法 source rows refit。`full_unfold` 读出已经采用；核长
比较已完成：

```text
K35: no-fine mean/sqrt/sum = 0.714/0.703/0.687
K65: no-fine mean/sqrt/sum = 0.683/0.683/0.669
K33: balanced/raw all = 0.623/0.623
provisional default = K35; unresolved strong control = K65
```

在相同 K35/K65 checkpoint 上追加 30 epoch 全源 EEG 联合微调后，learned-tempered
仅为 `0.688/0.676`。K35 learned 相对 no-fine mean 为 `-2.59 pp`，95% CI
`[-4.90,-0.14] pp`；fine-tuned fixed mean 更低 `-5.58 pp`，而 learned head
相对该受损 backbone 回收 `+2.99 pp`。结论是当前损失会破坏已有 trial 表征，
不是训练没跑全，也不是算力不足。该 v4 checkpoint 已物理归档；当前 v3 必须重训，
不得修改 metadata 后继续使用。

GTN 已被多轮查看，只能继续作 development。确认值需要新 target block/新采集。

BI+BNCI 的 5 导 common-CAR source 路径已进入匹配训练，而不再只是 loader 能力。
uniform joint hit@2=`0.0967`，相对 BI-only `0.1300` 为 `-3.33 pp`，95% CI
`[-5.90,-1.28] pp`。看到该结果后冻结的 BI 3x/BNCI 1x 探索臂为 `0.1239`：
相对 uniform 恢复 `+2.72 pp`，Holm `p=0.0052`，但相对 BI-only 仍 `-0.61 pp`
且 CI 跨 0。由于 3x/1x 同时增加每 epoch optimizer steps，它不能把恢复纯归因于
域权重。固定 uniform rows/steps、只用非 holdout BI source rows 拟合 input mean/std
后，hit@2=`0.0974`，相对 all-source 仅 `+0.07 pp`，95% CI
`[-0.36,+0.62] pp`；相对 BI-only 仍显著 `-3.26 pp`。因此负迁移不是公共统计主导。
2026-09-02 对归档旧 cache 的只读机制复审进一步发现：两域 target prior 都约
`1/6`，排除 label-prior shift；BI/BNCI 的 `target - nontarget` ERP 在完整 5 导张量
上的余弦为 `-0.0173`，subject-macro 后为 `-0.0787`，P3/Pz/P4 通道余弦分别约
`-0.247/-0.304/-0.220`。BNCI 每人固定 4200 行，BI 每人平均 953 行
（范围 420--1788），所以逐 epoch CE 让 8 个 BNCI participants 获得与 64 个 BI
participants 不成比例的梯度质量。旧 group split 在 seed 20260901/02/03 的 validation
中分别选到 0/1/1 个 BNCI participant；后两次该一人 4200 行接近 validation 行数一半，
模型选择目标随 seed 改变。binary AUC 与 decision hit 同时下降，故问题发生在 trial
表征/选择层，不是候选聚合层。

因此 v3 不直接跳到 GRL/PCGrad/stem。保持唯一行、batch、step 和 seed 不变，预注册：

```text
B0  BI-only，BI selection
J0  joint natural class-weighted epoch mass，BI selection
J1  joint BI/BNCI=80/20 epoch mass，BI selection
J2  joint BI/BNCI=80/20 participant-macro mass，BI selection
```

`J0-B0` 复核联合源净效应，`J1-J0` 隔离域质量，`J2-J1` 隔离统计单元。
只有 J2 仍低于 B0，才测量 matched BI/BNCI gradient cosine 并进入梯度冲突或
dataset-specific stem；不能引用 ERP 余弦直接声称 PCGrad 已被验证。

2026-09-02：该轴的 BNCI-target 方向已按同一设计在 causal-v3 CAR5 正式执行
（295942e；6 域联合源 = 5xBI + BNCI，selection-domain=BNCI，3 seeds x
8 ALS 被试 x 30 未知 decisions，100% 覆盖）。hit@8（chance=1/36）：
`B0 0.3417 / J1 0.2736 / J2 0.2667 / J0 0.1222`，三 seed 臂序一致。
预注册对照：`J0-B0 -21.94 pp`（CI `[-37.50,-6.25]`）、`J1-J0 +15.14 pp`
（CI `[+4.17,+27.36]`）、`J2-J1 -0.69 pp`（CI `[-1.94,+0.97]`）；Holm 后
hit@8 均不显著（n=8 欠功效），但 AUC `J0-B0 -9.62 pp`、`J1-J0 +7.95 pp`
均 `p=0.008` 且 8 个 evidence level 全部同号。J1/J2 仍低于 B0
（`-6.81/-7.50 pp`）。

两方向合成裁决：质量校正确为最大恢复项（两方向一致），统计单元无差异，
但校正后联合源在任何方向都未胜单域。带域质量 `alpha` 的 pooled CE 只能在
各域最优判别面的妥协段上取点；归档 ERP 余弦近正交说明该妥协成本不可由
标量 `alpha` 消除。**J2<B0 的预注册门已触发**：下一动作是 matched
BI/BNCI gradient cosine 诊断（无新训练），若确认冲突则单轴测试
per-domain classifier head（共享 trunk + 域专属头），不足再上 per-domain
stem；BI-target 方向的 v3 J 臂复跑降级为可选项，仅当梯度诊断显示方向
不对称时执行。BI-target v3 的 B0/checkpoint 仍随 Gate 3 缓存重建产生。
该正式运行的 manifest 步骤曾因合同投影缺陷两次失败，修复见 `ab6c8e7`；
数值待 manifest 重建后以归档产物为准（详见
[`research_status_report_20260902.zh.md`](research_status_report_20260902.zh.md)）。

### Gate 2-F：特征域迁移机制（2026-09-02 落地，预注册）

针对审计 F-01（当前没有 feature-domain alignment）实现的可执行机制，
全部为单轴臂，与既有 B0/J0/J1/J2 保持相同 rows、batch、optimizer step、
seed 与 selection-domain：

```text
S1   共享 trunk + per-domain identity-initialized residual adapter
A1   共享 trunk + class-conditional feature mean alignment（无 adapter）
AS1  S1 + A1 同时启用
```

机制合同（[`src/models/adapters.py`](../src/models/adapters.py)）：

- adapter 位置在 MST 拼接特征之后、pooling 之前（保留时空结构，不做
  展平伪卷积）；
- `H' = H + rho*tanh(a)*A(H)`，`A = W_up(ELU(W_down(DWConv(H))))`，
  `a=0` 零初始化使初始行为与 source trunk 严格逐位相等；`|alpha|<rho`
  有界；默认 rho=0.5、bottleneck=4、kernel=9（feature rate 32 Hz 下
  约 281 ms）；
- 无归一化状态：target 前向不估计任何 BN 统计；
- 域词表构造时冻结，未知域 fail-closed；保留槽 `__target__` 只允许
  subject adapter 内层注册；
- A1 的对齐损失是源域间**类条件**均值差的均方（真实标签、无
  pseudo-label），在 adapted 特征上计算，稀疏 cell（<8 行）跳过并记录
  active_terms 诊断。

反例不变量（已进测试，[`tests/test_domain_adapters.py`](../tests/test_domain_adapters.py)）：
identity-at-init 逐位相等、gate 有界、未知域 fail-closed、混合 batch 路由
保序、对齐损失对域重命名不变、`head_kind="adapter"` 内层环结束后 trunk 与
classifier 参数逐位不变（只有 `__target__` 残差变化）、bank checkpoint 的
strict 加载与 bankless 拒载。

裁决规则：任何臂的 latent 对齐距离下降而 target hit@R/AUC 下降，判负迁移
不得晋升；S1/A1/AS1 各自单独与 J1（当前最强联合臂）和 B0 比较，报告
subject-macro hit@R 与 AUC 的配对差与 CI。CLI 入口：
`run_pretrain_supervised.py --adapter-bottleneck N --feature-alignment-weight W`，
目标内层环 `--head adapter`（三个 transfer runner 均已暴露）。

### Gate 3：合法单被试校准

先用 BI2014a cross-decision 协议验证机制，再进入 BrainSync 9 选：

```text
zero-shot
pretrained classifier fine-tune (BN frozen vs adapt)
scratch head
source/target/shrinkage normalization
target QC none/fold-local
fixed epoch budget vs time-heldout selection + full-prefix refit
```

同一 GTN thought digit 的 O5 只作反例/诊断，不参与晋升。

物理归档中的 BI causal-v2 已完成 64 人、3 seeds 匹配比较。zero-shot/source stats 的
subject-macro operational hit@2=`0.1941`；classifier fine + shrinkage=`0.1958`，
full fine + shrinkage=`0.1989`，paired CI 均跨 0。target-prefix normalization
在 classifier/full/linear/MLP16 四种 head 上均下降，scratch linear 相对 zero-shot
显著下降 `4.22 pp`。因此当前默认保持 source classifier + source stats；少量已知
decisions 暂不用于更新网络或替换输入统计。

### Gate 4：无真值目标自适配与决策目标

GTN 可合法研究不使用 thought-number 真标签的目标适配：

- zero-shot prefix 先形成 target posterior；
- pseudo-target/latent-target 适配必须保留 no-adapt 对照；
- 任意 pseudo-target 都能被模型“学会”的反例必须失败；
- 已完成的 source-only 9-candidate all-evidence listwise 联合微调是负结果；后继
  只允许冻结/渐进解冻、显著更低 backbone LR 或梯度冲突控制的单轴消融，不能直接复用；
- 加 cumulative candidate listwise loss 时使用真实累计证据，不伪造同步 round；
- dynamic stopping 阈值只由 source/calibration 冻结，不能在 test suffix 选择。

### Gate 5：成人 BrainSync 90%

冻结所有 recipe 后，在多被试、多 session、多未知数字 decision 上一次性运行。
primary 为 subject-macro `hit@R`，并给 subject-cluster CI、coverage、abstain、刺激成本、
最差 decile、延迟和内存。只有该 Gate 有权产生产品 90% 结论。

## 7. 必备反例

1. candidate label permutation：同时重命名候选和 truth，命中不变；tie 不偏向数字 1；
2. early-window digit identity probe：若非 P300 早窗能识别 digit，检查 glyph/VEP shortcut；
3. pseudo-target control：任意指定数字为正类不应在真实未知 decision 获得同样增益；
4. target-switch：calibration 目标 A/B/...，test 目标必须变化；
5. causal startup：mV DC offset 不得在前几 trial 产生大 PTP transient；未来冲激不影响过去；
6. missing evidence：operational wait 必须增加刺激成本；fixed block 必须记 failure；
7. checkpoint permutation：通道顺序、参考或 preprocessing 改变时加载失败；
8. normalization：suffix 极值不能改变 source/prefix fitted statistics；
9. group objective：一个错误候选极端 logit 应被 candidate loss 明确惩罚；
10. target distribution：报告 uniform `1/9` 与 empirical-majority baseline，最终采集目标平衡。
11. source-risk duplication：复制同一辅助 participant 的 epoch 不得增加其域/participant
    coefficient mass；natural epoch mass 必须严格退化为原 CE。

## 8. 立即执行顺序

1. **闭合 295942e 证据链**：下载云端 96 个 checkpoint、
   `frozen/source_training_295942e9e94d.manifest.json`（含 tar.gz）与
   manifest 失败日志，用修复后代码（`ab6c8e7`）独立执行 manifest/analysis
   argv 重建产物，物理冻结为 `frozen/research_evidence_295942e_*.tar.gz`；
2. 已完成证据已物理压缩归档，活跃树不再包含旧合同 runner/manifest；
3. 从 raw 重建统一 v3 的 BI/GTN cache、common-CAR source 和 checkpoint
   （BNCI target/source/6 域联合 cache 已随 295942e 建成）；
4. v3 继续采用 `full_unfold + K35` 与 candidate mean，K65 保留强对照；
5. 归档 BI 结果不支持 5-decision personalization；新 v3 保留 zero-shot/source stats 基线；
6. 多源域质量/统计单元轴已在两方向裁决：联合不晋升、单元轴关闭；
   下一动作是 matched BI/BNCI gradient cosine 诊断，确认冲突后单轴测试
   per-domain head（不足再 stem）；BI-target v3 J 臂仅在诊断显示方向
   不对称时补跑；
6a. 特征域迁移机制已闭合（Gate 2-F）：在 295942e 同款 6 域联合源上
   预注册执行 S1/A1/AS1 单轴臂（与 J1 同 rows/batch/step/seed/
   selection-domain），主指标 BNCI-target hit@8 与 AUC；负迁移门：
   latent 对齐改善但决策指标下降的臂不得晋升；目标侧用
   `--head adapter` 内层环（只训练 `__target__` 残差）与
   zero_shot/classifier_fine 对照；
6b. 决策聚合分叉（审计 F-03）已消除：`models/decision.py`（旧
   subject-digit 后处理，含无人消费的 z-score 输出）删除，三套路径
   （subject-digit / generic candidate / row-column）统一到
   [`src/models/candidate_evidence.py`](../src/models/candidate_evidence.py)
   的单一聚合核心，语义含 fail-closed 词表与显式
   `missing_candidate_policy`（exclude=空候选仅不获胜；
   abstain=空候选判 incomplete miss，hit@R 分母语义）；冻结归档
   `source_before_feature_alignment_aa4b6af` 保留统一前行为；
7. decision-aligned 全参数 30-epoch recipe 已否决；后继只研究保护 backbone 的
   分阶段单轴策略、无真值 adaptation 与 target-switch personalization；
8. BI2014a v3 缓存就绪后推进 Gate 3 校准开放轴（BN adapt、time-heldout
   selection + full-prefix refit、fold-local target QC、shrinkage）；
9. 使用统一 v3 causal multi-session/target-switch 入口重新采集成人 BrainSync 数据；
10. 不把 O5 当未知数字校准。

## 9. 本轮外部依据与边界

2026-09-02 通过 OpenAlex/Crossref 公共 API 定向复核，不是系统综述。检索支持以下
工程判断，但不替代本项目实测：Ben-David et al. 的 domain-adaptation bound
（DOI `10.1007/s10994-009-5152-4`）要求同时考虑域差异与不可共同最优误差；
Wang et al. 的 negative-transfer 定义与 source 选择（DOI
`10.1109/CVPR.2019.01155`）支持必须保留 no-transfer 强基线；P300 cross-dataset
signal alignment（DOI `10.1088/1741-2552/ad430d`）支持显式信号对齐，但不证明
common-CAR 或联合训练必然增益；negative-transfer survey（DOI
`10.1109/JAS.2022.106004`）把 domain similarity、safe transfer 和 mitigation 分开。
PCGrad（arXiv `2001.06782`）只作为梯度冲突候选，不在未测真实 gradient 前实施。
Semantic Scholar 对部分 DOI 返回 429；P300 signal-alignment 本轮只核到可复核元数据，
不得写成完成摘要或全文系统审阅。
