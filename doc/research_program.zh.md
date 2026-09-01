# N2P3-Net 科研总纲：未知数字、多决策与 90% 目标

日期：2026-09-01
状态：living research guide
适用范围：oddball/P300 单试次检测、多试次候选聚合、跨被试预训练和单被试校准。
阶段结果摘要见 [`research_status_report_20260831.zh.md`](research_status_report_20260831.zh.md)。

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

工程入口现已支持 causal steady-state、单 session 多 block/selection、多 session
`started_utc` 排序、target-switch split/runner，以及显式公共通道 CAR 域适配。当前
默认固定为 128 Hz、0.1--30 Hz、`[-200,1200) ms`、179 samples；没有 2/800
降级选项。当前
本机 4 个历史 sessions 分别缺 recording、recording_error 或仍为
`analysis_ready=false`/target pending，均不能进入准确率分析；必须重新采集。

## 3. 2026-08-31 证据清算

| 证据 | 当前状态 | 允许结论 |
|---|---|---|
| BI2014a 128 Hz zero-phase LOSO assets | 已进入物理压缩归档 | 只作架构开发；不可作为当前 checkpoint |
| BI2014a causal-v2 cross-decision | 物理归档：64 人、12 checkpoints、13 arms x 3 seeds | 只保留机制结论；v3 必须重建重训 |
| BI2014a+BNCI2014_008 common-CAR5 source | 物理归档：`0.1300/0.0967/0.1239/0.0974` hit@2 | 负迁移机制证据；旧 128-sample cache 不可续用 |
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
| normalization | source stats / target-prefix stats / shrinkage | BI 已完成：source≈shrinkage，target-prefix 在所有 head 上下降 |
| multi-source normalization | all-source stats / target-dataset source stats | 已完成且等效：BI-source stats 相对 all-source `+0.07 pp`，CI 跨 0 |
| target QC | none / prefix-fit fold-local | source QC100 已冻结；target QC 待独立 decision 消融 |
| epoch budget | source/fixed budget / real-time target holdout + full-prefix refit | 代码闭合，性能待独立 decision |
| adaptation | zero-shot / pretrained-classifier fine / scratch head / full fine | BI 已完成：fine 无可靠增益，scratch 更差；BrainSync 待真实数据 |
| BatchNorm | frozen running stats / target adapt | 待合法 cross-decision |
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
下一臂必须保持唯一行、batch 和 step 不变，用归一化 per-row CE weight 实现约
80/20 域梯度质量，之后才判断是否需要 gradient conflict/stem。

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

## 8. 立即执行顺序

1. 已完成证据已物理压缩归档，活跃树不再包含旧合同 runner/manifest；
2. 从 raw 重建统一 v3 的 BI/BNCI/GTN cache、common-CAR source 和 checkpoint；
3. v3 继续采用 `full_unfold + K35` 与 candidate mean，K65 保留强对照；
4. 归档 BI 结果不支持 5-decision personalization；新 v3 保留 zero-shot/source stats 基线；
5. 归档 BI+BNCI 结果提示负迁移；v3 下一次固定 rows/steps，只改 normalized per-row domain loss weight；
6. decision-aligned 全参数 30-epoch recipe 已否决；后继只研究保护 backbone 的
   分阶段单轴策略、无真值 adaptation 与 target-switch personalization；
7. 使用统一 v3 causal multi-session/target-switch 入口重新采集成人 BrainSync 数据；
8. 不把 O5 当未知数字校准。
