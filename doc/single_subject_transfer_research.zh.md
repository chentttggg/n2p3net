# 单被试未知数字解码 90% 协议

日期：2026-09-01
状态：当前执行指导；总体状态与晋升由 `research_program.zh.md` 裁决。

## 1. 目标

单被试 90% 定义为：同一成人被试完成固定绝对预算的已知目标 calibration
decisions 后，在 later independent decisions/session 中猜此前未知的 9 选数字，
subject-macro `hit@R >= 0.90`。R 是预声明的最大刺激/时间预算，不默认等于 5。
失败、缺候选、tie 和 abstain 都进入分母。

这不是：

- LOSO trial BACC=0.90；
- 一个人一次 decision 的 cohort hit；
- 同一固定 target 内 prefix 标签训练后再猜 suffix；
- 由 trial AUC 高斯假设外推的 hit。

## 2. 三种不同估计量

### Z0：session-start zero-calibration

```text
source checkpoint 显式排除 target subject
target prefix used for fit = 0
test = earliest operational R valid evidence per candidate
```

GTN 可以执行 Z0。必须报告 `hit@all_balanced`、raw `hit@all` 和完整 `hit@R`
曲线，以及收齐各预算实际等待的 stimuli 和秒数。`hit@5` 仅为可比切片，不是标准。

### Z5：matched-time zero-target-fit

丢弃/等待前 M 个证据但不使用其标签，再评估 later suffix。它用于与适配 arm
共享时间位置，不能冒充 session-start zero calibration。增大 M 不增加训练数据；
若读取 GTN prefix 标签，则只是增强同 thought digit 的 oracle。

### S：监督校准到新 decision

```text
calibration = earlier known-target decisions
test        = later target-changing unknown decisions
```

BI2014a 可用 6x6 character decisions 验证这一机制；最终由 BrainSync 9 选数据裁决。
GTN 每人只有一个 thought digit，因此同 selection 的 S 模式只叫 oracle proxy。

## 3. 数据与时间隔离

每个 calibration/test 边界必须满足：

```text
max(calibration evidence_available_time)
    < min(test epoch_start_time)
```

重叠 trial 进入 embargo。online 路径使用 causal IIR steady-state initialization。
可另跑 offline split-local zero-phase arm，但不能先对整段连续记录 zero-phase 再切分。

GTN 是随机刺激流，不存在同步 1--9 round。默认 operational M/R 表示：

1. 等待每候选 M 个有效 evidence；
2. 越过全局 raw-time boundary；
3. 再等待每候选 R 个新有效 evidence；
4. 保存每候选原始 occurrence index、总 stimuli 和秒数。

## 4. 训练模式

| mode | target labels | 当前用途 |
|---|---:|---|
| zero_shot | 0 | GTN 合法主线 |
| classifier_fine | known calibration decisions | 首选监督适配候选 |
| linear/mlp scratch head | known calibration decisions | 负对照/容量对照 |
| full_fine | known calibration decisions | 高风险 arm，BN frozen/adapt 分开 |
| pseudo/latent target | 0 | GTN 无真值自适配研究 |

GTN source-only 决策目标实验已给出一个直接反例：从 target-block-excluded v4
checkpoint 出发，30 epoch 全参数 trial CE + 9-candidate listwise 联合微调并未提高
zero-shot all-evidence。K35 的 learned-tempered 为 `0.6884`，低于不微调
candidate mean 的 `0.7143`；fine-tuned backbone 即使用 fixed mean 也只有 `0.6585`。
因此 `full_fine` 继续保留为高风险消融，不再默认假设“算力更多/训练更久会更准”。

45 trial 随机重训 head 的旧失败不能证明所有校准无效。正确比较至少包括：

- 保留 source classifier；
- 固定 5 epoch / 低学习率 classifier fine；
- source stats / target-prefix stats / shrinkage；
- BN running stats frozen / adapt；
- target QC none / prefix-fit fold-local。

固定预算是准确率主臂：epoch 数在 source/development subjects 冻结，目标人全部
prefix 用于训练。`target_time_split` 是敏感性臂：真实时间 train/validation + embargo
选择 epoch，随后从同一初始化在全部 prefix refit。

## 5. 校准与聚合

weighted CE 输出转 LLR：

```text
LLR = (z - log(pos_weight) - logit(train_prior)) / T, T>0
```

不得让 tiny validation 的负 Platt slope 反转排序。固定完整 R 时，公共正仿射不改变
argmax。`trim0.2` 按排序固定裁 `floor(0.2*n)` 个两端值；R<5 不裁。

候选最高分 tie 一律 abstain/miss，不按数字编码破 tie。目标数字不平衡时同时报告
uniform `1/9` 和 empirical-majority baseline；正式采集目标必须随机平衡。

## 6. 必须保存

- target/source cache SHA 与 checkpoint SHA；
- ordered channels、reference、完整 preprocessing、causal initial state；
- source/holdout subject keys；
- 每 trial 的 logit/LLR、label、candidate、selection、时间、valid rank；
- calibration/test/embargo mask；
- QC 的 target/nontarget 分层 drop/mask；
- 每 decision hit@R、tie、abstain、stimuli 和秒数；
- requested/eligible/failed cohort ledger；
- seed、版本、环境、参数量、延迟和内存。

## 7. 当前执行顺序

1. GTN steady-state causal 2x2、Z0/Z5 与 source-QC 消融已经完成并审计；
2. 当前工程主线固定为 full-unfold+K35、0.1 Hz/1200 ms、source QC100、
   不做额外联合微调；all-evidence 使用全部 trial 的 candidate mean，K65 保留对照；
3. 使用已重建 BI candidate-v2 和冻结 block checkpoint，跑 early-known -> later-unknown calibration；
4. 只在独立 development decisions 选择 personalization/normalization/decision objective；
5. 冻结后采集成人 BrainSync 多 target-switch decisions；untouched BrainSync 才能裁决 90%。
