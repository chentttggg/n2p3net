# N2P3-Net 两条路线与 strict-past 分支边界

> 本文档是路线选择的操作入口。模型细节以 `blueprint.md` 为准，代码默认值以
> `src/train/recipe.py` 为准。用户口中的 `strictpass` 在代码和文档中统一写作
> `strict-past`。

## 1. 先选路线

当前只有一条正式路线，另一条是显式登记的研究路线。两者共享数据契约、PCW 主干和
GTN 重复证据协议，但不共享最终输出的默认语义。

| 项目 | 正式/生产路线 | strict-past 研究路线 |
|---|---|---|
| recipe | `neural_ride_v11_pcw_fail_closed` | `neural_ride_v11_strict_past_research` |
| 代码对象 | `NEURAL_RIDE_V11` | `NEURAL_RIDE_V11_STRICT_PAST_RESEARCH` |
| GTN 开关 | `--lambda-innovation 0`（默认） | `--lambda-innovation 1` 或其他正数 |
| 主神经判别量 | `logit_pcw` | 仍是 `logit_pcw` |
| 额外机制 | 无 strict-past likelihood | 两个类别假设各自的 strict-past 条件密度 |
| 最终输出 | `final = PCW` | 只有通过 audit、validation cross-fit 和 outer claim gate 才能融合；否则仍为 PCW |
| 当前地位 | 唯一正式默认 | opt-in 研究候选，当前未通过增量决策价值门槛 |

路线切换由 `--lambda-innovation` 唯一决定：为零走正式路线，大于零走 strict-past
研究 recipe。实验记录必须同时保存 resolved `recipe`、`lambda_innovation`、
`use_innovation_likelihood` 和最终是否实际启用 fusion；不能只看命令行标签或 run name。

`--lambda-recon`、`--lambda-morphology-l0` 等参数不构成第三条路线。它们只能作为单独
登记的解释性消融；ERP decoder 的输出不得进入正式分类 logit，也不能因为开启 decoder
就把一次运行称为 strict-past。

## 2. 正式/生产路线

正式路线解决的是离线 epoch-level 判别问题：PCW 保留完整时间轴，通过 N2/P3a/P3b
结构化窗口产生唯一神经分类 logit。它可以使用整个已观测 epoch，不承担 strict-past
逐样本概率预测的声明。

正式路线的硬边界：

- `final = PCW`；没有 global classifier、residual classifier 或 sensor-space bypass；
- 不启用 strict-past likelihood、低秩条件协方差或 prequential fusion；
- `component_decoder=False` 是默认正式配方；解释性 decoder 不能回写 PCW 分类路径；
- 结果按 GTN 的 ITT、`exact_llr@3`、coverage、校准和被试级配对协议报告；
- 正式结论不能借用 strict-past 分支的 NLL 改善来证明分类增益。

默认开发命令可以直接运行：

```powershell
.\.venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
  --lambda-innovation 0 --run-dir tmp\production
```

## 3. strict-past 研究路线

strict-past 不是旧 residual 路线的保留名，也不是把整个 PCW 分类器改成在线模型。
它是一个与 PCW 参数解耦的、按两个类别假设分别建模的条件密度分支：

1. 在 outer train 的 optimization subjects 上拟合 fold-local 类别均值、VAR(32) 和协方差 profile；
2. 对同一 trial 同时维护 target/non-target 两条 residual history，不能先猜类别再做 subtraction；
3. 使用正滞后 VAR、trial-adaptive AR(1) 和内部右移一采样点的因果 TCN；
4. 在 `D + U U^T` 协方差上计算严格按 observed scalar 归一化的 NLL，以及未中心化的 time-summed LLR；
5. 先由 untouched audit subjects 选择候选，再由 validation-subject LOSO cross-fit 学习非负 fusion 系数；
6. 任一 gate 失败都 fail closed，系数置零，最终输出退回 PCW。

strict-past 的实际声明是 `preprocessed_epoch_strict_past`：在完整 trial baseline
标准化之后，时刻 `t` 的密度参数只依赖 `x_<t`。上游连续滤波/重采样仍可能是非因果的，
所以不能把它写成 raw-streaming latency guarantee；PCW epoch classifier 也不受这个约束。

strict-past 研究运行示例：

```powershell
.\.venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
  --lambda-innovation 1 --audit-subjects 4 `
  --run-dir tmp\strict-past-research
```

这条路线当前不能直接替代正式路线。已有开发/锁定结果显示，后续锁定区间的 density
eligible 和 active fusion 均为 `0/5`，outer claim gate 未通过。因此报告中必须把
`PCW`、`prequential contribution` 和 `final` 分列，并明确 `final == PCW` 的原因。

## 4. 两条路线都不能带回的旧方法

以下内容不再是可选兼容路径：residual classifier、先分类再 ERP subtraction、
`alpha/ramp` 门控、identifiability hold，以及用 learned ERP waveform decoder 反向
制造分类证据。历史原因和实验结果只保留在 `issue2_strict_past_rewrite.md`；现行实现、
CLI 和指导文档不应重新提供这些开关。

## 5. 研究分支的升级与删除规则

strict-past 只有在完整的 locked outer protocol 中证明增量决策价值后，才可以申请进入
正式路线：至少 5 个完整 outer folds，audit 与 validation 分区保持独立，fusion 在严格
多数 fold 实际启用，并在 AUC、Brier 和预注册主终点上按被试配对改善。通过单元测试、
单折 NLL 或一次开发运行都不够。

新方法如果被证明确实正确、简单且替代关系成立，就大胆删除旧方法；不要为了保留而保留。
旧实现若只具有历史复现价值，移入 `archives/` 并从现行 import、CLI、默认 recipe、测试
和指导文档中移除。只有仍有独立科学问题或明确复现义务的内容，才作为带名称、带 gate 的
显式研究消融保留。
