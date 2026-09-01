# N2P3-Net 科研状态汇报

_更新至 2026-09-01；GTN development、单被试迁移链与 90% 产品目标阶段汇报_

---

## 结论摘要

- 当前工程主线为 causal-v3：`full_unfold + K35`、128 Hz、0.1--30 Hz、
  `[-200,1200) ms`、179 samples、forward steady-state、source QC 100 uV。
- 物理归档中准确率最高的 all-evidence 配方是不额外微调 v4 checkpoint，并对每个候选
  使用全部 trial 的 mean：K35 `71.43%`、K65 `68.30%`。K35 保持临时默认，
  K65 保留强对照，K33 退出主线。
- 24 个端到端联合微调 checkpoint 已按 30 epoch 全量训练完成。K35/K65 的
  learned-tempered 为 `68.84%/67.62%`，没有超过各自 no-fine mean；该 recipe 不采用。
- `hit@5` 不是通用标准。当前 all-evidence 开发默认使用全部 245 人和全部 trial 的
  candidate mean；balanced truncation、sqrt-count、raw sum 与完整 hit@R/cost
  作为兼容和机制对照。
- 物理归档中的 BI 64 人 candidate-v2、12 个 source checkpoints、13 arms x 3 seeds
  cross-decision 比较已完成。zero-shot/source stats 为 `19.41%` subject-macro hit@2；
  classifier/full fine 没有可靠增益，target-prefix normalization 明显下降。
- 物理归档中的 5 导 common-CAR BI+BNCI 实验显示 uniform joint 相对 BI-only
  为 `-3.33 pp`；后验 BI 3x/BNCI 1x 暴露恢复 `+2.72 pp`，但仍未胜 BI-only
  (`-0.61 pp`, CI 跨 0)。固定 uniform rows/steps 后改用 BI-source input stats 仅
  `+0.07 pp`，等同无变化。因此不采用朴素联合源，也不再归因于 mean/std 污染；
  下一轴隔离固定 step 下的域梯度权重。

## 实测结果

以下数值均来自 `d1db8e4` 物理冻结归档，用于保留科研结论，不代表当前 v3
cache/checkpoint 已生成。

### 信号与 source QC

以下为旧主干 `ms_flatten + K65` 的固定 `R=5` development 对照，用于冻结信号合同：

| 信号 arm | coverage | conditional hit@5 | operational hit@5 | AUC |
|---|---:|---:|---:|---:|
| 0.1 Hz / 800 ms | 230/245 | 0.370 | 0.347 | 0.657 |
| **0.1 Hz / 1200 ms** | **230/245** | **0.578** | **0.543** | **0.709** |
| 0.5 Hz / 800 ms | 230/245 | 0.300 | 0.282 | 0.632 |
| 0.5 Hz / 1200 ms | 230/245 | 0.522 | 0.490 | 0.674 |

source QC100 相对 no-QC 的 operational hit@5 提升 `+0.118`，95% CI
`[+0.053,+0.184]`，因此 source QC100 已冻结；target-prefix QC 保持关闭。

### BI2014a 合法跨 decision 校准

64 人 causal-v2 cache 使用前 5 个已知 character decisions 校准，后续未知
decisions 取每 row/column 前 2 repetitions；四个 target-block-excluded K35
full-unfold checkpoint、3 source/adaptation seeds。requested decisions=`1416`，
eligible=`964`，失败 `452` 全部保留在 operational 分母。

| Arm | subject-macro hit@2 |
|---|---:|
| zero-shot + source stats | **0.1941** |
| classifier fine + source / shrinkage | 0.1952 / 0.1958 |
| full fine + source / shrinkage | **0.1994** / 0.1989 |
| linear scratch + source / shrinkage | 0.1524 / 0.1518 |
| MLP16 scratch + source / shrinkage | 0.1831 / 0.1774 |
| classifier / full / linear / MLP16 + target stats | 0.1569 / 0.1590 / 0.1148 / 0.1294 |

- primary `classifier_fine_shrinkage - zero_shot_source`=`+0.18 pp`，95% CI
  `[-0.78,+1.04] pp`，Holm `p=1.0`；不支持 5-decision classifier personalization。
- `full_fine_shrinkage - zero_shot_source`=`+0.45 pp`，95% CI
  `[-0.52,+1.33] pp`，Holm `p=1.0`；同样不支持晋升。
- target stats 相对 source stats 在 classifier/full 上约 `-3.83/-4.04 pp`；
  full fine 的 Holm `p=0.043`。目标 prefix 统计不是当前默认。
- linear scratch shrinkage 相对 zero-shot `-4.22 pp`，95% CI
  `[-7.01,-1.52] pp`，Holm `p=0.043`。保留 source classifier 是必要基线。

该任务是 6x6 character (`chance=1/36`)，只能验证跨 decision 校准机制，不能与
BrainSync 9-choice 的 90% 指标直接换算。

### BI+BNCI 多数据集联合源

两臂父实验和一个后验探索臂都使用相同 BI target blocks、5 导
`CZ,P3,PZ,P4,OZ`、同子集 CAR、causal-v2、`full_unfold + K35`、100 epoch 上限、
3 seeds 和 operational hit@2。父实验在看到结果前冻结；3x/1x 是看到 uniform
负迁移后才冻结的新臂，不得写成确认性结果。

| Source arm | subject-macro hit@2 | 相对 BI-only |
|---|---:|---:|
| BI-only CAR5 | **0.1300** | reference |
| BI+BNCI uniform CAR5 | 0.0967 | `-3.33 pp` |
| BI+BNCI, BI 3x / BNCI 1x | 0.1239 | `-0.61 pp` |

- uniform joint - BI-only：95% CI `[-5.90,-1.28] pp`，sign-flip
  `p=0.00108`，三个 seed 均为负，确认当前朴素拼接产生负迁移；
- 3x/1x - uniform joint：`+2.72 pp`，95% CI `[+1.01,+4.68] pp`，Holm
  `p=0.0052`，三个 seed 均为正；
- 3x/1x - BI-only：95% CI `[-2.04,+0.80] pp`，`p=0.414`，没有联合源净增益；
- 每个 3x/1x fold 保留约 79.3k 唯一行，展开为约 170.8k optimizer rows；BI
  暴露约 80.3%。因此该臂同时改变域比例和每 epoch step 数，不能把恢复量纯归因于权重。

固定 uniform 的约 79.3k optimizer rows 和所有训练设置，只让 mean/std 由约 45.7k
非 holdout BI source rows 拟合后，hit@2=`0.0974`；相对 all-source stats 仅
`+0.07 pp`，95% CI `[-0.36,+0.62] pp`，`p=0.837`；相对 BI-only 仍 `-3.26 pp`，
95% CI `[-5.48,-1.39] pp`，Holm `p=0.00148`。因此公共输入统计不是负迁移主因。

当前结论是：BNCI 数据本身并非被证明“必然有害”，但其 uniform 分类梯度会损伤 BI
目标；3x/1x 的恢复更符合“增加 BI 梯度质量/步数”而不是 normalization 修复。
继续扫描 repeat 或 stats 没有价值。下一次应保持唯一行数、batch 和 step 不变，给
BI/BNCI per-row CE 设置归一化权重，使总梯度暴露约 80/20，以隔离域梯度比例。

### Full-unfold 核长

三核仅改变 ST kernel；其余 cache、block、训练、refit、QC、校准、Z0 和分母完全一致。

| Kernel | balanced-all | raw-all | AUC | hit@5 | hit@8 |
|---|---:|---:|---:|---:|---:|
| **K35** | **0.6694** | **0.6871** | **0.6938** | **0.4571** | **0.4490** |
| K65 | 0.6544 | 0.6694 | 0.6862 | 0.4340 | 0.4177 |
| K33 | 0.6231 | 0.6231 | 0.6807 | 0.4095 | 0.4027 |

- K35-K33：`+0.0463`，95% CI `[+0.0122,+0.0816]`，Holm `p=0.0346`，
  三 seed 差异均为正。
- K35-K65：`+0.0150`，95% CI `[-0.0122,+0.0435]`，Holm `p=0.3483`；
  seed 差异为 `+0.0286/-0.0653/+0.0816`。
- 严格唯一赢家仍未成立；准确率和参数量共同支持 K35 作为临时工程默认。

### All-evidence 计数校正与端到端训练

冻结 v4 trial ledger 上，单位 trial 权重的统一公式为：

```text
score_d = candidate_mean_d * effective_count_d^beta
```

| Kernel | no-fine mean (`beta=0`) | sqrt-count (`beta=0.5`) | sum (`beta=1`) |
|---|---:|---:|---:|
| **K35** | **0.7143** | 0.7034 | 0.6871 |
| K65 | 0.6830 | 0.6830 | 0.6694 |

mean 使用全部 trial，同时消除随机 candidate occurrence 数量对 argmax 的直接偏置。
K35-K65 mean 为 `+3.13 pp`，95% CI `[+0.41,+5.85] pp`，但 seed 差异
`+4.90/-4.90/+9.39 pp`，八个报告 contrasts 的 Holm `p=0.191`；因此仍不是
确认性唯一赢家。

端到端实验为 `2 kernels x 3 seeds x 4 target blocks = 24` 个 checkpoint；从 v4
target-block-excluded checkpoint 初始化，30 epoch，每 epoch 使用 QC100 后全部
`17,733--18,368` 条 source EEG。每 block 有 `166--169` 个完整九候选组进入
listwise CE；`14--16` 个缺候选组的剩余 EEG 仍进入 trial CE，不删 trial。

| Kernel | fine fixed mean | fine learned-tempered | no-fine mean |
|---|---:|---:|---:|
| K35 | 0.6585 | **0.6884** | **0.7143** |
| K65 | 0.6653 | **0.6762** | **0.6830** |

- K35 learned 相对同一 fine-tuned backbone 的 fixed mean：`+2.99 pp`，95% CI
  `[+1.09,+4.90] pp`，Holm `p=0.0185`。决策头本身有回收作用。
- K35 fine fixed mean 相对 no-fine mean：`-5.58 pp`，95% CI
  `[-7.76,-3.54] pp`，Holm `p<0.001`。主要损失来自 backbone 联合微调。
- K35 fine learned 相对 no-fine mean：`-2.59 pp`，95% CI
  `[-4.90,-0.14] pp`；八 contrasts Holm 后不显著，但点估计和区间均不支持晋升。
- fine learned 的 K35-K65 为 `+1.22 pp`，95% CI `[-1.09,+3.54] pp`，
  seed 差异 `+0.82/-7.76/+10.61 pp`。核长仍未严格分离。

因此这不是“训练太快所以没跑全”：RTX 5090 上 8 路并发使单臂 30 epoch 仅需
约 `26--61 s`，但每 epoch 的全行覆盖、optimizer step 和 backbone/classifier
非零参数改变量均写入产物。负结果来自当前目标/优化 recipe，不是训练缺失。

### 证据成本

| Endpoint | 有效 trials | scheduled stimuli | elapsed time |
|---|---:|---:|---:|
| balanced-all | 145.51 | 195.07 | 295.63 s |
| raw-all | 205.40 | 205.40 | 311.22 s |
| R=5 | 45 | 78.74 | 118.62 s |
| R=8 | 72 | 111.88 | 168.67 s |

`R=5` coverage 为 `230/245`，`R=8` 为 `209/245`。固定 R 的 operational accuracy
同时受模型准确率和预算覆盖影响；all-evidence 更接近当前完整 session 的准确率上限。

## 被推翻与保留的结论

| 旧判断 | 当前裁决 |
|---|---|
| zero-state forward IIR 可用于 prefix/suffix | 推翻；startup transient 使旧 causal 排名作废 |
| 同一 GTN selection 的 labelled prefix 是个人校准 | 推翻；只能称 O5 oracle proxy |
| K33 应替代 K35 | 推翻；K35 三 seed 稳定胜 K33 |
| `hit@5` 是自然主指标 | 推翻；它只是固定预算切片 |
| tiny Platt 可直接校准候选排序 | 推翻；负 slope 可反转排序，改用解析 offset + 正温度 |
| full-unfold 保留完整时间坐标 | 保留并采用；额外 MLP/二阶 readout 不晋升 |
| K35 是唯一确认核长 | 不成立；K65 仍未被统计分离 |
| all-evidence 必须 balanced 截断或直接 sum | 推翻；当前使用全部 trial 的 candidate mean 更高 |
| 全参数 listwise 联合训练应因目标更对齐而默认提升 | 推翻；30-epoch 全量实验对 K35 产生明显 backbone 负迁移 |

## 代码链闭环

- causal IIR 使用 `steady_state_first_sample`，旧 cache 版本被拒绝。
- checkpoint 绑定完整 architecture、通道顺序、参考、预处理、cache SHA、source/holdout、
  输入统计与 classifier supervision。
- source inner epoch 选择后在全部合法 source rows refit。
- BI 使用 raw Event 100/104 和 target-pair 变化恢复 selection；前 K 个 later decisions
  先冻结再判 eligibility，失败进入 subject 和 decision 分母。
- runner 支持 `--test-reps all`，输出 balanced-all、raw-all、完整 hit@R 与真实成本。
- 端到端 runner 强制每 epoch 全部合法 source EEG、subject/group 不拆 batch，保存
  trial/listwise loss、参数改变量、峰值显存和 245 人 all-evidence 结果。
- 软件结构默认为 `full_unfold + K35`；旧 v4 checkpoint 已压缩隔离，v3 尚待重训。
- 项目 `.venv` 最终验证：活跃 suite `343 passed`；6 个已完成实验专用 test
  modules 随 runner 进入物理归档。Ruff、compileall、uv lock 与 `git diff --check` 通过。

## 证据边界

- GTN 是儿童 3 导、每人一个固定 thought digit 和一次 selection 的 development cohort。
- 当前 subject bootstrap/sign-flip 只解释固定 3 seeds/4 blocks/checkpoints 下的受试者差异，
  不覆盖训练随机性的确认性推断。
- 本轮完成的是 K35/K65 同一 full-unfold 链路；没有做 `full_unfold + K35` 对
  `ms_flatten + K65` 的匹配多 seed
  pooling 比较；full-unfold 的采用是机制与工程决定，不是 GTN 产品冠军声明。
- 已有 BI candidate-v2 development 性能，但没有成人 BrainSync 多 target-switch 数据；因此不能声称
  “单人长期 90%”或“校准后未知数字 90%”。
- BrainSync causal multi-session loader、target-switch runner 和 common-channel CAR
  已通过 synthetic 反例；这属于工程验收，不是实际被试性能。现有 4 个 sessions
  均非 analysis-ready。
- 冻结材料只存在 `frozen/*.tar.gz`；活跃树没有旧 runner、manifest 或 loose evidence。

## 下一轮科研优先级

1. **重建统一 source**：从 raw 重建 causal-v3 BI/BNCI/GTN cache、common-CAR 和
   target-excluded K35/K65 checkpoint；旧 128-sample 对象不得改名复用。
2. **BrainSync 真实数据**：入口已支持 v3 causal multi-session/multi-decision 和
   target-switch runner；现有 4 个 session 均非 analysis-ready，需重新采集。
3. **保护已有表征**：当前强联合微调已否决；下一轮只做冻结 backbone、渐进解冻、
   更低 backbone LR 或梯度冲突控制的单轴比较。N200/P300 多窗辅助目标须单独归因。
4. **多源负迁移定位**：归档中 all-source 与 BI-source stats 等效；v3 下一步保持 uniform
   唯一行、batch 和 step 不变，用归一化 per-row CE weight 实现约 80/20 域梯度质量。
   若仍未胜 BI-only，再停止简单混合并进入 gradient conflict/stem 机制。
5. **证据效率**：在同一预测 ledger 上研究 latency tolerance、correlation-aware evidence
   和 dynamic stopping，以 hit-all/R/cost 曲线裁决，不再增加核长搜索。

## 证据入口

- [权威科研总纲](research_program.zh.md)
- [物理冻结归档 manifest](../frozen/research_evidence_through_20260901-d1db8e4.manifest.json)
- `../frozen/research_evidence_through_20260901-d1db8e4.tar.gz`
