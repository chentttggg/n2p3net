# 单被试 9 选数字解码 85% 研究预注册（含迁移训练）

日期：2026-08-28
状态：research proposal / preregistration
依赖：`doc/input_contract_math.zh.md`、`doc/ablation_20260828.zh.md`、
`doc/prior_free_unfolding.zh.md`

## 0. 目标定义（必须写死，否则无法判定成败）

“单被试 85%”不是 LOSO 单试次 binary BACC。正式目标为：

> 对同一个人的多次 oddball 数字刺激，训练数据只使用该人更早的试验；
> 测试数据使用该人更晚的试验；在固定重复次数 `R` 下，
> 9 选数字命中率 `hit@R >= 0.85`。

同层辅助指标：

| 指标 | 建议门槛 | 说明 |
|---|---|---|
| 单被试 binary trial AUC | >= 0.78 | 预测 9 选能力的前提 |
| `hit@8` | >= 0.85 | GTN 中位数每数字约 16 次，留 8 次测试 |
| `hit@5` | 仅报告 | 用于评估快系统 |
| calibration trials | <= 60% prefix | 允许迁移预训练减少校准量 |

若把目标误写成 LOSO 单试次 BACC=0.85，会在当前证据下先验失败：
本仓库 128 Hz 合同的最强 LOSO 结果是 EEGNet AUC=0.7396 / BACC=0.6754
（`doc/ablation_20260828.zh.md`）；外部 LOSO P300 对照同样落在
AUROC 0.72–0.78 区间（ERP-XTTN, ERP CORE P300，见本文件第 4 节）。

## 1. 可行性：由 binary AUC 推导 hit@R（本项目自己的判断）

假设 binary logit 近似正态、各试验近似独立、9 个候选数字中 1 个 target、
8 个 nontarget；target logit 均值为 `mu_t`，nontarget 均值为 `mu_n`，公共标准差
`sigma`。9 选聚合为每个数字在测试窗内累加 `R` 个 logit：

```text
d' = sqrt(2) * Phi^{-1}(AUC)
Z_target = R*mu_t + sqrt(R)*sigma*e_t
Z_other  = R*mu_n + sqrt(R)*sigma*e_k,  k=1..8
P(hit)   = P(Z_target > max_k Z_other)
         = P(e_t - max_k e_k > -sqrt(R) * d')
```

用这一模型（本项目自己的统计推演，不是文献数字）：

| binary AUC | hit@5 | hit@7 | hit@8 | hit@10 |
|---|---:|---:|---:|---:|
| 0.75 | 0.729 | 0.827 | 0.861 | 0.912 |
| 0.78 | 0.806 | 0.894 | 0.921 | 0.957 |
| 0.80 | 0.854 | 0.928 | 0.950 | 0.976 |
| 0.85 | 0.942 | 0.981 | 0.990 | 0.996 |

判断：

1. 85% 单被试数字命中不需要 binary AUC=0.85；`AUC>=0.78` 且 `R=8` 即可
   跨过 0.92 的模型上界，留出非平稳/校准漂移的余量。
2. 因此研究优先级是：**先让同人 prefix 训练的 binary AUC 稳定到 0.78–0.80，
   再用 `R=8` 决策**；不要先堆 9 选端到端网络。
3. 上述独立性假设会被试次间自相关、注意漂移和校准误差破坏，所以实际
   hit@R 必须用留出 suffix 实测，不能用此表替代实验。

## 2. 为什么单被试协议必须重新设计

### 2.1 当前代码不能直接做该协议

- `run_eeg_loso.py` 只有 LOSO，没有 per-subject prefix/suffix runner。
- `within_subject_folds` 现在要求完整 acquisition group；GTN 每个 subject 只有
  一个 group（245 groups / 242 subjects），因此无法按 run/block 分组。
- 主线滤波是 `phase=zero` 的非因果 IIR。把同一连续记录切成 train/test，
  train epoch 会含有 test 时期的滤波响应，形成时间泄漏。
- BI2014a 通用 MOABB 元数据未暴露 9 个 source block；GTN 的 candidate chain
  和 repetition index 才是可执行单被试数字协议的来源。

### 2.2 必须新增的协议

对每个 GTN/BrainSync subject：

1. 按 `event_timeline.onset_samples` 排序，按 repetition index 构造 prefix/suffix；
2. 不允许随机 trial 划分，不允许同一 repetition 同时出现在 train/test；
3. train = 前 `M` 个 repetition 的所有数字试验，test = 后 `R` 个 repetition；
4. binary 训练阶段的内层 early-stop/calibration 也用更早的 sub-prefix 分组，
   不用未来试验；
5. 9 选决策在 test suffix 内按数字累加 calibrated logit；
6. 对同一 raw 连续记录，单被试协议必须使用因果滤波
   （`filter_phase=forward` 或等效 FIR），并在 provenance 标记
   `online_causal=True`；零相位版本只能用于 LOSO 离线诊断。

### 2.3 GTN 当前数据量（旧 256 Hz cache 的直接统计）

- 242 subjects；每 subject 每数字中位数 16 trials；target trials 中位数 17。
- 建议主协议 `M=8, R=8`：用每个数字前 8 次训练、后 8 次测试；
  用 `M=12, R=4` 作为高校准量对照，`M=5, R=11` 作为低校准量对照。
- 每 subject 每数字不足 16 次的个体必须单独报告，不能混入主指标。

## 3. 大数据迁移训练：采用什么、不采用什么

### 3.1 主路线：P300 域内 SSL + 轻量 subject head

最有直接证据的蓝本是 SpellerSSL：

- 1D U-Net backbone，masked reconstruction + FFT 幅度一致 loss；
- in-domain P300 预训练优于 cross-domain MI 预训练；
- fine-tune 阶段对连续 repetition 做 G=2 聚合；
- II-B 上 94% CRR@7，单试次 accuracy 76.18%，校准量可减 60%。
- 来源：[SpellerSSL arXiv](https://arxiv.org/abs/2509.19401)、
  [SpellerSSL Scirate](https://scirate.com/arxiv/2509.19401)。

**批判保留**：SpellerSSL 的 "in-domain pretraining" 实际上是在 Subject A 上
预训练、在 Subject B 的 85 个字符校准后测试；它是单源被试迁移，不是大规模
多中心预训练。94% CRR@7 不能直接迁移到 GTN 9 选数字，但机制（重建预训练 +
聚合 + 轻量 head）值得在本项目用 matched protocol 复验。

建议迁移层级：

```text
T0  target-only: 只使用目标人 prefix。
T1  leave-one-target-subject-out: 在其余 GTN 被试上预训练，目标人 prefix 适配。
T2  cross-dataset: 加入 BI2014a、BNCI2014_008 等 P300 源；参考电极不一致必须先
    显式重参考，不能直接 concat。
T3  通用 EEG FM: 只作为探针/负对照，不作为 85% 目标的依赖路径。
```

### 3.2 域适配对照

- xDAWN + Riemannian transfer 是 P300 上最稳的 classical transfer floor：
  [MDPI xDAWN-Riemann transfer](https://www.mdpi.com/2076-3417/10/5/1804)。
- 小样本跨数据集 P300 上显式 MMD 对齐优于直接混训：
  [Adaptive Split-MMD P300](https://scirate.com/arxiv/2510.21969)。
- 本项目应跑 `xdawn_rg`、source-pretrain/fine-tune、MMD 三条 matched 路线；
  不能用“更多源数据”掩盖协议泄漏。

### 3.3 明确不押注的路线

- 通用 EEG foundation model 直接线性 probing：多篇独立基准显示短窗 BCI 上
  specialist 不差、linear probing 常不足：
  [EEG-FM-Compass](Paper/EEG_FM_Compass_arXiv_2601.17883.pdf)、
  [CHIL FM generalization](Paper/EEG_FM_Generalization_Framework_arXiv_2605.28563.pdf)、
  [NeuralBench](Paper/NeuralBench_arXiv_2605.08495.pdf)。
- 大规模 FM 表示里的 subject identity 和 1/f 低频捷径必须先审计：
  [Identity Trap](Paper/Identity_Trap_in_EEG_Foundation_Models_arXiv_2606.06647.pdf)、
  [Low-frequency bias](Paper/Understanding_and_Correcting_Low-Frequency_Bias_in_EEG_Foundation_Models_arXiv_2608.01898.pdf)。
- GAN/WGAN 数据增强不作为第一阶段；先做便宜且无分布的 time jitter +
  Gaussian noise + channel dropout。ERP-WGAN 证据存在
  ([ERP-WGAN](https://www.sciencedirect.com/science/article/pii/S0165027022001480))，
  但 ERP-XTTN 只用 jitter/noise 就达到与复杂 baseline 接近的 LOSO 表现。

## 4. 对当前模型搜索的外部校准

- ERP CORE P300 LOSO：full montage EEGNet AUROC=0.776 / BACC=0.693；
  3 通道 EEGNet AUROC=0.720 / BACC=0.640。
  [ERP-XTTN arXiv](Paper/ERP_XTTN_arXiv_2606.02939.pdf)。
- 本项目 128 Hz 合同 LOSO：EEGNet AUC=0.7396 / BACC=0.6754，与上述外部
  LOSO 水平一致；说明当前 pipeline 没有明显泄漏或虚高。
- GTN 旧文献（Vařeka 及后续比较）单试次 accuracy 约 61–77%，且多数是
  3 通道、旧预处理；不作为当前合同的目标上限。
  [Vařeka P300 CNN/RNN](Paper/102482.pdf)、
  [GTN architecture comparison](Paper/SHTI-292-SHTI220333.pdf)。
- P300 latency jitter 是单试次信息损失的主因之一，`[200,500] ms` 约束窗内
  单试次 realignment 可显著改善 ERP 形态；但必须同时报告 latency shift/jitter，
  防止把伪迹对齐误当增益：
  [JNE 2026 latency realignment](Paper/Improving_P300_Morphology_Single_Trial_Latency_Realignment_JNE_2026.pdf)。

## 5. 预注册实验矩阵

所有实验固定：同一 GTN 128 Hz 新 cache、同一 prefix/suffix 协议、同一 QC、
同一 outer test suffix、同一 seed manifest。

| ID | 训练数据 | 模型 | 决策 | 主判据 |
|---|---|---|---|---|
| W0 | target prefix | `window_lr`, `xdawn_rg` | hit@8 | classical floor |
| W1 | target prefix | `eegnet`, `ms_eegnet` | hit@8 | neural floor |
| W2 | target prefix + jitter/noise | W1 冠军 | hit@8 | augmentation 净效应 |
| T1 | 其他 GTN 被试 SSL | frozen trunk + subject head | hit@8 | leave-one-out transfer |
| T2 | T1 + BI/BNCI 源 | same | hit@8 | cross-dataset transfer |
| T3 | T1/T2 冠军 + MMD | same | hit@8 | explicit alignment |
| D1 | W/T 冠军 | sum vs mean vs precision-weight | hit@R, R=3..8 | decision aggregation |
| L1 | T 冠军 | prefix 大小 5/8/12 reps | hit@8 | calibration budget |

判定规则：

1. W1 必须至少与 W0 非劣；否则先修 baseline。
2. T1 相对 W1 必须报告同 prefix 训练量下的 delta 和 95% bootstrap CI；
   只在正 delta 且不劣于 W1 时晋升。
3. 只有 T2/T3 中更好者进入 D1；不允许用测试 suffix 回头调 R 或 head。
4. `hit@8 >= 0.85` 且 calibration <= 60% prefix 时，才写“达到单被试 85%”。

## 6. 需要的代码改动（按依赖排序）

1. **GTN 128 Hz 新 cache**：用当前合同重跑 `prepare_gtn_dataset.py`，
   并生成 SHA-256 attestation。
2. **causal filter 合同**：`PreprocessingSpec` 增加合法的
   `filter_phase=forward` / `online_causal=True` 单被试 profile；
   不允许把零相位滤波结果切成 prefix/suffix 后宣称无泄漏。
3. **per-subject prefix/suffix folds**：
   `experiments/run_within_subject.py` 或 `evaluate_within_subject.py`，
   输入 `(subject, repetition_index, stimulus)`，输出 hit@R 曲线。
4. **decision 层升级**：在 `models/decision.py` 增加
   repetition-weighted sum、mean、precision-weighted sum，并报告空集/低置信。
5. **transfer 模块**：
   - `src/transfer/ssl.py`：P300 masked reconstruction pretraining；
   - `src/transfer/adapters.py`：frozen trunk + subject head；
   - `src/transfer/mmd.py`：Split-MMD / MMD 对齐对照；
   - 全部遵守 `group_disjoint_validation_split`，禁止 subject identity 入特征。
6. **审计模块**：subject-axis linear probe、aperiodic/1/f ablation，
   作为 FM/SSL 候选晋升前置检查。

## 7. 结论

- 85% 必须定义为**同人 prefix→suffix 的 9 选 hit@8**，不是 LOSO 单试次
  BACC。
- 由 binary AUC 推导，单被试 binary AUC 0.78–0.80 时 hit@8 的理论上限已达
  0.92–0.95；因此工作重心是稳定单被试 binary AUC，而不是发明 9 选网络。
- 文献中 SpellerSSL 是机制最接近的迁移方案，但其数字不能直接搬；
  xDAWN-Riemann/MMD 是必须保留的 transfer floor；通用 FM 不押注。
- 先实现 GTN 新 cache + causal single-subject folds + hit@R evaluation，
  再按 T0→T1→T2 顺序做迁移；每层晋升都必须有同 prefix 训练量对照。
