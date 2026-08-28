# N2P3-Transfer v1 落地实现与鲁棒性门禁

日期：2026-08-28
状态：implementation spec
依赖：`doc/pretraining_head_comparison.zh.md`、
`doc/single_subject_transfer_research.zh.md`

## 0. 问题：为什么不能直接把 SpellerSSL/FAME/EEG2ERP 拼在一起

三家实验情景差异：

| 方案 | 原始数据 | 输入尺度 | 关键结构 | 直接搬到本项目的冲突 |
|---|---|---|---|---|
| SpellerSSL | II-A/II-B，64ch，240 Hz，667 ms | 160 samples | 1D U-Net，时间 mask，FFT L1，G=2 校准聚合 | 本项目 3ch/128 samples；换 U-Net 会丢掉已晋升 MS-EEGNet 的证据 |
| FAME | 多中心长时程 EEG | 长窗 | patch mask + 按频带均衡重建 | 1 s epoch 没有长时程；频带权重需按短窗 FFT 重定义 |
| EEG2ERP | ERP CORE 多范式，64ch | 多任务 ERP | split subject/task latent + bootstrap ERP + variance | 是 ERP 重建任务；本项目主目标是 binary 分类和 9 选决策 |

因此实现原则是：**借用“机制”，不借用“输入协议和骨干”；所有差异都转化为显式配置和消融轴。**

## 1. 模块边界

新增 `src/transfer/`，不改动当前 `N2P3Net` 的默认前向和已冻结实验。

```text
src/transfer/
  masking.py        # 时间 mask / channel mask / mask 比例策略
  heads.py          # DecoderHead, VarianceHead, SubjectProbeHead
  losses.py         # band-balanced FFT loss, NLL, identity penalty
  pretraining.py    # PretrainingConfig, PretrainingTask, train loop
  subject_adapter.py# FrozenTrunkAdapter, LinearHead, MLPHead, FullFine
  evaluation.py     # hit@R, calibration budget, robustness metrics
```

`N2P3Net` 只增加一个不改变默认行为的接口：

```python
def forward_features(self, x):
    # 返回 trunk 输出 H，不经过 classifier
    ...
```

`forward` 继续调用 `forward_features` 后接原分类头，保持旧模型权重、旧 manifest 和旧实验可复现。

## 2. 预训练任务：masked reconstruction，而不是分类

### 2.1 输入

- 只用源域：GTN 除目标人外 241 subjects；BrainSync/BI/BNCI 作为后续 T2。
- 输入仍是当前合同：128 Hz，2–30 Hz，`[-200,800) ms`，V，mean-only baseline。
- **不做 channel mask**。3 通道掩掉 1 个就丢失大量空间信息；8 通道同理。
  时间 mask 是主目标。
- 时间 mask：连续 block，block 长度 12–32 samples，总 mask 40–60%；
  对 128 sample epoch，短 mask 保留 N200/P300 局部形态，长 mask 强迫上下文补全。
- mask token：当前 trunk 是 Conv2d/Conv1d 结构，没有 tokenizer；
  用 `mask * x` 和 `mask` 通道一起送入一个轻量 mask embedding，避免引入 token 语义。

### 2.2 Decoder

不引入 U-Net。Decoder 是 trunk 的近似逆：

```text
H = trunk(x_masked)            # (B, S=4, T=32)
decoder:
  per-branch upsample x8  -> (B, S, T=128)
  pointwise mix S -> C
  output C x T
```

要求：decoder 参数 < trunk 参数，预训练后 `discard()`，不进入下游和部署。

### 2.3 Loss：SpellerSSL + FAME 的融合

```text
L = lambda_wave * L1(x, x_hat)
  + lambda_fft  * sum_b w_b * L1(|FFT_w(x)|_b, |FFT_w(x_hat)|_b)
```

具体规则：

1. 对每个 epoch 先 detrend + Hann window，再算 FFT 幅值；
2. 频带固定：delta 1–4, theta 4–8, alpha 8–13, beta 13–30 Hz；
3. `w_b` 由**源训练集**的每频带平均幅值倒数归一化得到；不使用目标人
   suffix，也不逐 subject 计算，避免把身份信息写进 loss；
4. `lambda_wave/lambda_fft` 初始 1/1，但每 N step 用 gradient-magnitude
   归一化保持两个 loss 同量级，不用验证集调这两个超参；
5. 增加总能量不变性测试：`L(x + c)` 与 `L(x)` 在 baseline-corrected 输入上
   必须接近，防止模型只学 DC/慢漂移。

这是 SpellerSSL 的 FFT 一致性 + FAME 的 band-balance，但明确去掉了
channel mask 和长窗 patch。

### 2.4 Subject identity 控制

- 每个源 subject 只出现在训练/验证之一；
- 在 trunk 输出后加一个可丢弃 `SubjectProbeHead`，线性探针；
- 若源验证集 subject probe balanced accuracy > 0.80，冻结前对 trunk 输出
  做线性 subject-axis erasure（DDP 目标为 `subject_id` 的梯度反转）再重测；
- 下游 fine-tune 前必须再次运行探针，把 probe 分数写进 record。

对应 Identity Trap 论文，但不是照搬其长时程临床协议，只保留可执行的
线性 probe + 1/f 消融两项。

## 3. 下游 subject head：冻结 trunk 是假设，不是默认

对每个目标人：

```text
prefix trials -> frozen trunk -> H -> aggregation -> classifier
```

head 轴：

| head | 说明 |
|---|---|
| `linear` | H 经 `ms_flatten` 后 40 维，直接 linear |
| `mlp16` | 40->16 GELU->2，容量对照 |
| `full_fine` | 低学习率全参 fine-tune |
| `xdawn_rg` | 相同 prefix 的 classical floor |

选择规则：只有 `linear` 或 `mlp16` 在相同 prefix 预算下不劣于 `full_fine`
和 `xdawn_rg`，冻结 trunk 才成立；否则报告“该迁移不成立”，不得写
“预训练有效”。

## 4. 后处理落地

### 4.1 binary 校准

- 用目标人 prefix 内更早的 sub-prefix 做 group-disjoint 校准；
- 候选：Platt、temperature、isotonic；选择 NLL/ECE 最好的，禁止用 suffix。

### 4.2 9 选聚合

所有聚合都作用在 calibrated logit 上：

```text
sum        baseline
mean       trial 数不齐对照
trim0.2    去掉每数字两侧 20% logit 后求和
precision  1/var(logits within digit) 加权；var 来自 3-seed ensemble
```

precision 不依赖 EEG2ERP 的 variance decoder，而是用 3 个不同 init/seed 的
head ensemble 近似单试次不确定性。若后续实现 variance head，再替换。

### 4.3 Dynamic stopping

- 只对 P/D 胜出模型启用；
- 阈值在 prefix 上按 hit@R 与平均 R 的 Pareto 曲线选择；
- 测试时若 margin 低于阈值则继续；达到最大 R 仍低则 abstain；
- 报告：hit@R、平均 R、abstain 率、ITR。

## 5. 鲁棒性门禁

每层晋升前必须全过以下门禁：

### G1 泄漏门禁

- `filter_phase=forward` 且 `online_causal=True`；
- train/calibration/test 的 stimulus onset 时间严格递增，无交叠 repetition；
- source 与 target 的 subject 集合不相交；
- 零相位 LOSO 结果不得与因果单被试结果混报。

### G2 统计门禁

- 242 目标人按可用 trial 分层报告，不只报告均值；
- 每模型至少 3 seed；对外层 subject 做 paired bootstrap CI；
- 多对比时报告 Holm 校正后的 p；
- 报告最差 decile subject，而不是只报中位数。

### G3 域鲁棒门禁

- subject probe 分数必须写入 record；
- 1/f 消融：用 aperiodic-removed 输入重跑，模型性能若显著下降，说明学的是
  慢漂移而非 ERP，直接判负；
- source 消融：T1a-only、T1a+T1b、T1a+T2 三组必须都报告，不允许只报最好。

### G4 对抗与分布偏移门禁

- 测试期注入 ±10/20 ms latency jitter 和 channel dropout；
- 报告性能下降曲线，超过预定 margin 判负；
- 对低 trial subject 单独报告，不得用“数据不足”删除后仍报主指标。

### G5 成本门禁

- 下游头参数 < 5k；
- batch-1 CPU 延迟和峰值内存必须与 `ms_flatten` 同协议报告；
- 预训练 decoder 不得进入部署路径，record 中写明 `discarded=True`。

## 6. 实验执行顺序

```text
Step 0  生成 GTN causal cache:
        python experiments/prepare_gtn_dataset.py \
          --filter-phase forward --output experiments/cache/gtn_causal_v1.npz

Step 1  W0/W1 target-only prefix -> suffix hit@R baseline

Step 2  P1 T1a 预训练:
        python experiments/run_pretrain.py --source-cache gtn_causal_v1.npz \
          --holdout-subject SUBJ --mask 0.4-0.6 --bands 1-4,4-8,8-13,13-30

Step 3  downstream:
        python experiments/run_within_subject_transfer.py \
          --pretrained-checkpoint ... --target-subject SUBJ \
          --prefix-reps 8 --test-reps 8 --heads linear,mlp16,full_fine,xdawn_rg

Step 4  decision:
        sum / mean / trim0.2 / precision; dynamic stopping threshold from prefix

Step 5  robustness gates G1-G5
```

## 7. 何时停止

- 若 `linear`/`mlp16` 在 60% prefix 下 hit@8 >= 0.85，且通过 G1–G5，晋升；
- 若只有 `full_fine` 达到，说明收益来自目标人训练而非迁移，不得称 SSL 有效；
- 若都达不到，回退到 `ms_flatten` + xDAWN-Riemann classical floor，并删除
  未通过门禁的 transfer 分支，不保留弱化语义。
