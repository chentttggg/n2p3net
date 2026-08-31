# N2P3-Transfer v1 落地实现与鲁棒性门禁

日期：2026-08-28
状态：historical mechanism spec；当前执行状态如下：source-supervised checkpoint
保留训练好的 classifier、inner epoch 选择后在全部 source rows refit，并固化完整输入
签名；masked-reconstruction checkpoint 的 classifier 未训练，不能 zero-shot 使用。
GTN labelled same-selection adaptation 已降级为 oracle proxy。当前研究顺序直接见
`research_program.zh.md`，本文不再提供独立执行默认值。
依赖：`doc/pretraining_head_comparison.zh.md`、
`doc/single_subject_transfer_research.zh.md`

## 0. 问题：为什么不能直接把 SpellerSSL/FAME/EEG2ERP 拼在一起

三家实验情景差异：

| 方案 | 原始数据 | 输入尺度 | 关键结构 | 直接搬到本项目的冲突 |
|---|---|---|---|---|
| SpellerSSL | II-A/II-B，64ch，240 Hz，667 ms | 160 samples | 1D U-Net，时间 mask，FFT L1，G=2 校准聚合 | 本项目 3ch/128 samples；且只有两个源被试，不能决定 GTN backbone |
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

## 2. 预训练原型：masked reconstruction 机制臂

更新后的主顺序是 source-supervised -> supervised+auxiliary SSL -> pure
reconstruction control。以下实现保留为 pure reconstruction 对照，不再预定为主线。

### 2.1 输入

- 源域人数不硬编码；从冻结的 block manifest 与 cache subject ledger 派生，并在
  checkpoint 中逐一固化 `training_subject_keys` 与 holdout subjects。
- 输入由 checkpoint 绑定的完整合同决定。当前 GTN matched factorial 是
  0.1/0.5 Hz x 800/1200 ms、128 Hz、V、mean-only baseline；chronological 路径
  必须使用 forward IIR + `steady_state_first_sample`，不得回退到通用 2 Hz/800 ms。
- **不做 channel mask**。3 通道掩掉 1 个就丢失大量空间信息；8 通道同理。
  时间 mask 是主目标。
- 时间 mask：连续 block，block 长度 12–32 samples，总 mask 40–60%；
  对 128 sample epoch，短 mask 保留 N200/P300 局部形态，长 mask 强迫上下文补全。
- 当前实现把 masked samples 置零，`keep mask` 只传给 decoder；没有把 mask 作为
  新通道送入 trunk。若增加 mask embedding，必须作为独立消融。

### 2.2 Decoder

不引入 U-Net。Decoder 是 trunk 的近似逆：

```text
H = trunk(x_masked)            # (B, S=4, T=32)
decoder:
  per-branch upsample x4  -> (B, S, T=128)
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
3. `w_b` 由**源训练集**中固定 seed、固定样本数子集的每频带全局平均幅值
   倒数归一化得到；不按 batch 分别求逆，不使用目标人 suffix，也不逐
   subject 计算，避免把 batch size 或身份信息写进 loss；
4. 当前实现固定 `lambda_wave/lambda_fft=1/1`；gradient-magnitude 动态归一化尚未
   实现，若加入必须记录轨迹并作静态权重对照；
5. 增加总能量不变性测试：`L(x + c)` 与 `L(x)` 在 baseline-corrected 输入上
   必须接近，防止模型只学 DC/慢漂移。
6. 每个频带先对 batch 和 channel 求均值，再按 `w_b` 加权；重复同一批样本
   不得改变 loss。Hann window 与频带归约矩阵按 `(T, sfreq, device, dtype)`
   缓存，禁止在 CUDA batch loop 中用 `int(tensor)` 检查频点数。

这是 SpellerSSL 的 FFT 一致性 + FAME 的 band-balance，但明确去掉了
channel mask 和长窗 patch。

### 2.4 Subject identity 控制

- 每个源 subject 只出现在训练/验证之一；
- `SubjectProbeHead` 目前只有组件接口，标准 runner 未启用、未传 subject IDs，也未
  优化/记录该 probe。正式门禁应在 stop-gradient frozen features 上独立训练 probe；
  高 probe 分数触发 erasure/GRL matched ablation，不自动判死或自动删除表示。

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
| `linear` | H 经 `ms_flatten` 后 16 维，直接 linear |
| `mlp16` | 16->16 GELU->2，容量对照 |
| `full_fine` | 低学习率全参 fine-tune |
| `xdawn_rg` | 相同 prefix 的 classical floor |

选择规则：只有 `linear` 或 `mlp16` 在相同 prefix 预算下不劣于 `full_fine`
和 `xdawn_rg`，冻结 trunk 才成立；否则报告“该迁移不成立”，不得写
“预训练有效”。

## 4. 后处理落地

### 4.1 binary 校准

- 用目标人 prefix 内更早且满足 evidence-available time 隔离的 sub-prefix 做校准；
- 当前合同是 weighted-CE prior offset correction + 正温度 `T>0`。普通 Platt
  可能在小验证集学出负斜率并翻转候选排序，禁止使用；若保留诊断 Platt，必须
  强制正 slope。isotonic 不作为当前小样本在线默认。

### 4.2 9 选聚合

所有聚合都作用在 calibrated logit 上：

```text
sum        baseline
mean       trial 数不齐对照
trim0.2    去掉每数字两侧 20% logit 后求和
precision  sum(w_i l_i)/sum(w_i), w_i=1/v_i；v_i 必须是逐 trial predictive variance
```

没有逐 trial ensemble variance 或 variance head 时，precision 必须 fail closed；
同一 candidate 内 logits 的样本方差不是 predictive variance，不能代替。

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

- 以 manifest 请求的完整 cohort 为 operational 分母，按实际 eligible/failed
  原因分层报告；不得硬编码 241/242，也不得只报告 eligible 均值；
- 每模型至少 3 seed；对外层 subject 做 paired bootstrap CI；
- 多对比时报告 Holm 校正后的 p；
- 报告最差 decile subject，而不是只报中位数。

### G3 域鲁棒门禁

- subject probe 分数必须写入 record；
- 1/f/aperiodic 移除只作敏感性分析。P300 本身包含低频瞬态，性能下降不能单独
  证明模型学习慢漂移，更不能直接判负；需结合时序打乱、subject probe 和 ERP
  morphology 一起解释；
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
Step 0  单个已冻结 GTN steady-state causal arm 的复现模板（2x2 factorial 已完成）:
        .venv/Scripts/python experiments/prepare_gtn_dataset.py \
          --root D:/path/to/GTN --cohort gtn --filter-phase forward \
          --l-freq 0.1 --tmax-ms 1200 --output CACHE.npz

Step 1  按冻结 4-block manifest 做 source-supervised checkpoint；每个 checkpoint
        排除完整 target block，并绑定 cache SHA、输入合同与 source ledger:
        .venv/Scripts/python experiments/run_pretrain_supervised.py \
          --source-cache CACHE.npz --cohort gtn --tmax-ms 1200 \
          --holdout-subjects BLOCK_KEYS --pooling-mode full_unfold \
          --temporal-kernel-size 35 \
          --epochs 100 --qc-ptp-uv 100 --checkpoint BLOCK.pt

Step 2  GTN 合法主结果只跑 target-excluded Z0；保存完整 trial ledger 后计算
        hit@all_balanced、raw hit@all 与全部 hit@R。Z5 等待相同 M 但不使用标签，
        只用于 time/coverage sensitivity:
        .venv/Scripts/python experiments/run_within_subject_transfer.py \
          --dataset-cache CACHE.npz --checkpoint BLOCK.pt --cohort gtn \
          --prefix-reps 0 --test-reps all --head zero_shot --aggregation sum \
          --target-subjects-file BLOCK.json --output Z0.json

Step 3  `--prefix-reps 5 --head zero_shot` 是 Z5。任何读取同 selection 标签的
        classifier_fine/linear/mlp16/full_fine 都是 O5 oracle proxy，必须带
        `--allow-oracle-same-selection-adaptation`，不得写成未知数字校准。

Step 4  监督校准主线转到 BI candidate-v2 的跨 decision split；64 人 causal-v2
        cache 已构建并 attested，性能结果须等待冻结 block checkpoint 后再报告。
        最终裁决转到成人 BrainSync 多 target-switch decisions。

Step 5  对冻结候选执行鲁棒性门禁 G1-G5。
```

## 7. 何时停止

- 只有成人 BrainSync 9-choice 多 target-switch 的合法 cross-decision arm 在预注册
  绝对 prefix/R 预算下 hit@R >= 0.90 且通过 G1–G5，才报告产品 point estimate 与 CI；
- 若只有 `full_fine` 达到，说明收益来自目标人训练而非迁移，不得称 SSL 有效；
- 若迁移头都达不到，回退到 adopted `full_unfold` zero-shot + xDAWN-Riemann
  classical floor；`ms_flatten` 仅保留结构基线，并删除
  未通过门禁的 transfer 分支，不保留弱化语义。
