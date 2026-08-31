# SpellerSSL 类预训练头与后处理横向对比

日期：2026-08-28
状态：historical mechanism comparison；执行指令已直接改由
`research_program.zh.md` 和 `single_subject_transfer_research.zh.md` 管理。
当前 checkpoint 已绑定 target/cache identity、通道、参考和 preprocessing；causal
IIR 使用稳态初值。GTN 同一 thought digit 的 labelled prefix 不再视为合法校准，
只允许 zero-shot 或无真值适配；监督校准由跨 decision target-switch 数据验证。
参考文献按“可复验程度”排序，不按论文宣称数字排序。

## 1. 预训练头/目标横向对比

| 方案 | 预训练目标 | 骨干 | 下游头 | 优点 | 风险/成本 |
|---|---|---|---|---|---|
| SpellerSSL | 50% 时间 mask；波形 L1 + FFT 幅值 L1 | 1D U-Net | 轻量 ERP-Head，subject 校准 fine-tune | P300 域内有效；G=2 聚合显著提高 CRR；校准减 60% | mask 只做时间，无 uncertainty/artifact 建模；II-B 单源迁移证据 |
| EEG2ERP / CSLP-AE | bootstrap ERP 重建；split subject/task latent；variance decoder；trial-count conditioning | CSLP-AE encoder-decoder | ERP 估计 + inverse-variance 聚合 | few-trial ERP 去噪、零样本新被试、显式不确定性 | 训练复杂；P300 上是 ERP 重建而非直接 binary 分类 |
| BENDR | 时域 mask + contextualized token 预测 | CNN encoder + Transformer | linear probe / fine-tune | 长时程上下文 | 1 s P300 epoch 上下文有限；线性探针增益不稳 |
| LaBraM/CoMET/FAME | patch mask 重建；FAME 做 band-balanced 频域重建 | Transformer | linear/full fine-tune | 数据规模大；FAME 缓解 1/f 低频偏置 | 短窗 BCI 增益有限；subject identity 捷径；部署重 |
| Contrastive ERP CNN | supervised/self-supervised contrastive 拉近同标签 ERP | compact CNN | 分类头 | 小样本稳定；目标类别语义强 | 需要 batch/pair 设计；可能与 reconstruction 重叠 |
| xDAWN-Riemann | 无预训练；空间增强 + 协方差黎曼几何 | 非参数 | tangent-space logistic | P300 transfer floor 强、轻量 | 单试次容量有限；但校准 score 可直接做 9 选聚合 |

来源：
- [SpellerSSL arXiv](https://arxiv.org/abs/2509.19401)；
- [EEG2ERP](https://openreview.net/forum?id=c6LgqDhpH0)、
  [EEG2ERP TMLR](../Paper/Estimating_Event_Related_Potential_from_Few_EEG_Trials_arXiv_2511.23162.pdf)；
- [BENDR](https://pmc.ncbi.nlm.nih.gov/articles/PMC8261053/)；
- [CoMET](https://ar5iv.labs.arxiv.org/html/2509.00314)、
  [FAME/Low-frequency bias](../Paper/Understanding_and_Correcting_Low-Frequency_Bias_in_EEG_Foundation_Models_arXiv_2608.01898.pdf)；
- [ERP contrastive CNN](https://arxiv.org/abs/2407.04738)；
- [xDAWN-Riemann transfer](https://www.mdpi.com/2076-3417/10/5/1804)。

## 2. 后处理/聚合横向对比

| 方案 | 输入 | 机制 | 适合本项目 | 风险 |
|---|---|---|---|---|
| logit sum（现方案） | 每个数字的所有 test logit | 对数似然比可加 | 基线必须保留 | 对坏 trial/异常 logit 不鲁棒 |
| logit mean | 同上 | 除以 trial 数 | 试次不齐时更稳 | 丢失 repetition 数量先验 |
| SpellerSSL G aggregation | fine-tune 时相邻 G 个 repetition 平均 | 提高 calibration 信噪比 | 应作为训练增强，不等同测试聚合 | G 过大降低类间方差 |
| precision/inverse-variance 加权 | EEG2ERP 逐 trial variance | 可靠性加权 | 适合有 uncertainty head 的迁移模型 | 无方差估计时不可用 |
| trimmed/median logit | trial-level logit | 去离群 trial | 对眼动/伪迹残留鲁棒 | 需要选择 trim 比例，必须 train-only |
| weighted-CE offset + positive temperature | source 或更早的独立 calibration-decision validation logits | 解析移除训练权重/先验偏移，再以 `T>0` 缩放证据 | 保持候选排序方向，适合可加 LLR | GTN 同 selection 标签不得用于未知数字校准；校准集不能含测试 decision/suffix |
| dynamic stopping | 当前 9 选 posterior | 置信度阈值 / 不确定即继续 | 报告 hit@R 的速率-精度曲线 | 阈值必须 prefix 选，不能用 suffix 回头调 |
| candidate-level z-score | 9 个候选分数 | 去常数 bias | 当前 `decide` 已有 | 中心化会破坏严格 LLR；只做诊断 |

来源：SpellerSSL（G 聚合）；EEG2ERP（inverse-variance）；动态停止
[Schreuder et al. JNE](https://pubmed.ncbi.nlm.nih.gov/25588137/)；
当前代码 `models/decision.py`、`baselines/calibration.py`。

## 3. 取各家之长：N2P3-Transfer v1 方案

### 3.1 主干与预训练头

读出已固定为线性 `full_unfold`；`ms_flatten` 仅作显式 MS-EEGNet 基线。当前
GTN 三 seed all-evidence 已将 K35 设为临时默认，K65 保留强对照，K33 退出主线。
BI2014a 与 GTN 共同支持 full-unfold+K35 的开发采用，但不能产生产品确认。

预训练时加一个可丢弃的 decoder：

```text
masked X -> MS-EEGNet trunk -> H (B, S=4, T=32)
        -> light decoder -> X_hat
loss = lambda_wave * |X - X_hat|_1
     + lambda_fft  * sum_b w_b * |FFT(X)_b - FFT(X_hat)_b|_1
```

- 时间 mask 比例：40–60%，不做 channel mask（3/8 通道太少，mask 通道会毁掉空间信息）。
- `w_b` 使用 band-normalized 权重：delta/theta/alpha/beta 各自先除以该频带
  训练能量，再归一化；这是把 FAME 的 band-balance 思想压缩进 P300 频段，
  避免 1/f 低频主导重建。
- 在 loss 外增加独立、stop-gradient 的 subject-axis linear probe 审计。高 probe
  分数只触发 erasure/GRL 的 matched ablation，不能自动删除可能与 P300 共线的表示。
  依据：[Identity Trap](../Paper/Identity_Trap_in_EEG_Foundation_Models_arXiv_2606.06647.pdf)。

### 3.2 预训练语料优先级

```text
T1a  GTN 冻结 manifest 中除 target block 外的 subjects（人数从 ledger 派生）
T1b  BrainSync-GTN（若可得，8 通道同设备）
T2   BI2014a + BNCI2014_008（16/8 通道，仅作 representation 辅助源）
```

T1a 必须先行。T1b/T2 除共同重参考外还必须解决通道数/坐标合同；固定
`n_channels` 空间卷积不能把 3/8/16 导直接混入一批。当前已实现第一条显式路径：
pairwise 公共真实通道子集 + 同子集 CAR，并重新计算 QC/重建 checkpoint；缺通道和
preprocessing mismatch 继续 fail closed。dataset-specific spatial stem 尚未实现，
不得与确定性 CAR 结果混报。BI+BNCI 匹配 causal 5 导实测中，uniform joint
显著低于 BI-only；BI 3x/BNCI 1x 只恢复到 BI-only 附近。因此 T2 不能再写成默认
representation 增益，下一轴须先隔离 BI-source normalization 与跨域梯度冲突。

### 3.3 下游 subject head

对每个目标人：

```text
frozen trunk -> adopted full_unfold features (4 x 32 = 128-dim at current input)
             -> Linear + weighted-CE offset + positive-temperature calibration
             -> target-only prefix fine-tune
```

并行消融：

1. `head_linear`：只训练 full-unfold feature -> 2 线性层；
2. `head_mlp16`：低容量非线性 head，作为 capacity 对照；
3. `full_fine`：低学习率全参数 fine-tune，BN 可选冻结；
4. `xdawn_rg`：同 prefix 的 classical floor。

只有 `head_linear/head_mlp16` 在更少 prefix 下不劣于 `full_fine`，
才允许冻结主干。这对应 EEG-FM-Compass 的 “linear probing 常不足” 警示。

### 3.4 决策后处理（默认推荐）

1. binary 阶段先解析移除 weighted-CE 权重/训练先验偏移，再以 `T>0` 做缩放；
   温度只能由 source validation 或更早、目标已知且与测试目标变化的 calibration
   decisions 拟合。GTN 同一 selection 的 labelled prefix 只允许标为 O5 oracle proxy；
2. 对测试 suffix 按数字聚合：
   - 默认使用全部可用 trial 的 candidate `mean`，消除随机 occurrence 数量偏置；
   - `sqrt-count` 与 `sum` 只作计数敏感性对照；`trimmed_mean(0.2)` 作鲁棒性对照；precision-weight 仅在模型
     输出逐 trial predictive variance 时启用；
3. 报告 hit@R，R=3..8；
4. dynamic stopping 作为可选层：后验最高候选的 margin 达到 prefix 选定
   阈值即停止，否则继续 repetition；
5. 空集数字保持 `-inf`；低置信且 margin 不足时允许 abstain，并单独报告
   abstain 率，不能把 abstain 算成命中。

### 3.5 迁移路线

```text
W0  target prefix classical floor
W1  adopted full-unfold+K35；K65 强对照；MS-EEGNet 结构基线
S1  T1a supervised pretrain -> frozen/full-fine subject head
J1  S1 + temporal/spatial auxiliary SSL
R1  pure masked reconstruction control
P2  J1/R1 + band-balanced spectral objective
P3  P2 + subject-axis erasure/contrastive penalty ablation
P4  P2 + compatible cross-dataset spatial stem
D1  P development leader + sum/mean/trimmed/(valid precision)
D2  D1 + dynamic stopping
```

每层晋升判据：

- 与 W1 相同 prefix 训练量下，主比较使用 operational `hit@all_balanced` 的
  subject-level bootstrap CI；raw all、完整 hit@R/cost curve 必须共同报告；
- P2/P3/P4 任一步不优于 P1，即停止该分支；
- 外测 suffix 只用一次，不能回头改 mask 比例、band 权重、head 结构或 R。

## 4. 与文献的关键分歧

1. 不照搬 SpellerSSL 的 U-Net：当前 MS-EEGNet trunk 已有同协议 matched
   evidence，换主干会让预训练收益无法归因。
2. 不采用 BENDR/LaBraM 级别 backbone：1 s epoch 对长程 Transformer 收益
   存疑；且通用 FM 的身份/低频捷径审计成本高于预期增益。
3. 不完全采用 EEG2ERP 的 split-latent 完整结构：首轮只借用
   bootstrap-ERP target 和 inverse-variance 决策；若 P1 在 few-prefix 下
   ERP 估计指标也提升，才升级为双 latent。
4. G=2 calibration aggregation 只作为训练期数据增强消融，不直接改测试
   repetition 语义。

## 5. 判定

- causal steady-state ingress、checkpoint 输入签名和 `hit@R` 计算链已实现；每次准确率
  结论仍须由该次 cache/checkpoint/prediction artifacts 完整审计。
- GTN full-unfold 三核比较已完成；K35 是 development default、K65 是 unresolved
  control，不产生产品确认。
- 只有成人 BrainSync 9-choice、多 target-switch、later independent decisions 才能裁决
  `hit@R >= 0.90`。必须同时报告固定绝对校准量、完整 cohort、失败分母和 subject-cluster
  CI；GTN/BI 结果及 60% 目标数据都不得写成产品 90% 或 few-shot 达成。
