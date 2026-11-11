> 历史参考材料：反映 2026 年 8 月的探索阶段。文中“当前”“正式”“默认”等均指当时版本；现行规则见本仓库六指南。
> 此为阅读副本：只整理导航并增加本提示，原始字节与 SHA-256 由上级清单和文字原文归档保存。
> [返回参考资料索引](../README.zh.md)

# PCW 时间平移等变研究协议

## 目标

PCW 的 `tau` 是分类路由参数，不是生理潜伏期。本研究只检验一个更窄的命题：
当输入中的时间模式作已知平移时，PCW 路由窗是否随之平移，并且这种约束是否提高外部分类泛化。

生产 v12 保持关闭。本协议是显式研究路线，不能据此作 latency 生理解释。

## 数学定义

令 `S_s` 表示整数样本平移，满足

```text
S_s X[t] = X[t - s]
```

对每个 trial `b`，有效重叠索引为

```text
I_s = { t | 0 <= t + s < T }
```

设 `A(X)` 是 PCW 的三个归一化软窗。为了不让边界填充值或截断质量影响目标，先在
`I_s` 上分别归一化：

```text
p_c(t) = A_c(X)[t] / sum_{u in I_s} A_c(X)[u]
q_c(t) = A_c(S_s X)[t + s] / sum_{u in I_s} A_c(S_s X)[u + s]
L_attention_shift = mean_{b,c} 1/2 * sum_{t in I_s} (p_c(t) - q_c(t))^2
```

同时保留旧的尺度锚定：

```text
L_jit = mean((tau(S_s X) - tau(X) - s / sfreq * 1000)^2) / tau_scale_ms^2
```

训练总损失新增：

```text
L = L_existing + lambda_jit * L_jit
               + lambda_attention_shift * L_attention_shift
```

为区分权重不足与训练过程冲突，`--shift-equivariance-ramp-epochs R` 可将两个
shift 权重在第 `e` 个零基 epoch 同比缩放为 `min(1, e/R)`。`R=0` 是固定权重；
`R>0` 的第一个 epoch 权重为零，避免从随机 PCW 路由直接施加强等变梯度。

## 反例和防护

1. 零填充捷径：模型可只检测全零端点得知 `s`。研究路径在 Stage 0 完成重参考和
   baseline 标准化后再移位，并用同 trial baseline 重采样填充无效端点。
2. baseline 重定义：在原始 epoch 上移位会改变 `forward` 用于标准化的前段基线。研究路径
   用 `input_preprocessed=True` 保证两个视图共享同一 Stage-0 变换。
3. 边界质量：损失只比较 `I_s`，并在该区域重新归一化；无效端点的任意质量不应改变损失。
4. 均匀窗：均匀 `A` 也能令 `L_attention_shift=0`。因此等变损失不是定位证据，必须以
   与 global-pool 的外部配对 AUC 消融作为收益判据。
5. 边界/时延饱和：`tau` 受 PCW 参数界约束，不能把 `L_jit` 的下降解释成真实 latency recovery。

## 运行和判定

实验从 named run 启动，例如：

```bash
python experiments/run_eeg_loso.py ... \
  --lambda-jit 0.05 \
  --lambda-attention-shift 0.10 \
  --jit-prob 0.5 \
  --jit-max-ms 20
```

这些是研究起点，不是调好的生产超参。仅在 inner validation 选权重；当前 BI2014a
64-fold 结果已用于发现 PCW 问题，不能再充当调参后的确认集。确认时要求：

1. 对 global-pool 的外部 fold 配对 AUC 增益达到预先声明阈值。
2. 增益不依赖少数被试，且 BACC 方向一致。
3. `L_attention_shift` 下降不能伴随窗宽退化或 `tau` 大面积触边。

要区分学习过程与权重尺度，在同一开发折运行相同终值的两组：固定权重（`R=0`）与
渐进权重（例如 `R=6`）。固定组获益说明主要是原始权重过弱；仅渐进组获益说明强约束
从训练开始即介入破坏了主任务优化；两者都无增益则否定该辅助目标的当前形式。
