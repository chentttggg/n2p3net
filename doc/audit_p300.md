# P300 PEC 审计模块

> v12 口径：本模块回答“representation 是否编码了 P300 特征”，不回答生理潜伏期。
> 潜伏期测量与 claim 走 [`blueprint.md`](blueprint.md) 的 LatencyMeasurement 对象；
> PCW attention soft-argmax 只作为 routing 对照，不得解释为生理 tau。


`src/audit_p300` 是一个独立的 P300 Probe--Erase--Closure 审计实现。它不导入
`src/models`、`src/baselines` 或任何训练代码，因此可以先审计现有模型，再决定是否把
适配器接回原实验流程。

## 审计口径

`P300AuditData.X` 必须是 `(N, C, T)` 的 stimulus-locked epoch，`time_ms` 是严格递增的
毫秒时间轴。`P300Split` 只接受调用方显式提供的 train/validation/test 行索引，默认要求
三部分 subject-disjoint；审计内部不会随机切分。

默认特征词典有 63 个 label-free 特征，分为 `time`、`frequency`、`time_frequency`、
`complexity`、`cross_frequency`、`cross_channel` 六个家族。默认窗口是 baseline
`[-200, 0)` ms、N2 `[150, 300)` ms、P3a `[250, 430)` ms、P3b `[300, 650)` ms、
analysis `[0, 800)` ms。没有刺激前 baseline 时，必须显式配置
`P300FeatureConfig(baseline_policy="first_window")`，不会静默伪造 baseline。

审计的三个阶段分别回答不同问题：

1. Probe：逐层 ridge probing，在 validation 上选 alpha 和 peak layer，并使用 shuffled-target
   与 Gaussian target 控制。
2. Erase：在训练 split 拟合 feature-correlated subspace，只通过 adapter 的显式
   `LayerIntervention` 注入模型；配合 random/shuffled/Gaussian erasure、paired bootstrap
   和 BH-FDR。
3. Closure：用被确认使用的透明 P300 特征训练 logistic regression，并与同维随机特征和
   原模型比较。可同时报告 trial-level binary AUC 与 subject-level 9-choice `digit_hit`。

## 最小调用

```python
from audit_p300 import (
    ArrayP300Adapter,
    P300AuditData,
    P300PECAuditor,
    P300Split,
)

data = P300AuditData(
    X=epochs,  # (N, C, T)
    target=target_non_target,  # 0/1
    subjects=subject_ids,
    time_ms=time_ms,
    digits=digits,  # optional, needed by digit_hit
    thought_numbers=thought_numbers,
    channel_names=("Fz", "Cz", "Pz"),
)
split = P300Split(train=train_idx, validation=validation_idx, test=test_idx)

adapter = ArrayP300Adapter(
    activation_cache={"readout": cached_layer},  # every layer is (N, D)
    score_from_activations=lambda layers: layers["readout"][:, 0],
)
report = P300PECAuditor().run(adapter, data, split, model_name="cached_model")
report.save_json("artifacts/p300_audit.json")
```

`ArrayP300Adapter` 的 cache 必须覆盖完整 `data.X`。Erase/runner 会在完整数组上执行
干预，再按 `split.test` 取分数，避免缓存 activation 与 test 子集错位。

## N2P3-Net Torch binding

N2P3-Net 的 forward 输入是 `X, E_chn, E_sub, ...`，输出是
`N2P3NetOutput`。最直接的审计层是 `tokenizer` 的 `(B,T,D)` 输出和 `encoder` 的
`Z2=(B,T,D)` 输出；默认 reduction 会沿 token 轴 mean-pool 成 `(B,D)`，intervention
会把 edit delta 广播回每个 token。若要审计完整 `(B,3,D)` 的 `H`，应为该输出另写一个
明确的 binding，而不是把默认 reduction 当作生理等价变换。

```python
import torch

from audit_p300 import TorchLayerBinding, TorchP300Adapter

e_chn = torch.as_tensor(channel_embedding, dtype=torch.float32)  # (C, d_chn)
e_sub = torch.as_tensor(subject_embedding_by_trial, dtype=torch.float32)  # (N, d_sub)


def input_fn(batch, batch_indices, device):
    return (
        torch.as_tensor(batch, dtype=torch.float32, device=device),
        e_chn.to(device),
        e_sub[batch_indices].to(device),
    )


adapter = TorchP300Adapter(
    model,
    bindings={
        "tokenizer": TorchLayerBinding("tokenizer", model.tokenizer),
        "encoder": TorchLayerBinding("encoder", model.encoder),
    },
    score_fn=lambda output: output.heads.logit_target,
    input_fn=input_fn,
    device=torch.device("cpu"),
)
```

如果不使用 subject embedding，可以省略 `input_fn`；默认输入只把 numpy epoch 转成
`(B,C,T)` float tensor。模型若需要 `channel_mask` 或 `domain_id`，应在 `input_fn` 中按
`batch_indices` 对齐并返回自定义输入对象，同时在模型前向签名处处理该对象。

## EEGNet 对接

项目 `DeepBaseline` 的 EEGNet 训练完成后，`model_` 是 braindecode EEGNet，输出为
`(B, 2)` class logits。EEGNet 的卷积输出通常是 `(B, F, 1, T')`，其特征维不在最后一轴，
所以不能直接使用默认 reduction；需要显式声明 reduction/lift：

```python
def pool_eegnet(tensor):
    return tensor.mean(dim=(2, 3))  # (B, F)


def lift_eegnet(native, original, edited):
    delta = (edited - original).unsqueeze(-1).unsqueeze(-1)
    return native + delta


binding = TorchLayerBinding(
    name="separable_point",
    module=model_.conv_separable_point,
    reduce=pool_eegnet,
    lift=lift_eegnet,
)
adapter = TorchP300Adapter(
    model_,
    bindings={"separable_point": binding},
    score_fn=lambda output: output[:, 1] - output[:, 0],
)
```

这里的 `lift_eegnet` 是“按 feature channel 广播 edit delta”的干预定义，不等同于对每个
时间点独立擦除。若研究问题要求时间定位，应改为保留 `(B,F,T')` 的 audit 表示并定义
对应的透明 reduction/lift；不要把默认 mean-pool 解释成时频定位证据。

## 结果解读边界

`selection_encoded` 只表示 validation 选层上的可读出证据；`representation_causal` 还要求
擦除造成显著性能下降、通过 null control、BH-FDR 和 residual probe 检查。它不是“特征是
神经生理因果变量”的证明，而是“该 feature-correlated representation subspace 对当前
任务决策有表示层贡献”的操作性结论。

Closure 比例也不是模型全部机制的比例。它只衡量当前 63 个手工特征词典能追回多少模型相对
随机特征基线的优势；未追回部分可能是词典遗漏、跨特征组合或 adapter/intervention 定义
不足，需要在独立实验中继续区分。
