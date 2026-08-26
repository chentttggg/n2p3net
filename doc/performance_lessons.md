# N2P3-Net 云端训练性能与通信教训

> 记录 2026-08-26 全云端训练流程审计与重建。这里的规则服务于
> `device-portability.md` 与 `CODING_WORKFLOW.md`，不改变科学协议。

## 已落地的教训

1. **数据只在 fold 开始上传一次。**
   `PreloadedDataLoader` / `GTNSetDataLoader` 持有同一份设备端 X/y，
   每 epoch 只做 `randperm + slice`。任何新 DataLoader 都不得把
   “每 epoch 重新 H2D” 带回来。

2. **per-batch 的设备→主机同步禁止进入热路径。**
   `bool(tensor.any()/all())`、`.item()`、`.cpu().tolist()` 在训练/验证
   batch 循环里会打断 GPU 流水线。数据契约只在 loader 构造时校验一次，
   之后通过 `TrialContext.prevalidated` / `SetMetadata.prevalidated`
   跳过值域复验。模型内的 finite 检查由 `N2P3Net.validate_input_finite`
   控制：Preloaded loader 一次全量验证后，Trainer 在 epoch 循环前关闭。

3. **与数据无关的 mask 用 CPU 一次算好。**
   `extract_quality_features` 的 FFT 频段 mask（nuisance/line/high）只依赖
   `sfreq/n_time`，改用 numpy 在 CPU 计算后上传为布尔张量，不再每 batch
   做 `mask.any()` 同步。

4. **训练期 AUC 不是早停判据，不得每 epoch 搬验证 logits。**
   `Trainer.fit` 的早停只使用 val loss；AUC 改为训练结束后用 best model
   做一次轻量验证 forward（`_evaluate_auc`）。`_evaluate_with_task` 只有
   `compute_auc=True` 时才累积 device 端 logits，并在最后一次 `.cpu()`。

5. **Set loader 的 rank 属于预处理，不属于每 epoch 迭代。**
   `GTNSetDataLoader` 在构造 `_sequences` 时一次性生成 repetition/sequence
   ranks；`__iter__` 只负责 shuffle 和拼 batch。

6. **repetition objective 的 `metadata.prevalidated` 快路径必须按 batch 担保。**
    `prevalidated_kmax` 是“本 batch 每个 GTN group 都覆盖的最小 K”，不是
    loader 级最大 K：ragged 覆盖时可能混有 K1/K2 序列。K ≤ 该值时，
    `repetition_multi_k_objective` 与 `additive_repetition_multi_k_objective`
    才跳过 `has_k.any()` / `active_weight` 同步；更大的 K 仍保留逐 K 空集
    保护，避免空 `cross_entropy` 产生 NaN。

## 尚未落地、留给下一次的重建项

7. **验证集每 epoch 仍有两趟 forward。**
   `_evaluate_with_task(val_loader)` 与 `_evaluate_set_with_task(val_set_loader)`
   使用同一份 `val_X_device`，但 set 覆盖与 trial 覆盖不同（K15 只有部分
   被试）。下一版应让 trial loader 携带 `SetMetadata`，一趟 forward 同时
   算 task/set 损失；或第一趟缓存 device 端 per-trial logits+quality。

8. **多进程 fold 仍为每个 task pickle 完整 X 与完整 event timeline。**
   full cohort 每个 task 约 157 MB X + 63 MB timeline。下一版使用
   `multiprocessing.shared_memory` / memmap，并把 timeline 切成折级子集。

9. **cache SHA-256 仍在加载后再次 `read_bytes()`。**
   应在 `run_gtn_baseline --prepare-cache-only` 时写 sidecar
   `<cache>.sha256`，训练入口优先读 sidecar；首次没有时再流式 hash。

10. **fit 后的 audit 仍多次 forward 验证/测试张量。**
     `predict_logit` / `predict_branches` / reliability audit / replay 可合并
     forward 结果；每次 fold 只发生一次，优先级低于 7–9。

## 修订记录

- 2026-08-26：完成 1–6；记录 7–10 作为下一阶段。
- 2026-08-26：修正 `prevalidated_kmax` 为 batch 内最小覆盖 K，封堵 ragged GTN 覆盖下空 `cross_entropy` NaN（有回归测试）。