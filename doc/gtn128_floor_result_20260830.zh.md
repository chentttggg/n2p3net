# GTN 128 Hz 历史地板与 steady-state zero-shot 结果

日期：2026-08-30；2026-08-31 按 steady-state successor 证据直接修订。zero-phase
LOSO 制品仍可复算，但该 outer cohort 已用于多轮开发，不能作确认集。旧 zero-state
forward causal 制品已从正文结果中移除，不再作为 baseline。当前 causal 证据位于
`doc/evidence/gtn_20260831/`。该目录是 Git 内的 compact audit index，不包含
`.npz/.pt`，不能单独重跑；边界见其 `README.md`。

## 0. 结论先行

1. **历史 zero-phase LOSO 开发观察**：EEGNet/MS-EEGNet/full_unfold 相对
   xDAWN-RG 的 ΔAUC ≈ +0.10（p<1e-5，两 seed 一致）；支持保留深度候选。
2. **full_unfold 未在该历史 LOSO recipe 复现 BI 优势**：对 ms_flatten 两 seed
   ΔAUC −0.0041(p=0.27) / +0.0026(p=0.54)；对 EEGNet −0.0041(p=0.25) /
   **−0.0129(p=0.0017)**。BI2014a 的 +0.0109 优势在 GTN 3 导儿童数据上不复现，
   且对 EEGNet 方向反转为负。`full_unfold` 已因 BI 预注册机制比较被采用为项目
   读出，但本表不支持其为 GTN 准确率冠军。
3. **旧 recipe 的 GTN LOSO hit@8 最好 0.624**。它不是天花板，也不等于
   成人多 decision 的产品目标。
4. **当前 steady-state Z0 baseline**：0.1 Hz/1200 ms、source QC100、ms K65 的
   coverage 为 `230/245`，conditional hit@5 `0.578`，operational hit@5 `0.543`，
   AUC `0.709`；未达到 0.90。
5. **信号与 QC 已裁决为开发合同**：1200 ms 平均提升约 `+20.20 pp`，0.1 Hz
   平均提升约 `+5.92 pp`；source QC100 对 no-QC 提升 `+11.84 pp`，95% CI
   `[+5.31,+18.37] pp`。固定 0.1 Hz/1200 ms/source QC100。
6. **核长开发比较已完成**：balanced-all K35/K65/K33=`0.669/0.654/0.623`。
   K35 临时默认、K65 保留对照；K35 未确认胜 K65，K33 退出主线。

## 1. 协议

- **历史缓存**（仅用于本节 zero-phase LOSO 复算）：
  `gtn_sf128_lf2_hf30_tm-0.2_tx0.8_mean_zero.npz`。同名旧 `..._mean_fwd.npz`
  使用 zero-state，永久拒绝；当前 causal asset 是 2026-08-31 steady-state bundle。
  50,562 epochs × 3 ch × 128 samples，245 subjects，candidate_chain=True。
- **协议 A（LOSO）**：245 折，`window_lr / xdawn_rg / eegnet / ms_eegnet /
  n2p3net_full_unfold`，seeds 20260828/20260829（+20260830 classical）。
  30 epochs、patience 6、batch 512、BF16、compile、fused Adam。
- 旧 zero-state M=5/R=5 protocol 已从当前结果删除。后继 causal protocol 使用
  target-excluded Z0、forward steady-state IIR、source-full-refit checkpoint 与
  requested operational denominator；Z5 只作 matched-time sensitivity。
- **QC 台账**：新增 `exclude_unusable_test_epochs`——全通道被 fold-local 策略
  mask 的 held-out epoch 从测试折剔除并计入
  `per_fold[].artifact_quality.n_test_all_channels_bad_excluded`。这是对 3 导
  数据的必要扩展：GTN 单折内存在全坏 held-out epoch，旧 fail-closed 直接崩溃。
- **hit@R**：`evaluate.py` 新增 subject-macro hit@1..R（每被试一票，
  calibrated LLR 前缀累加 argmax），`record.json` 直接输出曲线。

## 2. LOSO 全量结果（subject-macro，hit 曲线全量见 analysis json）

| 模型 | seed | AUC | BACC | hit@5 | hit@8 |
|---|---|---:|---:|---:|---:|
| window_lr | 28/29/30 | 0.5136 | 0.5084 | 0.135 | 0.155 |
| xdawn_rg | 28/29/30 | 0.5557 | 0.5340 | 0.204 | 0.224 |
| eegnet | 28 | 0.6637 | 0.6109 | 0.522 | 0.604 |
| eegnet | 29 | 0.6606 | 0.6090 | 0.539 | 0.624 |
| ms_eegnet | 28 | 0.6638 | 0.6116 | 0.539 | 0.584 |
| ms_eegnet | 29 | 0.6451 | 0.5968 | 0.465 | 0.535 |
| n2p3net_full_unfold K65 | 28 | 0.6597 | 0.6081 | 0.531 | 0.588 |
| n2p3net_full_unfold K65 | 29 | 0.6477 | 0.5989 | 0.457 | 0.539 |
| n2p3net_full_unfold K33 | 28 | 0.6568 | 0.6016 | 0.531 | 0.592 |

classical 三 seed 完全一致（无随机性，管道确定性 ✓）。

### 2.1 配对推断（subject-paired sign-flip + plus-one，100k perm）

| contrast | seed 28 | seed 29 |
|---|---|---|
| fu − ms_flatten | −0.0041 (p=0.269) | +0.0026 (p=0.538) |
| fu − eegnet | −0.0040 (p=0.251) | **−0.0129 (p=0.0017)** |
| fu − xdawn_rg | +0.1040 (p<1e-5) | +0.0940 (p<1e-5) |

两 seed 中 fu 对 eegnet 方向一致为负。不做"fu 更优"宣称；对 EEGNet 的
BI2014a 增益不迁移到 GTN。种子间深度模型 AUC 波动 ~0.02，与被试内 seed 聚合
后的判读一致：**三深度臂在 GTN LOSO 上无可靠排序**。

## 3. Steady-state causal successor（2026-08-31）

四个 `0.1/0.5 Hz x 800/1200 ms` cache 的 record attestation、NPZ CRC 与 SHA 均
通过；4-block manifest 覆盖 245 人，32 个 source-full-refit checkpoints 可加载，
120 个 Z0/Z5 JSON 与独立复算一致。

| signal arm | Z0 coverage | conditional hit@5 | operational hit@5 | AUC |
|---|---:|---:|---:|---:|
| 0.1 Hz / 800 ms | 230/245 | 0.370 | 0.347 | 0.657 |
| **0.1 Hz / 1200 ms** | **230/245** | **0.578** | **0.543** | **0.709** |
| 0.5 Hz / 800 ms | 230/245 | 0.300 | 0.282 | 0.632 |
| 0.5 Hz / 1200 ms | 230/245 | 0.522 | 0.490 | 0.674 |

Z5 仅等待 M=5、但不使用标签，是 matched-time sensitivity；最佳臂 operational
hit@5 `0.408`，coverage `175/245`。它相对 Z0 的下降反映 late-session/coverage，
不是校准损失或增益。GTN O5 使用同 thought digit 标签，只允许 oracle proxy。

## 4. 与既有结论的关系（含推翻）

- `full_unfold` 的采用回答读出结构，不回答核长或产品准确率；`ms_flatten` 保留显式基线。
- K35 的旧 sensitivity 领先在三 seed all-evidence 中得到方向支持；K33 被稳定淘汰，
  K65 与 K35 仍未严格分离。
- source classifier + QC 在合法 Z0 上显著优于 no-QC，但 `0.543` 离 0.90 仍远；
  合法监督校准必须进入 BI cross-decision 或 BrainSync target-switch。
- 245 人仍不能估计“每个单被试长期 90%”；每人需要多个独立未知 target decisions。

## 5. 已知缺口与下一步

1. 当前开发主线固定 `full_unfold + K35`、0.1 Hz/1200 ms、source QC100；
2. BI candidate-v2 仍因 raw source 缺失未构建；补齐后执行跨 decision calibration；
3. 直接推进 decision objective/personalization 试错，最终由成人 BrainSync 多
   target-switch decisions 裁决 90%。
