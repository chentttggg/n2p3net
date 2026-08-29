# GTN 128 Hz 全量地板与因果适配：第一轮结果

日期：2026-08-30。状态：immutable exploratory result（2 seeds LOSO + 1 seed causal）。
资产：`doc/assets/gtn128_floor_20260830/`（manifest、两 seed 配对分析、causal 三臂、
逐折 record、双 cache record）。云端：AutoDL RTX 5090，源码 `SOURCE_COMMIT=5598145` +
evalhit r2/r4/r5 改动，训练环境 Python 3.12 / torch 2.8.0+cu128 / BF16 / compile。

## 0. 结论先行

1. **Gate 1（深度 vs classical）通过**：EEGNet/MS-EEGNet/full_unfold 相对
   xDAWN-RG 的 ΔAUC ≈ +0.10（p<1e-5，两 seed 一致）。深度不能省。
2. **full_unfold 晋升路线被 GTN LOSO 否定**：对 ms_flatten 两 seed
   ΔAUC −0.0041(p=0.27) / +0.0026(p=0.54)；对 EEGNet −0.0041(p=0.25) /
   **−0.0129(p=0.0017)**。BI2014a 的 +0.0109 优势在 GTN 3 导儿童数据上不复现，
   且对 EEGNet 方向反转为负。`factory.py` 默认头保持 `ms_flatten` 的决定被数据支持。
3. **GTN LOSO 零校准 hit@8 天花板 ≈ 0.60**（最好臂 0.624）。0.85 目标与
   LOSO 零校准 estimand 不兼容；缺口必须由 transfer/预训练填补（Gate 3）。
4. **Causal M=5,R=5（176 可用组）全面接近 chance**：ms_flatten AUC 0.484 /
   hit@5 0.085；full_unfold 0.526 / 0.148；K33 0.511 / 0.119。
   45 个 prefix trials 校准 258 参数 head 不够用。
5. **Causal 协议下 readout 排序反转**：full_unfold − ms_flatten
   ΔAUC=+0.0418（p=0.0017，CI [0.016,0.067]）。保留时序分辨率的 readout 在
   少样本被试内适配下占优——与 LOSO 结论相反，机制分化真实存在。
6. **Arm C（K33）非劣确认（LOSO）**：AUC 0.657 vs K65 0.660，hit@8 0.592 vs
   0.588；causal 上 K65 方向占优但不显著（p=0.26）。W2 的压缩候选在 GTN 上
   保住了 LOSO 非劣性，没有拿到 causal 优势。

## 1. 协议

- **缓存**（新增，均带 record.json + SHA attestation，见 manifest）：
  `gtn_sf128_lf2_hf30_tm-0.2_tx0.8_mean_zero.npz`（LOSO，zero-phase 契约）
  与 `..._mean_fwd.npz`（causal，`filter_phase=forward`）。
  50,562 epochs × 3 ch × 128 samples，245 subjects，candidate_chain=True。
- **协议 A（LOSO）**：245 折，`window_lr / xdawn_rg / eegnet / ms_eegnet /
  n2p3net_full_unfold`，seeds 20260828/20260829（+20260830 classical）。
  30 epochs、patience 6、batch 512、BF16、compile、fused Adam。
- **协议 B（causal）**：`causal_prefix_suffix_split` M=5,R=5，raw-sample embargo，
  **usable=176 / excluded=69**（insufficient_suffix 54、insufficient_prefix 15）。
  SubjectAdapter 线性 head（trunk 冻结），30 epochs。
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

## 3. Causal M=5,R=5（seed 20260828，n=176）

| 臂 | AUC 均值 | hit@1 | hit@5 |
|---|---:|---:|---:|
| ms_flatten | 0.4839 | 0.102 | 0.085 |
| full_unfold K65 | 0.5258 | 0.153 | 0.148 |
| full_unfold K33 | 0.5105 | — | 0.119 |

- fu − ms：**+0.0418，p=0.0017，CI [0.0162, 0.0673]**（100k sign-flip，
  被试级 bootstrap）。
- fu − K33：+0.0152（p=0.258）。
- 全部臂的 hit@5 与 chance(0.111) 同量级：**scratch + 45-trial 校准不可行**。

## 4. 与既有结论的关系（含推翻）

- `doc/prior_free_unfold_result_20260828.zh.md` 第 0 节第 3 条的保留意见
  （"对 EEGNet 仅 AUC 显著，不得写取代"）——GTN 数据进一步反转为劣势方向，
  默认头晋升正式冻结。
- `doc/research_program.zh.md` H1（total-RF 重建）：K33 在 LOSO 非劣、causal
  略差；"更短 RF 恢复局部性"在 GTN 上没有显示收益，H1 降级为搁置。
- H2（group-level objective）与 H3（source-supervised pretraining）优先级
  上升：causal scratch 已到 chance，唯一有希望的杠杆是把多被试监督先验带进
  prefix 校准，而不是继续改 readout。
- 昨日 GTN 统计能力审计的修正：v3（不预剔除）口径下 M=5,R=5 usable=176
  （v2 预剔除口径 101）；hit=0.85 时 CI 半宽 ±0.053，"可靠 85%" 仍不可证。

## 5. 已知缺口与下一步

1. **causal 协议无 fold-local artifact QC**（runner 直接用 cache 张量）；
   LOSO 有。若 causal 进入确认阶段需补齐。
2. seeds：causal 仅 1 seed；LOSO 深度 2 seeds。Gate 2 正式判定需补第 3 seed
   并预注册 contrasts（当前数字只能定方向）。
3. 下一轮（按 ROI 排序）：
   a. **Gate 3-S1**：用 BI2014a(+其他) 做 source-supervised 预训练 checkpoint，
      leave-target-out，接入 causal zero/short adaptation——这是唯一能同时
      攻击 hit 天花板与 45-trial 校准不可行的杠杆；
   b. BI2014a 候选码恢复（`event_stimulus_ids` 24 个编码待映射），把 9 选因果
      证据量从 176 提到 1701；
   c. BNCI2014_008（8 导、T 的蒙太奇）作为目标域烟测。
4. 复现命令与代码差异：evalhit r2（hit@R + exclusion ledger）、r4（adapter
   pooling 放宽 + rep-block 校准组）、r5（causal --pooling-mode/--temporal-kernel-size），
   本地 `tmp/evalhit_r1.patch`、`tmp/causal_pooling_r3.tar.gz`、
   `tmp/causal_fix_r4.tar.gz`、`tmp/causal_k33_r5.tar.gz`；全部改动需在下一次
   commit 中固化为 git 历史。
