# BI2014a 无先验完整时序展开：matched 消融结果

日期：2026-08-28
状态：immutable single-dataset exploratory result；不得单独触发默认头晋升
预注册：`doc/prior_free_unfolding.zh.md`
结果目录：`experiments/runs/bi2014a_prior_free_unfold_ablation_20260828_r1/`

## 0. 结论先行

1. **去掉固定二级池化成立。** `full_unfold` 相对 `ms_flatten` 在 AUC 与 BACC 上
   同时显著（ΔAUC=+0.0109, p=0.0056；ΔBACC=+0.0096, p=0.0099）。这是本轮唯一
   在双指标上都站得住的增益，也是预注册的主判据。
2. **额外非线性与二阶容量均无增量。** `mlp_full_unfold` 与 `quadratic_full_unfold`
   相对 `full_unfold` 都不改善，二阶在 AUC 上还显著变差（Δ=−0.0044, p=0.0458）。
   因此增益来自"保留主干时序分辨率"，不是来自容量或非线性。
3. **对 EEGNet 的优势仅存在于 AUC。** ΔAUC=+0.0056 (p=0.035)，BACC 打平
   （Δ=+0.0020, p=0.443）。**在多种子确认之前，不得写"full_unfold 取代 EEGNet
   成为总冠军"。**
4. **将 `full_unfold` 注册为 GTN 候选，不晋升项目默认头。** 单个 BI2014a
   数据集只能支持机制方向；GTN 9-choice 和独立跨数据方向一致性才有晋升权。

## 1. 协议

与预注册一致，五个模型共享全部条件：

- 数据集：BI2014a，64 被试，16 通道，61,015 条 held-out 试次，逐折 `n_test_trials`
  五个模型完全一致。
- 输入合同：128 Hz，2–30 Hz，`[-200,800) ms`，128 samples，V，mean-only baseline，
  四阶零相位 IIR。
- 评估：64 折 LOSO；内层组不交叠的验证与校准（`val_group_frac=0.1`）。
- 训练：seed 20260828，BF16，batch 512，至多 30 epochs，early-stop patience 6，
  `pos_weight=8.0`。
- QC：一套 64 折折内策略，拟合一次后五个模型复用
  （`shared_artifact_qc.reused_across_models=true`）。
- 运行时：4 个独立 fold 进程，峰值显存 469.0 MiB（EEGNet 为 450.9 MiB）。

## 2. 完整结果

| 模型 | 参数 | AUC | BACC (折间 sd) | 平均 epochs | GPU 墙钟 |
|---|---:|---:|---:|---:|---:|
| `n2p3net_full_unfold` | 1,506 | **0.74511** | **0.67741** (0.074) | 28.64 | 501 s |
| `n2p3net_mlp_full_unfold` | 3,602 | 0.74403 | 0.67529 (0.073) | 26.31 | 485 s |
| `n2p3net_quadratic_full_unfold` | 3,570 | 0.74075 | 0.67556 (0.072) | 25.17 | 455 s |
| `eegnet` | 1,490 | 0.73951 | 0.67538 (0.070) | 28.47 | 568 s |
| `ms_eegnet` (`ms_flatten`) | 1,282 | 0.73421 | 0.66778 (0.064) | 29.30 | 516 s |

## 3. 配对外层被试检验

两-sided 配对 sign-flip，100,000 次 Monte Carlo 置换，带 plus-one 修正；
置信区间为 100,000 次被试级 bootstrap 重采样。

| 对比 (A − B) | 指标 | mean delta | 95% bootstrap CI | p |
|---|---|---:|---:|---:|
| full_unfold − ms_eegnet | AUC | +0.01090 | [0.00339, 0.01816] | **0.00559** |
| full_unfold − ms_eegnet | BACC | +0.00963 | [0.00261, 0.01659] | **0.00992** |
| full_unfold − eegnet | AUC | +0.00560 | [0.00046, 0.01056] | 0.03523 |
| full_unfold − eegnet | BACC | +0.00203 | [−0.00308, 0.00714] | 0.44349 |
| mlp_full_unfold − full_unfold | AUC | −0.00108 | [−0.00483, 0.00275] | 0.58828 |
| mlp_full_unfold − full_unfold | BACC | −0.00212 | [−0.00641, 0.00208] | 0.33568 |
| quadratic_full_unfold − full_unfold | AUC | −0.00436 | [−0.00856, −0.00018] | 0.04576 |
| quadratic_full_unfold − full_unfold | BACC | −0.00184 | [−0.00695, 0.00339] | 0.48714 |
| mlp_full_unfold − eegnet | AUC | +0.00451 | [−0.00063, 0.00951] | 0.08910 |
| quadratic_full_unfold − eegnet | AUC | +0.00124 | [−0.00361, 0.00615] | 0.61932 |

## 4. 按预注册判定顺序裁决

预注册 `doc/prior_free_unfolding.zh.md` 第 5 节的判定顺序：

1. **full_unfold 对 ms_flatten 检验"去掉二级池化"的净效应** →
   AUC/BACC 双显著，**成立**。
2. **full_unfold 对 EEGNet 在近似参数预算下是否有竞争力** →
   1,506 对 1,490 参数，AUC 显著胜出，BACC 打平。判定为**在 AUC 上不劣且略优，
   BACC 上无差异**。
3. **二阶 head 只与 full_unfold 比较增量** → 增量为负且 AUC 上显著，**二阶交互
   未获支持**。
4. **MLP 机制对照** → 无增量。既然二阶与 MLP 都不改善，排除"只是容量变大"的
   解释，也排除"一般非线性有效"的解释。
5. **完整展开是否退化** → 未退化。因此进入后续候选集；软件默认仍保持
   `ms_flatten`，直到 GTN 确认。

## 5. 脆性说明（晋升前必须读完）

### 5.1 多重比较

本表含 12 个检验。按 Holm 校正，最小 p 值 0.00559 需乘 12，得 0.0671 > 0.05，
**全部检验失去显著性**。因此：

- 只有预注册的一对一主判据（`full_unfold` vs `ms_flatten`）可以不做全族校正；
- 其余对比（尤其对 EEGNet 的 AUC）只能作为方向性证据，不能单独作为晋升依据；
- 晋升最终需要**多种子确认**，而不是在这张表上继续挖掘。

### 5.2 重跑噪声底

`doc/ablation_20260828.zh.md` 那一轮与本轮配置完全相同（同 preprocessing、
QC policy、seed、epochs、batch、lr、patience、precision），但同一模型两次运行的
AUC 仍有差异：

| 模型 | head ablation 轮 | prior-free 轮 | 差值 |
|---|---:|---:|---:|
| eegnet | 0.73955 | 0.73951 | 4.0e-5 |
| ms_eegnet | 0.73484 | 0.73421 | 6.3e-4 |

两轮的 `shared_artifact_qc.fit_seconds` 分别为 155.3 s 与 153.4 s，属 CPU 计时抖动，
QC 策略本身一致。故上述差异只能来自 BF16 数值非确定性与多进程调度顺序。

**取值 6.3e-4 作为当前协议的重跑噪声底**（单次重跑量级，非标准误）。据此：
`full_unfold` − `ms_flatten` 的 +0.0109 约为噪声底的 17 倍，稳健；
`full_unfold` − `eegnet` 的 +0.0056 约为 9 倍，可信但不宽裕；
`quadratic` − `full_unfold` 的 −0.0044 约为 7 倍，可信。

该噪声底是两次运行之差，不等同于多种子标准差；**只有多种子重复才能给出正式估计**。

## 6. 训练预算扫描：性能探索，不作晋升依据

以下四个 run 在预注册消融之后执行，扫描目标是训练预算而非 readout 结构。
`doc/prior_free_unfolding.zh.md` 第 5 节明确禁止用外层测试结果回头选择 epoch，
因此**以下结果一律标记为性能探索证据，不得用于任何晋升判定**。

| run | 参数 | AUC | BACC | 平均 epochs | 采样率 |
|---|---:|---:|---:|---:|---:|
| `epoch45_patience7` | 1,506 | 0.74523 | 0.67713 | 38.30 | 128 Hz |
| `epoch45` | 1,506 | 0.74439 | 0.67709 | 35.97 | 128 Hz |
| `epoch45_patience5` | 1,506 | 0.74226 | 0.67474 | 31.84 | 128 Hz |
| `sf256_epoch32_patience6` | 2,594 | 0.74187 | 0.67410 | 24.77 | 256 Hz |

要点：

- 扫描最优（0.74523）相对预注册配置（0.74511）的 ΔAUC 仅 +0.00012，**低于第 5.2 节
  的噪声底 6.3e-4**；BACC 反而更低（0.67713 vs 0.67741）。该"提升"是噪声。
- 256 Hz 对照参数从 1,506 增至 2,594，AUC 反而更低（0.74187）。在核跨度已按
  `doc/input_contract_math.zh.md` 第 3 节匹配的条件下提高采样率没有带来收益。
- 预注册的 30-epoch 配置予以保留。若日后要正式检验训练预算，须改为内层选择并
  重新预注册。

## 7. 未决事项

1. **GTN 多种子确认**：至少 3 个 seed 的 `full_unfold`、`eegnet`、`ms_flatten`
   在 chronological 9-choice 上配对比较。继续读取 BI outer 不能恢复独立确认性。
2. **代码默认头未晋升**：`src/train/factory.py` 的 `n2p3net_pooling_mode` 仍为
   `ms_flatten`。多种子确认通过后再改。
3. **文档状态**：入口已改为“BI 探索、GTN 最终”，历史 ablation 数字保持不变。
4. **跨数据集未验证**：本轮为单数据集、单 seed 结论。GTN 与 BNCI 上的确认仍缺失。

## 8. 复现方式

配对检验用仓库自带的 `baselines.evaluate.paired_permutation_test`；
bootstrap 为被试级重采样。五个模型的 `record.json` 中 `per_fold` 逐折对齐，
可直接按折索引配对。
