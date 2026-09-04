# 域迁移 v2.1 松弛门槛重跑结果（2026-09-05）

_状态：development/机制观察，非证据级结论。1 seed × 64 requested participants；BI2014a causal-v3。_
_运行：domain_transfer_v2_dev64_relaxed_1seed（代码冻结 8ddab10）与 domain_transfer_v2_relaxed_car16_dev64_1seed（代码冻结 2a75407，调度修复）。_

## 1. 口径

- 
equested_hit_rate：correct / requested，incomplete/fit_failure 保留在分母（旧主口径）。
- alid_hit_rate：correct / (correct + incorrect + tie)，排除 incomplete/fit_failure。
- 本批 requested=1416，R=8 完整可判 decisions=73；**valid 口径是机制读数**，requested 口径用于保持历史可比。
- 所有结论仅 1 seed，不能作为 EA/TSD/Dense/CAR16 有效或无效的证据。

## 2. Native 计划结果（64-fold × 1 seed）

| Arm | requested participant hit | R8 valid hit | AUC mean | AUC median |
|---|---:|---:|---:|---:|
| native__zero_shot | 0.0209 | 0.3562 | 0.7091 | 0.7017 |
| native__linear_target | 0.0071 | 0.1370 | 0.6164 | 0.6297 |
| native__full_fine_no_mmd | 0.0253 | **0.4384** | **0.7308** | 0.7338 |
| calibration_native__zero_shot | 0.0164 | 0.2603 | 0.6679 | 0.6619 |
| calibration_native__linear_target | 0.0038 | 0.0685 | 0.5554 | 0.5408 |
| calibration_native__full_fine_no_mmd | 0.0165 | 0.2740 | 0.6816 | 0.6804 |
| ea_target_calibration__zero_shot | 0.0161 | 0.2603 | 0.6658 | 0.6539 |
| ea_target_calibration__linear_target | 0.0000 | 0.0000 | 0.4418 | 0.4429 |
| ea_target_calibration__full_fine_no_mmd | 0.0161 | 0.2603 | 0.6618 | 0.6502 |
| tsd_n2000__zero_shot | 0.0226 | 0.3836 | 0.7045 | 0.7092 |
| tsd_n2000__linear_target | 0.0081 | 0.1507 | 0.5827 | 0.5727 |
| tsd_n2000__full_fine_no_mmd | 0.0233 | 0.3973 | 0.7187 | 0.7256 |
| dense_pds_n2000__zero_shot | 0.0219 | 0.3699 | 0.7022 | 0.7015 |
| dense_pds_n2000__linear_target | 0.0098 | 0.1781 | 0.6081 | 0.6204 |
| dense_pds_n2000__full_fine_no_mmd | 0.0233 | 0.3973 | 0.7217 | 0.7234 |

## 3. 预注册 contrasts（requested participant metric）

| Contrast | mean diff | paired bootstrap 95% CI |
|---|---:|---|
| ea_target_calibration_full_fine vs calibration_native_full_fine | -0.00044 | [-0.00400, +0.00340] |
| ea_target_calibration_zero_shot vs calibration_native_zero_shot | -0.00033 | [-0.00335, +0.00234] |
| tsd_n2000_linear vs native_linear | +0.00100 | [-0.00252, +0.00520] |
| dense_pds_n2000_linear vs tsd_n2000_linear | +0.00171 | [-0.00265, +0.00622] |

按 requested 口径全部含 0；valid 口径下 TSD/Dense 与 native 的差距也比旧 run 小得多。

## 4. 三项修复的效果

1. **硬门槛删除**：requested 从 613 扩到 1416；R8 valid 仍为 73，但 requested 口径不再把 540 个
   ineligible decision 的 incomplete 与 valid 混为一谈，decision_curve 每层分别报告。
2. **EA-target-cal 真臂**：相对 calibration_native 无可测量优势，AUC 0.662 vs 0.682；
   1-seed 下不能判有效。
3. **TSD/Dense 两阶段训练**：旧 run TSD/dense AUC≈0.51、valid hit≈3-8%；新 run
   TSD full_fine AUC 0.719 / valid hit 39.7%，Dense PDS full_fine AUC 0.722 / valid hit 39.7%，
   已接近 native full_fine 的 0.731 / 43.8%。旧 run 的低值主要是 2000 行从零训练的欠拟合。

## 5. CAR16 预处理对照（64-fold × 1 seed）

| Arm | requested participant hit | R8 valid hit | AUC mean | AUC median |
|---|---:|---:|---:|---:|
| car16__zero_shot | 0.0173 | 0.3014 | 0.6860 | 0.6749 |
| car16__linear_target | 0.0097 | 0.1781 | 0.5936 | 0.5822 |
| car16__full_fine_no_mmd | 0.0206 | 0.3699 | 0.7053 | 0.6995 |

- 同种子下 native full_fine 为 0.4384 / 0.7308；CAR16 full_fine 为 0.3699 / 0.7053。
- 1-seed 下 CAR16 未显示预处理收益，与文献 audit 中“CAR 可能改变 P300 ERP 幅度”的谨慎结论一致。

## 6. 性能调度修复

- 问题：result 阶段 4 workers × 1 thread，只用了 25 CPU 配额中的 4 个；GPU 0%。
- 修复（commit 2a75407）：uild_dag 在未显式 
esult_threads 时按
  
esolve_cpu_threads(result_max_workers) 自动填满 cgroup CPU 配额；计划新增
  checkpoint_batch_size=16 与 
esult_max_workers=8。
- 实测：native 5-arm run 65.0 min（旧调度）；CAR16 单臂 run 15.9 min（新调度），
  checkpoint 4 chunks GPU 100%，result 阶段 8 workers × 3 threads = 24/25 CPU 配额。
- CAR16 smoke 与 full 均 completed，0 failed。

## 6.5 参与者级 valid-hit 配对照（R8，bootstrap 2000）

| Contrast | mean diff | 95% CI | n |
|---|---:|---:|---:|
| EA-target-cal full vs calibration-native full | -0.0164 | [-0.0602, +0.0215] | 31 |
| TSD linear vs native linear | -0.0341 | [-0.1459, +0.0522] | 31 |
| Dense linear vs TSD linear | +0.0932 | [-0.0046, +0.2158] | 31 |
| TSD full vs native full | -0.0272 | [-0.0694, +0.0054] | 31 |
| Dense full vs native full | -0.0261 | [-0.0699, +0.0054] | 31 |
| CAR16 full vs native full（valid hit） | -0.0541 | [-0.1559, +0.0535] | 31 |
| CAR16 full vs native full（AUC） | **-0.0255** | **[-0.0381, -0.0139]** | 64 |
| native full vs native zero（valid hit） | **+0.0541** | **[+0.0138, +0.1046]** | 31 |

- 参与者级 native full valid-hit mean=0.4576（31 人），native zero=0.4035。
- 只有 native full vs zero 与 CAR16 vs native 的 AUC 差异排除了 0；其余含 0。
- 所有 CI 为条件于冻结模型/cohort 的参与者 bootstrap，不覆盖训练随机性。

## 6.6 hit@R 曲线（native full_fine）

| R | requested | valid | correct | requested hit | valid hit |
|---|---:|---:|---:|---:|---:|
| 1 | 1416 | 1413 | 276 | 19.5% | 19.5% |
| 2 | 1416 | 964 | 308 | 21.8% | 32.0% |
| 3 | 1416 | 628 | 222 | 15.7% | 35.4% |
| 4 | 1416 | 383 | 138 | 9.8% | 36.0% |
| 5 | 1416 | 258 | 114 | 8.1% | 44.2% |
| 6 | 1416 | 176 | 76 | 5.4% | 43.2% |
| 7 | 1416 | 111 | 48 | 3.4% | 43.2% |
| 8 | 1416 | 73 | 32 | 2.3% | 43.8% |

- 随机 36 选 1 基线 = 2.78%；R1 已有 19.5%，R2-R4 升到 32-36%，R5 起 plateau 在 ~43-44%。
- R8 虽然协议主层，但 valid denominator 只有 73；R5 有 258 个 valid decisions，是更稳定的诊断层。

## 7. 统计完整性

- 官方 p_values_computed=false；bootstrap CI 为 pointwise。
- 1 seed，n=64 participants；decision 端点 73 个，功效仍低。
- 所有结论为 development/机制观察，不形成性能或产品证据。
