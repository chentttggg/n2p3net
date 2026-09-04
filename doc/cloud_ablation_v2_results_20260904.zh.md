# 域迁移 v2 云端 development 结果分析（2026-09-04）

_状态：development/机制观察，非证据级结论。_
_运行：`domain_transfer_v2_dev64_1seed`；1 seed × 64 requested participants；BI2014a causal-v3；冻结 `ec2a9cb`；analysis SHA `ab6a3a4c…`。_

## 1. 口径

- 主指标：`planned_decision_operational_hit_rate`（requested denominator，R=8）。
- 64 个 requested participants 全部计入；其中 31 人有实际 planned decisions，33 人 `requested=0` 按合同记 0。
- 官方 `p_values_computed=false`；本报告补充 paired t / Wilcoxon 作为诊断，并以预注册 paired bootstrap interval 为主判读。
- 差异分布严重非正态（绝大多数为 0），主诊断用 Wilcoxon signed-rank。

## 2. Arm 描述统计（M, SD, median, 零值率）

| Arm | M | SD | median | zero% |
|---|---:|---:|---:|---:|
| native__zero_shot | 0.0209 | 0.0376 | 0 | 71.9 |
| native__linear_target | 0.0071 | 0.0235 | 0 | 89.1 |
| native__full_fine_no_mmd | 0.0238 | 0.0402 | 0 | 68.8 |
| ea_source__zero_shot | 0.0194 | 0.0345 | 0 | 71.9 |
| ea_source__linear_target | 0.0000 | 0.0000 | 0 | 100.0 |
| ea_source__full_fine_no_mmd | 0.0175 | 0.0325 | 0 | 73.4 |
| tsd_n2000__zero_shot | 0.0037 | 0.0195 | 0 | 95.3 |
| tsd_n2000__linear_target | 0.0016 | 0.0089 | 0 | 96.9 |
| tsd_n2000__full_fine_no_mmd | 0.0041 | 0.0204 | 0 | 95.3 |
| dense_pds_n2000__zero_shot | 0.0055 | 0.0159 | 0 | 89.1 |
| dense_pds_n2000__linear_target | 0.0041 | 0.0208 | 0 | 95.3 |
| dense_pds_n2000__full_fine_no_mmd | 0.0046 | 0.0144 | 0 | 90.6 |

## 3. 预注册配对 contrasts

### EA_source_full_fine vs native_full_fine
- n=64，非零差 7 对（全部右>左）；mean diff = −0.00629，SD = 0.01852，dz = −0.340。
- Bootstrap 95% CI [−0.01093, −0.00239]（不含 0）。
- paired t p = .00848；Wilcoxon p = .01776；Holm-corrected Wilcoxon p = .05327。
- 结论：EA(source) 在 development 单种子下略低于 native，方向与“EA 主效应改善”相反；效应幅度接近本设计可检测下限。

### TSD_n2000_linear vs native_linear
- n=64，非零差 9 对（2 左>右，7 右>左）；mean diff = −0.00551，SD = 0.02560，dz = −0.215。
- Bootstrap 95% CI [−0.01250, −0.00023]（边界刚跨过 0）。
- paired t p = .08994；Wilcoxon p = .06536；Holm p = .13073。
- 结论：无明显 TSD 优势，点估计为负。

### Dense_PDS_n2000_linear vs TSD_n2000_linear
- n=64，非零差 5 对（3 左>右，2 右>左）；mean diff = +0.00254，SD = 0.02289，dz = +0.111。
- Bootstrap 95% CI [−0.00249, +0.00858]（含 0）。
- paired t p = .37738；Wilcoxon p = .34523；Holm p = .34523。
- 结论：无区分证据。

## 4. 统计完整性

- 差异 Shapiro-Wilk 全部 p < 1e-13，正态性不满足，因此 bootstrap/Wilcoxon 为主。
- 3 个预注册 contrasts 用 Holm 校正；未做未预注册亚组。
- 灵敏度：n=64 paired、α=.05、power=.80 时可检测 dz ≥ 0.35；三个 observed |dz| 为 0.34/0.22/0.11，均处于或低于该下限。
- 所有数值仍为 1 seed 的 development 观察，不能作为 EA/TSD/Dense PDS 有效或无效的证据。
