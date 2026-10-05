# 其他 agent 深度科研实现核验（2026-10-05）

本次核验覆盖 `exact_covariance.py`、`information_bound.py`、E2 计划/runner 及其 23 项测试。范围是研究代码与合同审查；没有启动 E2 公共数据矩阵，没有改写旧结果或生产模型。

## 已确认成立的部分

- `ar_toeplitz_covariance` 的 AR(1)/AR(2)/AR(3) stationary 协方差与独立长序列经验自相关一致；AR(2) 不再使用未求解 lag 的自引用递归。
- `ExactPropagatedCovariance` 的 `K_prop=F K_AR Fᵀ` 与显式 dense operator 相等。有限双向滤波后的矩阵可以数值半正定并有近零特征方向；模块记录 numerical rank，不把原始矩阵宣称为满秩。
- 传播计算的时间复杂度和小网格内存门符合研究诊断用途；未接入生产调用链。
- 手动 shrinkage 已改为收缩到样本平均方差 `trace(Sigma)/p`，因此 d² 在 V→µV 单位变换下保持不变。
- E2 runner 的 gauge/unit 数学门在读取 MAT 前执行，并保留错误 ridge 的负对照。

## 已发现并修正的问题

1. E2 计划使用了 `family="toeplitz"`、`family="xdawn"`，但 `SelectionERP` 的正式值是 `toeplitz_lda`、`xdawn_erp`；xDAWN 还要求 `baseline_mode="mean_only"`。原计划的 96 个 anchor cell 会直接失败。计划已修正，并用 `SelectionERP._validate_declaration` 独立核验。
2. `mahalanobis_snr(..., shrinkage=s)` 原先向单位阵收缩。EEG 单位从 V 改成 µV 时，固定 `s>0` 会改变 d²；已改为向平均方差球收缩，并新增单位反例。
3. 精确协方差函数原先接受 `startup_mode="conditional_prefix"`，却仍构造 stationary Toeplitz 协方差；这会把条件 likelihood 当成边际 likelihood。现在明确拒绝该模式，要求先构造对应条件创新协方差。
4. 信息函数原先只以 candidate 0 的激活数/差值计算不均匀 flash 网格。现在 Monte Carlo 和 pairwise union 都对每个真实候选平均，避免 `F` 不能整除行列线数时偏置 candidate 0。

## 不应从这些模块宣称的结论

- `p_hit_monte_carlo` 是声明的 equal-covariance、independent-flash、known-parameter Gaussian 模型下的命中**估计**，不是任意 decoder 的信息论上界，也不是 BNCI/成人效能上限。
- `p_hit_lower_bound` 是该声明模型的 pairwise union lower bound；它不覆盖参数估计误差、跨 flash 协方差、刺激重叠、滤波有限支持或候选先验失配。
- `F K Fᵀ` 的 dense reference 目前使用相对 ridge 对奇异矩阵做数值白化；它不是已经接入 `OperatorGLS` 的完整 exact likelihood。null space 的物理处理、通道协方差 Kronecker 传播和候选核后验仍需单独设计。
- `spectral_discriminability` 采用 time/space Kronecker shrinkage 近似，不能当成全时空协方差的精确频谱分解。
- 旧 attachment 中“修正后 MC 与 0.771 一致”“Frobenius 误差 9–21×”属于特定参数/执行产物，不能从单元测试推广成新效能或普遍机制结论。

## 验收记录

- `tests/transfer/test_exact_covariance_and_bounds.py`: 23 passed。
- 与 `tests/transfer/test_temporal_noise.py` 合并：43 passed，1 条既有 SciPy 病态矩阵 warning，无失败。
- Ruff：`src/transfer/exact_covariance.py`、`src/transfer/information_bound.py`、相关测试及 E2 runner 通过。
- 独立检查 E2 两个 anchor 声明通过；未执行 E2 数据训练。

下一 agent 若继续工作，应先把 exact covariance 的 PSD/null-space 似然合同和通道协方差接线写成独立反例，再考虑将其消费到 E2；不要直接用当前 `whiten()` 结果替换 `OperatorGLS` 的 whitening，也不要把模型命中估计写成理论上界。
