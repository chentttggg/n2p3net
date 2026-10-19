# 审计报告 04：迁移学习数学/统计核心（transfer & stats & 数据算子）

审计分区：`src/transfer/{exact_covariance,information_bound,noise_model,operator_lda,temporal_noise,source_objective,target_experiments}.py`、`src/stats/transfer.py`、`src/data/{finite_filter_operator,harmonic,euclidean_alignment}.py`
审计时间：2026-10-19 | 只读审计，以当前工作树为准 | 编号 04 | 报告路径 `tmp/audit/04-transfer-math-stats.md`

**方法与可复现性**：所有数值验证用 `.venv\Scripts\python.exe`（3.14.3 / numpy 2.5.2 / scipy 1.18.1 / mne 1.12.1），`PYTHONPATH=src`、`PYTHONDONTWRITEBYTECODE=1`、`python -` 读 stdin，未写任何非审计文件；脚本片段内联在对应发现中，随机种子固定。

## 结论摘要

1. **[P1]** `information_bound` 的 d² 用同一批数据估均值与协方差（plug-in），在本项目现实的 p/n 区间上偏 **14–35 倍**；其上的 `p_hit_monte_carlo`、`required_d2_for_targets`、“距最优差距”因此不可用（F-05）。
2. **[P1]** 该界假设闪光间独立，而项目自身的 `temporal_noise` 明确同一记录残差为 AR(p) 相关；正相关下证据不能按 `sqrt(F)` 累加，命中率被高估（F-06）。
3. **[P1]** `stats/hierarchical.paired_interval` 只做 percentile bootstrap 且允许 n=2；实测名义 95% 区间在 n=2/3/5/10 的覆盖率为 **49.3% / 76.1% / 84.5% / 89.3%**（F-12）。
4. **[P1]** `exact_covariance.log_likelihood_correction` 的数值由 ridge 常数决定：实测 `delta_ll=+3241.3`，其中 det 项 `−3315.8`，解析量级 `(N−rank)·C·log(1/ridge)≈5049`（F-02）。
5. **[P2]** `d2_kronecker_approx` 不是 Mahalanobis 距离：Kronecker 模型下它等于 `d²/(a_t·a_s)`，把数据整体 ×1000 就从 0.92 掉到 1e-6（F-07）。
6. **[P2]** `relative_error_trace` 对任何带通滤波器恒 ≈1（实测 1.140），把“设计性带外功率损失”误读成“AR 近似很差”（F-01）。
7. **[P2]** `noise_model.fit` 只要求 3 个 draw，却允许估计 2p 维联合协方差；没有 draws/dimension 下限，LedoitWolf 收缩目标可支配整个“算子感知”协方差（F-15）。
8. **[P2]** `operator_lda` 用两套不一致的正则判据：秩底 `1e-10·λmax` 决定 `identifiable_projector`，权重却用 `1e-3·mean(λ)` 作 ridge，被投影器判为“可识别”的方向可被压制 5 个数量级（F-16）。
9. **[P2]** 统计聚合在“任一 seed pair 缺失即整被试剔除”的完全案例策略下运行；非劣效门限只用点估计判定（F-13、F-14）。
10. **[正面核实]** `F K F^T` 与稠密 `A K A^T` 一致到 2.9e-14；EA 的 `R^{-1/2}`、DCT 保护带、SupCon 多正样本式、unbiased MMD²、被试级抽样单位均正确（F-20）。

## 发现

### F-01 [P2] `relative_error_trace` 不是“AR 近似误差”，对带通恒为 O(1)
**证据**：`src/transfer/exact_covariance.py:150-153` 把 `||F K_ar F^T − K_ar||_F / ||K_ar||_F` 解释为“当前近似很差”；`:221-222` 计算该值。
**数值验证**（250 Hz、1–30 Hz 4 阶 Butterworth SOS，AR(2)=[0.5,0.3]，N=128）：`rel_err = 1.140`，偏差超过原协方差本身。带通按设计移除带外功率，差异主要来自“被滤掉的功率”，任何窄带滤波器都会给出 `≈1`。
**影响**：该列不可跨滤波器/被试比较；越窄的带通看起来越“差”，会误导滤波器设计评审。
**建议**：主指标改用 `whitening_gain()` 的奇异值谱，或 `||L_ar^{-1}(F K F^T)L_ar^{-T} − I||`；若保留 Frobenius 比率，分母改为 `||F K F^T||` 并注明“含设计性带外损失”。
**验证方式**：h_freq 30→15 Hz，rel_err 单调升而 whitening_gain 不变差。

### F-02 [P1] `log_likelihood_correction` 由 ridge 常数决定，不是数据似然
**证据**：`src/transfer/exact_covariance.py:396-440` 用 `log det(K_prop + ridge I)` 的 Cholesky 对角做行列式项，`ridge = max_eig·1e-12`（`:243`）；带通使 `K_prop` 秩亏 `N−rank` 维（`:237-239`）。
**数值验证**（N=128，rank=72，ridge=1.63e-10，C=4，残差取合法值 r=A·N(0,I)）：`quad_exact=174.7, quad_approx=25.6`，`logdet_exact=−784.1, logdet_approx=44.8`，det 项 `C(lp−la)=−3315.8`，quad 项 `−74.5`，合计 `delta_ll=+3241.3`。解析 `(N−rank)·C·log(1/ridge)≈5049` 与 det 项同量级 ⇒ ridge 1e-12→1e-10 会使 delta_ll 移动约 2500 nats。
**影响**：“精确传播模型比 AR 近似更/更不似然”的任何结论都不可用，符号由常数而非模型决定。
**建议**：在 `range(A)` 上做伪行列式（仅保留特征值 + 秩修正），或输出 ridge∈{1e-8,1e-10,1e-12} 的敏感性曲线；docstring 注明“非可比较边际似然”。
**验证方式**：固定 r 扫 ridge 三个量级，观察 delta_ll 变化远大于模型差异。

### F-03 [P2] `byte_budget` 容量说明算错，峰值内存被低估 3–4 倍
**证据**：`src/transfer/exact_covariance.py:185-188` 称 64 MiB “handles N up to ~2900”；判据 `2·8·N² ≤ 64 MiB` 实际给出 **N ≤ 2048**（`np.sqrt(64·1024²/16)=2048`）。`compute` 同时持有 `K_ar, FK, K_prop, eigh 工作区, eigenvectors, L, K_prop_regularized`，峰值约 6–8 个 N×N。
**影响**：N≈2000 时真实峰值约 256–320 MiB，预算却只有 64 MiB，存在 OOM 风险；按 docstring 规划的 N≈2800 会被直接拒绝。
**建议**：预算按 `8·N²·8` 计或分块；docstring 改为 N≤1400–2048 并注明峰值系数。

### F-04 [P3] 冻结 dataclass 的派生字段未声明，破坏相等性/序列化契约
**证据**：`src/transfer/exact_covariance.py:133-164` 只声明 9 个字段，`:292-296` 用 `object.__setattr__` 额外写入 `eigenvalues/eigenvectors/numerical_rank/null_dimension/regularization_ridge`；`assert_integrity`（`:305-313`）只覆盖 K_prop/K_ar/L，不覆盖特征对，而 `spectral_comparison` 直接报告它们。
**影响**：两实例 `==` 会逐字段比较 ndarray 并抛 “truth value of an array is ambiguous”；`asdict` 丢字段；特征对被篡改时完整性校验仍通过。
**建议**：显式声明 `compare=False` 字段，并把特征对摘要纳入 integrity payload。

### F-05 [P1] 信息界建立在大幅上偏的 plug-in d² 上（p/n 稍大即 10–35 倍）
**证据**：`src/transfer/information_bound.py:87-178` 用同一批数据估 `mu1,mu0` 与 `Sigma`（LedoitWolf），再算 `d²=δ^T Σ^{-1} δ`；`:515-518` 直接送入 bound。
**数值验证**（Σ=I，真 d²=0.5，40 次重复，两类等样本）：p=20,n=200 → 0.699（1.40×）；p=200,n=60 → 7.155（**14.3×**）；p=400,n=50 → 17.28（**34.6×**）；p=800,n=100 → 16.64（**33.3×**）。这正是 T×C 特征（如 100×8=800）与几十个 epoch 的现实区间。
**影响**：`p_hit_monte_carlo`、`required_d2_for_targets`、`efficiency` 与“距信息论最优”全部被系统性抬高；d² 越不可信报告越乐观。这是带未知上偏的 plug-in，不是界。
**建议**：交叉拟合 d²（Σ 与 δ 分离，K-fold 或 leave-one-selection-out），或偏差修正 `d²−p/n_eff` 量级；返回 `d2_bias_indicator=p/n` 并据此标“不可用”；禁止称其为 bound。
**验证方式**：同一数据分半，train 估 Σ、test 估 δ，比较 d² 下降幅度。

### F-06 [P1] “闪光间独立”假设与项目自身 AR 噪声模型冲突，界被高估
**证据**：`src/transfer/information_bound.py:39-49`（独立闪光、等协方差、已知参数）、`:296-306`（每 flash 抽 iid `N(0,d2)` 后 `T=noise@A`）；而 `src/transfer/temporal_noise.py:1-13` 明确同一记录残差为 AR(p)，行/列范式的闪光在同一段连续记录内、间隔远小于 epoch 长度（具体重叠需查 `configs/` 与输入契约）。
**影响**：正相关噪声下跨 flash 证据不能按 `sqrt(F)` 累加，MC 与 union bound 都高估命中率，可能把“需改特征/解码”的问题误判为“已接近最优”。
**建议**：MC 中按拟合 AR 协方差着色（`noise = L_AR @ iid`，或按 flash 间隔取相关矩阵），同时报告 iid 与 AR 两版；docstring 明确 iid 版是乐观参考而非界。

### F-07 [P2] `d2_kronecker_approx` 不是 Mahalanobis 距离，随物理单位整体缩放
**证据**：`src/transfer/information_bound.py:402-435` 分别用“按通道合并的时间协方差”和“按时间合并的空间协方差”白化后取 Frobenius 范数平方。Kronecker 模型下边际协方差为 `K·tr(S)/C` 与 `S·tr(K)/T`，故该范数 `= d²/(a_t·a_s)`（`a_t=tr(K)/T, a_s=tr(S)/C`）。
**数值验证**（T=40,C=6,AR(1)=0.8，真 d²=2.0）：解析边际白化范数 `1.0286`，与理论 `d²/(a_t a_s)=1.0286` **完全吻合**；代码估计 0.9196；同一数据 ×1000 后 `d2_kronecker_approx → 1e-6`，而 `mahalanobis_snr` 的 d² 保持 1.986。`d2_fraction`（比值）不受影响。
**影响**：绝对数值不可跨被试/跨单位比较（V vs µV 差 10^12）；若用于“保留了多少 d²”会得出完全错误的量级。
**建议**：重命名为 `whitened_mean_frobenius2`，或返回前乘回 `a_t·a_s`；`d2_fraction` 保留。

### F-08 [P2] efficiency 的文档定义与代码相反，且比例未截断
**证据**：`src/transfer/information_bound.py:489` 写 “Efficiency gap: (empirical_hit − bound)/(1 − bound)”；代码 `:535-546` 实算 `gap = best_possible − empirical`、`relative_gap_pct = 100·gap/(1−empirical)`；`fraction_of_room_used` 只做 `max(0,·)`，经验命中率高于 MC 时可 >1。
**影响**：读者会按相反符号理解“差距”；>1 的 “room used” 无含义。
**建议**：统一为 `(bound−emp)/(bound−chance)` 并 clamp 到 [0,1]，docstring 同步，并给 bootstrap CI（见 F-05）。

### F-09 [P3] 下界只用候选 0 的配对计数，与 MC 的全候选平均不一致
**证据**：`src/transfer/information_bound.py:286-287` 取 `base_diff_counts=pair_diff[0,1:]`；`:313-315` 的 union bound 只从候选 0 视角求 `per_true_fail_bound`；`:301` 的 MC 对全部 `true_candidate` 平均。F 不被 2L 整除时（`:204-215` 把余数给前几行）各行/列激活数不等。另 `:287` 的 `unique_counts` 赋值后未使用。
**影响**：余数非零的合成网格会给出与 MC 不可比的下界。
**建议**：对全部 true_candidate 求 `per_true_fail_bound` 再平均，删除未用变量。

### F-10 [P2] 数学模块零生产调用，且 11 个分区文件中 8 个未入库
**证据**：`git ls-files` 显示 `exact_covariance.py / information_bound.py / noise_model.py / operator_lda.py / temporal_noise.py / source_objective.py / finite_filter_operator.py / harmonic.py` 全为 `??`；`git grep` 显示 `ExactPropagatedCovariance`、`InformationBoundReport` 只出现在自身与 `tests/transfer/test_exact_covariance_and_bounds.py`；`TemporalNoise` 也仅被 `exact_covariance` 引用。
**影响**：公式错误不会污染生产结果（严重级别因此受限），但会污染研究结论与文档；未入库意味着无法回溯“某次报告用了哪版公式”，`_IMPLEMENTATION_SHA256` 只保证同文件内容。
**建议**：纳入 `src/` 入库并在技术文档标注 “diagnostic, no production import”，或移到 `experiments/`；报告必须绑定 commit。

### F-11 [P3] 推导文档与结论常量不一致
**证据**：`src/transfer/information_bound.py:8` 写 `I=(K/2)log2(1+d²/K_noise)`；K 次独立观测的正确形式是 `(1/2)log2(1+K·d²)`（仅 SNR→0 时同阶，前者对 K>1 系统性偏大）。`:26-27` 先说“等于 KL 的一半”，随后给出 `(1/8)d²`；等协方差高斯下 `KL=d²/2`，其一半是 `d²/4`，`d²/8` 只是小 SNR 的 MI 近似。代码用的 `1/4·d²·n_diff` 本身正确。
**影响**：被当作“第一性原理推导”引用时无法复现常数，且掩盖了真正前提（小 d²、等先验、等协方差、独立闪光）。
**建议**：改写为精确式 + 明确“小 d² 近似”与三条前提。

### F-12 [P1] 95% bootstrap 区间在 n=2–10 严重欠覆盖，却被当作推断区间
**证据**：`src/stats/hierarchical.py:47` 只要求 `len(values) ≥ 2`；`:62-71` 是纯 percentile bootstrap；`src/stats/transfer.py:190-192` 在 `len(values)>1` 时就调用并写入 `paired_interval`。
**数值验证**（真值 μ=1，σ=1，1500 次重复，每次 1000 重采样，seed=17）：n=2 覆盖 **0.493**；n=3 **0.761**；n=5 **0.845**；n=10 **0.893**；n=25 **0.928**。n=2 时 bootstrap 均值只有 3 个支撑点，区间就是 `[min,max]`（示例 `[-0.05,0.02]`）。
**影响**：报告中的 “95% CI” 在个位数被试时实际覆盖率可低至 49%，任何“显著/不显著”的读法都不成立；这是本分区最高风险的统计问题。
**建议**：改用 BCa 或 bootstrap-t；n<5 时不输出区间而输出 `insufficient_units`；`PairedInterval` 增加 `method` 与 `min_units_policy` 字段，报告必须带 n 与真实覆盖率说明。
**验证方式**：上述覆盖模拟可直接复现（12 行脚本）。

### F-13 [P2] 完全案例剔除被试，缺少可用配对敏感性分析
**证据**：`src/stats/transfer.py:162-174`：`seeds` 中任一 pair 缺失或非 `complete_pair`，该被试 `difference=None`、`status=incomplete_seed_grid`，随后 `_summary` 只统计 `complete_pair`（`:181`），仅在 `n_requested` 与 `failed_or_missing_seed_pairs` 中留痕。
**影响**：9 个 seed pair 全成对的要求会成批剔除被试；若失败与条件相关（某 arm 更慢/更易崩），估计量有选择偏差，且剔除数量不进入主表。
**建议**：增加“可用配对均值”敏感性估计（每被试对可用 pair 取均值并报告可用 pair 数分布），与主估计并列；主表直接给出被剔除被试数与比例。
**验证方式**：构造 10 被试、每人随机缺 1 个 pair 的合成 rows，比较两种估计之差。

### F-14 [P2] 非劣效门限只用点估计
**证据**：`src/stats/transfer.py:200` `replacement_decline_gate_passed = bool(values.mean() >= -0.01)`；`:197` 的 `decline_over_0_01_fraction` 只是描述比例，无区间。
**影响**：小样本下 `mean ≥ −0.01` 与 `< −0.01` 的差异可能完全在噪声内，却输出布尔 “gate passed”，容易被下游当作决策依据。
**建议**：改为单侧非劣效检验/区间下界 ≥ −0.01，或至少同时输出区间与 n；保留布尔值时改名 `point_estimate_gate`。

### F-15 [P2] 联合噪声模型 3 个 draw 即可估 2p 维协方差，无 draws/dimension 下限
**证据**：`src/transfer/noise_model.py:152-153` 仅要求 `len(raw) ≥ 3`；`:126-127` 的 provenance 也只校验 `n_draws ≥ 3`；`:167-173` 用 LedoitWolf，收缩强度随 `p/n` 增大自动趋近 1。对照 `src/transfer/temporal_noise.py:287-289` 明确要求 `minimum_samples_per_parameter`（≥2）与 `max(32, k·order)` 下限。
**影响**：n=3、p=200 时协方差几乎等于 `μI`，所谓“算子感知”LDA 实际用的是各向同性先验；`shrinkage` 虽记入 provenance，但没有 `n_draws/p` 比值与“是否数据可识别”判定，读者无法分辨。
**建议**：加 `n_draws ≥ max(10, ceil(2·dim))` 或显式 `insufficient_draws` 标记；provenance 增加 `n_draws_per_dimension` 与经验谱条件数；`shrinkage > 0.9` 时在 LDA 侧警告。

### F-16 [P2] operator_lda 双正则判据不一致，且缺少噪声模型拟合优度检查
**证据**：`src/transfer/operator_lda.py:58` 用 `values > λmax·1e-10` 定义可识别支持与 `identifiable_projector`；`:66-67` 却用 `ridge_relative·mean(λ_keep)`（默认 1e-3）作 ridge。于是被投影器判为“可识别”的方向（如 1e-9·λmax）会被 ridge 压制约 `1e-3·mean(λ)/λ ≈ 1e5` 倍；`:137-141` 的 `_validate_support` 用前者，实际权重空间由后者决定。此外没有检查投影协方差与校准特征散度是否一致（`w^T S_emp w` vs `w^T Σ w`）。
**影响**：`identifiable_projector` 高估真正参与判别的子空间，支持集校验放过了被正则化清零的方向；若 `raw_operator/state_operator` 单位或尺度错，权重方向错却无任何断言。
**建议**：用同一判据（建议 `λ > ridge`）同时定义 `keep` 与 `projector`；增加 `noise_fit_ratio=(w^T S_emp w)/(w^T Σ w)` 诊断，偏离 [0.5,2] 报警。
**验证方式**：构造 λ 跨 1e-9…1 的协方差，比较 `keep.sum()` 与权重中 `|w_i|>tol` 的方向数。

### F-17 [P2] `_conditional_mmd` 逐 cell 宏平均 + 无带宽下限
**证据**：`src/transfer/source_objective.py:388-421`：每个 (domain pair, class) cell 独立算 unbiased MMD²，`active = n,m ≥ min_cell_count`，`value=(terms·active).sum()/active.sum()`，2 样本 cell 与 200 样本 cell 等权；`:399-411` 每 cell 用自身中位数带宽，被平均的是不可比的 MMD²；`sigma` 无下限（`:409` 仅在 counts=0 时置 1），近重复特征会让 `exp(-d²/2σ²)` 下溢、梯度消失。
**影响**：小 cell 主导梯度、方差大；跨 cell 的“MMD 标量”不可解释为单一分布差异。
**建议**：按 cell 样本量加权（或每类做一次合并 MMD）；`sigma=max(median, eps·scale)`；diagnostics 报告 `min_sigma` 与 `active_cell_sizes`。

### F-18 [P3] harmonic 频率搜索的容差与代价不匹配
**证据**：`src/data/harmonic.py:113` 用 2 次 IRLS 的 `median|r|` 选网格点；`:124-126` `minimize_scalar(..., xatol=1e-10)`，而每次目标函数求值都重跑 `robust_iterations` 次全窗口 `lstsq`（`:118-121`）；物理分辨率约 `1/T`。`:138` 只报告 `frequency_at_search_edge`，最优点落在边界时仍可能 `applied=True`。
**影响**：约 30–50 次冗余最小二乘（长窗口下秒级开销），1e-10 精度远超目标函数平滑尺度，属无效精度；边界解未被拒绝。
**建议**：`xatol≈0.01/T` 或抛物线插值；`frequency_at_search_edge` 为真时拒绝该窗或扩大搜索带。

### F-19 [P3] 激活矩阵的非方阵回退把 group 当 lines 报告
**证据**：`src/transfer/information_bound.py:218-225` 回退分支 `return A, group`，而 `group=round(sqrt(Q))` 不是行/列数；`:335-357` 用它拼 `"Balanced {lines}-line row/column grid"`。
**影响**：非完全平方 Q 时解释串与事实不符（仅文字，不影响数值）。
**建议**：回退分支返回 `None` 并让 interpretation 分支处理。

### F-20 [正面核实，无缺陷] 以下公式与实现经核对/数值验证正确
- `exact_covariance` 的 AR 自协方差：AR(1) 与解析 `σ²a^k/(1−a²)` 相对误差 2.5e-16；AR(2) 的 `r0,r1,r2` 与 Yule–Walker 解析值一致；Toeplitz 矩阵对称且最小特征值 >0。
- `operator.apply` 是**线性**映射（加性误差 3.7e-14、齐次性 1.3e-13；`preprocess.py:441-442` 的 odd reflection `2x[0]−x[i]` 是齐次的，无仿射常数项），故 `F K F^T` 合法；`compute()` 与稠密 `A K A^T` 的相对 Frobenius 误差 **2.9e-14**。
- `whiten()` 的白化协方差在保留子空间上为 I（偏差仅 ridge/λ 量级），PSD 与秩亏处理逻辑正确。
- `stats/transfer.py` 以**被试**为抽样单位、先在每个被试内对 seed pair 取均值（`:155-175`）再 bootstrap，避免把 seed 当独立重复的伪重复；`sampling_unit` 与 `limitation` 如实声明“不覆盖新研究/源重采样/训练过程不确定性”（`:251-253`）；缺失结果的 `[Σw⁻, Σw⁺]` 敏感性界（`:203-215, 256-264`）方向正确并标注 `is_confidence_interval: False`。
- `standard_training_contrasts` 的全部权重和为 0（含 E1 的 `(2c−1)/2^{3−|axes|}` 主效应/交互），是合法的差异对比。
- `euclidean_alignment._reference_operator`：`covariance=Σ_n X_nX_n^T/(N·T)`、`scale=tr/C`、`R=V diag(λ·scale)V^T`、`R^{-1/2}=V diag(1/sqrt(λ·scale))V^T` 与标准 EA 一致；收缩在归一化坐标内、特征值下限相对化、条件数有上限、`from_record` 重算校验；`VALID_EA_SCOPES={"source_rows","target_calibration","unlabelled_buffer","per_domain"}` **不含** evaluation 作用域，未见把 EA 拟合到被评测 trials 的泄漏路径。
- `source_objective._supcon`：`logsumexp(candidates) − mean(s_pos)` 是多正样本 SupCon 的正确 `L_out` 形式，负样本排除同域同人同类、正样本要求跨域跨人，掩码无自环。
- `source_objective._conditional_mmd`：`n(n−1)` 分母配合 `off_diagonal` 是 unbiased MMD²，带宽用 detached 中位数（不传梯度）合理，`negative_mmd_terms` 如实记录无偏估计可为负。
- `temporal_noise` 的 Burg 反射系数符号与 `coefficients=−polynomial` 约定一致（`x_t=Σ a_j x_{t−j}+ε`），`_reflections` 的 step-down 与 Levinson 公式一致，`_autocovariance_sequence` 复用同一符号约定，跨模块无符号冲突。

## 未能验证与存疑

1. **未读**：`src/transfer/target_experiments.py`（53 KB）只做符号级浏览（`build_target_tasks/execute_target_tasks/freeze_target_batch` 等 7 个入口），未逐行核对其预算/聚合算术；`source_objective.py` 只审了 `SourceObjective` 类（`:242-431`），未审 `validate_source_objective_contract`（`:126-241`）与 recipes。
2. **无法验证**：d² 上偏对**最终报告数字**的实际影响，需要真实校准集（`artifacts/`、`experiments/runs/` 未读，42 GB 结果不应在本分区读取）；F-05/F-06 是方法层面成立，具体倍数取决于实际 T、C、epoch 数。
3. **存疑**：`d²` 是否在别处以“每 flash”之外的口径被复用（`grep` 仅见 `mahalanobis_snr` 与报告类内部使用）；若下游把 `d2_effective_per_selection=20·d²` 当成功率上界，误差会再放大一个量级。
4. **存疑**：`FiniteFilterOperator.padding_limit` 与生产 `filter_continuous(mne_reflect_limited)` 的 padding 是否一致未核实；若生产在连续记录上滤波、`exact_covariance` 在 epoch 上重滤，则 `F K F^T` 描述的是“逐 epoch 重滤”模型而非生产管线（该模块仅诊断用途，故未升级为发现）。
5. **未测**：`byte_budget` 的真实峰值内存未用 `tracemalloc` 实测（F-03 的 6–8× 系数按代码中同时存活的 N×N 数组推得）。
6. **未跑**：`tests/transfer/test_exact_covariance_and_bounds.py` 只读了文件名与导入，未执行（避免与 Lead 的测试分区重复）；`ruff check` 未在分区上运行，`unique_counts` 未使用等仅靠人工阅读判定，建议 Lead 统一跑一次 `ruff check src/transfer src/stats src/data`。
7. **未逐条比对**：`doc/input_contract_math.zh.md` 正文（时间预算内只通过代码 docstring 与 `temporal_noise`/`euclidean_alignment` 的滤波/对齐约定间接核对），这是本报告最主要的方法缺口。

## 已检查范围

| 文件 | 入库状态 | 方式 |
|---|---|---|
| `src/transfer/exact_covariance.py` (440 行) | 未跟踪 `??` | 全文精读 + 稠密矩阵数值验证（线性性、`A K A^T`、白化、log-lik） |
| `src/transfer/information_bound.py` (600 行) | 未跟踪 `??` | 全文精读 + d² 偏差模拟、Kronecker 尺度解析/数值、MC 结构核对 |
| `src/transfer/temporal_noise.py` (457 行) | 未跟踪 `??` | 全文精读（Burg/反射系数/白化/ledger 校验） |
| `src/transfer/noise_model.py` (241 行) | 未跟踪 `??` | 关键段精读（`validate_covariance`、`fit`、`project`） |
| `src/transfer/operator_lda.py` (197 行) | 未跟踪 `??` | 全文精读 |
| `src/transfer/source_objective.py` | 未跟踪 `??` | `SourceObjective.loss/_supcon/_conditional_mmd` 精读（`:242-431`） |
| `src/stats/transfer.py` (≈700 行) | 已跟踪 ` M` | `:1-409` 精读（权重表、seed 解析、被试聚合、bootstrap、缺失界） |
| `src/stats/hierarchical.py` (84 行) | 已跟踪 | 全文精读 + 覆盖率模拟 |
| `src/data/finite_filter_operator.py` (91 行) | 未跟踪 `??` | 全文精读 + 线性性数值验证 |
| `src/data/euclidean_alignment.py` (≈250 行) | 已跟踪 ` M` | `:20-211` 精读（`_reference_operator`、state 校验、fit） |
| `src/data/harmonic.py` (≈200 行) | 未跟踪 `??` | `:75-174` 精读（IRLS 谐波拟合与频率搜索） |
| `src/data/preprocess.py` 的 `filter_bidirectional_segment` | 已跟踪 ` M` | `:428-457` 精读（仅为核实 `apply` 的线性性） |
| `src/transfer/target_experiments.py` (53 KB) | 已跟踪 ` M` | **仅符号级浏览**，未逐行审计（见“未能验证” 1） |

**未纳入本报告**：`src/transfer/{subject_adapter,decision_loss,calibrated,calibration_candidates,checkpoint,quality,...}`、`src/stats/` 其他文件、`experiments/`、`formal/`。预计剩余风险集中在 `target_experiments` 的预算算术与 `decision_loss` 的候选集损失。
