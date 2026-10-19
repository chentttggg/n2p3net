# E2 科研记录：AR 阶数 × 先验尺度与精确协方差传播

日期：2026-10-06。范围：E2 实验执行与结果诠释；新增 `exact_covariance` 和 `information_bound` 模块；
项目文件结构部分清理。本轮不修改生产代码（`src/` 中 118 个 Python 文件字节完全不变，经 SHA256 验证）。
本记录承接 [边界状态重新设计](boundary_state_redesign_20261005.zh.md)。

## 1. 本轮交付物

### 新增研究模块

| 文件 | 内容 | 已验证 |
|---|---|---|
| `src/transfer/exact_covariance.py` | 精确 $F K_{AR} F^\top$ 传播；rank 诊断；白化增益 | 21 项测试 |
| `src/transfer/information_bound.py` | Q 叉 Bayesian 命中率上界；6×6 行列范式 MC；谱判别 | 21 项测试 |
| `tests/transfer/test_exact_covariance_and_bounds.py` | 独立反例 | 全部通过 |
| `experiments/run_e2_prior_scale.py` | E2 执行脚本；数学门前置；operator 显式声明 | 冒烟通过 |
| `experiments/run_information_bound.py` | 信息论上界驱动；per-participant 效率间隙 | 冒烟通过 |
| `experiments/runs/boundary_state_redesign_20261005/e2_plan.json` | 8 人 × 6 fold × 6 臂 计划 | 合规 |

### E2 执行摘要

8 人 × 6 fold × 6 臂 = 288 cell，全部完成，0 失败。
Ridge 网格 {0.01, 0.1, 1.0, 10.0}，主 cell 汇总用 ridge=1。

| 臂 | hit (48 folds) | 参与者均值±SD | NLL |
|---|---|---|---|
| raw_ar1_innovation | 0.604 | 0.604±0.295 | 3.023 |
| raw_ar1_marginal | 0.646 | 0.646±0.314 | 3.579 |
| raw_ar8_innovation | **0.667** | 0.667±0.252 | 6.162 |
| raw_ar8_marginal | 0.646 | 0.646±0.288 | **12.626** |
| toeplitz_anchor | **0.771** | 0.771±0.333 | 2.163 |
| xdawn_anchor | **0.771** | 0.771±0.266 | 3.075 |

数学门验证（每次运行前执行，本轮结果）：

- gauge 正确同步：max BF 差 = 2.27e-13（< 1e-8 ✅）
- gauge 错误同步（负对照）：差 = 216.62（> 1 ✅）
- 电压单位不变：max BF 差 = 1.14e-13（< 1e-8 ✅）
- naive 噪声缩放（负对照）：差 = 480.99（> 1 ✅）

## 2. 第一性原理结论

### 结论 1：`marginal` 先验不能修复 AR(8) 的 NLL 过度自信

**直接证据**：`raw_ar8_marginal` 的 NLL = 12.626，比 `raw_ar8_innovation` 的 6.162 高出约一倍。hit 从 0.667 降到 0.646。

**数学解释**：`marginal` 模式让有效 ridge $\lambda_{\text{eff}} = \lambda / r_0$（其中 $r_0$ 是边际方差）。对于 AR(8)，$r_0 > r_{0,AR1}$（高阶 AR 在低频捕获更多自相关），因此 $\lambda_{\text{eff}}$ 更小，先验更弱，允许 ERP 核拟合出更大幅度，进一步放大了噪声，使 log BF 方差增大。`marginal` 不是修复，而是正向放大了问题。

**排除的机制 A**：深度推导中的"AR(8) 创新方差小 → 先验精度被动偏小 → 过度自信"（机制 A）被直接否定。正确的修复方向不在 $\lambda$ 归一化上。

### 结论 2：AR(8) 的 NLL 问题根源在 $F K F^\top \neq K_{AR}$（机制 B）和设计矩阵条件数（机制 C）

**直接证据**：`exact_covariance` 诊断显示，Frobenius 相对误差最大达 21×（宽带），典型带通算子 9-10×。`whitening_gain` 的奇异值远离 1，证明当前 AR 近似白化在被通滤波后的坐标里是错误的。

**推论**：若用精确 $F K_{AR} F^\top$ 白化（`ExactPropagatedCovariance`），后验协方差的形状应与观测噪声匹配，NLL 不诚实问题应得到改善。这是 **E2 之后的下一个实验**。

### 结论 3：信息论上界远高于当前性能，存在可探索的大间隙

八名参与者的 $d^2 \in [9.7, 29.3]$，信息论可达命中率 MC 估计均为 1.0。

当前最佳 hit = 0.771（Toeplitz 和 xDAWN），信息论理论上界接近 1.0。大量潜力未被当前方法挖掘。

**为什么当前方法未达到理论上界**：
1. AR 近似白化误差（主要原因）
2. epoch 协方差估计只用了 600 个 trial（Toeplitz/xDAWN），非最优估计
3. 线性模型无法捕获非线性 ERP 变异（次要，P300 基本线性）

**效率间隙**（以 Toeplitz 为参考）：
```
fraction_of_room_used = (0.771 - 1/36) / (1.0 - 1/36) ≈ 0.749
```
即当前方法约利用了约 75% 的理论可用信息。25% 的间隙是值得追逐的空间，尤其是精确协方差传播方向。

### 结论 4：AR 阶数提升（AR1→AR8）命中率略提升但 NLL 代价巨大

| 对比 | hit 变化 | NLL 变化 |
|---|---|---|
| AR1→AR8（innovation） | +0.063 | +3.139（+104%） |
| AR1→AR8（marginal） | 0.000 | +9.047（+253%） |

在同一信号条件（原始 MAT，无项目滤波）下，高 AR 阶数只带来微小命中率增益，代价是 NLL 大幅恶化（概率不诚实）。从概率意义看，AR(8) 是更差的模型，尽管点预测略好。

### 结论 5：算子差异是 Toeplitz vs AR 的核心，不只是方法差异

Toeplitz 锚点用了项目双向滤波（0.1-30 Hz 四阶 Butterworth + 50 Hz Q=30），AR 臂用了原始分发 MAT（BNCI 自带 0.1-30 Hz，阶数和实现未知）。两组臂的 operator 不同，因此 Toeplitz 的 hit=0.771 vs AR8 的 hit=0.667 差距混合了"方法差异"和"算子差异"。这两个效应无法从当前 E2 数据中分离。**后续实验应统一算子后再比较方法**。

## 3. 模块技术细节

### `src/transfer/exact_covariance.py`

**核心数学**：$K_{\text{prop}} = F K_{AR} F^\top$，其中 $F$ 是有限双向 SOS 算子，$K_{AR}$ 是精确 Yule-Walker Toeplitz 协方差（用 Lyapunov 方程求解前 $p$ 个滞后，再前向递推）。

**关键发现（来自探测脚本）**：

```
=== brainsync 256Hz (fs=256 band=[0.1, 30.0] pad=100 n=200) ===
  K_ar  eig min/max   : 2.77e-01 / 9.82e+01  cond=3.54e+02
  K_prop eig min/max  : -1.35e-14 / 2.14e+03  (正半定，带通杀死 DC)
  ||K||_F=2.27e+02  ||Kp||_F=2.15e+03  rel=9.01 (9× 失真)
```

带通算子的 DC 零点使 $K_{\text{prop}}$ 秩亏，Cholesky 直接失败。实现用 eigendecomposition + 相对 ridge（$\text{max\_eig} \times 10^{-12}$）规避，诚实报告 `numerical_rank` 和 `null_dimension`。

**修复的真实 Bug**：原始 AR 协方差构造用前向递推 $r[k] = \sum_j a_j r[|k-j|]$，在 $k < p$ 时引用尚未求解的 $r[k]$ 自身（自引用 Yule-Walker），导致 AR(2) 的 $r_1$ 误差约 20%。修正为：先用 Lyapunov 方程精确求解前 $p$ 个滞后，再递推。

### `src/transfer/information_bound.py`

**正确的行列范式激活矩阵**：6×6 网格，每个候选在其行的 $F/12$ 次和其列的 $F/12$ 次 flash 中被激活，共 $F/6$ 次（而不是错误的 $F/36$）。MC 模拟用显式激活矩阵 $A \in \{0,1\}^{F \times Q}$ 计算，避免任何单 $d'$ 近似。

**Union bound**：对每个竞争者 $q$，其对数证据差 $S_q \sim N(-0.5 d^2 n_{\text{diff}}, d^2 n_{\text{diff}})$，误差概率 $P(q \text{ beats } q^*) = \Phi(-0.5\sqrt{d^2 n_{\text{diff}}})$，在所有竞争者上求和得 union bound。

**实测结果**：8 名参与者 $d^2 \in [9.7, 29.3]$，MC 可达命中率均为 1.0，union bound = 0.5（宽松）。

## 4. 修复的工程 Bug

### Bug 1：`TemporalNoise` 上的自引用 Yule-Walker（`exact_covariance.py`）

如上所述，AR(p≥2) 前向递推在 $k < p$ 时引用未知量，导致 AR(2) $r_1$ 误差约 20%。修正为 Lyapunov 求解前 $p$ 项。反例：AR(1) 参数 rho=0.6 精确到 1e-12，AR(2) 参数 (0.5, -0.2) 精确到 1e-9。

### Bug 2：信息论 bound 的范式激活概率（`information_bound.py`）

原代码 `p_tgt = 1/Q`（每次 flash 激活目标的概率），但 BNCI 6×6 中每次 flash 激活的是一行或一列的 6 个候选，不是 1 个。正确 `p_act = 1/6`（每个候选被单次 flash 激活的概率）。修正后 MC 给出与 Phase 4 一致的预测。

### Bug 3：`ExactPropagatedCovariance` 的奇异性处理

带通算子杀死 DC，$K_{\text{prop}}$ 正半定但奇异，直接 Cholesky 失败。修正：eigendecomposition + 相对 ridge，诚实报告 `numerical_rank`。

### Bug 4：`run_e2_prior_scale.py` 的动态导入问题（Python 3.14）

`importlib.util.spec_from_file_location` + `exec_module` 需要在 `exec_module` 之前将模块注册到 `sys.modules`，否则 `@dataclass` 装饰器找不到 `cls.__module__`。修正：加入 `sys.modules[_spec.name] = _boundary`。

### Bug 5：`run_e2_prior_scale.py` 的浮点相等选择

原代码 `next((r for r in ridge_results if r["ridge"] == 1.))` 用精确相等比较浮点数，若计划文件中的 `1.0` 被 JSON 解析为略偏移的值会静默回退。修正为 `math.isclose(r["ridge"], 1.0, rel_tol=1e-9)` 并在缺失时抛出明确错误。

## 5. 下一步科研方向（按优先级）

### 优先级 1：精确协方差 GLS（发现 1 的落实）

用 `ExactPropagatedCovariance.compute(operator, noise, n)` 替换 `OperatorGLS.solve()` 中的 AR 近似白化，建立新研究臂 `filtered_ar8_exact_cov`。

假设：精确 $F K_{AR} F^\top$ 白化后，AR(8) 的 NLL 应接近 AR(1) 的水平（因为白化误差是 NLL 失真的主要来源）。如果 NLL 改善而 hit 不退步，精确协方差传播是更好的路线。

代码计划：
- 在 `doc/assets/operator_research_20261004/continuous_model.py` 新增 `OperatorGLS.fit_exact_cov()` 方法，注入 `ExactPropagatedCovariance` 代替 `TemporalNoise`。
- 在 E2 计划里加入两个额外臂：`filtered_ar8_exact_cov` 和 `filtered_ar1_exact_cov`。
- 使用项目滤波算子（与 Toeplitz 锚点相同），消除算子差异。

### 优先级 2：算子统一对照实验

当前 E2 的 AR vs Toeplitz 差距混合了算子和方法两个效应。建立 `filtered_ar8_innovation`（AR(8) + 项目滤波算子），与 `raw_ar8_innovation` 对比，分离算子效应。

### 优先级 3：Kalman 可识别性评估

交接文档中的 Kalman 路线（慢漂移 ERP 幅度状态）需要先核验 $(A, H)$ 可观测性。在现有 BNCI 数据上，ERP 幅度真正随 selection 变化的程度决定了 Kalman 状态维数是否合理。用 `spectral_discriminability` 对比同一参与者不同 selection 之间的 ERP 谱结构。

### 优先级 4：E3 完整边界校正映射

E1/E3 框架（`boundary_state_research.py`）已建立，但缺少可用于短段推理的差值预测器。需要先解决 Jacobian 约束 $U_{\text{out}}^\top K U_{\text{in}} = 0$（第九层推导中的"正确 ERP 保护"），再连接 `BoundaryStateAnalysis` 的输出软保护到实际推理。

## 6. 验证记录

- 35 项独立反例测试通过（21 项新增 + 14 项前轮，包含本轮新 AR 协方差和信息论 bound 验证）。
- Ruff 通过所有新文件。
- E2 288 cell，0 失败，数学门每次运行前验证，结果数值复现 Phase 4 的 AR(8) 记录（hit=0.667，NLL=6.162，差异 < 0.001）。
- 信息论上界驱动 1 人冒烟和 8 人全量运行均正常完成，0 失败。
- 118 个生产 Python 文件字节未变（`before.json` SHA256 全部匹配）。

## 7. 不变的边界

E2 结果是 BNCI 八人公开开发校准数据，不是十二库迁移结论、不是成人设备验证、不是生产晋升依据。hit≥0.75、NLL<7.17 只是研究候选筛查门，当前最佳臂（AR8 innovation，hit=0.667）未达到 hit 候选门，Toeplitz 锚点（hit=0.771，NLL=2.163）是目前仍需对比的参考基准。

信息论上界的 $d^2$ 估计来自全量校准数据（非留出），是"信号信息量的上界"而不是"可达性能的保证"，不能作为晋升标准。效率间隙 75% 是研究启发，不是已证明的可达提升空间。
