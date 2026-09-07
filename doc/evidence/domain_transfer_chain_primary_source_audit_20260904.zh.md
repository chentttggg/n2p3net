# 主链公式与数值选择的一手来源审计

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-04  
本记录按当时源码与文献解释。被审计的旧`domain_alignment_replacement`等方案现已
[物理归档](../../frozen/technical_audit_before_20260906.manifest.json)，其中“主链”不再指当前默认路线。
触发：审计主链 `native reference → domain-specific feature lift → GALIA/AJD → conditional AS-MMD → PDS source sampling` 中每个关键公式与数值选择是否有一手来源支持，并与 `src/` 当前实现逐项比对。  
范围：`src/data/domain.py`、`src/data/domain_alignment.py`、`experiments/fit_galia_alignment.py`、`experiments/prepare_domain_features.py`、`src/transfer/alignment.py`、`src/baselines/deep.py`、`src/train/source_sampling.py`、`experiments/prepare_source_selection.py`、`doc/domain_alignment_replacement_20260904.zh.md`，以及 `.venv` 中 `pyriemann 0.12` 的安装源码与 sdist 测试。  
状态：本文只审计公式/数值选择，不修改任何代码，不提交任何内容，不授权性能结论。

---

## 1. 结论摘要

1. **CAR 与 native reference 本身的实现正确。** `src/data/domain.py` 的 CAR（第 79–80 行）是标准 `z = H_k x = x - (1/k)Σ_c x_c`，必然满足零和与 `rank(H_k)=k-1`；native reference（第 180 行）只做通道选择、不改参考。Kim et al. 2023（Frontiers in Signal Processing）确认这种“对已记录通道直接减均值”的平均参考会秩亏 1。当前代码没有丢弃一个通道，因此“保留 `C` 列、秩只有 `C-1`”的事实必须由下游显式承担。
2. **本地的 “GALIA” 不是 Altındış et al. 2023 的 GALIA。** 论文的 GALIA 是：对**每个域**构造 Riemannian tangent-space 特征向量（super-trial 协方差的 `S_ml=log(G_m^{-1/2} C_ml G_m^{-1/2})` 的上三角向量化）→ 类内 bootstrap surrogate features `T_mk` → 每域预白化矩阵 `W_m` → 交叉外积 `R_ijk=W_i^T T_ik T_jk^T W_j` → 每域 `U_m` 循环 AJD，目标为 `Σ_k Σ_{j≠i} ||off(U_i^T R_ijk U_j)||_F^2`，变换为 `z_ml = B_m^T v_ml`，`B_m=W_m U_m`；fast alignment 是式 (10) 的确定性 `U_x` 求解后 `z_xl=B_x^T v_xl`。本地的“共享 `Q` + 每域 `D_d=diag(QC_dQ^T)` + `z=D_d^{-1/2}Q(x-μ_d)`”是通用的 AJD 合同白化变体，`src/data/domain_alignment.py:6-7` 也明确如此描述；`doc/domain_alignment_replacement_20260904.zh.md` 中的 `min||QC_dQ^T-D_d||_F^2`、`||Q||_F^2=p` 不是论文式 (4)–(9)。论文的输入不是原始 EEG epoch，也不是 xDAWN 输出；是 super-trial 协方差的 tangent vector。论文使用 target fast alignment，但形式与本地不同。
3. **`pyriemann.ajd_pham` 的目标不是 Frobenius off-diagonal 最小化。** 安装源码 docstring 写明其优化基于 KL 散度的 log-likelihood 准则（Pham 2000）；`sample_weight` 会被归一化为和 1 且要求严格正；返回 `D = V X V^T`（实数情形），`V` 只有整体 Frobenius 缩放被本地固定到 `sqrt(P)`，行缩放/置换仍任意。本地 `_fit_ajd` 对 TypeError 静默去掉 `sample_weight` 的兜底会把加权拟合退化为等权拟合（已用 monkeypatch 复现）。
4. **RBF-MMD 公式与 Gretton et al. 2012 式 (3) 完全一致，且可负。** Gretton 原文明说 unbiased `MMD_u^2` 可以为负，并给出非负的 biased V-statistic 式 (5)。本地未 clamp/ReLU，`mmd_squared` 诊断可能为负；这不是实现 bug，但“无偏估计直接作训练损失”没有一手来源给出普遍推荐，AS-MMD 原文（式 7）确实直接用完整无偏式，DAN（Long et al. 2015）则用线性时间配对无偏式。median heuristic 与 Gretton 第 8 节的描述一致。
5. **本地 deep 训练路径不是 Chen & Delorme 2025 的 AS-MMD。** `adaptive_target_weight` 公式本身与论文式 (1)–(4) 一致，但**全仓库零调用**；`deep.py:1842-1859` 调的是 `conditional_domain_rbf_mmd_loss`，只在 source-domain 对之间按 `y=0/1` 对 **adapted features** 做条件 RBF-MMD，不是论文的 **logits 上、marginal、source-target** 的 MMD，也没有 target CE 加权和 λ_MMD(e) 路径。论文输入是 logits 且 MMD 为 marginal。
6. **本地 PDS 不是 Flores et al. 2024 的 Dense PDS。** 论文 Algorithm 1 是硬盘半径 `distance ≤ r0 → reject` 的 dart-throwing，Dense PDS 采样的是 class-mingling index `m ~ Cat(π)`；论文没有 `q_i=exp(-||x_i-μ_t||²/(2T²))` 目标相似度，也没有 Gaussian soft-repulsion、`target_temperature` 或 median bandwidth。本地 greedy `log q_i - λ Σ_j K(x_i,x_j) + tiebreak` 是未见于该论文的“PDS 风格”新目标。所有质量下溢为 0 时，`log(max(q,1e-300))=-690.7755` 吞掉 tie-break（ULP≈1e-14 >> 1e-16），选择与 seed 无关、按数组顺序取前 `n_select` 行（已复现）。
7. **“Fast Alignment” 的对角线-only 目标变换没有一手来源。** He & Wu 2020 EA（arXiv 1808.05464 式 10–11）与 Wu 2025 复盘（arXiv 2502.09203 式 1–3）都是 `x̃ = R̄^{-1/2} x`（原式**不减均值**），使均值协方差精确为 `I`；GALIA 论文的 fast alignment 是完整 `B_x=W_x U_x`，不是 `diag(QC_tQ^T)^{-1/2}`。本地 `transform_fast` 只保证 `Q` 坐标下目标协方差**对角**为 1，off-diagonal 残差原样保留（复算残差 Frobenius 1.712810）。
8. **CLI `source_weighted` 快对齐会写假目标行。** `experiments/fit_galia_alignment.py:110-135` 在 `source_weighted` 时用 `np.zeros((2,feature_dim))` 代替真实 `--target-calibration` 数据；输出 `.npz` 为 2 行完全相同的常数行（复现：真实 target 12 行被忽略，输出 2 行相同）。此外该分支在 CLI 显式传 `regularization` 时会加两次正则（第 402 行已加 `self.regularization`，第 412 行又加 `epsilon=args.regularization`）。

---

## 2. 各公式逐项核验表

| # | 本地公式 / 代码位置 | 一手来源公式 | 是否一致 | 备注 |
|---|---|---|---|---|
| 1 | CAR：`selected - selected.mean(axis=1, keepdims=True)`，`src/data/domain.py:79-80` | 标准平均参考 `z=(I_k-11^T/k)x`；`rank=k-1`、`ker=span{1}`；Kim et al. 2023, Front. Signal Process. 3:1064138 说明直接减已记录通道均值产生秩亏 1 | 计算一致 | 代码保留 `k` 列但不恢复被删秩；文档/下游需声明 `k-1` |
| 2 | Native reference：只选择公共通道、不改参考，`src/data/domain.py:180-181,196-204` | 无数学变换（恒等选择）；本仓库契约 `n2p3_native_reference_selection/1` | 一致（契约性） | fail-closed 要求 `source_reference` 非空 |
| 3 | 特征 lift：`einsum("pc,nct->npt",L,X)`、可选时基 `tq->npq`、flatten，`src/data/domain_alignment.py:92-153`；默认 identity flatten，`experiments/prepare_domain_features.py:61-68` | 标准 P300 空间 lift：xDAWN `X = D A + N`，`Â=argmin||X-DA||²`，`Û=argmax Tr(U^T Â^T D^T D Â U)/Tr(U^T X^T X U)`，解 `U=R_X^{-1}Ψ_s`（Rivet et al. 2009, 式 1–2, 9–12） | **不等价** | 本仓库只有“显式文件 lift”接口，没有 per-domain xDAWN/CSP 拟合生成器；identity flatten 不是 xDAWN |
| 4 | 文档声称的 GALIA 目标：`min_{Q,D_d} Σ_d ω_d||QC_dQ^T-D_d||_F^2, ||Q||_F^2=p`（`doc/domain_alignment_replacement_20260904.zh.md` “GALIA/AJD 数学合同”） | GALIA 式 (4)：`argmin_{U_1..U_M} Σ_k Σ_{j≠i} ||off(U_i^T R_ijk U_j)||_F^2`，约束 `u_i(p)^T M_i(p)u_i(p)=1`；式 (5)–(9) 为循环子问题（Altındış et al. 2023, p.5） | **不一致** | 论文没有共享 `Q`，没有 `D_d`；`R_ijk` 是每域预白化后的类内交叉外积，不是普通 domain covariance |
| 5 | 本地 `GaliaAdapter.fit` 实际目标：`ajd_pham`（KL/log-likelihood AJD）、或 `rjd`/`uwedge`；协方差 `C_d=(1/n)Σx̃x̃^T+εI`，`src/data/domain_alignment.py:555-590,617-663` | pyriemann 0.12 `geometry/ajd.py:118-147`：`ajd_pham` “optimizing a log-likelihood criterion based on the Kullback-Leibler divergence”（引用 Pham 2000）；`rjd` 正交 JADE；`uwedge` 加权最小二乘 AJD | **与文档式 (4) 不一致** | `ajd_pham` 不是 Frobenius offdiag 目标；`rjd/uwedge` 不吃非均匀权重，本地有 fail-closed 守卫 |
| 6 | 论文 GALIA 变换：本地文档写 `z=D_d^{-1/2}Q(x-μ_d)`；本地实现 `transform` 为 `z=(x-μ_d)Q^T/sqrt(diag(QC_dQ^T))`，`src/data/domain_alignment.py:361-370` | GALIA 式 (3)：`z_ml = B_m^T v_ml`，`B_m=W_m U_m`，每域一个 `B_m`，无共享 `Q`、无 `D_d`、无均值相减（论文 v 已做域内归一化） | **不一致** | 本地是“AJD 合同白化”变体；只有 AJD 完美对角化时近似等价 |
| 7 | 本地 Fast Alignment：`D_fast=diag(QC_tQ^T)`，`z=D_fast^{-1/2}Q(x-μ_t)`；`source_weighted` 用加权源混合均值/协方差，`src/data/domain_alignment.py:372-441` | GALIA 式 (10)：`argmin_{U_x} Σ_k Σ_j ||off(U_x^T R_xjk U_j)||_F^2`，`U_x` 为主广义特征向量，`B_x=W_xU_x`，`z_xl=B_x^T v_xl`（Altındış et al. 2023, p.6） | **不一致** | 论文 fast alignment 会优化 `U_x` 并做完整 `B_x`；本地只做 `Q` 坐标对角归一，**不保证 off-diagonal 对齐** |
| 8 | RBF-MMD：`1/(n(n-1))Σ_{i≠j}k(s_i,s_j)+1/(m(m-1))Σ_{i≠j}k(t_i,t_j)-2/(nm)Σ_{i,j}k(s_i,t_j)`，`src/transfer/alignment.py:85-96` | Gretton et al. 2012, JMLR 13:723–773 式 (3) `MMD_u^2` | **完全一致** | 原文明示 `MMD_u^2` 可负；本地不 clamp |
| 9 | median bandwidth：合并样本两两欧氏距离中位数，detached，`src/transfer/alignment.py:45-52` | Gretton 2012 第 8 节：Gaussian 核 `σ` 取 aggregate sample 内点间距离中位数；并说明这仍是 heuristic | **一致** | 全零距离时本地回退 `1.0`，为本地补充约定 |
| 10 | AS-MMD 权重：`base=clip(√(N_s/N_t),1,6)`，`α=min(1,e/W)`，`w_t=1+α(base-1)`，`src/transfer/alignment.py:214-235` | Chen & Delorme 2025 式 (1)–(4)：`α_e=min(1,e/W)`，`w_T^0=√(N_S/N_T)`，`w̃_T=clip(w_T^0,1,6)`，`w_T=1+α_e(w̃_T-1)`，`w_S=1`；`λ_MMD=α_eλ_0` | 权重公式**一致** | 但函数**零调用**；且本地没有 `λ_MMD=α_eλ_0` 调用路径 |
| 11 | 本地训练期 MMD：`conditional_domain_rbf_mmd_loss` 对 source-domain 对、按 label 0/1、在 adapted features 上做条件 RBF-MMD，`src/transfer/alignment.py:165-211`；`deep.py:1842-1859` 只调它 | AS-MMD 式 (6)–(9)：`k(u,v)=exp(-||u-v||²/(2σ²))`，`σ` median heuristic，**logits** 上 **marginal** MMD，source-target，`L=L_sup+λ_MMD MMD²` | **不一致（不同方法）** | 本地是 source-only 预训练的多源条件特征对齐；没有 paper 的 logit-level marginal target MMD |
| 12 | PDS：`q_i=exp(-||x_i-μ_t||²/(2T²))`，`penalty_i=Σ_{j∈S}exp(-||x_i-x_j||²/(2σ²))`，`score=log q_i-λ penalty+tiebreak`，`src/train/source_sampling.py:307-329` | Flores et al. 2024, arXiv:2412.17833 Algorithm 1：`distance ≤ r0 → reject`；Dense PDS 采样 mingling index `m~Cat(π)` | **论文中不存在此公式** | 本地为“PDS 风格”软排斥新目标；`target_temperature=1.0`、`diversity_weight=1.0` 无论文支持 |
| 13 | 本地 `transform_fast`：`z=diag(QC_tQ^T)^{-1/2}Q(x-μ_t)` | EA：`R̄=(1/N)ΣX_nX_n^T`，`X̃_n=R̄^{-1/2}X_n`，无均值相减（He & Wu 2020, 式 10–11；Wu 2025, arXiv:2502.09203 式 1–3） | **不一致** | 完整 EA 用**全矩阵**逆平方根，保证均值协方差 `=I`；本地只对角归一。对角线-only 形式未见一手来源 |
| 14 | 本地源域 `transform` 的数值效果：近似 `Cov(z_d)≈I` | AJD 白化逻辑成立（AJD 理想解使 `QC_dQ^T` 对角） | 数值一致（合成数据复算通过） | 见第 3 节；这只是通用 AJD 性质，不是 GALIA 论文的合同 |
| 15 | CAR 后 rank：本地不丢弃通道；验证只看通道均值接近 0（`src/data/domain.py:301-360`） | Kim et al. 2023：直接平均参考（method A）秩亏 1；正确做法应显式处理该秩 | **需下游处理** | 本文审计不判定“必须丢通道”，但应写入 fit/predict 契约 |
| 16 | xDAWN 本地状态：`prepare_domain_features.py` 无 xDAWN/CSP 生成；`spatial_kind="identity"` 或 `"explicit_file"` | Rivet et al. 2009 式 (9)/(12)；本地 `src/baselines/riemann.py` 有 pyriemann `Xdawn` 但不在主链 lift 中 | **主链缺口** | 需要显式 xDAWN/CSP 生成器或证明 identity lift 对当前 montage 足够 |

---

## 3. 关键数值复算（`.venv\Scripts\python.exe`，`sys.path.insert(0,'src')`）

以下输出均在 2026-09-04 于本仓库环境复算；脚本存放于临时目录，未写入仓库。

### 3.1 `rbf_mmd_loss` 与 brute-force unbiased MMD

- 数据：`n=7, m=5, d=4`，`torch.manual_seed(0)`，bandwidth 由 `_median_bandwidth` 得到。
- `bandwidth = 2.8060211838667626`
- `rbf_mmd_loss = 0.03175771236419678`
- brute-force `= 0.03175773429561679`
- `max_abs_diff = 2.1931420013387992e-08`（仅浮点求和顺序差异）→ **公式逐项一致**。
- 负值实例：`s=[[0],[2]]`，`t=[[1],[1]]`，`bandwidth=1.0` → `MMD_u^2 = -0.07772600650787354`；与 Gretton “`MMD_u^2` may be negative” 一致。

### 3.2 `GaliaState.transform` 白化源域 / `transform_fast` 仅对角归一

合成数据：`p=6`，共享真 `Q_true` 与对角 `D_d`，每域 2000 样本；`GaliaAdapter(method="ajd_pham", ε=1e-6)` 拟合。

| 指标（样本协方差，Frobenius） | 数值 |
|---|---|
| source d0 变换后 off-diagonal Fro | `0.000001` |
| source d0 变换后 `diag-1` Fro | `0.000002` |
| source d1 变换后 off-diagonal Fro | `0.000001` |
| source d1 变换后 `diag-1` Fro | `0.000002` |
| `state.ajd_residual` | `1.126152146116713e-21` |
| `||joint_diagonalizer||_F` | `2.4494897427831783 = sqrt(6)` |
| target 原始 off-diagonal Fro | `5.940245` |
| target `transform_fast` 后 off-diagonal Fro | **`1.712810`** |
| target `transform_fast` 后 `diag-1` Fro | `0.000002` |

结论：源域（参与 AJD 拟合）近似全白化；新目标域 fast 变换后协方差对角为 1，但 off-diagonal 残差仍然很大（复算矩阵中可见 `0.7013`、`-0.7291` 等元素），说明该变换**不保证 off-diagonal 对齐**。

### 3.3 `select_pds_rows` 全零质量下溢

`source = [[1000],[2000],[1500]]`，`target=[[0],[0],[0]]`，`diversity_weight=0`，`target_temperature=1`，`n_select=2`：

- 距离平方：`[1000000.0, 4000000.0, 2250000.0]`
- `quality = [0.0, 0.0, 0.0]`（`exp(-5e5)` 下溢为 0）
- seed `0, 1, 2, 42, 99` 全部输出 `selected=[0, 1]`。
- 原因：`log(max(q,1e-300)) = -690.7755279`，浮点 ULP≈1e-14；tie-break 在 `[0, 2.22e-16)` 内被加法吞掉，`argmax` 恒取第一个最大值。即：质量全下溢时 seed 无效、按数组顺序选择。

### 3.4 `fit_galia_alignment.py --fast-fit-scope source_weighted` 写假目标行

复现命令：

```text
experiments/fit_galia_alignment.py --source d1=s1.npz --source d2=s2.npz --output state.json \
  --target-calibration target.npz --fast-fit-scope source_weighted --fast-output fast_out.npz
```

- 真实 `target.npz`：shape `(12, 5)`，第一行 `[5.1753348434650706, 5.392094786665573, 4.622925549598904, 6.029179232697849, 5.210394938829637]`。
- `fast_out.npz`：shape `(2, 5)`，`domain_ids=['target','target']`；两行完全相同 `[-0.8168509602546692, -0.5669354200363159, -0.496511846780777, -0.3112829327583313, -0.2983919084072113]`；`row0 == row1 = True`，唯一行数 `1`。
- 原因：`experiments/fit_galia_alignment.py:112-113` 在 `source_weighted` 分支把 target 替换为 `np.zeros((2,feature_dim))`；真实 target 12 行完全没有进入输出。

### 3.5 `pyriemann.ajd_pham` 的权重语义与本地 TypeError 兜底

- 安装源码 `geometry/_check.py:81-129`：`sample_weight=None` → 等权；否则检查形状、`check_positivity=True` 拒绝 `<=0`，最后 `weights = weights / sum(weights)`（和为 1）。
- 源码 `geometry/ajd.py:118-240`：`ajd_pham` 在每个 pair 的更新中使用 `normalized_weight` 求和（第 199–203 行）；`D=V@X@V^T`（实数）。
- sdist 测试 `tests/test_geometry_ajd.py`：`None == uniform`；零权重必须报错；`1e-12` 权重近似等价于去掉该矩阵。
- 三矩阵 SPD 复算：uniform 与 `[0.8,0.1,0.1]` 的 `V` 不相等；`[0.8,0.1,0.1]` 与 `[0.1,0.8,0.1]` 最大差 `1.0884132136300948` → 权重确实改变 fit。
- TypeError 兜底复算（monkeypatch 让旧版 `ajd_pham` 遇到 `sample_weight` 抛 TypeError）：

```text
calls = [
  {'eps': 1e-06, 'n_iter_max': 100, 'sample_weight': array([0.25, 0.75])},
  {'eps': 1e-06, 'n_iter_max': 100}
]
weights_silently_dropped = True
```

即 `src/data/domain_alignment.py:642-646` 会把加权 fit 静默变成等权 fit。当前 pyriemann 0.12 的 `ajd_pham` 支持 `sample_weight`，所以正常路径不触发；这是 API 变更时的静默降级风险。

### 3.6 AJD 尺度模糊与本地归一化

- `ajd_pham` 返回 raw `V` 的 `||V||_F=2.5002324457819998`；本地 `B=V*sqrt(4)/||V||_F` 后 `||B||_F=2.0=sqrt(4)`，满足本地文档的 `||Q||_F^2=p`。
- 本地 `transformed = B C_d B^T`（`src/data/domain_alignment.py:657-660`）与解析式缩放后重算一致（`allclose=True`）。
- 对 `B` 行做正对角缩放 `S=diag(2,0.5,3,1.5)` 得到 `B'=SB`，`B' C_d B'^T` 仍保持对角（近似 AJD 意义）；用本地 `transform` 形式计算的白化向量在 `B` 与 `B'` 下完全相同。因此**最终变换对行缩放不变，但存储的 `joint_diagonalizer`/`diagonal_scales` 本身并不唯一**，只固定了整体 Frobenius 尺度。

---

## 4. 与实现的偏差清单

### 4.1 严重（名称/契约与一手来源不符）

- **D1**：`GaliaAdapter` 自称 GALIA，但未实现论文式 (3)–(10) 的 `W_m`、`T_mk`、`R_ijk`、`U_m` 结构；实现的是“共享 `Q` + 对角 `D_d` 的 AJD 合同白化”。若对外命名/账本继续叫 `GALIA`，应改为 `n2p3_ajd_congruence_whitening` 之类，或另行实现论文算法。
- **D2**：本地文档把 GALIA 目标写成 `min||QC_dQ^T-D_d||_F^2, ||Q||_F^2=p`，但论文目标是式 (4)（跨域交叉外积的 off-diagonal 平方），且 `ajd_pham` 也不是这个 Frobenius 目标。文档公式、代码算法、引用论文三者互不一致。
- **D3**：`prepare_domain_features.py` 默认 identity flatten 被称为 “domain-specific lift” 是不成立的；identity 只是公共 lift。主链没有 per-domain xDAWN/CSP 生成器；GALIA 论文的输入特征（super-trial tangent vector）也没有实现。
- **D4**：`deep.py` 的 “conditional AS-MMD” 不是 AS-MMD：没有 target stream、没有 logit-level marginal MMD、没有 `w_t(e)` CE 权重、没有 `λ_MMD(e)`。`adaptive_target_weight` 定义但零调用。
- **D5**：PDS 采样目标不在 Flores et al. 2024 论文中；论文是硬盘 dart-throwing + mingling index。当前实现应标注为 “PDS-inspired soft repulsion”，不能引用该论文作为公式依据。

### 4.2 数值缺陷/风险

- **N1**：`transform_fast` 只归一 `Q` 坐标的对角线，不保证新域 off-diagonal 对齐；复算残差 1.712810。文档把该变换放在 “Fast Alignment” 名称下会误导为论文式 (10) 的 `U_x` 解。
- **N2**：`fit_galia_alignment.py --fast-fit-scope source_weighted` 用 2 行零矩阵伪造 target 并输出重复常数行，同时忽略真实 `--target-calibration` 文件。任何下游把该 `.npz` 当 target 特征使用都会得到假数据。
- **N3**：同一路径显式传 `regularization` 时，`source_weighted` 协方差加两次 `εI`（`src/data/domain_alignment.py:402` 与 `:412`）。
- **N4**：`_fit_ajd` 的 `except TypeError: kwargs.pop("sample_weight")` 会把加权 fit 静默降级为等权 fit。应改为 fail-closed（只对已知不支持签名的版本显式处理，或重新抛出）。
- **N5**：`select_pds_rows` 用 `log(max(q,1e-300))` 防下溢，但当所有 `q=0` 时 `-690.7755279` 吞掉 `[0,ε)` 的 tie-break；选择退化为数组顺序且 seed 无效。建议对 `q` 做 log-sum-exp 型稳定化，或显式检测全零质量并按策略失败/均匀随机。
- **N6**：RBF-MMD 训练损失可负且未 clamp。Gretton 原文允许负值，但作为“距离”训练项，应在 ledger 中记录符号、或提供 biased/V-statistic 消融；`mmd_squared` 诊断名本身暗示非负，会误导读取者。
- **N7**：`ajd_residual` 只对 `B C_d B^T` 重算后的 off-diagonal 比例，属于实现自诊断，不是论文式 (4) 的成本；不能拿它声明 “GALIA 收敛”。

### 4.3 一致且可保留

- CAR 与 native reference 实现（`domain.py:79-80,180-181`）与标准公式/契约一致。
- `rbf_mmd_loss` 与 Gretton 式 (3) 一致；median heuristic 与 Gretton 第 8 节一致。
- `adaptive_target_weight` 的纯函数公式与 AS-MMD 式 (1)–(4) 一致（但未被调用）。
- `ajd_pham` 的归一化 `B*=sqrt(P)/||B||_F` 与 `B C B^T` 重算实现正确；最终白化变换对行缩放不变。
- 源域（参与 AJD 拟合）的 `transform` 在合成数据上近似全白化，符合通用 AJD 性质。

---

## 5. 参考文献 / 来源清单

### 5.1 论文（已获取并阅读全文 PDF）

1. Altındış F, Banerjee A, Phlypo R, Yılmaz B, Congedo M. *Transfer Learning for P300 Brain-Computer Interfaces by Joint Alignment of Feature Vectors.* IEEE Journal of Biomedical and Health Informatics, 27(10):4696–4706, 2023.  
   DOI: https://doi.org/10.1109/JBHI.2023.3299837  
   开放获取全文（HAL）：https://hal.science/hal-04255249/file/Altindis%20et%20al.pdf  
   已核公式：式 (1)–(10)、Algorithm 1、A=25/P=16、六库 333 sessions/177 subjects、平均提升 2.12±1.88%。
2. Gretton A, Borgwardt KM, Rasch MJ, Schölkopf B, Smola A. *A Kernel Two-Sample Test.* Journal of Machine Learning Research, 13:723–773, 2012.  
   URL: https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf  
   已核公式：unbiased 式 (3)、biased V-statistic 式 (5)、`MMD_u^2` 可为负、median heuristic（第 8 节）。
3. Long M, Cao Y, Wang J, Jordan MI. *Learning Transferable Features with Deep Adaptation Networks.* ICML 2015, PMLR 37:97–105.  
   URL: https://proceedings.mlr.press/v37/long15.pdf  
   已核：DAN 采用配对四元组的 linear-time unbiased MK-MMD，与完整 U-statistic 不同。
4. Chen W, Delorme A. *Adaptive Split-MMD Training for Small-Sample Cross-Dataset P300 EEG Classification.* arXiv:2510.21969；IEEE BIBM 2025.  
   arXiv PDF: https://arxiv.org/pdf/2510.21969  
   DOI: https://doi.org/10.1109/BIBM66473.2025.11356119  
   已核公式：式 (1)–(9)、Algorithm 1、Table I（`W=40, λ0=0.4, batch 32, 5×5 CV, seeds {42,123,456,789,321}`）；MMD 在 logits 上、marginal。
5. Flores C, Contreras M, Macedo I, Andreu-Perez J. *Transfer Learning with Active Sampling for Rapid Training and Calibration in BCI-P300 Across Health States and Multi-centre Data.* arXiv:2412.17833 (eess.SP), 2024.  
   URL: https://arxiv.org/pdf/2412.17833  
   已核：Section III-B（PDS 定义）、Section IV-C 与 Algorithm 1（`k, r0, M, π0`；`distance≤r0 → reject`；Dense PDS 采样 mingling index）。
6. He H, Wu D. *Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach.* IEEE Transactions on Biomedical Engineering, 67(2):399–410, 2020.  
   DOI: https://doi.org/10.1109/TBME.2019.2913914  
   arXiv PDF: https://arxiv.org/pdf/1808.05464  
   已核公式：式 (10) `R̄=(1/n)Σ X_iX_i^T`，式 (11) `X̃_i=R̄^{-1/2}X_i`，式 (12) 均值协方差 `=I`；原文无均值相减。
7. Wu D. *Revisiting Euclidean Alignment for Transfer Learning in EEG-Based Brain-Computer Interfaces.* arXiv:2502.09203 (v2), 2025.  
   URL: https://arxiv.org/pdf/2502.09203  
   已核公式：式 (1)–(3)，与 He & Wu 2020 一致（无均值相减）。
8. Kim H, Luo J, Chu S, Cannard C, Hoffmann S, Miyakoshi M. *ICA's bug: How ghost ICs emerge from effective rank deficiency caused by EEG electrode interpolation and incorrect re-referencing.* Frontiers in Signal Processing, 3:1064138, 2023.  
   DOI: https://doi.org/10.3389/frsip.2023.1064138  
   PDF: https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1064138/pdf  
   已核：直接平均参考的秩亏 1 结论与 method A/B/C/D 讨论。
9. Rivet B, Souloumiac A, Attina V, Gibert G. *xDAWN Algorithm to Enhance Evoked Potentials: Application to Brain-Computer Interface.* IEEE Transactions on Biomedical Engineering, 56(8):2035–2043, 2009.  
   DOI: https://doi.org/10.1109/TBME.2009.2012869  
   HAL PDF: https://hal.science/hal-00454568/file/2009_IEEE_BME.pdf  
   已核公式：式 (1)–(2) `X=DA+N` 与最小二乘 `Â`，式 (9) SSNR trace quotient，式 (11)–(12) `Q_D^TQ_X=ΦΛΨ^T`、`Û=R_X^{-1}Ψ_s`。
10. Pham D-T. *Joint Approximate Diagonalization of Positive Definite Hermitian Matrices.* SIAM Journal on Matrix Analysis and Applications, 22(4):1136–1152, 2000.  
    DOI: https://doi.org/10.1137/S089547980035689X  
    说明：本审计以 pyriemann 0.12 源码 docstring 对该文献的引用与实现为准，未单独解析其全文。

### 5.2 包源码 / 测试

- pyriemann 0.12（安装于 `.venv`）：  
  - `D:\bci\n2p3-net\.venv\Lib\site-packages\pyriemann\geometry\ajd.py`（`rjd` L23-115；`ajd_pham` L118-240；`uwedge` L243-347；`ajd` L360-415）  
  - `D:\bci\n2p3-net\.venv\Lib\site-packages\pyriemann\geometry\_check.py`（`check_weights` L81-129）  
  - 安装 wheel 未携带 `tests/`；从 PyPI 下载的 sdist `pyriemann-0.12.tar.gz` 中阅读了 `tests/test_geometry_ajd.py`（本审计解压于 `%TEMP%\pyriemann_src\pyriemann-0.12`）。
- 本地仓库文件（行号为 2026-09-04 当前 checkout）：  
  - `D:\bci\n2p3-net\src\data\domain.py`：CAR L79-80；native L180-181；CAR 验证 L301-360。  
  - `D:\bci\n2p3-net\src\data\domain_alignment.py`：模块声明 L1-9；lift L92-153；`GaliaConfig` L160-198；`transform` L361-370；`fast_alignment` L372-430；`transform_fast` L432-441；`GaliaAdapter.fit` L537-615；`_fit_ajd` L617-663。  
  - `D:\bci\n2p3-net\experiments\fit_galia_alignment.py`：source_weighted 假 target L110-135。  
  - `D:\bci\n2p3-net\experiments\prepare_domain_features.py`：默认 identity lift L61-68、record L69-104。  
  - `D:\bci\n2p3-net\src\transfer\alignment.py`：median bandwidth L45-52；`rbf_mmd_loss` L55-102；`conditional_rbf_mmd_loss` L105-162；`conditional_domain_rbf_mmd_loss` L165-211；`adaptive_target_weight` L214-235；条件均值对齐 L238-330。  
  - `D:\bci\n2p3-net\src\baselines\deep.py`：import L88-91；训练步调用 L1842-1859。全仓库搜索 `adaptive_target_weight|conditional_rbf_mmd_loss|rbf_mmd_loss` 结果：仅 `deep.py` 调用 `conditional_domain_rbf_mmd_loss`；`adaptive_target_weight` 与 `conditional_rbf_mmd_loss` 无调用点。  
  - `D:\bci\n2p3-net\src\train\source_sampling.py`：带宽估计 L197-221；软排斥 L224-243；`select_pds_rows` L245-373；质量/score L307-329。  
  - `D:\bci\n2p3-net\experiments\prepare_source_selection.py`：默认参数 L99-102；调用 L151-164。  
  - `D:\bci\n2p3-net\doc\domain_alignment_replacement_20260904.zh.md`：被审计的本地数学声称（GALIA/AJD、Fast Alignment、AS-MMD、PDS 合同）。
  - `D:\bci\n2p3-net\doc\car_math_audit_20260904.zh.md`：CAR 的本地严格推导（`H_k=I-11^T/k`、rank k-1）。

---

## 6. 给后续方案文档的输入（unresolved questions）

1. **GALIA 的命名与契约必须二选一。** 要么把 `src/data/domain_alignment.py` 改为真正的论文 GALIA（`T_mk/W_m/R_ijk/U_m/B_m` 结构、式 4/式 10、A=25/P=16 语义），要么把当前实现改名并重新立账本，禁止继续用 “GALIA” 引用论文性能。
2. **lift 缺口需要明确。** 主链第一步是 “domain-specific feature lift”，但仓库只有 identity/显式文件接口。后续文档应回答：xDAWN/CSP 生成器由哪个模块负责？是否复用 `src/baselines/riemann.py` 的 pyriemann `Xdawn`？生成器的标签使用范围（source train only）如何入 provenance？
3. **Fast Alignment 的形式。** 当前对角线-only 变换无一手来源；要么实现论文式 (10) 的 `U_x`，要么证明/标注该启发式的适用范围并改名为 `diagonal_fast_scaling`。同时需修复 `source_weighted` 假 target 输出与双正则问题。
4. **AS-MMD 的真实 arm 还没落地。** 若计划声称 “conditional AS-MMD”，需要新增 target stream：logit-level marginal MMD（faithful arm）或条件 MMD（项目 arm），target CE 用 `adaptive_target_weight`，`λ_MMD=α_eλ_0`，Split-BN target buffers 三要素成套；当前 deep.py 只有 source-source 条件特征 MMD。
5. **PDS 引用与公式。** 要么把 Flores et al. 2024 的 Algorithm 1（mingling index + 硬半径）作为 faithful arm，要么把当前软排斥采样标注为本地新方法并做 pre-registered 消融。`target_temperature`/`diversity_weight`/bandwidth 默认值 1.0/1.0/median 目前没有论文支持；全零质量时的 seed 失效必须修复。
6. **MMD 训练项的符号与估计器选择。** 是否接受可负的 unbiased `MMD_u^2` 直接进 loss？是否增加 biased V-statistic（Gretton 式 5）或 linear-time unbiased（DAN）作为消融 arm？ledger 的 `mmd_squared` 字段名是否改为 `mmd_u_squared` 并允许负值？
7. **AJD 的规范性与来源。** `ajd_pham` 的 `B` 不唯一（行缩放/置换）；当前只固定整体 Frobenius 尺度。后续文档应规定：是否要求 canonicalization（例如按每个域对角值排序/符号固定），否则 `state_digest` 虽可复现同一 pyriemann 版本，但不同版本或初始化下不可比。
8. **CAR/native 后的秩契约。** 平均参考秩 `C-1` 需要在 `EpochDataset`/fit 契约里显式声明；是否丢弃一个通道、或让下游模型知道零和约束，当前没有统一结论。

---

*本文档为审计留痕，不授权任何性能提升声明；主链任何 promotion 仍需走既有 held-out gate 与 operational `hit@R` 判定。*
