# 域迁移新主线 v2 具体实施方案（2026-09-04）

> 状态：**已批准，正在实施**（4 个决策点已由 T 全部同意）。
> 触发：T 要求对
> `native reference → domain-specific feature lift → GALIA/AJD → conditional AS-MMD → PDS source sampling`
> 进行公式/数值在线核验，自行科研确认，并给出最优解和可执行方案。
> 配套原始来源核验：`domain_transfer_chain_primary_source_audit_20260904.zh.md`（后台研究代理已完成；本文 §2–§3 已合并其结论）。

> 实施进度（2026-09-04 代码工作同步）：
> - ✅ **WP0 完成**：ACW 命名/schema（`n2p3_ajd_congruence_whitening/1`、`n2p3_acw_fit/1`）、solver 版本、`ajd_offdiag_residual`、fast-output 真实 target 行、AJD TypeError fail-closed、mean 模式 `alignment_inactive`、TSD log-quality + 有尺度 tie-break、`row_identity_digest` 绑定。
> - ✅ **WP1 完成**：`src/data/euclidean_alignment.py`（不减均值、`R̄^{-1/2}X`、四类 fit-scope、条件数门禁）+ `experiments/prepare_euclidean_alignment_cache.py`（attested cache 进、shape-preserving EA cache 出）+ 测试。
> - 🟡 **WP2 部分完成**：`src/transfer/asmmd.py`（Eq 1–9：target-weight、warmup 调度、logit-level marginal 无偏 RBF-MMD²、joint loss）；`SubjectAdapter` 已支持冻结 source logits/labels 的 joint adapter 训练；`run_candidate_cross_decision.py` 已加 `--as-mmd-*`/source-logits 并写入 adaptation procedure。**未完成**：live source-stream + SplitBN source/target 双 buffer 的完整 joint trunk 训练，以及 matrix plan 的 `as_mmd` 字段与 result_batch 下传。
> - 🟡 **WP3 部分完成**：TSD schema `n2p3_source_selection/2` + `sampler_kind="target_similarity_diverse"`、legacy schema 只读兼容、source_selection_record 的 row-identity 校验。**未完成**：`dense_pds` 独立采样器与 `--sampler` 开关。
> - 🟡 **WP4 部分完成**：matrix plan 已解析 `source_selection_record` 并写入 checkpoint batch argv；**未完成**：EA/`as_mmd`/`sampler_kind` plan 字段、`prepare_ea`/`prepare_selection` DAG 节点、result_batch 下传 `--source-stream-cache`/`--as-mmd-*`。
> - 🟡 **WP5 进行中**：四份指导文档已对齐至 EA/ACW/AS-MMD/TSD，旧 GALIA/PDS 描述标为历史；全量测试与 ruff 通过；新 source-only freeze `frozen/active_source_352932b.*` 已生成并提交（冻结对提交 `ab25c96`）；云端 dry-run/CUDA smoke 未做（E2 不启动）。

---

## 1. 结论摘要

1. 当前五个环节都有代码零件，但**不是一条可执行、可逐级消融的主线**：
   - GALIA/feature-lift 的输出是 `(N,P)` 特征矩阵，而 N2P3Net 只吃 `(N,C,T)`，没有下游消费端；
   - AS-MMD 只实现了 source-source 的 conditional feature MMD，论文的 target-weighted CE + **logit** MMD + Split-BN 没有接通；
   - PDS 只是手动 CLI，未进入 64-fold 矩阵计划。
2. 原始来源核验发现 4 个“名不副实”（后台代理 `domain_transfer_chain_primary_source_audit_20260904.zh.md` 逐项复核）：
   - **本地的 “GALIA” 不是 Altındış et al. 2023 的 GALIA**。论文是每域 `W_m` 预白化 + super-trial tangent 特征 `T_mk` + 交叉外积 `R_ijk` + 每域 `U_m` 循环 AJD（式 4–10）；本地是“共享 Q + 每域 D_d=diag(QC_dQᵀ)”的 AJD 合同白化。
   - 本地文档把 AJD 目标写成 Frobenius 最小二乘；`pyriemann.ajd_pham` 实际优化 Pham 的 log-likelihood/KL 准则。
   - 本地采样器名为 PDS，但 Flores 2024 的 Dense PDS 是 mingling-index + disk-radius 拒绝采样，与本地 target-prototype + RBF repulsion 贪心不是同一算法。
   - 本地 AS-MMD 与 Chen & Delorme 的 AS-MMD 公式差异：论文在 **logits** 上做 **marginal** RBF-MMD，并加 target-weighted CE；本地在 **features** 上做 source-source **conditional** MMD。
3. 数值/逻辑 bug 共 **5 个确定项** + **4 个 provenance/结构缺口**（详见 §3）；另外 3 项“可保留但必须改合同/命名”（详见 §4）。
4. 经核验后的**最优主线**改为：

\[
\boxed{
\text{reference/montage}
\rightarrow
\text{Euclidean Alignment (EA, default)}
\rightarrow
\text{N2P3Net trunk}
\rightarrow
\text{faithful AS-MMD (target CE + logit MMD + Split-BN)}
\rightarrow
\text{TSD source selection}
}
\]

   真 GALIA/AJD **不退出项目**，降级为：可选 feature-decoder 对照；本地现有 AJD 白化更名 ACW，只作 TSD 坐标/经典解码辅助，不冒充模型输入对齐。
   Dense PDS 保留为“论文忠实复现臂”，另立名称，不与 TSD 混用。

---

## 2. 来源核验（primary sources）

### 2.1 AS-MMD：本地 PDF 逐式核验

来源：
- Chen & Delorme, *Adaptive Split-MMD Training for Small-Sample Cross-Dataset P300 EEG Classification*, arXiv:2510.21969（本地 `Paper/Adaptive_Split_MMD_P300_arXiv_2510.21969.pdf`）。

论文公式：

\[
\alpha_e=\min(1,e/W),\quad
w_{T0}=\sqrt{N_S/N_T},\quad
\tilde w_T=\mathrm{clip}(w_{T0},1,6),\quad
w_T=1+\alpha_e(\tilde w_T-1),\quad w_S=1
\]

\[
L_{\sup}=w_S E_{(x,y)\sim D_S}[\ell(f(x;\theta),y)]
+w_T E_{(x,y)\sim D_T}[\ell(f(x;\theta),y)]
\]

\[
k(u,v)=\exp\left(-\frac{\|u-v\|_2^2}{2\sigma^2}\right),\quad
\widehat{MMD}^2(S,T)=\frac{1}{n(n-1)}\sum_{i\ne i'}k(s_i,s_{i'})
+\frac{1}{m(m-1)}\sum_{j\ne j'}k(t_j,t_{j'})
-\frac{2}{nm}\sum_{i,j}k(s_i,t_j)
\]

\[
L(e)=L_{\sup}+\lambda_{MMD}(e)\widehat{MMD}^2(S,T),\quad
\lambda_{MMD}(e)=\alpha_e\lambda_0.
\]

核验结论：

| 论文成分 | 本地代码 | 结论 |
|---|---|---|
| target-weight 公式 | `transfer/alignment.py: adaptive_target_weight` | 公式一致 ✅，但**无任何调用点** ❌ |
| MMD 无偏公式 | `rbf_mmd_loss` | 逐项一致，brute-force 复算一致 ✅ |
| MMD 输入 | 论文=logits，本地=features | **不一致** ❌ |
| 条件 vs 边际 | 论文=marginal logit MMD，本地=source-source conditional | **不一致** ❌ |
| Split-BN source/target 双 buffer | 本地有 SplitBN target buffer selector | 部分一致 ⚠️（AS-MMD 训练循环未接入） |

### 2.2 PDS：本地 PDF 逐式核验

来源：
- Flores, Contreras, Macedo, Andreu-Perez, *Transfer Learning with Active Sampling ... BCI-P300*, arXiv:2412.17833（本地 `Paper/Active_Sampling_Multicentre_P300_arXiv_2412.17833.pdf`）。

论文的 Dense PDS（Algorithm 1）是：

1. 指定采样数 `k`、拒绝半径 `r0`、mingling index `m`、类别混合分布 `π`；
2. 每次 `m ~ Cat(π)`，随机抽一个该 mingling 的点；
3. 若新点与已选点的欧氏距离 `≤ r0` 则拒绝，否则加入；
4. 输出新数据集 `B`。

本地 `src/train/source_sampling.py` 实际是：

\[
q_i=\exp\left(-\frac{\|x_i-\mu_{target}\|^2}{2T^2}\right),\quad
score_i=\log q_i-\lambda\sum_{j\in selected}\exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma^2}\right)
\]

结论：**本地算法不是论文 Dense PDS**。它是 target-prototype 相似度 + RBF repulsion 的贪心选样，应命名为 **TSD（Target-Similarity Diverse sampling）**。论文 PDS 可作为独立的“忠实复现/对照”臂，不应共用 schema 名和宣称。

### 2.3 AJD：pyriemann 0.12 源码核验

来源：本地安装 `pyriemann==0.12`，`pyriemann/geometry/ajd.py`。

- `ajd_pham`：直接实现 Pham 算法，优化 **log-likelihood/KL 准则**；`sample_weight` 会被 `check_weights` 归一化并要求严格正。
- 本地文档 `domain_alignment_replacement_20260904.zh.md` 写的目标是
  `min Σ_d ω_d || Q C_d Qᵀ − D_d ||_F²`。**pyriemann 的 ajd_pham 不优化这个 Frobenius 目标**；rjd 是正交 JADE，uwedge 是 U-WEDGE 准则。
- 本地 `_fit_ajd` 在 `TypeError` 时静默去掉 `sample_weight` 重试：这是 fail-open，必须改为 raise。
- `GaliaState` 只记录 `method`，不记录 solver 版本；应增加 `solver_name+solver_version` 或执行环境 digest。

### 2.4 GALIA：论文算法与本地实现逐项比对

来源（已由后台代理阅读全文并核对）：
- Altındış, Yilmaz, Congedo et al., *Transfer Learning for P300 Brain-Computer Interfaces by Joint Alignment of Feature Vectors*, IEEE JBHI 2023, DOI `10.1109/JBHI.2023.3299837`；HAL `hal-04255249`。

论文 GALIA（式 3–10）的结构是：
1. 输入不是 raw `(C,T)`，也不是 xDAWN 输出，而是每域 **super-trial 协方差的 tangent 向量**：`S_ml = log(G_m^{-1/2} C_ml G_m^{-1/2})` 的上三角向量化；
2. 每域先做**预白化矩阵** `W_m`，再构造类内 bootstrap surrogate `T_mk`；
3. 交叉外积 `R_ijk = W_i^T T_ik T_jk^T W_j`；
4. 对每域 `U_m` 做循环 AJD，目标为
   `argmin_{U_1..U_M} Σ_k Σ_{j≠i} ||off(U_i^T R_ijk U_j)||_F^2`；
5. 变换 `z_ml = B_m^T v_ml`，`B_m = W_m U_m`（每域一个 `B_m`，无共享 `Q`、无 `D_d`）；
6. target fast alignment 是式 (10) 的 `U_x` 优化，`B_x = W_x U_x`，不是对角线归一。

本地实现：
`C_d = (1/n)X̃ᵀX̃ + εI`，共享 `Q`，`z = diag(QC_dQᵀ)^{-1/2} Q (x-μ_d)`。

结论：**本地是“AJD 合同白化”，不是论文 GALIA**。论文式 (4)–(10) 与本地文档/代码三者互不一致。后续必须二选一：
- A) 把本地实现改名为 `ajd_congruence_whitening`（ACW），不再用 `GALIA` 命名/账本；
- B) 另起一个真正的 GALIA 实现（tangent-space 特征 + `W_m/T_mk/R_ijk/U_m`），作为独立 feature-decoder 路线。
本文推荐 A（主线用 EA），B 只作可选对照，不阻塞主线。

论文外部证据仍保留：6 个 P300 库、333 sessions、177 subjects，非临床平均 +2.12±1.88%，均值与方差同量级，不能作为 promotion 证据。

### 2.5 EA / average reference

来源：
- He & Wu 2020, *Transfer Learning for BCIs: A Euclidean Space Data Alignment Approach*, IEEE TBME, DOI `10.1109/TBME.2019.2913914`（式 10–11）；
- Wu 2025, *Revisiting Euclidean Alignment...*, arXiv:2502.09203（式 1–3）；
- MOABB `EuclideanAlignment` 官方实现。

EA 的标准形态：`R̄ = (1/N)Σ_n X_n X_nᵀ`，逐 trial `X̃_n = R̄^{-1/2} X_n`；原式**不减均值**，输出保持 `(N,C,T)`。因此 EA 可直接喂 N2P3Net，工程上优于 `(N,P)` 的 GALIA 输出。与本地 `transform_fast` 的差异：EA 用**全矩阵逆平方根**并保证均值协方差= I，本地只做 `diag(QC_tQᵀ)^{-1/2}`，不保证 off-diagonal 对齐（复算残差 Fro=1.712810）。
CAR 数学 `x_c − mean_c(x)` 与标准平均参考一致，rank 降 1 是预期性质，不是实现缺陷；是否保留 `C` 列但由下游声明 `C−1` 秩，必须写进 fit/predict 契约。

---

## 3. 本地实现审计：确定的 bug 与缺口

### 3.1 确定 bug（先修，不涉及科学路线）

| # | 位置 | bug | 复现 |
|---|---|---|---|
| B1 | `fit_galia_alignment.py:105-135` | `--fast-fit-scope source_weighted` 且给 `--fast-output` 时，忽略 `--target-calibration` 内容，用 zeros 拟合并输出 dummy；10 行 target 输入输出仅 1 个唯一行 | ✅ 已复现（后台代理同复现：12 行 target → 2 行相同） |
| B2 | `domain_alignment.py:640-646` | `ajd_pham` 传 `sample_weight` 抛 TypeError 时静默去权重重试，加权拟合被降级 | ✅ monkeypatch 复现 |
| B3 | `baselines/deep.py:1853-1867` + `transfer/alignment.py:238-331` | mean 模式未传 `min_cell_count`；mean diagnostics 无 `alignment_inactive`，`fail_closed` 对 mean 模式永不触发 | ✅ 已复现 |
| B4 | `source_sampling.py:307-312` | `exp(-d²/(2T²))` 下溢后 `log(1e-300)` 全同，tie-break `~2e-16` 打不破 tie，远样本按行序选择，seed 失效 | ✅ 已复现（seeds 0/1/2/42/99 同输出） |
| B5 | `domain_alignment.py:402,410-412` | `source_weighted` 分支协方差已加 `self.regularization`；CLI 显式传 `--regularization` 时又加一次 `εI` | 代码路径 + 后台代理确认 |

### 3.2 provenance / 结构缺口

| # | 位置 | 问题 | 处置 |
|---|---|---|---|
| G1 | `run_pretrain_supervised.py:678-688` | 只验证 `source_cache_sha256` 和索引范围；不验证 PDS feature 的 `source_digest`/行序。若 feature 由不同 lift 或不同行序构造，索引会作用到错误 epoch | record 增加 row-identity digest 并在 runner 校验 |
| G2 | matrix plan | plan 无 `source_selection_record`/`acw_state`/`feature_lift` 字段，TSD 不能作为计划内消融臂 | 见 §5.4 |
| G3 | 本地 ACW/原 “GALIA” 链 | `aligned_features` npz 无任何训练/评估消费端 | ACW 改辅助坐标；EA 承接数据级对齐 |
| G4 | AS-MMD | `adaptive_target_weight` 无调用；无 source+target joint training | 见 §5.2 |
| G5 | `GaliaState` | 不记录 solver 版本；`lift_ranks` 只记录不参与秩约束 | 增加版本；rank-limited 域须显式投影或 fail-closed |
| G6 | RBF-MMD 训练损失 | 无偏 `MMD_u²` 可负（复现 −0.0777）；本地直接作 loss 且诊断名 `mmd_squared` 暗示非负 | 不 clamp（保持 Gretton 定义），ledger 记录符号/负计数；提供 biased V-statistic 消融臂 |
| G7 | `ajd_residual` | 它是 `B C_d Bᵀ` 的实现自诊断，不是 GALIA 论文式 (4) 成本；不能据此声明 “GALIA 收敛” | 改名 `ajd_offdiag_residual`，只在 ACW 合同中使用 |

### 3.3 已通过、不误报的数学点

- `rbf_mmd_loss` 与 Gretton 式 (3) 及 brute-force 一致（代理复算 max diff 2.19e-08）；
- 本地 AJD 合同白化对**参与拟合的源域**近似全白化（offdiag Fro≈1e-6，diag−1 Fro≈2e-6）——这是通用 AJD 性质，不是 GALIA 论文合同；
- `ajd_pham` 归一化 `B *= sqrt(P)/||B||_F` 与 `B C Bᵀ` 重算正确；最终白化变换对行缩放不变；
- CAR 与 native reference provenance 校验严格；
- TSD 增量 repulsion 与完整 kernel 贪心等价（已有测试）。

---

## 4. 最优解及其理由

### 4.1 为什么数据级默认从 GALIA 换成 EA

1. EA 保持 `(N,C,T)` 形状，**N2P3Net 可以直接消费**；论文 EA 是 `R̄^{-1/2} x`（不减均值），有 He & Wu 2020、Wu 2025、MOABB 三方一致。
2. EA 无训练参数、计算成本低，便于 64-fold 消融；本地 `(N,P)` ACW 特征矩阵没有现有模型消费端。
3. 后台代理已确认本地 “GALIA” 不是论文算法；在论文全文未实现前，把它当默认会引入 solver 目标错配和 dead-end。把本地实现更名 `AJD congruence whitening (ACW)`，论文 GALIA 留作独立 feature-decoder 对照。
4. 决策规则：`EA = 默认数据对齐`；`ACW = TSD 坐标 + 经典特征解码对照`；`真 GALIA = 可选、不阻塞主线`；若未来要 N2P3Net 吃对齐特征，必须先把输出重新组织成 `(N,C,T)` 或换 decoder，不能静默伪装。

### 4.2 为什么深度适配默认用 faithful AS-MMD，而不是本地 conditional feature MMD

1. AS-MMD 是同任务（P300 跨数据集小样本）最直接的外部证据，公式已逐项核验。
2. 论文输入是 **logits**，目标函数含 target-weighted CE；本地 feature MMD 缺少监督目标、可能在 cell 不足时整 epoch 不生效。
3. 条件 feature MMD 保留为安全对照（防 marginal 类别塌缩），但不得继续叫 AS-MMD。

### 4.3 为什么 source selection 用 TSD 而非 Dense PDS

1. 论文 Dense PDS 是数据集内部的 mingling/半径拒绝采样，不是“按 target calibration 选 source rows”；直接套用会改变采样单位。
2. TSD 目标明确（target prototype 相似 + 源内多样性），可审计、可配额，适合本项目 source-row 选择。
3. Dense PDS 作为忠实复现臂单独实现，命名 `dense_pds`，结果只与 `uniform`/`TSD` 对照，不混入主线声明。

---

## 5. 具体实施方案

### WP0：先修 5 个 bug + 命名/合同 + 2 个 provenance 项（0.5–1 天）

文件/测试：

- `src/data/domain_alignment.py`：
  - `_fit_ajd` TypeError fallback 改为：非均匀 `weights` 时 `raise RuntimeError`，不静默降级；
  - 修复 `source_weighted` 显式 `regularization` 时 `εI` 加两次；
  - 将本地实现/state 命名从 `GALIA` 改为 `AJD congruence whitening (ACW)`：schema `n2p3_ajd_congruence_whitening/1`（旧 `n2p3_galia_alignment/1` 只读拒绝或显式迁移）；
  - `ajd_residual` 改名 `ajd_offdiag_residual`，并增加 `solver_name` + `solver_version`（`pyriemann.__version__`）。
- `experiments/fit_galia_alignment.py`（更名 `fit_acw_alignment.py` 或保留旧名但输出新 schema）：
  - `source_weighted` 与 `--fast-output` 组合时，必须读取真实 `--target-calibration` 行做 transform（fit 统计仍只用 source mixture）；禁止 dummy zeros；
  - fast record 增加 `target_offdiag_residual = ||offdiag(QC_tQᵀ)||_F / ||QC_tQᵀ||_F`，超过声明阈值即失败。
- `src/transfer/alignment.py` + `src/baselines/deep.py`：
  - `conditional_mean_alignment_loss` 返回 `alignment_inactive`，接受并传递 `min_cell_count`；
  - mean 模式也走 `fail_closed`；
  - RBF-MMD ledger 增加负值计数与符号；不 clamp（保持 Gretton 定义）。
- `src/train/source_sampling.py`：
  - `score_i = -d²_i/(2T²) - λ Σ_j K(x_i,x_j) + tie`，不再 exp 后取 log；
  - tie 改为有尺度随机量（如 `rng.uniform(0, 1e-9)`），并新增“远距离样本下 seed 可区分”的反例测试。
- `experiments/prepare_domain_features.py` / `prepare_source_selection.py` / `run_pretrain_supervised.py`：
  - feature record 增加 `row_identity_digest`（subject_ids + event_id/selection_id 的有序 digest）；
  - selection record 携带它；runner 用同法重算并校验，失配 fail-closed。
- 测试：`tests/test_domain_alignment.py`、`tests/test_domain_adapters.py`、`tests/test_source_sampling.py`、`tests/test_pretrain_supervised.py` 各加反例。

### WP1：EA 数据级对齐（1–2 天）

- 新 `src/data/euclidean_alignment.py`：
  - `EuclideanAlignmentConfig(reference_scope, shrinkage, regularization, max_condition_number, seed)`；
  - `fit_reference(epochs, domain_ids=None, scope="source_or_calibration")`；
  - `transform(epochs, reference)`，输出 `(N,C,T)`，逐 trial `x_i ← R̄^{-1/2} x_i`（**按 He & Wu 2020 / Wu 2025，不减均值**；如要减均值必须作为显式 `center=True` 变体并记录）；
  - provenance record：scope、covariance、condition、shrinkage、buffer digest。
- 新 `experiments/prepare_euclidean_alignment_cache.py`：
  - 输入 attested `EpochDataset`；
  - `--fit-scope {source_rows, target_calibration, per_domain}`；
  - `--reference-mode` 必须与上游 native/CAR cache 一致；
  - 输出新 `EpochDataset` cache（含 domain axis、identity table、lineage）。
- 消融：`none`（原 cache）、`ea_source`、`ea_target_calibration` 三臂；EA 只用声明 scope 的统计，禁止 target test。
- 测试：合成 SPD 矩阵，验证 transform 后 reference covariance≈I、条件数门禁、provenance fail-closed。

### WP2：faithful AS-MMD target-calibration 训练（2–3 天）

- 新 `src/transfer/asmmd.py`：
  - 输入 source logits + labels、target calibration logits + labels；
  - `target_weight(epoch, N_s, N_t, W, min=1, max=6)`（已核验公式）；
  - `mmd_schedule(epoch, W, lambda0)`；
  - `rbf_mmd_loss` 复用 `transfer/alignment.py`，但明确**logits**；
  - Split-BN 双 buffer 切换：source batch 用 source buffer，target batch 用 target buffer，eval 只用 target buffer。
- 扩展 `src/transfer/subject_adapter.py`：
  - `fit(source_X, source_y, target_X, target_y, ...)` 新 joint 模式；
  - `--as-mmd-lambda0`、`--as-mmd-warmup`、`--as-mmd-input {logits}`、`--as-mmd-min/max`；
  - 记录每 epoch target weight、MMD²、负 MMD 计数、bandwidth、buffer digest。
- 扩展 `experiments/run_candidate_cross_decision.py`：
  - 新 `--source-stream-cache`（与 target cache 同合同）、`--as-mmd-*`；
  - zero-shot/无 MMD 对照保持原路径。
- 计划 arm 命名：
  - `as_mmd`：faithful logit marginal MMD + target CE + SplitBN；
  - `cond_mmd_features`：本地安全对照（source-source conditional feature MMD，仍保留但不称 AS-MMD）；
  - `no_mmd`：matched control。
- 测试：公式与手工 PyTorch 实现逐位一致；target weight 曲线单调；SplitBN buffer 不泄漏 target test；负 MMD 批次有 ledger。

### WP3：ACW 辅助 + 真 GALIA 可选 + TSD 正式化（1–2 天）

- 保留本地 AJD 白化（更名 `AjdCongruenceWhitening` / ACW），用途改为：
  1. 为 `prepare_source_selection.py` 提供对齐特征（TSD 坐标）；
  2. 经典 xDAWN/LDA/Ridge decoder 对照（单独 runner，不碰 N2P3Net）。
- 真 GALIA（论文式 3–10：super-trial tangent 特征 + `W_m/T_mk/R_ijk/U_m`）作为**可选 feature-decoder 对照**，不进入主线；只有 T 明确批准后才实现。
- 重命名与 schema：
  - `n2p3_pds_source_selection/1` 升 `n2p3_source_selection/2`，字段 `sampler_kind="target_similarity_diverse"`；
  - 新 `dense_pds` 采样器（mingling index + radius）独立实现，schema 与 TSD 分开。
- `experiments/prepare_source_selection.py` 增加 `--sampler {target_similarity_diverse, dense_pds}`，各自可审计。
- 文档中所有“PDS 选样”改为“TSD 选样（PDS-inspired）”，论文 Dense PDS 列为对照。

### WP4：matrix plan 全链字段与 DAG（1 天）

在 `run_candidate_promotion_matrix.py` 增加 plan 字段：

| 字段 | 作用 |
|---|---|
| `source_selection_record` | 每 source arm 一个共享 TSD record；batch 全部 job 共用 |
| `euclidean_alignment` | `{mode: none|source|target_calibration, cache}`，由 source/target cache 路径体现 |
| `acw_state`（可选） | 仅 provenance/诊断；TSD 坐标已固化在 selection record |
| `as_mmd` | `{lambda0, warmup, min, max, splitbn_target_buffer_source}` |
| `sampler_kind` | `uniform|target_similarity_diverse|dense_pds` |

DAG 任务新增：`prepare_ea`（或要求离线 cache）、`prepare_selection`、`result_batch` 下传 `--source-stream-cache` 与 `--as-mmd-*`。plan digest 自动纳入新字段。

### WP5：文档、冻结、云验证（1 天）

- 更新 `doc/research_program.zh.md` §2.5/§9：主线改为 EA + AS-MMD + TSD，本地 AJD 白化更名 ACW、真 GALIA 降级为可选对照；
- 更新 `doc/domain_alignment_replacement_20260904.zh.md` 与 `domain_transfer_revision_20260904.zh.md` 的结论段；
- 全量 pytest；
- 新 source-only freeze（不可覆盖 `b96a6af`/`b86d558`）；
- 云端先在 BI2014a 64-subject cache 上做 EA/AS-MMD/TSD 的 dry-run + 1-seed CUDA smoke，再做正式 1-seed×64-fold。

---

## 6. 消融矩阵（一次只动一个轴）

基线全部同 seed、同 EPOCH、同 target 缓存；主指标 valid-subject `hit@R`，operational 同报。

| Arm | reference | 数据对齐 | 深度适配 | source selection | 目的 |
|---|---|---|---|---|---|
| B0 | 同参考 | none | zero-shot | uniform | no-transfer floor |
| A1 | native | none | pooled source CE | uniform | 复现负迁移 |
| A2 | CAR | none | 同 A1 | uniform | reference 对照 |
| EA0 | native | EA(source) | 同 A1 | uniform | EA 主效应 |
| EA1 | native | EA(source) | SplitBN target | uniform | EA+BN |
| AS0 | 同 A1 | none | no_mmd | uniform | AS-MMD 的 no-MMD 对照 |
| AS1 | 同 A1 | none | **as_mmd(logit marginal)** | uniform | **首选深度适配** |
| AS2 | 同 A1 | none | cond_mmd_features | uniform | 条件安全对照 |
| S0 | 同 A1 | none | AS1 | uniform | 选样对照 |
| S1 | 同 A1 | none | AS1 | **TSD** | 目标相似选样 |
| S2 | 同 A1 | none | AS1 | dense_pds | 论文忠实对照 |
| G1 | 同 A1 | ACW(aux) | AS1 | TSD(ACW coords) | 本地 AJD 白化辅助价值（不称 GALIA） |

优先 contrasts：
- `EA0 vs A1`、`AS1 vs AS0`、`S1 vs S0`、`A2 vs A1`、`G1 vs S1`。
其余为 exploratory。

---

## 7. 门禁与 provenance（不可省略）

1. EA/ACW/MMD bandwidth/TSD 的任何 target 统计只能来自 `target_calibration` 或 `unlabelled buffer`；target test 进入即 invalid。
2. 每个变换输出必须有：输入 cache sha256、fit scope、行 digest、state/transform digest、条件数、rank、AJD/offdiag residual。
3. TSD record 必须校验 row identity digest；索引失配 fail-closed。
4. AS-MMD ledger 必须记录每个 epoch 的 `w_T(e)`、`λ_MMD(e)`、MMD²、negative count、bandwidth、SplitBN buffer digest。
5. 负 MMD 不 clip（保持论文无偏定义），但计入 ledger 与训练诊断；若需作为 loss 可用，则在计划中显式选择 biased V-statistic，另立 arm。
6. ACW state 必须记录 solver 名称与版本；未来接口变更导致 TypeError 时禁止静默回退。
7. 真 GALIA 未实现前，任何 schema/文档/日志不得再把 ACW 称作 “GALIA”。
8. 每个 arm 的 plan digest 必须包含上述全部字段；重跑用同一 freeze。

---

## 8. 已批准决策（2026-09-04）

1. ✅ **EA 默认、ACW 辅助、真 GALIA 不进入主线**（本地现有实现已从 `GALIA` 更名为 ACW）。
2. ✅ **AS-MMD 按论文忠实用 logit marginal MMD**；条件 feature MMD 仅保留为安全对照。
3. ✅ **PDS 命名拆分**：TSD 为主线，Dense PDS 为忠实对照。
4. ✅ **负 MMD 的 biased V-statistic 仅作 exploratory loss 臂**；主线保留无偏式并记账。

---

## 9. 来源清单

- 后台代理一手来源审计：`doc/domain_transfer_chain_primary_source_audit_20260904.zh.md`（本文 §2–§3 已合并其结论）。
- Chen & Delorme, AS-MMD, arXiv:2510.21969（本地 PDF 已逐式核验 Eq.1–9）。
- Flores et al., Active Sampling / Dense PDS, arXiv:2412.17833（本地 PDF 已核验 Algorithm 1）。
- Gretton et al., *A Kernel Two-Sample Test*, JMLR 13:723–773, 2012（MMD 无偏式 3、biased 式 5、median heuristic）。
- Long et al., *Learning Transferable Features with Deep Adaptation Networks*, ICML 2015（线性时间无偏 MMD）。
- Altındış et al., GALIA, IEEE JBHI 2023, DOI `10.1109/JBHI.2023.3299837`；HAL `hal-04255249`（论文式 3–10 已由代理逐项比对）。
- Pham 2000, SIAM J. Matrix Anal. Appl., DOI `10.1137/S089547980035689X`；pyriemann==0.12 `geometry/ajd.py` / `geometry/_check.py` 与 sdist tests。
- He & Wu 2020, EA, IEEE TBME, DOI `10.1109/TBME.2019.2913914`；Wu 2025, arXiv:2502.09203；MOABB `EuclideanAlignment` 实现。
- Rivet et al. 2009, xDAWN, IEEE TBME, DOI `10.1109/TBME.2009.2012869`。
- Kim et al. 2023, average reference rank, Front. Signal Process., DOI `10.3389/frsip.2023.1064138`。
- Kilani et al., source sample selection in Euclidean space, Front. Neurosci. 2024, DOI `10.3389/fnins.2024.1360709`。
- 本地代码/数值复算：`src/*`、`experiments/*`、`tests/*`（文件与行号见 §3）。
