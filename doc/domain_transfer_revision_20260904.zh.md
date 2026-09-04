# 域迁移机制批判与第一轮修订（2026-09-04）

_日期：2026-09-04 · 触发：T 指出当前域迁移不及预期，“adapter 与 CAR 都不太好”_
_状态：已实施 reference policy、GALIA 候选缺陷修复、target adapter 剂量与 row/column 候选排序损失（含 plan/runner 消融接口）_

---

## 1. 一句话结论

当前迁移链的两大件都有**可定位的失效原因**，不是模糊的“域迁移很难”：

1. **CAR 被当成唯一的跨域对齐器，但它只是参考归一化，且当前实现无 bypass、无秩门**。
   本地证据显示它在 BI2014a 16 导上相对 native 参考是净损害；2 导 CAR 只剩一个差分自由度。
2. **target adapter 在现行配置下几乎没有被训练**：本地复算 5 epochs × lr 5e-4
   后门控尺度只有 `0.0050`（上界 `0.25`），suffix logits 与 zero-shot 差 `~2e-4`；
   同一模型 50 epochs × lr 1e-3 后尺度到 `0.141`、loss 由 0.702 降至 0.686。
   “adapter 无效”首先是**剂量不足**，其次才是目标函数错位（只优化 trial CE，不优化候选 top-1）。

本文把证据、外部文献、公式边界、反例和已落地的修订写清；下一轮实验仍须按
[`research_program.zh.md`](research_program.zh.md) 预注册。

---

## 2. 当前机制逐项批判

### 2.1 CAR：数学正确，但科研语义被拔高了

现有结论已部分写于
[`car_math_audit_20260904.zh.md`](car_math_audit_20260904.zh.md)，本文只补批判性收口：

| 事实 | 影响 |
|---|---|
| CAR = 每试次减去共同通道瞬时均值，数学上消除加性标量参考偏移 | 只对齐参考项，不对齐增益、噪声协方差、空间采样、潜伏期、范式与被试分布 |
| 修订前 `prepare_multidomain_source.py` 无条件调用 `ensure_common_channel_average_reference` | 相同参考的数据集也被强制 CAR；本地 16 导 CAR 把朴素 AUC 从 0.544 拉到 0.508 |
| `k=2` CAR 输出两行互为相反数，秩=1 | BI→GTN 的 `{Cz,Pz}` 路径被科学阻塞，但旧 builder 仍允许生成 |
| “先选公共通道再 CAR”与“各域全通道 CAR 再取交集”不等价 | 两种口径不可混报 |

**已落地修订**（代码已改，测试已过）：

- `src/data/domain.py` 新增 `n2p3_native_reference_selection/1`：
  `adapt_native_reference_selection` 只选公共通道、**不重参考**，并 fail-closed 要求
  父 provenance 有显式 `source_reference`、且输入不得已经过 domain adapter。
- `experiments/prepare_multidomain_source.py` 新增
  `--reference-mode {common_average,native_preserve}`；默认改为 `native_preserve`
  （CAR 成为显式 opt-in），且 `common_average` 默认 `--min-common-channels 3`，
  从 builder 层堵死 2 导 CAR。
- 测试：`tests/test_domain.py`、`tests/test_prepare_multidomain_source.py`
  覆盖 native 保留、拒绝 CAR 输入、幂等复用、provenance 篡改拒绝、显式 2 导 CAR 拒绝。

**尚未做（需数据与头模型）**：REST / 大 Laplacian。外部证据见 §4。

### 2.2 target adapter：三个缺陷，已复现其一

现行 adapter 是低秩残差 `H_t = Z_inv + rho_t tanh(a_t) A_t(Z_inv)`，`a_t=0`。
它满足零初始化契约，但：

1. **训练剂量被 plan 写死为 5 epochs × lr 5e-4（batch 16），更新量低于噪声底**。
   本地 probe（`tmp/l0_cache_bi2014a_v3_s12.npz`，subject 01，5 calibration selections，
   288 trials，`dev-01` smoke checkpoint）结果：

   | 配置 | target gate | 实际 scale | train loss 末值 |
   |---|---|---|---|
   | 5 epochs, lr 5e-4 | 0.0201 | **0.0050** | 0.7019 |
   | 50 epochs, lr 1e-3 | 0.6361 | **0.1406** | 0.6862 |

   对应 r1/r2 云端现象：adapter 与 zero-shot 的 AUC/hit 几乎逐位相同。
   该现象不能作为“adapter 机制无效”的证据，只能证明该剂量无效。
2. **目标函数错位**：adapter 只用 weighted binary CE；评估却是 12 候选的行/列
   聚合后 top-1。trial CE 不感知“候选排名”和跨 repetition 证据累积，
   这正是 [`architecture_decision_20260902.zh.md`](architecture_decision_20260902.zh.md)
   已经预言的失败模式。row/column 的 `L_candidate` 仍未实现。
3. **fixed-budget 训练无验证、无 refit**：plan 里 `epoch_selection=fixed_budget`
   关闭了时间顺序验证和 full-prefix refit；5 个 calibration selections 的
   时间顺序信息完全浪费。

**已实施（2026-09-04，第二轮）**：

- 新模块 `src/transfer/candidate_set_loss.py`：`row_column_candidate_loss`，
  公式 `L = mean_decisions (L_row + L_col)`，直接复用 `DecisionLevelLoss`。
  覆盖不全的 decision 记入 `skipped_count`，不产生伪零梯度；全部缺失时
  `SubjectAdapter` fail-closed 为 subject-local `NUMERICAL_FAILURE`。
- `SubjectAdapterConfig` 新增 `candidate_loss_weight` / `candidate_loss_temperature`；
  `fit` 接受 `candidate_ids/selection_ids/target_rows/target_columns` 与
  `candidate_loss_num_rows/columns`。每个 epoch 在 trial CE 之后执行一次
  full-calibration 候选排序更新，refit 阶段同样执行；history 记录
  `candidate_losses` 与 `refit_candidate_losses`。
- runner `run_candidate_cross_decision.py` 新增
  `--candidate-loss-weight/--candidate-loss-temperature`，并把候选元数据传入
  `SubjectAdapter.fit`；promotion matrix 的 evaluation arm 新增同名字段，
  DAG 会转发给 runner。**这是消融接口**：`weight=0` 是纯 CE，
  `weight∈[0,1]` 是 CE+候选排序，`temperature` 控制软标签锐度。
- 剂量协议仍是 arm 参数，不硬编码：新 plan 用
  `epoch_selection=target_time_split`、`epochs=30`、`lr=1e-3` 即可；
  旧 5×5e-4 fixed-budget 臂保留作诊断对照。
- 测试：`tests/test_candidate_set_loss.py`（4 个反例）、
  `tests/test_domain_adapters.py`（adapter 候选损失更新）、
  `tests/test_candidate_promotion_matrix.py`（plan 字段解析/转发/zero-shot 拒绝）

### 2.3 Split-BN / source residual / feature alignment

- P0 Split-BN 的软件实现完整，但**从未在真多域 cache 上运行**；不能判它无效。
- 外部同范式证据（AS-MMD，§4.1）显示 Split-BN **单独不够**：必须与
  target-weighted loss、logit-level RBF-MMD 组合才显著优于 pooled。
- P1 drop-to-consensus 的理论修正是对的，但 p 只控制共识路径频率，不控制残差幅度；
  未跑之前仍是合同，不是疗效。
- `conditional_mean_alignment` 只在 source 有标签时可用；它对 batch 内的
  (domain,class) cell 做 pair 均值差，二阶 CORAL 尚未启用。当前多源负迁移证据
  指向“域梯度暴露 + 判别方向冲突”，单一均值对齐预计不足以解决；应加


### 2.4 对同目录 `domain_alignment_replacement` 提案的批判（不照单全收）

[`domain_alignment_replacement_20260904.zh.md`](domain_alignment_replacement_20260904.zh.md)
提出 CAR 退役 → GALIA/AJD → conditional AS-MMD → PDS 的路线。方向有外部先例，但
把 GALIA 写成下一默认实现之前，必须先承认以下未闭合问题：

1. **GALIA 不是 raw `(C,T)` 变换，它的输入是已经存在的 feature vector。**
   本项目的 CNN 输入是 `(N,16,179)`；若 `feature_lift` 输出 `p_c*p_t`
   （例如 `16*179=2864`），AJD 要对每域 2864×2864 covariance 做联合对角化。
   每域通常只有几百到几千 trials，高维 AJD 的估计方差与 solver 收敛都没有
   在本项目数据规模上验证过。现有 doc 的 `kappa`/`eta_AJD` 门是必要的，但不充分。
2. **GALIA 论文的证据来自 xDAWN/LDA 类的经典特征管线**（6 数据库、333 sessions、
   177 subjects，平均 `+2.12±1.88%`）。该增益不能自动换算到本项目的 compact
   CNN + candidate `hit@R`；且 `±1.88` 的相对量级说明并非所有库/被试同向。
3. **AJD 是无监督二阶对齐**：covariance 相同而类别条件方向相反的反例下
   AJD loss 可以为零，任务仍 chance。现有 doc 自己写了这个反例，却仍把 GALIA
   放在 AS-MMD 之前；应反过来：先验证**条件监督对齐**（AS-MMD/class-conditional
   MMD 或 EA + 条件校正），再决定是否值得引入 AJD 的复杂度和 solver 依赖。
4. **更简单、更低风险的二阶候选是 Euclidean Alignment（EA）**：
   每试次 `X_tilde = R^{-1/2}(X-mu)`，`R` 为域内平均协方差。它对 `(C,T)`
   raw tensor 直接可定义、无迭代 solver、有跨被试 EEG 先例；其旋转不唯一性
   和“可能洗掉幅值”的边界也清楚。应把 EA 作为 GALIA 之前的廉价对照，而不是
   跳过高维 AJD。
5. **“native reference + GALIA”并不解决参考不一致**：native 门只能拒绝不同参考，
   GALIA 也不是参考恢复定理。不同参考的数据仍必须先显式统一参考或按参考分层。

结论：现有 GALIA 提案保留为**候选方案记录**，本轮只落地其中无争议的
“native 默认 + CAR opt-in + 秩门”；GALIA/AJD 必须与 EA 和 AS-MMD 成对预注册，
不能未经反例测试替换 builder。

对已落盘的 `src/data/domain_alignment.py` 通读还发现并修复两个实现缺陷：

- `rjd/uwedge` 不接收 `sample_weight`，但旧代码在显式 domain weights 下仍接受
  这两种 solver，导致 weights 被静默忽略。现在 fail-closed：显式 weights 只允许
  `ajd_pham`。
- `fast_alignment(fit_scope="source_weighted")` 的 source-mixture covariance
  未加 `regularization`，秩亏 source 可产生爆炸条件数。现在统一加 epsilon。
- 测试：`tests/test_domain_alignment.py` 新增两个反例（7/7 通过）。

  **source 选择/加权**与**logit MMD** 对照。

---

## 3. 反例与边界（防自我欺骗）

1. **native_preserve 不是“不处理参考”**：它只在多个 source 声明相同物理参考时合法。
   若参考缺失或不同，builder 必须 fail-closed；不能把 native 拼在一起假装对齐。
2. **CAR k≥3 是工程门，不是信息保留定理**：16 导 CAR 在本 cache 上仍被观测为
   损害，因此 CAR arm 必须永远与 native arm 成对跑，不能默认胜出。
3. **adapter 剂量加大可能过拟合**：50 epochs 的 train loss 下降不能解释为
   held-out 提升；必须时间 split + 早停 + refit，主判读只看 later independent decisions。
4. **logit MMD 是分布匹配不是因果对齐**：若目标候选先验不同，无条件 MMD 会
   把 target/non-target 混合比拉向 source；AS-MMD 原文用平衡小批量缓解，本项目
   必须记录每 batch 的类别组成，且只允许校准/元验证选 `lambda`。

---

## 4. 外部文献依据（联网检索 + 本地全文）

### 4.1 最直接先例：Adaptive Split-MMD（P300 跨数据集小样本）

[AS-MMD](https://ar5iv.labs.arxiv.org/html/2510.21969) / [IEEE](https://ieeexplore.ieee.org/document/11356119)
在 5 公共导、target 仅 10 trials/subject 的 P300 跨数据集设置中报告：
target-weighted CE（权重按 `sqrt(N_S/N_T)` 裁剪、warm-up）、Split-BN
（共享 affine、per-domain running stats）、以及 logit-level RBF-MMD
（median bandwidth、无新增参数）三件组合显著优于 target-only 与 pooled。
本项目已实现 Split-BN；缺的正是另外两项。该文与 BI→GTN 或 BI 家族迁移
同任务、同标签空间，外部迁移强度最高。

### 4.2 重参考：CAR 不是默认最优

[Does Re-referencing Matter? (arXiv:2510.10733)](https://arxiv.org/abs/2510.10733)
在 62 导 P300 上系统比较 no re-reference / CAR / small Laplacian / large
Laplacian，报告 large Laplacian > CAR > no re-reference。该结论支持
“reference 是显式实验因素”，但它是 62 导结果，不能直接外推到 8–16 导；
本项目的 16 导本地消融显示 native > CAR，因此必须保留 native arm。
[Dong 2019](https://doi.org/10.3389/fnins.2019.01068)、
[Hu 2018](https://doi.org/10.3389/fnins.2018.00297) 已在
`car_math_audit` 中作为参考敏感性/REST 边界。


### 4.2b GALIA 与 Euclidean Alignment 的候选边界

- [GALIA（Altındış et al., IEEE JBHI 2023）](https://ieeexplore.ieee.org/abstract/document/10197153)：
  6 个 P300 数据库、333 sessions、177 subjects，报告平均 `+2.12±1.88%`。
  它是**已提取 feature vector 的联合 AJD 对齐**，不是 raw `(C,T)` 的直接替代；
  且增益均值和方差同量级，不能作为本项目 promotion 证据。
- [Revisiting Euclidean Alignment for EEG transfer (arXiv:2502.09203)](https://ar5iv.labs.arxiv.org/html/2502.09203v1)：
  复核 EA 的旋转不唯一、与黎曼几何的关系等边界。EA 计算简单、可直接用于
  `(C,T)`，应作为 GALIA 之前的廉价数据级对照；不能把它当成理论最优。
- 因此数据级对齐的推荐顺序是：`native_preserve`（已落地）→ EA 对照 → GALIA
  仅当 EA 不足且 feature-lift 维度与条件数可审计时再上。


### 4.3 少校准 P300 迁移

- [Minimizing subject-dependent calibration for BCI with Riemannian transfer learning](https://ar5iv.labs.arxiv.org/html/2111.12071)：
  Riemannian 均值/质心对齐 + 最小校准，支持把 SPD 几何对齐列为可选臂。
- [xDAWN + Riemannian P300 transfer (2020)](https://jipmerlibrary.ovidds.com:443/discover/result?logSearchID=161752935&pubid=6957-oai%3Adoaj.org%2Farticle%3Ab6a5860300c04a4fbc7f55e14fd12e7f)：
  先 xDAWN 增强 P300，再 SPD affine 变换到 Riemannian mean，使被试间可比。
  项目已有 `src/baselines/riemann.py` 的 xDAWN+RG 基线，可复用做对照。
- [Inter-participant transfer with attention-based domain adversarial training for P300](https://www.sciencedirect.com/science/article/abs/pii/S0893608024005793)：
  注意 DANN 在 P300 上有阳性先例，但需要用本项目 matched fold 复现。
- [Source sample selection in Euclidean space for P300 BCI](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1360709/full)：
  源试次/源被试选择可以缓解多源负迁移，可作为 `source_domain_mass` 之外的
  数据驱动选择对照。

### 4.4 归一化与对齐的边界

- [SPD domain-specific batch normalization (arXiv:2206.01323)](https://arxiv.org/abs/2206.01323)：
  在 SPD 流形上做域特定 BN，是 Split-BN 的几何版本；本项目当前特征为欧氏 CNN
  特征，只列为候选，不替代 P0。
- [Revisiting Batch Normalization for Practical Domain Adaptation](https://ar5iv.labs.arxiv.org/html/1603.04779)：
  BN 统计替换（AdaBN 类）对 DA 的收益高度依赖问题与数据，必须与 buffer-swap
  机制证据一起判读，不能单独声称因果。
- [Cross-Subject Generalization for EEG Decoding survey](https://ar5iv.labs.arxiv.org/html/2604.27033)：
  Euclidean alignment、几何流形对齐、prototype matching 是当前跨被试 EEG 的
  三类主流；与上文共同支持“输入层对齐 + 域归一化 + 决策目标”三层而非单点修补。

---

## 5. 建议的下一轮机制对照（预注册骨架）

在 BI2015b（source，湿 32 导）→ BI2014a（target，干 16 导）或 GTN 被试群域
上，一次只变一个机制轴：

| Arm | 输入/参考 | 源训练 | target 校准 | 说明 |
|---|---|---|---|---|
| B0 | native/同参考或显式 CAR | single-domain target-only 或 LOSO | none | no-transfer floor |
| A1 | native_preserve | pooled source | zero-shot | 复现现负迁移基线 |
| A2 | CAR k≥3 | pooled source | zero-shot | CAR 参考对照 |
| EA | native + per-domain Euclidean alignment | pooled source | zero-shot | 廉价二阶数据对齐对照 |
| GALIA | native + audited feature lift + AJD | pooled source | zero-shot | 仅在 EA 后、维度可审计时 |
| P0 | 同 A1 | Split-BN pooled | predeclared target buffer | P0 首次真域 |
| P1 | 同 A1 | drop-to-consensus | none | P1 首次真域 |
| AS-MMD | 同 A1 | source + calibration joint，target-weighted CE + logit MMD + Split-BN | 已含 | 外部最强同任务先例 |
| Adapter-D | 同 A1 | frozen source | adapter 30–50 epochs, time-split, decision loss | 剂量修正 + 目标函数修正 |
| Riemannian | 同 A1 | xDAWN+RG per domain or tangent | 最小校准 | 强基线对照 |

主判读不变：matched fold/seed、participant-cluster CI、operational `hit@R`
（valid-subject 口径并存），AUC 只作支持。

---

## 6. 本次工作树留痕

- 修改：`src/data/domain.py`（native reference selection + ensure + schema）
- 修改：`experiments/prepare_multidomain_source.py`（reference-mode / min-common-channels）
- 新增：`experiments/prepare_reference_cache.py`（单 cache 派生 native/CAR，本地构建云端只传缓存）
- 修改：`src/data/domain_alignment.py`（fail-closed weights × solver、fast-alignment 正则化）
- 新增：`src/transfer/candidate_set_loss.py`（row/column decision-level surrogate）
- 修改：`src/transfer/subject_adapter.py`（candidate-loss weight/temperature + fit/refit 接入）
- 修改：`experiments/run_candidate_cross_decision.py`、
  `experiments/run_candidate_promotion_matrix.py`（消融参数接口 + DAG 转发）
- 修改：`tests/test_domain.py`、`tests/test_prepare_multidomain_source.py`、
  `tests/test_domain_alignment.py`、`tests/test_candidate_set_loss.py`、
  `tests/test_domain_adapters.py`、`tests/test_candidate_promotion_matrix.py`
- 诊断（已清理临时文件）：5×5e-4 vs 50×1e-3 adapter gate probe，
  subject 01 smoke checkpoint，288 calibration trials
- 隔离版本：性能改动已提交到独立 worktree 分支
  `perf/domain-transfer-20260904`，commit `92bf8141`；
  source-only freeze：`frozen/active_source_92bf814.source-only.tar.gz` +
  `frozen/active_source_92bf814.manifest.json`（sha256
  `f4697fbb6f4b0a05d6076e50354e754bfd4efcb4a90679a7ebb607394f1cf979`）。
  该分支只含性能链改动，不含未完成的 GALIA/MMD/source-sampling 代码。
- 默认行为变化：多源 builder 默认 `native_preserve`；`common_average` 必须显式指定，
  且少于 3 公共导时 fail-closed（2 导路径可显式 `--min-common-channels 2`，科研上仍不建议）
