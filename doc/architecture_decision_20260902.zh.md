# N2P3-Net 架构决策：从横向嵌套到纵向分层

_版本：2026-09-03 数学闭合修订 · 决策类型：架构重构 · 状态：框架已在 `ef7a40e` 落地，性能证据未完成_

> 📌 **文档角色：** 本文件保留架构裁决及其理由。新任务、实验门禁和当前结论以
> [`research_program.zh.md`](research_program.zh.md) 为准；P0/P1/P2 的代码链验收见
> [`framework_acceptance_20260903.zh.md`](framework_acceptance_20260903.zh.md)。

---

## 执行摘要

**决策**：放弃横向嵌套网络（K 个 EEGNet 当外层网络神经元），采用纵向分层架构（共享 trunk + 可丢弃域残差 + 可微决策集合网络）。

**三路文献调研核心发现**：
1. 嵌套网络/hypernetwork/FiLM：FiLM 在 EEG 上有实证（Défossez 2024, Brain's Bitter Lesson），hypernetwork 需额外静息态数据（P300 不适用），子网络当神经元无任何领域先例
2. EEG 基础模型端到端 vs 分阶段：SpellerSSL（arXiv:2509.19401, NeurIPS 2025）分阶段范式达 94% 准确率，基础模型在 P300 任务无优势证据且存在 identity trap 和低频偏置
3. 少样本适配机制：（等待第三路结果，但已有 MetaSleepLearner / MUPS-EEG 证据显示 MAML 可行）

**负迁移根因**：
- 不同域 class-conditional ERP direction 不一致，共享滤波器被迫取折中面
- BatchNorm 混合统计失配（st_temporal_norm / st_spatial_norm 用 batch 统计）

**最终方案**：分层而非横向，源域阶段编码器与决策层联合可微，目标适配阶段分开。

---

## 一、问题定位

### 1.1 现状观察

- 当前 BNCI-target 朴素多源 pooled 训练：-21.94pp hit@8 负迁移（相对单域基线）
- 80/20 域加权恢复 +15.14pp，但仍低于 B0
- 输入 mean/std 单独更换仅恢复 +0.07pp，说明公共统计不是主因

### 1.2 用户原始提议

"把 n2p3-net 当隐藏层神经元，做嵌套网络（或双层网络）"

---

## 二、横向嵌套网络的数学检验

### 2.1 三种可能解读

**解读 1：横向线性组合（Σ w_k f_k(X)）**

```
y = w_1 * EEGNet_1(X) + w_2 * EEGNet_2(X) + ... + w_K * EEGNet_K(X)
```

**判定**：❌ 本项目暂不采纳，但原先的数学否证过强。

- 多个固定宽度的非线性子网之和一般具有比单个同宽子网更大的函数类；不同
  子网的 Jacobian、初始化和特征区域不同，
  \(\nabla_{\theta_k}L=w_kJ_{f_k}^{\top}\nabla_yL\) 不能推出函数必然塌缩。
- 该方案仍因参数成本约为 K 倍、推理成本增加、样本不足时估计方差上升、以及
  缺少专家多样性/去相关约束而不进入当前主线。
- 若未来复评，必须预注册 expert diversity（例如特征/梯度去相关）、同预算参数
  对照和独立推理成本；“塌缩”只能作为待测经验结果，不能作为定理。

**解读 2：域绑定专家混合（每个源数据集一个 expert）**

```
y = Σ_d w_d(x) * EEGNet_d(X)    # w_d 是目标被试学习的混合权重
```

**判定**：⚠️ 数学成立但混合空间太窄

- 参数效率极好（每被试 K 个参数）
- 直接命中负迁移机制：取消折中面，让目标被试自由混合
- **致命缺陷**：每个专家只吃单域数据本身弱，K 受限于数据集数量（当前 K≈3），学不到 P300 精髓——精髓在共享部分，这个架构把共享完全丢了

**解读 3：纵向嵌套（内层对每个 epoch 编码，外层集合网络输出冻结 vocabulary 上的 posterior；BrainSync 为 9 选 1）**

```
# 对 decision 内第 i 个 epoch：
e_i = f_θ(x_i)                          # 内层：trial 级编码
s_d = (1/n_d) Σ_{i: c_i=d} e_i         # 外层：candidate 聚合
P = softmax(s)                          # 外层：9 选 1 posterior
```

**判定**：✅ 这才是真正的双层网络

- 与产品任务同构（多试次证据累积做 9 选 1）
- 是当前仓库最大结构性缺口

**梯度验证**（关键）：

设损失 L = -log P_{d*}，则

```
∂L/∂e_i = (P_{c_i} - 1[c_i = d*]) / (τ n_{c_i})
```

- 真候选的 epoch 梯度幅度正比于 (1 - P_{d*})，已经判对就几乎不再推
- 错候选的 epoch 梯度幅度正比于它当前的 posterior，排名第二的混淆候选拿最强负梯度

而 per-trial CE 下每个 non-target epoch 的梯度只取决于它自己的 logit，与"它正在和真答案抢第一"无关。

**这是两个不同的目标函数**，不是同一个东西的两种写法。你 doc 里反复出现的失败模式 trial AUC 升而 candidate hit@R 降，根因就是一直在优化代理量。

---

## 三、文献调研结论汇总

### 3.1 嵌套/Hypernetwork/FiLM 先例（第一路）

**已验证的 EEG 先例**：
- **HyperEEGNet**（Rajpura 2024, ICLR'25 投稿）：用静息态 EEG 生成 EEGNet 权重做运动想象分类。局限：需额外静息态采集、未验证 P300
- **FiLM 被试条件化**（Défossez《Brain's Bitter Lesson》arXiv:2406.04328, 2024）：大规模 MEG/EEG 语音解码，**FiLM 实测最优**，每被试参数增量仅 O(2d)

**无先例**：
- CNP 用于 EEG 分类无直接应用
- 子网络当神经元在任何领域均无成功先例

**判定**：few-shot 数据极端稀缺（每被试数十 decision），hypernetwork 需足够多样性的 meta-training episodes，FiLM 的每被试参数虽少但需稳定 subject embedding。

### 3.2 EEG 基础模型端到端 vs 分阶段（第二路）

**关键发现**：
- LaBraM/EEGPT/BENDR/BIOT 等基础模型**均未报告 P300/ERP 专项性能**
- **负面证据确凿**：
  - Identity Trap（arXiv:2606.06647）：基础模型依赖被试身份捷径
  - 低频偏置（arXiv:2608.01898）：重构预训练过度强调低频、低估高频振荡
  - EEG-FM-Bench：多任务负迁移，与你的 -21.94pp 一致

**成功案例：SpellerSSL**（arXiv:2509.19401, NeurIPS 2025）
- 机制：P300 聚合策略 + 自监督学习 + 1D U-Net，域内/跨域预训练后用轻量 ERP-Head 微调
- 性能：94% 字符识别率（仅 7 次重复）
- **关键**：分阶段范式（预训练 + 冻结主干 + 轻量适配）在 P300 任务有效

**文献外推边界（强制声明）**：以上文献均为跨任务证据——FiLM 来自 MEG/EEG 语音解码、MetaSleepLearner 来自睡眠分期、Reptile 证据主要来自运动想象、SpellerSSL 的 94% 是其自身协议下的结果（不含 unknown-subject、target-switch、operational denominator）。它们支持"候选机制值得实验"，不支持"当前方案已是 P300 few-shot 最优"。任何最优性结论只能由本项目 E2 运行事实产生。

**判定**：分阶段范式，任务导向 SSL（借鉴 SpellerSSL 的 P300 聚合）而非通用重构，冻结编码器 + 轻量分类头微调。

### 3.3 少样本适配机制（第三路，进行中）

已知文献片段：
- **MetaSleepLearner**（arXiv:2004.04157, IEEE JBHI 2021）：MAML 在生理信号少样本适配可行
- **MUPS-EEG**（arXiv:2003.06113, 2020）：结合 MAML 和 meta learner，仅需 10-20 个目标被试样本

等待第三路完整结果补充。

---

## 四、最终推荐架构（分层而非横向嵌套）

### 4.1 五层递进骨架

```
1. N2P3Net core（共享 trunk）
   ↓
2. 训练期：可丢弃的低秩域残差（随机置零 p=0.3）
   ├─ 每个源数据集一个 Δ_d
   └─ trunk 单独工作成为显式训练过的跨域共识模型
   ↓
3. Invariant/private 分解（当前已有）
   ├─ Z_inv：跨域可迁移 P300 evidence
   └─ Z_priv：吸收采集/被试/设备 style
   ↓
4. Target 低秩残差适配（当前已有）
   └─ 目标被试 inner-loop 只更新此层
   ↓
5. **可微决策集合网络（新增关键层）**
   └─ 输入：decision 内每个 epoch 的 evidence
   └─ 输出：9 选 1 posterior
   └─ 损失：L = L_trial + λ·L_decision
```

### 4.2 关键创新点

**1. 可丢弃域残差修正 source-domain adapter bank 的缺陷**

doc 第 8 节反例 8 的理由是训练启用 source adapter 而推理未 route 则结果无效——反例本身对，但证明的是实现有缺陷不是方向错。

**缺陷**：用 trunk + Δ_d 训练后，在未见目标上取 Δ=0，而 trunk 从未在 Δ=0 条件下被训练过，Δ=0 是 off-manifold 操作点。

**修法**：训练时对域残差随机置零（概率 p 丢弃 Δ_d），trunk 单独工作成为显式训练过的合法操作点（跨域共识模型），Δ_d 只承载域偏离量。

**2. 决策层可微化（candidate ranking surrogate）**

当前 candidate_evidence.py 是后处理聚合（mean / tempered_evidence），没有梯度回传到编码器。

新设计：ρ_ω(A_d,r, B_d,r, Q_d,r, N_d,r, coverage_d,r) 成为可学习模块，第一版用线性/温度模型。L_candidate 是 hit@R 的 top-1 ranking surrogate，不是 hit@R 的直接可微形式；晋升只看 operational hit@R。

**数学约束（必须写进契约）**：
- 决策级损失是加项不是替代：`L = L_trial + λ·L_decision`。membership 与目标候选已能确定全部 trial binary labels，因此保留 trial CE 是优化稳定性、局部监督、有限样本正则化和校准需求的预注册设计选择，不是标签数量导致的数学必然；`L_decision` 单独训练仍不作为晋升 arm。
- λ ∈ [0.1, 0.5] 是预注册搜索范围，不是理论推导上界

**3. BatchNorm 混合统计诊断（P0 优先级）**

代码里确认了 st_temporal_norm / st_spatial_norm 用 batch 统计，混合多域时每个域被错误归一化。Split-BN 在跨数据集 P300 上被反复提出正是针对这层。

建议做 Split-BN 实验臂；其对 21.94pp 差距的贡献必须通过成对实验和机制诊断检验，不能预先宣称会恢复其中相当部分。

### 4.3 分阶段训练（折中方案）

**Phase 1：源域多域训练（联合可微）**
- 编码器 + 可丢弃域残差 + 决策层**端到端联合训练**
- 损失：L_trial + λ_set·L_candidate_set + λ_align·L_conditional_align + λ_priv·L_domain_probe + λ_orth·L_orthogonality
- **关键**：决策层必须接进可微链条，优化 decision 级 ranking surrogate 而非仅 trial 代理量（surrogate ≠ hit@R，晋升仍只看 operational hit@R）

**Phase 2：冻结主干训决策层**
- trunk 冻结
- 只训练决策层的 ρ_ω 参数（候选 posterior 网络）

**Phase 3：目标 inner-loop 适配**
- 冻结 trunk + shared invariant map + source classifier
- 只更新 target 低秩残差 + evidence calibration
- 几十个标注 decision 只撑得起这个参数规模

**端到端还是分开的最终答案**：目标适配阶段必须分开（几十个标注 decision 撑不起端到端），源域阶段必须联合（否则一直优化代理量）。

---

## 五、已执行机制与后续待验证项

### P0：BatchNorm 混合统计诊断（框架已实现）
```python
# 在 src/models/n2p3net.py 加 Split-BN 实验臂
# st_temporal_norm / st_spatial_norm 改为 per-domain BN
# 贡献大小待通过成对实验与机制诊断验证，不预设恢复幅度
```

### P1：可丢弃域残差修正（框架已实现）
```python
# 在 src/models/adapters.py 的 FeatureDomainTransfer
# 训练时以概率 p 丢弃 source domain 残差
# trunk 单独工作成为合法操作点
# 这修正了 doc 第 8 节反例 8 的缺陷
```

### P2：决策层可微化（direct framework 已实现）
```python
# 在 src/models/candidate_evidence.py
# 把 CandidateAccumulator 从后处理聚合改为可学习模块
# ρ_ω(A_d,r, B_d,r, Q_d,r, N_d,r, coverage_d,r)
# 第一版用线性/温度模型，避免小数据下任意 MLP
```

---

## 六、数学公式总结（供下一步 doc 改写）

### 6.1 可丢弃域残差

源域训练时：

$$
Z_{inv} = H + \rho_{inv} \tanh(a_{inv}) A_{inv}(H)
$$

$$
H_{d,train} = Z_{inv} + \mathbb{1}_{\text{不丢弃}} \cdot \rho_d \tanh(a_d) \Delta_d(Z_{inv})
$$

其中 $\mathbb{1}_{\text{不丢弃}} \sim \text{Bernoulli}(1-p)$，推荐 $p \in [0.2, 0.5]$。

目标域推理时：

$$
H_{target} = Z_{inv} + \rho_t \tanh(a_t) \Delta_t(Z_{inv})
$$

trunk 单独工作（$\Delta=0$）是训练期被显式训练过的操作点。

### 6.2 决策级损失

对 decision 的第 i 个 epoch 归属候选 $c_i$，证据 $e_i = f_\theta(x_i)$，候选得分：

$$
s_d = \frac{1}{n_d} \sum_{i: c_i=d} e_i
$$

候选 posterior（\(\tau>0\)）：

$$
P(d \mid E) = \text{softmax}_{d\in V}(s_d/\tau)
$$

决策损失：

$$
L_{decision} = -\log P(d^* \mid E)
$$

总损失：

$$
L = L_{trial} + \lambda \cdot L_{decision}
$$

若训练 repetition 曲线，使用
\(L_{prefix}=\sum_{r=1}^{R_{max}}\alpha_rL_{decision,r}\)，并预注册
`alpha_8=1`（固定预算）或 `alpha_r=1/R_max`（整条曲线）。mean 聚合不自动表达
evidence 数量；evidence-aware 的 \(n_{eff}=B^2/Q\)、\(\beta\) 或 \(\tau_r\) 必须作为
独立 arm 由 calibration/meta-validation 冻结。

梯度：

$$
\frac{\partial L}{\partial e_i} = \frac{P(c_i \mid E) - \mathbb{1}[c_i = d^*]}{\tau\,n_{c_i}},\qquad \tau>0
$$

---

## 七、拒绝的方案及理由

### 7.1 横向线性组合
- 不采纳的理由是参数/推理成本、few-shot 方差和缺少预注册专家多样性约束，
  不是“无新表达力”或必然塌缩。多个固定宽度非线性网络的和一般扩大函数类；
  Jacobian、初始化和特征区域不同，不能由输出梯度只差标量推出相同函数。

### 7.2 域绑定专家混合（K=3）
- 混合空间太窄，丢失 P300 共性

### 7.3 端到端多源联合训练（不分阶段）
- identity trap、低频偏置、-21.94pp 负迁移
- SpellerSSL 证据支持分阶段

### 7.4 通用重构预训练（masked reconstruction）
- EEG-FM-Bench 明确指出与下游任务梯度冲突
- 低频偏置损害 P300 瞬态成分

## 九、外部依据与证据边界

- Gneiting & Raftery, “Strictly Proper Scoring Rules, Prediction, and Estimation”,
  JASA (2007), DOI: [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)。该文支持 proper scoring rule 与分类决策损失的概念区分；本项目的 NLL/0-1 反例仍是本地构造，不是该文的实验结果。
- Hansen & Salamon, “Neural network ensembles”, IEEE TPAMI (1990), DOI:
  [10.1109/34.58871](https://doi.org/10.1109/34.58871)。该文支持 ensemble 通过误差相关性影响泛化的基本论点，不证明本项目的横向方案有效。
- Chang et al., “Domain-Specific Batch Normalization for Unsupervised Domain Adaptation”,
  CVPR (2019), DOI: [10.1109/CVPR.2019.00753](https://doi.org/10.1109/CVPR.2019.00753)。该文支持按域隔离 normalization statistics 的工程先例；不支持把 buffer-swap 当作训练期因果否证。

以上 DOI 与题名已由 Crossref API 在 2026-09-03 查询核验；文献均为方法先例，不能替代本项目同 fold/seed、paired CI 与真实 target `hit@R` 证据。

---

## 八、待第三路结果补充（历史研究记录）

少样本适配机制（MAML / MoE / Prototypical / LoRA / RNN over trials）的完整文献综述仍是候选研究方向，不是当前实现门禁。预计补充：
- MAML 在 EEG 的稳定性证据
- 序列决策层联合可微训练的先例
- 参数效率与过拟合风险的最优解

---

## 参考文献

1. Défossez, A., et al. (2024). The Brain's Bitter Lesson: Scaling Speech Decoding With Self-Supervised Learning. arXiv:2406.04328.
2. Hong, J., Mackellar, G., & Ghane, S. (2025). SpellerSSL: Self-Supervised Learning with P300 Aggregation for Speller BCIs. NeurIPS 2025. arXiv:2509.19401.
3. Lin, X., Wu, Y., et al. (2026). The Identity Trap in EEG Foundation Models: A Diagnostic Audit. arXiv:2606.06647.
4. Understanding and Correcting Low-Frequency Bias in EEG Foundation Models. arXiv:2608.01898.
5. Xiong, W., et al. (2025). EEG-FM-Bench: A Comprehensive Benchmark for the Systematic Evaluation and Diagnostic Analyses of EEG Foundation Models. arXiv:2508.17742.
6. Rajpura, P., et al. (2024). From Rest to Action: Adaptive Weight Generation for Motor Imagery Classification from Resting-State EEG Using Hypernetworks. arXiv:2411.09789.
7. Phan, H., et al. (2021). MetaSleepLearner: A Pilot Study on Fast Adaptation of Bio-signals-Based Sleep Stage Classifier. IEEE J. Biomed. Health Inform. arXiv:2004.04157.
8. Dong, T., et al. (2020). Ultra Efficient Transfer Learning with Meta Update for Cross Subject EEG Classification. arXiv:2003.06113.

---

**文档状态**：架构决策已定；P0/P1/P2 framework 已在 `ef7a40e` 落地。真实数据效果和后续 target inner/outer 机制仍须按活动规范单独验收。
