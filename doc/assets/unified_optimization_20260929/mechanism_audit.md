# A1/Q1 机制只读审查与优化接口（2026-09-29）

本分支只读审查源码、规范与冻结清单，没有修改生产代码或运行训练。使用 `astra-research-workflow`（science/engineering）与 `scientific-critical-thinking`。本文件是父报告的原始审查材料，不替代六份活动指南。

## 身份与可归因范围

当前 HEAD 为 `f1a303c42c68173e8b1452ceb26bd84923b8a6aa`，工作区存在大量既有修改、未跟踪实现和测试目录迁移。HEAD 只是基准提交。

逐字节读取 `frozen/unified12_20260918_performance_v4/source.source-only.tar.gz` 后确认以下八个当前文件与执行 v4 完全一致：

| 当前路径 | SHA-256 |
|---|---|
| `src/transfer/source_objective.py` | `3c46c384442971b011724f6acf9063c5d4b3c1166e26a6ffcf90157a393d49ac` |
| `src/models/parameter_groups.py` | `c5634d05049618dae87c8e3bf75b47f2a140ee0f41baa34e25f493e5ee0c286b` |
| `src/transfer/decision_loss.py` | `29660eab1d9aa034a3baf799c6f5274ec5bb9ebf4ea31026bdd44af7ae4d6bc5` |
| `src/transfer/subject_adapter.py` | `3deebd464e02cf1959ed8d7c7da8378fe44b06ab36db7ee897426b22485e7dfe` |
| `src/transfer/calibration_candidates.py` | `f884675d795e6791ab1b18aeb407bf8d7e4d3edb52f1ec88631ba12ca218967d` |
| `src/transfer/recipes.py` | `71783a3a8c072b0aeb1fae2ac6b8960f0d960bf92d4d20f5b515c6c4d4d257bf` |
| `src/research/source_matrix.py` | `341cbee35249566a95918aef2b8fc3ee5871243867dde1998ebd6fb2968fd056` |
| `experiments/run_source_training_chain.py` | `ad22e2c8dbe55ef2a9c4e223e742a0f87264f3766725e925e0c070757622eb14` |

v4 archive 整包 SHA-256 已重新计算为 `c71a4d2109eaed23bea734c76845940bcf05d51a3af4478ac3c1db54fe75fd24`。另读取 `frozen/unified12_20260918_execution_v2/source.source-only.tar.gz`：上表七个 `src/` 文件也与 v2 逐字节一致，`run_source_training_chain.py` 不同。历史首折复用的 runtime_v2 仍属于它自身的执行身份，不能因核心机制相同而重写旧模型来源。

重要限制：A1 是 SupCon+MMD 的联合包，不是两个单组件的独立处理；Q1 在 BI2014a、BI2015a、Huebner2017 仅有 source prior，在 BNCI2014-008 才有 prior+decision。负交互可表述为冻结处理下的差中之差，不能归因为“SupCon 与 decision 必然冲突”。

## 真实调用链

1. `src/research/source_matrix.py:15-38` 固定残差架构 R=1，两源臂为 `(S,M)=(0,0)/(1,1)`；默认辅助权重均为 0.05，温度 0.1。不存在单独 SupCon 或单独 MMD 的本轮主臂。
2. `src/baselines/deep.py:1304-1321` 在同一 forward 的真实特征上把 CE 与辅助损失相加后一次反传。CE 的 row importance 与辅助项的 risk reduction 并非同一东西。
3. `src/transfer/source_objective.py:298-343` 将时序特征展平并 L2 归一化，仅在辅助项中使用；`353-377` SupCon 正例是跨域且跨人同类，异类才是负例；`380-421` MMD 对有至少两唯一行的域对/类别平均。重复采样的同一物理行仅在辅助项去重。不能声称主分类器也丢弃幅值或时序。
4. `experiments/run_source_training_chain.py:1101,1118-1151` 以 A1 参考源进行选轮及 lambda 开发，选出的共同 source exposure 与 lambda 再供 A0/A1 使用；`src/research/source_selection.py:184-220` 依据源内域/参与者等权 AUC，近似并列取较小 lambda。当前是“共享选择策略下包效应”，不是各方法各自最优超参数的比较。
5. `src/transfer/recipes.py:198-217` Q0/Q1 都采用 target-prefix 输入统计、calibration BN、可训练 affine、关闭 dropout、全模型固定 90 轮；Q1 添加 prior，存在真实 membership 才添加 decision weight=1。
6. `src/transfer/calibration_candidates.py:154-263` 从物理 timeline 构建授权组，禁止用 K 外行补完整组；direct 可按固定 repetitions 分窗，row/column 保留原 selection 的完整重复数；风险权重先人再原 selection 再窗口平均。
7. `src/transfer/subject_adapter.py:1663-1733` 把授权 decision 映射到 fit 范围，每个 trial batch 抽一组并追加组行。同一工作集同时供 Q0/Q1，BN buffer 被 `:795-800` 门禁要求冻结；重复 forward 导致额外 BN 更新并不是本轮已证 bug。
8. `src/models/candidate_evidence.py:65-102,148-191` 与 `src/transfer/decision_loss.py:692-725` 共用物理 membership 求和，候选 CE 对 `M^T e/T` 计算。训练 CE 允许平票；缺失或不可辨识组没有可训练 loss，但保留请求账本。
9. `src/models/parameter_groups.py:37-99` prior 保存实际源参数中心，对卷积/线性权重用源 RMS²+fan-in 方差，其他参数用单位尺度，先 tensor 平均、再 role 平均后相加；`subject_adapter.py:1172-1185` 三项损失一次反传。它是坐标正则，不是源函数保持保证。

## 五条可证伪假设

### H1：decision 风险的物理尺度与 trial 风险不一致，可能改变冲突方向及优化器历史

令单 trial evidence 为 `e_i=z_i1-z_i0`，组归属矩阵 `M∈{0,1}^{N×V}`，目标列 `c`：

`L_T=sum(w_i BCE(e_i,y_i))/sum(w_i)`，`L_D=CE(M^T e/T,c)`。

精确梯度为 `g_Ti=w_i(sigmoid(e_i)-y_i)/sum(w)`；`g_D=(M softmax(M^T e/T)-M[:,c])/T`。两项优化同一神经网络，不意味着它们在非零 evidence、不同工作集上方向一致。

最小候选：用冻结物理组在零 evidence 计算 `a_D=||g_T,D(0)||₂/||g_D(0)||₂`，试验 `L_T+gamma_bar*a_D*L_D+lambda*Omega`。它不使用拟合结果，不按库名开关。对完整 R10、6×6 row/column，`a_D=0.015`；direct 九选一 R5 为 `0.05625`。这是 09-16 已提出但未实施的候选，不是新发现的最佳系数。

必须保留的反证：零 evidence、同一对称工作集下两梯度共线；Adam 对正比例梯度首步近似不变。因此 66.67 倍逐 logit 梯度不能解释成 66.67 倍参数步长或损害。应同时报告同参数点下分项参数梯度 cosine、对 CE/decision 梯度的投影、固定 Adam 状态下的一步更新差。若规范化只改变 loss 数值却不改变方向、更新或独立后续风险，H1 不获支持。

另一个可证边界：`M^T delta=0` 时 `L_D(e+delta)=L_D(e)`，但 trial loss 可改变。row/column 的候选分数为行分数加列分数，36 类 CE 恰好分解为 6 类行 CE 与 6 类列 CE；36 候选不提供 36 个独立监督单位。decision CE 与 trial CE 不是同一个目标的简单重复。

### H2：少量真实 selection 被反复优化，导致 decision 梯度高方差与目标身份覆盖不足

冻结 K200/K800 原方案中，BNCI 分别只有 1/6 个完整 selection。来源见 `doc/evidence/chain_mechanism_20260916.zh.md:113-122`；当前 builder 与其一致。每 batch 抽一完整组且有放回 (`subject_adapter.py:1707`)。90 次或 180 次抽组不会创造新目标状态，R 子前缀也不是新的独立 selection。

数学可证点：对 6 组均匀抽 n 次，每组抽中次数方差为 `n*(1/6)*(5/6)`；按每轮完整组遍历可去掉覆盖次数不平衡，但不能增加有效独立样本。最小优化候选是独立 decision 风险调度与显式组级 exposure 账本，并保持原 trial exposure/optimizer 更新数，先分离 sampling variance 和 loss scaling 两个问题。

反证/判定：若固定所有合法组的准确期望梯度与原抽样估计相符，且不同抽样仍稳定，则抽组方差解释减弱；若 G=1，改成遍历本质无益。不要为“更多 decision”拆分同一 selection 后增大独立样本数。另设固定 G/目标覆盖的校准预算只能作为新 estimand，不能替换原 same-K 对照。

当前产品主指标已为 `hit@all`（`doc/research_program.zh.md:10-15`）。不能机械照搬旧文档的“把 R5 作为主训练 horizon”；原 BNCI 完整 R10 已匹配其 all-repetition 输出。若研究早期重复性能，才另设组内加权前缀风险，统计仍按原 selection 聚类。

### H3：A1 源参数坐标与 Q1 prior 的尺度共同变化，且 prior 没有约束先改变的统计坐标

当前 prior 为 `Omega=sum_r mean_t[||theta_t-theta_St||²/(2*n_t*s_t²)]`，其中 `s_t²=mean(theta_St²)+1/(3 fan_in)`（普通权重）。A0/A1 的参数中心和 `s_t²` 都不同，即使使用相同 lambda，也不是相同的二次惩罚矩阵。该事实提供负交互的竞争解释，不等于证明它是原因。

此外 target-prefix 标准化和 calibration BN 在更新前已经改变模型函数；`theta=theta_S` 时 penalty=0、penalty gradient=0，并不推出 target 坐标中的源输出不变。`subject_adapter.py:232-234,915-942` 和 `parameter_groups.py:64-85` 为直接依据。

最小可归因设计：固定一个 source、原统计与预算，做 prior 开/关 × decision 开/关。先量化每 role penalty Hessian/梯度、权重移动、BN/输入坐标改变前后预测；只在明确冲突证据下考虑 backbone-only prior 或固定 fan-in-only metric。不要直接增蒸馏、在线门控或库特定 lambda。

反证：等价网络重参数化可使函数保持而参数距离不同；输入统计改变也可使零参数距离对应不同函数。反过来，坐标 prior 不保证函数保持并不说明它无正则价值。若 lambda 实际选为 0，则该拟合中 prior 不能被归因为负效应；必须读取每折最终 selection。

### H4：跨域同类全拉近约束可能去掉可迁移的域条件形态

SupCon 的正对条件只有二元类、人、域；MMD 平均对齐每域类内分布。它们没有把 SOA、目标概率、注意状态、时延或参考差异当作条件。对真实 ERP，`P(h|y,d)` 可以是多个有用形态；类标签相同不保证应映射到同一方向/分布。

可证反例：设两个域的正类波形分别包含互补、对新域可用的时间成分，要求方向完全相同的辅助目标与保存这些成分的 CE 最优解可冲突。反证同样成立：若域差异全是 nuisance 且共享充分统计存在，则对齐可能有益。当前没有模型坍缩或缺容量的证据。

最小候选：维持当前小编码器与 CE 主线，先做 CE、CE+SupCon、CE+MMD、CE+两项的来源内独立诊断；源内留出上检查梯度冲突、类间 margin、时延/幅值扰动的可辨性。若固定权重的末期约束被证实冲突，再试唯一变化的固定退火至 0；预算、采样、架构和 target route 固定。不能凭 A1 包失败删除某单组件，也不能用 target test 选择辅助权重。

### H5：辅助 risk 的实际域/类别权重受 batch 可估性影响，未必对应期望宏平均目标

`source_objective.py:370-377` 对 active anchor 平均；positive 对当前 batch 可用同类跨域/跨人样本平均。`397-421` MMD 只对样本数至少 2 的域对/类别平均。uniform_rows 源 batch (`run_source_training_chain.py:709,1123`) 中，大域和常见类更易成为 active anchor/cell；“各 active cell 平均”不等于“所有请求域对等权的无偏风险”。CE 可拥有单独 importance 权重 (`deep.py:1204-1218,1313`)，这些权重没有传给 auxiliary reduction。

最小候选先是测量：按域×类记录请求/活跃概率、anchor mass、pair mass 与不同项梯度贡献，而非立刻换 source pool。若目标确为域/参与者等权，可保留 CE sampler，将辅助批次/权重作为单独声明的采样设计；必须推导 inclusion probability，不能用随 batch 自归一化权重冒称无偏。

反证：若每个域/类别始终完整可估、有效权重已接近冻结目标且梯度不偏向大域，则 H5 弱化。当前日志已有 active counts，但缺少足以还原每域/类贡献的账本，因此不能从源码直接量化它解释了多少性能差。

## 最小高价值实现规格（尚未实施）

优先做分量诊断 + 物理组 loss 尺度，不同时修改 source pool、BN、温度、学习率、训练时长、窗口或网络结构。

```python
def frozen_decision_loss_contract(group, target, balanced_trial_rule):
    # Reject missing/invalid physical schedule through the existing owner.
    M, y = group.membership, group.binary_labels_for(target)
    g_trial = balanced_trial_gradient_at_zero(y, balanced_trial_rule)
    g_decision = M @ uniform_candidate_prior() - M[:, target]
    # T=1 in this research arm; include T explicitly if a later arm changes it.
    scale = norm(g_trial) / norm(g_decision)
    return record(scale, membership_digest, target, risk_unit="original_selection")

trial_logits, group_logits = one_declared_workset_forward(...)
trial = matched_trial_risk(trial_logits)
decision = membership_decision_loss(group_logits, group, target)
prior = source_parameter_prior()
loss = trial + gamma_bar * frozen_scale * decision + lambda_ * prior
# Diagnostic-only: evaluate component gradients at the same theta and workset;
# clone fixed optimizer state for the theoretical one-step comparison.
```

需联动 owner：`SourceObjectiveConfig`/`SourceObjective`；`TargetRecipe`/`SubjectAdaptationConfig`；`calibration_decision_groups` 及其 risk ledger；`membership_decision_loss` 的 loss 版本与输入合同；`SubjectAdapter.fit/_refit_full_prefix` 两条路径；`target_experiments` task/manifest 与 source-only 选方；`TrainingRunContract`、checkpoint、个人 predictor 保存重放；结果分析器的 A/S/M/prior/decision/normalization 分量标签。不要在 CLI 分散计算 scale。

准入反例：任意候选联合置换；完整对称组 offset 不变性与非对称反例；缺失/QC 后不可借 scale 恢复可用性；zero-logit 规范化与非零梯度；row/column factorization；同 selection 多前缀的总 risk mass 不增加；prior 中心零梯度及改变统计的函数反例；Adam 正比例首步反例；空辅助 batch 完全退化至 CE。实际保存模型重放需覆盖新 loss/state 合同。

## EA 接线补审：不能把 fixed-checkpoint target whitening 当作标准双侧 EA

### 当前可核验代码

- `src/data/euclidean_alignment.py:199-220` 拟合平均未减均值 trial 二阶矩（可 shrinkage），求逆平方根；`:248-256` 只做 `X'=C_fit^{-1/2}X`。没有 target-to-source covariance coloring。
- `experiments/prepare_euclidean_alignment_cache.py:108-143` 支持 source/per-domain 或声明范围的整 cache 拟合与变换；`:157,164,174` 保留 epoch preprocessing，另写 `provenance.data_alignment` 和 lineage。`:145-147` 要求完整观测通道。
- `experiments/run_candidate_promotion_matrix.py:2475-2527` 审核 source-arm cache 是否有 EA provenance、schema、fit scope、reference。该准入有价值，但不是整个 target 推理路线。
- 已检索 `SubjectAdapter`、`calibrated`、`target_experiments`、`checkpoint` 和当前 unified runner，未发现 EA state 的目标校准 fit/独立 test replay 接线。
- `src/transfer/checkpoint.py:525-538,587-619` 比较 preprocessing execution、epoch 声明、通道顺序、reference；没有显式 `data_alignment` route 兼容检查。`src/data/epochs.py:671-685` 的 execution identity 只读 `preprocessing_execution`，不会把新增 `data_alignment` 自动纳入。故仅凭这些现有门禁，不能声称源 EA 权重与目标坐标已经一致。

### 正确路线身份

标准双侧 EA：对每个合法源 fit 单位 d，先 `X_Sd'=C_Sd^{-1/2}X_Sd`，在该坐标上重新训练源模型、源 input standardizer 与 BN；目标只用授权 calibration EEG 估 `C_T`，冻结 `C_T^{-1/2}` 后作用于独立 test。模型架构、原 raw rows、训练预算可以配对固定，实际源 checkpoint 必须有 route 身份；目标不能使用 plain 源 normalizer 假装坐标相同。源内验证的 covariance fit 单位及合法可见行也须预声明。

如果必须固定原 plain checkpoint，可研究另一个明确命名的二阶矩 transport：在模型实际输入统计坐标中计算 `A=C_S^{1/2}C_T^{-1/2}`（必要时另声明均值平移），再用冻结 source function；这不是现有 whitening 函数，也不是已实施 EA 路线。混合多源不存在唯一天然 `C_S`，需冻结聚合权重与参考坐标。输入标准化、reference 和空间矩阵一般不交换，所有算子顺序必须绑定。

简单反例：源与目标本就同为 `C=diag(4,1)`，线性分类器 `w=(1,1)`。输入 `x=(2,-1.5)` 原 logit 为 `0.5`；只白化目标后变 `-0.5`。双侧白化后重训对应 `w'=(2,1)` 才保持 `0.5`。若 source=target，coloring 为恒等。这个算子反例只反驳“形状保持且 checkpoint 相同就自动公平/等价”，不量化实际 N2P3 效应。

另一个必要反例：源与目标 covariance 都为 I，但 class-conditional mean 方向不同，EA 是恒等且不解决标签关系旋转。因此 covariance residual 小不是迁移效能验收。

`doc/evidence/zero_shot_transfer_routes_20260920.zh.md:45,103,112,117` 中固定 source checkpoint/state replacement 的文字应在新方案中校正：同 checkpoint 只能用于明确的目标侧变换实验；标准双侧 EA 应固定 architecture/budget/seed/data split 并绑定路线专属 checkpoint。历史 EA-target-cal 的弱/零收益也不能自动反驳尚未核验的双侧 EA。

EA 最小门禁需统一封装于可序列化 state/route owner：输入单位与reference、观测子空间/通道角色、矩阵维度/condition、fit row identity/time support、source与target统计坐标、算子顺序、source route digest、normalizer/BN/checkpoint identity；独立 test 只能 transform，不重新 fit。原始缺导不能补零后当作 EA 完整导；若在固定共同观测子空间做 EA，源/目标同子空间及秩必须证明。

## 本分支验证记录

执行了 Git 身份读取、当前/冻结八文件逐字节及 SHA-256 比较、源码与调用链只读检查。公式与反例在本文件为解析推导，未把它们声称为新增已运行数值测试。本分支未运行模型拟合、未调用云端、未消费新测试机会、未修改原数据及旧结果。父任务如执行独立数值探针，应另链接其结果。
