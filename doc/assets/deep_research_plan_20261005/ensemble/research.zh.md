# 神经与连续 ERP 专家：评分、融合和可验收实验规格

日期：2026-10-05。覆盖主计划 B00、B02、B03、B04、B12，以及愿望清单第七至十组。本轮交付研究规格、确定性数学反例和历史预测只读复算；没有拟合新神经模型、执行新效能矩阵或改变生产默认。原始工作树只读，研究文件位于隔离树。本文是研究证据附件，活动规范仍由六指南持有。

## 1. 裁决与历史证据的适用范围

保留 N2P3Net full-fine、同结构 scratch、Toeplitz、正确 ERP 原型增广 xDAWN 和 raw AR8。先建立**同一物理 decision 的不同合法数值视图**，再比较预声明的候选概率平均。等权是低选择成本的开发锚点，不能保证比最好单专家更准确。温度、非等权、按 K 的权重和质量 gating 都是另一个学习程序，必须独立归因。

原四周方案的“hit ≥ 0.771、NLL 不恶化”改为：用新同批三专家锚点做配对差值，保留 requested 队列、失败和 seed 结构。历史绝对数仅定位开发候选。原第一周任务需要 B00 评价、B01 物理输入、B02 专家评分合同先闭合；AR、先验、空间和 gating 会改变融合输入，不能称与混合完全独立。四库准入不等于四库都有已接通的 candidate 真值。

### 1.1 原 Phase 4 已复算的事实

[数学与复算结果](math_results.json)从原 `D:/BCI/n2p3-net/experiments/runs/operator_research_20261004/phase4_flexible_spatial_hybrid_v3/result.json` 读取 576 条保存预测，逐条重建 48 条三专家概率、winner、correct、NLL，再做 participant-macro：

| 项目 | 本轮只读复算 |
|---|---|
| 主候选 | `equal_raw_toeplitz_xdawn_mixture` |
| 实际组成 | `raw_ar8_lambda_1`、`toeplitz`、`xdawn`；各 1/3 |
| 统计单位 | 8 人；每人 6 个依赖的 selection 留一 fold |
| 宏平均 hit | 0.7708333333333333，即 37/48 |
| 宏平均 NLL | 1.0694548780609707，自然对数 |
| 相对 Toeplitz | 救回 3 次，丢掉 3 次，总 hit 相同 |
| 数值最大概率 tie | 本轮按 `rtol=atol=1e-12` 检查，0 次 |
| raw lambda=1 单专家 NLL | 原 Phase 4 报告 2.9607467761510025 |
| 原始预测文件 SHA-256 | `5dc85b86e791376836c78da7585fc9d593725d0268062c4c42b3bd2292c29ca2` |
| 原报告 JSON SHA-256 | `83906bb8a5aedeb30f2e9095271f0fb33977b85805b0d9220875625e5b262580` |

实际范围由原 `run_calibration_study.py:48` 的 `selection_views` 定义：C800 是授权上限；前 720 flash 构成 6 个完整 selection，80 个授权标签未用，每次在其他 5 组上拟合 **600 flash**、在 1 组上评分。6 个 held fold 是 C 内开发，不是 K800 拟合后 T 上的 48 个独立样本，也不是 8×6 名参与者。原算子报告中 raw AR8 NLL 6.162 来自此前不同研究臂；不能嫁接到本次 lambda=1 融合的成分解释。

### 1.2 实际执行来源仍有缺口

原 `phase4_hybrid_study.py:108` 调用从 `run_calibration_study` 导入的 `executing_files()`；后者 `run_calibration_study.py:42` 的实现仅捕获 `src/**/*.py`、自身、`continuous_model.py`、`selection_erp_model.py`，遗漏真实入口 `phase4_hybrid_study.py`。本轮检查 `result.json.source` 中该入口不存在。原 `phase4_manifest.json` 虽列入口哈希 `82afb55809da01a36011a390bc0a62dcc1b4abd770c465d69d0facec08db908b`，但事后当前文件哈希不能补证该入口在当时执行。原 helper 当前与 runtime 的哈希均为 `f93ad8f6027a46704672ac0650e0e97df77dffb0efe62e7d39b95d5084bac1aa`；当前 manifest 文件哈希为 `e5b4204c78e11377e68ee0e598125eacdd18ad77452962555649fbf68c2f8a2e`。逐文件关系保存于 `math_results.json.phase4_read_only_audit.provenance_files`。

保存 plan 的 `mixtures` 仅列 lambda hyperprior 和 raw+Toeplitz，未列 raw+Toeplitz+xDAWN；当前脚本中第三臂是固定等权，也不足以独立证明运行前预声明。因此本轮允许结论是“保存预测可复算、部分执行来源已绑定、三专家是历史开发候选”，不声称第三臂预声明或完整执行冻结已证实。原文件不改写。

新准入方式：在新物理支持下重建同批 C3 与 C4，冻结完整 plan、真实实验入口及全部导入研究 helpers；核对启动与每 worker 的文件哈希，持久化每个专家模型、预测、C 内选择轨迹和重载结果。新源快照不得只沿用生产 `src` 清单。该闭合工作属于 B00/B14，先于新效能比较。

## 2. 真实源码与神经专家的候选概率

已检查当前隔离源码，而非沿用旧技能的路径或架构裁决：`models/n2p3net.py:578` 定义小网络，`:1260` 返回双类 logits；`SubjectAdapter.predict_logit` 在 `transfer/subject_adapter.py:1788` 返回 **z1−z0**。`CalibratedPredictor.predict_logit` 在 `transfer/calibrated.py:764` 和 `fit_calibration_candidate` 在 `transfer/calibration_candidates.py:504` 共用实际拟合路径。`models/candidate_evidence.py:148` 的 `membership_decision_scores` 已集中完整物理 membership、词表、softmax、tie 和不可用原因。`transfer/decision_loss.py:692` 共用同一 scorer 的可微训练损失。

本轮直接构造 `N2P3Net(6,n_times=179,pooling_mode="full_unfold")` 为 **1202 参数**，相同配置 `global_average` 为 858。参数量来自具体 architecture，不用文档里的 1279 或技能中的“大模型”概括实际 recipe；附加 head、residual、不同输入和读出须重新记账。

### 2.1 从二分类 logit 到 LLR

令 `d_i=z_i1−z_i0`，训练分布 target 比率为 π，类别损失权重为 a1/a0。理想加权 CE 解满足：

\[
q_1(x)=\frac{a_1\eta(x)}{a_1\eta(x)+a_0(1-\eta(x))},\quad
d(x)=\log\frac{f_1(x)}{f_0(x)}+\underbrace{\log\frac{\pi}{1-\pi}+\log\frac{a_1}{a_0}}_{\rho}.
\]

因此 trial LLR 候选为 `ell_i=d_i−rho`。当前 target full-fine/scratch 在 `subject_adapter.py:1474` 以 `a1=n0/n1` 加权并保存 `training_prior_`，理论 rho=0；不应再重复减一次 `logit(π)`。原 Phase 4 的 `SelectionERP.score` 在研究 helper `selection_erp_model.py:297` 对 LDA `decision_function` 减去了拟合先验 odds，因为该 LDA 未使用相同的平衡 CE。

这只是风险最优点的推导，有限模型不自动校准。source 的层级风险、样本抽样和多个域混合会改变有效 π 与类条件分布，需由 `baselines/deep.py` 的实际 risk/sampling record 推导，不能使用整个缓存的原始 target 比率。类条件 shift 不会被一个先验 offset 消除；还要在合法 C 内独立组检验概率质量。只允许保存明确的 logit 语义、有效类别质量和校准状态，不给任意 neural score 贴“真实 LLR”标签。

### 2.2 完整 membership 聚合

令 M_iq=1 表示第 i 次实际刺激包含候选 q，词表 Q 固定。假设条件独立、各 trial 的 LLR 适当，则：

\[
S_q=\sum_i M_{iq}\ell_i,\qquad
P_m(q)=\operatorname{softmax}_q\left(S_q/\tau_m+\log\pi_q\right).
\]

独立 Bernoulli 似然的枚举给出同式，因为所有 q 共用的 `Σ_i log(1−sigmoid(ell_i))` 在归一化中消掉。BNCI 6×6 的 12 种刺激属于 **36 个物理候选**，不能对 12 个 flash code 做 softmax 后称 36 选一概率。BrainSync direct membership 属于九候选，不能从校准中出现的候选集合缩小词表。raw 连续模型直接输出候选联合 log evidence，它不需要再次经 trial membership 累加，但必须声明与 trial scorer 相同的物理 decision 和词表。

正 τ 在单专家均匀候选先验下保持 argmax；它改变置信度，且在概率混合中能改变最终 argmax。非均匀候选先验在 `S/τ+log πq` 中只加一次；不能既按 target binary π 改成后验 odds，又重复添加候选先验。τ 应针对完整 decision 的重复数、候选数和评分协议校准，不能把 R1 的 trial 温度无条件套到 R10；不同 Q 的均匀 NLL 为 log Q。

候选曝光数 n_q 相等时，常数 trial offset 只改变公共 `rho*n_q`，排序不变；曝光数不等时，它可以改变 winner。当前 `MembershipDecision` 检查实际排程而不假设任意 set membership 等曝光，故 offset、先验和 sum/mean 不能相互替代。重叠 epoch 更使条件独立不成立，softmax 是声明的评分近似，概率校准须由 held group 检验。

### 2.3 梯度及监督量

完整 decision 的 NLL 为 `L=−log P(y)`：

\[
\frac{\partial L}{\partial d_i}
=\frac1\tau\sum_q M_{iq}[P(q)-\mathbf1(q=y)],
\qquad \frac{\partial L}{\partial z_{i1}}=\frac{\partial L}{\partial d_i},
\quad \frac{\partial L}{\partial z_{i0}}=-\frac{\partial L}{\partial d_i}.
\]

这个梯度基于完整物理归属，不是“一次 flash 只属于一候选”的错误公式。若采用 mean，梯度另含 n_q 的分母，属于另一处理。当前统一评分是 `membership_llr_sum/1`，不能为了让融合数值接近就改回 mean。decision 标签数远少于 trial 数，额外 decision loss 与 CE-only full-fine/scratch 分开归因，不把新增 loss 同时并进“新增神经专家”首轮比较。本轮 autograd 与独立 NumPy/Bernoulli oracle 已比对，见检查 `membership_gradient`。

## 3. 混合、校准、互补性与专家失效

### 3.1 概率平均与 logit 平均是不同模型

算术概率池为 `P(q)=Σ_m w_m P_m(q)`。它满足 Jensen 界：

\[
-\log P(y)\le\sum_mw_m[-\log P_m(y)].
\]

该界针对**专家 NLL 的加权平均**，不针对最好专家，也不约束 hit。例：正确专家 `[.51,.49]` 加入错误专家 `[.01,.99]` 后等权池为 `[.26,.74]`，答对变答错。两次 decision 的反例还给出 hit 从 .5 到 0，而 NLL 从约 9.55 降到约 1.38。原 Phase 4 的“救回 3、丢掉 3”是现实开发样本中同一现象的实例。

logit 平均 `softmax(Σ_m w_m s_m)` 等于归一化几何概率池 `Π_m P_m(q)^wm`，强调专家共同支持，接近“同意才给概率”的机制。对 `[.8,.15,.05]` 和 `[.05,.3,.65]`，算术池 winner 是 0，几何池 winner 是 1。它不是可替换的实现细节。原 Phase 4 未做几何池主比较；新首轮固定算术池，几何池仅作预声明机制对照。

“两个群组等权”也不等于“四个专家等权”：`.5*NN+.5*C3` 给 NN .5、每经典专家 1/6；C4 给四个专家各 .25。首轮规定 C4 采用后者，避免把不同 neural 质量偷偷变成不同配方。

### 3.2 只在 C 内学习温度与权重

令 w=softmax(u)、混合真值概率 Py，则：

\[
\frac{\partial[-\log P_y]}{\partial u_m}
=w_m\left(1-\frac{P_m(y)}{P_y}\right).
\]

若 quality gating 为 `u=B h`，则梯度是上式乘 `h^T`；h 仅用当次 decision 的预声明标签无关输入特征，以及合法 C 中冻结的质量尺度。高 softmax 置信度本身不证明专家可靠，不能按 held 正确性选 expert。若端到端联合调整专家分数，梯度含责任 `r_m=w_m P_m(y)/P_y`：`∂L/∂s_m=r_m(P_m−onehot_y)`，容易压低已经很弱专家的训练信号。首轮只冻结专家再融合，不引入隐含联合网络。

固定专家概率后，最小化 `Σ_g−log(Σ_m w_m P_mg(yg)) + λ||w−wequal||²` 在 simplex 上为凸目标，每组Hessian为 `p_y p_y^T/P_y²` 加收缩项；但没有 hit 改善保证，也不能克服只有几组的统计容量。令β=1/τ，温度梯度为 `∂NLL/∂β=Σ_q P(q)S_q−S_y`，本轮另与autograd核对。λ、τ 范围、质量特征、按 K 共享方式和所有 weighting family 都在 outer held 之前声明。允许先选 hit 非劣的候选，再用 NLL 排序；不能选低 NLL、明显降低 hit 的模型。

**完整 C 内 nested group OOF 生命周期**：

1. 固定 outer C/T；T 的 EEG 不拟合任何统计，T 真值只由最终 evaluator 持有。
2. 在 C 中按独立 selection/session 分组，原始支持重叠或同 decision 的前缀放同组；隔离响应尾和滤波支持。
3. 对每个校准内评估组 V，用 C\V 的再内层 grouped cross-fit 产生专家 OOF；每个 OOF 分数的网络、xDAWN prototype、covariance、normalizer、BN、QC、raw noise 与先验只用自己的训练组拟合。
4. 温度/weight/gating 在这些 OOF 行上学习；候选超参数在 V 上比较。V 不进入拟合它所评估的 calibration map。不能在同一 OOF 上调许多 gate 并把训练 NLL 当选择依据。
5. 选择一个规则后，在全部 C 的 cross-fit OOF 上拟合最终校准 map；所有专家在全 C 独立 refit。冻结 map、专家、raw view、候选词表和失败规则，只在 T 评分一次。

在历史六 selection 开发中，outer held 留一后只有五组；再内层会产生至多五个独立 meta 训练样本，不是 600 个独立 gating 标签。4 个温度+3 个自由权重+多个 quality 系数在此规模不可识别。先使用固定等权与共享/强收缩温度；复杂 gating 必须累积独立 C 组，或在其他获准 source participants 的完整 decision 上学全局 policy，再固定应用目标，且保持 source 排除。

K200 在该 BNCI 分组中只有一个完整 selection，无法进行目标内独立 candidate cross-fit。该档继续 trial 训练，但候选温度/权重用固定等权或 source 内冻结规则，明确 `candidate_calibration_not_estimable`。拆同 selection trial 当 group 会让 held membership 和重叠 ERP 泄漏。K800/K2000 是否可选由实际独立组数决定，不从请求 K 猜测。

### 3.3 互补性以得失表验证

每个 neural 与 C3/C4 的 paired decision 保存：单方对/错、两方都对/错、救回、损害、候选 margin、错误时置信度、paired NLL 差，以及边界/通道/实际 K 分层。报告 `rescue−harm`、负迁移参与者比例与 tail；trial score 相关低或 disagreement 高只能说明差异，不能证明互补。把“存在某个 oracle 专家答对”列为能力上界，不作为可部署 hit。

full-fine+C3 相对 scratch+C3 估计源迁移对混合的条件贡献，前提是 NN 输入、架构、target 预算、优化、mask 和 seed 匹配。比较 C4 对 C3 时保留相同经典专家状态；C4 不得因为加入 NN 而重新选择 AR 阶数、lambda、xDAWN nfilter 或边界 penalty。

### 3.4 失效策略只在一个 Module 中解释

首轮 `all_declared_experts_required`：任一专家 expected fit/score failure 或完整物理证据不足，则整个 requested decision 不预测；保留 error、coverage 与 operational hit 分母。共同原始事件缺失，不能从其他专家猜成完整 decision；有限 prior-only posterior 也不产生 EEG prediction。invalid source identity、未授权时间支持、词表错序等是合同错误，启动前拒绝，不归类为低质量模型后继续训练。

未来可另立 `available_expert_renormalization`：只按**整个专家**可用性重归一权重，绝不按候选删列、填 uniform 并静默继续。权重学习也须用相同失效机制的 OOF；记录每次 activated expert mask、原权重、实际权重与失效原因，并同时报告严格 C4 和降级策略。这个程序可提高 operational coverage，但也可能选择性保留容易组；不能沿用 C4 名字替代其缺失项。避免每层 `try/except Exception` 和隐式 fallback，保留一个明确的期望失败类型与一个出口。

## 4. 小 Interface 与完整调用链

建议新增 `src/transfer/ensemble.py`，与主规格统一对外 `FittedEnsemble.fit(calibration_views, frozen_program)` 与 `FittedEnsemble.score(unlabelled_decision_views) -> CandidateEvidence`，保存/重载沿用现有 numeric state 模式。它承担原愿望的EnsembleScorer角色，不增设第二个同义类或fit入口。不要新建泛化 registry、每专家一套 runner 或多个 fallback 检查层。内部 Adapter 的真实变化包括 neural trial score、经典 epoch score、raw continuous log evidence，至少三类 Adapter 已有实际依据。

`FittedEnsemble` 持有并隐藏：完整物理任务、各专家 view/fit state、LLR 映射、τ、候选先验、pool 权重、失败策略与 frozen candidate ordering。score 不接收 held 真值，evaluator 才将结果与 truth 合并。每专家记录 `score_kind`、`probability_calibrated`、target physical event digest、独立 numeric input digest、fit scope、source origin、artifact hash；同一个共同物理任务不能只靠文件名或 candidate 长度来 join。

| 真实调用处 | 后续必须联动的行为 |
|---|---|
| `src/data/candidate_task.py` | 复用 `MembershipDecision`；完整词表、实际 flash 排程、组和 availability；不复制 validator |
| `src/models/candidate_evidence.py` | 复用 membership sum/softmax/tie；NN/LDA 提供正确 logit offset；raw Adapter 对齐联合 evidence |
| `src/transfer/recipes.py` | 增加唯一声明的 ensemble recipe；source/target/τ/pooling/失败策略进入 identity；不由 CLI 重写默认 |
| `src/transfer/calibration_candidates.py` | 共用 full-fine/scratch/ERP 拟合与 `CalibrationCandidateFitCache`；不同 inner group state 禁止复用 |
| `src/transfer/epoch_selection.py` | 复用合法 raw inner view 生命周期；候选 grouped nested cross-fit 是新增能力，现接口不能当已经具备 |
| `src/transfer/continuous_decision.py`、`raw_continuous.py` | 复用连续模型状态/score；旧研究 AR8 `OperatorGLS` 需明确 Adapter，不能重标成生产 model |
| `src/transfer/checkpoint.py` | NN source origin、真实 target 排除、输入声明/normalizer/BN 与实际 source numeric identity |
| `src/research/comparison.py` | `evaluation_input_identity`、`ContrastPlan` 区分共同物理证据与允许变化的数值视图 |
| `src/research/evaluation.py`、`transfer/result_reuse.py` | EvaluationRunContract 与 actual executed source；NN、C3、C4 复用键不同，旧历史结果不能贴新身份 |
| `src/transfer/target_experiments.py` | 计划产生全部 requested cells、K0 可达性、同结构 scratch 与 source→target seed；fit failure 留分母 |
| `src/transfer/calibrated.py` | 后续个人包必须携带全部专家与校准 map，重载只读；当前尚无 FittedEnsemble 消费者 |
| `src/transfer/outcomes.py`、`src/stats/transfer.py` | 单一 requested failure/tie/incomplete ledger，participant/seed/library 分层；不另建一套统计脚本口径 |
| `experiments/run_erp_comparator.py` 与实际 raw research runner | 相同 plan 连接全部臂；真实入参/执行脚本和导入 helper 被快照覆盖；保存单专家和混合模型重放 |

纯函数 scorer 最值得集中的是数学与失败语义，不是再包一层现有 epoch predictor。推理内部数据不含 target 字段；模型/Artifact 验证完成后使用可信对象，避免下游重复解析 dict、重复 hash 同一文件和重复 defensive validation。请求入口、physical join、保存重载三个真实不可信 seam 保留校验。

### 4.1 物理与数值身份关系

同 target decision 的 raw AR8 原生 256 Hz 连续输入、Toeplitz 无 baseline epoch、xDAWN/NN mean baseline epoch可以有不同数值数组；保留同 raw 来源、授权时间范围、物理 flash、truth 与 candidate vocabulary。`ContrastPlan` 显式声明 baseline、resampling、spatial 或 raw support 的变化。该比较估计完整算法 route，不称纯 model architecture 效应。

full-fine 对 scratch 要共享完全相同 target numeric input，才能估计源初始化收益。NN source 与 target 则应有不同 participant、记录和 raw digest；要求它们数值/物理事件相同会抹掉合法迁移。source/target 的**输入声明**必须兼容，source 旧 forward checkpoint 不能重贴新 selection 独立双向身份；不兼容就按新 source route 重训，并为 scratch 保留同 target view。

结果复用比模型声明兼容更严格，绑定**实际** source fit、target fit/score 数值与执行身份。融合 join 要核对同 decision 的物理 key，不能要求所有专家的 numeric hash 相等，也不能接受“行号相同”作为物理相同证明。raw 完整 selection 与已裁响应尾的 epoch mask还要共享预声明 requested 范围；若不同质量规则改变可用证据，作为 view/availability treatment 明记。

## 5. K0、四库、seed 和产品指标

### 5.1 K0 可达性

| 分支 | K0 允许的状态 |
|---|---|
| source NN zero-shot | 合规 source checkpoint、source statistics/BN；不能目标拟合、选温度、更新 BN 或用目标未标注总体统计 |
| full-fine | K>0 才称 target fine-tune；K0 另名 source NN zero-shot |
| target-only scratch、Toeplitz、xDAWN、raw AR8 | 当前均需目标有标签拟合；K0 不可用，不伪造“已拟合”或默默删专家 |
| C3/C4 | 当前 target-only经典专家使其 K0 不可达；保留 requested `not_applicable_zero_calibration`；K0 结果另列 source NN |
| source-only多专家 | 需各自真实 source 模型与匹配 target 观测布局；候选温度/weights 仅 source 内合法 membership 组，当前未实现 |

随机 scratch 输出或 uniform prior 可作明确诊断对照，但不是合法 K0 自我校准或 EEG prediction。固定 quality 特征可以在当次推理输入计算；其学习阈值和 gating 参数不能 K0 用 target 更新。

### 5.2 四库必须先做能力清单

历史统一四库主缓存中，BNCI 是实际使用完整 membership 的决策训练/评分库；另外三库在**该已执行视图**中是 binary-only，不能将此事实推成原始数据永远没有 candidate。

| 库/视图 | 当前已检查能力 | 新 candidate 面板所需退出证据 |
|---|---|---|
| BNCI2014-008 原 MAT | `bnci2014_008_candidate.py` 与 `raw_continuous.py` 有真实 flash/selection/member/truth | 当前统一 C/T 与新 source 排除、requested incomplete、各专家原始支持绑定 |
| BI2014a 原 CSV | `bi2014a_candidate.py` 已恢复 flash code、row/column、target pair、真实 episode boundary；不是天生 binary-only | 61 人规则；连续 selection Adapter、未知相邻响应、raw cutoff/phase 与三专家所需可用支持尚须联动核验 |
| BI2015a 当前 binary source view | 本轮没有确认完整 physical choice/decision/raw continuous chain | inventory 原 MAT/event protocol/阶段/成员 code，验证 original truth；未证实前保留 binary 面板 |
| Huebner2017 当前 binary/BIDS view | 原 BIDS events 有真实 binary 标签与时钟；本轮未证实足够的完整 candidate membership chain | inventory stimulus matrix、selection/target、run/session依赖和连续支持；不以重复句子或事件数量猜 candidate |

四库首先执行同物理试次的 full-fine/scratch/waveform/成熟 ERP binary AUC 面板，候选 hit/NLL 只在合法 membership 子面板执行。AUC 是排序统计量，不能从一个 AUC 标量造每个 decision 的 probability；未接通 raw/candidate 的库不跑虚构 C3/C4。需要升级 Adapter 的库作为明确依赖，不是“云资源就绪即可直接四库融合”。剩余 source 库同样 inventory，source binary labels可以服务 NN 预训练，不能提供并不存在的 candidate calibration labels。

### 5.3 主指标与拒判定义

令 U 是请求参与者，R 是预声明完整流程 seeds，D_u 是该参与者冻结 requested decisions：

\[
H_{u,r}=\frac{\sum_{d\in D_u}\mathbf1(\hat q_{u,r,d}=q^*_{u,d}\land\text{operational})}{|D_u|},\qquad
H=\frac1{|U|}\sum_u\frac1{|R|}\sum_rH_{u,r}.
\]

fit failure、缺失、incomplete、tie、abstain 的分子为 0，分母保留。完整概率覆盖、operational预测覆盖、完整物理覆盖分别记录；complete-only hit 不替代主指标。`hit@all` 消费该 decision 所有获准可用重复；不把 BNCI R10=all 与 BrainSync 45 flash 的九选一混成同一 Q/R 指标。若某库没有 candidate truth，主估计不可计算，不能补一个零或随机 truth 合并全库。

participant-macro 与 decision-micro不同：A参与者9/10、B参与者0/1，宏平均 .45，微平均 .81818。三个 seed 先在同一 participant 内平均，不把 n 名参与者变成 3n；同一被试的 overlapping CV fold/多个 K 前缀同样不独立。三个 complete pipeline seeds 包含 source→target 全链各自训练及选择；冻结 source 后仅换 target seed是另一估计量。

NLL 对实际输出合法概率的 requested decision 为 `−log P(q*)`；无概率项为缺失，不能令 NLL=0。conditional probability NLL按每人的有概率项平均再做宏平均，并同时记录每人 probability coverage；有人的概率全部缺失，全请求宏平均 NLL标为不可计算。两方法比较另在冻结requested roster的paired-complete概率子集计算，保存每人未配对清单；不能混用各自不同valid子集。为了分析拒判，可另列**明确约定** uniform-abstention NLL，将无概率项填 log Q；这不是模型自己给出的 posterior。完整概率但 tie 的 item仍可计算 NLL，operational hit计错。

保存log probability并以 `log P_mix(q)=logsumexp_m(log w_m+log P_m(q))` 稳定计算NLL。数学上真实零质量对应log概率为−∞、NLL为∞，不剪裁为tiny；概率数组有限精度下溢为0而稳定log probability仍有限时，不判为真实零质量。本轮log真值概率−1000/−1100使概率数组为0，稳定NLL1000.69314718与70位Decimal独立oracle一致。历史Phase4复算保持原脚本`max(tiny)`约定，只用于验证旧数字。

不同 Q 以 `NLL−log Q` 同时给出相对均匀的 excess loss。Brier采用 `Σ_q(Pq−onehot_q*)²`，不默默除以 Q；ECE 的 bins/样本权重先声明，小样本时只作诊断。错误高置信度用 `max(P)`、真值概率下尾和固定 confidence cutoff 下错误比例；平均 NLL低不保证错误高置信度尾消失。全拒判会改善 conditional NLL，故风险规则必须同时核对 coverage和uniform约定。

NLL高于 log Q 说明该面板对均匀概率预测平均更差，不能自动禁止 operational argmax；是否拒判属于另一个 C 内预声明策略。argmax accuracy可以高、极少数高置信度错误使 NLL极大。按 test NLL把不利被试删掉或事后调阈值均破坏主比较。

### 5.4 配对统计与晋升草案

首轮 primary contrast 固定 `C4_full−C3`，secondary为 `C4_full−C4_scratch`；singleton和两专家对照用于归因。participant cluster bootstrap 保留该人全部 seed/decision/fold/预算对应，不对 48 CV fold 独立 bootstrap。不把模型固定后的 resampling CI说成训练程序总体不确定性；同时报告每个完整 pipeline seed的 Δ、min/max，只有三seed时训练随机性精度仍有限。

取消历史绝对 hit门槛。本轮正式开发合同仍为同批 Δhit≥0、paired-complete ΔNLL≤0、无新增baseline成功项缺失与无逐人失败增加，具体以主计划JSON为准；这只决定研究继续。若宣布 hit增益，须预声明主要K/权重后 paired 95% CI 下界>0，并报告实际效果规模。实用效应目标草案为 Δhit≥.02；它是待确认的研究SESOI，不是已认证成人门槛。

允许以概率质量改善作为另一个结论，前提是已预声明 hit非劣界 δH（草案 .01）、95% CI下界>−δH、coverage门通过且 paired NLL改善。CI含0不等于 hit相同；精度不足就是未决。若hit下降超过界，拒绝融合晋升，即使 NLL显著改善。source机制 binary面板沿用活动四库等权mean ΔAUC≥.01 的独立研究门，不能以候选 NLL代替。

探索大量AR/先验/空间/gate后选最优会产生winner's curse。精确反例：两个真实正确率均 .5 的方法、各两个独立 dev decision，枚举16种状态后，最高 dev accuracy期望为 .6875，独立test期望仍 .5。原 BNCI八人反复用于改方案属于已观察development；三seed重跑同人不是独立确认。确认需要新独立参与者或先冻结路线后新完整决策；成人BrainSync另立新采集确认。

## 6. 可执行实验臂、依赖与拒绝条件

### 6.1 固定首轮矩阵

专家状态每个 participant/actual K/seed/source fold 独立拟合一次，混合复用该批真实状态和预测；不为每种混合重新训练经典专家。主六臂：

| 臂 | 定义 | 回答的问题 |
|---|---|---|
| E0 | source NN full-fine 单独 | neural在该物理视图的独立能力 |
| E1 | C3：AR8 lambda1 + Toeplitz + xDAWN，各1/3 | 新同批经典/连续锚点 |
| E2 | full NN + Toeplitz，各1/2 | 结构时间cov是否互补 |
| E3 | full NN + xDAWN，各1/2 | 空间ERP方向是否互补 |
| E4 | C4 full：NN + C3各成分，各1/4 | 加neural的主条件贡献 |
| E5 | C4 scratch：同结构scratch + 同批C3，各1/4 | 源迁移在混合中的贡献 |

必须同时保存 AR8、Toeplitz、xDAWN、waveform LDA、scratch NN singleton；这些不是被“已融合”替代的历史分数。CE-only普通full-fine和scratch作为首轮NN，decision loss、source prior/额外包单列后续机制，不堆入主臂。T=1固定首轮；C内温度校准C3/C4、非等权、gating和几何池每个都定义新臂，不把择优后结果覆盖E4。

请求K0/200/800/2000；记录 actual K、positive count、完整selection/target覆盖、阶段、分钟、raw支持、discarded/purged响应尾和独立C group数。K不足回退依据现有预算合同，不与真K2000合并。K0分支按可达性表执行；不可达三/四专家请求显示原因。三个 paired流程seed应在source/task计划冻结，用source seed与target seed一一对应，而非默认3×3 crossed；deterministic LDA复用不制造三次独立拟合。

### 6.2 四周退出证据而非日期承诺

| 顺序/优先级 | 独立交付 | 依赖与继续条件 | 拒绝或拆分条件 |
|---|---|---|---|
| 第1周 P0，B00/B01/B02 | 物理输入履历、共同task join、神经LLR→membership、完整artifact重放、同批C3 | 先补实际入口snapshot和candidate/raw能力；本轮13组数学检查是入口证据 | 无membership/错误source排除/输入合同不兼容/不完整组被归一成成功，停止该臂 |
| 第1周后半 P0，B03/B04 | E0–E5 + singleton合法BNCI开发；失败与rescue/harm表 | NN与source checkpoint有匹配view；内层fit/score交付时钟正确 | 不以0.771为门；救回不超过损害且复杂度增大则保留singleton，继续其他机制 |
| 第2周 P1，B03 | 固定等权对C内温度/收缩weights | 原始完整组数足够；nested OOF可复现；gate输入无target标签 | K200无独立组，拒绝学习型gate；OOF只有5组不拟合高维gate |
| 第2/3周 P1 | AR/边界/先验/空间单独研究臂 | 可并行实现，但各自通过算法及保真门；改变专家state则重新生成OOF概率 | 不复用旧τ/weights给新AR/basis/penalty/空间状态；不能只报告融合总收益 |
| 第4周 P0，B12/B14 | 四库binary与合法candidate面板，三个完整source→target seeds，逐人失败/tail/Pareto | 各库Adapter与预算可达；资源/启动/source freeze/重放先过门 | 没有raw连续链的库不伪造candidate；pilot之后完整已约定矩阵必须继续；成人另立 |

计算资源再多也不能增加独立selection数量、消除先验失配或恢复无记录的membership。吞吐工程应让准备、source fits、target calibration和冻结评分并行，遵守CPU≥25/GPU workers≥4的训练合同；NN内层拟合有新增成本，全量计划须计入nested fitting，不只计算外层model个数。本轮未启动云训练，相关资源技能与启动实测留给实际launch。

### 6.3 如果首轮融合不成立，保留的替代方案

| 可观测结果 | 下一条可归因路线 | 必须保持的对照 |
|---|---|---|
| NN singleton很强，C4未增加hit | 保留NN作为主专家，经典方法用于强锚点和诊断；不因“多专家”愿景强制保留融合 | full-fine、scratch、C3与运算/校准成本 |
| NN弱于Toeplitz，但C4稳定救回经典错误且不增加tail损害 | NN作为互补专家；仅研究收缩到等权的小范围weights | C3、C4 full、C4 scratch及逐人rescue/harm |
| NN只在某些高K有优势 | 预声明按K的固定共享rule；低K维持C3或固定C4，不能逐人由test挑路 | 每K都保留相同requested参与者；报告实际K和source训练成本 |
| NN表征有价值，neural head不强 | 已有features→收缩LDA路线可做独立顺序臂：冻结同一NN特征后拟合LDA，比较原NN head；仅由C内OOF选方 | 原NN、scratch features→LDA、原始waveform/Toeplitz；禁止先用全C拟合LDA再产生伪OOF |
| 所有静态融合都只有NLL收益 | 先研究单专家候选温度而不改hit；或采用明确hit非劣的概率质量结论 | 原argmax、C3固定温度与C3校准温度；不能用NLL替代命中晋升 |
| raw AR8频繁高置信度错误 | 保留raw似然研究和残差诊断，先修可识别噪声/先验而不加门控复杂度；两个成熟ERP专家另立固定混合 | AR8单专家、Toeplitz/xDAWN单独及C3；两个专家不能沿用C3标签 |
| quality解释存在但target完整组很少 | 在合法、来源排他的其他participant完整decision学全局gate，再固定应用；否则等权 | 等权和全局gate，同一source排除；target内学习型gate标不可估 |

source temporal basis＋target native head、固定cosine basis属于另一表征/布局路线，见主计划B09；其与NN特征路线可并行理论开发。只有实际输入、source权限和OOF输出合格，才作为新增专家进入一个**重新声明**的融合矩阵。它们不是上述六主臂的隐藏补丁。需要修改整个NN训练目标或引入decision set network时，重新定义监督量和容量预算；本轮未以小样本端到端gate替代当前小网络。

## 7. 论文定位与访问层级

本轮2026-10-05共10次公开URL访问成功：5个Crossref DOI元数据、2个论文出版社页面、1个作者README、2个公开PDF；没有读取密钥或使用认证Parallel，也没有安装包。公开原响应/访问状态与SHA在 [literature_retrieval.json](literature_retrieval.json)，只读全文页码摘录在 [literature_fulltext.json](literature_fulltext.json)。原始PDF/HTML缓存留本地`literature/`，不强制纳入Git。可分发ledger与全文页码摘录均在本目录根。不是系统综述，也没有对文献性能数字与本仓库做直接换算。

| 来源 | 访问与定位 | 本项目采纳的机制；不越界的结论 |
|---|---|---|
| Sosulski等，2022，[10.1088/1741-2552/ac9c98](https://doi.org/10.1088/1741-2552/ac9c98)，[作者代码](https://github.com/jsosulski/toeplitzlda) | Crossref摘要；本地14页全文，p1摘要、p4结构/收缩；作者README实际抓取 | block-Toeplitz、pooled residual shrinkage、taper与成熟ERP强基线；论文报告213人13协议，但不能据此保证此处C4收益 |
| Rivet等，2009，[10.1109/TBME.2009.2012869](https://doi.org/10.1109/TBME.2009.2012869) | 本轮仅Crossref元数据；不声称核实其全文 | xDAWN原始出处；本项目具体ERP模板增广实现另查源码，不从论文标题证明代码正确 |
| Li等，2020，[10.3390/app10051804](https://doi.org/10.3390/app10051804) | 本地13页全文，p1摘要与方法页摘录 | xDAWN/Riemannian作为可比较转移机制；不把其分类准确率等同本项目hit@all |
| Lawhern等，2018，[EEGNet](https://doi.org/10.1088/1741-2552/aace8c) | 本地30页作者稿全文的compact/P300摘录 | compact spatial/temporal CNN构造依据；不因此宣称N2P3Net优于成熟ERP |
| Guo等，2017，[On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | PMLR页面和10页PDF；p6 temperature scaling定义及验证集NLL | 单正温度不变单模型argmax、需要独立validation；本文进一步证明混合argmax可变，不能套“accuracy不变”结论到C4 |
| Wolpert，1992，[Stacked generalization](https://doi.org/10.1016/S0893-6080(05)80023-1) | 本轮仅Crossref元数据 | stacking文献定位；OOF隔离细节由本规格显式定义，不假装该论文已验证当前gate |
| Saerens等，2002，[10.1162/089976602753284446](https://doi.org/10.1162/089976602753284446) | Crossref摘要及元数据 | class-prior调整依据，需要说明类条件不变假设；不声称能修复任意跨域EEG shift |
| Cawley与Talbot，2010，[selection bias](https://jmlr.org/papers/v11/cawley10a.html) | JMLR页面和29页全文，p1–2选择过拟合/nested CV | 对有限验证准则调参可产生选择偏差；本轮16态精确枚举为独立数学反例 |
| Dietterich，2000，[Ensemble Methods in Machine Learning](https://doi.org/10.1007/3-540-45014-9_1) | 本轮Crossref元数据 | ensemble背景定位；没有把“多样性好”升级为命中单调保证 |

## 8. 本轮验收与未来工作边界

[verify_math.py](verify_math.py) 使用项目venv执行，显式将隔离 `src` 放在 `sys.path[0]` 并逐模块验证 `__file__`。13组确定性验证包括：独立Bernoulli枚举与NN映射、加权CE先验比率、membership梯度、词表置换/不完整组/全缺失/可训练tie、row/column聚合、等与不等曝光offset、混合hit下降与Jensen、概率池/几何池与温度、gate/专家梯度、宏平均/seed/NLL缺失及winner's curse精确枚举，另有温度梯度和稳定log-pool的Decimal oracle。历史预测复算是额外只读验证。

运行命令、退出码和文件哈希见 [validation.json](validation.json)。Ruff检查此脚本通过；95项现有membership、candidate evidence、decision loss和comparison合同测试通过。软件与数学通过说明实现合同和反例可重现，不是新神经/融合模型效能证据。E0–E5、C内校准、gating、四库/三seed和成人矩阵均未执行；FittedEnsemble及个人包接线是未来实施工作，不能写成已完成模型交付。
