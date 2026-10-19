# 完整请求终态、指标数学与近零对比系数反例

本文记录 V05.a 的规格、源码对应关系和软件合成反例。活动进度只在[统一 checkpoint](D:/BCI/n2p3-net/doc/assets/lean_gaussian_20261016/active_checkpoint.json)维护；本文不将候选证明数加入累计验收。原源码反例、隔离候选和后续 live 修复分别保留执行身份。

## 🔬 请求分母与实际消费者

证明目标在编写 Lean 前由[源码要求 01](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/source_requirements_01.zh.md)固定；[要求 02](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/source_requirements_02.json)补齐实际 decision 选择器的第 16 份源码身份。`Requested*.lean` 从原始 outcome 序列过滤出计数和终态集合，先检查 key，再推导分区；没有输入已经平衡的自由计数或预先穷尽的分区。

| 原始对象与源码 | 证明或审查结论 | 必须保留的范围 |
|---|---|---|
| [outcomes.py:280](D:/BCI/n2p3-net/src/transfer/outcomes.py:280) | requested 唯一、规范化等级非空/正数/唯一、observed pair 唯一、集合等于 requested×levels；每级六状态自然计数和等于 requested 数 | core helper 拒绝 missing，不负责自动补齐；raw level 先 `int`，不是 exact-integer parser |
| [outcomes.py:143](D:/BCI/n2p3-net/src/transfer/outcomes.py:143) | correct/incorrect/tie 的 coverage、target、prediction 条件；失败需要 reason 和无 prediction | abstain 可带相等 target/prediction；incomplete 不要求 coverage 不完整 |
| [contracts.py:303](D:/BCI/n2p3-net/src/research/contracts.py:303) | requested participants 是 planned 与 selection-failed 的不交并；planned 人实际有 decision，失败者无 planned decision | selection failure 是参与者终态，不是第七种 DecisionStatus |
| [candidate_within_subject.py:171](D:/BCI/n2p3-net/src/transfer/candidate_within_subject.py:171)、[runner:402](D:/BCI/n2p3-net/experiments/run_candidate_cross_decision.py:402) | 选择器先记录 potential requested tests；校准无效仍可使整个人 selection-failed；最终 frozen plan 用 failure 保留此人，省略其 entries | 不能把中间字典的并集直接认定为最终 D；requested-participant endpoint 仍为 0 |
| [runner:312](D:/BCI/n2p3-net/experiments/run_candidate_cross_decision.py:312)、[runner:346](D:/BCI/n2p3-net/experiments/run_candidate_cross_decision.py:346) | ineligible 与 fit failure factories 显式遍历各自 decision×全部等级，再由原 core accounting 验证完整性 | 只是生成器/调用链审查，没有执行拟合或实验 CLI |
| [analyzer:815](D:/BCI/n2p3-net/experiments/analyze_candidate_cross_decision.py:815) | 验证 task expected counts、冻结 target、各 key 的完整等级；重新构建 accounting；核对原失败账本与摘要 | coverage helper complete 本身未证明外部 task；来源与 frozen plan 匹配是额外义务 |
| [analyzer:850](D:/BCI/n2p3-net/experiments/analyze_candidate_cross_decision.py:850) | fit_failure ledger 要求同 key 所有 matching outcomes 为相同失败；incomplete ledger 只匹配 incomplete prefixes；失败 key 集合必须穷尽 | 浅层可 valid、深层 incomplete；不能把 helper 改读为所有等级必须同状态 |
| [analyzer:1004](D:/BCI/n2p3-net/experiments/analyze_candidate_cross_decision.py:1004) | selection-failed 人 endpoint=0，planned 人除以实际 planned decisions；macro 对完整 requested participants 等权 | pooled decision hit、valid coverage、data-eligible、evaluation-successful 分母分别定义 |
| [BrainSync runner:63](D:/BCI/n2p3-net/experiments/run_brainsync_cross_decision.py:63)、[runner:547](D:/BCI/n2p3-net/experiments/run_brainsync_cross_decision.py:547) | legacy 九选一失败 factory 与 requested-group ledger 也对全部等级补齐并调用 core accounting | 这条 legacy group ledger 不是新的 DecisionPlanContract 协议，不能据共享 helper 宣称协议同构 |
| [baselines.evaluate:247](D:/BCI/n2p3-net/src/baselines/evaluate.py:247) | operational hit 使用原 requested schedule 的全部 group，missing 为 miss；独立 fold AUC 汇总消费实际可用 fold | fold `nanmean` 不是无缺失 requested cohort 的形式化保证 |

逐等级 valid=C+I+Tie；tie 在 valid 分母内，同时在 operational 上非正确。自然域有 0≤C≤V≤N，且 helper 的零扩展满足 C/N=(V/N)(C/V)。analyzer 零分母用 `None`，不与 helper 的 0 合并。

固定三人例给出 participant macro=1/3、剔除选择失败者后=1/2、pooled decisions=1/10。单 key 随等级从 correct 变 incorrect 不违反 accounting；analyzer 的 progression 检查只有 observed counts 和时间不得递减，没有正确率单调性要求。

`EvidenceOutcomeCounts` 的任意直接构造与自然 Counter factory 分开：[八项 stdlib 探针](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/outcomes_source_probe_01.json)实际复现负 correct、平衡 accounted 后仍输出 -1 的直接构造；也复现空 coverage helper complete、[1.1]→[1] 的 raw int 规范化。探针没有把这些弱 helper 当作冻结计划消费者违约。

## 📐 AUC、NLL 和按库等权

[RequestedAUC.lean](D:/BCI/n2p3-net/formal/lean/N2P3/RequestedAUC.lean)把 AUC 写成正负样本对的 1、1/2、0 credit 和，分母为正数×负数；任一类为空则无值。固定 A=(positive1,negative0)、B=(positive101,negative100) 给出 subject-macro=1、pooled=3/4。该证明不精化 sklearn 的 rank 或 IEEE 实现。

[RequestedNLL.lean](D:/BCI/n2p3-net/formal/lean/N2P3/RequestedNLL.lean)对实际有限 log-evidence 与真实 target 证明：

$$
Z=\sum_j e^{e_j}>0,\qquad 0<p_y=e^{e_y}/Z\le1,
\qquad \log Z-e_y=-\log p_y\ge0.
$$

其 `Option` 列表直接过滤实际 `Some` NLL，证明 scored+missing=requested，全部无值时 conditional mean 为 `None`。真实 [continuous runner:110](D:/BCI/n2p3-net/experiments/run_bnci_continuous_study.py:110)在预测为 tie 时仍可保存 NLL，因而 NLL 域不等于 correct 或 unique-prediction 域。其稳定表达式使用 log-evidence；保存的浮点 probability 可下溢为 0，不能把理想实数概率定理认作序列化概率保证。

`requested_nll_with_explicit_uniform_abstention` 的填补项是 missing×logK，并除完整 requested row 数；它与实际 scored NLL 分开。预固定 p=9/10、另一个 missing、K=9 的例子证明条件值 -log(9/10) 严格小于填补均值。源码也明确将其命名为 reporting policy。[run_bnci_continuous_study.py:130](D:/BCI/n2p3-net/experiments/run_bnci_continuous_study.py:130)

[stats.transfer:155](D:/BCI/n2p3-net/src/stats/transfer.py:155)先要求一个 participant 的全部声明 seed pairs complete 才给 participant contrast；[summary:178](D:/BCI/n2p3-net/src/stats/transfer.py:178)在每库内平均 complete participants，再对有值的库等权。`full_study_estimate_available` 表示每个声明库有至少一个 complete participant，**不表示全部 requested participants 完成**；缺少整库时完整 point estimate 关闭，partial observed-studies 另报。数学 fixture 也明确 study-equal 与 pooled-participant mean 的区别。

缺失识别界使用每库完整请求分母，再对全部声明库等权；它不是 bootstrap confidence interval。下面的真实反例保留了源码 guard 允许的近零残差，未把原义务缩成精确零和。

## 🧪 近零系数：false_claim_with_evidence

原源码 [transfer.py:437](D:/BCI/n2p3-net/src/stats/transfer.py:437)用 `np.isclose(sum(coefficients),0)` 接受近零和系数，但 [transfer.py:203](D:/BCI/n2p3-net/src/stats/transfer.py:203)用正系数质量 B 给出缺失项区间 [-B,B]。精确实数下，令 r=Σc，允许 AUC 各在 [0,1]，实际锐界为

$$
L=\sum_{c_i<0}c_i=r-B,\qquad U=\sum_{c_i>0}c_i=B.
$$

只有 r=0 才能把下界替换成 -B。`np.isclose` 允许 r<0；因此原下界对其实际允许输入不普遍成立。

在[预固定探针计划](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/near_zero_contrast_probe_plan_01.json)中使用两组系数、空 observed cells、一个实际 K=200 的 synthetic requested participant、单 source seed/target seed、software_correctness scope。唯一公开入口为实际 `stats.transfer.analyze_transfer_contrasts`；内部经过 `_contrast_report`、`_participant_rows`、`_summary`。

| 固定输入 | 入口实际接受 | 可达 completion AUC | 原报告区间 | 结论 |
|---|---|---|---|---|
| [1,-1.000000001] | 是，`np.isclose` 为 True | [0,1]，contrast=-1.000000001 | [-1,1] | 可达值低于报告下界 |
| [1,-1] 独立 control | 是 | [0,1]，contrast=-1 | [-1,1] | control 恰在下界 |

[探针 03 回执](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/near_zero_contrast_probe_03.json)绑定原源码、脚本与计划 SHA；前两个仅导入格式不同的执行回执也保留，03 是通过 Ruff 的规范脚本。实际 binary residual 约 -1.000000082740371e-9；不将它写成 IEEE 精确的 -10^-9。全部生产源码及声明消费者前后摘要一致。

参数 `iterations=1000` 是公开入口允许的最小值。本例没有 complete participant；profile 证实 `stats.hierarchical`/`paired_interval` 调用为 0，**没有执行 bootstrap resampling**。没有数据集、真实结果、缓存读取，没有拟合、训练、实验 CLI、GPU 或网络调用。此项是软件合成反例，不是研究效能或历史实验结论。

## 🛠️ 隔离的一般界最小修复

生产 [src/stats/transfer.py](D:/BCI/n2p3-net/src/stats/transfer.py)在这些反例与隔离验证期间保持原哈希 `be7b44f24de1c4a8a01ff02e8fa3bc8f7faaffb9c54a9df00109218ab011a371`。可审查的[patch](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/near_zero_repair_candidate_01.patch)只作用于[隔离副本](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/near_zero_repair_candidate_01/transfer.py)，不重定义请求、不删除近零系数接受策略。

候选修复给 `_summary` 增加可选 `contrast_lower_bounds`，由 `_contrast_report` 传入每库负系数之和 L；已有正质量参数继续代表 U。若某库有 S 个已完成 contrast 的和、M 个缺失、N 个 requested participants，则区间改为

$$
[(S+ML)/N,\;(S+MU)/N].
$$

后续现有 declared-study mean 逐端点等权。旧字段 `contrast_bound` 保留为正系数质量，新增 `contrast_lower_bound`、`contrast_upper_bound` 明示一般端点；`assumption` 文本改为一般界。`_summary` 未传新参数的直接调用沿用对称 fallback，真实 public contrast 路径总传 L。因此“保持 exact-zero 输出”指已有区间数值不变；JSON 增加字段且说明文字更新，不能声称整个报告字节不变。

[隔离修复计划](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/near_zero_repair_probe_plan_01.json)预先固定相同两个 public fixtures，以及两库各一个 missing 的 macro fixture；[修复回执](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/near_zero_repair_probe_01.json)实际比较原入口与隔离入口：

| 对照 | 原数值区间 | 候选数值区间 |
|---|---|---|
| negative residual | [-1,1]，排除可达 completion | [-1.000000001,1]，包含相同 completion |
| exact-zero control | [-1,1] | [-1,1] |
| 两库 A近零、B系数[2,-2] | [-1.5,1.5] | [-1.5000000005,1.5] |

候选与探针均通过 Ruff；生产 source pins 前后保持，paired_interval/拟合仍为 0。一般实数数学证明覆盖 signed coefficient 范围、按实际 `Some`/`None` 填补的 requested 分母、完整 completion 的上下界，以及库等权保持每库界。它不包含 Python 求和、乘法、除法的 directed rounding；候选通过有限 IEEE 探针不等于任意规模浮点计算的保守外包络证明。该数值 refinement 仍单列。

## 🔗 后续 live 修复与 v4 身份

在上述原身份证据封存、并发检查全部终态之后，root 将独立审查通过的候选应用到 live `src/stats/transfer.py`。新源码 SHA256 为 `062e9471db589932587f1d155b54d6344ed1d6ba47eb23ecfb092d7562f2632a`；[v4 激活回执](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/source_v4_activation_01.json)记录更换前物理归档和同步屏障。[science_contract_v4.json](D:/BCI/n2p3-net/doc/assets/lean_personal_erp_20261018/science_contract_v4.json)的 SHA256 为 `454cc391bc6ffea1459f94a8567462dde72115792ec24f2d7c6721c85da6fc5c`，声明 201 份 source、190 份 runtime 身份；它没有覆盖或重写旧 v3、原 85/16 执行回执及原反例。

[正式 RED](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/near_zero_regression_red_01.json)绑定原 live：负残差、正残差和两库等权三项回归失败，exact-zero control 通过。[正式 GREEN](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/near_zero_regression_green_01.json)绑定修复后 v4：`test_transfer_statistics.py` 与 `test_hierarchical_stats.py` 两个完整 suite 共 43 项通过。这些是实际公共分析入口的软件合成验证；没有重新计算科研结果，也没有训练。隔离探针的 0 bootstrap 结论只属于其固定空 complete-participant fixtures，不扩写成这两个完整 suite 从未执行重采样。

修复的独立审查、正式回归及交付范围另由 root 的[修复交付回执](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/near_zero_fix_delivery_01.json)统一绑定；原失败版本不因 live 修复而改写为通过。

新 `_contrast_report` 把每库 `sum(c<0)` 传给 `_summary(...,contrast_lower_bounds=...)`，同时保留 `sum(c>0)` 的上界。原 `RequestedStudyStatistics` 中 `auc_contrast_general_bounds_preserve_coefficient_residual` 与 `signed_coefficient_bounds_are_the_source_negative_and_positive_sums` 给出此二端点的实数数学关系；`asymmetric_requested_bounds_contain_every_allowed_completion` 覆盖完整 requested 分母，`declared_study_equal_average_preserves_each_study_bound` 覆盖声明库等权组合。`exact_zero_sum_auc_contrast_is_bounded_by_positive_mass` 只保留为精确零和的特例。原 85 条数学源码及其原哈希不变；v4 对应关系通过独立 delta 附注绑定，不因版本迁移重跑整批成功证明。

[v4 delta 附注](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/v4_source_mapping_delta_01.json)核对 201 份 v4 source、原本拥有的 16 份源码中仅 stats 的变化，以及原 13 个证明/审计/负控文件完全不变。它是新增源码对应证据，不更改旧检查的 production identity。

## ⚖️ 补齐一般组合权重

原 85 条已有 `studyEqual` 的定义、缺库门槛和非均匀反例，但未一般推出完整 seed grid 过滤之后的逐行有效系数。这是原 V05.a 的数学/源码关系缺口，不能移到 V05.b 或数值预算之外。[权重源前合同](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/aggregation_weights_requirements_01.json)先固定实际入口、对象与错误替代，再实现 [RequestedAggregationWeights.lean](D:/BCI/n2p3-net/formal/lean/N2P3/RequestedAggregationWeights.lean)。

令显式声明的库集合为 S，解析后**实际声明的整数 seed pair 集合**为 R，每库原请求参与者集合为 P_s。`rows(s,p,r)` 是 `_contrast_report` 生成的 unit row 的数学投影：实际 `complete_pair` 给 Some δ，其余给 None。模块从 primary lookup 计算 missingPairs，再过滤完整参与者 C_s；没有输入一个已经完成的 C_s。它推导 selected 人的 observed pair 集合恰为 R，因此分母使用完整显式声明的 |R|，而非 surviving pairs 数。显式 diagonal pairs={(0,0),(1,1)} 有两个元素，不能擅自改成四元素 Cartesian product。

模块将源码的三层 mean 实现为固定实际支持上的算子 T；将 T 作用于某条原始 unit row 的指示函数，定义其有效系数。最终公式是由这个算子推出的结论：

$$
w_{s,p,r}=
\begin{cases}
\dfrac{1}{|S|\,|C_s|\,|R|}, &s\in S,\ p\in C_s,\ r\in R,\\
0,&\text{其他。}
\end{cases}
$$

实际 full-study gate 推出每个声明库 C_s 非空；在声明 S、R 非空的真实入口域内，再证明 w≥0、每库质量为 1/|S|、对**全部原请求行**的质量为 1，以及返回的 full point 等于 Σwδ。一个声明 pair 缺失会使该人所有 unit-row 系数为 0。整库没有完整人时，full estimate 返回 None；源码另报的 partial observed-studies mean 可以归一化剩余库，但不能冒充 full result。

`_contrast_report` 的 unit δ 本身是 Σ_arm c×AUC。[arm 映射合同](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/aggregation_arm_mapping_requirements_01.json)区分这两层：分配给原 AUC cell 的有符号系数为 w×c，其总质量是库等权的 Σc 残差，而非自动等于 1 或 0。对应一般有限和分配定理保留近零残差；精确零和只是一个特例。

这里的 conditional participant contrast 除以 |C_s|，requested missing bounds 除以 |P_s|。同一个实际 filter 另外推出 |C_s|+|P_s\setminus C_s|=|P_s|；它与已证明的一般非对称 completion bounds 对接。未知 actual K 的请求保留在 outer ledger，不虚构归属到某个实际 K panel。这一条件口径与 requested-participant operational hit 的全 P 分母分开。

NLL 还需另一条组合，不能套用“缺一个 seed 就排除整个人”的规则。[NLL 宏均值合同](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/nll_macro_weight_requirements_01.json)及 [RequestedNLLMacroWeights.lean](D:/BCI/n2p3-net/formal/lean/N2P3/RequestedNLLMacroWeights.lean)从每个实际 cell 的 Some NLL 过滤出 Q_p，再按 `_run_study` 的实际字段 gate 与 Q_p 非空筛出 A。这里的源前合同误把局部变量 `completed` 写成 status=='completed'；[独立勘误](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/nll_macro_gate_erratum_01.json)保留旧合同/执行身份，并将通用 Bool 参数正确绑定到 `"requested_hit_all" in cell`。该变量及 `scored_subjects` 计数不证明拟合成功，也不等于有 NLL 的 cell 数。通用数学对任意输入 Bool 都成立，故不修改或重跑已冻结证明。其系数仍由 mean 算子作用于 row indicator 推导：

$$
v_{p,d}=\frac{1}{|A|\,|Q_p|}\quad(p\in A,d\in Q_p),
$$

其他原请求行系数为 0。两个人的 |Q_p| 可以不同；一个人缺部分 decision NLL 仍可保留。模块推导非负、原请求行上总质量 1 和返回值展开；A 为空则报告 None。固定 NLL 例为 A 的 Some1/None、B 的三个 Some3：macro=2，pooled decision mean=5/2；实际 A 行系数 1/2、B 行 1/6、missing 行 0。它不将 uniform-filled requested NLL 重新命名成模型 NLL。

这些组合证明只处理实际 source 语义的有限实数关系。重复声明、canonical identity、source arm validity、actual-K/producer 匹配的入口检查分别有源码审查；跨 runner/模型/产物的整体身份组合仍属原 V05.b 的依赖项。不能据此把 V05.a 内尚缺的筛选、均值顺序或分母算法关系推迟到 V05.b。

## 🧩 独立复核补出的 actual-K policy 义务

首次独立复核确认，前三层 transfer 证明只对应一个 actual-K panel，当时尚未覆盖 [transfer.py:317](D:/BCI/n2p3-net/src/stats/transfer.py:317) 的 `budget_policy`：它先要求某人实际 K 集合内每个 participant row 全部 complete，再对该人不同实际 K 取均值；这个集合可以因人而异。其 primary unit-row 权重还需一个 1/|K_{s,p}| 因子，不能把 standard budget 集合当作每个人的实际 K 集合，也不能丢掉失败 K 后继续算这个人的 policy point。

[闭合提案修订 02](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/v05a_closure_proposal_amendment_02.json)保留原提案，并明确 85/16 与新增 48/9 可以作为有限切片审查，而不能据此宣告原 V05.a 完成。[policy 源前合同](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/budget_policy_weight_requirements_01.json)固定实际 seed→K→participant→study 四层计算、whole-person 失败 gate、unknown actual K 的 outer ledger 边界，以及错误替代。这个缺口仍属 V05.a 的算法数学，未转交 V05.b 或 IEEE 预算。

该追加切片由 [RequestedBudgetPolicyWeights.lean](D:/BCI/n2p3-net/formal/lean/N2P3/RequestedBudgetPolicyWeights.lean)补齐：从原 lookup 计算实际 K grid 和 whole-person policy gate，将四层算子作用于原行指示函数，再推出系数 1/(|S|·|C_s^policy|·|K_sp|·|R|)。模块证明每人、每库和全部 policy 行的质量与返回值展开；失败 K 排除整个人，unknown K 不构造 policy entry；common-standard-K cohort 的实际 K presence/完整 seed filter、后置 predicate、signed arm residual 和 contrast 凸界也分别给出一般关系。

[policy 作者回执 02](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/budget_policy_author_checks_02.json)绑定 32 条一般关系、5 条固定见证及 5 条严格 False 负控，原 v5 的 202 source/191 runtime 与当次 protected251 前后保持。总聚合、两个独立文件重查、公理审计和负控共 5 个检查均通过。[01 回执](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/budget_policy_author_checks_01.json)保留；02 只纠正收集脚本继承的 v4 注释为实际 v5，未重跑成功检查。[物理清单](D:/BCI/n2p3-net/frozen/lean_outcomes_20261018/budget_policy_manifest_candidate_01.json)冻结 5 个文件；[清理回执](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/budget_policy_freeze_cleanup_01.json)记录 6 个中间 build 目录、12 个文件归档核验后由 cmd 删除，最终 5 组活动日志保留。

[源码权重实例表](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/metric_weight_instantiations_02.json)明确每个实际 reducer 的输入 key、gate、value 与算子实例。baseline 的 hit_at_repetition_macro 实际为 pooled Σcorrect/Σtotal；未显式收到 requested_schedule 时只从当前 supplied groups 推断域，不能凭名称宣称冻结请求完整。

## 🎲 固定重采样的确定性数学

原 V05 第三目标还包括重采样实现与科学推广假设的区分。固定索引后的 mean 可使用前述通用算子，但此前 170/30 组件未专门证明两个 batch schedule 的覆盖/唯一更新，以及 default linear quantile 的排序/插值归约。[修订 04](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/v05a_closure_proposal_amendment_04.json)保留这个确定性 V05.a 义务；它没有把问题挪到 V05.b、随机数生成器或 CI 覆盖率。

[fixed-resample 源前合同](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/fixed_resample_requirements_01.json)从实际 paired_interval cap4000、summary 每库 stride1000、均值与分位数调用固定要求。[只读源码快照](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/fixed_resample_source_snapshot_01.json)核对本机 NumPy2.5.2 默认 linear 方法 h=(n−1)q 与 gamma 原值，不导入或执行项目/NumPy模块。

draw mean 以 draw 的位置为 key，原 unit 可重复抽中；有效原 unit 权重由均值对其指示函数的响应推出 multiplicity/n。fixed-study draw 对每库按自身 n_s 归约，再给 estimable studies 相同权重，不能合成 pooled-participant draw。每库 additive loop 对同一输出位置各更新一次，整体有若干次加法，不能误称全局只有一次写入。

linear quantile 使用保留全部位置与 ties 的排序置换接口，再按 floor(h)、相邻上界及 gamma 插值。要求从这个精确接口推出端点、顺序和凸界，而不预先输入 quantile 保证。不证明 RNG 分布、随机流批次等价、NumPy/IEEE 实现或 bootstrap 置信覆盖；也不根据 participant/replicate 字段标签推断科学独立性。

[本批作者回执](../assets/lean_outcomes_20261018/fixed_resample_author_checks_01.json)记录 47 条一般关系和 6 条合同预先固定的数学见证，以及 6 项严格归约到 `False` 的错误替代。新聚合 [RequestedResampleMetrics](D:/BCI/n2p3-net/formal/lean/N2P3/RequestedResampleMetrics.lean)、五个新模块的独立进程重查、公理审计和负控共 8 个检查通过，均绑定实际 v5 的 202 source/191 runtime、当次 protected293 与完整本地 import closure；原 170 条声明及其成功执行身份未改写、未因新基线重跑。47 条一般关系分为固定索引 mean 7 条、批次覆盖 11 条、linear quantile 15 条及真实参与者/区间接口 14 条。

[源码对应表](../assets/lean_outcomes_20261018/fixed_resample_source_correspondence_01.json)逐项绑定 paired 的位置均值、两个真实 batch width、study-equal additive draw、置信查询和 full/partial gate。[FixedResampleInterface](D:/BCI/n2p3-net/formal/lean/N2P3/FixedResampleInterface.lean)通过实际 complete participant filter 的双射枚举读取每个 `Some` 值，再把最终 draw 和排序前 raw 向量相连；不任意另给一个不相关的 quantile 输入。R01 保留重复抽样，R02/R03 的尾批次分别为 1，R04 区分 equal-study 的 1/2 与 pooled 的 1/4；R05/R06 用四点 quantile primitive 区分正确端点 3/20、237/20 与错误 raw-order 插值 111/10。这个四点例不是非法的 `paired_interval(iterations=4)` 执行。

[本批物理清单](../../frozen/lean_outcomes_20261018/fixed_resample_manifest_candidate_01.json)逐成员冻结 8 个证明/聚合/审计/负控文件；[清理回执](../assets/lean_outcomes_20261018/fixed_resample_freeze_cleanup_01.json)记录将 11 个中间构建目录中的 22 份 log/report 归档校验后，用 cmd 按明确路径逐目录删除并确认消失，8 组最终检查及全部源码/driver/extra-pin 快照保留。没有安装临时包或修改生产代码。本批补齐候选证据，原 V05.a 的整项闭合仍由独立审查和活动 checkpoint 的验收决定。

## 📋 证明和验收证据的解释

[RequestedSlice.lean](D:/BCI/n2p3-net/formal/lean/N2P3/RequestedSlice.lean)组合上述规格；[检查计划](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/requested_lean_check_plan_01.json)固定全部候选证明、逐文件独立进程检查、标准公理审计和严格 False 负控。旧 protected211 身份在检查前后另核对；未把历史成功检查改写到新基线。候选通过的事实与最终独立源码验收分别记账，全部 V05.a 的原生产义务仍必须显式处理实际近零分支缺陷。

本批[检查回执](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/requested_lean_checks_01.json)包含 85 条定理、16 条明确归约到 `False` 的负控，以及 build、11 个逐文件独立检查、公理审计和负控共 14 个终态进程；没有 Lean warning 或额外公理。最初聚合器未剥离公理列表中的换行，随后只修正日志解析并收集原 report，没有为此重跑成功 Lean，也没有把迁移后的 live source 身份倒填到旧检查。

候选 13 个证明/审计/负控文件已在[物理快照清单](D:/BCI/n2p3-net/frozen/lean_outcomes_20261018/manifest_candidate_01.json)逐成员核验。[清理回执](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/freeze_and_cleanup_01.json)记录当前构建确认后，将 24 个已被替代的构建目录中 48 个 log/report 归档校验，再经 cmd 逐目录删除并验证消失；活动 pinned 证据、`.lake`、依赖包和科研数据均保留。本批没有安装临时包。

后续一般权重批的[作者回执](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/weight_composition_author_checks_01.json)包含 38 条一般关系、10 条固定见证及 9 条严格 False 负控。总聚合 `RequestedCompleteMetrics`、四个新文件的独立进程重查、48 条公理审计和两组负控均通过；全过程用 `check_current.py` 绑定 v4、201 份 source、190 份 runtime、当次 protected215 和完整本地 import closure。只检查新证明与新聚合，没有因为 v4 迁移重跑原 85 条成功证明。

新 8 个证明/聚合/审计/负控文件已在[新批物理清单](D:/BCI/n2p3-net/frozen/lean_outcomes_20261018/weight_composition_manifest_candidate_01.json)逐成员核验。[本批清理回执](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/weight_composition_freeze_cleanup_01.json)记录将 12 个被替代的中间 build 目录、24 份 log/report 归档校验后用 cmd 删除；最终作者回执的 8 组活动检查日志和全部 current-check 源码/driver/extra-pin 快照保留原处。独立审查可据[原始 V05.a 闭合提案](D:/BCI/n2p3-net/doc/assets/lean_outcomes_20261018/v05a_closure_proposal_01.json)逐项检查 M01–M10；提案不自行更新累计验收或 checkpoint。

当前数学不证明：sklearn rank 实现、SciPy logsumexp、NumPy Counter/索引/求和或 Python构造器的完全精化；也不证明 bootstrap 覆盖率、新库推广或成人九选一效能。已有 Counter/coverage/constructor 的有限源码探针，以及这里的实际统计反例，分别绑定其精确执行身份。

## 请求与 policy 组件验收

[组件验收](../assets/lean_outcomes_20261018/requested_policy_component_acceptance_01.json)
纳入原终态/指标 85/16、一般 seed/person/study 与 NLL 48/9，以及实际 K policy 37/5，
合计 170 条定理、30 项严格错误替代。四层系数由实际筛选支持上的指示函数响应推出
`1/(|S|*|C_policy,s|*|K_s,p|*|R|)`，不把目标系数或质量一作为前提。
whole-person gate 拒绝任何实际 K 失败的人；标准 K 共同 cohort 则是另一项实际
presence/full-seed 交集，单个 per-K panel 不多除一个 K 数。

[policy 独立复核](../assets/lean_outcomes_20261018/independent_v05a_policy_review_01.json)
确认这项组合与 NLL 字段勘误已补齐；其最初整项闭合建议由
[重采样范围补审](../assets/lean_outcomes_20261018/independent_v05a_resampling_scope_review_01.json)
追加纠正，组件结论保留。该阶段原 V05.a 还缺固定 resample 索引下的重复槽计权、4000/1000
两种分批完整覆盖及最终 draw 向量的线性分位数接口。源码审查未发现这些循环的计算
错误；该数学缺口不转交科研侧修复，也不混同 RNG/native/IEEE 或 CI 覆盖率认证。

27 项作者检查和 root 一次联合构建分别保留实际执行身份。联合构建绑定 v5 的
202 份源码、191 runtime 和 251 个验收前保护成员，墙钟 32.54 秒。
[证明清单](../../frozen/lean_requested_policy_20261018/manifest.json)与
[源码证据清单](../../frozen/lean_requested_policy_20261018/source_and_evidence_manifest.json)
封存本批；旧 85 检查保留原 collector/guard 证据层级，不伪造现代逐次快照。
原始、一般权重和 policy 的 cmd 清理回执均经 root 核对归档及目录消失；后两批共清理
36 份旧 log/report、18 个空目录，最终检查与活动依赖保留。

## 固定重采样与原 V05.a 范围验收

[最终独立复核](../assets/lean_outcomes_20261018/independent_v05a_fixed_resample_review_01.json)
及[原范围验收](../assets/lean_outcomes_20261018/fixed_resample_acceptance_01.json)纳入
53 条定理、6 项严格错误替代，补齐上述 B03。重复抽中的人保留为不同 draw 槽，
系数由槽 multiplicity 除以原 n 推出；各库按自己的 n_s 重抽，再按库等权。
4000 while/min 与 1000 range/min 两种批次覆盖均保证每个结果位置恰有一次对应贡献。
最终分位数作用于完整 draw 向量，由排序置换、floor 与线性插值推出顺序、端点和凸界，
不会把 per-batch interval 的均值当作完整向量的分位数。

source 的 partial-observed-J interval 与 full-declared-S 输出 gate 分开。NumPy 2.5.2
默认 linear 的只读源码摘要也固定了实际方法接口，没有据此宣称原生排序、RNG 或
浮点实现已认证。V05.a 的原数学规格和源码对应范围据此完成，共 223 条定理、36 项
严格错误替代；V05.b 的物理配对/产物/producer 链和统计置信覆盖率分别保留。

八次作者检查与 root 一次联合构建绑定 v5、202 份源码、191 runtime 和当时 293 个
保护成员。root 联合构建墙钟 32.99 秒；未重跑先前 170 条成功证明。
[证明归档](../../frozen/lean_fixed_resample_20261018/manifest.json)及
[源码证据归档](../../frozen/lean_fixed_resample_20261018/source_and_evidence_manifest.json)
固定本片；[清理回执](../assets/lean_outcomes_20261018/fixed_resample_freeze_cleanup_01.json)
记录 22 份旧日志、11 个空目录归档核验后经 cmd 删除，root 再核对实际消失。
未安装包、运行 bootstrap/拟合/训练或修改生产算法。
