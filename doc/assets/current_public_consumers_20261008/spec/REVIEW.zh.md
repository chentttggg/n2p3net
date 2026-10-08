# 当前 public consumer 科研 Spec 评审

本评审接续已完成的 source `fd62c48b` 评审，该旧 report/capsule 不改签。本片从 main/作者共同基线 `a2975557d6e5e3ae99ff2f8d692ecc8a7e524ab2` 开始，只读作者 `codex/current-public-consumers`，作者单独修改生产；本目录保存独立数学反例、读取身份和验收结论。**当前 public consumer 软件 Spec 轴通过，零未闭合阻塞发现**，绑定 accepted source `e6499b90778644e0df3ffd899a39aaf004ab792b1302f29cf2105b02570955ff` / 439 成员。真实 source 到 K0/selection-OOF 的 P1 已通过原包 RED→accepted GREEN 关闭。公开效能、完整矩阵和成人确认仍未由本片证明。

## 当前 P1：真实 source 的 selection 路径被门禁拒绝

独立 Standards 发现后本评审实读确认：真实 independent source 必有完整 `input_preprocessing_execution`，其 `execution_identity` 非 None；`PreparedTransferCheckpoint` 在 checkpoint.py 539 要求 target `preprocessing_execution_identity(dataset)` 与它完全相同。K0 `CalibrationPrefix.trial_target` 仅有 reference 元数据，OOF `build_calibration_epoch_dataset` 有真实 `selection_epoch_execution/profile`，但没有接到该标准 owner，所以真实 source 的 K0 和 T/S OOF 都会拒绝。软件 checkpoint helper没有实际 source execidentity，因此此前测试没有触发这条真实路径。本评审此前把无 source critical 判断得过早，现明确撤回并由本 P1 hold。

独立源码推导：本次 source finite-zero/notch/`mne_reflect_limited` 实际从 `design_continuous_sos` 与 MNE padlimit进入 `filter_bidirectional_segment`；SelectionEpochProfile 使用同 SOS/padlimit/finitefilter。两者 closed-source-grid 后都调用 `transform_epoch_samples` 做 FFT/first-T/baseline，再 float32/Cz projection。同 canonical mathematical family可证明，但 full-record source 与授权有限 decision 的长度、端点/可用时间不同，不能称同一有限 `F_N` 或伪造 `mne_raw` 完成凭证。

修复应由 checkpoint/共享 input owner显式验证 source 原完整执行凭证与目标实际 selection profile/operator/executions，核 declared profile、实际 SOS/padding、closed grid/FFT/firstT/参考/输出/receipt关系；返回“不同 ingress/支持、同已证程序族”的 compatibility relation。K0只能提供未来推理 transform 的真实能力/profile描述，不能伪造已执行目标训练 receipt，实际 score另消费真实 execution。错误 profile/SOS/grid、无证据和非授权 raw仍 fail-closed；旧 source `fd62` 不重签、不 None fallback、不把 gate简单放宽。待用旧 tiny实际source checkpoint的零fit反例与 K0/OOF consumer关系确认出口。

当前修复实际已读：`validate_selection_checkpoint_execution` 验源完整原凭证及每原 native fs 下 SOS/padlimit；目标重编译 profile/grid/operator，限定真 float32/linear baseline 能力，actual channels采用同canonical电极名。K0只有futurecapability，不宣称校准执行；OOF核actualarray/mask numericaldigest、原physicalsupport/rawinput digest、outputreference、exact spans=`onsets+profileoffset+absolutescope`、epochgeometry、实际padding=`min(N−1,limit)`及行覆盖。持久test由真实execute_preprocessing产生非Noneexec再进softwarecheckpoint，检查K0/OOF/coldload与spans+1/缺cap拒；这组是softwaregate，独立原source使用另有下面的真实旧producer零fit探针。

原 source95c933真实 BDF β1 checkpoint→原A01 的 zero-target-fit probe 已实际成功：K0使用同capability对actualheld transform、sourceposterior保存逐数重放；C800前2个complete groups建立真实240行 OOF epoch data并经过 PreparedTransferCheckpoint，NPZ普通attested save/load gate成功；错numericalreceipt明确拒绝。修前 old `fd62c48b` 的实际执行 checkout 对同 checkpoint、同原A01 K0拒绝 `checkpoint requires compatible preprocessing execution evidence`，exit1且targetfits0，原RED receipt保留；隔离 accepted `e6499b...` checkout经 verified_exec 实际执行 GREEN，并额外拒绝 wrong_trained_channel_profile。checkpoint和其 `source_preprocessing_execution` 不改字节/不改producer95c933来源。本评审stdlib从原31文件preservation核checkpointSHA、targetactualprofile/ref/grid/outputreceipt、K0保存file和OOF cache/recordSHA、RED/GREEN同源及accepted439members；此probe NN=0，不能称C800目标训练或OOF calibration已执行。P1已关闭。

后续冻结 K0 predictor还持有 selection capability，并对训练合同实际 channels/reference/epoch geometry核验，不能经wrapper把另一个profile签成source输入。actual source receipt 的数学族核验按原nativefs缓存SOS编译，概率变换在载入时完成一次实际calibration验证/logβ/logπ缓存；推理不在每个batch重新hash完整source contract。优化没有更改score公式或授权scope。

## 范围与主张

本片包括实际 target β/π/固定 C class mass、冻结 raw/LLR/posterior 语义、binary NLL/Brier 与固定概率池、canonical Cz6 raw C/OOF/held 与 native8 single anchor、只对五个主 producer 的 C-only OOF 温度、共享 all-C final state、请求/失败分母。完整 source 十二模型、公共面板效能、成人确认不由本软件切片证明。旧矩阵不作必须兼容/必跑的要求；合法旧 package 仍按原语义重放。

当前 target NN/S 都 fixed32 epoch、batch64、lr1e-3、wd1e-4，输入统计/BN/dropout/更新范围相同，仅初始化不同；没有单侧早停。source 仍原 `fd62` β1 程序，不改变准备输入或重签原模型。

## 独立数学问题与修订

原 `SubjectAdapter` 用 `nn.CrossEntropyLoss(weight=[1,β_C])` 默认每 batch 类别质量分母。随机小 batch 的期望风险并不等于固定全 C 加权 CE；按单个样本类条件化，其有效类别比为 `β·E[1/(S+β)]/E[1/(S+1)]`，并非 β。独立 Fraction 无放回有限总体反例 N10/npos1/batch2/β9，实际期望风险有效 β=`81/41`，零信息最优 raw probability=`9/50`=.18，而 full-C balanced 风险为 .5。N200/npos20/batch64 的有效 β约8.56778，解析校正差约−.04922 logit。见 `target_probability_counterexample.py/json`，项目 venv exit0，不导入 torch、不 fit 模型。

root 已明确授权新 `calibration_balanced_fixed_mass`：`β_C=n_neg/n_pos`，`π_C=n_pos/n_C`，`Z_C=(n_neg+β_C·n_pos)/n_C`，每 minibatch `Σ w_i CE_i/(batch_len·Z_C)`，从各 fit 的实际合法 C labels 计算并冻结。全 C 风险精确等于 `.5 mean_negative_CE + .5 mean_positive_CE`。固定 model state 下，uniform 无放回 batch 的期望风险/梯度等于全 C；脚本独立 analytic gradient 全 C `−1/8`、所有 batch2 平均 `−1/8`。reshuffling 前批已影响模型时，conditional remaining population 不再是全 C 的均匀样本，同脚本 remaining gradient=`−1/3`，所以不声称每个 adaptive 条件 step 满 C 无偏，也不把有限 BN/dropout/regularized 神经网络当 Bayes optimum。

在明示 fixed-risk 理想 Bayes 下，`z_raw=LLR+lnβ_C+logitπ_C`，target balanced C 下后两项代数抵消；binary posterior logit=`z_raw−lnβ_C`，candidate LLR=`z_raw−lnβ_C−logitπ_C`。它们是声明预测分布/解析校正，不证明新 test posterior 已校准。source β1 输出的 posterior 是其冻结 source mixture；LLR 必须恢复实际 source prior 才能作为 binary posterior。不能从 model family 猜校正项，也不能以有界 sigmoid/权重归一作校准证据。

## 空间、混合与温度反例

`spatial_pool_counterexamples.py/json` 使用 Fraction 独立核 BNCI native channels `Fz,Cz,Pz,Oz,P3,P4,PO7,PO8` 到六 slot 的 `A6×8`，每行目标电极减实测 Cz，精确 rank6。共同参考常数偏移不影响输出；native PO7 单独信息在 six 投影下丢失，所以 native8 anchor 与 Cz6 raw 比较是观测信息/参考变化。A 满行秩不保证实际 projected residual Σ SPD（共模数据的投影可以全零）；真实 numerical/rank 仍由运行准入验证。

每个原生 label-free held view 仍是一条物理 observation authority；raw 的 A 派生数值 digest 与物理 support digest 分开，避免 Rcanonical6 与 trialnative8 profile 的 physical identity 脱节。C raw、OOF raw 和 held raw 消费相同 label-free owner；native8 raw single 保留原 bytes/ref/A/support。Cz6 raw256 unfiltered 与 trial128/179 finite-zero 仍是时间/算子路线差异，不能称相同 numerical input。

主 candidate flat weights：T/R 各1/3、Toe/XD各1/6；S 只作为替换 T 的对应控制，不混入 primary。binary 主权 T1/2、Toe/XD各1/4，S 对应替换。所有固定 probability pool 严格要求全声明 producer，失败不得使用成功 subset 重归权。反例权重和为1、丢 R 后质量2/3证明改权会改变程序。classical-three 等数值专家的 secondary 不冒称主机制类等权。

fixed 是 primary，grouped temperature 是 C-only OOF secondary。只五 primary expert IDs 校温，wave/nativeanchor 固定 T1 保留原数值/身份；未校准 controls 不标 OOF calibrated。任一声明 primary 的 OOF 失败仍使该 grouped strategy 不可用，不能暗缩温度 roster；fixed 保留。fixed/grouped 共用一份真实 all-C final expert fit，只 score transform 不同。

## 调用链覆盖与当前状态

| 要求 | 实际入口/owner/consumer | 当前状态 |
|---|---|---|
| actual β/π/Z 与 reduction | TargetRecipe → SubjectAdapterConfig → class_calibration → fit/refit | 源码完整关系及必要独立CE/梯度/均匀subsets/尾batch断言通过收据已绑定 |
| raw/LLR/posterior 冻结 | FittedCalibrationCandidate actual probability_fit → FrozenTrialPredictor `/2` → caller/reload | counts/actualprior/reduction、numericstate和sourcecontract绑定已核；旧 `/1` 原score，oldbatchmass不新增posteriorclaim |
| proper binary metrics/pools | comparator immutable source/catalog + same C/test + saved replay | 新caller读取raw/LLR/posterior一次并保存重放；pool复用ContrastPlan完整identity、singlebytesSHA、requested aliases，未按成功子集改权 |
| canonical raw / native8 anchor | profile/spatial owner → full-C、OOF groups、held score → ensemble package | 完整源码已核；原A01单C200空间fit/held重放收据及原bytes/archive关系已独立检查 |
| 温度只5primary | `_grouped_temperature` → shared final programs → save/load → score_experts | subset OOF只拟合declared primary，未校控制T1原身份；caller拟合前声明控制仅fixed，all-C最终状态共享 |
| 全请求与失败 | comparator requests / candidate DecisionPlan / fixed roster pool | 原strict requested owner保留；newpoolunavailable专家、requested_Ks与source/target配对索引保留，sameactualK别名不产生新fit |

## 需要继续核的实际关系

新增 `class_calibration` 不能只有多个 metadata mirror 与 numeric-state digest自一致。target calibration 必须与 actual fitted label counts/prior/reduction/digest及保存的 recipe 关系一致；generative LDA/ERP 的 prior 从 actual model / legal fitted rows出发。source_mixture 的 stored training contract 和实际 β关系核验，原 source prior 保留实际 producer lineage，不能声称可以只从权重恢复。

这些关系已经读到当前修复：target risk 与 probability_fit外层class_counts/prior及recipe.reduction交叉校验，generativeprior与实际fitclasses匹配，source校正β与保存的trainingcontract/digest一致，numericstate另绑digest。`probability_scores`明确拒绝没有actualcontract的legacy和旧batchmass target；候选score仍按各包合法原语义/新声明LLR解释。classic新predictor已去prior后 ensemble offset设0，避免重复扣除；旧 `/1` state与storedoffset继续旧程序，不改写原分数。

旧 `/1` package 载入保持其旧 logits/offset，没有 actual probability contract 的旧包不因新默认升级。`TargetRecipe.record` 的新 reduction 在旧 default 下应保留旧 identity，而明确新 reduction 则参与 fit cache/recipe/package身份。OOF class risk 必须只各 fit C，held 标签不能参与 β/π/Z；相同声明 policy 不意味着共享实际 prior。

验证策略：只读稳定源码/JSON、独立标准库数学反例、消费作者的 red-capable 最小必要行为/梯度/真实输入探针及稳定 capsule。没有运行 broad suite、启动 cloud source 或干扰本地 B08。生产仍仅由作者修改。

## 已有实际证据及独立检查

作者新必要测试已读：一份imbalanced 5:1 NN的真实短fit及 `/2` raw/LLR/posterior保存逐数重放、错prior拒绝；native BDF prepared-prefix comparator三paired software seeds得到15单cell、9次physical fits（6 NN+3 classic）与6个proper binary pools，classic复用不称新fit；synthetic MAT raw-Cz6 fit/save/held重放；选定温度producer的OOF且额外control拒绝OOF不拖primary。软件 source checkpoints/seed budgets属于fixtures，不称三完整公共 source→target 随机流程。

额外原生数据 probe 实际消费 `D:/BCI/n2p3-net/mne_data/MNE-bnci-data/~bci/database/008-2014/A01.mat`，raw SHA `cdf9bca4f48c61ee9c9ba998382a4e84392918c9dbc5a43f45ee0052614e25bd`。原执行 archive `ce9b3e2cc7e3d7e6ca8f08757952fd42ca2742ed9b03141d63b924a61f1e951e`、439成员，在一native CPU线程下 C200 对 raw AR8/Burg/cond32/ridge1实际fit；held selection17保存重放exact。prefix与held保持native8原view，model为Cz6/256Hz，A6×8 rank6/native ref/right-earlobe/output ref/CZ均在actualreceipt。只证明该单人/空间producer路径，不是NN、不用它作效能晋升或全矩阵结论。

本评审 `inspect_consumer_evidence.py` 项目venv纯stdlib exit0，独立核实际原raw SHA、完整439成员source archive、保存ensemble语义digest及模型file SHA、36维概率归一、physicalsupport digest与expected/observed20次计数、spatial matrix/ref/fs及当前14阅读文件稳定。软件温度保存3个原文件SHA也一致。没有读取科学数组或重放模型；native prefix byte guard与实际重放动作由原probe断言和源码定位支持，不冒称独立重算。

最终 stdlib 收据已更新核 **16 个当前阅读源码/测试文件**、accepted439成员与13 owned files SHA、actualA01原ce9b439成员、sourcegateaccepted439成员、same95source RED/GREEN、nativebinary15单cell与6pool文件SHA及原温度3文件SHA。`quality_receipt.json` 的12项概率/gradient/空间/binarycaller/frozen回归、10项executed-source ingress门禁、2项ensemble回归及1个candidate实际caller均exit0，Ruff/diffcheck exit0；预期oldgateRED exit1另记。未重复全集。

candidate caller的一份synthetic MAT、三对software source fixtures共1224 prediction cells、60 fit records，requestedK0/200、fixed/grouped程序、incomplete/strictroster失败分母均保留；这些records不等于60 physical fits，也不是三个完整公共source程序。nativebinary软件caller共15 singlecell、9physicalfits（6NN+3 deterministic classic）、6个pool，原保存模型/prediction/pool bytes独立核；K8与1epoch短probe不冒称当前32epoch全量效能。所有结果保留原producer，不因后来accepted版本而重签早先ce9b/95c933/3bc880/012255 probe。
