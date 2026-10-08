# 当前 public consumer：Standards 独立评审

作者树为 `C:/Users/chent/.codex/worktrees/calibration-g-prior/n2p3-net`，branch `codex/current-public-consumers`，基线 `a2975557d6e5e3ae99ff2f8d692ecc8a7e524ab2`。作者为唯一生产写者；本审查只读生产代码、设计与既有证据，不运行 Torch / 宽 pytest / 训练。source fd62 已完成的软件质量出口保持原身份，不追认为本切片。

**最终 Standards 结论：complete，0 项剩余关键发现。** 绑定 accepted archive `e6499b90778644e0df3ffd899a39aaf004ab792b1302f29cf2105b02570955ff`、439成员，manifest `75c2cb3ebb0da2d8a99be330705d887c6ba97343116479c9c75bece72ff69462`。独立 stdlib 检查 archive 每成员与当前源码、13 owned 文件 SHA、旧A01物理冻结439成员与保留软件三文件全部匹配，退出0。脚本/原始JSON为本目录 `inspect_consumer_artifacts.py` / `consumer_source_binding.json`，没有载入Torch或修改生产。

已复用 astra-research-workflow engineering 模块与 code-review Standards 轴；已读文档索引、六指南相关约束。当前设计输入为 root 授权和 `tmp/public_source_pipeline_20261008/current_experiment_design`，较早实验矩阵不作为必要范围。

## 入口到数值状态覆盖

| 关系 | 已读现状 / 新切片核验点 | 状态 |
|---|---|---|
| actual target loss → predictor | SubjectAdapter fit / refit 真正计算 `β_C=nnegative/npositive`、π_C，predict_logit 输出未校正 raw z；新冻结 state 要保存实际拟合数值与 fit authority，不能猜默认值。 | 原链已读，新 diff 待核 |
| posterior / LLR / source | binary posterior 为 sigmoid(raw z−logβ_C)，candidate 使用再扣 logitπ_C 的 LLR；source posterior需其原 source calibration，不能 sigmoid 已去 prior 的 LLR。 | 原链已读，新 diff 待核 |
| classic actual prior | waveform / Toeplitz / xDAWN fit 可能只消费真实可用 training rows；其 prior 必须从实际拟合行派生，不从未消费的完整 requested C 猜测。 | 原链已读，新 diff 待核 |
| spatial rawC → OOF → held | 现有 raw owner持native256Hz/8导，profile持none/128Hz/179；新 observed spatial owner须显式选真导与Cz差分，缺依赖不可补0，raw协方差不能含伪零槽。raw native8 anchor保持单独身份。 | 原链已读，新 diff 待核 |
| shared final / temperature subset | 原 fit_calibration_prefix_programs 共享一次all-C expert状态，OOF在完整C组内；新temperature_expert_ids须在OOF / record / save / load / score同一声明，只校primary5，不重归成功subset权重。 | 原链已读，新 diff 待核 |
| binary metric / fixed pool | comparator当前只有真实ranking AUC；新增NLL/Brier必须消费posterior并核rawbyte/participant/seed/K/test完整配对，固定pool缺producer要保留requested失败。 | 原链已读，新 diff 待核 |
| old/new package identities | 新数值与概率语义保存并绑定原执行源码；历史包按原identity解释，缺实际βπ的旧target包不认证为新proper-posterior。 | 新 diff 待核 |

已读完整核心：FrozenTrialPredictor、FittedCalibrationCandidate/SubjectAdapter实际fit与predict、StructuredERP fit/posterior、CalibrationExpertProgram/_FittedExpert/FittedEnsemble fit/score/save/load/programs、SelectionEpochProfile/CalibrationPrefix/raw group builders、两CLI source/matched controls与实际score入口、EnsembleScorer完整失败合同。数值输入、predictor calibration、raw spatial与温度的拟合/冻结/重载关系优先于新增字段名称。

## 当前发现与出口

当前为进行中，未把作者已接手的既有缺口重复列为新发现。新 diff 中独立确认并回交以下关系问题，当前均已读到静态修复，待必要行为收据与最终字节绑定：

- 新 proper-posterior 入口原本认证旧 `batch_class_mass`。现只允许 fixed full-C class mass 目标 risk；旧归约保留原 ranking / candidate 语义，不认证新 proper posterior。
- 新 `training_class_risk_` 原本不随初始化 / 失败清理，先成功后 early failure 会保留上一份 C 的风险。现与 weight / prior / history 一起初始化并清空。
- 三 paired seed 的 classic deterministic alias 原本不传播新增概率字段，后两 seed 的 pool 被误记 missing。现复用完整概率状态、风险、语义与原 artifact，不追加拟合。
- raw spatial dict 原本可被外部 mutation 改动已拟合程序，load 处未核实际 operator / channels。现 spec ingress 深冻结，并在实际 frozen expert 构造处核 projection receipt 与数值模型一次。
- `/1` 旧包附新合同原本能进入新概率程序。load 已接双向版本关系，且 origin / raw_score_kind 与实际 numerical kind 关联，旧模型不可用附加新字段重签转换身份。

binary pool 当前在原 row / label 对照之外消费既有 `ContrastPlan(())` 的完整实际比较身份、预测文件 SHA，输出 pool 文件 SHA；同一 subject / actual K / source-target seed 下 unique arm 由 comparator preflight 持有，actual-K alias只复用原cell，不额外散落重复检查。

raw spatial 当前 diff 已保持原 physical decision 支持，数值另绑 projection / original raw / native rate / output bytes。OOF只拟合显式 primary temperature subset，shared all-C model states保持同一对象；subset外控制温度为1且不改变 calibrated identity。candidate预测与最终metrics / contrasts两层均消费 mixture_calibration_strategies，fixed控制不进入 grouped treatment。binary actual / reload各一次同时返回 raw / LLR / posterior / probability，避免新 proper-probability 路径多次重复 forward。

新 diff 将按具体触发、结果及真实关系证据核查。源码稳定后才绑定文件 SHA、必要单项反例 / native probe 身份和最终结论。没有 findings 条数要求，不以风格偏好或宽测试计数替代关系审查。

## 已关闭：真实 source 的 K0 / OOF 执行身份

静态确认 P1：`validate_independent_source_input` 要求实际 source `execution_identity`；生产 source checkpoint 的 input preprocessing binding 因而非空。`PreparedTransferCheckpoint → checkpoint.py` 在载入 source / matched-scratch architecture 时要求目标 `preprocessing_execution_identity` 相同。但 K0 的 `CalibrationPrefix.trial_target` 只给 source_reference；OOF `build_calibration_epoch_dataset` 有 selection_epoch_profile / selection_epoch_execution，却刻意没有伪造 legacy mne_raw execution。两者当前 identity为None，真实已训练 source 因此拒绝 K0 与 NN / matched-scratch OOF。现三seed软件 caller 的 synthetic source checkpoint 没有实际 execution identity，无法覆盖此接入缺口。

修后采用 `validate_selection_checkpoint_execution` 同一 alternative owner：source 原合法 preprocessing receipts、实际 SOS / padding 与显式数值 family兼容；K0只声明编译的future profile能力，不宣称已预处理或拟合；OOF核真实 profile / operator / shape / dtype / reference / output bytes+mask、原物理支持摘要、实际source epoch spans / padding / availability。未把空执行身份视为compatible，未伪造mne_raw凭证，也未要求有限矩阵/物理观察相同。receipt tuple/list表示经统一规范，合法保存载入不误拒。

fresh零目标拟合收据 `source_ingress_receipt_v2.json` SHA `75cdf7bd3f8d1d4d30b9747d422c57e8a09d237bea4d0bb35c373fc3e83d67ef` 绑定消费者 executed capsule `3bc880af9662a38cd71ab8f09542f217033e1df6f434b587df7f3adca342c955`（439成员全部核验）。原 source checkpoint保持producer `95c93303…`；actual public A01作为真实输入而source来自软件BDF开发producer，二者角色分开。K0无targetfit、实际held概率保存重载exact；240行OOF ingress及保存NPZ后的cold attested load再准入成功；假numerical execution拒绝。旧v1收据原字节保留，不由finalcap倒签。

另已只读真实candidate CLI软件integration v2：一个软件participant、三个pairedseed、K0/K200、1224prediction cells终态completed；它使用自己的fixture源和程序，不宣称完整五primary公共效能矩阵。

最终 accepted capsule已fresh执行上述零fit source ingress，收据 SHA `174009c08545024ee510d56317ce3a68be7a0ff2aa7dad743bcec3f43506f9bc` 的 executed source为e649。K0 frozen capability另与实际source stem几何、trained preprocessing / orderedchannels / reference绑定，重签错trained profile明确拒绝。Frozen probability transform在constructor/load入口验证并编译为不可变数值状态，推理不重复训练合同验证。源级P1的原fd62 RED收据真实保存 `actual_source_execution_present=true`、合法K0被拒、targetfits0；不把修后执行身份倒填原RED。

作者必要质量收据分列软件12项、执行门禁10项、legacy ensemble2项、三seed CLI1项与真实A01单rawfit和accepted零targetfitprobe；均按各自执行源/数据role记账。Ruff全部12 owned Python与diff-check退出0。审查者逐项读取源码、现有收据与产物并做独立字节核验，不声称重跑作者NN / pytest。新固定C class mass风险/部分尾批、actualbeta/pi/sourceprior、旧格式拒properprob、真实physicalrawprojection与共享final/selectedOOF、请求alias/失败分母和对应save/reload关系已经闭合。

新程序的概率是声明模型/训练总体下的forecast，不证明有限NN已实证校准；source / target / candidate posterior估计量分开。公共全量两panel、实际source三流程成绩、云吞吐与成人确认保持 execution / research各自未完成出口。本审查没有使用软件通过或单participant数值probe为它们背书。consumer 与perf合并时，epochs / checkpoint共同合同须在新的实际源冻结后再验联合caller，旧probe保留原身份。
