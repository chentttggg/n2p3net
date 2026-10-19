# 审计 03 — src/transfer 核心（校准/评测隔离、配方、检查点、复用、适配器）

- 审计对象：`src/transfer/` 下 calibrated、recipes、checkpoint、quality、epoch_selection、alignment、decision_loss、candidate_set_loss、subject_adapter、calibration_candidates、calibration_split、result_reuse、binary、raw_continuous、continuous_decision、decision_experiments、source_objective、target_experiments
- 工作树状态：以当前工作树为准（含未跟踪文件）；用 `git ls-files` 判定入库状态
- 约束：只读；报告增量落盘
- 严重级别：P0 正确性/数据完整性/安全/可复现性阻断；P1 显著风险或高代价；P2 可维护性

## 检查计划
1. 入库状态与规模盘点（git ls-files / status）
2. 校准/评测隔离：测试折信息是否进入选模、归一化、早停、质量筛选
3. 被试层级聚合与预算守恒（九选一/候选集/split）
4. 配方与静默默认值、fallback
5. checkpoint 身份、兼容与指纹字段覆盖
6. result_reuse / quality 复用判定宽度
7. 适配器/损失形状与边界、重复实现、exception-pass
8. 未覆盖与存疑

## 结论摘要

1. **P1**：`transfer_only` 在源 checkpoint 不兼容时不会失败，而是经候选全失败路径静默回退为全校准 waveform LDA，arm 身份被降级（`calibrated.py:218-223, 1132-1146` + `calibration_candidates.py:832-856`）。
2. **P1**：公开池 target runner（`run_public_pool_targets.py`）没有挂载整库排除门 `assert_checkpoint_library_exclusion`，只做逐被试排除并信任 checkpoint 自声明的 study membership；主 runner `run_cross_dataset_transfer.py` 才有该门。
3. **P1**：checkpoint 的 contract/architecture/input 校验不绑定 `trunk_state_dict` 的字节；同形状替换权重可通过 checkpoint 内部全部校验，只能依赖外部 file sha256。
4. **P1**：selector 任务的 `fit_calls` 在 manifest 中按 1 计，实际为“候选数 + refit”（3-6 次），dry-run/预算/调度低估计算量。
5. **P1**：BrainSync schema 下传入 `FoldLocalArtifactPolicy/Model` 可绕过 `PersonalEpochRawQualityError` 的 channel-local continuous QC 强制。
6. **P1**：`load_checkpoint_payload` 与 result model artifact 用 `torch.load(weights_only=False)`；安全加载能力已在 calibrated 路径存在但未统一。
7. **P2**：target candidate loss 每步只选一个 decision，未使用 shared `DecisionBatchPlan`，与 source 路径的 `min_decisions_per_batch=2`/`min_rows_per_batch=128` 契约不一致（target 未声明该契约，故降为 P2）。
8. **P2 集群**：`unified_waveform_lda.seed` 硬编码 0（请求 seed=123 时实测仍为 0）；target result key 缺 training contract；拟合缓存 key 缺 block/selection metadata；epoch-selection 与 candidate-loss 的计划预算存在低估；continuous ledger 的删失计数用错 basis 维度；独立 ES 视图未比对标签；`TargetRecipe` 接受负 `weight_decay`。
9. **未发现 P0**：外层 test 行在 B0/最终评测路径基本被明确隔离，inner split 的恢复/时序检查总体严格；本分区无 `except: pass`/TODO，`ruff check src/transfer` 通过。

## 发现

### [P1] transfer_only 在源检查点不兼容时静默降级为全校准 waveform LDA

- 级别：P1（协议身份/科学结论）
- 证据：
  - `src/transfer/calibrated.py:172-197` `load_calibration_source_checkpoint` 捕获 `ValueError/RuntimeError`，返回 `source=None` 且 `gate["status"]="incompatible"`，不抛出。
  - `src/transfer/calibrated.py:218-223` `_calibration_source_failure` 只把 `unreadable` 视为失败；`incompatible` 仅在 `legacy_product`/`unified` 下失败，`transfer_only` 被放行。
  - `src/transfer/calibrated.py:1132-1146` `transfer_only` 的候选初始化集合是 `{"source"}`，随后调用 `select_calibration_candidate`。
  - `src/transfer/calibration_candidates.py:832-833` 候选拟合只捕获 `ValueError/ExpectedSubjectError` 并记为 `failed`；`850-856` 在无 completed 候选时无条件回退到 `waveform_lda` 全校准重拟合，写入 `selection_evidence=False, fallback="full_calibration_waveform_lda"`。
- 影响：声明为 `transfer_only` 的 arm 会以 `status="completed"` 产出 target-only LDA 分数；若下游按 arm 名聚合而不先筛 `selection_evidence/fallback`，会把“仅迁移”结论污染成“目标域 LDA 基线”。预检阶段 `fallback_planned` 有记录（`calibrated.py:398,433`），但执行路径没有硬失败。
- 建议：对候选集合仅含 `source` 的模式（`transfer_only` 及未来同类模式）禁止跨初始化回退；至少要求 `selected["initialization"]` 属于该模式声明的允许集合，否则抛 `ValueError` 或写 `fit_failed`。回退只允许发生在 `five_candidates`/`no_transfer` 这类显式包含 LDA 的模式。
- 验证方式：构造一个 schema 正确但 channel order 不兼容的 checkpoint，跑 `fit_calibrated_predictor(..., selector_mode="transfer_only")`；预期应失败，当前会返回 `model_kind="target_only_shrinkage_lda"`。本次审计未运行该拟合（只读/资源约束），结论来自静态控制流。

### [P1] 公开池 target runner 未执行整库排除门，只依赖逐被试门与自声明 membership

- 级别：P1（训练/评测隔离）
- 证据：
  - `src/transfer/checkpoint.py:115-136` 定义了 `assert_checkpoint_library_exclusion`（整库/整 study 排除），但 `src` 内没有调用者；实际调用仅在 `experiments/run_cross_dataset_transfer.py:272-277`。
  - `experiments/run_public_pool_targets.py:98-104` 的 `execute_job` 直接 `source_payload(...)` → `execute_target_tasks`，没有 `assert_checkpoint_library_exclusion`，也没有把 source task 的 `excluded` 与 target study 对比。
  - `experiments/run_public_pool_targets.py:94` 把 `payload["source_study_membership"]` 原样放进 context；`target_experiments.py:725` 只记录其 digest，不验证其真实性。
  - `src/transfer/checkpoint.py:926-932` `PreparedTransferCheckpoint.assert_target_excluded` 只检查当前 subject 的 authority key，不检查同一 target study 的其他被试。
- 影响：如果 source checkpoint 的训练 ledger 实际包含 target study 的其他被试（或 `excluded` 声明与 checkpoint 不一致），该 runner 不会拒绝；结果仍会以 `development` 证据产出。这直接破坏“整个 target library 不进入 source 预训练”的声明。
- 建议：在 `execute_job`/job 构建处强制调用 `assert_checkpoint_library_exclusion`，并以 source task 的 `excluded`/study membership proofs 校验；把 `source_study_membership` 从“自声明字段”改为“经 proof 验证的字段”。同时为 `assert_checkpoint_library_exclusion` 增加入口级测试覆盖该 runner。
- 验证方式：构造 payload，其 training ledger 含 target study 的另一个被试，但当前 subject 不在其中；当前 `run_public_pool_targets` 路径不会报错（需在测试中 mock 数据/checkpoint）。静态调用图已足够证明门未挂载。

### [P1] checkpoint 元数据不绑定 trunk 权重：契约完整时任意替换权重仍可通过内部校验

- 级别：P1（可复现性/身份完整性；外部 file sha256 仍可兜底）
- 证据：
  - `src/research/contracts.py:146-157` `TrainingRunContract` 字段只有 source cache/identity/snapshot、architecture、preprocessing、optimizer、validation、objective、seed、participant keys；没有 state/weights digest。
  - `src/research/source_selection.py:417-440` 写 checkpoint 时记录 `trunk_state_dict`、contract 与各 digest，但没有 `trunk_state_dict_sha256` 字段。
  - `src/transfer/checkpoint.py:357-369` `_validate_metadata` 只要求 `trunk_state_dict` 是非空 Mapping；后续校验 contract/architecture/preprocessing/decision ledger，均不触及权重值。
  - `src/transfer/checkpoint.py:896-906` 加载用 `strict=False`，只检查 missing/unexpected key；形状相同的替换权重不会触发。
- 影响：checkpoint 内部不是自证的。权重被替换/损坏但 key/形状不变时，checkpoint 自身全部 gate 通过；只有外部记录的 `checkpoint_sha256` 能发现文件变化。若产物被复制/重打包或错误地重新计算外部哈希，模型身份无法从内容恢复。
- 建议：在 checkpoint payload 和 `TrainingRunContract`（或独立字段）加入按名字/类型/形状排序的 `trunk_state_dict_sha256`，加载时重算并比对；同时把该 digest 带入 `source_checkpoint` 身份记录和 result reuse key。
- 验证方式：取任一 checkpoint，交换两个同形状参数的值并重写文件（临时副本），调用 `load_n2p3_trunk_checkpoint`；当前预期能通过（除文件哈希比较外）。本次审计未修改任何 checkpoint，结论来自字段/控制流检查。

### [P1] 任务清单的 fit_calls 对 selector 任务少算 3-6 倍候选拟合

- 级别：P1（预算守恒/高代价）
- 证据：
  - `src/transfer/target_experiments.py:228-231` 为 `legacy_product/transfer_only/no_transfer/five_candidates` 每个 arm 写入 `recipes` 列表（通常 1-5 个候选），但任务没有 `budget` 字段。
  - `src/transfer/target_experiments.py:261-264` `fit_calls=sum(task.get("selection_budget", {}).get("fit_calls_upper_bound", task["arm"] != "zero_shot") ...)`：selector 任务没有 `selection_budget`，默认按 1 计。
  - `src/transfer/target_experiments.py:625-635` 执行时 `select_calibration_candidate` 会逐个拟合 `task["recipes"]`，再对 `allowed` 全量重拟合一次。
  - `src/transfer/calibration_candidates.py:809-864` 每个候选一次 `fit_calibration_candidate`，最后 `fit_calls += 1` 做 refit。
- 影响：dry-run/plan 的 `fit_calls` 与 `cost_estimate`（`target_experiments.py:270-271`）对 selector 任务低估 2-6 倍；调度/配额/断点恢复按此数字分配会超时或超预算。该字段还被用于 manifest 级汇总。
- 建议：为 selector 任务生成 `budget` 和 `fit_calls_upper_bound = len(recipes)+1`（加上缓存命中语义），在 `apply_correctness_budget` 中同样覆盖 `"recipes"` 分支；执行结束后用真实 `fit_execution` 对账。
- 验证方式：对任一 `five_candidates` 任务的 `task["recipes"]` 计数，与 `execute_target_tasks` 的 `fit_execution.requested_fits` 对比；静态代码已显示前者为 5 个候选 + 1 次 refit，而 manifest 计 1。

### [P1] BrainSync 个人校准可绕过 personal raw-QC 强制：FoldLocal QC 在 BrainSync schema 下被静默接受

- 级别：P1（数据质量门/协议绕过）
- 证据：
  - `src/transfer/quality.py:40-42` `validate_personal_epoch_raw_quality` 在 `quality` 不是 `PersonalEpochQualityPolicy/Model` 时直接 `return`。
  - `src/transfer/quality.py:65-76` `_validate_dataset_raw_quality` 只在 `session_schema == BRAIN_SYNC_SESSION_SCHEMA` 时调用上述函数；若传入 `FoldLocalArtifactPolicy/Model`，该函数立即返回，连续的 channel-local QC 要求不再被检查。
  - `src/transfer/quality.py:118-130` `validate_epoch_quality_fit_scope` 明确允许两种 policy，并没有把 BrainSync schema 与 Personal policy 绑定。
  - `src/transfer/quality.py:141-143` `validate_frozen_epoch_quality` 也接受 `FoldLocalArtifactModel`。
- 影响：BrainSync 个人校准可以使用“public fold-local PTP”QC 模型，而不需要 `continuous_qc_model` 的 channel-local exact-constant 证据；这正是 `PersonalEpochRawQualityError` 想禁止的路径。结果仍会写入 `epoch_quality.lifecycle="frozen_explicit"` 并通过后续 `CalibratedPredictor` 校验。
- 建议：在 `validate_epoch_quality_fit_scope`/`fit_epoch_quality` 入口按 `session_schema` 强制 policy 类型：BrainSync 只允许 Personal；非 BrainSync 明确拒绝 Personal 或声明兼容性。对 `epoch_quality_model` 同样检查 `model` 类型与 dataset schema。
- 验证方式：用 BrainSync dataset + `FoldLocalArtifactPolicy` 调用 `validate_epoch_quality_fit_scope`；当前不会抛 `PersonalEpochRawQualityError`。本次审计未运行实际 fit，仅静态检查分支。

### [P1] 公共 checkpoint 入口使用 `torch.load(weights_only=False)`，结果 artifact 同样可执行任意 pickle

- 级别：P1（安全；取决于信任边界）
- 证据：
  - `src/transfer/checkpoint.py:53-56` `load_checkpoint_payload` 直接 `torch.load(path, map_location="cpu", weights_only=False)`；该函数被 20+ 个实验/研究入口调用（如 `experiments/run_cross_dataset_transfer.py:261`、`run_public_pool_targets.py:56`）。
  - `src/transfer/result_reuse.py:114-116` `read_result_artifacts` 对验证过 sha256 的 model artifact 也用 `weights_only=False` 加载。
  - 对比：`src/transfer/calibrated.py:165-169` 的校准路径已使用 `weights_only=True` + `safe_globals`，说明安全加载在本仓库可行。
- 影响：checkpoint/result 若来自共享云 worker、下载目录或被替换的文件，`torch.load` 可执行任意代码；sha256 校验只防篡改，不防“本来就是恶意产物”（record 内自洽的 hash 可被一起重算）。
- 建议：公共入口默认 `weights_only=True`，对已有 NumPy 标量等必要类型用 `safe_globals` 白名单；若必须保留旧格式，提供显式 `trusted=True` 参数并在 CLI 帮助/文档中标注信任边界。
- 验证方式：静态 API 检查；无需实际加载恶意文件。

### [P2] target 结果 key 不含 source training contract，权重相同但契约不同的 checkpoint 可互换复用

- 级别：P2（provenance/可复现性）
- 证据：
  - `src/transfer/target_experiments.py:548-563` 的 `execution_identity` 中 `source_state` 只含 `weights` 的 tensor digest、`statistics`，以及 `recipe`/`snapshot`/`device`，没有 `training_contract_digest` 或 `checkpoint_sha256`。
  - `src/transfer/result_reuse.py:303-322` `target_result_key` 同样只接收 `source_state` 与 `snapshot_sha256`，不接收 checkpoint 契约。
- 影响：两个 checkpoint 若权重/统计量/架构相同（例如同一模型被重新打包，或只改了 source membership/contract 元数据），target 结果缓存会命中，复用 cell 会被当前 `checkpoint_sha256` 重新标注，造成 source provenance 错配。
- 建议：在 `source_state` 中加入 `training_contract_digest` 与（可选）`checkpoint_sha256`；若希望跨文件复用，显式声明“按权重身份复用”并记录两组来源。
- 验证方式：构造两个 payload，weights 相同、`training_contract_digest` 不同，调用 `target_result_key`；当前 key 相同。

### [P2] 候选拟合缓存 key 漏掉 metadata.block_id / selection_repetitions，QC 或决策分组可能命中旧拟合

- 级别：P2（缓存正确性）
- 证据：
  - `src/transfer/calibration_candidates.py:395-415` `_fit_rows_digest` 只覆盖 rows/participants/event timeline 的 event_ids、evidence、session/run、onset/availability、X/y/observed/qc_features；没有 `block_id`、`selection_repetitions`、`calibration_available_timestamp_s` 等 metadata。
  - `src/transfer/quality.py:242-252` 单 session 时用 `metadata.block_id` 决定 QC 的 CV group；`calibration_candidates.py:231-236` 用 `selection_repetitions` 决定 row/column 决策分组。
  - `src/transfer/calibration_candidates.py:454-482` 缓存 key 仅由上述 digest + recipe + policy 等组成。
- 影响：当 X/y/events 相同但 block 或 selection_repetitions 元数据被修正/重标注时，缓存可能复用旧 QC 拟合或旧决策分组，产生“相同输入不同结果”或重复运行不可复现。
- 建议：把 `block_id`、`selection_repetitions` 以及当前 quality policy 实际读取的 metadata 列纳入 `_fit_rows_digest` 或缓存 key。
- 验证方式：构造两个 dataset，仅 `block_id` 不同，调用 `CalibrationCandidateFitCache.fit`；当前第二个会命中第一个的 key。

### [P2] epoch-selection 预算用父视图 inner 行数，independent selection view 行数不同时会低估/高估上限

- 级别：P2（预算守恒）
- 证据：
  - `src/transfer/epoch_selection.py:135-144` `epoch_selection_budget` 只用 `inner_rows` 参数计算 `inner` 的 updates/draws。
  - `src/transfer/target_experiments.py:214-215, 255-256` 传 `inner_rows=len(scope["inner_training_rows"])`（父 dataset 的 80% 行）。
  - `src/transfer/epoch_selection.py:182-194, 197-201` 当传入 `selection_data` 时，真正 fit 的是 `selection_data.dataset` 与 `selection_data.training_mask`，其行数可不同于父视图的 `inner_training_rows`。
- 影响：plan 中 `inner_epochs_upper_bound`、`optimizer_updates_upper_bound`、`trial_draws_upper_bound` 对 independent raw view 不是严格上界；当独立视图训练行更多时会超出声明预算。
- 建议：若存在 `scope_datasets.selection[scope_id]`，用其 training/validation 行数重算 selection budget；或在执行后把实际 `fit_budget` 与 plan 的 upper bound 对账并拒绝超限。
- 验证方式：查看 `IndependentEpochSelectionData.from_views` 的 `mask.sum()` 与 `scope["inner_training_rows"]` 是否可不同；代码未强制相等。

### [P2] 单记录 inner split 的“标签可用时间”检查在缺失 per-recording availability 时被跳过

- 级别：P2（校准时序泄漏的守卫缺口）
- 证据：
  - `src/transfer/calibration_split.py:104-108` 仅当 `label_available_times is not None and recording_start_times is not None` 时检查 inner-training 标签是否早于 validation recording 开始。
  - `src/transfer/calibration_split.py:169-182` `labels_ready` 只有在每个 recording 都有单值 `calibration_available_timestamp_s` 时才传入；否则传 `None`。
- 影响：多 recording 校准若缺少 per-recording availability 元数据，inner-training 的标签可能实际在 validation recording 开始之后才可用，仍会被用于“预测”较早的 validation 记录；这是校准时序泄漏，虽不涉及 outer test。
- 建议：多 recording 且缺少 availability 时直接判 `legal=False`（fail closed），而不是跳过检查；或在 dataset ingress 强制该字段。
- 验证方式：构造 starts 存在、labels_ready 缺失的两 recording 输入，调用 `calibration_inner_split`；当前返回 legal=True。

### [P2] 连续决策模型 ledger 的 right_censored_events 用 basis 行数而非响应样本长度

- 级别：P2（数据账本/删失声明）
- 证据：
  - `src/transfer/continuous_decision.py:225-230` ledger 记录 `right_censored_events = int(np.sum(onsets + len(basis) > len(X)))`；`basis` 是 `(n_basis, n_samples_per_response)`，`len(basis)` 是基函数个数（默认 8），不是响应窗口样本数。
  - `src/transfer/continuous_decision.py:98-104` `_event_design` 实际按 `basis.shape[0]`（响应样本数）截断；`raw_continuous.py:144-145` 也用 `width=response_seconds*SFREQ` 计算 `censored`。
- 影响：K-prefix 校准的最后若干 event 实际被截断，但 ledger 可能报告 `right_censored_events=0`；依赖该字段的审计/权重会误判完整响应。
- 建议：改为 `onsets + basis.shape[0] > len(X)`，并与 `raw_continuous._calibration_view` 的 `censored` 交叉校验。
- 验证方式：构造 stop 距最后 onset 约 0.1s、basis 默认 0.8s 的 calibration record；当前 ledger 计数为 0。

### [P2] target candidate loss 未使用 shared DecisionBatchPlan 的 min 2 decisions / 128 rows 契约

- 级别：P2（目标函数/采样实现与 shared 契约不一致；未证明 target 声明过该契约）
- 证据：
  - `src/transfer/decision_loss.py:93-107` `DecisionLossConfig` 默认 `min_decisions_per_batch=2`、`min_rows_per_batch=128`；`1300-1304` 由此推导每个 batch 的最小决策数与行数；`1165-1379` `build_decision_batch_plan` 是唯一的显式打包器。
  - `src/transfer/subject_adapter.py:1697-1733` 的 `_joint_training_batch` 每步只做 `rng.choice(len(groups))` 选 **一个** decision，再把该 group 的 `evidence_indices` 作为 `extra` 拼在普通 batch 后；没有 `DecisionBatchPlan`、没有 `min_decisions_per_batch`/`min_rows_per_batch` 检查。
  - `src/transfer/calibration_candidates.py:550-553` 只生成 `calibration_decision_groups`，然后 `631-635` 直接传给 `SubjectAdapter.fit(decision_groups=...)`；target 执行路径（`target_experiments.py:655-658`）也走这里。`build_decision_batch_plan` 在 `src` 内只被 `baselines/deep.py` 使用。
- 影响：目标域 P2/candidate-loss 的实际采样是“每 optimizer step 一个 decision”，通常 45 行左右，远小于声明的 128 行、也少于 2 个 decision；与 source P2 的 batch 契约不可比，且 `DecisionLossConfig.record()` 若被写入 ledger 会声称未执行的约束。梯度方差/有效学习率与预注册配方不一致。
- 建议：target adapter 也使用 `build_decision_batch_plan` 产出的 batch（或显式声明 target 使用单 decision sampler 并禁止 `min_decisions_per_batch` 字段），在 `fit_history` 记录实际 decision 数/行数；把 `DecisionLossConfig` 纳入 target recipe 或拒绝在 target 路径使用 source 的 batch 契约。
- 验证方式：对 `candidate_loss_weight>0` 的 `unified` 任务，检查 `fit_budget.candidate_draws` 与每个决策组大小；代码显示每步只选一个 group，且没有 `DecisionBatchPlan`。

### [P2] target candidate loss 的额外 decision 行不计入 `target_batch_budget` 计划

- 级别：P2（预算守恒）
- 证据：
  - `src/transfer/subject_adapter.py:1711-1717` `workset = torch.cat((rows, extra))`，其中 `extra` 是一个完整 decision 的可用行；`1188-1192` 才把 `decision_diagnostics["draws"]` 加到运行预算 `budget["draws"]`。
  - `src/transfer/recipes.py:170-189` `target_batch_budget` 只按 `n_fit/batch_size/epochs` 估算普通 CE 行，不含 decision 额外行。
  - `src/transfer/target_experiments.py:247-249` 任务预算直接使用 `target_batch_budget`；`apply_correctness_budget` 也只用它。
- 影响：带 candidate loss 的 target 任务（尤其 `unified`）实际 trial draws 高于 plan；决策组越大（例如 45 或 9×45），低估越明显。计划中的并行/GPU 配额与完成时间预测偏乐观。
- 建议：在 plan 中把每步 `decision_rows` 或 `candidate_draws_upper_bound` 加上（按每 epoch batch 数×决策组行数估算），并在执行后用 `fit_budget.candidate_draws` 对账；或让 `target_batch_budget` 接收 `decision_rows_per_batch`。
- 验证方式：运行一个短的 candidate-loss fit，比较 `fit_history["fit_budget"]["draws"]` 与同一 recipe 的 `target_batch_budget(... )["draws"]`；前者更大。

### [P2] legacy 校准路径在训练后重新读 checkpoint 文件计算 sha256（TOCTOU）

- 级别：P2（来源身份/可复现性）
- 证据：
  - `src/transfer/calibrated.py:945-951` 先用 `_load_calibration_checkpoint(source_checkpoint)` 读入并校验 payload，随后用该 payload 构建 source trunk。
  - `src/transfer/calibrated.py:1005` 训练结束后才执行 `hashlib.sha256(Path(source_checkpoint).read_bytes()).hexdigest()` 作为 `source_checkpoint_sha256`。
  - 对比 candidate 路径 `calibrated.py:179-181` 在读取字节后立即计算 `source_hash` 并把同一份 bytes 传给 `_load_calibration_checkpoint`。
- 影响：若 checkpoint 在两次读取之间被替换（多进程/共享存储/人工更新），模型 metadata 会记录新文件的 sha256，而实际权重来自旧文件；结果无法复现，且与 `source_checkpoint_sha256` 审计语义冲突。
- 建议：legacy 路径改为先读一次 bytes、计算 hash，再从同一 `BytesIO` 加载；或直接复用 `load_calibration_source_checkpoint` 返回的 `checkpoint_sha256`。
- 验证方式：静态路径对比；可用临时替换文件模拟（本次未执行）。

### [P2] `candidate_set_loss` 是未被生产代码使用的重复实现，且全不完整时返回零损失

- 级别：P2（重复实现/静默零梯度）
- 证据：
  - `src/transfer/candidate_set_loss.py:1-23` 自述它是历史 mean-scale row/column loss，并说统一 sum 方法已改用 `MembershipDecision.row_column`/`membership_decision_loss`；grep 显示 `row_column_candidate_loss` 只在 `tests/transfer/test_candidate_set_loss.py` 被调用，生产 `src` 无调用者。
  - `src/transfer/candidate_set_loss.py:174-182` 当 `complete_count == 0` 时返回 `zero`（无有效梯度）并只把 `complete_count/skipped_count` 放进 diagnostics；docstring 第 69-70 行却声称 “incomplete decisions ... do not contribute a silent zero gradient”。
- 影响：维护者可能误以为生产路径仍在使用该 loss；若未来接入，整批 decision 都无效时会静默训练（candidate 项无梯度），与文案相反。
- 建议：删除该模块及对应测试，或在 `complete_count==0` 时抛错/返回 `None` 让调用方显式处理；把 `MembershipDecision.row_column` 作为唯一实现并在文档索引中标注。
- 验证方式：grep 调用图；运行该函数构造全部 `selection_ids` 不完整，观察返回 0 标量。

### [P2] `TargetRecipe`/`SubjectAdapterConfig` 接受负 weight_decay，优化超参校验不完整

- 级别：P2（配置健壮性）
- 证据：`src/transfer/subject_adapter.py:96-165` 的 `SubjectAdapterConfig.__post_init__` 校验了 `epochs/batch_size/lr/max_updates/candidate_loss_*` 等，但没有校验 `weight_decay`、`seed`、`target_stat_weight` 之外的若干数值；`src/transfer/recipes.py:41-98` 的 `TargetRecipe.__post_init__` 只通过 `adapter_config()` 间接校验。
- 实测：`$env:PYTHONPATH='src'; .venv/Scripts/python.exe -c "from transfer.recipes import TargetRecipe; print(TargetRecipe('probe', weight_decay=-1.0).record())"` 输出包含 `'weight_decay': -1.0`，未抛异常。
- 影响：负 `weight_decay` 在 AdamW 中会放大参数（而非正则化），属于静默的错误配置；`seed` 非整数也会在后续 `SeedSequence` 处以不明确的 TypeError 失败。
- 建议：在 `SubjectAdapterConfig.__post_init__` 增加 `weight_decay >= 0`、`seed` 为 int、`target_stat_weight` 已存在校验；`TargetRecipe` 增加显式类型/范围校验并复用同一校验函数。
- 验证方式：上述单行 import probe（只读，无训练）。

### [P2] 独立 early-stopping 视图只校验事件归属，不校验标签与父校准一致

- 级别：P2（选模标签完整性）
- 证据：
  - `src/transfer/epoch_selection.py:72-76` `IndependentEpochSelectionData.from_views` 只校验 authority keys、preprocessing、channel_names 与两边类别都存在；没有把 `view.y` 与 `calibration.y` 或 `target_raw_view_identity` 逐事件比较。
  - `src/transfer/epoch_selection.py:186-194` `fit_with_epoch_selection` 只检查 `selection_data` 消费的 event_ids 是父校准已授权 event 的子集，不检查对应标签。
  - 同一函数 `90-92` 只把 `training_input`/`validation_input` 的 identity 写入 evidence，未与父 `target_raw_view_identity(dataset)` 做相等校验。
- 影响：若独立视图由错误的标签映射/重放脚本生成，inner selection 会在与父校准不一致的标签上早期停止/选 epoch，然后把选中 epoch 数用于父校准 refit；outer test 不泄漏，但选模证据已被污染。
- 建议：在 `from_views` 中对每个 event_id 比较 `y`，或比较 `target_raw_view_identity` 的 label digest；在 `fit_with_epoch_selection` 中对 `selection_data` 的 training/validation 标签与父 dataset 按 event_id 对齐验证。
- 验证方式：构造 selection_data，翻转一个 event 的 y 但保留 event_id/raw scope；当前会通过 `from_views` 与 event 子集检查。静态分支已足够，未运行 fit。

## 未能验证与存疑

- **未运行真实训练/校准/评测**：除只读 import 探针外，没有拟合任何模型；所有“会静默回退/会复用”的结论来自控制流与静态调用图，未构造端到端数据复现。
- **public pool 整库排除**：`source_matrix_tasks` 的 `excluded` 设计上会让 source pool 排除 target study，因此“绕过门”是 defense-in-depth/错误 checkpoint 场景；未证明当前实际 checkpoint 真的含 target study 其他被试。
- **checkpoint 权重替换**：未实际修改 checkpoint 文件验证“同形状权重可过内部校验”；结论基于 `TrainingRunContract` 字段表与 `_validate_metadata` 未计算 state digest。
- **`torch.load` 安全**：是否构成 P1 取决于威胁模型；如果所有 checkpoint/result 都只由本机可信进程产生，风险降为 P2。
- **target 结果 key 缺 training contract**：权重/统计量相同而契约不同的两个 checkpoint 是否实际存在，未在数据中查找；属于潜在身份混淆。
- **fit cache 缺 block_id**：只在 `block_id` 或 `selection_repetitions` 元数据被重写且 X/y/events 不变时触发；正常 pipeline 可能从不发生。
- **epoch-selection 预算**：没有实际运行 independent raw view 的 ES 任务；不清楚 `selection_data.training_mask.sum()` 与父 `inner_training_rows` 的实际差异幅度。
- **continuous censoring**：未用真实 BNCI continuous 记录核对 `right_censored_events`；结论是维度/字段语义错误，影响账本而非拟合数值。
- **未覆盖代码**：`raw_continuous.prepare_bnci_continuous` 的后半段、`decision_experiments.py`、`binary.dataset_calibration_splits` 的部分分支、`source_objective` 的完整验证逻辑、`subject_adapter` 的 1744 行中约一半未逐行阅读；`alignment` 已通读。
- **未验证测试**：没有运行 pytest（测试目录重组中，部分旧 `tests/test_*.py` 已删除）；只运行 `ruff check src/transfer`。
- **行号时效**：报告基于当前工作树（`src/transfer` 有 14 个 modified、9 个 untracked）；若之后提交/重排，行号会漂移。
- **没有 P0 结论**：本报告不声称仓库整体无 P0；只表示本分区已读代码中没有发现直接导致 test 泄漏或不可复现的阻断级缺陷。

## 已检查范围

### 代码

- 通读：`src/transfer/recipes.py`、`calibration_split.py`、`epoch_selection.py`、`quality.py`、`result_reuse.py`、`binary.py`、`alignment.py`、`candidate_set_loss.py`；`checkpoint.py` 的 header/identity/architecture/metadata/stats/loader 主要函数；`target_experiments.py` 的 build/apply/execute；`decision_loss.py` 的 config/metadata/plan/loss；`source_objective.py` 的 config/loss；`continuous_decision.py` 的 model/fit/score；`raw_continuous.py` 的 scope 准备。
- 定点精读：`calibrated.py` 的 preflight、legacy fit、candidate fit、source gate、hash/save/load；`calibration_candidates.py` 的 groups、fit、cache、score、select；`subject_adapter.py` 的 config、fit 主循环、refit、budget、decision batch；`decision_loss.py` 的 DecisionLevelLoss/packed_loss；`target_experiments.py` 的 source/execution/reuse 分支；`quality.py` 的 fit/validate/transform。
- 未逐行：`asmmd.py`、`batch_cohort.py`、`candidate_decision.py`、`candidate_within_subject.py`、`cohort.py`、`evaluation.py`、`exact_covariance.py`、`heads.py`、`information_bound.py`、`losses.py`、`masking.py`、`mask_probe.py`、`noise_model.py`、`operator_lda.py`、`outcomes.py`、`pretraining.py`、`temporal_noise.py`、`within_subject.py`（不属于本分区重点，未审计）。

### 命令与实测

- `git status --porcelain -- src/transfer`：14 个 modified、9 个 untracked；`git ls-files src/transfer`：28 个 tracked `.py`；`git ls-files` 用于判定入库状态。
- `ruff check src/transfer`：**All checks passed**（只读）。
- import 探针（PYTHONDONTWRITEBYTECODE=1，`.venv/Scripts/python.exe`）：
  - `unified_target_recipes(membership_available=True, source_prior_weight=0.5, seed=123)` 返回 `unified_waveform_lda.seed == 0`，其余 recipe seed 为 123。
  - `TargetRecipe('probe', weight_decay=-1.0)` 被接受，证明负 weight_decay 未被校验。
- 未运行 pytest；未运行任何训练/拟合；未修改、删除、重命名任何已有文件；未创建 scratch 脚本；唯一写入是本报文件（以及 `tmp/audit` 已存在目录内的本报告）。

### 对结论强度的标注

- 已证实：文件/行号存在、控制流可达、import 探针结果、ruff 结果。
- 疑似：public pool checkpoint 实际含 target study 其他被试、fit cache 实际命中旧 metadata、两个 checkpoint 权重相同契约不同——均有代码条件，但未构造实际数据。
- 无法验证：未在真实 BNCI/BrainSync 数据或 GPU 上执行；未验证历史结果产物；未审计 `frozen/`、`mne_data/`、`.git` 或 Lean 子项目。
