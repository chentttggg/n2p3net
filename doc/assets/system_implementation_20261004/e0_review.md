# E0 独立反例审查（2026-10-04）

本记录审查 root 第一轮 E0 实施，比较基线为已核验冻结 `frozen/system_implementation_20261004_before/source.source-only.tar.gz`。只读检查 production；新增探针与证据放本目录。审查快照 SHA 及规范化 changed hunks 在 [e0_review/review_snapshot.json](e0_review/review_snapshot.json) 和对应 `.diff` 文件。root 已收到实时发现并开始修复，因此以下“通过/接受”均指被审查快照，不冒充后续修复代码的现状。

Spec 为 `doc/evidence/system_research_20261004.zh.md` E0/S0：“分开 EvidenceCohortIdentity、MaterializedInputIdentity、ContrastPlan；同物理事件允许预声明算子变化，数值变化不得复用旧预测；fit 与 score 都绑定”。标准为 AGENTS 与活动技术/输入指南，强调 canonical owner、实际数值证据、保留历史、测试 seam。

## Spec 发现

### R1 — P1：独立 inner-validation 数值变化仍复用旧选轮结果

`target_experiments.execute_target_tasks` 新 execution key 已覆盖 full-calibration/test 的实际 X/y/mask，但独立 `scope_datasets.selection[scope].dataset` 只通过历史 `scope.inner_split` 声明进入 key，没有重新计算实际输入。

[probe_review.py](e0_review/probe_review.py) 使用真实 CPU checkpoint、真实 `fit_with_epoch_selection`、真实 `TargetResultReuse` 和真实 executor：first fit 成功后，仅将 inner-validation X×50、y 翻转；full-calibration/test 保持不变。第二次执行成功，execution identity 未变，`fit_calls=0, cache_hits=2`，沿用旧验证损失 `[0.6905062795, 0.6900655627]`。禁用 reuse 的新算得到 `[0.6921222210, 0.6906992197]`。

这不是手算 key 或仅模拟拟合函数。见 [probe_results.json](e0_review/probe_results.json)。需要在复用前对实际 inner-training / validation X、y、mask、物理行、scope 及选轮配置建立身份；验证失败不能变成 subject fit_failure 后继续宣称有效复用。独立 `recording_order` 参数也应进入实际拟合上下文。

### R2 — P1：遗漏候选监督上下文，同一 input digest 对应“有 Q1”与“无 Q1”

当前 `evaluation_input_identity` 只取部分 materialized timeline 字段，遗漏完整计划 timeline 的 `complete`、group/status、未 materialized 事件，以及 `candidate_vocabulary/candidate_task_contract` 等真实拟合输入。`calibration_decision_groups` 却实际消费这些字段。

[probe_decision_context.py](e0_review/probe_decision_context.py) 使用公共合成 candidate fixture 与真实 `calibration_decision_groups`：原数据产生 2 个可用 decision groups；只移除 vocabulary，返回 `membership_unavailable`；只将 timeline.complete 设为 false，返回 `incomplete_physical_timeline`。两种修改下 fit/score 的 `evaluation_input_identity.digest` 都完全不变。

见 [decision_context_results.json](e0_review/decision_context_results.json)。因此数值 X/y/mask 相同并不等于 Q1 的监督相同。复用身份应保守绑定完整计划排程、candidate 物理合同/词表和实际读取的相关 metadata；科研对比可另定义哪些语义允许改变，不能与复用等价关系混合。

### R3 — P1：任意或部分 consumed_inputs 被提升成“原始字节已绑定”

`comparison._raw_inputs` 收集任何角色的 `consumed_inputs`，`raw_bytes_bound=bool(raw)`。两个独立探针均通过本不应准入的 baseline operator contrast：

1. 只有 role=`events` 的 sidecar SHA，没有原始 EEG raw SHA，仍声明 raw bound=true。
2. 真实 `DataLineage.derive(... operation=verified_source_record_preparation)` 已知父与未知父拼接，只选未知父行 fit/score；无关已知父的一个 raw SHA 仍使当前 score raw bound=true。

见 [probe_results.json](e0_review/probe_results.json) 与 [coverage_results.json](e0_review/coverage_results.json)。应要求真实 raw role，并验证每个实际参与行对应的 raw parent 都有合法 producer 绑定；未知或部分覆盖只能明确降级，不能借旁支哈希放行。

### R4 — P1：声明 baseline 变化会连带允许未声明实现代码变化

`ContrastPlan.validate_pair` 对 `operator_implementation` 的检查只在无其他声明参数改变时触发。因此 baseline_mode 改变后，execution implementation source SHA 可同时改变而未声明。

不仅简化 identity dict 可复现；[probe_coverage.py](e0_review/probe_coverage.py) 对真实 Huebner preprocessing receipt 修改 baseline 声明与 `implementation.source_sha256[preprocess.py]`，重算 receipt SHA。两侧 `dataset.validate()` 都成功，baseline-only contrast 仍被接受。见 [coverage_results.json](e0_review/coverage_results.json)。

应将 implementation 身份与参数导致的 declaration/filter/grid 变化拆开规范化比较；参数轴允许变不等于源码版本也允许变。若 baseline 比较确实使用不同实现，调用者必须同时声明 operator_implementation，否则不是单因素 baseline 归因。

### R5 — P2：历史 fallback 不能证实完整“同输入”

旧记录 fallback 只要求 calibration row indices、test indices、test 数值/标签摘要相同。两份真实 dataset 仅改变 calibration X×3，生成的四个旧 digest 完全相同，`assert_matched_cells` 接受。见 [probe_results.json](e0_review/probe_results.json)。

这类历史证据仍可保留做审计与描述，但不能输出与完整 MaterializedInputIdentity 等同的空轴 plan 而不提示 fit-input unknown。建议显式 historical descriptive scope，或严格统计入口拒绝缺失 fit materialization 的记录；不能补写从未记录的历史 fit SHA。

### R6 — P2：P2 CLI 将一个 ContrastPlan 命名空间传入两类统计

`analyze_training_chain` 将同一 `analysis_options['contrast_plans']` 先传 target treatment contrasts，再原样传自动 `analyze_source_objective_contrasts`。两族 contrast 名称不同，非空 target plan 将在后者以 `input ContrastPlans must name declared contrasts` 失败。底层传播复现见 [p2_plan_propagation.json](e0_review/p2_plan_propagation.json)。

需要按 contrast family 分离或分发完整计划，并在入口对所有未知键统一拒绝；不要为修复此问题在底层静默丢掉未知计划。

## Standards 发现

核心方向符合 owner 集中化：`research.comparison` 把比较与复用分开，binary helper 与 full stats 共用配对准入。但 R1/R2 表明“实际拟合输入”的定义仍散落在 evaluator 与具体拟合器之间；这是已可复现的行为缺口，不是依据文件大小或抽象偏好的重构建议。建议由 fit owner 输出统一消费上下文，再由 execution key 使用，避免继续向 runner 人工追加一份不完整字段表。

候选 valid reducer 的 cohort 计算本轮改善正确。独立三臂×两重复探针：A/B 的 p1 完整，C 缺 r1。global complete cohort 正确为空，主指标与 replicate 均为 null，interval 为 not_estimable；`json.dumps(..., allow_nan=False)` 成功，未发现空 cohort NaN。见 [empty_valid_cohort.json](e0_review/empty_valid_cohort.json)。

不过 A-B 排除 p1 的 support 行中 `missing_left_replicates=[]`、`missing_right_replicates=[]`，只写 incomplete_global_grid，未指明 C/r1。作为“完整支持矩阵”仍不足：建议补 `missing_by_arm`，使第三臂造成的排除可解释。这是报告信息缺口，数值计算在此反例正确。

## 验证与边界

运行新增三个独立 probe，未修改 production，未跑全套测试或性能矩阵。真实 CPU executor 产物位于 `experiments/runs/system_implementation_20261004/e0_review/`。后续 coverage / candidate probe 使用固定的 `comparison_snapshot.py`，避免 root 并行修复污染审查基线。初始 review 模块 SHA 为 `a7e6115dc702658e4e6f28a5d6c0f6a8c7e4733d5a4b0ffa6ee507c90f9fa589`，target executor SHA 为 `21852af67118e7c46cb8ff4097ce88edf87ef9484d6498e987de986faed7977b`。

R1–R4 应在 S0 闭合前修复并用同类实际反例复验；本报告不因现有测试通过而放行。独立审查已确认原有 positive input复用路径存在且可执行，修复应保留同输入零额外拟合的能力。
