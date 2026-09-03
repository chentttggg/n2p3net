# N2P3-Net 下一轮 Agent D 代码链交叉审计

_版本：2026-09-03 · 审计时点：2026-09-03 12:26 +08:00 · 性质：集成前源码与产物审计，不是 P0/P1/P2 完成声明_

---

## 📋 源码事实

### 结论先行

截至本审计快照，P0/P1/P2 尚未形成从实验计划到结果 manifest 的闭合执行链。
P0 与 P1 在审计过程中出现了并发、未提交的模块实现，但五个活动 runner 和 promotion
plan schema 均未接入这些配置；P2 在集成前快照没有训练实现。局部模块存在、单元测试
存在或 checkpoint 能解析某个字段，均不能替代 CLI/config -> model -> trainer -> checkpoint
-> evaluation -> manifest 的端到端闭合。

已确认的最高优先级阻塞如下：

1. `run_pretrain_supervised.py` 仅在 `selection_domain` 非空时把
   `source_domain_ids` 传给 trainer；Split-BN 或 source residual 单独启用时没有 domain
   axis。当前云端模板恰好使用 `selection_domain=null`，同时打开 feature auxiliary loss，
   因而模板在真实训练中会确定性失败
2. `run_candidate_promotion_matrix.py` 的 plan schema 与 `TrainingConfig` 没有 P0/P1/P2
   字段，DAG argv 也不会传这些字段。模块内新增配置无法由正式入口启用
3. promotion DAG 固定调用 row/column runner，但当前 feature-transfer 云端模板把 target
   写成 BrainSync direct 9-choice cache。`run_candidate_cross_decision.py` 严格要求当前只支持
   row/column 的 `CandidateTaskContract`，该模板不能执行到有效结果
4. P2 公式 `y_i=1[c_i=d*]` 只适用于 direct candidate membership。当前
   `CandidateTaskContract` 只允许 `row_column_intersection`；一个 character truth 对应两个
   positive flash candidates，不能直接塞入单一 9-way CE
5. source/cache/checkpoint/plan 已有多段 digest，但 environment 未进入语义 contract，且
   source archive digest 没有证明 subprocess 实际执行的当前 checkout 等于该 archive。
   当前不存在一条完整的 source -> environment -> plan -> checkpoint -> result 证据链

### 审计快照与并发边界

阅读顺序严格遵循活动规范：

| 顺序 | 文件 | SHA-256 |
|---:|---|---|
| 1 | `doc/research_program.zh.md` | `472e2ae06ce663ab71e5fe6dc7eba552ff1f7f36d112502416d035b988d74740` |
| 2 | `doc/architecture_decision_20260902.zh.md` | `28f9c9d0472c33ac7213135e23f2579d8cdc10c91cce2e6a285996a55714d012` |
| 3 | `doc/implementation_plan_20260902.zh.md` | `f1e461e014a201472854ebce46262ce6a9caba875d06fced8a27ac369c98bf84` |
| 4 | `doc/math_verification_20260902.zh.md` | `e8464f22fe8ede2a1cd5793fc722103db3ca99012f90365e9ffdf8a20b574df0` |

审计启动时 HEAD 为 `b054b0b6b929758d32ba97344a22bc13f6de3748`。八个指定文件的
起始哈希如下；五个 runner 与 `contracts.py` 在第二次快照仍未变化：

| 文件 | 集成前 SHA-256 |
|---|---|
| `experiments/run_pretrain.py` | `d5ffda21f841bc05b54f3e65c446581da7944da02da410a4c1cb4d2acd51cec6` |
| `experiments/run_pretrain_supervised.py` | `26fd983dade3f12b2ac72bed1636f0025e4ce67d737c3c257bab224e3a42ad49` |
| `experiments/run_within_subject_transfer.py` | `5ed2f226d12e66d9ecaaa40b4f63395e808efce150ee5af8b116006efac5ca11` |
| `experiments/run_brainsync_cross_decision.py` | `4cc494717dd7a4ef394b90a2b84dd2987585fe30f56a424881987e6bae0e912e` |
| `experiments/run_candidate_cross_decision.py` | `681604e480e468ad974b57fd83841fc7e2d1f37be17a44838041c4d1f5f8e716` |
| `src/baselines/deep.py` | `f91416223a31afeb8c1949b02f3666196e8f4db70a0e7e3077b6f30217bc37f2` |
| `src/transfer/checkpoint.py` | `895da6e128289926ab72e30d1e34528db5dd6ee34cbbec8c7dd20667f07f44ae` |
| `src/research/contracts.py` | `1fbd4226f7320743a75b3c019e20bde575e02fb8ae6ac89971dc32603f8be40a` |

启动时除用户已有的 `pyproject.toml`、`requirements.txt` 外，指定源码与 HEAD 一致。
审计中 P0/P1 Agent 并发修改了 `deep.py`、`n2p3net.py`、`adapters.py`、
`checkpoint.py` 并新增 `splitbn.py` 与测试。本文件只把这些内容称为“并发观察”，不把
它们当作已验收实现；主线程集成后必须重新核对最终哈希、调用点与测试结果。

12:14--12:26 的并发中间快照为：`deep.py`
`815fb8e471d256854e2612f1d2199c4a75890728aa03e7c081947d09de4208b0`、
`baselines/n2p3net.py`
`727e75b830276ed6f3c74d022c87b72f7c057016427dababb21a9deda65bdb7f`、
`models/n2p3net.py`
`ef105539b52d609ededfc884b94b40e8c2d440433c4db3ee20a545181d377376`、
`adapters.py`
`0981fd90f39b2f8bffe4f8b14121eaf0df6aa62e19d7ec91d47d5990269c195f`、
`splitbn.py`
`1b0187ee5da5c5c95f25083140e9d882b6f527a28bd066ebdc6156e17921e3a0`、
`checkpoint.py`
`144314704802b133b054555a48265e0030a9c5cd40c9855258be4ba3d2b2ec94`。
这些哈希只定位审计证据，不是主线程的最终验收哈希。

### P0/P1/P2 字段现状

| 机制 | CLI/plan | model/trainer | checkpoint | evaluation/result | manifest | 集成前判定 |
|---|---|---|---|---|---|---|
| P0 Split-BN | 无字段 | 并发模块已有构造与 domain axis 门禁 | 并发 `/2` schema 可记录 normalization | zero-shot helper 可读取预声明 target buffer；target-adaptation 未接通 | 仅可通过 training contract 间接绑定 | 未闭合 |
| P1 source residual | 无三字段 | 并发模块已有 residual/drop/ratio 与内存 ledger | 集成中；起始 parser 会丢新字段 | 默认不 route source residual 是正确方向 | 无 durable `L(m=0)/L(m=1)` ledger | 未闭合 |
| P2 decision loss | 无字段 | 无 decision-group sampler/loss 接入 | 无 objective record | 只有训练后统一 aggregator | 无 P2 arm contract | 未实现 |

具体源码证据：

- `run_pretrain.py:94-151` 的 CLI 没有 P0/P1/P2；`run_pretrain.py:194-203`
  直接构造普通 core；`run_pretrain.py:389-466` 只记录 masked reconstruction contract
- `run_pretrain_supervised.py:313-369` 只有旧 feature-transfer/alignment 参数；
  `run_pretrain_supervised.py:478-490` 构造的配置没有 P1 三字段；
  `run_pretrain_supervised.py:544-559` 只有 `selection_domain` 存在才传 source-domain axis
- `run_pretrain_supervised.py:561-599` 的两次 `fit` 仍只有 flat epoch inputs；没有
  decision id、candidate membership、repetition 或 frozen vocabulary，所以 P2 不可能在
  该入口合法启用
- `run_pretrain_supervised.py:634-749` 会绑定 architecture、optimizer、objective 与
  training contract，但成功 checkpoint 没有持久化并发 P1 trainer 的
  `source_residual` counterfactual ledger
- `run_candidate_promotion_matrix.py:127-143`、`:585-678` 的 `TrainingConfig` 与严格
  parser 没有 P0/P1/P2 字段；`:902-960` 的 pretrain argv 也没有这些字段
- `src/research/contracts.py:350-365` 的 source-procedure allowlist 对 objective 只保留
  `name/qc_ptp_uv/input_statistics_scope`。若 P2 配置只写入 objective 而未更新投影，
  manifest 会把不同 P2 procedure 误认为同一 arm

### P2 的协议数学边界

对 direct 9-choice timeline，单个 epoch 只属于一个直接候选：

\[
c_i\in V_9,\qquad y_i=\mathbf 1[c_i=d^*].
\]

GTN 与 BrainSync 当前确实保留这种轴：`src/data/gtn_dataset.py:115-158` 和
`src/data/brainsync.py:95-109,167-170` 把 candidate、target、decision、repetition 写入
`ScheduledEventTimeline`；`src/data/epochs.py:732-738` 会把这些字段持久化到 cache，且
`src/data/epochs.py:336-347` 校验 timeline 存在 candidate set 时 binary label 等于
`candidate_id == target_candidate_id`。这是 direct P2 最小合法输入合同的现成基础。

当前 `CandidateTaskContract` 则在 `src/data/candidate_task.py:85-112` 明确只接受
`membership_kind="row_column"` 与 `target_representation="row_column_intersection"`。
对 6x6 row/column speller，设真 character 为 `(r*,q*)`，flash candidate 为
`c_i in {0,...,11}`，正确 binary 关系是：

\[
y_i=\mathbf 1[c_i=r^*\ \lor\ c_i=6+q^*],
\]

不是 `y_i=1[c_i=d*]`。一个 decision 每轮有两个 positive flash candidates。
当前 evaluator 也按两个独立轴工作：`src/transfer/candidate_decision.py:221-267` 分别对 row
和 column 做 unique top-1，再取交点。若要给 row/column 增加可微 objective，第一版应是
显式 factorized loss，例如：

\[
L_{rc}=\frac12\left[
-\log P_{row}(r^*)-\log P_{col}(q^*)
\right],
\]

而不是 9-way CE，也不是把 12 个 flash groups 当成 36 个 characters。

此外，`experiments/prepare_multidomain_source.py:84-88` 允许
`event_contract="binary_evidence"`，而 `src/data/domain.py:229-279` 会明确删除 timeline 的
candidate/target/repetition 字段。即便 DataFrame 里偶然还残留同名列，也不得绕过这个显式
投影重新解释为 P2 数据。混合 direct 与 row/column source 时必须保留每个 decision 的
typed task contract，并由 protocol-aware loss dispatch；当前 merged provenance 没有这个
合同。

## 🔗 调用链

```mermaid
flowchart LR
    accTitle: P0 P1 P2 execution chain
    accDescr: The formal experiment plan must preserve every mechanism field through training, checkpoint loading, evaluation, and manifest validation; red gates are currently missing or partial.

    plan[Experiment plan] --> dag[Promotion DAG argv]
    dag --> cli[Pretrain CLI]
    cli --> config[Typed configs]
    config --> trainer[Model and trainer]
    trainer --> checkpoint[Training contract and checkpoint]
    checkpoint --> target[Target runner]
    target --> outcome[Requested outcome ledger]
    outcome --> manifest[Manifest and analysis]

    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef risk fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef output fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    class plan,checkpoint,outcome,manifest output
    class dag,cli,config risk
    class trainer,target process
```

### 正式入口的确定性断链

`configs/feature_transfer_cloud_20260902.example.json:18-27,69-81` 设置多域 source、
`selection_domain=null`，同时把 alignment/private/orthogonality 设为非零。DAG 会原样不传
selection domain，而 supervised runner 又因此不传 `source_domain_ids`；当前 trainer 的
domain-aware loss 门禁会抛错。

同一模板把 target 指向 `brainsync_target_causal_v3.npz`，但
`run_candidate_promotion_matrix.py:983-1008` 固定启动
`run_candidate_cross_decision.py`。后者在 `run_candidate_cross_decision.py:261-269`
强制从 provenance 解析只支持 row/column 的 `CandidateTaskContract`。BrainSync cache 在
`src/data/brainsync.py:257-287` 声明的是 session/direct timeline，不含这个 row/column
contract。因此该模板不是尚未验证的“可能可跑”，而是当前接口下确定性不兼容。

> 2026-09-03 集成修正：promotion plan 已接入 P0/P1 配置轴。全局 `training`
> 记录 Split-BN target buffer 来源以及 source residual 的 drop probability/mode；
> `source_weighted` composition 则按 source arm 保存为 domain-keyed mapping，避免把
> BN buffer composition 与独立的 CE risk mass 绑成同一实验轴。parser 在两种 mapping
> 同时存在时立即核对 domain key，preflight 始终再以 cache 声明的 source-domain axis
> 复核，并在 P0/P1 cache 少于两个域时于任何 subprocess 前失败。字段缺省为 `null`，
> 不发 P0/P1 CLI 参数，仍走普通 BatchNorm/无 source residual 路径。
>
> 原云端示例已物理改写为带大写 `PLACEHOLDER` 路径与 row/column 身份名的
> **row/column development plan 模板**，固定交给 candidate runner；它不是 BrainSync
> 9-choice 产品实验计划，也不得替换路径后当作 BrainSync plan 使用。真正的 BrainSync
> 计划仍须走 direct-protocol runner，或等 protocol-dispatch 与下游 manifest 合同闭合后
> 再进入统一 promotion orchestrator。

### checkpoint 到 target evaluation

集成前 checkpoint 已将 architecture 与 `TrainingRunContract` 做语义 digest 一致性校验，
并把 checkpoint bytes SHA 与 training contract digest 放进 evaluation model origin：
`src/transfer/checkpoint.py:191-208`、`src/research/evaluation.py:19-27`。target runner 也会
复核 source snapshot：`run_candidate_cross_decision.py:387-412`、
`run_brainsync_cross_decision.py:263-278`。

并发 P0 观察中，direct Split-BN eval 缺 selector 会 raise，而 checkpoint-bound zero-shot
helper 从已签名 architecture contract 解析预声明 target buffer 并显式传给 core。这不是
pooled fallback。但是 `SubjectAdapter._features()` 当前没有 Split-BN eval selector；因此
linear/mlp/classifier/adapter 等非 zero-shot head 无法消费 Split-BN checkpoint。`full_fine`
路径只识别标准 `nn.BatchNorm*` 作为冻结对象，也没有合法 target domain index。P0 不能仅以
zero-shot checkpoint smoke 宣称 target transfer 已接通。

`target_calibration` buffer 的模块门禁会在未安装时 raise，这是正确的 fail-closed；目前没有
runner 从 calibration-only observations 构建、逐层安装、冻结并记录 buffer digest 的路径。

### requested denominator 与结果 ledger

| 路径 | participant denominator | decision x prefix ledger | coverage/tie/incomplete | fit failure | 判定 |
|---|---|---|---|---|---|
| candidate cross-decision | authority key；selection failure 的 participant endpoint 固定为 0 | 强制完整笛卡尔积 | typed 且完整 | 仅 `ExpectedSubjectError` | 主路径较完整 |
| BrainSync cross-decision | requested subject 保留；零 decision participant 进入 subject macro 0 | typed 且完整 | typed 且完整 | 仅 `ExpectedSubjectError` | 缺 decision plan digest/environment |
| legacy within-subject | requested group 有 operational denominator | 无 typed outcome ledger | coverage 有，tie/incomplete 未拆分 | 未捕获 | 不满足新主报告合同 |
| source training/P2 | 无 requested decision ledger | 无 | 无 | runner 失败即无 checkpoint/result | 未实现 |

`src/transfer/outcomes.py:277-323` 要求每个 requested decision 与每个 evidence level 都恰有
一个 outcome，并把 correct/incorrect/tie/abstain/incomplete/fit_failure 计入同一分母。
`run_candidate_cross_decision.py:795-893` 还把 participant selection failure 保留为
`planned_decisions=0`、endpoint `0.0`，没有伪造 decision；这是当前最完整的实现。

BrainSync 在 `run_brainsync_cross_decision.py:481-618` 也用 typed accounting，并同时输出
subject-macro 与 decision-micro operational hit。它没有 `DecisionPlanContract`；请求集合、
target truth 与 selection failure 没有独立 plan digest。`run_status` 也固定写 `completed`，
只能从内部 ledger 判断是否有失败。

legacy within-subject 在 `run_within_subject_transfer.py:627-703` 只保存 hit/eligible count；
`:700-917` 同时报 conditional 与 operational 曲线，但没有逐 decision typed status。
`SubjectAdapter` 的预期或非预期 fit error 都会中止整个 runner，因而不会生成一个把失败
留在 requested denominator 的结果文件。

### hit@R 语义

当前被审路径中没有 top-k/top-R selection 实现。`src/transfer/evaluation.py:84-110` 对每个
`r` 只选择 `repetition_index < r` 的 prefix，然后调用 unique argmax；缺 candidate 计 miss，
tie 的 winner 为 `None`。`src/transfer/candidate_decision.py:153-267` 对 row/column 每个
prefix 各做一次 top-1 后取交点。`run_candidate_cross_decision.py:699-743` 与 BrainSync
`:426-450` 都沿用这两个接口。

因此当前 `hit@R` 是 repetition prefix 后的 top-1 selection accuracy，不是“真类进入前 R
名”。legacy 字段 `hit_mean_by_repetition` 是 eligible 条件准确率，名字容易误读，但实现仍是
top-1；晋升应使用 requested-denominator operational 字段。

## ⚠️ 风险

### P0 阻塞：domain vocabulary 与 target head

- supervised runner 没有 Split-BN CLI，promotion plan 也没有对应字段
- trainer 的并发实现按 `sorted(set(source_domain_ids))` 生成整数 index，但 architecture
  只记录 positional `source_weights`，没有记录 domain name -> index 的有序映射。
  `source_weighted=[0.3,0.7]` 因而不能严格解释成哪两个 source 的 composition
- zero-shot target buffer 可以从 checkpoint 预声明读取，但 evaluation contract/result 没有
  重复记录 effective buffer source、estimand 与实际 buffer digest
- `target_calibration` 只有底层 install API，没有 calibration-only builder/runner
- non-zero-shot target heads 尚未向 Split-BN 传合法 target eval source；不得用 source-domain
  index 冒充 unseen target

### P1 阻塞：配置与 ledger 均未出 runner

- runner 没有 `source_domain_count/source_drop_p/source_drop_mode`，所以正式入口始终使用
  `source_domain_count=0`
- 即使手工构造 P1 baseline，成功 fit 的 `L(m=0)`、`L(m=1)`、ratio 分位数与 scales 只在
  trainer 内存状态；checkpoint 与 `TrainingRunContract` 不保存它们
- ratio 超限会 raise 并使训练 fail-closed，但 runner 不写 invalid run artifact，promotion
  只能看到一个失败 task，无法审计触发样本/forward 与 frozen denominator
- 并发 checkpoint parser 起始版本仍按旧字段手工重建配置；主线程已获知应复用严格 v2
  parser。最终集成必须以真实 P1 checkpoint round-trip 证明该缺口关闭
- 当前内存 ledger 保存 per-forward ratio summary 与 run max，未给出可验证的整次 run
  q50/q95/q99 汇总；更重要的是这些记录尚未持久化

### P2 阻塞：数据合同先于 loss 代码

- flat epoch sampler 不保证 batch 由多个完整 decisions 组成，也没有 `K*N>=128` 门禁
- 当前多域 source cache 可显式投影为 binary evidence；这种 cache 不再具有 P2 权限
- row/column 与 direct candidate 的 truth/membership 数学不同，必须用 typed protocol
  eligibility 分流
- 缺 decision 不应返回一个可参与平均的零 loss。对活动 source training，最简单的闭合是
  preflight 发现 incomplete planned decision 即整次 run invalid，并保存 ledger；不得缩小
  vocabulary 或从 loss denominator 删除
- P2 的 weight、temperature、prefix alpha、beta/tau arm、sampler contract 若只写普通日志，
  `training_procedure_record` 可能把它们投影掉，使不同训练程序在 manifest 中碰撞

### OOM 与 BatchNorm 门禁不闭合

并发中间快照的 `src/baselines/deep.py:805-856` 会在训练 OOM 后减半 batch 并从同一 seed
重启；成功后在
`last_runtime` 记录 effective batch 与 retry count。主动 memory scheduler 也可能在没有 OOM
异常时降低 requested batch。`run_pretrain_supervised.py:644-652` 保存 requested config 与
runtime telemetry，但 `src/research/contracts.py:350-365` 的 procedure projection 排除了
`selection_runtime/refit_runtime`。因此 matched cells 可以使用不同 effective batch、不同
optimizer step 数和不同 BN 更新轨迹，却仍被 manifest 当成同一 training procedure。

promotion plan 当前强制 `stage="development"`，这限制了结论强度，但没有解决 matched
development comparison 的语义偏移。confirmation 或严格 paired arm 应在 effective training
batch 改变时 invalid；探索性 development 至少要写 `scientific_deviation` 并从 matched
contrast 分开。

target `SubjectAdapter` 没有 OOM retry/typed OOM failure。candidate 与 BrainSync runner 只捕获
`ExpectedSubjectError`；CUDA OOM 或普通 `RuntimeError` 会让结果文件缺失。promotion journal
会记录 task failed，但后续 manifest 无法把该 participant/decision 作为 operational miss。

标准 BN 的 zero-shot/frozen head 使用 eval 模式，不更新 buffer；`full_fine +
adapt_batchnorm` 会在 calibration batches 更新 BN，且 runner 记录了布尔开关、seed 和 batch
size。但没有 before/after buffer digest、calibration provenance 或 test-order invariance gate。
Split-BN 自定义模块还必须被 SubjectAdapter 的 freeze 逻辑显式识别。

### target truth 与 same-selection calibration

当前 candidate/BrainSync 调用顺序没有把 suffix labels 传给 model fit 或 normalization：model
只接 calibration rows，suffix truth 在预测后进入 scorer。candidate evaluation contract 还记录
`calibration_truth_access="adapter_visible"`、`test_truth_access="scorer_only"`
（`run_candidate_cross_decision.py:921-953`）。这是良好结构，但不是 capability-level taint
证明。

仍有两条边界必须写清：

- `DecisionPlanContract` 是 runner 读取整份带 truth cache 后现场生成，再由后置 manifest
  检查跨 arm 一致；它不是 test 前外部签名的 plan
- BrainSync 的 `forced_switch`/`unseen_calibration_codes` 会使用 later target truth 定义 eligible
  set（`src/transfer/within_subject.py:248-286`）。这是预声明、truth-conditioned estimand，
  不是无条件 unknown-target 性能；BrainSync runner又没有 decision plan digest

GTN within-subject 非 zero-shot 模式已有强制 `--allow-oracle-same-selection-adaptation`，且
result/evaluation contract 标记 `oracle_label_same_selection_personalization_proxy` 与
`same_selection_oracle_proxy=true`（`run_within_subject_transfer.py:341-357,820-825`）。该门禁
有效；它不能被晋升为未知目标 confirmation。candidate 与 BrainSync cross-decision 使用 early
decisions 训练、later decisions 测试，方向正确，但仍需 frozen decision plan 与时间反例。

### digest 绑定矩阵

| 对象 | 当前绑定 | 缺口 |
|---|---|---|
| source archive | training/evaluation contract 记录 archive SHA；manifest 复核物理 archive | subprocess 从当前 `ROOT` 启动，未证明执行树等于 archive；source commit 与 bytes 对应关系未在运行时重验 |
| source cache | `TrainingRunContract.source_cache_sha256` + identity digest | 运行计划的 decoded cache record 只在 promotion journal；standalone runner 没有统一 bundle |
| target cache | `EvaluationRunContract.target_cache_sha256/target_identity_digest`；decision plan 再绑定 | within/BrainSync 没有统一 decision plan；BrainSync target truth plan 未单独 digest |
| checkpoint | result model origin 绑定 checkpoint bytes SHA + training contract digest；manifest 重算 | P0/P1 schema 集成后需重新 round-trip；旧 `/1` checkpoint 被 `/2` fail-closed 拒绝 |
| experiment plan | journal 记录 semantic plan SHA、resolved-input digest、DAG digest、dependency output hashes | result/evaluation contract 不含 promotion plan/DAG digest；standalone runner 无 plan binding |
| environment | candidate/within result 只有 Python/NumPy/Torch/device 字符串；training 有部分 runtime telemetry | 无 environment contract/digest、lockfile SHA、installed distribution digest、CUDA/driver/cuDNN 绑定；BrainSync result 连版本字符串也没有 |

因此答案不是“digests 已齐全”，而是“局部 provenance 强、全链 closure 未成立”。尤其不能用
source archive SHA 代替 executed-source identity，也不能用四个版本字符串代替环境复现合同。

## 🧪 建议反例测试

以下测试按阻塞优先级排序，重点验证字段不会在层间丢失，而不是重复模块 happy path。

1. **完整 round-trip 矩阵**：从 promotion plan 各启用一次 P0、P1、P2，断言 plan parser、
   DAG argv、runner args、typed config、model、trainer、TrainingRunContract、checkpoint loader、
   EvaluationRunContract、result 与 manifest 的 effective 值逐层相等。删除任意一层字段必须失败
2. **无 selection-domain 的 domain axis**：多域 source、`selection_domain=null`，单独启用
   P0 或 P1；断言 source ids 仍进入 trainer。当前 runner 应失败，此测试用于关闭确定性断链
3. **source-weighted 名称置换**：固定 named weight mapping 后交换 cache 中 domain row/order；
   结果必须保持语义或因 mapping mismatch fail-closed。裸 positional tuple 不得通过
4. **Split-BN target heads**：同一 checkpoint 分别运行 zero-shot、linear、adapter、
   full-fine-frozen；断言使用预声明 target buffer、source residual 永不 route、所有 BN buffer
   在 test 前后 byte-equal。当前非 zero-shot 路径应暴露 selector 缺失
5. **target-calibration buffer 污染攻击**：安装前 predict 必须失败；只用 calibration rows
   构建并冻结后可预测；加入、删除或重排 test rows不能改变 buffer digest
6. **P1 durable ledger**：真实 supervised runner 生成 checkpoint，重新加载后能取得 selection
   与 refit 的 `L(m=0)`, `L(m=1)`, delta、named domain scales、run q50/q95/q99/max；篡改
   任一字段或删除 ledger 必须拒绝
7. **P1 超限 run artifact**：构造单样本 `r_j>0.5`，断言 run status 为 invalid、没有可晋升
   checkpoint、failure ledger 保留 counterfactual `m=1` 与 requested run id，而不是只剩 subprocess
   return code
8. **P2 protocol eligibility**：row/column contract 传入 direct CE 必须拒绝；direct timeline
   满足 `y_i=1[c_i=d*]` 才通过；`binary_evidence` projection 即便 metadata 残留 candidate
   列也必须拒绝
9. **row/column factorized permutation**：同时置换 row、column 与 truth，factorized loss 和
   character hit 不变；把两个 positive flashes 错压成一个 target 时测试必须失败
10. **decision batch 完整性**：破坏一个 candidate、截断一个 repetition、让 `K*N<128`，
    sampler 必须产生 typed incomplete/invalid ledger，不能给零 loss 或缩小 vocabulary
11. **P2 procedure collision**：只改变 `lambda_candidate`、temperature、`alpha_r`、beta/tau
    或 sampler policy，两个 source-procedure digest 必须不同；否则 manifest 构建失败
12. **top-R 反例**：在 `R=2` 时让 truth 排第二，`hit@2` 必须为 0；到第三个 repetition
    truth 成为 unique top-1 后 `hit@3=1`。该测试直接阻止 rank-R 语义回流
13. **OOM matched-cell 攻击**：一个 arm 正常 batch，另一个 arm 首次 OOM 后减半成功；
    confirmation/matched contrast 必须 invalid，development 必须显式 deviation，不能只记录 telemetry
14. **target OOM denominator**：让一个 participant 的 SubjectAdapter OOM；结果仍须包含该
    participant 的全部 planned decision x prefix `fit_failure` outcomes，分析器把它们计 0
15. **test-truth permutation**：冻结 split/decision plan 后置换 test truth；training state、
    normalization、temperature、source selection 与 raw predictions digest 必须不变，只有 scorer
    outcome 改变。若 eligibility 因置换变化，必须被识别为另一个 truth-conditioned plan
16. **executed-source mismatch**：source archive 不变但修改当前 `ROOT` 的 runner byte；preflight
    必须在任何 subprocess 前拒绝
17. **environment mutation**：改变 lockfile、Torch/CUDA build 或 driver contract 后尝试 resume/
    manifest analysis，必须因 environment digest 不同而拒绝
18. **plan-result 绑定**：结果 bytes 不变但替换 promotion plan/DAG digest；journal、manifest 与
    analyzer 必须一致拒绝，不能从一组结果重新后置生成任意计划

## 🔍 未验证证据

### 本地活动数据与运行

- 工作区 `experiments/cache/` 只有物理隔离的 `legacy/pre_v3` cache；没有可用于本轮 P0/P1/P2
  的 active source/target cache
- `experiments/runs/row_column_p0_p1_development_placeholder` 与 `cloud_inputs/` 不存在；
  当前模板用大写 `PLACEHOLDER` 明示 row/column source snapshot/cache 与 target cache 尚待替换，
  它不是 BrainSync 产品计划
- `frozen/source_feature_transfer_c7bf860.source-only.tar.gz` 的物理 SHA-256 与 manifest 一致，
  但它冻结的是 `c7bf860`，不包含本轮并发 P0/P1/P2 改动。任何新实验都必须在集成验收后
  生成新的物理 archive/manifest，不能复用旧 digest
- 本审计没有 CUDA、真实 BrainSync multi-session/target-switch、Split-BN target-calibration
  或 P2 decision-group training 运行证据

### 历史结果包的证据边界

只读检查了 `tmp/formal_results.tgz`，其 SHA-256 为
`66654715dd74b33ae147aa86dbcbf18acc89cbe5044f62cbf0e6db79b5234bfb`。包内有 96 个
candidate result：每个都有 evaluation/decision-plan/candidate-task digest、source/cache/
checkpoint SHA、四项 environment 字符串与 requested accounting。共 23,040 个 outcome，实际
状态只有 `correct` 和 `incorrect`；tie/abstain/incomplete/fit_failure 都未在这批真实产物中
发生。因此它只能证明字段在成功路径出现，不能证明 failure path 在真实运行中闭合。

同一包内 `promotion.journal.json` 的最终状态是 `failed`：192 个 task completed，manifest
task failed，analysis pending；包内没有最终 `manifest.json` 或 `analysis.json`。journal 保存
plan semantic SHA、resolved-input digest 与 DAG digest，但失败 stderr 未随包提供，无法在本轮
从该包确定 manifest 失败根因。不得把此包描述为完整 confirmation bundle。

### 集成后必须追加的验收状态

主线程在 A/B/C 合并后至少追加：

- 最终 changed-file hashes 与 P0/P1/P2 配置 round-trip 表
- 上述阻塞项是 closed、open 还是 deliberately deferred
- focused pytest、全量 pytest、Ruff 与 `git diff --check` 的命令和结果
- 新 checkpoint schema 的真实 save/load/predict 证据
- 新 source freeze、cache、environment、plan 与 run bundle digest；若未实际运行，明确写
  “框架/contract evidence only”

在这些内容补齐前，本审计支持的结论仅是：已有统一 post-hoc candidate evaluator 与较强的
candidate requested-denominator contract；P0/P1/P2 的正式 runner、计划和证据链仍未闭合。

---

## 主线程集成验收（2026-09-03）

本节是上述“集成前”快照的后续记录，只替代 P0/P1/P2 框架与合同的当前判定；不把
synthetic/contract 测试写成跨域性能、BrainSync 产品或 confirmation 证据。验收基线为
`HEAD=b054b0b6b929758d32ba97344a22bc13f6de3748` 上的 dirty worktree。

### 已闭合的实现链

| 机制 | 已验收代码链 | 保持的 fail-closed 规则 |
|---|---|---|
| P0 Split-BN | supervised CLI -> cache-declared ordered domain axis -> `N2P3NetBaseline` -> temporal/spatial/MST 三处 SplitBN -> architecture/checkpoint -> target predict 与 `SubjectAdapter` 的预声明 buffer selector | 训练缺 `domain_indices`、eval 缺 buffer source、域轴/weights/checkpoint record 不一致均失败；被拒绝的 singleton-domain batch 在任何 running buffer 更新前失败；target eval 不更新 buffer |
| P1 drop-to-consensus | CLI `source-residual-drop-*` -> `FeatureTransferConfig` -> source-only residual route -> `m~Bernoulli(1-p)` -> counterfactual `m=0/m=1` ledger -> checkpoint/training-contract durable record | eval/target 永远走 consensus；ratio 用 feature 非 batch 维 L2、`eps_res=1e-6`、`r_max=0.5`、固定 `m=1` gate；超限写 typed invalid artifact；active P1 checkpoint 的 selection/refit ledger 任一无效即拒绝加载 |
| P2 direct-decision loss | direct timeline -> `DirectDecisionMetadata` -> complete-decision sampler -> `DecisionLevelLoss` -> `DeepBaseline.fit` -> checkpoint/training-contract record | vocabulary、coverage、membership 与 `y_i=1[c_i=d*]` 不成立即 invalid；每 batch 完整 decision、至少 128 rows；OOM 不允许 batch halving；显式配置即使 `lambda_candidate=0` 也必须带 metadata，不得静默回退 flat epoch batching；trial component 固定为 1 |

实现时发现并原位修正了四个会污染科学解释的缺口：

1. baseline architecture record 缺 `classifier_features`，导致严格 checkpoint 重建误拒绝；已由
   `N2P3Net.default_architecture_record()` 统一产生该字段。
2. `test_pretrain_supervised.py` 的 P2 fixture 把原有 supervised fixture 的 return 置于不可达
   路径；已恢复为两个独立 dataset builder。
3. P2 的 `lambda_candidate=0` matched control 可绕开 metadata/grouped-batch contract；已在
   `DeepBaseline.fit()` 的原有参数门禁处统一拒绝。
4. P1 checkpoint 原先只比较 ledger digest；攻击者同时改 payload 与 contract 后仍可能加载
   invalid residual run，或向无 P1 架构伪造 active ledger。loader 现要求 active
   selection/refit ledger 均为 `valid`、`run_max_ratio <= 0.5`，并与 architecture 的
   source-residual 参数空间一致。

### 仍然开放，且不得越界宣传

- `target_calibration` 目前只有 per-layer frozen install API。尚无 model-wide
  calibration-only collector、buffer digest/provenance 和 target runner 写入路径；因此未安装
  时必须失败，不能把该 arm 当作可运行的 target-calibration 实验。
- P2 的 standalone supervised runner 仅接受 direct-membership timeline。promotion matrix 的
  row/column runner 不能伪装为 direct 9-choice loss；在有 typed protocol dispatch 前，不把 P2
  写入该 row/column promotion plan。
- P0/P1 的 target result/manifest 尚未重复写入 effective buffer source、estimand 与 buffer
  digest；现有 checkpoint/predict/adapter contract 不等于完整 target evidence bundle。
- executed-source identity、lockfile/environment digest、target OOM 的 requested-denominator
  outcome 仍是上一节所列开放项。
- 当前没有 active real source/target cache、CUDA run、成人 BrainSync multi-session/target-switch
  或 matched fold/seed paired CI。所有本轮证据均为 framework/contract evidence；不支持
  “feature transfer 有效”“decision learning 有效”或“产品达到 90%”的结论。
- 本 dirty snapshot 尚未物理冻结；后续真正开跑前必须按新的 source-only archive/manifest
  固化实际执行树，不能复用旧 archive digest。

### 验收命令与结果

| 检查 | 结果 |
|---|---|
| `./.venv/Scripts/python.exe -m pytest --collect-only` | 774 tests collected |
| `./.venv/Scripts/python.exe -m pytest -q --disable-warnings` | `PYTEST_EXIT=0`；运行输出含 2 个 skip，无失败回溯 |
| P0/P1/P2 focused pytest | `tests/test_decision_loss.py tests/test_domain_adapters.py tests/test_splitbn.py tests/test_splitbn_subject_adapter.py tests/test_pretrain_supervised.py tests/test_transfer.py` 通过 |
| `./.venv/Scripts/ruff.exe check src tests experiments` | 通过 |
| `git diff --check` | 通过 |
| 外部方法学锚点 | Crossref 于 2026-09-03 核验 `10.1109/CVPR.2019.00753` 为 Domain-Specific Batch Normalization for Unsupervised Domain Adaptation；`10.1198/016214506000001437` 为 Strictly Proper Scoring Rules, Prediction, and Estimation |

### 最终核心文件哈希

```text
src/models/splitbn.py                         2DB495DCDC4698110B8468E9469B4010220B67E785253F0E9F77A2523773F1EF
src/models/n2p3net.py                         886D53058D3DEA1EC79CBBAE3103C9FF5071CCD38D09E700465B771A2AB6346F
src/models/adapters.py                         926DE0CFD89E01C314293ED04624EEBF33BEA7AE5445D533EFDA184CFF666E55
src/baselines/deep.py                          A6981C4DE8D78535F93B0D9679EFDB040D7AE688A86CCAFFCBD8E362B646015A
src/transfer/decision_loss.py                 1937B42D74D6D42AC7BAE73D6A402AF2307A8183C1ECBB5E387341AE1160DAD5
src/transfer/checkpoint.py                    82DAEE2D876D6A8C1CD3BF6C1BA7DB236C0AF9C2D84A965A9B94056C3FF33ED8
experiments/run_pretrain_supervised.py        EC9753D471ABBD7AE9FE8EB60C343DDAB4EA57E27EA20487E9B292D44BEA0ECB
experiments/run_candidate_promotion_matrix.py 0908AA247E172A28E0AD648B2795109AA8EFABD03A910C0371D4FB8B8CEDEB30
```
