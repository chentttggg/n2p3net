# 审计分区 G2：src/transfer 未深读模块 + train/{device,factory} + research/{contracts,domain_folds,recording_quality}

- 审计对象（18 个文件，全部 tracked）：
  - `src/transfer/`：asmmd.py(349)、batch_cohort.py(117)、candidate_decision.py(325)、candidate_within_subject.py(322)、cohort.py(57)、evaluation.py(388)、heads.py(89)、losses.py(202)、mask_probe.py(273)、masking.py(84)、outcomes.py(327)、pretraining.py(134)、within_subject.py(726)
  - `src/train/`：device.py(65)、factory.py(170)
  - `src/research/`：contracts.py(616)、domain_folds.py(201)、recording_quality.py(328)
- 工作树：非 HEAD（429 项未提交；main 停在 f1a303c）；本分区文件均已入库（`git ls-files --error-unmatch` 通过），与 05 报告的 P0 untracked import 问题无关。
- 约束：只读；未运行训练/长任务/安装依赖；python 只读 `-c`（`PYTHONDONTWRITEBYTECODE=1`）。
- 级别：P0 = 正确性/泄漏/复现阻断；P1 = 显著科学风险；P2 = 可维护性。
- 与既有分区去重：03 已覆盖 transfer 核心（calibrated/recipes/checkpoint/…），04 覆盖 transfer 数学（exact_covariance/information_bound/…），02 已读 device.py/factory.py 的构造与设备策略，05 覆盖 research 的 execution/resume/evaluation。本报告只写这些文件的**新**发现，重复处仅交叉引用。

## 检查计划

- [x] A. asmmd.py：AS-MMD 数学、核矩阵、形状、边界、数值稳定
- [x] B. masking.py + heads.py：掩码语义、head 形状、标签泄漏
- [x] C. losses.py + pretraining.py：损失定义、边界、归一化、死代码
- [x] D. cohort.py + batch_cohort.py：队列构建/被试键/时间边界
- [x] E. within_subject.py + candidate_within_subject.py：within-subject 划分泄漏
- [x] F. candidate_decision.py + outcomes.py：候选聚合/标签/阈值
- [x] G. evaluation.py + mask_probe.py：评测指标与探针
- [x] H. train/device.py + factory.py（仅新发现）：设备策略、构造身份
- [x] I. research/contracts.py：契约字段与 checkpoint/执行身份
- [x] J. research/domain_folds.py + recording_quality.py：折划分与质量门
- [x] K. 文档一致性、死代码/重复实现/异常吞噬汇总

## 结论摘要

1. **[P1] AS-MMD 源批抽样不可复现（G2-01）**：`subject_adapter.py:1639` 用 `torch.randperm` 选源行，既不受 seed 控制，也不写账本；目标批次却有逐批哈希。`as_mmd_lambda0>0` 的 arm 无法从记录 seed 重放。
2. **[P1] 预训练谱项在零填充掩码上做 FFT（G2-05）**：常数偏置几乎完全不可见（1e-7 偏移 → 谱项 1.6e-12、波形 1e-7），且比较的是 `|FFT(g·x)|` 而非掩码段带功率；同误差下谱项约为波形项的 3.5 倍，默认权重 1:1 未声明这种尺度差。
3. **[P1] 质量报告 provenance 失真（G2-27）**：`recording_quality.py:266` 允许 callable 就地拟合，但 `:158-170,236` 无条件写 `fit_performed: False`；“冻结 QC 模型”的叙述因此可能错误。
4. **[P2] 时间/泄漏语义不一致（G2-13、G2-15、G2-18）**：candidate prefix/suffix 只按重复序号切分、无时间禁运；同一模块对“合法重复计划”有两套定义（任意起点 vs 必须 0 起）；`hit_at_repetition` 用全局起点归一化，组起点不同时实测 level-2 命中率 1.0 vs 正确 0.5。
5. **[P2] 身份/账本缺口（G2-06、G2-17、G2-21、G2-24）**：谱权不入 `state_dict`，恢复后损失尺度改变；6×6 与二选一各有一套打分核心，与 evaluation 的“统一核心”声明冲突；mask probe dry-run 的 `state_sha256_after` 是复制值；经典基线超参数不入 record。
6. **[P2] 边界/静默降级集群**：`selection_repetition_floor` 死参数（G2-14）、`resolve_subject_scope` 不去重（G2-10）、B0 网格硬编码（G2-11）、`study` 键覆盖（G2-12）、ABSTAIN 无生产者与 `int()` 截断（G2-20）、deep override 静默覆盖（G2-25）、procedure seed 剥离（G2-32）、domain_folds 死变量/单被试（G2-31）、报告分段静默丢弃（G2-30）、`not_evaluable` 与 headline 矛盾（G2-28）、generator 哈希范围过窄（G2-29）、device 清理失败静默（G2-26）、AS-MMD 样本下限/dtype（G2-02/G2-03）。
7. **未发现 P0**；正向核实 7 项（AS-MMD 公式、域折覆盖、B0 失败关闭、mask 比例恒 0.5、mask_probe 冻结门、zero-phase 失败关闭、输入合同入口断言）。

## 发现

### A. AS-MMD（asmmd.py + subject_adapter 接线）

#### G2-01 [P1] AS-MMD 的源批抽样不受 seed 控制，也不进入任何账本
- 证据：纯函数 `src/transfer/asmmd.py:249-341` 无状态；唯一训练调用点在 `src/transfer/subject_adapter.py:1639`：`rows = torch.randperm(source_count, device=self.device)[: self.config.batch_size]`。目标批次序由 `_training_batches` 用 `np.random.default_rng(SeedSequence([seed, epoch_index]))` 生成并记录逐批 `batch_rows_sha256`（`subject_adapter.py:1601-1621`、`:1595-1598`）；源行既不用该 RNG，也无任何记录字段（`subject_adapter.py:1366-1376` 的 AS-MMD 账本只有 lambda/warmup/counts）。
- 影响：`as_mmd_lambda0 > 0` 的 arm（默认 0.0，`subject_adapter.py:82`；已由 `doc/evidence/full_chain_audit_20260922.zh.md:118` 指出主链未接线）不可从记录的 seed 复现源分布的经验估计。任何前序 torch RNG 消耗（模型初始化、dropout、其他 `randperm`）都会改变 MMD 梯度；doc 声称“AS-MMD 初始拟合和 refit 共用源上下文与联合损失”（`doc/research_program.zh.md:303`），但没有 source-batch 复现契约。
- 建议：源行改用 `np.random.default_rng(SeedSequence([seed, epoch_index, step]))` 或独立 `torch.Generator` 采样；把 `source_rows_sha256`（或批序号）写进每步诊断，refit 复用同一序列。
- 验证：固定 seed 连续两次构造 adapter、训练 1 epoch，比较 `as_mmd_losses[*]["mmd_squared"]`；当前应不同。

#### G2-02 [P2] AS-MMD 的样本下限只在单批上检查，1 行目标批会抛裸 ValueError
- 证据：`asmmd.py:127-128,176-177` 要求每个域至少 2 个样本；`subject_adapter.py:1608-1615` 只合并**目标**侧的 singleton 尾批，源侧 `source_count` 为 1 或整体 `count==1` 时不会预处理。`_training_loss`（`subject_adapter.py:1623-1661`）未捕获该 `ValueError`，异常不会转成候选拟合路径的 `fit_failure` 记录。
- 影响：极小 calibration 集或单行冒烟用例会在训练中途抛裸错；与 `calibration_candidates.py` 的失败账本语义不一致。
- 建议：在 `_training_loss` 入口显式校验两侧 `>=2` 并抛出带上下文的错误（或降级为跳过 MMD 项并记录）。

#### G2-03 [P2] `_prepare_logits` 只接受 float32/float64，未兼容 bf16 autocast
- 证据：`asmmd.py:123-124`。当前 `subject_adapter` 全程 float32（`subject_adapter.py:284,592,782,942`），因此不触发；但设备策略已为 BF16 前向做准备（`train/device.py:43-45` 注释；12-performance 报告）。
- 影响：一旦 AS-MMD 被接入 autocast 训练，第一步就会 `ValueError`，而不是显式降精度。
- 建议：入口 `logits.float()` 并在诊断中记录原 dtype；或要求调用方声明禁用 autocast。

#### G2-04 [正向] MMD 估计量、中位数启发式与目标权重公式正确
- 证据：`asmmd.py:194-209` 对角剔除、`n(n-1)`/`m(m-1)` 分母、cross 项 `2/(nm)`；`145-153` 合并样本上三角中位数；带宽 detach；`46-48,94-97` 的 `alpha=min(1,e/W)` 与 clip 边界；`216` 记录 `negative_mmd`。与 `doc/evidence/domain_transfer_chain_primary_source_audit_20260904.zh.md:20-21` 的 Gretton 无偏式（可为负）一致。未发现符号/形状错误。

### B. masking / heads / losses / pretraining

#### G2-05 [P1] 谱重建项在零填充的掩码信号上做 FFT：常数偏置不可见，且“只监督掩码段”的语义被门控谱污染
- 证据：`pretraining.py:80-91` 生成 `masked/keep`，`decoder(features, keep)`，`reconstruction_loss(..., sample_mask=mask[None,None,:])`；`losses.py:136-152` 先 `x*mask`、`x_hat*mask` 再 `_band_magnitude_tensor`（`losses.py:71-84`：去均值 + Hann + rfft + 带内均值 + 带权 L1）。
- 数值（`tmp/audit/scratch/loss_probe.py`，B=16,C=6,T=176,sfreq=250,5 µV 噪声，mask_fraction=0.5）：`x_hat=x` → 两项均 0；只在 kept 位置加 1e-7 → 两项均 0（被 mask 乘掉，符合“只重建掩码段”）；只在 masked 位置加 1e-7 → waveform 1.00e-7、spectral 3.47e-7；对整段加常数 1e-7 → waveform 1.00e-7、spectral 1.6e-12。
- 影响：(1) 谱项对常数偏置/慢漂移几乎完全不可见（去均值 + `freqs>=1 Hz` 排除 DC），而这是波形 L1 会惩罚的主要误差模式；两项的监督域因此互补但未被声明。(2) 因为门控同时作用于 x 与 x̂，谱项比较的是 `|FFT(g·x)|` 与 `|FFT(g·x̂)|`，即“门控后信号”的幅度谱；内部矩形门的泄漏不会被 Hann 消除，这不是被掩码段的带功率。若意图是只监督掩码段，应在掩码段拼接/逐段加窗后再变换。
- 建议：谱项改为对掩码段拼接后的连续信号（或逐段）做 FFT；docstring 写明“DC/常数偏置不在谱项监督内”；把实测的 waveform:spectral 尺度比写入 pretraining 记录。
- 验证：本次 scratch 探针（保留 `tmp/audit/scratch/loss_probe.py`）。

#### G2-06 [P2] 预训练谱权不可恢复：不是 buffer、不在 checkpoint，config 里有死字段
- 证据：`pretraining.py:53` `self._band_weights` 普通属性；`63-71` 从一批数据估计；`132-134` 只读属性。`state_dict()` 不含它 → 恢复 checkpoint 后为 `None`，`losses.py:145-146` 退回均匀权重，损失尺度改变。`PretrainingConfig.seed` 与 `band_weight_estimation_samples`（`pretraining.py:25-26`）除 `validate` 外无消费者。`discard_decoder`（`126-130`）把 `self.decoder=None`，但已有 optimizer 仍持有其参数引用。
- 建议：`register_buffer` 或把权重写入 checkpoint 的 pretraining record；删除/接线两个死字段；discard 后重建 optimizer 或记录“参数仍在 optimizer 中”。

#### G2-07 [P2] `apply_time_mask` 返回的是 keep 掩码，名字/docstring 说的是 float_mask
- 证据：`masking.py:68-84` 返回 `(masked, keep)`（1=保留），docstring 称 “float_mask”；`pretraining.py:81` 把第二项命名 `keep` 传给 decoder，把 `mask` 传给 loss（`:90`）。两个 0/1 张量形状相同，误用不会有任何运行时错误。
- 建议：返回命名结构（`TimeMask(mask, observed)`）或改 docstring 并加“decoder 收到 observed 掩码”的断言/单测。

#### G2-08 [P2] `WaveDecoderHead` 只在 `T % st_pool == 0` 时与 trunk 时间基一致
- 证据：`heads.py:34` `AvgPool1d(st_pool, stride=st_pool)` → `floor(T/pool)`；`41-45` 要求特征时长 `floor(T/pool)`；`55-60` `F.interpolate(size=T)`。若 T 非整除，被 AvgPool 丢弃的尾部样本在 upsample 时被线性插值补回，mask 条件信息在该尾部缺失；`PretrainingTask`（`pretraining.py:43-52`）没有 `n_times % st_pool_size == 0` 断言。
- 建议：`ceil_mode=True` 并显式对齐，或加构造期断言；docstring 注明尾部样本不受 mask 条件。

#### G2-09 [P2] `make_temporal_mask` 的 1000 次尝试上限与“恰好 mask_fraction”契约
- 证据：`masking.py:45-64`；达到 target 前最多 1000 次随机块放置，超限后若 `mask.sum()>0` 静默返回（不校验等于 target）。本次 2000 次 × {T=40,64,96,176} 抽样全部恰好 0.5（`tmp/audit/scratch/mask_split_probe2.py`），说明现实中未触发；但代码路径仍允许“少于目标比例”的掩码，且不记录 attempts。
- 建议：循环结束断言 `mask.sum()==target`（或显式返回 `attempts` 并记录降级），保证 `mask_fraction` 是可复现契约。
### C. cohort / batch_cohort

#### G2-10 [P2] `resolve_subject_scope` 允许重复请求，与 manifest 读取器的去重规则相反
- 证据：`cohort.py:26-27` 读 manifest 时拒绝重复；`cohort.py:44-56` 对 `requested_subjects` 不去重也不排序。重复条目会同时进入 `requested` 与 `selected_usable`，下游按被试聚合会双计同一人（分母膨胀），且 `unknown` 检查仍通过。
- 建议：与 `read_subject_manifest` 一致地 `dict.fromkeys` 去重（保序）或直接拒绝重复；并把 `usable` 中不在 available 的条目显式记为 `unknown_usable`（当前在 `:41,56` 被静默忽略）。

#### G2-11 [P2] B0 cohort 的 batch/arm 网格硬编码在验证器里，不在已声明的 cohort 中
- 证据：`batch_cohort.py:90-91`：`expected = {(*key, batch, arm) for batch in (32,128,512) for arm in ("source","scratch")}`；`declare_batch_cohort` 的 record（`:53-59`）不含该网格，`digest` 不覆盖它。改动实验设计（如 64/256）不会改变 cohort digest，旧记录会被新验证器判为“结果不全”。
- 同一记录里 `K==0` 的 scope 在 `_cases`（`:18-19`）被跳过，但 `verify_batch_cohort_manifest` 的 `people` 集合仍包含这些被试（`:75-76`），语义未写入 schema。
- 建议：把 `batch_sizes`/`arms` 写进 declare 记录并由 `verify_batch_cohort_results` 读取；在 docstring/schema 中声明 `K==0` 的处理。

#### G2-12 [P2] `protocol_exclusions` 的 `study` 键可被条目自身覆盖
- 证据：`batch_cohort.py:43-45`：`excluded.append({"study": study, **item})`。若 manifest 的 exclusion 条目自带 `study`，dict 展开会用条目值覆盖外层 study，排除项被归到错误的 study。
- 建议：`{**item, "study": study}` 并拒绝 item 中的保留键；或对 exclusion 条目做 schema 校验。

### D. within_subject / candidate_within_subject

#### G2-13 [P2] 两套 “prefix/suffix” 划分的泄漏防护强度不同，却共享同名概念
- 证据：`within_subject.causal_prefix_suffix_split`（`513-726`）用 `evidence_available_times_s` 做全局时间禁运：`boundary_s=max(...)`（`621-626`），suffix 需满足 `onset+offset > boundary`（`631-646`）；zero-phase cache 通过把 `evidence_available` 设为整段录音结束而失败关闭（`data/preprocess.py:972-977`），`tests/transfer/test_time_split.py:136` 有对应反例。
- `candidate_within_subject.candidate_prefix_suffix_split`（`261-322`）只按 `repetition_indices` 切分（`287-291`），没有任何时间边界检查；`candidate_calibration_decision_split` 只有 epoch 证据边界（`206-213`），没有 `within_subject` 的标签可用性门 `calibration_available=max(session_end, target_confirmed)`（`within_subject.py:189-205`）。
- 影响：两条路径都自称 chronological。candidate 路径的正确性依赖“重复序号 = 采集顺序”以及 `assert_causal_p300_input_contract`（`candidate_within_subject.py:48`）；数据准备若重排/压缩重复序号，它不会拒绝。此外 `within_subject` 同时返回原序号 `repetition_indices`（`:716`）与重编号的 `suffix_repetition_indices`（`:651-652`），candidate 路径只返回重编号结果（`:294-310`），原计划序号丢失。
- 建议：candidate 路径增加与 `within_subject` 相同的时间边界（用 `_epoch_times` 的 available），或在 docstring 明确“仅靠 causal 合同”；统一后缀序号的记录方式（scheduled + normalized 两列）。

#### G2-14 [P2] `selection_repetition_floor` 只验证不使用（死参数）
- 证据：`candidate_within_subject.py:114,128-132` 计算并校验 `floor`，其后无任何引用；docstring `119-123` 承认“retained for API compatibility but no longer a hard eligibility gate”。
- 影响：调用者会以为存在每决策最少重复下限；实际 `test_repetitions` 是唯一控制，且结果里没有记录两者差异。
- 建议：删除参数，或把它作为记录字段 `declared_selection_repetition_floor` 返回并写进 split record。

#### G2-15 [P2] 同一模块里两套“合法重复计划”定义：任意起点 vs 必须从 0 开始
- 证据：`within_subject.py:451-459` 允许重复序号从任意非负值开始、只要连续；`within_subject.py:233-243` 的 `complete()` 要求每个候选的序号等于 `arange(k)`（必须从 0）。若 calibration cache 缺 rep 0，前者接受、后者判 `incomplete_calibration_decision`。
- 建议：抽出单一 `validate_repetition_schedule(allow_offset: bool)`；或在 `complete` 中显式要求 offset==0 并注明“calibration 计划从 0 开始”的合同来源。

#### G2-16 [P2] `chronological_time_validation_split` 的候选边界切片表达式脆弱
- 证据：`within_subject.py:381`：`target_epoch_starts[: min_positive_per_partition - 1 : -1]`。它依赖“从末尾反向、跳过前 mpp-1 个”的巧合语义；mpp=1 时 `[:0:-1]` 也“恰好”工作。真正的正确性由循环内的类别计数与 `max(available_at[train]) < min(epoch_starts[validation])`（`386-395`）保证。
- 建议：改写为 `reversed(target_epoch_starts[mpp-1:])` 并加注释/单测锁定候选集合，避免后续改动把边界搜索悄然缩小。

### E. candidate_decision / outcomes / evaluation

#### G2-17 [P2] 同一个 `DecisionOutcome` 账本有两套打分核心，且与“统一核心”的模块声明冲突
- 证据：`evaluation.py:8-10` 声明 “All candidate scoring goes through the unified `models.candidate_evidence.candidate_evidence_scores` core”。但 `candidate_decision.py:150-219` 自己用 numpy cumsum + 轴内 argmax + `isclose` 平局判定，不导入该核心；`evaluation.py:209-215,316-330,356-362` 才使用统一核心。
- 调用分工：`decision_outcomes_at_repetition` 用于 candidate/6×6（`experiments/run_candidate_cross_decision.py:1282`，行/列轴分解），`candidate_decision_outcomes` 用于 BrainSync 二选一（`experiments/run_brainsync_cross_decision.py:486`）。两者产出同类型 `DecisionOutcome`，但平局规则、聚合方式、轴分解不同；跨 study 汇总 `decision_accounting` 时没有字段区分打分核心。
- 建议：在 outcome/accounting 里记录 `scoring_core`（`axis_argmax_sum` vs `candidate_evidence_scores:mean`），或把 6×6 轴分解实现为统一核心的 group 参数；至少修正 `evaluation.py` 的模块 docstring。

#### G2-18 [P2] `hit_at_repetition` 的“原始计划序号”支持在组起点不同时错位；且生产代码无调用者
- 证据：`evaluation.py:52-63` 用**全局** `unique_repetitions[0]` 归一化。若组 A 后缀为 8,9、组 B 为 9,10，则归一化后 A={0,1}、B={1,2}，同一 level 的证据量不同。数值验证（`tmp/audit/scratch/mask_split_probe2.py`，g2 在 rep10 转为错误）：原始起点输入返回 `{1:1.0, 2:1.0, 3:0.5}`，重编号输入返回 `{1:1.0, 2:0.5}`——level 2 命中率被抬高。
- 该函数在 `src/`、`experiments/` 无调用者，仅 `tests/transfer/test_decision_evaluation.py` 引用；但模块 docstring（`4-6`）明确宣称支持原始序号，测试未覆盖跨组偏移。
- 建议：按组内 `repetition - group_min` 归一化（或显式只接受 normalized 序号并 raise），补一个“不同起点组”的反例测试；若不再使用则删除。

#### G2-19 [P2] `candidate_evidence_endpoints` 要求序号从 0 开始，与 `hit_at_repetition` 的兼容声明不一致
- 证据：`evaluation.py:323,355` 用 `repetitions[rows] < k`；传原始序号（8,9,…）时 level 1 的 `selected` 为空，`candidate_evidence_scores` 收到 0 行（`evaluation.py:316-330`），行为取决于该核心的空输入处理；`hit_at_repetition` 则先归一化（`:63`）。现有唯一调用者传 `suffix_repetition_indices`（`experiments/run_within_subject_transfer.py:646`，已是 0 基），所以当前不触发。
- 建议：入口断言 `repetitions.min()==0` 并抛带修复提示的错误，或共用同一个 `_normalized_repetitions`。

#### G2-20 [P2] `DecisionStatus.ABSTAIN` 无生产者；`int(count)` 静默截断浮点计数
- 证据：`outcomes.py:20` 定义 ABSTAIN，但构造 `DecisionOutcome` 的路径只产生 CORRECT/INCORRECT/TIE/INCOMPLETE/FIT_FAILURE（`candidate_decision.py:271-324`、`evaluation.py:196-232`）；`EvidenceOutcomeCounts.abstain` 恒 0。`outcomes.py:43-45` 的 `count=int(count)` 把 `1.9` 截为 1 且负数检查在截断后。
- 建议：删除 ABSTAIN 或实现真正的 abstain 策略；`_canonical_counts` 改用 `operator.index` / 拒绝非整数类型。
### F. mask_probe

#### G2-21 [P2] dry-run 的 `state_sha256_after` 是复制而非实测；`_state_digest` 对 bf16 会抛错
- 证据：`mask_probe.py:178` 在构造输出时直接写 `"state_sha256_after": before`，`dry_run` 分支 `186-188` 提前返回，从未重算；非 dry-run 才在 `:270-272` 实测并校验。`_state_digest`（`45-50`）用 `tensor.detach().cpu().numpy()`，bfloat16 张量没有 numpy 视图（`TypeError`）；当前 checkpoint 载入为 float32，不触发。
- 影响：dry-run 记录声称“前后状态相同”，实际未运行任何 forward，属于自声明；若下游按该字段比较会得到虚假的“冻结已验证”。
- 建议：dry-run 把该字段设为 `None` + `"state_check": "not_evaluated"`，或直接重算 digest（无需 forward）；`_state_digest` 对非可 numpy dtype 先 `.float()` 并记录。

#### G2-22 [正向] 非 dry-run 的冻结门是实测的
- 证据：`mask_probe.py:103` 计算 before；`200-204` 只挂 eval 期 hook；`270-272` 重算 after 并在不等时 `RuntimeError`。state_dict 含 running stats，因此 BN buffer 漂移也会被捕获。

### G. train/{device,factory}

#### G2-23 [P2] `build_binary_model` 本身不校验输入合同，门在调用者
- 证据：`factory.py:48-53` 只做 `dataset.validate(require_labels=True)` 与标签 {0,1}；`within_subject.py:98,532`、`candidate_within_subject.py:48` 都有 personal/causal 合同断言。两个现有调用者确实各自断言：`experiments/run_eeg_loso.py:501-502`、`experiments/run_n2p3_sensitivity.py:336-337`。因此这是“单点构造入口不设门”的结构风险，而不是现存泄漏。
- 建议：给 `build_binary_model` 增加显式 `contract` 参数并调用 `assert_p300_input_contract`，或在 docstring 写明“调用者负责合同”，并在新入口清单里检查。

#### G2-24 [P2] `describe_binary_model` 对经典基线不记录超参数
- 证据：`factory.py:156-157` 只返回 `{"name","class"}`；而工厂内硬编码了 `TemplateMatching(window_ms=(250,600))`（`:100`）、`XdawnRiemann(nfilter=min(4,n_channels))`（`:102`）。修改窗口/滤波器数不会改变任何 record/model digest，arm 身份无法从结果账本复原。
- 建议：经典基线也返回 `hyperparameters` 子记录（窗口、nfilter、正则等），或把硬编码参数提升为 dataclass 并统一 `asdict`。

#### G2-25 [P2] `deep_config_overrides` 可静默覆盖显式参数
- 证据：`factory.py:64-71` 先放 `epochs/batch_size/seed/val_group_frac`，再 `values.update(overrides)`。调用者同时传 `seed=7` 与 overrides `{"seed": 0}` 时生效 0；`describe_binary_model` 记录 `asdict(config)`（`:162`）所以账本记的是实际值，但按函数参数写 manifest 的调用者会不一致。
- 建议：拒绝与显式参数冲突的 override 键（`set(overrides) & {"epochs","batch_size","seed","val_group_frac"}` → raise），或明确优先级并写入 docstring。

#### G2-26 [P2] `device.get_device` 返回无索引 cuda；`release_device_memory` 吞掉清理失败
- 证据：`train/device.py:17-21` 返回 `torch.device("cuda")`（当前设备），多卡机不按可见显存选择（02 报告已覆盖设备/确定性策略，这里只记“无设备索引/无记录”这一新点）。`:63-65` `except (AttributeError, RuntimeError): pass` 会隐藏 `ipc_collect`/`empty_cache` 失败且无日志；原始训练异常不会被覆盖（`48-54` 的 docstring 目的），但清理结果不可见。
- 建议：`get_device` 支持显式索引；清理失败记录到日志或返回 bool，由调用者决定是否忽略。

### H. research/{contracts,domain_folds,recording_quality}

#### G2-27 [P1] 质量报告无条件声明 `fit_performed: False`，但可调用 `epoch_quality()` 可能就地拟合
- 证据：`recording_quality.py:266`：`fitted = epoch_quality() if callable(epoch_quality) else epoch_quality`；`158-170` 把 `qc_record` 固定为 `{"status":"applied","fit_performed":False,...}`；顶层 `:236` 也写 `"fit_performed": False`。`transfer.quality` 本身已有 `fit_provenance` 字段（03 报告覆盖），但此报告不校验 callable 的 provenance。
- 影响：若调用者传入 `lambda: fit_epoch_quality(dataset)`（接口同时接受冻结对象与 callable），report.json 会声明“未拟合”，而门限实际来自本次记录；报告同时被 `:290,292,304-306` 的“冻结 QC 模型”叙述引用，provenance 失真。
- 建议：只接受已拟合对象（禁止 callable），或 callable 分支写 `fit_performed=True` 并透传 `fit_provenance`/`fit_scope`，在 `status`/`headline` 中标注“就地拟合”。
- 验证：用一个返回带 provenance 对象的 callable 生成报告，检查 `report["epoch_quality"]["fit_performed"]`（当前恒 False）。

#### G2-28 [P2] `except Exception` 把评估失败转成正常返回；`BrainSyncNoUsableEvidenceError` 分支的 status 与 headline 矛盾
- 证据：`recording_quality.py:317-321` 任意异常 → `status="not_evaluable"` + 写 report.json 后正常返回（`error` 字段有详情）；`:309-316` 专门分支只改 `headline`，未改 `status`（仍是 `:235` 的 `not_evaluable`），于是 headline 说“连续QC后没有可用epoch”、status 说“未能完成质量评估”。
- 影响：只检查文件存在/返回非 None 的调用者会把“未评估”当“已评估”；专用分支的空证据结论与状态字段不一致，归档检索会误分类。
- 建议：返回显式结果对象或要求检查 `status`；专用分支设 `status="no_usable_evidence"`（与 `attention` 区分）。

#### G2-29 [P2] `generator_sha256` 只哈希本文件，不覆盖实际计算链
- 证据：`recording_quality.py:239`：`"generator_sha256": file_sha256(Path(__file__))`；数值来自 `data.preprocess`、`data.qc_features`、`transfer.quality`（`:20-26`），`:238` 只记录 numpy/scipy/mne/matplotlib 版本。修改 `compute_epoch_qc_features` 或 `transform_epoch_quality` 不会改变 generator 哈希。
- 建议：复用 `research.evaluation` 的 executed-snapshot 摘要（05 报告已指出各入口快照校验不一致），或至少把 `data/preprocess.py`、`data/qc_features.py`、`transfer/quality.py` 三个文件哈希入账。

#### G2-30 [P2] 分段边界越界被静默丢弃，PSD 权重随之改变而不留记录
- 证据：`recording_quality.py:58-59`：`cuts=sorted({0, acquired, *round(t*rate) ...})`，`segments=[(a,b) for a,b in zip(cuts[:-1],cuts[1:]) if 0 <= a < b <= acquired]`。非单调/负值/超过 acquired 的 `acquisition_boundaries` 会被过滤，不报错；PSD 累计按 `windows` 加权（`:98-104`），因此被丢弃的边界会改变频谱估计，而报告不记录丢弃列表。
- 建议：越界边界 raise（BIDS 校验已拒绝则加注释），或记录 `dropped_boundaries` 与原因。

#### G2-31 [P2] `domain_folds` 有死变量；单被试 domain 使 ceiling 构造整体失败
- 证据：`domain_folds.py:58,66` `pairs=np.column_stack(...)` 后 `del pairs`，从未使用。`:138-141` 要求每个 domain 至少 2 名被试，否则 `raise`；数据池含单被试 study 时无法得到该 study 的 ceiling fold，也没有 excluded 记录。
- 建议：删除死变量；为 `domain_ceiling_folds` 增加 `allow_singleton_domains` 或返回 `(folds, skipped)`。

#### G2-32 [P2] `training_procedure_record` 的 seed 剥离与 allowlist 投影组合可能把新控制项静默排除
- 证据：`contracts.py:426-435` 递归删除任意层级的 `seed`/`random_seed`；`:456-472` 用 `_SOURCE_PROCEDURE_FIELDS` 投影；`:419-423` 在“没有任何 allowlist 键命中”时整段保留、命中≥1 时只保留命中键。docstring（`:447-453`）称“新控制必须显式注册”，但 seed 类键无论是否注册都会被剥离，且两种未知节的比较语义相反（整段 vs 子集）——共享 `lr` 但 `momentum` 不同的两个新优化器会被判为同一 procedure。
- 建议：投影时始终保留 scheme/kind 判别键；未知节整段保留并记录 `projection="whole_unknown"`；seed 剥离限定在已知随机键路径（如 `optimizer.seed`）而不是任意深度。
## 正向核实（本分区未发现 P0）

1. **AS-MMD 数学**：无偏 MMD² 的三项分母、对角剔除、cross 项 `2/(nm)`、中位数启发式（合并样本上三角）、`alpha` 与 clip 边界均与 Gretton 2012 / Chen & Delorme 2025 式 (1)-(4)(7)(9) 一致（`asmmd.py:46-48,94-97,145-153,194-209`）；`negative_mmd` 已入诊断，与 `doc/evidence/domain_transfer_chain_primary_source_audit_20260904.zh.md:20` 一致。
2. **域折覆盖**：LODO 与 ceiling 的 `_validate_domain_coverage`（`domain_folds.py:168-200`）保证 train/test 不相交、每个被选行恰好被测试一次、LODO 训练侧不含留出域、ceiling 训练侧不含留出被试；实测与静态检查一致。
3. **B0 cohort 失败关闭**：`batch_cohort.validate_batch_cohort` 重算语义哈希（`:62-68`），`verify_batch_cohort_manifest` 对实际 K/内层行/资格做全等比较（`:79-83`），结果验证要求完整网格（`:86-105`）；未发现“缺结果仍通过”的路径。
4. **mask 比例**：2000 次 × {T=40,64,96,176} 抽样中掩码比例恒为 0.5（`tmp/audit/scratch/mask_split_probe2.py`），说明 1000 次尝试上限在现实形状下不触发；G2-09 仅记录残余契约风险。
5. **mask_probe 冻结门**：非 dry-run 路径实测 before/after `state_dict` 摘要（含 BN running stats），不一致即 `RuntimeError`（`mask_probe.py:270-272`）。
6. **zero-phase 时间边界失败关闭**：`data/preprocess.py:972-977` 把 zero-phase cache 的 `evidence_available` 推到整段录音结束，`causal_prefix_suffix_split` 因此找不到合法后缀并整体报错（`within_subject.py:695-700`；`tests/transfer/test_time_split.py:136-152`）。
7. **输入合同**：`within_subject`/`candidate_within_subject` 分别在入口断言 BrainSync 双向 / 单被试 causal 合同（`within_subject.py:98,532`、`candidate_within_subject.py:48`）。

## 与既有分区的去重说明

- `device.py` 的 TF32/cudnn.benchmark、无确定性开关已由 02 报告 F-05 覆盖；本报告只补“无设备索引 / 清理失败静默”。
- `checkpoint.py`、`calibrated.py`、`quality.py`、`target_experiments.py` 的发现由 03 覆盖；本报告只在其调用方（`subject_adapter.py:1639`、`recording_quality.py:266`）记录接线缺口。
- `TrainingRunContract` 缺 weights digest 由 03 覆盖；本报告 G2-32 只补 procedure 投影的 seed/allowlist 交互。
- `research/evaluation.py` 的执行快照问题由 05 覆盖；本报告 G2-29 只补质量报告侧的 generator 哈希范围。

## 未覆盖 / 存疑

- 未读取真实 BrainSync/candidate cache，未执行 `calibration_decision_split` 或 `candidate_calibration_decision_split` 的端到端；时序结论来自静态控制流与 `test_time_split.py` 的既有断言。
- `candidate_evidence_scores` 的空输入行为未执行验证（属 models/candidate_evidence，02 范围），因此 G2-19 只断言“行为未定义/不一致”，不断言必然抛错。
- `contracts.py` 只精读与执行身份相关的 `_canonical_value/canonical_json_bytes`、procedure 投影、dataclass 摘要；其余字段校验未逐行。
- 未运行 `ruff check`（本分区只读报告不改代码）；未运行训练/长任务；未安装依赖。

## 复现与证据文件

- 数值探针：`tmp/audit/scratch/loss_probe.py`（G2-05）、`tmp/audit/scratch/mask_split_probe2.py`（G2-09、G2-18）。
- 运行方式：`$env:PYTHONDONTWRITEBYTECODE='1'; .venv\Scripts\python.exe <script>`，工作目录 `D:\BCI\n2p3-net`。
- 证据核对：所有 `文件:行号` 均对应当前工作树（18 个目标文件全部 `git ls-files --error-unmatch` 通过）。
