# 范式 × 域迁移代码审核

日期：2026-09-04 | 触发：T 问「不同实验范式的数据集，域迁移代码怎么处理」
方法：只读审核（`src/data`、`src/transfer`、`src/train`、`src/models`、`experiments/`）

---

## 1. 一句话结论

框架在**训练层完全范式无关**（只吃 epoch 级二分类 `(N,C,T) + y∈{0,1}`），范式差异被全部推到
**数据契约层**和**决策损失层**，而这两层的范式支持是**错配的**：

> 数据契约只认 `row_column`；P2 决策损失只认 `direct`。
> 两者互斥 ⇒ **没有任何一个范式能同时启用 P0/P1/P2 全套机制**。

四种范式因此落进**三条互不兼容的代码路径**（见 §3）。

---

## 2. 四范式对照

| 范式 | 数据集 | 候选语义 | 数据契约 | 决策评估 | 域迁移 runner | 状态 |
|---|---|---|---|---|---|---|
| 字符 6×6 speller | BI2014a（64 人 cache）；2014b/2015a/2015b 待建 | `row_column`，12 候选（6 行 + 6 列），target = 行列交点 | ✅ `CandidateTaskContract` | `row_column_target()` | `run_candidate_cross_decision.py` | **唯一完整接入** |
| row/column 6×6 | BNCI2014-008（ALS 人群） | `row_column`，12 候选 | ✅ 契约已建 | 同上 | 同上 | 契约就绪，**cache 待核** |
| 9 选 1（数字） | BrainSync 成人 multi-session | `direct`，vocab = 1–9，平衡设计（每 block 每候选等量） | ❌ **无契约，绕开** | `candidate_decision_outcomes(vocab=range(1,10))` | `run_brainsync_cross_decision.py` | **旁路接入**，P2 不可用 |
| 猜数字 9 选 1 | GTN（242 人，Fz/Cz/Pz） | `direct`，`thought_number` 为 ground truth | ❌ **无契约** | `evaluate_candidate_selection`（老 LOSO） | ❌ **无** | **完全在域迁移框架外** |

---

## 3. 分层范式耦合矩阵

| 层 | 文件 | 范式耦合 | 说明 |
|---|---|---|---|
| **L1 数据契约** | `data/candidate_task.py:88` | 🔴 **硬门 row_column** | `membership_kind != "row_column"` 直接 raise；`grid_shape` 需 ≥2×2；`candidate_ids` 必须是「先行后列」规范范围；`target_representation` 必须 `row_column_intersection` |
| **L2 源域构造** | `experiments/prepare_multidomain_source.py:58` | 🟡 可选 | `--event-contract {preserve, binary_evidence}`；`binary_evidence` 调 `project_binary_evidence_source_view()` **剥掉候选元数据**，只留二分类证据视图 |
| **L3 训练** | `transfer/subject_adapter.py:306`、`train/runtime.py:395` | 🟢 **完全无关** | 只吃 `X(N,C,T) + y∈{0,1}`；`MatrixBatchSource` 仅额外携带 `row_domains`（域 id），与范式无关 |
| **L4 决策聚合** | `models/candidate_evidence.py:110`、`transfer/evaluation.py:27` | 🟢 **词汇表驱动** | `candidate_evidence_scores(logits, codes, vocabulary)` 与 `hit_at_repetition` 对任意候选集通用；row_column 传 12 个 code，9 选 1 传 1–9 |
| **L5 P2 决策损失** | `transfer/decision_loss.py:6-7, 422-439` | 🔴 **只认 direct，且显式拒 row_column** | 只要传进 `candidate_task_contract` 就抛 `DirectDecisionContractError`（reason 里带 `membership_kind:target_representation`）；准入 `source_event_contract ∈ {preserve, scheduled_event_direct_membership}` |

**关键错位**：L4 明明是通用的，L1 却把它锁死在 row_column；L5 需要的 direct 恰恰是 L1 拒绝的。

---

## 4. 三条路径详解

### 路径 A · 契约驱动（row_column 唯一通道）

```
prepare_bi2014a_candidate.py / prepare_bnci2014_008_candidate.py
  → EpochDataset(metadata: candidate_id, row_code, col_code,
                 target_row, target_col, selection_id, repetition_index)
  → prepare_multidomain_source.py      # 通道子集 + 统一 CAR（参考/蒙太奇规范化）+ 域标注
  → run_pretrain.py                     # 自监督 → trunk + 输入统计量 → ckpt /2
  → run_candidate_cross_decision.py     # target 适配 + 决策计划 + hit@R
```

- 契约实例：`bi2014a_candidate.py:62-77`（6×6、12 候选、`raw_target_label_is_target={"1":False,"2":True}`）
- flash 码来源：`bi2014a_candidate.py:52-61`（col17 = flash group，col18 = 二值标签；边界码 100/104 是硬 episode 边界）
- 决策计划：`run_candidate_cross_decision.py:312` 用 `row_column_target(row, col)` 冻结目标 —— **行列交集语义写死**
- 标签一致性：`run_candidate_cross_decision.py:333-342` 调 `validate_candidate_membership_metadata`，逐行核对 `y_i == 1[该行/列 == target]`

### 路径 B · 旁路 direct（BrainSync 9 选 1）

```
prepare_brainsync_cache.py
  → run_brainsync_cross_decision.py     # 自带 splitter，不走 CandidateTaskContract
      candidate_vocab = range(1, 10)    # L239, L435
  → candidate_decision_outcomes(...)    # L426 通用评估
```

- 平衡设计校验：`brainsync_contract.py:216-257`（每 block 每候选恰好 `repetitions` 次，每个 cycle 是完整候选排列）
- **没有引用 `decision_loss` / `ScheduledEventTimeline`** ⇒ P2 决策损失在 9 选 1 上**完全不可用**

### 路径 C · 老 LOSO（GTN 猜数字）

```
prepare_gtn_dataset.py          # 输出 cache 带 event_timeline，支持 full candidate chain
  → run_eeg_loso.py             # L526-528 encoded_candidate_selection(require_full_chain=True)
                                # L667 evaluate_candidate_selection
  → run_pretrain_supervised.py
```

- GTN 在**数据结构上**已是 direct 候选链（1–9 数字，`thought_number` 为 truth）
- 但只有**单集内 LOSO**，无任何跨域/多域构造 ⇒ 域迁移机制（P0/P1）在此路径上不存在

---

## 5. 训练流程

### 阶段 1 · 源域预训练（`run_pretrain.py`）

| 项 | 值 |
|---|---|
| 目标 | 自监督：masked 波形重建 + 频谱带预测 |
| 关键参数 | `--mask-fraction 0.5`、`--min-block 12`、`--max-block 32`、`--bands 1-4,4-8,8-13,13-30`（默认） |
| 优化 | AdamW，`lr 1e-3`、`weight-decay 1e-4`、`epochs 20`、`batch 256` |
| 泄漏防护 | `--holdout-subjects`：这些被试永不参与预训练 |
| 输入规范化 | 上游 `prepare_multidomain_source.py`：公共通道子集 → 统一 CAR（只处理加性参考项）→ `annotate_source_domain`；不等同于已完成 feature alignment。2026-09-04 修订后默认 `native_preserve`，CAR 为显式 `--reference-mode common_average` 且公共导 `<3` 时 fail-closed |
| 产出 | checkpoint `/2`：trunk 权重 + 输入统计量（mean/std）+ 训练契约 |

### 阶段 2 · target 适配（`transfer/subject_adapter.py:306-597`）

1. **输入校验**：`X` 必须 `(N, n_channels, n_times)` 且有限；`y` 必须恰好 `{0,1}`
2. **验证切分**（三选一，互斥）：
   - `repetition_indices` → `chronological_validation_split`（时间前缀）
   - `validation_mask`（+ 可选 `training_mask`）
   - `group_ids` → `group_disjoint_validation_split`
3. **输入统计量**（`input_statistics`）：`auto` / `source`（用 checkpoint）/ `target_prefix`（用 target 训练段）/ `shrinkage`（两者按 `target_stat_weight` 混合）
4. **可训练性**：`head_kind == "full_fine"` 时 trunk 可训练；其余全部冻结 trunk，并**缓存 eval-mode 特征**（`cache_features`，`L481-491`）以省掉每 step 的 trunk 前传
5. **损失**：`CrossEntropyLoss(weight=[1, pos_weight])`，`pos_weight = n_neg / n_pos`（`L457-463`）
6. **优化**：`AdamW(lr, weight_decay)`，逐 epoch 随机排列 batch
7. **早停**：`patience` + `early_stop_min_delta`，保存 `best_state` 并在结束时回滚
8. **可选 refit**：`refit_full_prefix` 用选中的 `best_epoch_count` 在全前缀上重训（仅 chronological 切分）
9. **校准**：验证集 logit 差存为 `calibration_logits_`

`head_kind` 五种：`zero_shot` / `linear` / `mlp16` / `classifier_fine` / `full_fine` / `adapter`
（`run_candidate_cross_decision.py:363` 的 choices 为 auto + 六种）

### 阶段 3 · 决策聚合

1. target 测试段逐 epoch 出 logit → 转 LLR
2. 按 decision group（selection）累积：同一候选的 LLR 用 `sum` / `mean` / `tempered_evidence` / `trim0.2` / `precision` 聚合（`candidate_evidence.py:189-219`）
3. `center_logits=True`（默认）先按组自身加权均值中心化，防止恒定偏置在候选计数不等时泄漏成 `c * n_d`
4. 逐 repetition prefix `r = 1..R` 取 argmax → **hit@R = prefix 后的 top-1**（不是 top-R ranking）
5. 缺失候选的处理：`abstain` 策略下整组判 miss（保 requested 分母），`exclude` 下该候选永不获胜
6. **trial AUC 不受 R 影响**，是 rep 深度不足时的唯一有效对比维度

---

## 6. 缺口与风险（需 T 裁决）

| # | 缺口 | 影响 | 建议 |
|---|---|---|---|
| **G1** | **`direct` 无数据层实现**（`candidate_task.py:88` 硬门） | GTN / BrainSync 两个 9 选 1 数据集**永远进不了路径 A** ⇒ P0 Split-BN、P1 drop-to-consensus、P2 决策损失在产品线语义（9 选 1）上**全部无法启用** | 要么给 `CandidateTaskContract` 加 `direct` 分支（C6 的范畴），要么接受「先在 row_column 上验证机制」 |
| **G2** | 同一套 hit@R 语义在**两条路径重复实现**（路径 A 自管 accounting，路径 B 走 `candidate_decision_outcomes`） | 分母语义/缺失候选策略可能静默分叉，跨范式数字不可直接比 | 收敛到单一聚合核心 |
| **G3** | **GTN 完全在框架外**，但它是唯一撑得起 hit@R=8 的数据集（88.4% 被试支持 R≥8） | 主指标（hit@R=8）与机制验证（域适应）目前在不同的数据上、用不同的代码跑 | 见 `bi_family_transfer_census_20260904.zh.md` §6b 路线 C（两条腿） |
| **G4** | BI 家族 rep 深度浅（64 人实测仅 **5.2%** 达 R≥8） | 路径 A 上 hit@R=8 结构性归零（已实测 zs=ta=0.00717），非 bug | `R_dev` 降级（合同变更，`load_plan` 硬门 `test_reps≥8`）/ 换数据 / AUC-only |
| **G5** | `binary_evidence` 投影剥掉候选元数据 | 源域走这条路 → P2 在源侧无候选身份可用 | 与 G1 一并考虑 |

---

## 7. 本次审核未覆盖

- BNCI2014-008 的 cache 是否已实际构建（只确认了契约与 builder 存在）；
- `run_pretrain.py` 内部损失实现细节（只核到 argparse 参数与模块职责）；
- 路径 B / C 的 `hit@R` 分母语义与路径 A 是否逐位一致（G2 的根因，需单独比对）；
- `decision_loss.py` 完整 direct 元数据管线（本次只核到准入与拒绝分支）。
