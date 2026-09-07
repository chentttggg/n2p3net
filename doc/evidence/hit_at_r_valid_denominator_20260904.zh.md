# hit@R 口径修正与逐 R 曲线（2026-09-04）

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

> 2026-09-06路径更新：本文本地`tmp/l1_cloud/`结果现位于
> `experiments/results_history/tmp_20260906/retained/l1_cloud/`，脚本已物理归档。
> 见[清理与数据保留](workspace_cleanup_20260906.zh.md)；远端路径及历史数字不变。

> 起因：T 指出「逐 R 记录，R=2 到 R=8 都记下来；关键是 hit 的计算方法，
> 分母应该只有参与分子判定的那些 valid 被试」。
> 本文记录口径定义、两轮实测曲线、交叉验证、以及由此暴露的功效瓶颈。
> 代码冻结 `9f0b421e`；本轮**零训练**，全部从已落盘的 result 重算。

---

## 1. 两个 estimand 的定义

实验单元：被试（LOSO，每 partition 一个 target subject）。
一个被试持有若干 requested test selection，每个 selection 在证据深度
R=1..8 上各产生一个判定。

### 1.1 operational 口径（当前主指标，合同字段）

```
rate[s, R] = correct[s, R] / requested[s]          # ineligible 计为 miss
metric     = mean over ALL planned subjects s      # 含 rate=0 的被试
```

语义：部署视角——「我要求你判的每一题，你对了多少」。ineligible（数据不够）
按 fail-closed 记 0，这是诚实的行为。

### 1.2 valid-subject 口径（T 指定，本文采用）

```
valid[s, R]   = #{decisions at level R with status ∈ {correct, incorrect, tie}}
correct[s, R] = #{其中 status == correct}
rate[s, R]    = correct[s, R] / valid[s, R]        # 仅当 valid > 0
metric        = mean over subjects with valid[s, R] > 0   # 无判定者整体排除
```

语义：方法学视角——「在数据真正支持判定的地方，模型准不准」。
`abstain`（模型弃权）、`incomplete`（数据不足）、`fit_failure`（拟合失败）
三者都没有把候选放进 argmax，因此不进分母。

**两个口径都正当，用途不同，本文全部保留，不做删改。**

---

## 2. 关键发现：框架已有该口径，只是没被聚合为主指标

`experiments/run_candidate_cross_decision.py` L837 已经在每个 record 里写入：

```python
eligible_hit_rate_by_evidence = {
    counts.evidence_level: counts.correct / len(eligible_test)
    for counts in evaluated_accounting.counts_by_evidence
}
```

即 **per-subject × per-level，分母 = 该被试的 eligible decision 数**。
`analyze_candidate_cross_decision.py` L54 的 `SUPPORTED_METRIC` 选了
`requested_participant_operational_hit_rate`，于是 valid 口径被晾在一边。

⇒ **不需要改训练管线，也不需要从 outcomes 反推**——官方字段直接可用。

### 2.1 交叉验证（可信度证据）

对 `dev64_cuda_r2_e8/.../zero_shot/seed-20260901/dev-01.json`：

| R | 官方 `eligible_hit_rate_by_evidence` | 从 `decision_outcomes` 离线重算 |
|---|---|---|
| 1 | 1.0 | 1.0000 |
| 2 | 1.0 | 1.0000 |
| 3 | 1.0 | 1.0000 |
| 4 | 1.0 | 1.0000 |
| 5 | 1.0 | 1.0000 |
| 6 | 1.0 | 1.0000 |
| 7 | 1.0 | 1.0000 |
| 8 | 0.0 | 0.0000 |

**8/8 逐位一致。** 离线重算脚本（`tmp/l1_cloud/hit_by_r_valid.py`）与官方
字段互为独立实现，互相印证。

---

## 3. 逐 R 实测曲线

口径：valid-subject，subject-macro，三 seed 平均；CI = 被试聚类 bootstrap 5000 次
（取 seed-20260901 单次运行）。
**chance（36 符号 row_column）= 1/36 = 0.0278。**
复算脚本输出：`tmp/l1_cloud/hit_by_r_r{1_e2,2_e8}.json`。

### 3.1 r2（E=8，64 partitions × 3 seeds × 2 arms）

| R | valid 被试 | zero_shot | 95% CI | adapter | Δ(ad−zs) | p(sign-flip) | operational zs |
|---|---|---|---|---|---|---|---|
| 1 | 31 | 0.1204 | [0.052, 0.238] | 0.1204 | +0.00000 | 1.000 | 0.0152 |
| 2 | 31 | 0.1303 | [0.071, 0.291] | 0.1303 | +0.00000 | 1.000 | 0.0163 |
| 3 | 31 | 0.2202 | [0.081, 0.301] | 0.2202 | +0.00000 | 1.000 | 0.0212 |
| 4 | 31 | 0.2786 | [0.085, 0.327] | 0.2786 | +0.00000 | 1.000 | 0.0256 |
| 5 | 31 | 0.2602 | [0.109, 0.357] | 0.2637 | +0.00358 | 1.000 | 0.0305 |
| 6 | 31 | 0.3163 | [0.128, 0.390] | 0.3163 | +0.00000 | 1.000 | 0.0315 |
| 7 | 31 | 0.3364 | [0.138, 0.401] | 0.3364 | +0.00000 | 1.000 | 0.0348 |
| **8** | 31 | **0.3452** | [0.172, 0.439] | 0.3452 | +0.00000 | 1.000 | 0.0381 |

### 3.2 r1（E=2，同结构）

| R | valid 被试 | zero_shot | 95% CI | adapter | Δ(ad−zs) | operational zs |
|---|---|---|---|---|---|---|
| 1 | 31 | 0.0072 | [0.000, 0.016] | 0.0072 | +0.00000 | 0.0016 |
| 2 | 31 | 0.0471 | [0.000, 0.059] | 0.0471 | +0.00000 | 0.0060 |
| 3 | 31 | 0.0530 | [0.000, 0.070] | 0.0530 | +0.00000 | 0.0071 |
| 4 | 31 | 0.0668 | [0.022, 0.173] | 0.0721 | +0.00538 | 0.0114 |
| 5 | 31 | 0.0946 | [0.056, 0.255] | 0.0946 | +0.00000 | 0.0125 |
| 6 | 31 | 0.0884 | [0.057, 0.249] | 0.0884 | +0.00000 | 0.0136 |
| 7 | 31 | 0.1411 | [0.077, 0.277] | 0.1411 | +0.00000 | 0.0152 |
| 8 | 31 | 0.0897 | [0.016, 0.171] | 0.0897 | +0.00000 | 0.0125 |

### 3.3 剂量效应（E=2 → E=8）

| R | E=2 | E=8 | Δ |
|---|---|---|---|
| 1 | 0.0072 | 0.1204 | **+0.113** |
| 2 | 0.0471 | 0.1303 | +0.083 |
| 3 | 0.0530 | 0.2202 | +0.167 |
| 4 | 0.0668 | 0.2786 | +0.212 |
| 5 | 0.0946 | 0.2602 | +0.166 |
| 6 | 0.0884 | 0.3163 | +0.228 |
| 7 | 0.1411 | 0.3364 | +0.195 |
| 8 | 0.0897 | **0.3452** | **+0.256** |

---

## 4. 四条判读

### 4.1 模型在正常学习，此前的「0.0189」是分母污染

valid 口径下 R=8 hit = **0.3452 = 12.4 × chance**，且 R≥4 时 CI 下界
（0.085 / 0.109 / 0.128 / 0.138 / 0.172）全部高于 chance 0.0278。
operational 口径把同一批预测记成 0.0381——**稀释了 9.1 倍**。

### 4.2 曲线单调，证据累积物理正确

E=8：0.120 → 0.130 → 0.220 → 0.279 → 0.260 → 0.316 → 0.336 → 0.345。
R=5 的 −0.019 回落落在 CI 重叠区内，按噪声处理，不解读为效应。

### 4.3 训练不足确认，且未到平台

E=2 → E=8 全 R 段提升 +0.083 ~ +0.256，R=8 处放大 3.85 倍。
E=2 在 R=1~3 上 CI 下界为 0，即**与 chance 不可区分**——欠训练实锤。

### 4.4 adapter 效应在正确口径下确认为零

Δ(adapter − zero_shot) 在 R=1..8 上**全部 ≈ 0**，仅 R=4/5 出现 +0.005 / +0.004
的孤立正值（p=1.000）。两臂连「hit 完全相同」都做到了——不是「效应小到测不出」，
而是 36 选 1 的 argmax **几乎不翻转**。

这与早先在 AUC（秩统计量）上看到的 Δ≈0 汇合：
**target adapter（5 selections × 5 epochs @ lr 5e-4）在当前配置下无可测量增益。**
此前归因于「AUC 秩不敏感」的解释只对了一半——换成决策翻转维度后依然为零。

---

## 5. 由此暴露的功效瓶颈（本次最重要发现）

### 5.1 现象

`dev-01.json` 的 record：`requested_test_selections` 24 个，
`test_selections` **只有 1 个**（`subject_01:selection25`）。

汇总 seed-20260901 / zero_shot：
**valid decisions 73 / requested 613 = 11.9%**；31 个有效被试，
每被试平均仅 **2.35** 个判定（多数被试恰为 1 个）。

⇒ per-subject rate 退化成近乎 0/1 的伯努利，这就是 macro sd 高达 0.27~0.39、
CI 宽度 ±0.13 的直接原因。**31 次伯努利试验，SE ≈ 0.085。**

### 5.2 根因：split 层用 `test_repetitions` 而非 level R 做过滤（已修复）

> **修复记录（commit `e02e752`）**：第二道硬闸已删除；下文的旧实现片段只作
> 历史存档。现行 `candidate_calibration_decision_split` 对每个时间上不重叠的
> later selection 只做 `len(selected) == 0`、target 一致性检查，不再调用
> `_complete_repetition_block`；深 R 层由 `candidate_decision` 记 `incomplete`。

`src/transfer/candidate_within_subject.py`（`e02e752` 之前）L194-203：

```python
selected = rows[metadata.repetition_indices[rows] < test_repetitions]
if not _complete_repetition_block(selected, metadata, contract, count=test_repetitions):
    failed_test[str(selection)] = "insufficient_complete_test_repetitions"
```

过滤阈值是常量 `test_repetitions`（本轮 = 8），与证据深度 R 无关。
rep < 8 的 selection **连 R=1 的机会都被剥夺**——这解释了为什么 R=1 与 R=8
的 eligible 集合完全相同（都是那 73 个）。

这不是数据本性，是**过度收紧的过滤**：`candidate_decision_outcomes` 内部
已经在按 level 做 coverage 判定（fail-closed 正确），split 层的 8-rep 硬过滤
是冗余的第二道闸。

### 5.3 一个设计上的正面确认

逐 R 曲线的 selection 集合在各 R 上**固定不变**，仅证据深度变化 ⇒
跨 R 比较无构成偏差（无混杂）。这是干净的设计，应当保留。
代价是固定集合被钉在 rep≥8 的 11.9% 子集上。

### 5.4 降 test_reps 的预期功效增益（用已有普查数据估算）

BI2014a rep 分布（64 人 test 域 1416 selections，此前普查）：
rep=1 占 31.9%、rep=2 占 23.8%、rep=3 占 17.3%、**rep≥8 仅 5.2%**，
**rep≥2 占 68.1%**（中位每被试 16 个可判 selection）。

| 量 | test_reps=8（现状） | test_reps=2（预估） | 增益 |
|---|---|---|---|
| selection 通过率 | 11.9% | ≈ 68% | ×5.7 |
| 判定总数 | 73 | ≈ 420 | ×5.7 |
| 有效被试 | 31 | 接近 64 | ×2 |
| 每被试判定数 | 2.35（多为 1） | ≈ 13 | ×5.5 |
| 被试级 SE（粗估） | ≈ 0.085 | ≈ 0.045 | ÷1.9 |

⇒ **CI 宽度大致减半**，且 per-subject rate 不再是 0/1 退化值。

---

## 6. 下一步候选（按性价比排序）

### 候选 A：降 test_reps 重评（推荐优先）

- **做法**：`test_reps` 8 → 2（或 3），**复用 r2 的 E=8 checkpoint，只重跑 result 层**，
  pretrain 免重做，wall ≈ 30~40 min。
- **收益**：功效大幅提升（上表），且 R=2 本就是 T 指定的记录起点。
- **前置**：`experiments/run_candidate_promotion_matrix.py` L937 硬编码
  `test_reps >= MIN_PROMOTION_HIT_AT_R (= 8)` 会 fail-closed 拒绝 plan ⇒
  **需改合同常量或下放为 plan 参数（D-11，等 T 拍板，未改）**。
- **风险**：estimand 变化——浅 R 的 hit 绝对值会低于 R=8，不可与本文表 3.1 直接比较；
  需在报告里按 R 分层呈现。

### 候选 B：E 剂量轮（E=16 / 24）

- **依据**：E=2 → E=8 未到平台。
- **成本**：E=16 约 60~90 min（E=8 为 45 min，pretrain 近线性）。
- **风险**：在 11.9% 子集上，CI ±0.13，即使真涨 0.05 也可能测不出
  ⇒ **建议放在候选 A 之后**，否则大概率得到不可判读的结果。

### 候选 C：adapter 剂量诊断

- **依据**：Δ≈0 已在两个维度确认，怀疑是校准强度不足（5 selections / 5 epochs / lr 5e-4）。
- **成本**：小（只改 calibration 超参），但同样受功效限制。
- **价值**：负面结果也有价值——若强校准下仍无增益，可从研究主线中移除该分支。

---

## 7. 待 T 拍板

| # | 决策 | 说明 |
|---|---|---|
| **D-11** | `MIN_PROMOTION_HIT_AT_R = 8` 是否下放为 plan 参数 | 挡住候选 A 的唯一障碍。T 已申明 @8 非规范。promotion 轮下限保留多少需 T 指定 |
| D-13 | 主指标是否切换为 valid-subject 口径 | 建议**并存**：operational 记部署视角，valid 记方法学视角。若切换，`SUPPORTED_METRIC`（`analyze_candidate_cross_decision.py` L54）需改为可按 plan 选择 |
| D-14 | 候选 A/B/C 的排期 | 我的建议：A → B → C |

---

## 8. 本次产物

| 文件 | 内容 |
|---|---|
| `tmp/l1_cloud/hit_by_r_valid.py` | 离线重算脚本（valid 分母 + 逐 R + bootstrap CI + 配对 sign-flip），已与官方字段交叉验证 8/8 |
| `tmp/l1_cloud/hit_by_r_r1_e2.json` | r1 逐 R 完整结果（含 per-seed 明细） |
| `tmp/l1_cloud/hit_by_r_r2_e8.json` | r2 逐 R 完整结果（含 per-seed 明细） |
| `tmp/l1_cloud/r2_sample/dev-01.json` | 单 result 样本，用于结构核对与交叉验证 |

未改动任何 `src/` 或 `experiments/` 代码。

---

## 9. r3 预注册（发射前冻结，2026-09-04）

在 D-11 获批前，先发射**候选 B**（E 剂量轮）：它是唯一不触碰合同常量、
且与 r2 同口径可直接对比的实验。参数在发射前写死如下。

| 项 | 值 |
|---|---|
| 代码冻结 | `9f0b421e`（与 r1/r2 同一冻结树，零改动） |
| 唯一变量 | `training.epochs` 8 → **16** |
| 冻结项 | seeds = 20260901 / 20260902 / 20260903；64 partitions；2 arms（zero_shot / target_adapter）；test_reps = 8；bootstrap = 1000；cache / manifest / 通道 / 预处理全部不变 |
| plan digest | `8243c158`（r2 为 `a3161042`） |
| statistical_design digest | `fc201a66`（**未变**，与 r1/r2 同一统计合同） |
| output_root | `/root/autodl-tmp/n2p3-runs/dev64_cuda_r3_e16` |
| 预期 wall | 约 90 min（pretrain 近线性于 E；r2/E=8 为 45 min） |
| 主判读 | 与 §3.1 同口径比较 R=8 的 hit：若继续上升 ⇒ 训练仍是瓶颈；若平台 ⇒ 瓶颈转移到数据/功效（指向候选 A） |
| 不做的事 | 不在结果出来后回头改 epochs / seeds / test_reps；不因 E=16 结果改写 §3 的任何数字 |

**功效声明**：本轮仍在 11.9% 的 eligible 子集上（31 被试 × 平均 2.35 判定），
CI 宽度约 ±0.13。因此本轮**只能检测大效应**；若 E=8 与 E=16 的差小于约 0.10，
本轮无法给出判读，届时结论应记为「不可区分」而非「无差异」——这正是候选 A
需要优先的原因。
