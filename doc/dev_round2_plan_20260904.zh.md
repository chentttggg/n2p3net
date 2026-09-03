# development 第二轮方案（r2，2026-09-04）

> 授权依据：T 2026-09-04 指令"分析数据，得到进一步实验方案，然后自行开始云端
> 训练"——本轮参数选择由 T 授权自动决策，依据 `dev64_cuda_r1` 结果，选择逻辑
> 全部落档（§3），post-hoc 成分如实标注。
> 前置：`doc/cloud_run_plan_20260903.zh.md`（r1 方案与全部实测）；r1 plan
> digest `fc201a66…`；冻结树 `9f0b421e`。

## 0. r1 状态与收割（数据落地后填）

| 项 | 值 |
|---|---|
| 发射时间 | 2026-09-04 02:22（UTC+8），PID 70079，W=8 |
| 退出时间 / exit code | 03:06 / 0（`status: completed`，wall 44 min，389/389 任务，attempt 全 1） |
| journal 阶段计时 | pretrain 3 seed batch 并行 16.7 min；result 层 ≈27 min（W=8） |
| planned contrast Δ = adapter − zero_shot | hit@R=8：Δ = 0.0（结构性，见 §3.2）；trial AUC：Δ = +7.8e-4（dev-01） |
| participant bootstrap CI | （hit 全零退化，无解释力——见 §3.2/§3.3） |
| 绝对水平 zero_shot / adapter | hit@R=8 均 0.00717（64 人 × 3 seeds） |
| C1–C4 抽查 | C1 verified=true（commit 9f0b421e / archive `648789a9…`）✅ |

## 1. r2 的静态约束（不依赖 r1 结果，先钉死）

1. **代码冻结不变**：r2 继续跑 `9f0b421e` 冻结树，**零代码改动**。C2b / C6 /
   plan-parse fail-fast 等任何新代码都不进 r2——不重做冻结链，保持与 r1 同源
   可比。代码批次归入 C 系列（等 T 排期）。
2. **P0/P1 机制不可进 r2**（§3.2 实测裁决）：单域 cache 下 gate 以 `is not
   None` 触发，连 `drop_p=0` 都非法，唯一合法配置 = P0/P1 双 null。P0/P1 验证
   阻塞在**真多域 cache**（≥2 声明域）。域定义是研究设计决策，留给 T（§5 两
   选项利弊），本轮不擅自定。
3. **r2 合法自由度**（全部是 plan 参数，无代码面）：
   - `training.epochs`（E 终值，D3 真正的决策点）；
   - `training_replicates`（seeds 数，r1=3）；
   - `statistical_design.bootstrap_iterations`（r1=1000，dev 轮地板；
     10000 留给 evidence-grade L2 轮，§4.5 草案不变）；
   - `output_root`（`dev64_cuda_r2_e{E}`，命名带 E 便于消融对账）。
4. **exploratory→confirmatory 分层**：r1(E=2) 与 r2 均为 development 轮
   （exploratory）；基于 r1 结果选 E 属于 exploratory 设计决策（合法，落档）。
   confirmatory 主张只能来自 evidence-grade L2 轮（预注册 §4.5 全量实例化 +
   10000 bootstrap + Holm）。r2 的任何"显著"只作方向证据，不作结论。
5. **统计纪律沿用**：participant 配对、Holm、小 delta 对照重跑噪声底
   **6.3e-4**、主指标 operational hit@R（prefix top-1），AUC/BACC 仅支持指标。

## 2. 预声明决策树（在看到 r1 数据前写死，防 post-hoc 漂移）

记：A = zero_shot 主指标绝对水平，B = target_adapter，Δ = B − A（planned
contrast），chance = BI2014a v3 字符任务的随机水平（10 选 1 speller，0.1 量级，
以 result 字段实际口径为准）。

| 分支 | 条件 | r2 动作 |
|---|---|---|
| R1-充分性 | max(A, B) < 2×chance | E=2 严重不足 → r2 直接 E=8（上界 wall ≤ 2h @W=8），seeds 维持 3 |
| R2-充分无信号 | max(A,B) ≥ 2×chance 且 Δ 的 CI 含 0 且 \|Δ\| < max(噪声底×3, CI 半宽) | 加深不是第一杠杆 → r2 = 诊断轮：先做 per-subject Δ 分布 + AUC/BACC 方向一致性分析（只读，不发射），据分布形态决定 E 或 adapter 超参轮（超参改动 = 新预注册变更，单独列出） |
| R3-充分有信号 | max(A,B) ≥ 2×chance 且 Holm 后 p < 0.05 且 Δ 同向 | r2 = 剂量确认轮：E=4 或 8（按 §3 计时外推选 wall ≤ 1h 的最大 E）+ seeds 3→5 稳 CI |
| R4-异常/失败 | r1 任何非统计性失败 | 不发射 r2；沿"更有耐心找逻辑判定"源头修，修复 → 重跑 r1 同配置 |

分支 R2/R3 中若计时数据显示 pretrain batch 内串行是 wall 主导（非 result 层），
优先用 workers 提升 result 层并行即可，不碰 batch 内串行结构（改动 = 代码面，
冻结）。

**实际落点（2026-09-04）**：R4'（结构性无效，见 §3.3）——四分支判据均不可
计算（hit 全零）。R 修复属合同变更（D-10，需 T）；在授权范围内先发射 E=8
增量轮（§3.4），保持 AUC 维度信息流。

## 3. r1 判读与 r2 定稿（2026-09-04 03:15–04:00 收割后定稿）

### 3.1 r1 收割结果

| 项 | 值 |
|---|---|
| 退出 | 03:06，exit 0，`status: completed`，wall **44 min**（02:22–03:06） |
| 任务 | 389/389 completed，0 失败，全部 attempt=1 |
| pretrain wall | 16.7 min（3 seed batch 并行；hot cache ≈15.7 s/job，probe 的 33.9 s 含 Triton 冷启动） |
| **hit@R=8（主指标）** | zs = ta = **0.00717**（per-seed 逐一相同，配对差全 0） |
| binary AUC（trial 级） | zs 0.66406 vs ta 0.66484（dev-01；差异微小不翻转 argmax） |
| C1 | source_commit=9f0b421e、archive SHA `648789a9…` verified=true ✅ |

### 3.2 根因诊断：hit@R=8 与 BI2014a 数据结构本质失配（非 bug）

- decision_accounting：每被试 24 个 test decisions 中 **23 个 `incomplete`**
  （`insufficient_complete_test_repetitions`，`data_ineligible=23/24`）；
- cache metadata 的 `repetition_index` **完好**（12 候选 × rep 完整块）——不是索引
  bug；selection 切分（target pair 变化 + code 100/104 marker）也正确；
- **数据事实**：BI2014a 的字符块（selection）rep 数参差 1–8。64 人 test 域
  1416 个 selections 的分布：rep=1 占 31.9%、rep=2 23.8%、rep=3 17.3%、rep≥4
  仅 27.0%、rep≥8 仅 5.2%；
- `test_reps=8` 的 eligibility 要求完整 12×8 块 → 94.8% 的 test decisions 记
  incomplete → hit@R=8 结构性归零（`all_noncorrect_statuses_remain_in_the_
  denominator=true` 是正确的 fail-closed，不是事故）；
- **为何此前未暴露**：12 人 mini smoke 的"数值全 0 属预期（仅验结构）"掩盖了
  分母退化——**QC 盲区**：preflight QC 不含 per-selection rep 分布统计。
- timeline 层另一处真实缺陷（非本征阻塞）：`bi2014a_candidate.py` 用
  `observed_only_timeline`（`complete=False` 硬编码、repetition/candidate 字段
  落默认 −1），与 bnci2014_008 builder 的显式 timeline 不一致——下游若改从
  timeline 读 rep 会踩雷；顺带修复候选，随 R 决策批次走。

### 3.3 预声明决策树的落点

四分支都不命中（hit 全零使 R1/R2/R3 的判据不可计算）→ **R4'（结构性无效）**：
主指标在该数据+合同组合下无信息，但 trial AUC 维度完全有效（不受 R 影响）。

### 3.4 r2 定稿（T 授权范围内的参数决策，2026-09-04 04:00）

- **唯一变量：`training.epochs` 2 → 8**（一次一个变量；seeds/bootstrap/
  test_reps/arm 结构全部不变）；
- 理由：①hit@R 失配使 AUC 成为唯一有效观察维度，加深训练使该维度信息量
  最大（r1 AUC 0.664 vs 历史 64-fold 0.74–0.745，欠训练证据明确）；②E=8
  checkpoint 是未来 R 修正轮（r3）的直接前驱资产；③wall 预算 ≈ 80–95 min
  （pretrain 行边际 ×4 + result 层不变）；
- output_root = `/root/autodl-tmp/n2p3-runs/dev64_cuda_r2_e8`（云端绝对路径，
  --root 边界内）；
- plan digest：ca5b444f（r1）→ 见发射记录（statistical_design digest
  66423956 不变）。

### 3.5 待 T 拍板的新决策点（D-10，不阻塞 r2）

**BI2014a dev 轮的 hit@R 语义**：
- (a) **R_dev 降级**（合同变更）：development 轮允许 test_reps ∈ [2,8]（如
  R_dev=2：68.1% eligible、中位 16 decisions/被试），promotion gate R≥8
  不动；需改 `run_candidate_promotion_matrix.py` L937 的 plan 解析下限 +
  预注册变更记录；
- (b) **换数据**：hit@R=8 主张迁移到 rep 结构支持 R=8 的数据集（GTN 242 人
  的 rep 结构待查；BrainSync multi-session 本就按 R≥8 设计）；
- (c) **维持现状**：BI2014a 只做 AUC 维度 dev 监控，hit@R 主张全部留给
  (b) 数据。
另：QC 盲区补丁（preflight QC 增加 rep 分布统计）与 timeline 字段补齐
（§3.2 末条）随 (a)/(b) 批次落地。

## 4. r2 plan 实例化与发射清单

1. 本地生成 `tmp/l1_cloud/plan_dev64_cuda_r2.json`（模板 = r1 plan，仅改 §1.3
   自由度字段；digest 重算记录在案）；
2. 新预注册节（本文档 §3 定稿版）commit 进 git（pathspec 限本文档 + plan
   JSON + memory 日志，**不碰 T 的 staged 批次**），commit hash 即冻结锚；
3. scp 上云 → `run_candidate_promotion_matrix.py --dry-run --max-workers 1` 绿；
4. `nohup … --device cuda --max-workers 8 > dev64_r2.log 2>&1 &` 发射 + 远程
   守候（复用 r1 模式）；
5. 收割脚本 `harvest_dev64.py` 改路径复用。

## 5. 留给 T 的决策点（不阻塞 r2）

| # | 决策 | 选项 |
|---|---|---|
| D6 | 多域 cache 的"域"定义（P0/P1 验证前置） | (a) BI2014a session 伪域（3 session/被试，cache 重建中等，域语义 = session 间漂移，P0 Split-BN 直接有意义）；(b) BI2014a + GTN 跨数据集域（域语义 = 设备/被试群漂移，管线工作大，但直接服务最终产品线） |
| D7 | C2b / C6 / plan-parse fail-fast 代码批次排期 | 与 r2 并行或 r2 后 |
| D8 | ULP 容差裁决（r1 前遗留）：放宽 ≥2e-7 vs pin torch 2.11 | 待拍板 |
| D9 | 实例密码轮换（凭据进过聊天记录） | 待 T 操作 |
