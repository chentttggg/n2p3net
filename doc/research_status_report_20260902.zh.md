# N2P3-Net 科研状态汇报

_更新至 2026-09-02；BNCI-target 多源正式矩阵（295942e）云端结果对比分析与下一轮方向_

---

## 结论摘要

- 295942e 正式云端运行在 causal-v3 CAR5 合同下完成 BNCI2014-008 目标方向的
  多源矩阵：`bnci_only` / `joint_natural` / `joint_bnci80_epoch` /
  `joint_bnci80_participant` 四臂 x 3 seeds x 8 名 ALS 被试，96/96 checkpoint
  与 result 全部完成，覆盖 100%（无 tie/abstain/failure）。
- 主指标 hit@8（row/column pair，chance=1/36）：`bnci_only 0.3417` >
  `joint_bnci80_epoch 0.2736` > `joint_bnci80_participant 0.2667` >
  `joint_natural 0.1222`。三个 seed 的臂序完全一致。
- 预注册对照（subject-cluster sign-flip，n=8）：J0-B0 hit@8 `-21.94 pp`
  （CI `[-37.50,-6.25]`，Holm `p=0.110`）；J1-J0 `+15.14 pp`
  （CI `[+4.17,+27.36]`，Holm `p=0.151`）；J2-J1 `-0.69 pp`
  （CI `[-1.94,+0.97]`）。hit@8 在 n=8 下欠功效；支持性 AUC 上
  J0-B0 `-9.62 pp`（CI `[-14.12,-5.48]`，`p=0.008`）、J1-J0 `+7.95 pp`
  （CI `[+4.78,+11.52]`，`p=0.008`），且全部 8 个 evidence level 的
  J0-B0 均为负、J1-J0 均为正。
- 两方向合成裁决：朴素联合源负迁移在 BI-target（causal-v2 归档）与
  BNCI-target（本 v3 运行）均成立；域梯度质量 80/20 校正恢复大部分损失但
  两个方向都未超过单域基线；epoch/participant 统计单元无差异（J2≈J1）。
  预注册门禁"J2 仍低于 B0 才进入梯度冲突/stem"已触发。
- 云端 manifest 步骤两次失败（returncode 1）。根因已定位并修复
  （commit `ab6c8e7`）：`training_procedure_record` 曾整体投影
  optimizer/validation/objective，DeepConfig 派生的 selection/refit config
  内嵌 seed、且 per-partition 实测拟合结果（selected epoch、refit 长度、
  label counts、priors、risk、runtime）随 holdout 变化，拒绝本应均匀的臂。
  现为显式 allowlist 投影 + 全层 seed 剥离；checkpoint 生成改为按
  (arm, seed) 批处理（单次 cache 装载）并以 batch sentinel 溯源。
- 证据链尚未闭合：manifest/analysis 待用修复后代码在下载的 96 个
  checkpoint 上重建并物理冻结。本文数值为本地独立复算，属"探索性复算"
  层级；闭合后升级为归档证据。

## 实测结果

运行身份：plan `candidate_promotion_bnci_bi5_295942e_formal.json`
（sha256 `675bfbcf…`，执行前冻结），DAG digest `d3a298ab…`，
source snapshot `523fdff7…`，target cache `c1c5742a…`（causal-v3 CAR5，
`CZ,P3,PZ,P4,OZ`），`full_unfold + K35`、100 epoch 上限、batch 512、
precision auto、QC 关闭（四臂一致）。目标协议：前 5 个已知 character
decisions 为校准（zero_shot 未使用），后续 30 个未知 decisions 按
evidence level 1..8 记分，estimand
`known_early_decisions_to_later_unknown_decisions`，split_axis
`decision_time`。环境：Python 3.12.3 / torch 2.8.0+cu128 / CUDA。

联合源 cache 为 6 域混合（BI2013a/2014a/2014b/2015a/2015b + BNCI）；
J1/J2 给每 BI 域 4%（合计 20%）、BNCI 80%；J0 为自然 class-weighted
epoch mass；所有联合臂 `--selection-domain BNCI`（inner validation 仅用
BNCI group-disjoint 被试，v3 显式修复了归档 causal-v2 发现的验证域漂移）。

### 主表（subject-macro，3-seed 均值）

| Arm | AUC | hit@1 | hit@2 | hit@8 |
|---|---:|---:|---:|---:|
| B0 `bnci_only` | **0.6641** | **0.1125** | 0.1986 | **0.3417** |
| J0 `joint_natural` | 0.5679 | 0.0347 | 0.0486 | 0.1222 |
| J1 `joint_bnci80_epoch` | 0.6474 | 0.0750 | 0.1583 | 0.2736 |
| J2 `joint_bnci80_participant` | 0.6496 | 0.0806 | 0.1681 | 0.2667 |

chance=1/36≈0.0278。B0 hit@8 为 12.3x chance；B0 hit@2 `0.1986` 与
归档 BI-target causal-v2 zero-shot hit@2 `0.1941`（同为 1/36 任务）
量级一致。

### 预注册对照（hit@8 与 AUC，subject-cluster，seed-mean 后配对）

| Contrast | hit@8 | 95% CI | AUC | 95% CI |
|---|---:|---|---:|---|
| J0 - B0 | `-21.94 pp` | `[-37.50,-6.25]` | `-9.62 pp` | `[-14.12,-5.48]` |
| J1 - J0 | `+15.14 pp` | `[+4.17,+27.36]` | `+7.95 pp` | `[+4.78,+11.52]` |
| J2 - J1 | `-0.69 pp` | `[-1.94,+0.97]` | `+0.22 pp` | `[-0.06,+0.57]` |
| J1 - B0（事后） | `-6.81 pp` | `[-12.50,-0.83]` | `-1.67 pp` | `[-3.38,+0.13]` |
| J2 - B0（事后） | `-7.50 pp` | `[-12.92,-2.08]` | `-1.45 pp` | `[-2.98,+0.22]` |

三个 hit@8 预注册对照的 Holm 校正 `p>=0.11`（n=8 欠功效）；裁决依据为
点估计、区间、8 个 evidence level 的同号一致性、3 seed 臂序一致性
（B0 > J1 >= J2 > J0 全部成立）与 AUC 支持性显著。两联合臂不晋升，
零假设方向上无任何支持增益的证据。

### 被试结构（seed-mean hit@8）

| 被试 | B0 | J0 | J1 | J2 |
|---|---:|---:|---:|---:|
| A03 | 0.77 | 0.28 | 0.71 | 0.70 |
| A07 | 0.58 | 0.08 | 0.44 | 0.49 |
| A08 | 0.54 | 0.08 | 0.36 | 0.33 |
| A01 | 0.32 | 0.11 | 0.18 | 0.19 |
| A02 | 0.23 | 0.06 | 0.14 | 0.12 |
| A06 | 0.12 | 0.21 | 0.21 | 0.19 |
| A04 | 0.08 | 0.12 | 0.09 | 0.07 |
| A05 | 0.09 | 0.04 | 0.06 | 0.04 |

J0 的破坏集中于域信号强的被试（A03/A07/A08 分别 `-49/-50/-46 pp`）；
对近 chance 被试（A04/A06）联合源持平或略正。80/20 校正恢复强被试的
大部分但不满（A03 `-6 pp`、A07 `-14 pp`、A08 `-18 pp` vs B0）。

## 对比分析与机制解读

1. **质量主导、单元无关**：J1-J0 的大幅恢复（`+15.14 pp`）与 J2≈J1
   共同表明负迁移主要来自逐 epoch CE 的域梯度质量错位，统计单元
   （epoch vs participant）在该方向不产生可测差异。
2. **选择域已控制仍受损**：v3 联合臂的 inner validation/校准仅用
   BNCI 被试（显式 `--selection-domain`），排除了归档 causal-v2 发现的
   验证域漂移混杂；损害因此发生在训练梯度/表征层，与归档
   "binary AUC 与 decision hit 同时下降"的定位一致。
3. **剩余缺口是条件信号冲突**：80/20 后仍 `-6.8/-7.5 pp`。结合归档
   只读复审（BI/BNCI `target-nontarget` ERP 全张量余弦 `-0.0173`，
   subject-macro `-0.0787`，P3/Pz/P4 `-0.247/-0.304/-0.220`），
   两域判别方向近正交：带域质量 `alpha` 的 pooled 风险最小化只能在
   各域最优判别面的妥协段上取点，标量 `alpha` 无法消除该妥协成本，
   只能选择妥协位置（`alpha=1` 即 B0 为无妥协端点）。这与两个方向
   "校正后仍不胜单域"的观测定量自洽。
4. **方向不对称**：BNCI-target 残余缺口（`-6.8 pp`）大于 BI-target
   归档值（`-0.61 pp`）。注意两点不可直接换算：合同不同（v2 vs v3），
   混合源不同（归档为 BI2014a+BNCI 两域；本运行为 5xBI+BNCI 六域，
   BI 侧异质性更高）；且 BNCI 仅 8 名 ALS 被试，每 fold 目标域只剩
   7 人，冲突梯度相对目标域质量的比重更大。
5. **范式层结论**：跨数据集监督 pooled CE 在该数据体制下是错误抽象。
   多源若要继续，唯一有原理依据的路径是参数空间隔离（per-domain
   head/stem）或梯度冲突控制；这恰是总纲预注册的下一门（J2<B0 已触发）。

## 工程闭环

- 失败定位：manifest 步骤 argv 与两次失败 stderr（584 bytes）记录于
  journal `tasks['manifest']`；根因为旧 `training_procedure_record`
  的整段投影（seed 嵌入 config + per-partition 实测结果拒绝均匀臂）。
- 修复（`ab6c8e7`，已过全量测试 + Ruff）：
  - `training_procedure_record` 改为 `_SOURCE_PROCEDURE_FIELDS`
    显式 allowlist；实测拟合结果与 seed 字段剥离为声明变异轴；
    新控制字段必须显式注册，未知形态整段投影（防空段等价）。
  - `run_pretrain_supervised.py` 支持按 (arm, seed) 的 `--job` 批处理：
    单次 cache/snapshot 装载、每 job fit 前 re-seed（等价于独立运行）、
    batch sentinel 记录每个 checkpoint 的 SHA 与训练合同 digest。
  - promotion matrix DAG 由 96 个单 checkpoint 任务改为 12 个
    checkpoint_batch 任务；resume 时重验 checkpoint 字节与合同 digest；
    fresh run 拒绝未溯源 orphan checkpoint；manifest 增加单臂单
    source cache 钉扎。
- 未闭合：96 个 checkpoint（每个约 24 KB）、
  `frozen/source_training_295942e9e94d.manifest.json` 及其 tar.gz、
  两次 manifest 失败日志仍在云端；下载后本地重建
  `manifest.json` + `analysis.json`，再物理冻结为
  `frozen/research_evidence_295942e_*.tar.gz`。旧 journal 的 DAG
  digest 与新代码不兼容（checkpoint->checkpoint_batch），
  resume 不适用；重建 manifest/analysis 用其独立 argv 直接执行。

## 被推翻与保留的结论

| 旧判断 | 当前裁决 |
|---|---|
| 多源负迁移主因是公共输入统计（mean/std 污染） | 推翻（归档已证）；v3 进一步显示 selection-domain 控制后损害仍在，定位到梯度/表征层 |
| 域质量 80/20 校正可能使联合源净胜单域 | 两个方向均未实现；联合源在该轴上不晋升 |
| epoch vs participant 统计单元可能是恢复关键 | BNCI 方向 J2≈J1（CI 跨 0），该轴关闭 |
| v3 需在 BI-target 方向重跑完整 B0/J0/J1/J2 以裁决机制 | 降级：两方向机制结论一致；BI v3 缓存仍为 Gate 3 校准研究所需，J 臂仅在梯度余弦诊断显示方向差异时补跑 |
| promotion matrix 全链可在旧 journal 上 resume | 推翻：DAG 结构变更使 digest 不匹配；修复后需新输出根或独立执行 manifest/analysis argv |

## 下一轮科研优先级

1. **闭合 295942e 证据链（阻塞项）**：下载云端
   `checkpoints/`、`frozen/source_training_295942e9e94d.manifest.json`
   （含 tar.gz）与 manifest 失败日志；本地用修复后代码重建
   manifest + analysis；物理冻结并移除活跃路径外的 loose 产物。
2. **梯度冲突诊断（预注册门已开）**：在匹配 batch 上测 BI vs BNCI
   对共享 trunk 的梯度余弦（诊断性，无新训练）。若系统性非正：
   单轴测试 per-domain classifier head（共享 trunk + 域专属头，推理用
   目标域头）；不足再考虑 per-domain stem。PCGrad 仍为未测候选，
   不得引用 ERP 余弦声称已验证。
3. **BI2014a v3 校准主线（通往 90%）**：重建 v3 BI target cache 后，
   在合法 cross-decision 协议下跑尚开放的校准轴：BN frozen/adapt、
   time-heldout selection + full-prefix refit、fold-local target QC、
   shrinkage normalization；Gate 4 无真值 pseudo-target 适配须保留
   no-adapt 对照与"任意 pseudo-target 可被学会"反例。
4. **证据效率与被试分层**：本运行 hit 曲线（B0 hit@1 0.113 ->
   hit@8 0.342）与被试极化（A03 0.77 vs A05 0.04）支持 per-subject
   dynamic stopping / 判别性感知的证据预算分配；以闭合后的冻结
   ledger 为测试床，hit-all/R/cost 曲线裁决。
5. **BrainSync 采集（不变）**：90% 产品裁决仍只能在真实成人
   multi-session/multi-decision 数据上做出；多源预训练的先验现在
   明确偏负，产品路径以目标域校准与证据效率为主。

## 证据边界

- BNCI2014-008 为 8 名 ALS 被试（临床人群）、6x6 row/column、
  12 candidates、5 校准 + 30 测试 decisions/人；非成人 BrainSync
  9-choice 部署证据。
- n=8 的 hit@8 对照欠功效；本文裁决为"不晋升"决策（无增益证据 +
  一致负点估计），非"确认有害"的确认性结论；AUC 与 level-wise
  一致性为支持性证据。
- 四臂共享 QC 关闭与 source stats；GTN 线冻结的 source QC100 未
  在本矩阵内检验，属 GTN 开发合同与 BI/BNCI 线的已知差异，不是
  臂间混杂。
- 归档 BI-target 数字为 causal-v2 合同，与本 v3 运行不可直接数值
  比较；两方向比较仅限机制层。
- 本文全部数值来自本地对 96 份 result JSON 的独立复算
  （`tmp/analyze_formal_295942e.py`，seed-mean 后 subject-cluster
  sign-flip，20000 次重采样）；正式 manifest/analysis 重建后以
  归档产物为准。

## 证据入口

- [权威科研总纲](research_program.zh.md)
- 云端运行 journal：`tmp/n2p3-training-295942e-formal/promotion.journal.json`
  （plan sha `675bfbcf…`，DAG digest `d3a298ab…`）
- 修复提交：`ab6c8e7`（allowlist 投影 + checkpoint 批处理 + 溯源）
