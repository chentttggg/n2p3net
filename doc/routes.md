# N2P3-Net 两条路线与四个可证伪对象

> 本文档是路线选择的操作入口。模型细节以 `blueprint.md`（v12）为准，代码默认值以
> `src/train/recipe.py` 为准。v11 strict-past 路线文档已归档到
> [`../archives/legacy_v11_docs_2026-08-25/routes.md`](../archives/legacy_v11_docs_2026-08-25/routes.md)。

## 1. 先选路线

| 项目 | 正式/生产路线 | strict-past 研究路线 |
|---|---|---|
| recipe | `neural_ride_v12_pcw_fail_closed` | `neural_ride_v12_strict_past_research` |
| GTN 开关 | `--lambda-innovation 0`（默认） | `--lambda-innovation 1` 或其他正数 |
| 主神经判别量 | `logit_pcw` | 仍是 `logit_pcw` |
| 最终输出 | `final = PCW` | 只有嵌套 M0/M1 审计和 outer claim gate 通过才融合；否则仍为 PCW |
| 当前地位 | 唯一正式默认 | opt-in 研究候选，当前未通过增量决策价值门槛 |

`--lambda-recon`、`--lambda-morphology-l0` 等参数不构成第三条路线。

## 2. 正式/生产路线

硬边界不变：

- `final = PCW`；没有 global classifier、residual classifier 或 sensor-space bypass；
- 不启用 strict-past likelihood、低秩条件协方差或 prequential fusion；
- 结果按 GTN 的 ITT、`exact_llr@3`、coverage、校准和被试级配对协议报告；
- 正式结论不能借用 strict-past 分支的 NLL 改善来证明分类增益。

```powershell
.\.venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
  --lambda-innovation 0 --run-dir tmp\production
```

## 3. strict-past 研究路线：四个对象全部 fail-closed

strict-past 研究路线由 `blueprint.md` 的四个独立对象组成：

| 对象 | 可进入研究路线的条件 |
|---|---|
| L: LatencyMeasurement | S0-5 与合成/2-fold latency audit 全过；PCW 只消费 detached 后验 |
| R: RepetitionEvidence | additive LLR 主干恒在；state residual 通过 held-out log score 增量 gate |
| Q: Reliability | `fidelity` 恒在；`clean_probability` 只在硬标签生成模型 + prior-shift gate 通过后启用 |
| S: InnovationAudit + Stopping | 嵌套 `M0:a+bS` vs `M1:a+bS+cL` cluster bootstrap 通过才允许融合 |

**Fusion 激活条件（替代旧非负 alpha）**：

```text
c 无约束拟合 -> c 符号预注册 -> cluster 95% CI 不跨 0
-> subject-macro NLL 改善 >= 0.5% -> AUC 非劣界 -0.005。
```

普通 block permutation 不作零分布；`corr(S,L)` 符号翻转禁止；raw NLL 禁止直接
gate；普通 posterior 阈值不得宣称固定错误率控制。

strict-past 的实际声明仍为 `preprocessed_epoch_strict_past`，不是 raw streaming。

```powershell
.\.venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
  --lambda-innovation 1 --audit-subjects 4 `
  --run-dir tmp\strict-past-research
```

## 3b. full-Z2 auxiliary head：E5 claim-gate 对照（默认关闭）

为证伪「PCW-only 是否丢弃三个成分窗外的判别信息」，提供一条显式命名的研究
对照，不作为第三条生产路线：

```powershell
# A1：PCW + full-Z2（logit_target = logit_pcw + head_z2(Z2)）
.\.venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
  --z2-aux-head add --z2-aux-pool attention `
  --run-dir tmp\z2-aux-add-research

# A2：full-Z2-only（logit_target = head_z2(Z2)；PCW 仍作为 side readout 训练）
.\.venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
  --z2-aux-head replace --z2-aux-pool attention `
  --run-dir tmp\z2-aux-replace-research
```

- `off` 是生产默认；`add`/`replace` 只生成
  `neural_ride_v12_z2_aux_<mode>_research`（或 strict-past 组合名）recipe。
- 三臂必须固定 folds/seeds/预算，与 `A0=off` 做 subject-cluster bootstrap；
  `logit_pcw`、`tau/sigma/H` 在所有模式下保持输出并接受 `lambda_pcw` 训练。
- 只有预注册门槛通过，才能提出 constitution E5 修订案；在此之前该头不得由
  正式默认入口启用，也不得把 `head_z2` 解释为成分级证据。

## 4. 已归档且不得带回的旧方法

以下内容已归档，不再作为可执行路径：residual classifier、先分类再 ERP
subtraction、`alpha/ramp` 门控、identifiability hold、单参数非负 fusion、
soft-BCE(0.9/0.1) 的 rho 概率语义、`attention_softargmax` 作为生理潜伏期、
`known_time_shift` 零填充作为 latency audit 证据。历史原因见
[`../archives/legacy_v11_docs_2026-08-25/`](../archives/legacy_v11_docs_2026-08-25/)。

## 5. 升级与删除规则

新对象只有在完整的 locked outer protocol 中通过 `blueprint.md` 第 7 节门槛，
才可申请进入正式路线：至少 5 个完整 outer folds，audit 与 validation 分区保持
独立，strict-majority fold 改善，AUC/Brier 非劣，且 cluster bootstrap CI 支持。
通过单元测试、单折 NLL 或一次开发运行都不够。

通过替代性验证后，按 E4a 删除旧实现、旧入口、旧测试和旧文档；只有仍具独立
科学问题或明确复现义务的内容，才作为带名称、带 gate 的显式研究消融保留。