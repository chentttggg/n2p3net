# Neural-RIDE v12 recipe contract（当前架构）

> 代码单源仍是 `src/train/recipe.py`；本文件规定当前 v12 契约。v11 运行必须显式
> 使用 legacy recipe，不能当作生产默认。v11 recipe 文档已归档到
> [`../archives/legacy_v11_docs_2026-08-25/neural_ride_recipe.md`](../archives/legacy_v11_docs_2026-08-25/neural_ride_recipe.md)。

> 迁移状态（2026-08-26）：生产默认 recipe 为
> `neural_ride_v12_pcw_fail_closed`，直接实例化 `NeuralRideRecipe()` 也遵循该
> v12 语义；repetition v12 additive-LLR backbone 默认启用。state residual 仍默认
> gain=0，只有通过 held-out log-score gate 后才可激活。v11 仅通过显式的
> `NEURAL_RIDE_V11_LEGACY` 保留为历史/负对照，不能当作默认运行配置。

## Route selection

| `lambda_innovation` | `--z2-aux-head` | recipe | status |
|---:|---|---|---|
| `0` | `off` | `neural_ride_v12_pcw_fail_closed` | canonical production route |
| `>0` | `off` | `neural_ride_v12_strict_past_research` | opt-in research route, all new objects fail-closed |
| `0` | `add` / `replace` | `neural_ride_v12_z2_aux_<mode>_research` | E5 claim-gate contrast only; default disabled |
| `>0` | `add` / `replace` | `neural_ride_v12_strict_past_z2_aux_<mode>_research` | combined research contrast; default disabled |

`--z2-aux-head` 启用的是未约束 full-Z2 辅助试次头：`add` 把
`head_z2(Z2)` 加到 `logit_pcw`；`replace` 让 `logit_target=head_z2(Z2)`。
两种模式都保留 `logit_pcw`、`tau/sigma/H` 与早期证据/幅值头，PCW 仍由
`lambda_pcw` 训练。**它不得由生产默认入口启用**；是否进入生产必须通过预注册的
嵌套 M0/M1 cluster-bootstrap 门槛，并据此修订 constitution E5。

## 四个对象的默认契约

| 对象 | 默认 | 启用条件 |
|---|---|---|
| L: LatencyMeasurement | `measured_tau=None`；PCW `tau` 仅 routing | S0 合成 gate 全过 |
| R: RepetitionEvidence | additive LLR 主干；state residual=0 | residual held-out log score gate |
| Q: Reliability | `fidelity(q)`；`clean_probability=None` | 硬标签生成模型 + prior-shift + unseen corruption gate + 完整 digit-chain NLL/决策复验 |
| S: InnovationAudit/Stopping | `final=PCW`；只做经验 replay | 嵌套 M0/M1 cluster bootstrap 门槛 |

## PCW 先验 canonical 常量

所有 τ0/σ/dτ 默认值只来自 `src/models/component_window.py` 的
`PCW_CANONICAL_*`（成人先验）：

- `tau0_ms = (220, 300, 350)`
- `tau0_bounds = ((180,280), (250,380), (280,500))`
- `sigma_bounds = ((20,50), (20,80), (20,80))`
- `dtau_bounds = ((-30,30), (-30,0), (-50,150))`

GTN（儿童）使用 `GTN_CHILD_*` 命名覆盖：`tau0_P3b=460`、
`tau0_P3b ∈ [350,600]`、`sigma_P3b ∈ [20,150]`；只允许 GTN runner 引用，
不得进入通用默认。fold-calibrated prior 与 frozen prior 的 resolved 值必须写入
`record.json` 的 `erp_prior.resolved`。

## 正式路线默认

- PCW：width 64、depth-4 TCN（dilations 1/4/16/32）、BN、dropout 0.25；
- full-Z2 auxiliary head：默认关闭（`use_z2_aux_head=false`）；
- tokenizer：bandpass initialization，无 post normalization/activation；
- repetition：additive-LLR backbone 默认；state residual gain=0，audit gate 通过才可非零；
- L：`use_measurement_windows` 默认 false；启用时 fold 内 fit LatencyMeasurement、nested M0/M1 gate 后才把 detached 后验窗加入 PCW；
- innovation：research opt-in width 28、kernel 9、dilations 1/2/4/8/16；
- ERP decoder：disabled；
- `lambda_pcw=0.3`；`lambda_innovation` 默认 0；
- `L_amp`/`L_jit` 默认关闭；`L_tau` 仅作显式 identification 研究，且不作为
  生理潜伏期监督；
- variance warmup/ramp 5/10；early-stop patience 6。

## Research route evidence

- 分类增量只报告嵌套 `M0:a+bS` vs `M1:a+bS+cL` 的 subject-cluster bootstrap；
- 异常价值只报告两假设 typicality/conformal p-value，两类都拒绝才请求补采；
- 动态停止第一步只报告 first-crossing replay：错误率、未决率、expected flashes、
  risk-coverage 曲线。

## 数据集能力与 split contract

`GTN_DIGIT_TASK` 启用 digit-set 与 repetition objectives；`BINARY_ODDBALL_TASK`
置零。LOSO fold 四个角色保持 disjoint：optimization / audit / validation / test。

## Entry points

```powershell
.\.venv\Scripts\python.exe experiments/run_n2p3net_gtn.py `
  --subjects 12 --max-folds 5 --epochs 24 --batch-size 256 `
  --run-dir tmp/acceptance --run-name issue2_v12_5fold_seed0
```

Generic oddball 与 transfer 入口不变，仍要求 `n2p3net_epoch_dataset/2` 或
`n2p3net_gtn_cache/2` 缓存。

## Acceptance

每 fold 记录必须保存：四个对象各自的输入/输出/gate 结果、fail-closed 原因、
cluster bootstrap CI、prior-shift 与 coverage、PCW/final 分列。五到八个 outer
folds 是最小经验审核单元；一个 fold 或单元测试不构成 SOTA 证据。
