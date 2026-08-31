# N2P3-Net 项目批判性复审

日期：2026-09-01
范围：工程集成、BI2014a cross-decision、BrainSync 最终入口、跨域合同、迁移辅助目标、证据可复现性。

## 总判断

项目已经从“dirty worktree 上可运行”推进到 Git 可追溯的研究系统：核心 causal
合同、checkpoint loader、BI/GTN runner、decision-aligned 负结果和 compact evidence
均已进入提交历史。当前主瓶颈不再是代码缺线，而是：没有合法成人 BrainSync
analysis-ready 数据、跨域只有确定性公共通道 CAR 路径、SSL 没有 downstream 性能。

任何“接近产品 90%”“域迁移已打通”“个体校准有效”的表述目前都不成立。

## 已关闭的工程问题

| 项目 | 当前状态 | 验收证据 |
|---|---|---|
| dirty 主链 | 已关闭 | `f107bfa` 纳入 checkpoint/runner/tests/docs；后继 BI commits 可追溯 |
| 冻结证据混入源码 | 已关闭 | Git 仅保留 compact audit index；完整旧目录移出仓库；README 明示无 `.npz/.pt` |
| BI raw/candidate-v2 | 已关闭 | 64 CSV/MAT，cache `61013x16x128`，SHA `252237...75f5` |
| BI block checkpoint scope | 已关闭 | 4x16 被试，target-subject manifest 与 checkpoint holdout 精确核对 |
| BrainSync zero-phase 默认 | 已关闭 | 默认 generic causal 2-30 Hz/800 ms，forward IIR steady-state |
| BrainSync 单 selection | 已关闭 | marker `selection_id/block_id` 形成独立 group；per-selection target/repetition 校验 |
| BrainSync 单 session CLI | 已关闭 | `--session-dir` 可重复；多 session 由 timezone-aware `started_utc` 排序 |
| target-switch runner | 已关闭 | calibration decisions -> real-time embargo -> later 9-choice decisions；失败留分母 |
| 跨域确定性入口 | 部分关闭 | 公共真实通道子集 + 同子集 CAR；参考抵消和缺通道反例通过 |
| SSL visible-copy | 已关闭 | waveform/FFT 仅 masked region 计分 |
| subject probe 假实现 | 已关闭 | stop-gradient 独立 optimizer；逐 subject 20% held-out probe accuracy 入 checkpoint |

## 主要发现

### P0：没有可用于产品判断的 BrainSync 数据

本机现有 4 个历史 sessions：一个无 recording、一个 `recording_error`、两个仍为
`running` 且 `analysis_ready=false`；目标标签也未完整确认。即使 loader/runner
通过 synthetic 测试，也没有真实数据可产生 accuracy。下一步必须重新采集多
known-target calibration decisions 和 later target-changing decisions。

采集端仍记录 `software_repaint` timing、无硬件 trigger，并声明 posthoc algorithm
QC required。正式确认前必须至少保存 timing failure ledger，并将不完整/失败 decision
计入分母。

### P1：BI 合法校准没有显示净收益

BI causal-v2 使用 64 人、12 个 source checkpoints、13 arms、3 seeds：

- zero-shot/source stats subject-macro hit@2=`0.1941`；
- classifier fine + shrinkage=`0.1958`，paired delta `+0.18 pp`，CI 跨 0；
- full fine + shrinkage=`0.1989`，paired delta `+0.45 pp`，CI 跨 0；
- target-prefix normalization 在四种 head 上均下降；
- linear scratch + shrinkage 相对 zero-shot `-4.22 pp`，Holm `p=0.043`。

因此当前默认必须是 source classifier + source stats。更多 target optimizer steps、
full fine 或随机新 head 都没有证据支持。BI 是 6x6 character，不能换算 BrainSync
9-choice 90%。

### P1：BI coverage 只有 68.1%

`test_reps=2` 时 requested=`1416`、eligible=`964`、failed=`452`。conditional hit
约 25-28%，operational decision hit 约 17-19%。只报 eligible 会高估实际系统。
增加 R 会进一步降低覆盖；业务要更高 R，应增加每 decision 刺激量，而不是删失败。

### P1：跨域适配只完成可证明的 CAR 路径

公共通道 CAR 可消除共同参考：

```text
(x_c-r)-mean_j(x_j-r)=x_c-mean_j(x_j)
```

但它要求每条 trial 的全部选定通道都存在，并要求 source/target preprocessing 完全
一致。BI-BrainSync 可取 `CZ,P3,PZ,P4,OZ`，GTN-BrainSync 可取 `FZ,CZ,PZ`；三者
共同仅 `CZ,PZ`，不适合宣称统一多源模型。dataset-specific spatial stem 仍未实现，
也没有真实跨域性能。

### P1：Git 远端跟踪分支不存在

本地 commit 已闭合，但 `main` 显示 `origin/main: gone`，研究分支 upstream 也 gone。
当前 commits 只在本机和上传到云端的 Git bundle 中。是否重建/推送 GitHub branch
是外部发布动作，未在本轮擅自执行。正式协作前必须恢复受保护 remote branch。

### P2：证据包可审计但不能独立重跑

Git 内包含 manifest、分析、subject aggregate、checkpoint/cache hashes 和 source
commit；不包含 EEG `.npz`、checkpoint `.pt`、完整 trial ledgers 或容器镜像。它能
验证外部对象，不能生成对象。该边界已写入每个 evidence README，不再称独立
reproduction bundle。

### P2：SSL 辅助目标已修工程合同，仍无 downstream 结果

masked-only loss 与 held-out subject probe 已闭合；probe 不反向进入 trunk。
但是 masked checkpoint 的 classifier 仍明确 `classifier_trained=false`，不能用于
zero-shot。只有后续 supervised linear/full-fine head 在合法 cross-decision 数据上的
结果才能评估 SSL。当前不应把 reconstruction loss 下降解释为迁移提升。

### P2：GRL/pseudo/dynamic stopping 暂不实现是有意门禁

BI 的简单 supervised personalization 尚无净收益，BrainSync 又无合法数据。此时加入
GRL/erasure、pseudo/latent target 或动态停止会增加不可区分自由度。进入条件：

1. 新 BrainSync development subjects 有完整 target-switch decisions；
2. zero-shot/source-stats baseline 冻结；
3. dynamic stopping threshold 只在 calibration/development 冻结；
4. pseudo-target 通过任意标签反例；
5. erasure 同时报告 subject probe、ERP morphology 和 hit@R，不能只看 probe 下降。

### P3：依赖弃用警告

全回归仍有 MOABB/pyRiemann 与 Braindecode deprecation warnings，当前不改变结果，
但后续依赖升级可能破坏导入或 EEGNet 参数。`uv lock --check` 当前通过；升级应单独
提交并重跑全套，不与科研配方混合。

## 当前允许的结论

- `full_unfold + K35` 是 GTN development 的临时工程默认，不是确认冠军。
- GTN all-evidence candidate mean 优于当前 count-tempered/sum 默认。
- 当前 30-epoch decision-aligned full fine 对 K35 有负迁移。
- BI 5-decision personalization 没有可靠优于 zero-shot/source stats。
- BrainSync 工程入口已闭合，但真实准确率完全未知。
- 产品 90% 仍只能由新成人、多 session、多 target-switch prospective 数据裁决。

## 下一执行顺序

1. 采集并验收新的 BrainSync analysis-ready sessions，先冻结 zero-shot/source-stats。
2. 对 BI/BrainSync source-target 分别生成匹配 preprocessing 的 common-CAR caches，
   训练新的 source checkpoint；不得 `allow_input_domain_shift` 绕门禁。
3. 只在新 development decisions 比较 classifier fine 与 full fine；target stats 暂停。
4. 运行 masked SSL downstream 对照；保留 supervised source checkpoint 强基线。
5. 只有上述基线成立后再研究 spatial stem、pseudo adaptation 和 dynamic stopping。
