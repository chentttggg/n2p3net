# N2P3-Net 项目批判性复审

日期：2026-09-01
范围：工程集成、BI2014a cross-decision、BrainSync 最终入口、跨域合同、迁移辅助目标、证据可复现性。

## 总判断

项目已经从“dirty worktree 上可运行”推进到 Git 可追溯的研究系统：核心 causal
合同、checkpoint loader、BI/GTN runner 和 decision-aligned 负结果均已进入提交历史；
冻结材料已从活跃树压缩隔离。当前主瓶颈是：统一 v3 合同后的 source cache/checkpoint
尚未重建、没有合法成人 BrainSync
analysis-ready 数据、跨域联合源仍未超过单域 source、SSL 没有 downstream 性能。

任何“接近产品 90%”“域迁移已打通”“个体校准有效”的表述目前都不成立。

## 已关闭的工程问题

| 项目 | 当前状态 | 验收证据 |
|---|---|---|
| dirty 主链 | 已关闭 | `f107bfa` 纳入 checkpoint/runner/tests/docs；后继 BI commits 可追溯 |
| 冻结证据混入源码 | 已关闭 | loose evidence、dated runners/docs 已移入 checksummed `frozen/*.tar.gz` 并从活跃路径删除 |
| BI raw/candidate-v3 | 待重建 | builder 已统一到 179 samples；旧 `61013x16x128` cache 只存在冻结归档 |
| BI block checkpoint scope | 已关闭 | 4x16 被试，target-subject manifest 与 checkpoint holdout 精确核对 |
| BrainSync causal 默认 | 已关闭 | canonical v3：0.1-30 Hz/1200 ms，forward IIR steady-state；无 2/800 fallback |
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

### P1：归档 BI 合法校准没有显示净收益

物理归档中的 BI causal-v2 使用 64 人、12 个 source checkpoints、13 arms、3 seeds：

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

### P1：公共 CAR 已打通，但朴素联合源产生负迁移

公共通道 CAR 可消除共同参考：

```text
(x_c-r)-mean_j(x_j-r)=x_c-mean_j(x_j)
```

但它要求每条 trial 的全部选定通道都存在，并要求 source/target preprocessing 完全
一致。冻结的 128-sample BI2014a+BNCI2014_008 common-CAR 实验中，BI-only
hit@2=`0.1300`，uniform joint=`0.0967`，差 `-3.33 pp`，95% CI
`[-5.90,-1.28] pp`。因此“通道/参考可加载”不等于“域迁移有效”。

看到该负结果后注册的 BI 3x/BNCI 1x 探索臂为 `0.1239`：相对 uniform 恢复
`+2.72 pp`，Holm `p=0.0052`；相对 BI-only 仍 `-0.61 pp`，CI 跨 0。它还增加了
每 epoch optimizer steps，不能把恢复量纯解释为域权重。固定 uniform joint 数据与
steps、仅用 BI source rows 拟合 checkpoint 输入统计后，hit@2=`0.0974`，相对
all-source stats 仅 `+0.07 pp`，CI 跨 0；相对 BI-only 仍显著 `-3.26 pp`。因此
normalization contamination 不是主因，剩余损失来自跨域梯度/表征冲突的证据更强。
下一轴应先用固定 steps 的 normalized per-row domain loss weight 隔离梯度比例；若仍
无增益，再研究 gradient surgery 或 dataset-specific stem。三数据集公共仅 `CZ,PZ`，
仍不适合统一空间卷积。这些数值只能作为压缩归档中的机制证据；v3 当前链必须从
raw 重建 179-sample source cache 后重新训练，不能直接续跑旧 checkpoint。

### P1：Git 远端跟踪分支不存在

本地 commit 已闭合，但 `main` 显示 `origin/main: gone`，研究分支 upstream 也 gone。
当前 commits 只在本机和上传到云端的 Git bundle 中。是否重建/推送 GitHub branch
是外部发布动作，未在本轮擅自执行。正式协作前必须恢复受保护 remote branch。

### P2：冻结证据已物理隔离但不能独立重跑

`frozen/research_evidence_through_20260901-d1db8e4.tar.gz` 包含当时的 manifest、
分析、subject aggregate、checkpoint/cache hashes、runner、合同和文档快照；活跃树
不再暴露旧接口。压缩包不包含 EEG `.npz`、checkpoint `.pt`、完整 trial ledgers 或
容器镜像，因此只能验证外部对象，不能独立重跑。

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
- BI+BNCI uniform joint 有显著负迁移；80/20 行重复只能恢复到 BI-only 附近；
  BI-source stats 与 all-source stats 等效，未产生净增益。
- BrainSync 工程入口已闭合，但真实准确率完全未知。
- 产品 90% 仍只能由新成人、多 session、多 target-switch prospective 数据裁决。

## 下一执行顺序

1. 从 raw 重建 v3 BI/BNCI/GTN common-CAR source cache 与 target-excluded checkpoint。
2. 采集并验收新的 v3 BrainSync analysis-ready sessions，先冻结 zero-shot/source-stats。
3. 在新 BI+BNCI common-CAR cache 上保持 uniform 唯一行、batch 与 steps，比较
   unweighted CE 与归一化 80/20 per-row domain-weighted CE；不得再扫 repeat/stats。
4. 若固定-step domain weight 仍未胜 BI-only，停止简单混合，再进入梯度冲突控制；
   dataset-specific spatial stem 放在确定梯度负迁移之后。
5. 只在新 BrainSync development decisions 比较 classifier fine 与 full fine；target stats 暂停。
6. 运行 masked SSL downstream 对照；保留 supervised source checkpoint 强基线。
7. 只有上述基线成立后再研究 pseudo adaptation 和 dynamic stopping。
