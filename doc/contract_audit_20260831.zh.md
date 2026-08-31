# 数据与训练合同核对报告（2026-08-31）

状态：当前审计结论。科研指令已直接合并进 `research_program.zh.md`、
`single_subject_transfer_research.zh.md`、`blueprint.md` 和 `README.md`。

## 1. 已确认的硬错误

### 1.1 Causal IIR 初态

旧 `phase=forward` 从全零 SOS state 启动。GTN 原始通道带 mV 级 offset，实测
Experiment 341 首两 trial PTP 达约 1661/1361 uV，20 秒后才回到约 85--88 uV；
matched zero-phase 首两 trial 约 94/67 uV。该瞬态只污染 early prefix，直接制造
prefix/suffix domain shift。

现改为：

```text
causal_iir_initial_state = steady_state_first_sample
z0 = sosfilt_zi(sos) * first_sample
```

并升级 causal contract：generic v2、GTN v4、paper v2。该字段写入 cache JSON 并
参与 assert；旧 forward cache 默认解析为 `not_applicable`，会被新 causal contract 拒绝。

### 1.2 GTN estimand

GTN 每人一个固定 thought digit。用同 selection prefix 真标签适配，再猜 suffix 的
同一 digit 是 oracle-label proxy，且可能学习 digit glyph/VEP。它不能支持未知数字
监督校准结论。合法主路为 target-excluded Z0/Z5 或不使用 truth 的 pseudo/latent
target adaptation。

GTN 刺激是随机流，不是同步 1--9 round。candidate-local occurrence index 不能当
同步 block。当前 operational split 等待每候选 M/R 个有效 evidence，并记录每候选
原始 occurrence、总 stimuli 和秒数。

### 1.3 BI candidate repetition

旧构造器在每个 12-flash block 内计数，导致 repetition index 永远为 flash 位置
0..11；旧 475 MB causal cache虽 schema/attestation 合法，语义错误且 runner 无法执行。
构造器已改为 selection 内 repetition 0,1,2... 并消费 raw Event 100/104 边界；
但当前本地/远端缺 BI raw source，candidate-v2 尚未构建，没有 cross-decision 性能结果。

BI runner 已改成：前 K 个已知 character decisions 校准 -> raw-time embargo ->
later unknown character decisions 测试，不再在同一 character 内切 repetition。

### 1.4 Checkpoint 语义

旧 loader 只看 C/T/sfreq/tmin，且硬编码 ms_flatten/K65。现在 checkpoint 保存并校验：

- 完整 architecture；
- ordered channel names；
- reference 与完整 preprocessing；
- source cache SHA；
- training/holdout/cache-qualified subject keys；
- source input statistics、weighted-CE prior/weight；
- classifier 是否经过监督训练。

同号 subject 只有在同 dataset/cache 或显式 global participant identity 下才算重叠；
跨数据集同号不会误杀。任何通道/参考/recipe domain shift 必须通过显式 stem/adapter。

### 1.5 训练与评分

- `--temporal-kernel-size` 现在真正进入 supervised pretraining architecture；
- source inner validation 选 epoch 后，从同一初始化在全部合法 source rows refit；
- target time-split 选 epoch 后在全部 prefix refit，不永久损失 validation 数据；
- accuracy 主臂可使用预注册 fixed epoch budget，在全部 prefix 直接训练；
- 保留 source classifier 的 `classifier_fine` 不再随机重置 head；
- masked channel 在训练/预测均先标准化再 zero-fill；
- weighted CE 解析 offset + 正温度，禁止 tiny Platt 负 slope 反转排序；
- trim 按固定数量裁剪，R<5 不裁；tie 一律 abstain/miss；
- 无 predictive variance 时 CLI 不开放 precision aggregation。

## 2. 不是硬合同的准确率 recipe

以下按状态管理，不因“更严格”一刀切：

```text
high-pass:       0.1 vs 0.5 Hz              [GTN completed]
epoch end:       800 vs 1200 ms             [GTN completed]
offline/online:  split-local zero-phase vs forward steady-state
normalization:   source vs target-prefix vs shrinkage
target QC:       none vs prefix-fit fold-local
epoch select:    fixed budget vs real-time holdout+refit
BN:              frozen running stats vs target adapt
aggregation:     all-evidence mean vs tempered effective count vs sum vs fixed-count trim
```

旧 B0->C1 同时改 high-pass 与窗长，不能单独归因；后继 steady-state 2x2 已完成，
GTN development 冻结 0.1 Hz/1200 ms。其他目标域仍需独立验证。

## 3. 当前制品判定

| 制品 | 判定 |
|---|---|
| BI zero-phase LOSO records | 历史开发证据；outer 已花掉 |
| GTN zero-phase 2Hz/800 assets | 可复算旧 recipe，不作当前确认 |
| GTN causal v1/v3 caches/checkpoints | startup 初态错误，禁止复用 |
| BI candidate causal v1 cache | repetition 语义错误，禁止复用 |
| 旧 zero-state causal ranking | 同 suffix 后验选择且 startup 污染；数值不再作为 current baseline |
| GTN 2x2 steady-state bundle | 4 cache attestation/CRC、32 checkpoints、120 eval JSON 均通过；本地证据包已冻结 |
| 当前最佳已审计 fixed-R Z0 baseline | 0.1 Hz/1200 ms/source QC100；hit@5 coverage 230/245；operational 0.543，未达 0.90 |
| full-unfold 核长 | 36 checkpoints / 3 seeds 完成；K35/K65/K33 balanced-all 0.669/0.654/0.623 | K35 临时默认；严格 K35 vs K65 未决，K33 退出主线 |
| all-evidence count correction | 冻结 v4 ledger 上 K35/K65 candidate mean 0.714/0.683；mean 均不差于 sqrt-count/sum | 使用全部 trial 的 mean 为当前开发默认；balanced/sum 保留对照 |
| 端到端 decision fine-tune | 24 checkpoints，30 epoch，全源 EEG；K35/K65 learned 0.688/0.676 | 当前 recipe 不晋升；K35 backbone 负迁移最明显 |

## 4. 当前验证与下一步

本地 `.venv` 已覆盖：steady-state DC/future impulse、M=0、候选次数不等、真实时间
inner embargo、trim/tie、checkpoint signature、full-prefix refit、BI cross-decision。

下一步：

1. 当前工程 recipe 冻结为 `full_unfold + K35`、0.1 Hz/1200 ms、source QC100、
   无额外联合微调、all-evidence candidate mean；K65 保留强对照，K33 停止主线投入；
2. 补齐 BI raw source后重建 causal-v2 cache，跑跨 decision calibration；当前无性能结果；
3. 采集成人 BrainSync target-switch 多 decision 数据；只有该数据可裁决 90%。
