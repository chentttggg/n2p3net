# GTN 精度损失分解：工程契约审查与文献对照

日期：2026-08-30。状态：历史开发分析；当前指令以 `research_program.zh.md` 为准。
本文件中的 zero-phase LOSO 数字仍可说明当时的联合 recipe 现象；所有 forward
causal 数字使用了零状态 IIR，存在严重 startup transient，已失效用于当前性能比较。
同一 suffix 又被连续用于 QC、epoch、block 和 aggregation 选择，因此旧 causal
ranking 已从当前正文删除，不再保留为 baseline。
触发：EEGNet 历史复现 AUC≈0.78 / decision hit≈0.80，第一轮 GTN 128Hz 地板只到
AUC 0.66 / hit_all 0.690 / hit@8 0.62。本文审查"数学自洽但工程过严"的举措。

## 0. 事实基线（先分清口径）

| 口径 | 数值（当时 2 Hz/800 ms zero-phase 协议, eegnet, seed 28/29） |
|---|---|
| trial AUC | 0.664 / 0.661 |
| trial BACC | 0.611 / 0.609 |
| decision hit（全 repetition） | **0.690 / 0.694** |
| decision hit@8 | 0.604 / 0.624 |

历史 0.78/0.80 与该旧 recipe 的差距 = AUC −0.12、hit −0.11。旧 LOSO
fold-local QC 排除量虽小，但这不能外推到 source-pretraining QC；后继 matched
Z0 已显示 source QC100 相对 no-QC operational hit 提升 `+11.84 pp`。

## 1. 嫌疑清单与文献对照

### S1. 高通 2 Hz（嫌疑最大）

- 我们的论证：`DEFAULT_P300_DATA_CONTRACT` 2–30 Hz 是为"保留固定 65/5/17
  采样核的物理尺度"。但该论证只约束**采样率缩放时核宽换算**，并不约束
  滤波本身——核是学习出来的，不需要截止频率保护。
- 文献：Bougrain, Saavedra & Ranta (INRIA, TOBI Workshop 2012)，23 被试系统
  扫描 0.1–1 Hz 高通 × 8–60 Hz 低通：**[0.1, 15] Hz 最优**；1 Hz 高通在几乎
  所有低通下最差（如 60 Hz 低通时 46.4 vs 0.1 Hz 的 48.4）；并引 Duncan-Johnson
  经典结论——高0.1 Hz 的高通会使 P300 ERP 失真。
- P300 是 1–4 Hz 主导的慢波；2 Hz 高通直接削掉峰附近能量并引入相位失真
  （零相位 IIR 缓解相位，不缓解幅度损失）。
- 判定：**工程上无必要、文献上明确有害。应回到 0.1 Hz（或 0.5 Hz 上限）。**

### S2. 时窗 [-200, 800) ms（嫌疑次大）

- 我们的论证：紧凑输入、核尺度不变。推导时默认成人 P300（300–500 ms）。
- 文献：儿童 P300 潜伏期随发育递减——Goodin et al.：6–15 岁儿童 P3 潜伏期
  以 **18.4 ms/年** 递减（即 7 岁比 15 岁长约 150 ms，比成人长 200 ms+）；
  学龄儿童 P300 正成分要到 **~1000 ms** 才回到基线（Taylor et al. 2004）；
  Courchesne (1977)：儿童 P3 峰可在 600–900 ms。
- GTN 是 7–17 岁儿童。**[-200,800) 截断了大量目标证据**；legacy 1.2 s 窗合理。
- 判定：**对 GTN 人群过严。应扩至 [-200,1200) 或按年龄分段。**

### S3. decision 层 sum 无中心化（已被项目自己的决策记录预言）

- `models/decision.py` 的 D-center-logits 决策：常数偏置经 Σlogit 变
  `c·n_d` 进入 argmax，实测 GTN 30 被试 LOSO：SWLDA 0.467→0.667(sum)/
  0.700(mean)。而 `run_eeg_loso.py` 一直调用 `decide(..., center_logits=False)`，
  且 hit@R 聚合同样未中心化。
- artifact 排除后 n_d 天然不均 → 偏置项进入判定。这一项不解释 trial AUC
  0.66，但直接压 decision hit。
- 判定：centering A/B 已完成且无增益；校准 LLR 后保持关闭。

### S4. 128 Hz 重采样（嫌疑小）

- EEGNet/文献常在 128 Hz 下工作；采样率本身预计损失 <0.01 AUC。
- 但 256→128 与 S1/S2 交互（核宽缩放后频率响应不变，收益中性）。

### S5. artifact QC（必须区分位置）

- 旧 LOSO fold-local QC 只排除 0.5% train / 26 test epochs；对 3 导数据的
  fail-closed 崩溃已修。source-pretraining QC 是不同 estimand，后继 matched Z0
  已确认 QC100 显著优于 no-QC；target-prefix QC 仍关闭并保留独立消融。

### S6. 禁 per-epoch 标准差归一化（宪法禁令，暂不动）

- 文献中 P300 管线常做 per-epoch z-score（幅值归一）。宪法禁止的理由
  （QC 物理语义、V 单位）成立，且 EEGNet 有 BN。列为待 A/B 的低优先项，
  若 S1/S2 修复后仍差距大再测。

## 2. 受控 A/B（同代码、同 seed 20260828、同模型 EEGNet）

| 臂 | cache | decision | AUC | hit_all | hit@5 | hit@8 |
|---|---|---|---:|---:|---:|---:|
| B0（基线） | 128/2/800 | center=False | 0.6637 | 0.6898 | 0.522 | 0.604 |
| **B1** | **256/0.1/1200（legacy 风格）** | center=False | **0.7364** | **0.7796** | **0.653** | **0.759** |
| B2 | 128/2/800 | center=True | 0.6645 | 0.6776 | 0.510 | 0.600 |
| **C1** | **128/0.1/1200** | center=False | **0.7300** | **0.7918** | **0.649** | **0.743** |

### 2.1 判读（全部完成，三因素分解闭环）

1. **联合预处理 recipe 效应 = ΔAUC −0.066、Δhit_all −0.102、Δhit@8 −0.139**
   （C1 − B0 同时改变 l_freq 0.1→2 与 tmax 1200→800）。该对比不能分别
   归因给 high-pass 或窗长；后继 0.1/0.5 x 800/1200 steady-state factorial 已完成。
   B1 复现 AUC 0.736/hit 0.780，C1 复现 0.730/**0.792**——历史 0.78/0.80
   完全可复现，**不是旧代码 artifact**。T 的质疑成立且被精确归因。
2. **128 Hz 采样率清白**：C1 − B1 = ΔAUC −0.006、Δhit_all +0.012，噪声级。
   紧凑采样率可保留。
3. **候选机制，不是已锁定单因子**：
   - **l_freq=2 Hz**：P300 主能量在 1–4 Hz，2 Hz 高通削峰并失真（Bougrain
     2012：[0.1,15] 最优，>0.1 Hz 开始失真 P300，1 Hz 已是最差档）；
   - **tmax=800 ms**：GTN 为 7–17 岁儿童，P3 潜伏期随发育递减（Goodin：
     6–15 岁以 18.4 ms/年递减；学龄儿童正成分 ~1000 ms 才回基线），
     800 ms 窗硬截断目标成分。
4. **S3（decision centering）被否定**：B2 − B0 = −0.012 hit_all。校准后
   LLR 已除常数偏置，centering 无益；D-center 决策记录适用于未校准 logit。
5. 旧 LOSO fold-local QC 排除量小，不是该轮原因；这不适用于后继 source QC，
   后者 QC100 已显著优于 no-QC。采样率在 C1−B1 中无明显损失。

### 2.2 意义

- 0.78 → 0.66 的"退步"**不是模型或训练问题，是 2026-08-28 冻结的输入契约
  对 GTN（7–17 岁、3 导）过严**。"先证契约"流程本身工作正常——它忠实执行了
  一个错误的默认值。
- 昨晚/今晨所有 128Hz/2Hz/800ms 协议下的数字（含 full_unfold vs ms 的
  GTN 判定、causal chance 水平）都要在修订契约后重估：2Hz 高通削掉的
  delta 能量对不同 readout 的影响可能不对称（full_unfold 的绝对时序模板
  对波形失真可能更敏感）。修订契约复跑已完成，不能从旧 2 Hz 结果否定
  full-unfold；后继多 seed all-evidence 已将 K35 设为临时默认、K65 保留对照。
- 当时 hit@8=0.743 仍远低于当前 0.90 产品目标；且该 cohort/recipe 已用于开发，
  不能据此估计剩余可达性。

## 3. 修订决定（由 A/B 裁决）

1. **GTN/儿童开发 recipe**：后继 steady-state 2x2 已完成，冻结
   `l_freq=0.1`、`tmax=1200`；它是已查看 GTN cohort 的开发 winner，不是物理硬合同。
2. **成人/8 导契约**：窗宽可保留 800 ms（成人 P300 500–700 ms 回基线），
   但 l_freq 建议同样回到 0.1–0.5 Hz；最终值由 BI2014a 上的同款 A/B 裁决。
3. decision centering 保持 False（默认），不再列为嫌疑项。
4. 契约修订属"documented contract change"（CODING_WORKFLOW），旧结果
   不与新契约结果跨版本比较；128/2/800 轮的所有 record 保留并标记
   `contract=p300_ms_eegnet_input_v2`（受限版）。

## 4. 修订契约复跑结果（2026-08-30 下午，seed 20260828 先行）

契约：`gtn_ms_eegnet_input_v3`（128 Hz / **0.1 Hz** / [-200,**1200**) ms），
LOSO 与 causal 均已在新契约上重跑。

### 4.1 LOSO floor（seed 28，245 折，subject-paired）

| 模型 | AUC | BACC | hit_all | hit@5 | hit@8 |
|---|---:|---:|---:|---:|---:|
| window_lr | 0.596 | 0.567 | 0.404 | 0.257 | 0.359 |
| xdawn_rg | 0.613 | 0.573 | 0.506 | 0.327 | 0.376 |
| eegnet | 0.7302 | 0.6688 | 0.7918 | 0.653 | 0.739 |
| ms_eegnet | 0.7283 | 0.6701 | 0.7878 | 0.649 | 0.731 |
| n2p3net_full_unfold K65 | 0.7291 | 0.6673 | **0.8000** | 0.641 | 0.735 |

配对检验（100k sign-flip）：fu−ms p=0.64、fu−eegnet p=0.53（**打平**）；
fu−xdawn/window_lr p<1e-5。**修正上午的判定：修订契约下三深度臂 AUC 打平
（0.728–0.730），"full_unfold 被 GTN 否定"撤回**；hit_all 上 fu 最高
（+0.008–0.012；后续 seed29 未形成确认性优势）。旧契约 2 Hz 下 fu 的劣势与 causal 下的
伪优势均为损坏信号上的 artifact。

### 4.2 zero-state causal 结果已由 successor 替换

本节旧 M=5/R=5 表使用 zero-state forward IIR 且读取同 thought digit 标签，已删除。
当前合法 causal development 证据是 steady-state Z0：0.1 Hz/1200 ms/source QC100
的 coverage `230/245`、operational hit@5 `0.543`、AUC `0.709`。Z5 只作
matched-time sensitivity；O5 只作 oracle proxy。核长由 adopted full-unfold 下
三 seed比较已完成：K35/K65/K33 balanced-all=`0.669/0.654/0.623`。

### 4.3 契约修订落地

- `DEFAULT_GTN_DATA_CONTRACT` → `gtn_ms_eegnet_input_v3`（0.1 Hz / 1200 ms）；
  新增 `GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT`；
  `causal_prefix_suffix_split(contract=...)` 参数化（旧硬编码 assert 移除）。
- BI/成人默认契约 `DEFAULT_P300_DATA_CONTRACT` 未动，其 l_freq/窗宽由
  BI2014a 同款 A/B 另行裁决。
- 当前 causal successor 是 attested `gtn_causal_ss_lf0.1_t1200` 等四个 steady-state
  cache；旧 zero-state `...mean_fwd` 不再作为 current asset。

## 5. 历史 two-seed zero-phase 开发描述（各 245 折）

| 模型 | AUC s28/s29 | hit@8 s28/s29 |
|---|---:|---:|
| eegnet | 0.7302 / 0.7297 | 0.739 / 0.755 |
| ms_eegnet | 0.7283 / 0.7251 | 0.731 / 0.747 |
| full_unfold K65 | 0.7291 / 0.7266 | 0.735 / 0.722 |
| full_unfold K33 | 0.7299（s28） | 0.759；hit_all **0.804** |

配对（fu − ms）：+0.0009 (p=0.64) / +0.0015 (p=0.48)；（fu − eegnet）：
−0.0011 (p=0.53) / −0.0032 (p=0.10)。**开发描述：修订契约下四臂（EEGNet、
ms_flatten、fu-K65、fu-K33）在 trial AUC 与 hit@R 上均无可靠排序**；
K33 的 hit_all 0.804 是该次单 seed 点估计最高，不足以选择核长。
历史 0.78/0.80 的水平已恢复并小幅超越（hit_all 0.79–0.80）。
该开发轮联合 recipe 贡献 Δhit@8 ≈ +0.13；causal steady-state replacement 已完成，
当前 0.90 剩余缺口仍需 target-switch 多 decision 估计量，不能简单归因给 transfer。

云端 runs：`gtn128rev_loso_floor_seed2026082{8,9}/`、
`gtn128rev_loso_k33_seed20260828/`、`gtn128rev_causal_m5r5_*_seed20260828.json`、
`gtn128rev_floor_analysis_seed2026{0828,0829}.json`。

## 6. 历史 zero-phase 高通比较：原文 0.5 Hz vs 0.1 Hz

GTN 原文（gtn_unet_2023）高通为 **0.5 Hz**。同代码同 seed 的 eegnet 对照：

| 高通 | AUC | BACC | hit_all | hit@8 |
|---|---:|---:|---:|---:|
| 0.1 Hz | 0.7302 | 0.6688 | 0.7918 | 0.739 |
| **0.5 Hz（原文）** | 0.7177 | 0.6557 | 0.7551 | 0.722 |
| 2 Hz（已废弃） | 0.6637 | 0.6109 | 0.6898 | 0.604 |

历史 zero-phase 描述：0.5 Hz 相对 2 Hz 恢复了性能，但略低于 0.1 Hz。这个
LOSO 比较没有 causal startup 初态问题，可用于说明 2 Hz 不应成为 GTN 默认值；
它不能单独裁决在线 0.1/0.5 Hz 排序。后继 steady-state matched 2x2 已完成：
0.1 Hz 平均 operational hit 提升约 `+5.92 pp`，因此开发线冻结 0.1 Hz；
paper-aligned 锚点仍保留 0.5 Hz。
原文窗宽待查：若原文亦为 1200 ms 则 v3 窗宽已对齐。

## 7. 原文对齐全矩阵结果（0.5 Hz / 128 Hz / 1200 ms，seed 20260828）

### 7.1 LOSO floor（245 折，subject-paired）

| 模型 | AUC | BACC | hit_all | hit@5 | hit@8 |
|---|---:|---:|---:|---:|---:|
| window_lr | 0.578 | 0.547 | 0.363 | 0.253 | 0.282 |
| xdawn_rg | 0.608 | 0.561 | 0.449 | 0.331 | 0.396 |
| eegnet | 0.7177 | 0.6577 | 0.7633 | 0.645 | 0.722 |
| ms_eegnet | 0.7151 | 0.6559 | 0.7755 | 0.633 | 0.722 |
| full_unfold K65 | 0.7131 | 0.6518 | **0.7878** | 0.637 | 0.735 |
| full_unfold K33 | **0.7181** | 0.6545 | 0.7755 | **0.665** | 0.735 |

历史开发配对（fu-K65 基准）：fu−ms −0.002 (p=0.33)；fu−eegnet −0.005 (p=0.020)；
fu−K33 −0.005 (p=0.027)。**Holm（m=3）后全部不显著**（0.060/0.080>0.05）：
四深度臂在原文契约下依旧打平，K33 的 AUC 与 hit@5 均列深度臂最高。

### 7.2 paper-aligned steady-state Z0 successor

0.5 Hz/1200 ms 在同一 steady-state Z0 链上的 operational hit@5 为 `0.490`、
AUC `0.674`、coverage `230/245`；低于 0.1 Hz/1200 ms 的 `0.543/0.709`。
旧 zero-state M=5/R=5 表已删除，不再承担任何排序或校准结论。

### 7.3 双轨定稿

| 轨 | 契约 | 用途 | eegnet 锚点 |
|---|---|---|---|
| accuracy-development | v3（0.1 Hz） | 已查看 GTN cohort 的开发描述 | 0.7302/0.7918/0.739 |
| 原文线 | paper（0.5 Hz） | SOTA 对比表 | 0.7177/0.7633/0.722 |

原文线与开发线的历史 zero-phase 差距（ΔAUC 0.0125、Δhit_all 0.029）描述
paper-aligned profile 的代价。两条线都不是产品准确率；在线排序以独立的
steady-state causal matched arms 为准。

## 8. 当前 source-supervised causal 结论

旧 zero-state Gate 3-S1 与同 suffix 扫描数表已删除，避免继续充当 baseline。当前
独立审计的 0.1 Hz/1200 ms Z0 结果为：

| source arm | coverage | operational hit@5 | AUC |
|---|---:|---:|---:|
| no-QC | 230/245 | 0.424 | 0.665 |
| **QC100** | **230/245** | **0.543** | **0.709** |

QC100 的配对 operational 增益为 `+0.118`，95% CI `[+0.053,+0.184]`，因此
source QC100 已冻结；target-prefix QC 保持关闭。sum 与 mean 在每候选固定计数时
完全相同，不当作独立证据臂。linear `full_unfold` 已采用；当前只在同一完整链路下
三核结果已完成并报告 `hit@all_balanced`、raw `hit@all`、完整 hit@R 曲线及成本；
开发选择 K35，K65 保留强对照，K33 停止。

BI candidate-v2 已从 64 人 raw CSV/MAT 重建。合法监督校准由 BI cross-decision 与最终
BrainSync target-switch 数据裁决；GTN O5 永远只作 oracle proxy。
