# GTN 精度损失分解：工程契约审查与文献对照

日期：2026-08-30。状态：living analysis（A/B 受控实验进行中）。
触发：EEGNet 历史复现 AUC≈0.78 / decision hit≈0.80，第一轮 GTN 128Hz 地板只到
AUC 0.66 / hit_all 0.690 / hit@8 0.62。本文审查"数学自洽但工程过严"的举措。

## 0. 事实基线（先分清口径）

| 口径 | 数值（当前协议, eegnet, seed 28/29） |
|---|---|
| trial AUC | 0.664 / 0.661 |
| trial BACC | 0.611 / 0.609 |
| decision hit（全 repetition） | **0.690 / 0.694** |
| decision hit@8 | 0.604 / 0.624 |

历史 0.78/0.80 与当前差距 = AUC −0.12、hit −0.11。QC 排除量极小
（train drop 273/50k=0.5%，test excl 26），**QC 不是主因**。

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
- 判定：**修复（已实现 `--center-decision-logits`，A/B 验证中）。**

### S4. 128 Hz 重采样（嫌疑小）

- EEGNet/文献常在 128 Hz 下工作；采样率本身预计损失 <0.01 AUC。
- 但 256→128 与 S1/S2 交互（核宽缩放后频率响应不变，收益中性）。

### S5. fold-local artifact QC（已排除）

- 实测排除量 0.5% train / 26 test epochs。对 3 导数据的存在性问题是
  fail-closed 崩溃（已修），不是精度损失。

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

1. **预处理契约净效应 = ΔAUC −0.066、Δhit_all −0.102、Δhit@8 −0.139**
   （C1 − B0，单变量：l_freq 0.1→2 与 tmax 1200→800 两个默认值）。
   B1 复现 AUC 0.736/hit 0.780，C1 复现 0.730/**0.792**——历史 0.78/0.80
   完全可复现，**不是旧代码 artifact**。T 的质疑成立且被精确归因。
2. **128 Hz 采样率清白**：C1 − B1 = ΔAUC −0.006、Δhit_all +0.012，噪声级。
   紧凑采样率可保留。
3. **元凶锁定**：
   - **l_freq=2 Hz**：P300 主能量在 1–4 Hz，2 Hz 高通削峰并失真（Bougrain
     2012：[0.1,15] 最优，>0.1 Hz 开始失真 P300，1 Hz 已是最差档）；
   - **tmax=800 ms**：GTN 为 7–17 岁儿童，P3 潜伏期随发育递减（Goodin：
     6–15 岁以 18.4 ms/年递减；学龄儿童正成分 ~1000 ms 才回基线），
     800 ms 窗硬截断目标成分。
4. **S3（decision centering）被否定**：B2 − B0 = −0.012 hit_all。校准后
   LLR 已除常数偏置，centering 无益；D-center 决策记录适用于未校准 logit。
5. **QC 无罪**（排除量 0.5% train / 26 test epochs）；**采样率无罪**（C1−B1）。

### 2.2 意义

- 0.78 → 0.66 的"退步"**不是模型或训练问题，是 2026-08-28 冻结的输入契约
  对 GTN（7–17 岁、3 导）过严**。"先证契约"流程本身工作正常——它忠实执行了
  一个错误的默认值。
- 昨晚/今晨所有 128Hz/2Hz/800ms 协议下的数字（含 full_unfold vs ms 的
  GTN 判定、causal chance 水平）都要在修订契约后重估：2Hz 高通削掉的
  delta 能量对不同 readout 的影响可能不对称（full_unfold 的绝对时序模板
  对波形失真可能更敏感），**"full_unfold 被 GTN 否定"的结论暂缓，待
  修订契约复跑后再判**。
- hit@8=0.743（C1）与 0.85 目标的差距从"遥不可及"变为"一个契约修订 +
  transfer 预训练可争的距离"。

## 3. 修订决定（由 A/B 裁决）

1. **GTN/儿童契约**：`l_freq=0.1`、`tmax=1200`（即 C1 cache 契约）。生成
   causal 版 `--filter-phase forward` 后，GTN floor/causal 全部结论需在
   修订契约上复跑一轮再冻结。
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
（+0.008–0.012，待 seed 29 确认）。旧契约 2 Hz 下 fu 的劣势与 causal 下的
伪优势均为损坏信号上的 artifact。

### 4.2 causal M=5,R=5（n=175，seed 28）

| 臂 | AUC | hit@5 |
|---|---:|---:|
| ms_flatten | 0.5204 | 0.131 |
| full_unfold K65 | 0.4776 | 0.103 |
| full_unfold K33 | 0.5169 | 0.171 |

fu−ms = **−0.043（p=0.0037）**、fu−k33 = −0.039（p=0.0020）。**causal 排序
再次反转回 ms/k33 占优**——旧契约下 fu 的 causal 优势是损坏信号 artifact。
全臂仍近 chance：scratch + 45 trials 校准不可行的结论在修订契约下维持，
Gate 3（source-supervised 预训练 → 留出目标被试适配）动机不变。
注意 n=175 时三臂都挤在 0.48–0.52，单点显著要谨慎；causal 侧最终判定
应等多 seed 与 fold-local QC 补齐后再冻结。

### 4.3 契约修订落地

- `DEFAULT_GTN_DATA_CONTRACT` → `gtn_ms_eegnet_input_v3`（0.1 Hz / 1200 ms）；
  新增 `GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT`；
  `causal_prefix_suffix_split(contract=...)` 参数化（旧硬编码 assert 移除）。
- BI/成人默认契约 `DEFAULT_P300_DATA_CONTRACT` 未动，其 l_freq/窗宽由
  BI2014a 同款 A/B 另行裁决。
- 新 caches（云端，均带 record）：`gtn_sf128_lf0.1_tm-0.2_tx1.2_mean_zero/fwd.npz`。

## 5. 两 seed 终判（修订契约，LOSO 各 245 折）

| 模型 | AUC s28/s29 | hit@8 s28/s29 |
|---|---:|---:|
| eegnet | 0.7302 / 0.7297 | 0.739 / 0.755 |
| ms_eegnet | 0.7283 / 0.7251 | 0.731 / 0.747 |
| full_unfold K65 | 0.7291 / 0.7266 | 0.735 / 0.722 |
| full_unfold K33 | 0.7299（s28） | 0.759；hit_all **0.804** |

配对（fu − ms）：+0.0009 (p=0.64) / +0.0015 (p=0.48)；（fu − eegnet）：
−0.0011 (p=0.53) / −0.0032 (p=0.10)。**终判：修订契约下四臂（EEGNet、
ms_flatten、fu-K65、fu-K33）在 trial AUC 与 hit@R 上均无可靠排序**；
K33 的 hit_all 0.804 为全场最高，W2 的"压缩候选"在 GTN 上获得支持。
历史 0.78/0.80 的水平已恢复并小幅超越（hit_all 0.79–0.80）。
0.85 路线：契约修复贡献 Δhit@8 ≈ +0.13，剩余缺口归 Gate 3 transfer。

云端 runs：`gtn128rev_loso_floor_seed2026082{8,9}/`、
`gtn128rev_loso_k33_seed20260828/`、`gtn128rev_causal_m5r5_*_seed20260828.json`、
`gtn128rev_floor_analysis_seed2026{0828,0829}.json`。

## 6. 高通终值：原文 0.5 Hz vs 0.1 Hz（T 提供原文设定后补测）

GTN 原文（gtn_unet_2023）高通为 **0.5 Hz**。同代码同 seed 的 eegnet 对照：

| 高通 | AUC | BACC | hit_all | hit@8 |
|---|---:|---:|---:|---:|
| 0.1 Hz | 0.7302 | 0.6688 | 0.7918 | 0.739 |
| **0.5 Hz（原文）** | 0.7177 | 0.6557 | 0.7551 | 0.722 |
| 2 Hz（已废弃） | 0.6637 | 0.6109 | 0.6898 | 0.604 |

裁决：0.5 Hz 恢复大部分性能（vs 2 Hz：+0.054 AUC），较 0.1 Hz 低
0.0125 AUC / 0.037 hit_all，方向与 Bougrain 2012 一致。
**双轨决定**：
- 主契约 `gtn_ms_eegnet_input_v3` 维持 0.1 Hz（准确率优先，性能上限），
  作为 documented deviation 记录在案；
- 与原文 SOTA 的正式对比表将用 0.5 Hz 锚点臂重跑（EEGNet 锚点已有：
  0.7177/0.7551），保证预处理对齐的可比性。
原文窗宽待查：若原文亦为 1200 ms 则 v3 窗宽已对齐。
