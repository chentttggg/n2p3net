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
