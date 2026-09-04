# BI2014a → GTN 跨域迁移可行性裁决

日期：2026-09-04 | 触发：T 提问「BI 的数据做域迁移给 3 导的 GTN，而且情景不一样，这能不能行」
方法：只读契约核对 + 本地蒙太奇消融（12 人 BI cache，`tmp/l1_cloud/montage_ablation.py`）
状态：**裁决 = 当前 2 导 common-CAR 路径不进入活动迁移（科学阻塞）**，替代方案见 §5

---

## 1. 结论先行

**BI2014a（16 导，row_column 36 格）→ GTN（3 导，9 选 1 direct）的当前 2 导 common-CAR 路径，在科学上构成阻塞，不应进入活动迁移 promotion；这不是“所有 P300 都被数学判死”。**

- 「情景不一样」（任务语义）**不是**阻塞项——框架已有正式解法（§3）；
- 「通道不一样」**是**阻塞项：两数据集的通道交集只有 **2 导 {Cz, Pz}**，而跨域契约强制对交集施加统一平均参考（CAR）。2 导 CAR 等价于 `[(Cz−Pz)/2, (Pz−Cz)/2]`，只保留一个差分自由度，`(Cz+Pz)/2` 共模方向不可从输出恢复；本 cache 的组平均 class-difference wave 确实高度共模（`r = 0.9942`），但这不是所有 P300 的定理。

---

## 2. 三重失配逐条核对

| 维度 | BI2014a | GTN | 是否阻塞 | 依据 |
|---|---|---|---|---|
| **通道** | 16 导（Fp1,Fp2,F3,AFz,F4,T7,Cz,T8,P7,P3,Pz,P4,P8,O1,Oz,O2），512 Hz | 3 导（Fz, Cz, Pz） | **阻塞** | 交集 = {Cz, Pz}（BI 无 Fz，只有 AFz；`channel.py` L19 alias 表仅 T3/T4/T5/T6，不映射 AFz→Fz） |
| **参考** | 右耳垂 | 鼻尖 | 可处理 | `domain.py` 的 CAR 适配器即为统一参考而设计 |
| **任务语义** | row_column（12 候选 → 36 格） | direct（9 选 1 数字） | **不阻塞** | `project_binary_evidence_source_view` 把源域投影成纯二分类 P300 证据，剥掉候选/重复结构 |
| **范式物理** | 6×6 矩阵高亮闪烁 | 中央单个数字闪 | 部分 | ERP 形态/潜伏期有差异，属域漂移本身的一部分，可接受 |
| **evidence 深度** | rep 1–8，仅 5.2% selection 达 8 层 | 中位 >16 层，88.4% 支持 R≥8 | 偏向 GTN | `gtn_rep_census.json` |

**关键澄清**：语义失配（T 说的"情景不一样"）框架**已经设计覆盖**——源域走 `binary_evidence` 投影后只剩"这一闪是不是 target"，候选语义由目标域自己承载。这条路径是 `prepare_multidomain_source.py` 的一等公民（`--event-contract binary_evidence`）。当前阻塞来自公共空间采样和 2 导投影的秩亏，不是“情景”一词本身。

---

## 3. 框架既有跨域契约盘点（此前未通读，现补）

`src/data/domain.py`（schema `n2p3_common_channel_car/1`）：

- `common_channel_intersection` —— 取通道交集
- `ensure_common_channel_average_reference` —— **两侧施加同一组通道的 CAR**，带 provenance + lineage + 数值三重校验（L124–205）
- `project_binary_evidence_source_view` —— 剥掉 candidate/target/repetition 字段，投影为二分类证据视图（schema `n2p3_binary_evidence_source_view/1`）
- `annotate_source_domain` / `source_domain_ids` —— 显式域轴（schema `n2p3_source_domain_axis/1`）
- `namespace_epoch_dataset` —— 跨数据集拼接前的身份命名空间隔离

`experiments/prepare_multidomain_source.py`：

```
--source NS=cache.npz --source NS2=cache2.npz \
--target-channels "Cz,Pz" --event-contract binary_evidence --output merged.npz
```

流程：载入 → 统一 CAR → 命名空间 → 域标注 → 二分类投影 → 拼接。
**契约完整、可审计、设计意图明确。它的唯一前提是"存在一组有意义的公共通道"。BI×GTN 不满足这个前提。**

---

## 4. 实测：2 导 CAR 会删除多少 P300

脚本：`tmp/l1_cloud/montage_ablation.py`（只读，BI 12 人 cache，10884 epochs）
方法：280–600 ms 窗口均值作 score，**逐被试** target vs non-target 秩 AUC（无空间滤波、无时序模型——绝对值是朴素信噪比量度，只为**相对比较**）

| 蒙太奇 | 中位 AUC | 均值 | 范围 |
|---|---|---|---|
| (a) 原始右耳垂参考，mean(Cz,Pz) | **0.5440** | 0.5515 | 0.515–0.603 |
| (b) 16 导 CAR，mean(Cz,Pz) | 0.5082 | 0.5208 | 0.471–0.615 |
| **(c) 2 导 CAR，(Cz−Pz)/2** | **0.5047** | 0.5020 | 0.439–0.558 |
| (d) 原始参考，Cz 单导 | 0.5497 | 0.5473 | 0.493–0.606 |
| (e) 原始参考，Pz 单导 | 0.5338 | 0.5429 | 0.487–0.618 |

直接证据（组平均差异波，280–600 ms）：

- **Cz 与 Pz 的 P300 差异波相关 r = 0.9942**（几乎完全共模）
- 2 导 CAR 第一输出 `|(Cz-Pz)/2|` 的峰值 / 峰值 `|Cz|` = **0.1321**；原记录的 `0.2642` 漏除了 2
- 逐被试 AUC 损失（native minus CAR）：中位 **+0.0490**（0.544 → 0.505）；原记录写成 `−0.049` 是符号错误

**数学上能判死的部分**是信息可逆性：CAR 后两导之和恒为 0，选定集合的 `(Cz+Pz)/2` 方向在输出中不可恢复；下游模型只能使用差分方向。**不能判死的部分**是任务性能：若真正有用的 P300 判别信息主要在差分方向，任务仍可能工作，因此必须把该路径作为科学阻塞而非普适性能定理。

副产品观察：**16 导 CAR 也把朴素检测从 0.544 打到 0.508**。说明对这份 cache（右耳垂参考已合理）而言，额外 CAR 是净损害。若将来要走跨域路线，CAR 契约本身需要重新审视——不是"CAR 一定对"，而是"CAR 必须是同一组通道；`k>=2` 只保证代数可定义，通道数和任务信息保留仍需实证"。

---

## 5. 替代方案（推荐度排序）

### 方案 1（推荐）：GTN 单数据集内部定义域

- **通道零失配**（3 导统一、鼻尖参考统一）、**语义零失配**（9 选 1 direct 统一）
- 域 = **被试群**（242 人按采集批次或平衡随机切分；亦可按年龄/性别分层）
- 测的就是纯净的跨被试分布漂移 —— 正是 P0 Split-BN / P1 source residual 被设计要处理的对象
- **与产品线对齐**：GTN 的 Fz/Cz/Pz 是目标 8 导硬件方案的**子集**（Fz、Cz、Pz 均在 8 导内），9 选 1 语义与 BrainSync 目标一致
- 代价：GTN causal cache L0 构建 + direct 语义支持（框架硬门现仅 row_column，`candidate_task.py` L88）——direct 路线是产品线必经之路，一次投入长期复用

### 方案 2：放宽 CAR 契约，各自域内 CAR 后取共同导联

- BI 用 16 导 CAR、GTN 用 3 导 CAR，再各取 {Cz, Pz}
- 不再强制把 `Cz+Pz` 的选定集合共模投影为零（不像 2 导 CAR），但输出仍依赖各域全通道均值；两侧 CAR 的"均值"不同 → 通道语义仍跨数据集不一致
- 需改 `domain.py`（现有契约强制同组通道 CAR，L41–121）
- 风险：可解释性差——测到的迁移增益无法区分"真域适应"与"CAR 口径差"

### 方案 3：BI 仅作 source pretraining，空间层不迁移

- BI 投影为二分类证据（`project_binary_evidence_source_view`），只迁移时序/深层；GTN 侧空间层重新初始化
- 理论可行（经典做法），但 GTN 的 Fz 在 BI 中不存在 → 空间层不仅维度变，通道语义也变，属"重初始化 + 部分迁移"
- 工程量大、评估口径复杂，收益不确定

### 方案 4：BI 降级为历史证据线

- BI 的 LOSO 证据（EEGNet 0.7395 / full_unfold 0.7451）继续作为**架构选择**的证据，不参与跨域迁移
- 因果 cache 管线已在 BI 上跑通并完成工程验证，价值已兑现

---

## 6. 留给 T 的决策点

| # | 决策 | 选项 |
|---|---|---|
| **D-11** | **确认 hit@R 的 R 值不是规范** | T 已申明 @8 是随口说的指标，非合同。现代码中 `MIN_PROMOTION_HIT_AT_R = 8` 是**硬编码常量**（`run_candidate_promotion_matrix.py` L937），建议下放为 plan 层参数（dev 轮可填 2/3/4，promotion 轮仍受下限约束）——需要 T 拍板"promotion 轮的下限到底是多少"，我不擅自改 |
| **D-12** | 跨域路线取舍 | (1) 走方案 1（GTN 内被试群域，需 direct 语义 + L0 构建）；(2) 走方案 2（放宽 CAR 契约，保 BI→GTN）；(3) 走方案 4（BI 留作历史证据，等 BrainSync 真多 session 数据） |
| D-13 | GTN 域切分口径（若选方案 1） | (a) 按 Experiment 编号采集批次；(b) 平衡随机切分（预注册 seed）；(c) 按年龄/性别分层 |

---

## 7. dev64 两轮实际测的是什么（澄清：不是跨域）

T 追问「昨天晚上的训练用到跨域了吗？是不是用其他 BI 数据集来训练，剩下的一个 BI 集推理？」——
**都没有**。实际是**同一数据集内部的被试级留一（LOSO）**。

| 事实 | 证据 |
|---|---|
| 源与目标是同一个 cache 文件 | `target_cache` 与 `source_arms[0].cache` 同为 `l0_bi2014a_causal_v3_64subj.npz`（BI 只有这一个 64 人 cache，不存在"其他 BI 集"） |
| 每个 fold 训练时排除目标被试 | `target_subjects[i] = {target_subject: '01', source_holdout_subject: '01'}`，两者恒等 |
| 每个 fold **独立**训练一个模型 | `checkpoints/bi2014a_v3_dev/seed-*/` 每 seed **64 个 `.pt`**（dev-01…dev-64），3 seeds 共 **192 个**——非"训一次复用" |
| 有防泄漏策略 | `identity_exclusion_policy: "source_or_global"` |
| 跨域机制全关 | `source_domain_mass` / `splitbn_source_weights` / `selection_domain` 全 `null`；plan 名即 `single_domain` |

两臂（zero_shot / target_adapter）**共享同一个 source 训练**，差异只在目标侧：adapter 臂用目标被试的 5 个校准 selection（`calibration_selections: 5`）微调 adapter 头，5 epochs / lr 5e-4 / batch 16。

**含义**：这两轮测的是**跨被试泛化（within-dataset cross-subject）**，不是域漂移（domain shift）。
因此它们对 P0 Split-BN / P1 source residual 的验证价值为零（单域下这些机制连 `p=0` 都被 gate 拦掉）；
但 E=8 的 AUC 0.637 作为「跨被试 LOSO 基线」本身有效，可用于将来与域适应臂对照。

---

## 8. 附：本次未做的事（避免误解）

- 未改动任何 `src/` 代码，未改动 BI/GTN cache；
- 未验证 GTN 侧的实际 ERP 质量（无 GTN cache，需 L0 构建后才能做对称的蒙太奇诊断）；
- 朴素 AUC 的绝对值（≈0.54）**不代表模型性能**（历史全量训练 AUC ≈0.74），仅用于同方法下的相对比较。
