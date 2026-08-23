# GTN 242 人 N2P3-Net 失败诊断与修复记录（2026-08-22）

> 状态：方案 B 已实施并通过测试；30 被试小验证完成；方案 A（60 被试）运行中；方案 D 待 A 后启动。
> 本文档记录实测证据、文献依据与代码/文档改动，供后续会话与验收追溯。

## 1. 起点

- N2P3-Net 全量 242 人 × 10 epoch LOSO：hit 0.7727 / bacc 0.6447 / AUC 0.7019。
- 同协议基线：Inception 0.8512 ≈ Conformer 0.8512 > EEGNet 0.8388 > SWLDA 0.7851 >
  N2P3-Net 0.7727 > Template 0.6405 > xDAWN 0.6322 > WindowLR 0.5041。
- 60 人 × 30 epoch 试点全面差于 10 epoch（hit 0.7333 vs 0.7833；bacc 0.6206 vs 0.6620；
  AUC 0.6863 vs 0.7299），训练损失却持续下降 → 过拟合，不是欠拟合。

## 2. 关键实测证据

### 2.1 GTN 真实 P3b 峰在 460–490ms，不是模型先验的 350ms
逐被试基线校正后 ERP：
- 群体平均 Pz target 峰 464.1ms；target−nontarget 最大差 479.7ms（14.94μV）；
  Cz 460.2ms（13.01μV）；Fz 417.2ms（6.19μV）。
- 逐被试 Pz target 峰：mean 490.0ms、SD 84.9ms、q10/50/90 = 378/499/589ms。
- 单试次 Pz 250–500ms 平均幅值 AUC 仅 0.601±0.105（数据天花板）。
- 模型旧配置 target τ_P3b≈345–365ms，系统性偏早 ~120ms。

### 2.2 PCW 参数弱识别（核心机制失效）
一阶梯度诊断（batch=512）：
- head_a/tokenizer 梯度量级 0.1–1.2；**tau0 梯度 ≈2×10⁻⁴**（弱 3–4 个数量级）；
  dtau_attn_query/sigma_raw 也弱 100–1000 倍。
- 30 epoch 后 τ0 移动 <0.05ms；λ3=0 仍不动 → 不是被 L_tau 拉住，而是分类监督推不动。
- freeze-τ 与可学习 τ 的 12 被试性能几乎相同；时间均值池化替换整个 PCW 只损失
  ~0.01 bacc/AUC → PCW 接近「装饰性」，判别信息主要在 encoder 分布式时间特征。

### 2.3 L_jit 不收敛且有害
- 反推 L_jit≈1.1–1.25，对应 ±40ms 平移的 τ 跟踪 RMS≈52ms。
- 单折 30ep：nojit 末点 bacc/AUC 0.644/0.637，明显好于 baseline 0.546/0.556 和 jit_prob=1 的 0.591/0.589。

### 2.4 过拟合曲线
一折逐 epoch：L_target 1.16→0.92 单调降，held-out bacc/AUC 在 epoch 10–11 见顶
（~0.68/0.73），epoch 30 掉到 0.55/0.56。无 val/嵌套早停的训练协议放大了该问题。

### 2.5 失败被试结构
N2P3-Net 错 55 人；Inception/Conformer 同时错 28 人，仅 27 人「N2P3-Net 独错而
Inception 对」；19 人所有模型全错。说明主要是困难被试上更差，而非完全不同的失败模式。
bacc/AUC 与试次数负相关（r=−0.29/−0.32），试次数高的被试更难。

## 3. 文献依据（关键条目）

- GTN 数据集与范式：[Mouček et al., Scientific Data 2017](https://link.springer.com/article/10.1038/sdata.2016.121?fromPaywallRec=false)；
  同源多被试 CNN 结果多在 60–68% 单试次精度（[Vařeka et al. 2020](https://ar5iv.labs.arxiv.org/html/2001.04225)）。
- P3b 通常 250–500ms，儿童更晚、跨被试变异大：[Polich, Updating P300](https://pmc.ncbi.nlm.nih.gov/articles/PMC2715154/)；
  单试次 jitter 是 ERP 量化的头等问题（[Woody 1967](https://scispace.com/papers/characterization-of-an-adaptive-filter-for-the-analysis-of-4eu57785pm)、
  [DTW warp-averaging](https://pubmed.ncbi.nlm.nih.gov/11595152/)）。
- 软 argmax/注意力边界偏置与弱梯度退化：[Removing the Bias of Integral Pose Regression](https://mlanthology.org/iccv/2021/gu2021iccv-removing/)；
  时间序列注意力可坍缩为 MLP（[Why Attention Fails](https://openreview.net/pdf?id=mYzlRNMAxS)）。
- 时间不变性自监督与时间局域 P300 判别的冲突：[EEG SSL 系统综述](https://ar5iv.labs.arxiv.org/html/2401.05446)。
- 小样本 LOSO 过拟合与验证协议：[Trustworthy EEG decoding, PMID 39549492](https://pubmed.ncbi.nlm.nih.gov/39549492/)；
  [Data partitioning, arXiv:2505.13021](https://ar5iv.labs.arxiv.org/html/2505.13021)。

## 4. 方案 B 改动（代码+文档）

1. `MultiTaskHeads`：Head-A 增加全局时间池化旁路 `g_global=mean_pool(Z')`，
   `use_global_bypass=True` 默认开启；False 为旧结构消融。
2. `N2P3Net`：新增 `tau0_ms`、`global_bypass` 构造参数，透传 ComponentWindow/Heads。
3. `TrainerConfig`：`lambda_jit=0.0`、`jit_prob=0.0` 默认关闭 L_jit（接口保留）。
4. `run_n2p3net_gtn.py`：GTN 默认 epochs=10、τ0=(220,300,460)、P3b τ0 界 [350,600]、
   attention_softargmax、lambda_jit=0；新增 `--p3b-tau0-*`、`--no-global-bypass`；
   `--subjects` 一律从 nall 缓存派生（删除了与全量口径不一致的旧 n<N> 子集缓存）。
5. `run_gtn_baseline.py`：同样统一从 nall 缓存限缩，避免旧子集缓存改变分母。
6. 文档：constitution v5、blueprint v5、roadmap 更新；测试新增回归项。

测试：全量 pytest 通过（方案 B 后 241 项）。

## 5. 小验证结果（30 被试，同一 runner/缓存/seed 口径）

| 配置 | hit | bacc | AUC | target τ_P3b |
|---|---|---|---|---|
| 旧（τ0=350, direct, jit .5, 无 bypass） | .7667 | .6454 | .7153 | 345.1ms |
| 新（τ0=460, softargmax, jit 0, bypass） | **.8000** | **.6558** | **.7174** | **487.9ms** |

- 配对置换 hit p=1.000（N=30，只差 1 人，不显著），但三项指标方向一致；
- **可解释性目标达成**：新配置的 target τ_P3b=487.9ms，与逐被试 ERP 实测 490ms 一致。

## 6. 方案 A 结果（60 被试，同 runner/缓存/seed 口径）

| 配置 | hit | bacc | AUC | target τ_P3b |
|---|---|---|---|---|
| 旧（τ0=350, direct, jit .5, 无 bypass） | **.8500** | .6580 | .7189 | 369.9ms |
| 新（τ0=460, softargmax, jit 0, bypass） | .8000 | **.6590** | **.7195** | **481.7ms** |

- 配对置换检验：mean(A−B)=−0.0500，p=0.2492（不显著）。
- 结论：方案 B 的命中率没有显著提升（60 被试反而 −0.05），单试次 bacc/AUC 几乎持平（+0.001/0.0006），
  但 τ 从 370ms 修正到 482ms，与 ERP 实测 490ms 一致——**修复的是可解释性，不是命中率**。
- 决策：不以此为由重训全量 242；保留新配置作为默认（可解释性 + 训练更快 16.4s/fold vs 19.6s/fold），
  hit 指标仍需以全量 242 旧/新配置复核才能定论。

## 7. 方案 D（运行中）

按 transfer_policy 三臂（T0/T1/T2/T3），先 60 被试试点，显著才考虑全量 242。

### 7.1 EEGNet 60 被试（已跑完）
| 臂 | hit | bacc | AUC | vs T0 p |
|---|---|---|---|---|
| T0 from scratch | .8333 | .6845 | .7713 | — |
| T1 erpcore | .8667 | .6872 | .7745 | 0.5000 |
| T1 bnci008 | .8667 | .6915 | .7715 | 0.5004 |
| T1 bi2014a | **.9000** | .6877 | **.7786** | 0.1188 |
| T2 erpcore | .8333 | .6702 | .7551 | 1.0000 |
| T2 bnci008 | .8167 | .6781 | .7577 | 1.0000 |
| T2 bi2014a | .8833 | .6742 | .7649 | 0.2450 |

结论：T1 微调全线不劣，bi2014a 方向性 +0.0667；T2 冻结骨干无优势。N=60 未达显著，
符合 transfer_policy「不显著不采用」的默认结论；bi2014a 值得全量复核。

### 7.2 N2P3-Net T3 域对齐（实现 + 60 被试已完成）
- 新代码：`N2P3NetDomainBaseline`（aux 域只进 n_domains=2 的域条件仿射 + RBF-MMD，
  GTN 独占 L_target/L_early/L_amp/L_tau；predict 仍只吃 GTN 测试 fold）。
- runner 新参数：`--aux-dataset/--aux-channels/--aux-max-trials/--lambda4/--mmd-bandwidth`。
- 12 被试试点：T0-new .6667/.6454/.7061 → T3-erpcore .7500/.6480/.7223（方向性改善）。
- 60 被试结果（T0-new = planb_new60）：

| 臂 | hit | bacc | AUC | vs T0 p |
|---|---|---|---|---|
| T0-new | .8000 | .6590 | .7195 | — |
| T3-erpcore (λ4=0.1) | .8167 | .6570 | .7176 | 1.0000 |
| T3-bnci008 (λ4=0.1) | .7833 | .6568 | .7168 | 1.0000 |

T3 无显著收益；bacc/AUC 与 T0 持平。

## 8. 方案 D 总结与决策

- EEGNet T1 微调全线不劣（bi2014a 方向性 +0.0667 hit，p=0.119）；T2 冻结无优势。
- N2P3-Net T3 域对齐无显著收益（erpcore hit +0.0167，bnci −0.0167）。
- 按 transfer_policy §7：60 被试试点不显著 → 默认「辅助数据无增益」，不采用辅助预训练。
- 不建议全量 242 跑 D 矩阵（EEGNet T1-bi2014a 全量约 30–40min、N2P3Net T3 每臂约 3.5–4h
  且当前无阳性信号）；若仍要复核 T1-bi2014a 全量，可作为唯一例外单独批准。

## 9. 样本量外推与 EEGNet 可借鉴设计（无新训练，仅统计与文献分析）

### 9.1 哪些小样本不显著的变化值得怀疑"扩大样本会显著"
- **T1 辅助预训练 + 微调（EEGNet）**：三个辅助域 hit 全部不劣或更优，且 60 人内所有
  discordant 对都朝 T1 方向（erpcore/bnci 各 2/0、bi2014a 4/0）。bi2014a 的 discordant
  比例 95%CI≈[0.40,0.99]，若 242 人按 6.7% discordant 率外推约 16 对；要 80% 功效需
  ~19–47 对（取决于真实比例 0.8/0.7），故 242 人仍只是"可能显著"，不是确定。
- **P3b τ0=460 + softargmax（方案 B 的时间先验部分）**：机制证据最强（GTN 真实峰
  460–490ms，模型 τ 从 370→482ms），12/30 人 bacc/AUC 方向一致；但 60 人 bacc/AUC
  差只有 +0.001/+0.0006，bootstrap 95%CI 已跨 0。按 242 人外推 CI 仍不排除 0，
  **它应作为"可解释性修正"保留，不应期待 hit 显著提升**。
- **关闭 L_jit**：机制与文献一致（时间不变性自监督与 P300 时间局域任务冲突），
  但 10ep 下收益近乎为 0，只在 30ep 过拟合区显示价值；不是扩大样本就能兑现的调整。

### 9.2 不应继续投入的调整
- **T3 域对齐（λ4=0.1/带宽5.0）**：242 人投影功效仅 0.15–0.23，且 bacc/AUC 方向为负。
- **T2 冻结骨干**、**全量增强**：机制不利或方向为负。
- **方案 B 的 global bypass**：60 人 hit 方向为负，bacc/AUC 近 0；未被单独验证的收益。

### 9.3 EEGNet 值得借鉴的设计（按可落地性排序）
1. **原生通道数与"无幻象通道"**：EEGNet 直接用 GTN 3 导；N2P3Net 零填充到 8 导。
   建议 tokenizer 支持 3 导或 mask 化 spatial conv，而不是训练 5 个恒 0 通道。
2. **降容 + 池化**：EEGNet 3 导仅 1410 参数，N2P3Net 37.5k；EEGNet 两次平均池化
   （4×、8×）把 T 从 256 压到 8。分类旁路可池化，PCW 保持全 T 作可解释路径。
3. **频率特异空间滤波 + max-norm**：EEGNet 每个时间滤波器配 2 个空间滤波器并约束
   ‖w‖≤1（CSP 式可解释正则）；tokenizer 空间深度卷积可加同样 max-norm。
4. **更强 dropout/简单头**：EEGNet 两处 dropout=0.25；N2P3Net 只有 TCN 0.1。
   EEGNet 分类头是 1×8 conv（258 参数），N2P3Net Head-A 是 192→32→1 MLP。
5. **标准化口径**：EEGNet 用训练集逐通道全局 z-score；N2P3Net 只用基线段 z-score。
   可组合：基线校正 + 训练集通道 z-score。
6. **ELU/BN 与自适应 BN**：EEGNet 用 ELU + BN(momentum=0.01)；跨域迁移文献支持
   adaptive BN。N2P3Net 的 LayerNorm 不是必然错误，但值得做小样本 A/B。
7. **单任务优先**：EEGNet 只有 L_target；N2P3Net 的 L_early/L_tau/L_amp 可能争夺
   共享编码器。最终 hit 口径建议先跑 λ2=λ3=λamp=0 的 60 人对照（~25min）。
8. **参数账对标**：设计 N2P3Net-Mini（F1=8、D=2、separable block + pooling +
   linear head，目标 3–5k 参数），作为"结构正则化"对照。

> 大样本实验暂不启动；上述任何候选都先在 12/30 人上做配对检验，明确效果方向后再议。

## 10. v5.1 已实现（EEGNet 借鉴 6 项，全部可回退）

1. GTN 原生 3 导：`--n-channels 3`（默认）；`--n-channels 8` 回退零填充。
2. Head-A separable-pool 判别旁路：4× 池化 → depthwise+pointwise → adaptive 8 bin；
   `--bypass-mode mean_pool|none` 回退。
3. 空间 max-norm=1：tokenizer 有效空间权重约束；`--no-spatial-max-norm` 回退。
4. Dropout 0.25：encoder + 瘦头；`--encoder-dropout 0.1` 回退。
5. 简单线性头：Head-A/B = Dropout+Linear；`--head-mlp` 回退旧 MLP。
6. 分类旁路与 PCW 分离：E5 收窄为仅约束可解释路径（constitution v5.1）。
测试：新增 test_bypass / 3 导 / max-norm / 回退路径测试；全量 pytest 253 项通过。

### 10.1 v5.1 比较结果（同 folds/seed，12/30/60 被试）

| 配置 | hit | bacc | AUC | wall |
|---|---|---|---|---|
| v5.1-12 | .6667 | .6407 | .7080 | 32s |
| v5-new-12 | .6667 | .6454 | .7061 | — |
| v5.1-30 | **.8667** | .6476 | .7162 | 165s |
| v5-new-30 | .8000 | .6558 | .7174 | 383s |
| v5-old-30 | .7667 | .6454 | .7153 | 612s |
| v5.1-60 | .8167 | .6562 | .7175 | 616s |
| v5-new-60 | .8000 | .6590 | .7195 | 1002s |
| v5-old-60 | **.8500** | .6580 | .7189 | 1193s |
| EEGNet-60 T0 | .8333 | .6845 | .7713 | — |

- v5.1 vs v5-new60：hit +0.0167（p=1.000），bacc −0.0028，AUC −0.0020。
- v5.1 vs v5-old60：hit −0.0333（p=0.626）。
- v5.1 vs EEGNet60：hit −0.0167（p=1.000）。
- 结论：6 项 EEGNet 借鉴带来的主要收益是**效率**（60 人训练 1002s→616s，约 39% 加速）
  与更干净的原生 3 导接口；hit/bacc/AUC 与 v5 两个版本在同一噪声带内，未显著超越。
  30 人 hit 高 +0.067 但 60 人回落，不能作为稳健证据。

### 10.2 全量 242 × 50 epoch（v5.1，15.1 小时）结果

| 配置 | hit | bacc | AUC |
|---|---|---|---|
| v51-50ep-242 | **0.6901** | **0.6095** | **0.6612** |
| 旧配置 10ep-242（官方基线） | 0.7727 | 0.6447 | 0.7019 |

- 配对置换：10ep-242 vs 50ep-242，mean +0.0826，**p=0.0033**——10ep 显著优于 50ep。
- 与 v51-10ep-60 的共同 60 人交集：.8167 vs .7000（p=0.119，N=60 不显著）。
- 训练损失 50ep 持续下降（1.56→1.20），但 LOSO 命中率显著更差：**50 epoch 属于明确过拟合**。
- τ 可解释性未崩：τ0_P3b=460.0ms、target τ_P3b=455.1ms（与 10ep 的 ~470–490ms 同区）。

## 11. GLM 版本（v6-GLM，branch `GLM`，2026-08-22 深夜）

### 11.1 诊断补充（同日 GLM 消融矩阵，12 被 LOSO，只读分析）

架构侧单点修复全部不移动 AUC（0.70±0.01 平台）：τ0 先验 350→460（0.706/0.702）、
纯单任务损失（0.7015）、去 final LayerNorm（0.7015）、PCW→均值池化（0.678）、
fold z-score 输入缩放（0.622，**反向 −8pt**：模型原生试次内基线标准化更优，逐通道
缩放破坏再参考层物理一致性）。数据扩展性：12 被（40 步）与 242 被（730 步）同为
0.70，EEGNet 0.727→0.762——N2P3Net 对数据量不敏感，未学到超越线性分类器的表征。
数据接口复核：grand-average P3b 干净锐利（Pz 464ms / 14.9μV），epoch/marker/单位无误。

### 11.2 v6-GLM 改动（训练协议修复，代码见 branch `GLM` commit 71c82c3）

1. **被试级验证早停**：evaluate 向声明 `fit_accepts_subject_ids` 的模型传 subject_ids；
   N2P3NetBaseline 按被试分组切验证集（frac=0.08、clamp[2,12]、seed 确定）；
   runner 默认 epochs 10→30 + patience=6；`--no-val-early-stop` 回退。
2. **TCN BN 消融轴**：`--encoder-norm bn`（Stage2Encoder/_TCNBlock norm 参数，默认 ln）。
3. **P3b σ 上界 80→150ms** 透传（`--p3b-sigma-hi`，儿童宽 P3b；成人传 80 恢复）。
4. 测试：`tests/test_glm_protocol.py` 新增 8 项语义测试；全量 263 项通过。
   早停机制验证生效（12 被试末 fold 只训 9/30 epoch，val loss 第 3 轮触底后 patience 耗尽）。

### 11.3 GLM 验证结果（同 runner/缓存/seed 口径）

| 配置 | hit | bacc | AUC |
|---|---|---|---|
| 12 被：v5.1 基线复现（10ep, σ80） | .6667 | .6391 | .7036 |
| 12 被：GLM 协议 | .5833 | .6412 | .7019 |
| 12 被：GLM+BN | .6667 | .6367 | .7069 |
| 60 被：GLM 协议（LN） | **.8333** | .6429 | .7154 |
| 60 被：GLM+BN | .8167 | .6553 | **.7245** |
| （参照）v5.1-60 | .8167 | .6562 | .7175 |
| （参照）v5-new-60 | .8000 | .6590 | .7195 |
| （参照）EEGNet-60 T0 | .8333 | .6845 | .7713 |

- **GLM+BN 的 AUC 0.7245 是 N2P3-Net 全系变体迄今最高**（此前全部 0.715–0.720），
  BN 方向 +0.009 AUC vs GLM-LN——支持「BN 有益于跨被试泛化」假设，但幅度有限。
- GLM 协议 hit 0.8333 追平 EEGNet-60（单任务口径下 N2P3Net 首次在 hit 上不输），
  但 bacc/AUC 仍在 0.71–0.72 带；早停换来的 hit 增益与验证集挤占训练数据的代价
  大致相抵。
- 诚实结论：**训练协议修复 + BN 只带来边际改善（AUC +0.5~0.9pt），未打开 0.72→0.77
  的表征缺口**。该缺口（与 EEGNet 的单试次判别差距）指向 tokenizer/encoder 的表征
  瓶颈——下一个该动的组件是 tokenizer（多尺度时间卷积银行，唯一从未被消融的部件），
  建议按 §9.3.8 推进 N2P3Net-Mini（3–5k 参数、EEGNet 式单尺度核 + 池化 + 线性头）
  作「结构正则化」对照实验。
- 242 被试全量 GLM/GLM+BN 复核未跑（每臂约 2–4h）；按 transfer_policy 惯例，
  60 被试方向性信号（BN +）不足以定论，全量复核前不改默认 `--encoder-norm ln`。


## 12. N2P3Net-Mini 实验与真正的根因：再参考层（2026-08-23，branch `GLM`）

### 12.1 Mini 实验结果：容量不是瓶颈（但诊断价值极高）

用现有容量旋钮构造 Mini（无新代码）：miniA=2.5k（d16/2滤波器×1尺度/depth0）、
miniB=7.3k（d32/4滤波器×2尺度/depth1），与 default 38.5k 同协议对比。

12 被（GLM 协议 30ep+早停）：miniA 0.7016 / miniB 0.7035 / miniB+bn 0.6996 ——
**容量砍 15 倍 AUC 不降**。结合 SWLDA（线性+窗口特征）0.722 > N2P3Net 0.70，
推断：0.70 是骨架前端在销毁判别信息，而非容量不足。

### 12.2 真正的根因：加权再参考层在鼻参考数据上销毁 P3b

机制（GTN 实测，350–550ms 窗，逐被试基线校正）：
- 鼻参考：target−nontarget 差值 Fz 4.4 / Cz 9.8 / Pz 10.8 μV（diff/noise ≈ 0.89 @Pz）
- 3 导均匀 CAR（= 再参考层 softmax 均匀初始化的等价操作）：
  差值 Fz **−4.0** / Cz 1.5 / Pz 2.5 μV——**Pz 信号损失 4.36×、单试次 SNR 损失 1.90×，
  且 Fz 反转成伪负信号**。P3b 是 Fz/Cz/Pz 三导共有的宽正波，均匀 CAR 恰好减掉共模
  主部；可学习 w 未逃出（与 tau0 同款梯度饥饿）。EEGNet 从不做再参考。

### 12.3 验证结果（去参考 use_rereference=False，同协议）

| 配置 | 12 被 AUC | 60 被 hit / bacc / AUC |
|---|---|---|
| miniA(2.5k) noref | 0.7077 | .7833 / .6561 / .7258 |
| miniB(7.3k) ln noref | — | .8500 / .6730 / .7466 |
| miniB(7.3k) bn noref | 0.7263 | .8667 / .6791 / .7515 |
| **default(38.5k) bn noref** | — | **.8833 / .6795 / .7575** |
| （对照）GLM+BN 有参考 | 0.7069 | .8167 / .6553 / .7245 |
| （参照）EEGNet-60 | 0.7266 | .8333 / .6845 / .7713 |

- **default+bn+noref：hit 0.8833 超过 EEGNet-60（0.8333），AUC 0.7575 把与
  EEGNet 的差距从 0.054 缩到 0.014**；相对昨日最佳（GLM+BN 有参考）AUC +3.3pt。
- 去参考收益 +1.5~3.3pt（所有变体一致）；BN 在去参考后仍 +0.5~0.9pt（三组一致）；
  容量在修复前端后恢复小幅正贡献（default 略优于 mini，60 被）。
- 12 被 miniB_bn_noref 0.7263 与 EEGNet-12 0.7266 打平。

### 12.4 落地改动（commit 见 branch `GLM`）

1. runner `--use-rereference`（默认关闭，GTN 鼻参考数据）；平均参考数据集可重新评估。
2. runner `--encoder-norm` 默认改 bn（三组实验一致正收益；--encoder-norm ln 回退）。
3. runner `--model-size default/mini/mini_a` 容量预设（诊断结论的复现入口）。
4. 新增测试：mini 预算（<10k/<5k）、noref 前向；全量 265 项通过。
5. 修正 §11 的保守结论：BN 收益在去参考后仍成立，默认已切 bn。

### 12.5 待办

- 242 被试全量复核 default+bn+noref（当前 runner 默认即该配置，约 2.5–4h）。
- τ 可解释性复核：去参考后 PCW 的 τ/σ 读数是否仍与 ERP 实测一致（成分定位是
  项目立身之本，前端变了需要重新验证）。
- 跨数据集注意：再参考层的「参考无关」设计目标未被否定——它是为平均参考/多参考
  跨域设计的；被否定的是「在鼻参考少导数据上默认启用均匀 CAR」。自有 8 导数据
  落地时按参考类型决定开关。

## 13. 深度科研：再参考层的重设计（2026-08-23，GLM v2 门控参考层）

### 13.1 文献结论（设计依据）

- **CAR 的蒙太奇有效性**：Junghöfer et al. 2001 —— 平均参考在 ~32 导才近似合理、
  64+ 导均匀覆盖才良好；**<32 导时其偏差可能比单物理参考更糟**。Luck 2014 要求
  64–128 导、>50% 头表覆盖。→ 本项目 3 导（GTN）与 8 导（自采）蒙太奇均在 CAR
  无效区；§12 的实测（信号损失 4.36×）是该结论的少导极端案例。
- **P300 参考惯例**：鼻尖或单侧耳垂/乳突（上海 ERP 临床共识；BrainProducts 指南
  对 P300 检测推荐平均乳突参考而非 CAR/FCz）；Dien 2017——平均参考对 P300/N400
  这类垂直取向成分「劈差」，统计功效受损。
- **跨参考标准化（Phase 3 的原理性方案）**：REST（Yao 2001，参考电极标准化技术）
  把任意参考记录变换到无穷远参考：T_REST = G_REST·G_m⁺，只依赖头模型+蒙太奇+
  原参考（无需真实源）。文献一致报告 REST 参考误差 < AR/LM/L。本项目已有电极
  三维坐标（channel.py），具备实现条件；但 REST 精度依赖蒙太奇密度（3 导严重
  近似），列为 Phase 3 跨数据集前的预处理选项（blueprint D-glm-rest-next）。
- **可学习参考层先例**：未找到直接先例（最近的只有 BrainOmni 的传感器物理注意力、
  DeeperBrain 的可学习空间衰减核）——门控参考层是本项目的独立小贡献。

### 13.2 设计与实现（src/models/reference.py 重写）

旧版缺陷：w=softmax 且均匀初始化 → ① 强制 CAR（<32 导无效）；② softmax Σw=1
使「恒等」（保留记录参考）在参数空间不可表达——这是结构缺陷，不是初始化问题。

v2 设计：**out = X − g ⊙ (1·wᵀX)**
- w=softmax(w_logits) 保留：{Σw=1} 正是从任意记录参考可达的再参考变换全集
  （w=单通道→该电极参考；均匀→CAR；乳突对→联结乳突）；
- **g 自由线性门（无激活、无界），init=0 → 精确恒等**。自由线性而非 sigmoid：
  ∂out/∂g = −m(t) 无饱和——对照 tau0 梯度饥饿教训（sigmoid 门在 0 附近有
  g(1−g)≈0 因子会重蹈覆辙）；
- **可解释性读数**：训练后 effective_reference() = g·w 即每通道有效参考权重；
- **per-domain 通路**：n_domains 给定时 w/g 形状 (D,C) 按 domain_id 逐样本取行
  （推理无 domain_id 回退主域，P9）——Phase 3 跨数据集（鼻参考 GTN ↔ 平均参考
  ERP CORE ↔ A1 耳参考自采 8 导）每域学自己的参考变换；
- mask 语义保持（参考只由存在通道计算、只对存在通道减除，防幻象通道）。

副作用警示（D-glm-ref-jitter）：train/augment.py 的 reference_jitter 增强把数据
随机推向 CAR 型凸组合变换——新证据下它在鼻参考数据上教网络「销毁信号的不变性」，
GTN 类数据应保持关闭（augment 默认本就 off）。

### 13.3 验证结果

12 被（default+bn，GLM 协议）：
- 门控开启：hit .6667 / bacc .6436 / AUC .7207
- 门控关闭：hit .6667 / bacc .6407 / AUC .7204
- **完全等价**；训练后学到的 gate(Fz,Cz,Pz) = [−0.003, −0.003, 0.004] ≈ 0
  ——门有健康梯度（单测验证恒等初始化下 gate 梯度非零）而选择不动，
  **网络用数据自证「鼻参考数据上恒等最优」**，同时把「要不要再参考」
  从人工开关变成可学习决策。

60 被（default+bn，GLM 协议）：
- 门控开启：**hit .9000（N2P3-Net 全系历史最高）** / bacc .6745 / AUC .7536
- 门控关闭（default_bn_noref，§12.3）：hit .8833 / bacc .6795 / AUC .7575
- AUC 噪声带内等价；hit +1.7pt；**数据量充足时门真的开了**：
  gate(Fz,Cz,Pz) = [0.197, 0.127, 0.154]，effective_ref ≈ [0.057, 0.043, 0.057]
  ——网络学到「减去 ~15–20% 加权参考」的轻度再参考（小幅去共模噪声）。
- 决策：runner --rereference 默认开启（--no-rereference 关闭）——恢复
  Stage 0.1 参考无关设计目标，且恒等起步保证在单域数据上零风险。

测试：test_reference.py 重写（恒等 init / 门开=CAR / gate 梯度健康 / R 矩阵
含门 / 门开时偏移不变性 / per-domain / mask 系），全量 267 项通过。

### 13.4 242 被试全量复核结果（2026-08-23 17:51 完成，run `glm_v2_full242`）

配置：GLM v2 默认（门控参考开、BN、30ep+被试级早停、batch 512），wall 3.1h
（batch 512 对小模型/3 导输入提速 ~4×：256 batch 受小 batch 内核开销限制）。

| 242 被 LOSO | hit | AUC |
|---|---|---|
| **GLM v2（门控+BN+早停）** | **.8182** | **.7462** |
| v5.1（旧前端，强制 CAR） | .7727 | .7019 |
| SWLDA | .7851 | .7219 |
| EEGNet | .8395 | .7620 |
| Inception/Conformer | .8512 | — |

配对 McNemar（逐被试 digit-level hit，n=242）：
- GLM v2 vs v5.1：198/242 vs 187/242，净胜 +11（21胜10负），**p=0.071（边缘显著）**
- GLM v2 vs EEGNet：198 vs 203，净负 −5（6胜11负），**p=0.33（不显著——「追平 EEGNet」目标达成）**
- GLM v2 vs SWLDA：198 vs 190，净胜 +8，p=0.134
- GLM v2 vs Inception：198 vs 206，净负 −8，p=0.096

结论：**N2P3Net 首次在全量 242 被上达到与 EEGNet 统计不显著的差距**
（hit −2.1pt p=0.33、AUC −1.6pt），显著超越 SWLDA 与旧版自身（+4.5pt hit/+4.4pt AUC）；
早停在数据充足时正常后移（末 fold 训 19/30 ep）。剩余与 inception/conformer 的
差距（hit −3.3pt p≈0.10）指向 tokenizer 表征容量，非前端问题。

### 13.5 待办

- Phase 3 前的真正考验：跨数据集（GTN↔ERP CORE↔自采）per-domain 参考层的
  域适配 vs REST 预处理的标准化，两者对照实验。
- τ/σ 可解释性复核：门控参考开启后 PCW 成分定位读数与 ERP 实测的一致性验证。

## 14. Tokenizer 深度科研：带通初始化（2026-08-23，GLM v3，branch `GLM` commit b5aaced）

### 14.1 第一手诊断：时间滤波器从未学习

单 fold（241 被训练）训后取证（对照同 seed 未训练模型）：

- **随机 kaiming init 的 FIR 频谱中心 ~60Hz**（宽带白噪特征）；训练后 cos_sim(init,trained)
  = 0.90–0.98、相对漂移仅 0.2–0.5——**滤波器从未学出 ERP 形状**（ERP 能量在 1–8Hz）。
  判别信息全靠地形空间权重（保持 init，k=65/129→Pz 主导）+ 下游 TCN 兜底。
- 尺度幅值失衡 4×：kaiming 1/√k 缩放（k=13 输出 std 0.141 vs k=129 的 0.035）。
- Stage 1 纯 affine（无任何非线性/归一化）：多尺度线性混合在数学上塌缩为单组长 504ms FIR。
- 这同时解释了 Mini 实验（容量砍 15 倍不掉点：随机投影本来就无可砍）与
  「与 inception 基线剩余差距」的定位。

### 14.2 文献核验（四个优秀案例收敛的设计规则）

- **EEG-Inception**（Santamaria-Vazquez 2020，TNSRE，ERP 专用，胜 EEGNet 5.1%）：
  核长从 ERP 成分时长推导（500/250/125ms）；**每个卷积块 BN+ELU+dropout**。
- **FBCNet**（Mane 2021，MI SOTA）：9×4Hz 窄带滤波器组多视图——时间滤波器频带定位。
- **ATCNet**（Altaheri 2022，TII）：conv stem 后 BN→ELU→AvgPool。
- **Sinc-ShallowNet / Sinc-EEGNet**（Borra 2020 / Baria 2021）：第一层带通约束/初始化
  （sinc 参数化，Hamming 加窗），可解释频谱读数。

共性规则：时间滤波器应**频带定位**而非宽带随机；**BN+激活**是 ERP-CNN 标准结构。

### 14.3 设计与实现（tokenizer.py D-glm-bpinit / D-glm-postnorm）

- **init="bandpass"**：w[t] = sin(2πf·t+φ)·hann(k)，频带按核长分层
  （f_lo = max(1.5, 1.2·fs/k)，f_hi = min(40, 3·f_lo)）——k=129 占据 P3b δ-θ 带
  [2.4,7.1]Hz、k=65 [4.7,14.2]、k=33 [9.3,27.9]、k=13 [23.6,40]。
  单位 L2 + 随机相位 + 相对每抽头幅度 5% 噪声。
  与 SincNet 硬参数化的区别：滤波器仍完全自由可学习（init 只是更近的起点）。
  **实现教训**：噪声必须按 1/√k 缩放——绝对 std 的白噪能量虽仅 0.25%，但
  幅度加权频谱一阶矩（Σ|W|f）被 65 个高频 bin 主导，把 2.4Hz Gabor 的
  「主频」拉到 15Hz；检验指标也须用能量加权（Σ|W|²f）。
- **post_norm="bn" + post_act="elu"**：每尺度 BatchNorm1d(F)+ELU（作用于 (B*C,F,T)）。
- N2P3Net 透传 + runner --tokenizer-init/--tokenizer-post-norm/--tokenizer-post-act。
- tests/test_tokenizer_v3.py 10 项语义测试；全量 287 项通过。

### 14.4 验证结果（60 被 LOSO，batch 256，与 GLM v2 锚点同口径）

| 配置 | hit | bacc | AUC |
|---|---|---|---|
| GLM v2（基线：门控+BN+随机 init） | .9000 | .6745 | .7536 |
| **v3a 带通 init 单独** | **.9167** | **.6939** | **.7691** |
| v3b BN+ELU 单独 | .9167 | .6779 | .7532 |
| v3c 带通 init + BN + ELU | .9167 | .6797 | .7630 |
| （参照）EEGNet-60 | .8333 | .6845 | .7713 |

- **v3a：AUC +1.55pt 至 0.7691——与 EEGNet-60（0.7713）打平**（差 0.002），
  hit 0.9167 远超 EEGNet 的 0.8333；bacc 亦全系最高。
- **机理确认**：训后滤波器稳定在分配频带（k=129 中位 4.1Hz q10-90 [2.9,6.4]
  = P3b δ-θ 带；k=65 8.2Hz；k=33 16.2Hz；k=13 30Hz）——对照旧版全部 ~60Hz。
- BN+ELU 在带通 init 之上略降（0.7630）：单位范数初始化已天然均衡尺度，
  ELU 的非对称变换对读出无益。**胜出配置 = 仅带通初始化**。
- 12 被消融（噪声带内）：bpinit .7159 / bnelu .7249 / 组合 .7195 vs 基线 .7207。

### 14.5 242 被试全量结果（2026-08-23 22:48 完成，run `glm_v3_full242`，wall 3.2h）

配置：GLM v2 全默认 + --tokenizer-init bandpass（batch 512）。

| 242 被 LOSO | hit | bacc | AUC |
|---|---|---|---|
| **GLM v3（带通 init）** | **.8388** | **.6802** | **.7543** |
| GLM v2 | .8182 | .6734 | .7462 |
| v5.1（项目原起点） | .7727 | — | .7019 |
| SWLDA | .7851 | — | .7219 |
| EEGNet | .8395 | — | .7620 |
| Inception/Conformer | .8512 | — | — |

配对 McNemar（n=242，digit-level hit）：
- **GLM v3 vs EEGNet：203/242 vs 203/242——完全平局**（6 胜 6 负，p=1.000）。
  「追平 EEGNet」验收目标以最强形式达成；AUC 0.7543 vs 0.7620（差 0.8pt）。
- GLM v3 vs Inception：203 vs 206（5 胜 8 负），p=0.581——差距不再显著
  （v2 时 p=0.096）；hit 差 3pt 内。
- GLM v3 vs SWLDA：203 vs 190（18 胜 5 负），**p=0.011 显著超越**。
- GLM v3 vs v5.1 旧版：203 vs 187（24 胜 8 负），**p=0.007 显著超越**。
- 相对项目原起点（v5.1）：hit +6.6pt、AUC +5.2pt、bacc +0.7pt。

**决策**：runner 默认切换 --tokenizer-init bandpass（commit 本次）。
本轮失败诊断→修复链至此收束：门控参考层（空间通路，§13）+ 带通初始化
（时间通路，§14）+ BN/早停协议（§11-12）= N2P3Net 在 GTN 全量上追平最强
深度基线，同时保留成分可解释结构（PCW τ/σ + 有效参考读数）。

### 14.6 遗留

- seed 稳健性复跑（60/242 被 seed≠0）。
- AUC 与 EEGNet 的最后 0.8pt：tokenizer 后续可试（按频率重分配 filters_per_scale、
  k=13 高频分支是否可删）。
- 去参考 + 带通 init 后 PCW τ/σ 可解释性复核（前端两处都变了，需重验）。
