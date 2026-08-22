# N2P3-Net 深度设计蓝图（Blueprint）

> 本文档是 N2P3-Net 的施工蓝图，把 constitution 的原则（P/E 系列）落到可实现的张量流、模块、损失与超参。
> 张量形状、损失项、设计决策均在此钉死；实现以本文档为准，与本文档冲突需回改本文档。
> 版本：v6-GLM（v5.1 + 训练协议修复：被试级验证早停 + 30ep 预算、TCN BN 消融轴、
> P3b σ 上界 150ms（儿童宽 P3b）；架构不变，全部可通过 CLI 回退）。

## 0.5 GLM 协议决策记录（2026-08-22，v6）

D-glm-early-stop   训练协议：固定 10ep → 30ep 上限 + 被试级验证早停。证据：
                    失败诊断 §2.4 一折曲线显示 held-out 指标在 epoch 10–11 见顶后崩塌
                    （30ep bacc 0.68→0.55），固定 epoch 是「欠拟合赌博」；且「10ep 过拟合」
                    的结论来自小训练池（12/60 被）观察，从未在 242 被试全量 + 验证协议下验证。
                    早停锁定每 fold 的 val 峰值而非全局赌一个 epoch 数。验证集按被试分组切
                    （frac=0.08、clamp [2,12]，evaluate 传 subject_ids），试次级随机切分会有
                    同被试泄漏、验证损失高估泛化。Trainer 已支持 val_loader/早停，本次接入。
D-glm-bn           TCN block 归一化消融轴（norm=ln/bn，默认 ln 不变）。依据：跨被试 P300
                    文献反复报告 BN 是 CNN 泛化关键（Värbu 2020：ELU+dropout+BN 与最佳
                    CNN 性能相关）；LN 逐 token 归一化抹平单 token 幅值维度。bn 供 A/B 实测。
D-glm-sigma-hi     P3b σ 上界 80→150ms（GTN runner 默认）。依据：GTN 儿童的 P3b 宽达
                    300–650ms（ERP 实测 350–650ms 窗差值仍 11–14μV、逐被试峰潜伏期
                    SD 85ms），旧上界 80ms 使 PCW 窗物理上盖不住真实成分。成人数据传 80 恢复。
D-glm-evidence     2026-08-22 GLM 消融矩阵（12 被 LOSO）：架构侧单点修复（τ0 先验、读出
                    路径、PCW 移除、final-LN 移除、辅助损失开关、fold z-score 输入缩放）
                    全部不移动 AUC（0.70±0.01 平台；z-score 反而 −8pt）；v5.1 的 6 项 EEGNet
                    借鉴同样在同一噪声带。剩余可测假设按优先级：训练协议（本版）→ BN（本版）
                    → 表征瓶颈（tokenizer，未动）。

## 0. 张量与符号约定

- fs = 256 Hz（重采样后）。
- 试次窗口：刺激锁时 [-200, +800] ms，共 1000 ms → T = 256 点。
- 通道 C = 8（固定顺序）：Fz、Cz、P3、Pz、P4、PO7、PO8、Oz。
- 隐藏维 D = 64。
- 输入 X ∈ R^{B×C×T} = R^{B×8×256}，B 为 batch。
- 参数预算：Phase 2 单受试 ≤ 50k；Phase 3 跨域 + 自监督 ≤ 100k 且须归因收益（E4 落地）。

## 1. 关键设计决策（三思记录，指导后续工程）

D1 「9 选 1」是决策层，不是单试次 softmax。
   单试次任务只能是「target / non-target 二分类」（sigmoid），因一个试次只对应一个刺激数字 d，
   监督只有 1 bit（d 是否等于心选数字）。「猜中数字」= 对每个数字 d 累加其所有试次的
   p(target) 后 argmax。禁止单试次做 9 类 softmax。

D2 网络外允许项（修订版）：重采样 + 连续域高通（默认 0.1 Hz）+ epoch 切分 + 阈值法伪迹剔除。
   高通必须在「连续数据」上做（epoching 之前），因低截止频率 FIR 冲激响应达数秒，在 1 s
   epoch 内做在数学上无效且引入边缘伪影（review 1.3）。伪迹剔除是数据质量步骤，非特征工程。

D3 时间分辨率在**可解释路径**全程保留（E5 落地，v5.1 收窄）。Stage 1 卷积 stride=1 + padding=same，
   PCW 与 τ/σ 读数吃全 T 的 Z'；判别旁路（Head-A bypass）允许 EEGNet 式池化。

D4 潜伏期由参数化成分窗显式生成（τ 被直接监督，D8），监督 = 分类 + τ 软约束（L_tau）；
   方案 2 峰值弱标签仅作 τ0 初始化兜底（E3）。

D8（根治 —— 因果反转，参数化成分窗 PCW）。
   旧方案「free attention A → 事后 soft-argmax 得 τ」有结构性缺陷：H 与 τ 是 A 的两个并行统计量
   （加权和 vs 期望），分类只监督 H、不监督 τ 的绝对位置（软对齐的平移等变性使 H 对 A 整体平移
   近似不变）。这是因果反了，非调参问题，故不再用「诊断 + 升级」补救，直接换策略。
   根治：反转因果——先估计潜伏期 τ，再用 τ 生成参数化软窗 A(t)=Gauss(τ,σ)，软对齐读取 H。
   τ 从「事后统计量」变为「生成参数」，被分类经 H→A→τ 直接、单调监督（∂L/∂τ ≠ 0 且方向正确），
   逐试次 latency jitter 可被 τ 自然学到。
   由此简化：不再需要熵锐化（A 形状由 τ,σ 决定）、JSD 防坍缩（不同成分 τ0 不同，位置天然区分）、
   显式时移（无 circular 伪影）。Phase 2 仍保留模拟数据诊断 MAE(τ,τ_true)<40ms 作为验证（非补救）。

D9（不对称高斯窗 —— 优雅解形状受限 + σ 退化）。
   对称高斯窗两个缺陷：(1) 真实 P3b 不对称（上升 ~100ms 陡、下降 ~250ms 缓），对称高斯系统性
   偏估 τ（偏向对称质心而非峰值）；(2) σ 自由退化（→0 成脉冲、→∞ 成均匀）。
   优雅解（参数化本身升级，非补丁）：不对称高斯窗——左右独立宽度 σ_up/σ_down，σ 经 sigmoid
   软映射到 [20,80]ms（有界无 clamp）；τ 处 sigmoid 平滑过渡（处处可微）。τ 仍直接 = 峰值
   （左右对称中心在 τ，无偏斜耦合），符合 ERP「峰值潜伏期」惯例。
   比偏斜正态更优：偏斜正态 τ 非峰值（偏斜拉均值）+ 需 erf；左右 σ 的 τ 即峰值、无耦合、计算简单。
   退化兼容：σ_up=σ_down 退化为对称高斯（可作初始化/消融）。

D5（新增，人群假设）被试以成人为主，可采集年龄/性别等元数据。
   年龄/性别作为「subject metadata 嵌入」与坐标嵌入并列输入；P300 潜伏期随年龄递增
   （成人亦如此，Depuydt 2023），故先验中心数据驱动初始化并允许年龄协变量进入 Phase 4 回归。
   GTN（7–17 岁儿童）从「同源主数据」改定位为「跨年龄迁移源域」，其与成人目标存在年龄域差，
   跨数据集验收须增加「GTN 内部年龄分层」中间档。

D6（新增，容量）Stage 2 序列编码降级为消融轴 depth ∈ {0,1,2,3}，默认 3 层膨胀 TCN（2026-08-20
   决策，原「默认 depth=1 轻量 Conformer」因实测参数 58k 超 E4 50k 被否决，见 constitution E4 修订）。
   Stage 1 多尺度卷积已提供 ~500 ms 局部上下文，序列编码要补的只剩 N2→P3b ~150 ms 关系，
   ROI 低；GTN 上 CNN≈LDA 亦指向「容量非瓶颈、域差才是」。预算 ≤50k。

D7（新增，归一化）全程 InstanceNorm/LayerNorm/GroupNorm，不使用 BatchNorm；故删除 Split-BN
   （BN 不存在时 Split-BN 是空操作），跨域对齐改用「域条件仿射」（per-domain 可学习 scale/shift
   加于 LayerNorm 后）+ 特征级 RBF-MMD + 目标加权损失。

D10（新增，辅助数据协议，P9）其他 P300 数据只允许两种参与方式：
   (a) 辅助域 target/non-target 预训练特征提取器，保存 checkpoint；每个 GTN fold 从该初始化
       开始、用 GTN 微调；
   (b) 共享 Stage 0–2 编码器，辅助域只参与域条件仿射与 L_MMD；L_target/L_early/L_amp 只在
       GTN 试次上计算。
   禁止辅助试次与 GTN 拼接后联合优化主分类；禁止辅助梯度进入分类头/决策层。实验须按
   doc/transfer_policy.md 的 T0/T1/T2/T3 四臂协议执行。

## 2. Stage 0 —— 格式无关适配（Format Adapter）

网络外（MNE，均在 epoching 之前）：
- 重采样 → 256 Hz；
- 高通 0.1 Hz（连续域，去极慢漂移；0.5 Hz 仅作消融对照，见坑位对照 E1）；
- epoch 切分 [-200, +800] ms；
- 阈值法伪迹剔除（默认 ±100–150 μV，剔除率作为数据集元信息记录）。

网络内（可微）：
- 0.1 加权再参考：R = I − 1·wᵀ，w ∈ R^C 过 softmax（C=8 参数），X_ref = R @ X。
      语义「可学习加权再参考」：所有通道减同一个加权均值 1·(wᵀX)。**外积方向必须是 1·wᵀ**，
      不是 w·1ᵀ——后者展开为逐通道增益 w_c·(Σ_j x_j)，每通道减量不同、破坏「参考无关」语义
      （review v1 带进来的笔误，v3 修正）。替代原 C×C 自由矩阵（自由度大、与 Stage 1 空间卷积
      耦合、「参考无关」声明失守）。可选加每通道增益（8 参数，吸收设备增益差）。
- 0.2 基线校正：X_base = X_ref − mean(X_ref[:, :, :51], dim=2)（前 200 ms 基线）。
- 0.3 参考抖动增强（训练时）：以概率 p 随机重参考到随机通道/凸组合，零参数教网络参考不变性。
- 0.4 归一化：基线段归一化（默认）——用前 200ms 基线段（t<0）的逐通道均值/std 做 z-score，
      而非全窗 InstanceNorm。理由：target 试次因 P300 大幅值抬高全窗 std 会被压缩、non-target
      纯噪声被放大到单位方差，而两类试次的「evoked-vs-noise」幅值对比恰是核心判别特征
      （v3 盲区）。全窗 InstanceNorm 保留为消融轴。
- 0.5 坐标式通道身份：10-20 三维坐标 → 正弦嵌入（频率封顶 k≤n_freqs−1，默认 n_freqs=8，输出
      6·n_freqs 维；见 data/channel.D-freq-cap，P0②）→ models 层可学习投影到 D；缺失通道用
      可学习 mask 嵌入。正弦分支的维度由「频段数×2」决定，**不得硬堆高频**（float32 高频段是数值噪声）。
- 0.6 subject metadata 嵌入（新增）：年龄、性别 → 正弦嵌入（频率封顶，2·n_freqs+3 维）→ 可学习
      投影到 D，与通道坐标嵌入并列。性别数字编码遵循 MNE 惯例（0=unknown, 1=male, 2=female）。
- 诚实声明：再参考 + 参考抖动仅统计意义上吸收参考域差，不构成严格不变性。

输出：X0 ∈ R^{B×C×T}，E_chn ∈ R^{C×D}，E_sub ∈ R^D。（E_chn/E_sub 的可学习投影到 D 在此完成；
      data 层只产出频率封顶的正弦特征。）

## 3. Stage 1 —— ERP 感知时空 token 化（Spatio-Temporal Tokenizer）

- 1.1 多尺度时间卷积银行：核长 {13, 33, 65, 129} @256 Hz ≈ {51, 129, 254, 504} ms，
      跨通道共享；每核长 F=16 滤波器，stride=1 + padding=same，四尺度拼接 → B×64×T。
- 1.2 空间深度卷积（按尺度分地形先验初始化）：短核 → N2 地形（PO7/PO8/Oz 枕区负），
      长核 → P3b 地形（Pz/P3/P4 顶区正）；EEGNet 式 depthwise + pointwise。
      v5.1：支持原生 3 导（channel_names=("Fz","Cz","Pz")），不零填充；有效空间权重
      W=prior+coord_mod 后做 max-norm=1（Lawhern 2018；spatial_max_norm=None 回退）。
- 1.3 token 化：Z ∈ R^{B×T×D}（T=256, D=64）+ 时间位置编码（绝对潜伏期是时域任务关键）。
- 1.4 融合 E_chn（通道坐标嵌入）与 E_sub（subject metadata 嵌入）。

输出：Z ∈ R^{B×T×D}。

## 4. Stage 2 —— 成分感知编码器（Component-Aware Encoder）

- 2.1 序列编码（消融轴 depth ∈ {0,1,2,3}，默认 3 层膨胀 TCN，2026-08-20 决策）：
      3 层膨胀 TCN（dilation 1/4/16，感受野 ~168 ms，覆盖 N2→P3b ~150 ms 关系），或备选轻量 Conformer
      （FFN expansion=2，LayerNorm + 域条件仿射；depth=1 Conformer 自身约 33k，叠加 tokenizer 后全模型约 58k 超预算，仅作消融对照）。
      depth=0 时参数化成分窗直接吃 Stage 1 输出。
- 2.2 参数化成分窗 ×3（N2、P3a、P3b；砍掉无监督必然退化的 N1/P2）：
      对每个成分 c：先估计潜伏期 τ_c = τ0_c + Δτ_c（τ0_c 数据驱动初始化，Δτ_c 由 dtau_readout 估计（GTN 默认 attention_softargmax）
      再生成不对称高斯窗（D9）；attention_direct 模式下 Δτ 由时间质心经 tanh 软映射直接得到）：
      A_c(t) = softmax(−(t−τ_c)²/(2·σ_c(t)²))，σ_c(t) = σ_up + (σ_down−σ_up)·sigmoid((t−τ_c)/w)
      （w≈10ms 平滑过渡，处处可微）；σ_up/σ_down 独立 sigmoid 映射到 [20,80]ms（有界无 clamp）。
      软对齐读取 H_c = Σ_t A_c(t)·Z'(t)。
      因果反转（D8）：τ 是 A 的「生成参数」而非「事后统计量」，被分类监督。v5 失败诊断：GTN 实测 τ0/σ/Δτ 梯度比分类头弱 3–4 个数量级，分类监督不足以标定 τ0；L_jit 未收敛且与时间局域判别冲突，故默认关闭（§8），τ0 改用 GTN 数据驱动先验。
      **Δτ 界不对称（v4 修订）**：P3a Δτ∈[−30,0]ms（只前移）；
      P3b 放宽到 Δτ∈[−50,150]ms，覆盖真实 P3b 300–600ms。v5：GTN 儿童实测 P3b 峰值 460–490ms（逐被试 490±85ms），
       GTN runner 默认 τ0_P3b=460ms、τ0 界 [350,600]；成人默认 350ms/[280,500] 不变。
      注意：代码只约束相对 Δτ，不硬性保证 P3a≤300ms/P3b≥300ms；
      防互换依赖 τ0 中心与 Δτ 符号界，Phase 2 须实测 τ 顺序（audit 修订）。
      **global_pool 消融说明（v4）**：global_pool 会洗平时间位置信息，仅作消融；
      H_c=ΣA·Z' 读出路径未池化、保时间分辨率仍不变。Phase 2 合成诊断结果见 review 记录
      attention_direct + L_jit 达到 MAE 36.2ms；v5 后二者仅作显式消融，不再默认。
- 2.3 潜伏期即 τ_c 本身（显式参数，无需事后 soft-argmax）。

## 5. Stage 3 —— 时域多任务头（Multi-Task Heads）

- Head-A 主分类（target/non-target 二分类）：h = concat(H_P3b, H_P3a, g_bypass) → sigmoid → p_target。
       v5.1（EEGNet 借鉴）：g_bypass 默认来自 TemporalPoolingBypass（4× 池化 → depthwise+
       pointwise separable conv → adaptive 8 bin），头为 Dropout(0.25)+Linear；bypass_mode=
       mean_pool 回退旧方案 B 旁路，none 关闭旁路，use_mlp_heads=True 回退旧 MLP 头。
       依据：失败诊断实测 PCW 参数梯度弱 3–4 个数量级，判别信息主要在 encoder 分布式时间特征中。
      损失 L_target = BCE(pos_weight≈8)（target 占 1/9，无 pos_weight 会学「全判非目标」）。
- Head-B 早期证据头（原「N200 辅助」，重新定性）：输入 H_N2（参数化窗 τ0_N2≈220ms、
      σ∈[20,50]ms、Δτ∈±30ms，**单独收窄**限制在早期窗），→ MLP → sigmoid → p_early。
      N2 窗必须单独收窄（σ 上限 50ms、Δτ 上限 30ms，v3 P1）：若沿用 P3 的 σ∈[20,80]ms+Δτ±50ms，
      N2 窗尾部可达 ~350ms，P3a/P3b 前缘会泄漏进早期证据。
      它是「同标签的早期证据集成头」（multi-view ensemble + 正则），不是新任务。
      损失 L_early = BCE，λ2 用固定网格 {0.1, 0.3, 0.5}（不做不确定性加权，数千试次难拟合）。
- Head-C 潜伏期（参数化成分窗，根治版）：输出 τ_c、σ_up,c、σ_down,c（τ 是生成参数，直接监督）：
      (1) 分类直接监督：τ_c 是 A_c 的生成参数，∂L_target/∂τ_c 经 H→A→τ 直接、单调；
      (2) 软参数化（无 clamp）：σ_up/σ_down = lo + (hi−lo)·sigmoid(σ_raw)，
          天然有界 [20,80]ms；Δτ 按成分独立 sigmoid/tanh 映射到 dtau_bounds；
          L_tau = mean((τ_c−τ0_c)²)/50² 小正则，且 τ0 在读出路径中被 detach；
      (3) 方案 2 兜底：训练前几 epoch，τ0_c 用 Pz 峰值弱标签初始化（仅初始化，非监督）。
      不再需要熵锐化与 JSD（D8）；不对称窗解形状受限与 σ 退化（D9）。
- Head-D 幅值：Â = Σ_t A_P3b(t)·X_Pz(t)，被 L_target 隐式监督 + 可选重构。

## 6. 决策层（猜数字）

- score(d) = Σ_{trials of d} logit(p_target)（对数似然比累积，非平均概率）。
- 被试内标准化：每被试用其 9 个数字的 score 分布做 z-score（P300 speller 标准技巧，去校准偏差）。
  防边界（v3 P2）：伪迹剔除后某数字试次可能清零，此时 score(d) 未定义——规定空集 score=−∞
  （或该数字不参与 argmax）；且 z-score 的 std=0（9 数字 score 全同）时退化为不加标准化。
- d̂* = argmax_d score(d)。命中率 = P(d̂* = d*)，chance 11.1%。

## 7. Stage 4 —— 训练配方（辅助数据只预训练/域对齐，P9/D10）

- 7.1 辅助 P300 监督预训练（可选，Phase 3 默认路线）：用 Brain Invaders / BNCI008 / ERP CORE
      的 target/non-target 标签预训练特征提取器，保存 state_dict；目的只是「更好的初始化」。
      之后每个 GTN fold 从该初始化开始、用 GTN 微调；禁止辅助域与 GTN 联合优化主分类。
      加载须 strict=False + 显式 load_mapping，记录成功/跳过/形状不匹配的层。
- 7.2 自监督预训练（可选，Phase 3）：掩码时间点重建波形（Stage 0–1 + 轻量 decoder）。
      语料 = GTN + ERP CORE + 自有无标签数据（靠坐标嵌入 + 通道掩码吃异构通道数）。
      自有数千试次不足以学有意义表征，收益预期「小正则」，排在辅助监督预训练之后。
- 7.3 跨域对齐（仅 N2P3-Net 微调时）：域条件仿射（per-domain scale/shift）+ 特征级 RBF-MMD +
      目标加权损失。辅助域只进入 L_MMD；主分类/早期证据/幅值损失只由 GTN 试次计算。
- 7.4 ERP 感知数据增强：时间扭曲（PMB-TW）、幅值抖动、高斯噪声、通道 dropout、参考抖动。
      增强只作用于 GTN 训练 fold；辅助预训练可用独立增强配置，不得把 GTN 增强后的数据当辅助域。

## 8. 总损失（微调阶段）

L = λ1·L_target + λ2·L_early + λ3·L_tau + λ_jit·L_jit + λ4·L_MMD
- λ1=1.0；λ2∈{0.1,0.3,0.5} 网格；λ3≈1e-2（权重），L_tau = Σ_c(τ_c−τ0_c)²（损失，v3 分写避免混淆）；
  σ 由 sigmoid 软参数化天然有界，无需额外正则；λ4 跨域时启用。
- v5：λ_jit 默认 **0**（关闭 L_jit）。失败诊断实测 L_jit 未收敛（±40ms 平移的 τ 跟踪 RMS≈52ms），
  且时间不变性自监督与 P300 时间局域判别冲突；接口保留，显式传 λ_jit>0 可复现旧实验。
- P9 约束：L_target / L_early / L_amp 只在 GTN 试次上求梯度；辅助域最多进入 L_MMD。
  任何把辅助标签加入上述 BCE/MSE 的实现都属于协议违规。

## 9. 实现要点与坑位对照（v2 更新）

- E1 去漂移：默认 0.1 Hz 连续域高通 + 基线校正 + InstanceNorm；0.5 Hz 高通对 P3b 有失真风险
  （Tanner 2015 实测 ≥0.3 Hz 高通在慢成分前制造反极性伪峰），仅作消融对照（Phase 4）。
- E2 禁 Hungarian：参数化成分窗不涉及集合匹配，天然满足（软窗 + 位置参数化）。
- E3 禁窗口峰值当标签：τ 无真实标签，由参数化窗直接监督；方案 2 仅作 τ0 初始化兜底。
- E4 容量：Phase 2 ≤50k，Stage 2 降容 + depth 消融。
- E5 时间分辨率：Stage 1 无池化；τ 是显式参数（生成窗，非从 A 事后读出）。
- E6 跨域边际：跨数据集单试次接近随机，命中率靠决策层累加。
- 防坍缩（根治）：参数化窗位置寻址天然区分成分，无需 JSD（D8）。
- 新增类不平衡：pos_weight≈8 + 按 run/会话分组切分（review 3.7）。
