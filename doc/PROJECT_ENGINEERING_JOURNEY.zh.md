# N2P3-Net / Neural-RIDE 工程演进与踩坑复盘

> 复盘截点：2026-08-26。仓库 `HEAD=918fc08`，分支为
> `research/issue2-strict-past-rewrite`。本文同时检查了当前工作区，因此包含
> 尚未提交的 v12 代码和文档变化。提交状态必须以 `git status` 为准，不能把本文中
> 的工作区快照误当成某个 Git commit 的完整内容。

## 0. 这份文件怎么读

这不是只介绍模型的 README，而是从项目建立、基线、失败诊断、修复、性能工程，到
当前研究边界的一份工程交接记录。数字按以下证据等级使用：

- **Git**：来自提交历史、提交统计或历史诊断文档，可追溯到 commit。
- **Run**：来自 `tmp/**/record.json`、`progress.jsonl` 等实际运行产物，包含配置和环境。
- **Bench**：来自性能 benchmark 或本次独立复测。
- **Test**：来自单元/语义测试；证明代码契约，不等于证明真实数据性能。
- **推导**：由代码和数学关系得到的解释，不冒充实测效果。

凡是只有单次、少被试或 development 队列的结果，本文都标为“方向性/开发结果”。
当前 GTN 队列已经被查看过，不能用它支撑最终 confirmatory SOTA 声明。

## 1. 项目要解决什么问题

目标是用 EEG 的 ERP 证据，在 oddball “猜数字”范式下判断被试心选的 1--9 数字，
并保留 N2/P3a/P3b 等成分级的结构化读数。核心输入最终统一成：

```text
试次 EEG X: (N, C, T)
标签: target / non-target，以及 GTN 的 thought digit
物理时间轴、真实通道名、三维坐标、缺失通道 mask、被试元数据
```

项目的关键约束一直是：小样本、低 SNR、参考方式和通道数不统一、P300 潜伏期不能
直接标注、9 选 1 命中率不能用原始试次 accuracy 代替。当前正式主任务是 GTN 的
9 选 1 重复证据判定，普通 P300 数据集只作辅助域或独立二分类能力评估。

## 2. 一句话演进结论

项目从“有一份研究蓝图”演进成了一个带数据契约、LOSO 评估、可解释性 gate、运行
记录、dashboard、跨 montage 和 fail-closed 研究路线的完整工程。最重要的经验不是
某一个模块多了几层，而是：

1. 先把数据、单位、参考、时间轴和分母钉死，否则模型数字没有意义。
2. 先做最小基线和反例，再改模型；很多表面上的“容量不足”其实是前端信号被破坏、
   训练过拟合或评估聚合错误。
3. 任何新解释、融合或停止规则都要有独立 gate；没通过就零输出或回退 PCW。
4. 性能优化必须先证明语义等价，再看稳定 benchmark。API 变成矩阵/线性 API 不
   自动等于更快。

## 3. 时间线：从最开始到现在

| 时间 | 阶段和问题 | 工程动作 | 已核对结果/结论 |
|---|---|---|---|
| 2026-08-20 | 项目起点只有架构、使命、路线和工程约束 | 建立 `blueprint`、`constitution`、`mission`、`roadmap`、设备可移植性规范 | 初始 commit `b33d783` 是文档起点，尚不是完整代码实现 |
| 2026-08-22 | 需要把蓝图变成可测试代码 | 一次性落下 data、baseline、models、train、tests、experiments 等模块 | `71c82c3` 新增 80 个文件、12,441 行；早期评审记录 192 项测试全绿 |
| 2026-08-22 | 旧训练协议没有验证侧早停，长训练继续降 loss 但外测变差 | 引入被试级验证划分、早停和训练历史 | 242 人对照中 50 epoch 为 `hit=0.6901/AUC=0.6612`，10 epoch 为 `0.7727/0.7019`；配对置换 `p=0.0033`，确认是过拟合 |
| 2026-08-22 | GTN 是 NIX/HDF5，控制码和正例比例曾被误读 | 逐被试检查 NIX 实际事件；保留 `pos_weight≈8`，不把不存在的第 10 类算进去 | 实际主事件是 1--9，另有 `S13/S15` 控制码；target 约 1/9 |
| 2026-08-22 | sklearn 的窗口逻辑回归直接吃伏特 V，特征太小而被 L2 正则压掉 | `WindowLogisticRegression` 内部加 `StandardScaler` | GTN bacc 从约 `0.500` 回到 `0.564`；这类问题不影响 Pearson template、内部标准化的 SWLDA 和深度基线 |
| 2026-08-22 | N2P3-Net 全量早期结果落后 | 做 12/30/60/242 被试诊断，分别查看容量、训练、参考、tau、tokenizer | 10 epoch 旧配置 `hit=0.7727/AUC=0.7019`；训练 loss 下降不代表外测性能提升 |
| 2026-08-22 | 3 导 GTN 上均匀 CAR/可学习再参考销毁共同 P3b | 关闭默认再参考，并把参考层改为可表达恒等变换、可被 gate 的层 | GTN Pz 差值由约 `10.8 μV` 被 3 导 CAR 变成约 `2.5 μV`，信号损失约 `4.36x`，单试次 SNR 损失约 `1.90x`；去参考后 60 人 default+BN `AUC=0.7575` |
| 2026-08-23 | 网络需要支持不同 montage，不能把 3 导伪装成 8 导 | 使用原生通道数和 `channel_names`，空间先验按通道名解析；补坐标身份和 mask | GTN 直接按 `(N,3,T)` 输入；8 导结构仍可显式选择，但不再是 GTN 默认路径 |
| 2026-08-23 | 随机 temporal FIR 的频谱中心约 60 Hz，且短长核输出尺度相差约 4x | 增加 bandpass/Gabor 初始化、按尺度分配频带和可选 per-scale BN/ELU | 60 人消融中 bandpass 单独 `AUC=0.7691`，与 EEGNet `0.7713` 打平；旧随机初始化不是“学得慢”，而是训练后滤波器仍几乎不动 |
| 2026-08-23 | 242 人需要确认修复不是小样本偶然 | 完成 GLM v2/v3 的全量 LOSO 和 McNemar 配对比较 | v3 `hit=0.8388/AUC=0.7543`，EEGNet `0.8395/0.7620`；hit 为 `203/242` 对 `203/242`，`p=1.000`，相对旧版 `hit=0.7727/AUC=0.7019` 明显改善 |
| 2026-08-24 | BNCI epoch 从 0 ms 起，没有可用的 -200--0 ms 基线段 | 标准化策略按数据集物理时间轴选择，避免用极短的 4 点估计噪声尺度 | BNCI 单 fold 的 `base=.7365 -> mean_only=.8177 -> zscore=.8346`；全量修复后 N2P3-Net `bacc=.7102/AUC=.7993`，与 EEGNet `.7103/.7992` 打平 |
| 2026-08-24 | 多数据集专用脚本和结果管理容易漂移 | 统一 EpochDataset/manifest/通道坐标接口，增加 ERP 校准入口、模型独立 run 目录、progress 和 dashboard | GTN、BNCI、Brain Invaders 进入统一实验口径，专用默认值逐步移除 |
| 2026-08-24 | 需要验证干电极和不同 montage 的通用性 | 完成 GTN、BNCI-008、Brain Invaders 三数据集矩阵 | Brain Invaders 16 导 60 人：N2P3-Net `.6899/.7718`，EEGNet `.6803/.7541`，AUC 配对检验 `p=0.0019`；辅助 MMD 试点未证明 GTN 增益 |
| 2026-08-24 | 成分窗口的 tau 曾被过度解释为真实生理潜伏期 | 把 PCW tau 降级为 routing 参数；新增独立 latency measurement 规范和合成 gate | 全模型可分类不等于 tau 可识别；后续必须通过 known-shift、斜率、覆盖率和振幅混淆测试 |
| 2026-08-25 | strict-past 的残差/融合容易把开发集 NLL 改善误写成主任务收益 | 删除旧 residual classifier、ERP subtraction、单参数非负 alpha；研究分支 fail-closed | 60 人和后续锁定折未证明稳定增量，正式输出保持 `final=PCW` |
| 2026-08-25 | fold worker、finalization、父进程线程预算不一致 | 把 fold CPU budget 贯穿各路径，finalization 使用完整预算并在训练前校验，父进程初始化 8-core 调度 | Git commits `11c9375/2888953/3cbab69/204a1da`；运行记录现在包含线程数、fit wall time、峰值内存 |
| 2026-08-25 | tau 诊断需要能识别 offset tau0 的反例 | 加 synthetic latency identifiability audit，并增加 tau0 偏移 challenge | `52f7a23`、`918fc08`；不再把单纯分类成功当作 latency recovery 证据 |
| 2026-08-26 | v12 研究路线继续扩大，但不能让未验证对象污染正式路线 | 工作区加入 LatencyMeasurement、RepetitionEvidence、Reliability 双 estimand、InnovationAudit/Stopping 四对象和对应测试 | 当前默认仍是 PCW；四对象各有独立 gate，未过 gate 时保持零输出或 descriptive-only |
| 2026-08-26 | full-cohort 训练成本和 kernel 调度成为瓶颈 | tokenizer 时空算子融合、TCN pointwise linear 调度、GPU benchmark、梯度累积和可选 compile | 已测 tokenizer fused 约 `1.82x`；编译路径在本机 Triton/Windows 环境失败，因此默认保留 eager |

### 3.1 Git 提交索引

下面是截至 `HEAD` 的完整提交链。`7e1804f` 和 `540ce2a` 都曾追加同一节诊断文档，
随后 `dd7c1f9` 修复了沙箱重试导致的重复追加；这类历史操作也保留在索引中，方便
追踪为什么文档曾出现重复内容。

| Commit | 日期 | 主要内容 |
|---|---|---|
| `b33d783` | 08-20 | 文档起点：使命、蓝图、宪法、设备兼容和路线 |
| `71c82c3` | 08-22 | 第一版完整模块、训练入口、基线和测试；训练协议修复、TCN BN 消融、儿童 P3b 宽窗 |
| `5ac8446` | 08-23 | 60 被试结果和诚实结论：AUC `0.7245` 仍未打开表征缺口 |
| `e9e997b` | 08-23 | Mini 诊断定位再参考层销毁鼻参考数据上的 P3b；去参考和 BN 回升 |
| `b39bb5b` | 08-23 | 门控参考层，允许恒等变换和有效参考审计 |
| `7e52a7a` | 08-23 | 242 被试全量复核，首次与 EEGNet 达到统计不显著差距 |
| `b5aaced` | 08-23 | Tokenizer v3：带通初始化、per-scale BN/ELU 选项和语义测试 |
| `52e0e34` | 08-23 | 补齐 tokenizer 的诊断、文献、设计和验证链；60 人 AUC `0.7691` |
| `ffc3694` | 08-23 | LOSO 逐 fold 进度记录、dashboard 和训练可观测性 |
| `afab05c` | 08-23 | v3 全量定案，默认使用 bandpass tokenizer init |
| `58979df` | 08-24 | BNCI-008 全量入口和二分类 early-stop 路径修复 |
| `ff05c01` | 08-24 | 一键启动器、活动 run 自动定位、dashboard LOSS 记录 |
| `07bd8cf` | 08-24 | 自适应 ERP 成分窗校准工具，作为新数据集接入前置步骤 |
| `aaeb240` | 08-24 | BNCI 无基线段标准化修复，单 fold 最高约 `+9.8 pt` |
| `b48015e` | 08-24 | Brain Invaders/bi2014a 16 导原生通道规格 |
| `e3f3cbf` | 08-24 | 三数据集矩阵定案：GTN 打平、BNCI 打平、Brain Invaders 显著超越 |
| `7e1804f` / `540ce2a` | 08-24 | 多模型覆盖、PCW 解释复核和 bi2014a 失败被试画像；曾发生重复追加 |
| `dd7c1f9` | 08-24 | 删除重复追加的诊断文档内容 |
| `11c9375` | 08-25 | fold CPU budget 贯穿不同执行路径 |
| `2888953` | 08-25 | finalization 使用完整 CPU budget |
| `3cbab69` | 08-25 | 训练前校验 finalization CPU budget |
| `204a1da` | 08-25 | 初始化八核心父进程调度 |
| `52f7a23` | 08-25 | 加入 synthetic latency identifiability audit |
| `918fc08` | 08-25 | 增加 tau0 offset challenge，防止 latency recovery 过度乐观 |

## 4. 当前代码结构和职责边界

| 层 | 目录/入口 | 当前职责 |
|---|---|---|
| 数据 | `src/data/` | 通道身份、坐标、事件、epoch、manifest、NIX/MOABB/通用 Dataset、物理时间轴和质量信息 |
| 基线 | `src/baselines/` | SWLDA、xDAWN+RG、template、window LR、EEGNet/Inception/Conformer、校准、LOSO、配对检验 |
| 主模型 | `src/models/` | reference、tokenizer、TCN/Conformer encoder、PCW、heads、decision、跨 montage、ERP/重复证据模块 |
| 训练 | `src/train/` | 配方、Trainer、loss、增强、device、preloaded loader、multi-dataset、prequential 和 contracts |
| 测量/审计 | `src/measurement/`、`src/audit_p300/` | latency measurement、P300 表示层干预、fidelity/clean probability 和 gate |
| 实验入口 | `experiments/` | 数据准备、GTN baseline、N2P3-Net LOSO、多数据集 transfer、locked multiseed、gate、dashboard、benchmark |
| 测试 | `tests/` | 形状冒烟、语义反例、梯度、mask、泄漏边界、统计口径、协议和性能优化等回归测试 |

当前推荐阅读顺序是：

1. `doc/constitution.md`：科学与工程不可违反的边界。
2. `doc/routes.md`：正式 PCW 路线和 strict-past 研究路线。
3. `doc/blueprint.md`：v12 四对象架构及 gate。
4. `doc/evaluation_protocol.md`：LOSO、ITT、coverage、校准和 confirmatory 规则。
5. `doc/neural_ride_recipe.md`：默认训练配方。
6. `doc/CLI_MANUAL.zh.md`：实际命令。

## 5. 数据链路改进和踩坑

### 5.1 单位坑：MNE 输出是 V，不是 μV

MNE 原始数据单位是伏特。P300 的典型量级约为数个 `10^-6 V`。直接把窗均值特征
喂给带 L2 的逻辑回归时，特征太小，正则项会把有效权重压掉，模型接近随机。

解决方式：

- 在 `WindowLogisticRegression` 内部 fit-only 使用 `StandardScaler`。
- 不在 test fold 上 fit scaler。
- 记录模型是否自带 BatchNorm/InstanceNorm，避免重复标准化造成物理口径漂移。
- 把输入单位和标准化方式写入 run record，而不是只写在人的记忆里。

这次修复给出的证据是 GTN bacc 约 `0.500 -> 0.564`。它说明“模型差”之前必须
先排查单位和正则尺度。

### 5.2 标签比例坑：先核对实际事件，再决定 pos_weight

早期曾出现把 `1/9` 的 target 误推成第十个类别的想法。逐被试检查 NIX 事件后确认：

- 有效数字刺激是 `S1..S9`。
- `S13`、`S15` 是少量控制事件，不是第十个数字。
- target 基率约为 `1/9`，所以 `pos_weight≈8` 合理。

当前规则是每个外层训练折按训练侧 `n_neg/n_pos` 计算；只有固定为精确比值时，weighted
BCE 的输出才可能被讨论成 LLR。否则必须显式去掉 class weight/训练先验偏置并做
验证侧温度校准。

### 5.3 GTN 元数据和分母坑

GTN 不是 BrainVision 数据，而是 NIX/HDF5，并且 `the number thought` 位于配套 `.txt`
元数据中。被踩过的坑包括：

- `Data/` 里有多个 `.txt` 时取第一个，依赖未定义的文件系统顺序。
- `Experiment_611` 曾被误写成 NIX 损坏；复核后真正问题是没有 thought `.txt`。
- 重复的 NIX subject 会造成标签和分母覆盖。
- 全部 epoch 被质量阈值剔除的被试如果静默消失，会让命中率分母变小。

现在通过 NIX 内部 subject id 精确匹配 `.txt`，显式记录 duplicate、missing thought、
all epochs unavailable 等原因。当前 full-cohort record 的数据事实是：

```text
原始目录约 248 个
model-ready 被试 242 个
试次 36,625
输入形状 (36,625, 3, 358)
物理时间轴约 -200..1200 ms，256 Hz
target rate 约 0.1139
```

“242”是当前 N2P3-Net 可跑的 model-ready 数量，不等于所有原始目录；原始目录、ITT
全集和可用训练样本必须在记录中分字段保存。

### 5.4 缺失通道坑：零填充后增强会制造幻象通道

早期做法把 GTN 3 导补成 8 导零张量。这样即使模型入口知道 mask，训练增强中的
`reference_jitter` 或 `gaussian_noise` 仍可能把缺失通道变成非零；后续 baseline
standardization 会把它放大成类似真实信号的“幻象通道”。历史复核的量级是：

| 路径 | 缺失通道输出 std |
|---|---:|
| 无 mask 的 reference | `0.943` |
| reference 使用 mask | `0` |
| 先 reference jitter，再 mask | `0.998` |
| 先 gaussian noise，再 mask | `0.986` |

解决方式是贯穿整个链路，而不是只改模型 forward：

- 3 导 GTN 走原生 `(N,3,T)`，不伪造 8 导。
- `channel_mask` 传入 Trainer、reference、canonical GP 和增强。
- reference 只由存在通道计算，noise 只加到存在通道；增强后缺失通道强制为零。
- 任意布局必须有真实通道名和坐标，缺失数据记录为 mask/quality，而不是静默补值。

### 5.5 时间轴坑：不能固定前 51 个点代表 baseline

GTN 是 `tmin=-200 ms`，前 51 点约为 pre-stimulus baseline；BNCI 有从 0 ms 开始的
epoch，没有同样的基线。原先固定 `baseline_n=51` 会在 `T=128` 或 `tmin=0` 的缓存上
静默取错时间段。

当前做法是把 `sfreq/tmin/tmax/T` 写进 EpochDataset/cache，按数据集物理时间轴选择
`trial`、`global` 或显式的输入标准化方式。BNCI 的修复正是由这个差异定位出来的：
0 ms 起始数据用 4 点估计尺度会成为随机噪声放大器。

## 6. 模型演进和关键判断

### 6.1 参考层：从“默认再参考”变成“可门控、恒等可表达”

早期把 3 导鼻参考数据默认经过近似 CAR 的再参考层。3 导不满足平均参考的覆盖假设，
而 GTN 的 P3b 在 Fz/Cz/Pz 具有共同宽正波，减去均值会直接减掉任务信号。

实测：Pz target--non-target 差值约 `10.8 μV`，均匀 CAR 后约 `2.5 μV`；Fz 甚至
出现反号。这个失败不是“网络容量不够”，而是物理变换先把信息删了。

改进：

- 参考变换使用少量参数的加权形式，而不是自由 `C x C` 矩阵。
- 参数化必须能表达恒等变换。
- gate 开始时允许零变化；输出 effective reference 便于审计。
- mask 只允许存在通道参与参考。
- 对 GTN 鼻参考默认关闭；对其他参考类型按数据集实验决定。

这保留了“参考自适应”意图，但不强迫每个数据集都做 CAR。

### 6.2 Tokenizer：时间尺度和空间先验

当前 Stage 1 的核心是：

```text
(B,C,T)
 -> temporal Conv1d bank: kernels (13,33,65,129) @ 256 Hz
 -> per-scale spatial weights: ERP topology prior + coordinate modulation
 -> pointwise projection
 -> physical-time positional encoding + subject bias
 -> (B,T,D)
```

关键决定：

- 13/33/65/129 点约对应 51/129/254/504 ms，覆盖早期和 P3b 时间尺度。
- temporal filter 跨通道共享；不是每个通道一份互不共享的滤波器。
- 短核采用 N2 枕区负权先验，长核采用 P3b 顶区正权先验，之后仍可学习。
- 用 channel coordinates 生成 modulation，避免把固定通道名硬编码成唯一身份。
- spatial weight 可用 EEGNet 风格 row max-norm；这是约束，不是保证性能提升。
- 全程 stride 1、same padding、无池化，保持完整 T，避免把 latency 信息池化掉。

### 6.3 Tokenizer 初始化坑：随机 FIR 不是自然的 ERP filter bank

随机 Kaiming 初始化在不同核长上有两类问题：

1. 短核因为 `1/sqrt(k)` 缩放，输出 std 约为长核的 4 倍。
2. 随机滤波器的能量中心接近高频；训练后与初始化的余弦相似度仍为 `0.90--0.98`，
   没有主动学出 ERP 所在的 1--8 Hz 形状。

解决方式是 `init="bandpass"`：用 Gabor/sinusoid x Hann 生成按核长分层的可学习初值，
单位化并按 `1/sqrt(k)` 缩放噪声。注意这里是初始化，不是把滤波器硬编码死；训练后仍
可偏离先验。

一个重要二次教训是噪声尺度：绝对值 `0.05` 的噪声虽然看似小，却会在短核中把白噪声
频谱能量推到约 48 Hz；改成相对于每个抽头的 `noise/sqrt(k)` 后才符合预期。

结果：60 人消融里 bandpass-only `AUC=0.7691`，EEGNet `0.7713`；BN+ELU 组合为
`0.7630`，所以当前胜出配置是 bandpass 初始化单独开启，而不是“所有正则一起打开”。

### 6.4 Encoder：小模型、显式深度消融和可控感受野

Stage 2 保留 `depth ∈ {0,1,2,3,4}`：

- depth 0 是 identity 地板，判断 Stage 1 是否已经足够。
- TCN 使用 depthwise temporal + pointwise mixing。
- 当前工作区正式 recipe 默认 depth 4，dilation 为 `1/4/16/32`；depth 3 保留为轻量对照。
- D=64 时 3 层感受野约 168 ms，4 层约 418 ms；Stage 1 的 504 ms 长核补足更慢成分。
- 默认参数量仍受 `<=80k` 硬上限约束；当前 full-cohort record 中 depth3 为 `50,280`，
  depth4 为 `54,824`。

2026-08-26 的 5 被试 probe 是开发方向，不是定案：

| 配置 | hit | bacc | AUC | 备注 |
|---|---:|---:|---:|---|
| depth3, strict-past off | `1.000` | `0.6645` | `0.7499` | 5 被试，16 epoch，descriptive |
| depth4, strict-past off | `1.000` | `0.6798` | `0.7496` | 5 被试，16 epoch，descriptive |

full-cohort 两次运行的主决策口径和研究 gate 尚未形成确认性证据。因此 depth4 是当前
recipe 选择，不应写成已经被全量统计证明优于 depth3。

### 6.5 PCW 和 tau：分类能用到时间信息，不代表 tau 是生理真值

曾经把参数化 component window 的 tau 直接说成“单试次潜伏期”。合成和真实诊断
证明这个说法太强：Stage 1/2 可能自己重编码时间，使固定窗口也能完成分类；此时
分类 AUC 很高，tau 却不随真实 latency jitter 变化。

当前语义分层：

- `pcw_tau`：模型 routing/structured window 参数。
- `measured_tau_posterior`：只有独立 `LatencyMeasurement` 通过合成和真实数据 gate 后，
  才能用于生理潜伏期报告。
- 测量对象使用 fold-local 白化模板、amplitude profile likelihood 和显式规范锚。
- 不能用窗口内峰值、固定 tau0 或分类梯度伪装成逐试次物理标签。

这次修订牺牲了一部分“解释故事”，换来可证伪和可复核的测量口径。

## 7. 训练、评估和科学口径的改进

### 7.1 早停：loss 下降不等于泛化提升

GTN 的小样本场景中，50 epoch 的 train loss 可以持续下降，但 outer LOSO 指标明显变差。
因此：

- 按 subject 划分 validation，不能随机按 trial 混合。
- early stop 只看训练侧验证；正式 task endpoint 不能被辅助 loss 掩盖。
- record 保存每 epoch 的 task loss、density loss、best epoch 和是否触发 patience。
- 30/50 epoch 不是“更认真训练”，必须由验证侧证据决定。

实测 full-cohort：10 epoch `hit=.7727/AUC=.7019`，50 epoch `hit=.6901/AUC=.6612`，
`p=.0033`。这是项目里最清楚的“训练更久反而更差”反例。

### 7.2 决策层校准坑：先中心化，再按数字聚合

旧做法是直接按数字累加：

```text
score(d) = sum(logit_i for trial i of digit d)
```

如果模型有被试常数偏置 `c`，实际进入排名的是：

```text
sum(signal_i) + c * n_d
```

而每个数字 `n_d` 不完全相同。最后对 9 个 score 做 z-score 只是同一个仿射变换，
不能消掉 `c*n_d`。修复是每个 subject 先中心化 logit，再做 sum 或 mean；`mean` 作为
显式消融轴。

历史 GTN 30 人复核：

| 方法 | 未先中心化 | 先中心化后 sum | 先中心化后 mean |
|---|---:|---:|---:|
| SWLDA hit | `.467` | `.667` | `.700` |
| Template hit | `.767` | `.767` | `.800` |

当前 `src/models/decision.py` 还做了：固定 digit vocabulary、空数字 `-inf`、NaN
显式报错、`bincount` 桶化聚合。这样既修正了统计语义，也避免了对每个 subject/digit
重复扫全量数组。

### 7.3 评估不能只给一个平均 accuracy

当前正式协议区分：

- trial-level AUC 和 inductive balanced accuracy：看单试次辨别能力。
- subject-level 9-choice hit@K：看多次证据累积后的实际任务。
- ITT accuracy、coverage/N、conditional accuracy：不可用 subject 不能从分母消失。
- exact/prefix/flash/time/all budget semantics：不同证据预算不能混名。
- Brier、ECE、LLR reliability、expected flashes、risk-coverage：看校准和效率。
- McNemar/subject-paired bootstrap/cluster bootstrap：以 subject 为统计单元。

特别注意：K15 在完整缓存上的 coverage 约 29% 时，只能作次要分析；不能把有结果的
fold 当成 100% 可用。

### 7.4 辅助域边界

Brain Invaders、BNCI-008、ERP CORE 只能做：

- 辅助域监督预训练后逐 GTN fold 微调；
- 共享编码器的显式域对齐；
- 独立数据集的二分类 LOSO 能力评估。

不能把辅助 trial 和 GTN trial concatenate 后直接优化 GTN 主分类头，不能让辅助标签
进入 GTN 的主损失，也不能用辅助域精度替代 GTN 主验收。60 人 T0/T1/T2/T3 试点中，
预训练微调总体不劣但没有达到显著主任务增益，MMD 对 GTN 也没有稳定提升，因此默认
回到 GTN from-scratch 是正确的 fail-closed 选择。

## 8. 性能工程：已经做了什么，数据支持到什么程度

### 8.1 先纠正一个容易误写的说法

当前生产路径中没有“对每个样本、通道、时间点用两层 Python `for` 手写卷积”。
时间卷积由 PyTorch `nn.Conv1d` 执行，代码里的 `for` 主要只是遍历 4 个尺度、初始化
滤波器或构造模块。

真正做过的优化是把可交换的线性时空操作折叠成单个卷积，并消除大中间张量：

```text
legacy:
  Conv1d(X.reshape(B*C,1,T)) -> (B,C,F,T)
  einsum("bcft,fc->bft")

fused:
  K_eff[f,c,k] = W[f,c] * K[f,k]
  b_eff[f]    = b[f] * sum_c W[f,c]
  F.conv1d(X, K_eff, b_eff)
```

两条路径的输出和反向梯度都有语义回归测试。遇到 per-scale BN/ELU 或 canonical GP
这类非线性/坐标投影路径时自动回退 legacy，避免为了速度改变模型语义。

### 8.2 Tokenizer 时空融合 benchmark

证据文件：`tmp/gpu_schedule_b1024_final_eager.json`，benchmark 脚本：
`experiments/benchmark_gpu_schedule.py`。

环境和设置：RTX 5070 Laptop GPU，约 8.5 GB，PyTorch `2.13.0+cu132`，CUDA 13.2，
`B=1024,C=8,T=256,D=64`，预热 10 步、稳定测量 50 步、eager、训练 step。

| 路径 | ms/step | samples/s | peak allocated |
|---|---:|---:|---:|
| legacy Conv1d + spatial einsum | `117.97` | `8,679.84` | `1,440.59 MB` |
| fused effective Conv1d | `64.87` | `15,786.03` | `1,172.21 MB` |
| 变化 | **约 1.82x 快** | **约 +81.9%** | **约 -18.6%** |

这支持“算子融合减少 kernel launch/中间激活带来约 1.8 倍加速”，不支持“二维 for
被矩阵计算替换后固定提升几倍”这种脱离硬件和 shape 的泛化说法。另一份较早的
10-step benchmark 是 `110.71 -> 61.70 ms`，约 `1.79x`，方向一致。

### 8.3 TCN `1x1 Conv1d -> Linear` 调度

TCN pointwise 权重仍保持 `(D,D,1)`，只是把 `(B,D,T)` 转成 `(B*T,D)` 调用
`F.linear`，因此 checkpoint key/shape 不变。`tmp/gpu_schedule_b1024_pointwise*.json`
的两次反序 benchmark 为：

- linear `60.92 ms/step`，conv1d `66.21 ms/step`；
- 另一顺序为 linear `59.04`，conv1d `65.50` ms/step。

在这个 shape 上约为 **8--10%** 的稳定方向，但顺序和 warm-up 会影响绝对值，因此不
应写成精确固定收益。当前 recipe 选 linear，conv1d 保留为等价对照。

### 8.4 决策层向量化

这里可以复现出“矩阵/桶化比全量 mask 循环快”，但没有证据支持仓库 docstring 中的
`~18x` 在所有机器上成立。本次在当前 Windows 环境独立复测：

```text
N=50,000, subject=244, digit vocabulary=1..9
旧等价算法：subject x digit 反复构造全量布尔 mask
新算法：bincount bucket + 行广播 + argmax
三次中位数：约 9.68x
预测结果：完全一致
```

如果旧算法改成先切 subject 再在切片中循环，本机约为 2.26x；这说明“旧实现”定义
本身会影响 speedup。文档和论文中应写清 baseline algorithm、N、shape、warm-up、
硬件、线程数和统计方式。

### 8.5 GPU 调度、并行 fold 和编译坑

当前 full-cohort 运行记录给出以下量级：

| 运行 | 配置 | 总 wall | fit durations | peak fit memory |
|---|---|---:|---:|---:|
| depth3, 2 folds | `fold_jobs=2, batch=4096` | `920.3 s` | `853.5/846.9 s` | `~9,776 MB` |
| depth4, 2 folds | `fold_jobs=2, batch=4096` | `934.9 s` | `866.8/864.5 s` | `~10,714 MB` |

两个 fold 共用一块 GPU 时，`fold_jobs=2` 不是理论上的 2 倍吞吐；资源记录中 GPU
利用率达到 96--100%，并且每个 worker 仍有 CPU/BLAS 线程开销。当前做法是显式设置
父进程和 fold 线程预算、记录 `fit_sec`/内存，并按显存决定 batch/并行度。

`torch.compile` 的 benchmark 在当前 Windows/Triton 环境报：
`InductorError: 0 active drivers ([]). There should only be one.` 这不是模型数学错误，
而是编译后端/驱动环境问题；另一次 smoke 还遇到 Windows 默认 GBK 读取 Triton 模板的
`UnicodeDecodeError`。所以：

- 默认 `compile_mode=eager`，保证可复现。
- compile 只允许 `fold_jobs=1`，避免编译缓存和驱动争用。
- benchmark 必须把 cold start 和 steady step 分开。
- 只有在目标机器编译成功且与 eager 做数值/梯度对比后，才可改变默认值。

## 9. 当前 v12 研究路线：为什么要拆成四个对象

当前工作区把以前混在一条 strict-past 路径里的“解释、融合、可靠性、停止”拆开：

| 对象 | 负责什么 | 进入主研究路径的条件 |
|---|---|---|
| L: LatencyMeasurement | 估计可测的成分潜伏期后验 | S0 合成 known-shift、斜率、区间覆盖、初始化敏感度、振幅混淆全部通过；真实数据再做 split-half/峰值/mass-univariate 对照 |
| R: RepetitionEvidence | 按真实 flash 顺序累积 additive LLR，并可选 state residual | additive backbone 常开；residual 必须在 held-out log score 上有增量 |
| Q: Reliability | 连续 `fidelity` 与显式二元污染模型下的 `clean_probability` | fidelity 过未见被试/未见 corruption 排序 gate；clean probability 还要硬标签生成模型和 prior-shift gate |
| S: InnovationAudit + Stopping | 检验新证据是否真的提供分类增量，以及停止效率 | 嵌套 `M0:a+bS` vs `M1:a+bS+cL`、subject-cluster bootstrap、符号预注册、NLL/AUC 非劣；停止先做 replay，e-process 后置 |

正式路线仍是：

```text
final = PCW
lambda-innovation = 0
新对象没有通过 gate -> 输出 zero/descriptive-only，不改变 formal result
```

单参数非负 fusion 已删除；`soft-BCE(0.9/0.1)` 仅存在于
`NEURAL_RIDE_V11_LEGACY` 历史对照，生产默认是 v12 fidelity rank 目标；PCW attention
tau 最高只能称 `fold-calibrated routing window`；普通 posterior threshold 只做
descriptive replay，不得宣称固定错误率控制。

## 10. 当前已确认的结果和不能宣称的结果

### 可以写进阶段性报告的结果

- GTN 242 development LOSO：GLM v3 `hit=.8388/AUC=.7543`，EEGNet `.8395/.7620`，
  digit-level McNemar `p=1.000`，相对旧版明显提升。
- BNCI-008 修复标准化后：N2P3-Net `bacc=.7102/AUC=.7993`，EEGNet `.7103/.7992`。
- Brain Invaders 2014a 16 导 60 被：N2P3-Net `bacc=.6899/AUC=.7718`，EEGNet
  `.6803/.7541`，记录的 AUC 配对检验 `p=.0019`。
- tokenizer 时空融合 benchmark 在指定 RTX 5070 shape 上约 `1.82x`，峰值显存约降 `18.6%`。
- 当前独立决策层 benchmark 在明确的 full-mask loop 对照下约 `9.68x`，结果完全一致。

### 不能写成最终结论的结果

- 242 队列已用于开发和架构选择，不能作为未暴露 confirmatory SOTA。
- 5/12/30/60 被试结果只能说明方向，不能替代全量、多 seed、锁定协议。
- `hit` 高不等于 trial AUC 高；命中率还受每数字重复次数和聚合规则影响。
- 代码通过、单折 NLL 下降、reliability AUC 高，都不等于新对象对主 GTN 任务有增量。
- PCW tau 与 tau0 接近不等于真实单试次潜伏期已被识别。
- auxiliary 域结果不能替代 GTN 主任务；MMD 试点没有证明通用增益。
- depth4 当前是 recipe 选择，不是已经通过 full-cohort confirmatory 的性能优胜者。

## 11. 以后最容易再踩的坑

1. **不看 `git status` 就写报告**：当前有大量未提交的 v12/性能代码，必须记录 commit 或工作区快照。
2. **把 `record.json` 的 descriptive 字段当 formal**：先看 `claim_eligible`、coverage、gate 和 mode。
3. **把 loss 下降当泛化提升**：必须看 subject-disjoint validation 和 outer test。
4. **把全局 z-score 放在错误的位置**：决策层要先中心化 trial logit，再聚合；score 后 z-score 不能修 `c*n_d`。
5. **把所有 epoch 当成同一种时间轴**：标准化和 baseline 必须读取物理 `tmin/tmax/sfreq`。
6. **对缺失通道只在模型入口加 mask**：增强、reference、GP、标准化和保存格式都必须遵守 mask。
7. **把 3 导补成 8 导**：这会让空间先验和模型容量学习到不存在的传感器。
8. **默认做 CAR**：少导、鼻参考数据上可能直接删除 P3b；先做参考类型和 ERP 差值诊断。
9. **初始化噪声不按 kernel size 缩放**：短 FIR 会被白噪频谱主导。
10. **只凭 API 名字判断性能**：`Linear`、`matmul`、`einsum` 是否更快必须在固定 shape、warm-up、线程和硬件下测。
11. **为了跑快盲开 compile 或 fold 并行**：编译、驱动和显存争用可能比计算本身更先出问题。
12. **把研究支路偷偷混进正式支路**：任何 residual、fusion、reliability、stopping 新增项都必须有独立开关和 fail-closed 默认。

## 12. 推荐复现路径

### 12.1 先做测试和小规模 smoke

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

重点测试：`test_tokenizer.py`、`test_encoder.py`、`test_decision.py`、`test_gtn_cache.py`、
`test_experiment_protocol.py`、`test_s0_harness.py`、`test_latency_identifiability.py`。

### 12.2 运行正式 PCW development

```powershell
.\.venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
  --lambda-innovation 0 `
  --run-dir tmp\production-dev
```

正式 confirmatory 不能直接复用当前已经暴露的 GTN manifest；要用未查看的新 cohort、
至少 5 个 seed、一次性锁定的 config/cohort/cache hash。

### 12.3 运行 tokenizer benchmark

```powershell
.\.venv\Scripts\python.exe experiments\benchmark_gpu_schedule.py `
  --batch-size 1024 `
  --times 256 `
  --steps 50 `
  --warmup-steps 10 `
  --compile-mode eager `
  --output tmp\gpu_schedule_repro.json
```

跑 benchmark 时必须同时保存硬件、PyTorch/CUDA、shape、warm-up、steady step、吞吐、
显存和 error；不要只截图一个“快了几倍”。

## 13. 相关文件索引

- 现行文档入口：`doc/README.md`
- 科学/工程边界：`doc/constitution.md`
- v12 架构：`doc/blueprint.md`
- 路线选择：`doc/routes.md`
- 评估协议：`doc/evaluation_protocol.md`
- 数据契约：`doc/datasets.md`
- 训练配方：`doc/neural_ride_recipe.md`
- 辅助数据政策：`doc/transfer_policy.md`
- 性能 benchmark：`experiments/benchmark_gpu_schedule.py`
- tokenizer 融合实现：`src/models/tokenizer.py`
- 决策层向量化实现：`src/models/decision.py`
- TCN pointwise 调度：`src/models/encoder.py`
- 当前实验产物：`tmp/**/record.json`、`tmp/**/progress.jsonl`
- 历史 v11 文档和旧路线：`archives/legacy_v11_docs_2026-08-25/`

## 14. 最终交接判断

截至 2026-08-26，工程基础已经从“能跑模型”提升到“能追踪数据、协议、证据和资源”。
当前最可靠的阶段性结论是：

```text
N2P3-Net 的性能提升主要来自修复数据/参考/时间滤波/训练协议和评估语义，
而不是无条件增加容量；性能优化已经有指定硬件上的 1.82x tokenizer 证据，
但所有 speedup 都必须带 baseline、shape、硬件和测量协议。
```

下一阶段的重点不是继续堆新分支，而是把 v12 四对象在完整 locked protocol 中逐个
证伪或通过，并把当前工作区整理成一个明确 commit，再进行多 seed、未暴露 cohort 的
确认性运行。
