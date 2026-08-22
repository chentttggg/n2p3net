# 辅助 P300 数据使用政策（Transfer Policy）

> 本文档是「其他 P300 数据集辅助训练」的唯一实施规范。
> 优先级：constitution > 本文档 > blueprint/roadmap/techstack 中的训练配方。
> 版本：v1（2026-08-21，采纳「辅助数据只预训练/域对齐；GTN 独占分类头与验收」范式）。

## 1. 一句话政策

**辅助 P300 数据只允许改变模型的「初始化」或参与「域对齐」；不允许进入 GTN 主分类损失、分类头梯度或 GTN 测试 fold。最终每个 fold 仍在 GTN 上微调，所有验收指标只由 GTN 协议决定。**

## 2. 域角色定义

| 域 | 角色 | 说明 |
|---|---|---|
| GTN（242 名可评估被试） | **主域 / 最终实验情景** | 猜数字 9 选 1；LOSO 训练与验收 |
| Brain Invaders bi2014a | 辅助域 | 干电极 P300 target/non-target，最接近硬件噪声 |
| BNCI2014_008 | 辅助域 | 8 导、256 Hz，通道集合与项目蒙太奇完全一致 |
| ERP CORE P3 | 辅助域 | 成人 oddball P3，最接近成人目标人群 |
| 自有成人 8 导数据 | 部署目标域（后续） | 最终 zero/few-shot 验收，当前阶段不混入 GTN |

## 3. 允许的三种方式

### 方式 A：辅助域监督预训练 + GTN 逐 fold 微调（基线模型 EEGNet/Inception 首选）

1. 在辅助 P300 数据上预训练同一骨干的**特征提取部分**（target/non-target 二分类）；
2. 每个 GTN LOSO fold 新建模型，加载该预训练权重作为初始化；
3. 在 fold 训练集上微调全部或部分层；
4. 分类头可以在微调前重新初始化，也可以加载后微调；但最终参数必须经过 GTN 微调；
5. 评估只在 fold 测试集上做。

### 方式 B：共享编码器 + 域对齐，GTN 独占监督（N2P3-Net 首选）

1. 辅助域与 GTN 共享 Stage 0–2 编码器；
2. 辅助域只参与域条件仿射和/或 RBF-MMD 对齐损失 `L_MMD`；
3. `L_target`、`L_early`、`L_amp`、`L_tau`、`L_jit` 均只在 GTN 试次上计算；
4. 辅助域不向 `Head-A/Head-B/Head-D` 提供梯度；
5. GTN 每个 fold 仍在共享编码器基础上微调。

### 方式 C：辅助域预训练 + 冻结骨干 + GTN 只训分类头（严格对照）

1. 加载辅助域预训练权重后冻结全部特征层；
2. 只在 GTN 上训练分类头/决策相关参数；
3. 用作「最低污染」下界，评估预训练表示本身有多大价值。

## 4. 明令禁止

- **禁止**把辅助域试次与 GTN 试次直接 `concatenate` 成一个训练集联合优化主分类 BCE。
- **禁止**让辅助域标签进入 `L_target` / `L_early` / `L_amp` / `Head-A` / `Head-B`。
- **禁止**把任何辅助域试次放入 GTN 测试 fold，或把 GTN 测试 fold 用于辅助域训练。
- **禁止**在 GTN 上完成适应后，再用辅助数据继续训练并宣称同一模型代表主情景。
- **禁止**用辅助域单试次精度或辅助域命中率替代主任务验收指标。
- **禁止**根据 GTN 测试 fold 表现调辅助预训练超参；调参只能看 GTN 训练/验证划分或独立留出。
- **禁止**辅助预训练时混入「猜数字 thought number」信息；辅助域只有 target/non-target 标签。

## 5. 强制实验协议（三臂 + 对照）

所有方法必须同时报告：

| 臂 | 名称 | 定义 |
|---|---|---|
| T0 | GTN from scratch | 现有 242-fold LOSO 基线 |
| T1 | Aux pretrain + GTN fine-tune | 方式 A |
| T2 | Aux pretrain + frozen encoder + GTN head | 方式 C |
| T3 | Domain alignment + GTN-only head | 方式 B（N2P3-Net） |

要求：

1. T0/T1/T2/T3 使用**完全相同的 folds、metrics、seeds、早停规则**；
2. 主指标：9 选 1 命中率、balanced accuracy、AUC；
3. 差异显著性用 `paired_permutation_test`，不得只看均值差；
4. 同时报告辅助域在预训练后、GTN 微调前/后的单试次 AUC，作为域漂移观察量；
5. 若 T1/T2/T3 未显著超过 T0，默认结论为「辅助数据无增益」，退回 T0，不强行采用辅助预训练。

## 6. 工程实现要求

- 预训练权重必须保存为 checkpoint（`state_dict`），禁止只存在内存中。
- 加载时使用 `strict=False` + 显式 `load_mapping`，记录成功/跳过/形状不匹配的 key。
- 通道数不一致时：
  - 优先把辅助数据裁剪到与 GTN 相同的通道子集（如 Fz/Cz/Pz）再预训练；
  - 无法对齐的层（如 EEGNet 的空间 depthwise conv、Inception 第一层）跳过加载并在报告中列明；
  - N2P3-Net 使用坐标式通道适配，允许异构通道，但仍需记录 mask。
- 辅助预训练采用固定 epoch（不做依赖 GTN 测试的 early stopping），或使用辅助域自身 LOSO 选 epoch。
- 微调学习率建议为预训练阶段的 1/5–1/10；是否冻结早期层作为独立消融轴。
- 实验记录需包含：辅助数据集、预处理口径、epoch、lr、batch、加载/冻结的层、GTN 口径指标、显著性 p 值。

## 7. 验收判定

采用辅助数据的前提是同时满足：

1. 相对 T0，T1/T2/T3 在 GTN 主指标上有**配对置换检验显著**提升；
2. GTN 测试口径没有任何泄漏；
3. 辅助数据对主情景的「污染」可解释：T1 若显著优于 T2，说明微调必要；T2 若已优于 T0，说明预训练表示本身有迁移价值；
4. 最终部署/报告模型仍以 GTN（及后续自有成人数据）微调后的参数为准，不以辅助预训练权重为准。

## 8. 与项目文档的关系

- constitution P9 给出原则；
- blueprint Stage 4 规定具体损失与训练配方；
- roadmap Phase 3 规定交付与验收；
- 本文档规定「什么能做、什么不能做、怎么判胜」。
