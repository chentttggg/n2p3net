# 无生理时窗先验的完整时序展开

状态：BI2014a 探索性消融已完成；本文后半的“预注册”保留为历史方案。
确认性结论和 GTN 后续规则见 `research_program.zh.md`。

## 结论先行

本轮不再把 P300 窗、N200 窗、pre-stimulus reference、latency bank 或固定
segment 边界写进 head。保留已经验证的数据合同和 MS-EEGNet 风格主干，只比较
主干输出之后的 readout。这里的“无先验”是 head 级别的严格限定，不表示整个
网络没有卷积、局部性或第一阶段 `4x` 平均池化等结构假设。

在 128 Hz、128 点输入下，主干输出

```text
H in R^(B x S x T),  S=4, T=32.
```

`ms_flatten` 继续以 8 点固定平均池化把 `T: 32 -> 4`。新候选直接展开
`H` 的全部 128 个坐标；因此它保留的是主干分辨率，不是原始 128 Hz 分辨率。

## 固定池化为何不可恢复

把第二阶段固定池化记为线性映射

```text
P: R^T -> R^M,  M<T.
```

由秩-零化度定理，

```text
dim ker(P) = T - rank(P) >= T - M > 0.
```

所以存在 `delta != 0` 使 `P(x+delta)=P(x)`。对 8 点平均池化，同一 bin 内
第 2 点和第 5 点的等幅脉冲得到完全相同的池化结果；此后任何分类器都不可能
恢复两者的位置。测试把这个反例直接编码为张量，而不是只在文档中陈述。

## 模型 A：完整线性模板

```text
x = vec(H) in R^(S*T)
z = W0 x + b.
```

展开 `vec` 是坐标重排，因此不产生碰撞。对任意 `x1 != x2`，取
`w=x1-x2`，则

```text
w^T x1 - w^T x2 = ||x1-x2||_2^2 > 0.
```

这只证明任意一对不同表示可以被某个线性函数区分，不能推出任意带标签样本集
都线性可分，也不能推出有限样本泛化更好。这个 head 的实际含义是学习一个完全
自由的、位置相关的主干特征模板：

```text
z_k = b_k + sum_(s,t) W_(k,s,t) H_(s,t).
```

对应代码模式为 `full_unfold`。

## 模型 B：低秩二阶残差

线性模板不能表达“两个位置共同出现”这类二阶关系。加入 rank `R` 的分解二阶
项：

```text
l = A x,  r = B x
z = W0 x + b + C (l .* r).
```

对类别 `k`，

```text
z_k = b_k + w_k^T x + sum_(j=1)^R C_(k,j) (a_j^T x)(b_j^T x)
    = b_k + w_k^T x + x^T Q_k x,
Q_k = sum_(j=1)^R C_(k,j) a_j b_j^T.
```

它不指定哪些时点或尺度应交互，只把二阶矩阵限制为低秩。令 `C=0` 时严格退化
为模型 A，所以假设类包含完整线性模板。实现也把 `C` 零初始化，使优化从模型 A
的函数开始。

严格增益可由 XNOR 反例说明。四个点
`(1,1),(1,-1),(-1,1),(-1,-1)` 按两坐标同号与否分类，任何仿射超平面都不能
完全分开：把四个严格分类不等式中同号两式相加得到 `2b>0`，异号两式相加却
得到 `2b<0`，矛盾。一个 rank-1 项 `x1*x2` 的符号则恰好给出分类。对应代码模式为
`quadratic_full_unfold`。

这不是新的通用算法。它是 low-rank bilinear pooling / factorized quadratic
classification 在紧凑 ERP readout 上的残差化应用。

## 模型 C：同预算普通 MLP 对照

二阶候选比线性候选参数更多。为排除“只因容量变大”的解释，加入

```text
z = W0 x + b + V GELU(M x + c).
```

`V` 同样零初始化。hidden width 取 16，使其参数量与 rank-8 二阶 head 相差
不到 1%。它不是晋升候选，而是必要的机制对照；对应
`mlp_full_unfold`。

## 参数预算

对 16-channel、128-sample 输入：

| Head / model | Parameters | 可检验问题 |
|---|---:|---|
| MS-EEGNet `ms_flatten` | 1,282 | 固定二级池化基线 |
| `full_unfold` | 1,506 | 消除二级池化碰撞是否有效 |
| `quadratic_full_unfold`, rank 8 | 3,570 | 显式低秩二阶关系是否有效 |
| `mlp_full_unfold`, hidden 16 | 3,602 | 相近容量的一般非线性是否同样有效 |
| EEGNet | 1,490 | 当前整体冠军、近似参数匹配线性对照 |

## 文献检索后的取舍

检索覆盖 compact CNN、P300 专用网络、动态池化、attention/TCN、Transformer
和双线性 readout。以下只支持模型选择，不把不同数据集或指标的数字当成本项目
LOSO 证据。

1. [EEGNet](https://doi.org/10.1088/1741-2552/aace8c) 官方实现使用平均池化后
   Flatten，支持“保留位置再分类”是成熟做法，但其池化仍不可逆。
2. [Learnable Dynamic Temporal Pooling](https://doi.org/10.1609/aaai.v35i9.17008)
   用 soft-DTW 学习分段原型，明确针对全局池化丢失时序信息；但仍需预设输出段数，
   还引入专用对齐算子，因此不适合作为本轮最小 head 变量。
3. [Lightweight multi-scale CNN for P300](https://doi.org/10.3389/fnhum.2021.655840)、
   [A few filters are enough](https://doi.org/10.1016/j.neucom.2020.10.104) 和
   [large multi-subject P300 CNN evaluation](https://doi.org/10.1016/j.bspc.2019.101837)
   共同支持先检验紧凑模型，而不是默认扩展到大容量序列网络。
4. [Subject-independent P300 invariant patterns](https://doi.org/10.1109/TNSRE.2021.3083548)
   说明跨被试表示学习本身很重要，但其协议不能替代当前 64-subject matched LOSO。
5. [ST-CapsNet](https://doi.org/10.1109/TNSRE.2023.3237319)、
   [ATCNet](https://doi.org/10.1109/TII.2022.3197419) 和
   [EEG Conformer](https://doi.org/10.1109/TNSRE.2022.3230250) 表明 attention、
   TCN 和 Transformer 能保留序列并建模长程依赖；它们同时改变 trunk、容量和
   优化，本轮若混入就无法归因于展开策略。
6. [Low-Rank Bilinear Pooling](https://doi.org/10.1109/CVPR.2017.743) 给出分解
   二阶分类器的已有方法学基础。这里的贡献只能表述为 ERP readout 的具体适配与
   matched ablation，不能声称发明双线性机制。
7. 近期 P300 模型
   [ATCRN](https://doi.org/10.1016/j.jneumeth.2026.110727) 与
   [ST-GraphTRNet](https://doi.org/10.1088/1741-2552/ae3d68) 使用更复杂的
   TCN/attention 或 graph/transformer。其 repetition、character、within-subject
   或 cross-subject 指标不能和本项目 trial AUC/BACC 直接排序。
8. EEG 深度学习综述指出模型比较长期受数据规模、预处理与验证协议不一致影响：
   [Roy et al., 2019](https://doi.org/10.1088/1741-2552/ab0ab5)。因此外部论文只
   用于生成假设，晋升仍由同缓存、同折、同 QC、同训练预算的结果决定。

## 预注册消融与判定

首轮同时训练 `full_unfold`、`quadratic_full_unfold`、`mlp_full_unfold`、
MS-EEGNet 和 EEGNet，复用已冻结的 64 个 fold-local QC 模型。逐外层被试比较
AUC/BACC，使用被试级 bootstrap 区间和带 plus-one 修正的配对 sign-flip 检验。

判定顺序：

1. `full_unfold` 对 MS-EEGNet 检验“去掉二级池化”的净效应，并对 EEGNet 检验
   近似参数预算下是否有竞争力。
2. 二阶 head 只与 `full_unfold` 比较其增量；同时与 MLP 比较机制特异性。
3. 若二阶和 MLP 都改善且差异不清楚，只能结论为“容量/非线性有效”，不能结论为
   二阶交互有效。
4. 若只二阶改善，才获得低秩交互的支持；若完整展开退化，默认仍保持
   `ms_flatten`，不因理论表达力更强而晋升。
5. 外层测试结果只用于最终报告，不用于回头选择 rank、hidden width、epoch 或
   threshold。

## 已完成结果与后续边界

完整 64-subject / 61,015-trial BI2014a 结果：

| Head | AUC | BACC |
|---|---:|---:|
| linear `full_unfold` | 0.745109 | 0.677407 |
| quadratic `full_unfold` | 0.740752 | 0.675562 |
| MLP `full_unfold` | 0.744027 | 0.675290 |
| EEGNet | 0.739513 | 0.675382 |
| MS-EEGNet | 0.734211 | 0.667776 |

审计重算中，linear full-unfold 相对 MS 的 AUC 差约 `+0.01090`，相对 EEGNet
约 `+0.00560`；后者 BACC 区间跨零。该轮支持“保留主干时间坐标”，不支持
低秩二阶或 MLP 机制。K35 仅由后续 inner sensitivity 注册，未取得独立确认。

BI2014a outer test 已被后续 epoch/patience/sampling-rate 运行重复查看，不再用于
选择模型。下一次裁决必须在预注册 GTN chronological 9-choice 或新的独立数据上进行。
