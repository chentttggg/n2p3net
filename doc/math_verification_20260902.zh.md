# 数学严格推导与反例验证记录：P0 / P1 / P2

_版本：2026-09-03 闭合修订 · 文档角色：数学与反例记录 · 验证脚本：`tmp/counterexample_verification_20260902.py` · 运行环境：.venv torch 2.13.0+cpu · 结果：14/14 通过_

> 📌 **当前状态：** 推导已用于 `ef7a40e` 的 P0/P1/P2 framework/contract 实现；本文件记录
> 数学与反例边界，不独立证明真实数据增益。活动实验门禁以
> [`research_program.zh.md`](research_program.zh.md) 为准，代码验收以
> [`framework_acceptance_20260903.zh.md`](framework_acceptance_20260903.zh.md) 为准。

本文档做两件事：(1) 从第一性原理给出三项机制的严格推导；(2) 记录数值反例验证结果，其中三项"通过"实际暴露了实施清单 v1 的缺陷，对应修正已回写 [`implementation_plan_20260902.zh.md`](implementation_plan_20260902.zh.md)。正式测试映射与当前实现状态见 [`framework_acceptance_20260903.zh.md`](framework_acceptance_20260903.zh.md)。

---

## 1. P2 决策级损失：完整推导

### 1.1 设定

一个 decision 含 N 个 epoch。epoch i 归属候选 $c_i \in V = \{0,\dots,8\}$，候选 d 的 epoch 数 $n_d = |\{i: c_i = d\}|$。编码器给出标量证据 $e_i = f_\theta(x_i)$。聚合与 posterior：

$$
s_d = \frac{1}{n_d}\sum_{i:c_i=d} e_i,\qquad
P_d = \frac{e^{s_d/\tau}}{\sum_{d'\in V} e^{s_{d'}/\tau}},\qquad
L_{cand} = -\log P_{d^*}.
$$

### 1.2 梯度推导

链式法则经 softmax-CE 的标准结果 $\partial L/\partial s_d = (P_d - \mathbb 1[d=d^*])/\tau$，再经 mean 聚合 $\partial s_d/\partial e_i = \mathbb 1[c_i=d]/n_d$：

$$
\frac{\partial L_{cand}}{\partial e_i}
= \frac{P_{c_i} - \mathbb 1[c_i = d^*]}{\tau\, n_{c_i}}.
$$

若训练目标覆盖 repetition 轴，必须显式定义
\[
L_{\mathrm{prefix}}=\sum_{r=1}^{R_{\max}}\alpha_rL_{candidate,r},
\qquad \alpha_r\ge0,\quad\sum_r\alpha_r=1.
\]
固定 `hit@8` surrogate 预注册为 \(\alpha_8=1\)；整条曲线 arm 预注册为
\(\alpha_r=1/R_{\max}\)。由于 mean 聚合不会自动随 evidence 数量 sharpening，
evidence-aware arm 需另外预注册 \(n^{eff}_{d,r}=B_{d,r}^2/Q_{d,r}\) 与
\(s_{d,r}=\bar e_{d,r}(n^{eff}_{d,r})^{\beta}\)（默认 \(\beta=0.5\)，mean 对照为
\(\beta=0\)），或等价的 \(\tau_r=\tau_0/r^{\beta}\)；不得由 test truth 调节。

**数值验证**：与 autograd 比对，最大误差 1.49e-08（PASS #1）。

三条推论（均已数值确认）：

1. **梯度聚焦**（PASS #2）：错误候选 d 的总偏导质量为 $P_d/\tau$；对其执行梯度下降会把 evidence 压低，当前 posterior 排第二的混淆候选因此获得最大压低驱动力。trial CE 的梯度 $\partial L_{trial}/\partial e_i = \sigma(e_i) - y_i$ 只看自身 logit，无此性质。两者是不同的目标函数。
2. **判对饱和**：真候选梯度总质量 $-(1-P_{d^*})/\tau$，posterior 已判对时梯度自动衰减，容量分配给未解决的 decision。
3. **count-neutral**：$1/n_{c_i}$ 因子使不等候选计数下每候选的总梯度只由 posterior 决定，与出现次数无关。

### 1.3 常数偏置不变性（反例 2 的可微版本）

对 $e_i \mapsto e_i + b$：mean 聚合下 $s_d \mapsto s_d + b$，softmax 平移不变，$P$ 不变（PASS #3）。sum 聚合下 $s_d \mapsto s_d + b\,n_d$，不等计数时排序被破坏（PASS #4，数值确认不变性不成立）。

**结论（强制）**：可微决策损失的聚合算子必须是 count-neutral mean，sum 聚合直接违反反例 2。这与 `candidate_evidence.py` 冻结的 `mean` 默认语义一致。

### 1.4 置换等变性（反例 1）

候选与真值经同一置换 π 重命名后，softmax 按索引置换，$L$ 逐位不变（PASS #5，atol=1e-6）。实现中禁止任何按候选编码值的运算（排序、tie-break、embedding）。

### 1.5 覆盖不完整攻击（反例 4）——实施清单 v1 缺陷 A

数值实验：将候选 7 的全部 epoch 从 decision 中移除，然后只对**观测到的候选**做 softmax。结果 $P_{d^*}$ 从 0.0531 膨胀到 0.0665（PASS #6——攻击成立）。

这直接击中实施清单 v1 `DecisionLevelLoss` 伪代码中的 `torch.unique(candidate_membership)`：unique 会静默缩小 vocabulary，训练分布把"缺覆盖"当成"更容易的 decision"，模型学到被膨胀的 posterior，评估期遇到完整 vocabulary 时校准全错。

**修正（已回写实施清单）**：`DecisionLevelLoss` 构造时接收冻结 vocabulary；forward 检查观测候选集合，覆盖不完整的 decision 直接进入 `incomplete` ledger，不产生 posterior、CE、argmax 或 stopping decision。空桶记为 `-inf` 只能作为评分诊断值，不能在剩余候选上继续 softmax；这是反例 4 在训练路径上的延伸。

### 1.6 λ 搜索范围（非理论上界）

给定 membership 和目标候选，$y_i=\mathbb 1[c_i=d^*]$，所以一个 decision label 与 membership 一起确定所有 trial binary labels；trial labels 并非 N 份独立监督。保留 `L_trial` 基于优化稳定性、局部监督、有限样本正则化和 trial-level 校准的预注册设计选择。数值例（N=32 合成 decision）梯度比 ≈ 0.111（PASS #7）只是单点 sanity check：实际比例随 temperature、pos_weight、candidate count、logit 尺度、decision 间方差与 batch 组成变化，**不能由信息比特数推导出通用 λ 区间**。$\lambda_{set}\in[0.1,0.5]$ 的地位是预注册搜索范围。

`hit@8` 是 prefix `r=8` 后一次 9 选 1 top-1 selection accuracy，不是 top-8 ranking。概率损失与 0-1 决策损失可反向：模型 A 的正确/错误 decision 比例为 0.60/0.40，真类概率分别为 0.60/0.40，平均 NLL 约 0.673；模型 B 为 0.59/0.41，真类概率为 0.99/0.20（错误样本的最大非真类概率 0.30），平均 NLL 约 0.665，但命中率从 0.60 降至 0.59。反向原因是损失目标不同，不是 rank 8/9 事件。

---

## 2. P1 drop-to-consensus：完整推导

### 2.1 恒等初始化

$\Delta$ 的 gate 零初始化时 $\tanh(0)=0$，残差项逐位为零，$H_d = Z_{inv}$ 位精确成立（PASS #8，`torch.equal`）。反例 10 的 zero-init 等价性在数值上是位级的，不是近似的。

### 2.2 操作点覆盖引理（及其效力边界）

**引理**：设训练期 $H_d = Z + m\,g\,\Delta(Z)$，$m\sim\text{Bernoulli}(1-p)$，推理期 $H = Z$。当 $p>0$ 时训练目标包含分量 $p\cdot\mathbb E[L \mid m=0]$，trunk 参数因此直接收到无残差路径的梯度；当 $p=0$（旧 DomainAdapterBank）该分量为零，推理操作点从未被训练——off-manifold。

**效力边界（不可夸大）**：该引理只消除"推理操作点从未被训练"，不给出更多。训练目标是 $(1-p)L(m{=}1)+pL(m{=}0)$ 的混合而推理只用 $m=0$ 分量，因此 (a) 训练/推理分布不一致仍存在，(b) trunk 不保证收敛到最优 consensus，(c) 泛化结论只能由 held-out hit@R 判定。每个 run 必须分别报告 $L(m{=}0)$、$L(m{=}1)$ 及其差值作为机制诊断。

数值确认：m=0 的训练路径与无残差推理路径 `torch.equal`（PASS #9）；p=0 时残差引入的特征偏移相对范数 0.080，非零（PASS #10）——旧 bank off-manifold 间隙的直接测量。

**推论**：反例 15（trunk 单独路径与推理路径逐位一致）在 mask 乘法实现下自动成立；mask 实现优先于 `if` 分支（保持 RNG 消耗一致）。

**p 的语义**：p 只控制共识路径被训练的频率。残差幅度由 $\rho\tanh(a)\|\Delta(Z)\|$ 决定，与 p 无关；且 `max_residual` 只界定标量门 $|\rho\tanh(a)|$，不界定 $\|\Delta(Z)\|$。按样本、在 feature 非 batch 维度上固定 $\epsilon_{res}=10^{-6}$，计算
\[
r_j=\frac{\|m_j\rho\tanh(a)\Delta(Z_j)\|_2}{\max(\|Z_j\|_2,\epsilon_{res})}.
\]
报告 mean/q50/q95/q99/max，以 max 与 `r_max=0.5` 判定；source residual 检查固定 $m_j=1$。任一超限使整次 run `evaluation_failed`/invalid，禁止删除样本；零分母零分子定义为 0，零分母非零分子必超限。

### 2.3 实施清单 v1 缺陷 B：残差双加

`FeatureResidualAdapter.forward` 的语义是 $x \mapsto x + g\Delta(x)$（内含加法）。实施清单 v1 的伪代码写了

```python
residual = self.source_domain_residuals[domain_idx](features)
adapted = features + residual          # 错：features 被加了两次
```

数值确认 `features + adapter(features) != adapter(features)`（PASS #11）。正确写法二选一：

```python
adapted = self.source_domain_residuals[domain_idx](features)          # 直接用 forward
# 或显式残差：
adapted = features + scale * self.source_domain_residuals[domain_idx].residual(features)
```

drop mask 版本必须用第二种（residual() + 显式 scale + mask），否则 mask 无处安放：

```python
delta = ad.scale_tensor(features.dtype) * ad.residual(features)
adapted = features + m * delta        # m: (B,1,1) per-sample Bernoulli
```

**修正已回写实施清单**；per-sample mask（`source_drop_mode="sample"`）为默认，batch 级 mask 只作对照。

### 2.4 梯度流备注

m 是不可微离散 mask，直通即可（不需要 STE）：目标不是学 m，而是让两个操作点都被训练。$\mathbb E_m[\nabla_\theta L] = (1-p)\nabla_\theta L|_{m=1} + p\nabla_\theta L|_{m=0}$，p 就是两操作点的梯度混合权重——这给出 p 的解释：p 越大 trunk 共识路径权重越高。p ∈ [0.2,0.5] 保持两端都非退化。

---

## 3. P0 Split-BN：完整推导

### 3.1 混合统计偏差的量化

设域 0 特征 ~ N(0,1)，域 1 特征 ~ N(5,9)（不同采集尺度）。pooled batch 统计归一化后，域 0 特征变为 mean=-0.75、std=0.31（PASS #12）——既偏移又压缩，且该失配发生在网络内部（BN 层），输入层 mean/std 更换无法触及。这正是"输入 mean/std 只恢复 +0.07pp"与"BN 失配可能是大头"并存的机制解释。Split-BN（共享 affine、per-domain 统计）恢复各域标准化（PASS #13）。

### 3.2 Split-BN 作用域修正——实施清单 v1 缺陷 C（范围不足）

代码核查：N2P3Net 含 **三处** BatchNorm——`st_temporal_norm`（BatchNorm2d, n2p3net.py:615）、`st_spatial_norm`（BatchNorm2d, :624）、以及**每个 MST branch 内的 BatchNorm1d**（:285）。实施清单 v1 只列了前两处。MST branch 的 BN 同样吃混合域统计，Split-BN 臂必须三处同换，否则诊断结论被稀释。已回写实施清单。

Split-BN 的 buffer-swap 只能改变推理时采用的 running statistics，不能回溯训练期每个域独立统计对梯度、优化轨迹和正则化的影响；因此 swap 不变最多否定“推理 buffer 选择是主要机制”。unseen target 必须在 `pooled_source`、`source_weighted` 和 `target_calibration` 三种 buffer 来源中预注册其一，并将来源对应的 estimand 写入 ledger。

### 3.3 P0×P2 交叉约束——实施清单 v1 缺陷 D

P2 要求按 decision 分组组 batch；单 decision batch 只有 ~32 个 trial 且类别失衡（1 目标候选 vs 8）。数值确认：单 decision 尺寸的 batch 其 BN 批统计方差是 256-batch 的 8.1 倍（PASS #14）。若 P2 的 sampler 天真地"一个 batch = 一个 decision"，BN 统计（无论 pooled 还是 split）都会高方差漂移，P2 的增益会被 BN 噪声吃掉，且该失效与反例 9（BN buffer 随 batch 顺序产生未记录差异）同源。

**强制约束（已回写实施清单）**：P2 的 batch 必须由多个完整 decision 拼成（batch = K 个完整 decision，K·N ≥ 128），decision 边界以 index 张量传入损失函数；禁止单 decision batch 训练 BN 网络。

---

## 4. 汇总：v1 实施清单的四处修正

| # | 缺陷 | 反例证据 | 修正 |
|---|---|---|---|
| A | `torch.unique` 缩小 vocabulary | PASS #6：posterior 膨胀 0.0531→0.0665 | DecisionLevelLoss 绑定冻结 vocabulary，覆盖不完整 fail-closed |
| B | `features + adapter(features)` 双加 trunk | PASS #11 | 用 `residual()` + 显式 scale + per-sample mask |
| C | Split-BN 漏掉 MST branch BN | 代码核查 :285 | 三处 BN 同换 |
| D | 单 decision batch 破坏 BN 统计 | PASS #14：方差 8.1× | batch = K 个完整 decision，K·N ≥ 128 |

四处修正均不改变 2026-09-02b 架构裁决本身；它们是裁决到实现之间的契约收紧。

## 5. 外部审查修正记录（2026-09-02 第二轮，10 项）

用户方审查提出 10 项数学/逻辑问题，全部成立，修正已落盘：

| # | 问题 | 修正位置 |
|---|---|---|
| 1 | D0 对照 p 方向写反（Bernoulli(1-p) 下 p=1 是全丢弃）；"p 小则残差幅度小"混淆频率与幅度 | 实施清单 D0 改 p=0.0；p 语义澄清落三处 |
| 2 | "无 off-manifold 间隙"说得过强：混合目标 (1-p)L(m=1)+pL(m=0) 只保证推理路径被训练过 | "定理"降级为"操作点覆盖引理"+效力边界；强制分别报告 L(m=0)/L(m=1) |
| 3 | max_residual 只限标量门不限 ‖Δ(Z)‖，且 ‖Z‖=0/极小时比值未定义或不稳定 | 固定逐样本 feature 非 batch 维 L2、ε_res=1e-6、r=‖δ‖/max(‖Z‖,ε_res)，报告分位数并以 max 与 r_max=0.5 判定；source 检查固定 m=1，超限整次 run fail-closed |
| 4 | L_candidate 是 top-1 CE surrogate，不是 hit@R 的直接可微目标 | 全部"直接优化 hit@R"表述改为 ranking surrogate；晋升只看 operational hit@R |
| 5 | λ∈[0.1,0.5] 不能由信息比特数推导 | 降级为预注册搜索范围；比特数论证只保留"必须是加项" |
| 6 | 规范梯度公式漏 1/τ；"最强负梯度"符号表述不准 | 规范补 τ；改为"偏导为正、梯度下降压低其 evidence" |
| 7 | invariant/private 分解无可识别性保证（旋转不唯一、priv 可携带 task、可塌缩） | 规范 2.2 增"可识别性边界"：目标约束而非定理，四项证据缺一不可 |
| 8 | Split-BN ≥5pp 是工程门槛不是因果结论 | 验收改为成对 seed/fold + CI；反例 17 标注必要非充分 |
| 9 | learned ρ_ω 可能给缺失候选有限分数，且把 -inf 后继续 softmax 错当 fail-closed | 规范 4.2 分离评分诊断与状态门禁：B=0 可记 -inf，但 C=0 时 posterior/CE/argmax/stopping 均为 ⊥，decision 进入 incomplete ledger |
| 10 | 文献跨任务外推（FiLM=语音、MetaSleep=睡眠、Reptile=运动想象、SpellerSSL=自身协议） | 架构决策文档增"文献外推边界"强制声明：只支持"值得实验"，不支持"已是最优" |

## 6. 验证脚本维护

`tmp/counterexample_verification_20260902.py` 是一次性科研验证产物（tmp 层级）；P0-P2 实施时其中的断言必须迁移为 `tests/` 下的正式测试（对照实施清单第四节测试清单），并补两条本轮新增断言：(a) 缺失候选屏蔽（B=0 → -inf）；(b) 范数比 r 超限 fail-closed。迁移完成后本脚本可删除。

## 7. 2026-09-03 数学闭合修正记录

| # | 残留问题 | 闭合规则 |
|---|---|---|
| 1 | `hit@R` 被写成 top-R/rank 事件 | 统一为 prefix `r=R` 后的一次 9-choice top-1 selection accuracy；NLL 与 0-1 命中反向只归因于损失目标不等价 |
| 2 | 空候选 `-inf` 被误称 fail-closed | `-inf` 仅为评分诊断；coverage 不完整时 posterior/CE/argmax/stopping 均为 `⊥`，decision=`incomplete` 并进入 ledger |
| 3 | decision label 少于 trial label 被当作信息论结论 | 明确 `y_i=1[c_i=d*]`；保留 trial CE 是优化稳定性、局部监督、正则化和校准的预注册选择 |
| 4 | 横向集成“无新表达力/必然塌缩” | 改为成本、方差和多样性约束下暂不采纳；表达力与塌缩必须实证检验 |
| 5 | buffer-swap 被当作 Split-BN 训练作用的否证 | 只检验推理 buffer 选择；训练期统计对梯度/轨迹/正则化的影响仍需独立证据；target buffer 来源必须预注册并标注 estimand |
| 6 | residual 范数比在零分母和 dropout 下不闭合 | 固定逐样本 feature 非 batch 维 L2、`eps_res=1e-6`、max 判定、`r_max=0.5`、source `m=1` counterfactual；任一超限整次 run invalid |
| 7 | P2 缺少 repetition 轴 | 预注册 `L_prefix=sum alpha_r L_candidate,r`；固定预算 `alpha_8=1`，曲线 arm `alpha_r=1/R_max`；evidence sharpening 另设 beta/tau arm |
| 8 | 实施文档漏 `1/tau` 或写错 choice 数 | 统一为显式 `tau>0`、`1/tau` 梯度和冻结 vocabulary 的 9-choice softmax |
