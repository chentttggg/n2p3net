# 统一 trial → 候选集合决策：数学、反例与代码迁移方案

日期：2026-09-16。范围：统一模型的决策端设计；本文件没有实施生产修改，也没有改变正在运行的云任务。已核对当前活动研究规范，以及 `decision_loss.py`、`candidate_set_loss.py`、`candidate_evidence.py`、`evaluation.py`、个人校准接口与实际训练调用点。

## 1. 主选择

首选**共享神经网络 trial 证据 + 真实 membership 矩阵 + 可微求和决策头**。聚合本身不增训练参数，trial 编码器和证据头仍同时接受二分类与决策级损失的梯度。direct9、row/column、未来合法子集闪烁共用同一个矩阵公式。协议差别由输入的物理 membership 表达，不由“选哪个模型”的分支表达。

小型 learned set scorer 作为残差机制对照，暂不作为本人少量校准数据的默认头。理由是当前主要缺口为任务表示、概率尺度、缺失机制和训练接线；额外非线性不能替代这些约束。这个选择没有删掉已有可复用模块；只有固定头解释不了的、经过独立验证的交互残差，才需要额外可学习容量。

## 2. 一个候选空间，两种以上刺激协议

对 decision d，真实选择集合为 C_d={1,…,K_d}，排程包含 N_d 个事件。定义

\[
A_{ic}=\mathbf1[\text{事件 }i\text{ 的物理刺激集合包含选择 }c],
\quad A\in\{0,1\}^{N\times K}.
\]

`stimulus_code` 是闪烁事件的编码，`candidate_id` 是最终选择的身份，两者不能混用。真值 c* 只用于训练/评分，二分类标签必须满足 y_i=A_{i,c*}。membership builder 只读排程，不读真值生成 A。候选词表由协议冻结，不能从本批出现的编码或已保留 EEG 行推断。

- direct9：K=9，每个事件仅一个 A_ic=1，45-flash block 是五次完整候选排列。
- R×C row/column：K=R·C，候选是字符位置 (r,c)，事件是 R+C 个行/列刺激之一。某行闪烁时该行的 C 个字符 membership 为 1；某列闪烁时该列的 R 个字符为 1。不能把 R+C 个刺激代码误当成 R·C 个选择。
- 一般子集闪烁：沿用同一 A；若两个候选的观测 membership 列完全相同，当前数据无法区分它们。必须保留并列/不可识别状态，不能以标签编号破平。

当前 `data.candidate_task.CandidateTaskContract` 仅支持 row_column；`transfer.decision_loss.DirectDecisionMetadata` 仅支持 direct membership。建议在现有 candidate task / decision metadata 中统一这两个轴和构造规则，不新建第二套协议注册系统。大批训练可存稀疏 membership 或块索引，无需强制物化巨大稠密矩阵；其数学对象和序列化签名一致即可。

## 3. LLR 共同式及其适用边界

在给定协议/排程上下文 a_i 后，若事件条件密度只通过 y_i 依赖真实目标，且事件在给定目标及上下文后条件独立，定义

\[
e_i=\log\frac{p(x_i\mid y_i=1,a_i)}{p(x_i\mid y_i=0,a_i)}.
\]

则

\[
\log p(X\mid c,A)=\sum_i\log p(x_i\mid0,a_i)+\sum_i A_{ic}e_i,
\qquad s_c=\log\pi_c+(A^\top e)_c,
\qquad p(c\mid X,A)=\operatorname{softmax}(s)_c.
\]

公共减项与 c 无关，因此只需 Aᵀe。对当前猜数字任务，π_c=1/K；语言先验等只能作为显式、独立来源的协议输入，不从测试答案或校准数字频数偷学。这个后验推导不能自动赋予任意神经网络 logit、任意加权均值以概率含义；未核验条件独立与校准时，应叫**LLR 兼容判别分数**。

row/column 下，令 u_r 为第 r 行刺激证据之和、v_c 为第 c 列之和，则 s_(r,c)=u_r+v_c+log π_(r,c)。当 π_(r,c)=π_rπ_c 时：

\[
p(r,c\mid X)=p(r\mid X)p(c\mid X),\qquad
L_{joint}=L_{row}+L_{column}.
\]

这是联合候选 softmax 与两轴 CE 的精确等价，不是两个不同科学机制。一般非因子化先验、非行列子集协议不能强行沿用两轴分解。

迁移时还有一个尺度陷阱：完整 R 轮下，旧两轴均值分数是 (u_r+v_c)/R，而按每个字符的 membership 总数直接取均值会得到 (u_r+v_c)/(2R)，差一倍。新 sum 头若只为复现旧均值概率，应取 τ_new=R·τ_old；直接对每个字符取均值则需 τ_mean=τ_old/2。新研究默认 sum/τ=1 与这种历史复现是不同声明，不能通过同一个“mean”名字掩盖差异。

## 4. 二分类先验、类权重与等覆盖

设 trial 网络输出 f=logit q，训练抽样正例先验为 ρ，二分类权重为 w_1、w_0。在模型充分表达、优化达到对应期望风险最优、且权重只依赖类别等条件下：

\[
f^*(x)=\log\frac{p_{train}(x\mid1)}{p_{train}(x\mid0)}
+\operatorname{logit}(\rho)+\log(w_1/w_0).
\]

因此 b=logit(ρ)+log(w_1/w_0) 是到训练分布 LLR 的偏置修正。若多源风险权重改变了类条件域混合，单个全局 b 不能把训练分布 LLR 变成目标分布 LLR；目标域偏移也不能靠这一公式消除。加入决策损失后，f 不再必然是二分类风险的精确最优后验，仍需独立校准验证。

令 r_i 为外部质量门禁给出的事件可用性。忽略 b 是否影响候选结果，真正条件是

\[
A^\top(r\odot b)=k\mathbf1,
\]

其中 b 可为每事件不同的偏置。常数 b 时，退化为每个候选的有效 membership 数相同。完整 direct9 的 R 轮满足每列 R；完整 row/column 每个字符满足 2R。缺失、非平衡排程、协议依赖偏置或事件权重会破坏这一条件。

反例已计算：两个候选的 membership 次数为 1 与 2，网络 logit=(0.2,-0.2,-0.2)，ρ=1/9。未修正总分=(0.2,-0.4)，选第一个；减去 logit(1/9) 后总分=(2.27944,3.75888)，选第二个。不能声称“logit 常数永远不影响解码”。

等覆盖时 mean 与 sum 的 argmax 相同，但 softmax 概率不同。正比例缩放在均匀先验下保持 argmax，也会改变置信度；非均匀先验下，缩放证据而不同比例调整先验项可能连 argmax 都改变。若要保持原概率，温度必须按对应比例变换。**不能用固定温度的均值 posterior 解释每增加一轮积累了多少独立信息。**

新主头采用 sum 作为统计明确的基式；冻结旧 mean 模型按原声明解释，不能重标成 sum。对完整 R5 的旧/新纯聚合对照先确认 argmax 一致，再分别评价 NLL/Brier、校准和拒判。冻结单一正温度只允许从合法校准证据拟合；不能读取未知目标测试重新校准。训练温度、后处理温度和证据类型必须分开保存。

## 5. 重复测量与概率不能过度自信

sum 的概率解释要求前述独立性。共享低频漂移、注意状态、相近刺激间隔、重叠 epoch、同一数据的重复索引都可能造成相关。把同一证据复制一遍，会将 softmax(1,0) 的最大值从 0.73106 推到 0.88080，但没有增加新 EEG 信息。

在简化的等相关高斯模型中，组内 n 次测量协方差为 σ²[(1-ρ_e)I+ρ_e11ᵀ]，两类均值沿全一方向不同，则联合 LLR 为独立单次 LLR 总和除以 1+(n-1)ρ_e。ρ_e→1 时重复几乎不增加信息。该公式只说明相关性改变证据尺度，不建议从很少的本人 decision 生硬估计统一 ρ_e 后应用所有人/协议。

主方案先保留 sum 判别式并在独立 session 上检验概率可靠性；高置信停止暂不由该假设自动授权。实测相关造成的剩余误差可比较：冻结温度、显式低维相关模型、或下述共享 set 残差。不能将采样器重复 draw 当成决策中的新物理事件。

## 6. 缺失证据、非随机缺失及不能靠拒绝逃避损失

完整排程中的每个事件都有独立身份，r_i=0 只表示该次 EEG 不可用，A 的行和完整候选列仍保留。缺失导 mask 与缺失 trial 分开；只要一个 trial 满足既定输入合同，masked EEG 可以继续给出证据。

若缺失机制可忽略且分类条件满足，观测证据基式可写为 s=logπ+Aᵀ(r⊙e)。缺一列观测不会让对应真实目标的先验概率自动变为零。A 全无观测时内部数学返回 π，操作输出应是“没有可用证据/拒判”；不能凭非均匀先验称作 EEG 预测。处理 NaN 时先选择/where 清除无效位置，`NaN*0` 不成立。

**缺失非随机时，中性补零不等于正确边缘化。** 逐事件的完整观测模型应包含

\[
e_i^{obs}=\log\frac{p(x_i,R_i=1\mid y_i=1,a_i)}{p(x_i,R_i=1\mid y_i=0,a_i)},\qquad
e_i^{miss}=\log\frac{P(R_i=0\mid y_i=1,a_i)}{P(R_i=0\mid y_i=0,a_i)}.
\]

若 classifier 只在保留样本上校准，则还需区分 p(x|y,R=1) 与 p(x,R=1|y)；两者相差保留概率项。较大目标 ERP 更容易触发幅度门，正是实际可能出现的非随机缺失，标签盲 QC 不会自动解决它。

极简反例：仅刺激候选 1 的事件被拒绝，目标时拒绝概率 0.8、非目标时 0.1；均匀先验下，仅缺失状态便给出 (0.88889,0.11111)。中性 mask 给出 (0.5,0.5)。本例不是要求拿污染预测目标，而是证明中性缺失需要假设。现阶段不从有限校准估计 MNAR “补偿分数”；保留严格 coverage 主终点及类保留率、缺失模式诊断。部分覆盖输出可作单独预声明研究端点，不能替代原 strict operational R5。

用于训练的决定级资格 g_d 必须由冻结数据合同/外部 QC 决定，与网络 θ 无关。不能让网络把难样本、低 confidence 或大 CE 自己标为“不参与损失”。报告可用

\[
L_{requested}=\frac{\sum_d v_d[g_d L_d+(1-g_d)c_{fail,d}]}{\sum_d v_d},
\]

并分别报告可微项、常数失败项与 coverage。失败常数如 log K 只是一项预声明报告约定，不是缺失标签的似然。不能按各模型的成功子集重新分母来比较算法。

**固定失败惩罚本身也挡不住 learned reject escape。** 两个 decision NLL=(0.1,5)，平均 2.55；若模型能拒绝第二个并改记 log9，平均降到 1.14861。因此关键是冻结 g、保留全部请求和失败分母、禁止反向优化可用性；不能只加一个失败罚项便宣称问题解决。

可靠性权重 w_i 也应谨慎：Σ r_iw_iA_ic e_i 通常是幂似然/判别分数，不自动是精确 Bayes；偏置抵消条件随之变为 Aᵀ(r⊙w⊙b)=常数。默认不加入可学习的“抑制难 EEG”权重。

## 7. 置换与禁止学习 code 偏置

令 P 是候选列置换，Q 是事件存储行置换。必须满足

\[
s(AP,e,P^\top\pi)=P^\top s(A,e,\pi),\qquad
s(QA,Qe,\pi)=s(A,e,\pi).
\]

标签、mask、时间/质量上下文必须随相应行列共同置换。这里说的是存储重排；真实改变刺激发生顺序、刺激间隔或不应期上下文可以改变 EEG 和预测，不应强求生理不变性。

不添加按数字编号/字符槽位独立学习的 bias 或 embedding，不把 session 身份、已知校准目标编码输入预测器。共同标量 bias 在等覆盖头中本就不可识别，不应被重复解释为认知信息。明确的外部 prior 与 learned code bias 区分保存。

若需要 tiny set scorer，使用共享函数构造候选等变残差，例如

\[
u_c=\sum_i r_iA_{ic}\phi(e_i,h_i,q_i,t_i),\quad
v=\sum_c\psi(u_c),\quad
\delta_c=g(u_c,v)-K^{-1}\sum_k g(u_k,v),\quad
s_c=\log\pi_c+(A^\top(r\odot e))_c+\delta_c.
\]

所有 φ、ψ、g 跨候选共享，禁止 code embedding；末层零初始化使初态严格恢复基式。质量资格 g_d、r_i 不由这些参数产生。center δ 只消除共同偏置，不能阻止学习隐蔽质量/次数捷径；仍需等覆盖、候选重标、计数反事实及独立 session 验证。

Deep Sets 提供集合不变/等变结构的理论先例，不能证明这类小 EEG 样本上的统计收益，也不证明以上特定摘要足够表达全部时序相关。参数自由度、校准样本数和跨日泛化构成采用它的实测门槛。

## 8. 同一模型怎样统一训练

在合法候选数据上，使用一次联合损失与一次 optimizer step：

\[
L= L_{trial}+\lambda_{dec}\frac{\sum_d v_d\sum_R\alpha_Rg_{d,R}L_{d,R}}{\sum_d v_d}
+L_{other\ declared},\qquad
L_{d,R}=-\log\operatorname{softmax}(s_{d,R})_{c_d^*}.
\]

v_d 是预声明 decision/participant/domain 风险权重；它不等于把 trial 权重直接相加。α_R 为事先固定的前缀权重。只有 binary 标签而没有真实 membership 的公共源仍参加 trial/合法辅助目标，决策项 inactive；没有伪造候选身份，也没有切换成另一个模型。

固定 membership 与固定聚合并不阻断学习。温度为 τ 时：

\[
\frac{\partial L_d}{\partial e_i}
=\frac{r_i}{\tau}\sum_c A_{ic}(p_c-\mathbf1[c=c^*]).
\]

该梯度完整回传至 trial readout、时间/空间编码器和被声明可训练的适配模块。row/column 会自然得到对应行列边缘概率误差；无需另外发明一个协议专用神经头。

不再用“trial 标签个数远大于 decision 标签个数”证明 λ 必须落在 [0.1,0.5]；同一目标与排程确定的 trial 标签高度相关。λ 的范围应由风险归一化、梯度尺度及合法 inner 验证确定；比较 λ=0 与 λ>0 时保持批次、更新剂量和选方机会一致。

当前 group batching 对 min_decisions=2/min_rows=128 的硬要求是已有实现合同，不是数学定理。450-row decision 在 batch cap512 下无法装下两组；新设计须按真实支持规划完整 group 或有明确 BN 语义的微批累积，不能静默截掉部分事件。λ=0 对照必须共享新 batch 计划，否则 grouping/BN 与损失效应混杂。

双向滤波下 R1…R5 是完整记录结束后的回顾性证据。若以后研究真正动态停止，必须按截止时间重建可用原始输入并重新验证依赖，不得裁已消费未来样本的缓存。

### 8.1 两种采样流、一个更新：精确定义与控制

trial 风险与 decision 风险有不同统计单位，分别声明，不能按拼接后总行数混算。设合法训练 epoch 的全集为 I，外部冻结准入后的 trial 风险质量为 a_i≥0、Σ_i a_i=1；声明的完整请求 decision 全集为 D，风险质量为 b_d≥0、Σ_d b_d=1。a 可以包含明确的库/被试/类别权重；b 独立声明库/被试/decision 权重。两者属于同一授权训练阶段，inner 只取 T，final 才取 C。

每个 optimizer step 抽两个流：i_1,…,i_Bt 独立来自 p_i，d_1,…,d_Bd 独立来自 q_d；对所有正风险质量保持正采样概率。使用固定风险归一化的估计：

\[
\widehat L_{trial}=\frac1{B_t}\sum_{j=1}^{B_t}\frac{a_{i_j}}{p_{i_j}}\ell_{i_j},
\quad
\widehat L_{decision}=\frac1{B_d}\sum_{k=1}^{B_d}\frac{b_{d_k}}{q_{d_k}}
\left[g_{d_k}\ell_{d_k}+(1-g_{d_k})c_{fail,d_k}\right].
\]

当 p=a、q=b 时分别为普通采样均值；采用别的采样分布时用显式重要性系数。不能用“本批权重之和”随手替代 B_t/B_d 与已规范化风险质量：一般会得到自归一化比值估计，有限批次下不再是上述精确期望。若生产选择自归一化，必须另写其估计量与对照，不伪称完全相同。

上述点式无偏性以每个 ℓ_i/ℓ_d 不依赖其他抽中样本的 batch 上下文为条件。固定或预估后冻结 BN 能满足这一层条件；在线 BN 使单样本预测依赖联合 batch，此时训练目标应写成对**完整联合采样过程**的期望，不能宣称重要性权重已经恢复原先随机 row-only BN 的风险。此次设计不强制新增归一化层，但必须做同 batch/BN 的损失对照。

执行器先构造所需实际 EEG 的联合工作集、建立稳定索引映射，再前向并计算两个风险。decision 流需要该 decision 的完整排程及全部允许的可用 evidence；trial 流只对自己的抽样项计算 CE。**不能给所有 decision 流附带 EEG 自动再算一份 CE**：direct9 的 target 比例、row/column 的 target 比例和二分类源池不同，这会改变 trial 抽样先验并重复加权部分人/协议。

若同一物理 event 在两流或同一流多次出现，可以在一个更新内复用一次前向，但 trial 重复 draw 的 multiplicity 仍保留在上式；完整 decision 重复 draw 也按其抽样 multiplicity 计风险。复用使 dropout 结果在这些引用间共享，且可能改变 BN 的实际重复行计数；应作为明确的联合 batch 语义，在全部对照中保持。禁止仅凭相同信号数值合并不同 event，也不能把某个 event 的重复引用当成多次独立证据。

严格一次更新的最简伪代码是：

```python
row_draws, decision_draws = joint_plan.next_step()  # 与loss系数无关
workset, row_lookup, decision_lookup = materialize_same_joint_inputs(...)
optimizer.zero_grad(set_to_none=True)
logits = model(workset.X, mask=workset.channel_mask)
row_loss = declared_row_risk(logits, row_draws, row_lookup)
decision_loss = declared_decision_risk(logits, decision_draws, decision_lookup)
loss = lambda_trial * row_loss + lambda_decision * decision_loss + other_declared_losses
loss.backward()
optimizer.step()  # 只有这里一次
```

λ_decision=0 对照仍抽相同两流、构造相同工作集、按相同顺序前向；否则“加决策损失”的差值混入了 BN、dropout、行数和随机状态变化。λ_trial=0 的 CE 关闭对照同样如此。不能因某项系数为0就缩小前向 batch，也不能保留旧的“epoch 末额外一次 candidate optimizer.step”。实现若为性能跳过零系数项的标量计算，必须与字面联合目标核对梯度及 `grad is None`、weight decay、optimizer state；仅比较总 loss 数值不足够。

**B_d=1 数学上有效。** 一个完整 decision 的多类 CE 已定义，无需第二个目标标签才能训练。min_decisions/min_rows 应是经验证的工程/BN策略；对单组合法输入不能直接声称 loss 不成立。完整组超过显存或 batch cap 时，允许在明确且一致的 BN/dropout 语义下分块前向、汇集同一组 logits 并在最后一次 step 前累计梯度；不能将完整决策拆成几个独立 softmax，也不能以尾批不足为由静默删除事件。实现这一分块能力之前应明确拒绝不可执行预算，而非弱化组定义。

三种“partial”分别处置：

1. 排程/身份合同本身不完整：输入非法，不能猜补 membership。
2. 完整排程存在，部分 EEG 因外部冻结 QC 缺失：保存全部排程和 r，按预声明策略计算 g；strict 主终点下保留失败分母，不能重采直到抽到干净组。
3. 一个 trial 部分导不可用但达到既定最低观测合同：作为带通道 mask 的可用 trial，不能把它当成少一次刺激；mask/quality 特征与 EEG 同行对齐。

QC 可改变可用集合及其类别比例，这在 MNAR 情形不可能靠 dropout 或通用权重自动恢复。拟合前后应分别记录真实 label counts、抽样概率与加权有效先验 ρ_eff=Σ_i a_i y_i；source 的 CE 偏置解释来自实际执行的风险/抽样，而不是硬写1/9。若 a 已含类别权重，勿在先验修正中再重复加同一 class-weight 因子。普通神经 dropout 不删除训练标签或改变 r/g，其随机掩码不得由 y/c* 或是否难分类控制。

没有真实 membership 的 binary-only 库意味着该库没有定义 decision 风险；它继续进入 row 风险，不伪造 D。已经声明了 decision 请求但全部 g=0，则是另一种状态：请求风险及失败项仍存在，只是没有可用的决策梯度。两者的诊断不能合并成“本批无数据”。

## 9. 当前代码的具体迁移点

| 位置 | 当前事实 | 新方案应做的联动 |
|---|---|---|
| `data/candidate_task.py:membership_kind`、`transfer/decision_loss.py:DirectDecisionMetadata` | 分别只有 row_column 与 direct | 原模块增加共同选择词表/事件 membership 语义，保留两协议 builder；核验 y=A[:,truth]、物理事件身份、mask和真实时间支持 |
| `models/candidate_evidence.py:110` | NumPy 按 candidate code 分桶，mean 默认；支持其他聚合 | 作为唯一外部评分语义入口，接共同 A；保留历史 explicit mean 方法身份；torch可微核与NumPy结果/梯度参照交叉验证 |
| `transfer/decision_loss.py:681,776,907` | direct均值CE；packed前缀也除以重复数 | 共同 membership sum/明确温度，重新解释前缀概率；更新config/schema能力而非把旧mean文件重标成sum；移除无数学依据的λ硬区间前须更新调用、签名与测试 |
| `transfer/candidate_set_loss.py:49` | 行列两个均值loss相加 | 改为共同核的row_column builder/优化分解，并用联合候选NLL与梯度恒等验证；不保留第三种分数语义 |
| `baselines/deep.py:926` | fixed_updates拒绝decision_metadata/config；旧decision路径还拒绝部分源风险/辅助组合 | 实现同一步联合loss、完整group批计划、明确decision风险权重后再解除互斥；不可仅删除检查。复用现有固定更新器、source objective与性能路径 |
| `subject_adapter.py:122,1181,1675,1747` | fixed updates与candidate loss互斥；旧target每epoch额外candidate optimizer.step，且仅row/column metadata | 用同一batch的一次联合更新替代额外全训练集step；inner/final同loss；真实更新/曝光计数与缓存键同步；direct9也从同一metadata builder进入 |
| `transfer/recipes.py:78`、`calibration_candidates.py:436` | TargetRecipe不传decision loss；fit仅传X/y/channel mask | 将声明的决策配置和metadata范围沿配方→候选fit→final refit传递；无需改成另一个LDA/NN选择器 |
| `calibration_candidates.py:487`、`brainsync_model_task.py:124` | 产品分别手动聚合R5及推理曲线 | 共用决策核与完整请求账本；`scores`、概率资格、coverage、可识别性、tie/abstain理由由同一结果解释；未知目标入口不读truth |
| `transfer/evaluation.py:27,114`、`candidate_decision.py:81` | direct与row/column有多个评分路径 | 统一membership语义和请求分母；优化累计实现须与标量共同核一致，不能因缺组而消失 |
| `calibrated.py`保存/重载及checkpoint/fit-cache | 当前个人包主要保存binary模型和输入合同 | 保存冻结词表、协议/membership版本、score kind、prior/类权重修正、温度/相关修正范围、缺失政策及真实loss/update预算；旧包按旧模型解释 |

以上是设计迁移清单，未实施。对“统一模型”的必要数据资格判断应保留：缺少membership无法算决策监督，未知目标不能拟合，无效通道不能假装观测。这些是数据合同，不是按结果淘汰有用神经模块。

## 10. 已执行的数学反例

脚本：[decision_math.py](decision_math.py)。在仓库根目录执行 `.venv/Scripts/python.exe -B doc/assets/unified_model_research_20260916/decision_math.py`。脚本没有硬编码工作区绝对路径。项目 `.venv`，CPU float64，torch 2.13.0+cu130，退出码0；八组检查通过，输出JSON逐项PASS。

| 检查 | 结果 |
|---|---|
| direct9/rowcol显式条件似然乘积与Aᵀe softmax一致 | 两协议通过，atol/rtol 1e-12 |
| autograd与A(p−onehot)一致；事件重排与候选重标 | 两协议通过 |
| row/column联合NLL与两轴NLL、梯度相等 | 通过，示例NLL 0.6734845 |
| 等覆盖sum/mean | argmax相同，最大概率0.167738与0.121476不同 |
| 不等覆盖下漏先验修正 | winner翻转，见第4节 |
| 复制证据 | 最大概率0.731059→0.880797，说明重复独立性假设的重要性 |
| 非随机缺失 | 缺失信息给0.888889，neutral mask给0.5 |
| 固定失败惩罚仍可被学习型拒绝利用 | 2.55→1.148612；资格mask必须外部冻结 |

后续软件验收还需：全零/非有限/错行membership拒绝；不同K合法候选重标；缺一个row或column、重复物理事件、两个不可辨别列；任意candidate code不进入参数；训练与推理核相等；λ=0路径在同一batch下不额外forward/BN/step；保存重载不改变score/NaN/coverage/拒判。成人效果需新独立采集。

## 11. 已核验的原始文献与证据边界

网络检索共8次：Crossref 3次，Europe PMC 2次，arXiv 2次，全文请求1次。最后一个 Europe PMC fullTextXML 请求返回404；以下结论依赖已核验元数据/摘要，不声称逐式复核全文。本文公式为本次独立推导。

1. Throckmorton CS et al. *Bayesian Approach to Dynamically Controlling Data Collection in P300 Spellers* (2013). DOI **10.1109/TNSRE.2013.2253125**，PMID23529202，PMCID PMC3798004。Crossref与Europe PMC一致；摘要明确逐测量更新候选目标概率，并按概率阈值动态控制采集。它支持Bayesian决策/停止的研究先例，不证明当前神经logit已校准。https://doi.org/10.1109/TNSRE.2013.2253125
2. Mainsah BO et al. *Increasing BCI communication rates with dynamic stopping towards more practical use: an ALS study* (2015). DOI **10.1088/1741-2560/12/1/016013**，PMID25588137，PMCID PMC4631027。Europe PMC核验；摘要报告在线DS相对固定采集提高通信速度、维持准确率。人群、范式及因果在线输入不同，不能移植其数值作为BrainSync目标。https://doi.org/10.1088/1741-2560/12/1/016013
3. Mainsah BO, Collins LM, Throckmorton CS. *Using the detectability index to predict P300 speller performance* (2016). DOI **10.1088/1741-2560/13/6/066007**，PMID27705956。摘要将Bayesian DS多假设问题简化为似然比问题，并明确正态性条件；支持把概率假设写出。https://doi.org/10.1088/1741-2560/13/6/066007
4. Mainsah BO et al. *Optimizing the stimulus presentation paradigm design for the P300-based brain-computer interface using performance prediction* (2017). DOI **10.1088/1741-2552/aa7525**，PMID28548052。摘要明确刺激频率/不应期会影响ERP可靠性；支持不能把真实刺激顺序视为无关。https://doi.org/10.1088/1741-2552/aa7525
5. Zaheer M et al. *Deep Sets* (NeurIPS2017). **arXiv:1703.06114**，arXiv原站核验作者、标题、会议、摘要。为置换不变/等变集合结构提供先例；没有P300产品效果证据。https://arxiv.org/abs/1703.06114
6. Guo C et al. *On Calibration of Modern Neural Networks* (ICML2017). **arXiv:1706.04599**，arXiv原站核验。支持区分分类准确率与置信校准、测试温度缩放；其摘要实验为图像/文档，不等于EEG校准认证。https://arxiv.org/abs/1706.04599
7. Rubin DB. *Inference and missing data* (1976). DOI **10.1093/biomet/63.3.581**，Crossref核验；同次返回的1975 ETS原始报告摘要区分缺失机制可忽略条件，报告DOI **10.1002/j.2333-8504.1975.tb01053.x**。本文使用缺失观测联合似然直接展示问题，不把“标签盲”自动等同于MAR。https://doi.org/10.1093/biomet/63.3.581

另一条检索核实：*Dynamic Stopping Improves the Speed and Accuracy of a P300 Speller* 的确切书章作者为Verschore等，2012，DOI10.1007/978-3-642-33269-2_83；不是Mainsah2015论文的标题。未将二者混引。
