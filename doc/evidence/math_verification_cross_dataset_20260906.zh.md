# 跨数据集迁移的数学与反例记录

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-06。配套 [方案](architecture_decision_cross_dataset_20260906.zh.md)。
结果为合成/软件验证，不是新数据集迁移收益。运行脚本与盘点产物在本轮验证归档中。

## 1. 多源风险与梯度

设 \(A_{du}=\sum_i a_{y_{dui}}\)、\(A=\sum_{du}A_{du}\)，
现有 `resolve_source_risk_weights` 返回的逐行乘数为

\[
 w_{dui}=\frac{A\alpha_d}{|U_d|A_{du}}.
\]

故全数据风险

\[
 \frac1A\sum_{dui}w_{dui}a_y\,CE_i
 =\sum_d\frac{\alpha_d}{|U_d|}\sum_u\frac{\sum_i a_yCE_i}{A_{du}}.
\]

对二分类logit \(z_i\)，令 \(c_i=w_i a_{y_i}/A\)，有

\[
 \frac{\partial L}{\partial z_i}=c_i[\sigma(z_i)-y_i].
\]

反例使用两个域、三个不同trial数/类比例的被试、正类权重2.5，直接调用生产权重函数。
每被试总质量分别0.25/0.25/0.5，梯度与autograd以float64、atol1e-12逐项比较通过。
这不意味着每域正负类各半；不能把participant质量与class mass混为一谈。

### batch归约边界

当前 `_CrossEntropyObjective` 在batch B上使用

\[
 L_B=\frac{\sum_{i\in B}w_i a_{y_i}CE_i}{\sum_{i\in B}a_{y_i}}.
\]

按B均匀抽样一般有 \(E[N_B/D_B]\ne E[N_B]/E[D_B]\)。
真实生产objective的两组不同类质量batch验证：先分别取比例再等权平均不等于全数据风险；
累加numerator/denominator后再取比例则与全数据值一致。
这没有测定当前大batch训练的实际性能偏差，只否定“SGD每步精确域等权”的过强说法。
若需要无偏全风险SGD，均匀有放回抽样时可用固定训练均值分母
\(\bar a_{train}=A/N\)：\(L_B^*=(|B|\bar a_{train})^{-1}\sum_Bw_i a_yCE_i\)。
这是可选明确的新优化目标，实施时同步学习率/更新预算与验证归约，不能悄悄替换旧结果语义。

## 2. 边际对齐不足以确定标签语义

源域 \(X=2Y-1\)，目标域 \(X=1-2Y\)，\(P(Y=1)=1/2\)。
两域X的整个边际分布相同；任何仅检测边际MMD、均值或协方差的准则都无法区分。
源规则 \(\hat Y=1[X>0]\) 在目标错误率为1。代码反例验证通过。
因此零边际域差异不是P300可迁移性的充分条件；此例是机制边界，不代表真实数据发生标签翻转。

单纯先验差异也会改变边际：若 \(X=Y\)，则
\(E[X]=\pi\)、\(Var(X)=\pi(1-\pi)\)。1/6与1/9产生不同均值/居中方差，
但类条件分布完全一致。不能把所有边际差异都当应被消掉的设备噪声。
EA实现若用未居中二阶矩，需要按其确切定义分析，不能混用居中协方差证明。

只有 \(p_T(x|y)=p_S(x|y)\) 时，先验校正满足

\[
 \operatorname{logit}p_T(1|x)=\operatorname{logit}p_S(1|x)
 +\operatorname{logit}\pi_T-\operatorname{logit}\pi_S.
\]

若源训练使用类权重a，还需先移除加权posterior带来的\(\log(a_1/a_0)\)偏移；
复用 `checkpoint_scores_to_llr`，不新写互相冲突的校准器。
数据源混合的真实类条件不同则没有单个截距修正的保证。该公式未声称直接改善AUC。

## 3. 共同参考及缺导信息

两个记录 \(X^S_c=s_c-r_S\)、\(X^T_c=s_c-r_T\)，共同实测anchor a下

\[
 X^S_c-X^S_a=s_c-s_a=X^T_c-X^T_a.
\]

随机时变原参考的数值反例验证抵消恒等式。该假设要求真实共同参考和同一物理anchor；
未知上游通道特定处理不能凭数学恒等式豁免。CAR投影
\(P=I-11^T/C\)满足\(P1=0\)，所以纯共模判别信号可全部消失，代码验证通过。
共同参考有确定的不变性，也有确定的信息损失，实测收益需比较。

蒸馏反例：student唯一观测 \(X_o=0\)，teacher观测 \(X_h=Y\)，正负各半。
任何确定student对两样本输出相同，最优teacher-probability平方拟合误差为0.25，
不能恢复teacher的完美判别。蒸馏可改变有限样本学习，但不创造未观测导的信息。

## 4. 时间隔离

设最大K对应calibration的最晚证据可用时间
\(b=\max_{i\le K_{max}}t_i^{available}\)，所有预算共用

\[
 T=\{j>K_{max}:t_j+t_{min}\ge b\}.
\]

使用实际 `binary_time_split`、187.5ms SOA、[-200,1200)ms支持域，K48/96/192
均得到完全一致test row，且首test epoch不与最大calibration窗口重叠。
此测试只针对单recording；多run时钟归一与session排序仍待实施，不以该PASS替代。
固定因果IIR延续过去状态允许部署使用，不能把窗口无重叠说成统计独立。

## 5. 本轮验证记录

- 项目 `.venv/Scripts/python.exe` 执行7组反例；风险权重/梯度、标签翻转、先验差异、
  蒸馏信息限制、参考不变性/损失、实际固定suffix、生产batch归约全部通过。
- 盘点两套NEMAR全部630份事件表及通道表；两个代表性转换BDF可被MNE读取，
  样例前2048点有限；没有做全数据QC、完整BIDS validator或causal cache重建。
- BI六库246 ZIP中央目录可读；LLP 8 ZIP中央目录损坏。没有删除、修复或重下载原始数据。
- 首次盘点输出因NumPy整数不能JSON序列化失败；显式转为Python int后重跑。
  波形样例随后限定转换BDF，避免把sourcedata中的刺激状态幅值误报成EEG幅值。
  扩展生产CE反例时还统一了objective与logits的float64 dtype，修正探针自身的dtype错误。
- 脚本本轮只用于设计验证；未修改 `src`、训练CLI或正式test，无算法性能晋升。

完整记录：[design_audit.json](../assets/cross_dataset_20260906/design_audit.json)。
验证源码的物理压缩包及checksum见该目录的归档manifest；复核时解压到项目tmp后使用.venv执行。
