# 第二阶段完整矩阵：联合估计、字典端点与先验尺度

本轮沿用八人的授权C800开发范围，每折600标签拟合、120标签留出。13臂×8人×6折共624请求已逐项独立核查；
576份预测完成并用物理冻结源码、原模型字节和重新构造的合法C逐值重放，48份明确数值失败。
所有record/companion/source摘要通过，原五臂240份候选logits与第一阶段逐值相同。
第一阶段分析脚本已物理保留于`../analysis/executed_analysis/`并保存SHA；原分析数字没有改写。

## 完整结果与失败分母

八人等权；每臂请求48次。NLL是可评分决策的条件NLL，失败不伪造概率。

|固定臂|覆盖|正确/48|hit|NLL|held残差ACF1|
|---|---:|---:|---:|---:|---:|
|旧filtered-Y|1|29|0.6042|2.7966|0.8937|
|filtered-D|1|18|0.3750|6.3469|0.8938|
|raw-AR1|1|29|0.6042|3.0228|0.8693|
|raw-AR8|1|32|0.6667|6.1623|0.8053|
|raw-AR32|1|30|0.6250|7.3264|0.7979|
|raw-AR8，3轮|1|23|0.4792|2.9607|0.5301|
|raw-AR8，加权1轮|1|32|0.6667|5.8070|0.8223|
|raw-AR8，加权3轮|1|18|0.3750|3.4058|0.5545|
|raw-AR1，加权3轮|1|24|0.5000|2.5569|0.8685|
|raw-AR32，加权3轮|0|0|0|不可估|不可估|
|raw-AR8，标准化空间收缩+加权3轮|1|18|0.3750|3.4058|0.5545|
|raw-AR8，零端点1轮|1|28|0.5833|2.5084|0.5343|
|raw-AR8，零端点3轮|1|28|0.5833|2.5766|0.5301|

AR32加权3轮的48份失败均为`stationary startup covariance is numerically unidentifiable`。
这是当前平稳初态数值模型被拒绝，不能改写为AR32所有可行概率模型都无效；也没有提高condition阈值硬放行。
uniform-abstention约定给这些请求的报告损失为log36=3.5835，**不是该模型预测出的NLL**。

## 已声明机制对照

下列差异都先在每人六折内取均值，再在八人之间等权；bootstrap按人，不按折。

|对照（前者−后者）|Δhit|ΔNLL|八人bootstrap 95%区间（ΔNLL）|
|---|---:|---:|---|
|AR8三轮−AR8一轮|−0.1875|−3.2016|[−8.5698,+1.2298]|
|AR8加权一轮−无权重一轮|0|−0.3553|[−5.6814,+4.4153]|
|AR8加权三轮−无权重三轮|−0.1042|+0.4450|[+0.1214,+0.8270]|
|标准化空间收缩−普通空间收缩（均加权三轮）|0|+0.000012|[+0.000001,+0.000027]|
|零端点一轮−原字典一轮|−0.0833|−3.6539|[−9.0993,+0.5809]|
|零端点三轮−原字典三轮|+0.1042|−0.3842|[−0.6700,−0.1157]|

三轮把残差相关性从约0.80降到约0.53，但高阶臂又进入欠置信：AR8三轮的最大候选概率中位数仅0.0443，
加权三轮仅0.0326（均匀值0.0278）。NLL下降伴随hit下降，不能据此宣称完成正确概率估计。
零端点字典的较好点估计也不是独立确认；这是沿用同一C开展的自适应机制开发，且改变了字典和先验。
全部臂相对旧锚点及逐人结果仍完整保存在`analysis.json`，没有只呈现表现较好的组合。

## 更深的先验尺度问题

固定ridge=1并没有在不同AR形状之间保持同一ERP先验。
当前矩阵正态先验为\(H\mid\Sigma\sim MN(0,\lambda^{-1}I,\Sigma)\)，其中\(\Sigma\)是**创新**通道协方差。
若同一物理噪声写成\(K\otimes\Sigma=(cK)\otimes(\Sigma/c)\)，likelihood不变；
固定lambda却把H先验方差除以c。保持同一先验需要同时\(\lambda\mapsto\lambda/c\)。

独立反例把时间cov乘200、空间cov除200，固定ridge使最大log Bayes factor改变216.620；同步ridge除200后误差1.71e−13。
这不是“稍微调一个超参”，而是同一噪声的尺度表示不应暗中改变先验。

真实A01 fold0的平稳方差/创新方差r0从旧AR1约7.73、raw-AR8约1038增长到AR8三轮约236385。
零端点一轮有效核自由度仅0.607/16，旧臂为10.83/16，表明后续欠置信需要检查先验收缩，而非只继续调整噪声阶数。
尚不能把这一统计联系当作全部性能变化的唯一原因。

后续若采用按平稳边际噪声尺度定义的先验，在现有unit-innovation表示中应使用
\(\lambda_{\rm effective}=\lambda_{\rm declared}/r_0\)。这相对旧模型是**明确的新先验约定**；
只有纯尺度重参数化同时保留原先验时，才要求Bayes factor严格不变。

## 条件前缀：可检验的新估计量

为研究AR32的startup失效，独立验证了另一概率模型：在真实前p个观测给定的条件下只评分随后创新。
已知噪声、已知共同nuisance且prefix中所有候选响应都为零时，完整likelihood中prefix密度为候选共同项，
full/conditional Bayes factor差仅3.98e−13；prefix若含候选信息，反例差−3.865，不能随意删除。
未知共同offset先验若不随prefix条件更新，即使Dprefix=0，naive conditional与full marginal也不等价，反例差−5.64e−4。

`TemporalNoise`已增加显式`conditional_prefix`模式，保存为独立`/2`记录；实际prefix继续进入AR历史，
只返回后n−p行创新，不构造病态平稳初态矩阵。Burg反射系数和AR pole稳定性仍严格检查，没有放宽旧拒绝阈值。
GLS调用者必须另核prefix在首个响应前、无项目双向F、设计prefix为零，并让共同nuisance经过完整创新变换。
这条路径不能冒称与含未知nuisance的旧完整边际模型一般等价。

14项定向验证通过，包括独立条件Gaussian二次型、AR0/1/2默认record与第二阶段冻结源码逐值相同、
病态平稳矩阵仍拒绝而合法条件模型可重放、不稳定AR仍拒绝，以及高阶r0反射系数公式与可解Lyapunov对照。
条件模式用\(r_0=1/\prod_j(1-k_j^2)\)计算尺度，不为了求这个scalar重新构造病态矩阵。
这些是下一阶段的理论和实现前提，尚不构成新增真实效能结论。

材料：`analysis.json`、`diagnostic_cells.json`、`conditioning_probe.json`、`probe_conditioning.py`及`temporal_noise_conditional_tests.xml`。
所有held真候选残差均在封存预测后用于诊断，没有反向更新模型；源码、模型、raw范围与评分逐项绑定。

## 同一平稳模型的成熟数值解：Levinson / PARCOR创新

进一步推导后，优先研究与旧stationary模型同一估计量的稳定分解，而非首先删去prefix。
给定实际Burg反射系数\(k_1,\dots,k_p\)，第t个prefix样本用t阶预测器，创新方差为

\[
\nu_t=1/\prod_{j=t+1}^{p}(1-k_j^2),\qquad t=0,\dots,p-1.
\]

将该创新除以\(\sqrt{\nu_t}\)，后续接p阶单位创新，即得到完整平稳AR白化；保留全部观测、nuisance与原边际估计量。
不必先构造近奇异Toeplitz covariance再求Cholesky。实际执行对全序列使用PARCOR lattice递推，
避免仅prefix稳定、后续又回到数值敏感的展开多项式。

100位独立Yule-Walker/covariance参考验证AR1/2/4/8的prefix factor相对误差均≤3.52e−16。
重复pole=.99的AR4，舍入covariance的Cholesky factor已出现8.59e−4相对误差，而创新factor仍接近机器精度；
AR8的舍入covariance无法Cholesky。AR8展开成float polynomial后roots甚至报告半径1.0085，
但实际PARCOR全部严格小于1、100位计算证明平稳；这说明该roots值不能替代lattice的稳定性证书。

新`stationary_innovations`模式独立保存`/3`记录：真实fit PARCOR、prefix factor身份、各创新方差、r0及factor条件数。
拟合与重载不依赖最终coef逆stepdown；非法或非有限PARCOR直接拒绝。原`stationary /1`和`conditional /2`保留原数值路径，
原48份startup失败记录也原样保留，没有通过删除旧条件数阈值让它们变成成功。

20项定向验证通过。默认AR0/1/2的record和whiten与第二阶段物理快照逐值一致。
对同一float观测，新full lattice与100位参考的AR4误差为1.40e−11；极端AR8误差为0.00219。
AR8 factor条件数1.51e16、eps×condition=3.35，输入量化本身已损失信息；新的分解不能恢复这部分信息，
不能因factor公式准确就宣称所有病态数据可靠。这些精度指标进入模型记录，供下一阶段真实模型逐项检查。

独立推导与精度证据见`stationary_innovations_probe.json`、`probe_stationary_innovations.py`、
`lattice_execution_verification.json`和`stationary_innovations_tests.xml`。真实新模式效能仍需下一阶段固定矩阵。
