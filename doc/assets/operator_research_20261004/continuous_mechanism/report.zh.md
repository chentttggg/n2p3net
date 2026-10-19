# 连续候选的算子一致性与概率失配：机制证据

日期：2026-10-04。属于科研子任务证据；数学反例与软件实现已完成，真实校准内48组比较由统一执行器另行记录。
本文件没有读取或重新评分BNCI的旧T，也没有用旧T选择basis、AR阶数、ridge或温度。

## 当前发现与应当改变的研究路线

当前连续模型需要区分三个问题：数据的滤波与生成模板不一致、时间噪声形状不够、噪声参数自身不确定。
这三者都可以损坏概率，但不必同方向影响命中率；仅看hit或给所有logit统一降温无法区分。

本轮选择五个固定对照：原filtered-Y/因果D/AR1、filtered-Y/filtered-D/AR1，以及raw-Y/raw-D/Burg AR1、AR8、AR32。
前两臂专门检验均值算子失配，后三臂检验在原始坐标中识别时间噪声的可行性。它们不构成根据T选择的路由器。
重尾尺度积分保留为推导反例，尚未加入真实数据比较；真实AR不足时不能用Student-t包装成已校准概率。

## 1. 算子、模板和协方差必须一起解释

写原始生成模型为

\[
y=D_j h+Z\beta+\epsilon,\qquad \epsilon\sim N(0,\Sigma),
\]

其中j是候选，D含共同ERP与target差异ERP，Z为共同nuisance。若固定线性算子F可逆，

\[
Fy\mid j\sim N(FD_jh+FZ\beta,F\Sigma F^\top).
\]

全部候选共同的Jacobian抵消，所以log Bayes factor与raw坐标完全相同；这也包括正确变换核后验项
\(D_jV_hD_j^\top\)。**没有必要仅为计算而把可逆预处理后的高度相关数据冒充近白噪声。**
但真实高通可以有零空间，裁剪也会损失信息；因此该定理不证明真实raw路线必定更好，也不证明任意F可逆。
若滤波状态或F来自当前数据自适应估计，上述固定F的推导还没有积分算子估计误差。

反例使用36维已知高斯、3候选、400次观测、seed472901：

|计算|与raw赢家不同|NLL|命中率|
|---|---:|---:|---:|
|原坐标正确Gaussian|0|0.350626|0.8550|
|F同时变Y、均值和cov|0|0.350626|0.8550|
|仅变Y|97/400|0.750946|0.7250|
|变Y与均值，但沿用错cov|115/400|0.929528|0.6825|

完整变换的最大log Bayes factor数值差为1.35e−13。该反例排除了“只要补滤波模板就把整个概率模型修好”的解释。

## 2. 实际有限F不保持现有D字典的张成空间

用当前实际0.1–30Hz、50Hz/Q30、reflect_limited双向实现；256Hz、1.2s/8基、120flash、8500sample。
F的padlen实际2253sample。将FD投影回原D并使用相同constant/linear消除：

- FD整体不能由D重新估系数吸收的相对Frobenius残差为39.2615%。
- 单独构造平滑N200/P300形状，其filtered均值在原字典中的最佳拟合相对残差仍为13.8193%。
- 先裁剪再滤波与长scope滤波后再裁剪不交换，同一平滑构造相对差为16.2977%。

因此当前“filtered EEG + 有限因果D”不是精确生成模型。学习有效ERP核可以吸收部分变化，却不能保证完全吸收。
这些数值属于固定合成排程，不是BNCI中失配大小或准确率损失的估计；没有把39%称为真实噪声占比。

实际旧route中C独立滤波，但T先整段滤波再切decision；两者的F与边缘支持不同，算子identity名称相同不意味着相同有限矩阵。
新的C内对照明确每个完整selection独立切raw、独立滤波，FD使用同一个scope。nuisance的FZ也必须明确：
若继续投影原Z，应解释为输出坐标中的额外残余漂移假设，不能称为原raw模型的完整坐标变换。

## 3. AR1错误不一定降hit，却能把概率推到严重过置信

用固定稳定AR2过程产生6000次观测，参数为a1=1.862460、a2=−0.9216，创新方差1。
AR1用总体最优phi=0.969224，另保留当前clip到0.95；没有估计误差，没有使用评价数据拟合。
在不同预声明信号频率都固定真实Mahalanobis距离平方=4：

|判别方向|正确模型NLL|总体AR1 NLL|总体AR1最优logit校准斜率|
|---|---:|---:|---:|
|2Hz|0.3599|0.5631|5.8403（欠置信）|
|6Hz|0.3501|0.4681|0.4563|
|10Hz|0.3526|3.8672|0.0430（严重过置信）|
|16Hz|0.3559|0.5178|0.3783|
|24Hz|0.3579|0.3730|1.4577（欠置信）|

10Hz时AR1命中率仍84.62%，正确模型84.83%，但平均最大概率从84.42%变为99.32%；clipping phi对该失配几乎无帮助。
这表明实际过置信可以来自判别方向上的频谱比，而非简单的“样本太多”或单一全局温度错误。

## 4. 已实现的TemporalNoise及可识别性假设

`src/transfer/temporal_noise.py`提供一个深Module：`fit`、冻结`whiten`、纯数值`record/from_record`及`save/load`。
每个residual record保持独立，不跨record拼接lag。Burg汇总forward/backward残差，但不通过乘channels伪增最小时间样本数。
最少lag rows为max(32,20p)，每段至少p+2sample；它是显式数值准入，不是足够统计独立性的证明。

预测系数约定为 \(x_t=\sum_{l=1}^p a_lx_{t-l}+e_t\)。Burg通过反射系数递推；无pole clipping。
不稳定、退化能量、过短输入、不可辨的startup covariance直接拒绝并保留失败分母。
阶数0/1/8/32均可用；p1为Burg估计，不能冒称旧OLS+clip实现。

设状态transition为A，单位创新的stationary covariance P满足
\(P=APA^\top+ee^\top\)。前p行用stationary block的Cholesky逆，后续行用实际history做AR创新：

\[
W x=\begin{bmatrix}L_P^{-1}x_{0:p}\\
x_p-\sum_l a_lx_{p-l}\\ \vdots\end{bmatrix}.
\]

这等价于对未知平稳初态积分，而不是把初态设零。Module保存明确startup假设；物理记录从非平稳状态开始时该假设仍需验证。
W去除单位innovation的时间形状，**不再除估计innovation方差**；真实电压尺度由下游channel covariance估计。
保存的pooled innovation variance只作diagnostic，防止时间和空间两次除尺度。
Whiten每次调用是一份完整独立record，不能随意逐chunk重置后声称保持同一density。

9项针对性验证已通过，包括：AR0/1/2与独立20000项impulse-response构造的75×75协方差核对、Gaussian二次型一致、
多record AR2恢复、极端record断点不造lag、p32及uV→V不变、完整冻结和数值重载、非法scope/样本数/不稳定拒绝。
原TDD import缺失先失败，接口落地后转绿；这里的9项不是完整项目验收。

## 5. 不确定噪声尺度是另一条竞争解释

预声明24维、6自由度normal/inverse-gamma生成例，12000次draw：固定尺度Gaussian的NLL为0.453390，
正确积分共同scale后为0.388801；两者命中率都82.13%。每record只积分一个scale，不把每sample各自独立积分。
这个例子支持“估参不确定性可以独立影响概率”，不证明BNCI实际服从该先验。自由度必须从合法独立组或源数据确定；
有数万个连续sample不等于有数万个独立scale观测。主真实比较暂不增加这个自由度。

## 6. 固定真实C内方案与可证伪判据

方案先存为`calibration_group_plan.json`，再由root统一runner执行；8人、C800中6完整selection、6折leave-one-selection-out，
每次5组/600标注拟合，1组/120事件评分；最后80事件不用作完整组验证。C200仅1完整selection，明确组验证不可估。
全部五臂都报告，不从同一轮组预测选优后当确认。失败、coverage、requested分母、NLL、whitened ACF与held噪声能量一同保留。

- 若FD只改善均值残差而不能改善概率，单一模板失配解释不足。
- 若AR8/32大幅改善held innovation频谱但概率仍差，需要核非线性、非平稳、空间可分离性或参数不确定性的竞争解释。
- 若高阶模型训练残差变白而held残差不白，高阶拟合不能晋升；检查共同AR跨channel和跨selection的稳定性。
- 若平均NLL改善只由少数异常组贡献，报告逐人/逐组与尾部分布，不把它改写成普遍收益。
- 同一组内多次flash不是独立验证单位；未来全source→target与成人确认仍是不同任务。

## 文件与一手材料

- `probe_mechanisms.py` / `mechanism_results.json`：独立种子、baseline源码SHA、全部反例数值；只需项目venv。
- `temporal_noise_tests.xml`：当前9项验证；root最终全套以后另记。
- `sources.json`记录实际HTTP200及SHA；[Statsmodels Burg官方接口](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.burg.html)明确AR(p)系数与残差方差，未安装额外statsmodels依赖。
- [SciPy离散Lyapunov官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_lyapunov.html)明确\(AXA^H-X+Q=0\)及求解方式。
- 已核本地Unfold原文`doc/assets/system_research_20261004/literature/unfold_full.txt`：线性事件叠加与deconvolution假设。该文不为本项目AR阶数或收益背书。

baseline为`frozen/system_implementation_20261004_after`，archive SHA256 `043893ac95e4ff8e9b5f3db718e216ed6c33c4178af5a01e826f2237c4b3973a`。
probe所用continuous_decision与preprocess字节摘要均匹配其冻结清单。新AR模块是本轮研究实现；没有替换生产默认。

## 7. 新有限算子与GLS的独立验收

`tests/data/test_finite_filter_operator.py`新增10项：样本数1/2/19/257/2700与pad0/7/2253均验证
\(Qx+B\alpha=F(x)\)，与独立SciPy `sosfiltfilt(padtype='odd')`匹配；对Q验证独立构造\(H^\top H\)、
对称PSD和双线性伴随关系；检验同F作用于设计矩阵、有限矩阵内存预算。额外反例确认完整reflect算子F本身不必对称，
且不能和crop交换，避免将Q的性质错误归到F。

`verify_gls.py`使用四段长度240/259/275/208的合成record，最后一段显式右删失。
新GLS退回`operator=None, legacy_clipped_ar1`时，与旧ContinuousDecisionModel的mean、V_h、channel covariance、phi、
log evidence与概率全部逐值相同。这个桥接保留了原算法，五臂差异可归到声明的改变，不能归到悄悄不同的旧锚点。

独立以20000项impulse response构造AR0/1/2的稠密协方差，再对nuisance取正交补、组成两通道完整Gaussian密度；
与Woodbury路径最大log Bayes factor差为4.12e−13。filtered坐标中分别使用D和FD的独立Gaussian参考误差最大2.66e−15；
此处AR covariance定义在filtered坐标，并没有声称它就是原raw covariance的精确F传播。

五variant保存重载的全部candidate evidence逐位一致；candidate vocabulary/member联合置换保持结果；重命名训练signal仍拒绝。
`score`没有truth参数，scorer-only真值映射未进入decoder；完整真实runner还需检查调用链是否遵守这个接口。
全局uV→V转换后log evidence最大差1.14e−13，添加post-processing共同常数/线性漂移后差5.68e−14。

验收中发现training ACF最初在vstack residual上计算，制造了跨record的lag；owner已改为分record中心化并汇总合法配对，
现保存`cross_record_lag_pairs=0`。此项影响诊断读数，不影响此前已分record的Burg fit或score；修后重新执行上述全部GLS验证通过。
来源及具体版本见`gls_verification.json`，定向测试见`finite_filter_tests.xml`。这些验收仍不能替代真实C内机制区分。
