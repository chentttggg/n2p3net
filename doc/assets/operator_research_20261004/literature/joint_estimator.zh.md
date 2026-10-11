# 联合估计专项：AR、通道协方差与均值不能只接一次

本记录只回答2026-10-04新发现的估计器问题。输入问题由主线提供：48个合法校准fold中，原filtered AR1 residual ACF1均值约0.893，raw BurgAR8约0.804。这里没有重新计算这两个真实数值，也不把它们直接归因为单一bug；独立工作包括当前实现阅读、一手依据和合成反例。

**结论：应修正按原始通道幅值直接pool的时间估计，并迭代重拟均值/nuisance；但这最多关闭估计器缺口，不证明所有通道共享时间谱的separable模型正确。** 加权Burg是成熟可解释的近似更新，不是exact stationary likelihood的ML更新；与Ledoit–Wolf交替后不能无依据声称单调或全局最优。

## 一手来源核实

| 来源 | 本次实际读取的定位 | 能支持与边界 |
|---|---|---|
| Drton, Kuriki & Hoff，*Existence and Uniqueness of the Kronecker Covariance MLE*，[arXiv:2003.06024v2](https://arxiv.org/abs/2003.06024v2) | PDF p.3，Eq.1.2、1.3与紧随其后的flip-flop更新；pp.2–4讨论存在/唯一性/收敛 | 给出matrix-normal likelihood、两个因子的互逆scale自由度及协方差交替公式。其**无结构全时间矩阵**的样本量阈值不可直接拿来判定低参数AR模型准入 |
| Allen & Tibshirani，*Transposable regularized covariance models with an application to missing data imputation*，[10.1214/09-AOAS314](https://doi.org/10.1214/09-AOAS314)，[作者预印本0906.3465](https://arxiv.org/abs/0906.3465) | §3.2 Eq.3；§3.3 Eq.5、Proposition2，PDF pp.6–8 | 精确优化同一penalized likelihood的block updates可单调上升并到stationary point；一般不保证全局最大。不是把任意Burg、ridge、LW三个估计器轮流调用即获得该定理 |
| Dutilleul 1999，*The MLE algorithm for the matrix normal distribution*，10.1080/00949659908811970 | 本次Crossref核标题/DOI，且上两篇原文引用其算法；**未取得并实读原论文全文** | 仅作历史出处，不以未读全文支持额外细节 |
| Werner, Jansson & Stoica，*On Estimation of Covariance Matrices With Kronecker Product Structure*，10.1109/TSP.2007.907834 | Crossref核标题/DOI；**未实读全文** | 仅元数据候选，不提供本报告公式的证据 |
| [statsmodels v0.14.6作者实现](https://github.com/statsmodels/statsmodels/blob/v0.14.6/statsmodels/regression/linear_model.py) | `GLSAR.iterative_fit` L1301–1364：反复fit，再从`results.resid`更新AR；`whiten` L1366–1389：条件AR，丢最初p行；`burg` L1506后明确单序列Burg | 成熟可行GLS先例，AR在**原始坐标残差**重估。该例用Yule–Walker，不是多通道Burg；丢p行协议与本项目stationary startup不同 |

原始来源字节、访问状态、URL与SHA256保存在`access_joint_estimator*.json`。arXiv元数据API及PDF本次可读；Europe PMC全文500、Euclid200但1165字节的challenge/非PDF、EPMC PDF403均保留并降级，未当作论文已读。

## 1. 最小统计对象与scale约束

独立record i：

`Y_i = D_i H + N_i A_i + E_i`，`vec(E_i) ~ N(0, Σ ⊗ R_i(φ))`。

- `D_i`为获准完整事件设计，`N_i`为该record的constant/linear等nuisance。
- H共享，A_i按record拟合；不要把record边界拼接成一条AR序列。
- `R_i(φ)`为**innovation variance=1**的stationary AR时间covariance；所有物理通道量纲留在Σ。
- 一般`Σ⊗R = (Σ/a)⊗(aR)`，两个自由scale不可同时拟。当前unit-innovation约定可保留，Burg报告的pooled variance不能再除一次，否则double normalization。

记T为全部实际time rows数，c为实测通道数。无prior/penalty的负两倍log-likelihood（忽略常数）为

`Q = c Σ_i log|R_i(φ)| + T log|Σ| + Σ_i tr[Σ⁻¹ E_iᵀ R_i(φ)⁻¹ E_i]`。

这直接来自matrix-normal density，而不是新算法。

## 2. 正确的通道加权Burg

固定SPD `Σ=LLᵀ`，定义原始残差的通道白化`Etilde_i=E_i L⁻ᵀ`。对同一共享scalar AR的forward/backward errors f、b，使用

`κ_m = −2 Σ_{i,t} f_itᵀΣ⁻¹b_it / Σ_{i,t}(f_itᵀΣ⁻¹f_it + b_itᵀΣ⁻¹b_it)`。

它等价于在Etilde上运行现有pooled Burg。所有order使用同一固定Σ；每个order内部不能改变通道度量而继续套旧递推。SPD内积的Cauchy–Schwarz给`|κ_m|≤1`；退化等号/数值非有限/过差条件数应拒绝并入账，不把坏pole夹回后称已识别。非退化`|κ|<1`的Burg递推保留稳定性。

**通道z-score仅修正方差，完整L⁻ᵀ还处理通道相关。** 在正确separable总体模型中，各通道原本共享同一真实φ，未加权不必产生总体偏差，但会让高能量/重复相关通道支配有限样本目标；在现实非separable情况下，差异更显著。

Burg最小化逐阶forward/backward预测误差准则，不等于完整stationary Gaussian likelihood。后者还包括每record最初p个样本的stationary covariance quadratic与logdet。因此可以把加权Burg作为初始化或FGLS更新，不能称它已最小化上面的Q。

## 3. 最小可执行的两条路线

### 路线A：明确命名为iterated feasible GLS，改动较小

1. 以合法数据拟初始H、A_i，得到原始坐标`E_i=Y_i−D_iH−N_iA_i`。
2. 用训练残差得到SPD通道shape Σ，按L⁻ᵀ白化E_i，分record拟shared Burg φ。
3. 用新φ的stationary whitening同样作用于Y_i、D_i、N_i；重拟H，**重新拟A_i**。
4. 用该轮innovation估Σ；回到**原始坐标**E_i重估φ。不能把已temporal-white的残差直接再作为原始AR序列。
5. 记录每轮φ/κ、pole radius、startup condition、Σ spectrum/condition、H变化、残差ACF；达预声明收敛条件停止，末轮必须在实际保存的φ下再解H/A_i/Σ。

若保留固定ridge、相关收缩和自由度修正，这是一组有定义的FGLS估计规则；其迭代不自动继承ML收敛定理。若出现cycle/非收敛必须明示，不能只因达到maxiter就写converged。

**不要误修均值：** 所有通道共用同一D，每导有自由H列时，GLS对H的normal equations右侧Σ⁻¹会抵消。因此H解本身不需要额外通道加权；Σ须进入的是**共享时间参数φ**。若使用matrix-normal prior且其列covariance也是Σ，ridge均值解仍有该性质。只有跨导参数有额外约束/不匹配prior时才另当别论。

### 路线B：同一明确likelihood下的stationary block optimization

固定φ后，把`Z`写成共享D加block-record nuisance的整体设计，做GLS得到全部β=[H;A_i]；固定β/φ后

`Σhat=(Σ_i E_iᵀ R_i(φ)⁻¹ E_i)/T`。

固定β/Σ，直接最小化Q中的φ部分。AR(p)可以用`κ_j=tanh η_j`加稳定递推参数化，并用真实stationary startup covariance/logdet。此时不靠事后pole clipping。每步验证同一目标不增，并保留收敛状态；这种坐标法仍不保证AR约束下的全局最优。

如果采用conditional likelihood，则仅使用每record t≥p的innovation更新，φ的固定Σ更新是加权least squares：

`G_lm=Σ e_{t−l}ᵀΣ⁻¹e_{t−m}`，`b_l=Σ e_{t−l}ᵀΣ⁻¹e_t`，`φ=G⁻¹b`。

但该unconstrained least-squares解不保证stationary，且首p行是条件观察而不是stationary density，不能无记录地替换当前保留全行的模型。

无结构flip-flop的`Rhat∝Σ EΣ⁻¹Eᵀ`不能用于本项目万行record直接估一个任意T×T矩阵。保持低参数AR/预声明结构才有可计算、可估计的问题。

## 4. Ridge与nuisance积分不能混同普通ML

当前研究核用ridge，通道covariance用residual自由度修正加LW；这不是上节无penalty Q的MLE。若希望严格保留Bayesian核而估φ，可利用前份报告的matrix-normal共轭积分，但需要正确objective。

令W白化R，Q_N为WN的正交补基，`Ybar=Q_NᵀWY`、`Dbar=Q_NᵀWD`、`V=(λI+DbarᵀDbar)⁻¹`，`r=rank(N)`。原始坐标nuisance使用flat prior，H使用`MN(0,λ⁻¹I,Σ)`时，积分后的φ相关项包括

`c log|R| + c log|NᵀR⁻¹N| + c log|I+λ⁻¹DbarᵀDbar|`

以及`(T−r)log|Σ| + tr[Σ⁻¹S]`，其中

`S=YbarᵀYbar−YbarᵀDbar V DbarᵀYbar`。

这是可执行的低维行列式/Woodbury形式。固定φ时，部分项对候选q相同，可以在argmax中省略；**估φ时不能省略**。若仅比较投影残差平方，可能偏好用相关/nuisance把信息消掉的φ。参数积分、MAP、penalized likelihood和无偏covariance估计的分母也不同，需先选一个目标，不能拼接后统一叫ML。

## 5. 同单位不等于通道gain不变

所有通道一起V→µV时，pooled Burg分子分母共同乘常数平方，原实现本来就不变。这个测试不能覆盖某导单独gain变化。若`Ynew=YG`且G为可逆对角gain，则正确separable模型要求`Hnew=HG`、`Σnew=GᵀΣG`、φ不变；Gaussian log-likelihood只差公共Jacobian。

原始covariance直接做Ledoit–Wolf向`trace(Σ)I/c`收缩，通常不满足该对角congruence。若要检验这条鲁棒性，应用**合法训练残差**估`D=diag(std_j)`，对`D⁻¹ΣD⁻¹`收缩后恢复`D correlation_shrink D`；相同rule对独立非零通道gain可等变。它仍非任意空间混合/reference变化下的完全affine equivariance，不能扩大结论。near-zero channel、坏导、rank与量纲仍按实测合同处理。

此外，通道gain等变不是允许任意修正硬件gain或改输入身份；这里只是算法反例，实际preprocessing仍应存真实变换与source identity。

## 6. 已执行的独立反例

`joint_estimator_counterexamples.py`仅合成，固定seed1042026，项目venv执行；JSON保存实际脚本SHA。

| 检查 | 数值 | 结论 |
|---|---:|---|
| 八导φ=[0.05,0.92×7]，首导gain×100 | unweighted Burg φ：0.810294→0.058120 | 能量支配可掩盖其余七导噪声 |
| 上例其余七导innovation ACF1 | 约0.907–0.913 | high-order/稳定Burg本身不保证成功白化 |
| 完整通道covariance加权Burg，gain前后 | φ均0.811277823776 | 通道gain不变恢复 |
| 同一weighted结果的各导ACF1 | 第一导−0.459，其他约0.157–0.177 | 共同φ无法白化不同真实时间谱；估计器修复≠separable正确 |
| 全局V→µV | unweighted φ变化0 | 原有global scale测试不足 |
| raw Ledoit–Wolf经过逆gain恢复 | covariance相对差0.147323 | 必须联动通道covariance估计规则 |
| 相关矩阵LW再恢复量纲 | 相对差2.57e−16 | 对角gain等变，不是任意空间混合等变 |
| 真实sharedφ=.86的三段stationary AR1，联合GLS/Σ/φ ML | 5轮φ=.858794；Q：2302.96→1390.158 | 同一likelihood的交替估计可验证单调；未加ridge/LW |
| 对上述三导分别gain30/.2/4 | φ差6.3e−13，H/Σ相对等变差约2.5e−13/7.8e−14 | 包含startup、均值和covariance的完整反例 |
| `Σ⊗R`互逆scale | 误差2.22e−16 | 必须保留唯一scale约定 |
| ridge+flat nuisance积分：投影公式vs独立dense covariance公式 | φ=0/.5/.9的scatter与logdet误差均<1e−10 | 核验§4中的额外行列式与scatter，不能只优化投影残差 |

这些反例尚不能解释全部真实0.804 residual ACF。下一步应看每导ACF/PSD、每foldAR系数与pole、各导Burg分子/分母贡献、原始与GLS残差差异，以及是否真的有共同时间谱。若correctly weighted/iterated模型仍留强结构，再开放通道组不同时间shape、低秩动态factor或vector AR；不能在优化器问题未排除前直接宣判AR路线无效，也不能以增加AR阶数替代模型诊断。

平均核后验不确定性与每decision生理random effects仍保留为竞争解释；本专项先解决可单独检查的联合估计，不用它们转移当前估计器问题。
