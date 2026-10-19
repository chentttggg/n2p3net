# R1边缘状态可识别性：推导、真实合法前缀与受约束反例

日期：2026-10-04。研究证据，不是活动规范，也不是分类效能结论。

本轮没有把Γ/Ω参数表的存在视为科研完成。完成了三类估计策略的辨析、8人合法C800内1504个伪截断、48个留人验证fold、无约束与两种受约束条件修正、2016个固定ERP增量挑战，以及独立AR(3)/KKT反算。没有拟合或评价真实T；只修改本证据目录。

最有用的新发现是：**长短滤波差值能被明显预测，但低差值MSE并不等于保留ERP。** 无约束预测在边缘的MSE降低约81%，却把常规P300增量误差从2.28%提高到23.5%–24.3%。约束保护ERP后，离右端2秒/8秒的差值MSE仍降低39%–47%；紧贴边缘的优势消失。强制重建长参照响应则产生不可满足约束或严重噪声放大。这些正负结果共同约束下一步研究，不能只展示81%。

## 1. 随机量到底是什么

记原始观测段为x，固定有限双向滤波及epoch变换为F，较长但仍合法的母段为(x,x_out)，母段滤波后截取为z_H。有限输出为

\[
z_F=Fx,\qquad \delta=z_F-z_H.
\]

在给定x后，F以及代码中由x算出的padding/起始state都是确定的。这里δ表达相对于另一种上下文支持的输出差异；它并不是因为数值滤波器忽然增加了一份独立测量噪声。未知的实际外部前驱/后继state可以是随机量，但必须先给出它来自哪个原始过程、其观测模型和条件信息。

如果同一合法样本对上减去相容的类均值，得到r_F=r_H+d，则恒有

\[
\Sigma_F=\Sigma_H+\Sigma_d+\operatorname{Cov}(r_H,d)+\operatorname{Cov}(d,r_H).
\]

因此：

- 从有限输出r_F已拟合的残差covariance再加BΓBᵀ，会把同一边缘涨落重复计入。
- 从长参照r_H出发，只加Σ_d而忽略交叉项，同样没有一般依据；交叉项可以为正，也可以为负。
- δ的条件均值也必须单独处理。d的类内方差为零仍允许其类均值不同，Γ=0并不能证明没有类相关失真。
- 如果r_H仅是“减了二分类平均的epoch”，其中仍有重叠ERP、慢漂移和其他结构。其协方差可作为观测统计量，但不能据此宣称识别出了生理背景噪声。

独立反例令d=-0.8r_H：真实有限方差约0.04019，忽略交叉项得到1.64774，夸大41倍；在有限方差上再加Σ_d得到0.68321，夸大17倍。改为d=+0.8r_H，忽略交叉项又低估真实方差。见[counterexamples.json](counterexamples.json)。

## 2. Γ、Ω和坐标选择

设原始残差ε与真实外部state a联合高斯：

\[
\begin{bmatrix}\varepsilon\\a\end{bmatrix}\sim
\mathcal N\!\left(
\begin{bmatrix}0\\m_a\end{bmatrix},
\begin{bmatrix}\Sigma&\Omega\\\Omega^\top&\Gamma\end{bmatrix}\right).
\]

如果使用的是未条件化输出z=Aε+Ba，才能使用完整边缘covariance

\[
\operatorname{Cov}(z)=A\Sigma A^\top+B\Gamma B^\top+
A\Omega B^\top+B\Omega^\top A^\top.
\]

如果已经观察整个x=μ_y+ε，则state应条件化：

\[
E[a\mid x,y]=m_a+\Omega^\top\Sigma^{-1}(x-\mu_y),\qquad
\Gamma_{a\mid x}=\Gamma-\Omega^\top\Sigma^{-1}\Omega.
\]

不能同时声称条件于完整x，又保留无条件Γ当额外不确定性。若a就是x的某个已观测线性组合，相应Schur补精确为零；独立例子用a=x₀证实。若Σ奇异，需要在明确观测支持内求逆/求伪逆并验证联合PSD与范围条件，不能随意加大ridge制造未知state。

坐标a′=Ta、B′=BT⁻¹会使Γ′=TΓTᵀ、Ω′=ΩTᵀ，但输出联合协方差不变。对state坐标随手加λI并不具这种不变性：一个固定T=diag(1000,0.001)例子中，物理输出协方差仍相同，state各向同性ridge的输出能量却相差约735294倍。

这不意味着在任何条件下Γ都不可估计。若A/B、state起点/物理单位、原始残差定义全部固定，B有足够可观测秩，并有来自合法重复记录的配对state样本，则相应可观测Γ投影可以估计。若B有零空间或病态方向，数据只能约束BΓBᵀ和输出交叉项；对坐标Γ作强解释没有额外依据。

## 3. 三条估计策略和接口含义

### 策略A：成对输出统计，先验证分解

在合法源训练/C母段内预声明截点，得到(z_H,z_F)，在固定低维物理投影中联合估计(r_H,d)的均值和covariance。按record/participant聚类，保留两者交叉项。此路线无需把高维raw covariance显式展开，也无需给任意滤波state选择先验尺度。

它首先是诊断与低维正则化入口。因为Σ_H+Σ_d+cross与直接Σ_F相同，**完整分解本身不会凭空改善直接经验协方差**；收益必须来自经过验证的结构先验、不同支持的共享估计或更好的残差模型。把同一批有限残差拆开再合回去，不是新信息。

### 策略B：条件差值模型，并约束信号增量

利用截段内部可观测context u估计E[δ|u]，冻结后输出z_F−预测δ。预测时不需要真实外部state或长参照。均值模型及残差covariance分别拟合，训练残差需按participant交叉拟合，不能用同样本回归残差低估不确定性。

本轮固定标准化ridge，未选择超参数。其风险是条件均值可能利用ERP本身解释δ，因而降低MSE却破坏判别信号。增量字典U_ERP定义明确后，比较两种等式约束：

\[
K U_{ERP}=0,\qquad K U_{ERP}=\Delta_{ERP}.
\]

第一种保留现有有限算子对已声明ERP子空间的响应；第二种试图把该响应修正成已知长参照响应。二者不是同一目标，不可混写成“ERP保护”。

当前拟合还含源数据估计的截距，严格说是**仿射条件均值修正**。它不满足零输入零输出：本次无约束/保护臂零输入的输出feature RMS最高约0.700/0.695µV，强制一致臂达30.686µV。这是需要检验的源均值偏置，不应把它直接注册成固定线性F。约束只保护增量，不能为绝对均值偏移背书。

### 策略C：原始坐标的稳定AR过程与精确启动

在合法已知事件的重叠ERP设计中估计均值μ_y，残差采用稳定AR(p)状态空间s_t=Ms_{t−1}+Ge_t。给定稳定M和innovation covariance Q，stationary state covariance满足

\[
\Gamma=M\Gamma M^\top+GQG^\top.
\]

Ω由同一模型的lag covariance导出；不能另外独立估计一套不相容的Ω。预测时可以在raw坐标计算likelihood，也可以把μ和covariance同时经同一个固定F推送。后者在可逆支持内只改变坐标，不创造额外判别信息。

精确AR likelihood的前p个观测应作为一个**联合stationary Gaussian block**，后续才是独立innovations；不是把前p个点丢掉，也不是把它们当p个独立边缘分布。该启动块已经包括未观测前驱state的边缘化，不能再加BΓBᵀ。

独立稳定AR(3)例子：谱半径0.56116，前3联合块+后续innovations白化与dense Gaussian covariance的最大误差2.22e−16；200条观测的quadratic最大差4.26e−14，logdet差5.33e−15。这证明计算方案正确，不证明真实EEG满足AR(3)，更不能用这次固定系数例子代替实际合法残差识别。[独立验证](independent_verification.json)。

对Module设计，建议把stationary state/Ω推导、启动白化、缺样本边缘化收在一个原始残差似然Module内部。上层只提交原始支持、固定均值设计与授权fit ledger，消费frozen likelihood；避免让每个调用者自行选择Γ坐标、重复添加边缘covariance。

## 4. 真实C800实验与浅层漏洞

设计见[plan.json](plan.json)。依赖从已验证的`frozen/system_implementation_20261004_after`物理归档展开并实际导入；归档SHA为`043893ac95e4ff8e9b5f3db718e216ed6c33c4178af5a01e826f2237c4b3973a`。当前脏树的其他改动没有进入本探针。

读取八份官方BNCI2014-008 MAT原字节并核SHA；reader验证完整容器后，数值研究输入立即只保留0到第801次flash onset之前的C800，后续数组/标签对象删除。八人该母段均为259.0625秒。长参照也仅滤这一合法C800，没有使用测试fullrecord。

采用native八实测导、既有0.1–30Hz/50HzQ30有限SOS与既有epoch FFT/crop/baseline，固定每导前8个DCT坐标，共64维。截段长度16/64秒，epoch尾离右端0/2/8秒；每个几何在7人fit、1人诊断，目标参与者完全排除。1504个切点是组内重复，独立参与者仅8；没有将1504当作统计样本量。另80个预声明切点因几何越界如实记录，不补抽。

初版每24个flash抽一个切点，和12类/120flash刺激周期共振，1040个切点全为非目标，导致类均值NaN。JSON拒绝该结果，原脚本/计划/数据及失败说明保留在[initial_aliasing](initial_aliasing/failure.json)。纠正为与12/120互素的17步，并添加双类硬门禁；没有根据得分选择参与者或切点。

成对covariance与直接有限covariance最大逐元素差4.55e−12µV²，验证了相同样本上的代数恒等式。边缘差值covariance迹占有限covariance迹约20%；距边缘2秒约7.4%–7.6%，8秒约0.14%–0.80%。这些是当前投影和总体的观测比例，不是“原始EEG中噪声占比”。

无约束条件预测在所有48fold都降低δ MSE：0秒平均约81.1%–81.3%，2秒66.2%–66.5%，8秒44.1%–46.2%。它改善的是合法长短支持差值的预测，不是分类AUC或命中率。

另一个有迷惑性的结果：忽略cross或重复加Σ_d的错误分解，某些平均Gaussian NLL反而略低，约0.006–0.042/维。有限样本估计、域差与整体covariance失配可以使“额外膨胀”偶然有利；**较低NLL不能证明错误的物理分解变成正确**。需要把统计正则化与物理噪声解释分开，并使用独立物理组验收。[全部48fold](results.json)。

## 5. 保护与强制反演的真实分界

初版ERP挑战保留在[erp_increment_results.json](erp_increment_results.json)：16/64秒截段紧贴右端时，P300峰0.35秒原有限输出的增量误差2.28%，无约束修正后均值23.5%/24.3%；峰0.8秒原1.44%，变成39.4%/38.9%，最大58.0%。这足以拒绝其直接晋升。

随后先写[constrained_plan.json](constrained_plan.json)，固定每导8个0–1.2秒Gaussian smooth basis、一个负前窗basis、整母段常数和线性慢分量，共88个约束。所有模板由公式产生，不从held EEG学。仍用同48fold，比较原有全部注入并加邻近flash、早ERP和窄峰，共2016个模型×注入结果；32个固定白噪声raw draw/几何只作放大量诊断。未搜索λ、字典个数或阈值。

| 截段长度 | 距右端 | ERP增量保护：δ MSE平均降低 | 改善参与者 | 最大白噪声能量倍率 |
|---:|---:|---:|---:|---:|
| 16秒 | 0秒 | −0.76% | 4/8 | 1.0048 |
| 64秒 | 0秒 | −0.41% | 5/8 | 1.0041 |
| 16秒 | 2秒 | 39.22% | 8/8 | 0.9253 |
| 64秒 | 2秒 | 40.35% | 8/8 | 0.9487 |
| 16秒 | 8秒 | 44.97% | 8/8 | 0.9922 |
| 64秒 | 8秒 | 47.09% | 8/8 | 1.0005 |

保护臂在2/8秒条件下，新增ERP增量破坏基本消除，剩余误差回到原有限算子本身的约0.3%–2.0%；这不是“恢复真实ERP”，因为原有限算子仍可能有误差。0秒时保护成功但收益消失，说明这组观测context中可用于差值预测的信息与需要保留的ERP方向高度纠缠。

强制算子一致臂的结果更差：

- 0秒时88个约束全秩，等式残差约1e−13，数值方程可解；但白噪声能量最大放大794.87倍/157.64倍（16/64秒），实际δ MSE约变成原来的603.1倍/361.1倍，48个已声明fold中的此16fold均恶化。数学可解不等于物理稳定。
- 2/8秒时约束秩只有80/88，响应差不属于观测context能够表示的像空间；相对约束残差0.21–0.62，高于预声明1e−6门槛，32fold全部拒绝。报告保留诊断数字，但不把这些失败模型计作通过。
- Cholesky/SVD等式ridge与独立完整KKT解的最大差6.66e−16，排除了“刚好实现错了”这一解释。真实约束病态/不可表示不等于所有更丰富观测、正则逆问题或raw likelihood路线不可行。

完整结果见[constrained_results.json](constrained_results.json)。不以这些探针给两秒等待建议直接晋升：实际输入支持、等待延迟、未知后续flash以及标签授权都必须按新协议重新定义，不能把额外未来信号伪装成原同预算输入。

## 6. 下一步可以具体做什么

优先推进raw坐标重叠ERP+稳定AR(p)似然：它把均值、covariance及真实可用支持放在同一数学对象中，并自然处理stationary startup。先用合法C的物理组检验残差谱/相关、似然覆盖和ERP增量，再比较更高阶AR、厚尾或尺度边缘化；不得用当前T NLL选择阶数/温度。

保留ERP保护条件差值估计作为第二条研究路线。下一步需要在完整合法record/decision上扩展保护字典到已知调度的重叠响应，并检查是否仍留有可估计的nuisance方向；目前只保护单epoch固定模板，不能据此宣称完整连续decision的其他闪烁响应也受保护。若需输出分类score，传播修正后的均值和残差covariance，不能只替换波形。

直接输出联合covariance策略用作每条路线的独立账本与反例检查：同一事件/支持的r_H、δ、r_F和均值必须连得上，raw/state交叉项与conditioned state不得重复。若不能从合法组识别，则回到明确的经验收缩锚点并报告不可识别方向。

当前正证据只支持“存在可保留ERP增量的部分边界失真预测机会”，负证据排除了这两个特定的无约束/强制反演实现直接晋升。还没有证明A1/Q1唯一病因，也没有完成跨库、分类收益或成人设备验证。

## 一手来源与可复现文件

- [SciPy filtfilt官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)：padding/初始条件/Gustafsson的实现语义；不是无限上下文oracle等价保证。已保存网页字节。未把页面参考表中的卷号直接当原论文全文核实。
- [SciPy solve_discrete_lyapunov官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_lyapunov.html)：离散Lyapunov方程AXAᴴ−X+Q=0及其计算。
- Rasmussen与Williams，*Gaussian Processes for Machine Learning*，MIT Press，2006，[第二章原文](https://gaussianprocess.org/gpml/chapters/RW2.pdf)，式2.19、2.22–2.24：条件Gaussian均值和covariance的Schur形式。已下载并提取核验原文；这里只使用标准条件化数学，不声称本实验就是GP模型。
- [sources.json](sources.json)记录URL、HTTP状态、字节数和SHA；[gpml_equations_verified.txt](gpml_equations_verified.txt)保留相关原文。
- `probe.py`、`probe_erp_increment.py`、`probe_constrained.py`、`verify_independent.py`是实际科研实现；`execution.json`、各结果内源码SHA、`raw_support_ledger.json`、`projected_draws.npz`绑定实际输入/输出。四入口最终均执行成功；只做了本任务对应Ruff和独立数学/行为验算，没有以仓库全套绿灯替代上述科研证据。
- `research_execution_pass1.tar.gz`及manifest保存静态检查整理前的执行材料；initial_aliasing原失败另存。最终代码/结果清单见`final_manifest.json`。

本报告的Γ/Ω辨析、条件约束构造与实际数字是本轮推导/实验；引用文献用于核实基础公式和数值算法，不虚构现成方法已经在这八人或本产品证明收益。
