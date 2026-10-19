# BNCI分发信号与conditional AR估计量专项

日期2026-10-04。本记录保留前两份文献报告，追加两个独立问题。Stage2 rawAR32 weighted3的48cell因stationary startup covariance condition>1e12拒绝，是主线提供的既有运行结果；本分支没有改写这些记录，也未更改任何门禁。

## 一、MAT X的“raw”不能理解为未滤ADC

### 核查结果

**官方数据说明已经声明0.1–30 Hz带通，因此分发MAT不能无依据称为无算子ADC数据；同时现有来源不足以重建这段上游滤波的实施阶段、阶数、相位和初态。** 推荐把本项目raw臂表述为：

> 分发MAT连续信号，保留源采集/分发处理，仅不追加本项目的离线滤波。

| 证据层 | 本次实际核实 | 能断言什么/不能断言什么 |
|---|---|---|
| [BNCI官方数据页面](https://bnci-horizon-2020.eu/database/data-sets) | 第8项“P300 speller with ALS patients (008-2014)”直接链接A01–A08与description.pdf；本地抓取L276–289 | 下载说明与八个MAT属于同一官方条目 |
| [官方description.pdf](https://bnci-horizon-2020.eu/database/data-sets/008-2014/description.pdf) | pp.1–2，Experimental Protocol：g.MOBILAB、g.Ladybird、右耳参考、左乳突地、256 Hz，“digitized at 256 Hz and band-pass filtered between 0.1 and 30 Hz”；p.2定义X=[samples×channels] | 官方记录声明上游带通。文件未提供FIR/IIR阶数、模拟/数字、单向/双向、离线/设备内处理或MAT写出脚本；不能据文字恢复精确F，也不能把这些未说明项标为已知 |
| Riccio et al.2013，[10.3389/fnhum.2013.00732](https://doi.org/10.3389/fnhum.2013.00732)，PMC3825256全文 | Materials and methods→BCI task：设备、256 Hz、参考/地；Offline data analysis→P300 morphology：4阶Butterworth 0.1–10 Hz、50 Hz notch、1000 ms epochs、±70µV剔除、−200 ms baseline；Offline BCI control performance沿用该处理并12点均值降采样 | **0.1–10 Hz等是论文离线分析pipeline，不能搬到连续MAT X字段作为已执行历史。** 原论文采集段没有把0.1–30的具体实现说明清楚 |
| 实际安装MOABB1.6.1 reader | `bnci_2014.py` `_load_data_008_2014`只loadmat→`_convert_run_p300_sl`；`base.py` L491–513只X单位转换、转置、附stim/montage/line_freq | reader没有滤波、epoch剔除、baseline；`line_freq=50`是metadata赋值，不是notch调用 |
| reader数值核对 | A01 source SHA256 `cdf9bca4f48c61ee9c9ba998382a4e84392918c9dbc5a43f45ee0052614e25bd`，347704×8 float64；MOABB EEG输出与`X.T*1e−6`逐位一致，最大差0 | 证实reader透明，不证明X写出前没有处理；没有对测试数据拟合或打分 |
| 本仓库reader与artifact | `bnci2014_008_candidate.py`读取X为eeg_uv，合同给源参考/单位；既有artifact manifest核本地/远端SHA及八文件inventory，没有记录分发滤波实施链 | 原字节相同是文件身份依据，不是模拟前端或数字滤波的完整provenance |

安装版MOABB的metadata还写了“0.1–10 Hz、50 Hz notch、artifact rejection、baseline correction”，且docstring有与官方不符的linked mastoids、年龄/性别信息。这些**描述与实际reader执行分开**：新metadata不能改写文件字节来源，也不能覆盖更接近源数据的官方说明。当前metadata很可能把原论文离线分析步骤当成分发preprocessing；本次不能据此进一步断言MAT已做上述全部操作。

本次没有找到MAT生成脚本或BCI2000原始参数文件，因而没有把官方0.1–30的文字升级成逐样本数字实现证明。采集系统本身也有观测/抗混叠响应，不能把“不加项目F”解释为没有任何上游响应。母信号时谱已有低通可能有助于解释AR32的高条件数，但需要数据诊断，不能仅凭频带文字证明这就是唯一病因。

材料：`bnci_official_description.pdf/.txt`、`bnci_riccio_full.xml/.txt`、`bnci_reader_*.py`、`bnci_reader_snapshot.json`、`bnci_reader_numeric_audit.json`；请求与hash见`access_bnci_conditional*.json`。

## 二、conditional AR的正确观测与似然

令同一合法record的观测为`Y_t=μ_q,t(β)+e_t`，噪声满足固定AR(p)：

`e_t=Σ_{j=1}^p φ_j e_{t−j}+u_t`，`u_t~N(0,Σ)`且各t独立。

选择明确conditioning prefix长度m≥p。条件观测是`Y_0,...,Y_{m−1}`；评分/拟合的innovation rows为t=m,...,n−1：

`z_t=Y_t−Σ_j φ_jY_{t−j}`，

`μ*_q,t=μ_q,t−Σ_j φ_j μ_q,t−j`。

因此

`L_c(β,φ,Σ;q)=p(Y_m:n−1 | Y_0:m−1,β,φ,Σ,q)`，

`−2log L_c=(n−m)log|Σ|+Σ_{t=m}^{n−1}(z_t−μ*_q,t)ᵀΣ⁻¹(z_t−μ*_q,t)+const`。

所有Y、D、N都用同一个矩形`W_c(φ,m)`变换，不是只丢Y前p行，不是把残差先去均值后任意拼接。prefix仍作为后面p个innovation的lag条件值参与，不是从原始授权/标签账本中消失。该模型不需要stationary startup covariance Γ；它改变的是开始p个观测的概率处理，**没有恢复任何被滤掉的信息**。

成熟软件依据为[statsmodels v0.14.6 AutoReg](https://github.com/statsmodels/statsmodels/blob/v0.14.6/statsmodels/tsa/ar_model.py)：源L66明确conditional MLE/OLS与SARIMAX full MLE不同；`_setup_regressors` L350–385构造lag后应用hold_back；`loglike` L500–518只用实际保留的nobs。官方[`hold_back`文档](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html)明确可以固定不同AR阶数的估计样本。本项目是带候选设计/通道covariance/核先验的扩展，不能宣称单导AutoReg就是全部方案。

### 当前p=32、onset=51的含义

在本项目**假设的因果有限核设计**中，前32行D_q=0，且所有候选相同。该事实可以直接核设计矩阵。但它不证明物理信号前32点没有与候选有关的响应：分发滤波相位/边界未知，也可能有未建模的早期响应或前事件残留。因此后面的等价性只能在明确模型条件内陈述，不能转述为采集生理事实。

## 三、何时full Bayes factor与conditional相等

### A. 参数已知且prefix候选分布相同：相等

固定β、φ、Σ，若`p(Y_prefix|q,θ)`对q完全相同，则

`p(Y_all|q,θ)=p(Y_prefix|θ) p(Y_suffix|Y_prefix,q,θ)`，

公共startup因子抵消，full与conditional log-BF相同。固定噪声与nuisance，只有h存在独立Gaussian后验且D_prefix=0时，prefix也不会更新h；先积分h后仍可得到相同BF。

### B. 未知nuisance/噪声等共享参数：不能直接取消积分里的因子

令θ含未知h、nuisance a或噪声超参数。正确full Bayesian预测为

`p(Y_all|q,C)=∫p(Y_prefix|θ,q)p(Y_suffix|Y_prefix,θ,q)p(θ|C)dθ`。

若积分后的prefix概率对q相同，full BF等于**正确更新prefix后的**conditional predictive之比：

`p(Y_suffix|Y_prefix,q,C)=∫p(Y_suffix|Y_prefix,θ,q)p(θ|Y_prefix,q,C)dθ`。

它一般不等于`∫p(Y_suffix|Y_prefix,θ,q)p(θ|C)dθ`。D_prefix=0只排除了prefix直接观测h，不能排除它更新constant/linear nuisance、φ、Σ，或通过先验相关间接更新h。

使用flat nuisance时尤其明显：先在full stationary likelihood上积分a，和只用conditional rows积分a是两个不同统计对象。后者仍可作为**预声明的conditional integrated likelihood**研究路线，但不能称前者的数值等价实现。要恢复原full Bayes预测，必须按prefix更新a或其联合分布；这样可能再次需要原startup概率，因此不能同时许诺“完全绕开Γ且保留全部full-model信息”。

### C. 一个可直接实施的conditional模型

令`Z=W_cY`、`D*=W_cD_q`、`N*=W_cN`；Q为N*的正交补基，`Ybar=QᵀZ`、`Dbar=QᵀD*`。将nuisance以该conditional模型的flat prior积分，h的训练后验为`MN(H,V_h,Σ)`，则剩余行covariance为

`K_q=I+Dbar V_h Dbarᵀ`，

候选分数为

`−1/2{c log|K_q| + tr[Σ⁻¹(Ybar−DbarH)ᵀK_q⁻¹(Ybar−DbarH)]}`。

公共的`log|N*ᵀN*|`、Σ与2π项可在**固定φ/固定m/固定nuisance设计的候选比较**中消去；若优化φ、比较不同m的绝对likelihood、或nuisance随候选改变，则不能任意省略。有效评分维数为`n−m−rank(N*)`，拟合与noise df、核posterior必须同步使用conditional规则，不能拿stationary fit的记录冒充重拟conditional模型。

## 四、与现有门禁和对照的关系

- `stationary_AR_latent_initial_state`保持原条件数门禁与48cell拒绝记录；禁止失败后静默切conditional。
- 新计划/新artifact明确`startup=conditional_on_observed_prefix`、p、共同m、prefix/suffix物理样本索引、第一刺激onset及设计prefix检查。数值稳定性、AR系数/反射系数、合法输入、row identity等原有门禁继续适用。
- conditional概率在数学上不要求stationary prior，但本轮不能把这点当作放宽已有稳定pole要求的授权。研究估计量与数值准入分别写清楚。
- 比较AR阶数时建议所有conditional臂固定`m=32`，包括AR0/8/32；保持同样评分rows，并分别报告使用到的lag。若各阶默认m=p，差异还包含评分样本/前缀信息不同。
- 至少保留AR8 stationary↔AR8 conditional桥接。AR8 stationary↔AR32 conditional同时改变order与startup likelihood，不能单因素归因于高阶AR。
- prefix观测及等待仍计入真实成本，requested participant/decision/label分母不减少；这是更换似然建模，不是删除坏trial。

## 五、独立数值反例

脚本`conditional_likelihood_counterexamples.py`，seed43251，项目venv运行；Ruff通过。没有读取真实测试label、没有训练或修改production。

| 条件 | 验证结果 |
|---|---|
| 固定参数、candidate-independent prefix | full/conditional BF差5.33e−15 |
| 仅h不确定且D_prefix=0，噪声/nuisance已知 | 积分后BF差8.88e−15 |
| AR1=.8、prefix无candidate均值，但unknown flat constant nuisance | full BF7.2727，conditional-only BF2.56 |
| 同模型，Y=[4,1,−2] | full BF−24，conditional-only BF+1.92，候选排序直接翻转 |
| proper nuisance prior，正确由prefix更新 | full BF与conditional predictive均7.32203，差1.78e−15；沿用未更新prior仅2.96727 |
| candidate-dependent prefix | full与conditional BF相差23.04，恰等prefix BF；startup不再是公共因子 |

上述反例证明估计量边界，不能据此预判conditional路线在真实BNCI一定改善或一定下降。新路线是否可用，需要按主线已固定的校准内OOF计划检验，并保留已观察开发数据的身份。
