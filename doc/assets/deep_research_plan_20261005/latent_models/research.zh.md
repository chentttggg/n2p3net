# ERP 潜在响应、时间噪声与空间先验：深度研究规格

日期：2026-10-05，Asia/Shanghai。范围是原愿望清单第三、四、五、六组，以及四周路线的 AR/先验/空间部分。已完成当前源码核对、一手来源定向检索和 14 项独立数学检查；生产实现、公开数据性能矩阵、成人确认均为 **not_started**。本文件是独立研究证据，由六份活动指南索引，不构成另一套活动合同。

裁决：先把观测支持、候选生成模型、识别空间和先验固定下来，再增加模型容量。最先推进固定 hold-back 的条件 AR 对照与基函数/先验坐标一致性；ERP 随机效应和空间均值先验可以并行设计，但依赖同一观测几何。逐通道 AR、动态因子、VAR、重尾模型是分层扩展，不能直接塞进现有共享时间谱的矩阵正态后验。无限算力不能增加独立参与者，也不能辨识未观测或被参考投影丢弃的方向。

## 1. 当前事实、源码身份和实际调用者

隔离源码基于 `e442e4d6e007b41694a86f2220c463d0838b9b4c`。原树存在并发修改，只读。独立脚本明确优先隔离树 `src`，其数值导入路径及五个 owner 的 SHA-256 存在 [math_results.json](math_results.json)；原树研究 helper 和文献文本身份见 [read_source_manifest.json](read_source_manifest.json)。这是本轮实际读过的源，不给旧实验重新标源码。

| 职责 | 当前 owner / 调用者 | 当前能力与研究缺口 |
|---|---|---|
| 连续 ERP 与候选评分 | `src/transfer/continuous_decision.py`：`smooth_erp_basis`、`_event_design`、`ContinuousDecisionModel.fit/score`；`experiments/run_bnci_continuous_study.py`、`run_operator_decision_study.py` | 默认 8 个 Gaussian 基，common 与 target increment 各一组；共享 ridge 和 AR1；常数/线性趋势在白化后投影；保存核后验但未积分 AR/噪声估计与生理随机性 |
| 通用分段 AR | `src/transfer/temporal_noise.py`：`TemporalNoise.fit/whiten`；研究 helper、`src/transfer/exact_covariance.py` | pooled Burg、不拼接独立记录、stationary/conditional_prefix/PARCOR stationary_innovations 已存在；只支持共享时间谱，conditional_prefix 实际丢掉前 **p** 行，不是固定 32 行；白化不除以 pooled innovation variance |
| 算子 GLS | 原树 `doc/assets/operator_research_20261004/continuous_model.py`：`OperatorGLS`；`experiments/run_e2_prior_scale.py` 与本地算子研究脚本 | **研究 helper，不是 `src` 生产 owner，也不在旧 443-member 源码冻结中**。本轮 SHA-256 为 `30197962f36bf30a8281649f4f673b3a637f3651f9db53626d974e78c3a08be8`。已有 augmented-QR ridge、Burg、条件前缀、PARCOR、innovation/marginal prior convention；filtered 臂不是完整 `F K Fᵀ` 似然。不得复制它形成第二个生产连续模型 |
| 边界联合协方差 | `src/transfer/noise_model.py`：`JointNoiseModel`；`operator_lda` 等 | 估计成对 raw/state draws 的均值和联合协方差，保留交叉项；不自动推断真实外部 state。它不是时变动态因子的现成实现 |
| 成熟 ERP | `src/baselines/structured_erp.py`：`StructuredERP`；`transfer.calibration_candidates.fit_calibration_candidate`、`transfer.calibrated.CalibratedPredictor`、`experiments/run_erp_comparator.py` | Toeplitz/pooled/temporal LDA 与 xDAWN ERP，纯数值持久化；Toeplitz 要求无 baseline、等宽 bin、内部 channel-prime；xDAWN 用实际观测秩。这个 Interface 可继续作为强对照，不改名假装新方法 |
| 源时间基 | `src/baselines/temporal_transfer.py`：`TemporalBasis`、`resolve_temporal_basis`；`experiments/fit_source_temporal_basis.py`，temporal LDA 调用链 | 源参与者等权、归一化时间 Gram，只转移时间子空间；目标用实际原生通道。固定 cosine 是不同来源角色的对照。它是 epoch 时间投影，不直接等于连续 event-response 字典 |
| 神经模型先验 | `src/models/parameter_groups.py`：`SourceParameterPrior`；`SubjectAdapter` / 源训练目标 | 参数坐标 source-centered regularizer，不宣称网络重参数化不变。连续线性 g-prior 不自动适用于网络参数；N2P3Net 当前保持独立专家 |

旧 Phase 4 原始 JSON 的三专家开发结果为 8 人、6 个相关 selection fold/人，hit `0.7708333333333333`、NLL `1.0694548780609707`。它不证明 AR8 在新参与者上最优，也不提供本轮模型的非劣门槛。愿望清单中的“Pz 权重下降”“AR32 残差改善而判别变差”作为历史开发观察保留；本轮未复算完整旧预测，不能把它们变成普遍规律。

## 2. 一个统一的生成问题，而非越来越长的参数清单

对独立 selection `s` 的授权 raw 坐标观测 `Y_s ∈ R^{N_s×C_s}`，令 `t_i` 为完整已知刺激时间，`m_i(q)` 为候选 `q` 的公开 flash membership。用物理时间 `u=t−t_i` 定义响应：

\[
h_{i,s,c}(u,q)=h_{0,c}(u)+m_i(q)h_{\Delta,c}(u)
 +h_{\mathrm{line}(i),c}(u)+u_{s,c}(u)
 +m_i(q)v_{s,c}(u).
\]

`h0` 是所有 flash 共有的 ERP；`hΔ` 是目标增量；line 是行/列或刺激类别的预声明对比；`u_s` 是 selection 共有效应；`v_s` 是 selection 目标变化。各层是否需要由实验决定，不一次全部启用。离散卷积设计为

\[
Y_s=D_{0,s}B_0+D_{q,s}B_\Delta+D_{\mathrm{line},s}B_\mathrm{line}
     +Z_sU_s+Z_{q,s}V_s+N_sA_s+E_s.
\]

`N_s` 是漂移等 nuisance；已知有限观测算子 `F_s` 同时作用于信号、所有设计以及噪声/外部状态分布。若选择 raw 模型则 `F_s=I`。`Z_sU_s` 是真实 selection 变异，不能与有限滤波器确定 padding 误差混为一个随机状态。高斯随机效应积分后产生的协方差 `Z_s H_U Z_sᵀ` 与 kernel 估计不确定性 `D_q H_B D_qᵀ` 有不同含义、不同来源和不同可重复性，artifact 必须分别记录。

**拟合和预测目标：**

\[
p(q\mid Y_H,C)\propto\pi_q\int p(Y_H\mid q,\theta)\,p(\theta\mid C)\,d\theta.
\]

`C` 仅包含合法源训练/目标校准信息；`θ` 包含响应、噪声、随机效应和先验层参数。小模型先做条件于拟合噪声的精确 Gaussian 计算，再做超参数积分的独立臂。条件似然的参数不确定性覆盖度和 Gaussian 拟合残差是诊断，不是分类收益证明。

### 2.1 识别性在正则之前

common ERP 加每个 selection 无约束独立均值会产生重复方向：把共同系数加 `a`、所有 selection 系数减 `a`，观测均值不变。采用 `Σ_s u_s=0` 的 contrast 编码，或定义具有 proper 零均值分布的层级随机效应；后者用先验消除概率模型歧义，但有限样本仍可能弱识别，不能声称数据确定了各层。

候选信息应检查 **校准后验预测均值/协方差的实际差异**。只检查 `rank(D_q−D_q')>0` 不充分：该差异可能落在 nuisance 投影、参考 null space 或零目标核方向中。对于等协方差的已知均值模型可报告 `d²(q,q')=ΔμᵀC⁺Δμ` 与支持一致性；异协方差候选还需要均值和协方差完整密度比较，不套用等协方差信息上界。

## 3. ERP 字典、延迟和随机效应

### 3.1 按函数而非按列名定义字典

字典 `Φ(u)` 必须记录秒、采样率、支持 `[u_min,u_max)`、实际离散点、端点规则和坐标单位。连续 basis 不直接继承 FFT epoch 缓存的下标长度；源 epoch basis 进入连续卷积前须声明是否含刺激前 baseline、重采样和 `F`，不能把 `TemporalBasis.matrix` 截几行就当 raw kernel。

比较 Gaussian、B-spline、raised cosine、wavelet 和多尺度 basis 时先做 **同函数子空间的坐标变换对照**，再做不同子空间的容量对照。前者必须预测相同，后者比较 held waveform error、candidate hit/NLL 与实际有效自由度 `tr(D(DᵀD+Λ)⁻¹Dᵀ)`。相同 basis 个数不等于相同有效容量。Gaussian 基当前宽度在数组索引中定义；新 basis 用物理秒生成，必要时通过数值 quadrature/采样形成观测设计。

端点的三个研究臂分别是 `h(u_min)=h(u_max)=0` 的硬约束、自由端点、带 proper 方差的软端点。硬约束等于删除信号方向；对真实慢波或已截断响应可能造成系统偏差。端点选择与响应支持/滤波方式作交互，不能把更低 residual MSE 自动解释为更好 ERP。

### 3.2 延迟是局部非线性，导数是近似

\[
a_s h((u-\delta_s)/w_s)
 \approx a_s[h(u)-\delta_s h'(u)+\tfrac12\delta_s^2h''(u)
                -(w_s-1)u h'(u)].
\]

在独立 Gaussian 峰反例中，4 ms 延迟的一阶相对误差 `8.55e−4`，二阶 `2.00e−5`。这只证明小延迟 Taylor 展开有效，不能给真实 ERP 定义通用的“4 ms 足够”。大延迟、双峰、截断边界须使用显式 shifted-basis 或数值积分对照。随机 `δ_s` 的一阶积分产生 `Var(δ_s) h' h'ᵀ`，它是 **信号相关** 的低秩协方差，不是独立 stationary AR 噪声。

本轮优先 selection 级随机 amplitude，然后 selection 级 latency，最后 width；逐 flash 自由 latency 需要更多信息，容易追随噪声。训练估随机效应分布；held 评分时积分该分布，若按每个候选优化不同随机效应则必须有 proper prior 和同样积分/复杂度项。无惩罚 profile likelihood 会让错误候选也解释噪声。

### 3.3 完整 event history

对当前 selection 之外的已知公共 flash 仍建立 common design；未知目标 selection 的 increment 按其未知候选状态边缘化，或用公开训练得到的 state distribution。不能删除其事件然后称前缀无 ERP。固定 SOA 的卷积设计可能秩亏；需 SVD 与 event-phase 覆盖证据，不靠 ridge 数值可逆证明可分离。完整已知 history 可并行与 boundary state 研究，但两者共享 event/时间/预算合同。

## 4. 时间噪声的层级和固定支持比较

### 4.1 固定 hold-back=32 的条件创新：拟合/评分同支持

候选阶数集合为 `p ∈ {0,1,4,8,16,32}`，每个独立记录读取样本 `[0,N)`，条件前缀固定 `[0,32)`，只在 **同一 tail `[32,N)`** 计算训练残差诊断、创新方差、目标 likelihood 和评分。定义

\[
r_t(q,B)=Y_t-D_t(q)B,\quad
e_t=r_t-\sum_{j=1}^{p}\phi_j r_{t-j},\quad t=32,\ldots,N-1.
\]

因此 `W_p Y` 和 `W_p D_q` 都要在全记录中用实际前缀构造，再选择 `[32,N)`。先裁出 tail 再 whitening 会把前缀用掉的观测变成虚构启动；只白化 Y 会错误改变均值模型。本轮在 `N=180` 的所有六阶得到相同 148 行，设计/数据同步白化 residual 最大误差 `1.78e−15`；更改第 31 个样本改变实际创新，证明它确实被用作 history。

`TemporalNoise.whiten` 当前 conditional 模式返回 `[p,N)`；未来 **一个** observation-support owner 内负责 `whiten(full)[32-p:]` 与设计同行选取，调用者不各写一次切片。artifact 记 `read_samples=N`、`conditioning_samples=32`、`score_samples=N−32`、响应生成 history 与其授权来源。AR0 读取同一信息集合但其 Markov 模型不利用 prefix；不可因为它不利用而给其它臂免费多读 raw。

Burg 的不同阶数前后向 pair 数天然不同。因此把 Burg作为初始化后，精确条件 objective 的优化/创新方差/选择统计均使用相同 `[32,N)`；保存初始化数据范围。只比较 Burg结果的独立估计器臂必须承认其内部估计权重和 pair 数不同，不能宣称“同 row objective”。先不借 `minimum_samples_per_parameter=20` 这个工程门槛断言统计识别，改用设计秩、有效行/参数数、模型恢复模拟和 posterior uncertainty；保留现有入口门禁直到新数学合同经实现替换。

### 4.2 stationary marginal、conditional prefix、causal state 三个估计量

平稳模型 `r∼N(0,K_p⊗Σ_e)` 的 startup 来自稳定 AR state 的平稳协方差；stationary_innovations 的 PARCOR lattice 是同一平稳概率模型的数值实现，应与 dense 解一致，不是第四个估计量。

若比较相同 tail 的 stationary **marginal**，应使用 `p(Y_[32:N]|q,C)`，不读取/使用真实 prefix；不能把全记录 stationary log-density丢掉前 32 个 log项就说获得边际 tail。条件模型则是

\[
p(Y_{tail}\mid Y_{prefix},q,C)
=p(Y_{full}\mid q,C)/p(Y_{prefix}\mid q,C).
\]

固定 prefix 已知时的 Gaussian Schur 补均值/协方差必须同步处理响应不确定性。**候选无对比不等于参数无信息**：prefix即使只含 common ERP，也会更新该 ERP 核的 posterior；本轮独立高斯反例忽略这一步产生 `0.16056` 的 log-density 差。只有 prefix 均值已知、或经正确定义的完整联合预测密度相除，才能直接用纯创新公式完成条件化。若 prefix 有未知邻近 selection，须边缘化其状态；`onset[0]>=32` 不能证明外部事件的 response 已结束。

causal-state 研究读取合法过去，保存实际跨 chunk state、availability 与标签使用范围；其 state 初值/重置策略不同。独立记录之间不传递 state。前缀可能增加等待和 raw acquisition预算，三估计量的输入身份/延迟不能互换。

### 4.3 Burg、PARCOR 和精确优化

稳定 AR 用反射参数 `κ_j=tanh(η_j)`；Burg 得到实际 PARCOR初始化。接着分别以固定 tail 的条件 Gaussian objective和完整 stationary likelihood优化 `η`、创新协方差与均值先验参数。stationary objective保留 startup logdet、有效样本数和同一参照电压单位；条件 objective不加入 stationary startup项。当前可行 GLS/Burg+LW迭代不保证 likelihood单调，不能重标为精确MLE。

高阶expanded polynomial的 root-finding可能报告假不稳定；实际 lattice `|κ|<1` 是模型稳定参数条件，但接近单位边界仍可能数值病态。监测创新factor的 condition、log variance、precision 和实际 likelihood误差；用合成真AR恢复、dense小网格、长序列stationary ACF和高精度小阶参考交叉校验。不要给每个caller加 pole clipping、epsilon和PSD补丁。

### 4.4 从共享谱到非分离模型

| 层级 | 数学模型 | 参数/计算特点 | 研究裁决 |
|---|---|---|---|
| T0 | `E∼MN(0,K_p,Σ_e)`，所有通道同 `φ` | 标量 AR shape+空间innovation；沿用低秩核更新 | 第一优先锚点；先确认相同tail及噪声尺度 |
| T1 | `e_{t,c}=r_{t,c}−Σ_jφ_{c,j}r_{t-j,c}` | 独立channel谱且innovation独立时 block-diagonal；innovation相关时是 diagonal-lag VAR | 不再是 `K_time⊗Σ_space`。本轮两个channel的lag1为 `.8`与`−.4`，最佳共享lag相对误差`.5845`；不能继续使用一个MNrow后验 |
| T2 | `r_t=L f_t+u_t`；`f_t=A_f f_{t−1}+ξ_t`，私有 `u_t` 有各自AR | 低秩共享动态+私有谱；loading/factor有旋转gauge | 首先共享rank1/2、private AR0/1；与同容量T1配对，显式识别loading尺度/旋转 |
| T3 | `r_t=Σ_jA_j r_{t-j}+e_t`，稀疏/graph/group VAR | full VAR有 `C²p` 个lag参数；输入reference和缺导改变空间坐标 | 小C先VAR1/4，不直接从AR32跳到VAR32；正则从训练信息选择，观测缺导不可补零 |
| T4 | 连续state-space/Matérn GP，piecewise/nonstationary AR | rational spectrum可精确state-space；一般kernel需近似；change points可增加过拟合 | 与T0同条件支持对照；future smoother是离线协议，online filtering独立，不称等价 |
| T5 | 明确层级的Student-t/scale mixture/matrix-t | outlier local、selection全局、channel covariance不确定性含义不同 | 先one-row/短窗尺度冲击，后selection尺度；不将任意t分布称作matrix-t |

AR同谱假设是待证伪的容量限制：用训练残差跨channel normalized ACF/PSD差异与 held预测二次型分层；若差异只由未建模ERP造成，增加AR层级是在吸收信号，应先回到响应字典对照。动态因子/VAR需预测 `vec(Y)` 的一般协方差或状态空间innovations，不强行沿用当前MN特定Kronecker posterior formula。

### 4.5 重尾要放在正确层级

局部Gaussian scale mixture例如 `e_t|τ_t∼N(0,Σ_e/τ_t)`，`τ_t∼Gamma(ν/2,ν/2)`；这是时刻级multivariate Student-t。selection共享 `τ_s` 让整段同时增噪；channel私有scale改变空间分布；inverse-Wishart积分channel covariance得到matrix-variate预测模型，有别于把所有行flatten后套同一个scalar t。当前kernel covariance、noise covariance、boundary-state variance各自拥有scale，不重复积分同一variance来源。

独立反例的残差 `[12,.3]`，`ν=5`时global scale posterior precision为`.04695`，第二个局部干净coordinate为`1.17878`。它说明同样“t尾”会给出不同的downweighting，不证明某种t模型适合真实EEG。异常挑战须同时包含single-sample glitch、持续selection gain、channel污染、真实大P300：用无候选标签的尺度估计仍可能吞掉ERP。

## 5. g-prior、弱方向、收缩层次和概率混合

### 5.1 满秩、固定校准设计下的 g-prior

令同一合法校准设计白化/去proper或diffuse nuisance后为 `D_C`，`G_C=D_CᵀD_C`。基本MN先验为

\[
B\mid g,\Sigma_e\sim MN(M_0,H_g,\Sigma_e),\qquad H_g=gG_C^{-1}.
\]

换基 `D'_C=D_CA`、`B'=A^{-1}B` 必须有 `M'_0=A^{-1}M0` 和 `H'_g=A^{-1}H_gA^{-T}`，预测分布严格不变。独立反例随机非正交换基的最大predictive covariance误差 `1.11e−15`，固定ridge `H=λ⁻¹I`变化达到相对`831.65`；前者消除坐标假收益，但不会证明新子空间具有更高hit。

**G仅由训练C确定并被冻结**。每个test候选都使用同一个kernel posterior与物理prior。用 `D_qᵀD_q`随候选重造prior是另一个candidate-specific generative model，会使激活次数/设计尺度影响prior。独立幅度反例中冻结physical prior的logBF为`.61810`，per-q适应g-prior后为0。除非显式建立并验证候选条件prior，不能把这种归一化悄悄纳入评分。

### 5.2 common、target、selection 使用不同收缩

`D=[D0,DΔ,Dline]` 常相互共线。直接给full G乘不同group scalar通常不对称，不是covariance。可用分组proper covariance，或先将target对common/nuisance残差化，定义层级参数化，再令

\[
H=\operatorname{blockdiag}(g_0G_0^{-1},g_\Delta G_{\Delta\cdot0}^{-1},g_{line}G_{line\cdot0,\Delta}^{-1}).
\]

这是一个明确的ordered conditional prior，顺序和中心化属于模型定义，应与不残差化的physical GP prior作对照。它保证组内换基不变，**不保证跨组混合换基不变**，因为组表达生理语义。common强收缩与selection强收缩的相对顺序没有普遍答案；small K先缩selection variation，common低秩source先验与target增量分别做归因。只在相同观测/噪声下扫描 `(g0,gΔ,τ_selection)`，报告交互而非把三项一起提升。

### 5.3 秩亏和弱方向不能由伪逆补齐

`G`秩亏时，`G⁺`只定义训练prediction column-space中的g-prior；对系数null方向默认零variance是一项额外先验。任意非正交换基后，Moore-Penrose所选“最小norm系数”不同，held设计有训练null方向时预测不唯一。本轮同一训练projector误差`1.67e−16`，heldpriorvariance却从`.00064935`变`.00166234`。

采用SVD识别空间 `D_C=U_rS_rV_rᵀ`，先报告held设计在 `ker(D_C)` 上的norm及真实可观测响应。两条合法路线：

1. 将模型函数空间预声明为校准可识别subspace，held在外方向明确不受支持；这限制估计量，不能冒充完整原basis。
2. 给null/weak方向另一个以physical time/space函数定义的properprior，比如smooth GP或差分penalty，保存 `H_identifiable + H_unidentified` 两部分，承认外推依赖先验。

纯g-prior会在weak方向产生巨大coefficientvariance；本轮singular value`1e−5`、g=9时variance`9e10`，训练fitted函数每方向仍缩为`.9`。因此g-prior不是自动惩罚所有弱方向的万能ridge。数值rank容忍值按单位、noise与solver误差定义，并用敏感性/恢复模拟验证，不能把固定`1e−10`当成研究阈值。

### 5.4 MN gauge、电压单位与真实AR变化

若同一noise covariance分解改写为 `K→aK,Σe→Σe/a`，总`Σe⊗K`不变；MN coefficientprior的rowfactor须 `H→aH`。固定ridge即`λ→λ/a`；g-prior因`G_C→G_C/a`自动满足。若物理单位 `Y→sY`、`M→sM`、`Σe→s²Σe`，MNrowprior保持相同，不能再把λ乘`s²`。真正更改AR参数使Kshape改变，它是新概率模型，不是gauge，预先声明innovation/marginalpriorcoordinate才有意义。

反例correctgauge logBF误差`2.13e−14`，g-priorgauge误差`7.11e−15`，V→µV误差`2.84e−14`；错误不改λ的gauge差`1.66576`，按noise-scale改λ的单位差`1.84833`。artifact至少记录voltageunit、ARreference、unit-innovationconvention、channelcovariance、prior函数/row/columnscale和rank。ARmarginalvariance统计值是所声明stationaryreference，conditionalprefix不因此变成stationarylikelihood。

### 5.5 lambda mixture 的三种不同模型

固定probabilityensemble为 `p_mix(q|Y)=Σ_λw_λp_λ(q|Y)`，每个expert先归一化，w在held评分前冻结。它有效、可检验，**不是**把超参数从jointlikelihood积分掉。

Bayesianhyperparameter预测先用calibration evidence更新

\[
p(\lambda\mid C)\propto p(C\mid\lambda)p(\lambda),\quad
p(Y_H\mid q,C)=\sum_\lambda p(Y_H\mid q,C,\lambda)p(\lambda\mid C),
\]

再对q归一化。若将current无标签Y也用于modelposterior，联合混合规范为

\[
w_\lambda(Y)\propto p(\lambda\mid C)\sum_q\pi_q p(Y\mid q,C,\lambda).
\]

这里不能使用test真标签；candidate-independent常数在各λ模型中仍影响evidence，不能删完常数后把`log_evidence`直接混合。独立反例固定probabilitymix为`[.5,.5]`，Bayesmodelweights为`[.98039,.01961]`，Bayesposterior为`[.78824,.21176]`。

robustempiricalBayes在C里估λ/g/variance后冻结，是plug-in模型；crossfitcalibrationstacking是predictionrisk优化；weaklyinformativehyperprior积分是又一模型。分别保存拟合来源、归一化常数和超参数预算，不能都标`hyperprior_mixture`。适当priors和同physicalsupport是跨模型evidence比较前提；diffusenuisance若modeldimension或尺度不同，其任意常数使Bayesfactor失去定义，应改properprior或仅在同nuisance测度的已声明比较中使用。

## 6. 空间观测、Pz、graph 与成熟空间专家

物理模型是 `Y=A_obs S+noise`；`A_obs`包含实际测量/电压gain/reference/通道选择和合法投影。原生布局不同就用各自真实坐标。七导CAR的covariance秩最多6；Cz重参考再保留六个差分不一定在其六维坐标中继续秩亏，具体rank由 `A_obsΣA_obsᵀ`决定。native耳参考七导、CAR七导、Cz六差分分别声明，不能只根据最终通道数猜noise秩。

Pz信息若来自targetERP，只应先影响 **signalmean/meanprior**，而不是把residualprecision手工乘数。目标均值先验可定义

\[
\operatorname{vec}(B_\Delta)\sim N(m_\Delta,H_{time}\otimes S_{signal}),\quad
S_{signal}=(\lambda_0I+\lambda_LL_{graph})^{-1}
\]

或训练scope的ROI层级varianceprior。noise独立采用实际估计 `Σnoise`。若当前MN用 `Σnoise`作为kernelpriorcolumnfactor，那么加入独立Pz/graph信号columncovariance后通常**不再共轭为当前的单一MNrow后验**；必须用generalGaussianprecision或对whitenedspacechannelprior做谱分解，不能只给rowcovariance加一个Pz项。small C和低temporalrank使该generalGaussian问题可执行。

`Lgraph`由实测电极坐标定义，graph先验的常数null方向必须以proper λ0、参考子空间或explicitintercept处理。未知位置不按通道名猜、未观测Pz不补一列零；source-nativegraph通过显式observationmodel传播，不能伪造共同montage。给Pz大的priorvariance表示更宽容的ERP幅度，不等同于把其目标波形均值指定为正；reference可能改变符号，ROI理论只能提供有限先验约束。

训练估d-prime时按participant/session跨折收缩，不用一次heldselection选择sign或强度。稳定性用ERPtopography的reference-aware对比、空间subspace principalangles、预测`d²`、负迁移率；xDAWNfilter各列符号/尺度不唯一，不能直接对weight相关系数排名。xDAWN ERP的空间filter/模板是训练出的判别表示，不是raw残差precision，也不是现成ERP保护信号基。

联合空间与Toeplitz模型同时有时空结构、观察reference和baseline问题。先保持成熟Toeplitz无baseline基线，再比较等输入poolednoise、graphsignalprior与general时空covariance；不得因timeToeplitz成立就自动推导time-spaceKronecker分离。xDAWN空间expert与rawtemporalexpert的probabilitymix属于多模型路线，校准/coverage/打分常数需求见上节。

## 7. 原愿望逐项落实：问题、预测、实验和依赖

下表中P0是数学/输入依赖必须先完成，P1是第一批可执行机制研究，P2是后续容量/异构扩展。实验单元均为相同participant/完整selection的配对；公共binary-only数据只用于合法ERP/noise核学习，不伪造membership。未来实现/实验当前均not_started。

### 7.1 第三组：ERP 灵活字典（11项）

| ID / 原愿望 | 问题与公式 | 可证伪预测 / 机制实验 | owner与依赖 / 优先级 |
|---|---|---|---|
| E01 多种temporalbasis | 相同span与不同span的收益分开；`h=Φβ` | 同spanGL换基应相同预测；新增span仅在独立held响应/候选有效，不靠trainingfit | continuousbasis＋TemporalBasisadapter；先支持与g-prior；P1 |
| E02 一阶/二阶导数 | `h(u−δ)≈h−δh'+δ²h''/2` | 小δ恢复改善，大δ/双峰失配；导数臂vs显式shift积分、同自由度普通basis | continuousmeanowner；E01、时钟；P1 |
| E03 selectionamplitude/latency/width | `a_sh((u−δ_s)/w_s)`，proper随机效应 | 新selection预测覆盖改善；若只fitMSE降而hit/NLL无益则拒绝提升 | continuousposteriorowner；E02与properprior；P1→P2 |
| E04 common/target/selection分离 | `D0B0+DqBΔ+ZsUs`；centerselection | 恢复模拟应能分别估计；无center时出现exactnull；真实数据需层级增益 | 同一continuousowner；识别空间先行；P0 |
| E05 行/列响应差异 | `DlineBline`，sumcontrast | 行/列差异若稳定则heldcrossline预测改善；membership计数保持固定 | eventdesignowner；realphysicalmembership；P1 |
| E06 早期/P300/晚慢波 | 支持分段`Φearly⊕ΦP300⊕Φlate` | 去某段应产生可解释的候选contrast损失；同总dfmatchedcontrol | basis＋kernelprior；E01、nuisanceoverlap；P1 |
| E07 硬零/自由/软端点 | constrainedsubspacevsproperendpointvariance | 端点硬零在slowwave/censor挑战可能偏差；端点×window×filter配对 | basisartifact；响应支持；P1 |
| E08 完整eventhistory | 所有公共事件D0、未知邻域目标积分 | 外部事件有尾响应时忽略history产生系统残差/候选失配 | rawhistoryadapter＋boundarymodel；共享时间授权；P0 |
| E09 filtered与rawdictionary | 同`F`传递`Y,D,C`；learnedfilteredΦ另臂 | exact可逆coordinate下posterior相同；learnedΦ收益与operator收益可分 | continuousowner＋FiniteFilterOperator；观测协方差轴；P0→P1 |
| E10 物理时间/支持 | `Φ((n/sfreq)−t_i)`与实际grid | 改sample-rate保同physicalwaveform应同机制预测；错误下标负对照 | inputcontract＋basisconstructor；P0 |
| E11 字典自由度预算 | `df_eff=tr(G(G+Λ)⁻¹)`与rank | equaln_basis不保证capacity相同；等df路线是否仍提高heldtask | posteriordiagnostics＋plan；E01/Prior；P0 |

### 7.2 第四组：AR、Burg、PARCOR 与噪声（14项）

| ID / 原愿望 | 问题与公式 | 可证伪预测 / 机制实验 | owner与依赖 / 优先级 |
|---|---|---|---|
| T01 AR0/1/4/8/16/32 | fixedtail条件noise族 | 合成已知order恢复与out-of-samplelikelihood；真实hit不预断AR8最好 | TemporalNoise；T02；P1 |
| T02 hold-back公平 | `t>=32`数据/设计同步innovation | 所有order同行、prefix真实影响；裁tail再reset负对照 | observationsupport内部seam；E08；P0 |
| T03 Burg初始化+stationary精确更新 | `κ=tanhη`，含startup项优化 | 真parameter恢复、objectiveerror、初始化敏感性；Burg-only同scopecontrol | TemporalNoise内部optimizer；T01/04；P1 |
| T04 PARCOR lattice | stationaryinnovationfactor，同概率模型 | dense小网格等价且近边界高阶不靠expandedrootclip | TemporalNoise；数值误差预算；P0→P1 |
| T05 stationary/conditional/causal | marginal/conditioned/state三model | 相同tail分布不同；exactSchur与innovation对比；跨recordstate攻击 | observationsupport＋stateowner；E08；P0 |
| T06 每channel谱 | diagonal-lagVAR或independentblock | sharedspectrum被channelnormalizedACF否证时T1heldfit改善；检查ERP吸收 | generalnoiseAdapter；不继承MN公式；P2 |
| T07 低秩动态因子 | `r=L f+u` | 共享谱冲击能被factor捕获、privateERP不丢；loadingrotate预测不变 | state-spacenoise；T06和spaceobservations；P2 |
| T08 VAR/sparse/graph/groupAR | `ΣA_jr_{t−j}`，结构penalty | knowncross-laginteraction恢复；fullVARsmallK过拟合负对照 | generalnoiseowner；channelidentity+reference；P2 |
| T09 AR/Matérn/GP/Kalman | rationalPSD→state-space | 指定Matérn下denseGP与Kalman等价；非rationalkernel近似误差单列 | TemporalNoise/state-spaceAdapter；T05；P2 |
| T10 非平稳/change-point | `φ_s(t),Σ_s(t)`或state | 预声明changes恢复、falsechange率；slowERP不可被noisechange吞掉 | state-spacenoise；T01+robustmean；P2 |
| T11 Student-t/mixture/matrix-t | `e|τ`的层级明确 | glitchvsselectiongain需不同posteriorweights；clean大ERP保护 | posteriornoiseowner；T05及Prior；P1→P2 |
| T12 train/heldACF、PSD、方差、二次型、condition | diagnostics基于同residual和support | train改善但held无益揭示过拟合；PSD/ACF不能证明label正确 | sharedresidualdiagnostics；T02；P1 |
| T13 白化是否去判别 | 正确可逆白化保`d²`；错误projection可删`Δμ` | signal注入与pairwisecontrastenergy；过度nuisance投影负对照 | posteriorcontrastdiagnostics；E04+nuisance；P0 |
| T14 白化/分类独立门 | residualfit和requestedhit分别报告 | 允许white更好但hit更差的负结果；不得残差门替代产品门 | evaluationowner＋frozencomparisons；P0 |

### 7.3 第五组：空间与 Pz（10项）

| ID / 原愿望 | 问题与公式 | 可证伪预测 / 机制实验 | owner与依赖 / 优先级 |
|---|---|---|---|
| S01 不手工放大Pznoiseprecision | covariance由noisegeneratingmodel拥有 | 手工precision在无ERPnoisechannel产生过信；与signalprior区分 | spaceobservation＋posterior；P0 |
| S02 xDAWN/Riemannian/spatialLDA | 用actualtrainERPsubspace与模板交叉项 | barecovariance极性丢失负对照、templateERP保留；同input强对照 | StructuredERP、baselines.riemann；P1 |
| S03 trainROI d-prime强收缩 | hierarchical`d'_c`/signalvariance | crossfoldROI稳定性，smallKextremevariance收缩；独立participant才确认 | meanpriorowner；真实layout和Priors；P1 |
| S04 Pz进入ERPmeanprior | `Ssignal`与`Σnoise`独立 | PzstrongERPchallenge恢复、不改变noise证书；generalpriorvssharedMN | generalGaussianposterior；S01；P1 |
| S05 graphLaplacian | `(λ0I+λLL)⁻¹`properprior | smoothspatialERP改善、hemisphereasymmetrynegativecontrol；坐标错配拒绝 | layout＋signalpriorowner；S08与rank；P2 |
| S06 半球/后部/中央层级 | groupamplitude/variance+interaction | 后部规则若跨reference不稳则限制prior；actualelectrodegroup | signalpriorowner；S03/05；P2 |
| S07 reference/gain/missing/projection联合 | `AobsΣAobsᵀ`、`Aobsμ` | voltage/unitgain同步不变、CARrank6、missing不得当实测 | inputcontract/observationowner；P0 |
| S08 participant/source/reference稳定 | projection-awaremeancontrast | crossparticipant/reference负迁移分布；不相关filter列符号 | comparison＋spaceprior；S03、原生布局；P1 |
| S09 spatialweights+Toeplitz | 结构timecov与spaceprior独立 | baseline×reference×signalpriorfactorial；验证非Kronecker残差 | StructuredERPanchor＋generalposterior；T06、S07；P2 |
| S10 xDAWN+rawtemporal概率混合 | `Σw_mp_m(q)` | complementaryerrors配对，NLL与hit双读数；calibrationweight冻结 | commonexpertInterface；candidateprobabilitycontract；P1 |

### 7.4 第六组：先验族（11项）

| ID / 原愿望 | 问题与公式 | 可证伪预测 / 机制实验 | owner与依赖 / 优先级 |
|---|---|---|---|
| P01 ridge改explicitpriorfamily | `B∼N(M,H)`，非单scalar标签 | 相同priorfunction重参数化同预测；weakprior识别限制 | continuousposteriorowner；P0 |
| P02 g-prior | `H=gG_C⁻¹`，C-only | fullrank换基不变、rank亏外推不唯一、per-q重建负对照 | priorconstructor＋posterior；E04/T02；P0→P1 |
| P03 predeclaredlambda mixture | probabilitymix/likelihoodintegral分立 | 固定mix与Bayesweights的独立反例；不同常数不可混 | predictiveowner＋expertInterface；P0→P1 |
| P04 ERP层级variance | `τ²∼properprior` | smallK后验覆盖、variancezero边界和signal/noiseconfound | posteriorpriorowner；E03、P01；P1 |
| P05 common/selection不同收缩 | centeredgroupcovariance/conditionalprior | jointnull恢复、两组factorial比较；不只joint-package | sameposteriorowner；E04/P02；P1 |
| P06 lowrank+diagonalkernelcov | `H=LLᵀ+diag(v)` | 真低秩恢复、heldnull覆盖；diag过小失校准挑战 | kernelprior内部；P04；P2 |
| P07 noise/AR/ERP联合uncertainty | `∫p(Y|q,θ)p(θ|C)dθ` | simulationcoverage、plugin过信vs积分；低rankapproxerror | generalposterior/state-space；T03/P04；P2 |
| P08 robustEB/weakprior/heavytailposterior | C-onlyEB与properintegral不同arm | 少量污染改变EB参数但robust缓解；cleanERP不削弱 | priorfit＋robustnoise；T11/P04；P2 |
| P09 prior/innovation/voltageunitartifact | gauge、unit、ARmodel三分 | correctgauge与V→µVlogBF不变；错λ负对照 | singleartifactload/create；P0 |
| P10 lambda敏感性 | 完整grid的hit/NLL/maxp/residual/df | 未达到稳定区域则不可单值晋升；grid每臂保失败分母 | plans/evaluation；P03及相同support；P1 |
| P11 禁held选择温度/λ/order/basis | 预声明或nestedcalibrationfit | label-erasure与heldtruth不可达；selection成本/df全计 | fit授权与recipeowner；P0 |

共46项：第三组11、第四组14、第五组10、第六组11。原组数来自恢复源逐行计数，不将表格摘要当作新增愿望。

## 8. 机制矩阵、因果归因与四周依赖

### 8.1 分阶段实验而非一次打包所有创新

| 矩阵 | 固定条件 | 变化因素 | 直接回答 / 淘汰条件 |
|---|---|---|---|
| M0 数学/保存重放 | 小denseGaussian、knowntruth、同physicalunit | basisGL、gauge/unit、rank、prefix/history、candidatepermutation | 数学等价、prior来源、输入身份；任何等价性失败先修模型公式和Interface |
| M1 AR公平对照 | basis/先验、完整readscope、fixedtail32、同mean/nuisance | 六阶AR，Burg-only与exactconditional update；stationary另panel | 白化/variancecalibration与hit是否相容；不能从residualfit选冠军 |
| M2 ERP×prior归因 | 一种声明noise与fixedsupport | {当前Gaussian、新basis}×{fixedridge、C-gprior}；同spancontrol；随后derivative | basis容量与prior坐标假效应分别估计；数据中弱方向prior主导时限制结论 |
| M3 层级response | 与M2相同noise/dfledger | common/target分组收缩×selectionamplitude；随后latency | 形态randomness还是估计uncertainty；heldprofile-only拟合不得当收益 |
| M4 spaceprior | actualnativeobservation、相同tail/kernel | sharedMNsignalprior、independentROI、graph；noiseΣ固定/另臂 | Pz信息在均值还是precision；非conjugate实现必须dense等价 |
| M5 noisehierarchy | 最简合格ERP与同support | sharedAR→perchannel→dynamicfactor→smallVAR；Gaussian→局部t | sharedspectrum假设能否被helddata否证；noise吞ERP则回溯M2/M3 |
| M6 combination | 各expert同physicaldecision/预算 | 成熟xDAWN/Toeplitz、raw连续与N2P3Net的固定mix/校准gating | 互补信息与weight选择风险；mix概率不伪装Bayes evidence |

M1与M2可在M0/观测合同完成后并行；M4理论推导可同时开始，但数据arm依赖sameobservation。M3不能假称独立于prior，M5不能独立于ERPresponse。三条轴共享noise/mean/支持，因此原四周表中的“完全独立”不成立。算子精确传播/边界state是另一研究轴，但依赖共享observation和eventhistory。

未来性能矩阵使用已有预声明seed `[20260918,20260919,20260920]`。deterministic经典owner重复seed若模型完全相同不重复fit，但将同预测纳入matchedseedcomparison，记录复用身份。seed是算法随机性重复，不是3倍独立人数。统计按participantmacro，bootstrap以participant为最外层，并保留session/selection配对，六个留一selectionfold不可当六人。

K0只能用source/fixedprior，不能在目标上拟noise、ROI或ARshape；targetonlyg-prior依赖C，所以K0必须切到明确source/fixed先验或标不可执行。K200/800/2000的真实flash预算、是否完整selection、conditioningcontext和有效designrank分别记录；不能因为存在800个标签就假定有足够独立selection估randomvariance。

候选主指标为requestedparticipant-macro `hit@all`。coverage、fitfailure、tie、abstain保留在requested分母；NLL/Brier/maxp是二级概率读数；报告均匀fallback需显式schema，不能仅在scoredsubset平均。残差ACF、PSD、innovationvariance、预测quadratic、condition、contrastenergy、runtime/availability同时保存。正式晋升非劣/提升阈值由主计划在独立队列前冻结，本轴不凭旧`.771`或`log36`发明通用pass标准。

### 8.2 四周应是依赖窗口，不是到期自动晋升

第一周完成M0、观测support/priorconstruction的Interface设计及保存重放规格；成熟expert已经可复用，连续模型先维持独立researchrole。第二周并行M1和M2；stationary/conditional分成各自panel，不以总NLL混合排名。第三周先M3和ROImeanprior，再启动有充分channel/time信息的M4/M5小层级；group/graph/VAR不能仅按日历加入默认。第四周只集成通过数学与实现门的预声明候选，独立参与者/四库三seed的benchmark需要另行执行；未满足依赖就明确unfinished，不把已有八人开发再评分标独立确认。

## 9. 小而深的 Interface：减少 caller 防御

不新建一串只传metadata的wrapper。未来沿已有continuousowner深化一个拟合/预测seam，并保留真实varyingAdapters：不同responsebasis、conditional/stationarynoise已有两类行为才建立内部seam。

```python
# 研究规格，尚未实现；名称最终服从existing owner。
ContinuousDecisionModel.fit(calibration_view, recipe) -> FrozenContinuousModel
FrozenContinuousModel.predict(decision_view) -> CandidateDistribution
FrozenContinuousModel.export_state() -> NumericArtifact
```

`calibration_view/decision_view`来自现有raw/admissionowner，已经拥有physicaltime、actualchannel/reference、completeeventhistory、membership、授权read/condition/score支持和label可用范围。recipe拥有basis/prior/noiseestimator和fitprotocol；不会让每个caller传十几个可能冲突的scalar。

内部只保留三个集中Interface：

1. `ObservationSupport`生成一次全record与actualhistory的paireddata/design map，拥有固定32tail、nuisance/reference可识别space、budgetreceipt。它同时给noiseestimate和候选score，避免重复切片检查。
2. `ResponsePrior`从C的jointdesign和physicalbasis创建immutableprior，拥有groupsemantics、rank/null方向、voltage/gauge、propernormalizingmeasure与source/target角色。predict不能接入一份候选设计重新fit它。
3. `NoiseLikelihood`拥有一种estimand及其square-root innovations/logdet/posteriorconditioning；sharedMN与generalstate-space是真正不同Adapters。predictresult清楚说明kernel/noise/hyperparameter哪层被积分。

输入/外部artifact进入Interface时核identity/shape/time/finite/source一次；trusted内部保持numericarrays和receipt，不在每candidate循环重读hash和schema。solver集中square-root/QR，明确支持空间，失败保具体原因；不散落ridgefloor/poleclip/zero-fill。sharedscope幂等重放和mutation检测在artifactowner集中实现，评价owner接收同一CandidateDistribution并维护failuredenominator，caller不重新拼概率或抄callback fallback。

实施时优先用 `src/transfer/continuous_decision.py`作为唯一连续owner，研究helper已验证思想迁入它后应物理归档该helper并留manifest，而不是生产import`doc/assets`。本轮只给研究规格，不擅自执行归档/删除。StructuredERP、TemporalBasis、CalibratedPredictor沿既有接口复用；g-prior不复制到N2P3Net parameterprior。

## 10. 反例证据、来源边界与交付状态

复现命令（项目venv，隔离src显式优先）：

```powershell
D:\BCI\n2p3-net\.venv\Scripts\python.exe doc/assets/deep_research_plan_20261005/latent_models/verify_math.py
```

14/14通过，seed`20261005`。检查包括fullrankg-prior换基、rank亏及weak方向、MNgauge/电压unit、fixedholdback32真实prefix、每channelAR非分离、判别丢失、PSD/ACF不能代hit、latencyTaylor、common/selection识别、probabilitymix/Bayesevidence、referencerank/Pzprior、candidateadaptiveprior/prefixkernelconditional、Student-tscalehierarchy，以及真实colorednoise在heldrecord的ACF改善而candidatecontrast丢失。结果由 [verify_math.py](verify_math.py) 生成，不以模型实现本身作为独立预期。其中fixedsupport只调用当前TemporalNoise的lineartransform，并以独立denseGaussian核对；不代表新的连续posterior实现已经存在。

白化相关反例还澄清一个易误解事实：正确、可逆、均值协方差同步的坐标白化不会丢信息；损失来自不可逆projection、错误noise权重、mismodelling、finiteparameterfit等。本轮过度projection把signalcomponentvariance降到0，但requestedhit因全部tie降到0；错误给noise方向巨大weight时hit`.9345→.5331`。另用8000个独立held合成record、64点AR1噪声和常值ERP，白化加过宽nuisance投影把heldresidualACF从`.90055`降至`−.01488`，candidatecontrast却只剩`1.23e−16`，原hit`.956`变全tie。这里AR1是合成真值，不能视为公共数据AR8证据。同一residualPSD/ACF能在两组相反truth下分别hit1/0。因此“白化使分类下降”不是白化必坏的普遍定理。

文献表与获取状态： [source_ledger.json](source_ledger.json)、[query_log.json](query_log.json)、[read_source_manifest.json](read_source_manifest.json)。Crossref部分connectionreset/404和非相关dynamicfactorquery均保留，不把失败当不存在研究。以下只引用实际核验层级：

| 来源 | 本轮核验 | 支持的具体机制 / 不能支持 |
|---|---|---|
| Ehinger & Dimigen 2019, Unfold，[10.7717/peerj.7838](https://doi.org/10.7717/peerj.7838) | 原树既存PMC全文text只读，temporalbasis/overlap段已核；本轮官方toolbox站点可达，Crossref本次direct请求失败 | 连续事件deconvolution、time-spline/Fourier、regularization、linear/nonlinearpredictors；不证明本项目candidate性能 |
| Ouyang et al. 2011, RIDE，[10.1111/j.1469-8986.2011.01269.x](https://doi.org/10.1111/j.1469-8986.2011.01269.x) | Crossreftitlequery+EuropePMC摘要；未取全文 | ERPcomponentlatencyvariability及模板估计的先例；不支持把Taylorprior当成RIDE完整复现，也无本项目threshold |
| Liang et al. 2008, Mixtures of g Priors，[10.1198/016214507000001337](https://doi.org/10.1198/016214507000001337) | Crossrefbibliographicquery核题名/DOI/作者；authorPDF失败、EuropePMC无结果，**未核全文** | 作为g-prior/hyperprior的primarybibliographicpointer；本规格invariance公式由独立推导与反例支持，不引用未读全文的实验数字 |
| Hartikainen & Särkkä 2010，[10.1109/MLSP.2010.5589113](https://doi.org/10.1109/MLSP.2010.5589113) | 作者站点PDF6页全文已提取，Matérn/rationalspectrum与approximation段核读 | temporalGP与linearGaussianstate-space、Kalman/RTS关系；一般kernel需近似，不证明online/finiteprefix等价 |
| Shah, Wilson & Ghahramani 2014, Student-t Processes，[PMLR v33](https://proceedings.mlr.press/v33/shah14.html) | PMLR正式页及9页原文全文 | inverse-Wishartprocess边际得到TP、covariance uncertainty改变predictivebehavior；不等于任意localStudent-tnoise或矩阵t |
| Dawid 1981，[10.1093/biomet/68.1.265](https://doi.org/10.1093/biomet/68.1.265) | 正式Crossref元数据，未核全文 | matrix-variate理论的primarypointer；本轮不声称读过其中具体公式或其ERP效果 |
| Durbin & Koopman 2012，[state-spacebook](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001) | 正式Crossref元数据，未核全文 | state-space方法书籍来源；本规格ARconditionalSchur公式独立验证，不拿书籍元数据证明本项目性能 |
| Burg官方实现：[spectrum/burg.py](https://github.com/cokelaer/spectrum/blob/master/src/spectrum/burg.py) | 本轮官方repository代码200，forward/backward/PARCOR递归核读 | 作为Burg算法实现对照；项目segmentedmultichannel/objective和stationarystartup仍需本地独立证明 |
| AutoReg官方：[statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html) | 官方全文，hold_back说明已核 | 明确不同lags用同holdback才能同观察样本比较；本项目basis/随机effect/heldsupport仍额外声明 |
| Rivet et al. 2009 xDAWN，[10.1109/TBME.2009.2012869](https://doi.org/10.1109/TBME.2009.2012869) | 已成功取得EuropePMC摘要；MNE官方xdawn.py本轮200；重复摘要请求失败不抹掉已取得text | 空间filter增强evokedresponse、具体BCI先例；不证明Pzprior或每source/reference普遍最优 |
| Sosulski & Tangermann 2022 ToeplitzLDA，[10.1088/1741-2552/ac9c98](https://doi.org/10.1088/1741-2552/ac9c98) | 本轮Crossref正式元数据、作者repository tree及正确main路径toeplitzlda.py；原树existingofficialcode文本 | blockToeplitz协方差strongbaseline、implementation约束；不证明任意baseline/reference之后保持该结构 |
| Bayesian Signal Matching，[10.1080/01621459.2025.2563189](https://doi.org/10.1080/01621459.2025.2563189) | 本轮正式Crossref元数据；原树arXiv:2401.07111既存全文text只读；正式版全文未逐改核 | 从targetcalibration选择性借源的先例；montage/waveform共享仍需observationmodel，不是K0迁移证据 |

graphSPDE直接Crossref请求失败，本轮不以它作为已核primary支持；graphpropercovariance来自明确矩阵定义与support推导。dynamicfactorquery返回的工程力学条目与本任务无关，未进入引用表；T2是标准state-space形状的待验证研究候选，不伪造它已有ERPbenchmark。

研究规格 completed；独立数学 verification completed；官方实现和可取全文有限核验 completed；生产接口及新noise/prior/response实现 not_started；公开四库三seed矩阵 not_started；成人/设备确认 not_started。本轮没有训练、没有改productiondefault、没有给旧结果加新证据标签。
