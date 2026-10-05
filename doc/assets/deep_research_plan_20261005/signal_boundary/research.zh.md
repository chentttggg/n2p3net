# 信号履历、有限算子与边界条件不确定性研究规格

日期：2026-10-05。对应总计划 **B01 信号履历与观测支持、B06 边界条件均值/协方差与完整保真、B10 输入算子与污染分机制对照**；工程约束归入 B14。本文承接愿望清单第一、二组，属于独立研究证据，不替代六份活动指南。

## 1. 本轮裁决与交付范围

1. 首先保存真实观测支持和已经执行的有限处理映射，建立同物理样本的成熟 ERP 锚点。信号履历与可识别性是前置关系，**EA 不是这些路线的通用前置条件**。
2. 三个对象必须分别建模：确定 padding state `a_pad(X)`；不可得外部历史对应的预测不确定性；学习频率/权重/噪声/响应参数的不确定性。不能用一个额外 Γ 混合它们。
3. 优先路线是训练内条件预测差值，再验证完整 `T(X)`；平行保留在分发 raw 坐标中直接边际化邻近 ERP/慢状态的路线。Kalman 只有在实际观测可辨识后才增加，不作为全部实验的基础设施。
4. 输入 ERP 保护、输出校正幅度惩罚和长参照增量保真是不同目标。完整 Pareto 必须包含差值 MSE、独立 ERP 注入、完整白噪声 gain、coverage 和 participant-macro `hit@all`。没有可稳定识别的外部 state 时，保持已声明有限算子输出，并记录“未校正”；不因该研究臂失败删掉该参与者。
5. 旧 0/2/8 秒、penalty 1–10、重叠率 0.5、hit 0.75 等都不能直接变成新默认。距离是连续物理支持量，修正可信度还依赖左右端点、上下文、污染和训练组证据。

本轮完成当前 source 核对、16 个小矩阵/实际算子反例、23 个独立 ERP 形态挑战和一手来源访问账本。**未执行 E1–E3 公共矩阵、未拟合真实新模型、未评分新 test、未实施生产校正器或 Kalman decoder。** 另一个聊天中的边界研究继续独立进行；这里交付可审查的集成规格，不复制其后续实现。

数学执行根为 `C:/Users/chent/.codex/worktrees/deep-research-plan/n2p3-net`，Python 为 `D:/BCI/n2p3-net/.venv/Scripts/python.exe`；脚本把隔离树 `src` 放在 `sys.path[0]`。原 `D:/BCI/n2p3-net` 仅只读核对。

## 2. 当前可确认事实与来源身份

### 2.1 分发 MAT 与 ADC 的层级

官方 BNCI2014-008 description 声明 256 Hz、原生八 EEG 导、右耳参考、左乳突地以及 0.1–30 Hz band-pass。MAT reader 只把 `X` 的微伏转伏、解释事件轨迹；reader 没有实施该上游滤波。本文称该臂为“**官方分发 MAT 连续信号，未追加本项目滤波**”。它既不是未滤 ADC，也不是已知完整有限数字算子的输出。

官方说明没有给出上游滤波是模拟还是数字、设备内还是分发前、FIR/IIR 阶数、相位、启动状态和写出脚本。原论文的 0.1–10 Hz/50 Hz notch/epoch rejection/baseline 是离线分析步骤，不能追补成 MAT `X` 的历史。缺失字段必须保存 `unknown` 和来源范围。没有写出脚本或原始设备参数时，后续文档工作不会恢复这些事实；不做假定性逆滤波或“恢复 ADC”。

### 2.2 隔离 source 中的实际能力

| 文件与 Interface | 当前实际能力 | 不能据此宣称 |
|---|---|---|
| [finite_filter_operator.py](../../../../src/data/finite_filter_operator.py) `FiniteFilterOperator` | 固定 SOS/长度/observed odd reflection；`F X=Q X+B a_pad(X)`；S 个 SOS 时 B 为 `(N,4S)`；可导出有字节预算的 dense F | 初态是外部生理 state；padding 数等于 IIR 有限衰减支持；有限 F 对称/可逆 |
| [preprocess.py](../../../../src/data/preprocess.py) `filter_bidirectional_segment` | 实际双遍、两端 effective states、`pad=min(limit,N−1)`；连续采集段、epoch、resample/baseline 支持 | 比同参数声明即可证明实际算子等价；把整个记录未来都算作在线可用 |
| [noise_model.py](../../../../src/transfer/noise_model.py) `JointNoiseModel` | 从 caller 提供的合法成对 draw 估计联合均值/协方差，保留交叉项；保存 immutable 数值与来源 | 自动辨识真实外部 Γ/Ω；已经有 `condition_on_raw` 的生产 Interface |
| [operator_lda.py](../../../../src/transfer/operator_lda.py) `OperatorAwareLDA` | 传播已冻结 A/B 和联合模型；在其已识别 range 内判别 | 自动构造逐事件真实 A/B；可以同时累计 raw 与滤后独立似然 |
| [harmonic.py](../../../../src/data/harmonic.py) `clean_harmonic_segments` | 各合法 segment/window 独立估频、Huber 重权、低阶 quadrature envelope 和应用开关；只减估计 line | 整个映射是固定线性 F；频率中心是唯一可辨识的物理工频；任何 ERP 都落在受保护 DCT 子空间 |
| [raw_continuous.py](../../../../src/transfer/raw_continuous.py) | MAT 字节绑定，原生坐标；校准下一未知 flash 前截断、尾部右删失；prediction view 与 scoring truth 分开 | 校准完整尾已经观察；后续未授权 target 可以作已知设计；滤后不重叠 selections 相互独立 |
| [boundary_state_research.py](../../../../experiments/boundary_state_research.py) | 共同网格长短诊断、ERP 行基、输出软约束、候选距离 schedule、dense Gaussian gauge、计划生成 | 已经有实用差值预测器；框架中 null 的 ERP/noise 指标已经通过；schedule 已被独立验证 |
| [exact_covariance.py](../../../../src/transfer/exact_covariance.py) | stationary AR Toeplitz covariance 经确定 F 传播；以全空间小 ridge 后 Cholesky 白化 | ridge covariance 与原 PSD covariance 严格相同；null 白化安全；滤后 residual 拟合的 AR 可再次当 raw AR 传播 |

数学 JSON 的 `source_comparison_at_execution` 保存八个路径在隔离树和原树的完整 SHA256。执行时 `exact_covariance.py`、`raw_continuous.py`、`boundary_state_research.py`、`run_e2_prior_scale.py` 已与原树不同；这是并行工作的事实，**原树后续改动不追记为本轮执行 source**。当前谱/null 反例只针对 JSON 中的隔离版本，不能宣称另一个聊天仍然存在同样问题。

旧 Phase 4 数据留在原树。其报告中的 0 秒约 10%、2 秒约 2%–3%、8 秒约 0.4%–0.7% ERP 误差是原字典、原 injection、原校正映射的开发证据。愿望原文另外提及的 AR8 NLL 6.162 与 Phase 4 表内的 2.961 不具有同一配置，不能合成一个现模型读数。修复实施和重新评分必须另绑定实际 source，不改旧账本。

## 3. 数学对象与正确条件化

### 3.1 确定有限映射只传播一次

在固定 SOS、样本数、padding、采样/基线映射及通道参考之后，令

\[
Z=F_NX,\quad F_N=Q_N+B_NL_N,\quad a_{pad}=L_NX.
\]

这里 `L_N` 的单位随 SOS 的状态坐标而定；实际 `B_N a` 才是输出电压。给定完整 X 后，`a_pad` 的条件方差为零；未实现 X 作为随机量时，a 与 X 相关。令 `Σ=Cov(X)`、`Ω=ΣLᵀ`、`Γ=LΣLᵀ`，则

\[
\operatorname{Cov}(Z)=Q\Sigma Q^T+B\Gamma B^T
+Q\Omega B^T+B\Omega^TQ^T=F\Sigma F^T.
\]

删交叉项不成立。在 `FΣFᵀ` 之后再加 `BΓBᵀ`，则对同一确定作用重复计入方差。`Γ−ΩᵀΣ^{-1}Ω=0` 是确定 a 的检查，不等于外部 state 也为零。基线 `M=I−1bᵀ`、空间参考 W、重采样 R 等必须进入最终映射；例如 `MΣMᵀ` 一般不是 Toeplitz，W 非满秩时也不能许诺保留全部空间信息。

若 underlying `X=μ_q+η`，固定滤后候选模型同时使用 `Fμ_q` 和 `FΣFᵀ`。若只观察 Z 且 F 有非平凡 kernel，Gaussian 密度应在实际 range 上定义，或显式声明附加观测噪声。`F^{-1}`、全空间 `logdet(FΣFᵀ)` 不能未经秩检验直接使用。

同一 X 和确定 `Z=FX` 不是两个独立证据源：

\[
p(X,Z\mid q)=p(X\mid q)\,\delta_{Dirac}(Z-FX).
\]

将 raw 与滤后似然相乘会重复观测。多专家的合法概率混合也不等于该相乘模型。

### 3.2 外部 state 的条件均值必须保留候选依赖

令 raw residual η 和外部状态 residual ζ 的联合分布为

\[
\begin{bmatrix}X\\a_{ext}\end{bmatrix}\Bigm|q,C
\sim \mathcal N\left(
\begin{bmatrix}\mu_{x,q}\\\mu_{a,q}\end{bmatrix},
\begin{bmatrix}\Sigma&\Omega\\\Omega^T&\Gamma\end{bmatrix}\right),
\]

其中 C 是合法训练/校准支持，`Ω=Cov(η,ζ)`。在固定参数 Gaussian 条件下，

\[
m_{a\mid x,q}=\mu_{a,q}+\Omega^T\Sigma^{-1}(x-\mu_{x,q}),\qquad
V_{a\mid x,q}=\Gamma-\Omega^T\Sigma^{-1}\Omega.
\]

若 covariance 随候选不同，则 Σ/Ω/Γ 也带 q 下标。代码用解线性方程而非显式求逆；PSD Σ 则先声明可观察 range 与条件分布支持，不能用任意 ridge 假装原联合分布不变。

对于 `Z=AX+B a_ext`，若 X、a 未观测，应传播完整联合 covariance 与完整候选均值：

\[
\mu_{z,q}=A\mu_{x,q}+B\mu_{a,q},\quad
V_z=A\Sigma A^T+B\Gamma B^T+A\Omega B^T+B\Omega^TA^T.
\]

若给定 raw X，则预测均值为 `Ax+B m_{a|x,q}`，条件方差为 `B V_{a|x,q} Bᵀ`（另有独立真实观测噪声时才相加）。只做 Schur covariance 却保留无条件 `μ_a` 会改变 Bayes factor，甚至翻转排序。本轮 scalar 反例：正确 log BF 为 **−0.0294118**，省略候选条件均值后为 **+0.3**。

**条件预测不等于新增已观察证据。** 本项目只观察短 X 而未观察真正长 Z 时，上式用于预测/风险评估，不允许以合成的 Z 再计算第二份 likelihood。route B 在 raw 坐标边际化的 a，必须是进入 raw 观测方程的生理/nuisance 状态，而非凭空追加本项目数字滤波的外部初值。

### 3.3 学习参数后，Schur 补本身也有不确定性

令 θ 包含 ERP 核、协方差、频率、envelope、权重、状态动力学等。完整预测应为

\[
p(Z\mid q,C)=\int p(Z\mid q,\theta,C)p(\theta\mid C)\,d\theta,
\]

\[
\operatorname{Cov}(Z\mid q,C)
=\mathbb E_\theta[\operatorname{Cov}(Z\mid q,\theta,C)]
+\operatorname{Cov}_\theta[\mathbb E(Z\mid q,\theta,C)].
\]

需要给定 X 时使用 `p(θ|X,q,C)` 或明确不更新 θ 的冻结预测协议；不能既从 X 重估参数，又把其当无条件固定。已实现的 `θ_hat(X,C)` 在给定 X/C/seed 后也是确定函数：其数值本身不是一份独立随机量。这里 uncertainty 对应未知物理/模型参数或跨拟合数据的误差，不允许把它重复加到已经观察的确定 `T(X)`。plug-in Schur 是候选近似，不是 posterior covariance 的自动完整实现。本轮已知 x=2、`Var(θ)=0.25` 的标量映射 `Z=θx` 条件预测方差为 1，固定均值 θ 的 plug-in 方差为 0。

`JointNoiseModel.fit` 当前只核 caller 提供的成对 draw，不判断 draw 是否为真实噪声或能否代表 held state。必须在 C 内构造伪截断，留出完整 groups，反复拟合 θ 的 uncertainty 用 group/block bootstrap 或明确后验表达；对每 sample 乱洗牌会破坏时间相关，不能估计所需 uncertainty。

## 4. 候选机制与选择理由

| 路线 | 数学/统计目标 | 可接受的训练内实现 | 第一否证条件 | 优先级 |
|---|---|---|---|---|
| A0 保留有限 F | 精确已执行观测变换 | 原有限算子和成熟 ERP 均使用冻结输入身份 | 对照也没有合法真实支持/保存重放 | P0，所有后续 anchor |
| A1 预测外部差值后受约束修正 | 条件 MSE 与完整 ERP/gain 权衡 | C 内伪截断得到 `δ`，从短 X/合法 public schedule 预测 `E(δ|X,C)`；按 held groups 估计残余 covariance | 同 short 异 external 的 irreducible residual 未入账；或依赖 held 长参照推理 | P1 |
| A2 对差值 residual 边际化 | 将未知长参照作为不确定预测，不强行替换波形 | 在已识别的 B range 内估计条件均值/covariance；输出诊断或特定 generative model | 把随机校正输出再作为独立 q 证据；重复 Γ | P1，与 A1 共用辨识证据 |
| B1 raw 连续邻近响应边际化 | 在真实 raw 观测方程处理邻近未知 selection | `X=D_qβ+H_s a+η`，sum/integrate 未知 s/a/β；共同事件历史支持 | 把未知邻居当 target/non-target 已知；或直接遗漏其均值而只加白噪声 | P1，允许与 A1 平行 |
| B2 联合 ERP/慢状态 Kalman | 动态状态预测及观测 likelihood | 固定状态维数、训练内 Q/R，缺样由观测矩阵处理；filtering 与 smoothing 分别声明 | observability/实测条件数、Q/R与 ERP/漂移分离未辨识 | P2；先 small-matrix simulation |
| C1 固定设计 harmonic | 同一固定线性清理映射的可分析 baseline | 频率、权重、window/schedule 来自合法训练内冻结；只对新 X 线性估振幅 | 工频漂移覆盖不足；保护只声明有限子空间 | P1，B10 对照 |
| C2 自适应 harmonic | 对漂移/调制 line 的 segment-local 处理 | 实际重估 frequency/Huber/envelope/application；完整复跑注入与白噪声挑战 | 仍用一次输出的 H 矩阵宣称无条件 `HΣHᵀ`；切换点损伤 | P1，独立 numerical identity |

高算力只缩短计算时间，不能让同一组数据产生更多独立外部信息；也不能自动使 Q/R、未知 reference 或欠缺上游滤波参数可识别。

### 4.1 长短输出共同网格与可识别范围

令实际已观察长段 `X_M`，C 为短段选择，`R_l/R_s` 选择共同输出网格，则

\[
\delta=(R_lF_M-R_sF_N C)X_M.
\]

长短 F 不能直接相减。X_M 仅在真实已授权、同采集段、同独立 C group 内使用；长参照一般仍是有限算子，命名“长上下文诊断参照”，不称无限长真值。

伪截断 δ 可用 B_N 做 state-range 投影；分别保存 residual `r=δ−B a_hat` 与 numerical rank/condition。如果某方向不被实际 B 或模型表示，route A 不校正该方向；不把“该参与者拒判”当成去除分母的理由。用 SVD 输出方向作为参数化时要固定 gauge；不同 `B→BM`、`a→M^{-1}a` 只有同步 mean/covariance 和 prior 才等价。

边界距离定义为**整个使用输出支持**到左右短段端点的距离，而非只算刺激 onset；分别存 `d_left/d_right`、端点原因、可得 context 长度、事件密度和通道质量。padding limit 不是 IIR 衰减时间。低频 IIR 的有效恢复长度还需声明误差单位、输入/state envelope、容差和最终输出映射。

### 4.2 保护目标与完整 T 的验收

固定样本优先展平；校准 ERP 模板正交行基 U，`P=UᵀU`。

输出 penalty：

\[
u_\lambda=\arg\min_u\|u-\hat\delta\|^2+\lambda\|Uu\|^2
=\hat\delta-\frac{\lambda}{1+\lambda}P\hat\delta.
\]

它只限制当前校正输出在 ERP 方向的幅度。输入保护 `K U_inputᵀ=0` 则限制线性校正映射对被保护输入增量的作用。两者互不等价；保护一个有限字典不等于保护所有 amplitude/latency/width/topography 或相邻 flash 组合。

设实际推理 `T(X)=R_sF_NX+g(X)`。每次冻结/自适应算法均报告

\[
E_{short}(h;X)=\frac{\|T(X+h)-T(X)-R_sF_Nh\|}{\|R_sF_Nh\|}.
\]

另以合法长参照 `R_lF_M h_full` 定义 `E_long`；h_short 必须为同一个 synthetic h_full 的裁片，不能把边界外响应零补为已观察 raw。**E_short、E_long 是两个不同判据**：修正有限 F 的 ERP 衰减可能让 E_short 非零而使 E_long 更好。

已知 ERP 注入必须独立于构造 U 的 calibration 模板。挑战包括早期/晚期 ERP、幅度正负、潜伏期扰动、宽度、导数、多 topography、端点内外 onset、邻近 flash、glitch/EOG/参考污染，以及所有拒绝/开关/schedule 切换点。波形 MSE之外保存峰幅、电压偏置、最大逐点误差、潜伏期、积分面积和频谱。合成挑战保证形态敏感性，不证明真实成人 response 已保真。

当 `T(X)=GX` 线性时，完整白输入噪声指标为

\[
G_{mean}=\frac{\operatorname{tr}(GG^T)}{\operatorname{tr}(F_sF_s^T)},\qquad
G_{worst}=\|G\|_2^2,
\]

其中前者相对原有限输出，后者为绝对输入最坏方向放大，不混称一个数。非线性时使用 `T(X+ε)−T(X)`，对每个 X 和多 σ 重新运行整段 frequency/weights/application，报告平均/尾分位/最坏局部倍率；冻结 adaptive 参数所得 Jacobian 只能叫条件诊断。修正值 δ 的能量不等于完整 G 的噪声倍率。

距离 schedule 可比较固定 λ、连续 monotone λ(d) 和训练内 learned reliability。0/2/8 秒只留作旧协议桥接与报告分层；新的 `d_pivot/τ/λ_max` 从 inner groups 选择并冻结，或全网格预声明后不选 winner。若近端强保护，则 penalty 较大、ERP 幅度约束半径较小；不同时声称半径也随距离递减。保留不单调实际污染案例，不以距离远替代 signal-quality 证据。

### 4.3 工频和污染需要各自观测模型

持续工频、频率漂移、幅相调制、突发 glitch、慢漂移分别建立 challenge axes；每轴固定同 signal、事件、labels、mask、raw 范围与预算。频带/notch/phase 面板必须让滤波幅度响应与噪声/ERP均值同步传播，不能把双向平方响应当单向同滤波。

`clean_harmonic_segments` 即使某次拟合后可写条件线性矩阵，整体 `H_{θ(X)}X` 也非固定线性。frequency/envelope 有 trade-off，其中心频率不能单独解释为物理设备估计。本轮真实函数两条单 line 49.6/50.4 Hz 的叠加反例，相对非线性残差为 **0.634553**。与固定 A/B 传播组合前，必须选 C1 固定参数臂或 C2 完整非线性不确定性臂，不能默默串接。

空间观测写作 `Y_obs=M W G X_phys+H_eog a_eog+ε`：M 为实测 channel availability，W 为真实参考，G 为 gain；这些未知量分别声明。reference 污染可同时影响所有差分导，单 HEOG 不能表达完整眼动；缺导不作为已观测零电压。必要时保留 native head/原参考对照；插值作为有模型 uncertainty 的估计，不取得实测身份。

## 5. 邻近未知 selection 与时钟/删失

公开 flash onsets 与 membership 可以使用；邻近 selection 的 target 身份未授权时必须未知。当前 q 与邻居 s 的 joint likelihood 可写为

\[
p(Y\mid q,C)=\sum_s \pi(s\mid q,C)\int p(Y\mid q,s,\beta,a,\theta)
\,p(\beta,a,\theta\mid C)\,d\beta\,da\,d\theta.
\]

邻居独立 uniform prior 是预声明对照，不默认所有范式独立；若有重复目标/序列依赖需在训练范围辨识。用 `logsumexp` 聚合，不能以 `max_s` 代替边际化而不更改方法身份。较多未知 selections 可通过有限支持图/状态模型/可控 approximation 处理，保存截断误差和 compute latency；无限算力不授权读隐藏 target。

条件 AR 的 prefix 也必须更新共享 nuisance/θ 的 posterior。即使 prefix 没有直接 candidate ERP，先积分未知参数后它不必成为可取消的公共常数。固定 hold-back 令不同 AR order 使用相同评分 rows，但 prefix 观测、读取和等待仍计入成本。

四种支持分别保存：真正写入 sample clock 的采集支持；软件/硬件/光学事件时间；已授权标签/事件支持；数值算法使用和可用时间。缺硬件/光学证据不能把软件 onset 当已认证真实 stimulus onset；在声明误差范围内做 event jitter panel，不能从 test score 反推最优时钟。

最后授权 flash 后含下一未授权 flash 的 raw tail，不能用“仅取 ERP 尾”理由读入并只构建已知 events。现协议在下一 onset 前右删失，保留全部 K labels 和观察长度；一般左/右删失使用观测选择矩阵，只对真实已观察行做 likelihood，不用零填充当电压。非因果 F 会将窗口外/未来样本传入，censor 与 F 次序因此属于模型定义；每个 selection 独立 F 与整段 F 有不同身份、可用时间与 covariance，不可互换。

## 6. Kalman 分支的最小证明义务

先用 simulator 写明确状态：`z_t=[ERP amplitude/latency approximation, slow drift, noise state]`，已知 stimulation input `u_t`，

\[
z_{t+1}=A z_t+G u_t+w_t,\quad Y_t=H_tz_t+v_t.
\]

外部数字滤波 state 与真实生理 state 的状态坐标分别标注。固定 state 维数；距离改变 penalty/reliability，不随意改变 Kalman 向量长度。缺导/缺样对应 H_t 与观测缺失，未授权 target 仍是 latent input。latency 非线性先比较独立有限基/导数近似，不能默认 linear Kalman 能准确表达任意时移。

最小退出证据：实际可得 stimulus schedule 下 observability matrix/Gramian 的秩与 spectrum；ERP 与 slow drift 的混淆方向；Q/R/AR scale 可识别区间；训练独立 groups 数量；simulation coverage 与 posterior 校准；对实际 missing/clock scenarios 的精确 filtering/smoothing available time。`dim(z)<dim(Y)` 不是充分条件；本轮两个状态共用 H=[1,1]、A=I 的 observability rank 只有 1。A 的两个 pole 分开但只差 1e−5 时虽满秩，条件数为 **362001.8**。

filtering 只用截至当前可得观测；RTS smoothing 使用未来，按未来最后 sample/交付时间报告，不把后续独立 T 当 C 的 future context。若 Q/R 在可用训练中不能区分，保留 ridge/固定 noise family 对照，不输出“已识别 state uncertainty”。联合状态模型的优先级为 P2，不拖延 B01/B06/B10 的 P0/P1 工作。

## 7. 每个愿望的实验、验收与依赖

这里“验收”首先是可证伪的工程/数学退出条件；阈值效能是否成立仍需要后续真实配对验证。G0=同物理支持/授权/源绑定；G1=完整 T 的独立注入与 gain；G2=训练内选择/参数冻结；G3=requested outcomes/participant 配对与独立公共复核。四周可以并行，但不能把 G0/G1/G2 跳成第4周才补。

| 原愿望 | 分块实验 | 验收/否证与交付 | 优先依赖 |
|---|---|---|---|
| I.01 完整上游 signal history | B01-S1 官方文档、设备参数、分发脚本/字节、reader 执行分层账本 | 已知字段有源/hash；缺失项 explicit unknown；不补历史 | P0，无 EA 依赖 |
| I.02 MAT/项目滤波/ADC 三层 | B01-S2 带单位和 processing-stage 的 input identity | 本例可区分 official MAT 与新增 F；任何 ADC 用语需真实证据 | P0，S1 |
| I.03 每记录实际 F/padding/state/range | B01-S3 小矩阵 dense-reference + 长记录 matrix-free receipt | actual apply/SciPy/Q+B L 一致；长度、grid、SOS、edge、crop 和 state role 全绑定 | P0，S2/G0 |
| I.04 causal/zero/selection 独立三协议 | B10-S4 三臂同 raw 和 labels 面板 | 数值身份分开、完整可用时间、chunk state 及非因果支持正确 | P1，G0 |
| I.05 频带/notch/lowpass/phase | B10-S5 预声明 factorial panel | 同物理样本比较；mean/cov 同步变换，G1，报告 latency | P1，S3/G0 |
| I.06 line/频漂/调制/glitch/慢漂分治 | B10-S6 分污染 simulator + C 内真实诊断 | 各轴的 suppression、ERP/gain、failure 分开；glitch 不被悄然 inpaint | P1，S3/G1 |
| I.07 signal fidelity gate | B06-S7 独立 morphology/topography/event panel | 完整 T 两类参考、峰/潜伏/area/spectrum、worst/coverage，全保存 | P0，可与 S1 平行 |
| I.08 按距离修正强度 | B06-S8 fixed/continuous/reliability schedule | 0/2/8只桥接；inner group 冻结 schedule；挑战切换点与非单调污染 | P1，S7/G2 |
| I.09 ERP软约束+noise Pareto | B06-S9 input-vs-output protection 全网格 | MSE/G1/coverage/hit 全 Pareto；不从 held 拣 λ，不把 overlap 当损伤 | P1，S7/S8 |
| I.10 未知相邻 selection 边际化 | B06-S10 sum/integrate neighbor + omission/max 负对照 | public events使用而 hidden targets不进入；sum prior/clock/censor一致 | P1，G0/G2 |
| I.11 selection F vs continuous F | B10-S11 两套 finite-observation identity | 同网格差值/均值/covariance/可用时间分开；“不重叠”不宣称独立 | P1，S3/G0 |
| I.12 adaptive frequency/weights uncertainty | B10-S12 fixed-θ vs refit-pipeline bootstrap | law-of-total-covariance；整 pipeline 重跑 h/ε；不得只保存一次 conditional H | P1，S6/G1/G2 |
| I.13 坏导/ref/EOG/missing空间观测 | B01-S13 显式 MWG + native/reference挑战 | observed mask与单位/roles绑定；缺导0只是占位，参考依赖正确传播 | P0契约/P1实证，G0 |
| II.01 三种 state 对象 | B06-B1 state-role 枚举及负对照 | 确定padding无独立Γ；external和θ uncertainty不冒用同一字段 | P0，S3 |
| II.02 完整 Σ/Γ/Ω | B06-B2 C组 paired residual/state + gauge probes | jointly PSD、有paired draw证据、交叉项推送与 direct FΣFᵀ一致 | P1，B1/G0 |
| II.03 Schur 条件化 | B06-B3 raw-dependent mean/cov + q flip challenge | conditional mean必须含x−μq；确定Lx Schur=0；singular支持声明 | P0数学/P1估计，B2 |
| II.04 ERP/state同Kalman | B06-B4 最小 simulator 后 fitted-state臂 | observability、condition、Q/R辨识、时间许可均过；否则停留候选 | P2，B2/B3/S10 |
| II.05 差值条件mean/cov/保护方向 | B06-B5 cross-fitted pseudo-truncation | held长段只diagnostic；short predictor冻结；unexplained covariance不丢失 | P1，B2/B3/G2 |
| II.06 距离/context/density/quality可靠度 | B06-B6 reliability against held groups | 保存左右原因/支持；variance覆盖与reliability校准；不读held选阈值 | P1，B5/S7 |
| II.07 不可表示方向不修正 | B06-B7 range residual与不可识别fixture | δ−Ba residual入账；未修正回原F，requested participant分母不变 | P0数学/P1路径，B2/B5 |
| II.08 多目标MSE/ERP/noise/hit | B06-B8 同group外折全目标图 | 产品hit为主、概率风险为辅；无matched锚点不能称收益；全Pareto留存 | P1，S9/G3 |
| II.09 两路线差值校正/raw边际 | B06-B9 matched A1/B1 协议桥接 | 相同 raw/targets/K/test队列；不同数值/估计量身份；不重复p(X)p(FX) | P1，B5/S10/G0 |
| II.10 不同ERP子空间 | B06-B10 fixed-P300/early/slow/latency/neighbor辞典 | 独立challenge不复用训练字典；自由度/fit groups计账；不全信息守恒 | P1，S7/G2 |
| II.11 KU保护软化/分段/距离 | B06-B11 input-hard/input-soft/output-soft/schedule | 每种公式有独立reference；complete-map derivative与switch challenge | P1，B10/S8/S9 |
| II.12 残差/condition/白噪声记录 | B06-B12 complete-map numeric receipt | constraint residual、state表示误差、condition/rank、mean/worst gain全部有值 | P0接口/P1跑数，S3/S7 |

四周建议：第1周完成 B01/G0 与 G1 基架，同时成熟专家整合；第2周 A1 与固定/自适应工频研究并行；第3周 B1 邻居/删失与 reliability/ERP 字典并行，Kalman只做已可辨识的 simulator；第4周仅让通过门的固定路线进独立公共配对矩阵，三个 source→target seeds 与完整 requested outcomes 由总计划管理。某候选失败照样交付反例/Pareto，不能把它作为停掉整个月其他独立工作的理由。

## 8. 真实文件与调用链的后续改造清单

本轮没有修改下表生产文件。已有 owner 继续承接职责，新增 Interface 要消掉 caller 的重复知识，不堆 pass-through。

| owner 与真实调用点 | 具体实施增量 | 联动验证 |
|---|---|---|
| `data.bnci2014_008_candidate` → `transfer.raw_continuous` / source builders | history 分层、unknown 上游参数、事件/单位/参考/采集支持；不追补 ADC | MAT field/units/hash/授权负对照与原生8导身份 |
| `data.preprocess` → `execute_preprocessing` / BrainSync/MOABB raw builders | 返回一次验证后的已观察记录、实际有限算子 receipt、source/output网格与可用时间 | raw scope、closed epoch/FFT/baseline、pause/clock、causal chunk、selection split |
| `data.finite_filter_operator` → `exact_covariance` / 研究 probes | 保留确定state；matrix-free composition/covariance推送与bounded dense oracle；F/B/L state-role | 本轮交叉项、SciPy padding、crop非交换、长度/gauge/秩反例 |
| `transfer.noise_model` → `operator_lda` / `run_operator_decision_study` | 新条件预测 state artifact 接收真实C paired draws；返回条件mean、cov、range/未识别方向及fit rows；不是新增oracle状态入口 | q-conditioned mean、deterministicSchur、law-of-total-cov、save/load关系重放 |
| `transfer.exact_covariance` → 后续研究caller | 区分 fitted-before-F 与 fitted-after-F AR；exact PSD range likelihood 与显式 observation-noise/ridge两臂；避免null伪精度 | raw模型来源、均值同步F、rank、null perturbation、ridge敏感性、likelihood reference |
| `data.harmonic` → `run_operator_decision_study._harmonic` | protocol区分固定θ和逐segment自适应；保存实际窗口/估频/重权/开关与fit scope；若组合只执行已声明完整pipeline | 本轮真实叠加、segment隔离、δ/h/ε整pipeline、边界开关、污染分轴 |
| `transfer.continuous_decision` / `temporal_noise` → `raw_continuous` / `run_bnci_continuous_study` | 完整public event history；邻居latent target边际化、censor observation map、prefix posterior语义、frozen replay | neighbor omit/max负对照、未知label不可见、event置换、prefix/censor支持、availability |
| `experiments.boundary_state_research` → 新 E1/E3 research runner | 框架diagnostic与真实predictor分开；完整T的reference/ERP/gain不再null；读取获准真实C bytes而非旧结果猜状态 | inner group选择、同short异external、long/short共同grid、23独立形态挑战扩展 |
| `research.comparison` / `outcomes` / `executor` → 新实验manifest | 同物理证据、算子contrast、fit/score数值身份和完整requested ledger；参数选择scope与actualsource | paired失败、无输出NLL分母、coverage、source snapshot、保存模型重放 |
| 六活动指南（主代理集中维护） | B01证据边界进输入合同；B06路线进研究规范；实际实施状态进roadmap；deep Module owner进technical；验证/吞吐进engineering | 不另创竞争guide；证据索引和历史归档规则保持 |

BrainSync 采集端属于单独本地项目。新 device/光学时钟或 channel-observation 协议需要它的设备/recording/session owner 接入；后端本文没有修改前端，也不能仅靠后端字段证明新采集已经执行。

## 9. B14：减少重复防御而保留科学约束

建议 Interface：**record ingress 验证一次 → immutable 已观察输入/冻结 state →纯内部数学→ artifact save/load 边界核验**。caller 不逐层拼回相同 shape/单位/identity/fit scope；从同一验证结果取得 contract 和数值 receipt。错误在 owning ingress 给出一个清晰异常，不层层 catch 后降级成空结果。

| 检查/防御 | 裁决 | 具体证据或前提 |
|---|---|---|
| 原字节/observed support/participant/channel/reference/labels授权/clock/censor | 必须保留 | 它們定义科学对象；缺失会造成身份错配、future读取或伪样本 |
| fit/score identity、完整requested分母、source绑定、load字节/schema/digest | 必须保留在 owning Interface | 自述metadata不能替代数值关系；外部文件会变化/损坏 |
| covariance PSD/range、stable SOS/AR、finite数值、实际mask依赖 | 必须在构造/fit/load及新输入ingress验证；内部复用验证结果 | 这是模型可定义性，不是无意义防御；不能clip pole/rank或自动换协议 |
| 同一 trusted X 在 `decompose→zero_state→_values` 等内部重复全扫描 | 可收敛到 owning调用一次，后续纯内部方法消费已验证X | 实际重复调用存在；实施前保持所有public入口负对照与等价数值；此规划未删除检查 |
| immutable已拟合 state 每次候选/每次内部projection重算完整SHA/PSD | 满足deep immutability和one-owner后可减；fit/save/load/外部交付仍核 | `JointNoiseModel` byte-backed arrays+frozen metadata具备基础；需实际caller测量与不变性证明后删除内部重复；不是把所有assert_integrity删除 |
| `ExactPropagatedCovariance.provenance` 的运行时 integrity | **目前不能删** | 隔离版本provenance为可变dict；不能因dataclass frozen推断deep immutable |
| 给unknown状态默认0、unknown reference别名、fit失败后clip/ridge/conditional自动换臂、except→uniform概率 | 应从新Interface设计排除 | 本轮同short异external、rank/null、neighbor反例已证明会改变估计量或观测对象 |
| 重叠率>0.5/近边界固定2s/固定PTP门 | 不新建这种硬守卫 | 它们没有对应独立可验证效能资格；重叠不能证明损伤，距离不认证输入质量 |

没有证据的现有防御不做“一刀切删除”。先分别量化copy/scan/hash/fit/score开销，再把重复复杂度收回已有Module的Implementation；低算力复杂度不是目标，研究吞吐与科学意义同时保留。长N不保存dense F作为常规artifact，保存可执行operator与字节receipt，小N dense用于oracle；减少内存副本并为具有相同有效input/operator身份的工作复用缓存。

本分支只写 `signal_boundary/`，不commit、不修改原树或他人文件。产物使用 LF；主代理统一 Ruff、所需合同检查、`git diff --check`、source manifest与提交。无public/cloud训练时不用假冒CPU/GPU吞吐成绩；未来云端执行依总计划实际资源floor与25 CPU/4 GPU workers约束。

## 10. 本轮可重复数学证据

[verify_math.py](verify_math.py) 与 [math_results.json](math_results.json) 保存实际执行脚本/模块、source SHA、seed和环境。16项全部通过；其中23个 ERP形态注入来自与保护模板不同的center/width/topography，以及一阶/二阶导数和相邻响应组合。

| 检验 | 本轮合成/算子结果 | 所否定的推论 |
|---|---|---|
| actual F=Q+BL、SciPy同padding | 分解误差2.82e−13；SciPy最大差0 | padding仅是名义参数，无需验actual执行 |
| covariance交叉项 | 完整推送相对误差6.28e−14；删cross相对误差0.381086；重复Γ为0.622829 | 把a_pad作为独立state variance |
| deterministic Schur | norm2.37e−15 | 观测已定后仍额外加padding方差 |
| 同short异external | short逐位相同，δ差norm0.0241628 | 仅short可精确恢复所有long边界差值 |
| conditional mean | 正确BF−0.0294118；错mean后+0.3 | Schur covariance正确就足够 |
| learnedθ | predictive variance理论1，MC0.998215，plugin0 | 给定X后所有估参不确定性消失 |
| output ERP overlap | 100% overlap但long-ERP误差0；另例0% overlap却完全删除ERP增量 | overlap比例直接证明损伤/安全 |
| 完整white gain | 相同校正能量0.81，完整gain分别0.01和3.61 | δ能量等同noise放大 |
| 自适应switch | perturbation2e−6，增量偏差1 | 冻结一次K可保护跨开关增量 |
| 实际harmonic | superposition relative error0.634553 | 自适应回归仍是一份固定F |
| rank保护 | protected h误差0，但另candidate方向d′²由4→0 | 保住ERP字典就保住全部信息 |
| Kalman | rank1不可观察；近pole满rank条件数362001.8 | 低维或满rank就能辨识真实state |
| neighbor | 正确边际BF1.39e−11；遗漏neighbor后25 | 邻居无已知target就可以当不存在 |
| prebaseline covariance | rank19/20，diagonal spread1.36006 | baseline后仍自动Toeplitz |
| actual global ridge/null | 72维中rank70；unit null whitening norm287274.46 | ridge使exact PSD无条件信息守恒/安全 |

23形态完整mapping面板中，原有限F的E_short最坏2.43e−14；**专门构造的**state-projection/input-hard/output-soft fixtures最坏误差分别0.29167/0.34247/0.28539，相对白噪声平均gain约0.80844/0.81159/0.83130。它们是反方数学fixture，**不是经过训练的候选方法，也不是Phase4复跑，更不构成“soft比hard真实更优”的结论**。它们展示有限保护字典不能担保未知形态，噪声变小也不能担保ERP增量。

```powershell
& D:\BCI\n2p3-net\.venv\Scripts\python.exe -X utf8 `
  doc/assets/deep_research_plan_20261005/signal_boundary/verify_math.py
& D:\BCI\n2p3-net\.venv\Scripts\python.exe -m ruff check `
  doc/assets/deep_research_plan_20261005/signal_boundary/*.py
git diff --check
```

这些探针证明特定错误推论可被反例推翻；不能证明 A1/B1 的真实 Γ/Ω 可识别、不能证明真实ERP保真或命中率提升。后续独立公共矩阵需按G0–G3和总计划继续。

## 11. 一手资料与访问级别

[sources.json](sources.json) 保留每个实际query、requested/final URL、访问时间、状态、字节hash和access level；[retrieve_sources.py](retrieve_sources.py) 是有界查询入口。[cached_primary_sources.json](cached_primary_sources.json) 与 [inspect_cached_primary.py](inspect_cached_primary.py) 保存一手缓存字节核验、原下载访问记录和本轮精确文本点位。不是系统综述，不把元数据命中写成全文已读。

| 来源 | 当前实际访问 | 对本规格的用途与边界 |
|---|---|---|
| [SciPy sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)、[filtfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html) | 本轮官方HTML成功读取 | padtype/padlen、双遍及边缘处理；实际矩阵与SciPy独立数值核对，不是效能资料 |
| [BNCI官方description](https://bnci-horizon-2020.eu/database/data-sets/008-2014/description.pdf) | 本轮远端失败；原树2026-10-04一手PDF/访问账本可读 | 上游band-pass、设备、参考、采样率的明确范围；原版本缺失实施细节不追补 |
| Särkkä 2013 [作者全文](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf) | 原树缓存一手全文，原下载2026-10-04，本轮SHA逐位核原账本 | Appendix A.1，印刷pp.209–210/PDF pp.229–230，Lemma A.1/A.2及式A.5直接给出joint/conditional mean与Schur covariance；不证明EEG真实Gaussian或Γ可识别 |
| Gustafsson 1996 [10.1109/78.492552](https://doi.org/10.1109/78.492552) | 本轮Crossref标题/DOI已核，论文全文未取到 | 确认文献身份；实施事实使用SciPy官方实现，不编造原论文结论 |
| Widmann et al. [10.1016/j.jneumeth.2014.08.002](https://doi.org/10.1016/j.jneumeth.2014.08.002) | 本轮Crossref exact DOI元数据成功，未核全文 | 仅登记方法资料候选；本项目ERP损伤裁决由独立数值反例而非仅题名支撑 |
| Ehinger & Dimigen 2019 [Unfold, 10.7717/peerj.7838](https://doi.org/10.7717/peerj.7838) | 原树一手full XML与作者simulation可读；本轮作者文档网站最终成功，publisher XML远端失败 | XML DOI与段落点位已核；作者代码L55先`pop_eegfiltnew`，L58–59用−0.8..1s完整FIR lag。此例不自动等价本项目有限8维因果核，需实际F/字典对照 |
| `Paper/Bayesian_Signal_Matching_ERP_Transfer_arXiv_2401.07111.pdf` | 本轮本地全文解析/hash，页内mean/cov来源记录 | 可供灵活ERP核/均值先验对照；其数据/模型不证明本项目数字边界可辨识 |
| `Paper/Block_Toeplitz_LDA_ERP_JNE_2022.pdf`、`Paper/xDAWN_Riemannian_Transfer_P300_2020.pdf` | 本轮本地全文解析/hash | 成熟ERP空间/协方差锚点机制；classifier系数和spatial filter不能直接当信号保护字典 |

Crossref宽标题查询可能命中无关资料或限流，**不作为机制论据**；最终账本只据实际保存的标题/DOI记录登记文献身份，429/404/证书错误保留为失败，不作未见文献不存在的证据。本地一手副本以原下载时间/最终URL/hash交代 provenance，不能冒称本轮远端访问成功。Git分发账本保存URL、文件hash、页/段落定位和短excerpt，原始PDF/XML/HTML缓存留在原树，不复制到本分支。

## 12. 未完成实验与交接

- G0 input history/support与完整operator receipt的全库落地，真实paired残余/state构造与condition辨识，均待实施；本轮未对任何新real test拟合/评分。
- E1/E3真实C group伪截断、完整T的独立ERP/white panel、inner schedule/字典选择和actualpredictor冻结仍待跑。旧Phase4不可被这16个小矩阵替代。
- neighbor/censor/prefix posterior以及Kalman simulation需先完成time/support合同；任何新state方法都保留A0、固定/自适应harmonic、成熟ERP锚点与同输入scratch/神经专家关系。
- 通过G0/G1/G2才进独立公共participant/四库三seed，hit@all、coverage、paired failure、概率风险和负迁移全报告；成人设备/光学时钟确认另用新数据，不能由BNCI八名ALS或合成结果推出。
- B14工程删减先保证内部immutable与外部load关系核验；未证明的守卫不删，不向新Interface增加数据驱动硬门。主代理集中回写六指南、证据索引、plan/coverage和最后整洁提交。
