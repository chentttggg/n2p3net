# 算子科研：有限支持、条件噪声与连续候选似然的一手依据

日期：2026-10-04。范围：为本仓库 R1/R2 补充可实施的成熟数学和竞争解释，非系统综述、非新性能结果。

本次最大的推进不是再找一个滤波器名字，而是分清三种不同对象：**确定的有限输入算子、对未观测外侧过程的概率模型，以及用于分类的近似似然**。三者混在一起，会把均值失配误称为噪声，把确定的 padding 再加一次随机方差，或把有限样本正则收益误称为滤波增加信息。

## 1. 已核一手来源与边界

原始请求 URL、最终 URL、UTC 时间、HTTP 状态、字节数与 SHA256 在 `access_manifest.json`、`access_additional.json`、`access_extra.json`、`access_spm.json`。PDF 为作者或会议提供的公开版本；文本均为同文件提取，标有 PDF 页码。源代码固定到版本/commit。未进入下表的检索结果不作为论据。

| 来源 | 实读定位与支持 | 限度 |
|---|---|---|
| Särkkä, *Bayesian Filtering and Smoothing*, 2013，[作者全文](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf) | Appendix A.1, Lemma A.1/A.2，印刷 pp.209–210，PDF pp.229–230：线性高斯联合分布、条件均值和 Schur 补协方差；Chapter 4：Kalman prediction/update | 标准概率数学；不保证 EEG stationary Gaussian，也不给本项目真实 Γ |
| Rasmussen & Williams, *Gaussian Processes for Machine Learning*, 2006，[Chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf) | §2.1，Eq.2.5–2.9：核/权重后验及对权重积分的预测分布；§2.2：Gaussian conditioning | 核均值参数不确定性与每次生理响应变化不是同一层 |
| Minka, *Bayesian linear regression*, 1998/revised 2010，[作者全文](https://tminka.github.io/papers/minka-linear.pdf) | §5，PDF pp.8–9，Eq.47–62：未知输出协方差的 inverse-Wishart 后验、matrix-T evidence/predictive；Eq.62 为多个新样本的联合预测 | 使用条件独立 Gaussian residual；不能在未正确白化的 raw time samples 上直接套样本数；不是独立每样本 t 的乘积 |
| Shah, Wilson & Ghahramani, AISTATS 2014，[会议全文](https://proceedings.mlr.press/v33/shah14.pdf) | *Student-t Processes as Alternatives to Gaussian Processes*，层次 covariance 模型及 Student-t predictive；支持重尾来自明确积分层级 | 不等于任意 t score 自动校准；该论文不是 P300 性能依据 |
| Guo et al., ICML 2017，[会议全文](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) | §4 开头：需要 hold-out validation；§4.2，Eq.9：T>0 温度缩放，用 validation NLL 拟合且不改变 argmax | 成熟校准对照，只可能改善概率/选择性决策，不提升固定 argmax hit；本项目不能用已看过的 T 拟温度 |
| [SPM 官方实现](https://github.com/spm/spm/blob/9b8590ecaacdcee15ffb03eb084e6ce3a7a01048/spm_spm.m) | `spm_spm.m` L106–109、438、445、589：同一个 KW 用于 X、Y，且传播 KWVW′K′；`spm_filter.m` L21–22、70/82 为低频残差形成投影 | 成熟 neuroimaging GLM 工程先例；其 fMRI 噪声模型不是 EEG 的现成估计器 |
| Ehinger & Dimigen 2019, Unfold，[10.7717/peerj.7838](https://doi.org/10.7717/peerj.7838) | 前轮已核同一全文 `../../system_research_20261004/literature/unfold_full.xml`。本轮重点重读 Fig.6、Basic settings §Epoch size、time-frequency discussion；作者固定代码 `unfold_filter_simulation.m`：先滤 EEG，再用 −0.8..1 s 全 FIR lag | **作者示例没有把同一 F 直接作用在设计上。** 它拟合滤后的响应；全文要求覆盖完整响应并提示基函数/边界失真。不能作为本项目 0..1.2 s 八维因果基等价性的依据 |
| Crosse et al. 2016, mTRF，[10.3389/fnhum.2016.00604](https://doi.org/10.3389/fnhum.2016.00604)，PMC5127806 | 新核全文：Regularized linear regression、TRF optimization、worked example；作者代码 `mtrf_train.m` L115–164：明确 lag、Cxx/Cxy 与 ridge。正文案例滤 EEG 后用 −150..450 ms lag，并留50 ms边界空间 | 刺激自相关校正不等于观测 residual 的预白化；不支持声称其已经实现本项目 Γ/Ω，正文也没有该项目 Bayesian candidate likelihood |
| [SciPy v1.16.2 作者实现](https://github.com/scipy/scipy/blob/v1.16.2/scipy/signal/_signaltools.py) 及 [filtfilt 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html) | `sosfiltfilt` L5031–5143，尤其 zi×端点、正反滤波；`_filtfilt_gust` 明确初态线性求解 | 固定 SOS/长度/padding 为确定线性映射；不能据此把初态误称随机变量。Gustafsson1996本次仍未取到论文全文，仅用官方实现陈述实现事实 |

rERP II（10.1111/psyp.12320）元数据核实为 PMC5306445；全文 API 返回500，PMC网页虽HTTP200但实际是 reCAPTCHA。**没有将HTTP200计作全文可读，也未基于该全文作新主张。** Parallel CLI 可见，但未建立已登录可用的文献服务；本次使用公开作者源/出版社源/Europe PMC/GitHub API，无绕过付费墙或登录。

## 2. 第一条裁决：先核观测模型，再优化算子

令合法 raw 在候选 q 下满足

`x | q,h ~ N(D_q h, Σ)`，已知有限算子 `F`，观测 `z=Fx`。

正确处理是

`z | q,h ~ N(FD_q h, FΣFᵀ)`。

若 F 方阵可逆，`log p(Fx|q) = log p(x|q) − log|det F|`。公共 Jacobian 在候选比值中抵消，argmax 和归一化 posterior 均不变。若 F 降秩，基于正确同协方差 Gaussian 模型，候选均值差的 Mahalanobis 能量不能增加：

`δᵀFᵀ(FΣFᵀ)^+Fδ ≤ δᵀΣ⁻¹δ`。

这是信息与线性代数结论，不依赖 EEG 经验成绩。工程收益仍可能来自更稳的有限样本估计、条件数、正则、减轻模型失配或输入资源预算；必须明确归因。若换坐标后把原先先验重新设成各向同性 ridge，已改变统计模型，不能要求预测不变或把变化全归给 F。

**对当前源码的精确判断：** `raw_continuous` 在完整 C/T scope 上先双向滤波；`ContinuousDecisionModel` 从公共事件建立有限因果 D，只对 D 和 Y 同作 AR1/nuisance 变换，没有把实际原始滤波 F 作用到 D。它可以被解释为“直接学习滤后 ERP 核”的近似模型，因此不是仅凭遗漏函数调用就判代码错误；但有限边界、负时滞、固定低维基和邻近事件让其与原始生成模型一般不等价。这是应首先研究的具体失配。

尤其是完整 T 先双向滤波后再逐 decision 裁切：当前 decision 的数值样本可能包含相邻未知 decision 的滤波响应。若只给局部 D 套 F，滤波作用域仍不一致。可选成熟处理是：

1. 每个合法 decision scope 分别用明确有限 F，并对同一 scope 的设计/噪声共同传播；这是新算子协议，必须单列对照。
2. 在同一完整 T 的观测模型中表示所有公开 events，对邻近未知类别显式边缘化；不能读取它们的真标签，也不能把邻近决策当不存在。
3. 采用允许过去状态连续传递的因果协议；计入预热时间并保留状态审计，不沿用独立双向 scope 的旧结果身份。

## 3. 第二条裁决：Γ/Ω不是任何边界处理都必需的附加项

若预测就是执行已知的有限 F_N，固定端点延拓是确定函数，无须发明随机初态。其 raw 噪声经 F_NΣF_Nᵀ 已含端点噪声影响。

只有当研究目标另定义为未知外侧过程或随机初态，才引入

`z=Aε+Ba`，`Cov([ε,a])=[[Σ,Ω],[Ωᵀ,Γ]]`，

从而

`Cov(z)=AΣAᵀ+BΓBᵀ+AΩBᵀ+BΩᵀAᵀ`。

若 a 与观测 o 联合高斯，正确条件化还应同时改变均值与协方差：

`E[a|o]=m_a+Cov(a,o)Cov(o)⁻¹(o−m_o)`，

`Cov(a|o)=Γ−Cov(a,o)Cov(o)⁻¹Cov(o,a)`。

不能用 unconditional Γ 表示已给定边界观测后的 residual uncertainty，也不能把同一驱动噪声经 A 与 B 的两个路径当成独立后相加。Särkkä 的 Gaussian conditioning/Kalman 是现成完整解法，不需要先发明新收缩规则。

真实数据识别的可执行路线：在**合法训练/校准内部**布置预声明的伪边界，用较长的同一合法范围产生状态参照；对 raw residual 与状态误差成对建模。先按事件设计移除合法条件均值，宜通过校准内部 cross-fit 避免把拟合误差压没，再估低维联合 covariance。保持 windows 的物理重叠和 recording block 单位；高度重叠伪切不能算独立样本。参照只是“在可见较长上下文下的状态”，不是无限外侧真值。真实 ERP 尾、边缘阶跃与生理非平稳若被混进 Γ，模型可能把最有判别力的均值方向主动降权；需独立注入实验和候选 Mahalanobis 能量记录。

## 4. 第三条裁决：概率层的成熟解法与它解决不了的部分

当前 `kernel_row_covariance` 是平均核 h 的条件后验误差。预测积分 `D_q V_h D_qᵀ` 是合理的一层；但它没有包括：

- 噪声谱/AR参数的估计不确定性；
- 不同决策/flash的真实振幅或潜伏期变化；
- 不完备事件设计、未知邻接响应和错误滤波 scope；
- 删去 nuisance 后的候选信息损失。

Minka Eq.47–62 提供现成 matrix-normal/inverse-Wishart → matrix-T 联合预测。若只积分一个共享 scalar variance，inverse-Gamma 混合得到 multivariate t；这与每time point各自独立t、以及积分通道covariance得到的matrix-T是三种模型。应在公式和artifact中写清楚被积分的变量。对未白化的长raw直接把每样本计为独立校准，posterior degrees of freedom会虚高；t会很快近Gaussian，过度自信依然存在。

一个有信息增益的竞争模型是目标事件/decision的随机振幅与线性化潜伏期：`h_j≈h+u_j h+v_j h′`，对(u_j,v_j)施加训练估计的小协方差并边缘化。该候选低秩 covariance 描述真实响应变化，不会因为平均核估计很准而消失。它是标准线性随机效应与ERP的组合研究提案，本次没有找到能直接替代本项目验证的效能论文。独立flash、同decision共享、跨decision共享三个层级会给不同 covariance，不能按 test 成绩任选。

温度缩放是明确的成熟概率校准控制臂，T只在 source-held-out 或 calibration内部完整decision的OOF预测上拟合。T>0不改argmax，所以只能回答NLL/校准问题，不解释hit改善。在K只有一两个完整decision时，温度可识别性很差；应按预声明 source prior 或拒绝估计处理。固定 train/test同一次selection的不同prefix不算独立OOF。

## 5. 最短的决定性实验链

1. **软件/数学闭环：** 固定可逆F共同变换不变、降秩信息界、信号单独变换翻转反例、有限F与因果短核失配、Ω缺失反例、条件state协方差、global-scale t数值积分、候选置换与保存重放。
2. **纯校准诊断：** 按时间block交叉拟合核与噪声；报告设计rank、谱白化后残差ACF/PSD、边界距离与残差能量、训练/OOF的预测检验、参数稳定性。各诊断只支持对应假设，不用QQ漂亮作为全模型正确证明。
3. **固定公开开发对比：** 同一原始授权和requested账本上比较原近似continuous、同scope exact-F mean、mean+structured-noise、再加parameter marginalization。既有完整 T 已观察过，所有新结果都是 development，不能升级为独立确认。
4. **独立新测试：** 方案固定后检验requested hit、coverage、conditional NLL、Brier、延迟及participant配对差。若只NLL改善而hit不变，按校准收益记录；不能把准确性和校准互换。

先保留强成熟桥接（原waveform LDA/Toeplitz等），再裁决新算法是否有价值；强基线不得因一个NaN mask、单位/lag、保存重载或默认参数的浅层失误被否决。

## 6. 本轮独立数值证据

`methodology_counterexamples.py` 用仓库 `.venv/Scripts/python.exe` 执行，结果为 `methodology_counterexamples.json`，8类断言全部通过，源脚本hash随结果保存。

| 命题 | 结果 | 解释范围 |
|---|---:|---|
| 可逆F共同变换 | log-BF误差1.78e−15 | 数学实现正确性 |
| 降秩F | Mahalanobis能量6.277→1.044 | 不会创造population信息 |
| 仅filter signal，均值未变 | 正确log-BF +0.309→错误 −1.964 | 候选可被翻转；不是该真实库翻转率 |
| 有限双向F vs compact causal滤后核 | 合成均值相对差0.433，10.76%能量在onset前 | 反驳无条件等价，不是当前真实数据误差估计 |
| 精确有限矩阵乘法 | max误差2.02e−14 | 同一个有限实现仍为线性 |
| 删除raw/state cross terms | 真方差0.4→错误2.0 | 不能默认Ω=0 |
| 条件state uncertainty | Γ=1，给定相关观测后0.36 | 不应混同prior与conditional covariance |
| 共享IG scale积分 | 闭式、quadrature、SciPy logpdf差≤2e−15 | scalar mixture正确；不是matrix-T或EEG效能 |
| 正温度缩放 | 四个T argmax都相同 | 不可宣称此步骤带来hit增益 |

未修改production与六份活动指南。所有新增EDF/设备/成人效果均未涉及。
