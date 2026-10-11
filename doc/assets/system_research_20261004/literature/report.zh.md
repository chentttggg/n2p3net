# 2026-10-04：从有限信号支持到候选决策的算子与算法研究

本分支先独立提出四类方案，再查一手论文；顺序保存在 `independent_proposals.md`。结论是：不宜继续把“在很短且独立的合法 raw 上复刻无限上下文 0.1 Hz 双向 SOS”作为唯一目标。那个目标一般不可识别；应以授权范围内的判别信息、失真、延迟、算子稳定性和最终决策代价共同定义目标。成熟方法可以直接成为有力对照，新的组合仍须经反例和配对实验。这里没有声称产生新的公开数据或成人性能收益。

## 一、文献核验范围与获取记录

获取日期为 **2026-10-04，Asia/Shanghai**。Parallel CLI 0.7.1 已安装但未登录；因此采用公开 Crossref 与 Europe PMC API、作者代码和本地已存论文。Crossref 的部分查询返回 429 或连接重置，未绕过限制；后续以 Europe PMC 结果核对。检索结果中的无关主题和重复预印本未进入下表。原始查询、URL、HTTP 状态、UTC 获取时刻保存在同目录 `*_crossref.json`、`*_europepmc.json`、`*_epmc_extra.json`、`*_fetch.json`。这是一项针对方法决策的有限综述，不宣称系统综述完整性。

| 一手来源 | 核验层级 | 直接支持 | 不能支持 |
|---|---|---|---|
| Alday 2019, [10.1111/psyp.13451](https://doi.org/10.1111/psyp.13451) | Europe PMC 元数据和摘要 | baseline 可作为 GLM predictor，让系数由数据确定；传统减均值是特殊情形 | 不证明本项目无 baseline 必然提高 AUC，也不是小 K 分类器的现成落地验证 |
| Smith & Kutas 2015 I, [10.1111/psyp.12317](https://doi.org/10.1111/psyp.12317)；II, [10.1111/psyp.12320](https://doi.org/10.1111/psyp.12320) | 元数据和摘要；PMC XML 请求 500，未声称审全文 | rERP 将平均、协变量、非线性、重叠纳入回归框架 | 不等于任意固定节律和任意短段都可分离响应 |
| Ehinger & Dimigen 2019, [10.7717/peerj.7838](https://doi.org/10.7717/peerj.7838) | [PMC6815663 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6815663/) | 连续设计矩阵、线性叠加、正则、时间基函数；明确变化的时序/重叠与形态稳定假设 | 不证明缩短 epoch 优于长窗；不允许使用未知 test 真标签建 target/non-target 设计 |
| de Cheveigné & Arzounian 2018, [10.1016/j.neuroimage.2018.01.035](https://doi.org/10.1016/j.neuroimage.2018.01.035) | [PMC5915520 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5915520/) | 鲁棒去趋势、正交 sin/cos 回归；Fig. 5 显示 glitch 可破坏普通 50 Hz 回归而加权回归缓解；Fig. 2 显示高通失真 | 不保证短校准内所有 slow ERP 都能和漂移分开；不能直接推广高密度通道 inpainting 到六导 |
| de Cheveigné 2020, [10.1016/j.neuroimage.2019.116356](https://doi.org/10.1016/j.neuroimage.2019.116356) | Europe PMC 元数据和摘要 | ZapLine 用完美重构频段分解加空间滤波，仅清理受线噪污染的分支 | 不证明少导/时变拓扑/独立短段满足该方法假设 |
| Klug & Kloosterman 2022, [10.1002/hbm.25832](https://doi.org/10.1002/hbm.25832) | [PMC9120550 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9120550/) | Zapline-plus 按空间噪声稳定性分块、逐块估频和选成分；文中明确普通回归的幅相变化残留、空间投影损失脑信号风险 | 不保证其自动步骤可在 K200 独立短校准、六导中安全估计，也不构成跨全部 test 拟合的授权 |
| [EEGLAB CleanLine 官方文档](https://eeglab.org/plugins/cleanline/) | 官方算法/使用说明全文 | 2–4 秒滑窗、多窗谱正弦回归、窄带频率扫描和 F 检验；幅相确定性正弦加白噪声假设 | 不是本项目新算法的性能论文，窗口跨越 calibration/test 仍不合法 |
| Widmann, Schröger & Maess 2015, [10.1016/j.jneumeth.2014.08.002](https://doi.org/10.1016/j.jneumeth.2014.08.002)；Widmann & Schröger 2012, [10.3389/fpsyg.2012.00233](https://doi.org/10.3389/fpsyg.2012.00233) | 2015 摘要；[2012 PMC3391960 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3391960/) | 脉冲/频响、相位、因果性、过渡带和失真需一起检查 | FIR、零相位或特定阶数不是无条件正确答案 |
| Gustafsson 1996, [10.1109/78.492552](https://doi.org/10.1109/78.492552) | Crossref 元数据；相关 [SciPy filtfilt 官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html) | 初始状态选择是 forward-backward 算子的重要部分；官方实现区分 padding 与 Gustafsson | 不是对本项目高阶 SOS、小段、强工频下低通先行的性能证明 |
| Sosulski & Tangermann 2022, [10.1088/1741-2552/ac9c98](https://doi.org/10.1088/1741-2552/ac9c98) | 正式元数据/摘要；本地 arXiv:2202.02001v2 14 页全文；[作者代码](https://github.com/jsosulski/toeplitzlda) | 用 block-Toeplitz 结构改善 ERP LDA 协方差估计；论文 213 人/13 协议；官方要求无 baseline、等宽时间特征、channel-prime 顺序 | 外部“最高 6 AUC 点”不是本项目收益；原文主要同会话连续 run 分割且 0.5–16 Hz/40 Hz，与本项目不同 |
| Blankertz et al. 2011, [10.1016/j.neuroimage.2010.06.048](https://doi.org/10.1016/j.neuroimage.2010.06.048) | Europe PMC 元数据和摘要 | ERP 单 trial 分类中高维协方差估计是瓶颈，收缩 LDA 是成熟基线 | 不能证明把不同来源 cov 直接平均就能迁移 |
| Rivet et al. 2009, [10.1109/TBME.2009.2012869](https://doi.org/10.1109/TBME.2009.2012869) | Crossref + Europe PMC 元数据和摘要 | xDAWN 为增强事件诱发响应而估计空间子空间 | 不能把纯 epoch covariance 当成完整保留 ERP 波形的等价表示 |
| Li et al. 2020, [10.3390/app10051804](https://doi.org/10.3390/app10051804) | 本地 13 页正式全文 | xDAWN 与几何分类的同库跨人先例；文中定义是同 dataset 的 subject A→B | 两库中的最高平均 AUC 0.836 不能移作十二库留库或成人产品成绩 |
| [Tangent Space Alignment 2022, 10.3389/fnhum.2022.1049985](https://doi.org/10.3389/fnhum.2022.1049985) | [PMC9755175 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9755175/) | 矩形交叉积/SVD 支持不同维度；目标有标签锚点；明确 rank deficiency | 349 人/18 库实际结果是库内 subject pairs；Discussion 明说 cross-database 尚待研究 |
| Bayesian Signal Matching, [10.1080/01621459.2025.2563189](https://doi.org/10.1080/01621459.2025.2563189) | 正式 DOI/摘要由 Europe PMC 核实；本地 arXiv:2401.07111 36 页全文 | 使用目标校准决定是否向源人群借力；目标 baseline cluster 不强迫与全部源共享参数 | 未细核正式发表版每一修订；不是无校准，也不证明任意 montage 可直接共享波形 |

## 二、挑战错误目标：有限段没有唯一“真恢复”

若两个连续信号在合法区间完全相同、区间外不同，任何只读取区间内样本的算法都输出相同结果；但具有无限支持的双向 IIR 在该区间内可能输出不同结果。因此，任意有限算子不能对所有外侧延拓都复原无限上下文结果。全 record 仍可作诊断参照，不能被提升为合法 ground truth 或新的训练输入。

可执行的目标应分别报告：

1. 纯已知 nuisance 的抑制与误注入；
2. 合成 ERP 的幅度、形态、潜伏期和候选对比保真；
3. 原始真实记录的质量分布、靠边位置依赖、极端值，保留持续污染者；
4. 固定 participant/test cohort 的 AUC 和 requested `hit@all`，并记延迟/缺失代价。

## 三、优先可落地的成熟路线

### M1：工频专用回归与有限 SOS 的结构重排

把低频 drift 与 50 Hz nuisance 分开。对固定角频率，令 S=[cos(ωn),sin(ωn)]，P=S(SᵀS)⁻¹Sᵀ；F_N 为实际有限段双向算子，g=|H(e^{jω})|²。考虑：

`G = F_N(I−P) + gP`。

这是本项目可检验的组合提案；sin/cos 回归已有成熟先例，文献未验证这个精确组合在本项目优越。它有以下精确性质：

- 对任意纯线成分 Sa，`GSa=gSa`。
- `G−F_N=(gI−F_N)P`，改变的算子秩至多 2。
- 对真实 ERP s，改变恰为 `(gI−F_N)Ps`，因此不能只看线噪消失，还须检验 ERP 投影代价。
- 若频率、robust weights 或分块从当前输入估计，P 是数据依赖的；以上线性结论仅在条件固定 P 时成立。选频和清理只读当前合法 scope，不可跨 scope 合并估计。

SOS 各因子在无限 LTI 域相乘可交换，但各自 padding/初始状态/反向运行的有限段算子一般不交换。“低通先双遍，再处理其余双遍”必须被记作新算子，而不是无语义影响的重构。它可能降低工频在后续边界状态里的投影，也可能产生新的端点失真。固定频响一致性、纯线、慢坡、尖峰、ERP 注入、真实弱/强污染者均需覆盖。既有失败 B1/B2/B3 不应自动复活。

### M2：把真实成熟分类器提升为正式强基线

先落实正确的 ERP 原型增广 xDAWN 几何基线和 ToeplitzLDA；这是比继续叠加 A1/Q1 更有信息增益的对照。

`Cov(X)=Cov(−X)`；裸协方差对全局极性翻转完全不敏感，且对任意时间置换也不敏感。ERP 表示必须明确保留模板交叉项或时间编码。例如以**合法训练**均值模板 P 构造 `Cov([P;X])`；不能在 test 真标签上重建 P。

ToeplitzLDA 不能只换 estimator 名称。当前每 trial baseline 算子为 `B=I−1bᵀ`，协方差应变为 `BΣBᵀ`。即使 Σ 平稳 Toeplitz，输出通常不再 Toeplitz。两条可比较路径是：

- 从 raw 重建明确无 baseline 的特征，按成熟实现的 shrinkage/taper/顺序执行；
- 在 baseline 前估计 Σ_stationary，再把已声明的滤波、重采样、baseline 线性映射 A 传播为 `AΣ_stationaryAᵀ`，在输出可识别子空间估计/正则。

第二条是标准线性协方差传播与成熟结构估计的组合提案；本次有限检索未找到直接匹配的 ERP 实现，不能称为已有验证方法。若 cache 仅保留 BX，丢弃了原 baseline 数值，就不能任意假定原 Σ 可恢复。另有 `BΣBᵀ−Σ` 落在 span(1,Σb)，秩不超过 2；这提示可用低秩计算，但不提供缺失参数的识别依据。

不同库的 channel mask、参考、观测几何不能在此被抹平。优先报告 native target-only 与相同输入 transfer 的配对，另将 montage 扩展作为独立研究轴。

## 四、重难点创新路线

### R1：直接对连续 candidate 生成模型评分

改变“每次闪烁独立分类，然后假定独立累加”的基本分解。对一整个合法 block，训练时在 source 或 target calibration 估计非目标核 h₀、目标增量 h₁、噪声 C 和 drift/nuisance N。test 时对每个冻结候选 q，由已知物理 flash membership 构造事件卷积设计 D_q，而不使用隐藏目标标签：

`x = D_all h₀ + D_q h₁ + N a + ε`。

固定所有训练参数后，计算每个 q 的同一 raw 区间高斯或稳健似然；对 a 解析积分/剖面化，最后归一化候选后验。整个 block 中每个 raw 样本只计一次，可显式处理相邻 epoch 重叠和相关噪声。rERP/unfold 是连续响应估计先例；将其接入这个候选条件生成判别器是待验证组合，不能说 unfold 已证明九选一收益。

可识别性要求是所有候选对比在去 nuisance 后可区分，例如 `rank((I−P_N)[D_q−D_q′])>0`，同时校准设计对所估核有足够秩。固定 SOA 的公共 onset 响应可秩亏；随机 membership 有机会识别类别对比，但不能自动解决公共核或强共线。必须涵盖 target repetition refractoriness、相邻 target 间隔、事件时钟偏差和残差自相关。K200 不能估数百个自由 lag 参数；需 source 估时间核字典、target 只拟少量振幅/潜伏期或低秩系数，并计入内层选择成本。

读数协议：trial AUC 仍保留给传统臂；R1 的核心检验是合法 candidate `hit@all`、概率校准和 log-loss。不要为了与旧 trial 接口一致，伪造独立 trial 分数再把原问题带回来。没有真实 membership 的 binary-only 库只能训练事件核或噪声先验，不能制造候选监督。

### R2：在判别空间边际化边缘状态不确定性

用有限滤波器的可观测状态模态 U 表示边缘不确定项，而不是宣称恢复出真实波形：`z=μ_y+U a+ε`，`a~N(0,Γ)`。有效噪声为 `C_eff=C+UΓUᵀ`，线性判别方向为 `w=C_eff⁻¹(μ₁−μ₀)`。可以利用 Woodbury 避免高维全逆；Γ 必须由 source/calibration 或先声明保守范围确定，不能通过 test 标签选定。

极限 Γ→∞ 相当于完全忽略 U 子空间。如果真实 ERP 差异也在 U，信息必然一并丢失。因此优先软不确定性、报告每个候选对比保留的 Mahalanobis 能量和拒识成本；不把“去掉边缘高幅值”直接当作质量变好。该路线尤其适合与 M2 的结构协方差或 R1 的连续似然结合。

### R3：目标原生几何与选择性借力

十二库物理共同实测通道交集小，只否定相同六导物理 EA 的前提，不否定所有异构学习。TSA 的矩形锚点 SVD 可将各自原生 ERP 增广几何映射到共同判别子空间；BSM 提示借力强度应由目标校准与源信号相似性决定，而非强迫所有源共享。两个方法的贡献应分开评估：几何识别、是否借源、借多少。

二分类只用两类均值提供的锚点通常秩不足；增加 PCA 簇虽可增加表观锚点，却让小 K 每簇样本更少且难以跨 montage 定义匹配。候选方案宜只对能被数据识别的低维判别方向对齐，未识别方向保持 target-native；预测阶段不能因 test 表现挑 source。TSA 原文没有跨数据库性能确认，BSM 波形借用也需一致坐标或显式观测模型，二者不能作为绕过物理合同的口号。

### R4：重新开放一个采集/估计量层面的方案

对于未来 BrainSync 采集，可把真实 pre-roll/post-roll 或连续因果滤波状态写进任务协议，让所需物理上下文实际被采集并计入等待成本。因果状态从已知过去流入未来推理，在部署中与“每个独立 raw scope 重置状态”的估计量不同；它不必偷用未来，但需要单独保存状态、时间支持、延迟和重放凭证。不能拿新连续在线协议的输入替换历史独立 scope 后继续配对旧分数。此路线可能比无限增加离线边界算法复杂度更有价值，仍需选择实际可接受延迟和测量瞬态，不能凭空规定某个缓冲秒数已经足够。

## 五、独立反例与优先级

`method_counterexamples.py` 用项目 `.venv` 执行通过，结果保存在 `method_counterexamples.json`。它只证明数学性质，不是数据性能验证。

| 反例 | 实测数值/结论 | 对设计的限制 |
|---|---|---|
| AR(1) ρ=0.95、179 点、26 点均值 baseline | 输出相对 Toeplitz 偏离 0.460340 | baseline 与结构协方差必须联动 |
| SPD 矩阵普通对角均值 Toeplitz 投影 | 最小特征值从 +0.001 变 −0.386300 | 正定性不能靠“原来是 covariance”推定；保留成熟 taper/PSD 处理 |
| ERP 差异落在 nuisance 子空间 | 范数 1 降至 3.34e−16 | 绝对 nuisance 不变性没有无损保证 |
| X 与 −X | 裸 covariance 位级相同；模板增广 covariance 有差异 | ERP 几何需保留模板/时间信息 |
| 周期间隔 8、核长 24 的 steady-state 设计 | 设计秩仅 8 | 不能把 deconvolution 当自动恢复器 |
| montage 观测核空间 | 相同观测可对应相差 10 的未观测通道 | 任意补导仍引入先验，不能标记为实测 |

推荐顺序是 M1 与正确 M2 并行形成成熟强对照；通过物理与实现门禁后，再把 R1/R2 作为主要创新研究轴，R3 作为异构扩展，R4 作为产品协议层的独立路线。每一步应报告拒绝原因和负结果，采用 source-held-out 或 calibration 内嵌选择；最终相同物理测试队列上比较，requested 分母保持固定。不建议再次完整扫描 A1/Q1 调参组合，也不建议把未接模型的 EA artifact 修补称为全链算法优化。
