# 原愿望清单来源

来源聊天：01a0eb70-bcfb-7a92-b22b-1bfd6b541ddc

来源 turn：01a10a3c-0b62-7213-8602-c4373d27a36c；assistant message：msg_038ef2d01eb108d5016ac3222572cc87d09b51db571d389ca8。

以下为恢复的历史原文。数值与判断需要当前源文件独立核对，不视为本轮结论。

可以。下面是一份“愿望清单”，范围故意放宽，既包含近期高价值改进，也包含需要重新设计模型和实验体系的长期方向。它们目前都是研究候选，不能直接当成已验证方案。

**一、先把信号本身弄清楚**

- 建立完整的上游信号履历：设备滤波、分发滤波、参考电极、硬件抗混叠、MAT 写出过程都分别记录。
- 把“分发 MAT 信号”“项目新增滤波”“真正 ADC 原始信号”分成三个明确层级。
- 对每条记录保存实际有限滤波矩阵、padding 规则、初始状态和有效样本范围。
- 同时保留因果、双向和逐 selection 独立滤波三种协议，明确各自延迟和边界含义。
- 对 0.1–30 Hz、不同 notch、低通频率、滤波相位做完整输入对照。
- 对持续工频、频率漂移、幅相调制、突发 glitch、慢漂移分别处理，不能用一个 cleaner 解决所有污染。
- 发展“信号保真门”：已知 ERP 注入、波形增量、峰值、潜伏期、频谱变化都必须过门。
- 根据边界距离自动决定修正强度，贴边界时减少修正，远离边界时允许更多修正。
- 为边界修正加入“不伤害 ERP”的软约束和噪声放大约束，输出完整 Pareto 曲线。
- 研究相邻未知 selection 的响应边缘化，避免把邻近刺激当不存在。
- 研究每个 selection 独立滤波和整段连续滤波的真实差异，保留两套数据身份。
- 为滤波估计误差建立不确定性，而不是把自适应频率和权重当成固定线性算子。
- 为坏导、参考污染、EOG、通道缺失建立显式空间观测模型，避免补零或静默删除。

**二、重新设计边界状态**

- 将确定 padding state、未知外部生理 state、滤波估计误差分成三个对象。
- 对真实外部 state 使用完整的 \(\Sigma,\Gamma,\Omega\) 联合协方差。
- 对已经观察到的原始信号使用 Schur 补条件化，避免重复加入无条件 \(\Gamma\)。
- 把边界状态和 ERP 状态放进同一个 Kalman/状态空间模型。
- 估计边界差值的条件均值、条件协方差和 ERP 保护方向。
- 用距离边界、上下文长度、刺激密度、通道质量共同决定修正可信度。
- 对无法表示的边界方向直接拒绝修正，保留原有限算子结果。
- 将 MSE、ERP 保真、噪声放大和命中率一起作为多目标优化，而不是只最小化 MSE。
- 研究“先预测差值，再做受约束校正”和“直接在 raw 观测模型中边缘化边界状态”两条路线。
- 研究不同 ERP 子空间：固定 P300、早期 ERP、慢漂移、邻近 flash 响应、潜伏期变化。
- 对 \(K U_{\mathrm{ERP}}=0\) 的约束做软化、分段化和距离自适应化。
- 为每种边界策略保存可解释的约束残差、condition number 和白噪声放大倍数。

目前 Pareto 扫描已经显示：边界距离 2 秒和 8 秒存在比较好的折中，距离 0 秒没有稳定的免费收益。具体结果见[Phase 4报告](/D:/BCI/n2p3-net/doc/assets/operator_research_20261004/phase4_report.zh.md)。

**三、把 ERP 响应字典做得更灵活**

- 从固定 8 个 Gaussian 基扩展到 B-spline、raised cosine、wavelet、多尺度 temporal basis。
- 加入 ERP 一阶导数和二阶导数，表达潜伏期抖动。
- 加入每个 selection 的随机 amplitude、latency、width 变化。
- 将 common ERP、target increment、selection-specific ERP 分开建模。
- 对不同刺激行列分别学习响应差异。
- 对早期响应、P300 峰、晚期慢波设置独立子空间。
- 用带零端点、非零端点、软端点的基函数做预声明对照。
- 研究完整已知 event history 的卷积模型，避免只建当前 selection 的事件。
- 让模型学习“滤波后 ERP 字典”，但同时保留 raw 坐标模型，区分字典适应和算子收益。
- 用物理时间、采样点和响应支持共同定义字典，避免只在数组下标上做近似。
- 将响应字典的自由度纳入预算，避免高阶基函数只是增加过拟合能力。

**四、AR、Burg、PARCOR 和噪声模型继续深入**

- 继续比较 AR0、AR1、AR4、AR8、AR16、AR32。
- 用固定 hold-back 进行不同阶数的公平比较，避免 AR 阶数改变评分样本。
- 保留 Burg 作为稳定初始化，再用完整 stationary likelihood 做精确更新。
- 使用 PARCOR lattice 处理高阶 AR 的启动状态和数值稳定性。
- 对平稳启动、条件前缀和因果状态传递分别建立独立实验。
- 做逐通道 AR，而不是强行假设所有通道共享一个时间谱。
- 做低秩动态因子模型，让几个通道共享部分时间动态，其余通道保留私有谱。
- 研究 VAR、稀疏 VAR、图结构 VAR 和 channel-group AR。
- 研究 AR 与 Matérn Gaussian Process、状态空间模型、Kalman smoother 的关系。
- 研究非平稳 AR、change-point、selection 内不同噪声状态。
- 研究 Student-t、scale mixture 和 matrix-t，但只在明确的协方差层级上使用。
- 对每个 AR 模型同时报告训练残差 ACF、held 残差 ACF、频谱、创新方差、预测二次型和条件数。
- 检查白化后残差变好时，候选判别方向是否被一起消掉。
- 建立“白化收益”和“分类收益”两个独立门槛。

目前 AR8 的命中率较高，但 NLL 过度自信；AR32 可以降低残差相关，却可能丢掉判别信息。这说明 AR 应作为候选证据分支，而不是单独接管最终决策。

**五、空间建模和 Pz 的正确用法**

- 不直接把 Pz 作为残差 precision 的手工放大项。
- 通过 xDAWN、Riemannian ERP、空间 LDA 学习实际判别空间。
- 用训练组估计 Pz、Cz、Oz 等通道的 ERP d-prime，但做强收缩，防止小样本极端权重。
- 将 Pz 重要性放进 ERP 均值先验，而不是直接修改噪声精度。
- 使用头皮位置建立 graph Laplacian 或 spatial smoothness prior。
- 研究左右半球差异、后部通道组和中央通道组的分层模型。
- 将 reference、channel gain、通道缺失和空间投影一起建模。
- 评估 Pz 权重在不同 participant、不同源库、不同 reference 下是否稳定。
- 研究“空间权重 + Toeplitz 时间协方差”的联合模型。
- 研究“xDAWN 空间专家 + raw temporal 专家”的概率混合。

目前简单 Pz 权重和训练 d-prime 权重都降低了命中率。Pz 可能确实重要，但它更适合进入 ERP 空间建模和 xDAWN 类路线。

**六、先验要灵活，但不能失去约束**

- 把 ridge 从单一数值改成明确的先验族。
- 使用按设计信息归一化的 g-prior，减少基函数尺度变化带来的假差异。
- 使用预声明的 lambda mixture，而不是看 held 结果挑一个 lambda。
- 对 ERP 核使用层级 variance prior，让模型可以自己决定响应幅度。
- 将 common ERP 和 selection-specific ERP 使用不同强度的收缩。
- 对 kernel covariance 使用低秩加对角结构。
- 对 noise scale、AR 参数和 ERP 参数的不确定性做联合边际化。
- 研究 robust empirical Bayes、弱信息先验和重尾后验。
- 把先验尺度变换、创新尺度和物理电压单位写进 artifact。
- 设置先验敏感性报告，显示 lambda 改变后 hit、NLL、最大概率和残差如何变化。
- 禁止用单次 held selection 选择温度、lambda、AR 阶数或基函数数量。

当前等权 lambda mixture 改善了 raw 连续模型的 NLL，但没有超过成熟 ERP 的命中率。它适合作为灵活概率模型研究臂，暂时不适合成为默认模型。

**七、把 N2P3Net 和经典 ERP 统一起来**

这里不应该做二选一，而应该做多专家体系。

候选专家可以包括：

- N2P3Net full-fine；
- N2P3Net 同结构 scratch；
- waveform LDA；
- Toeplitz LDA；
- xDAWN ERP；
- raw AR 连续似然；
- source temporal basis + target native head；
- 固定 cosine temporal basis。

一个统一的候选分数可以写成：

\[
p(q)
=
\sum_m w_m p_m(q)
\]

其中 \(m\) 是不同专家，权重可以：

- 固定等权；
- 在训练 selection 内估计；
- 按校准预算 K 改变；
- 按信号质量、通道覆盖和边界距离改变；
- 由一个只使用 calibration 的轻量 gating model 决定。

当前 Phase 4 的 raw AR8 + Toeplitz + xDAWN 等权混合已经给出了 hit 0.771、NLL 1.069 的开发结果。下一步应该加入 N2P3Net，比较：

- N2P3Net 单独；
- 三个经典/连续专家混合；
- N2P3Net + Toeplitz；
- N2P3Net + xDAWN；
- N2P3Net + 三专家混合；
- 同结构 scratch + 三专家混合。

这样才能判断 N2P3Net 是：

- 主要专家；
- 在高 K 才有优势的专家；
- 适合做空间表征但不适合独立决策的专家；
- 还是适合通过混合提供互补信息的专家。

**八、Hit 和 NLL 怎样排序**

你的判断是对的：产品决策应该把 hit 放在第一位。

建议使用分层规则：

1. 先看 participant-macro `hit@all`；
2. 再看完整 coverage、失败配对和拒判率；
3. 在 hit 接近的模型中，用 NLL、Brier、最大置信度和校准误差做二级筛选；
4. 对 NLL 明显差于均匀基线、且错误时高度自信的模型设置风险门；
5. NLL 不能替代 hit，也不能因为 NLL 好就接受命中率明显下降。

NLL 的作用像一个“诚实度检查”：

- hit 问“答对了吗？”
- NLL 问“答错时是不是还非常确定？”

36 选 1 的均匀 NLL 是：

\[
\log 36\approx3.584
\]

当前 raw AR8 的 NLL 约 6.162，说明它虽然有时命中，但概率不诚实；三专家混合的 NLL 约 1.069，说明它在这组开发数据中同时提高了命中后的概率质量。

**九、实验体系还可以扩大**

- 增加独立 participant；
- 增加完整四库；
- 运行 3 个预声明 seed；
- K=0、200、800、2000 全部保留；
- 每个方法共享真实 raw scope、测试队列、channel mask 和预算；
- 训练组内选择参数，held 只做一次评分；
- 按 participant 做 bootstrap 和配对差值；
- 报告每个 participant 的失败、负迁移和不确定性；
- 增加完整 selection、邻近 selection、不同响应间隔的分层结果；
- 增加不同 reference 和通道缺失面板；
- 将算法选择、概率校准和产品晋升分成三个独立阶段；
- 先在公开数据完成预声明路线，再使用新的成人采集确认。

**十、当前最值得优先推进的路线**

目前我会把愿望清单压缩成这一条主线：

1. 保留 N2P3Net 作为神经专家；
2. 保留 Toeplitz 和 xDAWN 作为成熟空间/时间专家；
3. 保留 raw AR8 作为连续时间专家；
4. 用等权概率混合做稳定基线；
5. 在训练 selection 内研究 N2P3Net 与经典专家的 gating；
6. 用 hit@all 作为主目标；
7. 用 NLL、Brier、最大置信度和边界保真做安全门；
8. 对边界只在 2 秒以上的可识别区域启用软修正；
9. Pz 进入空间 ERP 建模，不直接手工放大残差精度；
10. 先完成独立数据和四库三 seed，再考虑接入产品默认。

这份愿望清单已经保存为研究证据的延伸部分：

- [Phase 4 报告](/D:/BCI/n2p3-net/doc/assets/operator_research_20261004/phase4_report.zh.md)
- [Phase 4 结构化结果](/D:/BCI/n2p3-net/doc/assets/operator_research_20261004/phase4_report.json)
- [Phase 4 结果清单](/D:/BCI/n2p3-net/doc/assets/operator_research_20261004/phase4_manifest.json)
- [完整算子科研报告](/D:/BCI/n2p3-net/doc/evidence/operator_research_20261004.zh.md)


