# 有限授权区间的双向滤波：候选与判定标准

2026-09-29。独立原型研究；禁止越界读取实际 test/context，禁止 NN 训练和按新 AUC 选方。使用 research-architecture-decision、scientific-brainstorming；先独立生成候选，再核对原始来源。当前输入合同固定 0.1–30 Hz、50 Hz notch、原采样率及相同双遍幅频目标。

## 先验可识别性边界

任何两个真实连续信号若在授权区间内完全相同、区间外不同，合法算法只能得到相同输出，而无限上下文双向滤波一般不同。因此“无假设恢复真实无限上下文”不可能；必须明确边界/信号模型、条件误差或拒绝声明保证。BI49/26还存在真实持续污染，消除BI01右端假坡不会自动清洁这些记录。

## 独立候选

| ID | 原理与公式 | 可接入性与反例 |
|---|---|---|
| B1 | SOS状态空间 Gustafsson：以观测矩阵O和反向传播S，求 `argmin_z ||[S^R−O, O^R−S]z−(y_bf^0−y_fb^0)||²`，再 `y=y_fb^0+[S^R,O^R]z` | 保持既有SOS和|H|²，可保存双端有效state。FB=BF不证明ERP保真；长时间常数/近不可观测模式可能病态，需要记录秩、条件数与残差。避免把SOS转高阶BA引入数值失稳。 |
| B2 | 物理谐波条件延拓：在授权的两端分别投影 `h(t)=a sin(2πft)+b cos(2πft)`，用 `h(t_out)+odd_extend[x−h](t_out)` 构造边缘，再执行原SOS双遍 | 中间实测样本完全不改；延拓仅从本区间边缘估计，不读取未来。固定f时为线性；估f时是预声明自适应信号算子。须挑战f漂移、幅度调制、step及ERP在拟合边缘的影响；记录每端f/coef/fit范围。不是宣称复现ZapLine。 |
| B3 | 有限区间自伴谱算子：`F_N=Q_DCT^T diag(|H(k fs/(2N))|²) Q_DCT`，Q为正交DCT-II | 直接声明even/Neumann边界，算子范数≤1；不依赖不稳定初态。可能镜像边界ERP/改变慢波；既有SOS恢复证书不能原样套用，必须新算子族或明确线性算子误差界。 |
| B4 | 显式信号+工频振子+慢漂移的状态空间平滑，返回后验均值与端点不确定性 | 把“不知道边界”建模而不是隐藏。需要预声明模型和先验；EEG信号/慢伪迹可能不可识别。当前只保留概念和反例，不以复杂度名义排除，也不在本轮任意估计新先验。 |

B1/B2/B3进入同一独立原型。成熟方向B1与原理不同的B3并行保留；不预先指定赢家。

## 数学与证据来源

- Gustafsson F. *Determining the initial states in forward-backward filtering*. 1996. DOI: https://doi.org/10.1109/78.492552 。本轮核实 Crossref 原始出版元数据及 SciPy官方实现；未获取IEEE全文，不夸称重读完整证明。SciPy参考实现：https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html 。其文档明确FB=BF、忽略padding，并警告边界瞬态一般不可避免。
- SciPy正交DCT定义：https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html 。B3为本项目提出的固定频率响应有限区间算子，不因引用DCT文档而声称有EEG疗效实证。
- de Cheveigné A. *ZapLine: A simple and effective method to remove power line artifacts*. NeuroImage, 2020. DOI: https://doi.org/10.1016/j.neuroimage.2019.116356 。本轮核实原摘要（Europe PMC原论文镜像）：完美重构滤波器组分离受工频污染分量，再空间投影清理后合回。它支持先分离线噪结构、保护其他信息的原则；B2的局部谐波条件延拓不是其算法复现。

## 预声明判定轴

1. 纯50/50.01Hz、8mV、16个相位的端点污染：报告每相位末1.4s峰值/RMS；5µV以内作为候选继续评估的严格工程目标，失败保持失败。
2. 已知ERP：幅度10µV、N200/P300双峰，峰位置覆盖两端与内部；比较独立区间输出和只供诊断的已知合成延长上下文输出。报告RMS误差、峰幅、潜伏期。ERP仍在区间内但靠端时不豁免；完整区间外不可辨识限制另写。
3. ERP+工频：比较差分 `F(line+ERP)−F(line)`，防止仅通过抹除所有信号通过污染门槛；同时比较F(ERP)的单独形状。非线性自适应方法尤须这一反例。
4. 0.05/0.2Hz慢波、线性趋势、阶跃：低频趋势不是默认应删的伪迹。检查响应与目标滤波不同之处，不能按输出更平滑判好。
5. 实际BI01/26/49、Huebner合法C200：只读取授权区间用于候选；原full-record作为诊断参照时单独命名，不作为部署可用输入。报告各端差异、已知ERP叠加响应与所有真实污染残留。
6. 必須检查时间反转对称性、线性/条件线性、恒定输入、输出单位、相同保存参数重放；B1记录秩条件，B2记录拟合频率范围。不得用分类成绩决定分支。

脚本和小摘要留doc/assets；大数组及逐case结果留Git忽略的 `experiments/runs/chain_improvements_20260929/boundary/`。串行执行，native线程1。
