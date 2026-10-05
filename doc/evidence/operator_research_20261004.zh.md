# 算子深度科研与连续决策复核（2026-10-05）

本记录补足上一轮只完成工程接线、没有完成算子科研的问题。所有结果属于公开数据校准组开发研究，使用改前冻结基准及本轮逐次保存的执行源码；没有读取 BNCI 测试选择参数，没有改写旧成绩，也没有把本轮结果当成人设备确认。

## 先回答算子到底是什么

对已经读取的有限数据段，双向 SOS、反射 padding 和起始状态是确定的线性映射 `z=F x`。如果候选均值和协方差同时变换为 `FD` 与 `FΣFᵀ`，可逆变换只增加候选共同的 Jacobian，Bayes factor 与归一化 posterior 不变；降秩变换不能凭空增加 Mahalanobis 信息。因而把确定 padding 再加一份独立 `Γ`，或只滤 `Y` 而不滤 `D`，都不是同一个观测模型。

当前 BNCI MAT 也不能称为 ADC 原始波形：官方分发说明声明 0.1–30 Hz band-pass，但具体阶段、阶数和相位未给出；Riccio 论文另述的 0.1–10 Hz、50 Hz notch、epoch rejection 和 baseline 是离线分析步骤，不能直接改写 MAT 合同。当前路线的准确称呼是“分发 MAT 信号，未追加本项目滤波”。来源与复核见[分发信号与条件 AR 专项](../assets/operator_research_20261004/literature/bnci_distribution_and_conditional_ar.zh.md)。

## R1：边缘状态不是可以随手加上的方差

在合法 C800 内构造 1504 个预声明伪截断、48 个留 participant fold，并做 2016 个固定 ERP 注入挑战。无约束的长短滤波差值预测可把差值 MSE 降约 81%，但把 P300 增量误差从 2.28% 放大到约 24%；`K U_ERP=0` 的保护约束在距边缘 2/8 秒仍有 39–47% 的差值 MSE 改善，却在紧贴边缘无收益。强制恢复长参照还产生最高约 795 倍白噪声放大及不可表示约束。

结论是：真实有限输出的残差已经包含边缘变化；再加 `BΓBᵀ` 可能双计。若 `a` 是外部未观测 state，必须同时定义 `Σ、Γ、Ω`、state 单位、`B` 的可观测秩，并用 Schur 补条件化；若 state 是从观测 `x` 算出来的 padding state，条件协方差不能继续当作独立未知量。无约束边缘修正路线保留为失败证据，未晋升。

## R2：把均值、时间噪声和先验一起检验

新增 `FiniteFilterOperator` 保存确定 SOS、边界规则和 state 分解；新增 `TemporalNoise` 支持分记录 Burg、平稳启动和 PARCOR lattice；`OperatorGLS` 将连续事件设计、有限算子、AR 噪声、通道协方差和 kernel 后验放入一个可重载研究模型。有限算子与独立 SciPy/dense 参考、AR0/1/2/3、单位变换、候选置换和旧连续模型桥接均通过，旧配方逐值一致。

真实组实验固定为八名 BNCI 参与者、每人六个完整 selection、留一组拟合其余五组。每折使用 600 个校准标签；C800 中另 80 个标签保留但不用；每个 held selection 使用完整 120 flashes。共执行三阶段开发矩阵：第一阶段 240 cell，第二阶段 624 cell，第三阶段 768 请求；第三阶段原始执行源的结果为 752 完成、16 个 `Matrix is not positive definite`，失败集中于高阶 AR32 边际先验。该失败已定位为 `DᵀD` 平方条件数导致的后验协方差数值失正定，不是 PARCOR 不稳定或算法已被证伪。

先验尺度是实际机制：固定 `ridge=1` 会随 AR innovation/marginal scale 改变 ERP 先验。数值 gauge 反例显示，保持物理噪声协方差不变而只改变坐标尺度，若 ridge 不同步，log Bayes factor 可改变约 216；同步 `λ_eff=λ/r0` 才保持同一先验。边际尺度路线在真实 C 内反而使 NLL 恶化到约 9–13，出现严重过度自信，不能当作校准修复。

三阶段结果的主要模式如下，NLL 是各模型自声明的近似后验损失，八人是统计单位，六折只是同一 participant 内的依赖重复：

| 路线 | 命中率 | 条件 NLL | 解释 |
|---|---:|---:|---|
| 旧 filtered-Y + 因果核 AR1 | 0.604 | 2.797 | 参考路线 |
| filtered-Y + filtered-D AR1 | 0.375 | 6.347 | 只匹配设计不能解决整体失配 |
| raw AR1 | 0.604 | 3.023 | 命中不变、概率仍失配 |
| raw AR8 | 0.667 | 6.162 | 命中上升但过度自信 |
| raw AR8 三轮均值/噪声更新 | 0.479 | 2.961 | NLL改善但命中下降 |
| raw AR8 零端点基三轮 | 0.583 | 2.577 | 仍有明显残差相关 |
| waveform LDA + baseline | 0.708 | 2.479 | 成熟 ERP 锚点 |
| waveform LDA，无 baseline | 0.708 | 2.268 | 成熟 ERP 锚点 |
| Toeplitz LDA，无 baseline | 0.771 | 2.163 | 当前 C 内最强成熟锚点之一 |
| xDAWN ERP + baseline | 0.771 | 3.075 | 命中强，概率损失较差 |

第三阶段的 AR32 lattice 普通路线命中为0、NLL约3.59，残差 ACF1 仍约0.39；它说明时间白化降低相关性不等于保留候选信息。QR 修复后，原16个失败 cell 以当前最终研究源码全部48 fold重跑完成，独立保存重载和log evidence一致；这组修复结果只解决数值可执行性，不追溯覆盖原失败账本。

## 成熟 ERP 对照的意义

SelectionERP wrapper 复用现有波形 LDA、StructuredERP Toeplitz 和 xDAWN owner。真实物理支持为 `[-51,+306]` 毫秒对应的 358 个 256 Hz 样本；共享 FFT 输出 179 点，再按真实闭区间合同保留 178 点，不读取 sample 307。mean-only baseline 使用 26 点，Toeplitz 使用 bin width 4 并显式丢弃尾 2 点，xDAWN 使用实测空间秩门禁。四个锚点 192/192 完成，原 owner 逐 trial logit 与保存重载一致。

这些锚点是同一公共 C 内的开发对照，不是四库迁移矩阵。它们的 membership 聚合 NLL 是条件独立近似，与连续 Gaussian NLL 不能直接当作同一个统计模型排序。

## 现在能下的科研结论

1. 直接添加边缘 `Γ`、只滤观测或只换 AR 阶数都没有形成可信的整体收益；必须同时说明 finite operator、均值字典、时间噪声和先验尺度。
2. 无约束边缘校正被 ERP 保真反例否定；受约束修正只在离边缘较远处显示诊断价值，不能注册为通用线性算子。
3. 三轮均值/噪声联合更新能降低 NLL，但命中下降；AR8 命中改善伴随概率恶化。命中与概率校准必须分开。
4. 当前真实 C 内成熟 Toeplitz/xDAWN 锚点强于连续研究路线，但这不是十二库三 seed 或成人设备结论。
5. R1 的真实 Γ/Ω 生理识别、逐事件 `A_i/B_i`、完整联合 decoder、十二库完整匹配矩阵和成人确认仍未完成。

当前最终字节还保留过两个已复现的消费层风险：模型工件必须绑定模块加载时的实现摘要，executor 启动异常必须封存为 failed；这些已在当前代码中修复，但旧 Stage3 结果仍按其原执行源码解释。最终 repair3 以当前源码完成48/48 AR32边际单元，不能把它与旧752/16账本混为一份执行。

所有失败单元、执行源码、模型/预测封印、阶段计划和独立复算分别保存在 `doc/assets/operator_research_20261004/` 及 `experiments/runs/operator_research_20261004/`；没有把失败路线从分母删除。

后续 Phase 4 又做了预声明软约束扫描和模型混合实验，见[Phase 4报告](../assets/operator_research_20261004/phase4_report.zh.md)。在同一 C800 开发组内，raw AR8、Toeplitz 和 xDAWN 的等权概率混合达到 hit 0.771、NLL 1.069；Pz 固定/训练权重反而降低命中。这个候选优先进入独立公共数据复核，仍不替换生产默认。
