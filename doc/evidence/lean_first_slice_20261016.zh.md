# Lean 首批验证：授权视图、有限 SOS 与协方差传播

日期：2026-10-16。执行范围：顺序方案 `V01.a → V02.a → V02.b` 的实数数学规格、
源码语义审查和合成行为对应；全部 Python/IEEE refinement 另行记录。

## 交付与数学关系

本轮从试验引理扩展为可独立检查的模块，并增加跨模块组合。
既有 14 条试验定理保留原身份；当前生产源码在执行前后核对实际摘要。
旧试验的 9 个证明/配置/检查器文件已按原回执哈希物理封存并逐成员校验，见
[归档清单](../../frozen/lean_pilot_20261016/manifest.json)；压缩包留在本地。

| 模块 | 已证明的核心关系 | 主要源码对应 |
|---|---|---|
| [RawScope](../../formal/lean/N2P3/RawScope.lean) | 非零 origin 的绝对/局部坐标、半开区间完整提取、原事件授权、嵌套前缀、purge、真实 availability 最大值、严格右删失；实际提取输入的非干扰 | `RawSampleScope`、`crop_raw_scope`、`prefix_raw_scopes`、`source_records`、`raw_continuous`、校准划分 |
| [FiniteSOS](../../formal/lean/N2P3/FiniteSOS.lean) | 实际 DFII-T 单节、任意级联和有限时间递推线性；观察内反射与双端有效状态；`F(x)=Qx+B a(x)`；单节/级联稳态初值平衡；真实展平次序与 4S 坐标数 | `filter_bidirectional_segment`、`FiniteFilterOperator` |
| [FiniteAdjoint](../../formal/lean/N2P3/FiniteAdjoint.lean) | 从实际递推推出有限因果 Toeplitz，继而 `RHR=Hᵀ`、`Q=HᵀH` 与 Q 半正定；不假定待证的 Toeplitz/伴随性质 | `FiniteFilterOperator.zero_state` 的实际两次递推及其数学注释 |
| [Covariance](../../formal/lean/N2P3/Covariance.lean) | 从二阶矩随机变量及真实 covariance 定义推出线性/联合传播；确定性状态 JX；L2 算子范数下的近似 PSD、残差与严格 ridge 界 | `JointNoiseModel.project`、`ExactPropagatedCovariance.compute` 与数值准入前提 |
| [FirstSlice](../../formal/lean/N2P3/FirstSlice.lean) | 从实际选择矩阵 E 与已构造有限滤波 F 得到 `FE`，证明先提取后处理的非干扰、矩阵表示及真实随机协方差 | 合法 raw 输入、有限滤波、状态和输出协方差的纵向组合 |

### 授权范围

原样本、原事件、原标签分别建模。输入记录的非零 `first_samp` 显式进入绝对/局部坐标转换，
保留的 evidence 只能映射回原授权事件；purge 不会增加 K。epoch 起止使用有符号整数，
所以刺激前负支持不会被自然数截断后错误放行。源网格的闭端点用 `+1` 转成半开区间。

通用 epoch 路线可授权完整响应尾部，可能包含下一个未授权事件的波形，但不授权其标签；
连续候选路线另在下一未授权 onset 前截止，缺失响应以右删失表达。二者分别证明并给出
错误混用的反例。实际标签何时可得、完整 metadata/文件解析和全部选模/refit 组合仍属于 V01.b。

### 有限滤波与确定性边界状态

SOS 采用实际归一化 DFII-T 方程，先证明单步线性，再递归级联和有限采样。
left pad、core、right tail、反向 tail、两端有效状态及 pad=0 分支按实际算法构造。
`sosfilt_zi` 对应的单节 DC 平衡和前级 DC 增益累计缩放有实数证明；SciPy 求解器是否
按该模型执行、根稳定门禁与系数误差则属于后续实现对应。

左、右状态展平坐标分别为 `2*section+coord` 与 `2*S+2*section+coord`，合计 4S 列。
一采样非零状态核的反例证明列数不保证满秩。`RHR=Hᵀ` 从因果递推的脉冲响应导出，
因而源码的 `Q=HᵀH` 精确数学关系已经补齐。重置分段分别初始化，不读取其他段数据。

### 协方差与数值边界

对有二阶矩的随机坐标 X、Z，不要求相互独立：

\[
\operatorname{Cov}(AX+BZ)=A\Gamma A^\top+ACB^\top+BC^\top A^\top+B\Omega B^\top.
\]

观察内 padding 状态若为 `Z=JX`，则证明输出协方差恰为
`(A+BJ) Γ (A+BJ)ᵀ`；不能另加独立边缘噪声。堆叠输出行同样保留重叠相关。
线性算子、范围和已拟合参数必须在该概率模型中固定；数据相关的自适应估参需要条件化或额外分析。

对称 `Σ̃≥−εI`、`ε≥0`，对称残差 E 的真实 L2 算子范数满足 `||E||₂≤η` 时，证明

\[
L\widetilde\Sigma L^\top+E\succeq -(\varepsilon\|L\|_2^2+\eta)I.
\]

给所有特征值的下界，并证明 ridge 严格大于该已传播界时正定。
`ApproxPSD` 与加 `εI` 后精确 PSD 的等价性也已证明。
这些定理没有把 `eigvalsh` 或 `norm` 的运行时读数当作已认证误差，也没有证明当前
`max_eig*1e-12` 自动满足所有输入下的充分条件。

## 独立源码与行为证据

[独立审查](lean_first_slice_source_review_20261016.zh.md)和
[结构化回执](../assets/lean_first_slice_20261016/source_review.json)分别记录源码、反例、
审查时证明快照和未闭合项。9 组纯内存行为探针全部通过，包含 15 个 SOS 网格/填充实例；
84 项相关已有测试全部通过。未读取原始 EEG、训练或改变生产算法。

最重要的可区分反例包括：

- 未授权波形置 NaN、标签置 999 后，合法裁剪和选标签保持不变；先全局滤波后裁剪则不同。
- 可准入的 `diag(1,-5e-11)` 经 `[0,1e6]` 投影产生 `-50` 方差；任意正 ridge 或恰等于界均不足。
- 指定对称残差 E 的实例验证传播界包含残差项；它不是库舍入误差的观测认证。
- 保留原始确定性 padding 状态时联合传播吻合；删除交叉项显著改变结果。
- 源网格的末采样、负 epoch 起点、右删失 onset 边界和状态零空间均有独立错误版本。

## 验收、范围与后续

首批[机器验收回执](../assets/lean_first_slice_20261016/acceptance.json)已全部通过：
六模块共 136 个定理（含试验原有 14 个）逐模块另起进程重查，所有定理公理依赖仅含
`propext`、`Classical.choice`、`Quot.sound`。5 个负例文件的 9 条错误证明分别因 False
目标未解决被拒，任何导入/类型错误不能计为成功。13 项命令检查合计 427.51 秒
（不含各命令等待共享编译锁的时间），进程树采样 RSS 峰值约 1.924 GiB；单线程 Lean。
189 份生产/实验源码及 21 份证明/检查器/配置文件在验收前后保持同一摘要。
所验收的证明源已物理压缩并逐成员复核，见[本轮归档清单](../../frozen/lean_first_slice_20261016/manifest.json)。

总验收入口为 [verify_stage.py](../../formal/lean/scripts/verify_stage.py)：
构建当前模块、每模块另起进程重新检查、逐声明审查公理，并要求每条故意错误证明
恰好因未解决的 False 目标被拒。多条错误证明不能只凭一个失败通过检查。
所有生产入口及证明/检查器的实际摘要在验收前后核对；数学证明只接受标准公理允许集。

本轮仍未完成整个 Python/MNE/SciPy 形式化 refinement、底层浮点误差认证、全链 V01.b，
也不证明成人效果或非凸全局收敛。方案中的这些维度独立保持未完成。

V03.a 的独立低秩代数准备已进入下一模块；其概率解释需注意 `_whiten_and_project`
移除了常数和趋势方向。实际投影空间的协方差为 `P+UUᵀ`，环境维度中 `I+UUᵀ` 是
对被消除方向的恒等扩展。条件密度必须在保留子空间上建立，并区分候选共同归一化项；
不能直接称投影后观测具有满环境维非退化高斯密度。
