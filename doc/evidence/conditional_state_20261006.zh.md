# B06：候选条件均值、Schur 协方差与真实 operator 调用

日期2026-10-06。实施起点为隔离主树1055982的三个B06文件；主线随后整合的trace core有独立来源，不属于本切片改动。未修改原D:/BCI/n2p3-net或其数据。本记录只说明软件及一个公开processed-EEG调用验收，不能认为已识别未知滤波边界state。

`transfer.noise_model.JointNoiseModel`仍是唯一joint residual covariance owner。新增`condition(raw_values, raw_means, state_means)`返回每观测/每候选的实际a|x,q均值及共享Schur协方差；raw/state means明示为包含任何noise mean的完整候选期望。公式为`m_aq+ΩᵀΣx_support^-1(x−m_xq)`和`Γ−ΩᵀΣx_support^-1Ω`，不能只缩协方差而沿用无条件均值。

owner只在标准化raw的已识别数值支持内解算，并核所有候选residual及Ω range；明确记录eigenvalue、rank、threshold和Schur舍入模式。常量raw可以是rank0；确定state=Lraw的无制造噪声模型得到零Schur。可分辨的独立弱电压方向保留。无含义不明的pinv、nullspace epsilon或静默降级。当前stochastic LDA要求共享affine支持，拒绝null类对比；这不是一般退化Gaussian在不同类支持上不可判别的证明。joint shrinkage的非零Schur只是拟合统计模型假设，不代表真实外部state噪声已确认。

`OperatorAwareLDA`的条件mode实际使用`J=A+BΩᵀΣx_support^-1`，以`z=y−Jx`及`m_yq−Jm_xq`评分，covariance同时改为`B Schur Bᵀ`。响应支持可奇异，regularization只作用在已表示方向。固定等类prior、不额外乘raw likelihood；prefix有类信息时不能称full p(q|x,y)。显式`ConditioningObservationSupport`绑定conditioning/scored native区间和可用样本上界，响应必须包含conditioner外已观察信息；区间声明仍须由真实producer的sample map/读数账本证明，不是自签权限。

noise `/2`与conditional operator `/2`保存实际gain、Schur/conditional covariance、均值、数值支持及source/fit/input/operator关系，载入重核数值关系与producer。旧unconditional operator保持单一pushforward程序；不能把完整F(X)当新独立观测。真实CLI已接`noise-condition`，`operator-lda-fit`接conditioner和support，`operator-lda-predict`只接受无标签features/observed_raw；held label、state oracle、错support、重签错误numeric/producer及降级隐藏payload均拒绝。

独立理论出口为12个NumPy反例，未导入生产代码；实现保留初始缺接口collection error、6项owner RED及3项caller RED，后续定向验证另记，见本地`tmp/conditional_state_20261006/{implementation,spec}/`。原失败日志不改写。精确测试计数、命令、actual SHA及独立复审由最终receipt和主线验收记录，不借旧source的全套成绩追认本实现。

独立Standards另以六条真实入口确认complex ndarray会丢虚部并产生与实数逐位相同的结果（P2），RED/source bytes保留在`standards/complex_ingress_red*`。修复由noise owner的`finite_real_array`统一外部real类型/有限性转换，operator和load入口复用；不在纯数值循环散落检查。修后新增八项meaningful回归，重新绑定实际producer/source并验收同一公开输入，不借旧capsule追认新源。

真实小范围入口为`experiments/probe_conditional_operator_native.py`：直接读A01官方MAT（SHA cdf9bca4f48c61ee9c9ba998382a4e84392918c9dbc5a43f45ee0052614e25bd），256Hz、权威原生八导、右耳参考，保持distributed processed EEG身份。C selection00–01共240draw，held selection02共120draw；明示sample×physical-channel线性map，conditioner在onset[-32,-8)，response在[64,160)，无本仓库额外filter/FFT/resample。每draw保存absolute supports、C/held cutoff、available_at、原window与feature SHA，noise/均值只由同C拟合。

conditional及同C marginal两arm均120输出逐值保存重放、模型预测前后字节不变；AUC读数为0.836/0.833，固定等prior NLL为0.540025/0.547431。仅作为一个subject的binary ERP真实caller验收，无公共收益、九选一hit、未知边界state或成人确认结论。实际执行从物理冻结materialized源码运行，源码/数据守卫与receipt保留；重跑同输入是源码验收，不增加独立样本。完整矩阵与unknown-state识别仍由主线另行验收。

最终质量：作者37项定向、独立Spec14项、Standards50项通过；完整主树2670通过、19跳过、0失败/错误，Ruff/diff-check0、413文件byte guard不变。Standards复现P2：复杂数数组被float cast静默丢弃虚部，真实CLI仍successful。修复将`finite_real_array`集中在noise owner，operator复用实际ingress；原6条RED同输入全部GREEN拒绝，新增8正规反例，数值核未散落重复防御。

独立两轴分别从原SciPy MAT重建360窗口、实际maps/labels和240C/120held绝对支持，核源码/原数据守卫、预测无held标签以及两包只读重放；dense Gaussian预测最大误差不超过1.53e-14。修前RED、namespace/sourceguard阻止的无效运行及修后私有capsule分别保存，不借旧source结果追认。最终source为noise `fe393308b556265f4fcce5df5a5002068744d6899a2bfb355d4674e0c2919e11`、operator `e8540d6d73864e65078b4c4b4b6c2ca85da696ba38aa1e4fce566e4b88aeb75d`；LF属性保证跨checkout实际字节稳定。完整[质量凭证](../assets/prior_control_20261006/quality_receipt.json)保留两个axis独立结论和原始收据指纹，集合重叠不相加。
