# 算子科研末轮规格复审（Spec轴）

日期：2026-10-05。复审基准为`frozen/system_implementation_20261004_after`，以及Stage1/2/3实际运行目录。此次复审只读，不修改`src/`、GLS、runner或SelectionERP wrapper；目标是判断声明的科学合同是否真的进入执行与消费，不以测试全绿替代规格审查。

## 结论

主要实验边界基本闭合：有限滤波是确定性输入算子；stationary、conditional-prefix和stationary-innovations是不同启动估计量；Stage3实际每个fold拟合600个标签、授权C800但保留80个未使用标签，held selection不进入fit；先验尺度确实进入GLS的设计Gram求解并写入保存记录；四个成熟ERP锚点使用同一selection原始支持，成功192/192，原owner逐trial logit复现一致。

仍有两个已复现的消费层缺陷，以及一个真实数值缺陷：

1. `OperatorGLS.load`和`TemporalNoise.from_record`允许把训练范围、实现SHA和先验语义改成伪造的、但重新签过哈希的元数据后继续加载。它们检查JSON自洽性，没有检查SHA是否等于当前实现，也没有对GLS provenance的`fit_scope/records/kernel_prior_scale/ridge`做完整语义校验。外部本次run的`source.json`和结果封存仍保护了历史成绩，但独立模型artifact无法自证自己的fit来源。这是**P1接口缺陷**。
2. Stage3的16个失败全部是`raw_ar32_lattice_marginal3`，不是PARCOR不稳定：`D^TD`条件数可达`6.69e14`，`np.linalg.inv`后对称化的row covariance出现最小特征值`-6.11e-10`，随后Cholesky拒绝。应使用增广设计的QR/SVD或稳定Cholesky解，不应把该臂的请求失败直接解释为算法无效。这是**实现数值缺陷，且使该预声明臂没有完整结果**。
3. 成功的stationary-innovations模型浮点条件指标最大`2.91e-8`，低于计划的`1e-6`警戒阈值；这只说明保存的lattice factor在当前数值尺度下未触发警戒，不证明高阶AR具有真实EEG生理解释。

## 已核对且满足规格的部分

### 1. 有限算子和状态

`FiniteFilterOperator`的`apply`、`zero_state`、`state_basis`和`decompose`均由给定SOS和观测段确定，没有把padding state当作额外随机变量。GLS的`filter`和`filter_design`分别决定对观测和设计矩阵是否作用同一F；其`operator.record()`随model保存。当前Stage3的raw臂不把无限长滤波输出冒充短段结果。

`TemporalNoise`的`stationary`使用单位innovation的离散Lyapunov启动协方差；`conditional_prefix`明确只返回观测前缀后的条件创新，并要求不同的输入语义；`stationary_innovations`使用实际Burg PARCOR lattice保留全部样本，expanded polynomial root仅作诊断。独立密集Gaussian、AR3和100位高精度重复极点反例均支持这些数值等价关系。conditional-prefix没有混进Stage3，这一点正确，因为它会改变估计量。

### 2. 均值、协方差和先验尺度

GLS拟合先在同一个`project(noise, design)`/`project(noise, samples)`坐标中估计均值，再从相同残差估计channel covariance；score沿同一noise、均值、row covariance和channel covariance计算。没有看到把有限F额外随机化或把状态Gamma再加一次的路径。

`kernel_prior_scale="innovation"`使用`ridge=1`；`"marginal"`保存每轮`marginal_variance_per_unit_innovation`并使用`ridge/r0`。对实际A01/fold0的raw_ar8_marginal3重算，保存的均值与独立设计Gram求解一致，`lambda_eff=4.220635972358091e-6`确实进入了求解。Stage3计划和保存provenance都记录了该选择；不能把两种先验当同一个算法。

### 3. 真实标签和支持

`run_calibration_study.py`从原始MAT字节校验C800，在每个完整selection构造独立raw范围，拒绝相邻未知响应重叠；前六个selection提供720授权标签，其中每个LOO fold训练5组=600标签、held组120标签，最后80个授权标签保留但不使用。训练`SelectionERP`和`OperatorGLS`只接收5个CalibrationRecord；truth仅在score返回后生成NLL/正确性字段。

Stage3请求768个cell，结果为752 completed、16 failed、0 not-executed；失败集中到单一`raw_ar32_lattice_marginal3`臂。不能把768写成768成功，也不能把16失败当作该方法的零分预测。

### 4. 成熟ERP强锚点

`SelectionERP`复用已有`target_shrinkage_features`/`fit_shrinkage_feature_model`和`StructuredERP`，没有另写同名仿版。真实source区间为`[-51,+306]`共358点@256Hz，shared FFT先产生179点，再按真实`PreprocessingSpec`保留178点；不读取sample+307。mean-only baseline使用26个输出点；Toeplitz使用`bin_width=4`并固定丢弃尾2点；xDAWN要求真实观测秩≥4。

四个软件正确性臂`waveform_lda/mean_only`、`waveform_lda/none`、`toeplitz_lda/none`、`xdawn_erp/mean_only`均通过原owner逐trial logit复现和NPZ重载。它们在Stage3已完成192/192请求，但这些是within-participant selection开发锚点，不能冒充跨库或成人性能。

## 反例与边界

### A. 可重签伪造训练来源（需修）

以下操作在内存完成，没有修改历史运行文件：

```text
OperatorGLS.load:
  provenance.fit_scope = "target_test"
  provenance.records = []
  provenance.ridge = 123456
  provenance.kernel_prior_scale = "fabricated_prior"
  重新计算 artifact_sha256
  -> 仍然加载成功

TemporalNoise.from_record:
  合法格式的 source_training ledger
  fake record_id / fake input SHA / implementation_sha256 = 64个0
  重新计算 artifact_sha256
  -> 仍然加载成功
```

当前构造函数只要求元数据有正确字段和哈希形式，没有要求`implementation_sha256 == 当前源码摘要`，也没有要求GLS的保存provenance与noise record、实际先验参数和拟合范围互相一致。解决时应保留历史记录原样，通过显式schema升级让加载器校验：当前consumer实现SHA、固定`fit_scope=target_calibration`、非空且完整record ledger、每轮实际`effective_ridge`、`kernel_prior_scale`、noise实现和operator identity的联动；不能依靠调用方的外部source封存代替模型自身门禁。

### B. 高阶AR后验矩阵失去正定（需修）

以A01/fold3的失败cell重放：`raw_ar32_lattice_marginal3`的`r0=3.669309e6`、`lambda_eff=2.725309e-7`，设计矩阵条件数约`2.74e7`，`D^TD`条件数约`6.69e14`。理论上加正的lambda后H仍正定，最小特征值约`2.51e-6`；但`np.linalg.inv(H)`的浮点误差把对称covariance最小特征值推到`-6.11e-10`，`OperatorGLS.__post_init__`的Cholesky遂失败。增广设计的谱条件约`2.59e7`，说明QR/SVD或带pivot的Cholesky能够避免平方条件数。

本结果只能支持“当前实现对该高阶先验数值不稳”，不能支持“stationary innovations/marginal先验科学路线无效”。修复后必须完整重跑16个原预声明cell，并保留原16个失败账本。

### C. 浮点警戒不是性能门

成功lattice cell的`floating_precision_condition_indicator`最大约`2.91e-8`，均低于`1e-6`计划警戒；这是数值稳定诊断，不是命中率或NLL收益。当前Stage3属于顺序开发，不是盲法确认；所有臂固定报告，未用held选择超参数。

## 规格裁决

| 项目 | 裁决 |
|---|---|
| 确定有限F/观测state | 满足；无需额外Γ随机化 |
| raw/finite均值和cov同步 | 满足当前GLS路径；高阶后验求解仍有数值缺陷 |
| stationary/conditional/lattice估计量区分 | 满足；conditional未进入Stage3 |
| 600标签与C800授权 | 满足；每fold训练600、剩余80未用 |
| prior λ/r0进入solver和重载 | 满足有效成功臂；重载语义可被重签metadata伪造 |
| 稳定性和precision边界 | PARCOR与lattice反例通过；16个高阶marginal cell因normal-equation数值失败 |
| 成熟ERP强锚点 | 满足输入/owner/grid/LLR接入，192/192软件验证/执行完成 |
| 整体收益 | 未证明。Stage1/2已有GLS改善NLL但hit下降；Stage3成熟锚点与连续臂仍是within-participant开发诊断 |

本审查不把已发现的未知漏洞隐藏在“source.json已封存”之后，也不把两个明确实现缺陷扩大解释成全部算子科学失败。下一步先修artifact语义门禁与稳定solver，再按原计划重跑受影响的16-cell；性能与产品晋升另需独立匹配矩阵。
