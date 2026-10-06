# 全C trace匹配isotropic prior对照

日期2026-10-06，开发树`codex/calibration-g-prior`，本切片基线`108c6b1`。承接[g-prior第一切片](calibration_g_prior_20261006.zh.md)及其修后Spec/Standards验收；原RED、初版127与修后143回归、原源码快照分别保留。当前只做代码与合成软件验证，没有启动公共或云训练，也没有改签冻结M1。

对照问题是区分prior几何与总reference预测能量。使用完全相同的授权C、basis坐标、AR0单位innovation reference和逐record diffuse intercept/trend，定义

\[
G=\sum_r\bar D_r^\top\bar D_r=R_C^\top R_C,\qquad
H_g=gG^{-1},\qquad
\lambda_C=\frac{\operatorname{tr}(G)}{kg},\qquad
H_{iso}=\lambda_C^{-1}I_k.
\]

两臂均有`tr(GH)=kg`，对应同一个reference上的每个白化输出通道的期望预测能量。匹配这一标量不会使两臂的协方差几何、final fitted-noise能量、有效自由度或性能自动相同。g增加使两臂H按相同比例增加；g不是1/λ，λ_C还依赖actual C与坐标尺度。

canonical kind为`calibration_trace_isotropic`，接口`KernelPriorSpec.calibration_trace_isotropic(g)`；wire只提供g及固定reference/support/nuisance/rank/mean字段，不提供precision/ridge。`transfer.kernel_prior`使用已有全C薄QR root的Frobenius norm生成`√λ_C I`，不重建G；`CompiledKernelPrior.derived_precision`只读冻结isotropic root，g-prior返回None。纯数值kernel的kind分派在owner内，callers只消费一次compiled root。

isotropic control在预声明的basis坐标内定义，明确不具一般可逆换基不变性。g的对应prior可按`T⁻¹H T⁻ᵀ`变换；isotropic臂在新坐标重新匹配trace后仍是scalar I，所以通常不同。归因比较必须固定同一basis、实际C、noise程序、conditioning=32和g，不能借trace匹配混用坐标。

两臂共享full-rank数值准入及original-design projection resolution。即使singular G仍有正trace，control也不单侧继续：完整paired g-prior不能定义，应保持`KernelPriorRankError.record()`的kind/spec/rank/source/binding。零reference与纯nuisance同样明确失败；没有pinv、completion或默认λ fallback。这个准入规则不宣称统计识别。

初始化响应拟合、真实segmented Burg residual、NumPy joint prefix/tail QR与Torch exact objective使用相同root；η循环不重造G或λ。Σ仍以Q/nTail profile，final kernel posterior读全部授权C。C-only是设计与授权标签范围：λ_C可依赖C的onsets/labels/basis，但不依赖C电压值、held signal、held真值或held候选。

g模型仍是`n2p3_continuous_decision_model/5`，trace control独立`/6`；两者均有九个numerical数组：四个base state、三个actual prior/reference/original-design roots、两个final-C posterior statistics。load核schema-kind、实际root/reference/g、rank/source/C/basis/input及posterior关系，重签wrapper不能把/5与/6互换。原source身份仍按实际执行bytes，不给旧模型或M1重新签名。

ensemble program/producer保存canonical kind/g与实际编译状态；grouped-temperature OOF每fold仅从其训练C编译，held record排除，final full-C另编译。source-training记录保持本身授权scope。保存模型的未知目标score只消费冻结posterior；改变held电压或候选设计也不触发compiler。

正式反例覆盖独立dense trace能量/Σ/posterior核对、g尺度、一般换基、singular/zero lineage、同derived λ的实际init/Burg/final fit、central-difference gradient、held无重建、source身份、OOF排除以及真实NPZ与ensemble保存重放。精确JUnit、命令/返回码和源码SHA记在本地`tmp/prior_control_20261006/`；独立两轴质量与完整suite由主线调度。软件测试不代替完整公共归因矩阵或新成人效能确认。
