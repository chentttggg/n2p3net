# 固定前缀条件AR：数学、实施与真实路径诊断

日期：2026-10-05。承接[15块科研方案](deep_research_plan_20261005.zh.md)的B05及[多专家主线](ensemble_implementation_20261005.zh.md)。实施起点为`096bd2a`的干净隔离树，改前源码已有本地物理冻结。原`D:/BCI/n2p3-net`并发工作、数据和历史产物保留。本记录随真实验收补充，不把软件或吞吐探针写成M1效能结果。

## 数学合同与代码落点

每个独立record读`[0,N)`，prefix为`[0,32)`，scored tail为`[32,N)`。六阶AR0/1/4/8/16/32使用同一readscope和tail；segmented Burg初始化仍有不同lag-pair数，范围另记。noise用stationary PARCOR参考建立prefix分布；同一full-record的W同步作用于Y、响应设计D和nuisance Z，不能先裁tail再白化。

令联合state为一个共享kernel B与每record自己的diffuse intercept/trend。MN先验column covariance为Σ，kernel row precision为ridge I，drift precision为0。所有C prefix的联合设计A_u给出

\[
P_u=P_0+A_u^\top A_u,\quad M_u=P_u^{-1}(P_0M_0+A_u^\top Y_u),\quad
Y_s\mid Y_u,C\sim MN(A_sM_u,I+A_sP_u^{-1}A_s^\top,\Sigma).
\]

prefix nuisance必须满列秩，使joint posterior proper。通过augmented QR求M_u和row root，以Woodbury正平方和求conditional scatter Q及row logdet；`Σ_profile=Q/n_tail`。AR目标为`0.5[n_tail log|Σ_profile|+C log|V_tail|]`，最终B posterior块来自全部授权C-prefix和tail。只计tail密度不等于只从tail估kernel；C-prefix标签/事件仍必须授权。

held评分重新条件化其真实prefix，但不修改冻结C模型、noise、温度或其它统计。AR0的Markov noise没有past依赖，未知kernel/drift仍可利用prefix；原设计对此的过强措辞以此合同更正。固定π(q)乘conditional-tail likelihood是明确的候选程序；若prefix design依q变化，它不同于full-data Bayes posterior，后者还需要π(q|prefix,C)。当前BNCI pre-roll51点，decision首onset51，prefix32内该decision设计无候选差异；外部history由原MAT完整flashledger的邻域准入负责，不能泛化为任意录制的证明。

主要owner：`continuous_decision`的joint posterior/conditional prediction与profile；`TemporalNoise.from_conditional_fit`保存真实Burg初始化和实际最终PARCOR；`CalibrationExpertProgram`冻结prefix/estimator，`_FittedExpert`核actual basis/order/ridge/support；model.digest覆盖完整实际模型身份。model/noise采用`/4`，旧clipped AR1 `/2`与stationary `/3`语义保留；历史严格source绑定包在对应frozen源上重放，未宣称跨版本自动兼容。

## 实际入口与失败账本

现有MAT runner接受`comparison_role=fixed_prefix_ar_order`，要求六阶×`burg`/`conditional_profile`完整roster、prefix32及相同basis、ridge和response support。Burg-only保留实际初始化shape；另一臂以η→tanhκ优化同一个精确条件目标。代码当前使用声明的有限L-BFGS-B搜索域/预算，记录local convergence，不称全局MLE。

模型/包/recipe/source与实际noise strategy一致；solver method从实际strategy与order派生，AR0不伪称调用非线性shape optimizer。合法不可识别、未收敛保存fit_failure并保留requested分母；合同错误传播并落父/子failed。每cell分别保存planned tail与实际read/condition/scored字段，失败时实际tail为None、实际scored样本0。

## 当前证据与尚未关闭项

- 独立理论oracle55项通过；joint-state、分层条件公式和dense full/prefix ratio三路一致，六阶最大density误差`1.42e-13`。
- 独立public Interface61项通过：六阶多record kernel/Σ、不同N与record置换、AR0 prefix、单位/gauge、right-censored C与held拒绝、readonly状态和保存重放。复数basis静默丢虚部已先复现再修复，均值与covariance也在同一real入口拒绝。
- noise producer新/旧64项通过；15种旧order/startup组合与基线数值位级一致。linked集成148项通过。
- 第一轮完整pytest2498通过、19跳过、0失败/错误，328.417秒；391文件在完整测试前后逐byte守卫不变。其实际source为`c058aec0...` continuous与`9fc7d122...` temporal，后续estimator/solver关系修复另做定向复验，不能把此全集包装成后续源上重跑。
- Standards独立确认estimator自述与actual noise optimization strategy可错贴；实际AR0又误称L-BFGS-B。现已统一派生/载入核对，完整最后复验随候选稳定追加。

原始证据在本地`tmp/continuous_fixed_prefix_20261005/`：theory_review、callchain_review、noise_state、standards_review、各JUnit/log与`full_suite_source.json`/`full_suite_receipt.json`。原始报告、src执行副本、RED脚本/结果保留；交付时物理封存和清单另记。以上测试范围有重叠，不相加。

真实A01路径仅是吞吐/重放探针：600授权labels、5×7975×8 raw fit，held一组只供无标签score/reload。单线程Burg AR0/8的fit约0.228/0.341秒，含保存/评分/重载约0.686/1.538秒（含profile仪器，不是正式跨环境速度承诺）。优化AR8触达200次迭代上限，进程exit1；失败收据与原log保留，未被当成完成或效能收益。

独立诊断已定位默认forward-FD在δ=1e-8时的gradient误差超过gtol；等价prefix居中尺度可改善drift/root条件数，exact autograd路线正在与同数据/预算/目标核对。当前未执行完整8人×6fold×六阶×两估计器M1矩阵，不从该探针选择算法或宣布泛化提升。下一步先稳定数值梯度和实测总墙钟，再冻结执行源、运行完整M1；四库/完整source→target seed、gating、其它机制与成人确认仍继续原依赖图。
