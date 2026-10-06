# 固定前缀条件AR：数学、梯度实施与代码质量

更新：2026-10-06。承接[15块科研方案](deep_research_plan_20261005.zh.md)的B05及[多专家主线](ensemble_implementation_20261005.zh.md)。实施起点为`096bd2a`的干净隔离树，joint条件模型已在`6c7c930`提交；后续稳定QR与精确梯度见下文。改前源码已有本地物理冻结。原`D:/BCI/n2p3-net`并发工作、数据和历史产物保留。本记录随真实验收补充，不把软件或吞吐探针写成M1效能结果。

## 数学合同与代码落点

每个独立record读`[0,N)`，prefix为`[0,32)`，scored tail为`[32,N)`。六阶AR0/1/4/8/16/32使用同一readscope和tail；segmented Burg初始化仍有不同lag-pair数，范围另记。noise用stationary PARCOR参考建立prefix分布；同一full-record的W同步作用于Y、响应设计D和nuisance Z，不能先裁tail再白化。

令联合state为一个共享kernel B与每record自己的diffuse intercept/trend。MN先验column covariance为Σ，kernel row precision为ridge I，drift precision为0。所有C prefix的联合设计A_u给出

\[
P_u=P_0+A_u^\top A_u,\quad M_u=P_u^{-1}(P_0M_0+A_u^\top Y_u),\quad
Y_s\mid Y_u,C\sim MN(A_sM_u,I+A_sP_u^{-1}A_s^\top,\Sigma).
\]

prefix nuisance必须满列秩，使joint posterior proper。drift在prefix居中尺度且full/prefix共用坐标；prefix→tail顺序augmented QR计算信息root，conditional scatter Q为最终残差平方和加prefix到full参数差平方和，避免Gram平方条件数和full/prefix二次型相减。row logdet来自两root的log对角差；`Σ_profile=Q/n_tail`。AR目标为`0.5[n_tail log|Σ_profile|+C log|V_tail|]`，最终B posterior块来自全部授权C-prefix和tail。只计tail密度不等于只从tail估kernel；C-prefix标签/事件仍必须授权。

held评分重新条件化其真实prefix，但不修改冻结C模型、noise、温度或其它统计。AR0的Markov noise没有past依赖，未知kernel/drift仍可利用prefix；原设计对此的过强措辞以此合同更正。固定π(q)乘conditional-tail likelihood是明确的候选程序；若prefix design依q变化，它不同于full-data Bayes posterior，后者还需要π(q|prefix,C)。当前BNCI pre-roll51点，decision首onset51，prefix32内该decision设计无候选差异；外部history由原MAT完整flashledger的邻域准入负责，不能泛化为任意录制的证明。

主要owner：`continuous_decision`的joint posterior/conditional prediction与profile；`TemporalNoise.from_conditional_fit`保存真实Burg初始化和实际最终PARCOR；`CalibrationExpertProgram`冻结prefix/estimator，`_FittedExpert`核actual basis/order/ridge/support；model.digest覆盖完整实际模型身份。model/noise采用`/4`，旧clipped AR1 `/2`与stationary `/3`语义保留；历史严格source绑定包在对应frozen源上重放，未宣称跨版本自动兼容。

## 实际入口与失败账本

现有MAT runner接受`comparison_role=fixed_prefix_ar_order`，要求六阶×`burg`/`conditional_profile`完整roster、prefix32及相同basis、ridge和response support。Burg-only保留实际初始化shape；另一臂以η→tanhκ优化同一个精确条件目标。L-BFGS-B固定`eta_max=8, maxiter=1000, maxcor=32, ftol=1e-10, gtol=1e-6`，使用CPU/float64精确autograd，不放宽原停止门限。达到数值停止条件不证明局部或全局最优。

`continuous_gradient.ConditionalGaussianObjective`拥有输入常量，保留独立record lattice、缓存zero drift块，不修改global threads/device。它返回每tail行原始profile目标与精确η梯度；调用者只在优化时减去Burg起点常数，初末点与独立NumPy实际Gaussian目标核对。生产solver声明绑定helper实际源码SHA。求值账本包含初始化、每次value/gradient、终点核对和最终posterior，最后计数遗漏经RED复现后修复；一次AR1例的实际10次与记录10次一致。

模型/包/recipe/source与实际noise strategy一致；solver method从实际strategy与order派生，AR0不伪称调用非线性shape optimizer。合法不可识别、未收敛保存fit_failure并保留requested分母；合同错误传播并落父/子failed。每cell分别保存planned tail与实际read/condition/scored字段，失败时实际tail为None、实际scored样本0。

## 软件验证与独立代码质量

- 独立理论oracle55项通过；joint-state、分层条件公式和dense full/prefix ratio三路一致，六阶最大density误差`1.42e-13`。
- 独立public Interface61项通过：六阶多record kernel/Σ、不同N与record置换、AR0 prefix、单位/gauge、right-censored C与held拒绝、readonly状态和保存重放。复数basis静默丢虚部已先复现再修复，均值与covariance也在同一real入口拒绝。
- noise producer新/旧64项通过；15种旧order/startup组合与基线数值位级一致。linked集成148项通过。
- 第一轮完整pytest2498通过、19跳过、0失败/错误，328.417秒；391文件在完整测试前后逐byte守卫不变。其实际source为`c058aec0...` continuous与`9fc7d122...` temporal，后续estimator/solver关系修复另做定向复验，不能把此全集包装成后续源上重跑。
- Standards独立确认estimator自述与actual noise optimization strategy可错贴；实际AR0又误称L-BFGS-B。已统一派生/载入核对；Burg与AR0不伪称调用gradient或保存helper身份。
- 稳定QR定向56项、gradient正式39项与fixed-prefix关联21项均通过；190项linked集成通过。独立梯度误差常规不超过`2.32e-15`，真实Burg8/32不超过`9.59e-11`，高精度中心差分不超过`4.56e-8`。
- 后续完整pytest为2540通过、19跳过、0失败/错误，共2559项，455.052秒；393文件前后byte守卫不变。实际continuous SHA为`4b99ddc11c2f29bc330f75b59d5f3f870dcde10b6268c73e5c9aedc038cfd20f`，helper为`6c03c259a9536c8111f875903cf341b2f697f78eaa5e45b9b6d59a440f385da5`。
- 最后的求值计数修复将continuous SHA改为`e982d26da8b0658d0ccb5895903bcc930c4286b01a5a5e24f8c495c1218f71a7`，86项定向通过。未在此delta后重跑全集；独立Standards另在提前物理捕获且运行中不变的最终源码上通过173项关联测试。
- Standards最终通过：剩余发现0、判断性smells 0。核对λ/Σ关系、记录边界、只读并发、helper SHA/wrong-tag拒绝、Burg/AR0自述、整包同字节重放与真实求值数；不能据此宣称M1效能或整体项目已完成。Spec与高相关AR数值检验独立保留，不用Standards代替数学结论。
- Spec/math最终通过，当前审查范围无剩余确认公式或模型缺陷。重新执行的公开生命周期61项、dense/Schur/80位高精度23项、当前计数源6项分别留源；真实κ8/32方向导数差不超过`1.12e-10`，匹配AR32平稳样本的logevidence绝对误差不超过`1.73e-7`（reference约48746）。

独立Spec保留一个通过numeric ingress的off-model iid尺度压力失败：reference density约`−7.9e12`，float64 absolute误差24.72–259.58，relative不超过`3.30e-11`；原固定absolute断言仍false。所测三个候选虽给相同概率`[0,1,0]`，log-BF误差约235，不能据relative小认证任意近tie或极端输入的NLL。该限制与匹配生成样本的验证分列，没有用合法样本通过抹去原失败。

文档链接核查另发现隔离树遗漏六份原有设计/manifest附件；已从原树按同字节复制补齐并逐SHA核对，原件未改动。这是导航修复，不重标历史设计或执行身份。

原始证据在本地`tmp/continuous_fixed_prefix_20261005/`：theory_review、callchain_review、noise_state、gradient_module、optimizer_diagnosis、standards_review、各JUnit/log与完整source/receipt。原始报告、src执行副本、RED脚本/结果保留；[紧凑验收账本](../assets/continuous_fixed_prefix_20261005/quality_receipt.json)保存版本/范围，原始证据物理封存与清单另记。以上测试范围有重叠，不相加。

最终独立数学出口另见[61/23/6分组凭证](../assets/continuous_fixed_prefix_20261005/final_math_receipt.json)。代码与指南提交`fe993c8033368f8b9b5849980751c328d6b420fc`；[物理执行源清单](../../frozen/continuous_fixed_prefix_20261005_delivery/manifest.json)的477成员已对照实际bytes、解包副本与镜像逐个验证。
[质量归档清单](../../frozen/continuous_fixed_prefix_20261005_delivery/quality_manifest.json)包含647成员、73012848压缩字节；每成员、原件前后与镜像SHA均一致，活动M1输出明确排除。Git只跟踪清单，原日志、模型与压缩包继续本地保留。

## 真实路径性能与M1执行合同

真实A01路径仅是吞吐/重放探针：600授权labels、5×7975×8 raw fit，held一组只供无标签score/reload。单线程Burg AR0/8含保存/评分/重载约0.686/1.538秒。原forward-FD AR8触达200次上限、exit1，失败收据保留。精确梯度AR8在136迭代达到停止条件，fit50.955秒、总54.014秒；最终1000/maxcor32生产AR32在261迭代达到停止条件，fit258.400秒、总264.035秒，保存重放精确一致。此AR32探针对应4b99源码，不重签为最后计数修复后的模型。

独立诊断确认默认forward-FD在δ=1e-8时gradient误差超过gtol；中心FD200也未收敛，AR8约4658次/417.8秒、AR32约15600次/4664秒，属于独立仪器窗口。独立exact-gradient AR32约308次/215.7秒达到ftol，projected gradient约`5.32e-5`且有微小负Hessian特征值，不能称最小值已证。exact-HVP trust-region更贵且失败、峰值约3.25GB，已否决该性能实现。不同source、callback、profile和并发窗口分别保存，不宣称统一倍速。

按`performance-optimization`与`get-available-resources`复核本地拓扑。数值窗口2workers×4threads为8.575次/秒，1×4为6.948、2×2为7.219；实际CPU约7.47核，双worker峰值1.939GB，数值差异最大`5.95e-14`。因此[完整M1计划](../assets/continuous_fixed_prefix_20261005/m1_plan.json)采用2个participant进程×4线程，入场available至少3GB；完整participant内存与总吞吐另由真实运行验收。

M1固定8人×6fold×六阶×两估计器，共576fit、每臂48个requested decisions；C800中720labels组成6个完整selection，fold内600训练/120held、80不用。共同prefix32、response1.2秒、basis8、ridge1；固定温度，baseline为ar8_burg。原MAT由canonical parser按八份原SHA准入，源码/计划由`n2p3_source_freeze/1`物理冻结并核对，从materialized源码执行，成功包逐cell保存重放，失败保留requested分母与未执行支持。全部12臂执行，不因pilot不利而减少。

本轮为单seed、target-only、公开校准开发。Windows本机venv为CPU-only Torch，未云端启动；云端25CPU/4GPU下限不改写成本地声明。M1需完整结束、核实际source/输入/覆盖/失败与participant-macro配对hit/NLL/Brier后另更新实测结论；四库/完整source→target seed、gating、其它机制与成人确认继续原依赖图。

M1已从上述materialized源码启动，命令/venv/原输入/plan SHA、资源快照与actual worker采样保存在本地`tmp/continuous_fixed_prefix_20261005/m1_20261006/`。该目录的完整终态、失败与风险结果另更新，进程启动不算矩阵验收。

## 2026-10-06：真实进程缺陷、包复用与续跑验收

第一次M1运行980.258秒后父进程exit1，错误为`BrokenProcessPool`。原worker的冻结元数据在pickle恢复时触发`TypeError: fitted metadata is frozen`；这不是拟合失败或完整矩阵结果。A02完整72格写盘，A01没有最终record但保存71包；父账本completed_subjects=0。独立按旧冻结源核验全部143包，program、600训练/120held/80unused、原manifest/model SHA、实际fit_digest与公开score重放通过，原件前后SHA不变。

修复将worker结果限定为已保存JSON的plain记录，内部state继续只读；真实双进程MAT入口在旧版先失败、修复后通过，14项关联检查通过。续跑使用[独立续跑计划](../assets/continuous_fixed_prefix_20261005/m1_resume_plan.json)，重新执行全部576逻辑cell，复用143个已验证包、剩433次新fit尝试。复用要求旧/新物理source freeze通过、每份`src/*.py`实际byte完全相同，科学/输入/资源计划一致；runner调度/通信来源另记。逐包重新核program、训练view和标签fit_digest、原model bytes，复制原byte后load/score/reload；未完成包新fit，完整但错误包拒绝且不fallback。每格保留physical attempt/success/reuse、原producer source与本轮score source，不重签原模型。

独立Standards另以合成真实包复现绝对model路径在拒绝前写到原件的问题；已在任何copy前限定relative basename并核destination parent，原byte与mtime均须不变。该发现和RED原样保留，不把先前通过解释为无进程或路径缺陷。
最终路径delta及真实双进程入口25项通过；紧凑源/验证分组见[续跑软件凭证](../assets/continuous_fixed_prefix_20261005/m1_resume_quality_receipt.json)。
独立Standards最终26项通过，无剩余已验证发现；原件实际write probe为0写入且mtime不变。另在真实双进程冷启动的合成resume入口复用4包、0次新fit、4次精确重放并传回plain JSON，原run所有byte/mtime不变。主线Spec核对完整576格、同src/配方/资源、来源分列和失败分母通过；两轴只确认实现，完整M1效能继续另验。

通信/续跑及B06完整映射的稳定候选上，完整pytest为2567通过、19跳过、0失败/错误，共2586项；396个受检源/测试/依赖文件前后byte不变，runner为`3a9f361661f81d9a1ed91475a08542d4a71a40e10a6a71dd2e26235ed8437411`。后续路径修复另有定向及独立复验，不能将该全集重新标为修复后执行。

第一次完整participant运行测得双worker RSS峰值1812799488字节，系统available最低129716224字节，实际CPU阶段波动；不将此前公平数值窗口的23.4%收益包装成此受共享内存压力影响的总吞吐。续跑入场提高到available至少4GB，仍为2×4，保留实际资源日志。完整M1结束后才进行participant-macro风险与全部失败/来源验收；[分析计划](../assets/continuous_fixed_prefix_20261005/m1_analysis_plan.json)固定8人cluster、20000次共享bootstrap与完整48概率缺失规则。

## 完整M1结果与下一步判定

完整续跑已结束，exit0、3008.964秒（约50.15分钟），487个冻结成员及计划前后bytes不变。独立验收确认576逻辑格、576成功包、576公开重放精确一致；143包复用、433新fit尝试且全部成功。A02原72格的保存证据与本次逐格完全一致；A01原无最终record的71包按包来源复用并重新评分。原件未重签，失败父运行仍单独保留。双worker RSS峰值1642909696字节、system available最低276201472字节；共享机器压力与阶段CPU波动仍存在，不据总墙钟反推公平数值窗口倍速。

每臂请求48个decision、8人，每fold600训练labels/120held，C800中80不用。全部12臂prediction/probability coverage为100%、fit失败0、tie0。以下风险有完整48概率支持，因此条件风险与完整请求风险数值相同；[结构化结果](../assets/continuous_fixed_prefix_20261005/m1_results.json)、[参与者表](../assets/continuous_fixed_prefix_20261005/m1_participant_metrics.csv)和[完整图](../assets/continuous_fixed_prefix_20261005/m1_ar_orders.pdf)保存全臂与不确定性，不能只报最好hit。

| 阶数 | Burg hit | Burg NLL | Burg Brier | 条件profile hit | 条件profile NLL | 条件profile Brier |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 70.83% | 25.226 | 0.574 | 70.83% | 25.226 | 0.574 |
| 1 | 58.33% | 3.016 | 0.621 | 47.92% | 2.524 | 0.658 |
| 4 | 66.67% | 6.919 | 0.590 | 64.58% | 5.417 | 0.657 |
| 8 | 66.67% | 6.151 | 0.569 | 33.33% | 3.438 | 0.963 |
| 16 | 64.58% | 7.946 | 0.601 | 0% | 3.698 | 0.979 |
| 32 | 62.50% | 7.308 | 0.583 | 0% | 3.578 | 0.974 |

预声明baseline为AR8 Burg。AR0救回3、损害1，hit增加4.17个百分点，但paired NLL增加19.074（participant cluster95%区间1.305–43.338）；hit差区间−4.17至10.42个百分点，不能以两个净正确decision宣布总体优势。AR4 Burg正确/错误集合与baseline相同，NLL/Brier变差。其它臂均降低hit；AR8 profile损害18、救回2，hit差−33.33个百分点（区间−43.75至−25）。AR16/32 profile分别损害baseline全部32个正确decision，coverage仍完整，所以它们的0%不是失败分母造成。

按“hit不降、paired NLL不增、baseline输出不丢”的开发点估计门，11个替代臂均不通过，未选择或晋升新程序。AR0虽有最高hit却有明显错时过度自信；高阶profile得到接近36类uniform的风险读数（uniform NLL约3.584、Brier约0.972），并未保留判别能力。降低C Gaussian目标/训练残差不等于提高产品命中，机制原因尚需噪声与ERP联合归因；不先把这一负结果归为代码缺陷或g-prior收益。

所有区间为固定fit outcomes、8个participant的20000次共享cluster bootstrap，不把六个相关fold当独立重复，不重训，不覆盖source随机性。AR16/32的经验hit区间[0,0]只反映已观察八人的全零结果，不能当未见人群成功率已确定。M1是公开C内开发，不是成人九选一确认。

另已对576成功模型完成truth-known residual诊断：训练按独立record投影drift，held用冻结C posterior加实际prefix的完整predictive row covariance与Σ白化；ACF/PSD不拼接records，输出创新协方差、条件数、预测二次型并与actual truth log-density核对。过程不fit、不改state、不用held选择，原始JSON与receipt留本地。完整source、模型/预测、重放审计、bootstrap索引和诊断继续物理归档，Git仅分发清单及紧凑表/图。B05逐通道谱、B06条件mean/Schur、B07–B11、K0、工作站和完整流程/成人确认仍为后续范围，不能将M1完成当全部15块已完成。

[完整终态物理归档](../../frozen/continuous_fixed_prefix_20261006_resume/complete_m1_manifest.json)已封存1268成员、84499150压缩字节；逐成员、原件前后与镜像SHA一致。原始M1失败父运行和300成员旧包归档继续保留，不以成功续跑覆盖原失败。图的PNG/SVG/PDF已检查标签、原始参与者点、区间与轴范围，表格提供数字替代。
