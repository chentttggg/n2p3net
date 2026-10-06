# B08：同 C 参考能量匹配的先验对照

日期2026-10-06。承接[固定C-reference第一切片](calibration_g_prior_20261006.zh.md)和[愿望清单B08](deep_research_plan_20261005.zh.md)。实施与两轴代码质量验收已完成，公开开发矩阵另记；本页锁定数学问题、矩阵与分析规则。M1及其原保存包继续按原执行源解释，不加载或重签为新先验程序。

第一问题是固定授权C中的信息几何是否影响候选判别；第二问题是预测能量的先验强度是否影响hit与概率风险。仅用g-prior对ridge1，两个因素同时变化，不能归因于先验形状。

同一个全C、AR0单位innovation、逐record独立diffuse affine nuisance剔除后的设计给出 `G=R_CᵀR_C`，核维数为k。固定g-prior为 `H_g=g G⁻¹`，参考能量 `tr(G H_g)=k g`。各向同性控制为 `H_I=I/λ_C`，其中 `λ_C=tr(G)/(k g)`，因此 `tr(G H_I)=k g`。该控制明确固定在声明basis坐标；g的通用换基不变性不扩展给它。

canonical `calibration_trace_isotropic` 程序由同一 `transfer.kernel_prior` owner编译，caller不能重建G或外部指定派生λ。g与control使用同full-rank准入，秩不足各自记录具名识别失败；这避免零信息C在一侧得到人为有效模型。匹配的是完整授权C参考能量，既不是逐held候选能量，也不是已拟合噪声中的能量或实际effective degrees of freedom。η优化期间prior不更新。

## 冻结开发设计

公共BNCI A01–A08；每人C800内六个完整selection，720标签使用、80不用。每fold600标签训练、120 held；相同物理decision、全C支持、事件历史、basis8（共同与target两部分合计k=16）、response1.2秒、observation308/256秒、pre-roll0.2秒和fixed32。target-only确定性seed20261006，不是完整source→target随机性复核。

噪声程序预声明AR0 Burg、AR8 Burg、AR8 conditional profile、AR32 conditional profile：分别提供无Markov噪声、M1基线、同阶联合优化和高阶失效端点。每个噪声程序固定七种prior：ridge1；g∈{1,9,99}；各自能量匹配isotropic control，共28臂×48fold=1344逻辑fit。所有臂均执行和报告，不由held结果选g、删人、提前停止或增加“最好”臂。

主要形状对照是每个噪声程序内12组g对其同g control；收缩强度对照在同一种prior内横比三个g，单独解释。AR8 Burg ridge1为同批总基线；以前M1数值不充作新基线。每臂输出单一专家、固定温度1，不在此panel加入混合或C-inner温度变动。

requested participant/decision分母固定，失败、弃权、平局与概率缺失保留；主要指标participant-macro operational hit。NLL、Brier、coverage及baseline成功输出丢失同列。若全部48概率可用，才给完整requested风险；配对风险只在双方实际概率交集计算，并明确缺失账本。候选下一阶段门仍为hit不降、paired NLL不增且baseline输出不丢失；这是开发点估计门，无成人效能含义。

分析按participant整簇、共享索引的20000次bootstrap，seed20261006、95%百分位区间；六个重叠训练fold不作为独立参与者。区间描述固定fit结果的参与者重采样，不覆盖source训练随机性；全28臂与12组形状对照均展示，不作显著性筛选或普遍因果结论。

## 实施与验收入口

`KernelPriorSpec` → 全C compiler → initialization/Burg → NumPy条件QR/Torch精确梯度 → 实际model与package → `run_candidate_ensemble_study.py` 的 `fixed_reference_kernel_prior` panel → 原requested指标owner与独立重放分析。

counterexamples检查reference能量恒等式、g与λ语义、basis变换、训练内隔离、rank failure、真实保存重载、源码/actual-C交叉绑定。runner明确拒绝不完整28格、重复cell、不同prefix/basis/response、错误g/λ和额外温度/混合。真实矩阵收据独立追加，不以代码通过表示已经训练。

## 代码质量验收

作者173项关联验证、独立Spec15项与Standards80项均通过，遗留确认缺陷与需整改代码味道均0。独立named-panel fixture确实经过MAT/fit/save/score入口：28臂56输出与12形状对照，其中44个实际零设计rank failure保留完整actual-C/source及requested分母。集合重叠，测试计数不相加。

主树合并B06后的完整pytest为2670通过、19跳过、0失败/错误，共2689项；Ruff/diff-check通过，413个源/测试/入口/指导文件byte guard不变。[质量凭证](../assets/prior_control_20261006/quality_receipt.json)绑定两轴、作者、原并发source-guard失败与有效复验。原失败源/日志和M1实际包保留；source checkpoints只表示父版本，实际SHA仍绑定执行byte。

本轮以canonical prior owner集中kind/reference规则、一次编译root、多消费者共享纯数值路径，未在caller复制G或λ推导。软件质量不建立公共收益或成人确认；完整计划见[1344格开发矩阵](../assets/prior_control_20261006/plan.json)和[分析规则](../assets/prior_control_20261006/analysis_plan.json)。

## 真实矩阵暴露的数值核验缺陷

首轮从435成员物理源启动2workers×4threads，在A02/fold0/AR32 profile g99触发后验关系ValueError。原Pool在父级取得exception后等待其余队列，已有worker继续其他participant；本轮核实进程身份后受控停止，101原包保留，退出15、源码与计划byte不变。原progress仍running不构成完成，独立分析器明确返回controlled_stopped/metrics_finalized=false。

一次必要的实际失败重现保存5个授权C的数组、design、prior/noise/M/H/statistics，83.52秒、原gate RED，未发布成功模型。实际W下独立simultaneous QR、Householder nuisance quotient和SVD支持数值误判：原均值绝对差8.53574e-17，固定2e-9 coefficient阈值却被相对量2.09576e-9越过；kernel信息root condition约1.0195e7，normal equations condition约1.039e14。posterior-standardized均值差平方约1e-16；同held全部36候选winner一致，归一化logp变化不超过9.43e-8。这个单格界限不追认为所有格效能。

修复使用nuisance-block QR，保存信息root与正交响应`Q_DᵀY`，method明确新语义；核验augmented QR的`R M=b`后向误差，未改实际fitted M/H、noise优化或H的原2e-9关系限。原捕获后向误差9.85e-16，低于`64 eps k=2.27e-13`；错误均值、H×1.01及真实g99改g9而保留旧posterior仍被拒绝。worker改滚动ready任务及共享abort signal，合同故障后peer在下一fit前停止，不将合法typed rank failure当父故障。

40项必要关联验证、真实spawn成功/停止、Ruff/diff-check通过，按用户要求不重复全集代替源码阅读。原2670/19完整测试与435源属于修前交付；修后源与独立两轴另记。[同1344格新计划](../assets/prior_control_20261006/plan_v2.json)只改变执行源清单，科学/资源字段不变；旧101包不重签或复用成新实现。完整新矩阵仍按原对照和失败分母执行。

独立Spec和Standards修后均通过。Standards另实际复现parent KeyboardInterrupt未落failed状态的P2，最外层与subject现在落盘并原样传播；同spawn边界RED→GREEN和三个进程行为回归通过，遗留确认缺陷0。actual A02/fold0/AR32 g99从[新436成员物理源](../../frozen/prior_control_20261006_execution_v2/manifest.json)重新fit/save/load/score成功，112.40秒、397求值/324迭代、精确重放；原捕获仅用于验证，未改签成功模型。完整新1344格已从该materialized source实际启动，结果结束后再验收。

[首轮停止与修复证据归档](../../frozen/prior_control_20261006_original_failure/manifest.json)逐成员验证710个成员并核镜像，含101旧包、真实RED、数值比较与修后两轴；原件保留，payload仅本地。原435源及4619成员质量归档也保持原身份。

## 第二次实际失败与唯一后验owner

v2真实终态exit1、1460.20秒，A01/fold1/AR8 profile trace1再次触发posterior关系，peer自动停止，72包保留；progress与独立分析器均为failed_run，不产生完整指标。一次必要实际捕获12.87秒保留原RED：H相对误差1.86e-14通过原2e-9，M后向相对量4.06e-13却越过64 eps k=2.27e-13。原joint系统condition13900.64，quotient15.36；两条算法路径各自有很低后向误差，跨路径固定阈值仍误拒。

独立same-W Householder quotient/SVD核对显示原M绝对差约9.95e-23，标准化差平方2.26e-25；held归一化logp差至多2.33e-10、winner一致。原noise W、先验与输入不变；未将这些尺度推广给其他格或发布capture为成功模型。

最终改为一个完整C kernel posterior owner：原joint prefix/tail仍决定noise objective、scatter/logdet与空间Σ；最终kernel M/H由同一个nuisance quotient压缩R/rhs加原precision root生成，fit provenance直接保存这一次统计量。载入核验共享同一factors函数，后向/H阈值均未放宽。60项相关验证通过，独立对两次实际失败数组核对新M/H与QR/SVD、真实改变M/H/先验仍拒绝；旧统计和模型只在原source重放。

[第三版完整计划](../assets/prior_control_20261006/plan_v3.json)只变执行源、科学与资源格仍为1344。原两批101/72包保留原producer，不改签复用；新源码实际fit/save/load/score与完整矩阵分别继续验收。总体优化还要落实B12独立C/test、source→target三完整seed、K0/source-artifact与工作站调用，不把本修复作为全项目完成。
