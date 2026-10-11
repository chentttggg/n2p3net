# 算子深度科研续接（进行中）

用户明确指出此前只完成工程/原型，要求深度科研；本轮主目标为R1可识别性、实际有限算子、R2均值/噪声/先验联合和真实公平对比。不得仅用测试通过结束。

## 基准与授权
- 原基准386文件frozen/system_implementation_20261004_after/source.manifest.json，启动时全文件一致，SHA043893ac95e4ff8e9b5f3db718e216ed6c33c4178af5a01e826f2237c4b3973a。
- 用户已授权实现、实验；仅本地CPU，32逻辑CPU、可用内存约3GB，逐人数据、2进程×2native线程；无云启动。
- skills：astra科研workflow/science/research/engineering、EEG科研(明确并行分方向)、codebase-design、experimental-design、get-available-resources。
- memory快速检索MEMORY.md58–69，最终需引用；不更新memory。

## 已完成科学证据
- literature/report.zh.md + claim_sources.json：10实读来源/23访问记录；Gaussian F必须同步作用D/cov，确定padding不是额外随机Γ。SPM、Särkkä、Minka/Unfold/mTRF等一手；不要说Unfold示例也显式FD（不是）。joint_estimator.zh.md与bnci_distribution_and_conditional_ar.zh.md已补联合Σ加权/FGLS非ML、BNCI上游0.1–30声明（阶段/相位未知）、conditional未知nuisance可翻转等反例。
- R1 state_identifiability/report.zh.md：8人仅C800、1504伪切、48留人fold、2016注入。无约束条件边缘校正MSE降81%却ERP误差2.28%→24%；KU=0保真约束2/8秒MSE降39–47%，紧贴边缘无收益；强制长参照约束噪声放大795倍/32fold不可表示。其回归含截距，严格仿射而非线性算子；不晋升。保留初版step24与stim周期共振全非目标的失败，用互素step17修复，未按性能选样。
- continuous_mechanism/report.zh.md及verification：完整可逆F BF不变；AR1可hit相似而NLL坏；平均核Vh不等于生理每decision变化。

## 生产新Module与root研究原型
- src/data/finite_filter_operator.py：真实有限SOS F、Q=H.T H、真实两端B/state分解、bounded dense export；10独立SciPy/dense测试。
- src/transfer/temporal_noise.py：segmented Burg，stationary /1（原严格cond门）、conditional_prefix /2（不同估计量，Stage3不跑）、stationary_innovations /3新lattice（正在末验收）。保留真实fit PARCOR；不以浮点expanded roots虚假不稳拒绝；r0 property marginal_variance_per_unit_innovation。20tests、100位反例，原路径不改。
- continuous_model.py：OperatorGLS研究fit/score/save，有限FD可选、raw AR1/8/32、matrix-normal kernel posterior、full vocab logdet；原default与旧ContinuousDecisionModel逐值相同。支持fixed FGLS3、channelΣ加权Burg、standardizedLW、kernel_prior_scale innovation/marginal（λ_eff=λ/r0）、新startup模式。
- 核心新发现：initial OLS残差拟AR近白，GLS改均值后ACF~.8→一次噪声/均值不自洽；fixed ridge1随innovation scale改变物理ERP先验（纯gauge例logBF改216而配套lambda应不变）。Stage3会检验此轴。
- selection_erp_model.py agent已稳定：复用原waveform target_shrinkage_features+fit_shrinkage_feature_model、StructuredERP Toeplitz/xDAWN。4臂原owner logit差0，NPZ重放exact，float32 rounding顺序保留。真实源[-51,+306]358点→FFT179→合法PreprocessingSpec裁178；baseline26点、Toeplitzbin4尾2不用，不额外读307。
- run_calibration_study.py：同8人C800前6完整selection720labels，LOO group train600/test120，80labels未用；不是K800或K200测试分数。每selection单独raw范围/F，无heldlabels fit。all requested ledger，truth仅scorer；prototype/所有src实际bytes随run保存。

## 真实实验与结论（旧产物不改）
- calibration_groups_v1首次在60cell后FrozenMetadata跨process pickle失败，240ledger含notexecuted保留；修return JSON plain bytes且failed CLI exit1，process_transport_repair.json。
- calibration_groups_v2完整240/240：old(filteredY+D+clippedAR1) hit.60417 NLL2.79664；FD hit.375/NLL6.34686；rawAR1 .60417/3.02284；raw8 .66667/6.16231；raw32 .625/7.32640。analysis/全sealed/source/logit与replay独立核，heldACF~.8-.89且held/train能量比~1，不能仅改variance。
- calibration_groups_stage2完整624请求，576 completed + raw_ar32_weighted3全部48因stationary covariance cond拒。analysis_stage2/全复算；旧5臂logits逐值同v2。
  raw8iter3 hit.47917 NLL2.96075；weighted3 .375/3.40578；smooth1 .58333/2.50838；smooth3 .58333/2.57657，ACF仍~.53；标准化spatial vsplain差很小。没有可靠统一收益。
- root/continuous_operator_mechanism完成analysis并bootstrap8人；6fold不当独立被试。stage1脚本副本物理保存，不能更新旧analysis文件。

## 下一阶段（本轮已完成）
- calibration_group_plan_stage3.json：16臂768cell已完成；原执行752成功/16数值失败，失败原件保留。QR修复后当前最终源码repair3对AR32边际臂48/48完成。
- startup_mode stationary_innovations由PARCOR前p条件创新factor保留全n，不改变stationary估计量；不放宽原cond门。不稳定PARCOR拒，factor eps*cond报告，>1e-6不得当稳健晋升。
- runner已接family分派和prior/startup参数；repair3保存当前最终执行源，48/48完成。旧768请求仍按旧执行身份保留，不能合并成一次768成功实验。
- agent continuous_operator_mechanism 正负责最终新GLS/16臂验证与后续analysis_stage3（要分ERP无连续ACF）；root尚未发送后续分析具体任务，可在ready后发送。
- root需要写doc/evidence/operator_research_20261004.zh.md综合报告、更新六现有指南/索引（尚未做），强区分有收益/失败/无法识别/未确认。
- 完成stage3后独立复算、红队门禁、Ruff、全套pytest和最终源码冻结。最终全套2293项：2285通过/8跳过/0失败；最终源码390成员，archive SHA `6d0cfbe41c2a45d340599fba38d5e1a8795839718d40421edcbc8c1ee884b5e6`。
- 若出现新严重数值/统计问题，修正确切机制并完整重复受影响的预声明矩阵，不用先导好坏停。

最终结论：本轮完成了算子理论、反例、真实C800组开发矩阵和成熟ERP锚点，但没有整体算子收益或产品晋升证据。
