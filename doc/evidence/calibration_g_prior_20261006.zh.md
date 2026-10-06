# 全C固定reference g-prior：第一切片代码实施

日期2026-10-06。开发树为独立 `codex/calibration-g-prior`，起点 `b04adcb`；冻结M1与原并发树未改。承接[潜在模型研究规格](../assets/deep_research_plan_20261005/latent_models/research.zh.md)5.1–5.5及23项独立小反例合同。只做软件实施/合成验证，没有真实或云训练；这不建立新效能结论。

`transfer.kernel_prior`集中不可变 `KernelPriorSpec`、`CompiledKernelPrior`和structured `KernelPriorRankError`。isotropic ridge的precision λ与calibration g的variance multiplier g互斥；未提供任何prior解析旧ridge1。g不是1/λ，旧ridge数值路径与`/2–/4`身份保持自己的含义。

新g程序固定 `AR0_unit_innovation` reference，使用全部实际授权C设计，逐record剔独立diffuse intercept/trend。增量thin QR产生R_C，`G_ref=R_CᵀR_C`，prior precision root为 `R0=R_C/√g`。原设计另存thin QR root，rank threshold以原设计/投影残差的较大尺度计入float64投影误差，不把nuisance roundoff判成信息；rank不足抛具名识别失败，`error.record()`保存原/残差谱与尺度、threshold、scope/source，不pinv、不fallback、不把阈值当统计识别证明。

同一R0先用于响应初始化再给segmented Burg产生真实residual，随后进入NumPy prefix/tail joint QR和Torch CPU/float64 exact gradient。η过程中root不更新；空间profile仍为Q/nTail，最终kernel posterior仍使用全部C。basis一般换基配套prior使实际初始化residual及Burg PARCOR不变，旧isotropic初始化未偷偷控制g。dynamic noise-dependent prior、quotient、proper completion仍不解锁。

新g模型保存`n2p3_continuous_decision_model/5`。数值NPZ载荷包含实际prior/reference/原design roots及final-C information/score，metadata保存g、reference、rank、design/basis/support/source binding。载入按同一原design尺度重核rank，交叉reference ledger与actual C的ID/N/read support/design SHA，核basis/actual input binding，并检查prior-root/g与posterior M/H关系；只改metadata无法保留旧posterior或把g数值载荷伪装成ridge。held只用冻结posterior条件化本record prefix，不接受prior builder或校准数据，不修改fit state。

`CalibrationExpertProgram.kernel_prior`接canonical spec；g wire只含kernel_prior、不含ridge。`_FittedExpert`核actual prior kind/strength与program；FittedEnsemble保存重载完整root/source及fit binding。grouped-temperature OOF在各自训练组重新编译prior，最终全C另编译，不泄漏held组的design/labels。BrainSync日常入口仍未接这个raw研究程序。

两项评审修复后，32项受影响定向测试和143项联动回归通过，0 failure/error/skip；仓库Ruff与diff-check通过。精确JUnit、命令/返回码、实际源码SHA由本地 `tmp/g_prior_implementation_20261006/receipt_rank_binding_fix.json`记录，原127项结果保留为初版证据。

最终[软件质量凭证](../assets/calibration_g_prior_20261006/quality_receipt.json)绑定实际修后源：完整pytest2576通过、19跳过、0失败/错误，共2595项；398文件前后byte guard不变。独立Spec/math26项、Standards34项均通过，剩余已复现缺陷0；集合重叠不相加。修后`/5`为9数值数组，新增原design root支持可重算rank分辨与载入核验。prior SHA为`2e0e5da9e1b5471facfab1634012b731ec7dc1d8e4a9e563e5a48a009806cc48`，continuous SHA为`4d33731cd18946583a36f78d02d845d9bb7df98209cdeeba09a5ff1312b6c0d1`。

全集在独立g树执行，合并主树后的caller联动另验；不将之前M1 source的成绩充作新g公开效能，不重签原模型。原始RED、源码副本与最终命令收据继续本地保留，主线review后接收提交。

已在主树接收，runner另补两处真正调用链：same-ridge AR阶数面板明确拒绝g程序，避免混入先验归因；`KernelPriorRankError.record()`进入失败格fit-origin，保留spec/rank/source证据、requested预算与actual scored0。59项联动通过；合并后完整pytest2604通过、19跳过、0失败/错误，共2623项，399文件前后byte guard不变。原独立g树2576/19与主树全集分别记录，原M1仍用自己的冻结e982执行源，不追签新g结果。

[最终执行源物理清单](../../frozen/calibration_g_prior_20261006_delivery/manifest.json)封存419成员并对照实际主树/解包副本逐byte验证；[质量归档](../../frozen/calibration_g_prior_20261006_delivery/quality_manifest.json)封存1501成员、15345417压缩字节，包括原理论、修前RED、作者与两轴独立探针、两个全集及caller检查。原件前后及镜像SHA一致，Git只发布清单，payload留本地；g公开矩阵和成人确认未执行。

主树runner整合清单（本树未改runner/resume）：

1. g panel使用`CalibrationExpertProgram.from_record`的新`kernel_prior`字典，删除该entry的ridge；已有M1 `fixed_prefix_ar_order` comparison仍只比较same ridge，不把g塞入该归因面板。
2. g/baseline对照预声明actual C、物理decision、basis、noise程序和fixed32；g字段与λ字段不作数值配对，held不选择g。
3. valid-design rank failure捕获`KernelPriorRankError`，现有fit_failure保留requested输出、actual scored0；另外保存`error.record()`的structured prior/rank证据。
4. execution snapshot包含`kernel_prior.py`与新gradient实际SHA；resume读取/复用包必须保持新/5 kind、g、C-root与program binding，不给旧M1重签。

正规测试含all-C/per-record信息矩阵独立dense核对、Σprofile、四阶实际Burg gauge、单位/columnΣ、exact gradient、rank typed failure、wrong prior/posterior binding、保存重放与OOF隔离。测试只说明这些行为，参数恢复、四库/三完整seed/成人确认仍需各自证据。

独立Standards在初版c74源确认两处缺陷并有RED：全nuisance投影roundoff被相对自身尺度误admit；reference IDs只自摘要而未与actual C交叉。修复增加原design numerical root/分辨尺度和canonical binding交叉规则。正规回归覆盖六个scaled nuisance rejection、scaled真实信息/可分辨weak contrast、rank-load重签以及reference ID/N/support/design和basis/input重签拒绝。原RED保留，新源/复验单独记录，不把旧127结果充作修后新source验证。
