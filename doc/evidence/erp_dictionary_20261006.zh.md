# B07 A0/A1：物理ERP字典与同ambient端点限制

日期2026-10-06。隔离branch codex/erp-dictionary从5d6862ae开始，随后显式接收root的f0ac67e QR-posterior/worker修复及4067e98 interruption状态修复（secondary commits ac581f5/36cfde1），不修改main、原D树或正在执行的冻结矩阵。本切片仅A0/A1，不交付导数潜伏期、邻居目标边缘化或selection随机效应，也不建立新的效能结论。

`transfer.response_dictionary.ResponseDictionary`是唯一物理字典owner，持有生成spec、immutable actual matrix/grid、ambient函数空间、端点evaluation和constraint map。Gaussian free锚点保留原smooth_erp_basis的逐值numeric，centre/width同时解释成physical seconds；cosine按actual半开grid生成。Trequested、L=round(fs*Trequested)、Tactual=L/fs及lastsample=(L−1)/fs分列；fs256/Trequested1.2对应307、1.19921875、1.1953125，端点在0/Tactual评价，不能将last sample当右端点。

hard_zero只限制同一ambient cosine Φ：E=[Φ(0);Φ(Tactual)]，固定orthonormal N∈ker(E)，实际basis为ΦN。m个ambient列变为m−2个function列，common/target合计少4个参数；没有暗换sine family或隐藏drop列。prior数值关系核NblockᵀΛambient Nblock=Λactual。g的ambient precision由同C reference信息/g给出；ridge同scalar lift；trace-isotropic在declared reduced空间校准实际scalar再lift，不能声称原ambient trace-control的reference-energy被保持。

`ContinuousDecisionModel`实际fit/gradient/predict共同使用该basis。既有AR0 diffuse reference-root算法集中在kernel_prior.calibration_reference_roots；actual fitted root消费原kernel sufficient-statistic owner，修后是QR response坐标。本切片不复制或弱化posterior验算。dictionary_geometry保存actual ambient/reference/fit roots和prior precision，df_ref、df_fit、reference-prior-energy分列；rank阈值参照原设计尺度。df来自实际QR/prior，不能以相同n_basis/g声称容量匹配，hard arm明确capacity_changed。

拥有字典的model保存真实times、ambient matrix、E/N、reference/fit/ambient/original roots及precision数值载荷；load核actual function/grid/source、constraint/prior pullback和capacity，dtype保持float64。旧numeric route留原frozen程序身份，不给M1或既有B08包重新签源。新增schema是实际数值载荷扩展，不能通过只改metadata隐藏这些arrays。

独立Spec/Standards在初版755d源确认实际容量关系缺口：改dict fitted-root并重签其自身df/root/hash账本，actual M/H/prior fitstats不变，仍会载入。真实free/hard包的RED与431旧cap保留。修补移除cal-prior重复dict fit/reference/original载荷，容量直接消费actual prior fit R/rhs与compiled reference/design；ridge从其真实solve的D/Y生成同QR统计并核prior precision/H关系，legacyφ使用自己的noise-program identity而非伪TemporalNoise。三个对应小反例先RED、修后GREEN；旧native与18项结果仍只属于旧source，最终统一posterior source/新capsule复验另记。

root后续af3f856统一最终posterior：实际noise先发布，同full-C quotient QR R/rhs只生成一次，`_kernel_posterior_factors`同时给fit与关系门使用；noise objective/选中W、prior和diffuse测度不变。secondary以3837542消费该修复，capacity从返回的同statistics取数，不另算fitroot。该实现的统计serializer和precision-root/noise-digest入口为ridge扩展集中复用，不能继续声称两个函数逐body等同最早5448 source；published-posterior solver/factors与root最终body一致。最终字典parity对照也基于同af3程序的ordinary numeric route，最早5d before-source结果保留其原版本。

`CalibrationExpertProgram.dictionary`拥有spec，wire不另发冲突n_basis/duration；默认仍由原Gaussian生成。actual predictor与spec、native rate、basis、prior、source交叉，grouped OOF每fold在自己的C编译prior，最终全C另fit。真实operator CLI可消费dictionary，fit/read-only predict/save-load已接；candidate runner的fixed_dictionary_endpoints preset固定same C/native support/fixed32/noise/prior，Gaussian free、cosine free/hard三arm，诚实报告容量变化。旧AR/prior对照panel也核实际dictionary spec，不能用同列数掩盖换family。

初始缺接口/真实caller失败RED保留在本地tmp/dictionary_implementation_20261006/；少量行为测试通过geometry/独立dense、default source逐值parity、真实OOF、CLI、whole package replay验证实现，不机械跑全集。最终actual SHA/command/timing/native小probe和独立Spec/Standards由同目录receipt及主线验收另记；source未stable前的成绩保留原版本，不追认后续源。

独立研究给出24个小NumPy反例，并有另代理20项反方核对，见主线tmp/dictionary_20261006/research/与independent/：常数SOA可被record drift混淆，hard endpoint可删慢波，导数可冗余/不等于latency，reference-energy与df无法一般由单scalar同时匹配。完整history必须truth-free公共schedule且mix full density后除mix prefix density；random amplitude乘不确定kernel一般非Gaussian。这些约束保留为后续A2/H/B前置，本次未自动扩展现overlap准入。

最小native验收已在物理冻结source上完成同A01原MAT、两个完整授权C组（240label）与一个无标签held view（120flash）上的三fit，保原生8导/256Hz/right-ear/volts、相同read/score/availabletime及现unknown-overlap拒绝。三包逐值replay、模型文件bytes不变；Gaussian/cosine free的df_ref均14.4，actual df_fit约3.16465/2.90893；hard的参数数16→12、df_ref10.8、df_fit约.893086，reference energy144→108。数值说明actual geometry/容量变化，不能宣称basis/端点提升hit。首次cap与最终复验分别保实际producer，精确timings/source/exit由receipt记录。四专家成熟基线/B12、完整公共矩阵和成人确认由主线继续，不由本机制切片替代。
