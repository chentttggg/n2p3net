# 个人 ERP 调用与当前守卫的 Lean 组件证据

本记录对应[组件验收](../assets/lean_personal_erp_20261018/component_acceptance_01.json)，
只接受所列数学模型、源码字段审查与有限行为证据。
完整 P01–P12、V04、V09 和整项目均未因此完成；当前执行状态仍只在
[活动 checkpoint](../assets/lean_gaussian_20261016/active_checkpoint.json)维护。

| 数学组件 | 本批定理 | 严格负控 | 已检查的有限语义 |
|---|---:|---:|---|
| PersonalERPRecipe | 60 | 14 | kwargs/defaults、ERP 配方与角色、当前 xDAWN reader 和数值调用顺序；含 CurrentERPState 的 15 条定理 |
| PersonalERPIdentity | 52 | 7 | participant/origin/global 同域并集、完整研究库排除、逐字段 temporal 回执 |
| PersonalERPAvailability | 70 | 15 | 固定 QC 后掩码、全请求行硬拒、有序视图、N 行回填和训练摘要操作图 |
| PersonalERPTransport | 34 | 8 | 逐数组解码字段、dtype 标签、单次前缀剥离及组件覆盖顺序 |
| WaveformFeatureWidth | 24 | 6 | selected 数目、原浮点算术操作接口、禁止升采样和完整系数形状条件 |
| OperatorRowSupport | 31 | 3 | 当前逐行缩放、投影、范数比较与 any；另证精确实数缩放几何 |

共 271 条数学定理、53 个严格负控。构建、六个独立进程复查和逐声明公理审计通过；
只出现 `propext`、`Classical.choice`、`Quot.sound`。
数学条数不表示项目覆盖比例，也不表示 Python、NumPy、Torch 或 IEEE 实现已经精化。

首次统一执行的[原报告](../../formal/lean/artifacts/personal_erp_final_01/report.json)整体失败，
原因是三份负控混入 `Decidable`、重写和嵌套子证明错误。
其原字节保留；[补充复查](../assets/lean_personal_erp_20261018/component_negative_rechecks_01.json)
复用 11 个依赖字节未变的成功检查，只新跑 3 份已修脚本的负控。
三个文件的 29 个命题逐字未改，修复仅涉及证明体；
新日志分别只有 15、8、6 条 `unsolved goals` 和对应 `False`。
[独立最终审查](../assets/lean_personal_erp_20261018/component_final_review_01.json)
核实该继承关系，没有把原失败改成成功，也没有声称重新执行了全部 14 个命令。

工作期间生产源码两次变化，原合同和旧验收未改。
当前采用[第三版合同](../assets/lean_personal_erp_20261018/science_contract_v3.json)，
源码快照逐成员检查 199 项，其中 189 项为运行源码。
新增守卫分别保留 xDAWN 退化 reader 拒绝、finite/Cholesky 异常、
Operator LDA 的逐行 maxabs/tiny 缩放和 waveform LDA 的实际物理维数。
历史批级稀释反例及单时间点 xDAWN 见证仍只属于原源码版本。

[实际 CPU 源码探针](../assets/lean_personal_erp_20261018/source_probe_v3_02.json)
完成 24 项检查和 44 项预期拒绝，没有拟合或训练。
原 N15 把 `h_freq` 设成了本来就有的 30，未产生合同差异；
[原失败与修正说明](../assets/lean_personal_erp_20261018/source_probe_correction_01.json)
保持独立。后续 producer 只将该输入改为 20，并先断言仅该字段改变，拒绝预期不变。
当前合法两时间点 xDAWN state 遇零输入会在 Cholesky 门失败；
未构造出当前源码接受并成功返回 NaN 的新见证。
可用性数学中的 NaN backend 只是一种接口反模型。

[P12 合成缓存探针](../assets/lean_personal_erp_20261018/p12_cache_mutation_probe_01.json)
通过真实 save/attested load 后，仅改变内存中一个 float32 单元。
普通验证与存储的 load-attestation accessor 仍返回，fresh 内容断言拒绝，磁盘两文件未变。
这说明原 load 身份与当前内存内容核验不同；它没有执行 source fit，
不能作为真实 `fit_plan` 失败、物理数据损坏或完整 P12 已证的证据。

[独立标量探针](../assets/lean_personal_erp_20261018/scalar_source_probe_01.json)
的 32 个固定案例均匹配源码预期：NumPy、`math.isfinite` 和普通比较的超大整数行为不同，
且 `temperature=0`、`prior=10**100` 时，prior 的数值异常先于循环后的温度正性拒绝。
实际个人构造仍先检查内部 ERP 状态，再进入 kwargs 配方。
这些有限案例用于修正后续形式模型的域与异常顺序，不认证 NumPy 的普适整数支持范围。

仍需补齐的源对应包括任意大 JSON 整数的数值操作失败、完整物理/执行/raw 入口、
原始行授权与实际字节、ERP-only 拟合 producer 组合、源样本到 Gram/谱/回放生产链。
各原语的浮点、dtype、alias、存储及物理真实性边界保持显式，
不以“域内数学已证”替代这些未完成义务。

[证明清单](../../frozen/lean_personal_erp_20261018/manifest.json)核验 140 个继承与新增成员。
替代检查确认后，已按[编译清理记录](../assets/lean_personal_erp_20261018/cleanup.json)
用 cmd 逐文件删除 52 个旧编译输出并移除 26 个空目录；
另按[合成缓存清理记录](../assets/lean_personal_erp_20261018/p12_cache_cleanup_01.json)
删除两份临时缓存及两个空目录。
两类原字节均先归档核验；活动依赖、尚在处理的模块、科学证据、冻结结果及暂停记录保留。
