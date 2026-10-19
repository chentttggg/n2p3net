# Lean：时间白化、完整状态回放与算子 LDA

验收日期：2026-10-18。主规范为[验证顺序与验收规则](../roadmap.zh.md#每个子项的统一验收)，
当前进展只在[活动 checkpoint](../assets/lean_gaussian_20261016/active_checkpoint.json)维护。
本记录保存本次固定源码与证明的证据，不替代原 V01–V10 范围。

[生产回执](../assets/lean_temporal_operator_20261016/acceptance.json)的10项检查全部通过，
包括构建、四个独立空日志重查、196条新定理的公理审查及6/8/6/6共26条严格负控。
新定理仅依赖 `propext`、`Classical.choice`、`Quot.sound` 允许集。
[独立语义审查](../assets/lean_temporal_operator_20261016/review.json)绑定四份正稿、四份负控、运行器与组合入口，
逐项确认前提、推导顺序、非空实例和源码对应。

| 模块 | 新定理 | 本次闭合的性质 |
|---|---:|---|
| [TemporalWhitening](../../formal/lean/N2P3/TemporalWhitening.lean) | 63 | 实际逐lag相减、前代startup及双buffer PARCOR递推；三模式行数、局部时间支持、线性和列间功能非干扰；unit-innovation尺度 |
| [TemporalState](../../formal/lean/N2P3/TemporalState.lean) | 47 | 三schema完整主状态及派生缓存、真实构造/训练账本守卫、摘要/contract/诊断顺序和规范生产记录重建 |
| [OperatorGeometry](../../formal/lean/N2P3/OperatorGeometry.lean) | 42 | 实际对称谱存在、严格截断与保留正性、支持内ridge、投影方程、解唯一及实际评分的保留坐标二次型公式 |
| [OperatorState](../../formal/lean/N2P3/OperatorState.lean) | 44 | 五数组读取与七字段metadata、噪声规范化、source/target账本、派生状态重建及完整批次评分准入 |

TemporalWhitening从源码循环推导有限和与分阶段输出，lattice使用实际reflection序列，未以任意函数代替递推。
`conditional_prefix`用实际前p行作历史但只输出n−p行；两种stationary模式保留n行。
诊断innovation variance不再除一次标准差；列间非干扰表示函数依赖，不表示统计独立。
`stationary_innovations`只要求存储coefficients与PARCOR展开式数值allclose，不能从构造接受推出两者精确相等。

TemporalState的回放前提是源码形状的具体构造函数正常返回；先分解出这次原语运算及字段结果，
再由独立writer/reader重新执行构造守卫，没有把整状态roundtrip或评分相等作为输入。
analytic reference必须是非空白文本；训练路径检查当前implementation身份、完整record ID、
每record `n_samples≥max(3,p+2)`、正列数、`per_parameter≥2`及lag行数门槛。
三种非空AR(1)模型和33样本产生32个可用lag行的合法账本实例均已证明。
原语接口保留Lyapunov、Cholesky、逆PARCOR/半径及浮点解码等数值实现边界；没有由原语成功直接推出未经证明的稳定性或PSD。
重封签的文本coefficients可被float规范化后接受，因此任意原始记录逐字段恒等并不成立；规范writer回放和原始输入规范化分别表述。

OperatorGeometry从固定Mathlib谱定理得到精确对称矩阵的谱表示。
源码使用 `lambda_i > lambda_max × 1e-10`，等号不保留；有正最大值时保留集合非空。
令V为保留正交列，`P=VVᵀ`，保留特征值按 `lambda_i+rho×mean(lambda)` 正则化得到G，则：

\[
w=V\frac{V^\top\delta}{\lambda+\rho\bar\lambda},\qquad Gw=P\delta,
\qquad b=-\tfrac12(\mu_0+\mu_1)^\top w.
\]

保留坐标中的G正定，环境空间可仍奇异；没有把支持内ridge改写成环境恒等ridge。
源码对类差和完整centered批次使用相对容差；类差通过不意味着严格属于range(P)。
精确有限和证明实际logit等于两个保留Mahalanobis能量之差的一半，不凭奇异环境协方差宣称完整环境密度。
同P和G时权重相等由唯一性推出；未宣称所有数值eigenvector旋转已自动保持这些输入。

OperatorState按真实顺序读五个primitive数组、核schema/metadata seal/array seal，
重建并对称化JointNoise，再检查算子和分类账本、重算P/G/w/b。
额外NPZ派生数组不参与加载，额外metadata仍进入封印；摘要没有被假设为单射或生产者认证。
目标噪声scope须与分类器一致；明确借用的source噪声可以来自不同scope。
Operator的implementation字段只检查格式，与Temporal严格比对当前implementation的路径不同。
合法非空一维fixture另证输入协方差为I、真实精确谱、w=2与评分4。

[真实NumPy探针](../assets/lean_temporal_operator_20261016/source_probe_01.json)绑定195份输入，其中189份是生产源码。
14项检查和7条拒绝在暂停前真实执行，恢复时重新核对源码及receipt身份；旧26/10预备证据只复用，没有冒充重新执行。
其中确认：

- `delta=[1,5e-7]`通过构造但有非零null分量；G为diag(3,0)，`Gw=[1,0]=Pdelta`。
- 单独null行会被拒绝，拼入足够大的in-support行后，同一null行可通过整个批次的相对Frobenius门槛。
- 恰位于相对谱阈值的方向被丢弃，略高方向保留；伪造的archive weights/G/P/b被忽略。
- 容差接受的轻微不对称raw协方差经加载后变为对称规范值，原array seal与重存array seal可不同。
- PARCOR执行值与仅allclose的存储AR展开式可有可测差异；拼接record也会改变第二record的startup。

探针无Torch、拟合、训练、磁盘模型和临时包；BytesIO调用实际public load，没有测试文件系统save。
这些是确定合成软件证据，不提供生理噪声有效性、公开数据收益或完整IEEE/数值求解器精化。

[114成员证明快照](../../frozen/lean_temporal_operator_20261018/manifest.json)与
[196成员源码/receipt快照](../../frozen/lean_temporal_operator_20261018/source_probe_manifest.json)逐成员解压校验，
继承的103个证明成员和189份生产源码保持身份。10项聚合检查墙钟148.45秒；本地计时不含agent构造和审查。
恢复时实测总内存约31.8GiB，检测可用约16–17GiB；实际检查沿用最多两个独立进程与独占构建，未由不同切片耗时推断整流程加速比。

[清理回执](../assets/lean_temporal_operator_20261016/cleanup.json)记录新验收确认后，用cmd逐文件删除
23个废弃构建/检查尝试目录中的46个文件；原诊断先压缩逐成员验证。
15个最终或暂停引用检查目录、活动依赖、科学原始证据、冻结结果和暂停checkpoint均保留。
旧leantar ZIP仍保留此前自动审批 `blocked by policy` 的删除拒绝结果，没有重试或绕过。

规范收敛前的8份文档、计划与checkpoint已[压缩核验留档](../assets/lean_temporal_operator_20261016/spec_convergence_before_20261018.json)。
主规范保留静态范围/依赖/验收规则，专题保存这些固定证据，分组和子项的动态进展迁入活动checkpoint。
