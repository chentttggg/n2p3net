# N2P3-Net Lean 形式化验证

本工程用 Lean 4.24.0 和 Mathlib 的固定提交验证选定实数数学规格。
`lake-manifest.json` 固定全部依赖；实验源码的实际字节摘要由检查器记录。
这是一组算法性质的形式化，不把 Python/BLAS/PyTorch 或完整训练流程作为已验证实现。

| 规格组 | 证明位置 | 对应项目表达式 |
|---|---|---|
| 联合协方差 | `joint_covariance_cross_terms`，两个块矩阵辅助引理 | `JointNoiseModel.project` 的拼接、乘法及 raw/state 交叉项 |
| 半正定与奇异性 | `covariance_pushforward_psd`、`projection_can_destroy_pd` | 固定算子传播协方差；输入正定不保证输出正定 |
| 对称平均 | `symmetrization_preserves_symmetric` | `(cov + cov.T) / 2` 在对称实矩阵上保持原值 |
| 缺失证据 | `unavailable_evidence_irrelevant` | `membership_evidence` 的 `where(available, ..., 0)` 与 sum |
| 候选偏置 | `scalar_offset_formula`、`equal_counts_preserve_difference`、`unequal_counts_offset_counterexample` | offset×有效membership计数；等计数充分条件及不等计数反例 |
| 时间支持 | `half_open_supports_disjoint`、`accepted_test_not_calibration`、`accepted_test_support_separated`、`half_open_overlap_iff` | `binary_time_split` 的前缀、purge谓词与输入合同的半开区间 |

时间证明要求每个校准 epoch 的可用终点不超过所取 boundary；源码用获准前缀的
`available.max()` 构造该上界。当前形式化证明接受这个上界作为输入，尚未形式化
NumPy max、实际索引数组的生成及浮点时间计算，也不推出滤波后的统计独立性。

原试验模块的协方差展开是有限矩阵代数等式，不包含随机变量、二阶矩存在性及 covariance
定义的概率论证明。PSD 定理显式假设输入 PSD；项目的近似特征值门禁不是这个
精确假设的形式化实现。候选证明只覆盖 `aggregation="sum"`，不覆盖 legacy mean。

## 环境与复现

使用固定 `lean-toolchain`，安装兼容 Lean/Lake；从此目录执行：

```powershell
lake exe cache get Mathlib.LinearAlgebra.Matrix.PosDef Mathlib.Data.Matrix.Block Mathlib.Tactic.FinCases Mathlib.Tactic.Linarith Mathlib.Tactic.NormNum Mathlib.Tactic.Ring
lake build N2P3Pilot
lake env lean -j1 Audit.lean
```

本机便携工具链位于 `D:/BCI/lean-tools/lean-4.24.0-windows/bin`。
在仓库根目录用项目 venv 运行带度量的完整检查；output 必须是尚不存在的目录：

```powershell
.\.venv\Scripts\python.exe formal/lean/scripts/verify.py --lake D:/BCI/lean-tools/lean-4.24.0-windows/bin/lake.exe --output formal/lean/artifacts/reproduction
.\.venv\Scripts\python.exe formal/lean/scripts/check_python_bridge.py --output formal/lean/artifacts/reproduction/python_bridge.json
```

`verify.py` 依次构建、重新 elaboration/kernel check 本工程证明、逐定理审查公理，
并检查两个故意错误的文件被拒绝。仅接受 `propext`、`Classical.choice`、
`Quot.sound`；本工程不允许未完成证明或额外计算信任构造。
它使用官方依赖缓存，没有以 trust=0 重新检查全部导入库。

内存报告区分 50 ms 采样的进程树 RSS 峰值与单进程 Windows peak working set。
CPU 时间为被采样进程的累计值，极短子进程可能未被采到；旧回执包括 Lake
启动开销，新的独立重查直接启动 Lean。每个检查受 `-j1` 约束；下载与依赖准备另行记账。

`negative/` 中的文件必须检查失败，故不属于默认构建目标。
`.lake/`、本地运行日志和构建产物由本目录 `.gitignore` 排除。
完整结论、实测与后续范围见 [试验记录](../../doc/evidence/history.zh.md#形式化验证)。

后续顺序以[执行计划](../../doc/roadmap.zh.md#依赖与交付顺序)为准；
[机器规格](verification_plan.json)记录同一方案的范围、依赖、验收规则和基准源码身份；
当前状态仅在[活动checkpoint](../../doc/assets/lean_gaussian_20261016/active_checkpoint.json)维护。

## 首批执行

首批构建目标为 `N2P3Program`，组合 `N2P3Pilot`、`N2P3.RawScope`、
`N2P3.FiniteSOS`、`N2P3.FiniteAdjoint`、`N2P3.Covariance`、`N2P3.FirstSlice`。
V01.a/V02.a/V02.b 的数学规格、源码审查和行为证据已验收；
详见[首批记录](../../doc/evidence/history.zh.md#形式化验证)。

```powershell
.\.venv\Scripts\python.exe formal/lean/scripts/verify_stage.py --output formal/lean/artifacts/stage_reproduction
```

从仓库根目录运行；output 使用全新目录。验证器自动生成 `AuditStage.lean`，
覆盖含属性的 theorem/lemma 声明，逐项核查标准公理并核对每条故意错误证明的拒绝。
首批回执使用 OS 文件锁串行执行；后续检查的启动与并发测量见
[性能证据](../../doc/evidence/history.zh.md#形式化验证)。

首次试验原文件及其原配置已按原回执摘要物理归档；当前配置改变不改写旧试验身份。
运行日志属于验收证据保留，旧构建仅在新构建确认后按用户要求用 cmd 清理。

## 连续候选层

当前默认构建目标为 `N2P3Continuous`，保留首批模块并加入 `GaussianLowRank`、
`ScaleGauge`、`NuisanceProjection` 和 `GaussianConditioning`。这些模块覆盖
低秩/密集评分、先验 gauge 与单位、实际 AR1 白化/保留观察坐标，以及完成平方的
非归一化高斯核；完整归一化条件概率和 Python/IEEE refinement 仍独立记录。

```powershell
.venv\Scripts\python.exe formal/lean/scripts/verify_continuous.py --output formal/lean/artifacts/continuous_reproduction
```

该检查器复核已接受首批证明与全部生产源码摘要，构建当前组合，并对新模块另起
进程重查、逐定理审查公理、逐错误证明确认拒绝原因。运行回执保留实际命令及等待时间。
数学/源码语义与行为证据见[本层记录](../../doc/evidence/history.zh.md#形式化验证)。

`N2P3.ProbabilitySlice` 进一步组合已经验收的有限维 Gaussian 归一化、实际矩阵正态坐标
及非冗余补空间存在性。它复用前 274 个定理的固定身份，另检查 53 个新定理及 5 个负例：

```powershell
.venv\Scripts\python.exe formal/lean/scripts/verify_probability.py --output formal/lean/artifacts/probability_reproduction
```

`N2P3.PosteriorSlice` 再组合已经验收的 83 个 Bayes 密度商、基无关多通道支持体积和
相关噪声协方差定理，新增 5 条错误证明拒绝检查：

```powershell
.venv\Scripts\python.exe formal/lean/scripts/verify_posterior.py --output formal/lean/artifacts/posterior_reproduction
```

`N2P3.ConditionalSlice` 再组合 149 个 joint/disintegration、完整预测积分、保留空间概率密度、
真实概率矩与源码投影拟合/评分连接定理，并增加 11 条严格错误证明拒绝检查：

```powershell
.venv\Scripts\python.exe formal/lean/scripts/verify_conditional.py --output formal/lean/artifacts/conditional_reproduction
```

`N2P3.CandidatesSlice` 接入候选概率/NLL、首 argmax 与绝对容差弃权、物理覆盖/拒绝优先级、
离线可用时间，以及真实 Gaussian 后验积分到候选归一化的 94 个定理：

```powershell
.venv\Scripts\python.exe formal/lean/scripts/verify_candidates.py --output formal/lean/artifacts/candidates_reproduction
```

`N2P3.JointStateSlice` 完成实际候选joint/disintegration、真实后验分类错误率/NLL，以及
连续/联合噪声全字段语义编解码和摘要检查顺序，共88条新定理和9条严格错误证明拒绝：

```powershell
.venv\Scripts\python.exe formal/lean/scripts/verify_joint_state.py --output formal/lean/artifacts/joint_state_reproduction
```

JointState层的8项聚合检查及84文件逐成员快照
见[现有证据记录](../../doc/evidence/history.zh.md#形式化验证)。实际噪声模型前提绑定、
数据估计超参数的不确定性、实现 refinement 和 IEEE 误差界仍在后续范围；
定理数量不表示项目覆盖比例。
候选联合测度的完整disintegration已从真实law质量推导。状态语义codec返回已解码构造输入，
完整源构造器、NPZ/JSON、数值和全模型族冻结合同仍待证明；不能由字段往返直接推断它们通过。
真实analytic账本反例及个人模型外部公开state修改后的双表示反例，均在现有证据中单独记录。

`N2P3.EventLedgerSlice`新增实际事件设计、公开事件史与已解码构造账本的88条定理，
错误证明固定3/3/5条；完整项目继续按原计划推进。

```powershell
.venv\Scripts\python.exe formal/lean/scripts/verify_event_ledger.py --output formal/lean/artifacts/event_ledger_reproduction
```

该运行器冻结继承84成员和189份生产源码身份，构建后执行三个独立空日志重查、
全部新定理公理审查与三组严格False控制；验收及93成员快照见[现有证据](../../doc/evidence/history.zh.md#形式化验证)。
账本域、一般生产者、真实Python行为证据分别记录；构造接受不推断完整评分就绪。

`N2P3.PersonalSlice`再加入PersonalState、PersonalLDA、PersonalAdmission和PersonalRestore，
125条新定理从实际命名张量接到同表示LDA评分，并保留正式入口与通用epoch API的不同边界。

```powershell
.venv\Scripts\python.exe formal/lean/scripts/verify_personal.py --output formal/lean/artifacts/personal_reproduction
```

该层九项聚合检查包含四个独立空日志重查、全125项标准公理审查和6/6/6错误证明拒绝。
本层[证据](../../doc/evidence/history.zh.md#形式化验证)与
[103成员快照](../../frozen/lean_personal_20261016/manifest.json)保留原生产身份。
构造输入封装、LDA子构造和typed featurewidth不等于五kind完整构造/前向验证；全项目目标继续进行。

`N2P3.TemporalOperatorSlice`连接实际时间白化、三schema规范回放、保留谱几何与五数组算子LDA状态。

```powershell
.venv\Scripts\python.exe formal/lean/scripts/verify_temporal_operator.py --output formal/lean/artifacts/temporal_operator_reproduction
```

该切片按构建、独立逐模块重查、逐定理公理和固定负控检查验收；来源及可信边界见
[专题证据](../../doc/evidence/history.zh.md#形式化验证)。当前累计数与下一步只查活动checkpoint。

`N2P3.ERPSlice`组合实际完整分箱、源时间对比Gram、Toeplitz双写与四族数值状态。

```powershell
.venv\Scripts\python.exe formal/lean/scripts/verify_erp.py --output formal/lean/artifacts/erp_reproduction
```

本批定理、软件探针、近似数值接口及剩余配方/个人入口边界见
[ERP专题](../../doc/evidence/history.zh.md#形式化验证)。新运行器保留历史producer和负控继承的分别计数。
