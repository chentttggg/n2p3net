# 边界状态重新设计：源码核对、数学反例与独立研究框架

日期：2026-10-05。范围是研究设计与 `experiments/` 独立框架；本轮不修改 `src/`，不运行新的公共效能矩阵，不晋升生产配方。用户附件中的公式与数值作为待核对输入，历史结论仍绑定原执行源码。本记录承接[算子复核](operator_research_20261004.zh.md)及[Phase 4](../assets/operator_research_20261004/phase4_report.zh.md)。

## 1. 三个实现前提的答案

| 问题 | 当前源码与产物的答案 | 本轮实施选择 |
|---|---|---|
| `state_basis` 的维度、padding 能否推断响应长度 | `src/data/finite_filter_operator.py` 中，令 SOS 节数为 S，单向 state 维度为 2S，双端 B 为 `(N,4S)`，a 为 `(4S,C)`；若把 `order` 定义为 2S，才可以写成 `(N,2×order)`。`padding_limit` 是反射样本数上限，实际取值还受 N−1 限制。稳定 IIR 通常没有有限的严格脉冲支持 | 距离按实际输出支持到左右裁剪端点计算；衰减长度须另声明误差单位、容差、state 包络和实际输出算子，不能从 padding 数字推断 |
| ERP 子空间是否来自 `SelectionERP` | `SelectionERP` 是波形 LDA/Toeplitz/xDAWN 的分类与保存重放 wrapper，没有导出 `(k,N×C)` 信号子空间。R1/Phase 4 保护字典在 `state_identifiability/probe_constrained.py` 和 `probe_constraint_pareto.py` 内独立构造：8 个平滑时间方向、刺激前脉冲、常数和线性趋势，逐通道提升；另有固定 ERP 注入挑战 | 复现旧约束保留原字典；新研究臂另从训练校准组构造信号模板并 SVD 正交化，绑定时间、通道、单位和 fit 行。xDAWN 空间滤波器或 LDA 系数不能直接当作该信号基 |
| 复用 repair3 数据还是重新建 raw | `calibration_groups_stage3_repair3` 含 plan、模型、预测、scopes 和 executed_source，是 AR32 数值修复账本，不是统一的 C800 原始输入缓存 | 复用其 plan 中八份 MAT 的路径/哈希与六组协议，从原 MAT 经已有 `run_calibration_study.selection_views` 重建合法校准组；核对旧 scopes 的输入摘要。无需重新下载，也不改旧模型/结果 |

原计划：八人，每人 C800 中前六个完整 selection，各 120 flash。每外折训练五组共 600 标签，留出一组共 120 标签，另 80 授权标签不用，共 48 折。八名参与者是统计单位，六折不是六名独立参与者。MAT 是官方分发的已带通数据，不称 ADC 原始波形。

## 2. 附件中需要更正的数学与证据

### 2.1 确定 padding、外部 state 与统计估计误差

确定算子分解成立：`F x = Q x + B a(x)`。固定完整 x 后，a(x) 无额外独立随机性。若把 x 当成尚未实现的随机信号，a=Lx 与 x 相关，应传播 `Cov(Fx)=FΣxFᵀ`；拆开传播须保留所有交叉项，不能当成独立噪声相加。

R1 的 2.28%→约24% 是无约束差值校正的 ERP 注入挑战结果，不是单独添加 Γ 的受控因果实验；确定 padding 的双计问题有数学依据，但不能把这个误差数字直接归因于“额外 Γ”。

这不意味着“观测了 x 就知道真实外部生理 state”。真实外部 a_ext 一般仍有条件不确定性。若 `(x,a_ext)` 联合 Gaussian，其条件协方差为

\[
\Gamma_{a\mid x}=\Gamma_a-\Omega_{ax}\Sigma_x^{-1}\Omega_{xa}.
\]

该式以联合协方差可识别、维度及正定性成立为前提。有限段估计误差也不会自动具有已知 Gaussian 分布或已知 Schur 补。允许训练校准数据在可识别模型中估计超参数；“来自同一段”本身不证明循环推断，真正的问题是未经辨识就把确定 padding 残差解释成独立外部方差。

附件 `y=Fx+ε` 在给定 x 后不含 q，因此无法单独提供候选区分。候选模型应明确 `X=D_q W+η`，随后 `Z=FX`、均值 `FD_qW`、噪声 `FΣFᵀ` 同步传播，或者直接使用未追加滤波的 X。不能用未证明可逆的 Q 写 `Q⁻¹`，也不能令“真实初态”等同 padding 状态。

### 2.2 长短算子必须放到同一网格

令 `x_full∈R^(M×C)` 是实际观测长上下文，C 为 short 裁剪映射，R_l/R_s 为共同输出网格的选择映射，则

\[
\delta=\bigl(R_lF_M-R_sF_N C\bigr)x_{full}.
\]

两个不同长度的 F 不能直接相减。只有长上下文确实已观测时，δ 才是已知的确定函数。若只观测 x_short，两个不同外部续接可以具有完全相同的 x_short 而产生不同 δ；此时只能研究条件预测或预声明外推，不能调用不可得的长段输出修正未知测试。

本轮 `BoundaryStateAnalysis.compute` 使用显式长、短和输出半开区间，比较同一输出网格。它是长参照诊断接口，不是推理清理器。完整长段双向输出也意味着实际可用时间至少到该段结束。

### 2.3 输出投影与 ERP 增量保真是两个约束

将输出按样本优先、通道次序固定展平为 `v∈R^(T C)`，令 U 为校准信号模板的正交行基，`P=UᵀU`。这样不会混淆 `(k,C)` 空间投影和 `(k,TC)` 时空投影。

附件的新输出软约束为

\[
\min_u\|u-\delta\|^2+\lambda\|Uu\|^2,
\qquad
u_\lambda=\delta-\frac{\lambda}{1+\lambda}P\delta.
\]

它惩罚的是校正值在 ERP 输出方向的幅度。旧 `K U_input=0` 约束的是校正映射对 ERP 输入增量的响应，本意是保留 ERP 增量；边界校正收益受限不能直接解释成“该约束抹掉 ERP”。二者不等价，不能把“soft penalty”换个写法就称为新保护机制。

若实用校正为 `T(x)=F_short x+g(x)`，应独立检验

\[
E_{ERP}(h)=
\frac{\|T(x+h)-T(x)-F_short h\|}{\|F_short h\|}
\]

作为有限输出增量保护；若目标是长参照保真，分子/分母中的参考改为 `R_lF_long h`，另报其指标。线性 `g=Kx` 时，前者由 `Kh` 决定；按 x 动态拒绝/换 penalty 的非线性 g 还需挑战切换点。白噪声倍率也必须来自完整 T 的作用，而不是从一次 δ 的能量推断。

`||Pδ||²/||δ||²` 是重叠诊断。100% 重叠也可能正好补回有限滤波衰减的 ERP，不能证明损伤或宣布校正“不合法”。0.5 阈值仅是待验证候选，未获得经验拒绝资格；δ=0 时比例未定义，不能记成安全证据。框架输出 `overlap_flag`，不从 held diagnostics 选择 recommended penalty。

### 2.4 距离 schedule 与容许误差的方向

`λ(d)=λ_max sigmoid((d_pivot−d)/τ)` 随 d 下降。给附件默认值，d=0 时是 88.0797 而非 100，d=1 时是 50。默认 pivot、宽度和上限是候选，未由 Phase 4 三个距离辨识；距离较远也不能保证实际污染或 ERP 不重叠。

如果想表达“近边界强保护”，固定幅度球约束 `||Uu||≤ε(d)` 的近端 ε 应较小；附件同时要求 ε 随距离递减，与上述方向相反。penalty 与约束半径的对应还取决于当前 δ，不能仅从 λ(d) 推出一个全局 ε(d)。本框架用数值稳定的 expit，并在诊断中采用输出支持到最近 short 端点的距离，后续须分开记录左右边界类型。

### 2.5 当前先验尺度的正确合同

现有 `ContinuousDecisionModel` 及研究 `OperatorGLS` 的形式是矩阵正态先验：

\[
Y\mid W\sim MN(DW,K,\Sigma),\quad
W\sim MN(0,\lambda^{-1}I,\Sigma),\quad
V=(D^TK^{-1}D+\lambda I)^{-1}.
\]

完整核先验协方差为 `Σ⊗(I/λ)`，不是独立于通道协方差的固定绝对精度。必须区分两种变换：

| 变换 | 保持同一物理模型需要的同步变化 |
|---|---|
| 只重分配噪声 Kronecker 尺度：`K→αK` | `Σ→Σ/α`，`λ→λ/α`，`V→αV`，均值与 predictive covariance 不变 |
| 电压单位：`Y→sY` | `Σ→s²Σ`，`W→sW`，λ 保持不变；各候选增加共同密度常数，log BF 不变 |
| 真正改变 AR 阶数或噪声频谱 | 通常改变 K 的形状，不是纯尺度等价；没有单个 scalar ridge 能保证所有候选 BF 不变 |

因此附件的 `ridge_relative×mean(diag(innovation_covariance))` 不能作为当前合同的普适修复：V→µV 时它会令 λ 乘 10¹²。若改用独立绝对核先验，单通道 λ=σ²/τ_W² 可在明确固定 τ_W 单位时推导；多通道有相关结构时一个 trace 比例也不等价于完整先验精度。不能将不同先验混称相同路线。

历史 gauge 反例保持物理协方差不变，α=200 时未同步 ridge 的最大 log BF 差为 216.6195，同步 λ/α 后差为 1.7053e−13（历史产物，不是本轮训练）。`OperatorGLS` 已有 `kernel_prior_scale="marginal"`，以 `λ/r0` 改写 innovation 坐标下的先验；这是相对于 innovation prior 的另一物理先验，历史结果反而有严重过度自信，不能称新改进或保证消除 AR8 的 NLL 问题。

本轮 `ScaleInvariantContinuousModel` 只实现有已冻结 K/Σ 的小规模 dense Gaussian/QR 数学参考，直接检验上述 gauge 与单位变换，限制矩阵字节预算。它没有拟合 AR1/AR8，没有继承生产模型，不估计完整超参数后验。后续 E2 仍消费现有研究 `OperatorGLS` 的校准内 AR 估计与冻结状态。

### 2.6 Kalman 可识别性与时间许可

可将跨 selection 慢漂移与 ERP 幅度设为低维状态，但幅度作为 state 不自动保护 ERP。须核对 `(A,H)` 的 observability、实际可用输入、nuisance/ERP 可区分性、过程/观测协方差的训练范围和有效独立组数。原生八导、八人、每人六组不是“8 selection”；维数远小于观测维数只是必要启发式，不能替代秩与条件数。

SOS 双端 state 为 4S，而不统一是“2×滤波器阶数”。按 d 激活 state 还需定义状态转换和缺失观测，不能随意改变 Kalman 维数。filtering 用过去，RTS smoothing 用未来；后者必须报告回顾性可用时间，且不得把后续未授权 test 当校准上下文。路线 B 暂列第二优先级，本轮不实现未经辨识的 Kalman decoder。

## 3. E1–E3 实验合同

本次只建立框架与可再生计划，下面不是已执行结果。历史 C800 已被观察，只能作为开发数据；独立公共 participant/数据及成人确认分别另做。

| 实验 | 训练内状态与对照 | 固定输出与否证条件 |
|---|---|---|
| E1 重叠诊断 | 保留旧独立保护字典，新校准信号模板另成研究臂；距离 0/0.5/1/2/4/8 秒；长参照仅从合法实际 C 内构造 | 记录重叠率与独立 ERP 增量误差的关系；不预设随距离单调，也不预设高重叠必定损伤。可构造同 short、异外部续接反例 |
| E2 先验尺度 | AR1/AR8；固定 innovation-relative 与已有 marginal-relative 两臂；ridge 网格 0.01/0.1/1/10。AR order、startup、basis、更新轮数与 spatial estimator 明示 | 先过 gauge/电压单位反例，再报告 requested hit@all、coverage、条件 NLL、最大 confidence；真实 AR 阶数变化不要求 BF 等价 |
| E3 两种保护约束 | 旧 K 对输入字典响应的 penalty 与新输出幅度保护分别报告；penalty 0/0.01/0.1/1/10/100；同 E1 距离网格 | 同时报 difference MSE、两类 ERP 增量参考误差、完整校正算子的白噪声倍率、条件数、请求/失败分母。不按 held Pareto 挑默认参数 |

同一外折训练五组内做内层分组选择，然后冻结到 held selection；或者网格预先固定、完整报告所有臂而不选胜者。ERP 基、重叠阈值、λ、distance schedule 与 AR 超参数均不得在 held selection 拟合。伪截断的长短范围必须全部位于同一已授权独立组；真实记录不足以容纳某距离时保留请求与不可用原因，不能取下一组或 T 来补。

新 E1/E3 比旧 Phase 4 更改了子空间/保护目标，不自动“复现”旧 Pareto。旧 24% 来自 R1 无约束 ERP 注入挑战；Phase 4 不同挑战集合中 d=0 的无约束误差约 10%，强 penalty 可降到约 4%。不能混用为一条同定义的误差曲线。

### 准入与统计

- 数学反例：同物理模型 gauge 与单位变换的 log BF 差使用 float64 `atol=1e−8`，并保留错误 ridge 的负对照；`差<1` 对精确等价过于宽松。
- ERP 候选门：独立预声明 latency/topography/相邻 flash/边界切换挑战，增量误差 5% 只是研究上限；逐例、覆盖率、最坏值及噪声放大同时报告，不能只报平均。
- 命中候选门：0.75 低于旧 Toeplitz 0.771，不能称“不退步”。48 个 decision 中 37/48=0.7708 与 36/48=0.75 只差一个；须按参与者等权给配对差及不确定性，正式非劣界限另预声明。
- NLL 候选门：`2log(36)=7.1670` 是宽松数值筛查，raw AR8 的旧 6.162 已经能通过。均匀分布的 NLL 是 `log(36)=3.5835`；“诚实”还需结合概率校准、coverage 和同一输出项上的锚点对照。
- requested hit@all 与 coverage 保留全部请求；NLL 只对实际有效概率输出报告，失败不能偷偷填均匀后混入条件 NLL。连续 Gaussian 与 membership 聚合概率的 NLL 可报告共同 predictive loss，但不是同一模型的似然/BF 排序。

所有产物绑定执行源、MAT 字节、训练/留出支持、事件、labels、membership、fit/score digest、通道顺序、单位与完整配置。保存重放与协议核验通过后才进入新公开矩阵；本轮数学测试不证明公开性能或成人效能。

## 4. 已交付框架与验证入口

- [独立数学组件与计划入口](../../experiments/boundary_state_research.py)：显式网格/observed context、ERP 正交化、输出软保护、distance schedule、小规模 gauge reference、八份原 MAT 哈希检查与新计划生成。未实现实用边界预测器、AR 新拟合器或 Kalman decoder。
- [独立反例测试](../../tests/research/test_boundary_state_research.py)：normal-equation/dense Gaussian 参照、双通道 Kronecker 顺序、候选置换、尺度负对照、单位负对照、同 short 异续接、零差值、重叠不能证明损伤、state 重构和距端点 schedule。
- 本地计划与验证产物位于 `experiments/runs/boundary_state_redesign_20261005/`；旧 Phase 4/repair3 目录不修改。`before.json` 保存本轮开始时 118 个生产 Python 文件的摘要，收尾逐字节核对。
- 一手 API 核对：[SciPy sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html) 与 [sosfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html) 于本轮成功访问；前者文档的 padlen 是扩展长度，后者为每 SOS 保留两维 zi。数学尺度关系由本记录推导并由独立 Gaussian 反例核验，不将 API 文档当成效能证据。

```powershell
.\.venv\Scripts\python.exe experiments/boundary_state_research.py `
  --baseline-plan experiments/runs/operator_research_20261004/calibration_groups_stage3_repair3/plan.json `
  --output experiments/runs/boundary_state_redesign_20261005/plan.final.json
.\.venv\Scripts\python.exe -m pytest tests/research/test_boundary_state_research.py
.\.venv\Scripts\python.exe -m ruff check experiments/boundary_state_research.py tests/research/test_boundary_state_research.py
git diff --check
```

生成入口仅核对文件身份、输出实验计划，不读取/评分新 test，不启动公共训练。完整 E1–E3 的数据执行、ERP 注入/白噪声验收与独立公共复核仍待后续实验；诊断中未计算的 ERP/noise 指标显式为 null，不伪造通过值。

## 5. 本轮实际验证

48 项相关测试通过（本轮 11 项反例及有限滤波、连续候选、时间噪声合同）；时间噪声的既有病态矩阵挑战产生 1 条预期范围内的 SciPy 数值警告，无测试失败。Ruff 与 `git diff --check` 通过。

本轮独立 dense Gaussian 探针复现未同步 ridge 的 log BF 差 216.6195025；正确 gauge 的最大差 2.2737e−13，V→µV 单位变换最大差 1.1369e−13；附件式 naive noise-scaled ridge 的单位反例差 480.98956。这些是合成数学验证，不是 BNCI 新模型成绩。

只读重建八份原 MAT 的 48 个 C800 selection，所有 scopes、标签行和数值输入摘要均与 repair3 的已存 receipt 完全一致。未拟合模型，未评分 T。结果见本地 `math_probe.json`、`rebuild_verification.json` 与最终 `plan.final.json`；118 个生产 Python 文件收尾摘要与本轮开始时一致。

覆盖交接：`delivery → science` 已核实三个事实前提和附件差异；`science → engineering` 将共同网格与矩阵正态尺度合同落实为独立数学组件/反例；`engineering → knowledge` 以本轮验证产物和源码未改核对更新活动索引。`execution` 本轮只做软件/数学验证，完整公开矩阵不适用；GitHub 发布、云训练和成人测试均不在本轮范围。
