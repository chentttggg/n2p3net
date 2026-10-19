# E2 成熟 ERP 基线与 R3 时间结构迁移：实施证据

## 第三阶段：强制可归因对照，而非仅提供可运行臂

最终 Spec 审查证实第二阶段仍允许 full-fine 独跑，或用不同预算/架构的 scratch 作为唯一对照。
该行为违反活动研究规范，不能因各臂都能训练保存就认定矩阵科学闭合。独立原始 RED 位于
`review_spec_closure/anchor_admission.json`；本支线补五个实际 runner 反例，先得到 **5 failed**：
遗漏scratch、遗漏原waveform、预算不匹配、实际架构不匹配、仅物理队列相同但数值输入不同。

修后每个可执行正K full-fine 的预核验均寻找同 participant/K/source_seed/target_seed 的 scratch 与原 waveform LDA。
控制必须通过空 allowed axes 的严格数值输入身份比较；scratch 还须匹配实际重建的 architecture_record
和除名字/初始化角色外的完整拟合配方，包括 epoch、batch、LR、权重衰减、BN与dropout。
scratch始终拟合目标normalizer；full显式使用source normalizer是允许保留的处理角色差异，不把source统计传给scratch。
找到的 controls 以 cell index/arm 写入结果。任一必需控制找不到，模型拟合与结果目录创建前即拒绝。
缺K仍为 `actual_K=None`，不会为通过对照门禁捏造实际输入；独立scratch-only研究仍可执行。

对照存在于矩阵中也不能为另一个错误 contrast 背书：
显式把 full 与不匹配 scratch 放入同一个统计 contrast 会拒绝；默认生成的此类 contrast 会省略并记录原因。
额外预算/架构探索臂可保留在 cells，但不能冒充强制matched source-effect对照。
结果独立输出 `matched_transfer_effects`，其逐人差值、负迁移标记及 incomplete_pair 只来自已核验的实际匹配。
full对waveform或其他ERP分类器的比较可继续保留，明确不等于单因素初始化效应。

门禁后 runner suite **8 passed / 26.53 s**；追加 BN-policy不匹配与“合法控制存在但contrast绕用另一scratch”反例
**2 passed / 8.20 s**；Ruff和限定文件 `git diff --check` 通过。原独立脚本由 Spec 审查者另存 GREEN重放，
不覆盖原RED。主任务随后负责最终全套验证与物理源码冻结。

BDF测试helper已迁移至 `tests/support/raw_prefix.py`，source-prefix、personal predictor与comparator三处直接导入；
原测试文件内定义已删除，无兼容别名。此前“待整理”事项现已完成。

## 第二阶段：独立审查后的实际闭合（2026-10-04）

第一阶段接口仍漏掉原 waveform LDA 桥接、source/scratch NN 锚点和缺 K 账本；
独立 Standards/Spec 审查还找到四项数值/状态漏洞。本节覆盖修正后的现状，后文保留首轮过程。

- `FittedCalibrationCandidate.predict_logit` 先检查 mask 为真正的、shape 对齐的 bool 数组；NaN/float 不再经 truthiness 被当作已观测。
- `TemporalBasis.from_record` 对 fixed_analytic 重新执行固定余弦公式校验；换成任意正交基并重签 hash 不能伪装成无源对照。
- `XdawnRiemann.fit` 在任何输入检查前清 `_fitted`；失败 refit 后不再返回旧模型分数。
- xDAWN 使用未正则化实际训练观测的空间秩，将每类 filter 数截到 `min(requested_nfilter, observed_rank)`；
  rank=1 的三份重复导只生成每类一条 filter，不再因 OAS 数值满秩生成六条假实测维度。
  `observed_rank/effective_nfilter` 写入纯数值 artifact，低秩保存重载已覆盖。

四项先加入 seam 反例并执行，确认为 **4 failed**，修复后聚焦 **7 passed**。
独立原反例由其他审查者继续重放；本支线没有通过修改独立审查脚本来改变结果。

`run_erp_comparator.py` 的实际执行链进一步改为：

1. 明确 `evidence_scope=software_correctness|public_development`，不再默认将合成容器标成公共性能。
   public_development 强制 `verified_executed_snapshot` 核验当前执行代码的物理 freeze。
2. 除四个 ERP family 外，直接支持原 `waveform_lda`、`full_fine`、`scratch`。
   原 waveform LDA 用现有固定特征与现有 shrinkage pipeline，不能用 pooled_lda 替代其桥接。
3. NN anchor 从精确读取的一份 checkpoint bytes 建立 `PreparedTransferCheckpoint`，
   participant 与完整 target study 留出都通过后才拟合；source seed 取真实 checkpoint 合同，保留源训练快照 hash。
   scratch 只取相同架构，重新初始化，独立目标统计；改变源所有浮点权重和 input_mean 后，scratch 逐预测完全相同。
4. 计划 NN epoch 预算必须显式给出；不按 test 分数挑预算。支持 `{source_seed: checkpoint}` 的完整 checkpoint grid 与显式 target_seeds。
5. 预核验和实际执行都一次只读取一个 participant/arm；不把全部人的多 K EEG 张量保留在 materialized 列表。
   预核验留下小体积身份记录，实际重载后再次比较输入身份；源 checkpoint bytes 在同一次运行内不变。
6. 请求 `[8,200]` 而 prepared 仅有 `[0,8,16]` 时，K8照常执行，K200为 `protocol_unavailable`、`actual_K=None`。
   不改称K16，不造输入hash；requested cells、requested participants 和 requested_budget_coverage 全部保留。
   statistics 只对真实已materialize的实际K调用共享分析，同时写外层 requested 分母与 requested-budget 覆盖。
7. 对 ERP、原 waveform LDA、NN 都实际保存纯数值状态并重载评分。
   waveform 标准化重放保留 sklearn 的输入dtype及两次舍入；验证时曾捕捉到约3e−6差异并沿实际transform修正。
   ERP/waveform要求逐值相同；NN记录1e−6容差和实际最大误差，并要求重载 AUC 完全相同。
8. 每 cell 写 canonical study、participant、source/target seed、当前 comparison identity 和证据级别，
   直接调用 `analyze_transfer_contrasts(..., contrast_plans=...)` 生成 statistics.json；
   deterministic control 在 source-seed 轴上明确标注为对照别名，不声称产生独立 source 拟合重复。

验证：第二阶段三项真实容器执行器验收 **3 passed / 12.76 s**，包含八臂（原waveform两输入已补）、
三臂NN/LDA锚点加缺K账本、无scope/无public执行freeze拒绝；另测source权重不进入scratch、
同库不同人仍被整库留出门禁拒绝、逐人loader调用与statistics接口。
修后完整定向 suite **112 passed / 51.83 s**；Ruff全部通过。
结构化执行摘要与文件SHA见 `baselines_phase2_evidence.json`；数据仍为合成 EEG 的真实 BDF/BIDS 容器，
这里不作真实公共数据算法收益声明。

最后的测试结构整理项：BDF helper 当前从 source-prefix 测试文件导入，已通知主任务迁移至统一 tests/support；
遵循主任务冻结要求，本支线在 validation pass2 冻结开始后停止全部 src/experiments/tests 编辑。

本记录对应 2026-10-04 科研路线。修改前源码由主任务冻结在 `frozen/system_implementation_20261004_before`。
本支线使用 codebase-design、tdd、diagnosing-bugs 的 Module/Interface 与反例先行方法。
实际检验覆盖 Baseline、TargetRecipe→校准、个人模型保存重载、公共独立 raw C/T 执行器四个 Interface。

## 已交付

1. `src/baselines/riemann.py` 保留既有 Baseline Interface，但改为正确 ERP prototype 增广协方差。
   xDAWN 的模板只从 fit 标签生成；预测拼接冻结模板与当前试次，交叉项保留极性与时间顺序。
   原实现对 `X` 和 `−X` 给出完全相同分数，新增实际反例先复现失败，再修复。
   数值结果与安装的 pyRiemann `XdawnCovariances(oas, xdawn_estimator=oas)→TangentSpace→LDA` 一致。
   拟合 schema `/2` 防止旧 bare-covariance 模型被当作新 ERP 模型加载。

2. `src/baselines/structured_erp.py` 集中 `xdawn_erp`、`pooled_lda`、`toeplitz_lda`、`temporal_lda`。
   所有 family 都有 fit/predict 与纯数值 export/from_state；个人包只保存 JSON 与 tensor，不引入任意 sklearn pickle。
   ToeplitzLDA 按官方算法先减类均值、合并残差、标准化收缩、恢复量纲、时间 block 平均、线性 taper。
   `sum(lag blocks)/T` 等价于 `mean(lag blocks)×(1−|lag|/T)`；可解释为零延拓序列的平稳滑动协方差，保留半正定性。
   最终用 Cholesky 检查正定，不用事后任意截负特征值伪装标准方法。
   默认输入 `(N,C,T)`；channel-prime 展平由 Module 承担，不让调用方猜 reshape 次序。
   固定整数 bin width；最后不足一个完整 bin 的尾部不合并成异宽 bin，丢弃点数进入 artifact。
   标准 Toeplitz 只接受 `baseline_mode=none`；已 baseline 的输入必须回到原始记录另建视图。

3. `TargetRecipe` 新增明确 ERP family、`erp_bin_width`、时间基类型/秩，旧 recipe record 默认仍省略新增空字段。
   `fit_calibration_candidate` 复用原 QC/授权/实测通道选择；静态与动态可用 mask 取交集。
   完整选择通道在预测缺失时按已有 QC 可用策略拒绝或保留 NaN，绝不把补零当观测。
   `CalibratedPredictor(selector_mode='fixed_erp')` 接通原始校准、保存与未知标签推理；不依据 test AUC 挑 family。
   源 NN checkpoint 与源时间基为不同角色，不互换。
   `target_only_structured_erp` 与 `source_temporal_target_erp` 明确区别源监督信息是否实际使用。

4. `src/baselines/temporal_transfer.py` 提供冻结的时间基。
   源路线仅从显式合法 source rows 的类均值差 `D_s` 构造
   `G = mean_s[D_sᵀ D_s / trace(D_sᵀ D_s)]`，取预声明 rank 的主特征空间。
   participant 等权；同一 participant 重复分片会拒绝，避免按 recording 数重复计权。
   通道交集仅在各 participant 内求，不要求十二库共有同一组电极；目标空间判别仍从目标实测通道拟合。
   源 study 与 participant origins/global identities 保存在 fit receipts；目标完整 study 留出与来源排除规则在使用时验证。
   数值 samples、标签、mask、事件、通道、父 cache attestation 均被记录；`verify_source` 重新拟合检查实际关系。
   固定余弦基是独立 source-free 对照，没有 source-training 声明。
   source rank 不可辨识、无 ERP 对比能量、无共同实测导、非有限值均失败，不随机补齐空间。

5. 两个可运行入口已贯通。
   - `experiments/fit_source_temporal_basis.py --plan ... --output ...` 从 attested caches 与显式 source rows 拟合、冻结、重载后再次复验。
   - `experiments/run_erp_comparator.py --plan ... --output ...` 从 `load_target_prefixes` 验证过的独立 raw calibration/test scopes 做对照。
   先核验全部预声明 pair 的物理事件队列及允许变化轴，再开始拟合。
   每 cell 写实际输入、split 和 comparison_identity；保存纯数值模型、重载逐分数完全相等后保存预测。
   失败仍在 requested cells 中；不执行自动 legacy 复用，不接受复用目录覆盖。
   执行开始保存当前 Python 源文件摘要，结束验证未变。正式性能矩阵应从新的不可变执行快照启动；before 归档不能冒充改后执行源码。

## 独立数值对照

官方来源 `https://github.com/jsosulski/toeplitzlda` 与 `https://github.com/jsosulski/blockmatrix` 的数值源码已经留存，见 `toeplitz_reference/`。
`verify_official.py` 执行原函数/类 AST，仅省略绘图/可选 Fortran import，并把 NumPy 2 移除的 `product` 名映射到 `prod`；未改其数值公式。
旧 sklearn `_validate_data` 已删除，故从官方已拟合 coef_/intercept_ 直接计算同一 binary score。
三组独立规模/单位（N=40,C=3,T=8；N=10,C=5,T=11,V尺度；N=100,C=1,T=5,百万尺度）
对比最大绝对 logit 差依次为 1.67e−15、1.07e−14、1.11e−15，协方差也通过对照。
源码 SHA256 和第一组冻结 golden scores 位于 `toeplitz_reference/parity.json`。
原版权与 BSD 免责位于 `src/baselines/structured_erp.LICENSE`。

## 验证

2026-10-04 定向 suite：107 passed / 61.12 s；随后新增空间正交变换/幅值缩放的时间子空间检验：1 passed。
范围为：

```text
tests/models/test_riemann.py
tests/models/test_structured_erp.py
tests/models/test_temporal_transfer.py
tests/transfer/test_calibration_candidates.py
tests/transfer/test_target_recipes.py
tests/brainsync/test_calibrated_predictor.py
tests/research/test_erp_comparator.py
```

覆盖不只“能运行”：极性、时间反转、官方 golden、V/µV等尺度、C>T或N<feature、小类、空间秩缺损、channel permutation、非有限值、几何错配、缺导、旧schema、篡改state、fit范围外的标签/数值变化、同participant/同study泄漏、source basis缓存失配。
公共 comparator 使用真实 BDF 容器和实际 raw 独立滤波，包含 ERPcov、SLDA baseline、SLDA无baseline、标准Toeplitz、固定时间基、源时间基六臂；所有保存模型的分数逐值重载一致。
源时间基准备使用另一份 BrainSync BIDS；另有 Huebner BDF source→BrainSync target 个人包正向测试；后续推理的 `y=None`。
这些容器内部是合成测试信号，证明接口与数值正确性，不证明真实 EEG 迁移收益。

## 仍需科研验证

- R3 是明确可否证的研究原型。source ERP Gram 可能偏向高SNR时间模式，也可能学习源库伪迹；当前固定秩与各 participant 归一化不等价于已证最优的统计估计。
- 它对共同潜在时间子空间、正交空间变换及整体增益有合理不变性；不承诺任意空间投影、缺失潜在成分、时延变化或不同输入滤波下完全不变。
- 本支线未运行新十二库性能矩阵，未改变产品默认 family；算法之间必须使用真实 raw 重建的 matched cohort 和 source-only 超参选择。
- 源时间基不是物理 electrode EpochDataset，也不是 EA 输入；不能用它绕过 source checkpoint/EA 的独立角色门禁。
- 完整连续候选似然与算子传播协方差由主任务/其他支线实施，本记录不宣称已经由 ERPcov/Toeplitz 一并解决。
