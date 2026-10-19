# 审计分区 02：src/models 与 src/train

状态：已完成（2026-10-19，基于工作树；证据均来自实际阅读与受限数值检查）

## 检查计划

- [x] src/models/n2p3net.py：前向、归一化、池化、dtype/device（损失定义不在本分区）
- [x] src/models/adapters.py：适配器/域对齐、梯度流
- [x] src/models/candidate_evidence.py：候选证据聚合数学
- [x] src/models/parameter_groups.py：参数分组/优化器超参
- [x] src/train/runtime.py：种子/确定性、线程/显存预算、批源、OOM 恢复（checkpoint 不在本分区）
- [x] src/train/source_sampling.py：采样口径与泄漏
- [x] src/train/early_stopping.py：early stopping 与指标口径
- [x] 与 doc 公式一致性（输入契约第 3 节）
- [x] ruff check 目标范围

## 结论摘要

1. **P1 训练轨迹依赖显存余量。** 同一命令行、同一 GPU：`can_preload` 决定行置换用 CUDA 还是 CPU Generator，`choose_batch_size` 决定有效 batch 与更新步数。空闲显存不同 → 行序和优化步数不同；有效 batch 虽写入 `last_runtime`，但"同种子可复现"仅限同一分配结果。
2. **P1 `pooled_source` SplitBN buffer 的估计量与记录不符。** 记录称 `source_mixture`，实现是"每 batch 一次"的指数滑动平均（batch 加权，非样本加权），混合域 batch 上的域频率会进入目标 buffer。
3. **P1 目标 adapter 的启用状态不可从 checkpoint/record 恢复。** `_target_adapter_enabled` 是普通属性，`state_dict` 与 `feature_transfer_record()` 都不含它；缺少 `set_target_adapter_enabled` 时推理静默走 invariant 路径。
4. **P1 Dense PDS 采样器的尝试预算可被空类绕过，且 GPU 距离路径无内存准入/无回退。** 空 mingling class 分支把刚加的 attempt 减回去，与 docstring 明确相反；同时任选 CUDA 可用即分配 float64 源矩阵用 `torch.cdist`。
5. **P2 无严格确定性开关。** `lease()` 每次强制 `cudnn.benchmark=True`、TF32、`float32_matmul_precision("high")`；仓库无 `use_deterministic_algorithms`。文档已声明 CUDA 重训非逐位可复现，但没有供回归比较的可选 deterministic profile。
6. **P2 LMBC 常量三处重复**，`default_architecture_record` 硬编码默认窗/offset/温度，与实例 `architecture_record` 可能漂移；且 pooled 特征上的 "pre-stimulus" 参考窗含左边界零填充的卷积感受野（疑似，未消融）。
7. **P2 默认 unified 架构每次 forward 多两次全量拷贝；`iter_batches` 快速路径 yield 复用 buffer 的 view；`batches()` 不校验 indices 为完整置换；默认 `private_domain_count=0` 时 private 分支仍计算。**
8. **P2 early stopping 只提供计数，不提供 best-state 恢复**；`record(stop_at_patience=False)` 会把已耗尽 patience 写成 `epoch_limit`；`reference_policy` 的 `target_max_epochs=1` 与 `target_epochs=90` 矛盾，且除构造期自校验外无训练路径消费它。
9. **已验证无问题（数值/契约）：** observed-channel 路径与普通路径在全通道观测时输出完全一致（max|diff|=0）；SplitBN 目标校准多 batch 合并与直接总体矩 ≤1e-9；LMBC 时间基与 `_pooled_time_samples` 一致（44 帧、7 个参考样本）；感受野公式与 doc 一致；`ruff check src/models src/train` 通过；models/train 定向 150 项测试通过。

## 发现

### F-01（P1）有效 batch 与行置换顺序依赖运行时显存余量

**证据**
- `src/train/runtime.py:1172-1185`：`can_preload` 用**当时的** `free_bytes` 与 `preload_memory_fraction` 决定是否整矩阵上卡；`runtime.py:1187-1227`：`choose_batch_size` 用 `free_bytes * batch_memory_fraction - 5*参数字节` 与启发式 `activation_bytes_per_row` 压低 batch。
- `runtime.py:562-571`：`make_generator` 在预载时创建 `torch.Generator(device=self.runtime.device)`，否则 CPU Generator；`runtime.py:573-582`：置换因此在 CUDA 与 CPU 两种 RNG 上生成。
- 消费方 `src/baselines/deep.py:1897-1902` 调用 `choose_batch_size`，`deep.py:1109-1120` 记录 `batch_size`/`preloaded`。
- 文档只声明了 CPU/CUDA 随机训练不逐位相等（`doc/evidence/brainsync_recovery_20260908.zh.md:132-133`、`doc/evidence/cross_dataset_start_20260906.zh.md:95`），未声明"同机器空闲显存不同 → 行序与步数不同"。

**影响**：同一 seed、同一数据的两次运行，只要构建模型时的空闲显存不同（其他进程占用、fragmentation），首个 epoch 的置换和 batch 组成就不同，优化轨迹与指标不可直接配对比较；论文/账本若把 seed 当作复现键会失真。有效 batch 已记录，缓解了审计不可见的问题，但科学结论仍依赖硬件状态。

**建议**：把"数据顺序"与"batch 划分"从资源策略中拆出：置换始终在 CPU 上由 seed 生成再搬到 device；为报告的实验固定 batch（或固定 updates）并把 `can_preload`/`choose_batch_size` 只在显式 throughput 模式下启用；在结果账本记录"effective_batch_size + permutation_seed + preloaded"三件套（已有前两项可补第三项）。

**验证方式**：同一 seed 下分别强制 `preload=False` 与 `preload=True`，比较首个 epoch 的 `torch.randperm` 结果与 `last_runtime["batch_size"]`；或把 `can_preload` 打桩为 False 重跑短 fit，比较 state_dict 哈希。

### F-02（P1）`pooled_source` 的估计量记录为 source mixture，实现是 batch-mixture EMA

**证据**
- `src/models/splitbn.py:17-21`：`"pooled_source": "source_mixture"`，即记录宣称按源域混合估计。
- `splitbn.py:358-370`：`_update_pooled_buffer` 对当前 batch 的 `inputs.detach()` 调一次 `F.batch_norm`，动量固定（默认 0.1），**与 batch 行数/域组成无关**；`splitbn.py:482-484` 每个训练 forward 都调用；`splitbn.py:452-455` eval 时直接返回该 buffer。
- 对照：per-domain 分支 `_normalize_domains`（`splitbn.py:398-430`）按域更新，因此"记录里 source mixture、实现里 batch mixture"只影响 pooled 路径。

**影响**：当 `target_buffer_source="pooled_source"` 时，评估用的归一化统计是按"batch 出现次数"加权的指数平均，而不是按样本/域权重；若某域在小 batch 中更常出现（例如采样不均、最后不足一个 batch），它的均值/方差会被高估。记录字段 `estimand` 会让读账本的人以为这是样本混合。

**建议**：改为累加充分统计量（每域/全局 sum、sumsq、count）后按样本数合成，或在 record 中把 estimand 改名为 `batch_mixture_exponential` 并记录更新次数与各域累计行数；仅在 `target_buffer_source=="pooled_source"` 时更新 pooled buffer，避免无关臂上的额外 BN 前向。

**验证方式**：构造域组成交替的 batch 序列，比较 `pooled_running_mean/var` 与直接按行合并的矩；现有测试 `tests/models/test_splitbn.py` 通过，说明该语义未被测试锁定。

### F-03（P1）目标 adapter 的启用状态不在 state_dict / record 中

**证据**
- `src/models/adapters.py:528-529`：构造时 `_target_adapter_enabled=False`、target adapter 参数 `requires_grad=False`；`adapters.py:575-585`：两个 setter 只改内存标志/requires_grad。
- `adapters.py:800-818`：前向根据 `_target_adapter_enabled` 选择 `target_adapter(invariant)` 或 invariant；`adapters.py:922-934`：`architecture_record()`/`feature_transfer_record()` 只含 `FeatureTransferConfig` 与通道数，没有该标志（`adapters.py:228-241` 的 config record 同样没有）。
- 调用方必须自己恢复：`src/transfer/subject_adapter.py:362`、`372-376`，`src/transfer/calibrated.py:606`。

**影响**：用 `architecture_record` 重建模型或从 state_dict 加载后，如果调用方漏掉 setter，推理会静默走 invariant 路径（权重存在但不生效），属于"无报错的错误预测"。跨进程/跨语言（Lean 导出、评测脚本）重建时尤其危险。

**建议**：把 `target_adapter_enabled`（或声明式 `target_adapter_mode`）写入 `feature_transfer_record`，加载时校验并设置；或在 forward 中要求显式传入 mode，禁止隐式默认。

**验证方式**：启用 adapter → `state_dict()` → 新建 wrapper `load_state_dict` → 同一输入比较 logits，当前会不同（可写成反例测试）。

### F-04（P1）Dense PDS：空类不消耗 attempt、GPU 距离无准入/无回退

**证据**
- docstring `src/train/source_sampling.py:742-744`："Class draws that find an empty class also consume one attempt."
- 实现 `source_sampling.py:908-916`：先 `pds_attempts += 1`，空类时 `pds_attempts -= 1; continue`（注释还写"not a dart-throwing attempt"），与 docstring 相反；`source_sampling.py:922-923` 又用 `pds_fallback_count` 回填 `pds_attempts`，把两种计数混在一起。
- GPU 路径 `source_sampling.py:835-847`：只要 `torch.cuda.is_available()` 且数值范围安全，就分配 `source.float64` 与 `selected_tensor(n_select,P)` 到 GPU；`862-870` 用 `torch.cdist`；无 try/except OOM，也不经过 `GpuPerformanceScheduler` 的预算。

**影响**：（a）某 mingling class 先被抽空后，每次抽到它都做一次 `np.flatnonzero(eligible & (labels==class))` 的全量 O(N) 扫描而不推进预算；若该类的 mixture 权重高、其余类又持续被半径拒绝，循环会长时间空转，且记录里的 `pds_attempts` 不反映真实 proposal 数。（b）数据准备阶段可能因 float64 源矩阵 + 选中矩阵 OOM，且选择结果依赖 CPU/CUDA 距离后端（`> pds_radius` 的严格比较会放大末位差异），虽然 `pds_distance_backend` 已入 record，但重跑不再等价。

**建议**：空类也消耗 attempt（直接删掉 `-=1`），或单独记 `empty_class_draws` 并把已耗尽类从 mixture 中移除后重归一化；GPU 改为显式 opt-in，OOM 时回退 CPU 并记录；`pds_attempts` 与 fallback 分字段记录。

**验证方式**：两类样本 (1, 50)、`n_select=20`、`pds_max_attempts=5`，断言运行在有限步内结束且 record 中 `pds_attempts` 与实际 proposal 数一致（需加计数器）；GPU 路径用 `CUDA_VISIBLE_DEVICES=""` 重跑比较 selected_indices。

### F-05（P2）没有可选确定性 profile；`lease()` 强制开启非确定性后端

**证据**
- `src/train/device.py:32-45`：`cudnn.benchmark=True`、`cudnn.allow_tf32=True` / `conv.fp32_precision="tf32"`、`matmul.allow_tf32=True`、`set_float32_matmul_precision("high")`。
- `src/train/runtime.py:1065-1068` + `1077-1089`：每次 `lease()` 都调用 `configure()`，即每次 fit 都重新打开上述开关；全仓库 `grep` 无 `use_deterministic_algorithms`/`cudnn.deterministic` 设置（只有 `transfer/calibration_candidates.py:474-481`、`transfer/result_reuse.py:319` 在**记录**这些值）。
- 文档已承认非逐位可复现（`doc/evidence/brainsync_recovery_20260908.zh.md:132-133`）。

**影响**：不是新的正确性错误，但意味着无法用"同 seed 重跑"做数值回归/二分；`cudnn.benchmark` 的算法选择还会随机器负载变化，使同一 GPU 的两次短跑也可能有差异，放大 F-01 的配对比较问题。

**建议**：增加 `determinism="throughput"|"strict"` 策略：strict 下 `benchmark=False`、`cudnn.deterministic=True`、关闭 TF32、设置 `CUBLAS_WORKSPACE_CONFIG`、`use_deterministic_algorithms(True, warn_only=True)`；把 effective 开关写入 run record（现有记录代码可复用）。

**验证方式**：strict profile 下连续两次短 fit 比较 `state_dict` 逐位哈希；throughput 下允许差异但记录。

### F-06（P2）LMBC 默认参数三处重复；类方法 record 与实例可能不一致；参考窗含零填充感受野

**证据**
- 同一组默认值出现在：`src/models/n2p3net.py:455-459`（`LatencyMarginalContrastPool` 签名）、`n2p3net.py:600-603`（`N2P3Net` 签名）、`n2p3net.py:952-960`（`default_architecture_record` 硬编码 `[250,600]/[-200,0]/offset/temperature`）。实例 `architecture_record`（`n2p3net.py:1069-1079`）从 `self.pool[0]` 读取真实值，两条路径因此可能给出不同记录。
- `N2P3ArchitectureConfig`（`n2p3net.py:52-69`）不含这些 LMBC 参数，`model_kwargs()`（`115-132`）也不传；`src/baselines/n2p3net.py:145-161` 因此无法构造非默认窗，但 `baselines/n2p3net.py:166` 仍用类方法生成 record；checkpoint 加载方 `src/transfer/checkpoint.py:219-241` 会按 record 的窗/offset/温度重建。
- 时间基：pooled 特征首帧 `tmin + (st_pool-1)/(2*sfreq) = -188.28 ms`（`n2p3net.py:750-752`）；K35 的卷积半宽 17 样本≈133 ms，故前几帧的 receptive field 越过 -200 ms 边界并由零填充补齐。LMBC 的 `reference_window_ms=(-200,0)`（`n2p3net.py:601`）在 pooled 分辨率上包含这些帧，**并非严格的"刺激前真实样本"**（静态推导，未做消融）。

**影响**：前任一调用方若用非默认 LMBC 窗构造模型而用类方法写 record，checkpoint 重建会静默换回默认窗，前向不再等价；参考窗的 padding 成分会给 contrast 引入常数偏移（程度未测）。目前默认路径（baseline 不支持自定义窗）未触发，属潜在缺陷。

**建议**：把 `evidence_window_ms/reference_window_ms/latency_offsets_ms/latency_temperature` 纳入 `N2P3ArchitectureConfig` 并只由实例状态生成 record；增加"类方法 record == 实例 record"的测试；在 record 中声明 padding 约定，或让参考窗排除感受野被截断的前若干帧（在 doc 中至少写明）。

**验证方式**：用 `N2P3Net(..., evidence_window_ms=(300,700))` 构造后比较 `model.architecture_record()` 与 `N2P3Net.default_architecture_record(...)`；对参考窗替换为 -150..0 做一次短消融看 logits/指标变化（待 GPU）。


### F-07（P2）unified/observed-channel 路径每次 forward 多两次全量拷贝（数值等价已验证）

**证据**：`src/models/n2p3net.py:1141-1146` 先建 `(B,C)` bool mask 并做 `selected = x[channel_mask]`（高级索引=拷贝），`n2p3net.py:1162-1164` 再 `new_zeros(B,C,F,T)`、散射、`permute(...).contiguous()`（第二次拷贝）。该路径由 `N2P3ArchitectureConfig.unified()` 默认开启（`n2p3net.py:135-144`）。
**影响**：全通道观测（最常见）也付出 2 次额外全量搬运与峰值显存；相比普通路径（`n2p3net.py:1187-1194` 直接 `unsqueeze(1)`）是纯开销。用随机张量对比两条路径输出 `max|diff|=0.0`，所以不是正确性问题。
**建议**：当 `channel_mask is None` 或 `channel_mask.all()` 时走原 layout；仅存在缺失通道时打包。补一个"全观测快路径 == 打包路径"的测试防漂移。
**验证方式**：同一 batch 下 `torch.cuda.max_memory_allocated` 对比两种 config；现有 `tests/models/test_n2p3net.py` 未覆盖该等价性。

### F-08（P2）批源快速路径返回复用 buffer 的 view；`batches()` 不校验完整置换

**证据**
- `src/train/runtime.py:802-810` 复用 `self._shuffle_x/_y`，`850-858` 直接把 `self._shuffle_x[start:stop]` 作为 batch yield；下一 epoch `index_select(..., out=self._shuffle_x)` 会覆盖（`802-804`）。
- `runtime.py:695-696` 对 `indices` 只校验 `ndim==1` 且 `len==n_rows`，不校验唯一/覆盖；`iter_index_batches`（`1011-1017`）才有覆盖证明，`gather_indices`（`948-949`）才查重。
**影响**：任何跨 epoch 保留 batch 的调用方（调试快照、延迟 backward、跨 epoch 梯度累积、可视化）会看到数据被静默改写；`batches(indices=...)` 传重复索引会静默少训样本且不报错。
**建议**：要么 yield `clone()`（可配置），要么在 docstring 明确"batch 生命周期仅限当前 step"；`batches` 复用 `torch.sort` 覆盖校验或要求调用方传 `random_permutation` 的返回值。
**验证方式**：保存第 1 epoch 最后一个 `xb`，跑第 2 epoch 首个 batch 后比较其值；向 `batches` 传 `[0]*n_rows` 观察是否报错（当前不报）。

### F-09（P2）early stopping 的记录语义与 reference policy 不一致

**证据**
- `src/train/early_stopping.py:35-38`：`reference_policy()` 构造 `GlobalEarlyStoppingPolicy(target_max_epochs=1)`，而同一 budget 的 `target_epochs=90`（`20`）；除 `UnifiedTrainingBudget.__post_init__`（`early_stopping.py:25`）的自校验外，全仓库没有训练路径消费它，但一旦被用于目标 fit 会只跑 1 epoch。
- `early_stopping.py:102-108`：`stop_reason` 在 `enabled=True, exhausted=True, stop_at_patience=False` 时输出 `"epoch_limit"`，把"patience 已耗尽但被要求继续"误记为 epoch 限制；三种原因（固定预算/patience/epoch 上限）没有独立字段。
- `PatienceStopper` 只维护计数（`85-100`）；best 权重快照由消费方自行处理（`src/transfer/subject_adapter.py:1122-1123,1243-1251`），模块本身不提供恢复契约。
**影响**：账本中的 stop_reason 可能误导；reference policy 是潜在 1-epoch 陷阱；不同调用方对"best checkpoint"的实现可能不一致。
**建议**：删除 `reference_policy` 的 target 字段或加断言；record 增加独立 `stopped_by: "patience"|"epoch_limit"|"fixed_budget"` 与 `best_epoch`；把"保存/恢复 best state"提升为该模块的明确契约（或在 doc 中声明由调用方负责）。
**验证方式**：构造 `PatienceStopper(patience=1)`，observe 两次同值后 `record(stop_at_patience=False)`，断言当前 `stop_reason=="epoch_limit"`。

### F-10（P2）`is_oom_error` 用字符串匹配，可能把非 OOM 错误当作显存不足并静默降级

**证据**：`src/train/runtime.py:42-49`：先判 `torch.OutOfMemoryError`，否则 `"out of memory" in message or "cuda oom" in message or "显存溢出"`。该函数用于 `MatrixBatchSource.__init__` 预载回退（`runtime.py:493-502`）与 shuffle 回退（`614-625`、`839-848`），消费方还会据此重试/降 batch（`baselines/deep.py` 的 `oom_retries`，已入 record）。
**影响**：第三方库抛出的带 "out of memory" 文案的 `ValueError`/`RuntimeError`（非 CUDA）会被当成 CUDA OOM；裸 `MemoryError` 因消息为空反而不会命中，判定不一致。实验继续跑，但 preload/shuffle 路径与 batch 改变，根因被掩盖。
**建议**：优先按异常类型/后端错误码判定（`torch.OutOfMemoryError`、CUDA `CUDA_ERROR_OUT_OF_MEMORY`）；字符串兜底时记录匹配到的异常类型与消息，并把"字符串判定"写进 record。
**验证方式**：构造 `RuntimeError("host out of memory")` 调 `is_oom_error`，当前返回 True；补反例测试。

### F-11（P2）SplitBN 辅助函数恢复 train 模式会覆盖调用方混合模式；冻结标志不可持久化；加权合成用无偏方差

**证据**
- `src/models/splitbn.py:656-657` `model.eval()` → `731-734` `finally: model.train(was_training)`：按顶层布尔恢复，会把调用方刻意设成 eval 的子模块（如 E1 冻结 BN、`dropout_mode="checkpoint"`）全部改回 train。
- `splitbn.py:251-255`：`_freeze_pooled_buffer` 是普通属性、不进 `state_dict`；加载后继续训练无法复现"冻结 pooled buffer"的语义（消费方 `src/transfer/subject_adapter.py:634-663` 每次临时设置，默认路径安全）。
- `splitbn.py:432-445`：用 per-domain `running_var`（PyTorch 保存的是**无偏**方差）+ `running_mean²` 合成二阶矩，再 `E[x²]-E[x]²`；与 eval 直接用方差的约定相比有 O(1/n) 正偏，且各域权重只按声明权重、不按样本数。
**影响**：辅助函数副作用可能改变后续训练/推理行为；resume 时冻结语义丢失；加权 buffer 有轻微统计偏差（不影响默认 pooled 路径）。
**建议**：逐模块保存/恢复 `training` 标志；把冻结标志写入 checkpoint 元数据或显式构造参数；在 record 里写清 variance convention，或先把无偏方差换算为总体方差再加权。
**验证方式**：构造一个 BN 处于 eval、其余处于 train 的模型，调 `install_splitbn_target_calibration_buffers` 后检查该 BN 的 `training` 已被改为 True。

### F-12（P2）默认配置下的死计算与两个小契约缺口

**证据**
- `src/models/adapters.py:500-510` 构造 `private_projection`/`private_domain_head`；`adapters.py:805-806` 每次 forward 都算 `private = ELU(proj(base)); mean(-1)`，而默认 `FeatureTransferConfig.private_domain_count=0`（`adapters.py:169`）且 `private_domain_logits` 在禁用时直接抛错（`889-892`）。`private_projection` 仍被 `adaptation_parameter_groups` 收进 `source_extensions`（`563-567`），`SourceParameterPrior`（`src/models/parameter_groups.py:46-91`）会对它施加先验梯度。
- `parameter_groups.py:78-91`：`_center/_coefficient` 是普通属性，构造时固定在参数所在 device；之后 `model.to(...)` 不会搬它们，`loss()` 会 device mismatch（默认路径先 `trunk.to(device)` 再建 prior，见 `src/transfer/subject_adapter.py:188,232`，故未触发）。
- `src/models/candidate_evidence.py:313,427`：`membership.astype(membership.dtype, copy=False)` 是 no-op，疑似原本要做 dtype 规范化/校验但被删空。
**影响**：默认配置下每 step 多一次 1×1 卷积 + 均值；未使用参数进入优化器/先验账本，增加"这个参数为何在动"的审计噪声；prior 在设备迁移后不可用。
**建议**：`private_domain_count==0` 时不构造/不计算 private 分支；prior 的 `_center/_coefficient` 用 `register_buffer` 或 loss 时 `.to(values.device)`；删掉 no-op 或恢复真正的 dtype 校验。
**验证方式**：默认 config 下 profile 一次 forward 的算子数；`prior = SourceParameterPrior(model); model.cpu(); prior.loss()` 观察 device 错误。

## 已核对且未发现问题的部分

- **observed-channel 与普通路径等价**：全通道观测时 `max|logit diff| = 0.0`（临时脚本 `tmp/audit/scratch/check_math_02.py`），且梯度路径经 boolean index/scatter 正常。
- **LMBC 时间基**：`n_times=179, st_pool=4` → 特征 44 帧；`_pooled_time_samples(179,4)=44`；每帧中心 `-188.28 + j*31.25 ms`；参考窗命中 7 帧（≥2 的契约成立）；5 个 offset 全部 active。与 `doc/input_contract_math.zh.md:301-324` 的感受野公式逐项一致（默认 RF 54/102 样本 = 414.06/789.06 ms；256 Hz 缩放 129/9/33 与代码 `scale_odd_kernel_preserving_span` 一致）。
- **SplitBN 目标校准合并**：多 chunk/多 minibatch 的 Welford 合并与直接总体矩比较，mean/var 误差 ≤9.3e-10（`tmp/audit/scratch/check_calib_02.py`），说明 `splitbn.py:692-717` 的 `delta*delta_after*count` 实现正确。
- **SplitBN per-domain running_var** 与 `torch.var(unbiased=True)` 一致（相对总体方差约 3% 偏差，符合 PyTorch 约定），不是 bug。
- **候选证据聚合**：`sum/mean/tempered/trim0.2/precision` 的数学、空候选 `-inf` 弃权、`1e-12` 平票弃权、`center_logits` 只在组内 >1 行时执行，均与 docstring 一致；未发现数值/泄漏错误。
- **最大范数空间卷积**：`effective_weight` 的逐输出核 L2 裁剪可微且 `clamp_min(1e-12)` 防零除；`max_norm` 仅作用于前向有效权重（记录的是原始权重，已由参数分组捕获）。
- **工具与测试**：`.venv\Scripts\ruff.exe check src/models src/train` 全绿；定向 pytest（`tests/models/test_n2p3net.py`、`test_splitbn.py`、`test_candidate_evidence.py`、`tests/train/test_runtime.py`、`test_global_early_stopping.py`、`test_source_sampling.py`）150 项、退出码 0（汇总行被 pytest 配置抑制，按测试点计数）。

## 未能验证与存疑

- **损失函数不在本分区**：`transfer/decision_loss.py`、`baselines/n2p3net.py`、`baselines/deep.py` 中的 CE/加权/决策级损失与文档的损失口径未核对；本报告只覆盖模型内部与 runtime 契约。
- **checkpoint 读写**在 `transfer/checkpoint.py` 与 `baselines/deep.py`；本报告只核对了 SplitBN/LMBC record 的消费点，未验证保存原子性、版本迁移与失败恢复。
- **GPU 行为未实测**：无 GPU 训练/长任务，F-01/F-04/F-05 的 GPU 差异与 OOM 回退均为静态推导；`benchmark` 与 TF32 的实际数值影响未量化。
- **LMBC 边界 padding 的影响未消融**：F-06 中"参考窗包含零填充感受野"是静态时间基推导，程度未测；需要 GPU 上对比"排除前 k 帧"的短实验。
- **`pooled_source` 偏差量级未测**：F-02 的域频率偏置需要真实源训练 batch 序列才能量化；本次只验证了更新公式本身。
- **`SourceParameterPrior` 的设备迁移**未在真实 `subject_adapter` 流程复现（仅静态确认默认顺序安全）。
- **指标口径**（AUC/BACC/校准）在 `baselines/evaluate.py`，不属本分区；early stopping 只消费 finite loss，未验证其与最终指标的单调关系。
- 工作树含 80 modified/118 deleted/230 untracked；本报告按工作树阅读，`src/models/parameter_groups.py` 与 `src/train/early_stopping.py` 为 untracked，`adapters.py/candidate_evidence.py/n2p3net.py/runtime.py/source_sampling.py` 为 modified（`git status --porcelain`）。

## 已检查范围

| 文件 | 行数 | 阅读方式 |
|---|---|---|
| `src/models/n2p3net.py` | 1278 | 全文分 4 段精读（前向/池化/记录/observed 路径） |
| `src/models/adapters.py` | 865 | 配置、record、残差 adapter、feature_views/forward 精读 |
| `src/models/candidate_evidence.py` | 420 | 全文精读（聚合、弃权、precision、group 决策） |
| `src/models/parameter_groups.py` | 89 | 全文精读 |
| `src/models/splitbn.py` | 701 | 全文精读（BN 语义、buffer 生命周期、校准安装）；不在派发清单但为 BN 语义必需 |
| `src/train/runtime.py` | 1109 | 设备/预算/批源/调度器精读；线程与 OOM 路径 |
| `src/train/source_sampling.py` | 877 | 概率、统计、TSD 与 Dense PDS 全读 |
| `src/train/early_stopping.py` | 87 | 全文精读 |
| `src/train/factory.py`、`src/train/device.py` | 154/52 | 全文精读（构造路径与设备策略） |

交叉证据：`src/baselines/deep.py`（lease/choose_batch_size 消费点）、`src/baselines/n2p3net.py:145-192`（record 消费）、`src/transfer/checkpoint.py:219-241`（LMBC 重建）、`src/transfer/subject_adapter.py`（adapter 状态与 prior 构造）、`doc/input_contract_math.zh.md:301-340`（公式核对）、`doc/evidence/brainsync_recovery_20260908.zh.md:132-133`（确定性声明）。

命令：`.venv\Scripts\ruff.exe check src/models src/train`（0 问题）；定向 pytest 150 项（6 个文件，退出码 0）；两个只读数值脚本（`tmp/audit/scratch/check_math_02.py`、`check_calib_02.py`）。未运行训练、未安装依赖、未修改仓库文件。

