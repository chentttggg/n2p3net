# 性能审计分区报告（12-performance）

- 审计对象：`D:\BCI\n2p3-net` 当前工作树（HEAD `f1a303c`，含未提交改动）
- 审计方式：只读（read/grep/glob、只读 git、`python -c`、tmp/audit/scratch 下微基准）；未修改任何现有文件
- 级别：P0 = 已造成或必然造成数量级吞吐损失/资源承诺失效；P1 = 显著吞吐损失或证据可复现性风险；P2 = 局部优化
- 每条结论均给 `文件:行号` 证据；实测数字来自本机（32 逻辑 CPU / 31.8 GiB RAM / RTX5070 8 GiB / `torch 2.13.0+cpu`，**CUDA 不可用**）

## 1. 结论摘要（按影响排序）

| # | 级别 | 结论 | 证据 |
|---|------|------|------|
| C1 | P1 | E2 评分每轮重建进程池并把整份 EpochDataset 通过 pickle 发给每个 worker；4 组嵌套后最多 16 个进程争 4 个 GPU 槽，12 个常驻空转 | `experiments/run_source_training_chain.py:405-406`、`:494`、`src/research/source_execution.py:219` |
| C2 | P1 | 报告 JSON 呈 O(n²) 写放大：每 5 秒（即使零进展）与每任务/每次选择后重写整份 report，`indent=2` + `fsync`；仓库已有 171.9 MB 单文件 | `run_source_training_chain.py:511-523,783,791,912,941,1302,1308`；`src/research/execution.py:42-53` |
| C3 | P1 | 每 optimizer step 对 22 个参数张量逐个 `torch.isfinite(grad).all()` 并串行 `&` → 每步多约 44 个 kernel；CPU 实测 +4.8%，GPU 上 kernel 占比更高 | `src/baselines/deep.py:1318-1321`；实测 46.78→49.03 ms/step |
| C4 | P1 | 每个评分/外层任务结束都 `torch.compiler.reset()` + `gc.collect()` + `empty_cache` + `ipc_collect`，摧毁编译缓存与分配器复用 | `src/research/source_execution.py:78-85`；`src/train/device.py:48-60`；调用点 `run_source_training_chain.py:376,398,1346` |
| C5 | P1 | 整份 CPU 矩阵 `pin_memory()`：一次性 423 MB 页锁定副本；且随机批走 `index_select` 后不再是 pinned，`to_device` 永久退回同步拷贝 | `src/train/runtime.py:508-517`、`:1237-1239`、`:333-339`、`:904-918` |
| C6 | P1 | `cloud_resource_floor` 只在唯一入口检查；`configs/cloud_training_resources.json` **未入库**，且没有任何 `src/experiments` 代码读取它 → 换机复现时 floor 静默变 None | `src/research/source_execution.py:206-213`；`git ls-files` 报 pathspec 不匹配 |
| C7 | P1 | `available_cpu_threads()` 在 Windows 无 affinity/cgroup，且不读 Job Object CPU 限额；被直接当 worker 上限用（32 个目标 worker） | `src/train/runtime.py:75-90`；`run_source_training_chain.py:605`；`run_candidate_promotion_matrix.py:3214` |
| C8 | P2 | 并发 worker 的显存预算靠调用方显式声明，声明点只有 2 处；E2 适配路径不声明，4 并发时各自按"独占整卡"计算 | `runtime.py:1127-1137,1184,1222`；`baselines/evaluate.py:541-544`；`research/source_selection.py:383,479` |
| C9 | P2 | attested 加载全量复制 + 重算 SHA + 105% 空闲门槛；实测 SHA 1.41 GB/s、复制 2.09 GB/s（0.2 s/157 MB），时间不是问题，**磁盘空间与每进程一份临时副本**才是 | `src/data/epochs.py:1073-1099,1210-1255`；实测见 §4.2 |
| C10 | P2 | 41 GB / 40956 文件 / 4398 目录：27 GB 原始交付 `.tar.gz` 在 runs 下，`shards.tar` 与解包 `shards/` 双份，22 份 checkout 贡献 6681 个 `.py` | 实测清单见 §5 |

## 2. 方法与限制

- 本机 `torch 2.13.0+cpu`，`torch.cuda.is_available()==False`：所有 GPU 结论只能由代码路径 + 预算算式推导，**不能**在本机复测 5090 数字；`experiments/benchmark_unified_training.py` 在本机不可运行（`--device cuda`）。
- 未运行训练、未安装依赖、未跑全量 pytest；仅跑 3 个 ~10 s 级只读探针（npz 加载/哈希/复制、单步扫描开销、模型参数计数）。
- 只读 git：`git status --short` / `git log -1` / `git ls-files --error-unmatch`。

## 3. 并行与资源计划

### 3.1 计划确实透传到进程（结论：基本透传，2 处断点）

链路：`configs/cloud_training_resources.json` → plan 内联字段 → `resolve_source_resources` → 子计划 → 实际 worker。

- 预算：`budget = min(available_cpu_threads(), cpu_budget)`（`src/research/source_execution.py:168`）。本机 32 核、预算 25 → 25，符合 25 CPU 下限。
- 组数：`groups = min(group_count, budget, source_workers=4)`（`:172`）→ `ProcessPoolExecutor(max_workers=resources["source_workers"])`（`run_source_training_chain.py:494`）。**透传成立**。
- 每 worker 原生线程：`native_threads = budget // max(groups, target) = 25//4 = 6`（`:214`）→ 子计划 `_native_threads_per_worker`（`run_source_training_chain.py:501`）→ `cpu_thread_budget(6)` 包裹整个组（`:462`）。**透传成立**，4×6=24≤25，与文档 `doc/engineering.zh.md:308` 一致。
- 组内 GPU 子 worker：`inner_workers_per_group = min(target, (budget-groups)//min(groups,4)) = 4`（`:219`）→ `child["target_workers"]=4`（`run_source_training_chain.py:505`）→ `_score_e2_candidates(workers=4)`（`:897`）与外层评估池 `max_workers=workers`（`:1304`）。**透传成立**。
- 断点 A：GPU 槽位 `BoundedSemaphore(4)`（`:491,372`）限制**并发**，不限制**常驻**。4 组 × 4 子 worker = 16 进程各自已 pickle 一份数据集并驻留，只有 4 个能进 `gpu_task_slot`（`source_execution.py:67-75`）。空转 12 个。
- 断点 B：`HostMemoryAdmission.target_workers` 在余量不足时返回 1（`source_execution.py:146`：`workers = int(allowed) if allowed >= 2 else 1`），**静默降级为单 worker**，而 `report["resource_policy"]` 只记录静态 `resources`（`run_source_training_chain.py:483`）。实际生效值仅写入 `scoring_progress.json`（`:417-419`），链级报告里看不到。这与 `doc/engineering.zh.md:309`"不能静默串行"的承诺存在证据缺口。
- 断点 C：CPU 记账把 1 个 source group 记 1 个 CPU、1 个 target worker 记 1 个 CPU（`source_execution.py:127,143-149`）。真实占用是 group 6 线程 + 子 worker 1 线程 ≈ 28 线程，记账 4+3=7。准入因此几乎不约束 CPU（真正的约束来自静态 `budget // max(groups,target)`）。结论：**嵌套进程没有被双重计数，而是被严重低计**；总线程 28 vs 预算 25，超 3 个（12%），可接受但应记账修正。

### 3.2 32 GB 卡上 4 GPU worker 的显存预算可达性（结论：可达，但预算不自洽）

以 32 GB 卡、`free=30 GB`、`gpu_task_memory_gib=4`、`gpu_workers=4` 代入：

- `slots = max(1, min(25, int(0.8*30/4))) = 6` → 被 `gpu_workers` 截为 4（`source_execution.py:194-201`）。
- 每 worker 常驻预算 `resident = 0.90*30/4 = 6.75 GB`（`:223-224`），4 份共 27 GB，留 3 GB 给 4 个 CUDA context（约 0.3–0.5 GB/个）+ 碎片。**偏紧但可达**。
- `can_preload` 用 `min(resident-allocated, free-512 MiB)`（`runtime.py:1180-1183`）。本仓库实测最大单份合并缓存 `domain_transfer_cloud_20260916/inputs/*.npz` 为 422.9/204.8/149.9/129.7 MB，4 份 preload 合计 < 1 GB，**不构成压力**。
- `choose_batch_size` 在该卡上给出 `0.55*30/4 = 4.125 GB/worker`（`runtime.py:1222-1227`）。模型实测仅 **1279 参数（5 KB）**；`activation_bytes_per_row = 6*179*4*16 = 68.7 KB`（C=6）到 `64*179*4*16 = 733 KB`（C=64）→ 可容纳 5 600–60 000 行，`cap=512` 永远先命中。**该启发式在本项目不构成约束，也不产生收益**（16× 保守系数纯属浪费，但无害）。
- 真正的风险在 `can_preload`：它读设备级 `free`，安全性完全依赖每个 worker 都先调用 `configure_shared_worker_budget(4)` 与 `configure_resident_memory_budget(6.75 GB)`。声明点只有 `baselines/evaluate.py:541-544`（`_fold_result`）与 `research/source_selection.py:383,479`（source 轨迹）。**E2 适配路径（`fit_calibration_candidate`，`run_source_training_chain.py:313`）不声明**，若 4 个 E2 worker 同时适配，各自按 0.55/0.30 独占整卡计算；当前模型太小不会 OOM，换大模型即超订。
- `recommended_concurrent_workers` 在 <24 GB 时返回 1（`runtime.py:1167-1170`），本机 8 GB 卡因此安全退化为单 worker；但它不校验"4 × 每 worker 预算 + context ≤ 总显存"，超订只能靠调用方自觉。

### 3.3 fail-closed 覆盖（结论：内存/floor 是 fail-closed，worker 数是静默降级）

- 内存不足：`ValueError("available host memory is below the declared worker requirement")`（`source_execution.py:188-189`）。本机实测 `avail=18.7 GiB`、`host_budget=17.8 GiB`、`group_budget=17.8-10=7.8 GiB < 18 GiB` → **本机直接拒绝运行**。这是正确的 fail-closed；同时也说明 18 GiB/组 × 4 组 + 10 GiB staging 的声明只有在 ≥ 92 GiB 空闲的机器上才成立（`86 GiB/0.95 + 10`），换小机器必须下调 `source_worker_memory_gib` 或组数。
- CPU/GPU 下限：`source_execution.py:206-213` 在 `budget < 25`、`slots < 4`、`target < 4`、`groups < min(group_count,4)` 任一成立时 `raise`。**正确**，但见 3.4。
- OOM 批回退：`deep.py:770-772,1153-1161` 每次把 batch 减半并**从头重跑整个 fit**，直到 1；属于 fail-safe，代价是全部已完成更新作废（记录在 `oom_retries`）。可接受，但应在 chain 报告里汇总 `oom_retries` 以区分"吞吐差"和"显存不足"。

### 3.4 floor 检查的入口覆盖（结论：只覆盖 1/6 入口）

`grep cloud_resource_floor` 仅命中 `src/research/source_execution.py` 与测试。实际接入只有 `experiments/run_source_training_chain.py`（经 `_execute_source_groups`）。以下入口既不解析 `configs/cloud_training_resources.json` 也不检查 floor：`run_domain_transfer_matrix.py`、`run_pretrain.py`、`run_pretrain_supervised.py`、`run_cross_dataset_transfer.py`、`run_eeg_loso.py`、`run_candidate_promotion_matrix.py`。

更严重的是 `configs/cloud_training_resources.json` **未被 git 跟踪**（`git ls-files --error-unmatch` → pathspec 不匹配），而 `tests/train/test_source_execution_resources.py:214` 直接 `read_text()` 它 → 干净克隆上该测试必然 `FileNotFoundError`；云端复现也必须手工内联这些字段，floor 一旦漏写就是 `floor is not None` 为假、静默不检查（`source_execution.py:206`）。

### 3.5 Windows CPU 高估与预算旁路

`available_cpu_threads()`（`runtime.py:75-90`）在本机返回 32（`os.cpu_count()`；Windows 无 `sched_getaffinity`，`_linux_cgroup_cpu_quota_threads` 直接 `return None`）。不读 Windows Job Object 的 CPU rate 限制（容器/CI 常用），也不区分 P/E 核；只要外部用 Job Object 限核就会高估。被当上限直接使用的两处：

- `run_source_training_chain.py:603-608`：`target_workers` 为 `auto/None` 时 `workers = available_cpu_threads()` → **32 个目标 worker**，绕过 25 预算（只在非 source-group 模式下触发；group 模式已写死 4）。
- `run_candidate_promotion_matrix.py:3214`：`cpu_quota = max(1, int(available_cpu_threads()))` 同类问题；`:1464` 的 `resolve_cpu_threads(n)` 未传 `total_threads` → `threads_per_worker = 64/n`（本机 32/n），同样绕过预算。

## 4. 数据与 IO

### 4.1 epoch 缓存格式（压缩比 vs 加载成本，实测）

- 分片缓存以 `compressed=False` 写出（`src/data/moabb.py:1201`），合并缓存默认 `compressed=True`（`epochs.py:939-943,1036`）。
- 实测 `domain_transfer_cloud_20260916/inputs/c03349a42f4b4a8a6177.npz`（157.2 MB，42 个成员，全部 deflate）：`X` = 57 437×6×179 float32（246.7 MB 原始）→ `np.load(...)['X']` **0.632 s**（90 946 行/s），含全部成员时压缩比 2.28。
- 分片实例：`Chailloux2020/shards/*.npz` 单个 7.23 MB，`X`=1791×6×179 float32（7.69 MB），`X.npy` 压缩后 7.10 MB（**仅 1.08×，compress_type=8**）。即对 float32 EEG，deflate 常只省 8%，却阻断 `mmap`、强制每次全量解压。
- 影响：每次 `load_epoch_dataset` 都把 `X` 实体化（`epochs.py:1418`），`release_cpu_after_preload` 只能在 H2D 之后才释放（`deep.py:1929-1948`、`runtime.py:504-507`）。因此一个 2 GB 级合并缓存 = 每进程 2 GB 常驻 + 一次约 5 s 解压。

### 4.2 attested 加载：复制 + SHA 的量化

- 路径：`_epoch_cache_archive_source` 打开 fd → 全量复制到 `TemporaryFile` 同时算 SHA → `np.load(snapshot)`（`epochs.py:1215-1255`）。要求空闲 `byte_size*1.05`（`:1088-1099`）。
- 实测（同一 157.2 MB 文件）：SHA-256 **1.41 GB/s**（0.111 s），顺序复制 **2.09 GB/s**（0.075 s），`np.load` 解压 0.632 s。合计 ≈ 0.82 s/157 MB，即约 **5.2 s/GB**。
- 结论：CPU 成本不是主因；主因是（a）每进程一份完整临时副本（4 组并发 = 4× 缓存大小），（b）105% 空间门槛在小盘上直接 fail-closed。
- 可直接消除：`os.dup(fd)` 后 `np.load(os.fdopen(dup,'rb'))`，配合复制前后 `fstat` 比对（现有 `_file_identity` 检查已具备）即可在**零拷贝**下获得同样的"稳定字节"证据，省掉复制与 105% 空间要求。

### 4.3 float64→float32 与整矩阵复制

- 正常路径已经是 float32：`save_epoch_dataset` 强制 `np.asarray(dataset.X, dtype=np.float32)`（`epochs.py:969`），加载再 cast（`:1461`），因此 loader 之后无 float64。
- 但每次 fit 仍有 3 个全矩阵 pass：`_prepare_input` 的 `np.subtract(out=prepared)`、`np.divide(out=prepared)`、`np.copyto(where=~mask)`（`deep.py:668-675`），即分配 423 MB 再写 2 遍。可合并为单 pass（先 mask 后标准化，或 `np.multiply(...); np.add(...)` 用同一 buffer）。预期省 0.2–0.4 s/次 fit（423 MB 量级），风险：需保持"先减均值再除标准差再按 mask 归零"的现有语义与既有测试。

### 4.4 逐样本循环 / MatrixBatchSource / gather_indices

- `MatrixBatchSource` 内部无逐样本循环，仅 batch 循环（`runtime.py:698-725`），与 self-claim 一致。
- `gather_indices` 每批做：`torch.unique(indices)`（排序）、越界 `.any()`、4–5 次 `index_select`、4–5 次 H2D（`runtime.py:940-984`）。固定更新轨迹每步都走这条路径（`deep.py:1345-1348`）。对 512 行、C=6 的批：`unique` 约 20–30 µs、4 次 H2D 约 2.3 MB。属于 P2：可把 `unique`/越界校验降为"每 epoch 一次"或只在 `require_full_coverage` 时做。
- 预载快速路径（一次 `index_select` 换整轮）默认关闭：`materialize_shuffled_epoch` 默认 `False`（`deep.py:259`），且还需 `decision_context_active` 为假、`can_preload` 二次通过（`:1976-1982`）。即默认仍是"每批 gather"，省内存但多 N−1 次稀疏读。对 423 MB 预载矩阵，开启后每 epoch 省 (n_batches−1) 次 gather，预期 5–15% 数据侧时间；风险：多一份全矩阵显存（可由 `can_preload` 守门）。
- `pin_memory`：见 C5。整矩阵 pin 之后，随机批经 `index_select` 得到的是普通可分页张量（`runtime.py:904,918`），`to_device` 检查 `_is_pinned_tensor`（`:1237-1238`）→ `non_blocking=False`。**训练主路径（固定更新 + 随机行）从未获得异步 H2D**，却付出 423 MB 页锁定 + 一次全量拷贝。顺带：4 进程各 423 MB 页锁定内存不可换出，直接吃掉 18 GiB/组的 host 预算。

### 4.5 缓存键与失效

- 缓存身份由 `write_epoch_dataset_record` 写侧 `_sha256_file` + `dataset.record()` 全字段生成（`epochs.py:1258-1278`），读侧 `_validate_attested_cache` 逐字段比对（`:1281-1300`）。键包含 byte 级 SHA 与预处理记录 → 失效语义正确，不会读到陈旧缓存。
- 代价：任何"派生缓存"（`prepare_*_cache.py` 系列）都要重新解压 + 重算 SHA + 全量校验，一次派生 ≈ 1 次读 + 1 次写 + 1 次 SHA。属可接受的正确性开销。

## 5. 训练 / 评估吞吐

### 5.1 AMP / compile / fused Adam（配置正确，路径有坑）

- BF16 选择：`resolve_precision` + `_supports_bf16`（`runtime.py:369-405`），TF32/cudnn.benchmark 在 `optimize_device_for_training`（`src/train/device.py:32-45`）。方向正确。
- `fused=True` 仅在 `optimizer_execution.fused_adam` 时传入（`deep.py:1282-1287`）；`capturable=True` 只在 CUDA Graph 模式（`:1285-1287`）。
- `torch.compile(train_step, fullgraph=False)` 把 `zero_grad/step` 一起编译（`deep.py:1305-1326`）。对 1279 参数模型，`reduce-overhead` 的 CUDA Graph 收益主要在 kernel launch；但 `fullgraph=False` 会因 Python 侧分支（`finite` 循环、`diagnostics` dict）产生 graph break，实际收益需实测。**没有实测门禁**（见 5.4）。
- 每步额外 device 同步：`loss_sum += loss.clone().float()`（`:1354`）与 `finite_updates.logical_and_`（`:1355`）本身不触发同步；真正的同步点是 `bool(finite_updates & state_finite & ...)`（`:1364`，仅 report 时）与 `float(loss_sum.cpu())`（`:1379`，每 100 步）——**同步策略是节制的**，这点符合 `host_sync_policy: epoch_boundary`。

### 5.2 每步梯度有限性扫描（C3，实测）

`deep.py:1318-1321`：`finite = isfinite(loss); for p in model.parameters(): finite = finite & isfinite(p.grad).all()`。参数张量 22 个 → 每步约 44 个额外 kernel（且读取全部梯度）。本机 CPU 实测（batch 512，6×179，4 线程，20 步均值）：无扫描 46.78 ms/step，有扫描 49.03 ms/step → **+4.8%**。GPU 上单步只有 1–3 ms，22 个 5 µs 级 kernel 的占比会显著高于 4.8%（预计 5–20%）。建议：把参数级扫描降到 `report_due` 频率（100 步；`:1358-1368` 已有同样的 state 扫描），保留逐步的 `isfinite(loss)`（1 个 kernel）。语义影响：NaN 梯度最多多应用 100 步，而现有实现本来也是"先 step 再报错"，检测时点改变但都不会回滚，风险可控；需补一个"注入 NaN 梯度仍失败"的合同测试。

### 5.3 E2 评分与外层评估的固定开销

- 每个 cohort 任务都新建 `PreparedTransferCheckpoint(path, dataset)` 并 `_sha256(path)`（`run_source_training_chain.py:290-292`）；每个 cell 两次 `_model_state_digest(trunk)`（`:308,326`）。文档记录该路径从 27.31 s→4.24 s/4 单元（`doc/engineering.zh.md:396-398`），当前 ≈1.06 s/单元固定成本，其中模型只有 1279 参数，说明开销在**合同/摘要/反序列化**而非算力。E2 单元数 = `candidates × cohort × seeds × axes × outers`，按 1 s/单元即可估算总时长，建议把 checkpoint 准备与 trunk 摘要按 (checkpoint, subject) 记忆化。
- `matched_input_digests` 只哈希 test 行（`src/transfer/binary.py:126-157`），不是全量 X，**没有 O(size) 放大**；这条与"合同校验 1.57 s"的历史问题不同，属已优化项。
- 外层评估：`_evaluate_outer_group` 在每个池 worker 内 `load_epoch_dataset(target_cache, attested)`（`:1327`），4 worker = 4 次 attested 加载（4 次复制 + SHA + 解压）。同理，`_execute_source_groups` 每个 (outer, layout) 组各加载一次同一 `outer["source_cache"]`（`:668`），2 layout 时同文件被加载 2 次。建议：目标/源缓存改为进程启动时一次性 `mmap` 只读共享（`multiprocessing.shared_memory` 或 `np.load(mmap_mode='r')`，后者要求非压缩格式）。

### 5.4 checkpoint I/O 与 resume 扫描

- `.pt` 共 4028 个 / 2.29 GB（均值 0.57 MB），单文件小，写入成本可忽略。
- resume 对每个 checkpoint **读两遍**：`SourceChainResume.verify` 全文 SHA（`src/research/source_resume.py:64-71`），随后 `load_checkpoint_payload` 再 `torch.load` 全文（`src/transfer/checkpoint.py:56`，`weights_only=False`）。全量 resume 的读放大 ≈ 2 × 2.3 GB ≈ 4.6 GB（顺序读，约 2–4 s），可接受；但 `verify` 已有 (size, mtime) 缓存，同一进程内重复调用不会重算。属 P2：可改为"边 `torch.load` 边校验"，或对 resume 只校验将要使用的 checkpoint。

### 5.5 基准方法学（`experiments/benchmark_unified_training.py`，未入库）

- 冷热分离**正确**：`samples[0]` 为 cold，`statistics.median(samples[1:])` 为 warm（`:64-76,110-111,128-129`）；每轮前 `synchronize`、计时含 `fit+predict`。
- 数值核验**到位**：`packed_prior` 与 scalar oracle 的值与梯度逐元素比对（`:52-62`）。
- 覆盖不足（脚本自身在 `scope` 里声明了，`:135-136`）：
  1. `X=(200, 6, 179)`、`batch_size=512` → **每 epoch 只有 1 个不足批（200 行）**，完全测不到 batch 调度、`gather_indices`、`pin_memory`、H2D 摊销——恰是 §4.4 的问题所在。
  2. 只有 6 通道，真实源 6–64 通道，激活/带宽差 10×。
  3. 无多 worker 并发（4 worker 的显存/带宽争抢、E2 池重建）测量。
  4. `compile_mode` 的编译时间被 warm 排除（合理），但**没有同时报告 cold**，无法估算一次性编译成本是否值得。
  5. **无回归门禁**：`--output` 必须指向新路径（`:150-151`），脚本不与任何基线比较、不设阈值、CI 不调用 → 性能回退不会被发现。建议加 `--baseline` 与 `--max-regression`，并对 `median_warm_seconds` 做 ±10% 门禁。

## 6. 存储吞吐（实测清单）

`experiments/runs`：**40.98 GB / 40 956 文件 / 4 398 目录 / 122 个 >20 MB**。

| 类型 | 文件数 | 体积 | 说明 |
|------|--------|------|------|
| `.gz` | 96 | 26.93 GB | `source_eog_20260909/delivery/*.tar.gz`（BI2013a 4.51、Huebner2017 4.48、BI2015a 4.06、EPFLP300 2.24、BI2014a 2.02、GTN 1.74 GB）+ 1.81/1.52 GB 结果包 |
| `.tar` | 11 | 4.31 GB | 与已解包目录**双份**：如 `local_prepared/Chailloux2020/shards.tar` 2.83 GB 与 `shards/` 2.63 GB 并存 |
| `.npz` | 17 312 | 5.43 GB | 多为分片缓存（Chailloux 382 个/2.63 GB；Huebner 342 个/0.42 GB） |
| `.pt` | 4 028 | 2.29 GB | 检查点 |
| `.json` | 9 780 | 1.53 GB | 最大单文件 **171.9 MB**（`unified_cloud_20260916/analysis/matrix_sources_source_chain_record.json`） |
| `.py` | 8 134 | 0.13 GB | 22 份 `*/checkout/` 合计 **6 681 个文件**（多数为 `.py`，共 0.11 GB），即约 16% 的文件数是源码副本 |
| `.pyc` | 614 | 0.01 GB | 源码副本的副产物 |

影响与建议：
1. **原始交付包移出 runs**（27 GB / 96 文件）：`frozen/` 或外部归档 + `sha256` manifest 留在 Git。收益：runs 41 GB→14 GB；任何全树备份/同步/杀毒的时间与 27 GB 成正比。风险：需先校验 manifest 再移动（`doc/engineering.zh.md` 已有归档流程，照做即可）。
2. **解包后删除 `shards.tar`**（2.83 GB）或保留 tar 删除解包目录——二者只需一份。收益 2.8 GB + 1 次 inode 扫描。
3. **checkout 改为 manifest 引用**（22 份→0 份）：文件数 40 956→34 275（−16%，按 checkout 总文件数计），目录数同步下降。收益：Windows 上每次全树 `stat`/杀毒扫描约与文件数线性。
4. 写放大：每次 `_save_chain_report` 的 `indent=2` 让 171.9 MB 记录的实际字节数比紧凑 JSON 多约 20–35%；叠加 C2 的重写频率，单次运行可产生数十 GB 无效写。压缩/紧凑编码 + 只写增量可同时解决。

## 7. 量化改进建议（收益 / 验证 / 风险）

| ID | 改动 | 预期收益 | 验证方法 | 风险 |
|----|------|----------|----------|------|
| R1 (C4) | `torch.compiler.reset()`/`empty_cache` 从每任务移到进程退出/空闲阈值；仅在 `mem_get_info` 低于阈值时清理 | 每次 E2 调用省 16×(gc+empty_cache)≈1–5 s；保留编译缓存避免重编译 | 对比 `worker_progress.json` 时间戳间隔、`nvidia-smi` 采样 | 常驻显存升高 → 用阈值守门 |
| R2 (C1) | E2 池按组复用（`_execute_source_groups` 级别），数据集用 `shared_memory`/`mmap` 只传句柄 | 去掉每次 (seed×axis) 的 N×423 MB pickle；16→4 常驻进程，host RAM 省 3–5 GB | 记录 worker 启动到首个 cell 的延迟 | 生命周期/清理复杂；Windows `shared_memory` 需显式 unlink |
| R3 (C2) | 报告写盘：状态变更才写 + ≥60 s 节流 + 紧凑 JSON；`fsync` 仅收尾 | O(n²)→O(n)；171.9 MB 记录每轮省 1–3 s 与等量写盘 | 统计写盘次数/字节 | 崩溃最多丢 60 s 进度（检查点目录才是证据，可接受） |
| R4 (C3) | 参数级 `isfinite` 降到 `report_due`（100 步）；保留每步 `isfinite(loss)` | GPU 单步 5–20%；CPU 实测 +4.8% 归零 | 在 5090 上跑 200 步 with/without（本机 CUDA 不可用） | NaN 检测延迟 ≤100 步；补合同测试 |
| R5 (C5) | 用单个 pinned batch buffer（gather 进 pinned）或在 `to_device` 判断源矩阵 pinned | 省 423 MB/进程页锁定与一次全量拷贝；恢复异步 H2D | 统计 `non_blocking` 命中率；`--device cuda` 下单步时间 | 多一次拷贝（与现状等价） |
| R6 (C6) | 把 `configs/cloud_training_resources.json` 入库；runner `--require-floor` 缺文件即失败；其余 5 个入口接入 | floor 从"1 个入口"到"全部入口" | 干净克隆跑 `tests/train/test_source_execution_resources.py` | 无（显式失败） |
| R7 (C7) | `available_cpu_threads()` 加 Windows Job Object 查询（`GetCurrentProcess`→`QueryInformationJobObject`）；`run_source_training_chain.py:605` 与 `run_candidate_promotion_matrix.py:1464,3214` 改为传 `total_threads=cpu_budget` | 防止 32 worker / 64 线程越界；CPU 预算真实生效 | 打印 `resolve_cpu_worker_plan().record()` | 需 Windows API，Linux 路径不变 |
| R8 (C8) | E2 适配前显式 `configure_shared_worker_budget(workers)` + `configure_resident_memory_budget`；把实际 `workers` 写进任务记录 | 4 并发显存从"各自独占"变"按份"，超订消除 | 断言 `shared_worker_count == workers` 的测试 | 无 |
| R9 (C9) | attested 加载改 `os.dup(fd)` + `np.load`，去掉复制与 105% 门槛 | 省 1 次全量写 + 空间门槛；0.075 s/157 MB 归零 | 复用 `_file_identity` 前后比对；跑加载合同测试 | zipfile 需可 seek；`fstat` 前后不等仍应失败 |
| R10 (4.1) | 合并缓存改 `compressed=False`（float32）并允许 `mmap_mode='r'` 切片 | 加载 0.632 s→~0.05 s（页缓存热），去掉 247 MB 实体化 | 对比 `load_epoch_dataset` 耗时与 RSS | 磁盘 +1.6×；格式变更需新缓存身份 |
| R11 (5.5) | benchmark 加 `--baseline/--max-regression`，并补 `batch<rows`、多 worker、64 通道三个场景 | 性能回退可被 CI 发现；覆盖真实 IO | 两次运行对比 | 合成数据不代表真实 raw |

## 8. 复核方法（只读、可重现）

```powershell
# 存储清单
Get-ChildItem experiments/runs -Recurse -File | Group-Object Extension |
  ForEach-Object { $s=($_.Group|Measure-Object Length -Sum).Sum; '{0} {1} {2:N2}GB' -f $_.Name,$_.Count,($s/1GB) }
# npz 压缩/加载/哈希实测
.venv\Scripts\python.exe tmp/audit/scratch/npz_probe.py
# 单步扫描开销
.venv\Scripts\python.exe tmp/audit/scratch/step_probe.py
# 资源解析（纯函数，不训练）
.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'src');from research.source_execution import resolve_source_resources;import json;print(json.dumps(resolve_source_resources(json.load(open('configs/cloud_training_resources.json')),8,'cuda',cpu_count=32,free_device_bytes=30*2**30,free_host_bytes=120*2**30),indent=1,default=str))"
# 资源合同测试
.venv\Scripts\python.exe -m pytest tests/train/test_source_execution_resources.py tests/train/test_runtime.py -q
```

### 8.1 与 Lead 给定事实的核对（复核者可直接对照）

- `runtime.py` 1248 行：**一致**（读到最后一行 1248）。
- `available_cpu_threads`/`resolve_cpu_worker_plan`/`threads_per_worker`/`cpu_thread_budget`：**一致**（`:75-208`）。
- `choose_batch_size` 公式 `free*0.55/workers` 与 `prod(shape)*4*16`、`params*5`：**一致**（`:1215-1227`），但本模型 1279 参数使 `params*5=26 KB` 可忽略，启发式永不 binding。
- `recommended_concurrent_workers` <24 GB 返回 1：**一致**（`:1167-1170`）。
- `pin_memory` 单缓冲：**部分不符**——实现是"整矩阵一次性 pin"（`:508-517`），且随机批路径丢失 pinned 性（见 C5）。
- `can_preload` 用 `preload_fraction/workers`：**一致**（`:1184-1185`），但存在 `resident_memory_budget` 分支优先（`:1180-1183`），4 worker 实际走后者。
- `source_execution.py` 119-221 行解析/floor：**一致**（实际 107-227）。
- `configs/cloud_training_resources.json` 数值：**一致**，但**未入库**且无代码读取（C6）。
- `experiments/runs` 41 GB/40956 文件/122 个 >20 MB：**一致**（40.98 GB / 40 956 / 122）；`.gz` 26.93 GB（Lead 记 27.6 GB，实测 26.93 GB，差 0.67 GB 可能因统计时点）。
- `epochs.py:1194-1255` 全量复制 + SHA + 105% 空闲：**一致**，但时间成本实测仅 ~0.5 s/GB（见 §4.2）。


## 9. 逐项核对清单（对照任务书"重点"）

| 重点项 | 结论 | 证据 |
|--------|------|------|
| worker 计划透传到实际 worker | 是（预算/组数/原生线程/子 worker 四层都落地）；2 处静默降级 | `source_execution.py:168,172,214,219`；`run_source_training_chain.py:462,494,501,505` |
| 嵌套进程是否双重计数 | 不是双重计数，而是**低计**：group 记 1 CPU 实占 6 线程，子 worker 记 1 CPU 实占 1 线程 → 记账 7 vs 实际 28 | `source_execution.py:127,143-149` |
| 32 GB 卡 4 GPU worker 显存可达性 | 可达（27 GB 常驻 + <1 GB preload + 3 GB context 余量）；batch 启发式不 binding | §3.2；`runtime.py:1222`；模型 1279 参数 |
| 失败/不足时 fail-closed 还是静默单 worker | 内存/floor **fail-closed**；`target_workers` **静默降到 1**且不写入链级报告 | `source_execution.py:146,188-189,206-213` |
| Windows 无 affinity/cgroup 是否高估 CPU | 会（返回 32，不读 Job Object CPU rate），且被 3 处直接当上限 | `runtime.py:75-90`；`run_source_training_chain.py:605`；`run_candidate_promotion_matrix.py:1464,3214` |
| `cloud_resource_floor` 是否覆盖所有入口 | 只覆盖 1 个入口；配置文件未入库 | §3.4 |
| epoch 缓存加载（npz/pt） | 0.632 s/247 MB（90 946 行/s）；压缩比 1.08–2.28；无 mmap | §4.1 |
| float64→float32 时机 | loader 已转 float32；每次 fit 仍有 3 个全矩阵 pass | `epochs.py:969,1461`；`deep.py:668-675` |
| 逐样本 Python 循环 | MatrixBatchSource 无逐样本循环；gather 每批做 unique+越界检查 | `runtime.py:698-725,940-951` |
| MatrixBatchSource 预载 / gather_indices 拷贝 | 预载走 `index_select` 新分配；快速整轮 shuffle 默认关闭 | `runtime.py:475-517,798-859`；`deep.py:259,1976-1982` |
| DataLoader / pin_memory | **无 DataLoader**；整矩阵 pin 且随机批丢失 pinned → 异步 H2D 从未生效 | `runtime.py:508-517,904-918,1237-1239` |
| 缓存键与失效 | 字节 SHA + 全字段记录，语义正确 | `epochs.py:1258-1300` |
| attested O(size) 放大 / 105% 门槛 | 时间 ~5.2 s/GB（可接受）；空间门槛与每进程副本是主要代价 | `epochs.py:1073-1099,1240-1255`；§4.2 |
| AMP / BF16 | 已实现，BF16 优先，TF32 + cudnn.benchmark 已开 | `runtime.py:369-405`；`device.py:32-45` |
| compile / fused Adam | 均已接线；`fullgraph=False` 会 graph break；无实测门禁 | `deep.py:1282-1287,1325-1326` |
| checkpoint I/O | 4028 个 / 2.29 GB，均值 0.57 MB，写成本可忽略；resume 读两遍 | `source_resume.py:64-71`；`checkpoint.py:56` |
| 评估向量化 | 无逐样本循环；每 cell 固定合同/摘要开销 ≈1 s | `run_source_training_chain.py:274-330`；`doc/engineering.zh.md:396-398` |
| 无谓 device sync | 同步策略节制（100 步/epoch 边界），但每步梯度扫描是多 kernel 开销 | `deep.py:1354-1379,1318-1321` |
| batch 选择启发式 | 形式正确、实际不 binding（模型 5 KB）；16× 保守系数无收益 | `runtime.py:1215-1227` |
| benchmark 冷热分离 | 正确（cold=samples[0]，warm=median(其余)）；数值 oracle 校验到位 | `benchmark_unified_training.py:64-76,52-62` |
| benchmark 回归门禁 | **无**：必须新输出路径、无基线比较、CI 不调用 | `benchmark_unified_training.py:150-151` |
| 41 GB runs 小文件 | 40 956 文件 / 4 398 目录 / 122 个 >20 MB | §6 |
| 结果写放大（1.56 GB JSON） | 实测 1.53 GB；最大单文件 171.9 MB；O(n²) 重写 | §6、C2 |
| resume 扫描成本 | 2× checkpoint 字节（≈4.6 GB 顺序读） | `source_resume.py:64-71` |
| 原始 tar.gz 存在 runs 下 | 是：27 GB / 96 个 `.gz`，另有 2.83 GB `shards.tar` 与解包目录双份 | §6 |

## 10. 关键代码摘录（便于复核）

```python
# src/baselines/deep.py:1318-1322  每步对全部参数做有限性扫描
finite = torch.isfinite(loss.detach())
for parameter in self.model_.parameters():      # 22 个参数张量
    if parameter.grad is not None:
        finite = finite & torch.isfinite(parameter.grad).all()   # 2 kernel/参数
optimizer.step()

# src/research/source_execution.py:78-85  每任务结束摧毁编译与分配器缓存
def release_source_worker(device):
    if torch.device(device).type == "cuda":
        torch.compiler.reset()                   # 编译缓存失效
        release_device_memory(torch.device(device))   # gc.collect + empty_cache + ipc_collect

# src/research/source_execution.py:142-149  余量不足时静默降为 1 个 worker
desired = min(self.target_limit, max(requested, self.cpu_budget // max(1, self.state.groups)))
allowed = min(desired, self.state.free // self.target_bytes, cpu_free, task_free)
workers = int(allowed) if allowed >= 2 else 1

# experiments/run_source_training_chain.py:405-406  每次评分重建池并 pickle 整份数据集
with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                         initializer=_initialize_e2_worker, initargs=(dataset,)) as pool:

# src/train/runtime.py:508-517 整矩阵 pin；随机批 index_select 后 pinned 性丢失
if not self.preloaded and is_accelerator(runtime.device):
    self.cpu_X = runtime.pin_memory(self.cpu_X)      # 全量拷贝到页锁定内存
...
# src/train/runtime.py:903-918
cpu_rows = permutation[start:stop].cpu()
selected_x = self.cpu_X.index_select(0, cpu_rows)    # 结果不再是 pinned
xb = self.runtime.to_device(selected_x)              # non_blocking=False
```

## 11. 未决/需在云端实测的项

1. C3 的 GPU 占比：本机 `torch 2.13.0+cpu`，需在 5090 上以 200 步 with/without 复测。
2. `torch.compile(mode="reduce-overhead")` 在 `fullgraph=False` + Python 分支下的真实收益：需 `TORCH_LOGS=graph_breaks` 统计 break 次数。
3. 4 worker 并发下的 GPU 利用率与实际显存曲线：需 `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1` 与 `worker_progress.json` 时间戳对齐。
4. 合并后的 4 研究 source 缓存真实大小（本机只有 `local_prepared` 分片与 domain_transfer 的 423 MB 单份）：决定 R10 的 mmmap 收益是否成立。

