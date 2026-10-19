# 审计分区 05：src/research、src/baselines、experiments 顶层脚本

> 审计者：子代理 05 · 时间：2026-10-19 · 基准：当前工作树（非 HEAD，80 modified / 118 deleted / 230 untracked）
> 范围：src/research/（12 py）、src/baselines/（12 py）、experiments/*.py（69 个顶层）
> 方法：grep 定位 + 精读；ruff；限定 pytest；--help 抽样；git ls-files 判定入库状态；未运行训练、未遍历 42GB 结果。

## 结论摘要

1. **P0 仓库可复现性阻断**：追踪代码在导入期依赖未入库模块。src/transfer/target_experiments.py:291/421/486、src/transfer/binary.py:19、src/stats/transfer.py:13 引用 research.comparison 与 research.source_matrix，而 comparison.py、source_catalog.py、source_matrix.py 三个模块均为 untracked（git ls-files 退出码 1）。全新 clone 或仅 HEAD 环境会 ImportError。
2. **P1 长任务无法就地续跑**：run_source_training_chain.py:649-650 与 :477-478 拒绝非空输出目录，run_domain_transfer_matrix.py:76 使用 exist_ok=False。中断后只能换目录重跑；SourceChainResume 需要额外的冻结 inventory 且指向旧记录。对照 run_source_pool_matrix.py:175-177 有基于 plan 同等性校验的续跑。
3. **P1 执行身份校验不一致**：run_pretrain.py:156-159、run_candidate_promotion_matrix.py:2500-2505、run_within_subject_transfer.py:302-306、run_binary_subject_transfer.py:140-141 只校验归档清单哈希；research/evaluation.py:79-83 明确说明该函数“不比较归档与正在执行的 checkout”。只有部分入口调用 verified_executed_snapshot（run_source_training_chain.py:646-648、run_source_pool_matrix.py:154、run_cross_dataset_transfer.py:254）。
4. **P1 结果覆盖/非原子写**：run_eeg_loso.py:582 以 "w" 截断 progress.jsonl，:736-741 直接替换 record.json；run_pretrain.py:441-442,491 直接 torch.save 到目标 checkpoint，无 --force 或原子替换。
5. **P2 资源准入含死代码**：research/source_execution.py:139-155 的 HostMemoryAdmission.target_workers 无任何调用者（grep 确认）；且 requested="auto" 时 max(str,int) 会 TypeError；workers==1 时不扣 target_bytes；task_free 与 cpu_free 对 source groups 重复计数。
6. **P2 锁/续跑校验边界**：process_file_lock 无超时与等待日志（source_execution.py:24-30）；SourceChainResume.verify 仅按 (size,mtime_ns) 缓存哈希（source_resume.py:64-71）；acquire 取 refit_key 的首个匹配（75-76），依赖字典顺序。
7. **P2 评估静默降级**：evaluate.py:611-613 将单类折记 NaN，:1364-1369 用 nanmean 后摘要不记录缺失折数；paired_permutation_test:1591-1600 不拒绝 NaN（NaN 差值得 p=1）。
8. **P2 基线加载安全**：baselines/deep.py:587 使用 torch.load(..., weights_only=False)；同仓库 transfer/checkpoint.py:56、transfer/result_reuse.py:116 为相同模式。
9. **P2 实验工程重复**：69 个脚本存在 7 种 sys.path 引导变体；atomic JSON 至少 3 份实现（research/execution.py:42、run_candidate_promotion_matrix.py:2050、run_pretrain_supervised.py:82）；run_domain_transfer_matrix.py:44 用 argv.index("--subjects") 解析私有 CLI 并复制分片规则。
10. **P2 数据治理量化**：experiments/runs 40.98GB/40,956 文件；.gz 26.93GB/96（source_eog_20260909/delivery 中 BI2013a 4.61GB、Huebner2017 4.59、BI2015a 4.16、EPFLP300 2.30、BI2014a 2.07、GTN 1.79、Chailloux2020 1.22），.npz 5.43GB、.tar 4.31GB、.pt 2.29GB。8,134 个 runs 内 py 中 5,951 个来自 22 个 checkout 快照（合计仅 0.11GB），2,183 个是逐 run 复制的分析脚本。42GB 主因是数据/结果归档复制，不是源码快照。

## 发现

### A. 仓库状态与执行身份

**A-1（P0）追踪代码导入未追踪模块，干净检出不可运行**
- 证据：git ls-files 判定 src/research/comparison.py、source_catalog.py、source_matrix.py 均未入库（退出码 1）；而 src/transfer/target_experiments.py:291、:421、:486，src/transfer/binary.py:19，src/stats/transfer.py:13 在模块顶层 import research.comparison / research.source_matrix。追踪的 experiments/run_source_training_chain.py:560/679/974/1044/1057 同样导入它们。
- 影响：当前工作树可运行，但任何 clone、CI、归档或协作方仅按 Git 还原会立即 ImportError；checkpoint 的 source_snapshot 声明无法在纯净环境复现。这是全仓库级可复现性阻断，不只是分区问题。
- 建议：把三个模块与对应 tests/research/test_source_catalog.py、test_source_pool_matrix.py、test_comparison_identity.py 一并入库；在 pre-commit 或 CI 增加“追踪模块不得导入未追踪模块”的静态检查（如 importlib 扫描 + git ls-files）。
- 验证方式：git ls-files --error-unmatch 逐个文件；在临时 worktree 中运行 python -c "import sys; sys.path.insert(0,'src'); import stats.transfer"。

**A-2（P1）训练入口的执行身份（executed snapshot）校验不一致**
- 证据：run_pretrain.py:156-159 调用 source_snapshot_sha256_from_archive_manifest；run_candidate_promotion_matrix.py:2500-2505 同样只校验归档；run_within_subject_transfer.py:302-306、run_binary_subject_transfer.py:140-141 使用 verified_source_snapshot。research/evaluation.py:79-83 注释明确“不比较归档与正在运行的 checkout”。对比 run_source_training_chain.py:646-648、run_source_pool_matrix.py:154、run_cross_dataset_transfer.py:254 调用 verified_executed_snapshot。
- 影响：checkpoint 可声明快照 A，而实际由改动后的代码 B 训练；source_snapshot_sha256 变成弱声明。不同入口的安全强度不一致，审计时无法用同一规则判断。
- 建议：提供单一 admit_executed_snapshot() 供所有训练/分析入口使用；默认强制 executed 校验，仅允许显式 --allow-dirty-source 并在记录中写明 base_commit + git_status。
- 验证方式：grep verified_executed_snapshot 的调用分布；对同一 manifest 修改一个 src 文件后重跑入口，观察是否拒绝。

**A-3（P2）verified_executed_snapshot 的严格性带来并发编辑成本**
- 证据：research/evaluation.py:167-169 要求 recorded 覆盖当前 src/**/*.py 与 experiments/*.py 的全部文件，并逐个比较哈希（173-178），随后再解压归档逐成员复核（179-184）。
- 影响：任何并发编辑（本仓库常态：80 modified）都会导致该入口失败；同时每次调用重复计算全部文件与归档哈希，长链启动成本高。
- 建议：冻结清单加缓存（manifest_sha256 + 索引 mtime 不参与判定），或在启动时一次性生成进程内快照校验；把“代码已变更”与“快照损坏”区分成不同错误码。
- 验证方式：对同一 manifest 连续调用两次计时；修改一个 experiments/*.py 观察错误信息。

### B. 调度、续跑与资源

**B-1（P1）主 source chain 与 domain matrix 无就地续跑**
- 证据：run_source_training_chain.py:649-650（if output.exists() and any(output.iterdir()): raise FileExistsError）、:477-478（并行组同样拒绝）；run_domain_transfer_matrix.py:76（output.mkdir(parents=True, exist_ok=False)）。完成组仅保存在内存 report 中，失败即 raise（run_source_training_chain.py:524-527）。
- 影响：数小时至数天的 source fit 在任一组失败后无法原地重试；用户被迫新建目录，shared trajectory registry 也位于旧 output（:504），已完成轨迹不能自动复用（除非另建 resume_source_run 清单）。这是高代价运维风险。
- 建议：引入与 promotion matrix 类似的 journal（已完成组 + 任务完成记录 + 输出哈希），--resume 时校验 plan/dag digest 后跳过 completed；或至少允许空目录之外的 output 并逐组检测完成。
- 验证方式：构造一个含 1 个完成组的旧 output，重跑同一命令观察是否立即 FileExistsError。

**B-2（P2）进程锁无超时、无等待诊断**
- 证据：research/source_execution.py:24-30 在 Windows 上循环 msvcrt.locking + sleep(0.05)，无最大等待；POSIX 分支 :39 直接阻塞 flock。
- 影响：共享 registry 被长 fit 持有时，第二个进程静默挂起，日志无“waiting for lock”记录，难以与死锁/无进度区分。
- 建议：带超时的 acquire，超时抛含 lock path/owner pid 的错误；周期性打印等待原因。
- 验证方式：两个进程同时 fit 同一 signature，观察第二个进程的日志与超时行为。

**B-3（P2）resume 校验缓存与 refit 选择依赖元数据/顺序**
- 证据：research/source_resume.py:64-71 用 (st_size, st_mtime_ns) 作为哈希缓存键；同尺寸且 mtime 被保留的改写会跳过复核。:74-76 用 next(...) 取第一个 refit_key 匹配，多任务同键时结果依赖 dict 顺序。
- 影响：极端情况下续跑接受已变更 artifact；refit 复用可能绑定到非预期任务。
- 建议：始终校验内容哈希（或纳入 st_ino/ctime），并要求 refit_key 在 inventory 内唯一。
- 验证方式：复制文件后保留 mtime 改内容，再走 acquire；构造两个同 refit_key 任务观察选择。

**B-4（P2）资源准入存在未使用且含缺陷的 target_workers**
- 证据：research/source_execution.py:139-155 定义 HostMemoryAdmission.target_workers；grep 在 src/ 内无调用者（仅定义与文档）。其 desired = min(limit, max(requested, ...)) 对 requested="auto" 会 TypeError；workers==1 时 reservation=0，未预留 target_bytes；task_free 与 cpu_free 都减 state.groups，而 target 预留只增加 child_cpus。
- 影响：当前不生效，但一旦接入就会错误准入/超额订阅；资源策略报告中的 target_workers 与实际启动的 worker 数可能脱节。
- 建议：删除死代码或补齐测试后接入；requested 先解析为 int；单 worker 也占用 target_bytes；统一 groups/child_cpus 的计账口径。
- 验证方式：对 target_workers("auto"|1|4) 写单测；grep 调用点。

### C. 实验脚本工程

**C-1（P1）LOSO 入口静默截断/覆盖既有结果**
- 证据：run_eeg_loso.py:579-582 对每个模型 output_dir.mkdir(..., exist_ok=True) 后以 "w" 打开 progress.jsonl；:736-741 写临时文件后直接 replace record.json；无 --force/--resume 或存在性检查。
- 影响：同名 --run-name 重跑会丢失上次的逐折进度与最终 record（且中途崩溃时 progress 已清空）；无法从断点继续，也无法保留失败证据。
- 建议：默认拒绝已存在的 record.json；提供 --resume 读取已完成 fold 并追加 progress；--force 才允许覆盖，且先备份。
- 验证方式：用不存在的 dataset 参数构造 --run-name 后重跑两次，比较 progress.jsonl 行数与 record.json mtime。

**C-2（P1）预训练 checkpoint 非原子写且无覆盖保护**
- 证据：run_pretrain.py:441-442 checkpoint.parent.mkdir(...)，:491 torch.save(payload, checkpoint)；无临时文件/rename，无存在性检查。
- 影响：保存中断会留下不可加载的截断 .pt；重跑静默覆盖旧预训练权重，无法追溯哪次代码/参数产生。
- 建议：写 .tmp + os.replace；默认拒绝覆盖；同时输出 checkpoint 旁车 record.json（含 manifest、seed、git_status、哈希）。
- 验证方式：在保存中途 kill，尝试 load_checkpoint_payload；重跑同命令观察覆盖。

**C-3（P2）domain matrix 以 argv 文本解析实验计划并复制分片规则**
- 证据：run_domain_transfer_matrix.py:43-47 从 entry["argv"] 中 index("--subjects")+1 取值，再按 all_subjects[shard::workers] 重新分片；注释自认“source runner declares subjects in the same strided shard order”。
- 影响：若上游改用 --subjects=a,b、改变 shard 规则或多值参数，目标分会静默错配或 ValueError；这是跨脚本隐式契约，无 schema 校验。
- 建议：在 outer_evaluation_jobs 中直接记录每个 shard 的 subjects 列表与 shard 总数；runner 只消费显式数据。
- 验证方式：构造 --subjects=a,b 形式的 plan，运行 target_job_groups 观察异常/错配。

**C-4（P2）source pool 编排异常只保留 repr，且进度文件高频重写**
- 证据：run_source_pool_matrix.py:245-250 except Exception as error 后仅记录 repr(error) 并 print；:253-255 每轮（含 10s 等待轮）重写 progress.json。
- 影响：丢栈导致无法定位子进程失败原因；长跑期间每 10 秒覆盖同一 JSON 造成不必要 IO 与 dashboard 抖动。
- 建议：记录 traceback（或子进程 log 路径）；仅状态变化时写进度。
- 验证方式：注入一个必然失败的 task，检查 progress.json 是否含 traceback；统计 60s 内 progress.json mtime 变化次数。

**C-5（P2）入口引导与原子 JSON 重复实现**
- 证据：69 个顶层脚本中有 7 种 ROOT/sys.path 引导变体（41 个 parent.parent、8 个 parents[1] 等）；atomic JSON 至少三份：research/execution.py:42 atomic_write_json、run_candidate_promotion_matrix.py:2050 _atomic_write_json、run_pretrain_supervised.py:82 _write_json_atomic；experiments 无 __init__.py，却以 from experiments... 跨脚本导入（run_domain_transfer_matrix.py:19、run_source_pool_matrix.py:36）。
- 影响：路径/编码/重试语义漂移；命名空间包行为依赖 ROOT 插入 sys.path，python -m 或安装式运行可能失败。
- 建议：新增 experiments/_bootstrap.py（ROOT、UTF-8 JSON、atomic_write、run dir 校验）并统一导入；把 experiments 变成显式包或改用 src 下的库函数。
- 验证方式：统计统一后的重复行数；在未插入 sys.path 的解释器中 import experiments.run_eeg_loso。

**C-6（P2）dashboard 终止端点仅静态头，端口探测失败静默**
- 证据：dashboard_server.py:540-542 仅检查 X-Dashboard-Confirm: yes，:579-590 默认 127.0.0.1 但允许 --bind 任意地址；run_with_dashboard.py:32-40 在 8812-8831 全忙时返回 start，:91 sleep(1.0) 后不检查 server.poll()。
- 影响：绑定非 loopback 时任何能访问该端口者可终止正在运行的实验；端口全忙时用户看到 URL 但服务未启动，实验照跑无监控。
- 建议：每次启动生成随机 capability token 写入 active_run.txt 并校验；非 loopback 绑定强制 token；启动后 connect 探测并失败即报错。
- 验证方式：占用全部端口后运行 wrapper；用伪造 Host 头 POST /api/kill 观察鉴权。

### D. 基线与评估

**D-1（P2）checkpoint 反序列化使用 weights_only=False**
- 证据：src/baselines/deep.py:587 torch.load(Path(path), map_location="cpu", weights_only=False)；同仓库 transfer/checkpoint.py:56 与 transfer/result_reuse.py:116 相同。
- 影响：加载共享/下载的 .pt 可执行任意 pickle 代码；本仓库 checkpoint 常来自 runs 目录与云端结果包，边界不清。
- 建议：state_dict 类负载改 weights_only=True；确需完整 payload 时先校验 SHA-256 白名单并在隔离进程加载。
- 验证方式：构造含 __reduce__ 副作用的 pickle（仅审计脚本，不执行）验证 weights_only=True 会拒绝。

**D-2（P2）评估摘要对 NaN 静默平均，置换检验不拒绝 NaN**
- 证据：evaluate.py:609-613 单类测试折 bacc/auc=nan；:1364-1369 用 np.nanmean/np.nanstd 汇总且 BinarySummary 无缺失折数字段；:1591-1600 delta 含 NaN 时 observed=NaN，比较全 False，p=(0+1)/(n+1) 而非报错。
- 影响：单类折被无声丢弃，均值看似合法但样本量/覆盖不明；统计检验对缺失输入给出误导性 p 值。
- 建议：BinarySummary 记录 n_valid/n_single_class 并拒绝在关键比较中静默丢折；permutation 先校验 np.isfinite。
- 验证方式：构造单类折与含 NaN 的 scores 调用 evaluate_binary/paired_permutation_test。

**D-3（P2）逐折 deepcopy 与未文档化的模型落盘 env**
- 证据：evaluate.py:541 每个 fold 对 prototype 做 copy.deepcopy（进程 worker 中对象已由 pickle 传入，仍再复制一次）；:617-649 读取 P300_SAVE_FOLD_MODELS_DIR，把每折 state_dict、输入统计、测试受试者与阈值写入 fold_XXXX.pt。
- 影响：GPU 模型深拷贝增加启动开销与显存峰值；环境变量可在不改变命令行/记录的情况下把模型与受试者信息写到任意目录，结果不可复现且存在隐私面。
- 建议：进程路径跳过 deepcopy（用 pickle 已隔离的副本），线程路径才复制；把模型导出改为显式 CLI/配置并写入 record。
- 验证方式：对比 process/thread 两种 backend 的 fit 启动时间；unset env 后确认无文件写出。

### E. 目录与数据治理

**E-1（P2）experiments 膨胀的成因与量化**
- 证据：Get-ChildItem 统计 experiments/runs = 40.98GB/40,956 文件；.gz 96 个 26.93GB，最大为 source_eog_20260909/delivery/BI2013a.tar.gz 4.61GB、Huebner2017 4.59GB、BI2015a 4.16GB、EPFLP300 2.30GB、BI2014a 2.07GB、GTN 1.79GB，以及 export/source_eog_20260909_results.tar.gz 1.86GB、domain_transfer_cloud_20260916/export/results.tar.gz 1.55GB、unified_cloud_20260916/export/verified_results.tar.gz 1.05GB。runs 内 py 8,134：22 个 checkout 目录含 5,951 py/6,681 文件（0.11GB），其余 2,183 为逐 run 复制的 analyze/prepare 脚本。checkout 可由 freeze_source_snapshot.py:56-60 的 --materialize 生成（该函数对同一归档调用两次 verified_executed_snapshot，:55、:60）。另统计 40 个 source.source-only.tar.gz，其中 13 组各有 2 份完全相同的哈希（runtime 目录与 export/execution_snapshots 各一份）。
- 影响：42GB 并非“每次 run 复制源码”，而是交付数据集包与结果包被复制进 gitignored runs；源码快照只占 0.11GB。真正的成本是同一数据包的多份拷贝、无内容寻址与无保留策略。
- 建议：把 delivery/*.tar.gz 移到 runs 外的内容寻址 store（sha256 命名）并以 manifest 引用；runs 内只留 manifest/checksum；对 export 结果包做去重（已发现 source.source-only.tar.gz 1.2MB 在同 run 的 runtime 与 export/execution_snapshots 各一份）；增加 retention 清单（保留最近 N 次或按 evidence 级别）。
- 验证方式：Get-FileHash 去重统计；按 run 输出 GB/evidence 级别报表；抽样校验 store 引用可解析。

**E-2（P2）runs 内脚本不可导入且与顶层脚本漂移**
- 证据：runs/*/ 下的 *.py 是执行时复制/生成的快照（如 bi64_protocol_factorial_20260906/analyze_study.py），无 __init__.py，依赖当时的 checkout；顶层 19/69 脚本 untracked。
- 影响：无法判断某个 run 的结果由哪份脚本产生（除 manifest 外），复制脚本修改后容易出现“同名不同逻辑”。
- 建议：每个 run 增加 execution.json（脚本相对路径 + sha256 + Python/依赖版本 + 命令行），并禁止在 runs 内修改脚本；分析阶段只读。
- 验证方式：随机抽 5 个 run，检查是否有 manifest 记录脚本哈希；与顶层同名脚本比对哈希。

**正面观察（非缺陷）**
- src/baselines/calibration.py:97-117、120-206 明确只在非测试观测上拟合阈值/温度，且校准数据缺失时拒绝重代入（209-224），设计边界清晰。
- src/baselines/structured_erp.py:156-217 对导出状态做键集合、形状、正定性与几何一致性校验，是良好的 artifact 契约示例。
- run_candidate_promotion_matrix.py:2080-2128 的 orchestrator 锁不自动偷取，:2795-2857 的 resume 逐项校验输出/日志/依赖哈希，是全仓库最好的续跑实现，可作为其他入口模板。

## 未能验证与存疑

- 未运行任何训练、GPU 或数据准备；所有科学结论（AUC、校准、泄漏）仅做代码路径审查，未复算数值。
- 未逐行审阅全部 69 个顶层脚本（精读/定点约 20 个，其余仅 grep 与函数清单）；deep.py（2,769+ 行）只定点阅读约 400 行，其完整训练环路未验证。
- 未遍历 42GB 结果内容；大小/去重为文件级统计，未确认 .gz 与 mne_data/frozen 中的数据是否跨分区重复（属其他分区）。
- 未验证 doc/ 六份正文与代码的一致性，仅核对模块 docstring。
- --help 抽样时退出码为 -1，怀疑是 PowerShell 管道提前关闭导致的 broken pipe，未据此判定入口失败；usage 正常输出。
- SourceChainResume 仅经单测（42 项通过）验证，未在真实旧 run 上执行 resume；其“非空目录拒绝”与 resume 的组合行为属推断。
- 118 个删除测试与 tests/ 新子目录的映射完整性未核（属测试分区）。
- 未读取或输出任何密钥值；未检查 .env/凭据文件。

## 已检查范围

- 全文精读：src/research/execution.py、source_resume.py、source_catalog.py、source_matrix.py、source_execution.py、evaluation.py、comparison.py；src/baselines/calibration.py、structured_erp.py；experiments/run_source_pool_matrix.py、run_domain_transfer_matrix.py、freeze_source_snapshot.py、result_batch_contract.py、run_with_dashboard.py、run_pretrain.py（除少量中段）。
- 定点阅读：src/research/source_selection.py:1-521；src/baselines/deep.py:92-200、239-359、374-474、560-640、1120-1190、1390-1440、2380-2440；src/baselines/evaluate.py:281-401、527-707、703-823、1140-1290、1286-1386、1586-1676；experiments/run_source_training_chain.py:27-205、423-547、547-845、1311-1396；experiments/run_candidate_promotion_matrix.py:2073-2130、2487-2636、2795-2857、3051-3194、3532-3591；experiments/run_eeg_loso.py:149-290、530-610、700-754；experiments/run_cross_dataset_transfer.py:149-300；experiments/prepare_eeg_dataset.py:200-258；experiments/dashboard_server.py:1-160、316-360、475-612。
- 仓库事实：git status --porcelain（分区内 28 项 modified/untracked）、git ls-files（research 9/12、baselines 10/12、顶层 experiments 50/69 入库）；tests 目录 118 项删除。
- 量化：experiments 顶层 69 py/25,568 行；runs 40.98GB/40,956 文件；py 8,134（checkout 5,951）；.gz/.tar/.npz/.pt 分类统计；SHA-256 重复抽样。
- 质量门：ruff check src/research src/baselines experiments → All checks passed（配置仅 E,F,I,N,W,UP,B，E501 忽略）；pytest tests/train/test_source_resume.py、test_source_execution_resources.py、tests/research/test_source_pool_matrix.py、test_comparison_identity.py、test_research_execution.py → 42 passed。
- 未修改任何已有文件；仅创建/更新 tmp/audit/05-research-experiments.md 与 tmp/audit/scratch/dup_experiments.py。
