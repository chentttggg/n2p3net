# 全仓库审计与改进建议

- 日期：2026-10-19（Asia/Shanghai）
- 审计对象：D:\BCI\n2p3-net 当前工作树（非 HEAD；HEAD 为 f1a303c）
- 约束：只读审计，**未修改任何源文件/测试/配置/指南**。本报告为新增文件；审计过程的 13 份分区报告保存在 `doc/assets/repository_audit_20261019/`（原 `tmp/audit/` 副本）。
- 方法：13 个只读审计分区（8 个 Lead 直管子代理 + 5 个后续分区）+ Lead 横向核查；全量 pytest、ruff、uv lock 检查、git/磁盘/文档/依赖静态分析；对数学与信号处理结论做小规模数值复算。
- 证据约定：每个结论给 `文件:行号`；标注“已证实/疑似/未验证”。分区报告给出了更完整的逐条证据与复现脚本。

## 0. 规模与运行状态（本次实测）

| 指标 | 数值 | 说明 |
|---|---|---|
| 工作树变更 | 428–429 项：80 modified、118 deleted、230–231 untracked | `git status --porcelain`；包含 109 个 tests 删除、9 个 frozen manifest 删除 |
| 源码 | `src` 120 py / 53.8k LOC；tests 159 个 `test_*.py` / 1593 个 `def test_` | 收集用例 2342 个 |
| 全量 pytest | **3 failed, 2331 passed, 8 skipped, 22 warnings, 439.48s** | 失败均在 `tests/transfer/test_preprocessing_execution_gate.py` |
| ruff | `ruff check src experiments tests` → All checks passed（399 文件） | `uv lock --check` 通过 |
| 工作区磁盘 | **154.9 GB / 237k 文件**（不含 94 GB 外部 junction） | mne_data 89.6 GB、experiments 45.0 GB、.git 8.0 GB、venv 4.15 GB、frozen 3.32 GB、formal 2.5 GB、cloud_inputs 2.0 GB、Paper 0.19 GB |
| .git | 8.04 GB：pack 205 MB + loose objects **7.26 GiB** + 2144 个 unreachable 对象 | 最大 loose 对象 1.2 GB、894 MB、453 MB、451 MB、354 MB |
| 依赖环境 | Python 3.14.3；`.venv` torch 2.13.0+cpu（0.98 GB）；`.venv-gpu` torch 2.13.0+cu130（3.17 GB，CUDA 可用） | `requires-python >=3.11`，ruff target py311；无 CI |

## 1. 总体结论

这是一个**合同与可复现性意识很强、但当前工作树处于“未收敛的中间态”**的研究仓库：数据准入、缓存 attestation、结果复用、预处理回执、Lean 证明链等设计明显高于一般研究代码；同时，**版本控制、测试收敛、文档/证据索引、统计诊断与工作区存储治理**是当前最薄弱的五个环节。

最需要优先处理的不是某个算法细节，而是：**当前代码状态无法由 Git 复现、全量测试不绿、文档链接与索引已出现系统性断链、以及若干会被当作科学结论引用的统计/数学诊断量不可信**。

### 优先级总表

| 级别 | 数量 | 主题 |
|---|---|---|
| P0 | 3 | ① 工作树未提交导致当前能力不可由 Git 重建；② 全量测试 3 失败 + 测试重组不可追溯；③ 当前工作树依赖未跟踪运行模块，仅备份 tracked 文件会 ImportError（与 ① 同源） |
| P1 | 24 | 工程/可复现性 6、数据/预处理 5、迁移/模型 6、数学统计 4、项目文件管理 2、安全 1 |
| P2 | 30+ | 代码结构、测试配置、文档维护、性能、Lean 工程化、许可与命名 |

## 2. P0 — 阻断级问题

### P0-1 当前工作树不可由 Git 重建，且 HEAD 与当前能力脱节（已证实）

**证据**
- 23 个 `src` 模块未跟踪，其中多个是当前代码的运行时依赖：`src/research/comparison.py`、`source_matrix.py`、`source_catalog.py` 被 `src/transfer/target_experiments.py:291,421,486`、`src/transfer/binary.py:19`、`src/stats/transfer.py:13` 直接 import；HEAD 版本的这些文件**不** import 它们，说明依赖只存在于工作树。
- 其余未跟踪源码含核心模块：`src/data/source_records.py`（61 KB）、`ea_artifact.py`（34 KB）、`source_parsers.py`、`source_admission.py`、`bidirectional.py`、`finite_filter_operator.py`、`harmonic.py`；`src/transfer/exact_covariance.py`、`information_bound.py`、`noise_model.py`、`operator_lda.py`、`temporal_noise.py`、`continuous_decision.py`、`raw_continuous.py`、`source_objective.py`、`decision_experiments.py`；`src/baselines/structured_erp.py`、`temporal_transfer.py`；`src/research/comparison.py` 等。
- 未跟踪的还有：19/69 个 `experiments/` 顶层脚本；4/9 个 `configs/`（含关键资源合同 `cloud_training_resources.json`、`unified_training_budget.json`）；整个 `formal/`（385 个 .lean 源文件、39 个 `Audit*.lean`、39 个 scripts、`verification_plan.json`）。
- 118 个 tracked 文件被删除（其中 `tests/` 109 个），`tests/` 已重组为 8 个子目录但**未暂存**；9 个 `frozen/*.manifest.json` 被删除且未提交。
**影响**：`git clone`/仅备份 tracked 文件无法构建当前系统；`git checkout`、`git stash`、分支切换会丢失当前能力；所有“测试/实验来自哪个源码”的执行身份依赖一个未入库的物理快照。这是全仓库最高风险。
**建议**：先对当前工作树做物理快照（复用 `experiments/freeze_source_snapshot.py`，或 `git stash create`/补丁 + 压缩归档），然后按 6 个原子提交入库：① EOL 归一化（单独提交）；② tests 迁移；③ 新 src 模块；④ configs；⑤ formal/（源码/脚本/manifest，不含 .lake/artifacts）；⑥ doc/assets + evidence + frozen manifest 修复。每个提交跑 ruff + 全量测试。
**验收**：`git status --porcelain` 为空（或只剩明确忽略项）；在临时 clone 中 `python -c "import transfer.target_experiments"` 成功。

### P0-2 全量测试 3 失败；测试重组缺少等价性记录（已证实）

**证据**
- 本次全量：`3 failed, 2331 passed, 8 skipped`（439.48 s），失败全部在未跟踪的新测试文件 `tests/transfer/test_preprocessing_execution_gate.py`：`test_new_personal_model_keeps_binding_on_save_load_and_admits_new_source_context`、`test_historical_personal_model_reloads_without_promoting_execution_evidence`、`test_personal_model_rejects_receipt_substitution_even_when_outer_metadata_is_resealed`。
- 直接原因（已定位）：这三个用例的 `_personal()` fixture 用 `n_channels × n_times` 构造 LDA state（`tests/transfer/test_preprocessing_execution_gate.py:70-72`），而工作树新增校验要求 `selected_count × downsampled_time_points(..., target_hz=20.0)`（`src/transfer/calibrated.py:650-658`）。生产 LDA 拟合路径 `target_shrinkage_features(..., sfreq)` 确实使用 20 Hz 降采样，故 fixture 陈旧；但“历史个人模型重载”用例因此失败，兼容约定需要人工确认。
- HEAD 的 1189 个测试函数中有 130 个唯一名称在当前树中消失；函数体模糊匹配仅 12 个可能是改写/改名，**119 个无近似替代**；消失名单含路径逃逸、哈希不一致、预算下限、forced-switch 账本等安全/合同测试（分区 06）。
**影响**：当前工作树不能宣称“验证通过”；测试重组是否等价无法由仓库自身证明；若直接提交，回归风险不可控。
**建议**：① 修 3 个 fixture（或回退该校验并单列迁移计划），恢复全绿；② 产出 `HEAD tests → 当前 tests` 的迁移映射（kept/merged/retired+reason），对 119 个无替代函数逐条确认，安全/预算类测试优先补回；③ 重组提交与行为变更提交分离，保留 `git log --follow`。
**验收**：`pytest` 全绿；迁移映射表中每个缺失函数有结论；`tests/transfer/test_preprocessing_execution_gate.py` 覆盖新校验的正反例。

### P0-3 备份/迁移边界不成立（P0-1 的直接后果，单列强调）

当前“源码=Git、数据/结果=本地”的分工在纸面上成立，但由于 23 个运行模块、19 个实验脚本、4 个关键 config、整个 `formal/` 不在 Git 中，**按文档执行“Git clone + 单独携带大数据”无法恢复当前系统**。任何机器迁移、云端重装或磁盘故障都会丢失核心能力。
**建议**：在 P0-1 的提交完成前，立即对 `src/`、`tests/`、`experiments/*.py`、`configs/`、`formal/lean`（不含 `.lake`、`artifacts`）、`doc/`、`frozen/*.json` 做一次带 SHA-256 清单的物理归档（参考 `experiments/freeze_source_snapshot.py`），并与 `frozen/worktree_recovery_20261016.json` 的记录方式保持一致。

## 3. P1 — 高优先级问题

### 3.1 工程与可复现性

**P1-E1 行尾未归一化 + 源码自哈希 → 跨平台 resume/证据闸门必失败（已证实）**
- `.gitattributes` 仅 6 行，无 `* text=auto`/`*.py eol=lf`；工作树 137 个 CRLF、69 个 mixed（`git ls-files --eol`）。11 个模块在 import 时对**自身文件字节**做 SHA-256：`src/data/preprocess.py:32`、`src/data/epochs.py:46`、`src/transfer/operator_lda.py:27`、`noise_model.py:19`、`exact_covariance.py:53`、`information_bound.py:79`、`continuous_decision.py:36`、`temporal_noise.py:34`、`src/research/recording_quality.py:239` 等。CRLF 与 LF 的同一文件哈希不同（分区 08 实测）。
- `src/data/source_records.py:98` 的 `data_modules_sha256` 对 `src/data` 下**所有** `*.py` 逐个哈希，`:207` 严格比较；任何无关改动都会使全部源缓存不可 resume；跨平台更直接失败。
- 建议：`.gitattributes` 增加 `* text=auto eol=lf`（至少 `*.py/*.toml/*.json/*.md`），独立提交 `git add --renormalize .`；哈希改用 LF 规范化字节或 `git hash-object`；`data_modules_sha256` 改为显式模块清单/模块级哈希。验收：Linux/WSL 与 Windows 对同一提交打印相同 `_IMPLEMENTATION_SHA256`。

**P1-E2 打包污染：9 个通用顶层包 + wheel 不自包含（已证实）**
- `pyproject.toml:25-26` 仅 `packages.find where=["src"]`，实际发布 `audit_p300, baselines, data, measurement, models, research, stats, train, transfer` 9 个顶层名（`src/n2p3_net.egg-info/top_level.txt`）。`data`/`models`/`transfer` 等是极通用名，装进共享 venv 会静默冲突；根目录若出现 `data/` 还可能遮蔽导入。
- `src/data/source_records.py:99` 读取 `experiments/prepare_source_records.py`，wheel 安装后该路径不存在 → `FileNotFoundError`。
- 建议：迁移到 `src/n2p3_net/...` 命名空间，或至少显式 `include=["n2p3_net*"]` 白名单；把实验脚本哈希改为可注入/随包清单；补 `py.typed`、LICENSE、readme、scripts。

**P1-E3 环境不可复现：lock 多解 + 手工 CUDA 覆盖（已证实）**
- `pyproject.toml:15-17`：Linux x86_64 且 Python<3.14 → `torch==2.8.0`，其他平台/版本 → `torch>=2.4`；`uv.lock` 同时锁 torch 2.8.0(cu12) 与 2.13.0(cu13)。本地 `.venv`=2.13.0+cpu、`.venv-gpu`=2.13.0+cu130 是文档化手工覆盖，lock 无 cu130 index；无 `[tool.uv] required-version`（uv 0.11.26 创建 venv、当前 0.12.9）；`requires-python` 无上界。
- 建议：固定 `uv required-version` 与 Python minor；用 `[tool.uv.sources]/[[tool.uv.index]]` 定义 `gpu` extra 使 cu130 可复现；给 torch marker 加上界并记录“本地/云端不同栈”。

**P1-E4 无 CI、pytest 配置过薄（已证实）**
- 无 `.github/`、无 pre-commit；`pyproject` 仅 `testpaths/pythonpath/addopts`；全树只用 `parametrize`/`skipif` 两个 marker；无 coverage/timeout/strict-markers/warning 策略。`addopts="-q"` 在 pytest 9.1.1 下不打印最终统计行（分区 06 实测）。
- 建议：最小 CI（ruff + pytest + `git diff --check` + lock 检查 + 文档链接检查）按 Python 3.11/3.14 或至少锁定的 3.14；注册 `slow/integration/gpu` markers；加 `pytest-timeout`、`-ra`、`--strict-markers`；覆盖率作为报告而非硬门禁直到基线稳定。

**P1-E5 requirements 漂移与云端引导脚本缺失（已证实）**
- `requirements.txt` 仅 `-e .[signals,baselines,dev]`（重新解析、绕过 lock）；`requirements-minimal.txt` 提及不存在的 `setup_cloud.sh`，且不含 torch；`threadpoolctl` 仅在 minimal 中出现（隐式依赖）。
- 建议：`uv export` 生成 requirements 或删除；补 `scripts/setup_cloud.sh` 入库并纳入文档；显式声明 `threadpoolctl` 或移除。

**P1-E6 关键配置未入库（已证实）**
- `configs/cloud_training_resources.json`（cpu_budget=25、gpu_workers=4、cloud_resource_floor）与 `configs/unified_training_budget.json` 均未跟踪，而 `src/research/source_execution.py:119-221` 直接消费它们。用户明确要求的 25/4 资源下限因此不可审计。
- 建议：随 P0 提交入库；在 CI 中校验 floor 字段与 AGENTS.md 一致。

### 3.2 数据、预处理与输入合同

**P1-D1 attested 缓存加载 O(size) 复制 + 重算哈希 + 105% 空间门槛（已证实）**
- `src/data/epochs.py:1215-1255` 每次 attested 加载都把整份 NPZ 复制到临时文件并重算 SHA；`:1073-1100` 要求 `byte_size*1.05+64MiB` 空闲；4 worker 下读放大与临时空间近翻倍。
- 建议：内容寻址只读对象（按 sha 命名、写一次）；进程内缓存验证结果（inode+size+mtime_ns+sha）；`105%` 可配置；提供显式 fast 模式并记录。

**P1-D2 MOABB 高层入口“声明即证据”（已证实）**
- `src/data/moabb.py:1783-1791` 由 MOABB 内部 `get_filter_pipeline` 执行 `raw.filter(method="iir")`，`:1985-1996` 的 provenance 只是把声明字段抄一遍，没有 `preprocessing_execution` 回执；MNE/MOABB 升级会静默改变算子；`picks="data"` 把 EOG 一并滤波。
- 建议：MOABB 入口也生成 `n2p3_preprocessing_execution/1`（实际 SOS/iir 参数、滤波前后样本、重采样函数、逐通道段）；版本白名单。

**P1-D3 BrainSync/BIDS 会话准入无内容哈希（疑似，需与采集端确认）**
- `src/data/bids_eeg.py:228-330`、`src/data/brainsync_contract.py:359-472` 只做元数据交叉校验；个人校准证据无法绑定到具体文件字节（公共源域则有逐文件 SHA-256）。
- 建议：session manifest 对每个文件记录 `(sha256, byte_size)` 并流式校验，或复用 `raw_artifacts.VerifiedSnapshot`。

**P1-D4 零相位滤波延拓长度两套后端相差 67–82 倍（已证实）**
- `src/data/preprocess.py:521-525`：`mne_reflect_limited` 的 padlen 实测 2201 样本（8.8 s，250 Hz）/1166（128 Hz），legacy 仅 27–33 样本（108–132 ms）。新旧缓存边缘/首末 trial 不可比；receipt 都写 zero-phase，只靠 `padding_limit_samples` 区分。
- 建议：把 `zero_phase_edge_policy/padding_limit` 提升为缓存一等身份与保真度门禁；legacy 缓存显式标注边缘限制，不与新缓存混用于边界敏感分析。

**P1-D5 epoch 网格与声明合同存在半采样偏移（已证实）**
- `src/data/preprocess.py:926-939` 用 MNE tmin/tmax；实测 128 Hz 的 `times[0]=-0.203125`、裁到 179 点后保留窗 `[-203.1,1187.5)` ms，而声明为 `[-200,1200)`；`contract.py:147-157` 的 `epoch_support_record` 只导出声明 tmin。
- 建议：provenance 增加 `grid_origin_ms`、`last_sample_ms`；或在建轴前按 `round(tmin*sf)` 对齐事件样本；补 128 Hz 路径的网格断言（MOABB 路径已有类似 0.5 采样校验）。

### 3.3 迁移、模型与训练

**P1-T1 transfer_only 静默降级为全校准 waveform LDA（已证实；静态控制流）**
- `src/transfer/calibrated.py:218-223` 只把 `unreadable` 视为失败，`incompatible` 在 `transfer_only` 被放行；`:1132-1146` 候选初始化仅 `{source}`；`src/transfer/calibration_candidates.py:832-856` 在无 completed 候选时无条件回退 `waveform_lda` 并写 `fallback="full_calibration_waveform_lda"`。
- 影响：声明“仅迁移”的 arm 可能实际是 target-only LDA，arm 身份污染。建议：对仅含 source 初始化的模式禁止跨初始化回退，否则抛错/写 `fit_failed`。

**P1-T2 公开池 target runner 未挂整库排除门（已证实）**
- `assert_checkpoint_library_exclusion`（`src/transfer/checkpoint.py:115-136`）在 `src` 内无调用者，仅 `experiments/run_cross_dataset_transfer.py:272-277` 使用；`experiments/run_public_pool_targets.py:94-104` 直接执行并原样信任 `source_study_membership` 自声明。
- 建议：在 job 构建/执行入口强制调用并校验 study membership proofs；为该 runner 补反例测试。

**P1-T3 checkpoint 不绑定 trunk 权重摘要（已证实）**
- `src/research/contracts.py:146-157` 的 `TrainingRunContract` 无 weights digest；`src/transfer/checkpoint.py:357-369` 只检查 state 是非空 Mapping；`:896-906` `strict=False` 只查 key 集合。同形状替换权重可通过内部校验，仅靠外部文件 sha256。
- 建议：checkpoint payload 增加按名字/类型/形状排序的 `trunk_state_dict_sha256` 并在加载时重算比对，纳入 reuse key。

**P1-T4 selector 任务的 fit_calls 预算低估 3–6 倍（已证实）**
- `src/transfer/target_experiments.py:228-231` 写 recipes 但无 budget；`:261-264` 对无 `selection_budget` 的任务按 1 计；`:625-635` 与 `calibration_candidates.py:809-864` 实际是“每候选一次 + 全量 refit 一次”。
- 建议：为 selector 任务生成 `fit_calls_upper_bound=len(recipes)+1`，并在执行后与真实 requested/executed 对账。

**P1-T5 BrainSync schema 下 FoldLocal QC 绕过 personal raw-QC（已证实）**
- `src/transfer/quality.py:40-42` 在非 Personal policy 时直接 return；`:65-76` 只在 BrainSync schema 调用该函数，于是 `FoldLocalArtifactPolicy/Model` 被静默接受；`:118-130,141-143` 明确允许两种 policy。
- 建议：按 `session_schema` 强制 policy 类型（BrainSync 只允许 Personal），并为该入口补反例测试。

**P1-T6 训练轨迹依赖运行时显存余量（已证实；静态+设计确认）**
- `src/train/runtime.py:562-582` 的 `can_preload` 决定行置换用 CUDA 还是 CPU Generator；`:1172-1227` `choose_batch_size` 按空闲显存压 batch。同 seed、不同空闲显存 → 行序与优化步数不同（有效 batch 记录在 `last_runtime`，但“同种子可复现”仅限同一分配结果）。
- 建议：记录 `determinism_profile`（batch/preload 是否被动态调整），为回归比较提供固定 batch/preload 的 deterministic profile；论文/证据中区分“同代码”与“同执行轨迹”。

### 3.4 数学与统计

**P1-M1 information_bound 的 d² 是大幅上偏的 plug-in（已证实，数值复算）**
- `src/transfer/information_bound.py:87-178` 用同一批数据估均值与协方差，再算 `d²=δ^TΣ^{-1}δ`，`:515-518` 直接作为“信息界”输入。合成实验（40 次重复）：p=200/n=60 上偏 14.3×，p=400/n=50 上偏 34.6×，p=800/n=100 上偏 33.3×；而 T×C 特征（数百维）+ 数十 epoch 正是该区间。
- 影响：`p_hit_monte_carlo`、`required_d2_for_targets`、`efficiency` 与“距信息论最优差距”全部系统性偏乐观，不能作为科学结论引用。
- 建议：交叉拟合/样本外估计 d²（协方差与均值分离，K-fold 或 leave-one-selection-out），或做解析偏差修正并报告 p/n；报告禁止称其为“界”。

**P1-M2 信息界的 iid 闪光假设与项目自身 AR 噪声模型冲突（已证实）**
- `src/transfer/information_bound.py:39-49,296-306` 假设闪光 iid；而 `src/transfer/temporal_noise.py:1-13` 明确残差为 AR(p) 相关，BNCI 100 ms 闪光在 800 ms epoch 内高度重叠。正相关下跨 flash 证据不能按 sqrt(F) 累加，MC 与 union bound 高估命中率。
- 建议：按拟合 AR 协方差对 MC 着色（`L_AR @ iid`），同时报告 iid 与 AR 两版；docstring 明确 iid 版是乐观参考。

**P1-M3 95% bootstrap 区间在 n=2–10 严重欠覆盖（已证实，数值复算）**
- `src/stats/hierarchical.py:47` 只要求 `len(values)≥2`，`:62-71` 为纯 percentile bootstrap；`src/stats/transfer.py:190-192` 在 `len>1` 就写入 `paired_interval`。1500 次模拟（1000 重采样）：名义 95% 区间实际覆盖 n=2 49.3%、n=3 76.1%、n=5 84.5%、n=10 89.3%、n=25 92.8%。
- 影响：个位数被试的“95% CI / 显著性”读法不成立。建议：改用 BCa 或 bootstrap-t；n<5 输出 `insufficient_units`；区间对象增加 `method` 与 `min_units_policy`，报告必须带 n 与覆盖率说明。

**P1-M4 exact_covariance 的“对数似然修正”由 ridge 常数支配（已证实，数值复算）**
- `src/transfer/exact_covariance.py:396-440` 对秩亏的 `K_prop` 用 `ridge=max_eig*1e-12`（`:243`）做 Cholesky logdet；实测 `delta_ll=+3241.3`（det 项 −3315.8、quad 项 −74.5），解析量级 `(N−rank)C·log(1/ridge)≈5049`；ridge 1e-12→1e-10 移动约 2500 nats。
- 影响：任何“精确传播 vs AR 近似谁更优”的结论不可用。建议：在 range(A) 子空间做伪行列式/秩修正，或把 ridge 敏感性曲线作为必报项，docstring 注明非可比较边际似然。

### 3.5 项目文件管理

**P1-F1 .git 对象库 8.04 GB / loose 7.26 GiB / 2144 unreachable（已证实）**
- pack 仅 205 MB，最大 loose 对象 1.2 GB、894 MB、453 MB、451 MB、354 MB；多为历史大文件/归档被 add 后移除。影响克隆/备份/迁移与日常 git 性能。
- 建议：备份 `.git` 后 `git gc --prune=now --aggressive`（先确认无未推送需要保留的 unreachable 对象；必要时先 `git fsck --lost-found` 检查）。长期：大二进制（Paper PDF、原始数据）用 LFS 或不入 Git；pre-commit 限制单文件大小。

**P1-F2 工作区 154.9 GB 无分层/保留策略（已证实）**
- mne_data 89.6 GB、experiments/runs 41.0 GB（26.9 GB 为原始数据交付 `tar.gz`；5.5 GB npz、2.3 GB pt、1.56 GB JSON、8134 个 `.py` 中 2183 个是逐 run 复制脚本、614 个 `.pyc`；4398 目录、122 个 >20 MB 文件）、.git 8.0 GB、两个 venv 4.15 GB、frozen 3.32 GB（另有 94 GB 外部 junction）、formal 2.5 GB、cloud_inputs 2.0 GB。
- 建议：定义四层存储（Git 文本 / 本地原始数据 / 派生缓存 / 证据归档）与配额；runs 加 manifest+保留期，原始 tar.gz 移出 runs（如 `cloud_inputs/raw_delivery/`）；`.pyc` 排除；定期 `du` 报表；frozen 归档统一压缩+manifest；删除/替换 `frozen/bidirectional_qc_20260909_acceptance/BrainSync_Tool_newest` 外部 junction（94 GB，会让递归工具/备份重复计算）。

**P1-F3 文档/证据索引与链接已系统性断链（已证实）**
- 32 个相对链接断链（18 个文件），多指向不存在或被删除的 frozen manifest/acceptance；`README.md` 指向的 `frozen/technical_audit_before_20260906.manifest.json` 已在工作树删除；111 个 evidence 中 8 个未被索引（`algorithm_chain_repairs_20260916`、`brainsync_transport_unification_20260917`、`code_quality_audit_20260908`、`e2_prior_scale_20261006`、`source_modules_20260918`、`unified_budget_execution_20260916`、`unified_compact_training_20260916`、`unified_model_implementation_20260916`）；59 个 `doc/assets` 目录 6 个未被引用（含空目录 `operator_research_202604`）。
- 建议：加一个 `experiments/check_docs.py`（链接存在性+evidence/assets 索引覆盖）并纳入 CI；恢复 9 个被删 manifest 或同步删除文档引用；清理空目录/重复 docx。

### 3.6 安全

**P1-S1 模型/结果加载使用 `torch.load(weights_only=False)`（已证实）**
- `src/transfer/checkpoint.py:56`（20+ 入口）、`src/transfer/result_reuse.py:116`（验证 sha 后仍全量 pickle）、`src/baselines/deep.py:587`。对照：`src/transfer/calibrated.py:165-169` 已用 `weights_only=True` + safe_globals，说明可行。
- 影响：来自共享云 worker、下载目录或被替换的 checkpoint 可执行任意代码；sha256 只防篡改、不防“自洽的恶意产物”。
- 建议：公共入口默认 `weights_only=True` + 必要类型白名单；保留旧格式需显式 `trusted=True` 并在 CLI/文档标注信任边界。

**P1-S2 云主机/用户/私钥名硬编码 + 关闭主机校验（已证实）**
- `open_dashboard.ps1:2-8` 硬编码 `connect.weste.seetacloud.com:35832`、user `root`、`/root/autodl-tmp/n2p3-net`、`~/.ssh/id_ed25519_n2p3_cloud_20260827`；`:46,98` 两处 `StrictHostKeyChecking=no`；远程命令由字符串拼接（`:43`）。虽然无密钥值泄露，但暴露基础设施与降低 MITM 防护。
- 建议：参数默认值改为占位符/环境变量；启用 host key 校验（首次 pin known_hosts）；把启动脚本移出仓库或加 `.example` 模板；文档补充。

### 3.7 测试基础设施

**P1-TS1 测试环境耦合与“同 commit 不同测试集”（已证实）**
- 30 个测试文件模块级 `import mne` 且 `importorskip`=0；GTN 92 GB 数据、同级 BrainSync 前端、node、CUDA 决定实际测试集；`addopts` 无 `-rs`，skip 原因不可见。收集 159 文件/2342 用例/0 错误/15.3 s。
- 建议：显式 `pytest.mark.integration/data/gpu`；数据缺失时 `skip` 并带原因（`-ra`）；CI 分“快速契约”与“全量硬件”两层。

### 3.8 文档与 Lean

**P1-L1 formal/ 完全未跟踪（已证实）**
- 385 个源 `.lean`（39 个 Audit*.lean、251 个 N2P3/）、39 个 scripts、`verification_plan.json`、`lakefile.toml`/`lean-toolchain`/`lake-manifest.json` 均未入库；`.lake` 2.47 GB。作为文档中活跃的验证主线，这是版本控制与证据链风险。
- 建议：随 P0 提交源码/脚本/manifest/plan（按 `formal/lean/.gitignore` 排除 `.lake/`、`artifacts/`）；在 CI 中跑 `lake build` + 公理白名单 + `negative/` 失败检查；evidence 引用的 `formal/lean/artifacts/...` 改为可提交的 receipt/summary。

**P1-L2 文档与工作树脱节（已证实）**
- `doc/engineering.zh.md` 首行“更新：2026-09-07”但含 2026-10-18 内容；其中的 `tests/run_headless.py`、`tests/verify_transport_ui.py`、`tests/verify_quality_report.py` 属于独立采集端项目，不在本仓库；`tests/test_preprocess.py` 等旧平铺路径已迁移；历史测试计数（1539/1936/2095/1404）与当前 2331 passed/3 failed 不一致；`doc/README.md` 173 行中约 130 行是状态编年史。
- 建议：六份正文只保留“当前行为+维护规则”，历史状态迁到 `doc/evidence/` 或 `CHANGELOG`；文首维护 `last_verified_commit` 与 `last_verified_suite`；链接检查进 CI。

## 4. P2 — 中优先级改进

### 4.1 代码结构
- 22 个函数 >200 行，最长 `src/baselines/deep.py:1438 _fit_attempt` 1328 行；`src/transfer/subject_adapter.py:756 fit` 632 行；`src/data/brainsync.py:176` 456 行。超长模块：deep.py 2830 行、moabb.py 2018、subject_adapter 1827、epochs 1795、evaluate 1601、raw_artifacts 1598、decision_loss 1403。建议按“校验/装配/落盘/拟合”拆分，至少为大函数加显式阶段函数与单测锚点。
- 21 处 `except ...: pass`（多为 OSError，可接受）与 10 处 `except Exception`（多数转域错误）；`src/data/gtn_dataset.py:199-213` 的宽泛捕获 + `allow_skipped` 会把代码缺陷记成源缺陷、静默缩小队列（建议收窄异常类型并保存 traceback）。
- 候选/日程规则在三处实现（`bi2014a_candidate.py`、`bnci2014_008_candidate.py`、`candidate_task.py`），存在规则漂移风险；建议抽共享模块 + 性质测试。
- 22 个 `src/transfer|models|train` 模块零生产调用或仅诊断（如 `exact_covariance/information_bound/noise_model/operator_lda/temporal_noise` 等）；建议在 README/技术结构标注“diagnostic/no production import”，或移到 `experiments/`。

### 4.2 测试
- `addopts="-q"` 吞最终统计行；无 `-ra`；无 coverage；`tests/__init__.py` 的 `PROJECT_ROOT` 使测试依赖“仓库根在 sys.path”（建议 fixture 或显式 pythonpath）。
- 断言：932 处 `pytest.raises` 中 100 处无 `match=`；11 个用例无任何断言；74 个用例断言全藏 helper；`array_equal` 287 次 vs `allclose` 138 次（数值比较应默认 allclose）。建议为错误契约加 match、将关键断言前置、统一数值容差风格。
- 无 `pytest-timeout`，个别测试 25 s（`tests/train/test_source_execution_resources.py`）；建议慢测试标记 + 超时。

### 4.3 数学/统计（其余）
- `relative_error_trace` 对带通恒 ≈1，不是“AR 近似误差”；建议改用白化增益谱/条件数（`exact_covariance.py:150-153,221-222`）。
- `d2_kronecker_approx` 随物理单位缩放（数据 ×1000 → 1e-6），不是 Mahalanobis 距离；建议重命名或乘回尺度（`information_bound.py:402-435`）。
- `efficiency` 文档定义与代码相反、比例未 clamp（`information_bound.py:489,535-546`）；`byte_budget` 容量说明算错（N≈2048 而非 2900）且峰值低估 3–4 倍（`:185-188`）。
- `noise_model` 3 draws 估 2p 维协方差无下限；`operator_lda` 两套正则判据（1e-10 与 ridge=1e-3·mean(λ)）不一致；`_conditional_mmd` 逐 cell 宏平均 + 无带宽下限（分区 04）。
- 校准切分：缺 per-recording availability 时跳过时序泄漏检查（`calibration_split.py:104-108,169-182`）；候选拟合缓存 key 缺 `block_id/selection_repetitions`（`calibration_candidates.py:395-415`）。
- 重采样抗混叠仅 −22 dB@70 Hz；被丢弃的 1200 ms 端点样本以 −60 dB 漏入基线窗；EOG 在新/legacy 路径滤波行为不同（分区 07）。

### 4.4 工程与依赖
- `ruff format --check` 有 273 个文件待格式化（ruff check 已通过）；建议在独立提交中 `ruff format` 并加入 CI。
- `pyproject` 无 `readme/license/authors/classifiers/urls/scripts`；根目录无 `LICENSE`；`src/baselines/structured_erp.LICENSE` 仅为第三方声明，建议补根 LICENSE/NOTICE 与 README“第三方代码与许可”小节。
- `.claude/settings.local.json` 设 `defaultMode=bypassPermissions` 且仅靠用户全局 ignore；备份文件未被忽略（建议仓库 .gitignore 显式覆盖 `.claude/`）。
- `open_dashboard.ps1` 未在 README/文档索引中出现（可发现性差）；`.venv` 含 6 个 lock 外手工包（pypdf/paramiko 等）。
- `requirements-minimal.txt` 与 pyproject 边界不清；`setup_cloud.sh` 缺失。

### 4.5 项目文件管理（其余）
- `Paper/` 59 个 PDF、191 MB 入 Git：既造成仓库膨胀，也存在再分发版权风险；建议移出 Git 或 LFS，并在 `doc/` 建文献索引（标题/DOI/许可）。
- `doc/assets` 173 MB/3610 文件未跟踪；含 6 个未被引用目录与空目录 `operator_research_202604`；`doc/progress_report_20260929*.docx` 与 md 重复；建议归档/去重/索引。
- `frozen` 100 个目录中 12 个无 manifest（lean_v07/v08 系列、lean_fold_artifact、local_history）；9 个 manifest 被删除但仍被文档引用；建议恢复或更新索引，并为新归档加 manifest 校验脚本。
- `frozen/worktree_recovery_20261016.json` 记录了过去的清理事故（“cleanup stopped; affected author trees restored”，含丢弃文件清单）；建议把工作树生命周期（创建/暂停/收尾/删除）写成固定清单并自动化检查独有提交与未提交源码。
- 命名：`operator_research_202604` 空目录疑似日期笔误；`doc/evidence/lean_outcomes_20261018.zh.md` 等使用绝对 `D:/BCI/...` 链接，不可移植。

### 4.6 安全（其余）
- dashboard_server 无鉴权（绑定 127.0.0.1，可接受；远程经 SSH 隧道）；`do_POST` 对 Content-Length/JSON 解析缺少健壮性处理；`_run_dir` 已用 `RUN_NAME.fullmatch`+包含校验（正面）。
- 无 `pip-audit`/依赖漏洞扫描；无 secret 扫描 pre-commit。
- 数据集许可（BI2014a/BNCI2014-008/GTN/Mainsah 等）未在文档/代码中系统声明；成人/设备确认与代码门禁的对应关系建议显式化。

### 4.7 性能（其余）
- `experiments/benchmark_unified_training.py`（未入库）只测合成 prepared-array，无 CI 回归门禁；建议把关键吞吐（source pack、target fit、attested load）做成可比较的 benchmark + 阈值。
- `GpuPerformanceScheduler.recommended_concurrent_workers` 在显存 <24 GB 时返回 1；`choose_batch_size` 使用固定系数（0.55/0.30、activation×16、persistent×5）；`pin_memory` 单缓冲。建议在 5090 上实测后回归，并记录“启发式→实测”的差距。
- Windows 无 cgroup/affinity 时 CPU 预算可能高估（`resolve_cpu_worker_plan` 只用 os.cpu_count）；建议在 Windows 用 Job Object/进程亲和性检测或显式 `--cpu-threads`。
- 41 GB runs 的 I/O 与 8134 个 py 副本/614 pyc 建议按 4.5 的保留策略处理。

## 5. 正面发现（建议保持）

- **合同与准入整体 fail-closed**：缓存 attested 加载、`raw_artifacts`（size/SHA/MD5+inode 前后比对）、`source_records` 台账、预处理执行回执、结果复用完整性校验都做了多重校验；错误路径明确终止而非静默补算。
- **数学实现质量总体高**：EA 的 `R^{-1/2}`、SupCon 多正样本式、无偏 MMD²、`F K F^T` 与稠密 `A K A^T` 一致到 2.9e-14、baseline 半开窗口、50 Hz/Q30 陷波 −3 dB 宽 1.6666 Hz、250→128 Hz 的 351→180→179 几何与文档一致、SplitBN 多 batch 合并 ≤1e-9、observed-channel 路径与普通路径全观测 max|diff|=0。
- **工程卫生**：ruff check 全通过（399 文件）；`uv lock --check` 通过；无裸 except、无可变默认参数；`tests/` 新模块覆盖：23/23 未跟踪 src 模块均被测试引用；数值比较大量使用 allclose。
- **资源合同**：`configs/cloud_training_resources.json` 与 `src/research/source_execution.py` 的 25 CPU/4 GPU floor、cgroup/affinity 解析、GPU 设备锁、AMP/BF16、内存快照均与用户要求一致（配置未入库是需要修的流程问题，不是设计问题）。
- **Lean 工程**：lean-toolchain/Mathlib commit 固定；无 `sorry/admit/axiom` 作为证明（仅注释提及）；公理白名单（propext/Classical.choice/Quot.sound）+ `negative/` 反例强制失败 + `verification_plan.json` 源码哈希绑定，方法学上是本项目最强的可复现设计之一。
- **安全基础**：未发现硬编码密钥/凭据/PII；`np.load(allow_pickle=False)` 全覆盖；`CalibratedPredictor` 路径已用 `weights_only=True+safe_globals`；dashboard 默认 localhost+run 名校验；README 入口表 19 个脚本全部存在。

## 6. 分阶段改进路线

### 第 0 阶段（立即，1–3 天）
1. 物理快照当前工作树（含未跟踪源码/配置/formal），生成 SHA-256 清单。
2. 修复 3 个 pytest fixture（或回退 LDA 特征校验并记 TODO），恢复全绿。
3. EOL 归一化独立提交：`.gitattributes` + `git add --renormalize` + 自哈希改为 LF 规范化。
4. 拆分提交：tests 迁移 → src 新模块 → configs → formal → doc/assets/evidence → frozen manifest 修复。
5. 备份后 `git gc` 清理 7.26 GiB loose objects；移除/迁移 94 GB junction。
6. 恢复 9 个被删 frozen manifest 或更新文档链接；补 8 个未索引 evidence 条目。
7. 安全：3 处 `weights_only=False` 默认收紧；`open_dashboard.ps1` 去硬编码/开主机校验；仓库 .gitignore 显式覆盖 `.claude/`。

### 第 1 阶段（2–6 周）
1. CI：ruff + pytest（快速/全量分层）+ `git diff --check` + `uv lock --check` + 文档链接检查 + Lean `lake build`（缓存）。
2. 测试迁移映射表（130 个消失函数逐条结论）与覆盖率报告。
3. 数学修复：d² 交叉拟合/偏差修正、AR 着色 MC、BCa/bootstrap-t + n<5 拒绝、logdet 秩修正/敏感性、统计报告统一口径。
4. 数据合同：MOABB 执行回执、BrainSync/BIDS 内容哈希、epoch 网格 origin、零相位 policy 身份化、attested 加载内容寻址。
5. 迁移/模型：transfer_only 回退硬失败、公开池整库排除门、checkpoint 权重摘要、selector 预算修正、BrainSync QC policy 绑定。
6. 文档治理：六份正文去编年史、状态迁 evidence/CHANGELOG、`last_verified_commit`、修复 32 条断链。
7. 打包：顶层命名空间或白名单、wheel 自包含、`[tool.uv]` 版本/索引固定、GPU extra。

### 第 2 阶段（2–3 个月）
1. 结构重构：拆分 1300/632 行函数与 2800 行模块，候选规则统一。
2. 性能：attested 缓存内容寻址、runs 保留策略与 manifest、5090 实测 benchmark + 回归阈值、原始数据移出 runs。
3. 仓库瘦身：Paper 移出 Git/LFS、frozen 归档统一压缩+manifest、doc/assets 去重、工作区存储配额。
4. Lean：formal 入库 + CI 证据链（artifact receipt 可提交）。
5. 安全/合规：根 LICENSE/NOTICE、数据集许可清单、secret 扫描、dashboard 鉴权选项。

### 统一验收命令
```powershell
..venvScriptspython.exe -m ruff check src experiments tests
..venvScriptspython.exe -m pytest            # 应 0 failed
uv lock --check
git diff --check
git status --porcelain                          # 应只剩明确忽略项
git ls-files --eol | Select-String 'w/crlf|w/mixed'   # 归一化后应无非预期项
```

## 7. 审计边界与未验证项

- 未运行任何训练、云任务或真实数据集实验；未连接 BrainSync 采集端（独立项目）。
- CUDA/5090 的实际吞吐、显存与并发行为未在当前机器复测；性能结论以静态代码+历史文档+合成基准阅读为主。
- 3 个失败测试的根因已定位到 fixture 与新增校验的维度不一致；未修改代码验证“修复后全绿”。
- BrainSync/BIDS 内容哈希缺失标为“疑似”：采集端可能另行记录哈希，需跨项目确认。
- 数学数值验证限于小规模合成数据；真实 T×C/epoch 规模下的 d² 偏差可能更大。
- 文档链接/索引/大小等为审计时点快照；`doc/assets` 与 runs 的具体保留价值需项目所有者裁定。
- 13 份分区报告含更多逐条证据与复现脚本，见 `doc/assets/repository_audit_20261019/`（对应 `tmp/audit/` 原始文件）。

## 8. 附录：证据索引

| 分区 | 报告 | 主要结论 |
|---|---|---|
| 01 | data-core.md | 缓存放大、MOABB 声明即证据、BIDS 无哈希、epoch 网格偏移 |
| 02 | models-train.md | 显存依赖轨迹、SplitBN EMA、adapter 状态、PDS 预算 |
| 03 | transfer-core.md | transfer_only 回退、整库排除门、checkpoint 权重摘要、fit_calls、QC 绕过 |
| 04 | transfer-math-stats.md | d² 上偏、iid/AR 冲突、bootstrap 覆盖、logdet ridge、单位缩放 |
| 05 | research-experiments.md | 未入库依赖、无法续跑、执行身份不一致、非原子写 |
| 06 | tests.md | 重组不可追溯、无 CI/覆盖/超时、环境耦合、断言质量 |
| 07 | math-contract.md | 零相位 padlen、时间轴半采样、切分预热、EOG/重采样 |
| 08 | engineering-tooling.md | EOL+自哈希、打包、环境不可复现、无 CI、format |
| 09 | repo-management.md | git/存储/归档/索引治理 |
| 10 | docs-consistency.md | 索引/链接/状态编年史/文档-代码一致性 |
| 11 | security-privacy.md | 凭据/隐私/许可/执行安全 |
| 12 | performance.md | 并行/IO/缓存/benchmark |
| 13 | formal-lean.md | formal 未跟踪、.lake/artifacts、CI/证明工程 |

> 本报告为审计建议，不构成对任何科学结论的重新认证；修复应在独立分支按上述阶段与验收命令执行。
