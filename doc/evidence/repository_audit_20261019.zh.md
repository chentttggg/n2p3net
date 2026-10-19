# 全仓库审计与改进建议

- 日期：2026-10-19（Asia/Shanghai）
- 对象：`D:\BCI\n2p3-net` 当前工作树（HEAD=`f1a303c`）及本地数据/归档/Git 对象库
- 约束：**只读审计，未修改任何源文件/测试/配置/指南**。本报告与 `doc/assets/repository_audit_20261019/` 下的分区报告为新增文件；审计前后 `src/tests/experiments` 共 368 个 py 文件 SHA-256 一致。
- 方法：13 个只读审计分区 + 3 个独立 red-team 复核 + 8 个补充分区（G1–G8）+ Lead 横向核查；全量 pytest、ruff、uv lock 检查、git/磁盘/文档/依赖分析；关键数学结论做合成数据数值复算。分区报告含逐条证据与复现脚本。

## 0. 总体结论

这是一个**合同、执行身份与可复现性设计明显强于一般研究代码**的仓库：缓存 attestation、raw artifact 认证、结果复用完整性、预处理执行回执、Lean 公理审查都做得相当扎实。但当前处于**危险的“历史与工作树双断裂”状态**：

1. **Git 历史断裂**：`main` 停在 2026-09-09，10 月真正的工作在 66 个分支与工作树中；另有 **112 个 dangling commit**，其中 **30 个无 ref/reflog 保护**（最新 `451af9d`，2026-10-12），一次 `gc --prune=now` 会立即销毁；另 82 个由 reflog 保护（`e148163` 约 2026-11-13 后过期）。
2. **工作树断裂**：428+ 项未提交、23 个 `src` 模块未跟踪（其中 10 个被 14 个已跟踪文件在工作树版本里 import）、184/185 个测试文件未跟踪、整个 `formal/` 未跟踪；当前代码无法由 Git 重建。
3. **验证断裂**：全量 pytest **3 failed / 2331 passed / 8 skipped**；1189 个 HEAD 测试函数中 130 个在当前树消失，119 个无同体替代且无迁移记录。
4. **证据链断裂**：约 31 条文档死链/17 文件（10 分区计 32/18）、8 个 evidence 未索引、Lean 1528 条 artifacts 引用中 753 条指向已删目录、`verify_erp.py` 锁定的 189 个源码哈希已有 6 个漂移。

**因此，最高优先级不是算法改进，而是“先救历史、再收敛工作树、后修合同”**。本报告按 P0/P1/P2 给出可直接执行的改进项与验收命令。

## 1. 现状快照（本次实测）

| 指标 | 数值 | 证据/说明 |
|---|---|---|
| 工作树状态 | 429–430 项：80 modified、118 deleted、230–231 untracked | `git status --porcelain` |
| main 最后提交 | `f1a303c` 2026-09-09 09:58 | `git log -1 main` |
| 分支 | 66 个（最新 `codex/research-director-resume-20261013` 2026-10-19 00:05） | `git for-each-ref` |
| Dangling commit | **112 个**，其中 30 个无 ref/reflog 保护（最新 `451af9d`，10-12）；82 个由 reflog 保护 | `git fsck --unreachable --no-reflogs` |
| 源码规模 | `src` 120 py / 53.8k LOC；tests 159 个 `test_*.py` / 1593 个 `def test_` | — |
| 工作区磁盘 | **154.9 GB / 237k 文件**（apparent，含 junction） | mne_data 89.6 GB、experiments 45.0 GB、.git 8.0 GB、venv 4.15 GB、frozen 自有约 3.3 GB + junction 约 92 GB、formal 2.5 GB、cloud_inputs 2.0 GB、Paper 0.19 GB |
| .git | 8.04 GB：pack 205 MB + loose **7.26 GiB**；1601 unreachable blob / 2144 unreachable 对象 | 前 5 大对象 1.22 GB/894 MB/453 MB/451 MB/354 MB，头部为 ZIP/GZIP |
| 全量 pytest | **3 failed, 2331 passed, 8 skipped, 22 warnings, 439.48 s** | `tmp/audit/pytest-full2.log`；失败在 `tests/transfer/test_preprocessing_execution_gate.py` |
| 静态检查 | `ruff check` 全过（399 文件）；`ruff format --check` 280 文件待格式化；`uv lock --check` 通过 | 分区 08 |
| 运行环境 | Python 3.14.3；`.venv` torch 2.13.0+cpu（0.98 GB）；`.venv-gpu` torch 2.13.0+cu130（3.17 GB，CUDA 可用） | `requires-python >=3.11`、ruff target py311、无 CI |

## 2. P0 — 阻断级（必须先处理）

### P0-1 Git 历史处于可丢失状态：main 停滞 + 112 个 dangling commit（已证实）

- `main` 最后提交 2026-09-09；此后提交分布在 66 个 `codex/*`、`local-recovery/*` 等分支上（4 个 remote ref 不含 10 月工作），**无远端副本**。
- `git fsck --unreachable --no-reflogs` 列出 2144 个对象 = **112 个 dangling commit + 431 tree + 1601 blob**；其中 **30 个 commit 未被任何 ref/reflog 保护**（最新 `451af9d`，2026-10-12），`git gc --prune=now` 会立即销毁；另 82 个由 reflog 保护（如 `e148163` 是 `codex/source-lineage-storage@{2026-10-14}`，约 11–13 天后过期才可回收）。默认 `git fsck --unreachable`（带 reflog）只显示 1812 个对象、331 个受保护对象。
- **影响**：一次误操作即可丢失 10 月 6–14 日的真实集成工作；这是全仓库最高风险。
- **建议（顺序不可颠倒）**：
  1. 先对 **30 个无 ref/reflog 保护的 commit** 逐个 `git branch rescue/<sha> <sha>`（用 `git fsck --unreachable | Select-String 'unreachable commit'` 取清单；`e148163` 受 reflog 保护，可作示例但非最紧急）；
  2. `git bundle create backup-20261019.bundle --all` 并异机保存；
  3. 核对 66 个分支中哪些已合并/可归档；
  4. 以上完成后才允许 `git gc`（见 P1-R2）。
- **验收**：`git branch --contains e148163` 输出非空；`git bundle verify backup-20261019.bundle` 成功。

### P0-2 当前工作树不可由 Git 重建（已证实）

- 23 个 `src` 模块未跟踪；**14 个已跟踪文件在工作树版本中形成 27 条 import 边、指向 11 个未跟踪目标**（HEAD 侧逐条复扫为 0）：`models/n2p3net.py:14` → `models.parameter_groups`（未跟踪）、`transfer/calibrated.py:23` → `baselines.structured_erp`、`transfer/calibration_candidates.py` → `baselines.structured_erp`/`temporal_transfer`、`transfer/recipes.py` → `structured_erp`、`baselines/deep.py:47` → `train.early_stopping`、`stats/transfer.py:12` → `research.comparison`、`transfer/binary.py`/`target_experiments.py` → `research.comparison`/`source_matrix`、`research/source_selection.py` → `research.source_catalog` 等。
- 未跟踪源码含核心实现：`source_records.py`（61 KB）、`ea_artifact.py`（34 KB）、`exact_covariance.py`、`information_bound.py`、`noise_model.py`、`operator_lda.py`、`temporal_noise.py`、`continuous_decision.py`、`raw_continuous.py`、`source_objective.py` 等；19/69 个 `experiments/` 顶层脚本、4/9 个 `configs/`（含 `cloud_training_resources.json`、`unified_training_budget.json`）、整个 `formal/` 也未跟踪。
- **注意口径**：HEAD 的旧代码本身不 import 这些未跟踪模块（已用 `git grep HEAD -- src` 抽查），所以“fresh clone HEAD 必 ImportError”不成立；正确表述是**当前工作树无法由 Git 重建**。
- **影响**：机器迁移、磁盘故障、cloud 重装、`git checkout` 都会丢失当前能力；所有“执行身份”依赖一个未入库物理快照。
- **建议**：先物理快照（`experiments/freeze_source_snapshot.py` 或补丁+压缩归档+SHA-256 清单），再按 6 个原子提交入库：① EOL 归一化；② tests 迁移；③ 新 src 模块；④ configs；⑤ formal（源码/脚本/manifest，不含 .lake/artifacts）；⑥ doc/assets+evidence+frozen manifest。每步跑 ruff + 全量测试。
- **验收**：`git status --porcelain` 仅剩明确忽略项；临时 clone 中 `python -c "import transfer.target_experiments, data.source_records"` 成功。

### P0-3 全量测试 3 失败 + 测试重组不可追溯（已证实）

- 失败均在未跟踪的新测试 `tests/transfer/test_preprocessing_execution_gate.py`：`test_new_personal_model_keeps_binding_on_save_load_and_admits_new_source_context`、`test_historical_personal_model_reloads_without_promoting_execution_evidence`、`test_personal_model_rejects_receipt_substitution_even_when_outer_metadata_is_resealed`，错误为 `ValueError: calibrated LDA feature state has invalid shape or scale`。
- 根因（已定位）：fixture 用 `n_channels × n_times`（实测 300 维）构造 LDA state（测试 `:70-72`），而工作树新增校验要求 `selected_count × downsampled_time_points(..., target_hz=20.0)`（`src/transfer/calibrated.py:650-661`；100 Hz/100 点示例期望 60=3×20）。生产 LDA 拟合路径使用 20 Hz 降采样特征，故 fixture 陈旧；但“历史个人模型重载”用例因此失败，兼容约定需人工确认。
- HEAD 1189 个测试函数（1188 唯一）中 130 个名称在当前树消失；difflib cutoff 0.83 下 11 个可视为改名、**119 个无近似替代**；消失名单含路径逃逸、哈希不一致、预算下限、forced-switch 账本等安全/契约测试（分区 06）。`tests/` 下 109 个 tracked 文件被删除、185 个磁盘 py 中约 184 个未跟踪。
- **影响**：不能宣称验证通过；重组等价性无法由仓库自证；直接提交会带入未知回归。
- **建议**：① 修 3 个 fixture 或回退校验并单列迁移计划，恢复全绿；② 产出 `HEAD tests → 当前 tests` 迁移映射（kept/merged/retired+reason），119 个无替代函数逐条确认；③ 重组与行为变更分两个 commit。
- **验收**：`pytest` 全绿；映射表每个缺失函数有结论。

### P0-4 Lean 形式化完全未入库 + 证据链已断（已证实）

- `formal/` 完全未跟踪：Git 可见 414 文件 / 3.63 MB（磁盘 formal 32,699 文件 / 2.53 GiB、8,419 个 `.lean`；其中 385 个源 `.lean`、39 个 `Audit*.lean`、39 个 scripts）、`verification_plan.json`、`lakefile.toml/lean-toolchain/lake-manifest.json` 零 Git 历史；`doc/assets/lean_*` 约 103.62 MB 证据也未跟踪。
- 证据链断裂：`doc` 中 1528 条 `formal/lean/artifacts/...` 引用中 **753 条（49%）指向已删目录**（`doc/assets/lean_gaussian_20261016/cleanup.json` 等鼓励删除构建产物，与 evidence 保留路径冲突）。
- 最严验证链已红：`verify_erp.py` 锁定 189 个源码 SHA-256，其中 6 个与当前树不符（`features.py`、`structured_erp.py`、`stats/transfer.py`、`calibrated.py`、`exact_covariance.py`、`operator_lda.py`），重跑必在 `verify_erp.py:151-153` 失败；缺 rebase 流程。
- 工具链锁死本机：`run_lean.py:415` 硬编码 `D:/BCI/lean-tools`、`:427` 硬编码 `MATHLIB_CACHE_DIR`，PATH 无 elan；`.lake` 2.67 GB/30887 文件。
- 其他：39 个 `Audit*.lean` 仅 12 个被脚本引用（11 个运行时生成），27 个无入口；93 个 `negative/` 中 55 个无脚本引用；`active_checkpoint` 自报 2890 条“已验收定理”（源码约 3,418 条 theorem，无 theorem→源码映射，无法独立验证），零 Git 历史。
- **建议**：立即提交源码/脚本/manifest/计划（按 `formal/lean/.gitignore` 排除 `.lake/`、`artifacts/`）；只提交 KB 级回执而非大 artifact；修 753 条断链；建立 6 源哈希 rebase 流程；CI 用 elan+缓存跑 `lake build`+公理白名单+`negative` 失败检查。

### P0-5 dashboard 零鉴权 + 关闭主机校验形成可注入链路（安全分区判定，条件触发）

- `experiments/dashboard_server.py:540` 唯一门槛是自定义头 `X-Dashboard-Confirm`（可被页面自行伪造）；`/api/monitor/*` 连它都不需要且可触发 `subprocess.Popen`；`/api/kill`、`tmp/` 与 `experiments/` 静态读取端点可用；`extra_run_roots`（`:589`）把整个 `tmp/` 变成 HTTP 可读；默认绑定 127.0.0.1。
- `open_dashboard.ps1:46,98` 两处 `StrictHostKeyChecking=no`；攻击者在网络路径上可冒充云主机，经隧道返回同源恶意 `dashboard.html`，从而调用 `api/kill` 杀训练并读走 `runs/` 全量结果。
- **影响**：在不可信网络 + 隧道运行场景下可造成训练中断与结果泄露；若只在本机 localhost 使用，风险显著降低。
- **建议**：dashboard 加启动即生成的 bearer token / 会话 cookie，写端点强制校验；`api/kill` 加二次确认与 CSRF token；`extra_run_roots` 收窄到 `tmp/runs`；`open_dashboard.ps1` 开启 host key 校验并去硬编码（见 P1-S2）。

## 3. P1 — 高优先级

### 3.1 工程与可复现性

**P1-E1 行尾未归一化 + 源码自哈希 → 跨平台 resume/证据闸门必失败（已证实）**
- `.gitattributes` 仅 6 行，无 `* text=auto`/`*.py eol=lf`；根因是本地 `core.autocrlf=true`（分区 09），工作树 157 个 `i/lf w/lf`、69 个 `w/mixed`。
- **8 个模块**在 import 时对自身字节做 SHA-256（另有 3 个运行期哈希点）：`src/data/preprocess.py:32`、`src/data/epochs.py:46`、`src/transfer/operator_lda.py:27`、`noise_model.py:19`、`exact_covariance.py:53`、`information_bound.py:79`、`continuous_decision.py:36`、`temporal_noise.py:34`，以及 `src/research/recording_quality.py:239` 等运行期点；CRLF/LF 同一文件哈希不同。
- 真正跨平台闸门是 `src/data/source_records.py:98` 的 `data_modules_sha256` 与 `:207` 的严格比较（`experiments/prepare_source_records.py:117` 的 `preparation_method`）：对 `src/data` 下所有 `*.py` 逐个哈希（43 个文件中 16 个 CRLF/mixed），任何无关改动或换行差异都会使全部源缓存不可 resume；epochs/preprocess 摘要只校验格式，`temporal_noise.py:219` 才是严格相等。
- 建议：`.gitattributes` 增加 `* text=auto eol=lf`（至少 py/toml/json/md），独立提交 `git add --renormalize .`；哈希改用 LF 规范化字节或 git blob；`data_modules_sha256` 改显式模块清单。

**P1-E2 打包污染：9 个通用顶层包 + wheel 不自包含（已证实）**
- `pyproject.toml:25-26` 的 `packages.find where=["src"]` 实际发布 `audit_p300, baselines, data, measurement, models, research, stats, train, transfer` 9 个顶层名（`src/n2p3_net.egg-info/top_level.txt`）；`data`/`models`/`transfer` 与 PyPI/其他项目冲突风险高。
- `src/data/source_records.py:99` 读 `experiments/prepare_source_records.py`，wheel 安装后不存在 → `FileNotFoundError`。
- 建议：`src/n2p3_net/` 命名空间或显式 `include` 白名单；实验脚本哈希改为可注入/随包清单；补 `py.typed`、LICENSE、readme、scripts。

**P1-E3 环境不可复现：lock 多解 + 手工 CUDA 覆盖（已证实）**
- torch marker：Linux x86_64 py<3.14 → `torch==2.8.0`，其他 → `torch>=2.4`；`uv.lock` 同时锁 2.8.0(cu12) 与 2.13.0(cu13)；本地 `.venv`=2.13.0+cpu、`.venv-gpu`=2.13.0+cu130 为文档化手工覆盖，lock 无 cu130 index；无 `[tool.uv] required-version`（`.venv` 由 uv 0.11.26 创建，`.venv-gpu` 与当前 uv 为 0.12.9）；`.venv` 含 6 个 lock 外手工包。
- 建议：固定 uv 版本与 Python minor；用 `[tool.uv.sources]/[[tool.uv.index]]` 定义 gpu extra；torch marker 加上界。

**P1-E4 无 CI、pytest 配置过薄（已证实）**
- 无 `.github/`、无 pre-commit；`pyproject` 仅 3 行 pytest 配置；全树只有 `parametrize`/`skipif` 两个 marker；无 coverage/timeout/strict/warning 策略；`addopts="-q"` 在 pytest 9.1.1 下不打印最终统计行。
- 建议：CI 至少 ruff + pytest + `git diff --check` + lock 检查 + 文档链接检查；注册 slow/integration/gpu markers；加 `-ra --strict-markers` 与 timeout。

**P1-E5 requirements 漂移与云端引导脚本缺失（已证实）**
- `requirements.txt` 仅 `-e .[signals,baselines,dev]`；`requirements-minimal.txt` 提及不存在的 `setup_cloud.sh` 且不含 torch；`threadpoolctl` 仅在该文件出现（隐式依赖）。
- 建议：`uv export` 生成或删除；补 `scripts/setup_cloud.sh` 入库；显式声明 threadpoolctl。

**P1-E6 关键资源合同未入库（已证实）**
- `configs/cloud_training_resources.json`（cpu_budget=25、gpu_workers=4、cloud_resource_floor）与 `configs/unified_training_budget.json` 未跟踪，而 `src/research/source_execution.py:119-221` 消费它们；用户要求的 25/4 下限不可审计。
- 建议：随 P0 入库；CI 校验 floor 字段与 AGENTS.md 一致。

### 3.2 数据、预处理与输入合同

**P1-D1 attested 缓存加载 O(size) 复制+哈希+105% 空间门槛（已证实）**
- `src/data/epochs.py:1215-1255` 每次 attested 加载全量复制 NPZ 并重算 SHA；`:1073-1100` 要求 `byte_size*1.05+64MiB` 空闲；4 worker 下读放大与临时空间近翻倍。
- 建议：内容寻址只读对象；进程内验证缓存；空间门槛可配置。

**P1-D2 MOABB 高层入口“声明即证据”（已证实）**
- `src/data/moabb.py:1783-1791` 由 MOABB 内部执行 `raw.filter(method="iir")`；`:1985-1996` 的 provenance 只是抄声明，无 `preprocessing_execution` 回执；MNE/MOABB 升级会静默改算子；`picks="data"` 把 EOG 一并滤波。
- 建议：MOABB 入口生成完整执行回执（实际 SOS/iir 参数、滤波前后样本、重采样函数、逐通道段）；版本白名单。

**P1-D3 BrainSync/BIDS 会话准入无内容哈希（疑似，需采集端确认）**
- `src/data/bids_eeg.py:228-330`、`src/data/brainsync_contract.py:359-472` 只做元数据交叉校验；个人校准证据无法绑定具体文件字节（公共源域有逐文件 SHA-256）。
- 建议：manifest 对每个文件记录 `(sha256, byte_size)` 并流式校验。

**P1-D4 零相位滤波延拓长度两套后端相差 67–82 倍（已证实）**
- `src/data/preprocess.py:521-525`：`mne_reflect_limited` 请求 padlen 实测 2201 样本（8.80 s @250 Hz）/1166（9.11 s @128 Hz），legacy 为 33 样本（132 ms @250 Hz）/27（108 ms @128 Hz）；实际 pad 受 `min(padlen, segment_len−1)` 截断（179 点 epoch 有效 pad=178）。独立复现：连续段前/后 0.5 s max|Δ|=0.379/0.681，中段 10 s 收敛到 0.0018（信号 RMS≈0.74）→ **边缘样本不可直接混用**，不是整段不可比。
- 建议：把 `zero_phase_edge_policy/padding_limit` 提升为缓存一等身份与保真度门禁；legacy 缓存显式标注。

**P1-D5 epoch 网格与声明合同存在半采样偏移（已证实）**
- `src/data/preprocess.py:926-939` 用 MNE tmin/tmax；**128 Hz 原生**实测 `times[0]=-0.203125`、最后采样点 1187.5 ms、半开支撑 `[-203.125,1195.3125)` ms；**250 Hz 原生重采样后**为 `[-200,1198.4375)` ms；两条路径窗长 179/128=1398.4375 ms，均比声明的 1400 ms 短 1.5625 ms，声明 1200 ms 端点未被保留。`src/data/contract.py:147-157` 的 `epoch_support_record` 只导出声明 tmin。
- 建议：provenance 增加 `grid_origin_ms/last_sample_ms`；或建轴前按 `round(tmin*sf)` 对齐；补 128 Hz 网格断言。

### 3.3 迁移、模型与训练

**P1-T1 transfer_only 静默降级为全校准 waveform LDA（已证实；静态控制流）**
- `src/transfer/calibrated.py:218-223` 只把 `unreadable` 视为失败，`incompatible` 在 `transfer_only` 被放行；`:1132-1146` 候选初始化仅 `{source}`；`src/transfer/calibration_candidates.py:832-856` 在无 completed 候选时无条件回退 `waveform_lda` 并写 `fallback="full_calibration_waveform_lda"`。
- 影响：声明“仅迁移”的 arm 可能实际是 target-only LDA。建议：仅含 source 初始化的模式禁止跨初始化回退。

**P1-T2 公开池 target runner 未挂整库排除门（已证实）**
- `assert_checkpoint_library_exclusion`（`src/transfer/checkpoint.py:115-136`）在 `src` 内无调用者，仅 `experiments/run_cross_dataset_transfer.py:272-277` 使用；`experiments/run_public_pool_targets.py:94-104` 直接执行并原样信任 `source_study_membership` 自声明。
- 建议：在 job 构建/执行入口强制调用并校验 study membership proofs。

**P1-T3 checkpoint 不绑定 trunk 权重摘要（已证实）**
- `src/research/contracts.py:146-157` 无 weights digest；`src/transfer/checkpoint.py:357-369` 只检查 state 非空；`:896-906` `strict=False` 只查 key。同形状替换权重可通过内部校验，仅靠外部文件 sha。
- 建议：增加按名字/类型/形状排序的 `trunk_state_dict_sha256`，加载时重算比对并纳入 reuse key。

**P1-T4 selector 任务 fit_calls 预算低估 3–6 倍（已证实）**
- `src/transfer/target_experiments.py:228-231` 写 recipes 但无 budget；`:261-264` 对无 `selection_budget` 的任务按 1 计；`:625-635` 与 `calibration_candidates.py:809-864` 实际是“每候选一次 + 全量 refit 一次”。
- 建议：`fit_calls_upper_bound=len(recipes)+1`，执行后与真实 requested/executed 对账。

**P1-T5 BrainSync schema 下 FoldLocal QC 绕过 personal raw-QC（已证实）**
- `src/transfer/quality.py:40-42` 非 Personal policy 直接 return；`:65-76` 只在 BrainSync schema 调用；`:118-130,141-143` 允许两种 policy → `FoldLocalArtifactPolicy/Model` 被静默接受。
- 建议：按 `session_schema` 强制 policy 类型，并补反例测试。

**P1-T6 训练轨迹依赖运行时显存余量（已证实）**
- `src/train/runtime.py:562-582` 的 `can_preload` 决定行置换用 CUDA 还是 CPU Generator；`:1172-1227` `choose_batch_size` 按空闲显存压 batch。同 seed、不同空闲显存 → 行序与优化步数不同。
- 建议：提供固定 batch/preload 的 deterministic profile，并在证据中区分“同代码”与“同执行轨迹”。

### 3.4 数学与统计

**P1-M1 information_bound 的 d² 是大幅上偏的 plug-in（已证实，独立复核修正口径）**
- `src/transfer/information_bound.py:87-178,515-518` 用同批数据估均值与协方差再算 `d²=δ^TΣ^{-1}δ`；独立复现（40 次重复，Σ=I，真 d²=0.5）：p=200、**每类 n=60（总 n=120）**上偏约 14.4×；若总 n=60 则约 28.6×；p=400、每类 n=50 约 33.5–34.4×（10 seed 均值 33.7×，原文 34.6× 为种子相关值）；p=800/每类 100 约 33.3×。T×C 特征（数百维）+ 数十 epoch 正在此区间。
- 交叉拟合（Σ 与 δ 分开）并未消除偏差（仍 14.7×/34.0×）：有限样本 δ̂ 满足 E||δ̂||²=d²+p/n；需解析偏差修正 d²−p/n_eff、独立大样本估 δ，或直接拒绝 p/n 过大的报告。
- 影响：`p_hit_monte_carlo`、`required_d2_for_targets`、`efficiency` 与“距信息论最优差距”全部系统性偏乐观，不能作为科学结论引用。建议：报告必须标注 n 是 per-class 还是 total、给出 p/n 与偏差修正；禁止称其为“界”。

**P1-M2 iid 闪光假设与项目自身 AR 噪声模型冲突（部分成立，措辞需改）**
- `information_bound.py:39-49,296-306` 假设 iid；`temporal_noise.py:1-13` 明确 AR(p) 相关；BNCI 100 ms 闪光在 800 ms epoch 内高度重叠（BI2014a 57,446 次 flash 在 1200 ms 窗内平均还有 2.22 个后续 onset）。正相关下证据不能按 sqrt(F) 累加。
- 代码自身已在 `:39-49` 声明“not a universal information-theoretic upper bound”，因此正确称呼是 **model estimate，不是界**；独立复现（Q=36,F=120,20k trials，真 d²=0.5）：iid MC 0.663，AR(1) ρ=0.3→0.460、ρ=0.5→0.340、ρ=0.8→0.211，即 iid 高估 0.20–0.45 绝对命中率。
- 建议：按拟合 AR 协方差对 MC 着色/块 bootstrap，同时报告 iid 与 AR 两版；报告删除“bound/最优”措辞。

**P1-M3 小 n bootstrap 区间严重欠覆盖（已证实，独立复核 50.47%）**
- `src/stats/hierarchical.py:47,62-71` 纯 percentile bootstrap，n=2 即允许；`src/stats/transfer.py:190-192` 在 `len>1` 就写 `paired_interval`。独立模拟：n=2 覆盖 **0.5047**（Wilson 95% [0.479, 0.530]；1500×1000），n=3 0.773、n=5 0.820；n=2 的解析覆盖率就是 50%（bootstrap 分布取 min/max 概率各 1/4）。
- 建议：n<5 直接输出 insufficient_units 或改用 bootstrap-t/BCa；把 `len>1` 门槛改为 n≥5；区间对象加 `method` 与 `min_units_policy`。

**P1-M4 exact_covariance 的“对数似然修正”由 ridge 常数支配（成立且更强，数字口径修正）**
- `exact_covariance.py:396-440` 对秩亏 `K_prop` 用 `ridge=max_eig*1e-12`（`:243`）做 Cholesky logdet。原报告 N=128/rank=72/C=4 的 +3241.1（det 项 +3315.6、quad −74.5；04 报告写 3241.3，差 0.2 为四舍五入）；但同配置下仅 det 项的 100× ridge 位移应为 `(N−rank)·C·ln100≈1031` nats，不是 2500（2500 需额外近零特征值的 quad 贡献，不能作为稳定量）。
- 更严重：若残差含零空间分量，quad 项被 1/ridge 放大，独立复现中 `delta_ll` 可达 **−2.16e12**（N=307 实验），ridge 1e-12→1e-10 移动 2.1e12。`byte_budget` 的 N≤2048 成立（64 MiB/两个 N×N 矩阵），docstring 的 ~2900 漏了系数 2。
- 建议：range(A) 子空间伪行列式/秩修正，或 ridge 敏感性曲线必报；docstring 注明非可比较边际似然；拒绝含零空间残差的该诊断。

### 3.5 项目文件管理

**P1-R1 工作区存储与 junction（已证实）**
- 总 apparent 154.9 GB：mne_data 89.6 GB、experiments/runs 41.0 GB（26.93 GB 为原始数据交付 `tar.gz`；5.4 GB npz、2.3 GB pt、1.5 GB JSON、8134 个 `.py`（1174 种内容，冗余副本约 6960、其中 3305 同源）、614 个 `.pyc`；4398 目录、122 个 >20 MB 文件）、.git 8.0 GB、venv 4.15 GB、frozen 自有 3.25 GB、formal 2.5 GB、cloud_inputs 2.0 GB。
- `frozen` apparent 95.4 GiB/102.5 GB = 自有 3.25 GiB + 3 个 junction：`BrainSync_Tool_newest`→`D:/BCI/BrainSync_Tool_newest`（1.58 GB）、`checkout/.venv`→仓库 `.venv`（0.98 GB）、`checkout/mne_data`→仓库 `mne_data`（89.56 GB）。这解释了 apparent ~94 GB，也意味着该“冻结 checkout”并非不可变（依赖活动 .venv/mne_data）；递归工具/备份会重复计算 90 GB。
- 建议：定义四层存储（Git 文本/本地原始数据/派生缓存/证据归档）与配额；runs 加 manifest+保留期；原始 tar.gz 移出 runs；`.pyc` 排除；定期磁盘报表；把 junction 改为显式外部依赖记录或删除。

**P1-R2 .git 对象库 8.04 GB（已证实；gc 前必须先救历史）**
- pack 205 MB、loose 7.26 GiB（11302 个）；`--no-reflogs` 下 2144 个 unreachable 对象（112 commit+431 tree+1601 blob；默认口径 1812）；前 4 大对象为 ZIP（约 3.0 GB，不可达、可回收），第 5 大 354 MB 的 `source-only.tar.gz` 已提交（`frozen/...`，不可 gc 回收）；来源是 2026-08-31 在 `.gitignore:28-32` 生效前被 force-add 后 reset 的大文件。
- 建议：**先执行 P0-1 的 branch rescue + bundle**；再检查是否需要保留 unreachable 对象（`git fsck --lost-found`）；确认后 `git gc --prune=now`；长期用 LFS/不入 Git + pre-commit 单文件大小限制。

**P1-R3 文档/证据索引与链接系统性断链（已证实）**
- 约 31 条相对链接断链/17 个文件（V1 复核；10 分区计 32/18）；约 28 个唯一目标中 **25 个从未在 HEAD**（只有 3 个是工作树删除），与 AGENTS.md:45-48、`frozen/README.md:3-8` 的“manifest 入库”承诺冲突。
- 111 个 evidence 中 8 个未被索引（`algorithm_chain_repairs_20260916`、`brainsync_transport_unification_20260917`、`code_quality_audit_20260908`、`e2_prior_scale_20261006`、`source_modules_20260918`、`unified_budget_execution_20260916`、`unified_compact_training_20260916`、`unified_model_implementation_20260916`）；59 个 `doc/assets` 目录中 **9 个**未被引用（含 60.4 MB/1149 文件的 `lean_current_checks_20261018`、空目录 `operator_research_202604`）。
- 2026-10-16 清理删除了 20,936 文件/1.43 GiB，连带 8 个 frozen 快照目录+归档消失；9 个被删 manifest 中仅 6 个被文档引用、3 个零引用；全部可从 HEAD 取回，但其 zip 从未入库。
- 建议：加 `experiments/check_docs.py`（链接存在性+evidence/assets 索引覆盖）并纳入 CI；恢复 9 个被删 manifest 或同步删除引用；清理空目录/重复 docx。

**P1-R4 已提交证据泄露机器身份（P1 隐私）**
- `doc/assets/code_quality_repair_20260908/validation.json:29` 含 Windows 账户名+**2 处** `C:/Users/...` 临时路径；`doc/assets/technical_audit_20260906/*.xml` **8/8** 个 JUnit 文件与部分 `frozen/*/manifest.json` 含 hostname；均在 Git 追踪/历史中（无密钥、无被试 PII）。
- 建议：新文件去除机器名/路径；历史是否改写取决于发布范围；在导出证据的脚本中统一脱敏（账户名、hostname、临时路径）。
- 另：`doc/assets` 有 3352 个未跟踪产物（172 MB，含 `.bdf/.pt/.log/.zip` 等），`.gitignore` 只忽略 `*.npz` → `git add -A` 会全部入库（`*.npz` 本身已被正确忽略，77 个 npz 0 个未跟踪；无“已跟踪文件被误 ignore”）。

**P1-R5 Paper PDF 入库与许可（已证实）**
- `Paper/` 54 个 PDF（Paper 下 59 个已跟踪文件）、191.4 MB 入 Git；根目录无 LICENSE（唯一许可是 `src/baselines/structured_erp.LICENSE`，且其所在目录三个文件未跟踪）；数据集许可（BI2014a/BNCI2014-008/GTN/Mainsah 等）未在文档/代码系统声明。
- 建议：Paper 移出 Git/LFS + 文献索引（标题/DOI/许可）；补根 LICENSE/NOTICE 与“第三方代码与许可”小节；明确数据集许可与再分发边界。

### 3.6 安全

**P1-S1 模型/结果加载使用 `torch.load(weights_only=False)`（已证实；若产物来自不可信来源应升为 P0）**
- 热路径：`src/transfer/checkpoint.py:56`（20+ 入口）、`src/transfer/result_reuse.py:116`（验证 sha 后仍全量 pickle）；`src/baselines/deep.py:587`（死代码/P2）。对照 `src/transfer/calibrated.py:165-169` 已用 `weights_only=True + safe_globals`。
- 影响：来自共享 worker/下载目录/被替换的 checkpoint 可执行任意代码；sha 只防篡改不防“自洽恶意产物”。
- 建议：默认 `weights_only=True` + 类型白名单；旧格式需显式 `trusted=True` 并在 CLI/文档标注。

**P1-S2 云主机/用户/私钥名硬编码 + 关闭主机校验（已证实）**
- `open_dashboard.ps1:2-8` 硬编码 `connect.weste.seetacloud.com:35832`、user `root`、`/root/autodl-tmp/n2p3-net`、`~/.ssh/id_ed25519_n2p3_cloud_20260827`；`:46,98` 两处 `StrictHostKeyChecking=no`；`:43` 远端命令为字符串拼接（root 命令注入模板）。
- 建议：参数默认值改占位符/环境变量；启用 host key 校验（首次 pin known_hosts）；脚本移出仓库或加 `.example` 模板。

### 3.7 性能

**P1-P1 E2 评分进程池反复重建 + 整份 pickle 数据集（已证实）**
- `experiments/run_source_training_chain.py:405-406` 每个 `(seed×axis)` 组合新建进程池，`initargs=(dataset,)` 把整份 EpochDataset 序列化给每个 worker；4 源组×4 子 worker=16 进程争 4 个 GPU 槽，12 个空转（`src/research/source_execution.py:219,67-75`）。
- 建议：按任务组复用池/共享只读矩阵；限制子 worker=GPU 槽；记录实际并发。

**P1-P2 报告 JSON O(n²) 写放大（已证实，复核量化）**
- 每 5 s 写一次（无进展也写）+ 每任务/每选择后整份重写、`indent=2` + fsync（`run_source_training_chain.py:511-523,783,791,912,941`；`src/research/execution.py:42-53`）。
- 复核：最大单文件 `runs/unified_cloud_20260916/analysis/matrix_sources_source_chain_record.json` = 180,221,713 B（171.87 MiB），35 tasks；该 run 至少 36 次全量写，按最终大小一半估算累计下界约 **3.1 GiB**（5 s tick 另计），相对最终文件约 18× 下界。
- 建议：增量 JSONL/分片 task 记录 + 原子替换小索引；只在状态变化时写；避免每 tick fsync。

**P1-P3 每步梯度 isfinite 扫描 + 每任务重置编译/分配器（部分成立，口径修正）**
- `src/baselines/deep.py:1318-1321` 每步对参数梯度做 isfinite+&；默认 EEGNet 是 **12 个参数张量/1,378 参数**（inception 38/14,890；conformer 110/202,306），不是 22；独立实测扫描约 **0.35–0.7%**（batch512 step≈51.7 ms/扫描 181 µs；batch256 step≈26 ms），不是 4.8%。
- `src/research/source_execution.py:78-85` 每任务 `torch.compiler.reset()+empty_cache`（调用点 376/398/1032/1346）成立；默认 `compile_mode='reduce-overhead'`，reset 会导致下一任务重编译，这项开销比 isfinite 扫描更值得关注。
- 建议：降低扫描频率（每 N 步/阶段末）；编译缓存跨任务复用；给出真实模型张量数/step 时间后再报百分比。

**P1-P4 资源下限合同覆盖不足（已证实）**
- `cloud_resource_floor` 只覆盖 1/6 入口；`configs/cloud_training_resources.json` 未入库且无 `src/experiments` 代码读取（仅测试读取）→ 漏写即静默跳过；`available_cpu_threads()` 在 Windows 返回 32 且不读 Job Object，`run_source_training_chain.py:605` 会开 32 个 target worker。
- 建议：所有入口统一解析并 fail-closed；Windows 用 Job Object/显式 `--cpu-threads`。

**P1-P5 当前所有 H2D 都是同步拷贝（部分成立，机制修正）**
- 独立复核：`p.pin_memory().is_pinned()`=True，`p[:10]` view=True，但 `p.index_select(0, idx).is_pinned()`=**False**；`to_device` 的 `non_blocking` 对这些 batch 为 False。
- 但主因更强：`MatrixBatchSource.__init__` 先 `to_device(self.cpu_X)`（`src/train/runtime.py:477`）后 `pin_memory`（`src/train/runtime.py:508-517`），而调用方传入 `torch.from_numpy(...)`（`deep.py:1275,1297,1937`）默认未 pinned → **preload 路径从不 pin、H2D 同步**；只有 preload 失败进入流式路径才 pin 整矩阵，且随机 `index_select` 又丢失 pinned。“每进程 423 MB”不是普遍值：代码无此常量，抽查的真实 X 为 (162,757×6×179) float32 = 666.8 MiB；`release_cpu_after_preload=True` 时 preload 成功后还会释放 CPU 矩阵。
- 建议：preload 路径显式 pin 并在 `to_device` 后用 pinned batch 拷贝；或删除无收益 pin；在 5090 上以端到端 step 时间复测。

### 3.8 补充分区新增 P1（第 2 轮并行审计，G1–G8）

**数据元数据（G1）**
- P1 `manifest.channels` 字符串被逐字符拆成通道：`manifest.py:179` 对 "Fz" 得 ('F','z') 且 load/validate 通过 → 通道合同输入类型未校验。
- P1 多会话事件台账不可拼接：event_ids=`f"{dataset_id}:{index}"` 不含 session/run（`events.py:585` vs `321`），同一 dataset_id 多录制连接时报 ValueError。
- P1 `identity` 的 `source_or_global` 证据完备性不对称（`identity.py:386-402`）：只登记 origin 时可静默放行跨研究同一被试。
- P1 混合坐标注册把 fiducial+ICP 压成恒等 transform 写入 provenance、`icp_iterations` 归零（`channel.py:481-489`）——审计证据失真。
- P1 年龄正弦编码实际 `π·2^k`（最高频周期≈1.6 岁），与通道 `geomspace(0.5,8)` 不同；`metadata.py:11-14` 的“复用同一编码”声明不成立。

**迁移/训练辅助（G2）**
- P1 AS-MMD 源批 `torch.randperm` 不受 seed 控制、无账本（`subject_adapter.py:1639`）；`as_mmd_lambda0>0` 的 arm 不可复现（目标批有 `batch_rows_sha256`）。
- P1 预训练谱项在零填充掩码上做 FFT（`src/transfer/losses.py:136-152`）：常数偏置 1e-7 时谱项 1.6e-12 vs 波形 1e-7，同误差谱/波形≈3.5×，默认 1:1 权重未声明尺度差。
- P1 质量报告无条件写 `fit_performed=False`，但 callable `epoch_quality` 可能就地拟合（`research/recording_quality.py:158-170,236,266`）→ provenance 失真。
- P2 candidate prefix/suffix 只按重复序号切分、无时间禁运（`src/transfer/candidate_within_subject.py:261-322`）；`hit_at_repetition` 用全局起点归一化，组起点不同时 level-2 命中率 1.0 vs 正确 0.5（`transfer/evaluation.py:52-63`）。
- P2 两套决策打分核心（与“统一核心”声明冲突）、谱权不入 `state_dict`、mask probe 的 `state_sha256_after` 是复制值、经典基线超参不入 record。

**配置（G5）**
- P1 workstation manifest 唯一 option 不传 `--preprocessing-config`，实际回退硬编码 `BRAIN_SYNC_PREPROCESSING`（`configs/brainsync_workstation.model.json:13`、`experiments/prepare_brainsync_cache.py:71`）→ 改配置不会改变实际预处理。
- P1 collection plan 两处 `rest_sec=0` 与采集端默认 20 s/block 冲突：10 block 少记 180 s（计划 6075 s vs 采集端 6258 s）。
- P1 preprocessing/epoch_support/workstation manifest 三份几何副本无交叉校验（仅 epoch_support 有测试 `tests/brainsync/test_acquisition_contract.py:13-21`）。
- P2 candidate 示例 7 个输入全缺失，dry-run 直接 FileNotFoundError；cloud 配置隐含 ≥86.32 GiB 主机内存/≥20 GiB 显存未文档化；无 runner 直接读该 JSON。

**代码结构（G6）**
- P1 反向依赖：`data→transfer` 7 条 deferred 边（`data/source_records.py:46,56,343,551,653,804`、`data/study.py:230`），靠函数内延迟导入隐藏；包级 7 包构成单一 SCC，eager 模块环 0。
- P2 复制不是主要债务（跨文件函数克隆 0 组；块级 221 行/0.21%）；真死 7 函数/141 行；88 个“公开但仅自用”符号 1,856 行；`src/measurement`、`src/audit_p300` 为空包；`deep.py` 2,686 SLOC、`_fit_attempt` 1,328 行。
- P1 重构优先级：把 `transfer.binary` 标签构造下沉到 data（消除 7 条反向边）；拆分 `_fit_attempt`（`tests/models/test_deep.py:18` 可作特征化基线）。

**CLI（G7）**
- P1 `--output` 二义：22 入口当目录、31 当文件；54/55 无 help；`save_epoch_dataset` 原子写但不拒绝已存在（`src/data/epochs.py:1037-1039`），7 个缓存脚本静默覆盖；17 个调用点 9 个无存在性检查；22 目录入口 16 `exist_ok=False`/9 True；全仓 0 个 `--force`、仅 1 个 `--resume`。
- P1 `--subjects` 四种互不兼容语法（int 人数、nargs 列表、逗号串、字面量 all）。
- P1 `retire_bi2014a_legacy_derivatives.py` 无 `__main__`、无 dry-run，模块层 parse_args 后直接 unlink（:9-13,43），且被 formal 计划与 frozen manifest 哈希固定。
- P2 `--device` 默认 cuda5/cpu4/auto7、`--cpu-threads` 4/32/2/None、10 种 worker 名；均不提示 AGENTS 的 ≥25 线程下限；退出码 2 被用法/已存在/复核失败共用；3 套 stdout 协议、0 logging、508/683 参数无 help、无 `--version`；67/67 `--help` 实测 rc=0（正面）。

**运行产物 lineage（G8）**
- P1 执行身份覆盖不足：52 个 run 中仅 9 个（17%）同时有源码快照字节+哈希声明；其余为仅源码字节/仅哈希声明/仅结果/两者都无（分层计数口径见 G8 报告；抽样 20 的分布为 8/3/7/1/1）；真正模型重放仅 3/20，失败/分母账本 18/52，包版本 16/52。
- P1 唯一可完整重算的链 `training_chain_auto` `execution.json:8-14` 与本地 manifest/archive 哈希一致（235 成员/1,059,425 B），但所有 commit 都标 `base_commit_only`，不能定位执行字节。
- P1 哈希悬空：`archive_sha256` 57/79 可本机解析、`manifest_sha256` 35/73；`system_*` 夹具写入 cccc…/dddd… 占位哈希且无 synthetic 标记。
- P1 路径/机器标识：39 个结果 JSON 中 28 个内嵌绝对路径；23 个 run 出现 autodl；`completion.json:17` 写 hostname；`source_eog` lineage 含 `D:\` 与私钥路径（仅路径）。
- 建议：18 字段 `run.json` 规范 + 6 门只读校验器 + R0/R1/R2 保留策略 + 内容寻址 store（≤8 MB 归档 551 文件仅 289 个不同内容，可去重 47.6%）。

**BrainSync 工作站后端（G3）**
- P1 `fixed_erp` 在 `brainsync_model_task` 中不可达：`--calibration-recipe` 只允许 unified 使用（`brainsync_model_task.py:250-269`），而 `fixed_erp` 必需该 recipe（`calibration_candidates.py:51-54`），传 recipe 实测 parser exit 2；但 `doc/workstation.zh.md:37-39` 声称该模式可保存/重载个人包。
- P1 文档化的 unified 入口依赖两个未跟踪配置（`unified_training_budget.json`、`brainsync_bidirectional_preprocessing.json`）；后者 SHA-256 正是 frozen manifests 引用的 `deffab5f…`，HEAD 克隆不可复现且无测试钉住。
- P1 `run_brainsync_cross_decision.py:705-706` 直接覆盖 `--output`，无新文件/路径身份检查；路径重合时可能把 `--dataset-cache`/`--checkpoint` 覆盖成 JSON（毁输入）。
- P1 参考更正缺后置校验：`correct_brainsync_reference.py:77-88` 只改值等于旧标签的 reference 行，无后置 validate；该可选列缺失或为混合值时脚本仍 exit 0，下一次加载才被 `src/data/brainsync_contract.py:169-172` 拒绝（测试夹具恰好缺该列而绕过）。
- P2 `prepare_brainsync_cache` 默认静默覆盖 npz+record.json（`data/epochs.py:1037-1040`）；quality report 对合同/参数错误一律 exit 0（`recording_quality.py:317-326`）；独立 epoch-QC 文件无 SHA 绑定；连续 QC manifest 的 `preprocessing` 绕过完整配置校验（`fit_brainsync_continuous_qc.py:36-38`）。
- 正面：无标签 `infer` 只写 `summary.json`；模型保存为临时文件+硬链接独占，加载校验 metadata/state digest 且 `weights_only=True`；推理 artifact 先哈希再从 `BytesIO` 加载（避免 TOCTOU）。

> 补充：G3b 二次审计（145 行）新增 1 条 P1（参考更正缺后置校验）与 7 条 P2（EDF 核对在回滚事务外、45 闪烁/5 重复协议未在后端强制、缓存固定 .tmp.npz 并发互踩、cross-decision 缺 ingress/skipped、连续 QC preprocessing 可回退默认、correction --output 可覆盖既有 manifest、EOG preprocessing 未接 CLI），并交叉复核 G3 的 13 条结论未发现误报。

**证据可回溯性（G4，正面为主）**
- 正面：BNCI 0.4051724/0.5905172、28/29、464→448、16 模型/448 decision 重放、34 封印文件/35 源码快照 SHA 全一致、0 拟合失败；P1/P2 45 源/16,599 拟合/32,238 cells/0 失败/288 重放/29,538 模型；compact 3126.7849 s/6,138 NN/246 LDA/4,416 复用/12,150 cells/10,800 模型。均可从本地产物回溯。
- P1 口径转述错误：README/doc 把“448 个 decision 重放”写成“448 个保存模型”（实际 16 个模型）；把 unified12 的“48/48 分片”写成“48 个模型”（实际 48 分片/3,632 模型/4,082 cells）；0.7057 不属于 unified12（其 K200 A0Q0=0.6995），实际来自 `training_chain_auto` primary_metrics 的 full_fine=0.7057075（scratch 0.6563828）。
- P1 unified12 重放容差 `rtol=atol=1e-4`、最大差 3.66e-4 未写入文档；11.17 GiB 归档仅云端收据、本地无副本。
- P2 BNCI `saved_reload_scores_exact` 只校验 nll 非空；README 16,599/32,238 未分解。

## 4. P2 — 中优先级

### 4.1 代码结构
- 约 22 个函数 >200 行（Lead AST 口径；G6 按 >150 行另计 28 个/829 个顶层函数）：最长 `src/baselines/deep.py:1438 _fit_attempt` 1328 行；`src/transfer/subject_adapter.py:756 fit` 632 行；`src/data/brainsync.py:176` 456 行；超长模块 deep.py 2830 行、moabb.py 2018、subject_adapter 1827、epochs 1795、evaluate 1601、raw_artifacts 1598、decision_loss 1403。建议按“校验/装配/落盘/拟合”拆分。
- 21 处 `except ...: pass`（多为 OSError，可接受）与 10 处 `except Exception`；`src/data/gtn_dataset.py:199-213` 宽泛捕获 + `allow_skipped` 会把代码缺陷记成源缺陷、静默缩小队列。
- 候选/日程规则三处实现（`bi2014a_candidate.py`、`bnci2014_008_candidate.py`、`candidate_task.py`）；建议抽共享模块+性质测试。
- 部分数学/诊断模块（`exact_covariance/information_bound` 等）零生产调用且未入库；建议在 README/技术结构标注 `diagnostic/no production import` 或移到 `experiments/`。

### 4.2 测试
- `addopts="-q"` 吞最终统计行；无 `-ra`；无 coverage；`tests/__init__.py` 的 `PROJECT_ROOT` 使测试依赖“仓库根在 sys.path”。
- 断言：932 处 `pytest.raises` 中 100 处无 `match=`；11 个用例无任何断言；74 个用例断言全藏 helper；`array_equal` 287 次 vs `allclose` 138 次。
- 30 个测试文件模块级 `import mne` 且 `importorskip`=0；GTN 92 GB 数据、同级 BrainSync 前端、node、CUDA 决定实际测试集；skip 原因不可见。
- 建议：显式 markers + `-ra` + timeout；快速契约/全量硬件两层；断言风格统一（错误契约加 match、数值默认 allclose）。

### 4.3 数学/统计（其余）
- `relative_error_trace` 对带通恒 ≈1，不是“AR 近似误差”（`exact_covariance.py:150-153,221-222`）；建议改用白化增益谱/条件数。
- `d2_kronecker_approx` 随物理单位缩放（数据 ×1000 → 1e-6），不是 Mahalanobis 距离（`src/transfer/information_bound.py:402-435`）；`efficiency` 文档定义与代码相反且未 clamp（`:489,535-546`）；`byte_budget` 在 `src/transfer/exact_covariance.py:185-198`（:187 docstring ~2900、:195 的 2*8*N²），容量算错（N≤2048 而非 2900）且峰值低估 3–4 倍。
- `noise_model` 3 draws 估 2p 维协方差无下限；`operator_lda` 两套正则判据（1e-10 与 ridge=1e-3·mean(λ)）不一致；`_conditional_mmd` 逐 cell 宏平均+无带宽下限（分区 04）。
- 校准切分：缺 per-recording availability 时跳过时序泄漏检查（`src/transfer/calibration_split.py:104-108,169-182`）；候选拟合缓存 key 缺 `block_id/selection_repetitions`（`calibration_candidates.py:395-415`）。
- 重采样抗混叠仅 −22 dB@70 Hz；被丢弃的 1200 ms 端点样本以 −60 dB 漏入基线窗；EOG 在新/legacy 路径滤波行为不同（分区 07）。

### 4.4 工程与文档
- `ruff format --check` 280 文件待格式化；建议独立提 `ruff format` + CI。
- `pyproject` 无 readme/license/authors/classifiers/urls/scripts；根目录无 LICENSE；`structured_erp.LICENSE` 未跟踪但被声明为 package-data。
- `.claude/settings.local.json` 设 `defaultMode=bypassPermissions` 且仅靠用户全局 ignore；备份文件未被忽略；`artifacts/` 无 ignore 规则。
- `open_dashboard.ps1` 未在 README/文档索引出现；`requirements-minimal.txt` 与 pyproject 边界不清。
- 文档：`doc/engineering.zh.md` 首行“更新：2026-09-07”但含 10-18 内容；`tests/run_headless.py` 等属于独立采集端项目；历史测试计数（1539/1936/2095/1404）与当前 2331 passed/3 failed 不一致；`doc/README.md` 173 行中约 142 行为状态流水（违反自身第 164 行规则）；发布体积存在 128.319 MiB/119,627,193 B/6.847 MiB/121.515 MiB 四套互斥数字；README 入口表实际 17 个脚本（+dashboard=18）全部存在。
- Lean 工程：`.lake` 2.67 GB/30887 文件未跟踪且被 ignore 覆盖（合理）；但 artifacts 被 ignore 而 evidence 引用；39 个 Audit 中 27 个无入口；工具链本机硬编码。

### 4.5 项目文件管理（其余）
- `frozen` 100 个目录中 **87 个有 manifest**（顶层 `manifest.json` 仅 53 个；13 个目录无；按任意层 *manifest*.json 计 89，口径需在建档规范中统一）；9 个被删 manifest 中 6 个被引用、可从 HEAD 恢复。
- `frozen/worktree_recovery_20261016.json` 记录过去清理事故（“cleanup stopped; affected author trees restored”，含丢弃文件清单）；建议把 worktree 生命周期写成固定清单并自动检查独有提交/未提交源码。
- 命名：`operator_research_202604` 空目录疑似日期笔误；`doc/evidence/lean_outcomes_20261018.zh.md` 等使用绝对 `D:/BCI/...` 链接，不可移植；`doc/progress_report_20260929*.docx` 与 md 重复且未跟踪。
- `doc/assets` 552 个 zip 与 `frozen` 376 个归档零同名重叠（无重复存储证据），但仍建议统一“证据资产索引”。

## 5. 正面发现（建议保持）

- **合同与准入 fail-closed**：缓存 attested 加载、`raw_artifacts`（size/SHA/MD5+inode 前后比对）、`source_records` 台账、预处理执行回执、结果复用完整性校验均多重校验；错误路径明确终止。
- **数学实现质量总体高**：EA 的 `R^{-1/2}`、SupCon 多正样本式、无偏 MMD²、`F K F^T` 与稠密 `A K A^T` 一致到 2.9e-14、baseline 半开窗口、50 Hz/Q30 陷波 −3 dB 宽 1.6666 Hz、250→128 Hz 的 351→180→179 几何、SplitBN 多 batch 合并 ≤1e-9、observed-channel 与普通路径 max|diff|=0。
- **工程卫生**：`ruff check` 全过（399 文件）；`uv lock --check` 通过；`src` 无裸 except、无可变默认参数；23/23 未跟踪 src 模块均被测试引用；数值比较常用 allclose。
- **资源合同设计**：`configs/cloud_training_resources.json` 与 `src/research/source_execution.py:119-221` 的 25 CPU/4 GPU floor、cgroup/affinity 解析、GPU 设备锁、AMP/BF16、内存快照均正确（配置未入库是流程问题）。
- **Lean 方法学**：lean-toolchain/Mathlib commit 固定；无 `sorry/admit/axiom` 作为证明（仅注释）；公理白名单（propext/Classical.choice/Quot.sound）+ `negative/` 反例强制失败 + `verification_plan.json` 源码哈希绑定；回执→清单→证明→工具四层哈希实测匹配。
- **证据可回溯（G4）**：BNCI/P1/P2/compact 四组主结果的模型数、拟合数、评分单元与本地归档 SHA 均可回溯（45 源/16,599 拟合/32,238 cells/0 失败；compact 3126.7849 s/10,800 模型）；G7 的 67/67 `--help` 全部 rc=0；3 个失败测试集中在单一未跟踪测试文件。
- **安全基础**：未发现硬编码密钥/凭据/被试 PII（但已提交证据含机器名/路径，见 P1-R4）；`np.load(allow_pickle=False)` 全覆盖；`CalibratedPredictor` 路径已用 `weights_only=True+safe_globals`；dashboard 路径穿越与 Content-Length 校验正确；README 入口表全部存在。

## 6. 分阶段改进路线

### 第 0 阶段（立即，1–3 天）
1. **救历史**：对 30 个无保护 commit 批量建 `rescue/*` ref → `git bundle create backup-20261019.bundle --all` → 异机保存；再核对 66 个分支（`e148163` 受 reflog 保护至约 2026-11-13）。
2. **物理快照当前工作树**（含未跟踪源码/配置/formal），生成 SHA-256 清单。
3. 修复 3 个 pytest fixture（或回退 LDA 校验并记 TODO），恢复全绿。
4. EOL 归一化独立提交：`.gitattributes` + `git add --renormalize` + 自哈希改为 LF 规范化。
5. 拆分提交：tests 迁移 → src 新模块 → configs → formal → doc/assets/evidence → frozen manifest 修复（每步 ruff+pytest）。
6. 备份后清理 .git（`git gc`）；评估/移除 3 个 junction。
7. 安全：`weights_only=False` 默认收紧；dashboard 加 token 与 kill 二次确认；`open_dashboard.ps1` 开主机校验/去硬编码；`.gitignore` 覆盖 `.claude/`、`artifacts/` 与 doc/assets 二进制产物。
8. 文档：恢复/更新 9 个 frozen manifest 与 32 条死链；补 8 个未索引 evidence。

### 第 1 阶段（2–6 周）
1. CI：ruff + pytest（快速/全量分层）+ `git diff --check` + `uv lock --check` + 文档链接检查 + Lean `lake build`（缓存）。
2. 测试迁移映射表（130 个消失函数逐条结论）+ 覆盖率报告；markers/timeout/-ra。
3. 数学修复：d² 交叉拟合/偏差修正、AR 着色 MC、BCa/bootstrap-t + n<5 拒绝、logdet 秩修正/敏感性、统计报告统一口径。
4. 数据合同：MOABB 执行回执、BIDS 内容哈希、epoch 网格 origin、零相位 policy 身份化、attested 加载内容寻址。
5. 迁移/模型：transfer_only 回退硬失败、公开池整库排除门、checkpoint 权重摘要、selector 预算修正、BrainSync QC policy 绑定。
6. 打包：命名空间或白名单、wheel 自包含、`[tool.uv]` 版本/索引固定、GPU extra。
7. 文档治理：六份正文去编年史、状态迁 evidence/CHANGELOG、`last_verified_commit`。

### 第 2 阶段（2–3 个月）
1. 结构重构：拆分 1300/632 行函数与 2800 行模块，候选规则统一。
2. 性能：E2 进程池/数据共享、JSON 增量写、isfinite 降频、编译缓存、pin_memory 复测、5090 benchmark 回归阈值。
3. 仓库瘦身：Paper 移出 Git/LFS、runs 保留策略与 manifest、doc/assets 去重、工作区存储配额。
4. Lean：formal 入库 + CI 证据链（KB 级 receipt 可提交）。
5. 安全/合规：根 LICENSE/NOTICE、数据集许可清单、secret/机器标识扫描 pre-commit。

### 统一验收命令
```powershell
.\.venv\Scripts\python.exe -m ruff check src experiments tests
.\.venv\Scripts\python.exe -m pytest            # 0 failed
uv lock --check
git diff --check
git status --porcelain                          # 仅明确忽略项
git ls-files --eol | Select-String 'w/crlf|w/mixed'   # 归一化后无非预期项
```

## 7. 审计边界与未验证项

- 未运行训练、云任务或真实数据集实验；未连接 BrainSync 采集端（独立项目）。验证环境为 Python 3.14.3 / torch 2.13.0+cpu（`.venv`），未在 3.11/3.12 复跑。
- CUDA/5090 实际吞吐、显存与并发未在本机复测（本机为 torch 2.13.0+cpu）；性能结论以静态代码+历史文档+合成基准为主。
- 3 个失败测试的根因已定位到 fixture 与新增校验的维度不一致；未修改代码验证“修复后全绿”。
- BIDS/BrainSync 内容哈希缺失标为“疑似”：采集端可能另行记录，需跨项目确认。
- 数学数值验证限于小规模合成数据；真实 T×C/epoch 规模下 d² 偏差可能更大。
- 文档链接/索引/大小/定理数为审计时点快照；doc/assets 与 runs 的具体保留价值需项目所有者裁定。
- 补充分区 G1–G8 于 2026-10-19 08:35–08:53 并行完成；G4 的数字口径修正（448/48 等）与 G8 的 lineage 覆盖数据同样属于审计时点快照。
- 本报告的若干早期判断已被独立复核修正：HEAD 本身不 import 未跟踪模块（P0-2 已改为“工作树不可由 Git 重建”）；94 GB junction 主要来自 checkout/mne_data（89.6 GB）而非 BrainSync 应用；frozen manifest 为 87/100；README 入口为 18 项；.gz 为 26.93 GB；`doc/assets/**/*.npz` 已正确忽略且无误 ignore 的 tracked 文件。
- V1 复核修正：import 边为 27 条/14 个文件/11 个未跟踪目标（HEAD 0）；自哈希 8 个而非 11 个，真正跨平台闸门是 `prepare_source_records.py:117`；未跟踪 231、总 429；Paper 54 个 PDF（191.4 MB）；frozen 87/100（顶层 53、13 个无；任意层 *manifest*.json 计 89）；断链约 31/17；被删 manifest 仅 6/9 被引用；.gz 路径为 `experiments/runs`；uv 0.11.26（.venv）/0.12.9；ruff format 280；无入链 assets 9 个；runs 内 py 副本口径改为 1174 种内容/6960 冗余/3305 同源。
- V2 复核修正 6 处数学/性能口径：d² 的 n 是 per-class（总 n=60 时约 28.6×，34.6× 为种子相关值，交叉拟合不能消偏）；iid MC 是 model estimate 而非界（ρ=0.5 时 0.663→0.340）；n=2 bootstrap 覆盖率 50.47%，n=3/5 为 77%/82%；logdet 的 100× ridge det-only 位移约 1031 nats 而非 2500，含零空间残差可达 −1e12；pin_memory 的 preload 路径根本没 pin、423 MB 非普遍值；isfinite 扫描为 12 张量/1378 参数、CPU 0.35–0.7%，非 22 张量/+4.8%。
- V3 复核修正：2144 unreachable 对象中仅 30 个 commit 无 ref/reflog 保护（最新 451af9d，立即风险），82 个由 reflog 保护；frozen apparent 为 95.4 GiB（自有 3.25 GiB）；formal Git 可见 414 文件/3.63 MB 只是子集（磁盘 32,699 文件/2.53 GiB、8,419 .lean）；2890 定理来自 active_checkpoint 自报、源码约 3,418 条 theorem、无映射；Lean artifacts 引用为 1530 条/754 断链（去重路径口径；按出现次数 2448/1128）。red-team 复核报告见 `doc/assets/repository_audit_20261019/V1-verification.md`、`V2-verification.md`、`V3-verification.md`。

## 8. 附录

### 8.1 分区报告索引

报告目录：[doc/assets/repository_audit_20261019/](../assets/repository_audit_20261019/)（下表中的文件均可用该链接访问）。

| 分区 | 报告 | 主要结论 |
|---|---|---|
| 00 | 00-consolidated.md | 审计汇总底稿（与 doc/evidence 定稿同源；过程版本） |
| 01 | 01-data-core.md | 缓存放大、MOABB 声明即证据、BIDS 无哈希、epoch 网格偏移 |
| 02 | 02-models-train.md | 显存依赖轨迹、SplitBN EMA、adapter 状态、PDS 预算 |
| 03 | 03-transfer-core.md | transfer_only 回退、整库排除门、checkpoint 权重摘要、fit_calls、QC 绕过 |
| 04 | 04-transfer-math-stats.md | d² 上偏、iid/AR 冲突、bootstrap 覆盖、logdet ridge、单位缩放 |
| 05 | 05-research-experiments.md | 未入库依赖、无法续跑、执行身份不一致、非原子写 |
| 06 | 06-tests.md | 重组不可追溯、无 CI/覆盖/超时、环境耦合、断言质量 |
| 07 | 07-math-contract.md | 零相位 padlen、时间轴半采样、切分预热、EOG/重采样 |
| 08 | 08-engineering-tooling.md | EOL+自哈希、打包、环境不可复现、无 CI、format |
| 09 | 09-repo-management.md | main 停滞/dangling、git/存储/归档/索引治理 |
| 10 | 10-docs-consistency.md | 索引/链接/状态编年史/文档-代码一致性 |
| 11 | 11-security-privacy.md | dashboard 鉴权、机器身份、许可、weights_only |
| 12 | 12-performance.md | 进程池/pickle、JSON 写放大、isfinite、pin_memory |
| 13 | 13-formal-lean.md | formal 未跟踪、证据链断、工具链、孤儿证明 |
| G3 | G3-brainsync-workstation.md | fixed_erp 不可达、未跟踪配置依赖、--output 覆盖、QC 证据绑定 |
| G1 | G1-data-metadata.md | channels 字符串拆字符、多会话事件台账、identity 不对称、坐标注册失真 |
| G2 | G2-transfer-helpers.md | AS-MMD 随机不可复现、谱项掩码尺度、QC provenance 失真、候选时间禁运 |
| G4 | G4-evidence-repro.md | BNCI/P1/P2/compact 可回溯；448/48 口径转述错误；0.7057 归属错误 |
| G5 | G5-configs.md | 预处理配置未接线、rest_sec 冲突、几何副本无交叉校验、示例输入缺失 |
| G6 | G6-code-structure.md | data→transfer 反向依赖、7 包 SCC、死代码少、deep.py 超长 |
| G7 | G7-cli-consistency.md | --output 二义/静默覆盖、--subjects 四语法、破坏性入口无围栏 |
| G8 | G8-run-lineage.md | 仅 17% run 有完整源码身份、哈希悬空、重放/账本覆盖低 |
| G3b | G3b-brainsync-workstation.md | 参考更正缺后置校验、协议未后置强制、缓存并发（交叉复核 G3） |
| R1 | R1-report-review.md | 对最终报告的独立审阅（36 条抽查；M1–M11 修订清单，已完成 M1–M7） |

### 8.2 审计新增文件与源文件完整性

- 新增：本报告（`doc/evidence/repository_audit_20261019.zh.md`）、`doc/assets/repository_audit_20261019/`（分区报告 + red-team 报告）、`doc/evidence/README.md` 索引一行；`tmp/audit/` 为被 gitignore 的过程目录。
- 完整性：审计前后 `src/*/*.py`、`tests/**/*.py`、`experiments/*.py` 共 368 个文件 SHA-256 全部一致（`tmp/audit/source-hashes-baseline.txt`）；未修改任何源文件/测试/配置/指南。

### 8.3 复现命令

```powershell
# 全量测试与静态检查
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src experiments tests
uv lock --check

# Git 状态与风险
git log -1 --date=iso --format='%H %ad %s' main
git for-each-ref --sort=-committerdate --format='%(committerdate:iso8601) %(refname:short)'
git fsck --unreachable --no-reflogs | Select-String 'unreachable commit'
git count-objects -vH
git ls-files --eol | Group-Object { ($_ -split '\s+')[1] }

# 工作树/存储
git status --porcelain
(Get-ChildItem .git -Recurse -File -Force | Measure-Object Length -Sum).Sum/1GB
```

> 本报告为审计建议，不构成对任何科学结论的重新认证；修复应在独立分支按上述阶段与验收命令执行。
