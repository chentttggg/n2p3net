# 06 — tests/ 审计报告

> 审计对象：`tests/`（185 py = 159 个测试文件 + 26 个辅助/conftest/`__init__`）、根 `conftest.py`、`pyproject.toml` 的 pytest 配置。
> 基线：当前工作树（非 HEAD）。审计只读；未修改任何被测文件。标记：**已证实** / **疑似** / **无法验证**。

## 结论摘要

1. **P0｜重组不可追溯**：HEAD 1189 个测试函数中 131 个在工作树消失；函数体模糊匹配显示 **0 个纯改名**、119 个无近似替代。消失名单含路径逃逸、哈希不一致、预算下限、forced-switch 账本等安全/契约测试。
2. **P1｜收集健康但规模已超手工**：159 个文件、**2342 个用例**、0 收集错误、15.3 s；参数化 302 处把 1593 个函数放大到 2342。两个代表性子集 138 例全过、130 例 2 skip，exit=0。
3. **P1｜无任何 CI/覆盖率/超时/分层**：无 `.github/`、无 tox/gitlab-ci；dev extras 仅 pytest+ruff；无 marker 注册（全树只用 `parametrize` 302 次、`skipif` 3 次）。
4. **P1｜同一 commit 在不同机器跑的是不同测试集**：30 个文件模块级 `import mne` 且全树 `importorskip` = 0；GTN 92 GB 数据、同级 BrainSync 前端、node、CUDA 均以 skip 决定是否执行，而 `-q` 默认不显示 skip 原因。
5. **P2｜`addopts = "-q"` 在 pytest 9 下吞掉最终统计行**：实测带 `-q` 无 "N passed / M skipped"，CI 日志只剩进度点与 exit code。
6. **P2｜错误契约偏弱、跨实现比较过严**：932 处 `pytest.raises` 有 100 处无 `match=`；11 个测试无任何断言记号；74 个测试的断言全部藏在 helper；287 次 `assert_array_equal` 对 138 次 `assert_allclose`。
7. **P2｜conftest 全局副作用**：import 时把 BLAS/OMP/MKL 线程固定为 1 并在仓库根 `mkdir .mne_home`（只读工作区会在 collection 阶段崩溃，且单线程掩盖并发缺陷）。
8. **P2｜测试包化隐性契约**：`tests/__init__.py` 暴露 `PROJECT_ROOT`，子目录 conftest 反向 `from tests import`，绑定 rootdir 注入与 prepend import 模式。
9. **正向**：23/23 个未跟踪新 src 模块都被测试文本引用；1593 个函数体 **0 组精确重复**；`tests/support/` 15 个 helper（约 1190 行）显著降低了重组后的重复。

## 发现

### P0-1 测试重组不是纯移动：HEAD 的 131 个测试函数实例（130 个唯一名）无同体替代（已证实）

**证据**
- HEAD：`git ls-tree -r --name-only HEAD -- tests` = 110 个 py，`def test*` 共 **1189** 个（唯一名 1188）。
- 工作树：159 个被 collect 的测试文件，`def test*` 共 **1593** 个（唯一名 1593，**无重名**）；顶层已无 `tests/test_*.py`（最后一个顶层 py 是 `tests/__init__.py`）。
- 旧名 1188 个中 130 个不再出现；对每个消失函数做函数体模糊匹配（difflib ratio，对比全部 1593 个新函数体，归一化空白/注释）：
  - ratio ≥ 0.95（纯改名）：**0 个**
  - 0.75–0.95（改名 + 改写）：12 个，例如 `tests/test_continuous_qc.py:37 test_recovery_annotation_rejects_epochs_without_resetting_again` → `tests/data/test_continuous_qc.py::test_epoch_exclusion_does_not_reset_continuous_filter_state`（0.921）。
  - < 0.75（体已改写/合并/删除）：**119 个**。
- 110 个 HEAD 文件中 109 个仍有至少一个同名函数存活；唯一整体无同名存活的是 `tests/test_recovery_contamination.py`（1 个函数），其行为由新文件 `tests/data/test_recovery_contamination.py` 承接（需人工确认等价）。
- `git status --porcelain -uall -- tests`：`D` **109** 个、`M` 2 个（`conftest.py`、`tests/__init__.py`）、`??` 8 个新目录。任务书所述"118 个 tracked 测试文件删除"中，另 9 个删除不在 `tests/` 下（`frozen/` 清单等），属其他分区。

**影响** 109 个文件确实搬入 8 个子目录并新增约 405 个函数，但 119 个旧测试的断言在文本层面无法追溯。它们可能被更强的参数化新测试覆盖，也可能被静默丢弃。消失名单包含 `test_bids_sidecars_reject_path_escape_and_bad_coordinates_before_raw_read`、`test_attested_cache_rejects_hash_disagreement`、`test_full_budget_is_included_only_with_sufficient_purged_test`、`test_forced_switch_returns_complete_failure_ledger_when_none_are_eligible` 这类**安全/预算契约**用例——正是最不该无记录消失的一类。

**建议**
1. 生成"旧文件 → 新文件"映射表（旧函数名与当前文件函数名交集可半自动生成），对 119 个无同体替代者逐个标注 `kept / merged(nodeid) / retired(reason)`，放入 `doc/evidence/` 并索引。
2. 优先补回安全/契约类：路径逃逸、哈希不一致、预算下限、forced-switch 账本、candidate split 门。
3. 以后"纯移动"与"行为修改"分两个 commit，便于 `git log --follow` 与覆盖率 diff。

**验证方式** `python tmp/audit/scratch/sim.py`；或 `git show HEAD:tests/test_brainsync.py | Select-String 'def test_'` 对照 `Select-String tests/brainsync/test_brainsync.py 'def test_'`。

### P1-1 pytest 配置过薄：无 marker 注册、无 warning 策略、无超时、无分层（已证实）

**证据** `pyproject.toml:31-34` 仅
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
```
- 全树 `@pytest.mark.*` 只有 `parametrize`（302 次）与 `skipif`（3 次）；无 `slow/integration/gpu/network` 分层，也未注册 `markers =`。
- 无 `filterwarnings`、无 `timeout`（pytest-timeout 未在 `pyproject.toml:23` 的 dev extras 中）、无 `--strict-markers`/`--strict-config`、无覆盖率配置。
- 根 `conftest.py:9-11` 在 import 时对仓库根 `.mne_home` 执行 `mkdir`（被 `.gitignore:9` 忽略）。

**影响** 无法按"快测 / 契约 / 需数据或工作站"选择性执行；AGENTS.md 要求广改动跑全量，但 2342 例无超时、无并发、无资源标记，任何外部依赖都可能拖死 CI；DeprecationWarning 无法升级为门禁，NumPy/PyTorch 升级漂移不会被捕获。

**建议** 增加 `markers = ["slow", "integration", "gpu"]`、`--strict-markers --strict-config -ra`、`filterwarnings = ["error::DeprecationWarning:n2p3net.*"]`，并加 pytest-timeout/cov/xdist（见 P1-2）。

### P1-2 无 CI、无覆盖率门禁、无超时，测试规模已超出"手工全量"能力（已证实）

**证据**
- 仓库根无 `.github/`、无 `tox.ini`、无 `.gitlab-ci.yml`（`Test-Path` 均 False）：**没有任何自动化执行入口**，AGENTS.md 只写"广泛改动跑全量"，靠人遵守。
- dev extras（`pyproject.toml:23`）= `packaging/pytest/ruff`，不含 pytest-timeout/cov/xdist；`doc/engineering.zh.md:25` 与 `README.md:50` 的正确安装命令是 `uv sync --locked --extra signals --extra baselines --extra dev`——即"只装 `[dev]`"会缺 mne/pandas 等测试期依赖。
- 实测 `pytest --collect-only`：**159 文件 / 2342 用例 / 0 收集错误 / 15.3 s**。
- 测试含真实外部动作：11 个文件调 `subprocess`；`tests/environment/test_dependency_contract.py:144-150` 起子进程验证 import 图（自带 `timeout=60`）；`tests/research/test_dashboard.py:69` 起 `Popen` 服务、`:159` 调 `node -e`；`tests/train/test_pretrain_supervised.py:72` 跑 CLI 入口。
- 子集实测：`tests/environment + models/test_candidate_evidence + data/test_channel + transfer/test_time_split + research/promotion/test_plan` 共 **138 例全过**（最慢 5.46 s = import-graph 子进程）；`transfer/test_candidate_transfer + brainsync/test_brainsync + train/test_pretrain_supervised` 共 **130 例、2 skip、exit=0**。

**影响** 2342 例回归完全依赖人工记忆；没有覆盖率基线，无法说明 24 个未跟踪 src 模块是否被执行；单个挂起子进程可拖死整轮。

**建议**
1. 建 `.github/workflows/tests.yml`：矩阵 `{ubuntu, windows} × {3.11, 3.14}`；步骤 `uv sync --locked --extra signals --extra baselines --extra dev` + `ruff check` + `pytest -m "not integration"`；工作站数据/CUDA/前端用例走本地 `pytest -m integration`。
2. 加 `--timeout=120`、`-n auto`（先验证 `tmp_path` 隔离）、`--cov≥基线`（按包分级，不要一刀切）。
3. 把"159 文件 / 2342 例 / 15 s 收集"写进 `doc/engineering.zh.md`，作为"全量"的可核对定义。

### P1-3 测试环境耦合：同一份代码在不同机器跑的是不同测试集（已证实）

**证据**
- 30 个测试文件模块级 `import mne`、16 个 `import pandas`、50 个 `import torch`，而 `pytest.importorskip` 全树 **0** 处：缺 `signals` extra 时是 collection error，不是 skip。
- `tests/data/test_gtn.py:22` `GTN_ROOT = PROJECT_ROOT / "mne_data/MNE-P3-data"`，`:88` `skipif(not HAS_GTN)`；本机该目录存在（约 92 GB），CI 上整文件 skip。
- `tests/brainsync/conftest.py:8-15` 要求同级 `../BrainSync_Tool_newest/BrainSync_Tool/.venv/Scripts/python.exe`，否则 skip；本机存在。
- `tests/research/test_dashboard.py:144-145` `shutil.which("node")`，无 node 即 skip；本机存在（`D:\Code\Nodejs\node.exe`）。
- 3 处 `skipif(not torch.cuda.is_available())`；另有 9 处 `pytest.skip(...)`（含 `"GTN 数据未下载"`、`"Experiment_611 不存在"`）。
- `addopts="-q"` 且未加 `-rs`，这些 skip 的**原因默认不显示**，只有进度条上的 `s`。

**影响** "全量通过"在本机包含真实 92 GB 数据与同级前端工程的集成路径，在 CI/他人机器上静默退化为更小集合；同一 commit 的绿色含义不可复现。这是可复现性风险，不只是便利性问题。

**建议** 引入 `integration`（外部数据/前端/node/CUDA）、`slow`（>5 s）、`gpu` 三个 marker；默认加 `-ra`；CI 用 `-m "not integration"` 并统计跳过分布；把 92 GB 数据依赖改为显式 fixture skip，并输出"跳过 N 例（原因分布）"。

### P2-1 测试包化（`tests/__init__.py`）是隐性全局契约（已证实）

**证据** `tests/__init__.py` 从空文件变为含 `PROJECT_ROOT = Path(__file__).resolve().parent.parent`；8 个子目录均新增 `__init__.py`；`tests/brainsync/conftest.py:5` 使用 `from tests import PROJECT_ROOT`；```pythonpath = ["src"]``` 未包含仓库根。

**影响** 测试依赖"仓库根在 `sys.path`"与 prepend import 模式；若改用 `importmode=importlib`、把 tests 移出根目录或在别处运行 `pytest tests/...`，`from tests import ...` 会失败。单文件执行之所以可用，靠的是 pytest 对包根的自动插入，而非显式配置。

**建议** 改用根 `conftest.py` 提供的 `project_root` fixture，或显式 `pythonpath = ["src", "."]`；保留 `__init__.py` 也可以，但应在 `doc/engineering.zh.md` 说明原因。

### P2-2 `addopts = "-q"` 使 pytest 9 不再打印最终统计行（已证实）

**证据** 同一文件两次实测：无 `-q` 输出末行 `1 passed in 1.01s`；带 `-q`（即 pyproject 默认）只有进度点与 `[100%]`，**没有 "N passed / M skipped" 行**；130 例子集同样只有 durations 与 `[100%]`，最终统计行缺失（exit code 仍为 0）。

**影响** CI 日志无法一眼看出通过/跳过数量；失败时只能靠 traceback 推断规模；与缺省 `-rs` 叠加后 skip 数量与原因都不可见。

**建议** 从 `addopts` 删除 `-q`（需要安静时命令行再加），并加 `-ra`。1 行改动、零风险。

### P2-3 断言风格：错误契约偏弱、跨实现比较过严、11 例无断言（已证实）

**证据**
- 全树 `assert` 4296 条 / 1593 测试（均值 2.7）；`pytest.raises` 932 处，其中 **100 处无 `match=`**（约 11%），只验证"抛了某种异常"。
- 11 个测试体内**没有任何** `assert`/`assert_*`/`pytest.raises`/`pytest.warns` 记号，例如 `tests/data/test_manifest.py::test_artifact_policy_accepts_the_gt_n_low_variance_floor`（仅调用 `FoldLocalArtifactPolicy(...).validate()`，不抛即过）、`tests/transfer/test_matched_transfer_controls.py::test_zero_calibration_and_pure_baselines_need_no_unfittable_scratch`（连续 4 次 `validate_*`，未断言返回值/异常）。
- 另有 74 例的断言全部藏在同文件私有 helper（如 `_assert_cross_segment_epoch_excluded`、`_reject_empty_partition_checkpoint`），失败只能定位到 helper 内一行，测试名与断言脱节。
- 数值比较：`assert_array_equal` 287 次 vs `assert_allclose` 138 次；97 处 `assert x == <float>`。多数 `array_equal` 用于"同一实现两条路径逐位一致"（合理且强），但也存在对 `sosfiltfilt` 参考实现逐位比较（`tests/data/test_filter_phase.py:31` 附近），对 SciPy/BLAS 版本敏感。

**影响** 错误契约测试若只断言"抛异常"，实现改成抛 `KeyError` 也会绿；逐位比较在依赖升级时产生与代码无关的红。

**建议** 新/改 `pytest.raises` 必带 `match=`；跨库/跨路径比较用 `assert_allclose(rtol=1e-6, atol=1e-9)`，仅"同一调用两次"用 `array_equal`；把 helper 中关键断言提升为测试内一行，helper 返回中间量供断言。

### P2-4 根 `conftest.py` 的全局副作用：单线程预算 + 仓库内建目录（已证实）

**证据** `conftest.py:6-7` 在任何 import 前 `setdefault` 把 `OPENBLAS/OMP/MKL_NUM_THREADS` 设为 `"1"`；`:9-11` 在 import 时对仓库根 `.mne_home` 执行 `mkdir(parents=True, exist_ok=True)` 并设置 `_MNE_FAKE_HOME_DIR`。`.mne_home/` 与 `.pytest_cache/` 均被忽略（`.gitignore:7,9`），不污染 `git status`。

**影响**
1. 测试永远在单线程 BLAS 下运行，无法暴露线程数相关的不稳定/竞争；而 AGENTS.md 明确要求云端 25 CPU 线程、性能优先，测试与生产执行环境不一致。
2. conftest 在**只读工作区**（CI 常见）会在 import 阶段 `mkdir` 失败——collection-time 崩溃，不是 skip。
3. 根目录出现工具目录，虽被忽略，清理/归档时易误带。

**建议** 用 `tmp_path_factory` 生成 MNE home；线程预算改为环境变量可覆盖（`N2P3_TEST_THREADS`，默认 1），并增加一个显式 `-m threads` 的并发一致性测试用 4/8 线程跑同一批数值断言。

### P2-5 正向观察：新模块覆盖与测试去重做得较好（已证实）

**证据**
- `git status --porcelain -uall -- src`：24 个未跟踪项（23 个 py）+ 49 个修改；23 个未跟踪模块名在 `tests/` 语料中**全部**出现（0 个完全无引用）。
- 120 个 src 模块中仅 `src/research/domain_folds.py` 在 tests 中零文本提及；它被 `src/research/source_selection.py:33` 引用，而 `source_selection` 在 tests 中出现 57 次——属间接覆盖，非明确缺口。
- 1593 个函数体**精确重复体 0 组**（归一化后 SHA1 比对），说明重组没有靠复制粘贴扩充用例。
- `tests/support/` 15 个 helper（约 1190 行）承载合成数据/快照/检查点构造，是重组的主要收益。

## 未能验证与存疑

- **行/分支覆盖率未知**：`pytest-cov` 未安装，安装依赖被禁止；本报告只能给出"模块被测试文本引用"的下界，不能证明每个分支被执行。
- **119 个消失测试是否被行为等价覆盖**：仅用函数体文本相似度判定；有些测试可能被改写成参数化/多个新测试，语义等价但文本差异大。需人工逐个确认（建议按 P0-1 的映射表处理）。
- **全量运行结果未知**：按指令未跑全量；只跑了 138 + 130 例两个子集（均 0 失败），因此不能断言 2342 例在当前工作树全绿。
- **flake 率未知**：未重复运行同一子集（时间盒限制）；`tests/models/test_baselines.py` 使用未播种 `np.random.randn` 构造形状/冒烟数据，但恰好未用于阈值断言（阈值用例的 `make_p300_data` 内置 `seed=0`，见 `tests/support/datasets.py:155-157`）。
- **慢测试分布未知**：只得到子集 durations（最大 5.46 s）；无全量 `--durations` 数据。
- **"fixture 污染"仅静态检查**：未构造乱序执行实验证明跨文件状态泄漏；已确认 `tests/` 内只有 `tests/brainsync/conftest.py` 1 个 session 级 fixture（外部运行时探测），无 `chdir` 泄漏（3 处 `monkeypatch` 相关均在 `test_moabb_adapter.py`），无 `os.environ[...] =` 写入（4 处均为断言读取）。

## 已检查范围

- 目录与入库状态：`tests/` 全树 185 py（159 测试 + 26 辅助）；`git status --porcelain -uall -- tests conftest.py`；HEAD 110 py 的逐文件 `git show`。
- 静态统计：1593 个 `def test*` 的断言数、`raises`、marker、函数体长度、重复体 SHA1；130 个消失函数与 1593 个现存函数体的 difflib 相似度（`tmp/audit/scratch/sim.py`）。
- 配置：`pyproject.toml:21-34`；根 `conftest.py` 全文；`tests/__init__.py`；`tests/brainsync/conftest.py` 全文；`tests/environment/*` 全文。
- 运行：`pytest --collect-only`（2342 例、0 错误、15.3 s）；子集 A（138 例）与子集 B（130 例，2 skip）实际执行；`pytest --version` = 9.1.1；venv Python = 3.14.3。
- 外部依赖探测：`mne_data/MNE-P3-data`、`../BrainSync_Tool_newest/...python.exe`、`node`、CUDA、`git check-ignore .mne_home/.pytest_cache/tmp`。
- 未覆盖：`experiments/` 测试入口本身的正确性、`tests/support/*` 的单元测试、并发/乱序执行实验、全量耗时画像、覆盖率数字。
- 审计副作用：运行 pytest 生成 `.pytest_cache/`、`.mne_home/`、`tmp/audit/scratch/pt*/`（均在 `.gitignore` 内）；未修改任何 tracked 文件（`conftest.py`/`pyproject.toml`/`tests/__init__.py` 的 `M` 为审计前既有状态）。
