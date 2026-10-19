# 08 — 工程与工具链审计（pyproject / uv.lock / requirements / 打包 / lint / pytest / venv / 脚本）

审计分区：`pyproject.toml`、`uv.lock`、`requirements*.txt`、`conftest.py`、`.gitattributes`、`open_dashboard.cmd/ps1`、`.claude/`、`.venv` 与 `.venv-gpu`、包/导入结构、工具链（无 CI）。

约束：只读（唯一写入为本报告与 `tmp/audit/scratch/`）。以当前工作树为准（非 HEAD）。

## 结论摘要

- **P1（跨平台复用时等价 P0）换行符未归一化 + 源码自哈希**：同一提交在 Windows 检出为 CRLF、Linux 为 LF，`implementation_sha256` 与 `data_modules_sha256` 因此不同（实测 `41e555fd…` vs `c722eedc…`），跨平台 resume/证据闸门必然误判“实现已变更”。
- **P1 打包命名空间污染**：`find(where=["src"])` 把 `data/models/train/stats/research/transfer/baselines` 等 9 个通用名发布为顶层包；安装形态下 `source_preparation_method()` 还依赖仓库内 `experiments/` 脚本，wheel 不可独立使用。
- **P1 环境不可复现**：同一 `uv.lock` 在 Linux py<3.14 解析 torch 2.8.0/CUDA 12，在 py>=3.14 解析 torch 2.13.0/CUDA 13；本地 `.venv`=2.13.0+cpu、`.venv-gpu`=2.13.0+cu130 是手工覆盖，lock 中没有 CUDA 13 index；无 `[tool.uv] required-version`（uv 0.11.26 与 0.12.9 并存）；`requires-python` 无上界。
- **P1 工作树 429 项未提交**：80 M / 118 D / 231 ??，159 个新测试文件与 1521 个测试函数只存在于工作树，HEAD 仍是旧测试布局；仓库历史上从未有 CI。
- **P2 无 CI/无 pre-commit，但补门禁成本极低**：全仓 `ruff check .` 已通过（399 个 py 文件）；`ruff format --check` 有 273 个文件待格式化；`pytest --collect-only` exit 0。
- **P2 `open_dashboard.ps1` 安全与可维护性**：硬编码云主机/端口/root/密钥名，两处 `StrictHostKeyChecking=no`，且 README/doc/AGENTS 完全没有引用（孤立运维脚本）。
- **P2 依赖契约覆盖不全**：只断言 Linux 3.11–3.13（cu12）与 Windows 3.14；Linux 3.14 的 cu13 分支、numpy 2.4.6/2.5.2 的 Python 分叉均无断言。
- **P2 `.venv` 漂移**：6 个 lock 外手工包（pypdf/paramiko/invoke/bcrypt/cryptography/pynacl），`uv sync --check --extra ...` 判定 outdated；两套 venv 合计约 4.45 GB。
- **P2 项目文件管理**：无 LICENSE/README/license 元数据；`requirements-minimal.txt` 引用不存在的 `setup_cloud.sh`；`.claude` 仅靠用户全局 ignore，备份文件已暴露为未跟踪；`src/__init__.py` 为空但 namespace 发现仍把空目录收进 top_level。
- **正向**：`uv lock --check` 通过（151 packages）；`ruff` 通过；`pytest tests/environment` 5 passed（含依赖契约与 Python 3.11 AST 兼容性）；`git diff --check` exit 0。

## 检查计划

1. `pyproject.toml` 全文（build-system / project / scripts / extras / tool.ruff / tool.pytest / setuptools 打包）
2. `uv lock --check`（应无输出=通过）；锁文件元数据（requires-python、torch 平台 marker、extras、版本数量）
3. `requirements*.txt` 与 pyproject 的漂移（逐项对照包名/版本/约束）
4. 打包与导入：setuptools `find where=src` 是否把 `data/models/train/...` 发布为顶层包；`import data` 命名冲突；测试 `pythonpath=src`；`__init__.py` 覆盖；package-data
5. Ruff：运行 `ruff check src experiments tests`；核对配置覆盖范围与未覆盖目录
6. pytest 配置：testpaths / markers / addopts / 收集可行性（限定范围 collect-only）
7. 无 CI / 无 pre-commit：确认 `.github` 缺失、`.pre-commit-config.yaml` 缺失，检查 `.git/hooks` 是否有本地钩子
8. 两个 venv：Python 版本、torch 版本、体积；与 `doc/engineering.zh.md` 声明的一致性；uv 版本未 pin
9. `.gitattributes`、`.claude/`、`open_dashboard.*` 可移植性；全局 gitignore 排除情况
10. 工程改进建议（CI 矩阵、锁定、打包、lint 门禁）按优先级

## 发现

### P1 — 换行符未归一化，源码自哈希导致跨平台/云端证明失效（已证实）

**证据**
- `.gitattributes` 仅 6 行，只有 `*.sh text eol=lf` 与 `doc/assets/** -text`，**没有** `* text=auto` 或 `*.py text eol=lf`（`.gitattributes:1-6`）。
- 全局 `core.autocrlf=true`（`git config --show-origin` → `C:/Users/chent/.gitconfig`）；工作树实际混用 CRLF/LF：`src/data/preprocess.py` 778 个 CRLF / 1028 个 LF，`src/transfer/operator_lda.py` 188 CRLF / 197 行，`pyproject.toml` 25 CRLF / 45 行；`conftest.py` 为纯 LF（本会话修改过）。
- 多个模块在 import 时对**自身字节**做 SHA256：`src/data/preprocess.py:32`、`src/data/epochs.py:46`、`src/transfer/operator_lda.py:27`、`src/transfer/noise_model.py:19`、`src/transfer/exact_covariance.py:53`、`src/transfer/information_bound.py:79`、`src/transfer/continuous_decision.py:36`、`src/transfer/temporal_noise.py:34`。
- 实测同一文件同一提交：工作树（CRLF）sha256=`41e555fd…a93b4f`，把 CRLF 归一为 LF 后=`c722eedc…e589d78d`（Python 直接读字节计算）。二者不可能同时匹配。
- 这些哈希被写进产物/证据并用于闸门：`src/transfer/temporal_noise.py:219` 对 `provenance["implementation_sha256"]` 做**严格不等**判定；`src/data/preprocess.py:771-772` 把 `preprocess.py`/`epochs.py` 哈希写入记录。
- 更严重的是 `src/data/source_records.py:91-100`：`data_modules_sha256` 对 `src/data` 下**所有** `*.py` 逐个哈希，外加 `directory.parents[1]/experiments/prepare_source_records.py`；`source_records.py:207` 用 `preparation_method != method` 严格比较，不一致即 `ValueError("source preparation implementation changed ...")`。

**影响**
- Windows 上生成的源缓存/证据（哈希基于 CRLF）在 Linux 云端（LF checkout）**必然**被判为“实现已变更”：resume 失败、已完成 shard 无法复用，或只能整批重跑。这正是“公开源预训练 + 个人校准”多机流程的核心可复现性阻断。
- 反之在 Linux 生成的工件回到 Windows 也无法通过同样的闸门；跨平台对照实验的 `implementation_sha256` 不可比。
- `data_modules_sha256` 还把所有 data 模块绑成一个整体：任何无关小改（哪怕注释）都会使全部源缓存不可 resume，代价高。

**建议**
- `.gitattributes` 增加 `* text=auto eol=lf`（至少 `*.py text eol=lf`、`*.toml text eol=lf`、`*.json text eol=lf`），然后在独立提交中 `git add --renormalize .` 并验证 `git diff --check`；要求所有平台 `core.autocrlf=false`。
- 哈希改为**规范化字节**（先 `b.replace(b"\r\n", b"\n")`）或对 git blob（`git hash-object`）取值；或只对语义内容（AST/常量）取哈希。
- `data_modules_sha256` 改为“声明式文件清单”（显式列出参与预处理的模块），或按模块级哈希分别校验，避免全目录绑定。

**验证方式**：`git ls-files --eol src/transfer/operator_lda.py`（应显示 `w/crlf`、`i/lf`）；在 Linux/WSL 克隆同一提交后运行 `python -c "import transfer.operator_lda as m; print(m._IMPLEMENTATION_SHA256)"`，与 Windows 值对比；或对同一文件分别用 CRLF/LF 计算 `hashlib.sha256`。

### P1 — setuptools 把 9 个通用顶层包发布到 site-packages，导入命名冲突与“源码树依赖”陷阱（已证实）

**证据**
- `pyproject.toml:25-26` 只有 `[tool.setuptools.packages.find] where = ["src"]`，没有 `include`/`namespaces` 限制；`src/{baselines,data,models,research,stats,train,transfer,measurement,audit_p300}/__init__.py` 均存在（`src/measurement`、`src/audit_p300` 无 `__init__.py`，但 editable 安装的旧元数据仍列出，见下）。
- 旧构建元数据 `src/n2p3_net.egg-info/top_level.txt` 明确列出 9 个顶层名：`audit_p300, baselines, data, measurement, models, research, stats, train, transfer`。
- 全仓库导入风格为顶层绝对导入（`from data.contract import …`、`from models.n2p3net import …`、`from transfer.calibrated import …`），grep 统计 `data.*` 出现数百次、`transfer.*`/`models.*` 各上百次；没有任何 `n2p3_net.` 前缀导入。
- `src/__init__.py` 存在但不会被打包（`find` 的根是 `src`），说明“src 作为包”的布局是半成品。
- 安装后 `src/data/source_records.py:99` 仍要读 `directory.parents[1] / "experiments/prepare_source_records.py"`：wheel 安装后该路径在 site-packages 下不存在，调用 `source_preparation_method()` 会 `FileNotFoundError`。

**影响**
- `data`、`models`、`train`、`stats`、`research`、`transfer` 是极通用的顶层名；装进任何共享 venv 都会与其他项目/PyPI 同名包冲突（`data` 在 PyPI 上已有包），且冲突是“后者覆盖前者”的静默导入错误。
- 该仓库明确预期本地存在 `/data/`（`.gitignore:10-11` 忽略 `/data/`、`/Data/`）。pytest 的 `pythonpath=["src"]` 与 rootdir conftest 都会改 `sys.path`；一旦根目录出现 `data/`（无 `__init__.py` 也可作为 namespace package 参与导入），`import data` 存在被本地数据目录遮蔽的风险（机制成立，当前根目录无 `data/`，故标“疑似”）。
- wheel 不是自包含的：`source_preparation_method` 依赖仓库内 `experiments/` 脚本，发布包在仓库外不可用，且没有测试覆盖安装形态。

**建议**
- 改为真正的 src-layout 命名空间：`src/n2p3_net/{data,models,train,...}`，全仓库改为 `from n2p3_net.data...`；或至少在 `packages.find` 中显式 `include = ["n2p3_net*"]` 并加 `py.typed`。
- 短期可加 `[tool.setuptools.packages.find] include = [...]` 白名单，避免 `audit_p300`/`measurement` 这类半成品目录被打包。
- 把 `experiments/prepare_source_records.py` 的哈希改为可注入的常量或随包发布的清单；安装形态下缺失时给出明确报错。

**验证方式**：`python -m build --wheel` 后 `unzip -l dist/*.whl | grep -E '^(data|models|train)/'`；在干净 venv 中 `pip install dist/*.whl` 并 `python -c "import data; print(data.__file__)"`。

### P1 — 两个 venv 与 lock 的 Torch/CUDA 分叉不可由 `uv sync --locked` 复现（已证实）

**证据**
- `pyproject.toml:15-17`：Linux x86_64 且 Python<3.14 → `torch==2.8.0`、`torchaudio==2.8.0`；其他平台/版本 → `torch>=2.4`。
- `uv.lock:3340-3430` 因此同时锁了 `torch 2.8.0`（cu12 NVIDIA 依赖，`uv.lock:3352-3365`）与 `torch 2.13.0`（cu13 依赖，`uv.lock:3394-3403`，含 `cuda-toolkit`/`nvidia-*-cu13`）；`uv.lock:4-16` 的 resolution-markers 还包含 `sys_platform == 'emscripten'` 分支。
- 实际环境：`.venv` = CPython 3.14.3 + torch **2.13.0+cpu**（`torch.version.cuda=None`）；`.venv-gpu` = 3.14.3 + torch **2.13.0+cu130**（CUDA 13.0，`cuda.is_available()=True`）。二者都不是 lock 直接产出的 `2.13.0` 本地版本；`.venv-gpu` 是按文档“先锁定安装再显式覆盖 +cu130”得到的（`doc/engineering.zh.md:458-459`）。
- `pyproject.toml` 没有 `[tool.uv]` 段：无 `required-version`、无 `[tool.uv.sources]`、无 PyTorch CUDA index 配置；`uv.lock` 全部 source 都是 `https://pypi.org/simple`。
- 两个 venv 由不同 uv 创建：`.venv/pyvenv.cfg` 记录 `uv = 0.11.26`，`.venv-gpu/pyvenv.cfg` 记录 `uv = 0.12.9`；当前 `uv --version` = 0.12.9。`pyproject.toml` 未 pin uv 版本。
- Python 版本本身也不在 lock 的语义内：`requires-python = ">=3.11"`（`pyproject.toml:9`、`uv.lock:3`），而 venv 用 3.14.3；同一 lock 在 3.11/3.12 上解析到 torch 2.8.0+cu12，在 3.14 上解析到 2.13.0+cu13。

**影响**
- “本地 GPU 验证 → Linux 云端 5090 训练”不是同一软件栈：云端按 lock 会得到 torch 2.8.0/CUDA 12（若 Python<3.14），本地是 2.13/CUDA 13；吞吐、算子实现、fused Adam/compile 行为都不可直接外推。文档第 12-13 行已承认“不能当作 RTX5090 训练验证”，但配置层没有任何强制或提示。
- `uv sync --locked` 无法重建 `.venv-gpu`：CUDA 13 轮子来自 PyTorch 专用 index，lock 里没有；重建只能靠口头/文档中的手工覆盖步骤，属于不可审计的环境真相。
- uv 版本漂移（0.11.26 创建、0.12.9 现在）会改变 resolver/lock 行为；没有 `required-version` 时，不同机器可能生成不同 lock。

**建议**
- 在 `pyproject.toml` 固化工具链：`[tool.uv] required-version = ">=0.12,<0.13"`（或 `=0.12.*`），并在文档写明 `uv python pin 3.12` 或 3.14 的唯一选择。
- 若要保留 CUDA 13 验证路径，增加显式 `[tool.uv.sources]`/extra（如 `gpu` extra + `torch = { index = "pytorch-cu130" }`）与 `[[tool.uv.index]]`，让 `.venv-gpu` 可由 `uv sync --extra gpu` 复现；否则删除 `.venv-gpu` 的“锁定环境”表述，明确它是手工覆盖。
- 统一 Python 版本（至少 CI/云端/本地同一 minor），把 torch 平台 marker 收窄成“仅 Linux x86_64 + 指定 Python”并补 `<3.15` 上界，避免未来 Python 版本静默落到 `torch>=2.4`。

**验证方式**：`uv sync --check --extra signals --extra baselines --extra dev`（应无 “Would install/uninstall” 差异；本次裸 `uv sync --check` 因未带 extras 报 “Would uninstall 102 packages”，属预期）；`uv pip check`；对比 `uv.lock` 中 torch 的 `resolution-markers` 与 `python --version`。

### P2 — requirements 与 pyproject/文档漂移，云端引导脚本缺失（已证实）

**证据**
- `requirements.txt:3` 只有 `-e .[signals,baselines,dev]`，不 pin 版本；与 `uv sync --locked` 是两条不同解析路径（pip 会重新解析 `torch>=2.4` 等）。注释自称“便捷入口”，但未说明与 lock 的差异边界。
- `requirements-minimal.txt:2` 写“Install CUDA-enabled Torch separately in setup_cloud.sh”，但 `setup_cloud.sh` **不存在**（`Get-ChildItem -Recurse -Filter setup_cloud.sh` 无结果；`git ls-files` 仅有 `requirements*.txt`）。
- `requirements-minimal.txt:3-7` 列 `numpy/scipy/mne/scikit-learn/threadpoolctl`，**不含 torch**，而“cached GTN MS-EEGNet training path”显然需要 torch；该文件无法独立构成训练环境。
- `threadpoolctl>=3.5` 只在 requirements-minimal 出现，`pyproject.toml:10-18` 未声明；若代码直接 `import threadpoolctl`，则属于未声明依赖（scikit-learn 会传递安装，故实际可用，但语义上是隐式依赖）。

**影响**：新机器/云端若照 requirements-minimal 安装会缺 torch；照 requirements.txt 安装会绕过 lock；照 `setup_cloud.sh` 安装则脚本不存在。三条路径没有一条被测试或 CI 覆盖。

**建议**：删除或重写 requirements-minimal（改成 `uv sync --locked --extra ...` 的说明或 `uv export` 产物）；把云端引导脚本入库（`scripts/setup_cloud.sh`）并纳入文档；`threadpoolctl` 若被直接 import 则提升为显式依赖，否则从 minimal 移除。

**验证方式**：`git ls-files '*setup_cloud*'` 为空；`grep -rn "import threadpoolctl" src experiments`；在干净容器按 requirements-minimal 安装后跑一个训练入口。

### P2 — `.claude/` 仅靠全局 ignore，备份文件已暴露为未跟踪（已证实）

**证据**
- `git check-ignore -v .claude/settings.local.json` → `C:/Users/chent/.config/git/ignore:1:**/.claude/settings.local.json`（**用户全局** ignore，不在仓库内）。
- 同目录还有 `.claude/settings.local.json.backup-20261011-232131`（62 字节，内容仅 `{"permissions": {"allow": [...]}}`，无密钥），**不匹配**该全局规则，因此 `git status` 显示 `?? .claude/`。
- 仓库 `.gitignore` 没有 `.claude/` 条目（`.gitignore:1-37`）。

**影响**：换一台机器/CI 克隆时，`settings.local.json` 不再被忽略，可能把本地权限配置（甚至未来加入的 env/token）提交入库；备份文件当前就可能被 `git add .` 误收。

**建议**：在仓库 `.gitignore` 加入 `.claude/settings.local.json` 与 `.claude/*.backup-*`（或整个 `.claude/`，若确定不共享）；不要依赖全局 ignore。若将来配置含密钥，加 `git secrets`/pre-commit 检查。

**验证方式**：`git check-ignore -v .claude/settings.local.json.backup-20261011-232131`（当前应无输出=未忽略）；`git status --porcelain --ignored .claude`。

### P2 — pyproject 工作树改动削弱了测试隔离，且未提交（已证实）

**证据**
- `git diff pyproject.toml`：`addopts = "-q -p no:cacheprovider"` → `addopts = "-q"`，删除了禁用 pytest cache 的参数（原注释说明是为受限/只读环境避免 `.pytest_cache` PermissionError）。
- `git diff conftest.py`：删除了说明 MNE 配置重定向原理的 docstring，新增 `OPENBLAS/OMP/MKL_NUM_THREADS` 的 `setdefault("1")`。
- 两处均为未提交修改（`git status`：` M conftest.py`、` M pyproject.toml`），但全仓库测试行为已随之改变；`.pytest_cache/` 目录已实际出现在根目录。

**影响**：任何“测试通过”的结论都依赖工作树而非 HEAD；在只读挂载/沙箱中跑测试会重新出现 cache 写入失败噪音。`setdefault` 让线程数取决于外部环境（用户设了 8 就用 8），数值型测试的线程相关波动不可控。

**建议**：把 `-p no:cacheprovider` 恢复为默认（或改用 `--cache-clear` 策略并文档化），线程变量改为**强制**赋值并加注释；将这两处改动作为一个独立提交，避免与审计/功能改动混在一起。

**验证方式**：`git diff --stat`；在只读目录运行 `pytest tests/environment -q` 观察是否产生 cache 警告。

### P2 — 打包元数据不完整：无 LICENSE/README/license 字段、无 console scripts（已证实）

**证据**
- `pyproject.toml:5-23` 无 `readme`、`license`、`authors`、`classifiers`、`[project.urls]`、`[project.scripts]`；`src/n2p3_net.egg-info/PKG-INFO` 也只有 Name/Version/Summary/Requires-Python/Requires-Dist。
- 根目录没有 `LICENSE`/`COPYING`/`NOTICE`/`CITATION`（`Get-ChildItem -File` 过滤为空）；唯一许可证文件是 `src/baselines/structured_erp.LICENSE`（1635 字节），通过 `pyproject.toml:28-29` 的 `package-data` 分发。
- `README.md` 中没有 license/许可/版权字样（grep 无命中）。

**影响**：仓库对外发布/复用时的授权边界不清；wheel 不带 README，`pip show` 无长描述；第三方 `structured_erp` 代码的许可证只在包内，根目录无 NOTICE，合规审计困难。

**建议**：补 `LICENSE`（与用户确认许可证类型）、`readme = "README.md"`、`license = {file = "LICENSE"}`（PEP 639 用 `license = "..."` + `license-files`）；在 README 增加“第三方代码与许可证”小节，列出 `structured_erp.LICENSE` 来源。

**验证方式**：`python -m build --wheel` 后检查 `.dist-info/METADATA` 是否含 `License-Expression`/`Description`；`unzip -l` 检查 `LICENSE` 是否在 wheel 根。

### 正向结论（已证实）

- `uv lock --check` 通过：exit 0，stderr 仅 “Resolved 151 packages in 1ms”；`pyproject` 的依赖声明与 `uv.lock` 一致（package-data 改动不影响 lock）。
- `ruff 0.16.5`：`ruff check src experiments tests` → `All checks passed!`（exit 0）。配置覆盖 `E,F,I,N,W,UP,B`，忽略 `E501/N803/N806/N812`，与数学张量命名约定一致。
- pytest 配置为 `testpaths=["tests"]`、`pythonpath=["src"]`、`addopts="-q"`（`pyproject.toml:31-34`），与 tests 重组为子目录后的布局兼容（tests 下有 brainsync/data/environment/models/research/support/train/transfer 八个子目录，159 个 `test_*.py`）。

### P1 — 工作树有 429 项未提交变更（含测试整体迁移），HEAD 与当前测试基线不一致（已证实）

**证据**
- `git status --porcelain`：80 `M`、118 `D`、231 `??`，合计 429 项（本报告审计时点）。
- 旧布局的 `tests/test_*.py` 有 118 个被删除（`D`），新布局 `tests/{brainsync,data,environment,models,research,train,transfer}/` 下有 159 个 `test_*.py`、1521 个 `def test_`，其中 `tests/` 相关未跟踪条目 109 个。
- `tests/__init__.py` 被修改为提供 `PROJECT_ROOT`（`git diff`：空文件 → 5 行），而 `tests/support/__init__.py` 仍是未跟踪；`tests/environment/test_dependency_contract.py:10` 依赖 `from tests import PROJECT_ROOT`。
- 未跟踪分布还包括 `doc/` 105 项、`frozen/` 68 项、`src/` 24 项、`experiments/` 19 项（`git status` 路径聚合）。
- 仓库从未有过 CI：`git ls-files '.github/*'` 为 0，`git log --all -- .github .gitlab-ci.yml .pre-commit-config.yaml` 无提交；`core.hooksPath` 未设置，`.git/hooks` 无自定义钩子。

**影响**
- 当前 159 个测试文件、1521 个测试函数只存在于工作树；任何误清理/回滚都会丢失整套新测试。HEAD 仍是旧布局，CI（若接入）会跑旧套件或直接失败。
- 无法用“提交 + diff”审阅这次迁移；`tests/` 与 `src/` 的新旧导入同时存在时，`from tests import PROJECT_ROOT` 这类跨包相对定位对布局变化极其敏感。
- 没有自动化门禁，`ruff`/`pytest`/`uv lock --check` 全靠个人记忆执行；本次审计即发现工作树已偏离 HEAD 的 pytest 配置。

**建议**：把迁移拆成可审阅提交（① 新增 tests 子目录与 `tests/__init__.py`；② 删除旧 `tests/test_*.py`；③ src 新模块；④ doc/evidence），每步跑一次 `pytest -q`；随后立刻加最小 CI（见下条）。在 `AGENTS.md` 增加“提交前必须 `uv lock --check` + `ruff check` + 定向 pytest”的检查单。

**验证方式**：`git status --porcelain | Measure-Object -Line`；`git ls-tree -r --name-only HEAD tests | Select-String 'test_' | Measure-Object` 与工作树 159 对比；`git stash list` 确认无备份。

### P2 — 无 CI / 无 pre-commit，格式化未纳入门禁（已证实）

**证据**
- 无 `.github/`、`.gitlab-ci.yml`、`.pre-commit-config.yaml`、`Makefile`、`tox.ini`、`noxfile.py`、`mypy.ini`/`pyrightconfig.json`、`.editorconfig`（根目录枚举均为空）。
- `ruff check src experiments tests` → `All checks passed!`（0.16.5）；`ruff check . --show-files` 扫描 399 个 py 文件，exit 0 → **全仓 lint 当前是干净的**，加门禁不会产生历史债。
- `ruff format --check src experiments tests` → `273 files would be reformatted, 101 files already formatted`，exit 1；配置中没有 `[tool.ruff.format]`，也没有格式检查步骤。
- 测试套件可收集：`pytest --collect-only -q` exit 0；`pytest tests/environment -q` → 5 passed（含依赖契约与 Python 3.11 AST 兼容性测试）。

**影响**：格式与 lint 漂移只能靠人工发现；`ruff format` 一次应用会产生 273 文件大 diff，越晚做越难 review（当前正是好时机，因为工作树已有 429 项未提交）。

**建议**：先单独提交一次 `ruff format src experiments tests`（纯格式，不改逻辑），随后在 CI/pre-commit 加 `ruff check`、`ruff format --check`、`uv lock --check`、`git diff --check`、定向 pytest。CI 矩阵建议：OS `ubuntu-latest` + `windows-latest`，Python `3.11/3.12/3.14`（与 lock 的三条分支对应），Linux 只跑 CPU wheel（避免 CI 拉 2GB CUDA 依赖），并加一个 `uv build` + 干净 venv `pip install dist/*.whl` + `python -c "import data"` 的打包冒烟任务。

**验证方式**：`ruff format --check` 当前必然失败；在临时分支应用格式化后重跑 `ruff check` 应仍通过；`pytest --collect-only` 已 exit 0。

### P2 — `open_dashboard.ps1` 硬编码云端基础设施并关闭主机密钥校验，且无文档引用（已证实）

**证据**
- `open_dashboard.ps1:2-8` 默认参数硬编码 `connect.weste.seetacloud.com`、端口 `35832`、用户 `root`、远程根 `/root/autodl-tmp/n2p3-net`、本地/远程端口 18812/8812、密钥文件名 `id_ed25519_n2p3_cloud_20260827`。
- 两处 SSH 调用都带 `-o StrictHostKeyChecking=no`：`open_dashboard.ps1:46`（远程启动 dashboard）与 `open_dashboard.ps1:98`（本地隧道）；第 104 行还注明“SSH may prompt for a password”。
- 脚本为 Windows 专用（`Get-NetTCPConnection`、`Get-CimInstance`、`ssh.exe`、`$env:USERPROFILE`），但远程侧用 `.venv/bin/python`（Linux）。
- `README.md`、`doc/*.md`、`AGENTS.md` 中 **没有** `open_dashboard` 的任何引用 → 脚本是孤立的运维入口，使用者只能读源码猜参数。

**影响**：公开/共享仓库泄露具体云主机、端口与 root 登录方式（虽无密钥值），降低攻击门槛；`StrictHostKeyChecking=no` 使隧道可被中间人劫持（dashboard 可能含实验指标与管理接口）。脚本不可移植到 Linux/macOS 操作端，也未说明“仅本地运维使用”。

**建议**：把 host/port/user/root 改为必填参数或读取 `$env:N2P3_SSH_*`/`configs/cloud_training_resources.json` 同级的本地配置文件（该文件应 gitignore）；删除 `StrictHostKeyChecking=no`，改用 `known_hosts` 固定指纹（或首次 `ssh-keyscan` 后校验）；在 `doc/engineering.zh.md` 增加“Dashboard 隧道”小节说明用法与安全边界；若不打算共享，移到 `tmp/`/`scripts/local/` 并 gitignore。

**验证方式**：`git grep -n open_dashboard`（仅脚本自身命中）；`ssh -o StrictHostKeyChecking=yes ...` 能否连通；用 `PSScriptAnalyzer` 检查未使用参数（`$Run` 仅拼 URL）。

### P2 — 依赖契约只覆盖 Linux 3.11–3.13 与 Windows 3.14，Python 3.14/Linux 的 CUDA 13 分支无人校验（已证实）

**证据**
- `tests/environment/test_dependency_contract.py:76-94`：对 Linux 3.11/3.12/3.13 断言 `torch==2.8.0`、`torchaudio==2.8.0`；对 Linux 3.14 只断言 `torch>=2.4`，没有断言 lock 中该分支解析到 `2.13.0`，也没有校验其 CUDA 13 依赖集合。
- 同一测试 `:97-119` 只校验 Linux 3.11–3.13 分支的 cu12 依赖；`uv.lock:3394-3403` 的 2.13.0 分支使用 `cuda-toolkit`/`nvidia-*-cu13`，完全不在断言范围内。
- `doc/engineering.zh.md:11-13` 声明 Linux x86_64 py<3.14 固定 2.8.0、其他平台按 `>=2.4`，“不能当作 RTX5090 训练验证”；`:458-459` 又记录本地 `.venv-gpu` 被手工覆盖为 `torch==2.13.0+cu130`。
- `uv.lock:1868-1876 / 1953-1964`：numpy 在 `python<3.12` 解析为 2.4.6，在 `>=3.12` 解析为 2.5.2；torch 同理 2.8.0（cu12）vs 2.13.0（cu13）。

**影响**：同一 `uv.lock` 在 Python 3.11 与 3.14 上给出不同的 numpy/torch/CUDA 主版本；“通过依赖契约测试”的绿色只覆盖 3.11–3.13 Linux，而实际本地与未来云端都可能落在 3.14 分支。科学结论（收敛、吞吐、数值容差）跨环境不可直接比较，文档中的“验证通过”容易被误读为覆盖全部矩阵。

**建议**：在 `test_dependency_contract.py` 增加 Linux 3.14 分支断言（torch 2.13.0 + cu13 依赖集合 + cp314 manylinux wheel 存在），或显式声明“不支持 Linux 3.14 GPU”，并在 pyproject marker 中排除；新增 `.python-version` 固定一个 minor（如 `3.14`），CI 矩阵与文档只承诺该版本；把 numpy 的 python-version 分叉收敛为单一版本（或记录“数值基线环境”）。

**验证方式**：运行 `pytest tests/environment/test_dependency_contract.py -q`（当前 4 passed）；用 `uv lock --check` 后 `uv export --python-version 3.14 --format requirements-txt` 查看实际解析结果。

### P2 — `.venv` 含 6 个 lock 之外的手工安装包，环境非严格由 lock 派生（已证实）

**证据**
- `uv sync --check --offline --extra signals --extra baselines --extra dev` 输出：`Would uninstall 7 packages`（bcrypt 5.0.0、cryptography 50.0.1、invoke 3.0.3、n2p3-net 0.1.0、paramiko 5.0.0、pynacl 1.6.2、pypdf 6.17.0），`Would install 1 package`，结论 `The environment is outdated`。
- 这 6 个包在 `uv.lock` 中不存在（grep 无结果），也不被 `src/experiments/tests/doc` 任何文件 import（正则搜索无命中）→ 是手工 `pip install`/`uv pip install` 残留，而非代码依赖。
- 体积：`.venv` 1,051,340,149 B（约 1.05 GB），`.venv-gpu` 3,399,707,146 B（约 3.40 GB），合计约 4.45 GB；`pyvenv.cfg` 记录创建者 uv 0.11.26 / 0.12.9，当前 uv 0.12.9。

**影响**：本地通过、干净环境失败的“幽灵依赖”风险；`uv sync` 会静默删除这些包，若有隐藏脚本依赖它们（如 pypdf 处理论文、paramiko 部署）会在同步后突然失效。两套 4.45GB venv 不可版本化，只能靠文档描述。

**建议**：重建 `.venv`：`uv sync --locked --extra signals --extra baselines --extra dev`，然后用 `uv tool install` 安装 pypdf/paramiko 等运维工具（与项目环境隔离）；把 `uv sync --check --extra ...` 纳入提交前检查；在 `doc/engineering.zh.md` 记录两套 venv 的用途、体积与重建命令。`.venv-gpu` 的重建需先解决上一条的 CUDA index 问题。

**验证方式**：重跑 `uv sync --check --extra signals --extra baselines --extra dev` 应输出 `The environment is up-to-date`；`uv pip list --outdated` 可选。

### P2 — 打包/项目元数据仍有缺口：空 `src/__init__.py`、无 `py.typed`、ruff 无上界、requires-python 无上界（已证实）

**证据**
- `src/__init__.py` 为 0 字节（`Get-Item` 长度 0），但 `src` 不是被 `find(where=["src"])` 发现的包，属于无意义占位；同时 `src/audit_p300`、`src/measurement` 没有 `__init__.py`（当前两者均为空目录、0 文件）却被陈旧的 `src/n2p3_net.egg-info/top_level.txt` 列为顶层包 → 说明 setuptools 的 namespace 发现（`namespaces = true` 默认）会把 `src/` 下任意目录收进发行包，只要将来出现文件。
- `pyproject.toml:9` `requires-python = ">=3.11"` 无上界；`dev = [... "ruff>=0.14"]`（`:23`）无上界，实际安装 0.16.5；`torch>=2.4`（`:17`）无上界。
- 无 `py.typed`、无 `readme`/`license`/`classifiers`（见上一条 P2 元数据）。
- `[tool.setuptools.package-data]` 仅声明 `baselines = ["structured_erp.LICENSE"]`；当前 `src` 下除该文件外没有需要随包分发的数据文件（已枚举非 `.py` 文件），故暂无缺包数据，但新增 JSON/模板时容易遗漏。

**建议**：删除空的 `src/__init__.py`；在 `packages.find` 设 `namespaces = false` 并显式 `include = ["baselines*", "data*", ...]`（或迁移到 `n2p3_net` 命名空间）；给 ruff/torch 加兼容上界；增加 `py.typed`（若计划类型检查）；建立“新增数据文件必须登记 package-data”的检查（可用测试断言 wheel 内容）。

**验证方式**：`python -c "import tomllib;..."` 读取配置；`python -m build --wheel` 后 `unzip -l` 核对顶层目录（本次未构建，见“未能验证”）。

## 未能验证与存疑

- **未跑全量 pytest**（约束只允许限定范围）：只做了 `pytest --collect-only`（exit 0）与 `pytest tests/environment`（5 passed），因此“测试全绿”未验证；doc 中“1530 通过/4 跳过”的旧记录不能当作本次证据。
- **未在 Linux/WSL 实际检出并计算哈希**：CRLF/LF 差异由 `git ls-files --eol`（`i/lf w/mixed`）与 Python 直接读字节推断；同一提交在 Linux 上应为 `w/lf`，未做端到端复核。
- **未构建 wheel**：构建会写 `dist/`，超出只读范围；顶层包结论基于 `src/n2p3_net.egg-info/top_level.txt` 与 setuptools `find` 语义，未用 `unzip -l` 验证。
- **`import data` 被根目录 `/data/` 遮蔽**：机制成立（pytest 会把 rootdir 与 `src` 都放入 `sys.path`），但当前根目录无 `data/`，未复现，标为“疑似”。
- **uv 0.11.26 能否读取 `uv.lock revision=3`**：无法安装旧版 uv 验证；只确认当前 0.12.9 的 `uv lock --check` 通过。
- **`.venv` 中 `torch-2.13.0.dist-info` 与 `torch.__version__=2.13.0+cpu` 的不一致**：未进一步查 wheel 元数据，不影响 lock 的 `torch==2.13.0` 判定，但说明本地版本标识与 lock 不完全同构。
- **`open_dashboard.ps1` 远程 dashboard 的暴露面**：只读了脚本，未审计 `experiments/dashboard_server.py` 的绑定地址/鉴权（属 experiments 分区）。
- **`.git` 8GB（loose objects 7.26GiB）**、`formal/`、`frozen/`、`experiments/runs`、`mne_data` 未审计（其他分区或非本报告范围）。

## 已检查范围

**文件（实际阅读）**
- `pyproject.toml`（全文 45 行）、`uv.lock`（头部 resolution-markers、torch/torchaudio/numpy/moabb 条目）、`requirements.txt`、`requirements-minimal.txt`、`conftest.py`、`.gitattributes`、`.gitignore`、`open_dashboard.cmd`、`open_dashboard.ps1`（146 行全文）、`.claude/settings.local.json.backup-*`（仅 JSON 键名）。
- `src/data/source_records.py:79-115,195-230`、`src/data/preprocess.py:32`、`src/transfer/operator_lda.py:27`、`src/transfer/temporal_noise.py:34,219` 等哈希相关行（用 Select-String 定位，未通读大文件）。
- `tests/__init__.py`、`tests/environment/test_dependency_contract.py`（149 行全文）、`tests/environment/test_python_compatibility.py`（23 行全文）、`src/n2p3_net.egg-info/{PKG-INFO,top_level.txt}`。
- `doc/engineering.zh.md` 仅环境相关行（`:8-13,25-28,454-467`）；`README.md` 仅安装/测试命令行（`:44-57,111-118`）。

**命令（只读）**
- Git：`status --porcelain`、`ls-files`、`diff`、`diff --check`、`check-ignore -v`、`ls-files --eol`、`log --all -- .github ...`、`config --get core.autocrlf/core.hooksPath`。
- uv：`uv --version`、`uv lock --check`、`uv sync --check --offline [--extra ...]`。
- Ruff：`ruff --version`、`ruff check src experiments tests`、`ruff check . --show-files/--statistics`、`ruff format --check`。
- pytest：`pytest --collect-only -q`（全量收集）、`pytest tests/environment -q`（5 项）。
- Python 小脚本：读取 torch/numpy 版本、计算 CRLF/LF 哈希、读 TOML 元数据；PowerShell 枚举/统计文件与体积。

**未触碰**：`formal/`、`frozen/` 内容、`experiments/runs`、`mne_data`、`.git` 对象、其他分区文件；未修改任何已有源文件；仅创建并增量更新本报告。审计时点：2026-10-19 08:02–08:45（Asia/Shanghai）。

## 附：按优先级的改进顺序（建议）

1. **P1 先做换行符归一化**：`.gitattributes` 加 `* text=auto eol=lf`，独立提交 `git add --renormalize .`；随后把自哈希改成 LF 规范化字节或 git blob 哈希（改动小、影响全仓库证据链）。
2. **P1 立刻补最小 CI 与 pre-commit**：`uv lock --check`、`ruff check`、`ruff format --check`（先单独提交一次格式化）、定向 `pytest`、wheel 构建 + `import data` 冒烟；Linux 只装 CPU wheel 控制 CI 成本。
3. **P1 把 429 项未提交变更拆成可审阅提交**（先提交新测试目录与 `tests/__init__.py`，再删旧测试），并在 `AGENTS.md` 写提交前检查单。
4. **P1 收紧打包面**：迁移到 `n2p3_net` 命名空间（或至少 `include` 白名单 + `namespaces=false`），并让 `source_preparation_method` 不再依赖仓库内 `experiments/` 路径。
5. **P1 统一环境真源**：`.python-version` 固定 minor；`[tool.uv] required-version`；为 CUDA 13 建 `gpu` extra + index（或删除手工覆盖的“锁定”表述）；`requires-python` 加 `<3.15`。
6. **P2 清理与合规**：重建 `.venv`/重建说明、删除或重写 `requirements-minimal.txt`、补 LICENSE/readme/license 元数据、把 `open_dashboard` 的主机与 `StrictHostKeyChecking` 改为配置/安全默认。
