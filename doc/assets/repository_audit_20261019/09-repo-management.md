# 09 — 项目文件管理 / 版本控制 / 存储分层审计

- 仓库: `D:\BCI\n2p3-net`（branch `main`, HEAD `f1a303c`, 最近提交 2026-09-09 09:58 +0800）
- 审计时间: 2026-10-19
- 范围: 仓库文件管理（版本控制、ignore/attributes、Git 卫生、存储分层、命名与重复、治理方案）
- 非范围: data / models / transfer / docs / security / performance（另有分区代理）
- 方法: 只读 `git status/ls-files/log/show/fsck/cat-file/count-objects` + `read/grep/glob` + 只读 PowerShell 统计；**未修改任何既有文件**
- 结论级别: **P0** 立即阻断/数据丢失风险 · **P1** 一周内处理 · **P2** 一月内/卫生项

## 0. 结论摘要

| # | 结论 | 级别 | 证据 |
|---|---|---|---|
| 1 | `main` 停在 `f1a303c`（2026-09-09），此后 0 次提交；全部 10 月工作只在 `codex/*` 分支与工作树中 | **P0** | `git log --since=2026-09-10 main` = 0 行；`git for-each-ref --sort=-committerdate` 最新 ref 为 `codex/research-director-resume-20261013`（2026-10-19） |
| 2 | 11 个**已跟踪** `src` 模块 import 9 个**未跟踪**模块；HEAD 的树无法导入，fresh clone 直接 ImportError | **P0** | `src/models/n2p3net.py:14`→`models.parameter_groups`；`src/transfer/calibrated.py:23`→`baselines.structured_erp`；`git ls-tree -r HEAD` 中无这些文件 |
| 3 | `tests/` 整体替换但未入库：HEAD 110 个测试中 109 个被删，磁盘 185 个中 184 个未跟踪，仅 `tests/__init__.py` 仍受跟踪 | **P0** | `git status` 118 D（109 在 tests）；`git ls-files tests`=110；磁盘 185；untracked tests=184 |
| 4 | `.git` 8.04GB 中约 **7.2GiB 是不可达垃圾**：1601 个 unreachable blob（解压后 7.72GiB），最大 5 个为 1.22GB/894MB/453MB/451MB/354MB 的 ZIP/GZIP | **P1** | `git fsck --unreachable --no-reflogs`=2144；`git count-objects -vH` loose 7.26GiB；对象头探测=ZIP/GZIP |
| 5 | 112 个 **dangling commit 含 2026-10-06→10-14 的真实集成工作**（"Integrate scientific numerical and execution contracts" ×12），当前不在任何分支上；`git gc --prune=now` 会直接销毁 | **P0** | `git fsck --unreachable` 按日期直方图 10-06..10-14 共 53 个；`git show -s` 抽样 e148163/d8ccbf4/594f2d2 |
| 6 | 8 个 frozen 快照目录（目录+manifest+归档）被物理删除，18 份 evidence 里 **32 条链接**指向不存在的路径；9 个被删 manifest 只是这场删除的可见部分 | **P1** | README.md:106；doc/evidence/brainsync_notch_20260909.zh.md:75；`Test-Path frozen/technical_audit_before_20260906`=False |
| 7 | CRLF 警告根因是 `core.autocrlf=true` + 仅 3 类 attributes 规则：157 个 `i/lf w/lf`、68 个 `i/lf w/mixed` 文件每次 `git add/diff` 都会刷警告 | **P2** | `git config core.autocrlf`=true；`git ls-files --eol` 统计；`git diff` 实测 stderr 出现 "\.gitignore', LF will be replaced by CRLF" |
| 8 | `formal/`（385 .lean，0 入库）、`configs/*.json`（含 AGENTS.md 指定的 `cloud_training_resources.json`）、19 个 `experiments/*.py` 全在版本控制之外 | **P1** | `git ls-files formal`=0；`git status` untracked configs=4 |
| 9 | frozen 100 个目录中 **87 个有 manifest、13 个没有**（其中 `local_history` 按设计排除，真正缺失 12 个，均 ≤2.6MB 的 20261018 lean 快照） | **P2** | 逐目录 `*manifest*.json` 计数：53/18/12/2/1/1 |
| 10 | 只读核对否定了两条 Lead 假设：`doc/assets/**/*.npz` **已被正确忽略**（77 个 npz，0 个 untracked）；`doc/assets` 与 `frozen` 的 zip **零同名重叠** | — | `git ls-files -i -c --exclude-standard` 为空；552 个 asset zip vs 376 个 frozen 归档，交集 0 |

> 说明：本报告所有计数均在本机当前工作树实测；Lead 提供的 41GB/98GB/92GB 等体量数字未重复全量遍历（见 §4 的验收命令）。

---

## 1. 版本控制风险量化

### 1.1 总体账本

| 指标 | 实测值 | 命令 |
|---|---|---|
| HEAD | `f1a303c42c68173e8b1452ceb26bd84923b8a6aa`（2026-09-09 09:58 +0800） | `git log -1` |
| main 之后提交 | **0** | `git log --since=2026-09-10 main` |
| 工作树条目 | 429：80 M / 118 D / 231 ?? | `git status --porcelain=v1` |
| 入库文件 | 570（.py 260 / .json 141 / .md 70 / .pdf 55） | `git ls-files` |
| 分支/引用 | 72 个本地+远程 ref，其中 `codex/*` 约 55 个 | `git branch -a` |
| reflog | 239 条，最早 2026-09-05 | `git reflog` |

结论：**版本控制已从"记录历史"退化为"记录 9 月 9 日的一个旧切片"**。10 月的 5 周工作（src 新增 24 模块、tests 全量重写、formal 全量新增、configs 新增）没有任何一条提交在 `main` 上。

### 1.2 只存在于未跟踪状态的源码（P0）

24 个未跟踪 `src` 文件 / 6183 行（占 src 49249 行的 12.6%）：

```
src/baselines/{structured_erp.py, structured_erp.LICENSE, temporal_transfer.py}
src/data/{bidirectional.py, ea_artifact.py, finite_filter_operator.py, harmonic.py,
          source_admission.py, source_parsers.py, source_records.py}
src/models/parameter_groups.py
src/research/{comparison.py, source_catalog.py, source_matrix.py}
src/train/early_stopping.py
src/transfer/{continuous_decision.py, decision_experiments.py, exact_covariance.py,
              information_bound.py, noise_model.py, operator_lda.py, raw_continuous.py,
              source_objective.py, temporal_noise.py}
```

**关键点不是"少提交了几个文件"，而是 HEAD 与工作树的 import 图不一致**。已跟踪文件对未跟踪模块的依赖：

主要依赖（已跟踪 → 未跟踪）：`models/n2p3net.py:14`→`models.parameter_groups`；`transfer/calibrated.py:23,24,51`→`baselines.structured_erp` / `baselines.temporal_transfer` / `train.early_stopping`；`transfer/calibration_candidates.py:15`、`transfer/recipes.py`、`transfer/subject_adapter.py`、`transfer/checkpoint.py`→`baselines.*`；`baselines/deep.py:47,81`→`train.early_stopping` / `transfer.source_objective`；`transfer/epoch_selection.py:30`、`transfer/target_experiments.py:290,291,486`→`data.source_records` / `research.source_matrix`；`data/euclidean_alignment.py:4`→`data.ea_artifact`；`research/source_selection.py`→`research.*`。

后果：`git clone` / `git worktree add HEAD` 得到的树 `import n2p3net.models.n2p3net` 即抛 `ModuleNotFoundError: models.parameter_groups`。任何 CI、容器构建、协作者复现、以及"从 HEAD 重跑一次"的审计动作都会失败——这不是风格问题，是**可复现性断裂**。

同时 `pyproject.toml:28-29` 新增 `[tool.setuptools.package-data] baselines = ["structured_erp.LICENSE"]`，而 `src/baselines/structured_erp.LICENSE` 未跟踪 → 即使 sdist 构建成功，许可证文件也会缺失。

### 1.3 tests 重组：118 删除的入库方案（P0）

- HEAD 跟踪 110 个测试；工作树删除了其中 **109 个**，仅剩 `tests/__init__.py`。
- 磁盘 185 个 `.py`，其中 **184 个未跟踪**（净增 75 个测试文件）。
- 这意味着 **rename 关系完全未记录**，Git 无法在 `git add` 时自动关联新旧文件；`git log --follow` 的历史将从 9 月 9 日断裂。

建议入库方式（只建议）：
1. `git add -A tests` 后**先看** `git status` 是否出现 `renamed:` 行；若显示为 `deleted + new file`，用 `git add -A --find-renames=40% tests`（或 commit 后 `git log --stat -M` 复核）。
2. 分两个 commit：① 纯 rename/移动；② 内容变更 —— 保住 `git blame` 的可回溯性。
3. 保留 `tests/__init__.py`（pytest rootdir 依赖，`pyproject.toml:31-34`），并在提交前跑全量 `pytest`：`pyproject.toml:33` 的 `addopts` 刚从 `"-q -p no:cacheprovider"` 改回 `"-q"`，受限环境会重新触发 `.pytest_cache` 写权限问题（原注释已说明，diff 中被删）。

### 1.4 未跟踪的配置与形式化资产（P1）

- `configs/` 4 个未跟踪：`cloud_training_resources.json`（**AGENTS.md 明文要求从这里起步**）、`unified_training_budget.json`、`brainsync_bidirectional_preprocessing.json`、`source_pool_expansion.example.json`。HEAD 里的 `configs/` 只有旧文件 → 任何按文档操作的代理/人都读不到资源下限配置。
- `experiments/*.py` 19 个未跟踪（`run_domain_transfer_matrix.py`、`benchmark_unified_training.py` 等）——这些是产出 10 月结果的实际入口，却不在历史里。
- `formal/`：**0 个文件入库**，385 个 `.lean`（39 个 `Audit*.lean`）全部未跟踪；`formal/lean/.gitignore:1` 排除 `.lake/` 是正确的，但这个 `.gitignore` 本身也没入库，一旦有人 `git add formal` 而漏掉它，会把 30435 个 `.lake` 文件卷进来。
- `artifacts/personal_erp_identity_06/{lean.log,report.json}` 未跟踪且**无任何 ignore 规则**——构建产物有被误提交的风险。


---

## 2. .gitignore / .gitattributes 审计

### 2.1 覆盖矩阵（实测，非推测）

| 路径 | 体量 | 规则 | 判定 |
|---|---|---|---|
| `.venv/ `.venv-gpu/` | 0.98 + 3.17 GB | `.gitignore:1` `.venv*/` | ✅ 正确 |
| `experiments/runs/` | 41 GB / 40956 文件 | `.gitignore:20` | ✅ 正确（含 8134 个 py 副本、614 pyc） |
| `mne_data/` | 92 GB | `.gitignore:22` | ✅ |
| `cloud_inputs/` | 2 GB | `.gitignore:24` | ✅ |
| `/tmp/` | 40 MB（含 `tmp/cloud_ssh` 的 pip --target 安装 14.9MB） | `.gitignore:12` | ✅ |
| `frozen/**/*.{zip,gz,tar,7z,bundle}` | 376 归档 / 27.0 GiB | `.gitignore:28-32` | ✅（本次审计的最大正确决策） |
| `frozen/**/checkout/` | — | `.gitignore:34` | ✅ |
| `frozen/.../BrainSync_Tool_newest/`（junction） | 目标 94 GB | `.gitignore:36`（**未提交的改动**） | ⚠️ 规则正确但只存在于工作树，HEAD 无此规则 |
| `frozen/local_history/` | 133 文件 / 2.85 GB | `.gitignore:26` | ✅ |
| `doc/assets/**/*.npz` | 77 个 npz，**0 个 untracked** | `.gitignore:37` | ✅ **Lead 的"漏忽略"假设不成立** |
| `formal/lean/.lake/` | 30435 文件 / 2.5 GB | `formal/lean/.gitignore:1`（嵌套，未跟踪） | ⚠️ 有效但自身未入库 |
| `.claude/` | 1 个 `settings.local.json.backup-20261011-232131` | **无规则** | ❌ 漏忽略（P2） |
| `artifacts/` | `lean.log` + `report.json` | **无规则** | ❌ 漏忽略（P2） |
| `doc/assets/**/*.zip` | 552 个 / 25 MiB | **无规则** | ⚠️ 一旦 `git add doc/assets` 全量收入（P2） |

**否定性证据**：`git ls-files -i -c --exclude-standard` 输出为空 → 没有任何**已跟踪**文件被 ignore 规则误伤，因此"误忽略 evidence md / assets 图"在当前工作树**不成立**；`doc/assets/**/*.npz` 的 77 个文件全部被正确拦下（`git status -uall` 中 npz untracked = 0）。真正的问题是 `.gitattributes` 让 `doc/assets` 下的文本类证据（58 个 .md、31 个 .xml、1605 个 .json）**按二进制存储**，见 2.3。

### 2.2 CRLF 警告的确切来源（P2）

- `git config core.autocrlf` = **true**；`core.safecrlf`、`core.excludesfile` 均未设置（因此 `.claude` 没有任何全局兜底）。
- `.gitattributes` 只有 6 行，仅覆盖 `*.sh`（`text eol=lf`）、`*.pdf`（`binary`）、`doc/assets/**`（`-text whitespace=cr-at-eol`）、`doc/assets/**/*.xml`（`-whitespace`）。**`.py/.md/.json/.toml` 没有任何规则** → 走 autocrlf 自动转换。
- 实测复现：`git diff -- .gitignore pyproject.toml` 的 stderr 输出
  `warning: in the working copy of '.gitignore', LF will be replaced by CRLF the next time Git touches it`（同理 pyproject.toml）。
- 存量：`git ls-files --eol` 统计 570 个跟踪文件 —— `i/lf w/lf` 157、`i/lf w/crlf` 119、`i/lf w/mixed` 68、`i/crlf w/crlf` 18、`i/-text w/-text` 76、`i/none w/none` 12、`i/mixed w/mixed` 1。
  - `w/lf` 的 157 个 = 每次 `git add` 都警告的对象；
  - `w/mixed` 的 69 个 = 未来 `git add` 会被整体重写为 CRLF，产生**与内容无关的假 diff**（review 时被噪音淹没）。

修复建议（不执行）：`.gitattributes` 增加 `* text=auto eol=lf`（或在仓库层面 `core.autocrlf=false`），然后 `git add --renormalize .` 单独成一个 commit，避免混入功能变更。风险：会让 **所有** 文本文件在首次 renormalize 时产生一次全量 diff（本仓库 570 文件，可接受），但不做的话每次提交都带噪音且 `w/mixed` 文件会持续漂移。

### 2.3 attributes 的副作用

`doc/assets/** -text`（`.gitattributes:4`）把整个 assets 树标成二进制：`git diff` 对其中 58 个 `.md`、31 个 `.xml` 只能显示 "Binary files differ"，code review 与 `git blame` 失效。`-whitespace` 只对 `.xml` 恢复空白检查。若目标是"保留捕获证据的字节"，更精确的写法是按扩展名（`*.npz`、`*.zip`、`*.png`、`*.log`）设 `-text`，让 `.md/.json/.xml` 仍可文本 diff。

---

## 3. Git 卫生与大对象

### 3.1 不可达对象：7.2GiB 的可回收空间（P1）

| 指标 | 值 |
|---|---|
| `git count-objects -vH` | count **11302** / size **7.26 GiB** / in-pack 3910 / size-pack 204.83 MiB / prune-packable **1145** |
| `git fsck --unreachable --no-reflogs` | **2144** 个（blob 1601 / tree 431 / commit 112） |
| unreachable 解压后总字节 | **7.72 GiB**（blob 1601 个占 7.72 GiB；其中 1573 个是 loose） |
| 计入 reflog 后 | 1813 个（reflog 仍保护 331 个） |

**最大 12 个 loose 对象全部 unreachable，且全部是 ZIP/GZIP**（对象头魔数探测：`50 4b 03 04` = ZIP，`1f 8b 08` = GZIP）：

| SHA（前 12） | 大小 | 类型 | 可达 |
|---|---|---|---|
| 2d81792625a5 | 1.22 GiB | ZIP | ❌ |
| 7739a6ae4a03 | 894 MiB | ZIP | ❌ |
| 29f938d86286 | 453 MiB | ZIP | ❌ |
| 0548cc6504e6 | 451 MiB | ZIP | ❌ |
| bd1368163c8c | 354 MiB | GZIP | ❌ |

其余 7 个（177/215/152/84/71/68/58 MiB，除 f689b57456f3 为 GZIP 外全为 ZIP）同样 100% unreachable。

来源判定：时间戳集中在 **2026-08-31 / 09-02**，正是 `.gitignore:28-32` 这批 `/frozen/**/*.zip|gz` 规则被加入之前。即当时 `git add frozen` 把冻结归档塞进了对象库，随后 `reset` 撤销，文件仍在 objects 里。今天 `.gitignore` 已挡住同类操作，但**历史垃圾没有被回收**——这就是 `.git` 8.04GB 与"pack 仅 205MB"的矛盾解释。

检出率：可达对象只有 1062 个（`git rev-list --objects --all`），而 pack 内有 3910 个 → **pack 中约 2848 个对象同样不可达**（早期 repack 保留了 dangling 对象）。

### 3.2 dangling commit：10 月的工作正在裸奔（P0）

112 个 unreachable commit 的提交日期分布：

| 日期 | 个数 |
|---|---|
| 2026-08-25 ~ 09-07 | 58 |
| **2026-10-06** | 10 |
| **2026-10-08 / 10-10 / 10-11 / 10-12 / 10-13 / 10-14** | 2 / 13 / 7 / 15 / 3 / 3 |

最新几个（`git show -s`）：
- `e148163` 2026-10-14 "Bind source reference size and public report schema at every ingress"
- `d8ccbf4` 2026-10-14 "Recheck original source size on transaction cache hits"
- `2c11e4b` 2026-10-14 "Preserve old producer replay and complete the declared public control plan"
- `594f2d2` / `413b08e` 2026-10-13 "Fix calibration selection on fixed requested pair risk"

即：**10-14 的工作比 HEAD 新 5 周，且没有任何 ref 指向它**。它们今天还能被 `git fsck` 找到，但
- `git gc`（默认 `gc.reflogExpire=90 days`）在 90 天后会连同 reflog 一起清掉；
- `git gc --prune=now` / `git prune` 立刻清掉；
- 一旦清掉，只能靠工作树里那 231 个未跟踪文件与 80 个修改文件重建。

**这是本仓库当前最高优先级的"可恢复性"风险**：在跑任何 gc 之前，先把最新 dangling commit 钉成 ref（`git branch rescue/dangling-20261014 e148163`），再考虑清理。

### 3.3 分支与引用卫生（P1）

- 72 个 ref：`codex/*` 约 55 个，其中大量同义重复：
  `codex/bi2014a-label-prefix` 与 `codex/retained-bi2014a-label-prefix-20261016`、`codex/joint-flat-ensemble` 与 `codex/retained-joint-flat-ensemble-20261016`、`codex/huebner-candidate` 与 `codex/retained-huebner-candidate-20261016`。
- 4 个分支被 worktree 占用（`git branch` 中的 `+`：bi2014a-label-prefix、huebner-candidate、joint-flat-ensemble、research-director-resume-20261013）——删除前必须先 `git worktree list` 确认，否则 `git branch -d` 会拒绝。
- 存在本地远程 `remotes/local-recovery/prefix-native-programme`（2026-10-16）——说明曾用"本地远端"方式抢救过工作，但没有文档说明它是什么、是否可信。
- `origin` 只有 `main` 与 `research/n2p3-transfer-ssl`；**10 月的工作一行都没有推到远端**，磁盘故障 = 全丢。无 CI（无 `.github/`，`git ls-files | grep -c github` = 0）。

### 3.4 gc / LFS / 历史改写建议与风险

| 动作 | 收益 | 风险 | 建议 |
|---|---|---|---|
| `git branch rescue/dangling-20261014 e148163` | 把 10 月工作钉住 | 无 | **立即** |
| `git reflog expire --expire-unreachable=now --all` + `git gc --prune=now` | 回收 ~7.2GiB，`.git` 降到 ~0.3GB | 销毁 112 个 dangling commit（含 10 月工作）、销毁 4GB ZIP 归档（可能是已删 frozen 快照的唯一副本） | 先做 rescue，再单独跑，,并在跑之前 `git bundle create` 备份 |
| `git gc --aggressive` | pack 更小 | 数小时、临时双倍空间 | 不必要，`--prune=now` 足够 |
| `git lfs migrate` 处理 Paper/ 55 PDF（191MB） | clone 变小 | 重写全部历史 hash，所有 codex 分支失效 | **不建议**（历史才 210 个 commit，收益 < 风险） |
| `git filter-repo` 删除历史中的大 ZIP | 永久瘦身 | 破坏协作与 reflog | 不建议 |
| `git bundle create backup.bundle --all` | 1 个文件备份 72 refs | 需 ~300MB | **立即**（无 CI、无推送，这是唯一低成本保险） |

备份/克隆影响：当前 `.git` 8.04GB，其中 90% 是垃圾；直接 `git clone` 会把这 8GB 一起搬走（clone 只传可达对象，实际约 205MB pack + 工作树 —— 这也是为什么"本地 8GB"与"远端 clone 很小"并不矛盾）。但如果用 `Copy-Item -Recurse` 备份目录，会连 7.2GiB 垃圾一起复制；用 `git bundle` 只含可达对象（**注意：bundle 会丢掉 112 个 dangling commit，所以 rescue 必须先做**）。


---

## 4. 存储分层判定

### 4.1 分层账本与判定

| 层 | 体量 | 入库状态 | 判定 | 验收命令（只读） |
|---|---|---|---|---|
| `.git` | 8.04 GB（pack 205 MiB + loose 7.26 GiB + 其余） | — | **压缩**：7.2GiB 为不可达垃圾（§3.1） | `git count-objects -vH`；`git fsck --unreachable --no-reflogs \| Measure-Object -Line` |
| `frozen/`（不含 junction） | 100 目录 / 376 归档 / **27.0 GiB** | 115 文件入库（manifest/README/delivery.json）；`*.zip/.gz/.tar` 被 ignore | **保留**（科学证据，本仓库的"账本"）；归档冷存、manifest 必须入库 | `(Get-ChildItem frozen -Recurse -File -Include *.zip,*.gz,*.tar \| Measure Length -Sum).Sum` |
| `frozen/.../BrainSync_Tool_newest`（junction） | 目标 94 GB | `.gitignore:36`（未提交改动） | **不计入仓库体量**；删除 junction 不丢数据 | `(Get-Item <path>).LinkType`→`Junction`；`Target`→`D:\BCI\BrainSync_Tool_newest`（实测） |
| `experiments/runs/` | 41 GB / 40956 文件（27.6GB .gz 交付包 + 5.5GB npz + 2.3GB pt + 1.56GB json + 8134 py 副本 + 614 pyc） | ignore | **冷存 + 裁剪**：交付包 .gz 移出工作盘；只留结果索引与复现所需最小集 | `(Get-ChildItem experiments/runs -Recurse -File \| Group-Object Extension \| Sort Count -Desc)` |
| `mne_data/` | 92 GB | ignore | **可重建缓存 → 删除/冷存**；但需在 evidence 记录 MNE 数据集版本 | `(Get-ChildItem mne_data -Directory).Name` |
| `cloud_inputs/` | 2 GB | ignore | **可重下 → 冷存/删除** | `(Get-ChildItem cloud_inputs -Recurse -File \| Measure Length -Sum).Sum` |
| `.venv` + `.venv-gpu` | 0.98 + 3.17 GB（Python 3.14.3，torch 2.8 cpu / cu130） | ignore | **保留一个**：3.14 上 wheel 稀少，重建成本 > 磁盘成本；另一个删除 | `.\.venv\Scripts\python -V` |
| `Paper/` | 55 PDF / 191.4 MB | **入库**（合理：文献是研究输入） | 保留 | `git ls-files Paper \| Measure -Line` |
| `doc/assets/` | 172.2 MB / 3352 untracked / 59 目录 | 仅 56 文件入库 | **选择性入库**：`.md/.json/.xml` 索引入库，`zip/npz/pt` 不入 | `git status -uall \| Where StartsWith '?? doc/assets/'` |
| `tmp/` | 40.3 MB（`audit` 26.5 + `cloud_ssh` 14.9） | ignore | 保留为本地草稿区（本报告即写在此） | `(Get-ChildItem tmp -Recurse -File \| Measure Length -Sum).Sum` |

> 体量口径修正：Lead 的 "frozen 98GB" 中 **94GB 是 junction 目标**（`D:\BCI\BrainSync_Tool_newest`），frozen 自身归档仅 27.0GiB。若用跟随 reparse point 的工具（PowerShell `-Recurse`、robocopy 默认、`tar -C frozen`）统计或备份，会把同一份 94GB 数两遍，甚至打包成两份。**验收/备份必须加 `/XJ`（robocopy）或 `-Attributes !ReparsePoint`。**

### 4.2 frozen manifest 与目录/归档的对应关系（抽查）

100 个目录中 **87 个含 manifest、13 个不含**（含 `local_history`，其按设计只有归档；真实缺失 12 个，见 §5.1）。

manifest 存在**两种 schema**，验收方式不同：
- **归档成员型**：`files` 是压缩包内部路径，例如 `frozen/brainsync_quality_20260907/manifest.json` 的键为 `original_rejected_session/.bidsignore`，附 `archive`+`archive_sha256`+`member_count`+`all_members_verified`。验收 = 比对 sha256 + `tar -tzf` 成员清单，**不是**比对工作树文件。
- **仓库快照型**：`files` 是仓库相对路径，例如 `frozen/bidirectional_qc_20260909_acceptance/source.manifest.json`（322 条，`AGENTS.md`、`configs/...`）。实测 **5 条已失效**：
  `tests/data/test_epoch_row_identity.py`、`tests/data/test_gtn_dataset.py`、`tests/data/test_ocular_strength_pipeline.py`、`tests/data/test_qc_features.py`、`tests/transfer/test_calibration_fit_reuse.py`
  —— 它们正是 §1.3 测试重组中被删掉的旧路径，而新树是扁平 `tests/test_*.py`。结论：**冻结验收记录与当前测试布局已经对不上**，重放验收会因为路径不存在而失败。
- 抽 28 个带 `files` 的 manifest、5919 条引用：1411 条按"目录相对"或"仓库相对"都解析不到——但这个数字**不能直接当缺陷率**，因为两种 schema 混在一起（归档成员型天然解析不到）。正确做法是逐 manifest 看有无 `archive` 键分派验收逻辑。

### 4.3 junction 的连带风险

`Get-Item` 实测：`Mode=d----l`、`LinkType=Junction`、`Target=D:\BCI\BrainSync_Tool_newest`。风险：
1. **磁盘统计**：同一 94GB 在两处出现，容量规划失真（本报告采用"frozen 自有 27GiB"口径）。
2. **备份**：`Copy-Item -Recurse` / `robocopy`（无 `/XJ`）会把 target 复制进备份，备份体积翻倍且产生两份可能不一致的副本。
3. **递归工具**：`tar`/`zip`/`du` 类工具跟随 reparse point 时可能长时间扫描或产生重复条目；Python `os.walk` 默认 `followlinks=False` 不跟随，行为不一致 → 同一脚本换语言结果就变。
4. **删除语义**：`Remove-Item -Recurse` 对 junction 在旧版 PowerShell 会删 target 内容（数据事故）。建议先 `(Get-Item x).Delete()` 或 `cmd /c rmdir x`。
5. 该 junction 的 ignore 规则（`.gitignore:36`）**尚未提交**，HEAD 中没有；一旦在未更新 ignore 的分支上 `git add -A`，Git 会把 junction 当普通目录尝试索引 94GB。

---

## 5. 命名 / 重复 / 索引

### 5.1 frozen 的 13 个无 manifest 目录

缺失清单（名称 × 文件数 / 大小）：`domain_transfer_cloud_20260916` 7/1.3MB、`lean_fold_artifact_20261018` 1/2.6MB、`lean_personal_erp_raw_loader_20261018` 8/0.3MB、`lean_v08_fit_20261018` 6/0.4MB，其余 8 个 `lean_v07_*_20261018` 均为 1 文件 / ~0MB（calibration_population、network_readout、observed_statistics、observed_temporal、parameter_ownership、splitbn_lifecycle、splitbn_momentum、splitbn_statistics）；`local_history` 133/2849.5MB。

判断：12 个真实缺失全是 **2026-10-18 当天的 Lean 小快照**（≤2.6MB，多为单文件），是 `.gitignore:26` 之外新产生的目录，尚未走"目录+manifest"流程；`local_history` 按 `.gitignore:26` 设计即只有归档、无 manifest，属正常。**P2**：统一由脚本补 manifest，或把这些目录并入既有 `lean_*` 目录。

### 5.2 空目录（Git 无法表达，需以 .gitkeep 或索引页代替）

实测 3 个：`doc/assets/deep_chain_audit_20260929/input/ea_static_mask`、`.../input/minimum_test_real_prefix`、`doc/assets/operator_research_202604/analysis_stage3_repair`。Lead 提到的 `operator_research_202604` 空目录是它的子目录；该父目录实际有 476 个文件。

### 5.3 重复

- `doc` 下 3 个 docx：`doc/progress_report_20260929.docx`（289,844 B）、`doc/progress_report_20260929_v2.docx`（292,322 B）**均未跟踪**；`doc/assets/progress_report_20260929/v2/source_report.docx` 与 v1 **同为 289,844 B**（疑似逐字节副本，建议 `Get-FileHash` 后删其一）。
- `doc/assets` 552 个 zip（25 MiB） vs `frozen` 376 个归档（27.0 GiB）：**basename 交集 = 0**，即不存在"同一归档两处存放"；所谓"doc/assets 与 frozen 重复"目前只在**证据类型**层面重叠（同一实验的 json/py 副本 + 归档），不是字节级重复。真正的重复是 `doc/assets` 内 747 个 `.py` 副本与 `experiments/runs` 的 8134 个 py 副本（后者已 ignore）。
- 55 个 `Paper/*.pdf` 191MB 全部入库，无重复文件名。

### 5.4 日期命名

同时存在三种约定：前缀式 `20261004_system_research` 类（少见）、后缀式 `system_research_20261004`（主流）、阶段式 `*_before_YYYYMMDD` / `*_validation`（frozen 快照）。`brainsync_notch_20260909_before` 与 `bidirectional_qc_20260909_acceptance` 说明"before/after/validation/acceptance"是语义后缀而非时间。**P2**：建议统一 `YYYYMMDD_<topic>_<stage>`，stage ∈ {before, after, validation, acceptance}，并在 `frozen/README.md` 固化。

### 5.5 2026-10-16 两条记录的含义（关键因果）

- `doc/evidence/workspace_cleanup_20261016.zh.md:15`：「删除文件共 20,936 份，原文件总计 1,539,921,492 字节，约 1.43 GiB」；`:12`「整理过程中用于校验的压缩包、逐文件库存和一次性脚本也已删除，只保留本页及简短操作清单」。→ **这正是 8 个 frozen 快照目录（目录+manifest+zip）消失的原因**：清理把"校验用压缩包"和快照一起删了，而 18 份 evidence 仍指向它们。
- `frozen/worktree_recovery_20261016.json`：`status = "further cleanup stopped; affected author trees restored"`，逐 worktree 记录 `restored_retained_files` 2344 / 762 / 104 与 `discarded_files_not_restored` 36 / 30 / 22，`source_snapshot` = 7e021ba / 55f1413 / 370f4f0（三个 commit 现均在 `codex/*` 分支上，实测可达）。→ 说明**科研 worktree 曾被误退役并部分恢复**，恢复时丢弃了 22–36 个文件/树；这也解释了 `codex/retained-*` 这批 20261016 重复分支的存在。

---

## 6. 分阶段治理方案（只建议，不执行）

### 立即（今天，全部只读或新增 ref）

1. **钉住 dangling commit**：`git branch rescue/dangling-20261014 e148163`（再用 `git log --oneline rescue/dangling-20261014` 确认 10 月链条完整）。这是后续任何 gc 的前置条件。
2. **整仓备份**：`git bundle create D:\BCI\backup\n2p3-20261019.bundle --all`（≈300MB；**必须在上一步之后**，否则 112 个 dangling commit 不进 bundle）。
3. **提交配置与源码**（不建议一次性 `git add -A`）：
   - `git add src configs formal/lean/.gitignore`，先看到 `formal/lean/.gitignore` 在列，再 `git add formal`（确认 `.lake` 未被加入：`git status --porcelain | Select-String '.lake'` 应为空）；
   - `git add tests` 并检查 rename 检测（`git status | Select-String renamed`）；
   - `.gitignore` 补 `.claude/`、`/artifacts/`；`.gitattributes` 补 `* text=auto eol=lf`（单独 commit）。
4. **合并 main**：把 `codex/research-director-resume-20261013` 或工作树对应分支 fast-forward/merge 到 `main`，并 `git push origin main`——当前 10 月工作零远端副本。

### 1 周内

5. **恢复 evidence 链**：对 18 份文档里的 32 条断链，二选一——① 从备份/回收站恢复 8 个快照目录；② 在文档中把链接改成"已按 20261016 清理移除"并登记 `frozen/README.md`，避免验收时踩空。被删的 9 个 manifest 可从 `git show HEAD:<path>` 直接取回（这正是它们还在 Git 里的价值）。
6. **修 manifest 漂移**：`frozen/bidirectional_qc_20260909_acceptance/source.manifest.json` 的 5 条 `tests/data/*`、`tests/transfer/*` 路径随测试重组失效 → 重新生成该 manifest 或加 `superseded_by` 字段。
7. **补 12 个 lean 快照 manifest**；`doc/assets` 增 `/doc/assets/**/*.zip` 与 `*.pt` 规则，只让 `.md/.json/.xml` 进库。
8. **CRLF 收敛**：`git add --renormalize .`（单独 commit），之后 `git diff` 不再刷 CRLF 警告。

### 1 月内

9. **回收 .git**：确认 `rescue/*`、`bundle` 就绪后，`git reflog expire --expire-unreachable=now --all` + `git gc --prune=now` → `.git` 从 8.04GB 降到 ~0.3GB。**先决条件：4GB 的 unreachable ZIP 若被认定为已删快照的唯一副本，必须先 `git cat-file blob <sha> > 恢复文件` 落盘再 gc。**
10. **分支收敛**：72 refs → 保留 `main` + 3–5 个活跃分支；`retained-*` 与原件比对（`git range-diff`）后删重复；删除前 `git worktree list` 解除 4 个 worktree 占用。
11. **存储分层落地**：`experiments/runs` 的 27.6GB .gz 交付包移外部冷存并写索引；`mne_data` 92GB、`cloud_inputs` 2GB 冷存/删除；venv 二选一（预计释放 100GB+）。
12. **建立最小 CI**（哪怕只是本地 pre-push hook）：`ruff check` + `pytest -q` + `git diff --check`，防止"HEAD 不可导入"再次发生。

---

## 附录 A：证据命令

```powershell
git status --porcelain=v1 -uall | Group-Object { $_.Substring(0,2).Trim() }  # 80/118/231
git ls-tree -r HEAD --name-only | Measure-Object -Line                       # 570
git ls-files --eol | Group-Object { ($_ -split '\s+')[0..1] -join ' ' }      # CRLF 分布
git ls-files -i -c --exclude-standard                                         # 空 = 无误忽略
git count-objects -vH ; git fsck --unreachable --no-reflogs | Measure -Line    # 7.26GiB / 2144
git rev-list --objects --all | Measure-Object -Line                            # 1062 可达
git branch -a ; git reflog --date=short                                       # 72 refs / 239 条
```

## 附录 B：方法与局限

- 全部结论来自当前工作树实测；未修改任何既有文件，未运行 git 写命令、未安装依赖、未启动训练。唯一写入：`tmp/audit/09-repo-management.md` 与 `tmp/audit/scratch/*.py`（3 个只读检查脚本）。
- 未全量遍历 41GB/98GB/92GB 目录（避免长任务）；这些层引用 Lead 实测值，并在 §4.1 给出可复现的验收命令。
- 链接检查脚本排除了指向源码行的 `D:/...py:NNN` 形式（49 条真实断链中已剔除这类），因此与"32 条 frozen 断链"口径一致。
- manifest 抽查受两种 schema 混杂影响，§4.2 已说明为何 1411/5919 不能当缺陷率。

