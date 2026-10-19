# V1 复核报告（红队逐条验证）

复核对象：tmp/audit 下 12 条工程/仓库/测试高影响结论。仓库 D:\BCI\n2p3-net，工作树（非 HEAD）。
复核时间：2026-10-19 08:16–08:40 (+08:00)。全程只读（read/grep/git 只读/ruff/uv lock --check/pytest 指定文件），仅写入 tmp/audit/ 与本报告。
可重复性注意：复核期间工作树被并发写入（doc/assets 文件数在两次测量间由 3514 变为 3529；出现 doc/evidence/repository_audit_20261019.zh.md）。涉及时点的计数已注明；结论未依赖单次快照的方向性判断。

## 总览

| # | 主题 | 判定 |
|---|---|---|
| 1 | 23 个未跟踪 src 模块 + 已跟踪模块 import 它们 | 成立（但严重低估：27 条边/14 文件） |
| 2 | 80 M / 118 D / 230 ?? 等精确计数 | 部分成立（?? 为 231，总 429） |
| 3 | CRLF/mixed、自哈希、跨平台 resume | 成立（数量对；"11 模块 import 自哈希"应为 8；影响点需改写） |
| 4 | .git 8.04GB / pack 205MB / loose 7.26GiB / unreachable 2144 / 大对象 | 成立（2144 需 --no-reflogs；默认 1812） |
| 5 | 全量 pytest 3F/2331P/8S，根因 fixture 维度 | 成立（3 个失败与根因复现；耗时未重跑） |
| 6 | HEAD 1189/1188，当前 1593，130 消失，119 无替代 | 成立（119 依赖 difflib≈0.83 口径） |
| 7 | Paper 59 PDF/doc/assets/frozen manifest/断链 | 部分成立（54 PDF；89 manifest；9→6 被引用；31/17/28/25） |
| 8 | experiments/runs 41GB 结构 | 成立（路径须写 experiments/runs；"2183 副本"不成立） |
| 9 | 两个 venv torch、uv、lock、required-version | 部分成立（uv 0.11.21 而非 0.11.26） |
| 10 | 打包 9 个顶层包 + source_records 路径 | 成立（namespace 发现 9 个；wheel 下必然 Missing） |
| 11 | ruff/lock/format/marker/addopts | 部分成立（format 280 而非 273；-q 抑制统计行属实） |
| 12 | README 17+1 脚本、8 未索引 evidence、6 无入链 assets | 部分成立（脚本与 evidence 对；无入链实测 9） |

## 逐条

### 1) 未跟踪 src 模块与被跟踪模块的 import（结论：成立，但低估）
证据：`git status --porcelain -uall` 下 src/ 未跟踪 24 个文件 = 23 个 .py + structured_erp.LICENSE（与"23 模块"一致）。用脚本把 23 个模块转成点分名后扫描 97 个已跟踪 src/*.py：**27 条 import 边、来自 14 个已跟踪文件、指向 11 个未跟踪模块**，远多于结论列的 3 个文件。`git show HEAD:<file>` 对全部 14 个文件重扫：**0 条**。
- target_experiments.py:291/421/486 属实（291 source_matrix、421 comparison、486 source_matrix）；binary.py:19 comparison 属实。
- 结论写 "stats/transfer.py:13" → 实际 `src/stats/transfer.py:12`（:13 是 `from stats.hierarchical import paired_interval`）。
- 结论漏掉的边示例：calibrated.py:23/24/51、calibration_candidates.py:14/16/30、epoch_selection.py:9/30、subject_adapter.py:24/31、target_experiments.py:109/116/290、recipes.py:11、checkpoint.py:81、models/n2p3net.py:14、data/continuous_qc.py:21、data/recovery.py:253/276、research/source_selection.py:147。
修正措辞："工作树有 23 个未跟踪 src 模块（24 个文件），被 14 个已跟踪模块以 27 条 import 边引用（含 target_experiments 291/421/486、binary 19、stats/transfer 12）；HEAD 版本这 14 个文件均为 0 条"。

### 2) 工作树计数（结论：部分成立）
`git status --porcelain=v1`：429 条 = 80 ` M` + 118 ` D` + **231** `??`（结论 230；总 429 正确）。`-uall` 时未跟踪文件为 4204 个（目录折叠口径才是 231）。
- tests 删除 109 ✓；frozen 删除 manifest 9 ✓（`manifest` 名匹配的第 10 个是 tests/test_manifest.py，非 frozen）。
- experiments 顶层 py 19/69 ✓（逐个列名核对）；configs 4/9 ✓（brainsync_bidirectional_preprocessing、cloud_training_resources、source_pool_expansion.example、unified_training_budget）。
修正："231 未跟踪目录项（-uall 4204 文件）"。

### 3) 行尾/自哈希/跨平台 resume（结论：成立，但机制需改写）
- `.gitattributes` 无 `* text=auto` ✓；`core.autocrlf=true`（用户全局）。
- 按 git 自身文本判定（`git grep -Il ""` → 371 个文本文件）：**137 CRLF / 69 mixed / 158 LF / 7 none**，与结论完全一致（按字节启发式会得 382 文件/137/75，故数量依赖 git 文本口径）。
- **import 时自哈希的模块是 8 个**：epochs.py:46、preprocess.py:32、continuous_decision.py:36、exact_covariance.py:53、information_bound.py:79、noise_model.py:19、operator_lda.py:27、temporal_noise.py:34。另有 3 处在运行期哈希自身源码：source_records.py:98-99（`src/data/*.py` 全部 43 个 + runner）、recording_quality.py:239、experiments/analyze_brainsync_recovery.py:134 → 结论"11 个模块 import 时"不准确，应为"8 个 import 时 + 3 个运行期"。
- CRLF/LF 哈希确实不同：operator_lda.py raw `41e555fd…` vs LF `c722eedc…`；epochs.py `dfe5347d…` vs `4ac99688…`（另有 noise_model/continuous_decision 同样不同；temporal_noise/exact_covariance/information_bound 当前纯 LF，相同）。
- **实际严格比较点（决定"是否会让 resume 失败"）**：`experiments/prepare_source_records.py:117` 全等比较 `result["preparation_method"] != method`，method 含 `data_modules_sha256`（43 个 src/data/*.py，其中 16 个当前 CRLF/mixed）与 runner_sha256；不一致即 `ValueError: source preparation method changed...`。同型比较见 source_records.py:207/356/565/442/457/464/483/502/692/826。另有 temporal_noise.py:219-220 严格相等（"fitted by a different implementation"）。
- 反面：epochs.py:560 对 receipt 的 preprocess.py/epochs.py 摘要**只校验 64-hex 格式**，PREPROCESSING_DECLARATION_SOURCE_SHA256 只写不比较；noise_model/operator_lda/continuous_decision 的 implementation_sha256 也只过 require_sha256 格式校验。
修正："8 个模块在 import 时自哈希；跨平台失败的具体闸门是 source preparation method（data_modules_sha256+runner_sha256）与 temporal noise 的 implementation_sha256；epochs/preprocess 摘要不参与相等比较"。

### 4) .git 体量与不可达对象（结论：成立，口径需补）
`git count-objects -vH`：count 11302 / size 7.26 GiB；in-pack 3910 / packs 6 / size-pack 204.83 MiB；prune-packable 1145。`.git` 合计 8.04 GB（其中 objects 7,648.5 MB，另有 github-preparation-20260907.bundle 586.2 MB、5 个 linked worktree）。
- `git fsck --unreachable`（默认含 reflog 根）= **1812**；`--no-reflogs` = **2144**（blob 1524/tree 258/commit 30 量级）。结论 2144 只在 --no-reflogs 口径成立。
- 最大 5 个 loose 对象：1218.8 / 893.8 / 452.7 / 450.7 / 353.6 MB，与结论一致。前 4 个为 PK zip，均在 fsck 不可达集合 → 可回收（loose 不可达；需 `gc --prune=now`，默认 2 周过期）；353.6 MB 那个**可达**（`frozen/source_feature_transfer_c7bf860.source-only.tar.gz`）→ gc 不能回收。默认口径下 loose 中不可达 1770 个/6.649 GiB。
修正："unreachable=1812（默认）/2144（--no-reflogs）；4 个最大 zip（约 3.0GB）不可达可 gc 回收，354MB 的 tar.gz 已提交不可回收"。

### 5) 全量 pytest（结论：成立）
复跑 `pytest tests/transfer/test_preprocessing_execution_gate.py -q`：**3 failed**，均为 `ValueError: calibrated LDA feature state has invalid shape or scale`（src/transfer/calibrated.py:661）。
根因实测：`binary_dataset(8)` → n_channels=3、n_times=100、sfreq=100；测试 :70-72 造 `n_channels*n_times=300` 维，而 calibrated.py:652-655 期望 `selected_count*downsampled_time_points(100,100,20)=3*20=60`。
总数：`pytest --collect-only -q` 合计 **2342** = 3F+2331P+8S，与结论自洽；439.48s/22 warnings 取自 tmp/audit/pytest-full2.log，未重跑全量。
修正："失败断言为 ValueError（calibrated.py:661），fixture 300 维 vs 期望 60 维（3 通道×20Hz 降采样 20 点）"。

### 6) 测试重组（结论：成立，方法依赖）
HEAD tests/ 110 文件：`def test_` **1189 次 / 1188 唯一**（重名 test_predict_proba_range）；当前磁盘 185 文件、**1593 唯一**；HEAD 名字消失 **130**。删除 tests 文件 109。
"119 无近似替代"：difflib SequenceMatcher 对每个消失名取当前最佳匹配，**cutoff 0.83 时恰为 11 有近似 / 119 无**；cutoff 0.80→17/113，0.75→32/98，0.70→47/83。属可复现但口径敏感。
修正："以 difflib≥0.83 为近似判据时 11 个疑似改名、119 个无近似替代；放宽到 0.80 则为 17/113"。

### 7) Paper/doc/assets/frozen/断链（结论：部分成立）
- Paper/：**54 个 PDF / 191.4 MB**；Paper/ 下已跟踪文件 **59**（5 个非 PDF）→ "59 PDF" 是把已跟踪文件数当成 PDF 数。
- doc/assets：172.2 MB；文件 3514（复核中途 3529，并发写入；203 个目录）；结论 3610 ≈ 文件+目录或旧快照，173MB 合理。
- frozen：100 个目录 ✓；含任一 `*manifest*.json` 的 **89** 个（结论 88，±1；其中 11 个完全没有 manifest，55 个恰为 manifest.json）。
- 9 个被删 manifest：**6 个**在已跟踪文本中有引用（brainsync_notch、quality source/frontend、recovery frontend、technical_audit_before/_app），**3 个零引用**（brainsync_producer_before_eog、source_feature_transfer_c5d670d、c7bf860）→ "9 个被引用"不成立。
- 断链：doc/*.md 中 **31 条 / 17 文件 / 28 唯一目标 / 25 个从未在 HEAD**（含根 README、Paper 为 34/20/28/25）；结论 32/18/27/24 相差 ±1–3，方向一致。已跟踪文档的 7 条断链全部指向被删的 frozen manifest。
修正："Paper 54 PDF/191MB（59 个已跟踪文件）；frozen 89/100 有 manifest；被删 manifest 仅 6/9 被引用；断链约 31 条/17 文件"。

### 8) experiments/runs（结论：成立，路径与"2183"需修正）
路径是 **experiments/runs**（仓库根无 runs/；根 runs 不存在曾致首轮测量为 0）。实测 **40.98 GB / 40956 文件 / 4398 目录**：.gz 26.93GB(96)、npz 5.43GB(17312)、tar 4.31GB(11)、pt 2.29GB(4028)、json 1.53GB(9780)、py 8134、pyc 614、>20MB 122 个；最大 4.51GB `experiments/runs/source_eog_20260909/delivery/BI2013a.tar.gz`（"~4.6GB"偏高，另有 4.48GB Huebner、4.06GB BI2015a）。
"8134 py（2183 逐 run 复制）"：8134 ✓，但 **2183 无法复现**——py 去重后 1174 种内容、冗余副本 **6960**、3305 个与仓库 src/experiments 源码逐字节相同。
修正："py 8134 个，仅 1174 种内容（冗余副本 6960；3305 与仓库源码完全相同）"。

### 9) 环境（结论：部分成立）
.venv：torch **2.13.0+cpu**，0.98 GB；.venv-gpu：**2.13.0+cu130**，3.17 GB（两者 Python 3.14.3）。uv：**0.12.9**（hermes）与 **0.11.21**（winget）→ 结论 "0.11.26" 不成立。uv.lock：torch 2.8.0（L3340）与 2.13.0（L3379）并存 ✓，torchaudio 同样两处；pyproject 无 `[tool.uv]`/required-version ✓。
修正："uv 0.12.9 vs 0.11.21"。

### 10) 打包（结论：成立）
`src/n2p3_net.egg-info/top_level.txt` = audit_p300, baselines, data, measurement, models, research, stats, train, transfer（9 个）。实测 `find_namespace_packages('src')` 恰为这 9 个（audit_p300/measurement 现为空目录），`find_packages` 只有 7 个 → setuptools 的 namespace 语义确会发布 9 个通用名。
source_records.py:98 在 site-packages 下只会 glob 到已安装模块；:99 `directory.parents[1]/"experiments/prepare_source_records.py"` 在 wheel 中解析为 `<venv>/Lib/experiments/...` → `file_sha256` 抛 FileNotFoundError（且 data_modules_sha256 与仓库内 43 文件版本必不相同）。触发点 `source_preparation_method`（source_records.py:91）。

### 11) 工具链（结论：部分成立）
ruff 0.16.5：`ruff check .` "All checks passed"，`--show-files` **399** ✓；`uv lock --check` 成功（Resolved 151 packages）✓；`ruff format --check` → **280 files would be reformatted**, 247 already（结论 273，已漂移）；全树 marker 只有 parametrize(3698)/skipif(93) ✓；pytest 9.1.1 + addopts=-q **确实不打印最后统计行**（通过和失败两种情形都验证：只留 FAILED 行，无 "N failed/N passed" 行）。收藏 2342 个测试。
修正："ruff format --check 280 个待格式化"。

### 12) 文档入口/索引/资产入链（结论：部分成立）
README 入口表 17 个脚本全部存在于 experiments/ ✓（缺 0），dashboard_server.py ✓ → 18 ✓。
doc/evidence：111 个 md，**8 个未索引** ✓（algorithm_chain_repairs_20260916、brainsync_transport_unification_20260917、code_quality_audit_20260908、e2_prior_scale_20261006、source_modules_20260918、unified_budget_execution_20260916、unified_compact_training_20260916、unified_model_implementation_20260916）。
assets 入链：60 个顶层目录，按"解析后的 markdown 链接指向该目录且来源不在目录内"实测 **9 个无入链**（结论 6）：chain_improvements_20260929、**lean_current_checks_20261018（1149 文件/60.4MiB，与结论完全一致）**、lean_selection_erp_20261018、lean_v06_risk_20261018、**operator_research_202604（空目录 ✓）**、progress_report_20260929、repository_audit_20261019、unified12_results_20260918、unified_budget_execution_20260916（后两个可能是并发新建/文档变更）。
修正："无入链 assets 目录实测 9 个（含 lean_current_checks 60.4MiB/1149 文件、operator_research_202604 空）"。

## 未能验证/边界
- 439.48s 全量耗时与 2331 passed 分布：未重跑全量（只复现 3 个失败 + collect 2342）；数值来自 tmp/audit/pytest-full2.log。
- uv "0.11.26"：本机两个 uv 为 0.12.9/0.11.21，仓库文本无 0.11.26。
- "2183 逐 run 复制"：三种可能口径（与仓库源码相同 3305、内容重复副本 6960、实验脚本同名副本 1169）均非 2183。
- doc/assets 精确文件数随时间变化（3514→3529），并发写入导致 ±1% 漂移。
