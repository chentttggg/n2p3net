# V3 独立复核（red team 逐条验证）
- 复核对象：tmp/audit/00-consolidated.md 与 09/11/13 等分册中的 12 条高影响结论。
- 复核时间：2026-10-19（工作树状态，非 HEAD）；仓库 D:\\BCI\\n2p3-net。
- 只读边界：仅 read/grep/glob、只读 git（log/fsck/for-each-ref/ls-files/cat-file/reflog/bundle list-heads）与 python 只读脚本；未跑任何 git 写命令、未装依赖、未训练。
- 数值口径：GiB=1024^3，GB=10^9；两者在原文中混用，下面逐条标注。

## 判定总表
| # | 主题 | 判定 |
|---|------|------|
| 1 | main 停在 f1a303c、66 分支、10 月工作在 codex/* | 成立（口径需注明） |
| 2 | 112 dangling commit、gc 即销毁 | 部分成立（82 条受 reflog 保护） |
| 3 | .git 8.04GB、205MB pack、7.26GiB loose、可回收 | 成立（补：可回收 6.65GiB） |
| 4 | 14 个已跟踪文件 import 10 个未跟踪模块 | 成立（精确复现） |
| 5 | tests 110 tracked / 109 删除 / 185 磁盘 | 成立（精确复现） |
| 6 | frozen 98GB apparent、3 junction、3.3GB 自有、87 manifest | 部分成立（95.4GiB；89/100 或 53/100） |
| 7 | 已提交文件含账户名/hostname | 成立 |
| 8 | dashboard 零鉴权 + StrictHostKeyChecking=no | 成立（需限定 loopback 与"头非鉴权"） |
| 9 | formal 未跟踪、414 文件/3.63MB、2890 定理 | 部分成立（2890 为自报字段） |
| 10 | 1528/753 断链、189/6 哈希漂移 | 成立（今为 1530/754） |
| 11 | Lean 工具链硬编码、12/39、55/93 | 成立 |
| 12 | LICENSE 缺失、3 文件未跟踪 | 成立 |

## 1) main 停在 f1a303c，此后 main 0 提交；66 分支；10 月工作主要在 codex/*
**结论摘要**：主干确实冻结在 2026-09-09。
**判定**：成立（"66 个分支"需注明口径）。
**证据**：
- `git log -1 main` = f1a303c42c68 2026-09-09T09:58:12+08:00；`git rev-list --count f1a303c..main` = 0；main 共 210 提交。
- refs/heads = 62，refs/remotes = 4，`git branch -a` = 66，tags = 0 → "66 分支"只在 branch -a 口径成立，本地分支实为 62。
- `--sort=-committerdate` 首位 codex/research-director-resume-20261013 2026-10-19 00:05:17，与原文一致。
- 10 月：全部 ref 627 条提交，main 0 条；54 个分支有 10 月提交，其中 52 个是 codex/*；工作树 4417 条变更（80 M / 118 D / 4219 ??，--untracked-files=all）。
**修正措辞**：把"66 个分支"改为"62 个本地分支 + 4 个远程跟踪分支（branch -a 计 66）"；其余可原样保留。

## 2) 112 dangling commit、最新 e148163、gc --prune=now 即销毁
**结论摘要**：数量与"不在任何 ref 上"成立；"任何 gc 即销毁"夸大。
**判定**：部分成立。
**证据**：
- `git fsck --unreachable --no-reflogs` = 2144 对象 = 112 commit + 431 tree + 1601 blob（与第 3 条同源）。
- 最新 e14816335ffd 2026-10-14T20:17:17+08:00，最旧 2026-08-25；按提交月：10 月 53、9 月 48、8 月 11。
- `git branch -a --contains e148163` = 0，`git tag --contains` = 0，`git for-each-ref --contains` 空 → 确实不在任何 ref 上。
- 但 `git fsck --unreachable`（带 reflog）= 1813 对象 / 30 commit：差额 331 对象 / 82 commit 仅由 reflog 保护；e148163 正是 `codex/source-lineage-storage@{2026-10-14 20:17:17}`。
- 真正无任何保护的是 30 条 commit（最新 451af9d1 2026-10-12，10 月 26 条），它们会被 `gc --prune=now` 立即回收；82 条会在 reflog 默认 30 天过期（约 2026-11-13）后的下一次 gc 才消失。
- 仓库内 .git/github-preparation-20260907.bundle（586MB，18 个 ref，main 指向 f9d8eab）只覆盖 9 月 7 日，不能兜底 10 月工作。
**Rescue（写命令，未执行）**：先落 ref 再 gc：
```
git fsck --unreachable --no-reflogs 2>$null | Select-String '^unreachable commit' |
  ForEach-Object { $h=($_.Line -split ' ')[-1]; git tag "rescue/$h" $h }
git config gc.pruneExpire never        # 或 git -c gc.pruneExpire=never gc
git bundle create tmp/audit/rescue-20261019.bundle --all   # 含新 tag，落到仓库外更佳
```
**修正措辞**："112 条不可达提交中 30 条已无 ref/reflog 保护，`gc --prune=now` 即销毁；其余 82 条仅靠 reflog 续命，reflog 过期（默认 30 天）后同样丢失；最新 e148163 属后者。"

## 3) .git 8.04GB 构成与"gc 可回收"前提
**结论摘要**：体积、包/松散分布、大对象类型、reflog 保护数均可复现。
**判定**：成立（补两个限定）。
**证据**：
- .git 全量 8,636,018,055 B = 8.04 GiB / 11,522 文件；pack 214,830,649 B（20 个 pack）= 204.9 MiB；loose 7,805,157,293 B = 7.27 GiB / 11,302 个。
- 前 5 大 loose 对象（磁盘字节）：1.278 / 0.937 / 0.475 / 0.473 / 0.371 GB；对象头解压后分别为 `PK\x03\x04`×4 与 `1f8b08`×1（ZIP/GZIP），与原文一致。
- 限定一：第 5 大（0.371GB，GZIP）不在 unreachable 集合，它被某个 ref 可达，不会随 gc 回收；前 4 大（3.16GB）才是不可达。
- 限定二：unreachable 对象占 loose 6.664 GiB（91.7%），而 reflog 保护的 331 个对象仅 15.8 MB → `gc --prune=now` 实收约 6.65 GiB，.git 预计从 8.04 GiB 降到约 1.0–1.4 GiB（另含 586MB bundle，可另行处理）。
**修正措辞**："前 5 大对象 3.53GB（4 个 ZIP、1 个 GZIP），其中第 5 大可达、不可回收；可回收上限 6.65GiB，而非全部 7.26GiB。"

## 4) 14 个已跟踪 src 文件 import 10 个未跟踪模块
**结论摘要**：数字精确，示例全部命中。
**判定**：成立。
**证据**：
- 脚本解析 src 下 97 个已跟踪 py、120 个磁盘 py、23 个未跟踪 py；14 个 importer、10 个未跟踪目标，恰好等于原文。
- 目标：baselines/{structured_erp,temporal_transfer}.py、data/{bidirectional,source_records}.py、models/parameter_groups.py、research/{comparison,source_catalog,source_matrix}.py、train/early_stopping.py、transfer/source_objective.py。
- 示例核对：models/n2p3net.py→models.parameter_groups ✓；transfer/calibrated.py→baselines.structured_erp（另含 temporal_transfer、train.early_stopping）✓；baselines/deep.py→train.early_stopping（另含 transfer.source_objective）✓；stats/transfer.py→research.comparison ✓。
- HEAD 侧：10 个目标在 HEAD 全部不存在（`git cat-file -e HEAD:<p>` 全 rc≠0），14 个 importer 的 HEAD 版本 import 命中 0。
**修正措辞**：可保留；建议补一句"HEAD 自身闭合，工作树因新增模块而不可导入"。

## 5) tests：110 tracked、109 删除、185 磁盘 / 184 未跟踪
**结论摘要**：完全一致。
**判定**：成立。
**证据**：`git ls-files tests`=110（全为 .py），`--deleted`=109；磁盘 185 个 py（根级仅 tests/__init__.py，且状态 M），`--others --exclude-standard` = 184。tests/ 下共 9 个子目录、370 个文件。
**修正措辞**：可原样保留；"只剩 tests/__init__.py"应说明它是被修改（M）而非干净保留。

## 6) frozen 98GB apparent、3 junction、3.3GB 自有、87/100 manifest
**结论摘要**：junction 数量/指向、自有体积成立；apparent 与 manifest 计数口径需修正。
**判定**：部分成立。
**证据**：
- 3 个 reparse point：bidirectional_qc_20260909_acceptance\\BrainSync_Tool_newest → D:\\BCI\\BrainSync_Tool_newest（1.577 GiB / 10,375 文件）；同目录 checkout\\.venv → D:\\BCI\\n2p3-net\\.venv（0.979 GiB）；checkout\\mne_data → D:\\BCI\\n2p3-net\\mne_data（89.563 GiB / 77,770 文件）。与原文 1.58GB / 89.6GB 一致（GiB 口径）。
- frozen 自有（不跟随 junction）= 3.253 GiB / 2,343 文件；PowerShell 独立复核 3.32 GiB → "约 3.3GB"成立。
- apparent（跟随 3 个 junction）= 95.44 GiB = 102.47 GB，非 98GB；若再扣除 .venv 则 94.4 GiB。
- manifest：100 个顶层目录；顶层有 manifest.json 的 53 个；任意层含 manifest.json 的 56 个；含任意 *manifest*.json 的 89 个。87 无法用单一标准复现。
**修正措辞**："frozen apparent 跟随 junction 为 95.4GiB（102.5GB），自有 3.25GiB；100 个归档目录中 53 个顶层有 manifest.json、89 个含任一 manifest 变体。"

## 7) 已提交文件含机器身份
**结论摘要**：位置与数量均核实，且都在 Git 追踪/历史中。
**判定**：成立。
**证据**（不打印具体值）：
- doc/assets/code_quality_repair_20260908/validation.json 第 29 行含 2 处 `C:/Users/<账户>/` 绝对临时路径 + 账户名；该文件 2 次提交、存在于 HEAD。
- doc/assets/technical_audit_20260906/ 下 8 个 JUnit xml 全部含 hostname（8/8），均存在于 HEAD（各 1 次提交）。
- frozen/*/manifest.json：被跟踪的 106 个中 3 个含 hostname（brainsync_qc_remaining_20260908、brainsync_reference_20260908 及其 post_merge 变体）。
- 全库被跟踪文件扫描（570 个）：11 个含 hostname、3 个含账户名。
**修正措辞**：可保留；建议写明"hostname 命中 11 个被跟踪文件，账户名 3 个"。

## 8) dashboard 零鉴权 + StrictHostKeyChecking=no 链路
**结论摘要**：鉴权分支、端点、extra roots、两处 ssh 选项全部核实；需补默认绑定与"头不是鉴权"两点。
**判定**：成立（带限定）。
**证据**：
- `do_POST`（:506-510）白名单仅 /api/kill、/api/monitor/{claim,heartbeat,release}；/api/monitor/* 在 :511-539 直接 return，不校验任何头；/api/kill 在 :540 只校验 `X-Dashboard-Confirm: yes`（可任意伪造，等于无鉴权）。
- /api/monitor/claim 会 `subprocess.Popen(watch_resources.py ...)`（:425），即未鉴权的进程启动面；`_run_dir` 用 RUN_NAME 正则（:20/:29）拒绝 `..`，无路径穿越。
- `translate_path`（:326-338）把 `runs/tmp/**` 映射到仓库 `tmp/`，且要求候选严格位于 tmp 下；`extra_run_roots={"tmp": <repo>/tmp}`（:589）→ 整个 tmp/ 可 GET。
- 静态根是 `--directory experiments`（:590），SimpleHTTPRequestHandler 默认目录列表 → experiments/ 全域（runs、checkout、logs）也可读，暴露面比"tmp/"更大。
- `--bind` 默认 127.0.0.1（:579），远端启动也用 `--bind 127.0.0.1`（open_dashboard.ps1:43）；对外仅经 SSH 隧道 -L 18812:127.0.0.1:8812。
- open_dashboard.ps1 第 46 行与第 98 行确为 `StrictHostKeyChecking=no`（两处）。
**修正措辞**："服务无任何身份校验：/api/kill 仅需可伪造的 X-Dashboard-Confirm，/api/monitor/* 连该头都不需要；默认只绑 127.0.0.1，因此 MITM 链的前提是攻击者能在 SSH 隧道建立时冒充云端主机（StrictHostKeyChecking=no 使其可行），随后以同源页面调用 /api/kill 并读取 runs/tmp。"

## 9) formal 未跟踪、414 文件/3.63MB、2890 定理
**结论摘要**：未跟踪与 414/3.63MB 成立，但"414 文件"只是 git 可见子集；2890 是自报字段。
**判定**：部分成立。
**证据**：
- `git ls-files formal` = 0；`git ls-files --others --exclude-standard formal` = 414 文件 / 3,631,683 B（3.63MB）= 385 个 .lean，其中 37 个 Audit*.lean（不是 39）。
- 磁盘上 formal/ 实为 32,699 文件 / 2.53 GiB / 8,419 个 .lean；差额被 formal/lean/.gitignore 排除（.lake 构建缓存、artifacts）。"2.67GB 缓存未入库"是同一事实的另一表述。
- doc/assets/lean_* = 25 个目录 / 2,371 文件 / 103,624,785 B（103.62MB，98.8MiB），0 个被跟踪 —— 与原文完全一致。
- 2890 出自 doc/assets/lean_gaussian_20261016/active_checkpoint.json 的 `accepted_theorems`（status=paused），是 checkpoint 自报，无逐条清单可核；源码 theorem 声明实测 3,348（N2P3）+14（根）+56（negative）=3,418 条（正则口径，审计文档记为 3342）。
**修正措辞**："formal/ 零跟踪；git 可见源码 414 文件/3.63MB（385 .lean、37 Audit*.lean），磁盘含缓存共 32,699 文件/2.53GiB；'2890 条已验收定理'是 checkpoint 自报字段，源码有约 3.4k 条 theorem 声明，二者无机器可读映射。"

## 10) Lean 证据链：1528/753 断链、189/6 哈希漂移
**结论摘要**：两组数字均可复现，漂移文件完全对得上。
**判定**：成立（当前为 1530/754）。
**证据**：
- 复跑原作者脚本 tmp/audit/scratch/p13_refs.py：`ARTIFACT_REFS 1530 OK 776 BROKEN 754`（49.3%）；口径为 doc/**/*.md+*.json 中**去重后**的 `formal/lean/artifacts/...` 路径，不是出现次数（按出现次数为 2448/1128）。
- 1528/753 与 1530/754 差 2/1，应为报告写就后新增引用；49% 结论不变。
- formal/lean/scripts/verify_erp.py:33 `EXPECTED_RUNTIME_SOURCE_COUNT=189`；acceptance.json 的 source_sha256 恰 189 条；按 verify.digest（sha256 of bytes）重算 6 条不符：src/baselines/features.py、src/baselines/structured_erp.py、src/stats/transfer.py、src/transfer/calibrated.py、src/transfer/exact_covariance.py、src/transfer/operator_lda.py。
- 失败判据核实：:151-153 `if digest(REPO/name) != h: raise ValueError("Production source changed: ...")`，在调用 Lean 之前即抛错；且脚本要求 `--output` 目录不存在（:113），重跑必失败。
**修正措辞**："当前 1530 条引用/754 条断链（49.3%）；189 条钉死源码中 6 条漂移，verify_erp 在 :153 抛 'Production source changed'，整条最严格验收链已不可重跑。"

## 11) Lean 工具链硬编码、Audit/negative 入口缺口
**结论摘要**：三处数字全部核实。
**判定**：成立。
**证据**：
- run_lean.py:415 `--lake` 默认 `D:/BCI/lean-tools/lean-4.24.0-windows/bin/lake.exe`；:427 无条件 `env["MATHLIB_CACHE_DIR"]="D:/BCI/lean-tools/mathlib-cache"`（无 CLI/env 覆盖路径）；同类硬编码另见 verify_erp.py:94。
- Audit*.lean 共 39（33 根级 + 4 个 N2P3/ + 2 个 artifacts/ 下）；27 个脚本中出现 12 个；12 个里 11 个由 verifier 运行时 `write_text` 生成（如 verify_erp.py:186 生成 AuditERP.lean），Audit.lean 为手写 → 27 个无入口。
- negative 93 个 .lean；脚本引用 38 个，55 个无引用。
**修正措辞**：可保留；补"另一处硬编码在 verify_erp.py:94，且 MATHLIB_CACHE_DIR 无覆盖开关"。

## 12) 根 LICENSE 缺失、三文件未跟踪
**结论摘要**：全部核实，且 pyproject 连 license 字段都没有。
**判定**：成立。
**证据**：
- `Test-Path LICENSE` = False，`git ls-files LICENSE` rc=1（历史上也从未跟踪）。
- src/baselines/structured_erp.py（12,786 B）、temporal_transfer.py（13,362 B）、structured_erp.LICENSE（1,635 B，第三方版权头）均在磁盘存在，`git ls-files` 全部 rc=1，HEAD 中同样不存在。
- pyproject.toml:29 `baselines = ["structured_erp.LICENSE"]` 声明为 package-data；全文无 `license` 字段。
- transfer/calibrated.py、calibration_candidates.py、recipes.py 等确实 import 这两个模块（见第 4 条清单）。
**修正措辞**："根 LICENSE 缺失且从未跟踪；pyproject 既无 license 元数据又把未跟踪的第三方 LICENSE 声明为 package-data；structured_erp/temporal_transfer 及其许可证三件套未跟踪，而工作树代码已依赖它们。"

## 复核附注
未验证/无法验证：2890 条定理的"已验收"身份（无逐条清单）、98GB 与 87 manifest 的原始取数方式（当前不可复现）。其余偏差均为口径/时点差异，未发现捏造；第 2、6、9 条需改写措辞，其余可保留。