# 工作区与文件夹管理审计及清理方案

- 日期：2026-10-19（Asia/Shanghai）
- 范围：`D:\BCI\n2p3-net`（当前工作树）及 `D:\BCI` 下的 artifacts / worktrees 等本地目录
- 约束：审计与清理**不修改任何源文件/测试/配置/指南**；数据处置遵循“先证据、后行动”；不可逆操作先备份/清单化。
- 方法：junction 感知的目录遍历；>20 MiB 文件 SHA-256 去重；归档成员与 `mne_data` 逐项（文件名+大小）比对；Git 对象可达性/年龄分析；代码路径引用扫描（`mne_data` 等）；所有结论可复算。
- 本次已执行的清理记录：`tmp/audit/cleanup_executed_20261019.json`、`tmp/audit/cleanup_partials_20261019.json`。

## 0. 结论摘要

1. **磁盘大头不在源码，而在数据与结果**：仓库 apparent 约 **147.5 GiB**（清理后）；其中 `mne_data` 89.56 GiB、`experiments` 44.97 GiB（`runs` 40.98 GiB）、`.git` 0.59 GiB（清理前 8.64 GiB）、frozen 3.32 GiB、两个 venv 4.15 GiB、`formal/lean/.lake` 2.50 GiB、`cloud_inputs` 1.98 GiB。整个 `src+tests+configs+doc` 只有约 0.2 GiB。
2. **字节级重复几乎为零，真正的冗余是语义与结构性的**：>20 MiB 文件 159 个/43.73 GiB，SHA-256 完全相同组 **0**；但 `experiments/runs/source_eog_20260909/delivery` 的 **20.80 GiB** 是对 `mne_data` 原始数据的重新打包（9 个归档逐成员文件名+大小 **100%** 匹配，其中 6 个与已解压文件完全对应、4 个与 mne_data 内嵌 zip 成员完全对应），并配有逐记录 SHA-256/原始路径的 `n2p3_source_records/1` 清单——属于可重建的冗余。
3. **Run 目录缺少保留策略**：52 个 run、40 956 文件；仅 16/52 有任意 manifest，9/52 同时有源码快照字节+哈希声明；8 134 个 `.py` 中仅 1 174 个唯一内容（6 960 个重复副本）、614 个 `.pyc`；导出/中间层重复叠加。
4. **本次已实际回收（D: 可用从 79.4 → 94.2 GiB）**：`git gc` 回收 **7.39 GB**（.git 8.64 GB → 0.59 GB 含移出历史 bundle）；删除 5 个 `.npz.gz.partial` 未完成副本 **12.38 GiB**（checkpoint 记录 `full_archive_verified=false`，原件在云端冷存）；清理 `.pyc`/缓存 **56.4 MB**。**未删除任何完成态科研数据/源码**。
5. **仍需用户批准的大项（合计约 80–95 GiB）**：delivery 归档 20.80 GiB（可重建）、不活跃原始数据 37.4 GiB（lee2019/BI2014b/llp/RomaniBF，公开可重下载）、run 派生缓存/导出/`local_prepared` 约 17 GiB、worktree 临时产物约 5–7 GiB、其他 artifacts/venv/.lake 等约 8 GiB。

## 1. 当前磁盘与目录快照

### 1.1 卷与 D:/BCI

| 位置 | 大小 | 说明 |
|---|---|---|
| D: 卷 | 551.6 GiB，已用 457.4 GiB，可用 **94.2 GiB**（清理后） | 清理前可用 79.4 GiB |
| `D:/BCI/n2p3-net` | ~147.5 GiB apparent（含 mne_data，不含 92 GiB junction） | 代码+数据+结果+归档混在一个目录 |
| `D:/BCI/artifacts` | 清理前 19.41 GiB → 清理后约 11.5 GiB | 含 git-unreachable 3.30 GiB、备份 bundle 0.59 GiB、apps 1.36 GiB、research-deliveries 1.88 GiB |
| `D:/BCI/worktrees` | 7.53 GiB | bi2014a-label-prefix 5.61、research-director 0.88、huebner 0.57、joint-flat 0.47 |
| `D:/BCI/lean-tools` | 2.40 GiB | Lean 工具链 |
| `D:/BCI/BrainSync_Tool_newest` / Release | 1.58 + 0.41 GiB | 独立采集端项目 |
| `D:/BCI/n2p3-audit-archive` / NRH-01 | 0.53 / 0.96 GiB | 历史归档 |

### 1.2 仓库内部（junction 目标不计入）

| 目录 | 大小 | 文件数 | 角色 | 问题 |
|---|---|---|---|---|
| `mne_data/` | **89.56 GiB** | 77 770 | 公开原始数据（MNE 布局） | 不应在代码仓库内；含 37.4 GiB 当前未使用数据集 |
| `experiments/runs/` | **40.98 GiB** | 40 956 | 执行记录/结果/模型 | 52 个 run 无保留策略；20.80 GiB 为 delivery 重打包 |
| `.git/` | 8.04 GiB → **0.59 GiB** | 11 522 → 134 | 版本库 | 本次 gc 已回收 7.39 GB；仍有 112 个 rescue ref |
| `frozen/` | 3.32 GiB | 2 343 | 冻结证据（自有部分） | local_history 2.78 GiB；3 个 junction 造成 apparent ~92 GiB |
| `.venv-gpu/` + `.venv/` | 4.15 GiB | 65 787 | 运行环境 | 两套环境应移到仓库外 |
| `formal/lean/.lake/` | 2.47 GiB | 30 435 | Lean 构建缓存 | 可重建（mathlib 构建约 5.5 h） |
| `cloud_inputs/` | 1.98 GiB | 21 | 云输入 npz | 与 runs/cache 有派生关系 |
| `Paper/` | 0.19 GiB | 59 | 文献 PDF | 可移 LFS/外部 |
| `doc/assets/` | 0.17 GiB | 3 541 | 证据图/JSON | 与 frozen 无字节重复；仍有 9 目录无入链 |
| `src+tests+experiments/*.py+configs` | ≈0.05 GiB | 约 500 | **应作为 Git 仓库主体** | — |

## 2. 冗余分类与证据

### 2.1 字节级重复：几乎没有（已证实）

- >20 MiB 文件（排除 .venv/mne_data/.git）：**159 个，43.73 GiB，SHA-256 完全相同组 0**。
- 缓存/cloud/history 中唯一“重复”是同一目录被遍历两次的统计误差（`experiments/cache/history/tmp_20260906` 与父目录重叠）；真实重复 0。
- frozen + doc/assets + Paper 中 30 组重复共 0.12 GiB，几乎都在 `frozen/.../BrainSync_Tool_newest`（外部应用的自带 .venv/_internal 副本）内，不应由本仓库处理。
- **结论**：不能靠“删重复文件”解决磁盘问题；必须按“可重建性/保留策略/目录分层”治理。

### 2.2 delivery 归档 = mne_data 重打包（已证实，可重建）

- `experiments/runs/source_eog_20260909/delivery`：19 个文件 20.80 GiB = 9 个 `<study>.tar.gz` + 9 个 `<study>.json` 清单（+1 uploaded）。
- 逐归档比对：Huebner2017、BI2014a、GTN、Chailloux2020、Speier2017、BNCI2014-008 的成员与 `mne_data` 已解压文件 **100%** 同名同大小；BI2013a、BI2015a、EPFLP300 的成员与 `mne_data` 内嵌 zip 成员 **100%** 同名同大小。9/9 归档均可由 mne_data 重建。
- 每个 `<study>.json` 为 `n2p3_source_records/1`，逐记录记录 `path/sha256/original_path/original_archive_member`；frozen 中还有 `delivery.json/source_eog_20260909_*.manifest.json` 记录归档哈希。因此是“可重建 + 有清单”的冗余，不是唯一数据。
- 建议：**批准后删除 ✓.tar.gz，保留 JSON 清单与 frozen manifest**；若未来需云传输，按清单从 mne_data 重新打包（字节级 tar 顺序/压缩可能不同，但逐记录 SHA-256 可复核）。

### 2.3 Run 目录的重复结构（已证实）

- `experiments/runs`：`.npz` 17 312/5.43 GiB、`.json` 9 780/1.53 GiB、`.pt` 4 028/2.29 GiB、`.py` 8 134/0.13 GiB（唯一 1 174，重复 6 960）、`.pyc` 614、`.gz` 96/26.93 GiB。
- 分类：delivery 20.80 GiB、`local_prepared` 8.28 GiB（内含 `shards.tar` 约 3.8 GiB）、`export` 5.24 GiB、`source_models` 0.62 GiB、其余 runtime/analysis/inputs ≈5.9 GiB。
- 52 个 run 中 45 个最近修改在 30–90 天前；仅 16/52 有任意 manifest，9/52 有完整源码快照字节+哈希；真正模型重放仅 3/20（抽样）。**保留价值差异极大，但当前没有分级标记。**

### 2.4 环境与构建缓存

- 两个 venv 共 4.15 GiB（`.venv` CPU 0.98 GiB、`.venv-gpu` GPU 3.17 GiB）；均为机器绑定，不应进入代码仓库/备份主体。
- `formal/lean/.lake` 2.47 GiB（30 435 文件），是 Mathlib 等构建缓存；可重建但耗时约 5.5 h，是否保留取决于 Lean 工作是否继续。
- `.pytest_cache`/`.ruff_cache`/`__pycache__`/`*.pyc`：本次已清 2 043 个 pyc/56.4 MB。

### 2.5 历史归档与 Git 对象

- `.git` 清理前 8.64 GB：pack 205 MB + loose 7.27 GiB（几乎全部为 49 天前 force-add 的大 ZIP/GZIP 对象）+ 0.57 GiB 历史 bundle。
- 本次执行 `git gc`（默认过期策略，保留 reflog）→ .git **0.59 GB**；回收 **7.39 GB**。同时把 `.git/github-preparation-20260907.bundle`（0.57 GiB）移出到 `D:/BCI/artifacts/history/`。
- 迁移前已导出 6 个 ≥100 MiB 的 unreachable blob（共 3.30 GiB，ZIP 头 `PK\x03\x04`）到 `D:/BCI/artifacts/git-unreachable-20261019/`，附 `index.json`（对象 SHA/大小/头）；**该目录可复核后删除以再回收 3.30 GiB**。
- 112 个 dangling commit 已由现有 `refs/codex-rescue/20261019/*` 保护（清理前已存在），且本次生成 `D:/BCI/artifacts/n2p3-git-backup-20261019.bundle`（0.59 GiB，`--all`）作为备份；未做历史改写。

### 2.6 junction 造成的“假占用”

- `frozen/bidirectional_qc_20260909_acceptance/checkout` 内有 3 个 junction：`BrainSync_Tool_newest` → `D:/BCI/BrainSync_Tool_newest`（1.58 GB）、`.venv` → 仓库 venv（0.98 GB）、`mne_data` → 仓库 mne_data（89.56 GB）。递归工具/备份会重复计算约 92 GiB；该“冻结 checkout”因此并非真正不可变。
- 建议：把 junction 改为显式外部路径登记（如 `frozen/external_paths.json`），或只在执行时临时建立、结束后删除；禁止长期放在仓库内。

## 3. 本次已执行（可回滚/有备份）

| 动作 | 对象 | 回收 | 证据/备份 |
|---|---|---|---|
| `git gc`（默认过期，不 --prune=now） | .git objects | **7.39 GB** | .git 8.64 → 0.59 GB；日志 `tmp/audit/cleanup_executed_20261019.json` |
| 移出历史 bundle | `.git/github-preparation-20260907.bundle` | 0.57 GiB（移出，不删除） | 现在 `D:/BCI/artifacts/history/git-github-preparation-20260907.bundle` |
| 导出不可达大对象 | 6 个 blob / 3.30 GiB | 保留（可复核后删） | `D:/BCI/artifacts/git-unreachable-20261019/index.json` |
| `git bundle --all` | 全部 refs | 备份 0.59 GiB | `D:/BCI/artifacts/n2p3-git-backup-20261019.bundle` |
| 删除未完成 partial | `artifacts/cold-storage-20261014/inactive-pools/*.partial` | **12.38 GiB** | `cleanup_partials_20261019.json`；checkpoint 记录 `full_archive_verified=false`、原件在云端 |
| 清理 pyc/缓存 | 2 043 pyc + .pytest_cache/.ruff_cache | 56.4 MB | `cleanup_executed_20261019.json` |

- 磁盘效果：D: 可用 **79.4 → 94.2 GiB**。
- 源文件完整性：审计前后 `src/tests/experiments` 368 个 py SHA-256 一致；`git status` 仍为 431 行（与本轮审计前的 429–430 行差异仅为本报告新增/索引修改）。

## 4. 待批准清理（按收益/风险排序）

> 所有删除均建议先写清单（路径/大小/SHA/证据），并使用回收站或“先移出、验证后再删”。以下命令为示例，实际执行应由用户确认。

### T1 delivery 归档（20.80 GiB，低风险，可重建）
- 命令（保留 JSON 清单）：`Remove-Item experiments/runs/source_eog_20260909/delivery/*.tar.gz`
- 依据：§2.2；删除前记录 `<study>.json` 的 SHA 与记录数；恢复时从 `mne_data` 按 JSON 的 `original_path/original_archive_member` 重新打包。
- 预计回收 **20.80 GiB**。

### T2 不活跃原始数据（37.4 GiB，中风险，公开可重下载）
- 候选：`mne_data/lee2019_erp_nemar_nm000323_v1.0.4` 31.51 GiB（src/experiments 零引用）、`MNE-braininvaders2014b-data` 4.36 GiB（仅历史结果引用）、`MNE-llp-data` 1.23 GiB（零引用）、`MNE-RomaniBF2025ERP-data` 0.29 GiB（零引用）。
- 保留：B7/B7+3 活跃源库（BI2014a/BI2015a/BI2013a/BNCI2014-008/Huebner2017/EPFLP300/Chailloux2020/Speier2017/GTN/BI2012/Cattan2019_VR）、`mainsah2025_s2_nemar_v1.0.3`（外部审计待评分）。
- 动作：优先移到外部冷存；若网络可重下载则删除。预计 **37.4 GiB**。

### T3 run 派生缓存/导出（约 17.3 GiB，中风险，可重建但耗时）
- `experiments/runs/source_eog_20260909/local_prepared` 8.28 GiB（`shards.tar` 3.8 GiB 等）
- `experiments/cache` 3.79 GiB（Huebner shards/four_dataset/legacy）
- run 内导出的 `.tar.gz` 5.74 GiB（source_eog results/lineage、domain_transfer results/sources、unified_cloud verified_results、training_chain completed_prediction_evidence）
- 建议：移到 `D:/BCI/artifacts/research-cache/` 冷存；或在对应 run 增加 manifest 后删除。恢复需重跑准备/导出，需评估时间成本。

### T4 worktree 临时产物（约 5–7 GiB，中风险）
- `D:/BCI/worktrees/bi2014a-label-prefix` 5.61 GiB，其中两个 **1.92 GiB 完全相同的 prefix_selection.json**（同一 SHA-256）与大量 tmp 模型；代码已在 `codex/bi2014a-label-prefix` 分支。
- 建议：校验分支/保留 ref 后，把 tmp 压缩为 `<worktree>-tmp.zip` 移到 artifacts，再 `git worktree remove`；预计 5–7 GiB。

### T5 其他 artifacts/缓存（约 8 GiB）
- `D:/BCI/artifacts/git-unreachable-20261019` 3.30 GiB（复核内容后删）。
- `D:/BCI/artifacts/research-deliveries` 1.88 GiB、旧 `apps` 版本等按需保留。
- `formal/lean/.lake` 2.47 GiB（若继续 Lean 工作则保留；否则删除后按需重建，约 5.5 h）。
- `frozen/local_history` 2.78 GiB 历史压缩包移到外部冷存。

### T6 环境/文档（移动，不直接释放磁盘）
- 两个 venv（4.15 GiB）移到 `D:/BCI/envs/`；（可选）Paper PDF（0.19 GiB）移 LFS 或外部文献库。

**合计可回收估算**：T1 20.80 + T2 37.4 + T3 17.3 + T4 5–7 + T5 8 ≈ **88–91 GiB**；叠加本次已回收约 20 GiB，D: 可用可回到约 180 GiB。

## 5. 目标目录结构

```
D:/BCI/
  n2p3-net/                  # Git：src、tests、experiments/*.py、configs、doc、scripts、pyproject；目标 <2 GiB
  data/                      # 原始与准备数据（mne_data、cloud_inputs、外部数据）
  runs/                      # 实验输出（原 experiments/runs），每个 run 带 run.json/manifest
  artifacts/                 # 交付物、报告、冷存、备份 bundle
  archive/                   # frozen 载荷、local_history、被替代的研究材料
  envs/                      # .venv / .venv-gpu
  worktrees/                 # 临时并行工作树（用后归档/删除）
```

- 代码中路径解析统一走环境变量：`MNE_DATA`（MNE 原生支持）、`N2P3_RUNS_ROOT`、`N2P3_ARCHIVE_ROOT`；缺省回退当前相对路径，避免一次性改 5 000+ 处引用。迁移期可在仓库内放目录 junction，但需登记在 `frozen/external_paths.json`，并在迁移完成后移除。
- 仓库 Git 主体只保留文本：`src/`、`tests/`、`experiments/*.py`、`configs/`、`doc/`、`frozen/*.json`、`formal/lean` 源码（不含 .lake/artifacts）；`.gitignore` 增加 `.venv*/`、`mne_data/`、`experiments/runs`、`cloud_inputs/`、`formal/lean/.lake/`、`artifacts/` 等（多数已有）。

## 6. 维护制度（防止再次膨胀）

1. **Run 目录规范**：每个 run 必须有 `run.json`（18 字段：schema、study、stage、created_utc、code snapshot 哈希、命令、seed、数据清单、预算、失败/分母账本、产物清单、replay 证据等，见 G8 报告）；缺 manifest 的 run 不得进入“已完成”。
2. **保留分级**：R0 活跃（不压缩）；R1 90 天内（压缩中间产物）；R2 归档（仅保留 manifest + 外部 payload 哈希）；R3 退役（manifest 入 Git、payload 移冷存/删除）。
3. **禁止项**：原始数据不进 `runs/`；不在仓库内建长期 junction；不把 venv/构建缓存纳入备份主体；不提交 >50 MiB 二进制（pre-commit 检查）。
4. **索引**：`doc/assets` 与 `frozen` 每项必须有 manifest/索引；`doc/evidence/README.md` 自动检查未索引项。
5. **自动化**：`scripts/du_report.py`（月度 du + 大文件/重复/未索引清单）、`scripts/check_retention.py`（超期/缺 manifest）、`scripts/verify_archive.py`（SHA 复核）、`git gc --auto`/季度人工 gc；CI 中禁止新增 >50 MiB 文件。
6. **验收**：仓库 apparent < 5 GiB（不含外部数据/结果）；D: 可用 >150 GiB；`git count-objects -vH` 的 size < 1.5 GiB；52 个历史 run 均带 manifest 或明确归档状态。

## 7. 风险与回滚

- T1/T3：均有清单与重建路径；删除前应保存 `*.json` 清单与 SHA；恢复需重跑准备/打包（时间成本）。
- T2：公开数据可重新下载（MNE `mne.datasets`），但受网络/版本影响；建议先移到外部盘而非直接删除。
- T4：代码在分支/ref 中；worktree 删除前确认 `git status --porcelain` 为空且独有提交已入 ref。
- 本次已执行的 `git gc` 未使用 `--prune=now`，保留 reflog；112 个 dangling commit 已由 `refs/codex-rescue/20261019/*` 保护，并已生成备份 bundle。
- 所有删除脚本默认 dry-run，执行需显式 `-Apply`，并先输出清单。

## 8. 复现与验收命令

```powershell
# 目录规模（junction 感知，见 tmp/audit/du_report.py）
..venvScriptspython.exe tmp/audit/du_report.py --root . --exclude-junctions
# Git 对象与备份
git count-objects -vH
git for-each-ref refs/codex-rescue --format='%(refname:short)'
git bundle verify D:/BCI/artifacts/n2p3-git-backup-20261019.bundle
# 重复/大文件（本次结果）
..venvScriptspython.exe tmp/audit/folder_dupes_report.py
# 剩余清理 dry-run
powershell -File tmp/audit/cleanup_remaining.ps1            # 默认只打印
powershell -File tmp/audit/cleanup_remaining.ps1 -Apply     # 需人工确认后执行
```

## 9. 给科研总管 agent 的交接要点（文件夹治理相关）

1. **先读本报告与 `docs/evidence/repository_audit_20261019.zh.md`**；本报告是执行依据，审计报告是证据索引。
2. 下一次启动训练/实验前：确认 `experiments/runs` 与 `mne_data` 的路径解析（迁移期用环境变量），不要直接在主目录新增原始数据副本。
3. 任何删除/迁移：先写 manifest（路径/大小/SHA/来源），再执行；Run 的 `run.json` 与 replay 证据必须保留。
4. `git gc` 已完成（.git 0.59 GB）；不要再执行 `git gc --prune=now`，除非先备份 `refs/codex-rescue` 并确认 112 个 rescue ref 可解析。
5. 当前 D: 可用 94.2 GiB；若批准 T1–T3，可回到约 150–180 GiB，需在批准后执行并更新本报告状态。

## 10. 附录

- 已执行日志：`tmp/audit/cleanup_executed_20261019.json`、`tmp/audit/cleanup_partials_20261019.json`
- 清理清单（机器可读）：`tmp/audit/folder_cleanup_plan.csv`
- 剩余清理脚本（默认 dry-run）：`tmp/audit/cleanup_remaining.ps1`
- 备份：`D:/BCI/artifacts/n2p3-git-backup-20261019.bundle`（0.59 GiB）、`D:/BCI/artifacts/git-unreachable-20261019/`（3.30 GiB）
- 关联审计：`doc/evidence/repository_audit_20261019.zh.md`、`doc/assets/repository_audit_20261019/`

> 本报告为文件夹管理建议与执行记录；T1–T6 中除已执行项外均需用户批准后执行。
