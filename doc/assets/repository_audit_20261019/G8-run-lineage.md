# 审计分区 G8：experiments/runs 产物执行身份（lineage）完整性抽样

- 审计对象：`D:\BCI\n2p3-net\experiments\runs`（52 个 run 目录；43.92 GB / 40,939 文件，只读盘点见 `tmp/audit/scratch/g8_inventory.json`）
- 仓库状态：HEAD `f1a303c`（main），`git status --porcelain` 431 行未提交状态（任务简报为 429 项）；formal/ 与 tests 184/185 未跟踪。
- 只读声明：本分区未修改/删除/重命名任何已有文件，未执行 git 写命令，未安装依赖，未启动训练；仅写入本报告与 `tmp/audit/scratch/` 下的只读脚本；未解析 .npz/.pt 内容（仅按文件名与大小统计）。
- 计数口径：`有源码快照` = run 目录内存在 src/experiments 源码副本或 execution_snapshots；`有 manifest` = 存在声明本次执行身份（快照哈希/配置/命令）的 JSON；`只有结果` = 仅有结果 JSON/模型/日志，无以上的声明性文件。
- 与既有分区去重：05 分区已覆盖 experiments 顶层脚本工程与 runs 的体积构成（E-1/E-2）；本报告只做**逐 run 的执行身份字段**统计、结果 JSON 的路径泄漏、manifest↔归档哈希一致性，并给出 lineage 字段规范与保留策略。

## 结论摘要

1. **源码身份覆盖率低且分层明显**：52 个 run 中，同时具备"源码快照字节 + 哈希声明"的只有 9 个（17%）；带源码字节共 12 个（23%）；只存哈希声明、无源码字节 13 个（25%）；只有结果 14 个（27%），两者都无 13 个（25%）。抽样 20 个 run 的分布为 8/3/7/1/1。
2. **存在可完整重算的正面样本**：`training_chain_auto_20260907/execution.json:8-14` 的 `manifest_sha256` 与本地 `source/source.manifest.json` 实测一致（4afea588…），`archive_sha256` 与 `source.source-only.tar.gz` 实测一致（5aa71634…，1,059,425 B / 235 成员）；但 `source_commit_role=base_commit_only`，commit 只标识基线，不能定位实际执行字节。
3. **哈希声明大量悬空**：对 run 内 ≤1.5 MB JSON 的统计，`archive_sha256` 79 个不同值中 57 个能在本机解析到内容一致的归档（72%），22 个不能；`manifest_sha256` 73 个不同值中 35 个可解析（48%）。另出现明显占位哈希 `cccccccc…`/`dddddddd…`（`system_implementation_20261004`、`system_research_20261004`，resolve 统计到至少 9 处引用），说明合成夹具与真实记录同构，审计脚本不能只按 64 位 hex 判定来源。
4. **结果 JSON 内嵌绝对路径是常态**：有结果 JSON 的 39 个 run 中 28 个（72%）至少一个 result/record/summary JSON 含绝对路径；全量 52 个中 23 个出现 `autodl` 主机标识；`bidirectional_qc_20260909/completion.json:17` 直接写入主机名 `"Tonger"`；`source_eog_20260909/export/lineage_local_verification.json:3` 把 `D:\…` 绝对路径写进本应可移植的 lineage 证明。
5. **随机性不可判定**：29/52 个 run 有显式 seed；抽样 20 个中 12 个明确、3 个部分（其余无 seed）。`boundary_state_redesign_20261005` 等确定性研究完全无 seed，也没有 `randomness: "none"` 声明，审计方无法区分"无随机性"与"漏记 seed"。
6. **环境记录两极分化**：仅 16/52 个 run 记录了 torch/cuda/python 中至少一项；完整环境块（python/torch/numpy/cuda_runtime/device_name）只在 `early_stopping_full512_20260907/probe/results/BI2014a_1/record.json:8` 与 `brainsync_recovery_20260908/environment.json` 见到。
7. **失败/分母账本覆盖 18/52**，抽样 12/20；多数只是汇总的 `failed: 0`，缺少"计划分母 vs 实际执行"的逐单元账本。正面例外：`boundary_state_redesign_20261005/e2_results/result.json:6`（requested_cells 288 + 每 cell status/error）、`domain_transfer_cloud_20260916/completion.json:8-32`（expected_physical_*_fits + failed=0）。
8. **真正的模型重放只有 3/20**：`domain_transfer_cloud_20260916/completion.json:39`（288 条重放，最大预测差 3.37e-4）、`bidirectional_qc_20260909/completion.json:5`（predictions_replayed=12、缓存重建 6/12）、`brainsync_recovery_20260908/deterministic_replay/`。其余所谓 verification 多为文件数/哈希/导出成员校验，不是模型重放。

## 1. 范围、计数口径与方法

- 对象：`experiments/runs` 下 52 个目录，43.92 GB / 40,939 文件（只读盘点 `tmp/audit/scratch/g8_inventory.json`）。未加载任何 .npz/.pt 内容，只解析 ≤1.5 MB 的 JSON 与文件名/大小。
- 四分类口径（第 3 节表格使用）：
  - **A 快照字节+声明**：run 内有 `source.manifest.json`/`source.source-only.tar.gz`/`checkout/` 等源码字节，且 JSON 声明 `archive_sha256`/`source_manifest_sha256`。
  - **B 仅快照字节**：有 `executed_source/` 或 `source/` 源码副本与 `source.json` 文件哈希，但无快照归档哈希/commit。
  - **C 仅声明哈希**：record/execution JSON 声明快照哈希，但 run 内没有源码字节（归档可能在 `frozen/` 或远端）。
  - **D 只有结果**：仅有 ckpt/npz/结果 JSON，无源码或声明。
  - **E 两者都无**：小目录（研究笔记、退役记录、下载清单），既无源码身份也无模型结果。
- 抽样 20 个覆盖：最大 3 个（`source_eog`、`domain_transfer_cloud`、`unified_cloud`）、云端训练 4 个、本地训练/研究 6 个、审计/交付 3 个、异常目录 4 个（见附录 A）。
- 与既有分区去重：05 分区已统计 runs 体积构成与 runs 内脚本漂移（E-1/E-2）；本报告不重复体积分析，只做逐 run 身份字段、路径泄漏、哈希对应与规范/保留策略。

## 2. 抽样 20 个 run 的执行身份七要素

图例：✓ 有且可验证；◐ 部分/仅汇总；— 未检出（不代表一定不存在，深度 ≤4、单文件 ≤1.5 MB）。

| # | run | 类型 | 源码快照 | 配置/命令 | git（commit/dirty） | seed | 包版本 | 失败/分母 | 重放 |
|---|-----|------|----------|-----------|--------------------|------|--------|-----------|------|
| 1 | source_eog_20260909 | 数据准备+源模型+交付 | A 26 个 manifest | 有 plan/preprocessing | ✓ f1a303c / ✓ | ✓ 20260909 | ◐ python 3.12.3 | ✓ failed_cells=0 | ◐ 导出 231 成员校验 |
| 2 | domain_transfer_cloud_20260916 | 云端矩阵 | C 归档哈希 | 有 argv | — | ✓ 20260916 | ◐ mne/numpy 记录 | ✓ expected fits + failed=0 | ✓ 288 条容差重放 |
| 3 | unified_cloud_20260916 | 云端 12 数据集 | C 结果包哈希 | 有 raw manifest/argv | ◐ 1 处 | ◐ 1 处 | — | —（仅 total_bytes） | ◐ 文件级哈希 |
| 4 | brainsync_qc_remaining_20260908 | 云端 QC | A 4 manifest + 源码 | 有 plan | ✓ f54e45c / ✓ | ✓ 20260908 | ✓ torch 2.8/cuda 12.8 | ✓ fits 72 / failures 0 | ◐ 计数验证 |
| 5 | brainsync_qc_redesign_20260908 | 云端 QC | A 2 manifest + 源码 | 有 plan | ✓ f54e45c / ✓ | ✓ 20260908 | ✓ torch 2.8/cuda 12.8 | ✓ fits 57/pred 114 | ◐ 计数验证 |
| 6 | training_chain_auto_20260907 | 训练链编排 | A manifest+归档 | 有 argv | ✓ f54e45c / ✓ | ✓ 20260906 | ◐ python | ✓ 0 failures | — |
| 7 | early_stopping_full512_20260907 | 训练 | A 双快照 | 有 plan | ✓ f54e45c / ✓ | ✓ 20260906 | ✓ 完整 env | ✓ failed_cells=0 | — |
| 8 | deep_chain_audit_20260929 | 审计 | C 脚本快照 | ◐ records | ◐ 1 处 | ◐ 3 处 | ◐ 实现信息 | ✓ 分母字段 | — |
| 9 | brainsync_s1_cloud_20260908 | 云端 S1 | A 3 manifest + 源码 | 有 plan | ✓ f54e45c / ✓ | ✓ 20260908 | ✓ torch/cuda | ✓ fits/pred+失败策略 | ◐ 计数验证 |
| 10 | operator_research_20261004 | 本地研究矩阵 | B executed_source | 有 plan | — | ✓ 20261004 | — | ◐ failure_policy | — |
| 11 | bi64_protocol_factorial_20260906 | 训练 factorial | C 每条 record 快照 | 有 plan | ✓ 145 处 / ✓ | ✓ 20260906 | — | ◐ trials 分母 | — |
| 12 | boundary_state_redesign_20261005 | 本地研究 | B executed_source | 有 plan | — | —（确定性未标注） | — | ✓ 288/288 + error | ◐ 数学复核 |
| 13 | brainsync_recovery_20260908 | 本地+GPU | A 4 manifest+zip | 有 plan/argv | ✓ f54e45c / ✓ | ✓ 20260908 | ✓ 完整 env | ✓ fits 35 / 0 failures | ✓ deterministic_replay |
| 14 | cross_dataset_start_20260906 | 云端启动 | C preflight 快照 | 有 plan | ✓ c8ee573 / ✓ | ✓ 20260906 | ✓ torch/cuda | ◐ 仅部分 | — |
| 15 | p300_mainline_20260906 | 云端 P300 | C 每条 record 快照 | 有 argv | ✓ 131 处 / ✓ | ✓ 148 处 | — | ◐ 仅 returncode | — |
| 16 | bidirectional_qc_20260909 | 本地 QC | D 仅 101 文件哈希 | 有 profiles | — | —（确定性未标注） | — | ✓ 6 模型/12 预测 | ✓ predictions_replayed=12 |
| 17 | system_implementation_20261004 | 本地实现 | B executed_source | 有 plan | — | ◐ 9 处 | — | ◐ cells error=None | — |
| 18 | _retired | 退役归档 | A 同 recovery | 有 | ✓ / ✓ | ✓ | ✓ | ✓ | ✓（复制自 recovery） |
| 19 | lda_head_bi61_20260906 | 云端 LDA | C 6 处快照 | 有 plan.journal | ✓ 6 处 / ✓ | ✓ 6 处 | — | ◐ integrity 77 条 | ◐ 模型/预测哈希 |
| 20 | dataset_expansion_audit_20260908 | 数据源审计 | E 无 | — | — | — | ◐ SDK 清单 | —（镜像清单） | — |

注：`early_stopping*`、`training_chain_auto` 的"源码快照字节"只有 1–2 个文件（manifest + 1.1 MB 归档），按 A 类计；`_retired` 是 `brainsync_recovery_20260908` 的物理归档副本，字段相同，统计时可能构成双重计数（见第 8 节）。


## 3. 全量 52 个 run 的身份覆盖率

| 分类 | 数量 | 占比 | 目录 |
|------|------|------|------|
| A 快照字节+声明 | 9 | 17% | source_eog_20260909, brainsync_qc_redesign, brainsync_qc_remaining, brainsync_recovery, brainsync_s1_cloud, early_stopping_full512, early_stopping_probe, training_chain_auto, _retired |
| B 仅快照字节 | 3 | 6% | boundary_state_redesign, operator_research, system_implementation |
| C 仅声明哈希 | 13 | 25% | bi64_protocol_factorial, brainsync_quality_20260909, cross_dataset_start, deep_chain_audit, domain_transfer_cloud, four_dataset, lda_head_bi61, lda_head_order, p300_mainline, target_epoch90_full512, training_chain_full512_metadata, unified12, unified_cloud |
| D 只有结果 | 14 | 27% | bidirectional_qc, brainsync_cause_audit, brainsync_p001, brainsync_quality_20260907, brainsync_reference, brainsync_s1_local, chain_improvements, filter_phase(×2), input_alignment_audit, legacy, system_research, training_chain_compact512, unified_model_implementation |
| E 两者都无 | 13 | 25% | algorithm_chain_validation, bi61_retirement, bnci_unseen_target, brainsync_quality_implementation/research, chain_attribution, cloud_delivery, dataset_expansion_audit, domain_transfer_implementation/research, lda_head_budget, refit_scope_audit, source_modules |

字段级出现率（对每个 run 的 ≤1.5 MB JSON 做正则探测，只代表"字段出现过"，是逐记录完整率的上界）：快照声明 21/52、40 位 commit 20/52、dirty 20/52、seed 29/52、torch/cuda/python 版本 16/52、失败/分母账本 18/52。commit 与 dirty 覆盖完全重合，因为二者都只由 `freeze_source_snapshot.py` 生成。

**P1｜C 类"声明哈希但无字节"占 25%，且大部分无法在本机复核。** 证据：`brainsync_qc_redesign_20260908/retrieval_verification.json:6-7` 记录 `archive_sha256=e9cc129e…`、`source_manifest_sha256=8809759b…`，但本机 8 MB 以下归档中找不到对应内容；`brainsync_s1_cloud_20260908` 的 `87cdfe64…`、`brainsync_p001_20260907/withdrawal.json` 的 `d69428ca…` 同样悬空。影响：云端产物回到本机后只能"相信"哈希，不能重算；一旦归档被清理，身份永久丢失。

**P2｜B 类有源码字节但无法定位来源。** `boundary_state_redesign_20261005/e2_results/result.json:4-5` 的 `plan_sha256` 可复算（= `e2_plan.json` 文件哈希），`source_sha256=6b007d79…` 也实测等于 `sha256(canonical JSON(source.json["files"]))`（124 个文件），但 `source.json` 没有 commit、dirty、生成时间与生成者，且该 digest 配方未写入产物——独立审计方必须读 `source.json` 的生成代码才能复算。`operator_research_20261004/*/source.json`、`system_implementation_20261004/*/execution_source.json` 同样只有 `files` 映射。

## 4. 结果 JSON 的绝对路径与机器名

- **绝对路径**：52 个 run 中 39 个（75%）至少一个 ≤1 MB JSON 含 `X:\…` 或 `/root|/home|/mnt|/autodl-tmp/…`；其中有结果 JSON 的 39 个 run 里 28 个（72%）的 result/record/summary 文件本身含绝对路径。
- Windows 盘符 27 个 run、POSIX 远端路径 24 个 run；23 个 run 出现 `autodl` 字样；主机名 `"Tonger"` 出现在 `bidirectional_qc_20260909/completion.json:17,27`（pytest 元数据）。
- 典型证据：
  - `bidirectional_qc_20260909/execution.json` 的 `plan.calibration_sessions` 与 `raw_hashes` **以 `D:\BCI\…` 绝对路径为字典键**（`execution.json:4-30` 区块），换机后键不可解析。
  - `brainsync_qc_remaining_20260908/controls/native7/native7_bp01_eog0/calibration.record.json:208-209` 在模型记录中写入本地 `D:\…\brainsync_p001_20260907\inputs\…`；该类记录在该 run 有 57/60 个含绝对路径。
  - `early_stopping_full512_20260907/probe/results/BI2014a_1/record.json:79-80` 同时保存 `source_snapshot_manifest=/root/autodl-tmp/…` 与快照哈希；路径与哈希并列，路径失效时只能靠哈希。
  - `unified_cloud_20260916/export/verified_results.manifest.json:3` 的 `archive` 是 `/root/autodl-fs/…`；结果包 manifest 自述远端路径，无法作为本地归档索引。
  - `source_eog_20260909/export/lineage_local_verification.json:3` 把 `D:\BCI\n2p3-net\…\source_eog_20260909_lineage.tar.gz` 写入"lineage 本地校验"文件，随归档分发即暴露目录结构与主机布局。
  - `source_eog_20260909/delivery_progress.json` 记录失败的 ssh 命令，含私钥路径 `C:\Users\chent\.ssh\id_ed25519_n2p3_cloud_20260827`（仅路径，未读取密钥值）；`dataset_expansion_audit` 的 `audit.json` 含 `D:\BCI\n2p3-net` 根路径。
- **P1｜跨机不可复现**：上述路径是"执行身份"的一部分但不是"可移植身份"；把 run 目录复制到另一台机器后，任何以路径为键的查找（raw_hashes、source_files）都会失效，重放器只能退化为按文件名猜。
- **P2｜机器名/硬件指纹**：`resource_snapshot.json` 仅 `platform.machine=x86_64, system=Linux, python_version`（`brainsync_qc_remaining_20260908/resource_snapshot.json`），未记录 hostname；hostname 反而由 pytest 元数据带出（上面 `Tonger`）。建议统一为 `host: {hostname, machine, system, gpu_name, gpu_total_bytes}`，并对 `hostname` 做可选哈希化（保留可区分性、降低指纹泄露）。
- 正面做法：云端 record（`domain_transfer_cloud_20260916`、`early_stopping_full512_20260907`）用 `n2p3-origin://STUDY/participant/ID` 逻辑标识，见 `early_stopping_full512_20260907/probe/results/BI2014a_1/record.json` 的 `expected_source_participant_keys`；应推广为所有 artifact 的默认引用形式，绝对路径仅作 `source_locator` 辅助字段。

## 5. manifest 与归档哈希的对应性

- 归档哈希：run 内声明的 79 个不同 `archive_sha256` 值，按内容寻址在 `experiments/runs`/`frozen`/`artifacts`/`cloud_inputs` 的 ≤8 MB 归档中匹配到 57 个（72%），1,437 次引用中 1,401 次（97.5%）可解析；22 个不可解析（含云端独有归档）。
- 清单哈希：73 个不同 `manifest_sha256` 值中 35 个可解析（48%）。注意其中大量 `manifest_sha256` 指向**原始输入 session manifest**（如 `fbb72879…` 在 `brainsync_qc_remaining` 被引用 104 次），不是源码快照；把两类同名字段混用本身是规范缺陷。
- 可完整重算的样本（正面）：
  - `training_chain_auto_20260907/execution.json:8-14` → `source/source.manifest.json:4-10`：`archive_sha256=5aa71634…`（实测等于归档哈希）、`manifest_sha256=4afea588…`（实测等于 manifest 文件哈希）、`member_count=235`、`source_commit=f54e45c…`、`git_status` 77 行 dirty。
  - `bi64_protocol_factorial_20260906/retrieved/.../original_runner_repeat/record.json:4-23`：source/evaluation 双快照 + `checkpoint_sha256=09bc2641…` + `source_snapshot_sha256=d7eb716c…`；后者实测解析到 `frozen/binary_transfer_20260906f.source-only.tar.gz`（1,326,306 B）。
  - `domain_transfer_cloud_20260916/launch.json:15` 的 `source_archive_sha256=aaa1514a…` 在本机未找到归档（云端构建），属"声明但不可复核"。
- **P2｜占位哈希进入运行记录（合成夹具未标记）**：`system_implementation_20261004/e0_review/{initial,changed_inner_fresh,changed_inner_reused}/record.json:7` 的 `archive_sha256` 为 64 个 `c`，同文件 `:1001` 的 `source_snapshot_sha256` 亦同；`system_research_20261004/interfaces/reuse_1/direct/record.json:7,294` 使用 64 个 `d`。这些目录是接口/复用测试夹具（`e0_review`、`interfaces/reuse_1`），但记录中没有 `evidence_scope`/`synthetic:true` 标记，与真实结果同构；审计脚本若只检查"字段存在且为 64 位 hex"会误判为已绑定源码。
- **P2｜同名不同义**：`manifest_sha256` 既表示源码清单也表示原始输入清单；`archive_sha256` 既表示源码归档也表示结果归档（如 `unified_cloud_20260916/export/verified_results.manifest.json`）。建议按 `source_archive_sha256`/`source_manifest_sha256`/`input_manifest_sha256`/`result_archive_sha256` 分名，并在 schema 中固定语义。


## 6. 最小可执行 lineage 规范（字段清单）

目标：不改现有训练入口，只增加"每次执行写一个 `run.json`"与"一个只读校验器"；沿用仓库已有 `n2p3_source_freeze/1` 清单（`experiments/freeze_source_snapshot.py:48-54`）与 `n2p3net_data_lineage/1` 的 canonical digest（`src/data/lineage.py:12,36-44`），不另造体系。

### 6.1 必填字段（18 项，缺一即校验失败）

```json
{
  "schema": "n2p3_run/1",
  "run_id": "20261019T083436Z_brainsync_qc_redesign_20260908",
  "created_utc": "2026-10-19T08:34:36Z",
  "evidence_scope": "public_development | adult_device_confirmation | synthetic_fixture",
  "command": {"argv": ["python", "experiments/run_brainsync_recovery.py", "..."], "cwd_rel": "n2p3-net", "entry_sha256": "<64hex>"},
  "config": {"path": "plans/plan.json", "sha256": "<64hex>", "seeds": {"source": 20260908, "target": 20260908},
             "randomness": "seeded | none | uncontrolled"},
  "source": {"manifest": "source/source.manifest.json", "manifest_sha256": "<64hex>",
             "archive": "source/source.source-only.tar.gz", "archive_sha256": "<64hex>",
             "member_count": 235, "git_commit": "<40hex>", "commit_role": "base_commit_only", "dirty": true, "git_status_sha256": "<64hex>"},
  "environment": {"python": "3.12.3", "torch": "2.8.0+cu128", "cuda": "12.8", "numpy": "2.5.2", "mne": "1.12.1",
                  "host": {"hostname_sha256": "<64hex>", "machine": "x86_64", "system": "Linux", "gpu_name": "NVIDIA GeForce RTX 5090", "gpu_total_bytes": 34190917632}},
  "denominators": {"planned_cells": 288, "attempted": 288, "completed": 288, "failed": 0, "skipped": 0,
                   "failure_policy": "any_requested_cell_failure_invalidates_candidate_score"},
  "ledger": {"path": "ledger.json", "sha256": "<64hex>"},
  "models": [{"id": "A_GTN_s20260909", "seed": 20260909, "checkpoint": "source_models/A_GTN_s20260909/source.u4000.pt", "checkpoint_sha256": "<64hex>", "protocol": {"sfreq": 128.0, "l_freq": 0.1, "h_freq": 30.0}}],
  "results": [{"path": "final_analysis.json", "sha256": "<64hex>", "bytes": 123456}],
  "replay": {"performed": true, "records": 288, "max_abs_prediction_diff": 0.0003366,
             "replay_script_sha256": "<64hex>", "status": "passed | failed | not_performed"},
  "producer_sha256": "<64hex>"
}
```

字段语义要点：
- **路径一律 run 相对路径**，禁止 `D:\`、`/root/`；外部输入用 `n2p3-origin://STUDY/participant/ID` + 可选 `source_locator`（只读提示，不参与校验）。
- `git_commit` 必须与 `commit_role: base_commit_only` 同时出现；**执行身份以 `source_manifest_sha256` + `archive_sha256` 为准**，commit 仅作人类可读基线（对应 `freeze_source_snapshot.py:52` 现状）。
- `dirty:true` 时 `git_status_sha256` 必填（对 `git status --short` 行做 canonical hash），否则无法区分"干净执行"与"带改动执行"。
- `randomness:"none"` 必须显式声明，用于确定性研究（boundary_state、bidirectional 这类无 seed run）；缺省不等于 none。
- `replay.status: not_performed` 是合法值，但必须显式写出，避免把"没重放"伪装成"已验证"。
- `evidence_scope` 强制区分真实运行、公开数据开发与合成夹具（消灭 `system_*` 占位哈希歧义）。

### 6.2 校验器行为（6 个门，全部只读）

1. `schema` 白名单；`run_id`/`created_utc` 格式。
2. 所有 `*.sha256` 必须为 64 位小写 hex，且拒绝 `c{64}`/`d{64}` 等退化值（正则 `^(?!([0-9a-f])\1{63}$)[0-9a-f]{64}$`）。
3. 逐个重算 `config`/`source.manifest`/`source.archive`/`ledger`/`results` 哈希；任一缺失 → `missing_artifact`（区别于 `hash_mismatch`）。
4. 从 `source.archive` 解包到临时目录，按 `source.manifest.files` 逐文件重算；成员数不一致 → 失败。
5. `denominators.completed + failed + skipped == attempted <= planned_cells`；`failed>0` 时必须有 `failure_policy`。
6. `replay.performed=true` 时要求 `replay_script_sha256` + `records` 与 `denominators.completed` 对得上；`max_abs_prediction_diff` 需给出容差并记录容差来源（现有 `domain_transfer_cloud_20260916/completion.json:39-43` 的 3.37e-4 可作首个通过样本）。

### 6.3 落地顺序（最小改动）

1. 先给 9 个 A 类 run 补 `run.json`（字段从现有 `execution.json`/`source.manifest.json` 直接映射，预计每个 ≤30 分钟）。
2. 新入口只在启动时写 `run.json`，训练代码零改动；云端 wrapper 在 `retrieved/` 回传时合并 `environment` 与 `replay`。
3. 校验器放 `experiments/verify_run_lineage.py`，CI 只跑"manifest 与 archive 可解析"的 run，悬空的 C 类 run 标记为 `unverifiable` 而不是失败。

## 7. 清理与保留策略

现状：43.92 GB / 40,939 文件；`source_eog_20260909` 34.5 GB、`domain_transfer_cloud` 2.8 GB、`unified_cloud` 1.3 GB 合计 38.6 GB（88%）。小归档去重空间明确：≤8 MB 归档 551 个文件只有 289 个不同内容（47.6% 重复），最大重复是 `source.source-only.tar.gz` 在 runtime/export 各一份（05 分区 E-1 已指出同源）；`export/source_eog_20260909_lineage.tar.gz` 单文件 254 MB。

分层保留（先 manifest、后删除；删除前必须通过 6.2 校验）：

| 层 | 内容 | 保留 | 依据 |
|----|------|------|------|
| R0 永久 | `run.json`、`source.manifest.json` + `source.source-only.tar.gz`（≤2 MB）、`ledger.json`、result JSON、checkpoint 哈希清单、replay 记录 | 永久，随 Git 清单归档 | 单 run 约 1–3 MB，52 个 run 合计 <0.2 GB；这是不可再生的身份 |
| R1 24 个月 | `.pt` checkpoint、`.npz` 预测/缓存 | 按 `sha256` 去重，只保留被已接收结果引用的版本 | checkpoint 可重训但成本高；同哈希多份无意义 |
| R2 90 天 | 逐 cell 中间缓存、重复 delivery `.tar.gz`、dashboard HTML/PNG、benchmark 原始输出 | 到期删除，先移到内容寻址 store | 可由 R0/R1 重放；占体积 90% 以上 |

执行规则：
1. **内容寻址 store**：`artifacts/store/sha256/<hash>`；run 目录只留 manifest 与指针。551→289 的去重可直接减少约 47% 小归档副本。
2. **删除前置条件**：`verify_run_lineage.py --run <dir>` 通过（archive 可解包、哈希一致、ledger 平衡），并在 `retention.json` 登记 `{run_id, kept, deleted, sha256, reason, verified_utc}`。
3. **不得删除**任何被 `frozen/` 清单、`doc/evidence/` 索引或正式结果引用的 artifact；退役走"物理归档 + 校验 + 清单"，与 `frozen/README.md` 的 archive 分发约定一致。
4. **云端回传**：只回传 R0 + R1（manifest/哈希/结果/必要 checkpoint），不回传 delivery 大包；`unified_cloud` 这类 6.7 MB results manifest 已足够重建索引。
5. **禁止**只凭 `failed: 0` 清理：失败/跳过账本必须与 planned 分母一起归档，否则无法证明"没有失败"还是"没有记录"。

## 8. 存疑与未验证

- **字段覆盖是下界**：扫描深度 ≤4、单 JSON ≤1.5 MB；`source_eog`（8,347 文件）、`operator_research`（6,813 文件）深层记录可能还有 seed/commit 字段，未逐一读取。
- **字段探测是上界**：`fits`/`failed` 等正则可能命中非账本字段（如 `resource_snapshot.memory` 的 `fits` 类词），所以"18/52 有账本"是"字段出现过"的上界，不是逐条完整率。
- **哈希解析只覆盖 ≤8 MB 归档**：大于 8 MB 的 delivery 包未哈希；因此 22 个"未解析"archive 中可能有一部分实际存在（例如云端包），需要单独按大小/名称复核。
- **未验证 254 MB lineage tar**：`source_eog_20260909/export/source_eog_20260909_lineage.tar.gz` 的 `archive_sha256`（`lineage_local_verification.json:4`）未重算；该文件同时是第 4 节"绝对路径写进 lineage 证明"的证据。
- **`_retired` 与 `brainsync_recovery_20260908` 重复**：前者是后者的物理归档副本，52 个 run 的统计因此包含 1 个重复身份；抽样同时选两者用于对照。
- **3 个 run 无 JSON**：`algorithm_chain_20260916_validation`、`brainsync_s1_local_20260908`、`lda_head_budget_20260906` 只有二进制/空目录，路径统计对其不适用。
- **`manifest_sha256` 语义靠上下文推断**：未逐一读生成代码确认每个 producer 的含义；第 5 节的"同名不同义"结论基于字段所在文件类型（source/session/result）判断。
- **时间信息缺失**：多数 `record.json` 只有 `started_unix`，无 `created_utc`；无法判断产物生成顺序与是否被事后覆盖（如 `plan_metadata_correction.json` 承认修正过原始计划）。
- **未读密钥值**：仅记录 `source_eog_20260909/delivery_progress.json` 中的私钥**路径**；未打开任何 `.ssh` 文件。

## 附录 A：抽样 20 个 run 的规模与类别

| run | 文件数 | MB | 四分类 | 主要身份载体 |
|-----|-------:|----:|--------|--------------|
| source_eog_20260909 | 8,347 | 34,498 | A | 26 个 `source.manifest.json` + `source_models/index.json` + export lineage |
| domain_transfer_cloud_20260916 | 114 | 2,823 | C | `launch.json`、`completion.json`、每 record 的 `source_archive_sha256` |
| unified_cloud_20260916 | 164 | 1,347 | C | `export/verified_results.manifest.json`（6.7 MB） |
| brainsync_qc_remaining_20260908 | 1,488 | 1,027 | A | 4 个 source manifest + `retrieved/execution.json` |
| brainsync_qc_redesign_20260908 | 888 | 1,010 | A | 2 个 source manifest + `retrieval_verification.json` |
| training_chain_auto_20260907 | 29 | 780 | A | `execution.json` + `source/source.manifest.json` |
| early_stopping_full512_20260907 | 6,263 | 506 | A | `execution.json` + 每 record 双快照 |
| deep_chain_audit_20260929 | 300 | 369 | C | `input/audit_script_snapshots/` + QC record 分母 |
| brainsync_s1_cloud_20260908 | 971 | 349 | A | 3 个 source manifest + `retrieved/execution.json` |
| operator_research_20261004 | 6,813 | 144 | B | `*/plan.json` + `*/source.json` + `executed_source/` |
| bi64_protocol_factorial_20260906 | 2,064 | 25 | C | 145 条 record 的 `source_snapshot` |
| boundary_state_redesign_20261005 | 1,769 | 28 | B | `e2_results/plan.json` + `source.json` + `executed_source/` |
| brainsync_recovery_20260908 | 1,605 | 53 | A | 4 个 source manifest + `source_snapshot.zip` + `deterministic_replay/` |
| cross_dataset_start_20260906 | 3,996 | 74 | C | `preflight.json` 的 `source_snapshot` + 每 record |
| p300_mainline_20260906 | 2,963 | 76 | C | 140 条 record 的 `source_snapshot`（131 处声明） |
| bidirectional_qc_20260909 | 102 | 41 | D | `execution.json`（101 文件哈希）+ `completion.json` |
| system_implementation_20261004 | 232 | 9 | B | `execution_source.json` + `executed_source/` |
| _retired | 444 | 41 | A | recovery 的归档副本 |
| lda_head_bi61_20260906 | 175 | 31 | C | `retrieved/*/integrity.json`（77 条 prediction+model 哈希） |
| dataset_expansion_audit_20260908 | 119 | 44 | E | 无（下载镜像清单） |

## 附录 B：方法与脚本（只读）

- 盘点/分类：`g8_inventory.py`、`g8_classify.py`、`g8_class2.py`、`g8_fourway.py`；身份字段：`g8_scan.py`、`g8_matrix.py`、`g8_decl.py`。
- 路径泄漏：`g8_paths.py`、`g8_paths2.py`、`g8_respath.py`；哈希对应：`g8_snapshots.py`、`g8_resolve.py`、`g8_hashmatch.py`。
- 全部脚本以 `PYTHONDONTWRITEBYTECODE=1` 运行，未写 run 目录，未执行 git 写命令，未读取密钥值。

