# 分区 G7：experiments/ 69 个顶层入口的 CLI 一致性审计

> 审计者：子代理 G7 · 时间：2026-10-19 · 基准：当前工作树（非 HEAD，429 项未提交；tests 184/185 未跟踪；formal/ 未跟踪）
> 范围：`experiments/*.py` 共 69 个顶层脚本的 argparse 接口（不覆盖 data/transfer/数学/测试/工程/性能等已由其他分区报告的主题）
> 方法：AST 静态提取（69 文件 / 683 个 add_argument）、`--help` 全量只读抽样（子进程 timeout 90s）、危险默认精读、退出码/输出格式统计；未执行任何任务、未写仓库文件、未启动训练
> 只读声明：本报告仅新增 `tmp/audit/G7-cli-consistency.md` 与 `tmp/audit/scratch/` 下临时脚本；无 git 写操作、无依赖安装

## 结论摘要

1. **P1 `--output` 二义且默认覆盖**：53 个文件 / 55 处 `--output`，**22 个当目录、31 个当文件**，54/55 无 help；底层 `save_epoch_dataset` 原子写但**不拒绝已存在**（`src/data/epochs.py:1037-1039`），7 个缓存入口静默覆盖（§4.1）。
2. **P1 写入口无统一覆盖契约**：17 个 `save_epoch_dataset` 调用点中 9 个无存在性检查；目录输出 16 处 `exist_ok=False` vs 9 处 `True`；全仓 0 个 `--force`，仅 1 个 `--resume`（§4.1-4.2）。
3. **P1 `--subjects` 同名四种语法**：int 人数（`run_eeg_loso.py:243`、`run_n2p3_sensitivity.py:273`）、`nargs="+"` 列表（`audit_bi2014a_window.py:104`）、逗号字符串（`prepare_*`）、字面量 `all`（`probe_checkpoint_masks.py:30`）；同一命令跨入口迁移会静默选错被试（§3）。
4. **P1 破坏性维护入口无围栏**：`retire_bi2014a_legacy_derivatives.py` 无 `__main__`、无 dry-run，模块层 `parse_args()` 后直接 `path.unlink()`（`:9-13,43`）；文件被 `formal/lean/verification_plan.json:62` 等 frozen manifest 哈希固定（§4.3）。
5. **P2 设备/资源默认分裂**：`--device` 默认 cuda 5 / cpu 4 / auto 7；`--cpu-threads` 默认 4/32/2/None/未声明；两者都不提示 `AGENTS.md` 的云端 ≥25 线程下限；worker 家族有 10 个名字（§2.4）。
6. **P2 退出码语义过载**：151 处 `parser.error`→2，与“输出已存在”“数据复核失败”共用 2；未捕获异常→1 + traceback；无入口声明退出码（§5）。
7. **P2 输出协议三套且无日志**：JSON 32 / 人类 19 / 混用 8 / `PROGRESS n` 3 / 静默 7（类别可重叠）；107 个 print 中 22 个无 flush；0 个 `logging`；status 词汇 32 个，`completed` 与 `complete` 并存（§6）。
8. **P2 无共享 CLI 层**：`src/` 中 0 个 `ArgumentParser`；69 个 parser / 683 个参数全部手写，508 个无 help、0 个 `--version`；缓存/manifest/plan/worker 四类名字各自演化（§2、§8）。
9. **P2 dry-run 不纯**：`run_source_training_chain.py:649-659` 与 `run_domain_transfer_matrix.py:76→85` 在 dry-run 下仍建目录/写记录；只有 `run_candidate_promotion_matrix.py:3560-3569` 是纯预演（§4.4）。
10. **入口本身健康、内容不合格**：`--help` 实测 67/67 rc=0、stderr 空、无 timeout（平均 3.94s），但 74% 参数无说明；`run_pretrain_supervised.py` 205 行 vs `audit_bidirectional_drift.py` 5 行（§7）。

## 1. 入口规模与解析器结构

- 69 个顶层 `.py` 中 **68 个是 CLI**；`result_batch_contract.py:1-7` 只是 3 个 schema 常量的共享模块，没有 `main`/参数，属于“实验目录里的库文件”。
- 静态统计：**69 个 `ArgumentParser` / 683 个 `add_argument` / 1 个位置参数 / 0 个短选项 / 81 个带 `choices` / 417 个带 `type` / 43 个 `store_true` / 10 个 `BooleanOptionalAction`**。
- `run_pretrain_supervised.py` 是唯一双解析器入口：`:403` 的 dispatch parser（`add_help=False, allow_abbrev=False`）先截获 `--source-chain-plan` 并转发给 `run_source_training_chain.main`，`:447` 才是 59 个参数的正式 parser。同名参数出现两次，但属于有意分流，不是重复定义 bug。
- 只有 2 个脚本使用子命令：`prepare_eeg_dataset.py:90`（moabb/manifest/brainsync，3 个）与 `run_operator_decision_study.py:225-245`（clean/fit/predict/noise-fit/noise-project/operator-lda-fit/operator-lda-predict，7 个）；其余 66 个是扁平选项。
- **67 个脚本有 `if __name__ == "__main__"` 守卫**（66 个定义 `main`，`run_e2_prior_scale.py` 在 guard 内内联）；`retire_bi2014a_legacy_derivatives.py` 没有（`:9-13` 在模块层直接 `parse_args()` 并执行删除，见 §4）。
- 可测试性不一致：**29 个 `main(argv=None)`** 可被 pytest 进程内调用（如 `run_pretrain_supervised.py:401`、`run_source_training_chain.py:1355`），另 37 个 `main()` 只能读 `sys.argv`，还有 2 个无 `main`（`retire_bi2014a_legacy_derivatives.py`、`run_e2_prior_scale.py`）。
- 全局缺失：**0 个 `--version`**、**0 个 `formatter_class`/`epilog`**、`allow_abbrev` 只在 dispatch parser 出现 1 次；argparse 默认前缀匹配（`--out` 可命中 `--output`）在全部 68 个入口生效。

## 2. 参数命名分类

### 2.1 输入：同一概念最多 6 种名字

| 概念 | 现有写法（文件数） |
|---|---|
| epoch 缓存 | `--dataset-cache`(16)、`--source-cache`(2)、`--cache`(1)、`--prepared`(4)、`--snapshot`(6) |
| 源/评估清单 | `--source-snapshot-manifest`(7)、`--evaluation-snapshot-manifest`(6)、`--manifest`(3)、`--raw-artifact-manifest`(2)、`--upstream-manifest`(1)、`--calibration-manifest`(1) |
| 计划 JSON | `--plan`(14)、`--matrix-plan`(2)、`--baseline-plan`(1)、`--fit-plan`(1)、`--training-plan`(1)、`--source-chain-plan`(1) |
| 会话目录 | `--session-dir`(5)、`--calibration-session-dir`(1)、`--inputs`(1) |
| 根目录 | `--root`(6)、`--directory`(1)、`--legacy-source-root`(1)、`--cache-root`(1)、`--runtime-checkout`(1) |

### 2.2 输出：`--output` 同时表示文件与目录（最严重）

- `--output` 出现在 **53 个文件 / 55 处**，其中 **53 处 `required=True`**，只有 `prepare_brainsync_cache.py:38-42`（走 `BRAIN_SYNC_SESSION_DIR`/`BRAIN_SYNC_OUTPUT_DIR`）与 `preflight_artifact_qc.py:187-188`（缺省只打印 JSON，给了才写文件）允许缺省。
- **22 个脚本把 `--output` 当目录**（`args.output.mkdir(...)`）：`run_domain_transfer_matrix.py:76`、`run_source_training_chain.py:479,651`、`run_operator_decision_study.py:250`、`run_cross_dataset_transfer.py:427` 等，完整清单见 §4。
- **31 个脚本把 `--output` 当文件**（`.npz`/`.json`）：`prepare_reference_cache.py:31,49`、`prepare_gtn_dataset.py:32,89`、`prepare_eeg_dataset.py:152,247`、`analyze_candidate_cross_decision.py` 等。
- 另有 3 种并行写法：`--output-dir`（`brainsync_model_task.py`、`prepare_candidate_subject_partitions.py`、`report_brainsync_quality.py`）、`--run-dir` + `--run-name`（`run_eeg_loso.py:245-246`、`run_n2p3_sensitivity.py:271-272`）、`--report`（两个 retire 脚本）。
- 结果：同一个参数名无法从 `--help` 判断是文件还是目录；argparse 不校验，传错时要么 `NotADirectoryError`，要么把 `.npz` 写成目录。

### 2.3 随机种子：10 个 `--seed` + 7 种衍生名

- `--seed` 出现在 10 个训练/分析入口，默认值三套：**0**（6 个：`prepare_source_selection.py:157`、`run_brainsync_cross_decision.py:203`、`run_candidate_cross_decision.py:593`、`run_eeg_loso.py:244`、`run_pretrain.py:137`、`run_within_subject_transfer.py:214`）、**20260906**（2 个：`brainsync_model_task.py:247`、`run_binary_subject_transfer.py:92`）、**20260828**（2 个：`run_n2p3_sensitivity.py:293`、`run_pretrain_supervised.py:518`）。
- 同一仓库里“默认种子”到底是 0 还是日期没有约定；跨入口比较时默认不可比。
- 衍生名：`--source-seeds`/`--target-seeds`（`analyze_training_chain.py`，必填）、`--seed-pair`/`--seed-design`（同上）、`--analysis-seed`、`--bootstrap-seed`、`--target-seed`（`run_cross_dataset_transfer.py`）。

### 2.4 设备与并行：1 个 device + 10 个 worker 名

- `--device`(16) 默认值三分：**cuda**（5：`benchmark_unified_training.py:141`、`run_candidate_promotion_matrix.py:3541`、`run_cross_dataset_transfer.py:154`、`run_domain_transfer_matrix.py:68`、`run_source_pool_matrix.py:145`）、**cpu**（4：`brainsync_model_task.py:236`、`probe_checkpoint_masks.py:35`、`run_binary_subject_transfer.py:94`、`run_source_training_chain.py:1362`）、**auto**（7）。
- `--cpu-threads`(11) 默认 **4 / 32 / 2 / None / 未声明** 五种；其中 4 与 32 都低于 `AGENTS.md` 要求的云端 ≥25 线程下限，且 `--help` 无一处说明该下限。
- worker 家族 10 个名字：`--workers`(6)、`--jobs`(2)、`--artifact-qc-jobs`(4)、`--source-workers`(2)、`--fold-jobs`(2)、`--gpu-fold-jobs`(2)、`--max-workers`(1)、`--subject-workers`(1)、`--result-workers`(1)、`--threads-per-worker`(1)。语义（进程数 / 线程数 / 任务数）混用，无统一单位。

### 2.5 行为开关

- `--dry-run` 11 个；`--resume` **仅 1 个**（`run_candidate_promotion_matrix.py`）；**没有任何 `--force` / `--overwrite` / `--yes`**。
- 显式放宽/删除类：`--allow-incomplete`、`--allow-skipped`、`--allow-software-correctness`、`--allow-oracle-same-selection-adaptation`、`--delete-original`（`retire_bi2014a_participants.py:21`）。命名风格不统一（allow-X vs delete-X vs resume）。
- `--uncompressed` 使用 `store_true`（12 处），而布尔覆盖参数在别处用 `BooleanOptionalAction`（10 处），同类开关两种表达。

## 3. required 与默认值

- 683 个参数中 **187 个 `required=True`（27%）**；required 集中在 `--output`(53)、`--dataset-cache`(15)、`--plan`(13)、`--source-snapshot-manifest`(6)、`--snapshot`(6)、`--prepared`(4)。
- 683 个参数中仅 1 个位置参数（`run_with_dashboard.py:70` 的 `cmd`，`argparse.REMAINDER` 透传实验命令），其余全部命名参数、0 个短选项，输入必须显式命名——这一点是好的，迁移时应保留。
- required 粒度不一致：`--dataset-cache` 在 15/16 个入口 required，唯一 optional 是 `run_binary_subject_transfer.py`（按 stage 校验，`:123` `parser.error`）；而 `run_eeg_loso.py:153`、`run_n2p3_sensitivity.py:270` 直接 required。
- 默认值把本机环境写死（换机器/换 CWD 即失真）：
  - `audit_bi2014a_window.py:102-104`：`--root` 默认 MNE 缓存绝对路径，`--subjects` 默认 `list(range(1,62))`；不带参数即对 61 人做窗口审计。
  - `prepare_bi2014a_candidate.py:49,51-55`：`--root` 默认 `DEFAULT_BI_ROOT`；`--subjects` 默认空串，help 写“default is eligible subjects 01-61”，只能靠阅读代码确认空串等价于全部。
  - `run_eeg_loso.py:245`、`run_n2p3_sensitivity.py:271`：`--run-dir` 默认相对路径 `experiments/runs`，依赖 CWD；从仓库外启动会写到另一棵目录树。
- **`--subjects` 有 4 种互不兼容的语法**：
  1. `type=int` 的**人数**：`run_eeg_loso.py:243`（`np.unique(subject_ids)[: args.subjects]`，`:519`）、`run_n2p3_sensitivity.py:273`（默认 64）；
  2. `nargs="+"` 的**ID 列表**：`audit_bi2014a_window.py:104`（默认 1-61）；
  3. 逗号/区间**字符串**：`prepare_bi2014a_candidate.py:51`、`prepare_bnci2014_008_candidate.py:77-80`、`prepare_eeg_dataset.py:109`；
  4. 字面量 `all`：`probe_checkpoint_masks.py:30`。
  同名参数传同一个值含义不同：`--subjects 7` 在 LOSO 是“前 7 人”，在 window 是“7 号被试”，在 prepare 是“7 号被试”，在 probe 是非法值。
- `--batch-size` 默认 6 种：32/64/256/512/深层默认/未声明；同族的 `run_pretrain.py:109`=256 与 `run_pretrain_supervised.py:514`=512 不一致。
- `--output` 类型：41 处 `type=Path`，12 处裸 `str`（`adapt_eeg_domain.py:90`、`prepare_eeg_dataset.py:152` 等），`--help` 无法体现路径类型，也与 §2.2 的文件/目录歧义叠加。

## 4. 危险默认与写/删行为

### 4.1 无存在性检查的写入口

- 逐个核对 17 个 `save_epoch_dataset` 调用点：**8 个有存在性检查、9 个没有**（涉及 7 个脚本），后者是 `adapt_eeg_domain.py:128`、`prepare_bi2014a_candidate.py:136`、`prepare_bnci2014_008_candidate.py:167`、`prepare_brainsync_cache.py:76`、`prepare_brainsync_eog_matrix.py:67,130-131`、`prepare_eeg_dataset.py:247`、`prepare_gtn_dataset.py:89`。
- 其余直接 `write_text`/`json.dump` 的入口同样少有检查，例如 `analyze_candidate_cross_decision.py`、`audit_bi2014a_window.py`、`build_candidate_cross_decision_manifest.py`。
- 底层 `save_epoch_dataset` 本身是**原子写**（`src/data/epochs.py:1037-1039` 写 `.tmp.npz` 后 `replace`），所以不会写坏文件，但**没有任何“目标已存在”拒绝**；CLI 层也不检查，等于默认允许覆盖。
- 有保护的实现风格也不统一：`prepare_reference_cache.py:38-42`、`prepare_multidomain_source.py:166-168`、`prepare_target_calibration_cache.py:61-62` 抛 `FileExistsError`（退出码 1 + traceback），而 `analyze_training_chain.py:78`、`brainsync_model_task.py:273` 用 `parser.error`（退出码 2 + usage）——同一语义两种退出码。

### 4.2 目录型输出的 `exist_ok` 分裂

- `--output` 作为目录的 22 个脚本共 25 处 `mkdir`，**16 处 `exist_ok=False`**（`run_domain_transfer_matrix.py:76`、`run_cross_dataset_transfer.py:427`、`run_information_bound.py:136`、`run_operator_decision_study.py:250` 等），**9 处 `exist_ok=True`**（`run_source_training_chain.py:651`、`run_source_pool_matrix.py:169`、`run_public_pool_targets.py:126`、`verify_pool_matrix.py:164` 等）。
- 前者重跑立即失败，后者把新结果静默并入旧目录；两者都没有 `--resume`/`--force` 解释语义（全仓仅 `run_candidate_promotion_matrix.py:3550` 有 `--resume`）。
- 结果是“同一命令能否重跑”取决于入口作者选了哪个 `exist_ok`，而不是统一契约。

### 4.3 破坏性脚本

- **`retire_bi2014a_legacy_derivatives.py` 是全仓唯一无 `__main__` 守卫的 CLI**：`:9-12` 在模块层 `parse_args()`，`:43` 对匹配的 `.npz` 执行 `path.unlink()`，无 `--dry-run`、无确认、无 `--force` 门槛；`:41-45` 每删一个就重写 `--report`。
  - 缓解：`--root`/`--report` 均 required，所以“直接 import 且无 argv”会在 `parse_args()` 处以退出码 2 终止，不会删文件；但只要调用方进程的 `sys.argv` 恰好带这两个参数（pytest 参数化、包装脚本），import 就会删除。
  - 该文件被 `formal/lean/verification_plan.json:62` 与多个 `frozen/**/manifest.json` 以哈希固定；修改它需要同步快照，不能当作普通脚本重构。
- `retire_bi2014a_participants.py` 是正面样例：删除必须显式 `--delete-original`（`:21`），且先写 replacement、reload 校验（`:84-89`）后才 `path.unlink()`（`:97`）。
- `remove_bi2014a_rejected_raw.ps1` 同为破坏性维护入口，不在 69 个 py 的 CLI 统计内。

### 4.4 `--dry-run` 不纯

- 11 个入口有 `--dry-run`，但语义是“不拟合”，不是“不写盘”：
  - `run_source_training_chain.py:649-659`：dry-run 仍要求输出目录为空/不存在，仍 `mkdir` 并写 `source_chain_record.json`；
  - `run_domain_transfer_matrix.py:76` 在 `:85` 的 dry-run 分支**之前**就 `mkdir(exist_ok=False)` 并写 `matrix.json`；dry-run 失败于“目录已存在”，也会留下新目录；
  - `run_candidate_promotion_matrix.py:3560-3569` 是正确样例：只打印 DAG 后 `return`，不落盘。
- 结论：`--dry-run` 无统一契约，不能作为“安全预演”依赖；需要区分 `--plan-only`（只读校验）与 `--dry-run`（跳过拟合但仍写计划）。

## 5. 退出码与错误通道

- 默认路径：`main()` 返回 None → 解释器退出 0；67 个 `__main__` 中只有 2 个用 `main()` 返回值驱动退出码（`run_information_bound.py:184,188`、`run_operator_decision_study.py:264,268`）。
- `--help` 实测 67/67 退出 0（§7）。
- 参数/前置校验失败：**151 处 `parser.error`（28 个文件）** → argparse 统一退出 2 并把 usage 打到 stderr；其余 40 个脚本没有显式校验，非法组合推迟到运行中途才失败。
- 未捕获异常：`FileExistsError`、`ValueError`、`RuntimeError` 等 → traceback 到 stderr + 退出 1（例：`prepare_reference_cache.py:38-42`、`run_domain_transfer_matrix.py:74,93`）。
- 显式退出：`brainsync_model_task.py:424,442`（1）、`preflight_artifact_qc.py:190`（`SystemExit(str)` → 1）、`retire_bi2014a_participants.py:108`（2）、`run_e2_prior_scale.py:535`（1）、`run_with_dashboard.py:116`（透传子进程 rc，含 Ctrl-C 的 130）。
- **退出码 2 被三种语义共用**：argparse 用法错误、输出已存在（`analyze_training_chain.py:78`、`brainsync_model_task.py:273`）、数据复核失败（`retire_bi2014a_participants.py:108`）。调用方无法区分“命令写错”与“结果需人工复核”。
- 没有任何入口在 `--help` 中声明退出码语义；除 `run_with_dashboard` 外也没有调用方级错误信封。

## 6. stdout 与日志格式

- 全仓 107 个 `print`，其中 85 个 `flush=True`、**22 个无 flush**（经管道/子进程收集时可能丢尾部输出）。
- 输出格式不统一（类别可重叠）：
  - **32 个以 JSON 为主**（`print(json.dumps(...))`）：`analyze_*`、`prepare_source_selection_vectors.py`、`verify_pool_matrix.py` 等；
  - **19 个人类可读 `[tag]` 行**（`[prepared]`、`[record]`、`[summary]`、`[dashboard]`），其中 3 个另用 `PROGRESS n`；
  - **8 个混用**（先 `[prepared]` 再 `json.dumps`，如 `prepare_bi2014a_candidate.py`、`run_pretrain.py`）；
  - `PROGRESS n` 无 schema、无时间戳、无总步数约定（`prepare_brainsync_cache.py:67,75,88`、`report_brainsync_quality.py`、`brainsync_model_task.py`）；
  - **7 个完全静默**（`run_domain_transfer_matrix.py`、`run_erp_comparator.py`、`fit_source_temporal_basis.py` 等），只写文件（`result_batch_contract.py` 无 CLI，不计入）。
- 全仓 **0 个 `logging` 调用**；仅 `brainsync_model_task.py:441` 把错误写到 stderr，其余靠 traceback。
- JSON 已有 23 个 `n2p3_*/[0-9]` schema 值（可复用），但 status 词汇表 **32 个值**，含 `completed`(13) 与 `complete`(2)、`failed`(3) 与 `fit_failed`/`fit_failure` 并存。
- 机器可读性不一致：只有 1/55 的 `--output` 有 help 文本；JSON 是否含 `output` 键也不统一（`run_source_training_chain.py:1392` 只打印 4 个字段，不含 output）；`run_domain_transfer_matrix.py` 只写文件不打印。

## 7. --help 抽样结果

- 方法：`python experiments/<f>.py --help`，`timeout=90s`、`PYTHONDONTWRITEBYTECODE=1`、全子进程；未执行任何任务。
- **67/67 抽样成功：rc=0、stderr 为空、无 timeout**；平均 3.94s，最慢 11.26s（`prepare_eeg_dataset.py`）。跳过 2 个：`result_batch_contract.py`（无 CLI）、`retire_bi2014a_legacy_derivatives.py`（模块层 `parse_args`，保守跳过）。
- 输出体量差异极大：<8 行 6 个（`audit_bidirectional_drift.py` 5 行、`watch_resources.py` 7 行、`freeze_source_snapshot.py` 6 行），8-15 行 23 个，16-30 行 23 个，>30 行 15 个；`run_pretrain_supervised.py` **205 行**、`run_eeg_loso.py` 151 行。
- usage 首行最长 78 字符（argparse 默认 80 列换行），全部使用默认 formatter，无 epilog/示例。
- 子命令入口的顶层 `--help` 只列子命令：`prepare_eeg_dataset.py` 仅 10 行，`usage: prepare_eeg_dataset.py [-h] {moabb,manifest,brainsync} ...`；具体参数须 `... moabb --help`。
- 可用性缺口：508/683 参数无 help；54/55 `--output` 无说明；无 `--version`；无示例；**没有任何一处说明“目标已存在时是覆盖还是拒绝”**。

## 8. 重复定义与共享辅助缺失

- **`src/` 中 0 个 `ArgumentParser`/`add_argument`**：68 个 CLI 全部在 experiments 内各自建 parser，683 个参数无任何共享构造器。
- 参数家族重复：6 种缓存名、6 种 manifest 名、6 种 plan 名、10 种 worker 名（§2.1/2.4），每个入口各写一遍 `add_argument`。
- 唯一同名重复是 `run_pretrain_supervised.py` 的 `--source-chain-plan`（`:404` dispatch、`:448` 正式），属有意分流，不是 bug；但它使该入口的 `--help` 语义分裂为“本脚本参数”与“转发给 source chain 的参数”两套。
- 解析辅助各自为政：`_parse_int_list`（`prepare_eeg_dataset.py:25`）、`_parse_subjects`（`prepare_bnci2014_008_candidate.py:48`）、`_parse_fold_indices`（`run_n2p3_sensitivity.py:238`）、`_parse_optional_float`（`prepare_eeg_dataset.py:63`）、`_parse_bands`（`run_pretrain.py:54`）分别实现；逗号分割在 `adapt_eeg_domain.py:100`、`prepare_multidomain_source.py:150`、`prepare_reference_cache.py:35` 各写一次。
- 已有重复项（不重复展开）：7 种 `sys.path` 引导、3 份 atomic JSON、`run_domain_transfer_matrix.py:44` 以 `argv.index("--subjects")` 解析其他脚本 CLI——见 `tmp/audit/05-research-experiments.md` §C。
- 后果：改一个参数语义要改 69 处；无法用单测锁定契约；也无法生成统一的 `--help`/`--version`。

## 9. 统一 CLI 约定建议

### 9.1 命名表（新入口强制，旧入口别名过渡）

| 语义 | 统一名 | 规则 |
|---|---|---|
| 单个输入文件 | `--input` | `type=Path`，必填；缓存类保留语义化别名（`--dataset-cache` 等）并 `help` 说明 |
| 输出文件 | `--output` | **必须是文件**且后缀受校验（`.npz`/`.json`） |
| 输出目录 | `--output-dir` | **不得**再用 `--output` 表示目录 |
| 运行目录 | `--run-dir` + `--run-name` | 仅训练/矩阵入口；两者同时给或同时缺省 |
| 随机种子 | `--seed` | `int`，默认 0；分析入口另设 `--analysis-seed`，不得复用 `--seed` |
| 设备 | `--device {auto,cpu,cuda}` | 默认 `auto`；禁止默认 `cuda` |
| CPU 线程 | `--cpu-threads` | 默认 `None`=进程配额；在 runtime 强制执行 AGENTS 的云端 ≥25 下限，CLI help 注明 |
| 进程数 | `--workers` | 统一“进程数”；任务类用 `--<kind>-workers`；线程用 `--threads-per-worker` |
| 覆盖 | `--force` | 默认拒绝已存在；`--force` 才覆盖；目录合并另设 `--resume` |
| 预演 | `--plan-only` / `--dry-run` | `--plan-only` 只读校验不落盘；`--dry-run` 允许写计划但必须声明写哪些文件 |

### 9.2 退出码与错误信封

- `0` 成功；`2` 仅 argparse 用法错误；`3` 输入/manifest 校验失败；`4` 输出已存在（未 `--force`）；`5` 任务部分失败（结果已写盘）；`130` 中断。
- 失败时 stdout 最后一行输出 `{"schema": "n2p3_cli_error/1", "status": "failed", "code": 3, "error": {"type", "message"}}`，traceback 仅进 stderr（`--verbose` 时）。
- 把 `parser.error` 从“输出已存在/数据复核”场景移出，改用上述专用码。

### 9.3 stdout 契约

- 正常结束：stdout 最后一个非空行必须是单个 JSON 对象，至少含 `schema`、`status`、`output`（或 `outputs`）、`elapsed_s`；人类进度改走 stderr 或 `--quiet`。
- 保留 `PROGRESS n` 但补 `schema: n2p3_progress/1`、`step`、`total`；`--json` 模式下不打印人类行。
- 所有 `print` 统一 `flush=True`；禁止在 stdout 混入库的 warning（改 `logging` 到 stderr）。

### 9.4 共享实现

- 新增 `experiments/_cli.py`：`build_parser()`, `add_common(parser, *, output_kind, seed, device, workers)`, `require_new_path(path, force)`, `resolve_workers/cpu_threads`, `write_json_atomic`, `run(main)`（捕获异常→错误信封+退出码）。
- 新增 `tests/research/test_cli_contract.py`：AST 扫描 68 个入口，断言 `--help` 可运行、`--output` 有 help、输出类参数必填、无模块层 `parse_args`、无 `default="cuda"`。
- 迁移顺序：先加 helper（不改行为）→ 新入口使用 → 旧入口逐个替换并保留别名 → 最后统一 `--help`。

## 10. 迁移清单

| 优先级 | 文件/范围 | 动作 | 风险/备注 |
|---|---|---|---|
| P1 | `retire_bi2014a_legacy_derivatives.py:9-13,43` | 加 `__main__` 守卫、`--dry-run`、`--confirm` 令牌；默认只报告不删除 | 文件被 frozen manifest 哈希固定，改动需同步快照 |
| P1 | 7 个静默覆盖缓存的入口（§4.1） | 调用 `require_new_path`；新增 `--force` | 会改变现有重跑习惯；`save_epoch_dataset` 本身保持原子写 |
| P1 | `--output` 的 22 个目录型入口 | 拆为 `--output-dir`，保留 `--output` 别名 + deprecation 警告 | 与 `run_*`/`prepare_*` 两条命名线合并 |
| P1 | `--subjects` 7 个入口 | 统一为逗号/区间字符串；`run_eeg_loso`/`run_n2p3_sensitivity` 改名 `--subject-limit` | 需同步 `run_domain_transfer_matrix.py:44` 的 argv 解析 |
| P2 | `--seed` 10 个入口 | 统一默认 0 或统一日期常量，并在 help 写明 | 影响默认结果的可比性，需记录在 evidence |
| P2 | `--device` 16 / `--cpu-threads` 11 | 默认 `auto`/`None`，runtime 执行 ≥25 下限 | 需 `train/runtime` 侧配合 |
| P2 | 全部 68 个入口 | 退出码与错误信封；`parser.error` 只保留用法错误 | 调用方需同步解析新码 |
| P2 | 107 个 print / 7 个静默入口 | 统一 JSON 尾行 + stderr 进度；补 `flush=True` | 日志采集脚本需改 |
| P2 | 683 个参数 | 补 help（优先 `--output`/`--device`/`--seed`/`--workers`）；加 `--version` 与 epilog 示例 | 纯文档改动，但会改变 frozen 哈希 |
| P2 | `run_domain_transfer_matrix.py:76,85` | `mkdir` 移到 dry-run 分支之后 | 保持目录语义不变 |
| P2 | `experiments/_cli.py` + contract test | 新增共享层与 CI 契约测试 | 新文件需入库并加入 `git ls-files` 检查 |

## 附录：证据文件

- `tmp/audit/scratch/g7_cli_inventory.json`：69 文件 / 683 参数的 AST 明细（opts、required、default、action、type、choices、help、行号）。
- `tmp/audit/scratch/g7_help.json`：67 个入口的 `--help` 实测（rc、耗时、stdout 行数、usage、stderr）。
- `tmp/audit/scratch/g7_cli_inventory.py`、`g7_help_sample.py`：可复跑的只读脚本（`PYTHONDONTWRITEBYTECODE=1`，未执行任务）。
