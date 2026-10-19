# G5：configs/ 9 个 JSON 及其消费者审计

- 分区：G5（configs 补充分区；不重复 data 核心、transfer 核心、数学统计、tests、工程、repo、docs、security、performance、Lean 已报告内容）
- 仓库：D:\BCI\n2p3-net；HEAD f1a303c；工作树 431 项未提交（4 个 configs 未跟踪）
- 方法：9 个 JSON 全文、消费者 grep/阅读、外部 BrainSync 采集端只读核对、只读 git、`python -c`/`tmp/audit/scratch` 探针（PYTHONDONTWRITEBYTECODE=1）
- 只读声明：未修改/删除/重命名任何已有文件，未运行 git 写命令，未安装依赖，未启动训练；本报告与 `tmp/audit/scratch/g5_*.py` 为唯一写入。

## 0. 结论摘要

1. **P1 预处理配置未接入实际交付路径**：`configs/brainsync_bidirectional_preprocessing.json` 与代码合同逐字段相等（探针），但工作站 manifest 的唯一 parameter_option 不传 `--preprocessing-config`（`configs/brainsync_workstation.model.json:13`），入口因此回退到硬编码 `BRAIN_SYNC_PREPROCESSING`（`experiments/prepare_brainsync_cache.py:71`、`src/data/brainsync.py:48-50`）。用户在采集端改配置或按文档复制配置都不会改变实际预处理。
2. **P1 采集计划与真实采集端冲突**：计划两次声明 `rest_sec: 0`（`configs/brainsync_collection_plan.json:8,20`），而采集端 `ExperimentConfig.rest_sec` 默认 20 s（外部 `gtn_workstation/models.py:396`），10 block 产生 9 次休息、总时长估算 `3 + 4050×1.5 + 9×20 = 6258 s`（外部 `ui.py:2168`），计划只给 6075 s SOA 预算且把休息写成 0。
3. **P1 三份几何/合同副本无交叉校验**：`bidirectional_preprocessing`、`epoch_support`、`workstation.model` 与本分区共享 tmin/tmax/128 Hz/179，但只有 epoch_support 有“等于代码合同”的测试（`tests/brainsync/test_acquisition_contract.py:13-21`）；preprocessing 配置无任何测试读取，manifest 也只校验 5 个字段（`tests/brainsync/test_brainsync.py:342-353`）。当前值一致（探针），但改一处不会触发失败。
4. **P1 4/9 配置未入库且未被默认消费**：`brainsync_bidirectional_preprocessing`、`cloud_training_resources`、`source_pool_expansion.example`、`unified_training_budget` 未跟踪；干净克隆下只有 cloud 配置会让测试硬失败（`tests/train/test_source_execution_resources.py:214`，performance 分区已报），其余 3 个是文档死路径或静默回退到代码默认值（今天恰好相等）。
5. **P2 示例/校验器缺口**：`candidate_promotion.example.json` 的 7 个输入全部不存在（dry-run 直接 `FileNotFoundError`）；`source_pool_expansion.example.json` 可通过解析（48 任务）但无测试；`cloud_training_resources`/`unified_training_budget` 无 schema、无未知字段校验；9 个文件无统一 JSON Schema/校验器。

**未发现 P0**（没有配置值会直接导致已执行训练/数据被错误解释；上述均为门禁/复现/预算缺口）。

## 1. 文件清单与元数据

| 文件 | 跟踪 | schema / version | 测试读取 | 运行消费者 | 核查结论 |
|---|---|---|---|---|---|
| brainsync_acquisition_reference.json | 是 | `brainsync-acquisition-reference/1` | 是（精确相等） | 外部前端副本 `models/acquisition_reference.json` | 值正确；仓库内无运行消费者 |
| brainsync_bidirectional_preprocessing.json | **否** | 无 schema，仅 `name: brainsync_bidirectional_v1` | **否** | `--preprocessing-config`（可选手工参数） | 与代码合同相等，但默认交付路径不读 |
| brainsync_collection_plan.json | 是 | `n2p3_brainsync_collection_plan/1` | 算术子集 | 无代码；仅文档/人工 | 4050/450/6075 自洽；rest_sec 与真实端冲突 |
| brainsync_epoch_support.json | 是 | `n2p3-epoch-support/1` | 是（等于合同） | 外部前端副本 `models/epoch_support.json` | 值正确；与 preprocessing 副本无联动 |
| brainsync_workstation.model.json | 是 | `brainsync-model-manifest/1`，version 4.1.0 | 部分（5 字段） | 外部前端启动子进程 | 路径存在；version/接口/参数选项未校验 |
| candidate_promotion.example.json | 是 | `n2p3_candidate_promotion_plan/2` | **否** | `run_candidate_promotion_matrix.py` | 结构可载入；7 个引用输入不存在，无法 dry-run |
| cloud_training_resources.json | **否** | 无 schema | 是（floor 路由） | 仅 `source_execution.resolve_source_resources`，且需手工内联进 plan | 25/4 数值正确；内存下限未声明 |
| source_pool_expansion.example.json | **否** | `n2p3_source_pool_plan/1` | **否** | `source_matrix_tasks`（经 `--matrix-plan`） | 可解析，48 任务；无测试 |
| unified_training_budget.json | **否** | 无 schema | **否** | `--training-budget`（可选手工参数） | 与 dataclass 默认值完全相等；非“默认加载文件” |

元数据补充：5 个已跟踪文件最后一次提交分别落在 908a0cd/f1a303c/98ce531；4 个未跟踪文件均未被 `.gitignore` 命中（`git check-ignore` 全部 not-ignored）。9 个文件 sha256 与 `doc/assets/deep_chain_audit_20260929/entry_manifest.json:6-14` **全部一致**，说明不是刚改动的“在途配置”，而是长期游离于版本控制之外。

## 2. 逐文件核查

### 2.1 brainsync_acquisition_reference.json
- 5 个字段与 `src/data/contract.py:23-31` 完全一致，测试逐字典相等（`tests/brainsync/test_acquisition_contract.py:24-34`）。
- `label` 用 “left earlobe”、`electrode` 用 “ear_clip”，语义上可共存；外部前端只消费 `label`（`gtn_workstation/models.py:70-76`）。无缺陷，仅提示：`site`/`electrode` 没有消费者校验。

### 2.2 brainsync_bidirectional_preprocessing.json
- 字段完整且合法：`parse_preprocessing_spec` 通过，`spec == BRAIN_SYNC_PREPROCESSING` 为 True；原始 JSON 与 `spec.record()` 双向无差集（探针 `tmp/audit/scratch/g5_probe.py`）。校验逻辑见 `src/data/epochs.py:230-249`（必填/未知字段）与 `:100-211`（值域）。
- **P1 未接线**：工作站 manifest 的唯一参数选项只传 `--session-dir/--output/--invalid-session`（`configs/brainsync_workstation.model.json:13`）；入口仅有 `--preprocessing-config` 时才读 JSON（`experiments/prepare_brainsync_cache.py:50-51,71`）。因此“显式配置为 `configs/...`”（`doc/workstation.zh.md:43`）与“从现行配置复制后修改”（`:28`）在实际 GUI 路径上不生效。
- **P1 无测试**：仓库内没有任何测试读取该文件；唯一名字相同的测试断言的是代码常量（`tests/brainsync/test_brainsync.py:563-575`）。它未跟踪，干净克隆中直接缺失。
- P2：无 `schema` 字段，只有 `name` 充当版本；若字段语义变化，`name` 不会自动失效。

### 2.3 brainsync_collection_plan.json
- 算术自洽：`9×10×45=4050`，`4050/9=450`，`4050×1.5=6075`；测试复核（`tests/brainsync/test_brainsync.py:356-392`），本分区复算一致。
- **P1 rest 冲突**：计划两处 `rest_sec: 0`（`configs/brainsync_collection_plan.json:8,20`），而真实采集端默认 20 s/block 间隔（外部 `gtn_workstation/models.py:392-396`），10 block 有 9 次休息（`tests/brainsync/test_acquisition_contract.py:103`）；采集端自身估算为 `3 + trials×soa/1000 + (blocks-1)×rest_sec`（外部 `ui.py:2168`），即 6258 s，计划名义 SOA 6075 s 少 180 s。计划是 `prospective_not_collected`（`:3`），这是下次采集会直接照做的协议记录。
- P2 语义含混：`decisions_per_participant: 9` 在乘法中实际充当“每个刺激的 9 次 flash/9 个数字”，而 block 数是 10（`:6-14`）；测试只验证 `len(target_sequence)==9`（`test_brainsync.py:361-362`），无法区分“9 个决策”与“9 个目标数字/9 次重复”。
- P2 测试阶段不可复算：`decisions_per_participant=27`、`decisions_per_target=3`、`primary_repetitions=5`、1 block×45（`:19-27`）之间没有说明 27 个决策如何由 45 个 epoch 支撑；测试只验 `9×3=27`（`test_brainsync.py:386-389`），未验决策与重复的映射。
- P2 文案：`target_sequence_role` 的 “participants_collection” 语法不成立（`:12`）。

### 2.4 brainsync_epoch_support.json
- 与 `contract.epoch_support_record()` 相等（`tests/brainsync/test_acquisition_contract.py:13-21`），也与 preprocessing 配置的 tmin/tmax/128/179 相等（本分区探针）。
- **P1 交叉校验缺失**：`epoch_support` 是 `SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT` 的导出（`src/data/contract.py:147-157`），而 BrainSync 实际用 `BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT`（`src/data/contract.py:124-130`）；二者今天几何相同纯属窗口参数相同。测试只把前者与代码比较，不把 `configs/brainsync_bidirectional_preprocessing.json` 与任一合同比较；改预处理窗口（例如 tmax 1000/n_times 154）仍可通过 `parse_preprocessing_spec`，但 epoch_support 与外部 app 的 guard 仍是 1200/179。
- P2 命名：同一量在 preprocessing 叫 `sfreq`，在这里叫 `model_sample_rate_hz`；schema 前缀用连字符 `n2p3-epoch-support/1`，而其它 n2p3 配置用下划线（见 §4）。

### 2.5 brainsync_workstation.model.json
- 路径真实存在：`../.venv/Scripts/python.exe`、`../experiments/prepare_brainsync_cache.py` 均解析成功（探针）；`session_schema` 与 `BRAIN_SYNC_SESSION_SCHEMA` 相等（探针）。
- 测试只断言 `session_schema`、输入通道/类型、输出 EEG 通道、`eog_in_classifier`（`tests/brainsync/test_brainsync.py:342-353`）；**`version`、`layout_schema`、`dataset`、`time_base`、`events`、`rest_policy`、`coordinates`、`epoch_cache`、`attestation` 均无校验**。若入口改名/参数改名，仓库测试不会发现，只有外部 GUI 加载时才报错。
- P2 manifest 形状漂移：同名 schema 的外部示例要求 `input_contract.session_manifest/recording_path/preprocessing_contract` 与 `runtime` 块（外部 `models/model_manifest.example.json:16-37`），本文件用 `dataset/time_base/rest_policy/coordinates` 且无 `runtime`；外部 GUI 当前只消费 name/version/python/entrypoint/tasks/parameter_options（外部 `ui.py:2725-2746`），所以不报错，但“schema /1”没有唯一字段契约。
- P2 version 4.1.0 与 `session_schema .../4` 无映射/变更记录；前端示例 version 是 0.1.0，版本语义未定义。

### 2.6 candidate_promotion.example.json
- 加载器结构校验很严：schema /2、stage=development、必填/可选键、arms/contrasts（`experiments/run_candidate_promotion_matrix.py:86,1287-1320,1332-1354`）。示例可被 `load_plan` 成功解析（探针：4 source arms、2 eval arms、6 contrasts）。
- **P2 输入全缺失**：`source_snapshot_manifest`、`target_cache`、3 个 source cache 共 7 个路径不存在；`--dry-run` 在 `_verify_inputs` 处 `FileNotFoundError`（`run_candidate_promotion_matrix.py:2497-2507`，本分区实跑复现）。作为模板尚可，但文件没有 README 说明“先准备这些产物”，且无测试读取它。
- P2 `--root` 语义：相对路径默认以 plan 目录（configs/）解析；一旦按帮助传 `--root <repo>`（`:3536-3539`），`../experiments/...` 会解析到仓库外并被 `output_root escapes` 拒绝（`:1379-1384`，本分区复现）。示例只在不传 `--root` 时可用，与其“可移植 plan”意图冲突。
- P2 示例的 `target_subjects` 仅 2 人（`:6-17`），与 `unified_training_budget.development_people_per_domain=2` 一致；但 `training.epochs=30` 与 unified `target_epochs=90` 属不同阶段，未见文档解释两者边界（易被误读为冲突）。

### 2.7 cloud_training_resources.json
- **P1 未入库**（详见 §5）；5 个入口不读它、只有测试读取字面路径（`tests/train/test_source_execution_resources.py:214`），performance 分区已就 25/4 floor 覆盖，本分区只补配置层面。
- P2 内存包络未声明：`source_worker_memory_gib=18`、`shared_staging_memory_gib=10`、`host_memory_fraction=0.95`（`:7-9`）意味着 4 组至少需要 `82/0.95≈86.32 GiB` 空闲主机内存；实测 86 GiB 失败、87 GiB 通过（探针）。`gpu_task_memory_gib=4` 还隐含 ≥20 GiB 空闲显存（0.8×20/4=4）。AGENTS 与文档只写 25 CPU/4 GPU，未写这两个内存下限。
- P2 无 schema/未知键校验：`resolve_source_resources` 逐键取值且允许未知键（`src/research/source_execution.py:166-213`）；`source_load_workers` 甚至不在 resolver 内，而由调用方读取（`experiments/run_source_training_chain.py:492`）。拼错键会静默使用默认值，而不是失败。
- P2 `source_load_workers=1`（`:5`）把 4 个 source group 的加载并发压到 1；代码允许但文档未解释，与 AGENTS“资源感知并行准备”存在张力（`run_source_training_chain.py:492-494`）。

### 2.8 source_pool_expansion.example.json
- 结构合法：`source_matrix_tasks(plan)` 成功生成 48 个任务（3 seeds × 4 targets × 2 pools × 2 doses），池 `B7` 与冻结定义逐项一致（`src/research/source_catalog.py:10-18`），`B7_plus_three` 是合法新名（`source_catalog.py:48-49`）。加载器还拒绝未知顶层键（`src/research/source_matrix.py:86-98`）。
- P2 无测试：仓库没有任何测试读取该文件；本次靠手工探针才验证。`product_pools=[]`/`standalone={}`（`:15-16`）使示例只覆盖 Bpublic，未示范文档声称的独立布局与 product 池（`doc/engineering.zh.md:101-102`）。
- P2 新增三库 `BI2012/Cattan2019_VR/BrainForm` 有 reader（`src/data/source_parsers.py:204-205`），但示例不代表数据已准入；文档已注明“模板”，可接受。

### 2.9 unified_training_budget.json
- 7 个字段与 `UnifiedTrainingBudget` 默认值逐项相等（`src/train/early_stopping.py:16-22`；探针 `config == dataclass().record()`）。校验在 dataclass：未知键会 TypeError，target_epochs/people 必须正整数，seed 非负（`:24-30`）。
- **P2 “默认文件”不成立**：`doc/engineering.zh.md:291` 称“默认文件 `configs/unified_training_budget.json`”，但 `--training-budget` 是可选的（`experiments/brainsync_model_task.py:241-260`），代码不会自动加载该路径；省略时使用 dataclass 默认值。今天两者相等，所以行为一致；一旦有人只改 JSON，省略参数的调用不会变。
- P2 无测试读取该文件，也没有“配置==默认值”的守门测试；值重复维护，改 dataclass 默认值不会提醒同步 JSON。



## 3. 跨文件冲突与一致性矩阵

| # | 冲突/缺口 | 证据 | 现实影响 | 级别 |
|---|---|---|---|---|
| X1 | 预处理配置未被交付路径消费 | `configs/brainsync_workstation.model.json:13` 无 `--preprocessing-config`；`experiments/prepare_brainsync_cache.py:50-51,71`；`src/data/brainsync.py:48-50` | 改 JSON/按文档复制配置不改变 GUI 实际预处理；模型与声明可静默不一致 | P1 |
| X2 | collection plan rest_sec=0 vs 采集端 20 s | `configs/brainsync_collection_plan.json:8,20`；外部 `gtn_workstation/models.py:392-396`；外部 `ui.py:2168` | 10 block 少记 180 s（6075 vs 6255/6258 s），协议记录与执行端不一致 | P1 |
| X3 | epoch_support ↔ preprocessing 无交叉测试 | `src/data/contract.py:124-130,147-157`；`tests/brainsync/test_acquisition_contract.py:13-21`；preprocessing 无测试 | 改一处窗口仍可通过各自校验，外部 app guard 与后端缓存可漂移 | P1 |
| X4 | manifest version/接口未校验 | `tests/brainsync/test_brainsync.py:342-353` 仅 5 字段；`configs/brainsync_workstation.model.json:4,17-33` | 入口/参数/版本改名仓库测试仍绿，GUI 运行才失败 | P2 |
| X5 | manifest input_contract 形状与前端示例不同 | 本文件 `:17-27` vs 外部 `models/model_manifest.example.json:16-37`（含 runtime 块） | 同 schema /1 两套字段契约；当前 GUI 忽略这些块所以不报错 | P2 |
| X6 | unified “默认文件”无默认加载 | `doc/engineering.zh.md:291`；`experiments/brainsync_model_task.py:241-260` | 只改 JSON 不影响省略 `--training-budget` 的调用；今天值相等所以难发现 | P2 |
| X7 | candidate 示例 7 个输入不存在 | `configs/candidate_promotion.example.json:4-5,21,28,35,45`；`run_candidate_promotion_matrix.py:2497-2507` | `--dry-run` 即 FileNotFoundError；示例不可执行且无测试 | P2 |
| X8 | candidate 示例与 `--root` 不兼容 | `:4-5,98` 的 `../`；loader `:1321,1379-1384` | 传 `--root` 即 `output_root escapes`；只能在 configs/ 为基准时用 | P2 |
| X9 | cloud 内存包络未声明 | `configs/cloud_training_resources.json:7-10,12`；`source_execution.py:178-190`；探针 86 GiB 失败/87 GiB 通过 | 25/4 达标但 <86.32 GiB 主机显式失败；<20 GiB 显存也失败，文档未写 | P2 |
| X10 | collection decisions 语义不可复算 | `configs/brainsync_collection_plan.json:6-14,19-27`；`test_brainsync.py:361-389` | 27 决策/45 epoch/3 per target/5 reps 的映射无定义，测试只验乘法 | P2 |

一致性通过项（避免误报）：
- `brainsync_bidirectional_preprocessing` 与 `BRAIN_SYNC_PREPROCESSING` **逐字段相等**；`epoch_support` 与 `epoch_support_record()` 相等；`acquisition_reference` 与合同相等；`unified_training_budget` 与 dataclass 默认值相等；`source_pool` 示例生成 48 个合法任务。以上均由本分区探针/测试复核。
- `cloud_training_resources` 的 25/4 数值与 floor 自洽：resolver 返回 `cpu_budget=25, source_workers=4, target_workers=4, gpu_slots=4, native_threads_per_worker=6`（探针，88 GiB/32 GiB 条件）。

## 4. schema、版本、单位与命名

| 维度 | 现状 | 证据 | 级别 |
|---|---|---|---|
| schema 覆盖 | 9 个中 6 个有 schema；preprocessing/cloud/unified 无 | 各文件首行；`brainsync_bidirectional_preprocessing.json:2` 只有 name | P2 |
| schema 命名 | 连字符与下划线混用：`n2p3-epoch-support/1` vs `n2p3_source_pool_plan/1`、`n2p3_brainsync_collection_plan/1`、`n2p3_candidate_promotion_plan/2` | 各文件 2 行 | P2 |
| 版本语义 | 只有 manifest 有 semver 4.1.0；其余用 schema 后缀；无变更记录/兼容矩阵 | `brainsync_workstation.model.json:4`；`src/data/brainsync_contract.py:28` | P2 |
| 单位 | 时间键均带 `_ms`/`_sec`/`_hz`/`_v`，内存键带 `_gib`；但 `sfreq`（preprocessing）与 `model_sample_rate_hz`（epoch_support）同量不同名 | `brainsync_bidirectional_preprocessing.json:3`；`brainsync_epoch_support.json:5` | P2 |
| 阈值命名 | `reject_threshold_v: null` 与代码“新入口拒绝固定阈值、仅读历史”一致 | `configs/...preprocessing.json:28`；`src/data/epochs.py:83-85` | 通过 |
| 布尔/可空 | `labels_required_for_inference=false`、`model_and_recipe_frozen_before_test=true` 与测试一致 | `collection_plan.json:29,31`；`test_brainsync.py:390-392` | 通过 |

## 5. 未跟踪配置的入库影响

- 跟踪状态：`git ls-files configs` = 5 个；未跟踪 = `brainsync_bidirectional_preprocessing.json`、`cloud_training_resources.json`、`source_pool_expansion.example.json`、`unified_training_budget.json`。全部 not-ignored，说明是遗漏而非有意本地文件。
- 干净克隆影响分级：
  1. **硬失败**：`tests/train/test_source_execution_resources.py:214` 直接 read_text cloud 配置 → FileNotFoundError；performance 分区已列为 C6，本分区不重复根因分析。
  2. **文档命令失败**：`doc/workstation.zh.md:5,28,43`、`doc/engineering.zh.md:101,291,307` 指向的文件在 clone 中不存在；照文档执行会先遇到缺文件。
  3. **静默回退**：preprocessing 与 unified 的代码默认值今天恰好等于 JSON（探针），所以缺文件不报错、行为不变；一旦 JSON 被本地修改，clone 与本地出现不可见分叉。
- 9 个文件 sha256 与 `doc/assets/deep_chain_audit_20260929/entry_manifest.json:6-14` 全部一致，说明至少自 2026-09-29 起未变；这不是“正在编辑”的暂时状态，而是长期治理缺口。
- 建议二选一：把 4 个文件 `git add` 并按现有跟踪配置管理；或明确标为 local-only 并加 `.gitignore` + 文档改写。当前“既不入库也不忽略”是最差组合。



## 6. 缺失的 example / 校验器清单

| 期望能力 | 现状 | 证据 | 级别 |
|---|---|---|---|
| 统一 JSON Schema 或 `validate_configs` | 无 `*.schema.json`，无通用校验器；各消费者手写字段检查 | `configs/` 目录；`src/data/epochs.py:230-249`、`src/research/source_matrix.py:86-98`、`source_execution.py:206-213` | P2 |
| 无 schema 文件的字段白名单 | cloud 允许未知键；unified 由 dataclass 构造器拒绝未知键；preprocessing 的 `parse_preprocessing_spec` 会拒绝未知键但无文件级 schema | `source_execution.py:166-213`；`src/train/early_stopping.py:24-30` | P2 |
| 跨文件几何校验 | 只有 acquisition_reference/epoch_support 对代码合同；preprocessing 与 epoch_support/manifest 三方无联动 | `test_acquisition_contract.py:13-34`；本报告 X3 | P1 |
| 采集计划对真实采集端校验 | 无；rest_sec 冲突即由此漏过 | 本报告 X2 | P1 |
| candidate 示例可用 | 结构可载入，7 个输入不存在，dry-run 失败 | 本报告 X7；`run_candidate_promotion_matrix.py:2497-2507` | P2 |
| source pool 示例守门 | 可解析出 48 任务，但无测试读取该文件 | 本分区探针；`src/research/source_matrix.py:75-128` | P2 |
| 每个配置至少一个测试 | 9 个中仅 5 个被测试读取；preprocessing/unified/candidate/source-pool 4 个零测试 | `tests/**` grep `configs/`；`tests/brainsync/conftest.py:8-14` 前端缺失时 skip | P2 |
| 外部副本一致性 | acquisition_reference 有前端相等测试（缺失前端则 skip）；epoch_support 无前端副本测试 | `test_acquisition_contract.py:37-40`；外部 `models/epoch_support.json` | P2 |

## 7. 与 AGENTS.md 的 25 CPU / 4 GPU 核对

- 要求原文：`AGENTS.md:28-33`——“CPU threads = 25 and GPU workers = 4 … Start from `configs/cloud_training_resources.json`; pass the total CPU budget through to actual workers and verify effective concurrency … do not silently downgrade it to a single worker.”
- 配置数值：`cpu_budget=25`、`gpu_workers=4`、`cloud_resource_floor={cpu_threads:25, gpu_workers:4}`（`configs/cloud_training_resources.json:2-4,12`），**逐项满足**。
- 解析器验证（本分区探针，cpu_count=25/free_device=32 GiB/free_host=88 GiB）：`cpu_budget=25, source_workers=4, target_workers=4, gpu_slots=4, native_threads_per_worker=6`；4×6=24 个数值线程 + 1 个调度线程，与 `doc/engineering.zh.md:306-309` 一致。floor 不满足时显式 `ValueError`（`source_execution.py:206-213`），测试覆盖 24 CPU 与 32 GiB 两种失败（`tests/train/test_source_execution_resources.py:220-223`）。
- 未满足的是**流程而非数值**：
  1. 配置未入库，干净克隆无法“Start from”该文件（§5）；performance 分区 C6 已证明 floor 只在 1/6 入口检查。
  2. 没有 runner 直接接收该 JSON 作为资源配置；必须手工把字段并入 plan（`source_execution.py:158-227` 只接受 dict；试验/证据中的 plan 资产是内联副本）。
  3. `source_load_workers=1` 把准备阶段加载串行化，虽可能为内存控制，但未在配置/文档声明理由。
- 结论：**25/4 的数值合同成立，可审计的“从配置到实际 worker”链路不成立**；补齐入库与入口接入前，AGENTS 的“verify effective concurrency”只能靠单次运行记录，不能靠仓库复现。

## 8. 建议（按优先级）

P1（下次采集/训练前）：
1. 在 `brainsync_workstation.model.json` 增加显式 `preprocessing_config` 参数选项，或让 `prepare_brainsync_cache.py` 默认读取 `configs/brainsync_bidirectional_preprocessing.json`；并加“配置==`BRAIN_SYNC_PREPROCESSING`”测试。
2. 修正 `brainsync_collection_plan.json` 的 `rest_sec`（与采集端 20 s 对齐，或明确写 0 并在计划中声明“采集端会另加 9×20 s”）；把总时长估算 `3 + trials×SOA + (blocks-1)×rest` 写进计划/测试。
3. 增加跨文件几何测试：`load_preprocessing_spec(configs/...preprocessing.json) == BRAIN_SYNC_PREPROCESSING`，并断言 epoch_support 与 preprocessing/manifest 的 tmin/tmax/rate/n_times 一致。
4. 将 4 个未跟踪配置纳入 Git（或明确本地化并改写文档）；至少让 `cloud_training_resources.json` 在干净克隆可被测试读取。

P2：
5. 为 cloud/unified/preprocessing 增加 schema 或集中字段白名单；cloud 拒绝未知键。
6. candidate 示例：补 README 说明 7 个前置产物，或改为可生成的最小 fixture；明确 `--root` 与 `../` 路径的互斥规则。
7. manifest 测试扩展到 version/layout_schema/entrypoint/tasks/parameter_options/epoch_cache/attestation；定义 version 4.1.0 与 session_schema /4 的映射。
8. unified：要么让 `--training-budget` 默认加载该文件，要么把文档“默认文件”改为“推荐显式传入”。
9. source pool 示例加入参数化测试（`source_matrix_tasks(example)` 非空、任务 id 唯一、池名冻结）。
10. 命名统一：`sfreq` vs `model_sample_rate_hz`；schema 连字符/下划线；在 README 列出每个配置的消费者、版本与单位。

## 9. 覆盖边界与复现

- 已做：9 个 JSON 全文；消费者 grep（src/experiments/tests/doc）；外部 BrainSync 采集端 `models.py/ui.py/model_manifest.example.json` 只读对照；`git ls-files/check-ignore/log`；sha256 对照 2026-09-29 审计清单；`tmp/audit/scratch/g5_probe.py`（配置解析/资源/池/算术）、`g5_candidate.py`（load_plan）、`g5_candidate_dry.py`（dry-run 失败）。
- 测试：5 个直接读取配置的测试全部通过（epoch_support、acquisition_reference、workstation manifest、collection plan、cloud floor），命令见报告生成记录；未跑全量 pytest（tests 分区负责）。
- 未做/不覆盖：data 核心的实际 epoch 支撑数学（data 分区）、transfer 预算语义（transfer 分区）、floor 在 6 个入口的覆盖（performance 分区 C6）、文档死链（docs 分区）、密钥（security 分区）、Lean（Lean 分区）。外部采集端仅作为消费者读取，未修改、未运行其训练。
- 行号均为当前工作树（HEAD f1a303c + 未跟踪文件）状态；未跟踪文件入库后行号可能变化。

