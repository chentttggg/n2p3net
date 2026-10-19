# G3b：BrainSync 后端入口与配置（二次审计）

- 日期：2026-10-19（Asia/Shanghai）
- 对象：`D:\BCI\n2p3-net` 当前工作树；HEAD=`f1a303c`（工作树另有未提交修改）
- 范围：`experiments/{brainsync_model_task,prepare_brainsync_cache,fit_brainsync_eog,fit_brainsync_continuous_qc,report_brainsync_quality,run_brainsync_cross_decision,correct_brainsync_reference}.py`、`configs/brainsync_*.json`、`doc/workstation.zh.md`
- 只读声明：未修改/删除/重命名仓库既有文件；未运行 git 写命令、安装、训练或子代理；未执行真实 `calibrate/infer/prepare/fit/correct`，只运行 7 个入口 `--help`（均退出 0）。本报告与 `tmp/audit/scratch/` 是唯一允许写入。
- 方法：逐行源码阅读、`grep/glob`、只读 `git diff/status`、库函数交叉核对；每个结论给出 文件:行号。**本报告不重复 G3 已覆盖的 20 条**（见 `tmp/audit/G3-brainsync-workstation.md`），只列二次核查新增/修正项，并在第 2 节交叉复核 G3 关键结论。
## 0. 结论摘要

- 新增分级：**P0 = 0；P1 = 1；P2 = 7**（均相对 G3 已报内容而言）。没有发现会直接删除原始 EDF、会话或既有模型包的默认路径。
- 最重要的三条：G3b-01（参考元数据更正后无后置校验，可产出内部不一致且仍 exit 0 的 BIDS 会话）；G3b-02（更正脚本的 EDF 后置核对在回滚事务之外）；G3b-04（后端不强制 45 闪烁/5 重复测试协议）。
- 正向结论不变：`calibrate` 的模型包采用硬链接独占发布，`infer` 只写 `summary.json`，不写模型/缓存/会话；`infer` 的参与者、校准记录复用、时间、模拟声明、采集参考和几何门禁均在原始 QC 之前执行。
- G3 的 3 条 P1 与 20 条 P2 本次未逐条重跑，但抽查 11 条控制流与当前工作树一致（第 2 节）；G3 未覆盖的缺口集中在“更正后置条件”“协议级门禁”“缓存并发原子性”“既有入口的新参数未接线”。
- 说明：本报告行号均为当前工作树；`experiments/brainsync_model_task.py` 等 5 个文件相对 HEAD 有未提交修改，结论以磁盘工作树为准。

## 1. 新增发现

### G3b-01（P1，已证实）参考更正脚本没有后置合同校验；`channels.tsv` 的 `reference` 可被静默留在旧值

- 证据：`experiments/correct_brainsync_reference.py:34-90` 只构造变更；`:68-75` 对 `eeg.json` 严格要求 `EEGReference == previous` 后才改写，但 `:77-88` 对 `channels.tsv` 只在存在 `reference` 列时改写“值恰好等于 previous”的行；没有检查是否每个 EEG 行都等于 previous，也没有在没有任何行匹配时报错。
- 证据：`:89` 再次写入 session JSON；`:114-123` 逐文件比较后写入；`:124-125` 只核对 EDF 哈希；`:126-135` 写 `manifest.json`；`:191-194` 的 `main()` 打印成功计数并返回 0。全文没有调用 `validate_brainsync_bids_session` 做后置验证（该导入只出现在未接 CLI 的 `:141`）。
- 证据（后果）：`src/data/brainsync_contract.py:169-172` 在下次加载时要求每个 EEG 行的 `reference`（缺省时取 `bids.source_reference`）都等于 `bids.source_reference`。若 `channels.tsv` 的 `reference` 是第三个值或混合值，更正脚本会把 session/eeg.json 改成新标签、把 `channels.tsv` 留在旧值，仍然 exit 0；下一次质量报告/缓存/模型构建才以 “BrainSync EEG channels must share the declared acquisition reference” 失败。
- 反证边界：测试夹具 `tests/support/brainsync.py:204-210` 的 `channels.tsv` 根本没有 `reference` 列，因此 `tests/brainsync/test_brainsync_reference.py:23-32` 只断言“2 个文件被改”，恰好绕过了“列存在但值不匹配”的分支；当前测试无法证明该场景安全。
- 触发条件：真实采集端的 `channels.tsv` 写出可选 `reference` 列，且其中任一 EEG 行不是 `previous`（例如人工改过、旧导出器写 `T7`、带空白或大小写差异）。`previous` 完全匹配的行会被正确改写，因此“部分行匹配”是最危险的混合状态。
- 影响：一个声明为“仅改元数据、不改样本”的更正操作，可以产出 session/eeg.json 已更正、channels.tsv 未同步、且带成功 manifest 的数据集；后续失败与更正动作之间没有直接错误链接，审计者也容易只读 manifest 判定成功。
- 建议：在任何写入前要求“列缺失，或所有 EEG 行的 `reference` 都等于 previous”二者之一；写入后、发布 manifest 前，对每个 root 运行 `validate_brainsync_bids_session`；失败则按 G3b-02 回滚。该结论不需要真实数据即可从控制流证实。

### G3b-02（P2，已证实）EDF 后置核对在回滚事务之外；失败会留下“已改元数据但无 manifest”的半成品

- 证据：`experiments/correct_brainsync_reference.py:106-112` 先写 zip 并逐成员回读校验；`:114-123` 的 `try/except Exception` 包住元数据写入并可在异常时回滚；但 `:124-125` 的 EDF 哈希核对在 `except` 之外。若该核对失败，session/BIDS 元数据已经被替换，函数在写 `:126-135` 的 manifest 之前抛错，归档目录里只有 zip，没有 manifest。
- 影响：并发改动 EDF 时（脚本自身不改 EDF），会留下“元数据已改、原始字节已变、无更正台账”的状态；重跑会因 `montage.ref_label != previous` 被 `:42-43` 拒绝（测试 `tests/brainsync/test_brainsync_reference.py:37-38` 固化了该拒绝），只能人工从 zip 恢复。触发概率低，但该脚本是明确的重写型工具，原子性应按最高标准要求。
- 建议：把 EDF 前后哈希核对移入 `try`，或将“写文件 + 核对 + 写 manifest”实现为可恢复事务（先写 `.new`，全部验证后 rename，最后原子写 manifest）；捕获 `BaseException` 以覆盖 Ctrl-C（G3-08 已报后者，本条只补 EDF 核对的位置）。

### G3b-03（P2；遗漏已证实，数值影响疑似）EOG 拟合的 `preprocessing` 参数没有接到两个 CLI 入口

- 证据：`experiments/fit_brainsync_eog.py:37-44` 新增了 `preprocessing` 参数，`:52` 只在 `load_brainsync_session` 的身份计算中使用；但 CLI `:146-163` 完全没有 `--preprocessing-config`，调用时也不传 `preprocessing`。`experiments/brainsync_model_task.py:318-319` 已加载完整 `preprocessing`，`:367-371` 调用 `fit_brainsync_eog_sessions(...)` 时仍未传 `preprocessing=preprocessing`。
- 影响（为何是疑似而非已证实的数值缺陷）：EOG 回归系数本身在原始样本域拟合（`src/data/ocular.py:229-266` 对 `read_raw` 的 EEG/EOG 直接回归），不消费 bandpass/notch 后的 `dataset.X`；当前缺失主要使 EOG 拟合内部的 `load_brainsync_session` 使用默认 `BRAIN_SYNC_PREPROCESSING`（`src/data/brainsync.py:153-161`），身份键通常仍相同。但 EOG 模型保存的记录没有 `preprocessing`/其 digest，`--preprocessing-config` 声明的合同没有在该拟合路径被执行；若未来预处理改变 `source_reference`/通道几何/身份键，拟合与校准会静默使用两套输入声明。
- 建议：给 `fit_brainsync_eog.py` 增加 `--preprocessing-config`，并把 `brainsync_model_task.py` 的 `preprocessing` 传入；在 `fit_provenance` 中记录 `preprocessing.record()` 与 digest，或在 `fit_eog_regression` 前调用 `validate_recovery_contract`/等价门禁。

### G3b-04（P2，已证实）后端不强制执行 45 闪烁/5 重复测试协议；9 闪烁的合法声明可产生 `predicted`

- 证据：`experiments/brainsync_model_task.py:120-150` 在未给 `--repetitions` 时用 `repeat.max()+1` 推 level；`:154-193` 对每个 level 只要每个候选计数达到 level 就输出 `status="predicted"`；全文没有 `scheduled_trials == 45`、`requested == 5` 或“协议合规”字段。`:235` 的 `--repetitions` 只在 `:295-296` 检查为正。
- 证据（输入合同允许）：`src/data/brainsync_contract.py:250-271` 只要求 `digits_per_block` 是 9 的倍数且 `repetitions_per_digit == digits_per_block/9`；`:274-300` 只校验每个 block 的平衡与置换。因此 `blocks=1, digits_per_block=9, repetitions_per_digit=1` 的 9 闪烁会话是合法输入，会得到 `requested_repetitions=1` 且可能 `status="predicted"`。
- 协议对照：`configs/brainsync_collection_plan.json:18-27` 定义测试为 1 block × 45、`primary_repetitions=5`、`diagnostic_repetitions=[1,2,3,4]`；`doc/workstation.zh.md:124` 写“新测试只采一个45闪烁block”。外部 GUI 会过滤 45 闪烁，但 CLI/直接调用后端没有同等门禁。
- 影响：一条被截断但自身平衡的 9/18 闪烁录制可以生成看似测试结果的 `summary.json`（含 `predicted`、`requested_repetitions=1/2`）；它不会写模型、缓存或会话，且 summary 保留计数，因此不是“伪造证据”，但缺少机器可读的 `protocol_compliant` 会让下游把诊断重复当成产品 R5 结果。建议在后端写 `protocol_compliant`，运营状态要求 45/5；1–4 重复仅允许显式诊断。



### G3b-05（P2，已证实）跨决策评估摘要不记录缓存的 ingress/skipped 会话，skip 模式掉队不可从结果本身审计

- 证据：`experiments/prepare_brainsync_cache.py:43-47` 暴露 `--invalid-session {error,skip}`；`:68-74` 交给 `load_brainsync_sessions_resilient`。失败时 `src/data/brainsync.py:846-854` 把 `ingress_policy` 与 `skipped_sessions` 写入缓存 provenance，exit 仍为 0（`:81-88` 只打印 skipped 行）。
- 证据：`experiments/run_brainsync_cross_decision.py:223` 用 attested 缓存加载；结果 summary `:648-704` 记录 `target_cache_sha256`、`requested_subjects`、`usable_subjects`、`excluded_subjects`，但没有复制 `dataset.provenance["ingress_policy"]` / `skipped_sessions`，也没有 `requested vs accepted` 的缓存级计数。
- 影响：若评估使用 skip 模式缓存，结果 JSON 本身不说明哪些请求会话在进入评估前已被丢弃；只有缓存文件（若仍保留且 SHA 可找到）才有该信息。不是数值错误，是结果 lineage 缺口，和本仓库“失败/缺失保留分母”的审计原则不一致。建议把两个 provenance 字段复制进 `evaluation_contract` 或 summary 顶层，并在 `--dataset-cache` 已 skip 时打印一行计数。

### G3b-06（P2，已证实）缓存保存的临时文件名固定；两个同输出并发任务可互踩临时文件，崩溃点在 npz/record 之间

- 证据：`src/data/epochs.py:1036-1040`：`temporary = path.with_suffix(path.suffix + ".tmp.npz")` 是按目标路径固定的；`writer(temporary, **payload); temporary.replace(path); write_epoch_dataset_record(path, dataset)`。`experiments/prepare_brainsync_cache.py:76-80` 直接把用户/环境给的 `--output` 交给它，没有锁。
- 影响：G3-05 已覆盖“覆盖既有缓存”；本条补并发/成对原子性：两个进程写同一输出时共享同一个 `.tmp.npz`，一个进程的 `replace` 可能发生在另一个仍在写临时文件时，产出截断/混合缓存；进程在 `replace` 与 `write_epoch_dataset_record` 之间崩溃，会留下“新 NPZ + 旧/缺 record”。attested 加载会 fail-closed，但该缓存不可用且旧缓存已被替换。
- 建议：临时文件加入 PID/UUID 并在 replace 前 fsync；把 `(npz, record.json)` 作为一个内容寻址对象发布，先写 record 临时文件，最后一次性提交两者，或显式加输出锁。`fit_brainsync_continuous_qc.py`/`fit_brainsync_eog.py` 的单 JSON 保存没有此问题。

### G3b-07（P2，已证实）连续 QC 校准 manifest 的 `preprocessing` 可省略并回退到代码默认，provenance 不记录实际生效配置

- 证据：`experiments/fit_brainsync_continuous_qc.py:36-38`：只有 `"preprocessing" in manifest` 才构造 `PreprocessingSpec`，否则使用 `BRAIN_SYNC_PREPROCESSING`（`src/data/brainsync.py:48-50`）。随后 `preprocessing.validate()` 不做“完整声明”检查；`fit_provenance`（`:88-93`）只记录 `manifest_sha256`，没有记录实际生效的 `preprocessing.record()`。
- 影响：同一个 manifest 文件（同一 `manifest_sha256`）在代码默认值改变后可产生不同的 QC 拟合，而模型记录无法单独证明当时用的预处理；对于非零相位/无 `epoch_recovery` 的因果路径，这个绑定更弱（零相位路径由 `epoch_recovery.preprocessing` 间接绑定，见 `src/data/continuous_qc.py:98-119`）。这是可复现性/审计缺口，不是当前数值错误。
- 建议：要求 manifest 显式给出完整 `preprocessing`（或记录其 canonical digest），用 `parse_preprocessing_spec` 而非 dataclass 构造；把 `preprocessing.record()` 纳入 fit provenance。与 G3-16（局部对象被 dataclass 默认补齐）相邻但不同：本条是“键整体缺失时静默回退代码默认”。

### G3b-08（P2，已证实）参考更正的 `--output` 不要求新目录；只保护 zip 名，`manifest.json` 可被覆盖，且可写进会话根目录

- 证据：`experiments/correct_brainsync_reference.py:100` `output.mkdir(parents=True, exist_ok=True)`；`:101-103` 只检查 `original_reference_metadata.zip` 不存在；`:135` `atomic_write_json(output / "manifest.json", manifest)` 会替换已有同名文件。全文没有 `output` 与 `roots` 的包含/相等检查（对比 `:61-62` 对 BIDS sidecar 有 `is_relative_to(root)`）。
- 影响：若操作者把 `--output` 指向一个已有审计目录（例如手工删除/移走旧 zip 后的重跑，或误指到另一个任务的目录），旧 `manifest.json` 会被静默替换；若指向某个会话根，zip/manifest 会被写进 BIDS raw 目录，污染原始数据集（虽然不一定被 BIDS 校验拒绝）。该脚本是显式写元数据的工具，输出目录应像模型包一样要求“新路径”。
- 建议：要求 `output` 不存在或为空，拒绝 `output == root` 或 `output.is_relative_to(root)`；若允许复用，先以独占方式发布 manifest（`x`/硬链接），并在 manifest 中记录被替换的旧文件摘要。

## 2. 对 G3 既有结论的复核（不重复细节）

| G3 ID | 复核结果 | 当前工作树证据/说明 |
|---|---|---|
| G3-01 fixed_erp 不可达 | 一致 | `experiments/brainsync_model_task.py:250-269` 只在 unified+calibrate 放行 `--calibration-recipe`；`SELECTOR_MODES` 仍含 fixed_erp；`doc/workstation.zh.md:37-39` 声明可保存重载。 |
| G3-02 文档依赖未跟踪配置 | 一致 | 当前 `git status` 显示 `?? configs/brainsync_bidirectional_preprocessing.json`；`doc/workstation.zh.md:4-6,28,43` 引用它。 |
| G3-03 cross-decision 覆盖输出 | 一致 | `experiments/run_brainsync_cross_decision.py:705-706` 直接 `atomic_write_json`；无 `output.exists()`/输入路径碰撞检查。 |
| G3-04 artifact 等于 summary 路径 | 一致 | `brainsync_model_task.py:270-273` 只查 summary；`:321-322` 只查 artifact；`:400` 保存后 `:415-417` 用 summary 覆盖同路径。 |
| G3-05 缓存静默覆盖 | 一致 | `prepare_brainsync_cache.py:76-80` + `src/data/epochs.py:1037-1040`；本条 G3b-06 补并发原子性。 |
| G3-06 calibrate 部分失败占目录 | 一致 | `brainsync_model_task.py:400-417` 先保存模型后写 summary；`:438-442` 失败也写 summary 并 exit 1。 |
| G3-09 短录制拒判 IndexError | 一致 | `brainsync_model_task.py:425` 用 `scheduled_trials // 9`；1–8 闪烁得 0，`:433` 的 `rows[-1]` 触发 IndexError。 |
| G3-11 质量报告错误 exit 0 | 一致 | `report_brainsync_quality.py:28-73` 无 `sys.exit`；`src/research/recording_quality.py:317-326` 捕获异常写 `not_evaluable` 后正常返回。 |
| G3-15 `ocular_fit_policy` 标签不实 | 一致 | `brainsync_model_task.py:102-106` 只看 `source_pretraining_used` 和非 unified；实际拟合范围在 `:361-371`，`five_candidates` 无源时也裁掉 20% 验证会话。 |
| G3-16/G3-17 连续 QC manifest 校验弱 | 一致 | `fit_brainsync_continuous_qc.py:31-45` 仅顶层 schema，`:36-38` 用 dataclass 构造；`:97-102` 直接 `**manifest["epoch_recovery"]`。 |
| G3-21 每 session 两类只查 legacy | 一致 | `src/transfer/calibrated.py:1060-1063` 仅 legacy 循环；unified 分支 `:1086-1107` 无逐 session 检查。 |
| G3-22 单类 test AUC 中止整轮 | 一致 | `run_brainsync_cross_decision.py:482` 直接 `roc_auc_score`，异常不在逐 subject try 内，summary 只在末尾写。 |
| G3-14 /1、/3 旧包兼容 | 未重跑，未发现反证 | `load_calibrated_predictor` 接受 set；本地无真实 /1 或 /3 包，沿用 G3 的“代码级有条件成立”结论。 |

- 本次没有发现 G3 的误报；新增的 G3b-01 是 G3 未覆盖的“更正后置条件”，G3b-06 是 G3-05 未覆盖的并发/成对原子性，G3b-03/04/05/07/08 是 G3 未列的入口参数/协议/lineage/输出目录缺口。



## 3. 已验证的正向合同（按审计焦点逐项）

### 3.1 CLI 默认值、覆盖与退出码

- 7 个入口 `--help` 均 exit 0（本次实测）；缺必填/非法组合由 argparse exit 2；`brainsync_model_task` 的运行时异常写 failed summary 后 `SystemExit(1)`（`:438-442`），`prepare`/`fit`/`correct` 多为裸 traceback exit 1，`report` 恒 0（G3-11）。没有“默认删除”路径：`prepare` 只替换输出缓存（G3-05），`correct` 是唯一就地改写，且先归档（`:106-112`）。
- `calibrate` 的输出目录与模型包都有独占门：summary 必须不存在（`:270-273`），artifact 必须不存在（`:321-322`），模型用临时文件+硬链接发布（`src/transfer/calibrated.py:1198-1217`）。G3-04/G3-06 的路径碰撞与半成品仍成立，但“默认覆盖既有模型”不成立。
- `fit_brainsync_eog` 在任何拟合前拒绝已存在输出（`:156-157`）；`fit_brainsync_continuous_qc` 只在保存时检查（`src/data/continuous_qc.py:627-632`），会先完成整轮拟合，属可改进但不破坏数据的默认。
- 路径方面，7 个脚本均以 `__file__` 推导 ROOT/SRC 并插入 `sys.path`；`grep` 未发现 `D:/`、`C:` 等机器绝对路径。`configs/brainsync_workstation.model.json:5-6` 使用 `../.venv/Scripts/python.exe` 与 `../experiments/...` 相对路径（Windows 专用，G3-18/G5 已报）；`doc/workstation.zh.md` 的外部 `D:/BCI/BrainSync...` 是采集端交付路径，不属于后端运行路径。

### 3.2 无标签推理是否可能写入“证据”

- 不写模型/缓存/会话：`infer` 分支只 `predictor.load_inference_session` + `inference_record` + `atomic_write_json(summary_path)`（`brainsync_model_task.py:403-417`）；全文没有 `save_calibrated_predictor`/`save_epoch_quality`/BIDS 写调用。模型字节只读一次并哈希，再从同一 `BytesIO` 加载（`:304-307`），避免“哈希后换文件”的 TOCTOU；summary 的 `artifact_sha256` 对应实际加载字节。
- 目标隔离：`load_brainsync_inference_session` 走 `validate_brainsync_bids_recording`（`src/data/brainsync.py:645-667`、`:311-314`），`labelled` 由 recording 类型决定，inference 的 `labels_array/targets` 为 `None`（`:187-230`）；`inference_record` 只读 `candidate_ids/evidence_indices/repetition_indices`，并写 `target_labels_used: False`（`:198-210`）。质量报告另有 `if dataset.y is not None: raise`（`src/research/recording_quality.py:262-263`）。
- 前置门禁顺序正确：`load_inference_session` 的 `recording_validator` 在原始 QC 之前执行（`src/data/brainsync.py:657-660`），核对参与者、session id 非校准复用、开始时间晚于校准可用时间、模拟声明、采集参考 details、通道几何（`calibrated.py:715-750`）；随后 `validate_input` 核对输入合同/执行凭证（`:752-766`），`validate_inference_scope` 再做一次会话范围（`:815-824`）。因此“连续 QC 完全拒绝”的 `BrainSyncNoUsableEvidenceError` 只会发生在上述门禁之后（这也解释了工作树删除重复异常分支后仍安全）。
- 剩余风险：校准会话身份只绑定 `session_id + started_timestamp_s`（`calibrated.py:819-824`），没有原始 EDF 字节哈希；复制并改写 session 元数据可伪装成独立记录。该缺口属于 G1/P1-D3 的“BIDS 无内容哈希”，本分区不重复，只确认推理入口没有额外的字节绑定。

### 3.3 模型保存/重载身份与兼容

- 当前 `CALIBRATED_PREDICTOR_SCHEMA = n2p3_calibrated_predictor/4`（`calibrated.py:92`）；加载器接受 `/4、/3、/1`（`:1220-1236`），验证 `metadata_digest`、`state_digest`，并用 `torch.load(..., weights_only=True)`（`:1222`）阻止任意代码；构造器再核对架构、输入统计、ERP/temporal basis、LDA 特征形状与非有限值（`:569-663`）。这支持“模型包身份可核验”。
- `calibrate` 保存后没有重载验证（G3-13）；只在 `:416` 读回字节算 SHA，若序列化/元数据在本机组合下不可读，要到下一次 `infer` 才失败。正向部分：模型元数据记录 participant、calibration sessions、input contract/request、preprocessing execution binding、模型参数摘要；inference 以同一份 metadata 重建输入。
- 当前 G3-14 的旧包兼容仍成立：加载器接受旧 schema，但 `_model_summary`/`load_inference_session` 无条件读 `input_request`/`selection` 等新字段；本地没有真实 /1、/3 包，未做端到端重载。

### 3.4 配置 schema 校验与 `--preprocessing-config` 接线

- 严格入口：`parse_preprocessing_spec` 要求完整字段集合、拒绝未知字段、要求 notch 两字段成对并执行 `validate()`（`src/data/epochs.py:230-254`）。三个入口已接线：`prepare_brainsync_cache.py:50-51,71`、`brainsync_model_task.py:57-60,297-298,319`、`report_brainsync_quality.py:36-37,53`；显式配置同时要求完整，不是“局部覆盖”。
- 未接线/弱校验：连续 QC manifest 用 dataclass 构造并仅 `validate()`（G3-16），且 `preprocessing` 键可整体省略回退代码默认（G3b-07）；`brainsync_workstation.model.json:13` 的唯一 parameter option 不传 `--preprocessing-config`（G5-X1），因此 GUI 的预处理配置不会改变实际缓存合同——这是已报结论，本分区只从入口侧确认。
- 配置值方面：`brainsync_acquisition_reference.json` 与 `brainsync_reference_record()` 有相等测试（`tests/brainsync/test_acquisition_contract.py:24-34`）；`brainsync_epoch_support.json` 与 `epoch_support_record()` 有相等测试（`:13-21`）；`brainsync_workstation.model.json` 仅 5 个物理字段被测试（G3-18）；`brainsync_bidirectional_preprocessing.json` 未跟踪且无测试（G3-02）。`brainsync_collection_plan.json` 当前未被后端代码消费，仅文档/测试使用（G5 已覆盖其 rest_sec 与语义问题，本分区不重复）。

### 3.5 校准→保存→未知目标推理的顺序/门禁与文档一致性

- 顺序：`calibrate` 先检查 output/artifact，再加载会话、拟合/应用参考、拟合模型、保存模型、写 summary（`:321-417`）；模型在 summary 之前发布，故 summary 失败会留下可用但无成功回执的模型（G3-06）。`infer` 先加载并哈希模型，再用同一字节重建输入，门禁在原始 QC 前，最后只写 summary（`:304-307,403-417`）。没有“先推理未知目标、后保存模型”的路径。
- 文档对照（`doc/workstation.zh.md`）：`:4-6` unified 命令与 CLI 相符（recipe 必填、budget 可选）；`:27-28` “完整预处理配置”对三个入口成立、对连续 QC manifest 不成立（G3-16）；`:37-39` fixed_erp 可保存重载不成立（G3-01）；`:77` “参数损坏明确失败”对 report 过强（G3-11）；`:100-103` 模型包重放/拒绝异人/时间/模拟基本成立（短录制例外 G3-09）；`:124` 45 闪烁协议后端未强制（G3b-04）；`:167-168` REF 左耳垂合同与 `src/data/contract.py` 常量一致；`:199-201` “epoch QC 必须配 /4 连续 profile”由 `validate_personal_epoch_raw_quality` 执行（`src/transfer/quality.py:40-62`），但 report 会把它降级为 `not_evaluable`（G3-11）；`:243` “每个校准 session 各自两类”仅 legacy 检查（G3-21）。

### 3.6 输入/输出原子性小结

| 写入物 | 现状 | 结论 |
|---|---|---|
| 模型包 | 临时文件 + `os.link` 独占（`calibrated.py:1198-1217`） | 通过（除 G3-04 路径碰撞） |
| summary/epoch_quality | `atomic_write_json`；epoch_quality 硬链接独占（`transfer/quality.py:201-216`）；summary 独占（`:270-273`） | 通过（calibrate 先模型后 summary，G3-06） |
| 缓存 npz+record | 固定临时名 `replace` 后写 record（`src/data/epochs.py:1036-1040`） | 覆盖门缺失（G3-05）；并发/成对原子性缺失（G3b-06） |
| 质量报告 | `report.json` 独占+原子；`report.html`/PNG 直接写（`recording_quality.py:207-231,324-326`） | report.json 通过；同名图片可覆盖（G3-12） |
| 参考更正 | zip 独占+逐成员校验+逐文件回滚；EDF 后置核对在回滚外（`correct...:106-135`） | 基本通过；G3b-01/G3b-02/G3b-08 补后置与输出目录 |
| 跨决策结果 | `atomic_write_json`，无独占/输入碰撞检查（`run...:705-706`） | 原子但不防覆盖（G3-03） |

## 4. 覆盖边界与限制

- 未执行任何真实 `calibrate/infer/prepare/fit/correct` 路径；未运行 pytest（由 06-tests 覆盖）；只运行 7 个 `--help` 与只读 `git`/源码/库函数阅读。所有“已证实”指控制流/合同可静态证实，不包含真实数据上的数值复现。
- 未读取任何密钥/原始数据；`fit_brainsync_continuous_qc` 的 manifest 样例、`/1`/`/3` 历史模型包、外部 BrainSync 采集端不在本次可达范围；G3-14 沿用 G3 的未重跑结论。
- 本报告与 `tmp/audit/scratch/g3b_chunk2.md`、`g3b_chunk3.md`（临时片段）是唯一写入；未修改任何既有文件。报告行号以 2026-10-19 当前工作树为准；若未跟踪配置入库或工作树继续变化，需重核 G3-02/`brainsync_workstation.model.json` 行号。
- 结论：本分区未发现 P0；P1 仅 G3b-01，建议在任何“更正 REF 后重建质量/缓存/模型”的流程前先修；P2 中 G3b-04（协议门禁）和 G3b-06（并发缓存）优先级最高，其余为可复现性/lineage 改进。

