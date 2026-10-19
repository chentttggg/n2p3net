# G3 BrainSync 工作站后端入口与配置审计

状态：完成（只读审计；未修改仓库文件、未运行真实训练/采集 CLI、未安装依赖、未创建子代理）
范围：`experiments/{brainsync_model_task,prepare_brainsync_cache,fit_brainsync_eog,fit_brainsync_continuous_qc,report_brainsync_quality,run_brainsync_cross_decision,correct_brainsync_reference,run_brainsync_pilot}.py`、`configs/brainsync_*.json`、`doc/workstation.zh.md`。
方法：逐行阅读、`grep/glob`、只读 `git`、各入口 `--help` 与参数拒绝探测、只读 `python -c` 配置探测、`ruff check`。未执行 `calibrate/infer` 的真实数据路径。

## 0. 基线与工作树状态

- HEAD = `f1a303c`。与 G3 相关的未提交状态：`experiments/brainsync_model_task.py`、`experiments/fit_brainsync_eog.py`、`experiments/fit_brainsync_continuous_qc.py`、`experiments/report_brainsync_quality.py`、`doc/workstation.zh.md` 为 M；`configs/brainsync_bidirectional_preprocessing.json` 与 `configs/unified_training_budget.json` 为 `??`（未跟踪）。其余四个 `configs/brainsync_*.json` 已跟踪。
- 环境：`.venv/Scripts/python.exe` = Python 3.14.3；`pyproject.toml:9` 要求 `>=3.11`，不冲突。
- `ruff check` 对上述 8 个脚本全部通过（无 lint 发现）。
- 8 个入口 `--help` 均退出 0；缺参数/非法组合退出 2；运行时异常退出 1（部分为裸 traceback，见 G3-09/G3-20）。

## 1. 结论摘要

- 级别计数：P0 = 0；P1 = 3；P2 = 20。
- 无 P0。模型/缓存/报告写入路径普遍有原子写与独占保护；无标签推理路径确实不写模型、缓存或会话元数据。主要风险集中在“文档声明与 CLI 可达能力不一致”、跨入口配置校验不一致、以及少数路径碰撞/覆盖门禁缺失。最重要的 5 条见第 11 节。

## 2. P1 发现

### G3-01 fixed_erp 在 `brainsync_model_task` 中不可达，与文档声明冲突

- 证据：`experiments/brainsync_model_task.py:250-269` 仅在 `--selector-mode unified` 时允许 `--calibration-recipe`，否则 `parser.error("calibration-recipe belongs to unified calibration")`；但 `--selector-mode` 的 choices 来自 `SELECTOR_MODES`（`transfer/calibration_candidates.py:42`），包含 `fixed_erp`。`calibration_candidate_recipes("fixed_erp")` 无 recipe 时抛 `ValueError("fixed_erp requires one explicit ERP TargetRecipe")`（`transfer/calibration_candidates.py:51-54`），`fit_calibrated_predictor` 的 fixed_erp 分支同样要求 recipe（`transfer/calibrated.py:1071-1076`）。
- 复现：`brainsync_model_task.py --task calibrate --selector-mode fixed_erp ... --calibration-recipe <json>` 退出 2 并报“calibration-recipe belongs to unified calibration”；不传 recipe 则 parser 放行，随后在预检/拟合阶段以 ValueError 失败（退出 1，输出 failed summary）。
- 文档冲突：`doc/workstation.zh.md:37-39` 声明“ERP/Toeplitz/时间基研究配方可经后端 `fixed_erp` 个人包保存重载并预测未知标签”。实测唯一入口无法提供该模式必需的 recipe；tests 只直接调用库函数（`tests/brainsync/test_calibrated_predictor.py:48,81,90`）。
- 建议：若保留 CLI 能力，应允许 `--calibration-recipe` 用于 `fixed_erp`（并拒绝与 `--training-budget` 组合），或在 `--selector-mode` choices 中移除 fixed_erp 并改文档为“仅库接口”。级别 P1（功能与文档均不可依赖）。

### G3-02 文档化的 unified 入口依赖未跟踪配置，HEAD 克隆不可复现

- 证据：`doc/workstation.zh.md:4-6` 指定 `--training-budget configs/unified_training_budget.json`；`doc/workstation.zh.md:28,43` 指定 `configs/brainsync_bidirectional_preprocessing.json` 为现行配置模板。两者均未跟踪（`git ls-files` 仅列出另外四个 configs）。`configs/brainsync_bidirectional_preprocessing.json` 的 SHA256 与多个 frozen manifest 记录的 `deffab5f...` 一致，说明冻结证据依赖该未跟踪文件。
- 影响：从 `f1a303c` 干净检出无法执行文档命令；若工作树配置被误改/丢失，冻结证据的配置身份无法从 Git 恢复。`tests/brainsync` 对 workstation/collection/epoch/reference 四个配置有等式测试（如 `tests/brainsync/test_brainsync.py:342-379`、`tests/brainsync/test_acquisition_contract.py:13-34`），但对 bidirectional 配置无任何测试钉住。
- 补充：探测显示该配置当前与运行时默认 `BRAIN_SYNC_PREPROCESSING` 完全相等（`load_preprocessing_spec` 相等为 True，名称 `brainsync_bidirectional_v1`、notch `(50.0,)`）；unified budget 文件也与 `UnifiedTrainingBudget()` 默认值完全相等。因此当前“文档值=默认值”，风险不是数值错误而是可复现性与身份。级别 P1（与 09-repo 的未跟踪清单相关，此处只报工作站入口后果）。

### G3-03 `run_brainsync_cross_decision` 的结果输出无存在性/路径身份门禁

- 证据：`experiments/run_brainsync_cross_decision.py:705-706` 直接 `atomic_write_json(output, summary)`；全文没有 `output.exists()` 或“新目录”检查（`grep` 仅 706 一处写）。`research/execution.py:42-63` 的 `atomic_write_json` 用 `replace` 覆盖既有文件。
- 影响 1：重复运行会静默覆盖上一次完整评估结果（含 evaluation_contract、逐 epoch logits/labels），破坏结果溯源；与 `brainsync_model_task`（summary/artifact 独占）、`run_brainsync_pilot`（`mkdir(exist_ok=False)`）、quality report（report.json 独占）的策略不一致。
- 影响 2：`--output` 可与 `--dataset-cache`、`--checkpoint`、snapshot manifest 等输入路径重合；输入在开头读取、输出在结尾写入，一次成功运行会把输入缓存/checkpoint 覆盖成 JSON 并退出 0。
- 建议：要求 `--output` 为新文件（`open("x")` 原子发布），并断言其 resolved 路径不同于所有 `--*cache/checkpoint/manifest` 输入。级别 P1（静默丢失结果；碰撞时丢失输入）。

## 3. P2：覆盖/写入与模型包安全

### G3-04 calibrate 允许 `--model-artifact` 等于 `<output-dir>/summary.json`，模型会被回执覆盖

- 证据：`experiments/brainsync_model_task.py:270-273` 只检查 `output_dir/summary.json` 是否已存在（此时不存在）；`:321-322` 只要求 artifact 不存在；`:400` 保存模型；`:415-417` 再向同一 `summary_path` 原子写回执。若用户把 artifact 指到该名字，保存后的模型是合法文件，随后被 JSON 覆盖，进程仍以 0 退出，且 `artifact_sha256` 记录的是被覆盖前的模型字节。触发面窄（仅 calibrate，且需精确同名），但结果是把唯一模型包静默变成回执；infer/inspect 因 `:272` 先命中已存在文件而拒绝。
- 建议：启动时断言 `args.model_artifact.resolve()` 不在 `{output_dir/summary.json, output_dir/epoch_quality.json}`，并拒绝 artifact 位于 output_dir 内（或为重定向输出另建 receipt 子目录）。级别 P2（触发即 P1 影响）。

### G3-05 `prepare_brainsync_cache` 静默覆盖既有缓存与 attestation，无 `--force`/新路径约束

- 证据：CLI 只把 `--output` 交给 `save_epoch_dataset`（`experiments/prepare_brainsync_cache.py:76-80`）；后者不检查存在性，直接 `temporary.replace(path)` 后重写 `.record.json`（`src/data/epochs.py:1037-1040`）。默认输出来自 `BRAIN_SYNC_OUTPUT_DIR`（`prepare_brainsync_cache.py:61-65`），重复调用会替换上次的 attested cache。
- 影响：旧缓存/其数据身份与预处理凭证被覆盖，未归档则无法恢复；两个并发预处理可能让读者看到新 npz+旧 record（load 会 fail-closed，但可用性受损）。
- 对比：模型 `save_calibrated_predictor`（`transfer/calibrated.py:1201-1202`）、EOG/连续 QC（`data/ocular.py:269-274`、`data/continuous_qc.py:627-632`）、epoch QC（`transfer/quality.py:201-216`）都拒绝已存在目标。建议缓存写入同样要求新路径或显式 `--overwrite` 并保留旧 attestation。级别 P2。

### G3-06 calibrate 失败会占用新输出目录；模型先落盘、回执后写

- 证据：任何 `:290` 之后的异常都会写 `summary.json` 后退出 1（`brainsync_model_task.py:438-442`）；而 `summary.json` 一旦存在，下次同目录运行被 `:272-273` 拒绝（“existing task records are immutable”）。模型在 `:400` 保存、回执在 `:417` 写入；若回执写失败，会留下无回执的 artifact，重试又因 artifact 已存在被 `:321-322` 拒绝。
- 影响：一次瞬态失败（OOM、磁盘、设备错误）迫使操作者换目录并可能换 artifact 名；失败记录与成功记录同样“不可变”。建议将 artifact 先写临时名、回执成功后提交；或允许 failed 记录在同一目录重试。级别 P2。

### G3-07 `run_brainsync_pilot` 的 plan 只有 schema 字符串校验，arm 名直接拼接为路径

- 证据：`experiments/run_brainsync_pilot.py:144-146` 仅检查 `plan["schema"]`；随后无校验地取 `plan["input_sha256"]`、`plan["caches"]`、`plan["device"]`、`plan["arms"]`（`:148-152`），缺键即 KeyError。arm 名唯一性只在 `:154-155` 检查，随后直接 `directory / f"{arm['name']}.pt"`（`:98`）与 `f"{arm['name']}.json"`（`:119`）：含 `../` 或以 `/` 开头的名字可写出 `--output` 之外。
- 另有 `:204` 的 `next(... waveform_lda ...)` 无兜底：plan 无 waveform_lda arm 时在 heldout 评估之后、`completed.json` 之前抛 StopIteration，整轮证据链缺失。仓库内无 plan 示例/结构文档，`tests/brainsync/test_brainsync_pilot.py` 只测 `validate_pilot_scopes`。
- 建议：为 `n2p3_brainsync_pilot/1` 定义必填键/类型校验，限制 arm 名为安全文件名并显式检查输出路径，去掉裸 `next`。级别 P2。

### G3-08 `correct_brainsync_reference` 就地改写的回滚不覆盖 BaseException

- 证据：`experiments/correct_brainsync_reference.py:106-113` 先写并校验 zip 归档，`:114-123` 的 `except Exception` 才回滚；Ctrl-C/进程崩溃（KeyboardInterrupt、SystemExit 或硬终止）不在回滚范围，多文件写到一半会留下部分已更正元数据。`manifest.json` 在最后写（`:126-135`），若该步失败则已有归档但无 manifest 映射。
- 正向：归档用 `"x"` 拒绝覆盖（`:102-103`）、逐文件比较后写、EDF 前后哈希核对（`:105,124-125`）、且不重写原始 EDF；这是本组入口中最完整的破坏性写入保护。建议把文件替换阶段放进可恢复事务（先写 `.new` 再 rename），或捕获 BaseException 回滚。级别 P2。

## 4. 无标签推理与“证据写入”核查

- 结论：`--task infer` 只写 `<output-dir>/summary.json`（`brainsync_model_task.py:411-417`）；不写模型、缓存、session.json/BIDS、不调用任何 save 函数。数据入口 `load_brainsync_inference_session` 明确是 target-free（`src/data/brainsync.py:645-667`），质量报告另有 `if dataset.y is not None: raise`（`src/research/recording_quality.py:262-263`）。summary 记录 `target_labels_used: False` 与逐重复预测/分数/计数，目标确认不在此后端路径。artifact 字节只读一次并哈希后从 `BytesIO` 加载（`brainsync_model_task.py:304-307`），避免“先哈希后换文件”的 TOCTOU。
- 剩余风险（非标签）：G3-09 的极短录制拒判路径、G3-11/12 的报告写出、以及 `--task infer` 未拒绝 `--batch-size/--seed/--selector-mode`（被静默忽略，不影响冻结模型）。推理不会把预测写成训练证据。

### G3-09 连续 QC 完全拒绝且排程 < 9 时，拒判回退会 `IndexError` 而非 abstain

- 证据：`experiments/brainsync_model_task.py:425` 用 `args.repetitions or error.scheduled_trials // 9` 推重复数；`error.scheduled_trials` 是 `len(validated.bids.stimuli)`（`src/data/brainsync.py:109`）。若录制只有 1–8 个已排程闪烁且未显式给 `--repetitions`，`requested = 0`，`:426-429` 生成空 `rows`，`:433` 的 `rows[-1]` 抛 IndexError，被 `:438` 捕获后写成 failed summary 并退出 1，而不是文档 `doc/workstation.zh.md:102` 承诺的“保留全部排程数、推理显示拒判”。
- 建议：`requested = max(1, ...)` 并对 0 排程单独定义；补一条 <9 闪烁的单测。级别 P2（边缘记录）。

### G3-10 `inspect` 静默忽略冻结处理参数，并记录未使用的 `--session-dir`

- 证据：`inspect` 分支（`brainsync_model_task.py:310-315`）只加载 artifact 与写 summary；`--source-checkpoint`、`--ocular-model`、`--continuous-qc-model`、`--epoch-qc-model`、`--reference-channel`、`--batch-rule` 均被接受但不使用；`:297-298` 只拒绝非 calibrate 的 `--preprocessing-config`。对比 infer 的严格冻结检查（`:406-408`）。`:288` 还把传入的 `session_dirs` 写进 summary，但 inspect 从不读取它们。
- 影响：GUI/脚本可能误以为“用这些覆盖参数检查了模型”，summary 却显示原模型合同与未使用的会话路径。建议 inspect 复用 infer 的覆盖拒绝列表，并拒收 `--session-dir`（或明确标记 unused）。级别 P2。

### G3-11 quality report 对合同/参数错误一律退出 0

- 证据：`experiments/report_brainsync_quality.py:28-73` 的 `main` 无 `sys.exit`/返回值；`create_recording_quality_report` 在 `:317-326` 捕获所有异常，写 `status="not_evaluable"` 的 report 后正常返回。异人模型、校准记录复用、`--epoch-qc-model` 损坏、连续 QC 模型不匹配等都在 report 内被降级成说明文字。
- 影响：文档 `doc/workstation.zh.md:77` 称“参数损坏会明确失败，不改用当前设置”。报告确实不改用当前设置，但进程退出码不能区分“报告成功”与“参数损坏”；调用方必须解析 `report.json.status`。建议对“输入 artifact 不可读/合同不匹配”返回专用非零码（如 3），把“录制本身不可评估”保留为 0 + not_evaluable。级别 P2。

### G3-12 quality report 未绑定独立 epoch-QC 文件摘要，且输出目录无会话包含约束

- 证据：`report_brainsync_quality.py:44-45` 的 `request` 只记录参数路径；只有 `--model-artifact` 额外写 `model_sha256`（`:51`）。独立 `--epoch-qc-model` 的 `qc_record` 只有 method/retained/fit_provenance/summary，无文件 SHA 或 model digest（`src/research/recording_quality.py:168-170`）；连续/EOG 处理记录至少带 `model_digest`/model（`:271-273`）。
- 另：输出目录只对 `report.json` 做独占（`:228-231`），`report.html` 与固定名 PNG 在 `:207-223,324-326` 直接写；未检查 output_dir 与 session_dir 的关系，调用方可把报告写进 BIDS 根并覆盖同名文件。建议记录 QC 文件 SHA，并要求 output_dir 位于专用 `analysis/quality_reports/<version>` 子目录或至少不等于 session_dir。级别 P2。

## 5. 模型保存/重载身份与兼容

- 正向：`save_calibrated_predictor` 用临时文件+硬链接独占发布（`transfer/calibrated.py:1198-1217`）；`load_calibrated_predictor` 校验 schema、metadata_digest、state_digest，构造器再核对架构、输入统计形状、ERP 基/配方、LDA 特征形状与非有限值（`:1220-1236, 569-663`）；`torch.load(..., weights_only=True)`（`:1222`）阻止工件内代码执行。inference scope 拒绝复用校准会话及早于校准就绪的录制（`:815-824`）。

### G3-13 calibrate 保存后不做重载验证，pilot 做而生产入口不做

- 证据：`run_brainsync_pilot.py:98-102` 保存后立即重载并比较 logits 逐位相等；`brainsync_model_task.py:400-417` 保存后只读回字节算 SHA，不重载。若序列化/元数据在本机版本组合下不可读，calibrate 仍会以 0 退出并给出 completed summary，下一次 infer 才失败。
- 建议：calibrate 结束前用同一 `load_calibrated_predictor` 重载并核对 metadata/state digest（或至少比较一次 logits），把结果写入 summary。级别 P2。

### G3-14 加载器接受 /1、/3，但 CLI 消费者无兼容分支，旧包可能 KeyError

- 证据：`load_calibrated_predictor` 接受 `n2p3_calibrated_predictor/1,3,4`（`transfer/calibrated.py:1223-1226`）。但 `_model_summary` 无条件读 `metadata["input_request"]`、`metadata["selection"]`、`calibration_*_timestamp_s`、`source_pretraining_used`（`brainsync_model_task.py:80-110`）；`load_inference_session` 无条件读 `metadata["input_request"]`（`calibrated.py:672-673`）。真正缺少这些键的旧包会以 KeyError 失败，而不是“不支持的旧包”说明。
- 测试只把当前 payload 的 schema 字段改成 /1//3（`tests/brainsync/test_personal_quality_dispatch.py:180-191,206-214`），未用历史 metadata 做 inspect/infer 回归。文档 `doc/workstation.zh.md:245` 的“仍可读取通过完整旧合同核验的 /3、/1 包”缺少可执行证据。建议加载器对旧 schema 做显式 metadata 迁移/必需键检查，并补旧包端到端测试。级别 P2。

### G3-15 summary 的 `ocular_fit_policy` 由源预训练标志推导，与真实 EOG 拟合范围不一致

- 证据：实际 EOG 拟合范围在 `brainsync_model_task.py:361-371`：非 unified、>1 会话且（有源 checkpoint 或非 legacy_product）时，只拟合前 80% 会话。summary 标签在 `:102-106` 却只在 `source_pretraining_used` 为真且模式非 fixed_unified_program 时写 `frozen_inner_training_sessions`，否则写 `authorized_calibration_sessions`。因此 `--selector-mode five_candidates/no_transfer` 且无源 checkpoint 时，EOG 实际排除了验证会话，标签却声称用了完整授权校准集。
- 影响：summary 与模型的 `ocular_fit_sessions`（`:101`，真实列表）自相矛盾；审计若只读 policy 字段会误判拟合范围。单会话情况下 `calibration_inner_split` 是记录内 80/20（`transfer/calibration_split.py:109-121`），而 EOG 仍拟合整段录制，标签也会写成 authorized。
- 建议：直接由 ocular fit provenance 的 `calibration_session_ids` 与会话集合推导 policy，或删除该冗余文本字段。级别 P2。

## 6. P2：配置 schema、路径与退出码

### G3-16 `fit_brainsync_continuous_qc` 的 `preprocessing` 绕过完整配置校验

- 证据：`experiments/fit_brainsync_continuous_qc.py:36-38` 直接 `PreprocessingSpec(**manifest["preprocessing"])` 后 `validate()`。`PreprocessingSpec` 每个字段都有默认值（`src/data/epochs.py:55-85`，含 `notch_freqs_hz=()`、`zero_phase_edge_policy="legacy"`），因此 `{"name":"x","sfreq":128,"tmin_ms":-200,"tmax_ms":1200,"n_times":179}` 这类局部配置会静默补齐 P300 默认并通过 `validate()`（只读探测已确认）。而 `prepare_brainsync_cache.py:71`、`brainsync_model_task.py:57-60`、`report_brainsync_quality.py:53` 都走 `load_preprocessing_spec` → `parse_preprocessing_spec`，后者对缺字段报错（探测：`{}` 与部分对象均被拒）。
- 影响：同一份“完整配置”规则在三个入口严格执行、在连续 QC manifest 入口不执行；漏写 `notch_freqs_hz`/`zero_phase_edge_policy` 会静默改变 QC 拟合的物理算子（之后与默认预处理的数据搭配时通常会因 SOS/recovery 不符而 fail-closed，但失败发生在更晚且信息更差）。建议改用 `parse_preprocessing_spec(manifest["preprocessing"])`。级别 P2。

### G3-17 连续 QC manifest 只有顶层 schema 校验，嵌套键错误直接 traceback

- 证据：`fit_brainsync_continuous_qc.py:31-45` 仅校验 `schema == "brainsync_continuous_qc_calibration/1"`；随后 `manifest["spec"]`、`manifest["sessions"]`、每一项的 `session_dir/clean_intervals_samples/clean_interval_evidence`、`manifest["epoch_recovery"]` 都被直接索引/解包（`:39,47-62,97-102`）。缺键为 KeyError，`epoch_recovery` 参数不匹配为 TypeError，均无 summary、无结构化错误。`--output` 是否存在只在保存时由 `save_continuous_qc` 检查（`data/continuous_qc.py:627-632`），错误形式为 FileExistsError traceback。
- 建议：给 manifest 加版本化必填/未知键校验，并把 CLI 错误统一为“打印一行原因 + 退出 2”。级别 P2。

### G3-18 `brainsync_workstation.model.json` 仅四个字段被测试，解释器为 Windows 专属相对路径，且未声明预处理合同

- 证据：`configs/brainsync_workstation.model.json:5` 的 `python_executable` 为 `../.venv/Scripts/python.exe`（Windows-only，相对基址未在仓库内定义）；`:6` entrypoint、`:7` tasks、`:13` 的 `arguments` 字符串、`:4` version 均无校验。`tests/brainsync/test_brainsync.py:342-353` 只断言 `session_schema`、采集通道/类型、模型 EEG 通道与 `eog_in_classifier`。
- `input_contract`（`:17-27`）与 `output_contract`（`:28-33`）都没有滤波/陷波/基线/epoch 参数；而 `prepare_brainsync_cache.py:71` 的默认预处理是 `BRAIN_SYNC_PREPROCESSING`。后端默认一旦变化（例如改变陷波），外部采集端 manifest 版本不会随之变化，缓存合同与 manifest 声明脱节。
- 建议：把预处理合同（或至少其 digest）纳入 manifest/测试；把 `arguments` 改为 argv 列表以避免引号/空格解析；显式声明解释器相对基址。级别 P2。

### G3-19 `--selection-protocol` 对 legacy_product 被静默忽略

- 证据：CLI 对非 unified 接受 `binary_auc|brainsync_operational_r5`（`brainsync_model_task.py:243-255`），不限制 selector；但 `fit_calibrated_predictor` 对 legacy_product 固定 `protocol="binary_auc"`（`transfer/calibrated.py:1143`），预检也固定报告 `legacy_epoch_validation_cross_entropy`（`:358-360`）。用户传 `--selection-protocol brainsync_operational_r5 --selector-mode legacy_product` 不会报错，也不会生效。
- 建议：对 legacy_product 拒绝显式 `--selection-protocol`（或仅允许 binary_auc），与 unified 的严格处理保持一致。级别 P2（静默无效参数，易误读结果）。

### G3-20 recipe/budget 的读取/解析发生在 try/回执机制之外

- 证据：`brainsync_model_task.py:256-269` 在 `:290` 的 try 之前执行 `read_text/json.loads/TargetRecipe(**...)`；文件缺失、JSON 损坏、未知字段或类型错误会直接抛 FileNotFoundError/JSONDecodeError/TypeError，进程以裸 traceback 退出 1，且不写 `n2p3_brainsync_model_task/2` summary。相对地，`:290` 之后的错误都会写 failed summary。
- 影响：采集端按 summary 判断任务结果时会看到“无结果/崩溃”，与文档 `doc/workstation.zh.md:84` 的“新任务结果必须通过…摘要核对”流程不一致。建议把这些解析移入 try 并写 failed summary（参数错误也可先用 parser.error 退出 2）。级别 P2。

### G3-21 unified/fixed_erp 不执行“每个校准 session 各自含两类”的检查

- 证据：仅 legacy_product 在 `transfer/calibrated.py:1060-1063` 逐 session 检查 `{0,1}`；unified 分支（`:1086-1107`）与 fixed_erp（`:1071-1085`）没有该循环。真正的类别门禁是 union 级 `set(unique(labels)) == {0,1}`（`transfer/calibration_candidates.py:579-581`），因此“一个单类 session + 一个双类 session”的校准集可以通过 unified 拟合。
- 文档 `doc/workstation.zh.md:243` 写“每个校准session各自应具备两类，不能拼接两个单类session伪造合格训练集”。对 unified 这条声明不成立。建议把逐 session 两类检查上移到 fit/preflight 的通用入口，或在文档限定为 legacy/选方路径。级别 P2。

### G3-22 `run_brainsync_cross_decision`：单类测试会中止整轮；部分参数对某些 head 静默无效

- 证据：`:482` 对每个 subject 直接 `roc_auc_score(dataset.y[test_rows], raw_logits)`，若某 subject 的 test_rows 只有单一类别（小 `--max-test-selections`、部分排程或失败会话）会抛 ValueError；该异常不在逐 subject 的 try 内，且 summary 只在最后写（`:705-706`），整轮结果全部丢失。
- 另：`--epochs`/`--lr` 对 `zero_shot`/`target_shrinkage_lda` 被 `:317-336` 静默置 0/0.0；CLI 不报“该 head 不使用这些参数”，contract 记录的是 0（执行值正确，用户输入被丢弃）。`--batch-size` 对 zero_shot 也无效。
- 建议：AUC 单类时记 `binary_auc=None` 并继续（与 fit_failure 记录风格一致）；对不适用 head 显式 parser.error；把每个 subject 的 partial record 增量落盘。级别 P2。

### G3-23 EOG/连续 QC 拟合重复实现采样掩码逻辑，存在与数据入口漂移风险

- 证据：`fit_brainsync_eog.py:83-99` 与 `fit_brainsync_continuous_qc.py:63-76` 各自复制了“clock.acquired_samples 截尾 + excluded_intervals 置 False + acquisition_boundaries 注释”的逻辑；数据入口 `data/brainsync.py` 的加载/切 epoch 也维护同一语义。任何一侧更新（例如边界/暂停语义）而另一侧未同步，会让 QC/EOG 在“看似合法”的样本上拟合。
- 建议：抽出 data-core 的“authorized sample mask”单一实现，两个入口只传 session + clean intervals。级别 P2（与 G6 结构审计相邻，此处只报拟合范围风险）。

## 7. CLI 写入/覆盖安全矩阵

| 入口 | 主要写入 | 覆盖/独占保护 | 退出码与错误形态 |
|---|---|---|---|
| model_task calibrate | model artifact、summary、可选 epoch_quality.json | artifact 必须新（`:321,400`），summary 必须新（`:272`）；但见 G3-04/G3-06 | 0/1（部分带 summary）/2；recipe 读失败为裸 traceback |
| model_task infer | 仅 summary | summary 必须新（`:272`） | 0（含拒判）/1/2 |
| model_task inspect | summary、可选 epoch_quality.json | 同上 | 0/1/2 |
| prepare_brainsync_cache | npz + record.json | **无**，静默 replace（G3-05） | 0/1；skip 策略仍 0 |
| fit_brainsync_eog | 一个 EOG JSON | 目标必须新（`:156-157` 与 save 双重） | 0/1/2 |
| fit_brainsync_continuous_qc | 一个连续 QC JSON | save 拒绝存在（race 级） | 0/1（traceback）/2 |
| report_brainsync_quality | report.html/json + PNG | 仅 report.json 独占（G3-12） | 0（含 not_evaluable）/2 |
| run_brainsync_cross_decision | 一个 summary JSON | **无**，可覆盖输入（G3-03） | 0/1（traceback） |
| correct_brainsync_reference | 就地改 session/BIDS 元数据 + zip + manifest | zip 必须新；逐文件比较+回滚（G3-08） | 0/1/2 |
| run_brainsync_pilot | 多目录多 .pt/.json | 输出目录必须新（`:172`）；arm 名未净化（G3-07） | 0/1（traceback）；dry-run 也写 preflight |

## 8. 配置与文档一致性对照

| 配置/声明 | 当前实现 | 结论 |
|---|---|---|
| `brainsync_workstation.model.json` | 跟踪；测试仅四个物理字段；Windows 相对解释器；无预处理合同 | G3-18 |
| `brainsync_collection_plan.json` | 跟踪；测试覆盖目标策略/词表/4050 trial/450 target/1 block | 通过（未被后端代码消费，仅文档/测试） |
| `brainsync_epoch_support.json` | 跟踪；与 `contract.epoch_support_record()` 相等（探测） | 通过 |
| `brainsync_acquisition_reference.json` | 跟踪；与 `brainsync_reference_record()` 相等（探测） | 通过 |
| `brainsync_bidirectional_preprocessing.json` | **未跟踪**；完整性由 `parse_preprocessing_spec` 强制；当前等于运行时默认；无测试钉住 | G3-02/G3-16 |
| `unified_training_budget.json` | **未跟踪**；等于 `UnifiedTrainingBudget()` 默认；CLI 可选（unified 未传则默认 90） | G3-02/G3-20 |
| 连续 QC calibration manifest | 只校验顶层 schema；preprocessing 绕严校验；嵌套缺键 KeyError | G3-16/G3-17 |
| pilot plan | 只有 schema 字符串；无示例/无 runner 测试 | G3-07 |
| `--calibration-recipe` | dataclass 默认值填充缺省字段；model_task 不记录文件路径/哈希（记录内容进 selection） | G3-20（lineage 弱） |
| doc:4-6 unified 命令 | recipe 必填、epochs==target_epochs 强制（`calibrated.py:231`）、默认 90；budget 可选 | 部分成立（G3-02/G3-20） |
| doc:37-39 fixed_erp 可保存重载 | CLI 不可达 | **不成立（G3-01）** |
| doc:27-28 预处理配置必须完整 | 三个入口成立；连续 QC manifest 不成立 | 部分成立（G3-16） |
| doc:77 参数损坏明确失败 | 报告降级为 not_evaluable 且退出 0 | 表述过强（G3-11） |
| doc:88-90 设备/线程记录、CUDA 失败不静默回退 | model_task 成立；报告固定 CPU/1 线程（未声明设备） | 通过（报告不在该声明范围） |
| doc:100-103 报告重放冻结合同、完全拒绝保留排程 | 报告经 `predictor.load_inference_session` 重放（工作树改动）；拒判保留计数 | 基本成立；短录制例外见 G3-09 |
| doc:243 每个校准 session 各自含两类 | 仅 legacy_product 检查 | 部分不成立（G3-21） |
| doc:245 可读取 /3、/1 旧包 | 加载器接受 schema；CLI 消费者无兼容分支 | 仅有条件成立（G3-14） |
| doc:252-253 自动 EOG 只在内部训练 session 拟合 | 多 session 选方路径成立；非源 selector 的 summary 文本错误 | G3-15 |

## 9. 未覆盖与限制

- 未运行真实 `calibrate/infer` 及 `prepare/fit/report/pilot` 数据路径，也未运行 pytest（由 06-tests 覆盖）；结论基于源码、`--help`/parser 拒绝探测、只读 `python -c` 与 `ruff check`（8 文件全过）。
- 未审计外部 BrainSync 采集端（不在本仓库），因此 `doc/workstation.zh.md` 中 UI/发布 EXE/队列相关声明只在后端可验证部分核对。
- 本地没有真实的 /1 或 /3 历史模型包，G3-14 是代码级结论；未读取任何密钥。
- 工作树非 HEAD，G3 结论对应 2026-10-19 的当前工作树；若未跟踪配置被提交/删除，G3-02 需重核。

## 10. 探测记录（只读）

- 8 个入口 `--help`：全部 exit 0。
- `--selector-mode fixed_erp --calibration-recipe ...`：exit 2，`error: calibration-recipe belongs to unified calibration`（G3-01 证据）。
- `--selector-mode unified` 不带 recipe 或无参数：均 exit 2（分别报缺 recipe、列出必填 `--task/--model-artifact/--output-dir`）。
- `parse_preprocessing_spec({})` 与 `{"name": ...}`：均被拒（`preprocessing config must be complete`）；对照 `PreprocessingSpec(**partial) + validate()` 通过（G3-16）。
- bidirectional 配置解析结果 == `BRAIN_SYNC_PREPROCESSING`，SHA256 以 `DEFFAB5F…` 开头且与 frozen manifests 一致；unified budget == `UnifiedTrainingBudget()`；epoch_support/reference 配置 == 运行时记录。
- `resolve_target_batch_rule({"batch_rule": freeze_target_batch([])})` → `(32, ...)`，文档“缺少规则回退 32”成立。
- `ruff check` 上述 8 文件：All checks passed。

## 11. 给 lead 的 5 条

1. **G3-01（P1）**：`fixed_erp` 出现在 `--selector-mode` 但 CLI 禁止其唯一必需的 `--calibration-recipe`，文档却声称可保存重载；要么修 CLI，要么撤文档/选项。
2. **G3-02（P1）**：文档化的 unified/bidirectional 入口依赖两个未跟踪配置；bidirectional 配置还是 frozen manifest 的哈希对象，HEAD 克隆不可复现。
3. **G3-03（P1）**：`run_brainsync_cross_decision --output` 无独占/路径身份检查，可静默覆盖既有结果，甚至覆盖 `--dataset-cache`/`--checkpoint` 输入。
4. **G3-05（P2）**：`prepare_brainsync_cache` 默认静默覆盖 npz + record.json；与模型/QC 的独占保存策略不一致，建议加新路径或显式 `--overwrite`。
5. **G3-11/G3-12（P2）**：quality report 对参数/合同错误一律退出 0，且独立 epoch-QC 文件无摘要绑定、输出目录无包含约束；调用方无法仅凭退出码/摘要核实“未换用当前设置”。
