# 分区 10：文档一致性审计报告

- 仓库：D:\BCI\n2p3-net（以当前工作树为准，非 HEAD）
- 审计时间：2026-10-19
- 范围：AGENTS.md、README.md、doc/README.md、六份正文、doc/evidence（111 个 evidence + README）、frozen/README.md、formal/lean/README.md、doc/assets 59 目录、doc/progress_report_*
- 方法：只读 read/grep/glob、只读 git、脚本静态解析 markdown 链接；未修改任何既有文件，唯一写入为本报告与 tmp/audit/scratch/links.json
- 级别：P0=阻断/误导科学结论；P1=入口失效或事实错误；P2=维护性/一致性问题

## 0. 结论摘要

1. **P1 断链是系统性的，不是个别笔误。** 111 个 evidence 中 8 个未被索引；32 个相对 markdown 链接（18 文件）指向 frozen manifest，27 个唯一目标中 24 个从未出现在 HEAD，仅 3 个是工作树删除（git status 显示 9 个 frozen 文件为 D）。这与 AGENTS.md:45-48、frozen/README.md:3-8 的“archive manifests stay in Git”直接矛盾。
2. **P1 “当前状态”中的测试结论已过期。** 文档反复引用 1539/1936/2095/1404/1530/1726/1650 等历史通过数，但 2026-10-19 全量日志为 **2331 passed, 3 failed, 8 skipped, 22 warnings**（439.48s，EXIT=1）；3 个失败均在 tests/transfer/test_preprocessing_execution_gate.py。全库 159 个 test 文件、1593 个 test 函数，文档从未记录当前结果。
3. **P1 同一发布物有四个体积数字。** 128.319 MiB（doc/README.md:67、engineering.zh.md:45）、119,627,193 B（evidence/README.md:59）、约 6.847 MiB（workstation.zh.md:67）、release-manifest 的 127,417,982 B/121.515 MiB。发布证据自身（brainsync_release_20260920.zh.md:58）又写 134,552,186 B/128.319 MiB；四个数字互斥。
4. **P1 索引缺口集中在新证据。** 唯一 2026-10-06 的 e2_prior_scale_20261006 完全未被任何活动文档提及；另 7 个 09-16/09-18 证据被一处 domain_transfer_major_20260920 链接“代替索引”。doc/assets 59 个目录中 6 个无任何入链，其中 lean_current_checks_20261018 为 60.4 MiB/1149 文件。
5. **P2 doc/README 已成编年史。** 173 行中 142 行（117 非空、21 个日期条目）是 2026-09-07 至 10-05 的状态流水，占 82%，与 evidence/roadmap 重复，并违反同文件 164 行“实施状态只更新执行计划”的自我规定。
6. **P2 文档承诺的入口在干净 clone 中不存在。** formal/ 完全未跟踪（git ls-files formal 为空），但 engineering.zh.md:74-85、roadmap.zh.md:3-129、formal/lean/README.md 把它当作仓库入口；2 个 progress_report docx 与 md 未跟踪且未被任何活动文档索引。
7. **P2 更新戳与实际内容脱节。** technical_structure 标“更新：2026-09-06”但内容到 10-16；engineering、workstation 标 09-07/09-06 但含 10-04/10-18 内容；roadmap 的“更新：2026-09-07”埋在 393 行。
8. **未发现 P0。** 文档没有把已证伪结论写成当前规范；主要风险是入口失效、状态过期、证据不可达。

## 1. 索引与链接完整性

### 1.1 未被索引的 evidence（8 个，已逐个核实）

判定口径：从 README.md、AGENTS.md、doc/README.md、doc/evidence/README.md 与六份正文的链接目标中查找；仅在 evidence 正文互相提及不算索引。

| evidence | 日期/主题 | 现状 | 建议入口 |
|---|---|---|---|
| e2_prior_scale_20261006.zh.md | 2026-10-06，E2 AR 阶数×先验尺度、exact_covariance/information_bound | 活动文档 0 次提及 | 在 evidence/README.md:21 的 10-05 条目后新增 10-06 条目；roadmap 的 E2 段加链接 |
| algorithm_chain_repairs_20260916.zh.md | 09-16，F1–F8、1936 通过 | 仅被 domain_transfer_major 摘要代替 | evidence/README.md:67 行“算法链八项缺陷是否修复”直接加链 |
| brainsync_transport_unification_20260917.zh.md | 09-17，串口/BLE 250 Hz | 仅被 domain_transfer_major 摘要代替 | evidence/README.md:58 行加链 |
| code_quality_audit_20260908.zh.md | 09-08，1 P1/6 P2 | 仅被 domain_transfer_major 摘要代替 | evidence/README.md:82 行加链 |
| source_modules_20260918.zh.md | 09-18，源接入/排程模块 | 仅被 domain_transfer_major 摘要代替 | evidence/README.md:57 行加链 |
| unified_budget_execution_20260916.zh.md | 09-16，分阶段预算执行器 | 仅被 domain_transfer_major 摘要代替 | evidence/README.md:63-64 行加链 |
| unified_compact_training_20260916.zh.md | 09-16，精简训练方案 | 仅被 domain_transfer_major 摘要代替 | evidence/README.md:64 行加链 |
| unified_model_implementation_20260916.zh.md | 09-16，统一模型逐模块实施 | 仅 engineering.zh.md:125 提及 assets 路径，无文档链接 | evidence/README.md:65 行加链 |

补充：evidence/README.md 共 117 个链接、108 个不同目标，domain_transfer_major_20260920 被 7 个不同问题行重复引用；这使“一个证据文件承担多组独立结论”的索引噪声很大。8 个缺口中有 7 个正是这种“摘要代替原文”的后果。

### 1.2 doc/assets 目录引用（59 个中 6 个无入链）

| 目录 | 规模 | 内容/判断 | 建议 |
|---|---:|---|---|
| lean_current_checks_20261018 | 60.4 MiB / 1149 文件 | 当前 artifact/CV 检查的 driver、zip、plan/result；活动 checkpoint 未提及 | 在 formal/lean/README 增加“当前检查输出”条目，或压缩归档到 frozen/ 后删除展开目录 |
| lean_selection_erp_20261018 | 7 文件 | freeze/nonfit/source_contract；无 md 入链 | 归入 Lean ERP 切片的证据表 |
| lean_v06_risk_20261018 | 6 文件 | closure audit/risk slice contract；无 md 入链 | 归入 checkpoint 的 V06.a 行或 Lean README |
| unified_budget_execution_20260916 | 14 文件 | acceptance/pytest/performance 日志；对应 evidence 本身未索引 | 先索引 evidence，再在文档中链到该目录 |
| unified12_results_20260918 | 1 文件 summary.json | 仅 evidence 名称出现，无 assets 入链 | 在 evidence/README.md:43 行补 assets 链接或删除重复摘要 |
| operator_research_202604 | 1 个空子目录 | analysis_stage3_repair 为空 | 删除空目录，或放入 README 说明 2026-04 研究已退役 |

补充：doc/assets 合计 172.2 MiB / 3514 文件，但 git 仅跟踪 56 个文件；.gitignore 只忽略 **/*.npz，其余（含 60 MiB 的 lean_current_checks）均为未跟踪。目录没有索引 README，也没有保留/归档规则。

### 1.3 断链完整清单（32 个相对链接 / 18 文件）

解析 181 个 md、1317 个链接，存在性检查失败 76 条；其中核心是 32 个指向 frozen/ 的相对链接，分布在 18 个文件：

```
README.md:106 -> frozen/technical_audit_before_20260906.manifest.json
doc/evidence/algorithm_chain_repairs_20260916.zh.md:109,110,111 -> frozen/algorithm_chain_20260916_{before,after}/source.manifest.json; .../validation/manifest.json
doc/evidence/brainsync_app_integration_20260906.zh.md:63 -> frozen/technical_audit_before_app_20260906.manifest.json
doc/evidence/brainsync_frontend_logic_20260916.zh.md:10 -> frozen/brainsync_frontend_logic_20260916_before/manifest.json
doc/evidence/brainsync_input_replay_20260909.zh.md:71,76 -> frozen/brainsync_input_replay_20260909_{validation,before}/manifest.json
doc/evidence/brainsync_recovery_20260908.zh.md:176 -> frozen/brainsync_recovery_before_frontend_20260908/frontend.manifest.json
doc/evidence/brainsync_transport_unification_20260917.zh.md:6,68 -> frozen/brainsync_transport_unification_20260917_{before,validation}/manifest.json
doc/evidence/completed_result_admission_20260909.zh.md:67,74 -> frozen/completed_result_admission_20260909_{validation,before}/manifest.json
doc/evidence/domain_transfer_chain_primary_source_audit_20260904.zh.md:8 -> frozen/technical_audit_before_20260906.manifest.json
doc/evidence/domain_transfer_implementation_20260916.zh.md:8,121 -> frozen/domain_transfer_20260915_before/manifest.json; .../20260916_validation/acceptance.json
doc/evidence/domain_transfer_plan_20260915.zh.md:5 -> frozen/domain_transfer_20260915_before/manifest.json
doc/evidence/preprocessing_contract_research_20260915.zh.md:254 -> frozen/preprocessing_contract_research_20260915/manifest.json
doc/evidence/source_modules_20260918.zh.md:74,76 -> frozen/source_modules_20260918_{evidence,before}/manifest.json
doc/evidence/technical_audit_20260906.zh.md:99 -> frozen/technical_audit_before_20260906.manifest.json
doc/evidence/unified_compact_training_20260916.zh.md:5,6,221,222 -> frozen/unified_compact_training_20260916_{before,acceptance}/*.manifest.json
doc/evidence/unified_model_implementation_20260916.zh.md:51,52,192,194 -> frozen/global_early_stopping_20260916_verified/*; frozen/unified_model_20260916_acceptance/*
doc/evidence/unified_model_plan_20260916.zh.md:499,638 -> frozen/unified_compact_training_20260916_before/previous_program.manifest.json; frozen/unified_model_plan_20260916_before/manifest.json
doc/evidence/unified_optimization_20260929.zh.md:201 -> tmp/counterexample_verification_20260929.py
```

逐目标核实（git cat-file -e HEAD:<path>）：
- **HEAD 中存在、工作树被删（3 个目标/5 条链接）**：frozen/technical_audit_before_20260906.manifest.json、frozen/technical_audit_before_app_20260906.manifest.json、frozen/brainsync_recovery_before_frontend_20260908/frontend.manifest.json。三者都在 git status 的 9 个 D frozen/ 文件中。
- **HEAD 中不存在（24 个目标/27 条链接）**：其余 frozen manifest/acceptance 路径。它们既不在工作树，也不在任何 Git 提交；不是“被删”，而是“从未入库”。这正是 AGENTS.md:45-48 与 frozen/README.md:3-8 承诺的反例。

修复建议：
1. 对 3 个 HEAD 存在的文件，用 `git restore --source=HEAD -- <path>` 恢复（本审计只读，未执行）；若确已废弃，改链到 frozen/README.md 并写明“本地归档，未分发”。
2. 对 24 个从未入库的目标，逐条选择：改链到 `frozen/major_domain_transfer_20260920/manifest.json`（该清单是 09-20 大版权威表，但只含 source_modules 等少数条目）；或删除“见 manifest”的链接，改成“归档保存在本地，manifest 未入库”。不能继续保留可点击的假链接。
3. README.md:106 改为 `frozen/README.md`，或恢复 manifest；README.md:101-103 的“GitHub contains ... evidence manifests”需与实际一致。

### 1.4 其他链接缺陷（不计入上述 32）

- **assets 内部断链 11 条**：progress_report_20260929/v2/source_report.zh.md:25,63,64,87,118,122,126（9 条，写成相对 doc/ 的 evidence/...，实际位于 doc/assets/.../v2/）；unified_compact_training_20260916/tests_transfer_consolidation.md:8；chain_improvements_20260929/boundary/candidates.md:14（t_out，疑似误识别）。
- **根路径写法 6 条**：operator_research_20261004/analysis_stage3_repair/report.zh.md:3 与 review_standards.md:11,15 使用 `/D:/BCI/...`，在 Windows/GitHub 均无法解析。
- **代码行号伪链接 27 条**：lean_outcomes_20261018.zh.md:11-20,39,41,43,49,136（20 条）与 source_requirements_01.zh.md:139-145（7 条）把 `D:/...py:280` 写成 markdown 链接目标；应改为反引号代码，不应作为链接。
- **反引号内的死路径 37 条**：包括 frozen/*manifest 20+ 条、tests/run_headless.py、tests/test_domain.py、tests/test_geometry_ajd.py、tests/test_impedance_display.py、configs/brainsync_offline_preprocessing.json 等。反引号不产生可点击断链，但同样误导“文件存在”。

### 1.5 归档契约冲突（P1）

- AGENTS.md:45-48：“Put independent evidence in doc/evidence/ and index it... archive manifests stay in Git.”
- frozen/README.md:3-8：“This directory tracks manifests... Historical archive payloads stay local.”
- 实际：111 个 evidence 有 8 个未索引；27 个被链接 manifest 中 24 个从未入库；git status 还有 9 个 frozen manifest 为 D。**规范、索引、归档三者的承诺不一致**。这是本分区最需要 Lead 统一决策的事项：要么补交 manifest，要么把规范改成“只保留 consolidated manifest，逐阶段 manifest 本地留存”。

## 2. 文档-代码一致性

### 2.1 README 入口表（17 个表格脚本，不是 19）

README.md:66-79 的 12 行表格实际列出 17 个反引号脚本；加 README.md:114 的 dashboard_server.py，共 18 个不同脚本。全部存在于 experiments/ 下（17/17 表格脚本 + dashboard_server.py）。任务前提的“19 个”未复现；若预期还有 2 个（例如 prepare_source_records.py、run_pretrain.py），它们不在入口表内，应补表或修正计数。README.md:81 “Scripts live under experiments/” 与实测一致。

### 2.2 technical_structure 模块表 vs src

- 91-104 行“阶段/实际模块与入口”表、392-400 行复用表涉及的全部 39 个 `data/models/train/transfer/research/stats/baselines` 包/模块均存在；experiments/boundary_state_research.py、run_domain_transfer_matrix.py、prepare_source_records.py、run_source_pool_matrix.py、run_public_pool_targets.py、verify_pool_matrix.py、run_bnci_continuous_study.py、run_brainsync_recovery.py、analyze_brainsync_recovery.py、correct_brainsync_reference.py、freeze_source_snapshot.py、benchmark_unified_training.py、run_source_training_chain.py 均存在。模块表在“包级”是准确的。
- 32-42、103-133 行的 transport.py、decoding.py、decoding_results.py、theme.py、quality_reports.py 属于采集端项目，不在本仓库；文中多数已标明“采集端”，但 103 行表格把 `采集端decoding.py` 与后端模块混在同一单元格，建议加“仓库”列。
- 主要问题不是模块表，而是**更新戳**：第 3 行“更新：2026-09-06”与第 9-77 行（10-05/10-04/09-20/09-18/09-17/09-16/09-15）、436-442 行（10-16）矛盾。technical_structure 已被写成“结构+实施日志”，违反 doc/README.md:164 的分工。

### 2.3 engineering 指南的 tests 路径与命令

- :28 `pytest tests/brainsync/test_calibrated_predictor.py tests/brainsync/test_brainsync.py`：两文件均存在，命令有效。
- :39 `python tests/run_headless.py`、:42 `tests/verify_transport_ui.py`、:70 `tests/verify_quality_report.py`：本仓库均无；实测存在于 `D:/BCI/BrainSync_Tool_newest/BrainSync_Tool/tests/`。:39 未写 cwd，容易被误读为本仓库根目录；:42 虽在“采集端”段落，但路径仍非仓库相对。建议三处统一写成 `cd D:/BCI/BrainSync_Tool_newest/BrainSync_Tool; python tests/...`，并注明“采集端独立项目，不在本仓库”。
- :60 `test_brainsync_input_replay.py`、`test_recording_quality_report.py`、`test_calibrated_predictor.py`：实际在 tests/brainsync/，裸文件名可读但应与 :28 的全路径写法统一。
- :168-169 `tests/data/test_filter_phase.py`、`test_bidirectional_recovery.py`、`test_preprocess.py`、`test_notch_filter.py`：四个文件均存在于 tests/data/；`test_preprocess.py` 不是旧平铺路径的幽灵，但后三个只写裸名，与前一个全路径不一致。建议全部写成 tests/data/...。
- :64、:132、:135、:238、:465 的测试计数与“当前状态”问题见 2.5。

### 2.4 workstation CLI 与外部路径

- :3-9 的 `brainsync_model_task.py --task calibrate --selector-mode unified --calibration-recipe ... --training-budget ...`：experiments/brainsync_model_task.py:215 定义 --task（calibrate/infer/inspect），:238 --selector-mode，:239 --calibration-recipe，:241 --training-budget；:259-269 还强制 unified 必须提供 recipe。文档命令与 argparse 一致。
- :49-50、:59-60、:65 的 Launch_BrainSync.cmd、Launch_Mock.cmd、RUN_GTN_WORKSTATION.md、BrainSync.exe 在 `D:/BCI/BrainSync_Tool_newest/...` 与 `D:/BCI/BrainSync_Release_20260920/` 实测均存在；这部分是准确的。
- :67 “排除运行时后的载荷约 6.847 MiB”与发布清单不符（见 2.6）。

### 2.5 测试计数与当前全量结果（P1）

| 位置 | 声称 | 语境 |
|---|---:|---|
| engineering.zh.md:64 | 1539 通过 / 4 跳过 | 2026-09-08 入口修复 |
| engineering.zh.md:132 | 2095 通过 / 4 跳过 | 2026-09-16 统一主链最终 |
| engineering.zh.md:135 | 1936 通过 / 4 跳过 | 2026-09-16 算法链修复 |
| engineering.zh.md:238 | 1404 项 / 1400 通过 / 4 跳过 | 2026-09-08 最终后端 |
| engineering.zh.md:465 | 1530 通过 / 4 跳过 | 2026-09-08 环境 |
| doc/README.md:86 | 1936 通过 / 4 跳过 | 2026-09-16 算法链 |
| roadmap.zh.md:375,380 | 1726 / 1650 通过，4 跳过 | 2026-09-09 |
| 当前 tmp/audit/pytest-full2.log | **2331 passed, 3 failed, 8 skipped, 22 warnings, 439.48s, EXIT=1** | 2026-10-19 全量 |

- 3 个失败：tests/transfer/test_preprocessing_execution_gate.py::test_new_personal_model_keeps_binding_on_save_load_and_admits_new_source_context、::test_historical_personal_model_reloads_without_promoting_execution_evidence、::test_personal_model_rejects_receipt_substitution_even_when_outer_metadata_is_resealed。
- 静态统计：159 个 test_*.py、1593 个 `def test_`；当前收集 2342 项（含参数化）。文档的 2095 与 2342 不可直接比较，但“4 跳过”已变为 8，说明所有计数都早于当前树。
- 结论：doc/README 的“当前状态”没有任何 2026-10 的测试行；读者会把 1936/4 当作当前绿灯。建议把历史计数全部标注日期与命令，并在 doc/README 顶部放一个自动生成的“最后验证”块（日期、命令、passed/failed/skipped、失败文件链接），失败项归属分区 06。

### 2.6 发布体积四套数字（P1）

| 来源 | 数值 | 口径 |
|---|---|---|
| doc/README.md:67 | 128.319 MiB | 不含开发入口/测试/缓存 |
| engineering.zh.md:45 | 128.319 MiB | 排除 `_internal` 后，小于 200 MiB |
| doc/evidence/README.md:59 | 119,627,193 B | “无独立 runtime” |
| workstation.zh.md:67 | 约 6.847 MiB | 排除运行时 |
| doc/evidence/brainsync_release_20260920.zh.md:58 | 134,552,186 B / 128.319 MiB | 不含 `_internal/` |
| Release 目录 release-manifest.json | **127,417,982 B / 121.515 MiB**；runtime 294.375 MiB | payload_bytes_excluding_runtime |
| 本次实测 Release 目录 | 121.604 MiB / 13 文件（不含 `_internal`）；总计 415.98 MiB / 499 文件 | 工作树实测 |

6.847 MiB 接近 BrainSync.exe 单文件 6.804 MiB，说明该行写的是“exe”而非“发布载荷”。四个数字中只有 release-manifest 是机器生成的权威值；其余应删除并改为引用清单。

### 2.7 更新戳与内容日期

- technical_structure.zh.md:3 “更新：2026-09-06”，内容含 2026-10-16（:436）。
- engineering.zh.md:3 “更新：2026-09-07”，内容含 2026-10-16（:482）、2026-10-04（:87）。
- workstation.zh.md:20 “更新：2026-09-06”，内容含 2026-10-04（:37）、2026-09-17（:56）。
- roadmap.zh.md:393 “更新：2026-09-07”出现在 1080 行文档的中部，前面已有 2026-10-16 的 Lean 方案（:3）。
四份正文都应把“更新”改为顶部元数据（last_reviewed + 依据 commit + 覆盖范围），或在 CI 中校验“更新日期 ≥ 文中最大日期”。

### 2.8 progress_report 的编号与索引

- doc/progress_report_20260929.zh.md:81 是“## 5. 阶段三”，紧接着 :83 是“### 6.1”，:114 又回到“## 7. 阶段四”；缺少阶段二/章节 6 的一级标题，编号断裂。
- 该报告在任何活动文档中出现 0 次；doc/progress_report_20260929.zh.md 与两个 docx 均未跟踪（git status 为 ??）。
- doc/assets/progress_report_20260929/v2/source_report.zh.md 有 9 个相对链接按 doc/ 根目录书写，放到 v2/ 后全部断链；若 v2 docx 由它生成，交付物内链接不可用。
- 建议：报告移入 doc/reports/ 或 doc/evidence/progress_report_20260929/，在 doc/README 的“交付物”段落索引；docx 若需分发用 LFS 或明确 gitignore，md 必须跟踪；修正 5→6 编号。

## 3. 维护性与结构

### 3.1 doc/README 的“当前状态”已成编年史

- 文件 173 行；`## 当前状态` 从 19 行到 160 行，共 142 行，其中 117 行非空、21 行含日期；占全文 82%。
- 内容从 2026-10-04 倒叙到 2026-09-07，但开头第 5 行又插入 2026-10-05 的边界状态链接，时间顺序不纯；第 21-23 行再插入 Lean 当前 checkpoint。新读者无法判断哪一段是“现在”。
- 该段与 evidence/README.md 的“先看哪些结果”表、roadmap 的“近期实施状态/已完成与未完成”大量重复；例如 1936 通过、48/48 重放、2.65 倍提速、35 源/12150 单元都在两处以上出现。
- 建议：把 19-160 行迁到 `doc/evidence/status_log_202610.md`（或 roadmap 的历史段），doc/README 只保留 15-20 行“当前状态”，字段固定为：最后验证日期、当前阶段、最近一次全量测试、活动 checkpoint、下一门槛。历史细节由 evidence 索引承担。

### 3.2 “实施状态只更新执行计划”的规则被四份正文违反

- doc/README.md:164 明确：“研究约束写入研究规范，实施状态只更新执行计划，使用方式写入工作站指南。”
- 实际：technical_structure.zh.md:9-77 有 10-05/10-04/09-20/09-18/09-17/09-16/09-15 的逐日实施段；engineering.zh.md:87-136、482 有 10-04/10-16 的实施段；workstation.zh.md:3-47 有 09-16/09-15/10-04 的接口状态段；doc/README 自身也有 142 行状态流水。
- 这不是“多写了一点”，而是职责边界失效：同一事实有四个维护点，更新时必然漏一处（本次已发现测试计数、发布体积、更新戳三类漂移）。
- 建议二选一：要么严格执行“roadmap 唯一状态源”，把其余正文的日期段压成“当前行为”并链到 roadmap；要么修改 164 行规则，明确“roadmap 管阶段状态，technical_structure 管当前接口，engineering 管可执行维护规则”，然后删除重复的日期叙述。

### 3.3 同一事实多处重复（抽样）

- 1936 通过：doc/README.md:86、engineering.zh.md:135、doc/evidence/algorithm_chain_repairs_20260916.zh.md:6/72、assets/algorithm_chain_20260916/source_changes.patch 多处。
- 128.319 MiB：doc/README.md:67、engineering.zh.md:45、evidence/brainsync_release_20260920.zh.md:58。
- 48/48 重放：doc/README.md:54、progress_report_20260929.zh.md:87、evidence/unified12_results_20260918.zh.md。
- 2.65 倍提速：doc/README.md:108、evidence/bidirectional_qc_20260909.zh.md。
- evidence/README.md 的 117 个链接中 domain_transfer_major_20260920 出现 7 次，承担“源模块/预算/统一模型/算法修复/代码质量/停止规则/发布前接口”等至少 7 组不同结论。
- 建议：数字只在 evidence 中定义一次；正文用“见 evidence/xxx#anchor”代替复制。索引表一行一个证据，禁止一个链接承担多组独立结论。

### 3.4 六份正文的体量与职责

| 文档 | 行数 | 主要职责 | 观察 |
|---|---:|---|---|
| research_program | 393 | 研究目标/合同/指标边界 | 第 359-393 行“历史与执行”含 09-06 结果（0.8136），与 roadmap/evidence 重叠 |
| technical_structure | 456 | 模块/接口/目录 | 第 9-77 行是实施日志，不是结构；更新戳 09-06 |
| input_contract_math | 443 | 单位/通道/参考/滤波/时间 | 合同主体清晰；第 410 行后附 09-07 预算约定，边界略混 |
| roadmap | 1080 | 阶段/配方/验收 | 体量最大，含大量实验设计和结果复述；第 361-395 行“近期实施状态”又插入 09-07 更新戳 |
| engineering | 532 | 环境/测试/性能/维护 | 测试计数与发布体积漂移的主要位置；含 10-04/10-16 历史 |
| workstation | 277 | 启动/采集/校准/推理 | 操作指南清楚，但第 3-47 行是接口状态史；发布体积错误 |

建议给六份正文加一个“职责矩阵”（谁拥有当前行为、谁拥有阶段状态、谁拥有历史），并在每份文件头部声明“不维护什么”。这比继续加长正文更能阻止重复。

### 3.5 doc/assets 的体积与保留策略

- 172.2 MiB / 3514 文件；git 仅跟踪 56 个；25 个 lean_* 目录占 98.8 MiB；lean_current_checks_20261018 单目录 60.4 MiB/1149 文件。
- 6 个目录无入链（1.2 节），其中 2026-04 目录为空；没有目录级 README、没有“生成物/证据/可删除”分类。
- .gitignore 只忽略 **/*.npz；大量 png/json/log/zip 处于未跟踪状态，既不在 Git，也没有本地归档清单。
- 建议：每个 assets 目录放 README.md（目的、生成命令、大小、保留期限、对应 evidence）；lean_current_checks 这类可重跑的大目录默认放 frozen/ 或本地 artifact 根，不进 doc/assets；在 .gitignore 增加明确的生成物规则，而不是只忽略 npz。

### 3.6 中英文与术语

- 根 README.md、AGENTS.md 为英文；doc/README.md 与六份正文为中文；这是可接受的分工，但没有一处声明语言策略，导致同一概念在两边命名不同（entrypoints vs 入口、evidence vs 证据）。
- 术语大小写不一致：Module/Interface/owner 常首字母大写，checkpoint/fit/recipe 小写；`活动checkpoint`（doc/README.md:22、formal/lean/README.md:57）缺空格；`当前状态` 有时指 Lean checkpoint，有时指全项目状态。
- 建议：在 doc/README 增加 10 行“语言与术语”小节，固定 Module/Interface/checkpoint/fit/recipe/evidence 的写法；中文正文首次出现英文术语时给中文解释，不另建词汇表文件。

### 3.7 progress_report 与 docx

- doc/progress_report_20260929.zh.md 21,388 B、progress_report_20260929.docx 289,844 B、*_v2.docx 292,322 B，三者均未跟踪；活动文档 0 次引用。
- 报告第 81-83 行编号从 5 跳到 6.1，第 114 行回到 7；这是可交付文档的结构错误。
- assets/progress_report_20260929/v2/source_report.zh.md 的 9 个断链说明 v2 是从 doc/ 根目录语境生成的；若直接转 docx，链接不可用。
- 建议：建立 doc/reports/ 或 doc/evidence/progress_report_20260929/；md 跟踪、docx 用 LFS 或 gitignore；在 doc/README 的“交付物”段索引；修正编号并把 v2 链接改为相对 assets 的路径。

## 4. Lean 文档

### 4.1 formal/ 完全未跟踪，但文档把它当仓库入口（P1）

- `git ls-files formal` 输出为空；formal/lean 下有 README.md、lean-toolchain、lakefile.toml、lake-manifest.json、N2P3/、scripts/、39 个 Audit*.lean、verification_plan.json 等；formal/ 总体 2.5 GB（含 .lake/artifacts）。
- engineering.zh.md:74-85 把 formal/lean 作为“选定数学规格的 Lean 验证入口”；roadmap.zh.md:3-129 是完整的 Lean 顺序方案；formal/lean/README.md:26-40 给出可执行命令。干净 clone 中这些路径全部不存在。
- 建议：要么跟踪源码子集（README、lean-toolchain、lakefile.toml、lake-manifest.json、N2P3/、scripts/、verification_plan.json、Audit*.lean；忽略 .lake/、artifacts/），要么在 engineering/roadmap/Lean README 顶部明确“formal/ 为本地未分发目录，clone 不含”。不能继续把本地路径写成仓库入口。

### 4.2 活动 checkpoint 的状态没有进入正文

- doc/assets/lean_gaussian_20261016/active_checkpoint.json：`status=paused`、`accepted_theorems=2890`、`last_updated_utc=2026-10-18T15:58:54Z`、`active_aggregation=N2P3.SavedControlManifestComponents`。
- doc/README.md:21-23 只说“当前进展见活动 checkpoint”，未写 paused/2890/日期；formal/lean/README.md:57 说“当前状态仅在活动 checkpoint 维护”，同样没有一行摘要。
- 建议：两处各加一行“截至 2026-10-18：paused，2890 条已验收定理，当前聚合 SavedControlManifestComponents；下一步见 checkpoint.next_actions”。读者不应为了知道是否暂停而打开 JSON。

### 4.3 39 个 Audit 文件、25 个 lean assets、3 个孤儿目录

- 39 个 Audit*.lean：34 个在 formal/lean 根目录，5 个在 artifacts/ 子目录；README 只点名 Audit.lean/AuditStage.lean，没有“切片 → Audit 文件 → verifier → evidence → assets”的索引。
- 25 个 lean_* assets 共 98.8 MiB；lean_current_checks_20261018（60.4 MiB/1149 文件）、lean_selection_erp_20261018（7 文件）、lean_v06_risk_20261018（6 文件）无任何 md 入链；checkpoint 也不提这三个目录名。
- 另有 16 个 lean_* evidence 文档，索引中只覆盖了一部分；Lean 证据的导航实际依赖 checkpoint JSON，而不是人可读的索引。
- 建议：在 formal/lean/README 增加一张表：切片名、Audit 文件、verifier、theorem 数、evidence、assets、状态（accepted/paused）。这张表也是 4.1 中“跟踪源码子集”的天然清单。

### 4.4 Lean README 的内部一致性

- 175 行；scripts/ 下有 13 个 verify*.py，README 引用其中 12 个（未提 verify_personal_erp.py）：verify.py、verify_stage.py、verify_continuous.py、verify_probability.py、verify_posterior.py、verify_conditional.py、verify_candidates.py、verify_joint_state.py、verify_event_ledger.py、verify_personal.py、verify_temporal_operator.py、verify_erp.py 均存在。
- 第 42 行 `verify.py` 是 prose 简称，实际文件在 scripts/verify.py；第 38 行命令也写 `formal/lean/scripts/verify.py`，可读但建议统一为 `scripts/verify.py` 并说明 cwd=formal/lean。
- 第 55-57 行链接 roadmap 的 Lean 锚点、active_checkpoint、frozen/lean_personal_20261016/manifest.json 均存在；Lean 文档本身的链接质量明显好于 evidence 的 frozen manifest 链接。
- 第 3 行 Lean 4.24.0 与 lean-toolchain 的 v4.24.0 一致；第 16-22 行的证明边界（不形式化 NumPy/BLAS/浮点）写得很清楚，应作为其他证据文档的模板。

## 5. 改进建议（按优先级）

### P1（本轮应处理）

1. **修 32 条 frozen 断链**：3 个 HEAD 存在的用 git restore 恢复；24 个从未入库的改链到 consolidated manifest 或删除链接并注明“本地归档”。同步修正 README.md:106 与 README.md:101-103 的“GitHub contains manifests”。
2. **统一归档契约**：在 AGENTS.md/frozen/README.md 明确“逐阶段 manifest 是否入库”；若不入库，禁止 evidence 写“见 frozen/...manifest.json”，改用“归档保存在本地，未随 Git 分发”。
3. **加“最后验证”块**：doc/README 顶部与 engineering:72 段各放一个自动生成块：日期、命令、2331 passed/3 failed/8 skipped、失败文件链接；历史 1539/1936/2095/1404 全部标注日期并移出“当前状态”。
4. **发布体积单一来源**：删除 doc/README:67、engineering:45、workstation:67、evidence/README:59 的数字，统一写“见 release-manifest.json 的 payload_mib_excluding_runtime（当前 121.515 MiB）”；修正 6.847 MiB 的 exe/载荷混淆。
5. **索引 8 个 evidence + 6 个 assets 目录**：按 1.1/1.2 的建议入口补链；e2_prior_scale_20261006 必须进入 evidence 索引与 roadmap E2 段。
6. **决定 formal/ 的分发策略**：跟踪源码子集，或在三处文档显著标注“本地未分发”；不能两者都不做。

### P2（下一轮）

7. **doc/README 状态段迁移**：19-160 行移到 doc/evidence/status_log_202610.md；README 保留 15-20 行当前状态 + 链接。把 164 行规则改成可执行的职责矩阵。
8. **更新戳治理**：每份正文头部加 last_reviewed/commit/coverage；CI 校验“更新日期 ≥ 文中最大日期”（可用脚本提取 20xx-xx-xx 最大值）。
9. **文档检查脚本 + CI**：新增 tools/check_docs.py：解析相对链接、反引号路径、evidence 索引覆盖、assets 目录引用、frozen manifest 白名单、更新戳；输出 JSON 并以非零退出。仓库当前无 CI，可先做 pre-commit 或手动脚本。
10. **证据索引模板**：每个 evidence 头部固定字段（日期、状态 active/superseded、scope、相关正文、assets、archive）；evidence/README 一行一证据，禁止 7 次重复同一链接。
11. **assets 保留策略**：目录级 README、生成物/证据/可删除分类、lean_current_checks 移出 doc/assets；.gitignore 增加明确规则。
12. **progress_report 治理**：移入 doc/reports/ 并索引；md 跟踪、docx 用 LFS 或 gitignore；修正 5→6 编号与 v2 的 9 条断链。
13. **Lean 索引表**：切片→Audit→verifier→theorem→evidence→assets→状态；补 3 个孤儿目录的归属。
14. **语言/术语**：doc/README 加 10 行语言策略与术语表；统一 Module/Interface/checkpoint/fit/recipe 的写法。

## 附录 A：复现与审计边界

- 链接解析：Node 脚本遍历 181 个 md（doc/ 全部、README、AGENTS、frozen/README、formal/lean/README），提取 `](target)` 1317 条；结果存 tmp/audit/scratch/links.json。核心 32 条 = 目标含 frozen/ 的相对链接；27 个唯一目标逐一用 `git cat-file -e HEAD:<path>` 核实。
- 证据覆盖：仅以 README、AGENTS、doc/README、evidence/README、六份正文的链接目标为准；evidence 之间互提不算索引。assets 覆盖以解析后的链接目标是否落在 doc/assets/<dir>/ 为准。
- 测试：未由本分区重跑全量；使用 Lead 的 tmp/audit/pytest-full2.log（UTF-16LE，2026-10-19 08:10 本地，2331 passed/3 failed/8 skipped/22 warnings/439.48s/EXIT=1）。静态计数 159 个 test_*.py、1593 个 `def test_`。
- 发布体积：读取 D:/BCI/BrainSync_Release_20260920/release-manifest.json 的 payload_bytes_excluding_runtime=127,417,982、payload_mib_excluding_runtime=121.515、runtime_mib=294.375；并实测目录 499 文件/415.98 MiB、不含 _internal 13 文件/121.604 MiB。
- 未做：未验证全部 markdown 锚点（只抽查 Lean/README 的 18 个带 # 链接，未见明显失效）；未审计其他分区；未修改任何既有文件；未运行 git 写命令；未安装依赖。

