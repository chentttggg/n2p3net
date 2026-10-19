# G6 代码结构量化审计（src / tests / experiments 顶层）

- 仓库：`D:\BCI\n2p3-net`，工作树非 HEAD（main 停 `f1a303c`，431 项未提交，tests/formal 未跟踪）
- 审计时间：2026-10-19；只读审计，仅写本报告与 `tmp/audit/scratch/`
- 范围：`src/**/*.py` 120 个（48,792 SLOC）、`tests/**/*.py` 185 个（34,519 SLOC）、`experiments/*.py` 顶层 69 个（23,619 SLOC），共 374 文件 106,930 SLOC；`experiments/runs`（8,134 个 .py 运行产物）按任务要求排除
- 去重声明：本分区只做"结构量化 + 重构收益排序"。data/transfer 的功能契约、数学统计、测试策略、工程/性能/Lean 结论以既有分区报告为准，此处不重复；与 02/06/08 报告重叠处仅补数字与代码定位。

## S1 方法与可复现性

| 项 | 口径 | 脚本 |
|---|---|---|
| 函数体指纹 | 去 docstring 后对 `ast.dump` 取 sha1；`Name`→`_`、`arg` 名→`_`（检测改名的整段复制），保留属性名与常量 | `tmp/audit/scratch/g6_ast.py` |
| 块级克隆 | 函数体内 8 行窗口，去注释、去缩进、去空行后哈希；跨文件出现 2–40 次者保留，相邻窗口合并为极大块 | `g6_ast2.py` |
| 近重复函数 | 同包顶层函数（span 25–1200、≥60 token），token 归一化（标识符→V，数字→N，字符串→S）后 `difflib.ratio ≥ 0.80` | `g6_ast4.py` |
| 引用计数 | 374 文件全 AST 的 `Name`/`Attribute`/`import` 名/字符串常量分词；区分"他用/本模块自用/仅测试/仅实验/字符串引用" | `g6_ast2.py`、`g6_ast3.py` |
| 导入图 | 逐条 import 解析到仓库模块；按"模块级(eager)/函数体内(deferred)"分类；Tarjan SCC（模块级与包级） | `g6_ast.py`–`g6_ast5.py` |
| 中间数据 | `tmp/audit/scratch/g6_data.json`、`g6_data2.json`、`g6_neardup.json` | — |

解析失败 0 个。默认 `python`（3.14，无 mne）仅做 AST；导入验证用仓库 `.venv`（3.14.3 / mne 1.12.1）。

## S2 重复函数体：函数级复制几乎为 0，块级集中在 experiments

1. **整函数归一化哈希：src 内跨文件克隆 0 组**（1014 个候选：535 顶层函数 + 310 方法 + 169 类；门槛 span≥8、语句≥4、指纹≥250 字符，无任何跨文件同哈希）。结论：src 没有"改名式整段复制"，此前担心的复制蔓延不成立。**P3（无需行动）**。
2. **块级克隆 19 段 / 221 行**，占 106,930 SLOC 的 0.21%，且分布极偏：

| 重复行 | 文件对 | 证据 |
|---:|---|---|
| 95 | experiments/run_candidate_cross_decision.py ↔ run_within_subject_transfer.py | 5 段：`run_candidate_cross_decision.py:1076/929/1001/1232/1488` ↔ `run_within_subject_transfer.py:543/375/485/597/762` |
| 22 | run_brainsync_cross_decision.py ↔ run_candidate_cross_decision.py | `:251`/`:342` ↔ `:734`/`:961` |
| 20 | run_candidate_cross_decision.py ↔ run_pretrain_supervised.py | `:95` ↔ `:145` |
| 18 | src/data/events.py ↔ tests/data/test_domain.py | `events.py:621` ↔ `test_domain.py:218`（2 段） |
| 11 | tests/train/test_source_execution_resources.py ↔ test_source_training_chain.py | `:145` ↔ `:416` |
| 11 | run_eeg_loso.py ↔ run_n2p3_sensitivity.py | `:552` ↔ `:366` |
| 9 | src/data/bi2014a_candidate.py ↔ bnci2014_008_candidate.py | `:365` ↔ `:648` |

3. **近重复函数 3 对（0.885–0.924），全部同文件**，是真正值得合并的重复实现：

| ratio | 函数 A | 函数 B |
|---:|---|---|
| 0.924 | `data/artifact.py:682 apply_fitted_artifact_model`（51 行） | `artifact.py:735 apply_fitted_artifact_model_with_test_exclusion` |
| 0.899 | `data/domain.py:43 adapt_common_channel_average_reference`（81 行） | `domain.py:126 adapt_native_reference_selection` |
| 0.885 | `baselines/evaluate.py:794 _run_shared_fold_task`（60 行） | `evaluate.py:911 _fit_shared_artifact_policy_task` |

   已读 artifact.py:682-741：两函数签名与开头逐行一致，ratio 0.924 表明主体近似（参数、校验、`fitted.transform`、mask 收尾顺序一致），仅 test-exclusion 分支不同 → 合并为"带策略参数的单一实现 + 薄封装"可消除漂移风险。
4. **实验脚本的 95 行重复已发生语义分叉**（不是纯复制）：`run_candidate_cross_decision.py:1083-1097` 与 `run_within_subject_transfer.py:550-563` 的 `SubjectAdapterConfig` 构造逐行相同，但前者写 `candidate_loss_weight`、后者写 `input_mean/input_std`。同一段配置在两个入口演化出不同字段 → **P2，配置漂移已在发生**。

## S3 无外部引用：真死代码 7 个函数，接口面比死代码更值得治理

引用分类（顶层符号，不含方法；`self_only`=`仅在定义它的模块内被引用`）：

| 类别 | 数量 | 行数 | 判读 |
|---|---:|---:|---|
| 无任何引用（真死） | 7 | 141 | P2，可直接删 |
| 仅测试引用 | 105 | 3,922 | 生产无消费者，属"测试支撑面" |
| 仅实验入口引用 | 27 | 597 | 研究入口 API，保留但应收窄导出 |
| 公开名但仅本模块自用 | 88 | 1,856 | P2，应改私有或补导出说明 |
| 私有名仅本模块自用 | 292 | 8,517 | 正常 |

真死 7 项：`research/domain_folds.py:124 domain_ceiling_folds`(42)、`baselines/validation.py:115 subject_disjoint_audit_split`(32)、`train/source_sampling.py:525 _update_min_distances`(23)、`data/metadata.py:129 build_subject_embeddings`(15)、`data/artifact.py:879 _relative_peak_to_peak`(14)、`data/channel.py:336 positions_from_raw`(11)、`data/contract.py:227 assert_paper_gtn_causal_input_contract`(4)。字符串动态引用命中 0，故"被 importlib 注册"的可能性低；但仍需 `grep -n 名称` 复核后删除。

模块级：**0 个孤儿模块**（每个非 `__init__` 的 src 模块都有导入者）。但有 **24 个模块没有任何 src 内部消费者**，只被入口脚本/测试导入，是"adapter 面"而非死码，最大者：`data/moabb.py`(1,881)、`baselines/evaluate.py`(1,509)、`data/domain_alignment.py`(661)、`data/ea_artifact.py`(560)、`transfer/information_bound.py`(473)、`transfer/evaluation.py`(363)、`transfer/mask_probe.py`(257)。这批模块不应按"死代码"清理（有测试与研究入口），但应在 `__init__` 明确不导出。

**空包**：`src/measurement/`、`src/audit_p300/` 均为 0 文件、`git ls-files` 无记录的空目录；`src/measurement` 在分层假设里还被当作 data 同层。**P2**：删除或补 README，避免结构误导。

## S4 导入环：无 import-time 环，6 个由函数内导入隐藏的环

- **纯 eager 环 0 个**（模块级 SCC 全为单点）。`.venv` 实测 `PYTHONPATH=src python -c "import data, transfer, baselines, models, research, stats, train; import data.source_records, data.study, transfer.target_experiments, baselines.deep, transfer.calibrated"` → `IMPORT_OK`。当前导入顺序安全。
- **含 deferred 边的环 6 个**（括号内为双向证据行）：

| 环 | eager 边 | deferred 边 |
|---|---|---|
| `models.adapters` ↔ `models.n2p3net` | adapters→n2p3net `adapters.py:19` | n2p3net→adapters `:109/:137/:804` |
| `data.epochs` ↔ `data.preprocess` | 无 | epochs→preprocess `:603`；preprocess→epochs `:636/:690` |
| `data.bidirectional` ↔ `data.recovery` | bidirectional→recovery `:21` | recovery→bidirectional `:253/:276` |
| `data.continuous_qc` ↔ `data.ocular` | 无 | qc→ocular `:515`；ocular→qc `:339` |
| **`data.study` ↔ `transfer.checkpoint`** | 无 | study→checkpoint `:230`；checkpoint→study `:109/:129` |
| `transfer.calibrated` ↔ `result_reuse` ↔ `target_experiments` | target_experiments→result_reuse `:14` | calibrated→target_experiments `:206`；result_reuse→calibrated `:272`；target_experiments→result_reuse `:438` |

判读：这些环靠"函数内延迟导入"维持可导入性，属于**脆弱平衡**——任一局部导入被上提到模块头即成真环（P2）。其中 `data.study ↔ transfer.checkpoint` 跨层（数据层↔迁移层），是 S5 反向依赖之一，**P1**。

- **包级只有一个 SCC**：`{baselines, data, models, research, stats, train, transfer}` 全部 7 个包强连通（2,133 条导入边中 src 415 条：329 eager + 86 deferred）。即：仓库**没有可拓扑排序的包分层**。这不是"某个包写错"，而是缺少共享内核包（见 S8 P2-3）。

## S5 分层违规：按导入方向多数派判定

仓库无书面分层契约，`doc/technical_structure.zh.md:272-273` 只明确了 `data → research.contracts/research.execution` 是允许的语义哈希/原子写依赖。故下表以**导入方向多数派 + 该条豁免**为准，不臆造更多规则。

包间边矩阵（eager/deferred，仅 src）：

| from \ to | data | models | transfer | train | baselines | research | stats |
|---|---|---|---|---|---|---|---|
| data | — | 0 | **0/7** | 0 | 0 | 10/0（内核） | 0 |
| models | 2/0 | — | 0 | 0 | 0 | 0 | 0 |
| transfer | 45/9 | 18/0 | — | 4/2 | **8/0** | **14/6** | 0/1 |
| baselines | 9/0 | 8/0 | **3/0** | **6/0** | — | 3/0 | 0 |
| train | 1/0 | 1/0 | 0 | — | **4/0** | 0/2 | 0 |
| research | 12/1 | 1/1 | 8/0 | 0/2 | 3/0 | — | 0 |
| stats | 0 | 0 | 0 | 0 | 0 | 1/0 | — |

真正需要处理的单向反向依赖（不是对称组合）：

1. **`data → transfer`：0 eager / 7 deferred，全部隐藏**。`data/source_records.py:46,56,343,551,653,804` 函数内 `from transfer.binary import ...`；`data/study.py:230` 函数内 `from transfer.checkpoint import ...`。数据层是基底，却依赖上层的二分类/检查点语义；因延迟导入而未暴露为环。**P1**（最高结构债务）。
2. **`transfer → research`（非内核）7 条**：`transfer/binary.py:19→research.comparison`、`transfer/mask_probe.py:18→research.evaluation`、`:19→research.source_selection`、`transfer/target_experiments.py:291/486→research.source_matrix`、`:421→research.comparison`、`:422→research.evaluation`。数值核心反过来依赖编排/研究层。**P2**。
3. **`baselines ↔ train`**：`train/factory.py:11-14` eager 导入 `baselines.{classic,deep,n2p3net,riemann}`；反向 `baselines/deep.py:46-60`、`baselines/evaluate.py:41`、`baselines/n2p3net.py:27` 导入 `train.{device,early_stopping,runtime,source_sampling}`。注册表与运行时互相依赖，`baselines` 无法独立安装/测试。**P2**。
4. **`transfer ↔ baselines`**：transfer→baselines 8（`transfer/binary.py:17`、`calibrated.py:22-24`、`calibration_candidates.py:15-16`、`recipes.py:12`、`subject_adapter.py:20`），baselines→transfer 3（`deep.py:67/71/81`）。两侧都"合理"，说明缺少共享协议模块（模型/特征协议应下沉）。**P2**。
5. `stats/transfer.py:13 → research.comparison`：统计层依赖研究层，**P2**。

不计违规：`data → research.contracts/execution`（10 eager，14+8 fan-in，属豁免内核）；`models → data`（单向，符合预期）。

## S6 测试与源码重复：比例健康，重复是"测试内自造轮子"

- 体量：tests 34,519 SLOC = src 的 **70.7%**（185 文件，中位 150–400 行区间）。
- 测试↔源码块级重复仅 2 段/18 行，最大者 `src/data/events.py:621 ↔ tests/data/test_domain.py:218`：测试复制了生产的事件时间线逻辑而非调用它——断言的是副本，contract 改动后测试可能仍绿。**P2**。
- 测试↔测试重复 3 段（`test_source_execution_resources.py:145`、`test_structural_adaptation.py`、`test_bnci2014_008_candidate.py`，各 8–11 行）：未复用 `tests/support/`，与 `doc/technical_structure.zh.md:412`（"共享造数归 support/，测试文件不互相导入"）的约定一致但执行不完全。**P3**。
- 105 个"仅测试引用"的生产符号（3,922 行）说明测试在保护一批生产未用的 API；其中 `transfer/exact_covariance.py:134 ExactPropagatedCovariance`(307)、`data/domain_alignment.py:264 AcwState`(291)、`data/ea_artifact.py:233 EAFeatureArtifact`(150) 最重，应确认是"待接线能力"还是"应下线"。

## S7 超长模块/函数（补充统计）

模块规模分布（374 文件）：<50 SLOC 47 个、50–150 119 个、150–400 133 个、400–800 47 个、800–1200 15 个、>1200 **13 个**。>1200 的 src 文件：

| SLOC | 文件 | 备注 |
|---:|---|---|
| 2,686 | src/baselines/deep.py | 含 1,328 行方法，全仓最大热点 |
| 1,881 | src/data/moabb.py | 两个 300+ 行函数 |
| 1,703 | src/transfer/subject_adapter.py | `fit` 632 行 |
| 1,660 | src/data/epochs.py | — |
| 1,509 | src/baselines/evaluate.py | 含近重复对 |
| 1,478 | src/data/raw_artifacts.py | — |
| 1,319 | src/transfer/decision_loss.py | — |
| 1,187 | src/models/n2p3net.py | `__init__` 228 行 |
| 1,161 | src/transfer/calibrated.py | — |
| 1,094 | src/train/runtime.py | — |
| 929 | src/data/preprocess.py | — |
| 921 | src/transfer/checkpoint.py | — |

函数长度（顶层）：829 个顶层函数中 28 个 >150 行、61 个 >100 行；719 个方法中 9 个 >150 行、2 个 >300 行。最长 `data/brainsync.py:176 _build_validated_brainsync_session` 456、`transfer/target_experiments.py:412 execute_target_tasks` 353、`data/moabb.py:1358 _prepare_causal_subjects` 347、`:1707 prepare_moabb_p300` 312、`transfer/within_subject.py:86 calibration_decision_split` 263。方法：`baselines/deep.py:1438 DeepBaseline._fit_attempt` **1,328**、`transfer/subject_adapter.py:756 SubjectAdapter.fit` **632**、`deep.py:896 DeepBaseline.fit` 272、`deep.py:1169 _fit_fixed_updates` 268。每包 >100 行顶层函数数：data 27、transfer 24、baselines 3、models 2、train 2、stats 2、research 1；中位 span 18–42 行，说明"少数巨函数 + 大量小函数"的双峰结构。experiments 顶层 `main` 有 1,192/1,183/770/602 行者。

## S8 重构优先级（收益/风险/验证）

| 优先级 | 动作 | 依据（文件:行） | 收益 | 验证 |
|---|---|---|---|---|
| **P1-1** | 把 `data.source_records` 需要的 `transfer.binary` 标签/正负构造下沉到 `data`（或新 `shared`），消除 7 条 deferred 反向边 | `data/source_records.py:46,56,343,551,653,804`；`data/study.py:230` | 数据层可独立导入；拆掉 data↔transfer 包级环的一半 | `.venv` 中 `PYTHONPATH=src python -c "import data.source_records"` 且 `grep -c "from transfer" src/data/*.py == 0`；`pytest tests/data -q`（47 文件） |
| **P1-2** | 拆分 `DeepBaseline._fit_attempt`（1,328 行）为 zero-shot / fixed-budget / time-split 三个阶段方法，`deep.py` 按模型族分文件 | `baselines/deep.py:1438`、`:896`、`:1169` | 全仓最大可维护性瓶颈；方法级测试可定位 | `pytest tests/models/test_deep.py tests/train/test_global_early_stopping.py -q`（`test_deep.py:18` 已直接导入该类，可作特征化基线） |
| **P2-1** | 合并 3 对近重复函数（artifact/domain/evaluate） | `artifact.py:682`+`:735`；`domain.py:43`+`:126`；`evaluate.py:794`+`:911` | 去重 ~190 行，消除 test-exclusion 分支漂移 | 先为两分支各加一条差异断言，再合并；`pytest tests/data/test_artifact.py tests/data/test_domain.py tests/models/test_domain_adapters.py` |
| **P2-2** | 抽取 experiments 的 `SubjectAdapterConfig`/训练循环装配为 src 内共享 builder | `run_candidate_cross_decision.py:1083-1097` vs `run_within_subject_transfer.py:550-563`（已分叉） | 消除 95 行重复且阻止配置继续分叉 | 两脚本同 seed 冒烟对比配置 JSON 相等；`pytest tests/train -q` |
| **P2-3** | 把 `research.contracts`/`research.execution` 提升为显式最低层包（如 `shared/contracts`），`research` 只留编排 | doc `technical_structure.zh.md:272`；30+ 处 import 站点（research.contracts fan-in 26、research.execution 8） | 打破 7 包单一 SCC；使 data/transfer→research 的"违规"消失 | 全量 import 冒烟 + `pytest tests -q`；机械改名可用 `git grep -l "research.contracts"` 逐一替换 |
| **P2-4** | 删除 7 个真死函数；88 个"公开但仅自用"符号改私有或进 `__all__` 说明 | S3 表；如 `data/bi2014a_schedule.py:98 BIFlashScheduleContract`（在 `__all__` 内但无外部引用） | 收窄 API 面 1,856 行 | `ruff check`（F401）+ `pytest tests -q` |
| **P3** | 删除空包 `src/measurement`、`src/audit_p300`；加"无 eager 模块环"守卫测试 | `git ls-files` 为空 | 防回归 | 复用 `g6_ast.py` 的 SCC 段作为测试断言 |

**不要做**：把 24 个"无 src 消费者"模块当死代码删除（`data/moabb.py` 1,881 行等由实验入口/测试使用，属研究能力）；把 experiments 的重复当纯复制合并（两处已语义分叉，需先对齐语义）。

## S9 残余风险与限制

1. 分层判定按导入方向多数派推断，仓库无成文分层规则；若维护者认为 `baselines` 本就高于 `transfer`，则 S5-4 应降为"共享协议缺失"而非违规。
2. 动态引用只覆盖字符串常量分词（命中 0），`importlib`/`getattr` 注册表式引用可能漏计；删除死代码前仍需 `grep` 复核。
3. 近重复筛查限同包顶层函数（span 25–1200），跨包近重复与 >1,200 行函数未纳入；块级克隆要求 8 行完全一致，语义克隆（改一两行）不计入，故 0.21% 是**下界**。
4. 默认 `python` 为 3.14 且无 `mne`，AST 统计不受影响；导入验证与后续 pytest 必须用 `.venv`。
5. `experiments/runs`（8,134 文件）与 `formal/`、`frozen/` 不在本分区统计内；历史运行副本中的重复不计入上表。

## S10 复现命令

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python tmp/audit/scratch/g6_ast.py    # 语料/edges/SCC/长函数/分层违规
python tmp/audit/scratch/g6_ast2.py   # 块克隆 + 引用分类
python tmp/audit/scratch/g6_ast3.py   # 死码分类 + eager 环 + 包边矩阵
python tmp/audit/scratch/g6_ast4.py   # 近重复 + 长函数 + 跨层边
python tmp/audit/scratch/g6_ast5.py   # 包级 SCC + 重复聚合 + 叶子模块
$env:PYTHONPATH='src'; .venv/Scripts/python.exe -c "import data, transfer, baselines, models, research, stats, train; print('IMPORT_OK')"
```

预期基线：eager SCC = 0；IMPORT_OK；块克隆 19 段/221 行；真死 7 项；包级 SCC 1 个（7 包）。统计脚本只读，仅写 `tmp/audit/scratch/*.json`。
