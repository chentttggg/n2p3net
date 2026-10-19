# tests/data 整理与覆盖对应

2026-09-16。按 `codebase-design` 的同一 interface 集中验证原则及工程指南执行。
整理前源码和全部测试已物理冻结于
`frozen/unified_compact_training_20260916_before/source.manifest.json`。
本模块只修改 `tests/data/`及本记录；生产源码、共享 support、其他测试目录均未改动。

## 文件合并

| 删除的活动文件 | 现 owner | 保留或替换的验证 |
|---|---|---|
| `test_timeline_group_validation.py` | `test_events.py` | 原交错排程的顺序不变、五种身份混组拒绝、时间倒序、连续原 repetition 与真缺口反例全部保留；合并与普通 timeline 重复的混 run 拒绝断言。 |
| `test_epoch_row_identity.py` | `test_events.py` | 被拒事件的 evidence 映射不改变 retained row 身份；共同支持集保留完整 scheduled 分母、原事件 ID 和 `-1`。 |
| `test_gtn_dataset.py` | `test_gtn.py` | 独立右端点数学检查保留。四 cohort 的静态转换比较、单 causal receipt 成功测试、两个无滤波 `_prepare_gtn_experiment` 成功测试，收敛到四 profile 的真实公开 builder 执行，同时验证独立物理参数预期、receipt、原事件样本、九选 labels/candidates/repetitions 和 causal 标志。 |
| `test_ocular_strength_pipeline.py` | `test_ocular.py` | 独立 EOG、已知神经信号与相关 EOG 的衰减反例原样保留；流水线改为调用已有完整 `PreprocessingSpec` interface，移除手工复制其十余个参数的低层调用。 |
| `test_qc_features.py` | `test_artifact.py`、`test_epochs.py` | 全局幅度与相对 PTP 的数学不变性保留于使用这些特征的 artifact 测试；特征保存重载并入现有 trial-channel mask 保存重载。 |

五个旧文件已实际删除；没有保留重定向入口，测试文件之间没有互导。
现有 `tests/support/datasets.py`、`signal.py`等继续复用。

## 删除重复执行的依据

| 原重复入口 | 覆盖 owner 和具体依据 | 执行变化 |
|---|---|---|
| `test_preprocess_n_times_alignment` | 默认179点仍由 `test_preprocess_shape_events_causality_and_copy`覆盖；不截断的180点由 `test_preprocessing_execution.test_low_level_none_sample_count_preserves_full_resampled_epoch`对独立 MNE epoch/resample 的完整数值参照覆盖。 | 删除两次只检查 shape 的预处理。 |
| `test_build_subject_uses_native_layout_without_padding` | `test_build_subject_preserves_shape_labels_metadata_and_times`的第二个参数案例直接走三导原生 `channels=None`，同时检查原生导顺序、实测 mask、标签、嵌入与时间；显式缺导拒绝另保留。 | 三次成功构建减少到两次。 |
| `test_epoch_cache_round_trip_persists_qc_features` + 原 trial-mask roundtrip | `test_epoch_dataset_trial_masks_and_qc_features_round_trip`一次保存带两处真实缺导 mask 的随机信号，同时检查 mask、相对 PTP、STD、全局尺度及 observed mask。 | 两次缓存写入/加载合并为一次。 |
| prefix 五类损坏反例的重复准备 | 模块范围的 `prepared_prefixes`只生成一次不可变原始 scope 缓存；四种 metadata 攻击各有自己的深拷贝 JSON，缺验证记录案例复制 NPZ 至自己的临时目录，不破坏共享产物。 | 五次相同 raw 解析、逐 scope 滤波和缓存写入减少到一次；五类拒绝案例仍分别收集执行。 |
| 单独 K0 成功案例 | 主体选择加载反例本来就准备 K0；该测试现同时检查 calibration 为 `{0: None}`、全部20个测试事件保留，再检查其他主体未加载与未知主体拒绝。底层 K0 范围反例继续保留。 | 少一次 K0 raw 处理与缓存写入。 |
| EOG 应用、持久化、强度、单位和错误参考场景重复 OLS 校准 | 同模块 `fitted_ocular`一次拟合；每次 apply 用独立 raw，强度变化使用不可变 `replace`，fixture 收尾确认原模型记录未变。改变 calibration 波形/掩码的拟合反例仍各自重新拟合。 | 五次相同校准减少到一次。 |

GTN 四种 profile 现在全部实际执行，增加了此前仅检查转换声明的数值覆盖。
没有为了缩小案例数而删减其 cohort 范围。

## 明确保留

原始字节/sidecar 身份、缓存篡改/替换、source/target fit 隔离、crop-before-filter、绝对样本与闭端支持、
一采样点边界、原 repetition 缺口、滤波非交换性/因果性、QC 数学量及失败分母继续存在。
独立公开数据读取和缺失元数据拒绝继续执行；已冻结模型的历史合同门禁有现实使用方，也继续保留。
未按文件日期、`legacy`命名或全套已通过来删测试。

## 验证和规模

与物理快照逐文件比较，`tests/data/`由 **46 个 Python 文件降至41个**（均含 `__init__.py`），
总行数 **10077 → 9962，净减115行**，包括全部辅助代码；测试函数 **441 → 436**。
修改范围为8个保留文件加5个删除文件，没有其他 data 文件内容变更。

相同职责集合的定向检查：整理前 **169 passed，21.78s**；整理后 **162 passed，14.76s**。
这是单次本机观测，不作为稳定吞吐基准。两次都实际读取已有 GTN，保留相同两条“多余 txt 已明确匹配”警告。
之后补齐 GTN 独立物理参数预期，4个相关 cohort 案例再次通过；prefix 精确一采样点/一行计数及其余损坏案例5项通过。
Ruff 检查整个 `tests/data/`通过；活动六指南没有遗留已删除测试路径，目录内没有测试文件互导。
没有运行全套，整体回归由根任务统一安排。
