# 审计 01 — src/data 数据契约、缓存准入与信号处理

- **分区**：`src/data/` 全部 45 个 Python 模块（含 `__init__.py`）
- **基线**：当前工作树（非 HEAD），只读审计，未修改任何源文件
- **方法**：grep 定位 → read 精读关键函数；`ruff check src/data`；AST 静态扫描（可变默认参数/裸 except/超长函数/重名）；`PYTHONDONTWRITEBYTECODE=1` 下的定点实验（MNE epoch 网格、MOABB 1.6.1 滤波器实现）
- **分级**：P0 正确性/数据完整性/安全/可复现性阻断；P1 显著风险或高代价；P2 可维护性

---

## 结论摘要

1. **P1** 缓存 attested 加载每次把整份 NPZ 复制到临时文件并重算 SHA-256，还要求 105% 空闲空间（`epochs.py:1194-1255, 1073-1100`）——在 42 GB 结果 + 4 GPU worker 规模下是直接吞吐瓶颈。
2. **P1** MOABB 高层入口（`moabb.py:1714-2000`）不产生 `preprocessing_execution` 回执，provenance 里的 filter 字段只是把**声明值**抄一遍；实际滤波由 MOABB→MNE 默认 `raw.filter(method="iir")` 执行（已核对 venv 内 moabb 1.6.1 源码），MNE 升级即可静默改变执行算子。
3. **P1（疑似）** BrainSync/BIDS 会话准入只做元数据交叉校验，**没有文件内容摘要**（`bids_eeg.py:228+`, `brainsync_contract.py:359`）；个人校准证据无法证明“读到的就是采集时的那份字节”。
4. **P2** 声明的 `[-200, 1200) ms` 与 MNE 实际 epoch 网格存在最多半采样（128 Hz 时 -3.125 ms）偏移，尾部被裁到 179 点（`preprocess.py:926-939, 626-627`）。
5. **P2** `save_epoch_dataset` 使用固定名 `.tmp.npz` 且 replace 前不 fsync（`epochs.py:1037-1039`），并发写同一路径会互相覆盖。
6. **P2** `PreparedSourceInventory` 的“已验证”指纹只含 `(dev,ino,size,mtime_ns)`（`source_admission.py:102-107`），回拨 mtime 可跳过重验。
7. **P2** GTN 构建的 `except Exception` + `allow_skipped`（`gtn_dataset.py:199-213`）会把程序错误记成“源缺陷”，队列可能静默缩小且不留 traceback。
8. **P2** 手写 preprocessing JSON 省略 `zero_phase_edge_policy` 会静默回落 `legacy`，切换到不同的零相位补边后端（`epochs.py:213-249`, `preprocess.py:759-764`），校验不报错。
9. **正面** 缓存准入链整体 fail-closed（哈希/size/sidecar/解码内容四重校验）；baseline 半开窗口正确；`epoch_operator_gain` 确为诱导 ∞-范数；ruff 全通过，src/data 无裸 except、无可变默认参数。

---

## 发现

### P1-1 缓存 attested 加载的 O(size) 复制/哈希放大与空间门槛
- **证据**：`src/data/epochs.py:1215-1255` 每次 `validation="attested"` 都用 `tempfile.TemporaryFile(dir=snapshot_directory)` 全量复制 NPZ 并重算 SHA-256 后与 sidecar 比对；`_verified_cache_snapshot_directory`（`epochs.py:1073-1100`）要求 `byte_size*1.05 + 64 MiB` 空闲，不足直接 ValueError；`verify_prepared_cache`（`source_records.py:72-88`）在 inventory 路径上还会再算一次文件 SHA。
- **影响**：42 GB `experiments/runs` + 大量 source cache；按 4 个 GPU worker、每个 worker 多次加载，读放大与临时空间近似翻倍，与“以实测吞吐优先”的目标直接冲突。
- **建议**：改为内容寻址只读对象（按 sha 命名、写一次）；同进程缓存 `(dev,ino,size,mtime_ns,sha)` 验证结果；5% 预留可配置；或提供显式的 `attested-fast` 模式并记录该模式。
- **验证**：对同一 NPZ 连续两次 attested 加载，用 Process I/O 计数器确认读字节 ≈ 2× 文件大小；磁盘填到 <105% 后复现 `insufficient free space`。

### P1-2 MOABB 高层入口“声明即证据”，执行算子不受回执保护
- **证据**：`moabb.py:1783-1791` 构造 `P300(fmin=l_freq, fmax=h_freq, tmin, tmax, baseline=None, resample=None)` 后由 `paradigm.get_data` 完成滤波；`moabb.py:1883` 用 `epochs.resample(...)`；`moabb.py:1901-1910` 再裁剪并 `apply_trial_baseline`。provenance 的 `"filter": {"method","order","phase"}`（`moabb.py:1985-1989`）与 `"resample": {...}`（1991-1996）全部来自声明对象；本文件没有 `execute_preprocessing`/`preprocessing_execution`（全仓 grep 仅 bi2014a/bnci2014_008/gtn_dataset/dataset.py 调用）。
- **已核对实际算子**：venv 内 moabb 1.6.1 的 `get_filter_pipeline` 为 `methodcaller("filter", l_freq=fmin, h_freq=fmax, method="iir", picks="data")`（`.venv/Lib/site-packages/moabb/datasets/preprocessing.py`）。当前默认恰好是 MNE 四阶 Butterworth 零相位，与声明一致——但这是版本巧合，不是被记录的事实；且 `picks="data"` 会把 EOG 一起滤波，而原生路径按通道/EOG 依赖分别构造滤波段（`preprocess.py:735-742`）。
- **影响**：MOABB 来源缓存无法与“不同 MNE/MOABB 版本、不同默认 iir 参数”的缓存区分；原生路径 receipt 则记录 `sos`、`padding_limit`、`source_context`。跨版本复现时 MOABB 缓存会“看起来合规”。
- **建议**：MOABB 入口也生成 `n2p3_preprocessing_execution/1` 回执（实际样本率、MNE 版本、iir 参数、滤波前后样本、重采样函数与输出长度、每通道滤波段），provenance 区分 declared 与 observed，并对 moabb 版本白名单。
- **验证**：用旧版 MNE 生成一个 MOABB 缓存，再用当前 env 读取，比较 `provenance["filter"]` 与实测冲激响应；当前代码不会报错。

### P1-3（疑似）BrainSync/BIDS 会话准入无内容摘要
- **证据**：`src/data/bids_eeg.py:228-330` 只校验 BIDSVersion/TaskName/SamplingFrequency/RecordingDuration/EEGReference/channels.tsv/events.tsv，全文 grep 无 `sha256/md5/digest/st_size`；`brainsync_contract.py:359` 只要求 `recording.bids` 是 file manifest，随后用 ts/样本/时长交叉校验（`brainsync_contract.py:422-472`）。`brainsync.py` 的 `semantic_sha256` 只用于 QC/ocular 模型摘要（560/583），不覆盖 EEG 文件。
- **影响**：个人校准原始 `.bdf` 被替换为同样头信息、同样时长、同样事件表的另一份记录时准入会通过；个人推断的可复现性绑定弱于公共源域（后者有逐文件 SHA-256 + MD5）。
- **建议**：在 session manifest 的 `recording.bids` 中为每个文件记录 `(sha256, byte_size)` 并流式校验，或复用 `raw_artifacts.VerifiedSnapshot`。
- **验证**：构造两份元数据完全相同、仅二进制若干采样不同的 BIDS 目录，`validate_brainsync_bids_session` 当前都会通过。
- **存疑**：独立仓库的采集 app 可能另行记录哈希；若已有，应把它纳入契约层。

### P2-1 声明时间轴与实际 epoch 网格的半采样偏移
- **证据**：`preprocess.py:926-936` 用 `mne.Epochs(tmin=-0.2, tmax=1.2)`；实测（mne 1.12.1）128 Hz 得 181 点、`times[0]=-0.203125`；512 Hz 得 717 点、`-0.19921875`；250/100 Hz 恰为 -0.2。`preprocess.py:626-627` 不重采样时直接 `data[..., :n_times]`（179），保留首样本、砍掉尾部两点 → 128 Hz 实际覆盖 `[-203.1, 1187.5]` ms，而契约 `n_times=floor(1.4*128)=179` 对应 `[-200, 1190.6]` ms。
- **影响**：跨数据集（不同源采样率）P300 潜伏期比较最差相差 ~3.9 ms；单数据集内是固定偏移，不影响分类，但影响以 ms 报告的潜伏期与跨设备对齐。`epoch_support_record()`（`contract.py:147-157`）只导出声明 tmin，采集 app 与下游都拿不到实际原点。
- **缓解证据**：receipt 记录了 `actual_source_tmin_s = start/source_rate`（`preprocess.py:783-786`）；MOABB 路径另做了 `epochs.times[0]` 的 0.5 采样校验（`moabb.py:1874-1879`），原生路径无对应断言。
- **建议**：provenance 增加 `grid_origin_ms`、`last_sample_ms`；或在 epoch 前把事件样本按 `round(tmin*sf)` 对齐后再以声明 tmin 建轴。
- **验证**：用 `sin(2πft)` 合成信号测 10 Hz 相位，128 Hz 路径应比 250 Hz 路径早 3.1 ms。

### P2-2 save_epoch_dataset 的固定临时名与缺 fsync
- **证据**：`epochs.py:1037-1039` `temporary = path.with_suffix(path.suffix + ".tmp.npz")` → writer → `temporary.replace(path)`；随后才写 `*.record.json`（1040）。
- **影响**：两个进程写同一缓存会互相覆盖（shard 按 subject 命名，冲突概率低但存在）；崩溃留下 `.tmp.npz` 垃圾；replace 前无 fsync，掉电可能留下未落盘 NPZ（record 未写 → 加载 fail-closed，可恢复但需人工清理）。
- **建议**：`tempfile.mkstemp(dir=path.parent)` + `os.fsync` + `os.replace`，启动时清理陈旧 `.tmp.npz`。

### P2-3 库存指纹只覆盖文件系统元数据
- **证据**：`source_admission.py:102-107` 指纹 = (ledger sha, (dev,ino,size,mtime_ns)×(cache, record))，命中则跳过 `verify_source_preparation`。
- **影响**：同尺寸改写 + `os.utime` 回拨可跳过台账级重验；最终 attested 加载仍重算哈希，故不构成完整性绕过，但故障发现从准备阶段推迟到训练启动阶段，42 GB 规模下代价高。
- **建议**：指纹加入内容哈希或 `ctime_ns`；或让 `inspect()` 返回是否命中缓存，供上游决定强制重验。

### P2-4 GTN 构建的宽泛捕获与 allow_skipped
- **证据**：`gtn_dataset.py:199-213` 每个 experiment 目录 `try: _preprocess_gtn_experiment(...) except Exception as exc: ... skipped.append(message)`，仅 `allow_skipped` 为真时继续；`skipped` 只存 `type(exc).__name__: str(exc)`，无 traceback、无阶段标记。`dataset.py:87` 的 `except Exception` 同理（但只做类型转换并保留 `from exc`）。
- **影响**：TypeError/AttributeError/KeyError（代码缺陷）与“缺 thought metadata/重复身份”（源缺陷）无法区分；`allow_skipped=True` 时队列静默缩小，下游只看到 `n_subjects` 变小。
- **建议**：捕获收窄到 `(ValueError, OSError, KeyError, FileNotFoundError)`；记录 traceback 与失败阶段；把 skipped 计数纳入 acceptance gate。
- **验证**：放入一个格式正确但字段缺失的 `.nix`，`allow_skipped=True` 时该 subject 消失而仅留字符串记录。

### P2-5 历史兼容字段的“静默 legacy”
- **证据**：`PreprocessingSpec.record()`（`epochs.py:213-227`）在 policy 为 `legacy` 时删除该字段；`parse_preprocessing_spec`（`epochs.py:230-249`）的必填集合来自 `PreprocessingSpec().record()`，因此新声明也可省略并静默得到 legacy。后端选择依赖它：`preprocess.py:759-764` 中 `mne_reflect_limited` 与非 notch 的 legacy 走 MNE reflect 补边，否则走 `scipy.sosfiltfilt_odd`，数值不同。
- **影响**：由 `PreprocessingSpec` 生成的 recipe 安全；手写/外部 JSON 漏字段会静默改变数值算子，且 `assert_p300_input_contract` 也会以 legacy 期望值通过。
- **建议**：声明加 `schema_version`，新入口要求显式写出该字段；历史缓存按版本走旧默认。
- **验证**：删除 bidirectional recipe 中的 `zero_phase_edge_policy`，解析不报错且执行改走 scipy odd 后端。

### P2-6 死代码与状态混淆
- **证据**：`preprocess.py:960` `dropped = np.array([], dtype=np.int64)`，1015 行原样放入 `PreprocessResult(dropped=...)`，阅读范围内未见写入；`preprocess.py:950-957` 只用 drop_log 生成状态，被注释/坏段丢弃的 epoch 统一标 `acquisition_rejected`（只有 NO_DATA/TOO_SHORT 才是 `boundary_dropped`）。
- **影响**：`dropped` 会让调用方误以为有被拒 epoch 列表；`acquisition_rejected` 把“MNE 注释拒绝”和“采集端拒绝”混为一谈。
- **建议**：删除该字段或填入真实索引；为 `reject_by_annotation` 丢弃增加独立状态。
- **存疑**：`mne.Epochs` 默认 `reject_by_annotation=True`，`preprocess.py:926-936` 未显式传入也未记录；公共源域 raw 一般无 BAD 注释，触发概率低。

### P2-7 超长函数与重复实现
- **证据**（AST 统计）：`brainsync.py:176 _build_validated_brainsync_session` 456 行；`moabb.py:1358 _prepare_causal_subjects` 347 行、`moabb.py:1707 prepare_moabb_p300` 312 行；`epochs.py:1303 _load_epoch_dataset` 245 行；`bids_eeg.py:228 validate_bids_eeg_recording` 233 行；`preprocess.py:801 preprocess` 228 行；共 40 个 ≥90 行函数。候选/日程解码分别落在 `bi2014a_candidate.py`（159 行）、`bnci2014_008_candidate.py`（109 行 `_decode_schedule`）、`candidate_task.py` 三处。
- **影响**：300-450 行函数难以单测与 review；候选逻辑三处实现容易规则漂移。
- **建议**：按“校验/装配/落盘”拆分；把候选/日程规则抽成共享模块并做性质测试（同一 flash 序列 → 同一 candidate 映射）。

### 正面结论（已证实，无需整改）
- **缓存准入 fail-closed**：`epochs.py:1194-1255` 复制后重算 SHA，前后 `fstat` 身份比对；`_validate_attested_cache`（1281-1300）逐字段比对 sidecar 与解码结果；`source_records.py:72-88` 同时校验 NPZ 与 record；`raw_artifacts.py:277-336` 校验尺寸/SHA/MD5 并在复制前后比对 inode。
- **baseline 半开正确**：`preprocess.py:589-590` 用 `ceil(x-1e-9)` 实现右开，0 ms 样本被排除；591-594 拒绝越界/过短窗口；重采样后按 128 Hz 施加（604-632）。
- **算子范数正确**：`epoch_linear_projection` 返回 `(source_samples, n_times)`（`recovery.py:67-76`），`epoch_operator_gain` 的 `|P|.sum(axis=0).max()` 恰为 `A=P^T` 的诱导 ∞-范数（`continuous_qc.py:92-95`），与 docstring 一致。
- **无低级缺陷**：`ruff check src/data` 全通过；AST 扫描无裸 except、无可变默认参数、无硬编码绝对路径；`source_records.py:130-143` 把未声明 companion 以 `identity_basis="preparation_start"` 冻结，resume 可比对。
- **单位一致**：V↔µV 只在显式解析入口转换（`source_parsers.py:148,195`、`bi2014a_candidate.py:284`、`bnci2014_008_candidate.py:533`），并强制 `signal_unit="V"`（`epochs.py:126-127`）；QC 阈值以 V 表示（`continuous_qc.py:134`）。

---

## 未能验证与存疑

1. **MOABB 缓存的真实执行算子**：旧缓存没有回执，无法从数据反推当时的 MNE/MOABB 版本与 iir 参数；只能确认当前版本“恰好”等价。
2. **`Epochs.resample` 与 `mne.filter.resample` 的数值差异**：两条路径都“先重采样再裁 179 点”，但未逐样本比对（pad/npad 语义可能不同）；标为**疑似**，建议同一 raw 走两条路径做 `allclose`。
3. **BrainSync 采集 app 是否另行记录哈希**：仓库外项目，无法检查。
4. **bidirectional/recovery 的边界包络数学**：只结构性核对了 ∞-范数与 `epoch_error_bound` 的维度语义，未重新推导 `fit_bidirectional_state_recovery` 的 warmup/probe 覆盖；疑似点是校准段先减首样本（`bidirectional.py:218`）而生产滤波不减常数，可能使界偏乐观（有 `inflation` 兜底）。
5. **未深读**：`events.py`（609 行）、`manifest.py`、`domain.py`/`domain_alignment.py`、`artifact.py`、`identity.py`/`lineage.py`、`moabb.py` 其余 ~1300 行。受 20-35 分钟窗口限制，请 Lead 决定是否追加。
6. **未运行**：任何训练/全量 pytest；只做了 ruff 与两个 python 定点实验，测试重组后的覆盖度未评估。

---

## 已检查范围

**精读**：`contract.py`(全)、`source_admission.py`(全)、`finite_filter_operator.py`(全)、`source_parsers.py`(全)、`euclidean_alignment.py`(全)、`harmonic.py`(全)、`bidirectional.py`(全)、`ocular.py`(全)、`preprocess.py`(540-1028)、`epochs.py`(30-520, 930-1330, 1044-1260)、`source_records.py`(55-185)、`raw_artifacts.py`(220-410)、`continuous_qc.py`(30-165, 552-602)、`recovery.py`(20-150)、`dataset.py`(55-165)、`gtn_dataset.py`(150-245)、`moabb.py`(1780-2003)、`bids_eeg.py`(228-298)、`bi2014a_candidate.py`(262-311)、`brainsync_contract.py`(grep 全量索引)。

**命令**：`ruff check src/data`（All checks passed）；AST 扫描脚本 `tmp/audit/scratch/data_stats.py`；MNE epoch 网格实验（128/250/512/100 Hz）；MOABB 1.6.1 `get_filter_pipeline` 源码核对。

**未写任何源文件**；仅写本报告与 `tmp/audit/scratch/data_stats.py`。

---

## 附：模块覆盖速览

| 模块 | 结论 | 依据 |
|---|---|---|
| contract.py | 契约字段完整；n_times=floor 正确；导出缺 grid origin（P2-1） | 全读 |
| epochs.py | 校验与准入强；attested 加载放大（P1-1）、save 原子性（P2-2）、legacy 缺省（P2-5） | 关键段精读 |
| preprocess.py | 半开 baseline 正确；网格取整未断言（P2-1）；dropped 空字段（P2-6） | 540-1028 |
| events.py | 未深读；仅确认 3 个公共 API | grep |
| manifest.py | 未深读 | — |
| brainsync_contract.py | manifest/BIDS 交叉校验细；无内容摘要（P1-3 疑似） | grep + 关键行 |
| bids_eeg.py | 仅元数据校验，无哈希（P1-3） | 228-330 |
| brainsync.py | 未深读；QC/ocular 模型有摘要 | grep |
| raw_artifacts.py | 物理文件认证最严格的一段（正面） | 220-410 |
| source_records.py | 台账/companion 冻结良好（正面） | 55-185 |
| source_admission.py | 指纹仅元数据（P2-3） | 全读 |
| recovery.py | 投影矩阵与范数语义正确（正面） | 20-150 |
| finite_filter_operator.py | SOS 稳定性/归一化校验完备（正面） | 全读 |
| bidirectional.py | 闭区间支持保守；校准减常数疑点（存疑 4） | 全读 |
| continuous_qc.py | 阈值单位 V 正确；差异/MAD 估计合理 | 30-165, 552-602 |
| ocular.py | 拟合/应用中心化约定有 DC 差，但被高通/基线消除 | 全读 |
| harmonic.py | WOLA 归一化正确；ledger 规模需关注 | 全读 |
| euclidean_alignment.py | 相对特征值下限 + trace 归一化自洽；输出 dimensionless | 全读 |
| candidate_task.py / bi2014a_candidate.py / bnci2014_008_candidate.py | 候选规则三处实现（P2-7）；主流程共用 execute_preprocessing | 局部 |
| dataset.py | 通用 MNE 适配器；异常类型统一（P2-4） | 55-165 |
| gtn_dataset.py | allow_skipped 宽泛捕获（P2-4） | 150-245 |
| moabb.py | 声明即证据（P1-2）；其余 1300 行未读 | 1780-2003 |
| domain*.py / artifact.py / identity.py / lineage.py / channel.py / study.py | 未审计 | — |

*（本表用于让 Lead 快速分配追加审计，不替代上文的证据与分级。）*
