# 审计 G1 — src/data 元数据/事件/域/工件层缺口

- **分区**：G1（补充分区，只读审计）
- **范围**：`src/data` 未深读模块 events.py、manifest.py、identity.py、lineage.py、domain.py、domain_alignment.py、artifact.py、artifact_sidecar.py、channel.py、study.py、metadata.py、exclusions.py、qc_features.py、slow_trend.py、personal_quality.py、bids_eeg.py、recording_clock.py、bi2014a_schedule.py，以及 `src/baselines/{calibration,progress,validation}.py`
- **基线**：工作树非 HEAD（429 项未提交；main = f1a303c）；审计期间不修改任何源文件
- **分级**：P0 数据/身份/契约正确性阻断；P1 显著风险或高代价；P2 可维护性
- **方法**：read 精读 + grep 调用点交叉；`ruff check`；PYTHONDONTWRITEBYTECODE=1 的定点 python -c；与 doc/input_contract_math.zh.md、doc/technical_structure.zh.md 声明对照

## 结论摘要

1. **P1 共 5 条**，其中两条已用定点实验复现：(a) `manifest` 把 `channels` 字符串逐字符拆成通道（`manifest.py:179`，实测 `"Fz"→('F','z')`）；(b) 同一 `dataset_id` 的多会话事件台账无法拼接（`events.py:585` vs `events.py:321`，实测 `ValueError`）。
2. 另两条 P1 属"身份/审计证据"缺口：`identity.source_or_global` 在训练表缺 `global_person_keys` 时静默放行跨研究同一被试（`identity.py:386-402`）；混合坐标注册把 fiducial+ICP 两条变换压成恒等矩阵写进 provenance（`channel.py:481-489`）。
3. 第五条 P1 是数值口径：年龄正弦编码实际使用 `π·2^k`（最高频周期≈1.6 岁），而通道坐标使用 `geomspace(0.5,8)`；`metadata.py:11-14` 声称"复用同一编码风格"不成立。
4. **P2 共 23 条**，集中在：退化 MAD 的阈值行为（`artifact.py:453-460`、`personal_quality.py:88`）、QC 特征 0.0 哨兵（`qc_features.py:172`）、event/schedule 版本兼容（`events.py:13` 死常量、`recording_clock.py:11-13` 缺省 legacy）、sidecar/进度文件的不可复现写入。
5. 整体质量高于预期：`identity/lineage/artifact/bids_eeg` 的拒绝路径与 digest 往返扎实，`ruff check` 对全部 21 个模块通过；`domain_alignment` 在 `src` 内已无消费者（仅一个 experiment 引用）。



## A. 身份、事件台账与清单契约

### P1-1 manifest 的 channels 接受字符串，被逐字符拆成通道
- **证据**：`manifest.py:179` `channels=(tuple(payload["channels"]) if payload.get("channels") else None)`；`validate()` 只检查元素为非空字符串（`manifest.py:73-77`），未检查容器类型。
- **实测**（`PYTHONDONTWRITEBYTECODE=1`，scratch 清单）：`"channels": "Fz"` → `DatasetManifest.channels == ('F', 'z')`，`load_manifest` 与 `validate` 均通过。
- **影响**：错误布局要到 `resolve_manifest_channels`/`_record_eeg_names` 读原始文件时才可能暴露；若原始记录恰有名为 `F`、`z` 的通道，会静默产出双通道数据集，与声明布局不符。属于"schema 拒绝路径缺口"。
- **建议**：`channels` 必须是 JSON 数组（`isinstance(payload["channels"], list)`），并对每项先 `canonical_channel_name` 再去重。

### P1-2 同一 dataset_id 的多会话事件台账无法拼接
- **证据**：`events.py:585` `event_ids=np.asarray([f"{dataset_id}:{index}" ...])`，event id 不含 session/run/selection；`events.py:321` 要求 `event_ids` 全局唯一。`concatenate_epoch_datasets` 在 `epochs.py:1757` 直接调用 `concatenate_event_timelines`。
- **实测**：两个 `observed_only_timeline(dataset_id='ds', session_ids=['a'|'b'])` 拼接 → `ValueError: event_ids must be globally unique and non-empty.`
- **影响**：同一数据集两次 session（BIDS 多 session 的常规形态）不能合并缓存；调用方若为绕过而伪造不同 `dataset_id`，就把"同一来源"拆成两个 study stream，反过来污染 `study.py` 的来源归组。虽然 fail-closed，但阻塞合法路径且诱导错误绕行。
- **建议**：event id 纳入 session/run/selection（字段已存在于 `events.py:100-102`），或拼接时按 `(dataset,session,run,selection,index)` 重编号。

### P1-3 `source_or_global` 的 global 证据完备性守卫不对称
- **证据**：`identity.py:391-397` 仅在 `policy == "global"` 时要求训练表每条记录都有 `global_person_keys`；`identity.py:386-402` 的 `source_or_global` 分支只用 `target_globals & training_globals` 求交。origin 分支相反：`identity.py:379-380` 对两种策略都要求 origin 完备。
- **影响**：跨研究同一被试（origin 不同、global 相同）在 `source_or_global` 下，若训练记录只登记了 origin（`source_verified`），交集为空 → 静默放行。这正是跨研究泄漏的主要形态，且与同策略下 origin 分支的严格度不一致。
- **建议**：两分支共用同一完备性守卫；证据不全时返回显式 `incomplete_global_evidence` 而不是视为"不同人"。

### P2-1 recording clock 的 Decimal 精确乘法没有采样容差
- **证据**：`recording_clock.py:47-54` `duration * rate != serialized` 为精确 Decimal 比较；同一 manifest 的其余时间校验用 `math.isclose(..., abs_tol=1e-6)`（`brainsync_contract.py:422-427`、`446-449`）。
- **影响**：`duration_seconds` 来自外部 JSON 时，任何四舍五入都会造成 ±<1 采样的假拒绝（例：96.445 s × 128 = 12344.96 ≠ 12345）。口径与相邻校验不一致，且报错文案 "duration disagree" 不指向根因。
- **建议**：允许 ±0.5 采样或 `abs_tol<=1e-6 s`；或在 sidecar 中直接以整数采样数为主键，duration 仅作校验值。

### P2-2 clock 缺省即 legacy，静默绕过全部时钟校验
- **证据**：`recording_clock.py:11-13` `clock is None` → 返回 `legacy_host_only`；`brainsync_contract.py:451` 只在 `clock` 非空时校验 pause 边界与 segment 边界的一致性。
- **影响**：从采集清单中删掉 `clock` 字段即可把 `delivery_anchored` 降级为 host-only 且仍算"通过"，与 data-core 报告 P2-5（legacy 默认）同类但发生在更上游的采集合同层。
- **建议**：采集端版本白名单，或要求显式 `clock_absent_reason` 并让其进入 QC 审计。

### P2-3 `LEGACY_EVENT_TIMELINE_SCHEMAS` 是死常量，v1 台账无迁移路径
- **证据**：`events.py:12-13` 定义 `/1` 集合；全仓 grep 仅命中定义处。`epochs.py:1367` 对缓存内 `event_schema` 只接受 `/2`，否则拒绝。
- **影响**：版本兼容"声明"与实际"直接拒绝"不符；`fingerprint()`（`events.py:514-521`）无条件写入 `/2`，v1 台账被静默重标为 v2。维护者会误以为存在迁移。
- **建议**：删除常量，或实现 `/1→/2` 迁移并在 fingerprint 中保留来源版本。

### P2-4 event 时间线部分变换方法不返回已校验对象
- **证据**：`events.py:452-455` `with_evidence_offset` 与 `events.py:457-486` `_slice` 均不调用 `validate()`；只有 `subset_rows`（450）与 `concatenate_event_timelines`（649）校验。
- **影响**：单独调用 `with_evidence_offset` 可得到 `evidence_indices` 非双射的台账对象；错误推迟到下游 `fingerprint`/写缓存时才出现，定位成本高。
- **建议**：与 `subset_rows` 保持一致，返回前 `validate(n_epochs=...)`。

### P2-5 origin 键解析存在两套实现，容忍度不同
- **证据**：`exclusions.py:37-40` 用 `len(parts)==3 and parts[1]=="participant"`，不匹配则**静默跳过**；`identity.py:58` 用 `startswith("/participant/")`，不匹配则**抛错**。
- **影响**：手写或第三方生成的畸形 origin 键会让 `excluded_identity_subjects` 漏掉被试而不报错，与被排除人群策略（`PARTICIPANT_EXCLUSION_POLICY`）冲突。
- **建议**：抽出单一 `parse_origin_key()`，畸形键一律抛错或计入审计。

### P2-6 study 的 domain_mass 一致性检查可能假拒绝
- **证据**：`study.py:293` 比较 `set(study_ids_for_identity(...)) != set(domain_mass)`；`domain_mass` 的键是 source stream id，而 study id 在合法归组时可与 stream id 不同（`study.py:253-266` 的 proof 分支可以给出不同 study 名）。
- **影响**：只要有一条 stream 通过 proof 归到不同的 study 名，集合比较必然失败 → 合法 checkpoint 被拒。**存疑**：需真实 grouped proof 样本复现，本次未构造。
- **建议**：比较 `{mapping[s] for s in domain_mass}`，而不是集合本身。

## B. 通道几何、域适配、QC 与工件

### P1-4 混合坐标注册把两条变换压成恒等矩阵，并抹掉 ICP 证据
- **证据**：`channel.py:481-489` 在"显式坐标 + montage 兜底"同时发生时构造新的 `CoordinateRegistrationSpec`：`transform_to_head=identity`、`input_frame="mixed"`、`method=A+B`，`fiducials_used` 回落为 `()`，`icp_iterations` 回落为 0。而该 spec 的类文档（`channel.py:37-51`）与 `manifest.py:380-391` 把它当作"可审计刚体注册"写进 provenance。
- **影响**：只要一个数据集里部分通道有显式坐标、部分走 montage，sidecar 就声称发生的是恒等变换，实际取了 fiducial 刚体变换（可能还叠加 ICP）；`method` 里的 "+rigid_icp" 与 `icp_iterations=0` 自相矛盾。下游无法据此复现或审计坐标来源。
- **建议**：要么拒绝混合注册（要求调用方先统一），要么把两条 spec 原样保留为列表并显式标注 composite。

### P1-5 年龄正弦编码与通道编码使用不同频基，注释声明不成立
- **证据**：`metadata.py:11-14`、`metadata.py:101` 声称复用 `channel.sinusoidal_encode_1d` 以"保证两类嵌入的编码风格一致"、并用 D-freq-cap 避免高频退化；但通道身份实际走 `channel.py:566-578` `sinusoidal_embedding`（`geomspace(0.5, 8)`），`sinusoidal_encode_1d`（`channel.py:559-563`，`π·2^k`，k=0..7）只被 `metadata.py` 使用。
- **数值**：年龄归一化 a=age/100，最高频 `π·128≈402 rad`，周期 `2/128=0.0156`（≈1.6 岁）；相邻 1 岁（Δa=0.01）相位差 4 rad，年龄编码在高频段已去相关，"年龄平滑调制潜伏期"的归纳偏置只在最低 2–3 个频段成立。
- **建议**：让两类嵌入共用同一频基，或在 metadata 中显式记录 `age_frequency_hz`/周期并改写 D-age-sin 的声明。

### P2-7 显式坐标映射的重复规范化键静默覆盖
- **证据**：`channel.py:412-415` 用 dict comprehension 建 `by_name`，两个不同的原始名规范化到同一键时后者静默获胜；montage 路径反而在 `channel.py:330-331` 对冲突位置抛错。
- **影响**：同一份配置在"显式坐标"与"montage"两条路径下对重复通道的处置不一致，重复键导致的错位无法发现。
- **建议**：构建时检测重复并抛错，复用 `_canonical_unique` 的语义。

### P2-8 `adapt_observed_montage` 把测量参考差分成恒零后仍标记为已观测
- **证据**：`domain.py:494-497` 先复制 `values[:, destination] = dataset.X[:, index]`，再对全部目标（含 anchor 自身）减 `reference_signal`；`observed[:, destination] = source_mask[:, index]` 仍为 True。
- **影响**：参考通道变成逐点恒零，却参与 `channel_mask`/QC：`artifact` 的 flatline 规则（`artifact.py:556`）每次都把它判为坏通道，bad_fraction 被系统性抬高；下游若把恒零当"真实零幅值"会得到错误协方差。
- **建议**：anchor 列置 `observed=False` 并在 `domain_adapter` 记录 `reference_channel_is_zeroed=True`。

### P2-9 domain 适配器的重复实现
- **证据**：`adapt_native_reference_selection`（`domain.py:126-226`）与 `adapt_common_channel_average_reference`（`domain.py:43-123`）在通道选择、观测掩码检查、`EpochDataset` 构造上约 90% 同构；`_existing_native_selection_matches`（229-278）与 `_existing_common_car_matches`（302-376）是两段近似的幂等校验。
- **影响**：新增一条适配臂时要同步改四处，规则易漂移（例如只在一处加入 `trial_channel_mask` 校验）。
- **建议**：抽出 `_select_channels()` 与 `_verify_idempotent_adapter(schema, reference)`。

### P2-10 global scale gate 的 eps 下限在退化 MAD 下变成过度删除
- **证据**：`artifact.py:453-460` `robust_std = max(1.4826*MAD, eps)`；`artifact.py:463-470` 门限 `|log scale - center| > 6*robust_std`。实测（scratch）4 个 epoch、尺度 [1e-4,1e-4,1e-4,2e-4] → `robust_std=2.22e-16`，2e-4 的 epoch 被判坏；`fit()` 随后可能触发 `artifact.py:724-725` "removed every training epoch"。
- **影响**：单被试/尺度高度同质的训练折里，MAD=0 是常态；eps 兜底把"无法估计离散度"翻译成"离散度为零"，产生物理上无意义的门限。
- **建议**：MAD 为 0 时跳过 global gate 并在 `cv_evidence` 记录 `global_scale_gate_disabled`；或用相对下限（如 `1e-6*|center|`）。

### P2-11 `_choose_quantile` 是无人调用的兼容包装，且口径与正式路径不同
- **证据**：`artifact.py:186-206`，全仓 grep 仅命中定义处（另有 doc 内嵌副本）。它自建 `EpochQCFeatures`，把 `epoch_scale_v` 固定为 1（`artifact.py:203`），与 `compute_epoch_qc_features` 的真实尺度不同；若被重新调用会得到与 `fit` 不同的阈值。
- **影响**：死代码 + 潜在口径陷阱。
- **建议**：删除；如确需单通道 API，改为调用 `_choose_quantiles` 后取列。

### P2-12 QC 特征的 0.0 同时是合法比值和缺测哨兵
- **证据**：`qc_features.py:172-173` 把未观测通道的 `relative_ptp`/`channel_std_v` 置 0，而 0 本身是合法取值（恒定通道）；`qc_features.py:114-121` 的 `record()` 只写文本口径，不写该哨兵约定。
- **影响**：`artifact` 依赖 `observed_mask` 才安全；任何只读特征数组的消费者（导出、绘图、外部脚本）无法区分"零幅值"与"没测"。
- **建议**：未观测置 NaN 并强制走 mask，或在 record 中声明哨兵字段。

### P2-13 候选分位数解析与坏通道比例解析的严格度不对称
- **证据**：`artifact.py:20-32` `parse_candidate_quantiles` 只查非空；`artifact.py:35-51` `parse_candidate_bad_channel_fractions` 校验范围 `[0,1)` 且要求有序唯一。
- **影响**：`artifact_quantiles="0.9,0.5"` 之类配置会一路传到 `policy.validate()` 才失败（`artifact.py:78-83`），错误信息不指向具体配置键；反例则立即报错。
- **建议**：两个 parser 共用同一组校验。

### P2-14 personal quality 接受但忽略 qc_features，MAD 也无下限
- **证据**：`personal_quality.py:143-148` `transform(..., qc_features=...)` 从不使用该参数（`fit` 只把它并入 digest）；`personal_quality.py:88` `spread=1.4826*MAD` 没有下限，MAD=0 时诊断区间 `[center,center]`。
- **影响**：调用方以为传入的 QC 特征会被交叉校验（artifact 路径确实校验，`artifact.py:484-486`）；MAD=0 时所有非中心值都被计入诊断计数，夸大"异常"数量（虽不参与拒绝）。
- **建议**：要么校验并透传，要么删除参数；spread 加与 artifact 相反方向的显式处理（MAD=0 → 记录 `degenerate_envelope`）。

### P2-15 fold artifact sidecar 的时间戳、静默降级与固定临时名
- **证据**：`artifact_sidecar.py:187` `"created_utc": datetime.now(UTC).isoformat()`；`artifact_sidecar.py:165-166` 捕获 `KeyError/TypeError/ValueError/JSONDecodeError` 后返回 `None`；`artifact_sidecar.py:191` `temporary = path.name + ".tmp"`。
- **影响**：同一 `fingerprint` 的 sidecar 字节每次不同（若被目录级清单/digest 覆盖则不可复现）；损坏 sidecar 被当作"不存在"并静默重拟合，丢失损坏证据；并发写同一路径互相覆盖（与 data-core P2-2 同类但发生在 sidecar）。
- **建议**：时间戳移出内容寻址文件（或写独立 audit 行）；损坏时至少回传原因；临时名加 pid/随机后缀。

## C. BIDS 入口、日程解码与基线工具

### P2-16 BIDS 入口强制每个 EEG 通道都有电极坐标，且该口径未写进合同
- **证据**：`bids_eeg.py:322-342` 对 `kind=="EEG"` 且 `electrodes.tsv` 无行的情况直接抛错；只有非 EEG 通道允许全 `n/a`。相比之下 `channel.resolve_channel_layout` 允许 montage 回退（`channel.py:448-490`）。
- **影响**：合法的 BIDS 数据集（未 digitize、或 electrodes.tsv 仅覆盖部分通道）会被采集入口拒绝，而同一批数据走 manifest 路径却可以通过；两条入口的物理坐标证据要求不一致。
- **建议**：把"是否允许平均模板回退"写成 `BidsEEGInputContract` 的显式字段，并在 provenance 记录选择了哪条。

### P2-17 BI 日程解码接受浮点整数，且逐事件 Python 解码
- **证据**：`bi2014a_schedule.py:116-120` `code=int(flash_code)`，随后 `if code != flash_code`；`20.0` 满足 `20 == 20.0` 故被接受，而报错文案是 "is not an integer code"。`bi2014a_schedule.py:167-171` `target_mask` 对每个事件调用 `decode`（Python 循环）。
- **影响**：非整数 dtype 的代码列（CSV 读成 float）会被静默接受，与拒绝语义不符；10^5 事件规模时是热路径。
- **建议**：用 `isinstance(code, (int, np.integer))` 或先做整列 dtype 校验再向量化映射。

### P2-18 raw label 审计键包含浮点原文
- **证据**：`bi2014a_schedule.py:199` `observed_key = f"code={...},raw_label={raw_label!s}"`；`bi2014a_schedule.py:211-217` 只保留 16 条 mismatch 示例，但 `to_record` 未声明截断。
- **影响**：浮点标签（如 1.0000001）会产生大量一次性键；审计记录被误读为"全部 mismatch 明细"。
- **建议**：键用规范化整数（`raw_label_int`），record 增加 `mismatch_examples_truncated_at`。

### P2-19 校准器不记录"参数命中边界"
- **证据**：`calibration.py:79-86` 温度在 `[ln0.05, ln20]` 有界优化，`calibration.py:183-190` Platt 在 `[ln1e-6, ln1e3]` 有界优化；只要 `optimum.success` 为真就直接采纳，`LogitCalibration`（`calibration.py:12-26`）没有 `at_bound` 字段。
- **影响**：验证集太小或分数退化时，温度会被钉在边界，模型仍被记录为"已校准"，下游无法区分"校准良好"和"优化失败但未报错"。
- **建议**：记录 `temperature_at_bound`/`slope_at_bound`，达到边界时降级为 `mode="uncalibrated"`。

### P2-20 `threshold_balanced_acc` 是同一批数据的重代入值
- **证据**：`calibration.py:115-117` 用选阈值时的同一 `(logits, y)` 重算 `balanced_accuracy_score`。
- **影响**：字段名容易被读成独立验证指标；实际是阈值搜索的样本内值。
- **建议**：改名或同时输出样本量/来源，并在文档声明非泛化估计。

### P2-21 验证划分的"稳定种子"依赖 numpy Generator 流，且划分本身不落盘
- **证据**：`validation.py:109-112` 用 `np.random.default_rng(seed).choice`；`GroupValidationSplit`（`validation.py:11-19`）只含 mask 与组名，不含 seed/numpy 版本。
- **影响**：numpy 不保证 `Generator` 流跨版本不变；缓存与 checkpoint 只保存 mask 时无法解释"为什么是这几个被试"，重跑可能得到不同划分。
- **建议**：把 seed、numpy 版本、候选组列表写入 split 记录。

### P2-22 审计划分在候选不足时静默关闭
- **证据**：`validation.py:139-141` `count<2` 时返回空 audit mask，无诊断字段。
- **影响**：调用方无法区分"审计被显式关闭"与"被试不足"，可能把未审计结果当作已审计。
- **建议**：返回 `audit_status`（`applied/insufficient_subjects/disabled`）。

### P2-23 进度 JSONL 清空旧文件、写 pid/时间戳且无锁
- **证据**：`progress.py:40` `target.write_text("")`；`progress.py:43-52` 每行写 `pid`、`ts` 并直接 append。
- **影响**：崩溃恢复/重跑同一 fold 会丢掉已有进度证据；文件内容不可复现；两个进程写同一 fold 会交错。
- **建议**：按 attempt 分子文件、追加而非清空，或写原子 rename 的 per-attempt 文件。

## D. 正面结论（已核实，建议保持）

- **静态检查全通过**：`ruff check` 对本期 21 个模块（含 `src/baselines/*`）全部 `All checks passed!`；无裸 `except`、无可变默认参数。
- **身份与谱系可双向校验**：`identity.py:234-269` 的 payload/digest 往返校验、记录排序与 `local_subject_id` 唯一性；`lineage.py:113-164` 的拓扑顺序、父实体可达性、NaN/inf 拒绝、`entity_digest`/`lineage_digest` 双重比对。`ParticipantIdentityRecord.__post_init__`（`identity.py:73-116`）对状态与键的组合做了穷举约束。
- **事件台账的拒绝路径扎实**：`events.py:144-147` 禁止隐式 dtype 转换（float→int 直接拒绝）、`events.py:357-375` 强制 evidence 索引双射且"可用时间不早于 onset"、`events.py:384-425` 逐组校验时间单调/单被试/单 selection/目标唯一/repetition 连续。`fingerprint`（`events.py:488-523`）覆盖全部字段与 truth。
- **fold-local 工件的拟合范围有实证校验**：`artifact.py:856-872` 用 `_fit_data_digest` 重算训练行/掩码/QC 特征，冻结模型换折即报错；held-out 全通道坏时 `artifact.py:717-722` fail-closed，显式排除版本带独立 ledger（735-795）。
- **BIDS 入口的路径与时间约束完整**：`bids_eeg.py:144-164` 防越界路径；`bids_eeg.py:370-373` 半采样 sample/onset 一致性；`bids_eeg.py:428-437` rest 不重叠且刺激不落入 rest。
- **BI2014a 只有一套解码**：`bi2014a_candidate.py:30,149,309` 全部复用 `BI2014A_FLASH_SCHEDULE`，未发现重复码表；`bi2014a_schedule.py:218-219` mismatch 直接抛 `BIFlashLabelContractError` 且带审计对象。
- **manifest 的未知字段/schema 拒绝严格**：`manifest.py:113-121,137-141`；`study.py:39-60` 的成员关系记录带自校验 digest。
- **数值合同明确不越界**：`slow_trend.py:163-177` 明确 `fit_performed=False`、`mask_changes=False`、`residual_mad_consistency_factor=1.0`、样本坐标原点与半开区间；`artifact.py` 全程"只丢训练 epoch、held-out 不静默删"。
- **domain_alignment 的状态机**：`domain_alignment.py:196-260`、`496-556` 的 schema+digest 往返、`285-356` 的对称性/正定/条件数/满秩/权重归一校验，`389-401` 显式禁止 target test 拟合（数学层由 04/05 分区覆盖，此处不重复评判）。

## E. 模块覆盖与未验证项

| 模块 | 结论 | 依据 |
|---|---|---|
| events.py | 多会话拼接受限（P1-2）、legacy 常量死代码（P2-3）、部分变换不校验（P2-4） | 全读 649 行 + 实测拼接 |
| manifest.py | channels 字符串被拆字符（P1-1）、provenance 源采样率按 subject 去重、位置跨被试均值 | 全读 + 实测 |
| identity.py | global 证据完备性不对称（P1-3） | 全读 |
| lineage.py | 无缺陷；digest 无缓存（性能由 12 分区） | 全读 |
| domain.py | 混合适配重复实现（P2-9）、参考通道恒零仍 observed（P2-8） | 全读 850 行 |
| domain_alignment.py | 仅被 `experiments/fit_galia_alignment.py` 调用，`src` 无消费者；schema/digest 完备 | 头部 + API/校验全扫 |
| artifact.py | eps 下限过度删除（P2-10）、死包装（P2-11） | 全读 892 行 |
| artifact_sidecar.py | 时间戳/静默降级/临时名（P2-15） | 全读 |
| channel.py | 混合注册证据被压平（P1-4） | 全读 618 行 + 名字实测 |
| study.py | domain_mass 一致性可能假拒绝（P2-6，存疑） | 全读 |
| metadata.py | 频基与声明不符（P1-5） | 全读 |
| exclusions.py | origin 解析双实现（P2-5） | 全读 |
| qc_features.py | 0 哨兵（P2-12） | 全读 |
| slow_trend.py | 未发现契约缺陷；边界/尾窗处理与 docstring 一致 | 全读 |
| personal_quality.py | 忽略 qc_features、MAD 无下限（P2-14） | 全读 |
| bids_eeg.py | 坐标强制口径（P2-16）；内容摘要缺口已由 01-data-core P1-3 覆盖，不重复 | 全读 460 行 |
| recording_clock.py | Decimal 精确乘法（P2-1）、缺省 legacy（P2-2） | 全读 + 调用点 |
| bi2014a_schedule.py | 浮点整数接受、循环解码、审计键（P2-17/18） | 全读 |
| baselines/calibration.py | 边界未记录、阈值指标口径（P2-19/20） | 全读 |
| baselines/validation.py | 种子可复现性声明、静默关闭审计（P2-21/22） | 全读 |
| baselines/progress.py | 清空/无锁/时间戳（P2-23） | 全读 |

**未验证**：
1. `study.py` domain_mass 假拒绝（P2-6）需真实 grouped proof checkpoint 才能复现，本次未构造。
2. artifact 的 eps 门限在真实 42 GB 缓存训练折中是否真的触发"删光训练 epoch"未跑训练验证；仅用 4 个合成尺度复现了门限行为。
3. 所有发现均为**只读静态审查 + 小规模 python -c 定点实验**，未运行任何 pytest（避免与其他分区重复与耗时）。
4. `manifest.py` 的 provenance `source_reference=next(iter({...}))`（`manifest.py:372-379`）在不同进程/随机哈希种子下顺序可能不同，未做跨进程实测，按 P2 记录在覆盖表中但未单列条目：**建议**改为 `sorted(...)[0]` 或显式记录多值。

## F. 与文档声明的对照（只列本分区新增结论）

- **不一致（P2）**：`doc/input_contract_math.zh.md:231` 声明 `PTP==0` 的实测导标记 `no_dynamic_signal`；`personal_quality.py:150,169-173,188` 只产出 `bad_channel_mask=constant` 与 `hard_rejection_basis="exact_constant_epoch_signal"`，全仓无 `no_dynamic_signal` 键。建议文档改用实际字段，或在 record 中补该标签。
- **一致（正面）**：`doc/input_contract_math.zh.md:236-240` 的 slow_trend 口径（原始参考/滤波前、1 s 完整 bin 中位数、5/15/30 s、Theil–Sen 配对斜率中位数、不跨断点/缺样、不拟合阈值、不改 X/mask）与 `slow_trend.py:58,85-87,109-131,163-177` 逐条对应；`doc:228-232` 的 personal quality 阈值（median±6·1.4826·MAD、组内≥4、包络、恒值导过半拒 epoch）与 `personal_quality.py:52-64,79-92,164` 一致。
- **一致（正面）**：`doc/technical_structure.zh.md:396` 要求 `domain_alignment` 的 fit 范围与参考可追溯；`domain_alignment.py:389-401,466-496` 的 `fit_scope`（禁 target test）与 state digest 往返满足该声明。但该模块在 `src` 内无消费者、仅 `experiments/fit_galia_alignment.py` 引用，文档宜注明"仅实验入口"；交叉引用（不重复）：`doc/technical_structure.zh.md:93` 把 `bids_eeg` 列入"字节来源"层，而 `bids_eeg.py` 只校验元数据与表格（无内容摘要）——该缺口已由 `tmp/audit/01-data-core.md` P1-3 记录，本期不重复计数。
