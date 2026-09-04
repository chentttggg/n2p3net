# 域迁移 v2 主线云端消融实验方案（2026-09-04）

_状态：预注册开发方案，待代码冻结与本地 dry-run 后执行。所有产出仅限 development/机制证据，不得作为性能或产品证据。_
_主线：EA 默认数据对齐 → N2P3Net trunk → faithful AS-MMD → TSD/Dense PDS 选样。_

---

## 0. 结论摘要

1. 代码链已补齐：EA cache、faithful AS-MMD（frozen-logits 与 live source-stream + SplitBN 双 buffer）、TSD `n2p3_source_selection/2`、Dense PDS 忠实采样、target-calibration cache、matrix plan 的 EA/AS-MMD/sampler 字段与 result_batch 下传。
2. 云端消融分三层，先便宜后贵：
   - **L1a dry-run**：2 target × 1 seed × short epoch，本地即可跑，只验 DAG/argv/provenance。
   - **L1b 单 worker CUDA smoke**：2 target × 1 seed，短 epoch，验 CUDA/精度/checkpoint/journal。
   - **L2 development 全跑**：64 target × 1 seed，按 §5 臂矩阵；AS-MMD live 臂要求多域 source cache，未就绪前**整臂 fail-closed 不启动**。
3. 单域 BI2014a 的 L2 臂**不含** live AS-MMD（需要 `splitbn_domain_count>=2`）；AS-MMD 的公式/双 buffer 行为先在本地合成测试与 CUDA smoke 中验收，再等 §6 多域 source cache 就绪后上 64-fold。
4. 所有 source-selection 的 target 统计必须来自 `target_calibration` cache；本方案新增 `prepare_target_calibration_cache.py`，从根上禁止 target-test 行进入 sampler。
5. **test-selection 硬门槛已删除（commit `e02e752`，冻结 `e02e752`）**：不再要求凑够 8 个完整 repetition；每个后续且时间不重叠的 decision 一律进入账本，hit@R 曲线照记 R=1..8，深度不足的层记 `incomplete`。旧 1-seed 开发跑（31/64 有可评 decision）作废重跑。
6. 本轮不启动 E2 wet-dry 比较；不覆盖 E1 冻结 `b96a6af`/`b86d558`。

---

## 1. 输入与冻结

| 输入 | 位置 | SHA-256 |
|---|---|---|
| 当前源码冻结 | `frozen/active_source_8ddab10.*`（source_commit `8ddab103d93e913628026f57b526eba7f225b960`，C1 三重互证；archive member_count=225，byte_size=1,106,315） | `fcbc6c11bced1271ca6235348c83b142c63eb4c540af6fbf6d2f6458b360d698` |
| 调度修复冻结（CAR16 对照） | `frozen/active_source_2a75407.*`（source_commit `2a75407ca8055f831233f75b1ffe983080d8ccd1`，member_count=225，byte_size=1,106,909） | `4f3ad1bbbe1f7b9fa33db2654201bebf8631743cfdbdeead320055b93fcb610d` |
| 上一冻结（已由 `e02e752` 取代） | `frozen/active_source_ec2a9cb.*`（source_commit `ec2a9cb`） | `7900582da556c3caf1eff359245702713c422bdabc511a63d18ba8a4dcebbe76` |
| BI2014a causal-v3 64 人 cache | `tmp/l0_bi2014a_causal_v3_64subj.npz` | `842466e45d63e017704a2fe4ebc1ee550c54eff7b400bdd27c7454bb4ae69fff` |
| BI2014a causal-v3 CAR16 cache | `tmp/l0_bi2014a_causal_v3_64subj_car16.npz` | `37710abb94255b53c9e2df478e642c688095ac93b23be627fa01abdedda4ef2b` |

E1 训练/评估锚点 `b96a6af`（native）/`b86d558`（CAR）只读，不复用其 checkpoint 做新臂；新臂全部从新 source-freeze 重新训练。

---

## 2. 本地派生 cache 的生成顺序

所有输出必须先过 `load_epoch_dataset(..., validation="attested")` 与相邻 `.record.json`，再上传；云端逐文件 SHA-256 复验。

```bash
# (1) target calibration-only cache（禁止 target test 行进入任何 sampler）
# 松弛门槛后：64/64 受试都有 calibration 行，usable_subjects=64。
python experiments/prepare_target_calibration_cache.py \
  --dataset-cache tmp/l0_bi2014a_causal_v3_64subj.npz \
  --output cloud_inputs/bi2014a_target_calibration_c5_t8_relaxed.npz \
  --calibration-selections 5 --test-repetitions 8

# (2) EA(target_calibration) source cache：EA 参考只来自已声明 calibration 行
python experiments/prepare_euclidean_alignment_cache.py \
  --dataset-cache cloud_inputs/bi2014a_target_calibration_c5_t8_relaxed.npz \
  --output cloud_inputs/bi2014a_ea_target_calibration_relaxed.npz \
  --fit-scope target_calibration --reference-mode native_preserve

# (3) 对齐特征矩阵
python experiments/prepare_domain_features.py \
  --dataset-cache cloud_inputs/bi2014a_target_calibration_c5_t8_relaxed.npz \
  --output cloud_inputs/bi2014a_target_cal_features_relaxed.npz
python experiments/prepare_domain_features.py \
  --dataset-cache tmp/l0_bi2014a_causal_v3_64subj.npz \
  --output cloud_inputs/bi2014a_source_features_relaxed.npz

# (4) 选样所需的 label/group 向量
python experiments/prepare_source_selection_vectors.py \
  --dataset-cache tmp/l0_bi2014a_causal_v3_64subj.npz \
  --output-labels cloud_inputs/bi2014a_source_labels_relaxed.npz \
  --output-groups cloud_inputs/bi2014a_source_subject_groups_relaxed.npz

# (5) TSD 选样（shared per-arm record；target 只来自 calibration features）
python experiments/prepare_source_selection.py \
  --source-features cloud_inputs/bi2014a_source_features_relaxed.npz \
  --target-features cloud_inputs/bi2014a_target_cal_features_relaxed.npz \
  --output cloud_inputs/bi2014a_tsd_selection_n2000_relaxed.json \
  --n-select 2000 --seed 20260904 --sampler target_similarity_diverse \
  --selection-scope target_calibration

# (6) Dense PDS 忠实选样：r0 用 source 特征最近邻中位数离线估计并冻结
python experiments/prepare_source_selection.py \
  --source-features cloud_inputs/bi2014a_source_features_relaxed.npz \
  --target-features cloud_inputs/bi2014a_target_cal_features_relaxed.npz \
  --output cloud_inputs/bi2014a_dense_pds_selection_n2000_relaxed.json \
  --n-select 2000 --seed 20260904 --sampler dense_pds \
  --source-labels cloud_inputs/bi2014a_source_labels_relaxed.npz \
  --pds-radius <r0_frozen> --selection-scope target_calibration
```

- 本消融采用 `N_TSD=N_DENSE=2000`；`r0` 以 source 特征最近邻中位数离线估计并在发射前写回本方案后冻结；未冻结不得启动 L2。
- 缓存生成必须带 CPU/GPU 利用率监测（云端按 Linux `%CPU` 计，>1000% 为 10+ 核；峰值 <1000% 要书面反思并决定是否分片并行）。
- CAR 对照臂使用 CAR16 cache 走同一 reference CLI，`--reference-mode common_average`。
- 每个派生 cache 的 `data_alignment`/`target_calibration_scope`/row-identity digest 必须由 matrix `_verify_inputs` 门禁复核。

---

## 3. L1a dry-run（本地，不花钱）

```bash
python experiments/run_candidate_promotion_matrix.py \
  --plan configs/domain_transfer_v2_ablation.example.json \
  --dry-run --max-workers 1
```

必须通过：plan digest、5 个 source-arm 的 EA/sampler 声明、checkpoint batch argv 的 `--source-selection-record`、result batch argv 的 `--source-selection-record`、零 lambda 臂无 AS-MMD 旗标。任何失败按 plan/缓存/代码三层定位后再跑。

---

## 4. L1b 单 worker CUDA smoke

```bash
python experiments/run_candidate_promotion_matrix.py \
  --plan <smoke_plan.json> --device cuda --max-workers 1
```

smoke plan 与 example 同构，但 `training.epochs=2`、各 fitted evaluation arm `epochs=2`、target_subjects 只保留 2 人、output_root 独立。验收：
- checkpoint `/2` payload、result/manifest/analysis 全链落盘；
- EA 臂 cache 被 `_verify_inputs` 接受；
- TSD/Dense PDS 臂的 `source_selection_contract`、`tsd_dropped_source_epochs` 与 selection record 一致；
- CUDA OOM/精度路径无异常；journal 可 resume。

---

## 5. L2 64-fold 臂矩阵（1 seed × 64 target）

`target_cache` 统一为 BI2014a causal-v3 64 人；每个 source arm 复用一个 checkpoint 批次，所有 evaluation arm 在同一 checkpoint 上评估。主指标 `valid_subject_operational_hit_rate`；`requested` hit@R 与 analysis 新增 `decision_curve` 的每层 `valid_hit_rate` 为诊断。**松弛门槛后 `requested` 口径固定为全部时间上可评的 later decisions；`incomplete` 只出现在单 decision 的深 R 层，不再整 decision 剔除。**

| Arm | source arm | depth adaptation | 主对照 |
|---|---|---|---|
| `B0` | bi2014a_native | zero_shot | no-transfer floor |
| `A1` | bi2014a_native | full_fine_no_mmd | 深度微调主效应 |
| `A1-cal` | bi2014a_calibration_native | full_fine_no_mmd | calibration 数据量匹配控制 |
| `EA-target-cal` | bi2014a_ea_target_calibration | full_fine_no_mmd | EA 主效应（对 `A1-cal`） |
| `S0` | bi2014a_native | linear_target | 选样对照 |
| `S1` | bi2014a_tsd_n2000（uniform 预训练→selected 继续训练） | linear_target | TSD 主效应 |
| `S2` | bi2014a_dense_pds_n2000（uniform 预训练→selected 继续训练） | linear_target | Dense PDS 忠实对照 |

优先 contrasts：`EA-target-cal vs A1-cal`、`S1 vs S0`、`S2 vs S1`。其余 exploratory。
CAR16 预处理对照用独立 plan（native CAR16 对 native 逐参与者比较）。

**AS-MMD live 臂（L2b，待多域 source cache）**：

```json
{
  "name": "as_mmd_source_stream",
  "head": "full_fine",
  "normalization": "source",
  "epoch_selection": "fixed_budget",
  "epochs": 40,
  "batch_size": 32,
  "lr": 0.0001,
  "target_stat_weight": 0.0,
  "fold_local_qc": false,
  "as_mmd": {
    "lambda0": 0.4,
    "warmup_epochs": 40,
    "target_weight_min": 1.0,
    "target_weight_max": 6.0,
    "input": "source_stream",
    "splitbn_target_buffer_source": "target_calibration"
  }
}
```

配套 `training.splitbn_target_buffer_source="target_calibration"`、source arm 声明真实多域 axis（≥2 域）。对照组 `no_mmd`（同 head/full_fine，无 `as_mmd`）与 frozen-logits 臂并行。负 MMD 不 clamp，ledger 记录 `negative_mmd`/buffer digests。

---

## 6. 多域 source cache 准备（AS-MMD L2b 前置）

- 来源候选：BI2015b（湿 32 导）与 BNCI2014-008 或 GTN 中可证明同参考、同 montage 的 cohort；最终域定义需在缓存构建前由 T 拍板。
- 命令：
  ```bash
  python experiments/prepare_multidomain_source.py \
    --source NAMESPACE_A=<cache_a.npz> --source NAMESPACE_B=<cache_b.npz> \
    --target-channels <shared channels> --reference-mode native_preserve \
    --output cloud_inputs/bi_multidomain_source.npz
  ```
- 域轴/参考/identity 三证必须通过 `preflight_artifact_qc.py` 与 matrix `_verify_inputs`；缺任一项不启动 L2b。
- live AS-MMD 的 `source_domain_ids` 顺序必须与 checkpoint `source_domain_axis` vocabulary 完全一致（runner 已 fail-closed 校验）。

---

## 7. 结果读取纪律

- 只报告预声明的 matched fold/seed contrasts 与 participant-cluster CI；AUC 只作支持。
- EA 臂必须同时报告 reference mode、fit scope、条件数、buffer digest、row-identity digest。
- TSD/Dense PDS 必须同时报告 `sampler_kind`、radius/mixture、attempts/fallback、source/target/selected digests。
- AS-MMD 必须同时报告每 epoch `w_T(e)`、`λ_MMD(e)`、MMD²、negative count、bandwidth、SplitBN source/target buffer digests。
- 任何 arm 的 target 统计只允许来自 `target_calibration` 或 unlabelled buffer；发现 target test 统计立即整 run invalid，不重试掩盖。

---

## 8. 冻结与上传

1. 代码/测试提交并生成新 source-only freeze + manifest，C1 三重互证（archive SHA ↔ manifest ↔ source_commit）。**已完成：`e3a9a33`，archive `2794c8fd…`，member_count 225。**
2. 派生 cache 上传后云端逐文件 SHA-256 复验，未复验不跑。
3. 冻结 `b96a6af`/`b86d558` 与历史 `6fc1df1`/`352932b`/`326c8fd`/`ae2706e` 只读。
4. 本轮不启动 E2 wet-dry 比较；L2b 只有在 §6 多域缓存与 T 裁决完成后才预注册发射。
