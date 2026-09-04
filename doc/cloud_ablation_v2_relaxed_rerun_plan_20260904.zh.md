# v2.1 松弛门槛 + 三项机制修复重跑方案（2026-09-04 深夜）

_状态：代码已实现、全量 pytest 绿；待生成新 freeze、上传云端、smoke、64-fold 发射。_
_所有产出仅限 development/机制证据。_

## 1. 变更总览

| # | 变更 | 代码位置 |
|---|---|---|
| 1 | 删除 8 个完整 repetition 硬门槛，hit@R 曲线照记 | candidate_within_subject.py |
| 2 | 结果/分析显式报告每层 R 的 alid 分母与 alid_hit_rate | outcomes.py、nalyze_candidate_cross_decision.py |
| 3 | 补上真正的 EA(target_calibration) 臂及其 calibration_native 对照 | 计划层，cache 已生成 |
| 4 | TSD/Dense PDS 改为「uniform 全源预训练 → 从该 checkpoint 初始化 → selected rows 继续训练」 | 
un_pretrain_supervised.py --initial-checkpoint、plan initial_checkpoint_arm |
| 5 | 新增 CAR16 预处理对照计划（独立 plan，逐参与者与 native 计划比较） | 	mp/make_plan_car16_e02.py |

## 2. 关键口径变化

- 
equested_hit_rate：仍保留，correct / requested（含 incomplete/fit_failure）。
- alid_hit_rate：correct / (correct + incorrect + tie)，排除 incomplete/fit_failure；
  每个 evidence level 都输出 
equested、alid、correct、两个 rate。
- analysis JSON 新增 decision_curve[arm][R]，R=1..8 汇总，旧口径不再作为唯一读法。
- 本地 CPU 纪律：本地任务总 CPU 不超过 90%；缓存生成峰值 <1000% 必须书面反思。
- 云端 CPU/GPU：TSD 峰值 >1000% 或 GPU 路径；Dense PDS 用 GPU float64 backend。

## 3. 臂矩阵（native 计划）

| 源臂 | 训练数据 | 机制 |
|---|---|---|
| bi2014a_native | 全源 63 人 | uniform |
| bi2014a_calibration_native | 63 人的 calibration rows | A1-cal 控制 |
| bi2014a_ea_target_calibration | 63 人的 calibration rows，EA(target_calibration) | EA 主效应 |
| bi2014a_tsd_n2000 | 先 uniform 全源预训练，再 TSD n=2000 继续训练 | 两阶段 TSD |
| bi2014a_dense_pds_n2000 | 先 uniform 全源预训练，再 Dense PDS n=2000 继续训练 | 两阶段 Dense PDS |

- 两阶段实现：selected checkpoint batch **依赖** uniform i2014a_native checkpoint batch，
  以每 fold 同 holdout 的 uniform checkpoint 作为 --initial-checkpoint；
  初始 checkpoint 必须同 source cache、同 source snapshot、且自身无 selection contract。
- CAR16 计划独立：i2014a_native_car16，target/source 均为 CAR16 cache；
  与 native 计划在 participant level 做 paired bootstrap 对照（单独脚本输出）。

## 4. 计划文件与输出根

- 	mp/plan_dt_v2_relaxed_smoke.json → /root/autodl-tmp/n2p3-runs/domain_transfer_v2_relaxed_smoke
- 	mp/plan_dt_v2_relaxed_dev64_1seed.json → /root/autodl-tmp/n2p3-runs/domain_transfer_v2_dev64_relaxed_1seed
- 	mp/plan_dt_v2_relaxed_car16_smoke.json → domain_transfer_v2_relaxed_car16_smoke
- 	mp/plan_dt_v2_relaxed_car16_dev64_1seed.json → domain_transfer_v2_relaxed_car16_dev64_1seed

## 5. 执行步骤

1. commit + 新 source-only freeze + C1 互证；
2. 上传 freeze 到 /root/autodl-tmp/n2p3-frozen/，解压 
2p3-e02e752 或新树；
3. 上传/复验 relaxed caches 与两个 plan JSON；
4. 云端 GPU 生成 Dense PDS relaxed（监控）；
5. native plan dry-run → 2 人 smoke → 64-fold 1 seed；
6. CAR16 plan smoke → 64-fold；
7. analysis 重算，报告 decision_curve 与三组预注册 contrasts + CAR16 paired 对照。

## 6. 冻结与已知状态

- 本版代码冻结 commit 与 archive 会在提交后生成，并替换 e02e752 作为唯一执行锚点。
- 旧 ec2a9cb 1-seed 结果仅作历史诊断，不作性能证据。
