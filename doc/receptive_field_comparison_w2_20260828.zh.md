# 总感受野机制矩阵 W2 实验报告

状态：BI2014a inner-development evidence，不是 outer-test 结论，也不是 GTN 确认。

## 1. 冻结合同

- source commit：`4fadd98d08e34597e67fe0007c7dce6cd3aa3708`
- cache SHA-256：`60f7c97fce94dc808784972f5acbfc326f7e1de4af61353c25e94eaa7e672d4b`
- cache：BI2014a，64 subjects，16 channels，128 samples，128 Hz
- development folds：`0-7,16-23,32-39,48-55`，共 32 个 held-out subjects
- seeds：`20260828,20260829`
- 训练：30 epochs，patience 6，physical batch 512，无梯度累积
- 执行：4 个 GPU fold workers；CPU/QC 数由 cgroup 配额解析为 25
- 选择指标：subject 内先平均 seeds，再比较 inner `best_task_val_loss`
- 次级指标：inner `final_task_val_auc`
- outer 指标虽由通用 fold evaluator 内部计算，但没有持久化，也没有用于排序或选择

固定五臂如下：

| arm | ST kernel / dilation | MST kernels | 总 RF samples | 16ch 参数 | 角色 |
|---|---:|---:|---:|---:|---|
| A | 65 / 1 | 5,17 | 84,132 | 1506 | broad dense reference |
| B | 35 / 1 | 5,17 | 54,102 | 1266 | 旧 sensitivity 领先；当前核长臂 |
| C | 33 / 1 | 5,17 | 52,100 | 1250 | local, same taps as D |
| D | 33 / 2 | 5,17 | 84,132 | 1250 | broad, same taps as C |
| E | 33 / 1 | 13,25 | 84,132 | 1506 | 与 A 等 RF、等参数的层间重分配 |

## 2. 完整性

共完成 `5 x 2 x 32 = 320` 个 fold fits，OOM retry 为 0。平均训练 28.73 epochs，
范围 13--30。fold records 显示 fused Adam 与 `reduce-overhead` compile 均实际启用。
QC 首次构建使用 25 个 worker，耗时 130.43 s；10 个 arm-seed 批次累计 wall time
为 1387.16 s。

本轮 manifest 请求 `precision=auto`，但专用 runner 当时没有把每 fold 的最终 precision
写入 inner-only record。因此不能从结果文件声称所有 fold 都执行 BF16；这是记录缺口，
不是训练失败。专用 runner 已在 W2 完成后补齐 precision 与显存字段；W2 原记录保持不变。

## 3. Arm 均值

以下均值先在每个 subject 内平均两个 seeds：

| arm | inner best val loss | inner final val AUC |
|---|---:|---:|
| A | 0.572378 | 0.746742 |
| B | 0.572048 | 0.747974 |
| C | 0.572365 | 0.747504 |
| D | 0.573731 | 0.745027 |
| E | 0.571586 | 0.747759 |

不得按这张均值表事后挑最高 AUC。预注册判读只使用下列三个 planned contrasts。

## 4. Subject-paired contrasts

对 32 个 subject-level 差值做 100,000 次 percentile bootstrap，随机种子
`20260828`。CI 未作跨三个 contrasts 的 simultaneous coverage 校正，因此属于开发证据。

| contrast | loss delta [95% CI] | AUC delta [95% CI] | 方向 |
|---|---:|---:|---|
| C-D | -0.001366 [-0.002615,-0.000163] | +0.002477 [+0.001022,+0.003986] | C 优于 D |
| D-A | +0.001353 [+0.000284,+0.002475] | -0.001715 [-0.003273,-0.000162] | D 劣于 A |
| E-A | -0.000792 [-0.001666,+0.000039] | +0.001017 [-0.000349,+0.002353] | 小差异，CI 跨 0 |

`C-D` 在两个 seeds 中方向一致，但 AUC 增益 `0.00248` 低于预设的 `0.005`
实质阈值。D 同时改变了总跨度和 dilation 采样格点，因而该结果不能证明“短 RF 本身
更优”；它更直接地否定了“用稀疏 dilation 免费恢复 K65 总跨度”这一实现。

`D-A` 表明相同总 RF 并不产生相同性能。系数密度、共享层 tap 数和优化路径仍然重要，
所以总 RF 不是充分解释变量。

`E-A` 说明等 RF、等总参数下重新分配 shared/branch taps 可能改善 loss，但 95% CI
仍跨 0，不能冻结为性能赢家。若继续该方向，研究问题应改为层间 factorization，而不是
局部 RF。

## 5. 冻结决定

不进入 W3。理由是主要差值均低于 `0.005` 实质阈值，继续增加 BI folds/seeds 更可能
精确估计一个很小的开发集效应，而不是改变工程决策。

探索性 `C-A` 非劣检查为：

- loss delta `-0.000013`，95% CI `[-0.00118,+0.00113]`；
- AUC delta `+0.000762`，95% CI `[-0.000592,+0.002136]`；
- C 参数为 1250，较 A 的 1506 少 17.0%；总 RF 从 `84/132` 缩为 `52/100`。

因此 C 证明 K33 是“更短 RF、较少参数、BI inner validation 对 K65 非劣”的机制臂，
不是已确认的性能赢家。B/K35 在 A/B/C 中有最高 mean inner AUC (`0.747974`)，
且 loss 优于 K33/K65，但 W2 未预注册 B-C 或 B-A 直接 contrast，不能据此选胜者，
也不能把 B 删除。D 停止；E 留作未来 factorization 研究。

## 6. GTN development 后继结果

相同 4-block、3-seed、steady-state Z0 链已完成：

| arm | balanced-all | raw-all | AUC | 判定 |
|---|---:|---:|---:|---|
| B / K35 | **0.669** | **0.687** | **0.694** | 临时工程默认 |
| A / K65 | 0.654 | 0.669 | 0.686 | 未分离强对照 |
| C / K33 | 0.623 | 0.623 | 0.681 | 退出主线 |

K35-K33=`+0.046`，95% CI `[+0.012,+0.082]`，三 seed 同方向；K35-K65=`+0.015`，
CI `[-0.012,+0.044]`，且 seed 方向反转。因此 K35 的旧 sensitivity 领先得到支持，
但唯一核长仍未确认。工程上采用 K35、保留 K65，不再继续 K33 或新增核长搜索。

## 7. 证据文件

- `doc/assets/receptive_field_comparison_w2_20260828/manifest.json`
- `doc/assets/receptive_field_comparison_w2_20260828/screening.json`
- `doc/assets/receptive_field_comparison_w2_20260828/paired_subject_analysis.json`
- `doc/assets/receptive_field_comparison_w2_20260828/run_audit.json`
- 完整 compact archive（本地 `tmp`，不入 Git）：
  `rf_mechanism_w2_summary_4fadd98.tar.gz`
- archive SHA-256：`e19d39cf064149483de2ecdb049675343e515e0299679aead3321699273f4848`
