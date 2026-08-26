# N2P3-Net 文档总览

本目录只描述当前可执行的 Neural-RIDE / N2P3-Net 实现。日期化评审、失败诊断、
历史实验记录和数据集专用入口已移入 [归档索引](../archives/README.md)，不得从归档目录导入代码；旧缓存已删除且不归档。
v11 strict-past 架构指导（blueprint/decision record/routes/recipe）已归档到
[`../archives/legacy_v11_docs_2026-08-25/`](../archives/legacy_v11_docs_2026-08-25/)；
现行唯一架构规范是 [`blueprint.md`](blueprint.md)（v12，四个独立可证伪对象）。

## 阅读顺序

1. [constitution.md](constitution.md)：不可违反的科学与工程边界。
2. [routes.md](routes.md)：正式路线与 strict-past 研究路线的选择和边界。
3. [blueprint.md](blueprint.md)：v12 现行架构——LatencyMeasurement、RepetitionEvidence、
   Reliability 双 estimand、InnovationAudit/Stopping 四个独立可证伪对象。
4. [evaluation_protocol.md](evaluation_protocol.md)：LOSO、泄漏边界和确认性评估。
5. [datasets.md](datasets.md)：通用 EEG 数据契约、格式和通道布局。
6. [neural_ride_recipe.md](neural_ride_recipe.md)：统一训练配方。
7. [CLI_MANUAL.zh.md](CLI_MANUAL.zh.md)：现行命令。
8. [performance_lessons.md](performance_lessons.md)：云端训练通信/重复计算教训与剩余重建项。

## 现行入口

| 入口 | 用途 |
|---|---|
| experiments/prepare_eeg_dataset.py | MOABB 或原始 EEG manifest 转通用 EpochDataset |
| experiments/run_eeg_loso.py | 任意二分类 oddball EpochDataset 的 LOSO |
| experiments/run_eeg_pretrain.py | 任意二分类 EpochDataset 的辅助域预训练 |
| experiments/run_gtn_baseline.py | GTN 基线与 GTN 最新缓存生成 |
| experiments/run_n2p3net_gtn.py | GTN Neural-RIDE 主入口 |
| experiments/run_multidataset_transfer.py | 多 montage EpochDataset transfer 验收 |
| experiments/run_locked_multiseed.py | GTN 锁定多 seed 复现；未暴露 cohort 的一次性确认性运行 |
| experiments/run_pcw_claim_gate.py | v11 PCW routing 对照；v12 生理潜伏期 claim 走 blueprint 对象 L |
| experiments/run_latency_measurement_gate.py | v12 LatencyMeasurement 双折合成 gate（Phase 1） |
| experiments/run_paired_test.py | 被试级配对比较 |

旧的 BNCI、Brain Invaders、ERP CORE 下载/LOSO/预训练脚本已删除。它们现在通过
MOABB 类名和同一套通用入口接入，不再拥有数据集专用训练默认值。

## 当前科学口径

- GTN 是 9 选 1 主任务锚点；跨模型主指标是 validation-calibrated `exact_llr@3` ITT hit。
- 当前 GTN eligibility 全集已开发暴露，只能提供 locked development/replication；确认性 SOTA
  声明需要新增受试者或从未查看过的外部 cohort。
- 普通 P300 数据集只做二分类 LOSO、预训练或显式域对齐，不能替代 GTN 主验收。
- 当前有两条明确路线：正式路线为 `neural_ride_v12_pcw_fail_closed`，研究路线为
  `neural_ride_v12_strict_past_research`。默认 `--lambda-innovation 0`；正数才进入
  strict-past 研究分支。另有默认关闭的 `--z2-aux-head add/replace` E5 claim-gate
  研究对照（full-Z2 auxiliary head），未过预注册嵌套门槛不得进入生产。v12 起研究分支的增量证据只接受嵌套 `M0:a+bS` 对
  `M1:a+bS+cL` 的 subject-cluster bootstrap 审计（详见
  [`blueprint.md`](blueprint.md) 第 5 节）；旧的单参数非负 fusion 契约已归档。
  正式部署输出 `final = PCW`，任何新对象默认 fail-closed。
- L_amp、L_jit 默认关闭；L_MMD 只用于单独登记的跨域实验。
- 标准数据入口不补电极、不替代电极。一个训练 run 必须解析为一个固定、真实、带坐标的布局。

## 维护规则

- 代码、CLI、数据 schema 或默认配方改变时，同步更新本目录。
- 历史事实只放归档；现行文档不保留已删除命令和复现兼容分支。
- 新方法经完整协议证明正确并替代旧方法后，删除旧实现、旧入口和旧文档；不要为了保留而保留。
- 新格式优先扩展 manifest/MOABB adapter，禁止新增以数据集名称分支的训练脚本。
- 任何缓存或运行记录必须写 schema、物理时间轴、通道坐标、数据来源和 resolved config。
