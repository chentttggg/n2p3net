# 目标早停消融实现与GPU试跑

2026-09-08全量已完成，3173/3173评分及3050个保存模型复核通过，0失败。
当前CE早停使AUC下降且含refit总更新增加；完整统计与结论见
[全量结果](target_early_stopping_results_20260908.zh.md)。以下保留实现、试跑与启动时点记录。

2026-09-07。用户要求在高预算迁移基准上实现早停消融，阶段依据沿用
[此前分析](four_study_standard_budgets_20260907.zh.md)：固定source，优先目标全网、头微调和
同K scratch；K0/LDA不早停，校准内选轮后从原初始化在全K重训。
有效方案见[roadmap](../roadmap.zh.md#2026-09-07-目标早停消融)。

## 共享实现

新增`transfer.epoch_selection.EpochSelectionPolicy`及`fit_with_epoch_selection`，公共builder、
两个CLI及分析器通过`--stage ES`进入。每个正K为三类NN×三策略及一个LDA，共10个结果；
K0仅源zero-shot。默认NN上限60遍、lr1e-3、batch上限512，patience6、min_delta1e-6。
full/scratch采用目标统计、在线BN和可训练affine；head只训练分类头，源BN buffer/affine冻结。

| 策略 | 内层 | 完整校准拟合 |
|---|---|---|
| fixed | 不选轮 | 原初始化全K固定60遍 |
| full_trajectory | 同一合法时间划分，跑满60遍按validation CE选轮 | 丢弃inner状态，原初始化全K重训选中轮数 |
| patience | 连续6轮无超过1e-6的CE改善时实际停止，保留此前最佳轮数 | 同上 |

多记录按真实时间留出末尾约20%的完整记录；单记录80/20并执行raw-support purge。
输入统计/BN/LDA仅在当次fit范围拟合；validation不参与梯度或训练态BN，最终test不参与选择。
非法inner划分执行相同固定高预算配方，明确记录fallback原因和`selection_evidence=false`。

`SubjectAdapter.validation_behavior=select_epoch_full_trajectory`选轮但不提前停止；既有
`select_epoch`保留patience行为，`diagnostics_only`只诊断。非有限选轮loss明确失败。
停止参数进入`CalibrationCandidateFitCache`键；不同inner策略分别真实训练，相同全K/选中轮数
refit可精确复用，scope结束释放缓存。目标runner复用Prepared接口避免反复解析源大账本。
旧TargetRecipe字段/默认值保持不变，产品E3五候选流程没有自动加入早停。

模型保存实际recipe、停止策略与两阶段账本；cell区分requested_recipe和实际recipe。
记录inner/selected/refit轮数、CE曲线、行摘要、总与物理updates/draws、两阶段总墙钟及缓存命中。
LDA拟合也进入计数。fixed与选轮比较包含内层划分和refit的整体影响；patience与full_trajectory
进一步比较提前终止。不能用缓存命中时间证明某策略单独运行更快。

## 软件验证

反例覆盖真实停止与完整轨迹继续扫描、相同选轮后与独立原初始化全K拟合精确一致、
改final test不改变选择/状态、改validation不改变inner训练loss/初始化/训练摘要、非法inner回退、
非法参数、跨策略refit复用、生产任务与模型保存、LDA及NN总计。
全套1208通过、4跳过、0失败，234.67秒，JUnit为`tmp/early_stopping_full_tests.xml`。
之后补齐LDA计数及请求/实际配方的两个生产集成测试通过。运行前全仓Ruff通过；两个公共CLI及
分析CLI从项目外目录验证ES入口。Spec审查发现并修复LDA漏计，Standards没有阻断项。
其他并发输入/QC实现改动全部保留。
收尾时本次早停修改文件的Ruff和`git diff --check`通过；同时进行的EOG/QC开发随后在其他
文件出现6个Ruff问题（5个import顺序、1个`field`名称遮蔽），未修改这些并发文件。
上述全套与GPU证据绑定已冻结的执行源码，不冒称正在变化的整棵工作树再次通过全套。

## GPU试跑

固定既有六导`S1_observed_rows_uniform_rows`、4000-update源模型，不新增source。
每库按声明subject标识排序取前两人，不按成绩挑选；8人、21个实际正K、218个评分。
请求K0/200/800/2000，不足按原规则回退/去重；精确K200/800/2000分别8/5/4人。
运行目录为`/root/n2p3-runs/early_stopping_probe_20260907/`。

执行快照240成员，SHA为`8f92f87cf22e9f3faca177963dc2a80add06a92e86858b0042e0d37fcd396db8`。
source保留原快照，评价从本次物化checkout执行。启动前性能/资源技能复核25核配额、90 GiB容器
上限、RTX5090，按实际可用内存/显存解析8个目标进程。进程日志确认8个CUDA CLI重叠执行；
本次未采样GPU利用率时间序列，不据此声称全程满载。

8个生产CLI dry-run通过；218/218评分成功、0失败，336次逻辑fit请求，262次真实fit、74次复用。
训练最慢分片58.28秒，含预检与分析总墙钟82.15秒。退出后GPU0%、2 MiB。
218份预测的AUC、行摘要、合同、NN状态、CE选轮、全K refit及预算独立复算一致。

| 模型 | patience提前停止比例 | 选中轮数中位数 | fixed总更新 | patience总更新，含refit |
|---|---:|---:|---:|---:|
| 全网迁移 | 47.6% | 54 | 2460 | 3482 |
| 头微调 | 61.9% | 40 | 2460 | 2877 |
| Scratch | 85.7% | 37 | 2460 | 2794 |

全网patience加refit后总更新比fixed多41.5%；缓存后的物理更新为2024，两种成本不能混淆。
全网patience减fixed的AUC差在K200/K800/K2000为-0.0308/-0.0106/-0.0167；完整轨迹选轮
减fixed为-0.0191/-0.0106/-0.0052。这个非随机8人小试没有显示早停收益，只支持流程与成本账本
可运行，不作总体推断或据此调停止超参数。固定高预算仍为基准。

## CLI与完整清单

```powershell
.venv/Scripts/python.exe experiments/run_cross_dataset_transfer.py --plan target_plan.json --subjects P01 --stage ES --dry-run --device cpu --output tmp/es_tasks
.venv/Scripts/python.exe experiments/run_binary_subject_transfer.py --training-plan target_plan.json --target-subjects P01 --stage ES --dry-run --output tmp/es_binary_tasks
.venv/Scripts/python.exe experiments/analyze_training_chain.py --stage ES --record run/record.json --studies BI2014a BI2015a BNCI2014-008 Huebner2017 --source-seeds 20260906 --target-seeds 20260906 --output analysis_es.json
```

plan沿用生产cache/checkpoint、来源/评价snapshot与协议，明确用户batch512规则；
`epoch_selection`仅接受`patience`和`min_delta`。真实执行去掉dry-run，从物化checkout调用。
完整四库dry-run已核验123人、305个正K、3173个评分，最多4880次逻辑拟合，0次新增source。
该清单随后按用户明确要求启动全量，见下文；小试本身仍只覆盖上述8人。

## 全量启动

用户明确“训练就是全量训练，必要验证自行完成后直接开始”。新控制器PID31745，运行目录
`/root/n2p3-runs/early_stopping_full512_20260907/`，使用8个CUDA worker。
沿用已通过1208项软件测试和GPU试跑的8f92f87c快照，从原物化checkout执行；没有将并发QC改动
临时混入本轮。配方、源模型、真实校准/test和batch512上限均保持原声明。

原小试全部文件按归档清单重算SHA；218个结果另核对配方、选轮政策、行摘要、seed、合同和
evaluation snapshot后保留原路径，续跑余下115人2955评分。完整目标仍为123人3173评分，
不是重新执行3173次加上旧218次。新`prepared.json`/`execution.json`记录复用来源、资源及任务，
完成后分析器合并已核验小试与新分片，保持完整队列和失败分母。
启动后约83秒采样：8个训练进程各约占用一个CPU核，GPU100%、显存6155 MiB；这是实际启动
采样而非全程平均。进程树、当前execution和资源快照已回传本地同名全量结果目录。

本地产物在`experiments/runs/early_stopping_probe_20260907/`，含plans、dry-run、预测、模型、
分析、进程记录与审计程序。`probe_results.tar.gz`为15634726字节，SHA为
`2938e090faffd4944fb047442b740e8eb70f67ed1bf5ffb8bc8546dc4ff8f0f8`；逐成员重算及云/本地SHA一致。
见[结果清单](../../frozen/early_stopping_20260907/probe_results.manifest.json)和
[源码清单](../../frozen/early_stopping_20260907/source.manifest.json)。随后完整dry-run清单单独保留
为`full_dry_run.json`，不倒填到先前试跑归档。不涉及Mainsah、额外seed或成人/设备确认。
