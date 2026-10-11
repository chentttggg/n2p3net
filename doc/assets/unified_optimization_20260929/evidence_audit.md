# 十二库统一框架：独立证据复算（2026-09-29）

只读复算已有参与者级 AUC 和 BNCI hit 记录，没有训练、预测重放或改写历史产物。项目 .venv Python；NumPy；20,000 次按固定库内参与者成组配对 bootstrap，库等权，analysis seed=20260929。新增区间和比较均属探索性；条件于单次训练 seed=20260918、固定源选择及固定四库。原参与者抽样无法覆盖训练随机性与新数据域。机器可读数据见同目录 evidence_audit.json。

## 四格结果

| 范围 | K | n | A0Q0 | A0Q1 | A1Q0 | A1Q1 | interaction及95%区间 |
|---|---:|---:|---:|---:|---:|---:|---|
| 普通四库 | 200 | 123 | 0.699487 | 0.697069 | 0.700328 | 0.689475 | -0.008436 [-0.012289, -0.004666] |
| 普通四库 | 800 | 72 | 0.766600 | 0.754627 | 0.767420 | 0.750750 | -0.004697 [-0.009275, -0.000024] |
| binary-only三库 | 200 | 115 | 0.709408 | 0.712137 | 0.708719 | 0.709316 | -0.002131 [-0.003043, -0.001221] |
| binary-only三库 | 800 | 64 | 0.770565 | 0.773318 | 0.773062 | 0.773745 | -0.002070 [-0.003256, -0.000925] |
| BNCI | 200 | 8 | 0.669721 | 0.651868 | 0.675155 | 0.629954 | -0.027348 [-0.042446, -0.012630] |
| BNCI | 800 | 8 | 0.754706 | 0.698553 | 0.750493 | 0.681765 | -0.012575 [-0.030097, +0.005766] |

interaction = (A1Q1-A1Q0) - (A0Q1-A0Q0)。四库权重是每库1/4，三库重算为每库1/3；不是按123人直接混池。Q1在binary-only三库为prior-only，在BNCI为prior+decision，因此四库Q1不是单一同质干预。

K200：I四库 = 3/4×(-0.002131327112) + 1/4×(-0.027348351001) = -0.008435583084。BNCI加权项为-0.006837087750，占负交互点估计81.05%。这是固定估计量的算术拆分，不是decision loss的因果归因比例。

原冻结分析K800 interaction区间为[-0.00915086,+0.00006372]；新复算为[-0.0092754,-0.0000245]。端点几乎为零，bootstrap Monte Carlo波动会改变是否过零，不能宣称K800交互已获稳健确认。

## BNCI simple effects

| 对比 | K200差值[95%CI] | K800差值[95%CI] |
|---|---|---|
| A1_minus_A0_at_Q0 | +0.005434 [-0.011145, +0.022535] | -0.004213 [-0.020729, +0.009490] |
| A1_minus_A0_at_Q1 | -0.021914 [-0.037959, -0.002220] | -0.016789 [-0.022982, -0.010631] |
| Q1_minus_Q0_at_A0 | -0.017853 [-0.047909, +0.012927] | -0.056153 [-0.072118, -0.041607] |
| Q1_minus_Q0_at_A1 | -0.045201 [-0.080571, -0.014117] | -0.068729 [-0.079451, -0.057405] |

binary-only Q1@A0在K200/K800为+0.002728/+0.002753，区间分别[0.002041,0.003448]/[0.001631,0.003848]；小正收益不支持把prior本身解释成已证实失效。A1Q1对A0Q0三库差值在K200/K800为-0.000093/+0.003180，均跨零。

## 决策、预算和标签核验

BNCI每人29个物理测试decision，28个完整可评价，8人合计232/224，coverage=0.965517。全部物理组原生10重复，所以R10=hit@all。不是BrainSync九选一或独立成人确认。

Q1校准每人K200只有1个不同物理完整selection、90次draw；K800只有6个、180次draw。这些selection的统计独立性不保证。candidate trial draws为10800/21600，不能冒充不同物理selection数。K800比K200更差不能单凭组数解释。

| BNCI hit@all | K200 | K800 |
|---|---:|---:|---:|
| A0Q0 | 0.456897 | 0.715517 |
| A1Q1 | 0.275862 | 0.508621 |
| CE scratch (ScratchCE) | 0.387931 | 0.586207 |
| decision scratch (Scratch) | 0.340517 | 0.500000 |

历史结果页的两行scratch标签反了；冻结recipes.py:213-214和analyze_unified_results.py:34确认映射。bnci_hit_all.json:268/294分别是Scratch/ScratchCE记录。

A1Q1-A0Q0的hit差：K200 -0.181034[-0.318966,-0.047414]，K800 -0.206897[-0.288793,-0.116379]。以参与者为bootstrap单位，不能把232个decision当232个独立受试者。

## 共同K800队列

72人=BI2014a20+BI2015a31+BNCI8+Huebner13。A1Q1-A0Q0在同72人的K200为-0.009946[-0.022194,+0.001886]，K800为-0.015850[-0.022244,-0.009587]；跨K差为-0.005904[-0.018915,+0.007757]。因此描述上仍更差，但差中差跨零，不能确认预算升高导致整包进一步恶化。

## λ与执行身份

冻结builder声明λ候选{0,.001,.01,.1}，选择只在源内K200做。冻结runner在外折/layout中先选择一次，再将同一selected_weight写入A0/A1与各目标K（run_source_training_chain.py:688及799）。本地拉回汇总没有实际selected_weight、normalized_cells.json或各折selection.json，无法验证每折具体λ及是否各折恰相同；不能把代码设计冒称实际数值验证。

v4压缩源已现场核SHA256=c71a4d2109eaed23bea734c76845940bcf05d51a3af4478ac3c1db54fe75fd24；首折复用v2身份5701d51f04dc1111e30f1bb9058f5e8992383abae3860250a1c8b87e2c3bcbce。当前recipes.py、source_selection.py、run_source_training_chain.py与v4冻结包逐字节一致；其他当前dirty文件不能据此推定一致。基线commit f1a303c42c68173e8b1452ceb26bd84923b8a6aa仅为base_commit。

现存results/archive_receipt.json指向云端11.18GiB归档，SHA256=db3456fada03f6cc7177c225b822c4884aef4e2204c646218c029ef0bd99c790；D:/BCI搜索未发现该完整训练归档，故本轮不称重新核验原预测或48/48模型重放。原记录的source_jobs=48来自分析器len(target_jobs)，实际是目标分片job计数，不能解释为48次源拟合。

## EA共同实测通道的可行性补核

此节只读已有prepared元数据、真实探针和交付manifest，没有重新准备输入或训练。六槽顺序为Fz/P3/Pz/P4/PO8/Oz，均是各位置减实测Cz，定义见`src/data/source_records.py:39`。`channel_mask`/`observed`不能当作已完成所有逐trial mask核验。

| 来源 | 已读mask（同一六槽顺序） | 证据范围 |
|---|---|---|
| BI2013a | 011101，292条 | 09-09本地prepared的全部record级observed字段 |
| BI2014a | 011101，61条 | 同上 |
| BI2015a | 011111，123条 | 同上 |
| Chailloux2020 | 111111，382条 | 同上 |
| EPFLP300 | 111101，192条 | 同上 |
| GTN | 101000，245条 | 同上；仅Fz-Cz/Pz-Cz |
| Huebner2017 | 111100，342条 | 同上 |
| Speier2017 | 101011，60条 | 同上 |
| BrainForm | 101011 | 09-18 P01单记录真实探针；不能代表全部167条 |
| Cattan2019_VR | 111101 | 09-18 subject_01_VR真实探针 |
| BI2012 | 011100 | 09-18 subject_01 training真实探针；Fz为ground |
| BNCI2014-008 | 本轮未逐prepared核验 | 不用猜测补齐表格 |

前八库依据`experiments/runs/source_eog_20260909/local_prepared/<study>/preparation.json`中全部记录的`observed`字段。它们是历史源准备元数据，不能冒充09-18最终云prepared逐trial核验。后三个实测探针见`tmp/source_modules_20260918/real_probes.json`及`doc/evidence/source_modules_20260918.zh.md:44-48`。

**已核实探针给出的上界：** BrainForm的101011与BI2012的011100相交为001000，即仅Pz-Cz。四个普通整库留出与全十二库部署都保留这两个来源，所以要求所有来源每条记录都完整实测的统一空间至多一维；不能直接在该池承诺同一实测六维SPD协方差。此上界仍不是全量Pz覆盖保证。

**加入交付声明坏导后，合同推导交集为空：** `experiments/runs/cloud_delivery_20260918/manifests/BrainForm.json`中，167条记录有164条无显式坏导、3条例外：P03/20241021_130354排除Oz（第122-129行），P02/20241021_120406排除Fz及Pz（第1062-1070行），P14/20241105_125757排除Pz（第1709-1716行）。这些排除有波形审计依据，见`doc/assets/dataset_expansion_20260908/channel_exclusions.json`。

BrainForm loader在`src/data/source_parsers.py:185-198`读取`excluded_channels`并写入`raw.info["bads"]`；`src/data/source_records.py:135-146`将坏导从输入剔除后才构造六槽mask。故P02该记录按声明只能保留PO8-Cz/Oz-Cz，即000011；与GTN的101000相交为空。所有四个普通折、BI家族联合排除折和全源部署组均保留GTN及BrainForm，所以**若完整交付manifest按该合同全部执行，跨全部记录的共同实测通道交集在这些折中均为空**，而非稳定保有Pz-Cz。

此推导绑定实际准备源快照：`frozen/source_modules_20260918_acceptance/source.source-only.tar.gz`本轮SHA256复核为`bac9654edfe63610f4775aa34569efc524c782eac81a0496b35a9e0ac79a54f7`；上述两个loader文件当前字节与该准备快照一致。`experiments/runs/unified12_20260918/preparation_status.txt:11,15`分别记BrainForm167条完成、GTN245条完成。但本轮未读取09-18最终云prepared每条记录/每个trial的实际mask，故“交集为空”保持为交付manifest与冻结loader合同的推导，最终输入接线及逐行mask仍待核验。不能通过补零、把坏导改为实测、静默删源/删记录或只选择P01探针来宣称标准EA可直接覆盖原十二库池。
