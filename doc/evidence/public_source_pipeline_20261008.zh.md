# 当前公共源程序：实施、质量与实际输入准备

本轮以独立C/test、多专家概率程序和三完整配对seed为当前范式。2026-10-08用户允许放弃几周前旧试验方案；历史B7剂量48、SupCon/MMD、固定60/90矩阵保留原证据，不作为本轮必跑任务或数值默认的依据。本文验收源程序与准备，不宣称目标效能或全部15块完成。

## 独立选择与完整重训

`source_program=independent_selection_refit`在原source-chain执行`P2_source` plain CE。四outer整库留出BI2014a、BI2015a、BNCI2014-008、Huebner2017，每折实际其余十一库。配对source/target seeds为20260918、20260919、20260920，不展开3×3组合。每seed独立留出source participant、记录验证曲线、选轮，再fresh same-seed完整source refit；12个nominal checkpoint对应首次24个logical NN fit，实际更新与物理复用分列。

源风险为库等权、库内participant等权、观察行内等权CE。显式`source_config={pos_weight:1,val_groups_min:2,val_groups_max:null}`，null从实际admitted authority解析最大单库人数，不从文件数猜。比例.2不再被旧12人上限截断，GTN245对应49个held participant。独立精确反例核实旧β8按被试class-mass归一后再次倾斜被试混合，减ln8不能恢复原等人分布；旧模型保留原producer，不签β1。

source有限开发程序预声明cap32、patience6、绝对CE min_delta1e-6、validation fraction.2、batch512。检查周期是`ceil(inner_rows/512)`次该uniform sampling更新，不保证每行恰遍历一次；cap命中为右删失训练范围，不称32最优或已收敛。inner/refit分别绑定实际rows/participants、norm、BN、初始化、optimizer updates/draws与执行snapshot。

`SourceSeedCatalog`/共享validator核实际来源；新公共消费者显式传`expected_source_config`，不能靠程序同名混入旧β8。generic consumer仍按原合同重放旧producer。完整stopping关系由实际程序/曲线验证，不从三个config字段反推。

## 调度、恢复和质量

完整outer group一次加载视图、依次执行三独立seed。准备、合并、加载、拟合共享ready DAG、CPU/GPU budget及HostMemoryAdmission；25 CPU为4×6原生线程加1调度，单RTX5090的4 GPU worker为任务容量，容量不等于实际利用率。依赖末端与group内顺序分别解释。

独立两轴关闭五项真实缺口：producer/catalog准入后才计completed；变预算续跑复用原冻结完整view；ready cache绑定本次manifest SHA；CPUbudget2不可派发时完整失败退出；失败fit计划上界与实际/未知compute分列。已返回fit的物理成本保留，即使后续producer准入失败。

最终14项必要定向验证、Ruff和diff检查通过；一个β1 tiny raw→inner/refit→保存/Catalog真实probe及245 authority→49held反例通过。software probe采用自身2周期程序，不签为公共32周期已跑。Spec/Standards均零未关闭关键发现。旧direct/DAG三seed真实运行保留不同archive、1124证据文件；新probe31文件另留，未重复旧六次NN fit或全集。

[质量与整合凭证](../assets/public_source_pipeline_20261008/integration_receipt.json)核主树14文件与受审bytes一致。作者commit79b3b42，主树整合a297555；实际云源码绑定[436成员物理冻结](../../frozen/public_source_pipeline_20261008/source.manifest.json)SHA `fd62c48b37de5a569ad2be629e61c508b931dc9b2e80bd2d3147cee2e25e94ea`，3f38fe12仅dirty冻结的base commit。归档/本地镜像一致，不用后续commit改签执行身份。

## 实际输入准备与执行边界

新输入自声明finite zero-phase、128Hz/179、−200..1200ms、0.1–30Hz、notch50/Q30、baseline none；父mean_only声明/缓存保留，不用命令覆盖冲突。源父1964明确准入1963：仅BI2012 subject3 online的P4全97252样本有限且0被排除，同人training保留。GTN245个NIX唯一key、manifest subject、TXT stem实际一致，245 metadata SHA重核；论文250独立参与者数据folder支撑源内匿名authority，全局person linkage未独立验证，不作人口属性聚类。

真实准备由18dafd6冻结源码的detached owner运行，系统盘保存环境/执行源码，数据盘保存原输入、新cache及收据。2026-10-08 04:52:17 UTC终态16/16成功、失败0：十二库1963记录、1430742 epoch；四库534条target记录按原协议生成61/41/8/13、合计123 participant group。真实准备逐条核raw/member SHA、companions、finite/nonconstant和实际预处理，先前header/CRC不替代该关。

准备25CPU为1调度+4×(1 CLI控制器+5 record worker)，数值进程各1原生线程。完成后仅通过实际输入/本任务产物的只读`POSIX_FADV_DONTNEED`提示释放干净文件页，size/mtime保持；未删数据、重置全局缓存或改模型。prepared method保持18dafd6实际runner/data身份，新训练消费已准入完整cache，不改签fd62。

完整12 source程序于2026-10-08 05:09:35 UTC从fd62 materialized source实际启动，解析25CPU/4GPU slots/4source groups；05:59:33 UTC因caller非法scope结束，四view完整保留。修scope后2394直接消费四view，06:53:03 UTC因worker突然终止失败，原始原因未知、已完成checkpoint0；[终态与完整输入基准](public_source_throughput_20261008.zh.md#完整输入基准与后续源运行终态)绑定26份原件与实际archive。字节准入/副本优化及退出诊断完成后恢复完整程序。目标fixed-mass、binary posterior、共享Cz raw空间和温度子集已由[当前消费者切片](current_public_consumers_20261008.zh.md)完成软件验收；公共效能、B05–B11后续机制和B13新成人确认各自验收。
