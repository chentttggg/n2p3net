# 研究与工程证据

本目录保存独立实验、推导和审计。正文中的“当前”“下一步”指写作时点，
不覆盖[研究规范](../research_program.zh.md)或[当前执行计划](../roadmap.zh.md)。
原始预测、JSON、日志、图片及数据留在各自结果目录和`../assets/`，本次没有改写。

## 先看哪些结果

| 问题 | 证据 | 解释边界 |
|---|---|---|
| 工频陷波是否真正加入算法 | [陷波实施与验证](brainsync_notch_20260909.zh.md) | BrainSync默认50Hz/Q30；实际滤波、恢复证书和模型签名联动，真实残留及已知ERP保真核验；未测识别率收益 |
| P001新增两份数据质量如何 | [六份记录质量复核](brainsync_p001_quality_20260909.zh.md) | 新100906原生QC保留340/450、七导齐全300；新103053存在同步瞬态、P003身份待核；无训练或QC重拟合 |
| 全库代码质量与 Module 设计是否可靠 | [代码质量核查与修复](code_quality_audit_20260908.zh.md) | 核查发现的七项问题已在原模块修复；追加CPU/CUDA、产物完整性和浏览器反例，最终验收见报告 |
| GTN和更多公开数据能否扩入域迁移 | [扩源审计与新增下载](dataset_expansion_audit_20260908.zh.md) | 四套新分发4.22 GiB及Speier原始DAT 483.4 MiB已核；BrainForm 24人99953个完整窗口与通道排除视图；BI2015b已删除；Lee两会话BDF及元数据38.63 GiB已启动下载，完整校验见账本；仍需逐折QC与接入 |
| 相对幅度是否把会话差异当作坏信号 | [QC重设计与验证](brainsync_qc_redesign_20260908.zh.md) | 已修正个人硬拒语义，57拟合/114预测复核通过；Pz保留恢复，R5准确率仍弱 |
| 其余四项前置质量问题 | [持续趋势、采样支持与处理对照](brainsync_qc_remaining_20260908.zh.md) | 已联动并完成12条件72拟合/144预测核验；趋势不作伪迹认证，恢复等待不任意缩短 |
| REF实际接法是什么，是否继续Cz重参考 | [左耳垂更正与算法裁决](brainsync_reference_20260908.zh.md) | 用户确认左耳耳垂耳夹；9目录/24元数据更正、18 EDF不变；原生七导本人基准与显式Cz迁移分开，未宣称七导已胜出 |
| 四库新源用于第二、第四次校准，第三、第五次分别测试 | [新源校准结果](brainsync_s1_calibration_20260908.zh.md) | 云端19拟合/38预测；独立会话留出、675训练epoch；新源未稳定优于scratch，严格R5均拒判 |
| BrainSync为何AUC低，迁移负效应来自什么 | [原始数据与模型归因核查](brainsync_auc_cause_audit_20260908.zh.md) | 35模型/56预测重载、四份EDF/六份源MAT；尺度与跨会话变化实测，单人负差不等于公开或成人总体负迁移 |
| 第三、第五份能保留多少，补充训练如何避免放宽质量标准 | [局部恢复与补充训练](brainsync_recovery_20260908.zh.md) | 247/450、229/450 epoch；逐导掩码、冻结QC、原R5分母；62反例中六个慢斜坡仍漏检 |
| 提高目标轮数到90遍是否改善结果 | [60对90遍结果](target_epoch90_20260908.zh.md) | 610拟合0失败，1525产物复核；全网K200/K800 AUC增0.0086/0.0096，K2000收益很小 |
| 早停全量是否改善准确率、是否减少总工作 | [早停全量结果](target_early_stopping_results_20260908.zh.md) | 123人3173评分/3050模型复核；当前CE选轮与patience降低AUC，含refit总更新增加；单seed公开开发 |
| 时钟、分段、连续污染与单 HEOG 如何处理 | [本次实施与验证](brainsync_quality_implementation_20260907.zh.md) | 软件及旧 P001 开发；450/450 排程支持已验证，恢复阈值/实机光学时钟/成人效能未确认 |
| 目标早停在哪一阶段做、是否节省总成本 | [早停消融实现与试跑](target_early_stopping_20260907.zh.md) | ES三模型×三策略，8人218评分复核；含refit总更新未减少，不能把试跑当总体结论 |
| P001 真实记录与云端本人迁移 | [P001 先导训练](brainsync_p001_pilot_20260907.zh.md) | 旧5会话pilot含已拒绝的0903_170721，缓存与个体模型已撤下；分数仅保留历史含义 |
| batch512全量各组件是否值得保留 | [完成结果与分析](training_chain_results_20260907.zh.md) | 95源任务、45134外层+7050内层预测复核；六导完整微调有小K收益，复杂源统计/选模增益有限；单seed开发 |
| compact512及后续优化全量实现 | [训练链条实现与验证](training_chain_implementation_20260907.zh.md) | compact1953/1953复核；扩展全量也已完成，历次快照、并发调度及精确复用见记录 |
| 新标准K0/200/800/2000完成结果 | [标准预算完整分析](four_study_standard_budgets_20260907.zh.md) | 123人1648单元逐预测复算；标准K、共同被试和回退档分开；单seed开发 |
| 四库迁移在各校准预算下有多少收益 | [单seed完整结果分析](four_study_results_20260907.zh.md)；[逐人详细矩阵](../../experiments/runs/four_dataset_20260907/analysis/detailed/report.zh.md) | 123×18矩阵、分布尾部、重复负迁移、漏检/误报与目标训练曲线；逐预测复算 |
| 四库整库留出及并发性能 | [四库运行](four_study_launch_20260907.zh.md) | 123人、单seed、六臂、2214单元完成；开发结果，非成人确认 |
| Mainsah参考和EDF/BDF转换是否可信 | [Mainsah准入](mainsah_s2_admission_20260907.md) | 288份完整SHA/波形/标签核验，24人仍封存未评分 |
| BI62-64 是否为作者拒收数据，后续是否使用 | [作者依据与删除记录](bi61_author_exclusion_20260906.zh.md) | 官方报告第4/9页明确拒收；活动队列永久01-61，旧64人结果不重标 |
| chance AUC 是否来自标签、时轴或预处理错位 | [跨库输入对齐审计](cross_dataset_input_alignment_audit_20260906.zh.md) | 全量缓存检查、两份 raw 重放；发现源尺度集中和时轴元数据风险，未定位唯一迁移失败根因 |
| 微调后接LDA还是先用LDA初始化，大K是否更好 | [LDA顺序与预算曲线](lda_head_order_20260906.zh.md) | 新61人准入，77组/231份预测；大K收益明显，LDA未稳定优于原head；旧64人结果单列历史 |
| 整库未见时迁移是否成立，GPU为何空闲 | [BI↔BNCI首轮](cross_dataset_start_20260906.zh.md) | 2,160单元，前缀统计/BN归因、1,186参数及CPU/GPU实测；单seed开发 |
| 公开ERP源与本人校准是否有用 | [主线实验结果](p300_transfer_results_20260906.zh.md)；[执行过程](p300_transfer_execution_plan_20260906.zh.md) | 包含pilot、配方诊断与被观察过的development，不合并成确认成绩 |
| 64人、通道和校准单位为什么不同 | [BI64结果](bi64_protocol_factorial_results_20260906.zh.md)；[配对协议](bi64_protocol_factorial_20260906.zh.md) | 源63→1与后续本人校准分开，16导/5导及共同test配对解释 |
| 内部选配方是否还能泛化到未校准目标 | [BNCI严格结果](bnci_unseen_target_results_20260906.zh.md)；[实验协议](bnci_unseen_target_plan_20260906.zh.md) | K600 AUC=0.8136，zero-shot=0.7782；仍为公开开发，不是成人九选一 |
| AUC是否受泄漏或选择偏差影响 | [初次AUC审计](p300_auc_leakage_audit_20260906.zh.md)；[后续refit复核与更正](finetune_leakage_review_20260906.zh.md) | 性能方向不能排除泄漏；混合均值、test选方与代码范围分别核查 |
| 早期迁移机制与长R读数 | [dev64结果](cloud_ablation_v2_results_20260904.zh.md)；[逐R与分母复算](hit_at_r_valid_denominator_20260904.zh.md) | CI含0不证明机制等效；BI长R与候选身份限制不能变成binary源准入门 |

早期结果中的R3/R8属于各自协议，当前产品主R5由研究规范定义。
BI K192强配方的0.8150是同一开发后缀选方并评估的读数；后续审计已纠正部分
BI/BNCI网格行数和混合均值，比较时以具体报告的真实队列为准，不照抄早期对话表格。

## 设计、数学与外部依据

| 材料 | 用途 |
|---|---|
| [跨库研究设计](architecture_decision_cross_dataset_20260906.zh.md) | 数据准入、空间桥接、目标风险、核心对照与外部评估的详细设计依据 |
| [跨库接口规格](implementation_plan_cross_dataset_20260906.zh.md) | 真实模块、必要增量及反例；状态以当前执行计划为准 |
| [跨库数学验证](math_verification_cross_dataset_20260906.zh.md) | 权重/梯度、对齐和参考反例、来源与split验证 |
| [72项候选实验](research_experiment_catalog_20260906.zh.md) | 条件化备选方案，不要求全部执行，不覆盖已完成状态 |
| [来源与文献核查](research_evidence_audit_20260906.zh.md) | 原始报告、MOABB、信号处理与统计依据，按核查等级使用 |
| [BI家族普查](bi_family_transfer_census_20260904.zh.md) | 六库通道、重复数和身份线索；旧candidate优先路线已被binary主线替代 |
| [CAR数学审计](car_math_audit_20260904.zh.md) | 参考投影的信息损失及两导反例，不把CAR当成完成域对齐 |
| [主链公式一手来源核查](domain_transfer_chain_primary_source_audit_20260904.zh.md) | 算法命名、公式与实现的可追溯性，缺陷是否修复需查后续源码 |
| [BIOT适用性审查](biot_applicability_assessment_20260904.zh.md) | 特定checkpoint、时窗与蒙太奇限制，不外推为全部基础模型不适用 |

## 工程、设备与数据

| 材料 | 用途 |
|---|---|
| [BrainSync 窗口与时轴核查](brainsync_epoch_timing_audit_20260907.zh.md) | 五份自采会话、2205 epoch 复算；固定输入、整段读出与 LMBC，采样零点和连续性缺口 |
| [第三、第五会话瞬态诊断与第一份撤回](brainsync_transient_quality_20260907.zh.md) | 第三份14.2 mV启动阶跃与IIR长尾；第五份局部污染/共同参考；保留两份待QC，旧模型物理归档 |
| [连续质量与滤波状态研究方案](brainsync_signal_quality_design_20260907.zh.md) | 文献/10组反例/实施顺序；原始污染区间、状态恢复、时钟和个人QC；更正休息后首trial基线缺口，方案尚未部署 |
| [技术审计](technical_audit_20260906.zh.md) | 结构、合同、调用链修复及阶段测试证据 |
| [BrainSync版本比较](brainsync_versions_20260906.zh.md) | 整合前两份源码优缺点与历史raw候选记录 |
| [工作站整合验收](brainsync_app_integration_20260906.zh.md) | Qt模拟全链、截图、前后端测试及交付快照 |
| [阻抗显示核查](impedance_display_review_20260906.zh.md) | 模拟范围与质量语义修正，真实设备测量仍待核实 |
| [设备与性能测量](device-portability.md) | 历史5090冷/热编译、并发和搬运记录；当前策略见工程指南 |
| [工作树清理与数据保留](workspace_cleanup_20260906.zh.md) | 16,389份历史文件、四份cache和结果的实际去向及SHA-256 |
| [工作树与BCI目录整理](workspace_cleanup_20260908.zh.md) | 4,816份临时/缓存文件压缩校验，16份旧传输文件集中保留；活动脚本及三个工作树原位保全 |
| [云端数据盘清理与保全](cloud_disk_cleanup_20260908.zh.md) | 释放16.843 GiB；62,256份历史文件压缩校验后退役，当前输入、模型与结果保全 |

## 被合并或退役的指导

2026-09-07用户将巨大训练链矩阵精简为compact512。替换前的九份活动文档已逐成员SHA-256核验，
见[归档清单](../../frozen/compact512_redesign_20260907/manifest.json)，本地压缩包为
`frozen/compact512_redesign_20260907/active_documents_before_compact512.zip`。
旧S1的32次、S2的33次及E2源训练/选模任务已取消本轮排程；能力保留不等于正式完成。

本轮修改前的37份文档（含项目根README）完整保存在
本地原文压缩包`frozen/docs_before_consolidation_20260906.zip`（[分发说明](../../frozen/README.md)），
[迁移清单](../../frozen/docs_consolidation_20260906.manifest.json)逐份记录原路径、SHA-256及当前去向。
25份保留报告只增加历史提示并调整可点击链接，原始字节在压缩包中可核验。
结果中的内嵌旧路径不回写，按manifest查找报告新位置。

| 原文件 | 处置理由与当前入口 |
|---|---|
| `mission.md` | 产品目标并入研究规范，删除重复权威 |
| `techstack.md`、`CODING_WORKFLOW.md` | 环境与工程纪律并入工程指南 |
| `critical_architecture_review_20260905.zh.md` | 探索性批判的待验证项已由后续结果和执行计划接管，原文压缩保留 |
| `evidence_floor_parameter_20260904.zh.md` | 包含已退役的旧门槛与后续发射计划；有效时间/分母约束由输入合同及研究规范维护 |
| `cross_dataset_bi_gtn_ruling_20260904.zh.md` | BI→儿童三导GTN的旧主线裁决不再指导成人BrainSync，CAR实测另有保留报告 |
| `paradigm_domain_transfer_audit_20260904.zh.md` | 旧框架状态/R8与GTN主线叙述已被实际技术结构、binary主线及后续审计替代 |

此前退役指导仍在[技术审计前归档](../../frozen/technical_audit_before_20260906.manifest.json)，
更早活动指导在`frozen/guidance_before_binary_mainline_20260906.zip`，
SHA-256为`6d0f5bee635dbf5287cd4ff39f00dd270149bd844178b9f7b2a4611f162d00ad`。
归档只用于追溯，不重新放回当前CLI、模型输入或默认配置。
