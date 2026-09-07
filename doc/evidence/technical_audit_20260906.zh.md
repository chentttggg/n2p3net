# 全项目技术结构审计与整理

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-06。范围：当前dirty工作树、顶层入口、核心实现、测试、配置、指导和数据目录。
先读取四个skill：improve-codebase-architecture、codebase-design、eeg-p300-research、
scientific-critical-thinking；三个子agent分别核查训练、数据/工作站、runner，并独立交叉复核。
旧skill和旧指导中的算法主线没有被直接当成当前实现事实。

## 主要判断

当前可复用结构已经覆盖公开ERP输入、源预训练、本人离线校准、后续decision评估与候选
聚合。没有发现具体模块静态导入环，不能据此声称所有运行时路径均无问题。
主要缺陷是入口之间的合同联动、训练阶段不一致及对产品完成度的过度描述，
不是必须重写主干或重排全部源码目录。

审计起初发现BrainSync产品末段缺失：内部选配方、个体预测器持久化及无标签R5接口。
在[两版比较](brainsync_versions_20260906.zh.md)之后，用户授权整合主app；本轮已将这些
接口实现，并以实际Qt模拟采集和公开源模型加EOG跑通保存/重载/未知目标block推理。
当前结果见[app实施与验证](brainsync_app_integration_20260906.zh.md)。
逐SDK chunk推理、设备样本时钟映射和真人准确率仍未验证；4份历史候选记录未改写。

完整保留链条和目录职责已合入[技术结构](../technical_structure.zh.md)。公开BNCI的0.8136
仍是独立保存的开发读数，没有因为本次代码整理重新训练、覆盖或升级为成人性能。

## 发现与处理

| 级别 | 可复现问题 | 处理结果 |
|---|---|---|
| P1 | `SubjectAdapter`初次拟合使用AS-MMD，完整prefix refit却只用CE | 共享`_AdaptationSource`、`_training_loss`和批次规则；两阶段保留相同损失与源流，分别导出ledger |
| P1 | AS-MMD在17行、batch=8时尾批只有1行，实际抛ValueError | 将尾行并入前批；不丢校准数据，配置batch=1提前报错 |
| P1 | candidate/within的QC先拟合完整outer校准，再选择inner validation | 对尚未正确实现的QC+内部选轮数组合提前拒绝；promotion同步，其他合法组合保留 |
| P1 | 从仓库外直接启动候选CLI报`ModuleNotFoundError: experiments` | 复用既有ROOT/SRC路径初始化，独立子进程回归通过 |
| P2 | 候选/within评估把源训练与评估源码混成一个hash | runner、batch、promotion、manifest builder和analyzer区分两种声明，checkpoint仍按源训练身份验证 |
| P2 | zero-shot memo未绑定评估代码，可把旧logit标成新评估结果 | 缓存键加入评估archive hash，错配时重算；不改写历史分数 |
| P2 | BIDS TSV少字段抛AttributeError绕过skip策略，多字段被静默接受 | 在共用`_read_tsv`统一检查列数并抛ValueError；坏session记录失败，正常session继续 |
| P1 | 晚确认被当成非法数据，校准标签实际可用时间未进入split | 接入允许晚确认，显式记录max(end,confirmed)，早于校准就绪的测试保留失败分母 |
| P1 | EOG预算检查把复合decision group当session ID | 从calibration mask对应的结构化timeline提取真实session，合法EOG runner已跑通 |
| P2 | 多个runner重复或直接写JSON，序列化失败可能留下不完整结果 | 统一调用`research.execution.atomic_write_json`，反例验证旧结果保留且临时文件移除 |
| P2 | README、旧方案与最新产品目标冲突，端到端描述超过实际能力 | 重写入口文档、主指导、实施状态及采集配置，区分离线能力与实时缺口 |
| P3 | preprocess保留无调用的`_canonical`、`highpass`、`reject_epochs` | 删除死接口及其专属旧测试，使用既有canonical命名与连续滤波接口 |

源码证据：`src/transfer/subject_adapter.py`、`src/data/bids_eeg.py`、
`experiments/run_candidate_cross_decision.py`、`run_within_subject_transfer.py`、
`run_candidate_promotion_matrix.py`、`run_candidate_result_batch.py`、
`build_candidate_cross_decision_manifest.py`、`analyze_candidate_cross_decision.py`。
错误v2/v3注释与BrainSync报错同时改为当前v4合同。

## 数学与反例

AS-MMD已经声明的目标为

\[
L_e=CE_S+w_T(N_S,N_C,e)CE_C+\lambda_e\widehat{MMD}^2(S,C).
\]

完整校准refit可以改变`N_C`与对应权重，但不能无声删除源CE/MMD项。
无偏MMD的类内项包含`1/[m(m-1)]`，所以单样本批次不能计算。合并尾批保留行数和监督，
比静默drop_last更符合固定校准预算。

7个AS-MMD新反例先全部失败，修复后通过：从同一源模型出发，比较“内部选轮数再refit”
和“用选中轮数直接训练完整授权prefix”的所有权重、BN、预测与损失逐位一致；覆盖
frozen logits、live pooled source、live target-calibration buffer、校准外后缀和尾批。
原`prefix_mask`范围守卫保留，inner embargo仍可回到已授权的完整校准范围。

BIDS反例分别复现缺字段`AttributeError`和多字段`DID NOT RAISE`，修复后通过真实
BrainSync builder及resilient接入检验；使用真实producer函数生成的合成EDF也完成往返。
这不是实机采集证据。

源码身份反例贯穿实际runner→manifest builder→analyzer，覆盖同snapshot默认、不同
训练/评估snapshot、错误绑定拒绝、batch一致性、promotion转发及memo失效。
JSON反例在已有有效结果上尝试写NaN，验证失败不覆盖旧文件。

学习型预处理的独立性依据也在线核对了[scikit-learn官方常见陷阱](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)：
变换的拟合必须只使用训练部分。在嵌套选择中，这一原则同样适用于inner train/validation。
本次QC问题是内部选择独立性缺陷，未观察到外部test参与该拟合；两者不能混称。
已核对严格BNCI实验脚本没有启用AS-MMD或此QC路径，不能据本轮发现否定其历史读数。

## 结构盘点

[AST盘点](../assets/technical_audit_20260906/structure.json)记录逐文件hash、大小、行数、
第一方导入位置、条件/函数内导入和强连通分量。范围为`src`及顶层`experiments`/`tests`，
不解析raw、cache、冻结源码或一次性runs脚本。

最初75份src、35份入口/辅助脚本、62份测试模块均可解析；追加个体模型模块后，
最终文件数与hash以刷新后的AST记录为准。当前具体模块图未发现静态环。
`data.epochs`、`data.contract`、`data.channel`有大量真实调用，是必须保持统一的接口。
`train`与`baselines`在包级双向引用不等于模块循环：factory组装baseline，baseline调用runtime。
没有发现src反向导入experiments。

大文件包括`baselines.deep`、`data.moabb`、`data.raw_artifacts`、`baselines.evaluate`和
`data.epochs`。本轮没有用行数阈值强拆它们；下一次重构应依据具体变更原因和接口负担。

## 工作树与归档

基线commit为`c8ee57399f3a081a14bab4ab914f07f86fb02184`，仅标识父提交。
修改前296份实际非忽略文件已压缩并逐成员校验：
[快照manifest](../../frozen/technical_audit_before_20260906.manifest.json)，archive SHA256：
`96641cdd4f4a98c9b2c5ee52a68aba5d76945f2da7c22f737cf648e45c5b24b6`。

从活动目录移除13份已被取代的指导/方案、2份日期化云模板，以及仍断言128 samples和
8 EEG的旧`patch.sh`；这些文件的原文仍在上述物理压缩包内。两个仅检验旧模板的测试
随模板删除，通用promotion模板和真实合同测试保留。

删除`tmp/l0_bi2014a_causal_v3_64subj.npz`的重复副本，保留`cloud_inputs/`同名文件；
两者SHA256均为`842466e45d63e017704a2fe4ebc1ee550c54eff7b400bdd27c7454bb4ae69fff`，
释放659577127字节。不同native/CAR/EA派生cache不是重复数据，均保持原身份。
其他唯一raw、cache、论文、旧结果和传输包未按名字批量删除；其层级和用途已写入技术指南。

没有reset、checkout、提交或覆盖先前研究改动。采集端先只读比较；后续按用户授权只整合newest，
旧目录和原始记录保留。未重启云训练。

## 验证记录

主app整合后的最终回归为后端440项、前端118项全部通过；实际Qt源模型+EOG+pending目标
全链也通过，见[整机记录](brainsync_app_integration_20260906.zh.md)。以下412项是整合前阶段记录，
不与最终用例重复相加。

最终15份测试文件的定向集成验证为**412通过、0失败、0跳过，70.12秒**，
写入[JUnit记录](../assets/technical_audit_20260906/focused_tests.xml)。
覆盖本轮更改的数据、校准、AS-MMD、候选评估、batch和promotion接口，不依赖全量旧测试。
已设置当前producer目录，跨项目合成EDF往返用例确实执行。5条warning来自合成EDF
末块填充和极小LDA fixture，不是实机结果，也未静默忽略失败。
静态检查覆盖活动src、experiments与tests；文档实际链接用Markdown解析器检查。
项目`.venv`的113个已安装包兼容检查通过，`uv lock --check --offline`通过，
没有为了本次审计改变依赖或锁文件。

## 剩余接口

1. **设备与逐chunk推理**：block完成后的无标签45-flash决策和个体推理包已经实现；
   每SDK chunk的状态化模型输入和设备时钟映射仍需另外验证，不能混称。
2. **内部学习型QC生命周期**：旧QC组合已明确拒绝，今后支持需inner train范围。
   新app的自动EOG采用已实现的inner-train拟合后冻结策略，不在refit时偷偷换处理器。
3. **历史记录准入**：旧PO7/EOG标签与未配准坐标不靠名字猜测迁移，保留raw并单独核对。
4. **实际执行源码核验**：snapshot验证器检查声明的归档完整性，未证明当前进程使用该归档。
   文案已更正；新实验需从冻结包运行或独立逐文件核验，历史现场核验按各自证据保留。
5. **真实效能与时长**：新版发现4份尚未准入的候选记录；AUC、眼电归因、重新佩戴/跨日、实时延迟和
   猜数字准确率仍待采集。4050个闪烁不等于4050次独立decision，1.5秒SOA校准约101分钟。

这些是明确记录的后续实现与实测工作，不计入本轮已完成修复，也不填充假在线接口掩盖缺口。
