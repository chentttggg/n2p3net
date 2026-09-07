# 跨数据集方案实施清单

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-06。配套 [研究设计](architecture_decision_cross_dataset_20260906.zh.md)。
本轮完成文档/数据普查/反例，下面未标完成的项目不是已有功能或已启动训练。

## 1. 分阶段交付

| 阶段 | 具体执行 | 验收产物 | 当前状态 |
|---|---|---|---|
| P0 | 工作树、活动指导、原始目录、真实接口及文献核对 | 数据盘点、7组反例、设计/实施/数学记录 | 完成 |
| P1 | BI2015a、Huebner准入；Mainsah来源/参考核查；旧分发去重 | 原始manifest、identity table、物理信号/标签/预处理核查报告 | 待实施 |
| P2 | 每库生成当前causal binary cache；产品mask/Cz变换；固定划分 | attested cache、同一row identity的多视图、cal/test清单 | 新数据待构建；复用BI/BNCI现存链 |
| P3 | 整库留出、多run校准、显式类别权重/预算的最小接口补齐 | 真实builder反例测试、CLI help、checkpoint contract | 待实施 |
| P4 | BI↔BNCI复现；开发四折单源/多源与强target-only | 配对预测、失败账本、实际source/target预算 | 设计已定，未开跑 |
| P5 | source-only选择增量机制，三seed复核，冻结配方 | source-only选择记录、整BI族留出、冻结评估计划 | 待P4 |
| P6 | Mainsah整库评估；之后独立生成全公开源部署候选 | 两种不同用途checkpoint及来源，不交叉冒用结果 | 待准入/冻结 |
| P7 | 真实BrainSync校准、后续未知目标决策及跨日确认 | 成人operational指标和设备replay | 无正式数据，保留未完成 |

先跑简洁路径；单个新增数据源暂时不合格时，其余合格阶段继续，但不足以支撑的
外层折明确缺失，不能把缩小的池仍命名为完成四库/跨族验证。

## 2. 数据链复用与必要修改

| 所有权位置 | 现有能力 | 需要落实的增量 |
|---|---|---|
| `src/data/raw_artifacts.py` | ZIP/source文件manifest、checksum、只读快照 | 给新库真实文件生成manifest；不把本轮目录盘点当attestation |
| `src/data/moabb.py`、`experiments/prepare_eeg_dataset.py` | pinned MOABB upstream parser、causal raw处理、manifest入口 | BI2015a走现有MOABB；NEMAR不冒充upstream MAT解析输入 |
| `src/data/manifest.py`、`src/data/dataset.py` | MNE支持的BDF/EDF+TSV、显式label map/session/run/reference | NEMAR采用声明式manifest；来源转换核查和hash进入原有provenance链 |
| `src/data/bids_eeg.py` | 当前产品BIDS合同含candidate/rest/decision要求 | 不拿该产品合同强迫公共binary源伪造candidate；只复用适用的解析/验证组件 |
| `src/data/identity.py`、`lineage.py` | authority `participant_key`、父来源、排他检查 | Huebner两分发同身份；同源导出不变成新域；BI未知身份不伪装已排除 |
| `src/data/domain.py`、`epochs.py` | binary投影、namespace、mask、参考变换及拼接 | 每库单独物理适配后再并库；每折source行覆盖检查；raw/coreg/转换记录保留 |
| `experiments/prepare_multidomain_source.py` | 多source显式域轴、measured_channel/Cz、mask | 复用CLI；namespace不能代替study identity；不改变target模板语义 |

Huebner / Mainsah每条manifest采用实际 `*_eeg.bdf` + 对应 `*_events.tsv`，
`label_map={"Target":1,"NonTarget":0}`；`sample`优先并与onset/sfreq交叉核对。
Mainsah代表性转换BDF与原始EDF的采样率/长度存在差别，转换核实通过之前，
该manifest只是待构建方案，不能直接将原始sidecar与转换信号混配后视为已准入。
辅助刺激列只进入provenance/决策解析，不能进入EEG。NEMAR与sourcedata只选择一个
已核实的规范信号分发用于训练；另一份作为provenance校验，不重复计样本。

当前generic event reader可读取trial_type，而NEMAR sidecar的duration可能记录epoch区间
而非物理flash时长；校准/推理成本不能直接加总该列。时间成本从源时钟和原始协议计算。
原生未知参考保持未知，待证据后选择显式共同anchor；不要在builder里写n/a→Cz别名。

## 3. 训练与评估接口

| 所有权位置 | 复用 | 有限增量/缺口 |
|---|---|---|
| `experiments/run_pretrain_supervised.py` | checkpoint、source holdout、group早停/refit、多域risk、source batch | 增加显式类别权重与固定update预算；完整数据集/研究家族排除写入真实训练合同 |
| `src/baselines/deep.py` | `resolve_source_risk_weights`、带权CE、BN、source训练 | 绑定risk/sampler/update语义；需要跨source域宏平均选择时扩展现有validation归约 |
| `src/train/runtime.py` | scheduler、共享batch、资源预算、compile/fused | 记录effective batch与实际updates，OOM不得悄悄改变科学预算 |
| `src/transfer/checkpoint.py` | identity/geometry/reference/preprocessing/训练head门禁 | 增加整库排他验证应走现有contract，不能只检查单目标人 |
| `src/transfer/binary.py`、`experiments/run_binary_subject_transfer.py` | 单recording固定prefix/suffix、scratch、source feature LDA | 增加真实多run协议：显式有序recording及C/T行清单；不要解除单stream检查后直接拼接 |
| `src/transfer/subject_adapter.py` | head/full fine、normalization、BN、`prefix_mask` refit | 复用calibration-only拟合；支持小预算下预声明fallback而不访问test |
| `src/research/evaluation.py`、`contracts.py` | 真正evaluation builder、源码/模型身份 | 将domain exclusion、recording分组、失败、预算、选择范围加入原有结构与校验 |
| `run_candidate_cross_decision.py`、`run_brainsync_cross_decision.py` | 真实candidate/目标隔离/聚合/成本 | binary source不依赖它们；仅在目标有可靠membership时进入该链 |

`--selection-domain`现有语义是**source内一个域的group分组epoch选择**，不是整域
从source训练排除的内层LODO。实现新选择时不得重解释旧开关；优先复用独立source job
与已有结果合同组合选择，把明确排除域与selection scope写入已有plan/artifact结构。

当前source runner还会以所选epoch从头refit全部source；这是合法source内部流程。
但若新增固定updates臂，选择过程与最终refit总updates需分别记录，不能把两者相加后
宣称每个模型经历了同样优化步数，也不能静默用最后epoch替换best epoch。

### 3.1 调度伪代码

```python
dev_sources = admitted_studies(plan.development_pool)
for target_study in plan.development_targets:
    source_studies = dev_sources - {target_study}
    source_rows = exclude_authority_and_study(source_studies, target_study)
    assert_source_channel_coverage(source_rows, plan.input_view)
    for seed in plan.seeds:
        model = fit_source(source_rows, source_only_selection=plan.source_selection)
        for participant in requested_participants(target_study):
            split = build_recording_calibration_test_plan(participant, plan.budgets)
            for budget in plan.budgets:
                for arm in plan.core_arms:
                    fitted = fit_on_calibration_only(model, arm, split.calibration(budget))
                    write_predictions_and_contract(fitted, split.common_test)
freeze_method_using_development_results()
refit_on_all_development_sources()
evaluate_predeclared_mainsah_protocol_once()
```

这是协议伪代码，函数名不是已存在API。正式实现优先扩展现有contract/builder/runner，
不另造抽象Runner框架，也不根据伪代码创建一串仅转发参数的新模块。

## 4. 反例测试与验收

算法实施后更新下列模块对应tests，移除仅验证已退役语义的断言；保留仍成立的不变量。
本轮没有生产算法改动，未删除旧test或改动并发工作树里的训练实现。

| 反例 | 必须观察到的行为 | 关联测试位置 |
|---|---|---|
| target-study中的另一被试混入source | 整库验证拒绝，即使当前目标人未重叠 | pretrain/evaluation builder测试 |
| Huebner换namespace后同时作两域 | canonical identity仍冲突，数据量不翻倍 | identity/lineage/manifest测试 |
| outer/inner排除后唯一Fz或PO8来源消失 | 训练前拒绝；不能使用target观测并集 | pretrain/source mask测试 |
| 缺失列参考变换后产生负Cz波形 | 仍为mask=false且零占位 | domain/prepare_multidomain测试 |
| 两份NEMAR的Target码与MOABB码不同 | 映射后每trial真值一致；绝不硬编码2=Target到全部raw | manifest/moabb测试 |
| run时钟重置、run-10字典序早于run-2 | 依赖原始顺序清单；无法确定顺序则拒绝该时序主张 | binary多recording测试 |
| 改动later test振幅、标签、目标身份 | 所有fitted统计、epoch选择、BN、refit结果不变 | binary/SubjectAdapter测试 |
| K增大改变test队列 | 同一评估计划内T的row digest严格相同 | binary/候选split测试 |
| 同一batch拆成不同microbatch | loss/gradient符合声明归约，不改变目标风险 | deep objective/runtime测试 |
| scratch复用source BN或normalizer | 源参数/统计任意扰动不改变scratch预测 | binary/BrainSync控制测试 |
| 无两类校准或拟合失败 | 预声明fallback/失败账本保留分母，无test补采样 | outcome/evaluation测试 |
| 蒸馏teacher读取外层目标 | 真实checkpoint identity+study排他拒绝 | teacher实现时新增对应测试 |

每个阶段仅运行相关反例及必要集成，不能用旧全量tests通过替代新协议验证。
执行数据与源码冻结为压缩archive和checksum，不覆盖本轮之前已有数据/模型/并发修改。

## 5. 执行规模与停止规则

首批四个目标域、三个seed、六个核心方法、三个K，共216个“域×seed×方法×K”汇总组合；
实际拟合cell还乘各库participant/run，Z0和确定性LDA可复用，不能把216当216名独立样本。
源单库比较额外按合法source roster计数；新增norm/BN/EA/蒸馏不是默认全笛卡尔积。

先在每库少量source参与者验证CPU构建、标签、事件、mask与一轮forward/backward；
通过后构建完整cache，再在现有GPU环境开展计划。当前本地.venv实测Torch2.13.0+cpu，
本轮未连接云端，也未把历史RTX5090可用性当当前事实。训练前重新读取设备、空间与进程。

若新机制在开发库没有足够的附加收益/稳定性，保留plain候选并记录未决；
若输入/身份不成立，停止相关折并继续其他合法折。没有正式BrainSync数据不制造产品性能。
