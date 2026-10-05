# 候选多专家主线：代码实施与质量验收

日期：2026-10-05。实施基线为隔离分支 `a6a3da5`，承接[分块科研方案](deep_research_plan_20261005.zh.md)。原 `D:/BCI/n2p3-net` 保持并发工作；本轮代码、测试和证据在 `codex/deep-research-plan` 交付。下述软件能力不等同于完整公共矩阵或成人确认。

## 已落实的代码行为

| 任务 | 实际 owner | 当前已验证行为 | 仍需后续证据 |
|---|---|---|---|
| B00 评价与风险 | `models.candidate_evidence`、`research.candidate_metrics` | 不可变无标签概率证据、固定 requested roster、seed 在人内平均、预测/概率覆盖分开、tie/fitfail 保留、严格单臂/配对宏NLL、救回/损害及开发门 | 独立公共数据的参与者级效应与完整流程随机性 |
| B01 物理与数值视图 | `transfer.ensemble_views` | MAT同字节核验、完整membership、canonical identity、授权标签/无标签view分离、真实raw支持/可用时间与lineage、178/179诚实区分、native/Cz布局显式声明 | 其他库适配与硬件/光学时钟确认 |
| B02 多专家接入 | `transfer.ensemble`、`ensemble_fit`、`calibration_state` | 真实raw/ERP/NN adapter、固定算术概率池、模型/输入/拟合身份、唯一producer绑定、整包保存与无标签重放 | source-only K0多专家与工作站CLI接入 |
| B03 校准 | `ensemble_fit` | 按完整组产生真实OOF，仅C内拟合每专家正温度，最终全C refit；固定权重和固定τ仍独立声明；小于3组拒绝学习型温度 | 复杂质量gating与独立效果归因；学习温度不称成人认证 |
| B04 神经贡献 | 既有checkpoint/recipe owner＋`ensemble_fit` | 真实source checkpoint准入、整目标库排除、同结构scratch不继承source值/统计，明确source权重和architecture-only角色 | 同输入waveform/scratch控制的完整公共配对矩阵 |
| B05 主线AR专家 | `transfer.continuous_decision`＋既有`TemporalNoise` | 显式AR0/1/4/8/16/32与stationary/PARCOR启动，Y/D同步白化，保存重载及dense Gaussian/GLS独立oracle；默认旧AR1语义保留 | fixed-holdback32、联合noise MLE和超参数积分独立研究 |
| B14 工程收敛 | `calibration_state`、执行入口 | 复用真实numeric snapshot，ERP comparator删除重复decode实现；每expert模型和可用性独立；入口/helper全源绑定，失败状态落盘；稳定LF仅作用本轮编辑文件 | 全源池资源DAG与当前云端吞吐实测 |

`EnsembleScorer`只接预声明的独立专家，字典key必须等于唯一producer身份；多专家输出保留所有leaf来源。层级权重直接在平坦roster声明，不能把已聚合结果伪装成某个基础expert。

每个专家使用共同物理decision、公开onsets/membership和deadline，分别保存自己的数值view。Toeplitz保持无trial baseline，xDAWN使用mean-only，NN按冻结profile，raw连续专家使用未追加本项目滤波的分发MAT信号。source checkpoint与178/179、reference、通道及执行凭证不一致时拒绝，不能改头/补样本绕门禁。

## 实际研究入口

`experiments/run_candidate_ensemble_study.py --plan <JSON> --output <全新目录>`提供完整MAT→校准组留出→真实expert fit→每专家保存重放→固定mixture→requested评价/配对contrast。每专家每fold只拟合一次，多个mixture复用其冻结预测；某个NN数值失败不抹掉成熟经典臂。

plan schema为`n2p3_candidate_ensemble_study/1`，显式声明records/path/SHA、authorized_labels、group_count、observation_response_seconds、pre_roll_seconds、experts、mixtures、baseline_mixture、seeds、source_checkpoints、calibration_strategy、device、workers和cpu_threads_per_worker。软件fixture必须`software_correctness`；公开校准开发必须`public_calibration_development`并提供真实执行源码冻结清单。

该入口的seed是**固定source下的target随机性**，输出`randomness_scope=target_only_fixed_source`。它不冒充三个完整source→target流程seed。BNCI默认C800前720标签形成6个完整组，每fold拟合600，80授权标签不用；K200仅1组不能据此产生独立group-CV温度。

新179点profile需要source359点，经FFT180后保留真实179；历史358点视图对应178。与标准NN179比较时本轮预声明共同观测tail为308/256秒，所有臂同支持重跑；历史37/48不作为这一新输入的匹配锚点。

## 独立反例与修复

- 连续模型在入口拒绝复数，避免隐式丢虚部；旧AR1具体均值/协方差快照保持。
- 错贴expert字典标签原可在非等权池翻转winner；先复现不抛错，再绑定实际producer。
- 单臂或配对宏NLL原过滤全缺概率的人/seed；红例得到错误的部分宏风险，修后完整requested宏风险为None，coverage与缺失仍保存。
- log-probability保留真实零和数值下溢的区别；不以消费端tiny clip美化NLL。
- 公共view/内部数组无法重新打开write flag；held truth变化不改变score，raw/record/participant范围错配拒绝。
- 实际模型内绑定expert配方/fit授权/source角色，重签manifest仍不能替换AR阶数、参与者或数值state；校验后的同一bytes直接解码，source一次冻结贯穿outer fold及OOF。
- 公共scorer/participant/temperature/state只读；完整微调必须在首次fit前绑定同结构scratch及waveform控制。具名不可识别数值失败只影响相关expert/mixture，合同错误保存父/子failed进度后传播。
- AR dense oracle、row/voltage gauge、候选置换、原MAT真实解析、旧接口、实际fit/save/load/CLI分别举证，不能以新测试数量替代真实入口。

## 已完成的软件验收

最终实施提交为`d3e602701716748d0ca561400699204268d4f8d6`，评审基线为`a6a3da5`。以下为不同范围的独立记录，不能相加为不同测试数量，也不能把早先完整测试包装成最终提交上重新执行。

| 验证 | 实际版本/范围 | 结果 |
|---|---|---|
| 完整pytest | `0f8074a`，2444项 | 2425通过、19跳过、0失败/错误，335.754秒 |
| 质量修复回归 | 测后提交为`0f8074a`，58项 | 全部通过 |
| 最终只读字段调整 | 测后提交为`d3e6027`，55项 | 全部通过，10.882秒 |
| 独立Standards复验 | `d3e6027`，8个相关测试文件及行为探针 | 117通过、0跳过/失败/错误；144文件摘要前后不变 |
| 独立Spec复验 | `d3e6027`，6个真实fit/score/保存/CLI探针 | 6命令退出0、13行为判定通过；129文件摘要前后不变 |

Ruff与`git diff --check`通过；最终软件评审时隔离树clean。完整测试的19项跳过为独立前端运行时8项、可选数据5项、Windows symlink权限2项和CUDA运行时4项，不记作这些集成已通过。

[独立质量报告](ensemble_quality_20261005.zh.md)分别保存Spec/Standards结论、已修缺口与检测范围；[机器可读验证收据](../assets/ensemble_implementation_20261005/validation_receipt.json)记录原JUnit摘要/哈希、复现命令、已记录命令的来源及退出码。原A01只读入库探针完成6组/720用/80不用和`[120,8,179]`，不含新的公开数据训练。

实际source/tests/configs/入口、六指南、科研规格与验证收据由[最终源码清单](../../frozen/ensemble_implementation_20261005/manifest.json)绑定物理压缩包并逐成员核验；[原始质量证据清单](../../frozen/ensemble_implementation_20261005/quality_manifest.json)独立保留两路探针、过程RED、原始日志和synthetic模型。压缩包及镜像留在本地，清单进入Git；源码身份按实际工作字节而非规范化Git blob认定。

本轮尚未运行完整四库/十二库三seed新矩阵，未取得成人独立确认；工作站生产默认未切换。继续优化以原123项/15块依赖图为准，不以这批core实现重定义整项目优化完成。
