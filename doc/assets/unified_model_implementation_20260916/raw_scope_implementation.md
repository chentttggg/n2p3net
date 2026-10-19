# 原始样本范围、逐 K 缓存与目标早停输入

2026-09-16。此记录对应统一模型输入准备模块的软件实施；没有执行完整数据重建或模型训练。
既有工作树改动保留，源码归档由统一实施 owner 汇总。

## 执行合同

`data.preprocess.RawSampleScope`声明绝对 MNE 样本半开区间及原事件账本中的授权索引。
`execute_preprocessing`先从原 raw 裁出该区间，再调用原滤波、epoch、重采样和基线 owner。
原闭端 epoch 支持统一表达为整数半开区间：

`[event_sample + round(tmin * fs), event_sample + round(tmax * fs) + 1)`。

源采样率、原始 `first_samp`、授权原事件编号、crop 后实际几何和执行顺序均进入凭证。
不同 K 的 C 区间分别处理；必要响应尾部包含的后续刺激不会自动获得标签授权。
标签授权数、边界剔除后可用标签数、实际拟合行数分别记录。
`build_subject`保持原始事件 ID、绝对时轴、session/run，不将同一物理记录的 T/V 伪造为不同 session。

原始支持规划和 80% 有限数据回退使用已有 `transfer.binary.resolve_calibration_splits`；
沿用其固定测试原则及小样本回退语义，未把 `minimum_test_trials`改成新的绝对准入下限。
最终有效标签预算用 `effective_budgets`表示，调整原因以请求预算为键，不能用 K200 标签标记实际 K16。

## 已连接入口

`experiments/prepare_source_records.py --preprocessing-config <完整 JSON>`，或 manifest 顶层/record 的
`preprocessing`声明，均进入实际执行器、准备方法身份和缓存 token。声明不一致直接拒绝。
旧 completed 目录不能在改变 profile、raw、sidecar、scope、源码身份后复用；缓存本体与完整验证记录均核验摘要。

增加 `--target-prefix-budgets 0 200 800 2000` 后，入口输出 `target_prefixes.json`。
单记录共用 Kmax 确定的测试 raw 后缀，每个 K 单独生成 C；读取/解析原 raw 一次，滤波独立执行。

manifest 明确提供 `target_protocol.kind=last_session_holdout`、`session_order`、`run_order`及
`test_session`时，按该顺序分配早期记录的标签。用尽一条早期记录时可消费其完整 raw；
未用尽的最后一条只处理到授权事件必要尾窗，后续早期记录不进入 C。测试使用声明的末会话完整记录。
此路径先做有界的单记录元数据读取，再逐记录复用 reader 处理各 K，内存中不同时保留整个参与者的所有 raw。

`load_target_prefixes(path, subjects=None)`返回逐主体的 `calibration[intK]`、固定 `test`及 `early_stopping_inner[intK]`。
它核验缓存摘要、原物理身份、scope 与实际执行凭证、事件预算、记录顺序、边界和实际剩余数。
绝对 cache 路径可以显式搬迁成 manifest 旁的相对路径；raw 字节身份和缓存摘要不随之改变。
并行 worker 可显式传入 `subjects=(subject,)`，先核验全 manifest 的结构与主体存在性，
再只解压该主体的缓存；不存在的主体报错，不让每个任务重复解压全研究数据。

## 全局早停

按用户最新的全局早停指令，保留并复用并行工作流已加入的 `early_stopping_inner`，固定剂量旧假设不再作为准入依据。
单记录在 C_K 内重新分配前 80% 授权事件，T/V 各自裁 raw 后处理，最终 C_K 另有独立缓存。
多条已授权记录优先将末 `ceil(0.2 * record_count)` 条交给 V，T 使用其余完整授权记录。
每个输入独立处理的原 raw 范围可复用，因此这一分配不需要再次滤波已独立处理的完整记录。
当 K 只涉及一条记录时，仍调用单记录嵌套 raw 范围 owner。

T/V 剔除后必须分别保留两类；不足时写 `status=unavailable`及明确原因，目标执行方不能退回 C 内 epoch 切分。
多记录 C 的最后一条部分记录允许只含一类；其标签仍可与前面记录合并。
该情况若使完整记录式 V 缺一类，就明确记为早停范围不可用。

`target_raw_record_scopes`把每个范围绑定到 study/subject/session/run、输入角色与原始摘要，
以及执行凭证摘要。不同物理 raw 可拥有相同几何/operator 凭证；凭证 bundle 的去重不等于删除物理记录。
多记录合并通过 `target_raw_view_identity`保存有序子缓存组合身份，类型明确为
`ordered_attested_cache_composition`，其摘要不是不存在的合并 NPZ 文件摘要。
`IndependentEpochSelectionData.from_views`已通过单记录和多记录实际集成验证。

## BNCI candidate 衔接

BNCI 原始入口复用严格 MAT parser 与 candidate builder，保留 MATLAB trial 边界、真实 selection、
membership、原 repetition 和 `selection_repetitions=10`。部分 selection 不因恰好包含完整 repetition 而获得完整组资格。
本模块调用另一个实施 owner 完成的 reader/builder；没有重新编写简化 candidate 解码器。

显式 source-record 准备链的既有输入 builder 在 `source_eog_20260909/build_inputs.py`按
官方文件 stem 定义 A01..A08，session/run 为 0；严格 candidate owner 使用同一合同。
MOABB 的数字 provider ID 属于另一个本地表示，`data.identity`未声明它与 A01 的别名映射。
本次不改写已有身份表或旧缓存，不凭文件名推断跨 provider 的同人关系。

## 验证与边界

相关测试：`test_raw_preprocessing_scope.py`、`test_source_prefix_preparation.py`、
`test_preprocessing_execution.py`、`test_source_records.py`、`test_dataset.py`、`test_epochs.py`、
`test_bnci2014_008_candidate.py`，合计 **183 passed in 36.56s，exit 0**。
随后 CLI 恢复定向回归 3 passed；按主体懒加载反例 1 passed，均 exit 0。
包含外部 EEG/标签替换不改变授权输入、精确一采样点边界、非整数采样率、原事件账本保留、
K0、预算回退、伪造计数/范围/profile/缓存验证记录拒绝、多记录顺序、单类末片段与 CLI 恢复。
Ruff 和相关 `git diff --check`通过。未将旧全套通过作为新时间隔离机制的证据。

实际读取了 Huebner2017 一人的 session0/run0、session0/run1、session2/run0，
在独立 probe 目录执行新双向 profile 和 K0/200/600；这只是三记录软件验证子集。
测试保持 476 行；C200 保留 200 行，inner T/V 为 160/35；C600 保留 600 行，inner T/V 为 476/124。
两组均通过 `IndependentEpochSelectionData.from_views`范围与身份检查。
原文件未修改，未拟合模型，不能从这些行数推断真实识别性能。
执行源码摘要、原始 sidecar 身份、cache 摘要及核验结果见 `raw_scope_probe_summary.json`；
完整缓存/账本在 `experiments/runs/unified_model_implementation_20260916/raw_scope_probe/`。

该 source 准备路径的 QC 仍为显式有限值/恒值及通道角色准入；未声称已经拟合 learned epoch QC 或 EOG。
`crop_raw_scope`为这些拟合 owner 提供合法原始样本入口，实际启用时仍需各自绑定 fit 范围和状态。
