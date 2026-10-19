# 算法链八项修复与完整验收

2026-09-16。用户在只读审计、数学规格及反例之后授权完整实施。工作树已有大量前置变更，
本轮在其基础上修改，未回退并发工作、未改原始数据，也未替换正在运行的云端执行快照。

**结果：F1–F8已落实，完整测试1936通过、4跳过、21条警告，退出码0。**
此外补齐独立复核发现的缓存凭证恢复缺口。全部实现与测试使用项目`.venv`。
该记录属于数学/软件验收，没有新的真实数据训练或成人九选一性能结论。

## 当前实现

| 项 | 原反例 | 实际修复与联动 |
|---|---|---|
| F1 拟合范围 | 旧非逐导QC在T+V拟合，dry-run拒绝而正式CLI进入adapter | 原`_validate_ocular_fit_scope`深化为公共`validate_calibration_raw_fit_scope`，个人preflight、旧fit、候选和fit缓存共同调用；嵌套来源及完整session门禁保留，固定LDA使用完整C |
| F2 辅助损失 | 单行尾批及无合法项的共享MMD无法求梯度；NaN重复行未明确拒绝 | 保留先前批量MMD，原特征/索引在normalize和dedup前验证；合法空项图连通零，配置和编码非法仍拒绝；基础双样本接口不放宽 |
| F3 重复设计 | 三对完整流程结果被当成九个交叉组合 | `resolve_seed_pairs`集中解析；源任务→目标计划→manifest/record携带完整`analysis_seed_pairs`，CLI核对显式声明与记录，不从成功行猜设计 |
| F4 源因素 | 多源目标函数共用target arm名导致重复key | P2默认按source_arm分组；`analyze_source_objective_contrasts`用两个因素的明确联合身份，复用配对统计引擎生成源差和相对scratch/LDA的交互 |
| F5 边界语义 | 全局伪迹与自然端点坐标重合使hard bound清零 | `continuous_filter_plan`同时保留span与原因集合；合并/重合不丢artifact；逐导恢复按原因认证完整双端界；receipt记录实际后端支持，旧坐标接口和legacy全局行为保留 |
| F6 坏导与来源 | bad Cz/Pz进入全True源缓存，旧completed只看BDF/NPZ | 源入口核验物理导规范名唯一、角色和坏导，参考坏即拒绝；永久坏非参考导在计算前剔除，最终montage零占位；实际sidecars及方法身份进入准备账本/lineage |
| F7 证据字段 | task scope循环覆盖证据等级字符串 | task_scope与record证据字段分开，零/多scope的真实CLI输出类型和等级稳定 |
| F8 配方分叉 | 同一legacy配方随QC路由改变输入统计/BN，显式weight decay丢失 | 路由前解析同一TargetRecipe，矛盾初始化/BN/dropout/refit/updates/policy两路拒绝；旧选方与refit流程保留，实际配方和状态摘要保存 |

实施过程中独立复核又发现：NPZ摘要正确但`.record.json`缺失时，source准备恢复仍会宣称completed。
新增四个RED反例后，shard及merged都绑定并核验NPZ和record原字节SHA，复用原attestation校验，
核对完整验证标记、大小及身份账本；恢复不复制/解压大型矩阵。缺失或篡改明确拒绝，不重新认证旧缓存。

## 数学契约落实

个人拟合状态来源F必须落在当前合法拟合范围；训练输入及mask的原始支持D不得跨入验证/测试。
源权重/统计按源排他合同核验，不误套本人T子集规则。固定LDA无inner选择时允许全部C；
raw QC/EOG冻结、显式epoch model冻结、epoch policy在final C重拟合分别保存其真实生命周期。
范围函数关闭旧分支漏检，物理身份和实际信号支持仍由既有loader/执行凭证共同核验。

MMD保持原去对角估计及同批detached median带宽：

\[
L_{aux}=|\mathcal T|^{-1}\sum_{t\in\mathcal T}\widehat M_t\quad(|\mathcal T|>0),
\qquad L_{aux}=0\quad(|\mathcal T|=0).
\]

负估计不clamp。空项是合法退化情形，NaN/Inf/非法索引是错误，不能混同。
同批带宽detach只阻断带宽梯度，不提供统计独立性；固定核无偏性需要额外抽样和表示前提。
CPU通过有限值验证后才进入损失，CUDA采用一次设备断言，保留无逐cell主机同步的批量核。
全inactive的五步对照逐步比较梯度及None状态、Adam状态、RNG、最终权重/BN，均与CE一致；
没有补行、丢尾或新增forward改变科研预算。

统计使用显式关系R={(source_seed,target_seed)}；先seed内配对、再参与者内平均、最后固定库等权。
缺任一预声明库时，全库均值/区间不可估计，`partial_observed_studies`保留实际子库读数。
已知actual K的整人无cell进入失败账本，未知K/未知分组不臆造；非法AUC不进入描述均值。
一般零和对比w的范围由B=sum(w正项)给出：普通两臂B=1、四项交互B=2。
缺失结果识别界先在各库计算，再按库等权；它不是置信区间。共享scratch/LDA不增加独立重复数。

边界原因集合保留完整双侧状态关系：

\[
\delta y=JF_NJO_N\delta z_L+JO_N\delta z_R.
\]

任一端为artifact/未分类实际reset都使用完整界；未知非reset注释不创建新重置。
混合gap原因并集是保守调整，可能新增部分拒绝；原分段数值顺序不变。持续污染的强迫项仍不受状态证书保证。
空间输出mask按全部真实依赖导的逻辑与传播；坏Cz不能置零替代参考，局部raw零填也不能代替分段滤波。
缺失导在最终表示才成为精确零，避免`NaN*0`和额外阶跃。

## 反例与验证

先完成对应RED再修改实现，覆盖真实个人CLI、实际source baseline fit、source prepare/save/attested load、
多轴结果分析、cache恢复、模型保存/重载和验证数据非干扰。新增断言进入原职责测试目录，
没有保留被替换的私有范围函数别名或错误语义测试。

| 验证 | 结果 |
|---|---|
| 全套 `python -B -m pytest -p no:cacheprovider` | 1936 passed、4 skipped、21 warnings，292.88秒，exit 0 |
| Ruff `check --no-cache src experiments tests` | 通过 |
| `git diff --check` | exit 0 |
| 校准/QC六文件及新增数值保持反例 | 132次定向测试执行通过，之后纳入全套 |
| 预处理/恢复两组文件 | 141次定向执行通过，之后纳入全套 |
| source坏导/域映射 | 初轮48通过；凭证缺口修后source_records 22通过，全部纳入全套 |
| source辅助及alignment/ASMMD | 49通过；五步optimizer/RNG强化反例另通过并纳入全套 |
| 统计与失败分母 | 全文件30通过，之后纳入全套 |
| 冻结实现对照 | 300次分段坐标相同，16次forward/双向滤波逐位相同 |
| 独立数学oracle | 12组通过，含梯度、mask、seed设计、缺失界及AdamW反例 |
| 本地CUDA/Dynamo | MMD/SupCon×singleton/重复行/正常批共6组通过；loss及gradient与CPU参考一致 |

上表定向执行有重叠，不能加总成额外独立测试数。全套4项依原测试条件跳过，没有计入通过。
警告原文保留于日志，主要为少样本协方差、现有GTN元数据兼容说明和第三方模型弃用提示。

本地设备为RTX5070 Laptop GPU、Torch2.13.0+cu130；Dynamo使用eager后端fullgraph捕获，
并非Linux Torch2.8 Inductor稳态吞吐比较。辅助CUDA Graph禁用身份仍沿原训练合同；没有替换云端训练。

独立数学复核还更正了研究论证：九选一五重复的trial标签给定目标/排程后只有九种完整模式，
不能凭45个标签推导固定λ；原梯度内积为正也不能直接推出AdamW使校准损失下降。
这些验证只约束后续研究推理，未自动启用新loss、改变采集协议或产生新效能结论。

## 兼容与后续执行

source准备账本为`n2p3_source_preparation/2`。方法/实际raw/sidecar改变、旧证据缺失或缓存凭证损坏时，
保留历史输出并使用新目录。首次准备可在解析前将未预声明sidecar绑定为`preparation_start`；
它不冒充原manifest中的冻结证据，completed恢复每次仍检查实际依赖。

新源任务自动传递分析设计。旧P1/P2结果若无设计字段，需显式声明seed关系；P2自动源对比还需完整source arm列表。
CLI与执行声明冲突则拒绝，不能依成功记录缩小设计。不同布局、协议或来源家族敏感性按原声明分组，不混成一个均值。

正在运行的云端P1/P2矩阵仍归其既有执行归档，本轮未部署/重启。
已完成模型可用新统计程序按原训练身份重新汇总；不能把它们的源码身份改写为本轮修复版。
后续新训练应按工程指南核验资源与吞吐后，从新冻结代码执行；短记录、决策损失与辅助目标的科研替代仍按先前门槛推进。

## 文件与物理证据

- 修复前源码：[manifest](../../frozen/algorithm_chain_20260916_before/source.manifest.json)。归档含当时dirty源码，不把HEAD当完整执行身份。
- 修复后源码：[manifest](../../frozen/algorithm_chain_20260916_after/source.manifest.json)。与当前源码逐文件校验。
- 验收归档：[manifest](../../frozen/algorithm_chain_20260916_validation/manifest.json)。物理压缩包保留完整日志、数学输出、源码改动差异及本记录。
- 结构化验收：[validation.json](../assets/algorithm_chain_20260916/validation.json)。
- [数学oracle](../assets/algorithm_chain_20260916/verify_contract_math.py)、[结果](../assets/algorithm_chain_20260916/math_verification.json)。
- [冻结滤波对照](../assets/algorithm_chain_20260916/verify_frozen_filter_geometry.py)、[滤波/mask推导探针](../assets/algorithm_chain_20260916/verify_filter_mask_math.py)。
- [CUDA/Dynamo探针](../assets/algorithm_chain_20260916/verify_cuda_source_objective.py)、[结果](../assets/algorithm_chain_20260916/cuda_source_objective.json)。

原始逐步日志位于本地`experiments/runs/algorithm_chain_20260916_validation/`。压缩payload保持本地，Git只分发清单及适合公开的源码/文档/结构化软件证据。
