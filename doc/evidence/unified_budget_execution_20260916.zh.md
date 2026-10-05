# 分阶段预算执行器与性能准备

2026-09-16。按[精简训练方案](unified_compact_training_20260916.zh.md)接通执行器，未启动真实全矩阵或云端任务。
修改前源码和测试见`frozen/unified_budget_execution_20260916_before/source.manifest.json`。

## 预算已经执行到真实训练循环

`train.early_stopping.UnifiedTrainingBudget`集中解释新预算，默认配置在
`configs/unified_training_budget.json`。它包含参考源的patience3、min_delta1e-4、最多12曝光周期，
源验证比例0.2、目标90轮、每来源库2名开发人员及一个固定开发seed。
参考policy内部的target上限不参与源计算，固定为1；修改个人90轮不会改变参考源的拟合签名。

源入口`run_source_training_chain.py --stages unified_source`和总入口
`run_domain_transfer_matrix.py --program unified`解析plan的`unified_training_budget`字段。
源条件只枚举R1/S0/M0、R1/S1/M1，默认校准预算K0/200/800。
源内H按实际合格C200/query与固定身份seed，每来源库选2人；H不进入参考源梯度、BN、输入统计和早停验证。
余下每库至少保留4人，以满足既有独立source验证接口，缺少支持在准备阶段报告不可执行。

每外折只使用一次A1参考：inner选轮后从原seed在S\H重训；三个正式source seed不重复开发搜索。
main的`fixed_updates=e*ceil(n_source/512)`写入DeepConfig，`source_budget`保存选中轮数、
reference checkpoint摘要、reference policy和fit范围摘要。实际fit检查更新数与曝光公式一致。
该字段随source训练合同、registry、resume、checkpoint与目标plan传递；已完成参考源也进入原恢复owner。
源中心选择允许λ=0，默认四候选，固定target轮数直接拟合C；有真实membership时使用Q1。
选择按seed、参与者、来源库逐级等权，任何请求失败都保留原分母。

公共目标使用`unified_q0`、`unified_q1`、同结构Q1监督的`unified_scratch`和确定性波形LDA。
两种source共同构成四格迁移处理；scratch/LDA通过现有`TargetResultReuse`跨source复用。
LDA配方使用固定seed0，通用复用键也移除其不参与数值拟合的seed字段；其他NN种子仍属于身份。
λ=0、D不适用时Q0/Q1是同一数值程序，配方名称不阻止已有精确复用。
每个逻辑请求保留自己的来源与复用记录，物理fit和统计重复分别计数。

个人`selector_mode=unified`改为完整C一次拟合，保存模式`fixed_unified_program`。
一条完整授权校准录制已能执行；预检、自动EOG的fit会话、QC、BN、保存重放共同使用完整授权C。
CLI使用`--training-budget configs/unified_training_budget.json`及一份明确的`--calibration-recipe`；
旧`--early-stopping-config`入口已移除，旧保存模型继续按其已保存状态推理。
配方epochs必须与预算target_epochs相同，拒绝同时声明额外更新上限或内部refit。

## 输入与复用的性能变化

`prepare_source_records.py --target-prefix-budgets 0 200 800`默认只准备实际使用的C_K和固定test。
只有显式`--early-stopping-inner`才准备个人内部T/V；改变这一准备模式会阻止错误的completed复用。
底层准备函数保留显式`prepare_inner`，其他早停研究可以继续调用同一raw范围owner。
各K仍独立裁raw再滤波，省去的是不再消费的inner算子。

`load_target_prefixes(..., load_inner=False)`避免解压未使用的内部视图；源内开发的父进程逐参与者检查后
只保存身份、资格和路径，训练reference时不驻留整个源池的C_K张量。
目标worker按实际分片的参与者集合加载前缀，组内各source/seed共用这些输入和结果复用owner。
不同source组沿用现有GPU信号量、CPU线程与内存准入并行；正式目标阶段按相同cache/参与者分片组织，
保留跨source、seed和BI家族的scratch/LDA复用。没有另建一套调度器。

## 源中心损失的实测优化

原先每个更新按tensor、角色分别计算均值，产生很多小CUDA算子。固定中心和系数现在只pack一次：

\[
\Omega(\theta)=\sum_j w_j(\theta_j-\theta_{S,j})^2,\qquad
w_j=\frac{1}{2|P_{g(j)}|n_{p(j)}s_{p(j)}^2}.
\]

该式与逐tensor均值、逐角色均值后求和相同。独立逐tensor oracle核验loss和每个参数梯度；
CPU合同测试及CUDA探针均通过。浮点归约次序改变，使用明确容差，未宣称训练轨迹逐位不变。

在项目`.venv`、Torch2.13.0+cu130、RTX5070 Laptop、单计算线程下，3次热运行中位数：

| 实际路径 | 逐tensor原式 | pack后 | 测量单位 |
|---|---:|---:|---|
| prior前向+反向 | 6.580ms | 0.821ms | 一次loss/backward |
| Q1完整30轮拟合+预测 | 0.702s | 0.498s | 含输入统计、BN重估、真实完整decision工作集 |

Q1约快1.41倍，耗时减少约29%。同一探针Q0为0.445s，A0/A1各30次source更新为0.398/0.733s。
这些是200行、六导179点、batch上限512的合成prepared-array实测，包含缺导pack与联合损失。
不包含全库raw准备、云端传输、多worker竞争及真实泛化评估，不能直接替代5090完整矩阵耗时。

可在目标环境复用同一入口：

```text
.venv/Scripts/python.exe experiments/benchmark_unified_training.py --device cuda --output probe.json
```

Linux用其项目环境对应python路径。`--compile-mode none|default|reduce-overhead`分别测量，
每次使用新的输出；冷启动与热运行分列。动态mask尚无CUDA Graph吞吐证明，默认探针为不编译路径。
正式云端4/8worker的完成率仍须在实际资源和真实scope上验证；原1.5–3.5小时区间继续作为条件估计。

## 验证和阶段归档

首批29项覆盖新source/target配方、单录制个人链、QC重载与raw准备，134项既有调用合同回归通过。
完整source执行反例证明：两个主seed产生1个reference和4个main，reference排除H，main重新包括全部合法source，
所有main用相同e*，没有各自重选轮；λ开发只用一个开发seed。
source checkpoint→prepared scopes→outer plan→target worker的实际集成还核验了跨target seed只拟合一次LDA，
同时NN保持独立seed拟合。两个扩展集成通过；此前测试遗漏`dry_run`参数已修正。

预算/个人/输入模块阶段快照见`frozen/unified_budget_execution_20260916_modules/source.manifest.json`。
完整测试日志、性能JSON和最终验收记录保存在`doc/assets/unified_budget_execution_20260916/`。
初次全套有两项旧测试夹具失配：手写source job缺少新预算，及按旧列表位置误选LDA。
夹具已补真实预算并改用具名Q0/Q1，相关29项复测通过。
最终稳定源码全套 **2080 passed、4 skipped，428.55秒，exit 0**；Ruff与标准`git diff --check`通过。
四项skip沿用两个独立工作站入口缺配置、两个Windows符号链接权限场景；没有新增跳过。
分批测试有重叠，不能相加作为独立总数，也不将测试总耗时当训练性能基准。

最终[源码与配置清单](../../frozen/unified_budget_execution_20260916_acceptance/source.manifest.json)和
[验证证据清单](../../frozen/unified_budget_execution_20260916_acceptance/evidence.manifest.json)
分别绑定物理压缩包，整体SHA-256及逐成员摘要均核验。原始数据、旧结果及并发工作保留。
