# 完成实验产物准入：实现与软件验证

日期：2026-09-09。范围为公共源池目标矩阵与本人EOG矩阵的完成结果复用，
不启动或续跑真实云训练，不改写既有模型、预测或冻结执行快照。

## 原问题与责任

原公共续跑只校验顶层record和checkpoint/cache/snapshot几个字段，未读取逐cell模型与预测。
原本人续跑只比较JSON中的任务、checkpoint及快照，未使用首次生成时保存的模型/预测摘要。
已经通过原始封存的文件被移除后，两个真实CLI仍可能记录`reused=true`。

现将准入集中于`transfer.result_reuse.admit_completed_result(path, kind, expected)`。
`expected`必须包含对应协议的完整声明字段；生产和消费使用同一builder，
不能用任意几个字段替代任务匹配。所有不完整、损坏或不匹配结果抛出`ResultIntegrityError`，
在续跑处中止，不转成可复用结果或被试拟合失败。

共享实现核对原封存记录、必需文件及摘要，并直接解码已验证字节。
公共与本人结果各有内部adapter，保留实际协议差异；未建立统一训练Runner。

## 公共矩阵

当前`target_context`与`build_manifest`同时提供执行和续跑的完整声明：
具体被试分片、train/test/purged行、任务顺序、完整recipe、预算调整、source任务与seed、
来源库、环境和实际执行快照。相同cache/checkpoint不足以复用不同任务。

顶层结果必须完成，`tasks.json`与内嵌manifest一致，cell数量和顺序与任务一致；
每份逐cell记录与顶层原声明一致，评价合同摘要有效。
完成cell必须有预测，拟合cell另需完整模型；train/test行与声明scope一致，公共score必须有限。
模型的recipe、fit history、状态摘要与cell交叉核验。

`fit_failed`仍计入原失败分母，不要求不存在的成功预测/模型。
失败前遗留但未被声明为证据的文件不自动成为可用模型，也不单凭其存在认定完整结果损坏。
completed/failed计数必须与全部cell一致。

## 本人矩阵

`personal_result_contract`绑定完整task、recipe、checkpoint、实际快照和运行环境。
task含输入cache摘要、EOG/view/scoring_event_ids、初始化、seed、通道和预声明测试会话。
新结果在首次发布时封存所有逐session NPZ以及正K个体包；旧缺原始seal结果保持历史可读，
不在新续跑时补摘要或重标为新证据。

预测必须覆盖全部声明session及各自请求事件；事件不重复，scores/raw_scores/标签/候选/组/repetition行对齐。
NaN和全部NaN都允许，保持原失败分母；Inf拒绝。
common view只按预声明scoring_event_ids屏蔽分数，其余事件及raw_scores保留。
K0无目标拟合模型且更新数为0；正K必须有经过原个体包加载门禁的模型。
模型内任务、recipe、fit history、训练行数、快照及原队列计划与结果交叉一致；
source初始化还核对源checkpoint与训练合同摘要，scratch/LDA不伪造source初始化字段。

选择性`--task`续跑会改变本次队列摘要，但不会改变已完成单任务的科学声明。
因此复用不要求新旧队列摘要相同；原模型和result中的plan来源保留。
新队列保存为`plan.<digest>.json`，首份`plan.json`不覆盖，progress指向当前计划。

## 验证与执行限制

新增准入回归62项通过，30.96秒；原结果复用、目标实验、源池及个体包相关86项通过。
使用真实生产封存、合成数据和两条真实CLI；CLI测试禁止提交训练任务，替换快照验证及外部输入准备，
不替换准入实现。真实source个体包fixture只用一次27条合成BIDS、CPU单线程、固定200 updates，
用于验证保存/加载与原源合同一致性，不构成公共数据开发成绩或成人效能验证。

最终全套为**1726通过、4跳过，482.00秒**；307份活动Python文件Ruff及`git diff --check`通过。
两个新矩阵CLI从仓库外目录运行`--help`通过；本次6份源码/测试在最终全套前后SHA-256一致。
4项跳过为两个未配置独立采集端根目录的集成测试，以及两个Windows权限不支持的符号链接测试。

首次全套有1722通过、4跳过、1失败：代理在测试期间更新了`doc/roadmap.zh.md`，
源码快照测试正确拒绝该变化。停止源文件及文档修改后完整重跑通过；未放宽快照门禁。
两次日志均已物理压缩并核验，最终源码摘要与检查记录见
[验收清单](../../frozen/completed_result_admission_20260909_validation/manifest.json)。

独立`verify_pool_matrix.py`还负责从指定冻结runtime读取cache/source并重新预测、复算指标，
本轮保留其接口。既有完成验证记录不能代替新一次伴随文件准入。
状态摘要实现集中于result_reuse，target_experiments保持既有导出供当前验证脚本及冻结runtime调用。

改动前10份相关源码、验证入口和活动指南已压缩并逐成员核验，见
[局部源码归档](../../frozen/completed_result_admission_20260909_before/manifest.json)。
该归档只覆盖清单中的文件，不声称是可执行训练快照。
