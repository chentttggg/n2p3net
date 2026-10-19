# 独立前缀、多概率程序与配对公共入口

2026-10-08的软件切片已实现并合入 `codex/deep-research-plan`。B12原分支的Spec/Standards独立评审均为0个遗留确认问题；受审源码427成员归档为45031d46…。主树合入B07物理字典时只在共同raw拟合入口接回已验收的ResponseDictionary编译，随后6项组合调用验证及新A01保存重放通过。该合并随后完成独立Spec源码/新包关系复核，遗留确认问题0；无新增拟合或宽测试。

## 行为与真实关系

- `CalibrationPrefix`在入口一次核实际K、raw截止/right-censor、trial行/标签/时序/profile/身份/执行收据，随后封存数字与事件数组。K200原生诊断真实使用200个raw标签和196个可用trial，不按完整selection取整。
- fixed与grouped-temperature分别声明；共同all-C最终专家只拟合、保存一次，grouped额外使用合法C-only OOF。完整组不足或OOF局部数值失败各自留账，fixed成功输出保留。程序名、strategy、固定单位温度及grouped范围在载入入口绑定。
- K0使用冻结源NN的权重、BN、输入统计及源LLR；target-fitted经典/raw臂具名不可用，固定混合仍保留完整专家权重。
- `SourceSeedCatalog`消费真实source producer owner：同seed/cache/source、participant-disjoint内层、patience轨迹和完整范围refit/实际updates相互绑定。两个公共消费者保留自己的证据角色，共用源scope；whole-study排除消费实际study membership。
- 新paired binary入口与candidate共用三组source→target seed和请求预算owner。200/800等请求映射到同actualK时各自保留原因、统计分母与outcome索引，共用一次fit/score；未获准请求不占合法复用键或scratch对照。经典确定性拟合的种子别名消费同一已重放数值包。

## 验证与归因

19项prefix/温度/binary关联验证对应f58归档；后续binary两处consumer修复另在最终450源跑11项，0失败。最终两轴均直接读取真实源码、原MAT及保存模型，未新增审查拟合。旧aad源的原生6包/612重放保留其原执行身份。

主树合并的ad4源另跑6项物理字典/OOF/独立caller组合验证。新A01实际拟合raw AR8和Toeplitz共2个模型、1个温度共用包，对34个物理请求的三个mixture执行102次精确重放；C200仍为200 raw/196 trial/一个完整组。Ruff与差分检查通过，没有重复全集。

质量与源身份见[紧凑收据](../assets/independent_prefix_20261008/quality_receipt.json)。源码归档、原始日志、模型与评审记录留本地，不能以Git父SHA或旧通过改签新程序。

## 后续完整范围

本切片不是B12公共效能或15块/123愿望的完整交付。十二库1964份原始记录声明已核存在，但新cache与实际K队列仍须准入。现source入口尚缺声明zero profile、每seed选择/refit及共享25 CPU/4 GPU-worker调度的合一接口；旧shared reference/λ程序不升级为三个完整选择流程。

新云实例已实证25 CPU quota、90 GiB内存额度、约32 GiB RTX5090及Torch2.8/CUDA12.8可用；预处理依赖与数据库存继续准备，未启动新公共训练。完整四库binary、合法membership候选和新成人/设备确认分别按活动路线验收。B08冻结1344格仍独立后台运行，评分数值源保持其原128文件，不能用本主树改签旧包。
