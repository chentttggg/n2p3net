# 2026-09-20 域迁移与 BrainSync 稳定大版

本轮大科研从 2026-10-04/05 的连续算子、专家、字典与 B00–B14 机制开始。此前最后完整公共训练架构是 09-18 十二库统一程序；BrainSync 的实际软件发布接口是 09-20。两者共同作为可恢复的稳定阶段保存，仍各自保留执行来源。

## 实际组成与恢复

大版只保留四个与最终恢复有直接关系的执行源：

| 角色 | 原执行源 SHA-256 | 保留原因 |
|---|---|---|
| 十二库最终主程序 | `c71a4d2109eaed23bea734c76845940bcf05d51a3af4478ac3c1db54fe75fd24` | 真实完整矩阵及默认共享预算/目标路径 |
| 首折已完成后复用源 | `5701d51f04dc1111e30f1bb9058f5e8992383abae3860250a1c8b87e2c3bcbce` | 最终程序实际复用其源模型，不能重标成主程序新 fit |
| 输入准备程序 | `bac9654edfe63610f4775aa34569efc524c782eac81a0496b35a9e0ac79a54f7` | 最终 raw/cache 输入的原准备身份 |
| 工作站实际默认源 producer | `f3267cd466efac946ee4a79ababa11a79472ba78119694e641cc592940ac05e4` | 09-20 默认 BNCI checkpoint 的真实训练来源 |

工作站默认 `bnci_cz6_seed20260906_A01.pt` SHA 为 `0105bb90dc4a5d9a2788a55ff0e183a230e8e5cf8df20dbd601848c67b2221be`，其 source seed 是 20260906，公共 A01 被排除。它与十二库部署源分别记录；前端普通校准任务没有传 `selector_mode=unified` 或新的校准 recipe。

发布 EXE 内 19 个 `gtn_workstation` 模块与所保源码的 Python3.10 code object 匹配；旧 09-17 快照的 app/ui 已与真实发布不同，所以不保留多个完整 frontend 微版本。环境分别使用后端 `pyproject.toml/uv.lock` 与前端 Python3.10 的独立依赖，发布 runtime/EXE 留在原发布目录。

[大版清单与恢复方法](../../frozen/major_domain_transfer_20260920/README.zh.md)记录 exact member/hash、外部模型/结果以及恢复操作。ZIP 95 成员、11,769,175 字节；逐成员 SHA 验证通过。已恢复后端 346 文件，用其真实 CLI inspect 原七导个人包 exit0；默认源严格重建 1,202 参数及 state 成功。这些检查没有新训练或设备操作。

## 保留的科研骨干

统一架构复用共享 EEG 网络、真实 observed mask、实际 membership 评分和完整 requested 失败分母；CE source 初始化、普通 full-fine、CE scratch、波形收缩 LDA 是独立对照。增强源与 prior/decision 是预声明研究臂，不能据各库测试分数动态选路线。

09-16 后稳定预算按阶段解释：source 参考模型使用 participant 验证 CE，patience3/min_delta1e-4/上限12曝光周期；正式 A0/A1 source 共享参考选出的 e*，从原 seed 初始化拟合；目标与源内 λ 开发固定90遍；K0零 fit，确定性 LDA按同输入复用。09-16 三 seed 是冻结 e*/λ 下的优化重复；09-18 完整矩阵单 seed20260918。它们不是三个独立选轮程序，也不是当前科研新的 source 政策。

原始范围与标签独立仍是必要合同：每 K 先裁合法连续 raw 再预处理，校准与 later test 具有独立实际支持；EOG/QC/输入统计只在相应合法校准/源范围估计。source/target recipe、模型 state、身份、原输入与预算共同决定复用，不能只重写结果摘要。

已落地的工程合同集中在源码与现有反例：实际标准化导出、有限数值失败清理、同 bytes 的结果完整性、Windows 无信号进程探测、parameter ownership、来源/通道/时序门禁、持久 cache 准入、worker 资源租约。旧源码 before/after、全部修复过程和逐次测试日志不作为日常架构入口。

## 最终结果与边界

[十二库最终结果](unified12_results_20260918.zh.md)保留 48/48 保存重放、4,082 逻辑评分单元/3,166 去重单元及全部真实失败/覆盖分母。A0Q0/full-fine 是后续神经比较锚点；完整 A1Q1 未显示整体胜过普通适配，不能晋升默认。BNCI 使用全部十次重复的 hit@all，不能将旧 R5 主张推广到 BrainSync 成人性能。

09-16 三 seed 开发程序与最终结果保留[执行/完整结果](unified_results_20260916.zh.md)，09-18 的资源与最终身份保留[最终运行记录](unified12_performance_20260918.zh.md)。工作站 09-20 EXE 模拟启动及前端 291 软件测试通过只证明所测软件路径，未验证真实串口/BLE录制、光学时序或成人识别效能。

大的最终模型/结果包与原 EEG/cache/model/final prediction 保留原位置，不复制进入每一个阶段冻结。过程记录、失败调试、重复源码副本按当前任务依赖与实际最终 producer 身份核对后退役；不能按日期或目录名直接判垃圾。
