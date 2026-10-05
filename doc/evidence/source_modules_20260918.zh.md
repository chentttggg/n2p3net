# 源数据接入与源池排程模块化验收

日期：2026-09-18。本次是源码、合同和输入探针验收，没有启动云训练或重新计算迁移效能。
B7的BI2013a/EPFL/Chailloux/Speier/GTN在09-09已完成扩源训练；09-16统一模型固定四库是实验设计。
既有九库结果见[源池完整结果](source_eog_training_20260909.zh.md)，不因本轮重构改写执行身份。

## 实现

| Module | Interface承担的行为 | 实际调用方 |
|---|---|---|
| `data.source_parsers` | 连续原件、真实事件/标签、物理身份及格式伴随文件；增加BrainForm/Cattan/BI2012 | `data.source_records`；准备CLI先验证manifest |
| `data.source_admission` | 完成账本、NPZ/record摘要、物理研究库、输入合同和数量核对；ready/pending/failed；运行输入绑定 | 源池调度器；源cache仍按attested不可变读取 |
| `research.source_catalog` | 源池成员及外折排除；新组合不能覆盖旧池名；声明与实际来源逐项相等 | 源池任务编译、P1/P2/统一source chain |
| `research.source_matrix` | 池、目标、排除、seed、全量/等量任务；从完整比较组冻结剂量 | 源训练和公共评估消费同一JSON计划 |

省略新计划时仍有原153任务。配置示例为[新源池模板](../../configs/source_pool_expansion.example.json)，
默认参数和历史研究池不因“candidate”标签新增禁用开关。任意来源数须通过显式source_studies或source_pool声明，
旧未声明的源链保持原三源/家族两源的合同。common-three继续要求真实完整通道，不伪造GTN缺测。
S2单源trial剂量由128乘实际来源数得到；旧三源仍为384。扩展池仍保留参与者数与trial剂量不同的解释。

等量任务的单任务探针会带入同外折所有比较池/seed定配额，避免只选一个任务改变其P/D。
准备失败、未完成与完全缺失分别记账，池不能静默缩小。显式有时限的等待支持准备与拟合重叠。
`input_bindings.json`保存实际manifest、cache、record及处理方法；输入改变后不得复用同一运行目录。
新源运行记录为`n2p3_explicit_source_matrix/2`，原/1产物保持原身份。

公共评估共享源池计划，以第一个声明池作为完整微调/scratch对照载体、首个seed执行确定性LDA，
避免新池和新seed遗漏控制。额外目标的真实记录顺序须由target-protocols显式提供。
未完成源输入会产生对应失败请求，最终有失败时不再报告completed；保存模型验证器继续按每份实际配方重放。
历史本人EOG矩阵保持原专项设计，通用个人校准仍通过已有checkpoint/SubjectAdapter消费新源。

## 新格式及原始输入

- BrainForm只取原CSV中按计数器和target/nontarget互补关系复核的校准前缀；保留原Pxx，
  不按原分类器成功与否筛选。按已审计BIDS规则裁1245个启动采样，事件仍保存原采样号。
  显式坏导必须有排除依据，Cz不得作为可修补坏导。推理段不制造监督标签。
- Cattan使用原22列连续MAT及PC/VR条件，不把block/repetition切成独立滤波起点。
  原subject_01_VR只能显式舍弃已知不完整末采样，不能裁任意有限尾部或丢掉刺激。
- BI2012复用既有MOABB连续parser和training/online原件；保留Fz为ground的坏导声明，
  源投影前剔除，不参与实测EEG输入。
- Chailloux实际消耗vhdr、eeg和vmrk，现在同时绑定三者摘要；标记文件改变也会拒绝旧准备复用。

真实数据无训练探针由本地`tmp/source_modules_20260918/probe.py`执行，同一共享预处理后保存并attested重载：

| 输入 | 保留事件/排程 | 标签0/1 | 六槽观测mask |
|---|---:|---:|---|
| BrainForm P01单记录 | 600/600 | 540/60 | 1,0,1,0,1,1 |
| Cattan subject_01_VR完整连续记录 | 720/720 | 600/120 | 1,1,1,1,0,1 |
| BI2012 subject_01 training | 768/768 | 640/128 | 0,1,1,1,0,0 |

这些探针验证输入接入及算子/保存链，不认证整库质量或模型准确率。
另从已审计BrainForm索引生成167条记录/24人的原始输入manifest，每个ZIP成员SHA256均重新核对；
保存于本地`experiments/runs/source_modules_20260918/manifests/BrainForm.json`，未据此声称全量cache已完成。
Lee下载账本仍无最终complete/全部文件和转换认证，本轮不填造其来源或标签。
Mainsah保持外部角色；两者的研究资格不由通用池定义替代。

## 验证与物理快照

定向测试覆盖旧153任务、新池/家族排除、等量任务选择不改变剂量、真实三/四源cache进入源链、
标签与缺导反例、伴随文件篡改、伪completed、身份错配、输入替换、CLI预检/续跑及新seed控制。
Ruff通过；从仓库外运行准备和公共评估CLI帮助入口通过；git diff --check通过。
最终源码下，**2092个不同测试已通过、4项原有环境跳过**，逐测试清单见
[验证账本](../assets/source_modules_20260918/validation.json)。这是隔离全套与本地缺失资源补测的并集，
不是把重复运行相加：隔离全套2080 passed、15 skipped，另1项因source-only快照不含dashboard.html而失败；
在字节一致的原项目上，22项采集端集成测试及6项HTML/真实数据测试全部通过，补齐其中11项额外跳过与1项缺文件失败。
原有4跳过为两个未配置独立producer入口及两个Windows符号链接权限场景，没有新增长期跳过。
先前工作树全套中的一项测试替身误传mp_context已修正，冻结后的对应测试通过。

最终源码已[冻结并物化](../../frozen/source_modules_20260918_acceptance/source.manifest.json)，
345个成员，归档SHA256 `bac9654edfe63610f4775aa34569efc524c782eac81a0496b35a9e0ac79a54f7`。
本地补测前，实际源码与该冻结副本逐文件核对一致；
[源码核验](../assets/source_modules_20260918/source_verification.json)与
[本轮文件范围](../assets/source_modules_20260918/change_scope.json)一并保存。
完整日志、原始输入探针及manifest生成脚本另行物理归档于
[证据清单](../../frozen/source_modules_20260918_evidence/manifest.json)。

修改前工作树已物理保存于[阶段开始快照](../../frozen/source_modules_20260918_before/source.manifest.json)，
其中包含当时尚未完成的修改，不视为已验收版本。既有并行工作、原始数据和训练结果未回退或覆盖。
