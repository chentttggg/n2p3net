# 科研与代码集成：2026-10-08

本记录保存本轮研究与集成进度，活动规则仍由六指南持有。源起点为`711e230346ed6c7c59539c7066119a4784011681`，独立科研分支`codex/scientific-mechanisms`。用户原树`D:/BCI/n2p3-net`的380条并发变化保持原状。

## 本轮任务与当前出口

用户要求持续优化、减少无用检测、实际审查代码、保持全局观和工作树整洁；科研子代理无我方期限/额度。科研窗口持有研究与最终代码集成，云端窗口持有执行、资源调度、杂务和旧B08验收。双方合计最多10并发：科研1主+5子、云端1主+3子；横向审查已归并科研主代理。

| 切片 | 当前动作 | 完成证据与待补出口 |
|---|---|---|
| B07字典可移植 | 编译数值态、物理grid/constraint/producer一次准入；load消费冻结数组 | 已定位旧重编译拒载；新artifact、完整模型/ensemble加载、跨平台和篡改反例仍在实施 |
| B08可识别prior | 物理metric、quotient预测范围、显式completion与换基/单位反例 | 理论及新owner在实施；旧full-rank失败含义保留，新真实fit/gradient/package/CLI与验证待闭合 |
| B05噪声归因 | 已知白化与估参/共享谱失配分开，真实Y/D及不确定性控制 | 研究诊断代码在实施；M1原576格保其既有结论，不由新原型晋升 |
| B06可观测性 | 实际观测map与任务泛函的null空间，完整T和不可识别witness | 新研究入口在实施；已知state软件出口不解释为未知state恢复 |
| B09空间signal prior | 物理reference/range与非separable vec Gaussian后验 | 新owner在实施；不将signal重要性写成手工noise precision |
| B14/B12云基础主线 | 云端作者合并caller用途、显式seed集合与共享资源执行 | 由云端单作者交稳定capsule和真实验证；科研总监逐文件审查与最终集成 |

以上是实际实施范围和未完成状态，不是全部15块/123愿望或公共效能已验收。

## 已确认输入与代码风险

旧B08 cloud replay已确认Gaussian basis重生有121元素末位变化（最大1.11e-16），原数组/模型SHA正确，但旧`ResponseDictionary.from_record`和`_FittedExpert`按本平台重新compile比较而拒载。旧冻结源码、模型及producer保持原身份，运输重放不作为未来生产可移植合同。

当前源码还在`ContinuousDecisionModel.__post_init__`从保存roots重算`dictionary_geometry`后逐metadata digest比较。QR/SVD/norm的容量诊断同样可能出现跨平台末位差。该第二层为已定位的静态风险，尚未完成跨平台全路径复现。修复须同时绑定actual basis、prior、fit统计量和原producer，不能只接受重新自签的summary/hash，也不能仅改一个decoder后宣布整包通过。

源码审查发现的外部准入与内部重复校验分别处理：外部artifact/spec/fit scope/时间支持须验证，已验证且不可变的内部数值态由owner共享。新compiler或模型schema必须记录新实际执行身份，旧产物不迁名或补签。

## seed与验证范围

科研云训练声明一对seed，当前使用20260918；四个独立source checkpoint流程各自选择与fresh refit，初始八个逻辑fit。大版本校验使用20260918/19/20三个独立source→target程序。source catalogue/caller按实际声明集合完整准入，政策由云侧source作者统一落实，科研侧不竞争修改其WIP。

单seed参与者cluster区间条件于冻结模型，不代表完整训练随机性。历史三seed计划/旧B08固定20261006矩阵保持原执行范围。必要数学、行为和资源验证交唯一云owner，保存实际冻结源码、命令、stdout/stderr、退出码与产物；本地仅相称的阅读、Ruff、compile和轻量检查。

当前没有本轮新增机制的云数值通过记录或新效能结果。完成状态将按实际代码、独立反例、完整调用链、保存重放和云证据更新。
