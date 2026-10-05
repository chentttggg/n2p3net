# 十二库统一框架数据分析的批判评估（2026-09-18）

## 总判断

这轮的**执行完整性高，统计推断强度中等偏低**。原始输入、source/target 身份、校准/test 行、模型保存重放和失败分母都做得很严谨；48/48 重放通过使“代码按声明执行”可信。当前分析可以支持公开数据上的条件性开发结论，但还不足以支持“完整联合框架优于普通 full-fine”“decision loss 已被验证有效”或“新增 source 数据带来收益”这类总体性结论。

现有结果最稳妥的表述是：在冻结的单 seed、固定四库目标和固定预算下，source 初始化相对 scratch 有明显收益；A1 源辅助目标和 Q1 prior/decision 适配没有显示稳定附加收益，部分场景出现负迁移。

## 已经做得好的地方

- source study 和 target study 按整库留出，目标 test 行有独立 digest；分析脚本还核对了每个 arm 的校准/test 行一致性。
- participant identity、source snapshot、model state、prediction 和 result record 都有摘要；保存模型重放 48/48 通过。
- 普通折与 BI 家族敏感性折分开；部署源没有被冒充成外层评价成绩。
- 采用 participant macro 并保留 study equal 权重，避免 BI2014a 的人数直接压过小库；同时报告逐库差异、负迁移比例和最差被试。
- source-only λ/选轮在 target test 之前完成，目标不按测试成绩挑 recipe；K=0/200/800、实际 K 和 raw scope 被写入任务记录。
- 将 CE scratch 和 decision scratch 分开接入，方向正确；BNCI 的物理 decision 指标也单独分析，没有把它伪装成成人九选一。

## 关键问题

### 1. Q1 在四个目标库上不是同一个处理

Q1 的定义是 trial CE + source prior；只有 BNCI2014-008 有真实 membership 时才额外加入 decision CE。BI2014a、BI2015a、Huebner2017 的 Q1 没有 decision loss。于是普通四库的 `A0Q1/A1Q1` 平均值把“prior-only 目标适配”和“prior+decision 目标适配”混在了一起。

同样，普通四库的 `Scratch` 汇总在 binary-only 目标上是普通 scratch，在 BNCI 上是 decision scratch；`ScratchCE` 又通过等价别名补入。`unified_minus_scratch` 因此不是一个监督定义一致的单一对照。

这不破坏代码执行，但会破坏处理效应的统一解释。当前报告中“统一包相对 decision scratch 的提升”不能当作四库共同结论。

修正方式是把 estimand 拆成三块：

1. binary-only 三库：Q0 vs prior-only Q1，CE scratch，LDA；
2. BNCI：Q0 vs prior+decision Q1，CE scratch vs decision scratch；
3. BI 敏感性：只做 binary-only 的 source/target 机制敏感性。

### 2. 单 seed 的训练不应被 participant sign-flip 当成多 seed 证据

当前只有一个 source/target seed=`20260918`。分析把每个 participant 的 paired AUC 差异拿来做 sign-flip p 值，这可以回答“在这组已经冻结的模型上，参与者差异是否围绕 0”，但不能回答“换一次初始化、shuffle、source fit 或 target fit 后方法是否稳定”。participant 是测试抽样单位，不是训练随机性的重复实验单位。

代码已经把区间标成 `conditional_frozen_models`，这是正确的限制；最终叙述仍应明确：p 值和 bootstrap 区间没有覆盖 model-training uncertainty。要评价方法稳定性，至少需要 3–5 个预先声明的配对训练 seeds，或者使用保存的多 seed 模型层级汇总。

### 3. 多重比较校正范围不够大

当前 Holm 只覆盖每个 panel 的三个 K=200 primary contrasts。与此同时，报告还查看了：

- K=800；
- ordinary 与 family 两个 panel；
- source-at-Q0、source-at-Q1、target-at-A0、target-at-A1、LDA 等 secondary contrasts；
- BNCI 的多个 arm、K 和 repetition R；
- 负迁移比例、K=800 的方向性判断。

如果这些结果都被当成 confirmatory，当前校正不够。应预先声明一个主要 estimand（例如 ordinary binary-only 的 A0Q0−A1Q1 at K=200），其余全部标为 exploratory，或者按“panel × budget × contrast family”完整控制 FDR/Holm。BNCI 的 arm×K×R 应使用 participant-clustered interval 和明确的探索性标签。

### 4. K=200 和 K=800 的完整被试集不同

普通折 K=200 有 123 个完整 participant，K=800 只有 72 个。当前报告把两档 AUC 并排解释为校准剂量变化，但它们同时改变了可分析 participant 组成。K=800 的差值不能直接当成同一批人的剂量曲线。

应该同时报告：

- exact-K estimand：接受不同完整 cohort，但明确是不同 cohort；
- common-K800 cohort：只保留 K=800 可用的人，再比较 K=200 与 K=800；
- participant × K 的层级模型或 cluster bootstrap。

否则“K=800 联合包进一步变差”可能部分来自短记录/难被试退出，而非 recipe 随预算变化。

### 5. 不能从当前矩阵归因“新增数据集带来收益”

十二库被放入 source，但当前对照主要是 A0 vs A1、Q0 vs Q1、scratch/LDA；没有同预算的旧 source 集合 vs 十二库集合，也没有 leave-one-new-source-out 或 source-set factor。因此当前数据能证明“十二库配置下发生了什么”，不能证明 BI2012、BrainForm、Cattan 等新增库各自贡献了什么，甚至不能证明扩源优于九库。

后续应固定 target、seed、source update budget 和 target recipe，加入 matched `old-source-set / twelve-source-set` 对照，并把新增库只用于 source 的角色写入 estimand。

### 6. sign-flip 和 study-equal bootstrap 的假设需要降级表述

participant sign-flip 依赖差异在零假设下近似可交换/对称；AUC 差异有边界、离散 test 样本和明显的 study/domain 异质性。当前 `paired_interval` 是 participant 内 bootstrap，overall interval 再对每个 study 分层后等权平均。它适合作为固定 study、固定模型的描述性区间，但不是完整的跨 study 泛化区间。

建议主报告增加 participant-nested-within-study 的 cluster bootstrap 或层级模型，并把现有结果命名为“conditional fixed-study interval”。不要把 `p<0.001` 读成跨数据集或跨训练过程的强证据。

### 7. BNCI decision 分析仍缺少成对不确定性和监督定义对照

当前 BNCI 报告了 hit、coverage、NLL、Brier，coverage 约 0.966，且按 R=1/5/10 展开，这些都是优点。但没有对 A0Q0、A0Q1、A1Q0、A1Q1、LDA、两种 scratch 做 participant-clustered paired interval 或主要对比检验；K 的多重查看也没有校正。

`hit_operational` 把物理上不完整的 decision 放进分母，`hit_complete` 又只看完整 decision。两个指标都应该保留，但不能只报点估计；应同时报每个 participant 的 complete/operational 分母、coverage CI、相对 36-choice chance level 的区间，并把 R=5 预声明为主要 repetition。

### 8. 选择不确定性没有进入最终区间

source λ 和 source exposure 选择是 source-only 的，避免了 target test 泄漏，这是正确的；但选出的 λ/e* 被当作固定真值。最终区间没有传播 source participant split、λ 候选和 early-stop 选择的不确定性。

这不要求重做当前训练，但报告中应把结果称为“conditional on the frozen source selection”。多 seed 或重复 source split 后，才能评价选择规则的稳定性。

### 9. 研究问题和产品问题仍需更强分离

AUC 是 binary ERP 目标，BNCI 是 36-choice physical membership 聚合；两者都不是 BrainSync 成人九选一 hit@R5。当前报告已经写了“不是成人确认”，但产品层面仍应禁止把 AUC、BNCI R5 或 deployment source model 作为工作站默认 recipe 的晋升证据。

## 推荐的下一轮分析顺序

1. 先重算三套清晰 estimand：binary-only、BNCI decision、BI sensitivity；删除混合 Q1/scratch 的总平均作为主结论。
2. 以 K=200 ordinary binary-only A0Q0−A1Q1 为唯一 primary contrast；K=800、family、LDA、BNCI 和所有机制拆分标 exploratory。
3. 输出 common-K800 cohort，报告 K×recipe 交互，而不是把两个 cohort 的均值直接当剂量曲线。
4. 对 BNCI 固定 R=all 为主，补 participant-clustered CI、coverage CI、chance-level 对比和 paired arm differences。
5. 做 3–5 个配对 training seeds；把 seed 作为训练重复单位，participant 作为测试抽样单位，使用两层 bootstrap 或层级模型。
6. 单独设计 old-source-set vs twelve-source-set 的 matched source ablation，才能回答新增数据集是否有贡献。
7. 保留现有 digest/replay/integrity gate；它们是当前分析最可靠的部分。

## 结论等级

- **高可信（执行层）**：数据和身份边界、固定 test 行、recipe 记录、失败分母、模型重放和归档完整。
- **中等可信（条件开发层）**：在固定单 seed 和固定 study 集合下，source 初始化优于 scratch；Q1/增强源没有稳定附加收益。
- **低可信（泛化层）**：多 seed 稳定性、跨 study 总体优势、新增 source 数据贡献、成人 BrainSync 产品收益。

这轮分析适合作为机制筛选和下一轮预注册设计的开发证据，不适合作为最终方法胜出或产品默认决策的统计确认。
