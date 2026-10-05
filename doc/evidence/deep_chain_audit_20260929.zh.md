# 历史无迁移与当前迁移不一致：全链深度审计

日期：2026-09-29。范围：历史LOSO、本人/公共目标scratch与LDA、旧四库迁移、training-chain、P1/P2、09-16与09-18统一矩阵，以及相关输入、候选评分、模型保存和复用接口。

本轮是调查，不是新性能实验。生产源码、原始EEG、旧模型、缓存、预测和历史归档均保留；仅新增独立复算、攻击性反例、诊断图和审计报告。入口时417份源码/配置/活动指南已有逐成员校验快照，见[入口身份](../assets/deep_chain_audit_20260929/entry_manifest.json)。已完成的实现审查不等于漏洞已修复。

## 1. 结论

**确实找到了隐藏问题，但没有证据把当前A1/Q1的负结果统一解释成“AUC算错”。** 已检查原始预测的主指标与保存记录一致；跨运行数值对不上有已证实的测试队列、输入算子、训练剂量和估计量变化。同时，EA、最小测试量、普通基线验证损失、候选评分、结果复用等路径存在能实际触发的实现或合同缺口。

最值得优先处理的是：

0. **当前独立短raw双向处理会把强带外工频干扰变成毫伏级边缘坡尾。** BI2014a-01的同一末尾事件，在独立C200处理后达到3.545 mV，完整record双向或独立C200 forward仅为几十µV量级。纯50 Hz、无ERP的独立算子反例也能产生最高5.369 mV，且与SciPy同padding逐值相同：问题是有限边缘策略与真实大干扰的组合，不能简单说自研滤波实现算错。
1. 同名K、full-fine、scratch、zero-shot并不绑定同一个测试集或训练过程。相同checkpoint的BI2014a-13，test从299条改为96条，AUC从0.57494变0.76172；共同96条上两边AUC完全相同。
2. `minimum_test_trials=200`不是有效的硬下限；回退分支只要求剩余test非空，甚至能返回不足200且调整记录为空。三代真实实验都出现声明200、实际仅85条test的成功单元。
3. EA绝对正则化下限`1e-6`直接作用于V²二阶矩。在真实Huebner切片上它约为平均二阶矩的4579倍，所谓EA几乎退化成统一乘1000。历史EA缓存还把变换坐标继续声明为V，存在人员排除后保留旧fit state及静态缺导被改成已观测的问题。
4. 普通基线的验证loss随验证batch划分改变，独立反例会反转候选模型排序；但统一矩阵固定更新源训练使用另一条正确的可加验证路径，不能把该漏洞归到所有源模型。
5. 候选总体hit与分R hit缺候选口径不一致；开启全组中心化时，后续重复分数可改变早期R的结果；valid分析在replicate缺失时改变比较支持集合。这些是实证API问题，尚未证明它们污染了十二库binary AUC。
6. compact跨运行结果导入可以复用旧预测，同时给cell和新evaluation contract写入新的evidence scope及执行snapshot。已通过真实executor无拟合复现，不是仅凭读代码推断。

**当前应先收敛比较身份及这些接口语义，再据匹配对照判断方法优劣。** 修正老报告的错误解释不应改写老预测；发现某条旁路有漏洞，也不能倒推所有旧成绩无效。

## 2. 审查方法与实际覆盖

使用astra-research-workflow、eeg-p300-research、scientific-critical-thinking、experimental-design、statistical-analysis、diagnosing-bugs；信号图使用scientific-visualization。三条并行审查分别负责输入、评分、可比性，主审查覆盖训练/重载/复用并逐项复核关键源码与反例；随后补个人模型链审查。

| 证据层 | 本轮实际动作 | 能说明什么 |
|---|---|---|
| 执行源码 | 初始417份文件快照；相关当前文件与产生结果的冻结源码逐字节比对 | 不把dirty HEAD或后来修改的计划冒称原执行源 |
| 原始预测 | 旧四库3862、09-07 chain已取子集37568、P1/P2 32238、09-16 unified12150份NPZ条目 | 合计85818条目；包含重复基线，不等于85818次独立实验；两分支交叉复核部分归档，不重复累计 |
| 独立评分算法 | 其中48250份核AUC/BACC；其中P1/P2与09-16 unified的44388份另核AP/Brier/NLL，不调用原指标实现 | 排除对应产物的评分公式、标签方向及保存指标不一致；旧3862份并未统一保存全部概率指标 |
| 原始EEG | Huebner真实BDF/events；BNCI八份MAT；BI2014a原CSV/MAT；BrainForm/Cattan真实切片 | 核单位、事件位置、binary/membership与接口处理；样本范围不冒充全库质量认证 |
| 真实算子差分 | 同一raw、同一事件，整段/隔离前缀处理；修改范围外raw；mask污染、重排、拼接、保存重载 | 能击穿“同名profile即同输入”和“shape相同即同物理含义”的假设 |
| 实际入口攻击 | 当前EA CLI、prepared verifier/loader、候选summary reducer、target executor与验证loss入口 | 不只测试自写公式；验证真实调用链可以接受错误语义 |

主指标复算：P1/P2与09-16 unified AUC/AP/BACC误差为双精度舍入级，旧四库AUC最大误差约`2.98e-8`；同run同人K跨臂/seed的已核test行和标签零异常。对同输入/seed/layout的4526组重复scratch/LDA，score数组完全一致。chain→P1的305个同cache/同人/同K/同行/同标签LDA单元，AUC全部相同；代表score数值相同而存储dtype不同，说明不同字节hash不能直接当性能变化。

**缺口必须保留：** 旧0.74511 LOSO目前找到的是冻结原报告，尚未找到原record/预测；09-18完整`verified_execution.tar.gz`（11,990,531,333 bytes）本地未找到，本轮没有重新核验它的逐cell模型/预测。09-18的分析摘要、执行脚本和收据可审，但不能以09-16的12150份复算替代09-18。缺少原包不等于其结果错误。

详细来源与脚本分别见[可比性报告](../assets/deep_chain_audit_20260929/comparability/report.zh.md)、[独立评分报告](../assets/deep_chain_audit_20260929/scoring/report.md)、[评分小摘要](../assets/deep_chain_audit_20260929/scoring/summary.json)。大型逐cell明细保存在本地`experiments/runs/deep_chain_audit_20260929/`，不进入Git分发。

## 3. 哪些变化已经足以解释“数字对不上”

### C1. 旧高分LOSO不是当前few-shot的无迁移对照

旧0.74511的冻结报告对应BI2014a 64人、16导、2–30 Hz、[-200,800) ms、128点、零相位IIR、最多30轮、折内QC及同库LOSO；后来排除62–64。当前是61人、Cz六差分与缺导mask、0.1–30 Hz、[-200,1200) ms、179点、整库留出、本人前缀校准后测后缀，且后来改为双向+notch。

“同库其他人的大量数据训练后留一个人”与“未见整个目标库、只有本人K个标签”是两个估计目标。既不能把0.74511直接要求为当前K200的下限，也不能拿现在的scratch低分说明旧模型从未有效。旧0.74511尚未原预测重算，最终仍需恢复其原件。

### C2. 同checkpoint的K0会随同批最高K改变测试队列

`src/transfer/binary.py:89-146`按最高实际K确定共同后缀，这保障**批内**各K公平；但最高K不同的两批，K0也测不同trial。

| BI2014a参与者 | 旧test/AUC | 新test/AUC | 共同test上的两边AUC |
|---|---|---|---|
| 13 | 299 / .574939 | 96 / .761719 | .761719 / .761719 |
| 02 | 300 / .570080 | 96 / .699219 | .699219 / .699219 |
| 25 | 408 / .521972 | 120 / .402000 | .402000 / .402000 |

123人中69人发生test数量变化；共同标签全部一致，score最大变化`4.77e-7`，共同test AUC全部一致。这是原产物解释，不是训练出来的新结论。修复比较方式应保存独立的cohort/原始事件身份，不能只以`participant×K`配对跨批分数，也不能倒改历史test。

### C3. “epochs=60”并不表示P1只训练60遍

| K | 09-07 chain | 09-16 P1/P2 | unified |
|---|---:|---:|---:|
| 200 | 60 updates / 12000 draws | 200 updates / 40000 draws | 90 updates / 18000 draws |
| 800 | 120 updates / 48000 draws | 200 updates / 80000 draws | 180 updates / 72000 draws |

P1 recipe同时设置`max_updates=200`，覆盖epoch上限含义；K200相当于200遍。顶层计划中的60不是实际预算。P1还用online BN和checkpoint dropout；unified改为校准开始时估计BN后冻结、关闭dropout。source端另改变滤波、notch、observed-channel BN及feature residual。

独立代码探针确认：unified冻结BN行为符合配方；人为将其上游卷积乘3后，真实输入方差增至9倍而buffer不更新。这说明它与旧online BN不是同一训练过程，不证明冻结BN必然有害。匹配scratch的1279参数架构和独立seed初始化则通过反证，没有发现偷继承source权重。

### C4. 同名双向滤波profile不等于同一有效输入

真实Huebner476个事件、112个target与events.tsv逐项匹配。对同一段raw的C64：先整record双向滤波再截取，与先隔离C64 raw范围再处理，最大差约30.66 µV、RMS约3.21 µV；forward路径这项差异为0。对C64范围之外注入10 mV扰动，当前独立raw视图的C64逐位不变。

这既支持新raw隔离实现，也解释为何换成独立短raw后，即使滤波参数JSON没改，旧LDA/NN分数也不应被期待原样保持。没有证据可把这3.21 µV直接折算成某个ΔAUC。

### C5. 归约单位会改变结论方向

旧K2000真实预测仅21人、两库。库等权/库内participant宏AUC的full−scratch为`+0.003961`；把同一批trial混在一起算一个AUC则为`−0.000524`。后者还比较了不同个人模型之间的绝对score偏置和尺度，不是同一估计量。

旧LOSO BACC使用校准阈值及`>=`，当前矩阵明确记录raw-zero阈值及`>`。字段含义不同，但本轮未发现这些阈值被错误送入主AUC。不能把BACC差异与AUC差异混为同一种退化。

## 4. 已复现的实现与接口问题

严重度按所影响的科学结论和入口决定；“可复现”与“历史已触发”分栏，不以漏洞数代替证据质量。

### F0. 合法的独立raw边界仍可产生严重信号失真

这是本轮进一步追查原始波形后发现的当前输入质量问题，优先于继续调A1/Q1损失。

BI2014a-01原始第200个flash（零基index199），sample64734、126.4336秒，独立C200 scope右端为sample65349，恰为该epoch最后所需样本之后。原始Cz存在约±8 mV强周期干扰，P3−Cz末10秒主要分量约50 Hz。当前0.1–30 Hz+50 Hz notch、`mne_reflect_limited`双向处理在该人工截止点附近生成缓慢上升尾部；同一事件经完整epoch重采样/基线后峰值3.545 mV。完整record双向和独立C200 forward均不出现同量级坡尾。

本轮实际算子分解：finite-minus-full在该事件P3−Cz上的RMS约1522.88 µV；原生P3贡献127.22 µV、Cz贡献1650.10 µV，逐点线性分解误差约`5.6e-15 V`。拟合到末10秒的50.012187 Hz单正弦振幅6687.6 µV，其边缘差贡献RMS1629.51 µV，残差106.63 µV。分量可相消，不能把7%残差换成“93%的性能损失已解释”。

在同200条事件上，C200独立与完整record零相上下文使P3−Cz输入列的裸总体std从约61.09 µV变为207.66 µV，约3.40倍；实际normalizer另加1 µV，分别约62.09/208.66 µV。这里的full-record零相不能与独立forward混称：后者实际normalizer约82.9 µV。normalizer公式即使完全正确，也会被上游端点失真改变。全record仅是算子参照，不是允许把未来test EEG接回校准的修复方案。

为什么50 Hz被notch声明抑制，却还能产生慢坡：`filter_bidirectional_segment`的右端odd延拓为`2*x_end−reverse(past)`。大幅正弦的截断端点值不为零（sinφ≠0）时，延拓包含新的常数/低频结构，稳态notch幅频响应不能保证这些有限段边缘项消失。0.1 Hz高通的长时间尺度进一步使影响进入最后多个epoch。

独立纯算子反例固定同一截止sample、8 mV纯50 Hz、16个截断相位，输入没有任何ERP：独立双向最后epoch峰值范围约0–5369.03 µV；延长30秒的诊断上下文最大1.187 µV，独立forward最大`3.72e-10 µV`。当前手工SOS实现与SciPy相同odd padding的最大差为0，排除该反例中“自研实现与标准算法不等价”的解释。证据：[probe_boundary_sinusoid.json](../assets/deep_chain_audit_20260929/training/probe_boundary_sinusoid.json)。

该问题不属于训练/测试标签泄漏；当前raw隔离反而已证正确。缺口是“时间授权合法、张量有限、执行hash一致”没有同时证明短支持输入的信号保真。现行指南也明确自然端点只作诊断，不能把这一设置读成已保证端点可靠。zero与forward还改变幅频响应（双遍为单遍幅度平方），全库phase差异不能自动等同纯边界效应；F0的边界定位另外依赖同一zero算子的full-context对照和纯正弦反例。

**原始强污染与新增右端效应并存。** 49号C200全部200个epoch的至少一导PTP>500 µV，full-record仍全部异常；最坏event46距右端约107秒，full/prefix逐点差0，原生Pz存在明显污染。26号prefix179/200异常、full和forward均173/200；最坏event2距右端约121秒，full/prefix逐点差0，原生Oz raw PTP约30.8 mV，同时该人另有约2.925 mV的尾端附加差异。零差只排除新增C200右端的贡献，**不能排除共有的录制左端初始化或其他滤波失真**，尤其26号event2仅在录制开始约6秒。01的独立forward全C200也仍有最高约1.405 mV的其他异常；只有所定位last event回到几十µV。

因此不能通过一句“改forward就解决”处置全部数据。500 µV等阈值仅用于固定描述性统计，不作为本轮自动剔除规则；全部参与者与原始记录保留。窗口研究也须区分raw支持范围和epoch读出窗口：短窗成绩改善可能仅仅躲开末端污染，不能直接解释成晚期ERP无用或噪声占比已知。

原始波形与同事件对照：[bi01_raw_extreme_anchor.png](../assets/deep_chain_audit_20260929/input/bi01_raw_extreme_anchor.png)；线性分解：[bi01_boundary_linearity.json](../assets/deep_chain_audit_20260929/input/bi01_boundary_linearity.json)。

#### 全61人校准侧扩展

对01–61每人按相同生产路径重建独立C200，zero和forward各12200 epoch，0失败；逐人raw SHA、授权scope、事件、标签和mask配对一致。共34/61人出现至少一个PTP>500 µV的epoch，21/61人出现>1000 µV；zero的宏平均>500比例7.52%，中位0.5%，差异高度异质。

zero下36人的最大峰值位于最后20事件，26人恰为event199；34名>500者中20人的峰值在末20，17人恰在event199，7人超限仅发生于末20。01的最后20事件贡献P3−Cz输入方差约90.9%、Pz−Cz73.1%、P4−Cz89.0%、Oz−Cz89.0%。这些是范围/分布读数，不是性能归因比例。

对照forward，末20的>500 epoch从209降至125，但全C200从918升至985。49号两phase均100%，26号89.5%→86.5%，51号53%→61.5%。phase差异包含幅频响应和左右端处理，不能据此挑一个“全库更干净”的默认phase。

完整固定规则、分母、逐人数据与图见[61人质量报告](../assets/deep_chain_audit_20260929/bi_quality/report.md)和[全员图](../assets/deep_chain_audit_20260929/bi_quality/bi61_c200_quality.png)。

#### 全61人实际测试侧扩展

进一步按生产`resolve_calibration_splits`及`prefix_raw_scopes`解析K=(0,200,800)、minimum=200，对61人的**实际test raw范围**与同record全长zero处理的相同事件进行诊断对照。61/61完成、0失败、0缺失；16,499个test事件的原sample、标签和mask逐项一致，没有模型拟合或新AUC评分。

test长度85–948，中位175，41/61低于声明minimum200；20人最高K800，其余按既有规则回退355–796。本轮真实重建与历史unified测试数量范围一致；旧chain/P1曾请求到K2000，不能拿其更短test误判为原始事件丢失。

四导每个人最大的scope/full差都在test首20事件；首20贡献差能量的跨人中位数为99.927%–99.947%，末20近零。每人最大绝对差的中位数93.26 µV，13人>500 µV、7人>1000 µV，最大约3665 µV。01的P3−Cz差RMS74.35 µV、最大1495.97 µV，最坏是test第0条，Pz−Cz首20的PTP>500比例为独立test35%、full-context0%。

| 同一16,499条test的诊断计数 | 独立test zero | 同zero full-context参照 |
|---|---:|---:|
| 首20、任一导PTP>500 µV | 96/1220 | 50/1220 |
| 末20、任一导PTP>500 µV | 90/1220 | 90/1220 |
| 首20、任一导PTP>1000 µV | 37/1220 | 13/1220 |
| 末20、任一导PTP>1000 µV | 42/1220 | 42/1220 |

这里两边phase相同，因此与前述zero/forward对照不同，直接定位了**支持范围**的影响；两边共有的原记录末端和原始污染没有被“消除”。full-context仍仅为oracle诊断，可能读取合法test起点之前的校准EEG，不能作为当前独立test协议的无说明替换。只证明波形输入变化及其位置，不量化它解释多少AUC或交互。

证据：[test_scope_summary.json](../assets/deep_chain_audit_20260929/bi_quality/test_scope_summary.json)；所有逐event统计保留在本地忽略目录。至此校准和测试侧均完成全61人固定规则波形核查，不等于全部record的信号质量认证。

### F1. 最小测试量放宽后仍沿用原minimum声明，部分情形不记原因

位置：`src/transfer/binary.py:118,131-144`。初始检查要求test≥minimum；进入回退后可行性只检查test>0，最后没有重新绑定实际测试量政策。若实际K没有改变，`adjustments`甚至为空。

- 无训练反例：N300、K200、minimum200，返回test100且`adjustments={}`；独立Train/Test情形也可触发。
- 真实Huebner：请求K0/200/800、minimum200，准备为K0/200/380，test91；`verify_prepared_prefixes`及`load_target_prefixes`均接受。
- 历史实证：三代BI2014a participant6/K200都有test85且completed，原plan minimum200。

用户已明确允许短数据约80%回退，**80%政策本身不是本报告指认的bug**。缺口是minimum到底是硬门槛还是触发回退的期望值未分开，放宽后声明和账本不完整。不得因此宣布旧85条结果计算错误；但它们有明确的小样本精度限制，不能被“至少200条测试”描述。

证据：[minimum_test_probe.json](../assets/deep_chain_audit_20260929/input/minimum_test_probe.json)。

### F2. EA的数值正则化使用了不合适的绝对V²尺度

位置：`src/data/euclidean_alignment.py:58-61,169-174`；EA CLI默认同为`1e-6`。代码执行

\[
R'=R+\max(10^{-6}-\lambda_{\min}(R),0)I,
\qquad W=(R')^{-1/2},
\]

而R是以V为单位输入的二阶矩，量纲V²。真实HuebnerC64的floor/平均二阶矩约4579.5，W与`999.904 I`相差仅0.0123%；输出白化残差`||WRWᵀ−I||F/√C=.99978`。剥离另一个normalizer绝对epsilon的影响后，EA前后逐导标准化信号仅约0.0118%相对RMS差：该条件下几何变换几乎被后续标准化抵消。

历史BI target/source EA state的白化残差为.9618/.9298；记录的condition1.45/2.43描述的是**加floor后的参考矩阵**，实际变换输出condition仍76.98/116.17。不能用漂亮的condition字段证明实际白化成功。

限定：正则化本来允许不精确白化，这里问题是默认尺度对真实单位产生压倒性作用、且不随V/µV表达等价变化。需要单位明确的相对尺度及实际输出残差，不是直接把floor设成零。未评估修正后任何真实效能收益。

### F3. EA输出仍被当作物理V，QC字段发生语义污染

位置：`experiments/prepare_euclidean_alignment_cache.py:149-170`保留`preprocessing=dataset.preprocessing`，对W变换后的数据重新生成`channel_std_v`等物理量。理想逆平方根白化后的坐标是无量纲/变换特征，并非原电极电压；也不再对应同一个头皮通道方向。

现存target eligible61 cache由当前attested loader接受，仍声明signal_unit=V；其中QC channel_std_v中位数约.0304，“峰值电压”约9.59 V。这些是变换坐标，不能再拿生理µV门限解释。

应分开原始物理EEG与模型特征state，并使模型、normalizer、QC消费者核验坐标合同。仅加`provenance.data_alignment`旁注不能让所有仍按V处理的消费者自动正确。

F2/F3证据：[ea_units_probe.json](../assets/deep_chain_audit_20260929/input/ea_units_probe.json)。

### F4. 删除不准入参与者的行，没有撤回他们对EA state的影响

真实`cloud_inputs/*_eligible61.record.json`的lineage顺序是先EA，再`exclude_author_rejected_participants`，且`retained_rows_unchanged=true`。

| 产物 | 当前行数 | EA fit sample_count | 排除后仍进入原fit的行 |
|---|---:|---:|---:|
| target calibration eligible61 | 12252 | 12900 | 648 |
| source rows eligible61 | 57437 | 61011 | 3574 |

过滤数据行与撤销上游拟合不是等价操作。当前元数据允许“当前只剩61人”与“W仍来自64人”共存；单看训练身份白名单会漏掉这类间接依赖。

**影响边界：**已定位旧64人EA执行记录，但当时的64人口径与后来排除政策不同；本轮尚未找到eligible61缓存被后续真实训练消费的record。不能倒扣旧64人开发成绩，也不能把这项问题归到不消费EA的unified12。修复应重新在合法行拟合派生state并换输入身份，旧cache保留为历史证据。

### F5. EA CLI漏检静态缺导，并把未测零槽标成已观测

位置：`prepare_euclidean_alignment_cache.py:145-147`只检查`trial_channel_mask`；`:157,166`又强制输出全True mask。合法`channel_mask=[1,1,0]`、`trial_channel_mask=None`没有被拒绝。

真实Huebner两导数据追加显式未测Oz零槽，经过合法保存、attested加载和实际EA CLI后，mask变为`[1,1,1]`，QC也把Oz写成observed=True，但该通道仍全零。反例由生产入口完成，没有mock掉缺导门禁。

已检查的10份cloud_inputs有效record没有这种静态缺导表示，尚不认定历史run触发。此前“EA CLI会拒绝全部缺导”的判断必须更正为“只覆盖动态mask”。证据：[ea_static_mask_probe.json](../assets/deep_chain_audit_20260929/input/ea_static_mask_probe.json)。

### F6. 普通验证loss的batch平均会反转模型选轮次序

位置：`src/baselines/deep.py:739-750`。每batch先做类权重CE归约，再乘batch长度平均，不能还原全验证集的类权重分母；`:2627-2639`据此选择best_state和patience。

同一6条验证数据，4负2正、pos_weight=2，两个固定候选的结果：

| 候选 | 正确全局加权CE / batch6 | 当前batch4归约 |
|---|---:|---:|
| A | .4500 | .5667 |
| B | .5000 | .3667 |

仅改变验证batch，赢家从A变B，score与label没有改变。OOM改变validation batch时也会改变监控量。

受影响：普通`DeepBaseline._fit_attempt → _validation_pass`等路径。**不受该反例直接影响：**unified源`_fit_fixed_updates`在`:1381-1394`用归一化row risk的可加sum；decision专用验证另累加numerator/denominator。修复只应替换错误归约，不能据此重标所有历史源模型。

证据：[probe_training_semantics.json](../assets/deep_chain_audit_20260929/training/probe_training_semantics.json)。

### F7. 候选主hit与分R hit在缺候选时用不同规则

位置：`baselines.evaluate.evaluate_candidate_selection:1443-1475`及`models.candidate_evidence`默认missing policy。
两个group，一个完整正确，一个只剩真实目标候选；真实summary reducer给`primary_hit_rate=1.0`，同一批预测的R1=.5，严格requested账本也是.5。总体分支让缺候选退出竞争，分R分支明确abstain。

生产调用来自`run_eeg_loso.py:680-691`；需要实际缺候选才触发，历史候选run是否触发尚缺原产物。不影响已核的binary AUC或BNCI统一membership评分核。

### F8. 开启中心化时，R3的score可以改变R2命中

位置：`baselines.evaluate.hit_correct_total_by_repetition:216-249`。先用全maxR的group均值中心化，再计算各R。候选在早期R计数不等时，全局中心不是共同抵消项。

仅将反例最后R3的score从0改100，R2所用原score保持不变，其hit却从1变0。默认center flag关闭，本轮未找到历史运行启用True的原声明；这是明确的条件入口缺陷。若保留该功能，应绑定每个prefix的可用范围，或严格将覆盖不完整的prefix记为incomplete，不能继续把该曲线描述成因果前缀证据。

### F9. valid比较会隐藏arm间不同的有效replicate集合

位置：`experiments/analyze_candidate_cross_decision.py:1137-1145,1178-1229`，默认metric为valid_subject。各arm先按可用replicate求均值，再按participant相减。

反例p1仅A/r1=.9与B/r2=.1有效，没有共同可评分replicate；p2两份真正配对差都是0。输出仍给p1差+.8、总体+.4。该值可以描述各自可用模型集合的均值差，却不能冒充同一冻结replicate集合上的对比，也没有充分暴露支持集合变化。

触发条件是多replicate且有效分母随arm变化；requested指标和当前单seed十二库主分析不直接触发此反例。修复应声明选择后的估计目标、支持矩阵及失败分母，不只给出“paired”标签。

F7–F9及较低优先级接口反例见[评分反例](../assets/deep_chain_audit_20260929/scoring/counterexamples.json)。

### F10. 跨运行导入保留旧预测，却生成新的执行证据身份

位置：`src/transfer/target_experiments.py:435-459,544-575,681-696`。显式compact导入只校部分source/cache/environment/recipe/split，未要求evidence_scope与evaluation_snapshot一致。

独立探针使用生产executor：先生成software_correctness、snapshot=C的K0结果；第二次禁止任何新prediction/fit，再导入旧结果。执行成功，cell与evaluation contract被标成development、snapshot=D；旧origin_contract只作为旁注保留。

这不是单纯相同预测正常复用的问题，而是消费入口接受旧计算、生成新执行声明。应保留原执行scope/snapshot并独立记录本次分析/消费身份，或用明确等价准入证明允许复用的变更。现有进程内`TargetResultReuse`键包含这些字段，不能自动弥补另一条导入路径。

本轮未证明09-18实际发生这种跨scope重标。证据：[probe_import_relabel.json](../assets/deep_chain_audit_20260929/training/probe_import_relabel.json)。

### F11. 其他独立接口缺口，不上升为当前矩阵根因

- `_validate_folds`的domain_lodo/domain_ceiling分支使用只在LOSO分支初始化的`test_counts`；合法两域折直接UnboundLocalError。当前统一迁移不走该入口。
- `stats.transfer`通用配对只比较行号摘要，不核不同input/label摘要；反例能接受不同输入的+.8差。本轮实际已检查矩阵没有配对标签/行错误，但producer以外的入口仍缺守卫。
- `candidate_evidence_scores`的trim/precision接受trial_weights却忽略零权重行的语义；零权重100分仍能改变赢家。未找到当前生产caller能同时传非平凡权重与这两模式，故为直接API缺口。

## 5. 已排除的怀疑与旧审计纠偏

1. **未发现主AUC的softmax轴/正类方向错误。** 主路径使用`logits[:,1]−logits[:,0]`或LDA decision_function；原预测复算一致。
2. **未发现匹配scratch偷用源参数，或重复baseline因不同source臂漂移。** 架构/seed反例与4526组真实重复预测分别支持这一点。
3. **未发现当前统一temporal BN把缺导零槽当实测。** 该主链显式pack实测槽；EA静态mask漏洞属于另一个入口，不能混同。
4. **非默认LMBC checkpoint参数丢失的旧疑点被推翻。** 实际architecture_record保存pool参数，loader恢复；非默认窗、offset和温度重建后前向逐位相同。不能只因通用dataclass未列pool字段就认定丢失。
5. **旧09-22 V1不能泛化为所有operational hit都高估。** within-subject runner已经另用完整requested分母重建operational字段；本轮真正确认的是F7那个不同入口。
6. **旧09-22 V5“Holm p没有产生路径”不成立。** 本地unified12独立分析脚本有50,000次sign-flip、+1修正及Holm；公共stats没调用某个函数不等于不存在分析来源。单seed不覆盖训练随机性的限制仍成立。
7. 当前masked source/target统计在真实缺导输入上数值一致；将未测槽灌入`1e12`没有影响合法统计。真实mask重排、拼接、subset与重载链没有复现标签/事件错位。
8. **11个既有个人包/重放实例没有显示主要漏状态或推理重拟合。** 6个LDA包的原raw→历史cache→原预测逐值一致，3个NN包重载与公共adapter输出逐值一致；另2个EOG/QC包availability一致，NN历史score最大差约`9.54e-7`，排序/AUC不变。时间/校准复用/参考/模拟/频带的30项攻击及6项摘要篡改被拒。真实raw推理`y=None`，没有拟合调用。详细证据见[个人模型审查](../assets/deep_chain_audit_20260929/personal/report.zh.md)。

## 6. 修复与重新比较的优先顺序

本轮没有修改生产算法。以下是可审查的实施顺序，不是已经执行的新配方：

1. **先固定比较合同。** 复用现有EvaluationRunContract/行身份模块，绑定raw recording、物理event、输入/标签/mask摘要、实际K、test cohort、预算单位、归约单位及真实执行snapshot。跨身份不直接相减；BACC明确阈值来源；requested/eligible/valid分开。
2. **先解决当前输入保真与确定性语义缺口。** 把强工频/参考污染和人工短raw端点联合纳入信号诊断；任何滤波/边界改造必须先通过无ERP带外干扰、真实同事件及已知ERP保真反例，不能偷接test上下文、直接删除高分/低分被试或按结果调阈值。随后修验证loss可加归约、minimum定义与回退账本、候选完整性/分母、valid支持集合、复用原执行身份及domain入口。保留每个本轮反例为真实入口回归证据。
3. **EA全链修复后重新准入。** 相对/单位一致正则化、静态与动态mask共同验证、物理EEG与变换特征分层、fit人员/行摘要贯穿所有派生state。新排除人员必须重拟合上游状态，不能只删行；也不通过补零或静默删源绕过十二库共同空间缺失。
4. **用少量严格匹配的桥接比较定位性能差额。** 冻结同raw、同person、同train/test物理事件、同K与seed，先只改60/90/200-update预算，再独立比较BN/dropout，最后比较合法raw处理合同。每次只改已声明轴；已有不可比绝对分数不作调参目标。该步骤尚未训练。
5. **恢复缺失的关键原件。** 旧0.74511原预测、09-18完整执行包、旧EA原manifest/cache内容SHA绑定仍需补齐；不能用文档文字补造原件。成人BrainSync确认另需独立后续记录。

## 7. 证据边界

当前可以说：大量旧/新成绩不一致有已证明的比较身份变化；若干隐藏接口缺口确实存在；现存09-16主矩阵的指标算术没有发现错误。当前不能说：已经找出A1/Q1负交互的唯一病因、所有旧数据都有错、修复后必然提高AUC，或已验证成人九选一效果。

任何修复均须保留旧产物和其真实执行身份；代码正确性、公共开发效能和新成人性能分别验收。本报告不消耗正式确认机会，也不将本轮局部反例或诊断图当作训练成绩。

## 8. 验证与可复现交付

已有`test_evaluate`、`test_candidate_evidence`、`test_euclidean_alignment`、`test_calibration_budgets`合计53项通过、0失败/跳过。它们通过并未覆盖本轮新反例，所以不把这53项当作“全链已正确”的结论。独立评分、真实入口反例、全61人处理、个人包重放与源身份检查分别保留脚本及数值账本。

所有新增审计脚本运行Ruff；文档链接、差异空白与本地大产物分发边界另行核验。新增缓存、逐cell表与roundtrip模型只留在Git忽略的`experiments/runs/deep_chain_audit_20260929/`，迁移前后逐文件SHA一致；原执行证据中旧绝对路径由迁移清单解释，没有改写旧记录冒充新执行路径。

主入口：[输入审查](../assets/deep_chain_audit_20260929/input/findings.md)、[历史可比性](../assets/deep_chain_audit_20260929/comparability/report.zh.md)、[评分审查](../assets/deep_chain_audit_20260929/scoring/report.md)、[全61人波形](../assets/deep_chain_audit_20260929/bi_quality/report.md)、[个人保存/推理](../assets/deep_chain_audit_20260929/personal/report.zh.md)。本轮生产算法未修复、未启动新训练；交付是可复核的问题与影响范围、排除证据和实施次序。
最终命令状态、410份生产/配置文件未变核验及交付哈希见[验证清单](../assets/deep_chain_audit_20260929/validation_manifest.json)。
