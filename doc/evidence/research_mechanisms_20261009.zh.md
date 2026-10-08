# 科研机制：真实数据、保存态与独立反例

2026-10-09。范围是本地数学/软件验收、A01预声明机制对照、八人48格signal-prior归因及有限历史参考。完整source→target两面板、概率校准和成人确认各自验收。当前指导见六指南，本页只保存本轮已落地的独立结果和来源。

## 执行身份与授权

原BNCI2014-008 A01 MAT SHA `cdf9bca4f48c61ee9c9ba998382a4e84392918c9dbc5a43f45ee0052614e25bd`，八EEG、right earlobe、256Hz，分发processed MAT转V，未反演上游滤波、未改BrainSync通道名或参考。读/fit角色均由已有公共calibration合同明确给出，原MAT执行前后只读hash相同。

B06/B10及最初native AR0基线实际source archive为`5ddeedd1980aaa382e8245a86548b38878405a2a90de823a57aaeffa51c2d397`，manifest `ea00449cc97e6f3c16fa9ef357916a24dbb6c7482bf58c6acf48e10193bf2020`，base `a545a01`，478成员。B09六格空间fit实际archive为`00fa94523433343f04a04754918c3136ca52ccb9acae834bc8ada2e70e520439`，manifest `12ddeb44cfedc166bda5dcfceb0021e8fae6a8fc39382aae024be3cf972053ae`，base `ae6b835`，478成员。每次原执行源before/after逐字节相同。

B09复用原成功AR0基线，原model digest `c8d2f660650cb2b2a32ca97c2cf500be933a49b1feb4d430bbfaf885b277ae88`及原fit来源保持；新baseline fits0、空间posterior fits6。保存包和数值关系按其真实producer解释，不将旧fit改签修后source。

## B09固定noise下的signal prior

同最前两个完整selection C240标签，第三selection120flash/36候选作为later-held；request800/actual fit240/unused560分开，held真值只作外部评价。所有格共享native AR0、fixed32 prefix、同Gaussian307点/8basis、同Σ_C和temporal prior H=I、固定温度1。noise不随signal prior或reference重新拟合。

noise-tied prior为S=Σ_C；独立预声明prior为S=a·diag(Pz4,其余1)，a=8/tr(Σ_C⁻¹W)只用C确定，native expected whitened spatial prior energy均8。projection后energy另报，不用held匹配或选倍数。先冻结两个prior再score。

| 实际观测 | signal prior | hit/requested1 | NLL | Brier | projection prior energy |
|---|---|---:|---:|---:|---:|
| native8 | noise-tied | 0/1 | 2.633525 | 1.723018 | 8 |
| native8 | independent diagonal/Pz4 | 1/1 | 0.105872 | 0.020185 | 8 |
| Cz7 | noise-tied | 1/1 | 3.8440e-7 | 2.9549e-13 | 7 |
| Cz7 | independent diagonal/Pz4 | 1/1 | 3.5041e-8 | 2.4535e-15 | 7.782674 |
| CAR8/rank7 | noise-tied | 1/1 | 3.8440e-7 | 2.9549e-13 | 7 |
| CAR8/rank7 | independent diagonal/Pz4 | 1/1 | 3.5041e-8 | 2.4535e-15 | 7.782674 |

六格全部完成、coverage1、typed failure/tie/abstain0，全部模型保存重放一致。native tied错时confidence0.928175。独立prior同时改变cross-channel covariance、diagonal结构和Pz倍率，单例救回不能单独归因Pz4，也不能选择默认或推断跨人收益。极小NLL不证明概率已校准。

预声明等价门全部通过：native tied与原MN的posterior precision-unit mean误差2.7603e-13、cov误差2.4003e-12、candidate BF差5.8208e-10；两prior下Cz/CAR同商空间BF/logprob差各1.1642e-10。source/read support、actualC/noise与六包hash由独立接收核验。

实际矩阵暴露并关闭QR zero/tiny entry误拒。修后按每个输入列的QR生成误差核，并保单位方向precision、Q正交、rhs/mean/H关系。浮点Gram最坏enclosure过宽时，rare分支用保存binary64 R/H的精确整数RHRᵀ，仍按原resolution1.49e-8准入；真实128维误差1.1701e-11、单次0.1654秒，wrong-H4×仍拒。不改prior/noise/阈值或score。

CPU1、单worker/GPU0；actualPython body墙钟16.632s/CPU16.375s，含启动22.534s，峰working set414.05MiB，主机CPU采样峰12.2%。只说明这次机制panel成本。

## 有限整selection历史参考

`research.response_history_identification` 已落实完整有限 joint label 状态、一个共享 MN response 后验和 full/prefix 两个边缘积分的比。给定状态的时间协方差为 `K+D H Dᵀ`，保留当前与邻居设计的共享 β 交叉项；邻居全部 flash 共用该 selection 的一个 target 标签。prefix 更新邻状态权重，不能以独立 flash Bernoulli、额外独立 Γ 或旧 prior 权重的 conditional 平均替代。

当前 source SHA `47bdc38cae0ffce67fd433d31d30d3effe8ccab442186a080ee9278e750ef97c`，test SHA `aac7c76f6d2745daaee0a575f188b75bfc122658b21d4d9c1f2650d8c3316c79`，提交 `a6e403b`。promotion 的五项必要反例实际通过，包含任意 current 轴、past/current/future 相关 prior、独立多通道 Gaussian 与 flat-nuisance oracle，以及固定 current-prior tail 评分不等于 full-data Bayes current posterior的反证。原型14+1日志没有执行源码绑定，保持其原身份，不补签为本次五项或整个17项的新运行。

独立科学复审接受当前声明模型，未发现新增确认缺陷。直接 β 的40×40 Gauss-Hermite积分与实现 full/prefix/conditional 最大 log 差分别为 `8.88e-16/0/8.88e-16`；双通道、独立 proper nuisance prior 的宽尺度极限按 `O(τ⁻²)` 收敛。这是两个独立小案例，不是整个数值域认证；12件原始来源和promotion receipt已按当前hash核对，未重跑作者suite。审阅详细记录保存在本轮 `REVIEWS/B07_HISTORY_SCIENCE.zh.md` 及相邻binding。

无 nuisance 时使用 proper Gaussian density；有 nuisance 时使用 full/prefix 相同坐标体积的共同 sigma-finite measure。prefix 识别共同 nuisance 后，比值定义 normalized conditional-tail density；单个full marginal仍不称proper密度，prefix缺列秩明确拒绝。future 状态可以因冻结 joint prior 与已观察 past 的相关性更新权重，未读取 future EEG。它是每状态 dense 小矩阵研究参考，完整历史、真实 source/channel/reference/time 与数据权限由外部 caller 证明；实际模型 adapter、保存重放、长历史扩展、latency、随机效应及真实效能另验。生产未知 overlap 拒绝保持。

## 实际模型到历史参考的接线

`research.continuous_history_adapter` 已由独立数学/调用链审阅后迁入长期research owner，source SHA `406aff8360813fa5967c788f97ab6ef9b6b523f1b5f6e854040696e23f91b14c` 保持原审字节。实际model integrity、原basis/M/R/Σ与stored noise runtime直接消费；以 `noise.whiten(I)` 的inverse取得同joint时间covariance，full/prefix保共同Z，不再乘innovation variance诊断值。rectangular启动和未发生actual public clocks明确拒绝；source/reference/历史完整性与权限仍由caller举证，生产未知overlap门未放宽。

原保存软件包及真实A01状态抽取的10项、无邻居原conditional-score parity1项各保原test版本；可分发module fixture改用canonical C、一次toy fit/save，必要迁移3项通过。它们有覆盖重叠，不合计成14独立反例或最终10项全集新通过。迁入测试只改package import及排序，SHA `7de69eca74ffd781d8edeb5efc50c6b5b7eb2e6b2bec4d644be9d3d170f717a0`；root编译与仓库外真实import通过，未重复三项绿色。独立结论见本轮 `REVIEWS/B07_ADAPTER_SCIENCE.zh.md` 与binding。它仍是dense小矩阵研究接口，没有新的可部署组合包、长期history吞吐、完整权限caller或新效能。

## 八人48格signal-prior归因

后续固定八人A01–A08 × native8/Cz7 × noise-tied Σ/energy-matched diagonal I/Pz4，共48格，CAR同商重复省去。实际source archive `f6101b585e6871595180a6147150833780a37054cf249210e8e357c42ddadd4a`，manifest `6e08c3fbab731351eede0097e852a78c5705c257983ed0d4a4c919f9f396c2df`，479成员、base `4ac444c`；新source before/after与八份原MAT均相同。48/48完成并独立验收，typed failure/reference-unavailable0，全包保存重放一致。

A01原四格按其source00fa及成功baseline5dde复用、不重评分；A01 I两格和其余七人42格新空间fit44，新baseline fit7。每人同C240/later-held120/request800，prior文件C-only且在held score前冻结。八人历史公共开发、A01已见；另列七人新variant读数，不称盲确认。

| 实际观测 | signal prior | hit/requested8 | 平均NLL | Brier |
|---|---|---:|---:|---:|
| native8 | Σ noise-tied | 6/8 | 2.594126 | 0.475504 |
| native8 | matched diagonal I | 7/8 | 0.922528 | 0.249504 |
| native8 | matched Pz4 | 7/8 | 1.128338 | 0.202325 |
| Cz7 | Σ noise-tied | 7/8 | 5.112400 | 0.250000 |
| Cz7 | matched diagonal I | 7/8 | 3.707681 | 0.250000 |
| Cz7 | matched Pz4 | 7/8 | 3.795221 | 0.250000 |

全coverage1。除去seen A01，三native prior命中均6/7：Σ→I救回A03、损伤A02，净命中收益0。七人native NLL为2.588497/1.046889/1.274405。A01 I自身已救回、NLL0.0520优于Pz4的0.1059，不能宣称Pz独有收益。Pz4相对I没有额外hit，native/Cz NLL分别增加0.205811/0.087540，均未过预声明hit/NLL联合开发门，不晋升默认。Cz的A03三prior均错误且置信度约1，NLL40.899/29.661/30.362；更高点命中不能证明reference或概率更可靠。

I与Pz4匹配原native总白化prior energy8。令π=(Σ⁻¹)Pz,Pz/tr(Σ⁻¹)，Pz方差相对I变4/(1+3π)，其它导变1/(1+3π)，是相对收缩重分配。A01实际倍率为2.33018/0.582544，不是独立放大Pz4倍。Cz projected energy仅报告，不另匹配或混称同能量。Σ→I改变整个noise-link/形状/正则分配，不能称纯decorrelation。

CPU1、GPU0，actual执行约126.25秒、主机CPU采样峰14.9%、峰working set约423.57MiB。该点估计依各人一个later decision和固定C-estimated noise；noise/超参数不确定性未积分，不能代替跨日、完整公共矩阵或成人确认。

权威结果：`B09/SPATIAL_ATTRIBUTION_RESULTS/matrix_receipt.json`、`B09/spatial_attribution_acceptance.json`和逐人48行`B09/spatial_attribution_participant_cells.csv`，均位于同轮本地研究目录。

## B06完整native T与B10实际自适应程序

B06 context[1800,1992)、short[1832,1960)、output[1864,1928)，实际额外研究SOS，不重构未知上游带通。短观测严格无法决定全部long-teacher输出；外部1831/Fz加1µV保持short字节相同，目标增量norm1.79236e-16V、有限差分舍入1.895e-20V。因此正确拒“精确恢复”，但不支持可测ERP损伤主张。K=b=0锚点的完整G与short baseline相同；relative white gain1不等于raw unit gain，raw平均variance gain约2.963、worst约15.863。

B10在同授权C内processed A01[1800,2056)跑真实完整harmonic程序。三独立预声明挑战：Pz主导320ms/80ms Gaussian2µV、50Hz八导line2µV、Fz单点glitch5µV；另4个seed20260918白噪声1µV draws。所有case保留，未取held标签或从结果选门。

ERP增量relative差1.66e-15、glitch1.69e-16，均在base和对应注入分支未启用处理时；glitch未被删除。line注入触发8导开关，完整refit增量相对injected line的能量0.012612、与line reference之差1.0205；包含对base拟合变化，不能直接叫纯50Hz剩能量或ERP损伤。四draw输出/输入二阶矩比0.999241，经验covariance rank≤3，非总体噪声增益或Gaussian likelihood。

B10小矩阵反例另外证实：isolated ERP清理误差0仍可在污染base上因开关造成完整增量误差2.085。应测T(x+h)−T(x)，不能只测T(h)或冻结一次矩阵。真实A01输入、单位、事件授权及保存经验矩均经独立只读审阅，未重复拟合。

## 科学代码与后续边界

B07五数组/物理表达式准入，B08逐functional quotient/completion与生产QR后验，TemporalNoise/5保存编译态和实际scale，B05/B09真实vec次序与private非分离noise均有独立反例及实际调用链。B11noise-only Gamma与独立proper Gaussian是数值参考，中等Gamma归一化消去已由70digit闭式反例修复、10项必要case通过；QUAD/解析tail/Stirling预算仍非all-floating证书，不把所有不确定性一起套t分布。

后续需预声明跨participant配对、signal prior形状归因、proper概率校准、joint AR/Σ或learned graph估计器与工作站授权接线。四source/单seed公共两面板、同版本完整crosshost及云25CPU/4GPU实效仍另验；本结果不晋升Pz/参考/清理默认。

本地权威入口：`tmp/research_line_20261008/B09/A01_NATIVE_VERIFIED/panel_receipt.json`、`B09/native_result_acceptance.json`、`B06/A01_NATIVE_RESULTS/receipt.json`、`B10/A01_NATIVE_RESULTS/audit_run/result.json`；科学结论、精确命令和独立审阅保其producer，当前过程副本按阶段落地后退役。
