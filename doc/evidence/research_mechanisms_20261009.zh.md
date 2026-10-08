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

## 噪声尺度的可识别投影参考

`research.noise_scale_identification` source SHA `8fcd27e0899457414a45d28a90906b5b0cb91690623e941ad8bbde3172d7549f` 已经独立数学与迁移接口审查后落地。给定fixed K0和完整ERP/state/nuisance span，以白化正交补消除共享β，包括其真正落在该span的Gaussian不确定性；只估投影里的相对τ²/明示ν。投影看不到blind noise方向，不能称freeΣ、未知state或full joint MLE。

初版静默rank截断曾将近共线满rank的纯ERP解释为noise；actual binary64精确rank与白化resolved rank不符时现明确拒绝，未隐加β幅度界。17个迁移相关unique case通过、6个仅收集，另原18组研究/高精度及独立直接Gamma/Beta积分保原源码，不签23全绿。正式test迁入只调整import排序，SHA `25ea1ff82370768118d45b35caa5e1367ceeb56d5a18c0f730c1860677302b41`，无新增运行依赖。

finiteν正尺度interior MLE要求 `νJ_positive>R_zero`，mixed zero也可能只有边界sup。单个decision自由τ²时ν profile严格增加、sup在Gaussian∞；这不否定t族的population结构可识别。模型内 `I_logτ²=rν/[2(ν+r+2)]` 随同decision维数r饱和到ν/2，时间/通道数不替代独立λ draw数。positive Fisher或optimizer收敛不能签真实ν稳定。ν≤2不报告finite covariance；explicit∞控制独立拟尺度，C folds按完整decision划分。

读取区间显式声明，不从flatten长度猜native时间；caller仍证明权限/physical map、fixed producer和真实noise独立。C估计K0是plugin，whole-program OOF须在inner train重拟它或明确conditional比较。更换noise law后，旧Gaussian C的ERP后验不能重签为新t exact posterior，完整C共享β/各λ联合更新正在下一研究切片。两个独立审阅见本轮 `REVIEWS/B11_IDENTIFICATION_SCIENCE.zh.md` 和 `B11_LASTING_OWNER.zh.md`；当前是moderate finiteν、小矩阵reference，无默认ν/生产接入/真实效能主张。

## 原M1完整状态的只读归因

原八人×六个C-inner fold×六阶×两估计器的576模型在另一执行工作树可达，全部文件与complete M1原档SHA `edcb65ae2405b3af67820b76b0708f170e0bd4eb31b1148c78ca9094b58e7539` 核对，143原包复用/433新fit身份保持。原continuous source `e982d26d…`、TemporalNoise `/4` source `9fc7d122…`，没有用当前 `/5` 构造/load旧模型、重评分或重fit。每臂48个相关C-fold状态来自8人，不称48独立participant。

| 原臂 | 原hit | 原NLL | increment norm / AR8 Burg，中位 | full target norm / AR8 Burg，中位 | given-noise kernel df，中位 |
|---|---:|---:|---:|---:|---:|
| AR8 Burg | .6667 | 6.1514 | 1 | 1 | 7.0937 |
| AR32 Burg | .6250 | 7.3081 | 1.0325 | 1.0140 | 7.3635 |
| AR8 profile | .3333 | 3.4380 | .009501 | .010745 | 5.7046 |
| AR16 profile | 0 | 3.6979 | .000800 | .000577 | 7.9592 |
| AR32 profile | 0 | 3.5781 | .000307 | .000755 | 8.1125 |

target increment为 `ΦM_delta`，完整target为 `Φ(M_common+M_delta)`，posterior保common/increment交叉块。AR32 profile相对同阶Burg的increment norm中位 `.00029829`，校准目标每tail行降低中位36.8596，但原效能联合门所有11替代臂均未过；不晋升默认，也不把拟合均值变小说成真实生理ERP消失。

固定AR形状/λ/C时，MN方程 `H=(G+λI)⁻¹, M=H S` 中Σ消去；Σ只变不会独自令M/H缩小，但会改变physical prior/posterior与概率尺度。实际profile一起改变AR的G/S、均值、Σ及noise-tied prior，是完整耦合程序。AR16/32的η没有普遍触8边界、df没有降到0；不能单因解释成过强白化或设计无信息。

C状态重建的最高8个信息方向与endpoint value/first-difference rowspace overlap，中位AR32 profile `.999946`、AR32 Burg `.719476`。它来自固定Gaussian坐标的已有H，与原raw C独立Gram复算不同；支持研究finite support/端点交互，尚不证明hard endpoint改善held。下一切片固定完整physical response/noise做可归因控制，再用一般vec C fit保持同一physical prior，最后明确不同函数空间与实际df。旧held已见，不据此选择新阶、prior或默认。

作者完整结果及绑定在本轮 `B05_ATTRIBUTION/ATTRIBUTION.zh.md`、`delivery.json`、全部576state与12臂summary。独立审阅已核全部model/ensemble身份、288个同阶initializer/目标关系、12臂原概率宏指标和11个原开发门；另从30份原NPZ独算波形/cross covariance/df/η与endpoint overlap，294项summary独立归约一致。未发现来源或公式错贴，endpoint overlap独算最大差 `3.37e-10`。其结论保本轮 `REVIEWS/B05_M1_ATTRIBUTION.zh.md` 与binding，不将状态归因签成新fitter或独有因果证明。

## 有限support、端点与noise precision的交互

端点切片完成六组明确生成程序的CPU1 oracle，数学/原包/文献经独立复审接受。已知可逆F同时传播Y/D/Z/noise可保Gaussian后验和信息；正确完整传播的synthetic response相对偏差 `.000199`，把filtered response截回raw长度后偏差 `.862746`，错误强制observed raw-T为零的偏差 `.747407`。滤波后合法tail占norm `.405369`，原raw硬零不授予filtered endpoint硬零。

在完全正确noise的另一合法非零边界例，hard约束造成两requested endpoint的coverage为0；真值位于hard空间时可降低风险，结构点质量coverage为1。它们不支持自动晋升hard。current hard_zero只限制同ambient cosine，在analytic `Tactual=L/fs`取E，不能用lastsample代替；cosine hard产生C1零延拓，curvature接缝仍可被更高阶差分强调。受限prior的precision/非零mean须正确pullback，dimension、physical energy和实际df分别记录。

完整设计未证明时，noise error contrast也会泄漏signal：正确完整设计的期望相对尺度为1，同维数错误截断设计变为 `4.590795`，不能把更多contrast称纯noise。原六份M1 noise/basis只在新的scalar/synthetic clock中诊断，Burg亦强读边界，不把端点集中归为profile专属病因。actual source与全部37file绑定见本轮 `B07_ENDPOINT_INTERACTION/delivery.json`；独立八sample反例与结论见 `REVIEWS/B07_ENDPOINT_INTERACTION.zh.md`。

三篇正文支撑检查滤波/完整response支持；Harris只核书目，失败PDF返回HTML不当正文。上述结果没有恢复未知processed MAT filter，也不是新公共效能。下一真实C对照须声明实际operator/support、同physical prior与函数空间，保持现B05同owner。

## 给定noise、固定physical prior的C拟合

`6128c69` 在原 `FittedChannelNoise` 加入 `fit_response_given_frozen_noise`，source SHA `307b3db6e738a7be680936a2fae14eca7be4e9a7b61d56acbe0b6e81b5df1177`、专属test SHA `b286e2065c9027cebbf4df14209a2ce2a21cb717b86aad033e8e46bd9a8bc701`。固定完整physical coefficient prior，noise arm更换不重标Σ；同actual whitening作用于Y/D/Z后流式QR，只加入一次prior。输出保每record information roots/responses、最终augmented root、完整response/drift及跨record covariance，由原FrozenERPResponse/score消费，无第二个生产fitter。

独立dense joint oracle支持shared MN极限、private非separable后验、Σ-only的两种prior区别、单位/非正交basis pullback和prefix/full共同测度评分。原型分阶段8/3/3/1项有重叠，existing-owner接线必要3项实际通过；不累计成15独立case或宣称最终11项全套新通过。root所合source/test逐字节等于该3项实际candidate，Ruff/编译/diff通过，数学与调用链独立审阅接受。详细分阶段来源在本轮 `B05_FIXED_PRIOR/delivery.binding.json`，独立结论见 `REVIEWS/B05_FIXED_PRIOR_SCIENCE.zh.md`。

它是在给定frozen noise/plugin参数下读全授权C的Gaussian后验，不估AR/Σ或积分hyperparameters。实际C labels/design/history、producer、绝对时间与读权限仍由外部caller证明。数组保存重放说明软件状态可恢复，尚不是正式持久包篡改/producer认证；真实归因矩阵与joint MLE另验。

## 原M1合法C导出与实际共用QR入口

原A01/fold0/AR8 Burg的五个C record已从原477-member源码恢复，600labels/100targets、absolute read/label/prefix、basis307×8、8EEG/right-earlobe/256Hz/V与原model逐项吻合。observed tail为308samples，不能以今天默认1.2秒猜其7975行。旧 `/4` 包不保存startup数组，原加载器重建后严格核其原数值哈希；导出明确是本次原程序数值重放，没有用当前 `/5` 改签旧包。

独立审阅确认两条实际关系漏门并闭合：协调签新export receipt仍能把label/prefix区间缩错，或把真实A02 MAT错贴给A01 C；现区间直接来自actual onsets/readstart+32，source字段来自原model真实training views。最终必要6项与正常五C/600label通过，原source、model、MAT与export未变。纯旧lattice源和原startup state逐bit核FY/FD/FZ，prewhite后空间Σ只消费一次；两个来源都进入同一private QR core，`2206ad5` 的source SHA `06a5c16731a03bbd316491d773ba109693d39e1e8c6e8fba2a589cd59aa616e3`，无第二fitter。

实际matched-MN入口在固定mean0/λ1/Σ_AR8 physicalprior上完成一个逻辑格。为了纠正venv launcher漏记子进程资源，执行两次physical fit，所有保存numeric arrays逐bit相同；不计两个seed。原posterior单位均值/完整cov最大差 `2.26e-12/1.08e-12`，common-increment与完整target cross covariance相对差 `1.85e-13/2.93e-14`，actual R/RHS关系亦通过。第二次fit 2.912秒、真worker约一核、Windows lifetime peakworking set约618.5MiB、host采样峰26.8%；第一轮launcher-only 4MiB不能当真实资源。

完整下一对照已固定48 Cunits×12原冻结noise×whole/shape-only/Σ-only=1728逻辑格，完全等价数值程序可复用、实际fit与requested另账；每unit用原预声明AR8Burg physicalprior。它是C-only参数机制研究，不重估噪声、读取旧held选方案或声称新效能。machine plan/当前执行源码尚在准备；原数据/model/producer和seed20261005保持，不能称一个新完整source→target seed。详细身份及实际onefold入口在本轮 `B05_M1_C_ADAPTER/`，最终只读clock/source复审见 `REVIEWS/B05_COMPILED_C_LIKELIHOOD.zh.md`。

## 全C共享response与noise尺度联合推断

`c94c0a1` 落实 `research.joint_response_scale`，physical β prior与noise scales独立；每完整decision一个λ，C与测试共享一个β。`m(C+full)/m(C+prefix)` 中C各出现一次，prefix保留对β、校准λ和测试λ的联合更新，不把旧Gaussian-C后验或moment-match Gaussian改签新模型。canonical source采用LF后SHA `7671997aefc4cdf046e546a301c97bb7a4d70079fdad2a51ca83132cf34ad273`，与作者309a proposal的AST相同；formal tests只调import排序/LF。

独立审查发现原dimension-tilted积分规则的有限ratio不正规：同一D0例96节点的某点误差约 `1e-15`，实际tail mass却为 `.999808249`。现只供应一个预先固定、正且归一的离散Γ近似measure，full/prefix消费同actualrule；独立原反例新mass误差约 `1e-15`，真正共享β/两C的finite dense oracle也通过。连续Γ误差另外按node/range核，prior排除概率不作posterior证书；低order的实际Eλ明显偏离1、有限极端尾为Gaussian，不冒称原t尾。旧614d source及10组记录保原字节，原反向t组件mass1不能追签旧finite程序。新六组与必要迁移分阶段执行，未重复旧套。

`c3268ec` 的 `research.joint_response_sampling` 共用上述canonical输入类，source SHA `2dbfe3d963b5d7da622d14f5c472cd01f93589816f044c480c296583c967a5a3`。Gibbs只更新一个global β和全部decision λ；一次准备C统计及测试白化。C样本的prefix权重先固定后平均proper条件组件，或由caller明示另跑C+prefix采样，再平均同条件组件；不用两批独立随机full/prefix分母，也无静默fallback。

原合成128C×8行、16β/129λ的两次CPU1运行各约4.7秒，predictive log差 `.000586`、combined MCSE估计 `.005119`；仅为该合成规模成本/重复性，没有高维exact oracle。实际多模态反例永久保留：overdispersed链Rhat约1.73/ESS6会拒；全从同mode起步却Rhat1.001/ESS2488，βmean3.981而exact symmetry mean0。诊断通过、MCSE和有限混合正规都不认证真实后验准确性。16个formal migration case分14+2通过，不重原10组/128C；新增prediction cache最高seam证明重复评分不重白化。

理论、原始论文、actual stages和两个独立结论分别在本轮 `B11_FULL_C/`、`B11_FULL_C_SCALING/` 与对应 `REVIEWS/`。当前未接实际K0/C权限、flat nuisance或产品包；实际EEG行数远大于8，dense noise root及全draw residual矩阵的内存问题已进入下一压缩QR/分批预测切片，不据此声称真实训练吞吐。

## 本轮统一软件入口核验

ERP入口的已删除私有digest调用已修正，原compact dtype/shape/bytes关系保持，真实仓库外CLI help通过。459个Python文件的4665个仓库内显式from-import静态检查未见缺符号；随后实际 `pytest --collect-only tests` exit0，收集217个测试模块/3179用例、源码前后不变。收集没有执行测试断言，完整integrated suite仍待实际足额内存；不降低原native资源声明或重启用户关闭的云。

长期owner落实后删除六份已确认等价的prototype/proposal源码与测试副本及首批bytecode，共80,582文本字节；原source producer、执行receipt、数据/cache、模型和最终结果保留。他人或下一批科研WIP未改。验证身份见本轮 `VALIDATION/research_ingress/`。

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
