# 科研机制：真实A01、保存态与独立反例

2026-10-09。范围是本地数学/软件验收与A01预声明机制对照；不是全公共矩阵、概率校准或成人确认。当前指导见六指南，本页只保存本轮已落地的独立结果和来源。

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

## B06完整native T与B10实际自适应程序

B06 context[1800,1992)、short[1832,1960)、output[1864,1928)，实际额外研究SOS，不重构未知上游带通。短观测严格无法决定全部long-teacher输出；外部1831/Fz加1µV保持short字节相同，目标增量norm1.79236e-16V、有限差分舍入1.895e-20V。因此正确拒“精确恢复”，但不支持可测ERP损伤主张。K=b=0锚点的完整G与short baseline相同；relative white gain1不等于raw unit gain，raw平均variance gain约2.963、worst约15.863。

B10在同授权C内processed A01[1800,2056)跑真实完整harmonic程序。三独立预声明挑战：Pz主导320ms/80ms Gaussian2µV、50Hz八导line2µV、Fz单点glitch5µV；另4个seed20260918白噪声1µV draws。所有case保留，未取held标签或从结果选门。

ERP增量relative差1.66e-15、glitch1.69e-16，均在base和对应注入分支未启用处理时；glitch未被删除。line注入触发8导开关，完整refit增量相对injected line的能量0.012612、与line reference之差1.0205；包含对base拟合变化，不能直接叫纯50Hz剩能量或ERP损伤。四draw输出/输入二阶矩比0.999241，经验covariance rank≤3，非总体噪声增益或Gaussian likelihood。

B10小矩阵反例另外证实：isolated ERP清理误差0仍可在污染base上因开关造成完整增量误差2.085。应测T(x+h)−T(x)，不能只测T(h)或冻结一次矩阵。真实A01输入、单位、事件授权及保存经验矩均经独立只读审阅，未重复拟合。

## 科学代码与后续边界

B07五数组/物理表达式准入，B08逐functional quotient/completion与生产QR后验，TemporalNoise/5保存编译态和实际scale，B05/B09真实vec次序与private非分离noise均有独立反例及实际调用链。B11noise-only Gamma与独立proper Gaussian是数值参考，其tight accuracy仍按独立审查收口，不把所有不确定性一起套t分布。

后续需预声明跨participant配对、signal prior形状归因、proper概率校准、joint AR/Σ或learned graph估计器与工作站授权接线。四source/单seed公共两面板、同版本完整crosshost及云25CPU/4GPU实效仍另验；本结果不晋升Pz/参考/清理默认。

本地权威入口：`tmp/research_line_20261008/B09/A01_NATIVE_VERIFIED/panel_receipt.json`、`B09/native_result_acceptance.json`、`B06/A01_NATIVE_RESULTS/receipt.json`、`B10/A01_NATIVE_RESULTS/audit_run/result.json`；科学结论、精确命令和独立审阅保其producer，当前过程副本按阶段落地后退役。
