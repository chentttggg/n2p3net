# 指标分母与字段审阅：独立概率复算

审阅范围主树 `b26c099`。脚本使用必填 `--root`、`--output`；放进其他artifact目录后仍消费明确指定的原仓库。只读保存JSON/CSV、binding和源码文本，本次fit、score、model load、EEG read、filter各0。

## 复算与核对

逐格读取原36维 `log_probability`，从原执行配置的DecisionPlan取得truth并核原vocabulary/complete roster。NLL直接取 `−log_p_true`；Brier由 `exp(log_p)`计算；winner/tie按当前generic源码中原log-space容差独立计算。没有使用旧row的nll/brier/correct作为计算输入，也没有归一化或clip原概率。行/列边缘质量与NLL链式分解同样复核。

shared8：5/8，NLL 7.3364765833，Brier 0.6790937536；private8：6/8，NLL 5.5540121911，Brier 0.4776785010。覆盖均100%。八人macro Δhit 0.125，ΔNLL -1.7824643922，ΔBrier -0.2014152525；救回1人、损害0人。八行ΔNLL和为 -14.2597151377，不可替代宏平均。A01贡献约 71.8229%。

独立结果与原row、paired、summary、三份CSV、accepted binding及保存generic metric owner逐项核对，共2427项比较；最大数值绝对差 8.88e-16。消费输入SHA前后不变；重复subject-arm、missing roster、P1 custom fields直接导入generic v1均拒绝。

## 调用数与分母

本次审阅0调用。原完成receipt合计14次score；A01两格复用72候选，其余14格评分504候选。原续评新增response/noise fit=0，原anchor/noise已有拟合，其历史拟合数本次没有重数。不能把本次0调用写成原训练/评分0调用。

P1每臂8人各1个C-held decision、36候选、全真值row5/col0；macro=micro只在此面板成立。public BNCI保持152/144，BI保持61人1287/1243且宏微coverage分列，BNCI C600/R10每臂240/240。后三项本次只核接受binding/文档，没有声称重算其全部概率向量。

P1是固定C-held诊断，非公共/成人效能；两臂NLL均高于ln36。`all_input_certificate=false`不是评分失败，row/column边缘概率不能替换joint p_true。每格producer/noise文件/consumer与score/receipt/plan SHA均进入账本。

B07 history adapter已在 `948c485` 接central；中央certificate/capsule独立复签进行中。本报告不写consumer未接，不声明新B07训练结果。

产物：metric_crosswalk.json、p1_error_ledger.json/.csv、verification.json。
