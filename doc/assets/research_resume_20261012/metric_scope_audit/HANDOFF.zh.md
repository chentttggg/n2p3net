# 指标分母与字段审阅：修后独立复算交接

本交付只更新原临时目录 `tmp/METRIC_SCOPE_AUDIT_20261012/`。root已复制的版本和 `doc/evidence/` 不由本次修改。

## 实际修复与程序身份

原脚本使用 `Path(__file__).parents[2]` 推断root，拷到doc/assets后会失效；并且从旧row的nll/brier/correct和旧paired字段求mean，未独立恢复概率计算。本轮明确修正这两点。

新脚本 `build_metric_audit.py` 的 `--root`、`--output` 均为必填，目录必须存在。脚本位置仅用于自身SHA和fixed diff；全部输入路径来自显式root。执行使用真实原科研树，不依赖工作目录或脚本所在层级。

最终脚本SHA：`66700e23b49c632e8a0ad369fa82e8dadc431f4ded80b34a995a0d4944845489`。原未闭合脚本保持在 `build_metric_audit.previous.py`，不追签为独立复算通过。固定diff在 `build_metric_audit.diff`。

## 完整输入与独立数学计算

共消费46份文件：原P1接受binding、analysis.json、execution_config.json、result.json、三份原CSV、16份score.json、16份receipt.json、candidate_evidence/candidate_metrics/continuous_history_adapter源码文本、cloud/BI/BNCI接受binding、public_evaluation文档。verification.json保存全部SHA-before/after，逐项完全相等。

每格直接从原score.json读取36维log_probability及candidate_ids；从execution_config内原producer_plan/DecisionPlan取真值、participant、decision和requested roster，核原held vocabulary、read support、完整physical counts和原receipt/score SHA关系。旧analysis rows/paired/summary只能作为核对对象，不参与计算NLL/Brier/correct。

计算：NLL=-log_probability[truth]；Brier=sum(exp(logp)-onehot)^2；按实际CandidateEvidence源码的log-space rtol/atol计算winner和tie，不重新归一化或clip。真值排名、winner confidence、entropy、row/column概率质量、稳定row logsumexp和NLL链式分解均从同一joint36向量重算。Q/logdet/error-budget诊断从原保存的candidate_diagnostics/log_density重组；这不是模型重新评分。

先计算16格，再按原同subject/truth/support配成8组shared/private，独立求Δhit/ΔNLL/ΔBrier和rescued/harmed，最后按八个原参与者等权平均。对原analysis row、paired、summary、三份CSV、accepted summary/paired、原result generic metric owner逐项核对，共2427项字段/数值比较，最大普通数值绝对差8.881784197001252e-16，最大概率差0。

重复subject-arm和missing roster由完整原请求集合拒绝。16行P1 custom schema/fields直接导入generic evidence v1均拒绝；另用generic字段形状正对照确认拒绝逻辑不是无条件抛错。正对照只验证schema/字段shape，不代表概率或科学authority准入，也没有调用真实生产constructor。

## 复算读数

shared8：5/8，NLL7.336476583291455，Brier0.6790937535735931。
private8：6/8，NLL5.554012191080322，Brier0.47767850103985904。

八人macro Δhit=0.125，ΔNLL=-1.7824643922111332，ΔBrier=-0.20141525253373407，救回1人、损害0人。8行ΔNLL和=-14.259715137689065，不能当宏平均；A01占行和改善71.82293423790772%。两臂概率/预测覆盖均100%；全真值row5/col0、每人1个C-held R10 decision，仍为固定C内诊断。

## 原调用与本次调用

本次新fit/score/model_load/EEG_read/filter分别为0。脚本只导入Python标准库，不导入任何scientific repository module，不读NPZ或EEG，不反序列化模型。

原完整rescoring receipt中的score_calls合计14；A01两格为0，复用72个旧候选，其余14格各评分1次、共504候选。原数值续评新增response/noise fit=0，但原anchor/noise已有拟合；其历史fit次数本次不重新计数。新账本使用 `saved_rescoring_score_calls`、`saved_rescoring_new_fit_calls` 和 `this_review_new_calls` 分别记账，避免把本次0调用重解释成原历史未训练/未评分。

## 字段、公共分母与最新B07范围

v2 crosswalk/ledger保原score/readout schema、每格original score/receipt/plan/readout路径和SHA。field_contracts逐组声明来源、公式和SHA引用；P1 custom fields明确不能直接写入generic v1。all_input_certificate=false不是评分失败，row/column边缘概率不替代joint p_true。

P1每臂8/8，public BNCI152 requested/144 probability，BI61人1287/1243和宏微coverage分列，BNCI C600/R10每臂240/240。后三项本次只检视接受binding和文档，JSON明确标注没有独立重算其全部保存概率；不存在声称本次复核了152/1287/240所有概率的升级。

B07状态纠正为：history adapter在 `948c485850c068083bc1240d9475b3fd382f8869` 已接中央primitive，脚本读取当前actual callsite核对；central certificate与capsule独立复签进行中。本交付不写“consumer未接”，也不声明新的B07公共训练结果。

审阅范围head保持b26c099；真实执行HEAD在verification和crosswalk中另记为268973007ba6a02720e630a095e54df4a416d20b，避免用文档head重签源输入。

## 命令、检查与交付

```text
.venv/Scripts/python.exe tmp/METRIC_SCOPE_AUDIT_20261012/build_metric_audit.py --root C:/Users/chent/.codex/worktrees/research-integration/n2p3-net --output C:/Users/chent/.codex/worktrees/research-integration/n2p3-net/tmp/METRIC_SCOPE_AUDIT_20261012
```

最终真实复算exit0；Ruff检查通过；py_compile检查通过；git diff --check通过。无full suite。新输出保持同一个已有目录：metric_crosswalk.json、p1_error_ledger.json/.csv、verification.json、REPORT.zh.md。delivery.binding.json汇总最终源码/产物SHA；root可据fixed diff和SHA审阅后替换其复制版本。
