# Performance Roadmap

权威总纲为 `research_program.zh.md`。准确率是当前第一目标；算力只在影响部署时
成为约束。

| Stage | 当前状态 | Exit condition |
|---|---|---|
| 输入/事件硬合同 | causal steady-state v2/v4 与 4 个真实 cache 已完成独立审计 | 保持 SHA/CRC/ledger 门禁 |
| GTN zero-shot | v4 checkpoint + all-evidence candidate mean：K35/K65=0.714/0.683；仍远低于 0.90 | K35 临时默认，K65 保留对照；不再用 K33 扩展搜索 |
| GTN decision-aligned fine-tune | 2 kernel x 3 seed x 4 block、30 epoch 全源 EEG 已完成；K35/K65 learned=0.688/0.676，均未胜各自 no-fine mean | 当前联合微调 recipe 不晋升；默认保留 source-supervised checkpoint + fixed mean |
| GTN oracle proxy | 仅机制诊断 | 永不用于未知数字 90% |
| BI cross-decision calibration | 64 人、13 arms x 3 seeds 已完成；zero-shot hit@2=0.194，fine 无可靠增益，target stats 下降 | 保留 source classifier/stats；不把 BI 6x6 外推为 9-choice 90% |
| Signal recipe | 2x2 已完成；冻结 0.1 Hz/1200 ms，source QC=100 uV | 新 target block 前不再用同一 GTN cohort选 signal recipe |
| Source transfer | source full refit/checkpoint signature 已实现；full-unfold+K35 临时默认；强联合微调有负迁移 | 进入合法 cross-decision personalization，并把 full-fine 作为高风险对照 |
| Unlabelled target adaptation | 未开始 | pseudo/latent target 胜 zero-shot 且过反例 |
| BrainSync adult 9-choice | causal multi-session/multi-decision loader、target-switch split/runner 已实现；现有 4 sessions 均非 analysis-ready | 重新采集后 subject-macro hit@R >=0.90，失败进入分母 |
| Cross-dataset domain path | BI+BNCI 5 导 common-CAR 已实训；uniform joint=0.0967，显著低于 BI-only=0.1300；BI 3x/BNCI 1x=0.1239，只恢复未超越 | 固定 uniform rows/steps 隔离 BI-source normalization；仍负再做 gradient/domain-balanced 机制 |
| Confirmation/deployment | 未开始 | untouched cohort CI、覆盖、鲁棒、延迟、内存全部通过 |

当前没有确认冠军或部署冠军。GTN 是儿童 3 导、每人一次 selection 的开发集；
最终 90% 只能由成人 BrainSync 多目标、多独立 decision/session 裁决。
