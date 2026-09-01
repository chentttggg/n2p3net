# N2P3-Net 数据与冻结层级

日期：2026-09-01

## 活动层级

1. `mne_data/`：官方或原始来源数据。不得因模型或合同升级而改写。
2. `experiments/cache/` 顶层：只放当前 `n2p3net_epoch_dataset/5`、统一 causal-v3
   派生数据。当前尚未重建正式 BI/BNCI/BrainSync promotion cache。
3. `experiments/cache/legacy/pre_v3/`：保留的旧派生数据，共 19 个文件、
   2,951,809,224 bytes。活动 loader 不支持其 schema；该目录只供历史核对。
4. `experiments/runs/` 顶层：只接收当前合同生成的新运行记录；当前没有 promotion run。
5. `experiments/runs/legacy/pre_v3/`：保留的旧运行数据，共 5 个目录、17 个文件、
   201,916 bytes。其中 `audit_baseline_8sub` 绑定旧 `/4` cache record，
   `bi2014a_128_contract_head_ablation_20260828_r1` 只有旧 cloud completion/log；其余
   3 个空目录仅表明曾创建运行位置，不构成完成证据。
6. `frozen/*.tar.gz`：旧源码、runner、tests、manifests 和 evidence 的物理隔离边界。
   活动 Python 路径不引用归档成员。
7. 新实验 evidence：必须绑定 source archive SHA、cache SHA、identity/lineage digest、
   TrainingRunContract、EvaluationRunContract、checkpoint registry 和完整 DecisionOutcome。

## 原始制品到 cache

公开数据 cache 不能只记录 dataset class 或下载目录。每套原始数据必须先通过
`n2p3_raw_artifact_manifest/1`，逐文件绑定官方来源、大小、MD5（若官方提供）、本地与
远端 SHA-256，以及实际 loader mapping。cache builder 将认证文件从单一打开的 descriptor
复制到只读、content-addressed snapshot，再只从该 snapshot 解析；原始路径在 hash 与解析
之间变化不会改变输入。ZIP 只抽取 manifest 明确指定且通过 CRC、成员路径和内容摘要检查的
loader member。

MOABB 仅用于已验证的信号解释，不再负责下载或解压。项目在 dataset instance 上安装只读
`data_path` proxy，指向 snapshot 物化目录，从而绕过 MOABB 1.6.1 中 BI2015a 的错误
`str.strip()` 路径推导，也避免 dataset-specific MNE config 将“验证 A”悄悄切换成“读取 B”。
BNCI2014-008 MAT 同样从认证 snapshot descriptor 读取，并显式记录 MATLAB 1-based trial
索引到内部 0-based sample 的转换。

## 旧数据启示

- causal-v2 BI cross-decision 曾显示 source normalization 优于 target-prefix，轻量
  personalization 没有可靠净收益。它只决定 v3 首轮保留哪些对照，不提供 v3 数值。
- BI+BNCI common-CAR v2 曾出现负迁移，说明物理通道可加载不代表梯度有益；v3 首轮
  必须保留 single-domain source，不直接采用联合源。
- 旧 8-subject baseline 与 BI head ablation 只说明旧 cache/QC/训练合同下哪些 baseline
  和 head 值得保留为 v3 对照；它们缺少当前 identity、lineage、TrainingRunContract、
  EvaluationRunContract 和 DecisionOutcome，不能与 v3 指标直接合并。
- GTN 旧开发支持 0.1 Hz/1200 ms、K35 和 candidate mean 进入新假设，但 GTN 是儿童
  3 导、单固定目标数据，不能裁决成人 BrainSync target-switch。
- 本机旧 BrainSync sessions 均未通过 analysis-ready 合同，只能作为严格 validator
  的反例，不能作为准确率样本。

这些结论必须以“旧合同下的开发启示”引用，不得复制旧 checkpoint、结果数字或
manifest 到新 promotion 目录冒充当前证据。

## 当前 candidate promotion 链

当前 BI2014a 与 BNCI2014-008 cross-decision 共用以下物理链：

1. runner 必须接收 `--source-snapshot-manifest`，该 manifest/相邻 tar.gz 必须通过
   `n2p3_source_freeze/1` 的完整 40 位 source commit、byte size、member count 和
   SHA-256 复核；禁止传裸 source SHA。
2. cache provenance 必须携带 `n2p3_candidate_task_contract/1`；活动 row-column
   producer 统一写 `candidate_id=0..11`、`row_code/col_code`、`target_row/target_col`、
   `selection_id` 与 `repetition_index`。训练标签从 candidate-membership 推导；
   `raw_is_target` 或 `raw_target_label` 只要存在就必须交叉审计，不因 cache 是 CAR
   等派生层而跳过。
3. `run_candidate_cross_decision.py` 的 `/1` result 必须携带中央
   `DecisionPlanContract`、`EvaluationRunContract`、authority participant keys、primary
   `DecisionOutcome` 和 typed `decision_failures`。失败 decision 仍保存冻结 target truth；
   promotion 的 `test_reps` 不得小于 8。
4. `build_candidate_cross_decision_manifest.py` 从真实 result 和 checkpoint 内嵌
   `TrainingRunContract` 构造 manifest，现场核 source/result/checkpoint 文件 hash，并为
   每个 arm 生成中央 `ArmContract`。主指标的 R 不得小于 8，禁止人工复制旧 partition
   JSON。
5. `analyze_candidate_cross_decision.py` 只接受 builder 生成的 `/1` manifest；同一 arm 的
   source training procedure 与 target adaptation procedure 不得改变，只允许显式的
   training replicate 和 participant partition 轴变化；result 必须完整包含 level 8。

decision accounting 分为 `data_eligible/data_ineligible` 和
`evaluation_successful/evaluation_failed`。前一组回答数据能否进入评估，后一组回答
适配/推断是否成功；fit failure 不得伪装成数据不合格，也不得从 operational denominator
中消失。旧 `subject_decision_ledger`、顶层 local-subject checkpoint 字段和重复 hit-rate
汇总不再是活动接口。

participant 与 decision 是两个不同 estimand。完整用户请求 cohort 写入
`DecisionPlanContract.requested_participant_keys`；没有可定义 test decision 的 participant
写入 `participant_selection_failures`，不得伪造 decision。它在
`requested_participant_operational_hit_rate` 中作为 0 保留，但不进入 planned-decision
accuracy 的 decision denominator。analyzer 分别输出 participant operational interval 和
显式 denominator 的 decision endpoints，禁止用同一个“accuracy”名称混报。

若一个 participant 没有任何完整 R8 test decision，runner 仍须完成 checkpoint 的目标域、
权重和逐 participant 身份排除验证，然后写出
`completed_with_selection_failures`：participant endpoint 为 0、decision denominator 为 0、
且不生成伪 decision。单个独立 participant 只能给点估计，区间必须标为
`not_estimable`，不得用退化 bootstrap 区间冒充不确定性证据。
