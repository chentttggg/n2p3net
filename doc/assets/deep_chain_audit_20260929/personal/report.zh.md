# 个人模型保存、重载与无标签推理语义审计

日期：2026-09-29。范围限定后端真实调用链；未启动 NN 训练、未连接设备、未修改生产源码。四个主文件的当前 SHA 均与本轮 `entry_manifest.json` 一致。

## 结论

**本轮没有发现“保存遗漏状态或重载偷偷换算法”足以解释当前低分的证据。** 已完成 6 个真实 LDA 个体包的原 raw→冻结处理→预测重放、3 个既有 NN 个体包的保存重载与公共 adapter 数学输出对照，以及 2 个包含 EOG/连续 QC/epoch QC 的真实个体包重放。主要差异来自明确保存的配方和评价单位，不能把公共 full-fine、legacy 产品选方、unified 固定校准、无源 LDA 都称为同一个方法。

一份历史 EOG+scratch NN 的本机重算 score 与原保存值最大相差 `9.536743e-7`；287 个可用样本的排序完全一致，AUC 均为 `0.4840686274509804`。同一当前环境下保存→重载完全一致。这个小差异不支持丢状态或性能下降归因；本轮未分离旧运行环境、算子版本和设备对数值舍入的各自作用。

## 1. 已执行的无训练探针

脚本：[replay_probe.py](replay_probe.py)；详细结果：[replay_probe.json](replay_probe.json)。所有探针将 `SubjectAdapter.fit` 替换成报错守卫，任何意外 NN 拟合都会立即失败。新生成的 roundtrip 个体包放在 Git 忽略的 `experiments/runs/deep_chain_audit_20260929/personal/`，原模型与原数据未写入。

| 原始个人条件 | 原 raw 重放 epoch 数 | 可用预测数 | 与保存 cache/score | 保存再加载 |
|---|---:|---:|---|---|
| Cz6 forward / range reject | 273 | 164 | 逐值相同 | 逐值相同 |
| Cz6 zero / range diagnostic | 212 | 83 | 逐值相同 | 逐值相同 |
| Cz6 zero / range reject | 159 | 68 | 逐值相同 | 逐值相同 |
| Native7 forward / range reject | 307 | 175 | 逐值相同 | 逐值相同 |
| Native7 zero / range diagnostic | 246 | 100 | 逐值相同 | 逐值相同 |
| Native7 zero / range reject | 194 | 82 | 逐值相同 | 逐值相同 |

以上 6 份模型来自 `experiments/runs/bidirectional_qc_20260909/`，共同推理记录为 `P001_20260907_150258_01`。重放同时核对 NaN/不可用位置，不是仅比较成功样本。全部原 raw 推理 dataset 的 `y is None`；没有把保存缓存的已知标签送进预测。

另外 3 份既有 NN：`product_lifecycle_v1/personal_model.pt`、`r2_plus_recovered_r3__source_seed20260908/model.pt`、对应 scratch 包。对 17 个合成 epoch，包含缺导 mask 和被遮罩导上的 NaN：

- 公共 `SubjectAdapter.predict_logit` 与个人包使用的 `predict_n2p3_checkpoint` 输出逐值相同。
- 保存→重载后输出、architecture record 和 state digest 均相同。
- 推理前后 trunk state，包括 BN 均值、方差、计数及 affine 参数均不变。
- 原包 state dtype 为 `torch.float32` 和 `torch.int64`；未出现 BF16 权重被悄然替换或保存遗漏。

这些 NN 探针使用的是已保存权重，合成的是输入，不产生新性能证据，也没有验证新的训练配方。

## 2. EOG 与两层 QC 的实物重放

补读 `brainsync_qc_remaining_20260908/retrieved/results/native7_target_only/` 的 `native7_bp01_eog1__waveform_lda` 与 `native7_bp01_eog1__scratch_seed20260908`。

两者均保存完整 EOG 模型、连续 QC 模型和 epoch QC 状态，并从真实原 raw 重建出 287 epoch：LDA 因完整所选通道要求可评分 167 个，NN 按缺导合同可评分 287 个。两者与旧结果的 availability 完全一致；LDA score 逐值相同，NN 最大差 `9.536743e-7`，排序/AUC 不变。证据见 [eog_replay_metrics.json](eog_replay_metrics.json)；CUDA 推理也检验了相同量级的差异，见 [gpu_replay.json](gpu_replay.json)，未训练。

把个体 metadata 的 `input_request.ocular_model` 去掉、保持 input contract 的已应用 EOG 状态，实际 `load_inference_session` 在 raw 读取前拒绝：`saved raw input request ocular_model differs from the model input contract`。这排除了“保存了名称、推理时实际漏用 EOG”的具体假设。

## 3. 保存和重载的真实状态路径

`calibration_candidates.py:303` 的 `FittedCalibrationCandidate.state` 保存所有 `trunk.*` state_dict 条目与最终 fitted input mean/std；LDA 保存 scaler mean/scale、线性系数/intercept 与固定选用通道。没有另建会覆盖原 classifier 的随机 head。当前产品候选使用 full-fine/共享 classifier；外部独立 linear-head 诊断不能冒充这一保存路径。

`calibrated.py:1112` 的保存载荷包括版本 schema、metadata、metadata digest、state、state digest；先写临时文件并 fsync，再以硬链接发布到不存在的新路径。已有模型路径拒绝覆盖。`:1134` 使用 `weights_only=True` 读取并检验两个摘要；`:515` 重建具体结构，并以 `strict=True` 加载 trunk state。

重建后还核对 architecture record、input statistics 的形状/正标准差、finite state、QC participant/reference/channels/preprocessing/session budget、预处理 execution binding。生产推理调用 `.eval()` 与 `torch.inference_mode()`，不执行 optimizer、BN 更新或 target statistics 重估。

本轮对 3 份 NN 个体包分别做 metadata/state 内容修改但不改摘要，共 6 个攻击均被拒绝。摘要是内容一致性约束，不是抗任意有权重写整个包者的数字签名；本轮不把它夸成来源认证。

## 4. 时间、身份与无标签边界

实际 CLI 链是：

`brainsync_model_task.py:307 load_calibrated_predictor → :409 predictor.load_inference_session → :411 inference_record`。

`CalibratedPredictor.load_inference_session`（`calibrated.py:609`）先核对冻结 `input_request` 与 input contract；通过 recording validator 在 raw 处理前核验 participant、校准后时间、session 复用、simulation、原采集参考与几何。后续恢复 EOG/QC、物理参考及通道顺序，再核对整个输入合同。推理 CLI 不允许覆盖频带、参考、EOG/QC 或另传 source checkpoint（`brainsync_model_task.py:407`）。

原 raw 入口 `data/brainsync.py:645` 使用 `validate_brainsync_bids_recording`；后者在 `brainsync_contract.py:311` 显式 `require_target=False`。构造 epoch 时 labelled 与 unlabelled 分开（`brainsync.py:187,228`），故真实推理 `y=None`，candidate truth 也不进入 `MembershipDecision`。`inference_record` 不调用评分真值函数，输出仅为预测/拒判和覆盖账本。

6 个真实 LDA 包分别攻击以下 5 项，共 30 项全部拒绝：早于校准就绪的记录；重用 calibration session；改变 simulation；改变物理参考；改变频带。另把已标注缓存的 y 单独翻转，会触发 candidate metadata 与 y 一致性检查。这是对已知标签缓存完整性的正确验证，不是模型读 target label；真正无标签入口上述 `y=None` 路径已单独实测。

直接调用底层 `predict_logit` 只验证输入合同，不承担 prospective session 边界；历史公开开发回顾评分会合法使用它。实际用户 CLI 在此前和 `inference_record` 内都校验时间/session，因此不能把未调用的底层 API 裸用方式当作当前产品泄漏。

## 5. 必须保留的配方差异

| 程序 | 实际含义 | 与公共评估的关系 |
|---|---|---|
| legacy 产品、有兼容 source | 在独立校准 session 中选 low/high 与 epoch，再从 source 全校准重拟合；batch64、source input statistics、source BN、checkpoint dropout | 与公共 online-BN/target-prefix full-fine 不同，旧公共收益不能直接验证该程序 |
| legacy 产品、未给 source | 完整授权校准上的固定 waveform shrinkage LDA，无内部方法选择 | 是明确的无源 LDA，不等于同结构 NN scratch |
| unified | 必须有显式 `TargetRecipe`；固定90轮、calibration BN、dropout off、全授权校准一次拟合；scratch prior 必为0 | 与统一训练机制相连，但本人direct-9任务/校准量不是 BNCI 36选一或公共K200 |
| no_transfer / five_candidates / transfer_only | 内层合法校准选方；无合法选方或候选全失败时可回退完整校准 waveform LDA | 应按记录的实际 `model_kind`、recipe、fallback、selection_evidence解释，不能只按请求名归组 |

最后一行的 fallback 是当前明确实现和已有合同测试覆盖的行为：`calibration_candidates.py:804` 设置 `fallback=full_calibration_waveform_lda`、reason 和 `selection_evidence=False`，最终 metadata 同时保存 `source_pretraining_used=False` 与实际 model_kind。`legacy_product` 源损坏/不兼容直接失败；`unified` 也拒绝不兼容 source，不走这条 fallback。它不是本轮新发现的隐蔽换法，但研究汇总若忽略实际 model_kind 就会混淆处理。

已加载的真实 `product_lifecycle_v1` 模型实际保存 `source_high`、2 epochs、batch64、source stats/source BN；另一份公开个人诊断 source 包保存 target-prefix/online BN、batch32、200 updates。两者即使都是 source-initialized N2P3，也不能当作同一次科学处理。

## 6. 决策聚合与概率含义

新个体包把 `decision_score` 保存为 `membership_llr_sum/1`、uniform prior、正 temperature、strict complete physical schedule、`probability_calibrated=False`。推理实际使用与公共统一评估相同的 `membership_decision_scores`（`models/candidate_evidence.py:148`）。这个核计算 `membership.T @ evidence`，再作 temperature/先验与 softmax；posterior 只是归一化证据，未被宣称是已校准置信度。

旧包没有该字段时继续显式 legacy mean，且每个候选必须具备完整重复数；不会因为缺一候选而在剩余候选中选冠军。无可用证据、覆盖不全或平局均拒判。

本轮直接9选一×5重复公式探针确认：完整等曝光时 legacy mean 与新 sum 的 winner 相同；加共同常数/正 temperature 不改变该探针 winner，temperature 可改变归一化分数；缺一 flash、全缺失、平局均拒判，即使全缺失 softmax 仍为有限均匀值。此结论是算术/合同验证，不表示 NN evidence 是精确 likelihood ratio，也不保证一般非等曝光或极小数值近似平局下结果相同。

公共 binary AUC、BNCI membership hit、个人九选一 hit 的监督单位和失败分母仍不同。个人选择器 `brainsync_operational_r5` 按每45-flash block计且保留失败，推理 `hit@all` 使用完成记录的所有声明重复；当前正常产品45-flash block上两者重复深度一致，历史450-flash同目标录制则不是同一个决策成本。

## 7. 结论范围

- 已排除本轮 11 个既有个体包/重放实例中的主要漏状态、漏 EOG/QC、推理重拟合和公共/个人普通 NN raw-logit 数学分叉假设。
- 已量化一个历史 NN 重放的约 `1e-6` 分数差，未改变可用性、排名或AUC；未借此解释当前整体性能。
- 未执行新的 NN 拟合，未获得新的成人/设备效能证据；没有把历史源包或合成输入当成当前 unified 个体模型性能。
- “保存正确”和“选用的配方有效”是两个问题；目前应优先处理主审计已确证的输入/预算/划分和研究口径问题，而非根据分数低臆测保存层丢参数。
