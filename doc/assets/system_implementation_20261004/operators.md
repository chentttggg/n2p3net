# 2026-10-04 算子与连续决策实施证据

本分支按 codebase-design、diagnosing-bugs、tdd，通过已约定的 fit→冻结→保存/重载→无标签预测和公开 CLI Seam 实施。没有修改默认 preprocess/epochs，也没有注册未经性能验证的默认生产路线。全部代码使用项目 .venv。代码前置冻结由主分支保存在 `frozen/system_implementation_20261004_before`。

## 交付的完整切片

1. `src/data/harmonic.py`：独立合法 segment 内的稳健窄带回归。输入 observed channels×samples（V），每个 segment 独立；输出同坐标清理数据和逐窗参数/摘要。默认 1 秒窗、50% overlap、50±0.5 Hz carrier 搜索、最高三阶 quadrature envelope，联合拟合 0–30 Hz DCT nuisance 和线性趋势，但只减去窄带项。低频字典用于保护信息，不是将低频从数据里减掉。频率在有时变 quadrature 时是局部 carrier center，不是独立可识别的物理工频估计。
2. `src/transfer/noise_model.py` 和 `operator_lda.py`：成对授权 source/calibration 的 raw residual/state draws→整体 PSD joint covariance（含 Ω）→声明的 A/B→calibration 类均值→可识别输出子空间相对 ridge→冻结 LDA logit→保存重载。未知 state 不进入预测 Interface。Σ、Γ、Ω 从同一组配对 draws 一次估计，不把两个边缘方差另加两次。追加恒零 state 坐标不改变活跃坐标的收缩估计。
3. `src/transfer/continuous_decision.py`：完整连续 calibration records→固定时间基上的 common ERP 与 target increment→calibration residual 估 AR(1) 和收缩 channel covariance→冻结模型→按公开 membership 构造每个候选设计、原始样本只计一次。matrix-normal ridge 的 kernel posterior covariance Vh 进入每候选 predictive covariance；用 Woodbury 计算二次型，同时计 candidate-dependent logdet。保存/重载包含 basis、kernel mean/Vh、channel covariance、AR φ、实际采样/通道顺序、operator identity 和拟合 ledger。

R1 的 OperatorAwareLDA 与 R2 的连续 AR 模型是分别可运行的切片；没有宣称已经把两者组合为一个模型。R1 的真实 Γ/Ω 识别尚需真实源截段/残差流程，软件不把 oracle states 包装成已估参数。

## 公开入口

`experiments/run_operator_decision_study.py` 提供七个命令：

- `clean --plan ... --output ...`：计划给 input NPZ(X)、sfreq、segments、scope_id、scope_role、config。test 输入必须显式声明 `test_offline_adaptive`，不能冒称固定状态或在线滤波。
- `fit --plan ... --output ...`：连续 labeled calibration NPZ(samples,onsets,target_mask)，固定 basis 或 basis 配方、sampling、channels、candidate_ids、operator_identity、fit_scope。
- `predict --model ... --input ... --metadata ... --output ...`：NPZ 仅允许 samples/onsets/membership，metadata 声明实际采样、通道顺序、candidate_ids列顺序、独立 record_id 和 operator identity；任何 truth/label 等附加载荷均拒绝。
- `noise-fit --plan ... --output ...`：成对 raw_residual_draws/state_draws，明确 scope 和唯一 draw IDs。
- `noise-project --model ... --input ... --output ...`：NPZ(A,B)，可将不同 epoch 的算子按行堆叠，保留 raw-overlap 及共享 state 的全部交叉块。
- `operator-lda-fit --plan ... --output ...`：冻结 noise artifact、A/B artifact、合法 calibration features/labels、scope/row IDs 和 input identity。
- `operator-lda-predict --model ... --input ... --metadata ... --output ...`：只接 features，拒绝 labels 和 oracle states，冻结 covariance 与均值不更新。

所有输出要求新目录，原数据不覆盖。包内数组与 metadata（包括 AR 参数/通道等）共同做 digest；不只 hash 权重矩阵。数字数组以 immutable bytes backing 保存，不能通过 setflags(write=True)解除冻结。scope IDs 和 draw IDs 是可审计声明，不能替代上游真实 participant/time authority；caller 仍须来自合法 split/plan。确切复用的 calibration signal 和 record ID 在 continuous score 中拒绝。

## 先反例再修复

- 首轮 4 秒窗、仿射 envelope 对固定频率和 glitch有效，真实 BI01/02 末事件大幅下降；但连续频漂/幅相调制产生 213/340/766 µV RMS残留，不能因固定 50.01 Hz 好看就结束。
- 缩至 1 秒、三阶 envelope 后合成调制通过，但真实 ERP 注入仍可因低频字典不足而失真。6人完整反证保存于 `harmonic_envelope_unprotected/`，其中 BI01/12 注入相对误差约20.8%/30.8%。
- 加入 10 Hz nuisance 保护明显改善，但 BI12 仍约7.2%。检查发现没有 apply 分支改变；尾8秒低频字典从10→20/30 Hz，raw注入最大误差从1.84→0.0255/0.0178 µV。见 `harmonic_lowband_probe.json`。最终保护带固定为仓库现有分类通带0–30 Hz，重新核对全部6人；不是按分类成绩挑被试或挑路线。
- 局部DCT慢波叠加不变性：旧窄带回归最大误差10.7 µV；当前保护机制的 Interface 测试要求误差<1 nV并通过。该性质只保证声明的局部有限低频子空间，不等于任何真实ERP均无损。
- continuous 首个 serialization 反例显示只 hash arrays 时修改φ仍可加载；已把全部 metadata 纳入 artifact digest。
- unsigned onset 的差分会 wrap，导致乱序事件通过；已用直接相邻比较和 Python int 尾部范围检查，反例通过。
- joint noise 追加恒零state维度会改变 Ledoit-Wolf 的活跃协方差；已只在正方差坐标中估收缩，保留零方向。

初版代码/全量日志保存在 `harmonic_initial/`、`harmonic_envelope_unprotected/`、`harmonic_protected10/`。它们是失败与中间证据，不作为默认结果。

## 验收及证据边界

正式定向测试包括频漂/幅调/相调/glitch、segment 隔离、低频保护、独立 dense Gaussian 对照（白噪与AR=.7）、候选覆盖/通道/采样/乱序门禁、冻结重载、metadata污染、Ω抵消、恒零坐标、可识别秩以及完整真实CLI解析/文件执行。公式对照直接构造正交补空间的密集 covariance 与 logdet，未调用模型的内部设计或白化函数作期望值。

`run_operator_cli_demo.py` 是单独的可重复软件示例；调用七次真正 Python 子进程执行公开CLI，truth 只留在外部 scorer。其产生的 `operator_cli_demo*/validation.json` 保留 argv、exitcode、stdout/stderr、预测和重载结果。合成预测正确不构成公开/成人性能证据。

`probe_harmonic_implementation.py` 对 BI2014a 固定1/2/12/26/49/60号的独立C200处理，读取原始源身份与原始电极映射，仅清理授权C200片段；保留原production epoch FFT/crop/baseline。这是信号质量/ERP注入诊断，没有训练分类器、按成绩删人或用全record oracle更新处理状态。

连续模型仍假定线性叠加、完整event history、共同class noise与AR(1)结构；Vh是给定经验估计noise的条件后验，没有积分AR/噪声超参数不确定性。共享constant/slope使用diffuse nuisance投影，可能丢失真实慢ERP。预测概率仍需源内独立校准/校验，不因实现Gaussian公式就称真人概率可信。邻接未知decision不得通过真标签构造context，caller必须声明独立完整支持或另做联合未知决策模型。

R1 的draw IDs唯一不代表draws独立；实际重叠/相关的noise采样需要上游分块。真实噪声状态能否合法识别、借用源noise是否匹配target，以及新路线能否提高AUC/hit，均仍需配对公开开发矩阵。本分支不擅自替换已有源模型输入合同。

附加组合限制：R1 的 A 必须实际固定线性。harmonic 的频率、robust 权重和开关在当前 segment 内自适应，不能把条件映射当无条件固定 A 而遗漏估参不确定性。当前三个切片没有自动级联；将来组合必须冻结合法状态或另建自适应残差模型与独立协议。

## 最终固定配置实测（完整6人已完成）
四条真实观测差分为P3−Cz、Pz−Cz、P4−Cz、Oz−Cz；不是六导全部输入，也不是61人总体。各人固定C200，不按结果挑人。
| BI编号 | 末事件峰值：原→清理后 (µV) | 全200事件最大峰值：原→后 (µV) | PTP>500µV事件：原→后 | ERP注入增量相对误差 |
|---|---|---|---|---|
| 01 | 3545.04 → 70.13 | 3545.04 → 997.40 | 22 → 14 | 0.0979% |
| 02 | 7150.20 → 186.68 | 7150.20 → 3726.26 | 43 → 26 | 0.1322% |
| 12 | 349.00 → 99.81 | 349.00 → 228.27 | 0 → 0 | 0.1180% |
| 26 | 2882.71 → 321.80 | 3577.97 → 2943.68 | 179 → 173 | 0.1086% |
| 49 | 569.34 → 569.34 | 3001.86 → 3001.86 | 200 → 200 | 0.0000% |
| 60 | 283.01 → 41.57 | 676.62 → 675.83 | 2 → 2 | 0.3538% |

最终合成固定频率49.8/50/50.01/50.2Hz的RMS残差均<0.00004µV；连续频漂0.0153µV、幅度调制0.0426µV、相位调制1.5340µV，line+20mV glitch（保留glitch后比较）0.00417µV。clean ERP的数值差异约0.000026µV。见 `harmonic_probe_results.json`。

六人四通道诊断PTP>500µV总事件446→415；BI26仍173/200、BI49仍200/200。结果支持清理一类工频与有限边缘相互作用，不支持解决持续污染，也不支持自动删除这些事件/参与者。六人注入增量误差最大0.3539%，不是对任意真实ERP的无损证明。

最终定向pytest **13通过**；Ruff、compileall、限定路径git diff --check通过。最终七个公开CLI命令真子进程均exit=0，见 `operator_cli_demo_final/validation.json`。所有source/config/draw/输入摘要在model、CLI JSON与本目录验证清单中；未训练公共模型或推断成人效能。
