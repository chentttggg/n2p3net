# N2P3-Net 下一轮多-agent 分发任务书

_版本：2026-09-03 · 用途：在当前 dirty worktree 上分发 P0/P1/P2 与代码链审计任务_

---

## 当前基线

- 工作目录：`D:\BCI\n2p3-net`
- 分支：`main`
- HEAD：`a9047dd fix: record source archive member count`
- 工作树：已有用户改动，必须保留；当前涉及 `Paper/README.zh.md`、`doc/`、`pyproject.toml`、`requirements.txt`，不得 reset、checkout 或覆盖无关改动。
- `.venv`：Windows 项目环境；使用 `.\.venv\Scripts\python.exe` 与 `.\.venv\Scripts\ruff.exe`。
- 验证基线：全量 `pytest` 已通过，约 560 个测试函数，2 个 skip；Ruff 全部通过；`git diff --check` 通过。
- 当前源码事实：
  - `src/models/n2p3net.py` 仍是普通 `BatchNorm2d/1d`，没有 Split-BN。
  - `src/models/adapters.py` 只有 invariant/private 与 target residual，没有 source-domain residual/drop-to-consensus；`FeatureTransferConfig` 尚无 `source_domain_count/source_drop_p/source_drop_mode`。
  - `src/baselines/deep.py` 已有 feature alignment/private/orthogonality 训练路径，但没有 `DecisionLevelLoss`、完整 decision sampler 或 `L(m=0)/L(m=1)` ledger。
  - `src/models/candidate_evidence.py` 有统一聚合、`-inf` 空候选评分和 `missing_candidate_policy="abstain"`；评估路径已有 coverage/incomplete 处理。不要另建 runner 私有 aggregator。
  - `src/transfer/checkpoint.py` 已拒绝旧 `domain_adapters`，但尚未绑定 Split-BN architecture contract 或新 source residual 字段。
  - `src/transfer/evaluation.py` 的 `hit_at_repetition` 已是每个 prefix 的 top-1 曲线；缺 candidate 会计为 incomplete/miss。不要改成 top-R ranking。
- 当前研究边界：P0/P1/P2 均未实现；不能宣称 feature transfer、decision-level learning 或 BrainSync 产品 90% 已成立。真实 active cache/成人 BrainSync target-switch confirmation 仍未完成。

## 统一阅读顺序

1. `doc/research_program.zh.md`（唯一活动规范，2026-09-03 数学闭合版）
2. `doc/architecture_decision_20260902.zh.md`
3. `doc/implementation_plan_20260902.zh.md`（r3）
4. `doc/math_verification_20260902.zh.md`（含 2026-09-03 修正记录）
5. 相关源码与测试

## Agent A：P0 Split-BN 实现

**范围**：只改 `src/models/splitbn.py`、`src/models/n2p3net.py`、必要的 `src/models/__init__.py`、`src/baselines/n2p3net.py`、`src/transfer/checkpoint.py` 与对应测试/最小文档状态。

**要求**：

- 实现 `SplitBN1d`/`SplitBN2d`：共享 affine，per-domain running mean/var；训练 forward 必须接收 `domain_indices`，eval 必须显式声明 buffer 来源/索引，缺失时 raise。
- `N2P3Net(splitbn_domain_count=None)` 默认路径逐行为旧 BatchNorm；启用时三处全部替换：ST temporal、ST spatial、每个 MST branch BN。
- 传递 `domain_indices` 的接口必须明确，不能静默回退 pooled BN。
- architecture/checkpoint record 必须记录 splitbn 模式、domain count 与 buffer source；旧/新不匹配 fail-closed。
- target buffer 来源只能从 `pooled_source`、`source_weighted`、`target_calibration` 中预声明一种，并记录 estimand；不得偷偷更新 target running state。

**测试**：混合域统计隔离、同分布近似 pooled BN、缺 domain indices raise、三处 BN 均替换、eval buffer source contract、checkpoint mismatch。

**验证**：`.\.venv\Scripts\python.exe -m pytest -q tests/test_n2p3net.py tests/test_domain_adapters.py <新增测试>`；Ruff。

## Agent B：P1 source residual/drop-to-consensus

**范围**：只改 `src/models/adapters.py`、必要的 `src/baselines/deep.py` 接口与对应测试；不得实现 P2。

**要求**：

- 增加 `source_domain_count`、`source_drop_p`、`source_drop_mode`，校验并写入 record。
- `source_domain_residuals` 使用 `residual()` + 显式 scale + mask；禁止 `features + adapter(features)` 双加。
- 训练期 `m~Bernoulli(1-p)`，eval/target inner-loop 永不 route source residual。
- residual ratio 固定为逐样本、feature 非 batch 维 L2，`eps_res=1e-6`，报告 mean/q50/q95/q99/max；source gate 固定 counterfactual `m=1`；max 超过 `r_max=0.5` 时整次 run invalid/fail-closed，不删样本。
- 提供可审计的 `L(m=0)`、`L(m=1)` 记录接口；不要把 dropout mask 当 residual amplitude bound。

**测试**：zero-init 位级 identity；全零 mask 与 eval path `torch.equal`；eval 不含 source residual；p=0/0.3/0.5 语义；ratio 零分母/超限；`L(m=0)/L(m=1)` ledger 存在。

**验证**：`.\.venv\Scripts\python.exe -m pytest -q tests/test_domain_adapters.py <新增测试>`；Ruff。

## Agent C：P2 decision loss 与 repetition 轴

**范围**：只改 `src/transfer/decision_loss.py`、`src/baselines/deep.py` 训练接入、必要的 sampler/contract 模块与对应测试；不得改 P0 BN 实现。

**要求**：

- `DecisionLevelLoss` 绑定冻结 vocabulary；coverage 不完整返回显式 incomplete/fail-closed 结果，不产生 posterior/CE/argmax；`-inf` 只能作为诊断评分。
- candidate aggregation 使用 count-neutral mean；posterior 为 `softmax(s/tau)`，梯度含 `1/tau`。
- decision label 与 membership 的关系必须写成 `y_i=1[c_i=d*]`；`L_trial` 保留是优化稳定性、局部监督、正则化和校准的预注册设计选择，不是信息论必然。
- 支持 prefix 轴：`L_prefix=sum_r alpha_r L_candidate,r`；固定预算 `alpha_8=1`，曲线 arm `alpha_r=1/R_max`；evidence-aware `beta/tau_r` 只能由 calibration/meta-validation 冻结。
- batch 必须由多个完整 decision 组成，`K*N>=128`；缺失 decision 写 ledger，不当零损失、不从分母删除。
- 只允许 `L=L_trial+lambda_candidate*L_candidate`；`lambda_trial=0` 拒绝。

**测试**：含 `tau != 1` 的 autograd 梯度、candidate/truth 同步置换、constant bias、coverage missing、prefix alpha、batch grouping、lambda guard。

**验证**：`.\.venv\Scripts\python.exe -m pytest -q <新增测试> tests/test_candidate_evidence.py tests/test_transfer.py`；Ruff。

## Agent D：代码链/runner/checkpoint 交叉审计

**范围**：只读审计为主；除非发现阻塞性契约缺口，不直接改 A/B/C 的实现文件。可新增 `doc/` 审计记录和针对性 failing tests，但先报告冲突。

**核对链路**：`experiments/run_pretrain.py`、`run_pretrain_supervised.py`、`run_within_subject_transfer.py`、`run_brainsync_cross_decision.py`、`run_candidate_cross_decision.py`、`src/baselines/deep.py`、`src/transfer/checkpoint.py`、`src/research/contracts.py`、结果 ledger。

**必须回答**：

- P0/P1/P2 的配置如何从 CLI/config 进入 model、trainer、checkpoint、evaluation 和 manifest？是否存在字段丢失或默认隐式变化？
- participant/decision requested denominator、coverage/incomplete/tie/fit failure 是否完整进入产物？
- `hit@R` 是否始终是 repetition prefix 后 top-1 selection？
- OOM batch 改变、BN buffer 更新、target truth 泄漏、same-selection calibration 是否有 fail-closed？
- source/cache/checkpoint/environment/plan digest 是否绑定？

**交付**：写 `doc/next_agent_audit_20260903.zh.md`，按“源码事实→调用链→风险→建议测试→未验证证据”记录；不得把文档计划写成已实现。

## 集成顺序

1. Agent A 先完成 P0 与测试。
2. Agent B 与 Agent D 可并行；B 不改 P0 文件，D 只读交叉审计。
3. Agent C 在 A/B 接口稳定后实施；若必须并行，严禁修改同一文件的重叠区域。
4. 主 agent 集成每个 agent 的变更后分别跑 focused pytest，再跑全量 pytest、Ruff、`git diff --check`。
5. 每个 agent 必须报告 changed files、测试命令/结果、未解决契约冲突；不得自行 commit、push、reset 或清理他人改动，除非主 agent 明确安排。

## 禁止事项

- 不得把 `hit@R` 写成 top-R/rank-R。
- 不得把 `-inf` 后继续 softmax 当作 fail-closed。
- 不得用“decision 标签少”证明必须保留 trial CE。
- 不得声称横向集成无新表达力或必然塌缩。
- 不得用 buffer-swap 单独否定 Split-BN 训练期作用。
- 不得用 `\|Z\|+epsilon` 代替已冻结的 `max(\|Z\|, eps_res)`，不得用 dropout 零 mask 稀释 source residual gate。
- 不得把 synthetic/development evidence 写成 confirmation/product evidence。
