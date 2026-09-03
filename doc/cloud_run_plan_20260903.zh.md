# 云端训练开跑方案（2026-09-03）

_版本：2026-09-03 20:55 +08:00 · 角色：执行方案，从属于 [`research_program.zh.md`](research_program.zh.md) §9.3 执行顺序 · 基线：工作树 HEAD `6c4abe8`（活动规范基线 `ef7a40e`，文档提交在其上）· 前置检验：[`supervisor_full_audit_20260903.zh.md`](supervisor_full_audit_20260903.zh.md) ·
更新（20:55）：C1–C4 已谨慎落地并通过聚焦测试（4/4）；BI2014a causal-v3 64 人 cache 已重建（tmp/，待 attestation）。详见 §4.1 与 §2。_
_更新（22:15）：本地 CPU mini-smoke 全链通过（§3.3）；L0 三证 attestation + cloud_inputs 清单完成（§2.2–2.3）；修复 P2 disabled-ledger 与 input_statistics_scope 两个真 bug（§3.3）。等 T：提交批次、提交后 §2.1 archive、cache 正式位置、D2/D3/D5。_

---

## 0. 结论：可以开跑，分三层走

代码与合同层已就绪（774 用例空闲全绿、P0/P1/P2 契约链逐条验证、flaky 四模式已处置）。
但按活动规范 §9.2/§9.3 的纪律，**"开跑"不是一键全量 evidence，而是三段**：

| 层 | 内容 | 就绪状态 | 产出定位 |
|---|---|---|---|
| **L0 数据与归档闭合** | 当前树 source-only archive、causal-v3 cache 重建、attestation | 🟡 进行中（BI2014a causal-v3 64 人 cache 已重建至 tmp/（2026-09-03 20:49，3m5s，含 candidate audit）；source-only archive 等待合同闭合 commit 落定后生成；attestation 未跑） | 一切后续 run 的输入 |
| **L1 云端 development 第一轮** | dry-run → 单 worker CUDA smoke → development 计划全跑 | ✅ 可立即执行（`stage: development` 模板已就绪） | 机制观察 + 工程验证，**不是** evidence bundle |
| **L2 evidence-grade paired arms** | matched fold/seed 对照 + participant-cluster CI | ✅ 前置 C1–C4 已落地（2026-09-03 20:55，测试验证见 §4.1）；剩 plan↔result 声明比对复核 | 可对外主张的 paired CI |

诚实边界：在 L2 的合同缺口闭合前，任何云端 run 只能写"development/机制证据"，
不得写"feature transfer 成立"或任何 `hit@R` 产品主张。

## 1. 与 §9.3 执行顺序的映射

| §9.3 条款 | 本方案落点 |
|---|---|
| 1. 生成当前执行树 source-only archive + manifest（禁复用旧归档） | §2.1 |
| 2. active cache 的 protocol/identity attestation；direct 与 row/column 不混用 P2 | §2.2–2.4 |
| 3. 绑定 source/environment/cache/plan/checkpoint 的 dry-run，再单 worker CUDA smoke | §3.2–3.3 |
| 4. 预声明 matched fold/seed 下跑 pooled-BN、P0、P1（及允许的 P2 direct arms），报 cluster CI | §4.2–4.3 |
| 5. target-calibration、row/column dispatch、failure ledger 闭合前不扩主张 | §5 D2/D4、§7 |

## 2. Stage 0：数据与归档闭合（L0，本地即可开工）

### 2.1 当前执行树 source-only archive

- 对工作树（**必须在合同闭合 commit 落定后**——manifest 的 `source_commit` 必须与
  archive 字节可互相印证，未提交的工作树冻结会破坏可复现性契约）生成
  `source-only.tar.gz` + manifest，沿用 frozen/ 现有 manifest 格式（`n2p3_source_freeze/1`，
  六个契约字段 + scope/replacement_policy 自述字段）；**不得复用**
  `c7bf860`/`e9b7a5e` 旧归档。
- 该 manifest 是 `run_pretrain_supervised.py --source-snapshot-manifest` 的必选输入，
  也会被 C1 的 `verified_source_snapshot` 全量校验。
- **✅ 已完成（2026-09-04 00:45，commit `3aa0e17` 落定后）**：
  `frozen/active_source_3aa0e17.source-only.tar.gz`（187 members，947,751 B，
  `git archive` 于 HEAD，scope=src/experiments/configs/tests/doc/build env，
  零数据/缓存/二进制泄漏）+ `frozen/active_source_3aa0e17.manifest.json`。
  C1 校验器三重身份互证通过：archive `8733fed7…` + manifest `1447920c…` +
  commit `3aa0e170c425bc2b…`。归档/清单未提交（留 T 下一批次）。

### 2.2 causal-v3 source/target cache 重建

- 对象：BI2014a（64 人；历史上唯一有 matched 结果的数据集）。从 raw 重建为当前
  `--cohort causal`（0.1 Hz / 1200 ms forward steady-state 契约）v3 缓存；
  历史 v2 缓存只作对照证据，不得复用。
- **状态（2026-09-03 21:25）：BI2014a 单域 v3 cache 已重建并完成 attestation 全链** ——
  `tmp/l0_bi2014a_causal_v3_64subj.npz`（629MB）+ `.candidate_audit.json`
  （`prepare_bi2014a_candidate.py` 直读 64 人 csv，绑定
  `SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT`；64 人 / 61,011 epochs /
  dropped_tail_flashes 总计 2）。attestation 三证齐全：
  ① verified load（`n2p3_verified_epoch_cache_load/1`，cache SHA-256
  `842466e4…`，full_contract_validated=true）；② named-contract 字段级一致
  （bind `--contract causal` 报 already-uses，物理预处理字段全匹配）；
  ③ preflight artifact QC（--full-contract-check，64 fold **零失败**）。
  cloud_inputs 清单（三文件 SHA-256 + 参数 + attestation 记录）已生成：
  `tmp/l0_cloud_inputs_manifest.json`。**cache 等列文件在 tmp/ 临时区，
  T 定正式位置后整体迁移。**
- 多域 source cache：`prepare_multidomain_source.py`（`--source NAMESPACE=cache.npz`，
  `--target-channels` 显式声明共享通道），共享通道、平均参考、域注释、binary-evidence
  投影由该工具强制执行。**单域 cache 不依赖此步；多域合并待 D2 域选择裁决。**
- direct 与 row/column 语义由 cache 的 event contract 决定，**以 preparation 输出的
  attestation 为准**，不事后解释。

### 2.3 attestation 与上传

- `preflight_artifact_qc.py` / `bind_preprocessing_contract.py` 做协议与身份证明
  —— **已完成（见 §2.2 三证）**。
- 生成 `cloud_inputs/` 清单（每个 npz 的 SHA-256 + 参数记录），随归档上传云端；
  云端 run 前逐文件复验哈希。—— **清单已生成（`n2p3_cloud_inputs_manifest/1`），
  待 T 定 cache 正式位置后随迁。**

## 3. Stage 1：云端 development 第一轮（L1）

### 3.1 环境锁定（最小记录形式）

完整 environment-digest 合同闭合前，development run 先以"记录在案"形式落盘到
run manifest：commit SHA、`uv.lock`/`requirements.txt` 哈希、`python --version`、
`pip freeze`、`nvidia-smi`（驱动/CUDA/型号）、种子。云端 shell 天然满足
"独立 shell" 验收纪律。

### 3.2 dry-run（不花钱的排雷）

`run_candidate_promotion_matrix.py --plan <plan.json> --dry-run --max-workers 1`

预期检查点（dry-run 的存在意义）：
- plan schema `n2p3_candidate_promotion_plan/2` 校验、DAG 构建、journal/checkpoint 路径；
- **P1 domain-axis gate**：development 模板 `training.source_residual_drop_p=0.3`，
  但 `source_arms[0]` 的 `selection_domain/source_domain_mass/splitbn_source_weights`
  均为 `null`。
  **✅ 已实测裁决（2026-09-03 21:50，本地 dry-run）**：单域 BI2014a v3 cache +
  原模板（P0 pooled_source + P1 drop_p=0.3）→ gate 拦截
  `enables Split-BN/source residual training but its cache does not declare
  at least two source domains`（`run_candidate_promotion_matrix.py` line ~1724，
  `indexed_domain_requested` 同时含 splitbn buffer source 与 drop_p 两条件）。
  **重要修正：gate 触发条件是 `is not None`，因此 (b) 变体的"drop_p=0 只关 P1"
  在单域下同样被拦——单域首轮必须 P0/P1 双 null**
  （`splitbn_target_buffer_source=null` + `source_residual_drop_p=null`）。
  实测双 null 变体 dry-run **完整通过**（DAG：checkpoint_batch → 4 result →
  manifest → analysis，plan digest `1e1afdcb…`）。
  两条路线按 gate 实测结果二选一：
  (a) 建真多域 cache（≥2 个声明域；BI2014a 单数据集下"域"的定义是研究设计
  决策，需 T 拍板——如 BI2014a+GTN 跨设备域）；
  (b) 单域首轮 P0/P1 双 null（feature-transfer 分解 development 照跑，
  P0/P1 机制验证推迟到多域 cache 就绪）。
  **注意：plan doc 原 (b) 描述"只跑 P0"已被实测否决。**

### 3.3 单 worker CUDA smoke

`--device cuda --max-workers 1` + 短 epoch（如 epochs=2）跑 1 个 fold/1 个 target：
验证 checkpoint `/2` 落盘、journal 状态机、buffer source 记录、CUDA/精度链路。
模型仅 ~1.5k–3.6k 参数，瓶颈预计在数据 IO 与 fold 数；**以 smoke 实测单 fold
时长为准再外推总预算**，本方案不预填编造的时长数字。

**✅ 前置：本地 CPU mini-smoke 全链通过（2026-09-03 22:10，dev_smoke5）**。
`--device cpu --max-workers 1`，12 人 BI2014a v3 cache，epochs=2，bootstrap=1000，
单域合法最小配置（P0/P1/三域损失全 null/0），完整链
pretrain → checkpoint_batch(/2) → candidate×4 → manifest → analysis
端到端 exit=0（smoke_run11，全链 ~110s）。产物核验：
- result 含 C1（source_commit + 双快照 SHA-256）、C2（environment 9 域 +
  environment_digest + 3 lockfile 哈希）、C3（splitbn 块，单域下
  buffer_source=null 属预期）、C4（oom_accounting 存在、decision_accounting
  分母保持）字段；
- analysis.json：planned_contrast（target_adapter vs zero_shot，participant
  配对）+ paired_bootstrap_interval "estimated"（n=2，iterations=1000）；
  数值退化（全 0）为 mini smoke 预期，仅验结构。

smoke 过程中钉死的两个真 bug（均已源头修复并回归验证）：
1. **P2 disabled-ledger 语义缝隙**：deep.py fit 开头无条件初始化
   `last_decision_loss` 为完整 disabled ledger，pretrain 透传进 checkpoint 的
   `decision_loss_selection/refit`；checkpoint 校验器二分合同〔P2 契约+激活
   ledger〕∨〔零 P2 字段〕——disabled ledger 落缝隙，导致 **P2-free checkpoint
   从不可被 candidate runner 读取**。修复：pretrain 写入端
   `_durable_decision_ledger`（status=="disabled" ⇒ None；其余原样，校验器
   继续强制），deep.py 序列化恢复 `dict | None` 忠实类型（不再 `or {}` 伪造）。
2. **`input_statistics_scope` realized 计数污染 procedure identity**：
   pretrain 曾把声明字符串 `"all_source_rows"` 膨胀为
   `{mode, rows, unique_subjects}`，rows/unique_subjects 随 partition holdout
   变化（9696 vs 10392），导致 manifest arm check 拒绝
   `changes model or training/adaptation procedure`。修复：恢复纯声明字符串
   （realized 遥测归 batch record/ledger，不进 procedure identity，与
   `training_procedure_record` allowlist 哲学一致）。

**新发现的上游 fail-fast 缺口（待 T 裁决是否入 C 系列）**：
`statistical_design.bootstrap_iterations` 在 plan 解析与 manifest 构建层均无
下限校验，analyze 层硬地板 ≥1000 才拒绝——本次 smoke 用 100 在全链第 6 步
（analysis）才爆。可考虑在 plan parse 处同步 1000 地板（fail-fast，附测试）。

**WorkBuddy 会话纪律注**：orchestrator 锁文件释放（unlink）会撞 bulk-delete
守卫（>50 文件阈值）产生 SystemExit 噪声，不影响计算正确性；独立 shell /
云端跑不受影响。

**⛔ 本地 CUDA smoke 受环境阻塞（2026-09-03 22:25 实测）**：本机有 GPU
（torch.cuda.is_available()=True），但 CUDA 路径 `torch.compile`
（`DEFAULT_COMPILE_MODE="reduce-overhead"`，runtime.py:32，DeepConfig 默认）
→ Inductor 需 Triton，Windows venv 无 triton → `TritonMissing` 崩溃
（dev_cuda1 残留失败 journal，归 D5 清理）。CPU 不经此路径（Inductor C++
codegen），故 CPU smoke 全绿。**pretrain CLI 无 compile 开关**；用
`TORCHDYNAMO_DISABLE=1` 强关会让 checkpoint 仍记录 reduce-overhead 而实际
eager 执行——record-vs-executed 不忠实，按记录纪律不采用。两条合法路径待 T：
(a) venv 安装 `triton-windows`（第三方轮子，改 uv.lock）；
(b) pretrain/plan 暴露 `--compile-mode`（含 None），smoke 显式关并如实记录。
云端 Linux 不受影响（torch 官方轮子自带 triton），D3 算力数据由云端 smoke
实测取得。

**✅ 云端 CUDA smoke 全链通过（2026-09-04 01:45，AutoDL westc RTX 5090 32GB，cuda_smoke3）**。
环境：208 核 / 754 GB RAM / CUDA 12.8 toolkit；venv 复用旧架构时期 site-packages
（torch 2.8.0+cu128——5090 sm_120 CUDA 实测可用；np 2.5.2 / sp 1.18.1 / mne 1.12.1 /
sk 1.9.0 / pd 3.0.5 全部满足 requirements 下限），`PYTHONPATH=src` 导入项目包。
上传 10 文件（归档+清单+64 人 cache+sidecar+12 人 cache+2 plan）双端 SHA-256 全对
（630 MB 上行 8m14s ≈ 1.3 MB/s，本地上行瓶颈）。云端布局：
`/root/autodl-tmp/{n2p3-frozen, n2p3-cloud-inputs, n2p3-runs, n2p3-3aa0e17}`。

**smoke 抓出第三个真 bug（已源头修复，deep.py）**：`torch.compile` 默认
`reduce-overhead` 在 CUDA 上启用 CUDA Graphs——编译区输出驻留**静态 buffer**，
下一次 invocation 即覆盖。train_step 返回的 metric 分量仅 `.detach()`（不拷
存储），`.float()` 在同 dtype 时返回别名 → 跨 step 累加链直接引用静态 buffer →
`RuntimeError: accessing tensor output of CUDAGraphs ... overwritten`。
排障路径：区内 clone 无效（torch 明示 "clone the tensor outside of
torch.compile()"——区内 clone 产物仍是静态 buffer，cuda_smoke2 实证）；
`cudagraph_mark_step_begin()` 更不可用（允许 buffer 复用 → 静默读到被覆盖的
旧值 = 数学腐蚀）。**正确修法：train_step 调用点之后的 eager 层对 5 个返回
张量 `detach().clone()` 快照**（值完全一致，仅取得存储所有权；CPU 路径无感）。
本地回归 test_deep + test_decision_loss 绿；cuda_smoke3 exit=0（全链 1m22s），
result C1–C3 核验通过（source_commit=`3aa0e170…`、device=cuda、
environment_digest、snapshot SHA `8733fed7…`、splitbn digest）。
⚠→✅ deep.py 变更已闭合（2026-09-04 02:00）：commit `9f0b421`（fix）+
`838a39c`（冻结对 `frozen/active_source_9f0b421e.*`，C1 三重互证
archive `648789a9…` / manifest `e9fdf682…` / commit `9f0b421e…`），已上传云端
并在新树 `/root/autodl-tmp/n2p3-9f0b421e` 完成端到端探针验证。3aa0e17 旧对
留在 frozen/ 未跟踪（D5 裁决去留）。

**CUDA vs CPU 实测（同 12 人 cache、epochs=2、1 seed、2 partition、2 arm）**：

| 阶段 | CPU（dev_smoke5） | CUDA 5090（cuda_smoke3） | 结论 |
|---|---|---|---|
| pretrain（checkpoint_batch） | 17.5 s/job | 24.8 s/job | GPU 慢 1.4×（模型太小，launch/compile 主导） |
| result | 10.4 s/单元 | 5.3 s/单元 | GPU 快 2.0× |
| manifest+analysis | ~12 s | 3.5 s | 聚合便宜 |

→ 单卡 CUDA 对该模型**不是整体加速**；真正的杠杆是 208 核 host 的
`--max-workers` 并行度（D3 一并拍板）。

### 3.4 development 全跑

- 实例化 [`configs/feature_transfer_cloud_20260902.example.json`](../configs/feature_transfer_cloud_20260902.example.json)：
  替换全部 `PLACEHOLDER` 路径 → `cloud_inputs/` 真实缓存；`output_root` 改为
  带日期与 commit 短哈希的目录名；`source_snapshot_manifest` 指向 §2.1 新归档。
- 默认臂：multidomain source（P0 `pooled_source` buffer + P1 `drop_p=0.3/sample`，
  视 §3.2 裁决）× 3 seeds × {zero_shot, target_adapter}。
- 中断恢复用 `--resume`（journal 幂等已测；输出 SHA-256 重算对账，篡改自动重跑）。

**D3 预算外推脚手架（CPU mini-smoke 实测基准，2026-09-03 dev_smoke5）**：
实测（12 人 cache、epochs=2、CPU）：pretrain 每 job ≈ 17.5 s（34.9 s/2 job 串行）、
result 每单元 ≈ 10.4 s、manifest+analysis ≈ 12 s/轮。外推公式（F folds、
S seeds、A result-arms、E epochs）：

```
T_pretrain ≈ S×F × 17.5s × (64/12) × (E/2)
T_results  ≈ A×S×F × 10.4s × (64/12) × (E/2)
T_aggregation ≈ 12s × 轮数（可忽略）
```

模板 epochs=null（**待 D3 一起拍板**）。

**dev64 规模实测（2026-09-04 02:15 cuda_probe1，64 人 cache、2 partition、1 seed、
E=2、CUDA 串行，树 = 9f0b421e 新归档）**：pretrain **33.9 s/job**（63 人 holdout
行；固定开销主导——行数 ×6.3 只贵 1.37×）、result **28.4 s/单元**（64 人 cache，
≈线性于行数）、聚合 3.7 s。result 合同核验：source_commit=`9f0b421ee367…`、
snapshot SHA=`648789a9…`（新归档链闭合）。dev64 外推（F=64、S=3、A=2）：
pretrain 192 job ≈ **1.81 h 串行**、results 384 单元 ≈ **3.03 h 串行**（E=2 实测）。
workers 并行：**W=8 → wall ≈ 40 min（E=2）**；E 增大只放大 pretrain 行边际
（E=8 上界 ≈ 线性 ×4 → wall ≤ 2 h @W=8）。成本 = 实例小时数 × 5090 时价，
个位数人民币。**D3 的真正决策点是 epochs 终值与 workers 数，不是预算。**

**模板 PLACEHOLDER → 实际产物映射（2026-09-04 00:50 全部就位）**：

| 模板字段 | 产物 | 状态 |
|---|---|---|
| `source_snapshot_manifest` | `frozen/active_source_3aa0e17.manifest.json` | ✅ 已生成并 C1 验证 |
| `target_cache` | `cloud_inputs/l0_bi2014a_causal_v3_64subj.npz`（SHA `842466e4…` 迁移复验一致） | ✅ 已迁正式位置 |
| `source_arms[].cache` | 同一 64 人 cache（单域 P0/P1 双 null） | ✅ 单域即用；多域待域定义 |

**实例化 plan 已就绪并通过本地 dry-run（2026-09-04 00:55）**：
`tmp/l1_cloud/plan_l1_development_single_domain.json`（64 partitions × 3 seeds ×
{zero_shot, target_adapter}，单域 P0/P1 双 null，digest `6f5b30d2…`）。
注：`training.epochs=2` 为**占位值**（smoke 实证值）——D3 拍板终值后替换
（替换即改 plan digest，属预期设计）；云端实例化将 `output_root` 换为云端
日期+commit 目录（同理重算 digest）。上传集见
`cloud_inputs/l0_cloud_inputs_manifest.json`（4 文件含 record.json sidecar，
全 SHA-256 复验；`cloud_inputs/` 已入 .gitignore 防大文件入库）。

注意：模板默认臂（multidomain + P0 `pooled_source` + P1 `drop_p=0.3`）只有在
多域 cache 存在时才可实例化；单域第一轮须按 §3.2(b) 改写 arm（P0/P1 双 null）。

## 4. Stage 2：evidence-grade arms（L2，合同闭合后）

### 4.1 前置 runner 侧合同闭合（代码改动，需 T 审批）

| # | 缺口 | 规模 | 阻塞什么 | 状态（2026-09-03 20:55） |
|---|---|---|---|---|
| C1 | executed-source identity（实际训练用的 source archive digest 写入 result 并与 plan 声明比对） | 中 | 一切 evidence 主张 | ✅ 已落地：`verified_source_snapshot` 全量校验（schema/commit/byte_size/member_count/双哈希），result 记录 `source_snapshot_sha256` + `source_snapshot_manifest_sha256` + `source_commit` |
| C2 | environment digest（lockfile+硬件快照进 manifest 的 fail-closed 版） | 小 | 同上 | ✅ 已落地：`_environment_record`（python/numpy/torch/device/platform/cuda/cudnn/gpu + uv.lock 等锁文件 SHA-256，缺 uv.lock 即 fail-closed）+ `environment_digest` |
| C3 | target result/manifest 写入 P0/P1 effective buffer source、estimand、buffer digest | 小–中 | 同上 | ✅ 已落地：result 记录 `splitbn` 块（present / effective_target_buffer_source / buffer_estimand / source_weights / bn_buffer_digest，digest 覆盖 trunk 全部 BN buffer） |
| C4 | target OOM requested-denominator | 小 | 同上 | ✅ 已落地：survive 重试梯子的 OOM 转预声明 `accelerator_oom` fit-failure 进分母（`oom_accounting` 块证明 requested==ledger slots）；非 OOM RuntimeError 原样上抛；`DirectDecisionContractError` 永不降级 |
| C5 | target_calibration model-wide collector + runner 写入路径 | 中–大 | 仅 `target_calibration` buffer 来源被使用前必须；`pooled_source` 不需要 | 未做（第一轮不需要，见 D4） |
| C6 | row/column typed protocol dispatch | 中–大 | 仅 row/column 路线启用 P2 前必须 | 未做 |

C1–C4 测试（`tests/test_candidate_transfer.py`）：793 号集成测试扩展 C1/C2/C3 断言；
新增 OOM 转换端到端测试（requested=4 全部 fit_failure、分母保持、by-level 计数）、
非 OOM 上抛测试（无产物落盘）、`verified_source_snapshot` 单元测试。聚焦 4/4 绿；
会话内全量 773/774（唯一失败为 `test_raw_artifacts` symlink 用例，系 WorkBuddy 会话
shim 将 `symlink_to` 劫持为静默 no-op 所致——判别实验证据：调用不抛错且目标路径
`exists()==False`；防护代码无责，独立 shell 验收不受影响）。

### 4.2 matched fold/seed 对照矩阵

预注册（看 test 结果前冻结）：arm 集、fold 集、seed 集（≥3）、`R_max=8`、
planned contrast。最小对照：
- pooled-BN ↔ P0 Split-BN（同 fold 同 seed 成对；含反例 17 的 buffer-swap 归因链）；
- P1 off ↔ on（drop_p 0 vs 预注册 0.3）；
- zero_shot ↔ target_adapter（development 模板已有的 planned contrast 骨架）。

### 4.3 统计与判读

- participant-cluster bootstrap CI（`statistical_design.bootstrap_iterations=10000`
  骨架已在模板）+ 配对 sign-flip；
- 判读以 operational `hit@R`（prefix top-1）为主指标；trial AUC/BACC 是支持指标；
- 小 delta 一律对照历史重跑噪声底 **6.3e-4**；
- P2 若启用：仅 direct 语义 arm（λ_trial=0 拒绝、λ_candidate∈{0}∪[0.1,0.5]），
  row/column+P2 在 C6 闭合前禁止。

### 4.4 plan↔result 声明比对复核（L2 前置，2026-09-03 完成）

只读代码审计 + 沙箱对抗探针（篡改副本，未触碰任何真实产物）。

| 声明/证据 | 记录位置 | 比对点 | 裁定 |
|---|---|---|---|
| source archive identity（C1） | result 顶层双 SHA-256 + commit | manifest 构建以 plan 声明的 archive manifest **现算哈希**对账（`build_candidate_cross_decision_manifest.py` L249–254）；analyze 层 result↔checkpoint 二次对账（L344） | ✅ fail-closed；探针 B：篡改 result 哈希 → 当场拒绝 |
| environment（C2） | result 9 域 + `environment_digest` | **无任何消费点**——manifest 与 analyze 全文零 `environment` 引用 | ❌ record-only；探针 A：device/torch/全 1 digest 全改 → **静默放行**（exit=0） |
| target cache/identity | result + decision plan | manifest L255–260 plan↔evaluation 对账 + L270–276 跨结果一致 | ✅ |
| training procedure identity | checkpoint payload | manifest L296–301 model_origin vs payload 重导 + 跨 partition arm check | ✅ |
| checkpoint 字节 | result.checkpoint_sha256 | manifest L290–291 现算对账 | ✅ |
| contract digests | result | manifest L120/132 + analyze L206 语义重导 | ✅ |
| resume 输出完整性 | journal | orchestrator L1959–1967 SHA-256 重算，篡改→自动重跑 | ✅（journal 信任有锚） |
| statistical design 数值 | plan→manifest 透传 | analyze 层仅下限校验 | 🟡 值透传，已知 fail-fast 缺口（见 §3.3） |

**裁决与建议**：C2 的"进 manifest"半边尚未实现——同 arm 跨 partition 混环境
（如 CPU+CUDA、不同 torch 版本）会被静默聚合。建议新增 **C2b（小规模）**：
manifest 构建强制同 manifest 全部 result 的 `environment_digest` 唯一，并把
environment 记录写进 manifest 输出（analyze 只读即得）；附一个篡改拒绝测试。
未闭合前 L2 第一轮要求同 arm 全 partition 同环境（人工纪律代替机器强制）。

### 4.5 预注册草案（**DRAFT——D2 拍板后实例化并冻结，冻结前不得看任何 development 结果后选型**）

依 §4.2 要求预注册（看 test 结果前冻结）。带 ⚠ 的项随 D2 分支实例化：

- **数据路线**：⚠ D2 拍板后定稿——按 D2 重述（2026-09-03），可行路线即
  BI2014a v3 row_column（64-fold LOSO matched folds）P2-free；direct/row_column+P2
  依赖 C6，不在第一轮。
- **arm 集**（超参加锚 development 模板 `configs/feature_transfer_cloud_20260902.example.json`
  + §3.2 裁决；⚠ 实例化时逐值列出并 commit）：最小对照
  ① pooled-BN ↔ P0 Split-BN（`pooled_source`；含反例 17 buffer-swap 归因链）；
  ② P1 off ↔ on（`drop_p` 0 vs 0.3/sample）；
  ③ zero_shot ↔ target_adapter（planned contrast 骨架已在 runner）。
  P2 延后（C6 前禁止 row/column+P2；direct 路线若启用仅 λ∈{0}∪[0.1,0.5]）。
- **seeds**：20260901 / 20260902 / 20260903（模板已定，≥3 满足）。
- **R_max=8**；主指标 operational `hit@R`（prefix top-1），AUC/BACC 仅支持指标。
- **统计**：participant-cluster bootstrap（10000 次迭代，与模板一致）+ 配对
  sign-flip；多重比较 Holm 校正（与 20260828 消融实践一致）；小 delta 对照
  重跑噪声底 **6.3e-4**。
- **判读规则**：主指标 Holm 后显著且方向预声明才可主张；支持指标只作一致性
  检查，不作独立主张；任何 post-hoc 分析标记为 exploratory。
- **冻结程序**：本节去掉 ⚠ 项后 commit 进 git（commit hash 即冻结锚）；此后
  改动必须新版本号 + 变更理由附录，不覆盖。

## 5. 需要 T 拍板的决策点

| # | 决策 | 建议 |
|---|---|---|
| D1 | C1–C4 合同闭合是否现在做（我先出改动方案再落地） | 建议现在做，与 L0 并行 |
| D2b | C2b（environment_digest 聚合层强制 + manifest 写入，见 §4.4）是否随本批落地 | 建议做——探针实证 record-only，改动小 |
| D2 | ~~第一轮数据路线：A) BI2014a direct 语义 vs B) row/column development~~ **重述（2026-09-03 22:55 实测契约核对）**：候选任务框架硬门仅支持 row_column（`candidate_task.py` L88），两个数据 builder（BI2014a v3、bnci2014_008）均为 row_column，数据层无 direct 语义实现；P2 仅 direct 且显式拒绝 row_column 契约（`decision_loss.py` L433）。**真实选项 = P2 是否进第一轮**：(i) 无 P2 → row_column cache 立即可跑（smoke 已端到端验证），C5/C6 均不需要；(ii) 要 P2 → 先 C6（row/column dispatch，中–大）或新建 direct 数据路线（更大）。 | 建议 (i)：P2-free 第一轮先行，P2 闭环并入 C6 批次 |
| D3 | 云端算力规格与预算 | smoke 实测单 fold 时长后外推再定 |
| D4 | `target_calibration` buffer 来源是否进入第一轮 | 建议不进（C5 未闭合，`pooled_source` 足够） |
| D5 | tmp/pytest-flaky-hunt-1/2 与 flaky_hunt 日志去留 | 建议保留至第一轮云端 run 复现纪律验证后 |

## 6. 风险与降级

- **cache 重建是最大不确定性**：v2→v3 契约变更点（1200 ms/0.1 Hz/平均参考）若与
  历史不一致，attestation 会拦住——拦住是正确行为，不是事故；
- **P1 domain-axis gate 可能在 dry-run 拦下 development 模板**：按 §3.2 的 (a)/(b) 处置；
- **云端与本地可比性**：同一 commit + 同一 cache 哈希 + 噪声底 6.3e-4 判读；
- **降级路径**：CUDA 不可用 → CPU 单 fold smoke 仍可验证合同链（模型小，可承受）；
  P1 被 gate 拦 → 先跑 P0-only development，P1 第二波补。

## 7. 纪律（开跑期间不变）

1. WorkBuddy 会话内不跑验收 pytest；验收复跑走云端/独立 shell；
2. development 结果不写成 evidence；L2 前不提 `hit@R` 产品主张；
3. final test 不参与任何选择；超参（p、λ、buffer source）只能预注册；
4. BrainSync confirmation 前不声称 90% 产品目标成立；
5. 每 run 落盘：plan、journal、checkpoint `/2`、seed、环境快照、cache/归档 SHA-256。
