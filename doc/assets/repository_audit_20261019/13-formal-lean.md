# 13 · Lean 形式化分区（formal/lean）审计

- 审计对象：`D:\BCI\n2p3-net` 工作树中的 `formal/`（完全未跟踪），重点 `formal/lean`。
- 审计时间：2026-10-19；基线 commit `f1a303c`（HEAD，未含 formal/）。
- 只读约束：未运行 `lake build`、未安装依赖、未修改任何既有文件；本报告是唯一写入。
- 证据格式：`路径:行号`，级别 P0/P1/P2（P0=证据链断裂/不可复现，P1=工程化缺口，P2=改进项）。

## 0. 结论摘要（先读这里）

1. **formal/ 完全未跟踪是 P0，不是形式问题**：3.63 MB 证明源码（414 文件）、103.62 MB 证据（2371 文件）、2890 条已验收定理无任何 Git 历史；工作树整体有 4402 条未提交变更（doc 3410、formal 414、tests 184、frozen 146），HEAD 仍是 f1a303c。
2. **证据链已实际断裂（P0）**：doc 中 1528 条 formal/lean/artifacts/... 引用有 753 条指向已删除目录（49%）；verify 每次生成新目录名（attempt/序号），清理即永久断链。
3. **最严格的验证链已红（P1，对外声明则 P0）**：verify_erp.py 钉住 189 个源哈希，其中 6 个与当前工作树不符（features.py、structured_erp.py、stats/transfer.py、calibrated.py、exact_covariance.py、operator_lda.py），重跑会在 verify_erp.py:151-153 直接失败。设计本身优秀（回执→清单→证明→工具字节四层哈希，实测 4 个回执哈希与 8 个证明哈希全部匹配），但缺少 rebase 流程。
4. **可复现性锁死在本机（P1）**：lake 默认路径硬编码 D:/BCI/lean-tools/...（run_lean.py:415、verify_erp.py:94），MATHLIB_CACHE_DIR 同样硬编码（:427），PATH 上无 elan/lean/lake；.lake 2.67 GB/30887 文件、mathlib 2592 个 olean 构建耗时约 5.5 小时（10-16 10:19→15:52）。版本 pin 本身合格（toolchain v4.24.0、mathlib rev f897ebc、9 依赖固定）。
5. **审计文件一半是孤儿（P1）**：39 个 Audit*.lean 中仅 12 个被脚本引用，11 个由 verifier 运行时 write_text 重写（生成物混在源码树、无 generated 标记）；27 个无自动化入口；.lake 中 Audit*.olean 为 0。negative/ 93 个文件中 55 个从未被任何验证器执行。
6. **公理与反例机制本身可信**：三处一致的白名单 {propext, Classical.choice, Quot.sound}（verify.py:27），禁用 sorry/admit/axiom/native_decide（:100，verify_stage.py:34 更严），反例拒绝要求「退出码非 0 + unsolved goals + ⊢ False + 非 unknown module」（verify.py:150-155），避免导入失败被误判为反例通过。
7. **定理数表述内部诚实、外部易误读（P2）**：N2P3/** 有 3342 条 theorem、pilot 14 条、negative 499 条 example；accepted 2890、base 1150。README.md:131 与计划都写明「定理数≠覆盖率」，但证据索引用「136/1150 个定理」作标题，建议强制带 17 个子目标分母。
8. **无 CI 是维护负担的放大器（P1）**：.github 不存在；pyproject.toml:32 的 testpaths 不含 formal/lean/scripts，8 个运行器测试从不随主套件跑；13 个 verify 脚本 + check_current + 2 个 bridge 全靠手工，输出目录已累积 792 个。

## 1. 可复现性（版本 pin、外部工具链、缓存、从零构建）

### 1.1 版本 pin：源码层面合格，升级路径缺失

- 工具链固定为 leanprover/lean4:v4.24.0（formal/lean/lean-toolchain:1）；mathlib 以 git rev f897ebcf72cd16f89ab4577d0c826cd14afaafc7 固定（lakefile.toml:8），并在 lake-manifest.json:10 重复固定；另有 8 个传递依赖各自固定 rev（lake-manifest.json:18-100），packagesDir 为 .lake/packages（:3）。这一层是合格的：仅凭这三个文件即可唯一确定依赖图。
- 四个 lean_lib 均加 moreLeanArgs = ["-j1"]（lakefile.toml:12,16,20,23），defaultTargets 为 N2P3Continuous（:3）。默认目标只覆盖 N2P3Continuous → N2P3Program → N2P3Pilot + 9 个 N2P3 模块（Program 引入 5 个，Continuous 再加 4 个；N2P3Program.lean:1-7，N2P3Continuous.lean:1-5），不含任何 Audit*/negative/Checks 文件。
- 升级路径没有任何记录：仓库无 .github（git ls-files .github 为空，目录不存在）、无 elan 配置、README 只写「安装兼容 Lean/Lake」（README.md:26）。升级 mathlib 需要人工 lake update，并重新推导每个 verify_*.py 里硬编码的 EXPECTED_* 常量（verify_erp.py:32-54 一次就有 8 类：成员数 114、源码数 189、命令数 10、定理数 1150、错误例 110/101、3 个回执哈希、4 个模块定理数、8 个证明哈希）。这是隐性的升级税：任何一次合法证明改动都要同步改验证器，否则 verify_erp.py:165-174 直接 ValueError。
- 小瑕疵：包名仍是 n2p3_pilot（lakefile.toml:1），与 defaultTargets 的 N2P3Continuous 语义不一致，属 P2 命名债。

### 1.2 外部工具链：证据只能在本机复现（P1）

- 默认 lake 路径硬编码在脚本里：run_lean.py:415、verify_erp.py:94、accept_repaired_continuous.py:30 都是 D:/BCI/lean-tools/lean-4.24.0-windows/bin/lake.exe；mathlib 缓存目录同样硬编码 MATHLIB_CACHE_DIR=D:/BCI/lean-tools/mathlib-cache（run_lean.py:427）；benchmark_lean.py:26 也写死该目录。
- 本机没有 elan/lean/lake 在 PATH 上（Get-Command elan,lean,lake 全空），工具链只存在于仓库外目录；README.md:34 以「本机便携工具链」承认这一点。换机/CI 必须显式传 --lake，否则默认值直接失败。
- 运行还依赖仓库根 .venv 的 Python（run_lean.py:275-277 硬编码 root.parents[1]/.venv），以及 psutil/numpy/torch（verify.py:15，check_python_bridge.py:12-13）。Python 侧依赖由 uv.lock 固定，Lean 侧靠外部目录，两侧可复现性不对称。
- .lake/verification-environment.json 缓存了 LEAN_PATH/LEAN_SRC_PATH/LEAN_SYSROOT，值是 D:\BCI\... 绝对路径（该文件 4-9 行）；它被 .gitignore 覆盖（formal/lean/.gitignore:1），所以不会入库，但也意味着缓存不可移植，且指纹绑定的是本机可执行文件字节（run_lean.py:255-271 把 lake.exe/lean.exe 的 sha256 计入指纹）。这提高了本机证据强度，同时把可复现性锁死在本机。
- 结论：作为「本机可复现」合格（同一台机器上能重跑并失败即停），作为「第三方可复现」不合格。级别 P1；若对外发布证据，升级为 P0。

### 1.3 .lake 构建缓存：2.67 GB / 30887 文件，被正确忽略，但重建代价被低估

- 实测 formal/lean/.lake 共 30887 个文件、2.67 GB（十进制）。构成：packages/mathlib 2176.63 MB（2592 个 .olean，1683.1 MB），build/lib 171.70 MB（247 个 .olean），aesop 103.34 MB，build/ir 86.76 MB，proofwidgets 52.61 MB，batteries 51.79 MB。
- mathlib 的 .olean 时间戳从 2026-10-16 10:19:52 到 15:52:45，即同一台机器上仅 mathlib 构建就约 5.5 小时；项目自身从 10:32 起、最后一个 RawLoaderVerification.olean 落在 10-18 23:55。这是「从零构建」的真实成本量级：清空 .lake 后要重新下载 9 个依赖并按 5 小时量级重建，且四个 lib 都被 -j1 限制（每进程单线程；lake 仍可跨模块并行，但没有并行度记录）。
- 缓存被 formal/lean/.gitignore:1 正确排除，体积不会进 Git。问题在于没有任何「缓存获取」文档：README.md:28-31 只给 lake exe cache get + build，没提 MATHLIB_CACHE_DIR；仓库外 mathlib-cache 目录有 2834 个 .ltar、仅 0.12 GB，说明它只是增量下载缓存而非完整 mathlib 缓存，清空 .lake 后 cache get 会走默认用户缓存目录，可能再次全量下载。
- .lake 里还有验证调度状态：verification-scheduler/tickets.jsonl（257 KB）、check-0/1.lock、coordinator.lock、verification-compile.lock（run_lean.py:187-193 创建）。这些是运行时状态，同样被忽略，属正确处置。

### 1.4 artifacts/ 被 ignore 但 evidence 大量引用：证据链断裂（P0）

- formal/lean/.gitignore:5 忽略 artifacts/；实测 formal/lean/artifacts 有 792 个子目录、1834 个文件、仅 4.5 MB（report.json + lean.log 为主）。verify_*.py 每次运行都要求 output 目录不存在（run_lean.py:436，verify.py:92，verify_erp.py:113），即一次一目录，天然膨胀。
- doc 下的证据文件共引用 1528 条 formal/lean/artifacts/... 路径，其中 753 条指向已不存在的目录（49%）。典型断链：doc/assets/lean_gaussian_20261016/cleanup.json 引用 formal/lean/artifacts/attempt01（已删）；doc/assets/lean_erp_20261018/cleanup.json 引用 formal/lean/artifacts/erp_state_01/{report.json,lean.log}（已删）；doc/assets/lean_fold_artifact_20261018/author_candidate_01.json 引用 artifact_cv_selection_current_01..03（现存仅 04）。
- 也就是说：「清理旧构建」这一被计划鼓励的动作（verification_plan.json 的 cleanup_acceptance 要求删除过时构建）与「evidence 保留产物路径」互相矛盾。产物目录不可再生（目录名带 attempt/序号，重跑会生成新名字），因此删掉即永久断链。级别 P0。

## 2. 仓库治理（未跟踪风险、体积、命名与生命周期）

### 2.1 formal/ 完全未跟踪，且这不是孤立问题（P0）

- git status --porcelain -uall 共 4402 行：formal 414 条未跟踪、doc 3410 条、tests 184 条、frozen 146 条；同时 src 49 个已修改、experiments 17 个已修改、tests 109 个已删除。HEAD 仍是 f1a303c，工作树与提交严重脱节。verification_plan.json 自己也标注 git_head_is_not_working_tree_identity=true（该文件 baseline 段），属诚实但被动的披露。
- 后果：formal/lean 的 3.63 MB 源码（414 个文件，排除 .lake/artifacts）没有任何版本历史；33 个 Audit*.lean、93 个 negative、251 个 N2P3 模块一旦误删无法恢复；doc/assets/lean_* 的 103.62 MB 证据（2371 文件）同样 0 个被 git 跟踪（git ls-files doc/assets 仅 56 个旧文件，lean_* 为 0）。
- 入库方案（建议）：提交 formal/lean 的 414 个源码 + scripts + lakefile.toml + lean-toolchain + lake-manifest.json + verification_plan.json + README；保持 .lake/、artifacts/ 忽略；doc/assets/lean_* 只保留每轮的 acceptance.json/manifest/摘要，原始 report/log 走 frozen 归档。按现有 .gitignore（formal/lean/.gitignore:1-5）该提交天然干净。

### 2.2 体积：源码 3.6 MB，证据 104 MB，缓存 2.67 GB

- 源码 414 文件 / 3.63 MB；artifacts 4.5 MB；doc/assets/lean_* 103.62 MB / 2371 文件（最大 lean_current_checks_20261018 63.28 MB / 1149 文件，lean_personal_erp_20261018 27.08 MB / 672 文件）；doc/assets 整体 180.54 MB。
- 把 104 MB 证据放进 Git 会让每次 clone/备份多 100 MB 量级，且这些文件多为一次性 report/log。建议：只提交 acceptance.json 与 manifest（KB 级），大体积原始日志压进 frozen 归档并保留 SHA-256 清单。
- 2.67 GB 的 .lake 已被忽略，对 Git 无影响，但对整机备份有影响（如果备份工具不读 .gitignore，会多备份 3 万文件）。建议把 .lake 纳入备份排除项或迁移到集中缓存目录。

### 2.3 与 frozen/lean_* 的重复与谱系漂移

- frozen 下有 51 个 lean_* 目录、106.42 MB；doc/assets 下 25 个 lean_* 目录、103.62 MB；同名重叠 18 个，frozen 独有 33 个（多为分片：lean_v07_*、lean_personal_erp_*、lean_temporal_operator_20261018 等），doc 独有 7 个（含 lean_current_checks_20261018、lean_state_20261016）。
- frozen 的 106 MB 里只有 115 个文件被 git 跟踪（git ls-files frozen），lean_* 目录内跟踪数为 0；这符合「归档清单入库、载荷本地」的约定（AGENTS.md 的 archive distribution 规则），但意味着 frozen 也不能单独恢复证据。
- 命名漂移实例：lean_gaussian_20261016 与 lean_gaussian_20261016_v2 并存；lean_temporal_operator_20261016（doc）与 lean_temporal_operator_20261018（frozen）并存，且 verify_erp.py:115-117 同时依赖这两个不同日期的目录。这类跨日期配对没有索引文件，只能靠脚本常量维系。

### 2.4 生命周期与增长速度

- 33 个根级 Audit*.lean 的 mtime 从 2026-10-16T10:15（Audit.lean）到 2026-10-18T23:51（AuditLatencyReadout.lean），两天内新增 32 个；同期新增 93 个 negative、25 个 lean_* 资产目录，checkpoint 累计验收 2890 条定理（active_checkpoint.json 的 accepted_theorems）。这是明显的冲刺式增长，缺少「一个子目标一个目录」的收口规则。
- 生成物与手写物混杂：AuditCandidates/AuditConditional/AuditContinuous/AuditERP/AuditEventLedger/AuditJointState/AuditPersonal/AuditPosterior/AuditProbability/AuditStage/AuditTemporalOperator 共 11 个由 verify_*.py 在运行时重写（如 verify_erp.py:185-190），它们是生成物；其余 21 个根级 Audit*.lean 不被任何脚本引用（脚本里只出现 12 个 Audit 名称，含 Audit.lean），是手写物。两类文件同名同目录、同样可能被提交，属 P2 结构债。
- artifacts 里存在完全相同的两份文件：personal_erp_fit_outcome_audit_01.lean 与 _02.lean，均为 2012 字节、sha256 相同（a7869254...），是复制粘贴而非再生成。级别 P2，但说明 artifacts 未做去重。

## 3. 证明工程（生成方式、重复、proof↔code 绑定、公理、反例）

### 3.1 39 个 Audit*.lean 的来源：11 个是生成物，27 个无人引用

- 39 个文件的构成：根级 33 个 + N2P3/ 下 4 个（AuditCalibrationPopulation、AuditNetworkReadoutHeads、AuditPersonalERPSourceComponents、AuditSplitBNLifecycle）+ artifacts/ 下 2 个（personal_erp_scalar_inputs_01/02 内的 AuditPersonalERPScalar.lean）。
- 其中 11 个由 verify_*.py 在每次运行时用 write_text 重写：AuditCandidates、AuditConditional、AuditContinuous、AuditERP、AuditEventLedger、AuditJointState、AuditPersonal、AuditPosterior、AuditProbability、AuditStage、AuditTemporalOperator。生成内容固定为「import 本切片模块 + 逐定理 #print axioms」（verify_erp.py:185-190 是最完整的样板；verify_continuous.py:72-82、verify_candidates.py:74-83 同构）。
- 另有 Audit.lean 由 verify.py 消费但不生成，对 14 个 pilot 定理逐条 #print axioms（Audit.lean:4-18），verify.py:158-168 要求 audited 集合与 N2P3Pilot.lean 的 theorem 名单完全相等且公理在白名单内。
- 剩余 27 个不被任何脚本字符串引用（21 个根级 + 4 个 N2P3/ + 2 个 artifacts/）；根级 21 个为：AuditERPContinuation、AuditERPTemporalCLI、AuditERPTemporalRawCLI、AuditERPTemporalRawExecution、AuditERPTemporalRawPreprocessing、AuditFixedResample、AuditLatencyReadout、AuditNeuralSavedControl、AuditNeuralSavedStateProjection、AuditObservedStatistics、AuditObservedTemporal、AuditOcularChain、AuditParameterOwnership、AuditPersonalERP、AuditRawManifestSnapshots、AuditRequested、AuditRequestedBudgetPolicy、AuditRequestedWeightComposition、AuditSpatialEffectiveMaxNorm、AuditSplitBNMomentum、AuditSplitBNStatistics。全部 audit 文件合计 2299 条 #print axioms，但这 27 个没有任何自动入口执行。
- 结构佐证：.lake/build/lib 下 Audit*.olean 数量为 0（含 N2P3/ 下 4 个），N2P3 下 244 个 olean 对应 251 个源文件。33 个根级与 2 个 artifacts/ 文件不在任何 lean_lib 的 glob 内（lakefile.toml:10-23 只声明 N2P3Pilot/N2P3/N2P3Program/N2P3Continuous）；N2P3/ 下 4 个虽在 N2P3 glob 内，但不被 N2P3Continuous 传递导入，默认目标也不会构建它们。
- 风险：生成物位于源码树、无 generated 标记、无 .gitignore 例外规则；如果这些文件被提交，一次 verify 就会静默改写工作树（verify_erp.py:185 不检查文件是否存在）。级别 P1。

### 3.2 重复：artifacts 内两份完全相同的 Lean 文件；doc/frozen 18 组同名目录

- artifacts/personal_erp_fit_outcome_audit_01.lean 与 _02.lean 均为 2012 字节，sha256 完全相同（a7869254b36cbeb3...），是复制而非再生成。级别 P2。
- doc/assets/lean_* 与 frozen/lean_* 有 18 组同名目录（lean_candidates_20261016、lean_gaussian_20261016、lean_personal_erp_20261018 等），内容未必同步；另有 33 个 frozen-only、7 个 doc-only。没有单一索引说明「哪个是权威副本」。级别 P1（证据唯一性）。

### 3.3 proof↔code 绑定：哈希链设计良好，但当前已与工作树脱节

- 设计：验收回执同时固定 189 个 runtime Python 源的 sha256、114 个证明/配置/runner 成员、1150 个定理名单、Lean/Lake 可执行文件 sha256，并把上游回执自身的 sha256 再钉进下游脚本（verify_erp.py:38-40 的 EXPECTED_BASE_*，:115-135 读取并校验）。这是「证据链」而非「一次性报告」，设计强度高于一般项目。
- 实测当前一致性：4 个被钉住的回执/清单哈希全部匹配（doc/assets/lean_temporal_operator_20261016/acceptance.json=662eaa5e...、frozen/lean_temporal_operator_20261018/manifest.json=f10144f5...、evidence_manifest.json=f7c07ba8...、doc/assets/lean_pilot_20261016/acceptance.json=688d0330...），8 个 ERP 证明哈希也全部匹配。链本身没坏。
- 但工作树已漂移：189 个 pin 住的源里 6 个与当前文件不符——src/baselines/features.py、src/baselines/structured_erp.py、src/stats/transfer.py、src/transfer/calibrated.py、src/transfer/exact_covariance.py、src/transfer/operator_lda.py。verify_erp.py:151-153 会直接抛「Production source changed」。也就是说，最新一次 ERP 验收无法在今天的工作树上重跑通过；它绑定的是 10-18 的源码快照。verification_plan.json 的 baseline 同样记录这 6 个漂移，说明漂移发生在计划创建之后。级别 P1；若对外声称「当前代码已验证」，则为 P0。
- 绑定语义边界是明确的：check_python_bridge.py 首行写明「Finite behavioral probes of real callers; these do not prove Python semantics」，输出 scope 为「not a formal refinement proof」，且 passed 字段是字面 True（断言失败会抛异常，但字段本身不反映结果）。计划策略同样写明 finite_behavior_tests_are_not_refinement_proofs=true、cached_dependencies_not_rechecked_with_trust_zero_in_pilot=true（verification_plan.json 的 proof_hole_and_trust_policy），README.md:45 也承认没有 trust=0 重查导入库。这是诚实的边界声明，不是缺陷；缺陷在于没有机器可读的「哪些定理对应哪个 Python 函数」映射表，只有散文式表格（README.md:7-14）。

### 3.4 公理与反例：白名单严格，但只覆盖被脚本点名的那部分

- 公理白名单三处一致：verify.py:27 定义 {propext, Classical.choice, Quot.sound}，其余 verify_*.py 从 verify 导入且 verify_erp.py:111-112 再断言一次；verify_personal_erp.py:120,130 额外要求日志中不出现 warning。禁用构造用正则拦截 sorry/admit/axiom/native_decide（verify.py:100，verify_stage.py:34 还含 unsafe/extern/implemented_by/run_tac/run_elab）。
- 反例强制失败的设计可信：verify.py:150-155 要求退出码非 0、日志同时含 unsolved goals 与 ⊢ False、且不含 unknown module，避免把「导入失败」误判为「反例被拒」。verify_erp.py:42 还为每个 negative 文件钉了期望 example 数（6/4/4/8）。
- 覆盖缺口：negative/ 共 93 个文件、499 条 example 声明，但脚本只引用 38 个文件，55 个从未被任何验证器执行（FalseCalibrationPopulation、FalseDecisionLoss、FalseERPTemporal* 系列等）。同理 27 个 Audit*.lean 无自动化入口。级别 P1。

### 3.5 定理数量的表述风险：内部诚实，外部摘要仍易误读

- 规模：N2P3/** 有 3342 条 theorem 声明（0 条 lemma），根级 N2P3Pilot.lean 14 条；negative 499 条 example。checkpoint 记录 accepted_theorems=2890，verify_erp 的 base 记录 1150，说明「已验收」是子集而非全部声明。
- README.md:131 明确写「定理数量不表示项目覆盖比例」；verification_plan.json 的 strict_completion_policy.counts 也写「theorem count is not project coverage」；checkpoint 的 original_subgoal_status 按 V01.a…V10.b 逐项标状态。内部口径是对的。
- 风险在外层摘要：doc/evidence 索引用「136 个定理完成重查」「1150 个定理」作标题式表述（doc/evidence/README.md:16-19），读者容易把它当成覆盖率。建议所有对外句子强制带分母（17 个子目标中完成 N 个）而不是定理数。级别 P2。

## 4. 维护负担与 CI

### 4.1 没有任何自动化入口（P1）

- 仓库没有 .github 目录（Test-Path .github=False，git ls-files .github 为空），因此没有 CI 会执行 lake build、公理审查或反例拒绝。全部验证靠 13 个 verify 脚本（verify.py + 12 个 verify_*.py）+ check_current.py + 2 个 bridge 手工运行；每次都要传 --lake 绝对路径、要求 --output 是不存在的新目录（run_lean.py:436，verify.py:92，verify_erp.py:113），失败后目录已存在，重跑必须先改名或删除。
- 运行器自身有 8 个 pytest 用例（formal/lean/scripts/test_lean_runner.py:84-287，覆盖双检查上限、FIFO、内存门、崩溃恢复、环境缓存、路径排序、独占写、配置漂移），但 pyproject.toml:32 的 testpaths=["tests"] 不包含 formal/lean/scripts，所以这些测试从不随主套件执行。这是「有测试、没接线」。
- 并发模型本身有明确边界且被测试：最多 2 个独立 check 或 1 个独占 build（run_lean.py:176-185），第二个 check 还要求可用内存 ≥1.25 GiB（run_lean.py:20，verification_plan.json 的 execution_policy.second_check_minimum_available_ram_bytes=1342177280）。这属于加分项，但没有 CI 就无法防止回归。

### 4.2 checkpoint / verification_plan 的可核验性：好，但依赖本地磁盘

- 实测可核验：active_checkpoint.json 的 pause_checkpoint_sha256=f2dc0571... 与 pause_wrapup_20261018_01.json 的实际 sha256 一致；cleanup.sha256、current_accepted_receipt（doc/assets/lean_personal_erp_20261018/saved_control_manifest_acceptance_01.json）都存在；verification_plan.json 的 pilot_receipt_sha256=688d0330... 与 doc/assets/lean_pilot_20261016/acceptance.json 一致。
- 内容也诚实：checkpoint status=paused、paused_utc=2026-10-18T15:58:54Z、maximum_subagents=3、accepted_theorems=2890、original_subgoal_status 逐项给出 complete/in_progress 与证据文件；plan 的 authorization 写明「只在 resumed 时执行」以及 10-18 从 4 降到 3 个子代理上限后用户要求 pause and wrap-up。
- 问题在于被核验对象 0 个进入 Git：上述 acceptance/manifest/pause 文件全部是 untracked（git ls-files --error-unmatch 全部失败）。clone 到新机器后 checkpoint 指向的文件不存在。级别 P1，和 2.1 同源。
- 另有一个语义缺口：checkpoint 的 accepted_theorems=2890，而源码中有 3342 条 theorem；两个数都真实，但没有任何机器可读映射说明「哪 452 条未验收」。目前只能靠 verifier 常量与 checkpoint 名单人工比对。

### 4.3 维护负担的量化

- 期望常量：verify_erp.py:32-54 一类常量在 13 个 verify 脚本中重复，任何合法证明增删都要改脚本；verify_erp.py:173 一旦数量不符直接失败。
- 自动生成文本：33 个 Audit*.lean 合计 2299 条 #print axioms；其中 11 个每次运行被整体重写，导致 diff 噪声与「源码被测试改写」的治理问题。
- 孤儿资产：27 个 Audit*.lean、55 个 negative/*.lean 无自动化入口，需要人工记忆维护；一旦 Lean 语法或 Mathlib 升级，它们不会被 CI 发现损坏。
- 环境：4 处硬编码外部路径、2.67 GB 本机缓存、5.5 小时 mathlib 重建时间、103 MB 未跟踪证据。换机成本约为「半天到一天」。

## 5. 工程化改进建议

### 5.1 P0（先做，直接决定结论可信度）

1. **formal/ 入库**：提交 formal/lean 下 414 个非缓存文件（3.63 MB：385 个 .lean、scripts/23 个 .py、lakefile.toml、lean-toolchain、lake-manifest.json、verification_plan.json、README.md、.gitignore）。现有 formal/lean/.gitignore:1-5 已正确排除 .lake/、*.olean、*.ilean、__pycache__/、artifacts/，无需改规则即可得到干净提交。
2. **证据入 Git 的边界**：只提交每轮的 acceptance.json / manifest.json / evidence_manifest.json / cleanup.json（KB 级）与一个 INDEX.md；103.62 MB 的 report/log 原始产物压入 frozen 归档，保留 sha256 清单。当前是「103 MB 全不跟踪 + 753 条断链」，应改为「小回执跟踪 + 大日志归档 + 索引可查」。
3. **修复 753 条断链**：在清理 artifacts 前生成 index.json（run 目录 → report.json sha256 → 引用它的 evidence 文件），并把 doc 中指向已删目录的链接改为指向回执或归档；或让 verifier 在结束时把 report.json 复制到 doc/assets/<slice>/receipts/<run>.json。
4. **给 6 个漂移源建 rebase 流程**：verify_erp.py 现在必然失败。需要一个显式脚本：重算 189 源哈希 → 生成新 acceptance.json → 更新 EXPECTED_BASE_* / EXPECTED_PROOF_SHA256 / 计数 → 归档旧回执并在 evidence 记一笔「rebase 原因」。没有这个流程，最严格的验证链是一次性的。

### 5.2 P1（工程化，决定长期可维护性）

5. **CI**：GitHub Actions 用 elan 安装 lean-toolchain 指定版本，缓存 ~/.elan 与 formal/lean/.lake（key = lake-manifest.json 的 hash），跑 `lake build N2P3Continuous` + verify.py + 2 个关键 verify_*（erp、continuous）+ pytest formal/lean/scripts/test_lean_runner.py。缓存命中时增量构建只需分钟级；未命中时用 5 小时量级的 nightly job。
6. **工具链可移植**：删除脚本里的 D:/BCI 默认值，改为 --lake 必填或读 LEAN_LAKE 环境变量，缺省用 shutil.which("lake")；MATHLIB_CACHE_DIR 改为环境变量并在 README.md:26-40 记录；把 lean-toolchain 作为唯一版本源。
7. **生成物隔离**：Audit*.lean 写到 .lake/generated/（或 artifacts/<run>/），通过 LEAN_PATH 或 -R 引入，而不是覆盖源码树；若必须留在根目录，则加 GENERATED 头注释 + .gitignore 规则，并让 verifier 先删后写。
8. **孤儿治理**：为 27 个 Audit*.lean 与 55 个 negative 建 coverage manifest（file → 所属子目标 → 使用它的脚本/证据），CI 校验「每个文件至少被引用一次」；不满足就删除或补入口。
9. **定理到源码的映射表**：把 README.md:7-14 的散文表升级为 machine-readable JSON（theorem → Python symbol → file:line → 假设边界），由 verifier 校验 symbol 存在且哈希匹配，替代当前的纯文本声明。

### 5.3 P2（整洁度与成本）

10. **目录结构**：formal/lean/{spec,audit,negative,checks,scripts} 分层，root 只留入口模块；同步调整 lakefile.toml 的 lean_lib glob，避免破坏 N2P3 的 251 文件布局。
11. **升级 runbook**：写 mathlib bump 步骤（改 rev → lake update → 全量重建 → 重推 8 类常量 → 归档旧回执 → 更新 CI 缓存 key），并在 doc/engineering.zh.md 的 Lean 段落（:74-82）加链接。
12. **体积**：doc/assets/lean_* 用 Git LFS 或只留摘要；.lake 加入整机备份排除项；artifacts 目录数（792）设上限并自动归档最旧者。
13. **命名**：包名 n2p3_pilot → n2p3_formal；统一 <slice>_<yyyymmdd>_<seq> 命名，消除 lean_gaussian_20261016 与 _v2、lean_temporal_operator_20261016 与 _20261018 这类跨日期配对。

### 5.4 「formal/ 未入库」的严重级别

**P0**。理由：(a) 3.63 MB 证明源码 + 103.62 MB 证据 + 2890 条已验收定理完全无版本历史，单盘故障即全损；(b) 证据链已有 753/1528 条断链，且最严格的 verify_erp 因 6 个源漂移已不可重跑，说明「未入库」不是形式问题而是已经造成可复现性损失；(c) 修复成本低（一次 3.63 MB 提交 + 一条 .gitignore 已就绪），收益是整条证据链可恢复。相对地，「2.67 GB 缓存未入库」是正确决策，不构成风险。

## 附录 A：证据清单

审计中实际读取/计算的对象（均为当前工作树）：

| 证据 | 用途 |
|---|---|
| formal/lean/lakefile.toml:1-23 | 包名、defaultTargets、mathlib rev、4 个 lean_lib 与 -j1 |
| formal/lean/lean-toolchain:1、lake-manifest.json:1-100 | 工具链与 9 个依赖的 rev pin |
| formal/lean/.gitignore:1-5 | .lake/*.olean/*.ilean/__pycache__/artifacts 排除 |
| formal/lean/scripts/run_lean.py:20,176-193,255-292,344-407,411-449 | 双检查/独占 FIFO、环境指纹、缓存、硬编码 lake 与 MATHLIB_CACHE_DIR |
| formal/lean/scripts/verify.py:20-27,98-110,128-190 | 5 个源码哈希、公理白名单、反例拒绝判据、运行期间不变性 |
| formal/lean/scripts/verify_erp.py:32-54,111-176,185-209 | 冻结计数/哈希、base 回执校验、Audit 生成、工具字节比对 |
| formal/lean/scripts/check_python_bridge.py:1,120-133 | 「does not prove Python semantics」边界与 passed=True 字面量 |
| formal/lean/scripts/test_lean_runner.py:84-287 | 8 个运行器测试（不在 pytest testpaths） |
| formal/lean/README.md:7-14,22,26-34,42-45,51-52,131 | 规格映射、工具链路径、复现命令、trust=0 声明、定理数警告 |
| formal/lean/Audit.lean:4-18、N2P3Continuous.lean:1-5、N2P3Program.lean:1-7 | 默认目标与 pilot 公理导出 |
| formal/lean/.lake/verification-environment.json:4-9 | 嵌入 D:\BCI 绝对路径的环境缓存 |
| formal/lean/verification_plan.json（baseline/current_state_checkpoint/execution_policy/strict_completion_policy/proof_hole_and_trust_policy） | 189 源哈希、回执哈希、策略与暂停授权 |
| doc/assets/lean_gaussian_20261016/active_checkpoint.json | status=paused、accepted_theorems=2890、pause/cleanup/contract 哈希 |
| doc/assets/lean_temporal_operator_20261016/{acceptance.json,evidence_manifest.json}、frozen/lean_temporal_operator_20261018/manifest.json | 4 个被钉哈希实测匹配 |
| doc/assets/lean_pilot_20261016/acceptance.json | pilot 回执，sha256=688d0330... 与 plan 一致 |
| doc/evidence/README.md:5-19 | 15 个 lean_* 证据条目与定理数表述 |
| pyproject.toml:31-34 | testpaths=["tests"]，不含 formal |
| git status --porcelain -uall | 4402 行；formal 414、doc 3410、tests 184、frozen 146 |
| tmp/audit/scratch/p13_*.py | 本次只读统计脚本（sizes/lake/refs/frozen/status/hashes/drift/neg/counts/cp） |

复算命令（只读，PYTHONDONTWRITEBYTECODE=1）：

    python tmp/audit/scratch/p13_lake.py      # .lake 文件数/体积/olean 构成
    python tmp/audit/scratch/p13_refs.py      # 1528 条 artifacts 引用、753 条断链
    python tmp/audit/scratch/p13_hashes2.py   # 4 个回执 + 8 个证明哈希
    python tmp/audit/scratch/p13_drift.py     # 189 源中 6 个漂移
    python tmp/audit/scratch/p13_neg.py       # 93 negative 中 55 个无脚本引用

未执行（按只读约束）：lake build、lake update、任何 pip/uv 安装、pytest、git add/commit。因此「从零构建耗时」是依据 .olean 时间戳的估算（mathlib 约 5.5 小时），不是实测。
