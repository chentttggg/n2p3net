# V2 独立复核（red team）

复核对象：另一组审计代理的 10 条数学/统计/性能/安全结论。仓库 `D:\BCI\n2p3-net` 当前工作树（非 HEAD）。
方法：每条先用 1–3 个最小脚本/命令独立复现；只读源码与产物，未修改任何既有文件；仅写本报告与 `tmp/audit/scratch/` 下临时脚本。
环境：`.venv` Python 3.14.3 / numpy 2.5.2 / scipy 1.18.1 / mne 1.12.1；pin_memory 用 `.venv-gpu` torch 2.13.0+cu130。

## 总结表

| # | 判定 | 一句话 |
|---|---|---|
| 1 | 成立（口径需修正） | plug-in d² 确为 14–34× 上偏；但原文的 n 是“每类 n”，不是总 n；34.6× 未精确复现（10 seed 均值 33.7×）；交叉拟合不能消除偏差。 |
| 2 | 部分成立 | iid 假设与 AR(p) 冲突、iid MC 高估命中率均成立；但代码已明确声明“不是 universal bound”，且高估量取决于正相关强度。 |
| 3 | 成立 | n=2 percentile bootstrap 覆盖率约 50%（我们 50.47%，原文 49.3% 在 MC 误差内）；n=3/5 也只有 77%/82%。 |
| 4 | 部分成立 | ridge 支配 log-likelihood 成立且比原文更强；+3241.3 算术自洽，但“100× ridge 移动 ~2500 nats”与自述 N=128/rank=72/C=4 不自洽（det-only 应约 1031）。 |
| 5 | 成立（限定边缘） | padlen=2201@250Hz / 1166@128Hz、legacy 27–33 均复现；新旧差异只在段边缘显著，中段收敛。 |
| 6 | 部分成立 | -203.125ms 只在 128Hz 原生成立；250Hz 原生重采样后起点是 -200ms。终点也不是 1187.5ms（那是最后采样点），半开支撑到 1195.3/1198.4ms。 |
| 7 | 部分成立 | “H2D 从未异步”结论成立，但机制不是“整矩阵 pin 后 index_select”：常见 preload 路径根本没 pin；423MB/进程只在流式回退路径成立。 |
| 8 | 部分成立 | 每步 isfinite 扫描、每任务 compiler.reset 均成立；但默认 EEGNet 只有 12 个参数张量/1378 参数，不是 22；CPU +4.8% 未复现（实测 0.35–0.7%）。 |
| 9 | 成立 | 5s/每任务/每选择整份重写、indent=2+fsync 均成立；最大单文件 180,221,713 B = 171.87 MiB，35 tasks，最小约 36 次全量重写。 |
| 10 | 成立 | 3 处 weights_only=False、硬编码云主机/root/私钥名、2 处 StrictHostKeyChecking=no、bypassPermissions 仅靠全局 ignore、备份未忽略，均复核为真。 |

## 1. information_bound 的 plug-in d² 偏差

**原结论**：p=200/n=60 上偏 14.3×、p=400/n=50 上偏 34.6×（40 次重复，Σ=I，真 d²=0.5）；代码路径 `information_bound.py:87-178,515-518`。

**判定：成立（量级与代码路径），但样本口径和精确倍数需修正。**

**证据**
- 代码确实同批数据：`mahalanobis_snr` 用 `mu1=tgt.mean`、`mu0=non.mean`，池化残差 `pooled` 后 LedoitWolf 估 `Sigma`（`information_bound.py:129-163`）；`InformationBoundReport.compute` 在 `:515-518` 直接把该 d² 送入 bound。
- 独立复现（`.venv`，40 次重复，seed=123，Σ=I，δ=[sqrt(0.5),0,...]）：
  - p=200，**每类 n=60（总 120）**：mean d²=7.18，14.37×；10 个 seed 均值 7.32，14.63×。
  - p=400，**每类 n=50（总 100）**：mean d²=16.75，33.50×；10 个 seed 均值 16.86，33.73×（范围 33.2–34.4×）。
  - p=800，每类 n=100：mean 16.66，33.31×。
- 若把“n=60”理解为**总样本 60（30/30）**，同一函数给出 mean d²=14.32，即 **28.65×**；p=400 总样本 50（25/25）给出 **69.2×**。所以原文的 n 是“每类 n”，不是总 n；读者若按总 n 理解会低估偏差约 2×。
- 交叉拟合（Σ 与 δ 分开，各用一半数据）并未消除偏差：p=200/每类60 仍为 7.34（14.68×），p=400/每类50 仍为 17.01（34.02×）。原因是 δ 本身用有限样本估计，E||δ̂||² = d² + p/n，交叉拟合只去掉“同一批”双重使用，去不掉这一项。

**修正措辞**
“在同批数据上 plug-in 的 d² 在 p=200、每类 60 个 epoch 时上偏约 14×（若总 n=60 则约 29×）；p=400、每类 50 时约 33–34×。原文的 n 应明确写成 per-class；34.6× 是种子相关值，不是稳定值。仅交叉拟合不够，需要独立大样本估 δ、解析偏差修正 d²−p/n_eff，或直接拒绝 p/n 过大的报告。”

## 2. iid 闪光假设 vs AR(p)

**原结论**：`information_bound.py:39-49,296-306` 假设闪光 iid，与 `temporal_noise.py:1-13` 的 AR(p) 冲突；iid MC 会高估命中率。

**判定：部分成立。冲突与高估方向成立；但“该界”措辞已被代码本身否定，且高估量依赖正相关强度。**

**证据**
- `information_bound.py:39-49` 明确写：“It is not a universal information-theoretic upper bound: it inherits the equal-covariance, independent-flash, known-parameter model and the estimated d².” 所以把它称作“界”是原审计的措辞问题，不是代码当前声称。
- `:296-306` 的 MC 每 flash 抽 iid `N(0,sqrt(d2))`，再 `T=noise@A`。
- `temporal_noise.py:1-13` 的模型是分段 Burg AR(p)，残差时间相关；`doc/progress_report_20260929.zh.md:64` 记录 BI2014a 57,446 次 flash 在 1200ms 窗内平均还有 2.22 个后续 onset，说明 epoch 重叠、闪光间噪声不可能独立。
- 同模型下换成 AR(1) 相关闪光（Q=36,F=120，20,000 trials，iid MC 与代码一致：d²=0.5 时 0.663）：
  - d²=0.5：iid 0.663；ρ=0.3 → 0.460；ρ=0.5 → 0.340；ρ=0.8 → 0.211。
  - d²=0.2：iid 0.345；ρ=0.5 → 0.173。
  - 即 iid MC 高估 0.20–0.45（绝对命中率），相对可达 2×。
- 代码自带的 union bound 在 d²=0.5 时返回 `p_hit_lower_bound=0.0`，本身已无信息量；它同样是 iid 模型下的量，不能当作 AR 噪声的界。

**修正措辞**
“该量是‘iid 闪光 + 等协方差 + 已知参数’的模型估计，不是界。仓库的 AR(p) 噪声模型与 iid 闪光假设冲突；在正相关下 iid MC 会明显高估命中率（示例 0.663 → 0.34，ρ=0.5）。应改称 model estimate，并用 AR 着色/块 bootstrap 重算 MC 与区间；在报告里删除 ‘bound/最优’ 措辞。”

## 3. 小 n percentile bootstrap 覆盖率

**原结论**：`stats/hierarchical.py:47,62-71` 纯 percentile bootstrap，n=2 覆盖率 49.3%（1500×1000）；`stats/transfer.py:190-192` 在 len>1 就写区间。

**判定：成立。**

**证据**
- `hierarchical.py:62-71` 只做 `rng.integers` 重采样 + `np.quantile`，无 BCa/t/偏差修正；`:47` 只要求 n≥2。`transfer.py:190-192` 的条件是 `len(values) > 1`，n=2 即写 `paired_interval`。
- 独立复现（真均值 0，正态）：
  - 1500 reps × 1000 iters：覆盖率 **0.5047**，Wilson 95% [0.4794, 0.5299]。
  - 300 reps × 1000 iters：0.4833，[0.4274, 0.5397]。
  - 原文 49.3% 落在该区间内；n=2 的解析覆盖率就是 50%（bootstrap 分布以 1/4 概率取 x1、1/4 取 x2、1/2 取均值，2.5%/97.5% 分位即 min/max，覆盖概率 1−2×1/4=0.5）。
  - n=3：0.773；n=5：0.820（300×1000），说明不只是 n=2 的问题。
- 注：`paired_interval` 强制 iterations≥1000，所以“300×500”无法走公开 API；我用 300×1000 替代。

**修正措辞**
“n=2 的 percentile bootstrap 区间实际是 [min,max]，覆盖率约 50%（49.3% 是 MC 抽样值，与 50% 无显著差异），不是 95%。应在 n<5 时拒绝或改用 bootstrap-t/BCa；`transfer.py` 的 `len>1` 门槛应改为 n≥5 或返回 `insufficient_units`。”

## 4. exact_covariance 的 ridge 与 byte_budget

**原结论**：`log_likelihood_correction` 由 `ridge=max_eig*1e-12` 支配（delta_ll≈+3241，ridge 变化 100 倍移动 ~2500 nats）；`:185-188` byte_budget 说明算错（N≤2048 而非 2900）。

**判定：部分成立。ridge 支配成立且可更强；+3241 算术自洽；但“2500 nats/100×”与自述配置不自洽，无法按原数复现。byte_budget 成立。**

**证据**
- 代码：`exact_covariance.py:243` `ridge=max_eig*1e-12`；`:396-440` 的 delta_ll = −0.5(quad_exact−quad_approx) − C(logdet_exact−logdet_approx)。
- 独立实验（N=307，C=3，AR(0.7)，2–15Hz@64Hz，padding=50）：
  - 残差取 `r=A·N(0,I)`（在算子值域内）：delta_ll=**+2104**，其中 det 项 **+2732**、quad 项 **−628**；ridge 1e-12→1e-10 时 +2104→+1997（det-only 变化 −109）。
  - 残差取普通白噪声（含零空间分量）：delta_ll=**−2.16e12**；ridge 1e-12→1e-10 时变为 −4.35e10（移动 2.1e12）。原因是零空间方向被 1/ridge 放大，quad 项主导。
- 原审计自述 N=128、rank=72、ridge=1.63e-10、C=4、r=A·N(0,I)：lp=−784.1、la=44.8 → det 项 = −4×(−828.9)=+3315.6，quad=−74.5，合计 +3241.1；算术自洽。
- 但同一配置下 100× ridge 的 det-only 位移应为 (N−rank)·C·ln100 = 56×4×4.605 ≈ **1031 nats**；要得到 2500 需要约 136 个 ridge 主导方向，超过 N=128。2500 只能来自近零特征值上的 quad 项或另一套 N/rank/C，不能作为该配置的通用数字。
- byte_budget：`:195` 条件是 `2*8*N² > 64MiB`；64MiB 下 N=2048 恰好等于预算（允许），N=2049 拒绝。docstring `:187` 的 ~2900 是“单个 N×N 矩阵”的 `sqrt(64MiB/8)=2896`，漏掉了“两个矩阵”的系数 2。

**修正措辞**
“delta_ll 的 det 项含 (N−rank)·C·log(1/ridge)，其数值由任意 ridge 决定，不是可比较的边际似然；在 N=128/rank=72/C=4 下，仅 det 项的 100× ridge 位移约 1031 nats，原报告的 2500 需要额外近零 quad 贡献，不能当作该配置的稳定量。对含零空间分量的残差，delta_ll 甚至可达 −1e12。byte_budget 应改为 N≤2048，或把默认预算/说明改成两个矩阵的公式。”

## 5. 零相位滤波 padlen

**原结论**：`mne_reflect_limited` padlen=2201@250Hz（8.8s）/1166@128Hz，legacy 27–33 样本；新旧缓存边缘不可比；`preprocess.py:521-525`。

**判定：成立（数值与路径），但“边缘不可比”应限定为边缘；2201 是请求上限，实际 pad 会被段长−1 截断。**

**证据**
- 用 `design_continuous_sos(rate,l_freq=0.1,h_freq=30,order=4,notch=(50,))` + `construct_iir_filter(dict(sos=sos))`：
  - 250Hz：sections=5，padlen=**2201**（8.804s）；128Hz：sections=5，padlen=**1166**（9.109s）。
  - legacy 公式 `3*(2*len(sos)+1−min(zeros)))`：5 sections（含 notch）=**33**；4 sections（无 notch）=**27**，故“27–33”成立。
- `preprocess.py:521-525` 确实调用 `filter_bidirectional_segment(..., padding_limit=zero_params["padlen"])`；`filter_bidirectional_segment` 在 `:439` 做 `pad=min(padding_limit, len−1)`。因此 179 点 epoch/短段的有效 pad 是 178，不是 2201；2201 只在长连续段生效。
- 边界差异实测（250Hz，60s 信号，new=reflect_limited 2201 vs old=sosfiltfilt padlen=33）：first 0.5s max|Δ|=0.379、last 0.5s 0.681；middle 10s max|Δ|=0.0018（信号 RMS≈0.74）。所以新旧缓存在边缘不可比，中段基本收敛；“整份不可比”会夸大。

**修正措辞**
“请求 padlen 为 2201/1166，legacy 为 27–33；实际 pad 受 min(padlen, segment_len−1) 截断。新旧策略在连续段两端产生 O(0.1–1) 的差异，中段收敛到 O(1e-3)；应把结论写成‘边缘样本不可直接混用’，并记录有效 pad 而非只记请求值。”

## 6. epoch 网格

**原结论**：128Hz 实际 `times[0]=-0.203125`、裁 179 点后 `[-203.1,1187.5)`ms vs 声明 `[-200,1200)`；`preprocess.py:926-939,626-627`。

**判定：部分成立。起点位移只在 128Hz 原生成立；终点 1187.5ms 是最后采样点，不是半开区间终点。**

**证据**
- 128Hz 原生：MNE `Epochs(tmin=-0.2,tmax=1.2)` 生成 **181** 点，`times[0]=-0.203125`、`times[-1]=1.203125`；裁 179 点后最后采样点 = −0.203125+178/128 = **1.1875s**，半开支撑 = `[-0.203125,1.1953125)`（1398.4375ms）。
- 250Hz 原生（BNCI 常见）：MNE 在 250Hz 上取到 `[-0.2,1.2]` 共 351 点（起点正好 −0.2）；`mne.filter.resample` 到 128Hz 得 180 点，再裁 179 点 → 半开 `[-0.200,1.1984375)`，最后采样点 1.190625s。
- 因此“128Hz 实际 −0.203125”正确；“250Hz 原生也是 −0.203125”不成立。两种路径的窗口长度都是 179/128=1398.4375ms，均比声明的 1400ms 短 1.5625ms；声明的 1200ms 端点未被保留。
- `preprocess.py:926-939` 用 `epochs.times[0]/[-1]` 算 offset；`:626-627` 只做 `data[..., :n_times]` 裁剪，没有对 tmin 做样本对齐补偿；`contract.py:71-72,88-90` 的 `n_times=floor(1400*128/1000+1e-9)=179`。

**修正措辞**
“128Hz 原生时实际起点 −203.125ms、最后采样点 1187.5ms、半开终点 1195.3125ms；250Hz 原生重采样后为 [−200,1198.4375)ms。声明 [-200,1200) 与两条实际网格都不一致；应报告实际 `source_epoch_start_samples/end_samples`，并把‘1187.5ms’标成最后采样点而非区间终点。”

## 7. runtime pin_memory / to_device

**原结论**：`runtime.py:508-517` 整矩阵 pin_memory 在随机 index_select 后不再是 pinned，to_device 从未异步；每个进程 ~423MB 页锁定。

**判定：部分成立。“H2D 从未异步”在当前调用路径成立；但机制描述只覆盖流式回退，且 423MB/进程不是普遍值。**

**证据**
- `x.pin_memory().is_pinned()` → True；`p[:10].is_pinned()`（view）→ True；`p.index_select(0, idx).is_pinned()` 和 `p[torch.arange(10)].is_pinned()` → **False**。所以随机 gather 确实丢失 pinned；`to_device` 的 `non_blocking = ... and _is_pinned_tensor(tensor)`（`:1237-1239`）对这些 batch 为 False。
- 但 `MatrixBatchSource.__init__` 的顺序是：先 `to_device(self.cpu_X)`（`:477`），后 `pin_memory`（`:508-517`）且仅在 `not preloaded` 时。调用方传入的是 `torch.from_numpy(...)`（`deep.py:1275,1297,1937`），默认未 pinned，所以 **preload 路径的整矩阵 H2D 也是同步的**，且根本没有执行 pin。若 preload 失败进入流式路径，才 pin 整矩阵，然后 batch `index_select` 又丢失 pinned。
- `release_cpu_after_preload=True`（source fit，`deep.py:1278`）在 preload 成功后释放 CPU 矩阵，不会留下 423MB 页锁定；只有流式回退才可能长期持有整份 pinned CPU 矩阵。
- “423MB”：代码没有这个常量；我检查的 `domain_transfer_cloud_20260916/inputs/a08...npz` 中 X 为 (162757,6,179) float32 = **666.8 MiB**（npz 文件本身 443MB，含元数据）。若该缓存流式训练，pin 的是 X 的 667MiB（或筛选后的子集），不是 423MB。需要给出“423MB”对应的具体矩阵/行子集。

**修正措辞**
“当前所有 H2D 都是同步：preload 路径传的是未 pinned 的 `from_numpy` 张量；流式回退路径虽 pin 整矩阵，但随机 batch 的 index_select 结果不再 pinned。因而没有获得异步拷贝；代价是 preload 路径一次同步全量拷贝，或回退路径整份页锁定 + 每批同步。‘每进程 423MB’仅在流式回退且矩阵为 423MB 时成立，本仓库检查到的 X 是 667MiB。”

## 8. 梯度 isfinite 扫描 / compiler.reset

**原结论**：`deep.py:1318-1321` 每步梯度 isfinite 扫描（22 个参数张量、CPU +4.8%）；每任务 `torch.compiler.reset()+empty_cache`（`source_execution.py:78-85`）。

**判定：部分成立。调用频率与 reset 成立；“22 个参数张量”不成立；CPU +4.8% 未复现。**

**证据**
- `deep.py:1318-1321` 在每个 `train_step` 内对 `loss` 和每个 `parameter.grad` 做 `torch.isfinite(...).all()`，随后 `:1364` 做一次 `bool(...)` 同步；频率是每 optimizer update，成立。
- 默认模型 `eegnet`（`deep.py:376`）：`EEGNet(n_chans=7,n_outputs=2,n_times=179)` 有 **12 个 parameter tensors / 1,378 参数**（state_dict 21 项，含 BN buffer）；`inception` 38/14,890；`conformer` 110/202,306。没有任何一个是 22 个 parameter tensors。若原审计数的是 state_dict keys，EEGNet 是 21，也不是 22。
- CPU 实测：12 个梯度的扫描 **181.2 µs/step**；batch=512 的 forward+backward+Adam 约 **51.7 ms** → 0.35%；默认 batch=256 约 26ms → 0.7%。CPU +4.8% 需要 step≈3.8ms，与默认配置不符。GPU 上 step 更短，相对开销可能到几个百分点，但需要实测，且前提仍是 12 个而非 22 个张量。
- `source_execution.py:78-85`：`torch.compiler.reset()` + `release_device_memory`（gc + empty_cache + ipc_collect）确实在每次 source trajectory fit 的 `finally` 调用（`run_source_training_chain.py:152`，另有 376/398/1032/1346）；默认 `compile_mode='reduce-overhead'`（`runtime.py:32`），reset 会使下一个任务重新编译。这个开销比 isfinite 扫描更值得关注。

**修正措辞**
“每步确实扫描梯度，但默认 EEGNet 是 12 个参数张量/1378 参数，不是 22；CPU 实测约 0.35–0.7%，不是 4.8%。每任务 `torch.compiler.reset()` 成立，且因默认 reduce-overhead 会导致下一任务重编译；应把性能结论改为‘每任务重编译开销’，并给出实际模型的张量数/step 时间。”

## 9. 报告 JSON 写放大

**原结论**：每 5s 写 + 每任务/选择后整份重写、indent=2+fsync（`run_source_training_chain.py:511-523,783,791,912,941`；`execution.py:42-53`）；已有 171.9MB 单文件。

**判定：成立。**

**证据**
- `_save_chain_report` 在 `:423-434` 每次写 `source_chain_record.json`（整份 report）+ `worker_progress.json`；`:511-523` 的 `wait(pending, timeout=5)` 循环每 5s 调一次，无论是否有新完成任务；`:783/791` 每任务、`:912` 每选择、`:941` 失败后都整份重写。
- `execution.py:42-53` 的 `atomic_write_json`：`json.dump(..., indent=2)` + `flush` + `os.fsync`，临时文件再 replace。确认。
- 现存最大文件 `experiments/runs/unified_cloud_20260916/analysis/matrix_sources_source_chain_record.json`：180,221,713 bytes = **171.87 MiB**；JSON 里 `task_count=35`、`tasks` 序列化 118.5 MiB、`selection=0`、`outer_evaluation_jobs=36`。
- 仅按“每任务后重写”计，最少 36 次全量写；若平均报告为最终大小的一半，累计 ≥0.5×171.9MiB×36 ≈ **3.1 GiB**（5s tick 还会增加）。这是任务数 T 的 O(T²) 放大；本 run 的 T=35，相对最终文件约 18× 下界。

**修正措辞**
“写频率与单文件大小成立：5s tick + 每任务/选择整份 `indent=2`+fsync 重写；该 run 35 个 task 至少 36 次全量写，累计下界约 3.1GiB，最终单文件 171.87MiB。应改为 append-only/分片 task 记录 + 原子替换小索引，或只在状态变化时写。”

## 10. 安全

**原结论**：`torch.load weights_only=False` 三处（`transfer/checkpoint.py:56`、`result_reuse.py:116`、`baselines/deep.py:587`）；`open_dashboard.ps1:2-8` 硬编码云主机/root/私钥名，`:46,98 StrictHostKeyChecking=no`；`.claude/settings.local.json defaultMode=bypassPermissions` 且仅靠全局 ignore、备份文件未忽略。

**判定：成立。**

**证据**
- `grep -rn "weights_only\s*=\s*False" src` 恰好 3 处：`src/baselines/deep.py:587`、`src/transfer/checkpoint.py:56`、`src/transfer/result_reuse.py:116`；另有 `calibrated.py:169,1222` 用 `weights_only=True`。
- `open_dashboard.ps1:2-8` 默认值硬编码 `connect.weste.seetacloud.com:35832`、`root`、`/root/autodl-tmp/n2p3-net`、`id_ed25519_n2p3_cloud_20260827`；`:46` 和 `:98` 都传 `-o StrictHostKeyChecking=no`。私钥内容未读取/输出；风险在主机验证被关闭 + root 远程执行。
- `.claude/settings.local.json` 内容：`{"permissions":{"allow":["Agent"],"defaultMode":"bypassPermissions"}}`。`git check-ignore -v` 显示它仅被 `C:\Users\chent/.config/git/ignore` 的 `**/.claude/settings.local.json` 忽略；仓库 `.gitignore` 无 `.claude` 规则。
- `.claude/settings.local.json.backup-20261011-232131`（62 B）未被任何 ignore 规则匹配；`git status --porcelain --ignored .claude` 输出 `?? .claude/`（因为 backup 是未跟踪且未忽略的文件）。backup 内容只有 allow Agent、无 `defaultMode`，但一旦 `git add .claude` 会被提交；主文件在没有全局 ignore 的机器上也会变成未跟踪可提交文件。

**修正措辞**
“三项均成立。应：把三处 `weights_only=False` 改为 `weights_only=True` + 显式安全 globals，或对 payload 做 schema/类型白名单；`open_dashboard.ps1` 移除默认主机/root/密钥名，改用 `known_hosts` 校验，删除 `StrictHostKeyChecking=no`；在仓库 `.gitignore` 加 `.claude/` 与 `*.backup-*`，并把 `defaultMode` 改为 `default`/`plan`，需要 bypass 时显式命令行开启。”

## 复核边界

- 未读真实校准/训练数据做端到端数值影响评估；第 1、2、4 条的倍数均为合成/受控实验，真实 T×C、epoch 数、AR 系数会改变具体数值。
- 第 4 条原报告的 +3241.3 无法在仓库中找到对应 fixture（无 N=128/rank=72 的构造），只验证了算术与代码路径；故标为“部分成立/具体数字无法精确复现”。
- 第 7 条的 423MB 未找到代码常量；给出的是实际 npz 中 X 的 667MiB，若审计另有 423MB 矩阵需补充路径。
- 未运行完整测试套件/训练；未修改/删除任何既有文件，未执行 git 写命令，未安装依赖，未创建子代理。
