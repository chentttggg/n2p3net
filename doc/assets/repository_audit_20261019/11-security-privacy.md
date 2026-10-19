# 11 — 安全与隐私审计（安全审计分区）

- 仓库：`D:\BCI\n2p3-net`（以当前工作树为准）
- 角色：只读代码审计子代理。**未修改任何既有文件**，未执行 git 写命令，未安装依赖，未启动训练，未创建子代理。
- 时间：2026-10-19（Asia/Shanghai）。级别：**P0** 立即 / **P1** 本周 / **P2** 长期治理。
- 证据格式 `路径:行号`。**不复制任何疑似密钥的值**，只报路径、类型、暴露面。

## 摘要

| # | 级别 | 问题 | 证据 |
|---|---|---|---|
| 1 | **P0** | dashboard 零鉴权 + 两次 `StrictHostKeyChecking=no` → 中间人可注入同源 JS，杀训练/读结果 | `dashboard_server.py:504,540`；`open_dashboard.ps1:46,98` |
| 2 | **P1** | 云主机 host/port/root/私钥名完整入库 | `open_dashboard.ps1:2-8` |
| 3 | **P1** | 已提交附件带出 Windows 账户名与主机名 | `doc/assets/code_quality_repair_20260908/validation.json:29`；`doc/assets/technical_audit_20260906/*.xml`（8 个 `hostname` 属性） |
| 4 | **P1** | `doc/assets` 3,352 个未跟踪产物（172 MB，含 .bdf/.pt/.log/.pyc），`git add -A` 即入库 | `.gitignore:37` 仅排除 `*.npz` |
| 5 | **P1** | 两处 `torch.load(weights_only=False)` 位于跨主机的 checkpoint / 结果复用路径 | `src/transfer/checkpoint.py:56`；`src/transfer/result_reuse.py:116` |
| 6 | **P1** | 无根 LICENSE；BSD 归属文件未跟踪却被声明为 package-data | `src/baselines/structured_erp.LICENSE`（??）；`pyproject.toml:29` |
| 7 | **P1** | `.claude` 的 `bypassPermissions` 只靠用户全局 ignore，仓库无规则；backup 可被 `git add` | `.gitignore` 无 `.claude` |
| 8 | P2 | 59 个论文 PDF（200.7 MB）版权无声明、`.git` 8.2 GB、无 CI/预提交、数据集许可全缺 | `Paper/`；`git rev-list --objects --all` |

**与 Lead 初筛的差异**：①"全仓未见用户名"不成立（§2.1：附件里有账户名与主机名）；②`np.load` 全为 `allow_pickle=False`、无 `eval/exec/shell=True` —— 初筛正确；③`src/baselines/structured_erp.py`、`temporal_transfer.py`、`structured_erp.LICENSE` **未跟踪**，HEAD 新鲜 clone 无法导入。

## 0. 范围与方法

审计对象：当前工作树 + Git 对象库（HEAD 与全部 ref）。**不审计** `.venv*/`、`mne_data/`、`experiments/runs/`、`doc/assets` 内第三方库代码、`tmp/cloud_ssh`。

1. 全部 570 个跟踪文件的正则扫描（私钥块 / AKIA / gh*_ / xox* / sk- / ssh 公钥 / 赋值型 secret / JWT / Bearer）——10 类全部零命中（§1.1）。
2. 主机名、`C:\Users`、`.ssh` 引用扫描（区分大小写与不区分大小写各跑一遍，后者才发现 §2.1）。
3. 历史：`git log --all --diff-filter=A` 扫敏感文件名；`git rev-list --objects --all` + `git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)'` 求最大 blob。
4. 危险 sink：`torch.load` / `np.load` / `pickle` / `yaml.load` / `eval` / `exec` / `shell=True` / `subprocess`。
5. 逐行复核 `experiments/dashboard_server.py`（612 行全文）、`open_dashboard.ps1`（146 行全文）、`src/transfer/checkpoint.py`、`src/transfer/result_reuse.py`、`watch_resources.py`、`run_with_dashboard.py`。
6. 许可：`git ls-files` 许可类文件、`pyproject.toml` package-data、`doc/` license 词表。

限制：未解包任何 `.pt`/`.bdf`/`.tar.gz` 做取证；未读取任何密钥值；未审计 `experiments/runs/` 内部。

## 1. 凭据与主机信息暴露

### 1.1 结论：仓库内**没有**凭据值，但有完整的云主机入口描述

| 模式 | 命中 |
|---|---|
| `-----BEGIN * PRIVATE KEY-----` | 0 |
| `AKIA[0-9A-Z]{16}` | 0 |
| `gh[pousr]_` / `xox[abp]-` / `sk-` | 0 |
| `ssh-(rsa\|ed25519) AAAA…` | 0 |
| `(api_key\|secret\|password\|token)\s*[:=]\s*"…"` | 0 |
| JWT `eyJ…` / `Authorization: Bearer` | 0 |

- `.gitignore:14-16` 覆盖 `.env`、`.env.*`（保留 `.env.example`）；工作树内**不存在**任何 `.env` 文件。
- 历史全 ref 的"新增文件"集合中**从未出现** `.env`/`.pem`/`.key`/`id_rsa`/`id_ed25519`/`credential`（唯一疑似命中 `src/models/tokenizer.py` 属词形误报）。
- 工作树内密钥类文件名扫描：仅命中 `.venv*/Lib/site-packages/{certifi,botocore}/cacert.pem`（CA 证书，非私钥）。
- `configs/*.json`（9 个文件）无 host/user/token/password 字段。
- `tmp/cloud_ssh` 顶层确认为 `pip --target` 安装（`paramiko`/`cryptography`/`invoke`/`bcrypt`/`cffi`/`nacl` + `.lock` + `_cffi_backend.cp314-win_amd64.pyd`），**无密钥文件**——Lead 判断成立。

### 1.2 P1 — `open_dashboard.ps1` 把云主机入口完整写进版本库

| 证据 | 内容类型 |
|---|---|
| `open_dashboard.ps1:2` | SSH 主机名（公网可达域名） |
| `open_dashboard.ps1:3` | 非标准 SSH 端口 35832 |
| `open_dashboard.ps1:4` | 登录用户 `root`（特权账户） |
| `open_dashboard.ps1:5` | 远端路径 `/root/autodl-tmp/n2p3-net` |
| `open_dashboard.ps1:8` | 私钥**文件名** `~/.ssh/id_ed25519_n2p3_cloud_20260827` |

全仓仅此一处出现主机名（`git grep -I -E "seetacloud|connect\.west|root@"` → 1 命中），说明其余文档已脱敏为 `autodl` 泛称。`autodl` 仍出现在 20+ 个 `doc/evidence/*.md` 与 `frozen/*/manifest.json`（如 `doc/engineering.zh.md:513`、`frozen/brainsync_s1_cloud_20260908/manifest.json:26`），属平台类型+路径前缀级别，单独不构成入口。

**为什么是 P1**：没有密钥值泄漏，但给出了定位三要素（公网 host、端口、特权用户名），使攻击面从"扫描 autodl 网段"收敛到"直接撞 35832 的 root"；私钥文件名带日期后缀，暴露了密钥轮换命名约定与本机 `~/.ssh` 的枚举线索；脚本以 root 登录并在远端 `nohup` 起服务（`:43`），落点是云主机 root + 全部训练数据。主机名最早随提交 `a4a9a1f`（dashboard harness）入库，**当前 HEAD 仍带**，删文件不足以消除历史。

**建议**：host/port/user/RemoteRoot/IdentityFile 全部参数化为 `$null` + 从 `~/.ssh/config` 的 `Host n2p3-cloud` 别名或环境变量读取；仓库内只留 `ssh n2p3-cloud`。若曾推送过远端，按"入口已公开"加固（改端口、禁 root 直登、fail2ban）。

### 1.3 P2 — 历史大 blob 使 `.git` 达 8.2 GB

- `.git` 实测 **8,236 MB**。历史最大 blob：`frozen/source_feature_transfer_c7bf860.source-only.tar.gz` **370.7 MB**、`frozen/source_feature_transfer_c5d670d.source-only.tar.gz` **185.3 MB**；其后是 19 个 `Paper/*.pdf`（最大 19 MB）与两版 `tmp/P1_STATIONARY_PROGRAM_STAGE06/plan.json`（13.6 MB）。
- `.gitignore:28-32` 已补 `/frozen/**/*.zip|gz|tar|7z|bundle`（并在 `:26` 忽略 `/frozen/local_history/`），但只对新增生效，556 MB 的 tarball 永久留在对象库。
- 本地 `frozen/local_history/` 另有 `tmp_history.zip` 1,342 MB、`tmp_20260906.zip` 1,249 MB（已忽略，未入库）。
- 影响：`git clone` 需拉 8 GB；若转公开，等于把两个完整源码快照 tar 一并分发。**未解包核对清单**，建议另派只读任务用 `tar -tzf` 检查是否含未跟踪的 `runs/`、`doc/assets` 私有产物。


## 2. 隐私与合规

### 2.1 结论：无真实被试 PII，但**机器身份**已入库（修正 Lead 初筛）

- 全部跟踪文本中**没有**姓名、邮箱、电话、出生日期、设备序列号（`[A-Za-z0-9._%+-]+@域名` 在 `doc/*.md`/`src`/`experiments`/`configs` 零命中）。被试只以公开数据集的伪名编号出现（`BI2014a`/`BNCI2014-008` 的 subject 序号、`p001` 类 participant key）。
- **但三处已提交附件带出机器身份**：

| 证据 | 泄漏类型 | 数量 |
|---|---|---|
| `doc/assets/code_quality_repair_20260908/validation.json:29` | Windows 账户名 + 完整 `C:\Users\<账户>\AppData\Local\Temp\pytest-of-<账户>\…` 路径（来自 `WinError 1314` 报错文本，含 symlink 测试临时目录） | 该文件 4 行含此路径形态 |
| `doc/assets/technical_audit_20260906/*.xml` | pytest JUnit 的 `hostname="<主机名>"` 属性，即本机主机名 | 8 个文件 |
| `…/backend_tests_initial_windows_reader_failure.xml`、`…/newest_producer_probe.xml` | 同时含 `C:\Users\<账户>` 路径 | 2 个文件 |

> 说明：本报告不复制账户名与主机名的具体值；值在对应文件的上述行内可直接查到。检测时必须用**不区分大小写**的 grep（`git grep -I -i "c:[\\/]+users"`），区分大小写版本会漏掉这三处。

**为什么是 P1**：主机名 + 账户名是横向定位的强线索（与 §1.2 的云主机入口组合，可确认"谁在本机开发、连哪台机器"）；`pytest-of-<账户>` 还泄漏了测试运行次数（`pytest-509`）。不含密钥、不含被试身份，故不升 P0。**建议**：提交前把路径替换为 `<tmp>`、删除 JUnit `hostname` 属性，并在 `doc/engineering.zh.md` 的证据附件小节写入硬规则。

### 2.2 P1 — `doc/assets` 有 3,352 个未跟踪产物（172 MB），含原始 EEG 格式与权重

- `git ls-files -- doc/assets` 只跟踪 **56** 个文件；`git status -uall -- doc/assets` 报 **3,352** 个未跟踪文件。`.gitignore` 只排除 `/doc/assets/**/*.npz`（`:37`）。
- 类型分布：`.json` 1,625、`.py` 749、`.zip` 552、`.log` 150、`.npz` 77、`.png` 46、`.xml` 39、`.txt` 38、**`.pyc` 29**、`.pt` 16、`.pdf` 15、`.csv` 7、**`.bdf` 1**、`.docx` 1。
- 高风险个案：
  - `doc/assets/system_implementation_20261004/review_spec/short_record/person_eeg.bdf`（59,524 B）——BDF 是真实 EEG 记录格式。文件极小、目录名 `review_spec/short_record`，**推测为合成夹具**，但仓库内无任何"合成/真实"标记，且不在忽略规则内。
  - `doc/assets/lean_state_20261016/source_probe_01/*.pt`（11,230 B ×2）等 16 个 `.pt`——pickle 格式权重，与 §3.3 的 `weights_only=False` 叠加即为可执行载荷载体。
  - 29 个 `.pyc`——字节码与解释器版本绑定，属纯噪声，且可能含绝对路径常量。
- **触发条件极普通**：`git add -A` / `git add doc` 会把这些（除 `*.npz`）全部暂存。这与 `README.md:101-103` 的自述（"GitHub contains source, tests, guidance and evidence manifests / A clone does not include datasets…"）直接矛盾。
- **建议**：`doc/assets` 改白名单入库（只放证据清单 `*.json`/`*.md`/`*.png`），显式忽略 `.bdf/.pt/.pyc/.log/.zip`。

### 2.3 P1 — 59 个第三方论文 PDF 入库（200.7 MB），版权无声明

- `git ls-files -- Paper` → 59 个 PDF；HEAD 中 `Paper/` 合计 **200,735,378 B（191.4 MiB）**。
- 既有 arXiv 预印本，也有**出版商正式版**：`Paper/CrossScale_Transformer_Domain_Rectified_RSVP_TNSRE_2024.pdf`（TNSRE）、`Paper/Improving_P300_Morphology_Single_Trial_Latency_Realignment_JNE_2026.pdf`（JNE）、`Paper/MS_EEGNet_P300_Frontiers_2021.pdf`（Frontiers）、`Paper/Calibration_Free_P300_2022_Accepted.pdf`。
- 仓库**无任何**再分发依据说明（无 `Paper/README.md`、无 license 字段）。IEEE/Frontiers 的 AAM 条款各不相同，默认不允许把 publisher PDF 放进公开仓库。
- 历史里另有小写 `paper/`（`paper/autoreject_2017.pdf`、`paper/prep_pipeline_2015.pdf`）——在大小写不敏感文件系统上是重命名，但 Git 对象里是两套路径。
- **建议**：私有可保留但补声明；公开前把出版商版移出 Git（`git rm --cached` + 忽略），只留 `Paper/index.md` 与 DOI/arXiv 链接。

### 2.4 P1 — 根 `LICENSE` 缺失，且 `structured_erp` 的 BSD 归属文件**未被跟踪**

- `git ls-files | Select-String 'LICEN|COPYING|NOTICE|CITATION'` → **空**：无根 LICENSE、无 CITATION.cff、无 NOTICE。默认"保留所有权利"，与"公开预训练 + 复用 BSD 代码 + 引用第三方数据集"的定位冲突。
- `src/baselines/structured_erp.LICENSE`（1,635 B，BSD-3-Clause 变体，`Copyright (c) 2022 Jan Sosulski`）**存在于磁盘但不在 Git**（`git cat-file -e HEAD:src/baselines/structured_erp.LICENSE` → 128）。同目录 `structured_erp.py`、`temporal_transfer.py` 同样 `??`。
- 更严重：`pyproject.toml:28-29` 已声明 `[tool.setuptools.package-data] baselines = ["structured_erp.LICENSE"]`——构建配置**依赖一个不在版本库里的文件**；换机 clone 后 sdist/wheel 缺该许可证，直接违反 BSD"源码再分发须保留版权声明"。
- 归属声明本身合格：`src/baselines/structured_erp.py:3-7` 写明"Sosulski et al. (JNE 2022)"、"adapted from jsosulski/toeplitzlda (BSD)"、"see structured_erp.LICENSE"，`ERP_FAMILIES` 也区分了改写与自研。问题只在**文件没入库**。
- 附带（超出本分区但影响可复现）：`src/transfer/calibrated.py:23`、`src/transfer/calibration_candidates.py:15`、`src/transfer/recipes.py:12` 都 `from baselines.structured_erp import …`，而该模块未跟踪 → **HEAD 的新鲜 clone 无法导入**，建议与相应分区交叉确认。

### 2.5 P2 — 数据集许可完全未声明；成人/设备确认是"文档声明"而非"代码门禁"

- **许可**：除 `doc/assets` 两份审计 JSON 外，全仓无数据集许可声明。`git grep -i "licen[cs]e|许可|CC-BY|非商业" -- doc README.md src experiments configs` 仅命中：
  - `doc/assets/cross_dataset_20260906/design_audit.json:28,203` —— 这是**被审计第三方数据集自带** `dataset_description.json` 的 `License: CC-BY-4.0`，不是本项目声明。
  - `doc/assets/dataset_expansion_20260908/acquisition_manifest.json:467` —— 记录 Chailloux2020/OpenNeuro `ds003190` 的 LICENSE 已下载。
  - `doc/evidence/biot_applicability_assessment_20260904.zh.md:185` —— **自认未核实**："SAMBA 的 checkpoint 是否公开、许可是否允许商用（未核实）"。
- 缺口具体到四个源：**BI2014a**（Brain Invaders）、**BNCI2014-008**（ALS 患者）、**GTN / MNE-P3-data**（三导儿童，248 目录）、**Mainsah**——均无"许可类型 / 是否允许再分发派生特征 / 是否允许商用"记录。真正的问题不是"用了数据"，而是**派生物（epoch 缓存、统计量、冻结快照）能否随仓库分发**没有依据；BNCI2014-008 是患者数据，条款通常更严。
- **成人/设备确认**：`configs/brainsync_collection_plan.json:44` 有 `"healthy_adult_scope_requires_age_evidence": true`，但**没有任何代码读取该文件**（`git grep collection_plan -- src experiments tests` 只命中它自身的 schema 行）→ 惰性配置，不构成门禁。真正的门禁只在文字：`AGENTS.md:42`、`doc/README.md:29`，以及 `doc/evidence/*` 30+ 处"不是成人确认"的免责句。文字执行得很好（`doc/evidence/README.md:59,62,65,76,77,92,99,107` 每行都带"非成人确认"），但**没有代码层断言**阻止用开发结果冒充成人结论。
- 被试身份/同意：`doc/workstation.zh.md:49` 说明采集端是独立本地项目、不在本仓库 → 知情同意与身份映射治理在仓库外。全仓无 `consent`/`IRB` 相关代码或文档（grep 零命中），发布说明应写清"本仓库不含任何采集端同意流程"。


## 3. 执行安全

### 3.1 P0 — `dashboard_server.py` 零鉴权，与 `StrictHostKeyChecking=no` 组成可利用链

先确认"没问题"的部分（Lead 初筛属实）：
- **路径穿越未打通**。`_run_dir`（`:24-43`）三重校验：`RUN_NAME.fullmatch`（`:20`）→ `(root/relative).resolve()` → `candidate == root or root not in candidate.parents` → `is_dir()`。注意 `RUN_NAME` 的字符类 `[A-Za-z0-9_.-]+` **允许 `..` 作为整段**，真正拦住的是 `:41` 的包含校验（`run=".."` → candidate == runs_root 被拒；`run="../.."` → 逃出但被 `root not in parents` 拒）。`translate_path`（`:326-338`）同样 `resolve()` + 包含校验，未解析的 `..` 由基类 `normpath` 丢弃。
- **Content-Length 解析正确**：`_read_payload`（`:566-573`）限定 `1 <= length <= 4096`、按 length 精确读、`isinstance(dict)` 校验；非法长度抛 `ValueError` 且被四处 `except` 覆盖。
- **监控子进程无命令注入**：`_claim_monitor`（`:425-432`）argv 列表 + `start_new_session=True` + 三路 `DEVNULL`，不经 shell。
- 默认 `--bind 127.0.0.1`（`:579`）；kill 前校验 `_process_belongs_to_run` 且拒绝自杀（`:550-553`）。

**问题**：
1. **零鉴权 + 零 Origin 校验**。整个 handler 无 token/session/Basic auth；`/api/kill` 唯一门槛是自定义头 `X-Dashboard-Confirm: yes`（`:540`），任何能发 HTTP 的本地进程均可伪造。
2. **CSRF 防护是"偶然"生效**。该头触发浏览器 CORS 预检，而 handler **没有 `do_OPTIONS`**（全文无此方法，基类亦无）→ 预检 501 → 浏览器放弃。**一旦为跨域便利补上 CORS/OPTIONS，唯一保护立即消失**。更直接的是 `/api/monitor/claim|heartbeat|release`（`:511-539`）**在 `:540` 之前 return**，不需要该头，可用 `Content-Type: text/plain` 发 CORS-simple 请求 → 任意网页可**盲发** POST 触发 `subprocess.Popen` 拉起 `watch_resources.py`，并用 heartbeat 无限续租（20 s 租约，`:21`、`:448-455`）。
3. **`extra_run_roots` 把整个 `tmp/` 变成 HTTP 可读**：`:589` `Handler.extra_run_roots = {"tmp": directory.parent / "tmp"}` → `GET /runs/tmp/<相对路径>` 映射到仓库根 `tmp/`。包含校验正确但**范围过大**：`tmp/cloud_ssh/`（15 MB 三方库）、`tmp/audit/*.md`（含本报告）、任何临时下载物都可被读。应收窄到 `tmp/runs`。
4. **静态目录全量可列**：`do_GET` 落回 `super().do_GET()`（`:504`），`SimpleHTTPRequestHandler` 默认开启目录列表 → 无鉴权浏览 `experiments/runs/**`（全部指标、checkpoint、record）。
5. `_request_port`（`:349-354`）把**未校验的 `Host` 头**解析后回写 JSON。当前仅用于显示，但这是 host-header 注入的现成原语。
6. 次要：`ThreadingHTTPServer` 无并发上限，`_read_payload` 在 body 不足时阻塞线程（慢速攻击可耗尽线程）；`monitor_clients`（`:320`）无条数上限（token 洪水可涨内存至 20 s 回收）；`int(resource["root_pid"])` 未显式要求 `pid > 1`（负数在 `:549` `_command_line` 处先失败，属侥幸）。

**利用链（把风险抬成 P0 的关键）**：`open_dashboard.ps1:46` 与 `:98` **两处** `StrictHostKeyChecking=no` → 隧道端主机身份不被校验。中间人可冒充云主机，经 127.0.0.1:18812 返回**恶意 `dashboard.html`**；该 JS 与 dashboard 同源，可直接带 `X-Dashboard-Confirm: yes` 调 `api/kill` 杀训练，并读取 `runs/` 全量结果外传。即：**关闭主机校验把"本地无鉴权面板"变成了"远端可注入的 RCE 入口 / 数据外泄通道"**。

**建议**：非 loopback 直接拒绝 bind；`/api/*` 加一次性 bearer token；`do_POST` 先校验 `Origin` 为 `http://127.0.0.1:<port>`；`/api/monitor/claim` 也要求确认头；补 `do_OPTIONS` 显式 501 并注明"这是 CSRF 防线"；`extra_run_roots` 收窄；静态服务白名单化（只管 `dashboard.html`、`assets/*`、`runs/**/*.jsonl|json`）；`_request_port` 用 `server_address` 忽略 Host；pid 断言 `> 1`。

### 3.2 P1 — `open_dashboard.ps1` 的远端命令拼接与主机校验

- `:43` 远端命令由 PowerShell 双引号插值 `$RemoteRoot`/`$RemotePort` 拼成**一个** argv 交给 `ssh`，远端子 shell 会再解释一次。`$RemoteRoot` 是脚本参数（`:5`），默认值安全，但任何 wrapper/环境变量注入 `;`、`$()`、反引号即等于**以 root 在远端执行任意命令**。修复：改为 `ssh … -- sh -c 'cd "$1" && …' _ "$RemoteRoot"`，或在 `:42` 前用 `^/[A-Za-z0-9_./-]+$` 校验 `$RemoteRoot`、`^[0-9]{1,5}$` 校验 `$RemotePort`。
- `:46`、`:98` 两处 `StrictHostKeyChecking=no`（搭配 `:78` 的 `IdentitiesOnly=yes` 是唯一正确项）。建议首次手工 `ssh-keyscan` 写 `known_hosts`，脚本内改 `accept-new` 或直接删掉。
- `:72-74` 只检查私钥文件存在，未检查 Windows ACL → 建议断言仅当前用户可读。
- 正面：`:52-65` 的 `Stop-Tunnel` 在 kill 前用 `Get-CimInstance` 校验 cmdline 含 `-L <local>:127.0.0.1:<remote>`，避免误杀无关监听进程，写法对症；`:104` 明确"口令不落盘"，未见 `sshpass`/`plink -pw`。

### 3.3 P1 — 三条 `torch.load(weights_only=False)` 的信任边界

| 位置 | 是否热路径 | 载荷为何必须 `False` | 信任边界 | 级别 |
|---|---|---|---|---|
| `src/baselines/deep.py:587` | **死代码**（`git grep load_state_dict_file` 仅命中定义本身） | `channel_mask` 是 NumPy 数组、`config` 是任意 dict（`:572-580`） | 无人调用 | P2（潜伏） |
| `src/transfer/checkpoint.py:56` | 热路径，42 处引用（含 `doc/assets/…/structure.json` 快照） | payload 含 `training_contract`、`training_identity_ledger` 嵌套结构（`:61-64`） | `experiments/runs/` 的 checkpoint，由**云主机 root** 产生并回传 | **P1** |
| `src/transfer/result_reuse.py:116` | 续跑准入路径 | 模型字节快照含 dict/数组对象 | 结果目录；`_read_artifact_bytes`（`:83-93`）只校验 **sha256 完整性**，而 sha256 **不是真实性**——能写目录的人同时能改哈希 | **P1** |

关键：**仓库里已有正确范式**。`src/transfer/calibrated.py:168-169` 用 `torch.serialization.safe_globals([scalar_constructor, np.dtype, type(np.dtype("U"))])` + `weights_only=True`（同文件 `:1222` 亦为 `True`），`doc/engineering.zh.md:15` 把该策略写成明文规范。所以修复不是研究问题，而是"把既有策略铺到另外三处"：

1. **首选（跨主机边界）**：state_dict 与 metadata 分离——`torch.save` 只存张量；`training_contract`/`identity_ledger`/`channel_mask` 写 JSON sidecar（`channel_mask.tolist()`）。这样 `weights_only=True` 无痛可用，元数据可人类 diff、可单独 schema 校验（现有 `checkpoint_training_contract` 本就基于 Mapping，迁移成本低）。
2. **次选**：`weights_only=True` + `add_safe_globals` 精确白名单。注意这**不是安全边界**——白名单里一旦出现可实例化的类，载荷仍能构造对象图；只适用于"来源可信、仅防误加载"。
3. `deep.py:587`：直接删除该死方法；若保留，改 `weights_only=True` 且 `channel_mask` 存 list。
4. `result_reuse.py`：模型快照改纯 state_dict + `weights_only=True`；若必须保留对象图，引入**签名**（如 hmac + 本地密钥）而非仅 sha256，否则 sha256 只挡传输损坏、不挡有意投毒。

其余 sink 复核（与 Lead 初筛一致，全部通过）：
- `np.load` 共 15 处（`src/data/dataset.py:154,157`、`src/data/epochs.py:1324`、`src/data/study.py:197`、`src/transfer/result_reuse.py:98,243`、`experiments/*` 等），**全部显式 `allow_pickle=False`**，无 `allow_pickle=True`。
- `eval(`/`exec(`/`pickle.load`/`yaml.load`/`shell=True`：**零命中**（结果里的 `.eval()` 全是 `nn.Module.eval()`）。
- 所有 `subprocess` 均为 argv 列表：`dashboard_server.py:425`、`watch_resources.py:39,117`、`run_with_dashboard.py:77,96`、`freeze_source_snapshot.py:45,46`、`run_binary_subject_transfer.py:118`、`run_candidate_promotion_matrix.py:3227`。
- CLI 不可信路径：`watch_resources.py:25-28` 的 `_resolve_run` 接受任意目录，但只用于 `ps`/`nvidia-smi` 统计，不落盘不拼 shell；`run_with_dashboard.py:70-72` 用 `argparse.REMAINDER` 收命令后 `subprocess.call(cmd)`（argv），无注入。全仓唯一"网络面解析器"就是 `dashboard_server.py`。


## 4. 本地工具配置治理（`.claude`）

### 4.1 P1 — `bypassPermissions` 只靠**用户全局** gitignore 兜底，仓库自身无防线

| 检查 | 结果 |
|---|---|
| `.claude/settings.local.json` 内容 | `"defaultMode": "bypassPermissions"` + `"allow": ["Agent"]`（110 B） |
| 是否被 Git 忽略 | **是**，但依据是**用户全局** ignore（`<用户目录>/.config/git/ignore:1`）的 `**/.claude/settings.local.json` |
| 仓库 `.gitignore` 含 `.claude`？ | **否**（全文 37 行） |
| `.git/info/exclude` | 仅注释 |
| `git check-ignore` 对 backup | **不忽略** |
| `git add -n .claude` | 会暂存 `settings.local.json.backup-20261011-232131` |
| 是否曾进入历史 | 否（`git log --all -- .claude` 为空） |
| CI / pre-commit | 均无（无 `.github`、无 `.gitlab-ci.yml`、无 `.pre-commit-config.yaml`） |

风险三层：
1. **可迁移性**：保护只存在于当前这台机器的全局 ignore。换机、换用户、CI runner、协作者 clone（该规则不随仓库分发）→ `git add -A` 会把 `settings.local.json`（权限绕过配置）与 backup 一起提交。这是"配置在开发者本机、防线也在本机"的典型失效。
2. **backup 已可入库**：`settings.local.json.backup-20261011-232131`（62 B，旧版 `allow:["Agent"]`）当前是 `??` 且不被忽略，任何一次 `git add -A` 都会带上。它虽不含 `bypassPermissions`，但把"本项目用 Agent 自动执行"写进了公开历史。
3. **语义风险**：`defaultMode=bypassPermissions` 使任何在本仓库工作的 agent 无审批执行命令；而 `AGENTS.md:35-38` 明确鼓励"自主跑测试、短探测，然后直接进全量训练"。两者叠加意味着：一条被污染的文档/issue 就能让 agent 直接执行 `open_dashboard.ps1`（§3.2 的远端 root 命令）或 `torch.load(weights_only=False)`（§3.3）——不需要人点确认。

补充：仓库**没有任何 CI/预提交钩子**，因此 §2.1 的机器名、§2.2 的 172 MB 产物、`.claude` 备份都没有自动拦截点。

### 4.2 处置建议（只建议）
- `.gitignore` 追加 `.claude/`（连 backup 覆盖），并在 `doc/engineering.zh.md` 写明"本地 agent 配置不入库"。
- 若要共享团队配置，改为提交 `.claude/settings.json`（无 `bypassPermissions`、allow 列表最小化），忽略 `.claude/settings.local.json*`。
- 加 `.pre-commit-config.yaml`（或 lefthook）：① `detect-secrets`/`gitleaks`；② 自定义 `grep -n 'C:\\Users|/home/[a-z]|hostname="'` 拒绝；③ `git diff --check`（`AGENTS.md:15-16` 已要求，但只在人记得时执行）。
- 不把 `bypassPermissions` 作为默认模式；至少对 `ssh`/`git push`/`torch.load` 相关路径保留确认。

### 4.3 P2 — 绝对路径/机器名是否带入文档：正文干净，附件不干净
- `doc/*.md`、`README.md`、`src`、`experiments` 中 `C:\Users` **零命中** → 正文脱敏做到了（主机只以 `autodl` 泛称出现）。
- 但 `doc/assets/` 的 3 个已提交附件带出真实路径与主机名（§2.1）→ "正文有规范、附件无规范"。建议在 `doc/engineering.zh.md` 的证据附件小节加硬规则，并在 evidence README 模板里加检查项。

## 5. 优先级处置清单（只建议，未执行任何修改）

### P0 — 立即（今天）
1. **给 `dashboard_server.py` 加鉴权并收窄暴露面**：`/api/*` 统一 bearer token；`do_POST` 先校验 `Origin`/`Sec-Fetch-Site`；`/api/monitor/claim` 也要求确认头；`extra_run_roots` 从整个 `tmp/` 收窄为 `tmp/runs`；静态服务白名单化；补 `do_OPTIONS` 并注释其 CSRF 作用（`experiments/dashboard_server.py:420-433,504,540,566-573,589`）。
2. **去掉两处 `StrictHostKeyChecking=no`**：先人工 `ssh-keyscan` 落 `known_hosts`，脚本改 `accept-new` 或删除（`open_dashboard.ps1:46,98`）。这是 §3.1 利用链的另一半。
3. **确认云主机暴露面**：`open_dashboard.ps1:2-5` 的 host/port/root 已随历史扩散；若仓库曾推送远端，按"入口已公开"加固（改端口、禁 root 直登、fail2ban）；同时把脚本参数化，仓库内只留 `ssh n2p3-cloud` 别名。

### P1 — 本周
4. **主机入口外部化**：`open_dashboard.ps1:2-8` 的 host/port/user/RemoteRoot/私钥名全部外部化。历史不可改写，重点是评估"曾推送否"，不是 `git rm`。
5. **清洗机器身份附件**：`doc/assets/code_quality_repair_20260908/validation.json:29`（账户名+临时路径）、`doc/assets/technical_audit_20260906/*.xml` 8 个文件的 `hostname` 属性（§2.1）。
6. **堵住 `doc/assets` 的 3,352 个未跟踪产物**：改白名单入库；显式忽略 `.bdf/.pt/.pyc/.log/.zip`；对 `doc/assets/system_implementation_20261004/review_spec/short_record/person_eeg.bdf` 作出"合成/真实"书面判定。
7. **补齐 `structured_erp` 三文件入库**：`src/baselines/structured_erp.py`、`temporal_transfer.py`、`structured_erp.LICENSE`（`pyproject.toml:29` 已声明依赖），否则 HEAD 不可导入且 BSD 归属丢失。
8. **补根 `LICENSE` + `CITATION.cff` + `NOTICE`**：明确项目许可，NOTICE 里列出 `jsosulski/toeplitzlda`（BSD）与各数据集许可。
9. **补数据集许可声明表**：在 `README.md` 的 Data And History 后新增一节，逐库写 BI2014a / BNCI2014-008 / GTN(MNE-P3-data/Chailloux2020) / Mainsah 的许可与"哪些派生物可随仓库分发"。
10. **处理 `Paper/` 版权**：先确认 59 个 PDF 的再分发依据；公开前把出版商版（TNSRE/JNE/Frontiers）移出 Git，只留 `Paper/index.md` 与 DOI/arXiv 链接。
11. **把 `torch.load` 三处改为既有范式**：优先生成 JSON sidecar 分离 state_dict（`src/transfer/checkpoint.py:56`、`src/transfer/result_reuse.py:116`），删除死代码 `src/baselines/deep.py:587`，对照 `src/transfer/calibrated.py:168-169` 的 `safe_globals` 写法。
12. **`.claude` 治理**：`.gitignore` 加 `.claude/`；提交无 `bypassPermissions` 的 `settings.json` 模板。

### P2 — 长期
13. **加 CI 与预提交**：`detect-secrets`/`gitleaks` + 机器路径/主机名检查 + `ruff` + `pytest`（当前 `.github` 不存在，`AGENTS.md:15-16` 的 Ruff/`git diff --check` 全凭自觉）。
14. **`.git` 8.2 GB 治理**：评估对 370 MB + 185 MB 的 `frozen/*.source-only.tar.gz` 做 `git filter-repo`（改写历史需全体协作者重克隆、改变所有 commit id，须与冻结清单策略一并决策）；短期至少把 `frozen/local_history/` 的 2.5 GB 归档移到仓库外。
15. **`dashboard_server.py` 韧性加固**：`ThreadingHTTPServer` 并发上限、`monitor_clients` 条数上限、`_read_payload` 读超时、`pid > 1` 断言、`_request_port` 忽略 `Host`（`:320,349-354,367-373,548,566-573`）。
16. **`open_dashboard.ps1:43` 远端命令改 `sh -c '…"$1"' _ "$RemoteRoot"`** 并加参数白名单校验（当前是 root 命令注入模板）。
17. **把"成人/设备确认"从文字变门禁**：`configs/brainsync_collection_plan.json:44` 的 `healthy_adult_scope_requires_age_evidence` 目前无代码读取；建议在结果准入处（与 `src/transfer/result_reuse.py` 的 `admit_completed_result` 同层）加断言：未声明 `adult_confirmed`/`device_confirmed` 的 run 不得产出对外结论字段。
18. **文档一致性**：`README.md:101-103` 称"GitHub contains source, tests, guidance and evidence manifests / A clone does not include datasets"，与 `Paper/` 200.7 MB 入库、`frozen/source_eog_training_20260909/source_eog_20260909_results.manifest.json` 达 9.4 MB 的事实不符，需改写。

## 附：未覆盖 / 需他人确认
- `tmp/cloud_ssh/` 仅确认目录结构为 `pip --target` 安装，**未逐文件比对 PyPI 哈希**，也未检查 `bin/` 下是否有可执行 shim。
- 两份 370 MB / 185 MB 的 `source-only.tar.gz` **未解包**核对清单（建议另派只读任务）。
- `experiments/runs/`（`.gitignore:20` 排除）内的实际结果未审计，仅通过 dashboard 读路径推断其可被无鉴权访问。
- 云主机实际配置（root 直登、35832 是否暴露公网、私钥是否带口令）**无法从仓库判定**，需运维核实。
- 未评估许可证与**商业使用**的兼容性（仅指出缺失），需法务判断。

