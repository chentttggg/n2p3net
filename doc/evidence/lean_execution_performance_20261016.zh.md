# Lean 检查启动、并发与资源测量

日期：2026-10-16。对同一冻结证明进行真实检查，未改生产算法。
采用 `performance-optimization` 的同输入对照方法，并用
`get-available-resources` 核对可用资源；未运行压力测试。

## 瓶颈与解释

本机有 16 个物理核、32 个逻辑线程，当前进程可用全部 32 个线程。
证明构造、源码语义审查与模型推理在远端 agent 完成；本机运行 Lean 检查器。
单个 `-j1` 检查即使占满一核，也只约为整机 3.125% CPU。
低总占用不代表检查器没有工作，也不能据此认定更多本地 CPU 会等比例加速。

测量发现两项可以消除的本地开销：每次经 Lake 启动、以及 Mathlib/标准库之前的
多个无关搜索目录导致的文件查找开销。早先全部检查串行，曾出现约 328 秒的
等待锁时间。前期基于 RSS 直接限制并发过于保守：RSS 包含大量共享库页面。
两进程私有提交内存与 USS 的实际测量支持有限并发。

本轮一次独立审查还遇到远端服务 502，重试后完成。尚未对完整 agent 推导与编译回路做统一计时，
不能给出二者的精确耗时占比。

## 同源测量

所有命令均退出 0，独立检查日志为空，前后源码 SHA-256 一致。
原始日志与完整运行回执保留在本地 `formal/lean/artifacts/`。

| 对照 | 墙钟秒 | 结论 |
|---|---:|---|
| `launcher_benchmark`，Lake→Lean j1，两次 | 70.86 / 72.36 | 启动基线 |
| 同轮直接 Lean j1，两次 | 61.99 / 59.28 | 省去重复 Lake 启动约 15% |
| 同轮直接 Lean j2 | 54.95 | 该文件增加线程有有限收益 |
| `path_benchmark_paired`，原路径，交替两次 | 50.85 / 50.50 | 相邻对照的原路径基线 |
| 同轮安全重排路径，交替两次 | 26.84 / 26.60 | 均值用时减少约 47%，约 1.90 倍 |
| `parallel_benchmark`，Covariance/FirstSlice 串行 | 58.29 | 两个相同文件，均使用重排路径 |
| 同轮两个独立进程 | 31.09 | 总吞吐约 1.87 倍 |

首次重排测试曾测得 17.39 / 14.18 秒；随后交替对照为 26.84 / 26.60 秒。
内存与系统缓存状态会使绝对时间变化，采用相邻交替对照的约 1.90 倍作为路径收益。
启动、路径和并发实验是不同测量，不能把各自倍数直接相乘当作全流程加速比。

双进程运行时，采样 RSS 相加峰值约 2.83 GiB（会重复计算共享页面），
私有提交合计峰值约 557 MiB、USS 合计峰值约 343 MiB；整机可用内存最低约
1.57 GiB。该证据支持两个独立检查，尚未证明更高并发有收益。

## 检查运行器

[run_lean.py](../../formal/lean/scripts/run_lean.py)缓存仅三个命名 Lean 环境变量，
按工具链/配置/依赖清单和两个可执行文件摘要失效。独立文件直接启动固定 Lean；
显式纳入标准库路径，只有顶层 namespace 不冲突才重排，未知目录保留原位置。
获得检查名额后再次从原始路径核对，配置变化则退出重新准备，避免排队期间的构建改变解析。

FIFO 队列最多接纳两个独立只读检查；第二名额要求实时可用内存至少 1.25 GiB，
构建和缓存保持独占，已排队的写任务阻止后来的读任务越过。
Windows 命令先挂起创建，再绑定 kill-on-close JobObject 并恢复，控制器退出时
实际 Lean/Lake 及后代随之回收；POSIX 分支只声明普通 context 的进程组清理。

[8 项针对性测试回执](../assets/lean_gaussian_20261016/runner_validation.json)已通过，
覆盖真实 child/grandchild 的控制器崩溃、PID 重用、并发上限、writer 公平和排斥、
内存门禁、缓存失效与入场时路径再验证。独立审查未发现剩余阻断问题。
实际证明的聚合重查单独记录，测试通过不替代 Lean 公理及错误证明拒绝审查。

库准备后的首次真实双进程检查分别耗时 127.18 / 118.76 秒，入场可用内存仅
1.45 / 1.38 GiB；同次系统采样观察到 304 次/秒页读和 3559 页/秒输入。
随后聚合中的四个新模块检查为 30.32–40.96 秒。首次集成与暖缓存基准的资源状态不同，
既不能把 1.87 倍吞吐当作所有时刻的保证，也不能据这两组不同条件宣称并发本身退化。
内存/文件缓存状态与依赖准备需要纳入后续成本估计。

## 复现

在仓库根目录，输出目录必须尚不存在：

```powershell
.venv\Scripts\python.exe formal/lean/scripts/benchmark_lean.py --output formal/lean/artifacts/launcher_reproduction
.venv\Scripts\python.exe formal/lean/scripts/benchmark_paths.py --environment-report formal/lean/artifacts/launcher_reproduction/report.json --output formal/lean/artifacts/paths_reproduction --paired
.venv\Scripts\python.exe formal/lean/scripts/benchmark_parallel.py --environment-report formal/lean/artifacts/paths_reproduction/report.json --output formal/lean/artifacts/parallel_reproduction
```

路径重排先验证所有已有搜索目录的顶层模块 namespace 不冲突；存在冲突时不能
改变模块优先级。构建/缓存写入和独立只读重查需要不同调度，不能同时改写导入产物。
完整定理及数值/源码对应范围仍见[执行方案](../roadmap.zh.md#2026-10-16-lean-形式化验证顺序方案)。

已接受的后续概率切片中，三个新模块的独立重查为 20.88、22.25、19.48 秒；
与低内存首次集成的 119–127 秒再次表明准备阶段和内存/文件缓存条件需要分别记账。

## 2026-10-18 ERP 阶段的资源与等待

[固定性能核查](../assets/lean_erp_20261018/performance_review.json)记录授权时整机CPU约23.7%、
可用内存约15.7GiB、32逻辑核；这些是该时刻快照，不代表之后持续占用。
单次独立检查曾仅排队0.0684秒，另一次在较早独占构建期间排队70.5871秒。
两个ready模块的合并Lake构建耗时150.0665秒、采样CPU96.625秒，不能把不同模块/条件直接作为吞吐对照。

两个模块的setup各引用相同2391份olean，磁盘文件合计1,541,049,696字节；
这是静态文件体积，不是测得的I/O量或RSS。只读检查表明同一Lean前端可直接输出olean，
但绕过Lake前必须保持依赖闭包、setup、选项和输出隔离；本轮没有替换构建或运行加速基准。
代理的远端推理/写证明时长未被此本地采样器计量，整机低CPU不能直接当作代理空闲。
新增第4个子代理处理独立实际源码探针与审查，检查器仍保持两个只读名额及构建独占。
