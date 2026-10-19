# P1 stationary cloud patch：author 交接

状态：代码与窄软件反例已交付；科学 full16、new root compile、fit 和 score 都未启动。云执行仍为 **HOLD**，依赖 sole cloud owner 的 late-orphan-safe controller 最终接受，以及云 preparation owner 的 exact-SHA relocation / sealed Windows export factory。

## 独审 F1/F2 修后接点

首轮 `4742f6ea` 被独审否证：decoded held expected由new plan自签，cloud入口缺跨attempt互斥。原20 facts及首轮HOLD在integration的 `tmp/STATIONARY_CLOUD_CODE_REVIEW/` 原字节保留；不能继续把首轮作者27/1当作授权关系或跨attempt安全证明。

F1新cloud plan必须显式添加 `original_export_plan={path, sha256}`，其SHA为实际Windows producer plan `0569777f5f99eaac753b0fb015ab0fcec761a2b4e9dcf14071044338f1f0485e`。这是本次8人固定程序的输入authority，不是通用Fitted或跨实验参数。它与native `scientific_plan.json` 不是同一个计划。core读取exact旧plan bytes，严格核new plan的整个participants、original native plan/input/capsule/loader/original_files/old reference/decision/request及zero-fit字段与它相等；只有新source/resources/cloud执行字段另持自己的身份。

`_held_authority`独立核实际input.json的rawSHA、唯一participant/5 C/1 held、完整held row、geometry、response model及 `held.view == native.views[5]`。worker沿该原view调用 `PredictionDecisionView.from_record(view,samples=actual_samples)`，核原raw/physical input digest；design expected始终来自原056 participant，array descriptor/keys从不可变原input.json准入。改变decoder或new plan writer不能互相给自己证明。

`_export_state`也独立核IOowner返回的plan SHA==原056以及export script SHA==原056 source_files中的script，locator用PureWindowsPath保持原Windows拼写；old Windows export JSON/NPZ不重签为新cloud score plan。factory仍须消费已sealed Windows manifest，不能从待验export body临时生成authority。

F2 `programme_attempt`复用 `source_execution.process_file_lock`，在任何actor/science写入前建立原子exclusive `controller.lease.json`，tuple含run_id、PID/birth、plan raw SHA、实际entry source SHA和previous attempt；持锁覆盖整个run、ACK与reduce。先后两次samebirth实际alive检查关闭争用；并发等待期间另一attempt已经完成时也拒绝本次竞争，不默默再启动第二轮。旧dead lease在历史门通过后另名保存，closed attempt另名保留，lease归档错误不覆盖原primary。

旧local handle/identity/history门仍复用 `assert_no_owned_live`。cloud历史树必须由共享controller公开接口核验：

```python
research.cloud_execution.assert_no_prior_owned_live(diagnostic_directory: pathlib.Path) -> None
```

它应消费自己的持久bootstrap/actual owned PID-birth/token/containment关系；live、unknown、读取错误或ownership变化raise RuntimeError/ValueError；只证明历史资源门，不编辑science。返回非None或接口缺失明确HOLD。本作者不实现或复制cloud树算法，不把c45等未接受source当accepted。该接口的真实Linuxpeer/descendant行为仍由sole cloud owner独立完成；软件stub通过不代表实际history门通过。

repair窄套为23 PASS/1 Linux SKIP/0 FAIL/0 ERROR，exit0：`tests/research/test_stationary_programme_lease.py`、已有native/control及cloudprogramme窄tests。JUnit `tmp/P1_CLOUD_REPAIR/software.xml` raw SHA `efb330a5a76526a9873af1c993eda4b2872efa7148a166eea929ba883923a3a9`。没有重跑全套。

真实A01 read-only repair probe：`D:\BCI\n2p3-net\.venv\Scripts\python.exe -B tmp/P1_CLOUD_REPAIR/contract.py`，OS0，11 facts。原actual local plan/newcloud whole authority正控过，原sealed Windows export保056，合法worker到native5 pool故意拦停；resigned sample/design expected、changed held reference、wrong array key/participant、actual wrong decoder sample/design均在pool/rootcompile/score前拒绝。receipt `contract.json` raw SHA `87cf1b461cfb5f47945e9fba915a7c654c84c4f90cb872b6b8f596fd1c8933d6`。fit0、old model load0、新 `/3` root compile0、score0。本probe没有重新执行old `/2` strict关系root重编；旧Windows strict-export的原关系重编与新 `/3` root counter是不同计数。

direct-script支持已加actual ROOT/src+ROOT引导。真实argv：`<locked-python> <new-freeze>/experiments/run_stationary_cloud_panel.py --plan <new-cloud-plan>`。仓库外cwd `C:\Users\chent\AppData\Local\Temp` 且unset PYTHONPATH运行实际 `--help` OS0；同一bootstrap下fresh spawn import的panel、entry、scorer路径均精确指向作者source，child exit0。proof `tmp/P1_CLOUD_REPAIR/bootstrap.json` raw SHA `4420bb45bf799dc868a328c2771d3bbff71dcbfc21f9904b6b4b560e4741cd18`。spawn继承由实际entry建立的source路径，不能靠editable install碰巧准入。

repair source/commit以root实际接收的最终冻结字节为准；下文旧source hash表保首轮作者证据含义，不能拿它替代repair freeze。

## 固定科学范围

A01–A08 × shared8/private8，共16格；seed `20260918`；原首五完整组 C600，第六完整组 held120，unused80 单列。AR8/prefix32、原 response M/R/Σ/F、PARCOR/Sigma/Temporal primitives、onsets、membership、candidate vocabulary、物理 read/conditioning/score 支持、完整 requested owner 与原指标归约都保留。

新 `/3` joint stationary law 由原 `compile_noise` 与已接受 compiler 生成；没有第二个 fitter。旧 `/2` compiled-W reference immutable，不根据新结果重签旧 producer。scorer/compiler 实际 bytes 常数分别仍为 `452b18bbe5ce3601f123c10638e2a5fdf6bfa351c0c5b038c240f09d2469f8b2` / `621badbd538bd724d24f91129f0c226db1c7001db90eeb1f013005dd81710c5f`。

旧冻结文件只读 hash 复核：

| 旧入口 | plan raw SHA | binding raw SHA | frozen source closure |
|---|---|---|---|
| stage05 | `0569777f5f99eaac753b0fb015ab0fcec761a2b4e9dcf14071044338f1f0485e` | `618cab811ba1be649dd22a80a3b1c90b3f5bd6baaa7c84fec51de9595831d625` | 158 members，0 mismatch |
| stage06 | `00f1975ef6be5fb536bb849ed480f23922c521acd72d672bb14414ae04118e4a` | `4454b1bea5efe6c05a31e9b467344144ffd18df629e5dccdc0a6d324d9020dbd` | 158 members，0 mismatch |

没有改写、删除旧 stage05/06 freeze、plan、binding、failure 或旧 state。author HEAD 起点为 `9497a632a39b84b37d20f87ea0532f62a3973df5`。新代码须由 root 与 accepted cloud owner 共同放入新的 freeze；旧目录不热改。

## A：实际 native budget

`stationary_panel.worker_native_threads` 严格读取 `plan.resources.native_threads`，拒绝 bool/string/zero/negative/noninteger，worker 在任何 export/compile/score 前准入。原 worker 的固定 `threadpool_limits(limits=1)` 改为该值；cell runtime 保存 declared value、实际 `threadpool_info()`、torch native/interop 值。

云端拓扑为四个 actor，每个 `PhaseResources('evaluation', cpu_threads=6, native_threads=5, interop_threads=1, device='cpu')`，一个 coordinator：`4*(5+1)+1=25`。原 native6+interop1 每 actor 的方案将总量变成29，不能准入。actor 原生环境和 torch interop 由共享 CloudExecutor 在 adapter import/execute 前建立，`cloud_worker` 再核其实际值。GPU claim=0；worker 容量4不表示这里有GPU计算。

original Windows strict export 的原 short load2 关系保持。export 与等待它的 parent actor 不同时进行科学 compute；原 loader 的其余单线程行为仍来自旧 frozen script。新云路径消费 sealed Windows 出口，不在 Linux 重跑 old strict loader。最终 metric reduce 也显式处于原 `cpu_thread_budget(coordinator1)`。

## B：共享执行器与科学 ACK

入口为 `experiments/run_stationary_cloud_panel.py --plan <new-cloud-plan.json>`，必须从新 plan 的 `source_root` 对应 frozen 文件执行，并使用该冻结 source 的 PYTHONPATH。

`StationaryCloudProgramme` 构造8个独立 subject task；每个 task 调用同一 module 的 `cloud_worker→worker`，该 subject 的 shared/private 两臂按原顺序完成。task arguments 只有 plan/factory/subject/task/thread 描述，不通过IPC携带EEG或live model。

`resolve_source_resources` 负责真实 CPU affinity/cgroup / host-memory 解析；`HostMemoryAdmission` 持有 CPU/memory/GPU 额度；`CloudExecutor` 持有 ready queue、actor/process tree、失败与诊断。这里没有复制 host-VM/CPU/RAM scheduler，没有修改这些 owner、source observer 或 prefix owner。

Cloud declaration 强制 CPU25、worker capacity4、native5、interop1、coordinator1、GPU claim0；解析后无法达到CPU25/四路容量时明确拒绝。资源配置从 `configs/cloud_training_resources.json` 的集中规则出发；本程序无 source fits，云 preparation 应为 CPU evaluation 的完整 resident peak 显式提供相符的 source/target memory budgets，避免沿用18 GiB source-fit预算制造虚假的容量缩减。名义线程/容量与真实CPU/RSS/墙钟不是同一验收项。

主进程 `on_finished` 串行核 `alive=false`、exit0、`process_tree_retirement.active_owned_processes_after_exit=0`、subject/task/result tuple，然后依次 `admit_cell(shared8)`、`admit_cell(private8)`，两格都成功才 ACK subject。失败 actor 不触碰原 primary；失败 ACK 由共享 executor 记录 artifact_admission，不补 numerical-unavailable。完整16 requested、已封存 cell、partial 及错误保留。

仅当全部8 task均完成、所有树已settle、所有8 subject均ACK后，`finish` 才调用原 `reduce_program`。queue失败输出仍保16 requested；reduction/基础执行异常有独立原 failure journal，journal自身失败以 note 附于原异常而非覆盖 primary。

`LiveControl.checkpoint` 只核 actual `N2P3_CLOUD_TASK_ID`，写 shared worker diagnostic phase 和当前 subject progress。它没有 local CPU pause 或另一套资源调度器；实际资源和actor停止仍由共享 executor 持有。

## Linux original loader 与 Windows export 边界

root 已明确：Linux old `/2` strict load 到原关系门失败，不能称跨host old loader通过。选择在 Windows 原 frozen loader 下导出全部8 participant，再以显式 cross-host binding 在 Linux消费相同字节。

`load_plan / worker / admit_cell / reduce_program` 添加可选 `input_owner=None` IO seam。local默认路径保持；cloud只通过 owner 解析原 Windows literal，原 plan/manifest/capsule/noise body不做全局文本替换、不重签。`load_plan` 仍由核心核 source/original/capsule hash、actual original native plan、participants/16 requested、decision/input authority。

cloud factory规格：

```python
spec = {
    'factory': 'cloud_preparation_module:factory',
    'source_sha256': '<actual factory module raw byte SHA>',
    'arguments': {'...': 'cloud preparation declared arguments'},
}
owner = factory(spec['arguments'])
owner.resolve(original_literal) -> Path
owner.bound_input(original_manifest_descriptor) -> arrays/file/verify_unchanged compatible input
owner.strict_export(plan_file, subject, export_directory)  # install sealed Windows export bytes only
owner.export_authority(subject) -> {
    'plan_file_sha256': '<actual original Windows export plan>',
    'export_script_sha256': '<actual original Windows export script>',
    'loader_sources': '<original capsule loader source descriptors>',
    'export_json_sha256': '<exact sealed export.json bytes>',
    'response_arrays_sha256': '<exact sealed response.npz bytes>',
}
```

factory模块自身 raw SHA 必须与新 plan spec 相符。云 preparation owner 仍需交付实际 factory/interface/binding，当前代码不是对尚未出现接口的科学接受。

`_export_state` 云路径先核 sealed export JSON/NPZ 字节与 Windows authority；export保原 plan/script/loader身份。然后仍核原 response record、实际mean/F、原 input digest、noise state/old state SHA/original file descriptor。新 cell receipt绑定新的cloud score plan，引用原 export raw SHA，不把export重签为新cloudplan。

bound_input解码后的 held samples/designs再直接核原 numeric array_digest，在rootcompile/score前拒绝 decoder/relocation错。原 subject/noise/response关系不是由alias猜测。

## Central alive 与 Linux zombie 反例

`local_program_control.actual_alive(pid,birth)` 集中核原0.01秒同birth门、is_running、notZOMBIE；PID不存在/被复用/已zombie返回False，权限/读取故障仍阻止准入。`read_control`、`OwnedProcessTree`、historical `assert_no_owned_live` 和 local controller lease 接点都消费它。

同birth launcher 即使是zombie，`OwnedProcessTree`仍发现其实际后代；historical recorded live child仍是resume blocker。不能通过leader退役把live orphan当消失。

真实Linux测试用 `os.fork` 生成未reap zombie，明确证实普通 `psutil.is_running()` 为True而 central predicate为False；核control拒绝、tree无该zombie、zombie handle可退役；加入一个samebirth live recorded child后resume必须拒绝。Windows只跳过该Linux case，不报告为通过。执行云端命令为：

`python -m pytest -q tests/research/test_stationary_cloud_control.py::test_linux_real_zombie_releases_lease_but_live_recorded_child_blocks_resume --junitxml=<new-owned-dir>/linux_zombie.xml --basetemp=<new-owned-dir>/linux_zombie_cases`

该case和 accepted controller 的 late-orphan retirement 是两个不同owner的门；前者通过不解除后者HOLD。

## 验证证据与下一步

实际local窄套：27 PASS / 1 Linux SKIP / 0 FAIL / 0 ERROR，exit0。JUnit `tmp/P1_CLOUD_PATCH_TESTS/final_junit.xml` raw SHA `dc21b69af501d13f98fc7f29dff6d02b46e77d69069e96102a50b59b8922e969`。命令：

`D:\BCI\n2p3-net\.venv\Scripts\python.exe -m pytest -q tests/research/test_stationary_cloud_control.py tests/research/test_stationary_cloud_programme.py tests/research/test_local_program_control.py tests/research/test_stationary_panel_path_contract.py --junitxml=tmp/P1_CLOUD_PATCH_TESTS/final_junit.xml --basetemp=tmp/P1_CLOUD_PATCH_TESTS/final_cases`

作者实际 source raw SHA（新 freeze须以它实际接收的字节重新绑定，不冒用旧冻结hash）：

| 文件 | raw SHA-256 |
|---|---|
| `src/research/stationary_panel.py` | `d10f440f9a3c41d8f888ae6810fdfd6afb6b937c3b159a900d4279aabc1bb4a5` |
| `src/research/local_program_control.py` | `55c2ad163dcc7c38d92a25441c3bae48ec11c0f3aafaa0384fac2fea89fc45bd` |
| `experiments/run_stationary_channel_noise_panel.py` | `3d13c13f08da0048f625aa2bcef85cd893884b71e8eafdabb8af2be94c9d1e3d` |
| `experiments/run_stationary_cloud_panel.py` | `141055298c6770e3f6f8c59989574b95799f0cafab02430e3ff52942a217331c` |

反例包括：native值非法拒绝/实际worker消费5/错held输入0compile拒绝、8 tasks/16 requested/CPU25、不可满足容量不静默降级、LiveControl actor关系、两arm串行ACK/缺ACK不reduce、unsettled actor拒ACK、cross-host旧export保原plan身份/字节篡改拒绝、实际共享executor spawn的ACK失败与原actor失败保持两个真实primary，tree退出后资源账本回原值。实际executor case是两路轻量software fixture，不是四路科研吞吐。

Ruff changed files、`git diff --check`、changed module compileall、新CLI help/import均exit0。原scientific scorer/compiler未修改；这次没有新科研成绩、训练、root compile或score。

执行前需 root 接受本patch逻辑；sole cloud owner交付并接受late-orphan-safe controller；cloud preparation owner完成实际 exact-SHA factory和全8 Windows export binding；root把最终 owner/source一起新freeze，重核source/plan/binding；云端运行真实Linux zombie与实际relocation/admission窄probe。只有这些出口满足后才进入已授权full16，实际CPU/RSS/进度与完整16 receipts另验收。
