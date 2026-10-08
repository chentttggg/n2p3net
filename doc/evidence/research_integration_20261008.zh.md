# 科研与代码集成

更新：2026-10-09。科研分支 `codex/scientific-mechanisms`，依赖起点 `711e230`；实际执行以各次源码字节和原产物为准。科研窗口持有科学合同及最终集成，另一个窗口持有云执行、资源调度与旧B08运输；两个窗口合计最多10并发，只同步重要交付、接口变化和阻塞。科研子代理无我方时间或额度上限。

用户明确科研可本地独立推进，不等云数值矩阵才进入下一项研究。当前云实例已关闭；本地小矩阵、数学反例、实际调用链与保存重放分别验收，不将其解释为新公共效能或成人确认。

## 当前骨干与出口

| 切片 | 当前实现和审阅 | 仍须完成的具体出口 |
|---|---|---|
| B07字典 | `/2`冻结五数组、固定标量算术和物理表达式/constraint/lift；owned模型与ensemble直接消费原dictionary，独立全路径审查闭合 | completion保存重放已通过，完整同版本跨host模型重放待验；有限整selection history参考已以共享β与冻结joint prior边缘化，proper/flat测度分开；生产完整history接入/derivative另验 |
| B08先验 | 物理equal-class response metric、quotient范围及显式completion；全C/prior生产Q/R/rhs/M保存，评分用precision factor，H是derived diagnostic | 真实completed-fit与弱方向攻击已验；quotient/completion新的公共配对及同版本跨host仍待验 |
| B05噪声归因 | shared/private与一般vec分开；原576状态归因、Σ-only均值取消及增量/fulltarget关系明确；同B05 owner新增fixed-physical-prior given-noise Cfit | 真实noise/response可归因控制、正式fit包与finite endpoint交互仍需验收；生产joint MLE未由状态诊断实现 |
| B06可观测性 | 实际O/L核空间及每个请求泛函独立归一；真实native有限T、ERP/噪声增量和外部未观测witness | 真实A01只读计划已完成且独立验收；未知state拟合、最终完整预处理和独立效能仍待各自验收 |
| B09空间 | native/reference/CAR结构与输出gain；一般vec signal prior、冻结noise因子传播、保存生产QR与只读诊断CLI | 真实A01六格和八人48格归因已完成并验收，Pz4无额外hit且NLL升；联合MLE及新效能另验；工作站入口需共同授权owner另接 |
| B14/B12 | 已审阅e51输入的24个owned文件逐一合入，epoch采样委托共同合同；requested任务、输入/产物预算和单seed源/目标共享owner | 最终composition已审阅合入；3项Linux并发case、真实25CPU/4GPU吞吐与完整公共两面板仍待验 |
| B10输入污染 | 已实现完整自适应程序增量和条件经验矩，复用实际harmonic/finite owner | 14项实际pipeline反例和原CLI审计入口已验；真实污染分轴配对及固定θ模型另验 |
| B11重尾参考 | noise-only Gamma与proper Gaussian积分比；fixed-shape projected尺度/ν与Fisher参考，rank截断误识noise已独立闭合 | 实际K0/独立noise/C权限、完整新β/λ后验和公共接线仍待验；不选择默认ν或自动部署 |

B07的完整模型测试原27项26通过、1个completion实际fit误拒，已独立诊断为显式Q生成误差未纳入逐列后向预算。修复保原producer，不重QR覆盖；9项必要completed-fit/保存/held及弱方向攻击已通过。实际源码、失败和通过日志分列，未合计为整仓通过。

B08 quotient在混合尺度请求下原本会漏掉小的unseen functional；独立4项先跑出3失败/1通过。逐行尺度修复后的4新反例与8关联换基/单位用例共12通过。B06同类全局尺度错误及恢复gain溢出已经修复，11项相关用例通过。科学可识别性和浮点resolution声明分别保留。

B09已有18项必要Windows测试通过，当前CLI新增Cal/held实际采样率门的两项反例通过；从仓库外目录启动help成功。它固定实际C拟合noise后重拟合signal prior，只支持这个可归因机制诊断，不能签联合MLE或真人效果。

## 真实来源、保存与失败边界

生产TemporalNoise已统一为显式compiled `/5`：保存并消费实际prefix数值态和scale，载入保producer并核其数学关系，不由参数重新编译取代原数值。旧1–4由原执行源码重放，旧0397运输不追认为新生产合同通过。新canonical编译12项、弱pivot/array-header各1项以及25项noise→模型→NumPy/Torch→字典/ensemble联动通过；保存原生产状态、协议与梯度目标分别验收。

ContinuousDecisionModel实际fit目前只接受相同implementation字节的恢复；字典与noise载荷允许保原producer，并不意味着任意新源码版本可以加载旧fit。跨host使用同一实际源码归档及LF规则，跨版本恢复用原执行源；不覆盖producer或删真实性门。源码冻结包含 `.gitattributes`。

单连续专家在合法held上数值不可识别，应只使该专家该decision不可用，保其它独立专家及完整requested分母；严格混合继续要求所有声明专家，不重新归一权重。artifact/input合同错误仍传播为真实失败。该接线已通过两专家及实际independent-prefix调用链，原生进程终止和资源异常仍由B14处理。

B05使用row-major vec(B)，B09使用column-major vec(B)；未来组合必须用真实坐标置换同时转均值、factor/precision与布局，private逐导AR不冒称shared temporal×nativeΣ。B09低层Gaussian诊断不签发新推理授权；生产接入仍由PredictionDecisionView/CalibrationPrefix核participant、只读和C重叠。

## 执行规则与整理

日常科研使用20260918一个完整source→target seed：四个source程序各自participant选择与fresh全源refit，初始八个逻辑fit，再接两个公共目标面板。20260918/19/20三完整程序只用于大版本校验。单seed参与者区间条件于其冻结程序，不能解释完整训练随机性。

原始EEG、cache、唯一模型/最终预测及并行WIP保留。已结束微冻结、失败日志、重复checkout和一次性assemble_source已退役；上轮可恢复成果见[域迁移与BrainSync大版](domain_transfer_major_20260920.zh.md)。本轮每个实际执行阶段只保必要源码身份、关键反例、结果及一个紧凑结论，落地后收口过程副本。

必要本地执行来源见 `tmp/research_line_20261008/VALIDATION/windows_diagnostics01/`；独立审阅与上游合入receipt见同轮 `REVIEWS/`。当前记录分别说明软件、机制和效能出口，不将全部15块/123愿望或公共效能标为完成。

[真实本地机制](research_mechanisms_20261009.zh.md)收B06/B10独立数据审阅与B09完整六格、两actualsource和原基线复用；八人48格已用noise-tied/diagonal-I/Pz4及native/Cz分解贡献，不重复等价CAR格；Pz4未过NLL开发门，全8及七人新variant读数仍是公共开发。

B07有限history参考保完整joint label states及shared β crosscov，以两个生成marginal的比更新prefix后邻状态权重。必要promotion/跨时间相关与固定currentprior反例5项通过，默认未知overlap拒绝未放宽；dense每state参考仅适用小矩阵，外部完整history/actualtime/source授权与生产保存接入尚未完成。

L1旧source-prior selector的nested spawn已在 `64481f0` 复用现有 `install_input_admission`，传父进程实际proxy和taskID，不另建资源owner或改变source科学配方。精准caller SHA `d7f4df09bf2d347cea150f5132e08180b8195669a7fff2fd70135a4c37d1e0bf` 与test SHA `72b967f557bddaac675c2b459b163736f2048d9ca58eff9d8386f9e34a2fcaa9` 均等于上游实际执行字节。两项最高selector接缝零fit探针原2失败、修后2通过：真实双spawn共享snapshot账，normal归还与native exit73后父 `retire_task` 回收分别验证；committed JSON继续收费。root只读复核、Ruff、编译和diff检查通过，未重跑这两个绿色case，未把替换的科学body当效能。一次最终整合suite与原pending三项native组合仍待足额内存环境，不重启已关闭云实例。

ERP比较入口漏改的 `_array_identity` 已在 `ecfcba5` 删除私有import，直接消费共同 `data.digests.compact_array_digest`；原encoding/保存label关系保留。实际旧入口ImportError、新caller仓库外help成功及最高parent标签dtype攻击验收各保来源；不恢复alias。静态遍历当时459个Python文件的4665个仓库内显式from-import未见其它缺失符号，不能替代完整运行套件。B07实际模型adapter已进入research owner；B11 projected尺度参考按独立数学/迁移delta审阅落地，源/test及17unique实际case/6未执行分别记录于[机制证据](research_mechanisms_20261009.zh.md)。
