# 愿望清单：当前科研进展与下一切片

2026-10-10。原愿望清单123项、B00–B14共15块继续全部保留；本轮以逐项状态核查推进，不能把已有设计或单个数学参考计为整体完成。原需求和依赖仍由[原清单](../assets/deep_research_plan_20261005/wishlist_source.zh.md)及[分块规格](deep_research_plan_20261005.zh.md)持有，当前逐项出口见[进展绑定](wishlist_progress_20261010.json)。

整合树为`C:/Users/chent/.codex/worktrees/wishlist-integration/n2p3-net`，分支`codex/wishlist-research-20261010`，起点`7bd984a`。原dirty主树、公共accepted496、历史M1/48/1728格及真实C模型按原身份保存；本轮科研不在运行中的来源上覆盖代码，不重复清理。

## 已验收的有限成果

| 研究块 | 当前交付与实际owner | 验收范围 | 尚缺出口 |
|---|---|---|---|
| B00置信度 | 原`candidate_evidence/candidate_metrics`及原报告consumer；最大置信度、错时唯一赢家置信度、层级分箱ECE、实际J的uniform NLL contrast | 两轴独立接受；91项整合树定向检查通过；不改变原hit、coverage、NLL/Brier或配对门 | hit接近时的选择程序、正式uniform/wrong-confidence阈值、binary诊断及新效能 |
| B09固定graph | 原`SpatialSignalPrior`的expected energy与energy-matched graph；原posterior、保存重放及B05因子 | 固定外部graph shape、同C/noise centered energy；33项定向合同通过；完整receipt一次冻结 | learned d-prime/组层级、新公共效能与跨人稳定性 |
| B07有限响应状态 | `research.latent_response_reference`复用原joint Gaussian QR；完整selection amplitude/latency/width与proper local deviation | 原独立数学接受后迁移；root逐读差分，25项dense/clock/端点行为通过；零状态消费保存matrix | 多channel真实caller、完整history/输入权限、持久包、连续latent精度与公共效能 |
| B07 cubic字典 | 原`dictionary_artifact/response_dictionary`的fixed degree3、open-clamped uniform B-spline free/hard value | 两轴独立接受；最终2652 symbolic scalar与8项真实软件caller验收，保旧Gaussian/cosine `/2`语义，cubic独立 `/3` | off-grid导数、hard slope、raised/multiscale/phase/wavelet、crosshost与新效能 |
| B03真实OOF | 原`_grouped_temperature`一次`C\held`评分后捕获完整概率，原FittedEnsemble持有并保存 | 两轴独立接受c5ba body；K13/完整组标签12、selected子集/不可用control、同一final包及数值/温度/source关联验收 | inner模型尚未持久重放；严格nested选择、gating及真实公共效能 |
| 条件评分数值 | 原`channel_noise_mechanism`的逐操作正算术界和local/global minimizer cap | 精确接受611b body；71项投影/物理后验/评分合同通过；精度预算和概率模型保留 | whitening/QR/SVD仍是ordinary estimates；非全输入区间证书；新body未重跑7975×8 |
| B11角向机制 | 五份原A01授权C的逐记录proper noise contrast | 原独立审阅接受；radius²/d为0.938–1.026、lag1角向比0.8056–0.8192；没有held读数 | temporal shape、ERP span与MAT上游处理仍是竞争解释；同C SD不是校准p值 |

B00原M1后处理保留原source与完整旧metric/contrast：AR0错时平均置信度约0.99118、NLL比log36高21.6423；AR16/32 profile虽hit为0却有较小ECE，直接反证“ECE小即可晋升”。当前候选ECE明确依赖分箱与available概率支持；任何requested participant/seed完全没有概率，固定roster汇总为null，不能删除该unit。

B09同energy不等于同df：原两次真实C信号fit仍属于a59历史producer，control/graph df为106.76322/109.27214，不追签新owner。receipt修后f17是前置审阅来源，当前owner迁移另签自己的数值与软件验证。

数值切片的local bound保QR optimality defect及`E.T r`项，取local/global两条合法cap的较小值。原3749的两个7975×8 synthetic候选通过记录仍只属于3749；611b修复正商下溢并只新增定向域/连接证据，本轮迁移不重签旧长记录。

## 正在整合与下一科研

B07有限selection amplitude/latency/width law已迁入明确research reference，复用`joint_response_scale`积分一次global beta与独立proper local deviation。物理支持、whole joint/prefix mixture比与soft/hard/free prior有独立反例；产品默认、真实完整history、包和public效能尚未由此完成。删除未消费的C设计镜像；local deviation直接使用unit-amplitude waveform，避免先乘amplitude再除回去的舍入。

B03已在已有`_grouped_temperature`捕获：每完整C组由真实`C\held`拟合owner评分，保完整OOF概率及producer/final温度/native layout/actualK关联，复用原FittedEnsemble数值包。fixed/grouped共享一份final数值态；新增OOF上下文另有model身份，原base数值身份明确保存。未保存inner模型，不声称可重放inner评分。private softmax gating原型仍是研究参考；六组不自动获得gating训练资格，nested选择与产品晋升另验。

下一批按实际依赖并行：W03.01多家族物理字典、W03.06早期/峰值/慢波子空间；W05.03强收缩通道d-prime与W05.06空间组层级；B11 fixed-input/physical-prior下的temporal-shape与expanded ERP span诊断。尚无代码/数据出口的条目保持partial或in_progress，不以子代理任务派发标complete。

多家族字典原型与owner草案经独立接受；随后cubic B-spline已进入唯一compiler、表达准入、实际C-prior/容量/continuous/OOF和保存链。新cubic仅固定degree3/open-clamped/uniform/free或hard-value，声明独立rational语义及 `/3`；旧Gaussian/cosine `/2`保存数值与producer关系保持。每scalar只计算至多10个局部Cox节点，删除整width逐列重复计算，无新compiler或全局cache。非零shift/width的cubic off-grid参考明确unsupported，不掉进Gaussian解释。

raised-cosine与独立coarse/detail chart仍只有数学原型，生产compiler/callers未实施。重复基会改变先验，所有尺度并列可秩冗余；Greville早/峰/晚组有精确time overlap，不当作独立时窗或独立后验。raised原型是sequential rounding；free全局causal crop可有端点jump，atom自身与hard-value延拓的C1边界分别记账，wavelet仍pending。

B11逐记录投影比共享beta的joint contrast额外丢弃512个信号一致性方向。恢复这些方向后，各record独立scale一般不与joint投影交换，所以不能直接套用原均匀球面角向law；必须分开局部scale不变诊断与shared-coefficient consistency诊断。

新source-bound consistency诊断的energy/512为7.51698；独立有理数证明local+consistency才等于完整joint projector，不把512方向当全部joint残差。

后续固定输入shape×span实际面板先声明spec/source，再完整执行20/20格及10/10对照并获独立接受。原saved basis延迟一采样点保完整末event `[7668,7975)`；新增16时间方向可解析，无额外读取或弱列删除。AR8扩展吸收1.19%–1.76%原contrast能量，lag1仍为0.8039–0.8177，未解释主要角向失配。外部AR0刻意保持同AR8 innovation Sigma、未估其marginal scale，不能按大半径否定完整AR0程序。随机对照只匹配rank/rawnorm而非白化频谱，故吸收能量差异也不唯一确认latency。新增response方向的proper prior completion仍未提供，本片仅诊断，没有新fit/held效能。

uniform风险门的两种政策参考已获独立数学接受：分别观察全部暴露中的高置信错答频率，或错答条件下的高置信尾部比例；阈值须预声明，未触发不授予晋升。辅助总体推导只给无界NLL的风险下界，不给安全上界，独立单位为participant。原owner允许的轻微负stored NLL在该`[0,B]`参考中明确unresolved，原point NLL不clip。当前OOF模型排除自身held标签，随后使用全部OOF标签拟合温度仍引入依赖；完整选择、温度、阈值与gating必须共同进入nested程序后才能验正式门，W08.04继续partial。

空间d-prime/组层级私有参考及后置修复已获独立接受。已复现的destination basis `Phi→2Phi`关系缺口由唯一effect-definition入口补齐：明确basis/window/grid/rate/operator/dictionary，source群体和destination必须核同物理定义；各participant的measurement noise law仍可不同。原19项及六个synthetic fit保原producer，修后16项只验新增binding关系，不拼成新body的全路径结果。`learn`仍unsupported，完整factor/chart、canonical数值合同、source独立权限与效能继续pending。

固定收缩权重不能界定异常值影响；独立scalar反例随异常效应平方增长无界。原2087倍读数比较的是不同人数/权重的两组合成样本，不能当同cohort纯异常效应。source mean-only与separable covariance completion是两种实际不同程序，新增人际方差与均值不确定性须区分，不把d-prime标准化向量直接按reference线性推送。

## 来源与验证

本轮使用项目venv、native1和受影响contract/数学反例，源码逻辑由root逐读。完整验证路径、原独立报告、本轮源码及测试哈希记在JSON；source/数学参考/真实开发/产品四层各自签收。没有重跑旧矩阵、长记录probe或整仓全集，也没有新成人/设备效能声明。

公共云accepted496完整单seed的来源manifest为`04ff8a07…`、archive为`4b075a2d…`，恢复后的121项必要Linux合同和actual mount probe通过，训练终态由云窗口另验。本轮新科研树不自动替代它的producer或扩大请求矩阵。双方只同步重要交付、共享接口和阻塞。
