# 当前公共目标程序：损失、概率与实测空间

本轮按当前独立前缀、多专家范式重新设计，旧剂量/60或90轮/大参数组合不作为必跑任务。source三seed的独立选择/refit保持自身程序。目标transfer与同结构scratch固定32轮、batch64、lr1e-3、weight_decay1e-4，仅初始化不同；这些是预声明有限开发参数，batch64来源为当前设计而非用户曾明示64，不称已测最优。

## 实际训练风险与概率

新`trial_loss_reduction=calibration_balanced_fixed_mass`保留C内balanced相对权重β=nneg/npos，Z=(nneg+β*npos)/N，每batch损失为Σw CE/(实际batchlen×Z)。旧batch-local class-mass分母会改变小batch风险，独立有限总体/解析梯度反例已验证；旧程序按原身份重放，不追认为新程序。固定模型参数下均匀无放回子集估计固定C风险；reshuffling的自适应逐步模型不能因此宣称每步条件无偏或有限NN已经校准。OOF的β/π/Z仅由该fold训练标签产生。

`FrozenTrialPredictor/2`保存实际fit风险、recipe、rows、state/input-normalizer与source合同。一份forward给raw、LLR、posterior和p：目标NN posterior logit=raw−lnβ，candidate LLR=raw−lnβ−logitπ；源K0使用真实源mixture prior。经典prior由实际fit保存。预测是声明模型的forecast，非真实target posterior的实证保证。旧/1保留原ranking/candidate分数，拒附加新proper合同。转换在load一次验证/编译，避免逐batch重复源合同哈希。

## 物理视图、温度与公共输出

raw C/OOF/held由同label-free owner投影实际BNCI原8导到Cz6：Fz、P3、Pz、P4、PO8、Oz各减实测Cz，A为6×8/rank6。只向raw模型传实际observed非零独立行，不补零制造SPD。原8导bytes、参考和physical support保留；derived numerical digest另绑定A。native8 raw为单独信息控制。raw仍native256Hz/unfiltered，trial为128Hz/179 finite-zero none，不能将不同时间输入解释成纯模型反差。

主候选机制类权重为`(T+R+(Toeplitz+xDAWN)/2)/3`，scratch仅替换T形成匹配对照，不进入主pool重复计票；waveform LDA独立控制。`temperature_expert_ids`只对预声明primary做C内OOF，wave/native控制为fixed T1。任一requested primary失败仍令整个grouped不可用，fixed及完整请求分母保留，不能按成功subset重归权。两个温度程序共享一次all-C实际模型。

binary输出实际posterior NLL/Brier、三种logit及概率artifact，池按声明权重在log域算术混合，绑定单输出SHA、共同physical/numerical comparison与行标签。三配对seed的确定性经典控制共享fit，并完整传播概率状态与风险；不能只使首seed有概率。四库没有raw二分类adapter，不造candidate ID来填面板。K0只有sourceNN可用，要求target-fit专家的pool明确unavailable。

## 真实来源门禁与验收

真实source保存非空执行身份，旧K0/OOF只有不同入口描述，曾被checkpoint拒绝；合成无执行身份source未覆盖这一路径。共享owner现在核原source完整receipt、实际nativefs的SOS/pad及target执行profile/operator/grid/ref/dtype/bytes/mask/spans。K0为已编译未来能力，不能伪造已执行凭证；OOF必须有实际选择执行证据。source fullrecord与target有限支持是相同数值程序族的不同有限算子，不说同一个F_N，也不以None放宽。future capability还绑定真实NN有序训练导与几何。

原fd62执行相同β1真实BDF源checkpoint产生合法K0拒绝RED；当前accepted源码以同一原producer、真实A01 K0/OOF240行/冷cache/load和概率重放GREEN，错误receipt与错误训练导能力被拒，target fit0。原source95c933不改签。另原A01 MAT SHA cdf9bca4…，C200/Cz6 rawAR8/Burg/fixed32/ridge1仅一次fit/save/首合法held重放成功，原执行ce9b…单列；不是完整公共收益。

软件固定质量损失、解析梯度、最后partial batch、空间/A/rank、实际小NN状态重放、三seed binary15cell/9fit/6pool、温度失败隔离及来源/版本反例均通过；独占持久目录保存必要证据。Spec与Standards零未关闭关键项，作者commit551ec33、主树整合6295cde。[质量/两轴/联合凭证](../assets/current_public_consumers_20261008/integration_receipt.json)与[439成员消费者冻结](../../frozen/current_public_consumers_20261008/source.manifest.json)绑定e6499b…，旧探针保持原producer。

主树另合并吞吐与scope前置，生成442成员2394ca…执行冻结；用真实原BDF源/A01完成一次零fit联合K0/OOF/冷cache检查和概率重放，未重复NN训练。完整十二source、四库binary和BNCI真实membership效能继续实际运行验收；15块其它机制和新成人数据确认未由本软件交付完成。
