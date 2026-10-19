# 响应模型、空间先验与噪声数值

## Source群体学习、联合噪声与完整历史

源baseline的条件Gaussian算术已独立推导并整合：给定已观测prefix，减去每record首样本只平移其flat截距，proper响应先验保持；prefix的压缩右端直接进入tail增广QR，右下残差根保留小的平方和。固定小矩阵的精确scatter为8，新算法得到约8；两个record的独立有理数profile目标与NumPy/Torch及导数吻合。Burg固定形状复用同一次评价，profile复用已核终点，初始化、预测cache、最终kernel统计和physical source block共用该坐标。代码关系、公式及反控见[条件算术整合](../assets/research_resume_20261012/source_conditioning_arithmetic_integration.json)。

完整source运行暴露的内存峰值随后按同一数学关系处理：每份记录先保存完整增广二次型的小R，再一起求解，所有kernel、各record漂移及response方向保持；密度仍按原实际观察行数归一。独立原始Gaussian有理数oracle核对变长record、非零proper prior、scatter、行数与后验；把压缩root行数误作观察数、丢掉response residual或拆成独立kernel三个负控均被区分。原P1布局的全局81列QR最多使用130行根合并，record本地49列白化仍按原数据执行。原错误中的2918282仅记录为allocation行数，真实tail分母从原record支持派生，见[内存生命周期整合](../assets/research_resume_20261012/source_memory_lifecycle_integration.json)。

source、joint、history三切片已按精确前像组合，原数学证据依源码/AST关系复用，见[整合绑定](../assets/research_resume_20261012/source_joint_history_integration.json)。source每人拟合一次并导出同次数据likelihood；留人折复用各个人包，源均值不确定性与人际差异分别保留。固定两曝光参考由各折其余源人建立，个人后验不作无噪声观测。

真实完整源已取得独立执行接受：13份个人数据因子、13套各用其余12人的LOSO forecast，以及原消费者载入均通过。原234条请求中213条、101,388个标签进入拟合，21条排除仍在完整账本；31导的496维响应方向、各496列的between/mean因子保留。实际冷读核13组baseline、noise和stationary moment，原执行树退出0并退休；验收新增拟合、QR、原EEG和目标读取均为0。12份原个人产物复用，缺少的1人沿原配方补齐；原producer与两次检查器拒绝均保原证据。完整目标矩阵直接消费这份[源完整接受绑定](../assets/research_resume_20261012/population_source_complete_acceptance.json)。

private AR1的新score先在完整后验mean/factor上形成创新，再累计二阶量，避免两个巨大raw二阶矩相减。完整startup、full Q、shared/local列及物理flat漂移保持。真实C8原返回点的梯度约3.741310，与独立objective有限差分约3.741309一致；这支持从原返回向量继续具名数值warm start。优化停止仍用原gtol=.001、maxiter=2000。

完整history以原record观测并集构造全候选目标，Gaussian算术按tile计算，原离散sum-product、proposal、SMC及trace `/2`保留。实际组合已核20个owner的导入来源，新同字节codec控制通过；真实拟合、首题保存重放和完整矩阵的结果沿各自实际producer登记。

## 原保存C噪声的真实复用

H1的K200/K800已由当前原source/C owner实际重新准入：分别200位/47正例/68570行，以及800位/188正例/171569＋115892行。原固定source均值、完整随机因子、均值误差、残差与噪声参数关系一致，source/noise/响应C新拟合均为0；两格保存关系和数值重放完成，各7项权限反控通过。

完整执行165.46秒，峰值RSS1.15GB、整机CPU最高42.9%，数值线程2；原完整noise加载约6.26/5.97秒。该复用保原fixed-source-mean开发对照身份，新G0/G1由各自联合校准程序给出参数。原producer、独审、真实执行及canonical关系见[整合绑定](../assets/research_resume_20261012/saved_public_noise_integration.json)。

## 人群先验与字符变化的通用核心

人群层直接消费每个人的测量likelihood，预测新人时把“人本来就不同”和“人群平均尚有估计误差”两组完整factor拼接。原个人posterior里的先验不再当作新测量；F-order到native C-order的列置换、绝对likelihood常数与每record观测范围明确保存。个人compact数据包在不同留人fold间只引用一次。

字符局部Gaussian变化与shared响应分开：同一次QR积分局部变量并保其条件后验，共同响应和record漂移跨全部观测保持同一变量。完整残差二阶矩保shared/local的交叉项，供联合AR1/full-Q噪声校准使用；两曝光GLS reference在目标优化前固定。原有Gaussian-local 37+9项、人群17项、数据导出8项与各自独审按相同源码复用。

canonical新增三项组合检查：原小数值模型的均值/完整factor逐bit冷读、两个fold共享相同个人对象；人口factor与local conditional组合对独立dense Gaussian最大误差5.55e-16；完整九候选C标签因子保8个真实零似然、候选先验只进入一次。新检查未读真实EEG或拟合模型，核心源码逐字节整合；真实public factory与完整held程序由各自执行验收。来源见[核心整合绑定](../assets/research_resume_20261012/general_response_core_integration.json)。

## 共享函数与字符变化的机制比较

完整Gaussian系数描述一个人跨字符共享的响应不确定性；给每字符新增零均值函数效应，才产生字符间独立变化。其跨字符covariance保共享项，字符内再加local项。独立反例表明，即使每字符总variance相同，错共享或错独立仍改变J2/9/32/36的完整候选概率；同一C观测乘两次也会虚增精度。Gaussian functional变化与exact时间warp有不同分布和支持，完整一二阶矩不等于warp mixture。

总监据此选择G0/G1作为第一项机制实验，分别以一次授权C联合定标响应与噪声；旧fixed-source-mean residualnoise另保开发baseline。原B07非线性delay/width四臂继续保留，H1的8.273秒间隔只用于law冻结后的支持核验。研究问题、七类数学反例、proper warp候选、联合marginal及可复用代码图见[科学裁决绑定](../assets/research_resume_20261012/response_law_science_direction.json)。

G1的外部物理reference以原dictionary和noise定义两个独立、零漂移的unit响应记录，non-target设计`[Phi,0]`、target设计`[Phi,Phi]`。若单曝光GLS误差根为F，则完整common/increment reference根为`[[F,0],[-F,F]]`，保负crosscovariance且不随source重复数作1/n收缩。它是prior的物理刻度，实际within尺度由授权C联合估计；原个人posterior F不作生理within协方差。该reference的完整秩、坐标pushforward、proper logvariance policy与新fit来源由具名联合校准owner保存。

## 有限输入算子的审计复核

实际backend复核纠正了两项笼统解释：纯带通legacy与新路径同用MNE的2192-sample延拓，人工输出逐字节相同；只有含50Hz notch的legacy显式SOS分支使用33-sample短pad，新完整SOS为2201 samples。四个EEG/HEOG人工控制中，typed EOG均保持原字节。source-origin与标称模型索引分别入回执，数值程序继续沿同一个source support与resampling owner执行。

351点、未先低通的70Hz人工短窗经FFT+edge-padding后，输出RMS比为−22.1978dB；整数周期500点、无padding的同频控制输出峰值约6.89e−14。两者反映有限窗口与延拓的不同投影；完整默认输入还包含连续30Hz前级。单位源端点到前26点的最大响应为9.5159e−4、均值为−8.90006e−6；以1mV端点换算，局部峰值约0.9516µV，均值扣除的全局偏移约−0.0089µV。这些是具名有限算子的人工控制，原真实数据和分母保留。

源码、独立微型控制和逐项裁决保存在`D:/BCI/artifacts/research-deliveries/repository-audit-math/signal-contract-review/REVIEW_RECEIPT.json`，SHA256为`bdf229c1af103ef77470a254d6be28b1a39dad665e793c1d34ce54d401567b0d`。本轮只修正合同解释；后续若选择新的边界程序，同步signal、template、noise传播与recovery的同一实际算子。

## 原生C均值的精确整数累计

实际热点来自重复事件的Fraction构造和累加。新路径复用原相对时间的dictionary查询，在共同有理分母下累计整数，再用原均值编译器收缩并正确舍入。三个完整C记录共356,031行×31通道的mean/error字节及原物理事件关系相同，原编译器cache命中/驱逐也相同。K800无profiler数学路径176.736→82.955秒，约快2.13倍；live授权与完整learner另按其执行计时。

独立9组Fraction插值/误差包络控制覆盖不同采样率、错开网格、重叠state、partial/blank、terminal半开边界及cache生命周期，70个约2^-1050弱项保留。完整测量prototype、稳定身份release与canonical导入分组修订分别保来源；计时不进入binding。每个新完整D的SHA等于旧residual的design SHA，在相同Y/I31/零nuisance程序下复用原noise参数与原fit身份。源码、独审及相等关系见[整合绑定](../assets/research_resume_20261012/native_mean_performance_integration.json)。

## 一般AR(p)平稳启动的创新分解

前p点服从原平稳联合分布，之后每时刻只加入新的跨channel创新。精确AR差分H满足`H K Hᵀ=diag(K_initial,Q,…)`，完整前缀根因而由两个小根和递推给出；实际保存根另用精确dyadic残差、checked inverse与原Decimal96误差验证同一law，保原门限和全部弱方向。p≥T沿原精确入口。

同一实际K200参数的新根构造7.129秒，原构造及准入阶段397.801秒。旧K200完整operand沿新consumer的原测量为6.290秒；root在canonical及当前公共adapter下再接同一原参数，加载与核验7.735秒，原record、初始31维根和operand身份均相同，完整992因子未重新构造。K800原root的新consumer测量5.380秒。新根是同参数law的新舍入producer，原根与旧执行记录继续保留。

作者27项新用例、4项既有codec例及独立手推AR2/重签错误law负控已通过；root仅增加canonical原参数接线及显式fit分派核验。测量对象为因子构造和加载，完整科研程序耗时按各自执行记录维护。数值范围、独审、来源和清理见[前缀整合](../assets/research_resume_20261012/markov_prefix_integration.json)。

## 原生公共校准噪声的完整拟合

原H2 source均值固定后，H1的K200、K800分别沿原fitter完成一次private AR1拟合：31条时间规律、完整31×31创新协方差、31维平稳初态及32点992维前缀因子均保留。原授权200与800个bits构造残差，η=1；两次fit/save为399.467、396.640秒，独立原codec cold/live核验为265.238、263.633秒。完整程序1057.886秒，进程树RSS峰2.339GB。

独审核原权限、全部参数、初态与992前缀的leading block，并以两通道手算例确认同record续接只用一次startup。source μ/F/prior前后相同，响应C更新在下一统一likelihood执行；原执行源码、完整结果、cold和清理均保其身份。来源见[原生公共整合](../assets/research_resume_20261012/public_native_input_integration.json)。

## 集合刺激的通用membership与缓存

物理响应现以原有序`membership[event,q]`决定target增量；一次集合刺激的全部成员各保完整增量，不除以成员数，common只加一次。known blank的membership全0，仍保共同响应；原未知中断曝光的`None`继续使用显式联合assignment及原质量。新public列轴由原history声明，旧BrainSync一热行逐项核candidate_id映射。

独审先复现两个实际缺口：原singleton equality漏掉集合中的成员；只修direct公式仍让template缓存错误复用，差分可假为0并错误取消误差。现direct/cache/difference共用一个membership owner。J9/32/36的全部q×两shape×两unknown状态对独立Fraction插值oracle核11088个标量，原216个一热数组逐byte保持。root另加四项手算永久回归通过，覆盖集合/blank及有真实目标信号的缓存差分；真实相同物理泛函仍精确取消。

原C issuer授权bit=1的数值贡献正控也已通过，两shape直接激励target列；该正控只消费原label权限和数值kernel，EEG样本读取、raw grant及C模型准入均0。原source参数codec/header保持，新plan/adapter另记producer，旧完整结果与恢复依赖保留。证据、独审、canonical验证和清理见[membership整合](../assets/research_resume_20261012/physical_membership_integration.json)。

## 源几何、实际电压差与完整参数冷读

源响应以真实共同电极的电压差进入目标坐标，其余方向保留明确的proper条件方差。新入口先合并真正要计算的泛函，再转换为浮点图像：source公共尺度为`2^60`时，两个条件残差仍具有方差`2/3`和交叉协方差`−1/3`；近共线源方向的差保留两个独立项及方差`2e²`。state-reference也在此时相减，逐项误差随图像交给后续数值owner。

三个canonical数学回归通过。实际保存源的496个潜变量加48个独立条件方向，共544维；完整source/B09/completion/dictionary由同原owners构造新canonical模型，在禁止旧作者及旧源目录读取的独立进程中冷读，三组混合泛函的exact矩、数值及误差数组逐bit相同。原九候选Gaussian桥另以rank3、16行和单一analytic contrast对照独立闭式积分，最大对数似然差约`1.07e−14`。各自作用域、代码规范迁移及原producer保持在[整合绑定](../assets/research_resume_20261012/source_transport_integration.json)。

零列source factor的一般边界另由独立标量反例核清：`x=5,y=1+2x+2z`，z为标准Normal，查询`x,y,2x−y`应保后两项方差4、交叉方差−4；完全确定的二维源也应返回完整3×3零covariance。旧tuple转置丢失零列矩阵的列数，返回三条空行并漏加条件创新。新查询行Gram自然保方形维数，Affine与exact-common消费者沿原数学给出一致矩。两个主树RED均复现、修复后连原三项抵消/弱方向控制共5项通过；完整transport/native及四种programme的保存、冷读、再次保存和原来源验证复用同一合法synthetic源codec，新增fit为0，18项关系反控拒绝。原producer与当前consumer分别记录，来源和清理见[零维潜变量绑定](../assets/research_resume_20261012/zero_latent_response_graph_integration.json)。

participant、原EEG字节与whole-study排除现由同一source owner处理，原native几何检查继续保留。实际原保存源的12项metadata控制验证了独立target、同participant、原source字节及整库策略；fresh source-data准入和raw/C权限分别由各自live issuer提供。

原Huebner participant2的C68已由原registered factory和catalog/data owner实际重新准入：20次source范围读取均在原`[0,19947)`内，68个onset/label及实际数组一致。旧ZIP中的213845字节marker与新提取文件逐byte相同；两套声明和三个实际reader/issuer关系复现原拟合prefix，唯一具名变化是声明aggregate `target_contract.source_sha256`。原source codec、模型、fit stage及原排除目标保持；新public data准入单独绑定。

规范入口要求显式artifact路径和期待SHA，原marker/prefix谓词与数据owner继续核其科学含义。新目标使用canonical合成EDF/BIDS issuer的真实七导几何、identity table和EEG字节；原31导native入口继续拒绝几何替代，新关系引用原完整model，compact绑定约5.4KB，无41MB新副本。七项排除/权限反例、实际target/cold配置验证及canonical正常namespace核验分别保存；cold参数不增加live权限。原29秒source实读证据复用，没有重新拟合，来源见[源准入整合](../assets/research_resume_20261012/source_admission_integration.json)。

C与held联合准入另以实际C3及新45次held录制执行：原source C68读取10次，完整24.63秒，拟合0。两者的实际EEG字节并集交原source排除；参与者、几何和同一声明scope共同核验。独审补齐了主EEG及伴随EEG文件的重复消费检查，防止不同session名把同一文件当成两份录制；原执行正例的字节集合已互斥，新窄反例直接区分旧错新对。保存关系另核inner held、outer union和原source一致，cold保持数值角色。原执行P250、cold修订与最终字节检查各保其producer，后续完整C-response/P2500程序单独执行，证据见[联合准入整合](../assets/research_resume_20261012/programme_admission_integration.json)。

## 精确Gaussian边缘化与完整任务开销

实际source的496个标准Gaussian潜变量经共同观测映射后，只有64个方向参与目标泛函；其余432个方向的正规密度积分恰为1。对实际因子图像的integer Gram作exact、无开方PSD分解，保留每个严格正pivot，再加48个独立条件方向，得到112维同一预测law。直觉上，它先算清“哪些随机方向根本照不到传感器”，再把这些方向解析积分；弱而可见的方向仍保留，1e−600正方差反例也通过。

模板与正确舍入加速保同一泛函。零M行、全零M和非可分泛函有exact通用路径；callback只提供affine时保原Prepared行为。七个正式边界案例通过，独审核数学、舍入、codec和泛用边界。保存描述符引用原完整模型并核SHA，原source时钟1000Hz与目标250Hz分别保留。

同一合成held-only完整45闪烁任务的544/112两程序分别耗时253.65/137.92秒，峰working set为2.93/0.884 GB；全部162个分支的P/F绝对积分及九候选概率在该fixture逐bit相同。两程序各有独立进程冷读，原producer快照保留。数学等价依据exact边缘化，计时与逐bit结论限于这份完整任务；来源见[边缘化整合](../assets/research_resume_20261012/marginal_response_integration.json)。

## 一次响应C更新到完整九选一

完整软件程序已连接实际公开source、合成BrainSync校准C、冻结七导C-noise、一次响应C更新，以及独立held完整45闪烁。每个端点使用
`J_E(q)=π(q)∫φ(u)L_C(u)L_H,E(q,u)du`，`E=P,F`：共同source只出现一次，C标签只约束C自身响应，全部九个held候选继续参与。C完整1375行同时进入P/F；held预声明prefix为2500/0行、full为7625/9875行。两个录制各自漂移，held两个噪声段共享其物理漂移，三段各保独立noise startup。

两种C形状×两种held形状×九种未知中断身份×九候选，共324个完整分支；每候选expected/observed均5次，45次completed及1次未知中断曝光全部保留。独立逐标量字典oracle核544项，最大差约2.48e−17，位于原误差界内。原noise参数和身份复用，source/noise新拟合均0，响应C更新计数1。

实际完整运行204.37秒、峰working set1.10 GB；独立冷恢复11.61秒，fresh原始输入父链278.03秒。三份Prepared身份、324分支P/F积分与发布关系逐bit恢复，六项live及两项cold权限反例通过。原始公开source与合成C/held构成此次软件证据；下一效能出口按冻结完整学习程序进行公共配对与成人九选一评价。完整C链和上节held-only计时属于不同工作量，来源见[完整学习程序整合](../assets/research_resume_20261012/complete_learner_integration.json)。

## 完整block与续段条件读数

在主规范声明的共同物理测度与正有限积分下，令`w_P=normalize(J_P)`。则
`normalize(w_P(q) J_F(q)/J_P(q))=normalize(J_F(q))`；把`w_P`换成原π，得到的是另一套续段评分。当前flat drift使用原Lebesgue测度，这个结论先按积分与归一化代数成立；proper条件生成law下，对应普通后验与条件密度。`J_P/π`对q为常数时两读数相同，前缀仍可更新共同响应及漂移等nuisance。

独立exact Fraction oracle覆盖J=2/9/32/36、均匀/非均匀先验及三种任意正证据倍率，共24组，调用当前真实`probabilities` owner，最大概率差2.22e−16。九选一等先验时，若前缀对某候选提供9倍证据、续段对各候选同样支持，完整读数保留9/17，原π重计权的续段读数回到1/9。该证据可完全来自零均值Gaussian的方差差；独立flat baseline积分为1，均值为零不等于对目标无信息。

同一完整观测只改变P/F切分时，whole保持相同，原π续段评分可改变；使用前缀候选权重的顺序计权恢复whole。候选重标号保持等变。这个区分性反例支持完整45闪烁采用全部观测读数，来源与下一研究方向见[完整任务科学裁决](../assets/research_resume_20261012/full_task_science_direction.json)。

## 共享物理漂移与完整候选积分

噪声分段和物理漂移分组分别声明。同一录制的噪声可以在暂停边界独立启动，而电压基线与慢漂移继续共享；各段likelihood先相乘，再对同一组物理参数积分一次，共同source及每个离散先验也各进入一次。独立四行绝对积分反例中，错误逐段独立漂移使完整九候选logp变化约0.08659；两段各一行在不同exact时刻联合识别基线和斜率，其积分体积为1/10。同一exact时刻的rank1数据由结构证据拒绝，浮点QR负责已识别空间内的数值坐标。

原external完整链为合成EDF/BIDS的90次completed与1次interrupted曝光、七EEG、两个private-AR2噪声段、一个14参数漂移组及rank3 source，完整枚举9q×2h1×2h2×9d=324分支。按每次曝光独立Fraction插值、完整段白化及增广QR的9条reference分支与新积分最大logZ差4.66e−10；两条额外非reference形状/未知身份分支逐bit相同。不同target-lock的C-only记录通过共同source改变当前完整logp约0.008754，独立dense QR差2.31e−14。保存积分、cold模型与原raw-parent重编均逐bit一致。原producer、原Q与结果保持实际执行身份。

canonical迁移的7个完整模块在规范化import、顺序和模块说明后AST一致，原outer其余15个定义保持。两项聚合回归实际经过新canonical四行绝对积分、弱候选logprior−900、联合秩、重复noise-ID、保存冷读及原parent重编；Ruff与diff检查通过。完整90次例复用原执行证据，新产品45次任务沿新manifest/issuer连接实际源与完整噪声。源码、原证据、独审及cmd清理结果见[共享漂移整合](../assets/research_resume_20261012/physical_frontier_integration.json)。

## C固定均值、七导噪声与响应更新

C首先为噪声估计提供残差，再在已冻结的噪声下更新响应。固定均值来自原source函数和已授事件bit；原`ResidualRecord → FittedChannelNoise.fit`拟合七套时间结构及完整7×7空间协方差，保留实际224×224启动root。方差倍率eta只由carrier应用一次，原source均值和F保持。噪声出口登记响应C更新0；完整learner使用原source prior与C likelihood一次，显式后验和同一联合积分可以分别表达这一程序。

新入口把字典内容和source时钟一起绑定到graph：相同列数但改标250Hz，或相同1000Hz却换基函数，均在mean编译前拒绝。exact查询缓存以完整有理行区分输入；`1/3`与`1/3+2^-100`虽舍入为同值，仍保各自误差。三项canonical聚合回归核这些边界、七导独立dense电压law中的eta和完整codec。canonical reader还在禁止fit下实际读取原C-noise，保同一identity、七套temporal、完整Σ/startup和eta=1.7。原1375×7 mean及其noise fit证据复用，未重新拟合。

新canonical callback与原producer各有自己的完整C输入身份；原strict parent replay继续比较全binding，重新加载参数只恢复数值。原fit/C输入、迁移AST关系、实际冷读、独审和cmd清理见[七导噪声整合](../assets/research_resume_20261012/target_noise_integration.json)。

## BrainSync原始权限与暂停时钟

canonical入口已通过三项以合成EDF/BIDS文件执行的MNE入口合同：无C标签时保完整九候选及全部会话请求，推理cap对隐藏target字段变化保持相同；部分C只用最早已授权事件，完整90个标签的候选约束需要两个实际segment的raw权限。七EEG与独立EOG逐数组等于原MNE读取，复制公开dataclass不会获得issuer注册。

两段原clock的边界相隔`501/250`秒，保留两秒真实暂停及一个采样间隔；row7750对先前completed和interrupted曝光的elapsed分别为3.5与2.7秒。完整请求为90次completed闪烁，实际物理账本保91次曝光，其中未知中断事件保九个选项并等待显式law。source/hash、实际前端调用关系及三项命令见[raw权限接受](../assets/research_resume_20261012/brainsync_raw_authority_integration.json)，本次为原输入读取、权限与时钟的软件验证。专属测试夹具30文件、2,951,652原逻辑字节已由cmd删除并核实。

本文按主题维护已复核的结论、代码接点和下一步。运行日期与源码身份留在各证据及绑定中；使用方式和工程规则由原六指南维护。

最终应用是九选一。响应模型统一写为实际事件membership驱动的信号，J列分别代表完整候选；改变J、重标候选或改变刺激集合时，原先验、观测likelihood与求和规则共同变换。下一推断核心以这套一般公式交付，主案例沿真实九候选刺激时钟、七EEG和独立EOG角色验证；公共任务各用其原词表、通道和授权。局部数值修复与软件状态预算的作用集中于工程可执行性，科研进展以通用推断、完整任务风险和实际九选一读数衡量。

## 通用候选求和与九选一协议

给定共同响应参数后，只把噪声白化后仍相互重叠的响应放在同一因子中，再逐个消去历史变量。直观上，先处理局部相互作用，把共同参数的不确定性留给外层积分。候选q在公开协议要求的多个block间共享，响应形状h按block独立；不同record共用响应系数，各自保留噪声启动和物理漂移。算法保完整状态，计算量由真实重叠图及消元宽度决定。跨record必须先合并likelihood再积分；独立积分各record会丢掉共同参数建立的联系。

结构化实现已用原保存启动root、各通道原PARCOR运算、空间root和噪声尺度核回完整原始Gaussian密度。独立小例覆盖短前缀、不同通道AR、重叠block、共享q/独立h、C-only记录及候选联合重标号，最大log密度差约7.11e-15。科研树集成后另以非零原始时钟、独立record漂移和非均匀先验对照原始密度枚举；两个新增合同通过，弱至log质量−900的候选仍保留其likelihood贡献。

完整协议数学主例保留250Hz、七EEG、45次闪烁、每数字五次、4秒完整响应及全部九候选×两形状。共18500个native样本，source因子112列；点上结果与18项直接原始Gaussian展开最大差2.33e-10，保存后输入/噪声身份与候选联合重标号均核回。状态图像实际17640000B，最大单份native设计2240000B；这些是数组组件读数。样例使用显式声明的数学先验、噪声和合成电压；成人命中率由冻结个人程序后的新记录评价。源码、独审、执行者和清理关系集中在[接受绑定](../assets/research_resume_20261012/structured_frontier_integration.json)，连续积分的实际出口沿同一主题补充。

### 完整概率积分与独立预采样

完整概率把共享响应系数和每record的漂移都积分掉。上述九选一主例已解析积分112个source坐标及14个物理漂移坐标：P/F各保留18个候选/形状分支，36项积分耗时87.06秒；保存的完整九维概率冷回读逐bit一致。解析核直接包含原Gaussian常数、噪声Jacobian及物理漂移坐标换元的体积项。

一般图采用共同正目标`Ψ=Σ_(T,q) J_Tq/s_Tq`，T为前缀P或完整观测F，各scale为有限正数。直观上，给每个候选分配积分注意力，包括概率很小的候选；最后乘回各自scale，恢复原概率定义。首次仅平衡P/F的样例中，赢家绝对概率差约3e-5，完整候选的最大log概率差达15.41；这说明完整九维精度提供了额外诊断。逐候选方案在已知解析proposal的九选一数值对照中将该差降至0.1253，该对照专门检验积分公式。

独立预采样版本从原物理坐标的proper参考分布学习proposal（已知密度的采样分布）及scale，正式采样使用另一个随机流，解析答案在正式结果完成后才构造。两个独立record、矩形source因子、C-only记录和共享q/各block独立h的真实图调用，whole/conditional完整候选最大log误差分别0.02364/0.02390；source/目标模型拟合均0。各重复运行的非归一积分先合并，再归一成概率；标准误是统计精度估计。保存积分重放逐bit一致。科研树新增的两个解析合同还核回跨1600 nats的P/F绝对积分、非均匀scale恢复及独立预采样校正。原执行、独审和代码关系见[连续积分接受](../assets/research_resume_20261012/continuous_frontier_integration.json)。

### 局部二次型与按记录组织的积分proposal

每record的观测设计、白化与局部状态关系只编译一次，保存共同项、单scope项和真实重叠项的二次型。共享q在pair中只占一个轴；图上的完整求和及source prior仍各一次。新九候选小例在12个相同参数点上，native调用与缓存最大log差2.84e-14；该例点调用由2.464秒降到0.03134秒，编译0.506秒。缓存还保存原flat QR/体积及拒绝状态，短P的可积性由原owner判定。

多record的proposal保共同u，并令每record的白化漂移满足`v_r=m_r+B_r(u−m_u)+L_rε_r`。这保留各记录通过u建立的联系，参数存储随record数线性增长。原物理漂移为`a_r=R_r^-1 v_r`，proposal密度加回`log|det R_r|`；独立预采样仅估计该数值分布，原likelihood与prior继续完整校正。三个新增合同以独立dense Gaussian核密度、抽样和pCN移动，另核弱候选绝对积分、保存codec及零source-rank情形。

新的12record完整九候选链已实际完成：source rank2、总连续维数50、含约−900 nats的弱候选。18个proposal component共76896B，同维度dense均值/平方根为367200B；局部likelihood cache共94080B。native与缓存点上最大log差1.48e-11，独立预采样后的whole/conditional全候选log误差分别0.11643/0.10852，均保统计估计身份。冷模型重放含完整proposal密度；原numeric-parent重编12个record后逐bit一致，重放拟合和重采样均0。这一出口使用显式合成数值输入，真实采集权限由后续registered caller承接；来源同[连续积分接受](../assets/research_resume_20261012/continuous_frontier_integration.json)。

完整数值proposal现可加入预先声明的Student-t厚尾分量：共享head与全部局部漂移共用一个Gamma尺度，离中心较远的区域仍获得抽样机会。实际密度保留原物理坐标的Jacobian，正式权重继续按原目标密度校正。对每个原flat已识别的完整有限模型，所有Gaussian分支的联合precision正定；厚尾t给出有限的目标/采样密度比上界，从而保有有限二阶矩结构。数值精度继续由独立重复的标准误描述。

新窗口复用原12份Q、原pilot及所有候选scale，加入质量0.1、自由度4的t分量，以seed618033执行24粒子×3重复；native/Q编译、pilot抽样及科学拟合均0。whole/conditional全候选最大log误差0.12315/0.13131，冷模型重放逐bit相同。采用依据为一般尾部覆盖与有限矩结构，原Gaussian窗口的结果和producer完整保留；同一[接受绑定](../assets/research_resume_20261012/continuous_frontier_integration.json)保存当前源码、原测试复用关系和独立审阅。

## 实际31导联合校准与保存root准入

B11联合模型已在原Huebner记录的500个授权标签上完成采样。直观上，它让各记录共享一套响应系数，同时分别估计噪声尺度。原128次预热、128次保存、两条链保持；独立重算全部622坐标的最大Rhat为1.0413，最小bulk ESS为52.14、tail ESS为98.97。Rhat衡量两条链是否走到相近分布，接近1较好；ESS把相关样本折算为有效独立样本数。三项均满足预声明1.2/5/5政策，原C500模型和样本已按完整fit绑定复用于后续OOF与future程序。source仍是唯一原C68拟合，participant2供source、participant1作target，属于同研究内排除目标参与者的设计。

这次收紧的是数值误差证书：用实际存储的Kronecker坐标、小正交见证、分块误差的向上hypot及漂移残差，计入真正进入公式的误差。点数组、先验、噪声、初始化及采样政策保持原定义。独立高精度oracle、实际采样与原包逐字节复用分别留证；完整预测效果由后续独立future读数给出。

前两个OOF（按时间只用更早校准组预测当前组）已完成并保留原字节。首个OOF的32个候选分子都由同一保留样本主导，分母约有256个有效贡献；第二个OOF的分子贡献ESS也只有1.47–1.53。两者均通过原采样政策，K68的全部559坐标与32个配对概率函数已独立重算。对首个OOF移除主导样本、分链和半样本后，赢家相同，任意候选对的对数赔率最大变化4.6725。对数赔率是两个候选概率比的自然对数，用来观察接近零的备选概率变化。

### 连续噪声尺度积分：已完成的单记录对照

独立积分完成原K0单record、完整32候选的64项P/PT积分。λ表示噪声精度，越大表示噪声方差越小。原前段P保留样本的λ范围为0.6676–0.6900；完整P+T积分的主要贡献位于0.7451–0.7475。直观上，后段观测把主要权重推到了原抽样范围之外，因此检查已有样本的分链/子集一致性还需要配合连续支持审计。

[贡献范围直观图](D:/BCI/artifacts/research-deliveries/joint-predictive-integration/lambda_support.png)对照原256样本范围、连续P与P+T的32候选峰值范围；图、底层端点表及来源manifest保存在同一稳定交付目录。

| 读数 | 原有限样本 | 连续积分 |
|---|---:|---:|
| 第一候选 | position:12 | position:12 |
| 第二候选 | position:13 | position:4 |
| position:12相对position:4的对数赔率 | 145.0415 | 164.6557 |
| 同一T测度的绝对对数密度 | 6634740.6485 | 6635691.1062 |

全部候选对的对数赔率最大变化73.9175，绝对对数密度增加950.4577。赢家保持、备选排序和相对权重改变，这把改进重点定位到了预测积分的覆盖。原有限样本结果保原producer；新结果接受范围为单记录/K0/current-only原连续law，来源见[预测积分接受](../assets/research_resume_20261012/joint_predictive_evidence_20261016.json)。

原C500授权内的marker读数已接上这条预测：首条真答案是position:9，原/连续NLL分别409.5816/470.3023，赢家position:12保持。第二条已完成OOF也为高置信错答，完整两条及C-only校准方向集中于[校准与选择](calibration_and_selection.zh.md)。后续同时推进正确积分、概率风险和响应/noise匹配。

计算保持一个不随λ缩放的proper响应先验β和原physical flat漂移测度。令`m_H,q(λ)`为已积分β/漂移的原QR对数核，`x=logλ`，则`I_H,q=∫exp(((N_H+ν)/2)x−νexp(x)/2+m_H,q(exp(x)))dx`；`N_H`用原scalar Gaussian行数，flat行列式已包含在`m`中。whole law按`π_q I_PT,q`归一，绝对T密度还恢复原P积分与Gaussian/noise常数。该实际输入的64个积分已用保存小根的误差界证明logλ全域严格凹，各有唯一峰；三档容差的log积分逐位相同，最大source误差估计跨度为5.87e−5 nats，两端有解析Gamma质量上界。独立80位聚合与原数组身份均已核回。

快速模块已实现预先消去漂移并分解响应矩阵，让每个λ点只计算标量有理项。single-record/fixed-q数值API经root及独审接受，15个区分性合同覆盖原矩阵/测度、弱/零方向、双峰、能量与source乘积下溢及typed溢出；旧失败与新成功消费者分别绑定，见[模块接受](../assets/research_resume_20261012/joint_fast_lambda_20261016.json)。完整新32候选/64积分消费者已核实际调用与吞吐，原QR结果作为保存对照。多record扩展继续保持ONE global β，并按正确P边缘分布平均其余record的λ及未定候选；完整公共预测链沿各自新producer验收。

完整新fast消费者现已接受64/64（P32/F32）：相对原保存QR最大log积分差5.24e−10，whole/fixed归一log概率差≤9.31e−10，原绝对density常数一致。数值阶段观测76.800秒，两个native1 worker实用平均1.533核，单worker峰值297283584B；最终writer零积分恢复耗时1.163秒并退出0。原控制器因JSON布尔写出失败的OS1、partial与全部数值cell保原身份。完整source/128份worker-received结果与80位归约独审均已核回；此计时只描述这批单record数值，不拼接成完整B11耗时倍率。

原有限采样programme的7/7条OOF（K0至K408）已全部保存并接受，每条原模型、capture、坐标与32配对概率函数均经独立复核并满足原政策。除K68的分子贡献ESS为1.47–1.53，其余行仍为1；新连续积分另核其支持覆盖。K272/K340/K408的逐state概率正规化舍入经80位量化，归一log概率扰动上界最高4.622e−10，低于既定1e−8归约容差；原数组与producer保持。逐行读数和误差关系集中保存在[预测积分接受](../assets/research_resume_20261012/joint_predictive_evidence_20261016.json)。原程序已进入独立future队列，首项按原C24 embargo返回不可用，后续future与完整图继续。

首份可评分future的保存capture也已独立复核：两个真实record、622坐标、32个配对概率函数及贡献表均满足原政策。完整模型关联将在原程序保存附有七行OOF记录的最终模型后闭合；当前接受其保存样本和数值归约。该份有限证据的最大贡献约占99.998%，对应当前record保留的最大精度λ；后续支持检查按真实双record模型处理。下一项ordinal002保留原数值不可用：配对概率函数的Rhat为1.61766、bulk ESS为3.34924，原政策要求分别≤1.2、≥5；两策略保留同一失败和完整请求。原采样返回及日志已独审入账，后续请求沿原队列继续，结果和来源集中保存在上述接受文件。

ordinal003也已完成保存态独审：622坐标最大Rhat1.04425、最小bulk/tail ESS44.01/53.78；32个配对概率函数分别为1.00578/256/245.63，最大批次MCSE为0.00381，均满足原政策。MCSE衡量该次有限采样给概率带来的波动。原贡献表与独立归约误差≤9.56e−10，最大贡献约98%，贡献ESS约1.04。本次保存态审阅覆盖future000–003：两项可评分，原embargo与混合不足各一项；完整OOF模型关联、实际输出比较与后续进度从checkpoint引用的任务记录读取。

源码及独审还确认：当前C draws只用于P sampler的初始点；实际P law从原source prior与C/P完整likelihood重新构造，同record的C不重复计入。因此具备上述条件的新连续programme可直接由真实earlier-prefix/history权限积分，省去纯初始化用途的C/P抽样。七条新连续前向概率沿这些真实权限完成，旧有限programme继续其原实验。

完整7条连续前向程序现已完成并独立接受，K依次为0/68/136/204/272/340/408。共448个逻辑P/F项：首条64项精确复用，后六条384项按完整root/误差/来源去重后实际积分204次，0新fit/采样。全部新数值阶段退出0，whole/fixed归一log概率误差估计最大0.01448/0.02033 nats；独立80位归约最大差2.94e−10。新六行两次数值阶段累计242.26秒，原输入准备另计，来源见[完整7行接受](../assets/research_resume_20261012/joint_fast_lambda_20261016.json)。

七份预测全部封存后，原issuer才以C500授权恢复7个完整组的476标签和第二record的24个partial标签。两种连续law的argmax诊断均为0/7，whole/fixed平均NLL为319.3663/318.6237，Brier约1.999888。准确积分已经给出清楚的模型开发信号：原source→target的响应与噪声匹配、候选排序及概率校准需要联合推进。后续有限delay/width研究将固定原basis/β坐标和noise，对照平均shape、完整矩匹配Gaussian与有限state；概率校准采用逐题只看此前结果的程序。

响应形状新切片已完成四臂引擎及完整软件调用链。每个selection可有预声明的非负额外delay与正width，四臂区分原模板、平均波形、同全一二阶矩Gaussian和完整finite law；同能量并不自动有同容量。矩控制令`β=μ+Fu`、`A=diag(sqrtπ)−πsqrtπᵀ`，局部factor为`A⊗[F,μ]`；全selection共享原β，新增selection只追加其独立局部prior。该控制匹配均值/协方差和Gaussian容量代理，非Gaussian的高阶关系由finite臂保留。唯一law正规化、全矩factor、compact收缩及原native阵列的具名off-grid延拓均获root/独审接受，来源集中在[整合绑定](../assets/research_resume_20261012/research_integration_acceptance.json)的`source_informed_finite_response`。

直观上，这四臂分别问：每次反应严格沿原模板、只允许平均波形、允许同样幅度的Gaussian波动、或保留几种形状各自的概率，会怎样改变候选判断。原source响应先验只进入一次；噪声沿原保存参数取Gaussian ηK，漂移用每导的截距与每秒斜率。C0使用共同平坦测度，非空C保存完整后验，首次真实评分重建C的QR并核对原保存态。whole直接计算保存C后验下的完整新增观测密度，conditional保留自身的前段条件化误差门。

实际合成原MAT软件纵向已完成四臂、两法、完整模型冷加载、before-group因果选择、无future真值的胶囊重放及原source/target MAT父级核验；篡改保存预测的负例被拒绝。八个专家×评分法组合中七个产出概率：whole四臂全部可用，conditional的矩匹配Gaussian保留原数值不可用。要求四臂齐备的两个固定混合池中一个可用，另一个保完整失败；partial-C119的finite单臂父级可用，实际判错。这里取得的是数学与软件全链出口，后续真实公共数据比较沿独立来源准入和资源账本推进。

公共计算按真实尺寸安排。若每次选择有H种形状，S次选择的完整离散组合按`H^S`增长，再乘尚未确定的候选数；矩匹配Gaussian的proper变量数为`r+SH(r+1)`，其中r是原source factor的列数。当前软件样例r=16，八导、八basis的原配置对应r=128，因此公共入口先核实际样本数、选择数和候选支持，再安排共享source闭包、流式计算及完整预算。现有预算式估计数值工作区；下一资源账本补齐常驻C根、保留状态与元数据，并结合实际峰值内存验收。

源码还明确了跨完整future队列的主要规模项：即使nominal/mean/moment，每个未标注history selection的36个合法q也进入全联合求和，j个这样的selection需要`36^j`状态；finite再乘全部selection的shape数。精确边缘化已结束的Gaussian局部变量可降低每状态维度，离散q的组合仍需独立推断方案。下一科学切片将比较有剩余质量证书的树求和、完整当前q支持的逐步粒子/Gaussian近似，以及在给定共享连续系数时按时间支持因子图先消去离散变量；各方案明确自己的估计量、误差和保存身份。当前generic软件完整请求通过预声明状态预算返回概率或类型化不可用，公共完整概率目标沿上述全局推断设计推进。

两种评分的直觉区别是：whole用完整观测一起判断候选；conditional固定当前候选先验，再比较后段在前段条件下的证据。若前段P严格早于当前刺激、两法均有定义，因非负delay和因果白化，P对当前q不含信息，`p(P|C,q)`就是共同常数，两法归一后的候选概率相同。P已经含有当前刺激时，whole还保留该段对q的信息，两种程序可产生不同概率。C0的conditional要求P能识别漂移，whole可由完整观测识别；各自的密度误差证书按对应观测范围验收。

原shared `/3`保存root也已实现快速严格准入：用小因子见证核验原992维矩阵，误差界9.386e-12低于原3.150e-7政策，保留原root与producer。实际完整source codec加载4.254秒，其中root部分1.130秒；较早同输入加载的457.8秒中，452.3秒集中在该验证。新见证是consumer准入元数据，原模型的概率计算仍消费原矩阵。来源见[整合绑定](../assets/research_resume_20261012/research_integration_acceptance.json)。

## Shared PARCOR平稳启动因子的显式入口

`shared_stationary_factor`只适用于各实测通道持有完全相同stored binary64 PARCOR的law。exact-real step-up给出同一AR多项式，unit-innovation平稳时间协方差为Γ；以`t*C+c`的time-major/channel-fast布局，完整启动law恰为`K=Γ⊗Σ`。其正对角下三角平方根在精确算术下为`chol(Γ)⊗chol(Σ)`。private unequal AR仍由原mixed Yule–Walker owner持有，不能套此分离。

小root的binary64值A/B分别来自原Decimal96 Cholesky，协方差与保存root均作为精确有理数准入。令小root的近似逆witness为W，`D=I-WA`、`η≥||D||F<1`，则`A⁻¹=(I-D)⁻¹W`；exact rational计算`W(Γ-AAᵀ)Wᵀ`，再乘`(1-η)⁻²`得到`a≥||A⁻¹ΓA⁻ᵀ-I||F`，空间同得b。witness的生成精度不作为前提，其缺陷由独立exact乘法核验。

设`R₀=A⊗B`为保存两小root的精确乘积，则

`e₀≥||R₀⁻¹ K R₀⁻ᵀ-I||F = ||Eₜ⊗I + I⊗Eₛ + Eₜ⊗Eₛ||F`

可取`e₀=√C*a+√P*b+a*b`。实际存储root为逐entry integer-nearest-ties-even的`R=R₀+Δ`，并非假定Kronecker乘法零误差。以可逆精确2幂缩放的两小root坐标，逐entry有`|Δ|≤eps*|A|⊗|B|+u`，其中eps为`2⁻⁵²`、u为最小binary64 subnormal。于是

`r = eps * || |A⁻¹||A| ||F * || |B⁻¹||B| ||F + PC*u*||A⁻¹||F*||B⁻¹||F`

上界`||R₀⁻¹Δ||F`。各逆范数由`||W||F/(1-η)`上界；乘积范数由`|| |W||A| ||F + η*||W||F*||A||F/(1-η)`上界。所有矩阵乘、残差和缺陷均用Fraction exact算术；各平方根的binary64上界用exact平方比较验证。两小因子的归一与最终physical rescale均须精确往返，不增加单位变换误差。

令`F=R₀⁻¹Δ`，`R=R₀(I+F)`，则实际law误差满足

`||R⁻¹ K R⁻ᵀ-I||F ≤ (e₀+2*r+r²)/(1-r)²`。

此式要求`r<1`，并继续使用原`1e-8*sqrt(P*C)`准入政策。无PSD修补、噪声参数重估、降通道或缩prefix。数学根计算为`P³+C³`主阶，完整actual root物化/核对仍有必要的`P²C²`项，small rational数的位长也影响时间。

`FittedChannelNoise.from_shared_stationary_factors`显式发布新`/4`包；原factory/fit/default constructor保持原`/3`。新包保存原A/B、actual R、证书及三项原producer源码SHA；consumer核固定语义、精确law和witness，保原源码身份。load直接准入保存的prefix因子和TemporalNoise启动状态；tail的空间Cholesky沿原owner构造。完整startup/tail whitening、leading prefix、log determinant、physical prior和flat da测度沿原函数消费actual root。

独立小维测试从AR0/1/2闭式YW组装raw K，用有理Gauss–Jordan直接计算完整`R⁻¹ K R⁻ᵀ-I`，覆盖原单位与极小/极大协方差尺度；另有private拒绝、改签/损坏state、逆缺陷方向及normalization反例。独审再从AR3 companion Lyapunov方程独立求raw K，P2/P5的真实误差均受证书覆盖。新noise通过B11实际`compile_owned_flat_decision → save_joint_flat_state → load_joint_flat_state → predict_flat_tail`，保存重放时禁用fit、重新白化及prefix/temporal启动编译，预测的component logs和prefix weights逐数组相等。这些检查验收数值law和保存重放，公共分类表现由实际面板记录。

固定源码`8ca8dd15`以原31通道shared-AR1/P32输入实测：新compile 2.076秒、constructor准入1.459秒、完整factory 3.814秒、save 0.833秒、load 2.281秒；native1、峰值RSS 322.0 MiB。law bound为`1.6504e-12`，原政策为`3.1496e-7`。原stored AR/Σ逐数组保持，新root与原保存dense root最大差`6.7763e-21`，startup logdet差0；新包record和白化逐bit重放。原source JSON只读并前后核SHA，无新fit。原观察约689秒为不同次执行的历史瓶颈，不构造配对倍率；同进程相同AR/Σ的P8C4经各一次warmup后三次生成，中位dense 25.87毫秒、factor 9.69毫秒，约2.67倍。完整[执行绑定](../assets/shared_stationary_factor_20261016/binding.json)保存原输入、实际执行source SHA、阶段耗时和软件验证。

产品舍入项的反例使精确`2⁻¹²⁰⁰` off-diagonal乘积存成0，独立有理pullback仍受界覆盖。原政策附近的近unit AR和近奇异Σ示例由新producer明确报typed unavailable；保守证书未承诺覆盖全部原dense可接受输入。调用者显式选择新producer，原保存态继续按原codec消费。

## P1完整16格读数

完整16格已完成并独审：8人各一个held selection05、R10、36选1，两噪声臂共576个候选分数，seed20260918，新增fit及response-fit均0。科学exit0、所属树0；[完成绑定](../assets/research_resume_20261011/p1_cloud_completion_binding.json)保原producer、新consumer、结果和两路独立复算。此处覆盖下文“A01已恢复、余14格待评”的阶段状态，原失败面板继续保原身份。

| 噪声臂 | 正确/请求 | hit | NLL | Brier | 请求覆盖 |
|---|---:|---:|---:|---:|---:|
| shared8 | 5/8 | 62.5% | 7.336477 | 0.679094 | 8/8 |
| private8 | 6/8 | 75.0% | 5.554012 | 0.477678 | 8/8 |

private仅救回A06，其余七人命中状态相同。A01贡献约71.82%的NLL改善；其真值概率从8.72e-22变为2.45e-17，两臂仍为近100%置信错误。Brier改善约75.71%来自A06，真值概率从0.203527变为0.925032。因此当前研究优先级是高置信错误尾部与C内概率校准，并用多selection/类别扩展固定响应噪声对照。

新旧stationary law的16格赢家全部相同，最大归一化概率变化约2.45e-8。private原密度修正接近候选共同偏移，softmax会抵消共同常数；quadratic与predictive logdet实际改变。A06救回在旧law中已经存在，属于噪声臂差异。数值修复完成了initial-state正确性、保存准入与完整执行闭环，其效能评价继续绑定真实候选概率。

原观察后八人配对bootstrap的95% Δhit区间为[0,+37.5]个百分点；仅一个discordant pair，精确双侧McNemar p=1。每人一项且八项真值均为同一候选类别，下一实验预声明多selection和类别。2026-10-13按保存报告舍入表的[派生](../assets/research_resume_20261011/cloud_completion_readout.json)保原范围；2026-10-14已恢复原P1结果、新16份score/receipt及旧参考`091fa663…`，核SHA、同decision_plan/roster、物理输入和原response/noise来源后独立复算。新576个log density重建概率与原指标一致；与旧参考log probability比较，16/16赢家相同，shared/private最大概率差分别`3.6915e-11`/`2.4476e-8`。新旧平均ΔNLL为`−5.8117e-11`/`−8.9791e-8`，数值关系与原比较一致。原拟合及producer保持，旧参考只保存归一化分数；此次直接核概率/赢家关系，旧绝对密度分解仍用原记录。来源及零新增拟合/模型重放范围见[恢复与复算绑定](../assets/research_resume_20261011/cloud_payload_recovery_binding.json)。

## 当前模型与数值任务

信息界诊断的`/2`已完成数学与状态审阅：保同C估计mean/inverse-shrinkage-covariance的raw plug-in d²，并完整保存实际或显式canonical activation及全部候选pair geometry。在已声明的独立flash、共同已知covariance和uniform prior模型下，pairwise error为`Phi(-sqrt(d²*h)/2)`；相同activation列按uniform tie，有限Monte Carlo另记录seed、trial数及相关candidate的trial-level误差。`Q=3/F=1`的解析高信噪极限为`2/3`，旧严格比较得到`1/3`的错误已修。候选曝光均衡与line-cycle均衡分列，`Q=9/F=45`每候选均15次，但line counts为8/8/8/7/7/7。

`ExactPropagatedCovariance/2`保存真实有限F对应的`F K_AR F.T`，SPD密度用无ridge的factor；显式正交支撑Q拥有自己的Euclidean测度，支撑外残差拒绝。数值分解未解决时保raw诊断，factor/密度明确不可用；ridge只通过独立显式诊断入口。白化偏差按`L_AR^-1 K_prop L_AR^-T`计算，修正旧square-root chart误差。信息界报告的完整JSON保存/load/model-only replay已闭合；exact类当前是内存诊断接口。相关验证与两轴独审分来源记录在[数学整合绑定](../assets/research_resume_20261012/diagnostic_mathematics_integration.json)，没有新增公共数据实验。

固定响应的完整物理factor由保存的M、上三角precision R和原Sigma导出：`F=kron(R^-1,chol(Sigma_ref))`。它保持common/increment系数及channel坐标，噪声变化时复用同一物理响应分布。

八人shared/private AR8、共同prefix32的16格native诊断已完成：每人600标签fit、120标签held评价、80授权标签未使用，8 anchor/16 noise fits均保存加载。16个held评分均触发数值可分辨性门，概率覆盖0；完整保存态已零拟合重现原失败。[执行绑定](../assets/research_resume_20261011/p1_native_panel_binding.json)保存原来源和完整分母。

数值稳定化已接受并接入科研树：full diffuse quotient保无量纲source-scale界，conditional score保nats界，仍用原两道阈值。实现包含prefix nuisance可逆换图、实际256-row短叶/平衡reduction、真实prefix/tail与列误差传播，以及完整posterior metric中的局部位移界。`Q.T b`显式按`abs(Q).T abs(b)`计算dot误差，logdet的factor2与raw normalizer/输出舍入都入账。

独立审阅核完整AR历史、flat测度/Jacobian、相关proper factor、inverse solve defect、prefix压缩和后验位移。109个独立案例通过；54个conditional案例中32获准、22具名拒绝，获准案例的110位oracle最大误差`1.395e-9 nats`且在各自估计界内。root逐读全部差异及最后finite保护，按原字节接入`5b827aa4…`，核依赖/证据SHA并通过六个新增反例、Ruff与diff检查，见[数值接受绑定](../assets/research_resume_20261011/p1_numerical_acceptance_binding.json)。

八人两臂的16格保存态续评已完成并独立复算，新增fit为0。A01的72候选保`22bf6782…`原评分来源，其余14格504候选由`5b827aa4…`评分；所有576候选满足原两道误差门。shared命中5/8、NLL7.336477、Brier0.679094；private命中6/8、NLL5.554012、Brier0.477679。private救回A06，未损害原正确决策，配对NLL下降1.782464；5/8人的NLL改善，其中A01贡献总改善的71.8%。两臂NLL都高于36候选均匀参考3.583519，严重错判的概率风险继续进入后续模型判断。

本面板每人仅一个C-held decision，八个预声明真值均为row5/col0；结果解释原冻结compiled-W Gaussian程序。真实平稳law修正采用新的producer另做同支持配对，不从本次held结果更改response、先验或候选。终态/独立逐概率复算/CSV与图见[续评接受绑定](../assets/research_resume_20261011/p1_numeric_panel_acceptance_binding.json)，原16格数值失败面板及两个执行失败run保持原身份。

这次A01消费的一次本机CPU采样为80.9%，按实际记录为超限；后续native1且检查总机余量，最多两个score workers。完整数值程序和原producer分开绑定；tree dot也改变新C-fit的舍入路径，原prior和统计模型保持。此处交付数值实现与真实保存态可用性，后续同物理支持的概率分析继续用原requested owner。

Windows控制文件读写竞态已独立复现，修复后原8份完整receipt逐字节复用、未封存A04/A05 private journal归档并重算，仅剩余8格新评分。实际续跑exit0、进程全部退役，CPU峰值44.6%；原terminal的64.1%对应整个历史追加日志。执行器的parent身份混搭缺口由最终独立检查器以原8份receipt SHA和固定script/config pair关闭，执行源码保持其真实历史字节。

真实 stationary-law 新producer 的 stage04 首次启动在严格旧capsule export 阶段发现冻结入口把 manifest 字符串传给只接受 `Path` 的 `BoundInput`，在 export/root compile/fit/score 前退出，owned tree 清理为0。stage05 已修复 export 与 worker 两个实际 caller，并以独立冻结 source/plan 通过 A01 原capsule strict export、独立审阅及7项 Path/资源生命周期验证；代码已整合到科研树。新实验保持原A01–A08两臂16格、seed20260918、C600/held120/unused80与完整原response/noise参数，只编译新的 `/3` stationary prefix root。

stage05 两次、stage06 一次本地实际执行均由 RAM 门记为 `resource_blocked`，exit1，owned tree 全部清理为0。三次都只有 A01 一个 active actor，已实际编译 shared8 partial root并完成1/5/4个候选的conditional resolution；完整score与cell receipt均为0，新增fit为0。stage05的RSS骤降与非同步采样是待核解释；stage06则是available RAM降至约1.46GiB，按reserve/growth公式触发负余量。单worker重规划没有闭合本地完整执行，后续交云窗口按真实资源与平台合同执行同一16格。旧 `/2` panel、三次资源终态和所有partial保持原身份，实际计数与交接见[执行绑定](../assets/research_resume_20261011/stationary_program_execution_binding.json)。

## 平稳前缀的真实模型关系

续评发现private旧入口的线程依赖：A01原保存root在native2重编逐bit相同并通过原关系门，native1重编却拒绝。更深入的两个独立100位oracle对全部36个cross-channel pair求真实平稳covariance，确认问题在旧companion Lyapunov/lag-power编译；全256维root的真实covariance pullback谱差为`0.0772181`，Frobenius差`0.167282`，而raw covariance相对差仅约`1.7583e-9`。原`1.6e-7`门核的是旧数值重编关系，弱方向需要真实law oracle。

尾部实际消费stored PARCOR lattice，新模型以每个binary64 PARCOR作为精确实数，经实数step-up确定AR；跨channel创新Sigma只入一次。完整pairwise mixed Yule–Walker与独立p² Sylvester在36个pair上一致；unequal AR1 closed form和diagonal scalar Yule–Walker另作参照。同真实law的高精度joint Cholesky仅舍入为binary64后，谱差`3.409e-11`、Frobenius差`1.109e-10`，支持保持原门实现稳定编译。

新compiler已由独立树完成并合入科研主树：`stationary_noise_artifact`以保存PARCOR的exact-real step-up、mixed Yule--Walker和joint Decimal96 Cholesky定义真实平稳law；saved root用独立向上误差证书验收，load不重编root。`/3`新state与旧`/2` compiled-W严格分开，旧P1续评仍只消费旧producer和旧law，不追签历史结果。A01新root的独立MP100 pullback谱/Frobenius误差为`3.409e-11`/`1.109e-10`，旧root为`0.0772181`/`0.167282`；stage04 underflow certificate、四种变换、save/load及实际caller均通过，0 fit/score。生产新state仍需绑定自己的producer和scope后再进入新实验。推导、两套oracle、实际旧state与源码绑定见[诊断绑定](../assets/research_resume_20261011/stationary_prefix_diagnosis_binding.json)及作者最终交接。

原Windows严格loader已导出全部8人的原响应与16份完整噪声状态。A01复用既有封存出口，A02–A08七个独立进程均exit0；最终33成员包为5,602,250 B，逐成员摘要、原130输入/158源码/158 capsule及完整请求经独立复审。原导出计划保持`056977…`，与新的云评分计划分别绑定；根窗口复核最终包33成员并写入[跨主机出口绑定](../assets/research_resume_20261011/stationary_crosshost_export_binding.json)。原`/2` strict load包含旧reference-root关系重编，新`/3` root编译、fit与held score均为0。

云科学接线已完成原导出计划对held样本/设计的独立授权及跨attempt互斥，修后独审33项与科研树集成5项分别通过，见[科学接受绑定](../assets/research_resume_20261011/stationary_cloud_scientific_acceptance_binding.json)。合法原A01可进入native5数值池，重签错误样本/设计及原输出争用均在计算前拒绝；原Linux旧loader失败与本地三次RAM失败保留自己的来源。

同支持的新stationary law云端16格已完成：8人×`shared8/private8`、每cell 36候选，共576 slots；每人一个`selection05` held decision，truth均为`row:5|column:0`。科学进程真实exit0，owned processes after exit为0；外层七项queue仍exit1，作为独立队列事实保留。两臂均prediction/probability coverage 8/8：shared命中5/8、NLL `7.3364765832`、Brier `0.6790937536`；private命中6/8、NLL `5.5540121013`、Brier `0.4776784949`。private相对shared救回A06、没有损害正确决策，paired NLL下降`1.7824644819`。A01贡献约71.82%的NLL改善但仍是接近1置信度的错误；A06贡献约75.71%的Brier改善。新旧stationary law在同一16格的winner完全不变；shared最大log-probability差`2.33e-10`，private最大差`8.86e-7`，实际概率最大差约`3.69e-11/2.45e-8`，law替换不是这次shared/private差异的来源。每cell fit/response-fit均为0，原response/noise与held支持保持。逐cell差异、terminal和原始result/queue SHA见[分析绑定](../assets/research_resume_20261012/p1_stationary_analysis.json)。

## P1 数值门诊断与小组协议评估

数值证据的第一、第二道门分别记录。前一道需要full-record quotient的正确算术估计，后一道需要prefix/tail各自的input与压缩误差；prefix basis、真实dot reduction和局部certificate按实际调用联动。只读推导及counterexample位于`tmp/P1_CHART_NEXT_HANDOFF_20261011/`，实码与独立审阅位于`tmp/P1_NUMERICS/`和`tmp/P1_NUMERICS_REVIEW/`。小组/单组结论集中在[校准与选择](calibration_and_selection.zh.md#小组部分标签与采样坐标)。

## B07：固定native新家族及真实坐标关系

唯一`dictionary_artifact/response_dictionary`compiler新增`/4` regular raised cosine和固定coarse6+fine detail3 cubic chart，free/hard-value端点；Gaussian/cosine `/2`及cubic `/3`各保原函数和数值语义。raised先在Decimal96中组合`(1+cos)/2`再一次binary64舍入，与旧原型先舍入cos再组合不同；atom边界精确为0，global free crop与hard-value连续性分开解释。

multiscale九列是fine9空间的可逆chart，并非新增span或正交wavelet。若`D_new=D_fine S`，保持同物理prior要求`Lambda_new=S.T Lambda_fine S`；同坐标ridge会改变prior及有效自由度，不能宣称同容量。g-reference沿真实C设计重编译并接受独立换图预测检验；hard端点为真实ambient chart内的约束。fs、`L=round(fs*T)`、实际端点`L/fs`、最后样本和半开支持继续分别保存。

实际consumer链为spec→compiler→event design→C-reference/prior→continuous posterior/容量→原OOF→ensemble save/load/score。五家族所有现有合法空间的精确off-grid δ/w评价及完整历史预测已完成下述软件切片。专用三臂dictionary comparison panel保持原Gaussian/cosine设计；下一出口为同支持公共评价、wavelet及导数/硬斜率，W03.01保持partial。

后续off-grid/history方向已通过源码、方法原件与独立反例审阅，见[方向绑定](../assets/research_resume_20261011/b07_offgrid_history_direction_binding.json)。12个小矩阵数学反例与7个时钟/授权反例分别验证：同native数组不足以定义连续函数；半开裁剪不能用Taylor或舍入onset替代；邻居与当前必须共用一次kernel后验并保跨事件协方差；正确条件混合为full与prefix各自先积分再相减，prefix同时更新邻居状态。Unfold原全文的完整事件设计/线性叠加给出方法依据，当前模型仍按自己的Gaussian/有限状态假设推导。

五家族精确δ/w评价、原decoder持有的截止内完整历史、冻结C后验的多通道条件预测及capsule保存重放已集成。当前decision完整flash覆盖与response尾部右删失分别记录；next-selection已发生前缀保完整候选词表和零贡献列。每个state的精确半开支持参与事件可见性判断，包络用于读取边界和ledger；恒零轴按完整实际joint质量边缘化，严格正质量不按阈值裁剪。原decoder整数event sample、response latency和width分别保存；实测fractional时钟由后续独立输入authority提供。

B05数值中枢与history adapter已接通：`native_history_marginal_program`消费原`TemporalNoise`、M/R/Σ及实际full/prefix观测，复用增广QR，按通道展开共同flat nuisance，生成full/prefix absolute potential、稳定conditional companion和原source-scale/nats估计。每状态及完整mixture的数值记录随prediction/capsule持久化。mixture先由prefix更新状态权重，再积分direct conditional；误差传播为`2*max(prefix_error)+max(conditional_error)`加实际算术估计，full误差保留且不重复相加。最终candidate scores先减max再logsumexp，保存概率保持归一关系；数值失败返回完整requested词表、capability和源码身份。

冻结C后验的合法state范围按实际C标签条件化的正joint质量判断：每个C-owned selection的正质量状态全部nominal，或它们的full/prefix design均精确为0。显式C-nominal/future-variable joint law获准；更宽C随机响应模型进入新posterior producer。主树54项相关software/synthetic检查通过，最终C-law六案例和多C失败账本另在实际主树复签；mixture独审11项数学案例与32组Decimal110扰动通过。软件验证包含合成拟合/加载/预测，独审曾误调用的4-case fixture按1次合成C fit、7次顶层load、6次prediction attempt单列，public fit/score为0。实际raw源码、行尾差异、验证范围及后续公共评价见[整合绑定](../assets/research_resume_20261012/history_central_integration.json)。真实A01 drift=0与drift=1e3/1e6的原接受/拒绝证据保留其来源。

### 新多通道校准潜状态后验

新 `multichannel_c_latent/1`已完成软件集成：同连续record由实际C binary labels定义合法selection q，全positive amplitude states与同global beta/record-local drift形成联合后验；实际startup/白化一次，原proper prior及physical flat measure各一次。C1一条合法标签可留下六q×两state，C119可唯一识别q，权限沿原compiler而来。fixed external参数角色明确，source/C学习参数及latency/width作为后续新程序接入。

原producer prior factor F及其error、后验R/RHS、flat图、C operands/noise identities保存，load不换图。futureprefix重新更新joint状态，conditional-tail保声明当前uniform prior，whole-observed法使用当前q-prefix证据；独立raw AR/actualselectiondesign对照误差≤1.78e-14。原normalizer误差改为实际small-log/scalar-operation计数，预算与全部state保持；最终int64索引、原upperflat图及C1权限单独验。

严格BNCI合成fixture119标签/八通道的正式三个完整出口OS0：八prediction、1044component延伸，旧Cmodel/MAT全SHA复用且零重拟合，nominal唯一新C fit一次；heldtruth攻击、model-only fit禁项、whole/censoredcapsule保存重放和正式CandidateEvidence均触发。scope是原decoder/factory/caller的纵向软件链。完整source、旧失败、新consumer和后续公共入口见[整合绑定](../assets/research_resume_20261012/research_integration_acceptance.json)。下一同支持公共比较与longhistory精确推断按新producer/资源声明推进。

## B09：可识别效应与source群体forecast

跨reference的共同物理效应输运已接通：直接在source真实joint likelihood中测目标reference-free泛函，保存BLUE均值、完整measurement covariance和native noise归一化，再进入群体forecast与目标prior。共同电极和参考关系来自共享authority；declared CAR的结构零与生成舍入一起核对。三种已声明source路径及source→群体→target→B05有独立逻辑和实际调用证据。最终入口修复冻结tuple/list的等价序列关系后，5项delta验证通过，见[整合绑定](../assets/research_resume_20261012/research_integration_acceptance.json)。

一般full-row source的结构识别已修为精确物理因子乘积：保存原空间map、时间design与record-local nuisance，并连接同次拟合的QR/噪声压缩关系。四导单行`[1,3,5,−9]`的真实beta-nullity为12，旧浮点归一化再Kronecker的舍入会误报0；新程序恢复12，微小真实未观测方向在直接BLUE和输运两入口均正确拒绝。general、native、实测参考和CAR共消费一次准入的后验，过渡白名单与重复输运入口核已删除。

原精确因子posterior `/5`、observation `/3`保存原支持坐标、归一map、尺度、体积和诊断；载入核原数学关系后恢复生产者数值，host SVD只做分辨率准入。1ulp诊断漂移保原记录，极端尺度D/Z先共同去单位再核QR关系。原数值拟合和BLUE方法复用，独审及新state14/实际consumer6/独立prototype分别绑定实际源码，见[整合绑定](../assets/research_resume_20261012/research_integration_acceptance.json)。真实长记录source builder可进一步利用因子row-space压缩，再接公共/ensemble消费者。

`ERPPhysicalEffect`绑定basis/window/grid/fs/operator/dictionary、真实reference与空间观测map。`ERPChannelEstimate`只从实际joint likelihood提取可识别BLUE，不把已有收缩posterior当独立measurement；measurement noise/dependence由其owner明确绑定。若压缩观测为`y=X beta+e`，物理泛函F必须存在witness`F=W X`，则效应mean为`W y`、covariance为`W W.T`。新关系检查逐entry使用原物理坐标，零观测列不从其它强列借容差；允许界来自浮点算法与运算尺度，不是全输入区间证明。

source group以participant为单位，对效应measurement covariance与人际variance分别记账，使用外部模板、强anchor与整程序上界。forecast covariance为`S_between+V_population_mean`，variance component为plug-in，不声称完整Bayesian积分。`mean_only`与`separable`是两种明确completion；不能把标准化d-prime向量按reference线性推送或手工放大噪声precision。

审查复现并修复tiny未观测泛函被全局sqrt(eps)吞掉、learned目标权限只在fit而可绕constructor/load、source scope/unit在load未完整核、效应coefficient-group关系、measurement schema/d-prime关系与zero-F的虚构population forecast。固定群体保全部participant；超出预声明bounds是整程序typed失败，不删异常人或截断贡献。

`transfer.spatial_programme`已将原baseline噪声、物理response refit、可识别BLUE、共同效应和source-group forecast接入真实目标OOF/partial-prefix、ensemble及保存重放。原continuous baseline持有实际noise-fit来源，新物理response只在同冻结噪声下拟合一次；source/controller/raw/label/study与prior-generation关系逐项关联。合成official-shaped MAT8→共同三导的纵向已通过，公共跨布局稳定性、d-prime机制及variance不确定性继续按新程序推进。

长记录结构识别已升级 `/6`：每record共同 `[Z D]` exact消自己的nuisance，再把原空间map配对时间quotient stream到shared beta；compact RREF/pivots保存，效应直接按exact rowspace membership判断。18000行实际posterior/BLUE/save-load和独立Fraction oracle已闭合，原numericQR/完整covariance保持。真实source/OOF/ensemble消费者已集成`ff131b8`，原producer保存修复已集成`138e2bf`；root在实际组合ensemble零fit/零QR载入旧包，原record、digest、baseline及文件字节保持，三项篡改反例拒绝。来源见[整合绑定](../assets/research_resume_20261012/research_integration_acceptance.json)。

### 成人七导的实际共同空间与完整接入

实际Huebner源为31导、nose参考；BrainSync七导使用left-ear参考，共享Fz/Cz/P3/Pz/P4五个电极，PO8/Oz未在源中测量。两边实际差分观测行空间的秩分别为31、7，联合秩34，交空间因而为4维。共同秩由真实观测行空间求出；这限定直接共享的物理量，完整31导仍可通过具名统计模型帮助预测七导。

一般迁移接口保留完整source人群law，并给未知参考差和缺电极响应一个明确的条件Gaussian：

\[
s=m_s+F_Bz_B+F_Mz_M,\qquad
w\mid s\sim N(a_0+G_ss,V_w),\qquad
\theta_t=B_ss+B_ww.
\]

其中人际差异因子F_B、总体均值估计误差因子F_M分别保存；w可取nose−ear、PO8−nose、Oz−nose，也可作精确可逆换坐标。真实七导的均值为`(Bs+Bw Gs)ms+Bw a0`，完整因子为`[(Bs+Bw Gs)FB,(Bs+Bw Gs)FM,Bw Fw]`，`Fw Fwᵀ=Vw`。Gs、a0、Vw的来源与量纲是新政策输入，保持全部源信息；34维表示潜在响应chart，实测源/目标仍分别为31/7导。跨成人联合时，F_M对应同一个共享均值变量，F_B逐人独立。

仅用共同边缘的完成是上述接口内的明确子模型。对目标共同映射L、源共同映射S，以及预声明reference `N(m0,K0)`，定义`J=K0 Lᵀ(L K0 Lᵀ)⁻¹`、`R0=K0−JL K0`。取`T Bw=I`并令`Gs=T(JS−Bs)`、`a0=T(m0−JLm0)`、`Vw=T R0 Tᵀ`，便精确恢复共同边缘完成；actual W和Gs先在精确算术中合并再舍入。一般模型中Gs可利用全部31导，不把四维共同空间设为信息上限。

proper表示概率归一化，允许Dirac等退化law。本次迁移另要求希望目标C继续学习的未知方向有完整条件支持，即Vw在其剩余坐标中正定；目标ambient中的R0自然为半正定。即使源人际差异为零，未知三向仍可有正变化；把残余方差乘以源between能量所得γ会在此处锁死支持，正倍数hyperprior也无法恢复。一般入口采用独立预声明的proper条件law，保原均值误差与人际差异；正域γ匹配作为另一具名政策保留，不用epsilon补方差。

给定任意共同law，保reference条件分布最小化其平均条件KL；共同KL有限时，这也最小化总KL。Dirac共同law下总KL可能全为无穷，此时条件KL仍能区分完成。固定reference允许同一次完整C更新响应；另学残余尺度需保存其proper层级和整个人共享残余的score。reference若随候选noise变化，其均值/gain/covariance的导数一起进入联合目标；把同一C的posterior再当prior乘C才会重复更新。

原population、chart、精确图、affine consumer和codec能力按各自职责复用。当前p=16时完整源参数496维，between/mean合计992因子列；source_map为identity的现有exact marginal先保完整496维源边缘及48维条件创新。进一步压缩应针对所有实际target查询与Gs的精确零空间，保各因子族的共享关系。旧软件fixture的3µV尺度保其数学造数身份，新reference政策另列来源。29组作者控制与8组独立Fraction/解析反例支持以上关系，科学设计和接口接受见[迁移数学绑定](../assets/research_resume_20261012/adult_population_transport_science_integration.json)。

成人完整多C/held输入和原parent恢复接口见[输入整合](../assets/research_resume_20261012/adult_programme_input_integration.json)。B已将完整execution的全部13个人数据likelihood、原source-only noise/reference/scale/seed、新成人排除、显式完整条件政策、actual W、保存/冷读及原parent重新准入连接到同一工厂。具名软件13人源为2导128Hz合成产物；它与实际七导250Hz的原EDF/BIDS输入完成9项控制，真实31导1000Hz的几何与完整源关系沿已接受数学和实际源码审阅核清。此处复用原控制及fixture的AST对应，主树另核六个模块实际导入，新增科研fit为0；新old-graph补充断言与三组独审数值脚本如实保未运行，接受依据及清理范围见[成人源迁移整合](../assets/research_resume_20261012/adult_population_source_integration.json)。

同人新会话已能只读复用原固定source，并按新A重新核全部源排除关系。7项软件控制实际通过，包括新会话、异人拒绝、别名/EEG/record污染、原发布字节与异常对应、原13人数据因子及13套forecast冷读；新增source/C拟合、fold发布均为0。根整合复用原控制，核18份独审引用、三份主树字节和六个实际模块导入；原producer及已冻结的科研runtime保原身份。接受范围与来源见[新会话源排除整合](../assets/research_resume_20261012/adult_personal_query_source_integration.json)。

后续全链连接一次完整C联合校准、保存后原C重新准入、完整history及typed个人入口；12人LOSO折保自身角色。完整切片定义见[成人接入交接](D:/BCI/artifacts/research-deliveries/source-response-states/public-joint-history-inference/ADULT_BRIDGE_HANDOFF.zh.md)，实施和运行状态由当前checkpoint维护。

### 源数据因子的充分状态与产物职责

在原噪声、字典、观测算子和flat测度固定时，同次压缩的全列A、响应r与完整常数c决定整条数据似然：`log L(θ)=c−||r−Aθ||²/2`。c保正交残差、nuisance体积、时间Jacobian与原物理行数；个体proper prior独立于这一数据因子。源moments实际只从原baseline的PARCOR/full Q取得，population、fold冷载及target随后消费compact因子与QR见证。因此下一版可由原reader/压缩owner直接发行这些充分状态，并为完整模型恢复任务另保具名载荷。来源链、精确结构支持、反例和完整实现切片见[源产物必要性研究](D:/BCI/artifacts/research-deliveries/source-response-states/public-population-source/SOURCE_ARTIFACT_PLAN.zh.md)；实现接受由对应代码审阅和真实运行记录给出。

独立预声明参考已接到实际压缩核心：八组控制覆盖完整高斯密度、F→C全列置换、平坦nuisance测度、空间体积、正交残差、极弱非零方向、全部实际行与异构张量空间和；完整密度的最大误差为7.11e-15。2^-500弱列继续保留，实际binary64白化后的结构由实际D/Z确定，见[独立数学对应](D:/BCI/artifacts/research-deliveries/source-response-states/public-population-source/independent-compact-factor/CANDIDATE_MATH_CONTROLS.json)。

在固定噪声与观测的对照中，将旧源个体后验的proper先验方差从9e-12换为1，后验随之变化，导出的A/r仍逐位相同、c差为0；新数据因子给出相同结果。该对照直接核实“数据证据与辅助个体先验分开”的实现关系，见[先验与发行权限控制](D:/BCI/artifacts/research-deliveries/source-response-states/public-population-source/compact-factor-delivery/PRIOR_AUTHORITY_CONTROLS.json)。

## B11：同记录漂移与噪声尺度

同record Gamma/flat科研程序已接受并集成`dd63d32`：global proper β prior一次，每真实record一个Gamma λ和原物理flat漂移测度。有效信息量按原raw行数减漂移rank计数，同一full-record白化用于C、prefix与tail。固定noise shape与动态λ分开，proper β不随test λ缩放。source参数从原保存Continuous模型恢复，原prior因子经等价QR转换，原noise-fit保持；C-plugin与source-only权限各用真实issuer。

尾部保prefix漂移后验，以直接增广QR计算，避免巨大中间量取消；原17个状态的energy/logdensity/logdet保持，实际误差估计由超预算降至约`3.32594e-5 nats`，低于原`.00111961`预算。最终7个定向案例和root旧包16draw的零fit/白化/QR重放通过，原package SHA和producer数值保持。此处为合成MAT真实软件链，原Gaussian∞包与finite Gamma law分别声明。下一切片接typed ensemble、候选prefix、模型包与public opt-in，连续Gamma预测精度沿真实law验收。

## 后验QR与有效自由度的反例

恢复增量使stationary fit均值、posterior covariance、保存R/RHS/Q、noise residual与score消费同一个compressed QR后验，避免两次QR的舍入差把合法取消均值拒绝。legacy clipped AR1保持原估计程序，数值值不承诺逐bit等同父source；旧包继续由其冻结producer解释。

独立审阅另复现重复basis的大尺度合法反例：3×110样本、2channel、ridge1、实际只有两信号方向。`trace((D.T D)H)`在AR0给26.2414407043，独立两维模型的df为2；错误residual df使Sigma放大8.1413%。legacy AR1在该反例给6.4695454025，Sigma放大1.4076%。这是数值算术错误，不是新生理机制。

修后保持原定义，使用同posterior root和实际information root计算

`df=trace(D.T D H)=||R_D R_posterior^-1||_F^2`。

不clip df，不改LedoitWolf与残差自由度科学定义，fixed32的nTail profile分母保持。独立两noise路径×六amplitude核查中，1e8幅度的df与独立模型差小于4e-14，Sigma倍率回到1；新正式反例先RED2后GREEN。保存mean/root关系门仍拒绝真实变更，不放宽阈值。

`H`仍是保存的factor-Gram诊断；held预测直接消费保留的R，不重新从H分解。fixed32只改变同一producer后验的数值生成入口，profile Sigma仍由原joint tail目标估计。

## P1后续方向：冻结响应与噪声程序的配对评分

下列方向记录对应实施前的决策；软件入口和16格native面板随后均已完成，结果见本页开头及下节。

独立逐读B05/B06/B11后，总监接受同一完整物理response后验下的shared/private AR8、共同C residual与32点prefix，接现有CandidateEvidence、DecisionPlan与paired metric owner。实现由独立`research-mechanism-paired`工作树交付；该树现继续承担数值稳定化。

方向依据包括8个数学/实际owner反例和一次current synthetic fit/save/load上的2个关系核验，均exit0。完整可逆白化不自动损失判别信息；非可逆投影可降低ACF却删除ERP；相同秩不排除错误span；Schur covariance正确而漏候选相关条件均值仍可翻转赢家；无候选信息的prefix仍更新共享不确定response。另实测近似平局在B05原门与共同CandidateEvidence门会得到不同赢家；删困难决策使可用NLL下降却降低coverage，被共同paired gate拒绝。新panel预声明共同赢家程序，原B05 winner仅作其原程序诊断，不追签旧输出。

现代Continuous保存的上三角row precision为R，完整系数M前半为common、后半为target increment；row-major系数×channel物理factor为`kron(R^-1, chol(原Sigma_ref))`，保持同一保存M/R和原spatial covariance。真实小型scorer图像关系误差约`1.39e-17`；误用R转置的图像误差约`0.04657`。不能从派生H重造现代producer，不能随新noise换response不确定性；这是模块化plugin预测程序，不称各臂joint posterior或joint MLE。

新physical入口须消费实际PredictionDecisionView/MembershipDecision的onsets、membership、source/support、事件counts和evidence level，design按原basis核验；旧generic数组manifest不能猜counts补身份。建议后续16格C内诊断为8实际participants×2noise，各用C800授权中最早6完整组、前5组600标签拟合、第6组120标签later-held，其余标签另列。它不称C800全fit、独立公共终测或成人确认。现代转换、共同指标与该实际输入先在代码/反例/保存重放中验收，再执行新panel。原ignored B11 payload未恢复，不重跑旧panel来补签；证据指针与hash见[方向绑定](../assets/research_resume_20261011/p1_direction_binding.json)。

## P1软件入口修后验收

`FrozenERPResponse.from_continuous_model`和paired audit `/2`已集成：保存M/R/原Σ转换为完整物理factor，公共入口从原MAT owner重建view、onsets/membership/design/support及标签权限，输出共同CandidateEvidence/DecisionPlan/paired指标。C-side残差明确不扣nuisance，可含drift/ERP失配；held原flat affine nuisance和noise数值程序保持原定义。本接口是固定base-C后验后的plugin比较，不把新noise当重估joint posterior，不从噪声诊断决定产品默认。

独立审阅否证了作者65项通过的初版：保两份实际source records却同时删A02 participant与plan能缩分母；另将C labels改成另一合法candidate并真正重拟modern R模型，也能越过public admission。修后公共requested集合来自全部actual source CalibrationSelection，resolved与selection-failure由共享DecisionPlan完整分割；每个C target/label逐项核原owner。两攻击均在noise.fit之前拒绝，合法失败participant仍保宏分母。原RED、原65-body与修后身份分开，不累加成一组证据。

修后作者16项、原独立主probe6项及专门admission3项通过；原wrong-label脚本因更早builder拒绝仍exit1，保原字节，另用旧wrong-label输入直接audit确认0fit拒绝。root逐读source、C、modern factor、held、指标及失败消费者后接受有限软件切片，Ruff/diff检查通过。两人四格诚实actual-owner synthetic输入完整fit/save/load/replay通过，不代表新native成绩。八原MAT的16格面板随后完成，但16个held cell均因完整proper-prior/noise chart的数值可分辨性门记为typed `not_available`，概率覆盖0、paired NLL/Brier为空；这不是分类错误率读数。执行身份、8 anchor/16 noise fit、保存态零拟合失败重放和资源边界见[native面板绑定](../assets/research_resume_20261011/p1_native_panel_binding.json)。原面板完整执行说明保留在本地 `tmp/P1_NATIVE_20261011/attempt02_absolute_paths/`。
