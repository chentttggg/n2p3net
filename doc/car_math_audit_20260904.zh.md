# 跨域 CAR 数学复核与证据边界

日期：2026-09-04  
范围：`src/data/domain.py` 的 `n2p3_common_channel_car/1`、
`experiments/prepare_multidomain_source.py`、`tmp/l1_cloud/montage_ablation.py`，以及
`doc/cross_dataset_bi_gtn_ruling_20260904.zh.md` 中的 BI2014a → GTN 裁决。  
状态：**CAR 的代数定义成立；“CAR 已完成跨域特征对齐”不成立；BI2014a → GTN 的
2 导路径仍不进入活动迁移实验。**

本文是本轮数学审计和证据留痕，不授权新的实验设计。活动门禁仍以
[`research_program.zh.md`](research_program.zh.md) 为准。

## 1. 审计结论

1. 对同一组、同一顺序、每个 trial 都完整存在的物理通道，代码执行的
   “先选通道，再减该组瞬时均值”正是标准 common-average projection。它能在加性
   标量参考模型下消除不同数据集的参考偏移。
2. CAR 不是可逆的域对齐器。它把每个时刻的通道向量投影到零和子空间，丢失一个
   空间自由度；参考偏移被消除与任务信息被保留是两个不同命题。
3. `k=2` 时投影秩只有 1，输出只是一个差分通道及其负值。它不能恢复选定两导的
   共模分量。这个事实足以把 BI2014a（`Cz,Pz` 交集）→ GTN 的 2 导 CAR 路径列为
   **科学阻塞项**，但不能把“2 导 CAR 必然使所有 P300 任务失效”写成数学定理。
4. 本地诊断有两处数值记录错误，已在 `tmp/l1_cloud/montage_ablation.py` 修正：
   残余比例漏除以 2，且 AUC 差值的符号解释反了。修正后为
   `0.1321`（不是 `0.2642`）和 native-minus-CAR 的中位损失 `+0.0490`。

## 2. 严格公式

### 2.1 参考模型和 CAR

对一个域 `d`、时刻 `t` 的 `k` 个有序通道，写成列向量

\[
x_d(t)=s_d(t)-r_d(t)\mathbf 1+\epsilon_d(t),
\qquad \mathbf 1=(1,\ldots,1)^\top .
\]

这里 `s_d` 是相对于任意共同零点的头皮电势，`r_d(t)` 是该域的标量活动参考，
`epsilon` 汇总通道噪声和模型未覆盖的误差。共同平均参考矩阵为

\[
H_k=I_k-\frac{1}{k}\mathbf 1\mathbf 1^\top,
\qquad z_d(t)=H_kx_d(t).
\]

因此

\[
H_k\mathbf 1=0,\quad H_k^\top=H_k,\quad H_k^2=H_k,
\quad \operatorname{rank}(H_k)=k-1,
\quad \ker(H_k)=\operatorname{span}\{\mathbf 1\}.
\]

对任意时变标量 `a(t)` 都有参考不变性

\[
H_k[x_d(t)+a(t)\mathbf 1]=H_kx_d(t).
\]

这证明的是“共同标量参考偏移不会进入输出”，不是“输出等于真实无限远参考的
电势”。向量的正交分解为

\[
x_d(t)=\underbrace{\bar x_d(t)\mathbf 1}_{\text{被删除的共模}}
       +\underbrace{H_kx_d(t)}_{\text{保留的相对空间分量}},
\qquad
\bar x_d(t)=\frac{1}{k}\mathbf 1^\top x_d(t).
\]

`H_k` 没有逆矩阵；给定 `H_kx`，可以加入任意 `b(t)\mathbf 1` 而得到同一个输出。
所以任何只接收 CAR 后张量的下游网络都无法从该张量重建被删除的共模。

### 2.2 两导特例

\[
H_2=\frac12
\begin{bmatrix}1&-1\\-1&1\end{bmatrix},
\qquad
H_2\begin{bmatrix}x_{Cz}\\x_{Pz}\end{bmatrix}
=\begin{bmatrix}(x_{Cz}-x_{Pz})/2\\(x_{Pz}-x_{Cz})/2\end{bmatrix}.
\]

两行线性相关，只有一个独立空间维度。被删除的是
`(Cz+Pz)/2` 这一选定集合的共模，而不是抽象意义上“所有 P300”。若 target ERP
主要表现为两导同向差异，信号会被削弱；若主要表现为 Cz-Pz 差异，CAR 可能保留
甚至改善可分性。这必须由任务数据验证。

### 2.3 跨域成立所需的联合假设

若两个域满足：

\[
\text{same ordered physical channels},\quad
\text{complete observations},\quad
\text{same units/preprocessing},\quad
x_a=s_a-r_a\mathbf1,\;x_b=s_b-r_b\mathbf1,
\]

则

\[
H_kx_a=H_ks_a,\qquad H_kx_b=H_ks_b.
\]

这只处理参考项。它不处理通道增益、噪声协方差、空间采样、刺激范式、潜伏期、
采样率、滤波状态、候选/决策协议或被试分布。因而 `domain.py` 的统一 CAR 应被
称为 **reference/montage normalization**，不能单独称为 feature-domain alignment。

此外，“先选公共通道再 CAR”与“每域全通道 CAR 后再取交集”一般不相等：

\[
S H_{K_{full}}x\ne H_{K_{common}}Sx
\]

除非全通道集合与公共集合恰好相同或满足特殊约束。当前实现选择前者，并在 provenance
中记录了这一操作，故不能把两种口径的结果混作一个实验。

## 3. 本地反例和复算

### 3.1 符号/数值不变量

本轮用 `.venv` 的 SymPy 做了精确核验：`H_k` 对 `k=2,3,5` 均对称、幂等，秩为
`k-1`，唯一零空间为全 1 向量；`H_2` 的两个输出通道互为相反数。对应的投影
不变量和真实 builder 行为已迁移到正式代码测试，覆盖了：

- 不同且时变的加性参考偏移经 CAR 后完全抵消；
- 输出通道和为零、两导输出等于 `+(Cz-Pz)/2` 与 `-(Cz-Pz)/2`；
- 只改变选定通道共模的两个输入在 CAR 后不可区分。

这些是变换不变量测试，不是 P300 性能证据。

### 3.2 BI2014a 12 人诊断复算

命令：

```powershell
.\.venv\Scripts\python.exe tmp\l1_cloud\montage_ablation.py
```

输入为 `tmp/l0_cache_bi2014a_v3_s12.npz`（10884 epochs，16 channels，128 Hz，
`[-200,1200) ms`），在 280--600 ms 窗口作每被试 target/non-target 秩 AUC。它是
朴素相对消融，不是训练模型结果：

| 蒙太奇 | 中位 AUC | 均值 | 范围 |
|---|---:|---:|---:|
| native reference，mean(Cz,Pz) | 0.5440 | 0.5515 | 0.5151--0.6027 |
| 16-channel CAR，mean(Cz,Pz) | 0.5082 | 0.5208 | 0.4713--0.6147 |
| 2-channel CAR，`(Cz-Pz)/2` | 0.5047 | 0.5020 | 0.4389--0.5575 |
| native reference，Cz only | 0.5497 | 0.5473 | 0.4930--0.6063 |
| native reference，Pz only | 0.5338 | 0.5429 | 0.4871--0.6182 |

组平均 class-difference wave 在该窗口的相关为 `r=0.9942`。更直接的分解是：

\[
\frac{\|\Delta_{diff}\|_2^2}{\|\Delta_{common}\|_2^2}=0.03186,
\qquad
\frac{\max_t|\Delta_{diff}(t)|}{\max_t|\Delta_{Cz}(t)|}=0.1321.
\]

这里 `Delta_common=(Delta_Cz+Delta_Pz)/2`，`Delta_diff=(Delta_Cz-Delta_Pz)/2`。
原记录的 `0.2642` 实际是未除以 2 的 `|Cz-Pz|/|Cz|`，不能叫作 2 导 CAR 输出
比例。native-minus-CAR 的每被试 AUC 差中位数为 `+0.0490`、均值 `+0.0495`；
“AUC drop = -0.049”是符号解释错误。该结果支持“这 12 人 cache 中，2 导 CAR
会明显改变朴素检测的空间证据”，不支持“所有 P300、所有被试、所有模型都必然
失效”。

## 4. 外部方法依据及其边界

- Dong et al. 对 ERP/ICA 的比较指出，不同参考对部分时间 ERP 和空间拓扑有非忽略
  影响，并报告 REST、平均参考、耳/乳突参考不是可互换的同一表示；这支持把参考
  作为显式实验因素，而不是把 CAR 当成无损标准化（[Dong et al., 2019](https://doi.org/10.3389/fnins.2019.01068)）。
- Hu et al. 将 AR 与 REST 放在统一 Bayesian 参考估计框架中，指出 AR 对应非信息
  生物物理先验、REST 使用头模型先验；这说明“平均参考 = 真零参考/无限远参考”
  不是一般定理（[Hu et al., 2018](https://doi.org/10.3389/fnins.2018.00297)）。
- Samantaray et al. 的 ERP 研究观察到 linked mastoid、CAR、REST 会改变 ERP 的
  幅度、极性和拓扑；该研究是 talk/listen 的 N1，不是本项目 P300，因此这里只能
  支持“参考会影响 ERP 测量”的一般机制，不能替代本项目验证
  （[Samantaray et al., 2025](https://doi.org/10.1007/s10548-025-01103-4)）。

外部文献共同支持的是“参考方案必须预声明并做敏感性分析”，不是“CAR 在本项目
跨域一定有效”。

## 5. 对活动合同的修正建议

### 已落实

1. 保留 `src/data/domain.py` 的通用 CAR 变换及其 provenance/lineage 校验；其代数
   语义正确，不因 BI→GTN 的 2 导不适用而删除通用接口。
2. 修正诊断脚本的 2 导输出命名、`1/2` 比例和 native-minus-CAR 符号。
3. 将 CAR 反例迁移到 `tests/test_domain.py`，防止后续重写成“差值未缩放”或把
   共模不可恢复性误写成性能定理。

### 必须保持的研究门禁

- `k>=2` 是代数最低要求，不是科学充分条件；任何 `k=2` 路径都必须单独报告秩、
  被删除共模、保留差分和任务级 held-out 结果。
- 不得只凭 `r=0.9942` 或一次朴素 AUC 消融宣称 P300 已被数学判死；正确表述是
  “在该 cache 中观察到目标/非目标差异主要位于被 2 导 CAR 删除的共模方向，故该
  路径不进入活动跨域 promotion”。
- 不得把统一 CAR 记录为 feature alignment 的充分证据。跨域 promotion 仍需
  同一 preprocessing、单位、通道顺序、reference provenance、causal state、
  source/target split，以及 participant-cluster 的 operational `hit@R`。
- 方案 2（各域全通道 CAR 后再取公共导联）与当前 common-subset CAR 是不同估计量，
  若未来比较必须作为独立 arm，不能复用当前 `n2p3_common_channel_car/1` 的同一
  结果解释。

## 6. 最终裁决

当前应删除的不是 CAR 代码，而是“CAR 自动完成跨域对齐”和“2 导 CAR 必然使 P300
失效”这两层过强语义。数学上可确认：CAR 是一个去除选定通道共模的正交投影；
跨域上可确认：在加性标量参考假设下它消除参考偏移；科研上尚不能确认：它保留了
足够的 P300 判别信息或能够产生迁移增益。BI2014a → GTN 的 2 导路径因此保持
fail-closed，主线转向同一 3 导/9-choice 语义内的跨被试域定义或等待具有足够公共
空间采样的真实 source/target 数据。
