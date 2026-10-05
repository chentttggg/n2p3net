# 输入合同与路线审查：复核与修正（2026-09-22）

本次按用户要求复核原审计并修改已确认问题。使用项目`.venv`，检查当前dirty工作树，
HEAD为`f1a303c42c68173e8b1452ceb26bd84923b8a6aa`；HEAD不代表全部实际执行源码。
应用`astra-research-workflow`、`eeg-p300-research`、`experimental-design`。
原审计与原09-20路线文档已逐成员压缩校验，见[归档清单](../../frozen/input_window_audit_20260922/manifest.json)。
本文件替换原审计的错误数字与推论，不覆盖历史训练结果。

## 逐项裁决

| 主张或建议 | 复核结论与处置 |
|---|---|
| 1200ms内存在后续刺激 | 成立。原始事件可证明重叠，不能证明长窗有害或已触及共同性能上限 |
| subject_01同rep中位数285.2ms、跨rep630.9ms | 未复现。完整ZIP与前400k样本分别重算，数字见下表；已更正 |
| 平均4个后续flash、71%输入为干扰 | 不成立。用`floor(1200/median(interval))`代替逐事件计数不是实测平均；窗口长度比例也不是信号/噪声比例 |
| 250–600ms是当前模型全部有效证据窗 | 不成立。该参数只进入可选LMBC；默认`full_unfold`用全部卷积特征，LMBC还含潜伏偏移及卷积感受野 |
| 旧默认窗证据混淆 | 成立。这段`2Hz/800ms`对`0.1Hz/1200ms`比较不能分离窗长作用；更正源码注释与输入指导，不宣称已验证短窗更优 |
| 先窗口、再EA/RPA/GALIA | 采纳为开发执行顺序，已更新roadmap与路线文档；理由是先解决输入选择的不确定性 |
| 6×6归并3×3即产品同构代理 | 不成立。新membership与原binary标签冲突；保留原合同，不重标cache。预测后粗定位可另列探索性指标 |
| 唯一主门槛mean ΔAUC≥0.01 | 已统一：四库等权、库内participant-macro；逐人正迁移率降为风险诊断 |
| 五个模块引用失效规范 | 成立。docstring已改为当前职责及六份指导中的相应入口 |
| within-session随机trial split风险 | 成立。相邻epoch共享原始样本；时间分块还须purge支持并隔离双向预处理。未执行新上界训练 |
| runner/历史快照堆积 | experiments顶层当前62个py文件；gitignore已排除runs/cache，Ruff通过。未删除或移动历史材料 |

## 原始数据复算

使用[可复算工具](../../experiments/audit_bi2014a_window.py)直接读取BI2014a ZIP内CSV，
零基第17列为flash code（人类计数第18列），零基第18列为raw label。
采样率512Hz来自现行原始数据合同，亦与[MOABB数据说明](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BI2014a.html)一致。
按规范码本逐事件交叉检查标签，每12次flash验证六行六列各一次、各含一个target。
统计同rep时排除每12次的边界；跨rep统计包含休息/selection间隔，不能当作固定SOA。

| subject_01统计范围 | 同rep间隔数 | median | mean | p10 | p90 | 跨rep median |
|---|---:|---:|---:|---:|---:|---:|
| 完整417925源样本、1188 flash、99 rep | 1089 | 349.609ms | 394.707ms | 199.219ms | 617.188ms | 1372.070ms |
| 前400000源样本、1160 flash | 1063 | 349.609ms | 395.540ms | 199.219ms | 617.188ms | 1370.117ms |

前400k共有1159个相邻间隔，其中96个跨rep；原审计写的1159并非同rep间隔数。
此处没有认定原审计错误的具体程序原因，只记录本地可复现差异。
ZIP SHA-256：`07c256d8d540b9cffafdcda08e1fd0f2307fdaa41ce04f9bdb6dd65e91fd9891`。
独立调用现行`recover_bi2014a_candidates`读取解压CSV得到相同的1188事件和两项间隔中位数。

逐事件直接计数`#{j: onset_i < onset_j < onset_i+tmax}`，不使用间隔中位数近似：

| subject_01响应截止 | 后续onset数mean | median | p10/p90 |
|---|---:|---:|---|
| 600ms | 1.037879 | 1 | 0 / 2 |
| 800ms | 1.431818 | 1 | 0 / 2 |
| 1200ms | 2.175926 | 2 | 1 / 3 |

这些是记录内的后续onset计数，包含尾部，未外推记录外未知刺激，也没有测量ERP能量。
61名作者准入被试共57446次flash，所有已记录flash标签交叉检查通过；2次不完整尾部flash
不进入完整rep间隔统计，但保留在onset重叠计数中。
逐人同rep中位数再取总体median为333.984ms，p10/p90为330.078/349.609ms；
1200ms后续onset计数的participant-macro mean为2.221972。
这些描述统计不代表61个参与者独立因果实验的效应量。

```powershell
.\.venv\Scripts\python.exe experiments/audit_bi2014a_window.py --output experiments/results/input_window_audit_20260922/raw_timing.json
```

完整逐人记录、ZIP摘要及执行工具/码本源码摘要在
[本地JSON](../../experiments/results/input_window_audit_20260922/raw_timing.json)。
JSON SHA-256：`6642176c33e64391ca4546b94edfa0d6fb0efeb191e5328193857c5b4d4a280a`。
原始数据和结果保持本地，不加入Git分发。

## 两条不能直接落实的修改

**不能据重叠直接将默认值换成600ms。** 实际输入可写为目标锁时响应与邻近刺激响应的叠加，
加上背景活动；邻近刺激反应既可能干扰，也可能含可利用的信息。原统计不能分解这些成分。
`350/1200`只是两个名义时长的比例，不能推导71%噪声，也不能解释容量消融或zero-shot的因果原因。
源码注释引用的两因素联合比较缺少窗长归因，足以撤下“1200ms已优于短窗”的表述，
却不足以反向认定600ms最优。0.1–30Hz与原窗口数值保留原执行身份，等待配对实验。

**不能把粗网格当作新的物理membership。** 取原target `(r,c)=(0,0)`：
row0和col0为target，row1和col1为non-target；合并后的粗目标 `(0,0)`却会把这四次全部判为target。
每rep原2个正类被错误改成4个，刺激正类比例从1/6变为1/3，仍不等于产品直接闪烁的1/9。
现行`validate_candidate_membership_metadata`已拒绝这种映射。9个输出类别与chance=1/9
不足以建立范式同构，61/8也不能直接当成统计功效倍数。
合法替代仅为36选一预测之后再计算`(r//2,c//2)`粗定位，必须保持原训练标签并另命名读数；本次未新增该探索任务。

## 窗口实验合同与可用时间

当前roadmap已将600/800/1200ms窗口族置于对齐路线前。主比较600对1200，800为次要敏感性。
已有cache的`X[..., :N]`可作廉价特征截取诊断，N=102/128/179；不修改父cache的preprocessing、
raw支持、证据可用时间、QC、身份或同一train/test行。LDA/target-only不要求新增source训练。
神经`full_unfold`的head维度随N改变，因此“其余一切固定”只能指配方与数据范围，
不能宣称参数量不变或直接复用长窗head。需要新source模型时须明确计入后续训练矩阵。

实际算子是连续滤波→源epoch→FFT重采样→截前T点→基线。
[Scipy FFT重采样说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html)
明确其频域截断/补零和周期延拓假设；本次还通过生产`transform_epoch_samples`做了数值反例：
仅在+900ms放入脉冲，长窗FFT之后的前600ms仍非零，真实短源epoch重建则为零。
即使滤波为forward，长窗FFT仍会把尾部信息传到保留前缀；双向滤波另有完整raw范围依赖。
相同基线时段不保证不同FFT长度下的基线数值相同。

within-session必须完整decision/时间分块、purge最长实际支持；双向数据先分raw再处理，
不能对完整记录双向cache仅做blocked trial split后声称fit/test隔离。
cache裁剪收益只回答当前特征选择问题，负结果也不能彻底关闭真实短窗路线。
若raw配对重建支持更短窗口，再同步builder、checkpoint、runner、评分、模型保存重载及成人确认。
原82.09%文献数值本次未重新核验其版本/划分/预处理，不作为同口径上界或项目验收阈值。

## 实施与验证

- `data.contract`：纠正默认窗的归因与GTN论文profile的0.5Hz/开发0.1Hz区别，默认数值未改。
- `baselines/{classic,deep,features,riemann}.py`与`data/gtn.py`：移除constitution/techstack/Phase旧契约引用。
- 输入指导、研究规范、roadmap和09-20路线文档：窗口诊断边界、粗网格限制、唯一效应门槛同步。
- 新增原始ZIP审计工具及9项反例，覆盖逐onset半开计数、错误事件轴、ZIP语义、重叠样本、FFT尾部依赖、非法3×3标签。
- 定向与集成验证188项通过，覆盖上述反例、BI candidate、预处理执行/凭证/raw隔离、epoch合同及N2P3Net。
- `python -m ruff check src tests experiments`通过；`git diff --check`退出0。未运行全套，不把局部验证写成全库验收。

本轮完成原始数据审计、合同表述与执行顺序修正、反例验证；未拟合新LDA/神经模型，
600/800/1200性能矩阵和within-session上界仍待执行，尚无窗口AUC增益或成人产品性能结论。
