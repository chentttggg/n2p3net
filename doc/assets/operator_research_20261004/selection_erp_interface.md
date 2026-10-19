# 成熟ERP selection锚点研究Interface

实现：`selection_erp_model.py`；验证：`verify_selection_erp_model.py`及`selection_erp_verification.json`。未改生产默认或GLS实现。

```python
from selection_erp_model import SelectionERP

model = SelectionERP.fit(
    calibration_records,  # distinct complete CalibrationRecord selections
    family="waveform_lda",  # or toeplitz_lda / xdawn_erp
    baseline_mode="mean_only",
    sfreq=256.0,
    channels=channels,
    candidate_ids=candidate_ids,
    operator=finite_filter_operator,
    expected_label_count=600,
)
result = model.score(
    raw_samples, onsets, public_membership,
    record_id=held_record_id, sfreq=256.0,
    channels=channels, candidate_ids=candidate_ids,
)
model.save(new_npz_path)  # exclusive create, numeric arrays + JSON
reloaded = SelectionERP.load(new_npz_path)
metadata = model.record()
```

四个预声明臂：

| family | baseline_mode | 真实训练owner | 额外参数 |
|---|---|---|---|
| waveform_lda | mean_only | `transfer.binary.target_shrinkage_features` + `fit_shrinkage_feature_model` | 原20Hz FFT，28点/导，StandardScaler→lsqr/auto LDA |
| waveform_lda | none | 同上 | 同上，仅baseline桥接 |
| toeplitz_lda | none | `baselines.structured_erp.StructuredERP` | bin_width=4，44个等宽bin；178点末2点固定不用 |
| xdawn_erp | mean_only | 同上，内部原xDAWN ERP协方差owner | nfilter=4；要求实际观测秩≥4，收缩不能制造此秩 |

每selection只消费调用者提供的独立raw支持：first onset相对位置必须51，末尾严格为last onset+307 exclusive。单epoch实际闭源区间为[-51,+306]，358点@256Hz。原`PreprocessingSpec`要求输出物理轴点数为floor(357×128/256)=178，因此采用**shared FFT先179点，再按真实合同保留前178点**。没有将tmax伪改307来强求179点，也不请求sample+307。

原始第一/最后采样为−199218.75µs/+1195312.5µs；保留输出第一/最后为−199218.75µs/+1183593.75µs。mean-only baseline严格用物理t<0的26个输出点。滤波operator只对每selection执行一次，epoch数值转换调用共享`transform_epoch_samples`。输出记录包含source支持、FFT179/保留178、baseline、operator、每条row起止及指纹。

候选分数是`membership.T @ (binary_logit - fitted_prior_log_odds)`，先去掉训练类prior再以float64累加。它是重叠epoch条件独立近似下的LLR聚合；候选prior固定均匀，不在held上调温，NLL不能解释为真实校准后验。返回现有`ContinuousDecisionResult`，记录真实独立raw样本总数。

原waveform owner的特征/预测是float32；保存状态不会改变它的StandardScaler舍入顺序与LDA计算dtype。独立直接owner重新拟合的逐trial logits与本wrapper四臂全部逐值一致，保存/重载候选evidence逐值一致。验证另覆盖不等曝光prior修正、错误/缺失源端点、非法rank、非bool membership、channel错序、训练bytes改名重用、重签空训练ledger及不可变状态。

`load`要求保存的实际源实现SHA与当前consumer一致，防止另一版owner读取后默默改变分数。所有模型只含数值NPZ和JSON，没有pickle。源码SHA来自module载入时的字节；物理执行快照/总矩阵证据scope继续由runner绑定。此验证仅为software_correctness，不是新的公共性能成绩。
