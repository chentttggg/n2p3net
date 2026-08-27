# Performance Blueprint

## Common Artifact

Each adapter produces `EpochDataset`:

```
eeg: float32 (N, C, T)       label: int64 (N,) in {0, 1}
subject/run/trial: string (N,) stimulus: string (N,)
decision_id: string (N,)      channel_mask: bool (C,)
sample_rate_hz: float         channel_names: tuple[str, ...]
qc_features: relative_ptp, channel_std, epoch_scale, observed_mask
```

The artifact stores provenance, source units, epoch bounds, preprocessing
configuration, and rejected-trial counts. Canonical resampling and filtering
are configuration choices evaluated inside the protocol, not hidden defaults.
Current standardized ingress stores unbaselined epochs (`baseline_mode=none`)
and fails closed on an unimplemented baseline transform. QC cache features are
unlabeled and fold-independent; every threshold, global scale calibration, and
epoch acceptance decision is fitted only on the outer training fold.

## Model Contract

Every classifier exposes `fit(train)`, `predict_logit(test)`, and
`parameter_count()`. It returns one target logit per epoch. The 9-choice layer
sums calibrated logits by `(decision_id, stimulus)` and selects the largest
candidate score.

### N2P3-Net Temporal Evidence

The compact N2P3-Net branch uses an MS-EEGNet-style trunk: an EEGNet-factorized
spatio-temporal block followed by two compressed separable temporal summaries.
Its default LMBC head receives the dataset's physical `sfreq` and `tmin`,
contrasts fixed post-stimulus P300 candidates against the pre-stimulus
reference, and softly marginalizes latency independently inside each temporal
scale. `ms_flatten` reproduces the paper-style head and `global_average` is a
matched aggregation ablation. All modes emit the same binary logits and leave
calibration and candidate aggregation unchanged. The complete hypothesis,
equations, counterexamples, and promotion criteria are in
`doc/latency_marginal_pooling.zh.md`.

## Initial Search Space

| Family | Search axes | Promotion condition |
|---|---|---|
| Linear time-domain | regularization, temporal decimation | required floor |
| xDAWN-RG | xDAWN components, covariance shrinkage | required classical baseline |
| EEGNet | temporal kernel, F1, depth multiplier, dropout | compact neural floor |
| P300-CNN | temporal filters, spatial filters, dropout | P300-specific comparison |
| Multi-scale CNN | 3-4 receptive fields, branch width, dropout | initial production candidate |

No auxiliary head is part of the default model. Any additional branch is an
independent candidate and must earn inclusion through the same nested protocol.

## Evaluation

The outer test fold is never used for tuning. Inner grouped folds choose
preprocessing, calibration, threshold-independent metrics, and hyperparameters.
Use a fixed seed manifest and subject/run identifiers. Bootstrap confidence
intervals resample at the outer independent-group level.

The champion minimizes compute subject to a preregistered non-inferiority
margin on AUC/BACC and decision hit rate. Measure inference in evaluation mode
after warm-up, with batch size 1 and the deployed decision batch size.
