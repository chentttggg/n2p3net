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
The mainline MS-EEGNet input contract is 128 Hz, 2-30 Hz, `[-200,800) ms`,
with an executed per-trial, per-channel `[-200,0) ms` mean correction. These
values preserve the physical scales of the fixed 65/5/17-sample temporal
kernels. Native acquisition remains first-class (250 Hz for BrainSync); raw
samples and event indices are preserved while only the derived model tensor is
resampled. Every adapter executes the same order: zero-phase fourth-order IIR
on continuous EEG, source-sample epoching, then epoch-domain resampling. QC
cache features are unlabeled and fold-independent; every threshold,
global scale calibration, and epoch acceptance decision is fitted only on the
outer training fold.

The model tensor remains in volts. Per-trial division by a baseline standard
deviation is prohibited before physical QC. Source reference and native sample
rate are provenance fields; datasets with different source references cannot be
concatenated until one explicit common re-reference has been executed and
recorded.

The executable equations, discrete-time conventions, counterexamples, and
source-to-decision code chain are specified in `doc/input_contract_math.zh.md`.

## Model Contract

Every classifier exposes `fit(train)`, `predict_logit(test)`, and
`parameter_count()`. It returns one target logit per epoch. The 9-choice layer
sums calibrated logits by `(decision_id, stimulus)` and selects the largest
candidate score.

### N2P3-Net Temporal Evidence

The compact N2P3-Net branch uses an MS-EEGNet-style trunk: an EEGNet-factorized
spatio-temporal block followed by two compressed separable temporal summaries.
The promoted project head is paper-style `ms_flatten`. LMBC receives the
dataset's physical `sfreq` and `tmin`, contrasts post-stimulus candidates
against the pre-stimulus reference, and remains available as a rejected
research hypothesis. `global_average` is the matched negative control. All
modes emit the same binary logits and leave calibration and candidate
aggregation unchanged. Equations and failure boundaries are in
`doc/latency_marginal_pooling.zh.md`; matched results are in
`doc/ablation_20260828.zh.md`.

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
