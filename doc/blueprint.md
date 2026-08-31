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
There is no universal P300 recipe. Executable profiles currently include:

| profile | purpose | current axes |
|---|---|---|
| default offline | adult/BI development | 128 Hz, 2-30 Hz, `[-200,800) ms`, zero phase |
| GTN performance | child/3-channel development | 128 Hz, 0.1 Hz, 1200 ms; matched 2x2 completed |
| GTN paper anchor | source-paper comparison | 128 Hz, 0.5 Hz, 1200 ms anchor |
| causal | chronological online estimate | forward IIR with persisted steady-state initial state |

All profiles execute per-trial, per-channel `[-200,0) ms` mean correction.
Native acquisition remains first-class (250 Hz for BrainSync); raw samples and
event indices are preserved while only the derived model tensor is resampled.
Every adapter executes continuous fourth-order IIR, source-sample epoching,
then epoch-domain resampling. Offline zero-phase and online forward causal are
different estimands, not interchangeable implementation details. QC
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
`parameter_count()`. It returns one target logit per epoch. For classifiers
trained with weighted CE, candidate evidence first removes the analytic
class-weight/training-prior offset. Any optional score rescaling uses a positive
temperature `T>0`; no calibrator may reverse candidate order. The 9-choice layer
aggregates by `(decision_id, stimulus)` under a declared mode. When candidate
counts differ, the current development default uses all trials and ranks the
candidate means (`count_power=0`). Raw `sum`, square-root count tempering and
balanced fixed-count truncation remain explicit controls. At a fixed equal R,
mean and sum have the same argmax. The layer selects the unique largest candidate
score. A tie is an abstention/miss, never a label-based
tie-break. Candidate-local occurrence counts are not synchronous rounds.

### N2P3-Net Temporal Evidence

The compact N2P3-Net branch uses an MS-EEGNet-style trunk: an EEGNet-factorized
spatio-temporal block followed by two compressed separable temporal summaries.
Linear `full_unfold` is the adopted N2P3-Net readout because the preregistered
BI2014a mechanism comparison favored retaining every post-trunk time coordinate;
paper-style `ms_flatten` remains an explicit MS-EEGNet baseline. Adoption does not
make either readout a product-accuracy champion. Architecture records expose both local kernel spans and
the total input-domain receptive field through ST convolution, pooling, and MST
convolution. LMBC uses physical `sfreq`/`tmin` and remains a conditional
latency hypothesis after failing only the BI2014a binary promotion test;
`global_average` is the matched negative control. All modes emit the same
binary logits and leave calibration and candidate aggregation unchanged.
K35 is the provisional default kernel under the adopted `full_unfold` readout.
The completed GTN development comparison removed K33 from the main line and retained
K65 as the unresolved broad reference. The strongest current all-evidence control
is the unchanged v4 checkpoint with candidate-mean scoring (K35 71.43%, K65
68.30%). Thirty-epoch joint backbone/listwise fine-tuning reduced K35 to 68.84%
and is not part of the default recipe. Equations, counterexamples, and the GTN development and
confirmation boundaries are in
`doc/research_program.zh.md`.

Prior-free research heads expose every post-trunk feature/time coordinate to a
linear, factorized-quadratic, or parameter-matched MLP readout. BI2014a supports
the linear full-unfold mechanism, not the quadratic or MLP extensions. Their
equations and historical preregistration are in `doc/prior_free_unfolding.zh.md`.

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

For a calibrated unknown-choice claim, calibration and test are distinct
target-changing decisions. Same-selection labelled prefix/suffix results are
oracle proxies. Only adult BrainSync 9-choice data with multiple target-switch,
later independent decisions can adjudicate the product target. Its primary is
subject-macro `hit@R >= 0.90` with coverage/failures in the denominator and a
fixed absolute calibration budget; AUC/BACC remain supporting metrics. GTN is
development-only and cannot produce a product 90% or confirmatory claim.

Accuracy is the primary optimization objective. Compute is a secondary
constraint unless latency/memory prevents the intended deployment. Measure
inference in evaluation mode after warm-up, with batch size 1 and the deployed
decision batch size.
