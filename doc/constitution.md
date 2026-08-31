# N2P3-Net Constitution

Version 5, 2026-08-31. This document contains stable scientific and engineering
principles; it is not the source of mutable defaults or model status. Executable
code/tests and attested run artifacts take precedence. The living hypothesis,
evidence, GTN development plan, and BrainSync promotion plan is `research_program.zh.md`.

## Objective

For oddball EEG cognitive decisions, maximize held-out P300 target detection
and 9-choice decision performance under small, heterogeneous datasets. Primary
metrics are AUC, BACC, decision hit rate, calibration cost, latency, peak
memory, and parameter count.

## Non-negotiable Rules

P1. The unit of scientific evidence is a held-out group, not a shuffled epoch.
All validation is grouped by run/session; LOSO and cross-dataset protocols are
required when their data exist.

For a calibrated unknown-choice claim, calibration and test are different
decisions with a changed/unknown target. Splitting repetitions inside one fixed
target decision is an explicitly labelled oracle personalization proxy.

P2. A model must beat xDAWN-RG and a regularized linear time-domain baseline
under the identical preprocessing, split, and decision aggregation contract.

P3. The default neural candidates are compact P300 CNN, EEGNet, and lightweight
multi-scale CNN. Capacity is an empirically tuned budget, not a goal. Larger
architectures need a measured Pareto improvement.

P4. Labels, preprocessing estimates, normalizers, calibrators, and tuning must
remain inside the training fold. Subject/session identifiers cannot be model
features unless the deployment protocol supplies them and an ablation proves
their legitimate benefit.

Target-excluded zero-shot uses no target prefix state. Unlabelled or pseudo-label
adaptation is reported separately from supervised calibration.

P5. Source formats remain first-class. Each adapter emits a documented common
epoch artifact with channel and provenance metadata; no adapter may silently
invent missing channels or labels.

P6. Every classifier produces one target logit per epoch. Decision inference
aggregates calibrated logits by candidate within its declared decision set.

Candidate-local occurrence indices are not synchronous rounds. Missing evidence,
ties, abstentions, and acquisition failures remain visible in coverage and
operational denominators.

P7. Device code is portable: CUDA, XPU, then CPU; no hard-coded device calls.
Every promoted model has measured inference latency and peak memory on its
declared device.

P8. Online causal filtering declares and persists its initial state. Whole-record
zero-phase filtering before chronological splitting is forbidden; offline
split-local zero-phase is a separate accuracy arm with boundary-safe padding.

## Prohibited

- Random epoch-level train/test splits when run/session data are available.
- Reporting imbalanced raw accuracy as the primary result.
- Claiming cross-subject or cross-dataset gains from within-subject results.
- Claiming unknown-number calibration from labels of the same fixed target decision.
- Breaking score ties by candidate label or silently dropping failed decisions.
- Adding a model family without a grouped-validation ablation against the
  current compact-model portfolio.
