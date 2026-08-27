# N2P3-Net Constitution

Version 3, 2026-08-27. This is the sole project contract.

## Objective

For oddball EEG cognitive decisions, maximize held-out P300 target detection
and 9-choice decision performance under small, heterogeneous datasets. Primary
metrics are AUC, BACC, decision hit rate, calibration cost, latency, peak
memory, and parameter count.

## Non-negotiable Rules

P1. The unit of scientific evidence is a held-out group, not a shuffled epoch.
All validation is grouped by run/session; LOSO and cross-dataset protocols are
required when their data exist.

P2. A model must beat xDAWN-RG and a regularized linear time-domain baseline
under the identical preprocessing, split, and decision aggregation contract.

P3. The default neural candidates are compact P300 CNN, EEGNet, and lightweight
multi-scale CNN. Capacity is an empirically tuned budget, not a goal. Larger
architectures need a measured Pareto improvement.

P4. Labels, preprocessing estimates, normalizers, calibrators, and tuning must
remain inside the training fold. Subject/session identifiers cannot be model
features unless the deployment protocol supplies them and an ablation proves
their legitimate benefit.

P5. Source formats remain first-class. Each adapter emits a documented common
epoch artifact with channel and provenance metadata; no adapter may silently
invent missing channels or labels.

P6. Every classifier produces one target logit per epoch. Decision inference
aggregates calibrated logits by candidate within its declared decision set.

P7. Device code is portable: CUDA, XPU, then CPU; no hard-coded device calls.
Every promoted model has measured inference latency and peak memory on its
declared device.

## Prohibited

- Random epoch-level train/test splits when run/session data are available.
- Reporting imbalanced raw accuracy as the primary result.
- Claiming cross-subject or cross-dataset gains from within-subject results.
- Adding a model family without a grouped-validation ablation against the
  current compact-model portfolio.
