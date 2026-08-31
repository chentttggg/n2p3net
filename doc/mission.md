# Mission

Build an oddball EEG decoder that identifies target trials and selects the
intended item from a 9-choice decision set. The deployment population is adult
BrainSync EEG, principally 8 scalp channels; EOG remains an independent QC
reference rather than a classifier channel.

The product objective is subject-macro `hit@R >= 0.90` after a fixed absolute
calibration budget, evaluated on later independent decisions whose target digit
was unknown during calibration. Each subject needs multiple target-switch test
decisions; failed, incomplete, and abstained decisions remain in the denominator.

Development also reports grouped within-subject/session, leave-one-subject-out,
and cross-dataset protocols where possible.
Each result reports trial AUC, BACC, 9-choice hit rate by repetition count,
parameter count, median/p95 inference latency, and peak memory. No universal
threshold is asserted before data are collected; the first milestone is a
leakage-free baseline.

The project is an EEG decoding system and does not make cognitive or medical
diagnostic claims. GTN is a child/3-channel development benchmark, not evidence
for adult BrainSync deployment or per-person long-run 90% accuracy.
