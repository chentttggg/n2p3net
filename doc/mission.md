# Mission

Build an offline oddball EEG decoder that identifies target trials and selects
the intended item from a 9-choice decision set. The target population is adult
EEG, principally 8 channels, while adapters support non-uniform recordings and
channel subsets.

Success is a reproducible result table under three protocols: grouped
within-subject/session, leave-one-subject-out, and cross-dataset where possible.
Each result reports trial AUC, BACC, 9-choice hit rate by repetition count,
parameter count, median/p95 inference latency, and peak memory. No universal
threshold is asserted before data are collected; the first milestone is a
leakage-free baseline.

The project is limited to offline EEG decoding and does not make cognitive or
medical diagnostic claims.
