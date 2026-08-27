# Performance Roadmap

## Phase 0: Research and Contract

Complete the literature catalog, data manifest schema, format adapters, and
grouped split audit. Exit only when every available source exposes subject and
run/session provenance or is explicitly excluded from grouped claims.

## Phase 1: Baselines

Implement regularized time-domain logistic regression and xDAWN-RG. Establish
within-subject grouped, LOSO, and decision-level aggregation reports.

## Phase 2: Compact Neural Search

Implement EEGNet, P300-CNN, and multi-scale CNN behind one classifier
interface. Run nested grouped selection, then publish the complete Pareto
table.

## Phase 3: Transfer and Deployment Cost

If an external corpus is available, test pretrain/fine-tune and calibration
budgets against scratch training and xDAWN-RG. Profile CPU/CUDA/XPU latency,
memory, and failure behavior.

## Phase 4: Confirmation

Freeze the winning configuration, rerun untouched outer tests, calculate
group-level confidence intervals, and export reproducible artifacts. Remove
any module that has no supported performance contribution.
