# GTN 2026-08-31 Evidence Boundary

This directory is a compact, Git-tracked audit index. It is not a standalone
reproduction bundle.

Included:

- frozen subject/block and experiment manifests;
- cache/checkpoint identities and runtime hashes;
- aggregate and paired result records;
- the subject-by-seed kernel summary;
- the frozen kernel evaluator required by its contract tests;
- independent audit and execution provenance.

Not included:

- EEG cache arrays (`.npz`);
- trained checkpoints (`.pt`);
- the complete trial ledgers and per-run logs;
- a portable Python/CUDA environment image.

The Git commit containing this file is the authoritative source identity for
the current runners. Hash records for omitted cache/checkpoint objects are
attestations only: they can verify an externally supplied object, but cannot
recreate it. A full rerun requires the external objects named by the manifests
and must reproduce their SHA-256 values before execution.

The former expanded audit directory and duplicate tar archives were moved out
of the repository. Keeping a million lines of repeated result JSON in Git would
not improve reproducibility without the corresponding EEG and model binaries.

Evidence status: GTN development evidence, conditional on the frozen cohort,
blocks, seeds, checkpoints, and preprocessing. It is not independent adult
BrainSync confirmation and does not establish the 90% product endpoint.
