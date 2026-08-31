# BI+BNCI uniform joint evidence

This compact audit package records the frozen two-arm BI-only versus uniform
BI+BNCI common-CAR5 experiment. The primary analysis is `analysis.json`; checkpoint
metadata, source/result hashes, input snapshots, and the source commit are retained
alongside it.

The original cloud archive was 8,887 bytes with SHA-256
`cf8d6918759454a1d3eb41b6bc4754547f874ac018b6c6623bec432cb41b2904`.
It does not contain the source `.npz` caches, checkpoint `.pt` files, or per-run
result JSON. Their hashes are audit pointers only. This directory is therefore an
auditable result package, not an independently rerunnable package.
