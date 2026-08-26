# N2P3-Net archives

Archives are immutable historical snapshots. They are not importable source
trees and must not be added to `PYTHONPATH`.

| Archive | Purpose | SHA-256 |
|---|---|---|
| `milestone_pre_glm_2026-08-24.tar.gz` | Pre-GLM milestone, excluding all caches | `ff2d31684081f21e9e7e5fe3972ec278028c9081898c79192deb0249a6f23a94` |
| `historical_experiments_pre_generic_2026-08-24.tar.gz` | Historical run records, diagnostics, and dataset-specific ingress removed by the generic EEG interface; all caches are excluded | `ca524b872376864f3a818d043ee87ac1bf24dfd71272260253c3e20bdf32a973` |
| `legacy_v11_docs_2026-08-25/` | v11 strict-past guidance docs (blueprint, issue2 decision record, routes, recipe, constitution, evaluation protocol, roadmap) superseded by doc/blueprint.md v12 | directory |

The second archive contains its own `README.md` and `MANIFEST.sha256` with one
entry per file. Restore only into a separate reproduction workspace.
