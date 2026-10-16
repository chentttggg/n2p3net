# Frozen Evidence Index

This directory tracks manifests, checksums and delivery records for physical
source and evidence snapshots. Historical archive payloads (`*.zip`, `*.tar.gz`,
`*.tar`, `*.7z`, `*.bundle`) stay local and are not distributed through GitHub.
`local_history/` also stays local. A fresh clone therefore does not contain these
archives, raw datasets, experiment caches, model checkpoints or run outputs.
Materialized `checkout/` directories also stay local; only their source manifests
are tracked. Per-event figure data (`doc/assets/**/*.npz`) stays local alongside
the published report figures and metadata.

Manifest archive paths refer to separately retained files. Obtain the matching
archive from its custodian and verify its recorded SHA-256 before inspection or
execution. Do not regenerate an archive from current code and reuse its old
identity. A source snapshot is not evidence that its experiments completed.

Historical `source_commit` values retain their original meaning. Publication
may remove archive payloads from unpublished Git history and change commit IDs;
the original history and commit mapping are retained locally. Snapshot hashes
and historical result identities must not be rewritten to match publication.

Start with the [evidence index](../doc/evidence/README.md) for research records,
or the [active guides](../doc/README.md) for current behavior.
