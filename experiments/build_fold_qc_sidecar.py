"""Build or load one persistent fold-local artifact-QC sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from baselines.evaluate import loso_folds, resolve_fold_local_artifact_models  # noqa: E402
from data.artifact import FoldLocalArtifactPolicy  # noqa: E402
from data.epochs import (  # noqa: E402
    load_epoch_dataset,
    read_epoch_cache_attestation,
)


def _subject_key(subject: str) -> tuple[int, int | str]:
    stripped = subject.strip()
    return (0, int(stripped)) if stripped.isdigit() else (1, stripped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--subjects", type=int, default=None)
    parser.add_argument("--artifact-qc-jobs", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=32)
    args = parser.parse_args()
    if args.subjects is not None and args.subjects < 2:
        parser.error("--subjects must be at least two")

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    X = dataset.X
    subject_ids = np.asarray(dataset.subject_ids).astype(str)
    trial_channel_mask = (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(np.asarray(dataset.channel_mask, dtype=bool), X.shape[:2]).copy()
    )
    qc_features = dataset.qc_features
    if qc_features is None:
        raise ValueError("Fold-QC sidecars require cached QC features.")
    if args.subjects is not None:
        subjects = sorted(np.unique(subject_ids).tolist(), key=_subject_key)[: args.subjects]
        keep = np.isin(subject_ids, subjects)
        X, subject_ids = X[keep], subject_ids[keep]
        trial_channel_mask = trial_channel_mask[keep]
        qc_features = qc_features.subset(keep)
    models, record = resolve_fold_local_artifact_models(
        X,
        subject_ids,
        loso_folds(subject_ids),
        cache_path=args.dataset_cache,
        cache_sha256=str(read_epoch_cache_attestation(args.dataset_cache)["sha256"]),
        trial_channel_mask=trial_channel_mask,
        qc_features=qc_features,
        artifact_policy=FoldLocalArtifactPolicy(),
        artifact_qc_jobs=args.artifact_qc_jobs,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps({**record, "fold_count": len(models)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
