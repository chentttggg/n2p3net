"""Audit fold-local artifact QC before a long LOSO training run."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.evaluate import loso_folds  # noqa: E402
from data.artifact import (  # noqa: E402
    FoldLocalArtifactPolicy,
    parse_candidate_bad_channel_fractions,
    parse_candidate_quantiles,
)
from data.epochs import load_epoch_dataset  # noqa: E402


def _parse_quantiles(value: str) -> tuple[float, ...]:
    try:
        return parse_candidate_quantiles(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_bad_channel_fractions(value: str) -> tuple[float, ...]:
    try:
        return parse_candidate_bad_channel_fractions(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _audit_fold_partition(
    dataset_cache: str,
    policy_values: dict[str, object],
    fold_indices: list[int],
) -> list[dict[str, object]]:
    dataset = load_epoch_dataset(dataset_cache, require_labels=True)
    if dataset.qc_features is None:
        raise ValueError("Dataset cache lacks QC v2 features; regenerate before preflight.")
    trial_channel_mask = (
        dataset.trial_channel_mask
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
    )
    folds = loso_folds(dataset.subject_ids)
    policy = FoldLocalArtifactPolicy(**policy_values)
    results = []
    for fold_index in fold_indices:
        train, test = folds[fold_index]
        fitted = policy.fit(
            dataset.X[train],
            dataset.subject_ids[train],
            trial_channel_mask[train],
            dataset.qc_features.subset(train),
        )
        transformed = fitted.transform(
            dataset.X[test],
            trial_channel_mask[test],
            dataset.qc_features.subset(test),
        )
        results.append(
            {
                "fold": fold_index,
                "held_out_subjects": sorted(set(dataset.subject_ids[test].astype(str))),
                "n_test_epochs": int(test.sum()),
                "n_all_channels_bad": int(transformed.all_channels_bad.sum()),
                "n_epochs_over_bad_channel_limit": int(transformed.drop_epoch_mask.sum()),
                "selected_quantiles": fitted.selected_quantiles.tolist(),
                "selected_bad_channel_fraction": fitted.selected_bad_channel_fraction,
                "n_global_epoch_scale": int(transformed.global_epoch_scale_mask.sum()),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--artifact-candidate-quantiles",
        type=_parse_quantiles,
        default=(0.90, 0.95, 0.975, 0.99),
    )
    parser.add_argument("--artifact-flat-quantile", type=float, default=0.005)
    parser.add_argument(
        "--artifact-candidate-bad-channel-fractions",
        type=_parse_bad_channel_fractions,
        default=(0.125, 0.25, 0.375, 0.5),
    )
    parser.add_argument("--artifact-global-scale-mad-z", type=float, default=6.0)
    parser.add_argument("--artifact-min-training-epoch-retention", type=float, default=0.70)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("--max-folds must be positive")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    policy = FoldLocalArtifactPolicy(
        candidate_quantiles=args.artifact_candidate_quantiles,
        flat_quantile=args.artifact_flat_quantile,
        candidate_bad_channel_fractions=args.artifact_candidate_bad_channel_fractions,
        global_scale_mad_z=args.artifact_global_scale_mad_z,
        min_training_epoch_retention=args.artifact_min_training_epoch_retention,
    )
    policy.validate()
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True)
    folds = loso_folds(dataset.subject_ids)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    fold_indices = list(range(len(folds)))
    partitions = [
        fold_indices[offset :: min(args.jobs, len(fold_indices))]
        for offset in range(min(args.jobs, len(fold_indices)))
    ]
    policy_values = asdict(policy)
    if len(partitions) == 1:
        results = _audit_fold_partition(args.dataset_cache, policy_values, partitions[0])
    else:
        with ProcessPoolExecutor(max_workers=len(partitions)) as executor:
            results = [
                result
                for partition in executor.map(
                    _audit_fold_partition,
                    [args.dataset_cache] * len(partitions),
                    [policy_values] * len(partitions),
                    partitions,
                )
                for result in partition
            ]
    results.sort(key=lambda result: int(result["fold"]))
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    failed = [result for result in results if result["n_all_channels_bad"]]
    summary = {
        "dataset_cache": str(args.dataset_cache),
        "n_folds": len(results),
        "policy": policy.__dict__,
        "jobs": min(args.jobs, len(fold_indices)),
        "n_failed_folds": len(failed),
        "failed_folds": failed,
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    if args.output is not None:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit("artifact QC preflight found held-out all-channel failures")


if __name__ == "__main__":
    main()
