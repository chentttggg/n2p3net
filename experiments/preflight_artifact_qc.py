"""Audit fold-local artifact QC before a long LOSO training run."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.evaluate import loso_folds  # noqa: E402
from data.artifact import FoldLocalArtifactPolicy, parse_candidate_quantiles  # noqa: E402
from data.epochs import load_epoch_dataset  # noqa: E402


def _parse_quantiles(value: str) -> tuple[float, ...]:
    try:
        return parse_candidate_quantiles(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _audit_fold_partition(
    dataset_cache: str,
    policy_values: dict[str, object],
    fold_indices: list[int],
) -> list[dict[str, object]]:
    dataset = load_epoch_dataset(dataset_cache, require_labels=True)
    folds = loso_folds(dataset.subject_ids)
    policy = FoldLocalArtifactPolicy(**policy_values)
    results = []
    for fold_index in fold_indices:
        train, test = folds[fold_index]
        fitted = policy.fit(dataset.X[train], dataset.subject_ids[train])
        transformed = fitted.transform(dataset.X[test])
        results.append(
            {
                "fold": fold_index,
                "held_out_subjects": sorted(set(dataset.subject_ids[test].astype(str))),
                "n_test_epochs": int(test.sum()),
                "n_all_channels_bad": int(transformed.all_channels_bad.sum()),
                "n_epochs_over_bad_channel_limit": int(transformed.drop_epoch_mask.sum()),
                "selected_quantiles": fitted.selected_quantiles.tolist(),
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
    parser.add_argument("--artifact-max-bad-channel-fraction", type=float, default=0.25)
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
        max_bad_channel_fraction=args.artifact_max_bad_channel_fraction,
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
