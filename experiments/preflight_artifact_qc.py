"""Audit fold-local artifact QC before a long LOSO training run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.evaluate import (  # noqa: E402
    loso_folds,
    precompute_fold_local_artifact_models,
)
from data.artifact import (  # noqa: E402
    FoldLocalArtifactPolicy,
    parse_candidate_bad_channel_fractions,
    parse_candidate_quantiles,
)
from data.epochs import (  # noqa: E402
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
    write_epoch_dataset_record,
)


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


def _audit_fitted_folds(
    X: np.ndarray,
    subject_ids: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    trial_channel_mask: np.ndarray,
    qc_features: object,
    fitted_models: dict[int, object],
) -> list[dict[str, object]]:
    results = []
    for fold_index, (_, test) in enumerate(folds):
        fitted = fitted_models[fold_index]
        transformed = fitted.transform(
            X[test],
            trial_channel_mask[test],
            qc_features.subset(test),
        )
        results.append(
            {
                "fold": fold_index,
                "held_out_subjects": sorted(set(subject_ids[test].astype(str))),
                "n_test_epochs": int(test.sum()),
                "n_all_channels_bad": int(transformed.all_channels_bad.sum()),
                "n_epochs_over_bad_channel_limit": int(transformed.drop_epoch_mask.sum()),
                "selected_quantiles": fitted.selected_quantiles.tolist(),
                "selected_bad_channel_fraction": fitted.selected_bad_channel_fraction,
                "n_global_epoch_scale": int(transformed.global_epoch_scale_mask.sum()),
            }
        )
    return results


def _load_verified_preflight_dataset(path: str, *, full_contract_check: bool):
    if full_contract_check:
        scanned = load_epoch_dataset(path, require_labels=True, validation="full")
        write_epoch_dataset_record(path, scanned, already_validated=True)
    dataset = load_epoch_dataset(path, require_labels=True, validation="attested")
    return dataset, loaded_epoch_cache_attestation(dataset)


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
    parser.add_argument(
        "--full-contract-check",
        action="store_true",
        help="Re-run the full tensor/event/QC contract scan before QC preflight.",
    )
    parser.add_argument(
        "--artifact-qc-jobs",
        "--jobs",
        dest="artifact_qc_jobs",
        type=int,
        default=1,
        help="CPU processes used to fit independent outer-fold QC policies.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Total CPU thread budget shared by artifact QC workers.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("--max-folds must be positive")
    if args.artifact_qc_jobs < 1:
        parser.error("--artifact-qc-jobs must be positive")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive when set")
    policy = FoldLocalArtifactPolicy(
        candidate_quantiles=args.artifact_candidate_quantiles,
        flat_quantile=args.artifact_flat_quantile,
        candidate_bad_channel_fractions=args.artifact_candidate_bad_channel_fractions,
        global_scale_mad_z=args.artifact_global_scale_mad_z,
        min_training_epoch_retention=args.artifact_min_training_epoch_retention,
    )
    policy.validate()
    dataset, verified_load = _load_verified_preflight_dataset(
        args.dataset_cache,
        full_contract_check=args.full_contract_check,
    )
    folds = loso_folds(dataset.subject_ids)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if dataset.qc_features is None:
        raise ValueError("Dataset cache lacks QC v2 features; regenerate before preflight.")
    trial_channel_mask = (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(np.asarray(dataset.channel_mask, dtype=bool), dataset.X.shape[:2])
    )
    fitted_models = precompute_fold_local_artifact_models(
        dataset.X,
        dataset.subject_ids,
        folds,
        trial_channel_mask=trial_channel_mask,
        qc_features=dataset.qc_features,
        artifact_policy=policy,
        artifact_qc_jobs=args.artifact_qc_jobs,
        cpu_threads=args.cpu_threads,
    )
    results = _audit_fitted_folds(
        dataset.X,
        dataset.subject_ids,
        folds,
        trial_channel_mask,
        dataset.qc_features,
        fitted_models,
    )
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    failed = [result for result in results if result["n_all_channels_bad"]]
    summary = {
        "dataset_cache": str(args.dataset_cache),
        "dataset_cache_verified_load": verified_load,
        "n_folds": len(results),
        "policy": policy.__dict__,
        "artifact_qc_jobs": min(args.artifact_qc_jobs, len(folds)),
        "cpu_threads": args.cpu_threads,
        "full_contract_check": args.full_contract_check,
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
