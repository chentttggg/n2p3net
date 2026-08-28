"""Screen local N2P3 hyperparameter sensitivity without persisting outer-test metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines.deep import DeepConfig  # noqa: E402
from baselines.evaluate import (  # noqa: E402
    evaluate_binary,
    loso_folds,
    precompute_fold_local_artifact_models,
)
from data.artifact import FoldLocalArtifactPolicy  # noqa: E402
from data.contract import (  # noqa: E402
    DEFAULT_P300_DATA_CONTRACT,
    assert_p300_input_contract,
    assert_p300_source_provenance,
)
from data.epochs import load_epoch_dataset  # noqa: E402
from models.n2p3net import (  # noqa: E402
    DEFAULT_N2P3_ARCHITECTURE,
    N2P3ArchitectureConfig,
    scale_architecture_preserving_spans,
)
from train.device import get_device  # noqa: E402
from train.factory import build_binary_model, describe_binary_model  # noqa: E402

DELTAS = (-0.15, -0.05, 0.05, 0.15)


@dataclass(frozen=True)
class SensitivityCandidate:
    name: str
    axis: str
    relative_delta: float
    batch_size: int
    deep_overrides: dict[str, float]
    architecture_overrides: dict[str, float | int]


def _delta_label(delta: float) -> str:
    return f"{'m' if delta < 0 else 'p'}{abs(round(delta * 100)):02d}"


def _nearest_odd(value: float) -> int:
    rounded = max(3, int(round(value)))
    if rounded % 2:
        return rounded
    lower, upper = max(3, rounded - 1), rounded + 1
    return min((lower, upper), key=lambda candidate: (abs(candidate - value), candidate))


def architecture_for_sample_rate(sample_rate_hz: float) -> N2P3ArchitectureConfig:
    if sample_rate_hz == DEFAULT_P300_DATA_CONTRACT.sample_rate_hz:
        return DEFAULT_N2P3_ARCHITECTURE
    return scale_architecture_preserving_spans(
        DEFAULT_N2P3_ARCHITECTURE,
        source_sample_rate_hz=DEFAULT_P300_DATA_CONTRACT.sample_rate_hz,
        target_sample_rate_hz=sample_rate_hz,
    )


def build_candidates(
    *,
    base_batch_size: int,
    base_architecture: N2P3ArchitectureConfig = DEFAULT_N2P3_ARCHITECTURE,
) -> list[SensitivityCandidate]:
    """Return one-factor candidates at 0%, +/-5%, and +/-15%, with integer deduplication."""

    deep = DeepConfig(batch_size=base_batch_size)
    architecture = base_architecture
    candidates = [
        SensitivityCandidate(
            name="baseline",
            axis="baseline",
            relative_delta=0.0,
            batch_size=base_batch_size,
            deep_overrides={},
            architecture_overrides={},
        )
    ]
    axes: tuple[tuple[str, str, float | int, object], ...] = (
        ("lr", "deep", deep.lr, lambda value: float(value)),
        ("weight_decay", "deep", deep.weight_decay, lambda value: float(value)),
        ("pos_weight", "deep", deep.pos_weight, lambda value: float(value)),
        ("batch_size", "batch", base_batch_size, lambda value: max(1, int(round(value)))),
        ("dropout", "architecture", architecture.dropout, lambda value: float(value)),
        (
            "spatial_max_norm",
            "architecture",
            architecture.spatial_max_norm,
            lambda value: float(value),
        ),
        (
            "temporal_kernel_size",
            "architecture",
            architecture.temporal_kernel_size,
            _nearest_odd,
        ),
        (
            "temporal_filters",
            "architecture",
            architecture.temporal_filters,
            lambda value: max(1, int(round(value))),
        ),
    )
    for axis, owner, base_value, transform in axes:
        seen = {base_value}
        for delta in DELTAS:
            value = transform(float(base_value) * (1.0 + delta))
            if value in seen:
                continue
            seen.add(value)
            candidates.append(
                SensitivityCandidate(
                    name=f"{axis}_{_delta_label(delta)}",
                    axis=axis,
                    relative_delta=delta,
                    batch_size=int(value) if owner == "batch" else base_batch_size,
                    deep_overrides={axis: float(value)} if owner == "deep" else {},
                    architecture_overrides={axis: value} if owner == "architecture" else {},
                )
            )
    return candidates


def _parse_fold_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fold indices must be comma-separated integers") from exc
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        raise argparse.ArgumentTypeError("fold indices must be unique non-negative integers")
    return indices


def _subject_sort_key(subject: str) -> tuple[int, int | str]:
    stripped = subject.strip()
    return (0, int(stripped)) if stripped.isdigit() else (1, stripped)


def main() -> None:
    defaults = DeepConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--run-dir", default="experiments/runs")
    parser.add_argument("--run-name", default="n2p3_local_sensitivity")
    parser.add_argument("--subjects", type=int, default=32)
    parser.add_argument("--sample-rate-hz", type=float, choices=(128.0, 256.0), default=128.0)
    parser.add_argument("--fold-indices", type=_parse_fold_indices, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--early-stop-patience", type=int, default=defaults.early_stop_patience)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--fold-jobs", type=int, default=4)
    parser.add_argument("--gpu-fold-jobs", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--artifact-qc-jobs", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.early_stop_patience < 1:
        parser.error("epochs, batch size, and patience must be positive")
    if args.max_candidates is not None and args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    if args.subjects is not None and args.subjects < 2:
        parser.error("--subjects must be at least two")

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    expected_contract = replace(
        DEFAULT_P300_DATA_CONTRACT,
        sample_rate_hz=args.sample_rate_hz,
    )
    assert_p300_input_contract(dataset.preprocessing, expected_contract)
    assert_p300_source_provenance(dataset)
    if dataset.qc_features is None:
        raise ValueError("Sensitivity screening requires cached QC v2 features.")
    X, y = dataset.X, dataset.y
    subject_ids = np.asarray(dataset.subject_ids).astype(str)
    trial_channel_mask = (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(np.asarray(dataset.channel_mask, dtype=bool), X.shape[:2]).copy()
    )
    qc_features = dataset.qc_features
    selected_subjects = np.asarray(
        sorted(np.unique(subject_ids).tolist(), key=_subject_sort_key)
    )
    if args.subjects is not None:
        selected_subjects = selected_subjects[: args.subjects]
        keep = np.isin(subject_ids, selected_subjects)
        X, y, subject_ids = X[keep], y[keep], subject_ids[keep]
        trial_channel_mask = trial_channel_mask[keep]
        qc_features = qc_features.subset(keep)
    all_folds = loso_folds(subject_ids)
    fold_indices = args.fold_indices
    if fold_indices is None:
        fold_indices = tuple(
            np.linspace(0, len(all_folds) - 1, num=min(8, len(all_folds)), dtype=int).tolist()
        )
    if max(fold_indices) >= len(all_folds):
        parser.error(f"fold index exceeds the available {len(all_folds)} LOSO folds")
    folds = [all_folds[index] for index in fold_indices]
    subjects = np.asarray(sorted(np.unique(subject_ids).tolist(), key=_subject_sort_key))
    anchor_subjects = [str(subjects[index]) for index in fold_indices]
    artifact_policy = FoldLocalArtifactPolicy()
    fitted_artifact_models = precompute_fold_local_artifact_models(
        X,
        subject_ids,
        folds,
        trial_channel_mask=trial_channel_mask,
        qc_features=qc_features,
        artifact_policy=artifact_policy,
        artifact_qc_jobs=args.artifact_qc_jobs,
        cpu_threads=args.cpu_threads,
    )
    device = get_device() if args.device == "auto" else torch.device(args.device)
    base_architecture = architecture_for_sample_rate(args.sample_rate_hz)
    candidates = build_candidates(
        base_batch_size=args.batch_size,
        base_architecture=base_architecture,
    )
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    output_root = Path(args.run_dir) / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "n2p3net_local_sensitivity/1",
        "selection_metric": "mean_inner_best_validation_loss",
        "outer_test_metrics_persisted": False,
        "deltas": [0.0, -0.05, 0.05, -0.15, 0.15],
        "selected_subject_count": len(selected_subjects),
        "selected_subjects": selected_subjects.tolist(),
        "selected_epoch_count": len(X),
        "sample_rate_hz": args.sample_rate_hz,
        "base_architecture": base_architecture.model_kwargs(),
        "fold_indices": list(fold_indices),
        "anchor_subjects": anchor_subjects,
        "epochs": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "seed": args.seed,
        "candidate_count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
        "dataset": dataset.record(validate=False),
        "started_utc": datetime.now(UTC).isoformat(),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_dir = output_root / candidate.name
        record_path = candidate_dir / "record.json"
        if record_path.exists():
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
            continue
        candidate_dir.mkdir(parents=True, exist_ok=True)
        architecture: N2P3ArchitectureConfig = replace(
            base_architecture, **candidate.architecture_overrides
        )
        overrides = {
            "lr": defaults.lr,
            "weight_decay": defaults.weight_decay,
            "pos_weight": defaults.pos_weight,
            "early_stop_patience": args.early_stop_patience,
            **candidate.deep_overrides,
        }
        model = build_binary_model(
            "n2p3net_full_unfold",
            dataset,
            epochs=args.epochs,
            batch_size=candidate.batch_size,
            seed=args.seed,
            validation_group_fraction=0.1,
            deep_config_overrides=overrides,
            device=device,
            n2p3net_architecture=architecture,
        )
        model.configure_epoch_progress(candidate_dir / "epochs")
        started = time.perf_counter()
        summary = evaluate_binary(
            model,
            X,
            y,
            subject_ids,
            folds,
            fold_protocol="inner_sensitivity_screen",
            n_jobs=args.fold_jobs,
            parallel_backend="process",
            max_gpu_jobs=args.gpu_fold_jobs,
            cpu_threads=args.cpu_threads,
            artifact_qc_jobs=args.artifact_qc_jobs,
            trial_channel_mask=trial_channel_mask,
            qc_features=qc_features,
            artifact_policy=artifact_policy,
            fitted_artifact_models=fitted_artifact_models,
        )
        inner_losses = np.asarray(
            [fold.best_task_val_loss for fold in summary.per_fold], dtype=float
        )
        inner_aucs = np.asarray(
            [fold.final_task_val_auc for fold in summary.per_fold], dtype=float
        )
        if not np.isfinite(inner_losses).all() or not np.isfinite(inner_aucs).all():
            raise RuntimeError(f"Candidate {candidate.name} produced non-finite inner metrics.")
        record = {
            "candidate_index": index,
            "candidate": asdict(candidate),
            "architecture": architecture.model_kwargs(),
            "model": describe_binary_model("n2p3net_full_unfold", model),
            "mean_inner_best_validation_loss": float(inner_losses.mean()),
            "mean_inner_final_validation_auc": float(inner_aucs.mean()),
            "mean_epochs": float(np.mean([fold.epochs_ran for fold in summary.per_fold])),
            "wall_seconds": time.perf_counter() - started,
            "per_fold": [
                {
                    "anchor_fold_index": int(fold_index),
                    "anchor_subject": subject,
                    "best_validation_loss": float(result.best_task_val_loss),
                    "final_validation_auc": float(result.final_task_val_auc),
                    "epochs_ran": result.epochs_ran,
                    "batch_size": result.batch_size,
                    "oom_retries": result.oom_retries,
                }
                for fold_index, subject, result in zip(
                    fold_indices, anchor_subjects, summary.per_fold, strict=True
                )
            ],
            "finished_utc": datetime.now(UTC).isoformat(),
        }
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
        print(
            f"[{index}/{len(candidates)}] {candidate.name} "
            f"val_loss={record['mean_inner_best_validation_loss']:.6f} "
            f"val_auc={record['mean_inner_final_validation_auc']:.6f}",
            flush=True,
        )
    ranking = sorted(
        records,
        key=lambda record: (
            record["mean_inner_best_validation_loss"],
            -record["mean_inner_final_validation_auc"],
        ),
    )
    (output_root / "screening.json").write_text(
        json.dumps(
            {
                **manifest,
                "finished_utc": datetime.now(UTC).isoformat(),
                "ranking": ranking,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
