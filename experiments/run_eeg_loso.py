"""Run performance-first LOSO P300 detection from an EpochDataset cache."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines.evaluate import (  # noqa: E402
    evaluate_binary,
    evaluate_candidate_selection,
    loso_folds,
)
from data.artifact import (  # noqa: E402
    FoldLocalArtifactPolicy,
    parse_candidate_bad_channel_fractions,
    parse_candidate_quantiles,
)
from data.contract import (  # noqa: E402
    assert_default_p300_input_contract,
    assert_p300_source_provenance,
)
from data.epochs import load_epoch_dataset  # noqa: E402
from models.n2p3net import POOLING_MODES  # noqa: E402
from train.device import get_device  # noqa: E402
from train.factory import (  # noqa: E402
    BINARY_MODEL_NAMES,
    build_binary_model,
    describe_binary_model,
)

RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
DEFAULT_MODELS = "swdla,window_lr,template,xdawn_rg,n2p3net,eegnet,inception"


def _safe_auto_run_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-") or "dataset"


def _validate_run_name(value: str) -> str:
    normalized = value.replace("\\", "/")
    if not RUN_NAME.fullmatch(normalized) or any(
        part in {"", ".", ".."} for part in normalized.split("/")
    ):
        raise ValueError("run name must contain safe relative path components using letters, digits, ._-")
    return normalized


def _parse_models(value: str, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    names = tuple(name.strip().lower() for name in value.split(",") if name.strip())
    if not names:
        parser.error("--models must contain at least one registered model")
    if len(names) != len(set(names)):
        parser.error("--models must not contain duplicate names")
    unknown = set(names) - set(BINARY_MODEL_NAMES)
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")
    return names


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


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return get_device()
    try:
        device = torch.device(choice)
    except RuntimeError as error:
        raise ValueError(f"Invalid --device value {choice!r}.") from error
    if device.type not in {"cuda", "xpu", "cpu"}:
        raise ValueError("--device must be auto, cpu, cuda[:INDEX], or xpu[:INDEX].")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("--device xpu was requested, but XPU is unavailable.")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device.index} is unavailable.")
    if device.type == "cpu" and device.index is not None:
        raise ValueError("CPU device indices are not supported.")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--validation-group-fraction", type=float, default=0.1)
    parser.add_argument(
        "--fold-jobs",
        type=int,
        default=2,
        help="concurrent fold workers; default 2 overlaps CPU preprocessing with GPU work",
    )
    parser.add_argument("--fold-backend", choices=("auto", "process", "thread"), default="auto")
    parser.add_argument(
        "--gpu-fold-jobs",
        type=int,
        default=None,
        help="maximum concurrent GPU fold processes; default is hardware-aware",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="total CPU thread budget shared across active fold workers; default uses the visible quota",
    )
    parser.add_argument(
        "--artifact-qc-jobs",
        type=int,
        default=None,
        help="CPU processes used to precompute fold-local artifact policies; default uses one worker per usable CPU thread, capped at 16",
    )
    parser.add_argument("--fold-offset", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--subjects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default="experiments/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda[:INDEX], or xpu[:INDEX]",
    )
    parser.add_argument(
        "--n2p3net-pooling",
        choices=sorted(POOLING_MODES),
        default="latency_marginal_contrast",
        help="N2P3-Net head: LMBC default, global_average matched ablation, ms_flatten paper-style head.",
    )
    parser.add_argument(
        "--artifact-candidate-bad-channel-fractions",
        type=_parse_bad_channel_fractions,
        default=(0.125, 0.25, 0.375, 0.5),
        help="Training-fold candidates for the maximum local-bad-channel fraction per epoch.",
    )
    parser.add_argument(
        "--artifact-candidate-quantiles",
        type=_parse_quantiles,
        default=(0.90, 0.95, 0.975, 0.99),
        help="Training-fold PTP threshold candidates, from least to most conservative.",
    )
    parser.add_argument(
        "--artifact-flat-quantile",
        type=float,
        default=0.005,
        help="Training-fold per-channel flatline quantile; zero keeps only true minimum-variance cases.",
    )
    parser.add_argument(
        "--artifact-global-scale-mad-z",
        type=float,
        default=6.0,
        help="Outer-training physical epoch-scale gate in robust standard-deviation units.",
    )
    parser.add_argument(
        "--artifact-min-training-epoch-retention",
        type=float,
        default=0.70,
        help="Minimum inner-CV training-epoch retention required for a kappa candidate.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.fold_jobs < 1:
        parser.error("epochs, batch size, and fold jobs must be positive")
    if args.gpu_fold_jobs is not None and args.gpu_fold_jobs < 1:
        parser.error("--gpu-fold-jobs must be positive when set")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive when set")
    if args.artifact_qc_jobs is not None and args.artifact_qc_jobs < 1:
        parser.error("--artifact-qc-jobs must be positive when set")
    if args.fold_offset < 0 or (args.max_folds is not None and args.max_folds < 1):
        parser.error("fold offset must be non-negative and max folds must be positive")
    if args.subjects is not None and args.subjects < 2:
        parser.error("--subjects must be at least two for LOSO")
    models = _parse_models(args.models, parser)
    artifact_policy = FoldLocalArtifactPolicy(
        candidate_quantiles=args.artifact_candidate_quantiles,
        flat_quantile=args.artifact_flat_quantile,
        candidate_bad_channel_fractions=args.artifact_candidate_bad_channel_fractions,
        global_scale_mad_z=args.artifact_global_scale_mad_z,
        min_training_epoch_retention=args.artifact_min_training_epoch_retention,
    )
    artifact_policy.validate()

    dataset = load_epoch_dataset(
        args.dataset_cache,
        require_labels=True,
        validation="attested",
    )
    assert_default_p300_input_contract(dataset.preprocessing)
    assert_p300_source_provenance(dataset)
    timeline = dataset.event_timeline
    trial_channel_mask = (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(np.asarray(dataset.channel_mask, dtype=bool), dataset.X.shape[:2]).copy()
    )
    X, y, subject_ids = dataset.X, dataset.y, dataset.subject_ids
    if dataset.qc_features is None:
        raise ValueError(
            "Dataset cache lacks QC v2 features. Regenerate the cache before training; "
            "legacy caches are not accepted by the QC v2 runner."
        )
    qc_features = dataset.qc_features
    if args.subjects is not None:
        selected_subjects = np.unique(subject_ids)[: args.subjects]
        keep = np.isin(subject_ids, selected_subjects)
        X, y, subject_ids = X[keep], y[keep], subject_ids[keep]
        trial_channel_mask = trial_channel_mask[keep]
        qc_features = qc_features.subset(keep)
        timeline_subjects = np.asarray(timeline.subject_ids).astype(str)
        timeline_groups = np.asarray(timeline.group_ids).astype(str)
        selected_groups = set(timeline_groups[np.isin(timeline_subjects, selected_subjects)].tolist())
        timeline = timeline.subset_groups(selected_groups)

    folds = loso_folds(subject_ids)[args.fold_offset :]
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise ValueError("No LOSO folds remain after --fold-offset/--max-folds.")
    candidate_selection = (
        timeline.encoded_candidate_selection(require_full_chain=True)
        if timeline.supports_full_candidate_chain
        else None
    )
    device = _resolve_device(args.device)
    run_name = (
        _validate_run_name(args.run_name)
        if args.run_name is not None
        else f"{_safe_auto_run_component(dataset.name)}_performance_{datetime.now(UTC):%Y%m%d_%H%M%SZ}"
    )

    for model_name in models:
        model = build_binary_model(
            model_name,
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            validation_group_fraction=args.validation_group_fraction,
            deep_config_overrides={"lr": args.lr, "early_stop_patience": args.early_stop_patience},
            device=device,
            n2p3net_pooling_mode=args.n2p3net_pooling,
        )
        output_dir = Path(args.run_dir) / run_name / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        model.configure_epoch_progress(output_dir / "epochs")
        manifest = {
            "run_name": run_name,
            "model": describe_binary_model(model_name, model),
            "dataset": dataset.record(validate=False),
            "fold_protocol": "partial_loso" if args.fold_offset or args.max_folds else "loso",
            "selection_mode": "candidate_selection" if candidate_selection is not None else "binary_oddball",
            "artifact_quality_policy": {"method": "fold_local_ptp_cv", **artifact_policy.__dict__},
            "environment": {
                "torch": torch.__version__,
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
                "visible_cpu_threads": os.cpu_count(),
            },
            "started_utc": datetime.now(UTC).isoformat(),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        started = time.perf_counter()
        common = {
            "fold_protocol": manifest["fold_protocol"],
            "n_jobs": args.fold_jobs,
            "parallel_backend": args.fold_backend,
            "max_gpu_jobs": args.gpu_fold_jobs,
            "cpu_threads": args.cpu_threads,
            "artifact_qc_jobs": args.artifact_qc_jobs,
            "fold_id_offset": args.fold_offset,
            "trial_channel_mask": trial_channel_mask,
            "qc_features": qc_features,
            "artifact_policy": artifact_policy,
        }
        if candidate_selection is None:
            summary = evaluate_binary(
                model,
                X,
                y,
                subject_ids,
                folds,
                **common,
            )
        else:
            summary = evaluate_candidate_selection(
                model,
                X,
                y,
                candidate_selection.candidate_codes,
                candidate_selection.group_ids,
                candidate_selection.truth_by_group,
                folds,
                candidate_vocab=tuple(range(len(candidate_selection.vocabulary))),
                fit_group_ids=subject_ids,
                event_timeline=timeline,
                **common,
            )
        record = {
            **manifest,
            "balanced_acc_mean": summary.balanced_acc_mean,
            "balanced_acc_std": summary.balanced_acc_std,
            "auc_mean": summary.auc_mean,
            "decision_hit_rate_mean": getattr(summary, "hit_rate_mean", None),
            "primary_decision_hit_rate": getattr(summary, "primary_hit_rate", None),
            "execution": {
                "requested_n_jobs": args.fold_jobs,
                "requested_backend": args.fold_backend,
                "max_gpu_jobs": args.gpu_fold_jobs,
                "requested_cpu_threads": args.cpu_threads,
                "requested_artifact_qc_jobs": args.artifact_qc_jobs,
                "effective_n_jobs": summary.effective_n_jobs,
                "backend": summary.execution_backend,
                "input_transport": summary.input_transport,
                "cpu_threads_per_worker": summary.cpu_threads_per_worker,
                "artifact_qc_workers": summary.artifact_qc_workers,
                "artifact_qc_cpu_threads_per_worker": summary.artifact_qc_cpu_threads_per_worker,
            },
            "per_fold": [fold.__dict__ for fold in summary.per_fold],
            "wall_seconds": time.perf_counter() - started,
            "finished_utc": datetime.now(UTC).isoformat(),
        }
        (output_dir / "record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[{model_name}] bacc={summary.balanced_acc_mean:.4f} "
            f"auc={summary.auc_mean:.4f} wall={record['wall_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
