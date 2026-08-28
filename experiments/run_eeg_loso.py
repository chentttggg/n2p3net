"""Run performance-first LOSO P300 detection from an EpochDataset cache."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines.deep import (  # noqa: E402
    DEFAULT_DEEP_EPOCHS,
    DEFAULT_EARLY_STOP_PATIENCE,
    DeepConfig,
)
from baselines.evaluate import (  # noqa: E402
    evaluate_binary,
    evaluate_candidate_selection,
    loso_folds,
    resolve_artifact_qc_workers,
    resolve_fold_local_artifact_models,
)
from data.artifact import (  # noqa: E402
    FoldLocalArtifactPolicy,
    parse_candidate_bad_channel_fractions,
    parse_candidate_quantiles,
)
from data.contract import (  # noqa: E402
    DEFAULT_P300_DATA_CONTRACT,
    assert_p300_input_contract,
    assert_p300_source_provenance,
)
from data.epochs import load_epoch_dataset, read_epoch_cache_attestation  # noqa: E402
from models.n2p3net import (  # noqa: E402
    DEFAULT_N2P3_ARCHITECTURE,
    POOLING_MODES,
    N2P3ArchitectureConfig,
    scale_architecture_preserving_spans,
)
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


def _parse_odd_kernel_sizes(value: str) -> tuple[int, ...]:
    try:
        kernels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        N2P3ArchitectureConfig(mst_kernel_sizes=kernels)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return kernels


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
    deep_defaults = DeepConfig()
    architecture_defaults = DEFAULT_N2P3_ARCHITECTURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--sample-rate-hz", type=float, choices=(128.0, 256.0), default=128.0)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_DEEP_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=deep_defaults.batch_size)
    parser.add_argument("--lr", type=float, default=deep_defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=deep_defaults.weight_decay)
    parser.add_argument("--pos-weight", type=float, default=deep_defaults.pos_weight)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=deep_defaults.early_stop_min_delta,
    )
    parser.add_argument(
        "--standardize-input",
        action=argparse.BooleanOptionalAction,
        default=deep_defaults.standardize_input,
    )
    parser.add_argument(
        "--max-update-batch-size",
        type=int,
        default=deep_defaults.max_update_batch_size,
    )
    parser.add_argument(
        "--batch-memory-fraction",
        type=float,
        default=deep_defaults.batch_memory_fraction,
    )
    parser.add_argument(
        "--preload-memory-fraction",
        type=float,
        default=deep_defaults.preload_memory_fraction,
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp32"),
        default=deep_defaults.precision,
        help="accelerator precision policy; auto selects BF16 when the backend supports it",
    )
    parser.add_argument(
        "--fused-adam",
        action=argparse.BooleanOptionalAction,
        default=deep_defaults.fused_adam,
        help="use CUDA fused Adam; enabled by default with portable non-CUDA fallback",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="none" if deep_defaults.compile_mode is None else deep_defaults.compile_mode,
        help="torch.compile mode; reduce-overhead is the CUDA default",
    )
    parser.add_argument(
        "--shuffle-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=deep_defaults.shuffle_each_epoch,
        help="optional one-shot device shuffle per epoch; off by default",
    )
    parser.add_argument("--validation-group-fraction", type=float, default=0.1)
    parser.add_argument(
        "--fold-jobs",
        type=int,
        default=4,
        help="concurrent fold workers; default 4 keeps the GPU fed through fold-boundary gaps",
    )
    parser.add_argument("--fold-backend", choices=("auto", "process", "thread"), default="auto")
    parser.add_argument(
        "--gpu-fold-jobs",
        type=int,
        default=None,
        help="maximum concurrent GPU fold processes; default adapts to accelerator memory",
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
        default="ms_flatten",
        help=(
            "N2P3-Net head: promoted ms_flatten, prior-free unfold candidates, "
            "LMBC rejected hypothesis, or global_average negative control."
        ),
    )
    parser.add_argument(
        "--n2p3net-temporal-filters",
        type=int,
        default=architecture_defaults.temporal_filters,
    )
    parser.add_argument(
        "--n2p3net-temporal-kernel-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n2p3net-spatial-depth-multiplier",
        type=int,
        default=architecture_defaults.spatial_depth_multiplier,
    )
    parser.add_argument(
        "--n2p3net-st-pool-size",
        type=int,
        default=architecture_defaults.st_pool_size,
    )
    parser.add_argument(
        "--n2p3net-mst-kernel-sizes",
        type=_parse_odd_kernel_sizes,
        default=None,
    )
    parser.add_argument(
        "--n2p3net-mst-features-per-scale",
        type=int,
        default=architecture_defaults.mst_features_per_scale,
    )
    parser.add_argument(
        "--n2p3net-mst-pool-size",
        type=int,
        default=architecture_defaults.mst_pool_size,
    )
    parser.add_argument(
        "--n2p3net-dropout",
        type=float,
        default=architecture_defaults.dropout,
    )
    parser.add_argument(
        "--n2p3net-spatial-max-norm",
        type=float,
        default=architecture_defaults.spatial_max_norm,
    )
    parser.add_argument(
        "--n2p3net-interaction-rank",
        type=int,
        default=architecture_defaults.interaction_rank,
    )
    parser.add_argument(
        "--n2p3net-mlp-hidden-features",
        type=int,
        default=architecture_defaults.mlp_hidden_features,
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
    architecture_defaults = scale_architecture_preserving_spans(
        DEFAULT_N2P3_ARCHITECTURE,
        source_sample_rate_hz=DEFAULT_P300_DATA_CONTRACT.sample_rate_hz,
        target_sample_rate_hz=args.sample_rate_hz,
    )
    deep_config_overrides = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pos_weight": args.pos_weight,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "standardize_input": args.standardize_input,
        "precision": args.precision,
        "fused_adam": args.fused_adam,
        "compile_mode": None if args.compile_mode == "none" else args.compile_mode,
        "shuffle_each_epoch": args.shuffle_each_epoch,
        "max_update_batch_size": args.max_update_batch_size,
        "batch_memory_fraction": args.batch_memory_fraction,
        "preload_memory_fraction": args.preload_memory_fraction,
    }
    try:
        DeepConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            val_group_frac=args.validation_group_fraction,
            **deep_config_overrides,
        )
        n2p3net_architecture = N2P3ArchitectureConfig(
            temporal_filters=args.n2p3net_temporal_filters,
            temporal_kernel_size=(
                architecture_defaults.temporal_kernel_size
                if args.n2p3net_temporal_kernel_size is None
                else args.n2p3net_temporal_kernel_size
            ),
            spatial_depth_multiplier=args.n2p3net_spatial_depth_multiplier,
            st_pool_size=args.n2p3net_st_pool_size,
            mst_kernel_sizes=(
                architecture_defaults.mst_kernel_sizes
                if args.n2p3net_mst_kernel_sizes is None
                else args.n2p3net_mst_kernel_sizes
            ),
            mst_features_per_scale=args.n2p3net_mst_features_per_scale,
            mst_pool_size=args.n2p3net_mst_pool_size,
            dropout=args.n2p3net_dropout,
            spatial_max_norm=args.n2p3net_spatial_max_norm,
            interaction_rank=args.n2p3net_interaction_rank,
            mlp_hidden_features=args.n2p3net_mlp_hidden_features,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
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
    dataset_record = dataset.record(validate=False)
    cache_sha256 = str(read_epoch_cache_attestation(args.dataset_cache)["sha256"])
    expected_contract = replace(
        DEFAULT_P300_DATA_CONTRACT,
        sample_rate_hz=args.sample_rate_hz,
    )
    assert_p300_input_contract(dataset.preprocessing, expected_contract)
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
    artifact_qc_workers = resolve_artifact_qc_workers(
        len(folds),
        artifact_qc_jobs=args.artifact_qc_jobs,
        cpu_threads=args.cpu_threads,
    )
    fitted_artifact_models, artifact_qc_sidecar = resolve_fold_local_artifact_models(
        X,
        subject_ids,
        folds,
        cache_path=args.dataset_cache,
        cache_sha256=cache_sha256,
        trial_channel_mask=trial_channel_mask,
        qc_features=qc_features,
        artifact_policy=artifact_policy,
        artifact_qc_jobs=args.artifact_qc_jobs,
        cpu_threads=args.cpu_threads,
    )
    artifact_qc_seconds = float(artifact_qc_sidecar["fit_seconds"])

    for model_name in models:
        model = build_binary_model(
            model_name,
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            validation_group_fraction=args.validation_group_fraction,
            deep_config_overrides=deep_config_overrides,
            device=device,
            n2p3net_pooling_mode=args.n2p3net_pooling,
            n2p3net_architecture=n2p3net_architecture,
        )
        output_dir = Path(args.run_dir) / run_name / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        model.configure_epoch_progress(output_dir / "epochs")
        progress_file = (output_dir / "progress.jsonl").open("w", encoding="utf-8")
        manifest = {
            "type": "manifest",
            "run_name": run_name,
            "model": describe_binary_model(model_name, model),
            "dataset": dataset_record,
            "fold_protocol": "partial_loso" if args.fold_offset or args.max_folds else "loso",
            "selection_mode": "candidate_selection" if candidate_selection is not None else "binary_oddball",
            "total_folds": args.fold_offset + len(folds),
            "batch_total_folds": len(folds),
            "batch_fold_offset": args.fold_offset,
            "epoch_progress": "epochs/fold_<fold>.jsonl",
            "epoch_index_base": 0,
            "trainer_kwargs": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "early_stop_patience": args.early_stop_patience,
            },
            "artifact_quality_policy": {"method": "fold_local_ptp_cv", **artifact_policy.__dict__},
            "shared_artifact_qc": {
                "reused_across_models": True,
                "workers": artifact_qc_workers,
                "fit_seconds": artifact_qc_seconds,
                "sidecar": artifact_qc_sidecar,
            },
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
        progress_file.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        progress_file.flush()
        folds_completed = 0

        def on_fold_end(fold_id: int, fold_result: object, *, _file=progress_file) -> None:
            nonlocal folds_completed
            folds_completed += 1
            hit_rate = getattr(fold_result, "hit_rate", None)
            line = {
                "type": "fold",
                "fold": fold_id,
                "n_folds_done": folds_completed,
                "fold_bacc": float(fold_result.balanced_acc),
                "fold_auc": float(fold_result.auc),
                "hit": None if hit_rate is None else float(hit_rate),
                "epochs_ran": int(fold_result.epochs_ran),
                "train_losses": [
                    round(float(value), 6) for value in fold_result.train_losses
                ][-12:],
                "val_losses": [
                    round(float(value), 6) for value in fold_result.val_losses
                ][-12:],
                "best_epoch": fold_result.best_epoch,
                "fit_sec": fold_result.fit_sec,
                "fit_peak_allocated_mb": fold_result.fit_peak_allocated_mb,
                "fit_peak_reserved_mb": fold_result.fit_peak_reserved_mb,
                "artifact_quality": fold_result.artifact_quality,
                "ts": datetime.now(UTC).isoformat(),
            }
            _file.write(json.dumps(line, ensure_ascii=False) + "\n")
            _file.flush()

        started = time.perf_counter()
        try:
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
                "fitted_artifact_models": fitted_artifact_models,
                "on_fold_end": on_fold_end,
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
        except BaseException as exc:
            progress_file.write(
                json.dumps(
                    {
                        "type": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "ts": datetime.now(UTC).isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            progress_file.flush()
            raise
        else:
            progress_file.write(
                json.dumps({"type": "done", "ts": datetime.now(UTC).isoformat()}) + "\n"
            )
            progress_file.flush()
        finally:
            progress_file.close()
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
