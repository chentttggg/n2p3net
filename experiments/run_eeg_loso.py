"""Run binary P300 LOSO from any universal EpochDataset cache."""

from __future__ import annotations

import argparse
import json
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

from baselines.evaluate import evaluate_binary, loso_folds  # noqa: E402
from baselines.evidence_protocol import row_acquisition_indices  # noqa: E402
from data.epochs import load_epoch_dataset  # noqa: E402
from models.heads import Z2_AUX_POOLS  # noqa: E402
from train.device import get_device  # noqa: E402
from train.factory import (  # noqa: E402
    BINARY_MODEL_NAMES,
    build_binary_model,
    describe_binary_model,
)
from train.recipe import (  # noqa: E402
    NEURAL_RIDE_V12,
    NEURAL_RIDE_V12_Z2_AUX_REPLACE_RESEARCH,
    NEURAL_RIDE_V12_Z2_AUX_RESEARCH,
)

RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


def _safe_auto_run_component(value: object) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return component or "dataset"


def _validate_run_name(value: str) -> str:
    normalized = value.replace("\\", "/")
    if not RUN_NAME.fullmatch(normalized) or any(
        part in {"", ".", ".."} for part in normalized.split("/")
    ):
        raise ValueError(
            "run name must contain safe relative path components using letters, digits, ._-"
        )
    return normalized


def _load_frozen_prior(path: str | None) -> dict | None:
    if path is None:
        return None
    prior = json.loads(Path(path).read_text(encoding="utf-8"))
    if prior.get("calibration_scope") != "independent_development":
        raise ValueError(
            "A frozen ERP prior must declare calibration_scope=independent_development."
        )
    return prior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument(
        "--models",
        default="n2p3net",
        help=f"comma-separated registered binary models: {','.join(BINARY_MODEL_NAMES)}",
    )
    parser.add_argument(
        "--z2-aux-head",
        choices=("off", "add", "replace"),
        default="off",
        help=(
            "research-only full-Z2 auxiliary trial head for N2P3-Net: "
            "off=PCW-only production default; add=PCW+aux; replace=aux-only"
        ),
    )
    parser.add_argument(
        "--z2-aux-pool",
        choices=Z2_AUX_POOLS,
        default="attention",
        help="pooling for --z2-aux-head != off",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--fold-jobs",
        type=int,
        default=1,
        help="number of folds trained concurrently; use 1 for serial GPU execution",
    )
    parser.add_argument(
        "--fold-offset",
        type=int,
        default=0,
        help="skip this many initial LOSO folds while keeping dashboard fold numbering",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="train at most this many folds after --fold-offset",
    )
    parser.add_argument("--subjects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-subject-fraction", type=float, default=0.1)
    parser.add_argument(
        "--early-stop-patience", type=int, default=NEURAL_RIDE_V12.early_stop_patience
    )
    parser.add_argument("--erp-calibration", choices=("fold", "fixed"), default="fold")
    parser.add_argument("--frozen-erp-prior", default=None)
    parser.add_argument("--lr", type=float, default=NEURAL_RIDE_V12.lr)
    parser.add_argument("--lambda2", type=float, default=NEURAL_RIDE_V12.lambda2)
    parser.add_argument("--lambda3", type=float, default=NEURAL_RIDE_V12.lambda3)
    parser.add_argument("--lambda-pcw", type=float, default=NEURAL_RIDE_V12.lambda_pcw)
    parser.add_argument("--lambda-recon", type=float, default=NEURAL_RIDE_V12.lambda_recon)
    parser.add_argument(
        "--lambda-morphology-l0",
        type=float,
        default=NEURAL_RIDE_V12.lambda_morphology_l0,
    )
    parser.add_argument(
        "--variance-warmup-epochs",
        type=int,
        default=NEURAL_RIDE_V12.variance_warmup_epochs,
    )
    parser.add_argument(
        "--variance-ramp-epochs",
        type=int,
        default=NEURAL_RIDE_V12.variance_ramp_epochs,
    )
    parser.add_argument(
        "--recon-bootstrap-samples",
        type=int,
        default=NEURAL_RIDE_V12.recon_bootstrap_samples,
    )
    parser.add_argument(
        "--recon-split-half-repeats",
        type=int,
        default=NEURAL_RIDE_V12.recon_split_half_repeats,
    )
    parser.add_argument("--lambda-amp", type=float, default=NEURAL_RIDE_V12.lambda_amp)
    parser.add_argument(
        "--amplitude-channel",
        default=None,
        help="physical channel for lambda-amp; required only when lambda-amp > 0",
    )
    parser.add_argument(
        "--fold-backend",
        choices=("auto", "process", "thread"),
        default="auto",
        help="parallel fold isolation backend; auto uses spawned processes on Linux",
    )
    parser.add_argument("--run-dir", default="experiments/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "xpu", "cpu"), default="auto")
    args = parser.parse_args()
    if args.fold_jobs < 1:
        parser.error("--fold-jobs must be positive")
    if args.fold_offset < 0:
        parser.error("--fold-offset must be non-negative")
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("--max-folds must be positive")
    if args.subjects is not None and args.subjects < 2:
        parser.error("--subjects must be at least two for LOSO")
    if args.lambda_amp > 0.0 and not args.amplitude_channel:
        parser.error("--amplitude-channel is required when --lambda-amp > 0")

    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True)
    dataset_acquisition_indices = row_acquisition_indices(dataset.event_timeline)
    dataset_trial_channel_mask = (
        np.asarray(dataset.trial_channel_mask, dtype=bool)
        if dataset.trial_channel_mask is not None
        else np.broadcast_to(
            np.asarray(dataset.channel_mask, dtype=bool),
            (dataset.n_epochs, dataset.n_channels),
        ).copy()
    )
    if args.subjects is not None:
        selected_subjects = np.unique(dataset.subject_ids)[: args.subjects]
        keep = np.isin(dataset.subject_ids, selected_subjects)
        X = dataset.X[keep]
        y = dataset.y[keep]
        subject_ids = dataset.subject_ids[keep]
        acquisition_indices = dataset_acquisition_indices[keep]
        trial_channel_mask = dataset_trial_channel_mask[keep]
    else:
        X, y, subject_ids, acquisition_indices, trial_channel_mask = (
            dataset.X,
            dataset.y,
            dataset.subject_ids,
            dataset_acquisition_indices,
            dataset_trial_channel_mask,
        )
    folds = loso_folds(subject_ids)
    if len(folds) < 2:
        raise ValueError("LOSO requires at least two subjects.")
    folds = folds[args.fold_offset :]
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        raise ValueError("No LOSO folds remain after --fold-offset/--max-folds.")
    display_fold_offset = args.fold_offset

    if args.device == "auto":
        resolved_device = get_device()
    else:
        resolved_device = torch.device(args.device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        if resolved_device.type == "xpu" and not (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        ):
            raise RuntimeError("--device xpu was requested, but XPU is unavailable.")

    model_names = tuple(name.strip().lower() for name in args.models.split(",") if name.strip())
    if not model_names:
        parser.error("--models must contain at least one registered model")
    if len(set(model_names)) != len(model_names):
        parser.error("--models must not contain duplicate model names")
    unknown = set(model_names) - set(BINARY_MODEL_NAMES)
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")
    frozen_prior = _load_frozen_prior(args.frozen_erp_prior)
    if args.erp_calibration == "fold" and frozen_prior is not None:
        parser.error("--erp-calibration fold and --frozen-erp-prior are mutually exclusive")

    if args.z2_aux_head == "off":
        n2p3net_recipe = NEURAL_RIDE_V12
    elif args.z2_aux_head == "replace":
        n2p3net_recipe = NEURAL_RIDE_V12_Z2_AUX_REPLACE_RESEARCH
    else:
        n2p3net_recipe = NEURAL_RIDE_V12_Z2_AUX_RESEARCH

    trainer_overrides = {
        "lr": args.lr,
        "lambda2": args.lambda2,
        "lambda3": args.lambda3,
        "lambda_pcw": args.lambda_pcw,
        "lambda_recon": args.lambda_recon,
        "lambda_morphology_l0": args.lambda_morphology_l0,
        "variance_warmup_epochs": args.variance_warmup_epochs,
        "variance_ramp_epochs": args.variance_ramp_epochs,
        "recon_bootstrap_samples": args.recon_bootstrap_samples,
        "recon_split_half_repeats": args.recon_split_half_repeats,
        "lambda_amp": args.lambda_amp,
        "amplitude_channel": args.amplitude_channel,
        "early_stop_patience": args.early_stop_patience,
    }
    protocol_name = (
        n2p3net_recipe.name
        if "n2p3net" in model_names
        else "binary_" + "_".join(model_names)
    )
    run_name = (
        _validate_run_name(args.run_name)
        if args.run_name is not None
        else (
            f"{_safe_auto_run_component(dataset.name)}_{protocol_name}_"
            + datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
        )
    )

    for model_name in model_names:
        adapter = build_binary_model(
            model_name,
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            validation_subject_fraction=args.validation_subject_fraction,
            fold_erp_calibration=args.erp_calibration == "fold",
            frozen_erp_prior=frozen_prior,
            trainer_overrides=trainer_overrides,
            model_overrides={
                "component_decoder": (
                    args.lambda_recon > 0.0 or args.lambda_morphology_l0 > 0.0
                ),
                "use_z2_aux_head": n2p3net_recipe.use_z2_aux_head,
                "z2_aux_head_mode": n2p3net_recipe.z2_aux_head_mode,
                "z2_aux_pool": args.z2_aux_pool,
            },
            deep_config_overrides={
                "lr": args.lr,
                "early_stop_patience": args.early_stop_patience,
            },
            device=resolved_device,
            recipe=n2p3net_recipe,
        )

        model_id = f"{model_name}_{dataset.n_channels}ch"
        output_dir = Path(args.run_dir) / run_name / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch_progress_dir = output_dir / "epochs"
        epoch_progress_dir.mkdir(parents=True, exist_ok=True)
        adapter.configure_epoch_progress(epoch_progress_dir)
        progress_file = (output_dir / "progress.jsonl").open("w", encoding="utf-8")
        manifest = {
            "type": "manifest",
            "run_name": run_name,
            "model": model_id,
            "dataset": dataset.record(),
            "selection": {
                "subjects_requested": args.subjects,
                "subjects_used": int(len(np.unique(subject_ids))),
                "fold_offset": display_fold_offset,
                "max_folds": args.max_folds,
            },
            "recipe": describe_binary_model(model_name, adapter, n2p3net_recipe),
            "total_folds": display_fold_offset + len(folds),
            "batch_total_folds": len(folds),
            "batch_fold_offset": display_fold_offset,
            "epoch_progress": "epochs/fold_<fold>.jsonl",
            "epoch_index_base": 0,
            "environment": {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
                "device": str(resolved_device),
            },
            "started_utc": datetime.now(UTC).isoformat(),
        }
        progress_file.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        progress_file.flush()
        folds_completed = 0

        def on_fold(
            fold_index: int,
            fold_result,
            *,
            _adapter=adapter,
            _file=progress_file,
        ) -> None:
            nonlocal folds_completed
            folds_completed += 1
            history = fold_result
            line = {
                "type": "fold",
                "fold": display_fold_offset + fold_index,
                "batch_fold": fold_index,
                "n_folds_done": folds_completed,
                "fold_bacc": float(fold_result.balanced_acc),
                "fold_auc": float(fold_result.auc),
                "epochs_ran": int(getattr(history, "epochs_ran", 0)),
                "train_losses": [round(float(v), 6) for v in history.train_losses][-12:],
                "val_losses": [round(float(v), 6) for v in history.val_losses][-12:],
                "val_objective_losses": [
                    round(float(v), 6) for v in history.val_objective_losses
                ][-12:],
                "task_val_aucs": [
                    None if v is None else round(float(v), 6)
                    for v in history.task_val_aucs
                ][-12:],
                "final_task_val_auc": history.final_task_val_auc,
                "phases": history.phases[-12:],
                "best_epoch": history.best_epoch,
                "best_task_epoch": history.best_task_epoch,
                "best_task_val_loss": history.best_task_val_loss,
                "task_patience_exhausted": history.task_patience_exhausted,
                "fit_sec": history.fit_sec,
                "fit_peak_memory_mb": history.fit_peak_memory_mb,
                "ts": datetime.now(UTC).isoformat(),
            }
            _file.write(json.dumps(line, ensure_ascii=False) + "\n")
            _file.flush()

        started = time.perf_counter()
        try:
            summary = evaluate_binary(
                adapter,
                X,
                y,
                subject_ids,
                folds,
                acquisition_indices=acquisition_indices,
                fold_protocol=(
                    "partial_loso"
                    if args.fold_offset or args.max_folds is not None
                    else "loso"
                ),
                n_jobs=args.fold_jobs,
                parallel_backend=args.fold_backend,
                fold_id_offset=display_fold_offset,
                trial_channel_mask=trial_channel_mask,
                on_fold_end=on_fold,
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
        wall_seconds = time.perf_counter() - started
        record = {
            **manifest,
            "balanced_acc_mean": summary.balanced_acc_mean,
            "balanced_acc_std": summary.balanced_acc_std,
            "auc_mean": summary.auc_mean,
            "transductive_balanced_acc_mean": summary.transductive_balanced_acc_mean,
            "per_fold": [fold.__dict__ for fold in summary.per_fold],
            "wall_seconds": wall_seconds,
            "finished_utc": datetime.now(UTC).isoformat(),
        }
        (output_dir / "record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[{model_id}] subjects={len(np.unique(subject_ids))} "
            f"bacc={summary.balanced_acc_mean:.4f} auc={summary.auc_mean:.4f} "
            f"wall={wall_seconds:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
