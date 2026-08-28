"""Run the fixed A-E receptive-field mechanism comparison on inner validation only."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    resolve_fold_local_artifact_models,
)
from data.artifact import FoldLocalArtifactPolicy  # noqa: E402
from data.contract import (  # noqa: E402
    DEFAULT_P300_DATA_CONTRACT,
    assert_p300_input_contract,
    assert_p300_source_provenance,
)
from data.epochs import load_epoch_dataset, read_epoch_cache_attestation  # noqa: E402
from models.n2p3net import N2P3ArchitectureConfig  # noqa: E402
from train.device import get_device  # noqa: E402
from train.factory import build_binary_model, describe_binary_model  # noqa: E402

ARM_ORDER = ("A", "B", "C", "D", "E")
ARM_ROLES = {
    "A": "broad_dense_reference",
    "B": "k35_bridge",
    "C": "local_same_taps",
    "D": "broad_same_taps",
    "E": "broad_redistributed_same_parameters",
}
PLANNED_CONTRASTS = (("C", "D"), ("D", "A"), ("E", "A"))
EXPECTED_GEOMETRY = {
    "A": (65, 1, (5, 17), (1, 1)),
    "B": (35, 1, (5, 17), (1, 1)),
    "C": (33, 1, (5, 17), (1, 1)),
    "D": (33, 2, (5, 17), (1, 1)),
    "E": (33, 1, (13, 25), (1, 1)),
}
MANIFEST_SCHEMA = "n2p3net_rf_mechanism_comparison/1"
SEED_RECORD_SCHEMA = "n2p3net_rf_mechanism_seed/1"
ARM_RECORD_SCHEMA = "n2p3net_rf_mechanism_arm/1"
SCREENING_SCHEMA = "n2p3net_rf_mechanism_screening/1"
RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
ARM_SELECTION_METRIC = "mean_inner_best_task_val_loss"
SCREENING_SELECTION_METRIC = "paired_mean_inner_best_task_val_loss"


def _parse_unique_nonnegative_ints(value: str, *, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise argparse.ArgumentTypeError(f"{label} must be unique non-negative integers")
    return values


def _parse_fold_indices(value: str) -> tuple[int, ...]:
    return _parse_unique_nonnegative_ints(value, label="fold indices")


def _parse_seeds(value: str) -> tuple[int, ...]:
    return _parse_unique_nonnegative_ints(value, label="seeds")


def _validate_run_name(value: str) -> str:
    normalized = value.replace("\\", "/")
    if not RUN_NAME.fullmatch(normalized) or any(
        part in {"", ".", ".."} for part in normalized.split("/")
    ):
        raise argparse.ArgumentTypeError(
            "run name must contain safe relative components using letters, digits, ._-"
        )
    return normalized


def _subject_sort_key(subject: str) -> tuple[int, int | str]:
    stripped = subject.strip()
    return (0, int(stripped)) if stripped.isdigit() else (1, stripped)


def _anchor_subjects_for_folds(
    subject_ids: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[str]:
    subjects = np.asarray(subject_ids).astype(str)
    anchors: list[str] = []
    for index, (_, test) in enumerate(folds):
        held_out = np.unique(subjects[np.asarray(test, dtype=bool)])
        if len(held_out) != 1:
            raise ValueError(f"Anchor fold {index} must hold out exactly one subject.")
        anchors.append(str(held_out[0]))
    return anchors


def _validate_rf_architectures(
    mapping: object,
) -> dict[str, N2P3ArchitectureConfig]:
    if not isinstance(mapping, Mapping):
        raise RuntimeError("RF_MECHANISM_ARCHITECTURES must be a mapping keyed by A-E.")
    if set(mapping) != set(ARM_ORDER):
        raise RuntimeError(
            "RF_MECHANISM_ARCHITECTURES must contain exactly the fixed arms A, B, C, D, E."
        )
    architectures: dict[str, N2P3ArchitectureConfig] = {}
    for arm in ARM_ORDER:
        architecture = mapping[arm]
        if not isinstance(architecture, N2P3ArchitectureConfig):
            raise RuntimeError(f"RF arm {arm} must be an N2P3ArchitectureConfig.")
        if not hasattr(architecture, "st_temporal_dilation") or not hasattr(
            architecture, "mst_dilations"
        ):
            raise RuntimeError(
                "RF_MECHANISM_ARCHITECTURES requires N2P3ArchitectureConfig fields "
                "st_temporal_dilation and mst_dilations."
            )
        observed = (
            int(architecture.temporal_kernel_size),
            int(architecture.st_temporal_dilation),
            tuple(int(value) for value in architecture.mst_kernel_sizes),
            tuple(int(value) for value in architecture.mst_dilations),
        )
        if observed != EXPECTED_GEOMETRY[arm]:
            raise RuntimeError(
                f"RF arm {arm} geometry drifted: expected {EXPECTED_GEOMETRY[arm]}, "
                f"got {observed}."
            )
        architectures[arm] = architecture
    return architectures


def _load_rf_architectures() -> dict[str, N2P3ArchitectureConfig]:
    module = importlib.import_module("models.n2p3net")
    mapping = getattr(module, "RF_MECHANISM_ARCHITECTURES", None)
    if mapping is None:
        raise RuntimeError(
            "models.n2p3net.RF_MECHANISM_ARCHITECTURES is unavailable. "
            "Use a source revision that provides the fixed A-E RF mechanism registry."
        )
    return _validate_rf_architectures(mapping)


def _resolve_source_commit(
    *,
    root: Path = ROOT,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    environment = os.environ if environ is None else environ
    supplied = str(environment.get("SOURCE_COMMIT", "")).strip()
    if supplied:
        return supplied, "SOURCE_COMMIT"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Cannot resolve the source commit from Git; set SOURCE_COMMIT explicitly."
        ) from exc
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("Git returned an empty source commit.")
    return commit, "git"


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return get_device()
    try:
        device = torch.device(choice)
    except RuntimeError as exc:
        raise ValueError(f"Invalid --device value {choice!r}.") from exc
    if device.type not in {"cpu", "cuda", "xpu"}:
        raise ValueError("--device must be auto, cpu, cuda[:INDEX], or xpu[:INDEX].")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU was requested but is unavailable.")
    return device


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _fingerprint(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _model_for_arm(
    architecture: N2P3ArchitectureConfig,
    dataset: object,
    *,
    args: argparse.Namespace,
    defaults: DeepConfig,
    seed: int,
    device: torch.device,
):
    overrides = {
        "lr": defaults.lr,
        "weight_decay": defaults.weight_decay,
        "pos_weight": defaults.pos_weight,
        "early_stop_patience": args.early_stop_patience,
        "precision": args.precision,
        "fused_adam": args.fused_adam,
        "compile_mode": None if args.compile_mode == "none" else args.compile_mode,
        "shuffle_each_epoch": args.shuffle_each_epoch,
        "max_update_batch_size": args.batch_size,
    }
    return build_binary_model(
        "n2p3net_full_unfold",
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
        validation_group_fraction=0.1,
        deep_config_overrides=overrides,
        device=device,
        n2p3net_architecture=architecture,
    )


def _arm_descriptor(
    arm: str,
    architecture: N2P3ArchitectureConfig,
    model: object,
) -> dict[str, object]:
    described = describe_binary_model("n2p3net_full_unfold", model)
    architecture_record = dict(described["architecture"])
    return {
        "arm": arm,
        "role": ARM_ROLES[arm],
        "configuration": asdict(architecture),
        "parameter_count": int(described["parameter_count"]),
        "receptive_field": {
            "branch_samples": architecture_record.get("mst_total_receptive_field_samples"),
            "branch_span_ms": architecture_record.get("mst_total_receptive_span_ms"),
        },
        "executed_architecture": architecture_record,
    }


def _inner_fold_record(
    result: object,
    *,
    fold_index: int,
    anchor_subject: str,
) -> dict[str, object]:
    return {
        "anchor_fold_index": int(fold_index),
        "anchor_subject": str(anchor_subject),
        "best_task_val_loss": float(result.best_task_val_loss),
        "final_task_val_auc": float(result.final_task_val_auc),
        "epochs_ran": int(result.epochs_ran),
        "batch_size": int(result.batch_size),
        "fused_adam_requested": bool(result.fused_adam_requested),
        "compile_mode_requested": result.compile_mode_requested,
        "fused_adam": bool(result.fused_adam),
        "compile_mode": result.compile_mode,
        "compile_scope": result.compile_scope,
        "optimizer_fallback_reason": result.optimizer_fallback_reason,
        "oom_retries": int(result.oom_retries),
    }


def _seed_record(
    *,
    arm: str,
    seed: int,
    source_commit: str,
    cache_sha256: str,
    fold_indices: Sequence[int],
    anchor_subjects: Sequence[str],
    model_record: Mapping[str, object],
    fold_results: Sequence[object],
    wall_seconds: float,
    run_fingerprint: str,
) -> dict[str, object]:
    if len(fold_results) != len(fold_indices) or len(fold_results) != len(anchor_subjects):
        raise ValueError("Seed fold results must align with fold indices and anchor subjects.")
    per_fold = [
        _inner_fold_record(result, fold_index=fold_index, anchor_subject=subject)
        for fold_index, subject, result in zip(
            fold_indices, anchor_subjects, fold_results, strict=True
        )
    ]
    losses = np.asarray([record["best_task_val_loss"] for record in per_fold], dtype=float)
    aucs = np.asarray([record["final_task_val_auc"] for record in per_fold], dtype=float)
    if not np.isfinite(losses).all() or not np.isfinite(aucs).all():
        raise RuntimeError(f"RF arm {arm} seed {seed} produced non-finite inner metrics.")
    return {
        "schema": SEED_RECORD_SCHEMA,
        "arm": arm,
        "role": ARM_ROLES[arm],
        "seed": int(seed),
        "source_commit": source_commit,
        "cache_sha256": cache_sha256,
        "run_fingerprint": run_fingerprint,
        "fold_indices": [int(value) for value in fold_indices],
        "anchor_subjects": [str(value) for value in anchor_subjects],
        "outer_test_metrics_persisted": False,
        "outer_test_metrics_used_for_selection": False,
        "selection_metric": ARM_SELECTION_METRIC,
        "model": dict(model_record),
        "mean_inner_best_task_val_loss": float(losses.mean()),
        "mean_inner_final_task_val_auc": float(aucs.mean()),
        "per_fold": per_fold,
        "wall_seconds": float(wall_seconds),
        "finished_utc": datetime.now(UTC).isoformat(),
    }


def _validate_resumed_seed(
    record: Mapping[str, object],
    *,
    arm: str,
    seed: int,
    source_commit: str,
    cache_sha256: str,
    fold_indices: Sequence[int],
    run_fingerprint: str,
) -> None:
    expected = {
        "schema": SEED_RECORD_SCHEMA,
        "arm": arm,
        "seed": int(seed),
        "source_commit": source_commit,
        "cache_sha256": cache_sha256,
        "run_fingerprint": run_fingerprint,
        "fold_indices": [int(value) for value in fold_indices],
        "outer_test_metrics_persisted": False,
    }
    mismatched = [key for key, value in expected.items() if record.get(key) != value]
    if mismatched:
        raise RuntimeError(f"Existing seed record cannot be resumed; mismatched {mismatched}.")


def _aggregate_arm_record(
    arm: str,
    descriptor: Mapping[str, object],
    seed_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not seed_records:
        raise ValueError("An RF arm record requires at least one seed record.")
    fold_records = [
        dict(fold)
        for seed_record in seed_records
        for fold in seed_record["per_fold"]  # type: ignore[index]
    ]
    losses = np.asarray([fold["best_task_val_loss"] for fold in fold_records], dtype=float)
    aucs = np.asarray([fold["final_task_val_auc"] for fold in fold_records], dtype=float)
    return {
        "schema": ARM_RECORD_SCHEMA,
        "arm": arm,
        "role": ARM_ROLES[arm],
        "outer_test_metrics_persisted": False,
        "outer_test_metrics_used_for_selection": False,
        "selection_metric": ARM_SELECTION_METRIC,
        "architecture": dict(descriptor),
        "seeds": [int(record["seed"]) for record in seed_records],
        "mean_inner_best_task_val_loss": float(losses.mean()),
        "mean_inner_final_task_val_auc": float(aucs.mean()),
        "per_seed": [dict(record) for record in seed_records],
        "finished_utc": datetime.now(UTC).isoformat(),
    }


def _paired_contrast(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> dict[str, object]:
    def indexed(record: Mapping[str, object]) -> dict[tuple[int, int], Mapping[str, object]]:
        output: dict[tuple[int, int], Mapping[str, object]] = {}
        for seed_record in record["per_seed"]:  # type: ignore[index]
            seed = int(seed_record["seed"])
            for fold in seed_record["per_fold"]:
                output[(seed, int(fold["anchor_fold_index"]))] = fold
        return output

    left_index = indexed(left)
    right_index = indexed(right)
    if set(left_index) != set(right_index):
        raise ValueError("Planned RF contrasts require identical seed/fold pairs.")
    keys = sorted(left_index)
    loss_delta = np.asarray(
        [
            left_index[key]["best_task_val_loss"] - right_index[key]["best_task_val_loss"]
            for key in keys
        ],
        dtype=float,
    )
    auc_delta = np.asarray(
        [
            left_index[key]["final_task_val_auc"] - right_index[key]["final_task_val_auc"]
            for key in keys
        ],
        dtype=float,
    )
    return {
        "contrast": f"{left['arm']}-{right['arm']}",
        "left": left["arm"],
        "right": right["arm"],
        "n_seed_fold_pairs": len(keys),
        "mean_best_task_val_loss_delta": float(loss_delta.mean()),
        "mean_final_task_val_auc_delta": float(auc_delta.mean()),
        "interpretation": "negative loss delta and positive AUC delta favor the left arm",
    }


def _screening_record(arm_records: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if set(arm_records) != set(ARM_ORDER):
        raise ValueError("Screening requires completed A-E arm records.")
    ranking = sorted(
        (
            {
                "arm": arm,
                "role": ARM_ROLES[arm],
                "mean_inner_best_task_val_loss": float(
                    arm_records[arm]["mean_inner_best_task_val_loss"]
                ),
                "mean_inner_final_task_val_auc": float(
                    arm_records[arm]["mean_inner_final_task_val_auc"]
                ),
            }
            for arm in ARM_ORDER
        ),
        key=lambda record: (
            record["mean_inner_best_task_val_loss"],
            -record["mean_inner_final_task_val_auc"],
            ARM_ORDER.index(str(record["arm"])),
        ),
    )
    contrasts = [
        _paired_contrast(arm_records[left], arm_records[right])
        for left, right in PLANNED_CONTRASTS
    ]
    return {
        "schema": SCREENING_SCHEMA,
        "evidence_scope": "inner_validation_only_development",
        "selection_metric": SCREENING_SELECTION_METRIC,
        "secondary_metric": "mean_inner_final_task_val_auc",
        "outer_test_metrics_persisted": False,
        "outer_test_metrics_used_for_selection": False,
        "bridge_arm": "B",
        "ranking": ranking,
        "planned_contrasts": contrasts,
        "finished_utc": datetime.now(UTC).isoformat(),
    }


def _build_parser(defaults: DeepConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--run-dir", default="experiments/runs")
    parser.add_argument(
        "--run-name",
        type=_validate_run_name,
        default="rf_mechanism_comparison",
    )
    parser.add_argument("--subjects", type=int, default=64)
    parser.add_argument("--fold-indices", type=_parse_fold_indices, default=None)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=(20260828, 20260829, 20260830),
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--early-stop-patience", type=int, default=defaults.early_stop_patience)
    parser.add_argument("--fold-jobs", type=int, default=4)
    parser.add_argument("--gpu-fold-jobs", type=int, default=None)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="total CPU budget; default uses the cgroup-aware available quota",
    )
    parser.add_argument(
        "--artifact-qc-jobs",
        type=int,
        default=None,
        help="QC process count; default adapts to folds and the effective CPU budget",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp32"),
        default=defaults.precision,
    )
    parser.add_argument(
        "--fused-adam",
        action=argparse.BooleanOptionalAction,
        default=defaults.fused_adam,
    )
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="none" if defaults.compile_mode is None else defaults.compile_mode,
    )
    parser.add_argument(
        "--shuffle-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=defaults.shuffle_each_epoch,
    )
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    defaults = DeepConfig()
    parser = _build_parser(defaults)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.early_stop_patience < 1:
        parser.error("epochs, batch size, and patience must be positive")
    if args.subjects is not None and args.subjects < 2:
        parser.error("--subjects must be at least two")
    if args.fold_jobs < 1:
        parser.error("--fold-jobs must be positive")
    if args.gpu_fold_jobs is not None and args.gpu_fold_jobs < 1:
        parser.error("--gpu-fold-jobs must be positive when set")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive when set")
    if args.artifact_qc_jobs is not None and args.artifact_qc_jobs < 1:
        parser.error("--artifact-qc-jobs must be positive when set")

    architectures = _load_rf_architectures()
    source_commit, source_commit_origin = _resolve_source_commit()
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    dataset_record = dataset.record(validate=False)
    cache_attestation = read_epoch_cache_attestation(args.dataset_cache)
    cache_sha256 = str(cache_attestation["sha256"])
    assert_p300_input_contract(dataset.preprocessing, DEFAULT_P300_DATA_CONTRACT)
    assert_p300_source_provenance(dataset)
    if dataset.qc_features is None:
        raise ValueError("RF mechanism comparison requires cached QC v2 features.")

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
            np.linspace(0, len(all_folds) - 1, num=min(16, len(all_folds)), dtype=int).tolist()
        )
    if max(fold_indices) >= len(all_folds):
        parser.error(f"fold index exceeds the available {len(all_folds)} LOSO folds")
    folds = [all_folds[index] for index in fold_indices]
    anchor_subjects = _anchor_subjects_for_folds(subject_ids, folds)
    artifact_policy = FoldLocalArtifactPolicy()
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
    device = _resolve_device(args.device)

    descriptors: dict[str, dict[str, object]] = {}
    for arm in ARM_ORDER:
        prototype = _model_for_arm(
            architectures[arm],
            dataset,
            args=args,
            defaults=defaults,
            seed=args.seeds[0],
            device=device,
        )
        descriptors[arm] = _arm_descriptor(arm, architectures[arm], prototype)

    training_record = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "early_stop_patience": args.early_stop_patience,
        "validation_group_fraction": 0.1,
        "precision": args.precision,
        "fused_adam": args.fused_adam,
        "compile_mode": args.compile_mode,
        "shuffle_each_epoch": args.shuffle_each_epoch,
        "fold_jobs": args.fold_jobs,
        "gpu_fold_jobs": args.gpu_fold_jobs,
        "cpu_threads": args.cpu_threads,
        "artifact_qc_jobs": args.artifact_qc_jobs,
        "device": str(device),
    }
    run_contract = {
        "schema": MANIFEST_SCHEMA,
        "source_commit": source_commit,
        "cache_sha256": cache_sha256,
        "selected_subjects": selected_subjects.tolist(),
        "fold_indices": list(fold_indices),
        "anchor_subjects": anchor_subjects,
        "seeds": list(args.seeds),
        "arms": descriptors,
        "training": training_record,
        "artifact_policy": asdict(artifact_policy),
    }
    run_fingerprint = _fingerprint(run_contract)
    output_root = Path(args.run_dir) / args.run_name
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "evidence_scope": "inner_validation_only_development",
        "selection_metric": SCREENING_SELECTION_METRIC,
        "secondary_metric": "mean_inner_final_task_val_auc",
        "outer_test_metrics_persisted": False,
        "outer_test_metrics_used_for_selection": False,
        "run_fingerprint": run_fingerprint,
        "source_commit": source_commit,
        "source_commit_origin": source_commit_origin,
        "cache_attestation": cache_attestation,
        "dataset": dataset_record,
        "selected_subject_count": len(selected_subjects),
        "selected_subjects": selected_subjects.tolist(),
        "selected_epoch_count": len(X),
        "fold_indices": list(fold_indices),
        "anchor_subjects": anchor_subjects,
        "seeds": list(args.seeds),
        "arm_order": list(ARM_ORDER),
        "arms": descriptors,
        "planned_contrasts": [f"{left}-{right}" for left, right in PLANNED_CONTRASTS],
        "bridge_arm": "B",
        "training": training_record,
        "artifact_qc_sidecar": artifact_qc_sidecar,
        "started_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(output_root / "manifest.json", manifest)

    arm_records: dict[str, dict[str, object]] = {}
    for arm in ARM_ORDER:
        arm_dir = output_root / arm
        seed_records: list[dict[str, object]] = []
        for seed in args.seeds:
            seed_dir = arm_dir / f"seed_{seed}"
            seed_path = seed_dir / "record.json"
            if seed_path.exists():
                record = _read_json(seed_path)
                _validate_resumed_seed(
                    record,
                    arm=arm,
                    seed=seed,
                    source_commit=source_commit,
                    cache_sha256=cache_sha256,
                    fold_indices=fold_indices,
                    run_fingerprint=run_fingerprint,
                )
                seed_records.append(record)
                continue
            model = _model_for_arm(
                architectures[arm],
                dataset,
                args=args,
                defaults=defaults,
                seed=seed,
                device=device,
            )
            model.configure_epoch_progress(seed_dir / "epochs")
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
            record = _seed_record(
                arm=arm,
                seed=seed,
                source_commit=source_commit,
                cache_sha256=cache_sha256,
                fold_indices=fold_indices,
                anchor_subjects=anchor_subjects,
                model_record=describe_binary_model("n2p3net_full_unfold", model),
                fold_results=summary.per_fold,
                wall_seconds=time.perf_counter() - started,
                run_fingerprint=run_fingerprint,
            )
            _write_json(seed_path, record)
            seed_records.append(record)
            print(
                f"[{arm} seed={seed}] val_loss={record['mean_inner_best_task_val_loss']:.6f} "
                f"val_auc={record['mean_inner_final_task_val_auc']:.6f}",
                flush=True,
            )
        arm_record = _aggregate_arm_record(arm, descriptors[arm], seed_records)
        _write_json(arm_dir / "record.json", arm_record)
        arm_records[arm] = arm_record

    screening = _screening_record(arm_records)
    screening.update(
        {
            "source_commit": source_commit,
            "cache_sha256": cache_sha256,
            "fold_indices": list(fold_indices),
            "seeds": list(args.seeds),
            "run_fingerprint": run_fingerprint,
        }
    )
    _write_json(output_root / "screening.json", screening)


if __name__ == "__main__":
    main()
