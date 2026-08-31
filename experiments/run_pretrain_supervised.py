"""Supervised source pretraining for the N2P3 trunk (Gate 3-S1).

Trains N2P3NetBaseline (the exact LOSO-floor training path) on every source
subject except the holdout block. The saved checkpoint carries the full model
state dict plus the auditable ``training_subject_keys`` ledger consumed by
``experiments/run_within_subject_transfer.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.calibration import fit_weighted_logit_temperature  # noqa: E402
from baselines.deep import DeepConfig  # noqa: E402
from baselines.n2p3net import N2P3NetBaseline  # noqa: E402
from data.contract import (  # noqa: E402
    DEFAULT_P300_DATA_CONTRACT,
    GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT,
    PAPER_GTN_CAUSAL_DATA_CONTRACT,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    assert_p300_input_contract,
)
from data.epochs import load_epoch_dataset, read_epoch_cache_attestation  # noqa: E402
from models.n2p3net import (  # noqa: E402
    DEFAULT_N2P3_ARCHITECTURE,
    DEFAULT_N2P3_POOLING_MODE,
    N2P3ArchitectureConfig,
)
from train.device import get_device  # noqa: E402
from train.runtime import GpuPerformanceScheduler  # noqa: E402

SOURCE_COHORT_CONTRACTS = {
    "default": DEFAULT_P300_DATA_CONTRACT,
    "p300_causal": SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    "gtn": GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT,
    "gtn_paper": PAPER_GTN_CAUSAL_DATA_CONTRACT,
}


def parse_subject_prefix_repeats(value: str) -> dict[str, int]:
    """Parse explicit subject-prefix exposure repeats from a CLI value."""

    if not value.strip():
        return {}
    repeats: dict[str, int] = {}
    for raw_item in value.split(","):
        item = raw_item.strip()
        prefix, separator, raw_repeat = item.partition("=")
        prefix = prefix.strip()
        raw_repeat = raw_repeat.strip()
        if not separator or not prefix or not raw_repeat:
            raise ValueError(
                "subject-prefix-repeat entries must use PREFIX=INTEGER, "
                "for example BI::=3,BNCI::=1."
            )
        if prefix in repeats:
            raise ValueError(f"duplicate subject-prefix-repeat prefix: {prefix!r}")
        try:
            repeat = int(raw_repeat)
        except ValueError as error:
            raise ValueError(
                f"subject-prefix-repeat for {prefix!r} must be an integer."
            ) from error
        if repeat < 1:
            raise ValueError(
                f"subject-prefix-repeat for {prefix!r} must be at least 1."
            )
        repeats[prefix] = repeat
    return repeats


def build_subject_prefix_exposure(
    subjects: np.ndarray,
    prefix_repeats: dict[str, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Expand row indices without splitting repeats across subject groups."""

    subjects = np.asarray(subjects).astype(str)
    if subjects.ndim != 1 or len(subjects) == 0:
        raise ValueError("subjects must be a non-empty one-dimensional array.")
    row_repeats = np.ones(len(subjects), dtype=np.int64)
    matched = np.zeros(len(subjects), dtype=bool)
    records: list[dict[str, object]] = []
    for prefix, repeat in prefix_repeats.items():
        prefix_mask = np.fromiter(
            (subject.startswith(prefix) for subject in subjects),
            dtype=bool,
            count=len(subjects),
        )
        if not bool(prefix_mask.any()):
            raise ValueError(
                f"subject-prefix-repeat prefix {prefix!r} matches no retained source rows."
            )
        if bool((matched & prefix_mask).any()):
            overlapping = sorted(set(subjects[matched & prefix_mask].tolist()))
            raise ValueError(
                "subject-prefix-repeat prefixes overlap for retained subjects: "
                f"{overlapping[:5]}"
            )
        matched |= prefix_mask
        row_repeats[prefix_mask] = repeat
        unique_rows = int(prefix_mask.sum())
        records.append(
            {
                "prefix": prefix,
                "repeat": repeat,
                "unique_physical_rows": unique_rows,
                "optimizer_rows": unique_rows * repeat,
                "unique_subjects": int(len(np.unique(subjects[prefix_mask]))),
            }
        )
    unmatched_rows = int((~matched).sum())
    if unmatched_rows:
        records.append(
            {
                "prefix": None,
                "repeat": 1,
                "unique_physical_rows": unmatched_rows,
                "optimizer_rows": unmatched_rows,
                "unique_subjects": int(len(np.unique(subjects[~matched]))),
            }
        )
    indices = np.repeat(np.arange(len(subjects), dtype=np.int64), row_repeats)
    optimizer_rows = int(len(indices))
    for record in records:
        record["optimizer_fraction"] = float(record["optimizer_rows"] / optimizer_rows)
    report: dict[str, object] = {
        "method": "deterministic_full_row_repeat",
        "unique_physical_rows": int(len(subjects)),
        "optimizer_rows_per_epoch": optimizer_rows,
        "all_unique_rows_retained": bool(len(np.unique(indices)) == len(subjects)),
        "prefixes": records,
    }
    return indices, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--holdout-subjects", default="", help="comma separated; never pretrain on these")
    parser.add_argument(
        "--cohort",
        choices=("default", "p300_causal", "gtn", "gtn_paper"),
        default="default",
        help=(
            "Causal contract family asserted for forward-phase source caches: "
            "'gtn' enforces the revised 0.1 Hz / 1200 ms child-cohort contract."
        ),
    )
    parser.add_argument(
        "--tmax-ms",
        type=float,
        default=None,
        help="Explicit epoch-end recipe override for matched factorials.",
    )
    parser.add_argument(
        "--pooling-mode",
        default=DEFAULT_N2P3_POOLING_MODE,
        choices=sorted({"ms_flatten", "full_unfold", "mlp_full_unfold", "quadratic_full_unfold"}),
    )
    parser.add_argument(
        "--temporal-kernel-size",
        type=int,
        default=DEFAULT_N2P3_ARCHITECTURE.temporal_kernel_size,
        help="ST temporal kernel width for the trunk (odd, >=3).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--subject-prefix-repeat",
        default="",
        help=(
            "Optional comma-separated PREFIX=INTEGER exposure contract. Every unique row is "
            "retained; matching rows are deterministically repeated before group-disjoint "
            "selection and full refit. Example: BI::=3,BNCI::=1."
        ),
    )
    parser.add_argument(
        "--qc-ptp-uv",
        type=float,
        default=0.0,
        help=(
            "Drop source epochs whose peak-to-peak amplitude on any channel "
            "exceeds this many microvolts (the GTN source study uses 100). "
            "0 disables QC; applies to pretraining rows only."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.source_cache, require_labels=True, validation="attested")
    source_cache_sha256 = str(read_epoch_cache_attestation(args.source_cache)["sha256"])
    expected_contract = SOURCE_COHORT_CONTRACTS[args.cohort]
    if args.tmax_ms is not None:
        expected_contract = replace(expected_contract, tmax_ms=float(args.tmax_ms))
    assert_p300_input_contract(dataset.preprocessing, expected_contract)
    holdout = {item.strip() for item in args.holdout_subjects.split(",") if item.strip()}
    subjects = np.asarray(dataset.subject_ids).astype(str)
    all_subjects = set(subjects.tolist())
    unknown = holdout - all_subjects
    if unknown:
        raise ValueError(f"holdout subjects absent from cache: {sorted(unknown)}")
    source_rows = ~np.isin(subjects, list(holdout))
    labels_all = np.asarray(dataset.y, dtype=np.int64)
    source_label_counts_before_qc = np.bincount(
        labels_all[source_rows], minlength=2
    ).astype(int)
    qc_dropped = 0
    qc_dropped_by_label = np.zeros(2, dtype=np.int64)
    if args.qc_ptp_uv > 0:
        threshold = float(args.qc_ptp_uv) * 1e-6
        X_all = np.asarray(dataset.X, dtype=np.float32)
        ptp = X_all.max(axis=2) - X_all.min(axis=2)  # (N, C)
        bad = (ptp >= threshold).any(axis=1)
        qc_dropped = int((source_rows & bad).sum())
        qc_dropped_by_label = np.bincount(
            labels_all[source_rows & bad], minlength=2
        ).astype(np.int64)
        source_rows = source_rows & ~bad
    if int(source_rows.sum()) < 1000:
        raise ValueError("too few source rows remain after holdout exclusion.")

    prefix_repeats = parse_subject_prefix_repeats(args.subject_prefix_repeat)
    physical_subjects = subjects[source_rows]
    exposure_indices, source_exposure = build_subject_prefix_exposure(
        physical_subjects,
        prefix_repeats,
    )

    runtime = GpuPerformanceScheduler(device, precision="fp32")
    config = DeepConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    architecture = N2P3ArchitectureConfig(temporal_kernel_size=args.temporal_kernel_size)
    baseline = N2P3NetBaseline(
        dataset.n_channels,
        dataset.n_times,
        dataset.preprocessing.sfreq,
        config=config,
        device=device,
        runtime=runtime,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode=args.pooling_mode,
        architecture=architecture,
    )

    X_physical = np.asarray(dataset.X[source_rows])
    y_physical = np.asarray(dataset.y)[source_rows]
    X = np.ascontiguousarray(X_physical[exposure_indices])
    y = y_physical[exposure_indices]
    src_subjects = physical_subjects[exposure_indices]
    started = time.perf_counter()
    baseline.fit(X, y, group_ids=src_subjects)
    fit_sec = time.perf_counter() - started
    selection_baseline = baseline
    selection_history = getattr(selection_baseline, "last_history", {}) or {}
    selected_epoch = selection_history.get("best_epoch")
    if selected_epoch is None:
        raise RuntimeError("source pretraining did not select an inner-validation epoch.")
    refit_epochs = int(selected_epoch) + 1
    refit_config = replace(config, epochs=refit_epochs, val_group_frac=None)
    baseline = N2P3NetBaseline(
        dataset.n_channels,
        dataset.n_times,
        dataset.preprocessing.sfreq,
        config=refit_config,
        device=device,
        runtime=runtime,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode=args.pooling_mode,
        architecture=architecture,
    )
    refit_started = time.perf_counter()
    baseline.fit(X, y, group_ids=None)
    refit_sec = time.perf_counter() - refit_started
    model = baseline.model_

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = selection_history
    source_selection_calibration = None
    if (
        selection_baseline.calibration_logits_ is not None
        and selection_baseline.calibration_labels_ is not None
    ):
        source_selection_calibration = asdict(
            fit_weighted_logit_temperature(
                selection_baseline.calibration_logits_,
                selection_baseline.calibration_labels_,
                pos_weight=float(selection_baseline.training_pos_weight_),
                train_prior=float(selection_baseline.training_prior_),
                source=str(
                    selection_baseline.calibration_source_ or "source_group_validation"
                ),
            )
        )
    source_calibration = {
        "pos_weight": float(baseline.training_pos_weight_),
        "train_prior": float(baseline.training_prior_),
        "temperature": 1.0,
        "source": "source_full_refit_weighted_ce_analytic",
    }
    payload = {
        "trunk_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_mean": np.asarray(baseline._input_mean, dtype=np.float32).squeeze().tolist(),
        "input_std": np.asarray(baseline._input_std, dtype=np.float32).squeeze().tolist(),
        "config": {
            "pooling_mode": args.pooling_mode,
            "temporal_kernel_size": args.temporal_kernel_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "subject_prefix_repeat": prefix_repeats,
            "training": "N2P3NetBaseline supervised (LOSO-identical path)",
        },
        "architecture": model.architecture_record(),
        "n_channels": int(dataset.n_channels),
        "n_times": int(dataset.n_times),
        "input_sample_rate_hz": float(dataset.preprocessing.sfreq),
        "input_tmin_s": float(dataset.preprocessing.tmin_ms) / 1000.0,
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_channel_names": list(dataset.channel_names),
        "input_source_reference": dataset.provenance.get("source_reference"),
        "source_cache_sha256": source_cache_sha256,
        "classifier_trained": True,
        "training_pos_weight": float(baseline.training_pos_weight_),
        "training_prior": float(baseline.training_prior_),
        "source_calibration": source_calibration,
        "source_selection_calibration": source_selection_calibration,
        "source_full_refit": True,
        "source_refit_epochs": refit_epochs,
        "source_cache": str(Path(args.source_cache).resolve()),
        "holdout_subjects": sorted(holdout),
        "source_dataset_name": dataset.name,
        "source_subjects": sorted(all_subjects),
        "training_subjects": sorted(all_subjects - holdout),
        "training_subject_keys": [
            f"{dataset.name}\0{subject}" for subject in sorted(all_subjects - holdout)
        ],
        "training_cache_subject_keys": [
            f"{source_cache_sha256}\0{subject}"
            for subject in sorted(all_subjects - holdout)
        ],
        "n_source_epochs_used": int(len(X)),
        "n_unique_source_epochs_used": int(len(X_physical)),
        "n_optimizer_source_rows_per_epoch": int(len(X)),
        "source_exposure": source_exposure,
        "source_label_counts_unique_after_qc": np.bincount(
            y_physical, minlength=2
        ).astype(int).tolist(),
        "source_label_counts_optimizer_rows": np.bincount(
            y, minlength=2
        ).astype(int).tolist(),
        "qc_ptp_uv": float(args.qc_ptp_uv),
        "qc_dropped_source_epochs": qc_dropped,
        "source_label_counts_before_qc": source_label_counts_before_qc.tolist(),
        "qc_dropped_source_epochs_by_label": qc_dropped_by_label.tolist(),
        "source_label_retention_by_label": (
            (source_label_counts_before_qc - qc_dropped_by_label)
            / np.maximum(source_label_counts_before_qc, 1)
        ).tolist(),
        "fit_seconds": fit_sec,
        "refit_seconds": refit_sec,
        "best_epoch": history.get("best_epoch"),
        "final_task_val_auc": history.get("final_task_val_auc"),
        "runtime": {"device": str(device)},
    }
    torch.save(payload, checkpoint)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "fit_seconds": fit_sec,
                "n_source_epochs_used": payload["n_source_epochs_used"],
                "n_unique_source_epochs_used": payload["n_unique_source_epochs_used"],
                "source_exposure": source_exposure,
                "training_subjects": len(payload["training_subjects"]),
                "best_epoch": history.get("best_epoch"),
                "final_task_val_auc": history.get("final_task_val_auc"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
