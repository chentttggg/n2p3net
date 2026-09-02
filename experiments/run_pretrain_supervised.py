"""Supervised source pretraining for the N2P3 trunk (Gate 3-S1).

Trains N2P3NetBaseline (the exact LOSO-floor training path) on every source
subject except the holdout block. The saved checkpoint carries the full model
state dict plus a structured participant-identity ledger derived from the rows
that actually reach fitting.
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
    SOURCE_COHORT_DATA_CONTRACTS,
    assert_p300_input_contract,
)
from data.domain import source_domain_ids  # noqa: E402
from data.epochs import load_epoch_dataset, loaded_epoch_cache_attestation  # noqa: E402
from data.identity import training_identity_ledger_from_rows  # noqa: E402
from models.n2p3net import (  # noqa: E402
    DEFAULT_N2P3_ARCHITECTURE,
    DEFAULT_N2P3_POOLING_MODE,
    N2P3ArchitectureConfig,
)
from research.contracts import TrainingRunContract  # noqa: E402
from research.evaluation import (  # noqa: E402
    source_snapshot_sha256_from_archive_manifest,
)
from train.device import get_device  # noqa: E402
from train.runtime import GpuPerformanceScheduler  # noqa: E402
from transfer.checkpoint import CHECKPOINT_SCHEMA  # noqa: E402


def parse_source_domain_mass(values: list[str]) -> dict[str, float]:
    """Parse exact DOMAIN=MASS entries without subject-prefix aliases."""

    masses: dict[str, float] = {}
    for value in values:
        domain, separator, raw_mass = value.rpartition("=")
        domain = domain.strip()
        if not separator or not domain or domain in masses:
            raise ValueError("source-domain-mass must use unique DOMAIN=FLOAT entries.")
        try:
            mass = float(raw_mass)
        except ValueError as error:
            raise ValueError("source-domain-mass values must be numeric.") from error
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("source-domain-mass values must be finite and positive.")
        masses[domain] = mass
    if masses and not np.isclose(sum(masses.values()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("source-domain-mass values must sum to one.")
    return masses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument(
        "--source-snapshot-manifest",
        type=Path,
        required=True,
        help="Manifest for the physical source archive used for this training run.",
    )
    parser.add_argument("--holdout-subjects", default="", help="comma separated; never pretrain on these")
    parser.add_argument(
        "--cohort",
        choices=tuple(SOURCE_COHORT_DATA_CONTRACTS),
        default="causal",
        help=(
            "Causal contract family asserted for forward-phase source caches: "
            "'causal' is the current 0.1 Hz / 1200 ms forward steady-state contract."
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
        "--precision",
        choices=("auto", "bf16", "fp32"),
        default="auto",
        help="Requested training precision; auto selects BF16 on supported accelerators.",
    )
    parser.add_argument(
        "--source-domain-mass",
        action="append",
        default=[],
        metavar="DOMAIN=FLOAT",
        help="Exact source-domain risk mass; repeat once for every prepared source domain.",
    )
    parser.add_argument(
        "--risk-within-domain-unit",
        choices=("epoch", "participant"),
        default="participant",
        help="Statistical unit receiving equal mass inside each source domain.",
    )
    parser.add_argument(
        "--selection-domain",
        default="",
        help="Exact source domain used for group-disjoint epoch selection and calibration.",
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
    source_snapshot_manifest = args.source_snapshot_manifest.resolve()
    source_snapshot_sha256 = source_snapshot_sha256_from_archive_manifest(
        source_snapshot_manifest
    )

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.source_cache, require_labels=True, validation="attested")
    source_cache_sha256 = str(loaded_epoch_cache_attestation(dataset)["sha256"])
    expected_contract = SOURCE_COHORT_DATA_CONTRACTS[args.cohort]
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

    if dataset.identity_table is None:
        raise ValueError("source cache lacks the required participant identity table.")
    training_identity_ledger = training_identity_ledger_from_rows(
        dataset.identity_table,
        dataset.subject_ids,
        source_rows,
    )
    src_subjects = subjects[source_rows]
    domain_mass = parse_source_domain_mass(args.source_domain_mass)
    source_domains = None
    selection_domain = args.selection_domain.strip() or None
    if dataset.provenance.get("source_domain_axis") is not None:
        source_domains = source_domain_ids(dataset)[source_rows]
        observed_domains = set(source_domains.tolist())
        if len(observed_domains) > 1:
            if selection_domain is None:
                raise ValueError(
                    "multi-domain source training requires an explicit selection-domain."
                )
            if domain_mass and set(domain_mass) != observed_domains:
                raise ValueError(
                    "source-domain-mass keys must equal retained source domains: "
                    f"{sorted(observed_domains)}."
                )
            if selection_domain not in observed_domains:
                raise ValueError("selection-domain is absent from retained source rows.")
        elif domain_mass or selection_domain is not None:
            raise ValueError("domain-risk options require a genuinely multi-domain source cache.")
    elif domain_mass or selection_domain is not None:
        raise ValueError("domain-risk options require an explicit source-domain axis.")

    config = DeepConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        precision=args.precision,
    )
    runtime = GpuPerformanceScheduler(
        device,
        precision=config.precision,
        batch_memory_fraction=config.batch_memory_fraction,
        preload_memory_fraction=config.preload_memory_fraction,
    )
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

    X = np.ascontiguousarray(dataset.X[source_rows])
    y = np.asarray(dataset.y, dtype=np.int64)[source_rows]
    source_input_stats_scope = {
        "mode": "all_source_rows",
        "rows": int(len(X)),
        "unique_subjects": int(len(np.unique(src_subjects))),
    }
    risk_fit_kwargs = {}
    if source_domains is not None and selection_domain is not None:
        risk_fit_kwargs.update(
            {
                "source_domain_ids": source_domains,
                "selection_domain": selection_domain,
            }
        )
    if domain_mass:
        risk_fit_kwargs.update(
            {
                "source_domain_mass": domain_mass,
                "risk_unit_ids": src_subjects,
                "risk_within_domain_unit": args.risk_within_domain_unit,
            }
        )
    started = time.perf_counter()
    baseline.fit(
        X,
        y,
        group_ids=src_subjects,
        **risk_fit_kwargs,
    )
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
    baseline.fit(
        X,
        y,
        group_ids=None,
        **risk_fit_kwargs,
    )
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
    assert dataset.identity_table is not None
    holdout_participant_keys = (
        dataset.identity_table.subset(holdout).authority_keys()
        if holdout
        else ()
    )
    training_contract = TrainingRunContract(
        source_cache_sha256=source_cache_sha256,
        source_identity_digest=dataset.identity_table.digest(),
        source_snapshot_sha256=source_snapshot_sha256,
        architecture=model.architecture_record(),
        preprocessing={
            "epoch": asdict(dataset.preprocessing),
            "channel_names": list(dataset.channel_names),
            "source_reference": dataset.provenance.get("source_reference"),
        },
        optimizer={
            "name": "torch.optim.Adam",
            "selection_config": asdict(config),
            "refit_config": asdict(refit_config),
            "selection_execution": selection_baseline.optimizer_execution.record(),
            "refit_execution": baseline.optimizer_execution.record(),
            "selection_runtime": selection_baseline.last_runtime,
            "refit_runtime": baseline.last_runtime,
            "optimizer_rows_per_epoch": int(len(X)),
        },
        validation={
            "strategy": "group_disjoint_epoch_selection_then_full_source_refit",
            "group_key": "local_subject_id",
            "selected_epoch_zero_based": int(selected_epoch),
            "refit_epochs": int(refit_epochs),
            "selection_calibration_source": selection_baseline.calibration_source_,
            "selection_domain": selection_domain,
            "full_source_refit": True,
        },
        objective={
            "name": "weighted_binary_cross_entropy",
            "effective_pos_weight": float(baseline.training_pos_weight_),
            "training_prior": float(baseline.training_prior_),
            "qc_ptp_uv": float(args.qc_ptp_uv),
            "input_statistics_scope": source_input_stats_scope,
            "label_counts": np.bincount(y, minlength=2).astype(int).tolist(),
            "source_risk_selection": selection_baseline.last_source_risk,
            "source_risk_refit": baseline.last_source_risk,
        },
        seed=int(args.seed),
        training_participant_keys=training_identity_ledger.authority_keys(),
        holdout_participant_keys=holdout_participant_keys,
    )
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "trunk_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_mean": np.asarray(baseline._input_mean, dtype=np.float32).squeeze().tolist(),
        "input_std": np.asarray(baseline._input_std, dtype=np.float32).squeeze().tolist(),
        "config": {
            "pooling_mode": args.pooling_mode,
            "temporal_kernel_size": args.temporal_kernel_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "source_domain_mass": domain_mass or None,
            "risk_within_domain_unit": args.risk_within_domain_unit if domain_mass else None,
            "selection_domain": selection_domain,
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
        "source_calibration": source_calibration,
        "source_selection_calibration": source_selection_calibration,
        "source_full_refit": True,
        "source_refit_epochs": refit_epochs,
        "source_cache": str(Path(args.source_cache).resolve()),
        "source_snapshot_manifest": str(source_snapshot_manifest),
        "source_snapshot_sha256": source_snapshot_sha256,
        "holdout_subjects": sorted(holdout),
        "source_dataset_name": dataset.name,
        "source_subjects": sorted(all_subjects),
        "training_identity_ledger": training_identity_ledger.payload(),
        "training_identity_ledger_digest": training_identity_ledger.digest(),
        "training_contract": training_contract.record(),
        "training_contract_digest": training_contract.digest(),
        "n_source_epochs_used": int(len(X)),
        "source_input_stats_scope": source_input_stats_scope,
        "source_label_counts_after_qc": np.bincount(y, minlength=2).astype(int).tolist(),
        "source_risk_selection": selection_baseline.last_source_risk,
        "source_risk_refit": baseline.last_source_risk,
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
        "runtime": {
            "device": str(device),
            "selection": selection_baseline.last_runtime,
            "refit": baseline.last_runtime,
        },
    }
    torch.save(payload, checkpoint)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "fit_seconds": fit_sec,
                "n_source_epochs_used": payload["n_source_epochs_used"],
                "source_input_stats_scope": source_input_stats_scope,
                "source_risk": payload["source_risk_selection"],
                "runtime": payload["runtime"],
                "training_participant_count": len(
                    training_identity_ledger.local_subject_ids
                ),
                "best_epoch": history.get("best_epoch"),
                "final_task_val_auc": history.get("final_task_val_auc"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
