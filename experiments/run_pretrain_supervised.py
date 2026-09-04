"""Supervised source pretraining for the N2P3 trunk and feature-transfer wrapper.

Trains N2P3NetBaseline (the exact LOSO-floor training path) on every source
subject except the holdout block. The saved checkpoint carries the full model
state dict plus a structured participant-identity ledger derived from the rows
that actually reach fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
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
from data.domain import source_domain_ids, source_domain_vocabulary  # noqa: E402
from data.epochs import (  # noqa: E402
    EpochDataset,
    epoch_row_identity_digest,
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
)
from data.identity import training_identity_ledger_from_rows  # noqa: E402
from models.adapters import (  # noqa: E402
    FeatureTransferConfig,
    SourceResidualContractError,
)
from models.n2p3net import (  # noqa: E402
    DEFAULT_N2P3_ARCHITECTURE,
    DEFAULT_N2P3_POOLING_MODE,
    N2P3ArchitectureConfig,
)
from models.splitbn import SPLITBN_TARGET_BUFFER_SOURCES  # noqa: E402
from research.contracts import TrainingRunContract, semantic_sha256  # noqa: E402
from research.evaluation import (  # noqa: E402
    source_snapshot_sha256_from_archive_manifest,
)
from train.device import get_device, release_device_memory  # noqa: E402
from train.runtime import GpuPerformanceScheduler  # noqa: E402
from train.source_sampling import (  # noqa: E402
    PDSSelection,
    load_pds_selection_record,
    pds_selection_from_record,
)
from transfer.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    checkpoint_training_contract,
    load_checkpoint_payload,
)
from transfer.decision_loss import (  # noqa: E402
    DecisionLossConfig,
    DirectDecisionContractError,
    DirectDecisionMetadata,
)

SOURCE_PRETRAIN_BATCH_SCHEMA = "n2p3_source_pretrain_batch/1"
SOURCE_PRETRAIN_INVALID_SCHEMA = "n2p3_source_pretrain_invalid/1"


@dataclass(frozen=True)
class SourcePretrainJob:
    holdout_subjects: str
    checkpoint: Path


_SOURCE_DATASET_CACHE: dict[Path, tuple[EpochDataset, dict[str, object]]] = {}
_SOURCE_SNAPSHOT_CACHE: dict[Path, str] = {}


def _load_source_dataset_once(path: str | Path) -> tuple[EpochDataset, dict[str, object]]:
    resolved = Path(path).resolve()
    cached = _SOURCE_DATASET_CACHE.get(resolved)
    if cached is None:
        dataset = load_epoch_dataset(resolved, require_labels=True, validation="attested")
        cached = (dataset, dict(loaded_epoch_cache_attestation(dataset)))
        _SOURCE_DATASET_CACHE[resolved] = cached
    return cached


def _source_snapshot_sha256_once(path: Path) -> str:
    resolved = path.resolve()
    cached = _SOURCE_SNAPSHOT_CACHE.get(resolved)
    if cached is None:
        cached = source_snapshot_sha256_from_archive_manifest(resolved)
        _SOURCE_SNAPSHOT_CACHE[resolved] = cached
    return cached


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _invalid_record_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path.with_suffix(path.suffix + ".invalid.json")


def _write_contract_failure(
    checkpoint: str | Path,
    *,
    failure: SourceResidualContractError | DirectDecisionContractError,
    source_cache: str | Path,
    source_cache_sha256: str,
    source_snapshot_manifest: Path,
    source_snapshot_sha256: str,
    holdout_subjects: set[str],
) -> Path:
    """Persist a typed invalid run before propagating a scientific-contract error."""

    output = _invalid_record_path(checkpoint)
    _write_json_atomic(
        output,
        {
            "schema": SOURCE_PRETRAIN_INVALID_SCHEMA,
            "run_status": "invalid",
            "checkpoint_requested": str(Path(checkpoint).resolve()),
            "checkpoint_written": False,
            "source_cache": str(Path(source_cache).resolve()),
            "source_cache_sha256": source_cache_sha256,
            "source_snapshot_manifest": str(source_snapshot_manifest),
            "source_snapshot_sha256": source_snapshot_sha256,
            "holdout_subjects": sorted(holdout_subjects),
            "failure_type": type(failure).__name__,
            "failure_ledger": dict(failure.record),
        },
    )
    return output


def _load_source_selection(
    path: str | Path,
    *,
    source_cache_sha256: str,
    row_identity_digest: str | None = None,
) -> tuple[PDSSelection, dict[str, object]]:
    """Load a PDS selection record bound to the attested source cache.

    The returned raw record is embedded in the training contract unchanged
    (except by the contract's canonical serializer), so the checkpoint carries
    the exact selection provenance rather than a reconstructed summary.
    """

    _, record = load_pds_selection_record(path)
    selection = pds_selection_from_record(
        record,
        expected_source_cache_sha256=source_cache_sha256,
    )
    if row_identity_digest is not None:
        feature_record = record.get("source_feature_record")
        if not isinstance(feature_record, Mapping):
            raise ValueError("source-selection record lacks its source_feature_record.")
        feature_row_digest = feature_record.get("row_identity_digest")
        if feature_row_digest != row_identity_digest:
            raise ValueError(
                "source-selection feature row_identity_digest disagrees with "
                "the loaded source cache."
            )
    return selection, record


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


def parse_splitbn_source_weights(values: list[str]) -> dict[str, float]:
    """Parse a named Split-BN source composition independently of CE risk."""

    weights: dict[str, float] = {}
    for value in values:
        domain, separator, raw_weight = value.rpartition("=")
        domain = domain.strip()
        if not separator or not domain or domain in weights:
            raise ValueError(
                "splitbn-source-weight must use unique DOMAIN=FLOAT entries."
            )
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError("splitbn-source-weight values must be numeric.") from error
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "splitbn-source-weight values must be finite and non-negative."
            )
        weights[domain] = weight
    if weights and not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("splitbn-source-weight values must sum to one.")
    return weights


def _single_job_argv(
    args: argparse.Namespace,
    job: SourcePretrainJob,
) -> list[str]:
    argv = [
        "--source-cache",
        str(args.source_cache),
        "--source-snapshot-manifest",
        str(args.source_snapshot_manifest),
        "--holdout-subjects",
        job.holdout_subjects,
        "--cohort",
        args.cohort,
        "--pooling-mode",
        args.pooling_mode,
        "--temporal-kernel-size",
        str(args.temporal_kernel_size),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--precision",
        args.precision,
        "--risk-within-domain-unit",
        args.risk_within_domain_unit,
        "--qc-ptp-uv",
        str(args.qc_ptp_uv),
        "--feature-transfer-invariant-bottleneck",
        str(args.feature_transfer_invariant_bottleneck),
        "--feature-transfer-private-channels",
        str(args.feature_transfer_private_channels),
        "--feature-transfer-target-bottleneck",
        str(args.feature_transfer_target_bottleneck),
        "--feature-transfer-target-kernel",
        str(args.feature_transfer_target_kernel),
        "--feature-transfer-max-invariant-residual",
        str(args.feature_transfer_max_invariant_residual),
        "--feature-transfer-max-target-residual",
        str(args.feature_transfer_max_target_residual),
        "--private-domain-weight",
        str(args.private_domain_weight),
        "--orthogonality-weight",
        str(args.orthogonality_weight),
        "--feature-alignment-weight",
        str(args.feature_alignment_weight),
        "--feature-alignment-mode",
        args.feature_alignment_mode,
        "--feature-alignment-min-cell-count",
        str(args.feature_alignment_min_cell_count),
        "--feature-alignment-inactive-policy",
        args.feature_alignment_inactive_policy,

        "--checkpoint",
        str(job.checkpoint),
        "--device",
        args.device,
    ]
    if args.source_selection_record is not None:
        argv.extend(
            ["--source-selection-record", str(args.source_selection_record)]
        )
    if args.tmax_ms is not None:
        argv.extend(["--tmax-ms", str(args.tmax_ms)])
    for value in args.source_domain_mass:
        argv.extend(["--source-domain-mass", value])
    if args.source_residual_drop_p is not None:
        argv.extend(
            [
                "--source-residual-drop-p",
                str(args.source_residual_drop_p),
                "--source-residual-drop-mode",
                args.source_residual_drop_mode,
            ]
        )
    if args.splitbn_target_buffer_source is not None:
        argv.extend(
            [
                "--splitbn-target-buffer-source",
                args.splitbn_target_buffer_source,
            ]
        )
    for value in args.splitbn_source_weight:
        argv.extend(["--splitbn-source-weight", value])
    if args.candidate_loss_weight is not None:
        argv.extend(
            [
                "--candidate-loss-weight",
                str(args.candidate_loss_weight),
                "--candidate-loss-temperature",
                str(args.candidate_loss_temperature),
                "--candidate-loss-prefix-mode",
                args.candidate_loss_prefix_mode,
                "--candidate-loss-max-repetitions",
                str(args.candidate_loss_max_repetitions),
            ]
        )
    if args.selection_domain:
        argv.extend(["--selection-domain", args.selection_domain])
    return argv


def _run_batch(args: argparse.Namespace, jobs: tuple[SourcePretrainJob, ...]) -> None:
    checkpoints = [job.checkpoint.resolve() for job in jobs]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("source pretrain batch checkpoint paths must be unique.")
    device = torch.device(args.device) if args.device != "auto" else get_device()
    records: list[dict[str, object]] = []
    for job in jobs:
        main(_single_job_argv(args, job))
        checkpoint = job.checkpoint.resolve()
        payload = load_checkpoint_payload(checkpoint)
        training = checkpoint_training_contract(payload)
        records.append(
            {
                "holdout_subjects": sorted(
                    item.strip() for item in job.holdout_subjects.split(",") if item.strip()
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "checkpoint_byte_size": checkpoint.stat().st_size,
                "training_contract_digest": training.digest(),
            }
        )
        release_device_memory(device)
    first = load_checkpoint_payload(checkpoints[0])
    batch_record = {
        "schema": SOURCE_PRETRAIN_BATCH_SCHEMA,
        "completed": True,
        "source_cache": str(Path(args.source_cache).resolve()),
        "source_cache_sha256": first["source_cache_sha256"],
        "source_snapshot_manifest": str(Path(args.source_snapshot_manifest).resolve()),
        "source_snapshot_sha256": first["source_snapshot_sha256"],
        "seed": int(args.seed),
        "job_count": len(records),
        "jobs": records,
    }
    _write_json_atomic(args.batch_record.resolve(), batch_record)
    print(
        json.dumps(
            {
                "batch_record": str(args.batch_record.resolve()),
                "job_count": len(records),
                "source_cache_sha256": batch_record["source_cache_sha256"],
                "source_snapshot_sha256": batch_record["source_snapshot_sha256"],
            }
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument(
        "--source-snapshot-manifest",
        type=Path,
        required=True,
        help="Manifest for the physical source archive used for this training run.",
    )
    parser.add_argument(
        "--holdout-subjects",
        default="",
        help="comma separated; never pretrain on these",
    )
    parser.add_argument(
        "--source-selection-record",
        type=Path,
        default=None,
        help=(
            "PDS source-row selection JSON from prepare_source_selection.py; "
            "selected rows are intersected with holdout/QC masks. Batch mode "
            "requires one record shared by every job so the arm procedure stays uniform."
        ),
    )
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
        "--splitbn-target-buffer-source",
        choices=sorted(SPLITBN_TARGET_BUFFER_SOURCES),
        default=None,
        help=(
            "Enable Split-BN and freeze the unseen-target buffer estimand. "
            "The source domain count and index order are derived from cache provenance."
        ),
    )
    parser.add_argument(
        "--splitbn-source-weight",
        action="append",
        default=[],
        metavar="DOMAIN=FLOAT",
        help=(
            "Named source-buffer composition; required exactly for "
            "--splitbn-target-buffer-source=source_weighted."
        ),
    )
    parser.add_argument(
        "--feature-transfer-invariant-bottleneck",
        type=int,
        default=0,
        help="Enable the shared invariant feature path with this residual bottleneck width; 0 keeps the plain core.",
    )
    parser.add_argument(
        "--feature-transfer-private-channels",
        type=int,
        default=4,
        help="Private style representation width when feature transfer is enabled.",
    )
    parser.add_argument(
        "--feature-transfer-target-bottleneck",
        type=int,
        default=4,
        help="Target-only residual adapter bottleneck width.",
    )
    parser.add_argument(
        "--feature-transfer-target-kernel",
        type=int,
        default=9,
        help="Target-only residual adapter temporal kernel (odd, >=3).",
    )
    parser.add_argument(
        "--feature-transfer-max-invariant-residual",
        type=float,
        default=0.25,
        help="Bound on the shared invariant residual scale.",
    )
    parser.add_argument(
        "--feature-transfer-max-target-residual",
        type=float,
        default=0.25,
        help="Bound on the target-only residual scale.",
    )
    parser.add_argument(
        "--source-residual-drop-p",
        type=float,
        default=None,
        help=(
            "Enable source-domain residuals with D0 p=0 or a preregistered "
            "drop probability in [0.2,0.5]."
        ),
    )
    parser.add_argument(
        "--source-residual-drop-mode",
        choices=("sample", "batch"),
        default="sample",
        help="Bernoulli mask granularity for an enabled source residual arm.",
    )
    parser.add_argument(
        "--candidate-loss-weight",
        type=float,
        default=None,
        help=(
            "Enable complete-decision batching; 0 is the matched grouped control and "
            "[0.1,0.5] adds the direct-candidate loss."
        ),
    )
    parser.add_argument(
        "--candidate-loss-temperature",
        type=float,
        default=1.0,
        help="Predeclared positive temperature for direct-candidate softmax.",
    )
    parser.add_argument(
        "--candidate-loss-prefix-mode",
        choices=("final", "uniform"),
        default="final",
        help="Use alpha_R=1 or uniform alpha over prefixes 1..R.",
    )
    parser.add_argument(
        "--candidate-loss-max-repetitions",
        type=int,
        default=None,
        help="Required frozen direct-decision evidence budget when P2 is enabled.",
    )
    parser.add_argument(
        "--feature-alignment-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the class-conditional invariant-feature mean-alignment loss "
            "across source domains; 0 disables this auxiliary objective."
        ),
    )
    parser.add_argument(
        "--feature-alignment-mode",
        choices=("mean", "conditional_rbf_mmd"),
        default="mean",
        help=(
            "Feature alignment objective: class-conditional mean distance or "
            "conditional RBF-MMD across source domains."
        ),
    )
    parser.add_argument(
        "--feature-alignment-min-cell-count",
        type=int,
        default=8,
        help="Minimum (domain, class) cell rows required for an active alignment term.",
    )
    parser.add_argument(
        "--feature-alignment-inactive-policy",
        choices=("record", "fail_closed"),
        default="record",
        help=(
            "record logs inactive batches and continues; fail_closed aborts the fold "
            "when an alignment-weighted batch has no active cell pair."
        ),
    )

    parser.add_argument(
        "--private-domain-weight",
        type=float,
        default=0.0,
        help="Weight of private-style source-domain supervision; requires feature transfer and multiple domains.",
    )
    parser.add_argument(
        "--orthogonality-weight",
        type=float,
        default=0.0,
        help="Weight of invariant/private cross-covariance penalty; requires feature transfer.",
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
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--job",
        action="append",
        nargs=2,
        default=[],
        metavar=("HOLDOUTS", "CHECKPOINT"),
        help="Batch mode job; repeat to train several holdouts after one cache load.",
    )
    parser.add_argument(
        "--batch-record",
        type=Path,
        help="Completed batch sentinel with checkpoint hashes and contract digests.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if (
        args.candidate_loss_weight is None
        and args.candidate_loss_max_repetitions is not None
    ):
        parser.error(
            "--candidate-loss-max-repetitions requires --candidate-loss-weight"
        )
    if (
        args.candidate_loss_weight is not None
        and args.candidate_loss_max_repetitions is None
    ):
        parser.error(
            "--candidate-loss-weight requires --candidate-loss-max-repetitions"
        )
    if args.candidate_loss_weight is None and (
        args.candidate_loss_temperature != 1.0
        or args.candidate_loss_prefix_mode != "final"
    ):
        parser.error(
            "candidate loss temperature/prefix options require --candidate-loss-weight"
        )
    if args.job:
        if args.checkpoint is not None or args.holdout_subjects:
            parser.error("--job cannot be combined with --checkpoint/--holdout-subjects")
        if args.batch_record is None:
            parser.error("--job requires --batch-record")
        jobs = tuple(
            SourcePretrainJob(holdout_subjects=holdout, checkpoint=Path(checkpoint))
            for holdout, checkpoint in args.job
        )
        _run_batch(args, jobs)
        return
    if args.checkpoint is None:
        parser.error("single-job mode requires --checkpoint")
    if args.batch_record is not None:
        parser.error("--batch-record requires at least one --job")
    source_snapshot_manifest = args.source_snapshot_manifest.resolve()
    source_snapshot_sha256 = _source_snapshot_sha256_once(source_snapshot_manifest)

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset, source_cache_attestation = _load_source_dataset_once(args.source_cache)
    source_cache_sha256 = str(source_cache_attestation["sha256"])
    expected_contract = SOURCE_COHORT_DATA_CONTRACTS[args.cohort]
    if args.tmax_ms is not None:
        expected_contract = replace(expected_contract, tmax_ms=float(args.tmax_ms))
    assert_p300_input_contract(dataset.preprocessing, expected_contract)
    source_selection = None
    source_selection_record: dict[str, object] | None = None
    if args.source_selection_record is not None:
        source_selection, source_selection_record = _load_source_selection(
            args.source_selection_record,
            source_cache_sha256=source_cache_sha256,
            row_identity_digest=epoch_row_identity_digest(dataset),
        )
        if source_selection.source_count != dataset.n_epochs:
            raise ValueError(
                "PDS source-selection source_count disagrees with the loaded cache "
                f"({source_selection.source_count} vs {dataset.n_epochs})."
            )
    holdout = {item.strip() for item in args.holdout_subjects.split(",") if item.strip()}
    subjects = np.asarray(dataset.subject_ids).astype(str)
    all_subjects = set(subjects.tolist())
    unknown = holdout - all_subjects
    if unknown:
        raise ValueError(f"holdout subjects absent from cache: {sorted(unknown)}")
    source_rows = ~np.isin(subjects, list(holdout))
    requested_source_rows = source_rows.copy()
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
    pds_dropped_source_epochs = 0
    pds_dropped_by_label = np.zeros(2, dtype=np.int64)
    if source_selection is not None:
        selection_mask = np.zeros(dataset.n_epochs, dtype=bool)
        selection_mask[source_selection.selected_indices] = True
        pds_dropped_mask = source_rows & ~selection_mask
        pds_dropped_source_epochs = int(pds_dropped_mask.sum())
        pds_dropped_by_label = np.bincount(
            labels_all[pds_dropped_mask], minlength=2
        ).astype(np.int64)
        source_rows = source_rows & selection_mask
    decision_metadata = None
    decision_loss_config = None
    p2_budget_dropped_rows = 0
    if args.candidate_loss_weight is not None:
        if args.source_domain_mass:
            raise ValueError(
                "direct decision batching cannot use epoch-level source-domain risk; "
                "define a decision-level weighting estimand first."
            )
        candidate_task_contract = dataset.provenance.get(
            "candidate_task_contract"
        )
        try:
            decision_metadata = DirectDecisionMetadata.from_timeline(
                dataset.event_timeline,
                requested_row_mask=requested_source_rows,
                retained_row_mask=source_rows,
                max_repetitions=args.candidate_loss_max_repetitions,
                source_event_contract=str(dataset.provenance.get("event_contract", "")),
                candidate_task_contract=candidate_task_contract,
            )
            decision_loss_config = DecisionLossConfig(
                candidate_vocabulary=decision_metadata.candidate_vocabulary,
                max_repetitions=args.candidate_loss_max_repetitions,
                lambda_candidate=args.candidate_loss_weight,
                temperature=args.candidate_loss_temperature,
                prefix_weight_mode=args.candidate_loss_prefix_mode,
            )
        except DirectDecisionContractError as error:
            _write_contract_failure(
                args.checkpoint,
                failure=error,
                source_cache=args.source_cache,
                source_cache_sha256=source_cache_sha256,
                source_snapshot_manifest=source_snapshot_manifest,
                source_snapshot_sha256=source_snapshot_sha256,
                holdout_subjects=holdout,
            )
            raise
        p2_rows = np.zeros(dataset.n_epochs, dtype=bool)
        p2_rows[decision_metadata.source_evidence_indices] = True
        if bool(np.any(p2_rows & ~source_rows)):
            raise RuntimeError("direct decision metadata escaped retained source rows.")
        p2_budget_dropped_rows = int(np.count_nonzero(source_rows & ~p2_rows))
        source_rows = p2_rows
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
    splitbn_source_weights = parse_splitbn_source_weights(
        args.splitbn_source_weight
    )
    if args.splitbn_target_buffer_source == "source_weighted":
        if not splitbn_source_weights:
            raise ValueError(
                "source_weighted Split-BN requires --splitbn-source-weight for every domain."
            )
    elif splitbn_source_weights:
        raise ValueError(
            "--splitbn-source-weight is only valid with source_weighted Split-BN."
        )
    source_domains = None
    source_vocabulary: tuple[str, ...] | None = None
    selection_domain = args.selection_domain.strip() or None
    if dataset.provenance.get("source_domain_axis") is not None:
        source_vocabulary = source_domain_vocabulary(dataset)
        source_domains = source_domain_ids(dataset)[source_rows]
        observed_domains = set(source_domains.tolist())
        if len(observed_domains) > 1:
            if selection_domain is None and domain_mass:
                raise ValueError(
                    "weighted multi-domain source training requires an explicit selection-domain."
                )
            if domain_mass and set(domain_mass) != observed_domains:
                raise ValueError(
                    "source-domain-mass keys must equal retained source domains: "
                    f"{sorted(observed_domains)}."
                )
            if (
                selection_domain is not None
                and selection_domain not in observed_domains
            ):
                raise ValueError("selection-domain is absent from retained source rows.")
        elif domain_mass or selection_domain is not None:
            raise ValueError("domain-risk options require a genuinely multi-domain source cache.")
    elif domain_mass or selection_domain is not None:
        raise ValueError("domain-risk options require an explicit source-domain axis.")
    if splitbn_source_weights and (
        source_vocabulary is None
        or set(splitbn_source_weights) != set(source_vocabulary)
    ):
        raise ValueError(
            "splitbn-source-weight keys must equal the cache-declared source domains."
        )
    indexed_domain_requested = (
        args.splitbn_target_buffer_source is not None
        or args.source_residual_drop_p is not None
        or (
            args.feature_transfer_invariant_bottleneck > 0
            and source_vocabulary is not None
            and len(source_vocabulary) >= 2
        )
    )
    if indexed_domain_requested and (
        source_vocabulary is None or len(source_vocabulary) < 2
    ):
        raise ValueError(
            "Split-BN/source residual training requires at least two declared source domains."
        )
    if (
        indexed_domain_requested
        and source_domains is not None
        and set(source_domains.tolist()) != set(source_vocabulary or ())
    ):
        raise ValueError(
            "holdout/QC rows removed an entire declared source domain; indexed domain "
            "mechanisms cannot silently change their vocabulary."
        )

    feature_transfer = None
    if args.feature_transfer_invariant_bottleneck > 0:
        feature_transfer = FeatureTransferConfig(
            invariant_bottleneck_channels=args.feature_transfer_invariant_bottleneck,
            private_channels=args.feature_transfer_private_channels,
            target_bottleneck_channels=args.feature_transfer_target_bottleneck,
            target_kernel_size=args.feature_transfer_target_kernel,
            max_invariant_residual=args.feature_transfer_max_invariant_residual,
            max_target_residual=args.feature_transfer_max_target_residual,
            private_domain_count=(
                len(source_vocabulary)
                if source_vocabulary is not None and len(source_vocabulary) >= 2
                else 0
            ),
            source_domain_count=(
                len(source_vocabulary)
                if args.source_residual_drop_p is not None
                and source_vocabulary is not None
                else 0
            ),
            source_drop_p=(
                args.source_residual_drop_p
                if args.source_residual_drop_p is not None
                else 0.3
            ),
            source_drop_mode=args.source_residual_drop_mode,
        )
    elif (
        args.private_domain_weight > 0.0
        or args.orthogonality_weight > 0.0
        or args.source_residual_drop_p is not None
    ):
        raise ValueError(
            "private-domain, orthogonality, and source-residual mechanisms require "
            "--feature-transfer-invariant-bottleneck > 0."
        )
    if args.private_domain_weight > 0.0 and (
        source_domains is None or len(set(source_domains.tolist())) < 2
    ):
        raise ValueError(
            "--private-domain-weight requires a genuinely multi-domain source cache "
            "with an explicit source-domain axis."
        )
    if args.feature_alignment_weight > 0.0 and (
        source_domains is None or len(set(source_domains.tolist())) < 2
    ):
        raise ValueError(
            "--feature-alignment-weight requires a genuinely multi-domain source "
            "cache with an explicit source-domain axis."
        )

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
        feature_transfer=feature_transfer,
        source_domain_vocabulary=source_vocabulary,
        splitbn_domain_count=(
            len(source_vocabulary)
            if args.splitbn_target_buffer_source is not None
            and source_vocabulary is not None
            else None
        ),
        splitbn_target_buffer_source=args.splitbn_target_buffer_source,
        splitbn_source_weights=splitbn_source_weights or None,
    )

    X = np.ascontiguousarray(dataset.X[source_rows])
    y = np.asarray(dataset.y, dtype=np.int64)[source_rows]
    # ``input_statistics_scope`` declares *which rows* normalization statistics
    # may be computed from; it must stay a pure control declaration. Realized
    # counts (rows/unique_subjects) vary with every partition's holdout, so
    # embedding them here would make the arm procedure disagree with itself
    # across partitions and get rejected by the manifest arm check.
    source_input_stats_scope = "all_source_rows"
    risk_fit_kwargs = {}
    if source_domains is not None:
        risk_fit_kwargs["source_domain_ids"] = source_domains
    if selection_domain is not None:
        risk_fit_kwargs["selection_domain"] = selection_domain
    if domain_mass:
        risk_fit_kwargs.update(
            {
                "source_domain_mass": domain_mass,
                "risk_unit_ids": src_subjects,
                "risk_within_domain_unit": args.risk_within_domain_unit,
            }
        )
    decision_fit_kwargs = (
        {}
        if decision_metadata is None or decision_loss_config is None
        else {
            "decision_metadata": decision_metadata,
            "decision_loss_config": decision_loss_config,
        }
    )
    started = time.perf_counter()
    try:
        baseline.fit(
            X,
            y,
            group_ids=src_subjects,
            feature_alignment_weight=args.feature_alignment_weight,
            feature_alignment_mode=args.feature_alignment_mode,
            feature_alignment_min_cell_count=args.feature_alignment_min_cell_count,
            feature_alignment_inactive_policy=args.feature_alignment_inactive_policy,

            feature_private_domain_weight=args.private_domain_weight,
            feature_orthogonality_weight=args.orthogonality_weight,
            **risk_fit_kwargs,
            **decision_fit_kwargs,
        )
    except (SourceResidualContractError, DirectDecisionContractError) as error:
        _write_contract_failure(
            args.checkpoint,
            failure=error,
            source_cache=args.source_cache,
            source_cache_sha256=source_cache_sha256,
            source_snapshot_manifest=source_snapshot_manifest,
            source_snapshot_sha256=source_snapshot_sha256,
            holdout_subjects=holdout,
        )
        raise
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
        feature_transfer=feature_transfer,
        source_domain_vocabulary=source_vocabulary,
        splitbn_domain_count=(
            len(source_vocabulary)
            if args.splitbn_target_buffer_source is not None
            and source_vocabulary is not None
            else None
        ),
        splitbn_target_buffer_source=args.splitbn_target_buffer_source,
        splitbn_source_weights=splitbn_source_weights or None,
    )
    refit_started = time.perf_counter()
    try:
        baseline.fit(
            X,
            y,
            group_ids=None,
            feature_alignment_weight=args.feature_alignment_weight,
            feature_alignment_mode=args.feature_alignment_mode,
            feature_alignment_min_cell_count=args.feature_alignment_min_cell_count,
            feature_alignment_inactive_policy=args.feature_alignment_inactive_policy,

            feature_private_domain_weight=args.private_domain_weight,
            feature_orthogonality_weight=args.orthogonality_weight,
            **risk_fit_kwargs,
            **decision_fit_kwargs,
        )
    except (SourceResidualContractError, DirectDecisionContractError) as error:
        _write_contract_failure(
            args.checkpoint,
            failure=error,
            source_cache=args.source_cache,
            source_cache_sha256=source_cache_sha256,
            source_snapshot_manifest=source_snapshot_manifest,
            source_snapshot_sha256=source_snapshot_sha256,
            holdout_subjects=holdout,
        )
        raise
    refit_sec = time.perf_counter() - refit_started
    model = baseline.model_
    architecture_record = baseline.architecture_record()
    if semantic_sha256(selection_baseline.architecture_record()) != semantic_sha256(
        architecture_record
    ):
        raise RuntimeError("selection and refit models disagree on architecture contract.")
    decision_loss_record = (
        None if decision_loss_config is None else decision_loss_config.record()
    )
    decision_metadata_record = (
        None
        if decision_metadata is None
        else decision_metadata.record(include_rows=False)
    )
    def _durable_decision_ledger(
        history_ledger: object,
    ) -> dict[str, object] | None:
        """Keep only durable P2 evidence for the checkpoint payload.

        The trainer emits a ``status="disabled"`` ledger when the P2 decision
        loss never trained; that is a process diagnostic, not evidence, and the
        checkpoint validator correctly requires a decision-loss contract for
        any decision payload it carries. Passing the disabled ledger through
        would therefore render every P2-free checkpoint unreadable. An active
        ledger (valid or otherwise) is preserved untouched so the validator
        keeps enforcing its full contract on it.
        """

        if not isinstance(history_ledger, Mapping) or history_ledger.get("status") == "disabled":
            return None
        return history_ledger

    decision_training_diagnostics = {
        "selection": _durable_decision_ledger(
            selection_baseline.last_history.get("decision_loss")
        ),
        "refit": _durable_decision_ledger(baseline.last_history.get("decision_loss")),
    }
    source_domain_axis = baseline.source_domain_axis_record()
    source_domain_axis_digest = architecture_record.get(
        "source_domain_axis_digest"
    )
    if source_domain_axis_digest is None:
        source_domain_axis = None
    elif source_domain_axis is None or source_domain_axis_digest != semantic_sha256(
        source_domain_axis
    ):
        raise RuntimeError("model source-domain axis disagrees with architecture digest.")

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
    source_selection_contract = (
        None if source_selection_record is None else dict(source_selection_record)
    )
    training_contract = TrainingRunContract(
        source_cache_sha256=source_cache_sha256,
        source_identity_digest=dataset.identity_table.digest(),
        source_snapshot_sha256=source_snapshot_sha256,
        architecture=architecture_record,
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
            "effective_batch_contract": {
                "requested_batch_size": int(args.batch_size),
                "selection_batch_size": selection_baseline.last_runtime.get(
                    "batch_size"
                ),
                "refit_batch_size": baseline.last_runtime.get("batch_size"),
                "selection_oom_retries": selection_baseline.last_runtime.get(
                    "oom_retries"
                ),
                "refit_oom_retries": baseline.last_runtime.get("oom_retries"),
                "matched_procedure_policy": "effective_batch_and_oom_must_match",
            },
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
            "source_risk_contract": {
                "domain_mass": domain_mass or None,
                "within_domain_unit": (
                    args.risk_within_domain_unit if domain_mass else None
                ),
                "selection_domain": selection_domain,
            },
            "source_domain_axis": source_domain_axis,
            "feature_transfer": feature_transfer.record() if feature_transfer is not None else None,
            "decision_loss": decision_loss_record,
            "decision_metadata": decision_metadata_record,
            "decision_loss_diagnostics": decision_training_diagnostics,
            "feature_alignment_contract": {
                "weight": float(args.feature_alignment_weight),
                "mode": args.feature_alignment_mode,
                "loss": (
                    (
                        "class_conditional_rbf_mmd"
                        if args.feature_alignment_mode == "conditional_rbf_mmd"
                        else "class_conditional_mean_alignment"
                    )
                    if args.feature_alignment_weight > 0.0
                    else None
                ),
                "min_cell_count": int(args.feature_alignment_min_cell_count),
                "inactive_policy": args.feature_alignment_inactive_policy,

                "private_domain_weight": float(args.private_domain_weight),
                "orthogonality_weight": float(args.orthogonality_weight),
            },
            "source_selection_contract": source_selection_contract,
            "feature_alignment_diagnostics": {
                "selection_epoch_mean_squared_distance": (
                    selection_baseline.last_history.get("feature_alignment") or {}
                ).get("epoch_mean_squared_distance"),
                "selection_epoch_mmd_squared": (
                    selection_baseline.last_history.get("feature_alignment") or {}
                ).get("epoch_mmd_squared"),
                "refit_epoch_mean_squared_distance": (
                    baseline.last_history.get("feature_alignment") or {}
                ).get("epoch_mean_squared_distance"),
                "refit_epoch_mmd_squared": (
                    baseline.last_history.get("feature_alignment") or {}
                ).get("epoch_mmd_squared"),
                "selection_epoch_private_domain_loss": (
                    selection_baseline.last_history.get("feature_alignment") or {}
                ).get("epoch_private_domain_loss"),
                "refit_epoch_private_domain_loss": (
                    baseline.last_history.get("feature_alignment") or {}
                ).get("epoch_private_domain_loss"),
                "selection_epoch_orthogonality_loss": (
                    selection_baseline.last_history.get("feature_alignment") or {}
                ).get("epoch_orthogonality_loss"),
                "refit_epoch_orthogonality_loss": (
                    baseline.last_history.get("feature_alignment") or {}
                ).get("epoch_orthogonality_loss"),
            },
            "source_residual_diagnostics": {
                "selection": selection_baseline.last_history.get("source_residual"),
                "refit": baseline.last_history.get("source_residual"),
            },
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
            "source_domain_vocabulary": (
                None if source_vocabulary is None else list(source_vocabulary)
            ),
            "splitbn_target_buffer_source": args.splitbn_target_buffer_source,
            "splitbn_source_weights": splitbn_source_weights or None,
            "decision_loss": decision_loss_record,
            "feature_transfer": feature_transfer.record() if feature_transfer is not None else None,
            "feature_alignment_weight": float(args.feature_alignment_weight),
            "feature_alignment_min_cell_count": int(args.feature_alignment_min_cell_count),
            "feature_alignment_inactive_policy": args.feature_alignment_inactive_policy,

            "feature_alignment_mode": args.feature_alignment_mode,
            "private_domain_weight": float(args.private_domain_weight),
            "orthogonality_weight": float(args.orthogonality_weight),
            "source_selection_contract": source_selection_contract,
            "training": "N2P3NetBaseline supervised (LOSO-identical path)",
        },
        "architecture": architecture_record,
        **(
            {}
            if source_domain_axis is None
            else {
                "source_domain_axis": source_domain_axis,
                "source_domain_axis_digest": source_domain_axis_digest,
            }
        ),
        "decision_loss": decision_loss_record,
        "decision_metadata": decision_metadata_record,
        "decision_metadata_digest": (
            None
            if decision_metadata_record is None
            else semantic_sha256(decision_metadata_record)
        ),
        "decision_loss_selection": decision_training_diagnostics["selection"],
        "decision_loss_refit": decision_training_diagnostics["refit"],
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
        "source_residual_selection": selection_baseline.last_history.get(
            "source_residual"
        ),
        "source_residual_refit": baseline.last_history.get("source_residual"),
        "p2_budget_dropped_source_epochs": p2_budget_dropped_rows,
        "source_selection_contract": source_selection_contract,
        "tsd_dropped_source_epochs": pds_dropped_source_epochs,
        "tsd_dropped_source_epochs_by_label": pds_dropped_by_label.tolist(),
        "pds_dropped_source_epochs": pds_dropped_source_epochs,
        "pds_dropped_source_epochs_by_label": pds_dropped_by_label.tolist(),
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
                "tsd_dropped_source_epochs": pds_dropped_source_epochs,
                "pds_dropped_source_epochs_legacy": pds_dropped_source_epochs,
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
