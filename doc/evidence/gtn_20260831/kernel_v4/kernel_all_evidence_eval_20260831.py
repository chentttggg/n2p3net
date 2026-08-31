"""Frozen all-evidence evaluator and paired analysis for the GTN kernel ablation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.contract import (  # noqa: E402
    GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT,
    assert_p300_input_contract,
)
from data.epochs import load_epoch_dataset, read_epoch_cache_attestation  # noqa: E402
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    checkpoint_input_stats,
    checkpoint_scores_to_llr,
    load_checkpoint_payload,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
)

CACHE_SHA256 = "3b4c6b0439dff13a8a47c28be8722c42d2585cea74956bf4b4675e30d5922a9a"
BASE_MANIFEST_SHA256 = "cc169523c02f9623be4b1ce24fc7f7e892e0dcdda40cf145f2988940f9a21d1f"
V3_MANIFEST_SHA256 = "e78668f0e2331824de2827f7d6d02be444b0ac0754ee47b443baf15a0c4c8124"
PARENT_MANIFEST_SHA256 = "52d3cfcf3a534d4062e99b0bff94926c388b3830fa8945a01b2c468725870744"
V4_SCHEMA = "n2p3_temporal_kernel_ablation_amendment/4"
KERNELS = (33, 35, 65)
SEEDS = (20260828, 20260829, 20260830)
BLOCKS = (0, 1, 2, 3)
REQUESTED_N = 245
BOOTSTRAP_ITERATIONS = 100_000
ANALYSIS_SEED = 20260831
PRACTICAL_DELTA = 0.02
MIN_POSITIVE_SEEDS = 2
SOURCE_QC_PTP_UV = 100.0
EXPECTED_BLOCK_SIZES = (62, 61, 61, 61)
PRIMARY_ENDPOINT = "requested-cohort operational hit@all_balanced"
CONDITIONAL_INFERENCE_SCOPE = (
    "subject-level paired inference conditional on the frozen three seeds, four target-excluded "
    "blocks, checkpoints, and GTN development cohort; seed stability is a separate descriptive gate"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_subjects(path: str | Path) -> list[str]:
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, list) or not decoded or not all(isinstance(x, str) for x in decoded):
        raise ValueError("target subject manifest must be a non-empty JSON string list")
    if len(set(decoded)) != len(decoded):
        raise ValueError("target subject manifest contains duplicate subjects")
    return decoded


def read_json_mapping(path: str | Path, *, label: str) -> dict[str, object]:
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON mapping")
    return decoded


def validate_v4_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], str]:
    manifest_path = Path(path)
    manifest_sha = sha256_file(manifest_path)
    if expected_sha256 and manifest_sha != expected_sha256:
        raise ValueError("frozen analysis manifest SHA-256 mismatch")
    manifest = read_json_mapping(manifest_path, label="v4 analysis manifest")
    if manifest.get("schema") != V4_SCHEMA:
        raise ValueError(f"analysis requires {V4_SCHEMA}, got {manifest.get('schema')!r}")
    chain = manifest.get("amendment_chain")
    if not isinstance(chain, list) or len(chain) != 2:
        raise ValueError("v4 manifest must bind exactly the v2 and v3 predecessor manifests")
    expected_chain = (
        ("n2p3_temporal_kernel_ablation/2", BASE_MANIFEST_SHA256),
        ("n2p3_temporal_kernel_ablation_amendment/3", V3_MANIFEST_SHA256),
    )
    actual_chain = tuple(
        (entry.get("schema"), entry.get("sha256"))
        for entry in chain
        if isinstance(entry, Mapping)
    )
    if actual_chain != expected_chain:
        raise ValueError("v4 predecessor manifest chain does not match the frozen v2/v3 hashes")
    data_lock = manifest.get("data_lock")
    if not isinstance(data_lock, Mapping):
        raise ValueError("v4 manifest lacks data_lock")
    cache = data_lock.get("cache")
    parent = data_lock.get("parent_block_manifest")
    if not isinstance(cache, Mapping) or cache.get("sha256") != CACHE_SHA256:
        raise ValueError("v4 manifest is not bound to the registered cache")
    if not isinstance(parent, Mapping) or parent.get("sha256") != PARENT_MANIFEST_SHA256:
        raise ValueError("v4 manifest is not bound to the registered parent block manifest")
    endpoint = manifest.get("endpoint_contract")
    if not isinstance(endpoint, Mapping) or endpoint.get("primary") != PRIMARY_ENDPOINT:
        raise ValueError("v4 manifest changed the frozen v3 primary endpoint")
    if endpoint.get("inherits_v3_sha256") != V3_MANIFEST_SHA256:
        raise ValueError("v4 endpoint contract does not inherit the frozen v3 amendment")
    return manifest, manifest_sha


def parent_block_subjects(
    parent_manifest_path: str | Path,
    v4_manifest: Mapping[str, object],
) -> tuple[dict[int, list[str]], str]:
    parent_sha = sha256_file(parent_manifest_path)
    data_lock = v4_manifest["data_lock"]
    assert isinstance(data_lock, Mapping)
    parent_lock = data_lock["parent_block_manifest"]
    assert isinstance(parent_lock, Mapping)
    if parent_sha != parent_lock.get("sha256"):
        raise ValueError("parent block manifest SHA-256 mismatch")
    parent = read_json_mapping(parent_manifest_path, label="parent block manifest")
    blocks = parent.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("parent block manifest lacks blocks")
    parsed: dict[int, list[str]] = {}
    for entry in blocks:
        if not isinstance(entry, Mapping):
            raise ValueError("parent block entry must be a mapping")
        block = int(entry.get("block", -1))
        subjects = entry.get("subjects")
        if block in parsed or block not in BLOCKS:
            raise ValueError(f"invalid or duplicate parent block {block}")
        if not isinstance(subjects, list) or not all(isinstance(x, str) for x in subjects):
            raise ValueError(f"parent block {block} subjects must be a string list")
        if len(subjects) != int(entry.get("n", -1)) or len(set(subjects)) != len(subjects):
            raise ValueError(f"parent block {block} has an invalid subject count")
        parsed[block] = list(subjects)
    if tuple(sorted(parsed)) != BLOCKS:
        raise ValueError("parent block manifest must contain blocks 0..3")
    if tuple(len(parsed[block]) for block in BLOCKS) != EXPECTED_BLOCK_SIZES:
        raise ValueError("parent block sizes differ from the frozen 62/61/61/61 contract")
    union = [subject for block in BLOCKS for subject in parsed[block]]
    if len(union) != REQUESTED_N or len(set(union)) != REQUESTED_N:
        raise ValueError("parent blocks must be disjoint and cover exactly 245 subjects")
    return parsed, parent_sha


def validate_target_block(
    target_subjects_file: str | Path,
    *,
    block: int,
    expected_blocks: Mapping[int, list[str]],
    v4_manifest: Mapping[str, object],
) -> tuple[list[str], str]:
    target_subjects = read_subjects(target_subjects_file)
    if target_subjects != expected_blocks[block]:
        raise ValueError(f"target subject file does not equal frozen parent block {block}")
    data_lock = v4_manifest["data_lock"]
    assert isinstance(data_lock, Mapping)
    block_locks = data_lock.get("block_manifests")
    if not isinstance(block_locks, list):
        raise ValueError("v4 manifest lacks block_manifests")
    lock = next(
        (entry for entry in block_locks if isinstance(entry, Mapping) and entry.get("block") == block),
        None,
    )
    target_sha = sha256_file(target_subjects_file)
    if lock is None or lock.get("sha256") != target_sha or int(lock.get("n", -1)) != len(target_subjects):
        raise ValueError(f"target block {block} file is not bound by the v4 manifest")
    return target_subjects, target_sha


def unique_prediction(scores: dict[str, float], vocabulary: tuple[str, ...]) -> str | None:
    values = np.asarray([scores.get(candidate, -np.inf) for candidate in vocabulary], dtype=float)
    if not np.isfinite(values).all():
        return None
    maximum = float(np.max(values))
    tied = np.flatnonzero(np.isclose(values, maximum, rtol=1e-12, atol=1e-12))
    return vocabulary[int(tied[0])] if len(tied) == 1 else None


def expected_architecture_record(dataset: object, kernel: int) -> dict[str, object]:
    architecture = N2P3ArchitectureConfig(temporal_kernel_size=int(kernel))
    model = N2P3Net(
        int(dataset.n_channels),
        n_times=int(dataset.n_times),
        sfreq=float(dataset.preprocessing.sfreq),
        tmin_s=float(dataset.preprocessing.tmin_ms) / 1000.0,
        pooling_mode="full_unfold",
        **architecture.model_kwargs(),
    )
    return model.architecture_record()


def validate_architecture_record(
    architecture: object,
    *,
    dataset: object,
    kernel: int,
) -> None:
    if not isinstance(architecture, Mapping):
        raise ValueError("checkpoint architecture must be a mapping")
    expected = expected_architecture_record(dataset, kernel)
    if dict(architecture) != expected:
        changed = sorted(
            key
            for key in set(architecture) | set(expected)
            if architecture.get(key) != expected.get(key)
        )
        raise ValueError(
            "checkpoint architecture differs outside the registered kernel arm: "
            f"{changed}"
        )


def _effective_trial_mask(dataset: object) -> np.ndarray:
    static = np.asarray(dataset.channel_mask, dtype=bool)
    if dataset.trial_channel_mask is None:
        return np.broadcast_to(static, dataset.X.shape[:2])
    return np.asarray(dataset.trial_channel_mask, dtype=bool) & static[None, :]


def _masked_input_stats(X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = mask[:, :, None]
    counts = mask.sum(axis=0, dtype=np.float64)[None, :, None] * X.shape[2]
    denominator = np.maximum(counts, 1.0)
    sums = np.sum(
        X,
        axis=(0, 2),
        dtype=np.float64,
        keepdims=True,
        where=observed,
    )
    mean = np.divide(sums, denominator, out=np.zeros_like(sums), where=counts > 0.0)
    variance = np.var(
        X,
        axis=(0, 2),
        dtype=np.float64,
        keepdims=True,
        where=observed,
        mean=mean,
    )
    std = np.where(counts > 0.0, np.sqrt(np.maximum(variance, 0.0)) + 1e-6, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def validate_checkpoint_payload_contract(
    payload: Mapping[str, object],
    dataset: object,
    *,
    target_subjects: list[str],
    cache_sha256: str,
    kernel: int,
    seed: int,
) -> dict[str, object]:
    required = {
        "trunk_state_dict",
        "input_mean",
        "input_std",
        "input_preprocessing",
        "input_channel_names",
        "input_source_reference",
        "source_cache_sha256",
        "classifier_trained",
        "training_subject_keys",
        "training_cache_subject_keys",
        "training_subjects",
        "source_subjects",
        "source_dataset_name",
        "holdout_subjects",
        "source_full_refit",
        "source_refit_epochs",
        "source_calibration",
        "training_pos_weight",
        "training_prior",
        "n_source_epochs_used",
        "qc_ptp_uv",
        "qc_dropped_source_epochs",
        "source_label_counts_before_qc",
        "qc_dropped_source_epochs_by_label",
        "source_label_retention_by_label",
        "best_epoch",
        "config",
        "architecture",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint lacks required v4 contract fields: {missing}")

    expected_config = {
        "pooling_mode": "full_unfold",
        "temporal_kernel_size": int(kernel),
        "epochs": 100,
        "batch_size": 512,
        "seed": int(seed),
        "training": "N2P3NetBaseline supervised (LOSO-identical path)",
    }
    config = payload.get("config")
    if not isinstance(config, Mapping) or dict(config) != expected_config:
        raise ValueError("checkpoint config does not exactly match the registered training arm")
    validate_architecture_record(payload.get("architecture"), dataset=dataset, kernel=kernel)

    if float(payload.get("qc_ptp_uv", math.nan)) != SOURCE_QC_PTP_UV:
        raise ValueError("checkpoint source QC is not exactly 100 uV")
    if payload.get("classifier_trained") is not True or payload.get("source_full_refit") is not True:
        raise ValueError("checkpoint classifier/full-source-refit contract is incomplete")
    if payload.get("source_cache_sha256") != cache_sha256 or cache_sha256 != CACHE_SHA256:
        raise ValueError("checkpoint is bound to the wrong cache")
    if payload.get("source_dataset_name") != dataset.name:
        raise ValueError("checkpoint source dataset name differs from the cache")

    cache_subjects = sorted(np.unique(np.asarray(dataset.subject_ids).astype(str)).tolist())
    cache_subject_set = set(cache_subjects)
    target_set = set(target_subjects)
    if not target_set <= cache_subject_set:
        raise ValueError("target block contains subjects absent from the cache")
    expected_training = sorted(cache_subject_set - target_set)
    if list(payload["holdout_subjects"]) != sorted(target_subjects):
        raise ValueError("checkpoint holdout subjects do not equal the frozen target block")
    if list(payload["source_subjects"]) != cache_subjects:
        raise ValueError("checkpoint source_subjects do not equal the cache subject set")
    if list(payload["training_subjects"]) != expected_training:
        raise ValueError("checkpoint training_subjects are not the exact holdout complement")
    expected_subject_keys = [f"{dataset.name}\0{subject}" for subject in expected_training]
    expected_cache_keys = [f"{cache_sha256}\0{subject}" for subject in expected_training]
    if list(payload["training_subject_keys"]) != expected_subject_keys:
        raise ValueError("checkpoint training_subject_keys are not the exact holdout complement")
    if list(payload["training_cache_subject_keys"]) != expected_cache_keys:
        raise ValueError("checkpoint training_cache_subject_keys are not the exact holdout complement")

    subjects = np.asarray(dataset.subject_ids).astype(str)
    labels = np.asarray(dataset.y, dtype=np.int64)
    source_before_qc = ~np.isin(subjects, target_subjects)
    X_all = np.asarray(dataset.X, dtype=np.float32)
    ptp = X_all.max(axis=2) - X_all.min(axis=2)
    bad = (ptp >= SOURCE_QC_PTP_UV * 1e-6).any(axis=1)
    source_rows = source_before_qc & ~bad
    counts_before = np.bincount(labels[source_before_qc], minlength=2).astype(int)
    dropped_by_label = np.bincount(labels[source_before_qc & bad], minlength=2).astype(int)
    retention = (counts_before - dropped_by_label) / np.maximum(counts_before, 1)
    if int(payload["n_source_epochs_used"]) != int(source_rows.sum()):
        raise ValueError("checkpoint source row count differs from the frozen holdout+QC complement")
    if int(payload["qc_dropped_source_epochs"]) != int((source_before_qc & bad).sum()):
        raise ValueError("checkpoint QC drop count differs from the cache")
    if list(payload["source_label_counts_before_qc"]) != counts_before.tolist():
        raise ValueError("checkpoint source label counts before QC differ from the cache")
    if list(payload["qc_dropped_source_epochs_by_label"]) != dropped_by_label.tolist():
        raise ValueError("checkpoint label-stratified QC drops differ from the cache")
    if not np.allclose(
        np.asarray(payload["source_label_retention_by_label"], dtype=float),
        retention,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("checkpoint label retention differs from the cache")

    best_epoch = payload.get("best_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, (int, np.integer)):
        raise ValueError("checkpoint best_epoch must be an integer")
    if not 0 <= int(best_epoch) < 100:
        raise ValueError("checkpoint best_epoch lies outside the 100-epoch selection run")
    if int(payload["source_refit_epochs"]) != int(best_epoch) + 1:
        raise ValueError("checkpoint source refit did not use best_epoch+1")

    expected_prior = float(labels[source_rows].mean())
    if not np.isclose(float(payload["training_pos_weight"]), 8.0, rtol=0.0, atol=1e-12):
        raise ValueError("checkpoint training_pos_weight differs from DeepConfig")
    if not np.isclose(float(payload["training_prior"]), expected_prior, rtol=0.0, atol=1e-12):
        raise ValueError("checkpoint training prior differs from the refit rows")
    calibration = payload.get("source_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("checkpoint source_calibration must be a mapping")
    expected_calibration = {
        "pos_weight": 8.0,
        "train_prior": expected_prior,
        "temperature": 1.0,
        "source": "source_full_refit_weighted_ce_analytic",
    }
    for key, expected_value in expected_calibration.items():
        actual = calibration.get(key)
        if isinstance(expected_value, float):
            if not np.isclose(float(actual), expected_value, rtol=0.0, atol=1e-12):
                raise ValueError(f"checkpoint calibration {key} differs from the refit contract")
        elif actual != expected_value:
            raise ValueError(f"checkpoint calibration {key} differs from the refit contract")

    trial_mask = _effective_trial_mask(dataset)[source_rows]
    expected_mean, expected_std = _masked_input_stats(X_all[source_rows], trial_mask)
    declared_stats = checkpoint_input_stats(payload, n_channels=dataset.n_channels, required=True)
    assert declared_stats is not None
    declared_mean, declared_std = declared_stats
    if not np.allclose(declared_mean, expected_mean, rtol=1e-6, atol=1e-8):
        raise ValueError("checkpoint input_mean was not fit on the exact source complement")
    if not np.allclose(declared_std, expected_std, rtol=1e-6, atol=1e-8):
        raise ValueError("checkpoint input_std was not fit on the exact source complement")
    return {
        "cache_subjects": cache_subjects,
        "training_subjects": expected_training,
        "n_source_epochs_used": int(source_rows.sum()),
        "best_epoch": int(best_epoch),
        "source_refit_epochs": int(payload["source_refit_epochs"]),
        "architecture": expected_architecture_record(dataset, kernel),
    }


def evidence_cost(
    selected_events: list[int],
    group_event_rows: np.ndarray,
    session_first_onset: float,
    *,
    scheduled_onsets: np.ndarray,
    scheduled_available: np.ndarray,
) -> dict[str, float | int | None]:
    if not selected_events:
        return {
            "available_trials": 0,
            "scheduled_stimuli_through_decision": 0,
            "elapsed_seconds": None,
            "decision_evidence_available_time_s": None,
        }
    available_time = float(max(scheduled_available[event] for event in selected_events))
    return {
        "available_trials": len(selected_events),
        "scheduled_stimuli_through_decision": int(
            np.count_nonzero(scheduled_onsets[group_event_rows] <= available_time)
        ),
        "elapsed_seconds": float(max(0.0, available_time - session_first_onset)),
        "decision_evidence_available_time_s": available_time,
    }


def summarize_costs(costs: list[Mapping[str, object]]) -> dict[str, float | None]:
    def mean(name: str) -> float | None:
        values = [float(cost[name]) for cost in costs if cost.get(name) is not None]
        return float(np.mean(values)) if values else None

    return {
        "mean_available_trials": mean("available_trials"),
        "mean_scheduled_stimuli": mean("scheduled_stimuli_through_decision"),
        "mean_elapsed_seconds": mean("elapsed_seconds"),
    }


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    eligible = [record for record in records if int(record["r_balanced_all"]) > 0]
    balanced_hits = int(sum(bool(record["balanced_all"]["hit"]) for record in records))
    raw_hits = int(sum(bool(record["raw_all"]["hit"]) for record in records))
    aucs = [float(record["binary_auc"]) for record in records if record["binary_auc"] is not None]
    max_r = max(int(record["r_balanced_all"]) for record in records)
    curve = {}
    for repetition in range(1, max_r + 1):
        at_r = [record for record in records if int(record["r_balanced_all"]) >= repetition]
        successes = int(sum(bool(record["hit_by_r"].get(str(repetition), False)) for record in at_r))
        costs = [record["cost_by_r"][str(repetition)] for record in at_r]
        curve[str(repetition)] = {
            "eligible": len(at_r),
            "coverage": len(at_r) / len(records),
            "hits": successes,
            "conditional_hit": successes / len(at_r) if at_r else None,
            "operational_hit": successes / len(records),
            "mean_available_trials": float(np.mean([cost["available_trials"] for cost in costs])) if costs else None,
            "mean_scheduled_stimuli": float(np.mean([cost["scheduled_stimuli_through_decision"] for cost in costs])) if costs else None,
            "mean_elapsed_seconds": float(np.mean([cost["elapsed_seconds"] for cost in costs])) if costs else None,
        }
    return {
        "requested": len(records),
        "eligible": len(eligible),
        "coverage": len(eligible) / len(records),
        "balanced_all_hits": balanced_hits,
        "balanced_all_conditional_hit": balanced_hits / len(eligible) if eligible else None,
        "balanced_all_operational_hit": balanced_hits / len(records),
        "raw_all_hits": raw_hits,
        "raw_all_conditional_hit": raw_hits / len(eligible) if eligible else None,
        "raw_all_operational_hit": raw_hits / len(records),
        "balanced_all_cost": summarize_costs(
            [record["balanced_all"]["cost"] for record in records]
        ),
        "raw_all_cost": summarize_costs([record["raw_all"]["cost"] for record in records]),
        "binary_auc_subject_macro": float(np.mean(aucs)) if aucs else None,
        "r_s_distribution": {str(k): int(v) for k, v in sorted(Counter(int(record["r_balanced_all"]) for record in records).items())},
        "hit_curve": curve,
    }


def prepare_checkpoint_context(
    args: argparse.Namespace,
) -> tuple[
    object,
    dict[str, object],
    object,
    tuple[np.ndarray, np.ndarray],
    list[str],
    str,
    str,
    str,
    dict[str, object],
]:
    manifest_path = Path(args.manifest)
    manifest, manifest_sha = validate_v4_manifest(
        manifest_path,
        expected_sha256=args.manifest_sha256,
    )
    expected_blocks, parent_sha = parent_block_subjects(args.parent_manifest, manifest)
    target_subjects, target_subjects_sha = validate_target_block(
        args.target_subjects_file,
        block=args.block,
        expected_blocks=expected_blocks,
        v4_manifest=manifest,
    )
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    assert_p300_input_contract(dataset.preprocessing, GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT)
    cache_sha = str(read_epoch_cache_attestation(args.dataset_cache)["sha256"])
    if cache_sha != CACHE_SHA256:
        raise ValueError(f"wrong cache SHA-256: {cache_sha}")
    cache_subjects = sorted(np.unique(np.asarray(dataset.subject_ids).astype(str)).tolist())
    frozen_subjects = sorted(subject for block in BLOCKS for subject in expected_blocks[block])
    if cache_subjects != frozen_subjects:
        raise ValueError("frozen parent blocks are not the exact cache subject set")
    payload = load_checkpoint_payload(args.checkpoint)
    checkpoint_contract = validate_checkpoint_payload_contract(
        payload,
        dataset,
        target_subjects=target_subjects,
        cache_sha256=cache_sha,
        kernel=args.kernel,
        seed=args.seed,
    )
    trunk, payload = load_n2p3_trunk_checkpoint(
        payload,
        dataset,
        target_subject=target_subjects[0],
        target_cache_sha256=cache_sha,
    )
    input_stats = checkpoint_input_stats(payload, n_channels=dataset.n_channels, required=True)
    assert input_stats is not None
    return (
        dataset,
        payload,
        trunk,
        input_stats,
        target_subjects,
        manifest_sha,
        parent_sha,
        target_subjects_sha,
        checkpoint_contract,
    )


def validate_checkpoint(args: argparse.Namespace) -> None:
    (
        _,
        _,
        _,
        _,
        target_subjects,
        manifest_sha,
        parent_sha,
        target_subjects_sha,
        checkpoint_contract,
    ) = prepare_checkpoint_context(args)
    print(
        json.dumps(
            {
                "status": "valid_v4_checkpoint",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "kernel": args.kernel,
                "seed": args.seed,
                "block": args.block,
                "target_subjects": len(target_subjects),
                "manifest_sha256": manifest_sha,
                "parent_manifest_sha256": parent_sha,
                "target_subjects_sha256": target_subjects_sha,
                "checkpoint_contract": checkpoint_contract,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def evaluate(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    (
        dataset,
        payload,
        trunk,
        input_stats,
        target_subjects,
        manifest_sha,
        parent_sha,
        target_subjects_sha,
        checkpoint_contract,
    ) = prepare_checkpoint_context(args)
    device = torch.device(args.device)

    timeline = dataset.event_timeline
    scheduled_groups = np.asarray(timeline.group_ids).astype(str)
    scheduled_subjects = np.asarray(timeline.subject_ids).astype(str)
    scheduled_candidates = np.asarray(timeline.candidate_ids).astype(str)
    scheduled_targets = np.asarray(timeline.target_candidate_ids).astype(str)
    scheduled_repetitions = np.asarray(timeline.repetition_indices, dtype=np.int64)
    scheduled_onsets = np.asarray(timeline.onset_times_s, dtype=float)
    scheduled_available = np.asarray(timeline.evidence_available_times_s, dtype=float)
    evidence_indices = np.asarray(timeline.evidence_indices, dtype=np.int64)
    vocabulary = tuple(sorted(np.unique(scheduled_candidates).tolist()))
    if len(vocabulary) != 9:
        raise ValueError(f"expected nine candidates, got {vocabulary}")

    groups_by_subject: dict[str, str] = {}
    for subject in target_subjects:
        groups = np.unique(scheduled_groups[scheduled_subjects == subject]).tolist()
        if len(groups) != 1:
            raise ValueError(f"subject {subject!r} maps to {len(groups)} groups")
        groups_by_subject[subject] = str(groups[0])
    block_event_mask = np.isin(scheduled_subjects, target_subjects) & (evidence_indices >= 0)
    event_rows = np.flatnonzero(block_event_mask)
    event_rows = event_rows[np.argsort(evidence_indices[event_rows], kind="stable")]
    epoch_rows = evidence_indices[event_rows]
    if len(np.unique(epoch_rows)) != len(epoch_rows):
        raise ValueError("evidence mapping is not one-to-one")
    if dataset.trial_channel_mask is None:
        trial_mask = np.broadcast_to(np.asarray(dataset.channel_mask, dtype=bool), dataset.X.shape[:2])
    else:
        trial_mask = np.asarray(dataset.trial_channel_mask, dtype=bool)
    raw_logits_chunks = []
    for start in range(0, len(epoch_rows), args.batch_size):
        rows = epoch_rows[start : start + args.batch_size]
        raw_logits_chunks.append(
            predict_n2p3_checkpoint(
                trunk,
                np.asarray(dataset.X[rows], dtype=np.float32),
                input_stats=input_stats,
                device=device,
                trial_channel_mask=trial_mask[rows],
            )
        )
    raw_logits = np.concatenate(raw_logits_chunks)
    llr_scores, calibration = checkpoint_scores_to_llr(payload, raw_logits)
    logit_by_event = {int(event): float(value) for event, value in zip(event_rows, raw_logits, strict=True)}
    llr_by_event = {int(event): float(value) for event, value in zip(event_rows, llr_scores, strict=True)}

    checkpoint_sha = sha256_file(args.checkpoint)
    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    available_occurrence_counter: Counter[tuple[str, str]] = Counter()
    with gzip.open(ledger_path, "wt", encoding="utf-8") as ledger:
        for event in event_rows:
            epoch = int(evidence_indices[event])
            subject = str(scheduled_subjects[event])
            candidate = str(scheduled_candidates[event])
            key = (subject, candidate)
            occurrence = int(available_occurrence_counter[key])
            available_occurrence_counter[key] += 1
            ledger.write(json.dumps({
                "kernel": args.kernel,
                "seed": args.seed,
                "block": args.block,
                "checkpoint_sha256": checkpoint_sha,
                "subject": subject,
                "group": str(scheduled_groups[event]),
                "scheduled_event_index": int(event),
                "epoch_row": epoch,
                "candidate": candidate,
                "target": str(scheduled_targets[event]),
                "scheduled_repetition_index": int(scheduled_repetitions[event]),
                "available_occurrence_index": occurrence,
                "onset_time_s": float(scheduled_onsets[event]),
                "evidence_available_time_s": float(scheduled_available[event]),
                "label": int(dataset.y[epoch]),
                "raw_logit": logit_by_event[int(event)],
                "llr_score": llr_by_event[int(event)],
            }, separators=(",", ":")) + "\n")

    records: list[dict[str, object]] = []
    for subject in target_subjects:
        group = groups_by_subject[subject]
        group_events = np.flatnonzero(scheduled_groups == group)
        if tuple(sorted(np.unique(scheduled_candidates[group_events]).tolist())) != vocabulary:
            raise ValueError(f"group {group!r} has incomplete scheduled vocabulary")
        targets = np.unique(scheduled_targets[group_events]).tolist()
        if len(targets) != 1 or targets[0] not in vocabulary:
            raise ValueError(f"group {group!r} has an invalid target")
        truth = str(targets[0])
        first_onset = float(np.min(scheduled_onsets[group_events]))
        ordered_available: dict[str, list[int]] = {}
        for candidate in vocabulary:
            rows = group_events[(scheduled_candidates[group_events] == candidate) & (evidence_indices[group_events] >= 0)]
            rows = rows[np.argsort(scheduled_onsets[rows], kind="stable")]
            ordered_available[candidate] = [int(row) for row in rows]
        counts = {candidate: len(rows) for candidate, rows in ordered_available.items()}
        r_s = min(counts.values())

        subject_events = [event for rows in ordered_available.values() for event in rows]
        labels = np.asarray([dataset.y[int(evidence_indices[event])] for event in subject_events], dtype=int)
        subject_logits = np.asarray([logit_by_event[event] for event in subject_events], dtype=float)
        auc = float(roc_auc_score(labels, subject_logits)) if len(np.unique(labels)) == 2 else None

        def decision(selected: dict[str, list[int]]) -> tuple[dict[str, float], str | None]:
            score = {
                candidate: float(sum(llr_by_event[event] for event in selected[candidate]))
                if selected[candidate] else -np.inf
                for candidate in vocabulary
            }
            return score, unique_prediction(score, vocabulary)

        def cost(
            selected_events: list[int],
            group_event_rows: np.ndarray,
            session_first_onset: float,
        ) -> dict[str, float | int | None]:
            return evidence_cost(
                selected_events,
                group_event_rows,
                session_first_onset,
                scheduled_onsets=scheduled_onsets,
                scheduled_available=scheduled_available,
            )

        raw_selected = ordered_available
        raw_scores, raw_prediction = decision(raw_selected)
        raw_event_rows = [event for rows in raw_selected.values() for event in rows]
        balanced_selected = {candidate: rows[:r_s] for candidate, rows in ordered_available.items()}
        balanced_scores, balanced_prediction = decision(balanced_selected)
        balanced_event_rows = [event for rows in balanced_selected.values() for event in rows]
        hit_by_r: dict[str, bool] = {}
        prediction_by_r: dict[str, str | None] = {}
        cost_by_r: dict[str, dict[str, float | int | None]] = {}
        for repetition in range(1, r_s + 1):
            selected = {candidate: rows[:repetition] for candidate, rows in ordered_available.items()}
            _, prediction = decision(selected)
            selected_rows = [event for rows in selected.values() for event in rows]
            hit_by_r[str(repetition)] = bool(prediction is not None and prediction == truth)
            prediction_by_r[str(repetition)] = prediction
            cost_by_r[str(repetition)] = cost(selected_rows, group_events, first_onset)
        records.append({
            "subject": subject,
            "group": group,
            "truth": truth,
            "candidate_counts": counts,
            "r_balanced_all": r_s,
            "binary_auc": auc,
            "balanced_all": {
                "prediction": balanced_prediction,
                "hit": bool(balanced_prediction is not None and balanced_prediction == truth),
                "scores": balanced_scores,
                "cost": cost(balanced_event_rows, group_events, first_onset),
            },
            "raw_all": {
                "prediction": raw_prediction,
                "hit": bool(raw_prediction is not None and raw_prediction == truth),
                "scores": raw_scores,
                "cost": cost(raw_event_rows, group_events, first_onset),
            },
            "hit_by_r": hit_by_r,
            "prediction_by_r": prediction_by_r,
            "cost_by_r": cost_by_r,
            "incomplete_reason": None if r_s > 0 else "at_least_one_candidate_has_zero_available_evidence",
        })

    if len(records) != len(target_subjects):
        raise AssertionError("requested target denominator changed during evaluation")
    summary = {
        "schema": "n2p3_kernel_all_evidence_block/2",
        "frozen_manifest": str(manifest_path.resolve()),
        "frozen_manifest_sha256": manifest_sha,
        "parent_block_manifest": str(Path(args.parent_manifest).resolve()),
        "parent_block_manifest_sha256": parent_sha,
        "dataset_cache": str(Path(args.dataset_cache).resolve()),
        "target_cache_sha256": CACHE_SHA256,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "target_subjects_file": str(Path(args.target_subjects_file).resolve()),
        "target_subjects_file_sha256": target_subjects_sha,
        "kernel": args.kernel,
        "seed": args.seed,
        "block": args.block,
        "pooling_mode": "full_unfold",
        "architecture": checkpoint_contract["architecture"],
        "training_contract_validated": True,
        "checkpoint_contract": checkpoint_contract,
        "source_qc_ptp_uv": SOURCE_QC_PTP_UV,
        "target_qc": False,
        "prefix_reps": 0,
        "candidate_vocabulary": list(vocabulary),
        "calibration": calibration,
        "ledger": str(ledger_path.resolve()),
        "ledger_sha256": sha256_file(ledger_path),
        "metrics": summarize_records(records),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "checkpoint": str(args.checkpoint),
        "kernel": args.kernel,
        "seed": args.seed,
        "block": args.block,
        "metrics": summary["metrics"],
    }), flush=True)


def paired_bootstrap_ci(differences: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = len(differences)
    output = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    offset = 0
    chunk = 4000
    while offset < BOOTSTRAP_ITERATIONS:
        take = min(chunk, BOOTSTRAP_ITERATIONS - offset)
        indices = rng.integers(0, n, size=(take, n))
        output[offset : offset + take] = differences[indices].mean(axis=1)
        offset += take
    return [float(x) for x in np.quantile(output, [0.025, 0.975])]


def sign_flip_p(differences: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(differences)))
    if np.allclose(differences, 0.0):
        return 1.0
    exceed = 0
    offset = 0
    chunk = 4000
    while offset < BOOTSTRAP_ITERATIONS:
        take = min(chunk, BOOTSTRAP_ITERATIONS - offset)
        signs = rng.integers(0, 2, size=(take, len(differences)), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * differences).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
        offset += take
    return float((exceed + 1) / (BOOTSTRAP_ITERATIONS + 1))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        value = min(1.0, (m - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def build_winner_selection(
    kernel_metrics: Mapping[str, Mapping[str, object]],
    primary_contrasts: Mapping[str, Mapping[str, object]],
    interactions: list[Mapping[str, object]],
) -> dict[str, object]:
    ranking = sorted(
        KERNELS,
        key=lambda kernel: float(
            kernel_metrics[str(kernel)]["balanced_all_operational_hit_seed_mean"]
        ),
        reverse=True,
    )
    top, runner_up = ranking[:2]
    checks: dict[str, dict[str, object]] = {}
    for other in KERNELS:
        if other == top:
            continue
        direct_name = f"K{top}-K{other}"
        reverse_name = f"K{other}-K{top}"
        if direct_name in primary_contrasts:
            contrast = primary_contrasts[direct_name]
            orientation = 1.0
            contrast_name = direct_name
        elif reverse_name in primary_contrasts:
            contrast = primary_contrasts[reverse_name]
            orientation = -1.0
            contrast_name = reverse_name
        else:
            raise ValueError(f"missing planned contrast for K{top} versus K{other}")
        stored_ci = contrast["balanced_all_paired_subject_bootstrap_ci95"]
        if orientation > 0:
            ci = [float(stored_ci[0]), float(stored_ci[1])]
        else:
            ci = [-float(stored_ci[1]), -float(stored_ci[0])]
        stored_seed_delta = contrast["balanced_all_operational_delta_by_seed"]
        seed_delta = {
            str(seed): orientation * float(stored_seed_delta[str(seed)]) for seed in SEEDS
        }
        positive_seeds = sum(delta > 0.0 for delta in seed_delta.values())
        no_practical_seed_reversal = all(
            delta > -PRACTICAL_DELTA for delta in seed_delta.values()
        )
        seed_stable = positive_seeds >= MIN_POSITIVE_SEEDS and no_practical_seed_reversal
        pair_interactions = [
            dict(item)
            for item in interactions
            if item.get("contrast") in {direct_name, reverse_name}
        ]
        delta = float(
            kernel_metrics[str(top)]["balanced_all_operational_hit_seed_mean"]
        ) - float(kernel_metrics[str(other)]["balanced_all_operational_hit_seed_mean"])
        requirements = {
            "practical_margin": delta >= PRACTICAL_DELTA,
            "paired_ci_excludes_zero": ci[0] > 0.0,
            "holm_adjusted_p_below_0.05": float(contrast["holm_adjusted_p"]) < 0.05,
            "no_practical_evidence_budget_reversal": not pair_interactions,
            "seed_direction_stable": seed_stable,
        }
        checks[str(other)] = {
            "top": top,
            "other": other,
            "stored_contrast": contrast_name,
            "top_minus_other": delta,
            "top_minus_other_ci95": ci,
            "top_minus_other_holm_p": float(contrast["holm_adjusted_p"]),
            "top_minus_other_by_seed": seed_delta,
            "positive_seed_count": positive_seeds,
            "minimum_seed_delta": min(seed_delta.values()),
            "evidence_budget_reversals": pair_interactions,
            "requirements": requirements,
            "qualified": all(requirements.values()),
        }
    winner_qualified = len(checks) == len(KERNELS) - 1 and all(
        bool(check["qualified"]) for check in checks.values()
    )
    runner_check = checks[str(runner_up)]
    return {
        "ordered_by_primary_point_estimate": ranking,
        "top_kernel": top,
        "runner_up": runner_up,
        "top_minus_runner_up": runner_check["top_minus_other"],
        "top_minus_runner_up_ci95": runner_check["top_minus_other_ci95"],
        "top_minus_runner_up_holm_p": runner_check["top_minus_other_holm_p"],
        "all_opponent_checks": checks,
        "seed_stability_rule": {
            "minimum_positive_seeds": MIN_POSITIVE_SEEDS,
            "practical_reversal_threshold": -PRACTICAL_DELTA,
            "strict_boundary": "a seed delta <= -0.02 fails",
            "formal_seed_p_value": None,
        },
        "winner_qualified_by_frozen_rule": winner_qualified,
        "conclusion": f"K{top}" if winner_qualified else "kernel_length_remains_unresolved",
    }


def analyze(args: argparse.Namespace) -> None:
    manifest, manifest_sha = validate_v4_manifest(
        args.manifest,
        expected_sha256=args.manifest_sha256,
    )
    expected_blocks, parent_sha = parent_block_subjects(args.parent_manifest, manifest)
    data_lock = manifest["data_lock"]
    assert isinstance(data_lock, Mapping)
    block_locks = {
        int(entry["block"]): entry
        for entry in data_lock["block_manifests"]
        if isinstance(entry, Mapping)
    }
    result_dir = Path(args.result_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    ledger_dir = Path(args.ledger_dir)
    loaded: dict[tuple[int, int, int], dict[str, object]] = {}
    for path in sorted(result_dir.glob("k*_seed*_blk*.json")):
        document = read_json_mapping(path, label=f"block result {path}")
        key = (int(document["kernel"]), int(document["seed"]), int(document["block"]))
        if key in loaded:
            raise ValueError(f"duplicate block result {key}")
        if document["frozen_manifest_sha256"] != manifest_sha:
            raise ValueError(f"result {path} is bound to another manifest")
        kernel, seed, block = key
        if document.get("schema") != "n2p3_kernel_all_evidence_block/2":
            raise ValueError(f"result {path} does not use the v4 block-result schema")
        if document.get("parent_block_manifest_sha256") != parent_sha:
            raise ValueError(f"result {path} is bound to another block partition")
        if document.get("target_cache_sha256") != CACHE_SHA256:
            raise ValueError(f"result {path} is bound to another cache")
        if document.get("pooling_mode") != "full_unfold":
            raise ValueError(f"result {path} changed the adopted readout")
        if float(document.get("source_qc_ptp_uv", math.nan)) != SOURCE_QC_PTP_UV:
            raise ValueError(f"result {path} changed source QC")
        if document.get("target_qc") is not False or int(document.get("prefix_reps", -1)) != 0:
            raise ValueError(f"result {path} is not the frozen Z0 target contract")
        if document.get("training_contract_validated") is not True:
            raise ValueError(f"result {path} did not pass the v4 checkpoint gate")
        if document.get("candidate_vocabulary") != [str(value) for value in range(1, 10)]:
            raise ValueError(f"result {path} changed the nine-digit candidate vocabulary")
        block_lock = block_locks.get(block)
        if block_lock is None or document.get("target_subjects_file_sha256") != block_lock.get(
            "sha256"
        ):
            raise ValueError(f"result {path} is not bound to frozen block {block}")
        records = document.get("records")
        if not isinstance(records, list):
            raise ValueError(f"result {path} lacks subject records")
        result_subjects = [str(record["subject"]) for record in records]
        if result_subjects != expected_blocks[block]:
            raise ValueError(f"result {path} subjects differ from frozen block {block}")
        checkpoint_path = checkpoint_dir / f"k{kernel}_seed{seed}_blk{block}.pt"
        ledger_path = ledger_dir / f"k{kernel}_seed{seed}_blk{block}.jsonl.gz"
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != document.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"result {path} checkpoint artifact hash mismatch")
        if not ledger_path.is_file() or sha256_file(ledger_path) != document.get("ledger_sha256"):
            raise ValueError(f"result {path} trial ledger hash mismatch")
        loaded[key] = document
    expected = {(kernel, seed, block) for kernel in KERNELS for seed in SEEDS for block in BLOCKS}
    if set(loaded) != expected:
        raise ValueError(f"missing/unexpected result blocks: missing={sorted(expected-set(loaded))} extra={sorted(set(loaded)-expected)}")

    reference_record: dict[str, dict[str, object]] = {}
    for record in loaded[(KERNELS[0], SEEDS[0], BLOCKS[0])]["records"]:
        reference_record[str(record["subject"])] = record
    for block in BLOCKS[1:]:
        for record in loaded[(KERNELS[0], SEEDS[0], block)]["records"]:
            reference_record[str(record["subject"])] = record
    if len(reference_record) != REQUESTED_N:
        raise ValueError("reference arm does not cover the requested cohort")
    for key, document in loaded.items():
        for record in document["records"]:
            subject = str(record["subject"])
            reference = reference_record[subject]
            invariant = (
                record.get("group"),
                record.get("truth"),
                record.get("candidate_counts"),
                int(record.get("r_balanced_all", -1)),
            )
            expected_invariant = (
                reference.get("group"),
                reference.get("truth"),
                reference.get("candidate_counts"),
                int(reference.get("r_balanced_all", -1)),
            )
            if invariant != expected_invariant:
                raise ValueError(
                    f"subject data invariants differ across arms for {subject!r} in {key}"
                )

    subject_records: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
    seed_metrics = []
    for kernel in KERNELS:
        for seed in SEEDS:
            records: list[dict[str, object]] = []
            for block in BLOCKS:
                records.extend(loaded[(kernel, seed, block)]["records"])
            by_subject = {str(record["subject"]): record for record in records}
            if len(records) != REQUESTED_N or len(by_subject) != REQUESTED_N:
                raise ValueError(f"arm K{kernel} seed {seed} does not have {REQUESTED_N} unique subjects")
            subject_records[(kernel, seed)] = by_subject
            metrics = summarize_records(records)
            seed_metrics.append({"kernel": kernel, "seed": seed, **metrics})
    reference_subjects = sorted(subject_records[(KERNELS[0], SEEDS[0])])
    if any(sorted(records) != reference_subjects for records in subject_records.values()):
        raise ValueError("subject sets differ between arms")

    kernel_subject: dict[int, dict[str, dict[str, object]]] = {}
    kernel_metrics: dict[str, dict[str, object]] = {}
    for kernel in KERNELS:
        aggregated: dict[str, dict[str, object]] = {}
        for subject in reference_subjects:
            rows = [subject_records[(kernel, seed)][subject] for seed in SEEDS]
            r_values = {int(row["r_balanced_all"]) for row in rows}
            if len(r_values) != 1:
                raise ValueError("R_s unexpectedly depends on training seed")
            max_r = next(iter(r_values))
            aggregated[subject] = {
                "r_s": max_r,
                "balanced_hit_seed_mean": float(np.mean([bool(row["balanced_all"]["hit"]) for row in rows])),
                "raw_hit_seed_mean": float(np.mean([bool(row["raw_all"]["hit"]) for row in rows])),
                "auc_seed_mean": float(np.mean([float(row["binary_auc"]) for row in rows if row["binary_auc"] is not None])),
                "hit_by_r_seed_mean": {
                    str(repetition): float(np.mean([bool(row["hit_by_r"].get(str(repetition), False)) for row in rows]))
                    for repetition in range(1, max_r + 1)
                },
            }
        kernel_subject[kernel] = aggregated
        seed_rows = [row for row in seed_metrics if row["kernel"] == kernel]
        max_r = max(max(int(key) for key in row["hit_curve"]) for row in seed_rows)
        curve = {}
        for repetition in range(1, max_r + 1):
            subjects_at_r = [subject for subject in reference_subjects if int(aggregated[subject]["r_s"]) >= repetition]
            hits = float(sum(aggregated[subject]["hit_by_r_seed_mean"][str(repetition)] for subject in subjects_at_r))
            curve[str(repetition)] = {
                "eligible": len(subjects_at_r),
                "coverage": len(subjects_at_r) / REQUESTED_N,
                "conditional_hit_seed_mean": hits / len(subjects_at_r) if subjects_at_r else None,
                "operational_hit_seed_mean": hits / REQUESTED_N,
            }
        kernel_metrics[str(kernel)] = {
            "balanced_all_operational_hit_seed_mean": float(np.mean([aggregated[s]["balanced_hit_seed_mean"] for s in reference_subjects])),
            "balanced_all_conditional_hit_seed_mean": float(np.mean([aggregated[s]["balanced_hit_seed_mean"] for s in reference_subjects if int(aggregated[s]["r_s"]) > 0])),
            "raw_all_operational_hit_seed_mean": float(np.mean([aggregated[s]["raw_hit_seed_mean"] for s in reference_subjects])),
            "raw_all_conditional_hit_seed_mean": float(np.mean([aggregated[s]["raw_hit_seed_mean"] for s in reference_subjects if int(aggregated[s]["r_s"]) > 0])),
            "binary_auc_subject_seed_macro": float(np.mean([aggregated[s]["auc_seed_mean"] for s in reference_subjects])),
            "coverage": float(np.mean([int(aggregated[s]["r_s"]) > 0 for s in reference_subjects])),
            "r_s_distribution": {str(k): int(v) for k, v in sorted(Counter(int(aggregated[s]["r_s"]) for s in reference_subjects).items())},
            "hit_curve": curve,
        }

    planned_pairs = ((35, 33), (35, 65), (33, 65))
    primary_contrasts: dict[str, dict[str, object]] = {}
    p_values = {}
    for index, (a, b) in enumerate(planned_pairs):
        name = f"K{a}-K{b}"
        delta = np.asarray([
            kernel_subject[a][subject]["balanced_hit_seed_mean"] - kernel_subject[b][subject]["balanced_hit_seed_mean"]
            for subject in reference_subjects
        ], dtype=float)
        raw_delta = np.asarray([
            kernel_subject[a][subject]["raw_hit_seed_mean"] - kernel_subject[b][subject]["raw_hit_seed_mean"]
            for subject in reference_subjects
        ], dtype=float)
        auc_delta = np.asarray([
            kernel_subject[a][subject]["auc_seed_mean"] - kernel_subject[b][subject]["auc_seed_mean"]
            for subject in reference_subjects
        ], dtype=float)
        rng = np.random.default_rng(ANALYSIS_SEED + index * 10)
        p_value = sign_flip_p(delta, rng)
        p_values[name] = p_value
        curve_delta = {}
        max_r = max(max(int(x) for x in kernel_metrics[str(a)]["hit_curve"]), max(int(x) for x in kernel_metrics[str(b)]["hit_curve"]))
        for repetition in range(1, max_r + 1):
            differences = []
            eligible = 0
            for subject in reference_subjects:
                r_s = int(kernel_subject[a][subject]["r_s"])
                if r_s >= repetition:
                    eligible += 1
                    av = float(kernel_subject[a][subject]["hit_by_r_seed_mean"][str(repetition)])
                    bv = float(kernel_subject[b][subject]["hit_by_r_seed_mean"][str(repetition)])
                else:
                    av = bv = 0.0
                differences.append(av - bv)
            curve_delta[str(repetition)] = {
                "eligible": eligible,
                "coverage": eligible / REQUESTED_N,
                "operational_delta": float(np.mean(differences)),
            }
        primary_contrasts[name] = {
            "a": a,
            "b": b,
            "balanced_all_operational_delta": float(np.mean(delta)),
            "balanced_all_operational_delta_by_seed": {
                str(seed): float(
                    np.mean(
                        [
                            bool(subject_records[(a, seed)][subject]["balanced_all"]["hit"])
                            - bool(subject_records[(b, seed)][subject]["balanced_all"]["hit"])
                            for subject in reference_subjects
                        ]
                    )
                )
                for seed in SEEDS
            },
            "balanced_all_paired_subject_bootstrap_ci95": paired_bootstrap_ci(delta, np.random.default_rng(ANALYSIS_SEED + index * 10 + 1)),
            "paired_sign_flip_p": p_value,
            "raw_all_operational_delta": float(np.mean(raw_delta)),
            "raw_all_paired_subject_bootstrap_ci95": paired_bootstrap_ci(raw_delta, np.random.default_rng(ANALYSIS_SEED + index * 10 + 2)),
            "auc_subject_seed_macro_delta": float(np.mean(auc_delta)),
            "auc_paired_subject_bootstrap_ci95": paired_bootstrap_ci(auc_delta, np.random.default_rng(ANALYSIS_SEED + index * 10 + 3)),
            "hit_curve_operational_delta": curve_delta,
        }
    adjusted = holm_adjust(p_values)
    for name, value in adjusted.items():
        primary_contrasts[name]["holm_adjusted_p"] = value

    interactions = []
    for name, contrast in primary_contrasts.items():
        primary = float(contrast["balanced_all_operational_delta"])
        primary_sign = 0 if abs(primary) < PRACTICAL_DELTA else int(np.sign(primary))
        for repetition, point in contrast["hit_curve_operational_delta"].items():
            delta = float(point["operational_delta"])
            monitored = repetition in {"5", "8"} or float(point["coverage"]) >= 0.80
            if monitored and primary_sign and abs(delta) >= PRACTICAL_DELTA and int(np.sign(delta)) != primary_sign:
                interactions.append({
                    "contrast": name,
                    "repetition": int(repetition),
                    "primary_delta": primary,
                    "curve_delta": delta,
                    "coverage": point["coverage"],
                })

    selection = build_winner_selection(kernel_metrics, primary_contrasts, interactions)

    subject_csv = Path(args.subject_csv)
    subject_csv.parent.mkdir(parents=True, exist_ok=True)
    with subject_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "kernel", "r_s", "balanced_hit_seed_mean", "raw_hit_seed_mean", "auc_seed_mean"])
        for kernel in KERNELS:
            for subject in reference_subjects:
                row = kernel_subject[kernel][subject]
                writer.writerow([subject, kernel, row["r_s"], row["balanced_hit_seed_mean"], row["raw_hit_seed_mean"], row["auc_seed_mean"]])

    result = {
        "schema": "n2p3_temporal_kernel_ablation_result/3",
        "frozen_manifest": str(Path(args.manifest).resolve()),
        "frozen_manifest_sha256": manifest_sha,
        "parent_block_manifest": str(Path(args.parent_manifest).resolve()),
        "parent_block_manifest_sha256": parent_sha,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "seed_aggregation": "subject-level mean correctness across seeds",
        "conditional_inference_scope": CONDITIONAL_INFERENCE_SCOPE,
        "requested_subjects": REQUESTED_N,
        "kernels": kernel_metrics,
        "per_seed_metrics": seed_metrics,
        "planned_primary_contrasts": primary_contrasts,
        "holm_family": list(primary_contrasts),
        "practical_delta_threshold": PRACTICAL_DELTA,
        "kernel_x_evidence_budget_interactions": interactions,
        "selection": selection,
        "subject_seed_aggregate_csv": str(subject_csv.resolve()),
        "subject_seed_aggregate_csv_sha256": sha256_file(subject_csv),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "selection": result["selection"], "kernels": kernel_metrics}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_checkpoint_contract_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--dataset-cache", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--target-subjects-file", required=True)
        command.add_argument("--parent-manifest", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--manifest-sha256", default=None)
        command.add_argument("--kernel", type=int, choices=KERNELS, required=True)
        command.add_argument("--seed", type=int, choices=SEEDS, required=True)
        command.add_argument("--block", type=int, choices=BLOCKS, required=True)

    validation = subparsers.add_parser("validate-checkpoint")
    add_checkpoint_contract_arguments(validation)
    validation.set_defaults(function=validate_checkpoint)

    evaluation = subparsers.add_parser("evaluate")
    add_checkpoint_contract_arguments(evaluation)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--batch-size", type=int, default=2048)
    evaluation.add_argument("--ledger", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(function=evaluate)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--result-dir", required=True)
    analysis.add_argument("--checkpoint-dir", required=True)
    analysis.add_argument("--ledger-dir", required=True)
    analysis.add_argument("--parent-manifest", required=True)
    analysis.add_argument("--manifest", required=True)
    analysis.add_argument("--manifest-sha256", default=None)
    analysis.add_argument("--subject-csv", required=True)
    analysis.add_argument("--output", required=True)
    analysis.set_defaults(function=analyze)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.function(parsed)
