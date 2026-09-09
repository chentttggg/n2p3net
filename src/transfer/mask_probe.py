"""Predeclared deletion probes on a frozen source model and fully observed BNCI EEG."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data.channel import build_channel_identity
from data.domain import adapt_observed_montage
from data.epochs import EpochDataset, epoch_row_identity_digest, materialize_dataset_lineage
from data.qc_features import effective_observed_mask
from data.study import checkpoint_study_membership, dataset_study_membership, study_ids_for_identity
from models.n2p3net import N2P3Net
from research.evaluation import build_evaluation_run_contract, checkpoint_model_origin
from research.source_selection import SOURCE_CHAIN_LAYOUTS
from transfer.checkpoint import (
    checkpoint_classifier_is_trained,
    checkpoint_input_stats,
    checkpoint_training_identity_ledger,
    load_n2p3_trunk_checkpoint,
    predict_n2p3_checkpoint,
)

MASK_PROBE_CHANNELS = build_channel_identity(SOURCE_CHAIN_LAYOUTS["six"]).names
MASK_PROBE_REMOVALS = (
    ("full", ()),
    ("remove_fz", ("FZ",)),
    ("remove_po8", ("PO8",)),
    ("remove_oz", ("OZ",)),
    ("common_three", tuple(name for name in MASK_PROBE_CHANNELS if name not in {"P3", "PZ", "P4"})),
)


def _array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256(f"{contiguous.dtype}:{contiguous.shape}".encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _state_digest(trunk: N2P3Net) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(trunk.state_dict().items()):
        digest.update(name.encode())
        digest.update(_array_digest(tensor.detach().cpu().numpy()).encode("ascii"))
    return digest.hexdigest()


def run_checkpoint_mask_probe(
    dataset: EpochDataset,
    payload: Mapping[str, object],
    *,
    target_subject: str,
    checkpoint_sha256: str,
    target_cache_sha256: str,
    evaluation_snapshot_sha256: str,
    device: torch.device | None = None,
    batch_size: int = 256,
    dry_run: bool = False,
    study_membership_proofs: Sequence[Mapping] = (),
) -> dict[str, object]:
    """Score five matched masks without fitting, selecting, or changing source state."""
    dataset.validate(require_labels=True)
    if dataset.identity_table is None:
        raise ValueError("mask probe requires production participant identities")
    if tuple(dataset.channel_names) != MASK_PROBE_CHANNELS:
        raise ValueError("mask probe requires the declared six-slot layout")
    # The idempotent production transform also checks its measured-reference lineage.
    adapt_observed_montage(dataset, MASK_PROBE_CHANNELS, reference_channel="Cz")
    target_identity = dataset.identity_table.subset([target_subject])
    target_membership = dataset_study_membership(dataset)
    if study_ids_for_identity(target_identity, target_membership) != ("BNCI2014-008",):
        raise ValueError("fixed mask probe requires actual BNCI2014-008 participant origin")
    source_identity = checkpoint_training_identity_ledger(payload)
    source_membership = checkpoint_study_membership(payload, proofs=study_membership_proofs)
    if "BNCI2014-008" in study_ids_for_identity(source_identity, source_membership):
        raise ValueError("the entire BNCI target library must be excluded from the source model")
    if not checkpoint_classifier_is_trained(payload):
        raise ValueError("mask probe requires a trained source classifier")
    rows = np.flatnonzero(dataset.subject_ids.astype(str) == str(target_subject))
    if not len(rows):
        raise ValueError("mask probe target participant has no rows")
    observed = effective_observed_mask(
        n_epochs=dataset.n_epochs,
        channel_mask=dataset.channel_mask,
        trial_channel_mask=dataset.trial_channel_mask,
    )[rows]
    if not bool(observed.all()):
        raise ValueError("all six slots must be actually observed in every selected BNCI row")
    if batch_size < 1:
        raise ValueError("probe batch size must be positive")
    trunk, checked_payload = load_n2p3_trunk_checkpoint(
        payload, dataset, target_subject=target_subject
    )
    if not isinstance(trunk, N2P3Net) or trunk.splitbn_config is not None:
        raise ValueError("fixed mask probe requires the plain N2P3Net source model")
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    trunk = trunk.eval() if dry_run else trunk.to(resolved_device).eval()
    before = _state_digest(trunk)
    inputs = dataset.X[rows]
    labels = np.asarray(dataset.y)[rows]
    participant = target_identity.record_for(target_subject).authority_key()
    masks = []
    arms = []
    for name, removed in MASK_PROBE_REMOVALS:
        mask = observed.copy()
        for channel in removed:
            mask[:, MASK_PROBE_CHANNELS.index(channel)] = False
        masks.append(mask)
        contract = build_evaluation_run_contract(
            arm_name=f"s1_mask_{name}",
            model_origin=checkpoint_model_origin(
                checked_payload, checkpoint_sha256=checkpoint_sha256
            ),
            dataset=dataset,
            target_cache_sha256=target_cache_sha256,
            source_snapshot_sha256=evaluation_snapshot_sha256,
            requested_subjects=(target_subject,),
            identity_policy="source_or_global",
            target_protocol={
                "kind": "fixed_model_mask_probe",
                "actual_target_library": "BNCI2014-008",
                "row_indices_sha256": _array_digest(rows),
                "rows": len(rows),
            },
            adaptation={
                "fit_rows": 0,
                "optimizer_updates": 0,
                "input_statistics": "source",
                "bn_buffers": "source_frozen",
                "parameters": "frozen",
                "removed_channels": list(removed),
                "mask_sha256": _array_digest(mask),
            },
            decision={"kind": "binary", "metric": "roc_auc", "selection": "none"},
            evidence_scope={
                "level": "development",
                "purpose": "fixed_model_deletion_probe",
                "limitation": "information_loss_and_mask_context_are_not_separated",
            },
        )
        arms.append(
            {
                "name": name,
                "removed_channels": list(removed),
                "retained_channels": [
                    channel for channel in MASK_PROBE_CHANNELS if channel not in removed
                ],
                "mask_sha256": _array_digest(mask),
                "participant_key": participant,
                "status": "planned" if dry_run else "pending",
                "evaluation_contract": contract.record(),
                "evaluation_contract_digest": contract.digest(),
            }
        )
    output = {
        "schema": "n2p3_fixed_checkpoint_mask_probe/1",
        "dry_run": dry_run,
        "participant_key": participant,
        "local_subject_id": target_subject,
        "row_indices": rows.tolist(),
        "row_identity_sha256": epoch_row_identity_digest(dataset),
        "input_sha256": _array_digest(inputs),
        "labels_sha256": _array_digest(labels),
        "observed_mask_sha256": _array_digest(observed),
        "checkpoint_sha256": checkpoint_sha256,
        "source_study_membership": source_membership,
        "target_study_membership": target_membership,
        "target_cache_sha256": target_cache_sha256,
        "evaluation_snapshot_sha256": evaluation_snapshot_sha256,
        "source_reference": dataset.provenance["source_reference"],
        "parent_lineage_sha256": materialize_dataset_lineage(dataset).entity_digest,
        "state_sha256_before": before,
        "state_sha256_after": before,
        "requested_device": str(resolved_device),
        "effective_device": None if dry_run else str(resolved_device),
        "inference_batch_size": batch_size,
        "fit_rows": 0,
        "optimizer_updates": 0,
        "arms": arms,
    }
    if dry_run:
        return output
    statistics = checkpoint_input_stats(checked_payload, dataset.n_channels)
    captures: dict[str, np.ndarray] = {}

    def capture(name):
        def hook(_module, _args, result):
            captures[name] = result.detach().cpu().numpy().copy()

        return hook

    def before_bn(_module, args):
        captures["bn_pre"] = args[0].detach().cpu().numpy().copy()

    handles = [
        trunk.st_temporal_norm.register_forward_pre_hook(before_bn),
        trunk.st_temporal_norm.register_forward_hook(capture("bn_post")),
        trunk.st_spatial.register_forward_hook(capture("spatial")),
    ]
    traces = [
        {
            "logits": [],
            "bn_pre_mean": [],
            "bn_pre_square": [],
            "bn_post_mean": [],
            "bn_post_square": [],
            "spatial_delta_rms": [],
        }
        for _ in arms
    ]
    try:
        for start in range(0, len(rows), batch_size):
            full_spatial = None
            for mask, trace in zip(masks, traces, strict=True):
                captures.clear()
                logits = predict_n2p3_checkpoint(
                    trunk,
                    inputs[start : start + batch_size],
                    input_stats=statistics,
                    device=resolved_device,
                    trial_channel_mask=mask[start : start + batch_size],
                )
                if not np.isfinite(logits).all():
                    raise ValueError("mask probe produced non-finite logits")
                trace["logits"].append(logits)
                for name in ("bn_pre", "bn_post"):
                    values = captures[name].astype(np.float64)
                    trace[f"{name}_mean"].append(values.mean(axis=(1, 3)))
                    trace[f"{name}_square"].append(np.square(values).mean(axis=(1, 3)))
                spatial = captures["spatial"].astype(np.float64)
                if full_spatial is None:
                    full_spatial = spatial
                trace["spatial_delta_rms"].append(
                    np.sqrt(np.square(spatial - full_spatial).mean(axis=(1, 2, 3)))
                )
    finally:
        for handle in handles:
            handle.remove()
    for arm, trace in zip(arms, traces, strict=True):
        logits = np.concatenate(trace["logits"])
        arm.update(
            {
                "status": "complete" if len(np.unique(labels)) == 2 else "auc_not_estimable",
                "logits": logits.tolist(),
                "labels": labels.astype(int).tolist(),
                "auc": float(roc_auc_score(labels, logits))
                if len(np.unique(labels)) == 2
                else None,
                "activations": {
                    **{
                        name: {
                            "channel_mean": np.concatenate(trace[f"{name}_mean"]).mean(0).tolist(),
                            "channel_rms": np.sqrt(
                                np.concatenate(trace[f"{name}_square"]).mean(0)
                            ).tolist(),
                        }
                        for name in ("bn_pre", "bn_post")
                    },
                    "spatial_delta_rms_by_trial": np.concatenate(
                        trace["spatial_delta_rms"]
                    ).tolist(),
                },
            }
        )
    output["state_sha256_after"] = _state_digest(trunk)
    if output["state_sha256_after"] != before:
        raise RuntimeError("frozen mask probe changed source parameters or buffers")
    return output
