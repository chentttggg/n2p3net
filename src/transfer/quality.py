"""Personal epoch quality fitted on declared calibration rows only."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from data.artifact import FoldLocalArtifactModel, FoldLocalArtifactPolicy, _fit_data_digest
from data.artifact_sidecar import artifact_model_from_record
from data.brainsync_contract import BRAIN_SYNC_SESSION_SCHEMA
from data.continuous_qc import (
    CONTINUOUS_QC_CONSTANT_SCHEMA,
    continuous_qc_from_record,
    minimum_usable_channel_count,
    validate_recovery_contract,
)
from data.epochs import EpochDataset, materialize_dataset_identity, materialize_dataset_lineage
from data.lineage import DataLineage
from data.personal_quality import (
    PersonalEpochQualityModel,
    PersonalEpochQualityPolicy,
    personal_epoch_quality_from_record,
)
from data.qc_features import compute_epoch_qc_features, effective_observed_mask
from research.contracts import semantic_sha256

EpochQualityPolicy = FoldLocalArtifactPolicy | PersonalEpochQualityPolicy
EpochQualityModel = FoldLocalArtifactModel | PersonalEpochQualityModel


class PersonalEpochRawQualityError(ValueError):
    """BrainSync personal signal evidence lacks its required raw quality stage."""


def validate_personal_epoch_raw_quality(processing: dict, quality) -> None:
    if not isinstance(quality, (PersonalEpochQualityPolicy, PersonalEpochQualityModel)):
        return
    message = (
        "BrainSync personal epoch QC requires applied channel-local continuous QC /4 "
        "with raw exact-constant detection enabled; provide a compatible --continuous-qc-model. "
        "Epoch amplitude diagnostics alone do not establish clean EEG."
    )
    if not isinstance(processing, dict) or processing.get("status") != "applied":
        raise PersonalEpochRawQualityError(message)
    record = processing.get("model")
    if not isinstance(record, dict) or (
        record.get("schema") != CONTINUOUS_QC_CONSTANT_SCHEMA
        or not isinstance(record.get("spec"), dict)
        or record["spec"].get("channel_local") is not True
        or record["spec"].get("flatline_min_duration_s") is None
        or processing.get("model_digest") != semantic_sha256(record)
    ):
        raise PersonalEpochRawQualityError(message)
    try:
        continuous_qc_from_record(record)
    except (KeyError, TypeError, ValueError) as error:
        raise PersonalEpochRawQualityError(f"{message} Invalid profile: {error}") from error


def _validate_dataset_raw_quality(dataset: EpochDataset, quality) -> None:
    if dataset.provenance.get("session_schema") == BRAIN_SYNC_SESSION_SCHEMA:
        processing = dataset.provenance.get("continuous_artifact_processing", {})
        validate_personal_epoch_raw_quality(processing, quality)
        if isinstance(quality, (PersonalEpochQualityPolicy, PersonalEpochQualityModel)) and dataset.preprocessing.filter_phase == "zero":
            try:
                frozen = continuous_qc_from_record(processing["model"])
                validate_recovery_contract(frozen, dataset.preprocessing,
                                           source_reference=dataset.provenance["source_reference"])
            except (KeyError, TypeError, ValueError) as error:
                raise PersonalEpochRawQualityError(
                    "zero-phase personal QC requires matching bidirectional boundary recovery") from error


def epoch_quality_from_record(record: dict[str, object]) -> EpochQualityModel:
    """Restore the declared algorithm without reinterpreting historical thresholds."""
    if not isinstance(record, dict):
        raise ValueError("epoch quality model must be a record")
    if record.get("method") == "personal_epoch_signal_evidence":
        return personal_epoch_quality_from_record(record)
    if record.get("method") == "fold_local_ptp_cv":
        return artifact_model_from_record(record)
    raise ValueError("unsupported epoch quality model method")


def epoch_quality_channel_count(fitted: EpochQualityModel) -> int:
    if isinstance(fitted, PersonalEpochQualityModel):
        return fitted.n_channels
    return len(fitted.ptp_thresholds)


def _aligned_events(dataset: EpochDataset) -> np.ndarray:
    evidence = np.asarray(dataset.event_timeline.evidence_indices, dtype=np.int64)
    events = np.flatnonzero(evidence >= 0)
    return events[np.argsort(evidence[events], kind="stable")]


def _input_contract(dataset: EpochDataset) -> dict[str, object]:
    reference = dataset.provenance.get("source_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Personal epoch quality requires an explicit source_reference.")
    identity = materialize_dataset_identity(replace(dataset))
    keys = identity.authority_keys()
    if len(keys) != 1:
        raise ValueError("Personal epoch quality requires exactly one authoritative participant.")
    return {
        "participant_key": keys[0],
        "source_reference": reference,
        "channel_names": list(dataset.channel_names),
        "preprocessing": json.loads(json.dumps(dataset.preprocessing.record())),
    }


def validate_epoch_quality_fit_scope(
    dataset: EpochDataset, training_mask: np.ndarray, *,
    policy: EpochQualityPolicy | None = None, model: EpochQualityModel | None = None,
) -> None:
    if policy is not None and model is not None:
        raise ValueError("epoch_quality_policy and epoch_quality_model are mutually exclusive")
    _validate_dataset_raw_quality(dataset, policy if policy is not None else model)
    if policy is not None:
        if not isinstance(policy, (FoldLocalArtifactPolicy, PersonalEpochQualityPolicy)):
            raise ValueError("unsupported epoch quality policy")
        policy.validate()
    if model is not None:
        validate_frozen_epoch_quality(dataset, model, training_mask)


def validate_frozen_epoch_quality(
    dataset: EpochDataset, fitted: EpochQualityModel, training_mask: np.ndarray,
) -> None:
    """Bind a frozen QC state to its original physical rows inside this fit scope."""
    dataset.validate()
    training = np.asarray(training_mask)
    if training.dtype != np.dtype(bool) or training.shape != (dataset.n_epochs,):
        raise ValueError("Frozen epoch quality requires an aligned boolean training mask.")
    if not isinstance(fitted, (FoldLocalArtifactModel, PersonalEpochQualityModel)):
        raise ValueError("Frozen epoch quality requires a fitted artifact model.")
    _validate_dataset_raw_quality(dataset, fitted)
    epoch_quality_from_record(fitted.record())
    for key, expected in _input_contract(dataset).items():
        if fitted.fit_provenance.get(key) != expected:
            raise ValueError(f"Frozen epoch quality {key} differs from the calibration input.")
    scope = fitted.fit_provenance
    events = scope.get("training_event_ids")
    if (
        scope.get("schema") != "n2p3_personal_epoch_quality_fit/1"
        or scope.get("scope") != "explicit_training_mask" or scope.get("labels_used") is not False
        or not isinstance(events, list) or not events
        or any(not isinstance(event, str) or not event for event in events)
        or len(events) != len(set(events)) or len(events) != fitted.fit_n_epochs
        or scope.get("training_event_ids_digest") != semantic_sha256(events)
        or fitted.fit_group_source != "explicit_fit_groups" or not fitted.fit_data_digest
    ):
        raise ValueError("Frozen epoch quality lacks consistent original fit event/digest evidence.")
    event_rows = _aligned_events(dataset)
    row_events = np.asarray(dataset.event_timeline.event_ids).astype(str)[event_rows]
    authorized = set(row_events[training])
    if not set(events) <= authorized:
        raise ValueError("Frozen epoch quality fit events are outside the authorized training rows.")
    index_by_event = {event: index for index, event in enumerate(row_events)}
    rows = np.asarray([index_by_event[event] for event in events], dtype=np.int64)
    sessions = np.asarray(dataset.event_timeline.session_ids).astype(str)[event_rows][rows]
    if scope.get("session_ids") != sorted(np.unique(sessions).tolist()):
        raise ValueError("Frozen epoch quality fit sessions disagree with its authorized events.")
    if scope.get("group_axis") == "session":
        groups = sessions
    elif scope.get("group_axis") == "block" and "block_id" in dataset.metadata:
        blocks = dataset.metadata.iloc[rows]["block_id"]
        if blocks.isna().any():
            raise ValueError("Frozen epoch quality requires declared original block identities.")
        groups = np.asarray([json.dumps([session, block], separators=(",", ":"))
            for session, block in zip(sessions, blocks.to_numpy(dtype=str), strict=True)])
    else:
        raise ValueError("Frozen epoch quality has no reconstructable session/block fit groups.")
    if fitted.fit_groups != tuple(sorted(np.unique(groups).tolist())):
        raise ValueError("Frozen epoch quality fit groups differ from the original calibration rows.")
    observed = effective_observed_mask(n_epochs=dataset.n_epochs, channel_mask=dataset.channel_mask,
                                       trial_channel_mask=dataset.trial_channel_mask)[rows]
    features = (dataset.qc_features.subset(rows) if dataset.qc_features is not None
                else compute_epoch_qc_features(dataset.X[rows], channel_mask=dataset.channel_mask,
                                               trial_channel_mask=observed))
    actual = _fit_data_digest(dataset.X[rows], np.asarray(dataset.subject_ids)[rows], groups, observed, features)
    if actual != fitted.fit_data_digest:
        raise ValueError("Frozen epoch quality fit data digest differs in X, masks, or QC features.")


def load_epoch_quality(path: str | Path) -> EpochQualityModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"model", "digest"} or (
        payload["digest"] != semantic_sha256(payload["model"])
    ):
        raise ValueError("epoch quality model digest is invalid")
    return epoch_quality_from_record(payload["model"])


def save_epoch_quality(path: str | Path, fitted: EpochQualityModel) -> Path:
    destination = Path(path)
    record = fitted.record()
    content = json.dumps({"model": record, "digest": semantic_sha256(record)},
                         indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def fit_epoch_quality(
    dataset: EpochDataset,
    training_mask: np.ndarray,
    policy: EpochQualityPolicy,
) -> EpochQualityModel:
    """Fit label-blind thresholds with real sessions, or blocks within one session.

    Only the authorized rows determine the grouping axis. A sole session/block
    is legal threshold calibration, but its fallback has no CV evidence.
    """
    dataset.validate()
    _validate_dataset_raw_quality(dataset, policy)
    training = np.asarray(training_mask)
    if training.dtype != np.dtype(bool) or training.shape != (dataset.n_epochs,):
        raise ValueError("training_mask must be an aligned boolean vector.")
    if training.sum() < 2:
        raise ValueError("Personal epoch quality requires at least two training epochs.")
    contract = _input_contract(dataset)
    event_rows = _aligned_events(dataset)
    sessions = np.asarray(dataset.event_timeline.session_ids).astype(str)[event_rows]
    selected_sessions = sessions[training]
    if np.any(np.char.strip(selected_sessions) == ""):
        raise ValueError("Personal epoch quality requires declared calibration session IDs.")
    groups = selected_sessions
    axis = "session"
    if len(np.unique(selected_sessions)) == 1 and "block_id" in dataset.metadata:
        blocks = dataset.metadata.loc[training, "block_id"]
        if blocks.isna().any() or np.any(np.char.strip(blocks.to_numpy(dtype=str)) == ""):
            raise ValueError("Personal epoch quality block_id must contain declared identities.")
        if blocks.nunique() > 1:
            groups = np.asarray([
                json.dumps([session, block], separators=(",", ":"))
                for session, block in zip(selected_sessions, blocks.to_numpy(dtype=str), strict=True)
            ])
            axis = "block"
    observed = effective_observed_mask(
        n_epochs=dataset.n_epochs, channel_mask=dataset.channel_mask,
        trial_channel_mask=dataset.trial_channel_mask,
    )
    features = None if dataset.qc_features is None else dataset.qc_features.subset(training)
    fitted = policy.fit(
        dataset.X[training], np.asarray(dataset.subject_ids)[training], observed[training],
        features, fit_groups=groups,
    )
    event_ids = np.asarray(dataset.event_timeline.event_ids).astype(str)[event_rows][training].tolist()
    provenance = {
        "schema": "n2p3_personal_epoch_quality_fit/1",
        "scope": "explicit_training_mask",
        **contract,
        "group_axis": axis,
        "session_ids": sorted(np.unique(selected_sessions).tolist()),
        "training_event_ids": event_ids,
        "training_event_ids_digest": semantic_sha256(event_ids),
        "labels_used": False,
    }
    return replace(fitted, fit_provenance=provenance)


def transform_epoch_quality(dataset: EpochDataset, fitted: EpochQualityModel):
    """Validate frozen QC and return its transformation and full rejection mask."""
    dataset.validate()
    _validate_dataset_raw_quality(dataset, fitted)
    contract = _input_contract(dataset)
    for key, expected in contract.items():
        if fitted.fit_provenance.get(key) != expected:
            raise ValueError(f"Personal epoch quality fitted {key} does not match the dataset.")
    observed = effective_observed_mask(
        n_epochs=dataset.n_epochs, channel_mask=dataset.channel_mask,
        trial_channel_mask=dataset.trial_channel_mask,
    )
    transformed = fitted.transform(dataset.X, observed, dataset.qc_features)
    rejected = transformed.drop_epoch_mask | transformed.all_channels_bad
    minimum = minimum_usable_channel_count(dataset.provenance.get("continuous_artifact_processing", {}), dataset.n_channels)
    rejected |= transformed.trial_channel_mask.sum(axis=1) < minimum
    return transformed, rejected


def apply_epoch_quality(dataset: EpochDataset, fitted: EpochQualityModel) -> EpochDataset:
    """Mask local channels and reject complete epochs while retaining every event.

    The empty-evidence case raises explicitly. Operational inference must retain
    the original timeline and abstain when no usable evidence remains.
    """
    transformed, rejected = transform_epoch_quality(dataset, fitted)
    retained = ~rejected
    if not retained.any():
        raise ValueError("Personal epoch QC left no usable evidence epochs; abstention is required.")

    timeline = dataset.event_timeline
    evidence = np.asarray(timeline.evidence_indices, dtype=np.int64).copy()
    available_events = evidence >= 0
    rejected_events = np.zeros(timeline.n_events, dtype=bool)
    rejected_events[available_events] = rejected[evidence[available_events]]
    new_indices = np.full(dataset.n_epochs, -1, dtype=np.int64)
    new_indices[retained] = np.arange(int(retained.sum()))
    evidence[available_events] = new_indices[evidence[available_events]]
    statuses = np.asarray(timeline.statuses, dtype=object).copy()
    details = np.asarray(timeline.status_details, dtype=object).copy()
    available_times = np.asarray(timeline.evidence_available_times_s, dtype=float).copy()
    statuses[rejected_events] = "artifact_rejected"
    details[rejected_events] = "frozen_personal_epoch_quality"
    available_times[rejected_events] = np.nan
    updated_timeline = replace(
        timeline, evidence_indices=evidence, statuses=statuses.astype(str),
        status_details=details.astype(str), evidence_available_times_s=available_times,
    )
    processing = {
        "status": "applied",
        "model_digest": semantic_sha256(fitted.record()),
        "fit_provenance": fitted.fit_provenance,
        "n_input_epochs": dataset.n_epochs,
        "n_retained_epochs": int(retained.sum()),
        "n_rejected_epochs": int(rejected.sum()),
        "n_scheduled_events": timeline.n_events,
        "rejected_event_ids": np.asarray(timeline.event_ids)[rejected_events].astype(str).tolist(),
        "quality_summary": transformed.summary(),
    }
    values = transformed.X[retained]
    trial_mask = transformed.trial_channel_mask[retained]
    identity = materialize_dataset_identity(replace(dataset))
    lineage = materialize_dataset_lineage(replace(dataset))
    result = replace(
        dataset,
        X=values,
        y=None if dataset.y is None else np.asarray(dataset.y)[retained],
        subject_ids=np.asarray(dataset.subject_ids)[retained],
        trial_channel_mask=trial_mask,
        qc_features=compute_epoch_qc_features(
            values, channel_mask=dataset.channel_mask, trial_channel_mask=trial_mask,
        ),
        metadata=(dataset.metadata.iloc[np.flatnonzero(retained)].reset_index(drop=True).copy()
                  if not dataset.metadata.empty else dataset.metadata.copy()),
        event_timeline=updated_timeline,
        provenance={**dataset.provenance, "epoch_artifact_processing": processing},
        identity_table=identity,
        lineage=DataLineage.derive(
            (lineage,), operation="personal_epoch_quality", parameters=processing,
        ),
        verified_cache_attestation=None,
    )
    result.validate()
    return result
