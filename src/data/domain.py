"""Deterministic cross-dataset channel and reference alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from data.channel import build_channel_identity, canonical_channel_name
from data.epochs import (
    EpochDataset,
    materialize_dataset_identity,
    materialize_dataset_lineage,
)
from data.lineage import DataLineage
from data.qc_features import compute_epoch_qc_features

DOMAIN_ADAPTER_SCHEMA = "n2p3_common_channel_car/1"
NATIVE_REFERENCE_SCHEMA = "n2p3_native_reference_selection/1"
OBSERVED_MONTAGE_SCHEMA = "n2p3_observed_montage/1"
BINARY_EVIDENCE_PROJECTION_SCHEMA = "n2p3_binary_evidence_source_view/2"
SOURCE_DOMAIN_AXIS_SCHEMA = "n2p3_source_domain_axis/1"
SOURCE_DOMAIN_COLUMN = "source_domain"
NAMESPACE_SEPARATOR = "::"
CAR_FLOAT32_ROUNDOFF_FACTOR = 4.0
CAR_VALIDATION_BLOCK_BYTES = 64 * 1024 * 1024


def common_channel_intersection(*datasets: EpochDataset) -> tuple[str, ...]:
    """Return canonical channels shared by every dataset in first-dataset order."""

    if len(datasets) < 2:
        raise ValueError("at least two datasets are required for a channel intersection.")
    for dataset in datasets:
        dataset.validate()
    common = set(datasets[0].channel_names)
    for dataset in datasets[1:]:
        common &= set(dataset.channel_names)
    return tuple(channel for channel in datasets[0].channel_names if channel in common)


def adapt_common_channel_average_reference(
    dataset: EpochDataset,
    target_channels: Sequence[str],
    *,
    name: str | None = None,
) -> EpochDataset:
    """Select a complete common montage and apply the same CAR in every domain."""

    dataset.validate(require_labels=dataset.y is not None)
    identity_table = materialize_dataset_identity(dataset)
    parent_lineage = materialize_dataset_lineage(dataset)
    channels = tuple(canonical_channel_name(value) for value in target_channels)
    if len(channels) < 2 or len(set(channels)) != len(channels):
        raise ValueError("target_channels must contain at least two unique channels.")
    if _existing_common_car_matches(dataset, channels):
        raise ValueError(
            "dataset already satisfies this common-channel CAR contract; "
            "use ensure_common_channel_average_reference to verify and reuse it."
        )
    indices_by_name = {name: index for index, name in enumerate(dataset.channel_names)}
    missing = [channel for channel in channels if channel not in indices_by_name]
    if missing:
        raise ValueError(f"dataset {dataset.name!r} lacks common channels {missing}.")
    indices = np.asarray([indices_by_name[channel] for channel in channels], dtype=np.int64)
    if dataset.trial_channel_mask is None:
        observed = np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])[:, indices]
    else:
        observed = np.asarray(dataset.trial_channel_mask, dtype=bool)[:, indices]
        observed &= np.asarray(dataset.channel_mask, dtype=bool)[indices][None, :]
    if not bool(observed.all()):
        unavailable = np.argwhere(~observed)
        first = unavailable[0].tolist()
        raise ValueError(
            "common-channel CAR requires every selected channel in every trial; "
            f"first missing trial/channel index={first}."
        )

    selected = np.asarray(dataset.X[:, indices, :], dtype=np.float64)
    values = (selected - selected.mean(axis=1, keepdims=True)).astype(np.float32)
    reference = "common_average[" + ",".join(channels) + "]"
    adapted = EpochDataset(
        name=name or f"{dataset.name}-common-car",
        X=np.ascontiguousarray(values, dtype=np.float32),
        y=None if dataset.y is None else np.asarray(dataset.y, dtype=np.int64),
        subject_ids=np.asarray(dataset.subject_ids).astype(str),
        channel_names=channels,
        channel_positions_m=np.asarray(dataset.channel_positions_m[indices], dtype=np.float32),
        channel_mask=np.ones(len(channels), dtype=bool),
        preprocessing=dataset.preprocessing,
        event_timeline=dataset.event_timeline,
        metadata=dataset.metadata.copy(),
        provenance={
            **dataset.provenance,
            "source": "explicit_common_channel_car_adapter",
            "parent_dataset_name": dataset.name,
            "parent_source_reference": dataset.provenance.get("source_reference"),
            "source_reference": reference,
            "domain_adapter": {
                "schema": DOMAIN_ADAPTER_SCHEMA,
                "target_channels": list(channels),
                "operation": "select channels then subtract their instantaneous mean",
                "preprocessing_identity": dataset.preprocessing.name,
            },
        },
        trial_channel_mask=None,
        qc_features=compute_epoch_qc_features(
            values,
            channel_mask=np.ones(len(channels), dtype=bool),
        ),
        identity_table=identity_table,
        lineage=DataLineage.derive(
            [parent_lineage],
            operation="common_channel_average_reference",
            parameters={
                "target_channels": list(channels),
                "reference": reference,
            },
        ),
    )
    adapted.validate(require_labels=dataset.y is not None)
    return adapted


def adapt_native_reference_selection(
    dataset: EpochDataset,
    target_channels: Sequence[str],
    *,
    name: str | None = None,
) -> EpochDataset:
    """Select a complete common montage without changing the reference.

    CAR is a reference *normalization*, not a lossless alignment. When two
    sources already declare the same physical reference, forcing CAR removes a
    spatial degree of freedom and has been observed to hurt task information in
    local ablations. This transform is the explicit ``preserve_native`` arm:
    it only selects the common channels and records that no re-referencing was
    performed, so downstream code can never silently mislabel CARed tensors as
    native.
    """

    dataset.validate(require_labels=dataset.y is not None)
    identity_table = materialize_dataset_identity(dataset)
    parent_lineage = materialize_dataset_lineage(dataset)
    channels = tuple(canonical_channel_name(value) for value in target_channels)
    if len(channels) < 2 or len(set(channels)) != len(channels):
        raise ValueError("target_channels must contain at least two unique channels.")
    adapter = dataset.provenance.get("domain_adapter")
    if adapter is not None:
        if not isinstance(adapter, Mapping):
            raise ValueError(
                "dataset declares an unsupported or malformed domain_adapter contract."
            )
        raise ValueError(
            "native reference selection cannot consume an already domain-adapted dataset; "
            f"found adapter schema {adapter.get('schema')!r}."
        )
    parent_reference = dataset.provenance.get("source_reference")
    if not isinstance(parent_reference, str) or not parent_reference.strip():
        raise ValueError(
            "native reference selection requires an explicit non-empty source_reference "
            "in the parent provenance."
        )
    indices_by_name = {name: index for index, name in enumerate(dataset.channel_names)}
    missing = [channel for channel in channels if channel not in indices_by_name]
    if missing:
        raise ValueError(f"dataset {dataset.name!r} lacks common channels {missing}.")
    indices = np.asarray([indices_by_name[channel] for channel in channels], dtype=np.int64)
    if dataset.trial_channel_mask is None:
        observed = np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])[:, indices]
    else:
        observed = np.asarray(dataset.trial_channel_mask, dtype=bool)[:, indices]
        observed &= np.asarray(dataset.channel_mask, dtype=bool)[indices][None, :]
    if not bool(observed.all()):
        unavailable = np.argwhere(~observed)
        first = unavailable[0].tolist()
        raise ValueError(
            "native common-channel selection requires every selected channel in every trial; "
            f"first missing trial/channel index={first}."
        )

    values = np.ascontiguousarray(dataset.X[:, indices, :], dtype=np.float32)
    reference = f"native:{parent_reference}"
    adapted = EpochDataset(
        name=name or f"{dataset.name}-native-selection",
        X=values,
        y=None if dataset.y is None else np.asarray(dataset.y, dtype=np.int64),
        subject_ids=np.asarray(dataset.subject_ids).astype(str),
        channel_names=channels,
        channel_positions_m=np.asarray(dataset.channel_positions_m[indices], dtype=np.float32),
        channel_mask=np.ones(len(channels), dtype=bool),
        preprocessing=dataset.preprocessing,
        event_timeline=dataset.event_timeline,
        metadata=dataset.metadata.copy(),
        provenance={
            **dataset.provenance,
            "source": "explicit_native_reference_channel_selection",
            "parent_dataset_name": dataset.name,
            "parent_source_reference": parent_reference,
            "source_reference": reference,
            "domain_adapter": {
                "schema": NATIVE_REFERENCE_SCHEMA,
                "target_channels": list(channels),
                "operation": "select channels without re-referencing",
                "preserved_source_reference": parent_reference,
                "preprocessing_identity": dataset.preprocessing.name,
            },
        },
        trial_channel_mask=None,
        qc_features=compute_epoch_qc_features(
            values,
            channel_mask=np.ones(len(channels), dtype=bool),
        ),
        identity_table=identity_table,
        lineage=DataLineage.derive(
            [parent_lineage],
            operation="native_reference_channel_selection",
            parameters={
                "target_channels": list(channels),
                "reference": reference,
            },
        ),
    )
    adapted.validate(require_labels=dataset.y is not None)
    return adapted


def _existing_native_selection_matches(
    dataset: EpochDataset,
    channels: tuple[str, ...],
) -> bool:
    adapter = dataset.provenance.get("domain_adapter")
    if adapter is None:
        return False
    if not isinstance(adapter, Mapping) or adapter.get("schema") != NATIVE_REFERENCE_SCHEMA:
        raise ValueError("dataset declares an unsupported or malformed domain_adapter contract.")
    raw_claimed_channels = adapter.get("target_channels")
    if not isinstance(raw_claimed_channels, Sequence) or isinstance(
        raw_claimed_channels,
        (str, bytes, bytearray),
    ):
        raise ValueError("domain_adapter.target_channels must be a channel-name sequence.")
    claimed_channels = tuple(canonical_channel_name(value) for value in raw_claimed_channels)
    parent_reference = adapter.get("preserved_source_reference")
    if not isinstance(parent_reference, str) or not parent_reference.strip():
        raise ValueError("native reference adapter lacks its preserved source reference.")
    reference = f"native:{parent_reference}"
    mismatches: list[str] = []
    if claimed_channels != channels:
        return False
    if dataset.channel_names != channels:
        mismatches.append("channel_names")
    if not bool(np.asarray(dataset.channel_mask, dtype=bool).all()):
        mismatches.append("channel_mask")
    if dataset.trial_channel_mask is not None:
        mismatches.append("trial_channel_mask")
    if dataset.provenance.get("source_reference") != reference:
        mismatches.append("source_reference")
    if adapter.get("preprocessing_identity") != dataset.preprocessing.name:
        mismatches.append("preprocessing_identity")
    expected_lineage_parameters = {
        "target_channels": list(channels),
        "reference": reference,
    }
    lineage = materialize_dataset_lineage(dataset)
    if not any(
        entity.operation == "native_reference_channel_selection"
        and dict(entity.parameters) == expected_lineage_parameters
        for entity in lineage.entities
    ):
        mismatches.append("lineage")
    if mismatches:
        raise ValueError(
            "native reference selection provenance disagrees with the dataset contract: "
            f"{mismatches}."
        )
    return True


def ensure_native_reference_selection(
    dataset: EpochDataset,
    target_channels: Sequence[str],
    *,
    name: str | None = None,
) -> EpochDataset:
    """Apply native channel selection once, or prove an existing one is complete."""

    dataset.validate(require_labels=dataset.y is not None)
    channels = tuple(canonical_channel_name(value) for value in target_channels)
    if len(channels) < 2 or len(set(channels)) != len(channels):
        raise ValueError("target_channels must contain at least two unique channels.")
    if _existing_native_selection_matches(dataset, channels):
        if name is not None and dataset.name != name:
            raise ValueError(
                "an already-adapted dataset cannot be renamed by an idempotent native check."
            )
        return dataset
    return adapt_native_reference_selection(dataset, channels, name=name)


def _existing_common_car_matches(
    dataset: EpochDataset,
    channels: tuple[str, ...],
) -> bool:
    adapter = dataset.provenance.get("domain_adapter")
    if adapter is None:
        return False
    if not isinstance(adapter, Mapping) or adapter.get("schema") != DOMAIN_ADAPTER_SCHEMA:
        raise ValueError("dataset declares an unsupported or malformed domain_adapter contract.")
    raw_claimed_channels = adapter.get("target_channels")
    if not isinstance(raw_claimed_channels, Sequence) or isinstance(
        raw_claimed_channels,
        (str, bytes, bytearray),
    ):
        raise ValueError("domain_adapter.target_channels must be a channel-name sequence.")
    claimed_channels = tuple(canonical_channel_name(value) for value in raw_claimed_channels)
    if claimed_channels != channels:
        return False

    reference = "common_average[" + ",".join(channels) + "]"
    mismatches: list[str] = []
    if dataset.channel_names != channels:
        mismatches.append("channel_names")
    if not bool(np.asarray(dataset.channel_mask, dtype=bool).all()):
        mismatches.append("channel_mask")
    if dataset.trial_channel_mask is not None:
        mismatches.append("trial_channel_mask")
    if dataset.provenance.get("source_reference") != reference:
        mismatches.append("source_reference")
    if adapter.get("preprocessing_identity") != dataset.preprocessing.name:
        mismatches.append("preprocessing_identity")
    expected_lineage_parameters = {
        "target_channels": list(channels),
        "reference": reference,
    }
    lineage = materialize_dataset_lineage(dataset)
    if not any(
        entity.operation == "common_channel_average_reference"
        and dict(entity.parameters) == expected_lineage_parameters
        for entity in lineage.entities
    ):
        mismatches.append("lineage")
    if mismatches:
        raise ValueError(
            f"common-channel CAR provenance disagrees with the dataset contract: {mismatches}."
        )

    maximum_relative_channel_mean = 0.0
    values_per_epoch = max(1, dataset.n_channels * dataset.n_times)
    epochs_per_block = max(
        1,
        CAR_VALIDATION_BLOCK_BYTES // (values_per_epoch * np.dtype(np.float64).itemsize),
    )
    for start in range(0, dataset.n_epochs, epochs_per_block):
        block = dataset.X[start : start + epochs_per_block]
        values = np.asarray(block, dtype=np.float64)
        scale = np.max(np.abs(values), axis=1)
        mean = np.abs(values.mean(axis=1))
        relative_mean = np.divide(
            mean,
            scale,
            out=np.zeros_like(mean),
            where=scale > 0.0,
        )
        maximum_relative_channel_mean = max(
            maximum_relative_channel_mean,
            float(np.max(relative_mean)),
        )
    tolerance = CAR_FLOAT32_ROUNDOFF_FACTOR * np.finfo(np.float32).eps * len(channels)
    if maximum_relative_channel_mean > tolerance:
        raise ValueError(
            "common-channel CAR provenance is contradicted by non-zero channel means: "
            f"max_relative_mean={maximum_relative_channel_mean:g}, tolerance={tolerance:g}."
        )
    return True


def ensure_common_channel_average_reference(
    dataset: EpochDataset,
    target_channels: Sequence[str],
    *,
    name: str | None = None,
) -> EpochDataset:
    """Apply CAR once, or prove that an existing identical CAR can be reused."""

    dataset.validate(require_labels=dataset.y is not None)
    channels = tuple(canonical_channel_name(value) for value in target_channels)
    if len(channels) < 2 or len(set(channels)) != len(channels):
        raise ValueError("target_channels must contain at least two unique channels.")
    if _existing_common_car_matches(dataset, channels):
        if name is not None and dataset.name != name:
            raise ValueError(
                "an already-adapted dataset cannot be renamed by an idempotent CAR check."
            )
        return dataset
    return adapt_common_channel_average_reference(dataset, channels, name=name)


def adapt_observed_montage(
    dataset: EpochDataset,
    target_channels: Sequence[str],
    *,
    name: str | None = None,
    reference_channel: str | None = None,
) -> EpochDataset:
    """Project observed EEG, optionally using a common measured reference."""
    dataset.validate(require_labels=dataset.y is not None)
    identity = build_channel_identity(target_channels, allow_missing_positions=False)
    channels = identity.names
    anchor = None if reference_channel is None else canonical_channel_name(reference_channel)
    if (
        anchor is not None
        and dataset.provenance.get("ocular_artifact_processing", {}).get("status") == "applied"
    ):
        raise ValueError(
            "choose the final reference before fitting and applying ocular regression."
        )
    if (
        anchor is not None
        and dataset.preprocessing.baseline_mode == "trial_reference"
        and dataset.preprocessing.trial_reference_center != "mean"
    ):
        raise ValueError(
            "measured reference invariance requires linear baseline preprocessing or raw rebuilding."
        )
    if len(channels) < 2:
        raise ValueError("a masked montage requires at least two declared channels.")
    previous = dataset.provenance.get("domain_adapter")
    if previous is not None:
        previous_reference = previous.get("reference", {}) if isinstance(previous, Mapping) else {}
        if not isinstance(previous_reference, Mapping):
            raise ValueError("the existing montage reference record is malformed.")
        reference_matches = (
            previous_reference == {"mode": "measured_channel", "channel": anchor}
            if anchor is not None
            else isinstance(previous_reference, Mapping)
            and previous_reference.get("mode") == "native_preserve"
            and isinstance(previous_reference.get("source_reference"), str)
            and bool(previous_reference["source_reference"].strip())
        )
        expected_reference = (
            "measured_channel:" + anchor
            if anchor is not None
            else "native:" + str(previous_reference.get("source_reference"))
        )
        if (
            isinstance(previous, Mapping)
            and previous.get("schema") == OBSERVED_MONTAGE_SCHEMA
            and previous.get("target_channels") == list(channels)
            and previous.get("operation") == "select_reference_and_observation_mask"
            and previous.get("preprocessing_identity") == dataset.preprocessing.name
            and reference_matches
            and dataset.channel_names == channels
            and np.allclose(dataset.channel_positions_m, identity.coords, atol=1e-6)
            and dataset.provenance.get("source_reference") == expected_reference
            and any(
                entity.operation == "observed_montage_projection"
                and dict(entity.parameters) == dict(previous)
                for entity in materialize_dataset_lineage(dataset).entities
            )
        ):
            if name is not None and name != dataset.name:
                raise ValueError("an existing masked montage cannot be implicitly renamed.")
            return dataset
        raise ValueError("masked montage requires raw native input or an identical projection.")
    reference = dataset.provenance.get("source_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("masked montage requires an explicit native reference.")
    source_positions = {channel: index for index, channel in enumerate(dataset.channel_names)}
    source_mask = (
        np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
        if dataset.trial_channel_mask is None
        else dataset.trial_channel_mask
    )
    reference_signal = None
    if anchor is not None:
        anchor_index = source_positions.get(anchor)
        if anchor_index is None or not bool(source_mask[:, anchor_index].all()):
            raise ValueError("the measured reference must be observed in every projected trial.")
        reference_signal = dataset.X[:, anchor_index]
    values = np.zeros((dataset.n_epochs, len(channels), dataset.n_times), dtype=np.float32)
    observed = np.zeros(values.shape[:2], dtype=bool)
    for destination, channel in enumerate(channels):
        index = source_positions.get(channel)
        if index is None:
            continue
        if bool(source_mask[:, index].any()) and not np.allclose(
            dataset.channel_positions_m[index], identity.coords[destination], atol=1e-6
        ):
            raise ValueError(
                "observed source coordinates differ from the declared standard montage."
            )
        values[:, destination] = dataset.X[:, index]
        if reference_signal is not None:
            values[:, destination] -= reference_signal
        observed[:, destination] = source_mask[:, index]
    if not bool(observed.any(axis=1).all()):
        raise ValueError("each projected trial must retain a measured EEG channel.")
    static_mask = observed.any(axis=0)
    np.copyto(values, 0.0, where=~observed[:, :, None])
    reference_record = (
        {"mode": "native_preserve", "source_reference": reference}
        if anchor is None
        else {"mode": "measured_channel", "channel": anchor}
    )
    output_reference = "native:" + reference if anchor is None else "measured_channel:" + anchor
    adapter = {
        "schema": OBSERVED_MONTAGE_SCHEMA,
        "target_channels": list(channels),
        "operation": "select_reference_and_observation_mask",
        "reference": reference_record,
        "preprocessing_identity": dataset.preprocessing.name,
    }
    output = replace(
        dataset,
        name=name or dataset.name + "-masked-montage",
        X=values,
        channel_names=channels,
        channel_positions_m=identity.coords,
        channel_mask=static_mask,
        trial_channel_mask=observed,
        qc_features=compute_epoch_qc_features(
            values, channel_mask=static_mask, trial_channel_mask=observed
        ),
        identity_table=materialize_dataset_identity(dataset),
        provenance={
            **dataset.provenance,
            "source_reference": output_reference,
            "parent_source_reference": reference,
            "domain_adapter": adapter,
            "source_acquisition_channels": list(dataset.channel_names),
            "missing_target_channels": [
                channel for index, channel in enumerate(channels) if not static_mask[index]
            ],
            "unobserved_values_are_placeholders": True,
        },
        lineage=DataLineage.derive(
            [materialize_dataset_lineage(dataset)],
            operation="observed_montage_projection",
            parameters=adapter,
        ),
        verified_cache_attestation=None,
    )
    output.validate(require_labels=dataset.y is not None)
    return output


def project_binary_evidence_source_view(dataset: EpochDataset) -> EpochDataset:
    """Drop candidate-only fields explicitly for binary supervised source pretraining."""

    dataset.validate(require_labels=True)
    labels = set(np.asarray(dataset.y, dtype=np.int64).tolist())
    if not labels <= {0, 1}:
        raise ValueError("binary evidence projection requires labels encoded as 0/1.")
    timeline = dataset.event_timeline
    candidate_columns = (
        "candidate_id",
        "candidate_ids",
        "target_candidate_id",
        "target_candidate_ids",
        "row_code",
        "col_code",
        "target_row",
        "target_col",
        "repetition_index",
        "repetition_indices",
    )
    dropped_columns = [column for column in candidate_columns if column in dataset.metadata]
    previous = dataset.provenance.get("binary_evidence_projection")
    if (
        isinstance(previous, Mapping)
        and previous.get("schema") == BINARY_EVIDENCE_PROJECTION_SCHEMA
        and dataset.provenance.get("event_contract") == "binary_evidence"
        and "candidate_task_contract" not in dataset.provenance
        and not dropped_columns
        and not timeline.has_candidate_ids
        and not timeline.has_candidate_sets
        and not timeline.has_repetition_structure
    ):
        return dataset

    parent_lineage = materialize_dataset_lineage(dataset)
    source_fingerprint = timeline.fingerprint()
    projected_timeline = replace(
        timeline,
        candidate_ids=None,
        target_candidate_ids=None,
        repetition_indices=None,
    ).validate(n_epochs=dataset.n_epochs)
    projection_record = {
        "schema": BINARY_EVIDENCE_PROJECTION_SCHEMA,
        "purpose": "binary_supervised_source_pretraining",
        "dropped_fields": [
            name
            for name, present in (
                ("candidate_ids", timeline.has_candidate_ids),
                ("target_candidate_ids", timeline.has_candidate_sets),
                ("repetition_indices", timeline.has_repetition_structure),
            )
            if present
        ],
        "source_event_fingerprint": source_fingerprint,
        "projected_event_fingerprint": projected_timeline.fingerprint(),
        "tensor_unchanged": True,
        "labels_unchanged": True,
        "dropped_metadata_columns": dropped_columns,
        "source_candidate_task_contract": dataset.provenance.get("candidate_task_contract"),
        "source_event_contract": dataset.provenance.get("event_contract"),
        "grouping_scope": "preserved_acquisition_provenance_not_candidate_supervision",
    }
    provenance = dict(dataset.provenance)
    provenance.pop("candidate_task_contract", None)
    projected = replace(
        dataset,
        event_timeline=projected_timeline,
        metadata=dataset.metadata.drop(columns=dropped_columns).copy(),
        provenance={
            **provenance,
            "event_contract": "binary_evidence",
            "binary_evidence_projection": projection_record,
        },
        lineage=DataLineage.derive(
            [parent_lineage],
            operation="project_binary_evidence_source_view",
            parameters=projection_record,
        ),
        verified_cache_attestation=None,
    )
    projected.validate(require_labels=True)
    return projected


def annotate_source_domain(dataset: EpochDataset, domain_id: str) -> EpochDataset:
    """Attach one explicit training-domain value to every epoch row."""

    dataset.validate(require_labels=dataset.y is not None)
    domain = str(domain_id).strip()
    if not domain or "\0" in domain:
        raise ValueError("source domain id must be non-empty and cannot contain NUL.")
    existing = dataset.provenance.get("source_domain_axis")
    has_column = SOURCE_DOMAIN_COLUMN in dataset.metadata
    if existing is not None or has_column:
        if not isinstance(existing, Mapping) or existing != {
            "schema": SOURCE_DOMAIN_AXIS_SCHEMA,
            "column": SOURCE_DOMAIN_COLUMN,
            "domains": [domain],
        }:
            raise ValueError("existing source-domain metadata is absent, forged, or conflicting.")
        values = dataset.metadata[SOURCE_DOMAIN_COLUMN].astype(str).to_numpy()
        if values.shape != (dataset.n_epochs,) or not bool((values == domain).all()):
            raise ValueError("source-domain metadata column disagrees with its provenance.")
        return dataset

    metadata = dataset.metadata.copy()
    if metadata.empty:
        metadata = metadata.reindex(range(dataset.n_epochs))
        metadata["subject"] = np.asarray(dataset.subject_ids).astype(str)
    if len(metadata) != dataset.n_epochs:
        raise ValueError("metadata must align with epoch rows before domain annotation.")
    metadata[SOURCE_DOMAIN_COLUMN] = domain
    record = {
        "schema": SOURCE_DOMAIN_AXIS_SCHEMA,
        "column": SOURCE_DOMAIN_COLUMN,
        "domains": [domain],
    }
    annotated = replace(
        dataset,
        metadata=metadata,
        provenance={**dataset.provenance, "source_domain_axis": record},
        lineage=DataLineage.derive(
            [materialize_dataset_lineage(dataset)],
            operation="annotate_source_domain",
            parameters=record,
        ),
        verified_cache_attestation=None,
    )
    annotated.validate(require_labels=dataset.y is not None)
    return annotated


def source_domain_vocabulary(dataset: EpochDataset) -> tuple[str, ...]:
    """Return the cache-declared domain order used to encode model indices."""

    dataset.validate(require_labels=dataset.y is not None)
    record = dataset.provenance.get("source_domain_axis")
    if not isinstance(record, Mapping) or record.get("schema") != SOURCE_DOMAIN_AXIS_SCHEMA:
        raise ValueError("dataset lacks the explicit source-domain axis contract.")
    if record.get("column") != SOURCE_DOMAIN_COLUMN:
        raise ValueError("source-domain axis declares an unsupported metadata column.")
    raw_domains = record.get("domains")
    if not isinstance(raw_domains, Sequence) or isinstance(raw_domains, (str, bytes, bytearray)):
        raise ValueError("source-domain axis domains must be a sequence.")
    domains = tuple(str(value).strip() for value in raw_domains)
    if not domains or any(not value for value in domains) or len(set(domains)) != len(domains):
        raise ValueError("source-domain axis domains must be unique non-empty strings.")
    return domains


def source_domain_ids(dataset: EpochDataset) -> np.ndarray:
    """Return the validated explicit row-domain axis of a prepared source cache."""

    domains = source_domain_vocabulary(dataset)
    if SOURCE_DOMAIN_COLUMN not in dataset.metadata:
        raise ValueError("source-domain axis metadata column is missing.")
    values = dataset.metadata[SOURCE_DOMAIN_COLUMN].astype(str).to_numpy()
    if values.shape != (dataset.n_epochs,) or set(values.tolist()) != set(domains):
        raise ValueError("source-domain row values disagree with declared domains.")
    return values


def namespace_epoch_dataset(
    dataset: EpochDataset,
    namespace: str,
    *,
    name: str | None = None,
) -> EpochDataset:
    """Qualify participant/event identity before cross-dataset concatenation."""

    dataset.validate(require_labels=dataset.y is not None)
    identity_table = materialize_dataset_identity(dataset)
    parent_lineage = materialize_dataset_lineage(dataset)
    prefix = str(namespace).strip()
    if not prefix or "\0" in prefix:
        raise ValueError("namespace must be non-empty and cannot contain NUL.")
    existing_namespace = dataset.provenance.get("subject_namespace")
    if existing_namespace is not None:
        raise ValueError(
            f"dataset already declares subject_namespace={existing_namespace!r}; "
            "use ensure_epoch_dataset_namespace to verify and reuse it."
        )
    subject_ids = np.asarray(
        [
            f"{prefix}{NAMESPACE_SEPARATOR}{value}"
            for value in np.asarray(dataset.subject_ids).astype(str)
        ]
    )
    subject_mapping = {
        value: f"{prefix}{NAMESPACE_SEPARATOR}{value}" for value in identity_table.local_subject_ids
    }
    timeline = replace(
        dataset.event_timeline,
        event_ids=np.asarray(
            [f"{prefix}{NAMESPACE_SEPARATOR}{value}" for value in dataset.event_timeline.event_ids]
        ),
        group_ids=np.asarray(
            [f"{prefix}{NAMESPACE_SEPARATOR}{value}" for value in dataset.event_timeline.group_ids]
        ),
        subject_ids=np.asarray(
            [
                f"{prefix}{NAMESPACE_SEPARATOR}{value}"
                for value in dataset.event_timeline.subject_ids
            ]
        ),
        dataset_ids=np.asarray(
            [
                f"{prefix}{NAMESPACE_SEPARATOR}{value}"
                for value in dataset.event_timeline.dataset_ids
            ]
        ),
    ).validate(n_epochs=dataset.n_epochs)
    metadata = dataset.metadata.copy()
    if "subject" in metadata:
        metadata["subject"] = subject_ids
    namespaced = EpochDataset(
        name=name or f"{prefix}{NAMESPACE_SEPARATOR}{dataset.name}",
        X=np.asarray(dataset.X, dtype=np.float32),
        y=None if dataset.y is None else np.asarray(dataset.y, dtype=np.int64),
        subject_ids=subject_ids,
        channel_names=dataset.channel_names,
        channel_positions_m=np.asarray(dataset.channel_positions_m, dtype=np.float32),
        channel_mask=np.asarray(dataset.channel_mask, dtype=bool),
        preprocessing=dataset.preprocessing,
        event_timeline=timeline,
        metadata=metadata,
        provenance={
            **dataset.provenance,
            "parent_dataset_name": dataset.name,
            "subject_namespace": prefix,
        },
        trial_channel_mask=(
            None
            if dataset.trial_channel_mask is None
            else np.asarray(dataset.trial_channel_mask, dtype=bool)
        ),
        qc_features=dataset.qc_features,
        identity_table=identity_table.relabel_local_subjects(subject_mapping),
        lineage=DataLineage.derive(
            [parent_lineage],
            operation="namespace_subject_axis",
            parameters={"namespace": prefix},
        ),
    )
    namespaced.validate(require_labels=dataset.y is not None)
    return namespaced


def ensure_epoch_dataset_namespace(
    dataset: EpochDataset,
    namespace: str,
    *,
    name: str | None = None,
) -> EpochDataset:
    """Apply one namespace or prove that the requested namespace is already complete."""

    dataset.validate(require_labels=dataset.y is not None)
    prefix = str(namespace).strip()
    if not prefix or "\0" in prefix:
        raise ValueError("namespace must be non-empty and cannot contain NUL.")
    existing = dataset.provenance.get("subject_namespace")
    if existing is None:
        return namespace_epoch_dataset(dataset, prefix, name=name)
    if existing != prefix:
        raise ValueError(
            f"dataset declares subject_namespace={existing!r}, not requested {prefix!r}."
        )
    if name is not None and dataset.name != name:
        raise ValueError(
            "an already-namespaced dataset cannot be renamed by an idempotent namespace check."
        )

    qualified_prefix = f"{prefix}{NAMESPACE_SEPARATOR}"
    axes = {
        "subject_ids": np.asarray(dataset.subject_ids).astype(str),
        "event_timeline.event_ids": np.asarray(dataset.event_timeline.event_ids).astype(str),
        "event_timeline.group_ids": np.asarray(dataset.event_timeline.group_ids).astype(str),
        "event_timeline.subject_ids": np.asarray(dataset.event_timeline.subject_ids).astype(str),
        "event_timeline.dataset_ids": np.asarray(dataset.event_timeline.dataset_ids).astype(str),
    }
    identity_table = materialize_dataset_identity(dataset)
    axes["identity_table.local_subject_ids"] = np.asarray(
        identity_table.local_subject_ids,
        dtype=str,
    )
    malformed = {
        axis: values[~np.char.startswith(values, qualified_prefix)].tolist()
        for axis, values in axes.items()
        if not bool(np.char.startswith(values, qualified_prefix).all())
    }
    if malformed:
        raise ValueError(
            "dataset subject_namespace provenance is inconsistent with qualified identity axes: "
            f"{malformed}"
        )
    if "subject" in dataset.metadata and not np.array_equal(
        dataset.metadata["subject"].astype(str).to_numpy(),
        np.asarray(dataset.subject_ids).astype(str),
    ):
        raise ValueError("namespaced metadata.subject disagrees with subject_ids.")
    return dataset
