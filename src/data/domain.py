"""Deterministic cross-dataset channel and reference alignment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from data.channel import canonical_channel_name
from data.epochs import (
    EpochDataset,
    materialize_dataset_identity,
    materialize_dataset_lineage,
)
from data.lineage import DataLineage
from data.qc_features import compute_epoch_qc_features

DOMAIN_ADAPTER_SCHEMA = "n2p3_common_channel_car/1"


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
    subject_ids = np.asarray(
        [f"{prefix}::{value}" for value in np.asarray(dataset.subject_ids).astype(str)]
    )
    subject_mapping = {
        value: f"{prefix}::{value}"
        for value in identity_table.local_subject_ids
    }
    timeline = replace(
        dataset.event_timeline,
        event_ids=np.asarray(
            [f"{prefix}::{value}" for value in dataset.event_timeline.event_ids]
        ),
        group_ids=np.asarray(
            [f"{prefix}::{value}" for value in dataset.event_timeline.group_ids]
        ),
        subject_ids=np.asarray(
            [f"{prefix}::{value}" for value in dataset.event_timeline.subject_ids]
        ),
        dataset_ids=np.asarray(
            [f"{prefix}::{value}" for value in dataset.event_timeline.dataset_ids]
        ),
    ).validate(n_epochs=dataset.n_epochs)
    metadata = dataset.metadata.copy()
    if "subject" in metadata:
        metadata["subject"] = subject_ids
    namespaced = EpochDataset(
        name=name or f"{prefix}::{dataset.name}",
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
