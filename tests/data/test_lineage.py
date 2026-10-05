from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.domain import adapt_common_channel_average_reference, namespace_epoch_dataset
from data.epochs import (
    EpochDataset,
    concatenate_epoch_datasets,
    load_epoch_dataset,
    materialize_dataset_lineage,
    save_epoch_dataset,
)
from data.lineage import DataLineage, LineageEntity
from tests.support.datasets import reference_dataset


def _dataset(*, subject: str = "s", reference: str = "A1") -> EpochDataset:
    dataset = reference_dataset(
        np.asarray([[[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]]], dtype=np.float32),
        reference=reference,
    )
    subjects = np.asarray([subject])
    return replace(
        dataset,
        name=f"toy-{subject}",
        y=np.asarray([1], dtype=np.int64),
        subject_ids=subjects,
        preprocessing=replace(dataset.preprocessing, name="toy-causal"),
        event_timeline=replace(
            dataset.event_timeline,
            event_ids=np.asarray([f"{subject}:event"]),
            group_ids=np.asarray([f"{subject}:decision"]),
            subject_ids=subjects,
            stimulus_ids=np.asarray([1]),
            onset_samples=np.asarray([10]),
            onset_times_s=np.asarray([1.0]),
            evidence_available_times_s=np.asarray([2.0]),
            dataset_ids=np.asarray(["toy-release"]),
            selection_ids=np.asarray(["decision"]),
        ),
        metadata=pd.DataFrame({"subject": subjects}),
    )


def test_lineage_digest_binds_operation_parameters_and_parent() -> None:
    source = DataLineage.source(parameters={"source": "toy", "event": "a"})
    first = DataLineage.derive(
        [source], operation="select_channels", parameters={"channels": ["CZ", "PZ"]}
    )
    changed = DataLineage.derive(
        [source], operation="select_channels", parameters={"channels": ["FZ", "PZ"]}
    )
    assert first.entity_digest != changed.entity_digest
    restored = DataLineage.from_payload(first.payload())
    assert restored.payload() == first.payload()
    corrupted = first.payload()
    corrupted["entities"][-1]["parameters"] = {"channels": ["FZ"]}  # type: ignore[index]
    with pytest.raises(ValueError, match="digest"):
        DataLineage.from_payload(corrupted)


def test_source_lineage_binds_stable_reference_provenance() -> None:
    first = materialize_dataset_lineage(_dataset(reference="A1"))
    second = materialize_dataset_lineage(_dataset(reference="A2"))
    assert first.entity_digest != second.entity_digest


def test_domain_steps_append_lineage_without_changing_identity() -> None:
    dataset = _dataset()
    source = materialize_dataset_lineage(dataset)
    car = adapt_common_channel_average_reference(dataset, ("Fz", "Cz", "Pz"))
    namespaced = namespace_epoch_dataset(car, "TOY")
    assert car.lineage is not None and namespaced.lineage is not None
    assert car.lineage.entities[-1].operation == "common_channel_average_reference"
    assert namespaced.lineage.entities[-1].operation == "namespace_subject_axis"
    assert source.entity_digest in car.lineage.entities[-1].parent_entity_digests
    assert namespaced.identity_table is not None
    assert namespaced.identity_table.record_for("TOY::s").origin_subject_keys


def test_save_load_and_concat_preserve_complete_parent_graph(tmp_path) -> None:
    first = namespace_epoch_dataset(_dataset(subject="s1"), "A")
    second = namespace_epoch_dataset(_dataset(subject="s2"), "B")
    merged = concatenate_epoch_datasets(
        (first, second),
        name="merged",
        provenance={"source": "test_concat", "source_reference": "A1"},
    )
    assert merged.lineage is not None
    assert merged.lineage.entities[-1].operation == "concatenate_epoch_datasets"
    assert len(merged.lineage.entities[-1].parent_entity_digests) == 2
    path = save_epoch_dataset(tmp_path / "merged.npz", merged)
    loaded = load_epoch_dataset(path)
    assert loaded.lineage is not None
    assert loaded.lineage.payload() == merged.lineage.payload()


def test_derived_lineage_rejects_missing_parent_history() -> None:
    orphan = LineageEntity(
        operation="select_channels",
        parent_entity_digests=("a" * 64,),
        parameters={"channels": ["CZ"]},
    )
    with pytest.raises(ValueError, match="unavailable parents"):
        DataLineage(entities=(orphan,))
