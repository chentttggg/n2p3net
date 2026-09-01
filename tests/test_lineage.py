from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.channel import build_channel_identity
from data.domain import adapt_common_channel_average_reference, namespace_epoch_dataset
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
    load_epoch_dataset,
    materialize_dataset_lineage,
    save_epoch_dataset,
)
from data.events import ScheduledEventTimeline
from data.lineage import DataLineage, LineageEntity


def _dataset(*, subject: str = "s", reference: str = "A1") -> EpochDataset:
    channels = ("Fz", "Cz", "Pz")
    channel_identity = build_channel_identity(channels, allow_missing_positions=False)
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"{subject}:event"]),
        group_ids=np.asarray([f"{subject}:decision"]),
        subject_ids=np.asarray([subject]),
        stimulus_ids=np.asarray([1], dtype=np.int64),
        onset_samples=np.asarray([10], dtype=np.int64),
        onset_times_s=np.asarray([1.0]),
        evidence_available_times_s=np.asarray([2.0]),
        evidence_indices=np.asarray([0], dtype=np.int64),
        statuses=np.asarray(["available"]),
        status_details=np.asarray([""]),
        dataset_ids=np.asarray(["toy-release"]),
        session_ids=np.asarray(["session"]),
        run_ids=np.asarray(["run"]),
        selection_ids=np.asarray(["decision"]),
        complete=True,
        online_causal=True,
        timing_source="toy",
    )
    return EpochDataset(
        name=f"toy-{subject}",
        X=np.asarray([[[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]]], dtype=np.float32),
        y=np.asarray([1], dtype=np.int64),
        subject_ids=np.asarray([subject]),
        channel_names=channel_identity.names,
        channel_positions_m=channel_identity.coords,
        channel_mask=np.ones(3, dtype=bool),
        preprocessing=PreprocessingSpec(
            name="toy-causal",
            sfreq=4.0,
            l_freq=None,
            h_freq=None,
            tmin_ms=-250.0,
            tmax_ms=750.0,
            n_times=4,
            filter_phase="forward",
            causal_iir_initial_state="steady_state_first_sample",
        ),
        event_timeline=timeline,
        metadata=pd.DataFrame({"subject": [subject]}),
        provenance={"source": "toy", "source_reference": reference},
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
