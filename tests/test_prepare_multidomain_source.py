from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.channel import build_channel_identity
from data.domain import (
    adapt_common_channel_average_reference,
    ensure_common_channel_average_reference,
    namespace_epoch_dataset,
)
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
    save_epoch_dataset,
)
from data.events import ScheduledEventTimeline
from experiments.prepare_multidomain_source import main


def _source_dataset(
    *,
    source_id: str,
    subject: str,
    offset: float,
    candidate_metadata: bool = False,
) -> EpochDataset:
    channels = ("Cz", "P3", "Pz", "P4", "Oz")
    identity = build_channel_identity(channels, allow_missing_positions=False)
    n_epochs = 2
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"{source_id}:event:{index}" for index in range(n_epochs)]),
        group_ids=(
            np.repeat(f"{source_id}:group", n_epochs)
            if candidate_metadata
            else np.asarray([f"{source_id}:group:{index}" for index in range(n_epochs)])
        ),
        subject_ids=np.repeat(subject, n_epochs),
        stimulus_ids=np.arange(n_epochs, dtype=np.int64),
        onset_samples=np.arange(n_epochs, dtype=np.int64),
        onset_times_s=np.arange(n_epochs, dtype=float),
        evidence_available_times_s=np.arange(n_epochs, dtype=float) + 1.0,
        evidence_indices=np.arange(n_epochs, dtype=np.int64),
        statuses=np.repeat("available", n_epochs),
        status_details=np.repeat("", n_epochs),
        dataset_ids=np.repeat(source_id, n_epochs),
        session_ids=np.repeat("session", n_epochs),
        run_ids=np.repeat("run", n_epochs),
        selection_ids=(
            np.repeat("selection", n_epochs)
            if candidate_metadata
            else np.asarray([f"selection:{index}" for index in range(n_epochs)])
        ),
        complete=True,
        online_causal=True,
        timing_source="synthetic",
        candidate_ids=(np.asarray(["left", "right"]) if candidate_metadata else None),
        target_candidate_ids=(
            np.asarray(["right", "right"]) if candidate_metadata else None
        ),
        repetition_indices=(
            np.asarray([0, 0], dtype=np.int64) if candidate_metadata else None
        ),
    )
    values = np.arange(n_epochs * len(channels) * 4, dtype=np.float32).reshape(
        n_epochs,
        len(channels),
        4,
    )
    return EpochDataset(
        name=source_id,
        X=(values + np.float32(offset)) * np.float32(1e-7),
        y=np.asarray([0, 1], dtype=np.int64),
        subject_ids=np.repeat(subject, n_epochs),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(len(channels), dtype=bool),
        preprocessing=PreprocessingSpec(
            name="causal-test",
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
        metadata=pd.DataFrame({"subject": np.repeat(subject, n_epochs)}),
        provenance={"source": source_id, "source_reference": "earlobe"},
    )


def _prepared_source(
    *,
    source_id: str,
    subject: str,
    offset: float,
    candidate_metadata: bool = False,
) -> EpochDataset:
    car = adapt_common_channel_average_reference(
        _source_dataset(
            source_id=source_id,
            subject=subject,
            offset=offset,
            candidate_metadata=candidate_metadata,
        ),
        ("Cz", "P3", "Pz", "P4", "Oz"),
    )
    return namespace_epoch_dataset(car, source_id)


def test_multidomain_builder_reuses_prepared_car_and_namespace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _prepared_source(source_id="A", subject="01", offset=0.0)
    second = _prepared_source(source_id="B", subject="02", offset=100.0)
    first_path = save_epoch_dataset(tmp_path / "first.npz", first)
    second_path = save_epoch_dataset(tmp_path / "second.npz", second)
    output_path = tmp_path / "merged.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_multidomain_source.py",
            "--source",
            f"A={first_path}",
            "--source",
            f"B={second_path}",
            "--target-channels",
            "Cz,P3,Pz,P4,Oz",
            "--output",
            str(output_path),
        ],
    )

    main()

    merged = load_epoch_dataset(output_path, require_labels=True, validation="attested")
    assert ensure_common_channel_average_reference(
        merged,
        ("Cz", "P3", "Pz", "P4", "Oz"),
    ) is merged
    np.testing.assert_array_equal(merged.X[: first.n_epochs], first.X)
    np.testing.assert_array_equal(merged.X[first.n_epochs :], second.X)
    assert merged.subject_ids.tolist() == ["A::01", "A::01", "B::02", "B::02"]
    assert not any("A::A::" in value or "B::B::" in value for value in merged.subject_ids)
    operations = [entity.operation for entity in merged.lineage.entities]
    assert operations.count("common_channel_average_reference") == 2
    assert operations.count("namespace_subject_axis") == 2
    assert operations.count("concatenate_epoch_datasets") == 1
    source_records = merged.provenance["sources"]
    assert [record["namespace_preexisting"] for record in source_records] == [True, True]
    assert source_records[0]["verified_cache_attestation"]["sha256"] == (
        loaded_epoch_cache_attestation(
            load_epoch_dataset(first_path, validation="attested")
        )["sha256"]
    )
    assert source_records[1]["verified_cache_attestation"]["sha256"] == (
        loaded_epoch_cache_attestation(
            load_epoch_dataset(second_path, validation="attested")
        )["sha256"]
    )
    output_sha256 = loaded_epoch_cache_attestation(merged)["sha256"]
    with pytest.raises(FileExistsError, match="requires a new output path"):
        main()
    assert loaded_epoch_cache_attestation(
        load_epoch_dataset(output_path, validation="attested")
    )["sha256"] == output_sha256


def test_multidomain_builder_requires_explicit_binary_view_for_mixed_event_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate = _prepared_source(
        source_id="CANDIDATE",
        subject="01",
        offset=0.0,
        candidate_metadata=True,
    )
    binary = _prepared_source(source_id="BINARY", subject="02", offset=100.0)
    candidate_path = save_epoch_dataset(tmp_path / "candidate.npz", candidate)
    binary_path = save_epoch_dataset(tmp_path / "binary.npz", binary)
    output_path = tmp_path / "binary-view.npz"
    rejected_path = tmp_path / "implicit-drop.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_multidomain_source.py",
            "--source",
            f"CANDIDATE={candidate_path}",
            "--source",
            f"BINARY={binary_path}",
            "--target-channels",
            "Cz,P3,Pz,P4,Oz",
            "--output",
            str(rejected_path),
        ],
    )
    with pytest.raises(ValueError, match="candidate_ids must be either absent or present"):
        main()
    assert not rejected_path.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_multidomain_source.py",
            "--source",
            f"CANDIDATE={candidate_path}",
            "--source",
            f"BINARY={binary_path}",
            "--target-channels",
            "Cz,P3,Pz,P4,Oz",
            "--event-contract",
            "binary_evidence",
            "--output",
            str(output_path),
        ],
    )

    main()

    merged = load_epoch_dataset(output_path, require_labels=True, validation="attested")
    assert merged.event_timeline.has_candidate_ids is False
    assert merged.event_timeline.has_candidate_sets is False
    assert merged.provenance["event_contract"] == "binary_evidence"
    source_records = merged.provenance["sources"]
    assert [record["candidate_metadata_projected"] for record in source_records] == [
        True,
        False,
    ]
    reloaded_candidate = load_epoch_dataset(candidate_path, validation="attested")
    assert reloaded_candidate.event_timeline.supports_full_candidate_chain is True
    np.testing.assert_array_equal(merged.X[: candidate.n_epochs], candidate.X)
    np.testing.assert_array_equal(merged.X[candidate.n_epochs :], binary.X)
