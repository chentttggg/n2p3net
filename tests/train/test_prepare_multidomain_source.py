from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.domain import (
    adapt_common_channel_average_reference,
    ensure_common_channel_average_reference,
    namespace_epoch_dataset,
    source_domain_ids,
    source_domain_vocabulary,
)
from data.epochs import (
    EpochDataset,
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
    save_epoch_dataset,
)
from experiments.prepare_multidomain_source import main
from experiments.prepare_reference_cache import main as derive_reference_cache_main
from tests.support.datasets import reference_dataset
from tests.support.preparation_cli import set_preparation_cli


def _source_dataset(
    *,
    source_id: str,
    subject: str,
    offset: float,
    channels: tuple[str, ...] = ("Cz", "P3", "Pz", "P4", "Oz"),
) -> EpochDataset:
    n_epochs = 2
    values = np.arange(n_epochs * len(channels) * 4, dtype=np.float32).reshape(
        n_epochs, len(channels), 4
    )
    dataset = reference_dataset(
        (values + np.float32(offset)) * np.float32(1e-7), reference="earlobe", channels=channels
    )
    subjects = np.repeat(subject, n_epochs)
    return replace(
        dataset,
        name=source_id,
        subject_ids=subjects,
        preprocessing=replace(dataset.preprocessing, name="causal-test"),
        event_timeline=replace(
            dataset.event_timeline,
            event_ids=np.asarray([f"{source_id}:event:{index}" for index in range(n_epochs)]),
            group_ids=np.asarray([f"{source_id}:group:{index}" for index in range(n_epochs)]),
            subject_ids=subjects,
            dataset_ids=np.repeat(source_id, n_epochs),
            selection_ids=np.asarray([f"selection:{index}" for index in range(n_epochs)]),
        ),
        metadata=pd.DataFrame({"subject": subjects}),
        provenance={"source": source_id, "source_reference": "earlobe"},
    )


def _prepared_source(dataset: EpochDataset) -> EpochDataset:
    car = adapt_common_channel_average_reference(dataset, dataset.channel_names)
    return namespace_epoch_dataset(car, dataset.name)


def test_multidomain_builder_reuses_prepared_car_and_namespace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _prepared_source(_source_dataset(source_id="A", subject="01", offset=0.0))
    second = _prepared_source(_source_dataset(source_id="B", subject="02", offset=100.0))
    first_path = save_epoch_dataset(tmp_path / "first.npz", first)
    second_path = save_epoch_dataset(tmp_path / "second.npz", second)
    output_path = tmp_path / "merged.npz"
    from experiments import prepare_multidomain_source as module

    original_prepare = module._prepare_source
    barrier = threading.Barrier(2)
    worker_names: list[str] = []

    def concurrent_prepare(source, **options):
        worker_names.append(threading.current_thread().name)
        barrier.wait(timeout=5)
        return original_prepare(source, **options)

    monkeypatch.setattr(module, "_prepare_source", concurrent_prepare)
    set_preparation_cli(
        monkeypatch,
        "prepare_multidomain_source.py",
        sources=(
            f"A={first_path}",
            f"B={second_path}",
        ),
        target_channels="Cz,P3,Pz,P4,Oz",
        reference_mode="common_average",
        output=output_path,
        source_workers="2",
        cpu_threads="2",
    )

    main()

    merged = load_epoch_dataset(output_path, require_labels=True, validation="attested")
    assert (
        ensure_common_channel_average_reference(
            merged,
            ("Cz", "P3", "Pz", "P4", "Oz"),
        )
        is merged
    )
    np.testing.assert_array_equal(merged.X[: first.n_epochs], first.X)
    np.testing.assert_array_equal(merged.X[first.n_epochs :], second.X)
    assert merged.subject_ids.tolist() == ["A::01", "A::01", "B::02", "B::02"]
    assert source_domain_ids(merged).tolist() == ["A", "A", "B", "B"]
    assert source_domain_vocabulary(merged) == ("A", "B")
    assert len(set(worker_names)) == 2
    assert all(name.startswith("multidomain-source") for name in worker_names)
    source_loading = merged.provenance["source_loading"]
    assert source_loading["task_count"] == 2
    assert source_loading["requested_workers"] == 2
    assert source_loading["effective_workers"] == 2
    assert source_loading["available_threads"] >= 2
    assert source_loading["total_thread_budget"] == 2
    assert source_loading["threads_per_worker"] == 1
    assert not any("A::A::" in value or "B::B::" in value for value in merged.subject_ids)
    operations = [entity.operation for entity in merged.lineage.entities]
    assert operations.count("common_channel_average_reference") == 2
    assert operations.count("namespace_subject_axis") == 2
    assert operations.count("concatenate_epoch_datasets") == 1
    source_records = merged.provenance["sources"]
    assert [record["namespace_preexisting"] for record in source_records] == [True, True]
    for record, path in zip(source_records, (first_path, second_path), strict=True):
        attestation = loaded_epoch_cache_attestation(
            load_epoch_dataset(path, validation="attested")
        )
        assert record["verified_cache_attestation"]["sha256"] == attestation["sha256"]
    output_sha256 = loaded_epoch_cache_attestation(merged)["sha256"]
    with pytest.raises(FileExistsError, match="requires a new output path"):
        main()
    assert (
        loaded_epoch_cache_attestation(load_epoch_dataset(output_path, validation="attested"))[
            "sha256"
        ]
        == output_sha256
    )


def test_multidomain_builder_requires_explicit_binary_view_for_mixed_event_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate = _source_dataset(source_id="CANDIDATE", subject="01", offset=0.0)
    candidate.event_timeline = replace(
        candidate.event_timeline,
        group_ids=np.repeat("CANDIDATE:group", candidate.n_epochs),
        selection_ids=np.repeat("selection", candidate.n_epochs),
        candidate_ids=np.asarray(["left", "right"]),
        target_candidate_ids=np.asarray(["right", "right"]),
        repetition_indices=np.asarray([0, 0], dtype=np.int64),
    )
    candidate = _prepared_source(candidate)
    binary = _prepared_source(_source_dataset(source_id="BINARY", subject="02", offset=100.0))
    candidate_path = save_epoch_dataset(tmp_path / "candidate.npz", candidate)
    binary_path = save_epoch_dataset(tmp_path / "binary.npz", binary)
    output_path = tmp_path / "binary-view.npz"
    rejected_path = tmp_path / "implicit-drop.npz"
    set_preparation_cli(
        monkeypatch,
        "prepare_multidomain_source.py",
        sources=(
            f"CANDIDATE={candidate_path}",
            f"BINARY={binary_path}",
        ),
        target_channels="Cz,P3,Pz,P4,Oz",
        reference_mode="common_average",
        output=rejected_path,
    )
    with pytest.raises(ValueError, match="candidate_ids must be either absent or present"):
        main()
    assert not rejected_path.exists()

    set_preparation_cli(
        monkeypatch,
        "prepare_multidomain_source.py",
        sources=(
            f"CANDIDATE={candidate_path}",
            f"BINARY={binary_path}",
        ),
        target_channels="Cz,P3,Pz,P4,Oz",
        event_contract="binary_evidence",
        reference_mode="common_average",
        output=output_path,
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


def test_multidomain_builder_preserves_native_reference_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _source_dataset(source_id="A", subject="01", offset=0.0)
    second = _source_dataset(source_id="B", subject="02", offset=100.0)
    first_path = save_epoch_dataset(tmp_path / "first.npz", first)
    second_path = save_epoch_dataset(tmp_path / "second.npz", second)
    output_path = tmp_path / "native-merged.npz"
    set_preparation_cli(
        monkeypatch,
        "prepare_multidomain_source.py",
        sources=(
            f"A={first_path}",
            f"B={second_path}",
        ),
        target_channels="Cz,P3,Pz,P4,Oz",
        output=output_path,
    )

    main()

    merged = load_epoch_dataset(output_path, require_labels=True, validation="attested")
    assert merged.provenance["source_reference"] == "native:earlobe"
    assert merged.provenance["domain_adapter"]["schema"] == "n2p3_native_reference_selection/1"
    assert merged.provenance["reference_mode"] == "native_preserve"
    assert all(
        record["reference_mode"] == "native_preserve" for record in merged.provenance["sources"]
    )
    np.testing.assert_array_equal(merged.X[: first.n_epochs], first.X)
    np.testing.assert_array_equal(merged.X[first.n_epochs :], second.X)


def test_multidomain_builder_rejects_two_channel_car_when_requested(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    first = _source_dataset(source_id="A", subject="01", offset=0.0)
    second = _source_dataset(source_id="B", subject="02", offset=100.0)
    first_path = save_epoch_dataset(tmp_path / "first.npz", first)
    second_path = save_epoch_dataset(tmp_path / "second.npz", second)
    set_preparation_cli(
        monkeypatch,
        "prepare_multidomain_source.py",
        sources=(
            f"A={first_path}",
            f"B={second_path}",
        ),
        target_channels="Cz,P3",
        reference_mode="common_average",
        output=tmp_path / "two-channel.npz",
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    assert (
        "common_average reference mode requires at least 3 common channels; got 2"
        in capsys.readouterr().err
    )


def test_reference_cache_derivation_builds_native_and_car_views(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source_dataset(source_id="RAW", subject="01", offset=0.0)
    source_path = save_epoch_dataset(tmp_path / "raw.npz", source)
    native_path = tmp_path / "native.npz"
    car_path = tmp_path / "car.npz"

    set_preparation_cli(
        monkeypatch,
        "prepare_reference_cache.py",
        dataset_cache=source_path,
        target_channels="Cz,P3,Pz,P4,Oz",
        reference_mode="native_preserve",
        output=native_path,
    )
    derive_reference_cache_main()
    native = load_epoch_dataset(native_path, require_labels=True, validation="attested")
    assert native.provenance["source_reference"] == "native:earlobe"
    np.testing.assert_array_equal(native.X, source.X)

    set_preparation_cli(
        monkeypatch,
        "prepare_reference_cache.py",
        dataset_cache=source_path,
        target_channels="Cz,P3,Pz,P4,Oz",
        reference_mode="common_average",
        output=car_path,
    )
    derive_reference_cache_main()
    car = load_epoch_dataset(car_path, require_labels=True, validation="attested")
    assert car.provenance["domain_adapter"]["schema"] == "n2p3_common_channel_car/1"
    np.testing.assert_allclose(car.X.sum(axis=1), 0.0, atol=1e-6)


def test_multidomain_builder_preserves_sparse_source_observations_in_full_montage(
    tmp_path, monkeypatch
):
    channels = ("Fz", "Cz", "P3", "Pz", "P4", "PO8", "Oz")
    sparse = _source_dataset(source_id="SPARSE", subject="01", offset=0.0)
    full = _source_dataset(source_id="FULL", subject="02", offset=100.0, channels=channels)
    sparse_path = save_epoch_dataset(tmp_path / "sparse.npz", sparse)
    full_path = save_epoch_dataset(tmp_path / "full.npz", full)
    output = tmp_path / "union.npz"
    set_preparation_cli(
        monkeypatch,
        "prepare_multidomain_source",
        sources=(
            f"SPARSE={sparse_path}",
            f"FULL={full_path}",
        ),
        target_channels=",".join(channels),
        event_contract="binary_evidence",
        reference_mode="native_preserve",
        missing_channel_policy="mask",
        output=output,
        source_workers="1",
    )
    main()
    combined = load_epoch_dataset(output, require_labels=True, validation="attested")
    assert combined.channel_mask.all()
    assert combined.X.shape == (4, 7, 4)
    absent = [combined.channel_names.index(name) for name in ("FZ", "PO8")]
    sparse_rows = slice(0, sparse.n_epochs)
    assert not combined.trial_channel_mask[sparse_rows][:, absent].any()
    assert np.all(combined.X[sparse_rows][:, absent] == 0.0)
    assert combined.trial_channel_mask[sparse.n_epochs :].all()
    np.testing.assert_array_equal(combined.y, np.r_[sparse.y, full.y])
    assert combined.identity_table is not None
    assert len(combined.identity_table.records) == 2
