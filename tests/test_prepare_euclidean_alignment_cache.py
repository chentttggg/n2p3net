from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.channel import build_channel_identity
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.domain import (
    SOURCE_DOMAIN_AXIS_SCHEMA,
    SOURCE_DOMAIN_COLUMN,
    ensure_common_channel_average_reference,
)
from data.epochs import (
    EpochDataset,
    epoch_row_identity_digest,
    load_epoch_dataset,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
)
from data.events import observed_only_timeline
from experiments.prepare_euclidean_alignment_cache import main

EA_FIT_SCHEMA = "n2p3_euclidean_alignment_fit/1"


def _make_dataset(
    name: str,
    subjects: tuple[str, ...],
    *,
    domains: tuple[str, ...] | None = None,
) -> EpochDataset:
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    n_epochs = len(subjects)
    rng = np.random.default_rng(20260904)
    scale = np.asarray([[2.0, 0.3, 0.1], [0.0, 1.8, 0.2], [0.1, 0.0, 2.2]])
    z = rng.normal(size=(n_epochs, scale.shape[0], SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times))
    x = np.einsum("cp,npt->nct", scale, z)
    timeline = observed_only_timeline(
        dataset_id=name,
        subject_ids=np.asarray(subjects),
        stimulus_ids=np.arange(n_epochs, dtype=np.int64),
        group_ids=np.asarray(subjects),
        online_causal=True,
        timing_source="ea_cache_cli_fixture",
    )
    metadata = pd.DataFrame({"subject": np.asarray(subjects)})
    provenance: dict[str, object] = {
        "source": "unit_test",
        "source_reference": "average",
    }
    if domains is not None:
        if len(domains) != n_epochs:
            raise ValueError("domains must align with subjects.")
        metadata[SOURCE_DOMAIN_COLUMN] = np.asarray(domains)
        provenance["source_domain_axis"] = {
            "schema": SOURCE_DOMAIN_AXIS_SCHEMA,
            "column": SOURCE_DOMAIN_COLUMN,
            "domains": sorted(set(domains)),
        }
    return EpochDataset(
        name=name,
        X=np.ascontiguousarray(x, dtype=np.float32),
        y=np.zeros(n_epochs, dtype=np.int64),
        subject_ids=np.asarray(subjects),
        channel_names=identity.names,
        channel_positions_m=np.asarray(identity.coords, dtype=np.float32),
        channel_mask=np.ones(len(identity.names), dtype=bool),
        preprocessing=preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT),
        event_timeline=timeline,
        metadata=metadata,
        provenance=provenance,
    )


def _run_ea_cli(tmp_path: Path, dataset: EpochDataset, fit_scope: str) -> tuple[Path, dict]:
    source = tmp_path / "source.npz"
    output = tmp_path / "source-ea.npz"
    save_epoch_dataset(source, dataset)
    main(
        [
            "--dataset-cache",
            str(source),
            "--output",
            str(output),
            "--fit-scope",
            fit_scope,
            "--reference-mode",
            "native_preserve",
        ]
    )
    record = json.loads(output.with_suffix(".record.json").read_text(encoding="utf-8"))
    return output, record


def test_prepare_ea_cache_cli_writes_source_rows_scope(tmp_path: Path) -> None:
    dataset = _make_dataset("source", ("s01", "s02", "s03", "s04"))
    source = tmp_path / "source.npz"
    output = tmp_path / "source-ea.npz"
    before_digest = epoch_row_identity_digest(dataset)
    save_epoch_dataset(source, dataset)

    main(
        [
            "--dataset-cache",
            str(source),
            "--output",
            str(output),
            "--fit-scope",
            "source_rows",
            "--reference-mode",
            "native_preserve",
            "--regularization",
            "1e-10",
        ]
    )

    record = json.loads(output.with_suffix(".record.json").read_text(encoding="utf-8"))
    alignment = record["provenance"]["data_alignment"]
    assert alignment["schema"] == EA_FIT_SCHEMA
    assert alignment["scope"] == "source_rows"
    assert alignment["reference_mode"] == "native_preserve"
    assert alignment["state"]["fit_scope"] == "source_rows"
    assert alignment["row_identity_digest"] == before_digest
    adapted = load_epoch_dataset(output, require_labels=True, validation="attested")
    assert adapted.X.shape == dataset.X.shape
    assert epoch_row_identity_digest(adapted) == before_digest
    np.testing.assert_array_equal(adapted.y, dataset.y)
    assert adapted.provenance["parent_dataset_name"] == "source"

    state = alignment["state"]
    inverse = np.asarray(state["inverse_sqrt_reference"], dtype=np.float64)
    assert inverse.shape == (3, 3)
    covariance = np.mean(
        np.einsum("nct,nut->ncu", adapted.X, adapted.X), axis=0
    ) / adapted.X.shape[2]
    assert np.allclose(covariance, np.eye(3), atol=1e-7)


def test_prepare_ea_cache_cli_writes_per_domain_scope(tmp_path: Path) -> None:
    dataset = _make_dataset(
        "multidomain",
        tuple(f"s{i:02d}" for i in range(8)),
        domains=("A", "A", "A", "A", "B", "B", "B", "B"),
    )
    output, record = _run_ea_cli(tmp_path, dataset, "per_domain")

    alignment = record["provenance"]["data_alignment"]
    assert alignment["schema"] == EA_FIT_SCHEMA
    assert alignment["scope"] == "per_domain"
    assert alignment["reference_mode"] == "native_preserve"
    assert set(alignment["states"]) == {"A", "B"}
    for domain, state in alignment["states"].items():
        assert state["schema"] == "n2p3_euclidean_alignment/1"
        assert state["fit_scope"] == "per_domain"
        assert state["domain_id"] == domain
    adapted = load_epoch_dataset(output, require_labels=True, validation="attested")
    assert epoch_row_identity_digest(adapted) == epoch_row_identity_digest(dataset)


def test_prepare_ea_cache_cli_rejects_per_domain_without_axis(tmp_path: Path) -> None:
    dataset = _make_dataset("single-domain", ("s01", "s02", "s03"))
    source = tmp_path / "source.npz"
    save_epoch_dataset(source, dataset)

    with pytest.raises(SystemExit):
        main(
            [
                "--dataset-cache",
                str(source),
                "--output",
                str(tmp_path / "source-ea.npz"),
                "--fit-scope",
                "per_domain",
                "--reference-mode",
                "native_preserve",
            ]
        )


def test_prepare_ea_cache_cli_rejects_existing_output(tmp_path: Path) -> None:
    dataset = _make_dataset("source", ("s01", "s02", "s03"))
    source = tmp_path / "source.npz"
    output = tmp_path / "source-ea.npz"
    save_epoch_dataset(source, dataset)
    output.write_bytes(b"occupied")

    with pytest.raises(FileExistsError, match="new output path"):
        main(
            [
                "--dataset-cache",
                str(source),
                "--output",
                str(output),
                "--fit-scope",
                "source_rows",
                "--reference-mode",
                "native_preserve",
            ]
        )



def test_prepare_ea_cache_cli_rejects_reference_mode_mismatch(tmp_path: Path) -> None:
    native = _make_dataset("native", ("s01", "s02", "s03"))
    native_cache = tmp_path / "native.npz"
    save_epoch_dataset(native_cache, native)

    with pytest.raises(ValueError, match="common_average"):
        main(
            [
                "--dataset-cache",
                str(native_cache),
                "--output",
                str(tmp_path / "native-ea.npz"),
                "--fit-scope",
                "source_rows",
                "--reference-mode",
                "common_average",
            ]
        )

    car = ensure_common_channel_average_reference(native, native.channel_names)
    car_cache = tmp_path / "car.npz"
    save_epoch_dataset(car_cache, car)
    with pytest.raises(ValueError, match="native_preserve"):
        main(
            [
                "--dataset-cache",
                str(car_cache),
                "--output",
                str(tmp_path / "car-ea.npz"),
                "--fit-scope",
                "source_rows",
                "--reference-mode",
                "native_preserve",
            ]
        )


def test_prepare_ea_cache_cli_accepts_declared_common_average(tmp_path: Path) -> None:
    native = _make_dataset("native", ("s01", "s02", "s03"))
    car = ensure_common_channel_average_reference(native, native.channel_names)
    car_cache = tmp_path / "car.npz"
    output = tmp_path / "car-ea.npz"
    save_epoch_dataset(car_cache, car)

    main(
        [
            "--dataset-cache",
            str(car_cache),
            "--output",
            str(output),
            "--fit-scope",
            "source_rows",
            "--reference-mode",
            "common_average",
        ]
    )

    record = json.loads(output.with_suffix(".record.json").read_text(encoding="utf-8"))
    alignment = record["provenance"]["data_alignment"]
    assert alignment["reference_mode"] == "common_average"
    adapted = load_epoch_dataset(output, require_labels=True, validation="attested")
    assert adapted.provenance["source_reference"] == car.provenance["source_reference"]
