from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.prepare_source_selection import _parse_pds_mixture, main
from train.source_sampling import (
    array_digest,
    load_pds_selection_record,
    pds_selection_from_record,
)

FEATURE_SCHEMA = "n2p3_domain_feature_matrix/1"


def _write_feature_matrix(
    path: Path,
    values: np.ndarray,
    *,
    cache_sha256: str,
    dataset_scope: dict[str, object] | None = None,
) -> dict[str, object]:
    np.savez_compressed(
        path,
        features=values,
        feature_key=np.asarray("features"),
        schema=np.asarray(FEATURE_SCHEMA),
    )
    record = {
        "schema": FEATURE_SCHEMA,
        "source_cache": "synthetic-cache.npz",
        "source_cache_sha256": cache_sha256,
        "dataset_scope": {} if dataset_scope is None else dataset_scope,
        "shape": list(values.shape),
        "spatial_lift": {"kind": "identity", "shape": None, "path": None, "key": "spatial_lift"},
        "temporal_basis": {
            "kind": "identity",
            "shape": None,
            "path": None,
            "key": "temporal_basis",
        },
        "channel_names": ["Fz", "Cz"],
        "n_times": int(values.shape[1]),
        "feature_dim": int(values.shape[1]),
        "labels_included": False,
        "domain_axis_included": False,
    }
    path.with_suffix(".record.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    return record


def test_prepare_source_selection_cli_writes_bound_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = np.asarray(
        [
            [-3.0, 0.0],
            [-2.5, 0.1],
            [-0.5, 0.0],
            [0.0, 0.2],
            [2.5, 0.0],
            [3.0, -0.1],
        ]
    )
    target = np.asarray([[2.8, 0.0], [2.9, 0.1]])
    source_path = tmp_path / "source_features.npz"
    target_path = tmp_path / "target_features.npz"
    _write_feature_matrix(source_path, source, cache_sha256="a" * 64)
    _write_feature_matrix(target_path, target, cache_sha256="b" * 64)
    labels_path = tmp_path / "labels.npz"
    groups_path = tmp_path / "groups.npz"
    np.savez(labels_path, labels=np.asarray([0, 0, 1, 1, 1, 1]))
    np.savez(groups_path, groups=np.asarray(["a", "a", "b", "b", "c", "c"]))
    output = tmp_path / "selection.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_source_selection.py",
            "--source-features",
            str(source_path),
            "--target-features",
            str(target_path),
            "--output",
            str(output),
            "--n-select",
            "4",
            "--seed",
            "17",
            "--source-labels",
            str(labels_path),
            "--source-groups",
            str(groups_path),
            "--min-per-label",
            "1",
            "--min-per-group",
            "1",
        ],
    )

    main()

    selection, record = load_pds_selection_record(output)
    groups = np.load(groups_path)["groups"]
    labels = np.load(labels_path)["labels"]
    assert len(selection.selected_indices) == 4
    assert set(groups[selection.selected_indices]) == {"a", "b", "c"}
    assert set(labels[selection.selected_indices]) == {0, 1}
    assert record["source_cache_sha256"] == "a" * 64
    assert record["target_cache_sha256"] == "b" * 64
    assert record["n_select"] == 4
    assert record["sampler_kind"] == "target_similarity_diverse"
    assert record["pds_radius"] is None
    assert record["pds_mingling_mixture"] is None
    assert record["pds_attempts"] == 0
    assert record["pds_fallback_count"] == 0
    assert pds_selection_from_record(
        record,
        expected_source_cache_sha256="a" * 64,
    ).selected_digest == selection.selected_digest


def test_prepare_source_selection_rejects_dimension_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source_features.npz"
    target_path = tmp_path / "target_features.npz"
    _write_feature_matrix(
        source_path,
        np.zeros((4, 3)),
        cache_sha256="a" * 64,
    )
    _write_feature_matrix(
        target_path,
        np.zeros((2, 2)),
        cache_sha256="b" * 64,
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_source_selection.py",
            "--source-features",
            str(source_path),
            "--target-features",
            str(target_path),
            "--output",
            str(output),
            "--n-select",
            "2",
        ],
    )

    with pytest.raises(ValueError, match="feature dimensions must match"):
        main()

def test_pds_mixture_cli_parser_formats_and_validation() -> None:
    assert _parse_pds_mixture("0.5, 0.5", [0, 1]) == (0.5, 0.5)
    assert _parse_pds_mixture("[0.25, 0.75]", [0, 1]) == (0.25, 0.75)
    assert _parse_pds_mixture('{"0": 0.25, "1": 0.75}', [0, 1]) == (0.25, 0.75)
    assert _parse_pds_mixture('{"10": 0.25, "20": 0.75}', [10, 20]) == (0.25, 0.75)

    with pytest.raises(ValueError, match="sum to one"):
        _parse_pds_mixture("0.5,0.4", [0, 1])
    with pytest.raises(ValueError, match="exactly"):
        _parse_pds_mixture('{"0": 0.5, "1": 0.5, "2": 0.0}', [0, 1])
    with pytest.raises(ValueError, match="numeric"):
        _parse_pds_mixture("0.5,target", [0, 1])


def test_prepare_source_selection_dense_pds_cli_writes_dense_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = np.asarray([[0.0], [0.4], [2.0], [2.4]])
    target = np.asarray([[2.0]])
    source_path = tmp_path / "source_features.npz"
    target_path = tmp_path / "target_features.npz"
    _write_feature_matrix(source_path, source, cache_sha256="a" * 64)
    _write_feature_matrix(target_path, target, cache_sha256="b" * 64)
    labels_path = tmp_path / "labels.npz"
    np.savez(labels_path, labels=np.asarray([0, 0, 1, 1]))
    output = tmp_path / "dense_pds_selection.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_source_selection.py",
            "--source-features",
            str(source_path),
            "--target-features",
            str(target_path),
            "--output",
            str(output),
            "--n-select",
            "3",
            "--seed",
            "0",
            "--sampler",
            "dense_pds",
            "--source-labels",
            str(labels_path),
            "--pds-radius",
            "0.5",
            "--pds-mixture",
            "0.5,0.5",
            "--pds-max-attempts",
            "10",
        ],
    )

    main()

    selection, record = load_pds_selection_record(output)
    labels = np.load(labels_path)["labels"]
    assert record["sampler_kind"] == "dense_pds"
    assert record["pds_radius"] == 0.5
    assert record["pds_mingling_mixture"] == [0.5, 0.5]
    assert record["pds_attempts"] > 0
    assert record["pds_fallback_count"] >= 0
    assert record["source_labels"]["digest"] == array_digest(labels)
    assert len(selection.selected_indices) == 3
    assert len(np.unique(selection.selected_indices)) == 3
    assert set(labels[selection.selected_indices]) == {0, 1}
    assert selection.sampler_kind == "dense_pds"
    assert selection.target_temperature == 1.0
    assert selection.diversity_weight == 0.0
    assert selection.bandwidth == 1.0
    assert pds_selection_from_record(
        record,
        expected_source_cache_sha256="a" * 64,
    ).pds_mingling_mixture == (0.5, 0.5)


def test_prepare_source_selection_dense_pds_rejects_missing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source_features.npz"
    target_path = tmp_path / "target_features.npz"
    labels_path = tmp_path / "labels.npz"
    _write_feature_matrix(source_path, np.zeros((4, 2)), cache_sha256="a" * 64)
    _write_feature_matrix(target_path, np.zeros((2, 2)), cache_sha256="b" * 64)
    np.savez(labels_path, labels=np.asarray([0, 0, 1, 1]))

    base = [
        "prepare_source_selection.py",
        "--source-features",
        str(source_path),
        "--target-features",
        str(target_path),
        "--output",
        str(tmp_path / "selection.json"),
        "--n-select",
        "2",
        "--sampler",
        "dense_pds",
    ]
    monkeypatch.setattr("sys.argv", base)
    with pytest.raises(SystemExit):
        main()

    monkeypatch.setattr("sys.argv", base + ["--pds-radius", "0.5"])
    with pytest.raises(SystemExit):
        main()


def test_prepare_source_selection_dense_pds_rejects_tsd_only_flags_and_bad_mixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source_features.npz"
    target_path = tmp_path / "target_features.npz"
    labels_path = tmp_path / "labels.npz"
    _write_feature_matrix(source_path, np.zeros((4, 2)), cache_sha256="a" * 64)
    _write_feature_matrix(target_path, np.zeros((2, 2)), cache_sha256="b" * 64)
    np.savez(labels_path, labels=np.asarray([0, 0, 1, 1]))

    base = [
        "prepare_source_selection.py",
        "--source-features",
        str(source_path),
        "--target-features",
        str(target_path),
        "--n-select",
        "2",
        "--sampler",
        "dense_pds",
        "--source-labels",
        str(labels_path),
        "--pds-radius",
        "0.5",
    ]

    for flag, value in (
        ("--target-temperature", "2.0"),
        ("--diversity-weight", "0.5"),
        ("--bandwidth", "1.0"),
        ("--min-per-label", "1"),
        ("--min-per-group", "1"),
    ):
        output = tmp_path / f"rejected_{flag[2:]}.json"
        monkeypatch.setattr(
            "sys.argv",
            [*base, "--output", str(output), flag, value],
        )
        with pytest.raises(SystemExit):
            main()

    output = tmp_path / "bad_mixture.json"
    monkeypatch.setattr(
        "sys.argv",
        [*base, "--output", str(output), "--pds-mixture", "0.5,0.4"],
    )
    with pytest.raises(SystemExit):
        main()



def test_prepare_source_selection_rejects_non_calibration_target_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = np.asarray([[-1.0, 0.0], [0.0, 0.2], [1.0, 0.0], [2.0, 0.1]])
    target = np.asarray([[0.5, 0.0], [0.6, 0.1]])
    source_path = tmp_path / "source_features.npz"
    target_path = tmp_path / "target_features.npz"
    _write_feature_matrix(source_path, source, cache_sha256="a" * 64)
    _write_feature_matrix(
        target_path,
        target,
        cache_sha256="b" * 64,
        dataset_scope={"schema": "n2p3_unrestricted_target_scope/1"},
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_source_selection.py",
            "--source-features",
            str(source_path),
            "--target-features",
            str(target_path),
            "--output",
            str(output),
            "--n-select",
            "2",
            "--selection-scope",
            "target_calibration",
        ],
    )

    with pytest.raises(ValueError, match="target-calibration-scoped"):
        main()
