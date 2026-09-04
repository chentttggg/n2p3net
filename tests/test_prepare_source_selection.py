from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.prepare_source_selection import main
from train.source_sampling import load_pds_selection_record, pds_selection_from_record

FEATURE_SCHEMA = "n2p3_domain_feature_matrix/1"


def _write_feature_matrix(
    path: Path,
    values: np.ndarray,
    *,
    cache_sha256: str,
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
