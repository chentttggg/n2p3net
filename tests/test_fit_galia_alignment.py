from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fit_galia_alignment import main


def test_fit_galia_cli_writes_state_and_aligned_feature_matrix(tmp_path: Path) -> None:
    rng = np.random.default_rng(41)
    source_a = tmp_path / "a.npz"
    source_b = tmp_path / "b.npz"
    np.savez(source_a, features=rng.normal(size=(20, 3)))
    np.savez(source_b, features=rng.normal(size=(20, 3)))
    state_path = tmp_path / "state.json"
    aligned_path = tmp_path / "aligned.npz"

    main(
        [
            "--source",
            f"A={source_a}",
            "--source",
            f"B={source_b}",
            "--output",
            str(state_path),
            "--aligned-output",
            str(aligned_path),
        ]
    )

    record = json.loads(state_path.read_text(encoding="utf-8"))
    assert record["schema"] == "n2p3_galia_fit/1"
    assert record["fit_scope"] == "source_only"
    state = record["state"]
    assert state["schema"] == "n2p3_galia_alignment/1"
    with np.load(aligned_path, allow_pickle=False) as archive:
        assert archive["schema"].item() == "n2p3_aligned_feature_matrix/1"
        assert archive["aligned_features"].shape == (40, 3)
        assert archive["domain_ids"].tolist() == ["A"] * 20 + ["B"] * 20
        assert archive["galia_state_digest"].item() == state["state_digest"]


def test_fit_galia_cli_rejects_raw_epoch_tensor_and_duplicate_domains(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez(first, features=np.zeros((5, 2, 3)))
    np.savez(second, features=np.zeros((5, 2)))
    with pytest.raises(ValueError, match="two-dimensional feature matrix"):
        main(
            [
                "--source",
                f"A={first}",
                "--source",
                f"B={second}",
                "--output",
                str(tmp_path / "state.json"),
            ]
        )

    np.savez(first, features=np.zeros((5, 2)))
    with pytest.raises(SystemExit):
        main(
            [
                "--source",
                f"A={first}",
                "--source",
                f"A={second}",
                "--output",
                str(tmp_path / "duplicate.json"),
            ]
        )
