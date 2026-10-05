from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.fit_galia_alignment import main
from tests.support.io import read_json


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

    record = read_json(state_path)
    assert record["schema"] == "n2p3_acw_fit/1"
    assert record["fit_scope"] == "source_only"
    state = record["state"]
    assert state["schema"] == "n2p3_ajd_congruence_whitening/1"
    with np.load(aligned_path, allow_pickle=False) as archive:
        assert archive["schema"].item() == "n2p3_aligned_feature_matrix/1"
        assert archive["aligned_features"].shape == (40, 3)
        assert archive["domain_ids"].tolist() == ["A"] * 20 + ["B"] * 20
        assert archive["acw_state_digest"].item() == state["state_digest"]


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


def test_fit_galia_cli_fast_aligns_target_calibration(tmp_path: Path) -> None:
    rng = np.random.default_rng(43)
    source_a = tmp_path / "a.npz"
    source_b = tmp_path / "b.npz"
    target = tmp_path / "target.npz"
    np.savez(source_a, features=rng.normal(size=(30, 4)))
    np.savez(source_b, features=rng.normal(size=(30, 4)))
    np.savez(target, features=rng.normal(size=(12, 4)))
    state_path = tmp_path / "state.json"
    fast_path = tmp_path / "target-fast.npz"

    main(
        [
            "--source", f"A={source_a}",
            "--source", f"B={source_b}",
            "--output", str(state_path),
            "--target-calibration", str(target),
            "--target-domain", "T",
            "--fast-fit-scope", "target_calibration",
            "--fast-output", str(fast_path),
        ]
    )

    record = read_json(state_path)
    fast = record["fast_alignment"]
    assert fast["schema"] == "n2p3_galia_fast_alignment/1"
    assert fast["fit_scope"] == "target_calibration"
    assert fast["domain_id"] == "T"
    assert len(fast["buffer_digest"]) == 64
    with np.load(fast_path, allow_pickle=False) as archive:
        assert archive["aligned_features"].shape == (12, 4)
        assert archive["domain_ids"].tolist() == ["T"] * 12
        assert archive["fast_alignment_state_digest"].item() == fast["state_digest"]
