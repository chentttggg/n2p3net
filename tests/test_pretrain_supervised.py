from __future__ import annotations

import numpy as np
import pytest

from experiments.run_pretrain_supervised import (
    build_subject_prefix_exposure,
    parse_subject_prefix_repeats,
)


def test_parse_subject_prefix_repeats_preserves_explicit_prefixes() -> None:
    assert parse_subject_prefix_repeats("BI::=3, BNCI::=1") == {
        "BI::": 3,
        "BNCI::": 1,
    }
    assert parse_subject_prefix_repeats("") == {}


@pytest.mark.parametrize("value", ["BI::", "=3", "BI::=0", "BI::=1.5", "BI::=2,BI::=3"])
def test_parse_subject_prefix_repeats_rejects_invalid_contracts(value: str) -> None:
    with pytest.raises(ValueError):
        parse_subject_prefix_repeats(value)


def test_subject_prefix_exposure_retains_rows_and_accounts_optimizer_rows() -> None:
    subjects = np.asarray(["BI::01", "BI::01", "BI::02", "BNCI::01", "other"])
    indices, report = build_subject_prefix_exposure(
        subjects,
        {"BI::": 3, "BNCI::": 1},
    )

    assert np.bincount(indices, minlength=len(subjects)).tolist() == [3, 3, 3, 1, 1]
    physical_values = np.arange(len(subjects), dtype=float)
    assert physical_values[indices].mean() == pytest.approx(
        (physical_values[:3].sum() * 3 + physical_values[3:].sum()) / 11
    )
    assert report["unique_physical_rows"] == 5
    assert report["optimizer_rows_per_epoch"] == 11
    assert report["all_unique_rows_retained"] is True
    assert report["prefixes"] == [
        {
            "prefix": "BI::",
            "repeat": 3,
            "unique_physical_rows": 3,
            "optimizer_rows": 9,
            "unique_subjects": 2,
            "optimizer_fraction": 9 / 11,
        },
        {
            "prefix": "BNCI::",
            "repeat": 1,
            "unique_physical_rows": 1,
            "optimizer_rows": 1,
            "unique_subjects": 1,
            "optimizer_fraction": 1 / 11,
        },
        {
            "prefix": None,
            "repeat": 1,
            "unique_physical_rows": 1,
            "optimizer_rows": 1,
            "unique_subjects": 1,
            "optimizer_fraction": 1 / 11,
        },
    ]


def test_subject_prefix_exposure_rejects_unknown_and_overlapping_prefixes() -> None:
    subjects = np.asarray(["BI::01", "BNCI::01"])
    with pytest.raises(ValueError, match="matches no retained source rows"):
        build_subject_prefix_exposure(subjects, {"GTN::": 2})
    with pytest.raises(ValueError, match="prefixes overlap"):
        build_subject_prefix_exposure(subjects, {"BI": 2, "BI::": 3})
