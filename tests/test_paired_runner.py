from __future__ import annotations

import json

import pytest

from baselines.experiment_protocol import canonical_sha256
from experiments.run_paired_test import _load_hits


def test_paired_loader_defaults_to_primary_records(tmp_path) -> None:
    path = tmp_path / "scores.json"
    payload = {
        "schema": "n2p3net_subject_scores/2",
        "model": "m",
        "seed": 0,
        "evaluation_mode": "development",
        "protocol_sha256": "protocol",
        "dataset_sha256": "dataset",
        "cohort_sha256": "cohort",
        "evaluation_units": ["s1", "s2"],
        "n_subjects": 2,
        "hit_rate_mean": 0.0,
        "records": [
            {"subject": "s1", "hit": 0},
            {"subject": "s2", "hit": 0},
        ],
        "primary_decision_metric": "exact_llr@3",
        "primary_hit_rate": 1.0,
        "primary_n_subjects": 2,
        "primary_records": [
            {
                "subject": "s1",
                "predicted": 1,
                "true": 1,
                "available": True,
                "hit": 1,
            },
            {
                "subject": "s2",
                "predicted": 2,
                "true": 2,
                "available": True,
                "hit": 1,
            },
        ],
    }
    payload["score_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _, hits, rate, metric = _load_hits(path)
    assert hits == {"s1": 1, "s2": 1}
    assert rate == 1.0 and metric == "exact_llr@3"


def test_old_score_file_requires_explicit_legacy(tmp_path) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "model": "old",
                "n_subjects": 1,
                "hit_rate_mean": 1.0,
                "records": [{"subject": "s1", "hit": 1}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy"):
        _load_hits(path)
    assert _load_hits(path, "legacy")[2:] == (1.0, "legacy_sum@all")
