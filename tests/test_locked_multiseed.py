from __future__ import annotations

import json

import pytest

from baselines.experiment_protocol import canonical_sha256
from experiments.run_locked_multiseed import (
    _require_fresh_score_path,
    _validate_forwarded_args,
    _validate_score_payload,
)


def _score_payload() -> dict:
    payload = {
        "schema": "n2p3net_subject_scores/2",
        "model": "eegnet",
        "seed": 0,
        "evaluation_mode": "confirmatory",
        "primary_decision_metric": "exact_llr@3",
        "dataset_sha256": "dataset",
        "protocol_sha256": "protocol",
        "source_sha256": "source",
        "runtime_sha256": "runtime",
        "external_assets_sha256": "assets",
        "confirmatory_id": "final-v1",
        "confirmatory_lock_sha256": "lock",
        "cohort_sha256": "cohort",
        "evaluation_units": ["s1", "s2"],
        "primary_n_subjects": 2,
        "primary_hit_rate": 0.5,
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
                "predicted": None,
                "true": 2,
                "available": False,
                "hit": 0,
            },
        ],
    }
    payload["score_sha256"] = canonical_sha256(payload)
    return payload


def test_locked_runner_rejects_exact_and_abbreviated_overrides() -> None:
    with pytest.raises(ValueError, match="locked options"):
        _validate_forwarded_args(["--device", "cpu"])
    with pytest.raises(ValueError, match="locked options"):
        _validate_forwarded_args(["--dev", "cpu"])


def test_locked_score_validation_binds_model_and_confirmatory_lock(tmp_path) -> None:
    path = tmp_path / "score.json"
    payload = _score_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")
    hits, availability, units, cohort = _validate_score_payload(
        payload,
        path=path,
        seed=0,
        mode="confirmatory",
        primary_metric="exact_llr@3",
        model="eegnet",
        dataset_sha256="dataset",
        protocol_sha256="protocol",
        source_sha256="source",
        runtime_sha256="runtime",
        external_assets_sha256="assets",
        confirmatory_id="final-v1",
        confirmatory_lock_sha256="lock",
    )
    assert hits == {"s1": 1.0, "s2": 0.0}
    assert availability == {"s1": True, "s2": False}
    assert units == ("s1", "s2") and cohort == "cohort"
    with pytest.raises(ValueError, match="frozen run identity"):
        _validate_score_payload(
            payload,
            path=path,
            seed=0,
            mode="confirmatory",
            primary_metric="exact_llr@3",
            model="conformer",
            dataset_sha256="dataset",
            protocol_sha256="protocol",
            source_sha256="source",
            runtime_sha256="runtime",
            external_assets_sha256="assets",
            confirmatory_id="final-v1",
            confirmatory_lock_sha256="lock",
        )


def test_locked_runner_rejects_stale_score_file(tmp_path) -> None:
    path = tmp_path / "score.json"
    path.write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale score"):
        _require_fresh_score_path(path, seed=3)
