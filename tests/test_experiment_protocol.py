from __future__ import annotations

import json

import pytest

from baselines.experiment_protocol import (
    EvaluationProtocol,
    aggregate_seed_subject_hits,
    canonical_sha256,
    claim_confirmatory_seed,
    frozen_truth_sha256,
    parse_seeds,
    reserve_confirmatory_lock,
    validate_confirmatory_lock,
    validate_eligibility_manifest,
)


def test_confirmatory_defaults_to_five_seeds() -> None:
    seeds = parse_seeds(None, mode="confirmatory")
    assert seeds == (0, 1, 2, 3, 4)
    EvaluationProtocol("confirmatory", seeds, "exact_llr@3", None).validate()


def test_confirmatory_rejects_subset_or_too_few_seeds() -> None:
    with pytest.raises(ValueError, match="five"):
        EvaluationProtocol("confirmatory", (0,), "exact_llr@3", None).validate()
    with pytest.raises(ValueError, match="complete"):
        EvaluationProtocol("confirmatory", (0, 1, 2, 3, 4), "exact_llr@3", 20).validate()


def test_confirmatory_lock_is_one_use(tmp_path) -> None:
    protocol = EvaluationProtocol("confirmatory", (0, 1, 2, 3, 4), "exact_llr@3", None).validate()
    path = reserve_confirmatory_lock(
        tmp_path,
        confirmatory_id="gtn-final-v1",
        dataset_sha256="abc",
        protocol=protocol,
        frozen_config={"encoder": "tcn"},
    )
    assert (
        json.loads(path.read_text(encoding="utf-8"))["protocol"]["primary_metric"] == "exact_llr@3"
    )
    with pytest.raises(RuntimeError, match="already locked"):
        reserve_confirmatory_lock(
            tmp_path,
            confirmatory_id="gtn-final-v1",
            dataset_sha256="abc",
            protocol=protocol,
            frozen_config={"encoder": "tcn"},
        )


def test_hierarchical_seed_subject_uncertainty_preserves_both_axes() -> None:
    hits = {
        0: {"s1": 1.0, "s2": 0.0, "s3": 1.0},
        1: {"s1": 1.0, "s2": 1.0, "s3": 0.0},
        2: {"s1": 0.0, "s2": 1.0, "s3": 1.0},
    }
    result = aggregate_seed_subject_hits(hits, n_bootstrap=200, bootstrap_seed=3)
    assert result.mean_hit_rate == pytest.approx(2.0 / 3.0)
    assert result.n_seeds == 3 and result.n_subjects == 3
    assert result.hierarchical_ci_low <= result.mean_hit_rate <= result.hierarchical_ci_high


def test_seed_aggregation_rejects_subject_intersection() -> None:
    hits = {0: {"u1": 1.0, "u2": 0.0}, 1: {"u1": 1.0}}
    with pytest.raises(ValueError, match="frozen universe"):
        aggregate_seed_subject_hits(hits, subject_universe=("u1", "u2"))


def test_confirmatory_rejects_ambiguous_legacy_budget() -> None:
    with pytest.raises(ValueError, match="Ambiguous legacy"):
        EvaluationProtocol("confirmatory", (0, 1, 2, 3, 4), "llr@15", None).validate()


def test_confirmatory_lock_binds_child_identity_and_seed_is_one_use(tmp_path) -> None:
    protocol = EvaluationProtocol(
        "confirmatory", (0, 1, 2, 3, 4), "exact_llr@3", None
    ).validate()
    frozen = {
        "runner": "baseline",
        "model": "eegnet",
        "dataset_sha256": "dataset",
    }
    protocol_sha256 = canonical_sha256(frozen)
    path = reserve_confirmatory_lock(
        tmp_path,
        confirmatory_id="final-v2",
        dataset_sha256="dataset",
        protocol=protocol,
        frozen_config=frozen,
    )
    payload, lock_hash = validate_confirmatory_lock(
        path,
        dataset_sha256="dataset",
        protocol_sha256=protocol_sha256,
        primary_metric="exact_llr@3",
        seed=0,
        runner="baseline",
        model="eegnet",
    )
    assert payload["confirmatory_id"] == "final-v2" and len(lock_hash) == 64
    claim_confirmatory_seed(path, seed=0, run_identity={"model": "eegnet"})
    with pytest.raises(RuntimeError, match="already consumed"):
        claim_confirmatory_seed(path, seed=0, run_identity={"model": "eegnet"})
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_confirmatory_lock(
            path,
            dataset_sha256="dataset",
            protocol_sha256=protocol_sha256,
            primary_metric="exact_llr@3",
            seed=0,
            runner="baseline",
            model="conformer",
        )


def test_eligibility_manifest_rejects_a_shrunk_cache_universe(tmp_path) -> None:
    truth = {"u1": 1, "u2": 2}
    path = tmp_path / "cohort.json"
    path.write_text(
        json.dumps(
            {
                "schema": "n2p3net_eligibility_manifest/1",
                "dataset": "gtn",
                "n_evaluation_units": 2,
                "truth_universe_sha256": frozen_truth_sha256("gtn", truth),
            }
        ),
        encoding="utf-8",
    )
    validate_eligibility_manifest(path, dataset="gtn", truth_by_unit=truth)
    with pytest.raises(ValueError, match="frozen eligibility universe"):
        validate_eligibility_manifest(path, dataset="gtn", truth_by_unit={"u1": 1})
