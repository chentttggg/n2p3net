"""Small independent Gaussian/schedule/JSON counterexamples for schema-2 diagnostics."""
import json

import numpy as np
import pytest
from scipy.special import ndtr

from transfer.information_bound import InformationBoundReport, mahalanobis_snr
from transfer.information_bound import theoretical_hit_rate_bound as hit_diagnostic
from transfer.noise_model import metadata_digest


def _report(**kwargs):
    rng = np.random.default_rng(123)
    return InformationBoundReport.compute(
        rng.normal(size=(12, 3)) + .2, rng.normal(size=(16, 3)),
        n_candidates=3, n_flashes_per_selection=40, membership_kind="single_item",
        sfreq=64., compute_spectral=False, mc_trials=200, **kwargs,
    )


def test_same_sample_inverse_covariance_does_not_have_fixed_true_covariance_null_bias():
    # Gaussian class-centered pooled scatter W~Wishart(I,n1+n0-2), independent
    # of the class means. Empirical covariance=W/(n1+n0), so its inverse expectation
    # is inflated by n/(n-p-3). Even this unshrunk case refutes p*(1/n1+1/n0).
    n1, n0, p = 8, 8, 4
    fixed_covariance_bias = p * (1 / n1 + 1 / n0)
    expected_fitted_inverse_d2 = fixed_covariance_bias * (n1 + n0) / (n1 + n0 - p - 3)
    assert expected_fitted_inverse_d2 - fixed_covariance_bias == pytest.approx(7 / 9)
    rng = np.random.default_rng(7)
    target = rng.normal(size=(n1, p))
    non = rng.normal(size=(n0, p))
    delta = target.mean(axis=0) - non.mean(axis=0)
    residuals = np.vstack([target - target.mean(axis=0), non - non.mean(axis=0)])
    raw_reference = delta @ np.linalg.solve(residuals.T @ residuals / (n1 + n0), delta)
    snr = mahalanobis_snr(target, non, shrinkage=0)
    assert snr["d2_mahalanobis"] == pytest.approx(raw_reference)
    assert "d2_mahalanobis_debiased" not in snr


@pytest.mark.parametrize("kind,Q,F,counts,balanced", [
    ("single_item", 3, 40, [14, 13, 13], False),
    ("row_column", 9, 45, [15] * 9, True),
    ("row_column", 9, 46, [16, 15, 15, 16, 15, 15, 16, 15, 15], False),
])
def test_uneven_schedule_keeps_every_candidate_geometry(kind, Q, F, counts, balanced):
    result = hit_diagnostic(.2, Q, F, membership_kind=kind, mc_trials=100)
    A = np.asarray(result["activation_matrix"])
    distances = ((A[:, :, None] - A[:, None, :]) ** 2).sum(axis=0)
    errors = ndtr(-.5 * np.sqrt(.2 * distances))
    np.fill_diagonal(errors, 0)
    assert result["activation_counts_per_candidate"] == counts
    assert result["activations_per_candidate"] == pytest.approx(np.mean(counts))
    assert result["d2_effective_per_selection"] == pytest.approx(.2 * np.mean(counts))
    assert result["pairwise_differing_flashes"] == distances.tolist()
    assert result["p_hit_lower_bound"] == pytest.approx(
        max(1 / Q, np.maximum(0, 1 - errors.sum(axis=1)).mean())
    )
    assert result["activation_counts_balanced"] is balanced
    assert result["schedule_cycles_balanced"] is False
    if not balanced:
        assert "Unequal" in result["interpretation"]


def test_incomplete_schedule_identical_candidates_have_correct_random_tie_limit():
    # Q3/F1 distinguishes q0 from {q1,q2}; a perfect detector reaches 2/3.
    result = hit_diagnostic(1e6, 3, 1, membership_kind="single_item", mc_trials=100)
    assert result["indistinguishable_candidates_per_candidate"] == [1, 2, 2]
    assert result["p_hit_per_candidate_monte_carlo"] == [1, .5, .5]
    assert result["identifiable_hit_ceiling"] == pytest.approx(2 / 3)
    for key in ("p_hit_lower_bound", "p_hit_upper_bound", "p_hit_monte_carlo"):
        assert result[key] == pytest.approx(2 / 3)
    assert result["required_d2_for_targets"]["p_hit_75"] is None
    assert result["model"]["decoder"] == "maximum_likelihood_uniform_random_ties"
    chance = hit_diagnostic(0., 3, 1, membership_kind="single_item", mc_trials=100)
    assert chance["p_hit_monte_carlo"] == 1 / 3


def test_binary_schedule_matches_closed_form_optimal_gaussian_hit():
    result = hit_diagnostic(.3, 2, 5, membership_kind="single_item", mc_trials=5000)
    exact = ndtr(.5 * np.sqrt(.3 * 5))
    assert result["p_hit_lower_bound"] == pytest.approx(exact)
    assert result["p_hit_upper_bound"] == pytest.approx(exact)
    assert abs(result["p_hit_monte_carlo"] - exact) < 4 * result["monte_carlo"]["standard_error"]


def test_mc_matches_direct_gaussian_log_likelihood_for_actual_uneven_membership():
    # Independent likelihood oracle: draw whitened observations under each true
    # candidate, then compute all candidate likelihoods directly. Same seed/draws
    # as the implementation lets us compare every MC candidate exactly.
    A = np.eye(3)[[2, 0, 0, 1, 2, 0, 0]]
    d2, trials, seed = .4, 600, 983
    result = hit_diagnostic(
        d2, 3, 7, membership_kind="single_item", activation_matrix=A,
        mc_trials=trials, mc_seed=seed,
    )
    epsilon = np.random.default_rng(seed).standard_normal((trials, len(A)))
    rates, all_hits = [], []
    for true in range(3):
        observations = epsilon + np.sqrt(d2) * A[:, true]
        scores = np.sqrt(d2) * observations @ A - .5 * d2 * A.sum(axis=0)
        hits = np.argmax(scores, axis=1) == true
        rates.append(hits.mean())
        all_hits.append(hits)
    assert result["p_hit_per_candidate_monte_carlo"] == rates
    expected_se = np.stack(all_hits, axis=1).mean(axis=1).std(ddof=1) / np.sqrt(trials)
    assert result["monte_carlo"]["standard_error"] == pytest.approx(expected_se)
    assert result["activation_scope"] == "caller_supplied_schedule"


def test_report_preserves_plugin_and_complete_membership_on_json_replay(tmp_path, monkeypatch):
    report = _report(empirical_hit_rate=.6)
    assert report.d2_per_flash == report.snr_analysis["d2_mahalanobis"]
    assert report.bound["d2_per_flash"] == report.d2_per_flash
    assert report.provenance["membership_kind"] == report.membership_kind == "single_item"
    record = report.to_dict()
    assert record["provenance"]["schema"] == "n2p3_information_bound/2"
    path = tmp_path / "diagnostic.json"
    report.save(path)
    with pytest.raises(FileExistsError):
        report.save(path)
    # Restoring saved diagnostics must not fit calibration again.
    monkeypatch.setattr("transfer.information_bound.mahalanobis_snr",
                        lambda *args, **kwargs: pytest.fail("unexpected covariance refit"))
    loaded = InformationBoundReport.load(path)
    assert loaded.to_dict() == record
    assert loaded.summary() == report.summary()
    assert loaded.replay_model()["p_hit_monte_carlo"] == report.bound["p_hit_monte_carlo"]
    with pytest.raises(TypeError, match="frozen"):
        loaded.bound["monte_carlo"]["seed"] = 0
    with pytest.raises(TypeError):
        loaded.bound["activation_counts_per_candidate"][0] = 0
    record["bound"]["monte_carlo"]["seed"] = 99
    assert loaded.bound["monte_carlo"]["seed"] != 99


@pytest.mark.parametrize("section,key,value", [
    ("bound", "p_hit_monte_carlo", .999),
    ("bound", "activation_counts_per_candidate", [1, 1, 1]),
    ("snr_analysis", "d2_mahalanobis", 0.),
    ("spectral_analysis", None, {"fabricated": 1}),
    ("provenance", "membership_kind", "row_column"),
])
def test_full_content_tampering_is_rejected(section, key, value):
    record = _report().to_dict()
    if key is None:
        record[section] = value
    else:
        record[section][key] = value
    with pytest.raises(ValueError, match="integrity"):
        InformationBoundReport.from_dict(record)


def test_load_and_replay_preserve_old_producer_instead_of_resigning():
    record = _report().to_dict()
    record["provenance"]["implementation_sha256"] = "1" * 64
    record["integrity_sha256"] = metadata_digest(
        {key: value for key, value in record.items() if key != "integrity_sha256"}
    )
    loaded = InformationBoundReport.from_dict(record)
    assert loaded.provenance["implementation_sha256"] == "1" * 64
    with pytest.raises(ValueError, match="original implementation"):
        loaded.replay_model()
    record["provenance"]["schema"] = "n2p3_information_bound/1"
    with pytest.raises(ValueError, match="schema"):
        InformationBoundReport.from_dict(record)


def test_report_model_replay_checks_saved_outputs_even_after_new_checksum():
    record = _report().to_dict()
    record["bound"]["p_hit_monte_carlo"] += .05
    record["integrity_sha256"] = metadata_digest(
        {key: value for key, value in record.items() if key != "integrity_sha256"}
    )
    loaded = InformationBoundReport.from_dict(record)
    with pytest.raises(ValueError, match="do not replay"):
        loaded.replay_model()


@pytest.mark.parametrize("method", ["summary", "to_dict", "replay_model"])
@pytest.mark.parametrize("bypass", ["dict", "object"])
def test_public_outputs_recheck_integrity_after_direct_mutation_bypass(method, bypass):
    report = _report()
    if bypass == "dict":
        dict.__setitem__(report.bound, "p_hit_monte_carlo", .99)
    else:
        object.__setattr__(report, "d2_per_flash", .99)
    with pytest.raises(ValueError, match="integrity"):
        getattr(report, method)()


def test_save_rejects_direct_tamper_before_creating_a_partial_artifact(tmp_path):
    report = _report()
    dict.__setitem__(report.bound, "p_hit_monte_carlo", .99)
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="integrity"):
        report.save(path)
    assert not path.exists()


def test_unreachable_mc_targets_are_strict_json():
    result = hit_diagnostic(.2, 3, 1, membership_kind="single_item", mc_trials=50)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("A,kind,Q,F", [
    ([[1, 1, 0]], "single_item", 3, 1),
    ([[1, 0, 0, 1]], "row_column", 4, 1),
    ([[1, 0, .5]], "single_item", 3, 1),
])
def test_actual_membership_respects_declared_physical_layout(A, kind, Q, F):
    with pytest.raises(ValueError, match="activate|rows|binary"):
        hit_diagnostic(.1, Q, F, membership_kind=kind, activation_matrix=A, mc_trials=50)


@pytest.mark.parametrize("value", [float("nan"), 1.1, -.01])
def test_empirical_comparison_is_a_probability(value):
    with pytest.raises(ValueError, match="empirical_hit_rate"):
        _report(empirical_hit_rate=value)
