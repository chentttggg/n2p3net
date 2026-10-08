"""Posterior direction binding and prediction through the actual factor owner."""
import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from scipy.linalg import solve_triangular

from transfer.continuous_decision import (
    CalibrationRecord,
    ContinuousDecisionModel,
    KernelPosteriorResolutionError,
    _dictionary_geometry_from_roots,
    _kernel_posterior_factors,
    _validate_dictionary_geometry,
    _validate_prior_posterior,
)
from transfer.kernel_prior import KernelPriorSpec
from transfer.noise_model import array_digest, metadata_digest
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


def statistics(information, score):
    return {"information_root": information.tolist(), "score": score.tolist(),
            "state_sha256": array_digest(information, score), "noise_digest": "independent-fixture",
            "method": "all_C_actual_noise_per_record_diffuse_QR_response"}


@pytest.mark.parametrize("weak_precision", [1e-8, 1e-50, 1e-100])
def test_huge_weak_posterior_variance_cannot_hide_fourfold_resolved_variance(weak_precision):
    prior = np.diag([1., weak_precision])
    stats = statistics(np.zeros((2, 2)), np.zeros((2, 1)))
    honest = np.diag([1., 1/weak_precision**2])
    _validate_prior_posterior(prior, stats, np.zeros((2, 1)), honest, "independent-fixture")
    wrong = honest.copy()
    wrong[0, 0] = 4.
    with pytest.raises(ValueError, match="posterior"):
        _validate_prior_posterior(prior, stats, np.zeros((2, 1)), wrong, "independent-fixture")


def test_actual_c_diagonal_posterior_matches_independent_exact_two_direction_oracle():
    prior = np.diag([2., 1e-8])
    information = np.diag([3., 1e-8])
    score = np.array([[6.], [1e8]])
    mean = np.array([[18/13], [5e15]])
    covariance = np.diag([1/13, 5e15])
    _validate_prior_posterior(prior, statistics(information, score), mean, covariance, "independent-fixture")
    wrong = covariance.copy()
    wrong[0, 0] *= 4.
    with pytest.raises(ValueError, match="posterior"):
        _validate_prior_posterior(prior, statistics(information, score), mean, wrong, "independent-fixture")


def test_honest_ill_conditioned_covariance_is_checked_as_rounded_factor_gram():
    prior = np.array([[1., 1.], [0., 1e-8]])
    inverse = np.array([[1., -1e8], [0., 1e8]])
    # Dense H has lost its unit eigenvalue at this floating-point scale. The
    # original triangular factor still defines the valid Gaussian model.
    covariance = inverse@inverse.T
    _validate_prior_posterior(prior, statistics(np.zeros((2, 2)), np.zeros((2, 1))),
                              np.zeros((2, 1)), covariance, "independent-fixture")


def test_unresolved_collinear_posterior_has_a_typed_numerical_exit_without_ridge():
    prior = np.array([[1., 1.], [0., 1e-16]])
    inverse = np.array([[1., -1e16], [0., 1e16]])
    with pytest.raises(KernelPosteriorResolutionError) as caught:
        _validate_prior_posterior(prior, statistics(np.zeros((2, 2)), np.zeros((2, 1))),
                                  np.zeros((2, 1)), inverse@inverse.T, "independent-fixture")
    assert caught.value.record()["kind"] == "kernel_posterior_numerical_resolution_failure"


@pytest.mark.parametrize("weak_precision", [1e-8, 1e-50, 1e-100])
def test_qr_column_budget_cannot_hide_coordinated_changes_to_a_small_coordinate(weak_precision):
    prior = np.diag([1., weak_precision])
    information, score = np.zeros((2, 2)), np.zeros((2, 1))
    root, response, _, _, orthogonal = _kernel_posterior_factors(prior, information, score, with_factor=True)
    root[1, 1] *= 2.
    inverse = np.diag([1/root[0, 0], 1/root[1, 1]])
    state = {"precision_root": root, "response": response, "orthogonal": orthogonal}
    # A global ||A|| bound could conceal this tiny but genuine direction.
    # Its own column budget scales with weak_precision and still rejects.
    with pytest.raises(ValueError, match="posterior QR"):
        _validate_prior_posterior(prior, statistics(information, score), np.zeros((2, 1)),
                                  inverse@inverse.T, "independent-fixture", posterior_state=state)
    state["orthogonal"] = orthogonal.copy()
    state["orthogonal"][:, 1] /= 2.
    # Coordinating Q to restore Q R=A now violates dimensionless Q orthogonality.
    with pytest.raises(ValueError, match="posterior QR"):
        _validate_prior_posterior(prior, statistics(information, score), np.zeros((2, 1)),
                                  inverse@inverse.T, "independent-fixture", posterior_state=state)


def _two_direction_fraction_capacity(reference, prior):
    def gram(array):
        values = [[Fraction.from_float(float(v)) for v in row] for row in array]
        return [[sum((row[i]*row[j] for row in values), Fraction()) for j in range(2)] for i in range(2)]

    def trace_inverse_product(matrix, data):
        a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        return (d*data[0][0]-b*data[1][0]-c*data[0][1]+a*data[1][1])/(a*d-b*c)

    information, precision = gram(reference), gram(prior)
    total = [[information[i][j]+precision[i][j] for j in range(2)] for i in range(2)]
    return float(trace_inverse_product(total, information)), float(trace_inverse_product(precision, information))


@pytest.mark.parametrize("exponent", [4, 8, 10, 12, 13, 14])
@pytest.mark.parametrize("g", [1., 3.])
def test_capacity_matches_independent_exact_two_by_two_oracle_without_rejecting_honest_values(exponent, g):
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("cosine", .5, 1), sample_rate_hz=64.)
    reference = np.array([[1.3, .7], [0., 10.**(-exponent)]])
    prior = reference/np.sqrt(g)
    roots = {"reference_information_root": reference, "fitted_information_root": reference,
        "original_reference_root": reference, "ambient_reference_information_root": reference,
        "ambient_prior_precision": prior.T@prior}
    actual = _dictionary_geometry_from_roots(dictionary, roots, prior, 40, 1)
    expected_df, expected_energy = _two_direction_fraction_capacity(reference, prior)
    assert actual["df_ref"] == expected_df
    assert actual["df_fit"] == expected_df
    assert actual["reference_prior_energy"] == expected_energy
    honest = {**actual, "df_ref": expected_df, "df_fit": expected_df, "reference_prior_energy": expected_energy}
    _validate_dictionary_geometry(honest, actual)
    rounded = {**honest, "df_fit": float(np.nextafter(expected_df, np.inf))}
    _validate_dictionary_geometry(rounded, actual)
    with pytest.raises(ValueError, match="capacity diagnostic"):
        _validate_dictionary_geometry({**honest, "df_fit": expected_df+1.}, actual)


def test_nonorthogonal_coordinate_and_voltage_changes_preserve_actual_posterior_binding():
    prior = np.diag([2., 1e-5])
    information = np.diag([3., 1e-5])
    score = np.array([[6.], [1e-5]])
    root, response, covariance, _, orthogonal = _kernel_posterior_factors(
        prior, information, score, with_factor=True)
    mean = solve_triangular(root, response)
    change = np.array([[2., .2], [0., .4]])
    inverse = np.linalg.inv(change)
    functional = np.diag([1., 1e-5])
    original_image = solve_triangular(root.T, functional.T, lower=True).T
    for units in (1., 1e6):
        # Pull back the complete producer state. Re-running QR on separately
        # rounded information/prior matrices is a different floating program;
        # it cannot silently replace the original numerical prediction state.
        state = {"precision_root": root@change, "response": units*response, "orthogonal": orthogonal}
        _validate_prior_posterior(prior@change, statistics(information@change, units*score),
                                  units*inverse@mean, inverse@covariance@inverse.T,
                                  "independent-fixture", posterior_state=state)
        image = solve_triangular(state["precision_root"].T, (functional@change).T, lower=True).T
        np.testing.assert_allclose(image@image.T, original_image@original_image.T, rtol=3e-11, atol=2e-11)
        np.testing.assert_allclose((functional@change)@(units*inverse@mean),
                                   units*functional@mean, rtol=3e-12, atol=units*2e-12)
        changed_mean = units*inverse@mean
        changed_mean[1] *= 4.
        with pytest.raises(ValueError, match="posterior mean"):
            _validate_prior_posterior(prior@change, statistics(information@change, units*score),
                                      changed_mean, inverse@covariance@inverse.T,
                                      "independent-fixture", posterior_state=state)


def _two_direction_fraction_mean(information, prior, score):
    rows = [[Fraction.from_float(float(v)) for v in row] for row in np.vstack((information, prior))]
    targets = [Fraction.from_float(float(v)) for v in np.r_[score[:, 0], np.zeros(2)]]
    gram = [[sum((row[i]*row[j] for row in rows), Fraction()) for j in range(2)] for i in range(2)]
    rhs = [sum((row[i]*target for row, target in zip(rows, targets, strict=True)), Fraction()) for i in range(2)]
    a, b, c, d = gram[0][0], gram[0][1], gram[1][0], gram[1][1]
    determinant = a*d-b*c
    return np.array([[float((d*rhs[0]-b*rhs[1])/determinant)],
                     [float((a*rhs[1]-c*rhs[0])/determinant)]])


@pytest.mark.parametrize("units", [1., 1e6])
def test_independently_rounded_coordinate_frame_cannot_relabel_the_ideal_real_gauge_mean(units):
    prior, information = np.diag([2., 1e-5]), np.diag([3., 1e-5])
    score = np.array([[6.], [1e-5]])
    change = np.array([[2., .2], [0., .4]])
    inverse = np.linalg.inv(change)
    ideal_mean = units*inverse@np.array([[18/13], [.5]])
    changed_prior, changed_information = prior@change, information@change
    actual_mean = _two_direction_fraction_mean(changed_information, changed_prior, units*score)
    # Actual binary information[0,1]=.6000000000000001 while prior[0,1]=.4.
    # The exact binary model differs from the real-matrix ideal, even before QR.
    assert abs(actual_mean[1, 0]-ideal_mean[1, 0]) > units*1e-6
    with pytest.raises(ValueError, match="posterior mean"):
        _validate_prior_posterior(changed_prior, statistics(changed_information, units*score),
            ideal_mean, inverse@np.diag([1/13, 5e9])@inverse.T, "independent-fixture")


@pytest.fixture(scope="module")
def actual_weak_model():
    rng = np.random.default_rng(810)
    # The labeled final target observes only Phi(0)=0. C genuinely contains
    # both labels but no target response information. No synthetic statistics.
    basis = (np.arange(13, dtype=float)/13)[:, None]
    records = [CalibrationRecord(f"actual-C-{i}", rng.normal(size=(31, 2)),
                                 np.array([0, 30]), np.array([0, 1]), right_censored=True)
               for i in range(3)]
    model = ContinuousDecisionModel.fit(records, sfreq=10., channels=("a", "b"),
        candidate_ids=("one", "two"), operator_identity="native-reference-actual-censored-C",
        basis=basis, noise_order=0, conditioning_samples=8,
        kernel_prior=KernelPriorSpec.calibration_completed_g(3., 1e16))
    assert model.kernel_prior.rank_ledger["rank"] == 1
    assert model.kernel_row_covariance[1, 1]/model.kernel_row_covariance[0, 0] > 1e12
    return model


def held(model):
    samples = np.random.default_rng(811).normal(size=(44, 2))
    onsets = np.array([0, 18])
    member = np.array([[1, 0], [0, 1]])
    return (samples, onsets, member), dict(sfreq=model.sfreq, channels=model.channels,
        candidate_ids=model.candidate_ids, operator_identity=model.operator_identity,
        record_id="independent-complete-held")


def test_actual_model_save_resign_load_rejects_small_direction_covariance(tmp_path, actual_weak_model):
    model = actual_weak_model
    original, changed = tmp_path/"actual.npz", tmp_path/"changed.npz"
    model.save(original)
    with np.load(original, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    covariance = arrays["kernel_row_covariance"]
    covariance[0, 0] *= 4.
    # Re-sign every producer payload/metadata checksum, keeping actual C,
    # prior, fitted noise, information root and response unchanged.
    metadata["state_sha256"] = array_digest(*(arrays[key] for key in model._state_arrays()))
    metadata["artifact_sha256"] = metadata_digest({k: v for k, v in metadata.items() if k != "artifact_sha256"})
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(changed, **arrays)
    with pytest.raises(ValueError, match="posterior"):
        ContinuousDecisionModel.load(changed)
    restored = ContinuousDecisionModel.load(original)
    values, settings = held(model)
    np.testing.assert_array_equal(restored.score(*values, **settings).log_evidence,
                                  model.score(*values, **settings).log_evidence)


def test_actual_conditional_score_consumes_cached_precision_and_preserves_diagnostic_roundoff(monkeypatch, actual_weak_model):
    import transfer.continuous_decision as owner

    model = actual_weak_model
    rounded = model.kernel_row_covariance.copy()
    rounded[1, 1] = np.nextafter(rounded[1, 1], np.inf)
    retained = replace(model, kernel_row_covariance=rounded)
    calls = []
    original = owner._conditional_prediction

    def capture(*args, **kwargs):
        assert kwargs["prior_precision_root"] is not None
        assert args[5] is None
        calls.append(kwargs["prior_precision_root"])
        return original(*args, **kwargs)

    def no_rebuild(*args, **kwargs):
        raise AssertionError("held score rebuilt the actual C posterior")

    monkeypatch.setattr(owner, "_conditional_prediction", capture)
    monkeypatch.setattr(owner, "_kernel_posterior_factors", no_rebuild)
    values, settings = held(model)
    before = model.score(*values, **settings)
    after = retained.score(*values, **settings)
    assert len(calls) == 4
    np.testing.assert_array_equal(before.log_evidence, after.log_evidence)
    assert retained.kernel_row_covariance[1, 1] == rounded[1, 1]


def test_load_preserves_producer_qr_and_mean_without_rebuilding_posterior(tmp_path, monkeypatch, actual_weak_model):
    import transfer.continuous_decision as owner

    model = actual_weak_model
    path = tmp_path/"producer.npz"
    model.save(path)

    def no_rebuild(*args, **kwargs):
        raise AssertionError("load regenerated producer posterior QR")

    monkeypatch.setattr(owner, "_kernel_posterior_factors", no_rebuild)
    loaded = ContinuousDecisionModel.load(path)
    np.testing.assert_array_equal(loaded.kernel_mean, model.kernel_mean)
    for key in ("precision_root", "response", "orthogonal"):
        np.testing.assert_array_equal(loaded.posterior_state[key], model.posterior_state[key])


def test_coordinated_changed_posterior_precision_covariance_cannot_keep_actual_c_binding(actual_weak_model):
    model = actual_weak_model
    state = {key: value.copy() for key, value in model.posterior_state.items()}
    state["precision_root"][1, 1] *= 2.
    inverse = np.linalg.inv(state["precision_root"])
    with pytest.raises(ValueError, match="posterior QR"):
        replace(model, posterior_state=state, kernel_row_covariance=inverse@inverse.T)


def test_nonconditional_actual_score_uses_precision_instead_of_dense_covariance_cholesky(monkeypatch):
    rng = np.random.default_rng(812)
    basis = (np.arange(13, dtype=float)/13)[:, None]
    records = [CalibrationRecord(f"legacy-C-{i}", rng.normal(size=(31, 2)),
                                 np.array([0, 30]), np.array([0, 1]), right_censored=True)
               for i in range(3)]
    model = ContinuousDecisionModel.fit(records, sfreq=10., channels=("a", "b"),
        candidate_ids=("one", "two"), operator_identity="native-reference-actual-censored-C",
        basis=basis, ridge=1e-16)
    original = np.linalg.cholesky

    def require_factor(array):
        if array is model.kernel_row_covariance:
            raise AssertionError("ordinary fitted score decomposed dense posterior covariance")
        return original(array)

    monkeypatch.setattr(np.linalg, "cholesky", require_factor)
    values, settings = held(model)
    assert np.isfinite(model.score(*values, **settings).log_evidence).all()
