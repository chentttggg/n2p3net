"""Independent response-space oracles for explicit quotient and completion."""
import copy

import numpy as np
import pytest

from transfer.kernel_prior import (
    CompiledKernelPrior,
    KernelPriorRankError,
    KernelPriorSpec,
    calibration_reference_roots,
    compile_kernel_prior,
)
from transfer.noise_model import array_digest, metadata_digest
from transfer.predictive_prior import (
    CompiledPredictivePrior,
    PhysicalResponseMetric,
    PredictivePriorSupportError,
)


def metric(width=1):
    t = np.linspace(0., 1., 13)
    basis = np.column_stack([t**i for i in range(width)])
    binding = {"sample_rate_hz": 10., "basis_sha256": array_digest(basis),
               "operator_identity": "actual_native_reference", "channels": ["Pz"],
               "fit_scope": "target_calibration"}
    return PhysicalResponseMetric.from_basis(basis, sample_rate_hz=10., binding=binding)


def metric_matrix(value):
    """Dense physical time quadrature, independent of the metric QR owner."""
    phi = value.basis
    physical = np.block([[phi, np.zeros_like(phi)], [phi, phi]])@value.coordinate_map
    return physical.T@physical/(2*len(phi))


def rank_one_geometry(tau=None, g=3., physical=None):
    physical = metric() if physical is None else physical
    # Exactly known reference information c*a*a^T, c=5, a=[1,2].
    reference = np.array([[np.sqrt(5), 2*np.sqrt(5)], [0., 0.]])
    return CompiledPredictivePrior(physical, reference, reference, 40, 1, g, tau)


def rank_one_oracle(physical, tau, g):
    """Closed form from one observed linear functional, without SVD or pinv."""
    inverse = np.linalg.solve(metric_matrix(physical), np.eye(2))
    a = np.array([1., 2.])
    v, scale = inverse@a, a@inverse@a
    observed = g*np.outer(v, v)/(5*scale**2)
    unseen = inverse-np.outer(v, v)/scale
    return observed if tau is None else observed+tau*unseen


def test_response_metric_includes_physical_class_cross_terms_and_actual_mean_grid():
    physical = metric(2)
    np.testing.assert_allclose(physical.precision_root.T@physical.precision_root,
                               metric_matrix(physical), rtol=2e-12, atol=2e-16)
    assert np.any(metric_matrix(physical)[:2, 2:] != 0)
    record = physical.record()
    assert record["actual_seconds"] == 1.3
    assert record["quadrature_weight"] == 1/26
    assert record["coefficient_units"] == "volts"


def test_quotient_matches_independent_estimable_variance_and_refuses_new_direction():
    quotient = rank_one_geometry()
    assert quotient.rank_ledger["rank"] == 1
    estimable = np.array([[1., 2.], [-2., -4.]])
    expected = estimable@rank_one_oracle(quotient.metric, None, 3.)@estimable.T
    np.testing.assert_allclose(quotient.predictive_covariance(estimable), expected, rtol=2e-12, atol=2e-15)
    assert quotient.precision_root is None
    with pytest.raises(PredictivePriorSupportError) as caught:
        quotient.predictive_covariance(np.array([[0., 1.]]))
    failure = caught.value.record()
    assert failure["kind"] == "kernel_quotient_prediction_failure"
    assert failure["unsupported_functional_rows"] == [0]
    assert failure["outside_relative_norm_by_row"][0] > failure["relative_tolerance_by_row"][0]


@pytest.mark.parametrize("scale", [1e-250, 1e-12, 1., 1e12, 1e250])
def test_quotient_support_does_not_change_when_functional_norm_would_underflow(scale):
    quotient = rank_one_geometry()
    assert quotient.prediction_support(scale*np.array([[1., 2.]]))["supported"]
    assert not quotient.prediction_support(scale*np.array([[0., 1.]]))["supported"]


@pytest.mark.parametrize("identified_scale,unseen_scale", [(1e20, 1.), (1., 1e-300), (1e250, 1e-300)])
def test_quotient_mixed_row_scales_cannot_hide_an_unseen_response(identified_scale, unseen_scale):
    # Equal-class h0=z0+z1, h1=z0-z1 has physical energy z0**2+z1**2.
    # C sees z0 only; every nonzero multiple of z1 remains unidentifiable.
    physical = PhysicalResponseMetric(np.ones((3, 1)), 64., {"scope": "independent_two_response_quotient"},
                                      np.array([[1., 1.], [0., -2.]]))
    reference = np.array([[1., 0.], [0., 0.]])
    quotient = CompiledPredictivePrior(physical, reference, reference, 40, 1, 3.)
    functional = np.array([[identified_scale, 0.], [0., unseen_scale], [0., 0.]])
    support = quotient.prediction_support(functional)
    assert not support["supported"]
    assert support["unsupported_functional_rows"] == [1]
    assert support["outside_relative_norm_by_row"][1] > support["relative_tolerance_by_row"][1]
    with pytest.raises(PredictivePriorSupportError):
        quotient.predictive_covariance(functional)


def test_quotient_zero_and_tiny_identified_rows_have_their_own_support_semantics():
    quotient = rank_one_geometry()
    assert quotient.prediction_support(np.array([[0., 0.], [1e-300, 2e-300], [1e20, 2e20]]))["supported"]
    assert quotient.prediction_support(np.zeros((3, 2)))["supported"]
    np.testing.assert_array_equal(quotient.predictive_covariance(np.zeros((3, 2))), np.zeros((3, 3)))


def test_completion_changes_only_unseen_response_and_agrees_with_rank_one_closed_form():
    held = np.eye(2)
    narrow, broad = rank_one_geometry(.2), rank_one_geometry(5.)
    for result in (narrow, broad):
        expected = rank_one_oracle(result.metric, result.completion_variance, 3.)
        np.testing.assert_allclose(result.predictive_covariance(held), expected, rtol=2e-12, atol=3e-14)
        np.testing.assert_allclose(result.precision_root.T@result.precision_root,
                                   np.linalg.inv(expected), rtol=2e-12, atol=3e-13)
    np.testing.assert_allclose(narrow.predictive_covariance([[1., 2.]]),
                               broad.predictive_covariance([[1., 2.]]), rtol=2e-12, atol=2e-15)
    assert broad.predictive_covariance([[0., 1.]])[0, 0] > narrow.predictive_covariance([[0., 1.]])[0, 0]


def test_nonorthogonal_mp_counterexample_and_physical_completion_congruence():
    change = np.array([[1., 1.], [0., 1.]])
    d, held = np.array([[1., 0.]]), np.array([[0., 1.]])
    # Explicit MP values: training projector unchanged, unseen held variance differs.
    mp_original = np.diag([1., 0.])
    mp_changed = np.ones((2, 2))/4
    assert (held@mp_original@held.T)[0, 0] == 0.
    assert ((held@change)@mp_changed@(held@change).T)[0, 0] == .25
    np.testing.assert_array_equal(d@mp_original@d.T, d@change@mp_changed@change.T@d.T)
    original = rank_one_geometry(.7)
    transformed_metric = original.metric.pullback(change, binding={"physical_parameterization": "S"})
    transformed = CompiledPredictivePrior(transformed_metric, original.reference_root@change,
        original.design_root@change, 40, 1, 3., .7)
    inverse = np.linalg.inv(change)
    np.testing.assert_allclose(transformed.predictive_covariance(np.eye(2)),
        inverse@original.predictive_covariance(np.eye(2))@inverse.T, rtol=2e-12, atol=4e-14)
    np.testing.assert_allclose(transformed.predictive_covariance(held@change),
                               original.predictive_covariance(held), rtol=2e-12, atol=4e-14)


def test_zero_information_quotient_has_no_response_claim_and_completion_is_physical_prior():
    physical = metric()
    for tau in (None, .6):
        result = CompiledPredictivePrior(physical, np.zeros((2, 2)), np.zeros((2, 2)), 40, 1, 3., tau)
        assert result.rank_ledger["rank"] == 0
        np.testing.assert_array_equal(result.predictive_covariance(np.zeros((2, 2))), np.zeros((2, 2)))
        if tau is None:
            with pytest.raises(PredictivePriorSupportError):
                result.predictive_covariance(np.eye(2))
        else:
            np.testing.assert_allclose(result.predictive_covariance(np.eye(2)),
                tau*np.linalg.inv(metric_matrix(physical)), rtol=2e-12, atol=8e-16)


def test_numerically_weak_resolved_direction_is_not_silently_ridged():
    physical = metric()
    # Build in physical-normalized coordinates so the singular values are known.
    for scale in (1e-12, 1., 1e12):
        root = scale*np.diag([1., 1e-10])@physical.precision_root
        result = CompiledPredictivePrior(physical, root, root, 40, 1, 3., .6)
        assert result.rank_ledger["rank"] == 2
        expected = np.linalg.solve(root, np.eye(2))
        expected = 3*expected@expected.T
        np.testing.assert_allclose(result.predictive_covariance(np.eye(2)), expected, rtol=3e-7)
    root = np.diag([1., 1e-16])@physical.precision_root
    result = CompiledPredictivePrior(physical, root, root, 40, 1, 3., .6)
    assert result.rank_ledger["rank"] == 1
    assert not result.rank_ledger["statistical_identifiability_claimed"]


def test_affine_projection_roundoff_cannot_become_calibration_information():
    physical = metric()
    values = [np.column_stack((np.ones(43), np.linspace(-1., 1., 43)))]
    reference, original = calibration_reference_roots(values)
    result = CompiledPredictivePrior(physical, reference, original, 43, 1, 3., .6)
    assert result.rank_ledger["rank"] == 0


def test_actual_legal_censored_event_history_does_not_identify_all_complete_held_candidates():
    from transfer.continuous_decision import _event_design, _validate_record

    physical = metric(2)
    # C has both event classes but its final labeled target is right censored
    # after one response sample. Its target time-slope coordinate is unseen.
    onsets = np.array([0, 30], dtype=np.int64)
    _validate_record(np.zeros((31, 1)), onsets, 1, len(physical.basis), allow_right_censor=True)
    design = _event_design(onsets, [0, 1], 31, physical.basis)
    reference, original = calibration_reference_roots([design])
    quotient = CompiledPredictivePrior(physical, reference, original, 31, 1, 3.)
    assert quotient.rank_ledger["rank"] == 3
    held_onsets = np.array([0, 18], dtype=np.int64)
    _validate_record(np.zeros((44, 1)), held_onsets, 1, len(physical.basis))
    for target in ([0, 1], [1, 0]):
        held = _event_design(held_onsets, target, 44, physical.basis)
        assert not quotient.prediction_support(held)["supported"]
        with pytest.raises(PredictivePriorSupportError):
            quotient.predictive_covariance(held)
        complete = CompiledPredictivePrior(physical, reference, original, 31, 1, 3., .7)
        assert np.isfinite(complete.predictive_covariance(held)).all()


def test_completion_reduces_to_original_g_when_physical_c_is_full_rank():
    physical = metric()
    t = np.linspace(-1., 1., 43)
    values = [np.column_stack((np.sin(4*t), np.cos(5*t)))]
    arguments = dict(record_ids=["C1"], fit_scope="target_calibration", binding=physical.binding)
    original = compile_kernel_prior(KernelPriorSpec.calibration_g(3.), values, **arguments)
    complete = compile_kernel_prior(KernelPriorSpec.calibration_completed_g(3., .6), values,
                                    physical_metric=physical, **arguments)
    np.testing.assert_allclose(complete.row_covariance, original.row_covariance, rtol=2e-12, atol=2e-16)
    assert complete.derived_precision is None
    assert complete.spec.rank_policy == "physical_response_completion"


def test_true_compiler_requires_explicit_completion_preserves_old_rank_exit_and_replays():
    physical = metric()
    t = np.linspace(-1., 1., 43)
    values = [np.column_stack((np.sin(4*t), 2*np.sin(4*t)))]
    arguments = dict(record_ids=["C1"], fit_scope="target_calibration", binding=physical.binding)
    with pytest.raises(KernelPriorRankError):
        compile_kernel_prior(KernelPriorSpec.calibration_g(3.), values, **arguments)
    spec = KernelPriorSpec.calibration_completed_g(3., .6)
    with pytest.raises(ValueError, match="physical"):
        compile_kernel_prior(spec, values, **arguments)
    complete = compile_kernel_prior(spec, values, physical_metric=physical, **arguments)
    assert complete.rank_ledger["rank"] == 1
    replay = CompiledKernelPrior.from_record(complete.record())
    assert replay.digest() == complete.digest()
    np.testing.assert_array_equal(replay.precision_root, complete.precision_root)
    with pytest.raises(ValueError):
        compile_kernel_prior(spec, values, physical_metric=physical, **{**arguments, "fit_scope": "held_test"})


def test_voltage_unit_and_covariance_factorization_gauge_are_distinct():
    complete = rank_one_geometry(.7)
    covariance = complete.predictive_covariance(np.eye(2))
    voltage_scale, gauge = 1e6, 5.
    sigma = np.array([[2.]])
    np.testing.assert_allclose(np.kron(voltage_scale**2*sigma, covariance),
                               voltage_scale**2*np.kron(sigma, covariance))
    changed = CompiledPredictivePrior(complete.metric, complete.reference_root/np.sqrt(gauge),
        complete.design_root/np.sqrt(gauge), 40, 1, 3., gauge*.7)
    np.testing.assert_allclose(np.kron(sigma/gauge, changed.predictive_covariance(np.eye(2))),
                               np.kron(sigma, covariance), rtol=2e-12, atol=1e-14)
    wrong = CompiledPredictivePrior(complete.metric, complete.reference_root/np.sqrt(gauge),
        complete.design_root/np.sqrt(gauge), 40, 1, 3., .7)
    assert not np.allclose(wrong.predictive_covariance(np.eye(2))/gauge, covariance)


def test_resigned_completion_covariance_and_physical_binding_forgery_are_rejected():
    complete = rank_one_geometry(.7)
    payload = copy.deepcopy(complete.record())
    payload["covariance_factor"][0][0] += .1
    payload["artifact_sha256"] = metadata_digest({k: v for k, v in payload.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError, match="numerical relation"):
        CompiledPredictivePrior.from_record(payload)
    with pytest.raises(ValueError, match="binding"):
        complete.metric.validate_binding({"operator_identity": "different_reference"})
    forged = complete.metric.record()
    forged["coordinate_map"][0][0] = 2.
    forged["artifact_sha256"] = metadata_digest({k: v for k, v in forged.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError, match="coordinate binding"):
        PhysicalResponseMetric.from_record(forged)


def test_huge_weak_variance_cannot_mask_a_resigned_change_to_small_variance():
    physical = metric()
    root = np.diag([1., 1e-8])@physical.precision_root
    complete = CompiledPredictivePrior(physical, root, root, 40, 1, 3., .7)
    payload = complete.record()
    # Column 1 represents the small-variance identified singular direction;
    # global ||H|| is dominated by column 2 at about 1e16. A 4x variance change
    # to column 1 would pass a sole Frobenius-relative comparison of H.
    for row in payload["covariance_factor"]:
        row[0] *= 2
    payload["artifact_sha256"] = metadata_digest({k: v for k, v in payload.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError, match="resolved direction"):
        CompiledPredictivePrior.from_record(payload)


@pytest.mark.parametrize("field", ["singular_values", "absolute_threshold", "rank"])
def test_resigned_rank_ledger_cannot_broadcast_or_change_declared_scalar_types(field):
    physical = metric()
    root = 4*physical.precision_root
    geometry = CompiledPredictivePrior(physical, root, root, 40, 1, 3., .7)
    payload = geometry.record()
    ledger = payload["rank_ledger"]
    if field == "singular_values":
        ledger[field] = ledger[field][:1]
    elif field == "absolute_threshold":
        ledger[field] = [ledger[field]]
    else:
        ledger[field] = float(ledger[field])
    payload["artifact_sha256"] = metadata_digest({k: v for k, v in payload.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError, match="rank"):
        CompiledPredictivePrior.from_record(payload)


@pytest.mark.parametrize("bad", [True, 0., -1., np.inf, 1+2j])
def test_completion_variance_cannot_be_an_implicit_invalid_regularizer(bad):
    with pytest.raises(ValueError):
        KernelPriorSpec.calibration_completed_g(3., bad)
