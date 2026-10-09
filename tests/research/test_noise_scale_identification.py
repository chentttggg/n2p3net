"""Behavioral seams and locked independent oracles for projected noise scales."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import multivariate_normal, multivariate_t

from research.noise_scale_identification import (
    ContrastNumericalError,
    UnidentifiedContrastError,
    calibration_contrast,
    calibration_group_cv,
    contrast_log_density,
    fit_contrast_shape,
    projected_t_fisher_information,
)


def _block(values, index, *, design=None, root=None, role="target_calibration", producer="fixed_K0"):
    y = np.asarray(values, dtype=float)
    return calibration_contrast(y, np.empty((len(y), 0)) if design is None else design,
        np.eye(len(y)) if root is None else root, decision_id=f"decision_{index}",
        recording_id="independent_synthetic_resets", read_support=(index * 20, index * 20 + 10),
        role=role, noise_shape_producer=producer)


def test_migration_projection_preserves_native_support_units_and_shared_erp():
    root = np.array([[1., 0., 0., 0.], [.2, .8, 0., 0.],
                     [.1, .2, 1.2, 0.], [.3, -.1, .1, .9]])
    design = np.column_stack((np.ones(4), np.arange(4)))
    y = np.array([1., -.3, 2., -.1])
    contrast = _block(y, 1, design=design, root=root)
    shifted = _block(y + design @ np.array([31., -14.]), 1, design=design, root=root)
    scaled = _block(y * 1e6, 1, design=design, root=root * 1e6)
    assert contrast.support == (20, 30)  # Explicit native scope, not len(vector)=4.
    assert contrast.dimension == 2
    np.testing.assert_allclose(contrast.projection @ design, 0., atol=1e-14)
    np.testing.assert_allclose(contrast.projection @ root @ root.T @ contrast.projection.T,
                               np.eye(2), atol=1e-14)
    np.testing.assert_allclose(contrast.projected, shifted.projected, atol=1e-13)
    np.testing.assert_allclose(contrast.projected, scaled.projected, atol=1e-14)
    # Shared beta Gaussian uncertainty D @ B vanishes as a whole, rather than
    # being re-drawn independently for every decision.
    np.testing.assert_allclose(contrast.projection @ (design @ [[1.], [.3]]), 0., atol=1e-14)
    with pytest.raises(ValueError):
        contrast.projected.flags.writeable = True
    with pytest.raises(ValueError):
        contrast.projection.flags.writeable = True


def test_migration_scale_and_curvature_match_locked_70digit_oracle():
    blocks = [_block([.3, -1.2], 1), _block([2.], 2), _block([-.5, .4, 1.1], 3)]
    fitted = fit_contrast_shape(blocks, degrees_of_freedom=4.)
    # Independent exact-binary64 input/70-digit root and derivative calculations
    # recorded in dated research evidence. No runtime/test mpmath dependency.
    assert fitted["relative_noise_shape"] == pytest.approx(
        1.04714606991425829009428332785876230320724991717869157168176846878457, abs=1e-12)
    assert fitted["log_likelihood"] == pytest.approx(
        -9.306904649996673802861508867581757983286666917228705682642616222343355, abs=1e-12)
    assert fitted["observed_log_shape_curvature"] == pytest.approx(
        1.916465731625468043236859212218095339033853163199634746781882177760553, abs=1e-12)
    assert abs(fitted["log_shape_score"]) < 1e-12
    assert fitted["independent_scale_draws"] == 3 and fitted["contrast_rows"] == 6
    assert fitted["noise_covariance_multiplier"] == pytest.approx(2 * fitted["relative_noise_shape"])
    gaussian = fit_contrast_shape(blocks, degrees_of_freedom=np.inf)
    assert gaussian["relative_noise_shape"] == pytest.approx(1.1916666666666667, abs=1e-14)
    assert gaussian["gaussian_limit"]
    assert fit_contrast_shape(blocks, degrees_of_freedom=2.)["noise_covariance_multiplier"] is None


def test_migration_complete_decision_cv_and_matched_gaussian_control():
    design = np.array([[1.], [0.], [0.]])
    raw = [[10., .2, -.7], [10., 1.3, .6], [10., -.9, .4],
           [10., .5, 1.8], [10., -1.2, .3], [10., .8, -.2]]
    blocks = [_block(values, i, design=design) for i, values in enumerate(raw)]
    folds = (0, 1, 2, 0, 1, 2)
    result = calibration_group_cv(blocks, degrees_grid=(4., 8., np.inf), folds=folds)
    shared_shift = [_block(np.asarray(values) + design.ravel() * 31., i, design=design)
                    for i, values in enumerate(raw)]
    shifted = calibration_group_cv(shared_shift, degrees_grid=(4., 8., np.inf), folds=folds)
    assert result == shifted
    for arm in result:
        assert arm["scored_independent_decisions"] == 6
        expected = []
        covariance_matched = []
        for fold in arm["fold_fits"]:
            train_ids, score_ids = set(fold["training_decisions"]), set(fold["scored_decisions"])
            assert not train_ids & score_ids and len(train_ids) == 4 and len(score_ids) == 2
            train = [v for v in blocks if v.decision_id in train_ids]
            gaussian_variance = sum(v.projected @ v.projected for v in train) / 8
            for value in (v for v in blocks if v.decision_id in score_ids):
                if arm["gaussian_limit"]:
                    expected.append(multivariate_normal.logpdf(value.projected,
                                    cov=np.eye(2) * gaussian_variance))
                    assert fold["relative_noise_shape"] == pytest.approx(gaussian_variance)
                else:
                    nu = arm["degrees_of_freedom"]
                    expected.append(multivariate_t.logpdf(value.projected, df=nu,
                                    shape=np.eye(2) * fold["relative_noise_shape"]))
                    covariance_matched.append(multivariate_t.logpdf(value.projected, df=nu,
                                    shape=np.eye(2) * gaussian_variance * (nu - 2) / nu))
        assert arm["mean_complete_decision_log_density"] == pytest.approx(np.mean(expected), abs=1e-12)
        if covariance_matched:
            assert arm["gaussian_covariance_matched_control_mean_log_density"] == pytest.approx(
                np.mean(covariance_matched), abs=1e-12)


@pytest.mark.parametrize("partial", [False, True])
def test_rank_admission_cannot_reclassify_unresolved_erp_as_noise(partial):
    design = np.array([[1., 1.], [1., 1. + 1e-13]])
    if partial:
        design = np.vstack((design, [0., 0.]))
    expected = ContrastNumericalError if partial else UnidentifiedContrastError
    with pytest.raises(expected):
        _block([0., 1., .3] if partial else [0., 1.], 0, design=design)
    redundant = _block([1., 2., .3], 0, design=np.ones((3, 2)))
    assert redundant.nuisance_rank == 1 and redundant.dimension == 2


@pytest.mark.parametrize("degrees", [4., np.inf])
def test_zero_residual_has_no_finite_positive_scale_mle(degrees):
    with pytest.raises(UnidentifiedContrastError, match="zero residual"):
        fit_contrast_shape([_block([0., 0.], 0)], degrees_of_freedom=degrees)


@pytest.mark.parametrize("zero_dimensions", [4, 5])
def test_mixed_zero_boundary_or_unbounded_supremum_is_not_an_interior_mle(zero_dimensions):
    blocks = [_block([0.] * zero_dimensions, 0), _block([1.], 1)]
    with pytest.raises(UnidentifiedContrastError, match="boundary"):
        fit_contrast_shape(blocks, degrees_of_freedom=4.)
    assert fit_contrast_shape(blocks, degrees_of_freedom=np.inf)["relative_noise_shape"] > 0


def test_blind_noise_and_an_omitted_erp_direction_remain_attributable():
    design = np.array([[1.], [0.]])
    first = _block([3., .4], 0, design=design, root=np.diag([1., 1.]))
    second = _block([3., .4], 0, design=design, root=np.diag([10., 1.]))
    np.testing.assert_array_equal(first.projected, second.projected)
    assert fit_contrast_shape([first], degrees_of_freedom=4.) == fit_contrast_shape(
        [second], degrees_of_freedom=4.)
    omitted = _block([0., 1.], 0, design=design)
    assert fit_contrast_shape([omitted], degrees_of_freedom=4.)["relative_noise_shape"] == pytest.approx(1.)
    with pytest.raises(UnidentifiedContrastError):
        _block([0., 1.], 0, design=np.eye(2))


def test_single_decision_scale_profile_retains_gaussian_limit_and_real_unit_count():
    blocks = [_block(np.full(128, np.sqrt(2.)), 0)]
    profile = [fit_contrast_shape(blocks, degrees_of_freedom=nu) for nu in (4., 8., 32., np.inf)]
    assert all(v["independent_scale_draws"] == 1 and v["contrast_rows"] == 128 for v in profile)
    assert all(v["relative_noise_shape"] == pytest.approx(2., abs=1e-13) for v in profile)
    assert all(left["log_likelihood"] < right["log_likelihood"]
               for left, right in zip(profile[:-1], profile[1:], strict=True))
    assert profile[0]["observed_log_shape_curvature"] == pytest.approx(1.9393939393939394, abs=1e-13)
    with pytest.raises(ValueError, match="folds"):
        calibration_group_cv(blocks, degrees_grid=(4., np.inf), folds=(0,))


def test_fisher_matches_locked_independent_beta_integrals_and_information_saturation():
    # 70-digit independent Beta-score integration: these are exact values,
    # not a second copy of the implementation's polygamma formula.
    fisher = projected_t_fisher_information(2, 4.)
    assert fisher["log_shape_information"] == pytest.approx(1 / 2, abs=1e-14)
    assert fisher["cross_information"] == pytest.approx(-1 / 24, abs=1e-14)
    assert fisher["nu_information"] == pytest.approx(1 / 96, abs=1e-14)
    assert fisher["nu_information_after_profile_shape"] == pytest.approx(1 / 144, abs=1e-14)
    larger = projected_t_fisher_information(128, 4.)
    assert larger["independent_scale_draws"] == 1
    assert larger["log_shape_information"] < 2.
    assert larger["nu_information_after_profile_shape"] < .04


def test_common_decision_scale_is_not_a_product_of_independent_time_scales():
    whole = contrast_log_density([5., .2], degrees_of_freedom=4., log_shape=np.log(.5))
    split = sum(contrast_log_density([value], degrees_of_freedom=4., log_shape=np.log(.5))
                for value in [5., .2])
    assert whole - split == pytest.approx(-1.1324979903789378, abs=1e-12)


@pytest.mark.parametrize("violation", ["role", "duplicate", "producer", "overlap", "no_Gaussian"])
def test_declared_calibration_input_ledger_rejects_scope_or_unit_changes(violation):
    first, second = _block([.3, .4], 0), _block([.7, -.2], 1)
    if violation == "role":
        second = replace(second, role="held_diagnostic")
    elif violation == "duplicate":
        second = first
    elif violation == "producer":
        second = replace(second, noise_shape_producer="different_K0")
    elif violation == "overlap":
        second = replace(second, support=(5, 15))
    if violation == "no_Gaussian":
        with pytest.raises(ValueError, match="Gaussian"):
            calibration_group_cv([first, second], degrees_grid=(4., 8.), folds=(0, 1))
    else:
        with pytest.raises(ValueError):
            fit_contrast_shape([first, second], degrees_of_freedom=4.)


@pytest.mark.parametrize("entry", ["projection", "density", "fit", "fisher"])
def test_nonrepresentable_reference_results_are_explicit_failures(entry):
    with pytest.raises(ContrastNumericalError):
        if entry == "projection":
            _block([1.], 0, root=[[1e-320]])
        elif entry == "density":
            contrast_log_density([1.], degrees_of_freedom=np.nextafter(0., 1.), log_shape=0.)
        elif entry == "fit":
            fit_contrast_shape([_block([1e300], 0)], degrees_of_freedom=4.)
        else:
            projected_t_fisher_information(2, 1e308)


def test_nonzero_energy_underflow_is_not_misreported_as_a_zero_noise_mle():
    with pytest.raises(ContrastNumericalError, match="energy"):
        fit_contrast_shape([_block([1e-300], 0)], degrees_of_freedom=4.)
