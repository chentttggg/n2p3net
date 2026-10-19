"""Independent Gamma-ratio and extreme radial witnesses through real consumers."""

import numpy as np
import pytest
from scipy.integrate import quad

from research.joint_response_sampling import (
    SamplingResolutionError,
    conditional_tail_components,
    prepare_model,
    prepare_prediction,
)
from research.joint_response_scale import DecisionBlock, FixedHyperparameters, GlobalGaussianPrior
from research.noise_scale_identification import ContrastNumericalError, contrast_log_density
from research.student_t_numerics import student_t_log_density


def _components(n, nu, observations):
    # No calibration, no ERP image, no Gibbs: only the actual public prediction
    # path can establish the normalization of its conditional-t components.
    prior = GlobalGaussianPrior([0.], [[1.]], "external")
    hyper = FixedHyperparameters(nu, 1., "externally_fixed")
    model = prepare_model(prior, (), hyper)
    test = DecisionBlock("T", observations, np.zeros((n, 1)), np.eye(n), "read_only_inference")
    prediction = prepare_prediction(model, test, conditioning_rows=1)
    result = conditional_tail_components(prediction, [[0.]])
    assert prediction.hyperparameters.degrees_of_freedom == nu
    return tuple(float(value[0]) for value in result)


@pytest.mark.parametrize("nu", [4., 1e8, 1e16, 1e30, 1e308, 1e-300, 1e-320])
def test_actual_two_row_density_has_exact_gamma_normalizer_for_every_finite_df(nu):
    # Gamma(a+1)/Gamma(a)=a, so the two-dimensional peak is exactly 1/(2*pi).
    full, _, _ = _components(2, nu, [0., 0.])
    assert full == -np.log(2 * np.pi)
    assert contrast_log_density([0., 0.], degrees_of_freedom=nu, log_shape=0.) == full


@pytest.mark.parametrize("n,nu,q,expected", [
    (1, 4., 1., -1.538688131297250626272188853226590258015512559128034345),
    (3, 4., 9., -6.720855255302623374917396705224783916217140573124514051),
    (3, 1e8, 1., -3.256815604614018217007655875883519336250865754246671667),
    (3, 1e16, 9., -7.256815599614017475340989209217465419584192420457350238),
    (9, 1e30, 1., -8.770446798842054676022967627639058758752577262968723901),
    (3, 1e-300, 0., 342.85673970213756179719128069078083960650837571018485),
    (3, 1e-320, 1., -739.3582651379431969439647972662011908383948518126913463),
    (9, 1e-320, 9., -750.1054465850266972461026255033457349455804750332496635),
    (3, 31.999, 1., -3.27235968074275654655417303868903707324176159338852667),
    (3, 32., 1., -3.272359197535421707921971435627500689430048217182125678),
])
def test_odd_dimensions_match_locked_independent_90_digit_log_gamma_formula(n, nu, q, expected):
    # Constants use exact binary64 nu/q interpreted by mpmath, with the direct
    # high-precision log-Gamma formula, not the implemented recurrence/Stirling.
    y = np.zeros(n)
    y[0] = np.sqrt(q)
    contrast = contrast_log_density(y, degrees_of_freedom=nu, log_shape=0.)
    if n == 1:
        _, actual, _ = _components(2, nu, [y[0], 0.])
    else:
        actual, _, _ = _components(n, nu, y)
    assert actual == pytest.approx(expected, rel=0, abs=5e-13)
    assert contrast == pytest.approx(expected, rel=0, abs=5e-13)


def test_underflowed_energy_over_df_preserves_radial_penalty_and_explicit_gaussian_control():
    # Center the exact two-row normalizer to expose a penalty much smaller than
    # one ulp of an ordinary uncentered log density. q/nu rounds to zero.
    q, nu = 1e-100, 1e308
    assert q / nu == 0.
    actual = student_t_log_density(q, degrees_of_freedom=nu, dimension=2,
                                  noise_logdet=-2 * np.log(2 * np.pi))
    assert actual == -.5 * q
    gaussian = student_t_log_density(q, degrees_of_freedom=np.inf, dimension=2,
                                    noise_logdet=-2 * np.log(2 * np.pi))
    assert gaussian == actual
    maximum = np.finfo(float).max
    large = student_t_log_density(maximum, degrees_of_freedom=maximum, dimension=2,
                                 noise_logdet=0.)
    assert np.isfinite(large)
    assert large == pytest.approx(-(.5 * maximum) * np.log(2), rel=2e-15)


@pytest.mark.parametrize("z,nu,log_shape,expected", [
    (1e154, 1e-300, 800., -1046.06677939985668586547380640100899330567882276424189882847),
    (1e-150, 4., -800., 129.423726395322263292246509332247214654649048794619189599461),
    (1e300, 4., 1380., -692.928122648821984117132815004146095733334643094750210365205),
    (1e-300, 4., -1380., 688.89004841828644274593031728209837546456909775911323333762),
])
def test_contrast_standardization_keeps_finite_physical_energy_without_raw_square_or_exp(z, nu, log_shape, expected):
    actual = contrast_log_density([z], degrees_of_freedom=nu, log_shape=log_shape)
    assert actual == pytest.approx(expected, rel=0, abs=5e-13)


def test_contrast_log_shape_uses_representable_density_even_if_twice_log_shape_overflows():
    assert contrast_log_density([0., 0.], degrees_of_freedom=4., log_shape=1e308) == -1e308
    with pytest.raises(ContrastNumericalError):
        contrast_log_density(np.zeros(4), degrees_of_freedom=4., log_shape=1e308)


@pytest.mark.parametrize("multiplier", [1., 3.])
def test_unrepresentable_gamma_shape_is_typed_in_both_actual_consumers(multiplier):
    nu = multiplier * np.nextafter(0., 1.)
    assert nu > 0 and 2 * (nu / 2) != nu
    with pytest.raises(SamplingResolutionError) as sampling:
        _components(2, nu, [0., 0.])
    with pytest.raises(ContrastNumericalError) as contrast:
        contrast_log_density([0., 0.], degrees_of_freedom=nu, log_shape=0.)
    assert "Gamma shape nu/2" in str(sampling.value.__cause__)
    assert "Gamma shape nu/2" in str(contrast.value.__cause__)


@pytest.mark.parametrize("nu", [4., 1e16, 1e-300, np.inf])
def test_actual_conditional_tail_density_integrates_to_one(nu):
    # Zero raw prefix and D=0 give a one-dimensional conditional t with df nu+1
    # and scale sqrt(nu/(nu+1)); infinity is the explicitly separate Gaussian.
    scale = 1. if np.isposinf(nu) else np.sqrt(nu / (nu + 1))
    def density(x):
        _, _, tail = _components(2, nu, [0., scale * x])
        return np.exp(tail + np.log(scale))
    mass, error = quad(density, -np.inf, np.inf, epsabs=2e-10, epsrel=2e-10)
    assert mass == pytest.approx(1., rel=0, abs=2e-10)
    assert error < 2e-10


def test_explicit_gaussian_components_retain_the_original_actual_normal_density():
    full, prefix, tail = _components(2, np.inf, [1., 3.])
    assert full == pytest.approx(-np.log(2 * np.pi) - 5., rel=0, abs=1e-15)
    assert prefix == pytest.approx(-.5 * np.log(2 * np.pi) - .5, rel=0, abs=1e-15)
    assert tail == pytest.approx(-.5 * np.log(2 * np.pi) - 4.5, rel=0, abs=1e-15)
