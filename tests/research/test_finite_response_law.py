"""New finite-law implementation correspondence and persistent controls."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from research.finite_response_law import (
    FiniteResponseLaw,
    FiniteResponseResolutionError,
    MeanResponseDesign,
    ResponseMomentComponents,
)
from research.latent_response_reference import ResponseState


def _law(masses=(1., 3.)):
    return FiniteResponseLaw.from_states(tuple(ResponseState(1, 1, i/32, 1+i/4, mass)
        for i, mass in enumerate(masses)))


def test_one_mass_owner_normalizes_overflowing_raw_sum_and_retains_original_records(tmp_path):
    law = _law((1e308, 1e308))
    assert [state.probability for state in law.states] == [1e308, 1e308]
    np.testing.assert_array_equal(law.probability, [.5, .5])
    np.testing.assert_allclose(law.log_probability, [-np.log(2)]*2, rtol=0, atol=1e-16)
    record = law.record()
    record["implementation_sha256"] = "f"*64
    old = FiniteResponseLaw.from_record(record)
    old.save(tmp_path/"law.json")
    restored = FiniteResponseLaw.load(tmp_path/"law.json")
    assert restored.record() == record and restored.digest() == old.digest()
    assert restored.log_probability.tobytes() == law.log_probability.tobytes()
    with pytest.raises(ValueError):
        restored.log_probability.setflags(write=True)
    changed = deepcopy(record)
    changed["log_probability"][0] = np.nextafter(changed["log_probability"][0], 0.)
    with pytest.raises(ValueError, match="original positive measure"):
        FiniteResponseLaw.from_record(changed)


def test_positive_log_mass_survives_unrepresentable_control_probability():
    law = _law((np.nextafter(0., 1.), 2.))
    assert np.isfinite(law.log_probability).all() and law.log_probability[0] < -740
    with pytest.raises(FiniteResponseResolutionError, match="unrepresentable"):
        _ = law.probability
    # A later likelihood can give this atom substantial posterior mass.
    logjoint = law.log_probability+np.array([750., 0.])
    assert np.exp(logjoint[0]) > 1
    assert FiniteResponseLaw.from_record(law.record()).log_probability.tobytes() == law.log_probability.tobytes()


@pytest.mark.parametrize("state", [ResponseState(1, 1, -.01, 1, 1), ResponseState(1, 1, 0, 0, 1),
    ResponseState(1, 1, 0, 1, 0), ResponseState(2, 1, 0, 1, 1)])
def test_first_programme_rejects_a_different_shape_or_amplitude_law(state):
    with pytest.raises(ValueError):
        FiniteResponseLaw.from_states((state,))


def test_actual_mean_design_keeps_shape_uncertainty_distinct_and_replays():
    law = _law((1, 1))
    designs = np.array([[[1.], [0.]], [[0.], [1.]]])
    actual = law.mean_design(designs)
    np.testing.assert_array_equal(actual.matrix, [[.5], [.5]])
    assert not np.array_equal(actual.matrix, designs[0])
    assert np.sum(actual.matrix**2) == .5
    assert np.mean(np.sum(designs**2, axis=(1, 2))) == 1
    saved = actual.record()
    saved["binding"]["compiler_sha256"] = "e"*64
    restored = MeanResponseDesign.from_record(saved, law=law, designs=designs)
    assert restored.record() == saved
    changed = deepcopy(saved)
    changed["matrix"][0][0] = .6
    with pytest.raises(ValueError, match="mean design"):
        MeanResponseDesign.from_record(changed, law=law, designs=designs)


def test_direct_factor_matches_full_mixture_and_keeps_cross_selection_process():
    law = _law((1, 2))
    mean = np.array([1., 2.])
    source_factor = np.array([[.5, 0], [.25, 1/3]])
    covariance = source_factor@source_factor.T
    components = law.moment_components(mean, source_factor)
    bank = components.bank(2)
    expected_mean = np.zeros(8)
    expected_second = np.zeros((8, 8))
    for states in product(range(2), repeat=2):
        selector = np.zeros((8, 2))
        for selection, atom in enumerate(states):
            selector[(selection*2+atom)*2:(selection*2+atom+1)*2] = np.eye(2)
        probability = law.probability[states[0]]*law.probability[states[1]]
        image = selector@mean
        expected_mean += probability*image
        expected_second += probability*(selector@covariance@selector.T+np.outer(image, image))
    expected_covariance = expected_second-np.outer(expected_mean, expected_mean)
    np.testing.assert_allclose(bank["mean"], expected_mean, atol=4e-16, rtol=0)
    np.testing.assert_allclose(bank["factor"]@bank["factor"].T, expected_covariance, atol=3e-15, rtol=0)
    assert expected_covariance[0, 4] == pytest.approx(1/36)
    assert np.linalg.matrix_rank(expected_covariance) == 6
    assert components.local_factor.shape == (4, 6)
    assert components.binding["zero_or_weak_directions_removed"] is False
    larger = components.bank(3)
    assert larger["factor"][:8, :14].tobytes() == bank["factor"].tobytes()
    assert np.count_nonzero(larger["factor"][:8, 14:]) == 0
    saved = components.record()
    saved["binding"]["compiler_sha256"] = "d"*64
    restored = ResponseMomentComponents.from_record(saved, law=law, mean=mean, factor=source_factor)
    assert restored.record() == saved
    changed = deepcopy(saved)
    changed["local_factor"][0][0] += 1e-5
    with pytest.raises(ValueError, match="moment image"):
        ResponseMomentComponents.from_record(changed, law=law, mean=mean, factor=source_factor)


def test_moment_compilation_errors_cover_actual_coefficient_operations():
    law = _law((1, 2))
    mean = np.array([.2, -.7])
    source_factor = np.array([[.1, 0.], [.3, .6]])
    result = law.moment_components(mean, source_factor)
    with localcontext() as context:
        context.prec = 100
        pi = [Decimal(1)/3, Decimal(2)/3]
        square_roots = [value.sqrt() for value in pi]
        for atom in range(2):
            for coefficient in range(2):
                row = atom*2+coefficient
                expected_mean = pi[atom]*Decimal.from_float(mean[coefficient])
                assert abs(Decimal.from_float(result.mean[row])-expected_mean) <= Decimal.from_float(result.mean_error[row])
                for contrast in range(2):
                    categorical = (square_roots[atom] if atom == contrast else Decimal(0))-pi[atom]*square_roots[contrast]
                    for coordinate in range(3):
                        value = source_factor[coefficient, coordinate] if coordinate < 2 else mean[coefficient]
                        expected = categorical*Decimal.from_float(value)
                        column = contrast*3+coordinate
                        assert abs(Decimal.from_float(result.local_factor[row, column])-expected) <= Decimal.from_float(result.local_factor_error[row, column])


def test_single_state_keeps_exact_source_and_zero_local_directions():
    law = _law((7,))
    mean, factor = np.array([-0., 1.]), np.array([[.5, 0.], [.25, 0.]])
    result = law.moment_components(mean, factor)
    assert result.mean.tobytes() == mean.tobytes()
    assert result.shared_factor.tobytes() == factor.tobytes()
    assert np.count_nonzero(result.local_factor) == np.count_nonzero(result.local_factor_error) == 0
    assert np.count_nonzero(result.mean_error) == 0
    designs = np.array([[[-0., 1.], [0., 2.]]])
    assert law.mean_design(designs).matrix.tobytes() == designs[0].tobytes()
    with pytest.raises(ValueError):
        result.shared_factor.setflags(write=True)


def test_exact_owner_log_rounding_error_covers_the_rational_normalizer():
    law = _law((.1, .2, .7))
    masses = [Fraction.from_float(state.probability) for state in law.states]
    total = sum(masses)
    with localcontext() as context:
        context.prec = 120
        for index, mass in enumerate(masses):
            exact = mass/total
            reference = (Decimal(exact.numerator)/Decimal(exact.denominator)).ln()
            difference = abs(Decimal.from_float(law.log_probability[index])-reference)
            assert difference <= Decimal.from_float(law.log_probability_error[index])


def test_compact_native_contraction_matches_explicit_state_bank_and_keeps_errors():
    law = _law((1, 2))
    mean = np.array([.2, -.7, .3, .8])
    source_factor = np.diag([.4, .3, .2, .6])
    source_factor[2, 0] = .1
    components = law.moment_components(mean, source_factor)
    designs = np.array([[[1., .2], [.4, -.3], [0., .8]], [[.1, .9], [.2, -.7], [.3, .2]]])
    actual = components.contract_designs(designs, channels=2, design_errors=np.full(designs.shape, 1e-12))
    explicit = np.column_stack([np.kron(design, np.eye(2)) for design in designs])
    np.testing.assert_allclose(actual.mean.ravel(), explicit@components.mean, atol=3e-16, rtol=0)
    np.testing.assert_allclose(actual.shared_image.reshape(6, 4), explicit@components.shared_factor, atol=2e-16, rtol=0)
    np.testing.assert_allclose(actual.local_image.reshape(6, 10), explicit@components.local_factor, atol=2e-16, rtol=0)
    assert np.all(actual.mean_error > 0) and np.all(actual.shared_image_error > 0)
    assert actual.local_image.shape == (3, 2, 10)
    assert actual.binding["channels"] == 2
    with pytest.raises(ValueError):
        actual.local_image.setflags(write=True)


@pytest.mark.parametrize("states", [1024, 4096])
def test_positive_source_error_reduction_bounds_its_own_lost_small_summands(states):
    law = FiniteResponseLaw.from_states((ResponseState(1, 1, 0, 1, 1),)*states)
    design = np.zeros((states, 1, 1))
    errors = np.full_like(design, states*1e-17)
    errors[0, 0, 0] = states
    actual = law.mean_design(design, design_errors=errors)
    exact = sum((Fraction.from_float(float(value))/states for value in errors.ravel()), Fraction(0))
    assert Fraction.from_float(actual.error[0, 0]) >= exact


def test_compact_positive_source_error_dot_bounds_long_native_coefficient_sum():
    # H=1 avoids a large state bank: the contraction still has k=1024 terms,
    # so the source-error dot must count its actual native reduction length.
    coordinates = 1024
    law = _law((1,))
    factor = np.zeros((coordinates, 1))
    factor[0, 0] = coordinates
    factor[1:, 0] = coordinates*1e-17
    components = law.moment_components(np.zeros(coordinates), factor)
    designs = np.zeros((1, 1, coordinates))
    errors = np.full_like(designs, 1/coordinates)
    actual = components.contract_designs(designs, channels=1, design_errors=errors)
    exact = sum((Fraction.from_float(float(value))/coordinates for value in factor.ravel()), Fraction(0))
    assert Fraction.from_float(actual.shared_image_error[0, 0, 0]) >= exact
