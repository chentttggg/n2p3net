"""Canonical actual-query behavior: strong cancellation and weak information."""
from fractions import Fraction as Q

import numpy as np

from research.conditional_response_graph import ExactConditionalResponseGraph
from research.numeric_response_consumer import AffineResponseSource, GraphAffineResponse
from transfer.source_response_transport import ResponseCoordinateChart


def _compiled_moments_are_enclosed(compiled, expected_mean, expected_covariance):
    values = compiled.values.reshape(-1, compiled.values.shape[-1])
    errors = compiled.absolute_error.reshape(values.shape)
    saved = tuple(tuple(Q.from_float(float(value)) for value in row) for row in values)
    error = tuple(tuple(Q.from_float(float(value)) for value in row) for row in errors)
    for i, row in enumerate(saved):
        assert abs(row[0] - expected_mean[i]) <= error[i][0]
        for j, other in enumerate(saved):
            covariance = sum((a * b for a, b in zip(row[1:], other[1:], strict=True)), Q())
            enclosure = sum((abs(a) * eb + abs(b) * ea + ea * eb for a, b, ea, eb in
                zip(row[1:], other[1:], error[i][1:], error[j][1:], strict=True)), Q())
            assert abs(covariance - expected_covariance[i][j]) <= enclosure


def _weak_source():
    e = 2.**-60
    graph = ExactConditionalResponseGraph(
        np.array([3., 7., -1.]),
        np.array([[1., 0., 0.], [e, e, 0.], [0., 0., e]]),
        ((Q(1), Q(1), Q(0)), (Q(1), Q(0), Q(0))),
        (Q(0), Q(0)), (), (),
        ResponseCoordinateChart(((Q(1), Q(0)), (Q(0), Q(1)))), 1,
        {"role": "independent_mathematical_source_weak_difference"})
    return AffineResponseSource(graph), Q(1, 2**60)


def test_actual_functional_cancels_strong_common_before_rounding():
    # Under an I3 reference, given z=sum(x), x2-z/3 and x3-z/3
    # have covariance [[2/3,-1/3],[-1/3,2/3]], independently of z.
    # Reference native mean (1,2,6) gives residual means (-1,3).
    source_sd = Q(2**60)
    graph = ExactConditionalResponseGraph(
        np.array([7.]), np.array([[float(source_sd)]]), ((Q(1),),),
        (Q(9), Q(2), Q(6)), ((Q(1, 3),), (Q(1, 3),)),
        ((Q(2, 3), Q(-1, 3)), (Q(-1, 3), Q(2, 3))),
        ResponseCoordinateChart(((Q(1), Q(1), Q(1)),)), 1,
        {"role": "independent_mathematical_strong_common_conditional_residual"})
    query = ((Q(1), Q(0), Q(0)), (Q(-1, 3), Q(1), Q(0)),
             (Q(-1, 3), Q(0), Q(1)))
    mean = (Q(7), Q(-1), Q(3))
    covariance = ((source_sd**2, Q(0), Q(0)),
                  (Q(0), Q(2, 3), Q(-1, 3)),
                  (Q(0), Q(-1, 3), Q(2, 3)))
    exact = graph.exact_linear_image(query)
    assert exact.mean == mean and exact.covariance == covariance
    source = AffineResponseSource(graph)
    compiled = source.compile(np.array(query, dtype=object)[:, None, :],
        query_binding={"case": "strong_source_common_and_two_conditional_residuals"})
    assert np.array_equal(compiled.values[:, 0, 1], [float(source_sd), 0., 0.])
    _compiled_moments_are_enclosed(compiled, mean, covariance)
    typed = compiled.gaussian_response_images(source.coordinate_ids)
    np.testing.assert_array_equal(typed.proper_error, compiled.absolute_error[:, :, 1:])


def test_actual_source_weak_contrast_retains_both_independent_terms():
    # z1=beta1+beta2 and z2=beta1. Their exact difference is beta2,
    # whose original source factor is e*(xi1+xi2), variance=2e^2.
    source, e = _weak_source()
    query = np.array([[[Q(1), Q(-1)]]], dtype=object)
    compiled = source.compile(query, query_binding={"case": "exact_source_weak_contrast"})
    np.testing.assert_array_equal(compiled.values[0, 0], [7., float(e), float(e), 0.])
    assert np.count_nonzero(compiled.absolute_error) == 0
    assert source.graph.exact_linear_image(query[0]).covariance == ((2 * e**2,),)


def test_state_reference_functionals_are_subtracted_before_numeric_images():
    class StateFunctionals:
        scope = ("target",)
        identity = "independent_weak_source_state_pair"

        def functionals(self, selected_states, *, native_start, native_stop):
            row = (Q(1), Q(0)) if selected_states["target"] else (Q(0), Q(1))
            count = native_stop - native_start
            return np.tile(np.array(row, dtype=object), (count, 1, 1)), {"states": selected_states}

    source, e = _weak_source()
    response = GraphAffineResponse(StateFunctionals(), source, 1)
    actual, reference = {"target": 1}, {"target": 0}
    direct, error, binding = response.affine_difference(actual, reference, native_start=0, native_stop=1)
    np.testing.assert_array_equal(direct[0, 0], [7., float(e), float(e), 0.])
    assert np.count_nonzero(error) == 0
    assert binding["actual_query_binding"]["operation"] == "exact_actual_functional_state_minus_reference"
    # A red control: each row is correctly rounded, but subtracting those
    # numerical images afterwards removes one of the two weak source terms.
    late = response.affine(actual, native_start=0, native_stop=1)[0].copy()
    late -= response.affine(reference, native_start=0, native_stop=1)[0]
    variance = sum((Q.from_float(float(x))**2 for x in late[0, 0, 1:]), Q())
    assert variance == e**2
