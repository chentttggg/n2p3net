from fractions import Fraction as Q

import numpy as np
import pytest

from research.conditional_response_graph import ExactConditionalResponseGraph
from research.exact_common_marginal import ExactCommonMarginalSource
from research.exact_template_response_consumer import ExactTemplateAffineSource
from research.numeric_response_consumer import AffineResponseSource
from transfer.source_response_transport import ResponseCoordinateChart


def _source(marginal):
    mapping = ((1, 0, 0, 0, 1), (0, 1, 0, 0, 0),
               (0, 0, 1, 0, 0), (1, 1, 0, 0, 0))
    graph = ExactConditionalResponseGraph(np.arange(1., 6.),
        np.diag([1., 1e-300, 2., 4., 0.]), mapping, (Q(),) * 4, (), (),
        ResponseCoordinateChart(((Q(1), Q(0)), (Q(0), Q(1)))), 2,
        {'role': 'independent_mathematical_source_zero_projection'})
    original = AffineResponseSource(graph)
    if not marginal:
        return original
    reduced = ExactCommonMarginalSource(original)
    assert reduced.rank == 3  # The positive 1e-600 variance remains represented.
    return reduced


@pytest.mark.parametrize('marginal', (False, True))
@pytest.mark.parametrize('spatial', (
    ((0, 0), (1, Q(-1, 2))),
    ((1, Q(-1, 2)), (0, 0)),
    ((0, 0), (0, 0)),
))
def test_zero_spatial_functionals_preserve_explicit_and_generic_queries(marginal, spatial):
    source = _source(marginal)
    template = ExactTemplateAffineSource(source, native_spatial_map=spatial,
        observed_channel_ids=('left', 'right'), maximum_cache_bytes=32768)
    temporal = ((Q(1), Q(0)), (Q(0), Q(1)), (Q(1, 2), Q(-1, 2)))
    query = np.array([[[value * weight for value in row for weight in channel]
        for channel in spatial] for row in temporal], dtype=object)
    expected = source.compile(query, query_binding={'role': 'direct_actual_functional'})
    generic = template.compile(query, query_binding={'role': 'generic_zero_projection'})
    explicit = template.compile_temporal(temporal, query_binding={'role': 'explicit_temporal'})
    for result in (generic, explicit):
        np.testing.assert_array_equal(result.values, expected.values)
        np.testing.assert_array_equal(result.absolute_error, expected.absolute_error)
    # A same-shape query outside M must retain its actual nonzero functional.
    changed = query.copy()
    zero_channel = next(i for i, row in enumerate(spatial) if not any(row))
    changed[0, zero_channel, 0] = Q(1)
    actual = template.compile(changed, query_binding={'role': 'mixed_actual_query'})
    direct = source.compile(changed, query_binding={'role': 'mixed_reference'})
    np.testing.assert_array_equal(actual.values, direct.values)
    np.testing.assert_array_equal(actual.absolute_error, direct.absolute_error)
    assert actual.values[0, zero_channel, 0] != 0
