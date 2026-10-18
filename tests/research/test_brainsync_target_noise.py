from fractions import Fraction as Q

import numpy as np
import pytest

from data.contract import BRAIN_SYNC_ACQUISITION_REFERENCE, BRAIN_SYNC_EEG_CHANNELS
from research.cached_response_consumer import CachedAffineResponseSource, ExactMeanResponseSource
from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry
from research.conditional_response_graph import ExactConditionalResponseGraph
from research.latent_response_reference import ResponseState
from research.numeric_response_consumer import AffineResponseSource
from research.source_functional_c_mean import SourceFunctionalCMean
from research.target_noise_operand import (
    NATIVE_OPERATOR,
    SEGMENT_LAW,
    TargetNoiseOperand,
    declare_complete_target_noise,
)
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import array_digest, metadata_digest
from transfer.source_response_transport import ResponseCoordinateChart


def _dictionary(matrix, rate=1000.):
    matrix = np.array(matrix, dtype=float)
    return NativeResponseDictionary.from_source_matrix(matrix, sample_rate_hz=rate,
        source_basis_sha256=array_digest(matrix), terminal_values=np.zeros(matrix.shape[1]))


def _graph(dictionary):
    spatial = tuple(tuple(Q(i == j) for j in range(7)) for i in range(7))
    source_map = tuple(tuple(Q(i == j) for j in range(14)) for i in range(14))
    return ExactConditionalResponseGraph(np.arange(1., 15.), np.eye(14), source_map,
        (Q(),) * 14, (), (), ResponseCoordinateChart(spatial), 2,
        {'function_source_clock_hz': dictionary.sample_rate_hz,
         'function_dictionary_sha256': dictionary.digest(), 'role': 'declared_mathematical_source'})


def test_cached_affine_and_mean_keep_distinct_exact_queries_before_rounding():
    graph = _graph(_dictionary([[1.], [2.], [-1.]]))
    original = AffineResponseSource(graph)
    cached = CachedAffineResponseSource(original, maximum_cache_bytes=32768)
    query = np.empty((3, 1, 14), dtype=object)
    query.fill(Q())
    query[:, 0, 0] = (Q(1, 3), Q(1, 3) + Q(1, 2**100), Q(1, 3))
    reference = original.compile(query, query_binding={'role': 'independent_reference'})
    image = cached.compile(query, query_binding={'role': 'first'})
    warm = cached.compile(query, query_binding={'role': 'warm'})
    np.testing.assert_array_equal(image.values, reference.values)
    np.testing.assert_array_equal(image.absolute_error, reference.absolute_error)
    np.testing.assert_array_equal(warm.values, reference.values)
    # These queries round to the same value but have different exact errors.
    assert image.values[0, 0, 0] == image.values[1, 0, 0]
    assert image.absolute_error[0, 0, 0] != image.absolute_error[1, 0, 0]
    mean, error, _ = ExactMeanResponseSource(graph).compile_mean(query, query_binding={'role': 'mean_only'})
    np.testing.assert_array_equal(mean, image.values[:, :, :1])
    np.testing.assert_array_equal(error, image.absolute_error[:, :, :1])
    for i in range(3):
        assert abs(Q.from_float(mean[i, 0, 0]) - query[i, 0, 0]) <= Q.from_float(error[i, 0, 0])


def test_c_mean_binds_original_source_function_clock_and_dictionary():
    dictionary = _dictionary([[1.], [2.], [-1.]])
    graph = _graph(dictionary)

    class NoCompilation:
        def __init__(self):
            self.graph = graph

        def compile_mean(self, *args, **kwargs):
            raise AssertionError('incompatible dictionary reached mean compilation')

    arguments = {'states_by_block': {1: ResponseState(1., 1., 0., 1., 1.)},
                 'clock_program': 'timeline_wall'}
    with pytest.raises(ValueError, match='function clock'):
        SourceFunctionalCMean(graph, _dictionary(dictionary.matrix, rate=250.), NoCompilation(), **arguments)
    with pytest.raises(ValueError, match='function dictionary'):
        SourceFunctionalCMean(graph, _dictionary(-dictionary.matrix), NoCompilation(), **arguments)
    callback = SourceFunctionalCMean(graph, dictionary, ExactMeanResponseSource(graph), **arguments)
    assert callback.coefficient_abs_bounds == tuple(tuple(Q(i * 7 + j + 1) for j in range(7))
        for i in range(2))


def test_complete_seven_channel_noise_codec_and_eta_match_dense_voltage_law(tmp_path):
    covariance = 1e-12 * .2**abs(np.arange(7)[:, None] - np.arange(7)[None, :])
    geometry = NoiseGeometry(tuple(BRAIN_SYNC_EEG_CHANNELS), 250., 'V',
        BRAIN_SYNC_ACQUISITION_REFERENCE, NATIVE_OPERATOR, 7)
    # Zero AR coefficients give an independent temporal law with correlated EEG.
    noise = FittedChannelNoise.from_parameters(np.zeros((7, 1)), covariance,
        geometry=geometry, conditioning_samples=4, reference_id='declared_seven_channel_AR0', family='private')
    operand = declare_complete_target_noise(noise, eta=1.7,
        declaration={'declaration_id': 'mathematical_seven_channel_law', 'source': 'known_covariance'},
        segment_law=SEGMENT_LAW)
    path = tmp_path / 'complete_noise.json'
    saved_sha = operand.save(path)
    cold = TargetNoiseOperand.load(path, expected_sha256=saved_sha)
    assert cold.identity == operand.identity
    assert metadata_digest(cold.noise.record()) == metadata_digest(noise.record())
    assert len(cold.noise.temporal) == 7 and cold.noise.innovation_covariance.shape == (7, 7)
    assert cold.programme['noise_C_fit_calls'] == cold.programme['response_C_updates_here'] == 0
    observed = np.arange(70).reshape(10, 7) * 1e-7 - 3e-6
    carrier = cold.carrier()
    white, _ = carrier.white_observation(observed)
    expected_quad = np.sum(observed * np.linalg.solve(1.7 * covariance, observed.T).T)
    expected = -.5 * (70 * np.log(2 * np.pi) + 10 * np.linalg.slogdet(1.7 * covariance)[1] + expected_quad)
    actual = carrier.leading_log_jacobian(10) - 35 * np.log(2 * np.pi) - .5 * (white @ white)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-10)
